"""Prepare the one shared feature table used by frozen classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from labels.build import CAMERA_GROUPS

CACHE_ROOT = Path("output/models/_cache")
CURRENT_SENSORS = (
    "ambient_temperature",
    "environment_relative_humidity",
    "water_in_temperature",
    "water_out_temperature",
    "water_temperature_setpoint",
    "water_flow",
    "evaporating_pressure",
    "coil_temperature",
    "suction_temperature",
    "superheat",
    "condensing_pressure",
    "condensing_temperature",
    "discharge_temperature",
    "plate_heat_exchanger_inlet_temperature",
    "plate_heat_exchanger_outlet_temperature",
    "compressor_frequency",
    "compressor_frequency_setpoint",
    "compressor_current",
    "compressor_power",
    "fan_speed",
    "fan_current",
    "exv_opening",
    "heating_capacity",
    "power_total",
    "cop",
    "evaporator_capacity",
    "pressure_ratio",
    "water_delta_temperature",
)
SLOPE_SENSORS = (
    "evaporating_pressure",
    "coil_temperature",
    "fan_current",
    "compressor_frequency",
    "exv_opening",
    "compressor_power",
    "power_total",
    "heating_capacity",
    "cop",
    "evaporator_capacity",
    "water_out_temperature",
    "water_delta_temperature",
    "suction_temperature",
    "superheat",
    "discharge_temperature",
    "pressure_ratio",
)


def even_sample_groups(
    rows: pd.DataFrame, group_columns: list[str], *, maximum_per_group: int
) -> pd.DataFrame:
    if maximum_per_group < 1:
        raise ValueError("maximum_per_group must be positive")
    ordered = rows.reset_index(drop=True).copy()
    ordered["_source_position"] = np.arange(len(ordered))
    sampled: list[pd.DataFrame] = []
    for _, group in ordered.sort_values("image_time").groupby(
        group_columns, sort=True, observed=True
    ):
        positions = np.linspace(
            0, len(group) - 1, min(len(group), maximum_per_group), dtype=int
        )
        sampled.append(group.iloc[positions])
    if not sampled:
        return ordered.drop(columns="_source_position")
    return (
        pd.concat(sampled)
        .sort_values("_source_position")
        .drop(columns="_source_position")
        .reset_index(drop=True)
    )


def image_color_features(path: Path) -> np.ndarray:
    """Return the established 34-value color-and-gradient descriptor."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB").resize((64, 36)), dtype=np.float32) / 255
    histograms = [
        np.histogram(pixels[..., channel], bins=8, range=(0, 1), density=True)[0] / 8
        for channel in range(3)
    ]
    grey = pixels.mean(axis=2)
    dx = np.abs(np.diff(grey, axis=1))
    dy = np.abs(np.diff(grey, axis=0))
    result: np.ndarray = np.concatenate(
        [
            *histograms,
            pixels.mean(axis=(0, 1)),
            pixels.std(axis=(0, 1)),
            [dx.mean(), dx.std(), dy.mean(), dy.std()],
        ]
    ).astype(np.float32)
    return result


def load_feature_shards(
    labels: pd.DataFrame, shards: Path, representation: str
) -> pd.DataFrame:
    """Relabel existing per-cycle deep features without rerunning a backbone."""
    prefix = f"{representation}_"
    keys = ["cycle_name", "camera_role", "file_name"]
    paths = [shards / f"{cycle}.parquet" for cycle in sorted(labels["cycle_name"].unique())]
    missing = [path.stem for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing feature shards: {', '.join(missing)}")
    available = pd.read_parquet(paths[0]).columns
    feature_columns = [column for column in available if str(column).startswith(prefix)]
    if not feature_columns:
        raise ValueError(f"feature shards do not contain {representation}")
    features = pd.concat(
        (pd.read_parquet(path, columns=[*keys, *feature_columns]) for path in paths),
        ignore_index=True,
    )
    return labels.merge(features, on=keys, how="inner", validate="one_to_one")


def causal_sensor_features(
    frame: pd.DataFrame,
    *,
    current_sensors: tuple[str, ...] = CURRENT_SENSORS,
    slope_sensors: tuple[str, ...] = SLOPE_SENSORS,
    bucket_seconds: int = 10,
) -> pd.DataFrame:
    """Mask imputed readings and add five-minute backward-only slopes."""
    values = frame.copy()
    values["sensor_timestamp"] = pd.to_datetime(
        values.pop("timestamp"), errors="coerce", format="mixed"
    ) + pd.Timedelta(seconds=bucket_seconds)
    for name in set(current_sensors) | set(slope_sensors):
        values[name] = pd.to_numeric(values[name], errors="coerce")
        marker = f"{name}__imputed"
        if marker in values:
            values[name] = values[name].mask(values[marker].fillna(True))
    groups = []
    for _, cycle in values.groupby("cycle_name", sort=False):
        cycle = cycle.sort_values("sensor_timestamp")
        past = cycle[["sensor_timestamp", *slope_sensors]].rename(
            columns={
                "sensor_timestamp": "past_timestamp",
                **{name: f"past_{name}" for name in slope_sensors},
            }
        )
        paired = pd.merge_asof(
            cycle.assign(
                slope_target_time=cycle["sensor_timestamp"] - pd.Timedelta(minutes=5)
            ),
            past,
            left_on="slope_target_time",
            right_on="past_timestamp",
            direction="backward",
            tolerance=pd.Timedelta(seconds=15),
        )
        elapsed = (
            paired["sensor_timestamp"] - paired["past_timestamp"]
        ).dt.total_seconds() / 60
        for name in slope_sensors:
            paired[f"{name}__slope_5min"] = (
                paired[name] - paired[f"past_{name}"]
            ) / elapsed
        groups.append(
            paired.drop(
                columns=[
                    "slope_target_time",
                    "past_timestamp",
                    *[f"past_{name}" for name in slope_sensors],
                ]
            )
        )
    return pd.concat(groups, ignore_index=True) if groups else values


def attach_causal_sensors(rows: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    """Attach the latest same-cycle sensor row to every image."""
    required = tuple(dict.fromkeys((*CURRENT_SENSORS, *SLOPE_SENSORS)))
    registry = json.loads((dataset_root / "channel_registry.json").read_text())
    bucket_seconds = int(registry["resample_interval_seconds"])
    sensor_frames = []
    for cycle_name in sorted(rows["cycle_name"].unique()):
        path = dataset_root / "cycles" / f"{cycle_name}.parquet"
        columns = ["cycle_name", "timestamp", *required]
        columns += [f"{name}__imputed" for name in required]
        sensor_frames.append(
            causal_sensor_features(
                pd.read_parquet(path, columns=columns), bucket_seconds=bucket_seconds
            )
        )
    sensors = pd.concat(sensor_frames, ignore_index=True)
    merged = []
    for cycle_name, images in rows.groupby("cycle_name", sort=False):
        cycle_sensors = sensors.loc[sensors["cycle_name"].eq(cycle_name)].sort_values(
            "sensor_timestamp"
        )
        merged.append(
            pd.merge_asof(
                images.assign(
                    image_time=pd.to_datetime(images["image_time"], format="mixed")
                ).sort_values("image_time"),
                cycle_sensors.drop(columns="cycle_name"),
                left_on="image_time",
                right_on="sensor_timestamp",
                direction="backward",
                tolerance=pd.Timedelta(seconds=15),
            )
        )
    return pd.concat(merged, ignore_index=True)


def prepare_cached_features(
    rows: pd.DataFrame,
    *,
    representation: str,
    camera: str,
    modality: str,
    state_column: str,
    maximum_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Select one camera and modality from the shared cached feature table."""
    full = rows.loc[rows["camera_role"].isin(CAMERA_GROUPS[camera])].copy()
    if modality != "rgb":
        full = full.loc[full["sensor_timestamp"].notna()].copy()
    training = even_sample_groups(
        full,
        [state_column, "camera_role", "cycle_name"],
        maximum_per_group=maximum_per_group,
    )
    columns = [column for column in full if column.startswith(f"{representation}_")]
    if modality in {"rgb_sensor", "rgb_sensor_slope"}:
        columns += list(CURRENT_SENSORS)
    if modality == "rgb_sensor_slope":
        columns += [f"{name}__slope_5min" for name in SLOPE_SENSORS]
    return training, full.reset_index(drop=True), columns


def _extract_handcrafted(
    rows: pd.DataFrame, absolute_paths: list[str], camera: str
) -> pd.DataFrame:
    roles = CAMERA_GROUPS[camera]
    role_index = {role: index for index, role in enumerate(roles)}
    role_eye: np.ndarray = np.eye(len(roles), dtype=np.float32)
    values = [
        np.concatenate(
            [
                image_color_features(Path(path)),
                role_eye[role_index[str(row.camera_role)]],
            ]
        )
        for row, path in zip(rows.itertuples(index=False), absolute_paths, strict=True)
    ]
    width = 34 + len(roles)
    matrix = np.stack(values) if values else np.empty((0, width), dtype=np.float32)
    result = pd.DataFrame({"absolute_path": absolute_paths})
    for index in range(width):
        result[f"feature_{index:03d}"] = matrix[:, index]
    return result


def prepare_features(
    rows: pd.DataFrame,
    *,
    dataset_root: Path,
    representation: str,
    camera: str,
    modality: str,
    state_column: str,
    maximum_per_group: int,
    cache_root: Path = CACHE_ROOT,
) -> tuple[pd.DataFrame, list[str]]:
    """Sample rows, reuse their image descriptor cache, and select one modality."""
    if representation != "handcrafted":
        raise ValueError(f"frozen features do not implement {representation}")
    sampled = even_sample_groups(
        rows,
        [state_column, "camera_role", "cycle_name"],
        maximum_per_group=maximum_per_group,
    )
    # Stable heating start is an observed physical boundary preceding each image;
    # this uses neither a future cycle end nor any sensor value.
    image_time = pd.to_datetime(sampled["image_time"], errors="raise", format="mixed")
    stable_start = pd.to_datetime(
        sampled["stable_heating_start"], errors="raise", format="mixed"
    )
    sampled["time_minutes"] = (image_time - stable_start).dt.total_seconds() / 60
    if modality == "time":
        return sampled.reset_index(drop=True), ["time_minutes"]

    cache_path = cache_root / representation / camera / "features.parquet"
    cached = pd.read_parquet(cache_path) if cache_path.is_file() else pd.DataFrame()
    absolute_paths = [
        str((dataset_root / str(path)).resolve()) for path in sampled["image_path"]
    ]
    feature_columns = [
        str(column) for column in cached.columns if str(column).startswith("feature_")
    ]
    reusable = (
        cached.get("absolute_path", pd.Series(dtype="string")).astype(str).tolist()
        == absolute_paths
        and len(feature_columns) == 34 + len(CAMERA_GROUPS[camera])
    )
    if not reusable:
        cached = _extract_handcrafted(sampled, absolute_paths, camera)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(cache_path, index=False)
        feature_columns = [
            str(column)
            for column in cached.columns
            if str(column).startswith("feature_")
        ]

    result = sampled.copy()
    result[feature_columns] = cached[feature_columns].to_numpy()
    selected_columns = (
        feature_columns if modality == "rgb" else [*feature_columns, "time_minutes"]
    )
    return result.reset_index(drop=True), selected_columns
