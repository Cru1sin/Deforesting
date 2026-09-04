"""Prepare the one shared feature table used by frozen classifiers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from image_labels.timing import CAMERA_GROUPS
from image_models.sensor_features import CURRENT_SENSORS, SLOPE_SENSORS

CACHE_ROOT = Path("output/image_models/_cache")


def limit_images_per_cycle_and_label(
    rows: pd.DataFrame, group_columns: list[str], *, max_images_per_cycle_label: int
) -> pd.DataFrame:
    if max_images_per_cycle_label < 1:
        raise ValueError("max_images_per_cycle_label must be positive")
    ordered = rows.reset_index(drop=True).copy()
    ordered["_source_position"] = np.arange(len(ordered))
    sampled: list[pd.DataFrame] = []
    for _, group in ordered.sort_values("image_time").groupby(
        group_columns, sort=True, observed=True
    ):
        positions = np.linspace(
            0, len(group) - 1, min(len(group), max_images_per_cycle_label), dtype=int
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


def extract_color_gradient_features(path: Path) -> np.ndarray:
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


def load_dinov2_feature_cache(
    labels: pd.DataFrame, cache: Path, image_feature: str
) -> pd.DataFrame:
    """Relabel existing per-cycle deep features without rerunning a backbone."""
    prefix = f"{image_feature}_"
    keys = ["cycle_name", "camera_role", "file_name"]
    paths = [cache / f"{cycle}.parquet" for cycle in sorted(labels["cycle_name"].unique())]
    missing = [path.stem for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing cached features: {', '.join(missing)}")
    available = pd.read_parquet(paths[0]).columns
    feature_columns = [column for column in available if str(column).startswith(prefix)]
    if not feature_columns:
        raise ValueError(f"feature cache does not contain {image_feature}")
    features = pd.concat(
        (pd.read_parquet(path, columns=[*keys, *feature_columns]) for path in paths),
        ignore_index=True,
    )
    return labels.merge(features, on=keys, how="inner", validate="one_to_one")


def prepare_cached_features(
    rows: pd.DataFrame,
    *,
    image_feature: str,
    camera: str,
    input_feature: str,
    label_column: str,
    max_images_per_cycle_label: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Select one camera and input_feature from the shared cached feature table."""
    full = rows.loc[rows["camera_role"].isin(CAMERA_GROUPS[camera])].copy()
    if input_feature != "image_only":
        full = full.loc[full["sensor_timestamp"].notna()].copy()
    training = limit_images_per_cycle_and_label(
        full,
        [label_column, "camera_role", "cycle_name"],
        max_images_per_cycle_label=max_images_per_cycle_label,
    )
    columns = [column for column in full if column.startswith(f"{image_feature}_")]
    if input_feature in {"image_plus_current_sensors", "image_plus_sensor_slopes"}:
        columns += list(CURRENT_SENSORS)
    if input_feature == "image_plus_sensor_slopes":
        columns += [f"{name}__slope_5min" for name in SLOPE_SENSORS]
    return training, full.reset_index(drop=True), columns


def _extract_color_gradient(
    rows: pd.DataFrame, absolute_paths: list[str], camera: str
) -> pd.DataFrame:
    roles = CAMERA_GROUPS[camera]
    role_index = {role: index for index, role in enumerate(roles)}
    role_eye: np.ndarray = np.eye(len(roles), dtype=np.float32)
    values = [
        np.concatenate(
            [
                extract_color_gradient_features(Path(path)),
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
    image_feature: str,
    camera: str,
    input_feature: str,
    label_column: str,
    max_images_per_cycle_label: int,
    cache_root: Path = CACHE_ROOT,
) -> tuple[pd.DataFrame, list[str]]:
    """Sample rows, reuse their image descriptor cache, and select one input_feature."""
    if image_feature != "color_gradient":
        raise ValueError(f"frozen features do not implement {image_feature}")
    sampled = limit_images_per_cycle_and_label(
        rows,
        [label_column, "camera_role", "cycle_name"],
        max_images_per_cycle_label=max_images_per_cycle_label,
    )
    # Stable heating start is an observed physical boundary preceding each image;
    # this uses neither a future cycle end nor any sensor value.
    image_time = pd.to_datetime(sampled["image_time"], errors="raise", format="mixed")
    stable_start = pd.to_datetime(
        sampled["stable_heating_start"], errors="raise", format="mixed"
    )
    sampled["time_minutes"] = (image_time - stable_start).dt.total_seconds() / 60
    if input_feature == "elapsed_time_only":
        return sampled.reset_index(drop=True), ["time_minutes"]

    cache_path = cache_root / image_feature / camera / "features.parquet"
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
        cached = _extract_color_gradient(sampled, absolute_paths, camera)
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
        feature_columns if input_feature == "image_only" else [*feature_columns, "time_minutes"]
    )
    return result.reset_index(drop=True), selected_columns
