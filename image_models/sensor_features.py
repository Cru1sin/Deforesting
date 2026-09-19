"""Build past-only sensor values and slopes for image timestamps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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


def build_past_only_sensor_statistics(
    frame: pd.DataFrame,
    *,
    current_sensors: tuple[str, ...] = CURRENT_SENSORS,
    bucket_seconds: int = 10,
    include_current: bool = False,
) -> pd.DataFrame:
    """Six statistics on available bucket endpoints in the last five minutes.

    Sample std and unbiased Fisher skew/kurtosis follow pandas conventions;
    slope is least squares per minute, entropy is natural-log histogram entropy.
    """
    values = frame.copy()
    values["sensor_timestamp"] = pd.to_datetime(values["timestamp"], format="mixed") + pd.Timedelta(
        seconds=bucket_seconds
    )
    outputs = []
    for _, cycle in values.groupby("cycle_name", sort=False):
        cycle = cycle.sort_values("sensor_timestamp").set_index("sensor_timestamp")
        result = {"cycle_name": cycle["cycle_name"]}
        for name in current_sensors:
            x = pd.to_numeric(cycle[name], errors="coerce")
            if f"{name}__imputed" in cycle:
                x = x.where(cycle[f"{name}__imputed"].eq(False))
            rolling = x.rolling("5min", closed="both", min_periods=1)
            if include_current:
                result[f"stat_{name}_current"] = x.ffill().where(
                    (pd.Series(cycle.index, index=cycle.index)
                     - pd.Series(cycle.index.where(x.notna()), index=cycle.index).ffill())
                    .dt.total_seconds().le(300)
                )
            for statistic in ("mean", "std", "skew", "kurt"):
                column = f"stat_{name}_{statistic}"
                # pandas rolling skew/kurt center on the entire series internally.
                # Window-local moments keep past results independent of future values.
                if statistic in {"skew", "kurt"}:
                    result[column] = rolling.apply(
                        lambda window, kind=statistic: getattr(pd.Series(window), kind)(),
                        raw=True,
                    )
                else:
                    result[column] = getattr(rolling, statistic)()
                result[f"{column}_missing"] = result[column].isna()

            def slope(window):
                valid = window.dropna()
                if len(valid) < 2:
                    return np.nan
                minutes = (valid.index.asi8 - valid.index.asi8[0]) / 60e9
                centered = minutes - minutes.mean()
                denominator = np.square(centered).sum()
                return (
                    np.dot(centered, valid.to_numpy() - valid.mean()) / denominator
                    if denominator
                    else np.nan
                )

            def entropy(window):
                valid = window[np.isfinite(window)]
                if len(valid) < 2:
                    return np.nan
                span = valid.max() - valid.min()
                if span == 0:
                    return 0.0
                counts = np.histogram((valid - valid.min()) / span, bins=10, range=(0.0, 1.0))[0]
                probabilities = counts[counts > 0] / len(valid)
                return -np.dot(probabilities, np.log(probabilities))

            for statistic, series in (
                ("slope", rolling.apply(slope, raw=False)),
                ("entropy", rolling.apply(entropy, raw=True)),
            ):
                column = f"stat_{name}_{statistic}"
                result[column] = series
                result[f"{column}_missing"] = series.isna()
            result[f"stat_{name}_valid_count"] = rolling.count()
            last_valid = pd.Series(cycle.index.where(x.notna()), index=cycle.index).ffill()
            result[f"stat_{name}_age_seconds"] = (
                pd.Series(cycle.index, index=cycle.index) - last_valid
            ).dt.total_seconds()
        outputs.append(pd.DataFrame(result, index=cycle.index).reset_index())
    return pd.concat(outputs, ignore_index=True)


def build_past_only_sensor_features(
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
            cycle.assign(slope_target_time=cycle["sensor_timestamp"] - pd.Timedelta(minutes=5)),
            past,
            left_on="slope_target_time",
            right_on="past_timestamp",
            direction="backward",
            tolerance=pd.Timedelta(seconds=15),
        )
        elapsed = (paired["sensor_timestamp"] - paired["past_timestamp"]).dt.total_seconds() / 60
        for name in slope_sensors:
            paired[f"{name}__slope_5min"] = (paired[name] - paired[f"past_{name}"]) / elapsed
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


def attach_latest_past_sensor_values(rows: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
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
            build_past_only_sensor_features(
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
