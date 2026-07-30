"""Bounded resampling, missing handling, physics, and baseline orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .baseline import add_baseline_residuals
from .config import Config
from .features import add_dynamic_features, calculate_derived_features


def process(
    prepared: pd.DataFrame,
    initial_summary: pd.DataFrame,
    config: Config,
    channels: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the fixed Process order and return processed rows plus cycle summary."""
    _require_keys(prepared, ["experiment_id", "timestamp", "cycle_id", "cycle_stage"])
    masked = _mask_duplicate_values(prepared, channels)
    interval_seconds = int(config.process.get("resample_interval_seconds", 10))
    resampled = _resample(masked, channels, interval_seconds)
    resampled = _recompute_cycle_coordinates(resampled, initial_summary)
    filled = _fill_missing(resampled, channels, config.process)
    derived = calculate_derived_features(filled, channels)
    baseline = add_baseline_residuals(
        derived,
        initial_summary,
        channels,
        config.process.get("baseline", {}),
    )
    featured = add_dynamic_features(
        baseline,
        channels,
        interval_seconds=interval_seconds,
        windows_minutes=[
            int(value)
            for value in config.process.get("features", {}).get("windows_minutes", [1])
        ],
    )
    featured = featured.sort_values(["experiment_id", "timestamp"], kind="stable").reset_index(
        drop=True
    )
    final_summary = _update_summary(initial_summary, featured)
    return featured, final_summary


def _mask_duplicate_values(
    frame: pd.DataFrame, channels: Mapping[str, Mapping[str, Any]]
) -> pd.DataFrame:
    result = frame.copy()
    for name in channels:
        duplicate = result.get(
            f"{name}__duplicate", pd.Series(False, index=result.index)
        ).astype(bool)
        conflict = result.get(
            f"{name}__conflict", pd.Series(False, index=result.index)
        ).astype(bool)
        if name in result:
            mask = duplicate | conflict
            if mask.any():
                result[name] = result[name].astype(object)
                result.loc[mask, name] = np.nan
    quality_columns = [
        column
        for column in result.columns
        if str(column).endswith(("__missing", "__invalid", "__duplicate", "__conflict"))
    ]
    return result.drop(columns=quality_columns)


def _resample(
    frame: pd.DataFrame, channels: Mapping[str, Mapping[str, Any]], interval_seconds: int
) -> pd.DataFrame:
    keys = ["experiment_id", "cycle_id", "cycle_stage"]
    rows: list[dict[str, object]] = []
    frequency = pd.Timedelta(seconds=interval_seconds)
    for group_values, group in frame.groupby(keys, sort=False, dropna=False):
        ordered = group.sort_values("timestamp", kind="stable")
        buckets = ordered["timestamp"].dt.floor(frequency)
        start = buckets.min()
        end = buckets.max()
        for timestamp in pd.date_range(start, end, freq=frequency):
            bucket = ordered.loc[buckets.eq(timestamp)]
            row: dict[str, object] = {
                key: value for key, value in zip(keys, group_values, strict=True)
            }
            row["experiment_date"] = (
                str(ordered["experiment_date"].iloc[0])
                if "experiment_date" in ordered
                else ""
            )
            for status_column in ("cycle_status", "cycle_status_reason"):
                if status_column in ordered:
                    row[status_column] = ordered[status_column].iloc[0]
            row["timestamp"] = timestamp
            for name, settings in channels.items():
                if name not in bucket:
                    continue
                values = pd.to_numeric(bucket[name], errors="coerce")
                is_last = str(settings.get("kind")) in {
                    "step",
                    "event",
                    "categorical",
                    "protected",
                } or str(settings.get("resample")) == "last"
                if is_last:
                    row[name] = (
                        bucket[name].dropna().iloc[-1]
                        if bucket[name].notna().any()
                        else np.nan
                    )
                else:
                    row[name] = values.mean()
            for image_column in ("image_path", "image_time", "image_camera_id"):
                row[image_column] = _nearest_image(bucket, timestamp, image_column)
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["experiment_id", "timestamp"], kind="stable").reset_index(drop=True)


def _nearest_image(bucket: pd.DataFrame, timestamp: pd.Timestamp, column: str) -> object:
    if column not in bucket:
        return np.nan
    valid = bucket.loc[bucket[column].notna(), ["timestamp", column]]
    if valid.empty:
        return np.nan
    distances = (valid["timestamp"] - timestamp).abs()
    return valid.loc[distances.idxmin(), column]


def _recompute_cycle_coordinates(
    frame: pd.DataFrame, cycle_summary: pd.DataFrame
) -> pd.DataFrame:
    result = frame.copy()
    result["cycle_elapsed_seconds"] = np.nan
    result["cycle_progress"] = np.nan
    for _, cycle in cycle_summary.iterrows():
        stable = cycle["stable_heating_start"]
        defrost = cycle["defrost_start"]
        if not isinstance(stable, pd.Timestamp) or not isinstance(defrost, pd.Timestamp):
            continue
        mask = result["experiment_id"].eq(cycle["experiment_id"]) & result["cycle_id"].eq(
            cycle["cycle_id"]
        )
        development = mask & result["cycle_stage"].eq("frost_development")
        elapsed = (result.loc[development, "timestamp"] - stable).dt.total_seconds()
        duration = (defrost - stable).total_seconds()
        if duration <= 0:
            continue
        result.loc[development, "cycle_elapsed_seconds"] = elapsed
        result.loc[development, "cycle_progress"] = (elapsed / duration).clip(0, 1)
    return result


def _fill_missing(
    frame: pd.DataFrame,
    channels: Mapping[str, Mapping[str, Any]],
    process_settings: Mapping[str, Any],
) -> pd.DataFrame:
    result = frame.copy()
    keys = ["experiment_id", "cycle_id", "cycle_stage"]
    interval = int(process_settings.get("resample_interval_seconds", 10))
    continuous_gap = float(process_settings.get("continuous_max_gap_seconds", 30))
    control_gap = float(process_settings.get("control_max_gap_seconds", 30))
    for name, settings in channels.items():
        if name not in result or str(settings.get("kind")) == "derived":
            continue
        result[f"{name}__imputed"] = False
        missing_policy = str(settings.get("missing", "none"))
        for _, group in result.groupby(keys, sort=False, dropna=False):
            indices = group.index
            original = result.loc[indices, name]
            is_continuous = str(settings.get("kind")) == "continuous"
            if missing_policy in {"interpolate", "linear"} and is_continuous:
                limit = max(1, int(continuous_gap // interval) - 1)
                values = pd.to_numeric(original, errors="coerce").copy()
                values.index = pd.DatetimeIndex(result.loc[indices, "timestamp"])
                filled = values.interpolate(method="time", limit=limit, limit_area="inside")
                result.loc[indices, name] = filled.to_numpy()
            elif missing_policy in {"forward_fill", "ffill"} and str(
                settings.get("kind")
            ) == "step":
                limit = max(1, int(control_gap // interval))
                result.loc[indices, name] = original.ffill(limit=limit).to_numpy()
            imputed = original.isna() & result.loc[indices, name].notna()
            result.loc[indices, f"{name}__imputed"] = imputed.to_numpy()
    return result


def _update_summary(summary: pd.DataFrame, processed: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    counts = (
        processed.groupby(["experiment_id", "cycle_id"], dropna=False)
        .size()
        .rename("processed_row_count")
        .reset_index()
    )
    return result.merge(
        counts,
        on=["experiment_id", "cycle_id"],
        how="left",
        validate="one_to_one",
    )


def _require_keys(frame: pd.DataFrame, keys: list[str]) -> None:
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise ValueError(f"process input missing columns: {missing}")
