"""Process Prepared rows with bounded, partition-local scientific semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .baseline import add_baseline_residuals
from .config import Config
from .features import add_dynamic_features, calculate_derived_features

_PARTITION_KEYS = ["experiment_id", "cycle_id", "cycle_stage"]
_SOURCE_QUALITY_SUFFIXES = ("__missing", "__invalid", "__duplicate", "__conflict")


def process(
    prepared: pd.DataFrame,
    initial_summary: pd.DataFrame,
    config: Config,
    channels: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the fixed Process order without modifying either input frame."""
    _require_keys(prepared, ["experiment_id", "timestamp", "cycle_id", "cycle_stage"])
    source = prepared.copy()
    source = source.loc[source["cycle_stage"].ne("partial")].copy()
    masked = _mask_duplicate_values(source, channels)
    interval_seconds = config.process.resample_interval_seconds
    resampled, excluded_transition_buckets = _resample(
        masked, initial_summary, channels, interval_seconds
    )
    coordinated = _recompute_cycle_coordinates(resampled, initial_summary)
    filled = _fill_missing(coordinated, channels, config)
    derived = calculate_derived_features(filled, channels)
    baselined, baseline_summary = add_baseline_residuals(
        derived, initial_summary, channels, config.process.baseline
    )
    featured = add_dynamic_features(
        baselined,
        channels,
        interval_seconds=interval_seconds,
        windows_minutes=list(config.process.feature_windows_minutes),
    )
    featured = featured.sort_values(["experiment_id", "timestamp"], kind="stable").reset_index(
        drop=True
    )
    final_summary = _update_summary(
        baseline_summary,
        featured,
        config,
        channels,
        excluded_transition_buckets,
    )
    return featured, final_summary


def _mask_duplicate_values(
    frame: pd.DataFrame, channels: Mapping[str, Mapping[str, Any]]
) -> pd.DataFrame:
    result = frame.copy()
    for name in channels:
        if name not in result:
            continue
        duplicate = _boolean_column(result, f"{name}__duplicate")
        conflict = _boolean_column(result, f"{name}__conflict")
        invalid = duplicate | conflict
        if invalid.any():
            result.loc[invalid, name] = np.nan
    quality_columns = [
        column
        for column in result.columns
        if str(column).endswith(_SOURCE_QUALITY_SUFFIXES)
        or str(column).endswith("__imputed")
    ]
    return result.drop(columns=quality_columns, errors="ignore")


def _boolean_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _resample(
    frame: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    channels: Mapping[str, Mapping[str, Any]],
    interval_seconds: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str], int]]:
    rows: list[dict[str, object]] = []
    excluded_transition_buckets: dict[tuple[str, str], int] = {}
    frequency = f"{interval_seconds}s"
    image_roles = _image_roles(frame)
    summary_lookup = _summary_lookup(cycle_summary)
    cycle_boundaries = _cycle_boundaries(cycle_summary)
    cycle_keys = ["experiment_id", "cycle_id"]
    for group_values, group in frame.groupby(cycle_keys, sort=False, dropna=False):
        ordered = group.sort_values("timestamp", kind="stable")
        buckets = ordered["timestamp"].dt.floor(frequency)
        grid = pd.date_range(buckets.min(), buckets.max(), freq=frequency)
        experiment_id, cycle_id = (str(value) for value in group_values)
        intervals = _cycle_stage_intervals(
            summary_lookup.get((experiment_id, cycle_id)), ordered
        )
        used_images: dict[str, set[str]] = {role: set() for role in image_roles}
        for timestamp in grid:
            if _crosses_cycle_boundary(
                timestamp, interval_seconds, cycle_boundaries.get(experiment_id, ())
            ):
                excluded_transition_buckets[(experiment_id, cycle_id)] = (
                    excluded_transition_buckets.get((experiment_id, cycle_id), 0) + 1
                )
                continue
            stage, overlap_tie = _winning_stage(timestamp, interval_seconds, intervals)
            if overlap_tie:
                excluded_transition_buckets[(experiment_id, cycle_id)] = (
                    excluded_transition_buckets.get((experiment_id, cycle_id), 0) + 1
                )
                continue
            if stage is None:
                continue
            bucket = ordered.loc[buckets.eq(timestamp) & ordered["cycle_stage"].eq(stage)]
            row = _identity_row(ordered, (experiment_id, cycle_id, stage), timestamp)
            for name, settings in channels.items():
                if str(settings.get("kind")) == "derived":
                    continue
                row[name] = _aggregate_channel(bucket, name, settings)
            for role in image_roles:
                _add_bucket_image(row, bucket, role, timestamp, used_images[role])
            rows.append(row)
    if not rows:
        empty = pd.DataFrame(columns=_processed_columns(channels, image_roles))
        return empty, excluded_transition_buckets
    result = pd.DataFrame(rows).sort_values(
        ["experiment_id", "timestamp"], kind="stable"
    )
    return result.reset_index(drop=True), excluded_transition_buckets


def _identity_row(
    ordered: pd.DataFrame, group_values: tuple[object, ...], timestamp: pd.Timestamp
) -> dict[str, object]:
    row = {key: value for key, value in zip(_PARTITION_KEYS, group_values, strict=True)}
    for column in ("experiment_date", "cycle_status", "cycle_status_reason"):
        if column in ordered:
            row[column] = ordered[column].iloc[0]
    row["timestamp"] = timestamp
    return row


def _summary_lookup(cycle_summary: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    lookup: dict[tuple[str, str], pd.Series] = {}
    for _, row in cycle_summary.iterrows():
        key = (str(row["experiment_id"]), str(row["cycle_id"]))
        lookup[key] = row
    return lookup


def _cycle_boundaries(cycle_summary: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, ...]]:
    boundaries: dict[str, list[pd.Timestamp]] = {}
    for _, row in cycle_summary.iterrows():
        experiment_id = str(row["experiment_id"])
        values = boundaries.setdefault(experiment_id, [])
        for column in ("heating_start", "defrost_end"):
            value = _as_timestamp(row.get(column))
            if value is not None:
                values.append(value)
    return {experiment_id: tuple(values) for experiment_id, values in boundaries.items()}


def _crosses_cycle_boundary(
    timestamp: pd.Timestamp,
    interval_seconds: int,
    boundaries: tuple[pd.Timestamp, ...],
) -> bool:
    bucket_end = timestamp + pd.Timedelta(seconds=interval_seconds)
    return any(timestamp < boundary < bucket_end for boundary in boundaries)


def _cycle_stage_intervals(
    summary: pd.Series | None, ordered: pd.DataFrame
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp | None]]:
    if summary is None:
        return _fallback_stage_intervals(ordered)
    heating = _as_timestamp(summary.get("heating_start"))
    stable = _as_timestamp(summary.get("stable_heating_start"))
    defrost = _as_timestamp(summary.get("defrost_start"))
    defrost_end = _as_timestamp(summary.get("defrost_end"))
    if heating is None or stable is None or defrost is None:
        return _fallback_stage_intervals(ordered)
    return {
        "recovery": (heating, stable),
        "frost_development": (stable, defrost),
        "defrost": (defrost, defrost_end),
    }


def _fallback_stage_intervals(
    ordered: pd.DataFrame,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp | None]]:
    stages = sorted(str(stage) for stage in ordered["cycle_stage"].dropna().unique())
    if len(stages) != 1:
        return {}
    start = pd.Timestamp(ordered["timestamp"].min())
    return {stages[0]: (start, None)}


def _winning_stage(
    timestamp: pd.Timestamp,
    interval_seconds: int,
    intervals: Mapping[str, tuple[pd.Timestamp, pd.Timestamp | None]],
) -> tuple[str | None, bool]:
    bucket_end = timestamp + pd.Timedelta(seconds=interval_seconds)
    overlaps: dict[str, float] = {}
    for stage, (start, end) in intervals.items():
        overlap_start = max(timestamp, start)
        overlap_end = bucket_end if end is None else min(bucket_end, end)
        seconds = (overlap_end - overlap_start).total_seconds()
        if seconds > 0:
            overlaps[stage] = seconds
    if not overlaps:
        return None, False
    maximum = max(overlaps.values())
    winners = [stage for stage, seconds in overlaps.items() if seconds == maximum]
    if len(winners) > 1:
        return None, True
    return winners[0], False


def _as_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value)


def _aggregate_channel(
    bucket: pd.DataFrame, name: str, settings: Mapping[str, Any]
) -> object:
    if name not in bucket:
        return np.nan
    values = bucket[name].dropna()
    if values.empty:
        return np.nan
    kind = str(settings.get("kind"))
    if kind in {"step", "event", "categorical"}:
        return values.iloc[-1]
    if kind == "protected" and str(settings.get("resample", "last")) != "mean":
        return values.iloc[-1]
    if str(settings.get("resample", "mean")) == "last":
        return values.iloc[-1]
    return pd.to_numeric(values, errors="coerce").mean()


def _image_roles(frame: pd.DataFrame) -> list[str]:
    prefix = "image_"
    suffix = "_path"
    roles = {
        str(column)[len(prefix) : -len(suffix)]
        for column in frame.columns
        if str(column).startswith(prefix) and str(column).endswith(suffix)
    }
    return sorted(roles)


def _add_bucket_image(
    row: dict[str, object],
    bucket: pd.DataFrame,
    role: str,
    timestamp: pd.Timestamp,
    used: set[str],
) -> None:
    path_column = f"image_{role}_path"
    time_column = f"image_{role}_time"
    offset_column = f"image_{role}_offset_seconds"
    if path_column not in bucket:
        row[path_column] = pd.NA
        row[time_column] = pd.NaT
        row[offset_column] = np.nan
        return
    candidate_columns = ["timestamp", path_column]
    if time_column in bucket:
        candidate_columns.append(time_column)
    candidates = bucket.loc[bucket[path_column].notna(), candidate_columns]
    if candidates.empty:
        row[path_column] = pd.NA
        row[time_column] = pd.NaT
        row[offset_column] = np.nan
        return
    candidate_rows: list[tuple[float, str, Any]] = []
    for _, candidate in candidates.iterrows():
        path = str(candidate[path_column])
        if path in used:
            continue
        distance = abs((pd.Timestamp(candidate["timestamp"]) - timestamp).total_seconds())
        image_time = (
            pd.to_datetime(candidate[time_column], errors="coerce")
            if time_column in bucket
            else pd.NaT
        )
        candidate_rows.append((distance, path, image_time))
    if not candidate_rows:
        row[path_column] = pd.NA
        row[time_column] = pd.NaT
        row[offset_column] = np.nan
        return
    _, path, image_time = min(candidate_rows, key=lambda value: (value[0], value[1]))
    used.add(path)
    row[path_column] = path
    row[time_column] = image_time
    row[offset_column] = (
        (pd.Timestamp(image_time) - timestamp).total_seconds()
        if pd.notna(image_time)
        else np.nan
    )


def _processed_columns(
    channels: Mapping[str, Mapping[str, Any]], image_roles: list[str]
) -> list[str]:
    columns = [
        *_PARTITION_KEYS,
        "experiment_date",
        "cycle_status",
        "cycle_status_reason",
        "timestamp",
    ]
    columns.extend(
        name for name, settings in channels.items() if settings.get("kind") != "derived"
    )
    for role in image_roles:
        columns.extend(
            [f"image_{role}_path", f"image_{role}_time", f"image_{role}_offset_seconds"]
        )
    return columns


def _recompute_cycle_coordinates(
    frame: pd.DataFrame, cycle_summary: pd.DataFrame
) -> pd.DataFrame:
    result = frame.copy()
    result["cycle_elapsed_seconds"] = np.nan
    result["cycle_progress"] = np.nan
    for _, cycle in cycle_summary.iterrows():
        stable = _timestamp_or_none(cycle.get("stable_heating_start"))
        defrost = _timestamp_or_none(cycle.get("defrost_start"))
        if stable is None or defrost is None:
            continue
        mask = result["experiment_id"].eq(cycle["experiment_id"]) & result["cycle_id"].eq(
            cycle["cycle_id"]
        )
        development = mask & result["cycle_stage"].eq("frost_development")
        if not development.any():
            continue
        elapsed = (result.loc[development, "timestamp"] - stable).dt.total_seconds()
        duration = (defrost - stable).total_seconds()
        if duration <= 0:
            continue
        result.loc[development, "cycle_elapsed_seconds"] = elapsed
        result.loc[development, "cycle_progress"] = (elapsed / duration).clip(0, 1)
    return result


def _timestamp_or_none(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value)


def _fill_missing(
    frame: pd.DataFrame, channels: Mapping[str, Mapping[str, Any]], config: Config
) -> pd.DataFrame:
    result = frame.copy()
    for name, settings in channels.items():
        if name not in result or str(settings.get("kind")) == "derived":
            continue
        imputed_column = f"{name}__imputed"
        result[imputed_column] = False
        policy = str(settings.get("missing", "none"))
        if policy not in {"interpolate", "linear", "forward_fill", "ffill"}:
            continue
        for _, group in result.groupby(_PARTITION_KEYS, sort=False, dropna=False):
            indices = group.index
            values = result.loc[indices, name].copy()
            timestamps = pd.Series(
                pd.to_datetime(result.loc[indices, "timestamp"].tolist(), errors="raise"),
                index=indices,
            )
            if str(settings.get("kind")) == "continuous" and policy in {
                "interpolate",
                "linear",
            }:
                filled = _fill_continuous(
                    values, timestamps, config.process.continuous_max_gap_seconds
                )
            elif str(settings.get("kind")) == "step" and policy in {"forward_fill", "ffill"}:
                filled = _fill_step(values, timestamps, config.process.control_max_gap_seconds)
            else:
                filled = values
            result.loc[indices, name] = filled.to_numpy()
            result.loc[indices, imputed_column] = (
                values.isna() & filled.notna()
            ).to_numpy()
    return result


def _fill_continuous(
    values: pd.Series, timestamps: pd.Series, maximum_gap_seconds: float
) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce").copy()
    observed_intervals = timestamps.diff().dt.total_seconds().dropna()
    if observed_intervals.empty:
        return result
    if maximum_gap_seconds < float(observed_intervals.median()):
        return result
    missing = result.isna().to_numpy()
    position = 0
    while position < len(result):
        if not missing[position]:
            position += 1
            continue
        end = _missing_run_end(missing, position)
        left = position - 1
        right = end + 1
        if left >= 0 and right < len(result):
            gap = (timestamps.iloc[right] - timestamps.iloc[left]).total_seconds()
            if gap <= maximum_gap_seconds:
                left_value = float(result.iloc[left])
                right_value = float(result.iloc[right])
                for index in range(position, end + 1):
                    fraction = (
                        (timestamps.iloc[index] - timestamps.iloc[left]).total_seconds() / gap
                    )
                    result.iloc[index] = left_value + fraction * (right_value - left_value)
        position = end + 1
    return result


def _fill_step(values: pd.Series, timestamps: pd.Series, maximum_gap_seconds: float) -> pd.Series:
    result = values.copy()
    missing = result.isna().to_numpy()
    position = 0
    while position < len(result):
        if not missing[position]:
            position += 1
            continue
        end = _missing_run_end(missing, position)
        left = position - 1
        if left >= 0:
            gap = (timestamps.iloc[end] - timestamps.iloc[left]).total_seconds()
            if gap <= maximum_gap_seconds:
                result.iloc[position : end + 1] = result.iloc[left]
        position = end + 1
    return result


def _missing_run_end(missing: np.ndarray, start: int) -> int:
    end = start
    while end + 1 < len(missing) and missing[end + 1]:
        end += 1
    return end


def _update_summary(
    summary: pd.DataFrame,
    processed: pd.DataFrame,
    config: Config,
    channels: Mapping[str, Mapping[str, Any]],
    excluded_transition_buckets: Mapping[tuple[str, str], int],
) -> pd.DataFrame:
    result = summary.copy()
    result["excluded_transition_bucket_count"] = [
        int(
            excluded_transition_buckets.get(
                (str(row["experiment_id"]), str(row["cycle_id"])), 0
            )
        )
        for _, row in result.iterrows()
    ]
    keys = ["experiment_id", "cycle_id"]
    if processed.empty:
        return result
    raw_names = [name for name, settings in channels.items() if settings.get("kind") != "derived"]
    records: list[dict[str, object]] = []
    for group_values, group in processed.groupby(keys, sort=False, dropna=False):
        available = group[raw_names].notna().mean().mean() if raw_names else np.nan
        imputed_columns = [f"{name}__imputed" for name in channels if f"{name}__imputed" in group]
        imputed_fraction = (
            float(group[imputed_columns].to_numpy(dtype=bool).mean())
            if imputed_columns
            else 0.0
        )
        gap = group["timestamp"].sort_values().diff().dt.total_seconds().max()
        records.append(
            {
                "experiment_id": group_values[0],
                "cycle_id": group_values[1],
                "processed_row_count": len(group),
                "processed_available_fraction": float(available),
                "processed_maximum_gap_seconds": float(gap) if pd.notna(gap) else 0.0,
                "imputed_fraction": imputed_fraction,
            }
        )
    metrics = pd.DataFrame(records)
    return result.merge(metrics, on=keys, how="left", validate="one_to_one")


def _require_keys(frame: pd.DataFrame, keys: list[str]) -> None:
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise ValueError(f"process input missing columns: {missing}")
