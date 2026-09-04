"""Process Prepared rows with bounded, partition-local scientific semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .baseline import add_baseline_residuals
from .dataset_settings import Config
from .features import calculate_derived_features
from .match_camera_images import image_columns, image_roles

_PARTITION_KEYS = ["experiment_id", "cycle_id", "cycle_stage"]
_SOURCE_QUALITY_SUFFIXES = ("__missing", "__invalid", "__duplicate", "__conflict")


def process(
    prepared: pd.DataFrame,
    initial_summary: pd.DataFrame,
    config: Config,
    channels: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the fixed Process order without modifying either input frame."""
    _require_keys(
        prepared,
        ["experiment_id", "timestamp", "cycle_id", "cycle_stage", "cycle_status"],
    )
    _validate_cycle_summary_input(prepared, initial_summary)
    scientific_source, fallback_source = _partition_process_inputs(prepared, initial_summary)
    scientific_source = scientific_source.loc[scientific_source["cycle_stage"].ne("partial")].copy()
    eligible_channels = _eligible_continuous_channels(prepared, channels)
    masked = _mask_duplicate_values(scientific_source, channels)
    interval_seconds = config.process.resample_interval_seconds
    (
        resampled,
        excluded_transition_buckets,
        low_coverage_buckets,
        eligible_channel_buckets,
        processed_cycles,
    ) = _resample(
        masked,
        initial_summary,
        channels,
        interval_seconds,
        expected_points_per_bucket=interval_seconds // config.expected_sensor_interval_seconds,
        minimum_coverage=config.process.minimum_continuous_bucket_coverage,
        eligible_channels=eligible_channels,
    )
    coordinated = recompute_cycle_coordinates(resampled, initial_summary)
    raw_system_quantities = _calculate_unfilled_system_quantities(coordinated, channels)
    filled = _fill_missing(coordinated, channels, config)
    derived = calculate_derived_features(filled, channels)
    for column in raw_system_quantities:
        name = str(column)
        derived[name] = raw_system_quantities[name]
        derived[f"{name}__imputed"] = False
    fallback = _resample_fallback(
        _mask_duplicate_values(fallback_source, channels),
        channels,
        interval_seconds,
    ).reindex(columns=derived.columns)
    for column in fallback.columns:
        if str(column).endswith("__imputed"):
            fallback[column] = False
    if derived.empty:
        derived = fallback.copy()
    elif not fallback.empty:
        derived = pd.concat([derived, fallback], ignore_index=True)
    derived = derived.sort_values(["experiment_id", "timestamp"], kind="stable").reset_index(
        drop=True
    )
    baselined, baseline_summary = add_baseline_residuals(
        derived, initial_summary, channels, config.process.baseline
    )
    final_summary = _update_summary(
        baseline_summary,
        baselined,
        excluded_transition_buckets,
        low_coverage_buckets,
        eligible_channel_buckets,
        processed_cycles,
    )
    return baselined, final_summary


def _partition_process_inputs(
    prepared: pd.DataFrame, cycle_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_lookup = _summary_lookup(cycle_summary)
    scientific_indices: list[object] = []
    fallback_indices: list[object] = []
    for group_values, group in prepared.groupby(
        ["experiment_id", "cycle_id"], sort=False, dropna=False
    ):
        key = (str(group_values[0]), str(group_values[1]))
        summary = summary_lookup[key]
        processable = _has_complete_boundaries(summary) and group["cycle_stage"].ne("partial").any()
        target = scientific_indices if processable else fallback_indices
        target.extend(group.index.tolist())
    return prepared.loc[scientific_indices].copy(), prepared.loc[fallback_indices].copy()


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
        if str(column).endswith(_SOURCE_QUALITY_SUFFIXES) or str(column).endswith("__imputed")
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
    *,
    expected_points_per_bucket: int,
    minimum_coverage: float,
    eligible_channels: Mapping[str, tuple[str, ...]],
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
    set[tuple[str, str]],
]:
    rows: list[dict[str, object]] = []
    excluded_transition_buckets: dict[tuple[str, str], int] = {}
    low_coverage_buckets: dict[tuple[str, str], int] = {}
    eligible_channel_buckets: dict[tuple[str, str], int] = {}
    processed_cycles: set[tuple[str, str]] = set()
    frequency = f"{interval_seconds}s"
    roles = image_roles(frame)
    summary_lookup = _summary_lookup(cycle_summary)
    cycle_keys = ["experiment_id", "cycle_id"]
    for group_values, group in frame.groupby(cycle_keys, sort=False, dropna=False):
        ordered = group.sort_values("timestamp", kind="stable")
        experiment_id, cycle_id = (str(value) for value in group_values)
        summary = summary_lookup[(experiment_id, cycle_id)]
        observed_end = ordered["timestamp"].max()
        if not _has_complete_boundaries(summary):
            continue
        grid = _cycle_grid(summary, frequency, observed_end=observed_end)
        buckets = ordered["timestamp"].dt.floor(frequency)
        processed_cycles.add((experiment_id, cycle_id))
        intervals = _cycle_stage_intervals(summary, observed_end=observed_end, frequency=frequency)
        image_records = {role: _image_records(ordered, role) for role in roles}
        cycle_key = (experiment_id, cycle_id)
        for timestamp in grid:
            stage, transition = _stage_for_complete_bucket(timestamp, interval_seconds, intervals)
            if transition:
                excluded_transition_buckets[cycle_key] = (
                    excluded_transition_buckets.get(cycle_key, 0) + 1
                )
                continue
            if stage is None:
                continue
            bucket = ordered.loc[buckets.eq(timestamp) & ordered["cycle_stage"].eq(stage)]
            row = _identity_row(ordered, (experiment_id, cycle_id, stage), timestamp)
            low_names, low_count, eligible_count = _coverage_for_bucket(
                bucket,
                eligible_channels.get(experiment_id, ()),
                expected_points_per_bucket,
                minimum_coverage,
            )
            eligible_channel_buckets[cycle_key] = (
                eligible_channel_buckets.get(cycle_key, 0) + eligible_count
            )
            low_coverage_buckets[cycle_key] = low_coverage_buckets.get(cycle_key, 0) + low_count
            for name, settings in channels.items():
                if str(settings.get("kind")) == "derived":
                    continue
                row[name] = (
                    np.nan if name in low_names else _aggregate_channel(bucket, name, settings)
                )
            for role in roles:
                _add_bucket_image(row, image_records[role], role, timestamp, interval_seconds)
            rows.append(row)
    if not rows:
        empty = pd.DataFrame(columns=_processed_columns(channels, list(roles)))
        return (
            empty,
            excluded_transition_buckets,
            low_coverage_buckets,
            eligible_channel_buckets,
            processed_cycles,
        )
    result = pd.DataFrame(rows).sort_values(["experiment_id", "timestamp"], kind="stable")
    return (
        result.reset_index(drop=True),
        excluded_transition_buckets,
        low_coverage_buckets,
        eligible_channel_buckets,
        processed_cycles,
    )


def _resample_fallback(
    frame: pd.DataFrame,
    channels: Mapping[str, Mapping[str, Any]],
    interval_seconds: int,
) -> pd.DataFrame:
    roles = image_roles(frame)
    if frame.empty:
        return pd.DataFrame(columns=_processed_columns(channels, list(roles)))
    source = frame.copy()
    cycle_origins = source.groupby(["experiment_id", "cycle_id"], sort=False, dropna=False)[
        "timestamp"
    ].transform("min")
    elapsed_seconds = (source["timestamp"] - cycle_origins).dt.total_seconds()
    source["_process_bucket"] = cycle_origins + pd.to_timedelta(
        (elapsed_seconds // interval_seconds) * interval_seconds,
        unit="s",
    )
    rows: list[dict[str, object]] = []
    keys = ["experiment_id", "cycle_id", "_process_bucket"]
    for group_values, bucket in source.groupby(keys, sort=False, dropna=False):
        ordered = bucket.sort_values("timestamp", kind="stable")
        experiment_id, cycle_id = (str(value) for value in group_values[:2])
        timestamp = pd.Timestamp(str(group_values[2]))
        stage = str(ordered["cycle_stage"].iloc[-1])
        row = _identity_row(ordered, (experiment_id, cycle_id, stage), timestamp)
        for name, settings in channels.items():
            if str(settings.get("kind")) != "derived":
                row[name] = _aggregate_channel(ordered, name, settings)
        for role in roles:
            _add_bucket_image(
                row,
                _image_records(ordered, role),
                role,
                timestamp,
                interval_seconds,
            )
        rows.append(row)
    result = (
        pd.DataFrame(rows)
        .sort_values(["experiment_id", "timestamp"], kind="stable")
        .reset_index(drop=True)
    )
    raw_system_quantities = _calculate_unfilled_system_quantities(result, channels)
    for column in raw_system_quantities:
        name = str(column)
        result[name] = raw_system_quantities[name].to_numpy()
        result[f"{name}__imputed"] = False
    return result


def _identity_row(
    ordered: pd.DataFrame, group_values: tuple[object, ...], timestamp: pd.Timestamp
) -> dict[str, object]:
    row = {key: value for key, value in zip(_PARTITION_KEYS, group_values, strict=True)}
    for column in ("experiment_date", "cycle_status", "cycle_status_reason"):
        if column in ordered:
            row[column] = ordered[column].iloc[0]
    row["timestamp"] = timestamp
    return row


def _cycle_stage_intervals(
    summary: pd.Series,
    *,
    observed_end: pd.Timestamp | None = None,
    frequency: str = "10s",
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    heating = _as_timestamp(summary.get("heating_start"))
    stable = _as_timestamp(summary.get("stable_heating_start"))
    preparation = _as_timestamp(summary.get("defrost_preparation_start"))
    defrost = _as_timestamp(summary.get("defrost_start"))
    defrost_end = _as_timestamp(summary.get("defrost_end"))
    if defrost_end is None and observed_end is not None:
        defrost_end = observed_end + pd.Timedelta(frequency)
    if heating is None or stable is None or defrost is None or defrost_end is None:
        raise ValueError("complete cycle is missing required stage boundaries")
    frost_end = preparation or defrost
    intervals = {
        "recovery": (heating, stable),
        "frost_development": (stable, frost_end),
        "defrost": (defrost, defrost_end),
    }
    if preparation is not None:
        intervals["defrost_preparation"] = (preparation, defrost)
    return intervals


def _stage_for_complete_bucket(
    timestamp: pd.Timestamp,
    interval_seconds: int,
    intervals: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[str | None, bool]:
    bucket_end = timestamp + pd.Timedelta(seconds=interval_seconds)
    boundaries = {boundary for interval in intervals.values() for boundary in interval}
    if any(timestamp < boundary < bucket_end for boundary in boundaries):
        return None, True
    for stage, (start, end) in intervals.items():
        if start <= timestamp < end:
            return stage, False
    return None, False


def _as_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value)


def _aggregate_channel(bucket: pd.DataFrame, name: str, settings: Mapping[str, Any]) -> object:
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


def _coverage_for_bucket(
    bucket: pd.DataFrame,
    channel_names: tuple[str, ...],
    expected_points: int,
    minimum_coverage: float,
) -> tuple[set[str], int, int]:
    low_names = {
        name
        for name in channel_names
        if (int(bucket[name].notna().sum()) if name in bucket else 0) / expected_points
        < minimum_coverage
    }
    return low_names, len(low_names), len(channel_names)


def _add_bucket_image(
    row: dict[str, object],
    records: list[tuple[str, pd.Timestamp]],
    role: str,
    timestamp: pd.Timestamp,
    interval_seconds: int,
) -> None:
    path_column, time_column, offset_column = image_columns(role)
    bucket_end = timestamp + pd.Timedelta(seconds=interval_seconds)
    candidates = [
        (path, image_time) for path, image_time in records if timestamp <= image_time < bucket_end
    ]
    if not candidates:
        row[path_column] = pd.NA
        row[time_column] = pd.NaT
        row[offset_column] = np.nan
        return
    path, image_time = min(
        candidates,
        key=lambda value: (abs((value[1] - timestamp).total_seconds()), value[0]),
    )
    row[path_column] = path
    row[time_column] = image_time
    row[offset_column] = (image_time - timestamp).total_seconds()


def _image_records(frame: pd.DataFrame, role: str) -> list[tuple[str, pd.Timestamp]]:
    path_column, time_column, _ = image_columns(role)
    if path_column not in frame or time_column not in frame:
        return []
    records = frame.loc[frame[path_column].notna(), [path_column, time_column]].copy()
    records[time_column] = pd.to_datetime(records[time_column], errors="coerce")
    records = records.dropna(subset=[time_column]).drop_duplicates()
    return [
        (str(path), pd.Timestamp(image_time))
        for path, image_time in records.itertuples(index=False)
    ]


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
    columns.extend(name for name, settings in channels.items() if settings.get("kind") != "derived")
    for role in image_roles:
        columns.extend(image_columns(role))
    return columns


def recompute_cycle_coordinates(frame: pd.DataFrame, cycle_summary: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["cycle_elapsed_seconds"] = np.nan
    result["cycle_progress"] = np.nan
    for _, cycle in cycle_summary.iterrows():
        stable = _timestamp_or_none(cycle.get("stable_heating_start"))
        defrost = _timestamp_or_none(cycle.get("defrost_start"))
        frost_end = _timestamp_or_none(cycle.get("defrost_preparation_start")) or defrost
        if stable is None or defrost is None:
            continue
        mask = result["experiment_id"].eq(cycle["experiment_id"]) & result["cycle_id"].eq(
            cycle["cycle_id"]
        )
        development = mask & result["cycle_stage"].eq("frost_development")
        if not development.any():
            continue
        elapsed = (result.loc[development, "timestamp"] - stable).dt.total_seconds()
        duration = (frost_end - stable).total_seconds()
        if duration <= 0:
            continue
        if (elapsed < 0).any():
            raise ValueError("frost_development cycle_elapsed_seconds must be nonnegative")
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
        missing_method = str(settings.get("missing", "none"))
        if missing_method not in {"interpolate", "linear", "forward_fill", "ffill"}:
            continue
        for _, group in result.groupby(_PARTITION_KEYS, sort=False, dropna=False):
            indices = group.index
            values = result.loc[indices, name].copy()
            timestamps = pd.Series(
                pd.to_datetime(result.loc[indices, "timestamp"].tolist(), errors="raise"),
                index=indices,
            )
            if str(settings.get("kind")) == "continuous" and missing_method in {
                "interpolate",
                "linear",
            }:
                filled = _fill_continuous(
                    values, timestamps, config.process.continuous_max_gap_seconds
                )
            elif str(settings.get("kind")) == "step" and missing_method in {
                "forward_fill",
                "ffill",
            }:
                filled = _fill_step(values, timestamps, config.process.control_max_gap_seconds)
            else:
                filled = values
            result.loc[indices, name] = filled.to_numpy()
            result.loc[indices, imputed_column] = (values.isna() & filled.notna()).to_numpy()
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
                        timestamps.iloc[index] - timestamps.iloc[left]
                    ).total_seconds() / gap
                    result.iloc[index] = left_value + fraction * (right_value - left_value)
        position = end + 1
    return result


def _fill_step(values: pd.Series, timestamps: pd.Series, maximum_gap_seconds: float) -> pd.Series:
    result = values.copy()
    last_observed_time: pd.Timestamp | None = None
    last_observed_value: object = np.nan
    for position in range(len(result)):
        value = result.iloc[position]
        if pd.isna(value):
            if last_observed_time is not None:
                gap = (timestamps.iloc[position] - last_observed_time).total_seconds()
                if gap <= maximum_gap_seconds:
                    result.iloc[position] = last_observed_value
            continue
        last_observed_time = pd.Timestamp(timestamps.iloc[position])
        last_observed_value = value
    return result


def _missing_run_end(missing: np.ndarray, start: int) -> int:
    end = start
    while end + 1 < len(missing) and missing[end + 1]:
        end += 1
    return end


def _update_summary(
    summary: pd.DataFrame,
    processed: pd.DataFrame,
    excluded_transition_buckets: Mapping[tuple[str, str], int],
    low_coverage_buckets: Mapping[tuple[str, str], int],
    eligible_channel_buckets: Mapping[tuple[str, str], int],
    processed_cycles: set[tuple[str, str]],
) -> pd.DataFrame:
    result = summary.drop(
        columns=["processed_available_fraction", "imputed_fraction"], errors="ignore"
    ).copy()
    summary_keys = [
        (str(row["experiment_id"]), str(row["cycle_id"])) for _, row in result.iterrows()
    ]
    result["excluded_transition_bucket_count"] = [
        int(excluded_transition_buckets.get(key, 0)) if key in processed_cycles else np.nan
        for key in summary_keys
    ]
    result["low_coverage_channel_bucket_count"] = [
        int(low_coverage_buckets.get(key, 0)) if key in processed_cycles else np.nan
        for key in summary_keys
    ]
    result["eligible_continuous_channel_bucket_count"] = [
        int(eligible_channel_buckets.get(key, 0)) if key in processed_cycles else np.nan
        for key in summary_keys
    ]
    keys = ["experiment_id", "cycle_id"]
    if processed.empty:
        return result
    records: list[dict[str, object]] = []
    for group_values, group in processed.groupby(keys, sort=False, dropna=False):
        gap = group["timestamp"].sort_values().diff().dt.total_seconds().max()
        records.append(
            {
                "experiment_id": group_values[0],
                "cycle_id": group_values[1],
                "processed_row_count": len(group),
                "processed_maximum_gap_seconds": float(gap) if pd.notna(gap) else np.nan,
            }
        )
    metrics = pd.DataFrame(records)
    return result.merge(metrics, on=keys, how="left", validate="one_to_one")


def _require_keys(frame: pd.DataFrame, keys: list[str]) -> None:
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise ValueError(f"process input missing columns: {missing}")


def _eligible_continuous_channels(
    frame: pd.DataFrame, channels: Mapping[str, Mapping[str, Any]]
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for experiment_id, group in frame.groupby("experiment_id", sort=False, dropna=False):
        names = [
            name
            for name, settings in channels.items()
            if str(settings.get("kind")) == "continuous"
            and f"{name}__missing" in group
            and not _boolean_column(group, f"{name}__missing").all()
        ]
        result[str(experiment_id)] = tuple(names)
    return result


def _has_complete_boundaries(summary: pd.Series) -> bool:
    return all(
        _as_timestamp(summary.get(column)) is not None
        for column in ("heating_start", "stable_heating_start", "defrost_start", "defrost_end")
    )


def _calculate_unfilled_system_quantities(
    frame: pd.DataFrame, channels: Mapping[str, Mapping[str, Any]]
) -> pd.DataFrame:
    settings = {
        name: value
        for name, value in channels.items()
        if name in {"cop", "evaporator_capacity"} and str(value.get("kind")) == "derived"
    }
    return calculate_derived_features(frame, settings).loc[:, list(settings)]


def _cycle_grid(
    summary: pd.Series,
    frequency: str,
    *,
    observed_end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    start = _as_timestamp(summary.get("heating_start"))
    end = _as_timestamp(summary.get("defrost_end"))
    open_cycle = end is None
    if open_cycle:
        end = observed_end
    if start is None or end is None:
        return pd.DatetimeIndex([])
    last = (
        end.floor(frequency) if open_cycle else (end - pd.Timedelta(nanoseconds=1)).floor(frequency)
    )
    return pd.date_range(start.floor(frequency), last, freq=frequency)


def _summary_lookup(cycle_summary: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    lookup: dict[tuple[str, str], pd.Series] = {}
    for _, row in cycle_summary.iterrows():
        key = (str(row["experiment_id"]), str(row["cycle_id"]))
        if key in lookup:
            raise ValueError(f"duplicate cycle summary for cycle {key[1]}")
        lookup[key] = row
    return lookup


def _validate_cycle_summary_input(prepared: pd.DataFrame, summary: pd.DataFrame) -> None:
    lookup = _summary_lookup(summary)
    grouped = prepared.groupby(["experiment_id", "cycle_id"], sort=False, dropna=False)
    prepared_keys: set[tuple[str, str]] = set()
    for group_values, group in grouped:
        key = (str(group_values[0]), str(group_values[1]))
        prepared_keys.add(key)
        if key not in lookup:
            raise ValueError(f"missing cycle summary for cycle {key[1]}")
        status = str(group["cycle_status"].iloc[0])
        if status not in {"valid", "invalid"}:
            raise ValueError(f"invalid cycle status for cycle {key[1]}")
    summary_only = set(lookup) - prepared_keys
    if summary_only:
        cycle_id = sorted(summary_only)[0][1]
        raise ValueError(f"cycle summary {cycle_id} is without Prepared rows")
