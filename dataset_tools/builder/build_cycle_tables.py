"""Process Prepared rows with bounded, partition-local scientific semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

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
    filled = coordinated
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
    from heat_smoothing.methods import process_measurements

    print(
        "[add] interpolate sensors, then smooth heat and pressure (offline + trailing 60 s)",
        flush=True,
    )
    derived = process_measurements(
        _mask_duplicate_values(prepared, channels), derived, channels=channels
    )
    final_summary = _update_summary(
        initial_summary,
        derived,
        excluded_transition_buckets,
        low_coverage_buckets,
        eligible_channel_buckets,
        processed_cycles,
    )
    if "user_side_fan_setting" in channels:
        from dataset_tools.cycle_metadata import user_side_fan_setting

        # Campaign metadata also applies to buckets without sensor observations.
        derived["user_side_fan_setting"] = user_side_fan_setting(
            config.experiment_date, derived.get("water_temperature_setpoint")
        )

    return derived, final_summary


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
    parts = []
    excluded_transition_buckets, low_coverage_buckets, eligible_channel_buckets = {}, {}, {}
    processed_cycles = set()
    frequency = f"{interval_seconds}s"
    roles = image_roles(frame)
    summary_lookup = _summary_lookup(cycle_summary)
    for key, group in frame.groupby(["experiment_id", "cycle_id"], sort=False, dropna=False):
        ordered = group.sort_values("timestamp", kind="stable")
        key = tuple(str(value) for value in key)
        summary = summary_lookup[key]
        if not _has_complete_boundaries(summary):
            continue
        grid = _cycle_grid(summary, frequency, observed_end=ordered.timestamp.max())
        intervals = _cycle_stage_intervals(
            summary, observed_end=ordered.timestamp.max(), frequency=frequency
        )
        stages = pd.Series(pd.NA, index=grid, dtype="string")
        transitions = np.zeros(len(grid), dtype=bool)
        for stage, (start, end) in intervals.items():
            stages.loc[(grid >= start) & (grid + pd.Timedelta(seconds=interval_seconds) <= end)] = (
                stage
            )
            for boundary in (start, end):
                transitions |= (grid < boundary) & (
                    boundary < grid + pd.Timedelta(seconds=interval_seconds)
                )
        excluded_transition_buckets[key] = int(transitions.sum())
        stages = stages.loc[stages.notna() & ~transitions]
        source = ordered.assign(_process_bucket=ordered.timestamp.dt.floor(frequency))
        matching = source._process_bucket.map(stages)
        source = source.loc[source.cycle_stage.eq(matching)]
        grouped = source.groupby("_process_bucket", sort=False)
        result = _aggregate_buckets(grouped, channels).reindex(stages.index)
        names = eligible_channels.get(key[0], ())
        low_count = 0
        for name in names:
            counts = (
                grouped[name].count().reindex(stages.index, fill_value=0)
                if name in source
                else pd.Series(0, index=stages.index)
            )
            low = counts / expected_points_per_bucket < minimum_coverage
            result.loc[low, name] = np.nan
            low_count += int(low.sum())
        low_coverage_buckets[key] = low_count
        eligible_channel_buckets[key] = len(stages) * len(names)
        processed_cycles.add(key)
        result["cycle_stage"] = stages
        result = _complete_bucket_table(result, ordered, key, roles, interval_seconds)
        parts.append(result)
    result = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=_processed_columns(channels, list(roles)))
    )
    return (
        result.sort_values(["experiment_id", "timestamp"], kind="stable").reset_index(drop=True),
        excluded_transition_buckets,
        low_coverage_buckets,
        eligible_channel_buckets,
        processed_cycles,
    )


def _resample_fallback(frame, channels, interval_seconds):
    roles = image_roles(frame)
    parts = []
    for key, group in frame.groupby(["experiment_id", "cycle_id"], sort=False, dropna=False):
        ordered = group.sort_values("timestamp", kind="stable")
        origin = ordered.timestamp.min()
        buckets = origin + pd.to_timedelta(
            ((ordered.timestamp - origin).dt.total_seconds() // interval_seconds)
            * interval_seconds,
            unit="s",
        )
        grouped = ordered.assign(_process_bucket=buckets).groupby("_process_bucket", sort=False)
        result = _aggregate_buckets(grouped, channels)
        result["cycle_stage"] = grouped.cycle_stage.last()
        result = _complete_bucket_table(result, ordered, key, roles, interval_seconds)
        parts.append(result)
    if not parts:
        return pd.DataFrame(columns=_processed_columns(channels, list(roles)))
    result = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["experiment_id", "timestamp"], kind="stable")
        .reset_index(drop=True)
    )
    raw_system_quantities = _calculate_unfilled_system_quantities(result, channels)
    for name in raw_system_quantities:
        result[name] = raw_system_quantities[name].to_numpy()
        result[f"{name}__imputed"] = False
    return result


def _aggregate_buckets(grouped, channels):
    """The same channel reducers for every interval, evaluated in one groupby."""
    reducers = {}
    for name, settings in channels.items():
        if str(settings.get("kind")) == "derived" or name not in grouped.obj:
            continue
        kind = str(settings.get("kind"))
        last = (
            kind in {"step", "event", "categorical"}
            or (kind == "protected" and settings.get("resample", "last") != "mean")
            or settings.get("resample", "mean") == "last"
        )
        reducers[name] = "last" if last else "mean"
    result = grouped.agg(reducers)
    for name, settings in channels.items():
        if str(settings.get("kind")) != "derived" and name not in result:
            result[name] = np.nan
    return result


def _complete_bucket_table(result, ordered, key, roles, interval_seconds):
    result = result.copy()
    result["experiment_id"], result["cycle_id"] = key
    for name in ("experiment_date", "cycle_status", "cycle_status_reason"):
        if name in ordered:
            result[name] = ordered[name].iloc[0]
    result.index.name = "timestamp"
    if roles:
        records = {role: _image_records(ordered, role) for role in roles}
        rows = []
        for timestamp in result.index:
            row = {}
            for role in roles:
                _add_bucket_image(row, records[role], role, timestamp, interval_seconds)
            rows.append(row)
        images = pd.DataFrame(rows, index=result.index)
        result = pd.concat([result, images], axis=1)
    return result.reset_index()


def _cycle_stage_intervals(
    summary: pd.Series,
    *,
    observed_end: pd.Timestamp | None = None,
    frequency: str = "1s",
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


def _as_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value)


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
