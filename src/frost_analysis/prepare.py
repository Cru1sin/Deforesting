"""Prepare raw sensor and image records without reconstructing measurements."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, resolved_config_sha256
from .cycles import label_cycles
from .images import match_images
from .io import discover_inputs, git_commit, optional_sha256, source_file_metadata


def prepare(
    config: Config, channels: Mapping[str, Mapping[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load raw observations, segment cycles, and attach one-shot image matches."""
    inputs = discover_inputs(config)
    if not inputs.sensor_files:
        raise ValueError(f"no sensor files found in {config.input_dir}")
    camera_roles = config.camera_roles
    channel_frames, invalid_timestamp_rows = _load_channel_frames(
        inputs.sensor_files, config.input_dir, channels, config.timestamp_column
    )
    available_source_channels = set(channel_frames)
    timestamps = _all_timestamps(channel_frames)
    if timestamps.empty:
        raise ValueError("sensor files contain no valid timestamps")
    prepared = pd.DataFrame({"timestamp": timestamps})
    prepared.insert(0, "experiment_date", config.experiment_date)
    prepared.insert(0, "experiment_id", config.experiment_id)
    unavailable_channels: list[str] = []
    channel_columns: dict[str, pd.Series] = {}
    for name, settings in channels.items():
        if str(settings.get("kind")) == "derived":
            continue
        if name not in channel_frames:
            unavailable_channels.append(name)
        values = _combine_channel(channel_frames.get(name, []), settings, timestamps)
        channel_columns[name] = values["value"]
        for suffix in ("__missing", "__invalid", "__duplicate", "__conflict"):
            channel_columns[f"{name}{suffix}"] = values[suffix]

    prepared = pd.concat(
        [prepared, pd.DataFrame(channel_columns, index=prepared.index)], axis=1
    )

    defrost_channel = config.cycles.defrost_channel
    if defrost_channel not in prepared:
        prepared[defrost_channel] = pd.Series(pd.NA, index=prepared.index, dtype="boolean")
        prepared[f"{defrost_channel}__missing"] = True
        prepared[f"{defrost_channel}__invalid"] = False
        prepared[f"{defrost_channel}__duplicate"] = False
        prepared[f"{defrost_channel}__conflict"] = False
    prepared, cycle_summary = label_cycles(
        prepared,
        defrost_channel,
        config.cycles,
        experiment_id=config.experiment_id,
        experiment_date=config.experiment_date,
    )
    image_matches = match_images(
        prepared["timestamp"],
        [path.relative_to(config.input_dir) for path in inputs.image_files],
        camera_roles=camera_roles,
        tolerance_seconds=config.image_match_tolerance_seconds,
    )
    for column in image_matches.columns:
        prepared[column] = image_matches[column].to_numpy()
    prepared = prepared.sort_values(["experiment_id", "timestamp"], kind="stable").reset_index(
        drop=True
    )
    cycle_summary = _add_cycle_summary_metrics(
        prepared,
        cycle_summary,
        channels,
        camera_roles,
        config.expected_sensor_interval_seconds,
        available_source_channels,
    )
    prepare_summary = {
        "experiment_id": config.experiment_id,
        "experiment_date": config.experiment_date,
        "input_dir": str(config.input_dir),
        "config_path": str(config.config_path) if config.config_path else None,
        "created_at": datetime.now(UTC).isoformat(),
        "sensor_files": [
            source_file_metadata(path, config.input_dir) for path in inputs.sensor_files
        ],
        "image_file_count": len(inputs.image_files),
        "sensor_file_count": len(inputs.sensor_files),
        "prepared_row_count": len(prepared),
        "cycle_count": len(cycle_summary),
        "config_sha256": optional_sha256(config.config_path),
        "defaults_path": str(config.defaults_path) if config.defaults_path else None,
        "defaults_sha256": optional_sha256(config.defaults_path),
        "resolved_config_sha256": resolved_config_sha256(config),
        "channels_path": str(config.channels_path),
        "channels_sha256": optional_sha256(config.channels_path),
        "discovered_camera_ids": sorted({path.parent.name for path in inputs.image_files}),
        "mapped_camera_ids": sorted(camera_roles),
        "missing_camera_ids": sorted(
            set(camera_roles) - {path.parent.name for path in inputs.image_files}
        ),
        "unavailable_channels": unavailable_channels,
        "invalid_timestamp_row_count": invalid_timestamp_rows,
        "git_commit": git_commit(config.project_root),
    }
    return prepared, cycle_summary, prepare_summary


def _load_channel_frames(
    paths: tuple[Path, ...],
    input_dir: Path,
    channels: Mapping[str, Mapping[str, Any]],
    timestamp_column: str,
) -> tuple[dict[str, list[pd.DataFrame]], int]:
    source_to_channels: dict[str, set[str]] = defaultdict(set)
    for name, settings in channels.items():
        if str(settings.get("kind")) == "derived":
            continue
        for source in settings.get("source_names", []):
            source_to_channels[str(source)].add(name)

    frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    invalid_timestamp_rows = 0
    for path in paths:
        table, invalid_count = _read_sensor_table(path, timestamp_column)
        invalid_timestamp_rows += invalid_count
        if table.empty:
            continue
        group = _parameter_group(path)
        for raw_column in table.columns:
            if raw_column == "timestamp":
                continue
            canonical = _canonical_name(group, str(raw_column))
            channel_names = source_to_channels.get(canonical, set())
            for name in channel_names:
                frames[name].append(
                    pd.DataFrame({"timestamp": table["timestamp"], "raw": table[raw_column]})
                )
    return frames, invalid_timestamp_rows


def _read_sensor_table(path: Path, timestamp_column: str) -> tuple[pd.DataFrame, int]:
    sample = path.read_bytes()[:131_072]
    if sample.startswith((b"\xd0\xcf\x11\xe0", b"PK\x03\x04")):
        raise ValueError(f"binary Excel workbook is not supported: {path}")
    encoding_errors = "strict"
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            decoded = sample.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        encoding = "gb18030"
        encoding_errors = "replace"
        decoded = sample.decode(encoding, errors=encoding_errors)
    delimiter = _detect_delimiter(decoded)
    table = pd.read_csv(
        path,
        sep=delimiter,
        encoding=encoding,
        dtype=str,
        keep_default_na=False,
        engine="python",
        encoding_errors=encoding_errors,
    )
    table.columns = [str(column).strip() for column in table.columns]
    if timestamp_column not in table.columns:
        raise ValueError(f"timestamp column {timestamp_column!r} not found in {path}")
    timestamps = pd.to_datetime(table.pop(timestamp_column), errors="coerce")
    invalid_count = int(timestamps.isna().sum())
    table = pd.concat([timestamps.rename("timestamp"), table], axis=1)
    return table.loc[table["timestamp"].notna()].reset_index(drop=True), invalid_count


def _combine_channel(
    frames: list[pd.DataFrame], settings: Mapping[str, Any], timestamps: pd.Series
) -> pd.DataFrame:
    if not frames:
        empty = pd.DataFrame(index=timestamps.index)
        empty["value"] = np.nan
        empty["__missing"] = True
        empty["__invalid"] = False
        empty["__duplicate"] = False
        empty["__conflict"] = False
        return empty
    records = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="stable")
    kind = str(settings.get("kind"))
    raw_values = records["raw"].astype("string").str.strip()
    cleaned, invalid = _parse_values(raw_values, kind, settings)
    work = pd.DataFrame(
        {
            "timestamp": records["timestamp"].to_numpy(),
            "value": cleaned.to_numpy(),
            "invalid": invalid.to_numpy(),
            "nonempty": (raw_values.ne("") & raw_values.notna()).to_numpy(),
        }
    )
    grouped = work.groupby("timestamp", sort=True)
    counts = grouped.size()
    invalid_any = grouped["invalid"].any()
    nonempty_any = grouped["nonempty"].any()
    first_value = grouped["value"].first()
    valid_values = work.loc[~work["invalid"] & work["value"].notna(), ["timestamp", "value"]]
    valid_values = valid_values.assign(value_key=valid_values["value"].astype("string"))
    unique_counts = valid_values.groupby("timestamp")["value_key"].nunique()
    index = pd.DatetimeIndex(timestamps)
    result = pd.DataFrame(index=index)
    counts_aligned = counts.reindex(index, fill_value=0)
    invalid_aligned = invalid_any.reindex(index, fill_value=False)
    nonempty_aligned = nonempty_any.reindex(index, fill_value=False)
    unique_counts_aligned = unique_counts.reindex(index, fill_value=0)
    result["value"] = first_value.reindex(index).where(
        counts_aligned.eq(1) & ~invalid_aligned, np.nan
    )
    result["__missing"] = ~nonempty_aligned
    result["__invalid"] = invalid_aligned
    result["__duplicate"] = counts_aligned.gt(1)
    result["__conflict"] = unique_counts_aligned.gt(1)
    result["__missing"] = result["__missing"].astype(bool)
    for column in ("__invalid", "__duplicate", "__conflict"):
        result[column] = result[column].astype(bool)
    return result.reset_index(drop=True)


def _parse_values(
    raw_values: pd.Series, kind: str, settings: Mapping[str, Any]
) -> tuple[pd.Series, pd.Series]:
    nonempty = raw_values.ne("") & raw_values.notna()
    if kind == "event":
        allowed = settings.get("allowed_values", {})
        mapping = {str(key).strip().upper(): bool(value) for key, value in allowed.items()}
        values = raw_values.map(lambda value: mapping.get(str(value).strip().upper(), np.nan))
        invalid = nonempty & values.isna()
        return values.astype("object"), invalid
    if kind == "categorical":
        return raw_values.where(nonempty, pd.NA).astype("object"), pd.Series(
            False, index=raw_values.index, dtype=bool
        )
    numeric = pd.to_numeric(raw_values.replace("", pd.NA), errors="coerce")
    invalid = nonempty & numeric.isna()
    scale = float(settings.get("scale", 1.0))
    offset = float(settings.get("offset", 0.0))
    numeric = numeric * scale + offset
    valid_range = settings.get("valid_range")
    if isinstance(valid_range, list) and len(valid_range) == 2:
        lower, upper = float(valid_range[0]), float(valid_range[1])
        out_of_range = numeric.notna() & ~numeric.between(lower, upper)
        invalid = invalid | out_of_range
        numeric = numeric.mask(out_of_range)
    return numeric.astype(float), invalid


def _all_timestamps(channel_frames: Mapping[str, list[pd.DataFrame]]) -> pd.Series:
    values = [frame["timestamp"] for frames in channel_frames.values() for frame in frames]
    if not values:
        return pd.Series(dtype="datetime64[ns]")
    unique = pd.concat(values, ignore_index=True).drop_duplicates().sort_values()
    return pd.Series(unique.to_numpy())


def _add_cycle_summary_metrics(
    prepared: pd.DataFrame,
    summary: pd.DataFrame,
    channels: Mapping[str, Mapping[str, Any]],
    camera_roles: Mapping[str, str],
    expected_interval_seconds: int,
    available_source_channels: set[str],
) -> pd.DataFrame:
    raw_channels = [
        name for name, settings in channels.items() if str(settings.get("kind")) != "derived"
    ]
    role_path_columns = [f"image_{role}_path" for role in sorted(set(camera_roles.values()))]
    role_time_columns = [f"image_{role}_time" for role in sorted(set(camera_roles.values()))]
    result = summary.copy()
    records: list[dict[str, object]] = []
    for _, row in result.iterrows():
        cycle_id = row["cycle_id"]
        group = prepared.loc[prepared["cycle_id"].eq(cycle_id)]
        raw_count = len(group)
        expected = _expected_row_count(row, expected_interval_seconds)
        observed_fraction = _observed_fraction(
            group, raw_channels, available_source_channels
        )
        maximum_gap = _maximum_gap(group["timestamp"])
        duplicate_count = _quality_row_count(group, raw_channels, "__duplicate")
        conflict_count = _quality_row_count(group, raw_channels, "__conflict")
        image_values = group[role_path_columns].to_numpy().ravel()
        image_count = int(pd.Series(image_values).dropna().nunique()) if image_values.size else 0
        role_count = sum(int(group[column].notna().any()) for column in role_path_columns)
        complete_fraction = role_count / len(role_path_columns) if role_path_columns else 0.0
        image_gaps = [
            _maximum_image_gap(group[column].dropna())
            for column in role_time_columns
            if column in group
        ]
        image_gaps = [value for value in image_gaps if pd.notna(value)]
        image_gap = max(image_gaps) if image_gaps else np.nan
        records.append(
            {
                "raw_row_count": raw_count,
                "expected_row_count": expected,
                "sensor_observed_fraction": observed_fraction,
                "maximum_timeline_gap_seconds": maximum_gap,
                "duplicate_observation_count": duplicate_count,
                "conflict_observation_count": conflict_count,
                "rgb_image_count": image_count,
                "rgb_role_count": role_count,
                "rgb_role_presence_fraction": complete_fraction,
                "maximum_rgb_gap_seconds": image_gap,
            }
        )
    return pd.concat([result.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def _expected_row_count(row: pd.Series, interval_seconds: int) -> object:
    start = row.get("heating_start")
    end = row.get("defrost_end")
    if not isinstance(start, pd.Timestamp) or not isinstance(end, pd.Timestamp):
        return np.nan
    return int(np.ceil((end - start).total_seconds() / interval_seconds))


def _observed_fraction(
    group: pd.DataFrame, channels: list[str], available_source_channels: set[str]
) -> float:
    available = [
        name for name in channels if name in available_source_channels and name in group
    ]
    if group.empty or not available:
        return 0.0
    return float(group[available].notna().mean().mean())


def _maximum_gap(timestamps: pd.Series) -> float:
    parsed = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()
    if len(parsed) < 2:
        return np.nan
    return float(parsed.diff().dt.total_seconds().dropna().max())


def _maximum_image_gap(timestamps: pd.Series) -> float:
    parsed = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()
    if len(parsed) < 2:
        return np.nan
    return float(parsed.diff().dt.total_seconds().dropna().max())


def _quality_row_count(group: pd.DataFrame, channels: list[str], suffix: str) -> int:
    columns = [f"{name}{suffix}" for name in channels if f"{name}{suffix}" in group]
    if group.empty or not columns:
        return 0
    return int(group[columns].astype(bool).any(axis=1).sum())


def _parameter_group(path: Path) -> str | None:
    match = re.search(r"参数(?P<group>\d+)", path.stem)
    return None if match is None else match.group("group")


def _canonical_name(group: str | None, raw_column: str) -> str:
    return f"p{group}__{raw_column}" if group else raw_column


def _detect_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        counts = {delimiter: sample.count(delimiter) for delimiter in ("\t", ",", ";")}
        delimiter, count = max(counts.items(), key=lambda item: item[1])
        if count == 0:
            raise ValueError("no supported delimiter") from None
        return delimiter
