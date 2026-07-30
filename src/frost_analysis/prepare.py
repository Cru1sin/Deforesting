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

from .config import Config
from .cycles import label_cycles
from .images import match_images
from .io import discover_inputs


def prepare(
    config: Config, channels: Mapping[str, Mapping[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load raw observations, segment cycles, and attach one-shot image matches."""
    inputs = discover_inputs(config)
    if not inputs.sensor_files:
        raise ValueError(f"no sensor files found in {config.input_dir}")
    channel_frames = _load_channel_frames(inputs.sensor_files, config.input_dir, channels)
    timestamps = _all_timestamps(channel_frames)
    if timestamps.empty:
        raise ValueError("sensor files contain no valid timestamps")
    prepared = pd.DataFrame({"timestamp": timestamps})
    prepared.insert(0, "experiment_date", config.experiment_date)
    prepared.insert(0, "experiment_id", config.experiment_id)
    for name, settings in channels.items():
        if str(settings.get("kind")) == "derived":
            continue
        values = _combine_channel(channel_frames.get(name, []), settings, timestamps)
        prepared[name] = values["value"].to_numpy()
        for suffix in ("__missing", "__invalid", "__duplicate", "__conflict"):
            prepared[f"{name}{suffix}"] = values[suffix].to_numpy()

    defrost_channel = str(config.cycles.get("defrost_channel", "defrost_active"))
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
        tolerance_seconds=float(config.cycles.get("image_tolerance_seconds", 2)),
    )
    for column in image_matches.columns:
        prepared[column] = image_matches[column].to_numpy()
    prepared = prepared.sort_values(["experiment_id", "timestamp"], kind="stable").reset_index(
        drop=True
    )
    prepare_summary = {
        "experiment_id": config.experiment_id,
        "experiment_date": config.experiment_date,
        "input_dir": str(config.input_dir),
        "created_at": datetime.now(UTC).isoformat(),
        "sensor_files": [str(path.relative_to(config.input_dir)) for path in inputs.sensor_files],
        "image_file_count": len(inputs.image_files),
        "sensor_file_count": len(inputs.sensor_files),
        "prepared_rows": len(prepared),
        "cycle_count": len(cycle_summary),
    }
    return prepared, cycle_summary, prepare_summary


def _load_channel_frames(
    paths: tuple[Path, ...], input_dir: Path, channels: Mapping[str, Mapping[str, Any]]
) -> dict[str, list[pd.DataFrame]]:
    source_to_channels: dict[str, set[str]] = defaultdict(set)
    for name, settings in channels.items():
        if str(settings.get("kind")) == "derived":
            continue
        for source in settings.get("source_names", []):
            source_to_channels[str(source)].add(name)

    frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for path in paths:
        table = _read_sensor_table(path)
        if table.empty:
            continue
        group = _parameter_group(path)
        for raw_column in table.columns:
            if raw_column == "timestamp":
                continue
            canonical = _canonical_name(group, str(raw_column))
            channel_names = _matching_channels(
                source_to_channels, canonical, str(raw_column)
            )
            for name in channel_names:
                frames[name].append(
                    pd.DataFrame(
                        {"timestamp": table["timestamp"], "raw": table[raw_column]}
                    )
                )
    return frames


def _read_sensor_table(path: Path) -> pd.DataFrame:
    sample = path.read_bytes()[:131_072]
    if sample.startswith((b"\xd0\xcf\x11\xe0", b"PK\x03\x04")):
        raise ValueError(f"binary Excel workbook is not a supported text export: {path}")
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            decoded = sample.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeError(f"cannot decode {path}")
    delimiter = _detect_delimiter(decoded)
    table = pd.read_csv(
        path,
        sep=delimiter,
        encoding=encoding,
        dtype=str,
        keep_default_na=False,
        engine="python",
    )
    table.columns = [str(column).strip() for column in table.columns]
    time_column = _time_column(table.columns)
    if time_column is None:
        raise ValueError(f"no timestamp column in {path}")
    timestamps = pd.to_datetime(table.pop(time_column), errors="coerce")
    table.insert(0, "timestamp", timestamps)
    return table.loc[table["timestamp"].notna()].reset_index(drop=True)


def _matching_channels(
    source_to_channels: Mapping[str, set[str]], canonical: str, raw_column: str
) -> set[str]:
    direct = set(source_to_channels.get(canonical, set()))
    if direct:
        return direct
    direct = set(source_to_channels.get(raw_column, set()))
    if direct:
        return direct
    return {
        name
        for source, names in source_to_channels.items()
        if source.endswith(f"__{raw_column}")
        for name in names
    }


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
    rows: list[dict[str, object]] = []
    for timestamp, group in records.groupby("timestamp", sort=True):
        raw_values = group["raw"].astype("string").str.strip()
        cleaned, invalid = _parse_values(raw_values, kind, settings)
        valid_values = cleaned.loc[~invalid & cleaned.notna()]
        unique_values = {str(value) for value in valid_values.tolist()}
        duplicate = len(group) > 1
        conflict = len(unique_values) > 1
        value: object = np.nan
        if len(group) == 1 and not bool(invalid.iloc[0]) and cleaned.iloc[0] is not pd.NA:
            value = cleaned.iloc[0]
        rows.append(
            {
                "timestamp": timestamp,
                "value": value,
                "__missing": bool(raw_values.iloc[0] == ""),
                "__invalid": bool(invalid.any()),
                "__duplicate": duplicate,
                "__conflict": conflict,
            }
        )
    result = pd.DataFrame(rows).set_index("timestamp")
    result = result.reindex(pd.DatetimeIndex(timestamps))
    result["__missing"] = result["__missing"].fillna(True).astype(bool)
    for column in ("__invalid", "__duplicate", "__conflict"):
        result[column] = result[column].fillna(False).astype(bool)
    return result.reset_index(drop=True)


def _parse_values(
    raw_values: pd.Series, kind: str, settings: Mapping[str, Any]
) -> tuple[pd.Series, pd.Series]:
    nonempty = raw_values.ne("") & raw_values.notna()
    if kind == "event":
        allowed = settings.get("allowed_values", {})
        mapping = {str(key).strip().upper(): bool(value) for key, value in allowed.items()}
        values = raw_values.map(lambda value: mapping.get(str(value).upper(), np.nan))
        invalid = nonempty & values.isna()
        return values.astype("object"), invalid
    numeric = pd.to_numeric(raw_values.replace("", pd.NA), errors="coerce")
    invalid = nonempty & numeric.isna()
    return numeric.astype(float), invalid


def _all_timestamps(channel_frames: Mapping[str, list[pd.DataFrame]]) -> pd.Series:
    values = [frame["timestamp"] for frames in channel_frames.values() for frame in frames]
    if not values:
        return pd.Series(dtype="datetime64[ns]")
    unique = pd.concat(values, ignore_index=True).drop_duplicates().sort_values()
    return pd.Series(unique.to_numpy())


def _parameter_group(path: Path) -> str | None:
    match = re.search(r"参数(?P<group>\d+)", path.stem)
    return None if match is None else match.group("group")


def _canonical_name(group: str | None, raw_column: str) -> str:
    return f"p{group}__{raw_column}" if group else raw_column


def _time_column(columns: pd.Index[str]) -> str | None:
    for position, column in enumerate(columns):
        normalized = re.sub(r"[\s_-]+", "", str(column)).lower()
        if normalized in {"时间", "time", "timestamp", "datetime"} or position == 0:
            return str(column)
    return None


def _detect_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        counts = {delimiter: sample.count(delimiter) for delimiter in ("\t", ",", ";")}
        delimiter, count = max(counts.items(), key=lambda item: item[1])
        if count == 0:
            raise ValueError("no supported delimiter") from None
        return delimiter
