"""Thin adapters for non-standard raw sensor formats."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_TEMPERATURE_PATTERN = re.compile(r"^T_SHT40_(?P<serial>.+)$")
_HUMIDITY_PATTERN = re.compile(r"^RH_SHT40_(?P<serial>.+)$")
_SIGNAL_COLUMNS = (
    "sensor_1_temperature",
    "sensor_1_humidity",
    "sensor_2_temperature",
    "sensor_2_humidity",
)


def read_edf_environment(
    edf_paths: Sequence[Path],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    pair_tolerance: pd.Timedelta,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read, pair, average, and time-clip two SHT40 streams from EDF files."""
    if pd.isna(pair_tolerance) or pair_tolerance <= pd.Timedelta(0):
        raise ValueError("pair_tolerance must be positive")
    if start_time > end_time:
        raise ValueError("start_time must not be after end_time")
    if not edf_paths:
        return _empty_environment_frame(), _empty_edf_summary()

    tables: list[pd.DataFrame] = []
    raw_rows = 0
    serials: tuple[str, str] | None = None
    for path in edf_paths:
        table, file_serials, file_raw_rows = _read_edf_file(path)
        if serials is None:
            serials = file_serials
        elif serials != file_serials:
            raise ValueError(f"EDF sensor serials do not match across files: {path}")
        tables.append(table)
        raw_rows += file_raw_rows

    combined = pd.concat(tables, ignore_index=True)
    deduplication_columns = ["timestamp", *_SIGNAL_COLUMNS]
    before_deduplication = len(combined)
    combined = combined.drop_duplicates(subset=deduplication_columns, ignore_index=True)
    duplicate_rows_removed = before_deduplication - len(combined)

    sensor_1, sensor_2 = _split_sensor_records(combined)
    paired, unmatched_sensor_1, unmatched_sensor_2, pair_deltas_ms = _pair_sensor_records(
        sensor_1,
        sensor_2,
        pair_tolerance,
    )
    paired_rows = len(paired)
    clipped = paired.loc[
        paired["timestamp"].between(start_time, end_time, inclusive="both")
    ].reset_index(drop=True)

    summary: dict[str, object] = {
        "raw_rows": raw_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
        "sensor_1_rows": len(sensor_1),
        "sensor_2_rows": len(sensor_2),
        "paired_rows": paired_rows,
        "unmatched_sensor_1_rows": unmatched_sensor_1,
        "unmatched_sensor_2_rows": unmatched_sensor_2,
        "pair_delta_median_ms": (
            float(np.median(pair_deltas_ms)) if pair_deltas_ms else None
        ),
        "pair_delta_max_ms": float(max(pair_deltas_ms)) if pair_deltas_ms else None,
        "rows_after_time_clip": len(clipped),
    }
    return clipped, summary


def _read_edf_file(path: Path) -> tuple[pd.DataFrame, tuple[str, str], int]:
    header_row: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle):
            if line.startswith("Epoch_UTC\t"):
                header_row = row_number
                break
    if header_row is None:
        raise ValueError(f"EDF data header not found: {path}")

    table = pd.read_csv(
        path,
        sep="\t",
        skiprows=header_row,
        dtype=str,
        keep_default_na=False,
        engine="python",
    )
    table.columns = [str(column).strip() for column in table.columns]
    required = {"Epoch_UTC", "Local_Date_Time"}
    if not required.issubset(table.columns):
        raise ValueError(f"EDF header is missing required columns: {path}")

    temperatures = {
        match.group("serial"): column
        for column in table.columns
        if (match := _TEMPERATURE_PATTERN.match(str(column))) is not None
    }
    humidities = {
        match.group("serial"): column
        for column in table.columns
        if (match := _HUMIDITY_PATTERN.match(str(column))) is not None
    }
    serials = tuple(sorted(set(temperatures) & set(humidities)))
    if len(serials) != 2:
        raise ValueError(f"EDF must contain exactly two complete SHT40 sensors: {path}")

    normalized = pd.DataFrame(
        {
            "timestamp": _local_wall_time(table["Local_Date_Time"]),
            "sensor_1_temperature": pd.to_numeric(
                table[temperatures[serials[0]]], errors="coerce"
            ),
            "sensor_1_humidity": pd.to_numeric(
                table[humidities[serials[0]]], errors="coerce"
            ),
            "sensor_2_temperature": pd.to_numeric(
                table[temperatures[serials[1]]], errors="coerce"
            ),
            "sensor_2_humidity": pd.to_numeric(
                table[humidities[serials[1]]], errors="coerce"
            ),
        }
    )
    return normalized, serials, len(table)


def _local_wall_time(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    try:
        return parsed.dt.tz_localize(None)
    except (AttributeError, TypeError):
        return parsed


def _split_sensor_records(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def complete_records(temperature: str, humidity: str) -> pd.DataFrame:
        values = table[["timestamp", temperature, humidity]].copy()
        values[temperature] = pd.to_numeric(values[temperature], errors="coerce")
        values[humidity] = pd.to_numeric(values[humidity], errors="coerce")
        valid = (
            values["timestamp"].notna()
            & values[temperature].notna()
            & values[humidity].notna()
            & np.isfinite(values[temperature])
            & np.isfinite(values[humidity])
        )
        return values.loc[valid, ["timestamp", temperature, humidity]].rename(
            columns={temperature: "temperature", humidity: "humidity"}
        ).sort_values("timestamp", kind="stable").reset_index(drop=True)

    return (
        complete_records("sensor_1_temperature", "sensor_1_humidity"),
        complete_records("sensor_2_temperature", "sensor_2_humidity"),
    )


def _pair_sensor_records(
    sensor_1: pd.DataFrame,
    sensor_2: pd.DataFrame,
    tolerance: pd.Timedelta,
) -> tuple[pd.DataFrame, int, int, list[float]]:
    rows: list[dict[str, Any]] = []
    unmatched_sensor_1 = 0
    unmatched_sensor_2 = 0
    pair_deltas_ms: list[float] = []
    left = 0
    right = 0
    while left < len(sensor_1) and right < len(sensor_2):
        left_time = pd.Timestamp(sensor_1.iloc[left]["timestamp"])
        right_time = pd.Timestamp(sensor_2.iloc[right]["timestamp"])
        delta_seconds = (left_time - right_time).total_seconds()
        if abs(delta_seconds) <= tolerance.total_seconds():
            rows.append(
                {
                    "timestamp": left_time + (right_time - left_time) / 2,
                    "environment_temperature": (
                        float(sensor_1.iloc[left]["temperature"])
                        + float(sensor_2.iloc[right]["temperature"])
                    )
                    / 2,
                    "environment_relative_humidity": (
                        float(sensor_1.iloc[left]["humidity"])
                        + float(sensor_2.iloc[right]["humidity"])
                    )
                    / 2,
                }
            )
            pair_deltas_ms.append(abs(delta_seconds) * 1000)
            left += 1
            right += 1
        elif left_time < right_time:
            unmatched_sensor_1 += 1
            left += 1
        else:
            unmatched_sensor_2 += 1
            right += 1

    unmatched_sensor_1 += len(sensor_1) - left
    unmatched_sensor_2 += len(sensor_2) - right
    paired = pd.DataFrame(rows, columns=["timestamp", *_ENVIRONMENT_COLUMNS])
    return paired, unmatched_sensor_1, unmatched_sensor_2, pair_deltas_ms


_ENVIRONMENT_COLUMNS = ["environment_temperature", "environment_relative_humidity"]


def _empty_environment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "environment_temperature": pd.Series(dtype="float64"),
            "environment_relative_humidity": pd.Series(dtype="float64"),
        }
    )


def _empty_edf_summary() -> dict[str, object]:
    return {
        "raw_rows": 0,
        "duplicate_rows_removed": 0,
        "sensor_1_rows": 0,
        "sensor_2_rows": 0,
        "paired_rows": 0,
        "unmatched_sensor_1_rows": 0,
        "unmatched_sensor_2_rows": 0,
        "pair_delta_median_ms": None,
        "pair_delta_max_ms": None,
        "rows_after_time_clip": 0,
    }
