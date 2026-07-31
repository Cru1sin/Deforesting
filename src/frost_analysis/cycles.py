"""Transparent cycle labeling for the Prepared stage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

_STAGES = {"recovery", "frost_development", "defrost", "partial"}
_STATUSES = {"valid", "incomplete", "invalid"}


def label_cycles(
    frame: pd.DataFrame,
    defrost_column: str,
    settings: Mapping[str, Any] | Any,
    *,
    experiment_id: str,
    experiment_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign cycle boundaries without interpreting unknown as OFF."""
    if not {"timestamp", defrost_column} <= set(frame.columns):
        raise ValueError(f"cycle labeling requires timestamp and {defrost_column}")
    labeled = frame.copy()
    labeled["timestamp"] = pd.to_datetime(labeled["timestamp"], errors="raise")
    labeled = labeled.sort_values("timestamp", kind="stable").reset_index(drop=True)
    raw_state = labeled[defrost_column].map(_normalize_state).astype("object")
    filled_state, long_gaps = _fill_short_state_gaps(
        labeled["timestamp"], raw_state, _setting(settings, "maximum_state_gap_seconds", 5)
    )
    debounced = _debounce_state(
        labeled["timestamp"], filled_state, _setting(settings, "debounce_seconds", 20)
    )
    events = _defrost_runs(labeled["timestamp"], debounced)
    cycles: list[dict[str, object]] = []
    cycle_ranges: list[tuple[dict[str, object], pd.Timestamp, pd.Timestamp | None]] = []

    for index in range(len(events) - 1):
        previous = events[index]
        following = events[index + 1]
        heating_start = previous["end"]
        defrost_start = following["start"]
        defrost_end = following["end"]
        cycle_id = f"cycle_{index + 1:03d}"
        status = "valid"
        reason = ""
        if heating_start is None:
            status, reason = "incomplete", "defrost_state_gap"
        elif defrost_end is None:
            status, reason = "incomplete", "data_ends_mid_cycle"
        elif _interval_intersects(heating_start, defrost_end, long_gaps):
            status, reason = "incomplete", "defrost_state_gap"
        elif not _duration_in_range(
            previous["duration"],
            _setting(settings, "minimum_defrost_seconds", 60),
            _setting(settings, "maximum_defrost_seconds", 1200),
        ) or not _duration_in_range(
            following["duration"],
            _setting(settings, "minimum_defrost_seconds", 60),
            _setting(settings, "maximum_defrost_seconds", 1200),
        ):
            status, reason = "invalid", "defrost_duration_out_of_range"
        else:
            heating_duration = (defrost_start - heating_start).total_seconds()
            if not _duration_in_range(
                heating_duration,
                _setting(settings, "minimum_heating_seconds", 1800),
                _setting(settings, "maximum_heating_seconds", 21600),
            ):
                status, reason = "invalid", "heating_duration_out_of_range"

        stable_start = (
            heating_start
            + pd.Timedelta(seconds=_setting(settings, "stable_heating_seconds", 180))
            if heating_start is not None
            else None
        )
        if stable_start is not None and stable_start >= defrost_start:
            status, reason = "invalid", "invalid_cycle_boundaries"
        row = _cycle_row(
            experiment_id,
            experiment_date,
            cycle_id,
            status,
            reason,
            heating_start,
            stable_start,
            defrost_start,
            defrost_end,
        )
        cycles.append(row)
        if heating_start is not None:
            cycle_ranges.append((row, heating_start, defrost_end))

    if not cycles:
        cycles.append(
            _cycle_row(
                experiment_id,
                experiment_date,
                "cycle_001",
                "incomplete",
                "defrost_state_gap" if long_gaps else "insufficient_cycle_boundaries",
                None,
                None,
                None,
                None,
            )
        )

    labeled["cycle_id"] = pd.Series(pd.NA, index=labeled.index, dtype="string")
    labeled["cycle_stage"] = pd.Series(pd.NA, index=labeled.index, dtype="string")
    labeled["cycle_status"] = pd.Series(pd.NA, index=labeled.index, dtype="string")
    labeled["cycle_status_reason"] = pd.Series(pd.NA, index=labeled.index, dtype="string")
    for row, cycle_start, cycle_end in cycle_ranges:
        cycle_id = str(row["cycle_id"])
        if cycle_end is None:
            mask = labeled["timestamp"].ge(cycle_start)
        else:
            mask = labeled["timestamp"].ge(cycle_start) & labeled["timestamp"].lt(cycle_end)
        labeled.loc[mask, "cycle_id"] = cycle_id
        labeled.loc[mask, "cycle_status"] = str(row["cycle_status"])
        labeled.loc[mask, "cycle_status_reason"] = str(row["cycle_status_reason"])
        labeled.loc[mask, "cycle_stage"] = _stage_for_times(
            labeled.loc[mask, "timestamp"],
            row["stable_heating_start"],
            row["defrost_start"],
            row["defrost_end"],
        ).to_numpy()

    _label_unassigned_rows(labeled, cycles, experiment_id, experiment_date)
    _add_cycle_coordinates(labeled, cycles)
    labeled["cycle_stage"] = labeled["cycle_stage"].astype("string")
    if not labeled["cycle_stage"].dropna().isin(_STAGES).all():
        raise ValueError("cycle_stage contains an unknown value")
    return labeled, pd.DataFrame(cycles, columns=_cycle_columns())


def _normalize_state(value: object) -> bool | float:
    if value is None or value is pd.NA or pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().upper()
    if text in {"ON", "1", "TRUE", "YES"}:
        return True
    if text in {"OFF", "0", "FALSE", "NO"}:
        return False
    return np.nan


def _fill_short_state_gaps(
    timestamps: pd.Series, state: pd.Series, maximum_seconds: float
) -> tuple[pd.Series, tuple[tuple[pd.Timestamp, pd.Timestamp], ...]]:
    result = state.copy()
    long_gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    missing = result.isna().to_numpy()
    position = 0
    while position < len(result):
        if not missing[position]:
            position += 1
            continue
        end = position
        while end + 1 < len(result) and missing[end + 1]:
            end += 1
        previous = position - 1
        following = end + 1
        if previous >= 0 and following < len(result):
            elapsed = (timestamps.iloc[following] - timestamps.iloc[previous]).total_seconds()
            same_state = result.iloc[previous] == result.iloc[following]
            if elapsed <= maximum_seconds and same_state:
                result.iloc[position : end + 1] = result.iloc[previous]
            else:
                long_gaps.append((timestamps.iloc[previous], timestamps.iloc[following]))
        position = end + 1
    return result, tuple(long_gaps)


def _debounce_state(timestamps: pd.Series, state: pd.Series, debounce_seconds: float) -> pd.Series:
    result = state.copy()
    for _ in range(len(result)):
        changed = False
        position = 0
        while position < len(result):
            value = result.iloc[position]
            end = position
            while end + 1 < len(result) and result.iloc[end + 1] == value:
                end += 1
            previous = position - 1
            following = end + 1
            bounded = (
                value is not np.nan
                and not pd.isna(value)
                and previous >= 0
                and following < len(result)
                and not pd.isna(result.iloc[previous])
                and result.iloc[previous] == result.iloc[following]
            )
            if bounded:
                duration = (timestamps.iloc[following] - timestamps.iloc[position]).total_seconds()
                if duration <= debounce_seconds:
                    result.iloc[position : end + 1] = result.iloc[previous]
                    changed = True
            position = end + 1
        if not changed:
            break
    return result


def _defrost_runs(
    timestamps: pd.Series, state: pd.Series
) -> list[dict[str, pd.Timestamp | float | None]]:
    events: list[dict[str, pd.Timestamp | float | None]] = []
    position = 0
    while position < len(state):
        if state.iloc[position] is not True:
            position += 1
            continue
        start = position
        while position + 1 < len(state) and state.iloc[position + 1] is True:
            position += 1
        next_index = position + 1
        end = (
            timestamps.iloc[next_index]
            if next_index < len(state) and state.iloc[next_index] is False
            else None
        )
        duration = (
            (end - timestamps.iloc[start]).total_seconds() if end is not None else None
        )
        events.append({"start": timestamps.iloc[start], "end": end, "duration": duration})
        position += 1
    return events


def _cycle_row(
    experiment_id: str,
    experiment_date: str,
    cycle_id: str,
    status: str,
    reason: str,
    heating_start: pd.Timestamp | None,
    stable_start: pd.Timestamp | None,
    defrost_start: pd.Timestamp | None,
    defrost_end: pd.Timestamp | None,
) -> dict[str, object]:
    heating_duration = (
        (defrost_start - heating_start).total_seconds()
        if heating_start is not None and defrost_start is not None
        else np.nan
    )
    defrost_duration = (
        (defrost_end - defrost_start).total_seconds()
        if defrost_start is not None and defrost_end is not None
        else np.nan
    )
    return {
        "experiment_id": experiment_id,
        "experiment_date": experiment_date,
        "cycle_id": cycle_id,
        "cycle_status": status,
        "cycle_status_reason": reason,
        "heating_start": heating_start,
        "stable_heating_start": stable_start,
        "defrost_start": defrost_start,
        "defrost_end": defrost_end,
        "heating_duration_seconds": heating_duration,
        "defrost_duration_seconds": defrost_duration,
    }


def _stage_for_times(
    times: pd.Series,
    stable_start: object,
    defrost_start: object,
    defrost_end: object,
) -> pd.Series:
    stage = pd.Series("partial", index=times.index, dtype="string")
    if not isinstance(stable_start, pd.Timestamp) or not isinstance(defrost_start, pd.Timestamp):
        return stage
    stage.loc[times.lt(stable_start)] = "recovery"
    stage.loc[times.ge(stable_start) & times.lt(defrost_start)] = "frost_development"
    if isinstance(defrost_end, pd.Timestamp):
        stage.loc[times.ge(defrost_start) & times.lt(defrost_end)] = "defrost"
    else:
        stage.loc[times.ge(defrost_start)] = "defrost"
    return stage


def _label_unassigned_rows(
    labeled: pd.DataFrame,
    cycles: list[dict[str, object]],
    experiment_id: str,
    experiment_date: str,
) -> None:
    unassigned = labeled["cycle_id"].isna().to_numpy()
    position = 0
    partial_index = 1
    while position < len(unassigned):
        if not unassigned[position]:
            position += 1
            continue
        end = position
        while end + 1 < len(unassigned) and unassigned[end + 1]:
            end += 1
        partial_id = f"partial_{partial_index:03d}"
        index = labeled.index[position : end + 1]
        labeled.loc[index, "cycle_id"] = partial_id
        labeled.loc[index, "cycle_stage"] = "partial"
        labeled.loc[index, "cycle_status"] = "incomplete"
        labeled.loc[index, "cycle_status_reason"] = "outside_complete_cycle"
        cycles.append(
            _cycle_row(
                experiment_id,
                experiment_date,
                partial_id,
                "incomplete",
                "outside_complete_cycle",
                None,
                None,
                None,
                None,
            )
        )
        partial_index += 1
        position = end + 1


def _add_cycle_coordinates(labeled: pd.DataFrame, cycles: list[dict[str, object]]) -> None:
    labeled["cycle_elapsed_seconds"] = np.nan
    labeled["cycle_progress"] = np.nan
    for row in cycles:
        stable_start = row["stable_heating_start"]
        defrost_start = row["defrost_start"]
        if not isinstance(stable_start, pd.Timestamp) or not isinstance(
            defrost_start, pd.Timestamp
        ):
            continue
        cycle_id = str(row["cycle_id"])
        mask = labeled["cycle_id"].eq(cycle_id) & labeled["cycle_stage"].eq(
            "frost_development"
        )
        elapsed = (labeled.loc[mask, "timestamp"] - stable_start).dt.total_seconds()
        duration = (defrost_start - stable_start).total_seconds()
        if duration <= 0:
            continue
        labeled.loc[mask, "cycle_elapsed_seconds"] = elapsed
        labeled.loc[mask, "cycle_progress"] = (elapsed / duration).clip(0, 1)


def _interval_intersects(
    start: pd.Timestamp,
    end: pd.Timestamp,
    gaps: tuple[tuple[pd.Timestamp, pd.Timestamp], ...],
) -> bool:
    return any(gap_start <= end and gap_end >= start for gap_start, gap_end in gaps)


def _duration_in_range(value: object, minimum: float, maximum: float) -> bool:
    return value is not None and minimum <= float(value) <= maximum


def _setting(settings: Mapping[str, Any] | Any, name: str, default: float) -> float:
    if isinstance(settings, Mapping):
        return float(settings.get(name, default))
    return float(getattr(settings, name, default))


def _cycle_columns() -> list[str]:
    return [
        "experiment_id",
        "experiment_date",
        "cycle_id",
        "cycle_status",
        "cycle_status_reason",
        "heating_start",
        "stable_heating_start",
        "defrost_start",
        "defrost_end",
        "heating_duration_seconds",
        "defrost_duration_seconds",
    ]
