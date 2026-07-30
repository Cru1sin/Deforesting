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
    settings: Mapping[str, Any],
    *,
    experiment_id: str,
    experiment_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign one stable cycle vocabulary without filling long event gaps."""
    if not {"timestamp", defrost_column} <= set(frame.columns):
        raise ValueError(f"cycle labeling requires timestamp and {defrost_column}")
    labeled = frame.copy()
    labeled["timestamp"] = pd.to_datetime(labeled["timestamp"], errors="raise")
    labeled = labeled.sort_values("timestamp", kind="stable").reset_index(drop=True)
    raw_state = labeled[defrost_column].map(_normalize_state).astype("object")
    filled_state, long_gap = _fill_short_state_gaps(
        labeled["timestamp"], raw_state, float(settings.get("maximum_state_gap_seconds", 5))
    )
    runs = _true_runs(labeled["timestamp"], filled_state)
    cycles: list[dict[str, object]] = []
    complete_ranges: list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for index in range(len(runs) - 1):
        previous = runs[index]
        following = runs[index + 1]
        if previous[2] is None or following[2] is None:
            continue
        cycle_id = f"cycle_{len(complete_ranges) + 1:03d}"
        heating_start = previous[2]
        stable_start = heating_start + pd.Timedelta(
            seconds=float(settings.get("stable_heating_seconds", 0))
        )
        defrost_start = following[0]
        defrost_end = following[2]
        status = "valid"
        reason = ""
        if stable_start >= defrost_start:
            status = "invalid"
            reason = "invalid_cycle_boundaries"
        elif long_gap and _interval_contains(
            labeled["timestamp"], heating_start, defrost_end, long_gap
        ):
            status = "incomplete"
            reason = "defrost_state_gap"
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
        complete_ranges.append(
            (cycle_id, heating_start, stable_start, defrost_start, defrost_end)
        )

    if not cycles:
        reason = "defrost_state_gap" if long_gap else "insufficient_cycle_boundaries"
        cycles.append(
            _cycle_row(
                experiment_id,
                experiment_date,
                "cycle_001",
                "incomplete",
                reason,
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
    for cycle_id, heating_start, stable_start, defrost_start, defrost_end in complete_ranges:
        summary = next(row for row in cycles if row["cycle_id"] == cycle_id)
        mask = labeled["timestamp"].between(heating_start, defrost_end)
        labeled.loc[mask, "cycle_id"] = cycle_id
        labeled.loc[mask, "cycle_status"] = str(summary["cycle_status"])
        labeled.loc[mask, "cycle_status_reason"] = str(summary["cycle_status_reason"])
        labeled.loc[mask, "cycle_stage"] = _stage_for_times(
            labeled.loc[mask, "timestamp"], stable_start, defrost_start, defrost_end
        ).to_numpy()

    _label_unassigned_rows(labeled, cycles)
    _add_cycle_coordinates(labeled, cycles)
    labeled["cycle_stage"] = labeled["cycle_stage"].astype("string")
    if not labeled["cycle_stage"].dropna().isin(_STAGES).all():
        raise ValueError("cycle_stage contains an unknown value")
    return labeled, pd.DataFrame(cycles, columns=_cycle_columns())


def _normalize_state(value: object) -> bool | float:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
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
) -> tuple[pd.Series, bool]:
    result = state.copy()
    long_gap = False
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
        can_fill = previous >= 0 and following < len(result)
        if can_fill:
            elapsed = (timestamps.iloc[following] - timestamps.iloc[previous]).total_seconds()
            same_state = result.iloc[previous] == result.iloc[following]
            if elapsed <= maximum_seconds and same_state:
                result.iloc[position : end + 1] = result.iloc[previous]
            else:
                long_gap = True
        position = end + 1
    return result, long_gap


def _true_runs(
    timestamps: pd.Series, state: pd.Series
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None]]:
    runs: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None]] = []
    is_true = state.eq(True)
    groups = is_true.ne(is_true.shift(fill_value=False)).cumsum()
    for _, part in is_true.groupby(groups, sort=False):
        if not bool(part.iloc[0]):
            continue
        start_index = int(part.index[0])
        end_index = int(part.index[-1])
        next_time = timestamps.iloc[end_index + 1] if end_index + 1 < len(timestamps) else None
        runs.append((timestamps.iloc[start_index], timestamps.iloc[end_index], next_time))
    return runs


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
    }


def _stage_for_times(
    times: pd.Series,
    stable_start: pd.Timestamp,
    defrost_start: pd.Timestamp,
    defrost_end: pd.Timestamp,
) -> pd.Series:
    stage = pd.Series("partial", index=times.index, dtype="string")
    stage.loc[times.lt(stable_start)] = "recovery"
    stage.loc[times.ge(stable_start) & times.lt(defrost_start)] = "frost_development"
    stage.loc[times.ge(defrost_start) & times.le(defrost_end)] = "defrost"
    return stage


def _label_unassigned_rows(labeled: pd.DataFrame, cycles: list[dict[str, object]]) -> None:
    unassigned = labeled["cycle_id"].isna()
    if not unassigned.any():
        return
    partial_id = "partial_001"
    labeled.loc[unassigned, "cycle_id"] = partial_id
    labeled.loc[unassigned, "cycle_stage"] = "partial"
    labeled.loc[unassigned, "cycle_status"] = "incomplete"
    labeled.loc[unassigned, "cycle_status_reason"] = "outside_complete_cycle"
    if not any(row["cycle_id"] == partial_id for row in cycles):
        cycles.append(
            _cycle_row(
                str(labeled["experiment_id"].iloc[0])
                if "experiment_id" in labeled
                else "",
                str(labeled["experiment_date"].iloc[0])
                if "experiment_date" in labeled
                else "",
                partial_id,
                "incomplete",
                "outside_complete_cycle",
                None,
                None,
                None,
                None,
            )
        )


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
        labeled.loc[mask, "cycle_elapsed_seconds"] = elapsed
        labeled.loc[mask, "cycle_progress"] = (elapsed / duration).clip(0, 1)


def _interval_contains(
    timestamps: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    gap: bool,
) -> bool:
    return bool(gap and timestamps.between(start, end).any())


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
    ]
