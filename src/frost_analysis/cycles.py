"""Transparent cycle labeling for the Prepared stage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

_STAGES = {"recovery", "frost_development", "defrost", "partial"}
_STATUSES = {"valid", "incomplete", "invalid"}
_RECOVERY_THRESHOLD_OFFSETS = (2.0, 3.0, 4.0)


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
        labeled["timestamp"], raw_state, _setting(settings, "maximum_state_gap_seconds", 0)
    )
    debounced = _debounce_state(
        labeled["timestamp"], filled_state, _setting(settings, "debounce_seconds", 20)
    )
    events = _defrost_runs(labeled["timestamp"], debounced, long_gaps)
    cycles, cycle_ranges = _build_cycles(
        events,
        long_gaps,
        settings,
        labeled,
        experiment_id=experiment_id,
        experiment_date=experiment_date,
    )
    labeled = _assign_cycle_ranges(labeled, cycle_ranges)

    _label_unassigned_rows(
        labeled,
        cycles,
        experiment_id,
        experiment_date,
        defrost_column=defrost_column,
        settings=settings,
    )
    _add_cycle_coordinates(labeled, cycles)
    labeled["cycle_stage"] = labeled["cycle_stage"].astype("string")
    if not labeled["cycle_stage"].dropna().isin(_STAGES).all():
        raise ValueError("cycle_stage contains an unknown value")
    return labeled, pd.DataFrame(cycles, columns=_cycle_columns())


def _build_cycles(
    events: list[dict[str, Any]],
    long_gaps: tuple[tuple[pd.Timestamp, pd.Timestamp], ...],
    settings: Mapping[str, Any] | Any,
    labeled: pd.DataFrame,
    *,
    experiment_id: str,
    experiment_date: str,
) -> tuple[
    list[dict[str, object]],
    list[tuple[dict[str, object], pd.Timestamp, pd.Timestamp | None]],
]:
    cycles: list[dict[str, object]] = []
    ranges: list[tuple[dict[str, object], pd.Timestamp, pd.Timestamp | None]] = []
    cycle_number = 1
    for index in range(len(events) - 1):
        heating_start = events[index]["end"]
        if not isinstance(heating_start, pd.Timestamp):
            continue
        row = _make_cycle_record(
            events[index],
            events[index + 1],
            long_gaps,
            settings,
            labeled,
            experiment_id,
            experiment_date,
            f"cycle_{cycle_number:03d}",
        )
        cycles.append(row)
        cycle_number += 1
        defrost_end = row["defrost_end"]
        if isinstance(defrost_end, pd.Timestamp) or defrost_end is None:
            ranges.append((row, heating_start, defrost_end))
    return cycles, ranges


def _make_cycle_record(
    previous: dict[str, Any],
    following: dict[str, Any],
    long_gaps: tuple[tuple[pd.Timestamp, pd.Timestamp], ...],
    settings: Mapping[str, Any] | Any,
    labeled: pd.DataFrame,
    experiment_id: str,
    experiment_date: str,
    cycle_id: str,
) -> dict[str, object]:
    heating_start = previous["end"]
    defrost_start = following["start"]
    defrost_end = following["end"]
    status, reason = _cycle_status(
        previous,
        following,
        heating_start,
        defrost_start,
        defrost_end,
        long_gaps,
        settings,
    )
    stable_start = _stable_start(labeled, heating_start, defrost_start, settings)
    if stable_start is None and status == "valid":
        status, reason = "incomplete", "recovery_end_not_observed"
    if (
        stable_start is not None
        and isinstance(defrost_start, pd.Timestamp)
        and stable_start >= defrost_start
    ):
        status, reason = "invalid", "invalid_cycle_boundaries"
    if status == "valid":
        status, reason = _operating_mode_status(
            labeled,
            heating_start,
            defrost_start,
            settings,
        )
    return _cycle_row(
        experiment_id,
        experiment_date,
        cycle_id,
        status,
        reason,
        heating_start,
        stable_start,
        defrost_start,
        defrost_end,
        previous["duration"],
        following["duration"],
    )


def _cycle_status(
    previous: dict[str, pd.Timestamp | float | None],
    following: dict[str, pd.Timestamp | float | None],
    heating_start: Any,
    defrost_start: Any,
    defrost_end: Any,
    long_gaps: tuple[tuple[pd.Timestamp, pd.Timestamp], ...],
    settings: Mapping[str, Any] | Any,
) -> tuple[str, str]:
    if heating_start is None or defrost_start is None:
        return "incomplete", "defrost_state_gap"
    if defrost_end is None:
        return "incomplete", "defrost_end_not_observed"
    if defrost_start <= heating_start or defrost_end <= defrost_start:
        return "invalid", "invalid_cycle_boundaries"
    if following.get("boundary_uncertain"):
        return "incomplete", "defrost_state_gap"
    if _interval_intersects(heating_start, defrost_end, long_gaps):
        return "incomplete", "defrost_state_gap"
    if not _duration_in_range(
        previous["duration"],
        _setting(settings, "minimum_defrost_seconds", 60),
        _setting(settings, "maximum_defrost_seconds", 1200),
    ):
        return "invalid", "preceding_defrost_duration_out_of_range"
    if not _duration_in_range(
        following["duration"],
        _setting(settings, "minimum_defrost_seconds", 60),
        _setting(settings, "maximum_defrost_seconds", 1200),
    ):
        return "invalid", "terminal_defrost_duration_out_of_range"
    heating_duration = (defrost_start - heating_start).total_seconds()
    if not _duration_in_range(
        heating_duration,
        _setting(settings, "minimum_heating_seconds", 1800),
        _setting(settings, "maximum_heating_seconds", 21600),
    ):
        return "invalid", "heating_duration_out_of_range"
    return "valid", ""


def _stable_start(
    frame: pd.DataFrame,
    heating_start: Any,
    defrost_start: Any,
    settings: Mapping[str, Any] | Any,
) -> pd.Timestamp | None:
    if not isinstance(heating_start, pd.Timestamp):
        return None
    required = {"timestamp", "water_out_temperature", "water_temperature_setpoint"}
    if not required <= set(frame.columns):
        return heating_start + pd.Timedelta(
            seconds=_setting(settings, "stable_heating_seconds", 180)
        )
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    water_out = pd.to_numeric(frame["water_out_temperature"], errors="coerce")
    setpoint = pd.to_numeric(
        frame["water_temperature_setpoint"], errors="coerce"
    )
    mask = timestamps.ge(heating_start) & timestamps.notna() & water_out.notna() & setpoint.notna()
    if isinstance(defrost_start, pd.Timestamp):
        mask &= timestamps.lt(defrost_start)
    if not mask.any():
        return None
    observations = pd.DataFrame(
        {
            "timestamp": timestamps.loc[mask],
            "water_out": water_out.loc[mask],
            "setpoint": setpoint.loc[mask],
        }
    ).sort_values("timestamp", kind="stable")
    for offset in _RECOVERY_THRESHOLD_OFFSETS:
        crossing = observations.loc[
            observations["water_out"].ge(observations["setpoint"] - offset),
            "timestamp",
        ]
        if not crossing.empty:
            return pd.Timestamp(crossing.iloc[0])
    return None


def _assign_cycle_ranges(
    labeled: pd.DataFrame,
    cycle_ranges: list[tuple[dict[str, object], pd.Timestamp, pd.Timestamp | None]],
) -> pd.DataFrame:
    result = labeled.copy()
    for column in ("cycle_id", "cycle_stage", "cycle_status", "cycle_status_reason"):
        result[column] = pd.Series(pd.NA, index=result.index, dtype="string")
    for row, cycle_start, cycle_end in cycle_ranges:
        if cycle_end is None:
            mask = result["timestamp"].ge(cycle_start)
        else:
            mask = result["timestamp"].ge(cycle_start) & result["timestamp"].lt(cycle_end)
        result.loc[mask, "cycle_id"] = str(row["cycle_id"])
        result.loc[mask, "cycle_status"] = str(row["cycle_status"])
        result.loc[mask, "cycle_status_reason"] = str(row["cycle_status_reason"])
        result.loc[mask, "cycle_stage"] = _stage_for_times(
            result.loc[mask, "timestamp"],
            row["stable_heating_start"],
            row["defrost_start"],
            row["defrost_end"],
        ).to_numpy()
    return result


def _normalize_state(value: Any) -> bool | float:
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
            elif elapsed > maximum_seconds:
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
    timestamps: pd.Series,
    state: pd.Series,
    long_gaps: tuple[tuple[pd.Timestamp, pd.Timestamp], ...],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    active_event: dict[str, Any] | None = None
    last_known_state: bool | None = None
    unknown_since_known = False
    long_gap_ends = {gap_end for _gap_start, gap_end in long_gaps}

    for position, value in enumerate(state):
        if pd.isna(value):
            unknown_since_known = True
            continue

        crossed_long_gap = unknown_since_known and timestamps.iloc[position] in long_gap_ends
        if crossed_long_gap:
            if active_event is not None:
                active_event["boundary_uncertain"] = True
                events.append(active_event)
                active_event = None
            last_known_state = None
        unknown_since_known = False

        if value is True:
            if active_event is None and last_known_state is not True:
                active_event = {
                    "start": timestamps.iloc[position],
                    "end": None,
                    "duration": None,
                }
                if crossed_long_gap:
                    active_event["boundary_uncertain"] = True
        elif value is False and active_event is not None:
            end = timestamps.iloc[position]
            active_event["end"] = end
            active_event["duration"] = (
                end - active_event["start"]
            ).total_seconds()
            events.append(active_event)
            active_event = None

        last_known_state = bool(value)

    if active_event is not None:
        events.append(active_event)
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
    preceding_defrost_duration: float | None = None,
    terminal_defrost_duration: float | None = None,
) -> dict[str, object]:
    heating_duration = (
        (defrost_start - heating_start).total_seconds()
        if heating_start is not None and defrost_start is not None
        else np.nan
    )
    return {
        "experiment_id": experiment_id,
        "experiment_date": experiment_date,
        "cycle_id": cycle_id,
        "segment_start": heating_start,
        "cycle_status": status,
        "cycle_status_reason": reason,
        "heating_start": heating_start,
        "stable_heating_start": stable_start,
        "defrost_start": defrost_start,
        "defrost_end": defrost_end,
        "heating_duration_seconds": heating_duration,
        "preceding_defrost_duration_seconds": preceding_defrost_duration,
        "terminal_defrost_duration_seconds": terminal_defrost_duration,
    }


def _operating_mode_status(
    frame: pd.DataFrame,
    heating_start: Any,
    defrost_start: Any,
    settings: Mapping[str, Any] | Any,
) -> tuple[str, str]:
    channel = _string_setting(settings, "operating_mode_channel", "")
    if not channel:
        return "valid", ""
    required = _string_setting(settings, "required_operating_mode", "3")
    if channel not in frame:
        return "incomplete", "missing_operating_mode"
    interval = frame.loc[
        frame["timestamp"].ge(heating_start) & frame["timestamp"].lt(defrost_start)
    ]
    observed = interval.loc[interval[channel].notna(), channel]
    for suffix in ("__duplicate", "__conflict"):
        quality_column = f"{channel}{suffix}"
        if quality_column in interval:
            quality = interval.loc[observed.index, quality_column].fillna(False).astype(bool)
            observed = observed.loc[~quality]
    if observed.empty:
        return "incomplete", "missing_operating_mode"
    if not observed.eq(required).all():
        return "invalid", "non_heating_mode_present"
    return "valid", ""


def _string_setting(settings: Mapping[str, Any] | Any, name: str, default: str) -> str:
    if isinstance(settings, Mapping):
        return str(settings.get(name, default))
    return str(getattr(settings, name, default))


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
    *,
    defrost_column: str,
    settings: Mapping[str, Any] | Any,
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
        segment = labeled.loc[index].copy()
        (
            segment_stages,
            heating_start,
            stable_start,
            defrost_start,
            defrost_end,
        ) = _partial_stage_context(segment, defrost_column, settings)
        labeled.loc[index, "cycle_id"] = partial_id
        labeled.loc[index, "cycle_stage"] = segment_stages.to_numpy()
        labeled.loc[index, "cycle_status"] = "incomplete"
        labeled.loc[index, "cycle_status_reason"] = "outside_complete_cycle"
        partial = _cycle_row(
            experiment_id,
            experiment_date,
            partial_id,
            "incomplete",
            "outside_complete_cycle",
            heating_start,
            stable_start,
            defrost_start,
            defrost_end,
        )
        partial["segment_start"] = heating_start
        cycles.append(partial)
        partial_index += 1
        position = end + 1


def _partial_stage_context(  # noqa: C901
    segment: pd.DataFrame,
    defrost_column: str,
    settings: Mapping[str, Any] | Any,
) -> tuple[
    pd.Series,
    pd.Timestamp | None,
    pd.Timestamp | None,
    pd.Timestamp | None,
    pd.Timestamp | None,
]:
    """Infer only the stage boundaries supported by an open segment.

    An open segment has no trusted cycle boundary on one side, but its observed
    water temperature and defrost state can still identify useful phases.  The
    status remains ``incomplete``; these labels are only the best supported
    stage facts for Process and publication.
    """
    times = pd.to_datetime(segment["timestamp"], errors="coerce")
    valid_times = times.dropna().sort_values(kind="stable")
    if valid_times.empty:
        return (
            pd.Series("partial", index=segment.index, dtype="string"),
            None,
            None,
            None,
            None,
        )

    heating_start = pd.Timestamp(valid_times.iloc[0])
    states = (
        segment[defrost_column].map(_normalize_state)
        if defrost_column in segment
        else pd.Series(np.nan, index=segment.index, dtype="float64")
    )
    active_times = times.loc[states.eq(True)].dropna().sort_values(kind="stable")
    defrost_start = (
        pd.Timestamp(active_times.iloc[0]) if not active_times.empty else None
    )
    defrost_end: pd.Timestamp | None = None
    if defrost_start is not None:
        inactive_after_start = times.loc[
            states.eq(False) & times.gt(defrost_start)
        ].dropna().sort_values(kind="stable")
        if not inactive_after_start.empty:
            defrost_end = pd.Timestamp(inactive_after_start.iloc[0])

    stable_start: pd.Timestamp | None = None
    required = {"timestamp", "water_out_temperature", "water_temperature_setpoint"}
    if required <= set(segment.columns):
        water_out = pd.to_numeric(segment["water_out_temperature"], errors="coerce")
        setpoint = pd.to_numeric(
            segment["water_temperature_setpoint"], errors="coerce"
        )
        observed = times.notna() & water_out.notna() & setpoint.notna()
        has_temperature_evidence = bool(observed.any())
        if has_temperature_evidence:
            stable_start = _stable_start(
                segment,
                heating_start,
                defrost_start,
                settings,
            )
    else:
        has_temperature_evidence = False

    if not has_temperature_evidence:
        # Temperature evidence is required for recovery/frost labels, but a
        # directly observed defrost state is an independent, useful boundary.
        # Keep the pre-defrost rows neutral and preserve the known defrost
        # interval instead of discarding it with the missing temperature data.
        stages = stages_for_partial(segment.index)
        if defrost_start is not None:
            active_interval = times.ge(defrost_start)
            if defrost_end is not None:
                active_interval &= times.lt(defrost_end)
            stages.loc[active_interval] = "defrost"
        return stages, heating_start, None, defrost_start, defrost_end

    stages = pd.Series("partial", index=segment.index, dtype="string")
    before_defrost = (
        times.lt(defrost_start) if defrost_start is not None else times.notna()
    )
    if stable_start is not None:
        stages.loc[before_defrost & times.lt(stable_start)] = "recovery"
        stages.loc[before_defrost & times.ge(stable_start)] = "frost_development"
    elif defrost_start is not None:
        # No temperature crossing means no evidence for frost development;
        # retain the known pre-defrost interval as recovery rather than hiding it.
        stages.loc[before_defrost] = "recovery"

    if defrost_start is not None:
        active_interval = times.ge(defrost_start)
        if defrost_end is not None:
            active_interval &= times.lt(defrost_end)
        stages.loc[active_interval] = "defrost"
    elif stable_start is not None:
        stages.loc[times.ge(stable_start)] = "frost_development"
    return stages, heating_start, stable_start, defrost_start, defrost_end


def stages_for_partial(index: pd.Index) -> pd.Series:
    """Return the neutral label used when an open segment has no temperature evidence."""
    return pd.Series("partial", index=index, dtype="string")


def _add_cycle_coordinates(labeled: pd.DataFrame, cycles: list[dict[str, object]]) -> None:
    labeled["cycle_elapsed_seconds"] = np.nan
    labeled["cycle_progress"] = np.nan
    for row in cycles:
        if str(row.get("cycle_status")) != "valid":
            continue
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


def _duration_in_range(value: Any, minimum: float, maximum: float) -> bool:
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
        "segment_start",
        "cycle_status",
        "cycle_status_reason",
        "heating_start",
        "stable_heating_start",
        "defrost_start",
        "defrost_end",
        "heating_duration_seconds",
        "preceding_defrost_duration_seconds",
        "terminal_defrost_duration_seconds",
    ]
