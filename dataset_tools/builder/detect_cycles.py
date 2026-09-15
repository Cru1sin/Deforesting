"""Transparent cycle labeling for the Prepared stage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .find_recovery_temperature_knee import find_global_knee

_STAGES = {
    "recovery",
    "frost_development",
    "defrost_preparation",
    "defrost",
    "partial",
}
_STATUSES = {"valid", "invalid"}


def label_cycles(
    frame: pd.DataFrame,
    defrost_column: str,
    settings: Mapping[str, Any] | Any,
    *,
    experiment_id: str,
    experiment_date: str,
    shutdown_gap_seconds: float = 60,
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
    shutdown_gaps = _shutdown_gaps(labeled, settings, shutdown_gap_seconds)
    events = _defrost_runs(labeled["timestamp"], debounced, long_gaps)
    cycles, cycle_ranges = _build_cycles(
        events,
        long_gaps,
        settings,
        labeled,
        experiment_id=experiment_id,
        experiment_date=experiment_date,
        shutdown_gaps=shutdown_gaps,
    )
    labeled = _assign_cycle_ranges(labeled, cycle_ranges)

    _label_unassigned_rows(
        labeled,
        cycles,
        experiment_id,
        experiment_date,
        defrost_column=defrost_column,
        settings=settings,
        shutdown_gaps=shutdown_gaps,
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
    shutdown_gaps: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = (),
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
        following_end = events[index + 1]["end"] or events[index + 1]["start"]
        if isinstance(following_end, pd.Timestamp) and _interval_intersects(
            heating_start, following_end, shutdown_gaps
        ):
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
    stable_start = resolve_stable_heating_start(
        labeled,
        heating_start,
        defrost_start,
        settings,
    )
    preparation_start = find_defrost_preparation_start(
        labeled, stable_start, defrost_start, settings
    )
    if stable_start is None and status == "valid":
        reason = reason or "recovery_end_not_observed"
    if (
        stable_start is not None
        and isinstance(defrost_start, pd.Timestamp)
        and stable_start >= defrost_start
    ):
        status, reason = "invalid", "invalid_cycle_boundaries"
    if status == "valid":
        mode_status, mode_reason = _operating_mode_status(
            labeled,
            heating_start,
            defrost_start,
            settings,
        )
        if mode_status == "invalid":
            status, reason = mode_status, mode_reason
        elif not reason:
            reason = mode_reason
    return _cycle_row(
        experiment_id,
        experiment_date,
        cycle_id,
        status,
        reason,
        heating_start,
        stable_start,
        preparation_start,
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
        return "valid", "defrost_state_gap"
    if defrost_start <= heating_start:
        return "invalid", "invalid_cycle_boundaries"
    if defrost_end is not None and defrost_end <= defrost_start:
        return "invalid", "invalid_cycle_boundaries"

    reason = ""
    if defrost_end is None:
        reason = "defrost_end_not_observed"
    elif following.get("boundary_uncertain") or _interval_intersects(
        heating_start, defrost_end, long_gaps
    ):
        reason = "defrost_state_gap"

    preceding_duration = previous["duration"]
    if preceding_duration is not None and not _duration_in_range(
        preceding_duration,
        _setting(settings, "minimum_defrost_seconds", 60),
        _setting(settings, "maximum_defrost_seconds", 1200),
    ):
        return "invalid", "preceding_defrost_duration_out_of_range"
    terminal_duration = following["duration"]
    if terminal_duration is not None and not _duration_in_range(
        terminal_duration,
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
    return "valid", reason


def resolve_stable_heating_start(
    frame: pd.DataFrame,
    heating_start: Any,
    defrost_start: Any,
    settings: Mapping[str, Any] | Any,
    *,
    mode: str = "criterion",
    seconds: int | None = None,
) -> pd.Timestamp | None:
    if mode == "seconds":
        if not isinstance(heating_start, pd.Timestamp) or seconds is None or seconds < 0:
            return None
        candidate = heating_start + pd.Timedelta(seconds=seconds)
        timestamps = pd.to_datetime(frame.get("timestamp", pd.Series(dtype="datetime64[ns]")))
        valid = timestamps.dropna()
        if valid.empty or valid.max() < candidate:
            return None
        return candidate
    if mode != "criterion":
        raise ValueError(f"unsupported recovery mode: {mode}")
    return find_stable_heating_start(frame, heating_start, defrost_start, settings)


def find_stable_heating_start(
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
    setpoint = pd.to_numeric(frame["water_temperature_setpoint"], errors="coerce")
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
    threshold = observations.loc[
        observations["water_out"].ge(observations["setpoint"] - 2.0),
        "timestamp",
    ]
    elapsed = (observations["timestamp"] - pd.Timestamp(heating_start)).dt.total_seconds().to_numpy(
        dtype=float
    ) / 60.0
    knee = find_global_knee(
        elapsed,
        observations["water_out"].to_numpy(dtype=float),
    )
    candidates = []
    if not threshold.empty:
        candidates.append(pd.Timestamp(threshold.iloc[0]))
    if knee is not None:
        candidates.append(pd.Timestamp(heating_start) + pd.Timedelta(minutes=knee))
    return min(candidates) if candidates else None


def find_defrost_preparation_start(
    frame: pd.DataFrame,
    stable_start: Any,
    defrost_start: Any,
    settings: Mapping[str, Any] | Any,
) -> pd.Timestamp | None:
    """Return the first compressor setpoint unload command before defrost."""
    channel = "compressor_frequency_setpoint"
    if not isinstance(defrost_start, pd.Timestamp) or channel not in frame:
        return None
    lookback = _setting(settings, "defrost_preparation_lookback_seconds", 120)
    threshold = _setting(settings, "defrost_preparation_setpoint_drop_hz", 10)
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    setpoint = pd.to_numeric(frame[channel], errors="coerce")
    start = defrost_start - pd.Timedelta(seconds=lookback)
    if isinstance(stable_start, pd.Timestamp):
        start = max(start, stable_start)
    observations = (
        pd.DataFrame({"timestamp": timestamps, "setpoint": setpoint})
        .loc[timestamps.ge(start) & timestamps.le(defrost_start)]
        .dropna()
    )
    observations = observations.sort_values("timestamp", kind="stable").drop_duplicates(
        "timestamp", keep="last"
    )
    gaps = observations["timestamp"].diff().dt.total_seconds()
    positive_gaps = gaps.loc[gaps.gt(0)]
    typical_gap = float(positive_gaps.median()) if len(positive_gaps) >= 3 else 0.0
    maximum_gap = max(
        _setting(settings, "maximum_state_gap_seconds", 5),
        typical_gap * 1.5,
        1,
    )
    groups = gaps.gt(maximum_gap).cumsum()
    for _, group in observations.groupby(groups, sort=False):
        drops = group["setpoint"].shift() - group["setpoint"]
        candidates = group.loc[drops.ge(threshold), "timestamp"]
        if not candidates.empty:
            return pd.Timestamp(candidates.iloc[0])
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
            row["defrost_preparation_start"],
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
            active_event["duration"] = (end - active_event["start"]).total_seconds()
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
    preparation_start: pd.Timestamp | None,
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
        "defrost_preparation_start": preparation_start,
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
        return "valid", "missing_operating_mode"
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
        return "valid", "missing_operating_mode"
    if not observed.eq(required).all():
        return "invalid", "non_heating_mode_present"
    return "valid", ""


def _shutdown_gaps(
    frame: pd.DataFrame,
    settings: Mapping[str, Any] | Any,
    maximum_seconds: float,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    """Find sensor gaps that also cross from heating mode into shutdown."""
    channel = _string_setting(settings, "operating_mode_channel", "")
    if not channel or channel not in frame:
        return ()
    valid = frame[channel].notna()
    for suffix in ("__duplicate", "__conflict"):
        quality = f"{channel}{suffix}"
        if quality in frame:
            valid &= ~frame[quality].fillna(False).astype(bool)
    observed = frame.loc[valid, ["timestamp", channel]]
    modes = observed[channel].astype(str).str.removesuffix(".0")
    required = _string_setting(settings, "required_operating_mode", "3").removesuffix(".0")
    shutdown = (
        observed["timestamp"].diff().dt.total_seconds().gt(maximum_seconds)
        & modes.shift().eq(required)
        & modes.ne(required)
    )
    return tuple(
        zip(
            observed["timestamp"].shift().loc[shutdown],
            observed.loc[shutdown, "timestamp"],
            strict=True,
        )
    )


def _string_setting(settings: Mapping[str, Any] | Any, name: str, default: str) -> str:
    if isinstance(settings, Mapping):
        return str(settings.get(name, default))
    return str(getattr(settings, name, default))


def _stage_for_times(
    times: pd.Series,
    stable_start: object,
    preparation_start: object,
    defrost_start: object,
    defrost_end: object,
) -> pd.Series:
    stage = pd.Series("partial", index=times.index, dtype="string")
    if not isinstance(stable_start, pd.Timestamp) or not isinstance(defrost_start, pd.Timestamp):
        return stage
    stage.loc[times.lt(stable_start)] = "recovery"
    frost_end = preparation_start if isinstance(preparation_start, pd.Timestamp) else defrost_start
    stage.loc[times.ge(stable_start) & times.lt(frost_end)] = "frost_development"
    if isinstance(preparation_start, pd.Timestamp):
        stage.loc[times.ge(preparation_start) & times.lt(defrost_start)] = "defrost_preparation"
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
    shutdown_gaps: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = (),
) -> None:
    unassigned = labeled["cycle_id"].isna().to_numpy()
    gap_ends = {end for _start, end in shutdown_gaps}
    position = 0
    partial_index = 1
    while position < len(unassigned):
        if not unassigned[position]:
            position += 1
            continue
        end = position
        while (
            end + 1 < len(unassigned)
            and unassigned[end + 1]
            and labeled["timestamp"].iloc[end + 1] not in gap_ends
        ):
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
        preparation_start = find_defrost_preparation_start(
            segment, stable_start, defrost_start, settings
        )
        if preparation_start is not None and defrost_start is not None:
            times = pd.to_datetime(segment["timestamp"], errors="coerce")
            segment_stages.loc[times.ge(preparation_start) & times.lt(defrost_start)] = (
                "defrost_preparation"
            )
        if defrost_start is not None and defrost_end is None:
            segment_end = pd.Timestamp(segment["timestamp"].max())
            following_starts = [
                value
                for row in cycles
                if isinstance(value := row.get("heating_start"), pd.Timestamp)
                and value > segment_end
            ]
            if following_starts:
                defrost_end = min(following_starts)
        labeled.loc[index, "cycle_id"] = partial_id
        labeled.loc[index, "cycle_stage"] = segment_stages.to_numpy()
        interrupted = end + 1 < len(unassigned) and labeled["timestamp"].iloc[end + 1] in gap_ends
        status = "invalid" if interrupted else "valid"
        reason = "shutdown_during_sensor_gap" if interrupted else "outside_complete_cycle"
        labeled.loc[index, "cycle_status"] = status
        labeled.loc[index, "cycle_status_reason"] = reason
        partial = _cycle_row(
            experiment_id,
            experiment_date,
            partial_id,
            status,
            reason,
            heating_start,
            stable_start,
            preparation_start,
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
    Structural limitations remain explicit in the reason and boundary fields;
    they do not create a third quality status.
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
    defrost_start = pd.Timestamp(active_times.iloc[0]) if not active_times.empty else None
    defrost_end: pd.Timestamp | None = None
    if defrost_start is not None:
        inactive_after_start = (
            times.loc[states.eq(False) & times.gt(defrost_start)]
            .dropna()
            .sort_values(kind="stable")
        )
        if not inactive_after_start.empty:
            defrost_end = pd.Timestamp(inactive_after_start.iloc[0])

    stable_start: pd.Timestamp | None = None
    required = {"timestamp", "water_out_temperature", "water_temperature_setpoint"}
    if required <= set(segment.columns):
        water_out = pd.to_numeric(segment["water_out_temperature"], errors="coerce")
        setpoint = pd.to_numeric(segment["water_temperature_setpoint"], errors="coerce")
        observed = times.notna() & water_out.notna() & setpoint.notna()
        has_temperature_evidence = bool(observed.any())
        if has_temperature_evidence:
            stable_start = resolve_stable_heating_start(
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
    before_defrost = times.lt(defrost_start) if defrost_start is not None else times.notna()
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
        frost_end = row.get("defrost_preparation_start")
        if not isinstance(frost_end, pd.Timestamp):
            frost_end = defrost_start
        if not isinstance(stable_start, pd.Timestamp) or not isinstance(
            defrost_start, pd.Timestamp
        ):
            continue
        cycle_id = str(row["cycle_id"])
        mask = labeled["cycle_id"].eq(cycle_id) & labeled["cycle_stage"].eq("frost_development")
        elapsed = (labeled.loc[mask, "timestamp"] - stable_start).dt.total_seconds()
        duration = (frost_end - stable_start).total_seconds()
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
        "defrost_preparation_start",
        "defrost_start",
        "defrost_end",
        "heating_duration_seconds",
        "preceding_defrost_duration_seconds",
        "terminal_defrost_duration_seconds",
    ]


RECOVERY_RULES = ("frequency-setpoint", "frequency-actual")
RECOVERY_DEFAULTS = {
    "recovery_rule": "frequency-setpoint",
    "startup_frequency_max": 42.0,
    "frequency_ramp": 1.0,
    "slow_fraction": 0.25,
    "minimum_state_observations": 3,
    "control_intervals": 3,
    "gap_seconds": 30.0,
}


def add_recovery_arguments(parser):
    parser.add_argument(
        "--recovery-rule", choices=RECOVERY_RULES, default=RECOVERY_DEFAULTS["recovery_rule"]
    )
    for name, default in RECOVERY_DEFAULTS.items():
        if name != "recovery_rule":
            parser.add_argument(
                "--" + name.replace("_", "-"),
                type=int if name in {"minimum_state_observations", "control_intervals"} else float,
                default=default,
            )


def recovery_settings(args):
    return {name: getattr(args, name) for name in RECOVERY_DEFAULTS}


def recovery_control_trace(frame: pd.DataFrame, settings: Mapping) -> pd.DataFrame:
    """Offline control-state segmentation from frequency changes and their intervals.

    A slower command and subsequent command pattern identify normal regulation.
    The ramp reference is the median of preceding intervals, not startup speed. A
    terminal plateau must outlast the observed ramp cadence; a truncated short
    hold cannot prove recovery. No confirmation delay is added to the boundary.
    """
    if set(settings) - set(RECOVERY_DEFAULTS):
        raise ValueError("obsolete recovery settings: use the offline frequency-state definition")
    cfg = {**RECOVERY_DEFAULTS, **settings}
    if cfg["recovery_rule"] not in RECOVERY_RULES:
        raise ValueError("offline recovery uses frequency setpoint or actual frequency")
    if any(float(cfg[key]) <= 0 for key in RECOVERY_DEFAULTS if key != "recovery_rule"):
        raise ValueError("recovery thresholds must be positive")
    if cfg["slow_fraction"] >= 1:
        raise ValueError("slow fraction must be less than one")
    values = frame.copy()
    values["timestamp"] = pd.to_datetime(values.timestamp, errors="coerce")
    values = (
        values.dropna(subset=["timestamp"])
        .sort_values("timestamp", kind="stable")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    signal = (
        "compressor_frequency_setpoint"
        if cfg["recovery_rule"] == "frequency-setpoint"
        else "compressor_frequency"
    )
    raw = pd.to_numeric(
        values.get(signal, pd.Series(np.nan, index=values.index)), errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    result = values[["timestamp"]].copy()
    for column in (
        "compressor_frequency_setpoint",
        "compressor_frequency",
        "condensing_pressure",
        "evaporating_pressure",
        "condensing_temperature",
        "plate_heat_exchanger_inlet_temperature",
        "water_temperature_setpoint",
    ):
        if column in values:
            result[column] = pd.to_numeric(values[column], errors="coerce")
    if {"condensing_pressure", "evaporating_pressure"} <= set(result):
        result["pressure_difference"] = result.condensing_pressure - result.evaporating_pressure
    result["normal_heating"] = False
    result[signal + "_slope_per_minute"] = np.nan
    result[signal + "_fast_reference"] = np.nan
    result["recovery_status"] = "missing_signal"
    observed = pd.DataFrame({"timestamp": values.timestamp, "value": raw}).dropna()
    if observed.empty:
        return result
    low = observed.loc[observed.value.between(0, cfg["startup_frequency_max"])]
    result["recovery_status"] = "awaiting_startup_observation"
    if low.empty:
        return result
    groups = observed.value.ne(observed.value.shift()).cumsum()
    states = observed.groupby(groups).agg(
        timestamp=("timestamp", "first"),
        end=("timestamp", "last"),
        value=("value", "first"),
        count=("value", "size"),
    )
    states = states.loc[states["count"].ge(cfg["minimum_state_observations"])]
    states = states.loc[
        states.timestamp.gt(low.timestamp.iloc[0]) & states.value.gt(cfg["startup_frequency_max"])
    ].reset_index(drop=True)
    result["recovery_status"] = "awaiting_sustained_ramp"
    count = cfg["control_intervals"]
    if len(states) <= count:
        return result
    intervals = states.timestamp.shift(-1).sub(states.timestamp).dt.total_seconds() / 60
    rates = states.value.shift(-1).sub(states.value).div(intervals)
    references = rates.expanding(min_periods=count).median().shift()
    sampling_minutes = float(observed.timestamp.diff().dt.total_seconds().median()) / 60
    increments = states.value.shift(-1).sub(states.value)
    reference_uppers = (
        increments.div((intervals - sampling_minutes).clip(lower=sampling_minutes))
        .expanding(min_periods=count)
        .median()
        .shift()
    )
    reference = float(references.iloc[count])
    reference_upper = float(reference_uppers.iloc[count])
    result[signal + "_fast_reference"] = reference
    result[signal + "_reference_upper"] = reference_upper
    result["sampling_interval_seconds"] = sampling_minutes * 60
    if not np.isfinite(reference) or reference < cfg["frequency_ramp"]:
        return result
    # The final hold is evidence only if the record covers a normal ramp update interval.
    cadence = float(intervals.loc[rates.gt(reference * cfg["slow_fraction"])].max())
    if (states.end.iloc[-1] - states.timestamp.iloc[-1]).total_seconds() / 60 >= cadence:
        rates.iloc[-1] = 0.0
    positions = (
        np.searchsorted(states.timestamp.to_numpy(), values.timestamp.to_numpy(), side="right") - 1
    )
    valid = positions >= 0
    result.loc[valid, signal + "_slope_per_minute"] = rates.to_numpy()[positions[valid]]
    # Command timestamps have one sampling interval of uncertainty at each interval estimate.
    lower_rates = increments.div(intervals + sampling_minutes)
    result.loc[valid, signal + "_slope_lower"] = lower_rates.to_numpy()[positions[valid]]
    lower_rates.iloc[-1] = rates.iloc[-1]
    # A small individual step is not a mode change: the subsequent command pattern
    # must also be slow. Offline evidence locates the start, without a time offset.
    following_rates = lower_rates.iloc[::-1].rolling(count, min_periods=1).median().iloc[::-1]
    thresholds = reference_uppers * cfg["slow_fraction"]
    eligible = (
        lower_rates.le(thresholds)
        & following_rates.le(thresholds)
        & references.ge(cfg["frequency_ramp"])
    )
    result["recovery_status"] = "transition_not_observed"
    if not eligible.any():
        if (
            observed.loc[observed.timestamp.ge(states.timestamp.iloc[0]), "timestamp"]
            .diff()
            .dt.total_seconds()
            .gt(cfg["gap_seconds"])
            .any()
        ):
            result["recovery_status"] = "transition_hidden_by_gap"
        return result
    position = int(np.flatnonzero(eligible)[0])
    boundary = states.timestamp.iloc[position]
    result[signal + "_fast_reference"] = references.iloc[position]
    result[signal + "_reference_upper"] = reference_uppers.iloc[position]
    support_position = min(position + count, len(states) - 1)
    support_end = states.timestamp.iloc[support_position]
    if support_position == len(states) - 1 and pd.notna(rates.iloc[-1]):
        support_end += pd.Timedelta(minutes=cadence)
    # A directly observed hold supplies local evidence without waiting for remote
    # commands. Its duration scales with the preceding command cadence, not seconds.
    hold_minutes = float(intervals.iloc[:position].median()) / cfg["slow_fraction"]
    hold_end = boundary + pd.Timedelta(minutes=hold_minutes)
    if states.end.iloc[position] >= hold_end:
        support_end = min(support_end, hold_end)
    evidence = observed.loc[observed.timestamp.between(states.timestamp.iloc[0], support_end)]
    if evidence.timestamp.diff().dt.total_seconds().gt(cfg["gap_seconds"]).any():
        result["recovery_status"] = "transition_hidden_by_gap"
        return result
    result["normal_heating"] = values.timestamp.ge(boundary)
    result["recovery_status"] = np.where(result.normal_heating, "identified_offline", "recovery")
    result["transition_support_end"] = support_end
    return result


def heating_episode_bounds(loader, cycle_name, frame):
    """Exclude pre-start standby only when no adjacent defrost precedes startup."""
    record = loader.get_cycle_record(cycle_name)
    bounds = dict(record["boundaries"])
    recorded = pd.to_datetime(bounds.get("recorded_heating_start", bounds.get("heating_start")))
    bounds.update(
        recorded_heating_start=recorded,
        heating_start=recorded,
        heating_origin="post_defrost",
        excluded_prestart_seconds=0.0,
    )
    cycles = loader.list_cycles()
    experiment = cycles.loc[cycles.experiment_id.eq(record["experiment_id"])].reset_index(drop=True)
    position = int(experiment.index[experiment.cycle_name.eq(cycle_name)][0])
    preceding_end = (
        pd.to_datetime(experiment.iloc[position - 1].get("defrost_end")) if position else pd.NaT
    )
    if pd.notna(preceding_end) and abs((recorded - preceding_end).total_seconds()) <= 60:
        return bounds
    bounds["heating_origin"] = "initial_recording" if position == 0 else "restart_recording"
    end = pd.to_datetime(bounds.get("defrost_preparation_start") or bounds.get("defrost_start"))
    observed = frame.reindex(columns=["timestamp", "compressor_frequency_setpoint"]).copy()
    observed["compressor_frequency_setpoint"] = pd.to_numeric(
        observed.compressor_frequency_setpoint, errors="coerce"
    )
    observed["timestamp"] = pd.to_datetime(observed.timestamp)
    observed = observed.loc[observed.timestamp.ge(recorded)]
    if pd.notna(end):
        observed = observed.loc[observed.timestamp.lt(end)]
    observed = observed.dropna().sort_values("timestamp").drop_duplicates("timestamp")
    if observed.empty or observed.compressor_frequency_setpoint.iloc[0] != 0:
        return bounds
    running = observed.loc[observed.compressor_frequency_setpoint.gt(0), "timestamp"]
    start = running.iloc[0] if len(running) else pd.NaT
    bounds.update(
        heating_start=start,
        heating_origin="cold_start" if len(running) else "not_started",
        excluded_prestart_seconds=(start - recorded).total_seconds(),
    )
    return bounds


def recovery_stages(frame, boundaries):
    """Apply a confirmed recovery boundary, preserving preparation/defrost observations."""
    times = pd.to_datetime(frame.timestamp)
    stage = frame.cycle_stage.copy()
    preparation = pd.to_datetime(boundaries.get("defrost_preparation_start"))
    before = (
        times.lt(preparation)
        if pd.notna(preparation)
        else ~stage.isin(["defrost", "defrost_preparation"])
    )
    stage.loc[before] = "recovery"
    if "heating_origin" in boundaries:
        stage.loc[times.lt(pd.to_datetime(boundaries.get("heating_start")))] = pd.NA
    stable = pd.to_datetime(boundaries.get("stable_heating_start"))
    if pd.notna(stable):
        stage.loc[before & times.ge(stable)] = "frost_development"
    return stage


def audit_recovery_cycle(loader, cycle_name, settings, output=None, *, figures=False):
    """Use the same detector for all recipes; keep raw controller codes as evidence."""
    record = loader.get_cycle_record(cycle_name)
    frame = loader.load_cycle_original(cycle_name)
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    bounds = heating_episode_bounds(loader, cycle_name, frame)
    start = pd.to_datetime(bounds.get("heating_start"))
    end = pd.to_datetime(bounds.get("defrost_preparation_start") or bounds.get("defrost_start"))
    heating = frame.loc[frame.timestamp.ge(start)] if pd.notna(start) else frame.iloc[:0]
    if pd.notna(end):
        heating = heating.loc[heating.timestamp.lt(end)]
    rows, traces, codes = [], {}, []
    for rule in RECOVERY_RULES:
        trace = recovery_control_trace(heating, {**settings, "recovery_rule": rule})
        traces[rule] = trace
        confirmed = trace.loc[trace.normal_heating, "timestamp"]
        boundary = confirmed.iloc[0] if len(confirmed) else pd.NaT
        evidence = {}
        for signal in (
            "compressor_frequency_setpoint",
            "compressor_frequency",
            "pressure_difference",
        ):
            column = signal + "_slope_per_minute"
            if pd.isna(boundary) or column not in trace:
                continue
            at_boundary = trace.loc[trace.timestamp.eq(boundary)].iloc[0]
            reference = at_boundary[signal + "_fast_reference"]
            after = trace.loc[
                trace.timestamp.ge(boundary)
                & trace.timestamp.le(boundary + pd.Timedelta(seconds=120))
            ]
            evidence[signal + "_reference"] = reference
            evidence[signal + "_reference_upper"] = at_boundary.get(signal + "_reference_upper")
            evidence[signal + "_next_command_slope_lower"] = at_boundary.get(
                signal + "_slope_lower"
            )
            evidence["sampling_interval_seconds"] = at_boundary.get("sampling_interval_seconds")
            evidence["transition_support_end"] = at_boundary.get("transition_support_end")
            evidence[signal + "_slope_at_boundary"] = at_boundary[column]
            observed_after = after.dropna(subset=[signal])
            if len(observed_after) >= 2:
                x = (observed_after.timestamp - boundary).dt.total_seconds().to_numpy() / 60
                evidence[signal + "_next120_raw_slope"] = float(
                    np.polyfit(x, observed_after[signal], 1)[0]
                )
        rows.append(
            {
                "cycle_name": cycle_name,
                "experiment_id": record["experiment_id"],
                "catalog_status": record["status"],
                "recovery_rule": rule,
                "heating_start": start,
                "recorded_heating_start": bounds["recorded_heating_start"],
                "heating_origin": bounds["heating_origin"],
                "excluded_prestart_seconds": bounds["excluded_prestart_seconds"],
                "stable_heating_start": boundary,
                "old_stable_heating_start": bounds.get("stable_heating_start"),
                **evidence,
                "recovery_minutes": (boundary - start).total_seconds() / 60,
                "recovery_status": trace.recovery_status.iloc[-1]
                if len(trace)
                else "no_heating_observations",
            }
        )
    for column in (
        "p1__压机状态'1_00",
        "p1__系统状态'1_00'",
        "p1__频率状态<1_00>",
        "p1__FF调节状态",
        "p1__四通阀",
    ):
        values = frame.get(column, pd.Series(dtype=object)).dropna()
        changes = values.ne(values.shift())
        for index in values.index[changes]:
            codes.append(
                {
                    "cycle_name": cycle_name,
                    "field": column,
                    "timestamp": frame.loc[index, "timestamp"],
                    "value": values.loc[index],
                    "coverage": len(values) / max(len(frame), 1),
                }
            )
    if output is not None:
        if figures:
            from plots.publication import render_recovery_audit

            render_recovery_audit(frame, record, traces, output / f"{cycle_name}.png")
        tables = []
        for row in rows:
            trace = traces[row["recovery_rule"]]
            stages = frame[["timestamp", "cycle_stage"]].copy()
            stages["cycle_stage"] = recovery_stages(
                frame, {**bounds, "stable_heating_start": row["stable_heating_start"]}
            )
            tables.append(
                stages.merge(trace, on="timestamp", how="left").assign(
                    recovery_rule=row["recovery_rule"]
                )
            )
        pd.concat(tables, ignore_index=True).to_parquet(
            output / f"{cycle_name}_states.parquet", index=False
        )
    return rows, codes
