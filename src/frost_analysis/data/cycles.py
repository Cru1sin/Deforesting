"""Explicit-state-first cycle segmentation with retained diagnostic evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from ..core.artifacts import FrameWriteResult, write_dataframe


@dataclass(frozen=True)
class SegmentationResult:
    frame: pd.DataFrame
    cycles: pd.DataFrame
    debounce_events: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CycleValidationResult:
    """Validated row labels, cycle records, sampling interval, and warnings."""

    frame: pd.DataFrame
    cycles: pd.DataFrame
    nominal_interval_seconds: float
    warnings: tuple[str, ...] = ()


def normalize_cycle_status(quality: object) -> str:
    """Map detailed segmentation quality to the three public cycle states."""
    status_mapping = {
        "complete": "valid",
        "contaminated": "invalid",
        "abnormal": "invalid",
        "excluded": "invalid",
        "partial": "incomplete",
    }
    return status_mapping.get(str(quality), "incomplete")


def append_issue(existing: object, issue: str) -> str:
    """Append one reason without serializing missing values as ``"nan"``."""
    previous = _text_or_empty(existing)
    return ";".join(value for value in (previous, issue.strip()) if value)


def _text_or_empty(value: object) -> str:
    """Render one optional scalar as readable text instead of ``"nan"``."""
    missing = value is None or bool(pd.isna(cast(Any, value)))
    return "" if missing else str(value).strip()


def infer_sampling_interval_seconds(
    frame: pd.DataFrame,
    *,
    expected_sampling_interval_seconds: float | None = None,
) -> float:
    """Infer the median positive timestamp delta or use an explicit fallback."""
    timestamps = pd.to_datetime(
        frame.get("timestamp", pd.Series(dtype=object)), errors="coerce"
    ).dropna()
    deltas = timestamps.sort_values().drop_duplicates().diff().dt.total_seconds()
    positive = deltas[deltas.gt(0)]
    if not positive.empty:
        inferred = float(positive.median())
        if np.isfinite(inferred) and inferred > 0:
            return inferred
    if expected_sampling_interval_seconds is not None:
        fallback = float(expected_sampling_interval_seconds)
        if np.isfinite(fallback) and fallback > 0:
            return fallback
    raise ValueError("cannot infer positive sampling interval")


def validate_cycles(
    segmentation: SegmentationResult,
    config: Mapping[str, object],
) -> CycleValidationResult:
    """Apply mode and gap checks after segmentation without reconstructing data."""
    # Copy the read-only Mapping once; validation may need ordinary dict access.
    settings = dict(config)
    # The median positive delta represents normal sampling better than a mean with gaps.
    nominal_interval = infer_sampling_interval_seconds(
        segmentation.frame,
        expected_sampling_interval_seconds=_optional_positive_float(
            settings.get("expected_sampling_interval_seconds")
        ),
    )
    # First reject non-heating rows, then classify unusually large observed gaps.
    labeled, validated, mode_warnings = _enforce_heating_mode(
        segmentation.frame,
        segmentation.cycles,
    )
    labeled, validated, gap_warnings = _mark_long_gap_cycles(
        labeled,
        validated,
        nominal_seconds=nominal_interval,
        factor=as_optional_float(settings.get("gap_warning_factor", 3.0)) or 3.0,
    )
    warnings = (*segmentation.warnings, *mode_warnings, *gap_warnings)
    # Keep warning order stable while removing duplicate messages.
    return CycleValidationResult(
        frame=labeled,
        cycles=validated,
        nominal_interval_seconds=nominal_interval,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def segment_cycles(  # noqa: C901 - explicit boundary cases remain auditable in one flow
    frame: pd.DataFrame,
    state_column: str,
    config: dict[str, Any],
) -> SegmentationResult:
    """Split post-defrost heating through the next defrost, preserving partials."""
    required = {"timestamp", state_column}
    if not required <= set(frame.columns):
        raise ValueError(f"segmentation requires columns: {sorted(required)}")
    labeled = frame.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    labeled["timestamp"] = pd.to_datetime(labeled["timestamp"], errors="raise")
    manual = config.get("manual_overrides", [])
    if manual:
        cycles = pd.DataFrame([_manual_cycle(record, config) for record in manual])
        return SegmentationResult(_label_rows(labeled, cycles), cycles, 0)

    raw_state = labeled[state_column].astype("string").str.strip().str.upper()
    on_values = {str(value).strip().upper() for value in config.get("defrost_on_values", ["ON"])}
    off_values = {str(value).strip().upper() for value in config.get("defrost_off_values", ["OFF"])}
    overlap = on_values & off_values
    if overlap:
        raise ValueError(f"defrost on/off values overlap: {sorted(overlap)}")
    state = raw_state.map(
        lambda value: "ON" if value in on_values else ("OFF" if value in off_values else pd.NA)
    ).astype("string")
    if not state.notna().any():
        raise ValueError("no recognized defrost states in configured state column")
    recognized = state.notna()
    state = state.ffill().bfill()
    debounced, debounce_events = _debounce(
        labeled["timestamp"], state, float(config.get("debounce_seconds", 20))
    )
    labeled["defrost_state_debounced"] = debounced
    on_runs = _on_runs(labeled["timestamp"], debounced)
    cycle_rows: list[dict[str, object]] = []
    warnings: list[str] = []

    if on_runs and labeled["timestamp"].iloc[0] < on_runs[0][0]:
        cycle_rows.append(
            _partial_row(
                "partial_leading",
                labeled["timestamp"].iloc[0],
                on_runs[0][0],
                "data_starts_mid_cycle",
            )
        )

    for index in range(len(on_runs) - 1):
        previous = on_runs[index]
        following = on_runs[index + 1]
        if previous[2] is None or following[2] is None:
            continue
        heating_start = previous[2]
        stable_start = heating_start + pd.Timedelta(
            seconds=float(config.get("recovery_seconds", 180))
        )
        defrost_start = following[0]
        defrost_end = following[2]
        cycle_id = f"cycle_{index + 1:03d}"
        cycle_rows.append(
            _automatic_cycle(
                labeled,
                cycle_id,
                heating_start,
                stable_start,
                defrost_start,
                defrost_end,
                config,
                state_coverage=float(
                    recognized.loc[
                        labeled["timestamp"].between(heating_start, defrost_end)
                    ].mean()
                ),
                warnings=warnings,
            )
        )

    if on_runs:
        last_end = on_runs[-1][2]
        if last_end is not None and last_end < labeled["timestamp"].iloc[-1]:
            cycle_rows.append(
                _partial_row(
                    "partial_trailing",
                    last_end,
                    labeled["timestamp"].iloc[-1],
                    "data_ends_before_next_defrost",
                )
            )
    elif not labeled.empty:
        cycle_rows.append(
            _partial_row(
                "partial_only",
                labeled["timestamp"].iloc[0],
                labeled["timestamp"].iloc[-1],
                "no_defrost_event",
            )
        )

    cycles = pd.DataFrame(cycle_rows, columns=_cycle_columns())
    return SegmentationResult(
        _label_rows(labeled, cycles), cycles, debounce_events, tuple(dict.fromkeys(warnings))
    )


def write_segmentation(
    result: SegmentationResult, processed_dir: Any, tables_dir: Any
) -> FrameWriteResult:
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    storage = write_dataframe(result.frame, processed_dir / "cycle_labeled_timeseries.parquet")
    result.cycles.to_csv(tables_dir / "cycle_table.csv", index=False)
    return storage


def _debounce(times: pd.Series, state: pd.Series, threshold_s: float) -> tuple[pd.Series, int]:
    if state.empty or threshold_s <= 0:
        return state, 0
    result = state.copy()
    events = 0
    changed = True
    passes = 0
    while changed:
        passes += 1
        if passes > len(result) + 1:
            raise RuntimeError("defrost state debounce failed to converge")
        changed = False
        run_ids = result.ne(result.shift()).fillna(True).cumsum()
        runs = list(result.groupby(run_ids, sort=False))
        for run_position, (_, run) in enumerate(runs):
            start = int(run.index[0])
            end = int(run.index[-1])
            nominal = (
                float(times.diff().dt.total_seconds().dropna().median()) if len(times) > 1 else 1.0
            )
            duration = float((times.iloc[end] - times.iloc[start]).total_seconds() + nominal)
            if duration >= threshold_s or run_position == 0:
                continue
            replacement = result.iloc[start - 1]
            if pd.isna(replacement) or result.iloc[start : end + 1].eq(replacement).all():
                continue
            result.iloc[start : end + 1] = replacement
            events += 1
            changed = True
            break
    return result, events


def _on_runs(
    times: pd.Series, state: pd.Series
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None]]:
    runs: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None]] = []
    run_ids = state.ne(state.shift()).fillna(True).cumsum()
    for _, part in state.groupby(run_ids, sort=False):
        if part.iloc[0] != "ON":
            continue
        start_index, end_index = int(part.index[0]), int(part.index[-1])
        next_off = times.iloc[end_index + 1] if end_index + 1 < len(times) else None
        runs.append((times.iloc[start_index], times.iloc[end_index], next_off))
    return runs


def _automatic_cycle(
    frame: pd.DataFrame,
    cycle_id: str,
    heating_start: pd.Timestamp,
    stable_start: pd.Timestamp,
    defrost_start: pd.Timestamp,
    defrost_end: pd.Timestamp,
    config: dict[str, Any],
    *,
    state_coverage: float,
    warnings: list[str],
) -> dict[str, object]:
    heating_duration = float((defrost_start - heating_start).total_seconds())
    defrost_duration = float((defrost_end - defrost_start).total_seconds())
    reasons: list[str] = []
    quality = "complete"
    if (
        not float(config.get("min_heating_seconds", 0))
        <= heating_duration
        <= float(config.get("max_heating_seconds", np.inf))
    ):
        quality = "excluded"
        reasons.append("heating_duration_out_of_range")
    if (
        not float(config.get("min_defrost_seconds", 0))
        <= defrost_duration
        <= float(config.get("max_defrost_seconds", np.inf))
    ):
        quality = "excluded"
        reasons.append("defrost_duration_out_of_range")
    frequency_column = str(config.get("compressor_frequency_column", "compressor_frequency"))
    if frequency_column in frame:
        normal = frame.loc[
            frame["timestamp"].between(heating_start, defrost_start, inclusive="left"),
            frequency_column,
        ]
        zero_run = normal.fillna(0).le(0)
        max_zero = (
            int(zero_run.groupby(zero_run.ne(zero_run.shift()).cumsum()).sum().max())
            if not zero_run.empty
            else 0
        )
        if max_zero >= int(config.get("shutdown_min_rows", 3)):
            quality = "abnormal"
            reasons.append("compressor_shutdown")
    evidence = "explicit_defrost_flag"
    corroboration = _corroboration(frame, defrost_start, config)
    interval = frame.loc[frame["timestamp"].between(heating_start, defrost_end)].sort_values(
        "timestamp"
    )
    interval_times = interval["timestamp"]
    deltas = interval_times.diff().dt.total_seconds().dropna()
    nominal = float(deltas[deltas.gt(0)].median()) if deltas.gt(0).any() else 1.0
    gap_columns = [
        column for column in config.get("corroboration_columns", []) if str(column) in interval
    ]
    gap_details: list[str] = []
    signal_gaps: list[float] = []
    for column_value in gap_columns:
        column = str(column_value)
        observed_times = interval.loc[interval[column].notna(), "timestamp"]
        observed_deltas = observed_times.diff().dt.total_seconds().dropna()
        signal_gap = float(observed_deltas.max()) if not observed_deltas.empty else 0.0
        signal_gaps.append(signal_gap)
        gap_details.append(f"{column}={signal_gap:.1f}")
    timestamp_gap = float(deltas.max()) if not deltas.empty else 0.0
    maximum_gap = max([timestamp_gap, *signal_gaps])
    gap_limit = nominal * float(config.get("gap_warning_factor", 3.0))
    gap_penalty = 0.0
    if maximum_gap > gap_limit:
        ratio = maximum_gap / max(gap_limit, 1e-9)
        gap_penalty = min(0.25, 0.05 + 0.05 * float(np.log10(max(ratio, 1.0))))
        warnings.append(
            f"long_gap:{cycle_id}:maximum_gap_seconds={maximum_gap:.1f}:"
            f"confidence_penalty={gap_penalty:.3f}"
        )
    status_penalty = {
        "corroborated": 0.0,
        "partial_support": 0.05,
        "no_signal_change": 0.15,
        "not_available": 0.20,
        "conflict": 0.25,
    }[str(corroboration["status"])]
    state_penalty = 0.4 * max(0.0, 1.0 - state_coverage)
    confidence = max(0.0, 0.95 - state_penalty - status_penalty - gap_penalty)
    penalties = (
        f"state={state_penalty:.3f};corroboration={status_penalty:.3f};gap={gap_penalty:.3f}"
    )
    return {
        "cycle_id": cycle_id,
        "heating_start": heating_start,
        "stable_heating_start": stable_start,
        "clean_start": pd.NaT,
        "clean_end": pd.NaT,
        "defrost_start": defrost_start,
        "defrost_end": defrost_end,
        "cycle_duration": float((defrost_end - heating_start).total_seconds()),
        "heating_duration": heating_duration,
        "defrost_duration": defrost_duration,
        "segmentation_method": "explicit_defrost_flag_debounced",
        "segmentation_confidence": confidence,
        "segmentation_evidence": f"{evidence};{corroboration['evidence']}",
        "corroboration_status": corroboration["status"],
        "state_coverage": state_coverage,
        "maximum_gap_seconds": maximum_gap,
        "gap_evidence": f"timestamp={timestamp_gap:.1f};" + ";".join(gap_details),
        "confidence_penalties": penalties,
        "quality_flag": quality,
        "exclusion_reason": ";".join(reasons),
    }


def _corroboration(
    frame: pd.DataFrame, boundary: pd.Timestamp, config: dict[str, Any]
) -> dict[str, str]:
    columns = [
        str(column) for column in config.get("corroboration_columns", []) if str(column) in frame
    ]
    if not columns:
        columns = [
            column for column in ("four_way_valve", "evidence_temperature") if column in frame
        ]
    if not columns:
        return {"status": "not_available", "evidence": "corroboration=not_available"}
    seconds = float(config.get("corroboration_window_seconds", 30))
    before = frame.loc[
        frame["timestamp"].between(
            boundary - pd.Timedelta(seconds=seconds), boundary, inclusive="left"
        )
    ]
    after = frame.loc[
        frame["timestamp"].between(
            boundary, boundary + pd.Timedelta(seconds=seconds), inclusive="both"
        )
    ]
    expected = {
        str(key): str(value).lower()
        for key, value in dict(config.get("corroboration_expected_directions", {})).items()
    }
    threshold = float(config.get("corroboration_min_normalized_change", 1.0))
    support = 0
    conflict = 0
    changed = 0
    details: list[str] = []
    for column in columns:
        left = pd.to_numeric(before[column], errors="coerce").dropna()
        right = pd.to_numeric(after[column], errors="coerce").dropna()
        if left.empty or right.empty:
            details.append(f"{column}:unavailable")
            continue
        combined = pd.concat([left, right])
        median = float(combined.median())
        scale = max(float((combined - median).abs().median() * 1.4826), 1e-6)
        delta = float(right.median() - left.median())
        normalized = delta / scale
        direction = "positive" if delta > 0 else "negative" if delta < 0 else "none"
        if abs(normalized) >= threshold:
            changed += 1
            expected_direction = expected.get(column)
            if expected_direction is None or direction == expected_direction:
                support += 1
            else:
                conflict += 1
        details.append(f"{column}:delta={delta:.6g}:z={normalized:.3f}:direction={direction}")
    if conflict and conflict >= support:
        status = "conflict"
    elif support >= int(config.get("corroboration_min_supporting_signals", 1)):
        status = "corroborated"
    elif changed:
        status = "partial_support"
    else:
        status = "no_signal_change"
    return {"status": status, "evidence": f"corroboration={status};" + "|".join(details)}


def _manual_cycle(record: dict[str, Any], config: dict[str, Any]) -> dict[str, object]:
    heating_start = pd.Timestamp(record["heating_start"])
    stable = pd.Timestamp(
        record.get(
            "stable_heating_start",
            heating_start + pd.Timedelta(seconds=float(config.get("recovery_seconds", 180))),
        )
    )
    defrost_start = pd.Timestamp(record["defrost_start"])
    defrost_end = pd.Timestamp(record["defrost_end"])
    return {
        "cycle_id": str(record["cycle_id"]),
        "heating_start": heating_start,
        "stable_heating_start": stable,
        "clean_start": pd.NaT,
        "clean_end": pd.NaT,
        "defrost_start": defrost_start,
        "defrost_end": defrost_end,
        "cycle_duration": float((defrost_end - heating_start).total_seconds()),
        "heating_duration": float((defrost_start - heating_start).total_seconds()),
        "defrost_duration": float((defrost_end - defrost_start).total_seconds()),
        "segmentation_method": "manual_override",
        "segmentation_confidence": 1.0,
        "segmentation_evidence": "configured_manual_override",
        "corroboration_status": "manual",
        "state_coverage": 1.0,
        "maximum_gap_seconds": 0.0,
        "gap_evidence": "manual_override",
        "confidence_penalties": "manual=0.000",
        "quality_flag": str(record.get("quality_flag", "complete")),
        "exclusion_reason": str(record.get("exclusion_reason", "")),
    }


def _partial_row(
    cycle_id: str, start: pd.Timestamp, end: pd.Timestamp, reason: str
) -> dict[str, object]:
    return {
        "cycle_id": cycle_id,
        "heating_start": start,
        "stable_heating_start": pd.NaT,
        "clean_start": pd.NaT,
        "clean_end": pd.NaT,
        "defrost_start": pd.NaT,
        "defrost_end": pd.NaT,
        "cycle_duration": float((end - start).total_seconds()),
        "heating_duration": np.nan,
        "defrost_duration": np.nan,
        "segmentation_method": "explicit_defrost_flag_debounced",
        "segmentation_confidence": 0.5,
        "segmentation_evidence": "dataset_boundary",
        "corroboration_status": "not_applicable",
        "state_coverage": np.nan,
        "maximum_gap_seconds": np.nan,
        "gap_evidence": "dataset_boundary",
        "confidence_penalties": "dataset_boundary",
        "quality_flag": "partial",
        "exclusion_reason": reason,
    }


def _label_rows(frame: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.copy()
    labeled["cycle_id"] = pd.Series(pd.NA, index=labeled.index, dtype="string")
    labeled["cycle_quality"] = pd.Series(pd.NA, index=labeled.index, dtype="string")
    labeled["stage"] = pd.Series(pd.NA, index=labeled.index, dtype="string")
    labeled["cycle_time_s"] = np.nan
    labeled["cycle_phase"] = np.nan
    for row in cycles.itertuples(index=False):
        start = pd.Timestamp(row.heating_start)  # type: ignore[arg-type]
        end = (
            pd.Timestamp(row.defrost_end)  # type: ignore[arg-type]
            if pd.notna(row.defrost_end)
            else start + pd.Timedelta(seconds=float(row.cycle_duration))
        )
        mask = labeled["timestamp"].between(start, end, inclusive="left")
        labeled.loc[mask, "cycle_id"] = row.cycle_id
        labeled.loc[mask, "cycle_quality"] = row.quality_flag
        if row.quality_flag == "partial":
            labeled.loc[mask, "stage"] = "partial"
            continue
        stable_start = pd.Timestamp(row.stable_heating_start)  # type: ignore[arg-type]
        defrost_start = pd.Timestamp(row.defrost_start)  # type: ignore[arg-type]
        recovery = mask & labeled["timestamp"].lt(stable_start)
        development = (
            mask
            & labeled["timestamp"].ge(stable_start)
            & labeled["timestamp"].lt(defrost_start)
        )
        defrost = mask & labeled["timestamp"].ge(defrost_start)
        labeled.loc[recovery, "stage"] = "recovery"
        labeled.loc[development, "stage"] = "frost_development"
        labeled.loc[defrost, "stage"] = "defrost"
        development_times = pd.DatetimeIndex(labeled.loc[development, "timestamp"])
        labeled.loc[development, "cycle_time_s"] = (
            development_times - stable_start
        ).total_seconds()
        duration = float((defrost_start - stable_start).total_seconds())
        if duration > 0:
            labeled.loc[development, "cycle_phase"] = (
                labeled.loc[development, "cycle_time_s"] / duration
            )
    return labeled


def _cycle_columns() -> list[str]:
    return [
        "cycle_id",
        "heating_start",
        "stable_heating_start",
        "clean_start",
        "clean_end",
        "defrost_start",
        "defrost_end",
        "cycle_duration",
        "heating_duration",
        "defrost_duration",
        "segmentation_method",
        "segmentation_confidence",
        "segmentation_evidence",
        "corroboration_status",
        "state_coverage",
        "maximum_gap_seconds",
        "gap_evidence",
        "confidence_penalties",
        "quality_flag",
        "exclusion_reason",
    ]


def build_cycle_summary(
    cycles: pd.DataFrame,
    frame: pd.DataFrame,
    multiview_index: pd.DataFrame,
    *,
    date: str,
    gap_warning_factor: float,
) -> pd.DataFrame:
    """Create the single human-readable sensor/RGB cycle quality table."""
    groups = _prepare_multiview_groups(multiview_index)
    image_times = groups["group_time"].drop_duplicates().sort_values()
    image_deltas = image_times.diff().dt.total_seconds().dropna()
    image_cadence = (
        float(image_deltas[image_deltas.gt(0)].median())
        if image_deltas.gt(0).any()
        else 30.0
    )
    image_gap_limit = max(45.0, image_cadence * 1.5)
    sensor_times = pd.to_datetime(
        frame.get("timestamp", pd.Series(dtype=object)), errors="coerce"
    )
    # Only registered sensor values define sensor coverage; cycle labels are metadata.
    sensor_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "timestamp",
            "cycle_id",
            "cycle_quality",
            "stage",
            "cycle_time_s",
            "cycle_phase",
            "cycle_stage",
            "cycle_status",
            "cycle_elapsed_seconds",
            "cycle_progress",
            "operating_mode",
            "is_heating",
            "defrost_flag",
            "defrost_state_debounced",
        }
        and not str(column).startswith("image_")
    ]
    sensor_observed = pd.Series(False, index=frame.index)
    for column in sensor_columns:
        sensor_observed |= pd.to_numeric(frame[column], errors="coerce").notna()
    sensor_observation_times = sensor_times.loc[sensor_observed]
    all_sensor_deltas = (
        sensor_times.dropna().drop_duplicates().sort_values().diff().dt.total_seconds()
    )
    nominal_sensor = (
        float(all_sensor_deltas[all_sensor_deltas.gt(0)].median())
        if all_sensor_deltas.gt(0).any()
        else 1.0
    )
    rows: list[dict[str, object]] = []
    for _, cycle in cycles.iterrows():
        cycle_id = str(cycle.get("cycle_id", ""))
        start = _coerce_cycle_time(cycle.get("heating_start"))
        end = _coerce_cycle_time(cycle.get("defrost_end"))
        if end is None:
            mask = frame.get("cycle_id", pd.Series(dtype=object)).astype("string").eq(cycle_id)
            end = _coerce_cycle_time(sensor_times.loc[mask].max()) if mask.any() else None
        scoped = (
            groups.loc[groups["group_time"].between(start, end, inclusive="both")]
            if start is not None and end is not None
            else groups.iloc[0:0]
        )
        scoped_sensor = (
            sensor_times.loc[sensor_times.between(start, end, inclusive="both")]
            if start is not None and end is not None
            else sensor_times.iloc[0:0]
        ).dropna().drop_duplicates().sort_values()
        sensor_count = int(len(scoped_sensor))
        sensor_duration = (
            float((end - start).total_seconds())
            if start is not None and end is not None
            else 0.0
        )
        scoped_observed_times = (
            sensor_observation_times.loc[
                sensor_observation_times.between(start, end, inclusive="both")
            ]
            if start is not None and end is not None
            else sensor_observation_times.iloc[0:0]
        )
        sensor_coverage = _time_span_fraction(scoped_observed_times, start, end)
        sensor_expected = (
            max(1, int(round(sensor_duration / max(nominal_sensor, 1e-9))) + 1)
            if sensor_duration > 0
            else sensor_count
        )
        sensor_gap, sensor_intervals = _sensor_gap_evidence(
            frame, cycle_id, start, end, nominal_sensor
        )
        times = scoped["group_time"].drop_duplicates().sort_values().reset_index(drop=True)
        gaps = times.diff().dt.total_seconds().dropna()
        boundary_gaps = _cycle_boundary_gaps(times, start, end)
        all_gaps = [*gaps[gaps.gt(0)].tolist(), *boundary_gaps]
        interruptions = _cycle_image_gaps(times, start, end, image_gap_limit, boundary_gaps)
        partial = _partial_image_intervals(scoped)
        complete = scoped.get(
            "all_cameras_present", pd.Series(dtype=bool, index=scoped.index)
        ).fillna(False).astype(bool)
        group_count = int(len(scoped))
        complete_count = int(complete.sum())
        # RGB coverage uses complete multi-camera groups, not partial camera groups.
        complete_group_times = scoped.loc[complete, "group_time"]
        rgb_coverage = _time_span_fraction(complete_group_times, start, end)
        multimodal_coverage = min(sensor_coverage, rgb_coverage)
        rgb_quality = (
            "missing"
            if group_count == 0
            else "complete" if not interruptions and not partial else "incomplete"
        )
        sensor_quality = _cycle_status(str(cycle.get("quality_flag", "")))
        rows.append(
            {
                "cycle_id": cycle_id,
                "date": date,
                "cycle_status": sensor_quality,
                "cycle_status_reason": _text_or_empty(cycle.get("exclusion_reason", "")),
                "heating_start": cycle.get("heating_start"),
                "stable_heating_start": cycle.get("stable_heating_start"),
                "defrost_start": cycle.get("defrost_start"),
                "defrost_end": cycle.get("defrost_end"),
                "cycle_duration_seconds": cycle.get("cycle_duration"),
                "clean_start": cycle.get("clean_start"),
                "clean_end": cycle.get("clean_end"),
                "max_sensor_gap_seconds": max(
                    _as_float(cycle.get("maximum_gap_seconds")), sensor_gap
                ),
                "sensor_observation_count": sensor_count,
                "sensor_expected_count": sensor_expected,
                "sensor_coverage_fraction": sensor_coverage,
                "sensor_interruption_count": len(sensor_intervals),
                "sensor_interruption_intervals": "; ".join(sensor_intervals),
                "rgb_image_count": int(scoped.get("camera_count", pd.Series(dtype=float)).sum()),
                "rgb_group_count": group_count,
                "rgb_complete_group_count": complete_count,
                "rgb_complete_fraction": complete_count / group_count if group_count else np.nan,
                "rgb_coverage_fraction": rgb_coverage,
                "multimodal_coverage_fraction": multimodal_coverage,
                "rgb_max_gap_seconds": max(all_gaps) if all_gaps else np.nan,
                "rgb_interruption_count": len(interruptions),
                "rgb_partial_group_count": int((~complete).sum()),
                "rgb_interruption_intervals": "; ".join(interruptions),
                "rgb_partial_intervals": "; ".join(partial),
                "rgb_quality": rgb_quality,
                "multimodal_quality": _multimodal_status(sensor_quality, rgb_quality),
            }
        )
    columns = [
        "cycle_id",
        "date",
        "cycle_status",
        "cycle_status_reason",
        "heating_start",
        "stable_heating_start",
        "defrost_start",
        "defrost_end",
        "cycle_duration_seconds",
        "clean_start",
        "clean_end",
        "max_sensor_gap_seconds",
        "sensor_observation_count",
        "sensor_expected_count",
        "sensor_coverage_fraction",
        "sensor_interruption_count",
        "sensor_interruption_intervals",
        "rgb_image_count",
        "rgb_group_count",
        "rgb_complete_group_count",
        "rgb_complete_fraction",
        "rgb_coverage_fraction",
        "multimodal_coverage_fraction",
        "rgb_max_gap_seconds",
        "rgb_interruption_count",
        "rgb_partial_group_count",
        "rgb_interruption_intervals",
        "rgb_partial_intervals",
        "rgb_quality",
        "multimodal_quality",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values("cycle_id", kind="stable")


def _enforce_heating_mode(
    frame: pd.DataFrame, cycles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Invalidate complete cycles containing a non-heating normal-stage row."""
    if "is_heating" not in frame or cycles.empty:
        return frame, cycles, []
    labeled = frame.copy()
    validated = cycles.copy()
    warnings: list[str] = []
    for index, cycle in validated.iterrows():
        if cycle.get("quality_flag") != "complete":
            continue
        cycle_mask = labeled["cycle_id"].eq(cycle["cycle_id"])
        normal_mask = labeled["stage"].isin(["stable_clean", "frost_development"])
        observed = labeled.loc[cycle_mask & normal_mask, "is_heating"].dropna()
        if not observed.empty and not observed.astype("boolean").all():
            validated.loc[index, "quality_flag"] = "abnormal"
            validated.loc[index, "exclusion_reason"] = append_issue(
                cycle.get("exclusion_reason"),
                "nonheating_mode_inside_cycle",
            )
            labeled.loc[cycle_mask, "cycle_quality"] = "abnormal"
            warnings.append(f"{cycle['cycle_id']}:nonheating_mode_inside_cycle")
    return labeled, validated, warnings


def _mark_long_gap_cycles(
    frame: pd.DataFrame,
    cycles: pd.DataFrame,
    *,
    nominal_seconds: float,
    factor: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Mark complete cycles whose observed channel gap exceeds the threshold."""
    labeled = frame.copy()
    validated = cycles.copy()
    warnings: list[str] = []
    limit = nominal_seconds * factor
    for index, cycle in validated.iterrows():
        cycle_id = str(cycle.get("cycle_id", ""))
        gap = as_optional_float(cycle.get("maximum_gap_seconds"))
        if pd.notna(cycle.get("heating_start")) and pd.notna(cycle.get("defrost_end")):
            channel_gap, _ = _sensor_gap_evidence(
                labeled,
                cycle_id,
                pd.Timestamp(cycle["heating_start"]),
                pd.Timestamp(cycle["defrost_end"]),
                nominal_seconds,
            )
            candidates = [
                value
                for value in (gap, channel_gap)
                if value is not None and np.isfinite(value)
            ]
            gap = max(candidates, default=0.0)
            validated.loc[index, "maximum_gap_seconds"] = gap
        if cycle.get("quality_flag") != "complete" or gap is None or gap <= limit:
            continue
        validated.loc[index, "quality_flag"] = "contaminated"
        validated.loc[index, "exclusion_reason"] = append_issue(
            cycle.get("exclusion_reason"),
            "long_gap",
        )
        cycle_mask = labeled["cycle_id"].eq(cycle_id)
        labeled.loc[cycle_mask, "cycle_quality"] = "contaminated"
        labeled.loc[cycle_mask, "cycle_phase"] = np.nan
        warnings.append(f"{cycle_id}:long_gap:{gap:.1f}s>{limit:.1f}s")
    return labeled, validated, warnings


def as_optional_float(value: object) -> float | None:
    """Convert one scalar to float while preserving missingness as ``None``."""
    number = pd.to_numeric(cast(Any, value), errors="coerce")
    if pd.isna(number):
        return None
    return float(number)


def _optional_positive_float(value: object) -> float | None:
    """Return a positive numeric option or ``None`` when it is not supplied."""
    number = as_optional_float(value)
    return number if number is not None and number > 0 else None


def _time_span_fraction(
    observed_times: pd.Series,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> float:
    """Measure observed time span against one cycle interval."""
    if start is None or end is None:
        return 0.0
    total_seconds = float((end - start).total_seconds())
    if total_seconds <= 0:
        return 0.0
    valid = pd.to_datetime(observed_times, errors="coerce").dropna().drop_duplicates()
    if valid.empty:
        return 0.0
    observed_seconds = float((valid.max() - valid.min()).total_seconds())
    return float(np.clip(observed_seconds / total_seconds, 0.0, 1.0))


def _sensor_gap_evidence(
    frame: pd.DataFrame,
    cycle_id: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    nominal: float,
) -> tuple[float, list[str]]:
    """Find gaps per channel, so one active channel cannot hide another's outage."""
    if start is None or end is None:
        return np.nan, []
    cycle_mask = frame.get("cycle_id", pd.Series(dtype=object)).astype("string").eq(cycle_id)
    scoped = frame.loc[cycle_mask & frame["timestamp"].between(start, end, inclusive="both")]
    reserved = {
        "timestamp",
        "cycle_id",
        "cycle_stage",
        "cycle_status",
        "cycle_elapsed_seconds",
        "cycle_progress",
        "is_heating",
        "operating_mode",
        "defrost_flag",
    }
    threshold = max(3.0 * nominal, 30.0)
    maximum = 0.0
    intervals: set[str] = set()
    for column in scoped.columns:
        if column in reserved or str(column).startswith("image_"):
            continue
        values = pd.to_numeric(scoped[column], errors="coerce")
        times = (
            scoped.loc[values.notna(), "timestamp"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )
        if len(times) < 2:
            continue
        deltas = times.diff().dt.total_seconds()
        positive = deltas[deltas.gt(0)]
        if positive.empty:
            continue
        maximum = max(maximum, float(positive.max()))
        for position, gap in enumerate(deltas):
            if position > 0 and float(gap) >= threshold:
                intervals.add(
                    _format_cycle_interval(times.iloc[position - 1], times.iloc[position])
                )
    return maximum, sorted(intervals)


def _as_float(value: object) -> float:
    number = as_optional_float(value)
    return number if number is not None else 0.0


def _prepare_multiview_groups(multiview: pd.DataFrame) -> pd.DataFrame:
    if multiview.empty or "group_time" not in multiview:
        return pd.DataFrame(
            {
                "group_time": pd.Series(dtype="datetime64[ns]"),
                "camera_count": pd.Series(dtype=float),
                "all_cameras_present": pd.Series(dtype=bool),
            }
        )
    result = multiview.copy()
    result["group_time"] = pd.to_datetime(result["group_time"], errors="coerce")
    return result.loc[result["group_time"].notna()].sort_values("group_time")


def _coerce_cycle_time(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(str(value), errors="coerce")
    return parsed if isinstance(parsed, pd.Timestamp) else None


def _cycle_status(quality: str) -> str:
    return normalize_cycle_status(quality)


def _cycle_boundary_gaps(
    times: pd.Series, start: pd.Timestamp | None, end: pd.Timestamp | None
) -> list[float]:
    if times.empty:
        return []
    result: list[float] = []
    if start is not None:
        result.append(float((times.iloc[0] - start).total_seconds()))
    if end is not None:
        result.append(float((end - times.iloc[-1]).total_seconds()))
    return result


def _cycle_image_gaps(
    times: pd.Series,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    threshold: float,
    boundaries: list[float],
) -> list[str]:
    if times.empty:
        return [_format_cycle_interval(start, end)] if start is not None and end is not None else []
    result: list[str] = []
    if boundaries and boundaries[0] >= threshold and start is not None:
        result.append(_format_cycle_interval(start, times.iloc[0]))
    gaps = times.diff().dt.total_seconds().dropna()
    for previous, current, gap in zip(times.iloc[:-1], times.iloc[1:], gaps, strict=False):
        if float(gap) >= threshold:
            result.append(_format_cycle_interval(previous, current))
    if len(boundaries) > 1 and boundaries[1] >= threshold and end is not None:
        result.append(_format_cycle_interval(times.iloc[-1], end))
    return result


def _partial_image_intervals(scoped: pd.DataFrame) -> list[str]:
    if scoped.empty or "all_cameras_present" not in scoped:
        return []
    partial = ~scoped["all_cameras_present"].fillna(False).astype(bool)
    if not partial.any():
        return []
    values = scoped.loc[:, ["group_time"]].copy()
    values["partial"] = partial.to_numpy()
    values["run"] = values["partial"].ne(values["partial"].shift()).cumsum()
    return [
        _format_cycle_interval(group["group_time"].iloc[0], group["group_time"].iloc[-1])
        for _, group in values.loc[values["partial"]].groupby("run")
    ]


def _format_cycle_interval(start: object, end: object) -> str:
    start_text = pd.Timestamp(str(start)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    end_text = pd.Timestamp(str(end)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return f"{start_text} -> {end_text}"


def _multimodal_status(sensor: str, rgb: str) -> str:
    if sensor == "valid" and rgb == "complete":
        return "complete_multimodal"
    if sensor == "valid":
        return "sensor_valid_rgb_incomplete"
    if rgb == "complete":
        return "sensor_incomplete_rgb_complete"
    return "sensor_and_rgb_incomplete"
