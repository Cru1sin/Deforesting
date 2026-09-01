"""V2.6.8-only Dataset cohorts, raw windows, and causal state features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from .boundaries import catalog_exclusion_reason, clean_anchor_exclusion_reason
from .fit_v2_6_8 import DYNAMIC_8

QUALITY_COVERAGE = 0.95
MAXIMUM_GAP_SECONDS = 30.0
PHASE_INTERVAL_CONVENTION = "[start,end)"
INTEGRAL_SAMPLING_CONVENTION = (
    "raw_observations_in_[start,end);_trapezoids_between_adjacent_retained_samples;"
    "right_boundary_sample_excluded;last_left_observation_zero_order_hold_to_end;"
    "hold_limited_by_robust_observation_cadence"
)
RAW_COLUMNS = (
    "timestamp",
    "power_total",
    "water_flow",
    "water_in_temperature",
    "water_out_temperature",
    "coil_temperature",
    "evaporating_pressure",
    "water_temperature_setpoint",
    "ambient_temperature",
    "compressor_frequency",
)


def timestamp(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(str(value), errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def catalog(loader: Any, *, valid_only: bool = False) -> pd.DataFrame:
    values = loader.list_cycles(statuses={"valid"} if valid_only else None).copy()
    for column in (
        "start_time",
        "heating_start",
        "stable_heating_start",
        "defrost_preparation_start",
        "defrost_start",
        "defrost_end",
    ):
        if column in values:
            values[column] = pd.to_datetime(values[column], errors="coerce")
    return cast(
        pd.DataFrame,
        values.sort_values(["experiment_id", "start_time"], kind="stable").reset_index(drop=True),
    )


def load_frame(loader: Any, cycle_name: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if cycle_name in cache:
        return cache[cycle_name]
    frame = loader.load_cycle_original(cycle_name, columns=list(RAW_COLUMNS)).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in frame.columns.drop("timestamp"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    cache[cycle_name] = frame.drop_duplicates("timestamp", keep="last")
    return cache[cycle_name]


def sorted_time_slice(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, *, end_inclusive: bool = True
) -> pd.DataFrame:
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    left = int(timestamps.searchsorted(start, side="left"))
    right = int(timestamps.searchsorted(end, side="right" if end_inclusive else "left"))
    return frame.iloc[left:right]


def water_heat(frame: pd.DataFrame) -> pd.Series:
    return (
        1.161
        * pd.to_numeric(frame["water_flow"], errors="coerce")
        * (
            pd.to_numeric(frame["water_out_temperature"], errors="coerce")
            - pd.to_numeric(frame["water_in_temperature"], errors="coerce")
        )
    )


def window_audit(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, column: str
) -> dict[str, object]:
    """Integrate adjacent raw samples in a strict half-open declared window."""
    values = sorted_time_slice(frame, start, end, end_inclusive=False).copy()
    values = values.sort_values("timestamp", kind="stable").drop_duplicates("timestamp")
    signal = (
        water_heat(values)
        if column == "water_heat"
        else pd.to_numeric(values[column], errors="coerce")
    )
    observed = pd.DataFrame({"timestamp": values["timestamp"], "value": signal}).dropna()
    dt = observed["timestamp"].diff().dt.total_seconds()
    short = dt.gt(0) & dt.le(MAXIMUM_GAP_SECONDS)
    increments = ((observed["value"] + observed["value"].shift()) / 2 * dt / 3600).where(short, 0.0)
    short_steps = dt.loc[short]
    cadence = float(short_steps.median()) if not short_steps.empty else np.nan
    last_row = observed.iloc[-1] if not observed.empty else None
    trailing = (
        max((end - pd.Timestamp(last_row["timestamp"])).total_seconds(), 0.0)
        if last_row is not None
        else 0.0
    )
    hold = min(trailing, cadence) if np.isfinite(cadence) and cadence > 0 else 0.0
    energy = float(increments.sum())
    if last_row is not None:
        energy += float(last_row["value"]) * hold / 3600
    duration = max((end - start).total_seconds(), 0.0)
    covered = float(dt.where(short, 0.0).sum() + hold)
    gaps = dt.dropna()
    first = observed["timestamp"].min() if not observed.empty else pd.NaT
    leading = (
        max((pd.Timestamp(first) - start).total_seconds(), 0.0) if pd.notna(first) else float("inf")
    )
    maximum_gap = (
        float(max(gaps.max(), leading, trailing))
        if not gaps.empty
        else float(max(leading, trailing))
        if last_row is not None
        else float("inf")
    )
    last = observed["timestamp"].max() if not observed.empty else pd.NaT
    start_fresh = pd.notna(first) and abs((first - start).total_seconds()) <= 30
    end_fresh = pd.notna(last) and abs((end - last).total_seconds()) <= 30
    coverage = covered / duration if duration > 0 else 0.0
    valid = bool(
        np.isfinite(energy)
        and coverage >= QUALITY_COVERAGE
        and maximum_gap <= MAXIMUM_GAP_SECONDS
        and start_fresh
        and end_fresh
    )
    return {
        "energy": float(energy),
        "coverage": min(float(coverage), 1.0),
        "maximum_gap_seconds": maximum_gap,
        "integral_sampling_convention": INTEGRAL_SAMPLING_CONVENTION,
        "start_fresh": bool(start_fresh),
        "end_fresh": bool(end_fresh),
        "valid": valid,
    }


def candidate_integral_table(
    frame: pd.DataFrame, start: pd.Timestamp, candidates: Sequence[pd.Timestamp], column: str
) -> pd.DataFrame:
    return pd.DataFrame([window_audit(frame, start, end, column) for end in candidates])


def pre_action_features(
    frame: pd.DataFrame, candidates: Sequence[pd.Timestamp], heating_start: pd.Timestamp
) -> pd.DataFrame:
    """Extract state medians and Pe slope from samples strictly before tau."""
    state_names = (
        "water_in_temperature",
        "water_out_temperature",
        "coil_temperature",
        "evaporating_pressure",
        "water_temperature_setpoint",
        "ambient_temperature",
        "compressor_frequency",
    )
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    seconds: np.ndarray = timestamps.view("i8") // 1_000_000_000
    arrays = {
        name: pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        for name in state_names
    }
    rows: list[dict[str, float | bool]] = []
    for candidate in pd.DatetimeIndex(candidates):
        right = int(timestamps.searchsorted(candidate, side="left"))
        left = int(timestamps.searchsorted(candidate - pd.Timedelta(seconds=60), side="left"))
        slope_left = int(timestamps.searchsorted(candidate - pd.Timedelta(minutes=5), side="left"))
        row: dict[str, float | bool] = {}
        counts: dict[str, int] = {}
        for name in state_names:
            values = arrays[name][left:right]
            finite = np.isfinite(values)
            row[name] = float(np.median(values[finite])) if finite.any() else np.nan
            counts[name] = int(np.unique(seconds[left:right][finite]).size)
            row[f"{name}_valid_second_count"] = counts[name]
        row["mean_water_temperature"] = (
            float(row["water_in_temperature"]) + float(row["water_out_temperature"])
        ) / 2
        row["setpoint_outlet_difference"] = float(row["water_temperature_setpoint"]) - float(
            row["water_out_temperature"]
        )
        row["ambient_coil_difference"] = float(row["ambient_temperature"]) - float(
            row["coil_temperature"]
        )
        row["heating_elapsed_minutes"] = (candidate - heating_start).total_seconds() / 60
        pressure = arrays["evaporating_pressure"][slope_left:right]
        finite = np.isfinite(pressure)
        slope_seconds = int(np.unique(seconds[slope_left:right][finite]).size)
        slope = np.nan
        if slope_seconds >= 240 and finite.sum() >= 2:
            elapsed = (timestamps.view("i8")[slope_left:right][finite] - candidate.value) / 60e9
            centered = elapsed - elapsed.mean()
            denominator = float(np.square(centered).sum())
            if denominator > 0:
                slope = float(
                    (centered * (pressure[finite] - pressure[finite].mean())).sum() / denominator
                )
        row["evaporating_pressure_slope_5m"] = slope
        row["evaporating_pressure_slope_valid_second_count"] = slope_seconds
        row["pre_action_window_valid"] = bool(
            all(counts[name] >= 48 for name in state_names)
            and slope_seconds >= 240
            and np.isfinite([row[name] for name in DYNAMIC_8]).all()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_candidate_boundaries(
    cycle_name: str,
    experiment_id: str,
    heating_start: pd.Timestamp,
    preparation_start: pd.Timestamp,
) -> pd.DataFrame:
    first = heating_start + pd.Timedelta(minutes=10)
    if preparation_start < first:
        raise ValueError(f"candidate interval is empty for {cycle_name}")
    candidates = list(pd.date_range(first, preparation_start, freq="min"))
    if not candidates or candidates[-1] != preparation_start:
        candidates.append(preparation_start)
    return pd.DataFrame(
        {
            "cycle_name": cycle_name,
            "experiment_id": experiment_id,
            "candidate_time": candidates,
            "candidate_elapsed_minutes": [
                (value - heating_start).total_seconds() / 60 for value in candidates
            ],
            "integration_start": heating_start + pd.Timedelta(minutes=9),
            "integration_start_rule": "fixed_post_defrost_9min",
            "heating_start": heating_start,
            "actual_preparation_time": preparation_start,
        }
    )


def event_outcomes(
    current: pd.DataFrame,
    recovery: pd.DataFrame,
    *,
    preparation_start: pd.Timestamp,
    defrost_start: pd.Timestamp,
    defrost_end: pd.Timestamp,
    recovery_end: pd.Timestamp,
) -> dict[str, object]:
    """Observe preparation, defrost, and fixed recovery targets without clipping Q."""
    windows = {
        "prep": (current, preparation_start, defrost_start),
        "D": (current, defrost_start, defrost_end),
        "R": (recovery, defrost_end, recovery_end),
    }
    result: dict[str, object] = {}
    for phase, (frame, start, end) in windows.items():
        for quantity, column in (("E", "power_total"), ("Q", "water_heat")):
            audit = window_audit(frame, start, end, column)
            result[f"{quantity}_{phase}_kwh"] = audit["energy"]
            for field in ("coverage", "maximum_gap_seconds", "start_fresh", "end_fresh", "valid"):
                result[f"{quantity}_{phase}_{field}"] = audit[field]
    partition = preparation_start < defrost_start < defrost_end < recovery_end
    result["E_T_observed_kwh"] = float(
        np.sum([float(result[f"E_{phase}_kwh"]) for phase in windows])  # type: ignore[arg-type]
    )
    result["Q_T_observed_kwh"] = float(
        np.sum([float(result[f"Q_{phase}_kwh"]) for phase in windows])  # type: ignore[arg-type]
    )
    result["event_duration_minutes"] = (recovery_end - preparation_start).total_seconds() / 60
    result["phase_partition_valid"] = bool(partition)
    result["phase_interval_convention"] = PHASE_INTERVAL_CONVENTION
    result["integral_sampling_convention"] = INTEGRAL_SAMPLING_CONVENTION
    result["event_valid"] = bool(
        partition
        and all(
            bool(result[f"{quantity}_{phase}_valid"])
            for quantity in ("E", "Q")
            for phase in windows
        )
    )
    return result


def build_event_table(loader: Any) -> pd.DataFrame:  # noqa: C901
    """Retain every real defrost record, including incomplete exclusions."""
    values = catalog(loader)
    next_rows: dict[str, pd.Series] = {}
    for _, experiment in values.groupby("experiment_id", sort=False):
        ordered = experiment.sort_values("start_time", kind="stable").reset_index(drop=True)
        for index in range(len(ordered) - 1):
            next_rows[str(ordered.loc[index, "cycle_name"])] = ordered.iloc[index + 1]
    cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    real = values.loc[
        values["defrost_preparation_start"].notna()
        | (
            values["status"].eq("valid")
            & values["defrost_start"].notna()
            & values["defrost_end"].notna()
        )
    ]
    boundary_names = ("heating_start", "defrost_preparation_start", "defrost_start", "defrost_end")
    for _, record in real.iterrows():
        name = str(record["cycle_name"])
        row: dict[str, object] = {
            "event_id": name,
            "cycle_name": name,
            "experiment_id": str(record["experiment_id"]),
            **{boundary: timestamp(record.get(boundary)) for boundary in boundary_names},
        }
        missing = [boundary for boundary in boundary_names if row[boundary] is None]
        following = next_rows.get(name)
        reasons = [f"missing_{boundary}" for boundary in missing]
        if following is None:
            reasons.append("missing_following_cycle")
        elif str(following["experiment_id"]) != row["experiment_id"]:
            reasons.append("following_cycle_experiment_mismatch")
        elif isinstance(row["defrost_end"], pd.Timestamp):
            recorded_end = row["defrost_end"]
            next_start = timestamp(following.get("heating_start"))
            if next_start is None or abs((next_start - recorded_end).total_seconds()) > 60:
                reasons.append("following_cycle_not_adjacent")
        if reasons:
            row.update(event_valid=False, event_invalid_reason=";".join(reasons))
            rows.append(row)
            continue
        heating, preparation = row["heating_start"], row["defrost_preparation_start"]
        defrost, defrost_end = row["defrost_start"], row["defrost_end"]
        assert isinstance(heating, pd.Timestamp) and isinstance(preparation, pd.Timestamp)
        assert isinstance(defrost, pd.Timestamp) and isinstance(defrost_end, pd.Timestamp)
        assert following is not None
        current = load_frame(loader, name, cache)
        recovery = load_frame(loader, str(following["cycle_name"]), cache)
        observed = event_outcomes(
            current,
            recovery,
            preparation_start=preparation,
            defrost_start=defrost,
            defrost_end=defrost_end,
            recovery_end=defrost_end + pd.Timedelta(minutes=9),
        )
        features = cast(
            dict[str, object],
            pre_action_features(current, [preparation], heating).iloc[0].to_dict(),
        )
        audit_reasons: list[str] = []
        if not observed["phase_partition_valid"]:
            audit_reasons.append("phase_partition")
        if not observed["event_valid"]:
            for quantity in ("E", "Q"):
                for phase in ("prep", "D", "R"):
                    prefix = f"{quantity}_{phase}"
                    if float(observed[f"{prefix}_coverage"]) < QUALITY_COVERAGE:  # type: ignore[arg-type]
                        audit_reasons.append(f"{prefix}_coverage")
                    if float(observed[f"{prefix}_maximum_gap_seconds"]) > MAXIMUM_GAP_SECONDS:  # type: ignore[arg-type]
                        audit_reasons.append(f"{prefix}_continuous_gap")
                    if not bool(observed[f"{prefix}_start_fresh"]):
                        audit_reasons.append(f"{prefix}_start_boundary")
                    if not bool(observed[f"{prefix}_end_fresh"]):
                        audit_reasons.append(f"{prefix}_end_boundary")
        row.update(
            next_cycle_name=str(following["cycle_name"]),
            recovery_end_fixed9=defrost_end + pd.Timedelta(minutes=9),
            recovery_duration_minutes=9.0,
            **features,
            **observed,
        )
        row["event_valid"] = not audit_reasons
        row["event_invalid_reason"] = ";".join(audit_reasons)
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_cohort(loader: Any, parameter_experiments: set[str]) -> tuple[list[str], int]:
    """Apply catalog metadata and raw clean-anchor gates without external tables."""
    selected: list[str] = []
    rows = 0
    values = catalog(loader, valid_only=True)
    for _, record in values.iterrows():
        record_dict = cast(dict[str, object], record.to_dict())
        if catalog_exclusion_reason(record_dict, parameter_experiments) is not None:
            continue
        name = str(record["cycle_name"])
        frame = loader.load_cycle_original(
            name,
            columns=[
                "timestamp",
                "water_flow",
                "water_in_temperature",
                "water_out_temperature",
                "power_total",
            ],
        )
        anchor_record = cast(dict[str, object], record.to_dict())
        heating_anchor = timestamp(record["heating_start"])
        if heating_anchor is not None:
            anchor_record["stable_heating_start"] = heating_anchor + pd.Timedelta(minutes=9)
        if clean_anchor_exclusion_reason(frame, anchor_record) is not None:
            continue
        heating = timestamp(record["heating_start"])
        preparation = timestamp(record["defrost_preparation_start"])
        assert heating is not None and preparation is not None
        selected.append(name)
        rows += len(
            build_candidate_boundaries(name, str(record["experiment_id"]), heating, preparation)
        )
    return selected, rows
