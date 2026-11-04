"""Dataset measurements, complete defrost events, and causal state features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from heat_smoothing.methods import PROCESSED_CHANNELS

from .event_electricity_ridge import THREE_INPUTS

QUALITY_COVERAGE = 0.95
MAXIMUM_GAP_SECONDS = 30.0
PHASE_INTERVAL_CONVENTION = "[start,end)"
INTEGRAL_SAMPLING_CONVENTION = (
    "raw_observations_in_[start,end);_trapezoids_between_adjacent_retained_samples;"
    "right_boundary_sample_excluded;last_left_observation_zero_order_hold_to_end;"
    "hold_limited_by_robust_observation_cadence"
)
OFFLINE_INTEGRAL_SAMPLING_CONVENTION = (
    "offline_interpolated_observations;linear_window_boundaries;"
    "trapezoids_between_samples;terminal_half_open_bin_hold_limited_by_observation_cadence"
)
RAW_COLUMNS = (
    "timestamp",
    "power_total",
    "compressor_power",
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


def load_frame(
    loader: Any, cycle_name: str, cache: dict[str, pd.DataFrame], *, extra_columns=()
) -> pd.DataFrame:
    if cycle_name in cache:
        return cache[cycle_name]
    columns = (
        list(RAW_COLUMNS)
        + list(extra_columns)
        + [name + "_processed" for name in PROCESSED_CHANNELS]
    )
    frame = loader.load_cycle_measurements(cycle_name)
    processing = frame.attrs.get("measurement_processing")
    frame = frame.reindex(columns=columns).copy()
    if processing:
        frame.attrs["measurement_processing"] = processing
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


def _interpolate_boundaries(frame, boundaries, column):
    """Evaluate offline window boundaries from the same published linear signals."""
    values = frame.set_index("timestamp")
    values = values.reindex(
        values.index.union(pd.DatetimeIndex(boundaries)).sort_values().rename("timestamp")
    ).copy()
    if column in values:
        values[column] = pd.to_numeric(values[column], errors="coerce").interpolate(
            method="time", limit_area="inside"
        )
    result = values.reset_index()
    result.attrs.update(frame.attrs)
    return result


def _window_segments(frame, start, end, column):
    """Shared segment construction for single-window and batched integrals."""
    offline = frame.attrs.get("measurement_processing") == "dataset_offline_linear_60s"
    if offline:
        times = pd.DatetimeIndex(frame.timestamp)
        left = max(int(times.searchsorted(start, side="left")) - 1, 0)
        right = min(int(times.searchsorted(end, side="right")) + 1, len(frame))
        values = _interpolate_boundaries(frame.iloc[left:right], [start, end], column)
        values = sorted_time_slice(values, start, end, end_inclusive=True)
    else:
        values = sorted_time_slice(frame, start, end, end_inclusive=False).copy()
    values = values.sort_values("timestamp", kind="stable").drop_duplicates("timestamp")
    signal = pd.to_numeric(values[column], errors="coerce")
    observed = pd.DataFrame({"timestamp": values["timestamp"], "value": signal})
    observed = observed.dropna()
    dt = observed["timestamp"].diff().dt.total_seconds()
    short = dt.gt(0) if offline else dt.gt(0) & dt.le(MAXIMUM_GAP_SECONDS)
    increments = ((observed["value"] + observed["value"].shift()) / 2 * dt / 3600).where(short, 0.0)
    return observed, dt, short, increments


def window_audit(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    column: str,
) -> dict[str, object]:
    """Integrate a half-open window using its declared measurement processing."""
    offline = frame.attrs.get("measurement_processing") == "dataset_offline_linear_60s"
    convention = OFFLINE_INTEGRAL_SAMPLING_CONVENTION if offline else INTEGRAL_SAMPLING_CONVENTION
    if start == end:
        return {
            "energy": 0.0,
            "coverage": 1.0,
            "maximum_gap_seconds": 0.0,
            "integral_sampling_convention": convention,
            "start_fresh": True,
            "end_fresh": True,
            "valid": True,
        }
    observed, dt, short, increments = _window_segments(frame, start, end, column)
    short_steps = dt.loc[short]
    cadence = float(short_steps.median()) if not short_steps.empty else np.nan
    last_row = observed.iloc[-1] if not observed.empty else None
    trailing = (
        max((end - pd.Timestamp(last_row["timestamp"])).total_seconds(), 0.0)
        if last_row is not None
        else 0.0
    )
    offline = frame.attrs.get("measurement_processing") == "dataset_offline_linear_60s"
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
    start_fresh = pd.notna(first) and abs((first - start).total_seconds()) <= (0 if offline else 30)
    end_fresh = pd.notna(last) and abs((end - last).total_seconds()) <= (
        cadence if offline and np.isfinite(cadence) else 30 if not offline else 0
    )
    coverage = covered / duration if duration > 0 else 0.0
    valid = bool(
        np.isfinite(energy)
        and coverage >= QUALITY_COVERAGE
        and (offline or maximum_gap <= MAXIMUM_GAP_SECONDS)
        and start_fresh
        and end_fresh
    )
    return {
        "energy": float(energy),
        "coverage": min(float(coverage), 1.0),
        "maximum_gap_seconds": maximum_gap,
        "integral_sampling_convention": convention,
        "start_fresh": bool(start_fresh),
        "end_fresh": bool(end_fresh),
        "valid": valid,
    }


def candidate_integral_table(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    candidates: Sequence[pd.Timestamp],
    column: str,
) -> pd.DataFrame:
    # All candidates share a start: compute each observed segment once. The
    # search excludes the candidate timestamp, preserving half-open causality.
    ends = pd.DatetimeIndex(candidates)
    if not len(ends):
        return pd.DataFrame()
    offline = frame.attrs.get("measurement_processing") == "dataset_offline_linear_60s"
    if offline:
        # Include every query boundary so batch and single-window integrals agree.
        frame = _interpolate_boundaries(frame, ends, column)
    observed, dt, short, increments = _window_segments(frame, start, ends.max(), column)
    duration = np.maximum((ends - start).total_seconds().to_numpy(), 0.0)
    energy, coverage = np.zeros(len(ends)), np.zeros(len(ends))
    maximum_gap = np.full(len(ends), np.inf)
    start_fresh, end_fresh = np.zeros(len(ends), bool), np.zeros(len(ends), bool)
    if len(observed):
        times = pd.DatetimeIndex(observed.timestamp)
        positions = times.searchsorted(ends, side="right" if offline else "left") - 1
        available = positions >= 0
        indices = positions[available]
        cadence = dt.where(short).expanding().median().to_numpy()
        trailing = (ends[available] - times[indices]).total_seconds().to_numpy()
        hold = np.where(np.isfinite(cadence[indices]), np.minimum(trailing, cadence[indices]), 0.0)
        energy[available] = (
            increments.fillna(0).cumsum().to_numpy()[indices]
            + observed.value.to_numpy()[indices] * hold / 3600
        )
        covered = dt.where(short, 0.0).cumsum().to_numpy()[indices] + hold
        coverage[available] = np.divide(
            covered, duration[available], out=np.zeros(len(indices)), where=duration[available] > 0
        )
        leading = max((times[0] - start).total_seconds(), 0.0)
        gaps = dt.fillna(0).cummax().to_numpy()[indices]
        maximum_gap[available] = np.maximum(np.maximum(gaps, leading), trailing)
        start_fresh[available] = abs((times[0] - start).total_seconds()) <= (0 if offline else 30)
        end_fresh[available] = trailing <= (np.nan_to_num(cadence[indices]) if offline else 30)
    valid = (
        np.isfinite(energy)
        & (coverage >= QUALITY_COVERAGE)
        & (offline | (maximum_gap <= MAXIMUM_GAP_SECONDS))
        & start_fresh
        & end_fresh
    )
    zero = ends == start
    energy[zero], coverage[zero], maximum_gap[zero] = 0.0, 1.0, 0.0
    start_fresh[zero], end_fresh[zero], valid[zero] = True, True, True
    return pd.DataFrame(
        dict(
            energy=energy,
            coverage=np.minimum(coverage, 1.0),
            maximum_gap_seconds=maximum_gap,
            integral_sampling_convention=(
                OFFLINE_INTEGRAL_SAMPLING_CONVENTION if offline else INTEGRAL_SAMPLING_CONVENTION
            ),
            start_fresh=start_fresh,
            end_fresh=end_fresh,
            valid=valid,
        )
    )


def extract_pre_defrost_features(
    frame: pd.DataFrame,
    candidates: Sequence[pd.Timestamp],
    heating_start: pd.Timestamp,
) -> pd.DataFrame:
    """Three model inputs strictly before each action, with 60-second coverage.

    Ambient temperature uses the preceding-window median. Frequency and the
    stage-smoothed coil temperature use the latest observation within 30 seconds.
    """
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    seconds = timestamps.view("i8") // 1_000_000_000
    arrays = {
        name: pd.to_numeric(
            frame["coil_temperature_processed"] if name == "coil_temperature" else frame[name],
            errors="coerce",
        ).to_numpy(dtype=float)
        for name in THREE_INPUTS
    }
    rows = []
    for candidate in pd.DatetimeIndex(candidates):
        right = int(timestamps.searchsorted(candidate, side="left"))
        left = int(timestamps.searchsorted(candidate - pd.Timedelta(seconds=60), side="left"))
        row = {}
        for name in THREE_INPUTS:
            values = arrays[name][left:right]
            finite = np.isfinite(values)
            row[f"{name}_valid_second_count"] = int(np.unique(seconds[left:right][finite]).size)
            if name == "ambient_temperature":
                row[name] = float(np.median(values[finite])) if finite.any() else np.nan
                continue
            valid = np.flatnonzero(finite)
            latest = left + int(valid[-1]) if valid.size else None
            row[name] = (
                float(arrays[name][latest])
                if latest is not None
                and timestamps[latest] >= heating_start
                and (candidate - timestamps[latest]).total_seconds() <= MAXIMUM_GAP_SECONDS
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def measure_defrost_event_quantities(
    current: pd.DataFrame,
    *,
    preparation_start: pd.Timestamp,
    defrost_start: pd.Timestamp,
    defrost_end: pd.Timestamp,
) -> dict[str, object]:
    """Measure preparation and defrost electricity on their half-open windows."""
    result: dict[str, object] = {}
    for phase, start, end in (
        ("prep", preparation_start, defrost_start),
        ("D", defrost_start, defrost_end),
    ):
        audit = window_audit(current, start, end, "power_total")
        result[f"E_{phase}_kwh"] = audit["energy"]
        for field in ("coverage", "maximum_gap_seconds", "start_fresh", "end_fresh", "valid"):
            result[f"E_{phase}_{field}"] = audit[field]
    partition = preparation_start <= defrost_start < defrost_end
    result["defrost_event_electricity_observed_kwh"] = result["E_prep_kwh"] + result["E_D_kwh"]
    result["defrost_event_duration_observed_minutes"] = (
        defrost_end - preparation_start
    ).total_seconds() / 60
    result["event_duration_minutes"] = result["defrost_event_duration_observed_minutes"]
    result["phase_partition_valid"] = bool(partition)
    result["phase_interval_convention"] = PHASE_INTERVAL_CONVENTION
    result["integral_sampling_convention"] = (
        OFFLINE_INTEGRAL_SAMPLING_CONVENTION
        if current.attrs.get("measurement_processing") == "dataset_offline_linear_60s"
        else INTEGRAL_SAMPLING_CONVENTION
    )
    result["energy_event_valid"] = partition and all(
        result[f"E_{phase}_valid"] for phase in ("prep", "D")
    )
    result["duration_event_valid"] = bool(partition)
    result["event_valid"] = result["energy_event_valid"]
    return result


def observed_defrost_records(values):
    """Observed event entries shared by Ridge and event-pair diagnostics."""
    return values.loc[
        (
            values["defrost_preparation_start"].notna()
            | (values["defrost_start"].notna() & values["defrost_end"].notna())
        )
    ]


def build_defrost_event_training_table(
    loader: Any,
) -> pd.DataFrame:  # noqa: C901
    """Extract observed events independently of cycle quality; audit each event window."""
    values = catalog(loader)
    cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    real = observed_defrost_records(values)
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
        reasons = [f"missing_{boundary}" for boundary in missing]
        if reasons:
            row.update(
                event_valid=False,
                energy_event_valid=False,
                duration_event_valid=False,
                event_invalid_reason=";".join(reasons),
            )
            rows.append(row)
            continue
        heating, preparation = row["heating_start"], row["defrost_preparation_start"]
        defrost, defrost_end = row["defrost_start"], row["defrost_end"]
        assert isinstance(heating, pd.Timestamp) and isinstance(preparation, pd.Timestamp)
        assert isinstance(defrost, pd.Timestamp) and isinstance(defrost_end, pd.Timestamp)
        current = load_frame(loader, name, cache)
        observed = measure_defrost_event_quantities(
            current,
            preparation_start=preparation,
            defrost_start=defrost,
            defrost_end=defrost_end,
        )
        features = cast(
            dict[str, object],
            extract_pre_defrost_features(current, [preparation], heating).iloc[0].to_dict(),
        )
        audit_reasons: list[str] = []
        if not observed["phase_partition_valid"]:
            audit_reasons.append("phase_partition")
        if not observed["event_valid"]:
            for phase in ("prep", "D"):
                prefix = f"E_{phase}"
                if float(observed[f"{prefix}_coverage"]) < QUALITY_COVERAGE:  # type: ignore[arg-type]
                    audit_reasons.append(f"{prefix}_coverage")
                if float(observed[f"{prefix}_maximum_gap_seconds"]) > MAXIMUM_GAP_SECONDS:  # type: ignore[arg-type]
                    audit_reasons.append(f"{prefix}_continuous_gap")
                if not bool(observed[f"{prefix}_start_fresh"]):
                    audit_reasons.append(f"{prefix}_start_boundary")
                if not bool(observed[f"{prefix}_end_fresh"]):
                    audit_reasons.append(f"{prefix}_end_boundary")
        row.update(
            defrost_event_scope="preparation_defrost",
            **features,
            **observed,
        )
        row["event_valid"] = not audit_reasons
        row["event_invalid_reason"] = ";".join(audit_reasons)
        rows.append(row)
    return pd.DataFrame(rows)
