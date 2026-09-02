#!/usr/bin/env python3
# ruff: noqa: E501
"""Estimate empirical defrost optima from unsmoothed original cycle data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import wilcoxon

from dataloader.dataloader import DatasetLoader
from dataloader.metadata import following_cycle_names
from frost_analysis.cost.core import (
    count_true_runs,
    find_recovery_time,
    integrate_energy_curve_kwh,
    integrate_energy_kwh,
    optimize_cycle_cop_cost,
    optimize_renewal_cost,
    water_side_heating_kw,
)

RAW_COLUMNS = [
    "timestamp",
    "water_flow",
    "water_in_temperature",
    "water_out_temperature",
    "water_temperature_setpoint",
    "power_total",
    "heating_capacity",
    "cycle_stage",
    "evaporating_pressure",
    "coil_temperature",
    "ambient_temperature",
    "p1__T3o'2_20",
    "p1__DefTim1'2_20",
    "p1__DefTim2'2_20",
]
ANCHOR_SECONDS = 60
MINIMUM_HEATING_MINUTES = 10
CANDIDATE_STEP_MINUTES = 1
RECOVERY_FRACTION = 0.9
RECOVERY_SECONDS = 30
MINIMUM_INTEGRATION_COVERAGE = 0.95
FIXED_RECOVERY_ELECTRICITY_KWH = 0.279901897467
PE_FOLD_COLUMNS = [
    "fold_intercept_kwh",
    "fold_linear_kwh_per_mpa",
    "fold_quadratic_kwh_per_mpa2",
    "fold_train_pe_min_mpa",
    "fold_train_pe_max_mpa",
]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def _timestamp(value: object) -> pd.Timestamp | None:
    result = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(result) else pd.Timestamp(result)


def _raw(loader: DatasetLoader, cycle_name: str) -> pd.DataFrame:
    frame = loader.load_cycle_original(cycle_name, columns=RAW_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["q_heating_kw"] = water_side_heating_kw(frame)
    frame["q_unit_kw"] = pd.to_numeric(frame["heating_capacity"], errors="coerce")
    frame["power_total"] = pd.to_numeric(frame["power_total"], errors="coerce")
    return frame.sort_values("timestamp", kind="stable").drop_duplicates("timestamp")


def _rb_case(t4_c: float) -> int:
    if t4_c >= -2:
        return 1
    if t4_c >= -5:
        return 2
    if t4_c >= -8:
        return 3
    if t4_c >= -10:
        return 4
    return 5


def _rb_limits(t4_c: float, twout_c: float, t3o_c: float) -> tuple[float, float]:
    case = _rb_case(t4_c)
    time_limits = {
        1: (40, 35, 30),
        2: (40, 38, 33),
        3: (80, 60, 40),
        4: (90, 70, 50),
        5: (150, 120, 90),
    }
    water_bin = 0 if twout_c >= 35 else 1 if twout_c >= 25 else 2
    t3_limit = t3o_c - (3 if case == 1 else 5) if case <= 3 else t4_c - 5
    return time_limits[case][water_bin], t3_limit


def _rb_first_trigger(
    frame: pd.DataFrame, heating_start: pd.Timestamp, observation_end: pd.Timestamp
) -> dict[str, object]:
    """Replay the automatic RB controller on an unfilled one-second grid."""
    columns = {
        "coil_temperature": "T3_C",
        "ambient_temperature": "T4_C",
        "water_out_temperature": "Twout_C",
        "p1__T3o'2_20": "T3o_C",
        "p1__DefTim1'2_20": "T1_min",
        "p1__DefTim2'2_20": "T2_min",
    }
    values = frame.loc[
        pd.to_datetime(frame["timestamp"], errors="coerce").ge(heating_start)
        & pd.to_datetime(frame["timestamp"], errors="coerce").lt(observation_end),
        ["timestamp", *columns],
    ].copy()
    values["timestamp"] = pd.to_datetime(values["timestamp"], errors="coerce").dt.floor("s")
    values = values.dropna(subset=["timestamp"]).drop_duplicates("timestamp").set_index("timestamp")
    grid = pd.date_range(pd.Timestamp(heating_start).ceil("s"), pd.Timestamp(observation_end).ceil("s"), freq="s", inclusive="left")
    values = values.reindex(grid).rename(columns=columns)
    values = values.apply(pd.to_numeric, errors="coerce")
    values["T3o_C"] /= 10
    values[["T1_min", "T2_min"]] /= 60

    case = values["T4_C"].map(lambda value: _rb_case(value) if pd.notna(value) else np.nan)
    limits = [
        _rb_limits(t4, twout, t3o) if pd.notna(t4) and pd.notna(twout) and pd.notna(t3o) else (np.nan, np.nan)
        for t4, twout, t3o in values[["T4_C", "Twout_C", "T3o_C"]].itertuples(index=False, name=None)
    ]
    t1_limit = pd.Series([item[0] for item in limits], index=grid)
    t3_limit = pd.Series([item[1] for item in limits], index=grid)

    historical_t3_max = values["T3_C"].shift(50).rolling(551, min_periods=1).max()
    c1 = (
        values["T1_min"].gt(35)
        & values["T2_min"].ge(6)
        & values["T3_C"].le(-1)
        & historical_t3_max.sub(values["T3_C"]).ge(1)
    )
    c2_temperature = values["T3_C"].lt(t3_limit)
    c2_confirmed = c2_temperature.rolling(20, min_periods=20).sum().eq(20)
    c2 = values["T2_min"].ge(6) & values["T1_min"].ge(t1_limit) & c2_confirmed
    c7_temperature = values["T3_C"].le(-10) & values["T3_C"].le(0.8 * values["T4_C"] - 12)
    c7 = values["T1_min"].ge(30) & c7_temperature.rolling(20, min_periods=20).sum().eq(20)
    c8 = values["T1_min"].ge(150)

    triggered = c1 | c2 | c7 | c8
    if not triggered.any():
        return {
            "t_RB": pd.NaT,
            "rb_status": "right_censored",
            "trigger_type": "",
            "case": np.nan,
            "T4_C": np.nan,
            "Twout_C": np.nan,
            "T1_min": np.nan,
            "T2_min": np.nan,
            "T3_C": np.nan,
            "T3o_C": np.nan,
            "t_observation_end": observation_end,
        }
    timestamp = triggered[triggered].index[0]
    trigger_type = (
        "Condition1" if c1.loc[timestamp]
        else f"Case{int(case.loc[timestamp])}" if c2.loc[timestamp]
        else "Case7" if c7.loc[timestamp]
        else "Case8"
    )
    return {
        "t_RB": timestamp,
        "rb_status": "triggered",
        "trigger_type": trigger_type,
        "case": int(case.loc[timestamp]) if pd.notna(case.loc[timestamp]) else np.nan,
        **values.loc[timestamp, ["T4_C", "Twout_C", "T1_min", "T2_min", "T3_C", "T3o_C"]].to_dict(),
        "t_observation_end": observation_end,
    }


def _rb_trigger_table(
    catalog: pd.DataFrame, frames: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for row in catalog.itertuples(index=False):
        frame = frames[row.cycle_name]
        heating_start = _timestamp(row.heating_start) or pd.Timestamp(frame["timestamp"].min())
        observation_end = _timestamp(row.defrost_preparation_start)
        observation_end_source = "actual_preparation"
        if observation_end is None:
            observation_end = pd.Timestamp(frame["timestamp"].max()).floor("s") + pd.Timedelta(seconds=1)
            observation_end_source = "raw_cycle_end"
        trigger = _rb_first_trigger(frame, heating_start, observation_end)
        rows.append(
            {
                "cycle_name": row.cycle_name,
                "cycle_id": row.cycle_id,
                "experiment_id": row.experiment_id,
                "observation_end_source": observation_end_source,
                **trigger,
            }
        )
    return pd.DataFrame(rows)


def _rb_candidate_costs(
    results: pd.DataFrame, candidate_curves: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    eligible = candidate_curves.loc[
        candidate_curves["optimization_eligible"].fillna(False)
    ].copy()
    eligible["candidate_time"] = pd.to_datetime(eligible["candidate_time"], errors="coerce")
    for row in results.itertuples(index=False):
        match = eligible.loc[eligible["cycle_name"].eq(row.cycle_name)].copy()
        rb_time = _timestamp(row.t_RB) if row.rb_status == "triggered" else None
        if rb_time is not None and not match.empty:
            distance = (match["candidate_time"] - rb_time).abs().dt.total_seconds() / 60
            nearest = match.loc[distance.idxmin()]
            if float(distance.min()) <= 0.51:
                rows.append(
                    {
                        "cycle_name": row.cycle_name,
                        "rb_candidate_time": nearest["candidate_time"],
                        "rb_inverse_cop": nearest["inverse_cop"],
                        "rb_relative_regret": nearest["relative_regret"],
                    }
                )
                continue
        rows.append(
            {
                "cycle_name": row.cycle_name,
                "rb_candidate_time": pd.NaT,
                "rb_inverse_cop": np.nan,
                "rb_relative_regret": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _anchor(frame: pd.DataFrame, start: pd.Timestamp | None) -> dict[str, object]:
    if start is None:
        return {"valid": False, "invalid_reason": "missing_stable_heating_start"}
    window = frame.loc[
        frame["timestamp"].ge(start)
        & frame["timestamp"].lt(start + pd.Timedelta(seconds=ANCHOR_SECONDS))
    ]
    valid = window[["q_heating_kw", "power_total"]].dropna()
    if len(valid) < 0.8 * ANCHOR_SECONDS:
        return {"valid": False, "invalid_reason": "clean_anchor_coverage_below_80pct"}
    q_kw = float(valid["q_heating_kw"].median())
    power_kw = float(valid["power_total"].median())
    if q_kw <= 0 or power_kw <= 0:
        return {"valid": False, "invalid_reason": "nonpositive_clean_anchor"}
    return {
        "valid": True,
        "invalid_reason": "",
        "anchor_start": start,
        "q_clean_kw": q_kw,
        "power_clean_kw": power_kw,
        "cop_clean": q_kw / power_kw,
        "observed_points": len(valid),
    }


def _reference_kw(
    timestamps: pd.Series,
    start_time: pd.Timestamp,
    start_kw: float,
    end_time: pd.Timestamp,
    end_kw: float,
) -> np.ndarray:
    x = pd.to_datetime(timestamps).astype("int64").to_numpy(dtype=float)
    return np.interp(
        x,
        [float(start_time.value), float(end_time.value)],
        [start_kw, end_kw],
    )


def _candidate_pressure_features(
    frame: pd.DataFrame,
    end: pd.Timestamp,
    *,
    seconds: int = 60,
    maximum_gap_seconds: float = 5.0,
) -> dict[str, float | int]:
    """Summarize the strict pre-action Pe window on a 1 s grid."""
    if "evaporating_pressure" not in frame:
        return {
            "evaporating_pressure_mpa": np.nan,
            "pe_raw_valid_seconds": 0,
            "pe_interpolated_valid_seconds": 0,
            "pe_interpolated_coverage": 0.0,
            "pe_internal_gap_interpolated": False,
            "pe_extrapolated_valid_seconds": 0,
            "pe_endpoint_extrapolated": False,
        }
    if isinstance(frame.index, pd.DatetimeIndex) and "timestamp" not in frame:
        values = frame[["evaporating_pressure"]]
    else:
        values = frame[["timestamp", "evaporating_pressure"]].copy()
        values["timestamp"] = pd.to_datetime(values["timestamp"], errors="coerce")
        values["evaporating_pressure"] = pd.to_numeric(
            values["evaporating_pressure"], errors="coerce"
        )
        values = values.dropna(subset=["timestamp"]).sort_values(
            "timestamp", kind="stable"
        ).drop_duplicates("timestamp").set_index("timestamp")
    grid = pd.date_range(end - pd.Timedelta(seconds=seconds), periods=seconds, freq="s")
    raw_window = values["evaporating_pressure"].reindex(grid)
    interpolation_index = values.index.union(grid).sort_values()
    interpolated = values["evaporating_pressure"].reindex(interpolation_index)
    raw_valid = int(raw_window.notna().sum())
    interpolated = interpolated.interpolate(method="time", limit_area="inside").reindex(grid)
    finite = values["evaporating_pressure"].dropna().sort_index()
    endpoint_mask = pd.Series(False, index=grid)
    if len(finite) >= 2:
        finite_time = finite.index
        finite_values = finite.to_numpy(dtype=float)
        left_slope = (finite_values[1] - finite_values[0]) / (
            (finite_time[1] - finite_time[0]).total_seconds()
        )
        right_slope = (finite_values[-1] - finite_values[-2]) / (
            (finite_time[-1] - finite_time[-2]).total_seconds()
        )
        left = grid < finite_time[0]
        right = grid > finite_time[-1]
        grid_ns = grid.view("i8").astype(float)
        interpolated.loc[left] = finite_values[0] + left_slope * (
            grid_ns[left] - finite_time[0].value
        ) / 1e9
        interpolated.loc[right] = finite_values[-1] + right_slope * (
            grid_ns[right] - finite_time[-1].value
        ) / 1e9
        endpoint_mask = pd.Series(left | right, index=grid)
    interpolated_valid = int(interpolated.notna().sum())
    extrapolated_valid = int((endpoint_mask & interpolated.notna()).sum())
    left_position = finite.index.searchsorted(pd.Timestamp(end), side="right") - 1
    right_position = left_position + 1
    pe_internal_gap = False
    if 0 <= left_position < right_position < len(finite.index):
        left = finite.index[left_position]
        right = finite.index[right_position]
        pe_internal_gap = (
            left < pd.Timestamp(end)
            and pd.Timestamp(end) < right
            and (right - left).total_seconds() > maximum_gap_seconds
        )
    return {
        "evaporating_pressure_mpa": (
            float(interpolated.median()) if interpolated_valid else np.nan
        ),
        "pe_raw_valid_seconds": raw_valid,
        "pe_interpolated_valid_seconds": interpolated_valid,
        "pe_interpolated_coverage": interpolated_valid / seconds,
        "pe_internal_gap_interpolated": pe_internal_gap,
        "pe_extrapolated_valid_seconds": extrapolated_valid,
        "pe_endpoint_extrapolated": bool(extrapolated_valid),
    }


def _candidate_state_features(
    frame: pd.DataFrame, end: pd.Timestamp, *, seconds: int = 60
) -> dict[str, float]:
    """Return candidate-available state medians from the strict pre-action window."""
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    window = frame.loc[
        timestamps.ge(end - pd.Timedelta(seconds=seconds)) & timestamps.lt(end)
    ]
    result = {}
    for column in (
        "water_in_temperature",
        "water_out_temperature",
        "coil_temperature",
        "water_temperature_setpoint",
    ):
        values = (
            pd.to_numeric(window[column], errors="coerce").dropna()
            if column in window
            else pd.Series(dtype=float)
        )
        result[column] = float(values.median()) if not values.empty else np.nan
    return result



def _read_pe_folds(path: Path) -> pd.DataFrame:
    """Read one unique experiment-held-out Pe coefficient tuple per experiment."""
    source = pd.read_csv(path)
    required = ["experiment_id", *PE_FOLD_COLUMNS]
    missing = set(required) - set(source)
    if missing:
        raise ValueError(f"LOEO Pe source is missing columns: {sorted(missing)}")
    unique_counts = source.groupby("experiment_id")[PE_FOLD_COLUMNS].nunique(
        dropna=False
    )
    if not unique_counts.eq(1).all().all():
        raise ValueError("each experiment requires one unique LOEO Pe fold tuple")
    folds = source.groupby("experiment_id", as_index=False)[PE_FOLD_COLUMNS].first()
    if not np.isfinite(folds[PE_FOLD_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("LOEO Pe fold coefficients and support must be finite")
    return folds.set_index("experiment_id")


def _apply_pe_fold(candidates: pd.DataFrame, fold: pd.Series) -> pd.DataFrame:
    """Attach the quadratic LOEO Pe ticket and model-continuation audit labels."""
    values = candidates.copy()
    for column in PE_FOLD_COLUMNS:
        values[column] = float(fold[column])
    pe = pd.to_numeric(values["evaporating_pressure_mpa"], errors="coerce")
    lower = float(fold["fold_train_pe_min_mpa"])
    upper = float(fold["fold_train_pe_max_mpa"])
    values["support_status"] = np.select(
        [pe.isna(), pe.lt(lower), pe.gt(upper)],
        ["missing", "below", "above"],
        default="supported",
    )
    values["pe_extrapolation_distance_mpa_signed"] = np.select(
        [pe.lt(lower), pe.gt(upper)], [pe - lower, pe - upper], default=0.0
    )
    values.loc[pe.isna(), "pe_extrapolation_distance_mpa_signed"] = np.nan
    values["pe_extrapolation_distance_mpa_absolute"] = values[
        "pe_extrapolation_distance_mpa_signed"
    ].abs()
    values["pe_supported"] = values["support_status"].eq("supported")
    values["integration_eligible"] = values["integration_coverage"].ge(
        MINIMUM_INTEGRATION_COVERAGE
    )
    values["predicted_preparation_defrost_electricity_kwh"] = (
        float(fold["fold_intercept_kwh"])
        + float(fold["fold_linear_kwh_per_mpa"]) * pe
        + float(fold["fold_quadratic_kwh_per_mpa2"]) * pe.pow(2)
    )
    values["fixed_recovery_electricity_kwh"] = FIXED_RECOVERY_ELECTRICITY_KWH
    values["dynamic_ticket_electricity_kwh"] = (
        values["predicted_preparation_defrost_electricity_kwh"]
        + values["fixed_recovery_electricity_kwh"]
    )
    values["optimization_eligible"] = values["integration_eligible"] & pe.notna()
    return values



def _candidate_eligibility_audit(candidates: pd.DataFrame) -> dict[str, float | int]:
    """Return distinct Pe-support and joint-optimization candidate counts."""
    count = len(candidates)
    pe_supported = int(candidates["pe_supported"].fillna(False).sum())
    integration_eligible = int(
        candidates["integration_eligible"].fillna(False).sum()
    )
    optimization_eligible = int(
        candidates["optimization_eligible"].fillna(False).sum()
    )
    pe_fraction = pe_supported / count if count else 0.0
    integration_fraction = integration_eligible / count if count else 0.0
    optimization_fraction = optimization_eligible / count if count else 0.0
    return {
        "pe_supported_candidate_count": pe_supported,
        "pe_supported_candidate_fraction": pe_fraction,
        "integration_eligible_candidate_count": integration_eligible,
        "integration_eligible_candidate_fraction": integration_fraction,
        "optimization_eligible_candidate_count": optimization_eligible,
        "optimization_eligible_candidate_fraction": optimization_fraction,
        "supported_candidate_count": pe_supported,
        "support_coverage_fraction": pe_fraction,
    }


def _pe_support_summary(candidates: pd.DataFrame) -> dict[str, int]:
    """Count support, finite-domain extrapolation, and missing Pe separately."""
    status = candidates["support_status"].astype(str)
    return {
        "supported_count": int(status.eq("supported").sum()),
        "extrapolated_count": int(status.isin(["above", "below"]).sum()),
        "missing_count": int(status.eq("missing").sum()),
    }


def _no_eligible_failure_reason(candidates: pd.DataFrame) -> str:
    """Distinguish absence of Pe support from absence of joint eligibility."""
    return (
        "no_optimization_eligible_candidates"
        if candidates["pe_supported"].fillna(False).any()
        else "no_pe_supported_candidates"
    )


def _dynamic_report_statistics(
    points: pd.DataFrame, curves: pd.DataFrame
) -> dict[str, float | int | str]:
    """Return support, eligibility, and fixed-ticket sensitivity audit statistics."""
    shifts = pd.to_numeric(
        points["dynamic_vs_fixed_ticket_shift_minutes"], errors="coerce"
    ).abs()
    maximum = points.loc[shifts.idxmax()]
    fully_supported = points.loc[
        pd.to_numeric(
            points["pe_supported_candidate_fraction"], errors="coerce"
        ).eq(1.0)
    ]
    fully_supported_shifts = pd.to_numeric(
        fully_supported["dynamic_vs_fixed_ticket_shift_minutes"], errors="coerce"
    ).abs()
    return {
        "candidate_count": len(curves),
        "pe_supported_candidate_count": int(
            curves["pe_supported"].fillna(False).sum()
        ),
        "integration_eligible_candidate_count": int(
            curves["integration_eligible"].fillna(False).sum()
        ),
        "optimization_eligible_candidate_count": int(
            curves["optimization_eligible"].fillna(False).sum()
        ),
        "maximum_shift_cycle": str(maximum["cycle_name"]),
        "maximum_shift_minutes": float(shifts.max()),
        "maximum_shift_optimization_fraction": float(
            maximum["optimization_eligible_candidate_fraction"]
        ),
        "fully_pe_supported_cycle_count": len(fully_supported),
        "fully_pe_supported_shift_median": float(fully_supported_shifts.median()),
        "fully_pe_supported_shift_p90": float(fully_supported_shifts.quantile(0.9)),
        "fully_pe_supported_shift_maximum": float(fully_supported_shifts.max()),
    }


def _fixed_ticket_optimum(
    candidates: pd.DataFrame, ticket_kwh: float, observed_end: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate a fixed ticket on the full domain while excluding low-coverage rows."""
    values = candidates.copy()
    values["pe_supported"] = True
    values["integration_eligible"] = values["integration_coverage"].ge(
        MINIMUM_INTEGRATION_COVERAGE
    )
    values["optimization_eligible"] = values["integration_eligible"]
    return optimize_cycle_cop_cost(
        values,
        defrost_recovery_electricity_kwh=ticket_kwh,
        required_end_time=observed_end,
    )


def _candidate_costs(
    frame: pd.DataFrame,
    *,
    stable_start: pd.Timestamp,
    candidate_end: pd.Timestamp,
    q_start_kw: float,
    next_stable_start: pd.Timestamp,
    q_end_kw: float,
    lambda_q: float,
) -> pd.DataFrame:
    heating = frame.loc[
        frame["timestamp"].ge(stable_start) & frame["timestamp"].le(candidate_end)
    ].copy()
    heating["q_reference_kw"] = _reference_kw(
        heating["timestamp"], stable_start, q_start_kw, next_stable_start, q_end_kw
    )
    heating["thermal_shortfall_kw"] = np.maximum(
        heating["q_reference_kw"] - heating["q_heating_kw"], 0.0
    )
    heating["equivalent_power_kw"] = (
        heating["power_total"] + lambda_q * heating["thermal_shortfall_kw"]
    )
    heating["effective_user_heating_kw"] = heating["q_heating_kw"].clip(lower=0)
    heating["effective_unit_heating_kw"] = heating["q_unit_kw"].clip(lower=0)
    first = stable_start + pd.Timedelta(minutes=MINIMUM_HEATING_MINUTES)
    candidates = list(pd.date_range(first, candidate_end, freq=f"{CANDIDATE_STEP_MINUTES}min"))
    if candidates and candidates[-1] != candidate_end:
        candidates.append(candidate_end)
    legacy = integrate_energy_curve_kwh(
        heating["timestamp"],
        heating["equivalent_power_kw"],
        candidates,
        bridge_internal_gaps=True,
        extrapolate_endpoints=True,
    )
    electricity = integrate_energy_curve_kwh(
        heating["timestamp"],
        heating["power_total"],
        candidates,
        bridge_internal_gaps=True,
        extrapolate_endpoints=True,
    )
    user_heat = integrate_energy_curve_kwh(
        heating["timestamp"],
        heating["effective_user_heating_kw"],
        candidates,
        bridge_internal_gaps=True,
        extrapolate_endpoints=True,
    )
    unit_heat = integrate_energy_curve_kwh(
        heating["timestamp"],
        heating["effective_unit_heating_kw"],
        candidates,
        bridge_internal_gaps=True,
        extrapolate_endpoints=True,
    )
    pressure = frame[["timestamp", "evaporating_pressure"]].copy() if (
        "evaporating_pressure" in frame
    ) else pd.DataFrame(columns=["timestamp", "evaporating_pressure"])
    pressure["timestamp"] = pd.to_datetime(pressure["timestamp"], errors="coerce")
    pressure["evaporating_pressure"] = pd.to_numeric(
        pressure["evaporating_pressure"], errors="coerce"
    )
    pressure = pressure.dropna(subset=["timestamp"]).sort_values(
        "timestamp", kind="stable"
    ).drop_duplicates("timestamp").set_index("timestamp")
    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        pressure_features = _candidate_pressure_features(pressure, candidate)
        state_features = _candidate_state_features(frame, candidate)
        rows.append(
            {
                "candidate_time": candidate,
                "heating_hours": (candidate - stable_start).total_seconds() / 3600,
                "heating_electricity_kwh": electricity.iloc[index]["energy_kwh"],
                "user_heating_kwh": user_heat.iloc[index]["energy_kwh"],
                "water_heating_kwh": user_heat.iloc[index]["energy_kwh"],
                "unit_heating_kwh": unit_heat.iloc[index]["energy_kwh"],
                "unit_heating_coverage": unit_heat.iloc[index]["coverage"],
                "heating_cost_kwh": legacy.iloc[index]["energy_kwh"],
                "integration_coverage": min(
                    legacy.iloc[index]["coverage"],
                    electricity.iloc[index]["coverage"],
                    user_heat.iloc[index]["coverage"],
                ),
                "candidate_in_interpolated_gap": bool(
                    legacy.iloc[index]["bridged_internal_gap"]
                    or electricity.iloc[index]["bridged_internal_gap"]
                    or user_heat.iloc[index]["bridged_internal_gap"]
                    or unit_heat.iloc[index]["bridged_internal_gap"]
                    or pressure_features["pe_internal_gap_interpolated"]
                ),
                "candidate_in_extrapolated_endpoint": bool(
                    legacy.iloc[index]["extrapolated_endpoint"]
                    or electricity.iloc[index]["extrapolated_endpoint"]
                    or user_heat.iloc[index]["extrapolated_endpoint"]
                    or unit_heat.iloc[index]["extrapolated_endpoint"]
                    or pressure_features["pe_endpoint_extrapolated"]
                ),
                **state_features,
                **pressure_features,
            }
        )
    return pd.DataFrame(rows)


def _compare_heat_bases(
    candidates: pd.DataFrame, required_end_time: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    """Evaluate water and canonical unit heat on an unchanged candidate domain."""
    ticket = candidates["dynamic_ticket_electricity_kwh"]
    water_curve, water_optimum = optimize_cycle_cop_cost(
        candidates,
        defrost_recovery_electricity_kwh=ticket,
        required_end_time=required_end_time,
    )
    unit_candidates = candidates.copy()
    unit_candidates["user_heating_kwh"] = unit_candidates["unit_heating_kwh"]
    unit_curve, unit_optimum = optimize_cycle_cop_cost(
        unit_candidates,
        defrost_recovery_electricity_kwh=ticket,
        required_end_time=required_end_time,
    )
    water_curve["inverse_cop_water"] = water_curve["inverse_cop"]
    water_curve["inverse_cop_unit"] = unit_curve["inverse_cop"]
    eligible = water_curve["optimization_eligible"].fillna(False)
    water_curve["relative_regret_water"] = (
        water_curve["inverse_cop_water"] / float(water_optimum["inverse_cop"]) - 1
    ).where(eligible)
    water_curve["relative_regret_unit"] = (
        water_curve["inverse_cop_unit"] / float(unit_optimum["inverse_cop"]) - 1
    ).where(eligible)
    return water_curve, water_optimum, unit_optimum


def _heat_ratio_progress(
    frame: pd.DataFrame,
    cycle_name: str,
    stable_start: pd.Timestamp,
    preparation_start: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Integrate both heat quantities in ten normalized pre-action time bins."""
    values = frame.loc[
        frame["timestamp"].between(stable_start, preparation_start),
        [
            "timestamp",
            "q_heating_kw",
            "q_unit_kw",
            "water_in_temperature",
            "water_out_temperature",
            "water_flow",
        ],
    ].copy()
    values["water_delta_temperature_C"] = pd.to_numeric(
        values["water_out_temperature"], errors="coerce"
    ) - pd.to_numeric(values["water_in_temperature"], errors="coerce")
    flow = pd.to_numeric(values["water_flow"], errors="coerce")
    q_water = pd.to_numeric(values["q_heating_kw"], errors="coerce")
    q_unit = pd.to_numeric(values["q_unit_kw"], errors="coerce")
    valid_offset = flow.gt(0) & q_water.gt(0) & q_unit.gt(0)
    values["equivalent_delta_t_offset_C"] = (
        (q_water - q_unit) / (1.161 * flow)
    ).where(valid_offset)
    edges = pd.date_range(stable_start, preparation_start, periods=11)
    rows: list[dict[str, object]] = []
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        part = values.loc[values["timestamp"].between(start, end)]
        water_energy, water_coverage = integrate_energy_kwh(
            part["timestamp"], part["q_heating_kw"]
        )
        unit_energy, unit_coverage = integrate_energy_kwh(
            part["timestamp"], part["q_unit_kw"]
        )
        rows.append(
            {
                "cycle_name": cycle_name,
                "progress_bin": index + 1,
                "progress_start": index / 10,
                "progress_end": (index + 1) / 10,
                "progress_midpoint": (index + 0.5) / 10,
                "bin_start": start,
                "bin_end": end,
                "water_heating_kwh": water_energy,
                "unit_heating_kwh": unit_energy,
                "water_heating_coverage": water_coverage,
                "unit_heating_coverage": unit_coverage,
                "unit_to_water_heat_ratio": (
                    unit_energy / water_energy if water_energy > 0 else np.nan
                ),
                "median_water_delta_temperature_C": part[
                    "water_delta_temperature_C"
                ].median(),
                "median_water_flow": pd.to_numeric(
                    part["water_flow"], errors="coerce"
                ).median(),
                "equivalent_delta_t_offset_C": part[
                    "equivalent_delta_t_offset_C"
                ].median(),
            }
        )
    bins = pd.DataFrame(rows)
    early, late = bins.iloc[:2], bins.iloc[-2:]
    early_ratio = early["unit_heating_kwh"].sum() / early["water_heating_kwh"].sum()
    late_ratio = late["unit_heating_kwh"].sum() / late["water_heating_kwh"].sum()
    early_offset = early["equivalent_delta_t_offset_C"].median()
    late_offset = late["equivalent_delta_t_offset_C"].median()
    return bins, {
        "heat_ratio_early": float(early_ratio),
        "heat_ratio_late": float(late_ratio),
        "heat_ratio_late_minus_early": float(late_ratio - early_ratio),
        "equivalent_delta_t_offset_early_C": float(early_offset),
        "equivalent_delta_t_offset_late_C": float(late_offset),
        "equivalent_delta_t_offset_late_minus_early_C": float(late_offset - early_offset),
    }


def _observed_cycle_boundary_reason(row: pd.Series) -> str:
    """Return why a cycle lacks the observed facts needed for a counterfactual curve."""
    if row.get("status") != "valid":
        return f"catalog_{row.get('status')}"
    if _timestamp(row.get("stable_heating_start")) is None:
        return "missing_stable_heating_start"
    if _timestamp(row.get("defrost_start")) is None:
        return "missing_defrost_start"
    if _timestamp(row.get("defrost_end")) is None:
        return "missing_defrost_end"
    return ""


def _preparation_candidate_boundary_reason(row: pd.Series) -> str:
    reason = _observed_cycle_boundary_reason(row)
    if reason:
        return reason
    stable = _timestamp(row.get("stable_heating_start"))
    preparation = _timestamp(row.get("defrost_preparation_start"))
    defrost = _timestamp(row.get("defrost_start"))
    if preparation is None:
        return "missing_defrost_preparation_start"
    if stable is None or defrost is None or not stable < preparation < defrost:
        return "invalid_defrost_preparation_boundary_order"
    return ""


def _ticket_boundary_reason(row: pd.Series, following: str | None) -> str:
    reason = _observed_cycle_boundary_reason(row)
    if reason:
        return reason
    return "missing_next_cycle_recovery" if following is None else ""


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    svg = base.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _near_optimal_segments(curve: pd.DataFrame) -> list[tuple[float, float]]:
    """Return disconnected <=5% regret intervals in plotted minutes."""
    selected = curve["relative_regret"].le(0.05) & curve.get(
        "optimization_eligible", True
    )
    groups = selected.ne(selected.shift(fill_value=False)).cumsum()
    return [
        (float(rows["minutes"].min()), float(rows["minutes"].max()))
        for _, rows in curve.loc[selected].groupby(groups[selected], sort=False)
    ]


def _near_optimal_segment_rows(
    cycle_name: str, curve: pd.DataFrame, *, fraction: float = 0.05
) -> pd.DataFrame:
    """Return contiguous eligible near-optimal segments for one cycle."""
    values = curve.sort_values("candidate_time", kind="stable").copy()
    eligible = values.get(
        "optimization_eligible", pd.Series(True, index=values.index)
    ).fillna(False).astype(bool)
    selected = eligible & pd.to_numeric(
        values["relative_regret"], errors="coerce"
    ).le(fraction)
    groups = selected.ne(selected.shift(fill_value=False)).cumsum()
    optimum_index = pd.to_numeric(
        values.loc[eligible, "relative_regret"], errors="coerce"
    ).idxmin()
    rows = []
    for segment_index, (_, segment) in enumerate(
        values.loc[selected].groupby(groups[selected], sort=False), start=1
    ):
        start = pd.Timestamp(segment["candidate_time"].min())
        end = pd.Timestamp(segment["candidate_time"].max())
        rows.append(
            {
                "cycle_name": cycle_name,
                "relative_regret_threshold": fraction,
                "segment_index": segment_index,
                "segment_start": start,
                "segment_end": end,
                "segment_width_minutes": (end - start).total_seconds() / 60,
                "contains_t_star": bool(optimum_index in segment.index),
            }
        )
    return pd.DataFrame(rows)


def _plot_cycle(
    frame: pd.DataFrame,
    result: pd.Series,
    curve: pd.DataFrame,
    *,
    q_start_kw: float,
    next_stable_start: pd.Timestamp,
    q_end_kw: float,
    atlas: PdfPages | None = None,
) -> None:
    stable = pd.Timestamp(result["t_heating_stable"])
    actual = _timestamp(result.get("t_actual_preparation", result.get("t_actual_defrost")))
    candidate_end = pd.Timestamp(result["candidate_end"])
    optimum = pd.Timestamp(result["t_star"])
    shown = frame.loc[frame["timestamp"].between(stable, candidate_end)].copy()
    shown["minutes"] = (shown["timestamp"] - stable).dt.total_seconds() / 60
    shown["q_reference_kw"] = _reference_kw(
        shown["timestamp"], stable, q_start_kw, next_stable_start, q_end_kw
    )
    curve = curve.copy()
    curve["minutes"] = (pd.to_datetime(curve["candidate_time"]) - stable).dt.total_seconds() / 60
    x_star = (optimum - stable).total_seconds() / 60
    x_actual = (actual - stable).total_seconds() / 60 if actual is not None else None
    x_rb = pd.to_numeric(result.get("rb_minutes_from_stable"), errors="coerce")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    axes[0].plot(
        shown["minutes"],
        shown["q_heating_kw"],
        color="#7A8793",
        lw=0.45,
        alpha=0.75,
        label="Raw water-side $Q_h$",
    )
    axes[0].plot(
        shown["minutes"], shown["q_reference_kw"], color="#2166AC", lw=1.3, label="$Q_{ref}$"
    )
    axes[0].set_ylabel("Heating capacity (kW)")
    axes[0].legend(loc="best")
    eligible = (
        curve["optimization_eligible"].fillna(False).astype(bool)
        if "optimization_eligible" in curve
        else pd.Series(True, index=curve.index)
    )
    pe_supported = (
        curve["pe_supported"].fillna(False).astype(bool)
        if "pe_supported" in curve
        else eligible.copy()
    )
    integration_eligible = (
        curve["integration_eligible"].fillna(False).astype(bool)
        if "integration_eligible" in curve
        else eligible | ~pe_supported
    )
    axes[1].plot(
        curve["minutes"],
        curve["inverse_cop"].where(eligible),
        color="#4C78A8",
        lw=1.3,
    )
    if pd.notna(x_rb):
        axes[1].axvline(
            float(x_rb), color="#2E7D5B", lw=0.9, ls="--", label="RB baseline"
        )
        rb_value = np.interp(float(x_rb), curve["minutes"], curve["inverse_cop"])
        axes[1].plot(float(x_rb), rb_value, "o", ms=3, color="#2E7D5B")
    unsupported = curve.loc[~pe_supported & curve["inverse_cop"].notna()]
    if not unsupported.empty:
        axes[1].scatter(
            unsupported["minutes"],
            unsupported["inverse_cop"],
            marker="x",
            s=10,
            color="#A7ADB3",
            linewidths=0.6,
            label="Outside Pe support",
        )
    insufficient_integration = curve.loc[
        pe_supported & ~integration_eligible & curve["inverse_cop"].notna()
    ]
    if not insufficient_integration.empty:
        axes[1].scatter(
            insufficient_integration["minutes"],
            insufficient_integration["inverse_cop"],
            marker="+",
            s=12,
            color="#C66A00",
            linewidths=0.7,
            label="Insufficient integration coverage",
        )
    interpolated_gap = curve.loc[
        curve.get(
            "candidate_in_interpolated_gap", pd.Series(False, index=curve.index)
        ).fillna(False).astype(bool)
        & curve["inverse_cop"].notna()
    ]
    if not interpolated_gap.empty:
        axes[1].scatter(
            interpolated_gap["minutes"],
            interpolated_gap["inverse_cop"],
            marker="s",
            s=16,
            facecolors="none",
            edgecolors="#9AA0A6",
            linewidths=0.65,
            label="Internal-gap interpolation",
        )
    for segment_index, (start, end) in enumerate(_near_optimal_segments(curve)):
        half_step = 0.5 * CANDIDATE_STEP_MINUTES if start == end else 0.0
        for axis in axes:
            axis.axvspan(
                start - half_step,
                end + half_step,
                color="#9ECAE1",
                alpha=0.28,
                lw=0,
                label="5% near-optimal" if segment_index == 0 and axis is axes[1] else None,
            )
    for axis in axes:
        axis.axvline(x_star, color="#D95F02", lw=1.1, label="Empirical optimum")
        if x_actual is not None:
            axis.axvline(
                x_actual,
                color="#222222",
                lw=0.9,
                ls="--",
                label="Observed preparation",
            )
    axes[1].set_ylabel("Cycle electricity / user heat")
    axes[1].set_xlabel("Minutes from stable heating")
    axes[1].legend(loc="best", ncol=3)
    boundary = "right-censored sensor end" if bool(result["is_censored"]) else "observed policy"
    fig.suptitle(f"{result['cycle_name']} · {result['cohort_tier']} · {boundary}", fontsize=8)
    fig.tight_layout()
    if atlas is not None:
        atlas.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_main(
    representative_frame: pd.DataFrame,
    representative: pd.Series,
    representative_curve: pd.DataFrame,
    valid_results: pd.DataFrame,
    tickets: pd.DataFrame,
    *,
    q_start_kw: float,
    next_stable_start: pd.Timestamp,
    q_end_kw: float,
    output: Path,
) -> None:
    stable = pd.Timestamp(representative["t_heating_stable"])
    actual = pd.Timestamp(
        representative.get("t_actual_preparation", representative["t_actual_defrost"])
    )
    optimum = pd.Timestamp(representative["t_star"])
    shown = representative_frame.loc[
        representative_frame["timestamp"].between(stable, actual)
    ].copy()
    shown["minutes"] = (shown["timestamp"] - stable).dt.total_seconds() / 60
    shown["q_reference_kw"] = _reference_kw(
        shown["timestamp"], stable, q_start_kw, next_stable_start, q_end_kw
    )
    curve = representative_curve.copy()
    curve["minutes"] = (pd.to_datetime(curve["candidate_time"]) - stable).dt.total_seconds() / 60
    x_star = (optimum - stable).total_seconds() / 60
    x_actual = (actual - stable).total_seconds() / 60
    x_rb = pd.to_numeric(representative.get("rb_minutes_from_stable"), errors="coerce")

    base = output.parent

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot(shown["minutes"], shown["q_heating_kw"], color="#7A8793", lw=0.45, alpha=0.75)
    ax.axvline(x_star, color="#D95F02", lw=1.0)
    ax.axvline(x_actual, color="#222222", lw=0.9, ls="--")
    ax.set(xlabel="Time from stable heating start (min)", ylabel="Heating capacity (kW)")
    ax.legend(
        handles=[
            Line2D([], [], color="#7A8793", lw=0.8, label="Raw $Q_h$"),
            Line2D([], [], color="#D95F02", lw=1.0, label="Optimum"),
            Line2D([], [], color="#222222", lw=0.9, ls="--", label="Observed"),
        ],
        loc="best",
        ncol=2,
        fontsize=5.5,
    )
    _save_figure(fig, base / "figure_1a_representative_heating_capacity")

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    eligible = curve.get("optimization_eligible", pd.Series(True, index=curve.index))
    ax.plot(
        curve["minutes"],
        curve["inverse_cop"].where(eligible),
        color="#4C78A8",
        lw=1.35,
    )
    unsupported = curve.loc[~eligible]
    if not unsupported.empty:
        ax.scatter(
            unsupported["minutes"],
            unsupported["inverse_cop"],
            marker="x",
            s=10,
            color="#A7ADB3",
            linewidths=0.6,
        )
    near = curve.get(
        "is_near_optimal",
        curve["inverse_cop"].le(1.05 * curve.loc[eligible, "inverse_cop"].min()),
    )
    ax.fill_between(
        curve["minutes"],
        0,
        1,
        where=near,
        transform=ax.get_xaxis_transform(),
        color="#9ECAE1",
        alpha=0.35,
    )
    ax.axvline(x_star, color="#D95F02", lw=1.0)
    ax.axvline(x_actual, color="#222222", lw=0.9, ls="--")
    if pd.notna(x_rb):
        ax.axvline(
            float(x_rb), color="#2E7D5B", lw=0.9, ls="--", label="RB baseline"
        )
        ax.plot(
            float(x_rb),
            np.interp(float(x_rb), curve["minutes"], curve["inverse_cop"]),
            "o",
            ms=3,
            color="#2E7D5B",
        )
    ax.set(
        xlabel="Time from stable heating start (min)",
        ylabel=r"Unit heating electricity $J=1/COP_{cyc}$",
    )
    ax.legend(
        handles=[
            Line2D([], [], color="#9ECAE1", lw=5, alpha=0.45, label="5% near-optimal"),
            Line2D([], [], color="#D95F02", lw=1.0, label="Optimum"),
            Line2D([], [], color="#2E7D5B", lw=0.9, ls="--", marker="o", ms=3, label="RB baseline"),
            Line2D([], [], color="#222222", lw=0.9, ls="--", label="Observed"),
        ],
        loc="best",
        fontsize=5.5,
    )
    _save_figure(fig, base / "figure_1b_representative_cycle_inverse_cop")

    colors = valid_results["minimum_location"].map(
        {
            "interior": "#4C78A8",
            "left_boundary": "#E6A34A",
            "right_observed": "#8C8C8C",
            "right_support_limited": "#A7ADB3",
            "right_integration_limited": "#C66A00",
            "right_boundary": "#8C8C8C",
        }
    )
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.scatter(
        valid_results["minutes_from_stable"],
        valid_results["actual_minutes_from_stable"],
        c=colors,
        s=18,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.3,
    )
    limit = float(
        np.nanmax(valid_results[["minutes_from_stable", "actual_minutes_from_stable"]].to_numpy())
    )
    ax.plot([0, limit], [0, limit], color="#777777", lw=0.8, ls="--")
    ax.set(
        xlim=(0, limit * 1.03),
        ylim=(0, limit * 1.03),
        xlabel="Estimated minimum from stable heating (min)",
        ylabel="Observed preparation from stable heating (min)",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color="#4C78A8", label="Interior"),
            Line2D([], [], marker="o", ls="", color="#8C8C8C", label="Observed right"),
            Line2D(
                [], [], marker="o", ls="", color="#C66A00", label="Integration-limited"
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3,
        fontsize=5.5,
    )
    ax.text(0.97, 0.04, f"n = {len(valid_results)}", transform=ax.transAxes, ha="right")
    _save_figure(fig, base / "figure_1c_optimum_vs_observed_defrost")

    advance = valid_results["minutes_earlier_than_actual"].dropna()
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.boxplot(
        advance,
        orientation="horizontal",
        widths=0.35,
        patch_artist=True,
        boxprops={"facecolor": "#C6DBEF", "edgecolor": "#4C78A8"},
        medianprops={"color": "#D95F02", "linewidth": 1.2},
        whiskerprops={"color": "#4C78A8"},
        capprops={"color": "#4C78A8"},
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": "#7A8793",
            "markeredgecolor": "none",
        },
    )
    jitter = np.random.default_rng(0).uniform(-0.07, 0.07, len(advance))
    ax.scatter(advance, 1 + jitter, s=8, color="#4C78A8", alpha=0.45, zorder=3)
    ax.axvline(0, color="#777777", lw=0.8, ls="--")
    ax.set(yticks=[], xlabel="Advance relative to observed preparation (min)")
    ax.text(
        0.97,
        0.92,
        f"n = {len(valid_results)}\nmedian = {advance.median():.1f} min",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    _save_figure(fig, base / "figure_1d_defrost_advance_distribution")


def _boundary_limited(values: pd.DataFrame) -> pd.Series:
    """Identify valid optima constrained by any observed search boundary."""
    left_support = values.get(
        "left_support_removed", pd.Series(False, index=values.index)
    ).eq(True)
    left_integration = values.get(
        "left_integration_removed", pd.Series(False, index=values.index)
    ).eq(True)
    return values["valid"].eq(True) & (
        values["minimum_location"].ne("interior") | left_support | left_integration
    )


def _optimum_classification_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize optimum status over the complete observed-defrost cohort."""
    values = results.loc[results["complete_observed_defrost"].fillna(False)].copy()
    valid = values["valid"].fillna(False).astype(bool)
    boundary = _boundary_limited(values)
    counts = [int((valid & ~boundary).sum()), int(boundary.sum()), int((~valid).sum())]
    total = len(values)
    return pd.DataFrame(
        {
            "category": [
                "Interior optimum",
                "Boundary-limited optimum",
                "No valid estimate",
            ],
            "count": counts,
            "fraction": [count / total for count in counts],
        }
    )


def plot_dynamic_optimal_time_distribution(  # noqa: C901
    results: pd.DataFrame,
    segments: pd.DataFrame,
    output: Path,
    *,
    comparison: str = "water_unit_rb",
) -> None:
    """Compare water/unit cost optima with the causal RB trigger by cycle."""
    if comparison not in {"water_unit_rb", "unit_rb", "water_rb"}:
        raise ValueError(f"unknown comparison: {comparison}")
    values = results.loc[
        results["complete_observed_defrost"].fillna(False)
        & results["valid"].fillna(False)
    ].copy()
    values["minutes_from_stable"] = pd.to_numeric(
        values["minutes_from_stable"], errors="coerce"
    )
    has_unit = "t_star_unit" in values
    if has_unit:
        values["unit_minutes"] = (
            pd.to_datetime(values["t_star_unit"], errors="coerce")
            - pd.to_datetime(values["t_heating_stable"], errors="coerce")
        ).dt.total_seconds() / 60
        comparable = values.get(
            "heat_basis_comparable", pd.Series(True, index=values.index)
        ).eq(True)
        values = values.loc[comparable & values["unit_minutes"].notna()].copy()
    if comparison == "unit_rb" and not has_unit:
        raise ValueError("unit heat optimum is required for the unit/RB figure")
    show_water = comparison != "unit_rb"
    show_unit = has_unit and comparison != "water_rb"
    sort_column = "unit_minutes" if comparison == "unit_rb" else "minutes_from_stable"
    values = values.sort_values(
        [sort_column, "cycle_name"], kind="stable"
    ).reset_index(drop=True)
    boundary = _boundary_limited(values)

    fig, axis = plt.subplots(figsize=(183 / 25.4, max(3.8, 0.10 * len(values) + 1.2)))
    water_y_offset = -0.13 if show_unit else -0.10 if has_unit else 0.0
    unit_y_offset = -0.10 if comparison == "unit_rb" else 0.0
    rb_y_offset = 0.13 if show_water and show_unit else 0.10 if has_unit else 0.0
    for y, row in values.iterrows():
        axis.add_patch(
            Rectangle(
                (0, y - 0.26),
                float(row["actual_minutes_from_stable"]),
                0.52,
                facecolor="#E5E7E9",
                edgecolor="none",
                zorder=0,
            )
        )
        if show_water:
            axis.scatter(
                [float(row["minutes_from_stable"])],
                [y + water_y_offset],
                s=19,
                marker="D",
                facecolors="white" if boundary.loc[y] else "#1F5F99",
                edgecolors="#1F5F99",
                linewidths=0.8,
                zorder=3,
            )
        if show_unit:
            axis.scatter(
                [float(row["unit_minutes"])],
                [y + unit_y_offset],
                s=18,
                marker="s",
                color="#D17A22",
                edgecolors="white",
                linewidths=0.35,
                zorder=4,
            )
        rb_minutes = pd.to_numeric(row.get("rb_minutes_from_stable"), errors="coerce")
        if pd.notna(rb_minutes):
            axis.plot(
                [float(rb_minutes), float(rb_minutes)],
                [y + rb_y_offset - 0.08, y + rb_y_offset + 0.08],
                color="#2E7D5B",
                lw=0.85,
                zorder=5,
                label="_rb_tick",
            )
            axis.plot(
                [float(rb_minutes)],
                [y + rb_y_offset],
                marker="o",
                ms=2.8,
                color="#2E7D5B",
                lw=0,
                zorder=6,
                label="_rb_trigger",
            )
        elif row.get("rb_status") == "right_censored":
            axis.plot(
                [float(row["actual_minutes_from_stable"])],
                [y + rb_y_offset],
                marker=">",
                ms=4,
                markerfacecolor="white",
                markeredgecolor="#2E7D5B",
                lw=0,
                zorder=6,
                label="_rb_censored",
            )

    tick_positions = ([0] if len(values) else []) + list(range(9, len(values), 10))
    axis.set_yticks(tick_positions, [str(position + 1) for position in tick_positions])
    axis.invert_yaxis()
    maximum = pd.to_numeric(
        values["actual_minutes_from_stable"], errors="coerce"
    ).max()
    axis.set_xlim(0, max(float(maximum) * 1.03, 1.0))
    axis.set_xlabel("Heating duration before defrost initiation (min)")
    axis.set_ylabel(
        "Cycle (sorted by unit-reported optimum)"
        if comparison == "unit_rb"
        else "Cycle (sorted by water-side optimum)" if has_unit
        else "Cycle (sorted by optimal defrost time)"
    )
    axis.grid(axis="x", color="#D9DDE1", lw=0.45, alpha=0.75)
    handles = [
        Patch(facecolor="#E5E7E9", edgecolor="none", label="Observed heating period")
    ]
    if show_water:
        handles.append(
            Line2D(
                [], [], marker="D", color="#1F5F99", lw=0, ms=4,
                label="Water-side optimum" if has_unit else "Optimal defrost time",
            )
        )
    if show_unit:
        handles.append(
            Line2D(
                [], [], marker="s", markerfacecolor="#D17A22",
                markeredgecolor="white", color="#D17A22", lw=0, ms=4,
                label="Unit-reported optimum",
            )
        )
    handles.append(
        Line2D(
            [], [], marker="o", color="#2E7D5B", lw=0.9, ls="--", ms=3,
            label="RB defrost time",
        )
    )
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=len(handles),
        fontsize=6.2,
        columnspacing=1.2,
        handlelength=1.6,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.10, top=0.93)
    _save_figure(fig, output)


def plot_optimum_classification_summary(results: pd.DataFrame, output: Path) -> None:
    """Plot why complete observed cycles do or do not yield an interior optimum."""
    summary = _optimum_classification_summary(results)
    fig, axis = plt.subplots(figsize=(3.5, 2.2))
    bars = axis.barh(
        summary["category"],
        100 * summary["fraction"],
        color=["#5B8DB8", "#9C83B8", "#B8BDC2"],
        height=0.55,
    )
    axis.invert_yaxis()
    for bar, row in zip(bars, summary.itertuples(index=False), strict=True):
        axis.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"n = {row.count} ({row.fraction:.1%})",
            va="center",
            fontsize=6.5,
        )
    axis.set_xlabel("Share of complete observed-defrost cycles (%)")
    axis.set_xlim(0, max(100, 100 * summary["fraction"].max() + 22))
    axis.grid(axis="x", color="#D9DDE1", lw=0.45, alpha=0.75)
    _save_figure(fig, output)


def plot_rb_minus_optimal_by_trigger_type(
    results: pd.DataFrame, output: Path
) -> None:
    """Show which RB branches the economic optimum changes most."""
    values = results.loc[
        results["valid"].fillna(False)
        & pd.to_numeric(results["rb_minus_optimal_minutes"], errors="coerce").notna()
    ].copy()
    order = [
        name
        for name in ["Condition1", "Case1", "Case2", "Case3", "Case4", "Case5", "Case7", "Case8"]
        if values["trigger_type"].eq(name).any()
    ]
    fig, axis = plt.subplots(figsize=(3.5, 2.5))
    rng = np.random.default_rng(0)
    for x, name in enumerate(order):
        group = pd.to_numeric(
            values.loc[values["trigger_type"].eq(name), "rb_minus_optimal_minutes"],
            errors="coerce",
        ).dropna()
        jitter = rng.uniform(-0.10, 0.10, len(group))
        axis.scatter(
            x + jitter,
            group,
            s=14,
            color="#2E7D5B",
            alpha=0.60,
            edgecolor="white",
            linewidth=0.3,
            zorder=2,
        )
        axis.plot([x - 0.20, x + 0.20], [group.median()] * 2, color="#174B38", lw=1.7)
        axis.text(x, 0.98, f"n={len(group)}", transform=axis.get_xaxis_transform(), ha="center", va="top")
    axis.axhline(0, color="#777777", lw=0.8, ls="--", zorder=0)
    axis.set_xticks(range(len(order)), order)
    axis.set_ylabel(r"RB timing minus cost optimum, $t_{RB}-t^*$ (min)")
    axis.set_xlabel("First RB trigger")
    fig.tight_layout()
    _save_figure(fig, output)


def plot_diagnostic_dynamic_optimal_time_distribution(
    results: pd.DataFrame, segments: pd.DataFrame, output: Path
) -> None:
    """Plot observed preparation lengths, disconnected windows, optima and failures."""
    values = results.loc[results["complete_observed_defrost"].fillna(False)].copy()
    values["t_star"] = pd.to_datetime(values["t_star"], errors="coerce")
    values["valid"] = values["valid"].fillna(False).astype(bool)
    values["minutes_from_stable"] = pd.to_numeric(
        values["minutes_from_stable"], errors="coerce"
    )
    left_support_removed = values.get(
        "left_support_removed", pd.Series(False, index=values.index)
    ).eq(True)
    left_integration_removed = values.get(
        "left_integration_removed", pd.Series(False, index=values.index)
    ).eq(True)
    limited = values["valid"] & (
        values["minimum_location"].isin(
            {"right_support_limited", "right_integration_limited"}
        )
        | left_support_removed
        | left_integration_removed
    )
    values["_order"] = np.select(
        [
            values["valid"] & ~limited,
            limited,
        ],
        [0, 1],
        default=2,
    )
    values = values.sort_values(
        ["_order", "minutes_from_stable", "cycle_name"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    height = max(4.2, 0.14 * len(values) + 1.0)
    fig, axis = plt.subplots(figsize=(183 / 25.4, height))
    segment_values = segments.copy()
    if "relative_regret_threshold" not in segment_values:
        segment_values["relative_regret_threshold"] = 0.05
    for column in ("segment_start", "segment_end"):
        if column in segment_values:
            segment_values[column] = pd.to_datetime(
                segment_values[column], errors="coerce"
            )
    for y, row in values.iterrows():
        if not row["valid"]:
            transform = transforms.blended_transform_factory(axis.transAxes, axis.transData)
            axis.plot(
                [-0.025],
                [y],
                marker="x",
                ms=4.2,
                mew=0.9,
                color="#8A8F94",
                linestyle="none",
                transform=transform,
                clip_on=False,
                label="_failed_status",
            )
            continue
        observed = float(row["actual_minutes_from_stable"])
        axis.add_patch(
            Rectangle(
                (0.0, y - 0.34),
                observed,
                0.68,
                facecolor="#EEF0F2",
                edgecolor="none",
                zorder=0,
            )
        )
        stable = pd.Timestamp(row["t_heating_stable"])
        cycle_segments = segment_values.loc[
            segment_values["cycle_name"].eq(row["cycle_name"])
        ]
        for threshold, color, width, zorder in (
            (0.05, "#77A9D4", 5.0, 2),
            (0.01, "#7B5AA6", 2.2, 3),
        ):
            threshold_segments = cycle_segments.loc[
                cycle_segments["relative_regret_threshold"].eq(threshold)
            ]
            for segment in threshold_segments.itertuples(index=False):
                left = (
                    pd.Timestamp(segment.segment_start) - stable
                ).total_seconds() / 60
                right = (
                    pd.Timestamp(segment.segment_end) - stable
                ).total_seconds() / 60
                axis.plot(
                    [left, right],
                    [y, y],
                    color=color,
                    lw=width,
                    solid_capstyle="round",
                    zorder=zorder,
                )
        axis.plot(
            [observed, observed],
            [y - 0.25, y + 0.25],
            color="#555A60",
            lw=0.8,
            label="_actual_preparation",
            zorder=3,
        )
        location = str(row["minimum_location"])
        marker = {
            "interior": "D",
            "left_boundary": "<",
            "right_observed": ">",
            "right_support_limited": ">",
            "right_integration_limited": ">",
        }[location]
        row_left_support = bool(row.get("left_support_removed", False))
        row_left_integration = bool(row.get("left_integration_removed", False))
        pe_limited = row_left_support or location == "right_support_limited"
        integration_limited = (
            row_left_integration or location == "right_integration_limited"
        )
        hollow = pe_limited or integration_limited
        edge_color = "#C66A00" if integration_limited else "#1F5F99"
        axis.scatter(
            [float(row["minutes_from_stable"])],
            [y],
            s=22,
            marker=marker,
            facecolors="white" if hollow else "#1F5F99",
            edgecolors=edge_color,
            linewidths=0.9,
            zorder=4,
        )
    axis.set_yticks(range(len(values)), values["cycle_name"], fontsize=5.5)
    axis.invert_yaxis()
    maximum = pd.to_numeric(
        values.loc[values["valid"], "actual_minutes_from_stable"], errors="coerce"
    ).max()
    axis.set_xlim(0, max(float(maximum) * 1.03, 1.0))
    axis.set_xlabel("Time from stable heating start (min)")
    axis.set_ylabel("Complete observed-defrost cycle")
    axis.grid(axis="x", color="#D9DDE1", lw=0.45, alpha=0.75)
    axis.legend(
        handles=[
            Patch(facecolor="#EEF0F2", edgecolor="none", label="Observed length"),
            Line2D([], [], color="#555A60", lw=0.8, label="Observed preparation"),
            Line2D(
                [],
                [],
                marker="x",
                color="#8A8F94",
                lw=0,
                ms=4,
                label="Failed / no estimate",
            ),
            Line2D([], [], color="#77A9D4", lw=5, label="5% near-optimal segment"),
            Line2D([], [], color="#7B5AA6", lw=2.2, label="1% near-optimal segment"),
            Line2D([], [], marker="D", color="#1F5F99", lw=0, ms=4, label="Interior"),
            Line2D(
                [],
                [],
                marker="<",
                markerfacecolor="#1F5F99",
                markeredgecolor="#1F5F99",
                color="none",
                ms=4,
                label="Left",
            ),
            Line2D(
                [],
                [],
                marker=">",
                markerfacecolor="#1F5F99",
                markeredgecolor="#1F5F99",
                color="none",
                ms=4,
                label="Observed right",
            ),
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="white",
                markeredgecolor="#1F5F99",
                color="none",
                ms=4,
                label="Pe-support-limited search",
            ),
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="white",
                markeredgecolor="#C66A00",
                color="none",
                ms=4,
                label="Integration-limited search",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=5,
        fontsize=5.7,
        columnspacing=1.0,
        handlelength=1.5,
    )
    fig.subplots_adjust(left=0.24, right=0.985, bottom=0.08, top=0.90)
    _save_figure(fig, output)


def plot_heat_quantity_ratio_progress(progress: pd.DataFrame, output: Path) -> None:
    """Plot cycle trajectories and the cohort median heat-quantity ratio."""
    fig, axis = plt.subplots(figsize=(89 / 25.4, 72 / 25.4))
    for _, cycle in progress.groupby("cycle_name", sort=False):
        axis.plot(
            cycle["progress_midpoint"],
            cycle["unit_to_water_heat_ratio"],
            color="0.80",
            lw=0.55,
            alpha=0.65,
            zorder=1,
        )
    summary = progress.groupby("progress_midpoint")["unit_to_water_heat_ratio"].agg(
        median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
    )
    axis.fill_between(
        summary.index,
        summary["q25"],
        summary["q75"],
        color="#77A9D4",
        alpha=0.35,
        lw=0,
        label="IQR",
        zorder=2,
    )
    axis.plot(
        summary.index,
        summary["median"],
        color="#1F5F99",
        lw=1.5,
        label="Median",
        zorder=3,
    )
    axis.axhline(1, color="0.25", lw=0.7, ls="--")
    axis.set(xlim=(0, 1), xlabel="Frosting progress to observed preparation", ylabel=r"$Q_{unit}/Q_{water}$")
    axis.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, output)


def plot_heat_basis_optimum_comparison(results: pd.DataFrame, output: Path) -> None:
    """Compare cycle optima under the water and canonical unit heat quantities."""
    values = results.loc[
        results["valid"].eq(True) & results["heat_basis_comparable"].eq(True)
    ].copy()
    values["water_minutes"] = (
        pd.to_datetime(values["t_star_water"])
        - pd.to_datetime(values["t_heating_stable"])
    ).dt.total_seconds() / 60
    values["unit_minutes"] = (
        pd.to_datetime(values["t_star_unit"])
        - pd.to_datetime(values["t_heating_stable"])
    ).dt.total_seconds() / 60
    limit = float(values[["water_minutes", "unit_minutes"]].max().max())
    lower = float(values[["water_minutes", "unit_minutes"]].min().min())
    padding = max((limit - lower) * 0.06, 1.0)
    bounds = (max(0.0, lower - padding), limit + padding)
    fig, axis = plt.subplots(figsize=(89 / 25.4, 72 / 25.4))
    axis.plot(bounds, bounds, color="0.45", lw=0.8, ls="--", zorder=1)
    axis.scatter(
        values["water_minutes"], values["unit_minutes"], s=17, color="#1F5F99", zorder=2
    )
    within_ten = values["heat_basis_abs_delta_minutes"].le(10).mean()
    axis.text(
        0.04,
        0.96,
        f"n = {len(values)}\nmedian |Δt*| = {values['heat_basis_abs_delta_minutes'].median():.1f} min\nwithin 10 min = {within_ten:.1%}",
        transform=axis.transAxes,
        va="top",
        fontsize=6.5,
    )
    axis.set(
        xlim=bounds,
        ylim=bounds,
        aspect="equal",
        xlabel=r"Water-side optimum $t^*$ (min)",
        ylabel=r"Unit-reported optimum $t^*$ (min)",
    )
    fig.tight_layout()
    _save_figure(fig, output)


def analyze(dataset_root: Path, output_root: Path) -> None:  # noqa: C901
    loader = DatasetLoader(dataset_root)
    catalog = loader.list_cycles().sort_values(["experiment_id", "start_time"], kind="stable")
    records = {row["cycle_name"]: row for _, row in catalog.iterrows()}
    ordered = list(catalog["cycle_name"].astype(str))
    next_cycle = following_cycle_names(catalog)

    frames = {name: _raw(loader, name) for name in ordered}
    rb_triggers = _rb_trigger_table(catalog, frames)
    anchors: dict[str, dict[str, object]] = {}
    anchor_rows: list[dict[str, object]] = []
    for name in ordered:
        start = _timestamp(records[name].get("stable_heating_start"))
        anchors[name] = _anchor(frames[name], start)
        anchor_rows.append({"cycle_name": name, **anchors[name]})
    anchor_table = pd.DataFrame(anchor_rows)
    clean_cop = anchor_table.loc[anchor_table["valid"], "cop_clean"].median()
    if not np.isfinite(clean_cop) or clean_cop <= 0:
        raise ValueError("No positive clean-anchor COP is available")
    lambda_q = 1 / float(clean_cop)

    ticket_rows: list[dict[str, object]] = []
    for name in ordered:
        row = records[name]
        reason = ""
        following = next_cycle.get(name)
        defrost_start = _timestamp(row.get("defrost_start"))
        defrost_end = _timestamp(row.get("defrost_end"))
        reason = _ticket_boundary_reason(row, following)
        if not reason and (not anchors[name]["valid"] or not anchors[following]["valid"]):
            reason = "invalid_clean_anchor"
        if reason:
            ticket_rows.append({"cycle_name": name, "valid": False, "invalid_reason": reason})
            continue

        assert following is not None and defrost_start is not None and defrost_end is not None
        next_stable = _timestamp(records[following].get("stable_heating_start"))
        assert next_stable is not None
        event = pd.concat([frames[name], frames[following]], ignore_index=True).sort_values(
            "timestamp"
        )
        recovery_search = event.loc[event["timestamp"].between(defrost_end, next_stable)]
        recovery = find_recovery_time(
            recovery_search["timestamp"],
            recovery_search["q_heating_kw"],
            reference_kw=float(anchors[following]["q_clean_kw"]),
            threshold_fraction=RECOVERY_FRACTION,
            continuous_seconds=RECOVERY_SECONDS,
        )
        if recovery is None:
            ticket_rows.append(
                {"cycle_name": name, "valid": False, "invalid_reason": "recovery_not_observed"}
            )
            continue
        event = event.loc[event["timestamp"].between(defrost_start, recovery)].copy()
        event["q_reference_kw"] = _reference_kw(
            event["timestamp"],
            pd.Timestamp(anchors[name]["anchor_start"]),
            float(anchors[name]["q_clean_kw"]),
            pd.Timestamp(anchors[following]["anchor_start"]),
            float(anchors[following]["q_clean_kw"]),
        )
        event["thermal_shortfall_kw"] = np.maximum(
            event["q_reference_kw"] - event["q_heating_kw"], 0.0
        )
        defrost_event = event.loc[event["timestamp"].between(defrost_start, defrost_end)]
        recovery_event = event.loc[event["timestamp"].between(defrost_end, recovery)]
        defrost_electricity, defrost_power_coverage = integrate_energy_kwh(
            defrost_event["timestamp"], defrost_event["power_total"]
        )
        recovery_electricity, recovery_power_coverage = integrate_energy_kwh(
            recovery_event["timestamp"], recovery_event["power_total"]
        )
        electricity = defrost_electricity + recovery_electricity
        assert np.isclose(electricity, defrost_electricity + recovery_electricity)
        shortfall, heat_coverage = integrate_energy_kwh(
            event["timestamp"], event["thermal_shortfall_kw"]
        )
        signed_transient_heat, signed_heat_coverage = integrate_energy_kwh(
            event["timestamp"], event["q_heating_kw"]
        )
        power_coverage = min(defrost_power_coverage, recovery_power_coverage)
        valid = power_coverage >= MINIMUM_INTEGRATION_COVERAGE
        ticket_rows.append(
            {
                "cycle_name": name,
                "defrost_start": defrost_start,
                "recovery_stable": recovery,
                "electricity_kwh": electricity,
                "defrost_electricity_kwh": defrost_electricity,
                "recovery_electricity_kwh": recovery_electricity,
                "measured_signed_transient_user_heat_kwh": signed_transient_heat,
                "assumed_primary_user_heat_kwh": 0.0,
                "thermal_shortfall_kwh": shortfall,
                "equivalent_cost_kwh": electricity + lambda_q * shortfall,
                "duration_minutes": (recovery - defrost_start).total_seconds() / 60,
                "integration_coverage": power_coverage,
                "defrost_electricity_coverage": defrost_power_coverage,
                "recovery_electricity_coverage": recovery_power_coverage,
                "thermal_shortfall_coverage": heat_coverage,
                "signed_transient_user_heat_coverage": signed_heat_coverage,
                "valid": valid,
                "invalid_reason": "" if valid else "event_integration_coverage_below_95pct",
            }
        )
    tickets = pd.DataFrame(ticket_rows)
    valid_tickets = tickets.loc[tickets["valid"]]
    if valid_tickets.empty:
        raise ValueError("No valid empirical defrost tickets are available")
    mean_ticket_cost = float(valid_tickets["equivalent_cost_kwh"].mean())
    mean_ticket_hours = float(valid_tickets["duration_minutes"].mean() / 60)
    median_ticket_cost = float(valid_tickets["equivalent_cost_kwh"].median())
    median_ticket_hours = float(valid_tickets["duration_minutes"].median() / 60)
    mean_ticket_electricity = float(valid_tickets["electricity_kwh"].mean())
    median_ticket_electricity = float(valid_tickets["electricity_kwh"].median())
    mean_defrost_electricity = float(valid_tickets["defrost_electricity_kwh"].mean())
    mean_recovery_electricity = float(valid_tickets["recovery_electricity_kwh"].mean())
    pe_folds = _read_pe_folds(
        output_root / "证据" / "pe_quadratic_ridge_fold_coefficients.csv"
    )

    result_rows: list[dict[str, object]] = []
    curves: list[pd.DataFrame] = []
    band_rows: list[dict[str, object]] = []
    segment_tables: list[pd.DataFrame] = []
    progress_tables: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for name in ordered:
        row = records[name]
        following = next_cycle.get(name)
        stable = _timestamp(row.get("stable_heating_start"))
        actual = _timestamp(row.get("defrost_preparation_start"))
        complete_observed_defrost = _observed_cycle_boundary_reason(row) == ""
        reason = _preparation_candidate_boundary_reason(row)
        if not reason and not anchors[name]["valid"]:
            reason = "invalid_clean_anchor"
        experiment_id = str(row.get("experiment_id"))
        if not reason and experiment_id not in pe_folds.index:
            reason = "missing_experiment_loeo_fold"
        if not reason and (actual - stable) < pd.Timedelta(minutes=MINIMUM_HEATING_MINUTES):
            reason = "heating_interval_shorter_than_10min"
        if reason:
            result_rows.append(
                {
                    "cycle_name": name,
                    "experiment_id": experiment_id,
                    "complete_observed_defrost": complete_observed_defrost,
                    "t_heating_stable": stable,
                    "t_actual_preparation": actual,
                    "t_actual_defrost": _timestamp(row.get("defrost_start")),
                    "valid": False,
                    "primary_analysis": False,
                    "failure_reason": reason,
                    "invalid_reason": reason,
                }
            )
            audit_rows.append({"cycle_name": name, "included": False, "reason": reason})
            continue

        assert stable is not None and actual is not None
        candidate_end = actual
        candidate_end_source = "observed_preparation"
        is_censored = False
        reference_end = (
            _timestamp(records[following].get("stable_heating_start"))
            if following is not None and anchors[following]["valid"]
            else candidate_end
        )
        assert reference_end is not None
        q_end_kw = (
            float(anchors[following]["q_clean_kw"])
            if following is not None and anchors[following]["valid"]
            else float(anchors[name]["q_clean_kw"])
        )
        cohort_tier = "complete_observed_cycle"
        candidates = _candidate_costs(
            frames[name],
            stable_start=stable,
            candidate_end=candidate_end,
            q_start_kw=float(anchors[name]["q_clean_kw"]),
            next_stable_start=reference_end,
            q_end_kw=q_end_kw,
            lambda_q=lambda_q,
        )
        if candidates.empty:
            reason = "no_candidate_rows"
            result_rows.append(
                {
                    "cycle_name": name,
                    "experiment_id": experiment_id,
                    "complete_observed_defrost": complete_observed_defrost,
                    "t_heating_stable": stable,
                    "t_actual_preparation": actual,
                    "t_actual_defrost": _timestamp(row.get("defrost_start")),
                    "valid": False,
                    "primary_analysis": False,
                    "failure_reason": reason,
                    "invalid_reason": reason,
                }
            )
            audit_rows.append({"cycle_name": name, "included": False, "reason": reason})
            continue
        candidates = _apply_pe_fold(candidates, pe_folds.loc[experiment_id])
        candidates = candidates.sort_values("candidate_time", kind="stable")
        candidates.insert(0, "cycle_name", name)
        candidates["experiment_id"] = experiment_id
        candidates["cohort_tier"] = cohort_tier
        candidates["candidate_end_source"] = candidate_end_source
        candidates["is_censored"] = is_censored
        candidates["actual_preparation_time"] = actual
        if not candidates["optimization_eligible"].any():
            reason = _no_eligible_failure_reason(candidates)
            candidate_audit = _candidate_eligibility_audit(candidates)
            candidates["relative_regret"] = np.nan
            candidates["is_near_optimal"] = False
            candidates["minimum_location"] = np.nan
            curves.append(candidates)
            result_rows.append(
                {
                    "cycle_name": name,
                    "experiment_id": experiment_id,
                    "complete_observed_defrost": complete_observed_defrost,
                    "t_heating_stable": stable,
                    "t_actual_preparation": actual,
                    "t_actual_defrost": _timestamp(row.get("defrost_start")),
                    "candidate_end": candidate_end,
                    "valid": False,
                    "primary_analysis": False,
                    "failure_reason": reason,
                    "invalid_reason": reason,
                    "candidate_count": len(candidates),
                    **candidate_audit,
                }
            )
            audit_rows.append({"cycle_name": name, "included": False, "reason": reason})
            continue
        curve, optimum, unit_optimum = _compare_heat_bases(candidates, candidate_end)
        _, fixed_optimum = _fixed_ticket_optimum(
            candidates, mean_ticket_electricity, candidate_end
        )
        legacy_candidates = candidates.copy()
        legacy_candidates.loc[
            legacy_candidates["integration_coverage"].lt(MINIMUM_INTEGRATION_COVERAGE),
            "heating_cost_kwh",
        ] = np.nan
        legacy_curve, legacy_optimum = optimize_renewal_cost(
            legacy_candidates,
            ticket_cost_kwh=mean_ticket_cost,
            ticket_duration_hours=mean_ticket_hours,
            required_end_time=candidate_end,
        )
        curve["renewal_cost_kw"] = legacy_curve["renewal_cost_kw"].reindex(curve.index)
        curve["relative_regret"] = curve["relative_regret_water"]
        curve["is_near_optimal"] = curve["optimization_eligible"] & curve[
            "relative_regret"
        ].le(0.05)
        curve["minimum_location"] = optimum["minimum_location"]
        curve["actual_preparation_time"] = actual
        curve_segments = _near_optimal_segment_rows(name, curve, fraction=0.05)
        segment_tables.extend(
            [_near_optimal_segment_rows(name, curve, fraction=0.01), curve_segments]
        )
        for fraction in (0.01, 0.02, 0.05, 0.10):
            selected = curve["optimization_eligible"] & curve["relative_regret"].le(
                fraction
            )
            band = curve.loc[selected, "candidate_time"]
            band_rows.append(
                {
                    "cycle_name": name,
                    "relative_regret_threshold": fraction,
                    "band_start": band.min(),
                    "band_end": band.max(),
                    "band_width_minutes": (band.max() - band.min()).total_seconds()
                    / 60,
                    "segment_count": count_true_runs(selected.tolist()),
                }
            )
        curves.append(curve)
        t_star = pd.Timestamp(optimum["candidate_time"])
        t_star_unit = pd.Timestamp(unit_optimum["candidate_time"])
        eligible_heat_basis = curve["optimization_eligible"].fillna(False)
        unit_coverage = float(
            curve.loc[eligible_heat_basis, "unit_heating_coverage"].min()
        )
        heat_basis_comparable = unit_coverage >= MINIMUM_INTEGRATION_COVERAGE
        cost_rank_spearman = curve.loc[
            eligible_heat_basis, ["inverse_cop_water", "inverse_cop_unit"]
        ].corr(method="spearman").iloc[0, 1]
        water_at_unit_regret = float(
            curve.loc[
                curve["candidate_time"].eq(t_star_unit), "relative_regret_water"
            ].iloc[0]
        )
        unit_at_water_regret = float(
            curve.loc[curve["candidate_time"].eq(t_star), "relative_regret_unit"].iloc[0]
        )
        progress, progress_summary = _heat_ratio_progress(
            frames[name], name, stable, actual
        )
        progress_tables.append(progress)
        fixed_ticket_t_star = pd.Timestamp(fixed_optimum["candidate_time"])
        legacy_t_star = pd.Timestamp(legacy_optimum["candidate_time"])
        _, median_ticket_optimum = _fixed_ticket_optimum(
            candidates, median_ticket_electricity, candidate_end
        )
        median_ticket_t_star = pd.Timestamp(median_ticket_optimum["candidate_time"])
        constant_reference_candidates = _candidate_costs(
            frames[name],
            stable_start=stable,
            candidate_end=candidate_end,
            q_start_kw=float(anchors[name]["q_clean_kw"]),
            next_stable_start=reference_end,
            q_end_kw=float(anchors[name]["q_clean_kw"]),
            lambda_q=lambda_q,
        )
        constant_reference_candidates.loc[
            constant_reference_candidates["integration_coverage"].lt(
                MINIMUM_INTEGRATION_COVERAGE
            ),
            "heating_cost_kwh",
        ] = np.nan
        _, constant_reference_optimum = optimize_renewal_cost(
            constant_reference_candidates,
            ticket_cost_kwh=mean_ticket_cost,
            ticket_duration_hours=mean_ticket_hours,
            required_end_time=candidate_end,
        )
        constant_reference_t_star = pd.Timestamp(constant_reference_optimum["candidate_time"])
        near_start = pd.Timestamp(optimum["near_opt_start"])
        near_end = pd.Timestamp(optimum["near_opt_end"])
        t_star_segment = curve_segments.loc[curve_segments["contains_t_star"]].iloc[0]
        pe_supported = curve.loc[curve["pe_supported"]]
        optimization_eligible = curve.loc[curve["optimization_eligible"]]
        candidate_audit = _candidate_eligibility_audit(curve)
        result_rows.append(
            {
                "cycle_name": name,
                "experiment_id": experiment_id,
                "complete_observed_defrost": complete_observed_defrost,
                "t_heating_stable": stable,
                "t_actual_preparation": actual,
                "t_actual_defrost": _timestamp(row.get("defrost_start")),
                "candidate_end": candidate_end,
                "candidate_end_source": candidate_end_source,
                "is_censored": is_censored,
                "cohort_tier": cohort_tier,
                "primary_analysis": not is_censored,
                "t_star": t_star,
                "t_star_water": t_star,
                "t_star_unit": t_star_unit,
                "heat_basis_delta_minutes": (t_star_unit - t_star).total_seconds() / 60,
                "heat_basis_abs_delta_minutes": abs(
                    (t_star_unit - t_star).total_seconds() / 60
                ),
                "heat_basis_cost_rank_spearman": cost_rank_spearman,
                "water_regret_at_unit_optimum": water_at_unit_regret,
                "unit_regret_at_water_optimum": unit_at_water_regret,
                "unit_heating_coverage": unit_coverage,
                "heat_basis_comparable": heat_basis_comparable,
                **progress_summary,
                "fixed_ticket_t_star": fixed_ticket_t_star,
                "dynamic_vs_fixed_ticket_shift_minutes": (
                    t_star - fixed_ticket_t_star
                ).total_seconds()
                / 60,
                "t_star_median_ticket": median_ticket_t_star,
                "median_ticket_shift_minutes": (median_ticket_t_star - t_star).total_seconds() / 60,
                "legacy_time_average_t_star": legacy_t_star,
                "main_vs_legacy_shift_minutes": (legacy_t_star - t_star).total_seconds() / 60,
                "legacy_constant_reference_t_star": constant_reference_t_star,
                "legacy_constant_reference_shift_minutes": (
                    constant_reference_t_star - legacy_t_star
                ).total_seconds()
                / 60,
                "minutes_from_stable": (t_star - stable).total_seconds() / 60,
                "actual_minutes_from_stable": (
                    (actual - stable).total_seconds() / 60 if actual is not None else np.nan
                ),
                "minutes_earlier_than_actual": (
                    (actual - t_star).total_seconds() / 60 if actual is not None else np.nan
                ),
                "inverse_cop_min": optimum["inverse_cop"],
                "cycle_cop_max": optimum["cycle_cop"],
                "t_star_support_status": optimum.get("support_status", np.nan),
                "t_star_pe_supported": bool(optimum.get("pe_supported", False)),
                "t_star_extrapolated": not bool(optimum.get("pe_supported", False)),
                "t_star_pe_extrapolation_distance_mpa_signed": optimum.get(
                    "pe_extrapolation_distance_mpa_signed", np.nan
                ),
                "t_star_pe_extrapolation_distance_mpa_absolute": optimum.get(
                    "pe_extrapolation_distance_mpa_absolute", np.nan
                ),
                "defrost_recovery_electricity_kwh": optimum[
                    "dynamic_ticket_electricity_kwh"
                ],
                "predicted_preparation_defrost_electricity_kwh": optimum[
                    "predicted_preparation_defrost_electricity_kwh"
                ],
                "fixed_recovery_electricity_kwh": FIXED_RECOVERY_ELECTRICITY_KWH,
                "near_opt_start": near_start,
                "near_opt_end": near_end,
                "near_opt_width_minutes": (near_end - near_start).total_seconds() / 60,
                "near_opt_segment_count": len(curve_segments),
                "near_opt_t_star_segment_start": t_star_segment["segment_start"],
                "near_opt_t_star_segment_end": t_star_segment["segment_end"],
                "observed_length_minutes": (actual - stable).total_seconds() / 60,
                "search_length_minutes": (
                    candidate_end - pd.Timestamp(curve["candidate_time"].min())
                ).total_seconds()
                / 60,
                "supported_length_minutes": (
                    pd.Timestamp(pe_supported["candidate_time"].max())
                    - pd.Timestamp(pe_supported["candidate_time"].min())
                ).total_seconds()
                / 60,
                "candidate_count": len(curve),
                **candidate_audit,
                "supported_start": pe_supported["candidate_time"].min(),
                "supported_end": pe_supported["candidate_time"].max(),
                "optimization_eligible_start": optimization_eligible[
                    "candidate_time"
                ].min(),
                "optimization_eligible_end": optimization_eligible[
                    "candidate_time"
                ].max(),
                "left_support_removed": optimum["left_support_removed"],
                "left_integration_removed": optimum["left_integration_removed"],
                "minimum_location": optimum["minimum_location"],
                "valid": True,
                "failure_reason": "",
                "invalid_reason": "",
            }
        )
        audit_rows.append(
            {
                "cycle_name": name,
                "included": True,
                "reason": "",
                "cohort_tier": cohort_tier,
                "candidate_end_source": candidate_end_source,
                "is_censored": is_censored,
            }
        )

    results = pd.DataFrame(result_rows).merge(
        rb_triggers[["cycle_name", "t_RB", "rb_status", "trigger_type", "case"]],
        on="cycle_name",
        how="left",
        validate="one_to_one",
    )
    results["rb_minutes_from_stable"] = (
        pd.to_datetime(results["t_RB"], errors="coerce")
        - pd.to_datetime(results["t_heating_stable"], errors="coerce")
    ).dt.total_seconds() / 60
    results["rb_minus_optimal_minutes"] = (
        pd.to_datetime(results["t_RB"], errors="coerce")
        - pd.to_datetime(results["t_star"], errors="coerce")
    ).dt.total_seconds() / 60
    rb_triggers = rb_triggers.merge(
        results[["cycle_name", "t_heating_stable", "t_star", "rb_minutes_from_stable", "rb_minus_optimal_minutes"]],
        on="cycle_name",
        how="left",
        validate="one_to_one",
    )
    candidate_curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    rb_costs = _rb_candidate_costs(results, candidate_curves)
    results = results.merge(rb_costs, on="cycle_name", how="left", validate="one_to_one")
    rb_triggers = rb_triggers.merge(
        rb_costs, on="cycle_name", how="left", validate="one_to_one"
    )
    band_sensitivity = pd.DataFrame(band_rows)
    near_optimal_segments = (
        pd.concat(segment_tables, ignore_index=True)
        if segment_tables
        else pd.DataFrame(
            columns=[
                "cycle_name",
                "relative_regret_threshold",
                "segment_index",
                "segment_start",
                "segment_end",
                "segment_width_minutes",
                "contains_t_star",
            ]
        )
    )
    heat_ratio_progress = (
        pd.concat(progress_tables, ignore_index=True) if progress_tables else pd.DataFrame()
    )
    source = output_root / "源数据"
    figures = output_root / "图表"
    source.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    anchor_table.to_csv(source / "clean_anchor_summary.csv", index=False)
    tickets.to_csv(source / "defrost_ticket_events.csv", index=False)
    rb_triggers.to_csv(source / "rb_defrost_triggers.csv", index=False)
    results.to_csv(source / "cycle_optimal_points.csv", index=False)
    candidate_curves.to_parquet(source / "candidate_cost_curves.parquet", index=False)
    heat_ratio_progress.to_csv(source / "heat_quantity_ratio_progress.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(source / "cohort_audit.csv", index=False)
    band_sensitivity.to_csv(source / "near_optimal_band_sensitivity.csv", index=False)
    near_optimal_segments.to_csv(source / "near_optimal_segments.csv", index=False)
    pd.DataFrame(
        [
            {
                "clean_cop": clean_cop,
                "lambda_q": lambda_q,
                "valid_ticket_count": len(valid_tickets),
                "mean_ticket_cost_kwh_equivalent": mean_ticket_cost,
                "median_ticket_cost_kwh_equivalent": median_ticket_cost,
                "mean_ticket_electricity_kwh": mean_ticket_electricity,
                "median_ticket_electricity_kwh": median_ticket_electricity,
                "mean_defrost_electricity_kwh": mean_defrost_electricity,
                "mean_recovery_electricity_kwh": mean_recovery_electricity,
                "fixed_recovery_plugin_electricity_kwh": FIXED_RECOVERY_ELECTRICITY_KWH,
                "dynamic_ticket_protocol": "experiment_LOEO_quadratic_Ridge_Pe_plus_full_data_fixed_recovery",
                "mean_ticket_duration_minutes": mean_ticket_hours * 60,
                "median_ticket_duration_minutes": median_ticket_hours * 60,
            }
        ]
    ).to_csv(source / "empirical_policy_summary.csv", index=False)

    valid_results = results.loc[results["valid"]].copy()
    if valid_results.empty:
        raise ValueError("No valid cycle optimum is available")
    primary_results = valid_results.loc[valid_results["primary_analysis"]].copy()
    with PdfPages(figures / "cycle_atlas.pdf") as atlas:
        for _, result in valid_results.iterrows():
            name = str(result["cycle_name"])
            if bool(result["is_censored"]):
                reference_end = pd.Timestamp(result["candidate_end"])
                q_end_kw = float(anchors[name]["q_clean_kw"])
            else:
                following = next_cycle.get(name)
                reference_end = (
                    _timestamp(records[following].get("stable_heating_start"))
                    if following is not None and anchors[following]["valid"]
                    else pd.Timestamp(result["candidate_end"])
                )
                assert reference_end is not None
                q_end_kw = (
                    float(anchors[following]["q_clean_kw"])
                    if following is not None and anchors[following]["valid"]
                    else float(anchors[name]["q_clean_kw"])
                )
            _plot_cycle(
                frames[name],
                result,
                candidate_curves.loc[candidate_curves["cycle_name"].eq(name)],
                q_start_kw=float(anchors[name]["q_clean_kw"]),
                next_stable_start=reference_end,
                q_end_kw=q_end_kw,
                atlas=atlas,
            )

    interior = primary_results.loc[primary_results["minimum_location"].eq("interior")]
    pool = interior if not interior.empty else primary_results
    rb_pool = pool.loc[pool["rb_status"].eq("triggered")]
    pool = rb_pool if not rb_pool.empty else pool
    median_advance = pool["minutes_earlier_than_actual"].median()
    representative = pool.iloc[
        (pool["minutes_earlier_than_actual"] - median_advance).abs().argmin()
    ]
    representative_name = str(representative["cycle_name"])
    following = next_cycle.get(representative_name)
    next_stable = (
        _timestamp(records[following].get("stable_heating_start"))
        if following is not None and anchors[following]["valid"]
        else pd.Timestamp(representative["candidate_end"])
    )
    assert next_stable is not None
    _plot_main(
        frames[representative_name],
        representative,
        candidate_curves.loc[candidate_curves["cycle_name"].eq(representative_name)],
        primary_results,
        tickets,
        q_start_kw=float(anchors[representative_name]["q_clean_kw"]),
        next_stable_start=next_stable,
        q_end_kw=(
            float(anchors[following]["q_clean_kw"])
            if following is not None and anchors[following]["valid"]
            else float(anchors[representative_name]["q_clean_kw"])
        ),
        output=figures / "figure_1_empirical_optimal_defrost",
    )
    plot_dynamic_optimal_time_distribution(
        results,
        near_optimal_segments,
        figures / "figure_1e_dynamic_optimal_time_distribution",
    )
    plot_dynamic_optimal_time_distribution(
        results,
        near_optimal_segments,
        figures / "figure_1e_unit_rb_optimal_time_distribution",
        comparison="unit_rb",
    )
    plot_dynamic_optimal_time_distribution(
        results,
        near_optimal_segments,
        figures / "figure_1e_water_rb_optimal_time_distribution",
        comparison="water_rb",
    )
    plot_optimum_classification_summary(
        results,
        figures / "figure_1f_optimum_classification_summary",
    )
    plot_rb_minus_optimal_by_trigger_type(
        results,
        figures / "figure_1g_rb_minus_optimal_by_trigger_type",
    )
    plot_diagnostic_dynamic_optimal_time_distribution(
        results,
        near_optimal_segments,
        figures / "figure_diagnostic_dynamic_optimal_time_distribution",
    )
    plot_heat_quantity_ratio_progress(
        heat_ratio_progress, figures / "figure_heat_quantity_ratio_progress"
    )
    plot_heat_basis_optimum_comparison(
        results, figures / "figure_heat_basis_optimum_comparison"
    )

    counts = primary_results["minimum_location"].value_counts()
    complete_results = results.loc[results["complete_observed_defrost"].fillna(False)]
    failed_complete = complete_results.loc[~complete_results["valid"].fillna(False)]
    optimum_status = _optimum_classification_summary(results).set_index("category")
    fixed_ticket_shift = primary_results["dynamic_vs_fixed_ticket_shift_minutes"].abs()
    report_statistics = _dynamic_report_statistics(primary_results, candidate_curves)
    candidate_count = int(report_statistics["candidate_count"])
    pe_supported_count = int(report_statistics["pe_supported_candidate_count"])
    pe_support_summary = _pe_support_summary(candidate_curves)
    pe_extrapolated_count = pe_support_summary["extrapolated_count"]
    pe_missing_count = pe_support_summary["missing_count"]
    integration_eligible_count = int(
        report_statistics["integration_eligible_candidate_count"]
    )
    optimization_eligible_count = int(
        report_statistics["optimization_eligible_candidate_count"]
    )
    candidate_gap_marker = candidate_curves.get(
        "candidate_in_interpolated_gap",
        pd.Series(False, index=candidate_curves.index),
    ).fillna(False).astype(bool)
    candidate_gap_marker_count = int(candidate_gap_marker.sum())
    candidate_gap_marker_cycles = int(
        candidate_curves.loc[candidate_gap_marker, "cycle_name"].nunique()
    )
    rb_triggered = rb_triggers["rb_status"].eq("triggered")
    rb_complete_triggered = complete_results["rb_status"].eq("triggered")
    rb_comparable = primary_results["rb_minus_optimal_minutes"].dropna()
    rb_cost_regret = primary_results["rb_relative_regret"].dropna()
    rb_group_summary = primary_results.dropna(
        subset=["rb_minus_optimal_minutes"]
    ).groupby("trigger_type")["rb_minus_optimal_minutes"].agg(["count", "median"])
    rb_group_text = ", ".join(
        f"{name} n={int(row['count'])}, median={row['median']:.1f} min"
        for name, row in rb_group_summary.iterrows()
    )
    rb_type_counts = ", ".join(
        f"{name} {count}"
        for name, count in rb_triggers.loc[rb_triggered, "trigger_type"].value_counts().items()
    )
    heat_basis = primary_results.loc[
        primary_results["heat_basis_comparable"].eq(True)
    ]
    heat_ratio_pairs = primary_results.dropna(
        subset=["heat_ratio_early", "heat_ratio_late"]
    )
    ratio_test = wilcoxon(
        heat_ratio_pairs["heat_ratio_early"],
        heat_ratio_pairs["heat_ratio_late"],
        alternative="two-sided",
    )
    ratio_late_lower = int(
        heat_ratio_pairs["heat_ratio_late"].lt(heat_ratio_pairs["heat_ratio_early"]).sum()
    )
    offset_pairs = primary_results.dropna(
        subset=[
            "equivalent_delta_t_offset_early_C",
            "equivalent_delta_t_offset_late_C",
        ]
    )
    summary = rf"""# Pe 动态经验最优除霜准备点

## 结论边界

本分析在完整、已观测除霜循环中搜索除霜准备动作时刻。现有 publication 和 RGB 标签由“实验留一 Pe 二次 Ridge + 固定恢复电耗均值”产生。动态恢复模型是未来改进，不改变当前主分析与标签。

Pe 关系只在真实 preparation 边界训练。把该关系迁移到任意候选时刻 `tau` 是探索性反事实假设，不构成因果最优控制策略。

## 固定方法

- 水侧制热量：`1.161 × water_flow × (water_out_temperature - water_in_temperature)`，单位 kW。
- `Pe(tau)`：在完整循环的有限 Pe 时间点与 `[tau-60 s, tau)` 的 1 s 网格并集上做 time interpolation，再截取严格的 `[tau-60 s, tau)` 窗口取中位数；内部缺口线性插值，若窗口越过有限观测首/尾，则用相邻两点线性外推，并以 `pe_endpoint_extrapolated` 单独标记。
- 候选：稳定制热后 {MINIMUM_HEATING_MINUTES} min 起，按 {CANDIDATE_STEP_MINUTES} min 搜索至实际 `defrost_preparation_start`；精确实际边界始终作为末候选。
- 当前主票价：`K_hat(tau) = b0_LOEO(exp) + b1_LOEO(exp) × Pe(tau) + b2_LOEO(exp) × Pe(tau)^2 + {FIXED_RECOVERY_ELECTRICITY_KWH:.12f}` kWh。最后一项是当前正式采用的恢复电耗均值；动态恢复模型留作未来改进。
- 主目标：`J(tau) = [E_heating(tau) + K_hat(tau)] / Q_U^(water)(tau) = 1/COP_cyc(tau)`；制热电耗与水侧正向供热代理量严格只积分到 `tau`，preparation 不重复积分。
- 口径一致性实验只替换上述分母：主结果及 `inverse_cop` 使用水侧 `Q_U^(water)`，它不等同于房间或建筑负荷；机组口径使用 `cycles_original.heating_capacity`。两者共享完全相同的候选时刻、动态票价与原始 eligibility。机组热量覆盖低于 {MINIMUM_INTEGRATION_COVERAGE:.0%} 的循环标为不可比较，但不改变水侧主结果。
- 结霜进程定义为 stable heating 到已观测 controller preparation 动作的归一化相对时间，分 10 箱后分别对两套热量做 gap-aware 能量积分；这不是真实霜质量。箱内比值为能量比，不对逐秒比值求均值。
- 主 argmin 使用 Pe 有限且积分覆盖合格的候选；Pe 位于 LOEO fold 训练支持域外时，采用同一 LOEO 二次模型延拓，不 clip、不另行插值。`support_status` 与 `pe_extrapolation_distance_mpa_*` 保留为审计标签；延拓段仍进入成本曲线和近优搜索。
- 能量积分对电功率、用户侧热量、机组热量及等效功率均启用内部缺口线性桥接；窗口越过有限观测首/尾时，用最近两点线性外推，并将 `candidate_in_extrapolated_endpoint=True` 单独保留 provenance。内部桥接候选在成本图中以灰色空心方块、端点外推候选以紫色空心菱形标记；两者均不改变候选的 argmin eligibility。
- RB baseline：直接回放 `cycles_original` 原始传感器与机内计时器。T3/T4/Twout 已是 °C，T3o 按原值/10 转为 °C，DefTim1/2 按原值/60 转为 min。每个循环从 heating start 扫描到 preparation 前一秒，按 timestamp 排序去重并落到 1 s 网格；缺秒不插值，任一所需量缺测时该秒相关条件为 False。
- RB 在线因果窗口：C1 只读取 `[t-600 s,t-50 s]`；C2/C7 的“持续20 s”严格解释为尾随 20 个 1 s 样本 `[t-19 s,t]` 全部成立，T1 时间门槛只在当前秒判断。首次触发立即停止；同秒优先级为 Condition1 > Case1–5 > Case7 > Case8；人工强制指令不作为输入。
- RB 成本映射：仅在同循环 `optimization_eligible=True` 的候选中匹配离 `t_RB` 最近者，且时间差必须 ≤0.51 min；否则 `rb_candidate_time/rb_inverse_cop/rb_relative_regret` 均留空。Pe 支持域外候选若 Pe 有限则采用二次模型延拓；内部缺口桥接候选仍保留其 provenance，不把桥接值当作实测值。

### RB baseline 判据

- $C_1=(T_1>35)\land(T_2\ge6)\land(T_3\le-1)\land\exists\Delta t\in[50,600]:T_3(t-\Delta t)-T_3(t)\ge1$。
- $C_2=(T_2\ge6)\land(T_1\ge T_{{1,lim}})\land[T_3<T_{{3,lim}}]_{{20s}}$。
- $C_7=(T_1\ge30)\land[T_3\le-10\land T_3\le0.8T_4-12]_{{20s}}$；$C_8=(T_1\ge150)$。
- 自动触发：$D(t)=C_1\lor C_2\lor C_7\lor C_8$；时间单位为 min，温度单位为 °C。

| Case | $T_4$ 范围 (°C) | $T_{{1,lim}}$：$T_{{wout}}\ge35$ / $25\le T_{{wout}}<35$ / $T_{{wout}}<25$ (min) | $T_{{3,lim}}$ (°C) |
|---|---|---|---|
| 1 | $T_4\ge-2$ | 40 / 35 / 30 | $T_{{3o}}-3$ |
| 2 | $-5\le T_4<-2$ | 40 / 38 / 33 | $T_{{3o}}-5$ |
| 3 | $-8\le T_4<-5$ | 80 / 60 / 40 | $T_{{3o}}-5$ |
| 4 | $-10\le T_4<-8$ | 90 / 70 / 50 | $T_4-5$ |
| 5 | $T_4<-10$ | 150 / 120 / 90 | $T_4-5$ |

## 当前结果

- complete/observed-defrost 队列：{len(complete_results)}；有效动态最优点：{len(primary_results)}；failed：{len(failed_complete)}。
- Pe 模型支持内：{pe_supported_count}/{candidate_count}（{pe_supported_count / candidate_count:.1%}）；支持域外二次模型延拓：{pe_extrapolated_count}/{candidate_count}（{pe_extrapolated_count / candidate_count:.1%}）；Pe 缺测：{pe_missing_count}/{candidate_count}（{pe_missing_count / candidate_count:.1%}）。
- 积分 eligible（积分覆盖 ≥ {MINIMUM_INTEGRATION_COVERAGE:.0%}）：{integration_eligible_count}/{candidate_count}（{integration_eligible_count / candidate_count:.1%}）。
- 联合 optimization eligible（Pe 有限且积分覆盖 ≥ {MINIMUM_INTEGRATION_COVERAGE:.0%}）：{optimization_eligible_count}/{candidate_count}（{optimization_eligible_count / candidate_count:.1%}）。
- 含至少一条内部缺口桥接或候选时刻 Pe 落在长内部缺口中的候选：{candidate_gap_marker_count}/{candidate_count}（{candidate_gap_marker_count / candidate_count:.1%}），涉及 {candidate_gap_marker_cycles}/{len(primary_results)} 个有效循环；端点外推单独记入 `candidate_in_extrapolated_endpoint`，不与内部桥接混淆。
- minimum location：interior {int(counts.get("interior", 0))}；left boundary {int(counts.get("left_boundary", 0))}；right observed {int(counts.get("right_observed", 0))}；right support-limited {int(counts.get("right_support_limited", 0))}；right integration-limited {int(counts.get("right_integration_limited", 0))}。
- publication 分类（分母为 {len(complete_results)} 个 complete observed-defrost cycles）：内部最优 {int(optimum_status.loc["Interior optimum", "count"])}；边界/搜索域受限 {int(optimum_status.loc["Boundary-limited optimum", "count"])}；无有效估计 {int(optimum_status.loc["No valid estimate", "count"])}。边界受限定义为 valid 且 minimum location 非 interior，或左侧 Pe/积分可行域被删减。
- `t*` 距稳定制热起点：P10 {primary_results["minutes_from_stable"].quantile(0.1):.1f} min；中位数 {primary_results["minutes_from_stable"].median():.1f} min；P90 {primary_results["minutes_from_stable"].quantile(0.9):.1f} min。
- 5% near-opt envelope 中位宽度 {primary_results["near_opt_width_minutes"].median():.1f} min；{int(primary_results["near_opt_segment_count"].gt(1).sum())} 个循环含多个连续段。
- 相对旧固定综合票价，最优点绝对移动：中位数 {fixed_ticket_shift.median():.1f} min；P90 {fixed_ticket_shift.quantile(0.9):.1f} min；最大 {fixed_ticket_shift.max():.1f} min。
- 最大位移来自 `{report_statistics["maximum_shift_cycle"]}`，其 optimization eligible 覆盖仅 {float(report_statistics["maximum_shift_optimization_fraction"]):.1%}；这是支持/积分域受限的敏感值，不代表一般模型变化。
- 完全 Pe-supported 子集（n={int(report_statistics["fully_pe_supported_cycle_count"])}）的绝对位移中位数/P90/最大值为 {float(report_statistics["fully_pe_supported_shift_median"]):.1f}/{float(report_statistics["fully_pe_supported_shift_p90"]):.1f}/{float(report_statistics["fully_pe_supported_shift_maximum"]):.1f} min，作为不额外建模的稳健性参照。
- RB 回放覆盖：全 catalog 为 {int(rb_triggered.sum())}/{len(rb_triggers)} triggered、{int((~rb_triggered).sum())} right-censored；论文 complete/observed-defrost 队列为 {int(rb_complete_triggered.sum())}/{len(complete_results)} triggered。存在 actual preparation 时截止于该动作前；缺 preparation 的 partial cycle 截止于 raw cycle end，并以 `observation_end_source` 明示。所有 right-censored 均不以观测端点伪造触发。首次触发类型：{rb_type_counts}。
- RB 与成本最优可比较循环 n={len(rb_comparable)}；`t_RB - t*` 的 P10/中位数/P90 为 {rb_comparable.quantile(0.1):.1f}/{rb_comparable.median():.1f}/{rb_comparable.quantile(0.9):.1f} min（正值表示 RB 晚于成本最优点）。
- RB 时刻成本可比较 n={len(rb_cost_regret)}；相对最优成本惩罚 `100 × rb_relative_regret` 的中位数/P90 为 {100 * rb_cost_regret.median():.2f}%/{100 * rb_cost_regret.quantile(0.9):.2f}%。按首次触发类型的时间差：{rb_group_text}。
- 热量口径决策可比较 n={len(heat_basis)}；`|t*_unit-t*_water|` 中位数 {heat_basis["heat_basis_abs_delta_minutes"].median():.1f} min，{int(heat_basis["heat_basis_abs_delta_minutes"].le(10).sum())}/{len(heat_basis)} 在 10 min 内；候选成本排序 Spearman 中位数 {heat_basis["heat_basis_cost_rank_spearman"].median():.3f}。水侧最优代入机组口径、机组最优代入水侧口径的交叉 regret 中位数分别为 {100 * heat_basis["unit_regret_at_water_optimum"].median():.2f}%/{100 * heat_basis["water_regret_at_unit_optimum"].median():.2f}%。
- 归一化进程早期/晚期 `Q_unit/Q_water` 中位数为 {heat_ratio_pairs["heat_ratio_early"].median():.3f}/{heat_ratio_pairs["heat_ratio_late"].median():.3f}，循环内 late−early 中位数 {heat_ratio_pairs["heat_ratio_late_minus_early"].median():.3f}；{ratio_late_lower}/{len(heat_ratio_pairs)} 个循环 late 低于 early。以 cycle 为重复的配对 Wilcoxon 双侧检验 p={ratio_test.pvalue:.3g}（n={len(heat_ratio_pairs)}，唯一确认性描述）。
- 同秒有效点折算的等效水侧 ΔT 偏置早期/晚期跨循环中位数为 {offset_pairs["equivalent_delta_t_offset_early_C"].median():.3f}/{offset_pairs["equivalent_delta_t_offset_late_C"].median():.3f} °C，late−early 中位数 {offset_pairs["equivalent_delta_t_offset_late_minus_early_C"].median():.3f} °C。固定 ΔT 零偏不足以单独解释阶段变化；因此只称“机组上报相对于水侧的阶段性偏差”，不指定任一口径为绝对真值。

## 可追溯输出

- `源数据/cycle_optimal_points.csv`：循环级有效/failed、边界、支持覆盖、动态与旧固定票价位移。
- `源数据/rb_defrost_triggers.csv`：每个 catalog cycle 一行的首次 RB 触发/删失状态、触发类型、Case、触发时传感器与计时器，以及相对稳定起点和 `t*` 的时间差。
- `源数据/candidate_cost_curves.parquet`：候选级 Pe、插值/端点外推覆盖、fold、支持状态、动态票价、eligibility 与 regret。
- `源数据/heat_quantity_ratio_progress.csv`：逐循环 10 箱的两套 gap-aware 能量、覆盖、能量比、水温差/流量中位数与等效 ΔT 偏置。
- `源数据/near_optimal_segments.csv`：每个连续 1%/5% near-opt segment 一行，并显式记录 regret 阈值。
- `源数据/clean_anchor_summary.csv`：clean anchor 与 COP。
- `源数据/cohort_audit.csv`：队列纳入审计。
- `源数据/empirical_policy_summary.csv`：当前固定恢复均值、动态票价协议与换算系数。
- `源数据/near_optimal_band_sensitivity.csv`：1%、2%、5%、10% regret 阈值对应的 envelope 宽度与连续段数。
- `图表/figure_1e_dynamic_optimal_time_distribution.*`：仅有效完整循环，按水侧最低点排序；灰条为已观测制热期，蓝色菱形、橙色方形与绿色圆点分别表示水侧最低点、机组侧最低点与 RB 首次触发，绿色空心 `>` 表示观测截止前未触发。
- `图表/figure_1e_unit_rb_optimal_time_distribution.*`：同一队列按机组侧最低点排序，仅保留机组侧最低点与 RB 的比较。
- `图表/figure_1e_water_rb_optimal_time_distribution.*`：同一队列按水侧最低点排序，仅保留水侧最低点与 RB 的比较。
- `图表/figure_1f_optimum_classification_summary.*`：complete observed-defrost 队列中内部最优、边界/搜索域受限及无有效估计的数量与比例。
- `图表/figure_1g_rb_minus_optimal_by_trigger_type.*`：有效且两时刻可比循环中，按首次 RB 触发类型展示 `t_RB-t*` 原始点、中位数与样本量。
- `图表/figure_diagnostic_dynamic_optimal_time_distribution.*`：保留 1%/5% 双阈值、具体限制原因、端点位置与 failed cycle 的完整诊断图，不作为 publication 主图。
- `图表/figure_heat_quantity_ratio_progress.*`：浅灰循环轨迹及中位数/IQR，检验相对热量偏差是否随归一化结霜进程漂移。
- `图表/figure_heat_basis_optimum_comparison.*`：同尺度比较水侧与机组口径的循环最优时刻及 1:1 线。
- `图表/循环图/全部循环/*_J_unit.png`：全部有效循环的机组热量口径逐循环 publication；除成本曲线、5% near-optimal 区间、最优线与最优 RGB 时刻切换到 unit 口径外，其余 COP、水温、RB、时间轴、RGB 与水侧版完全复用，且不覆盖原水侧文件。
- `图表/cycle_atlas.pdf`：全部有效循环紧凑审阅图册；正式逐循环 publication 只写 report 树，不覆盖 Dataset PNG。
"""
    (output_root / "报告.md").write_text(summary, encoding="utf-8")
    figure_qa = f"""# Figure 1 QA contract

- Core conclusion: valid cycle optima have a cross-cycle timing structure that can be compared directly with the causal industrial RB baseline.
- Claim boundary: candidate-time Pe transfer is an exploratory counterfactual assumption, not a causal optimal policy or a fully held-out dynamic ticket.
- Archetype: quantitative cohort distribution plus categorical and rule-branch summaries.
- Backend: Python/matplotlib only.
- Final size: Figure 1e is 183 mm wide with cohort-scaled height; Figures 1f/1g are 89 mm wide.
- Heat-basis figures: `figure_heat_quantity_ratio_progress` and `figure_heat_basis_optimum_comparison` are separate 89 mm Python/matplotlib panels, with editable SVG/PDF text and 300 dpi PNG review copies.
- n definition: {len(complete_results)} complete observed-defrost cycles; {len(primary_results)} valid optima and {len(failed_complete)} failed rows; no censored cycle is reintroduced.
- Statistics: descriptive P10/median/P90 only; no hypothesis test is claimed.
- Source data: `源数据/cycle_optimal_points.csv`, `源数据/rb_defrost_triggers.csv`, `源数据/candidate_cost_curves.parquet`, `源数据/near_optimal_segments.csv`, and `源数据/heat_quantity_ratio_progress.csv`.
- Editable exports: SVG text preserved; PDF uses TrueType fonts; PNG is retained for review. TIFF is generated only at submission if required.
- Publication encoding: Figure 1e includes valid complete observed-defrost cycles only, sorted by the water-side optimum; pale grey bands show the observed heating period, blue diamonds show the water-side optimum, orange squares show the unit-reported optimum, green ticks/circles show causal RB triggers, and hollow green `>` marks RB right-censoring. The unit/RB and water/RB companions use the same cohort, sort by the displayed optimum, and retain only that optimum and RB. Cost panels retain gray `×` markers for Pe-support-domain candidates, gray hollow squares for internal-gap interpolation, and purple hollow diamonds for endpoint continuation; the line and argmin retain explicit provenance. Figure 1f uses the full complete observed-defrost denominator ({len(complete_results)} cycles): interior means valid, minimum_location=interior and no left Pe/integration deletion; boundary-limited means any other valid estimate; no-valid contains the remainder.
- Rule-branch comparison: Figure 1g uses valid cycles with both `t_RB` and `t*`; raw points and median bars show `t_RB-t*` by first-trigger type, with no distribution smoothing or significance test.
- Diagnostic retention: `figure_diagnostic_dynamic_optimal_time_distribution.*` preserves cycle IDs, 1%/5% intervals, explicit endpoint and limitation encodings, and failed rows for audit rather than publication.
- Image integrity: no microscopy or selective image manipulation; sensor panels use unsmoothed original values. Pe uses internal interpolation plus nearest-two-point endpoint extrapolation on the complete cycle time axis before extracting `[tau-60 s, tau)`, and energy curves use the same endpoint rule. All extrapolated candidates retain explicit provenance markers.
- Representative cycle: {representative_name}, selected as the interior-minimum cycle nearest the cohort median preparation advance.
"""
    (output_root / "图表质检.md").write_text(figure_qa, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("output/test/成本函数/其他/经验经济窗口"))
    parser.add_argument("--figure-1e-only", action="store_true")
    args = parser.parse_args()
    if args.figure_1e_only:
        source, figures = args.output / "源数据", args.output / "图表"
        results = pd.read_csv(source / "cycle_optimal_points.csv")
        segments = pd.read_csv(source / "near_optimal_segments.csv")
        plot_dynamic_optimal_time_distribution(
            results, segments, figures / "figure_1e_dynamic_optimal_time_distribution"
        )
        plot_dynamic_optimal_time_distribution(
            results,
            segments,
            figures / "figure_1e_unit_rb_optimal_time_distribution",
            comparison="unit_rb",
        )
        plot_dynamic_optimal_time_distribution(
            results,
            segments,
            figures / "figure_1e_water_rb_optimal_time_distribution",
            comparison="water_rb",
        )
        return
    analyze(args.dataset, args.output)


if __name__ == "__main__":
    main()
