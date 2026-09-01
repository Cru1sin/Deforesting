"""V2.6.8 fixed-nine-minute diagnostic cost recipe and curve assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .cost_curve import validate_recipe
from .fit_v2_6_8 import load_artifacts, predict_independent_targets
from .v2_6_8_data import (
    RAW_COLUMNS,
    build_event_table,
    candidate_cohort,
    event_outcomes,
    pre_action_features,
)
from .v2_6_8_data import (
    sorted_time_slice as _sorted_time_slice,
)
from .v2_6_8_data import (
    window_audit as _window_audit,
)

__all__ = [
    "DEFAULT_RECIPE",
    "_sorted_time_slice",
    "_window_audit",
    "build_candidate_boundaries_v268",
    "build_event_table",
    "calculate",
    "calculate_cycle",
    "candidate_cohort",
    "event_outcomes",
    "finalize_v268_curve",
    "pre_action_features",
]
from .v2_6_8_data import (
    build_candidate_boundaries as build_candidate_boundaries_v268,
)
from .v2_6_8_data import (
    candidate_integral_table as _candidate_integral_table,
)
from .v2_6_8_data import (
    timestamp as _timestamp,
)

Q_MIN_KWH = 0.01

DEFAULT_RECIPE: dict[str, object] = {
    "base_cost": "v2.6.8",
    "version": "v2.6.8",
    "variant": None,
    "label_eligible": False,
    "heat_basis": "water",
    "event_scope": "fixed_post_defrost_9min_to_actual_preparation",
    "heating_start_rule": "fixed_post_defrost_9min",
    "integration_protocol": "strict_causal",
    "state_protocol": "strict_causal",
    "candidate_start_rule": "heating_start_plus_10_minutes",
    "candidate_end_rule": "observed_defrost_preparation_start",
    "candidate_cadence": "1_minute_plus_exact_endpoint",
    "state_window": "[tau-60s,tau)",
    "slope_window": "[tau-5m,tau)",
    "transition_scope": "preparation_defrost_recovery",
    "transition_window": "observed_preparation_defrost_fixed_9m_recovery",
    "transition_provenance": "offline_diagnostic_fixed9_full_transition_target",
    "transition_breakdown": "not_decomposed",
    "decision_rule": "corrected_supported_minimum",
    "heating_energy_model": "measured_total_power",
    "heating_heat_model": "measured_water_heat",
    "transition_energy_model": "ticket_ridge_dynamic8",
    "transition_heat_model": "ticket_ridge_dynamic8",
}


def _long_support_runs(times: pd.Series, selected: pd.Series) -> pd.Series:
    """Retain connected supported runs spanning at least five actual minutes."""
    result = pd.Series(False, index=selected.index)
    chosen = np.flatnonzero(selected.to_numpy(dtype=bool))
    if not chosen.size:
        return result
    parsed = pd.to_datetime(times, errors="coerce")
    breaks = np.flatnonzero(
        (np.diff(chosen) != 1)
        | (
            parsed.iloc[chosen[1:]].to_numpy() - parsed.iloc[chosen[:-1]].to_numpy()
            > np.timedelta64(90, "s")
        )
    )
    starts, ends = np.r_[0, breaks + 1], np.r_[breaks, len(chosen) - 1]
    for left, right in zip(starts, ends, strict=True):
        positions = chosen[left : right + 1]
        if parsed.iloc[positions[-1]] - parsed.iloc[positions[0]] >= pd.Timedelta(minutes=5):
            result.iloc[positions] = True
    return result


def finalize_v268_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """Apply the corrected measurement gate and connected optimum basins."""
    result = curve.sort_values("candidate_time", kind="stable").reset_index(drop=True).copy()
    inverse = pd.to_numeric(result["inverse_cop"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    base = (
        result["heating_measurement_valid"].fillna(False)
        & result["ET_supported"].fillna(False)
        & result["QT_supported"].fillna(False)
        & result["pre_action_window_valid"].fillna(False)
        & result["physical_valid"].fillna(False)
        & inverse.notna()
    )
    result["model_supported"] = result["ET_supported"].fillna(False) & result[
        "QT_supported"
    ].fillna(False)
    result["continuous_support"] = _long_support_runs(result["candidate_time"], base)
    result["optimization_eligible"] = base & result["continuous_support"]
    result["diagnostic_minimum"] = pd.NaT
    for percent in (1, 5):
        result[f"basin_{percent}pct_start"] = pd.NaT
        result[f"basin_{percent}pct_end"] = pd.NaT
        result[f"basin_{percent}pct_width_minutes"] = np.nan
    eligible = result["optimization_eligible"]
    if eligible.any():
        optimum = int(result.index[eligible & inverse.eq(inverse.loc[eligible].min())][0])
        optimum_time = _timestamp(result.loc[optimum, "candidate_time"])
        assert optimum_time is not None
        result["diagnostic_minimum"] = optimum_time
        for percent in (1, 5):
            within = eligible & inverse.le(float(inverse.iloc[optimum]) * (1 + percent / 100))
            left = right = optimum
            while left and bool(within.iloc[left - 1]):
                left -= 1
            while right + 1 < len(result) and bool(within.iloc[right + 1]):
                right += 1
            start, end = result.loc[[left, right], "candidate_time"].map(pd.Timestamp)
            result[f"basin_{percent}pct_start"] = start
            result[f"basin_{percent}pct_end"] = end
            result[f"basin_{percent}pct_width_minutes"] = (end - start).total_seconds() / 60
    for phase in ("preparation", "defrost", "recovery"):
        result[f"{phase}_energy_kwh"] = np.nan
        result[f"{phase}_heat_kwh"] = np.nan
    result["relative_regret"] = np.nan
    if eligible.any():
        result.loc[eligible, "relative_regret"] = (
            inverse.loc[eligible] / inverse.loc[eligible].min() - 1
        )
    result["near_optimal_1pct"] = eligible & result["relative_regret"].le(0.01)
    result["near_optimal_5pct"] = eligible & result["relative_regret"].le(0.05)
    result["recommended_time"] = pd.NaT
    result["hard_label_eligible"] = False
    result["label_eligible"] = False
    return result


def calculate_cycle(
    loader: Any,
    cycle_name: str,
    recipe: Mapping[str, object] | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Execute boundary/cohort → EH → QH → ET → QT → final curve."""
    checked = validate_recipe(DEFAULT_RECIPE if recipe is None else recipe)
    record = loader.get_cycle_record(cycle_name)
    nested = record.get("boundaries")
    boundary_source = nested if isinstance(nested, Mapping) else record
    heating = _timestamp(boundary_source.get("heating_start"))
    preparation = _timestamp(boundary_source.get("defrost_preparation_start"))
    if heating is None or preparation is None:
        raise ValueError(f"V2.6.8 boundaries are incomplete for {cycle_name}")

    boundary = build_candidate_boundaries_v268(
        cycle_name, str(record["experiment_id"]), heating, preparation
    )
    frame = loader.load_cycle_original(cycle_name, columns=list(RAW_COLUMNS)).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
    candidates = [pd.Timestamp(value) for value in boundary["candidate_time"]]
    start = pd.Timestamp(boundary["integration_start"].iloc[0])

    eh_audit = _candidate_integral_table(frame, start, candidates, "power_total")
    eh = pd.DataFrame(
        {"heating_energy_kwh": eh_audit["energy"], "heating_energy_supported": eh_audit["valid"]}
    )
    qh_audit = _candidate_integral_table(frame, start, candidates, "water_heat")
    qh = pd.DataFrame(
        {"heating_heat_kwh": qh_audit["energy"], "heating_heat_supported": qh_audit["valid"]}
    )
    features = pre_action_features(frame, candidates, heating)
    source = dict(artifacts or load_artifacts())
    energy_model = source["models"][str(checked["transition_energy_model"])]["energy"]
    heat_model = source["models"][str(checked["transition_heat_model"])]["heat"]
    transition = predict_independent_targets(
        energy_model, heat_model, features, str(record["experiment_id"])
    )
    et = transition[["transition_energy_kwh", "E_support_distance", "ET_supported"]]
    qt = transition[
        ["transition_heat_kwh", "Q_support_distance", "QT_supported", "model_supported"]
    ]

    curve = pd.concat([boundary, eh, qh, features, et, qt], axis=1)
    curve["heating_measurement_valid"] = eh_audit["valid"] & qh_audit["valid"]
    numerator = curve["heating_energy_kwh"] + curve["transition_energy_kwh"]
    denominator = curve["heating_heat_kwh"] + curve["transition_heat_kwh"]
    curve["physical_valid"] = numerator.gt(0) & denominator.gt(Q_MIN_KWH)
    finite_ratio = np.isfinite(numerator) & np.isfinite(denominator) & denominator.ne(0)
    curve["inverse_cop"] = (numerator / denominator).where(finite_ratio)
    curve["algorithm"] = curve["base_cost"] = "v2.6.8"
    curve["variant"] = checked["variant"]
    curve["transition_scope"] = "preparation_defrost_recovery"
    curve["transition_breakdown"] = "not_decomposed"
    return finalize_v268_curve(curve)


def calculate(
    loader: Any, cycle_names: Sequence[str], recipe: Mapping[str, object] | None = None
) -> pd.DataFrame:
    """Calculate V2.6.8 curves for the requested Dataset-native cohort."""
    tables = [calculate_cycle(loader, name, recipe) for name in cycle_names]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
