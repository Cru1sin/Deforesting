"""Frozen V2.6.8 inverse-COP diagnostic retained for historical comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from defrost_event_models.ridge_models import load_defrost_event_models
from defrost_event_models.training_data import timestamp

from ..candidate_quantities import build_candidate_quantities
from ..selection_results import add_selected_time_fields, cycle_ratio, five_minute_support_runs

Q_MIN_KWH = 0.01
DEFAULT_RECIPE: dict[str, object] = {
    "base_cost": "v2.6.8",
    "version": "v2.6.8",
    "run_name": None,
    "label_eligible": False,
    "heat_basis": "water",
    "integration_protocol": "strict_causal",
    "state_protocol": "strict_causal",
    "heating_heat_model": "measured_water_heat",
    "transition_energy_model": "ridge_dynamic_state_8",
    "transition_heat_model": "ridge_dynamic_state_8",
}


def calculate_cycle(
    loader: Any,
    cycle_name: str,
    recipe: Mapping[str, object] | None = None,
    models: Mapping[str, Any] | None = None,
    *,
    candidate_step_seconds: int = 60,
) -> pd.DataFrame:
    checked = validate_recipe(DEFAULT_RECIPE if recipe is None else recipe)
    curve = build_candidate_quantities(
        loader,
        cycle_name,
        models,
        candidate_step_seconds=candidate_step_seconds,
        defrost_event_electricity_model=str(checked["transition_energy_model"]),
        defrost_event_heat_model=str(checked["transition_heat_model"]),
    )
    curve["heating_measurement_valid"] = (
        curve["pre_defrost_electricity_measurement_valid"]
        & curve["pre_defrost_heat_measurement_valid"]
    )
    curve["pre_action_window_valid"] = curve["pre_defrost_feature_window_valid"]
    curve["defrost_event_electricity_evaluable"] = curve[
        "defrost_event_electricity_prediction_available"
    ]
    curve["defrost_event_net_heat_evaluable"] = curve["defrost_event_net_heat_prediction_available"]
    curve = cycle_ratio(curve)
    curve["physical_valid"] = curve["total_energy_kwh"].gt(0) & curve["total_heat_kwh"].gt(
        Q_MIN_KWH
    )
    curve["algorithm"] = curve["base_cost"] = "v2.6.8"
    curve["run_name"] = checked["run_name"]
    return finalize_curve(curve)


def calculate(
    loader: Any, cycle_names: Sequence[str], recipe: Mapping[str, object] | None = None
) -> pd.DataFrame:
    models = load_defrost_event_models()
    tables = [calculate_cycle(loader, name, recipe, models) for name in cycle_names]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def validate_recipe(recipe: Mapping[str, object]) -> dict[str, object]:
    value = dict(DEFAULT_RECIPE)
    for key in ("run_name", "transition_energy_model", "transition_heat_model"):
        if key in recipe:
            value[key] = recipe[key]
    allowed = {
        "experiment_balanced_mean",
        "ridge_basic_state_5",
        "ridge_physical_state_6",
        "ridge_dynamic_state_8",
    }
    for key in ("transition_energy_model", "transition_heat_model"):
        if value[key] not in allowed:
            raise ValueError(f"V2.6.8 does not implement {key}={value[key]}")
    return value


def finalize_curve(curve: pd.DataFrame) -> pd.DataFrame:
    result = (
        curve.sort_values("candidate_defrost_time", kind="stable").reset_index(drop=True).copy()
    )
    inverse = pd.to_numeric(result["inverse_cop"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    base = (
        result["heating_measurement_valid"].fillna(False)
        & result["defrost_event_electricity_evaluable"].fillna(False)
        & result["defrost_event_net_heat_evaluable"].fillna(False)
        & result["defrost_event_electricity_in_training_domain"].fillna(False)
        & result["defrost_event_net_heat_in_training_domain"].fillna(False)
        & result["pre_action_window_valid"].fillna(False)
        & result["physical_valid"].fillna(False)
        & inverse.notna()
    )
    result["model_supported"] = result["defrost_event_electricity_in_training_domain"].fillna(
        False
    ) & result["defrost_event_net_heat_in_training_domain"].fillna(False)
    result["support_rule"] = "require_empirical_support"
    result["continuous_support"] = five_minute_support_runs(result["candidate_defrost_time"], base)
    result["optimization_eligible"] = base & result["continuous_support"]
    result["diagnostic_minimum"] = pd.NaT
    for percent in (1, 5):
        result[f"basin_{percent}pct_start"] = pd.NaT
        result[f"basin_{percent}pct_end"] = pd.NaT
        result[f"basin_{percent}pct_width_minutes"] = np.nan
    eligible = result["optimization_eligible"]
    if eligible.any():
        optimum = int(result.index[eligible & inverse.eq(inverse.loc[eligible].min())][0])
        optimum_time = timestamp(result.loc[optimum, "candidate_defrost_time"])
        result["diagnostic_minimum"] = optimum_time
        for percent in (1, 5):
            within = eligible & inverse.le(float(inverse.iloc[optimum]) * (1 + percent / 100))
            left = right = optimum
            while left and bool(within.iloc[left - 1]):
                left -= 1
            while right + 1 < len(result) and bool(within.iloc[right + 1]):
                right += 1
            start, end = result.loc[[left, right], "candidate_defrost_time"].map(pd.Timestamp)
            result[f"basin_{percent}pct_start"] = start
            result[f"basin_{percent}pct_end"] = end
            result[f"basin_{percent}pct_width_minutes"] = (end - start).total_seconds() / 60
    result["relative_regret"] = np.nan
    if eligible.any():
        result.loc[eligible, "relative_regret"] = (
            inverse.loc[eligible] / inverse.loc[eligible].min() - 1
        )
    result["near_optimal_1pct"] = eligible & result["relative_regret"].le(0.01)
    result["near_optimal_5pct"] = eligible & result["relative_regret"].le(0.05)
    for phase in ("preparation", "defrost", "recovery"):
        result[f"{phase}_energy_kwh"] = np.nan
        result[f"{phase}_heat_kwh"] = np.nan
    result["recommended_time"] = pd.NaT
    result["hard_label_eligible"] = False
    result["label_eligible"] = False
    selected = pd.to_datetime(result["candidate_defrost_time"]).eq(
        pd.to_datetime(result["diagnostic_minimum"]).iloc[0]
    )
    return add_selected_time_fields(
        result,
        selected,
        method="supported_inverse_cop_minimum",
        selected_reason="continuous_supported_minimum",
        abstain_reason="no_continuous_supported_minimum",
        score=inverse,
        model_supported=result["model_supported"],
    )
