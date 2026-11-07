"""Calculate cycle COP and select each explicitly requested objective independently."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .selection_results import five_minute_support_runs


def _finish_objective(
    result: pd.DataFrame,
    name: str,
    *,
    measurements_valid: pd.Series,
    physically_valid: pd.Series,
    predictions_available: pd.Series,
    predictions_in_training_domain: pd.Series,
    require_support_run: bool = True,
) -> pd.DataFrame:
    finite = np.isfinite(result[name])
    state_valid = result["pre_defrost_feature_window_valid"].eq(True).fillna(False)
    base = measurements_valid & physically_valid & predictions_available & state_valid & finite
    in_domain = base & predictions_in_training_domain
    # Model-domain flags annotate predictions; they never gate scientific calculations.
    allowed = base
    result[f"{name}_measurements_valid"] = measurements_valid
    result[f"{name}_physically_valid"] = physically_valid
    # Diagnostic column only; never use it as a selector or label support mask.
    result[f"{name}_eligible_without_extrapolation"] = in_domain
    result[f"{name}_eligible"] = allowed & (
        five_minute_support_runs(result["candidate_defrost_time"], allowed)
        if require_support_run
        else True
    )
    result[f"{name}_uses_model_extrapolation"] = (
        result[f"{name}_eligible"] & ~predictions_in_training_domain
    )
    return result


def calculate_cycle_cop(
    table: pd.DataFrame,
    *,
    effective: bool = False,
) -> pd.DataFrame:
    """Calculate full-cycle heat divided by full-cycle electricity."""
    result = table.copy()
    total_heat = result["pre_defrost_heat_kwh"] + result["defrost_event_net_heat_kwh"]
    total_electricity = (
        result["pre_defrost_electricity_kwh"] + result["defrost_event_electricity_kwh"]
    )
    result["cycle_cop"] = total_heat / total_electricity
    predictions_available = result["defrost_event_electricity_prediction_available"].eq(
        True
    ).fillna(False) & result["defrost_event_net_heat_prediction_available"].eq(True).fillna(False)
    predictions_in_domain = result["defrost_event_electricity_in_training_domain"].eq(True).fillna(
        False
    ) & result["defrost_event_net_heat_in_training_domain"].eq(True).fillna(False)
    measurements = result["pre_defrost_electricity_measurement_valid"].eq(True).fillna(
        False
    ) & result["pre_defrost_heat_measurement_valid"].eq(True).fillna(False)
    physical = (
        np.isfinite(total_electricity)
        & total_electricity.gt(0)
        & np.isfinite(total_heat)
        & total_heat.gt(0.01)
    )
    if effective:
        physical &= result["defrost_event_electricity_kwh"].gt(0)
        physical &= result["defrost_event_net_heat_kwh"].ge(0)
    return _finish_objective(
        result,
        "cycle_cop",
        require_support_run=not effective,
        measurements_valid=measurements,
        physically_valid=physical,
        predictions_available=predictions_available,
        predictions_in_training_domain=predictions_in_domain,
    )


def _connected_basin(
    result: pd.DataFrame, name: str, optimum: int, eligible: pd.Series, percent: int
) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    values = result[name]
    threshold = float(values.iloc[optimum]) - abs(float(values.iloc[optimum])) * percent / 100
    within = eligible & values.ge(threshold)
    left = right = optimum
    while left and bool(within.iloc[left - 1]):
        left -= 1
    while right + 1 < len(result) and bool(within.iloc[right + 1]):
        right += 1
    start, end = result.loc[[left, right], "candidate_defrost_time"].map(pd.Timestamp)
    return start, end, (end - start).total_seconds() / 60


def add_single_objective_optima(
    objectives: pd.DataFrame,
    names=("cycle_cop",),
    *,
    selection_mask=None,
) -> pd.DataFrame:
    """Add each objective's own optimum and connected near-optimal basins."""
    result = objectives.copy()
    for name in names:
        result[f"{name}_t_star"] = pd.NaT
        for percent in (1, 2, 5):
            result[f"{name}_basin_{percent}pct_start"] = pd.NaT
            result[f"{name}_basin_{percent}pct_end"] = pd.NaT
            result[f"{name}_basin_{percent}pct_width_minutes"] = np.nan
        eligible = result[f"{name}_eligible"]
        selectable = eligible if selection_mask is None else eligible & selection_mask
        if not selectable.any():
            continue
        optimum = int(
            result.index[selectable & result[name].eq(result.loc[selectable, name].max())][0]
        )
        result[f"{name}_t_star"] = result.loc[optimum, "candidate_defrost_time"]
        for percent in (1, 2, 5):
            start, end, width = _connected_basin(result, name, optimum, eligible, percent)
            result[f"{name}_basin_{percent}pct_start"] = start
            result[f"{name}_basin_{percent}pct_end"] = end
            result[f"{name}_basin_{percent}pct_width_minutes"] = width
    return result


def select_cycle_optimum(result):
    """Select on the action grid; model extrapolation is annotation only."""
    result = add_single_objective_optima(
        result,
        names=("cycle_cop",),
        selection_mask=result.get("on_action_grid", pd.Series(True, index=result.index)),
    )
    result["is_selected"] = pd.to_datetime(result.candidate_defrost_time).eq(
        result.cycle_cop_t_star
    )
    result["selected_point_in_training_domain"] = (
        result.is_selected
        & result.cycle_cop_eligible
        & ~result.get(
            "cycle_cop_uses_model_extrapolation", pd.Series(False, index=result.index)
        ).fillna(False)
    )
    result["t_star"] = result.cycle_cop_t_star
    result["t_star_model_supported"] = result.selected_point_in_training_domain.any()
    return result
