"""Calculate three independent candidate-time performance objectives.

Paper notation: C -> cycle_cop, H -> cycle_heating_rate_kw, and
O -> cycle_evaporator_capacity_kw. O is diagnostic only and never selects time.
"""

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
    allow_model_extrapolation: bool,
) -> pd.DataFrame:
    finite = np.isfinite(result[name])
    state_valid = result["pre_defrost_feature_window_valid"].fillna(False)
    base = measurements_valid & physically_valid & predictions_available & state_valid & finite
    in_domain = base & predictions_in_training_domain
    allowed = base & (predictions_in_training_domain | allow_model_extrapolation)
    result[f"{name}_measurements_valid"] = measurements_valid
    result[f"{name}_physically_valid"] = physically_valid
    result[f"{name}_eligible_without_extrapolation"] = in_domain & five_minute_support_runs(
        result["candidate_defrost_time"], in_domain
    )
    result[f"{name}_eligible"] = allowed & five_minute_support_runs(
        result["candidate_defrost_time"], allowed
    )
    result[f"{name}_uses_model_extrapolation"] = (
        result[f"{name}_eligible"] & ~predictions_in_training_domain
    )
    return result


def calculate_cycle_cop(
    table: pd.DataFrame, *, allow_model_extrapolation: bool = False
) -> pd.DataFrame:
    """Calculate full-cycle heat divided by full-cycle electricity."""
    result = table.copy()
    total_heat = result["pre_defrost_heat_kwh"] + result["defrost_event_net_heat_kwh"]
    total_electricity = (
        result["pre_defrost_electricity_kwh"] + result["defrost_event_electricity_kwh"]
    )
    result["cycle_cop"] = total_heat / total_electricity
    predictions_available = (
        result["defrost_event_electricity_prediction_available"].fillna(False)
        & result["defrost_event_net_heat_prediction_available"].fillna(False)
    )
    predictions_in_domain = (
        result["defrost_event_electricity_in_training_domain"].fillna(False)
        & result["defrost_event_net_heat_in_training_domain"].fillna(False)
    )
    measurements = (
        result["pre_defrost_electricity_measurement_valid"].fillna(False)
        & result["pre_defrost_heat_measurement_valid"].fillna(False)
    )
    physical = (
        np.isfinite(total_electricity)
        & total_electricity.gt(0)
        & np.isfinite(total_heat)
        & total_heat.gt(0.01)
    )
    return _finish_objective(
        result,
        "cycle_cop",
        measurements_valid=measurements,
        physically_valid=physical,
        predictions_available=predictions_available,
        predictions_in_training_domain=predictions_in_domain,
        allow_model_extrapolation=allow_model_extrapolation,
    )


def calculate_cycle_heating_rate(
    table: pd.DataFrame, *, allow_model_extrapolation: bool = False
) -> pd.DataFrame:
    """Calculate full-cycle delivered heat per elapsed cycle hour."""
    result = table.copy()
    elapsed_hours = (
        pd.to_datetime(result["candidate_defrost_time"])
        - pd.to_datetime(result["heating_accounting_start"])
    ).dt.total_seconds() / 3600
    duration_hours = elapsed_hours + pd.to_numeric(
        result["defrost_event_duration_minutes"], errors="coerce"
    ) / 60
    total_heat = result["pre_defrost_heat_kwh"] + result["defrost_event_net_heat_kwh"]
    result["cycle_heating_rate_kw"] = total_heat / duration_hours
    predictions_available = (
        result["defrost_event_net_heat_prediction_available"].fillna(False)
        & result["defrost_event_duration_prediction_available"].fillna(False)
    )
    predictions_in_domain = (
        result["defrost_event_net_heat_in_training_domain"].fillna(False)
        & result["defrost_event_duration_in_training_domain"].fillna(False)
    )
    measurements = result["pre_defrost_heat_measurement_valid"].fillna(False)
    physical = np.isfinite(duration_hours) & duration_hours.gt(0) & np.isfinite(total_heat)
    return _finish_objective(
        result,
        "cycle_heating_rate_kw",
        measurements_valid=measurements,
        physically_valid=physical,
        predictions_available=predictions_available,
        predictions_in_training_domain=predictions_in_domain,
        allow_model_extrapolation=allow_model_extrapolation,
    )


def calculate_cycle_evaporator_capacity(
    table: pd.DataFrame, *, allow_model_extrapolation: bool = False
) -> pd.DataFrame:
    """Calculate compressor-subtracted heat per elapsed cycle hour for reference."""
    result = table.copy()
    elapsed_hours = (
        pd.to_datetime(result["candidate_defrost_time"])
        - pd.to_datetime(result["heating_accounting_start"])
    ).dt.total_seconds() / 3600
    duration_hours = elapsed_hours + pd.to_numeric(
        result["defrost_event_duration_minutes"], errors="coerce"
    ) / 60
    net_output = (
        result["pre_defrost_heat_kwh"]
        - result["pre_defrost_compressor_electricity_kwh"]
        + result["defrost_event_net_heat_kwh"]
        - result["defrost_event_compressor_electricity_kwh"]
    )
    result["cycle_evaporator_capacity_kw"] = net_output / duration_hours
    predictions_available = (
        result["defrost_event_net_heat_prediction_available"].fillna(False)
        & result["defrost_event_duration_prediction_available"].fillna(False)
        & result["defrost_event_compressor_electricity_prediction_available"].fillna(False)
    )
    predictions_in_domain = (
        result["defrost_event_net_heat_in_training_domain"].fillna(False)
        & result["defrost_event_duration_in_training_domain"].fillna(False)
        & result["defrost_event_compressor_electricity_in_training_domain"].fillna(False)
    )
    measurements = (
        result["pre_defrost_heat_measurement_valid"].fillna(False)
        & result["pre_defrost_compressor_electricity_measurement_valid"].fillna(False)
    )
    physical = np.isfinite(duration_hours) & duration_hours.gt(0) & np.isfinite(net_output)
    return _finish_objective(
        result,
        "cycle_evaporator_capacity_kw",
        measurements_valid=measurements,
        physically_valid=physical,
        predictions_available=predictions_available,
        predictions_in_training_domain=predictions_in_domain,
        allow_model_extrapolation=allow_model_extrapolation,
    )


def calculate_performance_objectives(
    candidate_table: pd.DataFrame, *, allow_model_extrapolation: bool = False
) -> pd.DataFrame:
    """Calculate C, H and O without combining or cross-filtering their eligibility."""
    result = candidate_table.sort_values(
        "candidate_defrost_time", kind="stable"
    ).reset_index(drop=True)
    result = calculate_cycle_cop(
        result, allow_model_extrapolation=allow_model_extrapolation
    )
    result = calculate_cycle_heating_rate(
        result, allow_model_extrapolation=allow_model_extrapolation
    )
    return calculate_cycle_evaporator_capacity(
        result, allow_model_extrapolation=allow_model_extrapolation
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


def add_single_objective_optima(objectives: pd.DataFrame) -> pd.DataFrame:
    """Add each objective's own optimum and connected near-optimal basins."""
    result = objectives.copy()
    for name in ("cycle_cop", "cycle_heating_rate_kw", "cycle_evaporator_capacity_kw"):
        result[f"{name}_t_star"] = pd.NaT
        for percent in (1, 2, 5):
            result[f"{name}_basin_{percent}pct_start"] = pd.NaT
            result[f"{name}_basin_{percent}pct_end"] = pd.NaT
            result[f"{name}_basin_{percent}pct_width_minutes"] = np.nan
        eligible = result[f"{name}_eligible"]
        if not eligible.any():
            continue
        optimum = int(
            result.index[eligible & result[name].eq(result.loc[eligible, name].max())][0]
        )
        result[f"{name}_t_star"] = result.loc[optimum, "candidate_defrost_time"]
        for percent in (1, 2, 5):
            start, end, width = _connected_basin(result, name, optimum, eligible, percent)
            result[f"{name}_basin_{percent}pct_start"] = start
            result[f"{name}_basin_{percent}pct_end"] = end
            result[f"{name}_basin_{percent}pct_width_minutes"] = width
    return result
