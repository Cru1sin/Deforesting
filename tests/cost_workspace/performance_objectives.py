"""Independent historical C/H/O comparisons; never combine their selectors."""

import numpy as np
import pandas as pd

from defrost_decision.performance_objectives import _finish_objective, calculate_cycle_cop


def calculate_cycle_heating_rate(table: pd.DataFrame) -> pd.DataFrame:
    """Calculate full-cycle delivered heat per elapsed cycle hour."""
    result = table.copy()
    elapsed_hours = (
        pd.to_datetime(result["candidate_defrost_time"])
        - pd.to_datetime(result["heating_accounting_start"])
    ).dt.total_seconds() / 3600
    duration_hours = (
        elapsed_hours
        + pd.to_numeric(result["defrost_event_duration_minutes"], errors="coerce") / 60
    )
    total_heat = result["pre_defrost_heat_kwh"] + result["defrost_event_net_heat_kwh"]
    result["cycle_heating_rate_kw"] = total_heat / duration_hours
    predictions_available = result["defrost_event_net_heat_prediction_available"].fillna(
        False
    ) & result["defrost_event_duration_prediction_available"].fillna(False)
    predictions_in_domain = result["defrost_event_net_heat_in_training_domain"].fillna(
        False
    ) & result["defrost_event_duration_in_training_domain"].fillna(False)
    measurements = result["pre_defrost_heat_measurement_valid"].fillna(False)
    physical = np.isfinite(duration_hours) & duration_hours.gt(0) & np.isfinite(total_heat)
    return _finish_objective(
        result,
        "cycle_heating_rate_kw",
        measurements_valid=measurements,
        physically_valid=physical,
        predictions_available=predictions_available,
        predictions_in_training_domain=predictions_in_domain,
    )


def calculate_cycle_evaporator_capacity(table: pd.DataFrame) -> pd.DataFrame:
    """Calculate compressor-subtracted heat per elapsed cycle hour for reference."""
    result = table.copy()
    elapsed_hours = (
        pd.to_datetime(result["candidate_defrost_time"])
        - pd.to_datetime(result["heating_accounting_start"])
    ).dt.total_seconds() / 3600
    duration_hours = (
        elapsed_hours
        + pd.to_numeric(result["defrost_event_duration_minutes"], errors="coerce") / 60
    )
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
    measurements = result["pre_defrost_heat_measurement_valid"].fillna(False) & result[
        "pre_defrost_compressor_electricity_measurement_valid"
    ].fillna(False)
    physical = np.isfinite(duration_hours) & duration_hours.gt(0) & np.isfinite(net_output)
    return _finish_objective(
        result,
        "cycle_evaporator_capacity_kw",
        measurements_valid=measurements,
        physically_valid=physical,
        predictions_available=predictions_available,
        predictions_in_training_domain=predictions_in_domain,
    )


def calculate_performance_objectives(candidate_table: pd.DataFrame) -> pd.DataFrame:
    """Calculate C, H and O without combining or cross-filtering their eligibility."""
    result = candidate_table.sort_values("candidate_defrost_time", kind="stable").reset_index(
        drop=True
    )
    result = calculate_cycle_cop(result)
    result = calculate_cycle_heating_rate(result)
    return calculate_cycle_evaporator_capacity(result)
