from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _candidates(periods: int = 7) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01")
    return pd.DataFrame(
        {
            "candidate_defrost_time": pd.date_range(
                start + pd.Timedelta(minutes=1), periods=periods, freq="min"
            ),
            "heating_accounting_start": start,
            "pre_defrost_electricity_kwh": 2.0,
            "pre_defrost_heat_kwh": 6.0,
            "pre_defrost_compressor_electricity_kwh": 1.0,
            "defrost_event_electricity_kwh": 1.0,
            "defrost_event_net_heat_kwh": 3.0,
            "defrost_event_compressor_electricity_kwh": 0.5,
            "defrost_event_duration_minutes": 2.0,
            "defrost_event_electricity_prediction_available": True,
            "defrost_event_net_heat_prediction_available": True,
            "defrost_event_compressor_electricity_prediction_available": True,
            "defrost_event_duration_prediction_available": True,
            "defrost_event_electricity_in_training_domain": True,
            "defrost_event_net_heat_in_training_domain": True,
            "defrost_event_compressor_electricity_in_training_domain": True,
            "defrost_event_duration_in_training_domain": True,
            "pre_defrost_feature_window_valid": True,
            "pre_defrost_electricity_measurement_valid": True,
            "pre_defrost_heat_measurement_valid": True,
            "pre_defrost_compressor_electricity_measurement_valid": True,
        }
    )


def test_objectives_compute_exact_values_and_independent_support() -> None:
    from defrost_decision.performance_objectives import (
        add_single_objective_optima,
        calculate_performance_objectives,
    )

    values = _candidates()
    values.loc[0, "defrost_event_electricity_in_training_domain"] = False
    objectives = calculate_performance_objectives(values)
    assert not any(column.endswith("_t_star") for column in objectives)
    result = add_single_objective_optima(objectives)

    hours = 1 / 60 + 2 / 60
    assert result.loc[0, "cycle_cop"] == pytest.approx(3.0)
    assert result.loc[0, "cycle_heating_rate_kw"] == pytest.approx(9.0 / hours)
    assert result.loc[0, "cycle_evaporator_capacity_kw"] == pytest.approx(7.5 / hours)
    assert not result.loc[0, "cycle_cop_eligible_without_extrapolation"]
    assert result.loc[0, "cycle_heating_rate_kw_eligible_without_extrapolation"]
    assert result.loc[0, "cycle_evaporator_capacity_kw_eligible_without_extrapolation"]
    assert not result.loc[0, "cycle_cop_eligible"]
    assert result["cycle_heating_rate_kw_eligible"].all()
    assert result["cycle_evaporator_capacity_kw_eligible"].all()
    for prefix in ("cycle_cop", "cycle_heating_rate_kw", "cycle_evaporator_capacity_kw"):
        assert result[f"{prefix}_t_star"].notna().all()
    assert result["cycle_cop_basin_5pct_width_minutes"].iloc[0] == 5


def _policy_objectives() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=8, freq="min")
    frame = pd.DataFrame(
        {
            "candidate_defrost_time": times,
            "cycle_cop": [1.0, 1.4, 1.7, 2.0, 0.9, 0.8, 0.7, 0.6],
            "cycle_heating_rate_kw": [2.0, 1.95, 1.6, 1.0, 0.9, 0.8, 0.7, 0.6],
            "cycle_evaporator_capacity_kw": np.arange(8, dtype=float),
            "pre_defrost_feature_window_valid": True,
        }
    )
    for prefix in ("cycle_cop", "cycle_heating_rate_kw", "cycle_evaporator_capacity_kw"):
        frame[f"{prefix}_measurements_valid"] = True
        frame[f"{prefix}_physically_valid"] = True
        frame[f"{prefix}_eligible_without_extrapolation"] = True
        frame[f"{prefix}_eligible"] = True
        frame[f"{prefix}_uses_model_extrapolation"] = False
    return frame


def test_policy_selects_chord_knee_ignores_o_and_reports_reference_window() -> None:
    from defrost_decision.pareto_selection import select_cop_heating_rate_pareto_knee

    objectives = _policy_objectives()
    first = select_cop_heating_rate_pareto_knee(objectives)
    changed = objectives.copy()
    changed["cycle_evaporator_capacity_kw"] = [np.nan, -1e9, 1e12, 5.0, -5.0, 7.0, 8.0, 9.0]
    changed["cycle_evaporator_capacity_kw_eligible"] = [
        False,
        False,
        True,
        True,
        False,
        True,
        False,
        True,
    ]
    second = select_cop_heating_rate_pareto_knee(changed)

    assert (
        first.loc[first["is_selected_pareto_point"], "candidate_defrost_time"].item()
        == objectives.loc[1, "candidate_defrost_time"]
    )
    assert second["is_selected_pareto_point"].equals(first["is_selected_pareto_point"])
    assert not first.loc[1, "within_5pct_of_best_cop_and_heating_rate"]
    assert first.loc[1, "is_selected_pareto_point"]
    assert not first["within_5pct_of_best_cop_and_heating_rate"].any()
    assert first["selected_defrost_time"].eq(objectives.loc[1, "candidate_defrost_time"]).all()
    assert first["is_selected"].equals(first["is_selected_pareto_point"])
    assert first["selection_method"].eq("cop_heating_rate_pareto_knee").all()
    assert first["selection_status"].eq("selected").all()
    assert first["selection_reason"].eq("normalized_chord_distance").all()
    assert not {"cycle_cop_native_eligible", "cycle_cop_model_supported"} & set(first)


def test_policy_abstains_without_h_and_applies_minimum_time_before_pareto() -> None:
    from defrost_decision.pareto_selection import select_cop_heating_rate_pareto_knee

    objectives = _policy_objectives()
    no_h = objectives.copy()
    no_h["cycle_heating_rate_kw_eligible"] = False
    abstained = select_cop_heating_rate_pareto_knee(no_h)
    assert not abstained["is_selected_pareto_point"].any()
    assert abstained["selection_method"].eq("cop_heating_rate_pareto_knee").all()
    assert abstained["selection_reason"].eq("no_common_domain").all()
    assert abstained["selected_defrost_time"].isna().all()

    minimum = objectives.loc[2, "candidate_defrost_time"]
    restricted = select_cop_heating_rate_pareto_knee(objectives, minimum_time=minimum)
    assert not restricted.loc[:1, "is_cop_heating_rate_pareto_point"].any()
    assert (
        restricted.loc[restricted["is_selected_pareto_point"], "candidate_defrost_time"].item()
        >= minimum
    )


def test_policy_rejects_stale_native_rows_with_nonfinite_ch_values() -> None:
    from defrost_decision.pareto_selection import select_cop_heating_rate_pareto_knee

    objectives = _policy_objectives()
    objectives.loc[1, ["cycle_cop", "cycle_heating_rate_kw"]] = np.nan

    result = select_cop_heating_rate_pareto_knee(objectives)

    assert not result.loc[1, "cycle_cop_eligible"]
    assert not result.loc[1, "cycle_heating_rate_kw_eligible"]
    assert not result.loc[1, "is_cop_heating_rate_pareto_point"]
    assert not result.loc[1, "is_selected_pareto_point"]


def test_policy_extrapolation_requires_valid_pre_action_state() -> None:
    from defrost_decision.pareto_selection import select_cop_heating_rate_pareto_knee

    objectives = _policy_objectives()
    objectives.loc[1, "pre_defrost_feature_window_valid"] = False
    for prefix in ("cycle_cop", "cycle_heating_rate_kw", "cycle_evaporator_capacity_kw"):
        objectives.loc[1, f"{prefix}_eligible"] = False
        objectives.loc[1, f"{prefix}_eligible_without_extrapolation"] = False

    result = select_cop_heating_rate_pareto_knee(objectives)

    assert not result.loc[1, "cycle_cop_eligible"]
    assert not result.loc[1, "cycle_heating_rate_kw_eligible"]
    assert not result.loc[1, "cycle_evaporator_capacity_kw_eligible"]
    assert not result.loc[1, "is_cop_heating_rate_pareto_point"]
    assert not result.loc[1, "is_selected_pareto_point"]


def test_policy_extrapolation_still_requires_five_continuous_minutes() -> None:
    from defrost_decision.pareto_selection import select_cop_heating_rate_pareto_knee

    objectives = _policy_objectives()
    for prefix in ("cycle_cop", "cycle_heating_rate_kw", "cycle_evaporator_capacity_kw"):
        objectives[f"{prefix}_eligible"] = False
        objectives.loc[1, f"{prefix}_eligible"] = True

    result = select_cop_heating_rate_pareto_knee(objectives)

    assert not result.loc[1, "cycle_cop_eligible"]
    assert not result.loc[1, "cycle_heating_rate_kw_eligible"]
    assert not result.loc[1, "cycle_evaporator_capacity_kw_eligible"]
    assert not result["is_selected_pareto_point"].any()


def test_objectives_require_explicit_permission_for_extrapolation() -> None:
    from defrost_decision.performance_objectives import calculate_performance_objectives

    candidates = _candidates()
    candidates["defrost_event_electricity_in_training_domain"] = False

    default = calculate_performance_objectives(candidates)
    allowed = calculate_performance_objectives(candidates, allow_model_extrapolation=True)

    assert not default["cycle_cop_eligible"].any()
    assert allowed["cycle_cop_eligible"].all()
    assert allowed["cycle_cop_uses_model_extrapolation"].all()
