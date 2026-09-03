from __future__ import annotations

import pandas as pd
import pytest

from cost.cost_curve import build_historical_cost_curve, cycle_ratio
from cost.cost_function_v1 import validate_recipe


def _recipe(**changes: object) -> dict[str, object]:
    recipe: dict[str, object] = {
        "base_cost": "v1",
        "version": "v1",
        "variant": None,
        "label_eligible": True,
        "heat_basis": "unit",
        "event_scope": "stable_heating_start_to_actual_preparation",
        "heating_start_rule": "stable_heating_start",
        "integration_protocol": "historical_reconstruction",
        "state_protocol": "historical_interpolation",
        "candidate_start_rule": "stable_heating_start_plus_10_minutes",
        "candidate_end_rule": "observed_defrost_preparation_start",
        "candidate_cadence": "1_minute_plus_exact_endpoint",
        "state_window": "[tau-60s,tau)",
        "transition_scope": "preparation_defrost_recovery",
        "transition_window": "candidate_state_at_tau",
        "transition_provenance": "candidate_time_state_plus_fixed_recovery",
        "decision_rule": "supported_argmin_inverse_cop",
        "heating_energy_model": "measured_total_power",
        "heating_heat_model": "measured_unit_heat",
        "transition_energy_model": "pe_quadratic_plus_fixed_recovery",
        "transition_heat_model": "zero_transition_heat",
    }
    recipe.update(changes)
    return recipe


def test_validate_recipe_accepts_canonical_v1_defaults() -> None:
    assert validate_recipe(_recipe()) == _recipe()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"heat_basis": "water"}, "heat basis"),
        ({"variant": "trial"}, "variant"),
    ],
)
def test_validate_recipe_rejects_incompatible_canonical_recipe(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_recipe(_recipe(**change))


def test_component_override_requires_a_named_variant() -> None:
    with pytest.raises(ValueError, match="variant"):
        validate_recipe(_recipe(transition_heat_model="linear_qprep_plus_signed_quadratic_qd"))

    recipe = _recipe(
        variant="trial",
        transition_heat_model="linear_qprep_plus_signed_quadratic_qd",
        transition_window="observed_preparation_and_defrost_durations",
        transition_provenance=(
            "offline_diagnostic_future_boundary_observed_durations_plus_fixed_recovery"
        ),
    )
    assert validate_recipe(recipe)["variant"] == "trial"
    assert validate_recipe(recipe)["label_eligible"] is False

    normalized = validate_recipe(
        _recipe(variant="trial", transition_heat_model="linear_qprep_plus_signed_quadratic_qd")
    )
    assert normalized["transition_window"] == "observed_preparation_and_defrost_durations"
    assert normalized["transition_provenance"].endswith("_plus_fixed_recovery")


@pytest.mark.parametrize(
    "protocol",
    [
        {"integration_protocol": "strict_causal"},
        {"state_protocol": "strict_causal"},
        {"integration_protocol": "strict_causal", "state_protocol": "strict_causal"},
    ],
)
def test_noncanonical_measurement_protocol_requires_label_ineligible_variant(
    protocol: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="variant"):
        validate_recipe(_recipe(**protocol))

    checked = validate_recipe(_recipe(variant="strict_audit", **protocol))

    assert checked["label_eligible"] is False


def test_variant_still_rejects_unknown_or_incompatible_components() -> None:
    with pytest.raises(ValueError, match="transition heat model"):
        validate_recipe(_recipe(variant="trial", transition_heat_model="experimental"))
    with pytest.raises(ValueError, match="heat basis.*heating heat model"):
        validate_recipe(_recipe(variant="trial", heat_basis="water"))


def test_fixed_recipe_metadata_is_not_a_runtime_component() -> None:
    checked = validate_recipe(
        _recipe(
            heating_start_rule="bogus",
            event_scope="bogus_to_actual_preparation",
            unexpected_parameter=True,
        )
    )

    assert checked["heating_start_rule"] == "stable_heating_start"
    assert checked["event_scope"] == "stable_heating_start_to_actual_preparation"
    assert "unexpected_parameter" not in checked


def test_build_historical_cost_curve_uses_joint_support_positive_heat_and_argmin() -> None:
    times = pd.date_range("2026-01-01", periods=4, freq="min")
    boundaries = pd.DataFrame(
        {
            "cycle_name": "cycle_a",
            "candidate_time": times,
            "candidate_elapsed_minutes": [10.0, 11.0, 12.0, 13.0],
        }
    )
    eh = pd.DataFrame(
        {
            "heating_energy_kwh": [1.0, 1.0, 2.0, 1.0],
            "heating_valid": [True, True, True, False],
            "strict_heating_energy_supported": False,
            "heating_energy_model": "measured_total_power",
            "heating_energy_rule": "stable_heating_start",
            "heating_energy_status": "supported",
        }
    )
    qh = pd.DataFrame(
        {
            "heating_heat_kwh": [1.0, 2.0, -1.0, 2.0],
            "strict_heating_heat_supported": False,
            "heating_heat_model": "measured_unit_heat",
            "heating_heat_rule": "stable_heating_start",
            "heating_heat_status": "supported",
        }
    )
    et = pd.DataFrame(
        {
            "transition_energy_kwh": [0.0, 0.0, 0.0, 0.0],
            "preparation_energy_kwh": 0.0,
            "defrost_energy_kwh": 0.0,
            "recovery_energy_kwh": 0.0,
            "ET_evaluable": [True, True, True, True],
            "ET_in_support": [False, False, False, False],
            "strict_ET_supported": False,
            "transition_energy_model": "pe_quadratic_plus_fixed_recovery",
            "transition_energy_rule": "strict_pre_action_60s",
            "transition_energy_status": "supported",
        }
    )
    qt = pd.DataFrame(
        {
            "transition_heat_kwh": 0.0,
            "preparation_heat_kwh": 0.0,
            "defrost_heat_kwh": 0.0,
            "recovery_heat_kwh": 0.0,
            "QT_evaluable": [True, True, True, True],
            "QT_in_support": [False, False, False, False],
            "QT_physical_valid": [True, True, True, True],
            "strict_QT_supported": False,
            "transition_heat_model": "zero_transition_heat",
            "transition_heat_rule": "none",
            "transition_heat_status": "supported",
        }
    )

    result = build_historical_cost_curve(boundaries, eh, qh, et, qt, _recipe())

    assert result["supported"].tolist() == [True, True, False, False]
    assert result["optimization_eligible"].tolist() == [True, True, False, False]
    assert result["support_policy"].eq("allow_historical_extrapolation").all()
    assert result["inverse_cop"].iloc[:2].tolist() == pytest.approx([1.0, 0.5])
    assert result["is_optimum"].tolist() == [False, True, False, False]
    assert result["selected"].equals(result["is_optimum"])
    assert result["selected_time"].eq(times[1]).all()
    assert result["selection_policy"].eq("argmin_inverse_cop").all()
    assert result["selection_status"].eq("selected").all()
    assert result["selection_reason"].eq("historical_eligible_minimum").all()
    assert result.loc[result["selected"], "selection_score"].item() == 0.5
    assert not result["selection_model_supported"].any()
    assert result["relative_regret"].iloc[:2].tolist() == pytest.approx([1.0, 0.0])
    assert result["near_optimal_1pct"].tolist() == [False, True, False, False]
    assert result["near_optimal_5pct"].tolist() == [False, True, False, False]
    assert result["base_cost"].eq("v1").all()
    assert result["variant"].isna().all()
    assert result["label_eligible"].all()


def test_build_historical_cost_curve_rejects_positive_defrost_heat() -> None:
    boundary = pd.DataFrame(
        {"cycle_name": ["cycle_a"], "candidate_time": [pd.Timestamp("2026-01-01")]}
    )
    eh = pd.DataFrame({"heating_energy_kwh": [1.0], "heating_valid": [True]})
    qh = pd.DataFrame({"heating_heat_kwh": [2.0]})
    et = pd.DataFrame({"transition_energy_kwh": [0.0], "ET_evaluable": [True]})
    qt = pd.DataFrame(
        {
            "transition_heat_kwh": [0.1],
            "defrost_heat_kwh": [0.1],
            "QT_evaluable": [True],
            "QT_physical_valid": [True],
        }
    )

    with pytest.raises(ValueError, match="defrost_heat_kwh"):
        build_historical_cost_curve(boundary, eh, qh, et, qt, _recipe())


def test_cycle_ratio_names_totals_and_masks_nonfinite_or_zero_heat() -> None:
    curve = pd.DataFrame(
        {
            "heating_energy_kwh": [1.0, 1.0, float("inf")],
            "transition_energy_kwh": [0.5, 0.5, 0.5],
            "heating_heat_kwh": [2.0, 0.0, 2.0],
            "transition_heat_kwh": [1.0, 0.0, 1.0],
        }
    )

    result = cycle_ratio(curve)

    assert result["total_energy_kwh"].tolist()[:2] == [1.5, 1.5]
    assert result["total_heat_kwh"].tolist() == [3.0, 0.0, 3.0]
    assert result["inverse_cop"].iloc[0] == pytest.approx(0.5)
    assert result["inverse_cop"].iloc[1:].isna().all()


def test_build_historical_cost_curve_does_not_invent_missing_transition_breakdown() -> None:
    boundary = pd.DataFrame(
        {"cycle_name": ["cycle_a"], "candidate_time": [pd.Timestamp("2026-01-01")]}
    )
    eh = pd.DataFrame({"heating_energy_kwh": [1.0], "heating_valid": [True]})
    qh = pd.DataFrame({"heating_heat_kwh": [2.0]})
    et = pd.DataFrame({"transition_energy_kwh": [0.0], "ET_evaluable": [True]})
    qt = pd.DataFrame(
        {
            "transition_heat_kwh": [0.0],
            "QT_evaluable": [True],
            "QT_physical_valid": [True],
        }
    )

    result = build_historical_cost_curve(boundary, eh, qh, et, qt, _recipe())

    for phase in ("preparation", "defrost", "recovery"):
        assert f"{phase}_energy_kwh" not in result
        assert f"{phase}_heat_kwh" not in result
