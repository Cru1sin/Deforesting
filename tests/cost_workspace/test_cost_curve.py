from __future__ import annotations

import pandas as pd
import pytest

from cost.cost_curve import build_cost_curve, validate_recipe


def _recipe(**changes: object) -> dict[str, object]:
    recipe: dict[str, object] = {
        "base_cost": "v1",
        "version": "v1",
        "variant": None,
        "label_eligible": True,
        "heat_basis": "unit",
        "event_scope": "stable_heating_start_to_actual_preparation",
        "heating_start_rule": "stable_heating_start",
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
        ({"event_scope": "heating_start_to_actual_preparation"}, "event scope"),
        ({"heating_start_rule": "heating_start"}, "start rule"),
        ({"variant": "trial"}, "canonical recipe"),
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

    recipe = _recipe(variant="trial", transition_heat_model="linear_qprep_plus_signed_quadratic_qd")
    assert validate_recipe(recipe)["variant"] == "trial"
    assert validate_recipe(recipe)["label_eligible"] is False


def test_variant_still_rejects_unknown_or_incompatible_parameters() -> None:
    with pytest.raises(ValueError, match="transition heat model"):
        validate_recipe(_recipe(variant="trial", transition_heat_model="experimental"))
    with pytest.raises(ValueError, match="heat basis.*heating heat model"):
        validate_recipe(_recipe(variant="trial", heat_basis="water"))
    with pytest.raises(ValueError, match="event scope|start rule"):
        validate_recipe(
            _recipe(
                variant="trial",
                heating_start_rule="bogus",
                event_scope="bogus_to_actual_preparation",
            )
        )
    with pytest.raises(ValueError, match="unexpected"):
        validate_recipe(_recipe(unexpected_parameter=True))


def test_build_cost_curve_uses_joint_support_positive_heat_and_argmin() -> None:
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
            "heating_energy_model": "measured_total_power",
            "heating_energy_rule": "stable_heating_start",
            "heating_energy_status": "supported",
        }
    )
    qh = pd.DataFrame(
        {
            "heating_heat_kwh": [1.0, 2.0, -1.0, 2.0],
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
            "ET_supported": [True, True, True, True],
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
            "QT_supported": [True, True, True, True],
            "transition_heat_model": "zero_transition_heat",
            "transition_heat_rule": "none",
            "transition_heat_status": "supported",
        }
    )

    result = build_cost_curve(boundaries, eh, qh, et, qt, _recipe())

    assert result["supported"].tolist() == [True, True, False, False]
    assert result["optimization_eligible"].tolist() == [True, True, False, False]
    assert result["inverse_cop"].iloc[:2].tolist() == pytest.approx([1.0, 0.5])
    assert result["is_optimum"].tolist() == [False, True, False, False]
    assert result["relative_regret"].iloc[:2].tolist() == pytest.approx([1.0, 0.0])
    assert result["near_optimal_1pct"].tolist() == [False, True, False, False]
    assert result["near_optimal_5pct"].tolist() == [False, True, False, False]
    assert result["base_cost"].eq("v1").all()
    assert result["variant"].isna().all()
    assert result["label_eligible"].all()


def test_build_cost_curve_rejects_positive_defrost_heat() -> None:
    boundary = pd.DataFrame(
        {"cycle_name": ["cycle_a"], "candidate_time": [pd.Timestamp("2026-01-01")]}
    )
    eh = pd.DataFrame({"heating_energy_kwh": [1.0], "heating_valid": [True]})
    qh = pd.DataFrame({"heating_heat_kwh": [2.0]})
    et = pd.DataFrame({"transition_energy_kwh": [0.0], "ET_supported": [True]})
    qt = pd.DataFrame(
        {
            "transition_heat_kwh": [0.1],
            "defrost_heat_kwh": [0.1],
            "QT_supported": [True],
        }
    )

    with pytest.raises(ValueError, match="defrost_heat_kwh"):
        build_cost_curve(boundary, eh, qh, et, qt, _recipe())
