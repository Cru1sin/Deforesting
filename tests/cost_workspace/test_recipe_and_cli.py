from __future__ import annotations

import pytest

import main_cost
from main_cost import build_parser


def test_v268_recipe_exposes_fixed9_dynamic8_protocol() -> None:
    from cost.cost_function_v2_6_8 import DEFAULT_RECIPE

    assert DEFAULT_RECIPE == {
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


def test_cli_lists_ticket_feature_models_and_v268() -> None:
    parser = build_parser()
    actions = {action.dest: action for action in parser._actions}
    expected = {
        "experiment_mean",
        "ticket_ridge_static5",
        "ticket_ridge_physical6",
        "ticket_ridge_dynamic8",
    }
    assert expected <= set(actions["transition_energy_model"].choices)
    assert expected <= set(actions["transition_heat_model"].choices)
    assert "v2.6.8" in actions["cost"].choices


def test_v268_canonical_is_cli_defaults_and_overrides_require_variant() -> None:
    args = build_parser().parse_args(["--action", "calculate", "--cost", "v2.6.8"])
    assert main_cost._recipe(main_cost.cost_function_v2_6_8, args) == (
        main_cost.cost_function_v2_6_8.DEFAULT_RECIPE
    )
    args = build_parser().parse_args(
        [
            "--action",
            "calculate",
            "--cost",
            "v2.6.8",
            "--transition-energy-model",
            "ticket_ridge_static5",
        ]
    )
    with pytest.raises(ValueError, match="variant"):
        main_cost._recipe(main_cost.cost_function_v2_6_8, args)
    args.variant = "static_energy"
    assert main_cost._recipe(main_cost.cost_function_v2_6_8, args)["label_eligible"] is False
