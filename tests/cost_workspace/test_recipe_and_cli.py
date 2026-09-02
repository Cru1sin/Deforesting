from __future__ import annotations

import pytest

import main_cost
from cost.cost_function_v2_6_8 import validate_recipe
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


def test_cli_hides_fixed_recipe_fields() -> None:
    actions = {action.dest for action in build_parser()._actions}

    assert not {
        "heat_basis",
        "event_scope",
        "heating_start_rule",
        "heating_energy_model",
    } & actions


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


@pytest.mark.parametrize("cost", ["v1", "v2.5"])
@pytest.mark.parametrize(
    ("option", "model"),
    [
        ("--transition-energy-model", "ticket_ridge_dynamic8"),
        ("--transition-heat-model", "experiment_mean"),
    ],
)
def test_legacy_versions_reject_unimplemented_ticket_components(
    cost: str, option: str, model: str
) -> None:
    args = build_parser().parse_args(
        ["--action", "calculate", "--cost", cost, "--variant", "ticket", option, model]
    )

    with pytest.raises(ValueError, match=f"{cost} does not implement"):
        main_cost._recipe(main_cost.COST_MODULES[cost], args)


@pytest.mark.parametrize(
    "overrides",
    [
        ["--heating-heat-model", "measured_unit_heat"],
        ["--integration-protocol", "historical_reconstruction"],
        ["--state-protocol", "historical_interpolation"],
    ],
)
def test_v268_rejects_named_overrides_that_execution_does_not_implement(
    overrides: list[str],
) -> None:
    args = build_parser().parse_args(
        ["--action", "calculate", "--cost", "v2.6.8", "--variant", "fake", *overrides]
    )

    with pytest.raises(ValueError, match="v2.6.8 does not implement"):
        main_cost._recipe(main_cost.cost_function_v2_6_8, args)


def test_v268_accepts_independent_ticket_component_selection() -> None:
    args = build_parser().parse_args(
        [
            "--action",
            "calculate",
            "--cost",
            "v2.6.8",
            "--variant",
            "mixed",
            "--transition-energy-model",
            "ticket_ridge_static5",
            "--transition-heat-model",
            "experiment_mean",
        ]
    )

    recipe = main_cost._recipe(main_cost.cost_function_v2_6_8, args)

    assert recipe["transition_energy_model"] == "ticket_ridge_static5"
    assert recipe["transition_heat_model"] == "experiment_mean"


def test_v268_rejects_unimplemented_candidate_rule_even_for_named_variant() -> None:
    recipe = dict(main_cost.cost_function_v2_6_8.DEFAULT_RECIPE)
    recipe.update(
        variant="fake",
        candidate_start_rule="stable_heating_start_plus_10_minutes",
    )

    with pytest.raises(ValueError, match="v2.6.8 does not implement"):
        validate_recipe(recipe)
