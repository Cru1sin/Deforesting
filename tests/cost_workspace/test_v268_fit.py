from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _events() -> pd.DataFrame:
    rows = []
    for group_index, experiment in enumerate(("a", "b", "c", "heldout")):
        for event_index in range(4):
            x = float(group_index + event_index)
            rows.append(
                {
                    "event_id": f"{experiment}_{event_index}",
                    "experiment_id": experiment,
                    "x": x,
                    "defrost_event_electricity_observed_kwh": 1 + 2 * x,
                    "defrost_event_net_heat_observed_kwh": -2 + 0.5 * x,
                    "defrost_event_compressor_electricity_observed_kwh": 0.5 + 0.25 * x,
                    "defrost_event_duration_observed_minutes": 10 + x,
                }
            )
    return pd.DataFrame(rows)


def test_outcome_targets_are_the_single_explicit_four_target_contract() -> None:
    from defrost_event_models.ridge_models import OUTCOME_TARGETS, OUTCOME_VALIDITY

    assert OUTCOME_TARGETS == {
        "event_electricity": "defrost_event_electricity_observed_kwh",
        "event_net_heat": "defrost_event_net_heat_observed_kwh",
        "event_compressor_electricity": "defrost_event_compressor_electricity_observed_kwh",
        "event_duration": "defrost_event_duration_observed_minutes",
    }
    assert OUTCOME_VALIDITY == {
        "event_electricity": "energy_event_valid",
        "event_net_heat": "heat_event_valid",
        "event_compressor_electricity": "compressor_event_valid",
        "event_duration": "duration_event_valid",
    }


def test_each_outcome_selects_its_own_valid_events() -> None:
    from defrost_event_models.ridge_models import select_valid_events_for_quantity

    events = pd.DataFrame(
        {
            "event_id": ["energy_only", "heat_only"],
            "energy_event_valid": [True, False],
            "heat_event_valid": [False, True],
            "defrost_event_electricity_observed_kwh": [1.0, np.nan],
            "defrost_event_net_heat_observed_kwh": [np.nan, 2.0],
        }
    )

    assert select_valid_events_for_quantity(events, "event_electricity")["event_id"].tolist() == [
        "energy_only"
    ]
    assert select_valid_events_for_quantity(events, "event_net_heat")["event_id"].tolist() == [
        "heat_only"
    ]


def test_experiment_weights_equalize_group_mass() -> None:
    from defrost_event_models.ridge_models import experiment_weights

    groups = pd.Series(["a", "a", "a", "b", "c", "c"])
    weights = experiment_weights(groups)
    assert weights.sum() == pytest.approx(6)
    np.testing.assert_allclose(pd.Series(weights).groupby(groups).sum(), [2, 2, 2])


def test_heldout_targets_cannot_change_fold_model_or_alpha() -> None:
    from defrost_event_models.ridge_models import fit_model_for_heldout_experiment

    events = _events()
    before = fit_model_for_heldout_experiment(
        events, "heldout", ("x",), "defrost_event_electricity_observed_kwh"
    )
    changed = events.copy()
    changed.loc[
        changed["experiment_id"].eq("heldout"), "defrost_event_electricity_observed_kwh"
    ] += 10000
    after = fit_model_for_heldout_experiment(
        changed, "heldout", ("x",), "defrost_event_electricity_observed_kwh"
    )
    assert before.alpha == after.alpha
    np.testing.assert_allclose(before.scaler.mean_, after.scaler.mean_)
    np.testing.assert_allclose(before.ridge.coef_, after.ridge.coef_)


def test_nested_loeo_fails_closed_when_three_experiments_leave_no_inner_fold() -> None:
    from defrost_event_models.ridge_models import fit_model_for_heldout_experiment

    events = _events().loc[lambda frame: frame["experiment_id"].isin(["a", "b", "c"])]

    with pytest.raises(ValueError, match="no evaluable inner LOEO folds"):
        fit_model_for_heldout_experiment(
            events, "c", ("x",), "defrost_event_electricity_observed_kwh"
        )


def test_experiment_balanced_mean_api_builds_one_target_model_without_outer_folds() -> None:
    from defrost_event_models.ridge_models import mean_outcome_model

    training = _events().loc[lambda frame: ~frame["experiment_id"].eq("heldout")]
    artifact = mean_outcome_model(training, "defrost_event_electricity_observed_kwh")

    assert "folds" not in artifact
    assert artifact["target"] == "defrost_event_electricity_observed_kwh"
    assert artifact["training_experiment_count"] == 3
    assert artifact["intercept"] == pytest.approx(6.0)


def test_artifact_round_trip_and_missing_fold_fail_closed() -> None:
    from defrost_event_models.ridge_models import (
        assemble_target_model,
        fit_model_for_heldout_experiment,
        fit_model_on_all_experiments,
        predict_with_heldout_event_model,
    )

    events = _events()
    folds = {
        experiment: fit_model_for_heldout_experiment(
            events, experiment, ("x",), "defrost_event_electricity_observed_kwh"
        )
        for experiment in events["experiment_id"].unique()
    }
    artifact = assemble_target_model(
        "defrost_event_electricity_observed_kwh",
        ("x",),
        folds,
        fit_model_on_all_experiments(events, ("x",), "defrost_event_electricity_observed_kwh"),
    )
    candidates = pd.DataFrame({"x": [1.5, 3.5]})
    memory = folds["heldout"].predict(candidates)
    replay = predict_with_heldout_event_model(artifact, candidates, "heldout")
    np.testing.assert_allclose(replay["prediction"], memory, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="no retrospective fold"):
        predict_with_heldout_event_model(artifact, candidates, "unknown")


def test_independent_support_is_intersection() -> None:
    from defrost_event_models.ridge_models import predict_independent_targets

    fold = {
        "feature_order": ["x"],
        "imputer_median": [0.0],
        "scaler_mean": [0.0],
        "scaler_scale": [1.0],
        "coefficients": [1.0],
        "intercept": 0.0,
        "training_standardized_references": [[0.0]],
    }
    e_artifact = {
        "folds": {"a": {**fold, "support_threshold": 1.0}},
    }
    q_artifact = {
        "folds": {"a": {**fold, "support_threshold": 0.5}},
    }
    result = predict_independent_targets(
        e_artifact, q_artifact, pd.DataFrame({"x": [0.2, 0.8]}), "a"
    )
    assert result["defrost_event_electricity_in_training_domain"].tolist() == [True, True]
    assert result["defrost_event_net_heat_in_training_domain"].tolist() == [True, False]


def test_validation_replays_only_targets_present_in_each_model(monkeypatch) -> None:
    from defrost_event_models import validation as validate_v2_6_8

    events = pd.DataFrame(
        {
            "event_id": ["new", "excluded"],
            "cycle_name": ["cycle_new", "cycle_excluded"],
            "experiment_id": ["a", "a"],
            "event_valid": [True, False],
            "event_invalid_reason": ["", "bad_boundary"],
            "defrost_event_electricity_observed_kwh": [1.0, np.nan],
            "defrost_event_net_heat_observed_kwh": [2.0, np.nan],
            "defrost_event_compressor_electricity_observed_kwh": [3.0, np.nan],
            "defrost_event_duration_observed_minutes": [4.0, np.nan],
        }
    )
    models = {
        "models": {
            "legacy": {
                "event_electricity": {"name": "event_electricity"},
                "event_net_heat": {"name": "event_net_heat"},
            },
            "current": {
                name: {"name": name}
                for name in (
                    "event_electricity",
                    "event_net_heat",
                    "event_compressor_electricity",
                    "event_duration",
                )
            },
        }
    }
    values = {
        "event_electricity": 10.0,
        "event_net_heat": 20.0,
        "event_compressor_electricity": 30.0,
        "event_duration": 40.0,
    }

    monkeypatch.setattr(
        validate_v2_6_8,
        "predict_with_heldout_event_model",
        lambda artifact, *_: pd.DataFrame(
            {"prediction": [values[artifact["name"]]], "support_distance": [0.25]}
        ),
    )

    result = validate_v2_6_8.build_validation_table(events, models)

    assert len(result) == 3
    legacy = result.loc[result["model_name"].eq("legacy")].iloc[0]
    assert legacy["defrost_event_electricity_prediction_kwh"] == 10
    assert legacy["defrost_event_net_heat_prediction_kwh"] == 20
    assert pd.isna(legacy["defrost_event_compressor_electricity_prediction_kwh"])
    current = result.loc[result["model_name"].eq("current")].iloc[0]
    assert current["defrost_event_compressor_electricity_prediction_kwh"] == 30
    assert current["defrost_event_duration_prediction_minutes"] == 40
    assert current["defrost_event_compressor_electricity_training_distance"] == 0.25
    assert current["defrost_event_duration_training_distance"] == 0.25
    assert result.loc[result["model_name"].eq("excluded_event"), "event_id"].tolist() == [
        "excluded"
    ]


def test_validation_keeps_an_event_valid_for_only_one_outcome(monkeypatch) -> None:
    from defrost_event_models import validation as validate_v2_6_8

    events = pd.DataFrame(
        {
            "event_id": ["energy_only"],
            "cycle_name": ["cycle"],
            "experiment_id": ["a"],
            "event_valid": [False],
            "energy_event_valid": [True],
            "heat_event_valid": [False],
            "defrost_event_electricity_observed_kwh": [1.0],
            "defrost_event_net_heat_observed_kwh": [np.nan],
            "event_invalid_reason": ["Q_R_coverage"],
        }
    )
    models = {"models": {"model": {"event_electricity": {"name": "event_electricity"}}}}
    monkeypatch.setattr(
        validate_v2_6_8,
        "predict_with_heldout_event_model",
        lambda *_: pd.DataFrame({"prediction": [1.1], "support_distance": [0.2]}),
    )

    result = validate_v2_6_8.build_validation_table(events, models)

    assert result.loc[0, "model_name"] == "model"
    assert result.loc[0, "defrost_event_electricity_prediction_kwh"] == 1.1
    assert result.loc[0, "energy_event_valid"]
    assert not result.loc[0, "heat_event_valid"]


def test_selection_composes_candidate_quantities_objectives_and_pareto(monkeypatch) -> None:
    import select_defrost_time as selection_command

    base = pd.DataFrame(
        {
            "candidate_defrost_time": pd.date_range("2026-01-01", periods=2, freq="10s"),
            "feature": [1.0, 2.0],
        }
    )
    models = {
        "models": {
            "ridge_dynamic_state_8": {
                "event_electricity": {},
                "event_net_heat": {},
                "event_compressor_electricity": {"folds": {"exp": {"support_threshold": 0.5}}},
                "event_duration": {"folds": {"exp": {"support_threshold": 0.5}}},
            }
        }
    }

    class Loader:
        def get_cycle_record(self, _: str) -> dict[str, object]:
            return {
                "experiment_id": "exp",
                "boundaries": {"stable_heating_start": "2026-01-01 00:00:10"},
            }

    calls: dict[str, object] = {}
    monkeypatch.setattr(
        selection_command,
        "build_candidate_quantities",
        lambda loader, cycle, source, *, candidate_step_seconds: (
            calls.update(step=candidate_step_seconds, source=source) or base.copy()
        ),
    )

    def predict(model, values, experiment):
        calls.setdefault("predictions", []).append((model, values.copy(), experiment))
        return pd.DataFrame({"prediction": [3.0, 4.0], "support_distance": [0.1, 0.7]})

    monkeypatch.setattr(selection_command, "predict_with_heldout_event_model", predict)
    monkeypatch.setattr(
        selection_command,
        "calculate_performance_objectives",
        lambda values, **kwargs: (
            calls.update(objectives=values.copy(), objective_options=kwargs)
            or values.assign(cycle_cop=1.0, cycle_heating_rate_kw=2.0)
        ),
    )
    monkeypatch.setattr(selection_command, "add_single_objective_optima", lambda values: values)
    monkeypatch.setattr(
        selection_command,
        "select_cop_heating_rate_pareto_knee",
        lambda values, **kwargs: (
            calls.update(selection_options=kwargs)
            or values.assign(
                selected_defrost_time=pd.NaT,
                selection_method="cop_heating_rate_pareto_knee",
            )
        ),
    )

    result = selection_command.calculate_cycle(
        Loader(),
        "cycle",
        models,
        candidate_step_seconds=10,
        allow_extrapolation=False,
    )

    objectives = calls["objectives"]
    assert objectives["defrost_event_compressor_electricity_kwh"].tolist() == [3.0, 4.0]
    assert objectives["defrost_event_duration_minutes"].tolist() == [3.0, 4.0]
    assert objectives["defrost_event_compressor_electricity_in_training_domain"].tolist() == [
        True,
        False,
    ]
    assert objectives["defrost_event_duration_in_training_domain"].tolist() == [True, False]
    assert calls["selection_options"] == {
        "minimum_time": "2026-01-01 00:00:10",
    }
    assert calls["objective_options"] == {"allow_model_extrapolation": False}
    assert calls["step"] == 10
    assert result["selection_method"].eq("cop_heating_rate_pareto_knee").all()
    assert "decision_method" not in result
    assert "label_eligible" not in result
