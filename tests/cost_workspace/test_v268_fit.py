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
                    "E_T_observed_kwh": 1 + 2 * x,
                    "Q_T_observed_kwh": -2 + 0.5 * x,
                    "E_comp_T_observed_kwh": 0.5 + 0.25 * x,
                    "D_T_observed_minutes": 10 + x,
                }
            )
    return pd.DataFrame(rows)


def test_outcome_targets_are_the_single_explicit_four_target_contract() -> None:
    from cost.fit_v2_6_8 import OUTCOME_TARGETS, OUTCOME_VALIDITY

    assert OUTCOME_TARGETS == {
        "energy": "E_T_observed_kwh",
        "heat": "Q_T_observed_kwh",
        "compressor_energy": "E_comp_T_observed_kwh",
        "duration": "D_T_observed_minutes",
    }
    assert OUTCOME_VALIDITY == {
        "energy": "energy_event_valid",
        "heat": "heat_event_valid",
        "compressor_energy": "compressor_event_valid",
        "duration": "duration_event_valid",
    }


def test_each_outcome_selects_its_own_valid_events() -> None:
    from cost.fit_v2_6_8 import valid_outcome_events

    events = pd.DataFrame(
        {
            "event_id": ["energy_only", "heat_only"],
            "energy_event_valid": [True, False],
            "heat_event_valid": [False, True],
            "E_T_observed_kwh": [1.0, np.nan],
            "Q_T_observed_kwh": [np.nan, 2.0],
        }
    )

    assert valid_outcome_events(events, "energy")["event_id"].tolist() == ["energy_only"]
    assert valid_outcome_events(events, "heat")["event_id"].tolist() == ["heat_only"]


def test_experiment_weights_equalize_group_mass() -> None:
    from cost.fit_v2_6_8 import experiment_weights

    groups = pd.Series(["a", "a", "a", "b", "c", "c"])
    weights = experiment_weights(groups)
    assert weights.sum() == pytest.approx(6)
    np.testing.assert_allclose(pd.Series(weights).groupby(groups).sum(), [2, 2, 2])


def test_heldout_targets_cannot_change_fold_model_or_alpha() -> None:
    from cost.fit_v2_6_8 import fit_outcome_fold

    events = _events()
    before = fit_outcome_fold(events, "heldout", ("x",), "E_T_observed_kwh")
    changed = events.copy()
    changed.loc[changed["experiment_id"].eq("heldout"), "E_T_observed_kwh"] += 10000
    after = fit_outcome_fold(changed, "heldout", ("x",), "E_T_observed_kwh")
    assert before.alpha == after.alpha
    np.testing.assert_allclose(before.scaler.mean_, after.scaler.mean_)
    np.testing.assert_allclose(before.ridge.coef_, after.ridge.coef_)


def test_nested_loeo_fails_closed_when_three_experiments_leave_no_inner_fold() -> None:
    from cost.fit_v2_6_8 import fit_outcome_fold

    events = _events().loc[lambda frame: frame["experiment_id"].isin(["a", "b", "c"])]

    with pytest.raises(ValueError, match="no evaluable inner LOEO folds"):
        fit_outcome_fold(events, "c", ("x",), "E_T_observed_kwh")


def test_experiment_mean_api_builds_one_target_model_without_outer_folds() -> None:
    from cost.fit_v2_6_8 import mean_outcome_artifact

    training = _events().loc[lambda frame: ~frame["experiment_id"].eq("heldout")]
    artifact = mean_outcome_artifact(training, "E_T_observed_kwh")

    assert "folds" not in artifact
    assert artifact["target"] == "E_T_observed_kwh"
    assert artifact["training_experiment_count"] == 3
    assert artifact["intercept"] == pytest.approx(6.0)


def test_artifact_round_trip_and_missing_fold_fail_closed() -> None:
    from cost.fit_v2_6_8 import (
        assemble_target_artifact,
        fit_full_outcome,
        fit_outcome_fold,
        predict_from_artifact,
    )

    events = _events()
    folds = {
        experiment: fit_outcome_fold(events, experiment, ("x",), "E_T_observed_kwh")
        for experiment in events["experiment_id"].unique()
    }
    artifact = assemble_target_artifact(
        "E_T_observed_kwh",
        ("x",),
        folds,
        fit_full_outcome(events, ("x",), "E_T_observed_kwh"),
    )
    candidates = pd.DataFrame({"x": [1.5, 3.5]})
    memory = folds["heldout"].predict(candidates)
    replay = predict_from_artifact(artifact, candidates, "heldout")
    np.testing.assert_allclose(replay["prediction"], memory, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="no retrospective fold"):
        predict_from_artifact(artifact, candidates, "unknown")


def test_independent_support_is_intersection() -> None:
    from cost.fit_v2_6_8 import predict_independent_targets

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
    assert result["ET_in_support"].tolist() == [True, True]
    assert result["QT_in_support"].tolist() == [True, False]


def test_validation_replays_only_targets_present_in_each_model(monkeypatch) -> None:
    from cost import validate_v2_6_8

    events = pd.DataFrame(
        {
            "event_id": ["new", "excluded"],
            "cycle_name": ["cycle_new", "cycle_excluded"],
            "experiment_id": ["a", "a"],
            "event_valid": [True, False],
            "event_invalid_reason": ["", "bad_boundary"],
            "E_T_observed_kwh": [1.0, np.nan],
            "Q_T_observed_kwh": [2.0, np.nan],
            "E_comp_T_observed_kwh": [3.0, np.nan],
            "D_T_observed_minutes": [4.0, np.nan],
        }
    )
    artifacts = {
        "models": {
            "legacy": {"energy": {"name": "energy"}, "heat": {"name": "heat"}},
            "current": {
                name: {"name": name}
                for name in ("energy", "heat", "compressor_energy", "duration")
            },
        }
    }
    values = {"energy": 10.0, "heat": 20.0, "compressor_energy": 30.0, "duration": 40.0}

    monkeypatch.setattr(
        validate_v2_6_8,
        "predict_from_artifact",
        lambda artifact, *_: pd.DataFrame(
            {"prediction": [values[artifact["name"]]], "support_distance": [0.25]}
        ),
    )

    result = validate_v2_6_8.build_validation_table(events, artifacts)

    assert len(result) == 3
    legacy = result.loc[result["model_name"].eq("legacy")].iloc[0]
    assert legacy["E_T_prediction_kwh"] == 10
    assert legacy["Q_T_prediction_kwh"] == 20
    assert pd.isna(legacy["E_comp_T_prediction_kwh"])
    current = result.loc[result["model_name"].eq("current")].iloc[0]
    assert current["E_comp_T_prediction_kwh"] == 30
    assert current["D_T_prediction_minutes"] == 40
    assert current["Ecomp_support_distance"] == 0.25
    assert current["D_support_distance"] == 0.25
    assert result.loc[result["model_name"].eq("excluded_event"), "event_id"].tolist() == [
        "excluded"
    ]


def test_validation_keeps_an_event_valid_for_only_one_outcome(monkeypatch) -> None:
    from cost import validate_v2_6_8

    events = pd.DataFrame(
        {
            "event_id": ["energy_only"],
            "cycle_name": ["cycle"],
            "experiment_id": ["a"],
            "event_valid": [False],
            "energy_event_valid": [True],
            "heat_event_valid": [False],
            "E_T_observed_kwh": [1.0],
            "Q_T_observed_kwh": [np.nan],
            "event_invalid_reason": ["Q_R_coverage"],
        }
    )
    artifacts = {"models": {"model": {"energy": {"name": "energy"}}}}
    monkeypatch.setattr(
        validate_v2_6_8,
        "predict_from_artifact",
        lambda *_: pd.DataFrame({"prediction": [1.1], "support_distance": [0.2]}),
    )

    result = validate_v2_6_8.build_validation_table(events, artifacts)

    assert result.loc[0, "model_name"] == "model"
    assert result.loc[0, "E_T_prediction_kwh"] == 1.1
    assert result.loc[0, "energy_event_valid"]
    assert not result.loc[0, "heat_event_valid"]


def test_cho_composes_shared_candidates_four_predictions_objectives_and_policy(monkeypatch) -> None:
    from cost import cho

    base = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01", periods=2, freq="10s"),
            "feature": [1.0, 2.0],
        }
    )
    artifacts = {
        "models": {
            "ticket_ridge_dynamic8": {
                "energy": {},
                "heat": {},
                "compressor_energy": {"folds": {"exp": {"support_threshold": 0.5}}},
                "duration": {"folds": {"exp": {"support_threshold": 0.5}}},
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
        cho.cost_function_v2_6_8,
        "build_candidate_outcomes",
        lambda loader, cycle, recipe, source, *, candidate_step_seconds: (
            calls.update(step=candidate_step_seconds, recipe=recipe, source=source) or base.copy()
        ),
    )

    def predict(artifact, values, experiment):
        calls.setdefault("predictions", []).append((artifact, values.copy(), experiment))
        return pd.DataFrame({"prediction": [3.0, 4.0], "support_distance": [0.1, 0.7]})

    monkeypatch.setattr(cho, "predict_from_artifact", predict)
    monkeypatch.setattr(
        cho,
        "build_objectives",
        lambda values: calls.update(objectives=values.copy()) or values.assign(C=1.0, H=2.0),
    )
    monkeypatch.setattr(cho, "add_single_objective_diagnostics", lambda values: values)
    monkeypatch.setattr(
        cho,
        "select_ch_pareto_knee",
        lambda values, **kwargs: calls.update(policy=kwargs) or values.assign(selected_time=pd.NaT),
    )

    result = cho.calculate_cycle(Loader(), "cycle", artifacts, step_seconds=10)

    objectives = calls["objectives"]
    assert objectives["transition_compressor_energy_kwh"].tolist() == [3.0, 4.0]
    assert objectives["transition_duration_minutes"].tolist() == [3.0, 4.0]
    assert objectives["EcompT_in_support"].tolist() == [True, False]
    assert objectives["DT_in_support"].tolist() == [True, False]
    assert calls["policy"] == {
        "minimum_time": "2026-01-01 00:00:10",
        "allow_extrapolation": False,
    }
    assert calls["step"] == 10
    assert result["algorithm"].eq("ch_pareto_knee").all()
    assert result["base_method"].eq("ch_pareto_knee").all()
    assert not result["label_eligible"].any()
    assert not result["hard_label_eligible"].any()
