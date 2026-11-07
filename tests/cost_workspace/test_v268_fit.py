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


def test_experiment_weights_equalize_group_mass() -> None:
    from defrost_event_models.event_electricity_ridge import experiment_weights

    groups = pd.Series(["a", "a", "a", "b", "c", "c"])
    weights = experiment_weights(groups)
    assert weights.sum() == pytest.approx(6)
    np.testing.assert_allclose(pd.Series(weights).groupby(groups).sum(), [2, 2, 2])


def test_heldout_targets_cannot_change_fold_model_or_alpha() -> None:
    from defrost_event_models.event_electricity_ridge import fit_model_for_heldout_experiment

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
    from defrost_event_models.event_electricity_ridge import fit_model_for_heldout_experiment

    events = _events().loc[lambda frame: frame["experiment_id"].isin(["a", "b", "c"])]

    with pytest.raises(ValueError, match="no evaluable inner LOEO folds"):
        fit_model_for_heldout_experiment(
            events, "c", ("x",), "defrost_event_electricity_observed_kwh"
        )


def test_artifact_round_trip_and_missing_fold_fail_closed() -> None:
    from defrost_event_models.event_electricity_ridge import (
        assemble_target_model,
        fit_model_for_heldout_experiment,
        fit_model_on_all_experiments,
        predict_with_event_model,
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
    replay = predict_with_event_model(
        artifact, candidates, "heldout", prediction_mode="cross-fitted"
    )
    np.testing.assert_allclose(replay["prediction"], memory, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="no retrospective fold"):
        predict_with_event_model(artifact, candidates, "unknown", prediction_mode="cross-fitted")


def test_full_model_predicts_an_unseen_experiment() -> None:
    from defrost_event_models.event_electricity_ridge import predict_with_event_model

    parameters = {
        "feature_order": ["x"],
        "imputer_median": [0.0],
        "scaler_mean": [0.0],
        "scaler_scale": [1.0],
        "coefficients": [2.0],
        "intercept": 1.0,
        "support_threshold": 3.0,
        "training_standardized_references": [[0.0]],
    }
    model = {"folds": {}, "full_data_model": parameters}

    result = predict_with_event_model(
        model,
        pd.DataFrame({"x": [2.0]}),
        "new_experiment",
        prediction_mode="full-model",
    )

    assert result.loc[0, "prediction"] == pytest.approx(5.0)
    assert result.loc[0, "support_threshold"] == pytest.approx(3.0)


def test_joint_event_training_saves_only_admitted_events_and_disjoint_folds(tmp_path):
    import json
    from types import SimpleNamespace

    from defrost_event_models.event_electricity_ridge import (
        THREE_INPUTS,
        fit_preparation_defrost_model,
        prepare_training_events,
    )

    events = _events().assign(E_prep_valid=True, E_D_valid=True, E_prep_kwh=0.1)
    events["E_D_kwh"] = 0.5 + events.x / 10
    for index, feature in enumerate(THREE_INPUTS):
        events[feature] = events.x + index
        events[feature + "_valid_second_count"] = 60
    events.loc[0, "E_prep_valid"] = False
    loader = SimpleNamespace(
        registry={"resample_interval_seconds": 1, "recovery_edit": {"mode": "dataset"}},
        list_cycles=lambda: events[["experiment_id"]],
    )
    fit_preparation_defrost_model(
        loader, prepare_training_events(events), tmp_path, workers=1, figures=False
    )
    saved = json.loads((tmp_path / "candidate_model_parameters.json").read_text())
    assert saved["training_events"] == 15
    for experiment, phase in saved["phase_models"]["folds"].items():
        model = phase["E_prep_D_kwh"]
        assert experiment not in model["training_experiment_ids"]
        assert events.event_id.iloc[0] not in model["training_event_ids"]
    validation = pd.read_csv(tmp_path / "model_validation.csv")
    assert len(validation) == 15
    np.testing.assert_allclose(validation.E_prep_D_kwh, validation.E_prep_kwh + validation.E_D_kwh)
    assert np.isfinite(validation.predicted_E_prep_D_kwh).all()


def test_selection_uses_only_supported_cop_and_earliest_tie(monkeypatch):
    from defrost_decision import candidate_quantities as command

    start = pd.Timestamp("2026-01-01")

    class Loader:
        def get_cycle_record(self, _):
            return {
                "experiment_id": "exp",
                "status": "valid",
                "boundaries": {
                    "heating_start": start,
                    "stable_heating_start": start,
                    "defrost_preparation_start": start + pd.Timedelta(minutes=6),
                },
            }

        def load_cycle_measurements(self, _):
            return pd.DataFrame({"timestamp": [start, start + pd.Timedelta(minutes=6)]})

    def candidates(frame, times, *args, **kwargs):
        return pd.DataFrame(
            {
                "candidate_defrost_time": times,
                "cycle_cop": 3.0,
                "cycle_cop_eligible": True,
                "cycle_heating_rate_kw": range(len(times)),
            }
        )

    monkeypatch.setattr(command, "effective_candidate_cop", candidates)
    result = command.calculate_cycle(
        Loader(),
        "cycle",
        {},
        rb={"t_RB": start + pd.Timedelta(minutes=6), "rb_status": "triggered"},
    )
    assert result.is_selected.sum() == 1
    assert result.loc[result.is_selected, "candidate_defrost_time"].iloc[0] == start + pd.Timedelta(
        minutes=5
    )
    assert result.algorithm.eq("effective_cop_without_trunc").all()
