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
                }
            )
    return pd.DataFrame(rows)


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
    assert result["model_supported"].tolist() == [True, False]
