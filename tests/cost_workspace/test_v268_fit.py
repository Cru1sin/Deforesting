from __future__ import annotations

import copy

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


def test_artifact_round_trip_and_missing_fold_fail_closed() -> None:
    from cost.fit_v2_6_8 import build_model_artifact, predict_from_artifact

    artifact = build_model_artifact(_events(), ("x",), "E_T_observed_kwh")
    candidates = pd.DataFrame({"x": [1.5, 3.5]})
    memory = artifact["_models"]["heldout"].predict(candidates)
    serializable = copy.deepcopy(artifact)
    serializable.pop("_models")
    replay = predict_from_artifact(serializable, candidates, "heldout")
    np.testing.assert_allclose(replay["prediction"], memory, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="no retrospective fold"):
        predict_from_artifact(serializable, candidates, "unknown")


def test_independent_support_is_intersection() -> None:
    from cost.fit_v2_6_8 import predict_independent_targets

    e_artifact = {
        "folds": {"a": {"support_threshold": 1.0}},
    }
    q_artifact = {
        "folds": {"a": {"support_threshold": 0.5}},
    }
    e = pd.DataFrame({"prediction": [1.0, 1.0], "support_distance": [0.2, 0.2]})
    q = pd.DataFrame({"prediction": [1.0, 1.0], "support_distance": [0.2, 0.8]})
    result = predict_independent_targets(
        e_artifact, q_artifact, pd.DataFrame(index=range(2)), "a", replay=(e, q)
    )
    assert result["model_supported"].tolist() == [True, False]
