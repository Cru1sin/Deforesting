from __future__ import annotations

import pandas as pd

from frost_analysis.rgb_evaluation import (
    add_cycle_time_features,
    bootstrap_mean_interval,
    experiment_prediction_metrics,
    leave_one_experiment_out_predictions,
)


def test_leave_one_experiment_out_predicts_every_row_once() -> None:
    rows = []
    for experiment, offset in (("a", 0.0), ("b", 0.2), ("c", -0.2)):
        for target, value in ((0, -1.0 + offset), (1, 1.0 + offset)):
            for repeat in range(3):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "cycle_name": f"{experiment}_{repeat}",
                        "camera_role": "top",
                        "target": target,
                        "feature_000": value + repeat * 0.01,
                    }
                )

    predictions = leave_one_experiment_out_predictions(pd.DataFrame(rows))

    assert len(predictions) == len(rows)
    assert predictions.groupby("experiment_id")["held_out_experiment"].nunique().eq(1).all()
    assert predictions["experiment_id"].eq(predictions["held_out_experiment"]).all()
    assert set(predictions["predicted_target"]) == {0, 1}


def test_add_cycle_time_features_uses_candidate_domain() -> None:
    rows = pd.DataFrame(
        {"cycle_name": ["cycle"], "image_time": [pd.Timestamp("2026-01-01 00:05:00")]}
    )
    candidates = pd.DataFrame(
        {
            "cycle_name": ["cycle", "cycle"],
            "candidate_time": pd.to_datetime(
                ["2026-01-01 00:00:00", "2026-01-01 00:10:00"]
            ),
        }
    )

    enriched = add_cycle_time_features(rows, candidates)

    assert enriched["time_elapsed_minutes"].iloc[0] == 5
    assert enriched["time_candidate_progress"].iloc[0] == 0.5


def test_experiment_metrics_balance_classes_and_regret() -> None:
    predictions = pd.DataFrame(
        {
            "experiment_id": ["a"] * 4,
            "cycle_name": ["cycle"] * 4,
            "target": [0, 0, 1, 1],
            "predicted_target": [0, 1, 0, 1],
            "decision_score": [-2.0, 1.0, -1.0, 2.0],
            "relative_regret": [0.1, 0.2, 0.3, 0.4],
        }
    )

    metrics = experiment_prediction_metrics(predictions).iloc[0]

    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["balanced_misclassification_regret"] == 0.125
    assert metrics["image_count"] == 4


def test_bootstrap_mean_interval_is_reproducible() -> None:
    first = bootstrap_mean_interval(pd.Series([0.7, 0.8, 0.9]), repeats=200, seed=3)
    second = bootstrap_mean_interval(pd.Series([0.7, 0.8, 0.9]), repeats=200, seed=3)

    assert first == second
    assert first["lower"] <= first["estimate"] <= first["upper"]
