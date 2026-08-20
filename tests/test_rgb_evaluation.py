from __future__ import annotations

import math
import warnings

import pandas as pd

from frost_analysis.rgb_evaluation import (
    MODEL_NAMES,
    add_cycle_time_features,
    bootstrap_mean_interval,
    experiment_prediction_metrics,
    high_confidence_coverage,
    leave_one_experiment_out_predictions,
    representation_columns,
    retain_high_confidence_rows,
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


def test_all_locked_models_share_the_same_held_out_protocol() -> None:
    rows = []
    for experiment, offset in (("a", 0.0), ("b", 0.1), ("c", -0.1)):
        for target, value in ((0, -2.0 + offset), (1, 2.0 + offset)):
            for repeat in range(12):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "cycle_name": experiment,
                        "camera_role": "top",
                        "target": target,
                        "feature_000": value + repeat * 0.01,
                        "feature_001": value * 0.5,
                    }
                )
    frame = pd.DataFrame(rows)

    for model_name in MODEL_NAMES:
        predictions = leave_one_experiment_out_predictions(frame, model_name=model_name)
        assert len(predictions) == len(frame)
        assert predictions["model"].eq(model_name).all()
        assert predictions["experiment_id"].eq(predictions["held_out_experiment"]).all()
        assert predictions["decision_score"].notna().all()


def test_add_cycle_time_features_uses_candidate_domain() -> None:
    rows = pd.DataFrame(
        {"cycle_name": ["cycle"], "image_time": [pd.Timestamp("2026-01-01 00:05:00")]}
    )
    candidates = pd.DataFrame(
        {
            "cycle_name": ["cycle", "cycle"],
            "candidate_time": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:10:00"]),
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


def test_retain_high_confidence_rows_uses_pointwise_regret() -> None:
    rows = pd.DataFrame({"relative_regret": [0.005, 0.01, 0.011, 0.05]})

    retained = retain_high_confidence_rows(rows, threshold=0.01)

    assert retained["relative_regret"].tolist() == [0.011, 0.05]


def test_experiment_metrics_mark_single_class_as_not_evaluable() -> None:
    predictions = pd.DataFrame(
        {
            "experiment_id": ["a", "a"],
            "cycle_name": ["cycle", "cycle"],
            "target": [0, 0],
            "predicted_target": [0, 1],
            "decision_score": [-1.0, 1.0],
            "relative_regret": [0.1, 0.2],
        }
    )

    metrics = experiment_prediction_metrics(predictions).iloc[0]

    assert not metrics["evaluable"]
    assert math.isnan(metrics["balanced_accuracy"])
    assert math.isnan(metrics["auroc"])


def test_bootstrap_mean_interval_handles_no_evaluable_experiments() -> None:
    with warnings.catch_warnings(record=True) as caught:
        interval = bootstrap_mean_interval(pd.Series([float("nan")]))

    assert all(math.isnan(value) for value in interval.values())
    assert not caught


def test_high_confidence_coverage_uses_all_candidate_domain_states() -> None:
    balance = pd.DataFrame(
        {
            "regret_threshold": [0.01] * 3,
            "camera_group": ["all"] * 3,
            "cost_state": ["pre_optimal", "near_optimal", "post_optimal"],
            "image_count": [30, 40, 30],
        }
    )

    assert high_confidence_coverage(balance, "all", 0.01) == 0.6


def test_representation_columns_select_only_requested_features() -> None:
    frame = pd.DataFrame(
        {
            "feature_000": [1.0],
            "feature_001": [2.0],
            "dinov2_000": [3.0],
            "efficientnet_000": [4.0],
            "target": [0],
        }
    )

    assert representation_columns(frame, "handcrafted") == ["feature_000", "feature_001"]
    assert representation_columns(frame, "dinov2") == ["dinov2_000"]
    assert representation_columns(frame, "efficientnet") == ["efficientnet_000"]


def test_leave_one_experiment_out_uses_requested_representation() -> None:
    rows = []
    for experiment in ("a", "b", "c"):
        for target, value in ((0, -1.0), (1, 1.0)):
            for repeat in range(3):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "cycle_name": experiment,
                        "camera_role": "top",
                        "target": target,
                        "feature_000": 0.0,
                        "dinov2_000": value + repeat * 0.01,
                    }
                )

    predictions = leave_one_experiment_out_predictions(
        pd.DataFrame(rows), model_name="logistic", representation="dinov2"
    )

    assert predictions["predicted_target"].eq(predictions["target"]).all()
    assert predictions["representation"].eq("dinov2").all()
