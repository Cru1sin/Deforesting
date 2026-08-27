from __future__ import annotations

import inspect
import math
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

import frost_analysis.rgb_evaluation as rgb_evaluation
from frost_analysis.rgb_evaluation import (
    MODEL_NAMES,
    add_cycle_time_features,
    bootstrap_mean_interval,
    evaluate_holdout_task,
    experiment_prediction_metrics,
    fit_predict_rgb_model,
    high_confidence_coverage,
    leave_one_experiment_out_predictions,
    make_rgb_model,
    map_cost_state_targets,
    representation_columns,
    retain_high_confidence_rows,
)


def test_cost_state_targets_support_binary_and_three_class_tasks() -> None:
    states = pd.Series(["pre_optimal", "near_optimal", "post_optimal"])

    binary = map_cost_state_targets(states, "binary")
    three = map_cost_state_targets(states, "three")

    assert binary.tolist() == [0, pd.NA, 1]
    assert three.tolist() == [0, 1, 2]
    assert str(binary.dtype) == str(three.dtype) == "Int64"


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

    predictions = leave_one_experiment_out_predictions(pd.DataFrame(rows), expected_classes=(0, 1))

    assert len(predictions) == len(rows)
    assert predictions.groupby("experiment_id")["held_out_experiment"].nunique().eq(1).all()
    assert predictions["experiment_id"].eq(predictions["held_out_experiment"]).all()
    assert set(predictions["predicted_target"]) == {0, 1}


def test_task_result_marks_missing_fold_class_invalid() -> None:
    frame = pd.DataFrame(
        {
            "experiment_id": ["a", "a", "b", "b"],
            "cycle_name": ["a", "a", "b", "b"],
            "target": [0, 0, 0, 1],
            "feature_000": [0.0, 0.1, 0.0, 1.0],
        }
    )

    result, predictions = evaluate_holdout_task(
        frame, "a", model_name="logistic", expected_classes=(0, 1)
    )

    assert result["status"] == "invalid"
    assert "test classes" in result["message"]
    assert predictions["predicted_target"].isna().all()


def test_convergence_warning_is_counted_and_fails_the_task(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        {
            "experiment_id": ["a", "a", "b", "b"],
            "cycle_name": ["a", "a", "b", "b"],
            "target": [0, 1, 0, 1],
            "feature_000": [0.0, 1.0, 0.0, 1.0],
        }
    )

    def warns(*args, **kwargs):  # type: ignore[no-untyped-def]
        warnings.warn("did not converge", ConvergenceWarning, stacklevel=2)
        test = args[1]
        return test["target"].to_numpy(), test["target"].to_numpy(float), np.array([0, 1])

    monkeypatch.setattr(rgb_evaluation, "fit_predict_rgb_model", warns)

    result, _ = evaluate_holdout_task(frame, "a", model_name="mlp", expected_classes=(0, 1))

    assert result["status"] == "failed"
    assert result["warning_count"] == 1
    assert result["error_type"] == "ConvergenceWarning"


def test_leave_one_experiment_out_reports_each_completed_fold() -> None:
    frame = pd.DataFrame(
        [
            {
                "experiment_id": experiment,
                "cycle_name": experiment,
                "target": target,
                "feature_000": float(target),
            }
            for experiment in ("a", "b")
            for target in (0, 1)
        ]
    )
    progress = []

    leave_one_experiment_out_predictions(
        frame,
        model_name="logistic",
        expected_classes=(0, 1),
        progress=lambda *values: progress.append(values),
    )

    assert [(fold, total, experiment) for fold, total, experiment, _ in progress] == [
        (1, 2, "a"),
        (2, 2, "b"),
    ]
    assert all(elapsed >= 0 for *_, elapsed in progress)


def test_parallel_holdouts_report_completion_order_but_return_deterministically(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        [
            {
                "experiment_id": experiment,
                "cycle_name": experiment,
                "target": target,
                "feature_000": float(target),
            }
            for experiment in ("a", "b", "c")
            for target in (0, 1)
        ]
    )
    inner_jobs = []

    def fake_fit(train, test, model_name, representation, *, n_jobs):  # type: ignore[no-untyped-def]
        inner_jobs.append(n_jobs)
        time.sleep({"a": 0.06, "b": 0.03, "c": 0.0}[test["experiment_id"].iat[0]])
        return (
            test["target"].to_numpy(),
            test["target"].to_numpy(dtype=float),
            np.asarray([0, 1]),
        )

    monkeypatch.setattr(rgb_evaluation, "fit_predict_rgb_model", fake_fit)
    progress = []

    predictions = leave_one_experiment_out_predictions(
        frame,
        model_name="logistic",
        expected_classes=(0, 1),
        jobs=3,
        progress=lambda *values: progress.append(values),
    )

    assert [experiment for _, _, experiment, _ in progress] == ["c", "b", "a"]
    assert [fold for fold, *_ in progress] == [1, 2, 3]
    assert predictions["experiment_id"].tolist() == ["a", "a", "b", "b", "c", "c"]
    assert inner_jobs == [1, 1, 1]


def test_serial_holdouts_keep_model_internal_parallelism(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        {
            "experiment_id": ["a", "a", "b", "b"],
            "cycle_name": ["a", "a", "b", "b"],
            "target": [0, 1, 0, 1],
            "feature_000": [0.0, 1.0, 0.0, 1.0],
        }
    )
    inner_jobs = []

    def fake_fit(train, test, model_name, representation, *, n_jobs):  # type: ignore[no-untyped-def]
        inner_jobs.append(n_jobs)
        return (
            test["target"].to_numpy(),
            test["target"].to_numpy(dtype=float),
            np.asarray([0, 1]),
        )

    monkeypatch.setattr(rgb_evaluation, "fit_predict_rgb_model", fake_fit)

    leave_one_experiment_out_predictions(
        frame, model_name="random_forest", expected_classes=(0, 1), jobs=1
    )

    assert inner_jobs == [-1, -1]


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

    for model_name in set(MODEL_NAMES) - {"window_logistic"}:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            predictions = leave_one_experiment_out_predictions(
                frame, model_name=model_name, expected_classes=(0, 1)
            )
        assert len(predictions) == len(frame)
        assert predictions["model"].eq(model_name).all()
        assert predictions["experiment_id"].eq(predictions["held_out_experiment"]).all()
        assert predictions["decision_score"].notna().all()


def test_mlp_uses_fixed_full_fold_training_budget() -> None:
    classifier = make_rgb_model("mlp")[-1]

    assert not classifier.early_stopping
    assert classifier.max_iter == 1000
    assert classifier.n_iter_no_change == 15


def test_parallel_random_forest_uses_only_sklearn_parallel_wrapper() -> None:
    rows = []
    for experiment in ("a", "b", "c"):
        for target in (0, 1):
            for repeat in range(4):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "cycle_name": experiment,
                        "target": target,
                        "feature_000": target + repeat / 10,
                    }
                )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        leave_one_experiment_out_predictions(
            pd.DataFrame(rows),
            model_name="random_forest",
            expected_classes=(0, 1),
            jobs=3,
        )

    assert not any("sklearn.utils.parallel.delayed" in str(item.message) for item in caught)


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_holdout_parallel_backends_preserve_results(backend) -> None:  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        [
            {
                "experiment_id": experiment,
                "cycle_name": experiment,
                "target": target,
                "feature_000": float(target),
            }
            for experiment in ("a", "b", "c")
            for target in (0, 1)
        ]
    )

    result = leave_one_experiment_out_predictions(
        frame,
        model_name="logistic",
        expected_classes=(0, 1),
        jobs=2,
        backend=backend,
    )

    assert result["experiment_id"].tolist() == ["a", "a", "b", "b", "c", "c"]


def test_holdout_frames_have_stable_dtypes_before_concat(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    rows = []
    for experiment, targets in (("a", (0, 1, 2)), ("b", (0, 1)), ("c", (0, 1, 2))):
        for target in targets:
            rows.append(
                {
                    "experiment_id": experiment,
                    "cycle_name": experiment,
                    "target": target,
                    "feature_000": float(target),
                }
            )
    original_concat = pd.concat

    def audited_concat(frames, *args, **kwargs):  # type: ignore[no-untyped-def]
        frames = list(frames)
        if frames and all("predicted_target" in frame for frame in frames):
            assert all(str(frame["predicted_target"].dtype) == "Int64" for frame in frames)
            assert all(
                str(frame[column].dtype) == "float64"
                for frame in frames
                for column in frame
                if column.startswith("decision_score")
            )
        return original_concat(frames, *args, **kwargs)

    monkeypatch.setattr(rgb_evaluation.pd, "concat", audited_concat)

    leave_one_experiment_out_predictions(
        pd.DataFrame(rows), model_name="logistic", expected_classes=(0, 1, 2)
    )


def test_window_logistic_returns_ordered_three_class_probabilities() -> None:
    assert "window_logistic" in MODEL_NAMES
    model = make_rgb_model("window_logistic")
    features = pd.DataFrame({"feature": [-3.0, -2.0, -0.1, 0.1, 2.0, 3.0]})
    target = pd.Series([0, 0, 1, 1, 2, 2])

    model.fit(features, target)
    probabilities = model.predict_proba(features)

    assert model.classes_.tolist() == [0, 1, 2]
    assert probabilities.shape == (6, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.array_equal(model.predict(features), model.classes_[probabilities.argmax(axis=1)])


def test_window_logistic_requires_all_three_window_classes() -> None:
    model = make_rgb_model("window_logistic")

    with pytest.raises(ValueError, match="window_logistic requires classes 0,1,2"):
        model.fit([[0.0], [1.0]], [0, 1])


def test_window_logistic_is_a_standard_sklearn_classifier() -> None:
    classifier = make_rgb_model("window_logistic")[-1]

    assert is_classifier(classifier)
    assert tuple(inspect.signature(classifier.fit).parameters) == ("X", "y")


def test_multiclass_svc_keeps_raw_decision_scores() -> None:
    frame = pd.DataFrame(
        {
            "feature_000": [-2.0, -1.8, 0.0, 0.2, 1.8, 2.0],
            "feature_001": [0.0, 0.2, 2.0, 1.8, 0.2, 0.0],
            "target": [0, 0, 1, 1, 2, 2],
        }
    )
    model = make_rgb_model("rbf_svm").fit(frame[["feature_000", "feature_001"]], frame["target"])

    _, scores, classes = fit_predict_rgb_model(frame, frame, "rbf_svm")

    assert classes.tolist() == [0, 1, 2]
    assert np.allclose(
        scores,
        model.decision_function(frame[["feature_000", "feature_001"]]),
    )


def test_multiclass_auroc_is_the_macro_average_of_raw_ovr_rankings() -> None:
    values = pd.DataFrame(
        {
            "target": [0, 0, 1, 1, 2, 2],
            "decision_score_0": [4.0, 3.0, 2.0, 1.0, 0.0, -1.0],
            "decision_score_1": [0.0, 1.0, 4.0, 3.0, -1.0, 2.0],
            "decision_score_2": [-2.0, 0.0, 1.0, -1.0, 4.0, 3.0],
        }
    )
    expected = np.mean(
        [
            roc_auc_score(values["target"].eq(class_name), values[f"decision_score_{class_name}"])
            for class_name in (0, 1, 2)
        ]
    )

    assert rgb_evaluation._auroc(values) == pytest.approx(expected)


def test_window_logistic_uses_the_held_out_evaluation_path() -> None:
    rows = []
    for experiment in ("a", "b", "c"):
        for target, value in ((0, -3.0), (1, 0.0), (2, 3.0)):
            for repeat in range(3):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "cycle_name": experiment,
                        "target": target,
                        "feature_000": value + repeat * 0.01,
                        "feature_001": -3.0 if target == 1 else 3.0,
                    }
                )

    predictions = leave_one_experiment_out_predictions(
        pd.DataFrame(rows),
        model_name="window_logistic",
        expected_classes=(0, 1, 2),
    )

    assert len(predictions) == len(rows)
    assert predictions["predicted_target"].eq(predictions["target"]).all()
    assert {"decision_score_0", "decision_score_1", "decision_score_2"} <= set(predictions)


def test_add_cycle_time_features_uses_stable_heating_start() -> None:
    rows = pd.DataFrame(
        {"cycle_name": ["cycle"], "image_time": [pd.Timestamp("2026-01-01 00:15:00")]}
    )
    candidates = pd.DataFrame(
        {
            "cycle_name": ["cycle", "cycle"],
            "candidate_time": pd.to_datetime(["2026-01-01 00:10:00", "2026-01-01 00:20:00"]),
            "heating_hours": [1 / 6, 1 / 3],
        }
    )

    enriched = add_cycle_time_features(rows, candidates)

    assert math.isclose(enriched["time_elapsed_minutes"].iloc[0], 15)
    assert not {"candidate_start", "candidate_end", "time_candidate_progress"} & set(enriched)


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


def test_three_class_metrics_report_per_state_recall_and_balanced_regret() -> None:
    predictions = pd.DataFrame(
        {
            "experiment_id": ["a"] * 6,
            "cycle_name": ["cycle"] * 6,
            "target": [0, 0, 1, 1, 2, 2],
            "predicted_target": [0, 1, 1, 1, 0, 2],
            "decision_score_0": [0.8, 0.2, 0.1, 0.1, 0.6, 0.1],
            "decision_score_1": [0.1, 0.7, 0.8, 0.8, 0.2, 0.1],
            "decision_score_2": [0.1, 0.1, 0.1, 0.1, 0.2, 0.8],
            "relative_regret": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )

    metrics = experiment_prediction_metrics(predictions).iloc[0]

    assert metrics["evaluable"]
    assert metrics["recall_before"] == 0.5
    assert metrics["recall_within"] == 1.0
    assert metrics["recall_after"] == 0.5
    assert math.isclose(metrics["balanced_accuracy"], 2 / 3)
    assert math.isclose(metrics["macro_f1"], (0.5 + 0.8 + 2 / 3) / 3)
    assert not math.isnan(metrics["auroc"])
    assert math.isclose(metrics["balanced_misclassification_regret"], 7 / 60)


def test_three_class_metrics_make_every_metric_na_when_a_class_is_missing() -> None:
    predictions = pd.DataFrame(
        {
            "experiment_id": ["a"] * 4,
            "cycle_name": ["cycle"] * 4,
            "target": [0, 0, 1, 1],
            "predicted_target": [0, 1, 0, 1],
            "decision_score_0": [0.8, 0.2, 0.7, 0.2],
            "decision_score_1": [0.1, 0.7, 0.2, 0.7],
            "decision_score_2": [0.1, 0.1, 0.1, 0.1],
            "relative_regret": [0.1, 0.2, 0.3, 0.4],
        }
    )

    metrics = experiment_prediction_metrics(predictions).iloc[0]

    assert not metrics["evaluable"]
    for metric in (
        "recall_before",
        "recall_within",
        "recall_after",
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "balanced_misclassification_regret",
    ):
        assert math.isnan(metrics[metric])


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


def test_three_class_coverage_uses_label_and_candidate_domain_counts() -> None:
    labels = pd.DataFrame(
        {
            "cycle_name": ["cycle"] * 5,
            "camera_role": ["front"] * 5,
            "image_time": pd.to_datetime(
                [
                    "2025-12-31 23:55",
                    "2026-01-01 00:00",
                    "2026-01-01 00:05",
                    "2026-01-01 00:10",
                    "2026-01-01 00:15",
                ]
            ),
            "relative_regret": [0.1, 0.1, float("nan"), 0.2, 0.3],
        }
    )
    candidates = pd.DataFrame(
        {
            "cycle_name": ["cycle", "cycle"],
            "candidate_time": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:10"]),
        }
    )

    coverage = rgb_evaluation.three_class_eligible_image_coverage(labels, candidates, ("front",))

    assert coverage == 2 / 3


def test_representation_columns_select_only_requested_features() -> None:
    frame = pd.DataFrame(
        {
            "feature_000": [1.0],
            "feature_001": [2.0],
            "dinov2_000": [3.0],
            "efficientnet_000": [4.0],
            "mobilenet_v3_small_000": [5.0],
            "repvit_m0_9_000": [6.0],
            "convnext_tiny_000": [7.0],
            "target": [0],
        }
    )

    assert representation_columns(frame, "handcrafted") == ["feature_000", "feature_001"]
    assert representation_columns(frame, "dinov2") == ["dinov2_000"]
    assert representation_columns(frame, "efficientnet") == ["efficientnet_000"]
    assert representation_columns(frame, "mobilenet_v3_small") == ["mobilenet_v3_small_000"]
    assert representation_columns(frame, "repvit_m0_9") == ["repvit_m0_9_000"]
    assert representation_columns(frame, "convnext_tiny") == ["convnext_tiny_000"]


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
                        "file_name": f"{target}_{repeat}.jpg",
                        "image_time": pd.Timestamp("2026-01-01") + pd.Timedelta(days=repeat),
                        "cost_state": "pre_optimal" if target == 0 else "post_optimal",
                        "relative_regret": 0.1,
                        "assets": {"unused": "large nested metadata"},
                        "target": target,
                        "feature_000": 0.0,
                        "dinov2_000": value + repeat * 0.01,
                    }
                )

    predictions = leave_one_experiment_out_predictions(
        pd.DataFrame(rows),
        model_name="logistic",
        representation="dinov2",
        expected_classes=(0, 1),
    )

    assert predictions["predicted_target"].eq(predictions["target"]).all()
    assert predictions["representation"].eq("dinov2").all()
    assert not any(column.startswith(("feature_", "dinov2_")) for column in predictions)
    assert "assets" not in predictions
    assert {
        "experiment_id",
        "cycle_name",
        "camera_role",
        "file_name",
        "image_time",
        "cost_state",
        "relative_regret",
        "target",
        "predicted_target",
        "decision_score",
    } <= set(predictions)


def test_leave_one_experiment_out_supports_three_classes() -> None:
    rows = []
    for experiment in ("a", "b", "c"):
        for target, value in ((0, -2.0), (1, 0.0), (2, 2.0)):
            for repeat in range(4):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "cycle_name": experiment,
                        "camera_role": "front",
                        "target": target,
                        "feature_000": value + repeat * 0.01,
                        "relative_regret": 0.1,
                    }
                )

    predictions = leave_one_experiment_out_predictions(
        pd.DataFrame(rows), model_name="logistic", expected_classes=(0, 1, 2)
    )
    metrics = experiment_prediction_metrics(predictions)

    assert predictions["predicted_target"].eq(predictions["target"]).all()
    assert {"decision_score_0", "decision_score_1", "decision_score_2"} <= set(predictions)
    assert metrics["auroc"].eq(1.0).all()


def test_multiclass_scores_use_the_model_classes() -> None:
    rows = []
    for experiment in ("a", "b", "c"):
        for target, value in ((2, -2.0), (4, 0.0), (7, 2.0)):
            for repeat in range(4):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "cycle_name": experiment,
                        "target": target,
                        "feature_000": value + repeat * 0.01,
                    }
                )
    frame = pd.DataFrame(rows)

    predicted, score, classes = fit_predict_rgb_model(
        frame.loc[frame["experiment_id"].ne("a")],
        frame.loc[frame["experiment_id"].eq("a")],
        "logistic",
    )
    predictions = leave_one_experiment_out_predictions(
        frame, model_name="logistic", expected_classes=(2, 4, 7)
    )

    assert len(predicted) == len(frame.loc[frame["experiment_id"].eq("a")])
    assert score.shape[1] == 3
    assert classes.tolist() == [2, 4, 7]
    assert {"decision_score_2", "decision_score_4", "decision_score_7"} <= set(predictions)
    assert "decision_score_0" not in predictions


def test_three_class_holdout_marks_fold_when_test_misses_an_expected_class() -> None:
    rows = []
    for experiment, targets in (("a", (0, 1, 2)), ("b", (0, 1)), ("c", (0, 1, 2))):
        for target in targets:
            for repeat in range(3):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "cycle_name": experiment,
                        "target": target,
                        "feature_000": float(target) + repeat * 0.01,
                        "relative_regret": 0.1,
                    }
                )

    predictions = leave_one_experiment_out_predictions(
        pd.DataFrame(rows), model_name="logistic", expected_classes=(0, 1, 2)
    )

    assert set(predictions["held_out_experiment"]) == {"a", "b", "c"}
    assert predictions.groupby("held_out_experiment")["fold_evaluable"].first().to_dict() == {
        "a": True,
        "b": False,
        "c": True,
    }
    missing = predictions.loc[predictions["held_out_experiment"].eq("b"), "predicted_target"]
    assert missing.isna().all()
    assert {"decision_score_0", "decision_score_1", "decision_score_2"} <= set(predictions)
    metrics = experiment_prediction_metrics(predictions)
    assert metrics.set_index("experiment_id").loc[["a", "c"], "evaluable"].all()


def test_three_class_holdout_handles_every_fold_missing_a_class() -> None:
    rows = []
    for experiment, targets in (("a", (0, 1)), ("b", (1, 2)), ("c", (0, 2))):
        for target in targets:
            rows.append(
                {
                    "experiment_id": experiment,
                    "cycle_name": experiment,
                    "target": target,
                    "feature_000": float(target),
                    "relative_regret": 0.1,
                }
            )

    predictions = leave_one_experiment_out_predictions(
        pd.DataFrame(rows), model_name="logistic", expected_classes=(0, 1, 2)
    )
    metrics = experiment_prediction_metrics(predictions)

    assert len(predictions) == len(rows)
    assert not predictions["fold_evaluable"].any()
    assert len(metrics) == 3
    assert not metrics["evaluable"].any()
    assert {
        "recall_before",
        "recall_within",
        "recall_after",
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "balanced_misclassification_regret",
    } <= set(metrics)


def test_three_class_holdout_does_not_infer_binary_from_observed_cohort() -> None:
    rows = []
    for experiment in ("a", "b", "c"):
        for target in (0, 1):
            rows.append(
                {
                    "experiment_id": experiment,
                    "cycle_name": experiment,
                    "target": target,
                    "feature_000": float(target),
                    "relative_regret": 0.1,
                }
            )
    frame = pd.DataFrame(rows)

    predictions = leave_one_experiment_out_predictions(
        frame,
        model_name="logistic",
        expected_classes=(0, 1, 2),
    )
    metrics = experiment_prediction_metrics(predictions)

    assert len(predictions) == len(frame)
    assert not predictions["fold_evaluable"].any()
    assert predictions["predicted_target"].isna().all()
    scores = predictions[["decision_score_0", "decision_score_1", "decision_score_2"]]
    assert scores.isna().all().all()
    assert len(metrics) == 3
    assert not metrics["evaluable"].any()
    assert (
        metrics[
            [
                "recall_before",
                "recall_within",
                "recall_after",
                "balanced_accuracy",
                "macro_f1",
                "auroc",
                "balanced_misclassification_regret",
            ]
        ]
        .isna()
        .all()
        .all()
    )
