from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score


def _module():  # type: ignore[no-untyped-def]
    path = Path("scripts/rgb/evaluate_rgb_learning_curves.py")
    spec = importlib.util.spec_from_file_location("rgb_learning_curves", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_experiment_subsets_are_nested_and_exclude_test() -> None:
    module = _module()
    subsets = module.nested_training_sets(["a", "b", "c", "d", "e"], "c", [2, 4], 0)

    assert "c" not in subsets[4]
    assert set(subsets[2]).issubset(subsets[4])
    assert len(subsets[2]) == 2
    assert len(subsets[4]) == 4


def test_binary_learning_curve_defaults_exclude_window_logistic() -> None:
    module = _module()

    assert tuple(
        name for name in module.MODEL_NAMES if name != "window_logistic"
    ) == module.DEFAULT_MODELS
    assert tuple(module.MODEL_COLORS) == module.MODEL_NAMES


def test_three_class_plots_include_window_logistic(tmp_path, monkeypatch) -> None:
    from matplotlib.axes import Axes

    module = _module()
    summary = pd.DataFrame(
        {
            "camera_group": ["front", "front"],
            "model": ["window_logistic", "window_logistic"],
            "training_experiment_count": [2, 2],
            "metric": ["balanced_accuracy", "balanced_misclassification_regret"],
            "estimate": [0.8, 0.01],
            "lower": [0.7, 0.005],
            "upper": [0.9, 0.02],
        }
    )
    labels = []
    plot = Axes.plot

    def record_label(axis, *args, **kwargs):  # type: ignore[no-untyped-def]
        labels.append(kwargs.get("label"))
        return plot(axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", record_label)

    module.plot_learning_curves(summary, tmp_path)
    module.plot_camera_grid(summary, tmp_path, "balanced_accuracy", "Balanced accuracy")

    assert labels.count("window logistic") == 3


def test_binary_cli_rejects_explicit_window_logistic(monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_rgb_learning_curves.py", "--models", "window_logistic"],
    )

    with pytest.raises(SystemExit, match="window_logistic requires --task three"):
        module.main()


def test_cli_default_models_follow_the_classification_task(tmp_path, monkeypatch) -> None:
    module = _module()
    shards = tmp_path / "shards"
    shards.mkdir()
    pd.DataFrame(
        {
            "cost_state": ["pre_optimal", "near_optimal", "post_optimal"],
            "relative_regret": [0.1, 0.0, 0.1],
        }
    ).to_parquet(shards / "cycle.parquet")
    captured = {}

    for task in ("binary", "three"):
        def capture(*args, _task=task, **kwargs):  # type: ignore[no-untyped-def]
            captured[_task] = tuple(kwargs["models"])
            raise RuntimeError("captured")

        monkeypatch.setattr(module, "evaluate", capture)
        monkeypatch.setattr(
            "sys.argv",
            ["evaluate_rgb_learning_curves.py", "--shards", str(shards), "--task", task],
        )
        with pytest.raises(RuntimeError, match="captured"):
            module.main()

    assert captured == {
        "binary": module.DEFAULT_MODELS,
        "three": module.MODEL_NAMES,
    }


def test_learning_curve_auroc_accepts_raw_multiclass_decision_scores() -> None:
    module = _module()
    values = pd.DataFrame(
        {
            "target": [0, 0, 1, 1, 2, 2],
            "predicted_target": [0, 0, 1, 1, 2, 2],
            "relative_regret": [0.1] * 6,
            "decision_score_0": [4.0, 3.0, 2.0, 1.0, 0.0, -1.0],
            "decision_score_1": [0.0, 1.0, 4.0, 3.0, -1.0, 2.0],
            "decision_score_2": [-2.0, 0.0, 1.0, -1.0, 4.0, 3.0],
        }
    )
    expected = sum(
        roc_auc_score(values["target"].eq(class_name), values[f"decision_score_{class_name}"])
        for class_name in (0, 1, 2)
    ) / 3

    assert module.score_predictions(values, (0, 1, 2))["auroc"] == pytest.approx(expected)


def test_summary_averages_repeats_before_experiment_bootstrap() -> None:
    module = _module()
    rows = []
    for held_out in ("a", "b"):
        for repeat in (0, 1):
            rows.append(
                {
                    "camera_group": "front",
                    "model": "logistic",
                    "training_experiment_count": 2,
                    "held_out_experiment": held_out,
                    "recall_before": 0.8,
                    "recall_within": float("nan"),
                    "recall_after": 0.8,
                    "balanced_accuracy": 0.8 + 0.1 * repeat,
                    "macro_f1": 0.8,
                    "auroc": 0.9,
                    "balanced_misclassification_regret": 0.01,
                    "fit_predict_seconds": 1.0,
                    "training_cycle_count": 4,
                    "training_image_count": 100,
                }
            )
    summary = module.summarize(pd.DataFrame(rows))
    balanced = summary.loc[summary["metric"].eq("balanced_accuracy")].iloc[0]

    assert balanced["estimate"] == pytest.approx(0.85)
    assert balanced["held_out_experiment_count"] == 2


def test_summary_counts_only_evaluable_experiments_per_metric() -> None:
    module = _module()
    results = pd.DataFrame(
        {
            "camera_group": ["front", "front"],
            "model": ["logistic", "logistic"],
            "training_experiment_count": [2, 2],
            "held_out_experiment": ["a", "b"],
            "recall_before": [0.8, float("nan")],
            "recall_within": [float("nan"), float("nan")],
            "recall_after": [0.8, float("nan")],
            "balanced_accuracy": [0.8, float("nan")],
            "macro_f1": [0.8, 0.7],
            "auroc": [0.9, float("nan")],
            "balanced_misclassification_regret": [0.01, float("nan")],
            "fit_predict_seconds": [1.0, 0.0],
            "training_cycle_count": [4, 4],
            "training_image_count": [100, 100],
        }
    )

    summary = module.summarize(results).set_index("metric")

    assert summary.loc["balanced_accuracy", "held_out_experiment_count"] == 1
    assert summary.loc["macro_f1", "held_out_experiment_count"] == 2


def test_data_requirement_uses_smallest_size_within_full_accuracy_margin() -> None:
    module = _module()
    summary = pd.DataFrame(
        {
            "camera_group": ["front"] * 3,
            "model": ["logistic"] * 3,
            "training_experiment_count": [2, 4, 6],
            "metric": ["balanced_accuracy"] * 3,
            "estimate": [0.80, 0.89, 0.90],
        }
    )

    requirement = module.data_requirements(summary, margin=0.02).iloc[0]

    assert requirement["required_training_experiment_count"] == 4
    assert requirement["full_training_experiment_count"] == 6


def test_data_requirement_keeps_na_record_when_no_fold_is_evaluable() -> None:
    module = _module()
    summary = pd.DataFrame(
        {
            "camera_group": ["front", "front"],
            "model": ["logistic", "logistic"],
            "training_experiment_count": [2, 4],
            "metric": ["balanced_accuracy", "balanced_accuracy"],
            "estimate": [float("nan"), float("nan")],
        }
    )

    requirement = module.data_requirements(summary).iloc[0]

    assert requirement["full_training_experiment_count"] == 4
    assert pd.isna(requirement["full_balanced_accuracy"])
    assert pd.isna(requirement["required_training_experiment_count"])
    assert pd.isna(requirement["required_balanced_accuracy"])


def test_evaluate_records_requested_representation() -> None:
    module = _module()
    rows = []
    for experiment in ("a", "b", "c"):
        for target, value in ((0, -1.0), (1, 1.0)):
            for repeat in range(3):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "cycle_name": experiment,
                        "camera_role": "front",
                        "target": target,
                        "relative_regret": 0.1,
                        "feature_000": 0.0,
                        "dinov2_000": value + repeat * 0.01,
                    }
                )

    results = module.evaluate(
        pd.DataFrame(rows),
        camera_groups=["front"],
        models=["logistic"],
        representations=["dinov2"],
        training_sizes=[2],
        repeats=1,
        expected_classes=(0, 1),
    )

    assert results["representation"].eq("dinov2").all()
    assert results["balanced_accuracy"].eq(1.0).all()


def test_evaluate_records_actual_training_experiment_count() -> None:
    module = _module()
    rows = []
    for experiment in ("a", "b", "c"):
        for target, value in ((0, -1.0), (1, 1.0)):
            rows.append(
                {
                    "experiment_id": experiment,
                    "cycle_name": experiment,
                    "camera_role": "front" if experiment != "b" else "top",
                    "target": target,
                    "relative_regret": 0.1,
                    "feature_000": value,
                }
            )

    results = module.evaluate(
        pd.DataFrame(rows),
        camera_groups=["front"],
        models=["logistic"],
        representations=["handcrafted"],
        training_sizes=[2],
        repeats=1,
        expected_classes=(0, 1),
    )

    assert results.set_index("held_out_experiment")["training_experiment_count"].to_dict() == {
        "a": 1,
        "b": 2,
        "c": 1,
    }


def test_learning_curve_rejects_cycle_split_across_experiments() -> None:
    module = _module()
    frame = pd.DataFrame(
        {
            "experiment_id": ["a", "b"],
            "cycle_name": ["same", "same"],
            "camera_role": ["front", "front"],
            "target": [0, 1],
            "relative_regret": [0.1, 0.1],
            "feature_000": [0.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="multiple experiments"):
        module.evaluate(
            frame,
            camera_groups=["front"],
            models=["logistic"],
            representations=["handcrafted"],
            training_sizes=[1],
            repeats=1,
            expected_classes=(0, 1),
        )


def test_learning_curves_use_actual_model_classes(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    module = _module()
    rows = []
    for experiment in ("a", "b", "c"):
        for target, value in ((2, -2.0), (4, 0.0), (7, 2.0)):
            for repeat in range(3):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "cycle_name": experiment,
                        "camera_role": "front",
                        "target": target,
                        "relative_regret": 0.1,
                        "feature_000": value + repeat * 0.01,
                    }
                )
    score_columns = []
    score_predictions = module.score_predictions

    def inspect_score_columns(values, expected_classes):  # type: ignore[no-untyped-def]
        score_columns.append(
            {column for column in values if column.startswith("decision_score_")}
        )
        return score_predictions(values, expected_classes)

    monkeypatch.setattr(module, "score_predictions", inspect_score_columns)

    results = module.evaluate(
        pd.DataFrame(rows),
        camera_groups=["front"],
        models=["logistic"],
        representations=["handcrafted"],
        training_sizes=[2],
        repeats=1,
        expected_classes=(2, 4, 7),
    )

    assert results["evaluable"].all()
    assert score_columns
    assert all(
        columns == {"decision_score_2", "decision_score_4", "decision_score_7"}
        for columns in score_columns
    )


def test_learning_curves_keep_missing_class_fold_with_na_metrics() -> None:
    module = _module()
    rows = []
    for experiment, targets in (("a", (0, 1, 2)), ("b", (0, 1, 2)), ("c", (0, 1))):
        for target in targets:
            for repeat in range(3):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "cycle_name": experiment,
                        "camera_role": "front",
                        "target": target,
                        "relative_regret": 0.1,
                        "feature_000": float(target) + repeat * 0.01,
                    }
                )

    results = module.evaluate(
        pd.DataFrame(rows),
        camera_groups=["front"],
        models=["logistic"],
        representations=["handcrafted"],
        training_sizes=[2],
        repeats=1,
        expected_classes=(0, 1, 2),
    )

    complete = results.loc[results["held_out_experiment"].isin(("a", "b"))]
    missing = results.loc[results["held_out_experiment"].eq("c")].iloc[0]
    metrics = [
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "balanced_misclassification_regret",
        "recall_before",
        "recall_within",
        "recall_after",
    ]
    assert complete["evaluable"].all()
    assert complete[["recall_before", "recall_within", "recall_after"]].notna().all().all()
    assert not missing["evaluable"]
    assert missing[metrics].isna().all()


def test_learning_curve_cli_accepts_shards_without_cost_source_hash(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    module = _module()
    shards = tmp_path / "shards"
    shards.mkdir()
    rows = []
    for experiment in ("a", "b", "c"):
        for state, value in (
            ("pre_optimal", -1.0),
            ("post_optimal", 1.0),
        ):
            for repeat in range(3):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "cycle_name": experiment,
                        "camera_role": "front",
                        "cost_state": state,
                        "relative_regret": 0.1,
                        "feature_000": value + repeat * 0.01,
                    }
                )
    pd.DataFrame(rows).to_parquet(shards / "cycle.parquet")
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_rgb_learning_curves.py",
            "--shards",
            str(shards),
            "--camera-groups",
            "front",
            "--models",
            "logistic",
            "--training-sizes",
            "2",
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
    )

    module.main()

    results = pd.read_csv(output / "fold_results.csv")
    assert len(results) == 3
    assert results["evaluable"].all()
