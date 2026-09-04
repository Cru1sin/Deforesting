from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from image_models.classifiers import (
    build_classifier,
    split_heldout_experiment,
    train_frozen_feature_fold,
)


def _rows(*, missing_heldout_class: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment_index, experiment in enumerate(("a", "b", "c")):
        targets = (0,) if missing_heldout_class and experiment == "c" else (0, 1)
        for target in targets:
            rows.append(
                {
                    "experiment_id": experiment,
                    "cycle_name": f"cycle_{experiment}",
                    "camera_role": "front",
                    "image_path": f"{experiment}-{target}.png",
                    "image_time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=target),
                    "relative_regret": 0.2,
                    "target": target,
                    "feature_000": float(target * 3 + experiment_index * 0.01),
                    "feature_001": np.nan if target == 0 else float(target),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "rbf_svm"])
def test_make_head_uses_the_required_small_sklearn_pipeline(name: str) -> None:
    pipeline = build_classifier(name, seed=17)

    assert list(pipeline.named_steps) == ["imputer", "scaler", "classifier"]
    if name == "random_forest":
        assert pipeline.named_steps["classifier"].n_jobs == 1
    if name in {"logistic_regression", "random_forest"}:
        assert pipeline.named_steps["classifier"].random_state == 17


def test_split_heldout_never_leaks_the_test_experiment_into_training() -> None:
    train, test = split_heldout_experiment(_rows(), "b")

    assert set(train["experiment_id"]) == {"a", "c"}
    assert set(test["experiment_id"]) == {"b"}
    assert not set(train["experiment_id"]) & set(test["experiment_id"])


def test_small_frozen_fold_really_trains_and_returns_predictions() -> None:
    result = train_frozen_feature_fold(
        _rows(),
        ["feature_000", "feature_001"],
        heldout_experiment="c",
        classifier="rbf_svm",
        image_feature="color_gradient",
        camera="front",
        input_feature="image_only",
        task="binary",
        return_model=True,
    )

    assert result["metrics"]["status"] == "ok"
    assert result["metrics"]["train_images"] == 4
    assert result["metrics"]["test_images"] == 2
    assert result["model"] is not None
    predictions = result["predictions"]
    assert len(predictions) == 2
    assert {
        "experiment_id",
        "cycle_name",
        "camera_role",
        "image_time",
        "held_out_experiment",
        "image_feature",
        "classifier",
        "input_feature",
        "target",
        "prediction",
        "decision_score",
    }.issubset(predictions.columns)
    assert predictions["held_out_experiment"].eq("c").all()
    assert predictions["camera"].eq("front").all()


def test_missing_test_class_is_still_evaluated_with_full_label_metrics() -> None:
    result = train_frozen_feature_fold(
        _rows(missing_heldout_class=True),
        ["feature_000", "feature_001"],
        heldout_experiment="c",
        classifier="logistic_regression",
        image_feature="color_gradient",
        camera="front",
        input_feature="image_only",
        task="binary",
    )

    assert result["metrics"]["status"] == "ok"
    assert result["metrics"]["macro_f1"] == pytest.approx(0.5)
    assert result["metrics"]["balanced_accuracy"] == pytest.approx(0.5)
    assert len(result["predictions"]) == 1
    assert result["model"] is None


def test_missing_training_class_returns_clear_invalid_result() -> None:
    rows = _rows()
    rows = rows.loc[rows["experiment_id"].eq("c") | rows["target"].eq(0)].reset_index(drop=True)

    result = train_frozen_feature_fold(
        rows,
        ["feature_000", "feature_001"],
        heldout_experiment="c",
        classifier="logistic_regression",
        image_feature="color_gradient",
        camera="front",
        input_feature="image_only",
        task="binary",
    )

    assert result["metrics"]["status"] == "invalid"
    assert "train classes" in result["metrics"]["message"]
    assert result["predictions"].empty


def test_binary_random_forest_decision_score_is_positive_class_probability() -> None:
    rows = _rows()
    columns = ["feature_000", "feature_001"]
    result = train_frozen_feature_fold(
        rows,
        columns,
        heldout_experiment="c",
        classifier="random_forest",
        image_feature="color_gradient",
        camera="front",
        input_feature="image_only",
        task="binary",
        return_model=True,
    )
    _, test = split_heldout_experiment(rows, "c")

    expected = result["model"].predict_proba(test[columns])[:, 1]
    assert np.allclose(result["predictions"]["decision_score"], expected)


def test_binary_logistic_decision_score_is_positive_class_probability() -> None:
    rows = _rows()
    columns = ["feature_000", "feature_001"]
    result = train_frozen_feature_fold(
        rows,
        columns,
        heldout_experiment="c",
        classifier="logistic_regression",
        image_feature="color_gradient",
        camera="front",
        input_feature="image_only",
        task="binary",
        return_model=True,
    )
    _, test = split_heldout_experiment(rows, "c")

    expected = result["model"].predict_proba(test[columns])[:, 1]
    assert np.allclose(result["predictions"]["decision_score"], expected)
    assert result["predictions"]["decision_score"].between(0, 1).all()
