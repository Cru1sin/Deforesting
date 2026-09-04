"""Sklearn classifiers and one frozen-feature LOEO fold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from image_models.evaluation import classification_metrics


def build_classifier(name: str, seed: int = 0) -> Pipeline:
    classifiers = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1_000, random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1
        ),
        "rbf_svm": SVC(kernel="rbf", class_weight="balanced"),
    }
    if name not in classifiers:
        raise ValueError(f"unknown frozen classifier: {name}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifiers[name]),
        ]
    )


def split_heldout_experiment(
    rows: pd.DataFrame, heldout_experiment: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    heldout = rows["experiment_id"].astype(str).eq(heldout_experiment)
    return rows.loc[~heldout].copy(), rows.loc[heldout].copy()


def _base_metrics(
    *,
    train_images: int,
    test_images: int,
    heldout_experiment: str,
    image_feature: str,
    classifier: str,
    camera: str,
    input_feature: str,
) -> dict[str, Any]:
    return {
        "held_out_experiment": heldout_experiment,
        "image_feature": image_feature,
        "classifier": classifier,
        "camera": camera,
        "input_feature": input_feature,
        "train_images": train_images,
        "test_images": test_images,
        "status": "ok",
        "message": "",
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
    }


def train_frozen_feature_fold(
    rows: pd.DataFrame,
    feature_columns: list[str],
    *,
    heldout_experiment: str,
    classifier: str,
    image_feature: str,
    camera: str,
    input_feature: str,
    task: str,
    return_model: bool = False,
    seed: int = 0,
    test_rows: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Fit one frozen-feature fold and return data for the main process to write."""
    train, _ = split_heldout_experiment(rows, heldout_experiment)
    _, test = split_heldout_experiment(rows if test_rows is None else test_rows, heldout_experiment)
    expected = set(range(2 if task == "binary" else 3))
    metrics = _base_metrics(
        train_images=len(train),
        test_images=len(test),
        heldout_experiment=heldout_experiment,
        image_feature=image_feature,
        classifier=classifier,
        camera=camera,
        input_feature=input_feature,
    )
    present = set(train["target"].astype(int))
    if present != expected:
        metrics.update(
            status="invalid",
            message=f"train classes {sorted(present)}; expected {sorted(expected)}",
        )
        return {"metrics": metrics, "predictions": pd.DataFrame(), "model": None}

    model = build_classifier(classifier, seed=seed)
    model.fit(train[feature_columns], train["target"])
    prediction = model.predict(test[feature_columns]).astype(int)
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(test[feature_columns]))
    else:
        scores = np.asarray(model.decision_function(test[feature_columns]))
        if scores.ndim == 1:
            scores = 1 / (1 + np.exp(-scores))

    prediction_columns = [
        column
        for column in (
            "experiment_id",
            "cycle_name",
            "camera_role",
            "image_path",
            "file_name",
            "image_time",
            "relative_regret",
            "selected_defrost_time",
            "target",
        )
        if column in test
    ]
    predictions = test[prediction_columns].reset_index(drop=True).copy()
    predictions["held_out_experiment"] = heldout_experiment
    predictions["camera"] = camera
    predictions["image_feature"] = image_feature
    predictions["classifier"] = classifier
    predictions["input_feature"] = input_feature
    predictions["prediction"] = prediction
    if scores.ndim == 1:
        predictions["decision_score"] = scores
    elif scores.shape[1] == 2:
        predictions["decision_score"] = scores[:, 1]
    else:
        predictions["decision_score"] = scores.max(axis=1)
        for class_index in range(scores.shape[1]):
            predictions[f"decision_score_{class_index}"] = scores[:, class_index]

    target = test["target"].to_numpy(dtype=int)
    metrics.update(classification_metrics(target, prediction, task))
    return {
        "metrics": metrics,
        "predictions": predictions,
        "model": model if return_model else None,
    }
