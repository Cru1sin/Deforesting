"""Sklearn heads and one frozen-feature LOEO fold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_head(name: str) -> Pipeline:
    classifiers = {
        "logistic": LogisticRegression(
            class_weight="balanced", max_iter=1_000, random_state=0
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=0, n_jobs=1
        ),
        "rbf_svm": SVC(kernel="rbf", class_weight="balanced"),
    }
    if name not in classifiers:
        raise ValueError(f"unknown frozen head: {name}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifiers[name]),
        ]
    )


def split_heldout(
    rows: pd.DataFrame, heldout_experiment: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    heldout = rows["experiment_id"].astype(str).eq(heldout_experiment)
    return rows.loc[~heldout].copy(), rows.loc[heldout].copy()


def _base_metrics(
    *,
    train_images: int,
    test_images: int,
    heldout_experiment: str,
    representation: str,
    head: str,
    camera: str,
    modality: str,
) -> dict[str, Any]:
    return {
        "held_out_experiment": heldout_experiment,
        "representation": representation,
        "head": head,
        "camera": camera,
        "modality": modality,
        "train_images": train_images,
        "test_images": test_images,
        "status": "ok",
        "message": "",
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
    }


def train_frozen_fold(
    rows: pd.DataFrame,
    feature_columns: list[str],
    *,
    heldout_experiment: str,
    head: str,
    representation: str,
    camera: str,
    modality: str,
    task: str,
    return_model: bool = False,
) -> dict[str, Any]:
    """Fit one frozen-feature fold and return data for the main process to write."""
    train, test = split_heldout(rows, heldout_experiment)
    expected = set(range(2 if task == "binary" else 3))
    metrics = _base_metrics(
        train_images=len(train),
        test_images=len(test),
        heldout_experiment=heldout_experiment,
        representation=representation,
        head=head,
        camera=camera,
        modality=modality,
    )
    for split_name, split in (("train", train), ("test", test)):
        present = set(split["target"].astype(int))
        if present != expected:
            metrics.update(
                status="invalid",
                message=f"{split_name} classes {sorted(present)}; expected {sorted(expected)}",
            )
            return {"metrics": metrics, "predictions": pd.DataFrame(), "model": None}

    model = make_head(head)
    model.fit(train[feature_columns], train["target"])
    prediction = model.predict(test[feature_columns]).astype(int)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(test[feature_columns]))
    else:
        scores = np.asarray(model.predict_proba(test[feature_columns]))

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
            "target",
        )
        if column in test
    ]
    predictions = test[prediction_columns].reset_index(drop=True).copy()
    predictions["held_out_experiment"] = heldout_experiment
    predictions["representation"] = representation
    predictions["head"] = head
    predictions["modality"] = modality
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
    metrics.update(
        accuracy=float(accuracy_score(target, prediction)),
        balanced_accuracy=float(balanced_accuracy_score(target, prediction)),
        macro_f1=float(
            f1_score(
                target,
                prediction,
                labels=sorted(expected),
                average="macro",
                zero_division=0,
            )
        ),
    )
    return {
        "metrics": metrics,
        "predictions": predictions,
        "model": model if return_model else None,
    }
