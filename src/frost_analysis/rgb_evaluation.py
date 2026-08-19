"""Experiment-held-out evaluation for compact frost-image features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def retain_high_confidence_rows(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Exclude images whose pointwise cost regret lies in the ambiguity region."""
    return frame.loc[frame["relative_regret"].gt(threshold)].copy()


def experiment_prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score held-out predictions with one row per independent experiment."""
    rows = []
    for experiment, values in predictions.groupby("experiment_id", sort=True):
        evaluable = values["target"].nunique() == 2
        if evaluable:
            incorrect_regret = values["relative_regret"].where(
                values["target"].ne(values["predicted_target"]), 0.0
            )
            scores = {
                "balanced_accuracy": balanced_accuracy_score(
                    values["target"], values["predicted_target"]
                ),
                "macro_f1": f1_score(
                    values["target"], values["predicted_target"], average="macro"
                ),
                "auroc": roc_auc_score(values["target"], values["decision_score"]),
                "balanced_misclassification_regret": incorrect_regret.groupby(
                    values["target"]
                ).mean().mean(),
            }
        else:
            scores = dict.fromkeys(
                (
                    "balanced_accuracy",
                    "macro_f1",
                    "auroc",
                    "balanced_misclassification_regret",
                ),
                float("nan"),
            )
        rows.append(
            {
                "experiment_id": experiment,
                "evaluable": evaluable,
                **scores,
                "image_count": len(values),
                "cycle_count": values["cycle_name"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_interval(
    values: pd.Series, repeats: int = 5000, seed: int = 0
) -> dict[str, float]:
    """Return a percentile interval for a mean across independent experiments."""
    array = values.dropna().to_numpy(dtype=float)
    if not len(array):
        return dict.fromkeys(("estimate", "lower", "upper"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(repeats, len(array)), replace=True).mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {"estimate": float(array.mean()), "lower": float(lower), "upper": float(upper)}


def add_cycle_time_features(frame: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Add elapsed time and normalized position in each candidate domain."""
    bounds = (
        candidates.assign(candidate_time=pd.to_datetime(candidates["candidate_time"]))
        .groupby("cycle_name", as_index=False)["candidate_time"]
        .agg(candidate_start="min", candidate_end="max")
    )
    result = frame.merge(bounds, on="cycle_name", how="left", validate="many_to_one")
    image_time = pd.to_datetime(result["image_time"])
    elapsed = (image_time - result["candidate_start"]).dt.total_seconds()
    duration = (result["candidate_end"] - result["candidate_start"]).dt.total_seconds()
    result["time_elapsed_minutes"] = elapsed / 60
    result["time_candidate_progress"] = elapsed / duration
    return result


def leave_one_experiment_out_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit the locked RBF-SVM on all but one experiment at a time."""
    feature_columns = [column for column in frame if column.startswith("feature_")]
    predictions = []
    for experiment in sorted(frame["experiment_id"].unique()):
        test = frame.loc[frame["experiment_id"].eq(experiment)].copy()
        train = frame.loc[~frame["experiment_id"].eq(experiment)]
        if train["target"].nunique() < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            SVC(C=2.0, class_weight="balanced", random_state=0),
        )
        model.fit(train[feature_columns], train["target"])
        test["predicted_target"] = model.predict(test[feature_columns])
        test["decision_score"] = model.decision_function(test[feature_columns])
        test["held_out_experiment"] = experiment
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True)
