"""Experiment-held-out evaluation for compact frost-image features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_NAMES = ("logistic", "random_forest", "rbf_svm", "hist_gradient_boosting", "mlp")
REPRESENTATIONS = ("handcrafted", "dinov2", "efficientnet")
REPRESENTATION_PREFIXES = {
    "handcrafted": "feature_",
    "dinov2": "dinov2_",
    "efficientnet": "efficientnet_",
}
CAMERA_GROUPS = {
    "top": ("top",),
    "top_close": ("top_close",),
    "left": ("left",),
    "left_close": ("left_close",),
    "front": ("front",),
    "extreme": ("extreme",),
    "top_pair": ("top", "top_close"),
    "left_pair": ("left", "left_close"),
    "all": ("top", "top_close", "left", "left_close", "front", "extreme"),
}


def representation_columns(frame: pd.DataFrame, representation: str) -> list[str]:
    """Return feature columns belonging to one image representation."""
    prefix = REPRESENTATION_PREFIXES[representation]
    return [column for column in frame if column.startswith(prefix)]


def make_rgb_model(name: str):  # type: ignore[no-untyped-def]
    """Return one locked compact classifier for the shared 40D feature protocol."""
    if name == "logistic":
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
    elif name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            n_jobs=-1,
            random_state=0,
        )
    elif name == "rbf_svm":
        classifier = SVC(C=2.0, class_weight="balanced", random_state=0)
    elif name == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            max_iter=100, class_weight="balanced", random_state=0
        )
    elif name == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(32,),
            alpha=0.001,
            early_stopping=True,
            max_iter=300,
            n_iter_no_change=15,
            random_state=0,
        )
    else:
        raise ValueError(f"unknown RGB model: {name}")
    return make_pipeline(StandardScaler(), classifier)


def fit_predict_rgb_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    representation: str = "handcrafted",
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one compact model and return class predictions plus ranking scores."""
    feature_columns = representation_columns(train, representation)
    model = make_rgb_model(model_name)
    model.fit(train[feature_columns], train["target"])
    predicted = model.predict(test[feature_columns])
    score = (
        model.decision_function(test[feature_columns])
        if hasattr(model, "decision_function")
        else model.predict_proba(test[feature_columns])[:, 1]
    )
    return np.asarray(predicted), np.asarray(score)


def high_confidence_coverage(
    label_balance: pd.DataFrame, camera_group: str, threshold: float
) -> float:
    """Return retained pre/post images as a fraction of all candidate-domain images."""
    rows = label_balance.loc[
        label_balance["camera_group"].eq(camera_group)
        & label_balance["regret_threshold"].eq(threshold)
        & label_balance["cost_state"].isin(("pre_optimal", "near_optimal", "post_optimal"))
    ]
    retained = rows.loc[rows["cost_state"].ne("near_optimal"), "image_count"].sum()
    return float(retained / rows["image_count"].sum())


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


def leave_one_experiment_out_predictions(
    frame: pd.DataFrame,
    model_name: str = "rbf_svm",
    representation: str = "handcrafted",
) -> pd.DataFrame:
    """Fit one locked model on all but one experiment at a time."""
    predictions = []
    for experiment in sorted(frame["experiment_id"].unique()):
        test = frame.loc[frame["experiment_id"].eq(experiment)].copy()
        train = frame.loc[~frame["experiment_id"].eq(experiment)]
        if train["target"].nunique() < 2:
            continue
        test["predicted_target"], test["decision_score"] = fit_predict_rgb_model(
            train, test, model_name, representation
        )
        test["held_out_experiment"] = experiment
        test["model"] = model_name
        test["representation"] = representation
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True)
