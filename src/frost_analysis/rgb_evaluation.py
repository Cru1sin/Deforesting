"""Experiment-held-out evaluation for compact frost-image features."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from contextlib import nullcontext
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import parallel_config
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.parallel import Parallel, delayed
from threadpoolctl import threadpool_limits

from .rgb_deep_features import DEEP_REPRESENTATIONS

MODEL_NAMES = (
    "logistic",
    "window_logistic",
    "random_forest",
    "rbf_svm",
    "hist_gradient_boosting",
    "mlp",
)
REPRESENTATIONS = ("handcrafted", *DEEP_REPRESENTATIONS)
REPRESENTATION_PREFIXES = {
    "handcrafted": "feature_",
    **{name: f"{name}_" for name in DEEP_REPRESENTATIONS},
}
PREDICTION_IDENTITY_COLUMNS = (
    "experiment_id",
    "experiment_date",
    "cycle_id",
    "cycle_uid",
    "cycle_name",
    "camera_role",
    "file_name",
    "frame_index",
    "image_time",
    "cost_state",
    "relative_regret",
    "target",
    "split",
)
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


def map_cost_state_targets(states: pd.Series, task: str) -> pd.Series:
    """Map the shared state labels without silently producing float/NaN targets."""
    names = (
        ("pre_optimal", "post_optimal")
        if task == "binary"
        else ("pre_optimal", "near_optimal", "post_optimal")
    )
    if task not in {"binary", "three"}:
        raise ValueError(f"unknown classification task: {task}")
    return states.map({name: index for index, name in enumerate(names)}).astype("Int64")


def representation_columns(frame: pd.DataFrame, representation: str) -> list[str]:
    """Return feature columns belonging to one image representation."""
    prefix = REPRESENTATION_PREFIXES[representation]
    return [column for column in frame if column.startswith(prefix)]


class _WindowLogisticClassifier(ClassifierMixin, BaseEstimator):
    """Classify within-window first, then before versus after."""

    def fit(self, X, y):  # type: ignore[no-untyped-def]
        y = np.asarray(y)
        if set(y) != {0, 1, 2}:
            raise ValueError("window_logistic requires classes 0,1,2")
        self.within_ = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=0
        ).fit(X, y == 1)
        outside = y != 1
        self.before_after_ = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=0
        ).fit(X[outside], y[outside])
        self.classes_ = np.asarray([0, 1, 2])
        return self

    def predict_proba(self, features):  # type: ignore[no-untyped-def]
        within = self.within_.predict_proba(features)[:, 1]
        before_after = self.before_after_.predict_proba(features)
        outside = 1 - within
        return np.column_stack((outside * before_after[:, 0], within, outside * before_after[:, 1]))

    def predict(self, features):  # type: ignore[no-untyped-def]
        return self.classes_[self.predict_proba(features).argmax(axis=1)]


def make_rgb_model(name: str, *, n_jobs: int = -1):  # type: ignore[no-untyped-def]
    """Return one locked compact classifier for the shared 40D feature protocol."""
    if name == "logistic":
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
    elif name == "window_logistic":
        classifier = _WindowLogisticClassifier()
    elif name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            n_jobs=n_jobs,
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
            early_stopping=False,
            max_iter=1000,
            n_iter_no_change=15,
            random_state=0,
        )
    else:
        raise ValueError(f"unknown RGB model: {name}")
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), classifier)


def fit_predict_rgb_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    representation: str = "handcrafted",
    *,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one compact model and return predictions, scores, and fitted classes."""
    feature_columns = representation_columns(train, representation)
    model = make_rgb_model(model_name, n_jobs=n_jobs)
    model.fit(train[feature_columns], train["target"])
    predicted = model.predict(test[feature_columns])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(test[feature_columns])
        score = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities
    else:
        score = np.asarray(model.decision_function(test[feature_columns]))
    return np.asarray(predicted), np.asarray(score), np.asarray(model.classes_)


def _auroc(values: pd.DataFrame) -> float:
    """Score either binary or multiclass decision columns."""
    score_columns = sorted(
        (column for column in values if column.startswith("decision_score_")),
        key=lambda column: int(column.removeprefix("decision_score_")),
    )
    if score_columns:
        return float(
            np.mean(
                [
                    roc_auc_score(
                        values["target"].eq(int(column.removeprefix("decision_score_"))),
                        values[column],
                    )
                    for column in score_columns
                ]
            )
        )
    return float(roc_auc_score(values["target"], values["decision_score"]))


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


def three_class_eligible_image_coverage(
    labels: pd.DataFrame,
    candidates: pd.DataFrame,
    camera_roles: tuple[str, ...],
) -> float:
    """Return labeled images as a fraction of the real candidate-domain rows."""
    bounds = (
        candidates.assign(
            candidate_time=pd.to_datetime(
                candidates["candidate_time"], errors="coerce", format="mixed"
            )
        )
        .groupby("cycle_name")["candidate_time"]
        .agg(candidate_start="min", candidate_end="max")
    )
    rows = labels.loc[labels["camera_role"].isin(camera_roles)].join(bounds, on="cycle_name")
    image_time = pd.to_datetime(rows["image_time"], errors="coerce", format="mixed")
    inside = image_time.between(rows["candidate_start"], rows["candidate_end"])
    denominator = int(inside.sum())
    if not denominator:
        return float("nan")
    return float(rows.loc[inside, "relative_regret"].notna().sum() / denominator)


def retain_high_confidence_rows(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Exclude images whose pointwise cost regret lies in the ambiguity region."""
    return frame.loc[frame["relative_regret"].gt(threshold)].copy()


def experiment_prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score held-out predictions with one row per independent experiment."""
    rows = []
    for experiment, values in predictions.groupby("experiment_id", sort=True):
        score_columns = sorted(
            (column for column in values if column.startswith("decision_score_")),
            key=lambda column: int(column.removeprefix("decision_score_")),
        )
        expected_classes = (
            tuple(int(column.removeprefix("decision_score_")) for column in score_columns)
            if score_columns
            else (0, 1)
        )
        evaluable = (
            "fold_evaluable" not in values or bool(values["fold_evaluable"].all())
        ) and set(values["target"].unique()) == set(expected_classes)
        if evaluable:
            incorrect_regret = values["relative_regret"].where(
                values["target"].ne(values["predicted_target"]), 0.0
            )
            recalls = recall_score(
                values["target"],
                values["predicted_target"],
                labels=list(expected_classes),
                average=None,
                zero_division=0,
            )
            scores = {
                "recall_before": recalls[0],
                "recall_within": recalls[1] if len(recalls) == 3 else float("nan"),
                "recall_after": recalls[-1],
                "balanced_accuracy": balanced_accuracy_score(
                    values["target"], values["predicted_target"]
                ),
                "macro_f1": f1_score(
                    values["target"],
                    values["predicted_target"],
                    labels=list(expected_classes),
                    average="macro",
                ),
                "accuracy": accuracy_score(values["target"], values["predicted_target"]),
                "positive_f1": f1_score(
                    values["target"], values["predicted_target"], pos_label=1
                ),
                "precision": precision_score(
                    values["target"], values["predicted_target"], pos_label=1, zero_division=0
                ),
                "recall": recall_score(
                    values["target"], values["predicted_target"], pos_label=1, zero_division=0
                ),
                "auroc": _auroc(values),
                "balanced_misclassification_regret": incorrect_regret.groupby(values["target"])
                .mean()
                .mean(),
            }
        else:
            scores = dict.fromkeys(
                (
                    "recall_before",
                    "recall_within",
                    "recall_after",
                    "balanced_accuracy",
                    "macro_f1",
                    "accuracy",
                    "positive_f1",
                    "precision",
                    "recall",
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
    return pd.DataFrame(
        rows,
        columns=(
            "experiment_id",
            "evaluable",
            "recall_before",
            "recall_within",
            "recall_after",
            "balanced_accuracy",
            "macro_f1",
            "accuracy",
            "positive_f1",
            "precision",
            "recall",
            "auroc",
            "balanced_misclassification_regret",
            "image_count",
            "cycle_count",
        ),
    )


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
    """Add elapsed time since stable heating began in each cycle."""
    stable_heating_starts = (
        (
            pd.to_datetime(candidates["candidate_time"])
            - pd.to_timedelta(candidates["heating_hours"], unit="h")
        )
        .groupby(candidates["cycle_name"])
        .first()
    )
    result = frame.copy()
    elapsed = pd.to_datetime(result["image_time"]) - result["cycle_name"].map(stable_heating_starts)
    result["time_elapsed_minutes"] = elapsed.dt.total_seconds() / 60
    return result


def evaluate_holdout_task(  # noqa: C901
    frame: pd.DataFrame,
    experiment: object,
    *,
    model_name: str,
    representation: str = "handcrafted",
    expected_classes: tuple[int, ...] | list[int],
    n_jobs: int = -1,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate one held-out experiment without side effects."""
    started = perf_counter()
    expected_classes = tuple(expected_classes)
    expected = set(expected_classes)
    test = frame.loc[frame["experiment_id"].eq(experiment)].copy()
    train = frame.loc[~frame["experiment_id"].eq(experiment)]
    missing = []
    if set(train["target"].unique()) != expected:
        missing.append("training classes")
    if set(test["target"].unique()) != expected:
        missing.append("test classes")
    status = "invalid" if missing else "ok"
    message = "missing " + " and ".join(missing) if missing else ""
    error_type = ""
    caught: list[warnings.WarningMessage] = []
    try:
        assert train.index.intersection(test.index).empty, "train/test rows overlap"
        test["fold_evaluable"] = status == "ok"
        if status == "ok":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                predicted, score, classes = fit_predict_rgb_model(
                    train, test, model_name, representation, n_jobs=n_jobs
                )
            test["predicted_target"] = predicted
            if score.ndim == 1:
                test["decision_score"] = score
            else:
                for class_name, class_score in zip(classes, score.T, strict=True):
                    test[f"decision_score_{class_name}"] = class_score
            convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
            if convergence:
                status = "failed"
                error_type = "ConvergenceWarning"
                message = str(convergence[-1].message)
        else:
            test["predicted_target"] = pd.NA
            if len(expected_classes) == 2:
                test["decision_score"] = float("nan")
            else:
                for class_name in expected_classes:
                    test[f"decision_score_{class_name}"] = float("nan")
    except Exception as error:
        status = "failed"
        error_type = type(error).__name__
        message = str(error)
        test["fold_evaluable"] = False
        test["predicted_target"] = pd.NA
        if len(expected_classes) == 2:
            test["decision_score"] = float("nan")
        else:
            for class_name in expected_classes:
                test[f"decision_score_{class_name}"] = float("nan")
    test["held_out_experiment"] = experiment
    test["model"] = model_name
    test["representation"] = representation
    test["predicted_target"] = pd.array(test["predicted_target"], dtype="Int64")
    for column in test:
        if column.startswith("decision_score"):
            test[column] = test[column].astype(float)
    generated = [
        column
        for column in test
        if column.startswith("decision_score")
        or column
        in {
            "fold_evaluable",
            "predicted_target",
            "held_out_experiment",
            "model",
            "representation",
        }
    ]
    test = test[[column for column in PREDICTION_IDENTITY_COLUMNS if column in test] + generated]
    return {
        "status": status,
        "elapsed": perf_counter() - started,
        "warning_count": len(caught),
        "error_type": error_type,
        "message": message,
    }, test


def leave_one_experiment_out_predictions(  # noqa: C901
    frame: pd.DataFrame,
    model_name: str = "rbf_svm",
    representation: str = "handcrafted",
    *,
    expected_classes: tuple[int, ...] | list[int],
    jobs: int = 1,
    backend: str = "threading",
    progress: Callable[[int, int, object, float], None] | None = None,
) -> pd.DataFrame:
    """Fit one locked model on all but one experiment at a time."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    if backend not in {"threading", "loky"}:
        raise ValueError("backend must be 'threading' or 'loky'")
    predictions = []
    expected_classes = tuple(expected_classes)
    report_progress = progress or (lambda *_: None)
    experiments = sorted(frame["experiment_id"].unique())

    def evaluate_fold(fold_index, experiment):  # type: ignore[no-untyped-def]
        result, test = evaluate_holdout_task(
            frame,
            experiment,
            model_name=model_name,
            representation=representation,
            expected_classes=expected_classes,
            n_jobs=1 if jobs > 1 else -1,
        )
        return fold_index, experiment, result["elapsed"], test

    limits = threadpool_limits(limits=1) if backend == "threading" else nullcontext()
    configuration = (
        parallel_config(backend="loky", inner_max_num_threads=1)
        if backend == "loky"
        else nullcontext()
    )
    with limits, configuration:
        completed = Parallel(
            n_jobs=jobs,
            backend=backend,
            return_as="generator_unordered",
            batch_size="auto",
            pre_dispatch="2*n_jobs",
        )(
            delayed(evaluate_fold)(fold_index, experiment)
            for fold_index, experiment in enumerate(experiments, start=1)
        )
        for completed_index, (fold_index, experiment, elapsed, test) in enumerate(
            completed, start=1
        ):
            predictions.append((fold_index, test))
            report_progress(completed_index, len(experiments), experiment, elapsed)
    if not predictions:
        empty = frame.iloc[:0].copy()
        empty["predicted_target"] = pd.Series(dtype=frame["target"].dtype)
        empty["fold_evaluable"] = pd.Series(dtype=bool)
        if len(expected_classes) != 2:
            for class_name in expected_classes:
                empty[f"decision_score_{class_name}"] = pd.Series(dtype=float)
        else:
            empty["decision_score"] = pd.Series(dtype=float)
        empty["held_out_experiment"] = pd.Series(dtype=frame["experiment_id"].dtype)
        empty["model"] = pd.Series(dtype="string")
        empty["representation"] = pd.Series(dtype="string")
        return empty
    result = pd.concat(
        [test for _, test in sorted(predictions, key=lambda item: item[0])],
        ignore_index=True,
    )
    result["predicted_target"] = result["predicted_target"].astype("Int64")
    return result
