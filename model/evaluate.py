"""Recompute fold metrics and summarize trained runs."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

SETTING_COLUMNS = ("representation", "head", "camera", "modality")
FOLD_COLUMNS = (*SETTING_COLUMNS, "held_out_experiment")
METRIC_COLUMNS = ("accuracy", "balanced_accuracy", "macro_f1")
METRICS_REQUIRED = (
    *FOLD_COLUMNS,
    "train_images",
    "test_images",
    "status",
    "message",
)
PREDICTIONS_REQUIRED = (*FOLD_COLUMNS, "experiment_id", "target", "prediction")
SUMMARY_COLUMNS = (
    *SETTING_COLUMNS,
    "total_folds",
    "valid_folds",
    "total_test_images",
    "accuracy_mean",
    "accuracy_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
)


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} are missing required columns: {', '.join(missing)}")


def _key(row: pd.Series[Any]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in FOLD_COLUMNS)


def _expected_labels(task: str) -> list[int]:
    if task == "binary":
        return [0, 1]
    if task == "three":
        return [0, 1, 2]
    raise ValueError(f"unknown task: {task}")


def _prediction_groups(predictions: pd.DataFrame) -> dict[tuple[str, ...], list[int]]:
    groups: dict[tuple[str, ...], list[int]] = {}
    for index in range(len(predictions)):
        groups.setdefault(_key(predictions.iloc[index]), []).append(index)
    return groups


def _fold_metrics(predictions: pd.DataFrame, labels: list[int]) -> dict[str, float]:
    target = pd.to_numeric(predictions["target"], errors="raise").astype(int)
    prediction = pd.to_numeric(predictions["prediction"], errors="raise").astype(int)
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(
            recall_score(
                target,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                target,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for values, folds in metrics.groupby(list(SETTING_COLUMNS), dropna=False, sort=False):
        values = values if isinstance(values, tuple) else (values,)
        valid = folds.loc[
            folds["status"].astype(str).eq("ok")
            & folds[list(METRIC_COLUMNS)].notna().all(axis=1)
        ]
        row = dict(zip(SETTING_COLUMNS, values, strict=True))
        row.update(
            total_folds=len(folds),
            valid_folds=len(valid),
            total_test_images=pd.to_numeric(folds["test_images"], errors="coerce").sum(),
        )
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = valid[metric].mean()
            row[f"{metric}_std"] = valid[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _validate_groups(
    metrics: pd.DataFrame,
    metric_rows: dict[tuple[str, ...], list[int]],
    groups: dict[tuple[str, ...], list[int]],
) -> None:
    for key, indices in metric_rows.items():
        if len(indices) > 1:
            raise ValueError(f"duplicate metrics fold key: {key}")
    for key in groups:
        if len(metric_rows.get(key, [])) != 1:
            raise ValueError(
                f"prediction group {key} does not map to exactly one metrics row"
            )
    for key, indices in metric_rows.items():
        if str(metrics.at[indices[0], "status"]) == "ok" and key not in groups:
            raise ValueError(f"ok metrics row {key} has no prediction group")


def evaluate_run(
    metrics: pd.DataFrame, predictions: pd.DataFrame, task: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute one training run's fold metrics and setting summaries."""
    _require_columns(metrics, METRICS_REQUIRED, "metrics")
    if not predictions.empty:
        _require_columns(predictions, PREDICTIONS_REQUIRED, "predictions")
        if not (
            predictions["experiment_id"].astype(str)
            == predictions["held_out_experiment"].astype(str)
        ).all():
            raise ValueError("prediction experiment_id must equal held_out_experiment")

    labels = _expected_labels(task)
    metrics = metrics.reset_index(drop=True).copy()
    predictions = predictions.reset_index(drop=True)
    metric_rows: dict[tuple[str, ...], list[int]] = {}
    for index in range(len(metrics)):
        metric_rows.setdefault(_key(metrics.iloc[index]), []).append(index)
    groups = _prediction_groups(predictions)
    _validate_groups(metrics, metric_rows, groups)
    for metric in METRIC_COLUMNS:
        metrics[metric] = float("nan")
    for key, indices in groups.items():
        metric_index = metric_rows[key][0]
        if str(metrics.at[metric_index, "status"]) == "ok":
            expected_count = pd.to_numeric(
                metrics.at[metric_index, "test_images"], errors="raise"
            )
            if len(indices) != expected_count:
                raise ValueError(
                    f"prediction rows for {key} do not match test_images "
                    f"({len(indices)} != {expected_count})"
                )
            metrics.loc[metric_index, list(METRIC_COLUMNS)] = _fold_metrics(
                predictions.loc[indices], labels
            )
    return metrics, _summary(metrics)
