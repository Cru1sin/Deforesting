#!/usr/bin/env python3
"""Estimate experiment-level RGB learning curves without frame leakage."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from frost_analysis.rgb_evaluation import (
    CAMERA_GROUPS,
    MODEL_NAMES,
    bootstrap_mean_interval,
    fit_predict_rgb_model,
)


def nested_training_sets(
    experiments: list[str],
    held_out: str,
    sizes: list[int],
    repeat: int,
) -> dict[int, tuple[str, ...]]:
    """Return deterministic nested experiment subsets for one held-out experiment."""
    available = sorted(set(experiments) - {held_out})
    seed = repeat + sum((index + 1) * ord(character) for index, character in enumerate(held_out))
    order = np.asarray(available)[np.random.default_rng(seed).permutation(len(available))]
    return {size: tuple(order[: min(size, len(order))]) for size in sizes}


def score_predictions(values: pd.DataFrame) -> dict[str, float]:
    """Return classification and cost-weighted errors for one held-out experiment."""
    incorrect_regret = values["relative_regret"].where(
        values["target"].ne(values["predicted_target"]), 0.0
    )
    return {
        "balanced_accuracy": balanced_accuracy_score(
            values["target"], values["predicted_target"]
        ),
        "macro_f1": f1_score(values["target"], values["predicted_target"], average="macro"),
        "auroc": roc_auc_score(values["target"], values["decision_score"]),
        "balanced_misclassification_regret": incorrect_regret.groupby(
            values["target"]
        ).mean().mean(),
    }


def evaluate(
    features: pd.DataFrame,
    *,
    camera_groups: list[str],
    models: list[str],
    training_sizes: list[int],
    repeats: int,
) -> pd.DataFrame:
    """Evaluate nested experiment-count learning curves."""
    rows = []
    experiments = sorted(features["experiment_id"].unique())
    for camera_group in camera_groups:
        scoped = features.loc[features["camera_role"].isin(CAMERA_GROUPS[camera_group])]
        for held_out in experiments:
            test = scoped.loc[scoped["experiment_id"].eq(held_out)].copy()
            if test["target"].nunique() < 2:
                continue
            for repeat in range(repeats):
                subsets = nested_training_sets(experiments, held_out, training_sizes, repeat)
                for training_size, selected in subsets.items():
                    if training_size == len(experiments) - 1 and repeat > 0:
                        continue
                    train = scoped.loc[scoped["experiment_id"].isin(selected)]
                    if train["target"].nunique() < 2:
                        continue
                    for model_name in models:
                        started = time.perf_counter()
                        predicted, decision = fit_predict_rgb_model(train, test, model_name)
                        elapsed = time.perf_counter() - started
                        scored = test.copy()
                        scored["predicted_target"] = predicted
                        scored["decision_score"] = decision
                        rows.append(
                            {
                                "camera_group": camera_group,
                                "model": model_name,
                                "held_out_experiment": held_out,
                                "repeat": repeat,
                                "training_experiment_count": len(selected),
                                "training_cycle_count": train["cycle_name"].nunique(),
                                "training_image_count": len(train),
                                "test_image_count": len(test),
                                "fit_predict_seconds": elapsed,
                                **score_predictions(scored),
                            }
                        )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Average repeated subsets within experiment, then bootstrap independent experiments."""
    metrics = [
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "balanced_misclassification_regret",
        "fit_predict_seconds",
        "training_cycle_count",
        "training_image_count",
    ]
    held_out = (
        results.groupby(
            ["camera_group", "model", "training_experiment_count", "held_out_experiment"],
            as_index=False,
        )[metrics]
        .mean()
    )
    rows = []
    for keys, values in held_out.groupby(
        ["camera_group", "model", "training_experiment_count"], sort=True
    ):
        camera_group, model_name, training_size = keys
        for metric in metrics:
            rows.append(
                {
                    "camera_group": camera_group,
                    "model": model_name,
                    "training_experiment_count": training_size,
                    "metric": metric,
                    **bootstrap_mean_interval(values[metric]),
                    "held_out_experiment_count": len(values),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards", type=Path, default=Path("report/rgb_full_feature_shards/cycles")
    )
    parser.add_argument(
        "--camera-groups",
        nargs="+",
        choices=tuple(CAMERA_GROUPS),
        default=["front", "all"],
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--training-sizes", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("report/rgb_learning_curves"))
    args = parser.parse_args()
    paths = sorted(args.shards.glob("frost_cycle_*.parquet"))
    if not paths:
        raise SystemExit("no full feature shards")
    features = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    features = features.loc[
        features["relative_regret"].gt(0.01)
        & features["cost_state"].isin(("pre_optimal", "post_optimal"))
    ].copy()
    features["target"] = features["cost_state"].map({"pre_optimal": 0, "post_optimal": 1})
    results = evaluate(
        features,
        camera_groups=args.camera_groups,
        models=args.models,
        training_sizes=args.training_sizes,
        repeats=args.repeats,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output / "fold_results.csv", index=False)
    summarize(results).to_csv(args.output / "summary.csv", index=False)


if __name__ == "__main__":
    main()
