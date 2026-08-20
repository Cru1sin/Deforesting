#!/usr/bin/env python3
"""Estimate experiment-level RGB learning curves without frame leakage."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from frost_analysis.rgb_evaluation import (
    CAMERA_GROUPS,
    MODEL_NAMES,
    REPRESENTATIONS,
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
        "balanced_accuracy": balanced_accuracy_score(values["target"], values["predicted_target"]),
        "macro_f1": f1_score(values["target"], values["predicted_target"], average="macro"),
        "auroc": roc_auc_score(values["target"], values["decision_score"]),
        "balanced_misclassification_regret": incorrect_regret.groupby(values["target"])
        .mean()
        .mean(),
    }


def evaluate(
    features: pd.DataFrame,
    *,
    camera_groups: list[str],
    models: list[str],
    representations: list[str],
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
                    for representation in representations:
                        for model_name in models:
                            started = time.perf_counter()
                            predicted, decision = fit_predict_rgb_model(
                                train, test, model_name, representation
                            )
                            elapsed = time.perf_counter() - started
                            scored = test.copy()
                            scored["predicted_target"] = predicted
                            scored["decision_score"] = decision
                            rows.append(
                                {
                                    "camera_group": camera_group,
                                    "representation": representation,
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
    results = results.assign(representation=results.get("representation", "handcrafted"))
    metrics = [
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "balanced_misclassification_regret",
        "fit_predict_seconds",
        "training_cycle_count",
        "training_image_count",
    ]
    held_out = results.groupby(
        [
            "camera_group",
            "representation",
            "model",
            "training_experiment_count",
            "held_out_experiment",
        ],
        as_index=False,
    )[metrics].mean()
    rows = []
    for keys, values in held_out.groupby(
        ["camera_group", "representation", "model", "training_experiment_count"],
        sort=True,
    ):
        camera_group, representation, model_name, training_size = keys
        for metric in metrics:
            rows.append(
                {
                    "camera_group": camera_group,
                    "representation": representation,
                    "model": model_name,
                    "training_experiment_count": training_size,
                    "metric": metric,
                    **bootstrap_mean_interval(values[metric]),
                    "held_out_experiment_count": len(values),
                }
            )
    return pd.DataFrame(rows)


def data_requirements(summary: pd.DataFrame, margin: float = 0.02) -> pd.DataFrame:
    """Return the smallest observed experiment count within a fixed full-data margin."""
    summary = summary.assign(representation=summary.get("representation", "handcrafted"))
    accuracy = summary.loc[summary["metric"].eq("balanced_accuracy")].copy()
    rows = []
    for (camera_group, representation, model_name), values in accuracy.groupby(
        ["camera_group", "representation", "model"], sort=True
    ):
        values = values.sort_values("training_experiment_count")
        full = values.iloc[-1]
        eligible = values.loc[values["estimate"].ge(full["estimate"] - margin)]
        required = eligible.iloc[0]
        rows.append(
            {
                "camera_group": camera_group,
                "representation": representation,
                "model": model_name,
                "accuracy_margin": margin,
                "full_training_experiment_count": int(full["training_experiment_count"]),
                "full_balanced_accuracy": full["estimate"],
                "required_training_experiment_count": int(required["training_experiment_count"]),
                "required_balanced_accuracy": required["estimate"],
            }
        )
    return pd.DataFrame(rows)


def plot_learning_curves(summary: pd.DataFrame, output: Path) -> None:
    """Plot classification and control-value learning curves."""
    models = list(MODEL_NAMES)
    colors = dict(zip(models, ["#4C78A8", "#72B7B2", "#F58518", "#B279A2", "#E45756"], strict=True))
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    cameras = [camera for camera in ("front", "all") if camera in summary["camera_group"].unique()]
    fig, axes = plt.subplots(
        len(cameras),
        2,
        figsize=(7.2, 2.45 * len(cameras)),
        squeeze=False,
        sharex=True,
        gridspec_kw={"hspace": 0.30, "wspace": 0.28},
    )
    metrics = (
        ("balanced_accuracy", "Balanced accuracy"),
        ("balanced_misclassification_regret", "Balanced misclassification regret"),
    )
    for row, camera in enumerate(cameras):
        for column, (metric, ylabel) in enumerate(metrics):
            axis = axes[row, column]
            values = summary.loc[summary["camera_group"].eq(camera) & summary["metric"].eq(metric)]
            for model in models:
                points = values.loc[values["model"].eq(model)].sort_values(
                    "training_experiment_count"
                )
                if points.empty:
                    continue
                x = points["training_experiment_count"]
                axis.plot(
                    x,
                    points["estimate"],
                    "o-",
                    color=colors[model],
                    ms=3,
                    lw=1,
                    label=model.replace("_", " "),
                )
                axis.fill_between(
                    x,
                    points["lower"],
                    points["upper"],
                    color=colors[model],
                    alpha=0.12,
                )
            axis.set(ylabel=ylabel, title=f"{camera} · {ylabel}")
            axis.set_xlabel("Training experiments")
            axis.grid(axis="y", color="#DDDDDD", lw=0.45)
            axis.text(
                -0.14,
                1.04,
                chr(ord("a") + row * 2 + column),
                transform=axis.transAxes,
                fontsize=9,
                fontweight="bold",
            )
    axes[0, 0].legend(fontsize=5.5, ncol=2, loc="lower right")
    fig.suptitle(
        "Model performance and control loss versus independent training experiments",
        fontsize=8,
    )
    source = output / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    summary.to_csv(source / "figure_7_learning_curves.csv", index=False)
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 300},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(
            output / f"figure_7_learning_curves{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


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
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="handcrafted")
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
        representations=[args.representation],
        training_sizes=args.training_sizes,
        repeats=args.repeats,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output / "fold_results.csv", index=False)
    summary = summarize(results)
    summary.to_csv(args.output / "summary.csv", index=False)
    data_requirements(summary).to_csv(args.output / "data_requirements.csv", index=False)
    plot_learning_curves(summary, args.output)


if __name__ == "__main__":
    main()
