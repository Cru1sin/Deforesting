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
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from frost_analysis.rgb_evaluation import (
    CAMERA_GROUPS,
    MODEL_NAMES,
    REPRESENTATIONS,
    bootstrap_mean_interval,
    fit_predict_rgb_model,
    map_cost_state_targets,
)

FOLD_METRICS = (
    "recall_before",
    "recall_within",
    "recall_after",
    "balanced_accuracy",
    "macro_f1",
    "auroc",
    "balanced_misclassification_regret",
)
DEFAULT_MODELS = tuple(name for name in MODEL_NAMES if name != "window_logistic")
MODEL_COLORS = dict(
    zip(
        MODEL_NAMES,
        ("#4C78A8", "#54A24B", "#72B7B2", "#F58518", "#B279A2", "#E45756"),
        strict=True,
    )
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


def score_predictions(
    values: pd.DataFrame, expected_classes: tuple[int, ...] | list[int]
) -> dict[str, float]:
    """Return classification and cost-weighted errors for one held-out experiment."""
    expected_classes = tuple(expected_classes)
    incorrect_regret = values["relative_regret"].where(
        values["target"].ne(values["predicted_target"]), 0.0
    )
    score_columns = sorted(
        (column for column in values if column.startswith("decision_score_")),
        key=lambda column: int(column.removeprefix("decision_score_")),
    )
    recalls = recall_score(
        values["target"],
        values["predicted_target"],
        labels=list(expected_classes),
        average=None,
        zero_division=0,
    )
    try:
        auroc = (
            np.mean(
                [
                    roc_auc_score(
                        values["target"].eq(
                            int(column.removeprefix("decision_score_"))
                        ),
                        values[column],
                    )
                    for column in score_columns
                ]
            )
            if score_columns
            else roc_auc_score(values["target"], values["decision_score"])
        )
    except ValueError:
        auroc = float("nan")
    return {
        "recall_before": recalls[0],
        "recall_within": recalls[1] if len(recalls) == 3 else float("nan"),
        "recall_after": recalls[-1],
        "balanced_accuracy": balanced_accuracy_score(values["target"], values["predicted_target"]),
        "macro_f1": f1_score(
            values["target"],
            values["predicted_target"],
            labels=list(expected_classes),
            average="macro",
        ),
        "auroc": auroc,
        "balanced_misclassification_regret": incorrect_regret.groupby(values["target"])
        .mean()
        .mean(),
    }


def evaluate(  # noqa: C901
    features: pd.DataFrame,
    *,
    camera_groups: list[str],
    models: list[str],
    representations: list[str],
    training_sizes: list[int],
    repeats: int,
    expected_classes: tuple[int, ...] | list[int],
) -> pd.DataFrame:
    """Evaluate nested experiment-count learning curves."""
    if features.groupby("cycle_name")["experiment_id"].nunique().gt(1).any():
        raise ValueError("one or more cycles belong to multiple experiments")
    rows = []
    experiments = sorted(features["experiment_id"].unique())
    expected_class_set = set(expected_classes)
    for camera_group in camera_groups:
        scoped = features.loc[features["camera_role"].isin(CAMERA_GROUPS[camera_group])]
        for held_out in experiments:
            test = scoped.loc[scoped["experiment_id"].eq(held_out)].copy()
            for repeat in range(repeats):
                subsets = nested_training_sets(experiments, held_out, training_sizes, repeat)
                for training_size, selected in subsets.items():
                    if training_size == len(experiments) - 1 and repeat > 0:
                        continue
                    train = scoped.loc[scoped["experiment_id"].isin(selected)]
                    fold_evaluable = (
                        set(train["target"].unique()) == expected_class_set
                        and set(test["target"].unique()) == expected_class_set
                    )
                    for representation in representations:
                        for model_name in models:
                            started = time.perf_counter()
                            if fold_evaluable:
                                predicted, decision, classes = fit_predict_rgb_model(
                                    train, test, model_name, representation
                                )
                                scored = test.copy()
                                scored["predicted_target"] = predicted
                                if decision.ndim == 1:
                                    scored["decision_score"] = decision
                                else:
                                    for class_name, class_score in zip(
                                        classes, decision.T, strict=True
                                    ):
                                        scored[f"decision_score_{class_name}"] = class_score
                                scores = score_predictions(scored, expected_classes)
                            else:
                                scores = dict.fromkeys(FOLD_METRICS, float("nan"))
                            elapsed = time.perf_counter() - started
                            rows.append(
                                {
                                    "camera_group": camera_group,
                                    "representation": representation,
                                    "model": model_name,
                                    "held_out_experiment": held_out,
                                    "repeat": repeat,
                                    "training_experiment_count": train[
                                        "experiment_id"
                                    ].nunique(),
                                    "training_cycle_count": train["cycle_name"].nunique(),
                                    "training_image_count": len(train),
                                    "test_image_count": len(test),
                                    "fit_predict_seconds": elapsed,
                                    "evaluable": fold_evaluable,
                                    **scores,
                                }
                            )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Average repeated subsets within experiment, then bootstrap independent experiments."""
    results = results.assign(representation=results.get("representation", "handcrafted"))
    metrics = [
        "recall_before",
        "recall_within",
        "recall_after",
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
                    "held_out_experiment_count": int(values[metric].notna().sum()),
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
        required = eligible.iloc[0] if not eligible.empty else None
        rows.append(
            {
                "camera_group": camera_group,
                "representation": representation,
                "model": model_name,
                "accuracy_margin": margin,
                "full_training_experiment_count": int(full["training_experiment_count"]),
                "full_balanced_accuracy": full["estimate"],
                "required_training_experiment_count": (
                    int(required["training_experiment_count"])
                    if required is not None
                    else float("nan")
                ),
                "required_balanced_accuracy": (
                    required["estimate"] if required is not None else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_learning_curves(summary: pd.DataFrame, output: Path) -> None:
    """Plot classification and control-value learning curves."""
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
            for model in MODEL_NAMES:
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
                    color=MODEL_COLORS[model],
                    ms=3,
                    lw=1,
                    label=model.replace("_", " "),
                )
                axis.fill_between(
                    x,
                    points["lower"],
                    points["upper"],
                    color=MODEL_COLORS[model],
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
    source = output / "源数据"
    source.mkdir(parents=True, exist_ok=True)
    summary.to_csv(source / "figure_7_learning_curves.csv", index=False)
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 300},
    }.items():
        fig.savefig(
            output / f"figure_7_learning_curves{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def plot_camera_grid(summary: pd.DataFrame, output: Path, metric: str, ylabel: str) -> None:
    """Plot one learning-curve metric for all nine camera groups."""
    camera_order = list(CAMERA_GROUPS)
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7.2, 6.4),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.30, "wspace": 0.20},
    )
    for index, (camera, axis) in enumerate(zip(camera_order, axes.flat, strict=True)):
        values = summary.loc[summary["camera_group"].eq(camera) & summary["metric"].eq(metric)]
        for model in MODEL_NAMES:
            points = values.loc[values["model"].eq(model)].sort_values("training_experiment_count")
            if points.empty:
                continue
            axis.plot(
                points["training_experiment_count"],
                points["estimate"],
                "o-",
                color=MODEL_COLORS[model],
                ms=2.5,
                lw=0.9,
                label=model.replace("_", " "),
            )
        axis.set_title(camera.replace("_", " "), fontsize=7, fontweight="bold")
        axis.grid(axis="y", color="#DDDDDD", lw=0.4)
        axis.text(
            -0.15,
            1.03,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
        )
    fig.supxlabel("Training experiments", fontsize=7, y=0.045)
    fig.supylabel(ylabel, fontsize=7, x=0.015)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=5.8)
    fig.subplots_adjust(bottom=0.12, left=0.08)
    base = output / f"{metric}_all_camera_learning_curves"
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 300},
    }.items():
        fig.savefig(
            base.with_suffix(suffix),
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards", type=Path, default=Path("outputs/RGB特征缓存/全量手工特征/cycles")
    )
    parser.add_argument(
        "--camera-groups",
        nargs="+",
        choices=tuple(CAMERA_GROUPS),
        default=["front", "all"],
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="handcrafted")
    parser.add_argument("--task", choices=("binary", "three"), default="binary")
    parser.add_argument("--training-sizes", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("report/03_RGB标签与模型/九机位学习曲线")
    )
    args = parser.parse_args()
    args.models = args.models or list(
        DEFAULT_MODELS if args.task == "binary" else MODEL_NAMES
    )
    if args.task == "binary" and "window_logistic" in args.models:
        raise SystemExit("window_logistic requires --task three")
    paths = sorted(args.shards.glob("*.parquet"))
    if not paths:
        raise SystemExit("no full feature shards")
    features = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True).drop(
        columns="cost_source_sha256", errors="ignore"
    )
    if args.task == "binary":
        features = features.loc[features["relative_regret"].gt(0.01)].copy()
    features["target"] = map_cost_state_targets(features["cost_state"], args.task)
    features = features.loc[features["target"].notna()].copy()
    features["target"] = features["target"].astype(int)
    results = evaluate(
        features,
        camera_groups=args.camera_groups,
        models=args.models,
        representations=[args.representation],
        training_sizes=args.training_sizes,
        repeats=args.repeats,
        expected_classes=(0, 1) if args.task == "binary" else (0, 1, 2),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output / "fold_results.csv", index=False)
    summary = summarize(results)
    summary.to_csv(args.output / "summary.csv", index=False)
    data_requirements(summary).to_csv(args.output / "data_requirements.csv", index=False)
    plot_learning_curves(summary, args.output)
    plot_camera_grid(summary, args.output, "balanced_accuracy", "Balanced accuracy")
    plot_camera_grid(
        summary,
        args.output,
        "balanced_misclassification_regret",
        "Balanced cost regret",
    )


if __name__ == "__main__":
    main()
