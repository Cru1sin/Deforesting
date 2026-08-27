#!/usr/bin/env python3
"""Create publication figures for the complete-cohort three-class RGB models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

REPRESENTATIONS = [
    "handcrafted",
    "dinov2",
    "efficientnet",
    "mobilenet_v3_small",
    "repvit_m0_9",
    "convnext_tiny",
    "dinov3",
    "siglip2",
]
MODELS = [
    "logistic",
    "random_forest",
    "rbf_svm",
    "hist_gradient_boosting",
    "mlp",
    "window_logistic",
]
CAMERAS = ["all", "top", "top_close", "left", "left_close", "front", "extreme"]
REP_LABELS = {
    "handcrafted": "Handcrafted",
    "dinov2": "DINOv2",
    "efficientnet": "EfficientNet",
    "mobilenet_v3_small": "MobileNetV3-small",
    "repvit_m0_9": "RepViT-M0.9",
    "convnext_tiny": "ConvNeXt-Tiny",
    "dinov3": "DINOv3",
    "siglip2": "SigLIP2",
}
MODEL_LABELS = {
    "logistic": "Logistic†",
    "random_forest": "Random forest",
    "rbf_svm": "RBF SVM",
    "hist_gradient_boosting": "Hist. gradient boosting",
    "mlp": "MLP†",
    "window_logistic": "Window logistic",
}
CAMERA_LABELS = {
    "all": "All views",
    "top": "Top",
    "top_close": "Top close",
    "left": "Left",
    "left_close": "Left close",
    "front": "Front",
    "extreme": "Extreme",
}
WARNING_MODELS = {"logistic", "mlp"}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro-F1",
}
CLASS_LABELS = {0: "Before", 1: "Within", 2: "After"}
BEST = {
    "representation": "mobilenet_v3_small",
    "model": "random_forest",
    "camera_group": "all",
    "modality": "rgb",
    "regret_threshold": 0.01,
}
BLUE, PALE, PINK, DARK = "#7884B4", "#E4E4F0", "#E4CCD8", "#484878"
FUSION_LABELS = {
    "rgb": "RGB",
    "rgb_state": "RGB + state sensors",
    "rgb_all_sensor": "RGB + all sensors",
}
FUSION_COLORS = {"rgb": DARK, "rgb_state": BLUE, "rgb_all_sensor": PINK}
FUSION_METRICS = {"macro_f1": "Macro-F1", "positive_f1": "1% near-optimal F1"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def _export(fig: plt.Figure, stem: Path, *, tight: bool = True) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", **options)
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", **options)
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white", **options)


def _bootstrap(values: pd.Series, seed: int) -> tuple[float, float, float]:
    x = values.dropna().to_numpy(float)
    if not len(x):
        return np.nan, np.nan, np.nan
    means = np.random.default_rng(seed).choice(x, (10_000, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), *np.quantile(means, [0.025, 0.975])


def _scores(target: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    recalls, f1s = [], []
    for label in (0, 1, 2):
        tp = np.sum((target == label) & (predicted == label))
        fn = np.sum((target == label) & (predicted != label))
        fp = np.sum((target != label) & (predicted == label))
        recalls.append(tp / (tp + fn))
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return {
        "accuracy": float(np.mean(target == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        **{f"recall_{label}": float(recalls[label]) for label in (0, 1, 2)},
    }


def _matrix_sources(summary: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    keys = ["representation", "model", "camera_group", "modality", "regret_threshold"]
    configs = manifest.loc[manifest["stage"].eq("MATRIX"), keys]
    rows = summary.loc[summary["metric"].eq("macro_f1")].merge(configs, on=keys, how="inner")
    out = []
    for row in rows.itertuples(index=False):
        out.append(
            {
                "representation": row.representation,
                "model": row.model,
                "representation_label": REP_LABELS[row.representation],
                "model_label": MODEL_LABELS[row.model],
                "macro_f1": row.estimate,
                "lower_95": row.lower,
                "upper_95": row.upper,
                "evaluable_experiment_count": row.evaluable_experiment_count,
                "convergence_warning_noted": row.model in WARNING_MODELS,
                "warning_note": "A small number of ConvergenceWarning events occurred."
                if row.model in WARNING_MODELS
                else "",
            }
        )
    return pd.DataFrame(out)


def _best_prediction_sources(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = np.ones(len(predictions), dtype=bool)
    for column, value in BEST.items():
        mask &= predictions[column].eq(value)
    data = predictions.loc[mask]
    per_experiment = []
    for camera in CAMERAS:
        subset = data if camera == "all" else data.loc[data["camera_role"].eq(camera)]
        for experiment, group in subset.groupby("experiment_id"):
            if (
                not group["fold_evaluable"].all()
                or group["predicted_target"].isna().any()
                or set(group["target"]) != {0, 1, 2}
            ):
                continue
            score = _scores(group["target"].to_numpy(int), group["predicted_target"].to_numpy(int))
            per_experiment.append({"camera": camera, "experiment_id": experiment, **score})
    per_experiment = pd.DataFrame(per_experiment)
    metric_rows, recall_rows = [], []
    for camera_index, camera in enumerate(CAMERAS):
        group = per_experiment.loc[per_experiment["camera"].eq(camera)]
        for metric_index, metric in enumerate(METRIC_LABELS):
            estimate, lower, upper = _bootstrap(
                group[metric], 2100 + camera_index * 10 + metric_index
            )
            metric_rows.append(
                {
                    "camera": camera,
                    "camera_label": CAMERA_LABELS[camera],
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "estimate": estimate,
                    "lower_95": lower,
                    "upper_95": upper,
                    "evaluable_experiment_count": len(group),
                }
            )
        for label in (0, 1, 2):
            estimate, lower, upper = _bootstrap(
                group[f"recall_{label}"], 3100 + camera_index * 10 + label
            )
            recall_rows.append(
                {
                    "camera": camera,
                    "camera_label": CAMERA_LABELS[camera],
                    "class_id": label,
                    "class_label": CLASS_LABELS[label],
                    "recall": estimate,
                    "lower_95": lower,
                    "upper_95": upper,
                    "evaluable_experiment_count": len(group),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(recall_rows)


def _plot_matrix(source: pd.DataFrame, output: Path) -> None:
    matrix = source.pivot(index="representation", columns="model", values="macro_f1").reindex(
        index=REPRESENTATIONS, columns=MODELS
    )
    cmap = LinearSegmentedColormap.from_list("nmi", [PALE, "#B4C0E4", DARK])
    fig, ax = plt.subplots(figsize=(7.2, 4.45), constrained_layout=True)
    image = ax.imshow(matrix, cmap=cmap, vmin=0.55, vmax=0.90, aspect="auto")
    ax.set_xticks(
        np.arange(len(MODELS)), [MODEL_LABELS[x] for x in MODELS], rotation=30, ha="right"
    )
    ax.set_yticks(np.arange(len(REPRESENTATIONS)), [REP_LABELS[x] for x in REPRESENTATIONS])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            count = source.loc[
                (source.representation.eq(REPRESENTATIONS[row]))
                & (source.model.eq(MODELS[column])),
                "evaluable_experiment_count",
            ].iloc[0]
            ax.text(
                column,
                row,
                f"{value:.3f}\nn={int(count)}",
                ha="center",
                va="center",
                fontsize=5.8,
                color="white" if value > 0.78 else "#303030",
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Experiment-macro F1")
    ax.set(xlabel="Classification head", ylabel="Feature representation")
    ax.text(
        0,
        -0.26,
        "† A small number of ConvergenceWarning events occurred; "
        "these models are not used for the primary conclusion.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        color="#606060",
    )
    _export(fig, output / "rgb_model_matrix_macro_f1")
    plt.close(fig)


def _plot_top10(source: pd.DataFrame, output: Path) -> pd.DataFrame:
    top = source.nlargest(10, "macro_f1").sort_values("macro_f1").copy()
    top["combination_label"] = top["representation_label"] + " + " + top["model_label"]
    y = np.arange(len(top))
    colors = [PINK if row.model == "random_forest" else BLUE for row in top.itertuples()]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.hlines(y, top["lower_95"], top["upper_95"], color=colors, lw=1.4)
    ax.scatter(top["macro_f1"], y, color=colors, edgecolor="#484848", linewidth=0.4, s=24, zorder=3)
    ax.set_yticks(y, top["combination_label"])
    ax.set(xlabel="Experiment-macro F1 (mean and 95% bootstrap CI)", xlim=(0.68, 0.91))
    ax.grid(axis="x", color="#E5E5E5", lw=0.6)
    _export(fig, output / "rgb_top10_macro_f1")
    plt.close(fig)
    return top


def _plot_view_metrics(source: pd.DataFrame, output: Path) -> None:
    y = np.arange(len(CAMERAS))
    offsets = {"accuracy": -0.20, "balanced_accuracy": 0, "macro_f1": 0.20}
    colors = {"accuracy": "#B4C0E4", "balanced_accuracy": BLUE, "macro_f1": DARK}
    fig, ax = plt.subplots(figsize=(7.2, 4.45), constrained_layout=True)
    for metric in METRIC_LABELS:
        rows = source.loc[source["metric"].eq(metric)].set_index("camera").reindex(CAMERAS)
        ax.errorbar(
            rows["estimate"],
            y + offsets[metric],
            xerr=[rows["estimate"] - rows["lower_95"], rows["upper_95"] - rows["estimate"]],
            fmt="o",
            ms=4,
            capsize=2,
            elinewidth=1,
            color=colors[metric],
            label=METRIC_LABELS[metric],
        )
    counts = (
        source.drop_duplicates("camera")
        .set_index("camera")
        .reindex(CAMERAS)["evaluable_experiment_count"]
    )
    ax.set_yticks(
        y, [f"{CAMERA_LABELS[camera]}  (n={int(counts.loc[camera])})" for camera in CAMERAS]
    )
    ax.invert_yaxis()
    ax.set(xlabel="Experiment-level performance (mean and 95% bootstrap CI)", xlim=(0.68, 0.96))
    ax.grid(axis="x", color="#E5E5E5", lw=0.6)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    _export(fig, output / "rgb_best_model_view_metrics")
    plt.close(fig)


def _plot_recall(source: pd.DataFrame, output: Path) -> None:
    matrix = source.pivot(index="camera", columns="class_id", values="recall").reindex(
        index=CAMERAS, columns=[0, 1, 2]
    )
    cmap = LinearSegmentedColormap.from_list("recall", ["#F0E0D0", "#E0E0F0", DARK])
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    image = ax.imshow(matrix, cmap=cmap, vmin=0.45, vmax=1.0, aspect="auto")
    counts = (
        source.drop_duplicates("camera")
        .set_index("camera")
        .reindex(CAMERAS)["evaluable_experiment_count"]
    )
    ax.set_xticks(np.arange(3), [CLASS_LABELS[x] for x in (0, 1, 2)])
    ax.set_yticks(
        np.arange(len(CAMERAS)), [f"{CAMERA_LABELS[c]}  (n={int(counts.loc[c])})" for c in CAMERAS]
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if value > 0.80 else "#303030",
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Experiment-macro recall")
    ax.set(xlabel="Frost-cycle class", ylabel="Camera view")
    _export(fig, output / "rgb_best_model_class_recall")
    plt.close(fig)


def _plot_sensor_fusion_f1(source: pd.DataFrame, output: Path) -> None:
    order = [(metric, modality) for metric in FUSION_METRICS for modality in FUSION_LABELS]
    rows = source.set_index(["metric", "modality"]).loc[order].reset_index()
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(89 / 25.4, 78 / 25.4))
    fig.subplots_adjust(left=0.34, right=0.97, bottom=0.23, top=0.88)
    for yi, row in zip(y, rows.itertuples(), strict=True):
        color = FUSION_COLORS[row.modality]
        ax.errorbar(
            row.estimate,
            yi,
            xerr=[[row.estimate - row.lower], [row.upper - row.estimate]],
            fmt="o",
            ms=4.2,
            capsize=2,
            elinewidth=1.1,
            color=color,
            markeredgecolor=DARK,
            markeredgewidth=0.45,
        )
    ax.set_yticks(
        y,
        [FUSION_LABELS[row.modality].removesuffix(" sensors") for row in rows.itertuples()],
    )
    ax.axhline(2.5, color="#D8D8D8", lw=0.7)
    ax.set(xlabel="F1 score", xlim=(0.52, 0.85))
    ax.grid(axis="x", color="#E5E5E5", lw=0.6)
    ax.text(
        0,
        1.02,
        "Mean and 95% bootstrap CI; n = 14 experiments",
        transform=ax.transAxes,
        fontsize=6,
        color="#606060",
    )
    fig.text(
        0.34,
        0.03,
        "All sensors includes COP, power and capacity residual;\nassociation is not vision-only.",
        fontsize=5.5,
        color="#606060",
    )
    fig.text(0.02, 0.82, "Macro-F1", fontsize=6, weight="bold")
    fig.text(0.02, 0.51, "1% near-optimal F1", fontsize=6, weight="bold")
    _export(fig, output / "rgb_binary_sensor_fusion_f1", tight=False)
    plt.close(fig)


def _plot_sensor_fusion_delta(source: pd.DataFrame, output: Path) -> None:
    order = [
        (metric, comparison)
        for metric in FUSION_METRICS
        for comparison in ("rgb_state_minus_rgb", "rgb_all_sensor_minus_rgb")
    ]
    rows = source.set_index(["metric", "comparison"]).loc[order].reset_index()
    rows[["estimate", "lower", "upper"]] *= 100
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(89 / 25.4, 62 / 25.4))
    fig.subplots_adjust(left=0.36, right=0.97, bottom=0.27, top=0.86)
    for yi, row in zip(y, rows.itertuples(), strict=True):
        modality = row.comparison.removesuffix("_minus_rgb")
        color = FUSION_COLORS[modality]
        ax.errorbar(
            row.estimate,
            yi,
            xerr=[[row.estimate - row.lower], [row.upper - row.estimate]],
            fmt="o",
            ms=4.2,
            capsize=2,
            elinewidth=1.1,
            color=color,
            markeredgecolor=DARK,
            markeredgewidth=0.45,
        )
    ax.axvline(0, color="#767676", ls="--", lw=0.9)
    ax.set_yticks(
        y,
        [
            "{} · {}".format(
                FUSION_METRICS[row.metric],
                FUSION_LABELS[row.comparison.removesuffix("_minus_rgb")]
                .removeprefix("RGB + ")
                .removesuffix(" sensors"),
            )
            for row in rows.itertuples()
        ],
    )
    ax.set(xlabel="Paired ΔF1 (percentage points)", xlim=(-0.5, 6.0))
    ax.grid(axis="x", color="#E5E5E5", lw=0.6)
    ax.text(
        0,
        1.02,
        "Fusion − RGB; 95% bootstrap CI; n = 14 experiments",
        transform=ax.transAxes,
        fontsize=6,
        color="#606060",
    )
    fig.text(
        0.36,
        0.03,
        "All sensors includes COP, power and capacity residual;\nassociation is not vision-only.",
        fontsize=5.5,
        color="#606060",
    )
    _export(fig, output / "rgb_binary_sensor_fusion_delta", tight=False)
    plt.close(fig)


def _render_sensor_fusion(results: Path, output: Path, summary: pd.DataFrame) -> None:
    source_dir = output.parent / "源数据"
    source_dir.mkdir(parents=True, exist_ok=True)
    source = summary.loc[
        summary["metric"].isin(FUSION_METRICS) & summary["modality"].isin(FUSION_LABELS)
    ].copy()
    assert len(source) == 6 and source["experiment_count"].eq(14).all()
    source["input_label"] = source["modality"].map(FUSION_LABELS)
    source["metric_label"] = source["metric"].map(FUSION_METRICS)
    source["reviewer_note"] = (
        source["modality"]
        .map(
            {
                "rgb_all_sensor": "Includes COP, compressor power and evaporator-capacity "
                "baseline residual; not vision-only."
            }
        )
        .fillna("")
    )
    source.to_csv(source_dir / "rgb_binary_sensor_fusion_f1.csv", index=False)

    delta = pd.read_csv(results / "modality_deltas.csv")
    delta = delta.loc[
        delta["metric"].isin(FUSION_METRICS)
        & delta["comparison"].isin(("rgb_state_minus_rgb", "rgb_all_sensor_minus_rgb"))
    ].copy()
    assert len(delta) == 4 and delta["evaluable_experiment_count"].eq(14).all()
    delta["metric_label"] = delta["metric"].map(FUSION_METRICS)
    delta["estimate_percentage_points"] = 100 * delta["estimate"]
    delta["lower_percentage_points"] = 100 * delta["lower"]
    delta["upper_percentage_points"] = 100 * delta["upper"]
    delta["reviewer_note"] = (
        delta["comparison"]
        .map(
            {
                "rgb_all_sensor_minus_rgb": "Includes COP, compressor power and "
                "evaporator-capacity baseline residual; not vision-only."
            }
        )
        .fillna("")
    )
    delta.to_csv(source_dir / "rgb_binary_sensor_fusion_delta.csv", index=False)
    _plot_sensor_fusion_f1(source, output)
    _plot_sensor_fusion_delta(delta, output)


def render(results: Path, output: Path) -> None:
    _style()
    summary = pd.read_csv(results / "summary_metrics.csv")
    if set(summary["modality"]) == set(FUSION_LABELS):
        _render_sensor_fusion(results, output, summary)
        return
    manifest = pd.read_csv(results / "experiment_manifest.csv")
    predictions = pd.read_parquet(results / "predictions.parquet")
    source_dir = output.parent / "源数据"
    source_dir.mkdir(parents=True, exist_ok=True)

    expected = summary.loc[
        summary["metric"].eq("macro_f1")
        & summary["camera_group"].eq("all")
        & summary["modality"].eq("rgb")
        & summary["regret_threshold"].eq(0.01)
    ]
    assert expected[["representation", "model"]].drop_duplicates().shape[0] == 48
    matrix = _matrix_sources(summary, manifest)
    assert matrix.shape[0] == 48 and matrix["evaluable_experiment_count"].between(1, 14).all()
    matrix.to_csv(source_dir / "rgb_model_matrix_macro_f1.csv", index=False)
    _plot_matrix(matrix, output)
    top10 = _plot_top10(matrix, output)
    top10.to_csv(source_dir / "rgb_top10_macro_f1.csv", index=False)

    view_metrics, recalls = _best_prediction_sources(predictions)
    assert view_metrics.shape == (21, 8) and recalls.shape == (21, 8)
    view_metrics.to_csv(source_dir / "rgb_best_model_view_metrics.csv", index=False)
    recalls.to_csv(source_dir / "rgb_best_model_class_recall.csv", index=False)
    _plot_view_metrics(view_metrics, output)
    _plot_recall(recalls, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("output/model/rgb_full_cohort_latest"))
    parser.add_argument("--output", type=Path, default=Path("output/test/model/论文图表/图表"))
    args = parser.parse_args()
    render(args.results, args.output)


if __name__ == "__main__":
    main()
