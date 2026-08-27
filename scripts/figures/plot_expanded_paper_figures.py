#!/usr/bin/env python3
"""Plot the model comparison and visual-concentration evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER = ["logistic", "random_forest", "rbf_svm", "hist_gradient_boosting", "mlp"]
CAMERA_ORDER = [
    "top",
    "top_close",
    "left",
    "left_close",
    "front",
    "extreme",
    "top_pair",
    "left_pair",
    "all",
]
COLORS = dict(
    zip(MODEL_ORDER, ["#4C78A8", "#72B7B2", "#F58518", "#B279A2", "#E45756"], strict=True)
)

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


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_model_comparison(summary: pd.DataFrame, output: Path) -> None:
    values = summary.loc[summary["metric"].eq("balanced_accuracy")].copy()
    values["camera_group"] = pd.Categorical(values["camera_group"], CAMERA_ORDER, ordered=True)
    values["model"] = pd.Categorical(values["model"], MODEL_ORDER, ordered=True)
    values = values.sort_values(["camera_group", "model"])
    source = output.parent / "源数据"
    source.mkdir(parents=True, exist_ok=True)
    values.to_csv(source / "figure_5_model_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"wspace": 0.32})
    offsets = np.linspace(-0.26, 0.26, len(MODEL_ORDER))
    for offset, model in zip(offsets, MODEL_ORDER, strict=True):
        rows = values.loc[values["model"].eq(model)].set_index("camera_group").reindex(CAMERA_ORDER)
        x = np.arange(len(CAMERA_ORDER)) + offset
        axes[0].errorbar(
            x,
            rows["estimate"],
            yerr=[rows["estimate"] - rows["lower"], rows["upper"] - rows["estimate"]],
            fmt="o",
            ms=2.8,
            capsize=1.5,
            lw=0.7,
            color=COLORS[model],
            label=model.replace("_", " "),
        )
    axes[0].set_xticks(range(len(CAMERA_ORDER)), CAMERA_ORDER, rotation=45, ha="right")
    axes[0].set(ylabel="Balanced accuracy", xlabel="Camera group")
    axes[0].set_ylim(0.78, 1.01)
    axes[0].legend(fontsize=5.4, ncol=2, loc="lower left")

    camera_colors = plt.cm.tab10(np.linspace(0, 0.85, len(CAMERA_ORDER)))
    offsets = np.linspace(-0.3, 0.3, len(CAMERA_ORDER))
    for offset, camera, color in zip(offsets, CAMERA_ORDER, camera_colors, strict=True):
        rows = values.loc[values["camera_group"].eq(camera)].set_index("model").reindex(MODEL_ORDER)
        x = np.arange(len(MODEL_ORDER)) + offset
        axes[1].plot(x, rows["estimate"], "o", ms=2.5, color=color, label=camera)
    axes[1].set_xticks(range(len(MODEL_ORDER)), [name.replace("_", "\n") for name in MODEL_ORDER])
    axes[1].set(ylabel="Balanced accuracy", xlabel="Model")
    axes[1].set_ylim(0.78, 1.01)
    axes[1].legend(fontsize=4.8, ncol=3, loc="lower left")
    for label, axis in zip("ab", axes, strict=True):
        axis.text(-0.16, 1.03, label, transform=axis.transAxes, fontsize=9, fontweight="bold")
    fig.suptitle("Same frozen RGB features · leave-one-experiment-out", fontsize=8)
    _save(fig, output / "figure_5_model_camera_comparison")


def plot_concentration(
    optima: pd.DataFrame, concentration: pd.DataFrame, output: Path
) -> None:
    primary = optima.loc[optima["cohort_tier"].eq("A_observed_policy")].copy()
    concentration = concentration.copy()
    concentration["camera_group"] = pd.Categorical(
        concentration["camera_group"], CAMERA_ORDER, ordered=True
    )
    concentration = concentration.sort_values("camera_group")
    source = output.parent / "源数据"
    source.mkdir(parents=True, exist_ok=True)
    primary[["cycle_name", "minutes_from_stable", "minimum_location"]].to_csv(
        source / "figure_6_optimal_times.csv", index=False
    )
    concentration.to_csv(source / "figure_6_visual_concentration.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.34})
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.08, 0.08, len(primary))
    axes[0].scatter(jitter, primary["minutes_from_stable"], s=11, color="#4C78A8", alpha=0.65)
    axes[0].boxplot(
        primary["minutes_from_stable"],
        positions=[0],
        widths=0.3,
        patch_artist=True,
        boxprops={"facecolor": "#C6DBEF", "edgecolor": "#4C78A8"},
        medianprops={"color": "#D95F02"},
    )
    axes[0].set(
        xticks=[0], xticklabels=["Observed-policy\ncycles"], ylabel="Empirical optimum (min)"
    )
    dispersion = concentration["time_optimum_iqr_over_median"].iloc[0]
    axes[0].text(
        0.03,
        0.97,
        f"n={len(primary)}\nIQR/median={dispersion:.3f}",
        transform=axes[0].transAxes,
        va="top",
    )

    y = np.arange(len(concentration))
    axes[1].axvline(0, color="#555555", lw=0.8, ls="--")
    axes[1].errorbar(
        concentration["estimate"],
        y,
        xerr=[
            concentration["estimate"] - concentration["lower"],
            concentration["upper"] - concentration["estimate"],
        ],
        fmt="o",
        color="#D95F02",
        ms=3.5,
        capsize=2,
        lw=0.9,
    )
    axes[1].set_yticks(y, concentration["camera_group"].astype(str))
    axes[1].set(
        xlabel="Optimum-neighbourhood minus fixed-time\ncentroid dispersion",
        ylabel="Camera group",
    )
    axes[1].text(
        0.02,
        0.98,
        "<0 supports visual concentration",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
    )
    for label, axis in zip("ab", axes, strict=True):
        axis.text(-0.17, 1.04, label, transform=axis.transAxes, fontsize=9, fontweight="bold")
    fig.suptitle(
        "Optimal timing varies, but single-frame visual concentration is not supported",
        fontsize=8,
    )
    _save(fig, output / "figure_6_time_visual_concentration")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=Path,
        default=Path("report/03_RGB标签与模型/五模型比较/summary_metrics.csv"),
    )
    parser.add_argument(
        "--optima",
        type=Path,
        default=Path("report/02_经济除霜窗口/经验经济窗口/源数据/cycle_optimal_points.csv"),
    )
    parser.add_argument(
        "--concentration",
        type=Path,
        default=Path("report/03_RGB标签与模型/视觉状态集中性/summary.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("report/04_论文图表/图表"))
    args = parser.parse_args()
    plot_model_comparison(pd.read_csv(args.models), args.output)
    plot_concentration(pd.read_csv(args.optima), pd.read_csv(args.concentration), args.output)


if __name__ == "__main__":
    main()
