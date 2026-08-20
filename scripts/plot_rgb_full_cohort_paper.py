#!/usr/bin/env python3
"""Plot complete-cohort RGB increments and held-out failure modes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frost_analysis.paper_figures import (
    full_cohort_figure_3_sources,
    full_cohort_figure_4_sources,
)

BLUE = "#3775BA"
ORANGE = "#E28E2C"
GREY = "#777777"
RED = "#B64342"
COLORS = {"rgb": BLUE, "time": GREY, "rgb_time": ORANGE}
LABELS = {"rgb": "RGB", "time": "Retrospective time", "rgb_time": "RGB + time"}
CAMERAS = [
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
CAMERA_LABELS = {
    "top": "top",
    "top_close": "top close",
    "left": "left",
    "left_close": "left close",
    "front": "front",
    "extreme": "extreme",
    "top_pair": "top + top close",
    "left_pair": "left + left close",
    "all": "all views",
}


def export(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg = stem.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def render_figures(
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    experiment_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    source = output.parent / "源数据"
    source.mkdir(parents=True, exist_ok=True)
    figure_3 = full_cohort_figure_3_sources(summary, deltas)
    for name, frame in figure_3.items():
        frame.to_csv(source / f"figure_3_{name}.csv", index=False)
    _plot_figure_3(figure_3, output)

    figure_4 = full_cohort_figure_4_sources(experiment_metrics, predictions)
    for name, frame in figure_4.items():
        frame.to_csv(source / f"figure_4_{name}.csv", index=False)
    _plot_figure_4(figure_4, output)


def _plot_figure_3(sources: dict[str, pd.DataFrame], output: Path) -> None:
    performance = sources["camera_performance"]
    delta = sources["camera_deltas"]
    tradeoff = sources["threshold_tradeoff"]
    available = [camera for camera in CAMERAS if camera in set(performance["camera_group"])]
    y = np.arange(len(available))
    offsets = {"rgb": -0.19, "time": 0.0, "rgb_time": 0.19}

    fig = plt.figure(figsize=(7.2, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.7, 1.0))
    ax_a = fig.add_subplot(grid[0, 0])
    for modality in ("rgb", "time", "rgb_time"):
        rows = performance.loc[performance["modality"].eq(modality)].set_index("camera_group")
        rows = rows.reindex(available)
        ax_a.errorbar(
            rows["estimate"],
            y + offsets[modality],
            xerr=[rows["estimate"] - rows["lower"], rows["upper"] - rows["estimate"]],
            fmt="o",
            ms=3.5,
            capsize=2,
            color=COLORS[modality],
            label=LABELS[modality],
        )
    ax_a.set_yticks(y, [CAMERA_LABELS[name] for name in available])
    ax_a.invert_yaxis()
    ax_a.set(xlabel="Experiment-macro balanced accuracy", xlim=(0.5, 1.01))
    ax_a.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=6.5
    )

    ax_b = fig.add_subplot(grid[0, 1])
    comparisons = (
        ("rgb_minus_time", "RGB − time", BLUE, -0.11),
        ("rgb_time_minus_time", "RGB + time − time", ORANGE, 0.11),
    )
    for comparison, label, color, offset in comparisons:
        rows = delta.loc[delta["comparison"].eq(comparison)].set_index("camera_group")
        rows = rows.reindex(available)
        ax_b.errorbar(
            rows["estimate"],
            y + offset,
            xerr=[rows["estimate"] - rows["lower"], rows["upper"] - rows["estimate"]],
            fmt="o",
            ms=3.5,
            capsize=2,
            color=color,
            label=label,
        )
    ax_b.axvline(0, color="#222222", lw=0.8, ls="--")
    ax_b.set_yticks(y, [])
    ax_b.invert_yaxis()
    ax_b.set(xlabel="Paired balanced-accuracy difference")
    ax_b.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=6.5
    )

    ax_c = fig.add_subplot(grid[1, :])
    for modality in ("rgb", "time", "rgb_time"):
        rows = tradeoff.loc[tradeoff["modality"].eq(modality)].sort_values(
            "regret_threshold"
        )
        ax_c.errorbar(
            rows["regret_threshold"] * 100,
            rows["estimate"],
            yerr=[rows["estimate"] - rows["lower"], rows["upper"] - rows["estimate"]],
            marker="o",
            ms=3.5,
            capsize=2,
            color=COLORS[modality],
            label=LABELS[modality],
        )
    coverage = tradeoff.drop_duplicates("regret_threshold").sort_values("regret_threshold")
    ax_cover = ax_c.twinx()
    ax_cover.plot(
        coverage["regret_threshold"] * 100,
        coverage["eligible_image_coverage"] * 100,
        color="#222222",
        marker="s",
        ms=3,
        ls=":",
        label="Eligible image coverage",
    )
    ax_c.set(
        xlabel="Relative-regret exclusion threshold (%)",
        ylabel="Balanced accuracy",
        ylim=(0.5, 1.02),
    )
    ax_cover.set(ylabel="Eligible image coverage (%)", ylim=(0, 75))
    for row in coverage.itertuples(index=False):
        ax_c.text(
            row.regret_threshold * 100,
            0.515,
            f"n={int(row.evaluable_experiment_count)}",
            ha="center",
            va="bottom",
            fontsize=5.8,
        )
    handles, labels = ax_c.get_legend_handles_labels()
    cover_handles, cover_labels = ax_cover.get_legend_handles_labels()
    ax_c.legend(
        handles + cover_handles,
        labels + cover_labels,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=6.3,
    )

    for label, axis in zip("abc", (ax_a, ax_b, ax_c), strict=True):
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontsize=10, fontweight="bold")
    export(fig, output / "figure_3_rgb_increment")
    plt.close(fig)


def _plot_figure_4(sources: dict[str, pd.DataFrame], output: Path) -> None:
    experiments = sources["experiment_metrics"]
    failures = sources["cycle_failures"].head(10).sort_values(
        "mean_misclassification_regret"
    )
    order = sorted(experiments["experiment_id"].unique())
    x = np.arange(len(order))
    short_labels = [value.replace("exp_20", "") for value in order]
    fig = plt.figure(figsize=(7.2, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    ax_a = fig.add_subplot(grid[0, :])
    for modality in ("rgb", "time", "rgb_time"):
        rows = experiments.loc[experiments["modality"].eq(modality)].set_index(
            "experiment_id"
        ).reindex(order)
        ax_a.plot(
            x,
            rows["balanced_accuracy"],
            marker="o",
            ms=3.5,
            lw=1,
            color=COLORS[modality],
            label=LABELS[modality],
        )
    ax_a.set_xticks(x, short_labels, rotation=45, ha="right")
    ax_a.set(ylabel="Balanced accuracy", ylim=(0.5, 1.02))
    ax_a.legend(ncol=3, loc="lower left", fontsize=6.5)

    ax_b = fig.add_subplot(grid[1, 0])
    for modality in ("rgb", "time", "rgb_time"):
        rows = experiments.loc[experiments["modality"].eq(modality)].set_index(
            "experiment_id"
        ).reindex(order)
        ax_b.plot(
            x,
            rows["balanced_misclassification_regret"] * 100,
            marker="o",
            ms=3.5,
            lw=1,
            color=COLORS[modality],
            label=LABELS[modality],
        )
    ax_b.set_xticks(x, short_labels, rotation=45, ha="right")
    ax_b.set(ylabel="Class-balanced error regret (%)")

    ax_c = fig.add_subplot(grid[1, 1])
    labels = failures["cycle_name"].str.replace("frost_cycle_", "", regex=False)
    values = failures["mean_misclassification_regret"] * 100
    ax_c.barh(np.arange(len(failures)), values, color=RED, alpha=0.82)
    ax_c.set_yticks(np.arange(len(failures)), labels)
    ax_c.set(xlabel="Mean misclassification regret (%)", ylabel="Highest-regret cycles")
    for position, (value, error_rate) in enumerate(
        zip(values, failures["error_rate"], strict=True)
    ):
        ax_c.text(value, position, f" {error_rate:.0%} errors", va="center", fontsize=5.8)

    for label, axis in zip("abc", (ax_a, ax_b, ax_c), strict=True):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontsize=10, fontweight="bold")
    export(fig, output / "figure_4_failure_audit")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path, default=Path("report/03_RGB标签与模型/全量模态比较")
    )
    parser.add_argument("--output", type=Path, default=Path("report/04_论文图表/图表"))
    args = parser.parse_args()
    render_figures(
        pd.read_csv(args.results / "summary_metrics.csv"),
        pd.read_csv(args.results / "modality_deltas.csv"),
        pd.read_csv(args.results / "experiment_metrics.csv"),
        pd.read_parquet(args.results / "predictions.parquet"),
        args.output,
    )


if __name__ == "__main__":
    main()
