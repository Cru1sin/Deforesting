#!/usr/bin/env python3
"""Plot the publication evidence linking cost regret to RGB labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frost_analysis.paper_figures import regret_threshold_summary

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

BLUE = "#3775BA"
ORANGE = "#E28E2C"
GREY = "#767676"
PALE = "#DDEAF4"


def export(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg = stem.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cost_to_rgb_evidence(
    curves: pd.DataFrame,
    bands: pd.DataFrame,
    balance: pd.DataFrame,
    splits: pd.DataFrame,
    output: Path,
) -> None:
    """Export four standalone figures, one conclusion per figure."""
    threshold_summary = regret_threshold_summary(bands, balance)
    source = output.parent / "源数据"
    source.mkdir(parents=True, exist_ok=True)
    threshold_summary.to_csv(source / "threshold_summary.csv", index=False)
    split_summary = (
        splits.groupby("split", as_index=False)
        .agg(experiment_count=("experiment_id", "nunique"), cycle_count=("cycle_name", "nunique"))
        .set_index("split")
        .reindex(["train", "validation", "test"])
        .reset_index()
    )
    split_summary.to_csv(source / "split_summary.csv", index=False)

    representative = str(curves.groupby("cycle_name").size().idxmax())
    curve = curves.loc[curves["cycle_name"].eq(representative)].copy()
    curve["minutes"] = (
        pd.to_datetime(curve["candidate_time"]) - pd.to_datetime(curve["candidate_time"]).min()
    ).dt.total_seconds() / 60
    optimum = curve.loc[curve["inverse_cop"].idxmin(), "candidate_time"]
    curve["state_01pct"] = np.where(
        curve["relative_regret"].le(0.01),
        "near-optimal",
        np.where(
            pd.to_datetime(curve["candidate_time"]).lt(optimum),
            "pre-optimal",
            "post-optimal",
        ),
    )
    curve.to_csv(source / "representative_inverse_cop_curve.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    ax.plot(curve["minutes"], curve["inverse_cop"], color=GREY, lw=1.3, zorder=1)
    colors = {"pre-optimal": BLUE, "near-optimal": ORANGE, "post-optimal": "#B64342"}
    for state, rows in curve.groupby("state_01pct", sort=False):
        ax.scatter(rows["minutes"], rows["inverse_cop"], s=10, color=colors[state], label=state)
    ax.set(xlabel="Candidate time from search start (min)", ylabel="Cycle inverse COP (-)")
    ax.legend(ncol=1, loc="best", fontsize=7)
    export(fig, output / "figure_2_inverse_cop_example")

    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    rng = np.random.default_rng(0)
    positions = np.arange(len(threshold_summary))
    for pos, threshold in zip(positions, threshold_summary["regret_threshold"], strict=True):
        values = bands.loc[
            bands["relative_regret_threshold"].eq(threshold), "band_width_minutes"
        ].dropna()
        ax.scatter(
            pos + rng.uniform(-0.12, 0.12, len(values)),
            values,
            s=9,
            alpha=0.45,
            color=BLUE,
            linewidth=0,
        )
        median = float(values.median())
        ax.plot([pos - 0.22, pos + 0.22], [median, median], color="#272727", lw=1.5)
    ax.set_xticks(positions, [f"{value:.0%}" for value in threshold_summary["regret_threshold"]])
    ax.set(xlabel="Relative-regret threshold", ylabel="Near-optimal envelope width (min)")
    export(fig, output / "figure_3_near_optimal_width")

    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    ax.plot(
        threshold_summary["regret_threshold"] * 100,
        threshold_summary["eligible_image_coverage"] * 100,
        marker="o",
        color=BLUE,
    )
    for row in threshold_summary.itertuples(index=False):
        ax.annotate(
            f"{row.eligible_image_coverage:.1%}",
            (row.regret_threshold * 100, row.eligible_image_coverage * 100),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    ax.set(
        xlabel="Relative-regret threshold (%)",
        ylabel="High-confidence image coverage (%)",
        ylim=(0, 75),
    )
    export(fig, output / "figure_4_label_coverage")

    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    x = np.arange(len(split_summary))
    width = 0.36
    ax.bar(x - width / 2, split_summary["experiment_count"], width, color=BLUE, label="Experiments")
    ax.bar(
        x + width / 2,
        split_summary["cycle_count"],
        width,
        color=PALE,
        edgecolor=BLUE,
        label="Cycles",
    )
    ax.set_xticks(x, ["Train", "Validation", "Test"])
    ax.set(ylabel="Independent units")
    ax.legend(fontsize=7)
    export(fig, output / "figure_5_split_independence")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/test/model/论文图表/图表"))
    args = parser.parse_args()
    plot_cost_to_rgb_evidence(
        pd.read_parquet(
            "output/test/成本函数/其他/经验经济窗口/源数据/candidate_cost_curves.parquet"
        ),
        pd.read_csv(
            "output/test/成本函数/其他/经验经济窗口/源数据/near_optimal_band_sensitivity.csv"
        ),
        pd.read_csv("output/label/cost_function_v1_binary/label_balance.csv"),
        pd.read_csv("output/label/cost_function_v1_binary/cycle_splits.csv"),
        args.output,
    )


if __name__ == "__main__":
    main()
