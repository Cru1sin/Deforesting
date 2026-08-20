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
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("report/paper_figures"))
    args = parser.parse_args()

    curves = pd.read_parquet(
        "report/raw_optimal_defrost/source_data/candidate_cost_curves.parquet"
    )
    bands = pd.read_csv(
        "report/raw_optimal_defrost/source_data/near_optimal_band_sensitivity.csv"
    )
    balance = pd.read_csv("report/rgb_cost_labels/label_balance.csv")
    splits = pd.read_csv("report/rgb_cost_labels/cycle_splits.csv")
    threshold_summary = regret_threshold_summary(bands, balance)

    source = args.output / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    threshold_summary.to_csv(source / "figure_2_threshold_summary.csv", index=False)
    split_summary = (
        splits.groupby("split", as_index=False)
        .agg(experiment_count=("experiment_id", "nunique"), cycle_count=("cycle_name", "nunique"))
        .set_index("split")
        .reindex(["train", "validation", "test"])
        .reset_index()
    )
    split_summary.to_csv(source / "figure_2_split_summary.csv", index=False)

    # Illustrative only: this cycle contains substantial pre/near/post support.
    representative = "frost_cycle_000020"
    curve = curves.loc[curves["cycle_name"].eq(representative)].copy()
    curve["minutes"] = (
        pd.to_datetime(curve["candidate_time"]) - pd.to_datetime(curve["candidate_time"]).min()
    ).dt.total_seconds() / 60
    optimum = curve.loc[curve["renewal_cost_kw"].idxmin(), "candidate_time"]
    curve["state_01pct"] = np.where(
        curve["relative_regret"].le(0.01),
        "near-optimal",
        np.where(
            pd.to_datetime(curve["candidate_time"]).lt(optimum),
            "pre-optimal",
            "post-optimal",
        ),
    )
    curve.to_csv(source / "figure_2_representative_curve.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(curve["minutes"], curve["renewal_cost_kw"], color=GREY, lw=1.3, zorder=1)
    colors = {"pre-optimal": BLUE, "near-optimal": ORANGE, "post-optimal": "#B64342"}
    for state, rows in curve.groupby("state_01pct", sort=False):
        ax.scatter(rows["minutes"], rows["renewal_cost_kw"], s=10, color=colors[state], label=state)
    ax.set(xlabel="Candidate time from search start (min)", ylabel="Renewal cost (kW-eq.)")
    ax.legend(ncol=1, loc="best", fontsize=7)

    ax = axes[0, 1]
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

    ax = axes[1, 0]
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

    ax = axes[1, 1]
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

    for label, ax in zip("abcd", axes.flat, strict=True):
        ax.text(-0.16, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold")
    export(fig, args.output / "figure_2_regret_to_rgb_labels")
    plt.close(fig)


if __name__ == "__main__":
    main()
