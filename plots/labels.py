"""Plot the publication evidence linking cost regret to RGB labels."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from labels.build import high_confidence_coverage

__all__ = ["plot_label_figures"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

_BLUE = "#3775BA"
_ORANGE = "#E28E2C"
_GREY = "#767676"
_PALE = "#DDEAF4"


def regret_threshold_summary(
    bands: pd.DataFrame, label_balance: pd.DataFrame
) -> pd.DataFrame:
    """Summarize timing ambiguity and retained image coverage by regret threshold."""
    summary = (
        bands.groupby("relative_regret_threshold", as_index=False)
        .agg(median_width_minutes=("band_width_minutes", "median"))
        .rename(
            columns={
                "relative_regret_threshold": "regret_threshold",
            }
        )
    )
    summary["eligible_image_coverage"] = [
        high_confidence_coverage(label_balance, "all", threshold)
        for threshold in summary["regret_threshold"]
    ]
    return summary


def _export(fig: plt.Figure, stem: Path, formats: tuple[str, ...] = ("png",)) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for figure_format in formats:
        path = stem.with_suffix(f".{figure_format}")
        fig.savefig(
            path,
            dpi=300 if figure_format == "png" else None,
            bbox_inches="tight",
        )
        if figure_format == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def _band_widths(cost: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    eligible = cost.loc[cost["optimization_eligible"].fillna(False)].copy()
    eligible["candidate_time"] = pd.to_datetime(
        eligible["candidate_time"], errors="coerce", format="mixed"
    )
    eligible["relative_regret"] = pd.to_numeric(eligible["relative_regret"], errors="coerce")
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        selected = eligible.loc[eligible["relative_regret"].le(float(threshold))]
        for cycle_name, curve in selected.groupby("cycle_name", sort=True):
            rows.append(
                {
                    "cycle_name": cycle_name,
                    "relative_regret_threshold": float(threshold),
                    "band_width_minutes": (
                        curve["candidate_time"].max() - curve["candidate_time"].min()
                    ).total_seconds()
                    / 60,
                }
            )
    return pd.DataFrame(rows)


def plot_label_figures(
    *,
    cost: pd.DataFrame,
    labels: pd.DataFrame,
    balance: pd.DataFrame,
    thresholds: Sequence[float],
    output: Path,
    source_output: Path,
    figure_formats: tuple[str, ...] = ("png",),
) -> None:
    """Export four standalone figures, one conclusion per figure."""
    bands = _band_widths(cost, thresholds)
    threshold_summary = regret_threshold_summary(bands, balance)
    splits = labels[["experiment_id", "cycle_name", "split"]].drop_duplicates()
    split_summary = (
        splits.groupby("split", as_index=False)
        .agg(experiment_count=("experiment_id", "nunique"), cycle_count=("cycle_name", "nunique"))
        .set_index("split")
        .reindex(["train", "validation", "test"])
        .reset_index()
    )

    representative = str(cost.groupby("cycle_name").size().idxmax())
    curve = cost.loc[cost["cycle_name"].eq(representative)].copy()
    curve["candidate_time"] = pd.to_datetime(
        curve["candidate_time"], errors="coerce", format="mixed"
    )
    curve["minutes"] = (
        curve["candidate_time"] - curve["candidate_time"].min()
    ).dt.total_seconds() / 60
    optimum = curve.loc[curve["inverse_cop"].idxmin(), "candidate_time"]
    curve["state_01pct"] = np.where(
        curve["relative_regret"].le(0.01),
        "near-optimal",
        np.where(curve["candidate_time"].lt(optimum), "pre-optimal", "post-optimal"),
    )

    source_output.mkdir(parents=True, exist_ok=True)
    bands.to_csv(source_output / "near_optimal_band_widths.csv", index=False)
    threshold_summary.to_csv(source_output / "threshold_summary.csv", index=False)
    split_summary.to_csv(source_output / "split_summary.csv", index=False)
    curve.to_csv(source_output / "representative_inverse_cop_curve.csv", index=False)

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
    ax.plot(curve["minutes"], curve["inverse_cop"], color=_GREY, lw=1.3, zorder=1)
    colors = {"pre-optimal": _BLUE, "near-optimal": _ORANGE, "post-optimal": "#B64342"}
    for state, rows in curve.groupby("state_01pct", sort=False):
        ax.scatter(
            rows["minutes"],
            rows["inverse_cop"],
            s=10,
            color=colors[str(state)],
            label=state,
        )
    ax.set(xlabel="Candidate time from search start (min)", ylabel="Cycle inverse COP (-)")
    ax.legend(ncol=1, loc="best", fontsize=7)
    _export(fig, output / "figure_2_inverse_cop_example", figure_formats)

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
            color=_BLUE,
            linewidth=0,
        )
        median = float(values.median())
        ax.plot([pos - 0.22, pos + 0.22], [median, median], color="#272727", lw=1.5)
    ax.set_xticks(positions, [f"{value:.0%}" for value in threshold_summary["regret_threshold"]])
    ax.set(xlabel="Relative-regret threshold", ylabel="Near-optimal envelope width (min)")
    _export(fig, output / "figure_3_near_optimal_width", figure_formats)

    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    ax.plot(
        threshold_summary["regret_threshold"] * 100,
        threshold_summary["eligible_image_coverage"] * 100,
        marker="o",
        color=_BLUE,
    )
    for row in threshold_summary.itertuples(index=False):
        ax.annotate(
            f"{row.eligible_image_coverage:.1%}",
            (
                cast(float, row.regret_threshold) * 100,
                cast(float, row.eligible_image_coverage) * 100,
            ),
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
    _export(fig, output / "figure_4_label_coverage", figure_formats)

    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    x = np.arange(len(split_summary))
    width = 0.36
    ax.bar(
        x - width / 2,
        split_summary["experiment_count"],
        width,
        color=_BLUE,
        label="Experiments",
    )
    ax.bar(
        x + width / 2,
        split_summary["cycle_count"],
        width,
        color=_PALE,
        edgecolor=_BLUE,
        label="Cycles",
    )
    ax.set_xticks(x, ["Train", "Validation", "Test"])
    ax.set(ylabel="Independent units")
    ax.legend(fontsize=7)
    _export(fig, output / "figure_5_split_independence", figure_formats)
