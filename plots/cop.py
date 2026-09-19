"""Shared effective COP curves; extrapolated predictions are dashed."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

COP_COLOR = "#3C6E8F"
OPTIMAL_COLOR = "#C77836"
RB_COLOR = "#287D68"

_STYLES = {
    "cycle_cop": (COP_COLOR, "-", "cycle COP"),
    "cycle_heating_rate_kw": ("#B64A50", "-", "Heating rate [kW]"),
    "cycle_evaporator_capacity_kw": ("#2A788E", "--", "Evaporator capacity [kW]"),
}


def _plot_supported_curve(axis, minutes, raw, eligible, *, color, label):
    """Interpolate display gaps; keep the original candidate support unchanged."""
    displayed = pd.Series(raw.to_numpy(), index=minutes).interpolate(
        method="index", limit_area="inside"
    )
    axis.plot(minutes, displayed, color=color, linestyle="--", linewidth=1.2)
    return axis.plot(
        minutes,
        displayed.where(eligible.to_numpy()),
        color=color,
        linestyle="-",
        linewidth=1.35,
        label=label,
    )[0]


def plot_objectives(
    axis: Any,
    values: pd.DataFrame,
    origin: pd.Timestamp,
    stage_spans: list[tuple[str, float, float]] | None = None,
    shader: Callable[..., None] | None = None,
    metrics=("cycle_cop", "cycle_heating_rate_kw", "cycle_evaporator_capacity_kw"),
) -> list[Any]:
    """Plot raw C/H/O trajectories without changing their eligibility or selection."""
    curve = values.copy()
    curve["candidate_defrost_time"] = pd.to_datetime(
        curve["candidate_defrost_time"], errors="coerce"
    )
    curve["minutes"] = (curve["candidate_defrost_time"] - origin).dt.total_seconds() / 60
    axes = [axis] + [axis.twinx() for _ in metrics[1:]]
    if len(axes) > 2:
        axes[2].spines["right"].set_position(("axes", 1.09))
        axes[2].spines["right"].set_visible(True)
    if shader is not None and stage_spans:
        shader(axis, stage_spans, [])
    handles = []
    for target, metric in zip(axes, metrics, strict=True):
        color, linestyle, label = _STYLES[metric]
        raw = pd.to_numeric(curve[metric], errors="coerce")
        eligible = curve[f"{metric}_eligible"].fillna(False).astype(bool)
        if len(metrics) == 1:
            handles.append(
                _plot_supported_curve(
                    target,
                    curve["minutes"],
                    raw,
                    eligible
                    & ~curve.get(
                        f"{metric}_uses_model_extrapolation", pd.Series(False, index=curve.index)
                    ).fillna(False),
                    color=color,
                    label="Effective cycle COP",
                )
            )
        else:
            target.plot(
                curve["minutes"], raw, color=color, linestyle=linestyle, linewidth=0.8, alpha=0.25
            )
            handles.append(
                target.plot(
                    curve["minutes"],
                    raw.where(eligible),
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.35,
                    label=label.split(" [", maxsplit=1)[0],
                )[0]
            )
            target.scatter(
                curve.loc[~eligible, "minutes"],
                raw.loc[~eligible],
                s=9,
                marker="x",
                linewidths=0.5,
                color=color,
                alpha=0.35,
            )
        target.set_ylabel(label, fontsize=8, labelpad=8)
        target.tick_params(axis="y", labelsize=6.5)
        display_maximum = raw.max()
        if np.isfinite(display_maximum) and display_maximum > 0:
            lower, upper = 0.75 * display_maximum, 1.02 * display_maximum
            ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(lower, upper)
            target.set_ylim(lower, upper)
            target.set_yticks(ticks[(ticks >= lower) & (ticks <= upper)])
            target.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        fontsize=7,
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
    )
    axis.grid(axis="x", alpha=0.12)
    return axes


def plot_normalized(
    axis: Any,
    values: pd.DataFrame,
    origin: pd.Timestamp,
    stage_spans: list[tuple[str, float, float]] | None = None,
    shader: Callable[..., None] | None = None,
    metrics=tuple(_STYLES),
) -> None:
    """Plot every objective relative only to its own eligible maximum."""
    curve = values.copy()
    curve["candidate_defrost_time"] = pd.to_datetime(
        curve["candidate_defrost_time"], errors="coerce"
    )
    curve["minutes"] = (curve["candidate_defrost_time"] - origin).dt.total_seconds() / 60
    if shader is not None and stage_spans:
        shader(axis, stage_spans, [])
    for metric in metrics:
        color, linestyle, label = _STYLES[metric]
        raw = pd.to_numeric(curve[metric], errors="coerce")
        eligible = curve[f"{metric}_eligible"].fillna(False).astype(bool)
        best = raw.where(eligible).max()
        normalized = 100 * raw / best if np.isfinite(best) and best != 0 else raw * np.nan
        if len(metrics) == 1:
            _plot_supported_curve(
                axis,
                curve["minutes"],
                normalized,
                eligible
                & ~curve.get(
                    f"{metric}_uses_model_extrapolation", pd.Series(False, index=curve.index)
                ).fillna(False),
                color="black",
                label="Normalized cycle COP",
            )
        else:
            axis.plot(
                curve["minutes"],
                normalized.where(eligible),
                color=color,
                linestyle=linestyle,
                linewidth=1.35,
                label=label.split(" [", maxsplit=1)[0],
            )
    axis.axhline(100, color="#7A7A7A", linestyle=":", linewidth=0.75)
    for loss, level in ((1, 99), (2, 98), (5, 95)):
        axis.axhline(level, color="#9CA3AF", linestyle="--", linewidth=0.55, alpha=0.7)
        axis.text(
            0.995,
            level + 0.08,
            f"{loss}%",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=5.5,
            color="#6B7280",
        )
    axis.set_ylim(90, 100.8)
    axis.set_yticks([90, 92, 94, 96, 98, 100])
    axis.set_ylabel(
        "Cycle COP / optimum [%]" if len(metrics) == 1 else "Relative to best performance [%]",
        fontsize=8,
        labelpad=8,
    )
    axis.tick_params(labelsize=6.5)
    axis.grid(axis="x", alpha=0.12)
    axis.legend(
        frameon=False,
        ncol=3,
        fontsize=7,
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        columnspacing=1.2,
    )
