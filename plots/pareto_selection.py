"""Render already-selected cycle-performance and Pareto results."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

COP_COLOR = "#3C6E8F"
OPTIMAL_COLOR = "#C77836"
RB_COLOR = "#287D68"

_STYLES = {
    "cycle_cop": (COP_COLOR, "-", "cycle COP"),
    "cycle_heating_rate_kw": ("#B64A50", "-", "Heating rate [kW]"),
    "cycle_evaporator_capacity_kw": ("#2A788E", "--", "Evaporator capacity [kW]"),
}
def clip_point_to_axes(
    axis: Any, x: float, y: float, *, pad_fraction: float = .025
) -> tuple[float, float, str]:
    """Keep an out-of-view marker at the frame edge and return its direction."""
    xlim, ylim = axis.get_xlim(), axis.get_ylim()
    dx = -1 if x < xlim[0] else 1 if x > xlim[1] else 0
    dy = -1 if y < ylim[0] else 1 if y > ylim[1] else 0
    arrows = {
        (-1, -1): r"$\swarrow$", (0, -1): r"$\downarrow$", (1, -1): r"$\searrow$",
        (-1, 0): r"$\leftarrow$", (0, 0): "", (1, 0): r"$\rightarrow$",
        (-1, 1): r"$\nwarrow$", (0, 1): r"$\uparrow$", (1, 1): r"$\nearrow$",
    }
    xpad, ypad = pad_fraction * np.diff(xlim)[0], pad_fraction * np.diff(ylim)[0]
    return (
        float(xlim[0] + xpad if dx < 0 else xlim[1] - xpad if dx > 0 else x),
        float(ylim[0] + ypad if dy < 0 else ylim[1] - ypad if dy > 0 else y),
        arrows[(dx, dy)],
    )


def _plot_supported_curve(axis, minutes, raw, eligible, *, color, label):
    """Interpolate display gaps; keep the original candidate support unchanged."""
    displayed = pd.Series(raw.to_numpy(), index=minutes).interpolate(
        method="index", limit_area="inside"
    )
    axis.plot(minutes, displayed, color=color, linestyle="--", linewidth=1.2)
    return axis.plot(
        minutes, displayed.where(eligible.to_numpy()), color=color,
        linestyle="-", linewidth=1.35, label=label,
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
            handles.append(_plot_supported_curve(
                target, curve["minutes"], raw, eligible, color=color, label="Effective cycle COP"
            ))
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
                axis, curve["minutes"], normalized, eligible,
                color="black", label="Normalized cycle COP",
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
        "Cycle COP / optimum [%]" if len(metrics) == 1
        else "Relative to best performance [%]", fontsize=8, labelpad=8
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


def plot_cop_heating_rate_pareto(
    axis: Any,
    values: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    rb_time: object | None = None,
    local: bool = False,
    local_window_minutes: float = 15,
    local_center_time: object | None = None,
    selection_label: str = "Optimal defrost point",
) -> None:
    """Plot the stored C/H Pareto result, using O only as point colour."""
    curve = values.copy()
    curve["candidate_defrost_time"] = pd.to_datetime(
        curve["candidate_defrost_time"], errors="coerce"
    )
    curve["minutes"] = (curve["candidate_defrost_time"] - origin).dt.total_seconds() / 60
    valid = (
        curve["cycle_cop_eligible"].fillna(False).astype(bool)
        & curve["cycle_heating_rate_kw_eligible"].fillna(False).astype(bool)
        & curve[["cycle_cop", "cycle_heating_rate_kw"]]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .all(axis=1)
    )
    pareto = curve.loc[
        valid & curve["is_cop_heating_rate_pareto_point"].fillna(False).astype(bool)
    ].sort_values("candidate_defrost_time", kind="stable")
    selected = curve.loc[curve["is_selected_pareto_point"].fillna(False).astype(bool)]
    rb_timestamp = pd.to_datetime(rb_time, errors="coerce")
    rb = curve.iloc[0:0]
    if pd.notna(rb_timestamp) and valid.any():
        rb_index = (
            (curve.loc[valid, "candidate_defrost_time"] - pd.Timestamp(rb_timestamp)).abs().idxmin()
        )
        rb = curve.loc[[rb_index]]

    if not valid.any():
        axis.set_title("Pareto unavailable", loc="left", fontsize=6.5, color="#4B5563")
        axis.text(
            .5, .5, "No eligible C/H candidates\nNo selection; excluded from statistics",
            transform=axis.transAxes, ha="center", va="center", fontsize=7, color="#667085",
        )
        axis.set_box_aspect(1)
        axis.set_xlabel("cycle COP", fontsize=8)
        axis.set_ylabel("Heating rate [kW]", fontsize=8)
        axis.tick_params(labelsize=6.5)
        axis.grid(alpha=0.15)
        return

    center = pd.to_datetime(local_center_time, errors="coerce")
    if pd.isna(center) and not selected.empty:
        center = pd.Timestamp(selected.iloc[0]["candidate_defrost_time"])
    if local and pd.isna(center):
        axis.set_title(
            "Optimal-point detail unavailable", loc="left", fontsize=6.5, color="#4B5563"
        )
        axis.text(
            .5, .5, "No optimal defrost point\nLocal time view unavailable",
            transform=axis.transAxes, ha="center", va="center", fontsize=7, color="#667085",
        )
        axis.set_box_aspect(1)
        axis.set_xlabel("cycle COP", fontsize=8)
        axis.set_ylabel("Heating rate [kW]", fontsize=8)
        axis.tick_params(labelsize=6.5)
        axis.grid(alpha=0.15)
        return

    focus = valid.copy()
    if local and pd.notna(center):
        half_window = pd.Timedelta(minutes=local_window_minutes)
        focus = valid & curve["candidate_defrost_time"].between(
            center - half_window,
            center + half_window,
        )

    for setter, metric in ((axis.set_xlim, "cycle_cop"), (axis.set_ylim, "cycle_heating_rate_kw")):
        shown = curve.loc[focus, metric].dropna()
        if not shown.empty:
            lower, upper = float(shown.min()), float(shown.max())
            span = max(upper - lower, max(abs(lower), abs(upper)) * 0.04, 1e-6)
            setter(lower - 0.02 * span, upper + 0.02 * span)

    colour_source = pd.to_numeric(
        curve.loc[focus, "cycle_evaporator_capacity_kw"], errors="coerce"
    ).dropna()
    if colour_source.empty:
        norm = Normalize(0.0, 1.0, clip=True)
    else:
        midpoint = float(colour_source.median())
        vmin, vmax = float(colour_source.min()), float(colour_source.max())
        minimum_span = max(0.02, 0.01 * abs(midpoint))
        if vmax - vmin < minimum_span:
            vmin, vmax = midpoint - minimum_span / 2, midpoint + minimum_span / 2
        norm = Normalize(vmin, vmax, clip=True)

    window = curve.loc[focus].sort_values("candidate_defrost_time", kind="stable")
    common = curve.loc[valid].sort_values("candidate_defrost_time", kind="stable")
    axis.plot(
        common["cycle_cop"],
        common["cycle_heating_rate_kw"],
        color="#D1D5DB",
        linewidth=0.55,
        zorder=0,
    )
    coloured = focus & pd.to_numeric(
        curve["cycle_evaporator_capacity_kw"], errors="coerce"
    ).notna()
    points = axis.scatter(
        curve.loc[coloured, "cycle_cop"],
        curve.loc[coloured, "cycle_heating_rate_kw"],
        c=curve.loc[coloured, "cycle_evaporator_capacity_kw"],
        cmap="viridis",
        norm=norm,
        s=24,
        alpha=0.52,
        linewidths=0,
    )
    if coloured.any():
        colourbar = axis.figure.colorbar(
            points, cax=axis.inset_axes([1.02, 0, 0.018, 1], transform=axis.transAxes)
        )
        colourbar.set_label("Evaporator capacity [kW]", fontsize=8)
        colourbar.ax.tick_params(labelsize=6.5)
    shown_pareto = pareto.loc[pareto.index.intersection(curve.index[focus])]
    if not shown_pareto.empty:
        axis.scatter(
            shown_pareto["cycle_cop"], shown_pareto["cycle_heating_rate_kw"], s=56,
            facecolors="none", edgecolors="#475467", linewidths=0.55,
            zorder=3,
        )
    if not selected.empty:
        axis.scatter(
            selected["cycle_cop"], selected["cycle_heating_rate_kw"], s=80, marker="D",
            facecolors="none", edgecolors="#D97706", linewidths=1.0,
            label=selection_label, zorder=4,
        )
    if not rb.empty:
        rb_x, rb_y, rb_direction = clip_point_to_axes(
            axis, rb["cycle_cop"].iloc[0], rb["cycle_heating_rate_kw"].iloc[0]
        )
        axis.scatter(
            rb_x,
            rb_y,
            s=64,
            marker="s",
            facecolors="none",
            edgecolors="#2E7D5B",
            linewidths=1.0,
            label="RB trigger",
            zorder=4,
        )
        axis.annotate(
            f"RB {rb['minutes'].iloc[0]:.0f} {rb_direction}".rstrip(),
            (rb_x, rb_y),
            xytext=(6, -7),
            textcoords="offset points",
            fontsize=5.5,
            color="#2E7D5B",
        )

    labels = []
    label_x: list[float] = []
    label_y: list[float] = []
    if not pareto.empty:
        common_minutes = window["minutes"].dropna()
        title = (
            f"Knee time window (±{local_window_minutes:g} min) · O local · "
            if local and pd.notna(center) else
            "No optimal point · full Pareto domain · "
            if local else
            "Full Pareto domain · O global · "
        )
        axis.set_title(
            title + f"{common_minutes.min():.0f}–{common_minutes.max():.0f} min",
            loc="left",
            fontsize=6.5,
            color="#4B5563",
            pad=3,
        )
        xlim, ylim = axis.get_xlim(), axis.get_ylim()
        select_labeled_images = curve.loc[
            curve["is_cop_heating_rate_pareto_point"].fillna(False).astype(bool)
            & focus
            & curve["cycle_cop"].between(*xlim)
            & curve["cycle_heating_rate_kw"].between(*ylim)
        ].copy()
        if not local and not common.empty:
            select_labeled_images = pd.concat(
                [select_labeled_images, common.iloc[[0, -1]]], ignore_index=True
            )
        select_labeled_images["_time_label"] = select_labeled_images["minutes"].map(
            lambda value: f"{value:.0f}"
        )
        select_labeled_images = (
            select_labeled_images.sort_values(
                ["is_selected_pareto_point", "minutes"], ascending=[False, True]
            )
            .drop_duplicates("_time_label")
            .sort_values("minutes", kind="stable")
        )
        for _, row in select_labeled_images.iterrows():
            label = axis.text(
                row["cycle_cop"],
                row["cycle_heating_rate_kw"],
                row["_time_label"],
                ha="center",
                va="center",
                fontsize=5.5,
                color="#D97706" if bool(row["is_selected_pareto_point"]) else "#667085",
            )
            label.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground="white")])
            labels.append(label)
            label_x.append(float(row["cycle_cop"]))
            label_y.append(float(row["cycle_heating_rate_kw"]))
    axis.set_box_aspect(1)
    axis.set_xlabel("cycle COP", fontsize=8)
    axis.set_ylabel("Heating rate [kW]", fontsize=8)
    axis.tick_params(labelsize=6.5)
    axis.grid(alpha=0.15)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(frameon=False, fontsize=6.5, ncol=2 if not rb.empty else 3)
    if labels:
        adjust_text(
            labels,
            x=window["cycle_cop"].to_numpy(dtype=float),
            y=window["cycle_heating_rate_kw"].to_numpy(dtype=float),
            target_x=label_x,
            target_y=label_y,
            ax=axis,
            expand=(1.3, 1.5),
            force_text=(0.8, 1.0),
            force_static=(0.5, 0.8),
            force_pull=(0.01, 0.01),
            force_explode=(0.5, 0.8),
            prevent_crossings=True,
            ensure_inside_axes=True,
            min_arrow_len=0,
            arrowprops={"arrowstyle": "-", "linewidth": 0.35, "alpha": 0.6},
        )
