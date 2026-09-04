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

_STYLES = {
    "cycle_cop": ("#484878", "-", "cycle COP"),
    "cycle_heating_rate_kw": ("#B64A50", "-", "Heating rate [kW]"),
    "cycle_evaporator_capacity_kw": ("#2A788E", "--", "Evaporator capacity [kW]"),
}


def plot_objectives(
    axis: Any,
    values: pd.DataFrame,
    origin: pd.Timestamp,
    stage_spans: list[tuple[str, float, float]] | None = None,
    shader: Callable[..., None] | None = None,
) -> list[Any]:
    """Plot raw C/H/O trajectories without changing their eligibility or selection."""
    curve = values.copy()
    curve["candidate_defrost_time"] = pd.to_datetime(
        curve["candidate_defrost_time"], errors="coerce"
    )
    curve["minutes"] = (curve["candidate_defrost_time"] - origin).dt.total_seconds() / 60
    axes = [axis, axis.twinx(), axis.twinx()]
    axes[2].spines["right"].set_position(("axes", 1.09))
    axes[2].spines["right"].set_visible(True)
    if shader is not None and stage_spans:
        shader(axis, stage_spans, [])
    handles = []
    for target, metric in zip(
        axes, ("cycle_cop", "cycle_heating_rate_kw", "cycle_evaporator_capacity_kw"), strict=True
    ):
        color, linestyle, label = _STYLES[metric]
        raw = pd.to_numeric(curve[metric], errors="coerce")
        eligible = curve[f"{metric}_eligible"].fillna(False).astype(bool)
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
) -> None:
    """Plot every objective relative only to its own eligible maximum."""
    curve = values.copy()
    curve["candidate_defrost_time"] = pd.to_datetime(
        curve["candidate_defrost_time"], errors="coerce"
    )
    curve["minutes"] = (curve["candidate_defrost_time"] - origin).dt.total_seconds() / 60
    if shader is not None and stage_spans:
        shader(axis, stage_spans, [])
    for metric, (color, linestyle, label) in _STYLES.items():
        raw = pd.to_numeric(curve[metric], errors="coerce")
        eligible = curve[f"{metric}_eligible"].fillna(False).astype(bool)
        best = raw.where(eligible).max()
        normalized = 100 * raw / best if np.isfinite(best) and best != 0 else raw * np.nan
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
    axis.set_ylabel("Relative to best performance [%]", fontsize=8, labelpad=8)
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

    if selected.empty:
        display_focus = valid.copy()
    else:
        point = selected.iloc[0]
        display_focus = (
            valid
            & curve["cycle_cop"].between(0.95 * point["cycle_cop"], 1.05 * point["cycle_cop"])
            & curve["cycle_heating_rate_kw"].between(
                0.95 * point["cycle_heating_rate_kw"], 1.05 * point["cycle_heating_rate_kw"]
            )
        )
    display_focus.loc[rb.index] = True
    for setter, metric in ((axis.set_xlim, "cycle_cop"), (axis.set_ylim, "cycle_heating_rate_kw")):
        focus = curve.loc[
            display_focus | curve["is_cop_heating_rate_pareto_point"].fillna(False).astype(bool),
            metric,
        ].dropna()
        if not focus.empty:
            lower, upper = float(focus.min()), float(focus.max())
            reference = float(selected.iloc[0][metric]) if not selected.empty else upper
            span = max(upper - lower, abs(reference) * 0.04, 1e-6)
            setter(lower - 0.12 * span, upper + 0.12 * span)

    local_pareto_o = pd.to_numeric(
        curve.loc[
            curve["is_cop_heating_rate_pareto_point"].fillna(False).astype(bool) & display_focus,
            "cycle_evaporator_capacity_kw",
        ],
        errors="coerce",
    ).dropna()
    local_o = pd.to_numeric(
        curve.loc[display_focus, "cycle_evaporator_capacity_kw"], errors="coerce"
    ).dropna()
    colour_source = local_pareto_o if local_pareto_o.nunique() >= 3 else local_o
    if colour_source.empty:
        colour_source = pd.to_numeric(
            curve.loc[valid, "cycle_evaporator_capacity_kw"], errors="coerce"
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

    common = curve.loc[valid].sort_values("candidate_defrost_time", kind="stable")
    axis.plot(
        common["cycle_cop"],
        common["cycle_heating_rate_kw"],
        color="#D1D5DB",
        linewidth=0.55,
        zorder=0,
    )
    coloured = valid & pd.to_numeric(curve["cycle_evaporator_capacity_kw"], errors="coerce").notna()
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
    axis.scatter(
        pareto["cycle_cop"],
        pareto["cycle_heating_rate_kw"],
        s=56,
        facecolors="none",
        edgecolors="#475467",
        linewidths=0.55,
        label="Pareto front",
        zorder=3,
    )
    axis.scatter(
        selected["cycle_cop"],
        selected["cycle_heating_rate_kw"],
        s=80,
        marker="D",
        facecolors="none",
        edgecolors="#D97706",
        linewidths=1.0,
        label="Selected",
        zorder=4,
    )
    if not rb.empty:
        axis.scatter(
            rb["cycle_cop"],
            rb["cycle_heating_rate_kw"],
            s=64,
            marker="s",
            facecolors="none",
            edgecolors="#2E7D5B",
            linewidths=1.0,
            label="RB trigger",
            zorder=4,
        )
        axis.annotate(
            f"RB {rb['minutes'].iloc[0]:.0f}",
            (rb["cycle_cop"].iloc[0], rb["cycle_heating_rate_kw"].iloc[0]),
            xytext=(6, -7),
            textcoords="offset points",
            fontsize=5.5,
            color="#2E7D5B",
        )

    labels = []
    label_x: list[float] = []
    label_y: list[float] = []
    if not pareto.empty:
        axis.set_title(
            "Local Pareto view · local O scale · "
            f"full range {pareto['minutes'].min():.0f}–{pareto['minutes'].max():.0f} min",
            loc="left",
            fontsize=6.5,
            color="#4B5563",
            pad=3,
        )
        xlim, ylim = axis.get_xlim(), axis.get_ylim()
        select_labeled_images = curve.loc[
            curve["is_cop_heating_rate_pareto_point"].fillna(False).astype(bool)
            & curve["cycle_cop"].between(*xlim)
            & curve["cycle_heating_rate_kw"].between(*ylim)
        ].copy()
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
    axis.legend(frameon=False, fontsize=6.5, ncol=2 if not rb.empty else 3)
    if labels:
        adjust_text(
            labels,
            x=common["cycle_cop"].to_numpy(dtype=float),
            y=common["cycle_heating_rate_kw"].to_numpy(dtype=float),
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
