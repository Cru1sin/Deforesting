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


def original_missing_spans(frame, origin, *, columns=None):
    """Half-open input-interpolation intervals at sensor resolution.

    A metric supplies its own inputs; unrelated sensors must not gray its curve.
    The shared row flag remains available for whole-Dataset panels.
    """
    if frame.empty:
        return []
    if columns is None:
        if "is_interpolated" not in frame:
            return []
        missing = frame.is_interpolated.fillna(False).to_numpy(dtype=bool)
    else:
        missing = np.zeros(len(frame), dtype=bool)
        for name in columns:
            for suffix in ("_interpolated", "_filled"):
                if name in frame and name + suffix in frame:
                    missing |= (frame[name].isna() & frame[name + suffix].notna()).to_numpy()
    times = (pd.to_datetime(frame.timestamp) - origin).dt.total_seconds().to_numpy() / 60
    ends = np.r_[times[1:], times[-1] + (np.median(np.diff(times)) if len(times) > 1 else 1 / 60)]
    starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
    stops = np.flatnonzero(missing & ~np.r_[missing[1:], False])
    return list(zip(times[starts], ends[stops]))


def _plot_supported_curve(
    axis,
    minutes,
    raw,
    eligible,
    *,
    color,
    label,
    interpolate=True,
    dashed_alpha=None,
    extrapolated=None,
    missing_spans=(),
):
    """Interpolate display gaps; keep the original candidate support unchanged."""
    displayed = pd.Series(raw.to_numpy(), index=minutes)
    if interpolate:
        displayed = displayed.interpolate(method="index", limit_area="inside")
    if extrapolated is not None:
        # Split at missing-span boundaries, not just the coarser COP candidate grid.
        t = np.asarray(minutes, dtype=float)
        boundaries = np.asarray(missing_spans, dtype=float).reshape(-1)
        x = np.unique(np.r_[t, boundaries[(boundaries > t[0]) & (boundaries < t[-1])]])
        y = np.interp(x, t, displayed.to_numpy(dtype=float))
        middle = (x[:-1] + x[1:]) / 2
        left = np.clip(np.searchsorted(t, middle, side="right") - 1, 0, len(t) - 1)
        right = np.minimum(left + 1, len(t) - 1)
        outside = np.asarray(extrapolated, dtype=bool)
        dashed = outside[left] | outside[right]
        gray = np.zeros(len(middle), dtype=bool)
        for start, end in missing_spans:
            gray |= (middle >= start) & (middle < end)
        kinds = gray.astype(int) * 2 + dashed.astype(int)
        lines = []
        for kind, (ink, style) in enumerate(
            ((color, "-"), (color, "--"), ("0.6", "-"), ("0.6", "--"))
        ):
            mask = kinds == kind
            starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
            stops = np.flatnonzero(mask & ~np.r_[mask[1:], False])
            px, py = [], []
            for start, stop in zip(starts, stops):
                px.extend([*x[start : stop + 2], np.nan])
                py.extend([*y[start : stop + 2], np.nan])
            lines.append(
                axis.plot(
                    px,
                    py,
                    color=ink,
                    linestyle=style,
                    linewidth=1.35,
                    label=label if kind == 0 else "_nolegend_",
                )[0]
            )
        return lines[0]
    axis.plot(minutes, displayed, color=color, linestyle="--", linewidth=1.2, alpha=dashed_alpha)
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
                    eligible,
                    color=color,
                    label="Effective cycle COP",
                    extrapolated=curve.get(f"{metric}_uses_model_extrapolation", ~eligible).fillna(
                        False
                    ),
                    missing_spans=curve.attrs.get("original_missing_spans", ()),
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
    *,
    x_column: str | None = None,
    percent: bool = True,
) -> None:
    """Plot every objective relative only to its own eligible maximum."""
    curve = values.copy()
    curve["candidate_defrost_time"] = pd.to_datetime(
        curve["candidate_defrost_time"], errors="coerce"
    )
    curve["minutes"] = (curve["candidate_defrost_time"] - origin).dt.total_seconds() / 60
    if shader is not None and stage_spans:
        shader(axis, stage_spans, [])
    scale = 100 if percent else 1
    for metric in metrics:
        color, linestyle, label = _STYLES[metric]
        raw = pd.to_numeric(curve[metric], errors="coerce")
        eligible = curve[f"{metric}_eligible"].fillna(False).astype(bool)
        best = raw.where(eligible).max()
        normalized = scale * raw / best if np.isfinite(best) and best != 0 else raw * np.nan
        if x_column is not None:
            # A state coordinate can reverse: retain chronological order and gaps.
            gap = curve.candidate_defrost_time.diff().gt(pd.Timedelta(seconds=30))
            if "image_time" in curve:
                gap |= pd.to_datetime(curve.image_time).diff().gt(pd.Timedelta(seconds=45))
            for j, (_, part) in enumerate(curve.groupby(gap.cumsum(), sort=False)):
                axis.plot(
                    part[x_column],
                    normalized.loc[part.index].where(eligible.loc[part.index]),
                    color="black",
                    linewidth=1.35,
                    label="Normalized cycle COP" if j == 0 else "_nolegend_",
                )
        elif len(metrics) == 1:
            _plot_supported_curve(
                axis,
                curve["minutes"],
                normalized,
                eligible,
                color="black",
                label="Normalized cycle COP",
                extrapolated=curve.get(f"{metric}_uses_model_extrapolation", ~eligible).fillna(
                    False
                ),
                missing_spans=curve.attrs.get("original_missing_spans", ()),
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
    axis.axhline(scale, color="#7A7A7A", linestyle=":", linewidth=0.75)
    for loss, level in ((1, 99), (2, 98), (5, 95)):
        level = level * scale / 100
        axis.axhline(level, color="#9CA3AF", linestyle="--", linewidth=0.55, alpha=0.7)
        if x_column is None:
            axis.text(
                0.995,
                level + 0.0008 * scale,
                f"{loss}%",
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=5.5,
                color="#6B7280",
            )
    axis.set_ylim(0.9 * scale, 1.008 * scale)
    axis.set_yticks(np.array([0.9, 0.92, 0.94, 0.96, 0.98, 1]) * scale)
    axis.set_ylabel(
        ("Cycle COP / optimum" if len(metrics) == 1 else "Relative to best performance")
        + (" [%]" if percent else " [-]"),
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
