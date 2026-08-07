"""Pure renderers for self-contained Dataset artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

_PANELS = (
    (("compressor_frequency", "compressor_frequency_setpoint"), "Compressor frequency [Hz]"),
    (("heating_capacity",), "Heating capacity [kW]"),
    (("cop",), "COP [-]"),
    (
        ("water_in_temperature", "water_out_temperature", "water_temperature_setpoint"),
        "Water temperature [degC]",
    ),
    (
        ("ambient_temperature", "coil_temperature", "evaporating_temperature"),
        "Temperature [degC]",
    ),
)
_COLORS = {
    "compressor_frequency": "#0072B2",
    "compressor_frequency_setpoint": "#7A9CC6",
    "heating_capacity": "#D55E00",
    "cop": "#009E73",
    "ambient_temperature": "#374151",
    "water_in_temperature": "#E69F00",
    "water_out_temperature": "#CC79A7",
    "evaporating_temperature": "#56B4E9",
    "coil_temperature": "#7B2CBF",
    "water_temperature_setpoint": "#6B7280",
}
_STAGE_COLORS = {
    "recovery": "#78A6BC",
    "frost_development": "#F2A35E",
    "defrost": "#70B184",
}


def render_cycle_publication(
    cycle_frame: pd.DataFrame,
    cycle_record: Mapping[str, object],
    output_path: Path,
) -> None:
    """Render the Dataset's cycle-level scientific overview."""
    frame = cycle_frame.sort_values("timestamp", kind="stable").copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    origin = frame["timestamp"].min()
    minutes = (frame["timestamp"] - origin).dt.total_seconds() / 60.0
    humidity = [
        str(column)
        for column in frame
        if "humidity" in str(column).lower()
        and not str(column).endswith("__imputed")
        and _observed_values(frame, str(column)).notna().any()
    ]
    row_count = 1 + len(_PANELS) + bool(humidity)
    figure, axes = plt.subplots(
        row_count,
        1,
        figsize=(7.2, 10.8 + 1.25 * bool(humidity)),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [0.18, *([1.0] * (row_count - 1))], "hspace": 0.58},
    )
    stage_axis = axes[0]
    stage_spans = _stage_spans(frame, minutes)
    _plot_stage_ribbon(stage_axis, stage_spans)

    boundaries = cycle_record.get("boundaries")
    baseline = boundaries if isinstance(boundaries, Mapping) else cycle_record
    baseline_start = pd.to_datetime(baseline.get("baseline_start"), errors="coerce")
    baseline_end = pd.to_datetime(baseline.get("baseline_end"), errors="coerce")
    if not pd.isna(baseline_start) and not pd.isna(baseline_end):
        left = (pd.Timestamp(baseline_start) - origin).total_seconds() / 60.0
        right = (pd.Timestamp(baseline_end) - origin).total_seconds() / 60.0
    else:
        left = right = np.nan

    for axis, (channels, label) in zip(axes[1:6], _PANELS, strict=True):
        _plot_cycle_panel(axis, frame, minutes, channels, label, stage_spans, left, right)

    if humidity:
        axis = axes[-1]
        for channel in humidity:
            axis.plot(minutes, _observed_values(frame, channel), linewidth=1.2, label=channel)
        axis.set_ylabel("Relative humidity [%]", fontsize=8)
        axis.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.01))

    axes[-1].set_xlabel("Time from cycle start [min]", fontsize=8)
    figure.suptitle(
        f"{cycle_record.get('cycle_name', cycle_record.get('cycle_id', 'Cycle'))} | "
        f"{cycle_record.get('status', cycle_record.get('cycle_status', ''))}",
        x=0.12,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.20, right=0.98, bottom=0.06, top=0.94)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_stage_ribbon(axis: Any, spans: list[tuple[str, float, float]]) -> None:
    for stage, start, end in spans:
        axis.axvspan(start, end, color=_STAGE_COLORS.get(stage, "#BDBDBD"))
        axis.text(
            (start + end) / 2,
            0.5,
            stage.replace("_", " ").title(),
            ha="center",
            va="center",
            fontsize=7,
        )
    axis.set_yticks([])
    axis.tick_params(axis="x", labelbottom=False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _plot_cycle_panel(
    axis: Any,
    frame: pd.DataFrame,
    minutes: pd.Series,
    channels: tuple[str, ...],
    label: str,
    stage_spans: list[tuple[str, float, float]],
    baseline_start: float,
    baseline_end: float,
) -> None:
    for channel in channels:
        values = _observed_values(frame, channel)
        if values.notna().any():
            axis.plot(
                minutes,
                values,
                color=_COLORS[channel],
                linestyle="--" if channel.endswith("setpoint") else "-",
                linewidth=1.2,
                label=channel.replace("_", " ").title(),
            )
    for _stage, start, end in stage_spans:
        axis.axvspan(start, end, color="#D9DEE5", alpha=0.12, zorder=0)
    if np.isfinite(baseline_start) and np.isfinite(baseline_end):
        axis.axvspan(baseline_start, baseline_end, color="#6B7280", alpha=0.12, zorder=0)
    axis.set_ylabel(label, fontsize=8)
    axis.grid(axis="x", alpha=0.12)
    if axis.lines:
        axis.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.01))


def _observed_values(frame: pd.DataFrame, channel: str) -> pd.Series:
    if channel not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[channel], errors="coerce")
    imputed = frame.get(f"{channel}__imputed")
    if imputed is not None:
        values = values.mask(imputed.astype("boolean").fillna(False))
    return values.replace([np.inf, -np.inf], np.nan)


def _stage_spans(
    frame: pd.DataFrame, minutes: pd.Series
) -> list[tuple[str, float, float]]:
    if "cycle_stage" not in frame or frame.empty:
        return []
    step = float(minutes.diff().dropna().median()) if len(minutes) > 1 else 0.0
    groups = frame["cycle_stage"].astype("string").ne(frame["cycle_stage"].shift()).cumsum()
    return [
        (
            str(group["cycle_stage"].iloc[0]),
            float(minutes.loc[group.index].iloc[0]),
            float(minutes.loc[group.index].iloc[-1] + step),
        )
        for _, group in frame.groupby(groups, sort=False)
        if pd.notna(group["cycle_stage"].iloc[0])
    ]


def render_rgb_coverage_intervals(
    cycle_name: str,
    cycle_start: pd.Timestamp,
    cycle_end: pd.Timestamp,
    intervals: Mapping[str, Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]]],
    output_path: Path,
    *,
    sensor_intervals: Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]] | None = None,
) -> None:
    """Draw merged available/missing intervals using one coverage calculation."""
    start = pd.Timestamp(cycle_start)
    end = pd.Timestamp(cycle_end)
    duration_minutes = max((end - start).total_seconds() / 60.0, 1.0)
    rows: list[tuple[str, Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]]]] = []
    if sensor_intervals is not None:
        rows.append(("sensor_overall", sensor_intervals))
    rows.extend((str(role), value) for role, value in sorted(intervals.items()))
    if not rows:
        rows.append(("sensor_overall", {"available": [], "missing": [(start, end)]}))

    width = 183.0 / 25.4
    height = max(1.65, 0.43 * len(rows) + 0.78)
    figure, axis = plt.subplots(figsize=(width, height), dpi=300)
    axis.set_facecolor("white")
    bar_height = 0.56
    available_color = "#304B63"
    missing_face = "#F4F6F8"
    missing_edge = "#AEB8C3"
    for row_index, (label, role_intervals) in enumerate(rows):
        y = len(rows) - row_index - 1
        axis.add_patch(
            Rectangle(
                (0.0, y - bar_height / 2.0),
                duration_minutes,
                bar_height,
                facecolor=missing_face,
                edgecolor=missing_edge,
                linewidth=0.55,
                hatch="///",
                zorder=1,
            )
        )
        for available_start, available_end in role_intervals.get("available", []):
            left = (pd.Timestamp(available_start) - start).total_seconds() / 60.0
            right = (pd.Timestamp(available_end) - start).total_seconds() / 60.0
            if right > left:
                axis.add_patch(
                    Rectangle(
                        (left, y - bar_height / 2.0),
                        right - left,
                        bar_height,
                        facecolor=available_color,
                        edgecolor=available_color,
                        linewidth=0.35,
                        zorder=2,
                    )
                )
        axis.text(
            -duration_minutes * 0.014,
            y,
            "Sensor overall" if label == "sensor_overall" else label,
            ha="right",
            va="center",
            fontsize=7.2,
            fontweight="bold" if label == "sensor_overall" else "normal",
            color="#1F2933",
        )

    axis.set_xlim(0.0, duration_minutes)
    axis.set_ylim(-0.7, len(rows) - 0.3)
    axis.set_yticks([])
    axis.set_xlabel("Time from cycle start [min]", fontsize=7.5, labelpad=7)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=4))
    axis.tick_params(axis="x", labelsize=7.0, width=0.55, length=3, pad=2)
    axis.grid(axis="x", color="#D9DEE5", linewidth=0.35, alpha=0.55)
    axis.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_linewidth(0.6)
    axis.set_title(
        str(cycle_name),
        loc="left",
        fontsize=8.5,
        fontweight="bold",
        color="#17212B",
        pad=24,
    )
    axis.legend(
        handles=[
            Patch(
                facecolor=available_color,
                edgecolor=available_color,
                label="Available",
            ),
            Patch(
                facecolor=missing_face,
                edgecolor=missing_edge,
                hatch="///",
                label="Missing",
            ),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.10),
        ncol=2,
        frameon=False,
        fontsize=7.0,
        handlelength=1.5,
        columnspacing=1.3,
        borderaxespad=0.0,
    )
    figure.subplots_adjust(left=0.22, right=0.985, bottom=0.22, top=0.78)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    plt.close(figure)
