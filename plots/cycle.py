"""Cycle measurements, stage boundaries, and RGB panels."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from dataset_tools.local_images import RGB_PANEL_MAX_OFFSET
from plots.publication import (
    _PANELS,
    _RGB_PANEL_FONT,
    _RGB_PANEL_LABELS,
    _STAGE_COLORS,
    _display_label,
    _observed_values,
    _plot_availability_panel,
    _plot_cycle_panel,
    _plot_stage_ribbon,
    _shade_cycle_stages,
    _stage_spans,
)

matplotlib.use("Agg")


def build_rgb_panel_targets(
    cycle_record: Mapping[str, object], cycle_frame: pd.DataFrame
) -> list[dict[str, object]]:
    """Build the fixed eight review slots from existing cycle facts."""
    raw_timestamps = (
        cycle_frame["timestamp"] if "timestamp" in cycle_frame else pd.Series(dtype=object)
    )
    timestamps = pd.to_datetime(raw_timestamps, errors="coerce").dropna()
    boundaries = cycle_record.get("boundaries")
    values = boundaries if isinstance(boundaries, Mapping) else {}

    def boundary(name: str, fallback: pd.Timestamp | None = None) -> pd.Timestamp | None:
        parsed = pd.to_datetime(cast(Any, values.get(name)), errors="coerce")
        return fallback if pd.isna(parsed) else pd.Timestamp(parsed)

    first = pd.Timestamp(timestamps.min()) if not timestamps.empty else None
    last = pd.Timestamp(timestamps.max()) if not timestamps.empty else None
    start = boundary("start_time", first)
    end = boundary("end_time", last)
    heating_start = boundary("heating_start")
    stable_start = boundary("stable_heating_start")
    defrost_start = boundary("defrost_start")
    preparation_start = boundary("defrost_preparation_start")
    defrost_end = boundary("defrost_end")
    stages = set(cycle_frame.get("cycle_stage", pd.Series(dtype="string")).dropna().astype(str))
    has_recovery = "recovery" in stages
    has_frost = "frost_development" in stages
    has_defrost = "defrost" in stages
    frost_start = stable_start or heating_start or start
    frost_end = preparation_start or (
        defrost_start if has_defrost and defrost_start is not None else end
    )
    frost_valid = (
        has_frost and frost_start is not None and frost_end is not None and frost_end > frost_start
    )

    def slot(label: str, target: pd.Timestamp | None, enabled: bool = True) -> dict[str, object]:
        return {"label": label, "target_time": target if enabled else None, "enabled": enabled}

    frost_targets: list[pd.Timestamp | None]
    if frost_valid and frost_start is not None and frost_end is not None:
        frost_targets = [
            frost_start + fraction * (frost_end - frost_start) for fraction in (0.25, 0.5, 0.75)
        ]
    else:
        frost_targets = [None, None, None]
    return [
        slot(_RGB_PANEL_LABELS[0], start, start is not None),
        slot(_RGB_PANEL_LABELS[1], stable_start, has_recovery and stable_start is not None),
        *(
            slot(label, target, frost_valid)
            for label, target in zip(_RGB_PANEL_LABELS[2:5], frost_targets, strict=True)
        ),
        slot(_RGB_PANEL_LABELS[5], defrost_start, has_defrost and defrost_start is not None),
        slot(
            _RGB_PANEL_LABELS[6],
            defrost_start + 0.5 * (defrost_end - defrost_start)
            if defrost_start is not None and defrost_end is not None
            else None,
            has_defrost and defrost_start is not None and defrost_end is not None,
        ),
        slot(_RGB_PANEL_LABELS[7], end, end is not None),
    ]


def select_rgb_panel_cells(
    images: pd.DataFrame,
    targets: list[dict[str, object]],
    camera_roles: list[str],
) -> dict[str, list[Path | None]]:
    """Select the nearest image for every enabled camera/slot pair."""
    result: dict[str, list[Path | None]] = {}
    for camera_role in camera_roles:
        camera = images.loc[images["camera_role"].astype(str).eq(camera_role)].copy()
        raw_times = camera["image_time"] if "image_time" in camera else pd.Series(dtype=object)
        camera["image_time"] = pd.to_datetime(raw_times, errors="coerce")
        camera = camera.dropna(subset=["image_time"])
        cells: list[Path | None] = []
        for target in targets:
            target_time = target.get("target_time")
            if not bool(target.get("enabled")) or target_time is None or camera.empty:
                cells.append(None)
                continue
            distance = (camera["image_time"] - pd.Timestamp(cast(Any, target_time))).abs()
            nearest = distance.idxmin()
            cells.append(
                Path(cast(Any, camera.loc[nearest, "path"]))
                if distance.loc[nearest] <= RGB_PANEL_MAX_OFFSET
                else None
            )
        result[camera_role] = cells
    return result


def render_rgb_panel(
    cycle_record: Mapping[str, object],
    cycle_frame: pd.DataFrame,
    images: pd.DataFrame,
    intervals: Mapping[str, Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]]],
    camera_roles: list[str] | tuple[str, ...],
    output_path: Path,
) -> None:
    """Render the per-cycle RGB review panel from Dataset-resident facts."""
    targets = build_rgb_panel_targets(cycle_record, cycle_frame)
    role_order = {role: index for index, role in enumerate(("front", "left", "right", "top"))}
    roles = sorted(
        set(camera_roles) | set(images.get("camera_role", pd.Series(dtype=str)).astype(str)),
        key=lambda role: (
            role_order.get(role.split("_", 1)[0], len(role_order)),
            role,
        ),
    )
    cells = select_rgb_panel_cells(images, targets, roles)
    frame = cycle_frame.sort_values("timestamp", kind="stable").copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    start = pd.Timestamp(frame["timestamp"].min())
    time_steps = frame["timestamp"].sort_values().diff().dropna().dt.total_seconds()
    positive_steps = time_steps.loc[time_steps > 0]
    step = float(positive_steps.median()) if not positive_steps.empty else 1.0
    end = pd.Timestamp(frame["timestamp"].max()) + pd.Timedelta(seconds=step)
    duration = max((end - start).total_seconds() / 60.0, 1.0)
    minutes = (frame["timestamp"] - start).dt.total_seconds() / 60.0
    stage_spans = _stage_spans(frame, minutes)
    row_count = 1 + 2 * max(1, len(roles))
    heights = [0.16] + sum(([1.0, 0.13] for _ in range(max(1, len(roles)))), [])
    figure = plt.figure(figsize=(14.4, max(3.0, 2.15 * len(roles) + 0.8)), dpi=300)
    grid = figure.add_gridspec(row_count, 8, height_ratios=heights, hspace=0.08, wspace=0.04)
    stage_axis = figure.add_subplot(grid[0, :])
    _plot_stage_ribbon(stage_axis, stage_spans)
    stage_axis.set_xlim(0.0, duration)

    for column, target in enumerate(targets):
        stage_axis.text(
            (column + 0.5) / 8,
            1.65,
            str(target["label"]),
            transform=stage_axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#1F2933",
        )

    for row, role in enumerate(roles):
        for column, path in enumerate(cells[role]):
            axis = figure.add_subplot(grid[1 + row * 2, column])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#D5DBE1")
                spine.set_linewidth(0.5)
            if column == 0:
                axis.set_ylabel(
                    role,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=8,
                    labelpad=12,
                    fontfamily=_RGB_PANEL_FONT,
                )
            if path is not None and path.is_file():
                axis.imshow(np.rot90(plt.imread(path)), aspect="auto")

        coverage_axis = figure.add_subplot(grid[2 + row * 2, :])
        for stage, left, right in stage_spans:
            coverage_axis.axvspan(
                left,
                right,
                color=_STAGE_COLORS.get(stage, "#BDBDBD"),
                alpha=0.22,
            )
        coverage_axis.add_patch(
            Rectangle(
                (0.0, 0.1),
                duration,
                0.8,
                facecolor="none",
                edgecolor="#AEB8C3",
                linewidth=0.45,
                hatch="///",
            )
        )
        role_intervals = intervals.get(role, {})
        for available_start, available_end in role_intervals.get("available", []):
            available_left = (pd.Timestamp(available_start) - start).total_seconds() / 60.0
            available_right = (pd.Timestamp(available_end) - start).total_seconds() / 60.0
            for stage, stage_left, stage_right in stage_spans:
                left = max(available_left, stage_left)
                right = min(available_right, stage_right)
                coverage_axis.add_patch(
                    Rectangle(
                        (left, 0.1),
                        max(0.0, right - left),
                        0.8,
                        facecolor=_STAGE_COLORS.get(stage, "#BDBDBD"),
                        edgecolor="none",
                    )
                )
        coverage_axis.set_xlim(0.0, duration)
        coverage_axis.set_ylim(0.0, 1.0)
        coverage_axis.axis("off")

    figure.suptitle(
        f"{cycle_record.get('cycle_name', 'Cycle')}   |   {cycle_record.get('status', '')}",
        x=0.105,
        y=0.99,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.105, right=0.99, bottom=0.03, top=0.91)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(figure)


def render_cycle_publication(
    cycle_frame: pd.DataFrame,
    cycle_record: Mapping[str, object],
    output_path: Path,
    *,
    sensor_intervals: Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]] | None = None,
    rgb_intervals: Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]] | None = None,
) -> None:
    """Render the Dataset's cycle-level scientific overview."""
    frame = cycle_frame.sort_values("timestamp", kind="stable").copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    bounds = cycle_record.get("boundaries", {})
    origin = frame["timestamp"].min()
    if bounds.get("heating_origin") in {"cold_start", "not_started"}:
        origin = pd.to_datetime(bounds.get("heating_start"))
        frame = frame.loc[frame.timestamp.ge(origin)].copy()
        if frame.empty:
            return

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
        figsize=(7.2, 10.8 + 1.25 * (len(_PANELS) - 5 + bool(humidity))),
        dpi=300,
        sharex=True,
        gridspec_kw={
            "height_ratios": [0.42, *([1.0] * (row_count - 1))],
            "hspace": 0.58,
        },
    )
    stage_spans = _stage_spans(frame, minutes)

    boundaries = cycle_record.get("boundaries")
    boundaries = boundaries if isinstance(boundaries, Mapping) else cycle_record
    if "recovery_status" in cycle_record:
        boundary = pd.to_datetime(boundaries.get("stable_heating_start"))
        if pd.notna(boundary):
            transition = (boundary - origin).total_seconds() / 60
            stage_spans = [
                (
                    stage,
                    transition if stage == "frost_development" else start,
                    transition if stage == "recovery" else end,
                )
                for stage, start, end in stage_spans
            ]
    if bounds.get("heating_origin") == "cold_start":
        stage_spans = [
            (
                "cold_start" if stage == "recovery" else stage,
                0.0 if stage == "recovery" else start,
                end,
            )
            for stage, start, end in stage_spans
        ]
    _plot_availability_panel(
        axes[0],
        origin,
        stage_spans,
        sensor_intervals or {"available": [], "missing": []},
        rgb_intervals or {"available": [], "missing": []},
    )

    missing_spans = [
        (
            (pd.Timestamp(start) - origin).total_seconds() / 60.0,
            (pd.Timestamp(end) - origin).total_seconds() / 60.0,
        )
        for start, end in (sensor_intervals or {}).get("missing", [])
    ]

    for axis, (channels, label) in zip(axes[1 : 1 + len(_PANELS)], _PANELS, strict=True):
        _plot_cycle_panel(
            axis,
            frame,
            minutes,
            channels,
            label,
            stage_spans,
            missing_spans,
        )

    if humidity:
        axis = axes[-1]
        for channel in humidity:
            axis.plot(
                minutes,
                _observed_values(frame, channel),
                linewidth=1.2,
                label=_display_label(channel),
            )
        _shade_cycle_stages(axis, stage_spans, missing_spans)
        axis.set_ylabel("Relative humidity [%]", fontsize=8)
        axis.grid(axis="x", alpha=0.12)
        axis.legend(
            frameon=False,
            fontsize=7,
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncol=len(humidity),
        )

    axes[-1].set_xlabel("Time from cycle start [min]", fontsize=8)
    recovery_note = (
        " | Recovery boundary unidentified"
        if cycle_record.get("recovery_status") not in (None, "identified_offline")
        else ""
    )
    from dataset_tools.cycle_metadata import cycle_operating_condition

    figure.suptitle(
        f"{cycle_record.get('cycle_name', cycle_record.get('cycle_id', 'Cycle'))} | "
        f"{cycle_operating_condition(frame)} | "
        f"{cycle_record.get('status', cycle_record.get('cycle_status', ''))}{recovery_note}",
        x=0.12,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.20, right=0.98, bottom=0.06, top=0.94)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_recovery_audit(frame, record, traces, output_path):
    """Show each offline boundary against the same observed startup signals."""
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    origin = pd.Timestamp(record["boundaries"].get("heating_start") or frame.timestamp.min())
    minute = (frame.timestamp - origin).dt.total_seconds() / 60
    frame["pressure_difference"] = frame.condensing_pressure - frame.evaporating_pressure
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True, dpi=150)
    panels = (
        (("compressor_frequency_setpoint", "compressor_frequency"), "Frequency [Hz]"),
        (("pressure_difference",), "Pc − Pe [MPa]"),
        (("condensing_temperature", "water_temperature_setpoint"), "Temperature [degC]"),
    )
    for axis, (channels, label) in zip(axes[:3], panels, strict=True):
        _plot_cycle_panel(axis, frame, minute, channels, label, [], [])
    colors = ("#0072B2", "#D55E00", "#7B2CBF", "#009E73")
    for index, ((rule, trace), color) in enumerate(
        zip(traces.items(), colors[: len(traces)], strict=True)
    ):
        x = (trace.timestamp - origin).dt.total_seconds() / 60
        axes[-1].plot(x, trace.normal_heating.astype(float) * 0.65 + index, color=color, label=rule)
        confirmed = trace.loc[trace.normal_heating, "timestamp"]
        if len(confirmed):
            t = (confirmed.iloc[0] - origin).total_seconds() / 60
            for axis in axes[:3]:
                axis.axvline(t, color=color, lw=0.8, ls="--")
    axes[-1].set_yticks(range(4), list(traces), fontsize=7)
    axes[-1].set_xlabel("Time from heating start [min]")
    axes[-1].set_xlim(0, min(60, max(float(minute.max()), 1)))
    fig.suptitle(f"{record['cycle_name']} | ramp → regulation | offline boundaries", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_recovery_overviews(loader, boundaries, output):
    """Review every detected startup together, including delayed and missing transitions."""
    names = list(boundaries.cycle_name.unique())
    for first in range(0, len(names), 18):
        fig, axes = plt.subplots(6, 3, figsize=(15, 15), dpi=160)
        for axis, name in zip(axes.flat, names[first : first + 18], strict=False):
            frame = loader.load_cycle_original(name)
            frame.timestamp = pd.to_datetime(frame.timestamp)
            rows = boundaries.loc[boundaries.cycle_name.eq(name)]
            start = pd.Timestamp(rows.heating_start.iloc[0])
            end = pd.to_datetime(rows.stable_heating_start).max()
            xmax = max(25.0, (end - start).total_seconds() / 60 + 4) if pd.notna(end) else 40.0
            x = (frame.timestamp - start).dt.total_seconds() / 60
            axis.plot(x, frame.compressor_frequency_setpoint, color="#0072B2", lw=0.8)
            pressure_axis = axis.twinx()
            pressure_axis.plot(
                x,
                _observed_values(frame, "pressure_difference"),
                color="#7B2CBF",
                lw=0.6,
                alpha=0.5,
            )
            pressure_axis.tick_params(axis="y", labelsize=5, colors="#7B2CBF")
            colors = dict(
                zip(
                    (
                        "frequency-setpoint",
                        "frequency-actual",
                        "pressure-difference",
                        "frequency-pressure",
                    ),
                    ("#0072B2", "#D55E00", "#7B2CBF", "#009E73"),
                    strict=True,
                )
            )
            for _, row in rows.iterrows():
                color = colors[row.recovery_rule]
                t = pd.to_datetime(row.stable_heating_start)
                if pd.notna(t):
                    axis.axvline((t - start).total_seconds() / 60, color=color, lw=0.7, ls="--")
            axis.set_xlim(0, min(xmax, max(x.max(), 1)))
            axis.set_title(name, fontsize=9)
            axis.tick_params(labelsize=7)
        for axis in list(axes.flat)[len(names[first : first + 18]) :]:
            axis.set_visible(False)
        fig.suptitle(
            "All-cycle review | frequency [Hz, left], Pc−Pe [MPa, right] | "
            "blue=setpoint, orange=actual, purple=pressure, green=joint"
        )
        fig.tight_layout()
        fig.savefig(output / f"overview_{first // 18 + 1:02d}.png", bbox_inches="tight")
        plt.close(fig)
