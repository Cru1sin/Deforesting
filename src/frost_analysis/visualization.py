"""Pure renderers for self-contained Dataset artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Rectangle

_PANELS = (
    (("compressor_frequency", "compressor_frequency_setpoint"), "Compressor frequency [Hz]"),
    (
        ("heating_capacity", "evaporator_capacity", "compressor_power", "power_total"),
        "Capacity / power [kW]",
    ),
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
    "evaporator_capacity": "#B24C63",
    "compressor_power": "#4D4D4D",
    "power_total": "#7884B4",
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
    "defrost_preparation": "#A78BBA",
    "defrost": "#70B184",
}
_RGB_PANEL_LABELS = (
    "Start",
    "Recovery End",
    "Frost 25%",
    "Frost 50%",
    "Frost 75%",
    "Defrost Start",
    "Defrost Mid",
    "End",
)
_RGB_PANEL_FONT = next(
    (
        family
        for family in ("Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans")
        if any(font.name == family for font in font_manager.fontManager.ttflist)
    ),
    "DejaVu Sans",
)
_RGB_PANEL_MAX_OFFSET = pd.Timedelta(minutes=2)


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
        has_frost
        and frost_start is not None
        and frost_end is not None
        and frost_end > frost_start
    )

    def slot(label: str, target: pd.Timestamp | None, enabled: bool = True) -> dict[str, object]:
        return {"label": label, "target_time": target if enabled else None, "enabled": enabled}

    frost_targets: list[pd.Timestamp | None]
    if frost_valid and frost_start is not None and frost_end is not None:
        frost_targets = [
            frost_start + fraction * (frost_end - frost_start)
            for fraction in (0.25, 0.5, 0.75)
        ]
    else:
        frost_targets = [None, None, None]
    return [
        slot(_RGB_PANEL_LABELS[0], start, start is not None),
        slot(_RGB_PANEL_LABELS[1], stable_start, has_recovery and stable_start is not None),
        *(
            slot(label, target, frost_valid)
            for label, target in zip(
                _RGB_PANEL_LABELS[2:5], frost_targets, strict=True
            )
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
                if distance.loc[nearest] <= _RGB_PANEL_MAX_OFFSET
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
    cost_curve: pd.DataFrame | None = None,
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
    has_cost = cost_curve is not None and not cost_curve.empty
    row_count = 1 + len(_PANELS) + bool(has_cost) + bool(humidity)
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
    _plot_availability_panel(
        axes[0],
        origin,
        stage_spans,
        sensor_intervals or {"available": [], "missing": []},
        rgb_intervals or {"available": [], "missing": []},
    )

    boundaries = cycle_record.get("boundaries")
    baseline = boundaries if isinstance(boundaries, Mapping) else cycle_record
    baseline_start = pd.to_datetime(str(baseline.get("baseline_start")), errors="coerce")
    baseline_end = pd.to_datetime(str(baseline.get("baseline_end")), errors="coerce")
    if not pd.isna(baseline_start) and not pd.isna(baseline_end):
        left = (cast(pd.Timestamp, baseline_start) - origin).total_seconds() / 60.0
        right = (cast(pd.Timestamp, baseline_end) - origin).total_seconds() / 60.0
    else:
        left = right = np.nan

    missing_spans = [
        (
            (pd.Timestamp(start) - origin).total_seconds() / 60.0,
            (pd.Timestamp(end) - origin).total_seconds() / 60.0,
        )
        for start, end in (sensor_intervals or {}).get("missing", [])
    ]

    for axis, (channels, label) in zip(
        axes[1 : 1 + len(_PANELS)], _PANELS, strict=True
    ):
        _plot_cycle_panel(
            axis,
            frame,
            minutes,
            channels,
            label,
            stage_spans,
            missing_spans,
            left,
            right,
        )

    if has_cost:
        _plot_cost_panel(
            axes[1 + len(_PANELS)],
            cast(pd.DataFrame, cost_curve),
            origin,
            stage_spans,
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
        if np.isfinite(left) and np.isfinite(right):
            axis.axvspan(left, right, color="#6B7280", alpha=0.12, zorder=0)
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


def _plot_cost_panel(
    axis: Any,
    cost_curve: pd.DataFrame,
    origin: pd.Timestamp,
    stage_spans: list[tuple[str, float, float]],
) -> None:
    """Plot empirical cost only where the cycle is in frost development."""
    curve = cost_curve.copy()
    curve["candidate_time"] = pd.to_datetime(curve["candidate_time"], errors="coerce")
    curve["minutes"] = (curve["candidate_time"] - origin).dt.total_seconds() / 60.0
    curve["renewal_cost_kw"] = pd.to_numeric(curve["renewal_cost_kw"], errors="coerce")
    frost_spans = [
        (left, right)
        for stage, left, right in stage_spans
        if stage == "frost_development"
    ]
    is_frost = pd.Series(False, index=curve.index)
    for left, right in frost_spans:
        is_frost |= curve["minutes"].ge(left) & curve["minutes"].lt(right)
    curve = curve.loc[is_frost].dropna(subset=["minutes", "renewal_cost_kw"])

    _shade_cycle_stages(axis, stage_spans, [])
    if curve.empty:
        axis.text(0.5, 0.5, "No frosting cost candidates", transform=axis.transAxes, ha="center")
    else:
        axis.plot(
            curve["minutes"],
            curve["renewal_cost_kw"],
            color="#3775BA",
            linewidth=1.25,
            label="Empirical cost",
        )
        minimum_index = curve["renewal_cost_kw"].idxmin()
        minimum_x = float(curve.loc[minimum_index, "minutes"])
        axis.axvline(minimum_x, color="#E28E2C", linewidth=1.05, label="Minimum")
        regret = (
            pd.to_numeric(curve["relative_regret"], errors="coerce")
            if "relative_regret" in curve
            else curve["renewal_cost_kw"] / curve["renewal_cost_kw"].min() - 1.0
        )
        near = curve.loc[regret.le(0.01)]
        if not near.empty:
            axis.axvspan(
                float(near["minutes"].min()),
                float(near["minutes"].max()),
                color="#E28E2C",
                alpha=0.18,
                label="1% window",
                zorder=0.2,
            )
        location = (
            "left boundary"
            if minimum_index == curve.index[0]
            else "right boundary"
            if minimum_index == curve.index[-1]
            else "interior"
        )
        axis.text(
            0.01,
            0.95,
            f"Minimum: {location}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#4B5563",
        )

    defrost_starts = [left for stage, left, _ in stage_spans if stage == "defrost"]
    if defrost_starts:
        axis.axvline(
            defrost_starts[0],
            color="#777777",
            linewidth=0.9,
            linestyle="--",
            label="Observed defrost",
        )
    axis.set_ylabel("Renewal cost [kW-eq.]", fontsize=8)
    axis.grid(axis="x", alpha=0.12)
    if axis.lines:
        handle_count = len(axis.get_legend_handles_labels()[0])
        axis.legend(
            frameon=False,
            fontsize=7,
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncol=min(handle_count, 4),
        )


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


def _plot_availability_panel(
    axis: Any,
    origin: pd.Timestamp,
    stage_spans: list[tuple[str, float, float]],
    sensor: Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
    rgb: Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
) -> None:
    duration = max((right for _stage, _left, right in stage_spans), default=1.0)
    for y, (label, values) in enumerate((("RGB", rgb), ("Sensor", sensor))):
        axis.add_patch(
            Rectangle(
                (0.0, y - 0.32),
                duration,
                0.64,
                facecolor="#F4F6F8",
                edgecolor="#AEB8C3",
                linewidth=0.45,
                hatch="///",
            )
        )
        for available_start, available_end in values.get("available", []):
            left = (pd.Timestamp(available_start) - origin).total_seconds() / 60.0
            right = (pd.Timestamp(available_end) - origin).total_seconds() / 60.0
            for stage, stage_left, stage_right in stage_spans:
                clipped_left = max(left, stage_left)
                clipped_right = min(right, stage_right)
                if clipped_right > clipped_left:
                    axis.add_patch(
                        Rectangle(
                            (clipped_left, y - 0.32),
                            clipped_right - clipped_left,
                            0.64,
                            facecolor=_STAGE_COLORS.get(stage, "#BDBDBD"),
                            edgecolor="none",
                        )
                    )
        axis.text(-duration * 0.015, y, label, ha="right", va="center", fontsize=7)
    for stage, left, right in stage_spans:
        label_x = (
            right - duration * 0.004
            if stage == "defrost_preparation"
            else left + duration * 0.004
            if stage == "defrost"
            else (left + right) / 2.0
        )
        alignment = (
            "right"
            if stage == "defrost_preparation"
            else "left"
            if stage == "defrost"
            else "center"
        )
        axis.text(
            label_x,
            1.47,
            _display_label(stage),
            ha=alignment,
            va="bottom",
            fontsize=7,
            color="#1F2933",
        )
    axis.set_xlim(0.0, duration)
    axis.set_ylim(-0.55, 1.78)
    axis.axis("off")


def _plot_cycle_panel(
    axis: Any,
    frame: pd.DataFrame,
    minutes: pd.Series[Any],
    channels: tuple[str, ...],
    label: str,
    stage_spans: list[tuple[str, float, float]],
    missing_spans: list[tuple[float, float]],
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
                label=_display_label(channel),
            )
    _shade_cycle_stages(axis, stage_spans, missing_spans)
    if np.isfinite(baseline_start) and np.isfinite(baseline_end):
        axis.axvspan(baseline_start, baseline_end, color="#6B7280", alpha=0.12, zorder=0)
    axis.set_ylabel(label, fontsize=8)
    axis.grid(axis="x", alpha=0.12)
    if axis.lines:
        axis.legend(
            frameon=False,
            fontsize=7,
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncol=len(axis.lines),
            columnspacing=1.2,
        )
    if channels == ("cop",):
        values = _observed_values(frame, "cop")
        stage = frame.get("cycle_stage", pd.Series(index=frame.index, dtype="string"))
        normal = values.mask(stage.astype("string").eq("recovery")).dropna()
        if not normal.empty:
            lower = float(normal.min())
            upper = float(normal.max())
            padding = max((upper - lower) * 0.08, max(abs(lower), abs(upper)) * 0.03, 0.1)
            axis.set_ylim(lower - padding, upper + padding)


def _display_label(channel: str) -> str:
    if channel == "environment_relative_humidity":
        return "Relative Humidity"
    return channel.replace("_", " ").title()


def _shade_cycle_stages(
    axis: Any,
    stage_spans: list[tuple[str, float, float]],
    missing_spans: list[tuple[float, float]],
) -> None:
    for stage, start, end in stage_spans:
        axis.axvspan(
            start,
            end,
            facecolor=(
                "#76528F"
                if stage == "defrost_preparation"
                else _STAGE_COLORS.get(stage, "#D9DEE5")
            ),
            alpha=0.20 if stage == "defrost_preparation" else 0.10,
            edgecolor="none",
            zorder=0,
        )
    for start, end in missing_spans:
        axis.axvspan(
            start,
            end,
            facecolor="#E5E7EB",
            edgecolor="#9CA3AF",
            alpha=0.46,
            hatch="////",
            linewidth=0.0,
            zorder=0.1,
        )


def _observed_values(frame: pd.DataFrame, channel: str) -> pd.Series[Any]:
    if channel not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[channel], errors="coerce")
    imputed = frame.get(f"{channel}__imputed")
    if imputed is not None:
        values = values.mask(imputed.astype("boolean").fillna(False))
    return values.replace([np.inf, -np.inf], np.nan)


def _stage_spans(
    frame: pd.DataFrame, minutes: pd.Series[Any]
) -> list[tuple[str, float, float]]:
    if "cycle_stage" not in frame or frame.empty:
        return []
    step = float(minutes.diff().dropna().median()) if len(minutes) > 1 else 0.0
    stages = frame["cycle_stage"].astype("string")
    changes = stages.ne(stages.shift()).fillna(True)
    groups = changes.cumsum()
    return [
        (
            str(group["cycle_stage"].iloc[0]),
            float(minutes.loc[group.index].iloc[0]),
            float(minutes.loc[group.index].iloc[-1] + step),
        )
        for _, group in frame.groupby(groups, sort=False)
        if pd.notna(group["cycle_stage"].iloc[0])
    ]
