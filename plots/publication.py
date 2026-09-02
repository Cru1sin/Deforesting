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

from cost.energy_models import water_side_heating_kw
from dataloader.images import RGB_PANEL_MAX_OFFSET

_PANELS = (
    (("compressor_frequency", "compressor_frequency_setpoint"), "Compressor frequency [Hz]"),
    (
        ("heating_capacity", "evaporator_capacity", "compressor_power", "power_total"),
        "Capacity / power [kW]",
    ),
    (("cop", "water_cop"), "COP [-]"),
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
    "water_cop": "#0072B2",
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

    for axis, (channels, label) in zip(axes[1 : 1 + len(_PANELS)], _PANELS, strict=True):
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


def match_decision_rgb_images(
    metadata: pd.DataFrame,
    images: pd.DataFrame,
    targets: Mapping[str, object],
    *,
    max_offset: pd.Timedelta = RGB_PANEL_MAX_OFFSET,
) -> pd.DataFrame:
    """Match the nearest physical front frame to each causal decision time.

    Metadata is kept separate from the physically materialized images so that a
    missing local file is reported instead of silently being treated as missing
    metadata.  The same two-minute limit as the existing RGB panel is used.
    """
    metadata = metadata.loc[
        metadata.get("camera_role", pd.Series(dtype=str)).astype(str).eq("front")
    ].copy()
    if not metadata.empty:
        metadata["image_time"] = pd.to_datetime(metadata["image_time"], errors="coerce")
        metadata = metadata.dropna(subset=["image_time"]).sort_values(
            ["image_time", "file_name"], kind="stable"
        )
    physical = images.loc[
        images.get("camera_role", pd.Series(dtype=str)).astype(str).eq("front")
    ].copy()
    physical_by_name = {
        str(row["file_name"]): Path(cast(Any, row["path"]))
        for _, row in physical.iterrows()
        if "file_name" in row and "path" in row
    }
    rows: list[dict[str, object]] = []
    for target_type in ("rb", "optimal"):
        raw_target = targets.get(target_type)
        target_time = pd.to_datetime(raw_target, errors="coerce")
        target_time = pd.NaT if pd.isna(target_time) else pd.Timestamp(target_time)
        if pd.isna(target_time):
            rows.append(
                {
                    "target_type": target_type,
                    "target_time": pd.NaT,
                    "image_time": pd.NaT,
                    "offset_seconds": np.nan,
                    "file_name": "",
                    "image_path": "",
                    "available": False,
                    "status": ("rb_right_censored" if target_type == "rb" else "no_valid_optimal"),
                }
            )
            continue
        if metadata.empty:
            rows.append(
                {
                    "target_type": target_type,
                    "target_time": target_time,
                    "image_time": pd.NaT,
                    "offset_seconds": np.nan,
                    "file_name": "",
                    "image_path": "",
                    "available": False,
                    "status": "front_metadata_missing",
                }
            )
            continue
        distance = (metadata["image_time"] - target_time).abs()
        nearest = metadata.loc[distance.idxmin()]
        image_time = pd.Timestamp(nearest["image_time"])
        offset_seconds = float(abs((image_time - target_time).total_seconds()))
        file_name = str(nearest.get("file_name", ""))
        path = physical_by_name.get(file_name)
        if offset_seconds > max_offset.total_seconds():
            status = "offset_exceeds_2min"
            available = False
            image_path = ""
        elif path is None or not path.is_file():
            status = "physical_image_missing"
            available = False
            image_path = ""
        else:
            status = "matched"
            available = True
            image_path = str(path)
        rows.append(
            {
                "target_type": target_type,
                "target_time": target_time,
                "image_time": image_time,
                "offset_seconds": offset_seconds,
                "file_name": file_name,
                "image_path": image_path,
                "available": available,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def cost_curve_optimal_time(
    cost_curve: pd.DataFrame,
    origin: pd.Timestamp,
    stage_spans: list[tuple[str, float, float]],
) -> pd.Timestamp | None:
    """Return the same eligible minimum used by ``_plot_cost_panel``."""
    curve = cost_curve.copy()
    if "candidate_time" not in curve:
        return None
    curve["candidate_time"] = pd.to_datetime(curve["candidate_time"], errors="coerce")
    curve["minutes"] = (curve["candidate_time"] - origin).dt.total_seconds() / 60.0
    metric = "inverse_cop" if "inverse_cop" in curve else "renewal_cost_kw"
    if metric not in curve:
        return None
    curve[metric] = pd.to_numeric(curve[metric], errors="coerce")
    frost_spans = [
        (left, right) for stage, left, right in stage_spans if stage == "frost_development"
    ]
    is_frost = pd.Series(False, index=curve.index)
    for left, right in frost_spans:
        is_frost |= curve["minutes"].ge(left) & curve["minutes"].le(right)
    curve = curve.loc[is_frost & curve["minutes"].notna()].copy()
    eligible = (
        curve["optimization_eligible"].fillna(False).astype(bool)
        if "optimization_eligible" in curve
        else pd.Series(True, index=curve.index)
    )
    curve.loc[~eligible, metric] = np.nan
    values = curve.loc[eligible, metric].dropna()
    if values.empty:
        return None
    return pd.Timestamp(curve.loc[values.idxmin(), "candidate_time"])


def render_decision_publication(
    cycle_frame: pd.DataFrame,
    cycle_record: Mapping[str, object],
    cost_curve: pd.DataFrame,
    decision_images: Mapping[str, Mapping[str, object]],
    output_path: Path,
    *,
    optimal_label: str = "Economic optimum",
    cost_label: str = "Cycle inverse COP [-]",
    full_candidate_domain: bool = False,
    display_metric: str | None = None,
    minimum_label: str = "Minimum",
    minimum_support_label: str | None = None,
) -> None:
    """Render two decision frames above the reusable COP, water and cost panels."""
    frame = cycle_frame.sort_values("timestamp", kind="stable").copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    origin = pd.Timestamp(frame["timestamp"].min())
    minutes = (frame["timestamp"] - origin).dt.total_seconds() / 60.0
    stage_spans = _stage_spans(frame, minutes)
    duration = max(float(minutes.max()), 1.0)
    boundaries = cycle_record.get("boundaries")
    boundaries = boundaries if isinstance(boundaries, Mapping) else cycle_record
    stable = pd.to_datetime(boundaries.get("stable_heating_start"), errors="coerce")
    stable = pd.NaT if pd.isna(stable) else pd.Timestamp(stable)
    baseline_start = pd.to_datetime(boundaries.get("baseline_start"), errors="coerce")
    baseline_end = pd.to_datetime(boundaries.get("baseline_end"), errors="coerce")
    baseline_left = (
        (pd.Timestamp(baseline_start) - origin).total_seconds() / 60.0
        if not pd.isna(baseline_start)
        else np.nan
    )
    baseline_right = (
        (pd.Timestamp(baseline_end) - origin).total_seconds() / 60.0
        if not pd.isna(baseline_end)
        else np.nan
    )

    figure = plt.figure(figsize=(7.2, 8.25), dpi=300)
    grid = figure.add_gridspec(
        4,
        2,
        height_ratios=[2.15, 0.92, 1.05, 1.08],
        hspace=0.62,
        wspace=0.08,
    )
    image_axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    image_specs = (("rb", "RB trigger"), ("optimal", optimal_label))
    for axis, (target_type, label) in zip(image_axes, image_specs, strict=True):
        info = decision_images.get(target_type, {})
        _plot_decision_image(axis, info, label, origin, stable)

    panel_axes = [
        figure.add_subplot(grid[1, :]),
        figure.add_subplot(grid[2, :]),
        figure.add_subplot(grid[3, :]),
    ]
    _plot_cycle_panel(
        panel_axes[0],
        frame,
        minutes,
        ("cop", "water_cop"),
        "COP [-]",
        stage_spans,
        [],
        baseline_left,
        baseline_right,
    )
    _plot_cycle_panel(
        panel_axes[1],
        frame,
        minutes,
        ("water_in_temperature", "water_out_temperature", "water_temperature_setpoint"),
        "Water temperature [degC]",
        stage_spans,
        [],
        baseline_left,
        baseline_right,
    )
    curve = cost_curve if cost_curve is not None else pd.DataFrame()
    _plot_cost_panel(
        panel_axes[2],
        curve,
        origin,
        stage_spans,
        cost_label=cost_label,
        full_candidate_domain=full_candidate_domain,
        display_metric=display_metric,
        minimum_label=minimum_label,
        minimum_support_label=minimum_support_label,
    )
    for axis in panel_axes:
        axis.set_xlim(0.0, duration)
    _plot_decision_markers(panel_axes, decision_images, origin, optimal_label)
    labels = panel_axes[2].get_legend_handles_labels()
    if labels[0]:
        panel_axes[2].legend(
            *labels,
            frameon=False,
            fontsize=6.5,
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncol=min(len(labels[0]), 4),
        )
    cycle_name = str(cycle_record.get("cycle_name", cycle_record.get("cycle_id", "Cycle")))
    figure.suptitle(cycle_name, x=0.08, ha="left", fontsize=10, fontweight="bold")
    panel_axes[-1].set_xlabel("Time from cycle start [min]", fontsize=8)
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.06, top=0.92)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_decision_image(
    axis: Any,
    info: Mapping[str, object],
    label: str,
    origin: pd.Timestamp,
    stable: pd.Timestamp,
) -> None:
    path_value = info.get("image_path", info.get("path", ""))
    path = Path(str(path_value)) if path_value else None
    available = bool(info.get("available")) and path is not None and path.is_file()
    axis.set_xticks([])
    axis.set_yticks([])
    if available:
        axis.imshow(np.rot90(plt.imread(path)), aspect="auto")
    else:
        axis.set_facecolor("#ECEFF1")
        status = str(info.get("status", "unavailable")).replace("_", " ")
        axis.text(
            0.5,
            0.5,
            f"RGB unavailable\n{status}",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#59636E",
        )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#2E7D5B" if label.startswith("RB") else "#E28E2C")
        spine.set_linewidth(1.0)
    image_time = pd.to_datetime(info.get("image_time"), errors="coerce")
    if available and not pd.isna(image_time):
        cycle_minutes = (pd.Timestamp(image_time) - origin).total_seconds() / 60.0
        stable_minutes = (
            (pd.Timestamp(image_time) - stable).total_seconds() / 60.0
            if not pd.isna(stable)
            else np.nan
        )
        stable_text = (
            f"stable +{stable_minutes:.1f} min" if np.isfinite(stable_minutes) else "stable n/a"
        )
        title = (
            f"{label}\n"
            f"image {pd.Timestamp(image_time):%H:%M:%S} · cycle +{cycle_minutes:.1f} min · "
            f"{stable_text} · Δ {float(info.get('offset_seconds', np.nan)):.1f} s"
        )
    else:
        title = (
            f"{label}\nRGB unavailable · {str(info.get('status', 'unavailable')).replace('_', ' ')}"
        )
    axis.set_title(title, fontsize=7.2, pad=5, loc="left")


def _plot_decision_markers(
    axes: list[Any],
    decision_images: Mapping[str, Mapping[str, object]],
    origin: pd.Timestamp,
    optimal_label: str,
) -> None:
    markers = (
        ("rb", "RB RGB frame", "#2E7D5B"),
        ("optimal", f"{optimal_label} RGB frame", "#E28E2C"),
    )
    for target_type, label, color in markers:
        info = decision_images.get(target_type, {})
        image_time = pd.to_datetime(info.get("image_time"), errors="coerce")
        if not bool(info.get("available")) or pd.isna(image_time):
            continue
        x = (pd.Timestamp(image_time) - origin).total_seconds() / 60.0
        for axis_index, axis in enumerate(axes):
            axis.axvline(
                x,
                color=color,
                linestyle=":",
                linewidth=0.9,
                label=label if axis_index == len(axes) - 1 else "_nolegend_",
                zorder=4,
            )


def _plot_cost_panel(  # noqa: C901
    axis: Any,
    cost_curve: pd.DataFrame,
    origin: pd.Timestamp,
    stage_spans: list[tuple[str, float, float]],
    *,
    cost_label: str = "Cycle inverse COP [-]",
    full_candidate_domain: bool = False,
    display_metric: str | None = None,
    minimum_label: str = "Minimum",
    minimum_support_label: str | None = None,
) -> None:
    """Plot empirical cost only where the cycle is in frost development."""
    curve = cost_curve.copy()
    curve["candidate_time"] = pd.to_datetime(curve["candidate_time"], errors="coerce")
    curve["minutes"] = (curve["candidate_time"] - origin).dt.total_seconds() / 60.0
    metric = "inverse_cop" if "inverse_cop" in curve else "renewal_cost_kw"
    curve[metric] = pd.to_numeric(curve[metric], errors="coerce")
    frost_spans = [
        (left, right) for stage, left, right in stage_spans if stage == "frost_development"
    ]
    is_frost = pd.Series(False, index=curve.index)
    for left, right in frost_spans:
        is_frost |= curve["minutes"].ge(left) & curve["minutes"].lt(right)
    curve = curve.loc[
        curve["minutes"].notna() & (True if full_candidate_domain else is_frost)
    ].copy()
    curve["_raw_metric"] = curve[metric]
    if display_metric is not None:
        if display_metric not in curve:
            raise ValueError(f"missing display-only metric: {display_metric}")
        curve[display_metric] = pd.to_numeric(curve[display_metric], errors="coerce")
    eligible = (
        curve["optimization_eligible"].fillna(False).astype(bool)
        if "optimization_eligible" in curve
        else pd.Series(True, index=curve.index)
    )
    pe_supported = (
        curve["pe_supported"].fillna(False).astype(bool)
        if "pe_supported" in curve
        else curve["support_status"].eq("supported")
        if "support_status" in curve
        else eligible.copy()
    )
    if "model_supported" in curve:
        support_state = curve["model_supported"].astype("boolean")
        model_supported = support_state.eq(True).fillna(False)
        outside_support = support_state.eq(False).fillna(False)
        outside_support_label = "Outside empirical-model support"
    else:
        support_state = None
        model_supported = pe_supported
        outside_support = ~model_supported
        outside_support_label = "Outside Pe support"
    integration_eligible = (
        curve["integration_eligible"].fillna(False).astype(bool)
        if "integration_eligible" in curve
        else eligible | ~pe_supported
    )
    insufficient_integration = pe_supported & ~integration_eligible
    interpolated_gap = (
        curve["candidate_in_interpolated_gap"].fillna(False).astype(bool)
        if "candidate_in_interpolated_gap" in curve
        else pd.Series(False, index=curve.index)
    )
    extrapolated_endpoint = (
        curve["candidate_in_extrapolated_endpoint"].fillna(False).astype(bool)
        if "candidate_in_extrapolated_endpoint" in curve
        else pd.Series(False, index=curve.index)
    )
    curve.loc[~eligible, metric] = np.nan

    _shade_cycle_stages(axis, stage_spans, [])
    if curve.loc[eligible, metric].dropna().empty:
        axis.text(
            0.5,
            0.5,
            "No optimization-eligible candidates",
            transform=axis.transAxes,
            ha="center",
        )
    else:
        axis.plot(
            curve["minutes"],
            curve[metric],
            color="#3775BA",
            linewidth=1.25,
            label="Cycle inverse COP" if metric == "inverse_cop" else "Empirical cost",
        )
        minimum_index = curve.loc[eligible, metric].idxmin()
        minimum_x = float(curve.loc[minimum_index, "minutes"])
        axis.axvline(
            minimum_x,
            color="#E28E2C",
            linewidth=1.05,
            label=minimum_label,
            zorder=4,
        )
        regret = (
            pd.to_numeric(curve["relative_regret"], errors="coerce")
            if "relative_regret" in curve
            else curve[metric] / curve[metric].min() - 1.0
        )
        near = eligible & regret.le(0.01)
        groups = near.ne(near.shift(fill_value=False)).cumsum()
        for segment_index, (_, segment) in enumerate(
            curve.loc[near].groupby(groups[near], sort=False), start=1
        ):
            axis.axvspan(
                float(segment["minutes"].min()),
                float(segment["minutes"].max()),
                color="#E28E2C",
                alpha=0.18,
                label=f"1% near-optimal segment {segment_index}",
                zorder=0.2,
            )
        if "minimum_location" in curve:
            location = str(curve.loc[minimum_index, "minimum_location"]).replace("_", " ")
        else:
            location = (
                "left boundary"
                if minimum_index == curve.index[0]
                else "right boundary"
                if minimum_index == curve.index[-1]
                else "interior"
            )
        if minimum_support_label is not None:
            support_text = minimum_support_label
        elif support_state is not None:
            minimum_support = support_state.loc[minimum_index]
            support_text = (
                "support unknown"
                if pd.isna(minimum_support)
                else "within support"
                if bool(minimum_support)
                else "extrapolated"
            )
        axis.text(
            0.01,
            0.95,
            (
                f"Minimum: {location} · "
                f"{support_text}"
                if minimum_support_label is not None or support_state is not None
                else f"Minimum: {location}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#4B5563",
        )
        decision_time = pd.to_datetime(
            curve.get("t_star", pd.Series(dtype=object)).iloc[:1], errors="coerce"
        )
        if not decision_time.empty and pd.notna(decision_time.iloc[0]):
            decision_index = (curve["candidate_time"] - decision_time.iloc[0]).abs().idxmin()
            if decision_index != minimum_index:
                axis.axvline(
                    float(curve.loc[decision_index, "minutes"]),
                    color="#B64A50",
                    linestyle="--",
                    linewidth=1.05,
                    label="Selected decision",
                    zorder=4,
                )
                status = str(curve.get("decision_status", pd.Series(["near-optimal"])).iloc[0])
                axis.text(
                    0.01,
                    0.87,
                    f"Decision: {status.replace('_', ' ')} · "
                    f"+{100 * float(regret.loc[decision_index]):.2f}%",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="#B64A50",
                )

    if display_metric is not None and curve[display_metric].notna().any():
        axis.plot(
            curve["minutes"],
            curve[display_metric],
            color="#A69AA8",
            linestyle="--",
            linewidth=0.8,
            marker=".",
            markersize=2.5,
            alpha=0.75,
            label="Unsupported model extension, display only",
        )

    _plot_cost_quality_markers(
        axis,
        curve,
        outside_support,
        outside_support_label,
        insufficient_integration,
        interpolated_gap,
        extrapolated_endpoint,
    )

    _plot_rb_trigger(axis, curve, origin, metric, eligible)

    preparation = (
        pd.to_datetime(curve["actual_preparation_time"], errors="coerce").dropna()
        if "actual_preparation_time" in curve
        else pd.Series(dtype="datetime64[ns]")
    )
    if not preparation.empty:
        axis.axvline(
            (pd.Timestamp(preparation.iloc[0]) - origin).total_seconds() / 60.0,
            color="#777777",
            linewidth=0.9,
            linestyle="--",
            label="Observed preparation",
        )
    else:
        defrost_starts = [left for stage, left, _ in stage_spans if stage == "defrost"]
        if defrost_starts:
            axis.axvline(
                defrost_starts[0],
                color="#777777",
                linewidth=0.9,
                linestyle="--",
                label="Observed defrost",
            )
    axis.set_ylabel(
        cost_label if metric == "inverse_cop" else "Renewal cost [kW-eq.]",
        fontsize=8,
    )
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


def _plot_cost_quality_markers(
    axis: Any,
    curve: pd.DataFrame,
    outside_support: pd.Series,
    outside_support_label: str,
    insufficient_integration: pd.Series,
    interpolated_gap: pd.Series,
    extrapolated_endpoint: pd.Series,
) -> None:
    valid_metric = curve["_raw_metric"].notna()
    marker_specs = (
        (
            outside_support,
            11,
            "x",
            {"linewidths": 0.65, "color": "#A7ADB3"},
            outside_support_label,
            3,
        ),
        (
            insufficient_integration,
            13,
            "+",
            {"linewidths": 0.75, "color": "#C66A00"},
            "Insufficient integration coverage",
            3,
        ),
        (
            interpolated_gap,
            16,
            "s",
            {"facecolors": "none", "edgecolors": "#9AA0A6", "linewidths": 0.65},
            "Internal-gap interpolation",
            2,
        ),
        (
            extrapolated_endpoint,
            18,
            "D",
            {"facecolors": "none", "edgecolors": "#76528F", "linewidths": 0.7},
            "Endpoint linear extrapolation",
            2,
        ),
    )
    for mask, size, marker, style, label, zorder in marker_specs:
        visible = mask & valid_metric
        if visible.any():
            x = curve.loc[visible, "minutes"]
            y = curve.loc[visible, "_raw_metric"]
        elif mask.any():
            # Keep the diagnostic label when the corresponding cost is NaN.
            x = []
            y = []
        else:
            continue
        axis.scatter(
            x,
            y,
            s=size,
            marker=marker,
            label=label,
            zorder=zorder,
            **style,
        )


def _plot_rb_trigger(
    axis: Any,
    curve: pd.DataFrame,
    origin: pd.Timestamp,
    metric: str,
    eligible: pd.Series,
) -> None:
    """Mark the recorded first RB trigger without changing its causal timestamp."""
    if "rb_status" not in curve or "t_RB" not in curve:
        return
    status = curve["rb_status"].dropna().astype(str)
    rb_times = pd.to_datetime(curve["t_RB"], errors="coerce").dropna()
    if status.empty or status.iloc[0] != "triggered" or rb_times.empty:
        return
    rb_time = pd.Timestamp(rb_times.iloc[0])
    axis.axvline(
        (rb_time - origin).total_seconds() / 60.0,
        color="#2E7D5B",
        linewidth=0.9,
        linestyle="--",
        label="RB trigger",
        zorder=2.5,
    )
    valid_cost = eligible & curve[metric].notna()
    if valid_cost.any():
        nearest = (curve.loc[valid_cost, "candidate_time"] - rb_time).abs().idxmin()
        if abs(curve.loc[nearest, "candidate_time"] - rb_time) > pd.Timedelta(minutes=0.51):
            return
        axis.scatter(
            curve.loc[nearest, "minutes"],
            curve.loc[nearest, metric],
            s=18,
            color="#2E7D5B",
            label="Nearest eligible cost",
            zorder=3.5,
        )


def _plot_stage_ribbon(axis: Any, spans: list[tuple[str, float, float]]) -> None:
    minimum_label_width = max((end for _stage, _start, end in spans), default=0.0) * 0.05
    for stage, start, end in spans:
        axis.axvspan(start, end, color=_STAGE_COLORS.get(stage, "#BDBDBD"))
        if end - start < minimum_label_width:
            continue
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
    if "cop" in channels:
        values = pd.concat([_observed_values(frame, channel) for channel in channels], axis=1)
        stage = frame.get("cycle_stage", pd.Series(index=frame.index, dtype="string"))
        frost = stage.astype("string").eq("frost_development")
        normal = (
            (
                values.where(frost, axis=0)
                if frost.any()
                else values.mask(stage.astype("string").eq("recovery"), axis=0)
            )
            .stack()
            .dropna()
        )
        if not normal.empty:
            lower = float(normal.min())
            upper = float(normal.max())
            padding = max((upper - lower) * 0.08, max(abs(lower), abs(upper)) * 0.03, 0.1)
            axis.set_ylim(lower - padding, upper + padding)


def _display_label(channel: str) -> str:
    if channel == "cop":
        return "Refrigerant-side COP"
    if channel == "water_cop":
        return "Water-side COP"
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
                "#76528F" if stage == "defrost_preparation" else _STAGE_COLORS.get(stage, "#D9DEE5")
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
    if channel == "water_cop":
        dependencies = (
            "water_flow",
            "water_in_temperature",
            "water_out_temperature",
            "power_total",
        )
        if not set(dependencies) <= set(frame):
            return pd.Series(np.nan, index=frame.index, dtype=float)
        observed = pd.concat(
            [_observed_values(frame, dependency) for dependency in dependencies], axis=1
        )
        power = observed.iloc[:, -1].where(observed.iloc[:, -1].gt(0))
        return water_side_heating_kw(frame).mask(observed.isna().any(axis=1)).div(power)
    if channel not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[channel], errors="coerce")
    imputed = frame.get(f"{channel}__imputed")
    if imputed is not None:
        values = values.mask(imputed.astype("boolean").fillna(False))
    return values.replace([np.inf, -np.inf], np.nan)


def _stage_spans(frame: pd.DataFrame, minutes: pd.Series[Any]) -> list[tuple[str, float, float]]:
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
