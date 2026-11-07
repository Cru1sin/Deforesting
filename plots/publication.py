"""Shared publication style, figure export, and reusable measurement panels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Rectangle

from plots.cop import (
    COP_COLOR,
    _plot_supported_curve,
)

_PANELS = (
    (("compressor_frequency", "compressor_frequency_setpoint"), "Compressor frequency [Hz]"),
    (("pressure_difference",), "Pc − Pe [MPa]"),
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
        (
            "condensing_temperature",
            "plate_heat_exchanger_inlet_temperature",
            "water_temperature_setpoint",
        ),
        "Condensing / inlet temperature [degC]",
    ),
    (
        ("ambient_temperature", "coil_temperature", "evaporating_temperature"),
        "Temperature [degC]",
    ),
)
_COLORS = {
    "visual_frost_score": COP_COLOR,
    "slingshot_grade": "#222222",
    "compressor_frequency": "#0072B2",
    "compressor_frequency_setpoint": "#7A9CC6",
    "heating_capacity": "#D55E00",
    "evaporator_capacity": "#B24C63",
    "compressor_power": "#4D4D4D",
    "power_total": "#7884B4",
    "defrost_event_electricity_kwh": "#B08342",
    "cop": "#4C8A83",
    "water_cop": "#657BA5",
    "ambient_temperature": "#374151",
    "water_in_temperature": "#5E8CA8",
    "water_out_temperature": "#C98B69",
    "condensing_temperature": "#D55E00",
    "plate_heat_exchanger_inlet_temperature": "#009E73",
    "pressure_difference": "#7B2CBF",
    "evaporating_temperature": "#56B4E9",
    "coil_temperature": "#7B2CBF",
    "water_temperature_setpoint": "#6B7280",
}


_STAGE_COLORS = {
    "recovery": "#78A6BC",
    "cold_start": "#78A6BC",
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


def water_side_heating_kw(frame: pd.DataFrame) -> pd.Series:
    """Return raw water-side heating capacity in kW."""
    return (
        1.161
        * pd.to_numeric(frame["water_flow"], errors="coerce")
        * (
            pd.to_numeric(frame["water_out_temperature"], errors="coerce")
            - pd.to_numeric(frame["water_in_temperature"], errors="coerce")
        )
    )


def save_publication(
    fig, output, name, caption, *, formats=("png", "pdf", "svg"), layout=True, dpi=250
):
    """Export a quantitative figure with its evidence caption and editable text."""
    if caption:
        fig.text(0.02, 0.015, caption, fontsize=7, va="bottom")
    if layout:
        fig.tight_layout(rect=(0, 0.1, 1, 0.98))
    for ext in formats:
        fig.savefig(output / f"{name}.{ext}", dpi=dpi, facecolor="white")
    plt.close(fig)


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
) -> None:
    for channel in channels:
        values = _observed_values(frame, channel)
        if values.notna().any():
            if (
                channel == "defrost_event_electricity_kwh"
                and "defrost_event_electricity_in_training_domain" in frame
            ):
                available = frame.defrost_event_electricity_prediction_available.eq(True)
                _plot_supported_curve(
                    axis,
                    minutes,
                    values,
                    available,
                    color=_COLORS[channel],
                    label=_display_label(channel),
                    interpolate=False,
                    extrapolated=available
                    & ~frame.defrost_event_electricity_in_training_domain.eq(True),
                    missing_spans=frame.attrs.get("original_missing_spans", ()),
                )
                continue
            axis.plot(
                minutes,
                values,
                color=_COLORS[channel],
                linestyle="--" if channel.endswith("setpoint") else "-",
                linewidth=1.2,
                label=_display_label(channel),
            )
    _shade_cycle_stages(axis, stage_spans, missing_spans)
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
    if channel == "visual_frost_score":
        return "结霜程度 reference"
    if channel == "slingshot_grade":
        return "Slingshot 霜层轨迹等级"
    if channel == "plate_heat_exchanger_inlet_temperature":
        return "Plate HX Inlet Temperature"
    if channel == "cop":
        return "Refrigerant-side COP"
    if channel == "pressure_difference":
        return "Pc − Pe"
    if channel == "defrost_event_electricity_kwh":
        return "Ridge preparation + defrost electricity"
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
    if channel == "pressure_difference":
        return _observed_values(frame, "condensing_pressure") - _observed_values(
            frame, "evaporating_pressure"
        )
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
