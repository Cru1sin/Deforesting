"""Pure renderers for self-contained cycle Dataset artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from .dataset_loader import DatasetLoader


def render_cycle_publication(
    cycle_frame: pd.DataFrame,
    cycle_record: Mapping[str, object],
    output_path: Path,
) -> None:
    """Render a compact publication figure from Dataset-owned values only."""
    humidity_columns = [
        column
        for column in cycle_frame.columns
        if "humidity" in str(column).lower() and not str(column).endswith("__imputed")
    ]
    panels: list[tuple[str, tuple[str, ...]]] = [
        ("Compressor frequency [Hz]", ("compressor_frequency",)),
        ("Heating capacity [kW]", ("heating_capacity",)),
        ("COP [-]", ("cop",)),
        ("Water temperature [°C]", ("water_in_temperature", "water_out_temperature")),
        (
            "Temperature [°C]",
            ("ambient_temperature", "coil_temperature", "evaporating_temperature"),
        ),
    ]
    if humidity_columns:
        panels.append(("Relative humidity [%]", tuple(humidity_columns)))

    figure, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        figsize=(7.2, 1.45 * len(panels) + 1.0),
        dpi=220,
        squeeze=False,
    )
    flat_axes = list(axes[:, 0])
    timestamps = _timestamps(cycle_frame, cycle_record)
    for axis, (label, columns) in zip(flat_axes, panels, strict=True):
        plotted = False
        for column in columns:
            if column not in cycle_frame:
                continue
            values = pd.to_numeric(cycle_frame[column], errors="coerce")
            if len(values) != len(timestamps):
                continue
            axis.plot(timestamps, values, linewidth=0.9, label=str(column))
            plotted = True
        if not plotted:
            axis.text(
                0.01,
                0.5,
                "No observed values",
                transform=axis.transAxes,
                color="#666666",
                fontsize=8,
                va="center",
            )
        axis.set_ylabel(label, fontsize=8)
        axis.grid(axis="x", alpha=0.16)
        if plotted and len(columns) > 1:
            axis.legend(frameon=False, fontsize=7, loc="upper left", ncol=3)

    if timestamps:
        flat_axes[-1].set_xlabel("Time")
    status = str(cycle_record.get("status", cycle_record.get("cycle_status", "unknown")))
    cycle_name = str(cycle_record.get("cycle_name", "cycle"))
    figure.suptitle(f"{cycle_name} · {status}", fontsize=11, x=0.02, ha="left")
    figure.subplots_adjust(left=0.19, right=0.98, bottom=0.08, top=0.94, hspace=0.46)
    _save_figure(figure, output_path)


def _timestamps(frame: pd.DataFrame, record: Mapping[str, object]) -> list[pd.Timestamp]:
    if "timestamp" in frame:
        values = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
        return [pd.Timestamp(value) for value in values]
    starts = pd.to_datetime(
        pd.Series([record.get("start_time")], dtype="object"), errors="coerce"
    ).dropna()
    ends = pd.to_datetime(
        pd.Series([record.get("end_time")], dtype="object"), errors="coerce"
    ).dropna()
    if not starts.empty:
        end = ends.iloc[0] if not ends.empty else starts.iloc[0]
        return [pd.Timestamp(starts.iloc[0]), pd.Timestamp(end)]
    return []


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=figure.dpi, bbox_inches="tight")
    finally:
        plt.close(figure)


def generate_cycle_publication(loader: DatasetLoader, cycle_name: str) -> Path:
    """Load one cycle through DatasetLoader and write its publication figure."""
    from .dataset_loader import DatasetLoader
    from .dataset_manifest import refresh_cycle_asset_hashes

    if not isinstance(loader, DatasetLoader):
        raise TypeError("generate_cycle_publication requires DatasetLoader")
    path = loader.publication_path(cycle_name)
    render_cycle_publication(
        loader.load_cycle(cycle_name), loader.get_cycle_record(cycle_name), path
    )
    refresh_cycle_asset_hashes(loader.dataset_root, cycle_name)
    return path


def generate_rgb_coverage(loader: DatasetLoader, cycle_name: str) -> Path:
    """Load one cycle and its current role folders through DatasetLoader."""
    from .dataset_coverage import render_rgb_coverage
    from .dataset_loader import DatasetLoader
    from .dataset_manifest import refresh_cycle_asset_hashes

    if not isinstance(loader, DatasetLoader):
        raise TypeError("generate_rgb_coverage requires DatasetLoader")
    path = loader.rgb_coverage_path(cycle_name)
    render_rgb_coverage(
        loader.load_cycle(cycle_name),
        loader.load_cycle_images(cycle_name),
        loader.get_cycle_record(cycle_name),
        path,
    )
    refresh_cycle_asset_hashes(loader.dataset_root, cycle_name)
    return path
