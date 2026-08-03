"""RGB/sensor temporal coverage summaries and rendering."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def render_rgb_coverage(
    cycle_frame: pd.DataFrame,
    cycle_images: pd.DataFrame,
    cycle_record: Mapping[str, object],
    output_path: Path,
) -> None:
    """Render sensor coverage and one row for every current camera role."""
    roles = sorted(
        str(value)
        for value in cycle_images.get("camera_role", pd.Series(dtype=str)).dropna().unique()
    )
    timestamps = _timestamps(cycle_frame, cycle_record)
    figure, axes = plt.subplots(
        max(1, len(roles) + 1),
        1,
        figsize=(8.0, max(2.2, 0.45 * (len(roles) + 1) + 1.2)),
        sharex=True,
        dpi=220,
        squeeze=False,
    )
    flat_axes = list(axes[:, 0])
    _draw_sensor_row(flat_axes[0], timestamps)
    flat_axes[0].set_ylabel("sensor", fontsize=8)
    for axis, role in zip(flat_axes[1:], roles, strict=True):
        role_times = pd.to_datetime(
            cycle_images.loc[cycle_images["camera_role"].eq(role), "image_time"],
            errors="coerce",
        ).dropna()
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_ylabel(role, fontsize=8, rotation=0, ha="right", va="center")
        if not role_times.empty:
            axis.scatter(role_times, [0.5] * len(role_times), s=8, color="#0072B2")
        axis.grid(axis="x", alpha=0.16)
    flat_axes[-1].set_xlabel("Time")
    cycle_name = str(cycle_record.get("cycle_name", "cycle"))
    figure.suptitle(f"{cycle_name} · sensor/RGB coverage", fontsize=10, x=0.02, ha="left")
    figure.subplots_adjust(left=0.20, right=0.98, bottom=0.12, top=0.90, hspace=0.28)
    path = output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=figure.dpi, bbox_inches="tight")
    finally:
        plt.close(figure)


def image_summary(
    cycle_frame: pd.DataFrame,
    cycle_images: pd.DataFrame,
) -> dict[str, object]:
    """Return small factual image coverage metadata for one cycle."""
    by_role: dict[str, dict[str, object]] = {}
    timestamp_values = cycle_frame.get(
        "timestamp", pd.Series(dtype="datetime64[ns]")
    )
    all_times = pd.to_datetime(timestamp_values, errors="coerce").dropna()
    groups: Any = (
        cycle_images.groupby("camera_role", sort=True) if not cycle_images.empty else []
    )
    for role, group in groups:
        times = pd.to_datetime(group["image_time"], errors="coerce").dropna().sort_values()
        by_role[str(role)] = {
            "image_count": int(len(times)),
            "coverage_ratio": _coverage_ratio(all_times, times),
            "first_image_time": _iso(times.iloc[0]) if not times.empty else None,
            "last_image_time": _iso(times.iloc[-1]) if not times.empty else None,
            "leading_missing_seconds": _edge_gap(all_times, times, leading=True),
            "trailing_missing_seconds": _edge_gap(all_times, times, leading=False),
            "largest_gap_seconds": _largest_gap(times),
            "gap_count_over_threshold": int((_gaps(times) > 30.0).sum()),
        }
    return {
        "has_images": bool(len(cycle_images)),
        "total_image_count": int(len(cycle_images)),
        "camera_role_count": len(by_role),
        "has_unassigned_camera": any(name.startswith("unassigned_") for name in by_role),
        "by_camera_role": by_role,
    }


def _draw_sensor_row(axis: Any, timestamps: list[pd.Timestamp]) -> None:
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.grid(axis="x", alpha=0.16)
    if timestamps:
        axis.plot(timestamps, [0.5] * len(timestamps), linewidth=2.0, color="#444444")


def _timestamps(frame: pd.DataFrame, record: Mapping[str, object]) -> list[pd.Timestamp]:
    if "timestamp" in frame:
        values = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
        if not values.empty:
            return [pd.Timestamp(value) for value in values]
    value = pd.to_datetime(
        pd.Series([record.get("start_time"), record.get("end_time")], dtype="object"),
        errors="coerce",
    ).dropna()
    return [pd.Timestamp(item) for item in value]


def _coverage_ratio(sensor_times: pd.Series, image_times: pd.Series) -> float:
    if sensor_times.empty:
        return 0.0
    if image_times.empty:
        return 0.0
    covered = int(image_times.between(sensor_times.min(), sensor_times.max()).sum())
    return float(covered / len(sensor_times))


def _edge_gap(sensor_times: pd.Series, image_times: pd.Series, *, leading: bool) -> float | None:
    if sensor_times.empty or image_times.empty:
        return None
    boundary = sensor_times.min() if leading else sensor_times.max()
    image_boundary = image_times.min() if leading else image_times.max()
    seconds = (image_boundary - boundary).total_seconds()
    return float(max(0.0, seconds if leading else -seconds))


def _gaps(times: pd.Series) -> pd.Series:
    if len(times) < 2:
        return pd.Series(dtype=float)
    return pd.Series(
        times.sort_values().diff().dropna().dt.total_seconds(), dtype="float64"
    )


def _largest_gap(times: pd.Series) -> float:
    gaps = _gaps(times)
    return float(gaps.max()) if not gaps.empty else 0.0


def _iso(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return pd.Timestamp(str(value)).isoformat()
