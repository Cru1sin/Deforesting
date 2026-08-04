"""RGB/sensor temporal coverage summaries and rendering."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


def render_rgb_coverage_legacy(
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


# ---------------------------------------------------------------------------
# Canonical v3 interval coverage
# ---------------------------------------------------------------------------

def coverage_intervals(
    frame: pd.DataFrame,
    image_metadata: pd.DataFrame,
    *,
    max_image_gap_seconds: float = 40.0,
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Return merged available and missing RGB intervals for one cycle."""
    timestamps = pd.to_datetime(
        frame.get("timestamp", pd.Series(dtype="datetime64[ns]")),
        errors="coerce",
    )
    timestamps = timestamps.dropna().sort_values().drop_duplicates()
    if timestamps.empty:
        return [], []
    positive = timestamps.diff().dropna().dt.total_seconds()
    sample_seconds = float(positive.median()) if not positive.empty else 1.0
    cycle_start = pd.Timestamp(timestamps.iloc[0])
    cycle_end = pd.Timestamp(timestamps.iloc[-1]) + pd.Timedelta(seconds=sample_seconds)
    image_times = pd.to_datetime(
        image_metadata.get("image_time", pd.Series(dtype="datetime64[ns]")),
        errors="coerce",
    ).dropna().sort_values().drop_duplicates()
    image_times = image_times.loc[image_times.ge(cycle_start) & image_times.lt(cycle_end)]
    if image_times.empty:
        return [], [(cycle_start, cycle_end)]
    max_gap = float(max_image_gap_seconds)
    if max_gap <= 0:
        raise ValueError("max_image_gap_seconds must be positive")
    available: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    missing: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    first = pd.Timestamp(image_times.iloc[0])
    if first > cycle_start:
        missing.append((cycle_start, first))
    else:
        available.append((cycle_start, first))
    previous = first
    for current_value in image_times.iloc[1:]:
        current = pd.Timestamp(current_value)
        gap = (current - previous).total_seconds()
        if gap <= max_gap:
            available.append((previous, current))
        else:
            available.append((previous, previous))
            missing.append((previous, current))
        previous = current
    if previous < cycle_end:
        missing.append((previous, cycle_end))
    else:
        available.append((previous, cycle_end))
    return _merge_time_intervals(available), _merge_time_intervals(missing)


def coverage_ratio(
    frame: pd.DataFrame,
    image_metadata: pd.DataFrame,
    *,
    max_image_gap_seconds: float = 40.0,
) -> float:
    """Compute RGB coverage as available time divided by cycle duration."""
    timestamps = pd.to_datetime(
        frame.get("timestamp", pd.Series(dtype="datetime64[ns]")),
        errors="coerce",
    ).dropna()
    if timestamps.empty:
        return 0.0
    positive = timestamps.sort_values().diff().dropna().dt.total_seconds()
    sample_seconds = float(positive.median()) if not positive.empty else 1.0
    duration = (
        timestamps.max() + pd.Timedelta(seconds=sample_seconds) - timestamps.min()
    ).total_seconds()
    available, _ = coverage_intervals(
        frame, image_metadata, max_image_gap_seconds=max_image_gap_seconds
    )
    covered = sum((end - start).total_seconds() for start, end in available)
    return min(1.0, max(0.0, covered / duration)) if duration > 0 else 0.0


def sensor_overall_mask(frame: pd.DataFrame, registry: Mapping[str, Any]) -> pd.Series:
    """Return the 10-s sensor mask defined by coverage_required channels."""
    required = [
        str(name)
        for name, settings in dict(registry.get("channels", {})).items()
        if isinstance(settings, Mapping) and bool(settings.get("coverage_required", False))
    ]
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for name in required:
        if name not in frame or f"{name}__imputed" not in frame:
            return pd.Series(False, index=frame.index, dtype=bool)
        mask &= frame[name].notna()
        mask &= frame[f"{name}__imputed"].eq(False).fillna(False).astype(bool)
    return mask


def render_rgb_coverage(
    frame: pd.DataFrame,
    image_metadata: pd.DataFrame,
    record: Mapping[str, object],
    output_path: Path,
    *,
    registry: Mapping[str, Any] | None = None,
) -> None:
    """Render a publication-style interval raster for sensor and RGB coverage.

    The bars are built from merged time intervals, so the figure represents
    temporal availability rather than image-marker density. The same interval
    construction is used by :func:`coverage_ratio` and the manifest summary.
    """
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    registry_value = registry or {}
    coverage_settings = registry_value.get("image_coverage", {})
    max_gap = float(
        coverage_settings.get("max_image_gap_seconds", 40.0)
        if isinstance(coverage_settings, Mapping)
        else 40.0
    )
    timestamp_values = pd.to_datetime(
        frame.get("timestamp", pd.Series(dtype="datetime64[ns]")),
        errors="coerce",
    )
    timestamps = timestamp_values.dropna().sort_values(kind="stable").drop_duplicates()
    roles = sorted(
        image_metadata.get("camera_role", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .unique()
    )
    labels = ["sensor_overall", *roles]
    figure_height = max(1.85, 1.08 + 0.34 * len(labels))
    figure, axis = plt.subplots(figsize=(7.2, figure_height), dpi=300)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    if timestamps.empty:
        axis.text(0.5, 0.5, "unavailable", ha="center", va="center", fontsize=7)
        axis.axis("off")
    else:
        from .report import (
            _cycle_time_origin,
            _defrost_state_gap_intervals,
            _infer_cycle_stage_boundaries,
            _shade_cycle_stages,
            _shade_defrost_state_gaps,
        )

        timestamp_frame = frame.loc[frame["timestamp"].notna()].copy()
        timestamp_frame["timestamp"] = pd.to_datetime(
            timestamp_frame["timestamp"], errors="coerce"
        )
        timestamp_frame = timestamp_frame.loc[timestamp_frame["timestamp"].notna()]
        timestamp_frame = timestamp_frame.sort_values("timestamp", kind="stable")
        timestamp_frame = timestamp_frame.drop_duplicates("timestamp", keep="first")
        cycle = _infer_cycle_stage_boundaries(
            timestamp_frame, pd.Series(dict(record))
        )
        origin = _cycle_time_origin(timestamp_frame, cycle)
        positive = timestamps.diff().dropna().dt.total_seconds()
        sample_seconds = float(positive.median()) if not positive.empty else 1.0
        cycle_start = pd.Timestamp(timestamps.iloc[0])
        cycle_end = pd.Timestamp(timestamps.iloc[-1]) + pd.Timedelta(
            seconds=sample_seconds
        )
        left = (cycle_start - origin).total_seconds() / 60.0
        right = (cycle_end - origin).total_seconds() / 60.0

        gap_frame = timestamp_frame.copy()
        if "defrost_active" in gap_frame:
            gap_frame["defrost_active__missing"] = gap_frame["defrost_active"].isna()
            for suffix in ("__invalid", "__duplicate", "__conflict"):
                gap_frame[f"defrost_active{suffix}"] = False
        _shade_cycle_stages([axis], cycle, origin)
        _shade_defrost_state_gaps(
            [axis], _defrost_state_gap_intervals(gap_frame), origin
        )

        sensor_mask = sensor_overall_mask(timestamp_frame, registry_value)
        interval_sets: list[
            tuple[list[tuple[pd.Timestamp, pd.Timestamp]], list[tuple[pd.Timestamp, pd.Timestamp]]]
        ] = [
            (
                _mask_intervals(timestamps, sensor_mask),
                _mask_intervals(timestamps, ~sensor_mask),
            )
        ]
        for label in roles:
            role_images = image_metadata.loc[
                image_metadata["camera_role"].astype(str).eq(label)
            ]
            interval_sets.append(
                coverage_intervals(
                    timestamp_frame,
                    role_images,
                    max_image_gap_seconds=max_gap,
                )
            )

        available_color = "#0F4D92"
        missing_color = "#D6D8DB"
        for position, (available, missing) in enumerate(interval_sets):
            _draw_time_bands(
                axis, available, origin, position, available_color, False
            )
            _draw_time_bands(axis, missing, origin, position, missing_color, True)

        axis.set_ylim(-0.5, len(labels) - 0.5)
        axis.invert_yaxis()
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels, fontproperties=_camera_role_font(), fontsize=7)
        axis.set_xlim(min(0.0, left), max(right, left + 1.0))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{value:g}")
        )
        axis.set_xlabel("Time from cycle start [min]", fontsize=7)
        axis.tick_params(
            axis="both", which="major", labelsize=6.5, width=0.55, length=2.5
        )
        axis.tick_params(axis="y", length=0, pad=5)
        axis.grid(axis="x", color="#B8C2CC", linewidth=0.45, alpha=0.55)
        axis.set_axisbelow(True)
        axis.set_title(
            f"{record.get('cycle_name', 'cycle')} · sensor and RGB coverage",
            loc="left",
            fontsize=8,
            fontweight="bold",
            pad=16,
        )
        axis.legend(
            handles=[
                Patch(facecolor=available_color, edgecolor="none", label="available"),
                Patch(
                    facecolor=missing_color,
                    edgecolor="#8A8F94",
                    hatch="///",
                    linewidth=0.35,
                    label="missing",
                ),
            ],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncols=2,
            handlelength=1.5,
            handleheight=0.8,
            columnspacing=1.3,
            handletextpad=0.45,
            borderaxespad=0.0,
            fontsize=6.5,
            frameon=False,
        )
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.spines["left"].set_linewidth(0.65)
        axis.spines["bottom"].set_linewidth(0.65)
    figure.subplots_adjust(left=0.24, right=0.99, bottom=0.23, top=0.78)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)


def _mask_intervals(
    timestamps: pd.Series, mask: pd.Series
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    values = pd.to_datetime(timestamps, errors="coerce")
    mask_values = pd.Series(mask).fillna(False).astype(bool).to_numpy()
    if len(values) != len(mask_values):
        raise ValueError("timestamps and mask must have the same length")
    ordered = pd.DataFrame(
        {"timestamp": values, "present": mask_values}
    ).dropna(subset=["timestamp"])
    if ordered.empty:
        return []
    ordered = ordered.sort_values("timestamp", kind="stable").drop_duplicates("timestamp")
    positive = ordered["timestamp"].diff().dropna().dt.total_seconds()
    step = float(positive.median()) if not positive.empty else 1.0
    result: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start: pd.Timestamp | None = None
    previous: pd.Timestamp | None = None
    for timestamp, present in ordered.itertuples(index=False):
        timestamp = pd.Timestamp(timestamp)
        present = bool(present)
        if present:
            if start is None:
                start = timestamp
        elif start is not None and previous is not None:
            result.append((start, previous + pd.Timedelta(seconds=step)))
            start = None
        previous = timestamp
    if start is not None and previous is not None:
        result.append((start, previous + pd.Timedelta(seconds=step)))
    return _merge_time_intervals(result)


def _draw_time_bands(
    axis: Any,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    origin: pd.Timestamp,
    row: int,
    color: str,
    hatch: bool,
) -> None:
    for start, end in intervals:
        left = (start - origin).total_seconds() / 60.0
        width = max(0.0, (end - start).total_seconds() / 60.0)
        if width <= 0:
            continue
        axis.broken_barh(
            [(left, width)],
            (row - 0.32, 0.64),
            facecolors=color,
            edgecolors="#888888" if hatch else "none",
            hatch="///" if hatch else None,
            linewidth=0.35,
        )


def _merge_time_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _camera_role_font() -> font_manager.FontProperties | None:
    """Use an installed CJK font for source camera folder names when available."""
    for family in ("Heiti TC", "Hiragino Sans GB", "Songti SC", "Noto Sans CJK SC"):
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family),
                fallback_to_default=False,
            )
        except (FileNotFoundError, ValueError):
            continue
        return font_manager.FontProperties(fname=path)
    return None
