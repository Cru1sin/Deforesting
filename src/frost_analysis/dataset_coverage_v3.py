"""Coverage masks and role summaries for Dataset v3."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd


def coverage_ratio(frame: pd.DataFrame, image_metadata: pd.DataFrame) -> float:
    """Count unique matched grid timestamps over all cycle timestamps."""
    eligible = set(pd.to_datetime(frame["timestamp"], errors="coerce").dropna())
    covered = set(
        pd.to_datetime(
            image_metadata.get(
                "matched_timestamp", pd.Series(dtype="datetime64[ns]")
            ),
            errors="coerce",
        ).dropna()
    )
    if not eligible:
        return 0.0
    return min(1.0, len(eligible.intersection(covered)) / len(eligible))


def sensor_overall_mask(frame: pd.DataFrame, registry: Mapping[str, Any]) -> pd.Series:
    """Return the source-sensor presence mask from coverage_required channels."""
    required = [
        name
        for name, settings in dict(registry.get("channels", {})).items()
        if isinstance(settings, Mapping) and bool(settings.get("coverage_required", False))
    ]
    if not required:
        return pd.Series(True, index=frame.index, dtype=bool)
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for name in required:
        if name not in frame:
            return pd.Series(False, index=frame.index, dtype=bool)
        mask &= frame[name].notna()
        imputed = f"{name}__imputed"
        if imputed not in frame:
            return pd.Series(False, index=frame.index, dtype=bool)
        mask &= frame[imputed].eq(False).fillna(False).astype(bool)
    return mask


def render_rgb_coverage(
    frame: pd.DataFrame,
    image_metadata: pd.DataFrame,
    record: Mapping[str, Any],
    output_path: Path,
    *,
    registry: Mapping[str, Any] | None = None,
) -> None:
    """Render one row per sensor/camera mask with explicit missing intervals."""
    from .report import (
        _cycle_time_origin,
        _defrost_state_gap_intervals,
        _shade_cycle_stages,
        _shade_defrost_state_gaps,
    )

    timestamp_frame = frame.loc[frame["timestamp"].notna()].drop_duplicates(
        "timestamp", keep="first"
    )
    timestamps = pd.to_datetime(timestamp_frame["timestamp"], errors="coerce")
    cycle = pd.Series(dict(record))
    roles = sorted(
        image_metadata.get("camera_role", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .unique()
    )
    labels = ["sensor_overall", *roles]
    figure, axis = plt.subplots(figsize=(14, max(2.5, 0.45 * len(labels) + 1.2)))
    if timestamps.empty:
        axis.text(0.5, 0.5, "unavailable", ha="center", va="center")
        axis.axis("off")
    else:
        origin = _cycle_time_origin(timestamp_frame, cycle)
        gap_frame = timestamp_frame.copy()
        if "defrost_active" in gap_frame:
            imputed = gap_frame.get(
                "defrost_active__imputed", pd.Series(False, index=gap_frame.index)
            )
            gap_frame["defrost_active__missing"] = gap_frame["defrost_active"].isna() | (
                imputed.fillna(False).astype(bool)
            )
            for suffix in ("__invalid", "__duplicate", "__conflict"):
                gap_frame[f"defrost_active{suffix}"] = False
        gap_intervals = _defrost_state_gap_intervals(gap_frame)
        _shade_cycle_stages([axis], cycle, origin)
        _shade_defrost_state_gaps([axis], gap_intervals, origin)
        sensor_mask = sensor_overall_mask(timestamp_frame, registry or {})
        elapsed = (timestamps - origin).dt.total_seconds().div(60.0)
        for position, label in enumerate(labels):
            if label == "sensor_overall":
                present = sensor_mask.to_numpy(dtype=bool)
            else:
                role_times = set(
                    pd.to_datetime(
                        image_metadata.loc[
                            image_metadata["camera_role"].eq(label),
                            "matched_timestamp",
                        ],
                        errors="coerce",
                    ).dropna()
                )
                present = timestamps.isin(role_times).to_numpy(dtype=bool)
            _draw_mask(axis, elapsed, present, position)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlim(cast(Any, elapsed.min()), cast(Any, elapsed.max()))
        axis.set_xlabel("Time from heating start [min]")
        axis.set_title(f"{record.get('cycle_name', 'cycle')} · sensor/RGB coverage")
        axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def summarize_cycle_roles(
    frame: pd.DataFrame, image_metadata: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Return per-role coverage facts using cycle-aware denominators."""
    result: dict[str, dict[str, Any]] = {}
    frame_cycle_names = (
        set(frame["cycle_name"].dropna().astype(str))
        if "cycle_name" in frame
        else set()
    )
    total_cycles = len(frame_cycle_names) if frame_cycle_names else 1
    for role, group in image_metadata.groupby("camera_role", sort=True):
        ratios: list[float] = []
        if frame_cycle_names and "cycle_name" in group:
            for cycle_name, cycle_images in group.groupby("cycle_name", sort=True):
                cycle_frame = frame.loc[
                    frame["cycle_name"].astype(str).eq(str(cycle_name))
                ]
                if not cycle_frame.empty:
                    ratios.append(coverage_ratio(cycle_frame, cycle_images))
        else:
            ratios.append(coverage_ratio(frame, group))
        result[str(role)] = {
            "image_count": int(len(group)),
            "cycle_count": len(ratios),
            "mean_coverage_ratio": float(sum(ratios) / len(ratios)) if ratios else 0.0,
            "minimum_coverage_ratio": float(min(ratios)) if ratios else 0.0,
            "fully_covered_cycle_count": sum(ratio == 1.0 for ratio in ratios),
            "has_gap_cycle_count": sum(ratio < 1.0 for ratio in ratios),
            "missing_role_cycle_count": total_cycles - len(ratios),
        }
    return result


def _draw_mask(axis: Any, timestamps: pd.Series, present: Any, row: int) -> None:
    values = list(present)
    for index, timestamp in enumerate(timestamps):
        color = "#2ca25f" if bool(values[index]) else "#de2d26"
        axis.plot([timestamp, timestamp], [row - 0.32, row + 0.32], color=color, linewidth=3)
