"""Pure renderers for self-contained Dataset artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator


def render_cycle_publication(
    cycle_frame: pd.DataFrame,
    cycle_record: Mapping[str, object],
    output_path: Path,
) -> None:
    """Render a cycle with the established publication renderer."""
    from .report import _infer_cycle_stage_boundaries, _plot_one_cycle_publication

    boundaries = cycle_record.get("boundaries")
    flattened: dict[str, object] = dict(cycle_record)
    if isinstance(boundaries, Mapping):
        flattened.update(boundaries)
    flattened["cycle_status"] = cycle_record.get(
        "status", cycle_record.get("pipeline_status", cycle_record.get("cycle_status"))
    )
    flattened["cycle_status_reason"] = cycle_record.get(
        "status_reason", cycle_record.get("pipeline_status_reason")
    )
    flattened["cycle_id"] = cycle_record.get("cycle_id")
    report_cycle = _infer_cycle_stage_boundaries(cycle_frame, pd.Series(flattened))
    _plot_one_cycle_publication(
        cycle_frame,
        cycle_frame,
        report_cycle,
        output_path,
        [],
        processed_only=True,
        include_humidity=True,
    )


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
