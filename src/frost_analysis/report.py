"""Read-only scientific QA reporting.

This module may read, select, group, mask, count, format, and render values
already stored in formal run outputs. It must not reconstruct measurements or
recompute cycle segmentation, resampling, imputation, derived quantities,
baselines, residuals, statistical evidence, thresholds, or decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from .analysis import imputed_column_for_value
from .validation import validate_analysis, validate_prepared, validate_processed

_REQUIRED_FILES = (
    "prepared_data.parquet",
    "processed_data.parquet",
    "cycle_summary.csv",
    "candidate_channel_evidence.csv",
)
_SUMMARY_TIME_COLUMNS = (
    "heating_start",
    "stable_heating_start",
    "defrost_start",
    "defrost_end",
    "baseline_start",
    "baseline_end",
)
_FIXED_OVERVIEW_CHANNELS = (
    "compressor_frequency",
    "heating_capacity",
    "cop",
    "ambient_temperature",
    "water_in_temperature",
    "water_out_temperature",
    "evaporating_pressure",
    "evaporating_temperature",
)
_BASELINE_DIAGNOSTIC_CHANNELS = (
    "ambient_temperature",
    "water_in_temperature",
    "water_out_temperature",
    "compressor_frequency",
)
_QUALITY_SUFFIXES = ("__missing", "__invalid", "__duplicate", "__conflict")
_EXCLUDED_COVERAGE_COLUMNS = {
    "experiment_id",
    "experiment_date",
    "timestamp",
    "cycle_id",
    "cycle_stage",
    "cycle_status",
    "cycle_status_reason",
    "cycle_elapsed_seconds",
    "cycle_progress",
}
_STAGE_COLORS = {
    "recovery": "#9ecae1",
    "frost_development": "#fdae6b",
    "defrost": "#a1d99b",
    "partial": "#bdbdbd",
}


def generate_report(input_dir: Path, output_dir: Path, overwrite: bool = False) -> Path:
    """Render a read-only QA report from one formal run directory."""
    prepared, processed, summary, evidence, manifest = _load_report_inputs(input_dir)
    _validate_image_columns(prepared, processed)
    _validate_processed_quality_columns(processed, evidence)

    output_dir = output_dir.resolve()
    input_dir = input_dir.resolve()
    if output_dir == input_dir:
        raise ValueError("report output directory must differ from input directory")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"report output is not a directory: {output_dir}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"report output already exists: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    warnings: list[dict[str, str]] = []
    figures: list[str] = []
    try:
        figures.extend(_plot_cycle_overviews(prepared, processed, summary, temporary_dir, warnings))
        figures.append(_plot_coverage(prepared, summary, temporary_dir, warnings))
        figures.append(_plot_baseline(processed, summary, temporary_dir, warnings))
        figures.append(_plot_candidates(processed, summary, evidence, temporary_dir, warnings))
        report_summary = _report_summary(input_dir, manifest, figures, warnings)
        (temporary_dir / "report_summary.json").write_text(
            json.dumps(report_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _publish_report(temporary_dir, output_dir, overwrite)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    return output_dir


def _load_report_inputs(
    input_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"report input directory does not exist: {input_dir}")
    missing = [name for name in _REQUIRED_FILES if not (input_dir / name).is_file()]
    if missing:
        raise ValueError(f"report input is incomplete: missing {missing}")
    prepared = pd.read_parquet(input_dir / "prepared_data.parquet")
    processed = pd.read_parquet(input_dir / "processed_data.parquet")
    summary = pd.read_csv(input_dir / "cycle_summary.csv")
    evidence = pd.read_csv(input_dir / "candidate_channel_evidence.csv")
    _parse_time_column(prepared, "timestamp")
    _parse_time_column(processed, "timestamp")
    for column in _SUMMARY_TIME_COLUMNS:
        if column in summary:
            _parse_time_column(summary, column)
    validate_prepared(prepared, summary)
    validate_processed(processed, summary)
    validate_analysis(evidence)
    manifest = _read_manifest(input_dir / "manifest.json")
    _validate_summary_contract(summary)
    _validate_run_identity(prepared, processed, summary, evidence, manifest)
    return prepared, processed, summary, evidence, manifest


def _parse_time_column(frame: pd.DataFrame, column: str) -> None:
    values = frame[column]
    parsed = pd.to_datetime(values, errors="coerce")
    if values.notna().any() and parsed.loc[values.notna()].isna().any():
        raise ValueError(f"{column} contains unparseable timestamps")
    frame[column] = parsed


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid manifest.json: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("manifest.json must contain a JSON object")
    return value


def _validate_image_columns(prepared: pd.DataFrame, processed: pd.DataFrame) -> None:
    if set(_image_roles(prepared)) != set(_image_roles(processed)):
        raise ValueError("Prepared and Processed image roles do not match")
    for frame in (prepared, processed):
        role_columns: dict[str, set[str]] = {}
        for column in frame.columns:
            match = re.fullmatch(r"image_(.+)_(path|time|offset_seconds)", str(column))
            if match:
                role_columns.setdefault(match.group(1), set()).add(match.group(2))
        for role, suffixes in role_columns.items():
            if suffixes != {"path", "time", "offset_seconds"}:
                raise ValueError(f"image role columns are incomplete for {role}")
            path_present = frame[f"image_{role}_path"].notna()
            time_present = frame[f"image_{role}_time"].notna()
            offset_present = frame[f"image_{role}_offset_seconds"].notna()
            if not (path_present.eq(time_present) & path_present.eq(offset_present)).all():
                raise ValueError(f"image role row is incomplete for {role}")
            parsed_times = pd.to_datetime(frame[f"image_{role}_time"], errors="coerce")
            if time_present.any() and parsed_times.loc[time_present].isna().any():
                raise ValueError(f"image role time is invalid for {role}")
            offsets = pd.to_numeric(frame[f"image_{role}_offset_seconds"], errors="coerce")
            if offset_present.any() and offsets.loc[offset_present].isna().any():
                raise ValueError(f"image role offset is invalid for {role}")


def _validate_summary_contract(summary: pd.DataFrame) -> None:
    required = {"cycle_id", "cycle_status"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"cycle summary missing report fields: {missing}")
    boundary_columns = ("heating_start", "stable_heating_start", "defrost_start", "defrost_end")
    for _, cycle in summary.iterrows():
        status = str(cycle["cycle_status"])
        if status not in {"valid", "incomplete", "invalid"}:
            raise ValueError(f"cycle summary contains invalid cycle status: {status}")
        values = [cycle.get(column) for column in boundary_columns]
        present = [pd.notna(value) for value in values]
        if status in {"valid", "invalid"} and not all(present):
            raise ValueError("cycle summary has incomplete cycle boundaries")
        ordered = [pd.Timestamp(value) for value in values if pd.notna(value)]
        if ordered != sorted(ordered):
            raise ValueError("cycle summary has invalid cycle boundaries")


def _validate_run_identity(
    prepared: pd.DataFrame,
    processed: pd.DataFrame,
    summary: pd.DataFrame,
    evidence: pd.DataFrame,
    manifest: dict[str, Any] | None,
) -> None:
    frames = (
        ("Prepared", prepared),
        ("Processed", processed),
        ("Cycle Summary", summary),
        ("Evidence", evidence),
    )
    experiment_ids: set[str] = set()
    experiment_dates: set[str] = set()
    for name, frame in frames:
        experiment_ids.update(_nonempty_values(frame, "experiment_id", name))
        if "experiment_date" in frame:
            experiment_dates.update(_nonempty_values(frame, "experiment_date", name))
    if len(experiment_ids) != 1 or len(experiment_dates) > 1:
        raise ValueError("report inputs have inconsistent experiment identity")
    experiment_id = next(iter(experiment_ids))
    for name, frame in frames:
        values = _nonempty_values(frame, "experiment_id", name)
        if values and values != {experiment_id}:
            raise ValueError("report inputs have inconsistent experiment identity")
    if manifest is not None:
        for key, expected in (
            ("experiment_id", experiment_id),
            ("experiment_date", next(iter(experiment_dates), None)),
        ):
            actual = manifest.get(key)
            if actual is not None and expected is not None and str(actual) != expected:
                raise ValueError("manifest has inconsistent experiment identity")


def _nonempty_values(frame: pd.DataFrame, column: str, name: str) -> set[str]:
    if column not in frame:
        raise ValueError(f"{name} is missing {column}")
    if not frame.empty and frame[column].isna().any():
        raise ValueError(f"{name} has null experiment identity")
    return set(frame[column].dropna().astype(str))


def _validate_processed_quality_columns(
    processed: pd.DataFrame, evidence: pd.DataFrame
) -> None:
    for column in processed.columns:
        if str(column).endswith("__baseline_residual"):
            quality = imputed_column_for_value(str(column))
            if quality not in processed:
                raise ValueError(f"processed value column lacks quality column: {quality}")
    for candidate in evidence.get("channel", pd.Series(dtype=str)).astype(str):
        residual = f"{candidate}__baseline_residual"
        if residual in processed:
            quality = imputed_column_for_value(residual)
            if quality not in processed:
                raise ValueError(f"processed value column lacks quality column: {quality}")


def _plot_cycle_overviews(
    prepared: pd.DataFrame,
    processed: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    warnings: list[dict[str, str]],
) -> list[str]:
    cycle_dir = output_dir / "cycles"
    cycle_dir.mkdir()
    figures: list[str] = []
    for _, cycle in summary.iterrows():
        cycle_id = str(cycle["cycle_id"])
        prepared_cycle = prepared.loc[_cycle_mask(prepared, cycle)]
        processed_cycle = processed.loc[_cycle_mask(processed, cycle)]
        path = cycle_dir / f"{cycle_id}_overview.png"
        _plot_one_cycle(prepared_cycle, processed_cycle, cycle, path, warnings)
        figures.append(str(path.relative_to(output_dir)))
    return figures


def _plot_one_cycle(
    prepared: pd.DataFrame,
    processed: pd.DataFrame,
    cycle: pd.Series,
    path: Path,
    warnings: list[dict[str, str]],
) -> None:
    figure, axes = plt.subplots(7, 1, figsize=(15, 20), sharex=True)
    cycle_id = str(cycle["cycle_id"])
    _plot_prepared_line(
        axes[0],
        prepared,
        "compressor_frequency",
        "Compressor frequency",
        cycle_id,
        warnings,
    )
    _plot_prepared_line(
        axes[1], prepared, "heating_capacity", "Heating capacity", cycle_id, warnings
    )
    _plot_processed_line(axes[2], processed, "cop", "COP", cycle_id, warnings)
    for channel, color in zip(
        ("ambient_temperature", "water_in_temperature", "water_out_temperature"),
        ("#3182bd", "#e6550d", "#31a354"),
        strict=True,
    ):
        _plot_prepared_line(axes[3], prepared, channel, channel, cycle_id, warnings, color)
    for channel, color in zip(
        ("evaporating_pressure", "evaporating_temperature"),
        ("#756bb1", "#636363"),
        strict=True,
    ):
        _plot_prepared_line(axes[4], prepared, channel, channel, cycle_id, warnings, color)
    _plot_stage_and_defrost(axes[5], prepared, cycle_id, warnings)
    axes[6].axis("off")

    boundaries = {
        "heating_start": cycle.get("heating_start"),
        "stable_heating_start": cycle.get("stable_heating_start"),
        "defrost_start": cycle.get("defrost_start"),
        "defrost_end": cycle.get("defrost_end"),
    }
    for axis in axes[:6]:
        for _name, value in boundaries.items():
            if pd.notna(value):
                axis.axvline(pd.Timestamp(value), color="#333333", alpha=0.35, linewidth=0.8)
        _shade_baseline(axis, cycle)
    _format_cycle_metadata(figure, axes[6], prepared, processed, cycle)
    figure_name = str(path.relative_to(path.parent.parent))
    for role, count in _cycle_image_counts(prepared, cycle_id).items():
        if count == 0 and not prepared.empty:
            _record_warning(
                warnings,
                "empty_camera_role",
                figure_name,
                cycle_id,
                role,
                "Camera role exists but has no matched images in this cycle.",
            )
    if cycle.get("cycle_status") == "valid" and processed.empty:
        _record_warning(
            warnings,
            "cycle_without_processed_rows",
            figure_name,
            cycle_id,
            message="Valid cycle has no Processed rows.",
        )
    start = _format_timestamp(cycle.get("heating_start"))
    end = _format_timestamp(cycle.get("defrost_end"))
    duration = _cycle_duration(cycle)
    figure.suptitle(
        f"{cycle_id} | {cycle.get('cycle_status', 'N/A')}\n"
        f"{start} → {end} | Duration: {duration}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_prepared_line(
    axis: Any,
    frame: pd.DataFrame,
    channel: str,
    label: str,
    cycle_id: str,
    warnings: list[dict[str, str]],
    color: str | None = None,
) -> None:
    if channel not in frame:
        _missing_panel(axis, label, "channel not present in Prepared", warnings, cycle_id, channel)
        return
    required = [f"{channel}{suffix}" for suffix in _QUALITY_SUFFIXES]
    if not set(required).issubset(frame.columns):
        _missing_panel(axis, label, "quality columns unavailable", warnings, cycle_id, channel)
        return
    if frame.empty:
        _empty_panel(
            axis,
            label,
            "no Prepared rows",
            warnings,
            cycle_id,
            channel,
            record_warning=False,
        )
        return
    values = _prepared_observed_series(frame, channel)
    if not values.notna().any():
        _empty_panel(axis, label, "no observed values", warnings, cycle_id, channel)
        return
    axis.plot(frame["timestamp"], pd.to_numeric(values, errors="coerce"), label=label, color=color)
    axis.set_ylabel(label)
    axis.legend(loc="upper right", fontsize=8)


def _plot_processed_line(
    axis: Any,
    frame: pd.DataFrame,
    channel: str,
    label: str,
    cycle_id: str,
    warnings: list[dict[str, str]],
    color: str | None = None,
) -> None:
    if channel not in frame:
        _missing_panel(axis, label, "channel not present in Processed", warnings, cycle_id, channel)
        return
    quality = f"{channel}__imputed"
    if quality not in frame:
        raise ValueError(f"processed value column lacks quality column: {quality}")
    if frame.empty:
        _empty_panel(
            axis,
            label,
            "no Processed rows",
            warnings,
            cycle_id,
            channel,
            record_warning=False,
        )
        return
    values = _processed_observed_series(frame, channel)
    if not values.notna().any():
        _empty_panel(axis, label, "no observed values", warnings, cycle_id, channel)
        return
    axis.plot(frame["timestamp"], pd.to_numeric(values, errors="coerce"), label=label, color=color)
    axis.set_ylabel(label)
    axis.legend(loc="upper right", fontsize=8)


def _plot_stage_and_defrost(
    axis: Any,
    frame: pd.DataFrame,
    cycle_id: str,
    warnings: list[dict[str, str]],
) -> None:
    if "cycle_stage" in frame:
        stage_codes = {stage: index for index, stage in enumerate(_STAGE_COLORS)}
        for stage, code in stage_codes.items():
            mask = frame["cycle_stage"].eq(stage)
            if mask.any():
                axis.scatter(
                    frame.loc[mask, "timestamp"],
                    [code] * int(mask.sum()),
                    color=_STAGE_COLORS[stage],
                    s=12,
                )
        axis.set_yticks(list(stage_codes.values()), list(stage_codes))
        axis.set_ylabel("stage")
    else:
        _missing_panel(
            axis,
            "stage",
            "cycle_stage not present in Prepared",
            warnings,
            cycle_id,
            "cycle_stage",
        )
    if "defrost_active" in frame:
        quality = _prepared_quality_available(frame, "defrost_active")
        if quality:
            values = _prepared_observed_series(frame, "defrost_active").astype("boolean")
            axis.step(
                frame["timestamp"],
                values.astype(float),
                where="post",
                color="#de2d26",
                alpha=0.7,
            )
            axis.text(0.01, 0.02, "red: defrost_active", transform=axis.transAxes, fontsize=8)
        else:
            _missing_panel(
                axis,
                "defrost_active",
                "quality columns unavailable",
                warnings,
                cycle_id,
                "defrost_active",
            )


def _format_cycle_metadata(
    figure: Any,
    axis: Any,
    prepared: pd.DataFrame,
    processed: pd.DataFrame,
    cycle: pd.Series,
) -> None:
    counts = [
        f"{role}: {count}"
        for role, count in _cycle_image_counts(prepared, str(cycle["cycle_id"])).items()
    ]
    lines = [
        f"status: {cycle.get('cycle_status', 'N/A')}",
        f"reason: {cycle.get('cycle_status_reason', '') or 'N/A'}",
        (
            f"baseline: {_format_timestamp(cycle.get('baseline_start'))} → "
            f"{_format_timestamp(cycle.get('baseline_end'))}"
        ),
        f"processed rows: {len(processed)}" if not processed.empty else "processed rows: N/A",
        "images: " + (", ".join(counts) if counts else "none"),
    ]
    axis.text(0.01, 0.95, "\n".join(lines), va="top", transform=axis.transAxes, fontsize=10)


def _plot_coverage(
    prepared: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    warnings: list[dict[str, str]],
) -> str:
    channels = _coverage_channels(prepared)
    roles = _image_roles(prepared)
    labels = [*channels, *(f"camera:{role}" for role in roles)]
    figure_height = max(4.0, 0.32 * max(len(labels), 1) + 1.5)
    figure, axis = plt.subplots(figsize=(15, figure_height))
    if not labels:
        axis.text(0.5, 0.5, "No observable channels or image roles", ha="center", va="center")
        axis.axis("off")
    else:
        if not roles:
            axis.text(
                0.01,
                0.98,
                "No image roles exported",
                transform=axis.transAxes,
                va="top",
            )
        display_labels = list(labels)
        for index, channel in enumerate(channels):
            values = _prepared_observed_series(prepared, channel)
            mask = values.notna()
            if not mask.any():
                display_labels[index] = f"{channel} (empty)"
                _record_warning(
                    warnings,
                    "empty_visual_channel",
                    "coverage.png",
                    field=channel,
                    message="No observed values in Prepared.",
                )
            axis.scatter(
                prepared.loc[mask, "timestamp"],
                [index] * int(mask.sum()),
                marker="|",
                s=32,
            )
        for offset, role in enumerate(roles, start=len(channels)):
            times = (
                pd.to_datetime(prepared[f"image_{role}_time"], errors="coerce")
                .dropna()
                .drop_duplicates()
            )
            axis.scatter(times, [offset] * len(times), marker="|", s=60, label=role)
        axis.set_yticks(range(len(display_labels)), display_labels)
        axis.set_title("Prepared observed time coverage (display only)")
        axis.grid(axis="x", alpha=0.2)
        _add_summary_status_text(axis, summary)
    if not roles:
        _record_warning(
            warnings,
            "no_image_roles_exported",
            "coverage.png",
            message="No image roles were exported.",
        )
    figure.tight_layout()
    path = output_dir / "coverage.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return str(path.relative_to(output_dir))


def _plot_baseline(
    processed: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    warnings: list[dict[str, str]],
) -> str:
    figure, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    unavailable_lines: list[str] = []
    for _, cycle in summary.iterrows():
        if cycle.get("cycle_status") == "valid" and cycle.get("baseline_status") == "unavailable":
            reason = str(cycle.get("baseline_failure_reason", "baseline unavailable"))
            cycle_id = str(cycle["cycle_id"])
            _record_warning(
                warnings,
                "baseline_unavailable",
                "baseline.png",
                cycle_id,
                message=reason,
            )
            unavailable_lines.append(f"{cycle_id}: {reason}")
    if unavailable_lines:
        axes[0].text(
            0.01,
            0.05,
            "Baseline unavailable\n" + "\n".join(unavailable_lines),
            transform=axes[0].transAxes,
            fontsize=8,
        )
    for axis, channel in zip(axes, _BASELINE_DIAGNOSTIC_CHANNELS, strict=True):
        _plot_baseline_panel(axis, processed, channel, warnings)
        for _, cycle in summary.iterrows():
            if pd.notna(cycle.get("baseline_start")) and pd.notna(cycle.get("baseline_end")):
                axis.axvspan(
                    cycle["baseline_start"], cycle["baseline_end"], color="#74c476", alpha=0.25
                )
    figure.suptitle("Baseline diagnostic channels and saved baseline windows")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = output_dir / "baseline.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return str(path.relative_to(output_dir))


def _plot_baseline_panel(
    axis: Any,
    processed: pd.DataFrame,
    channel: str,
    warnings: list[dict[str, str]],
) -> None:
    if channel not in processed:
        _missing_panel(
            axis,
            channel,
            "channel not present in Processed",
            warnings,
            "all",
            channel,
            "baseline.png",
        )
        return
    quality = f"{channel}__imputed"
    if quality not in processed:
        raise ValueError(f"processed value column lacks quality column: {quality}")
    if processed.empty:
        _empty_panel(
            axis,
            channel,
            "no Processed rows",
            warnings,
            "all",
            channel,
            "baseline.png",
            record_warning=False,
        )
        return
    values = _processed_observed_series(processed, channel)
    if not values.notna().any():
        _empty_panel(
            axis,
            channel,
            "no observed values",
            warnings,
            "all",
            channel,
            "baseline.png",
        )
        return
    axis.plot(processed["timestamp"], pd.to_numeric(values, errors="coerce"), label=channel)
    axis.set_ylabel(channel)
    axis.legend(loc="upper right", fontsize=8)


def _plot_candidates(
    processed: pd.DataFrame,
    summary: pd.DataFrame,
    evidence: pd.DataFrame,
    output_dir: Path,
    warnings: list[dict[str, str]],
) -> str:
    count = len(evidence)
    rows = max(1, math.ceil(count / 3))
    figure, axes = plt.subplots(rows, 3, figsize=(18, rows * 4.2), squeeze=False)
    axes_flat = list(axes.flat)
    experiment_id = str(summary["experiment_id"].iloc[0])
    eligible = set(
        summary.loc[
            summary["cycle_status"].eq("valid") & summary["baseline_status"].eq("available"),
            "cycle_id",
        ]
    )
    colors = plt.get_cmap("tab20")
    cycle_ids = sorted(str(value) for value in eligible)
    cycle_color = {cycle_id: colors(index % 20) for index, cycle_id in enumerate(cycle_ids)}
    if evidence.empty:
        axes_flat[0].text(0.5, 0.5, "No candidate evidence rows", ha="center", va="center")
        axes_flat[0].axis("off")
        _record_warning(
            warnings,
            "no_candidate_evidence_rows",
            "candidate.png",
            message="Evidence contains no candidate rows.",
        )
    for index, (_, row) in enumerate(evidence.iterrows()):
        axis = axes_flat[index]
        candidate = str(row["channel"])
        residual = f"{candidate}__baseline_residual"
        quality = imputed_column_for_value(residual)
        if residual not in processed:
            _missing_panel(
                axis,
                candidate,
                "residual not present in Processed",
                warnings,
                "all",
                residual,
            )
        elif quality not in processed:
            raise ValueError(f"processed value column lacks quality column: {quality}")
        else:
            points = processed.loc[
                processed["experiment_id"].eq(experiment_id)
                & processed["cycle_id"].isin(eligible)
                & processed["cycle_stage"].eq("frost_development")
                & ~processed[quality].fillna(False).astype(bool)
            ]
            for cycle_id, group in points.groupby("cycle_id", sort=False):
                axis.scatter(
                    group["cycle_progress"],
                    pd.to_numeric(group[residual], errors="coerce"),
                    color=cycle_color.get(str(cycle_id), "#636363"),
                    s=14,
                    label=str(cycle_id),
                )
            if not points[residual].notna().any():
                _record_warning(
                    warnings,
                    "candidate_without_points",
                    "candidate.png",
                    message=f"No displayable points for {candidate}.",
                )
                axis.text(0.5, 0.5, "No eligible Processed observations", ha="center", va="center")
            axis.axhline(0, color="#636363", linewidth=0.8)
        axis.set_title(candidate)
        axis.set_xlabel("cycle_progress")
        axis.set_ylabel("baseline residual")
        annotation = (
            f"trend n={row['trend_cycle_count']}\n"
            f"effect={row['trend_effect']}\n"
            f"direction={row['direction_consistency']}\n"
            f"future={row['future_performance_association']}\n"
            f"context={row['median_max_abs_context_spearman']}\n"
            f"{row['decision']}\n{row['reason']}"
        )
        axis.text(0.02, 0.98, annotation, transform=axis.transAxes, va="top", fontsize=7)
    for axis in axes_flat[count:]:
        axis.axis("off")
    if cycle_color:
        figure.legend(
            handles=[
                Line2D([], [], marker="o", linestyle="", color=color, label=cycle_id)
                for cycle_id, color in cycle_color.items()
            ],
            loc="upper center",
            ncols=min(6, len(cycle_color)),
            fontsize=8,
        )
    figure.tight_layout(rect=(0, 0, 1, 0.94 if cycle_color else 1))
    path = output_dir / "candidate.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return str(path.relative_to(output_dir))


def _coverage_channels(frame: pd.DataFrame) -> list[str]:
    channels: list[str] = []
    for column in frame.columns:
        name = str(column)
        if name in _EXCLUDED_COVERAGE_COLUMNS or name.startswith("image_"):
            continue
        if name.endswith(_QUALITY_SUFFIXES) or "__" in name:
            continue
        if all(f"{name}{suffix}" in frame for suffix in _QUALITY_SUFFIXES):
            channels.append(name)
    return channels


def _cycle_mask(frame: pd.DataFrame, cycle: pd.Series) -> pd.Series:
    return frame["experiment_id"].eq(cycle["experiment_id"]) & frame["cycle_id"].eq(
        cycle["cycle_id"]
    )


def _cycle_image_counts(frame: pd.DataFrame, cycle_id: str) -> dict[str, int]:
    cycle = frame.loc[frame["cycle_id"].astype(str).eq(str(cycle_id))]
    return {
        role: int(cycle[f"image_{role}_path"].nunique(dropna=True))
        for role in _image_roles(frame)
    }


def _image_roles(frame: pd.DataFrame) -> list[str]:
    roles = {
        match.group(1)
        for column in frame.columns
        if (match := re.fullmatch(r"image_(.+)_path", str(column)))
    }
    return sorted(roles)


def _prepared_quality_available(frame: pd.DataFrame, channel: str) -> bool:
    return all(f"{channel}{suffix}" in frame for suffix in _QUALITY_SUFFIXES)


def _prepared_observed_series(frame: pd.DataFrame, channel: str) -> pd.Series:
    values = frame[channel].copy()
    invalid = pd.Series(False, index=frame.index)
    for suffix in _QUALITY_SUFFIXES:
        invalid = invalid | frame[f"{channel}{suffix}"].fillna(False).astype(bool)
    return values.mask(invalid)


def _processed_observed_series(frame: pd.DataFrame, channel: str) -> pd.Series:
    values = frame[channel].copy()
    imputed = frame[f"{channel}__imputed"].fillna(False).astype(bool)
    return values.mask(imputed)


def _missing_panel(
    axis: Any,
    label: str,
    message: str,
    warnings: list[dict[str, str]],
    cycle_id: str,
    field: str,
    figure: str = "cycle overview",
) -> None:
    axis.text(
        0.5,
        0.5,
        f"{label}\nUnavailable\n{message}",
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    axis.set_ylabel(label)
    _record_warning(warnings, "missing_visual_channel", figure, cycle_id, field, message)


def _empty_panel(
    axis: Any,
    label: str,
    message: str,
    warnings: list[dict[str, str]],
    cycle_id: str,
    field: str,
    figure: str = "cycle overview",
    record_warning: bool = True,
) -> None:
    axis.text(
        0.5,
        0.5,
        f"{label}\nEmpty\n{message}",
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    axis.set_ylabel(label)
    if record_warning:
        _record_warning(warnings, "empty_visual_channel", figure, cycle_id, field, message)


def _record_warning(
    warnings: list[dict[str, str]],
    code: str,
    figure: str,
    cycle_id: str | None = None,
    field: str | None = None,
    message: str = "",
) -> None:
    warning = {"code": code, "figure": figure, "message": message}
    if cycle_id is not None:
        warning["cycle_id"] = cycle_id
    if field is not None:
        warning["field"] = field
    if warning not in warnings:
        warnings.append(warning)


def _add_summary_status_text(axis: Any, summary: pd.DataFrame) -> None:
    metric_columns = (
        "excluded_transition_bucket_count",
        "low_coverage_channel_bucket_count",
        "eligible_continuous_channel_bucket_count",
    )
    statuses = []
    for _, row in summary.iterrows():
        line = f"{row.get('cycle_id')}: {row.get('cycle_status')}"
        reason = row.get("cycle_status_reason")
        if pd.notna(reason) and str(reason):
            line += f" ({reason})"
        for column in metric_columns:
            value = row.get(column)
            if pd.notna(value):
                line += f"\n  {column}={value}"
        statuses.append(line)
    if statuses:
        axis.text(1.01, 0.5, "\n".join(statuses), transform=axis.transAxes, va="center", fontsize=8)


def _shade_baseline(axis: Any, cycle: pd.Series) -> None:
    if pd.notna(cycle.get("baseline_start")) and pd.notna(cycle.get("baseline_end")):
        axis.axvspan(cycle["baseline_start"], cycle["baseline_end"], color="#74c476", alpha=0.2)


def _cycle_duration(cycle: pd.Series) -> str:
    for column in ("cycle_duration_seconds", "duration_seconds", "cycle_duration"):
        if column in cycle and pd.notna(cycle[column]):
            return _format_duration(float(cycle[column]))
    start = cycle.get("heating_start")
    end = cycle.get("defrost_end")
    if pd.notna(start) and pd.notna(end):
        return _format_duration((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds())
    return "N/A"


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours} h {minutes} min {secs} s"


def _format_timestamp(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _report_summary(
    input_dir: Path,
    manifest: dict[str, Any] | None,
    figures: list[str],
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    input_files = {
        name: {"sha256": _sha256(input_dir / name)} for name in _REQUIRED_FILES
    }
    provenance: dict[str, Any] = {}
    if manifest is not None:
        for key in ("experiment_id", "experiment_date", "git_commit"):
            if key in manifest:
                provenance[key] = manifest[key]
        if "config_provenance" in manifest:
            provenance["config_provenance"] = manifest["config_provenance"]
    return {
        "status": "success_with_warnings" if warnings else "success",
        "manifest_present": manifest is not None,
        "input_directory": str(input_dir),
        "input_files": input_files,
        "provenance": provenance,
        "figures": figures,
        "warnings": warnings,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_report(temporary_dir: Path, output_dir: Path, overwrite: bool) -> None:
    backup: Path | None = None
    old_output_moved = False
    new_output_published = False
    try:
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(f"report output already exists: {output_dir}")
            backup = output_dir.with_name(f".{output_dir.name}.backup")
            if backup.exists():
                shutil.rmtree(backup)
            output_dir.rename(backup)
            old_output_moved = True
        os.replace(temporary_dir, output_dir)
        new_output_published = True
    except Exception:
        if new_output_published and output_dir.exists():
            shutil.rmtree(output_dir)
        if old_output_moved and backup is not None and backup.exists() and not output_dir.exists():
            backup.rename(output_dir)
        raise
    else:
        if backup is not None and backup.exists():
            with suppress(OSError):
                shutil.rmtree(backup)
