"""Transparent candidate evidence for valid, baseline-backed cycles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .dataset_loader import DatasetLoader

EVIDENCE_COLUMNS = [
    "experiment_id",
    "experiment_date",
    "channel",
    "trend_cycle_count",
    "reset_pair_count",
    "future_cycle_count",
    "context_cycle_count",
    "trend_effect",
    "direction_consistency",
    "reset_effect",
    "reset_evidence_status",
    "reset_evidence_reason",
    "future_performance_association",
    "median_max_abs_context_spearman",
    "decision",
    "reason",
]

_DECISIONS = {
    "trend_supported_candidate",
    "partial_evidence",
    "insufficient_coverage",
}


def imputed_column_for_value(column: str) -> str:
    """Map a value or baseline residual column to its source quality column."""
    base = column.removesuffix("__baseline_residual")
    return f"{base}__imputed"


def analyze(
    processed: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    config: Any,
    channels: Mapping[str, Mapping[str, Any]],
    *,
    respect_cycle_status: bool = True,
) -> pd.DataFrame:
    """Compute one evidence row per experiment and configured candidate channel."""
    candidates = _candidate_names(channels)
    settings = _analysis_settings(config)
    _require_analysis_quality_columns(processed, candidates, channels, settings)
    if not candidates:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)
    experiments = _experiments(processed, config)
    rows: list[dict[str, object]] = []
    for experiment_id, experiment_date in experiments:
        current = processed.loc[processed["experiment_id"].eq(experiment_id)].copy()
        cycles = cycle_summary.loc[cycle_summary["experiment_id"].eq(experiment_id)].copy()
        for channel in candidates:
            trend = _trend_effects(
                current,
                cycles,
                channel,
                channels[channel],
                settings,
                respect_cycle_status=respect_cycle_status,
            )
            context_count, context_effect = _context_association(
                current,
                cycles,
                channel,
                channels,
                settings,
                respect_cycle_status=respect_cycle_status,
            )
            future_count, future_effect = _future_association(
                current,
                cycles,
                channel,
                settings,
                respect_cycle_status=respect_cycle_status,
            )
            trend_effect = _median_or_nan(trend)
            direction = _direction_consistency(trend)
            decision, reason = _decision(
                len(trend),
                trend_effect,
                direction,
                settings,
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "experiment_date": experiment_date,
                    "channel": channel,
                    "trend_cycle_count": len(trend),
                    "reset_pair_count": 0,
                    "future_cycle_count": future_count,
                    "context_cycle_count": context_count,
                    "trend_effect": trend_effect,
                    "direction_consistency": direction,
                    "reset_effect": np.nan,
                    "reset_evidence_status": "not_evaluated",
                    "reset_evidence_reason": "independent_reference_unavailable",
                    "future_performance_association": future_effect,
                    "median_max_abs_context_spearman": context_effect,
                    "decision": decision,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)


def _candidate_names(channels: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        name
        for name, settings in channels.items()
        if bool(settings.get("analysis_candidate", False))
    ]


def _experiments(processed: pd.DataFrame, config: Any) -> list[tuple[str, str]]:
    if processed.empty:
        return [(str(config.experiment_id), str(config.experiment_date))]
    columns = ["experiment_id", "experiment_date"]
    values = processed[columns].drop_duplicates().itertuples(index=False, name=None)
    return [(str(experiment_id), str(experiment_date)) for experiment_id, experiment_date in values]


def _analysis_settings(config: Any) -> Any:
    settings = config.analysis
    return settings


def _trend_effects(
    frame: pd.DataFrame,
    cycles: pd.DataFrame,
    channel: str,
    channel_settings: Mapping[str, Any],
    settings: Any,
    *,
    respect_cycle_status: bool,
) -> list[float]:
    residual = f"{channel}__baseline_residual"
    if residual not in frame or "cycle_progress" not in frame:
        return []
    quality = imputed_column_for_value(residual)
    eligible = _eligible_cycle_ids(cycles, respect_cycle_status=respect_cycle_status)
    development = frame.loc[
        frame["cycle_id"].isin(eligible)
        & frame["cycle_stage"].eq("frost_development")
    ]
    effects: list[float] = []
    for _, group in development.groupby(["experiment_id", "cycle_id"], sort=False):
        observed = ~_quality_mask(group, quality)
        correlation = _spearman_with_minimum(
            group.loc[observed, "cycle_progress"],
            group.loc[observed, residual],
            settings.minimum_points_per_cycle,
        )
        if correlation is None:
            continue
        direction = str(channel_settings.get("expected_frost_direction", ""))
        effects.append(correlation if direction == "increase" else -correlation)
    return effects


def _eligible_cycle_ids(
    cycles: pd.DataFrame, *, respect_cycle_status: bool = True
) -> set[object]:
    if "baseline_status" not in cycles:
        return set()
    mask = cycles["baseline_status"].eq("available")
    if respect_cycle_status and "cycle_status" in cycles:
        mask &= cycles["cycle_status"].eq("valid")
    return set(cycles.loc[mask, "cycle_id"])


def _future_association(
    frame: pd.DataFrame,
    cycles: pd.DataFrame,
    channel: str,
    settings: Any,
    *,
    respect_cycle_status: bool,
) -> tuple[int, float]:
    residual = f"{channel}__baseline_residual"
    target = str(settings.performance_target)
    if residual not in frame or target not in frame:
        return 0, np.nan
    candidate_quality = imputed_column_for_value(residual)
    target_quality = imputed_column_for_value(target)
    eligible = _eligible_cycle_ids(cycles, respect_cycle_status=respect_cycle_status)
    development = frame.loc[
        frame["cycle_id"].isin(eligible) & frame["cycle_stage"].eq("frost_development")
    ]
    horizon = pd.Timedelta(minutes=settings.future_horizon_minutes)
    effects: list[float] = []
    for _, group in development.groupby(["experiment_id", "cycle_id"], sort=False):
        target_by_time = pd.Series(
            pd.to_numeric(group[target], errors="coerce").to_numpy(),
            index=pd.DatetimeIndex(group["timestamp"]),
        )
        target_imputed_by_time = pd.Series(
            _quality_mask(group, target_quality).to_numpy(),
            index=pd.DatetimeIndex(group["timestamp"]),
        )
        future_timestamps = group["timestamp"] + horizon
        future = future_timestamps.map(target_by_time)
        future_imputed = future_timestamps.map(target_imputed_by_time).astype("boolean")
        future_imputed = future_imputed.fillna(False)
        observed = (
            ~_quality_mask(group, candidate_quality)
            & future.notna()
            & ~future_imputed
        )
        correlation = _spearman_with_minimum(
            group.loc[observed, residual], future.loc[observed], settings.minimum_points_per_cycle
        )
        if correlation is not None:
            effects.append(correlation)
    return len(effects), _median_or_nan(effects)


def _context_association(
    frame: pd.DataFrame,
    cycles: pd.DataFrame,
    channel: str,
    channels: Mapping[str, Mapping[str, Any]],
    settings: Any,
    *,
    respect_cycle_status: bool,
) -> tuple[int, float]:
    residual = f"{channel}__baseline_residual"
    context_names = [
        name
        for name, channel_settings in channels.items()
        if channel_settings.get("role") == "context" and name in frame
    ]
    if residual not in frame or not context_names:
        return 0, np.nan
    candidate_quality = imputed_column_for_value(residual)
    eligible = _eligible_cycle_ids(cycles, respect_cycle_status=respect_cycle_status)
    development = frame.loc[
        frame["cycle_id"].isin(eligible) & frame["cycle_stage"].eq("frost_development")
    ]
    cycle_maxima: list[float] = []
    for _, group in development.groupby(["experiment_id", "cycle_id"], sort=False):
        candidate_observed = ~_quality_mask(group, candidate_quality)
        associations = [
            abs(correlation)
            for context in context_names
            if (
                correlation := _spearman_with_minimum(
                    group.loc[
                        candidate_observed & ~_quality_mask(
                            group, imputed_column_for_value(context)
                        ),
                        residual,
                    ],
                    group.loc[
                        candidate_observed & ~_quality_mask(
                            group, imputed_column_for_value(context)
                        ),
                        context,
                    ],
                    settings.minimum_points_per_cycle,
                )
            )
            is not None
        ]
        if associations:
            cycle_maxima.append(max(associations))
    return len(cycle_maxima), _median_or_nan(cycle_maxima)


def _decision(
    trend_count: int,
    trend_effect: float,
    direction: float,
    settings: Any,
) -> tuple[str, str]:
    if trend_count < settings.minimum_valid_cycles:
        return "insufficient_coverage", "trend_cycles_below_minimum"
    if (
        np.isfinite(trend_effect)
        and trend_effect >= settings.minimum_trend_effect
        and np.isfinite(direction)
        and direction >= settings.minimum_direction_consistency
    ):
        return "trend_supported_candidate", "trend_evidence_meets_threshold"
    return "partial_evidence", "trend_evidence_partial"


def _direction_consistency(effects: list[float]) -> float:
    if not effects:
        return np.nan
    return float(np.mean([effect > 0 for effect in effects]))


def _spearman_with_minimum(
    left: pd.Series, right: pd.Series, minimum_points: int
) -> float | None:
    x = pd.to_numeric(left, errors="coerce")
    y = pd.to_numeric(right, errors="coerce")
    valid = x.notna() & y.notna()
    if int(valid.sum()) < minimum_points:
        return None
    x = x.loc[valid]
    y = y.loc[valid]
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return None
    value = x.corr(y, method="spearman")
    return None if pd.isna(value) else float(value)


def _median_or_nan(values: list[float]) -> float:
    return float(np.median(values)) if values else np.nan


def _quality_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("boolean").fillna(False).astype(bool)


def run_analysis(
    loader: DatasetLoader,
    *,
    statuses: set[str] | None = None,
    experiment_ids: set[str] | None = None,
    cycle_names: set[str] | None = None,
    output_dir: Path,
) -> Path:
    """Run the scientific evidence analysis from DatasetLoader-selected cycles."""
    from .dataset_loader import DatasetLoader
    from .io import ensure_output_outside_input

    if not isinstance(loader, DatasetLoader):
        raise TypeError("run_analysis requires DatasetLoader")
    ensure_output_outside_input(output_dir, loader.dataset_root)
    selected = loader.list_cycles(statuses=statuses, experiment_ids=experiment_ids)
    if cycle_names is not None:
        selected = selected.loc[selected["cycle_name"].isin(cycle_names)]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    statistics: list[dict[str, object]] = []
    image_statistics: list[dict[str, object]] = []
    for cycle_name in selected["cycle_name"].astype(str):
        record = loader.get_cycle_record(cycle_name)
        frame = loader.load_cycle(cycle_name)
        images = loader.load_cycle_images(cycle_name)
        statistics.extend(_cycle_numeric_statistics(frame, record))
        image_statistics.append(
            {
                "cycle_name": cycle_name,
                "status": _assessment_status(record),
                "image_count": len(images),
                "camera_role_count": images["camera_role"].nunique()
                if not images.empty
                else 0,
            }
        )
    if loader.schema_version == 3:
        evidence = _analyze_dataset_cycles(loader, selected)
        evidence.to_csv(output_dir / "candidate_channel_evidence.csv", index=False)
    pd.DataFrame(statistics).to_csv(output_dir / "cycle_statistics.csv", index=False)
    pd.DataFrame(image_statistics).to_csv(
        output_dir / "image_sensor_alignment.csv", index=False
    )
    return output_dir


def _analyze_dataset_cycles(
    loader: DatasetLoader, selected: pd.DataFrame
) -> pd.DataFrame:
    """Adapt canonical Dataset tables to the existing evidence function."""
    registry = loader.registry
    raw_channels = registry.get("channels", {})
    raw_settings = registry.get("analysis_settings", {})
    if not isinstance(raw_channels, Mapping):
        raise ValueError("Dataset registry channels must be a mapping")
    if not isinstance(raw_settings, Mapping):
        raise ValueError("Dataset registry analysis_settings must be a mapping")
    channels = {
        str(name): dict(value)
        for name, value in raw_channels.items()
        if isinstance(value, Mapping)
    }
    if not any(bool(value.get("analysis_candidate", False)) for value in channels.values()):
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)
    from .config import AnalysisSettings

    settings = AnalysisSettings.from_mapping(raw_settings)
    frames = [loader.load_cycle(str(name)) for name in selected["cycle_name"].astype(str)]
    if not frames:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)
    processed = pd.concat(frames, ignore_index=True)
    summary = selected.copy()
    summary = summary.rename(columns={"cycle_name": "dataset_cycle_name"})
    if "experiment_date" not in summary:
        summary["experiment_date"] = processed["experiment_date"].iloc[0]
    config = SimpleNamespace(
        analysis=settings,
        experiment_id=str(processed["experiment_id"].iloc[0]),
        experiment_date=str(processed["experiment_date"].iloc[0]),
    )
    return analyze(
        processed,
        summary,
        config,
        channels,
        respect_cycle_status=False,
    )


def _cycle_numeric_statistics(
    frame: pd.DataFrame,
    record: Mapping[str, object],
) -> list[dict[str, object]]:
    cycle_name = str(record["cycle_name"])
    rows: list[dict[str, object]] = []
    for column in frame.select_dtypes(include="number").columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        rows.append(
            {
                "cycle_name": cycle_name,
                "column": str(column),
                "row_count": int(len(values)),
                "mean": float(values.mean()) if not values.empty else np.nan,
                "minimum": float(values.min()) if not values.empty else np.nan,
                "maximum": float(values.max()) if not values.empty else np.nan,
            }
        )
    return rows


def _assessment_status(record: Mapping[str, object]) -> str | None:
    assessment = record.get("assessment")
    if isinstance(assessment, dict):
        value = assessment.get("status")
        return None if value is None else str(value)
    return None


def _require_analysis_quality_columns(
    frame: pd.DataFrame,
    candidates: list[str],
    channels: Mapping[str, Mapping[str, Any]],
    settings: Any,
) -> None:
    missing_values: set[str] = set()
    required: set[str] = set()
    for candidate in candidates:
        residual = f"{candidate}__baseline_residual"
        if residual not in frame:
            missing_values.add(residual)
        else:
            required.add(imputed_column_for_value(residual))
    target = str(settings.performance_target)
    if target not in frame:
        missing_values.add(target)
    else:
        required.add(imputed_column_for_value(target))
    for name, channel_settings in channels.items():
        if channel_settings.get("role") == "context" and name in frame:
            required.add(imputed_column_for_value(name))
    missing = sorted((*missing_values, *(column for column in required if column not in frame)))
    if missing:
        raise ValueError(f"analysis requires value and quality columns: {missing}")
