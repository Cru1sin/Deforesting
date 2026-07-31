"""Transparent candidate evidence for valid, baseline-backed cycles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

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
) -> pd.DataFrame:
    """Compute one evidence row per experiment and configured candidate channel."""
    candidates = _candidate_names(channels)
    if not candidates:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)
    settings = _analysis_settings(config)
    _require_analysis_quality_columns(processed, candidates, channels, settings)
    experiments = _experiments(processed, config)
    rows: list[dict[str, object]] = []
    for experiment_id, experiment_date in experiments:
        current = processed.loc[processed["experiment_id"].eq(experiment_id)].copy()
        cycles = cycle_summary.loc[cycle_summary["experiment_id"].eq(experiment_id)].copy()
        for channel in candidates:
            trend = _trend_effects(current, cycles, channel, channels[channel], settings)
            context_count, context_effect = _context_association(
                current, cycles, channel, channels, settings
            )
            future_count, future_effect = _future_association(
                current, cycles, channel, settings
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
) -> list[float]:
    residual = f"{channel}__baseline_residual"
    if residual not in frame or "cycle_progress" not in frame:
        return []
    quality = imputed_column_for_value(residual)
    eligible = _eligible_cycle_ids(cycles)
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


def _eligible_cycle_ids(cycles: pd.DataFrame) -> set[object]:
    if "baseline_status" not in cycles:
        return set()
    return set(
        cycles.loc[
            cycles["cycle_status"].eq("valid") & cycles["baseline_status"].eq("available"),
            "cycle_id",
        ]
    )


def _future_association(
    frame: pd.DataFrame, cycles: pd.DataFrame, channel: str, settings: Any
) -> tuple[int, float]:
    residual = f"{channel}__baseline_residual"
    target = str(settings.performance_target)
    if residual not in frame or target not in frame:
        return 0, np.nan
    candidate_quality = imputed_column_for_value(residual)
    target_quality = imputed_column_for_value(target)
    eligible = _eligible_cycle_ids(cycles)
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
    eligible = _eligible_cycle_ids(cycles)
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


def _require_analysis_quality_columns(
    frame: pd.DataFrame,
    candidates: list[str],
    channels: Mapping[str, Mapping[str, Any]],
    settings: Any,
) -> None:
    required: set[str] = set()
    for candidate in candidates:
        residual = f"{candidate}__baseline_residual"
        if residual in frame:
            required.add(imputed_column_for_value(residual))
    target = str(settings.performance_target)
    if target in frame:
        required.add(imputed_column_for_value(target))
    for name, channel_settings in channels.items():
        if channel_settings.get("role") == "context" and name in frame:
            required.add(imputed_column_for_value(name))
    missing = sorted(column for column in required if column not in frame)
    if missing:
        raise ValueError(f"analysis requires quality columns: {missing}")
