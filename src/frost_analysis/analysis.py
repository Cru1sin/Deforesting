"""Cycle-level candidate evidence without fixed weights or global ranking."""

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
    "future_performance_effect",
    "max_abs_context_spearman",
    "decision",
    "reason",
]


def analyze(
    processed: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    config: Any,
    channels: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Compute one transparent evidence row per experiment and candidate channel."""
    candidates = [
        name
        for name, settings in channels.items()
        if bool(settings.get("analysis_candidate", False))
    ]
    if not candidates:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)
    rows: list[dict[str, object]] = []
    experiments = (
        processed[["experiment_id", "experiment_date"]]
        .drop_duplicates()
        .to_dict("records")
    )
    if not experiments:
        experiments = [
            {"experiment_id": config.experiment_id, "experiment_date": config.experiment_date}
        ]
    for experiment in experiments:
        experiment_id = str(experiment["experiment_id"])
        experiment_date = str(experiment["experiment_date"])
        current = processed.loc[processed["experiment_id"].eq(experiment_id)].copy()
        for channel in candidates:
            trend_effects = _trend_effects(current, channel)
            context_count, context_effect = _context_association(current, channel, channels)
            reset_count, reset_effect = _reset_evidence(
                current,
                cycle_summary.loc[cycle_summary["experiment_id"].eq(experiment_id)],
                channel,
                channels[channel],
                int(config.analysis.get("reset_pre_window_minutes", 5)),
            )
            future_count, future_effect = _future_evidence(
                current,
                channel,
                str(config.analysis.get("performance_target", "")),
                int(config.analysis.get("future_horizon_minutes", 10)),
            )
            trend_count = len(trend_effects)
            trend_effect = _median_or_nan(trend_effects)
            direction = _direction_consistency(trend_effects, trend_effect)
            decision, reason = _decision(
                trend_count,
                trend_effect,
                direction,
                context_effect,
                config.analysis,
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "experiment_date": experiment_date,
                    "channel": channel,
                    "trend_cycle_count": trend_count,
                    "reset_pair_count": reset_count,
                    "future_cycle_count": future_count,
                    "context_cycle_count": context_count,
                    "trend_effect": trend_effect,
                    "direction_consistency": direction,
                    "reset_effect": reset_effect,
                    "future_performance_effect": future_effect,
                    "max_abs_context_spearman": context_effect,
                    "decision": decision,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)


def _trend_effects(frame: pd.DataFrame, channel: str) -> list[float]:
    residual = f"{channel}__baseline_residual"
    if residual not in frame or "cycle_progress" not in frame:
        return []
    effects: list[float] = []
    development = frame.loc[frame["cycle_stage"].eq("frost_development")]
    for _, group in development.groupby(["experiment_id", "cycle_id"], sort=False):
        x = pd.to_numeric(group["cycle_progress"], errors="coerce")
        y = pd.to_numeric(group[residual], errors="coerce")
        correlation = _spearman(x, y)
        if correlation is not None:
            effects.append(correlation)
    return effects


def _reset_evidence(
    frame: pd.DataFrame,
    cycles: pd.DataFrame,
    channel: str,
    channel_settings: Mapping[str, Any],
    pre_window_minutes: int,
) -> tuple[int, float]:
    status = f"{channel}__baseline_status"
    if cycles.empty or f"{channel}__baseline_residual" not in frame or status not in frame:
        return 0, np.nan
    ordered = cycles.copy()
    ordered["heating_start"] = pd.to_datetime(ordered["heating_start"], errors="coerce")
    ordered = ordered.sort_values("heating_start", kind="stable").reset_index(drop=True)
    effects: list[float] = []
    residual = f"{channel}__baseline_residual"
    for index in range(len(ordered) - 1):
        current_cycle = ordered.iloc[index]
        next_cycle = ordered.iloc[index + 1]
        if pd.isna(current_cycle.get("defrost_start")):
            continue
        current_id = current_cycle["cycle_id"]
        next_id = next_cycle["cycle_id"]
        current_mask = frame["cycle_id"].eq(current_id)
        next_mask = frame["cycle_id"].eq(next_id)
        defrost_start = pd.Timestamp(current_cycle["defrost_start"])
        pre = frame.loc[
            current_mask
            & frame["cycle_stage"].eq("frost_development")
            & frame["timestamp"].between(
                defrost_start - pd.Timedelta(minutes=pre_window_minutes),
                defrost_start,
                inclusive="left",
            ),
            residual,
        ]
        post = frame.loc[
            next_mask
            & frame["cycle_stage"].eq("recovery")
            & frame[status].eq("accepted"),
            residual,
        ]
        if pre.empty or post.empty or not pre.notna().any() or not post.notna().any():
            continue
        raw_effect = float(post.median() - pre.median())
        direction = 1.0 if channel_settings.get("expected_frost_direction") == "decrease" else -1.0
        effects.append(raw_effect * direction)
    return len(effects), _median_or_nan(effects)


def _future_evidence(
    frame: pd.DataFrame, channel: str, target: str, horizon_minutes: int
) -> tuple[int, float]:
    if channel not in frame or target not in frame:
        return 0, np.nan
    effects: list[float] = []
    horizon = pd.Timedelta(minutes=horizon_minutes)
    development = frame.loc[frame["cycle_stage"].eq("frost_development")]
    for _, group in development.groupby(["experiment_id", "cycle_id"], sort=False):
        target_by_time = pd.Series(
            pd.to_numeric(group[target], errors="coerce").to_numpy(),
            index=pd.DatetimeIndex(group["timestamp"]),
        )
        current_values: list[float] = []
        future_values: list[float] = []
        for _, row in group.iterrows():
            current = pd.to_numeric(pd.Series([row[channel]]), errors="coerce").iloc[0]
            future = target_by_time.get(pd.Timestamp(row["timestamp"]) + horizon, np.nan)
            if pd.notna(current) and pd.notna(future):
                current_values.append(float(current))
                future_values.append(float(future))
        correlation = _spearman(pd.Series(current_values), pd.Series(future_values))
        if correlation is not None:
            effects.append(correlation)
    return len(effects), _median_or_nan(effects)


def _context_association(
    frame: pd.DataFrame,
    channel: str,
    channels: Mapping[str, Mapping[str, Any]],
) -> tuple[int, float]:
    context_names = [
        name
        for name, settings in channels.items()
        if settings.get("role") == "context" and name in frame
    ]
    residual = f"{channel}__baseline_residual"
    if residual not in frame or not context_names:
        return 0, np.nan
    associations: list[float] = []
    cycle_count = 0
    for _, group in frame.loc[frame["cycle_stage"].eq("frost_development")].groupby(
        ["experiment_id", "cycle_id"], sort=False
    ):
        cycle_associations: list[float] = []
        for context in context_names:
            correlation = _spearman(group[residual], group[context])
            if correlation is not None:
                cycle_associations.append(abs(correlation))
        if cycle_associations:
            cycle_count += 1
            associations.extend(cycle_associations)
    return cycle_count, max(associations, default=np.nan)


def _decision(
    trend_count: int,
    trend_effect: float,
    direction: float,
    context_effect: float,
    settings: Mapping[str, Any],
) -> tuple[str, str]:
    minimum_cycles = int(settings.get("minimum_valid_cycles", 3))
    if trend_count < minimum_cycles:
        return "insufficient_coverage", "trend_cycles_below_minimum"
    maximum_context = float(settings.get("maximum_context_association", 0.8))
    if np.isfinite(context_effect) and context_effect >= maximum_context:
        return "high_context_association", "context_association_above_threshold"
    minimum_effect = float(settings.get("minimum_absolute_trend_effect", 0.3))
    minimum_direction = float(settings.get("minimum_direction_consistency", 0.7))
    if (
        np.isfinite(trend_effect)
        and abs(trend_effect) >= minimum_effect
        and np.isfinite(direction)
        and direction >= minimum_direction
    ):
        return "trend_supported_candidate", "trend_evidence_meets_threshold"
    return "partial_evidence", "trend_evidence_partial"


def _direction_consistency(effects: list[float], overall: float) -> float:
    if not effects or not np.isfinite(overall) or overall == 0:
        return np.nan
    overall_sign = np.sign(overall)
    return float(np.mean([np.sign(effect) == overall_sign for effect in effects]))


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    x = pd.to_numeric(left, errors="coerce")
    y = pd.to_numeric(right, errors="coerce")
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 2:
        return None
    left = x.loc[valid]
    right = y.loc[valid]
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return None
    value = left.corr(right, method="spearman")
    return None if pd.isna(value) else float(value)


def _median_or_nan(values: list[float]) -> float:
    return float(np.median(values)) if values else np.nan
