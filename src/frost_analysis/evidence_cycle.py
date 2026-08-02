"""Cycle-local calculations for frost evidence.

This module owns the complete cycle grid and the distinction between raw
trend evidence and reference-dependent level/threshold evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation, spearmanr, theilslopes

from .config import EvidencePolicy

CYCLE_KEYS = ("experiment_id", "experiment_date", "cycle_id")
STAGE = "frost_development"
TREND_VARIANT = "past_slope_5min"


@dataclass(frozen=True)
class CycleSlice:
    key: tuple[str, str, str]
    frame: pd.DataFrame
    frost: pd.DataFrame
    grid: pd.DatetimeIndex
    summary: pd.Series
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    grid_coverage: float
    cycle_status: str
    cycle_status_reason: str | None
    eligible: bool
    exclusion_reason: str | None

    @property
    def eligibility_status(self) -> str:
        if not self.eligible:
            return "excluded"
        if self.cycle_status == "incomplete":
            return "eligible_exploratory"
        return "eligible"


@dataclass(frozen=True)
class ResolvedReference:
    residual: pd.Series
    source: str
    center: float
    scale: float
    observed_fraction: float
    valid_from: pd.Timestamp
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class CycleChannelEvidence:
    values: pd.Series
    imputed: pd.Series
    target_valid: pd.Series
    reference: ResolvedReference
    analysis_residual: pd.Series
    onset_elapsed_minutes: float
    onset_progress: float
    past_slope_5min: pd.Series | None


def duration_buckets(duration_seconds: int, grid_interval_seconds: int) -> int:
    if grid_interval_seconds <= 0:
        raise ValueError("grid_interval_seconds must be positive")
    if duration_seconds <= 0 or duration_seconds % grid_interval_seconds:
        raise ValueError(
            f"duration {duration_seconds}s must be a positive multiple of "
            f"grid interval {grid_interval_seconds}s"
        )
    return duration_seconds // grid_interval_seconds


def expected_grid(
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval_seconds: int,
) -> pd.DatetimeIndex:
    """Return Process fixed-bucket left labels fully contained in ``[start, end)``."""
    if interval_seconds <= 0 or end <= start:
        return pd.DatetimeIndex([])
    first = start.ceil(f"{interval_seconds}s")
    last = end - pd.Timedelta(seconds=interval_seconds)
    if last < first:
        return pd.DatetimeIndex([])
    return pd.date_range(first, last, freq=f"{interval_seconds}s")


def build_cycle_slices(
    processed: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    interval_seconds: int,
) -> list[CycleSlice]:
    summary_lookup: dict[tuple[str, str, str], pd.Series] = {}
    for _, row in cycle_summary.iterrows():
        key = cast(tuple[str, str, str], tuple(str(row[name]) for name in CYCLE_KEYS))
        if key in summary_lookup:
            raise ValueError(f"duplicate cycle summary key: {key}")
        summary_lookup[key] = row

    normalised_processed = processed.copy()
    for name in CYCLE_KEYS:
        normalised_processed[name] = normalised_processed[name].astype(str)
    processed_groups: dict[tuple[str, str, str], pd.DataFrame] = {}
    for values, group in normalised_processed.groupby(
        list(CYCLE_KEYS), sort=False, dropna=False
    ):
        key = cast(tuple[str, str, str], tuple(str(value) for value in values))
        processed_groups[key] = (
            group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        )
    empty_frame = processed.iloc[0:0].copy()
    keys = sorted(set(processed_groups) | set(summary_lookup))
    slices: list[CycleSlice] = []
    for key in keys:
        summary = summary_lookup.get(key, pd.Series(dtype=object))
        frame = processed_groups.get(key, empty_frame).copy()
        frost = frame.loc[frame["cycle_stage"].eq(STAGE)].copy()
        if frost["timestamp"].duplicated().any():
            raise ValueError(f"duplicate processed timestamp in cycle {key}")
        frost_times = pd.DatetimeIndex(pd.to_datetime(frost["timestamp"], errors="raise"))
        start = _timestamp(summary.get("stable_heating_start"))
        end = _timestamp(summary.get("defrost_start"))
        grid = expected_grid(start, end, interval_seconds) if start and end else pd.DatetimeIndex([])
        grid_available = not grid.empty
        actual = set(frost_times)
        grid_coverage = float(len(actual & set(grid)) / len(grid)) if len(grid) else 0.0
        status = _cycle_status(summary, frame)
        status_reason = _cycle_status_reason(summary, frame)
        boundary_complete = start is not None and end is not None and start < end
        boundary_mismatch = False
        grid_mismatch = False
        if len(frost_times) and boundary_complete:
            assert start is not None and end is not None
            boundary_mismatch = bool(
                (frost_times < start).any() or (frost_times >= end).any()
            )
            grid_mismatch = bool(
                grid_available
                and not boundary_mismatch
                and not frost_times.isin(grid).all()
            )
        eligible = bool(
            boundary_complete
            and grid_available
            and not frost.empty
            and not boundary_mismatch
            and not grid_mismatch
            and (
                status == "valid"
                or (status == "incomplete" and status_reason == "defrost_end_not_observed")
            )
        )
        exclusion: str | None = None
        if not eligible:
            exclusion = _cycle_exclusion_reason(
                frame,
                frost,
                status,
                status_reason,
                boundary_complete,
                grid_available,
                boundary_mismatch,
                grid_mismatch,
            )
        slices.append(
            CycleSlice(
                key=key,
                frame=frame,
                frost=frost,
                grid=grid,
                summary=summary,
                start=start,
                end=end,
                grid_coverage=grid_coverage,
                cycle_status=status,
                cycle_status_reason=status_reason,
                eligible=eligible,
                exclusion_reason=exclusion,
            )
        )
    return slices


def resolve_analysis_reference(
    frame: pd.DataFrame,
    *,
    channel: str,
    cycle_start: pd.Timestamp,
    formal_frost_start: pd.Timestamp,
    formal_frost_end: pd.Timestamp,
    configured_baseline_available: bool,
    configured_residual: pd.Series | None,
    configured_baseline_mask: pd.Series | None,
    window_minutes: int,
    minimum_observed_fraction: float,
    maximum_gap_seconds: float,
    interval_seconds: int,
    configured_baseline_end: pd.Timestamp | None = None,
    analysis_grid: pd.DatetimeIndex | None = None,
) -> ResolvedReference:
    """Resolve a configured or initial-grid reference.

    ``cycle_start`` is retained for public API compatibility and is not used
    by the current reference definition.
    """
    output_on_analysis_grid = analysis_grid is not None
    grid = analysis_grid
    if grid is None:
        grid = expected_grid(formal_frost_start, formal_frost_end, interval_seconds)
    if channel not in frame:
        return _unavailable(frame.index, "reference_channel_missing", formal_frost_start)
    values = pd.to_numeric(frame[channel], errors="coerce")
    imputed = _imputed_mask(frame, channel)
    if configured_baseline_available and configured_residual is not None:
        configured = _configured_reference(
            frame,
            configured_residual,
            configured_baseline_mask,
            formal_frost_start,
            interval_seconds,
            grid,
            output_on_analysis_grid,
            imputed,
            configured_baseline_end,
        )
        if configured is not None:
            return configured
    return _auto_reference(
        frame,
        channel,
        formal_frost_start,
        formal_frost_end,
        window_minutes,
        minimum_observed_fraction,
        maximum_gap_seconds,
        interval_seconds,
        grid,
        output_on_analysis_grid,
        values,
    )


def build_channel_evidence(
    cycle: CycleSlice,
    channel: str,
    policy: EvidencePolicy,
    *,
    target: bool,
    interval_seconds: int,
) -> CycleChannelEvidence:
    grid = cycle.grid
    values = _grid_values(cycle.frost, channel, grid)
    imputed = _grid_imputed(cycle.frost, channel, grid)
    target_valid = _grid_target_valid(cycle.frost, channel, grid)
    if not cycle.eligible:
        reference = _unavailable(
            grid,
            cycle.exclusion_reason or "cycle_not_eligible",
            cycle.start if cycle.start is not None else pd.NaT,
        )
    elif cycle.start is None or cycle.end is None or grid.empty:
        reference = _unavailable(grid, "frost_development_unavailable", pd.NaT)
    else:
        reference = _resolve_cycle_reference(cycle, channel, policy, interval_seconds)
    residual = reference.residual.reindex(grid).astype(float)
    if reference.source != "unavailable":
        residual.loc[grid < reference.valid_from] = np.nan
    else:
        residual[:] = np.nan
    onset_values = residual.where(target_valid) if target else residual
    onset_elapsed, onset_progress = _onset(
        grid,
        onset_values,
        cycle.start,
        cycle.end,
        reference,
        policy,
        interval_seconds,
    )
    slope = None if target else _past_slope(values, grid, interval_seconds)
    return CycleChannelEvidence(
        values=values,
        imputed=imputed,
        target_valid=target_valid,
        reference=reference,
        analysis_residual=residual,
        onset_elapsed_minutes=onset_elapsed,
        onset_progress=onset_progress,
        past_slope_5min=slope,
    )


def feature_metric_record(
    cycle: CycleSlice,
    feature: str,
    registry: Mapping[str, Any],
    evidence: CycleChannelEvidence,
    policy: EvidencePolicy,
    interval_seconds: int,
) -> dict[str, object]:
    record: dict[str, object] = {
        "experiment_id": cycle.key[0],
        "experiment_date": cycle.key[1],
        "cycle_id": cycle.key[2],
        "feature": feature,
        "registry_role": registry.get("role", ""),
        "expected_frost_direction": registry.get("expected_frost_direction", pd.NA),
        "cycle_eligible": cycle.eligible,
        "metric_status": "unavailable",
        "metric_exclusion_reason": cycle.exclusion_reason or "",
        "observed_fraction": np.nan,
        "imputed_fraction": np.nan,
        "maximum_consecutive_gap_seconds": np.nan,
        "reference_source": evidence.reference.source,
        "reference_exclusion_reason": evidence.reference.exclusion_reason or "",
        "reference_center": evidence.reference.center,
        "reference_scale": evidence.reference.scale,
        "reference_observed_fraction": evidence.reference.observed_fraction,
        "reference_valid_from": evidence.reference.valid_from,
        "early_observed_fraction": np.nan,
        "middle_observed_fraction": np.nan,
        "late_observed_fraction": np.nan,
        "signed_sensitivity": np.nan,
        "absolute_sensitivity": np.nan,
        "onset_elapsed_minutes": evidence.onset_elapsed_minutes,
        "onset_progress": evidence.onset_progress,
        "global_spearman": np.nan,
        "early_slope_per_min": np.nan,
        "middle_slope_per_min": np.nan,
        "late_slope_per_min": np.nan,
        "late_minus_early_slope": np.nan,
    }
    if not cycle.eligible or cycle.grid.empty:
        return record

    values = evidence.values
    real = values.notna() & ~evidence.imputed
    usable = values.notna()
    record["observed_fraction"] = float(real.mean())
    record["imputed_fraction"] = float(evidence.imputed.mean())
    record["maximum_consecutive_gap_seconds"] = _maximum_false_run_seconds(
        values.notna().to_numpy(dtype=bool), interval_seconds
    )
    progress = _progress(cycle.grid, cycle.start, cycle.end)
    for name in ("early", "middle", "late"):
        mask = _segment_mask(progress, name)
        record[f"{name}_observed_fraction"] = _coverage(real, mask)
        record[f"{name}_slope_per_min"] = _segment_slope(
            cycle.grid, values, progress, name, policy
        )
    record["global_spearman"] = _global_spearman(
        cycle.grid, values, policy, interval_seconds
    )
    sensitivity = _sensitivity(cycle, evidence, progress, policy)
    record["signed_sensitivity"] = sensitivity
    record["absolute_sensitivity"] = abs(sensitivity) if _finite_number(sensitivity) else np.nan
    early = record["early_slope_per_min"]
    late = record["late_slope_per_min"]
    if _finite_number(early) and _finite_number(late):
        record["late_minus_early_slope"] = float(cast(Any, late)) - float(cast(Any, early))
    metric_values = [
        record["global_spearman"],
        record["early_slope_per_min"],
        record["middle_slope_per_min"],
        record["late_slope_per_min"],
        record["signed_sensitivity"],
        record["onset_elapsed_minutes"],
    ]
    if any(_finite_number(value) for value in metric_values):
        record["metric_status"] = "available"
        record["metric_exclusion_reason"] = ""
    elif not usable.any():
        record["metric_exclusion_reason"] = "insufficient_feature_data"
    else:
        record["metric_exclusion_reason"] = "no_metric_available"
    return record


def future_records(
    cycle: CycleSlice,
    feature: str,
    target: str,
    feature_cache: CycleChannelEvidence,
    target_cache: CycleChannelEvidence,
    policy: EvidencePolicy,
    interval_seconds: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for variant in ("residual_level", TREND_VARIANT):
        for horizon in policy.horizons_minutes:
            horizon_buckets = duration_buckets(horizon * 60, interval_seconds)
            for target_type in ("future_level", "future_change"):
                records.append(
                    _future_record(
                        cycle,
                        feature,
                        target,
                        variant,
                        horizon,
                        target_type,
                        feature_cache,
                        target_cache,
                        policy,
                        horizon_buckets,
                    )
                )
    return records


def _future_record(
    cycle: CycleSlice,
    feature: str,
    target: str,
    variant: str,
    horizon: int,
    target_type: str,
    feature_cache: CycleChannelEvidence,
    target_cache: CycleChannelEvidence,
    policy: EvidencePolicy,
    horizon_buckets: int,
) -> dict[str, object]:
    feature_reference = feature_cache.reference
    target_reference = target_cache.reference
    feature_source = (
        feature_reference.source if variant == "residual_level" else "not_required"
    )
    target_source = target_reference.source if target_type == "future_level" else "not_required"
    base = {
        "experiment_id": cycle.key[0],
        "experiment_date": cycle.key[1],
        "cycle_id": cycle.key[2],
        "feature": feature,
        "feature_variant": variant,
        "feature_reference_source": feature_source,
        "target": target,
        "target_reference_source": target_source,
        "horizon_minutes": horizon,
        "target_type": target_type,
        "effect_metric": "within_cycle_spearman",
        "effect": np.nan,
        "expected_anchor_count": 0,
        "valid_pairs": 0,
        "pair_coverage": np.nan,
        "lead_time_minutes": np.nan,
        "lead_time_status": "not_applicable",
        "metric_status": "unavailable",
        "exclusion_reason": cycle.exclusion_reason or "cycle_not_eligible",
    }
    if not cycle.eligible or cycle.grid.empty:
        return base

    grid = cycle.grid
    future_positions = np.arange(len(grid)) + horizon_buckets
    in_grid = future_positions < len(grid)
    anchor = pd.Series(in_grid, index=grid)
    if variant == "residual_level":
        anchor &= grid >= feature_reference.valid_from
        feature_values = feature_cache.analysis_residual
        if feature_reference.source == "unavailable":
            return _with_future_reason(base, "unavailable", "feature_reference_unavailable")
    else:
        anchor &= grid >= grid[0] + pd.Timedelta(seconds=5 * 60)
        feature_values = feature_cache.past_slope_5min
    if target_type == "future_level":
        if target_reference.source == "unavailable":
            return _with_future_reason(base, "unavailable", "target_reference_unavailable")
        future_times = _shift_grid(grid, horizon_buckets)
        anchor &= future_times >= target_reference.valid_from
        outcome = target_cache.analysis_residual.shift(-horizon_buckets)
        target_pair_valid = target_cache.target_valid.shift(-horizon_buckets).fillna(False)
    else:
        current = target_cache.values
        outcome = current.shift(-horizon_buckets) - current
        target_pair_valid = (
            target_cache.target_valid
            & target_cache.target_valid.shift(-horizon_buckets).fillna(False)
        )
    expected = int(anchor.sum())
    valid_feature = feature_values.notna() if feature_values is not None else pd.Series(False, index=grid)
    valid = anchor & valid_feature & target_pair_valid & outcome.notna()
    valid_pairs = int(valid.sum())
    coverage = valid_pairs / expected if expected else np.nan
    effect = np.nan
    status = "unavailable"
    reason = "no_structural_anchors"
    if expected:
        if valid_pairs < policy.min_valid_pairs:
            status, reason = "insufficient_pairs", "valid_pairs_below_minimum"
        elif coverage < policy.min_pair_coverage:
            status, reason = "insufficient_coverage", "pair_coverage_below_minimum"
        else:
            effect = _spearman(feature_values.loc[valid], outcome.loc[valid], policy.min_valid_pairs)
            if np.isfinite(effect):
                status, reason = "available", ""
            else:
                status, reason = "not_evaluated", "zero_variability"
    base.update(
        {
            "effect": effect,
            "expected_anchor_count": expected,
            "valid_pairs": valid_pairs,
            "pair_coverage": coverage,
            "metric_status": status,
            "exclusion_reason": reason,
        }
    )
    if (
        variant == "residual_level"
        and target_type == "future_level"
        and horizon == policy.primary_horizon_minutes
    ):
        base["lead_time_minutes"], base["lead_time_status"] = _lead_time(
            feature_cache,
            target_cache,
        )
    return base


def _lead_time(
    feature_cache: CycleChannelEvidence,
    target_cache: CycleChannelEvidence,
) -> tuple[float, str]:
    feature_onset = feature_cache.onset_elapsed_minutes
    target_onset = target_cache.onset_elapsed_minutes
    if np.isfinite(feature_onset) and np.isfinite(target_onset):
        return float(target_onset - feature_onset), "available"
    if feature_cache.reference.source == "unavailable":
        return np.nan, "feature_onset_unavailable"
    if target_cache.reference.source == "unavailable":
        return np.nan, "target_reference_unavailable"
    return np.nan, "onset_unavailable"


def _resolve_cycle_reference(
    cycle: CycleSlice,
    channel: str,
    policy: EvidencePolicy,
    interval_seconds: int,
) -> ResolvedReference:
    baseline_status = str(cycle.summary.get("baseline_status", ""))
    residual_name = f"{channel}__baseline_residual"
    baseline_mask = _baseline_window_mask(cycle.frame, cycle.summary)
    return resolve_analysis_reference(
        cycle.frame,
        channel=channel,
        cycle_start=cycle.start or pd.Timestamp("NaT"),
        formal_frost_start=cycle.start or pd.Timestamp("NaT"),
        formal_frost_end=cycle.end or pd.Timestamp("NaT"),
        configured_baseline_available=baseline_status == "available" and residual_name in cycle.frame,
        configured_residual=cycle.frame.get(residual_name),
        configured_baseline_mask=baseline_mask,
        window_minutes=policy.auto_reference_window_minutes,
        minimum_observed_fraction=policy.auto_reference_min_observed_fraction,
        maximum_gap_seconds=policy.auto_reference_max_gap_seconds,
        interval_seconds=interval_seconds,
        configured_baseline_end=_timestamp(cycle.summary.get("baseline_end")),
        analysis_grid=cycle.grid,
    )


def _configured_reference(
    frame: pd.DataFrame,
    configured_residual: pd.Series,
    baseline_mask: pd.Series | None,
    formal_start: pd.Timestamp,
    interval_seconds: int,
    grid: pd.DatetimeIndex,
    output_on_analysis_grid: bool,
    imputed: pd.Series,
    configured_baseline_end: pd.Timestamp | None,
) -> ResolvedReference | None:
    if baseline_mask is None:
        return None
    source = pd.to_numeric(configured_residual, errors="coerce").reindex(frame.index)
    mask = baseline_mask.reindex(frame.index).fillna(False).astype(bool)
    observed = source.notna() & ~imputed & mask
    expected = int(mask.sum())
    center, scale, fraction = _reference_statistics(source.loc[observed], expected)
    if center is None:
        return None
    residual = source - center
    grid_residual = _timestamp_indexed_series(frame, residual).reindex(grid)
    valid_from = _configured_valid_from(
        frame,
        mask,
        formal_start,
        interval_seconds,
        grid,
        configured_baseline_end,
    )
    return ResolvedReference(
        residual=grid_residual if output_on_analysis_grid else residual,
        source="configured_baseline",
        center=center,
        scale=scale,
        observed_fraction=fraction,
        valid_from=valid_from,
    )


def _auto_reference(
    frame: pd.DataFrame,
    channel: str,
    formal_start: pd.Timestamp,
    formal_end: pd.Timestamp,
    window_minutes: int,
    minimum_observed_fraction: float,
    maximum_gap_seconds: float,
    interval_seconds: int,
    grid: pd.DatetimeIndex,
    output_on_analysis_grid: bool,
    values: pd.Series,
) -> ResolvedReference:
    count = duration_buckets(window_minutes * 60, interval_seconds)
    if len(grid) < count:
        return _unavailable(frame.index, "reference_window_outside_frost_development", formal_start)
    reference_grid = grid[:count]
    window_values = _grid_values(frame, channel, reference_grid)
    window_imputed = _grid_imputed(frame, channel, reference_grid)
    observed = window_values.notna() & ~window_imputed
    observed_count = int(observed.sum())
    fraction = observed_count / count
    maximum_gap = _maximum_false_run_seconds(observed.to_numpy(dtype=bool), interval_seconds)
    if observed_count < ceil(count * minimum_observed_fraction):
        return _unavailable(frame.index, "reference_observed_coverage", formal_start)
    if fraction < minimum_observed_fraction:
        return _unavailable(frame.index, "reference_observed_fraction", formal_start)
    if maximum_gap > maximum_gap_seconds:
        return _unavailable(frame.index, "reference_observation_gap", formal_start)
    center, scale, _ = _reference_statistics(window_values.loc[observed], count)
    if center is None:
        return _unavailable(frame.index, "reference_values_empty", formal_start)
    valid_from = grid[count] if len(grid) > count else formal_end
    residual = values - center
    grid_residual = _grid_values(frame, channel, grid) - center
    return ResolvedReference(
        residual=grid_residual if output_on_analysis_grid else residual,
        source="auto_cycle_initial_reference",
        center=center,
        scale=scale,
        observed_fraction=fraction,
        valid_from=valid_from,
    )


def _configured_valid_from(
    frame: pd.DataFrame,
    baseline_mask: pd.Series,
    formal_start: pd.Timestamp,
    interval_seconds: int,
    grid: pd.DatetimeIndex,
    configured_baseline_end: pd.Timestamp | None,
) -> pd.Timestamp:
    if grid.empty:
        return formal_start
    cutoff = configured_baseline_end
    if cutoff is None:
        selected = pd.to_datetime(frame.loc[baseline_mask, "timestamp"], errors="coerce").dropna()
        if selected.empty:
            return grid[0]
        cutoff = selected.max() + pd.Timedelta(seconds=interval_seconds)
    if cutoff <= formal_start:
        return grid[0]
    positions = np.flatnonzero(grid >= cutoff)
    return grid[int(positions[0])] if len(positions) else cutoff


def _onset(
    grid: pd.DatetimeIndex,
    residual: pd.Series,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    reference: ResolvedReference,
    policy: EvidencePolicy,
    interval_seconds: int,
) -> tuple[float, float]:
    if (
        reference.source == "unavailable"
        or not np.isfinite(reference.scale)
        or reference.scale <= 0
        or start is None
        or end is None
        or grid.empty
    ):
        return np.nan, np.nan
    window_buckets = duration_buckets(policy.onset_window_seconds, interval_seconds)
    persistence_buckets = duration_buckets(policy.onset_persistence_seconds, interval_seconds)
    series = residual.reindex(grid).astype(float)
    series.loc[grid < reference.valid_from] = np.nan
    smoothed = series.rolling(window_buckets, min_periods=window_buckets).median()
    qualifying = smoothed.abs().gt(policy.onset_mad_multiplier * reference.scale).fillna(False)
    for end_index in range(persistence_buckets - 1, len(grid)):
        if not bool(qualifying.iloc[end_index - persistence_buckets + 1 : end_index + 1].all()):
            continue
        onset_time = grid[end_index - persistence_buckets + 1]
        elapsed = (onset_time - start).total_seconds() / 60
        progress = (onset_time - start).total_seconds() / (end - start).total_seconds()
        return float(elapsed), float(progress)
    return np.nan, np.nan


def _past_slope(
    values: pd.Series,
    grid: pd.DatetimeIndex,
    interval_seconds: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=grid, dtype=float)
    window = duration_buckets(5 * 60, interval_seconds)
    for index in range(window, len(grid)):
        segment = values.iloc[index - window + 1 : index + 1]
        if len(segment) != window or segment.isna().any() or segment.nunique() < 2:
            continue
        x = np.arange(window, dtype=float) * interval_seconds / 60
        result.iloc[index] = float(np.polyfit(x, segment.to_numpy(dtype=float), 1)[0])
    return result


def _global_spearman(
    grid: pd.DatetimeIndex,
    values: pd.Series,
    policy: EvidencePolicy,
    interval_seconds: int,
) -> float:
    valid = values.notna()
    expected = len(grid)
    count = int(valid.sum())
    if expected == 0 or count < policy.min_segment_points or count / expected < policy.min_segment_coverage:
        return np.nan
    x = pd.Series(
        np.arange(expected, dtype=float) * interval_seconds / 60,
        index=grid,
    )
    return _spearman(x.loc[valid], values.loc[valid], policy.min_segment_points)


def _segment_slope(
    grid: pd.DatetimeIndex,
    values: pd.Series,
    progress: pd.Series,
    name: str,
    policy: EvidencePolicy,
) -> float:
    mask = _segment_mask(progress, name)
    expected = int(mask.sum())
    valid = mask & values.notna()
    count = int(valid.sum())
    if expected == 0 or count < policy.min_segment_points or count / expected < policy.min_segment_coverage:
        return np.nan
    x = (grid[valid.to_numpy()] - grid[valid.to_numpy()][0]).total_seconds() / 60
    y = values.loc[valid].to_numpy(dtype=float)
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return float(theilslopes(y, x).slope)


def _sensitivity(
    cycle: CycleSlice,
    evidence: CycleChannelEvidence,
    progress: pd.Series,
    policy: EvidencePolicy,
) -> float:
    reference = evidence.reference
    if reference.source == "unavailable" or not np.isfinite(reference.scale) or reference.scale <= 0:
        return np.nan
    after_reference = pd.Series(cycle.grid >= reference.valid_from, index=cycle.grid)
    late = _segment_mask(progress, "late") & after_reference
    valid = late & evidence.analysis_residual.notna()
    expected = int(_segment_mask(progress, "late").sum())
    count = int(valid.sum())
    if expected == 0 or count < policy.min_segment_points or count / expected < policy.min_segment_coverage:
        return np.nan
    return float(evidence.analysis_residual.loc[valid].median() / reference.scale)


def _grid_values(frame: pd.DataFrame, channel: str, grid: pd.DatetimeIndex) -> pd.Series:
    if channel not in frame:
        return pd.Series(np.nan, index=grid, dtype=float)
    indexed = frame.set_index("timestamp")
    return pd.to_numeric(indexed[channel], errors="coerce").reindex(grid).astype(float)


def _timestamp_indexed_series(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    return pd.Series(
        pd.to_numeric(values.reindex(frame.index), errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(frame["timestamp"]),
        dtype=float,
    )


def _grid_imputed(frame: pd.DataFrame, channel: str, grid: pd.DatetimeIndex) -> pd.Series:
    column = f"{channel}__imputed"
    if column not in frame:
        return pd.Series(False, index=grid, dtype=bool)
    indexed = frame.set_index("timestamp")
    return indexed[column].astype("boolean").fillna(False).astype(bool).reindex(grid, fill_value=False)


def _grid_target_valid(frame: pd.DataFrame, channel: str, grid: pd.DatetimeIndex) -> pd.Series:
    result = _grid_values(frame, channel, grid).notna() & ~_grid_imputed(frame, channel, grid)
    indexed = frame.set_index("timestamp")
    for suffix in ("__quality_valid", "__valid"):
        column = f"{channel}{suffix}"
        if column in frame:
            result &= indexed[column].astype("boolean").fillna(False).astype(bool).reindex(
                grid, fill_value=False
            )
    return result


def _baseline_window_mask(frame: pd.DataFrame, summary: pd.Series) -> pd.Series | None:
    start = _timestamp(summary.get("baseline_start"))
    end = _timestamp(summary.get("baseline_end"))
    if start is None or end is None:
        return None
    return frame["timestamp"].ge(start) & frame["timestamp"].lt(end)


def _progress(
    grid: pd.DatetimeIndex,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.Series:
    if start is None or end is None or end <= start:
        return pd.Series(np.nan, index=grid, dtype=float)
    return pd.Series(
        (grid - start).total_seconds() / (end - start).total_seconds(),
        index=grid,
        dtype=float,
    )


def _segment_mask(progress: pd.Series, name: str) -> pd.Series:
    if name == "early":
        return progress.ge(0) & progress.lt(0.25)
    if name == "middle":
        return progress.ge(0.25) & progress.lt(0.75)
    return progress.ge(0.75) & progress.lt(1.0)


def _coverage(values: pd.Series, mask: pd.Series) -> float:
    expected = int(mask.sum())
    return float((values & mask).sum() / expected) if expected else 0.0


def _spearman(left: pd.Series, right: pd.Series, minimum_points: int) -> float:
    valid = left.notna() & right.notna()
    if int(valid.sum()) < minimum_points:
        return np.nan
    x = pd.to_numeric(left.loc[valid], errors="coerce")
    y = pd.to_numeric(right.loc[valid], errors="coerce")
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    value = spearmanr(x.to_numpy(dtype=float), y.to_numpy(dtype=float)).statistic
    return float(value) if np.isfinite(value) else np.nan


def _reference_statistics(values: pd.Series, expected: int) -> tuple[float | None, float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty or expected <= 0:
        return None, np.nan, 0.0
    center = float(numeric.median())
    scale = float(
        median_abs_deviation(numeric.to_numpy(dtype=float), scale="normal", nan_policy="omit")
    )
    return center, scale, float(len(numeric) / expected)


def _imputed_mask(frame: pd.DataFrame, channel: str) -> pd.Series:
    column = f"{channel}__imputed"
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].astype("boolean").fillna(False).astype(bool)


def _maximum_false_run_seconds(values: np.ndarray[Any, Any], interval_seconds: int) -> float:
    maximum = 0
    current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return float(maximum * interval_seconds)


def _finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(cast(Any, value))))
    except (TypeError, ValueError):
        return False


def _unavailable(
    index: pd.Index | pd.DatetimeIndex,
    reason: str,
    valid_from: pd.Timestamp,
) -> ResolvedReference:
    return ResolvedReference(
        residual=pd.Series(np.nan, index=index, dtype=float),
        source="unavailable",
        center=np.nan,
        scale=np.nan,
        observed_fraction=0.0,
        valid_from=valid_from,
        exclusion_reason=reason,
    )


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value)


def _cycle_status(summary: pd.Series, frame: pd.DataFrame) -> str:
    if "cycle_status" in summary.index and pd.notna(summary.get("cycle_status")):
        return str(summary.get("cycle_status"))
    if "cycle_status" in frame and not frame.empty:
        return str(frame["cycle_status"].iloc[0])
    return "invalid"


def _cycle_status_reason(summary: pd.Series, frame: pd.DataFrame) -> str | None:
    value: Any = summary.get("cycle_status_reason") if "cycle_status_reason" in summary.index else None
    if value is None or pd.isna(value) or not str(value).strip():
        if "cycle_status_reason" in frame and not frame.empty:
            value = frame["cycle_status_reason"].iloc[0]
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    return str(value)


def _cycle_exclusion_reason(
    frame: pd.DataFrame,
    frost: pd.DataFrame,
    status: str,
    status_reason: str | None,
    boundary_complete: bool,
    grid_available: bool,
    boundary_mismatch: bool,
    grid_mismatch: bool,
) -> str:
    if frame.empty:
        return "processed_cycle_unavailable"
    if boundary_mismatch:
        return "frost_stage_boundary_mismatch"
    if boundary_complete and not grid_available:
        return "no_complete_frost_grid_bucket"
    if grid_mismatch:
        return "frost_stage_grid_mismatch"
    if status == "invalid":
        return "cycle_invalid"
    if status == "incomplete" and status_reason != "defrost_end_not_observed":
        return status_reason or "cycle_incomplete"
    if not boundary_complete:
        return "missing_formal_frost_boundaries"
    if frost.empty:
        return "frost_development_unavailable"
    return "cycle_not_eligible"


def _shift_grid(grid: pd.DatetimeIndex, buckets: int) -> pd.Series:
    values = list(grid[buckets:]) + [pd.NaT] * min(buckets, len(grid))
    return pd.Series(values[: len(grid)], index=grid)


def _with_future_reason(
    record: dict[str, object], status: str, reason: str
) -> dict[str, object]:
    result = dict(record)
    result["metric_status"] = status
    result["exclusion_reason"] = reason
    return result
