"""Cycle-level, reference-aware frost evidence calculations.

The module intentionally keeps the analysis linear.  A cycle is the unit of
calculation, dates are the unit of cross-run weighting, and no point-level
inference is performed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation, spearmanr, theilslopes

from .config import EvidencePolicy, EvidenceSettings

GRID_INTERVAL_SECONDS = 10
CYCLE_KEYS = ["experiment_id", "experiment_date", "cycle_id"]

CYCLE_ELIGIBILITY_COLUMNS = [
    "experiment_id",
    "experiment_date",
    "cycle_id",
    "complete_boundary",
    "frost_development_status",
    "frost_development_start",
    "frost_development_end",
    "frost_development_duration_minutes",
    "frost_development_grid_coverage",
    "eligible_feature_count",
    "total_candidate_count",
    "eligibility_status",
    "exclusion_reason",
]

FEATURE_CYCLE_METRIC_COLUMNS = [
    "experiment_id",
    "experiment_date",
    "cycle_id",
    "feature",
    "registry_role",
    "expected_frost_direction",
    "cycle_eligible",
    "metric_status",
    "metric_exclusion_reason",
    "observed_fraction",
    "imputed_fraction",
    "maximum_consecutive_gap_seconds",
    "reference_source",
    "reference_center",
    "reference_scale",
    "reference_observed_fraction",
    "reference_valid_from",
    "early_observed_fraction",
    "middle_observed_fraction",
    "late_observed_fraction",
    "signed_sensitivity",
    "absolute_sensitivity",
    "onset_elapsed_minutes",
    "onset_progress",
    "global_spearman",
    "early_slope_per_min",
    "middle_slope_per_min",
    "late_slope_per_min",
    "late_minus_early_slope",
]

FUTURE_ASSOCIATION_COLUMNS = [
    "experiment_id",
    "experiment_date",
    "cycle_id",
    "feature",
    "feature_variant",
    "feature_reference_source",
    "target",
    "target_reference_source",
    "horizon_minutes",
    "target_type",
    "effect_metric",
    "effect",
    "expected_anchor_count",
    "valid_pairs",
    "pair_coverage",
    "lead_time_minutes",
    "lead_time_status",
    "metric_status",
    "exclusion_reason",
]

FEATURE_PROFILE_COLUMNS = [
    "feature",
    "registry_role",
    "expected_frost_direction",
    "reference_scope",
    "configured_baseline_cycle_count",
    "auto_reference_cycle_count",
    "trend_valid_cycle_count",
    "trend_valid_date_count",
    "signed_sensitivity_median",
    "signed_sensitivity_iqr",
    "global_spearman_median",
    "global_spearman_iqr",
    "trend_cycle_sign_agreement",
    "trend_date_sign_agreement",
    "onset_minutes_median",
    "onset_minutes_iqr",
    "trend_evidence_status",
    "primary_future_valid_cycle_count",
    "primary_future_valid_date_count",
    "primary_future_effect_median",
    "primary_future_effect_iqr",
    "primary_future_sign_agreement",
    "lead_valid_cycle_count",
    "lead_time_median",
    "lead_time_iqr",
    "primary_future_evidence_status",
    "evidence_reason",
]

FEATURE_PAIR_SIMILARITY_COLUMNS = [
    "feature_a",
    "feature_b",
    "dynamic_spearman_median",
    "dynamic_spearman_iqr",
    "similarity_sign_agreement",
    "valid_cycle_count",
    "valid_date_count",
    "pair_coverage_median",
    "definition_dependency",
    "similarity_status",
    "similarity_reason",
]


@dataclass(frozen=True)
class EvidenceBundle:
    """The five stable outputs produced by the evidence analysis."""

    cycle_eligibility: pd.DataFrame
    feature_cycle_metrics: pd.DataFrame
    future_association: pd.DataFrame
    feature_profile: pd.DataFrame
    feature_pair_similarity: pd.DataFrame


@dataclass(frozen=True)
class ResolvedReference:
    """One channel's reference residual and its causal validity boundary."""

    residual: pd.Series
    source: str
    center: float
    scale: float
    observed_fraction: float
    valid_from: pd.Timestamp
    exclusion_reason: str | None = None


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
) -> ResolvedReference:
    """Resolve a formal baseline or the fixed five-minute initial reference.

    The returned residual is centered exactly once.  Callers must use
    ``valid_from`` to prevent the retrospective automatic reference window
    from entering predictive calculations.
    """
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
            imputed,
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
        values,
    )


def _configured_reference(
    frame: pd.DataFrame,
    configured_residual: pd.Series,
    configured_baseline_mask: pd.Series | None,
    formal_frost_start: pd.Timestamp,
    imputed: pd.Series,
) -> ResolvedReference | None:
    if configured_baseline_mask is None:
        return None
    source = pd.to_numeric(configured_residual, errors="coerce").reindex(frame.index)
    observed = source.notna() & ~imputed
    baseline_mask = configured_baseline_mask.reindex(frame.index).fillna(False).astype(bool)
    observed &= baseline_mask
    expected = int(baseline_mask.sum())
    if expected == 0:
        return None
    center, scale, fraction = _reference_statistics(source.loc[observed], expected)
    if center is None:
        return None
    return ResolvedReference(
        residual=source - center,
        source="configured_baseline",
        center=center,
        scale=scale,
        observed_fraction=fraction,
        valid_from=_configured_valid_from(
            frame,
            configured_baseline_mask,
            formal_frost_start,
        ),
    )


def _auto_reference(
    frame: pd.DataFrame,
    channel: str,
    formal_frost_start: pd.Timestamp,
    formal_frost_end: pd.Timestamp,
    window_minutes: int,
    minimum_observed_fraction: float,
    maximum_gap_seconds: float,
    interval_seconds: int,
    values: pd.Series,
) -> ResolvedReference:

    window_end = formal_frost_start + pd.Timedelta(minutes=window_minutes)
    if formal_frost_end < window_end:
        return _unavailable(
            frame.index,
            "reference_window_outside_frost_development",
            formal_frost_start,
        )
    grid_start = formal_frost_start.ceil(f"{interval_seconds}s")
    expected_times = pd.date_range(
        grid_start,
        window_end - pd.Timedelta(seconds=interval_seconds),
        freq=f"{interval_seconds}s",
    )
    window = frame.loc[
        frame["timestamp"].ge(formal_frost_start)
        & frame["timestamp"].lt(window_end)
        & frame["cycle_stage"].eq("frost_development")
    ].set_index("timestamp")
    expected = int(len(expected_times))
    if expected == 0:
        return _unavailable(frame.index, "reference_window_empty", formal_frost_start)
    window_values = pd.to_numeric(window[channel], errors="coerce").reindex(expected_times)
    if f"{channel}__imputed" in window:
        window_imputed = (
            window[f"{channel}__imputed"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
            .reindex(expected_times, fill_value=False)
        )
    else:
        window_imputed = pd.Series(False, index=expected_times, dtype=bool)
    observed = window_values.notna() & ~window_imputed
    observed_count = int(observed.sum())
    observed_fraction = observed_count / expected
    maximum_gap = _maximum_false_run_seconds(observed.to_numpy(dtype=bool), interval_seconds)
    minimum_points = ceil(expected * minimum_observed_fraction)
    if observed_count < minimum_points:
        return _unavailable(frame.index, "reference_observed_coverage", formal_frost_start)
    if observed_fraction < minimum_observed_fraction:
        return _unavailable(frame.index, "reference_observed_fraction", formal_frost_start)
    if maximum_gap > maximum_gap_seconds:
        return _unavailable(frame.index, "reference_observation_gap", formal_frost_start)
    center, scale, _ = _reference_statistics(window_values.loc[observed], expected)
    if center is None:
        return _unavailable(frame.index, "reference_values_empty", formal_frost_start)
    return ResolvedReference(
        residual=values - center,
        source="auto_cycle_initial_reference",
        center=center,
        scale=scale,
        observed_fraction=observed_fraction,
        valid_from=window_end,
    )


def build_evidence_bundle(
    processed: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    settings: EvidenceSettings | EvidencePolicy,
    channels: Mapping[str, Mapping[str, Any]],
) -> EvidenceBundle:
    """Build all cycle-level and date-balanced evidence tables."""
    _require_columns(processed, ["timestamp", "cycle_id", "cycle_stage"])
    frame = processed.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = _ensure_identity_columns(frame, cycle_summary)
    summary = _ensure_identity_columns(cycle_summary.copy(), frame)
    summary["cycle_id"] = summary["cycle_id"].astype(str)
    policy = _policy(settings)
    candidates = _candidate_names(channels, policy.targets)
    targets = policy.targets
    cycles = _make_cycles(frame, summary)

    eligibility_rows: list[dict[str, object]] = []
    references: dict[tuple[tuple[str, str, str], str], ResolvedReference] = {}
    cycle_metrics_rows: list[dict[str, object]] = []
    future_rows: list[dict[str, object]] = []

    for cycle in cycles:
        key = cycle["key"]
        cycle_references, cycle_metric_records, cycle_future_records = _cycle_outputs(
            cycle, candidates, targets, channels, policy
        )
        references.update(cycle_references)
        eligibility_record = _eligibility_record(cycle, len(candidates))
        eligibility_record["eligible_feature_count"] = int(
            sum(cycle_references[(key, feature)].source != "unavailable" for feature in candidates)
        )
        eligibility_rows.append(eligibility_record)
        cycle_metrics_rows.extend(cycle_metric_records)
        future_rows.extend(cycle_future_records)

    cycle_metrics = pd.DataFrame(cycle_metrics_rows, columns=FEATURE_CYCLE_METRIC_COLUMNS)
    future = pd.DataFrame(future_rows, columns=FUTURE_ASSOCIATION_COLUMNS)
    eligibility = pd.DataFrame(eligibility_rows, columns=CYCLE_ELIGIBILITY_COLUMNS)
    profiles = _aggregate_profiles(cycle_metrics, future, candidates, channels, policy)
    similarity = _compute_pair_similarity(cycles, candidates, channels, references, policy)
    return EvidenceBundle(
        cycle_eligibility=eligibility,
        feature_cycle_metrics=cycle_metrics,
        future_association=future,
        feature_profile=profiles,
        feature_pair_similarity=similarity,
    )


def _cycle_outputs(
    cycle: dict[str, Any],
    candidates: list[str],
    targets: tuple[str, ...],
    channels: Mapping[str, Mapping[str, Any]],
    policy: EvidencePolicy,
) -> tuple[
    dict[tuple[tuple[str, str, str], str], ResolvedReference],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    references = _resolve_cycle_references(cycle, candidates, targets, policy)
    if (
        not cycle["eligible"]
        or cycle["frost"].empty
        or cycle["start"] is None
        or cycle["end"] is None
    ):
        metrics = _empty_cycle_metrics(cycle, candidates, channels, references, policy)
        futures = _empty_cycle_future(cycle, candidates, targets, policy, references)
        return references, metrics, futures
    metrics = [
        _cycle_metric_record(
            cycle,
            feature,
            channels[feature],
            references[(cycle["key"], feature)],
            policy,
        )
        for feature in candidates
    ]
    futures = [
        _future_record(
            cycle,
            feature,
            feature_variant,
            target,
            horizon,
            target_type,
            references[(cycle["key"], feature)],
            references[(cycle["key"], target)],
            policy,
        )
        for feature in candidates
        for feature_variant in ("residual_level", "past_slope_5min")
        for target in targets
        for horizon in policy.horizons_minutes
        for target_type in ("future_level", "future_change")
    ]
    return references, metrics, futures


def _resolve_cycle_references(
    cycle: dict[str, Any],
    candidates: list[str],
    targets: tuple[str, ...],
    policy: EvidencePolicy,
) -> dict[tuple[tuple[str, str, str], str], ResolvedReference]:
    key = cycle["key"]
    names = [*candidates, *targets]
    if cycle["frost"].empty or cycle["start"] is None or cycle["end"] is None:
        reason = cycle["reason"] or "frost_development_unavailable"
        valid_from = cycle["start"] if cycle["start"] is not None else pd.Timestamp("NaT")
        return {
            (key, name): _unavailable(cycle["frame"].index, reason, valid_from) for name in names
        }
    return {
        (key, name): _resolve_cycle_reference(
            cycle["frame"],
            cycle["summary"],
            name,
            cycle["start"],
            cycle["end"],
            policy,
        )
        for name in names
    }


def _empty_cycle_metrics(
    cycle: dict[str, Any],
    candidates: list[str],
    channels: Mapping[str, Mapping[str, Any]],
    references: Mapping[tuple[tuple[str, str, str], str], ResolvedReference],
    policy: EvidencePolicy,
) -> list[dict[str, object]]:
    return [
        _cycle_metric_record(
            cycle,
            feature,
            channels[feature],
            references[(cycle["key"], feature)],
            policy,
        )
        for feature in candidates
    ]


def _empty_cycle_future(
    cycle: dict[str, Any],
    candidates: list[str],
    targets: tuple[str, ...],
    policy: EvidencePolicy,
    references: Mapping[tuple[tuple[str, str, str], str], ResolvedReference],
) -> list[dict[str, object]]:
    return [
        _empty_future_record(
            cycle,
            feature,
            feature_variant,
            target,
            horizon,
            target_type,
            references[(cycle["key"], feature)],
            references[(cycle["key"], target)],
        )
        for feature in candidates
        for feature_variant in ("residual_level", "past_slope_5min")
        for target in targets
        for horizon in policy.horizons_minutes
        for target_type in ("future_level", "future_change")
    ]


def _policy(settings: EvidenceSettings | EvidencePolicy) -> EvidencePolicy:
    return settings.policy if isinstance(settings, EvidenceSettings) else settings


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"evidence input missing columns: {missing}")


def _ensure_identity_columns(frame: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "experiment_id" not in result:
        if "experiment_id" in other and other["experiment_id"].nunique(dropna=True) == 1:
            result["experiment_id"] = other["experiment_id"].iloc[0]
        else:
            raise ValueError("evidence input requires experiment_id")
    if "experiment_date" not in result:
        if "experiment_date" in other and other["experiment_date"].nunique(dropna=True) == 1:
            result["experiment_date"] = other["experiment_date"].iloc[0]
        else:
            raise ValueError("evidence input requires experiment_date")
    result["experiment_id"] = result["experiment_id"].astype(str)
    result["experiment_date"] = result["experiment_date"].astype(str)
    return result


def _make_cycles(frame: pd.DataFrame, summary: pd.DataFrame) -> list[dict[str, Any]]:
    summary_lookup = {
        tuple(str(row[name]) for name in CYCLE_KEYS): row for _, row in summary.iterrows()
    }
    cycles: list[dict[str, Any]] = []
    grouped = frame.groupby(CYCLE_KEYS, sort=True, dropna=False)
    for values, group in grouped:
        key = tuple(str(value) for value in values)
        ordered = group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        summary_row = summary_lookup.get(key, pd.Series(dtype=object))
        frost = ordered.loc[ordered["cycle_stage"].eq("frost_development")].copy()
        start, end = _formal_bounds(summary_row)
        expected = _expected_grid(start, end)
        actual = set(pd.to_datetime(frost["timestamp"]).tolist())
        grid_coverage = (
            float(len(actual.intersection(set(expected))) / len(expected)) if len(expected) else 0.0
        )
        complete = start is not None and end is not None and start < end
        eligible = bool(complete and not frost.empty)
        reason = "" if eligible else _cycle_exclusion_reason(complete, frost)
        cycles.append(
            {
                "key": key,
                "frame": ordered,
                "frost": frost,
                "summary": summary_row,
                "start": start,
                "end": end,
                "grid_coverage": grid_coverage,
                "eligible": eligible,
                "reason": reason,
            }
        )
    return cycles


def _formal_bounds(summary: pd.Series) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    # These are the Process/cycle-summary fields that define the formal
    # frost-development interval in this repository.  ``cycle_stage`` remains
    # the sole source for stage membership; these timestamps only provide the
    # complete-grid coordinates for progress and audit output.
    start = _first_timestamp(summary, ("stable_heating_start",))
    end = _first_timestamp(summary, ("defrost_start",))
    return start, end


def _first_timestamp(summary: pd.Series, names: tuple[str, ...]) -> pd.Timestamp | None:
    for name in names:
        if name in summary.index and pd.notna(summary.get(name)):
            return pd.Timestamp(summary[name])
    return None


def _expected_grid(start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DatetimeIndex:
    if start is None or end is None or end <= start:
        return pd.DatetimeIndex([])
    grid_start = start.ceil(f"{GRID_INTERVAL_SECONDS}s")
    return pd.date_range(
        grid_start,
        end - pd.Timedelta(seconds=GRID_INTERVAL_SECONDS),
        freq="10s",
    )


def _cycle_exclusion_reason(complete: bool, frost: pd.DataFrame) -> str:
    if not complete:
        return "missing_formal_frost_boundaries"
    if frost.empty:
        return "frost_development_unavailable"
    return "cycle_not_eligible"


def _eligibility_record(cycle: dict[str, Any], total_candidates: int) -> dict[str, object]:
    start = cycle["start"]
    end = cycle["end"]
    return {
        "experiment_id": cycle["key"][0],
        "experiment_date": cycle["key"][1],
        "cycle_id": cycle["key"][2],
        "complete_boundary": bool(start is not None and end is not None and start < end),
        "frost_development_status": "available" if not cycle["frost"].empty else "unavailable",
        "frost_development_start": start,
        "frost_development_end": end,
        "frost_development_duration_minutes": (
            (end - start).total_seconds() / 60 if start is not None and end is not None else np.nan
        ),
        "frost_development_grid_coverage": cycle["grid_coverage"],
        "eligible_feature_count": np.nan,
        "total_candidate_count": total_candidates,
        "eligibility_status": "eligible" if cycle["eligible"] else "excluded",
        "exclusion_reason": cycle["reason"],
    }


def _resolve_cycle_reference(
    cycle_frame: pd.DataFrame,
    summary: pd.Series,
    channel: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    policy: EvidencePolicy,
) -> ResolvedReference:
    residual_name = f"{channel}__baseline_residual"
    baseline_status = str(summary.get("baseline_status", ""))
    baseline_mask = _baseline_window_mask(cycle_frame, summary)
    return resolve_analysis_reference(
        cycle_frame,
        channel=channel,
        cycle_start=start,
        formal_frost_start=start,
        formal_frost_end=end,
        configured_baseline_available=baseline_status == "available"
        and residual_name in cycle_frame,
        configured_residual=cycle_frame.get(residual_name, None),
        configured_baseline_mask=baseline_mask,
        window_minutes=policy.auto_reference_window_minutes,
        minimum_observed_fraction=policy.auto_reference_min_observed_fraction,
        maximum_gap_seconds=policy.auto_reference_max_gap_seconds,
        interval_seconds=GRID_INTERVAL_SECONDS,
    )


def _baseline_window_mask(frame: pd.DataFrame, summary: pd.Series) -> pd.Series | None:
    start = _first_timestamp(summary, ("baseline_start",))
    end = _first_timestamp(summary, ("baseline_end",))
    if start is None or end is None:
        return None
    return frame["timestamp"].ge(start) & frame["timestamp"].lt(end)


def _cycle_metric_record(
    cycle: dict[str, Any],
    feature: str,
    registry: Mapping[str, Any],
    reference: ResolvedReference,
    policy: EvidencePolicy,
) -> dict[str, object]:
    frost = cycle["frost"].copy()
    if (
        not cycle["eligible"]
        or frost.empty
        or cycle["start"] is None
        or cycle["end"] is None
    ):
        return _empty_metric_record(cycle, feature, registry, reference)
    progress = _analysis_progress(frost, cycle["start"], cycle["end"])
    frost["analysis_progress"] = progress
    residual = reference.residual.reindex(frost.index)
    observed = _observed_mask(frost, feature)
    imputed = _imputed_mask(frost, feature)
    # This audit field is the remaining-NaN gap after Process interpolation;
    # real-observation coverage and imputed_fraction are reported separately.
    raw_present = (
        pd.to_numeric(frost[feature], errors="coerce").notna()
        if feature in frost
        else pd.Series(False, index=frost.index, dtype=bool)
    )
    metric_status = (
        "available" if cycle["eligible"] and reference.source != "unavailable" else "unavailable"
    )
    reason = cycle["reason"] if not cycle["eligible"] else reference.exclusion_reason
    early_fraction = _segment_observed_fraction(frost, progress, "early", observed)
    middle_fraction = _segment_observed_fraction(frost, progress, "middle", observed)
    late_fraction = _segment_observed_fraction(frost, progress, "late", observed)
    slopes = {
        name: _segment_slope(
            frost,
            progress,
            residual,
            name,
            reference.valid_from,
            policy,
        )
        for name in ("early", "middle", "late")
    }
    global_spearman = _spearman(
        residual.loc[frost["timestamp"].ge(reference.valid_from)],
        frost.loc[frost["timestamp"].ge(reference.valid_from), "timestamp"].map(
            lambda value: (pd.Timestamp(value) - cycle["start"]).total_seconds() / 60
        ),
        policy.min_segment_points,
    )
    sensitivity = np.nan
    if reference.source != "unavailable" and np.isfinite(reference.scale) and reference.scale > 0:
        late = residual.loc[
            frost["timestamp"].ge(reference.valid_from) & progress.ge(0.75)
        ].dropna()
        late_expected = int((progress.ge(0.75)).sum())
        if (
            len(late) >= policy.min_segment_points
            and late_expected
            and len(late) / late_expected >= policy.min_segment_coverage
        ):
            sensitivity = float(late.median() / reference.scale)
    onset_minutes, onset_progress = _onset(
        frost,
        residual,
        cycle["start"],
        cycle["end"],
        reference,
        policy,
    )
    if reference.source == "unavailable":
        reason = reason or "reference_unavailable"
    elif not np.isfinite(reference.scale) or reference.scale <= 0:
        reason = reason or "not_evaluated_zero_reference_variability"
    return {
        "experiment_id": cycle["key"][0],
        "experiment_date": cycle["key"][1],
        "cycle_id": cycle["key"][2],
        "feature": feature,
        "registry_role": registry.get("role", ""),
        "expected_frost_direction": registry.get("expected_frost_direction", pd.NA),
        "cycle_eligible": bool(cycle["eligible"]),
        "metric_status": metric_status,
        "metric_exclusion_reason": reason,
        "observed_fraction": float(observed.mean()) if len(observed) else 0.0,
        "imputed_fraction": float(imputed.mean()) if len(imputed) else 0.0,
        "maximum_consecutive_gap_seconds": _maximum_false_run_seconds(
            raw_present.to_numpy(dtype=bool), GRID_INTERVAL_SECONDS
        ),
        "reference_source": reference.source,
        "reference_center": reference.center,
        "reference_scale": reference.scale,
        "reference_observed_fraction": reference.observed_fraction,
        "reference_valid_from": reference.valid_from,
        "early_observed_fraction": early_fraction,
        "middle_observed_fraction": middle_fraction,
        "late_observed_fraction": late_fraction,
        "signed_sensitivity": sensitivity,
        "absolute_sensitivity": abs(sensitivity) if np.isfinite(sensitivity) else np.nan,
        "onset_elapsed_minutes": onset_minutes,
        "onset_progress": onset_progress,
        "global_spearman": global_spearman,
        "early_slope_per_min": slopes["early"],
        "middle_slope_per_min": slopes["middle"],
        "late_slope_per_min": slopes["late"],
        "late_minus_early_slope": (
            slopes["late"] - slopes["early"]
            if np.isfinite(slopes["late"]) and np.isfinite(slopes["early"])
            else np.nan
        ),
    }


def _empty_metric_record(
    cycle: dict[str, Any],
    feature: str,
    registry: Mapping[str, Any],
    reference: ResolvedReference,
) -> dict[str, object]:
    return {
        "experiment_id": cycle["key"][0],
        "experiment_date": cycle["key"][1],
        "cycle_id": cycle["key"][2],
        "feature": feature,
        "registry_role": registry.get("role", ""),
        "expected_frost_direction": registry.get("expected_frost_direction", pd.NA),
        "cycle_eligible": False,
        "metric_status": "unavailable",
        "metric_exclusion_reason": cycle["reason"] or "frost_development_unavailable",
        "observed_fraction": 0.0,
        "imputed_fraction": 0.0,
        "maximum_consecutive_gap_seconds": np.nan,
        "reference_source": reference.source,
        "reference_center": reference.center,
        "reference_scale": reference.scale,
        "reference_observed_fraction": reference.observed_fraction,
        "reference_valid_from": reference.valid_from,
        "early_observed_fraction": np.nan,
        "middle_observed_fraction": np.nan,
        "late_observed_fraction": np.nan,
        "signed_sensitivity": np.nan,
        "absolute_sensitivity": np.nan,
        "onset_elapsed_minutes": np.nan,
        "onset_progress": np.nan,
        "global_spearman": np.nan,
        "early_slope_per_min": np.nan,
        "middle_slope_per_min": np.nan,
        "late_slope_per_min": np.nan,
        "late_minus_early_slope": np.nan,
    }


def _candidate_names(
    channels: Mapping[str, Mapping[str, Any]], targets: tuple[str, ...]
) -> list[str]:
    target_set = set(targets)
    result: list[str] = []
    for name, settings in channels.items():
        if not bool(settings.get("analysis_candidate", False)):
            continue
        role = str(settings.get("role", ""))
        if role in {"performance", "performance_target"} or name in target_set:
            continue
        dependencies = settings.get("depends_on", settings.get("dependencies", [])) or []
        if set(map(str, dependencies)).intersection(target_set):
            continue
        if str(settings.get("alias_of", "")) in target_set:
            continue
        result.append(str(name))
    return result


def _analysis_progress(
    frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None
) -> pd.Series:
    if start is None or end is None or end <= start:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return ((frame["timestamp"] - start).dt.total_seconds() / (end - start).total_seconds()).clip(
        0, 1
    )


def _segment_observed_fraction(
    frame: pd.DataFrame, progress: pd.Series, name: str, observed: pd.Series
) -> float:
    mask = _segment_mask(progress, name)
    return float(observed.loc[mask].mean()) if int(mask.sum()) else 0.0


def _segment_slope(
    frame: pd.DataFrame,
    progress: pd.Series,
    residual: pd.Series,
    name: str,
    valid_from: pd.Timestamp,
    policy: EvidencePolicy,
) -> float:
    segment = _segment_mask(progress, name)
    expected = int(segment.sum())
    segment &= frame["timestamp"].ge(valid_from)
    valid = segment & residual.notna()
    if expected == 0 or int(valid.sum()) < policy.min_segment_points:
        return np.nan
    if int(valid.sum()) / expected < policy.min_segment_coverage:
        return np.nan
    x = (
        frame.loc[valid, "timestamp"] - frame.loc[valid, "timestamp"].min()
    ).dt.total_seconds() / 60
    y = residual.loc[valid].astype(float)
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(theilslopes(y.to_numpy(), x.to_numpy()).slope)


def _segment_mask(progress: pd.Series, name: str) -> pd.Series:
    if name == "early":
        return progress.ge(0) & progress.lt(0.25)
    if name == "middle":
        return progress.ge(0.25) & progress.lt(0.75)
    return progress.ge(0.75) & progress.le(1.0)


def _onset(
    frame: pd.DataFrame,
    residual: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    reference: ResolvedReference,
    policy: EvidencePolicy,
) -> tuple[float, float]:
    if (
        reference.source == "unavailable"
        or not np.isfinite(reference.scale)
        or reference.scale <= 0
    ):
        return np.nan, np.nan
    allowed = frame["timestamp"].ge(reference.valid_from)
    work = pd.DataFrame(
        {"timestamp": frame.loc[allowed, "timestamp"], "residual": residual.loc[allowed]}
    )
    work = work.sort_values("timestamp", kind="stable").reset_index(drop=True)
    buckets = max(1, round(policy.onset_window_seconds / GRID_INTERVAL_SECONDS))
    persistence = max(1, round(policy.onset_persistence_seconds / GRID_INTERVAL_SECONDS))
    smooth = work["residual"].rolling(buckets, min_periods=buckets).median()
    qualifying = smooth.abs().gt(policy.onset_mad_multiplier * reference.scale).fillna(False)
    for end_index in range(persistence - 1, len(work)):
        window = qualifying.iloc[end_index - persistence + 1 : end_index + 1]
        if not bool(window.all()):
            continue
        smooth_timestamps = work.loc[
            end_index - buckets + 1 : end_index, "timestamp"
        ]
        if len(smooth_timestamps) != buckets or smooth_timestamps.diff().dropna().dt.total_seconds().ne(
            GRID_INTERVAL_SECONDS
        ).any():
            continue
        timestamps = work.loc[end_index - persistence + 1 : end_index, "timestamp"]
        if timestamps.diff().dropna().dt.total_seconds().ne(GRID_INTERVAL_SECONDS).any():
            continue
        onset_time = pd.Timestamp(timestamps.iloc[0])
        elapsed = (onset_time - start).total_seconds() / 60
        progress = (onset_time - start).total_seconds() / (end - start).total_seconds()
        return float(elapsed), float(np.clip(progress, 0, 1))
    return np.nan, np.nan


def _future_record(
    cycle: dict[str, Any],
    feature: str,
    feature_variant: str,
    target: str,
    horizon: int,
    target_type: str,
    feature_reference: ResolvedReference,
    target_reference: ResolvedReference,
    policy: EvidencePolicy,
) -> dict[str, object]:
    if not cycle["eligible"] or cycle["frost"].empty:
        return _empty_future_record(
            cycle,
            feature,
            feature_variant,
            target,
            horizon,
            target_type,
            feature_reference,
            target_reference,
        )
    frost = cycle["frost"].sort_values("timestamp", kind="stable")
    feature_values = _feature_variant(frost, feature_reference, feature_variant, policy)
    timestamps = pd.DatetimeIndex(frost["timestamp"])
    future_timestamps = timestamps + pd.Timedelta(minutes=horizon)
    same_stage = pd.Series(future_timestamps.isin(timestamps), index=frost.index)
    structural = timestamps >= feature_reference.valid_from
    if feature_variant == "past_slope_5min":
        structural &= timestamps >= feature_reference.valid_from + pd.Timedelta(minutes=5)
    target_source, target_structural, outcome, target_valid = _future_target_data(
        frost,
        timestamps,
        future_timestamps,
        target,
        target_type,
        target_reference,
    )
    structural &= same_stage & target_structural
    expected = int(structural.sum())
    valid_feature = feature_values.notna()
    valid = structural & valid_feature & target_valid
    valid_pairs = int(valid.sum())
    coverage = valid_pairs / expected if expected else np.nan
    effect, status, reason = _association_effect(
        feature_reference,
        target_reference,
        target_type,
        expected,
        valid_pairs,
        coverage,
        feature_values.loc[valid],
        outcome.loc[valid],
        policy,
    )
    lead_time, lead_status = _lead_time(
        cycle,
        frost,
        feature_variant,
        target_type,
        horizon,
        feature_reference,
        target_reference,
        policy,
    )
    return {
        "experiment_id": cycle["key"][0],
        "experiment_date": cycle["key"][1],
        "cycle_id": cycle["key"][2],
        "feature": feature,
        "feature_variant": feature_variant,
        "feature_reference_source": feature_reference.source,
        "target": target,
        "target_reference_source": target_source,
        "horizon_minutes": horizon,
        "target_type": target_type,
        "effect_metric": "within_cycle_spearman",
        "effect": effect,
        "expected_anchor_count": expected,
        "valid_pairs": valid_pairs,
        "pair_coverage": coverage,
        "lead_time_minutes": lead_time,
        "lead_time_status": lead_status,
        "metric_status": status,
        "exclusion_reason": reason,
    }


def _empty_future_record(
    cycle: dict[str, Any],
    feature: str,
    feature_variant: str,
    target: str,
    horizon: int,
    target_type: str,
    feature_reference: ResolvedReference,
    target_reference: ResolvedReference,
) -> dict[str, object]:
    target_source = "not_required" if target_type == "future_change" else target_reference.source
    return {
        "experiment_id": cycle["key"][0],
        "experiment_date": cycle["key"][1],
        "cycle_id": cycle["key"][2],
        "feature": feature,
        "feature_variant": feature_variant,
        "feature_reference_source": feature_reference.source,
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
        "exclusion_reason": cycle["reason"] or "cycle_not_eligible",
    }


def _association_effect(
    feature_reference: ResolvedReference,
    target_reference: ResolvedReference,
    target_type: str,
    expected: int,
    valid_pairs: int,
    coverage: float,
    feature_values: pd.Series,
    outcome: pd.Series,
    policy: EvidencePolicy,
) -> tuple[float, str, str]:
    if feature_reference.source == "unavailable":
        return np.nan, "unavailable", "feature_reference_unavailable"
    if target_type == "future_level" and target_reference.source == "unavailable":
        return np.nan, "unavailable", "target_reference_unavailable"
    if expected == 0:
        return np.nan, "unavailable", "no_structural_anchors"
    if valid_pairs < policy.min_valid_pairs:
        return np.nan, "insufficient_pairs", "valid_pairs_below_minimum"
    if coverage < policy.min_pair_coverage:
        return np.nan, "insufficient_coverage", "pair_coverage_below_minimum"
    effect = _spearman(feature_values, outcome, policy.min_valid_pairs)
    if not np.isfinite(effect):
        return np.nan, "not_evaluated", "zero_variability"
    return effect, "available", ""


def _lead_time(
    cycle: dict[str, Any],
    frost: pd.DataFrame,
    feature_variant: str,
    target_type: str,
    horizon: int,
    feature_reference: ResolvedReference,
    target_reference: ResolvedReference,
    policy: EvidencePolicy,
) -> tuple[float, str]:
    canonical = (
        feature_variant == "residual_level"
        and target_type == "future_level"
        and horizon == policy.primary_horizon_minutes
    )
    if not canonical:
        return np.nan, "not_applicable"
    feature_onset, _ = _onset(
        frost,
        feature_reference.residual.reindex(frost.index),
        cycle["start"],
        cycle["end"],
        feature_reference,
        policy,
    )
    target_onset, _ = _onset(
        frost,
        target_reference.residual.reindex(frost.index),
        cycle["start"],
        cycle["end"],
        target_reference,
        policy,
    )
    if np.isfinite(feature_onset) and np.isfinite(target_onset):
        return float(target_onset - feature_onset), "available"
    if feature_reference.source == "unavailable":
        return np.nan, "feature_onset_unavailable"
    if target_reference.source == "unavailable":
        return np.nan, "target_reference_unavailable"
    return np.nan, "onset_unavailable"


def _future_target_data(
    frost: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    future_timestamps: pd.DatetimeIndex,
    target: str,
    target_type: str,
    target_reference: ResolvedReference,
) -> tuple[str, pd.Series, pd.Series, pd.Series]:
    if target_type == "future_level":
        residual = target_reference.residual.reindex(frost.index)
        outcome = _map_series(future_timestamps, timestamps, residual)
        structural = (target_reference.source != "unavailable") & pd.Series(
            future_timestamps >= target_reference.valid_from,
            index=frost.index,
        )
        target_valid = _target_valid_mask(frost, target)
        future_valid = _map_boolean(future_timestamps, timestamps, target_valid)
        return target_reference.source, structural, outcome, outcome.notna() & future_valid
    raw = (
        pd.to_numeric(frost[target], errors="coerce")
        if target in frost
        else pd.Series(np.nan, index=frost.index)
    )
    future = _map_series(future_timestamps, timestamps, raw)
    target_valid = _target_valid_mask(frost, target)
    future_valid = _map_boolean(future_timestamps, timestamps, target_valid)
    valid = raw.notna() & target_valid & future.notna() & future_valid
    return "not_required", pd.Series(True, index=frost.index), future - raw, valid


def _feature_variant(
    frame: pd.DataFrame,
    reference: ResolvedReference,
    variant: str,
    policy: EvidencePolicy,
) -> pd.Series:
    residual = reference.residual.reindex(frame.index)
    if variant == "residual_level":
        return residual.where(frame["timestamp"].ge(reference.valid_from))
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for index, timestamp in frame["timestamp"].items():
        if timestamp < reference.valid_from + pd.Timedelta(minutes=5):
            continue
        mask = frame["timestamp"].gt(timestamp - pd.Timedelta(minutes=5)) & frame["timestamp"].le(
            timestamp
        )
        values = residual.loc[mask].dropna()
        if (
            len(values) < policy.min_segment_points
            or len(values) / 30 < policy.min_segment_coverage
        ):
            continue
        x = (frame.loc[mask & residual.notna(), "timestamp"] - timestamp).dt.total_seconds() / 60
        y = residual.loc[mask & residual.notna()].astype(float)
        if len(y) >= 2 and x.nunique() >= 2:
            result.at[index] = float(np.polyfit(x.to_numpy(), y.to_numpy(), 1)[0])
    return result


def _aggregate_profiles(
    metrics: pd.DataFrame,
    future: pd.DataFrame,
    candidates: list[str],
    channels: Mapping[str, Mapping[str, Any]],
    policy: EvidencePolicy,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    primary = future.loc[
        future["target"].eq(policy.primary_target)
        & future["target_type"].eq(policy.primary_target_type)
        & future["horizon_minutes"].eq(policy.primary_horizon_minutes)
        & future["feature_variant"].eq(policy.primary_feature_variant)
    ]
    lead = future.loc[
        future["target"].eq(policy.lead_target)
        & future["target_type"].eq("future_level")
        & future["horizon_minutes"].eq(policy.primary_horizon_minutes)
        & future["feature_variant"].eq("residual_level")
    ]
    for feature in candidates:
        feature_metrics = metrics.loc[metrics["feature"].eq(feature)].copy()
        trend = feature_metrics.loc[
            feature_metrics["metric_status"].eq("available")
            & (
                feature_metrics["global_spearman"].map(np.isfinite)
                | feature_metrics["early_slope_per_min"].map(np.isfinite)
                | feature_metrics["late_slope_per_min"].map(np.isfinite)
            )
        ]
        primary_rows = primary.loc[
            primary["feature"].eq(feature)
            & primary["metric_status"].eq("available")
            & primary["effect"].map(np.isfinite)
        ]
        lead_rows = lead.loc[
            lead["feature"].eq(feature) & lead["lead_time_minutes"].map(np.isfinite)
        ]
        sensitivity_dates, sensitivity_values = _date_balanced_values(trend, "signed_sensitivity")
        spearman_dates, spearman_values = _date_balanced_values(trend, "global_spearman")
        onset_dates, onset_values = _date_balanced_values(trend, "onset_elapsed_minutes")
        future_dates, future_values = _date_balanced_values(primary_rows, "effect")
        lead_dates, lead_values = _date_balanced_values(lead_rows, "lead_time_minutes")
        trend_dates = sorted(str(value) for value in trend["experiment_date"].dropna().unique())
        sources = set(str(value) for value in trend["reference_source"].dropna())
        rows.append(
            {
                "feature": feature,
                "registry_role": channels[feature].get("role", ""),
                "expected_frost_direction": channels[feature].get(
                    "expected_frost_direction", pd.NA
                ),
                "reference_scope": _reference_scope(sources),
                "configured_baseline_cycle_count": int(
                    trend["reference_source"].eq("configured_baseline").sum()
                ),
                "auto_reference_cycle_count": int(
                    trend["reference_source"].eq("auto_cycle_initial_reference").sum()
                ),
                "trend_valid_cycle_count": int(len(trend)),
                "trend_valid_date_count": len(trend_dates),
                "signed_sensitivity_median": _median(sensitivity_values),
                "signed_sensitivity_iqr": _iqr(sensitivity_values),
                "global_spearman_median": _median(spearman_values),
                "global_spearman_iqr": _iqr(spearman_values),
                "trend_cycle_sign_agreement": _sign_agreement(
                    trend["signed_sensitivity"].tolist(), _median(trend["signed_sensitivity"])
                ),
                "trend_date_sign_agreement": _sign_agreement(
                    sensitivity_values, _median(sensitivity_values)
                ),
                "onset_minutes_median": _median(onset_values),
                "onset_minutes_iqr": _iqr(onset_values),
                "trend_evidence_status": _evidence_status(len(trend), len(trend_dates), policy),
                "primary_future_valid_cycle_count": _unique_cycle_count(primary_rows),
                "primary_future_valid_date_count": len(future_dates),
                "primary_future_effect_median": _median(future_values),
                "primary_future_effect_iqr": _iqr(future_values),
                "primary_future_sign_agreement": _sign_agreement(
                    future_values, _median(future_values)
                ),
                "lead_valid_cycle_count": _unique_cycle_count(lead_rows),
                "lead_time_median": _median(lead_values),
                "lead_time_iqr": _iqr(lead_values),
                "primary_future_evidence_status": _evidence_status(
                    _unique_cycle_count(primary_rows), len(future_dates), policy
                ),
                "evidence_reason": _profile_reason(trend, primary_rows),
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_PROFILE_COLUMNS)


def _compute_pair_similarity(
    cycles: list[dict[str, Any]],
    candidates: list[str],
    channels: Mapping[str, Mapping[str, Any]],
    references: Mapping[tuple[tuple[str, str, str], str], ResolvedReference],
    policy: EvidencePolicy,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_a, feature_b in combinations(candidates, 2):
        cycle_values: list[dict[str, object]] = []
        for cycle in cycles:
            if not cycle["eligible"]:
                continue
            key = cycle["key"]
            ref_a = references.get((key, feature_a))
            ref_b = references.get((key, feature_b))
            if ref_a is None or ref_b is None:
                continue
            frost = cycle["frost"].sort_values("timestamp", kind="stable")
            slope_a = _feature_variant(frost, ref_a, "past_slope_5min", policy)
            slope_b = _feature_variant(frost, ref_b, "past_slope_5min", policy)
            common = frost["timestamp"].ge(
                max(ref_a.valid_from, ref_b.valid_from) + pd.Timedelta(minutes=5)
            )
            expected = int(common.sum())
            valid = common & slope_a.notna() & slope_b.notna()
            count = int(valid.sum())
            coverage = count / expected if expected else np.nan
            effect = (
                _spearman(slope_a.loc[valid], slope_b.loc[valid], policy.min_valid_pairs)
                if expected
                and count >= policy.min_valid_pairs
                and coverage >= policy.min_pair_coverage
                else np.nan
            )
            if np.isfinite(effect):
                cycle_values.append(
                    {"experiment_date": key[1], "effect": effect, "coverage": coverage}
                )
        cycle_frame = pd.DataFrame(cycle_values)
        dates, effects = _date_balanced_values(cycle_frame, "effect")
        _, coverages = _date_balanced_values(cycle_frame, "coverage")
        valid_cycle_count = len(cycle_frame)
        median_effect = _median(effects)
        dependency = _definition_dependency(feature_a, feature_b, channels)
        if not effects:
            status, reason = "no_valid_evidence", "no_valid_cycle_pairs"
        elif valid_cycle_count < policy.min_valid_cycles:
            status, reason = "insufficient_cycles", "valid_cycles_below_minimum"
        elif abs(median_effect) >= policy.similarity_threshold:
            status, reason = "high_dynamic_similarity", "absolute_dynamic_spearman_meets_threshold"
        else:
            status, reason = (
                "no_high_dynamic_similarity",
                "absolute_dynamic_spearman_below_threshold",
            )
        rows.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "dynamic_spearman_median": median_effect,
                "dynamic_spearman_iqr": _iqr(effects),
                "similarity_sign_agreement": _sign_agreement(effects, median_effect),
                "valid_cycle_count": valid_cycle_count,
                "valid_date_count": len(dates),
                "pair_coverage_median": _median(coverages),
                "definition_dependency": dependency,
                "similarity_status": status,
                "similarity_reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_PAIR_SIMILARITY_COLUMNS)


def _date_balanced_values(frame: pd.DataFrame, column: str) -> tuple[list[str], list[float]]:
    if frame.empty or column not in frame:
        return [], []
    values = pd.to_numeric(frame[column], errors="coerce")
    work = frame.loc[values.notna(), ["experiment_date"]].copy()
    work["value"] = values.loc[work.index].astype(float)
    if work.empty:
        return [], []
    grouped = work.groupby("experiment_date", sort=True)["value"].median()
    return [str(value) for value in grouped.index], grouped.astype(float).tolist()


def _evidence_status(cycles: int, dates: int, policy: EvidencePolicy) -> str:
    if cycles == 0:
        return "no_valid_evidence"
    if cycles < policy.min_valid_cycles:
        return "insufficient_cycles"
    return "within_date_exploratory" if dates == 1 else "cross_date_exploratory"


def _reference_scope(sources: set[str]) -> str:
    sources.discard("unavailable")
    if sources == {"configured_baseline"}:
        return "configured_only"
    if sources == {"auto_cycle_initial_reference"}:
        return "auto_only"
    if sources == {"configured_baseline", "auto_cycle_initial_reference"}:
        return "mixed"
    return "unavailable"


def _profile_reason(trend: pd.DataFrame, primary: pd.DataFrame) -> str:
    if trend.empty and primary.empty:
        return "no_valid_trend_or_future_evidence"
    if trend.empty:
        return "no_valid_trend_evidence"
    if primary.empty:
        return "no_valid_primary_future_evidence"
    return ""


def _unique_cycle_count(frame: pd.DataFrame) -> int:
    return (
        int(frame[["experiment_id", "experiment_date", "cycle_id"]].drop_duplicates().shape[0])
        if not frame.empty
        else 0
    )


def _sign_agreement(values: list[float], reference: float) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite or not np.isfinite(reference) or reference == 0:
        return np.nan
    sign = np.sign(reference)
    return float(np.mean([np.sign(value) == sign for value in finite]))


def _definition_dependency(a: str, b: str, channels: Mapping[str, Mapping[str, Any]]) -> bool:
    left = channels.get(a, {})
    right = channels.get(b, {})
    left_dependencies = set(map(str, left.get("depends_on", left.get("dependencies", [])) or []))
    right_dependencies = set(map(str, right.get("depends_on", right.get("dependencies", [])) or []))
    return bool(
        str(left.get("alias_of", "")) == b
        or str(right.get("alias_of", "")) == a
        or b in left_dependencies
        or a in right_dependencies
    )


def _spearman(left: pd.Series, right: pd.Series, minimum_points: int) -> float:
    x = pd.to_numeric(left, errors="coerce")
    y = pd.to_numeric(right, errors="coerce")
    valid = x.notna() & y.notna()
    if int(valid.sum()) < minimum_points:
        return np.nan
    x = x.loc[valid].astype(float)
    y = y.loc[valid].astype(float)
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    value = spearmanr(x.to_numpy(), y.to_numpy()).statistic
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


def _configured_valid_from(
    frame: pd.DataFrame, baseline_mask: pd.Series | None, formal_start: pd.Timestamp
) -> pd.Timestamp:
    if baseline_mask is None or not baseline_mask.any():
        return formal_start
    end = frame.loc[baseline_mask, "timestamp"].max()
    return (
        pd.Timestamp(end) + pd.Timedelta(seconds=GRID_INTERVAL_SECONDS)
        if end >= formal_start
        else formal_start
    )


def _imputed_mask(frame: pd.DataFrame, channel: str) -> pd.Series:
    column = f"{channel}__imputed"
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].astype("boolean").fillna(False).astype(bool)


def _observed_mask(frame: pd.DataFrame, channel: str) -> pd.Series:
    if channel not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return pd.to_numeric(frame[channel], errors="coerce").notna() & ~_imputed_mask(frame, channel)


def _target_valid_mask(frame: pd.DataFrame, target: str) -> pd.Series:
    result = ~_imputed_mask(frame, target)
    for column in (f"{target}__quality_valid", f"{target}__valid"):
        if column in frame:
            result &= frame[column].astype("boolean").fillna(False).astype(bool)
    return result


def _map_series(
    target_times: pd.DatetimeIndex, source_times: pd.DatetimeIndex, values: pd.Series
) -> pd.Series:
    source_values = pd.to_numeric(values, errors="coerce").to_numpy()
    mapping = dict(zip(source_times, source_values, strict=True))
    mapped = [mapping.get(timestamp, np.nan) for timestamp in target_times]
    return pd.Series(mapped, index=values.index, dtype=float)


def _map_boolean(
    target_times: pd.DatetimeIndex, source_times: pd.DatetimeIndex, values: pd.Series
) -> pd.Series:
    mapping = dict(zip(source_times, values.astype(bool).to_numpy(), strict=True))
    mapped = [mapping.get(timestamp, False) for timestamp in target_times]
    return pd.Series(mapped, index=values.index, dtype=bool)


def _maximum_false_run_seconds(values: np.ndarray, interval_seconds: int) -> float:
    maximum = 0
    current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return float(maximum * interval_seconds)


def _unavailable(index: pd.Index, reason: str, valid_from: pd.Timestamp) -> ResolvedReference:
    return ResolvedReference(
        residual=pd.Series(np.nan, index=index, dtype=float),
        source="unavailable",
        center=np.nan,
        scale=np.nan,
        observed_fraction=0.0,
        valid_from=valid_from,
        exclusion_reason=reason,
    )


def _median(values: list[float] | pd.Series) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else np.nan


def _iqr(values: list[float] | pd.Series) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(numeric.quantile(0.75) - numeric.quantile(0.25)) if not numeric.empty else np.nan
