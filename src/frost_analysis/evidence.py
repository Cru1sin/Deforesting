"""Orchestration and stable schemas for exploratory frost-cycle evidence.

The outputs describe within-cycle and cross-date temporal evidence.  They do
not establish causality, independent predictive gain, absolute frost amount,
or a final sensor-selection decision.  ``auto_cycle_initial_reference`` is a
cycle-local relative reference, not a frost-free baseline; onset is a
retrospective statistical response start, not first physical frost.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import EvidencePolicy, EvidenceSettings, validate_evidence_timing
from .evidence_cycle import (
    CYCLE_KEYS,
    CycleChannelEvidence,
    CycleSlice,
    ResolvedReference,
    build_channel_evidence,
    build_cycle_slices,
    feature_metric_record,
    future_records,
    resolve_analysis_reference,
)
from .evidence_summary import aggregate_feature_profiles, compute_pair_similarity

CYCLE_ELIGIBILITY_COLUMNS = [
    "experiment_id",
    "experiment_date",
    "cycle_id",
    "frost_boundary_complete",
    "cycle_status",
    "cycle_status_reason",
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
    "sensitivity_valid_cycle_count",
    "sensitivity_valid_date_count",
    "sensitivity_cycle_sign_agreement",
    "sensitivity_date_sign_agreement",
    "onset_valid_cycle_count",
    "onset_valid_date_count",
    "onset_minutes_median",
    "onset_minutes_iqr",
    "trend_evidence_status",
    "primary_future_valid_cycle_count",
    "primary_future_valid_date_count",
    "primary_future_effect_median",
    "primary_future_effect_iqr",
    "primary_future_sign_agreement",
    "lead_valid_cycle_count",
    "lead_valid_date_count",
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
    "evaluated_cycle_count",
    "valid_cycle_count",
    "valid_date_count",
    "pair_coverage_median",
    "definition_dependency",
    "similarity_status",
    "similarity_reason",
]


@dataclass(frozen=True)
class EvidenceBundle:
    cycle_eligibility: pd.DataFrame
    feature_cycle_metrics: pd.DataFrame
    future_association: pd.DataFrame
    feature_profile: pd.DataFrame
    feature_pair_similarity: pd.DataFrame


def build_evidence_bundle(
    processed: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    settings: EvidenceSettings | EvidencePolicy,
    channels: Mapping[str, Mapping[str, Any]],
    *,
    grid_interval_seconds: int | None = None,
) -> EvidenceBundle:
    """Build all evidence outputs from Processed data and cycle summaries."""
    policy = settings.policy if isinstance(settings, EvidenceSettings) else settings
    interval = 10 if grid_interval_seconds is None else grid_interval_seconds
    validate_evidence_timing(policy, interval)
    frame = _normalise_identity(processed.copy(), cycle_summary)
    summary = _normalise_identity(cycle_summary.copy(), frame)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if frame.duplicated([*CYCLE_KEYS, "timestamp"]).any():
        raise ValueError("duplicate processed cycle timestamp")
    if summary.duplicated(list(CYCLE_KEYS)).any():
        raise ValueError("duplicate cycle summary key")
    for column in ("stable_heating_start", "defrost_start", "baseline_start", "baseline_end"):
        if column in summary:
            summary[column] = pd.to_datetime(summary[column], errors="coerce")
    cycles = build_cycle_slices(frame, summary, interval)
    candidates = _candidate_names(channels, policy.targets)
    targets = list(policy.targets)
    caches: dict[tuple[tuple[str, str, str], str], CycleChannelEvidence] = {}
    metric_rows: list[dict[str, object]] = []
    future_rows: list[dict[str, object]] = []
    eligibility_rows: list[dict[str, object]] = []
    for cycle in cycles:
        for channel in [*candidates, *targets]:
            caches[(cycle.key, channel)] = build_channel_evidence(
                cycle,
                channel,
                policy,
                target=channel in targets,
                interval_seconds=interval,
            )
        for feature in candidates:
            cache = caches[(cycle.key, feature)]
            metric_rows.append(
                feature_metric_record(
                    cycle,
                    feature,
                    channels[feature],
                    cache,
                    policy,
                    interval,
                )
            )
            for target in targets:
                future_rows.extend(
                    future_records(
                        cycle,
                        feature,
                        target,
                        cache,
                        caches[(cycle.key, target)],
                        policy,
                        interval,
                    )
                )
        eligibility_rows.append(_eligibility_record(cycle, candidates, caches))

    metrics = pd.DataFrame(metric_rows, columns=FEATURE_CYCLE_METRIC_COLUMNS)
    future = pd.DataFrame(future_rows, columns=FUTURE_ASSOCIATION_COLUMNS)
    eligibility = pd.DataFrame(eligibility_rows, columns=CYCLE_ELIGIBILITY_COLUMNS)
    profile = aggregate_feature_profiles(
        metrics,
        future,
        candidates,
        channels,
        policy,
        FEATURE_PROFILE_COLUMNS,
    )
    similarity = compute_pair_similarity(
        cycles,
        candidates,
        caches,
        channels,
        policy,
        FEATURE_PAIR_SIMILARITY_COLUMNS,
        interval,
    )
    return EvidenceBundle(eligibility, metrics, future, profile, similarity)


def _eligibility_record(
    cycle: CycleSlice,
    candidates: list[str],
    caches: Mapping[tuple[tuple[str, str, str], str], CycleChannelEvidence],
) -> dict[str, object]:
    start = cycle.start
    end = cycle.end
    return {
        "experiment_id": cycle.key[0],
        "experiment_date": cycle.key[1],
        "cycle_id": cycle.key[2],
        "frost_boundary_complete": bool(start is not None and end is not None and start < end),
        "cycle_status": cycle.cycle_status,
        "cycle_status_reason": cycle.cycle_status_reason,
        "frost_development_status": "available" if not cycle.frost.empty else "unavailable",
        "frost_development_start": start,
        "frost_development_end": end,
        "frost_development_duration_minutes": (
            (end - start).total_seconds() / 60 if start is not None and end is not None else np.nan
        ),
        "frost_development_grid_coverage": cycle.grid_coverage,
        "eligible_feature_count": int(
            sum(
                caches[(cycle.key, feature)].reference.source != "unavailable"
                for feature in candidates
            )
            if cycle.eligible
            else 0
        ),
        "total_candidate_count": len(candidates),
        "eligibility_status": cycle.eligibility_status,
        "exclusion_reason": cycle.exclusion_reason or "",
    }


def _candidate_names(
    channels: Mapping[str, Mapping[str, Any]], targets: tuple[str, ...]
) -> list[str]:
    target_set = set(targets)
    candidates: list[str] = []
    for name, settings in channels.items():
        if not bool(settings.get("analysis_candidate", False)):
            continue
        role = str(settings.get("role", settings.get("registry_role", "")))
        if role in {"performance", "performance_target"} or name in target_set:
            continue
        dependencies = settings.get("depends_on", settings.get("dependencies", [])) or []
        if set(map(str, dependencies)).intersection(target_set):
            continue
        if str(settings.get("alias_of", "")) in target_set:
            continue
        candidates.append(str(name))
    return candidates


def _normalise_identity(frame: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    for name in ("experiment_id", "experiment_date"):
        if name not in frame:
            if name not in other or other[name].nunique(dropna=True) != 1:
                raise ValueError(f"evidence input requires {name}")
            frame[name] = other[name].dropna().iloc[0]
        frame[name] = frame[name].astype(str)
    if "cycle_id" not in frame:
        raise ValueError("evidence input requires cycle_id")
    frame["cycle_id"] = frame["cycle_id"].astype(str)
    return frame


# Compatibility exports for callers that previously imported these symbols
# from the monolithic evidence module.
__all__ = [
    "EvidenceBundle",
    "build_evidence_bundle",
    "resolve_analysis_reference",
    "ResolvedReference",
]
