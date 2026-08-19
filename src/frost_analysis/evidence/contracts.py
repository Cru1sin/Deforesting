"""Stable public contracts for Dataset-native Evidence."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ANALYSIS_VERSION = "frost-cycle-evidence-v2.3"
AGGREGATION_METHOD = "date_balanced_median_of_cycle_medians_v1"
TARGET_DEGRADATION_DIRECTION = {
    "heating_capacity": "decrease",
    "cop": "decrease",
}

CYCLE_ELIGIBILITY_COLUMNS = [
    "cycle_name",
    "cycle_uid",
    "experiment_id",
    "experiment_date",
    "status",
    "analysis_duration_minutes",
    "eligible",
    "exclusion_reason",
]
FEATURE_CYCLE_METRIC_COLUMNS = [
    "cycle_name",
    "cycle_uid",
    "experiment_id",
    "experiment_date",
    "feature",
    "observed_fraction",
    "spearman",
    "signed_effect",
    "trend_slope_per_min",
    "metric_status",
    "exclusion_reason",
]
FUTURE_ASSOCIATION_COLUMNS = [
    "cycle_name",
    "cycle_uid",
    "experiment_id",
    "experiment_date",
    "feature",
    "target",
    "horizon_minutes",
    "effect",
    "degradation_support",
    "valid_pairs",
    "pair_coverage",
    "metric_status",
    "exclusion_reason",
]
FUTURE_HORIZON_SUMMARY_COLUMNS = [
    "feature",
    "target",
    "horizon_minutes",
    "effect",
    "degradation_support",
    "valid_cycle_count",
    "valid_date_count",
    "aggregation_method",
    "metric_status",
    "exclusion_reason",
]
FEATURE_PROFILE_COLUMNS = [
    "feature",
    "trend_valid_cycle_count",
    "trend_valid_date_count",
    "signed_effect",
    "direction_consistency",
    "trend_slope_per_min",
    "primary_target",
    "primary_horizon_minutes",
    "primary_future_degradation_support",
    "primary_future_valid_cycle_count",
    "primary_future_valid_date_count",
]
FEATURE_PAIR_SIMILARITY_COLUMNS = [
    "feature_a",
    "feature_b",
    "valid_cycle_count",
    "valid_date_count",
    "median_abs_spearman",
    "metric_status",
    "exclusion_reason",
]
TARGET_AUDIT_COLUMNS = [
    "cycle_name",
    "experiment_date",
    "target",
    "baseline_value",
    "target_observed_fraction",
    "event_5_elapsed_minutes",
    "event_10_elapsed_minutes",
    "event_15_elapsed_minutes",
    "primary_event_elapsed_minutes",
    "primary_event_status",
    "censor_elapsed_minutes",
    "valid_pairs_5min",
    "valid_pairs_10min",
    "valid_pairs_20min",
    "metric_status",
    "exclusion_reason",
]
READINESS_SPLIT_COLUMNS = [
    "split_id",
    "held_out_cycle",
    "held_out_date",
    "feature",
    "target",
    "horizon_minutes",
    "signal_onset_elapsed_minutes",
    "performance_event_elapsed_minutes",
    "lead_minutes",
    "lead_status",
    "expected_anchor_count",
    "valid_anchor_count",
    "anchor_coverage",
    "train_cycle_count",
    "train_date_count",
    "mae_m0",
    "mae_m1",
    "mae_m2",
    "mae_m3",
    "skill_context_vs_time",
    "skill_level_vs_context",
    "skill_dynamic_vs_level",
    "metric_status",
    "exclusion_reason",
]
READINESS_SUMMARY_COLUMNS = [
    "feature",
    "target",
    "horizon_minutes",
    "trend_valid_cycle_count",
    "trend_valid_date_count",
    "trend_effect",
    "trend_direction_consistency",
    "lead_valid_cycle_count",
    "lead_median_minutes",
    "lead_q25_minutes",
    "positive_lead_fraction",
    "level_skill_median",
    "level_improvement_fraction",
    "dynamic_skill_median",
    "dynamic_improvement_fraction",
    "readiness_status",
    "readiness_reason",
]


@dataclass(frozen=True)
class EvidenceBundle:
    """The nine tables produced by one Evidence analysis."""

    cycle_eligibility: pd.DataFrame
    feature_cycle_metrics: pd.DataFrame
    future_association: pd.DataFrame
    future_horizon_summary: pd.DataFrame
    feature_profile: pd.DataFrame
    feature_pair_similarity: pd.DataFrame
    target_audit: pd.DataFrame
    readiness_split: pd.DataFrame
    readiness_summary: pd.DataFrame
