"""Stable public contracts for Dataset-native Evidence."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ANALYSIS_VERSION = "frost-cycle-evidence-v2.2"
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


@dataclass(frozen=True)
class EvidenceBundle:
    """The six tables produced by one Evidence analysis."""

    cycle_eligibility: pd.DataFrame
    feature_cycle_metrics: pd.DataFrame
    future_association: pd.DataFrame
    future_horizon_summary: pd.DataFrame
    feature_profile: pd.DataFrame
    feature_pair_similarity: pd.DataFrame
