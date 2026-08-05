"""Stable Evidence table column contracts."""

from __future__ import annotations

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
    "onset_minutes",
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
    "onset_minutes",
    "primary_target",
    "primary_horizon_minutes",
    "primary_future_effect",
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
