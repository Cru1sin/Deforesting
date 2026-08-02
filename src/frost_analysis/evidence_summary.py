"""Date-balanced aggregation for cycle-level frost evidence."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import EvidencePolicy
from .evidence_cycle import CycleChannelEvidence, CycleSlice, duration_buckets


def aggregate_feature_profiles(
    metrics: pd.DataFrame,
    future: pd.DataFrame,
    candidates: list[str],
    channels: Mapping[str, Mapping[str, Any]],
    policy: EvidencePolicy,
    profile_columns: list[str],
) -> pd.DataFrame:
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
    rows: list[dict[str, object]] = []
    for feature in candidates:
        feature_metrics = metrics.loc[metrics["feature"].eq(feature)].copy()
        eligible_refs = feature_metrics.loc[
            feature_metrics["cycle_eligible"].eq(True)
            & feature_metrics["reference_source"].isin(
                ["configured_baseline", "auto_cycle_initial_reference"]
            )
        ]
        trend = _finite_rows(feature_metrics, "global_spearman")
        sensitivity = _finite_rows(feature_metrics, "signed_sensitivity")
        onset = _finite_rows(feature_metrics, "onset_elapsed_minutes")
        primary_rows = _finite_rows(
            primary.loc[primary["feature"].eq(feature)], "effect"
        )
        lead_rows = _finite_rows(lead.loc[lead["feature"].eq(feature)], "lead_time_minutes")

        trend_cycle_values = _finite_values(trend, "global_spearman")
        trend_dates, trend_values = date_balanced_values(trend, "global_spearman")
        sensitivity_cycle_values = _finite_values(sensitivity, "signed_sensitivity")
        sensitivity_dates, sensitivity_values = date_balanced_values(
            sensitivity, "signed_sensitivity"
        )
        onset_dates, onset_values = date_balanced_values(onset, "onset_elapsed_minutes")
        future_dates, future_values = date_balanced_values(primary_rows, "effect")
        lead_dates, lead_values = date_balanced_values(lead_rows, "lead_time_minutes")
        sources = set(eligible_refs["reference_source"].astype(str))
        trend_median = _median(trend_values)
        sensitivity_median = _median(sensitivity_values)

        rows.append(
            {
                "feature": feature,
                "registry_role": channels[feature].get("role", ""),
                "expected_frost_direction": channels[feature].get(
                    "expected_frost_direction", pd.NA
                ),
                "reference_scope": reference_scope(sources),
                "configured_baseline_cycle_count": int(
                    eligible_refs["reference_source"].eq("configured_baseline").sum()
                ),
                "auto_reference_cycle_count": int(
                    eligible_refs["reference_source"].eq("auto_cycle_initial_reference").sum()
                ),
                "trend_valid_cycle_count": len(trend),
                "trend_valid_date_count": len(trend_dates),
                "signed_sensitivity_median": sensitivity_median,
                "signed_sensitivity_iqr": _iqr(sensitivity_values),
                "global_spearman_median": trend_median,
                "global_spearman_iqr": _iqr(trend_values),
                "trend_cycle_sign_agreement": sign_agreement(
                    trend_cycle_values, _median(trend_cycle_values)
                ),
                "trend_date_sign_agreement": sign_agreement(
                    trend_values, trend_median
                ),
                "sensitivity_valid_cycle_count": len(sensitivity),
                "sensitivity_valid_date_count": len(sensitivity_dates),
                "sensitivity_cycle_sign_agreement": sign_agreement(
                    sensitivity_cycle_values, _median(sensitivity_cycle_values)
                ),
                "sensitivity_date_sign_agreement": sign_agreement(
                    sensitivity_values, sensitivity_median
                ),
                "onset_valid_cycle_count": len(onset),
                "onset_valid_date_count": len(onset_dates),
                "onset_minutes_median": _median(onset_values),
                "onset_minutes_iqr": _iqr(onset_values),
                "trend_evidence_status": evidence_status(
                    len(trend), len(trend_dates), policy
                ),
                "primary_future_valid_cycle_count": unique_cycle_count(primary_rows),
                "primary_future_valid_date_count": len(future_dates),
                "primary_future_effect_median": _median(future_values),
                "primary_future_effect_iqr": _iqr(future_values),
                "primary_future_sign_agreement": sign_agreement(
                    future_values, _median(future_values)
                ),
                "lead_valid_cycle_count": unique_cycle_count(lead_rows),
                "lead_valid_date_count": len(lead_dates),
                "lead_time_median": _median(lead_values),
                "lead_time_iqr": _iqr(lead_values),
                "primary_future_evidence_status": evidence_status(
                    unique_cycle_count(primary_rows), len(future_dates), policy
                ),
                "evidence_reason": profile_reason(trend, primary_rows),
            }
        )
    return pd.DataFrame(rows, columns=profile_columns)


def compute_pair_similarity(  # noqa: C901
    cycles: list[CycleSlice],
    candidates: list[str],
    caches: Mapping[tuple[tuple[str, str, str], str], CycleChannelEvidence],
    channels: Mapping[str, Mapping[str, Any]],
    policy: EvidencePolicy,
    pair_columns: list[str],
    interval_seconds: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    anchor_buckets = duration_buckets(5 * 60, interval_seconds)
    for feature_a, feature_b in combinations(candidates, 2):
        evaluated: list[dict[str, object]] = []
        valid_effects: list[dict[str, object]] = []
        for cycle in cycles:
            if not cycle.eligible or cycle.grid.empty:
                continue
            cache_a = caches.get((cycle.key, feature_a))
            cache_b = caches.get((cycle.key, feature_b))
            if cache_a is None or cache_b is None:
                continue
            if cache_a.past_slope_5min is None or cache_b.past_slope_5min is None:
                continue
            anchor = cycle.grid >= cycle.grid[0] + pd.Timedelta(
                seconds=anchor_buckets * interval_seconds
            )
            expected = int(anchor.sum())
            if expected == 0:
                continue
            evaluated.append({"experiment_date": cycle.key[1], "coverage": np.nan})
            valid = anchor & cache_a.past_slope_5min.notna() & cache_b.past_slope_5min.notna()
            count = int(valid.sum())
            coverage = count / expected
            evaluated[-1]["coverage"] = coverage
            if count < policy.min_valid_pairs or coverage < policy.min_pair_coverage:
                continue
            slope_a = cache_a.past_slope_5min.loc[valid]
            slope_b = cache_b.past_slope_5min.loc[valid]
            if slope_a.nunique() < 2 or slope_b.nunique() < 2:
                continue
            effect = _spearman(slope_a, slope_b, policy.min_valid_pairs)
            if np.isfinite(effect):
                valid_effects.append(
                    {"experiment_date": cycle.key[1], "cycle_key": cycle.key, "effect": effect}
                )
        coverage_frame = pd.DataFrame(evaluated)
        effect_frame = pd.DataFrame(valid_effects)
        _, coverages = date_balanced_values(coverage_frame, "coverage")
        effect_dates, effects = date_balanced_values(effect_frame, "effect")
        median_effect = _median(effects)
        valid_cycles = len(valid_effects)
        if not effects:
            status, reason = "no_valid_evidence", "no_valid_cycle_pairs"
        elif valid_cycles < policy.min_valid_cycles:
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
                "similarity_sign_agreement": sign_agreement(effects, median_effect),
                "evaluated_cycle_count": len(evaluated),
                "valid_cycle_count": valid_cycles,
                "valid_date_count": len(effect_dates),
                "pair_coverage_median": _median(coverages),
                "definition_dependency": definition_dependency(feature_a, feature_b, channels),
                "similarity_status": status,
                "similarity_reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=pair_columns)


def date_balanced_values(frame: pd.DataFrame, column: str) -> tuple[list[str], list[float]]:
    if frame.empty or column not in frame:
        return [], []
    numeric = pd.to_numeric(frame[column], errors="coerce")
    mask = numeric.notna() & frame["experiment_date"].notna()
    if not mask.any():
        return [], []
    grouped = frame.loc[mask].assign(_value=numeric.loc[mask]).groupby(
        "experiment_date", sort=True
    )["_value"].median()
    return [str(value) for value in grouped.index], grouped.astype(float).tolist()


def evidence_status(cycles: int, dates: int, policy: EvidencePolicy) -> str:
    if cycles == 0:
        return "no_valid_evidence"
    if cycles < policy.min_valid_cycles:
        return "insufficient_cycles"
    return "within_date_exploratory" if dates == 1 else "cross_date_exploratory"


def reference_scope(sources: set[str]) -> str:
    if sources == {"configured_baseline"}:
        return "configured_only"
    if sources == {"auto_cycle_initial_reference"}:
        return "auto_only"
    if sources == {"configured_baseline", "auto_cycle_initial_reference"}:
        return "mixed"
    return "unavailable"


def definition_dependency(a: str, b: str, channels: Mapping[str, Mapping[str, Any]]) -> bool:
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


def _finite_rows(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    values = pd.to_numeric(frame[column], errors="coerce")
    return frame.loc[values.notna()].copy()


def _finite_values(frame: pd.DataFrame, column: str) -> list[float]:
    if frame.empty:
        return []
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return [float(value) for value in values.to_numpy(dtype=float)]


def profile_reason(trend: pd.DataFrame, primary: pd.DataFrame) -> str:
    if trend.empty and primary.empty:
        return "no_valid_trend_or_future_evidence"
    if trend.empty:
        return "no_valid_trend_evidence"
    if primary.empty:
        return "no_valid_primary_future_evidence"
    return ""


def unique_cycle_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame[["experiment_id", "experiment_date", "cycle_id"]].drop_duplicates().shape[0])


def sign_agreement(values: list[float], reference: float) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite or not np.isfinite(reference) or reference == 0:
        return np.nan
    return float(np.mean([np.sign(value) == np.sign(reference) for value in finite]))


def _spearman(left: pd.Series, right: pd.Series, minimum_points: int) -> float:
    valid = left.notna() & right.notna()
    if int(valid.sum()) < minimum_points:
        return np.nan
    x = left.loc[valid].astype(float)
    y = right.loc[valid].astype(float)
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    value = float(spearmanr(x.to_numpy(), y.to_numpy()).statistic)
    return value if np.isfinite(value) else np.nan


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else np.nan


def _iqr(values: list[float]) -> float:
    if not values:
        return np.nan
    numeric = np.asarray(values, dtype=float)
    return float(np.quantile(numeric, 0.75) - np.quantile(numeric, 0.25))
