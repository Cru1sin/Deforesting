"""Date-balanced Evidence summaries and feature-pair similarity."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from .contracts import (
    AGGREGATION_METHOD,
    FEATURE_PAIR_SIMILARITY_COLUMNS,
    FEATURE_PROFILE_COLUMNS,
    FUTURE_HORIZON_SUMMARY_COLUMNS,
)
from .metrics import spearman
from .settings import EvidenceSettings


def future_horizon_summary(
    associations: pd.DataFrame,
    features: Sequence[tuple[str, str]],
    settings: EvidenceSettings,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature, _ in features:
        for target in settings.targets:
            for horizon in settings.horizons_minutes:
                selected = associations.loc[
                    associations["feature"].eq(feature)
                    & associations["target"].eq(target)
                    & associations["horizon_minutes"].eq(horizon)
                ]
                effect, cycle_count, date_count = date_balanced_median(selected, "effect")
                degradation_support, _, _ = date_balanced_median(
                    selected, "degradation_support"
                )
                available = date_count > 0
                rows.append(
                    {
                        "feature": feature,
                        "target": target,
                        "horizon_minutes": horizon,
                        "effect": effect,
                        "degradation_support": degradation_support,
                        "valid_cycle_count": cycle_count,
                        "valid_date_count": date_count,
                        "aggregation_method": AGGREGATION_METHOD,
                        "metric_status": "available" if available else "unavailable",
                        "exclusion_reason": "" if available else "no_valid_dates",
                    }
                )
    return _frame(rows, FUTURE_HORIZON_SUMMARY_COLUMNS)


def feature_profile(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    features: Sequence[tuple[str, str]],
    settings: EvidenceSettings,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature, _ in features:
        selected = metrics.loc[
            metrics["feature"].eq(feature) & metrics["metric_status"].eq("available")
        ]
        signed_effect, cycle_count, date_count = date_balanced_median(
            selected, "signed_effect"
        )
        slope, _, _ = date_balanced_median(selected, "trend_slope_per_min")
        direction_consistency = _direction_consistency(selected)
        future = summary.loc[
            summary["feature"].eq(feature)
            & summary["target"].eq(settings.primary_target)
            & summary["horizon_minutes"].eq(settings.primary_horizon_minutes)
        ]
        if future.empty:
            future_degradation_support = np.nan
            future_cycle_count = 0
            future_date_count = 0
        else:
            future_row = future.iloc[0]
            future_degradation_support = future_row["degradation_support"]
            future_cycle_count = int(future_row["valid_cycle_count"])
            future_date_count = int(future_row["valid_date_count"])
        rows.append(
            {
                "feature": feature,
                "trend_valid_cycle_count": cycle_count,
                "trend_valid_date_count": date_count,
                "signed_effect": signed_effect,
                "direction_consistency": direction_consistency,
                "trend_slope_per_min": slope,
                "primary_target": settings.primary_target,
                "primary_horizon_minutes": settings.primary_horizon_minutes,
                "primary_future_degradation_support": future_degradation_support,
                "primary_future_valid_cycle_count": future_cycle_count,
                "primary_future_valid_date_count": future_date_count,
            }
        )
    return _frame(rows, FEATURE_PROFILE_COLUMNS)


def feature_pair_similarity(
    pair_inputs: Sequence[tuple[str, str, dict[str, dict[float, float]]]],
    features: Sequence[tuple[str, str]],
    settings: EvidenceSettings,
) -> pd.DataFrame:
    """Return dynamic co-trend similarity, not a redundancy decision."""
    rows: list[dict[str, object]] = []
    feature_names = [name for name, _ in features]
    for feature_a, feature_b in combinations(feature_names, 2):
        cycle_values: list[dict[str, object]] = []
        for cycle_name, experiment_date, values in pair_inputs:
            first = values.get(feature_a, {})
            second = values.get(feature_b, {})
            common = sorted(set(first).intersection(second))
            if len(common) < settings.minimum_feature_points:
                continue
            correlation = spearman(
                np.asarray([first[value] for value in common], dtype=float),
                np.asarray([second[value] for value in common], dtype=float),
            )
            if np.isfinite(correlation):
                cycle_values.append(
                    {
                        "cycle_name": cycle_name,
                        "experiment_date": experiment_date,
                        "value": abs(correlation),
                    }
                )
        selected = pd.DataFrame(cycle_values)
        effect, cycle_count, date_count = date_balanced_median(selected, "value")
        available = date_count > 0
        rows.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "valid_cycle_count": cycle_count,
                "valid_date_count": date_count,
                "median_abs_spearman": effect,
                "metric_status": "available" if available else "unavailable",
                "exclusion_reason": "" if available else "no_valid_dates",
            }
        )
    return _frame(rows, FEATURE_PAIR_SIMILARITY_COLUMNS)


def date_balanced_median(frame: pd.DataFrame, value_column: str) -> tuple[float, int, int]:
    if frame.empty or value_column not in frame:
        return np.nan, 0, 0
    if "metric_status" in frame:
        frame = frame.loc[frame["metric_status"].eq("available")]
        if frame.empty:
            return np.nan, 0, 0
    values = pd.to_numeric(frame[value_column], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    columns = ["experiment_date", value_column]
    if "cycle_name" in frame:
        columns.insert(1, "cycle_name")
    selected = frame.loc[finite, columns].copy()
    if selected.empty:
        return np.nan, 0, 0
    selected[value_column] = pd.to_numeric(selected[value_column], errors="coerce")
    if "cycle_name" in selected:
        cycle_values = (
            selected.groupby(["experiment_date", "cycle_name"], sort=False)[value_column]
            .median()
            .reset_index()
        )
        cycle_count = len(cycle_values)
    else:
        cycle_values = selected
        cycle_count = len(cycle_values)
    date_values = cycle_values.groupby("experiment_date", sort=False)[value_column].median()
    return float(date_values.median()), cycle_count, int(date_values.index.nunique())


def _direction_consistency(frame: pd.DataFrame) -> float:
    if frame.empty or "signed_effect" not in frame:
        return np.nan
    selected = frame.loc[:, ["experiment_date", "signed_effect"]].copy()
    if "cycle_name" in frame:
        selected["cycle_name"] = frame["cycle_name"]
    selected["signed_effect"] = pd.to_numeric(selected["signed_effect"], errors="coerce")
    selected = selected.loc[np.isfinite(selected["signed_effect"].to_numpy())]
    if selected.empty:
        return np.nan
    if "cycle_name" in selected:
        selected = (
            selected.groupby(["experiment_date", "cycle_name"], sort=False)[
                "signed_effect"
            ]
            .median()
            .reset_index()
        )
    date_medians = selected.groupby("experiment_date", sort=False)["signed_effect"].median()
    return float((date_medians > 0).mean())


def _frame(rows: list[dict[str, object]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns))
