"""Strict no-frost baseline estimation with explicit failure reasons."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

BASELINE_FAILURE_REASONS = {
    "no_candidate_window",
    "missing_required_anchor",
    "insufficient_observed_coverage",
    "too_much_imputation",
    "unstable_anchor",
}


def add_baseline_residuals(
    frame: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    channels: Mapping[str, Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    """Add accepted baseline and ``current - baseline`` residual columns."""
    result = frame.copy()
    stage = str(settings.get("stage", "recovery"))
    minimum_coverage = float(settings.get("minimum_observed_coverage", 0.5))
    maximum_imputed_fraction = float(settings.get("maximum_imputed_fraction", 0.2))
    required_anchors = [str(value) for value in settings.get("required_anchor_channels", [])]
    eligible = _eligible_channels(channels)
    for name in eligible:
        result[f"{name}__baseline"] = np.nan
        result[f"{name}__baseline_residual"] = np.nan
        result[f"{name}__baseline_status"] = pd.Series(
            pd.NA, index=result.index, dtype="string"
        )
    for _, cycle in cycle_summary.iterrows():
        experiment_id = cycle["experiment_id"]
        cycle_id = cycle["cycle_id"]
        cycle_mask = result["experiment_id"].eq(experiment_id) & result["cycle_id"].eq(cycle_id)
        window_mask = cycle_mask & result["cycle_stage"].eq(stage)
        for name in eligible:
            status, baseline = _estimate_one(
                result.loc[window_mask],
                name,
                minimum_coverage=minimum_coverage,
                maximum_imputed_fraction=maximum_imputed_fraction,
                required_anchors=required_anchors,
                frame=result,
                window_mask=window_mask,
                maximum_baseline_std=float(settings.get("maximum_baseline_std", float("inf"))),
            )
            result.loc[cycle_mask, f"{name}__baseline_status"] = status
            if baseline is None:
                continue
            result.loc[cycle_mask, f"{name}__baseline"] = baseline
            values = pd.to_numeric(result.loc[cycle_mask, name], errors="coerce")
            result.loc[cycle_mask, f"{name}__baseline_residual"] = values - baseline
    return result


def _eligible_channels(channels: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        name
        for name, settings in channels.items()
        if str(settings.get("kind")) in {"continuous", "step", "derived"}
        and (
            bool(settings.get("analysis_candidate", False))
            or settings.get("role") == "performance"
        )
    ]


def _estimate_one(
    window: pd.DataFrame,
    name: str,
    *,
    minimum_coverage: float,
    maximum_imputed_fraction: float,
    required_anchors: list[str],
    frame: pd.DataFrame,
    window_mask: pd.Series,
    maximum_baseline_std: float,
) -> tuple[str, float | None]:
    if name not in window or window.empty:
        return "no_candidate_window", None
    for anchor in required_anchors:
        if anchor not in frame:
            return "missing_required_anchor", None
        anchor_values = pd.to_numeric(frame.loc[window_mask, anchor], errors="coerce")
        if not anchor_values.notna().any():
            return "missing_required_anchor", None
    values = pd.to_numeric(window[name], errors="coerce")
    observed = values.notna()
    if not observed.any():
        return "no_candidate_window", None
    coverage = float(observed.mean()) if len(values) else 0.0
    if coverage < minimum_coverage:
        return "insufficient_observed_coverage", None
    imputed = window.get(f"{name}__imputed", pd.Series(False, index=window.index)).astype(bool)
    if float(imputed.mean()) > maximum_imputed_fraction:
        return "too_much_imputation", None
    baseline = float(values.loc[observed].median())
    if not np.isfinite(baseline):
        return "no_candidate_window", None
    if float(values.loc[observed].std(ddof=0)) > maximum_baseline_std:
        return "unstable_anchor", None
    return "accepted", baseline
