"""Cycle-local early stable baseline proxy selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from .config import BaselineSettings

BASELINE_FAILURE_REASONS = {
    "no_candidate_window",
    "missing_required_anchor",
    "insufficient_observed_coverage",
    "too_much_imputation",
    "unstable_anchor",
}
BASELINE_REFERENCE_TYPE = "cycle_local_early_stable_proxy"


def add_baseline_residuals(
    frame: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    channels: Mapping[str, Mapping[str, Any]],
    settings: BaselineSettings | Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add common-window baselines and return the enriched cycle summary."""
    rules = _settings(settings)
    result = frame.copy()
    summary = cycle_summary.copy()
    eligible = _eligible_channels(channels)
    for name in eligible:
        result[f"{name}__baseline"] = np.nan
        result[f"{name}__baseline_residual"] = np.nan
    summary = _initialise_summary(summary)
    for index, cycle in summary.iterrows():
        cycle_mask = result["experiment_id"].eq(cycle["experiment_id"]) & result[
            "cycle_id"
        ].eq(cycle["cycle_id"])
        if cycle.get("cycle_status") != "valid":
            summary.loc[index, "baseline_status"] = "not_applicable"
            summary.loc[index, "baseline_failure_reason"] = "cycle_not_valid"
            continue
        window, reason = _find_common_window(result.loc[cycle_mask], cycle, rules)
        if window is None:
            summary.loc[index, "baseline_status"] = "unavailable"
            summary.loc[index, "baseline_failure_reason"] = reason
            continue
        start, end, anchor_window = window
        summary.loc[index, "baseline_status"] = "available"
        summary.loc[index, "baseline_failure_reason"] = ""
        summary.loc[index, "baseline_start"] = start
        summary.loc[index, "baseline_end"] = end
        unavailable: list[str] = []
        for name in eligible:
            baseline = _channel_baseline(anchor_window, name, rules)
            if baseline is None:
                unavailable.append(name)
                continue
            result.loc[cycle_mask, f"{name}__baseline"] = baseline
            values = pd.to_numeric(result.loc[cycle_mask, name], errors="coerce")
            result.loc[cycle_mask, f"{name}__baseline_residual"] = values - baseline
        summary.at[index, "baseline_unavailable_channels"] = cast(Any, unavailable)
    return result, summary


def _settings(settings: BaselineSettings | Mapping[str, Any]) -> BaselineSettings:
    if isinstance(settings, BaselineSettings):
        return settings
    return BaselineSettings.from_mapping(settings)


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


def _initialise_summary(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    defaults: dict[str, object] = {
        "baseline_status": pd.NA,
        "baseline_failure_reason": pd.NA,
        "baseline_reference_type": BASELINE_REFERENCE_TYPE,
        "baseline_start": pd.NaT,
        "baseline_end": pd.NaT,
        "baseline_unavailable_channels": None,
    }
    for column, value in defaults.items():
        if column not in result:
            if column == "baseline_unavailable_channels":
                result[column] = pd.Series([None] * len(result), index=result.index, dtype=object)
            else:
                result[column] = cast(Any, value)
    return result


def _find_common_window(
    cycle_frame: pd.DataFrame, cycle: pd.Series, settings: BaselineSettings
) -> tuple[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame], str] | tuple[None, str]:
    stable = _timestamp_or_none(cycle.get("stable_heating_start"))
    defrost = _timestamp_or_none(cycle.get("defrost_start"))
    if stable is None:
        return None, "no_candidate_window"

    start = stable
    end = stable + pd.Timedelta(seconds=settings.baseline_seconds)
    timestamp_values = (
        cycle_frame["timestamp"]
        if "timestamp" in cycle_frame
        else pd.Series(dtype="datetime64[ns]")
    )
    timestamps = pd.to_datetime(timestamp_values, errors="coerce").dropna()
    if timestamps.empty or timestamps.max() < end:
        return None, "no_candidate_window"
    if defrost is not None and end > defrost:
        return None, "no_candidate_window"
    stage = cycle_frame.loc[cycle_frame["cycle_stage"].eq(settings.stage)].copy()
    window = stage.loc[stage["timestamp"].ge(start) & stage["timestamp"].lt(end)]
    valid, failure = _anchors_are_stable(window, settings)
    if valid:
        return (start, end, window), ""
    return None, failure


def _anchors_are_stable(
    window: pd.DataFrame, settings: BaselineSettings
) -> tuple[bool, str]:
    if window.empty:
        return False, "no_candidate_window"
    for anchor in settings.required_anchor_channels:
        if anchor not in window:
            return False, "missing_required_anchor"
        values = pd.to_numeric(window[anchor], errors="coerce")
        imputed = _imputed_column(window, anchor)
        if not values.notna().any():
            return False, "missing_required_anchor"
        if imputed.any():
            return False, "too_much_imputation"
        coverage = float(values.notna().mean())
        if coverage < settings.minimum_observed_coverage:
            return False, "insufficient_observed_coverage"
        maximum_std = settings.anchor_maximum_std.get(anchor)
        if maximum_std is not None and float(values.dropna().std(ddof=0)) > maximum_std:
            return False, "unstable_anchor"
    return True, ""


def _channel_baseline(
    window: pd.DataFrame, name: str, settings: BaselineSettings
) -> float | None:
    if name not in window:
        return None
    values = pd.to_numeric(window[name], errors="coerce")
    observed = values.notna() & ~_imputed_column(window, name)
    coverage = float(observed.mean()) if len(values) else 0.0
    if coverage < settings.minimum_observed_coverage:
        return None
    finite = values.loc[observed]
    if finite.empty:
        return None
    baseline = float(finite.median())
    if not np.isfinite(baseline):
        return None
    return baseline


def _imputed_column(frame: pd.DataFrame, name: str) -> pd.Series:
    column = f"{name}__imputed"
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _timestamp_or_none(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value)
