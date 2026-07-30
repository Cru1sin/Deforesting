"""Physical and strictly past-looking derived features."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def calculate_derived_features(
    frame: pd.DataFrame, channels: Mapping[str, Mapping[str, Any]]
) -> pd.DataFrame:
    """Calculate only formulas named by the channel contract."""
    result = frame.copy()
    for name, settings in channels.items():
        if str(settings.get("kind")) != "derived":
            continue
        formula = str(settings.get("formula", ""))
        dependencies = [str(value) for value in settings.get("dependencies", [])]
        if formula != "cop" or len(dependencies) != 2:
            raise ValueError(f"unsupported derived formula for {name}: {formula}")
        numerator, denominator = dependencies
        imputed = _imputed_column(result, numerator) | _imputed_column(result, denominator)
        if numerator not in result or denominator not in result:
            result[name] = np.nan
            result[f"{name}__imputed"] = imputed
            continue
        denominator_values = pd.to_numeric(result[denominator], errors="coerce")
        numerator_values = pd.to_numeric(result[numerator], errors="coerce")
        valid_denominator = denominator_values.gt(0)
        result[name] = numerator_values.div(denominator_values).where(valid_denominator)
        result[f"{name}__imputed"] = imputed
    return result


def _imputed_column(frame: pd.DataFrame, name: str) -> pd.Series:
    column = f"{name}__imputed"
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].astype(bool)


def add_dynamic_features(
    frame: pd.DataFrame,
    channels: Mapping[str, Mapping[str, Any]],
    *,
    interval_seconds: int,
    windows_minutes: list[int],
) -> pd.DataFrame:
    """Add rolling, slope, and lag columns using observations strictly in the past."""
    result = frame.sort_values(
        ["experiment_id", "cycle_id", "cycle_stage", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    keys = ["experiment_id", "cycle_id", "cycle_stage"]
    for name, settings in channels.items():
        if not bool(settings.get("analysis_candidate", False)) or name not in result:
            continue
        for minutes in windows_minutes:
            steps = max(1, round(minutes * 60 / interval_seconds))
            grouped = result.groupby(keys, sort=False)[name]
            result[f"{name}__lag_{minutes}min"] = grouped.shift(steps)
            result[f"{name}__slope_{minutes}min"] = result[name] - grouped.shift(steps)
            shifted = grouped.shift(1)
            result[f"{name}__rolling_mean_{minutes}min"] = (
                shifted.groupby(
                    [result[key] for key in keys], sort=False, group_keys=False
                )
                .rolling(steps, min_periods=1)
                .mean()
                .reset_index(level=list(range(len(keys))), drop=True)
                .sort_index()
            )
    return result
