"""Fixed physical formulas and strictly past-looking dynamic features."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

_FORMULAS = {
    "cop",
    "evaporator_capacity",
    "pressure_ratio",
    "water_delta_temperature",
    "superheat_calculated",
}


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
        if formula not in _FORMULAS:
            raise ValueError(f"unsupported derived formula for {name}: {formula}")
        result[name] = _calculate_formula(result, formula, dependencies)
        result[f"{name}__imputed"] = _dependency_imputed(result, dependencies)
    return result


def _calculate_formula(frame: pd.DataFrame, formula: str, dependencies: list[str]) -> pd.Series:
    if any(dependency not in frame for dependency in dependencies):
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = [pd.to_numeric(frame[dependency], errors="coerce") for dependency in dependencies]
    if formula == "cop":
        return values[0].div(values[1]).where(values[1].gt(0))
    if formula == "evaporator_capacity":
        return values[0] - values[1]
    if formula == "pressure_ratio":
        return values[0].div(values[1]).where(values[1].gt(0))
    if formula == "water_delta_temperature":
        return values[0] - values[1]
    if formula == "superheat_calculated":
        return values[0] - values[1]
    raise ValueError(f"unsupported derived formula: {formula}")


def _dependency_imputed(frame: pd.DataFrame, dependencies: list[str]) -> pd.Series:
    result = pd.Series(False, index=frame.index, dtype=bool)
    for dependency in dependencies:
        column = f"{dependency}__imputed"
        if column in frame:
            result = result | frame[column].fillna(False).astype(bool)
    return result


def recompute_dynamic_features(
    frame: pd.DataFrame,
    channels: Mapping[str, Mapping[str, Any]],
    *,
    interval_seconds: int,
    windows_minutes: list[int],
) -> pd.DataFrame:
    """Add lag, delta, and full past-only rolling means within each partition."""
    result = frame.sort_values(
        ["experiment_id", "cycle_id", "cycle_stage", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    for name, settings in channels.items():
        if not bool(settings.get("analysis_candidate", False)) or name not in result:
            continue
        for minutes in windows_minutes:
            steps = max(1, round(minutes * 60 / interval_seconds))
            _add_window_features(result, name, minutes, steps)
    return result


def _add_window_features(frame: pd.DataFrame, name: str, minutes: int, steps: int) -> None:
    keys = ["experiment_id", "cycle_id", "cycle_stage"]
    lag_column = f"{name}__lag_{minutes}min"
    delta_column = f"{name}__delta_{minutes}min"
    rolling_column = f"{name}__rolling_mean_{minutes}min"
    frame[lag_column] = np.nan
    frame[delta_column] = np.nan
    frame[rolling_column] = np.nan
    for _, group in frame.groupby(keys, sort=False, dropna=False):
        indices = group.index
        values = pd.to_numeric(group[name], errors="coerce")
        lag = values.shift(steps)
        rolling = values.shift(1).rolling(steps, min_periods=steps).mean()
        frame.loc[indices, lag_column] = lag.to_numpy()
        frame.loc[indices, delta_column] = (values - lag).to_numpy()
        frame.loc[indices, rolling_column] = rolling.to_numpy()
