"""Fixed physical formulas for Process."""

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
    """Calculate only formulas named by the channel definitions."""
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
