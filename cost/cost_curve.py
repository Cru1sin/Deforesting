"""Validate recipes and assemble the four cost blocks into a standard curve."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def cycle_ratio(curve: pd.DataFrame) -> pd.DataFrame:
    """Return EH + ET, QH + QT, and their finite nonzero-denominator ratio."""
    result = curve.copy()
    result["total_energy_kwh"] = result["heating_energy_kwh"] + result["transition_energy_kwh"]
    result["total_heat_kwh"] = result["heating_heat_kwh"] + result["transition_heat_kwh"]
    valid = (
        np.isfinite(result["total_energy_kwh"])
        & np.isfinite(result["total_heat_kwh"])
        & result["total_heat_kwh"].ne(0)
    )
    result["inverse_cop"] = (result["total_energy_kwh"] / result["total_heat_kwh"]).where(valid)
    return result


def build_historical_cost_curve(  # noqa: C901
    boundaries: pd.DataFrame,
    heating_energy: pd.DataFrame,
    heating_heat: pd.DataFrame,
    transition_energy: pd.DataFrame,
    transition_heat: pd.DataFrame,
    recipe: Mapping[str, object],
) -> pd.DataFrame:
    """Apply the shared V1/V2.5 allow_historical_extrapolation curve policy."""
    checked = dict(recipe)
    lengths = {
        len(part)
        for part in (boundaries, heating_energy, heating_heat, transition_energy, transition_heat)
    }
    if len(lengths) != 1:
        raise ValueError("cost blocks must contain the same candidate rows")
    parts = [
        part.reset_index(drop=True)
        for part in (boundaries, heating_energy, heating_heat, transition_energy, transition_heat)
    ]
    column_sets = [set(map(str, part.columns)) for part in parts]
    duplicate = set().union(
        *(
            left & right
            for index, left in enumerate(column_sets)
            for right in column_sets[index + 1 :]
        )
    )
    if duplicate:
        raise ValueError(f"cost blocks contain duplicate columns: {sorted(duplicate)}")
    curve = pd.concat(parts, axis=1)
    if "defrost_heat_kwh" in curve and curve["defrost_heat_kwh"].gt(0).any():
        raise ValueError("defrost_heat_kwh must be signed and non-positive")
    curve = cycle_ratio(curve)
    positive = np.isfinite(curve["total_heat_kwh"]) & curve["total_heat_kwh"].gt(0)
    if "heating_valid" not in curve:
        curve["heating_valid"] = True
    for column in ("heating_energy_supported", "heating_heat_supported"):
        if column in curve:
            curve["heating_valid"] &= curve[column].fillna(False)
    et_evaluable = curve["ET_evaluable"].fillna(False)
    qt_evaluable = curve["QT_evaluable"].fillna(False)
    qt_physical = curve["QT_physical_valid"].fillna(False)
    curve["supported"] = (
        curve["heating_valid"].fillna(False)
        & et_evaluable
        & qt_evaluable
        & qt_physical
        & positive
    )
    curve["optimization_eligible"] = curve["supported"]
    curve["support_policy"] = "allow_historical_extrapolation"
    curve["relative_regret"] = np.nan
    curve["is_optimum"] = False
    for _, positions in curve.groupby("cycle_name", sort=False).groups.items():
        eligible = curve.index.isin(positions) & curve["optimization_eligible"]
        if not eligible.any():
            continue
        optimum = curve.loc[eligible, "inverse_cop"].idxmin()
        minimum = float(curve.loc[optimum, "inverse_cop"])
        curve.loc[eligible, "relative_regret"] = curve.loc[eligible, "inverse_cop"] / minimum - 1
        curve.loc[optimum, "is_optimum"] = True
    curve["near_optimal_1pct"] = curve["optimization_eligible"] & curve["relative_regret"].le(0.01)
    curve["near_optimal_5pct"] = curve["optimization_eligible"] & curve["relative_regret"].le(0.05)
    curve["base_cost"] = checked["base_cost"]
    curve["variant"] = checked["variant"]
    curve["label_eligible"] = checked["label_eligible"]
    return curve
