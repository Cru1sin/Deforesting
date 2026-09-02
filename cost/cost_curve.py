"""Validate recipes and assemble the four cost blocks into a standard curve."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def build_cost_curve(  # noqa: C901
    boundaries: pd.DataFrame,
    heating_energy: pd.DataFrame,
    heating_heat: pd.DataFrame,
    transition_energy: pd.DataFrame,
    transition_heat: pd.DataFrame,
    recipe: Mapping[str, object],
) -> pd.DataFrame:
    """Combine EH, QH, ET, and QT and mark supported optima per cycle."""
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
    for column in (
        "preparation_energy_kwh",
        "defrost_energy_kwh",
        "recovery_energy_kwh",
        "preparation_heat_kwh",
        "defrost_heat_kwh",
        "recovery_heat_kwh",
    ):
        if column not in curve:
            curve[column] = 0.0
    numerator = curve["heating_energy_kwh"] + curve["transition_energy_kwh"]
    denominator = curve["heating_heat_kwh"] + curve["transition_heat_kwh"]
    positive = np.isfinite(denominator) & denominator.gt(0)
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
    curve["inverse_cop"] = (numerator / denominator).where(positive)
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
