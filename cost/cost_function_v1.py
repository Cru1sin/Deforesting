"""V1 uses [stable_heating_start, tau] for EH/QH.

Candidates begin at stable+10 min and end exactly at actual preparation.
Pe features use the strict interval [tau-60 s, tau).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .boundaries import build_candidate_boundaries
from .cost_curve import build_cost_curve, validate_recipe
from .energy_models import heating_energy, transition_energy
from .heat_models import heating_heat, transition_heat_v2_5, zero_transition_heat

DEFAULT_RECIPE: dict[str, object] = {
    "base_cost": "v1",
    "variant": None,
    "heat_basis": "unit",
    "event_scope": "stable_heating_start_to_actual_preparation",
    "heating_start_rule": "stable_heating_start",
    "heating_energy_model": "measured_total_power",
    "heating_heat_model": "measured_unit_heat",
    "transition_energy_model": "pe_quadratic_plus_fixed_recovery",
    "transition_heat_model": "zero_transition_heat",
}


def calculate_cycle(
    loader: Any, cycle_name: str, recipe: Mapping[str, object] | None = None
) -> pd.DataFrame:
    """Calculate one V1 or named V1-based variant curve from Dataset raw data."""
    checked = validate_recipe(DEFAULT_RECIPE if recipe is None else recipe)
    if checked["base_cost"] != "v1":
        raise ValueError("V1 module requires base_cost v1")
    frame = loader.load_cycle_original(cycle_name)
    record = loader.get_cycle_record(cycle_name)

    boundary = build_candidate_boundaries(loader, cycle_name, str(checked["heating_start_rule"]))
    eh = heating_energy(frame, boundary)
    qh = heating_heat(
        frame,
        boundary,
        "unit" if checked["heating_heat_model"] == "measured_unit_heat" else "water",
    )
    et = transition_energy(
        frame,
        boundary,
        str(record["experiment_id"]),
        include_fixed_recovery=checked["transition_energy_model"]
        == "pe_quadratic_plus_fixed_recovery",
    )
    qt = (
        zero_transition_heat(len(boundary))
        if checked["transition_heat_model"] == "zero_transition_heat"
        else transition_heat_v2_5(frame, boundary, record)
    )
    return build_cost_curve(boundary, eh, qh, et, qt, checked)


def calculate(
    loader: Any,
    cycle_names: Sequence[str],
    recipe: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Calculate V1 curves for the requested Dataset cycles."""
    tables = [calculate_cycle(loader, cycle_name, recipe) for cycle_name in cycle_names]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
