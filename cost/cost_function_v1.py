"""Canonical V1 reconstructs EH/QH from stable heating through tau.

Candidates begin at stable+10 min and end exactly at actual preparation.
Canonical Pe reproduces the formal historical interpolation; strict variants use [tau-60 s, tau).
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
    "version": "v1",
    "variant": None,
    "label_eligible": True,
    "heat_basis": "unit",
    "event_scope": "stable_heating_start_to_actual_preparation",
    "heating_start_rule": "stable_heating_start",
    "integration_protocol": "historical_reconstruction",
    "state_protocol": "historical_interpolation",
    "candidate_start_rule": "stable_heating_start_plus_10_minutes",
    "candidate_end_rule": "observed_defrost_preparation_start",
    "candidate_cadence": "1_minute_plus_exact_endpoint",
    "state_window": "[tau-60s,tau)",
    "transition_scope": "preparation_defrost_recovery",
    "transition_window": "candidate_state_at_tau",
    "transition_provenance": "candidate_time_state_plus_fixed_recovery",
    "decision_rule": "supported_argmin_inverse_cop",
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
    integration_protocol = str(checked["integration_protocol"])
    state_protocol = str(checked["state_protocol"])
    eh = heating_energy(frame, boundary, integration_protocol)
    qh = heating_heat(
        frame,
        boundary,
        "unit" if checked["heating_heat_model"] == "measured_unit_heat" else "water",
        integration_protocol,
    )
    et = transition_energy(
        frame,
        boundary,
        str(record["experiment_id"]),
        include_fixed_recovery=checked["transition_energy_model"]
        == "pe_quadratic_plus_fixed_recovery",
        state_protocol=state_protocol,
    )
    qt = (
        zero_transition_heat(len(boundary))
        if checked["transition_heat_model"] == "zero_transition_heat"
        else transition_heat_v2_5(frame, boundary, record, state_protocol)
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
