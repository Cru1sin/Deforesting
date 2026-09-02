"""Calculate J = (EH + ET) / (QH + QT) for canonical V2.5 candidates.

Candidates run from stable heating + 10 min through actual preparation. EH and QH are
measured heating blocks; ET and QT are the transition blocks. The decision is the minimum
eligible inverse COP, with 1% and 5% near-optimal sets. EH/QH include the observed heating
prefix; canonical states use formal interpolation and strict variants use [tau-60 s, tau).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .boundaries import build_candidate_boundaries
from .cost_curve import build_historical_cost_curve
from .energy_models import heating_energy, transition_energy
from .heat_models import heating_heat, transition_heat_v2_5, zero_transition_heat

DEFAULT_RECIPE: dict[str, object] = {
    "base_cost": "v2.5",
    "version": "v2.5",
    "variant": None,
    "label_eligible": False,
    "heat_basis": "water",
    "event_scope": "heating_start_to_actual_preparation",
    "heating_start_rule": "heating_start",
    "integration_protocol": "historical_reconstruction",
    "state_protocol": "historical_interpolation",
    "candidate_start_rule": "stable_heating_start_plus_10_minutes",
    "candidate_end_rule": "observed_defrost_preparation_start",
    "candidate_cadence": "1_minute_plus_exact_endpoint",
    "state_window": "[tau-60s,tau)",
    "transition_scope": "preparation_defrost_recovery",
    "transition_window": "observed_preparation_and_defrost_durations",
    "transition_provenance": "offline_diagnostic_future_boundary_observed_durations",
    "decision_rule": "supported_argmin_inverse_cop",
    "heating_energy_model": "measured_total_power",
    "heating_heat_model": "measured_water_heat",
    "transition_energy_model": "pe_quadratic",
    "transition_heat_model": "linear_qprep_plus_signed_quadratic_qd",
}


def calculate_cycle(
    loader: Any, cycle_name: str, recipe: Mapping[str, object] | None = None
) -> pd.DataFrame:
    """Calculate one V2.5 or named V2.5-based variant curve from Dataset raw data."""
    checked = validate_recipe(DEFAULT_RECIPE if recipe is None else recipe)
    frame = loader.load_cycle_original(cycle_name)
    record = loader.get_cycle_record(cycle_name)

    boundary = build_candidate_boundaries(loader, cycle_name, str(checked["heating_start_rule"]))
    integration_protocol = str(checked["integration_protocol"])
    state_protocol = str(checked["state_protocol"])
    historical_start = pd.Timestamp(boundary["stable_heating_start"].iloc[0])
    eh = heating_energy(
        frame,
        boundary,
        integration_protocol,
        historical_start=historical_start,
    )
    qh = heating_heat(
        frame,
        boundary,
        "unit" if checked["heating_heat_model"] == "measured_unit_heat" else "water",
        integration_protocol,
        historical_start=historical_start,
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
    return build_historical_cost_curve(boundary, eh, qh, et, qt, checked)


def calculate(
    loader: Any,
    cycle_names: Sequence[str],
    recipe: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Calculate V2.5 curves for the requested Dataset cycles."""
    tables = [calculate_cycle(loader, cycle_name, recipe) for cycle_name in cycle_names]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def validate_recipe(recipe: Mapping[str, object]) -> dict[str, object]:
    """Validate only recipe combinations implemented by the V2.5 executor."""
    value = dict(recipe)
    if value.keys() != DEFAULT_RECIPE.keys():
        raise ValueError("recipe has missing or unexpected parameters")
    if value["base_cost"] != "v2.5" or value["version"] != "v2.5":
        raise ValueError("V2.5 module requires base_cost and version v2.5")
    if (value["heat_basis"], value["heating_heat_model"]) not in {
        ("unit", "measured_unit_heat"),
        ("water", "measured_water_heat"),
    }:
        raise ValueError("heat basis and heating heat model are incompatible")
    choices = {
        "integration_protocol": {"historical_reconstruction", "strict_causal"},
        "state_protocol": {"historical_interpolation", "strict_causal"},
        "transition_energy_model": {"pe_quadratic_plus_fixed_recovery", "pe_quadratic"},
        "transition_heat_model": {
            "zero_transition_heat",
            "linear_qprep_plus_signed_quadratic_qd",
        },
    }
    invalid = next((key for key, allowed in choices.items() if value[key] not in allowed), None)
    if invalid:
        raise ValueError(f"v2.5 does not implement {invalid.replace('_', ' ')}={value[invalid]}")
    fixed_recovery = value["transition_energy_model"] == "pe_quadratic_plus_fixed_recovery"
    observed = value["transition_heat_model"] == "linear_qprep_plus_signed_quadratic_qd"
    if observed:
        value["transition_window"] = "observed_preparation_and_defrost_durations"
        value["transition_provenance"] = (
            "offline_diagnostic_future_boundary_observed_durations"
            + ("_plus_fixed_recovery" if fixed_recovery else "")
        )
    else:
        value["transition_window"] = "candidate_state_at_tau"
        value["transition_provenance"] = "candidate_time_state" + (
            "_plus_fixed_recovery" if fixed_recovery else ""
        )
    optional = {
        *choices,
        "heat_basis",
        "heating_heat_model",
        "transition_window",
        "transition_provenance",
    }
    fixed = DEFAULT_RECIPE.keys() - optional - {"variant", "label_eligible"}
    if any(value[key] != DEFAULT_RECIPE[key] for key in fixed):
        raise ValueError("v2.5 does not implement fixed recipe overrides")
    variant = value["variant"]
    if variant is not None and (not isinstance(variant, str) or not variant.strip()):
        raise ValueError("variant must be a non-empty string")
    changed = any(value[key] != DEFAULT_RECIPE[key] for key in optional)
    if changed != (variant is not None):
        raise ValueError("named variant is inconsistent with recipe overrides")
    if variant is None and value["label_eligible"] is not False:
        raise ValueError("canonical label_eligible status cannot be changed")
    value["label_eligible"] = False
    return value
