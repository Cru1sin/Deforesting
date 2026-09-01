"""Validate recipes and assemble the four cost blocks into a standard curve."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def validate_recipe(recipe: Mapping[str, object]) -> dict[str, object]:  # noqa: C901
    """Validate a V1/V2.5 canonical recipe or a named single-base variant."""
    value = dict(recipe)
    base = value.get("base_cost")
    canonical = {
        "v1": {
            "version": "v1",
            "label_eligible": True,
            "heat_basis": "unit",
            "event_scope": "stable_heating_start_to_actual_preparation",
            "heating_start_rule": "stable_heating_start",
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
        },
        "v2.5": {
            "version": "v2.5",
            "label_eligible": False,
            "heat_basis": "water",
            "event_scope": "heating_start_to_actual_preparation",
            "heating_start_rule": "heating_start",
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
        },
    }
    if base not in canonical:
        raise ValueError("base_cost must be v1 or v2.5")
    required = {"base_cost", "variant", *canonical[str(base)]}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"recipe is missing parameters: {sorted(missing)}")
    unexpected = value.keys() - required
    if unexpected:
        raise ValueError(f"recipe has unexpected parameters: {sorted(unexpected)}")
    domains = {
        "heat_basis": {"unit", "water"},
        "event_scope": {
            "stable_heating_start_to_actual_preparation",
            "heating_start_to_actual_preparation",
        },
        "heating_start_rule": {"stable_heating_start", "heating_start"},
        "candidate_start_rule": {"stable_heating_start_plus_10_minutes"},
        "candidate_end_rule": {"observed_defrost_preparation_start"},
        "candidate_cadence": {"1_minute_plus_exact_endpoint"},
        "state_window": {"[tau-60s,tau)"},
        "transition_scope": {"preparation_defrost_recovery"},
        "transition_window": {
            "candidate_state_at_tau",
            "observed_preparation_and_defrost_durations",
        },
        "transition_provenance": {
            "candidate_time_state_plus_fixed_recovery",
            "offline_diagnostic_future_boundary_observed_durations",
        },
        "decision_rule": {"supported_argmin_inverse_cop"},
    }
    for key, choices in domains.items():
        if value[key] not in choices:
            raise ValueError(f"unknown {key.replace('_', ' ')}")
    if value["version"] != base:
        raise ValueError("version must match base_cost")
    if not isinstance(value["label_eligible"], bool):
        raise ValueError("label_eligible must be Boolean")
    allowed = {
        "heating_energy_model": {"measured_total_power"},
        "heating_heat_model": {"measured_unit_heat", "measured_water_heat"},
        "transition_energy_model": {"pe_quadratic_plus_fixed_recovery", "pe_quadratic"},
        "transition_heat_model": {
            "zero_transition_heat",
            "linear_qprep_plus_signed_quadratic_qd",
        },
    }
    for key, choices in allowed.items():
        if value[key] not in choices:
            raise ValueError(f"unknown {key.replace('_', ' ')}")
    expected_heat_model = f"measured_{value['heat_basis']}_heat"
    if value["heating_heat_model"] != expected_heat_model:
        raise ValueError("heat basis and heating heat model are incompatible")
    expected_scope = f"{value['heating_start_rule']}_to_actual_preparation"
    if value["event_scope"] != expected_scope:
        raise ValueError("event scope and start rule are incompatible")
    differences = {
        key
        for key, expected in canonical[str(base)].items()
        if key != "label_eligible" and value.get(key) != expected
    }
    variant = value.get("variant")
    if not variant:
        labels = {
            "heat_basis": "heat basis",
            "event_scope": "event scope",
            "heating_start_rule": "start rule",
        }
        for key in differences:
            if key in labels:
                raise ValueError(f"incompatible {labels[key]}")
        if differences:
            raise ValueError("component override requires a named variant")
    if not differences and variant:
        raise ValueError("canonical recipe cannot set variant")
    if variant:
        value["label_eligible"] = False
    elif value["label_eligible"] != canonical[str(base)]["label_eligible"]:
        raise ValueError("canonical label_eligible status cannot be changed")
    if variant is not None and (not isinstance(variant, str) or not variant.strip()):
        raise ValueError("variant must be a non-empty string")
    return value


def build_cost_curve(  # noqa: C901
    boundaries: pd.DataFrame,
    heating_energy: pd.DataFrame,
    heating_heat: pd.DataFrame,
    transition_energy: pd.DataFrame,
    transition_heat: pd.DataFrame,
    recipe: Mapping[str, object],
) -> pd.DataFrame:
    """Combine EH, QH, ET, and QT and mark supported optima per cycle."""
    checked = validate_recipe(recipe)
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
    curve["supported"] = (
        curve["heating_valid"].fillna(False)
        & curve["ET_supported"].fillna(False)
        & curve["QT_supported"].fillna(False)
        & positive
    )
    curve["optimization_eligible"] = curve["supported"]
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
