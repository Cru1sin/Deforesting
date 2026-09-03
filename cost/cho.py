"""Thin offline CHO composition over the shared V2.6.8 candidates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from . import cost_function_v2_6_8
from .fit_v2_6_8 import predict_from_artifact
from .objectives import add_single_objective_diagnostics, build_objectives
from .policy import TIE_BREAK, select_ch_pareto_knee

DEFAULT_POLICY_RECIPE: dict[str, object] = {
    "base_method": "ch_pareto_knee",
    "heat_basis": "water",
    "label_eligible": False,
    "outcome_model": "ticket_ridge_dynamic8",
    "candidate_step_seconds": 10,
    "allow_extrapolation": False,
    "reference_window": "within_5pct_of_both_ideals",
    "minimum_time_boundary": "stable_heating_start_when_available",
    "O_role": "reference_only",
    "tie_break": TIE_BREAK,
}


def _add_outcome(
    candidates: pd.DataFrame,
    artifact: dict[str, Any],
    experiment_id: str,
    *,
    value_column: str,
    prefix: str,
) -> None:
    prediction = predict_from_artifact(artifact, candidates, experiment_id)
    threshold = float(artifact["folds"][experiment_id]["support_threshold"])
    candidates[value_column] = prediction["prediction"].to_numpy()
    candidates[f"{prefix}_support_distance"] = prediction["support_distance"].to_numpy()
    candidates[f"{prefix}T_evaluable"] = np.isfinite(prediction["prediction"])
    candidates[f"{prefix}T_in_support"] = prediction["support_distance"].le(threshold).to_numpy()


def calculate_cycle(
    loader: Any,
    cycle_name: str,
    artifacts: Mapping[str, Any],
    *,
    step_seconds: int = 10,
    allow_extrapolation: bool = False,
) -> pd.DataFrame:
    """Build shared candidates, add EcompT/DT, then apply objectives and CH policy."""
    model_name = str(DEFAULT_POLICY_RECIPE["outcome_model"])
    model_set = artifacts["models"][model_name]
    candidates = cost_function_v2_6_8.build_candidate_outcomes(
        loader,
        cycle_name,
        cost_function_v2_6_8.DEFAULT_RECIPE,
        artifacts,
        candidate_step_seconds=step_seconds,
    )
    record = loader.get_cycle_record(cycle_name)
    experiment_id = str(record["experiment_id"])
    _add_outcome(
        candidates,
        model_set["compressor_energy"],
        experiment_id,
        value_column="transition_compressor_energy_kwh",
        prefix="Ecomp",
    )
    _add_outcome(
        candidates,
        model_set["duration"],
        experiment_id,
        value_column="transition_duration_minutes",
        prefix="D",
    )
    nested = record.get("boundaries")
    boundary_source = nested if isinstance(nested, Mapping) else record
    result = select_ch_pareto_knee(
        add_single_objective_diagnostics(build_objectives(candidates)),
        minimum_time=boundary_source.get("stable_heating_start"),
        allow_extrapolation=allow_extrapolation,
    )
    result["algorithm"] = result["base_method"] = "ch_pareto_knee"
    result["label_eligible"] = result["hard_label_eligible"] = False
    return result
