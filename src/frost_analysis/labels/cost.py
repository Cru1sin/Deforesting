"""Assign image states from candidate-level empirical cost regret."""

from __future__ import annotations

import pandas as pd

from labels.build import (
    _curve_support,
    assign_image_cost_states,
    complete_catalog_cycle_names,
    complete_observed_cycle_names,
    curve_label_exclusion_reason,
    high_confidence_coverage,
)

__all__ = [
    "_curve_support",
    "assign_image_cost_states",
    "complete_catalog_cycle_names",
    "complete_observed_cycle_names",
    "curve_label_exclusion_reason",
    "high_confidence_coverage",
    "map_cost_state_targets",
]


def map_cost_state_targets(states: pd.Series, task: str) -> pd.Series:
    """Map shared cost states to dense binary or three-class targets."""
    names = (
        ("pre_optimal", "post_optimal")
        if task == "binary"
        else ("pre_optimal", "near_optimal", "post_optimal")
    )
    if task not in {"binary", "three"}:
        raise ValueError(f"unknown classification task: {task}")
    return states.map({name: index for index, name in enumerate(names)}).astype("Int64")
