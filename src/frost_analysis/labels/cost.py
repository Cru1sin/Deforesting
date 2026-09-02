"""Assign image states from candidate-level empirical cost regret."""

from __future__ import annotations

import pandas as pd

from labels.build import (
    _curve_support,
    assign_image_cost_states,
    complete_catalog_cycle_names,
    complete_observed_cycle_names,
    curve_label_exclusion_reason,
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


def high_confidence_coverage(
    label_balance: pd.DataFrame, camera_group: str, threshold: float
) -> float:
    """Return retained pre/post images as a fraction of candidate-domain images."""
    rows = label_balance.loc[
        label_balance["camera_group"].eq(camera_group)
        & label_balance["regret_threshold"].eq(threshold)
        & label_balance["cost_state"].isin(("pre_optimal", "near_optimal", "post_optimal"))
    ]
    retained = rows.loc[rows["cost_state"].ne("near_optimal"), "image_count"].sum()
    return float(retained / rows["image_count"].sum())
