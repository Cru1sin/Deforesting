"""Select the cycle-COP–heating-rate Pareto compromise.

Pareto dominance uses only cycle_cop and cycle_heating_rate_kw.
cycle_evaporator_capacity_kw is diagnostic and never moves the selection.
Equal scores select the earlier candidate defrost time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .selection_results import add_selected_time_fields, five_minute_support_runs

TIE_BREAK = "earliest_candidate"


def select_cop_heating_rate_pareto_knee(
    objectives: pd.DataFrame,
    *,
    minimum_time: object = None,
) -> pd.DataFrame:
    """Select a two-objective Pareto knee without allowing O to move it."""
    result = objectives.sort_values("candidate_defrost_time", kind="stable").reset_index(drop=True).copy()
    after_minimum = pd.Series(True, index=result.index)
    if minimum_time is not None:
        after_minimum = pd.to_datetime(result["candidate_defrost_time"]).ge(pd.Timestamp(minimum_time))
    for name in ("cycle_cop", "cycle_heating_rate_kw", "cycle_evaporator_capacity_kw"):
        candidate = after_minimum & result[f"{name}_eligible"].fillna(False) & np.isfinite(
            result[name]
        )
        eligible = candidate & five_minute_support_runs(result["candidate_defrost_time"], candidate)
        result[f"{name}_eligible"] = eligible
        result[f"{name}_uses_model_extrapolation"] &= eligible

    result["within_5pct_of_best_cop_and_heating_rate"] = False
    c_eligible = result["cycle_cop_eligible"]
    if c_eligible.any():
        best_c = float(result.loc[c_eligible, "cycle_cop"].max())
        h_eligible = result["cycle_heating_rate_kw_eligible"]
        if h_eligible.any():
            best_h = float(result.loc[h_eligible, "cycle_heating_rate_kw"].max())
            result["within_5pct_of_best_cop_and_heating_rate"] = (
                c_eligible
                & h_eligible
                & result["cycle_cop"].ge(0.95 * best_c)
                & result["cycle_heating_rate_kw"].ge(0.95 * best_h)
            )

    result["is_cop_heating_rate_pareto_point"] = False
    result["is_latest_pareto_point"] = False
    result["is_selected_pareto_point"] = False
    result["pareto_selection_score"] = np.nan
    result["pareto_selection_method"] = "abstain_no_common_domain"
    common = result["cycle_cop_eligible"] & result["cycle_heating_rate_kw_eligible"]
    if not common.any():
        return add_selected_time_fields(
            result,
            result["is_selected_pareto_point"],
            method="cop_heating_rate_pareto_knee",
            selected_reason="normalized_chord_distance",
            abstain_reason="no_common_domain",
            score=result["pareto_selection_score"],
            model_supported=result["cycle_cop_eligible_without_extrapolation"]
            & result["cycle_heating_rate_kw_eligible_without_extrapolation"],
        )

    positions = result.index[common].to_numpy()
    c_values = result.loc[positions, "cycle_cop"].to_numpy(dtype=float)
    h_values = result.loc[positions, "cycle_heating_rate_kw"].to_numpy(dtype=float)
    front = np.ones(len(positions), dtype=bool)
    for index, (c_value, h_value) in enumerate(zip(c_values, h_values, strict=True)):
        front[index] = not np.any(
            (c_values >= c_value)
            & (h_values >= h_value)
            & ((c_values > c_value) | (h_values > h_value))
        )
    front_positions = positions[front]
    result.loc[front_positions, "is_cop_heating_rate_pareto_point"] = True
    result.loc[front_positions[-1], "is_latest_pareto_point"] = True

    front_c = result.loc[front_positions, "cycle_cop"].to_numpy(dtype=float)
    front_h = result.loc[front_positions, "cycle_heating_rate_kw"].to_numpy(dtype=float)
    c_span = float(front_c.max() - front_c.min())
    h_span = float(front_h.max() - front_h.min())
    method = "relative_ideal_distance_fallback"
    scores = -np.hypot(
        (front_c.max() - front_c) / max(abs(float(front_c.max())), 1e-12),
        (front_h.max() - front_h) / max(abs(float(front_h.max())), 1e-12),
    )
    if len(front_positions) >= 3 and c_span > 0 and h_span > 0:
        chord = (
            (front_c - front_c.min()) / c_span
            + (front_h - front_h.min()) / h_span
            - 1
        ) / np.sqrt(2)
        if float(chord.max()) > 1e-12:
            scores = chord
            method = "normalized_chord_distance"
    result.loc[front_positions, "pareto_selection_score"] = scores
    best = float(scores.max())
    knee = int(front_positions[np.flatnonzero(scores == best)[0]])
    result.loc[knee, "is_selected_pareto_point"] = True
    result["pareto_selection_method"] = method
    return add_selected_time_fields(
        result,
        result["is_selected_pareto_point"],
        method="cop_heating_rate_pareto_knee",
        selected_reason=method,
        abstain_reason="no_common_domain",
        score=result["pareto_selection_score"],
        model_supported=result["cycle_cop_eligible_without_extrapolation"]
        & result["cycle_heating_rate_kw_eligible_without_extrapolation"],
    )
