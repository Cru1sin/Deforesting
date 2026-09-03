"""Pure CH Pareto-knee policy; O is diagnostic only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cost_curve import add_selection_contract, five_minute_support_runs

TIE_BREAK = "earliest_candidate"


def select_ch_pareto_knee(
    objectives: pd.DataFrame,
    *,
    minimum_time: object = None,
    allow_extrapolation: bool = False,
) -> pd.DataFrame:
    """Select a two-objective C/H Pareto knee without allowing O to move it."""
    result = objectives.sort_values("candidate_time", kind="stable").reset_index(drop=True).copy()
    after_minimum = pd.Series(True, index=result.index)
    if minimum_time is not None:
        after_minimum = pd.to_datetime(result["candidate_time"]).ge(pd.Timestamp(minimum_time))
    state_valid = result["pre_action_window_valid"].fillna(False)
    for prefix in ("C", "H", "O"):
        native = result[f"{prefix}_native_eligible"].fillna(False)
        model = result[f"{prefix}_model_supported"]
        measurement = result[f"{prefix}_measurement_eligible"].fillna(False)
        physical = result[f"{prefix}_physical_valid"].fillna(False)
        finite = np.isfinite(result[prefix])
        extrapolated = (
            bool(allow_extrapolation)
            & model.eq(False)
            & measurement
            & physical
            & state_valid
            & finite
        )
        candidate = after_minimum & state_valid & finite & (native | extrapolated)
        eligible = candidate & five_minute_support_runs(result["candidate_time"], candidate)
        result[f"{prefix}_eligible"] = eligible
        result[f"{prefix}_extrapolated"] = eligible & ~native

    result["within_5pct_of_both_ideals"] = False
    c_eligible = result["C_eligible"]
    if c_eligible.any():
        best_c = float(result.loc[c_eligible, "C"].max())
        h_eligible = result["H_eligible"]
        if h_eligible.any():
            best_h = float(result.loc[h_eligible, "H"].max())
            result["within_5pct_of_both_ideals"] = (
                c_eligible
                & h_eligible
                & result["C"].ge(0.95 * best_c)
                & result["H"].ge(0.95 * best_h)
            )

    result["pareto"] = False
    result["pareto_latest"] = False
    result["pareto_knee"] = False
    result["pareto_knee_score"] = np.nan
    result["pareto_knee_method"] = "abstain_no_common_domain"
    common = result["C_eligible"] & result["H_eligible"]
    if not common.any():
        return add_selection_contract(
            result,
            result["pareto_knee"],
            policy="ch_pareto_knee",
            selected_reason="normalized_chord_distance",
            abstain_reason="no_common_domain",
            score=result["pareto_knee_score"],
            model_supported=result["C_model_supported"] & result["H_model_supported"],
        )

    positions = result.index[common].to_numpy()
    c_values = result.loc[positions, "C"].to_numpy(dtype=float)
    h_values = result.loc[positions, "H"].to_numpy(dtype=float)
    front = np.ones(len(positions), dtype=bool)
    for index, (c_value, h_value) in enumerate(zip(c_values, h_values, strict=True)):
        front[index] = not np.any(
            (c_values >= c_value)
            & (h_values >= h_value)
            & ((c_values > c_value) | (h_values > h_value))
        )
    front_positions = positions[front]
    result.loc[front_positions, "pareto"] = True
    result.loc[front_positions[-1], "pareto_latest"] = True

    front_c = result.loc[front_positions, "C"].to_numpy(dtype=float)
    front_h = result.loc[front_positions, "H"].to_numpy(dtype=float)
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
    result.loc[front_positions, "pareto_knee_score"] = scores
    best = float(scores.max())
    knee = int(front_positions[np.flatnonzero(scores == best)[0]])
    result.loc[knee, "pareto_knee"] = True
    result["pareto_knee_method"] = method
    return add_selection_contract(
        result,
        result["pareto_knee"],
        policy="ch_pareto_knee",
        selected_reason=method,
        abstain_reason="no_common_domain",
        score=result["pareto_knee_score"],
        model_supported=result["C_model_supported"] & result["H_model_supported"],
    )
