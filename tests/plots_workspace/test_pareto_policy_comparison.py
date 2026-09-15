from __future__ import annotations

import numpy as np
import pandas as pd


def test_select_policy_times_compares_all_nine_methods() -> None:
    from plots.pareto_policy_comparison import (
        METHOD_LABELS,
        select_policy_times,
        selection_similarity,
    )

    times = pd.date_range("2026-01-01", periods=8, freq="min")
    candidates = pd.DataFrame(
        {
            "candidate_defrost_time": times,
            "stable_heating_start": times[0],
            "cycle_cop": [8, 10, 9, 8, 9, 6, 5, 4],
            "cycle_heating_rate_kw": [4, 5, 7, 10, 9, 8, 7, 6],
            "cycle_evaporator_capacity_kw": [3, 4, 5, 7, 10, 9, 8, 7],
        }
    )
    for objective in (
        "cycle_cop",
        "cycle_heating_rate_kw",
        "cycle_evaporator_capacity_kw",
    ):
        candidates[f"{objective}_eligible"] = True

    selected = select_policy_times(candidates)

    assert tuple(selected) == METHOD_LABELS
    assert selected["C"] == times[1]
    assert selected["H"] == times[3]
    assert selected["O"] == times[4]
    assert selected["CH → max O"] == times[4]
    assert selected["CO → max H"] == times[4]
    assert selected["HO → max C"] == times[4]
    assert all(pd.notna(value) for value in selected.values())
    assert np.isfinite(candidates["cycle_cop"]).all()

    results = pd.DataFrame(
        {
            "cycle_name": "cycle",
            "method": METHOD_LABELS,
            "selected_time": times[4],
        }
    )
    similarity = selection_similarity(results)
    assert np.diag(similarity).tolist() == [100.0] * len(METHOD_LABELS)
