from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _candidates(periods: int = 7) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01")
    return pd.DataFrame(
        {
            "candidate_time": pd.date_range(
                start + pd.Timedelta(minutes=1), periods=periods, freq="min"
            ),
            "integration_start": start,
            "heating_energy_kwh": 2.0,
            "heating_heat_kwh": 6.0,
            "heating_compressor_energy_kwh": 1.0,
            "transition_energy_kwh": 1.0,
            "transition_heat_kwh": 3.0,
            "transition_compressor_energy_kwh": 0.5,
            "transition_duration_minutes": 2.0,
            "ET_evaluable": True,
            "QT_evaluable": True,
            "EcompT_evaluable": True,
            "DT_evaluable": True,
            "ET_in_support": True,
            "QT_in_support": True,
            "EcompT_in_support": True,
            "DT_in_support": True,
            "pre_action_window_valid": True,
            "heating_energy_measurement_valid": True,
            "heating_heat_measurement_valid": True,
            "heating_compressor_measurement_valid": True,
        }
    )


def test_objectives_compute_exact_values_and_independent_support() -> None:
    from cost.objectives import build_objectives

    values = _candidates()
    values.loc[0, "ET_in_support"] = False
    result = build_objectives(values)

    hours = 1 / 60 + 2 / 60
    assert result.loc[0, "C"] == pytest.approx(3.0)
    assert result.loc[0, "H"] == pytest.approx(9.0 / hours)
    assert result.loc[0, "O"] == pytest.approx(7.5 / hours)
    assert not result.loc[0, "C_model_supported"]
    assert result.loc[0, "H_model_supported"]
    assert result.loc[0, "O_model_supported"]
    assert not result.loc[0, "C_native_eligible"]
    assert result["H_native_eligible"].all()
    assert result["O_native_eligible"].all()
    for prefix in ("C", "H", "O"):
        assert result[f"{prefix}_t_star"].notna().all()
    assert result["C_basin_5pct_width_minutes"].iloc[0] == 5


def _policy_objectives() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=4, freq="min")
    frame = pd.DataFrame(
        {
            "candidate_time": times,
            "C": [1.0, 1.4, 1.7, 2.0],
            "H": [2.0, 1.95, 1.6, 1.0],
            "O": [0.0, 1.0, 2.0, 3.0],
            "pre_action_window_valid": True,
        }
    )
    for prefix in ("C", "H", "O"):
        frame[f"{prefix}_native_eligible"] = True
        frame[f"{prefix}_model_supported"] = True
        frame[f"{prefix}_measurement_eligible"] = True
        frame[f"{prefix}_physical_valid"] = True
    return frame


def test_policy_selects_chord_knee_and_ignores_o_and_guardrail() -> None:
    from cost.policy import select_ch_pareto_knee

    objectives = _policy_objectives()
    first = select_ch_pareto_knee(objectives, guardrail=0.0)
    changed = objectives.copy()
    changed["O"] = [np.nan, -1e9, 1e12, 5.0]
    changed["O_native_eligible"] = [False, False, True, True]
    second = select_ch_pareto_knee(changed, guardrail=0.99)

    assert first.loc[first["pareto_knee"], "candidate_time"].item() == objectives.loc[
        1, "candidate_time"
    ]
    assert second["pareto_knee"].equals(first["pareto_knee"])
    assert not first.loc[1, "within_guardrail"]
    assert first.loc[1, "pareto_knee"]
    assert not first["within_guardrail"].any()
    assert first["pareto_knee_method"].eq("normalized_chord_distance").all()
    assert first["selected_time"].eq(objectives.loc[1, "candidate_time"]).all()
    assert not {"C_native", "C_model", "C_measurement", "C_physical"} & set(first)


def test_policy_abstains_without_h_and_applies_minimum_time_before_pareto() -> None:
    from cost.policy import select_ch_pareto_knee

    objectives = _policy_objectives()
    no_h = objectives.copy()
    no_h["H_native_eligible"] = False
    abstained = select_ch_pareto_knee(no_h)
    assert not abstained["pareto_knee"].any()
    assert abstained["pareto_knee_method"].eq("abstain_no_common_domain").all()
    assert abstained["selected_time"].isna().all()

    minimum = objectives.loc[2, "candidate_time"]
    restricted = select_ch_pareto_knee(objectives, minimum_time=minimum)
    assert not restricted.loc[:1, "pareto"].any()
    assert restricted.loc[restricted["pareto_knee"], "candidate_time"].item() >= minimum


def test_policy_rejects_stale_native_rows_with_nonfinite_ch_values() -> None:
    from cost.policy import select_ch_pareto_knee

    objectives = _policy_objectives()
    objectives.loc[1, ["C", "H"]] = np.nan

    result = select_ch_pareto_knee(objectives)

    assert not result.loc[1, "C_eligible"]
    assert not result.loc[1, "H_eligible"]
    assert not result.loc[1, "pareto"]
    assert not result.loc[1, "pareto_knee"]


def test_policy_extrapolation_requires_valid_pre_action_state() -> None:
    from cost.policy import select_ch_pareto_knee

    objectives = _policy_objectives()
    objectives.loc[1, "pre_action_window_valid"] = False
    for prefix in ("C", "H", "O"):
        objectives.loc[1, f"{prefix}_native_eligible"] = False
        objectives.loc[1, f"{prefix}_model_supported"] = False

    result = select_ch_pareto_knee(objectives, allow_extrapolation=True)

    assert not result.loc[1, "C_eligible"]
    assert not result.loc[1, "H_eligible"]
    assert not result.loc[1, "O_eligible"]
    assert not result.loc[1, "pareto"]
    assert not result.loc[1, "pareto_knee"]
