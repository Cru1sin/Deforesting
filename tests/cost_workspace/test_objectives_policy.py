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
    from cost.objectives import add_single_objective_diagnostics, build_objectives

    values = _candidates()
    values.loc[0, "ET_in_support"] = False
    objectives = build_objectives(values)
    assert not any(column.endswith("_t_star") for column in objectives)
    result = add_single_objective_diagnostics(objectives)

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
    times = pd.date_range("2026-01-01", periods=8, freq="min")
    frame = pd.DataFrame(
        {
            "candidate_time": times,
            "C": [1.0, 1.4, 1.7, 2.0, 0.9, 0.8, 0.7, 0.6],
            "H": [2.0, 1.95, 1.6, 1.0, 0.9, 0.8, 0.7, 0.6],
            "O": np.arange(8, dtype=float),
            "pre_action_window_valid": True,
        }
    )
    for prefix in ("C", "H", "O"):
        frame[f"{prefix}_native_eligible"] = True
        frame[f"{prefix}_model_supported"] = True
        frame[f"{prefix}_measurement_eligible"] = True
        frame[f"{prefix}_physical_valid"] = True
    return frame


def test_policy_selects_chord_knee_ignores_o_and_reports_reference_window() -> None:
    from cost.policy import select_ch_pareto_knee

    objectives = _policy_objectives()
    first = select_ch_pareto_knee(objectives)
    changed = objectives.copy()
    changed["O"] = [np.nan, -1e9, 1e12, 5.0, -5.0, 7.0, 8.0, 9.0]
    changed["O_native_eligible"] = [False, False, True, True, False, True, False, True]
    second = select_ch_pareto_knee(changed)

    assert first.loc[first["pareto_knee"], "candidate_time"].item() == objectives.loc[
        1, "candidate_time"
    ]
    assert second["pareto_knee"].equals(first["pareto_knee"])
    assert not first.loc[1, "within_5pct_of_both_ideals"]
    assert first.loc[1, "pareto_knee"]
    assert not first["within_5pct_of_both_ideals"].any()
    assert first["pareto_knee_method"].eq("normalized_chord_distance").all()
    assert first["selected_time"].eq(objectives.loc[1, "candidate_time"]).all()
    assert first["selected"].equals(first["pareto_knee"])
    assert first["selection_policy"].eq("ch_pareto_knee").all()
    assert first["selection_status"].eq("selected").all()
    assert first["selection_reason"].eq("normalized_chord_distance").all()
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


def test_policy_extrapolation_still_requires_five_continuous_minutes() -> None:
    from cost.policy import select_ch_pareto_knee

    objectives = _policy_objectives()
    for prefix in ("C", "H", "O"):
        objectives[f"{prefix}_native_eligible"] = False
        objectives[f"{prefix}_model_supported"] = False
        objectives[f"{prefix}_measurement_eligible"] = False
        objectives.loc[1, f"{prefix}_measurement_eligible"] = True

    result = select_ch_pareto_knee(objectives, allow_extrapolation=True)

    assert not result.loc[1, "C_eligible"]
    assert not result.loc[1, "H_eligible"]
    assert not result.loc[1, "O_eligible"]
    assert not result["pareto_knee"].any()


def test_policy_requires_explicit_permission_for_extrapolation() -> None:
    from cost.policy import select_ch_pareto_knee

    objectives = _policy_objectives()
    for prefix in ("C", "H", "O"):
        objectives[f"{prefix}_native_eligible"] = False
        objectives[f"{prefix}_model_supported"] = False

    default = select_ch_pareto_knee(objectives)
    allowed = select_ch_pareto_knee(objectives, allow_extrapolation=True)

    assert not default["pareto_knee"].any()
    assert allowed["pareto_knee"].any()
