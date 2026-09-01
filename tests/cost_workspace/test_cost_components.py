from __future__ import annotations

import pandas as pd
import pytest

from cost.boundaries import build_candidate_boundaries
from cost.energy_models import heating_energy, transition_energy
from cost.heat_models import heating_heat, transition_heat_v2_5


class FakeLoader:
    def __init__(self) -> None:
        self.record = {
            "cycle_name": "cycle_a",
            "experiment_id": "exp_20260714",
            "boundaries": {
                "heating_start": "2026-01-01 00:00:00",
                "stable_heating_start": "2026-01-01 00:02:00",
                "defrost_preparation_start": "2026-01-01 00:14:30",
                "defrost_start": "2026-01-01 00:15:00",
                "defrost_end": "2026-01-01 00:20:00",
            },
        }

    def get_cycle_record(self, cycle_name: str) -> dict[str, object]:
        assert cycle_name == "cycle_a"
        return self.record


def _frame() -> pd.DataFrame:
    timestamp = pd.date_range("2026-01-01", periods=1201, freq="s")
    elapsed = (timestamp - timestamp[0]).total_seconds()
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "power_total": 6.0,
            "heating_capacity": 12.0,
            "water_flow": 1.0,
            "water_in_temperature": 40.0 + elapsed / 10000,
            "water_out_temperature": 45.0 + elapsed / 10000,
            "water_temperature_setpoint": 50.0,
            "coil_temperature": -10.0,
            "evaporating_pressure": 0.3,
        }
    )
    frame.loc[frame["timestamp"].ge("2026-01-01 00:15:00"), "coil_temperature"] = 20.0
    return frame


def test_candidates_start_stable_plus_ten_and_include_exact_preparation_end() -> None:
    values = build_candidate_boundaries(FakeLoader(), "cycle_a", "stable_heating_start")

    assert values["candidate_time"].tolist() == [
        pd.Timestamp("2026-01-01 00:12:00"),
        pd.Timestamp("2026-01-01 00:13:00"),
        pd.Timestamp("2026-01-01 00:14:00"),
        pd.Timestamp("2026-01-01 00:14:30"),
    ]
    assert values["candidate_elapsed_minutes"].tolist() == [10.0, 11.0, 12.0, 12.5]


def test_v25_candidates_keep_domain_but_elapsed_and_integration_start_at_cycle_heating() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "heating_start")
    energy = heating_energy(_frame(), boundaries)
    heat = heating_heat(_frame(), boundaries, "water")

    assert boundaries["candidate_time"].iloc[0] == pd.Timestamp("2026-01-01 00:12:00")
    assert boundaries["candidate_elapsed_minutes"].iloc[0] == 12.0
    assert energy["heating_energy_kwh"].iloc[0] == pytest.approx(1.2)
    assert heat["heating_heat_kwh"].iloc[0] == pytest.approx(1.161)


def test_v1_heating_blocks_integrate_from_stable_and_use_unit_heat() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "stable_heating_start")
    energy = heating_energy(_frame(), boundaries)
    heat = heating_heat(_frame(), boundaries, "unit")

    assert energy["heating_energy_kwh"].iloc[0] == pytest.approx(1.0)
    assert heat["heating_heat_kwh"].iloc[0] == pytest.approx(2.0)
    assert energy["heating_energy_supported"].all()
    assert heat["heating_heat_supported"].all()


def test_transition_energy_uses_strict_pre_action_window_and_frozen_fold() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "stable_heating_start")
    frame = _frame()
    first = boundaries["candidate_time"].iloc[0]
    frame.loc[frame["timestamp"].eq(first), "evaporating_pressure"] = 0.9

    result = transition_energy(
        frame,
        boundaries.iloc[:1],
        "exp_20260714",
        include_fixed_recovery=True,
    )

    expected_ed = 0.109160437898849 - 0.0524311159925975 * 0.3 - 0.2089549607749103 * 0.3**2
    assert result["evaporating_pressure_mpa"].iloc[0] == pytest.approx(0.3)
    assert result["defrost_electricity_kwh"].iloc[0] == pytest.approx(expected_ed)
    assert result["recovery_electricity_kwh"].iloc[0] == pytest.approx(0.279901897467)
    assert result["transition_energy_kwh"].iloc[0] == pytest.approx(expected_ed + 0.279901897467)
    assert result["ET_supported"].iloc[0]


def test_transition_energy_does_not_fill_strict_window_from_tau_or_earlier_data() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "stable_heating_start")
    frame = _frame()
    first = boundaries["candidate_time"].iloc[0]
    strict = frame["timestamp"].ge(first - pd.Timedelta(seconds=60)) & frame["timestamp"].lt(first)
    frame.loc[strict, "evaporating_pressure"] = float("nan")
    frame.loc[frame["timestamp"].eq(first), "evaporating_pressure"] = 0.3

    result = transition_energy(
        frame,
        boundaries.iloc[:1],
        "exp_20260714",
        include_fixed_recovery=False,
    )

    assert pd.isna(result["evaporating_pressure_mpa"].iloc[0])
    assert not result["ET_supported"].iloc[0]


def test_heating_coverage_is_anchored_to_declared_integration_start() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "heating_start")
    delayed = _frame().loc[lambda values: values["timestamp"].ge("2026-01-01 00:05:00")]

    energy = heating_energy(delayed, boundaries.iloc[:1])
    heat = heating_heat(delayed, boundaries.iloc[:1], "water")

    assert energy["heating_energy_coverage"].iloc[0] == pytest.approx(7 / 12)
    assert heat["heating_heat_coverage"].iloc[0] == pytest.approx(7 / 12)
    assert not energy["heating_energy_supported"].iloc[0]
    assert not heat["heating_heat_supported"].iloc[0]


def test_v25_transition_heat_uses_strict_window_and_emits_signed_qd() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "heating_start")
    frame = _frame()
    first = boundaries["candidate_time"].iloc[0]
    frame.loc[frame["timestamp"].eq(first), "coil_temperature"] = 100.0

    result = transition_heat_v2_5(frame, boundaries.iloc[:1], FakeLoader().record)

    assert result["preparation_heat_kwh"].iloc[0] > 0
    assert result["defrost_heat_kwh"].iloc[0] <= 0
    assert result["transition_heat_kwh"].iloc[0] == pytest.approx(
        result["preparation_heat_kwh"].iloc[0] + result["defrost_heat_kwh"].iloc[0]
    )
    assert result["recovery_heat_kwh"].iloc[0] == 0
    assert result["coil_temperature"].iloc[0] == pytest.approx(-10.0)
