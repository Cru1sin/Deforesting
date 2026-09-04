from __future__ import annotations

import pandas as pd
import pytest

from defrost_decision.baselines.electricity import heating_energy, transition_energy
from defrost_decision.baselines.heat import heating_heat, transition_heat_v2_5
from defrost_decision.candidate_times import build_candidate_boundaries


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

    assert values["candidate_defrost_time"].tolist() == [
        pd.Timestamp("2026-01-01 00:12:00"),
        pd.Timestamp("2026-01-01 00:13:00"),
        pd.Timestamp("2026-01-01 00:14:00"),
        pd.Timestamp("2026-01-01 00:14:30"),
    ]
    assert values["minutes_since_heating_start"].tolist() == [10.0, 11.0, 12.0, 12.5]


def test_v25_candidates_keep_domain_but_elapsed_and_heating_accounting_start_at_cycle_heating() -> (
    None
):
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "heating_start")
    energy = heating_energy(_frame(), boundaries)
    heat = heating_heat(_frame(), boundaries, "water")

    assert boundaries["candidate_defrost_time"].iloc[0] == pd.Timestamp("2026-01-01 00:12:00")
    assert boundaries["minutes_since_heating_start"].iloc[0] == 12.0
    assert energy["pre_defrost_electricity_kwh"].iloc[0] == pytest.approx(1.2)
    assert heat["pre_defrost_heat_kwh"].iloc[0] == pytest.approx(1.161)


def test_v25_historical_prefix_bridges_without_endpoint_extrapolation() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "heating_start")
    historical_start = pd.Timestamp(boundaries["stable_heating_start"].iloc[0])
    frame = _frame()
    prefix_nans = frame["timestamp"].isin(
        [pd.Timestamp("2026-01-01 00:00:00"), pd.Timestamp("2026-01-01 00:01:00")]
    )
    frame.loc[prefix_nans, "power_total"] = float("nan")
    frame.loc[prefix_nans, "water_flow"] = float("nan")

    energy = heating_energy(
        frame,
        boundaries.iloc[:1],
        historical_start=historical_start,
    )
    heat = heating_heat(
        frame,
        boundaries.iloc[:1],
        "water",
        historical_start=historical_start,
    )

    assert energy["pre_defrost_electricity_kwh"].iloc[0] == pytest.approx(6 * 719 / 3600)
    assert heat["pre_defrost_heat_kwh"].iloc[0] == pytest.approx(1.161 * 719 / 720)
    assert energy["heating_energy_supported"].iloc[0]
    assert heat["heating_heat_supported"].iloc[0]


def test_v1_heating_blocks_integrate_from_stable_and_use_unit_heat() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "stable_heating_start")
    energy = heating_energy(_frame(), boundaries)
    heat = heating_heat(_frame(), boundaries, "unit")

    assert energy["pre_defrost_electricity_kwh"].iloc[0] == pytest.approx(1.0)
    assert heat["pre_defrost_heat_kwh"].iloc[0] == pytest.approx(2.0)
    assert energy["heating_energy_supported"].all()
    assert heat["heating_heat_supported"].all()


def test_transition_energy_uses_strict_pre_action_window_and_frozen_fold() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "stable_heating_start")
    frame = _frame()
    first = boundaries["candidate_defrost_time"].iloc[0]
    frame.loc[frame["timestamp"].eq(first), "evaporating_pressure"] = 0.9

    result = transition_energy(
        frame,
        boundaries.iloc[:1],
        "exp_20260714",
        include_fixed_recovery=True,
    )

    expected_ed = 0.109160437898849 - 0.0524311159925975 * 0.3 - 0.2089549607749103 * 0.3**2
    assert result["evaporating_pressure_mpa"].iloc[0] == pytest.approx(0.3)
    assert result["preparation_energy_kwh"].iloc[0] == 0
    assert result["defrost_energy_kwh"].iloc[0] == pytest.approx(expected_ed)
    assert result["recovery_energy_kwh"].iloc[0] == pytest.approx(0.279901897467)
    assert result["defrost_event_electricity_kwh"].iloc[0] == pytest.approx(
        expected_ed + 0.279901897467
    )
    assert result["defrost_event_electricity_evaluable"].iloc[0]
    assert "ET_supported" not in result


def test_transition_energy_separates_evaluable_from_empirical_support() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "stable_heating_start")
    frame = _frame()
    frame["evaporating_pressure"] = 0.5

    result = transition_energy(
        frame,
        boundaries.iloc[:1],
        "exp_20260714",
        include_fixed_recovery=False,
    )

    assert result["defrost_event_electricity_evaluable"].iloc[0]
    assert not result["defrost_event_electricity_in_training_domain"].iloc[0]
    assert "ET_supported" not in result
    assert result["transition_energy_status"].iloc[0] == "above_support"


def test_transition_energy_does_not_fill_strict_window_from_tau_or_earlier_data() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "stable_heating_start")
    frame = _frame()
    first = boundaries["candidate_defrost_time"].iloc[0]
    strict = frame["timestamp"].ge(first - pd.Timedelta(seconds=60)) & frame["timestamp"].lt(first)
    frame.loc[strict, "evaporating_pressure"] = float("nan")
    frame.loc[frame["timestamp"].eq(first), "evaporating_pressure"] = 0.3

    result = transition_energy(
        frame,
        boundaries.iloc[:1],
        "exp_20260714",
        include_fixed_recovery=False,
        state_protocol="strict_causal",
    )

    assert pd.isna(result["evaporating_pressure_mpa"].iloc[0])
    assert not result["defrost_event_electricity_evaluable"].iloc[0]
    assert not result["defrost_event_electricity_in_training_domain"].iloc[0]
    assert "ET_supported" not in result


def test_strict_heating_coverage_is_anchored_to_declared_heating_accounting_start() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "heating_start")
    delayed = _frame().loc[lambda values: values["timestamp"].ge("2026-01-01 00:05:00")]

    energy = heating_energy(delayed, boundaries.iloc[:1])
    heat = heating_heat(delayed, boundaries.iloc[:1], "water")

    assert energy["heating_energy_coverage"].iloc[0] == 1
    assert heat["heating_heat_coverage"].iloc[0] == 1
    assert energy["heating_energy_supported"].iloc[0]
    assert heat["heating_heat_supported"].iloc[0]
    assert energy["strict_heating_energy_coverage"].iloc[0] == pytest.approx(7 / 12)
    assert heat["strict_heating_heat_coverage"].iloc[0] == pytest.approx(7 / 12)
    assert not energy["strict_heating_energy_supported"].iloc[0]
    assert not heat["strict_heating_heat_supported"].iloc[0]


def test_later_signal_sample_changes_only_offline_historical_reconstruction() -> None:
    start = pd.Timestamp("2026-01-01")
    boundaries = pd.DataFrame(
        {
            "candidate_defrost_time": [
                start + pd.Timedelta(seconds=30),
                start + pd.Timedelta(seconds=40),
            ],
            "heating_accounting_start": start,
            "heating_accounting_start_rule": "heating_start",
        }
    )
    frame = pd.DataFrame(
        {
            "timestamp": [start, start + pd.Timedelta(seconds=40)],
            "power_total": [1.0, 2.0],
            "heating_capacity": [3.0, 4.0],
        }
    )
    changed = frame.copy()
    changed.loc[1, ["power_total", "heating_capacity"]] = [200.0, 400.0]

    before_eh = heating_energy(frame, boundaries)
    after_eh = heating_energy(changed, boundaries)
    before_qh = heating_heat(frame, boundaries, "unit")
    after_qh = heating_heat(changed, boundaries, "unit")

    assert (
        before_eh["pre_defrost_electricity_kwh"].iloc[0]
        != after_eh["pre_defrost_electricity_kwh"].iloc[0]
    )
    assert before_qh["pre_defrost_heat_kwh"].iloc[0] != after_qh["pre_defrost_heat_kwh"].iloc[0]
    assert (
        before_eh["strict_pre_defrost_electricity_kwh"].iloc[0]
        == after_eh["strict_pre_defrost_electricity_kwh"].iloc[0]
    )
    assert (
        before_qh["strict_pre_defrost_heat_kwh"].iloc[0]
        == after_qh["strict_pre_defrost_heat_kwh"].iloc[0]
    )


def test_channel_leading_nans_are_only_extrapolated_by_historical_reconstruction() -> None:
    start = pd.Timestamp("2026-01-01")
    timestamps = pd.date_range(start, periods=61, freq="s")
    missing = timestamps < start + pd.Timedelta(seconds=30)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "power_total": pd.Series(6.0, index=range(61)).mask(missing),
            "water_flow": pd.Series(1.0, index=range(61)).mask(missing),
            "water_in_temperature": 40.0,
            "water_out_temperature": 45.0,
        }
    )
    boundaries = pd.DataFrame(
        {
            "candidate_defrost_time": [start + pd.Timedelta(seconds=60)],
            "heating_accounting_start": start,
            "heating_accounting_start_rule": "heating_start",
        }
    )

    energy = heating_energy(frame, boundaries)
    heat = heating_heat(frame, boundaries, "water")

    assert energy["pre_defrost_electricity_kwh"].iloc[0] == pytest.approx(6 * 60 / 3600)
    assert heat["pre_defrost_heat_kwh"].iloc[0] == pytest.approx(1.161 * 5 * 60 / 3600)
    assert energy["heating_energy_coverage"].iloc[0] == 1
    assert heat["heating_heat_coverage"].iloc[0] == 1
    assert energy["heating_energy_supported"].iloc[0]
    assert heat["heating_heat_supported"].iloc[0]
    assert energy["strict_pre_defrost_electricity_kwh"].iloc[0] == pytest.approx(6 * 30 / 3600)
    assert heat["strict_pre_defrost_heat_kwh"].iloc[0] == pytest.approx(1.161 * 5 * 30 / 3600)
    assert energy["strict_heating_energy_coverage"].iloc[0] == pytest.approx(0.5)
    assert heat["strict_heating_heat_coverage"].iloc[0] == pytest.approx(0.5)
    assert not energy["strict_heating_energy_supported"].iloc[0]
    assert not heat["strict_heating_heat_supported"].iloc[0]


def test_sparse_gap_is_bridged_canonically_but_not_in_strict_diagnostics() -> None:
    start = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame(
        {
            "timestamp": [start, start + pd.Timedelta(seconds=60)],
            "power_total": [6.0, 6.0],
            "heating_capacity": [12.0, 12.0],
        }
    )
    boundaries = pd.DataFrame(
        {
            "candidate_defrost_time": [start + pd.Timedelta(seconds=60)],
            "heating_accounting_start": start,
            "heating_accounting_start_rule": "heating_start",
        }
    )

    energy = heating_energy(frame, boundaries)
    heat = heating_heat(frame, boundaries, "unit")

    assert energy["pre_defrost_electricity_kwh"].iloc[0] == pytest.approx(0.1)
    assert heat["pre_defrost_heat_kwh"].iloc[0] == pytest.approx(0.2)
    assert energy["heating_energy_coverage"].iloc[0] == 1
    assert heat["heating_heat_coverage"].iloc[0] == 1
    assert energy["heating_energy_supported"].iloc[0]
    assert heat["heating_heat_supported"].iloc[0]
    assert energy["strict_pre_defrost_electricity_kwh"].iloc[0] == 0
    assert heat["strict_pre_defrost_heat_kwh"].iloc[0] == 0
    assert energy["strict_heating_energy_coverage"].iloc[0] == 0
    assert heat["strict_heating_heat_coverage"].iloc[0] == 0
    assert not energy["strict_heating_energy_supported"].iloc[0]
    assert not heat["strict_heating_heat_supported"].iloc[0]


def test_one_finite_pe_second_is_canonical_but_not_strictly_supported() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "stable_heating_start")
    frame = _frame()
    candidate = boundaries["candidate_defrost_time"].iloc[0]
    strict = frame["timestamp"].ge(candidate - pd.Timedelta(seconds=60)) & frame["timestamp"].lt(
        candidate
    )
    frame.loc[strict, "evaporating_pressure"] = float("nan")
    frame.loc[
        frame["timestamp"].eq(candidate - pd.Timedelta(seconds=30)), "evaporating_pressure"
    ] = 0.3

    result = transition_energy(
        frame,
        boundaries.iloc[:1],
        "exp_20260714",
        include_fixed_recovery=False,
    )

    assert result["defrost_event_electricity_evaluable"].iloc[0]
    assert "ET_supported" not in result
    assert not result["strict_ET_supported"].iloc[0]
    assert result["transition_energy_status"].iloc[0] == "supported"
    assert result["strict_transition_energy_status"].iloc[0] == "incomplete"


def test_one_finite_state_second_is_canonical_but_not_strictly_supported() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "heating_start")
    frame = _frame()
    candidate = boundaries["candidate_defrost_time"].iloc[0]
    strict = frame["timestamp"].ge(candidate - pd.Timedelta(seconds=60)) & frame["timestamp"].lt(
        candidate
    )
    features = [
        "water_in_temperature",
        "water_out_temperature",
        "coil_temperature",
        "evaporating_pressure",
    ]
    frame.loc[strict, features] = float("nan")
    frame.loc[frame["timestamp"].eq(candidate - pd.Timedelta(seconds=30)), features] = [
        45.0,
        50.0,
        -10.0,
        0.3,
    ]
    frame.loc[
        frame["timestamp"].between("2026-01-01 00:15:00", "2026-01-01 00:18:19"),
        "coil_temperature",
    ] = -10.0

    result = transition_heat_v2_5(frame, boundaries.iloc[:1], FakeLoader().record)

    assert result["defrost_event_net_heat_evaluable"].iloc[0]
    assert result["QT_physical_valid"].iloc[0]
    assert "QT_supported" not in result
    assert not result["strict_QT_supported"].iloc[0]
    assert result["transition_heat_status"].iloc[0] == "supported"
    assert result["strict_transition_heat_status"].iloc[0] == "incomplete"


def test_transition_heat_separates_evaluable_from_empirical_support() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "heating_start")
    result = transition_heat_v2_5(_frame(), boundaries.iloc[:1], FakeLoader().record)

    assert result["defrost_event_net_heat_evaluable"].iloc[0]
    assert not result["defrost_event_net_heat_in_training_domain"].iloc[0]
    assert "QT_supported" not in result
    assert result["transition_heat_status"].iloc[0] == "outside_empirical_support"


def test_v25_transition_heat_uses_strict_window_and_emits_signed_qd() -> None:
    boundaries = build_candidate_boundaries(FakeLoader(), "cycle_a", "heating_start")
    frame = _frame()
    first = boundaries["candidate_defrost_time"].iloc[0]
    frame.loc[frame["timestamp"].eq(first), "coil_temperature"] = 100.0

    result = transition_heat_v2_5(frame, boundaries.iloc[:1], FakeLoader().record)

    assert result["preparation_heat_kwh"].iloc[0] > 0
    assert result["defrost_heat_kwh"].iloc[0] <= 0
    assert result["defrost_event_net_heat_kwh"].iloc[0] == pytest.approx(
        result["preparation_heat_kwh"].iloc[0] + result["defrost_heat_kwh"].iloc[0]
    )
    assert result["recovery_heat_kwh"].iloc[0] == 0
    assert result["coil_temperature"].iloc[0] == pytest.approx(-10.0)
