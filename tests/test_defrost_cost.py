from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frost_analysis.defrost_cost import (
    candidate_domain_end,
    count_true_runs,
    find_recovery_time,
    integrate_energy_kwh,
    optimize_renewal_cost,
    water_side_heating_kw,
)


def test_candidate_domain_distinguishes_observed_and_right_censored_cycles() -> None:
    observed = pd.Timestamp("2026-01-01 02:00")
    record_end = pd.Timestamp("2026-01-01 03:00")

    assert candidate_domain_end(observed, record_end) == (
        observed,
        "observed_defrost",
        False,
    )
    assert candidate_domain_end(None, record_end) == (
        record_end,
        "sensor_record_end",
        True,
    )


def test_candidate_domain_requires_a_real_end_time() -> None:
    with pytest.raises(ValueError, match="candidate end"):
        candidate_domain_end(None, None)


def test_count_true_runs_preserves_disconnected_near_optimal_regions() -> None:
    assert count_true_runs([True, True, False, True, False]) == 2


def test_water_side_heating_uses_raw_water_measurements() -> None:
    frame = pd.DataFrame(
        {
            "water_flow": [2.0],
            "water_in_temperature": [30.0],
            "water_out_temperature": [35.0],
        }
    )

    assert water_side_heating_kw(frame).iloc[0] == 11.61


def test_energy_integration_does_not_bridge_missing_intervals() -> None:
    time = pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:01", "2026-01-01 00:00:10"])
    power = pd.Series([3.6, 3.6, 3.6])

    energy, coverage = integrate_energy_kwh(time, power, maximum_gap_seconds=2)

    assert np.isclose(energy, 0.001)
    assert np.isclose(coverage, 1 / 10)


def test_energy_integration_accepts_irregular_raw_samples_within_gap_limit() -> None:
    time = pd.date_range("2026-01-01", periods=3, freq="s")
    power = pd.Series([3.6, np.nan, 3.6])

    energy, coverage = integrate_energy_kwh(time, power, maximum_gap_seconds=2)

    assert np.isclose(energy, 0.002)
    assert coverage == 1.0


def test_recovery_requires_30_continuous_seconds_above_threshold() -> None:
    time = pd.date_range("2026-01-01", periods=50, freq="s")
    heat = pd.Series([8.0] * 10 + [9.2] * 20 + [8.0] + [9.2] * 19)

    assert find_recovery_time(time, heat, reference_kw=10.0) is None
    heat[:] = 8.0
    heat.iloc[20:50] = 9.2
    assert find_recovery_time(time, heat, reference_kw=10.0) == time[20]


def test_renewal_cost_can_identify_an_interior_minimum() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.to_datetime(
                ["2026-01-01 01:00", "2026-01-01 02:00", "2026-01-01 03:00"]
            ),
            "heating_hours": [1.0, 2.0, 3.0],
            "heating_cost_kwh": [2.0, 3.0, 8.0],
        }
    )

    curve, optimum = optimize_renewal_cost(
        candidates,
        ticket_cost_kwh=3.0,
        ticket_duration_hours=1.0,
    )

    assert curve["renewal_cost_kw"].tolist() == [2.5, 2.0, 2.75]
    assert optimum["candidate_time"] == pd.Timestamp("2026-01-01 02:00")
    assert optimum["minimum_location"] == "interior"


def test_renewal_cost_rejects_a_truncated_candidate_domain() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.to_datetime(["2026-01-01 01:00", "2026-01-01 02:00"]),
            "heating_hours": [1.0, 2.0],
            "heating_cost_kwh": [2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="candidate domain does not reach"):
        optimize_renewal_cost(
            candidates,
            ticket_cost_kwh=3.0,
            ticket_duration_hours=1.0,
            required_end_time=pd.Timestamp("2026-01-01 03:00"),
        )
