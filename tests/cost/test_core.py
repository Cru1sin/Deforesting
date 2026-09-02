from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cost import energy_models
from frost_analysis.cost.core import (
    build_partial_pool_curves,
    candidate_domain_end,
    count_true_runs,
    find_recovery_time,
    integrate_energy_curve_kwh,
    integrate_energy_kwh,
    optimize_cycle_cop_cost,
    optimize_renewal_cost,
    water_side_heating_kw,
)


def test_legacy_energy_helpers_reexport_root_implementations() -> None:
    assert water_side_heating_kw is energy_models.water_side_heating_kw
    assert integrate_energy_kwh is energy_models.integrate_energy_kwh
    assert integrate_energy_curve_kwh is energy_models.integrate_energy_curve_kwh


def test_partial_pool_curves_reuse_existing_experiment_identity() -> None:
    curves = pd.DataFrame(
        {
            "cycle_name": ["c1", "c2"],
            "experiment_id": ["a", "b"],
            "heating_cost_kwh": [1.0, 1.5],
            "heating_hours": [1.0, 1.0],
        }
    )
    events = pd.DataFrame(
        {
            "experiment_id": ["a", "a", "b", "b"],
            "equivalent_cost_kwh": [0.2, 0.3, 0.4, 0.5],
            "duration_minutes": [5.0, 6.0, 7.0, 8.0],
        }
    )
    catalog = pd.DataFrame(
        {"cycle_name": ["c1", "c2"], "experiment_id": ["a", "b"]}
    )

    result = build_partial_pool_curves(curves, events, catalog)

    assert result["experiment_id"].tolist() == ["a", "b"]
    assert "experiment_id_x" not in result


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


def test_energy_curve_matches_pointwise_gap_aware_integration() -> None:
    time = pd.to_datetime(
        ["2026-01-01 00:00:00", "2026-01-01 00:00:01", "2026-01-01 00:00:10"]
    )
    power = pd.Series([3.6, 3.6, 3.6])
    candidates = pd.to_datetime(["2026-01-01 00:00:01", "2026-01-01 00:00:10"])

    curve = integrate_energy_curve_kwh(time, power, candidates, maximum_gap_seconds=2)

    assert np.allclose(curve["energy_kwh"], [0.001, 0.001])
    assert np.allclose(curve["coverage"], [1.0, 0.1])


def test_energy_curve_can_bridge_internal_gap_with_provenance() -> None:
    time = pd.date_range("2026-01-01", periods=12, freq="s")
    power = pd.Series([3.6, 3.6, *([np.nan] * 8), 7.2, 7.2])
    candidates = time[[1, 5, 10, 11]]

    curve = integrate_energy_curve_kwh(
        time,
        power,
        candidates,
        maximum_gap_seconds=2,
        bridge_internal_gaps=True,
    )

    assert curve["bridged_internal_gap"].tolist() == [False, True, False, False]
    assert curve.loc[1, "energy_kwh"] == pytest.approx(0.001 + 4.4 * 4 / 3600)
    assert curve.loc[2, "energy_kwh"] == pytest.approx(0.001 + 5.4 * 9 / 3600)
    assert curve.loc[2, "coverage"] == pytest.approx(1.0)


def test_energy_curve_can_linearly_extrapolate_endpoint_with_provenance() -> None:
    time = pd.date_range("2026-01-01", periods=5, freq="s")
    power = pd.Series([np.nan, np.nan, 3.6, 3.6, 3.6])
    candidates = time[[1, 2, 4]]

    curve = integrate_energy_curve_kwh(
        time,
        power,
        candidates,
        maximum_gap_seconds=2,
        extrapolate_endpoints=True,
    )

    assert curve["extrapolated_endpoint"].tolist() == [True, False, False]
    assert curve.loc[0, "energy_kwh"] == pytest.approx(0.001)
    assert curve.loc[0, "coverage"] == pytest.approx(1.0)


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


def test_cycle_cop_cost_uses_total_electricity_per_user_heat() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.to_datetime(
                ["2026-01-01 01:00", "2026-01-01 02:00", "2026-01-01 03:00"]
            ),
            "heating_electricity_kwh": [2.0, 4.0, 7.0],
            "user_heating_kwh": [4.0, 10.0, 15.0],
        }
    )

    curve, optimum = optimize_cycle_cop_cost(
        candidates,
        defrost_recovery_electricity_kwh=1.0,
    )

    assert np.allclose(curve["inverse_cop"], [0.75, 0.5, 8 / 15])
    assert np.allclose(curve["cycle_cop"], [4 / 3, 2.0, 15 / 8])
    assert optimum["candidate_time"] == pd.Timestamp("2026-01-01 02:00")
    assert optimum["minimum_location"] == "interior"


def test_cycle_cop_cost_broadcasts_rowwise_transient_ticket() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.to_datetime(
                ["2026-01-01 03:00", "2026-01-01 01:00", "2026-01-01 02:00"]
            ),
            "heating_electricity_kwh": [7.0, 2.0, 4.0],
            "user_heating_kwh": [15.0, 4.0, 10.0],
        }
    )
    tickets = pd.Series([1.0, 1.0, 3.0], index=candidates.index)

    curve, optimum = optimize_cycle_cop_cost(
        candidates,
        defrost_recovery_electricity_kwh=tickets,
    )

    assert curve["candidate_time"].is_monotonic_increasing
    assert curve["cycle_electricity_kwh"].tolist() == [3.0, 7.0, 8.0]
    assert optimum["candidate_time"] == pd.Timestamp("2026-01-01 03:00")


def test_cycle_cop_cost_includes_defrost_recovery_heat_in_denominator() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.to_datetime(
                ["2026-01-01 02:00", "2026-01-01 01:00"]
            ),
            "heating_electricity_kwh": [4.0, 2.0],
            "user_heating_kwh": [6.0, 0.0],
        }
    )

    curve, _ = optimize_cycle_cop_cost(
        candidates,
        defrost_recovery_electricity_kwh=pd.Series([2.0, 1.0]),
        defrost_recovery_heat_kwh=pd.Series([2.0, 4.0]),
    )

    assert curve["cycle_user_heating_kwh"].tolist() == [4.0, 8.0]
    assert curve["inverse_cop"].tolist() == pytest.approx([0.75, 0.75])


def test_cycle_cop_cost_accepts_scalar_defrost_recovery_heat() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.to_datetime(["2026-01-01 01:00"]),
            "heating_electricity_kwh": [2.0],
            "user_heating_kwh": [4.0],
        }
    )

    curve, _ = optimize_cycle_cop_cost(
        candidates,
        defrost_recovery_electricity_kwh=1.0,
        defrost_recovery_heat_kwh=1.0,
    )

    assert curve.loc[0, "cycle_user_heating_kwh"] == 5.0
    assert curve.loc[0, "inverse_cop"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("recovery_heat", "error"),
    [
        (-2.0, "positive user heating"),
        (-3.0, "positive user heating"),
        (np.nan, "finite energy"),
        (np.inf, "finite energy"),
    ],
)
def test_cycle_cop_cost_rejects_invalid_total_user_heat(
    recovery_heat: float, error: str
) -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.to_datetime(["2026-01-01 01:00"]),
            "heating_electricity_kwh": [2.0],
            "user_heating_kwh": [2.0],
        }
    )

    with pytest.raises(ValueError, match=error):
        optimize_cycle_cop_cost(
            candidates,
            defrost_recovery_electricity_kwh=1.0,
            defrost_recovery_heat_kwh=recovery_heat,
        )


def test_cycle_cop_cost_ignores_invalid_total_user_heat_when_ineligible() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01 01:00", periods=2, freq="h"),
            "heating_electricity_kwh": [2.0, 3.0],
            "user_heating_kwh": [4.0, 1.0],
            "optimization_eligible": [True, False],
        }
    )

    curve, _ = optimize_cycle_cop_cost(
        candidates,
        defrost_recovery_electricity_kwh=1.0,
        defrost_recovery_heat_kwh=pd.Series([0.0, np.nan]),
    )

    assert pd.isna(curve.loc[1, "cycle_user_heating_kwh"])
    assert curve.loc[0, "inverse_cop"] == pytest.approx(0.75)


def test_cycle_cop_cost_resets_duplicate_indices_after_stable_sort() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.to_datetime(
                ["2026-01-01 03:00", "2026-01-01 01:00", "2026-01-01 02:00"]
            ),
            "heating_electricity_kwh": [7.0, 2.0, 4.0],
            "user_heating_kwh": [15.0, 4.0, 10.0],
        },
        index=[0, 0, 1],
    )
    tickets = pd.Series([1.0, 1.0, 3.0], index=candidates.index)

    curve, optimum = optimize_cycle_cop_cost(
        candidates,
        defrost_recovery_electricity_kwh=tickets,
    )

    assert curve["candidate_time"].is_monotonic_increasing
    assert curve.index.tolist() == [0, 1, 2]
    assert curve["cycle_electricity_kwh"].tolist() == [3.0, 7.0, 8.0]
    assert np.allclose(curve["inverse_cop"], [0.75, 0.7, 8 / 15])
    assert optimum["candidate_time"] == pd.Timestamp("2026-01-01 03:00")
    assert optimum["minimum_location"] == "right_observed"


def test_cycle_cop_cost_keeps_unsupported_rows_out_of_argmin_but_finite() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01 01:00", periods=3, freq="h"),
            "heating_electricity_kwh": [2.0, 3.0, 3.5],
            "user_heating_kwh": [4.0, 8.0, 20.0],
            "pe_supported": [True, True, False],
            "integration_eligible": [True, True, True],
            "optimization_eligible": [True, True, False],
        }
    )

    curve, optimum = optimize_cycle_cop_cost(
        candidates,
        defrost_recovery_electricity_kwh=pd.Series([1.0, 1.0, 0.0]),
    )

    assert len(curve) == 3
    assert curve.loc[
        2, ["cycle_electricity_kwh", "inverse_cop", "cycle_cop"]
    ].tolist() == pytest.approx([3.5, 0.175, 1 / 0.175])
    assert optimum["candidate_time"] == pd.Timestamp("2026-01-01 02:00")
    assert optimum["minimum_location"] == "right_support_limited"
    assert optimum["left_support_removed"] is False


def test_cycle_cop_cost_keeps_finite_extrapolated_rows_in_curve() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01 01:00", periods=2, freq="h"),
            "heating_electricity_kwh": [2.0, 3.0],
            "user_heating_kwh": [4.0, 8.0],
            "pe_supported": [False, True],
            "integration_eligible": [True, True],
            "optimization_eligible": [True, True],
        }
    )

    curve, _ = optimize_cycle_cop_cost(
        candidates,
        defrost_recovery_electricity_kwh=pd.Series([1.0, 1.0]),
    )

    values = curve.loc[0, ["cycle_electricity_kwh", "inverse_cop", "cycle_cop"]]
    assert values.tolist() == pytest.approx([3.0, 0.75, 4 / 3])


def test_cycle_cop_cost_distinguishes_support_and_integration_right_limits() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01 01:00", periods=2, freq="h"),
            "heating_electricity_kwh": [2.0, 3.0],
            "user_heating_kwh": [4.0, 8.0],
            "pe_supported": [True, False],
            "integration_eligible": [True, True],
            "optimization_eligible": [True, False],
        }
    )

    _, support_limited = optimize_cycle_cop_cost(
        candidates,
        defrost_recovery_electricity_kwh=1.0,
    )
    assert support_limited["minimum_location"] == "right_support_limited"

    candidates["pe_supported"] = True
    candidates["integration_eligible"] = [True, False]
    _, integration_limited = optimize_cycle_cop_cost(
        candidates,
        defrost_recovery_electricity_kwh=1.0,
    )
    assert integration_limited["minimum_location"] == "right_integration_limited"


def test_cycle_cop_cost_prioritizes_observed_right_for_single_eligible_candidate() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01 01:00", periods=3, freq="h"),
            "heating_electricity_kwh": [2.0, 3.0, 3.5],
            "user_heating_kwh": [4.0, 8.0, 10.0],
            "pe_supported": [False, False, True],
            "integration_eligible": [True, True, True],
            "optimization_eligible": [False, False, True],
        }
    )

    curve, optimum = optimize_cycle_cop_cost(
        candidates,
        defrost_recovery_electricity_kwh=1.0,
    )

    assert curve.loc[:1, "inverse_cop"].tolist() == pytest.approx([0.75, 0.5])
    assert optimum["minimum_location"] == "right_observed"
    assert optimum["left_support_removed"] is True
    assert optimum["left_integration_removed"] is False


def test_cycle_cop_cost_tracks_left_integration_removal() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01 01:00", periods=3, freq="h"),
            "heating_electricity_kwh": [2.0, 3.0, 9.0],
            "user_heating_kwh": [4.0, 8.0, 10.0],
            "pe_supported": [True, True, True],
            "integration_eligible": [False, True, True],
            "optimization_eligible": [False, True, True],
        }
    )

    _, optimum = optimize_cycle_cop_cost(
        candidates,
        defrost_recovery_electricity_kwh=1.0,
    )

    assert optimum["minimum_location"] == "left_boundary"
    assert optimum["left_support_removed"] is False
    assert optimum["left_integration_removed"] is True


def test_cycle_cop_cost_distinguishes_observed_right_and_zero_support() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01 01:00", periods=3, freq="h"),
            "heating_electricity_kwh": [2.0, 3.0, 3.5],
            "user_heating_kwh": [4.0, 8.0, 10.0],
            "optimization_eligible": [True, True, True],
        }
    )

    _, optimum = optimize_cycle_cop_cost(
        candidates,
        defrost_recovery_electricity_kwh=1.0,
    )
    assert optimum["minimum_location"] == "right_observed"

    candidates["optimization_eligible"] = False
    with pytest.raises(ValueError, match="no_supported_candidates"):
        optimize_cycle_cop_cost(
            candidates,
            defrost_recovery_electricity_kwh=1.0,
        )


def test_cycle_cop_cost_rejects_nonpositive_user_heat() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.to_datetime(["2026-01-01 01:00"]),
            "heating_electricity_kwh": [2.0],
            "user_heating_kwh": [0.0],
        }
    )

    with pytest.raises(ValueError, match="positive user heating"):
        optimize_cycle_cop_cost(
            candidates,
            defrost_recovery_electricity_kwh=1.0,
        )


@pytest.mark.parametrize(
    ("electricity", "user_heat"),
    [(np.nan, 2.0), (np.inf, 2.0), (2.0, np.nan), (2.0, np.inf)],
)
def test_cycle_cop_cost_rejects_nonfinite_energy(electricity: float, user_heat: float) -> None:
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.to_datetime(["2026-01-01 01:00"]),
            "heating_electricity_kwh": [electricity],
            "user_heating_kwh": [user_heat],
        }
    )

    with pytest.raises(ValueError, match="finite energy"):
        optimize_cycle_cop_cost(
            candidates,
            defrost_recovery_electricity_kwh=1.0,
        )
