from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


def _analysis_module():
    from frost_analysis.exploration import optimal_defrost

    return optimal_defrost


def _rb_frame(
    seconds: int,
    *,
    t1_seconds: float = 0,
    t2_seconds: float = 360,
    t3: float | list[float] = 0,
    t4: float = 0,
    twout: float = 30,
    t3o_raw: float = 50,
) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01")
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=seconds, freq="s"),
            "coil_temperature": t3,
            "ambient_temperature": t4,
            "water_out_temperature": twout,
            "p1__T3o'2_20": t3o_raw,
            "p1__DefTim1'2_20": t1_seconds,
            "p1__DefTim2'2_20": t2_seconds,
        }
    )


@pytest.mark.parametrize(
    ("t4", "twout", "t3o", "expected"),
    [
        (-2, 35, 4, (40, 1)),
        (-2, 25, 4, (35, 1)),
        (-2, 24, 4, (30, 1)),
        (-5, 35, 4, (40, -1)),
        (-5, 25, 4, (38, -1)),
        (-5, 24, 4, (33, -1)),
        (-8, 35, 4, (80, -1)),
        (-8, 25, 4, (60, -1)),
        (-8, 24, 4, (40, -1)),
        (-10, 35, 4, (90, -15)),
        (-10, 25, 4, (70, -15)),
        (-10, 24, 4, (50, -15)),
        (-10.1, 35, 4, (150, -15.1)),
        (-10.1, 25, 4, (120, -15.1)),
        (-10.1, 24, 4, (90, -15.1)),
    ],
)
def test_rb_case_boundaries_and_limits(
    t4: float, twout: float, t3o: float, expected: tuple[float, float]
) -> None:
    analysis = _analysis_module()

    assert analysis._rb_limits(t4, twout, t3o) == expected


@pytest.mark.parametrize(
    ("delta_seconds", "triggered"),
    [(49, False), (50, True), (600, True), (601, False)],
)
def test_rb_condition1_history_window_boundaries(
    delta_seconds: int, triggered: bool
) -> None:
    analysis = _analysis_module()
    t3 = [np.nan] * (delta_seconds + 1)
    t3[0] = 0
    t3[-1] = -1
    frame = _rb_frame(
        len(t3), t1_seconds=36 * 60, t3=t3, t3o_raw=-100
    )

    result = analysis._rb_first_trigger(
        frame,
        frame.iloc[0]["timestamp"],
        frame.iloc[-1]["timestamp"] + pd.Timedelta(seconds=1),
    )

    assert result["rb_status"] == ("triggered" if triggered else "right_censored")


def test_rb_condition1_uses_50_to_600_second_history() -> None:
    analysis = _analysis_module()
    t3 = [0.0] * 50 + [-1.0] * 2
    frame = _rb_frame(52, t1_seconds=36 * 60, t3=t3, t3o_raw=-100)

    trigger = analysis._rb_first_trigger(
        frame, frame.iloc[0]["timestamp"], frame.iloc[-1]["timestamp"] + pd.Timedelta(seconds=1)
    )

    assert trigger["trigger_type"] == "Condition1"
    assert trigger["t_RB"] == frame.iloc[50]["timestamp"]


def test_rb_condition2_temperature_requires_20_seconds_but_time_gate_does_not() -> None:
    analysis = _analysis_module()
    frame = _rb_frame(25, t1_seconds=34 * 60, t3=1, t3o_raw=50)
    frame.loc[frame.index >= 1, "coil_temperature"] = 0
    frame.loc[frame.index >= 19, "p1__DefTim1'2_20"] = 35 * 60

    trigger = analysis._rb_first_trigger(
        frame, frame.iloc[0]["timestamp"], frame.iloc[-1]["timestamp"] + pd.Timedelta(seconds=1)
    )

    assert trigger["trigger_type"] == "Case1"
    assert trigger["t_RB"] == frame.iloc[19]["timestamp"]


def test_rb_case7_case8_and_same_second_priority() -> None:
    analysis = _analysis_module()
    frame = _rb_frame(20, t1_seconds=29 * 60, t2_seconds=0, t3=-20, t4=-5)
    frame.loc[19, "p1__DefTim1'2_20"] = 150 * 60

    trigger = analysis._rb_first_trigger(
        frame, frame.iloc[0]["timestamp"], frame.iloc[-1]["timestamp"] + pd.Timedelta(seconds=1)
    )

    assert trigger["trigger_type"] == "Case7"
    assert trigger["t_RB"] == frame.iloc[19]["timestamp"]
    assert trigger["T1_min"] == pytest.approx(150)
    assert trigger["T3o_C"] == pytest.approx(5)

    case8 = _rb_frame(1, t1_seconds=150 * 60, t2_seconds=0)
    forced = analysis._rb_first_trigger(
        case8,
        case8.iloc[0]["timestamp"],
        case8.iloc[0]["timestamp"] + pd.Timedelta(seconds=1),
    )
    assert forced["trigger_type"] == "Case8"


def test_rb_first_trigger_is_unchanged_by_future_data() -> None:
    analysis = _analysis_module()
    frame = _rb_frame(80, t1_seconds=36 * 60, t3o_raw=-100)
    frame.loc[50:, "coil_temperature"] = -1
    start = frame.iloc[0]["timestamp"]
    end = frame.iloc[-1]["timestamp"] + pd.Timedelta(seconds=1)
    expected = analysis._rb_first_trigger(frame, start, end)
    changed = frame.copy()
    changed.loc[51:, ["coil_temperature", "ambient_temperature", "water_out_temperature"]] = 999

    actual = analysis._rb_first_trigger(changed, start, end)

    assert (actual["t_RB"], actual["trigger_type"]) == (
        expected["t_RB"],
        expected["trigger_type"],
    )


def test_rb_returns_one_right_censored_row_when_no_rule_triggers() -> None:
    analysis = _analysis_module()
    frame = _rb_frame(30)
    start = frame.iloc[0]["timestamp"]
    end = frame.iloc[-1]["timestamp"] + pd.Timedelta(seconds=1)

    trigger = analysis._rb_first_trigger(frame, start, end)

    assert trigger["rb_status"] == "right_censored"
    assert pd.isna(trigger["t_RB"])
    assert trigger["t_observation_end"] == end


def test_rb_case8_triggers_without_temperature_inputs() -> None:
    analysis = _analysis_module()
    frame = _rb_frame(
        1,
        t1_seconds=150 * 60,
        t2_seconds=0,
        t3=np.nan,
        t4=np.nan,
        twout=np.nan,
        t3o_raw=np.nan,
    )

    trigger = analysis._rb_first_trigger(
        frame,
        frame.iloc[0]["timestamp"],
        frame.iloc[0]["timestamp"] + pd.Timedelta(seconds=1),
    )

    assert trigger["trigger_type"] == "Case8"
    assert pd.isna(trigger["case"])


def test_rb_cost_uses_nearest_eligible_candidate_within_half_minute() -> None:
    analysis = _analysis_module()
    start = pd.Timestamp("2026-01-01")
    results = pd.DataFrame(
        {
            "cycle_name": ["matched", "support_gap", "too_far"],
            "t_RB": [
                start + pd.Timedelta(minutes=10, seconds=29),
                start + pd.Timedelta(minutes=20),
                start + pd.Timedelta(minutes=30, seconds=31),
            ],
            "rb_status": ["triggered"] * 3,
        }
    )
    curves = pd.DataFrame(
        {
            "cycle_name": ["matched", "matched", "support_gap", "too_far"],
            "candidate_time": [
                start + pd.Timedelta(minutes=10),
                start + pd.Timedelta(minutes=11),
                start + pd.Timedelta(minutes=20),
                start + pd.Timedelta(minutes=30),
            ],
            "optimization_eligible": [True, True, False, True],
            "inverse_cop": [0.50, 0.52, 0.60, 0.55],
            "relative_regret": [0.02, 0.06, 0.20, 0.10],
        }
    )

    costs = analysis._rb_candidate_costs(results, curves).set_index("cycle_name")

    assert costs.loc["matched", "rb_candidate_time"] == start + pd.Timedelta(minutes=10)
    assert costs.loc["matched", "rb_inverse_cop"] == pytest.approx(0.50)
    assert costs.loc["matched", "rb_relative_regret"] == pytest.approx(0.02)
    assert pd.isna(costs.loc["support_gap", "rb_candidate_time"])
    assert pd.isna(costs.loc["too_far", "rb_candidate_time"])


def test_rb_minus_optimal_plot_uses_valid_comparable_cycles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = _analysis_module()
    captured: dict[str, object] = {}
    original_save = analysis._save_figure

    def save_and_capture(fig: object, base: Path) -> None:
        captured["figure"] = fig
        original_save(fig, base)

    monkeypatch.setattr(analysis, "_save_figure", save_and_capture)
    results = pd.DataFrame(
        {
            "valid": [True, True, True, False, True],
            "trigger_type": ["Case2", "Condition1", "Case2", "Case1", "Case1"],
            "rb_minus_optimal_minutes": [5.0, -2.0, 7.0, 99.0, np.nan],
        }
    )
    output = tmp_path / "figure_1g_rb_minus_optimal_by_trigger_type"

    analysis.plot_rb_minus_optimal_by_trigger_type(results, output)

    axis = captured["figure"].axes[0]
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["Condition1", "Case2"]
    assert sum(len(collection.get_offsets()) for collection in axis.collections) == 3
    assert any(np.allclose(line.get_ydata(), [0, 0]) for line in axis.lines)
    assert {text.get_text() for text in axis.texts} == {"n=1", "n=2"}
    for suffix in (".png", ".svg", ".pdf"):
        assert output.with_suffix(suffix).is_file()


def test_save_figure_strips_svg_trailing_whitespace(tmp_path) -> None:
    analysis = _analysis_module()
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])

    analysis._save_figure(figure, tmp_path / "figure")

    lines = (tmp_path / "figure.svg").read_text().splitlines()
    assert all(line == line.rstrip() for line in lines)


def test_near_optimal_segments_preserve_disconnected_time_bands() -> None:
    analysis = _analysis_module()
    curve = pd.DataFrame(
        {
            "minutes": [10, 11, 12, 13, 14],
            "relative_regret": [0.01, 0.02, 0.08, 0.03, 0.09],
        }
    )

    assert analysis._near_optimal_segments(curve) == [(10.0, 11.0), (13.0, 13.0)]


def test_candidate_costs_integrate_electricity_and_positive_user_heat() -> None:
    analysis = _analysis_module()
    stable = pd.Timestamp("2026-01-01 00:00:00")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(stable, periods=602, freq="s"),
            "power_total": 3.6,
            "q_heating_kw": [7.2] * 601 + [-7.2],
            "q_unit_kw": [6.0] * 601 + [-6.0],
        }
    )

    candidates = analysis._candidate_costs(
        frame,
        stable_start=stable,
        candidate_end=stable + pd.Timedelta(minutes=10),
        q_start_kw=7.2,
        next_stable_start=stable + pd.Timedelta(minutes=20),
        q_end_kw=7.2,
        lambda_q=0.5,
    )

    assert np.isclose(candidates.iloc[0]["heating_electricity_kwh"], 0.6)
    assert np.isclose(candidates.iloc[0]["user_heating_kwh"], 1.2)
    assert np.isclose(candidates.iloc[0]["water_heating_kwh"], 1.2)
    assert np.isclose(candidates.iloc[0]["unit_heating_kwh"], 1.0)


def test_candidate_costs_marks_internal_gap_bridge_across_cost_paths() -> None:
    analysis = _analysis_module()
    stable = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(stable, periods=661, freq="s"),
            "power_total": 3.6,
            "q_heating_kw": 7.2,
            "q_unit_kw": 6.0,
            "evaporating_pressure": 0.3,
        }
    )
    frame.loc[541:600, ["power_total", "q_heating_kw", "q_unit_kw"]] = np.nan

    candidates = analysis._candidate_costs(
        frame,
        stable_start=stable,
        candidate_end=stable + pd.Timedelta(minutes=11),
        q_start_kw=7.2,
        next_stable_start=stable + pd.Timedelta(minutes=20),
        q_end_kw=7.2,
        lambda_q=0.5,
    )

    assert candidates["candidate_in_interpolated_gap"].tolist() == [True, False]
    assert candidates.loc[0, "heating_electricity_kwh"] == pytest.approx(0.6)
    assert candidates.loc[0, "user_heating_kwh"] == pytest.approx(1.2)
    assert candidates.loc[0, "unit_heating_kwh"] == pytest.approx(1.0)


def test_candidate_pressure_window_is_left_closed_and_inside_interpolated() -> None:
    analysis = _analysis_module()
    end = pd.Timestamp("2026-01-01 00:01:00")
    frame = pd.DataFrame(
        {
            "timestamp": [end - pd.Timedelta(seconds=59), end - pd.Timedelta(seconds=1), end],
            "evaporating_pressure": [0.2, 0.2, 99.0],
        }
    )

    values = analysis._candidate_pressure_features(frame, end)

    assert values["evaporating_pressure_mpa"] == pytest.approx(0.2)
    assert values["pe_raw_valid_seconds"] == 2
    assert values["pe_interpolated_valid_seconds"] == 60
    assert values["pe_interpolated_coverage"] == 1.0
    assert values["pe_internal_gap_interpolated"] is False
    assert values["pe_extrapolated_valid_seconds"] == 1
    assert values["pe_endpoint_extrapolated"] is True


def test_candidate_pressure_uses_full_cycle_endpoints_without_extrapolation() -> None:
    analysis = _analysis_module()
    start = pd.Timestamp("2026-01-01")
    end = start + pd.Timedelta(minutes=2)
    frame = pd.DataFrame(
        {
            "timestamp": [start + pd.Timedelta(seconds=59), end + pd.Timedelta(seconds=1)],
            "evaporating_pressure": [0.2, 0.4],
        }
    )

    values = analysis._candidate_pressure_features(frame, end)

    assert values["pe_raw_valid_seconds"] == 0
    assert values["pe_interpolated_valid_seconds"] == 60
    assert values["evaporating_pressure_mpa"] == pytest.approx(0.2983870968)
    assert values["pe_internal_gap_interpolated"] is True


def test_candidate_pressure_extrapolates_after_last_finite_observation() -> None:
    analysis = _analysis_module()
    end = pd.Timestamp("2026-01-01 00:02:00")
    frame = pd.DataFrame(
        {
            "timestamp": [
                end - pd.Timedelta(minutes=3),
                end - pd.Timedelta(minutes=2),
                end - pd.Timedelta(seconds=30),
            ],
            "evaporating_pressure": [0.2, 0.3, np.nan],
        }
    )

    values = analysis._candidate_pressure_features(frame, end)

    assert values["pe_raw_valid_seconds"] == 0
    assert values["pe_interpolated_valid_seconds"] == 60
    assert values["pe_extrapolated_valid_seconds"] == 60
    assert values["pe_endpoint_extrapolated"] is True
    assert values["evaporating_pressure_mpa"] == pytest.approx(0.4491667)


def test_candidate_pressure_marker_ignores_short_internal_gaps() -> None:
    analysis = _analysis_module()
    end = pd.Timestamp("2026-01-01 00:02:00")
    frame = pd.DataFrame(
        {
            "timestamp": [
                end - pd.Timedelta(minutes=2),
                end - pd.Timedelta(minutes=1),
                end - pd.Timedelta(seconds=57),
                end - pd.Timedelta(seconds=1),
                end,
            ],
            "evaporating_pressure": [0.2, 0.2, 0.3, 0.3, 0.3],
        }
    )

    values = analysis._candidate_pressure_features(frame, end)

    assert values["pe_interpolated_valid_seconds"] > values["pe_raw_valid_seconds"]
    assert values["pe_internal_gap_interpolated"] is False


def test_candidate_pressure_marker_only_flags_candidate_inside_long_gap() -> None:
    analysis = _analysis_module()
    start = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame(
        {
            "timestamp": [
                start,
                start + pd.Timedelta(minutes=1),
                start + pd.Timedelta(minutes=2),
                start + pd.Timedelta(minutes=2, seconds=1),
                start + pd.Timedelta(minutes=2, seconds=2),
                start + pd.Timedelta(minutes=2, seconds=30),
                start + pd.Timedelta(minutes=2, seconds=31),
            ],
            "evaporating_pressure": [0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4],
        }
    )

    inside = analysis._candidate_pressure_features(
        frame, start + pd.Timedelta(minutes=1, seconds=30)
    )
    recovered = analysis._candidate_pressure_features(
        frame, start + pd.Timedelta(minutes=2, seconds=31)
    )

    assert inside["pe_internal_gap_interpolated"] is True
    assert np.isfinite(inside["evaporating_pressure_mpa"])
    assert 0.2 <= inside["evaporating_pressure_mpa"] <= 0.4
    assert recovered["pe_interpolated_valid_seconds"] > 0
    assert recovered["pe_internal_gap_interpolated"] is False


def test_candidate_costs_append_exact_observed_preparation_boundary() -> None:
    analysis = _analysis_module()
    stable = pd.Timestamp("2026-01-01")
    preparation = stable + pd.Timedelta(minutes=10, seconds=30)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(stable, preparation, freq="s"),
            "power_total": 3.6,
            "q_heating_kw": 7.2,
            "q_unit_kw": 3.6,
            "evaporating_pressure": 0.3,
        }
    )

    candidates = analysis._candidate_costs(
        frame,
        stable_start=stable,
        candidate_end=preparation,
        q_start_kw=7.2,
        next_stable_start=preparation,
        q_end_kw=7.2,
        lambda_q=0.5,
    )

    assert candidates["candidate_time"].tolist() == [
        stable + pd.Timedelta(minutes=10),
        preparation,
    ]
    assert candidates["user_heating_kwh"].tolist() == pytest.approx([1.2, 1.26])
    assert candidates["water_heating_kwh"].tolist() == pytest.approx([1.2, 1.26])
    assert candidates["unit_heating_kwh"].tolist() == pytest.approx([0.6, 0.63])
    assert candidates["unit_heating_coverage"].tolist() == pytest.approx([1.0, 1.0])


def test_heat_basis_comparison_keeps_domain_ticket_and_eligibility_but_can_move_argmin() -> None:
    analysis = _analysis_module()
    end = pd.Timestamp("2026-01-01 00:12")
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01 00:10", periods=3, freq="min"),
            "heating_electricity_kwh": [1.0, 2.0, 3.0],
            "user_heating_kwh": [2.0, 4.0, 5.0],
            "water_heating_kwh": [2.0, 4.0, 5.0],
            "unit_heating_kwh": [2.0, 3.0, 6.0],
            "unit_heating_coverage": [1.0, 1.0, 1.0],
            "dynamic_ticket_electricity_kwh": [1.0, 1.0, 1.0],
            "pe_supported": [True, True, True],
            "integration_eligible": [True, True, True],
            "optimization_eligible": [True, True, True],
        }
    )
    original = candidates.copy(deep=True)

    curve, water, unit = analysis._compare_heat_bases(candidates, end)

    pd.testing.assert_series_equal(candidates["candidate_time"], original["candidate_time"])
    pd.testing.assert_series_equal(
        candidates["dynamic_ticket_electricity_kwh"],
        original["dynamic_ticket_electricity_kwh"],
    )
    pd.testing.assert_series_equal(
        candidates["optimization_eligible"], original["optimization_eligible"]
    )
    assert water["candidate_time"] == candidates.loc[1, "candidate_time"]
    assert unit["candidate_time"] == candidates.loc[2, "candidate_time"]
    assert curve["inverse_cop_water"].tolist() == pytest.approx([1.0, 0.75, 0.8])
    assert curve["inverse_cop_unit"].tolist() == pytest.approx([1.0, 1.0, 2 / 3])
    assert curve["relative_regret_water"].iloc[1] == pytest.approx(0.0)
    assert curve["relative_regret_unit"].iloc[2] == pytest.approx(0.0)


def test_fixed_ticket_comparison_keeps_observed_boundary_as_ineligible() -> None:
    analysis = _analysis_module()
    end = pd.Timestamp("2026-01-01 00:12")
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01 00:10", periods=3, freq="min"),
            "heating_electricity_kwh": [1.0, 1.1, 1.2],
            "user_heating_kwh": [2.0, 2.5, 3.0],
            "integration_coverage": [1.0, 1.0, 0.9],
        }
    )

    curve, optimum = analysis._fixed_ticket_optimum(candidates, 0.3, end)

    assert curve["candidate_time"].max() == end
    assert curve["pe_supported"].tolist() == [True, True, True]
    assert curve["integration_eligible"].tolist() == [True, True, False]
    assert curve["optimization_eligible"].tolist() == [True, True, False]
    assert optimum["minimum_location"] == "right_integration_limited"


def test_pe_fold_reader_requires_one_unique_tuple_per_experiment(tmp_path: Path) -> None:
    analysis = _analysis_module()
    source = tmp_path / "folds.csv"
    rows = pd.DataFrame(
        {
            "experiment_id": ["a", "a"],
            "fold_intercept_kwh": [0.1, 0.1],
            "fold_linear_kwh_per_mpa": [-0.2, -0.2],
            "fold_quadratic_kwh_per_mpa2": [-0.4, -0.4],
            "fold_train_pe_min_mpa": [0.2, 0.2],
            "fold_train_pe_max_mpa": [0.4, 0.4],
        }
    )
    rows.to_csv(source, index=False)
    assert len(analysis._read_pe_folds(source)) == 1

    rows.loc[1, "fold_quadratic_kwh_per_mpa2"] = -0.3
    rows.to_csv(source, index=False)
    with pytest.raises(ValueError, match="unique LOEO Pe fold"):
        analysis._read_pe_folds(source)


def test_pe_fold_application_extrapolates_outside_training_support() -> None:
    analysis = _analysis_module()
    candidates = pd.DataFrame(
        {
            "evaporating_pressure_mpa": [np.nan, 0.1, 0.3, 0.5],
            "integration_coverage": [1.0, 1.0, 1.0, 1.0],
        }
    )
    fold = pd.Series(
        {
            "fold_intercept_kwh": 0.1,
            "fold_linear_kwh_per_mpa": -0.2,
            "fold_quadratic_kwh_per_mpa2": -0.4,
            "fold_train_pe_min_mpa": 0.2,
            "fold_train_pe_max_mpa": 0.4,
        }
    )

    result = analysis._apply_pe_fold(candidates, fold)

    assert set(analysis.PE_FOLD_COLUMNS).issubset(result.columns)
    assert result["support_status"].tolist() == ["missing", "below", "supported", "above"]
    assert result["pe_supported"].tolist() == [False, False, True, False]
    assert result["integration_eligible"].tolist() == [True, True, True, True]
    assert result["optimization_eligible"].tolist() == [False, True, True, True]
    assert result["pe_extrapolation_distance_mpa_signed"].tolist()[1:] == pytest.approx(
        [-0.1, 0.0, 0.1]
    )
    modeled = [
        "predicted_preparation_defrost_electricity_kwh",
        "dynamic_ticket_electricity_kwh",
    ]
    assert result.loc[result["pe_supported"], modeled].iloc[0].tolist() == pytest.approx(
        [0.004, 0.004 + analysis.FIXED_RECOVERY_ELECTRICITY_KWH]
    )
    below = result.loc[result["support_status"].eq("below"), modeled].iloc[0]
    assert below.tolist() == pytest.approx(
        [0.076, 0.076 + analysis.FIXED_RECOVERY_ELECTRICITY_KWH]
    )
    above = result.loc[result["support_status"].eq("above"), modeled].iloc[0]
    assert above.tolist() == pytest.approx(
        [-0.1, -0.1 + analysis.FIXED_RECOVERY_ELECTRICITY_KWH]
    )


def test_candidate_audit_splits_pe_support_and_optimization_eligibility() -> None:
    analysis = _analysis_module()
    candidates = pd.DataFrame(
        {
            "pe_supported": [True, True, True, False],
            "integration_eligible": [True, False, True, True],
            "optimization_eligible": [True, False, True, False],
        }
    )

    audit = analysis._candidate_eligibility_audit(candidates)

    assert audit == {
        "pe_supported_candidate_count": 3,
        "pe_supported_candidate_fraction": 0.75,
        "integration_eligible_candidate_count": 3,
        "integration_eligible_candidate_fraction": 0.75,
        "optimization_eligible_candidate_count": 2,
        "optimization_eligible_candidate_fraction": 0.5,
        "supported_candidate_count": 3,
        "support_coverage_fraction": 0.75,
    }


def test_pe_support_summary_excludes_missing_pressure_from_extrapolation() -> None:
    analysis = _analysis_module()

    summary = analysis._pe_support_summary(
        pd.DataFrame(
            {"support_status": ["supported", "above", "below", "missing", "missing"]}
        )
    )

    assert summary == {
        "supported_count": 1,
        "extrapolated_count": 2,
        "missing_count": 2,
    }


def test_no_eligible_failure_reason_distinguishes_pe_support_from_integration() -> None:
    analysis = _analysis_module()

    assert (
        analysis._no_eligible_failure_reason(
            pd.DataFrame(
                {
                    "pe_supported": [False, False],
                    "optimization_eligible": [False, False],
                }
            )
        )
        == "no_pe_supported_candidates"
    )
    assert (
        analysis._no_eligible_failure_reason(
            pd.DataFrame(
                {
                    "pe_supported": [True, False],
                    "optimization_eligible": [False, False],
                }
            )
        )
        == "no_optimization_eligible_candidates"
    )


def test_report_statistics_separate_support_and_flag_sensitive_maximum() -> None:
    analysis = _analysis_module()
    points = pd.DataFrame(
        {
            "cycle_name": ["cycle_1", "cycle_2", "frost_cycle_000064"],
            "dynamic_vs_fixed_ticket_shift_minutes": [1.0, -3.0, 71.0],
            "pe_supported_candidate_fraction": [1.0, 1.0, 0.5],
            "optimization_eligible_candidate_fraction": [1.0, 0.95, 0.01],
        }
    )
    curves = pd.DataFrame(
        {
            "pe_supported": [True, True, False],
            "integration_eligible": [True, True, False],
            "optimization_eligible": [True, False, False],
        }
    )

    statistics = analysis._dynamic_report_statistics(points, curves)

    assert statistics == {
        "candidate_count": 3,
        "pe_supported_candidate_count": 2,
        "integration_eligible_candidate_count": 2,
        "optimization_eligible_candidate_count": 1,
        "maximum_shift_cycle": "frost_cycle_000064",
        "maximum_shift_minutes": 71.0,
        "maximum_shift_optimization_fraction": 0.01,
        "fully_pe_supported_cycle_count": 2,
        "fully_pe_supported_shift_median": 2.0,
        "fully_pe_supported_shift_p90": 2.8,
        "fully_pe_supported_shift_maximum": 3.0,
    }


def test_preparation_candidate_boundary_never_falls_back_to_defrost_start() -> None:
    analysis = _analysis_module()
    row = pd.Series(
        {
            "status": "valid",
            "stable_heating_start": "2026-01-01 00:00",
            "defrost_preparation_start": None,
            "defrost_start": "2026-01-01 01:00",
            "defrost_end": "2026-01-01 01:05",
        }
    )

    assert (
        analysis._preparation_candidate_boundary_reason(row)
        == "missing_defrost_preparation_start"
    )
    row["defrost_preparation_start"] = "2026-01-01 01:01"
    assert (
        analysis._preparation_candidate_boundary_reason(row)
        == "invalid_defrost_preparation_boundary_order"
    )
    row["defrost_preparation_start"] = "2026-01-01 00:59"
    assert analysis._preparation_candidate_boundary_reason(row) == ""


def test_unsupported_hole_splits_near_optimal_segments() -> None:
    analysis = _analysis_module()
    curve = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01", periods=4, freq="min"),
            "relative_regret": [0.01, 0.02, np.nan, 0.03],
            "optimization_eligible": [True, True, False, True],
        }
    )

    segments = analysis._near_optimal_segment_rows("cycle", curve)

    assert segments["segment_index"].tolist() == [1, 2]
    assert segments["segment_start"].tolist() == [
        pd.Timestamp("2026-01-01 00:00"),
        pd.Timestamp("2026-01-01 00:03"),
    ]
    assert segments["contains_t_star"].tolist() == [True, False]


def test_dual_near_optimal_segments_keep_thresholds_nested_and_split_holes() -> None:
    analysis = _analysis_module()
    curve = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01", periods=6, freq="min"),
            "relative_regret": [0.0, 0.008, 0.03, np.nan, 0.009, 0.06],
            "optimization_eligible": [True, True, True, False, True, True],
        }
    )

    segments = pd.concat(
        [
            analysis._near_optimal_segment_rows("cycle", curve, fraction=fraction)
            for fraction in (0.01, 0.05)
        ],
        ignore_index=True,
    )

    assert set(segments["relative_regret_threshold"]) == {0.01, 0.05}
    one_percent = segments.loc[segments["relative_regret_threshold"].eq(0.01)]
    five_percent = segments.loc[segments["relative_regret_threshold"].eq(0.05)]
    assert one_percent["segment_start"].tolist() == [
        pd.Timestamp("2026-01-01 00:00"),
        pd.Timestamp("2026-01-01 00:04"),
    ]
    assert all(
        any(
            outer.segment_start <= inner.segment_start
            and inner.segment_end <= outer.segment_end
            for outer in five_percent.itertuples(index=False)
        )
        for inner in one_percent.itertuples(index=False)
    )


def test_diagnostic_dynamic_distribution_exports_three_formats_and_marks_failed_off_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = _analysis_module()
    captured: dict[str, object] = {}
    original_save = analysis._save_figure

    def save_and_capture(fig: object, base: Path) -> None:
        captured["figure"] = fig
        original_save(fig, base)

    monkeypatch.setattr(analysis, "_save_figure", save_and_capture)
    stable = pd.Timestamp("2026-01-01")
    fast_stable = stable + pd.Timedelta(days=2)
    results = pd.DataFrame(
        {
            "cycle_name": [
                "ordinary_slow",
                "ordinary_fast",
                "observed_right",
                "left_boundary",
                "support_limited",
                "integration_limited",
                "left_removed",
                "failed",
            ],
            "complete_observed_defrost": [True] * 8,
            "valid": [True, True, True, True, True, True, True, False],
            "t_heating_stable": [
                stable,
                fast_stable,
                stable,
                stable,
                stable,
                stable,
                stable,
                stable,
            ],
            "t_actual_preparation": [
                stable + pd.Timedelta(minutes=40),
                fast_stable + pd.Timedelta(minutes=30),
                stable + pd.Timedelta(minutes=32),
                stable + pd.Timedelta(minutes=34),
                stable + pd.Timedelta(minutes=50),
                stable + pd.Timedelta(minutes=45),
                stable + pd.Timedelta(minutes=35),
                pd.NaT,
            ],
            "t_star": [
                stable + pd.Timedelta(minutes=20),
                fast_stable + pd.Timedelta(minutes=10),
                stable + pd.Timedelta(minutes=11),
                stable + pd.Timedelta(minutes=12),
                stable + pd.Timedelta(minutes=5),
                stable + pd.Timedelta(minutes=6),
                stable + pd.Timedelta(minutes=7),
                pd.NaT,
            ],
            "minutes_from_stable": [20.0, 10.0, 11.0, 12.0, 5.0, 6.0, 7.0, np.nan],
            "actual_minutes_from_stable": [
                40.0,
                30.0,
                32.0,
                34.0,
                50.0,
                45.0,
                35.0,
                np.nan,
            ],
            "minimum_location": [
                "interior",
                "interior",
                "right_observed",
                "left_boundary",
                "right_support_limited",
                "right_integration_limited",
                "interior",
                np.nan,
            ],
            "left_support_removed": [False, False, False, False, False, False, True, False],
            "left_integration_removed": [False] * 8,
            "failure_reason": ["", "", "", "", "", "", "", "missing_preparation"],
        }
    )
    five_percent_segments = pd.DataFrame(
        {
            "cycle_name": [
                "ordinary_slow",
                "ordinary_fast",
                "observed_right",
                "left_boundary",
                "support_limited",
                "integration_limited",
                "left_removed",
            ],
            "segment_index": [1] * 7,
            "segment_start": [
                stable + pd.Timedelta(minutes=15),
                fast_stable + pd.Timedelta(minutes=8),
                stable + pd.Timedelta(minutes=10),
                stable + pd.Timedelta(minutes=11),
                stable + pd.Timedelta(minutes=4),
                stable + pd.Timedelta(minutes=5),
                stable + pd.Timedelta(minutes=6),
            ],
            "segment_end": [
                stable + pd.Timedelta(minutes=22),
                fast_stable + pd.Timedelta(minutes=12),
                stable + pd.Timedelta(minutes=11),
                stable + pd.Timedelta(minutes=12),
                stable + pd.Timedelta(minutes=5),
                stable + pd.Timedelta(minutes=6),
                stable + pd.Timedelta(minutes=7),
            ],
            "contains_t_star": [True] * 7,
        }
    )
    one_percent_segments = five_percent_segments.copy()
    one_percent_segments["segment_start"] += pd.Timedelta(minutes=1)
    one_percent_segments["segment_end"] -= pd.Timedelta(minutes=1)
    five_percent_segments["relative_regret_threshold"] = 0.05
    one_percent_segments["relative_regret_threshold"] = 0.01
    segments = pd.concat(
        [five_percent_segments, one_percent_segments], ignore_index=True
    )

    output = tmp_path / "figure_1e_dynamic_optimal_time_distribution"
    analysis.plot_diagnostic_dynamic_optimal_time_distribution(results, segments, output)

    for suffix in (".png", ".svg", ".pdf"):
        assert output.with_suffix(suffix).is_file()
    axis = captured["figure"].axes[0]
    assert [label.get_text() for label in axis.get_yticklabels()] == [
        "ordinary_fast",
        "observed_right",
        "left_boundary",
        "ordinary_slow",
        "support_limited",
        "integration_limited",
        "left_removed",
        "failed",
    ]
    assert sorted(patch.get_width() for patch in axis.patches) == [
        30.0,
        32.0,
        34.0,
        35.0,
        40.0,
        45.0,
        50.0,
    ]
    failed = next(line for line in axis.lines if line.get_label() == "_failed_status")
    assert failed.get_xdata().tolist() == [-0.025]
    assert failed.get_clip_on() is False
    assert failed.get_transform() != axis.transData
    legend = axis.get_legend()
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert len(legend.legend_handles) == 10
    assert {
        "Observed length",
        "Observed preparation",
        "Failed / no estimate",
        "1% near-optimal segment",
        "5% near-optimal segment",
        "Pe-support-limited search",
        "Integration-limited search",
    }.issubset(legend_labels)
    legend_handles = dict(zip(legend_labels, legend.legend_handles, strict=True))
    assert legend_handles["Interior"].get_marker() == "D"
    assert legend_handles["Left"].get_marker() == "<"
    assert legend_handles["Observed right"].get_marker() == ">"
    assert legend_handles["Left"].get_markerfacecolor() == "#1F5F99"
    assert legend_handles["Observed right"].get_markerfacecolor() == "#1F5F99"
    assert legend_handles["Pe-support-limited search"].get_markerfacecolor() == "white"
    assert legend_handles["Pe-support-limited search"].get_markeredgecolor() == "#1F5F99"
    assert legend_handles["Integration-limited search"].get_markerfacecolor() == "white"
    assert legend_handles["Integration-limited search"].get_markeredgecolor() == "#C66A00"
    near_optimal_lines = [
        line for line in axis.lines if line.get_color() in {"#77A9D4", "#7B5AA6"}
    ]
    assert [line.get_color() for line in near_optimal_lines[:2]] == [
        "#77A9D4",
        "#7B5AA6",
    ]
    assert near_optimal_lines[0].get_linewidth() > near_optimal_lines[1].get_linewidth()

    point_styles = {
        label.get_text(): collection
        for label, collection in zip(
            axis.get_yticklabels(), axis.collections, strict=False
        )
    }
    assert point_styles["observed_right"].get_facecolors()[0].tolist() == pytest.approx(
        analysis.mpl.colors.to_rgba("#1F5F99")
    )
    assert point_styles["left_boundary"].get_facecolors()[0].tolist() == pytest.approx(
        analysis.mpl.colors.to_rgba("#1F5F99")
    )
    assert point_styles["support_limited"].get_facecolors()[0].tolist() == pytest.approx(
        analysis.mpl.colors.to_rgba("white")
    )
    assert point_styles["integration_limited"].get_edgecolors()[0].tolist() == pytest.approx(
        analysis.mpl.colors.to_rgba("#C66A00")
    )


def test_publication_dynamic_distribution_prioritizes_sorted_valid_optima(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = _analysis_module()
    captured: dict[str, object] = {}
    original_save = analysis._save_figure

    def save_and_capture(fig: object, base: Path) -> None:
        captured["figure"] = fig
        original_save(fig, base)

    monkeypatch.setattr(analysis, "_save_figure", save_and_capture)
    stable = pd.Timestamp("2026-01-01")
    results = pd.DataFrame(
        {
            "cycle_name": ["late", "failed", "partial", "early", "constrained"],
            "complete_observed_defrost": [True, True, False, True, True],
            "valid": [True, False, True, True, True],
            "t_heating_stable": [stable] * 5,
            "minutes_from_stable": [30.0, np.nan, 5.0, 10.0, 20.0],
            "t_star_unit": [
                stable + pd.Timedelta(minutes=32),
                pd.NaT,
                stable + pd.Timedelta(minutes=7),
                stable + pd.Timedelta(minutes=12),
                stable + pd.Timedelta(minutes=18),
            ],
            "heat_basis_comparable": [True, False, True, True, True],
            "actual_minutes_from_stable": [40.0, np.nan, 12.0, 25.0, 35.0],
            "minimum_location": ["interior", np.nan, "interior", "interior", "interior"],
            "left_support_removed": [False, False, False, False, True],
            "left_integration_removed": [False] * 5,
            "rb_status": [
                "triggered",
                "right_censored",
                "triggered",
                "triggered",
                "right_censored",
            ],
            "rb_minutes_from_stable": [33.0, np.nan, 7.0, 14.0, np.nan],
        }
    )
    segments = pd.DataFrame(
        {
            "cycle_name": ["late", "early", "constrained"] * 2,
            "relative_regret_threshold": [0.01] * 3 + [0.05] * 3,
            "segment_start": [
                stable + pd.Timedelta(minutes=28),
                stable + pd.Timedelta(minutes=9),
                stable + pd.Timedelta(minutes=18),
                stable + pd.Timedelta(minutes=20),
                stable + pd.Timedelta(minutes=5),
                stable + pd.Timedelta(minutes=12),
            ],
            "segment_end": [
                stable + pd.Timedelta(minutes=32),
                stable + pd.Timedelta(minutes=11),
                stable + pd.Timedelta(minutes=22),
                stable + pd.Timedelta(minutes=35),
                stable + pd.Timedelta(minutes=15),
                stable + pd.Timedelta(minutes=25),
            ],
        }
    )

    output = tmp_path / "figure_1e_dynamic_optimal_time_distribution"
    analysis.plot_dynamic_optimal_time_distribution(results, segments, output)

    axis = captured["figure"].axes[0]
    assert sorted(patch.get_width() for patch in axis.patches) == [25.0, 35.0, 40.0]
    assert axis.get_xlabel() == "Heating duration before defrost initiation (min)"
    assert axis.get_ylabel() == "Cycle (sorted by water-side optimum)"
    assert all("early" not in tick.get_text() for tick in axis.get_yticklabels())
    assert [collection.get_offsets()[0, 0] for collection in axis.collections] == [
        10.0, 12.0, 20.0, 18.0, 30.0, 32.0
    ]
    colors = [line.get_color() for line in axis.lines]
    assert "#7B5AA6" not in colors
    assert "#77A9D4" not in colors
    assert colors.count("#2E7D5B") == 4
    rb_lines = [line for line in axis.lines if line.get_label() == "_rb_trigger"]
    assert [line.get_xdata().tolist() for line in rb_lines] == [[14.0], [33.0]]
    censored = next(line for line in axis.lines if line.get_label() == "_rb_censored")
    assert censored.get_marker() == ">"
    assert censored.get_markerfacecolor() == "white"
    labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert labels == [
        "Observed heating period",
        "Water-side optimum",
        "Unit-reported optimum",
        "RB defrost time",
    ]

    analysis.plot_dynamic_optimal_time_distribution(
        results, segments, tmp_path / "unit_rb", comparison="unit_rb"
    )
    unit_axis = captured["figure"].axes[0]
    assert unit_axis.get_ylabel() == "Cycle (sorted by unit-reported optimum)"
    assert sorted(patch.get_width() for patch in unit_axis.patches) == [25.0, 35.0, 40.0]
    assert [collection.get_offsets()[0, 0] for collection in unit_axis.collections] == [
        12.0, 18.0, 32.0
    ]
    assert [text.get_text() for text in unit_axis.get_legend().get_texts()] == [
        "Observed heating period",
        "Unit-reported optimum",
        "RB defrost time",
    ]

    analysis.plot_dynamic_optimal_time_distribution(
        results, segments, tmp_path / "water_rb", comparison="water_rb"
    )
    water_axis = captured["figure"].axes[0]
    assert water_axis.get_ylabel() == "Cycle (sorted by water-side optimum)"
    assert sorted(patch.get_width() for patch in water_axis.patches) == [25.0, 35.0, 40.0]
    assert [collection.get_offsets()[0, 0] for collection in water_axis.collections] == [
        10.0, 20.0, 30.0
    ]
    assert [text.get_text() for text in water_axis.get_legend().get_texts()] == [
        "Observed heating period",
        "Water-side optimum",
        "RB defrost time",
    ]


def test_optimum_classification_summary_uses_complete_observed_cohort() -> None:
    analysis = _analysis_module()
    results = pd.DataFrame(
        {
            "complete_observed_defrost": [True, True, True, True, False],
            "valid": [True, True, True, False, True],
            "minimum_location": [
                "interior",
                "interior",
                "right_observed",
                np.nan,
                "interior",
            ],
            "left_support_removed": [False, True, False, False, False],
            "left_integration_removed": [False] * 5,
        }
    )

    summary = analysis._optimum_classification_summary(results)

    assert summary["category"].tolist() == [
        "Interior optimum",
        "Boundary-limited optimum",
        "No valid estimate",
    ]
    assert summary["count"].tolist() == [1, 2, 1]
    assert summary["fraction"].tolist() == pytest.approx([0.25, 0.5, 0.25])


def test_publication_cycle_ticks_use_cycle_numbers_not_zero_based_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = _analysis_module()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        analysis,
        "_save_figure",
        lambda fig, _output: captured.setdefault("figure", fig),
    )
    stable = pd.Timestamp("2026-01-01")
    count = 11
    results = pd.DataFrame(
        {
            "cycle_name": [f"cycle_{index}" for index in range(count)],
            "complete_observed_defrost": True,
            "valid": True,
            "t_heating_stable": stable,
            "minutes_from_stable": np.arange(10, 10 + count),
            "actual_minutes_from_stable": 30.0,
            "minimum_location": "interior",
            "left_support_removed": False,
            "left_integration_removed": False,
        }
    )
    segments = pd.DataFrame(
        {
            "cycle_name": results["cycle_name"],
            "relative_regret_threshold": 0.01,
            "segment_start": stable + pd.to_timedelta(results["minutes_from_stable"], unit="min"),
            "segment_end": stable + pd.to_timedelta(results["minutes_from_stable"], unit="min"),
        }
    )

    analysis.plot_dynamic_optimal_time_distribution(results, segments, tmp_path / "ticks")

    assert [tick.get_text() for tick in captured["figure"].axes[0].get_yticklabels()] == [
        "1",
        "10",
    ]


def test_observed_cycle_filter_ignores_partial_name_but_requires_all_boundaries() -> None:
    analysis = _analysis_module()
    row = pd.Series(
        {
            "cycle_name": "partial_cycle_70",
            "status": "valid",
            "stable_heating_start": "2026-01-01 00:00",
            "defrost_start": "2026-01-01 01:00",
            "defrost_end": "2026-01-01 01:05",
        }
    )

    assert analysis._observed_cycle_boundary_reason(row) == ""
    assert analysis._ticket_boundary_reason(row, following="cycle_71") == ""
    assert analysis._ticket_boundary_reason(row, following=None) == "missing_next_cycle_recovery"
    row["defrost_start"] = None
    assert analysis._observed_cycle_boundary_reason(row) == "missing_defrost_start"
    row["defrost_start"] = "2026-01-01 01:00"
    row["status"] = "partial"
    assert analysis._observed_cycle_boundary_reason(row) == "catalog_partial"


def test_publication_main_exports_four_standalone_figures(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = _analysis_module()
    captured: dict[str, object] = {}
    original_save = analysis._save_figure

    def save_and_capture(fig: object, base: Path) -> None:
        captured[base.name] = fig
        original_save(fig, base)

    monkeypatch.setattr(analysis, "_save_figure", save_and_capture)
    stable = pd.Timestamp("2026-01-01 00:00:00")
    actual = stable + pd.Timedelta(minutes=60)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(stable, actual, periods=61),
            "q_heating_kw": np.linspace(8.0, 5.0, 61),
        }
    )
    result = pd.Series(
        {
            "t_heating_stable": stable,
            "t_actual_defrost": actual,
            "t_star": stable + pd.Timedelta(minutes=35),
            "rb_status": "triggered",
            "rb_minutes_from_stable": 44.0,
        }
    )
    curve = pd.DataFrame(
        {
            "candidate_time": [
                stable + pd.Timedelta(minutes=10),
                stable + pd.Timedelta(minutes=35),
                actual,
            ],
            "inverse_cop": [0.60, 0.50, 0.53],
        }
    )
    results = pd.DataFrame(
        {
            "minutes_from_stable": [35.0, 60.0, 55.0],
            "actual_minutes_from_stable": [60.0, 60.0, 60.0],
            "minutes_earlier_than_actual": [25.0, 0.0, 5.0],
            "minimum_location": [
                "interior",
                "right_boundary",
                "right_integration_limited",
            ],
        }
    )

    analysis._plot_main(
        frame,
        result,
        curve,
        results,
        pd.DataFrame({"valid": [True, True]}),
        q_start_kw=8.0,
        next_stable_start=actual + pd.Timedelta(minutes=10),
        q_end_kw=7.0,
        output=tmp_path / "figure_1_empirical_optimal_defrost",
    )

    names = [
        "figure_1a_representative_heating_capacity",
        "figure_1b_representative_cycle_inverse_cop",
        "figure_1c_optimum_vs_observed_defrost",
        "figure_1d_defrost_advance_distribution",
    ]
    for name in names:
        for suffix in (".svg", ".pdf", ".png"):
            assert (tmp_path / f"{name}{suffix}").is_file()
        svg_lines = (tmp_path / f"{name}.svg").read_text().splitlines()
        assert all(line == line.rstrip() for line in svg_lines)
    cost_axis = captured["figure_1b_representative_cycle_inverse_cop"].axes[0]
    rb = next(line for line in cost_axis.lines if line.get_label() == "RB baseline")
    assert list(rb.get_xdata()) == [44.0, 44.0]
    assert rb.get_color() == "#2E7D5B"
    assert rb.get_linestyle() == "--"


def test_cycle_atlas_builds_reference_from_passed_anchors() -> None:
    analysis = _analysis_module()
    stable = pd.Timestamp("2026-01-01")
    actual = stable + pd.Timedelta(minutes=12)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(stable, actual, freq="min"),
            "q_heating_kw": np.linspace(8.0, 6.0, 13),
        }
    )
    result = pd.Series(
        {
            "cycle_name": "cycle",
            "t_heating_stable": stable,
            "t_actual_preparation": actual,
            "candidate_end": actual,
            "t_star": stable + pd.Timedelta(minutes=11),
            "is_censored": False,
            "cohort_tier": "complete_observed_cycle",
        }
    )
    curve = pd.DataFrame(
        {
            "candidate_time": pd.date_range(
                stable + pd.Timedelta(minutes=10), periods=3, freq="min"
            ),
            "inverse_cop": [0.5, 0.4, 0.45],
            "relative_regret": [0.25, 0.0, 0.125],
            "optimization_eligible": [True, True, True],
        }
    )

    analysis._plot_cycle(
        frame,
        result,
        curve,
        q_start_kw=8.0,
        next_stable_start=actual + pd.Timedelta(minutes=5),
        q_end_kw=7.0,
    )

    assert "q_reference_kw" not in frame


def test_cycle_plot_distinguishes_pe_and_integration_ineligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _analysis_module()
    stable = pd.Timestamp("2026-01-01")
    actual = stable + pd.Timedelta(minutes=12)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(stable, actual, freq="min"),
            "q_heating_kw": np.linspace(8.0, 6.0, 13),
        }
    )
    result = pd.Series(
        {
            "cycle_name": "cycle",
            "t_heating_stable": stable,
            "t_actual_preparation": actual,
            "candidate_end": actual,
            "t_star": stable + pd.Timedelta(minutes=11),
            "is_censored": False,
            "cohort_tier": "complete_observed_cycle",
        }
    )
    curve = pd.DataFrame(
        {
            "candidate_time": pd.date_range(
                stable + pd.Timedelta(minutes=9), periods=4, freq="min"
            ),
            "inverse_cop": [0.55, 0.50, 0.40, 0.45],
            "relative_regret": [np.nan, np.nan, 0.0, 0.125],
            "pe_supported": [False, True, True, True],
            "integration_eligible": [True, False, True, True],
            "optimization_eligible": [False, False, True, True],
        }
    )
    original_close = analysis.plt.close
    monkeypatch.setattr(analysis.plt, "close", lambda _figure: None)

    analysis._plot_cycle(
        frame,
        result,
        curve,
        q_start_kw=8.0,
        next_stable_start=actual + pd.Timedelta(minutes=5),
        q_end_kw=7.0,
    )

    figure = analysis.plt.gcf()
    labels = {collection.get_label() for collection in figure.axes[1].collections}
    assert labels >= {
        "Outside Pe support",
        "Insufficient integration coverage",
    }
    original_close(figure)


def test_candidate_costs_use_only_pre_candidate_state_window() -> None:
    analysis = _analysis_module()
    stable = pd.Timestamp("2026-01-01")
    candidate = stable + pd.Timedelta(minutes=10)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(stable, periods=602, freq="s"),
            "power_total": 3.6,
            "q_heating_kw": 7.2,
            "q_unit_kw": 6.0,
            "evaporating_pressure": 0.3,
            "water_in_temperature": 40.0,
            "water_out_temperature": 45.0,
            "coil_temperature": -8.0,
            "water_temperature_setpoint": 50.0,
        }
    )
    frame.loc[frame["timestamp"].ge(candidate), [
        "evaporating_pressure",
        "water_in_temperature",
        "water_out_temperature",
        "coil_temperature",
        "water_temperature_setpoint",
    ]] = 999.0

    row = analysis._candidate_costs(
        frame,
        stable_start=stable,
        candidate_end=candidate,
        q_start_kw=7.2,
        next_stable_start=stable + pd.Timedelta(minutes=20),
        q_end_kw=7.2,
        lambda_q=0.5,
    ).iloc[0]

    assert row["evaporating_pressure_mpa"] == pytest.approx(0.3)
    assert row["water_in_temperature"] == pytest.approx(40.0)
    assert row["water_out_temperature"] == pytest.approx(45.0)
    assert row["coil_temperature"] == pytest.approx(-8.0)
    assert row["water_temperature_setpoint"] == pytest.approx(50.0)
