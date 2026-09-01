from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frost_analysis.cost.iterations import marginal_dinkelbach, select_final_basin
from frost_analysis.cost.selected import (
    DURATION_BY_EXPERIMENT,
    ED_BY_EXPERIMENT,
    FIXED_RECOVERY_ELECTRICITY_KWH,
    QD_COEFFICIENTS,
    _latest_supported_in_optimal_basin,
    _predict_candidate_duration,
    build_cost_function_table,
    write_cost_function_csv,
)


class FakeLoader:
    def __init__(self) -> None:
        self.start = pd.Timestamp("2026-01-01")
        self.frames = {cycle: self._frame() for cycle in ("cycle_a", "cycle_b")}

    def _frame(self) -> pd.DataFrame:
        timestamps = pd.date_range(self.start, periods=721, freq="s")
        state = np.where(timestamps < self.start + pd.Timedelta(minutes=10), 41.0, 42.0)
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "water_in_temperature": state,
                "water_out_temperature": state + 5,
                "water_flow": 1.0,
                "power_total": 2.0,
                "heating_capacity": 4.0,
                "coil_temperature": state - 50,
                "water_temperature_setpoint": 50.0,
            }
        )
        terminal = pd.DataFrame(
            {
                "timestamp": [
                    self.start + pd.Timedelta(minutes=20),
                    self.start + pd.Timedelta(minutes=20, seconds=10),
                ],
                "coil_temperature": [-5.0, 25.0],
            }
        )
        return pd.concat([frame, terminal], ignore_index=True)

    def get_cycle_record(self, cycle_name: str) -> dict[str, object]:
        assert cycle_name in self.frames
        return {
            "heating_start": self.start,
            "stable_heating_start": self.start + pd.Timedelta(minutes=10),
            "defrost_preparation_start": self.start + pd.Timedelta(minutes=19, seconds=30),
            "defrost_start": self.start + pd.Timedelta(minutes=20),
            "defrost_end": self.start + pd.Timedelta(minutes=21),
        }

    def load_cycle_original(
        self, cycle_name: str, *, columns: list[str] | None = None
    ) -> pd.DataFrame:
        frame = self.frames[cycle_name].copy()
        for column in columns or []:
            if column not in frame:
                frame[column] = np.nan
        return frame if columns is None else frame[columns]


class FailingLoader:
    def get_cycle_record(self, cycle_name: str) -> dict[str, object]:
        raise AssertionError("v1 must not load original cycle data")

    def load_cycle_original(self, *args, **kwargs) -> pd.DataFrame:
        raise AssertionError("v1 must not load original cycle data")


class MixedSetpointLoader(FakeLoader):
    def load_cycle_original(
        self, cycle_name: str, *, columns: list[str] | None = None
    ) -> pd.DataFrame:
        frame = super().load_cycle_original(cycle_name, columns=columns)
        if cycle_name == "cycle_b":
            frame["water_temperature_setpoint"] = 55.0
        return frame


@pytest.fixture
def tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp("2026-01-01")
    times = [start + pd.Timedelta(minutes=10), start + pd.Timedelta(minutes=11)]
    base = pd.DataFrame(
        [
            {
                "cycle_name": cycle,
                "candidate_time": candidate,
                "heating_electricity_kwh": electricity,
                "water_heating_kwh": water_heat,
                "unit_heating_kwh": unit_heat,
                "evaporating_pressure_mpa": 0.2,
                "integration_coverage": 1.0,
                "water_reference": f"{cycle}-reference",
            }
            for cycle in ("cycle_a", "cycle_b")
            for candidate, electricity, water_heat, unit_heat in zip(
                times, [1.0, 1.2], [5.0, 9.0], [4.0, 8.0], strict=True
            )
        ]
    )
    points = pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "cycle_b"],
            "experiment_id": ["exp_20260714", "exp_20260714"],
            "t_heating_stable": [start, start],
            "t_actual_preparation": times,
            "t_RB": times[::-1],
            "rb_status": ["triggered", "right_censored"],
            "trigger_type": ["Case1", ""],
            "actual_minutes_from_stable": [10.0, 11.0],
        }
    )
    return base, points


def test_selected_models_are_embedded() -> None:
    assert "exp_20260714" in ED_BY_EXPERIMENT
    assert QD_COEFFICIENTS["intercept"] > 0
    assert build_cost_function_table.__module__ == "frost_analysis.cost.selected"


def test_v1_uses_unit_heat_without_loading_cycle_data(tables) -> None:
    base, points = tables
    result = build_cost_function_table(base, points, FailingLoader(), "v1")

    intercept, linear, quadratic, *_ = ED_BY_EXPERIMENT["exp_20260714"]
    defrost = intercept + linear * 0.2 + quadratic * 0.2**2
    first = result.iloc[0]
    assert first["inverse_cop"] == pytest.approx(
        (1.0 + defrost + FIXED_RECOVERY_ELECTRICITY_KWH) / 4.0
    )
    assert first["water_reference"] == "cycle_a-reference"
    assert result["valid"].all()
    assert not result.duplicated(["cycle_name", "candidate_time"]).any()


def test_v2_uses_pre_action_state_and_reconstructed_rule_duration(tables) -> None:
    base, points = tables
    result = build_cost_function_table(base, points, FakeLoader(), "v2")

    cycle = result.loc[result["cycle_name"].eq("cycle_b")].reset_index(drop=True)
    observed_duration = (9 + 40) / 60
    assert cycle["observed_rule_defrost_duration_minutes"].unique() == pytest.approx(
        [observed_duration]
    )
    assert cycle.loc[0, "water_in_temperature"] == pytest.approx(41.0)
    assert cycle.loc[1, "water_in_temperature"] == pytest.approx(42.0)
    assert cycle["recovery_electricity_kwh"].eq(0.250930).all()
    assert cycle["recovery_heat_kwh"].eq(0.804970).all()
    assert cycle["valid"].all()


def test_v2_marks_missing_qd_state_unsupported(tables) -> None:
    base, points = tables
    loader = FakeLoader()
    frame = loader.frames["cycle_a"]
    before_first = frame["timestamp"].lt(loader.start + pd.Timedelta(minutes=10))
    frame.loc[before_first, "water_in_temperature"] = np.nan

    result = build_cost_function_table(base, points, loader, "v2")
    missing = result.loc[result["cycle_name"].eq("cycle_a")].iloc[0]

    assert not missing["qd_supported"]
    assert "water_in_temperature:missing" in missing["qd_outside_terms"].split(",")
    assert np.isnan(missing["qd_max_normalized_extrapolation"])
    assert not missing["qd_eligible"]
    assert not missing["optimization_eligible"]


def test_v21_uses_unit_heat_and_signed_transient_heat(tables) -> None:
    base, points = tables
    result = build_cost_function_table(base, points, FakeLoader(), "v2.1")
    v2 = build_cost_function_table(base, points, FakeLoader(), "v2")

    first = result.loc[result["cycle_name"].eq("cycle_a")].iloc[0]
    expected_preparation_heat = -0.049851 - 0.001875 * 41.0 + 0.002498 * 46.0 + 0.142823 * 0.5
    assert first["preparation_duration_minutes"] == pytest.approx(0.5)
    assert first["preparation_heat_kwh"] == pytest.approx(expected_preparation_heat)
    assert first["defrost_heat_kwh"] == pytest.approx(-first["defrost_absorbed_heat_kwh"])
    assert first["recovery_heat_kwh"] == pytest.approx(1.057650730994152)
    assert first["user_heating_kwh"] == pytest.approx(
        first["unit_heating_kwh"] + first["preparation_heat_kwh"] + first["defrost_heat_kwh"]
    )
    assert first["cycle_user_heating_kwh"] == pytest.approx(
        first["user_heating_kwh"] + first["recovery_heat_kwh"]
    )
    assert np.allclose(
        result[
            [
                "defrost_electricity_kwh",
                "recovery_electricity_kwh",
                "cycle_electricity_kwh",
            ]
        ],
        v2[
            [
                "defrost_electricity_kwh",
                "recovery_electricity_kwh",
                "cycle_electricity_kwh",
            ]
        ],
    )


def test_v22_uses_water_heat_for_heating_and_recovery(tables) -> None:
    base, points = tables
    result = build_cost_function_table(base, points, FakeLoader(), "v2.2")
    v21 = build_cost_function_table(base, points, FakeLoader(), "v2.1")

    first = result.loc[result["cycle_name"].eq("cycle_a")].iloc[0]
    assert first["recovery_heat_kwh"] == pytest.approx(0.804970)
    assert first["user_heating_kwh"] == pytest.approx(
        first["water_heating_kwh"] + first["preparation_heat_kwh"] + first["defrost_heat_kwh"]
    )
    assert first["cycle_user_heating_kwh"] == pytest.approx(
        first["user_heating_kwh"] + first["recovery_heat_kwh"]
    )
    assert np.allclose(
        result[
            [
                "defrost_electricity_kwh",
                "recovery_electricity_kwh",
                "cycle_electricity_kwh",
            ]
        ],
        v21[
            [
                "defrost_electricity_kwh",
                "recovery_electricity_kwh",
                "cycle_electricity_kwh",
            ]
        ],
    )


def test_v23_uses_nine_minute_recovery_for_both_setpoints(tables) -> None:
    base, points = tables
    result = build_cost_function_table(base, points, MixedSetpointLoader(), "v2.3")

    recovery = result.groupby("cycle_name").first()
    assert recovery["recovery_duration_minutes"].eq(9.0).all()
    assert recovery.loc["cycle_a", "recovery_electricity_kwh"] == pytest.approx(0.25093046783625733)
    assert recovery.loc["cycle_a", "recovery_heat_kwh"] == pytest.approx(0.80496951375)
    assert recovery.loc["cycle_b", "recovery_electricity_kwh"] == pytest.approx(0.2515340107709751)
    assert recovery.loc["cycle_b", "recovery_heat_kwh"] == pytest.approx(0.786563833239796)
    first = result.loc[result["cycle_name"].eq("cycle_b")].iloc[0]
    assert first["user_heating_kwh"] == pytest.approx(
        first["water_heating_kwh"] + first["preparation_heat_kwh"] + first["defrost_heat_kwh"]
    )


def test_v24_starts_heating_at_the_nine_minute_cycle_boundary(tables) -> None:
    base, points = tables
    v23 = build_cost_function_table(base, points, FakeLoader(), "v2.3")
    v24 = build_cost_function_table(base, points, FakeLoader(), "v2.4")

    assert v24["recovery_duration_minutes"].eq(9.0).all()
    assert np.allclose(
        v24["heating_electricity_kwh"] - v23["heating_electricity_kwh"],
        2.0 / 60,
    )
    assert np.allclose(
        v24["water_heating_kwh"] - v23["water_heating_kwh"],
        1.161 * 5.0 / 60,
    )


def test_v25_uses_current_cycle_water_heat_without_a_future_ticket(tables) -> None:
    base, points = tables
    v23 = build_cost_function_table(base, points, FakeLoader(), "v2.3")
    v25 = build_cost_function_table(base, points, FakeLoader(), "v2.5")

    assert v25["recovery_electricity_kwh"].eq(0.0).all()
    assert v25["recovery_heat_kwh"].eq(0.0).all()
    assert np.allclose(
        v25["heating_electricity_kwh"] - v23["heating_electricity_kwh"],
        2.0 * 10 / 60,
    )
    assert np.allclose(
        v25["water_heating_kwh"] - v23["water_heating_kwh"],
        1.161 * 5.0 * 10 / 60,
    )
    assert np.allclose(
        v25["user_heating_kwh"],
        v25["water_heating_kwh"] + v25["preparation_heat_kwh"] + v25["defrost_heat_kwh"],
    )


def test_v26_changes_only_current_cycle_heat_from_water_to_unit(tables) -> None:
    base, points = tables
    v23 = build_cost_function_table(base, points, FakeLoader(), "v2.3")
    v25 = build_cost_function_table(base, points, FakeLoader(), "v2.5")
    v26 = build_cost_function_table(base, points, FakeLoader(), "v2.6")

    assert np.allclose(v26["cycle_electricity_kwh"], v25["cycle_electricity_kwh"])
    assert v26["recovery_electricity_kwh"].eq(0.0).all()
    assert v26["recovery_heat_kwh"].eq(0.0).all()
    assert np.allclose(
        v26["unit_heating_kwh"] - v23["unit_heating_kwh"],
        4.0 * 10 / 60,
    )
    assert np.allclose(
        v26["user_heating_kwh"],
        v26["unit_heating_kwh"] + v26["preparation_heat_kwh"] + v26["defrost_heat_kwh"],
    )
    assert v26["model_supported"].equals(
        v26[["pe_supported", "qd_supported", "qprep_supported"]].fillna(False).all(axis=1)
    )
    assert v26.groupby("cycle_name")["t_star_model_supported"].nunique().eq(1).all()


def test_v261_is_a_labeled_v26_alias(tables, tmp_path: Path) -> None:
    base, points = tables
    v26 = build_cost_function_table(base, points, FakeLoader(), "v2.6")
    v261 = build_cost_function_table(base, points, FakeLoader(), "v2.6.1")

    pd.testing.assert_frame_equal(v261.drop(columns="algorithm"), v26.drop(columns="algorithm"))
    assert v261["algorithm"].eq("v2.6.1").all()
    assert write_cost_function_csv(v261, tmp_path, "v2.6.1").name == ("cost_function_v2.6.1.csv")


def test_v262_closes_the_cycle_without_reusing_observed_recovery(tables) -> None:
    base, points = tables
    v26 = build_cost_function_table(base, points, FakeLoader(), "v2.6")
    original_v26 = v26.copy(deep=True)
    result = build_cost_function_table(base, points, FakeLoader(), "v2.6.2")

    first = result.loc[result["cycle_name"].eq("cycle_a")].iloc[0]
    expected_electricity = (
        first["stable_heating_electricity_kwh"] + first["defrost_electricity_kwh"] + 0.250930
    )
    expected_heat = (
        first["stable_unit_heating_kwh"]
        + first["preparation_heat_kwh"]
        - first["defrost_absorbed_heat_kwh"]
        + 0.804970
    )

    assert first["observed_prefix_recovery_electricity_kwh"] == pytest.approx(2 / 6)
    assert first["observed_prefix_recovery_unit_heat_kwh"] == pytest.approx(4 / 6)
    assert first["stable_heating_electricity_kwh"] == pytest.approx(1.0)
    assert first["stable_unit_heating_kwh"] == pytest.approx(4.0)
    assert first["projected_recovery_electricity_kwh"] == pytest.approx(0.250930)
    assert first["projected_recovery_heat_kwh"] == pytest.approx(0.804970)
    assert first["recovery_electricity_kwh"] == pytest.approx(0.250930)
    assert first["recovery_heat_kwh"] == pytest.approx(0.804970)
    assert first["transition_electricity_kwh"] == pytest.approx(
        first["defrost_electricity_kwh"] + 0.250930
    )
    assert first["transition_service_heat_kwh"] == pytest.approx(
        first["preparation_heat_kwh"] - first["defrost_absorbed_heat_kwh"] + 0.804970
    )
    assert first["cycle_electricity_kwh"] == pytest.approx(expected_electricity)
    assert first["cycle_user_heating_kwh"] == pytest.approx(expected_heat)
    assert first["inverse_cop"] == pytest.approx(expected_electricity / expected_heat)
    assert first["algorithm"] == "v2.6.2"
    assert first["model_protocol"] == "stable_to_stable_projected_post_defrost_recovery"
    assert first["t_star_model_supported"] == first["model_supported"]
    assert result["valid"].all()
    assert not result.duplicated(["cycle_name", "candidate_time"]).any()
    assert "stable_heating_electricity_kwh" not in v26
    pd.testing.assert_frame_equal(v26, original_v26)


def test_v263_separates_baseline_heating_and_transition_degradation(tables) -> None:
    base, points = tables
    v262 = build_cost_function_table(base, points, FakeLoader(), "v2.6.2")
    original_v262 = v262.copy(deep=True)
    result = build_cost_function_table(base, points, FakeLoader(), "v2.6.3")

    cycle = result.loc[result["cycle_name"].eq("cycle_a")].reset_index(drop=True)
    baseline = (
        cycle.loc[1, "stable_heating_electricity_kwh"] / cycle.loc[1, "stable_unit_heating_kwh"]
    )
    expected_heating_loss = (
        cycle["stable_heating_electricity_kwh"] - baseline * cycle["stable_unit_heating_kwh"]
    ).clip(lower=0)
    expected_transition_loss = (
        cycle["transition_electricity_kwh"] - baseline * cycle["transition_service_heat_kwh"]
    ).clip(lower=0)

    assert cycle["baseline_inverse_cop"].eq(baseline).all()
    assert cycle["baseline_candidate_count"].eq(2).all()
    assert np.allclose(cycle["heating_degradation_electricity_kwh"], expected_heating_loss)
    assert np.allclose(cycle["transition_excess_electricity_kwh"], expected_transition_loss)
    assert np.allclose(
        cycle["total_excess_electricity_kwh"],
        expected_heating_loss + expected_transition_loss,
    )
    assert np.allclose(
        cycle["inverse_cop"],
        baseline + cycle["total_excess_electricity_kwh"] / cycle["stable_unit_heating_kwh"],
    )
    assert cycle["algorithm"].eq("v2.6.3").all()
    assert cycle["model_protocol"].eq("baseline_normalized_degradation").all()
    assert result["valid"].all()
    assert not result.duplicated(["cycle_name", "candidate_time"]).any()
    pd.testing.assert_frame_equal(v262, original_v262)


def test_v264_uses_loeo_shadow_and_five_point_marginal_balance() -> None:
    start = pd.Timestamp("2026-01-01")
    rows = []
    for experiment, cycle, excess in (
        ("exp_a", "cycle_a", [1, 2, 3, 4, 5, 6]),
        ("exp_b", "cycle_b", [2, 4, 6, 8, 10, 12]),
    ):
        for minute, total_excess in enumerate(excess, 1):
            rows.append(
                {
                    "experiment_id": experiment,
                    "cycle_name": cycle,
                    "candidate_time": start + pd.Timedelta(minutes=minute),
                    "t_heating_stable": start,
                    "stable_unit_heating_kwh": float(minute),
                    "total_excess_electricity_kwh": float(total_excess),
                    "baseline_inverse_cop": 0.4,
                    "optimization_eligible": True,
                    "model_supported": True,
                }
            )

    result = marginal_dinkelbach(pd.DataFrame(rows))
    cycle = result.loc[result["cycle_name"].eq("cycle_a")].reset_index(drop=True)

    assert cycle["shadow_excess_per_heating_kwh"].eq(2.0).all()
    assert cycle.loc[0, "marginal_window_minutes"] == pytest.approx(1.0)
    assert cycle.loc[5, "marginal_window_minutes"] == pytest.approx(5.0)
    assert cycle.loc[5, "marginal_delta_excess_electricity_kwh"] == pytest.approx(5.0)
    assert cycle.loc[5, "marginal_delta_heating_kwh"] == pytest.approx(5.0)
    assert cycle.loc[5, "marginal_delta_g_kwh"] == pytest.approx(-5.0)
    assert cycle.loc[5, "inverse_cop"] == pytest.approx(1.4)
    assert result["algorithm"].eq("v2.6.4").all()
    assert result["marginal_eligible"].all()
    assert not result.duplicated(["cycle_name", "candidate_time"]).any()


def test_v265_predicts_unclipped_candidate_duration_and_support() -> None:
    coefficients = DURATION_BY_EXPERIMENT["exp_20260714"]
    candidates = pd.DataFrame({"coil_temperature": [-10.0, 100.0]})

    result = _predict_candidate_duration(candidates, "exp_20260714")

    assert coefficients[:2] == pytest.approx((3.6857790568881903, -0.07108475653641054))
    assert result.loc[0, "predicted_rule_defrost_duration_minutes"] == pytest.approx(
        coefficients[0] - 10 * coefficients[1]
    )
    assert result.loc[1, "predicted_rule_defrost_duration_minutes"] == pytest.approx(
        coefficients[0] + 100 * coefficients[1]
    )
    assert result["candidate_duration_supported"].tolist() == [True, False]


def test_v265_selects_latest_confirmed_point_and_marks_censoring() -> None:
    start = pd.Timestamp("2026-01-01")
    curve = pd.DataFrame(
        {
            "candidate_time": pd.date_range(start, periods=5, freq="min"),
            "near_optimal_1pct": [False, True, True, True, False],
            "model_supported": [True, True, True, True, True],
            "marginal_delta_g_kwh": [-1.0, -1.0, 0.1, 0.2, 0.3],
            "inverse_cop": [1.1, 1.0, 1.005, 1.008, 1.2],
        }
    )

    selected = select_final_basin(curve, start + pd.Timedelta(minutes=1), "interior")
    censored = select_final_basin(curve, start + pd.Timedelta(minutes=1), "right_observed")

    assert selected["t_star"] == start + pd.Timedelta(minutes=3)
    assert selected["decision_status"] == "supported_optimal"
    assert selected["hard_label_eligible"]
    assert censored["decision_status"] == "right_censored_lower_bound"
    assert not censored["hard_label_eligible"]


def test_v265_full_table_contract(tables) -> None:
    base, points = tables
    points = points.copy()
    points.loc[points["cycle_name"].eq("cycle_b"), "experiment_id"] = "exp_20260715"

    result = build_cost_function_table(base, points, FakeLoader(), "v2.6.5")

    required = {
        "raw_t_star",
        "decision_regret",
        "decision_status",
        "hard_label_eligible",
        "candidate_duration_supported",
        "t_star_model_supported",
    }
    assert required <= set(result)
    assert result["valid"].all()
    assert result.groupby("cycle_name")["t_star"].nunique().eq(1).all()
    assert np.allclose(
        result["inverse_cop"],
        result["baseline_inverse_cop"]
        + result["total_excess_electricity_kwh"] / result["stable_unit_heating_kwh"],
    )
    assert result.groupby("cycle_name")["decision_regret"].first().le(0.01).all()
    assert result.loc[result["hard_label_eligible"], "t_star_model_supported"].all()
    assert (
        result.loc[result["hard_label_eligible"], "decision_status"].eq("supported_optimal").all()
    )
    assert not result.duplicated(["cycle_name", "candidate_time"]).any()


def test_v3_uses_closed_cycle_lower_heat_and_loeo_risk_margins(tables) -> None:
    base, points = tables
    result = build_cost_function_table(base, points, FakeLoader(), "v3")

    first = result.loc[result["cycle_name"].eq("cycle_a")].iloc[0]
    transient_heat = first["preparation_heat_kwh"] + first["defrost_heat_kwh"]
    nominal_water = first["water_heating_kwh"] + transient_heat
    nominal_unit = first["unit_heating_kwh"] + transient_heat
    nominal_electricity = first["heating_electricity_kwh"] + first["defrost_electricity_kwh"]

    assert first["nominal_cycle_user_heating_kwh"] == pytest.approx(
        min(nominal_water, nominal_unit)
    )
    assert first["robust_cycle_user_heating_kwh"] == pytest.approx(
        min(nominal_water, nominal_unit) - 0.03579830227974917
    )
    assert first["nominal_cycle_electricity_kwh"] == pytest.approx(nominal_electricity)
    assert first["robust_cycle_electricity_kwh"] == pytest.approx(
        nominal_electricity + 0.004859499035670531
    )
    assert first["inverse_cop"] == pytest.approx(
        first["robust_cycle_electricity_kwh"] / first["robust_cycle_user_heating_kwh"]
    )
    assert first["excess_electricity_kwh"] >= 0
    assert first["model_supported"] == (
        first["pe_supported"] and first["qd_supported"] and first["qprep_supported"]
    )


def test_latest_supported_choice_stays_in_the_optimum_basin() -> None:
    start = pd.Timestamp("2026-01-01")
    curve = pd.DataFrame(
        {
            "candidate_time": pd.date_range(start, periods=7, freq="min"),
            "near_optimal_1pct": [False, True, True, False, True, True, True],
            "model_supported": [False, False, True, False, True, True, True],
        }
    )

    chosen, status = _latest_supported_in_optimal_basin(curve, start + pd.Timedelta(minutes=1))

    assert chosen == start + pd.Timedelta(minutes=2)
    assert status == "latest_supported_in_1pct_basin"


def test_writer_uses_one_clear_filename(tables, tmp_path: Path) -> None:
    base, points = tables
    result = build_cost_function_table(base, points, FailingLoader(), "v1")

    path = write_cost_function_csv(result, tmp_path / "nested", "v1")

    assert path == tmp_path / "nested" / "cost_function_v1.csv"
    assert path.is_file()
