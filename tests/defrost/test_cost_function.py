from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _analysis_module():
    path = Path("scripts/defrost/analyze_raw_optimal_defrost.py")
    spec = importlib.util.spec_from_file_location("cost_function_analysis", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cost_module(monkeypatch: pytest.MonkeyPatch):
    analysis = _analysis_module()
    monkeypatch.setitem(sys.modules, "analyze_raw_optimal_defrost", analysis)
    path = Path("scripts/defrost/cost_function.py")
    spec = importlib.util.spec_from_file_location("cost_function_cli", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeLoader:
    def __init__(self) -> None:
        self.start = pd.Timestamp("2026-01-01")
        self.frames = {cycle: self._frame(cycle) for cycle in ("cycle_a", "cycle_b")}

    def _frame(self, cycle: str) -> pd.DataFrame:
        timestamps = pd.date_range(self.start + pd.Timedelta(minutes=9), periods=181, freq="s")
        state = np.where(timestamps < self.start + pd.Timedelta(minutes=10), 1.0, 2.0)
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "water_in_temperature": state,
                "water_out_temperature": state + 10,
                "coil_temperature": state - 5,
                "water_temperature_setpoint": 50.0,
            }
        )
        if cycle == "cycle_b":
            terminal = pd.DataFrame(
                {
                    "timestamp": [
                        self.start + pd.Timedelta(minutes=20),
                        self.start + pd.Timedelta(minutes=20, seconds=10),
                    ],
                    "coil_temperature": [-5.0, 25.0],
                }
            )
            frame = pd.concat([frame, terminal], ignore_index=True)
        return frame

    def list_cycles(self) -> pd.DataFrame:
        return pd.DataFrame({"cycle_name": ["cycle_a", "cycle_b"]})

    def get_cycle_record(self, cycle_name: str) -> dict[str, object]:
        assert cycle_name in self.frames
        return {
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
    def load_cycle_original(self, *args, **kwargs) -> pd.DataFrame:
        raise AssertionError("v1 must not load original cycle data")


@pytest.fixture
def tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, FakeLoader]:
    start = pd.Timestamp("2026-01-01")
    candidate_times = [start + pd.Timedelta(minutes=10), start + pd.Timedelta(minutes=11)]
    base = pd.DataFrame(
        [
            {
                "cycle_name": cycle,
                "candidate_time": candidate,
                "heating_electricity_kwh": electricity,
                "water_heating_kwh": water_heat,
                "unit_heating_kwh": unit_heat,
                "predicted_preparation_defrost_electricity_kwh": 0.2,
                "evaporating_pressure_mpa": 0.5,
                "optimization_eligible": True,
                "integration_eligible": True,
                "pe_supported": True,
                "support_status": "supported",
                "water_reference": f"{cycle}-reference",
            }
            for cycle in ("cycle_a", "cycle_b")
            for candidate, electricity, water_heat, unit_heat in zip(
                candidate_times, [1.0, 1.2], [5.0, 9.0], [4.0, 8.0], strict=True
            )
        ]
    )
    base.loc[0, "water_in_temperature"] = 99.0
    points = pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "cycle_b"],
            "experiment_id": ["exp_a", "exp_b"],
            "t_heating_stable": [start, start],
            "t_actual_preparation": candidate_times,
            "t_RB": candidate_times[::-1],
            "rb_status": ["triggered", "right_censored"],
            "trigger_type": ["Case1", ""],
            "actual_minutes_from_stable": [10.0, 11.0],
        }
    )
    tickets = pd.DataFrame(
        {"cycle_name": ["cycle_a"], "rule_defrost_duration_minutes": [2.0]}
    )
    terms = _analysis_module().QD_TERMS
    qd = pd.DataFrame(
        {
            "term": terms,
            "coefficient": [
                0.5
                if term == "intercept"
                else 0.1
                if term == "rule_defrost_duration_minutes"
                else 0.0
                for term in terms
            ],
            "training_min": np.nan,
            "training_max": np.nan,
        }
    )
    support = {
        "water_in_temperature": (1.0, 1.5),
        "water_out_temperature": (11.0, 11.5),
        "rule_defrost_duration_minutes": (0.5, 2.5),
        "coil_temperature": (-4.5, -3.5),
        "evaporating_pressure": (0.4, 0.6),
    }
    for term, (lower, upper) in support.items():
        qd.loc[qd["term"].eq(term), ["training_min", "training_max"]] = lower, upper
    return base, points, tickets, qd, FakeLoader()


def test_build_v1_cost_table_uses_unit_heat_and_broadcasts_metadata(tables) -> None:
    analysis = _analysis_module()
    base, points, tickets, qd, loader = tables

    result = analysis.build_cost_function_table(base, points, tickets, qd, loader, "v1")

    assert not result.duplicated(["cycle_name", "candidate_time"]).any()
    assert result[["cycle_name", "candidate_time"]].values.tolist() == sorted(
        result[["cycle_name", "candidate_time"]].values.tolist()
    )
    first = result.loc[result["cycle_name"].eq("cycle_a")].iloc[0]
    expected_ticket = 0.2 + analysis.FIXED_RECOVERY_ELECTRICITY_KWH
    assert first["inverse_cop"] == pytest.approx((1.0 + expected_ticket) / 4.0)
    assert first["water_reference"] == "cycle_a-reference"
    assert first["water_in_temperature"] == pytest.approx(99.0)
    assert result["water_out_temperature"].isna().all()
    assert set(result["valid"]) == {True}
    assert set(result["failure_reason"]) == {""}
    for column in (
        "experiment_id",
        "t_heating_stable",
        "actual_preparation_time",
        "t_RB",
        "rb_status",
        "trigger_type",
        "actual_minutes_from_stable",
        "t_star",
        "minimum_location",
        "cycle_electricity_kwh",
        "cycle_user_heating_kwh",
        "relative_regret",
        "near_optimal_1pct",
        "near_optimal_5pct",
        "optimization_eligible",
        "support_status",
        "model_protocol",
        "water_reference_t_star",
    ):
        assert column in result
    for _, cycle in result.groupby("cycle_name"):
        assert cycle["t_star"].nunique() == 1
        assert cycle["minimum_location"].nunique() == 1
        eligible = cycle.loc[cycle["optimization_eligible"]]
        expected = eligible.loc[eligible["water_reference_inverse_cop"].idxmin(), "candidate_time"]
        assert cycle["water_reference_t_star"].eq(expected).all()


def test_build_v1_cost_table_does_not_read_candidate_states(tables) -> None:
    analysis = _analysis_module()
    base, points, tickets, qd, _ = tables

    result = analysis.build_cost_function_table(
        base, points, tickets, qd, FailingLoader(), "v1"
    )

    assert result["valid"].all()


def test_candidate_state_features_returns_nan_without_empty_slice_warning() -> None:
    analysis = _analysis_module()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="s"),
            **{column: [np.nan, np.nan] for column in (
                "water_in_temperature",
                "water_out_temperature",
                "coil_temperature",
                "water_temperature_setpoint",
            )},
        }
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        result = analysis._candidate_state_features(
            frame, pd.Timestamp("2026-01-01 00:00:02")
        )

    assert all(np.isnan(value) for value in result.values())


def test_build_cost_table_only_records_data_value_errors(
    tables, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = _analysis_module()
    base, points, tickets, qd, loader = tables

    def fail(*args, **kwargs):
        raise RuntimeError("programming failure")

    monkeypatch.setattr(analysis, "_candidate_states_from_loader", fail)
    with pytest.raises(RuntimeError, match="programming failure"):
        analysis.build_cost_function_table(base, points, tickets, qd, loader, "v2")

    def invalid(*args, **kwargs):
        raise ValueError("bad cycle data")

    monkeypatch.setattr(analysis, "_candidate_states_from_loader", invalid)
    result = analysis.build_cost_function_table(base, points, tickets, qd, loader, "v2")
    assert not result["valid"].any()
    assert set(result["failure_reason"]) == {"bad cycle data"}


def test_build_v2_cost_table_uses_pre_action_state_ticket_and_terminal_fallback(tables) -> None:
    analysis = _analysis_module()
    base, points, tickets, qd, loader = tables

    result = analysis.build_cost_function_table(base, points, tickets, qd, loader, "v2")

    cycle_a = result.loc[result["cycle_name"].eq("cycle_a")]
    cycle_b = result.loc[result["cycle_name"].eq("cycle_b")]
    assert cycle_a["observed_rule_defrost_duration_minutes"].unique() == pytest.approx([2.0])
    expected_fallback = (9 + 40) / 60
    assert cycle_b[
        "observed_rule_defrost_duration_minutes"
    ].unique() == pytest.approx([expected_fallback])
    assert cycle_b["defrost_absorbed_heat_kwh"].tolist() == pytest.approx(
        [0.5 + 0.1 * expected_fallback] * 2
    )
    first = cycle_b.iloc[0]
    assert first["recovery_electricity_kwh"] == pytest.approx(0.250930)
    assert first["recovery_heat_kwh"] == pytest.approx(0.804970)
    assert first["cycle_electricity_kwh"] == pytest.approx(1.0 + 0.2 + 0.250930)
    assert first["cycle_user_heating_kwh"] == pytest.approx(
        5.0 - first["defrost_absorbed_heat_kwh"] + 0.804970
    )
    assert first["water_in_temperature"] == pytest.approx(1.0)
    assert cycle_b.iloc[1]["water_in_temperature"] == pytest.approx(2.0)


def test_v2_audits_qd_support_without_changing_cost_or_eligibility(tables) -> None:
    analysis = _analysis_module()
    base, points, tickets, qd, loader = tables

    result = analysis.build_cost_function_table(base, points, tickets, qd, loader, "v2")
    cycle_a = result.loc[result["cycle_name"].eq("cycle_b")].reset_index(drop=True)

    assert cycle_a["qd_supported"].tolist() == [True, False]
    assert cycle_a.loc[0, "qd_outside_terms"] == ""
    assert cycle_a.loc[0, "qd_max_normalized_extrapolation"] == pytest.approx(0.0)
    assert set(cycle_a.loc[1, "qd_outside_terms"].split(",")) == {
        "water_in_temperature",
        "water_out_temperature",
        "coil_temperature",
    }
    assert cycle_a.loc[1, "qd_max_normalized_extrapolation"] == pytest.approx(1.0)
    assert cycle_a["optimization_eligible"].tolist() == [True, True]
    assert cycle_a["defrost_absorbed_heat_kwh"].nunique() == 1
    qd_heat = cycle_a.loc[1, "defrost_absorbed_heat_kwh"]
    assert cycle_a.loc[1, "inverse_cop"] == pytest.approx(
        (1.2 + 0.2 + 0.250930) / (9.0 - qd_heat + 0.804970)
    )


def test_v2_marks_missing_qd_state_as_unsupported(tables) -> None:
    analysis = _analysis_module()
    base, points, tickets, qd, loader = tables
    frame = loader.frames["cycle_a"]
    before_first_candidate = frame["timestamp"].lt(loader.start + pd.Timedelta(minutes=10))
    frame.loc[before_first_candidate, "water_in_temperature"] = np.nan
    base.loc[base["cycle_name"].eq("cycle_a"), "water_in_temperature"] = np.nan

    result = analysis.build_cost_function_table(base, points, tickets, qd, loader, "v2")
    missing = result.loc[result["cycle_name"].eq("cycle_a")].iloc[0]

    assert not missing["qd_supported"]
    assert "water_in_temperature:missing" in missing["qd_outside_terms"].split(",")
    assert np.isnan(missing["qd_max_normalized_extrapolation"])
    assert not missing["qd_eligible"]
    assert not missing["optimization_eligible"]


def test_write_cost_function_csv_uses_exact_algorithm_filename(tables, tmp_path: Path) -> None:
    analysis = _analysis_module()
    base, points, tickets, qd, loader = tables
    v1 = analysis.build_cost_function_table(base, points, tickets, qd, loader, "v1")
    v2 = analysis.build_cost_function_table(base, points, tickets, qd, loader, "v2")

    assert set(v1) == set(v2)
    for column in (
        "water_reference_inverse_cop",
        "water_reference_relative_regret",
        "water_reference_t_star",
        "observed_rule_defrost_duration_minutes",
        "qd_eligible",
        "heat_balance_eligible",
    ):
        assert column in v1 and column in v2

    first = analysis.write_cost_function_csv(v1, tmp_path / "nested", "v1")
    second = analysis.write_cost_function_csv(v2, tmp_path / "nested", "v2")

    assert first == tmp_path / "nested" / "cost_function_v1.csv"
    assert second == tmp_path / "nested" / "cost_function_v2.csv"
    assert sorted(path.name for path in (tmp_path / "nested").iterdir()) == [
        "cost_function_v1.csv",
        "cost_function_v2.csv",
    ]


def test_cli_rejects_any_failed_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _cost_module(monkeypatch)

    cli._require_valid(
        pd.DataFrame({"cycle_name": ["a", "a", "b"], "valid": [True, True, True]}),
        "v1",
    )
    with pytest.raises(RuntimeError, match="v2.*failed cycles.*b"):
        cli._require_valid(
            pd.DataFrame(
                {
                    "cycle_name": ["a", "a", "b", "b"],
                    "valid": pd.Series([True, True, False, pd.NA], dtype="boolean"),
                }
            ),
            "v2",
        )
