from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frost_analysis.cost.selected import (
    ED_BY_EXPERIMENT,
    FIXED_RECOVERY_ELECTRICITY_KWH,
    QD_COEFFICIENTS,
    build_cost_function_table,
    write_cost_function_csv,
)


class FakeLoader:
    def __init__(self) -> None:
        self.start = pd.Timestamp("2026-01-01")
        self.frames = {cycle: self._frame() for cycle in ("cycle_a", "cycle_b")}

    def _frame(self) -> pd.DataFrame:
        timestamps = pd.date_range(
            self.start + pd.Timedelta(minutes=9), periods=181, freq="s"
        )
        state = np.where(
            timestamps < self.start + pd.Timedelta(minutes=10), 41.0, 42.0
        )
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "water_in_temperature": state,
                "water_out_temperature": state + 5,
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


def test_writer_uses_one_clear_filename(tables, tmp_path: Path) -> None:
    base, points = tables
    result = build_cost_function_table(base, points, FailingLoader(), "v1")

    path = write_cost_function_csv(result, tmp_path / "nested", "v1")

    assert path == tmp_path / "nested" / "cost_function_v1.csv"
    assert path.is_file()
