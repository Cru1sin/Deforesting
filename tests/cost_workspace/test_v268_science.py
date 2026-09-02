from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _raw_frame(start: pd.Timestamp, periods: int = 601) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq="s")
    elapsed = np.arange(periods) / 60
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "power_total": 2.0,
            "compressor_power": 1.0,
            "heating_capacity": 6.0,
            "water_flow": 1.0,
            "water_in_temperature": 35.0,
            "water_out_temperature": 40.0,
            "coil_temperature": -8.0,
            "evaporating_pressure": 0.4 + 0.01 * elapsed,
            "water_temperature_setpoint": 50.0,
            "ambient_temperature": 2.0,
            "compressor_frequency": 70.0,
        }
    )


def test_half_open_integral_excludes_right_boundary_and_does_not_bridge_gap() -> None:
    from cost.v2_6_8_data import window_audit

    start = pd.Timestamp("2026-01-01")
    frame = _raw_frame(start, 61)
    frame.loc[60, "power_total"] = 100.0
    end = start + pd.Timedelta(seconds=60)
    audit = window_audit(frame, start, end, "power_total")
    assert audit["energy"] == pytest.approx(2 * 60 / 3600)

    gapped = frame.drop(index=range(11, 50))
    audit = window_audit(gapped, start, end, "power_total")
    assert audit["energy"] == pytest.approx(2 * 20 / 3600)
    assert not audit["valid"]


def test_features_are_strictly_pre_action_and_require_counts() -> None:
    from cost.cost_function_v2_6_8 import pre_action_features

    start = pd.Timestamp("2026-01-01")
    frame = _raw_frame(start)
    tau = start + pd.Timedelta(minutes=10)
    frame.loc[frame["timestamp"].eq(tau), "water_in_temperature"] = 999
    result = pre_action_features(frame, [tau], start).iloc[0]
    assert result["water_in_temperature"] == 35.0
    assert result["evaporating_pressure_slope_5m"] == pytest.approx(0.01)
    assert bool(result["pre_action_window_valid"])

    sparse = frame.drop(index=range(540, 553))
    assert not bool(pre_action_features(sparse, [tau], start).iloc[0]["pre_action_window_valid"])


def test_signed_water_heat_is_not_clipped() -> None:
    from cost.v2_6_8_data import window_audit

    start = pd.Timestamp("2026-01-01")
    frame = _raw_frame(start, 61)
    frame["water_out_temperature"] = 34.0
    result = window_audit(frame, start, start + pd.Timedelta(seconds=60), "water_heat")
    assert result["energy"] < 0


def test_window_maximum_gap_includes_leading_window_gap() -> None:
    from cost.v2_6_8_data import window_audit

    start = pd.Timestamp("2026-01-01")
    frame = _raw_frame(start, 61).iloc[25:]

    result = window_audit(frame, start, start + pd.Timedelta(seconds=60), "power_total")

    assert result["maximum_gap_seconds"] == 25


def test_fixed9_candidates_start_at_heating_plus_ten_and_keep_exact_end() -> None:
    from cost.v2_6_8_data import build_candidate_boundaries

    heating = pd.Timestamp("2026-01-01")
    preparation = heating + pd.Timedelta(minutes=12, seconds=30)
    result = build_candidate_boundaries("cycle", "experiment", heating, preparation)
    assert result["integration_start"].iloc[0] == heating + pd.Timedelta(minutes=9)
    assert result["candidate_time"].tolist() == [
        heating + pd.Timedelta(minutes=10),
        heating + pd.Timedelta(minutes=11),
        heating + pd.Timedelta(minutes=12),
        preparation,
    ]


def test_measurement_gate_breaks_support_run_and_connected_basin() -> None:
    from cost.cost_function_v2_6_8 import finalize_v268_curve

    times = pd.date_range("2026-01-01", periods=8, freq="min")
    curve = pd.DataFrame(
        {
            "candidate_time": times,
            "inverse_cop": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.45, 0.5],
            "heating_measurement_valid": [True, True, True, False, True, True, True, True],
            "ET_evaluable": True,
            "QT_evaluable": True,
            "ET_in_support": True,
            "QT_in_support": True,
            "pre_action_window_valid": True,
            "physical_valid": True,
        }
    )
    result = finalize_v268_curve(curve)
    assert not result["optimization_eligible"].any()
    assert result["diagnostic_minimum"].isna().all()

    curve["heating_measurement_valid"] = True
    result = finalize_v268_curve(curve)
    assert result["diagnostic_minimum"].iloc[0] == times[5]
    assert result["basin_1pct_width_minutes"].iloc[0] == 0


def test_v268_requires_in_support_even_when_transition_is_evaluable() -> None:
    from cost.cost_function_v2_6_8 import finalize_v268_curve

    times = pd.date_range("2026-01-01", periods=6, freq="min")
    curve = pd.DataFrame(
        {
            "candidate_time": times,
            "inverse_cop": np.linspace(1.0, 0.5, 6),
            "heating_measurement_valid": True,
            "ET_evaluable": True,
            "QT_evaluable": True,
            "ET_in_support": False,
            "QT_in_support": True,
            "pre_action_window_valid": True,
            "physical_valid": True,
        }
    )

    result = finalize_v268_curve(curve)

    assert result["ET_evaluable"].all()
    assert not result["model_supported"].any()
    assert not result["optimization_eligible"].any()
    assert result["support_policy"].eq("require_empirical_support").all()


def test_main_curve_decomposition_is_nan_and_labels_are_disabled() -> None:
    from cost.cost_function_v2_6_8 import finalize_v268_curve

    times = pd.date_range("2026-01-01", periods=6, freq="min")
    curve = pd.DataFrame(
        {
            "candidate_time": times,
            "inverse_cop": np.linspace(1.0, 0.5, 6),
            "heating_measurement_valid": True,
            "ET_evaluable": True,
            "QT_evaluable": True,
            "ET_in_support": True,
            "QT_in_support": True,
            "pre_action_window_valid": True,
            "physical_valid": True,
        }
    )
    result = finalize_v268_curve(curve)
    assert result["recommended_time"].isna().all()
    assert not result["hard_label_eligible"].any()
    for phase in ("preparation", "defrost", "recovery"):
        assert result[f"{phase}_energy_kwh"].isna().all()
        assert result[f"{phase}_heat_kwh"].isna().all()


def test_calculate_cycle_executes_declared_independent_ticket_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cost import cost_function_v2_6_8 as module

    heating = pd.Timestamp("2026-01-01")

    class Loader:
        def get_cycle_record(self, _: str) -> dict[str, object]:
            return {
                "experiment_id": "experiment",
                "heating_start": heating,
                "defrost_preparation_start": heating + pd.Timedelta(minutes=15),
            }

        def load_cycle_original(self, _: str, *, columns: list[str] | None = None) -> pd.DataFrame:
            return pd.DataFrame({"timestamp": [heating]})

    monkeypatch.setattr(
        module,
        "candidate_integral_table",
        lambda _frame, _start, candidates, quantity: pd.DataFrame(
            {"energy": 1.0 if quantity == "power_total" else 3.0, "valid": True},
            index=range(len(candidates)),
        ),
    )
    monkeypatch.setattr(
        module,
        "pre_action_features",
        lambda _frame, candidates, _heating: pd.DataFrame(
            {"pre_action_window_valid": True}, index=range(len(candidates))
        ),
    )
    selected: dict[str, object] = {}

    def predict(energy, heat, values, experiment_id):
        selected.update(energy=energy, heat=heat, experiment_id=experiment_id)
        return pd.DataFrame(
            {
                "transition_energy_kwh": 1.0,
                "transition_heat_kwh": 2.0,
                "E_support_distance": 0.0,
                "Q_support_distance": 0.0,
                "ET_evaluable": True,
                "QT_evaluable": True,
                "ET_in_support": True,
                "QT_in_support": True,
                "ET_supported": True,
                "QT_supported": True,
            },
            index=values.index,
        )

    monkeypatch.setattr(module, "predict_independent_targets", predict)
    static_energy, mean_heat = object(), object()
    artifacts = {
        "models": {
            "ticket_ridge_static5": {"energy": static_energy},
            "experiment_mean": {"heat": mean_heat},
        }
    }
    recipe = dict(module.DEFAULT_RECIPE)
    recipe.update(
        variant="mixed",
        transition_energy_model="ticket_ridge_static5",
        transition_heat_model="experiment_mean",
    )

    result = module.calculate_cycle(Loader(), "cycle", recipe, artifacts)

    assert selected == {
        "energy": static_energy,
        "heat": mean_heat,
        "experiment_id": "experiment",
    }
    assert result["transition_energy_kwh"].eq(1.0).all()
    assert result["transition_heat_kwh"].eq(2.0).all()


def test_event_audit_retains_any_defrost_with_missing_preparation() -> None:
    from cost.v2_6_8_data import build_event_table

    class Loader:
        def list_cycles(self, **_: object) -> pd.DataFrame:
            start = pd.Timestamp("2026-01-01")
            return pd.DataFrame(
                [
                    {
                        "cycle_name": "arbitrary_defrost_name",
                        "experiment_id": "exp",
                        "status": "valid",
                        "start_time": start,
                        "heating_start": start,
                        "defrost_preparation_start": pd.NaT,
                        "defrost_start": start + pd.Timedelta(minutes=10),
                        "defrost_end": start + pd.Timedelta(minutes=11),
                    },
                    {
                        "cycle_name": "following",
                        "experiment_id": "exp",
                        "status": "invalid",
                        "start_time": start + pd.Timedelta(minutes=11),
                        "heating_start": start + pd.Timedelta(minutes=11),
                        "defrost_preparation_start": pd.NaT,
                        "defrost_start": pd.NaT,
                        "defrost_end": pd.NaT,
                    },
                ]
            )

    result = build_event_table(Loader())
    assert result["cycle_name"].tolist() == ["arbitrary_defrost_name"]
    assert result["event_invalid_reason"].iloc[0] == "missing_defrost_preparation_start"
    assert pd.notna(result["defrost_start"].iloc[0])
    assert pd.notna(result["defrost_end"].iloc[0])


def test_candidate_cohort_excludes_experiments_without_parameter_folds() -> None:
    from cost.v2_6_8_data import candidate_cohort

    start = pd.Timestamp("2026-01-01")

    class Loader:
        def list_cycles(self, **_: object) -> pd.DataFrame:
            rows = []
            for index, experiment in enumerate(("with_fold", "missing_fold")):
                heating = start + pd.Timedelta(hours=index)
                rows.append(
                    {
                        "cycle_name": experiment,
                        "experiment_id": experiment,
                        "status": "valid",
                        "start_time": heating,
                        "heating_start": heating,
                        "stable_heating_start": heating + pd.Timedelta(minutes=9),
                        "defrost_preparation_start": heating + pd.Timedelta(minutes=15),
                        "defrost_start": heating + pd.Timedelta(minutes=16),
                        "defrost_end": heating + pd.Timedelta(minutes=18),
                    }
                )
            return pd.DataFrame(rows)

        def load_cycle_original(
            self, cycle_name: str, *, columns: list[str] | None = None
        ) -> pd.DataFrame:
            heating = start + pd.Timedelta(hours=int(cycle_name == "missing_fold"), minutes=9)
            frame = pd.DataFrame(
                {
                    "timestamp": pd.date_range(heating, periods=60, freq="s"),
                    "water_flow": 1.0,
                    "water_in_temperature": 40.0,
                    "water_out_temperature": 45.0,
                    "power_total": 2.0,
                }
            )
            return frame if columns is None else frame[columns]

    assert candidate_cohort(Loader(), {"with_fold"}) == (["with_fold"], 6)
