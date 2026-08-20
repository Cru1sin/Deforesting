from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pandas as pd
import pytest

from frost_analysis.defrost_cost import partial_pool_group_estimates


def _module():  # type: ignore[no-untyped-def]
    path = Path("scripts/analyze_optimal_window_evidence.py")
    spec = importlib.util.spec_from_file_location("optimal_window_evidence", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preceding_features_use_raw_instantaneous_cop() -> None:
    module = _module()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="20s"),
            "water_flow": [1.0] * 4,
            "water_in_temperature": [30.0] * 4,
            "water_out_temperature": [35.0] * 4,
            "power_total": [2.0] * 4,
        }
    )

    values = module.preceding_features(
        frame,
        pd.Timestamp("2026-01-01 00:01:00"),
        include_dynamics=True,
    )

    assert values["q_heating_kw"] == pytest.approx(5.805)
    assert values["cop"] == pytest.approx(2.9025)


def test_preceding_features_exposes_level_slope_and_iqr() -> None:
    module = _module()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="20s"),
            "water_flow": [1.0] * 4,
            "water_in_temperature": [30.0] * 4,
            "water_out_temperature": [34.0, 35.0, 36.0, 37.0],
            "power_total": [2.0] * 4,
        }
    )

    values = module.preceding_features(
        frame,
        pd.Timestamp("2026-01-01 00:01:00"),
        include_dynamics=True,
    )

    assert values["water_out_temperature"] == pytest.approx(36.0)
    assert values["water_out_temperature_slope_per_min"] == pytest.approx(3.0)
    assert values["water_out_temperature_iqr"] == pytest.approx(1.0)


def test_preceding_features_silently_marks_all_missing_channel() -> None:
    module = _module()
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01")],
            "water_flow": [1.0],
            "water_in_temperature": [30.0],
            "water_out_temperature": [35.0],
            "power_total": [2.0],
            "ambient_temperature": [float("nan")],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        values = module.preceding_features(frame, pd.Timestamp("2026-01-01"))

    assert pd.isna(values["ambient_temperature"])
    assert not caught


def test_ticket_predictions_leave_out_whole_experiment() -> None:
    module = _module()
    events = pd.DataFrame(
        {
            "experiment_id": ["a", "a", "b", "b"],
            "equivalent_cost_kwh": [1.0, 1.0, 3.0, 3.0],
            "duration_minutes": [10.0, 10.0, 20.0, 20.0],
            "electricity_kwh": [0.2, 0.2, 0.4, 0.4],
            "thermal_shortfall_kwh": [2.0, 2.0, 4.0, 4.0],
            "minutes_from_stable": [20.0, 30.0, 20.0, 30.0],
            "cop": [2.0, 2.1, 2.0, 2.1],
            "cop_slope_per_min": [0.1, 0.1, -0.1, -0.1],
        }
    )

    result = module.leave_one_experiment_out_ticket_predictions(
        events,
        ["minutes_from_stable", "cop"],
        ["minutes_from_stable", "cop", "cop_slope_per_min"],
    )

    assert result.loc[result["experiment_id"].eq("a"), "predicted_mean_cost"].eq(3.0).all()
    assert result.loc[result["experiment_id"].eq("b"), "predicted_mean_cost"].eq(1.0).all()
    assert result["training_event_count"].eq(2).all()
    assert {
        "predicted_dynamic_cost",
        "predicted_nonlinear_cost",
        "predicted_component_cost",
    } <= set(result)


def test_conditional_curve_returns_window_and_earliest_minimum() -> None:
    module = _module()
    curves = pd.DataFrame(
        {
            "cycle_name": ["cycle"] * 4,
            "candidate_time": pd.date_range("2026-01-01", periods=4, freq="1min"),
            "renewal_cost_conditional": [2.0, 1.0, 1.005, 1.2],
        }
    )

    result = module.conditional_optimal_points(curves, threshold=0.01).iloc[0]

    assert result["t_star_conditional"] == pd.Timestamp("2026-01-01 00:01:00")
    assert result["near_opt_start_conditional"] == pd.Timestamp("2026-01-01 00:01:00")
    assert result["near_opt_end_conditional"] == pd.Timestamp("2026-01-01 00:02:00")


def test_partial_pool_cost_is_between_experiment_and_global_means() -> None:
    events = pd.DataFrame(
        {
            "experiment_id": ["a", "a", "a", "b", "b", "b"],
            "equivalent_cost_kwh": [0.8, 1.0, 1.2, 2.8, 3.0, 3.2],
            "duration_minutes": [10.0, 11.0, 12.0, 18.0, 19.0, 20.0],
        }
    )

    estimates = partial_pool_group_estimates(events)

    cost_a = estimates.set_index("experiment_id").loc["a", "partial_pool_cost"]
    assert events.loc[events["experiment_id"].eq("a"), "equivalent_cost_kwh"].mean() < cost_a
    assert cost_a < events["equivalent_cost_kwh"].mean()


def test_window_overview_plot_accepts_missing_image_path_from_csv(tmp_path: Path) -> None:
    module = _module()
    overview = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000001"],
            "minimum_location": ["interior"],
            "near_opt_start_minutes": [20.0],
            "near_opt_end_minutes": [30.0],
            "minutes_from_stable": [25.0],
            "cop_at_t_star_60s": [2.1],
            "front_image_path": [float("nan")],
            "front_image_available": [False],
        }
    )

    module.plot_window_cop_rgb(overview, tmp_path / "overview")

    assert (tmp_path / "overview.png").is_file()
