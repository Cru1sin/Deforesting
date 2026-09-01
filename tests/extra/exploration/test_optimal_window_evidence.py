from __future__ import annotations

import importlib.util
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from frost_analysis.cost.core import partial_pool_group_estimates
from frost_analysis.dataset.metadata import following_cycle_names


def _module():  # type: ignore[no-untyped-def]
    path = Path("scripts/exploration/analyze_optimal_window_evidence.py")
    spec = importlib.util.spec_from_file_location("optimal_window_evidence", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_following_cycle_names_stays_within_experiment() -> None:
    catalog = pd.DataFrame(
        {
            "cycle_name": ["b2", "a2", "b1", "a1"],
            "experiment_id": ["b", "a", "b", "a"],
            "start_time": pd.to_datetime(
                [
                    "2026-01-02 01:00",
                    "2026-01-01 01:00",
                    "2026-01-02 00:00",
                    "2026-01-01 00:00",
                ]
            ),
        }
    )

    assert following_cycle_names(catalog) == {"a1": "a2", "b1": "b2"}


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


def test_preceding_features_keeps_coil_temperature_distinct_from_evaporating_temperature() -> None:
    module = _module()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="20s"),
            "water_flow": [1.0] * 4,
            "water_in_temperature": [30.0] * 4,
            "water_out_temperature": [35.0] * 4,
            "power_total": [2.0] * 4,
            "coil_temperature": [-8.0, -7.0, -6.0, 999.0],
            "evaporating_temperature": [-12.0] * 4,
        }
    )

    values = module.preceding_features(frame, pd.Timestamp("2026-01-01 00:01:00"))

    assert "coil_temperature" in module.RAW_COLUMNS
    assert values["coil_temperature"] == pytest.approx(-7.0)
    assert values["evaporating_temperature"] == pytest.approx(-12.0)


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

    assert values["water_out_temperature"] == pytest.approx(35.0)
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


def test_ticket_features_derive_defrost_duration_and_power_from_catalog() -> None:
    module = _module()

    class Loader:
        @staticmethod
        def load_cycle_original(_cycle: str, columns: list[str]) -> pd.DataFrame:
            frame = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-01-01 00:59", periods=4, freq="20s"),
                    "water_flow": [1.0] * 4,
                    "water_in_temperature": [30.0] * 4,
                    "water_out_temperature": [35.0] * 4,
                    "power_total": [2.0] * 4,
                    "coil_temperature": [-8.0] * 4,
                }
            )
            return frame.reindex(columns=columns)

    tickets = pd.DataFrame(
        {
            "cycle_name": ["good", "bad"],
            "valid": [True, True],
            "defrost_start": ["1999-01-01", "1999-01-01"],
            "defrost_electricity_kwh": [0.5, 0.5],
        }
    )
    points = pd.DataFrame(
        {"cycle_name": ["good", "bad"], "actual_minutes_from_stable": [60.0, 60.0]}
    )
    catalog = pd.DataFrame(
        {
            "cycle_name": ["good", "bad"],
            "experiment_id": ["a", "a"],
            "defrost_start": ["2026-01-01 01:00", "2026-01-01 01:05"],
            "defrost_end": ["2026-01-01 01:05", "2026-01-01 01:05"],
        }
    )

    result = module.build_ticket_features(Loader(), tickets, points, catalog).set_index(
        "cycle_name"
    )

    assert result.loc["good", "defrost_duration_minutes"] == pytest.approx(5.0)
    assert result.loc["good", "mean_defrost_power_kw"] == pytest.approx(6.0)
    assert pd.isna(result.loc["bad", "defrost_duration_minutes"])
    assert pd.isna(result.loc["bad", "mean_defrost_power_kw"])


def test_defrost_power_rows_interpolate_seconds_and_use_stage_local_time() -> None:
    module = _module()

    class Loader:
        @staticmethod
        def load_cycle_original(_cycle: str, columns: list[str]) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        [
                            "2026-01-01 00:00:00",
                            "2026-01-01 00:00:02",
                            "2026-01-01 00:00:03",
                            "2026-01-01 00:00:04",
                        ]
                    ),
                    "power_total": [1.0, 3.0, 4.0, 5.0],
                    "coil_temperature": [-10.0, 10.0, 20.0, 25.0],
                }
            ).reindex(columns=columns)

    events = pd.DataFrame(
        {
            "cycle_name": ["cycle"],
            "experiment_id": ["experiment"],
            "catalog_defrost_start": ["2026-01-01 00:00:00"],
            "catalog_defrost_end": ["2026-01-01 00:00:05"],
            "defrost_electricity_kwh": [0.01],
        }
    )

    rows = module.build_defrost_power_rows(Loader(), events)

    assert rows["elapsed_s"].tolist() == [0, 1, 2, 3, 4]
    assert rows["elapsed_min"].tolist() == pytest.approx([0, 1 / 60, 2 / 60, 3 / 60, 4 / 60])
    assert rows["power_kw"].tolist() == pytest.approx([1, 2, 3, 4, 5])
    assert rows["stage"].tolist() == [1, 1, 1, 2, 2]
    assert rows["u_min"].tolist() == pytest.approx([0, 1 / 60, 2 / 60, 0, 1 / 60])
    assert rows["cross20_s"].eq(3).all()
    assert rows["duration_rule_s"].eq(43).all()
    assert rows["duration_actual_s"].eq(5).all()
    assert rows["stage_weight"].tolist() == pytest.approx([1 / 3] * 3 + [1 / 2] * 2)


def test_defrost_power_weighting_and_markus_design_are_literal() -> None:
    module = _module()
    duration_min, u_min = np.meshgrid([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    rows = pd.DataFrame(
        {
            "duration_actual_s": duration_min.ravel() * 60,
            "u_min": u_min.ravel(),
            "power_kw": [0.2, 0.9, 2.1, 0.4, 1.8, 3.5, 1.0, 2.7, 5.2],
        }
    )
    weights = pd.Series([0.2, 0.3, 0.5, 0.1, 0.3, 0.6, 0.4, 0.4, 0.2])

    coefficients, condition = module._fit_defrost_power(
        rows, "duration_actual_s", "u_min", weights
    )
    design_for_fit = np.column_stack(
        [
            np.ones(len(rows)),
            duration_min.ravel(),
            u_min.ravel(),
            duration_min.ravel() * u_min.ravel(),
            u_min.ravel() ** 2,
            duration_min.ravel() * u_min.ravel() ** 2,
        ]
    )
    root_weight = np.sqrt(weights.to_numpy())
    expected = np.linalg.lstsq(
        design_for_fit * root_weight[:, None],
        rows["power_kw"].to_numpy() * root_weight,
        rcond=None,
    )[0]
    wrong = np.linalg.lstsq(
        design_for_fit * weights.to_numpy()[:, None],
        rows["power_kw"].to_numpy() * weights.to_numpy(),
        rcond=None,
    )[0]
    design = module._defrost_power_design(2.0, np.array([0.0, 1.0]))

    assert coefficients.tolist() == pytest.approx(expected)
    assert not np.allclose(coefficients, wrong)
    assert np.isfinite(condition)
    assert design.tolist() == [[1, 2, 0, 0, 0, 0], [1, 2, 1, 2, 1, 2]]
    assert (design @ module.MARKUS_POWER_COEFFICIENTS_KW).tolist() == pytest.approx(
        [0.4, 0.5983]
    )


def test_defrost_power_fit_rejects_rank_deficient_and_nonfinite_data() -> None:
    module = _module()
    rank_deficient = pd.DataFrame(
        {
            "duration_actual_s": [60.0] * 6,
            "u_min": [0.0] * 6,
            "power_kw": [1.0] * 6,
        }
    )
    with pytest.raises(ValueError, match="rank-deficient"):
        module._fit_defrost_power(
            rank_deficient,
            "duration_actual_s",
            "u_min",
            pd.Series([1.0] * 6),
        )

    nonfinite = rank_deficient.copy()
    nonfinite.loc[0, "power_kw"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        module._fit_defrost_power(
            nonfinite,
            "duration_actual_s",
            "u_min",
            pd.Series([1.0] * 6),
        )


def test_defrost_power_grid_caps_rule_and_does_not_extrapolate_edges() -> None:
    module = _module()

    class Loader:
        @staticmethod
        def load_cycle_original(_cycle: str, columns: list[str]) -> pd.DataFrame:
            seconds = np.r_[0, 0, np.arange(2, 355)]
            return pd.DataFrame(
                {
                    "timestamp": pd.Timestamp("2026-01-01")
                    + pd.to_timedelta(seconds, unit="s"),
                    "power_total": np.r_[1.0, 9.0, np.ones(353)],
                    "coil_temperature": -12.0 + 0.1 * seconds,
                }
            ).assign(
                power_total=lambda values: values["power_total"].mask(
                    values.index == 354
                )
            ).reindex(columns=columns)

    events = pd.DataFrame(
        {
            "cycle_name": ["cycle"],
            "experiment_id": ["experiment"],
            "catalog_defrost_start": ["2026-01-01 00:00:00"],
            "catalog_defrost_end": ["2026-01-01 00:05:55"],
            "defrost_electricity_kwh": [355 / 3600],
        }
    )

    rows = module.build_defrost_power_rows(Loader(), events)

    assert rows["cross20_s"].eq(320).all()
    assert rows["duration_rule_s"].eq(350).all()
    assert rows["duration_actual_s"].eq(355).all()
    assert rows.loc[rows["elapsed_s"].eq(0), "power_kw"].item() == 1.0
    assert rows.loc[rows["elapsed_s"].eq(1), "power_kw"].item() == 1.0
    assert 354 not in rows["elapsed_s"].values


def test_defrost_power_loeo_isolates_fold_and_integrates_exact_actual_seconds() -> None:
    module = _module()
    chunks = []
    for experiment in ("a", "b", "c"):
        for event_number, duration_s in enumerate(range(100, 106)):
            elapsed_s = np.arange(duration_s)
            stage = np.where(elapsed_s < 50, 1, 2)
            chunks.append(
                pd.DataFrame(
                    {
                        "cycle_name": f"{experiment}{event_number}",
                        "experiment_id": experiment,
                        "elapsed_s": elapsed_s,
                        "elapsed_min": elapsed_s / 60,
                        "stage": stage,
                        "u_min": np.where(stage == 1, elapsed_s, elapsed_s - 50) / 60,
                        "power_kw": 1.0,
                        "cross20_s": 50,
                        "duration_rule_s": 90,
                        "duration_actual_s": duration_s,
                        "actual_energy_kwh": duration_s / 3600,
                    }
                )
            )
    rows = pd.concat(chunks, ignore_index=True)
    rows["stage_weight"] = 1 / rows.groupby(["cycle_name", "stage"])[
        "stage"
    ].transform("size")
    assert rows.groupby(["cycle_name", "stage"])["stage_weight"].sum().eq(1).all()

    predictions, coefficients = module.leave_one_experiment_out_defrost_power(
        rows, "actual"
    )
    changed = rows.copy()
    changed.loc[changed["experiment_id"].eq("a"), ["power_kw", "actual_energy_kwh"]] = 99.0
    _, changed_coefficients = module.leave_one_experiment_out_defrost_power(
        changed, "actual"
    )

    held_a = coefficients.loc[coefficients["held_out_experiment"].eq("a")]
    changed_held_a = changed_coefficients.loc[
        changed_coefficients["held_out_experiment"].eq("a")
    ]
    pd.testing.assert_frame_equal(
        held_a.reset_index(drop=True), changed_held_a.reset_index(drop=True)
    )
    fitted = predictions.loc[~predictions["model"].eq("markus_original")]
    expected_duration = fitted["cycle_name"].str[1:].astype(int).add(100)
    assert fitted["prediction_duration_s"].tolist() == expected_duration.tolist()
    assert fitted["predicted_energy_kwh"].tolist() == pytest.approx(
        fitted["prediction_duration_s"] / 3600, abs=1e-12
    )
    assert module._integrate_power_kwh(np.ones(7)) == pytest.approx(7 / 3600)


def test_defrost_power_rule_cap_reaches_held_out_prediction_and_integral() -> None:
    module = _module()
    chunks = []
    for experiment in ("a", "b", "c"):
        for event_number, duration_rule_s in enumerate(range(330, 351, 4)):
            cross20_s = 320 if duration_rule_s == 350 else duration_rule_s - 40
            duration_actual_s = duration_rule_s + 5
            elapsed_s = np.arange(duration_actual_s)
            stage = np.where(elapsed_s < cross20_s, 1, 2)
            chunks.append(
                pd.DataFrame(
                    {
                        "cycle_name": f"{experiment}{event_number}",
                        "experiment_id": experiment,
                        "elapsed_s": elapsed_s,
                        "elapsed_min": elapsed_s / 60,
                        "stage": stage,
                        "u_min": np.where(
                            stage == 1, elapsed_s, elapsed_s - cross20_s
                        )
                        / 60,
                        "power_kw": 1.0,
                        "cross20_s": cross20_s,
                        "duration_rule_s": duration_rule_s,
                        "duration_actual_s": duration_actual_s,
                        "actual_energy_kwh": duration_actual_s / 3600,
                    }
                )
            )
    rows = pd.concat(chunks, ignore_index=True)
    rows["stage_weight"] = 1 / rows.groupby(["cycle_name", "stage"])[
        "stage"
    ].transform("size")

    predictions, _ = module.leave_one_experiment_out_defrost_power(rows, "rule")
    fitted = predictions.loc[
        ~predictions["model"].eq("markus_original")
        & predictions["prediction_duration_s"].eq(350)
    ]

    assert fitted["prediction_duration_s"].eq(350).all()
    assert fitted["predicted_second_count"].eq(350).all()
    assert fitted["predicted_energy_kwh"].tolist() == pytest.approx(
        [350 / 3600] * len(fitted), abs=1e-12
    )


def test_defrost_power_duplicate_t3_is_deterministic_and_edges_are_not_filled() -> None:
    module = _module()

    class DuplicateLoader:
        @staticmethod
        def load_cycle_original(_cycle: str, columns: list[str]) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        [
                            "2026-01-01 00:00:00",
                            "2026-01-01 00:00:00",
                            "2026-01-01 00:00:00",
                            "2026-01-01 00:00:02",
                            "2026-01-01 00:00:03",
                            "2026-01-01 00:00:04",
                        ]
                    ),
                    "power_total": [1.0, 9.0, 8.0, 3.0, 4.0, 5.0],
                    "coil_temperature": [-10.0, 99.0, np.nan, 10.0, 20.0, 25.0],
                }
            ).reindex(columns=columns)

    events = pd.DataFrame(
        {
            "cycle_name": ["cycle"],
            "experiment_id": ["experiment"],
            "catalog_defrost_start": ["2026-01-01 00:00:00"],
            "catalog_defrost_end": ["2026-01-01 00:00:05"],
            "defrost_electricity_kwh": [0.01],
        }
    )
    rows = module.build_defrost_power_rows(DuplicateLoader(), events)

    assert rows["cross20_s"].eq(3).all()
    assert rows.loc[rows["elapsed_s"].eq(0), "power_kw"].item() == 1.0
    assert rows.loc[rows["elapsed_s"].eq(1), "power_kw"].item() == 2.0

    for missing_index in (0, 4):
        class EdgeMissingLoader:
            missing = missing_index

            @classmethod
            def load_cycle_original(
                cls, _cycle: str, columns: list[str]
            ) -> pd.DataFrame:
                coil = pd.Series([-10.0, 0.0, 10.0, 20.0, 25.0]).mask(
                    lambda values: values.index == cls.missing
                )
                return pd.DataFrame(
                    {
                        "timestamp": pd.date_range(
                            "2026-01-01", periods=5, freq="s"
                        ),
                        "power_total": 1.0,
                        "coil_temperature": coil,
                    }
                ).reindex(columns=columns)

        with pytest.raises(ValueError, match="incomplete defrost trace"):
            module.build_defrost_power_rows(EdgeMissingLoader(), events)


def test_duration_t3_energy_fit_is_physical_and_leaves_out_complete_experiment() -> None:
    module = _module()
    event_rows = []
    duration_rows = []
    for experiment_index, experiment in enumerate(("a", "b", "c")):
        for event_index in range(3):
            cycle_name = f"{experiment}{event_index}"
            duration_rule_s = 240 + 10 * event_index + 5 * experiment_index
            duration_actual_s = duration_rule_s + 1
            t3_pre60 = -15 + 4 * event_index + experiment_index
            energy = (duration_rule_s / 3600) * (0.8 + 0.01 * t3_pre60)
            event_rows.append(
                {
                    "cycle_name": cycle_name,
                    "experiment_id": experiment,
                    "coil_temperature": t3_pre60,
                    "defrost_electricity_kwh": energy,
                    "predicted_t3_rule_defrost_electricity": 0.9 * energy,
                }
            )
            duration_rows.append(
                {
                    "cycle_name": cycle_name,
                    "experiment_id": experiment,
                    "duration_rule_s": duration_rule_s,
                    "duration_actual_s": duration_actual_s,
                }
            )
    events = pd.DataFrame(event_rows)
    duration_rows = pd.DataFrame(duration_rows)

    beta, condition = module._fit_duration_t3_energy(
        duration_rows["duration_rule_s"],
        events["coil_temperature"],
        events["defrost_electricity_kwh"],
    )
    predictions, coefficients = module.leave_one_experiment_out_duration_t3_energy(
        events, duration_rows, "rule"
    )
    actual_predictions, _ = module.leave_one_experiment_out_duration_t3_energy(
        events, duration_rows, "actual"
    )

    assert beta.tolist() == pytest.approx([0.8, 0.01])
    assert np.isfinite(condition)
    assert set(predictions["model"]) == {
        "additive_sensitivity",
        "fixed_mean_energy",
        "duration_only",
        "old_t3_duration",
        "duration_t3_physical",
    }
    physical = predictions.loc[predictions["model"].eq("duration_t3_physical")]
    assert physical["t3_pre60_c"].tolist() == events["coil_temperature"].tolist()
    assert physical["predicted_energy_kwh"].tolist() == pytest.approx(
        physical["actual_energy_kwh"]
    )
    expected_old = events.set_index("cycle_name")[
        "predicted_t3_rule_defrost_electricity"
    ]
    for values in (predictions, actual_predictions):
        old = values.loc[values["model"].eq("old_t3_duration")].set_index(
            "cycle_name"
        )
        assert old["predicted_energy_kwh"].tolist() == pytest.approx(
            expected_old.loc[old.index]
        )
    metrics = module.duration_t3_energy_metrics(predictions).set_index("model")
    assert pd.notna(
        metrics.loc["additive_sensitivity", "improvement_vs_duration_only_pct"]
    )
    assert pd.notna(metrics.loc["additive_sensitivity", "improved_experiment_count"])

    changed = events.copy()
    changed.loc[changed["experiment_id"].eq("a"), "coil_temperature"] = 99.0
    changed.loc[
        changed["experiment_id"].eq("a"), "defrost_electricity_kwh"
    ] = 99.0
    _, changed_coefficients = module.leave_one_experiment_out_duration_t3_energy(
        changed, duration_rows, "rule"
    )
    pd.testing.assert_frame_equal(
        coefficients.loc[coefficients["held_out_experiment"].eq("a")].reset_index(
            drop=True
        ),
        changed_coefficients.loc[
            changed_coefficients["held_out_experiment"].eq("a")
        ].reset_index(drop=True),
    )


def test_duration_t3_energy_fit_rejects_rank_deficient_and_nonfinite_data() -> None:
    module = _module()
    duration = pd.Series([240.0, 240.0, 240.0])
    t3 = pd.Series([-10.0, -10.0, -10.0])
    energy = pd.Series([0.05, 0.06, 0.07])

    with pytest.raises(ValueError, match="rank-deficient"):
        module._fit_duration_t3_energy(duration, t3, energy)
    with pytest.raises(ValueError, match="non-finite"):
        module._fit_duration_t3_energy(
            duration, pd.Series([-10.0, np.nan, -5.0]), energy
        )


def test_predefrost_sensor_increment_uses_fold_training_imputation_and_loeo() -> None:
    module = _module()
    rows = []
    for experiment_index, experiment in enumerate(("a", "b", "c")):
        for event_index in range(3):
            duration = 4.0 + event_index + 0.2 * experiment_index
            t3 = -15.0 + 2 * event_index + experiment_index
            ambient = -5.0 + 3 * event_index**2 + experiment_index
            rows.append(
                {
                    "cycle_name": f"{experiment}{event_index}",
                    "experiment_id": experiment,
                    "defrost_duration_minutes": duration,
                    "coil_temperature": t3,
                    "defrost_electricity_kwh": (
                        0.02 + 0.001 * duration + 0.002 * t3 + 0.03 * ambient
                    ),
                    "ambient_temperature": ambient,
                    "water_in_temperature": 30.0 + event_index,
                }
            )
    events = pd.DataFrame(rows)
    events.loc[events["cycle_name"].eq("a0"), "ambient_temperature"] = np.nan

    summary, predictions = module.evaluate_predefrost_sensor_increment(
        events,
        ["ambient_temperature", "water_in_temperature"],
        include_nested=False,
    )

    ranked = summary.set_index("feature")
    assert ranked.loc["ambient_temperature", "status"] == "ok"
    assert pd.notna(ranked.loc["ambient_temperature", "mse_kwh2"])
    held_a = predictions.loc[
        predictions["experiment_id"].eq("a")
        & predictions["feature"].eq("ambient_temperature")
    ].sort_values("cycle_name")
    training_median = events.loc[
        ~events["experiment_id"].eq("a"), "ambient_temperature"
    ].median()
    assert held_a.iloc[0]["feature_value_used"] == pytest.approx(training_median)
    assert held_a["feature_residual_oof"].notna().all()

    changed = events.copy()
    changed.loc[changed["experiment_id"].eq("a"), "defrost_electricity_kwh"] = 99.0
    _, changed_predictions = module.evaluate_predefrost_sensor_increment(
        changed,
        ["ambient_temperature", "water_in_temperature"],
        include_nested=False,
    )
    changed_a = changed_predictions.loc[
        changed_predictions["experiment_id"].eq("a")
    ].sort_values(["feature", "cycle_name"])
    original_a = predictions.loc[predictions["experiment_id"].eq("a")].sort_values(
        ["feature", "cycle_name"]
    )
    assert changed_a["predicted_energy_kwh"].tolist() == pytest.approx(
        original_a["predicted_energy_kwh"]
    )


def test_literature_main_candidates_exclude_deterministic_te_duplicate() -> None:
    module = _module()

    assert "evaporating_pressure" in module.LITERATURE_SENSOR_FEATURES
    assert not any(
        feature.startswith("evaporating_temperature")
        for feature in module.LITERATURE_SENSOR_FEATURES
    )
    assert set(module.LITERATURE_AUDIT_FEATURES) == {
        "evaporating_temperature",
        "evaporating_temperature_slope_per_min",
    }


def test_fixed_literature_screen_applies_thresholds_and_load_family_deduplication() -> None:
    module = _module()
    summary = pd.DataFrame(
        {
            "feature": [
                "__baseline__",
                "evaporating_pressure",
                "evaporating_temperature",
                "cop",
                "power_total",
                "compressor_power",
                "compressor_frequency",
                "q_heating_kw",
            ],
            "mse_kwh2": [0.100, 0.094, 0.080, 0.095, 0.090, 0.092, 0.093, 0.090],
            "macro_experiment_mse_kwh2": [
                0.100,
                0.094,
                0.080,
                0.095,
                0.090,
                0.092,
                0.093,
                0.096,
            ],
            "improvement_vs_baseline_pct": [0, 6, 20, 5, 10, 8, 7, 10],
            "improved_experiment_count": [0, 8, 15, 8, 10, 9, 9, 12],
        }
    )

    assert module.select_fixed_literature_features(summary) == [
        "evaporating_pressure",
        "cop",
        "power_total",
    ]


def test_fixed_literature_combination_is_frozen_across_outer_folds() -> None:
    module = _module()
    rows = []
    for experiment_index, experiment in enumerate(("a", "b", "c")):
        for event_index in range(5):
            duration = 4.0 + 0.6 * event_index + 0.2 * experiment_index
            t3 = -18.0 + 0.7 * event_index**2 + 1.1 * experiment_index
            pressure = 0.25 + 0.015 * event_index + 0.004 * experiment_index**2
            cop = 2.2 + 0.03 * event_index**3 + 0.05 * experiment_index
            power = 0.8 + 0.02 * (event_index - experiment_index) ** 2
            rows.append(
                {
                    "cycle_name": f"{experiment}{event_index}",
                    "experiment_id": experiment,
                    "defrost_duration_minutes": duration,
                    "coil_temperature": t3,
                    "evaporating_pressure": pressure,
                    "cop": cop,
                    "power_total": power,
                    "defrost_electricity_kwh": (
                        0.02
                        + 0.006 * duration
                        + 0.0005 * t3
                        + 0.03 * pressure
                        - 0.005 * cop
                        + 0.01 * power
                    ),
                }
            )
    events = pd.DataFrame(rows)
    events.loc[events["cycle_name"].eq("a0"), "cop"] = np.nan

    summary, predictions = module.evaluate_predefrost_sensor_increment(
        events,
        ["evaporating_pressure"],
        include_nested=False,
        fixed_combinations={
            module.FIXED_LITERATURE_COMBINATION: module.FIXED_LITERATURE_FEATURES
        },
    )
    fixed = predictions.loc[
        predictions["feature"].eq(module.FIXED_LITERATURE_COMBINATION)
    ]

    assert fixed["selected_features"].unique().tolist() == [
        "evaporating_pressure;cop;power_total"
    ]
    assert len(fixed) == len(events)
    assert np.isfinite(
        summary.set_index("feature").loc[
            module.FIXED_LITERATURE_COMBINATION, "max_condition_number"
        ]
    )
    assert not fixed["predicted_energy_kwh"].lt(0).any()

    changed = events.copy()
    changed.loc[changed["experiment_id"].eq("a"), "defrost_electricity_kwh"] = 99.0
    _, changed_predictions = module.evaluate_predefrost_sensor_increment(
        changed,
        ["evaporating_pressure"],
        include_nested=False,
        fixed_combinations={
            module.FIXED_LITERATURE_COMBINATION: module.FIXED_LITERATURE_FEATURES
        },
    )
    changed_a = changed_predictions.loc[
        changed_predictions["experiment_id"].eq("a")
        & changed_predictions["feature"].eq(module.FIXED_LITERATURE_COMBINATION)
    ].sort_values("cycle_name")
    original_a = fixed.loc[fixed["experiment_id"].eq("a")].sort_values("cycle_name")
    assert changed_a["predicted_energy_kwh"].tolist() == pytest.approx(
        original_a["predicted_energy_kwh"]
    )


def test_preparation_network_cohort_starts_from_valid_catalog_and_keeps_complete_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    catalog = pd.DataFrame(
        {
            "cycle_name": ["complete", "missing_input", "missing_clean", "invalid", "no_end"],
            "experiment_id": ["a", "a", "b", "b", "b"],
            "status": ["valid", "valid", "valid", "invalid", "valid"],
            "defrost_preparation_start": ["2026-01-01"] * 5,
            "defrost_end": ["2026-01-01 00:05"] * 4 + [None],
        }
    )
    values = {
        "cycle_name": ["complete", "missing_input", "missing_clean"],
        "experiment_id": ["a", "a", "b"],
        "t3_prepreparation_c": [-10.0, -11.0, -12.0],
        "evaporating_pressure": [0.3, np.nan, 0.4],
        "cop": [2.0, 2.1, 2.2],
        "compressor_power": [0.5, 0.6, 0.7],
        "evaporator_capacity_ratio_clean": [0.8, 0.9, np.nan],
        "fan_current": [1.0, 1.1, 1.2],
        "ambient_temperature": [3.0, 4.0, 5.0],
        "inclusive_energy_kwh": [0.08, 0.09, 0.10],
    }
    seen = []

    def fake_builder(_loader, tickets, _catalog):  # type: ignore[no-untyped-def]
        seen.extend(tickets["cycle_name"])
        audit = tickets[["cycle_name"]].assign(status="included", reason="")
        return pd.DataFrame(values), audit

    monkeypatch.setattr(module, "build_preparation_inclusive_events", fake_builder)
    events, audit = module.build_preparation_network_cohort(object(), catalog)

    assert seen == ["complete", "missing_input", "missing_clean"]
    assert events["cycle_name"].tolist() == ["complete"]
    reasons = audit.set_index("cycle_name")["reason"]
    assert reasons["missing_input"] == "incomplete_network_input_window"
    assert reasons["missing_clean"] == "incomplete_network_clean_baseline"


def test_preparation_network_uses_fixed_scaled_loeo_models_without_duration() -> None:
    module = _module()
    rows = []
    for experiment_index, experiment in enumerate(("a", "b", "c", "d")):
        for event_index in range(6):
            features = {
                "t3_prepreparation_c": -20 + 0.4 * event_index**2 + experiment_index,
                "evaporating_pressure": 0.2
                + 0.01 * event_index
                + 0.003 * experiment_index**2,
                "cop": 2.0 + 0.04 * event_index**3 + 0.02 * experiment_index,
                "compressor_power": 0.5
                + 0.03 * (event_index - experiment_index) ** 2,
                "evaporator_capacity_ratio_clean": 0.7
                + 0.02 * event_index
                + 0.004 * experiment_index * event_index,
                "fan_current": 0.8 + 0.01 * event_index**2 + 0.007 * experiment_index,
                "ambient_temperature": -5 + 1.5 * experiment_index + 0.1 * event_index,
            }
            rows.append(
                {
                    "cycle_name": f"{experiment}{event_index}",
                    "experiment_id": experiment,
                    **features,
                    "inclusive_energy_kwh": (
                        0.05
                        + 0.02 * features["evaporating_pressure"]
                        - 0.002 * features["cop"]
                        + 0.01 * features["compressor_power"]
                        + 0.004 * features["fan_current"]
                    ),
                }
            )
    events = pd.DataFrame(rows)

    summary, predictions = module.evaluate_preparation_network(events)

    assert module.PREPARATION_NETWORK_FEATURES == [
        "t3_prepreparation_c",
        "evaporating_pressure",
        "cop",
        "compressor_power",
        "evaporator_capacity_ratio_clean",
        "fan_current",
        "ambient_temperature",
    ]
    assert module.PREPARATION_NETWORK_HIDDEN_LAYERS == (4,)
    assert set(summary["model"]) == {
        "train_mean",
        "pe_linear",
        "pe_quadratic_ridge",
        "ridge_7",
        "ridge_7_squared",
        "mlp_7_4_1",
    }
    assert predictions.groupby("model").size().eq(len(events)).all()
    assert "defrost_duration_minutes" not in events
    assert summary.set_index("model").loc[
        ["pe_linear", "pe_quadratic_ridge", "ridge_7", "ridge_7_squared"],
        "max_condition_number",
    ].notna().all()
    assert summary.set_index("model").loc[
        ["train_mean", "mlp_7_4_1"], "max_condition_number"
    ].isna().all()

    held_a = predictions.loc[predictions["experiment_id"].eq("a")].sort_values(
        ["model", "cycle_name"]
    )
    expected_mean = events.loc[~events["experiment_id"].eq("a"), "inclusive_energy_kwh"].mean()
    assert held_a.loc[held_a["model"].eq("train_mean"), "predicted_energy_kwh"].eq(
        expected_mean
    ).all()
    changed = events.copy()
    changed.loc[changed["experiment_id"].eq("a"), "inclusive_energy_kwh"] = 99.0
    _, changed_predictions = module.evaluate_preparation_network(changed)
    changed_a = changed_predictions.loc[
        changed_predictions["experiment_id"].eq("a")
    ].sort_values(["model", "cycle_name"])
    assert changed_a["predicted_energy_kwh"].tolist() == pytest.approx(
        held_a["predicted_energy_kwh"]
    )


def test_absorbed_heat_loeo_includes_five_input_squared_model() -> None:
    module = _module()
    rows = []
    for experiment_index, experiment in enumerate(("a", "b", "c", "d")):
        for event_index in range(6):
            tin = 25.0 + event_index + 0.2 * experiment_index
            tout = 30.0 + 0.7 * event_index - 0.1 * experiment_index
            duration = 1.0 + 0.4 * event_index + 0.05 * experiment_index
            t3 = -15.0 + event_index + experiment_index
            pe = 0.20 + 0.025 * event_index + 0.004 * experiment_index
            rows.append(
                {
                    "cycle_name": f"{experiment}{event_index}",
                    "experiment_id": experiment,
                    "defrost_electricity_kwh": 0.1,
                    "defrost_duration_minutes": duration,
                    "rule_defrost_duration_minutes": duration,
                    "coil_temperature": t3,
                    "water_in_temperature": tin,
                    "water_out_temperature": tout,
                    "evaporating_pressure": pe,
                    "defrost_absorbed_heat_kwh": (
                        0.3
                        + 0.01 * tin
                        - 0.012 * tout
                        + 0.05 * duration
                        - 0.004 * t3
                        - 0.2 * pe
                        + 0.002 * duration**2
                        - 0.8 * pe**2
                    ),
                }
            )
    events = pd.DataFrame(rows)

    predictions = module.leave_one_experiment_out_defrost_predictions(events)
    metrics = module.defrost_model_metrics(predictions)

    prediction = "predicted_water_duration_t3_pe_squared_defrost_absorbed_heat"
    assert prediction in predictions
    heat_overall = metrics.loc[
        metrics["outcome"].eq("defrost_absorbed_heat")
        & metrics["experiment_id"].eq("__overall__")
    ].set_index("strategy")
    assert "water_duration_t3_pe_squared" in heat_overall.index
    assert heat_overall.loc["water_duration_t3_pe_squared", "mse"] < heat_overall.loc[
        "water_duration_t3_pe", "mse"
    ]


def test_preparation_network_rejects_missing_inputs_instead_of_imputing() -> None:
    module = _module()
    events = pd.DataFrame(
        {
            "cycle_name": ["a", "b"],
            "experiment_id": ["x", "y"],
            **{feature: [1.0, 2.0] for feature in module.PREPARATION_NETWORK_FEATURES},
            "inclusive_energy_kwh": [0.08, 0.09],
        }
    )
    events.loc[0, "fan_current"] = np.nan

    with pytest.raises(ValueError, match="complete-case"):
        module.evaluate_preparation_network(events)


def test_pe_linear_cycle_fit_exports_61_loeo_points_and_five_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    experiments = [f"e{index:02d}" for index in range(16)]
    experiment_ids = [
        experiment
        for index, experiment in enumerate(experiments)
        for _ in range(4 if index < 13 else 3)
    ]
    pe = np.linspace(0.19, 0.40, 61)
    actual = 0.125 - 0.17 * pe + np.sin(np.arange(61)) / 1000
    residual = np.linspace(-0.012, 0.009, 61)
    events = pd.DataFrame(
        {
            "cycle_name": [f"cycle_{index:03d}" for index in range(61)],
            "experiment_id": experiment_ids,
            "status": "included",
            "evaporating_pressure": pe,
            "inclusive_energy_kwh": actual,
        }
    )
    predictions = pd.DataFrame(
        {
            "cycle_name": events["cycle_name"],
            "experiment_id": experiment_ids,
            "model": "pe_quadratic_ridge",
            "predicted_energy_kwh": actual + residual,
        }
    )
    output = tmp_path / "证据"
    output.mkdir()
    events.to_csv(output / "preparation_inclusive_network_events.csv", index=False)
    predictions.to_csv(
        output / "preparation_inclusive_network_predictions.csv", index=False
    )
    saved = {}

    def capture_save(fig: object, base: Path, **kwargs: object) -> None:
        saved.update(fig=fig, base=base, kwargs=kwargs)

    monkeypatch.setattr(module, "_save_figure", capture_save)

    source = module.write_pe_linear_cycle_fit(output)

    assert len(source) == 61
    assert source["experiment_id"].nunique() == 16
    expected_labels = set(
        source.nlargest(5, "absolute_loeo_residual_kwh")["cycle_name"]
    )
    assert set(source.loc[source["label_largest_residual"], "cycle_name"]) == expected_labels
    for experiment, fold in source.groupby("experiment_id"):
        train = events.loc[events["experiment_id"].ne(experiment)]
        design = np.column_stack(
            [
                train["evaporating_pressure"],
                train["evaporating_pressure"].pow(2),
            ]
        )
        scaler = module.StandardScaler().fit(design)
        model = module.Ridge(alpha=module.PREPARATION_NETWORK_RIDGE_ALPHA).fit(
            scaler.transform(design), train["inclusive_energy_kwh"]
        )
        expected = model.coef_ / scaler.scale_
        expected_intercept = model.intercept_ - expected @ scaler.mean_
        assert fold["fold_intercept_kwh"].nunique() == 1
        assert fold["fold_linear_kwh_per_mpa"].nunique() == 1
        assert fold["fold_quadratic_kwh_per_mpa2"].nunique() == 1
        assert fold["fold_intercept_kwh"].iloc[0] == pytest.approx(expected_intercept)
        assert fold["fold_linear_kwh_per_mpa"].iloc[0] == pytest.approx(expected[0])
        assert fold["fold_quadratic_kwh_per_mpa2"].iloc[0] == pytest.approx(expected[1])
        assert fold["fold_train_pe_min_mpa"].iloc[0] == pytest.approx(
            train["evaporating_pressure"].min()
        )
        assert fold["fold_train_pe_max_mpa"].iloc[0] == pytest.approx(
            train["evaporating_pressure"].max()
        )
    assert (output / "pe_linear_cycle_fit_source.csv").is_file()
    assert (output / "pe_quadratic_ridge_fold_coefficients.csv").is_file()
    assert saved["base"] == tmp_path / "图表" / "figure_pe_linear_cycle_fit"
    assert saved["kwargs"] == {"bbox_inches": None, "tiff": True}
    figure = saved["fig"]
    assert len(figure.axes) == 1
    assert figure.get_size_inches().tolist() == pytest.approx(
        [183 / 25.4, 118 / 25.4], abs=0.01
    )
    assert len(figure.axes[0].lines) == 17
    labels = {
        text.get_text()
        for text in figure.axes[0].texts
        if text.get_text().startswith("Cycle ")
    }
    assert labels == {
        f"Cycle {int(cycle_name.rsplit('_', maxsplit=1)[-1])}"
        for cycle_name in expected_labels
    }
    module.plt.close(figure)


def test_preparation_inclusive_events_move_preparation_from_features_to_target() -> None:
    module = _module()

    class Loader:
        @staticmethod
        def load_cycle_original(_cycle: str, columns: list[str]) -> pd.DataFrame:
            seconds = np.arange(65)
            before = seconds < 60
            frame = pd.DataFrame(
                {
                    "timestamp": pd.Timestamp("2026-01-01")
                    + pd.to_timedelta(seconds, unit="s"),
                    "power_total": np.where(before, seconds, 10.0),
                    "coil_temperature": np.where(before, -20 + seconds / 10, 99.0),
                    "water_flow": np.where(before, 1 + seconds / 60, 99.0),
                    "water_in_temperature": np.where(before, 20.0, 99.0),
                    "water_out_temperature": np.where(before, 25.0, 99.0),
                    "compressor_frequency": np.where(before, 30 + seconds, 99.0),
                    "compressor_power": np.where(before, 0.5, 99.0),
                    "fan_current": np.where(before, 1 + seconds / 60, 99.0),
                    "evaporating_pressure": np.where(before, 0.3, 99.0),
                    "ambient_temperature": np.where(before, 2.0, 99.0),
                    "environment_relative_humidity": np.where(before, 80.0, 99.0),
                    "evaporating_temperature": np.where(
                        before, -10 + seconds / 30, 99.0
                    ),
                }
            )
            return frame.reindex(columns=columns)

    tickets = pd.DataFrame(
        {
            "cycle_name": ["included", "missing_boundary"],
            "experiment_id": ["a", "a"],
        }
    )
    catalog = pd.DataFrame(
        {
            "cycle_name": ["included", "missing_boundary"],
            "defrost_preparation_start": ["2026-01-01 00:01:00", None],
            "defrost_start": ["2026-01-01 00:01:02", "2026-01-01 00:01:02"],
            "defrost_end": ["2026-01-01 00:01:05", "2026-01-01 00:01:05"],
            "baseline_start": ["2026-01-01 00:00:00", None],
            "baseline_end": ["2026-01-01 00:01:00", None],
        }
    )

    events, audit = module.build_preparation_inclusive_events(
        Loader(), tickets, catalog
    )
    event = events.iloc[0]

    assert event["cycle_name"] == "included"
    assert event["inclusive_duration_minutes"] == pytest.approx(5 / 60)
    assert event["preparation_duration_s"] == 2
    assert event["inclusive_energy_kwh"] == pytest.approx(50 / 3600)
    assert event["preparation_energy_kwh"] == pytest.approx(20 / 3600)
    assert event["t3_prepreparation_c"] == pytest.approx(
        np.median(-20 + np.arange(60) / 10)
    )
    assert event["coil_temperature_slope_per_min"] == pytest.approx(6.0)
    assert event["q_heating_ratio_clean"] == pytest.approx(1.0)
    assert event["evaporator_capacity_ratio_clean"] == pytest.approx(1.0)
    assert event["cop_ratio_clean"] == pytest.approx(1.0)
    assert event["fan_current_slope_per_min"] == pytest.approx(1.0)
    assert event["power_total_slope_per_min"] == pytest.approx(60.0)
    assert audit.set_index("cycle_name").loc["missing_boundary", "status"] == "excluded"


@pytest.mark.parametrize(
    ("water_out_temperature", "expected_kwh"),
    [(35.0, 0.09675), (25.0, -0.09675)],
)
def test_preparation_events_integrate_signed_water_heat(
    water_out_temperature: float, expected_kwh: float
) -> None:
    module = _module()

    class Loader:
        @staticmethod
        def load_cycle_original(_cycle: str, columns: list[str]) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-01-01", periods=122, freq="s"),
                    "power_total": 2.0,
                    "coil_temperature": -5.0,
                    "water_flow": 1.0,
                    "water_in_temperature": 30.0,
                    "water_out_temperature": water_out_temperature,
                }
            ).reindex(columns=columns)

    tickets = pd.DataFrame(
        {"cycle_name": ["cycle"], "experiment_id": ["experiment"]}
    )
    catalog = pd.DataFrame(
        {
            "cycle_name": ["cycle"],
            "defrost_preparation_start": ["2026-01-01 00:01:00"],
            "defrost_start": ["2026-01-01 00:02:00"],
            "defrost_end": ["2026-01-01 00:02:02"],
            "baseline_start": [None],
            "baseline_end": [None],
        }
    )

    events, _ = module.build_preparation_inclusive_events(Loader(), tickets, catalog)

    assert events.loc[0, "preparation_signed_heat_kwh"] == pytest.approx(expected_kwh)


def test_preparation_heat_models_leave_out_whole_experiments() -> None:
    module = _module()
    events = pd.DataFrame(
        {
            "cycle_name": ["a1", "a2", "b1", "b2", "c1", "c2"],
            "experiment_id": ["a", "a", "b", "b", "c", "c"],
            "preparation_signed_heat_kwh": [-0.2, -0.1, 0.1, 0.2, 0.3, 0.4],
            "water_in_temperature": [30, 31, 32, 33, 34, 35],
            "water_out_temperature": [29, 30, 33, 34, 36, 37],
            "preparation_duration_minutes": [1, 1, 2, 2, 3, 3],
            "t3_prepreparation_c": [-8, -7, -6, -5, -4, -3],
            "evaporating_pressure": [0.30, 0.31, 0.32, 0.33, 0.34, 0.35],
        }
    )

    summary, predictions = module.evaluate_preparation_heat_models(events)

    held_a = predictions.loc[
        predictions["experiment_id"].eq("a")
        & predictions["model"].eq("train_mean")
    ]
    assert held_a["predicted_heat_kwh"].eq(0.25).all()
    assert set(summary["model"]) == {
        "train_mean",
        "water",
        "water_duration",
        "water_duration_t3_pe",
        "water_duration_t3_pe_squared",
    }
    assert summary["event_count"].eq(len(events)).all()
    assert summary["experiment_count"].eq(3).all()
    assert predictions.groupby("model").size().eq(len(events)).all()


def test_defrost_rule_predictions_leave_out_experiment_and_keep_recovery_separate() -> None:
    module = _module()
    events = pd.DataFrame(
        {
            "cycle_name": ["a1", "a2", "b1", "b2", "c1", "c2", "c3"],
            "experiment_id": ["a", "a", "b", "b", "c", "c", "c"],
            "defrost_electricity_kwh": [0.2, 0.8, 0.2, 0.9, 0.5, 1.4, 0.3],
            "recovery_electricity_kwh": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            "defrost_duration_minutes": [2.0, 4.0, 3.0, 6.0, 5.0, 7.0, 8.0],
            "coil_temperature": [-100.0, 100.0, 3.0, 6.0, 5.0, float("nan"), 8.0],
            "defrost_end": pd.date_range("2026-01-01 01:02", periods=7, freq="min"),
            "coil_temperature_end": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
        }
    )

    predictions = module.leave_one_experiment_out_defrost_predictions(events)
    held_a = predictions.loc[predictions["experiment_id"].eq("a")]
    assert held_a["training_mean_defrost_power_kw"].tolist() == pytest.approx(
        [198 / 29, 198 / 29]
    )
    assert held_a["predicted_fixed_defrost_electricity"].tolist() == pytest.approx(
        [0.66, 0.66]
    )
    assert held_a["predicted_known_duration_defrost_electricity"].tolist() == pytest.approx(
        [33 / 145, 66 / 145]
    )
    assert held_a["predicted_t3_duration_minutes"].tolist() == pytest.approx([3.0, 8.0])
    assert predictions.loc[
        predictions["cycle_name"].eq("c2"), "predicted_t3_duration_minutes"
    ].iloc[0] == pytest.approx(3.75)
    assert not any(
        "recovery" in column
        for column in predictions.columns
        if column.startswith("predicted_")
    )
    changed = events.copy()
    changed.loc[changed["experiment_id"].eq("a"), "defrost_duration_minutes"] = [200.0, 400.0]
    changed.loc[changed["experiment_id"].eq("a"), "defrost_end"] = pd.Timestamp("2099-01-01")
    changed.loc[changed["experiment_id"].eq("a"), "coil_temperature_end"] = 999.0
    changed.loc[changed["experiment_id"].eq("a"), "recovery_electricity_kwh"] = [1000.0, 2000.0]
    changed_a = module.leave_one_experiment_out_defrost_predictions(changed).loc[
        lambda values: values["experiment_id"].eq("a"),
        "predicted_t3_rule_defrost_electricity",
    ]
    assert changed_a.tolist() == pytest.approx(
        held_a["predicted_t3_rule_defrost_electricity"].tolist()
    )

    metrics = module.defrost_model_metrics(predictions)
    overall = metrics.loc[metrics["experiment_id"].eq("__overall__")].set_index("strategy")
    assert set(overall.index) == {"fixed", "known_duration", "t3_rule"}
    expected = {
        "fixed": (0.3464285714285715, 0.3416666666666666, 0.0),
        "known_duration": (0.3061690034103827, 0.2771971706454465, 2 / 3),
        "t3_rule": (0.2975987037607961, 0.28414053929321925, 2 / 3),
    }
    for strategy, values in expected.items():
        assert overall.loc[strategy, "event_weighted_mae"] == pytest.approx(values[0])
        assert overall.loc[strategy, "macro_mae"] == pytest.approx(values[1])
        assert overall.loc[strategy, "improved_experiment_fraction"] == pytest.approx(
            values[2]
        )


def test_rule_only_main_calls_only_ticket_evidence_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls = []
    monkeypatch.setattr(module, "write_defrost_rule_evidence", lambda *args: calls.append(args))
    monkeypatch.setattr(module, "analyze", lambda *_args: pytest.fail("full analysis called"))
    monkeypatch.setattr(sys, "argv", ["analyze_optimal_window_evidence.py", "--defrost-rule-only"])

    module.main()

    assert calls == [
        (
            Path("dataset"),
            Path("output/test/成本函数/其他/经验经济窗口/源数据"),
            Path("output/test/成本函数/其他/经验经济窗口/证据"),
        )
    ]


def test_defrost_energy_comparison_source_uses_prespecified_methods_and_baselines() -> None:
    module = _module()
    power = pd.DataFrame(
        {
            "duration_mode": ["rule"] * 3 + ["actual"] * 3,
            "model": ["markus_original", "one_stage_ls", "two_stage_ls"] * 2,
            "event_count": [68] * 6,
            "experiment_count": [15] * 6,
            "mse_kwh2": [10.0, 8.0, 9.0, 1.0, 1.0, 1.0],
            "macro_experiment_mse_kwh2": [5.0, 4.0, 4.5, 1.0, 1.0, 1.0],
        }
    )
    duration = pd.DataFrame(
        {
            "duration_mode": ["rule"] * 5 + ["actual"] * 5,
            "model": [
                "fixed_mean_energy",
                "duration_only",
                "old_t3_duration",
                "duration_t3_physical",
                "additive_sensitivity",
            ]
            * 2,
            "event_count": [68] * 10,
            "experiment_count": [15] * 10,
            "mse_kwh2": [30.0, 20.0, 15.0, 12.0, 10.0] + [1.0] * 5,
            "macro_experiment_mse_kwh2": [15.0, 10.0, 8.0, 7.0, 6.0]
            + [1.0] * 5,
        }
    )
    literature = pd.DataFrame(
        {
            "feature": [
                "__baseline__",
                "evaporating_pressure",
                "cop",
                "power_total",
                "__fixed_pe_cop_power_total__",
            ],
            "valid_n": [59] * 5,
            "mse_kwh2": [10.0, 6.0, 7.0, 8.0, 9.0],
            "macro_experiment_mse_kwh2": [5.0, 3.0, 3.5, 4.0, 4.5],
        }
    )
    network = pd.DataFrame(
        {
            "model": [
                "train_mean",
                "pe_linear",
                "pe_quadratic_ridge",
                "ridge_7",
                "ridge_7_squared",
                "mlp_7_4_1",
            ],
            "event_count": [61] * 6,
            "experiment_count": [16] * 6,
            "mse_kwh2": [8.870014779364519e-05, 3.0, 3.5, 4.0, 4.5, 5.0],
            "macro_experiment_mse_kwh2": [
                7.950072839228545e-05,
                2.0,
                2.25,
                2.5,
                2.75,
                3.0,
            ],
        }
    )

    source = module.build_defrost_energy_method_source(
        power, duration, literature, network
    )

    assert len(source) == 19
    assert source.groupby("group_id", sort=False).size().to_dict() == {
        "A": 3,
        "B": 5,
        "C": 5,
        "D": 6,
    }
    baselines = source.loc[source["method_role"].eq("baseline")]
    assert baselines["method_id"].tolist() == [
        "markus_original",
        "duration_only",
        "__baseline__",
        "train_mean",
    ]
    assert baselines["event_relative_mse_pct"].eq(100.0).all()
    assert baselines["macro_relative_mse_pct"].eq(100.0).all()
    roles = source.set_index(["group_id", "method_id"])["method_role"]
    assert roles.loc[("D", "pe_linear")] == "comparison"
    assert roles.loc[("D", "pe_quadratic_ridge")] == "selected_deployable"
    assert roles.loc[("C", "__fixed_pe_cop_power_total__")] == "rejected_complex"
    assert roles.loc[("D", "mlp_7_4_1")] == "rejected_complex"
    labels = source.set_index(["group_id", "method_id"])["method_label"]
    assert labels.str.contains("\n", regex=False).all()
    assert not labels.str.contains("selected", case=False).any()
    assert labels.loc[("C", "evaporating_pressure")] == (
        "Duration + coil temperature + Pe\nLinear regression"
    )
    assert labels.loc[("D", "train_mean")] == (
        "No event-level inputs\nHistorical mean (reference)"
    )
    assert labels.loc[("D", "pe_linear")] == "Pe only\nLinear regression"
    assert labels.loc[("D", "pe_quadratic_ridge")] == (
        "Pe + Pe²\nQuadratic Ridge regression"
    )
    assert labels.loc[("D", "ridge_7")] == "7 physical variables\nRidge regression"
    assert labels.loc[("D", "ridge_7_squared")] == (
        "7 variables + individual squares\nRidge regression"
    )
    assert labels.loc[("D", "mlp_7_4_1")] == (
        "7 physical variables\nMLP (7-4-1)"
    )
    filters = source.groupby("group_id")["source_filter"].unique()
    assert filters.loc["A"].tolist() == ["duration_mode=rule"]
    assert filters.loc["B"].tolist() == ["duration_mode=rule"]


def test_defrost_energy_comparison_plots_actual_mse_on_one_shared_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    source = pd.DataFrame(
        {
            "group_id": ["A", "B"],
            "group_title": ["First comparison", "Second comparison"],
            "target": ["energy during defrost"] * 2,
            "event_count": [10, 11],
            "experiment_count": [3, 4],
            "method_id": ["first", "second"],
            "method_label": ["First method", "Second method"],
            "method_role": ["baseline", "comparison"],
            "event_mse_kwh2": [1.0e-4, 2.0e-5],
            "macro_mse_kwh2": [8.0e-5, 3.0e-5],
            "event_relative_mse_pct": [100.0, 200.0],
            "macro_relative_mse_pct": [100.0, 150.0],
        }
    )
    captured = {}
    monkeypatch.setattr(
        module,
        "_save_figure",
        lambda fig, output, **kwargs: captured.update(
            fig=fig, output=output, kwargs=kwargs
        ),
    )

    module.plot_defrost_energy_method_comparison(source, tmp_path / "comparison")

    figure = captured["fig"]
    axis = figure.axes[0]
    assert [collection.get_offsets()[0, 0] for collection in axis.collections] == (
        pytest.approx([1.0e-4, 8.0e-5, 2.0e-5, 3.0e-5])
    )
    assert axis.get_xlabel() == "Held-out MSE (kWh²; lower is better)"
    assert axis.xaxis.get_major_formatter()._powerlimits == (0, 0)
    assert not any(
        np.allclose(line.get_xdata(), [100.0, 100.0]) for line in axis.lines
    )
    assert not any("reference = 100%" in text.get_text() for text in figure.texts)
    module.plt.close(figure)


def test_only_modes_are_mutually_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "write_defrost_power_evidence", lambda *_args: None)
    monkeypatch.setattr(module, "write_defrost_rule_evidence", lambda *_args: None)
    monkeypatch.setattr(module, "analyze", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_optimal_window_evidence.py",
            "--defrost-rule-only",
            "--defrost-power-only",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        module.main()


def test_rule_writer_preserves_sentinel_and_writes_only_two_csvs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    dataset = tmp_path / "dataset"
    source = tmp_path / "source"
    output = tmp_path / "output"
    dataset.mkdir()
    source.mkdir()
    output.mkdir()
    pd.DataFrame({"cycle_name": ["cycle"]}).to_csv(
        source / "defrost_ticket_events.csv", index=False
    )
    pd.DataFrame({"cycle_name": ["cycle"]}).to_csv(
        source / "cycle_optimal_points.csv", index=False
    )
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    class Loader:
        @staticmethod
        def list_cycles() -> pd.DataFrame:
            return pd.DataFrame({"cycle_name": ["cycle"]})

    predictions = pd.DataFrame({"cycle_name": ["cycle"], "prediction": [1.0]})
    metrics = pd.DataFrame({"strategy": ["fixed"], "mae": [0.1]})
    monkeypatch.setattr(module, "DatasetLoader", lambda _path: Loader())
    monkeypatch.setattr(
        module,
        "build_ticket_evidence",
        lambda *_args: (predictions, metrics),
    )

    module.write_defrost_rule_evidence(dataset, source, output)

    assert {path.name for path in output.iterdir()} == {
        "sentinel.txt",
        "ticket_event_features_and_predictions.csv",
        "ticket_model_metrics_by_experiment.csv",
    }
    assert sentinel.read_text(encoding="utf-8") == "keep"
    pd.testing.assert_frame_equal(
        pd.read_csv(output / "ticket_event_features_and_predictions.csv"), predictions
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(output / "ticket_model_metrics_by_experiment.csv"), metrics
    )


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


def test_component_optimum_recomputes_cost_with_strict_joint_support() -> None:
    module = _module()
    candidates = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01", periods=3, freq="1min"),
            "heating_electricity_kwh": [1.0, 1.5, 2.0],
            "water_heating_kwh": [1.0, 3.0, 5.0],
            "integration_eligible": [True, True, True],
            "pe_supported": [True, True, True],
            "qd_supported": [False, True, True],
            "t_star": pd.Timestamp("1999-01-01"),
        }
    )

    curve, point = module._recompute_component_optimum(
        candidates,
        ed_kwh=pd.Series([0.0, 0.0, 0.0]),
        qprep_kwh=pd.Series([0.0, 0.0, 0.0]),
        qd_kwh=pd.Series([0.0, 0.0, 0.0]),
        recovery_electricity_kwh=0.0,
        recovery_heat_kwh=0.0,
    )

    assert point["candidate_time"] == pd.Timestamp("2026-01-01 00:02:00")
    assert curve["optimization_eligible"].tolist() == [False, True, True]
    assert point["candidate_time"] != candidates.loc[0, "t_star"]


def test_renewal_water_table_flags_a_fully_extrapolated_cycle() -> None:
    module = _module()
    candidates = pd.DataFrame(
        {
            "cycle_name": ["cycle"] * 2,
            "candidate_time": pd.date_range("2026-01-01", periods=2, freq="1min"),
            "heating_electricity_kwh": [1.0, 2.0],
            "water_heating_kwh": [3.0, 5.0],
            "defrost_electricity_kwh": [0.1, 0.1],
            "revised_qprep_kwh": [0.1, 0.1],
            "revised_qd_kwh": [0.2, 0.2],
            "integration_eligible": True,
            "pe_supported": True,
            "qd_supported": True,
            "qprep_supported": False,
        }
    )

    result = module._renewal_water_table(candidates)

    assert not result["abstain"].any()
    assert result["inverse_cop"].notna().all()
    assert result["optimization_eligible"].all()
    assert not result["model_supported"].any()
    assert not result["t_star_model_supported"].any()


def test_renewal_water_table_marks_but_keeps_extrapolated_candidates() -> None:
    module = _module()
    candidates = pd.DataFrame(
        {
            "cycle_name": ["cycle"] * 3,
            "candidate_time": pd.date_range("2026-01-01", periods=3, freq="1min"),
            "heating_electricity_kwh": [1.0, 2.0, 3.0],
            "water_heating_kwh": [2.0, 4.0, 6.0],
            "defrost_electricity_kwh": [0.1] * 3,
            "revised_qprep_kwh": [0.1] * 3,
            "revised_qd_kwh": [0.2] * 3,
            "integration_eligible": True,
            "pe_supported": True,
            "qd_supported": True,
            "qprep_supported": [False, True, True],
        }
    )

    result = module._renewal_water_table(candidates)

    assert not result["abstain"].any()
    assert result["valid"].all()
    assert result["optimization_eligible"].all()
    assert result["model_supported"].tolist() == [False, True, True]
    assert result["inverse_cop"].notna().all()
    assert result["t_star"].notna().all()
    assert result["relative_regret"].notna().all()


def test_candidate_features_skip_failed_curves_without_valid_stable_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=61, freq="s"),
            "water_flow": 1.0,
            "water_in_temperature": 30.0,
            "water_out_temperature": 35.0,
            "power_total": 2.0,
        }
    )
    monkeypatch.setattr(module, "_load_raw", lambda _loader, _cycle: raw)
    curves = pd.DataFrame(
        {
            "cycle_name": ["valid", "failed"],
            "experiment_id": ["a", "b"],
            "candidate_time": pd.to_datetime(
                ["2026-01-01 00:01", "2026-01-01 00:01"]
            ),
            "inverse_cop": [0.5, np.nan],
        }
    )
    points = pd.DataFrame(
        {
            "cycle_name": ["valid"],
            "t_heating_stable": [pd.Timestamp("2026-01-01")],
        }
    )
    catalog = pd.DataFrame(
        {
            "cycle_name": ["valid", "failed"],
            "experiment_id": ["a", "b"],
        }
    )

    result = module.build_candidate_features(object(), curves, points, catalog)

    assert result["cycle_name"].unique().tolist() == ["valid"]
    assert result["experiment_id"].unique().tolist() == ["a"]
    assert "experiment_id_x" not in result


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
            "cycle_name": [
                "frost_cycle_000001",
                "frost_cycle_000002",
                "frost_cycle_000003",
                "frost_cycle_000004",
            ],
            "minimum_location": [
                "interior",
                "right_observed",
                "right_support_limited",
                "right_integration_limited",
            ],
            "near_opt_start_minutes": [20.0, 21.0, 22.0, 23.0],
            "near_opt_end_minutes": [30.0, 31.0, 32.0, 33.0],
            "minutes_from_stable": [25.0, 26.0, 27.0, 28.0],
            "cop_at_t_star_60s": [2.1, 2.0, 1.9, 1.8],
            "front_image_path": [float("nan")] * 4,
            "front_image_available": [False] * 4,
        }
    )

    module.plot_window_cop_rgb(overview, tmp_path / "overview")

    assert (tmp_path / "overview.png").is_file()


def test_all_cost_publications_write_report_assets_with_inverse_cop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    seen = []

    class Loader:
        dataset_root = tmp_path

        @staticmethod
        def list_cycles() -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "cycle_name": ["cycle_a", "cycle_b"],
                    "status": ["valid", "valid"],
                    "stable_heating_start": ["2026-01-01", "2026-01-01"],
                    "defrost_start": ["2026-01-01 01:00", "2026-01-01 01:00"],
                    "defrost_end": ["2026-01-01 01:05", "2026-01-01 01:05"],
                }
            )

        @staticmethod
        def get_cycle_record(cycle_name: str) -> dict[str, object]:
            return {
                "cycle_name": cycle_name,
                "assets": {"publication": f"cycles/{cycle_name}.png"},
            }

    monkeypatch.setattr(
        module,
        "render_publication_asset",
        lambda _root, record, **kwargs: seen.append((record, kwargs)),
    )
    curves = pd.DataFrame(
        {
            "cycle_name": ["cycle_b", "cycle_a", "cycle_b"],
            "candidate_time": pd.date_range("2026-01-01", periods=3, freq="1min"),
            "inverse_cop": [0.5, 0.6, 0.4],
            "cycle_cop": [2.0, 1.5, 2.5],
            "relative_regret": [0.25, 0.0, 0.0],
            "optimization_eligible": [True, True, True],
            "support_status": ["supported", "supported", "supported"],
            "minimum_location": ["interior", "interior", "right_observed"],
            "actual_preparation_time": pd.Timestamp("2026-01-01 01:00"),
            "t_RB": pd.to_datetime(
                ["2026-01-01 00:30", "2026-01-01 00:35", "2026-01-01 00:30"]
            ),
            "rb_status": ["triggered", "triggered", "triggered"],
            "trigger_type": ["Case1", "Condition1", "Case1"],
            "is_censored": [False, False, False],
        }
    )

    output = tmp_path / "report"
    cycles = module.render_all_cost_publications(Loader(), curves, output)

    assert cycles == ["cycle_a", "cycle_b"]
    assert len(seen) == 2
    assert {call[1]["output_path"] for call in seen} == {
        output / "cycle_a.svg",
        output / "cycle_b.svg",
    }
    assert all(
        set(call[1]["cost_curve"]) == {
            "candidate_time",
            "inverse_cop",
            "cycle_cop",
            "relative_regret",
            "optimization_eligible",
            "support_status",
            "minimum_location",
            "actual_preparation_time",
            "t_RB",
            "rb_status",
            "trigger_type",
        }
        for call in seen
    )


def test_render_cost_publication_matches_rb_and_cost_minimum_to_front_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    times = pd.date_range("2026-01-01", periods=5, freq="1min")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "cycle_stage": ["recovery", *(["frost_development"] * 4)],
            "cop": [2.2] * 5,
            "water_in_temperature": [30.0] * 5,
            "water_out_temperature": [35.0] * 5,
            "water_temperature_setpoint": [35.0] * 5,
        }
    )
    image_path = tmp_path / "front.jpg"
    plt.imsave(image_path, np.ones((4, 4, 3)))
    metadata = pd.DataFrame(
        {
            "camera_role": ["front", "front"],
            "file_name": ["front.jpg", "later.jpg"],
            "image_time": [times[2] + pd.Timedelta(seconds=5), times[3]],
        }
    )
    images = metadata.iloc[[0]].copy()
    images["path"] = [image_path]

    class Loader:
        dataset_root = tmp_path

        @staticmethod
        def get_cycle_record(_cycle_name: str) -> dict[str, object]:
            return {
                "cycle_name": "cycle_a",
                "status": "valid",
                "boundaries": {"stable_heating_start": times[1].isoformat()},
            }

        @staticmethod
        def load_cycle(_cycle_name: str) -> pd.DataFrame:
            return frame

        @staticmethod
        def load_cycle_images(_cycle_name: str) -> pd.DataFrame:
            return images

        @staticmethod
        def load_image_metadata(_cycle_name: str) -> pd.DataFrame:
            return metadata

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        module,
        "render_decision_publication",
        lambda _frame, _record, _curve, decision, _output, **kwargs: seen.update(
            decision_images=decision, render_options=kwargs
        ),
    )
    curves = pd.DataFrame(
        {
            "cycle_name": ["cycle_a"] * 4,
            "candidate_time": times[1:],
            "inverse_cop": [0.5, 0.4, 0.45, 0.6],
            "cycle_cop": [2.0, 2.5, 2.2, 1.7],
            "relative_regret": [0.25, 0.0, 0.125, 0.5],
            "optimization_eligible": [True] * 4,
            "support_status": ["supported"] * 4,
            "minimum_location": ["interior"] * 4,
            "actual_preparation_time": [times[-1]] * 4,
            "t_RB": [times[2]] * 4,
            "rb_status": ["triggered"] * 4,
            "trigger_type": ["Case1"] * 4,
        }
    )

    matches = module.render_cost_publication(Loader(), "cycle_a", curves, tmp_path / "cycle.svg")

    assert matches.set_index("target_type").loc["rb", "status"] == "matched"
    assert matches.set_index("target_type").loc["optimal", "status"] == "matched"
    assert seen["decision_images"]["optimal"]["target_time"] == times[2]
    assert seen["render_options"]["full_candidate_domain"] is True


def test_render_all_cost_publications_unit_heat_uses_unit_metric_and_png_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    seen: list[tuple[Path, pd.DataFrame]] = []

    class Loader:
        dataset_root = tmp_path

        @staticmethod
        def list_cycles() -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "cycle_name": ["frost_cycle_000001"],
                    "status": ["valid"],
                    "stable_heating_start": ["2026-01-01"],
                    "defrost_start": ["2026-01-01 01:00"],
                    "defrost_end": ["2026-01-01 01:05"],
                }
            )

        @staticmethod
        def get_cycle_record(cycle_name: str) -> dict[str, object]:
            return {"cycle_name": cycle_name}

    monkeypatch.setattr(
        module,
        "render_publication_asset",
        lambda _root, _record, **kwargs: seen.append(
            (kwargs["output_path"], kwargs["cost_curve"].copy())
        ),
    )
    curves = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000001"] * 2,
            "candidate_time": pd.date_range("2026-01-01 00:30", periods=2, freq="min"),
            "inverse_cop": [0.4, 0.5],
            "inverse_cop_unit": [0.6, 0.3],
            "cycle_cop": [2.5, 2.0],
            "relative_regret": [0.0, 0.25],
            "relative_regret_unit": [1.0, 0.0],
            "optimization_eligible": [True, True],
            "support_status": ["supported", "supported"],
            "minimum_location": ["interior", "interior"],
            "actual_preparation_time": pd.Timestamp("2026-01-01 01:00"),
            "t_RB": pd.Timestamp("2026-01-01 00:40"),
            "rb_status": "triggered",
            "trigger_type": "Case1",
        }
    )

    module.render_all_cost_publications(
        Loader(), curves, tmp_path / "all", unit_heat=True
    )

    assert [path.name for path, _ in seen] == ["frost_cycle_000001_J_unit.png"]
    assert seen[0][1]["inverse_cop"].tolist() == [0.6, 0.3]
    assert seen[0][1]["relative_regret"].tolist() == [1.0, 0.0]


def test_render_all_cost_publications_passes_explicit_cloud_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    import frost_analysis.dataset.images as dataset_images

    class Loader:
        dataset_root = tmp_path / "dataset"

        def load_cycle(self, _cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame()

        @staticmethod
        def list_cycles() -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "cycle_name": ["frost_cycle_000006"],
                    "status": ["valid"],
                    "stable_heating_start": ["2026-01-01"],
                    "defrost_start": ["2026-01-01 01:00"],
                    "defrost_end": ["2026-01-01 01:05"],
                }
            )

    curves = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000006"],
            "candidate_time": [pd.Timestamp("2026-01-01 00:30")],
        }
    )
    matches = pd.DataFrame(
        {
            "target_type": ["rb"],
            "status": ["physical_image_missing"],
        }
    )
    prepared = ({"cycle_name": "frost_cycle_000006"}, curves, pd.DataFrame(), {}, matches)
    monkeypatch.setattr(module, "_prepare_cost_publication", lambda *_args: prepared)
    monkeypatch.setattr(
        module, "render_decision_publication", lambda *_args, **_kwargs: None
    )
    seen: dict[str, object] = {}

    @contextmanager
    def fake_materialize(_dataset_dir: Path, _cycle_name: str, **kwargs: object):
        seen.update(kwargs)
        yield tmp_path

    monkeypatch.setattr(dataset_images, "materialize_cycle_images", fake_materialize)

    module.render_all_cost_publications(
        Loader(),
        curves,
        tmp_path / "output",
        fetch_cloud=True,
        cloud_root=tmp_path / "onedrive",
        cleanup_downloaded=True,
        minimum_free_gib=35,
    )

    assert seen == {
        "fetch_cloud": True,
        "cloud_root": tmp_path / "onedrive",
        "cleanup_downloaded": True,
        "minimum_free_gib": 35,
    }


def test_render_all_cost_publications_range_materializes_only_missing_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()

    class Loader:
        dataset_root = tmp_path / "dataset"

        def load_cycle(self, _cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame()

        @staticmethod
        def list_cycles() -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "cycle_name": ["frost_cycle_000006"],
                    "status": ["valid"],
                    "stable_heating_start": ["2026-01-01"],
                    "defrost_start": ["2026-01-01 01:00"],
                    "defrost_end": ["2026-01-01 01:05"],
                }
            )

    curves = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000006"],
            "candidate_time": [pd.Timestamp("2026-01-01 00:30")],
        }
    )
    initial_matches = pd.DataFrame(
        {
            "target_type": ["rb", "optimal"],
            "status": ["physical_image_missing", "offset_exceeds_2min"],
            "file_name": ["rb.jpg", "same.jpg"],
        }
    )
    prepared = ({"cycle_name": "frost_cycle_000006"}, curves, pd.DataFrame(), {}, initial_matches)
    prepare_calls: list[tuple[object, ...]] = []

    def fake_prepare(*args: object) -> tuple[object, ...]:
        prepare_calls.append(args)
        return prepared

    monkeypatch.setattr(module, "_prepare_cost_publication", fake_prepare)
    monkeypatch.setattr(module, "_render_prepared_cost_publication", lambda *_args: initial_matches)
    monkeypatch.setattr(
        module,
        "complete_observed_cycle_names",
        lambda *_args: ["frost_cycle_000006"],
    )
    seen: dict[str, object] = {}

    @contextmanager
    def fake_range_materialize(
        _dataset_dir: Path, _cycle_name: str, names: list[str], **kwargs: object
    ):
        seen["names"] = names
        seen.update(kwargs)
        yield tmp_path / "range"

    monkeypatch.setattr(module, "materialize_cycle_image_members", fake_range_materialize)

    module.render_all_cost_publications(
        Loader(),
        curves,
        tmp_path / "output",
        fetch_cloud=True,
        minimum_free_gib=35,
    )

    assert seen == {
        "names": ["rb.jpg"],
        "fetch_cloud": True,
        "cloud_root": None,
        "minimum_free_gib": 35,
    }
    assert len(prepare_calls) == 2
    assert prepare_calls[1][-1] == tmp_path / "range"


def test_rb_point_columns_merge_once_from_points_into_publication_curves() -> None:
    module = _module()
    curves = pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "cycle_a", "cycle_b"],
            "candidate_time": pd.date_range("2026-01-01", periods=3, freq="1min"),
        }
    )
    points = pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "cycle_b"],
            "t_RB": pd.to_datetime(["2026-01-01 00:30", None]),
            "rb_status": ["triggered", "right_censored"],
            "trigger_type": ["Case1", ""],
        }
    )

    result = module.merge_rb_points(curves, points)

    assert result.loc[result["cycle_name"].eq("cycle_a"), "t_RB"].eq(
        pd.Timestamp("2026-01-01 00:30")
    ).all()
    assert result.loc[result["cycle_name"].eq("cycle_b"), "t_RB"].isna().all()
    assert result["rb_status"].tolist() == ["triggered", "triggered", "right_censored"]


def test_inverse_cop_curves_are_exported_one_cycle_per_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    saved: list[object] = []
    original_save = module._save_figure

    def save_and_capture(fig: object, base: Path) -> None:
        saved.append(fig)
        original_save(fig, base)

    monkeypatch.setattr(module, "_save_figure", save_and_capture)

    class Loader:
        @staticmethod
        def list_cycles() -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "cycle_name": ["partial_complete", "censored", "failed"],
                    "status": ["valid", "valid", "valid"],
                    "stable_heating_start": ["2026-01-01"] * 3,
                    "defrost_start": ["2026-01-01 01:00"] * 3,
                    "defrost_end": ["2026-01-01 01:05"] * 3,
                }
            )

    curves = pd.DataFrame(
        {
            "cycle_name": ["partial_complete"] * 3 + ["censored", "failed"],
            "candidate_time": pd.to_datetime(
                [
                    "2026-01-01 00:10",
                    "2026-01-01 00:20",
                    "2026-01-01 00:30",
                    "2026-01-01 00:10",
                    "2026-01-01 00:10",
                ]
            ),
            "inverse_cop": [0.6, 0.5, 0.1, 0.4, 0.2],
            "relative_regret": [0.2, 0.0, -0.8, 0.0, 0.0],
            "optimization_eligible": [True, True, False, True, False],
            "is_censored": [False, False, False, True, False],
        }
    )

    exported = module.plot_inverse_cop_curves(Loader(), curves, tmp_path)

    assert exported == ["partial_complete"]
    assert (tmp_path / "partial_complete.svg").is_file()
    assert not (tmp_path / "censored.svg").exists()
    assert not (tmp_path / "failed.svg").exists()
    axis = saved[0].axes[0]
    assert list(axis.lines[0].get_xdata()) == [10.0, 20.0, 30.0]
    assert axis.collections[0].get_offsets()[:, 0].tolist() == [20.0]
    assert axis.collections[1].get_offsets()[:, 0].tolist() == [20.0]
    assert axis.get_xlabel() == "Time from stable heating start [min]"


def test_stale_cycle_figure_cleanup_is_exact(tmp_path: Path) -> None:
    module = _module()
    keep = "frost_cycle_000001"
    stale = "frost_cycle_000002"
    for stem in (keep, stale):
        for suffix in (".svg", ".pdf", ".png"):
            (tmp_path / f"{stem}{suffix}").write_text("generated")
    sentinel = tmp_path / "representative_publication_cost.svg"
    sentinel.write_text("keep")
    unrelated = tmp_path / "frost_cycle_000002.txt"
    unrelated.write_text("keep")

    removed = module._remove_stale_cycle_figures(tmp_path, {keep})

    assert {path.name for path in removed} == {
        f"{stale}.svg",
        f"{stale}.pdf",
        f"{stale}.png",
    }
    assert all((tmp_path / f"{keep}{suffix}").is_file() for suffix in (".svg", ".pdf", ".png"))
    assert sentinel.is_file()
    assert unrelated.is_file()
