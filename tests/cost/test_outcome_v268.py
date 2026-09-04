from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class SyntheticV268Loader:
    def __init__(self) -> None:
        self.frames: dict[str, pd.DataFrame] = {}
        records: list[dict[str, object]] = []
        self.point_rows: list[dict[str, object]] = []
        origin = pd.Timestamp("2026-02-01 00:00:00")
        for experiment_index, experiment in enumerate(("a", "b", "c", "d")):
            start = origin + pd.Timedelta(days=experiment_index)
            stable = start + pd.Timedelta(minutes=9)
            preparation = start + pd.Timedelta(minutes=15)
            defrost = preparation + pd.Timedelta(minutes=1)
            defrost_end = defrost + pd.Timedelta(minutes=1)
            current_name = f"{experiment}_event"
            recovery_name = f"{experiment}_recovery"
            records.extend(
                [
                    {
                        "cycle_name": current_name,
                        "experiment_id": experiment,
                        "experiment_date": start.date().isoformat(),
                        "start_time": start,
                        "heating_start": start,
                        "stable_heating_start": stable,
                        "defrost_preparation_start": preparation,
                        "defrost_start": defrost,
                        "defrost_end": defrost_end,
                    },
                    {
                        "cycle_name": recovery_name,
                        "experiment_id": experiment,
                        "experiment_date": start.date().isoformat(),
                        "start_time": defrost_end,
                        "heating_start": defrost_end,
                        "stable_heating_start": defrost_end + pd.Timedelta(minutes=9),
                        "defrost_preparation_start": pd.NaT,
                        "defrost_start": pd.NaT,
                        "defrost_end": pd.NaT,
                    },
                ]
            )
            self.point_rows.append(
                {
                    "cycle_name": current_name,
                    "experiment_id": experiment,
                    "valid": True,
                    "candidate_end": preparation,
                    "t_actual_preparation": preparation,
                    "t_RB": preparation,
                    "rb_status": "triggered",
                }
            )
            current_time = pd.date_range(start, defrost_end, freq="s")
            recovery_time = pd.date_range(
                defrost_end, defrost_end + pd.Timedelta(minutes=13), freq="s"
            )
            offset = float(experiment_index)

            def make_frame(
                timestamps: pd.DatetimeIndex,
                *,
                defrosting: bool,
                defrost: pd.Timestamp = defrost,
                defrost_end: pd.Timestamp = defrost_end,
                offset: float = offset,
            ) -> pd.DataFrame:
                elapsed = (timestamps - timestamps[0]).total_seconds() / 60
                in_defrost = (
                    (timestamps >= defrost) & (timestamps <= defrost_end)
                    if defrosting
                    else np.zeros(len(timestamps), dtype=bool)
                )
                return pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "power_total": 2.0 + 0.1 * offset,
                        "compressor_power": 1.0 + 0.05 * offset,
                        "heating_capacity": np.where(in_defrost, 0.0, 6.0 + 0.1 * offset),
                        "water_flow": 1.0,
                        "water_in_temperature": 35.0 + 0.1 * offset,
                        "water_out_temperature": np.where(in_defrost, 34.0, 40.0) + 0.1 * offset,
                        "coil_temperature": -8.0 - offset - 0.01 * elapsed,
                        "evaporating_pressure": 0.45 - 0.01 * offset - 0.0002 * elapsed,
                        "water_temperature_setpoint": 50.0,
                        "ambient_temperature": 2.0 - 0.2 * offset,
                        "compressor_frequency": 70.0 + 2 * offset,
                    }
                )

            self.frames[current_name] = make_frame(current_time, defrosting=True)
            self.frames[recovery_name] = make_frame(recovery_time, defrosting=False)
        self.catalog = pd.DataFrame(records)

    def list_cycles(self) -> pd.DataFrame:
        return self.catalog.copy()

    def load_cycle_original(self, cycle_name: str, *, columns=None) -> pd.DataFrame:
        frame = self.frames[cycle_name].copy()
        return frame if columns is None else frame[list(columns)].copy()


def test_build_v268_is_cross_fitted_and_exports_all_cost_views() -> None:
    from frost_analysis.cost.outcome import build_v268_table

    loader = SyntheticV268Loader()
    points = pd.DataFrame(loader.point_rows)
    table, artifacts = build_v268_table(points, loader, bootstrap_replicates=3, n_jobs=2)

    assert set(table["algorithm"]) == {"v2.6.8"}
    assert set(table["cycle_name"]) == set(points["cycle_name"])
    assert table.groupby("cycle_name").size().eq(6).all()
    assert table[["J_model", "J_ts_model", "J_rr_model"]].notna().all().all()
    assert table["J_model"].equals(table["J"])
    assert table["E_T_hat_kwh"].notna().all()
    assert table["Q_T_hat_kwh"].notna().all()
    assert table["model_training_experiment_ids"].notna().all()
    leaked = table.apply(
        lambda row: row["experiment_id"] in row["model_training_experiment_ids"].split(","),
        axis=1,
    )
    assert not leaked.any()

    validation = artifacts["validation"]
    assert set(validation["model_name"]) == {
        "mean_baseline",
        "static_5",
        "physical_static_6",
        "dynamic_8",
    }
    assert validation.groupby("cycle_name")["model_name"].nunique().eq(4).all()
    assert validation[["E_T_observed_kwh", "Q_T_observed_kwh"]].notna().all().all()
    assert (
        validation.loc[validation["model_name"].eq("dynamic_8"), "E_prep_prediction_kwh"]
        .notna()
        .all()
    )
    assert (
        validation.loc[validation["model_name"].eq("dynamic_8"), "Q_D_prediction_kwh"].lt(0).all()
    )

    bootstrap = artifacts["bootstrap"]
    assert set(bootstrap["cycle_name"]) == set(points["cycle_name"])
    assert bootstrap["repeat_count"].eq(3).all()


def test_build_v27_reuses_v268_candidates_and_keeps_native_metric_directions(
    tmp_path: Path,
) -> None:
    from frost_analysis.cost.evaluation import build_v27_tables

    loader = SyntheticV268Loader()
    points = pd.DataFrame(loader.point_rows)

    trajectory = tmp_path / "trajectory.parquet"
    tables, artifacts = build_v27_tables(
        points,
        loader,
        bootstrap_replicates=2,
        bootstrap_trajectory=trajectory,
        bootstrap_final_only=True,
        n_jobs=2,
        candidate_step_seconds=10,
    )

    assert set(tables) == {"v2.7.0", "v2.7.1", "v2.7.2", "v2.7.3", "v2.7.4"}
    grids = {
        metric: tuple(
            pd.to_datetime(
                table.drop_duplicates(["cycle_name", "candidate_time"])["candidate_time"]
            )
        )
        for metric, table in tables.items()
    }
    assert len(set(grids.values())) == 1
    grid = tables["v2.7.0"].drop_duplicates(["cycle_name", "candidate_time"])
    candidate_times = pd.to_datetime(grid["candidate_time"])
    cadence = candidate_times.groupby(grid["cycle_name"]).diff()
    assert cadence.dropna().eq(pd.Timedelta(seconds=10)).all()
    assert grid.groupby("cycle_name").size().eq(31).all()
    assert set(tables["v2.7.0"]["metric_id"]) == {"eta_e_cyc", "cop_e"}
    assert set(tables["v2.7.1"]["metric_id"]) == {
        "epsilon_hl",
        "epsilon_hl_t0_proxy",
    }
    assert set(tables["v2.7.4"]["metric_id"]) == {"eta_h_cyc"}
    assert tables["v2.7.4"]["optimization_direction"].eq("max").all()
    eta_h = tables["v2.7.4"].loc[lambda frame: frame["metric_id"].eq("eta_h_cyc")]
    np.testing.assert_allclose(eta_h["epsilon_hl_closed"], 1 - eta_h["objective_value"])
    assert tables["v2.7.0"]["optimization_direction"].eq("max").all()
    assert tables["v2.7.1"]["optimization_direction"].eq("min").all()
    assert tables["v2.7.2"]["optimization_direction"].eq("max").all()
    assert tables["v2.7.3"]["optimization_direction"].eq("min").all()
    assert tables["v2.7.0"]["cycle_elapsed_minutes"].ge(10).all()
    assert tables["v2.7.0"]["source_idea"].notna().all()
    assert tables["v2.7.0"]["project_definition"].notna().all()
    assert {
        "healthy_water_heat_alpha",
        "healthy_compressor_power_alpha",
        "E_comp_T_alpha",
        "event_duration_alpha",
        "L_T_dynamic_alpha",
        "L_T_t0_alpha",
        "healthy_water_heat_support_threshold",
        "E_comp_T_model_training_experiment_ids",
        "v27_model_provenance",
    } <= set(tables["v2.7.0"])
    assert "diagnostic_minimum" not in tables["v2.7.3"]
    assert "legacy_v268_diagnostic_minimum" in tables["v2.7.3"]
    assert "raw_t_star" not in tables["v2.7.3"]
    assert "J" not in tables["v2.7.3"]

    common = tables["v2.7.2"].loc[lambda frame: frame["objective_value"].notna()]
    assert np.allclose(common["objective_value"], 1 / common["J_rr_model"])
    assert pd.to_datetime(common["t_star"], errors="coerce").reset_index(drop=True).equals(
        pd.to_datetime(common["rr_diagnostic_minimum"], errors="coerce").reset_index(drop=True)
    )

    leaked = tables["v2.7.0"].apply(
        lambda row: row["experiment_id"]
        in str(row["model_training_experiment_ids"]).split(","),
        axis=1,
    )
    assert not leaked.any()
    assert {
        "E_comp_T_observed_kwh",
        "event_duration_observed_minutes",
        "L_T_dynamic_observed_kwh",
        "L_T_dynamic_reference_derived_kwh",
        "L_T_dynamic_observed_provenance",
        "L_T_dynamic_observed_is_direct_measurement",
        "L_T_t0_observed_kwh",
    } <= set(artifacts["validation"])
    assert artifacts["validation"]["L_T_dynamic_observed_provenance"].eq(
        "cross_fitted_healthy_water_heat_reference_minus_observed_Q_T"
    ).all()
    assert not artifacts["validation"]["L_T_dynamic_observed_is_direct_measurement"].any()
    assert set(artifacts["validation"]["model_name"]) == {
        "mean_baseline",
        "static_5",
        "physical_static_6",
        "dynamic_8",
    }
    assert artifacts["validation"][
        ["J_w_observed", "J_w_prediction", "J_w_residual"]
    ].notna().all().all()
    assert artifacts["validation"]["event_duration_prediction_minutes"].notna().all()
    assert artifacts["bootstrap"]["repeat_count"].eq(2).all()
    assert artifacts["bootstrap"]["bootstrap_method"].eq(
        "experiment_refit_excluding_target"
    ).all()
    bootstrap_curves = pd.read_parquet(trajectory)
    assert set(bootstrap_curves["replicate_id"]) == {0, 1}
    assert {
        "cop_cyc_evt",
        "eta_h_cyc",
        "eta_e_cyc",
        "cop_cyc_evt_eligible",
        "eta_h_cyc_eligible",
        "eta_e_cyc_eligible",
        "support_Q_T",
        "support_Qw0",
        "support_D_T",
        "support_Pcomp0",
        "support_E_comp_T",
        "eta_h_cyc_base_eligible",
        "eta_e_cyc_base_eligible",
    } <= set(bootstrap_curves)
    assert set(artifacts["bootstrap_draws"]["replicate_id"]) == {0, 1}
    assert {
        "heldout_experiment_id",
        "source_experiment_id",
        "draw_count",
    } <= set(artifacts["bootstrap_draws"])
    assert {
        "strict_tan_qhc1",
        "wang_nominal_capacity",
        "da_silva_frost_mass_efficiency",
        "project_two_anchor",
    } <= set(artifacts["identifiability"]["audit_id"])


def test_v27_validation_keeps_event_experiments_without_candidate_curves() -> None:
    from frost_analysis.cost.evaluation import build_v27_tables

    loader = SyntheticV268Loader()
    points = pd.DataFrame(loader.point_rows[:-1])

    tables, artifacts = build_v27_tables(points, loader, bootstrap_replicates=0)

    assert "d" not in set(tables["v2.7.0"]["experiment_id"])
    assert "d" in set(artifacts["validation"]["experiment_id"])


def test_v27_t0_proxy_reference_integrates_from_proxy_start_to_candidate() -> None:
    from frost_analysis.cost.evaluation import build_v27_tables

    loader = SyntheticV268Loader()
    points = pd.DataFrame(loader.point_rows)
    tables, _ = build_v27_tables(points, loader, bootstrap_replicates=0)

    row = tables["v2.7.1"].loc[
        lambda frame: frame["metric_id"].eq("epsilon_hl_t0_proxy")
        & frame["cycle_name"].eq("a_event")
        & frame["cycle_elapsed_minutes"].eq(10.0)
    ].iloc[0]
    proxy = float(row["t0_proxy_water_heat_kw"])

    assert row["t0_proxy_candidate_valid"]
    assert row["t0_proxy_healthy_heating_kwh"] == pytest.approx(proxy * 10 / 60)


def test_v268_event_table_uses_fixed_nine_minute_recovery_and_raw_water_heat() -> None:
    from frost_analysis.cost.outcome import build_event_table

    loader = SyntheticV268Loader()
    events = build_event_table(loader)
    valid = events.loc[events["event_valid"]]

    assert len(valid) == 4
    assert valid["recovery_duration_minutes"].eq(9.0).all()
    assert valid["E_prep_kwh"].gt(0).all()
    assert valid["Q_D_kwh"].lt(0).all()
    assert valid["Q_T_kwh"].eq(valid["Q_prep_kwh"] + valid["Q_D_kwh"] + valid["Q_R_kwh"]).all()
    assert valid["pre_action_window_valid"].all()


def _synthetic_event_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp]]:
    preparation = pd.Timestamp("2026-01-01 00:00:00")
    defrost = preparation + pd.Timedelta(minutes=1)
    defrost_end = defrost + pd.Timedelta(minutes=1)
    recovery_end = defrost_end + pd.Timedelta(minutes=9)

    current_time = pd.date_range(preparation, defrost_end, freq="10s")
    current = pd.DataFrame(
        {
            "timestamp": current_time,
            "power_total": 2.0,
            "compressor_power": 1.0,
            "water_flow": 1.0,
            "water_in_temperature": 30.0,
            "water_out_temperature": np.where(current_time < defrost, 31.0, 29.0),
        }
    )
    recovery_time = pd.date_range(defrost_end, recovery_end, freq="10s")
    recovery = pd.DataFrame(
        {
            "timestamp": recovery_time,
            "power_total": 2.0,
            "compressor_power": 1.0,
            "water_flow": 1.0,
            "water_in_temperature": 30.0,
            "water_out_temperature": 31.0,
        }
    )
    return (
        current,
        recovery,
        {
            "preparation": preparation,
            "defrost": defrost,
            "defrost_end": defrost_end,
            "recovery_end": recovery_end,
        },
    )


def test_event_outcomes_include_preparation_energy_and_preserve_signed_qd() -> None:
    from frost_analysis.cost.outcome import event_outcomes

    current, recovery, boundary = _synthetic_event_frames()
    result = event_outcomes(
        current,
        recovery,
        preparation_start=boundary["preparation"],
        defrost_start=boundary["defrost"],
        defrost_end=boundary["defrost_end"],
        recovery_end=boundary["recovery_end"],
    )

    assert result["E_prep_kwh"] > 0
    assert result["E_comp_T_kwh"] == pytest.approx(11.0 / 60)
    assert result["event_duration_minutes"] == pytest.approx(11.0)
    assert result["Q_D_kwh"] < 0
    assert result["E_T_kwh"] == result["E_prep_kwh"] + result["E_D_kwh"] + result["E_R_kwh"]
    assert result["Q_T_kwh"] == result["Q_prep_kwh"] + result["Q_D_kwh"] + result["Q_R_kwh"]
    assert result["E_T_kwh"] == pytest.approx(22 / 60)
    assert result["phase_interval_convention"] == "[start,end)"
    assert "right_boundary_sample_excluded" in result["integral_sampling_convention"]


def test_event_outcomes_reject_non_partitioned_phase_intervals() -> None:
    from frost_analysis.cost.outcome import event_outcomes

    current, recovery, boundary = _synthetic_event_frames()

    result = event_outcomes(
        current,
        recovery,
        preparation_start=boundary["preparation"],
        defrost_start=boundary["defrost_end"] + pd.Timedelta(seconds=10),
        defrost_end=boundary["defrost_end"],
        recovery_end=boundary["recovery_end"],
    )

    assert not result["phase_partition_valid"]
    assert not result["event_valid"]


def test_experiment_weights_preserve_sample_scale_and_equal_group_mass() -> None:
    from frost_analysis.cost.outcome import experiment_weights

    groups = pd.Series(["a", "a", "a", "b", "c", "c"])
    weights = experiment_weights(groups)

    assert weights.sum() == pytest.approx(len(groups))
    mass = pd.Series(weights).groupby(groups).sum()
    np.testing.assert_allclose(mass, np.repeat(len(groups) / 3, 3))


def test_weighted_scaler_and_ridge_use_the_same_weights() -> None:
    from frost_analysis.cost.outcome import experiment_weights, fit_weighted_ridge

    frame = pd.DataFrame(
        {
            "experiment_id": ["a", "a", "a", "b"],
            "x": [0.0, 0.0, 0.0, 10.0],
            "target": [0.0, 0.0, 0.0, 10.0],
        }
    )
    model = fit_weighted_ridge(frame, ("x",), "target", alpha=1.0)
    expected = np.average(frame["x"], weights=experiment_weights(frame["experiment_id"]))

    assert model.scaler.mean_[0] == pytest.approx(expected)
    assert model.sample_weight_sum == pytest.approx(len(frame))


def test_heldout_targets_cannot_change_fold_model_or_alpha() -> None:
    from frost_analysis.cost.outcome import fit_outcome_fold

    rows = []
    for experiment, offset in (("heldout", 0.0), ("a", 1.0), ("b", 2.0), ("c", 3.0)):
        for index in range(3):
            rows.append(
                {
                    "experiment_id": experiment,
                    "x": offset + index,
                    "target": 2 * (offset + index) + 1,
                }
            )
    events = pd.DataFrame(rows)
    before = fit_outcome_fold(events, "heldout", ("x",), "target")
    changed = events.copy()
    changed.loc[changed["experiment_id"].eq("heldout"), "target"] += 10000
    after = fit_outcome_fold(changed, "heldout", ("x",), "target")

    assert before.alpha == after.alpha
    np.testing.assert_allclose(before.scaler.mean_, after.scaler.mean_)
    np.testing.assert_allclose(before.ridge.coef_, after.ridge.coef_)
    assert before.support_threshold == pytest.approx(after.support_threshold)




def test_short_support_run_keeps_full_curve_but_has_no_diagnostic_minimum() -> None:
    from frost_analysis.cost.outcome import finalize_v268_curve

    times = pd.date_range("2026-01-01", periods=5, freq="min")
    curve = pd.DataFrame(
        {
            "candidate_time": times,
            "J_model": [0.30, 0.20, 0.10, 0.15, 0.25],
            "supported": True,
            "pre_action_window_valid": True,
            "physical_valid": True,
        }
    )
    result = finalize_v268_curve(curve)

    assert result["J_model"].notna().all()
    assert not result["optimization_eligible"].any()
    assert result["diagnostic_minimum"].isna().all()


def test_v268_minimum_requires_measurement_eligible_candidate() -> None:
    from frost_analysis.cost.outcome import finalize_v268_curve

    start = pd.Timestamp("2026-01-01")
    curve = pd.DataFrame(
        {
            "candidate_time": pd.date_range(start, periods=8, freq="min"),
            "J_model": [0.1, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
            "supported": True,
            "pre_action_window_valid": True,
            "measurement_eligible": [False, True, True, True, True, True, True, True],
            "physical_valid": True,
        }
    )

    result = finalize_v268_curve(curve)

    assert not result.loc[0, "optimization_eligible"]
    assert pd.Timestamp(result["t_star"].iloc[0]) == start + pd.Timedelta(minutes=7)


def test_five_minute_support_run_selects_minimum_without_clipping_curve() -> None:
    from frost_analysis.cost.outcome import finalize_v268_curve

    times = pd.date_range("2026-01-01", periods=8, freq="min")
    curve = pd.DataFrame(
        {
            "candidate_time": times,
            "J_model": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.45, 0.5],
            "supported": [False, True, True, True, True, True, True, False],
            "pre_action_window_valid": True,
            "physical_valid": True,
        }
    )
    result = finalize_v268_curve(curve)

    assert result["J_model"].notna().all()
    assert result["optimization_eligible"].sum() == 6
    assert result["diagnostic_minimum"].iloc[0] == times[5]
    assert result["basin_1pct_width_minutes"].iloc[0] == 0
    assert result["basin_5pct_width_minutes"].iloc[0] == 0


def test_selected_dispatches_v268_and_keeps_export_artifacts(monkeypatch) -> None:
    import frost_analysis.cost.outcome as outcome
    from frost_analysis.cost.selected import build_cost_function_table

    expected = pd.DataFrame({"algorithm": ["v2.6.8"]})
    artifacts = {
        "validation": pd.DataFrame({"model_name": ["dynamic_8"]}),
        "bootstrap": pd.DataFrame({"repeat_count": [200]}),
        "events": pd.DataFrame({"event_valid": [True]}),
    }
    captured = {}

    def fake_build(points, loader, **kwargs):
        captured.update(kwargs)
        return expected.copy(), artifacts

    monkeypatch.setattr(outcome, "build_v268_table", fake_build)

    result = build_cost_function_table(
        pd.DataFrame(),
        pd.DataFrame(),
        object(),
        "v2.6.8",
        bootstrap_replicates=0,
    )

    pd.testing.assert_frame_equal(result, expected)
    assert result.attrs == artifacts
    assert captured["bootstrap_replicates"] == 0


def test_v268_writer_exports_only_the_three_public_tables(tmp_path) -> None:
    import importlib.util

    path = Path(__file__).parents[2] / "scripts/cost/build.py"
    spec = importlib.util.spec_from_file_location("cost_build_v268", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = pd.DataFrame({"cycle_name": ["cycle"]})
    table.attrs.update(
        {
            "validation": pd.DataFrame({"model_name": ["dynamic_8"]}),
            "bootstrap": pd.DataFrame({"repeat_count": [200]}),
            "events": pd.DataFrame({"event_valid": [True]}),
        }
    )

    module._write_v268_artifacts(table, tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "cost_function_v2.6.8_bootstrap.csv",
        "cost_function_v2.6.8_validation.csv",
    ]


def test_v27_writer_exports_shared_validation_identifiability_and_bootstrap(tmp_path) -> None:
    import importlib.util

    path = Path(__file__).parents[2] / "scripts/cost/build.py"
    spec = importlib.util.spec_from_file_location("cost_build_v27", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    artifacts = {
        "validation": pd.DataFrame({"cycle_name": ["cycle"]}),
        "identifiability": pd.DataFrame({"audit_id": ["strict_tan_qhc1"]}),
        "bootstrap": pd.DataFrame({"repeat_count": [200]}),
        "healthy_samples": pd.DataFrame({"sample": [1]}),
    }

    module._write_v27_artifacts(artifacts, tmp_path)

    assert {"v2.7.0", "v2.7.1", "v2.7.2", "v2.7.3"} <= set(module.ALGORITHM_CHOICES)
    assert sorted(file.name for file in tmp_path.iterdir()) == [
        "cost_function_v2.7_bootstrap.csv",
        "cost_function_v2.7_identifiability.csv",
        "cost_function_v2.7_validation.csv",
    ]


def test_event_table_retains_real_events_with_missing_boundaries_as_exclusions() -> None:
    from frost_analysis.cost.outcome import build_event_table

    loader = SyntheticV268Loader()
    missing = loader.catalog.iloc[0].copy()
    missing["cycle_name"] = "missing_defrost_start"
    missing["start_time"] = pd.Timestamp("2026-01-31")
    missing["heating_start"] = pd.Timestamp("2026-01-31")
    missing["defrost_preparation_start"] = pd.Timestamp("2026-01-31 00:15:00")
    missing["defrost_start"] = pd.NaT
    missing["defrost_end"] = pd.NaT
    loader.catalog = pd.concat([loader.catalog, missing.to_frame().T], ignore_index=True)

    events = build_event_table(loader)
    excluded = events.loc[events["cycle_name"].eq("missing_defrost_start")].iloc[0]

    assert not bool(excluded["event_valid"])
    assert excluded["event_invalid_reason"] == "missing_defrost_start;missing_defrost_end"


def test_validation_csv_retains_excluded_events_without_training_on_them() -> None:
    from frost_analysis.cost.outcome import build_event_table, build_validation_table

    loader = SyntheticV268Loader()
    events = build_event_table(loader)
    excluded = events.iloc[[0]].copy()
    excluded["cycle_name"] = "excluded"
    excluded["event_valid"] = False
    excluded["event_invalid_reason"] = "Q_R_continuous_gap"
    validation, _ = build_validation_table(pd.concat([events, excluded], ignore_index=True))

    excluded_rows = validation.loc[validation["cycle_name"].eq("excluded")]
    assert len(excluded_rows) == 1
    assert excluded_rows.iloc[0]["model_name"] == "excluded_event"
    assert excluded_rows.iloc[0]["event_invalid_reason"] == "Q_R_continuous_gap"


def test_sorted_time_slice_returns_only_the_boundary_window() -> None:
    from frost_analysis.cost.outcome import _sorted_time_slice

    timestamps = pd.date_range("2026-01-01", periods=100_000, freq="s")
    frame = pd.DataFrame({"timestamp": timestamps, "value": np.arange(len(timestamps))})
    start = timestamps[50_000]
    end = timestamps[50_060]

    result = _sorted_time_slice(frame, start, end)

    assert result["timestamp"].iloc[0] == start
    assert result["timestamp"].iloc[-1] == end
    assert len(result) == 61


def test_candidate_integral_table_matches_individual_raw_window_integrals() -> None:
    from frost_analysis.cost.outcome import _candidate_integral_table, _window_audit

    timestamps = pd.date_range("2026-01-01", periods=601, freq="s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "power_total": 2.0 + np.sin(np.arange(len(timestamps)) / 60),
        }
    )
    candidates = pd.DatetimeIndex([timestamps[120], timestamps[300], timestamps[600]])

    batched = _candidate_integral_table(frame, timestamps[0], candidates, "power_total")
    individual = [
        _window_audit(frame, timestamps[0], candidate, "power_total") for candidate in candidates
    ]

    np.testing.assert_allclose(batched["energy"], [row["energy"] for row in individual])
    np.testing.assert_allclose(batched["coverage"], [row["coverage"] for row in individual])
    assert batched["valid"].tolist() == [row["valid"] for row in individual]


def test_half_open_integrals_exclude_right_boundary_sample_from_trapezoid() -> None:
    from frost_analysis.cost.outcome import _candidate_integral_table, _window_audit

    start = pd.Timestamp("2026-01-01")
    end = start + pd.Timedelta(seconds=60)
    timestamps = pd.date_range(start, periods=61, freq="s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "power_total": np.r_[np.ones(60), 100.0],
        }
    )

    individual = _window_audit(frame, start, end, "power_total")
    batched = _candidate_integral_table(frame, start, [end], "power_total").iloc[0]

    # The post-boundary 100-kW sample at ``end`` must not enter the phase
    # integral.  The last left-side 1-kW sample supports the final second by
    # an explicit zero-order hold, so a complete 1-Hz [start, end) interval
    # still has 60 s of support rather than losing one sampling interval.
    expected = 60.0 / 3600.0
    assert individual["energy"] == pytest.approx(expected)
    assert batched["energy"] == pytest.approx(expected)
    assert individual["coverage"] == pytest.approx(1.0)
    assert batched["coverage"] == pytest.approx(1.0)


def test_half_open_integral_does_not_hide_a_trailing_observation_gap() -> None:
    from frost_analysis.cost.outcome import _window_audit

    start = pd.Timestamp("2026-01-01")
    # The last left-side observation is 31 s before ``end``.  A 1-Hz cadence
    # may support only one final second; the remaining tail is uncovered and
    # must fail both the coverage and maximum-gap quality audit.
    timestamps = pd.date_range(start, periods=30, freq="s")
    frame = pd.DataFrame({"timestamp": timestamps, "power_total": 1.0})
    end = start + pd.Timedelta(seconds=60)

    result = _window_audit(frame, start, end, "power_total")

    assert result["coverage"] < 0.95
    assert result["maximum_gap_seconds"] == pytest.approx(31.0)
    assert not result["valid"]
    assert "zero_order_hold" in result["integral_sampling_convention"]


def test_pre_action_feature_table_is_strictly_before_each_candidate() -> None:
    from frost_analysis.cost.outcome import _pre_action_feature_table

    timestamps = pd.date_range("2026-01-01", periods=601, freq="s")
    elapsed_minutes = np.arange(len(timestamps)) / 60
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "water_in_temperature": 35.0,
            "water_out_temperature": 40.0,
            "coil_temperature": -8.0,
            "evaporating_pressure": 0.4 + 0.01 * elapsed_minutes,
            "water_temperature_setpoint": 50.0,
            "ambient_temperature": 2.0,
            "compressor_frequency": 70.0,
        }
    )
    frame.loc[frame["timestamp"].eq(timestamps[600]), "water_in_temperature"] = 999.0

    result = _pre_action_feature_table(frame, [timestamps[300], timestamps[600]], timestamps[0])

    assert result["water_in_temperature"].tolist() == [35.0, 35.0]
    np.testing.assert_allclose(result["evaporating_pressure_slope_5m"], [0.01, 0.01])
    assert result["pre_action_window_valid"].all()


def test_cross_objective_regret_never_changes_a_metrics_own_decision() -> None:
    from frost_analysis.cost.benchmark import FINAL_METRICS, cross_objective_regret

    times = pd.date_range("2026-01-01", periods=3, freq="min")
    tables = {}
    for metric, optimum in zip(FINAL_METRICS, (2, 1, 0), strict=True):
        objective = np.array([1.0, 2.0, 3.0]) if optimum == 2 else np.array([3.0, 2.0, 1.0])
        if optimum == 1:
            objective = np.array([1.0, 3.0, 2.0])
        tables[metric] = pd.DataFrame(
            {
                "cycle_name": "cycle_001",
                "candidate_time": times,
                "objective_value": objective,
                "optimization_eligible": [True, True, metric != "eta_h_cyc"],
                "t_star": times[optimum],
                "near_optimal_1pct": [False, False, True],
            }
        )

    result = cross_objective_regret(tables, basin_percents=(1,))
    cop_w1 = result.loc[
        result["selector_metric"].eq("cop_cyc_evt")
        & result["decision_type"].eq("latest_W1")
    ]

    assert cop_w1["decision_time"].eq(times[2]).all()
    assert set(cop_w1["target_metric"]) == {"cop_cyc_evt", "eta_e_cyc"}
