from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = (
    "water_in_temperature",
    "water_out_temperature",
    "coil_temperature",
    "evaporating_pressure",
    "water_temperature_setpoint",
)


def _events() -> pd.DataFrame:
    rows = []
    for experiment, offset in (("heldout", 0.0), ("a", 1.0), ("b", 2.0)):
        for event in range(2):
            x = offset + event / 10
            rows.append(
                {
                    "cycle_name": f"{experiment}_{event}",
                    "experiment_id": experiment,
                    **{feature: x + index for index, feature in enumerate(FEATURES)},
                    "E_T_observed_kwh": 1 + x,
                    "Q_T_observed_kwh": 2 + 2 * x,
                }
            )
    return pd.DataFrame(rows)


def test_frozen_ticket_formula() -> None:
    from frost_analysis.cost.ticket import ticket_cost

    assert ticket_cost(2.0, 8.0, 1.0, 4.0) == 0.25
    assert np.isnan(ticket_cost(2.0, 8.0, 1.0, -8.0))


def test_loeo_predictions_ignore_heldout_terminal_targets() -> None:
    from frost_analysis.cost.ticket import fit_ticket_fold

    events = _events()
    before = fit_ticket_fold(events, "heldout")
    events.loc[events.experiment_id.eq("heldout"), "E_T_observed_kwh"] += 1000
    events.loc[events.experiment_id.eq("heldout"), "Q_T_observed_kwh"] -= 1000
    after = fit_ticket_fold(events, "heldout")

    row = events.loc[events.experiment_id.eq("heldout"), FEATURES].iloc[[0]]
    for target in ("E_T", "Q_T"):
        np.testing.assert_allclose(
            before[target]["model"].predict(row),
            after[target]["model"].predict(row),
            atol=1e-12,
        )


def test_joint_support_and_status_priority() -> None:
    from frost_analysis.cost.ticket import classify_cycle, prediction_in_support

    support = {feature: (0.0, 1.0) for feature in FEATURES}
    assert prediction_in_support(pd.Series(dict.fromkeys(FEATURES, 0.5)), support)
    assert not prediction_in_support(pd.Series(dict.fromkeys(FEATURES, 1.1)), support)

    curve = pd.DataFrame(
        {
            "measurement_eligible": [True, True, True],
            "model_supported": [False, True, True],
            "optimization_eligible": [False, True, True],
        }
    )
    assert classify_cycle(curve, 1) == "model_support_limited"
    curve.loc[0, "measurement_eligible"] = False
    assert classify_cycle(curve, 1) == "measurement_limited"
    curve.loc[0, ["measurement_eligible", "model_supported"]] = True
    assert classify_cycle(curve, 2) == "right_censored"


def test_single_eligible_is_not_a_boundary_and_component_blocks_right_censor() -> None:
    from frost_analysis.cost.ticket import classify_cycle

    curve = pd.DataFrame(
        {
            "measurement_eligible": [True, True, True],
            "component_eligible": [True, True, True],
            "model_supported": [False, True, False],
            "optimization_eligible": [False, True, False],
        }
    )
    assert classify_cycle(curve, None) == "model_support_limited"

    curve["model_supported"] = True
    curve["component_eligible"] = [False, True, True]
    curve["optimization_eligible"] = [False, True, True]
    assert classify_cycle(curve, 2) == "unidentifiable_component"


def test_finalize_selects_earliest_exact_tie_and_never_decides() -> None:
    from frost_analysis.cost.ticket import finalize_curve

    times = pd.date_range("2026-01-01", periods=3, freq="min")
    curve = pd.DataFrame(
        {
            "candidate_time": times,
            "J": [0.2, 0.2, 0.3],
            "measurement_eligible": True,
            "model_supported": True,
            "optimization_eligible": True,
        }
    )
    result, status = finalize_curve(curve)

    assert result.raw_t_star.iloc[0] == times[0]
    assert status == "left_boundary_limited"
    assert result.recommended_time.isna().all()
    assert not result.hard_label_eligible.any()
    assert result.decision_status.eq("abstain_v267_identification_only").all()


def test_oracle_columns_cannot_change_main_curve() -> None:
    from frost_analysis.cost.ticket import finalize_curve

    curve = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01", periods=3, freq="min"),
            "J": [0.3, 0.2, 0.4],
            "measurement_eligible": True,
            "model_supported": True,
            "optimization_eligible": True,
            "oracle_J": [0.1, 0.9, 0.8],
        }
    )
    before, before_status = finalize_curve(curve)
    curve["oracle_J"] = [9.0, 8.0, 0.1]
    after, after_status = finalize_curve(curve)

    assert before_status == after_status
    pd.testing.assert_series_equal(before.raw_t_star, after.raw_t_star)
    pd.testing.assert_series_equal(before.J, after.J)


def test_bootstrap_uses_requested_repeat_count_and_seed() -> None:
    from frost_analysis.cost.ticket import bootstrap_audit

    events = _events()
    candidates = pd.DataFrame(
        {
            "cycle_name": ["heldout_0"] * 3,
            "experiment_id": ["heldout"] * 3,
            "cycle_status": ["identified_curve"] * 3,
            "candidate_time": pd.date_range("2026-01-01", periods=3, freq="min"),
            **{feature: [0.0, 0.05, 0.1] for feature in FEATURES},
            "heating_electricity_kwh": [1.0, 1.1, 1.2],
            "unit_heating_kwh": [5.0, 5.2, 5.4],
            "measurement_eligible": True,
            "basin_5pct_start": pd.Timestamp("2026-01-01"),
            "basin_5pct_end": pd.Timestamp("2026-01-01 00:02:00"),
        }
    )
    first = bootstrap_audit(candidates, events, replicates=3, seed=267)
    second = bootstrap_audit(candidates, events, replicates=3, seed=267)

    pd.testing.assert_frame_equal(first, second)
    assert first.repeat_count.iloc[0] == 3


def test_q_target_survives_invalid_e_and_uses_raw_preparation_heat() -> None:
    from frost_analysis.cost.ticket import _ticket_events

    preparation = pd.Timestamp("2026-01-01 00:01:00")
    timestamps = pd.date_range(preparation - pd.Timedelta(seconds=60), periods=121, freq="s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "power_total": 2.0,
            "heating_capacity": 3.0,
            **{feature: 1.0 for feature in FEATURES},
        }
    )

    class Loader:
        def list_cycles(self):
            return pd.DataFrame(
                {
                    "cycle_name": ["cycle"],
                    "experiment_id": ["experiment"],
                    "defrost_preparation_start": [preparation],
                    "defrost_start": [preparation + pd.Timedelta(seconds=60)],
                }
            )

        def load_cycle_original(self, cycle_name, *, columns=None):
            assert cycle_name == "cycle"
            return frame.copy() if columns is None else frame[columns].copy()

    sources = {
        "preparation": pd.DataFrame(
            {
                "cycle_name": ["cycle"],
                "experiment_id": ["experiment"],
                "defrost_preparation_start": [preparation],
                "defrost_start": [preparation + pd.Timedelta(seconds=10)],
                "preparation_signed_heat_kwh": [999.0],
            }
        ),
        "tickets": pd.DataFrame(
            {
                "cycle_name": ["cycle"],
                "experiment_id": ["experiment"],
                "valid": [False],
                "defrost_electricity_kwh": [0.01],
                "defrost_electricity_coverage": [0.0],
                "defrost_absorbed_heat_kwh": [0.02],
                "defrost_signed_heat_coverage": [1.0],
            }
        ),
        "recovery": pd.DataFrame(
            {
                "cycle_name": ["cycle"],
                "experiment_id": ["experiment"],
                "recovery_valid": [True],
                "recovery_electricity_kwh": [0.04],
                "recovery_electricity_coverage": [1.0],
                "recovery_water_heat_kwh": [0.03],
                "recovery_water_heat_coverage": [1.0],
            }
        ),
    }

    event = _ticket_events(Loader(), sources, {}).iloc[0]

    assert np.isnan(event.E_T_observed_kwh)
    assert event.Qprep_observed_kwh == 0.05
    assert event.Q_T_observed_kwh == 0.06


def test_catalog_restores_cycles_excluded_by_legacy_preparation_table() -> None:
    from pathlib import Path

    from dataloader.loader import DatasetLoader
    from frost_analysis.cost.ticket import _default_sources, _ticket_events

    events = _ticket_events(DatasetLoader(Path("dataset")), _default_sources(), {})
    restored = {
        "frost_cycle_000051",
        "frost_cycle_000056",
        "frost_cycle_000057",
        "frost_cycle_000088",
        "frost_cycle_000091",
        "frost_cycle_000095",
    }

    assert not events["cycle_name"].duplicated().any()
    assert restored <= set(events.loc[events["Q_T_observed_kwh"].notna(), "cycle_name"])


def test_oracle_resets_all_main_model_state() -> None:
    from frost_analysis.cost.ticket import _oracle_curve

    curve = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01", periods=3, freq="min"),
            "heating_electricity_kwh": [1.0, 1.1, 1.2],
            "unit_heating_kwh": [5.0, 5.2, 5.4],
            "measurement_eligible": True,
            "model_supported": False,
            "component_eligible": False,
            "optimization_eligible": False,
            "J": np.nan,
            "E_T_hat_kwh": 999.0,
            "Q_T_hat_kwh": 999.0,
            "E_T_model_supported": False,
            "Q_T_model_supported": False,
            "E_T_model_provenance": "main-model",
            "Q_T_training_event_count": 40,
            "failure_reason": "model_support_limited",
            "formula": "main",
        }
    )
    oracle = _oracle_curve(
        curve, pd.Series({"E_T_observed_kwh": 0.5, "Q_T_observed_kwh": -0.2})
    )

    assert oracle["cycle_electricity_kwh"].eq(
        oracle["heating_electricity_kwh"] + 0.5
    ).all()
    assert oracle["cycle_net_heat_kwh"].eq(oracle["unit_heating_kwh"] - 0.2).all()
    assert oracle["valid"].equals(oracle["optimization_eligible"])
    assert oracle["component_prediction_valid"].equals(oracle["component_eligible"])
    assert oracle["formula"].eq("(EH+E_T_observed)/(QH+Q_T_observed)").all()
    assert oracle["E_T_model_supported"].isna().all()
    assert oracle["E_T_hat_kwh"].isna().all()
    assert oracle["E_T_model_provenance"].isna().all()
    assert oracle["Q_T_training_event_count"].isna().all()
    assert not oracle["failure_reason"].eq("model_support_limited").any()


def test_macro_mse_is_ratio_of_experiment_mean_mses() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[2] / "scripts/cost/build.py"
    spec = importlib.util.spec_from_file_location("cost_build_v267", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = pd.DataFrame(
        {
            "experiment_id": ["a", "b"],
            "residual_kwh": [0.0, 2.0],
            "baseline_residual_kwh": [1.0, 3.0],
        }
    )

    metrics = module._mse_metrics(values)

    assert metrics["macro_ratio"] == 0.4
    assert module._safe_ratio(0.0, 0.0) == 0.0
    assert np.isinf(module._safe_ratio(1.0, 0.0))

    values.loc[0, "residual_kwh"] = np.nan
    invalid = module._mse_metrics(values)
    assert np.isinf(invalid["ratio"])
    assert np.isinf(invalid["macro_ratio"])


def test_v267_writer_includes_bootstrap_failure_evidence(tmp_path) -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[2] / "scripts/cost/build.py"
    spec = importlib.util.spec_from_file_location("cost_build_writer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = pd.DataFrame({"cycle_name": ["cycle"]})
    table.attrs.update(
        {
            "cycle_audit": pd.DataFrame({"cycle_name": ["cycle"]}),
            "ticket_loeo": pd.DataFrame({"target": ["E_T"]}),
            "v1r": pd.DataFrame({"oracle_only": [True]}),
            "v1r_audit": pd.DataFrame({"comparable": [False]}),
            "bootstrap_audit": pd.DataFrame({"repeat_count": [200]}),
        }
    )

    module._write_v267_artifacts(table, tmp_path)

    assert (tmp_path / "cost_function_v2.6.7_bootstrap_audit.csv").exists()


def test_ticket_events_reject_experiment_ownership_mismatch() -> None:
    from frost_analysis.cost.ticket import _ticket_events

    class Loader:
        def list_cycles(self):
            return pd.DataFrame(
                {
                    "cycle_name": ["cycle"],
                    "experiment_id": ["catalog_experiment"],
                    "defrost_preparation_start": [pd.Timestamp("2026-01-01")],
                    "defrost_start": [pd.Timestamp("2026-01-01 00:01:00")],
                }
            )

    tickets = pd.DataFrame(
        {
            "cycle_name": ["cycle"],
            "experiment_id": ["wrong_experiment"],
            "defrost_electricity_kwh": [1.0],
            "defrost_electricity_coverage": [1.0],
            "defrost_absorbed_heat_kwh": [1.0],
            "defrost_signed_heat_coverage": [1.0],
        }
    )
    recovery = pd.DataFrame(
        {
            "cycle_name": ["cycle"],
            "experiment_id": ["catalog_experiment"],
            "recovery_valid": [True],
            "recovery_electricity_kwh": [1.0],
            "recovery_electricity_coverage": [1.0],
            "recovery_water_heat_kwh": [1.0],
            "recovery_water_heat_coverage": [1.0],
        }
    )

    with np.testing.assert_raises(ValueError):
        _ticket_events(Loader(), {"tickets": tickets, "recovery": recovery}, {})

    class DuplicateCatalog(Loader):
        def list_cycles(self):
            return pd.concat([super().list_cycles()] * 2, ignore_index=True)

    tickets["experiment_id"] = "catalog_experiment"
    with np.testing.assert_raises(ValueError):
        _ticket_events(DuplicateCatalog(), {"tickets": tickets, "recovery": recovery}, {})
