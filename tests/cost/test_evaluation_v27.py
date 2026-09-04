from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frost_analysis.cost.evaluation import (
    _metric_copy,
    _metric_measurement_support,
    _required_model_support,
    earliest_t0_proxy,
    finalize_metric_curve,
    project_two_anchor_loss,
    select_two_anchors,
    t0_proxy_reference_kwh,
)


def test_final_objectives_use_only_their_native_model_support() -> None:
    support = {
        "E_T": np.array([False, True]),
        "Q_T": np.array([True, True]),
        "Qw0": np.array([True, True]),
        "Pcomp0": np.array([False, True]),
        "E_comp_T": np.array([False, True]),
        "D_T": np.array([True, True]),
    }

    assert _required_model_support("cop_cyc_evt", support).tolist() == [False, True]
    assert _required_model_support("eta_h_cyc", support).tolist() == [True, True]
    assert _required_model_support("eta_e_cyc", support).tolist() == [False, True]


def test_final_objectives_use_native_measurement_support() -> None:
    frame = pd.DataFrame(
        {
            "heating_electricity_measurement_eligible": [False, True],
            "water_heating_measurement_eligible": [True, True],
            "heating_compressor_measurement_eligible": [False, True],
        }
    )

    assert _metric_measurement_support(frame, "cop_cyc_evt").tolist() == [False, True]
    assert _metric_measurement_support(frame, "eta_h_cyc").tolist() == [True, True]
    assert _metric_measurement_support(frame, "eta_e_cyc").tolist() == [False, True]


def test_metric_copy_renames_inherited_v268_inverse_cop() -> None:
    start = pd.Timestamp("2026-01-01")
    curve = pd.DataFrame(
        {
            "cycle_name": "cycle",
            "candidate_time": pd.date_range(start, periods=7, freq="min"),
            "objective_value": np.linspace(0.8, 1.0, 7),
            "inverse_cop": np.linspace(1.0, 1.2, 7),
            "supported": True,
            "pre_action_window_valid": True,
            "measurement_eligible": True,
            "physical_valid": True,
        }
    )

    result = _metric_copy(
        curve,
        "cop_e",
        curve["objective_value"],
        curve["supported"],
        curve["physical_valid"],
        measurement_eligible=curve["measurement_eligible"],
    )

    assert "inverse_cop" not in result
    np.testing.assert_allclose(result["legacy_v268_inverse_cop"], curve["inverse_cop"])


@pytest.mark.parametrize(
    ("direction", "values", "expected_time"),
    [
        ("min", [1.20, 1.10, 1.02, 1.00, 1.005, 1.04, 1.10], "00:03:00"),
        ("max", [0.80, 0.90, 0.98, 1.00, 0.995, 0.94, 0.88], "00:03:00"),
    ],
)
def test_direction_aware_extreme_and_connected_basins(
    direction: str, values: list[float], expected_time: str
) -> None:
    start = pd.Timestamp("2026-01-01")
    curve = pd.DataFrame(
        {
            "candidate_time": pd.date_range(start, periods=7, freq="min"),
            "objective_value": values,
            "supported": True,
            "pre_action_window_valid": True,
            "measurement_eligible": True,
            "physical_valid": True,
            "identifiable": True,
        }
    )

    result = finalize_metric_curve(curve, direction, curve["measurement_eligible"])

    optimum = start + pd.Timedelta(expected_time)
    assert pd.Timestamp(result["t_star"].iloc[0]) == optimum
    assert result.loc[3:4, "near_optimal_1pct"].all()
    assert not result.loc[[2, 5], "near_optimal_1pct"].any()
    assert result["basin_1pct_width_minutes"].iloc[0] == pytest.approx(1.0)


def test_rr_measurement_mask_is_independent_of_fixed9_mask() -> None:
    start = pd.Timestamp("2026-01-01")
    base = pd.DataFrame(
        {
            "cycle_name": "cycle",
            "candidate_time": pd.date_range(start, periods=6, freq="min"),
            "objective_value": 1.0,
            "supported": True,
            "pre_action_window_valid": True,
            "physical_valid": True,
            "identifiable": True,
        }
    )
    rr_eligible = base.assign(
        measurement_eligible=False,
        rr_measurement_eligible=True,
    )
    rr_ineligible = base.assign(
        measurement_eligible=True,
        rr_measurement_eligible=False,
    )

    counted = _metric_copy(
        rr_eligible,
        "cop_cyc_k",
        rr_eligible["objective_value"],
        rr_eligible["supported"],
        rr_eligible["physical_valid"],
        measurement_eligible=rr_eligible["rr_measurement_eligible"],
    )
    skipped = _metric_copy(
        rr_ineligible,
        "cop_cyc_k",
        rr_ineligible["objective_value"],
        rr_ineligible["supported"],
        rr_ineligible["physical_valid"],
        measurement_eligible=rr_ineligible["rr_measurement_eligible"],
    )

    assert counted["optimization_eligible"].any()
    assert not skipped["optimization_eligible"].any()


def test_t0_proxy_uses_earliest_raw_120_second_stable_window() -> None:
    start = pd.Timestamp("2026-01-01")
    timestamps = pd.date_range(start, periods=150, freq="s")
    water_heat = np.r_[np.full(10, 12.0), np.full(140, 10.0)]
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "water_flow": 1.0,
            "water_in_temperature": 40.0,
            "water_out_temperature": 40.0 + water_heat / 1.161,
        }
    )

    proxy = earliest_t0_proxy(frame, start)

    assert proxy["valid"] is True
    assert proxy["start"] == start + pd.Timedelta(seconds=10)
    assert proxy["end"] == start + pd.Timedelta(seconds=130)
    assert proxy["water_heat_kw"] == pytest.approx(10.0)
    assert proxy["coverage"] == pytest.approx(1.0)
    assert proxy["maximum_gap_seconds"] <= 30


def test_t0_proxy_reference_uses_proxy_start_and_rejects_future_candidates() -> None:
    start = pd.Timestamp("2026-01-01 00:00:10")
    end = start + pd.Timedelta(seconds=120)

    assert t0_proxy_reference_kwh(
        pd.Timestamp("2026-01-01 00:02:10"), start, end, 10.0
    ) == pytest.approx(10.0 * 120 / 3600)
    assert np.isnan(
        t0_proxy_reference_kwh(
            pd.Timestamp("2026-01-01 00:02:09"), start, end, 10.0
        )
    )


def test_two_anchor_selection_excludes_target_and_requires_ordered_distinct_cycles() -> None:
    events = pd.DataFrame(
        {
            "cycle_name": ["target", "mild", "severe", "other_experiment"],
            "experiment_id": ["exp_a", "exp_a", "exp_a", "exp_b"],
            "attenuation_fraction": [0.20, 0.06, 0.33, 0.05],
            "heating_elapsed_minutes": [25.0, 12.0, 30.0, 10.0],
            "L_T_t0_kwh": [0.3, 0.1, 0.5, 99.0],
            "event_valid": True,
        }
    )

    anchors = select_two_anchors(events, "target", "exp_a")

    assert anchors["valid"] is True
    assert anchors["anchor_5_cycle"] == "mild"
    assert anchors["anchor_35_cycle"] == "severe"
    assert "target" not in {anchors["anchor_5_cycle"], anchors["anchor_35_cycle"]}
    assert project_two_anchor_loss(21.0, anchors) == pytest.approx(0.3)


def test_bootstrap_refit_separates_training_and_heldout_anchor_pools(monkeypatch) -> None:
    """Bootstrap models exclude heldout data while anchors use heldout siblings."""
    import frost_analysis.cost.evaluation as evaluation

    class DummyModel:
        alpha = 1.0

        def predict(self, values):
            return np.zeros(len(values), dtype=float)

    heldout_times = pd.date_range("2026-01-01 00:10:00", periods=6, freq="min")
    common = pd.DataFrame(
        {
            "cycle_name": "target",
            "experiment_id": "heldout",
            "candidate_time": heldout_times,
            "stable_start_fixed9": heldout_times[0] - pd.Timedelta(minutes=1),
            "pre_action_window_valid": True,
            "measurement_eligible": True,
        }
    )
    curves = {
        "v2.7.3": common.assign(
            metric_id="epsilon_hl_2a",
            basin_5pct_start=heldout_times[0],
            basin_5pct_end=heldout_times[-1],
        )
    }
    events = pd.DataFrame(
        {
            "cycle_name": [
                "a_event",
                "b_event",
                "heldout_mild",
                "heldout_severe",
                "target",
            ],
            "experiment_id": ["a", "b", "heldout", "heldout", "heldout"],
            "event_valid": True,
            "event_duration_observed_minutes": 10.0,
            "Q_T_observed_kwh": 1.0,
            "attenuation_fraction": [0.05, 0.35, 0.06, 0.33, 0.20],
            "heating_elapsed_minutes": [10.0, 20.0, 12.0, 30.0, 25.0],
            "L_T_t0_kwh": [0.1, 0.5, 0.1, 0.5, 0.3],
        }
    )
    healthy_samples = pd.DataFrame(
        {
            "cycle_name": ["a_healthy", "b_healthy", "heldout_healthy"],
            "experiment_id": ["a", "b", "heldout"],
        }
    )
    folds = {
        "heldout": {
            "healthy_water_heat_kw": DummyModel(),
            "healthy_compressor_power_kw": DummyModel(),
            "E_comp_T_observed_kwh": DummyModel(),
            "event_duration_observed_minutes": DummyModel(),
            "L_T_t0_kwh": DummyModel(),
            "E_T_observed_kwh": DummyModel(),
            "Q_T_observed_kwh": DummyModel(),
            "E_PD_kwh": DummyModel(),
            "Q_PD_kwh": DummyModel(),
        }
    }
    dynamic_loss_folds = {"heldout": DummyModel()}
    seen_events: list[pd.DataFrame] = []
    seen_training: list[tuple[str, pd.DataFrame]] = []

    monkeypatch.setattr(
        evaluation,
        "_fit_bootstrap_model",
        lambda training, target, alpha, features=evaluation.DYNAMIC_8: (
            seen_training.append((target, training.copy())) or DummyModel()
        ),
    )

    def capture_anchor_input(candidates, anchor_events, experiment_id, q0_model):
        seen_events.append(anchor_events.copy())
        return candidates.assign(
            L_T_two_anchor_hat_kwh=0.0,
            anchor_identifiable=True,
        )

    monkeypatch.setattr(evaluation, "_bootstrap_two_anchor_curve", capture_anchor_input)
    monkeypatch.setattr(
        evaluation,
        "_bootstrap_objectives",
        lambda candidates: {
            "epsilon_hl_2a": np.full(len(candidates), 0.1),
        },
    )
    monkeypatch.setattr(
        evaluation,
        "_bootstrap_metric_status",
        lambda candidates, models: {
            "epsilon_hl_2a": (
                np.ones(len(candidates), dtype=bool),
                np.ones(len(candidates), dtype=bool),
                np.ones(len(candidates), dtype=bool),
            ),
        },
    )

    evaluation._bootstrap_refit_summary(
        common,
        curves,
        events,
        healthy_samples,
        folds,
        folds,
        folds,
        dynamic_loss_folds,
        replicates=3,
        seed=270,
    )

    assert len(seen_events) == 3
    for anchor_events in seen_events:
        assert set(anchor_events["source_experiment_id"]) == {"heldout"}
        assert "target" not in set(anchor_events["cycle_name"])
        assert set(anchor_events["experiment_id"]).issubset({"draw_000", "draw_001"})

    assert seen_events[0]["cycle_name"].tolist() != seen_events[1]["cycle_name"].tolist()
    for _, training in seen_training:
        assert "source_experiment_id" in training
        assert "heldout" not in set(training["source_experiment_id"])
        assert "heldout" not in set(training["experiment_id"])


def test_select_two_anchors_uses_source_experiment_and_distinct_siblings() -> None:
    """Draw labels must not hide the original experiment or duplicate sibling."""
    events = pd.DataFrame(
        {
            "cycle_name": ["mild", "mild", "severe", "target"],
            "experiment_id": ["draw_000", "draw_001", "draw_001", "draw_001"],
            "source_experiment_id": ["heldout"] * 4,
            "attenuation_fraction": [0.06, 0.06, 0.33, 0.20],
            "heating_elapsed_minutes": [12.0, 12.0, 30.0, 25.0],
            "L_T_t0_kwh": [0.1, 0.1, 0.5, 0.3],
            "event_valid": True,
        }
    )

    anchors = select_two_anchors(events, "target", "heldout")

    assert anchors["valid"] is True
    assert anchors["anchor_5_cycle"] == "mild"
    assert anchors["anchor_35_cycle"] == "severe"

    duplicate_only = events.iloc[[0, 1, 3]].copy()
    invalid = select_two_anchors(duplicate_only, "target", "heldout")
    assert invalid["valid"] is False
    assert invalid["reason"] == "fewer_than_two_sibling_events"
