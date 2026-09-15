from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from image_models import stopping_loss_comparison as stopping
from image_models.stopping_loss_comparison import (
    ARCHITECTURES,
    _fold_reference,
    _saved_fold_results,
    action_rows,
    after_optimum_loss,
    architecture_action_rows,
    architecture_contract,
    build_model,
    cop_stopping_loss,
    cycle_batches,
    fit_policy,
    policy_cycle_metrics,
    positive_logit,
    predict_policy,
    replay_cycles,
    select_cop_threshold,
    stopping_distribution,
    summarize_metrics,
)


def _curve():
    return pd.DataFrame({
        "cycle_name": ["a"] * 5,
        "candidate_defrost_time": pd.date_range("2026-01-01", periods=5, freq="30s"),
        "cycle_cop": [9.0, 10.0, 10.0, 8.0, 7.0],
        "cycle_cop_eligible": [False, True, True, True, True],
        "model_input_available": [True, True, True, False, True],
        "physically_allowed": [False, True, True, True, True],
    })


def test_action_rows_freeze_input_cop_and_physical_support_and_take_earliest_tie():
    selected = action_rows(_curve())
    assert selected.index.tolist() == [0, 1, 2]
    assert selected.candidate_defrost_time.tolist() == list(pd.to_datetime([
        "2026-01-01 00:00:30", "2026-01-01 00:01:00", "2026-01-01 00:02:00",
    ]))
    assert selected.after_optimum_target.tolist() == [1.0, 1.0, 1.0]
    assert selected.optimal_time.nunique() == 1
    assert selected.optimal_time.iloc[0] == pd.Timestamp("2026-01-01 00:00:30")
    with pytest.raises(ValueError, match="physically_allowed"):
        action_rows(_curve().drop(columns="physically_allowed"))


def test_new_sensor_action_domain_does_not_depend_on_rgb_history():
    time = pd.Timestamp("2026-01-01 00:01:00")
    rows = pd.DataFrame({
        "cycle_name": ["a"], "candidate_defrost_time": [time],
        "cycle_cop": [3.0], "cycle_cop_eligible": [True],
        "sensor_timestamp": [time - pd.Timedelta(seconds=1)],
        "stat_ambient_temperature_current": [5.0],
        "development_input_available": [False],
        "stable_heating_start": [time - pd.Timedelta(minutes=1)],
        "observed_defrost_preparation_start": [time + pd.Timedelta(minutes=1)],
    })
    assert len(architecture_action_rows(rows, "new_sensor")) == 1
    assert architecture_action_rows(
        rows.drop(columns="observed_defrost_preparation_start"), "new_sensor"
    ).empty


def test_fold_reference_restores_authoritative_physical_boundaries(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from image_models import relative_cop

    rows = pd.DataFrame({
        "cycle_name": ["a"], "value": [1],
        "stable_heating_start": ["stale"], "t_RB": ["stale"],
    })
    captured = {}

    def fake_builder(*args, **kwargs):
        captured.update(kwargs)
        return rows.copy(), {}

    monkeypatch.setattr(relative_cop, "build_fold_rows", fake_builder)
    cohort = pd.DataFrame({
        "cycle_name": ["a"], "experiment_id": ["e"],
        "stable_heating_start": ["2026-01-01 00:01:00"],
        "observed_defrost_preparation_start": ["2026-01-01 00:05:00"],
        "observation_end": ["2026-01-01 00:06:00"],
        "t_RB": ["2026-01-01 00:02:00"],
    })
    args = SimpleNamespace(output=tmp_path, data=tmp_path)
    result = _fold_reference(args, cohort, pd.DataFrame(), ("e",), tmp_path)
    assert result.observation_end.iloc[0] == "2026-01-01 00:06:00"
    assert result.stable_heating_start.iloc[0] == "2026-01-01 00:01:00"
    assert result.t_RB.iloc[0] == "2026-01-01 00:02:00"
    assert captured["include_history"] is True


def test_stopping_distribution_forces_the_last_legal_action():
    logits = torch.tensor([0.0, 0.0, -100.0])
    probabilities = stopping_distribution(logits)
    torch.testing.assert_close(probabilities, torch.tensor([0.5, 0.25, 0.25]))
    assert probabilities.sum() == pytest.approx(1.0)


def test_cycle_losses_use_cop_regret_and_present_class_balancing():
    logits = torch.zeros(3)
    cop = torch.tensor([1.0, 3.0, 2.0])
    assert cop_stopping_loss(logits, cop) == pytest.approx(1.25)

    targets = torch.tensor([0.0, 1.0, 1.0])
    assert after_optimum_loss(logits, targets) == pytest.approx(np.log(2))
    only_after = torch.ones(3)
    assert after_optimum_loss(logits, only_after) == pytest.approx(np.log(2))


def test_replay_is_first_positive_with_forced_final_and_threshold_selected_by_cop():
    times = pd.date_range("2026-01-01", periods=3, freq="30s")
    rows = pd.DataFrame({
        "cycle_name": ["a"] * 3 + ["b"] * 3,
        "candidate_defrost_time": list(times) * 2,
        "cycle_cop": [1.0, 4.0, 3.0, 2.0, 3.0, 5.0],
        "probability": [.4, .8, .9, .2, .6, .7],
        "optimal_time": [times[1]] * 3 + [times[2]] * 3,
    })
    replay = replay_cycles(rows, .75)
    assert replay.selected_time.tolist() == [times[1], times[2]]
    assert replay.selected_cop.tolist() == [4.0, 5.0]
    assert replay.forced_final.tolist() == [False, True]
    assert replay.early_trigger.tolist() == [False, False]

    selected, grid = select_cop_threshold(rows, thresholds=(.5, .75))
    assert selected == .75
    assert grid.loc[grid.threshold.eq(.75), "mean_cycle_cop"].item() == pytest.approx(4.5)


def test_cycle_batches_never_split_a_trajectory():
    rows = pd.DataFrame({"cycle_name": ["a", "a", "b", "c", "c", "c"]})
    batches = list(cycle_batches(rows, cycles_per_batch=2, seed=4))
    assert sorted(np.concatenate(batches).tolist()) == list(range(len(rows)))
    assert all(rows.iloc[index].cycle_name.nunique() <= 2 for index in batches)
    assert all(
        sum(rows.iloc[index].cycle_name.eq(name).sum() for index in batches)
        == rows.cycle_name.eq(name).sum()
        for name in rows.cycle_name.unique()
    )


def test_five_architectures_have_paired_seed_identical_initialization():
    assert tuple(ARCHITECTURES) == (
        "r_sensor", "r_sensor_rgb", "chen_rgb", "new_sensor", "new_rgb_difference",
    )
    for name in ARCHITECTURES:
        first = build_model(name, numeric_width=7, visual_width=5, seed=11)
        second = build_model(name, numeric_width=7, visual_width=5, seed=11)
        assert first.state_dict().keys() == second.state_dict().keys()
        for key in first.state_dict():
            torch.testing.assert_close(first.state_dict()[key], second.state_dict()[key])
    sensor = build_model("new_sensor", numeric_width=7, visual_width=0, seed=0)
    assert not any("visual" in name for name in sensor.state_dict())


def test_architecture_contracts_reuse_the_five_existing_input_shapes():
    sizes = {
        name: tuple(map(len, architecture_contract(name))) for name in ARCHITECTURES
    }
    assert sizes == {
        "r_sensor": (199, 0),
        "r_sensor_rgb": (199, 384),
        "chen_rgb": (0, 384),
        "new_sensor": (191, 0),
        "new_rgb_difference": (191, 768),
    }


def test_policy_metrics_use_supported_rb_and_report_headroom_and_soft_hard_gap():
    times = pd.date_range("2026-01-01", periods=3, freq="30s")
    predicted = pd.DataFrame({
        "cycle_name": ["a"] * 3,
        "experiment_id": ["e"] * 3,
        "candidate_defrost_time": times,
        "cycle_cop": [2.0, 4.0, 3.0],
        "probability": [.2, .8, .9],
        "optimal_time": [times[1]] * 3,
    })
    reference = predicted.assign(
        cycle_cop_eligible=True, t_RB=times[0], model_input_available=True
    )
    metrics = policy_cycle_metrics(predicted, reference, .5, loss_name="cop_stopping")
    row = metrics.iloc[0]
    assert row.selected_cop == 4
    assert row.rb_cop == 2
    assert row.gain_vs_rb_pct == pytest.approx(100)
    assert row.headroom_captured_pct == pytest.approx(100)
    assert np.isfinite(row.soft_hard_gap)


def test_summary_computes_ratios_from_mean_cops_not_mean_cycle_percentages():
    rows = pd.DataFrame({
        "cycle_name": ["a", "b"], "selected_cop": [4.0, 8.0],
        "rb_cop": [2.0, 6.0], "optimal_cop": [5.0, 10.0],
        "evaluation_status": ["evaluated", "evaluated"],
        "early_trigger": [False, True], "soft_hard_gap": [1.0, 3.0],
    })
    result = summarize_metrics(rows)
    assert result["mean_cycle_cop"] == 6
    assert result["gain_vs_rb_pct"] == pytest.approx(50)
    assert result["headroom_captured_pct"] == pytest.approx(100 * 2 / 3.5)
    assert result["headroom_remaining_pct"] == pytest.approx(100 - 100 * 2 / 3.5)


def test_chen_two_logits_are_converted_to_class_one_log_odds():
    output = torch.tensor([[2.0, 3.0], [4.0, 1.0]])
    torch.testing.assert_close(positive_logit(output), torch.tensor([1.0, -3.0]))
    torch.testing.assert_close(positive_logit(torch.tensor([2.0, 3.0])), torch.tensor([2.0, 3.0]))


@pytest.mark.parametrize("loss_name", ["after_optimum", "cop_stopping"])
def test_fit_policy_uses_cycle_batches_and_cop_selected_threshold(loss_name):
    times = pd.date_range("2026-01-01", periods=3, freq="30s")
    rows = pd.DataFrame({
        "cycle_name": ["a"] * 3 + ["b"] * 3,
        "experiment_id": ["e1"] * 3 + ["e2"] * 3,
        "candidate_defrost_time": list(times) * 2,
        "cycle_cop": [1.0, 3.0, 2.0, 2.0, 3.0, 4.0],
        "optimal_time": [times[1]] * 3 + [times[2]] * 3,
        "after_optimum_target": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        "x": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
    })
    fitted = fit_policy(
        rows, rows, architecture="r_sensor", loss_name=loss_name,
        numeric_columns=["x"], visual_columns=[], seed=3, maximum_epochs=2,
        patience=2, cycles_per_batch=1, thresholds=(.25, .75),
    )
    assert fitted["selected_epoch"] in (1, 2)
    assert fitted["threshold"] in (.25, .75)
    assert fitted["losses"].optimizer_steps.eq(2).all()
    rng = torch.random.get_rng_state()
    prediction = predict_policy(rows, fitted)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    assert prediction.probability.between(0, 1).all()


def test_saved_fold_results_collect_every_resumable_fold(tmp_path):
    import pickle

    for name in ("a", "b"):
        with (tmp_path / f"{name}.pkl").open("wb") as stream:
            pickle.dump({"heldout_experiment": name}, stream)
    assert [row["heldout_experiment"] for row in _saved_fold_results(tmp_path)] == [
        "a", "b",
    ]


def test_reliable_pool_filters_boundaries_and_events(monkeypatch, tmp_path):
    from image_models import relative_cop

    reliable = pd.DataFrame({
        "cycle_name": ["b", "a"], "experiment_id": ["e2", "e1"],
    })

    class Loader:
        def __init__(self, _):
            pass

        def list_valid_cycles(self, *, require_rgb):
            assert require_rgb is True
            return reliable

    monkeypatch.setattr(relative_cop, "DatasetLoader", Loader)
    boundaries = pd.DataFrame({
        "cycle_name": ["rogue", "a", "b"],
        "experiment_id": ["e9", "e1", "e2"], "boundary": [0, 1, 2],
    })
    events = pd.DataFrame({
        "cycle_name": ["a", "rogue", "b"], "experiment_id": ["e1", "e9", "e2"],
    })
    cohort, filtered_events = stopping.reliable_stopping_inputs(
        SimpleNamespace(dataset=tmp_path), boundaries, events,
    )
    assert cohort.cycle_name.tolist() == ["a", "b"]
    assert filtered_events.cycle_name.tolist() == ["a", "b"]


def test_two_of_three_uses_uncompressed_reference_clock_and_forces_final():
    times = pd.date_range("2026-01-01", periods=5, freq="30s")
    predicted = pd.DataFrame({
        "cycle_name": ["a"] * 3, "experiment_id": ["e"] * 3,
        "candidate_defrost_time": times[[0, 3, 4]], "cycle_cop": [1.0, 4.0, 3.0],
        "probability": [.9, .9, .1], "optimal_time": [times[3]] * 3,
        "optimal_cop": [4.0] * 3,
    })
    reference = pd.DataFrame({
        "cycle_name": ["a"] * 5, "experiment_id": ["e"] * 5,
        "candidate_defrost_time": times, "cycle_cop": [1., 2., 3., 4., 3.],
        "cycle_cop_eligible": True,
    })
    replay = replay_cycles(predicted, .5, strategy="two_of_three", reference=reference)
    assert replay.selected_time.item() == times[4]
    assert replay.forced_final.item()


def test_frozen_probabilities_select_independent_controller_thresholds():
    times = pd.date_range("2026-01-01", periods=5, freq="30s")
    rows = pd.DataFrame({
        "cycle_name": ["a"] * 5, "candidate_defrost_time": times,
        "cycle_cop": [1., 1.99, 2., 1.8, 1.7],
        "probability": [.2, .9, .95, .3, .1], "optimal_time": [times[2]] * 5,
    })
    selected, grid = stopping.select_controller_thresholds(rows, thresholds=(.5, .9, .95))
    assert set(selected) == {"first_positive", "two_of_three"}
    assert selected["first_positive"] == .95
    assert selected["two_of_three"] == .9
    assert set(grid.strategy) == {"first_positive", "two_of_three"}
