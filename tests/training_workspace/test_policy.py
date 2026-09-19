import numpy as np
import pandas as pd
import pytest
import torch
from image_models.policy import (
    cycle_batches,
    positive_logit,
    replay_cycles,
    select_cop_threshold,
    summarize_metrics,
)


def test_replay_is_first_positive_with_forced_final_and_threshold_selected_by_cop():
    times = pd.date_range("2026-01-01", periods=3, freq="30s")
    rows = pd.DataFrame(
        {
            "cycle_name": ["a"] * 3 + ["b"] * 3,
            "candidate_defrost_time": list(times) * 2,
            "cycle_cop": [1.0, 4.0, 3.0, 2.0, 3.0, 5.0],
            "probability": [0.4, 0.8, 0.9, 0.2, 0.6, 0.7],
            "optimal_time": [times[1]] * 3 + [times[2]] * 3,
        }
    )
    replay = replay_cycles(rows, 0.75)
    assert replay.selected_time.tolist() == [times[1], times[2]]
    assert replay.selected_cop.tolist() == [4.0, 5.0]
    assert replay.forced_final.tolist() == [False, True]
    assert replay.early_trigger.tolist() == [False, False]

    selected, grid = select_cop_threshold(rows, thresholds=(0.5, 0.75))
    assert selected == 0.75
    assert grid.loc[grid.threshold.eq(0.75), "mean_cycle_cop"].item() == pytest.approx(4.5)


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


def test_summary_computes_ratios_from_mean_cops_not_mean_cycle_percentages():
    rows = pd.DataFrame(
        {
            "cycle_name": ["a", "b"],
            "selected_cop": [4.0, 8.0],
            "rb_cop": [2.0, 6.0],
            "optimal_cop": [5.0, 10.0],
            "evaluation_status": ["evaluated", "evaluated"],
            "early_trigger": [False, True],
            "soft_hard_gap": [1.0, 3.0],
        }
    )
    result = summarize_metrics(rows)
    assert result["mean_cycle_cop"] == 6
    assert result["gain_vs_rb_pct"] == pytest.approx(50)
    assert result["headroom_captured_pct"] == pytest.approx(100 * 2 / 3.5)
    assert result["headroom_remaining_pct"] == pytest.approx(100 - 100 * 2 / 3.5)


def test_chen_two_logits_are_converted_to_class_one_log_odds():
    output = torch.tensor([[2.0, 3.0], [4.0, 1.0]])
    torch.testing.assert_close(positive_logit(output), torch.tensor([1.0, -3.0]))
    torch.testing.assert_close(positive_logit(torch.tensor([2.0, 3.0])), torch.tensor([2.0, 3.0]))


def test_two_of_three_uses_uncompressed_reference_clock_and_forces_final():
    times = pd.date_range("2026-01-01", periods=5, freq="30s")
    predicted = pd.DataFrame(
        {
            "cycle_name": ["a"] * 3,
            "experiment_id": ["e"] * 3,
            "candidate_defrost_time": times[[0, 3, 4]],
            "cycle_cop": [1.0, 4.0, 3.0],
            "probability": [0.9, 0.9, 0.1],
            "optimal_time": [times[3]] * 3,
            "optimal_cop": [4.0] * 3,
        }
    )
    reference = pd.DataFrame(
        {
            "cycle_name": ["a"] * 5,
            "experiment_id": ["e"] * 5,
            "candidate_defrost_time": times,
            "cycle_cop": [1.0, 2.0, 3.0, 4.0, 3.0],
            "cycle_cop_eligible": True,
        }
    )
    replay = replay_cycles(predicted, 0.5, strategy="two_of_three", reference=reference)
    assert replay.selected_time.item() == times[4]
    assert replay.forced_final.item()
