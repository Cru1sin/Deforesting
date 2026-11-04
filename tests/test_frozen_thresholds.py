import pandas as pd

from tests.cost_workspace.frozen_thresholds import choose_threshold, loeo_replay, replay_curve


def test_first_crossing_is_not_moved_and_training_uses_cop():
    curve = pd.DataFrame(
        dict(
            cycle_name="a",
            experiment_id="day",
            Ts=50.0,
            candidate_defrost_time=pd.date_range("2026-01-01", periods=4, freq="30s"),
            visual_frost_score=[0.1, 0.4, 0.5, 0.8],
            cycle_cop=[2.0, 2.8, 3.0, 2.9],
            cycle_cop_eligible=[True, False, True, True],
            cop_norm=[2 / 3, float("nan"), 1.0, 2.9 / 3],
        )
    )
    result = replay_curve(curve, [0.3, 0.5, 0.7])
    assert result.loc[result.threshold.eq(0.3), "status"].item() == "unsupported_cop"
    assert pd.isna(result.loc[result.threshold.eq(0.3), "selected_cop"].item())
    assert choose_threshold(result)[0] == 0.5
    other = result.copy()
    other["cycle_name"] = "b"
    other.loc[other.threshold.eq(0.5), "selected_cop"] = 1.0
    assert choose_threshold(pd.concat([result, other]))[0] == 0.7
    assert choose_threshold(result.loc[result.threshold.eq(0.3)])[0] is None
    other = result.copy()
    other["cycle_name"] = "b"
    other["experiment_id"] = "held"
    grid = pd.concat([result, other], ignore_index=True)
    _, before = loeo_replay(grid, ["a", "b"], "test")
    grid.loc[grid.experiment_id.eq("held"), "selected_cop"] += 10
    _, after = loeo_replay(grid, ["a", "b"], "test")
    pd.testing.assert_series_equal(
        before.loc[before.experiment_id.eq("held"), "threshold"],
        after.loc[after.experiment_id.eq("held"), "threshold"],
    )
