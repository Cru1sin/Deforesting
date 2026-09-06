import numpy as np
import pandas as pd

from plots.pareto_learning import evaluate_predictions, render_figures


def evidence():
    rows = []
    for cycle, logits in enumerate(([1., -1., 1.], [-1., -1., -1.], [-1., 0., 1.]), 1):
        for minute, logit in enumerate(logits):
            rows.append({
                "cycle_name": f"frost_cycle_{cycle:03d}", "experiment_id": "exp_20260101",
                "heldout_experiment": "exp_20260101", "method": "baseline", "seed": 0,
                "image_time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=minute),
                "heating_start": pd.Timestamp("2026-01-01"),
                "stable_heating_start": pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=30),
                "teacher_time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=1),
                "target": [0, .5, 1][minute], "logit": logit,
                "prediction": int(logit >= 0), "decision_score": 1 / (1 + np.exp(-logit)),
                "economic_c": np.nan if minute == 0 else 2 + minute / 10,
                "economic_h": np.nan if minute == 0 else 5 + minute / 10,
                "cycle_cop_eligible_without_extrapolation": minute > 0,
                "cycle_heating_rate_kw_eligible_without_extrapolation": minute > 0,
            })
    predictions = pd.DataFrame(rows)
    teachers = predictions.loc[predictions.image_time.eq(predictions.teacher_time)].copy()
    return predictions, teachers


def test_first_trigger_retains_misses_and_unknown_economics():
    predictions, teachers = evidence()
    cycles, frames, summary = evaluate_predictions(predictions, teachers)
    assert cycles.trigger_error_minutes.tolist()[0] == -1
    assert pd.isna(cycles.trigger_time.iloc[1])
    assert cycles.trigger_domain.iloc[0] == "unknown"
    assert pd.isna(cycles.delta_c.iloc[0])
    assert cycles.delta_c.iloc[2] == 0
    assert cycles.trigger_extra_observed_delay_minutes.iloc[:2].isna().all()
    assert cycles.trigger_extra_observed_delay_minutes.iloc[2] == 0
    assert cycles.persistent_low.tolist() == [False, True, False]
    assert frames.frame_count.sum() == 6  # fractional knee is not a hard class
    assert summary.cycles.iloc[0] == 3
    assert summary.no_trigger_fraction.iloc[0] == 1 / 3
    assert summary.brackets_knee_trigger_fraction.iloc[0] == 2 / 3
    assert summary.brackets_knee_median_error_minutes.iloc[0] == -.5


def test_abstaining_teacher_keeps_native_replay_denominator():
    predictions, teachers = evidence()
    absent = predictions.cycle_name.eq("frost_cycle_003")
    predictions.loc[absent, "teacher_time"] = pd.NaT
    predictions.loc[absent, "target"] = np.nan
    teachers = teachers.loc[~teachers.cycle_name.eq("frost_cycle_003")]
    cycles, _, summary = evaluate_predictions(predictions, teachers)
    assert len(cycles) == 3
    assert cycles.trigger_time.notna().sum() == 2
    assert summary.teacher_covered_cycles.iloc[0] == 2
    assert cycles.observation_status.iloc[2] == "teacher_absent"
    assert pd.isna(cycles.trigger_extra_observed_delay_minutes.iloc[2])
    assert summary.teacher_absent_cycles.iloc[0] == 1
    assert pd.isna(cycles.loc[cycles.cycle_name.eq("frost_cycle_003"), "delta_c"]).all()


def test_shared_plotter_synthetic_smoke(tmp_path):
    predictions, teachers = evidence()
    losses = pd.DataFrame([
        {"method": "baseline", "seed": 0, "heldout_experiment": "exp_20260101",
         "epoch": epoch, "split": split, "loss": 1 / epoch}
        for epoch in (1, 2) for split in ("train_side", "val_side", "train_rank")
    ])
    render_figures(predictions, teachers, losses, pd.DataFrame(), tmp_path)
    assert (tmp_path / "decision_summary.png").is_file()
    assert (tmp_path / "trigger_error_first_positive_front.png").is_file()
    assert (tmp_path / "representative_streams.png").is_file()


def test_cropped_stream_observation_status_keeps_all_cycle_denominators():
    predictions, teachers = evidence()
    original, _, _ = evaluate_predictions(predictions, teachers)
    cropped = predictions.loc[
        (~predictions.cycle_name.eq("frost_cycle_001")
         | predictions.image_time.gt(predictions.teacher_time))
        & (~predictions.cycle_name.eq("frost_cycle_002")
           | predictions.image_time.lt(predictions.teacher_time))
    ]
    cycles, _, summary = evaluate_predictions(cropped, teachers)
    assert "observation_status" in cycles
    assert cycles.observation_status.tolist() == [
        "starts_after_knee", "ends_before_knee", "brackets_knee"
    ]
    assert cycles.first_native_relative_teacher_minutes.tolist() == [1., -1., -1.]
    assert cycles.last_native_relative_teacher_minutes.tolist() == [1., -1., 1.]
    assert original.loc[1, "observed_post_knee"]
    assert not cycles.loc[1, "observed_post_knee"]
    assert pd.isna(original.loc[1, "trigger_time"]) and pd.isna(cycles.loc[1, "trigger_time"])
    assert summary.cycles.iloc[0] == 3
    assert summary.triggered_cycles.iloc[0] == 2
    assert summary.no_trigger_fraction.iloc[0] == 1 / 3
    assert summary.brackets_knee_cycles.iloc[0] == 1
    assert summary.brackets_knee_trigger_fraction.iloc[0] == 1
    assert summary.brackets_knee_median_error_minutes.iloc[0] == 0


def test_internal_gap_exposes_observation_delay_and_preserves_plot_points(tmp_path, monkeypatch):
    from plots import pareto_learning

    predictions, teachers = evidence()
    selected = predictions.cycle_name.eq("frost_cycle_003")
    origin = predictions.image_time.min()
    predictions.loc[selected, "image_time"] = [
        origin + pd.Timedelta(minutes=value) for value in (0, 10, 10.5)
    ]
    predictions.loc[selected, "logit"] = [-1., -1., 1.]
    predictions.loc[selected, "prediction"] = [0, 0, 1]
    predictions.loc[selected, "target"] = [0, 1, 1]
    cycles, frames, summary = evaluate_predictions(predictions, teachers)
    row = cycles.loc[cycles.cycle_name.eq("frost_cycle_003")].iloc[0]
    assert "nearest_pre_knee_relative_minutes" in cycles
    assert row.nearest_pre_knee_relative_minutes == -1
    assert row.nearest_post_knee_relative_minutes == 9
    assert row.trigger_extra_observed_delay_minutes == .5
    assert row.trigger_error_minutes == 9.5
    assert summary.cycles.iloc[0] == 3 and frames.frame_count.sum() == 7
    figures = []
    monkeypatch.setattr(pareto_learning, "_export", lambda figure, path: figures.append(figure))
    pareto_learning._representatives(
        predictions.loc[selected], cycles.loc[cycles.cycle_name.eq("frost_cycle_003")], tmp_path
    )
    lines = [line for line in figures[0].axes[0].lines if line.get_marker() == "."]
    assert sorted(len(line.get_xdata()) for line in lines) == [1, 2]
    assert sum(len(line.get_xdata()) for line in lines) == 3
    assert all(np.diff(line.get_xdata()).max(initial=0) <= .75 for line in lines)


def test_representative_uses_saved_grid_and_keeps_outlying_trigger(tmp_path, monkeypatch):
    from plots import pareto_learning

    predictions, teachers = evidence()
    predictions = predictions.loc[predictions.cycle_name.eq("frost_cycle_003")].copy()
    predictions.loc[predictions.logit.ge(0), "economic_c"] = 9.0
    cycles, _, _ = evaluate_predictions(predictions, teachers)
    grid = predictions.iloc[[0, 1, 2]].copy().assign(
        candidate_defrost_time=predictions.image_time.to_numpy(),
        is_teacher_candidate=True, is_knee=[False, True, False],
        pareto_selection_score=[.1, .9, np.nan],
        cycle_cop=[2.0, 2.1, 2.2], cycle_heating_rate_kw=[5., 5.1, 5.2],
    )
    seen, figures = [], []
    def original_renderer(axis, values, origin):
        seen.append(values.copy())
        axis.set(xlim=(1, 3), ylim=(4, 6))
    monkeypatch.setattr(pareto_learning, "plot_cop_heating_rate_pareto", original_renderer)
    monkeypatch.setattr(pareto_learning, "_export", lambda fig, path: figures.append(fig))
    pareto_learning._representatives(predictions, cycles, tmp_path, teacher_curves=grid)
    assert len(seen[0]) == 3
    assert seen[0].is_cop_heating_rate_pareto_point.tolist() == [True, True, False]
    assert seen[0].is_selected_pareto_point.tolist() == [False, True, False]
    assert figures[0].axes[1].get_xlim()[1] > 9
    assert (tmp_path / "representative_pareto.csv").exists()
