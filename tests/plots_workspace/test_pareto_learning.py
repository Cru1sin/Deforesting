import numpy as np
import pandas as pd
import pytest

from plots.pareto_learning import (
    METHOD_NAMES,
    actual_action_ch_calibration,
    cross_input_comparisons,
    evaluate_predictions,
    event_calibration_metrics,
    maximum_relative_performance_loss,
    performance_consequences,
    relation_stability_metrics,
    render_figures,
    render_neural_sensitivity,
    render_outcome_figures,
    render_performance_coverage,
    trigger_is_dominated,
)


def test_loss_plot_limits_do_not_sacrifice_all_curves_to_one_divergent_point():
    from plots.pareto_learning import _loss_plot_limits

    lower, upper, clipped = _loss_plot_limits(
        pd.Series([*np.linspace(.05, .8, 1000), 2e9])
    )

    assert lower > 0
    assert upper < 2
    assert clipped == 1


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


def test_trigger_dominance_uses_the_frozen_teacher_grid_without_moving_selection():
    grid = pd.DataFrame({
        "cycle_cop": [2.0, 2.1, 1.9],
        "cycle_heating_rate_kw": [5.0, 4.9, 5.2],
        "is_teacher_candidate": [True, True, False],
    })
    assert trigger_is_dominated(1.9, 4.8, grid) is True
    assert trigger_is_dominated(2.1, 4.9, grid) is False


def test_event_calibration_keeps_seed_repeats_separate():
    rows = pd.DataFrame({
        "event_id": ["a", "b", "a", "b"],
        "experiment_id": ["e1", "e2", "e1", "e2"],
        "representation": ["model"] * 4,
        "seed": [0, 0, 1, 1],
        "target": [1.0, 2.0, 1.0, 2.0],
        "predicted_target": [1.1, 1.9, 1.2, 1.8],
    })

    metrics = event_calibration_metrics(rows, ["target"])

    assert metrics.events.tolist() == [2, 2]
    assert metrics.seed.tolist() == [0, 1]


def test_actual_action_ch_calibration_reuses_shared_objective_formula():
    target = {
        "defrost_event_electricity_observed_kwh": 1.0,
        "defrost_event_net_heat_observed_kwh": -1.0,
        "defrost_event_compressor_electricity_observed_kwh": .5,
        "defrost_event_duration_observed_minutes": 30.0,
    }
    starts = pd.DataFrame([{
        "event_id": "a", "cycle_name": "a", "experiment_id": "e",
        "candidate_defrost_time": pd.Timestamp("2026-01-01 02:00"),
        "heating_accounting_start": pd.Timestamp("2026-01-01 00:00"),
        "pre_defrost_heat_kwh": 10.0, "pre_defrost_electricity_kwh": 5.0,
        "pre_defrost_compressor_electricity_kwh": 2.0,
        "pre_defrost_heat_measurement_valid": True,
        "pre_defrost_electricity_measurement_valid": True,
        "pre_defrost_compressor_electricity_measurement_valid": True,
        "pre_defrost_feature_window_valid": True,
        **target,
    }])
    predictions = starts[["event_id", "cycle_name", "experiment_id", *target]].copy()
    predictions["representation"] = "visual"
    predictions["seed"] = 0
    predictions["predicted_defrost_event_electricity_observed_kwh"] = 2.0
    predictions["predicted_defrost_event_net_heat_observed_kwh"] = -2.0
    predictions["predicted_defrost_event_compressor_electricity_observed_kwh"] = .6
    predictions["predicted_defrost_event_duration_observed_minutes"] = 60.0

    result = actual_action_ch_calibration(predictions, starts).iloc[0]

    assert result.observed_c == pytest.approx(9 / 6)
    assert result.observed_h == pytest.approx(9 / 2.5)
    assert result.predicted_c == pytest.approx(8 / 7)
    assert result.predicted_h == pytest.approx(8 / 3)


def test_relation_stability_compares_s4_with_its_fold_matched_bce():
    time = pd.date_range("2026-01-01", periods=5, freq="min")
    common = {
        "cycle_name": "frost_cycle_001", "heldout_experiment": "exp_20260101",
        "seed": 0, "image_time": time, "teacher_time": time[2],
    }
    matched = pd.DataFrame({**common, "method": "s1", "logit": [-1., -.2, .1, -.1, .2]})
    relation = pd.DataFrame({**common, "method": "s4", "logit": [-1., -.9, -.8, .1, .2]})
    selected = pd.DataFrame({
        "heldout_experiment": ["exp_20260101"], "selected_for_s4": ["s1"]
    })

    result = relation_stability_metrics(
        pd.concat([matched, relation], ignore_index=True), selected, local_window_minutes=2
    ).iloc[0]

    assert result.reference_method == "s1"
    assert result.zero_crossings_matched == 3
    assert result.zero_crossings_s4 == 1
    assert result.repeated_crossings_matched == 2
    assert result.repeated_crossings_s4 == 0
    assert result.local_total_variation_matched == pytest.approx(1.6)
    assert result.local_total_variation_s4 == pytest.approx(1.2)
    assert result.median_absolute_step_s4 < result.median_absolute_step_matched


def test_first_positive_is_primary_and_retains_misses_and_unknown_economics():
    predictions, teachers = evidence()
    cycles, frames, summary = evaluate_predictions(predictions, teachers)
    assert cycles.trigger_error_minutes.tolist()[0] == -1
    assert pd.isna(cycles.trigger_time.iloc[1])
    assert cycles.trigger_domain.iloc[0] == "unknown"
    assert pd.isna(cycles.delta_c.iloc[0])
    assert cycles.delta_c.iloc[2] == pytest.approx(0.)
    assert pd.isna(cycles.trigger_extra_observed_delay_minutes.iloc[0])
    assert cycles.trigger_extra_observed_delay_minutes.iloc[2] == 0
    assert pd.isna(cycles.trigger_extra_observed_delay_minutes.iloc[1])
    assert cycles.persistent_low.tolist() == [False, True, False]
    assert frames.frame_count.sum() == 6  # fractional knee is not a hard class
    assert summary.cycles.iloc[0] == 3
    assert summary.audited_cycles.iloc[0] == 3
    assert summary.no_trigger_fraction.iloc[0] == 1 / 3
    assert summary.brackets_knee_trigger_fraction.iloc[0] == 2 / 3
    assert summary.brackets_knee_median_error_minutes.iloc[0] == pytest.approx(-.5)


def test_abstaining_teacher_keeps_native_replay_denominator():
    predictions, teachers = evidence()
    absent = predictions.cycle_name.eq("frost_cycle_003")
    predictions.loc[absent, "teacher_time"] = pd.NaT
    predictions.loc[absent, "target"] = np.nan
    teachers = teachers.loc[~teachers.cycle_name.eq("frost_cycle_003")]
    cycles, _, summary = evaluate_predictions(predictions, teachers)
    assert len(cycles) == 3
    assert cycles.trigger_time.notna().sum() == 1
    assert summary.teacher_covered_cycles.iloc[0] == 2
    assert summary.cycles.iloc[0] == 2
    assert summary.audited_cycles.iloc[0] == 3
    assert summary.evaluation_cycles.iloc[0] == 2
    assert cycles.observation_status.iloc[2] == "teacher_absent"
    assert not cycles.evaluation_included.iloc[2]
    assert pd.isna(cycles.trigger_time.iloc[2])
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
    assert (tmp_path / "trigger_error_two_of_three_front.png").is_file()
    assert {path.stem for path in (tmp_path / "cycles").glob("*.png")} == {
        "frost_cycle_001", "frost_cycle_002", "frost_cycle_003"
    }
    assert not list((tmp_path / "cycles").glob("*.svg"))
    assert not (tmp_path / "representative_streams.png").exists()
    assert (tmp_path / "cycle_streams.csv").is_file()
    assert (tmp_path / "pareto_candidates.csv").is_file()


def test_shared_plotter_writes_only_evaluation_included_cycles(tmp_path):
    predictions, teachers = evidence()
    excluded = predictions.cycle_name.eq("frost_cycle_003")
    predictions.loc[excluded, "teacher_time"] = pd.NaT
    predictions.loc[excluded, "target"] = np.nan
    teachers = teachers.loc[~teachers.cycle_name.eq("frost_cycle_003")]
    cycle_output = tmp_path / "cycles"
    cycle_output.mkdir()
    (cycle_output / "stale.png").write_bytes(b"old")

    render_figures(
        predictions,
        teachers,
        pd.DataFrame(),
        pd.DataFrame(),
        tmp_path,
    )

    assert {path.stem for path in cycle_output.glob("*.png")} == {
        "frost_cycle_001",
        "frost_cycle_002",
    }


def test_neural_sensitivity_keeps_independent_knees_and_missing_selection(tmp_path):
    time = pd.date_range("2026-01-01", periods=4, freq="10s")
    rows = pd.DataFrame({
        "row_id": [f"a:{index}" for index in range(4)],
        "cycle_name": "frost_cycle_001", "experiment_id": "exp_20260101",
        "heldout_experiment": "exp_20260101", "candidate_defrost_time": time,
        "is_teacher_candidate": True, "teacher_time": time[1],
        "neural_teacher_time": pd.NaT,
        "defrost_event_electricity_kwh": [1., 2., 3., 4.],
        "neural_defrost_event_electricity_kwh": [1.1, 2.1, 3.1, 4.1],
        "defrost_event_net_heat_kwh": [-1., -.8, -.6, -.4],
        "neural_defrost_event_net_heat_kwh": [-.9, -.7, -.5, -.3],
        "defrost_event_compressor_electricity_kwh": [.2, .3, .4, .5],
        "neural_defrost_event_compressor_electricity_kwh": [.3, .4, .5, .6],
        "defrost_event_duration_minutes": [4., 5., 6., 7.],
        "neural_defrost_event_duration_minutes": [4.1, 5.1, 6.1, 7.1],
        "cycle_cop": [2., 2.1, 2.2, 2.3],
        "neural_cycle_cop": [2.1, 2.2, 2.3, 2.4],
        "cycle_heating_rate_kw": [5., 5.1, 5.2, 5.3],
        "neural_cycle_heating_rate_kw": [5.1, 5.2, 5.3, 5.4],
    })

    knees = render_neural_sensitivity(rows, tmp_path)

    assert knees.selection_status.tolist() == ["ridge_only"]
    assert knees.knee_difference_minutes.isna().all()
    assert (tmp_path / "neural_outcome_propagation.png").is_file()
    assert (tmp_path / "neural_pareto_knee_difference.png").is_file()
    assert (tmp_path / "neural_sensitivity_source.csv").is_file()
    assert (tmp_path / "neural_knees_by_cycle.csv").is_file()


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
    assert summary.cycles.iloc[0] == 1
    assert summary.audited_cycles.iloc[0] == 3
    assert summary.evaluation_cycles.iloc[0] == 1
    assert summary.triggered_cycles.iloc[0] == 1
    assert summary.no_trigger_fraction.iloc[0] == 0
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
    predictions.loc[selected, "logit"] = [-1., 1., 1.]
    predictions.loc[selected, "prediction"] = [0, 1, 1]
    predictions.loc[selected, "target"] = [0, 1, 1]
    cycles, frames, summary = evaluate_predictions(predictions, teachers)
    row = cycles.loc[cycles.cycle_name.eq("frost_cycle_003")].iloc[0]
    assert "nearest_pre_knee_relative_minutes" in cycles
    assert row.nearest_pre_knee_relative_minutes == -1
    assert row.nearest_post_knee_relative_minutes == 9
    assert row.trigger_extra_observed_delay_minutes == 0
    assert row.trigger_error_minutes == 9
    assert summary.cycles.iloc[0] == 3 and frames.frame_count.sum() == 7
    figures = []
    monkeypatch.setattr(
        pareto_learning, "_export", lambda figure, path, *args: figures.append(figure)
    )
    pareto_learning._cycle_figures(
        predictions.loc[selected], cycles.loc[cycles.cycle_name.eq("frost_cycle_003")], tmp_path
    )
    lines = [line for line in figures[0].axes[0].lines if line.get_marker() == "."]
    assert sorted(len(line.get_xdata()) for line in lines) == [1, 2]
    assert sum(len(line.get_xdata()) for line in lines) == 3
    assert all(np.diff(line.get_xdata()).max(initial=0) <= .75 for line in lines)


def test_cycle_figure_uses_saved_grid_and_keeps_outlying_trigger(tmp_path, monkeypatch):
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
    def original_renderer(axis, values, origin, **kwargs):
        seen.append((values.copy(), kwargs))
        axis.set(xlim=(1, 3), ylim=(4, 6))
    monkeypatch.setattr(pareto_learning, "plot_cop_heating_rate_pareto", original_renderer)
    monkeypatch.setattr(pareto_learning, "_export", lambda fig, path, *args: figures.append(fig))
    rb = pd.DataFrame({"cycle_name": ["frost_cycle_003"], "t_RB": [predictions.image_time.iloc[0]]})
    pareto_learning._cycle_figures(
        predictions, cycles, tmp_path, teacher_curves=grid, rb_triggers=rb,
        repeat_trigger_legends=False,
    )
    assert len(seen) == 2
    assert [kwargs["local"] for _, kwargs in seen] == [False, True]
    assert [kwargs["local_window_minutes"] for _, kwargs in seen] == [15, 15]
    assert all(kwargs["rb_time"] == rb.t_RB.iloc[0] for _, kwargs in seen)
    assert seen[0][0].is_cop_heating_rate_pareto_point.tolist() == [True, True, False]
    assert seen[0][0].is_selected_pareto_point.tolist() == [False, True, False]
    assert figures[0].axes[1].get_xlim()[1] > 9
    assert figures[0].get_size_inches().tolist() == pytest.approx([7.2, 5.0])
    assert figures[0].axes[0].get_position().width > 1.7 * figures[0].axes[1].get_position().width
    assert {text.get_text() for axis in figures[0].axes[:3] for text in axis.texts} >= {
        "a", "b", "c"
    }
    assert "Baseline" not in figures[0].axes[1].get_legend_handles_labels()[1]
    assert "Baseline" not in figures[0].axes[2].get_legend_handles_labels()[1]
    assert r"$\rightarrow$" in {text.get_text() for text in figures[0].axes[2].texts}
    assert all("seed" not in text.get_text() for axis in figures[0].axes for text in axis.texts)
    assert (tmp_path / "pareto_candidates.csv").exists()


def test_incomplete_rgb_suppresses_selected_and_statistics(tmp_path, monkeypatch):
    from plots import pareto_learning

    predictions, teachers = evidence()
    selected = predictions.cycle_name.eq("frost_cycle_003")
    predictions = predictions.loc[~selected | predictions.image_time.lt(predictions.teacher_time)]
    cycles, _, summary = evaluate_predictions(predictions, teachers)
    incomplete = cycles.loc[cycles.cycle_name.eq("frost_cycle_003")].iloc[0]
    assert incomplete.observation_status == "ends_before_knee"
    assert not incomplete.evaluation_included
    assert pd.isna(incomplete.trigger_time)
    assert summary.evaluation_cycles.iloc[0] == 2

    grid = evidence()[0].loc[selected].copy().assign(
        candidate_defrost_time=lambda x: x.image_time,
        is_teacher_candidate=True, is_knee=[False, True, False],
        pareto_selection_score=[.1, .9, np.nan],
        cycle_cop=[2., 2.1, 2.2], cycle_heating_rate_kw=[5., 5.1, 5.2],
    )
    seen = []
    monkeypatch.setattr(
        pareto_learning, "plot_cop_heating_rate_pareto",
        lambda _axis, values, _origin, **_kwargs: seen.append(values.copy()),
    )
    monkeypatch.setattr(pareto_learning, "_export", lambda *_args: None)
    pareto_learning._cycle_figures(
        predictions.loc[predictions.cycle_name.eq("frost_cycle_003")],
        cycles.loc[cycles.cycle_name.eq("frost_cycle_003")], tmp_path,
        teacher_curves=grid,
    )
    assert len(seen) == 2
    assert not any(frame.is_selected_pareto_point.any() for frame in seen)
    audit = pd.read_csv(tmp_path / "pareto_triggers.csv")
    assert not audit.evaluation_included.iloc[0]
    assert audit.observation_status.iloc[0] == "ends_before_knee"


def test_cycle_legends_are_above_axes_and_labels_omit_seed(tmp_path, monkeypatch):
    from plots import pareto_learning

    predictions, teachers = evidence()
    cycles, _, _ = evaluate_predictions(predictions, teachers)
    figures = []
    monkeypatch.setattr(
        pareto_learning, "_export", lambda figure, _path, *args: figures.append(figure)
    )
    pareto_learning._cycle_figures(
        predictions.loc[predictions.cycle_name.eq("frost_cycle_001")],
        cycles.loc[cycles.cycle_name.eq("frost_cycle_001")], tmp_path,
    )
    legend = figures[0].axes[0].get_legend()
    assert legend.get_bbox_to_anchor()._bbox.y0 > 1
    assert all("seed" not in text.get_text() for text in legend.get_texts())


def test_decision_summary_includes_boxplots(tmp_path, monkeypatch):
    from plots import pareto_learning

    predictions, teachers = evidence()
    cycles, _, _ = evaluate_predictions(predictions, teachers)
    figures = []
    monkeypatch.setattr(pareto_learning, "_export", lambda figure, _path: figures.append(figure))

    pareto_learning._summary_plot(cycles, tmp_path)

    assert all(axis.patches for axis in figures[0].axes)


def test_outcome_figures_compare_neural_representation_with_ridge(tmp_path):
    targets = (
        "defrost_event_electricity_observed_kwh",
        "defrost_event_net_heat_observed_kwh",
        "defrost_event_compressor_electricity_observed_kwh",
        "defrost_event_duration_observed_minutes",
    )
    rows = []
    for representation in ("visual", "nonvisual"):
        for index in range(4):
            row = {
                "event_id": str(index), "cycle_name": str(index),
                "experiment_id": f"exp_{index % 2}", "representation": representation,
            }
            for target_index, target in enumerate(targets):
                row[target] = index + target_index
                row[f"predicted_{target}"] = index + target_index + .1
                row[f"ridge_predicted_{target}"] = index + target_index + .2
            rows.append(row)
    losses = pd.DataFrame([
        {"epoch": epoch, "split": split, "loss": 1 / epoch,
         "representation": representation, "heldout_experiment": "exp_0"}
        for representation in ("visual", "nonvisual")
        for split in ("train", "validation") for epoch in (1, 2)
    ])
    render_outcome_figures(pd.DataFrame(rows), losses, tmp_path)
    assert (tmp_path / "outcome_prediction.png").is_file()
    assert (tmp_path / "outcome_training_loss.png").is_file()
    metrics = pd.read_csv(tmp_path / "outcome_metrics.csv")
    assert set(metrics.model) == {"visual", "nonvisual", "ridge"}


def test_outcome_figures_keep_fixed_seeds_separate(tmp_path, monkeypatch):
    from plots import pareto_learning

    targets = (
        "defrost_event_electricity_observed_kwh",
        "defrost_event_net_heat_observed_kwh",
        "defrost_event_compressor_electricity_observed_kwh",
        "defrost_event_duration_observed_minutes",
    )
    predictions = []
    for seed in (0, 1):
        row = {
            "event_id": f"event-{seed}", "cycle_name": f"cycle-{seed}",
            "experiment_id": "exp_0", "representation": "multimodal_time_linear",
            "seed": seed,
        }
        for target in targets:
            row[target] = 1.
            row[f"predicted_{target}"] = 1. + seed
        predictions.append(row)
    losses = pd.DataFrame([
        {"epoch": epoch, "split": split, "loss": 1 / epoch,
         "representation": "multimodal_time_linear", "seed": seed,
         "heldout_experiment": "exp_0"}
        for seed in (0, 1) for split in ("train", "validation") for epoch in (1, 2)
    ])

    figures = []
    monkeypatch.setattr(
        pareto_learning, "_export", lambda figure, path: figures.append((figure, path))
    )
    render_outcome_figures(pd.DataFrame(predictions), losses, tmp_path)

    metrics = pd.read_csv(tmp_path / "outcome_metrics.csv")
    assert set(metrics.seed) == {0, 1}
    assert len(metrics) == 8
    outcome = next(figure for figure, path in figures if path.name == "outcome_prediction")
    labels = outcome.axes[0].get_legend_handles_labels()[1]
    assert labels == ["Time-linear event head · seed 0", "Time-linear event head · seed 1"]


def test_readable_method_names_cover_every_stopping_recipe():
    assert {"s0", "s1", "s2", "s3", "s4", "n2", "s1_neural"} < set(METHOD_NAMES)
    assert METHOD_NAMES["s3"].name == "multimodal_ridge_cho_history"
    assert METHOD_NAMES["s3"].label == "RGB+sensor + Ridge C/H/O history"
    assert METHOD_NAMES["ridge_head_neural_ch"].label == "Ridge head · Neural C/H"


def test_readable_method_names_cover_every_vcnet_recipe():
    from defrost_event_models.ridge_models import OUTCOME_TARGETS
    from plots.pareto_learning import OUTCOME_LABELS, VCNET_LABELS, VCNET_STYLES

    assert set(VCNET_LABELS) == set(VCNET_STYLES)
    assert VCNET_LABELS["multimodal_time_linear"] == "Time-linear event head"
    assert VCNET_LABELS["multimodal_time_varying_regularized"] == (
        "Time-varying + coefficient regularization"
    )
    assert set(OUTCOME_LABELS) == set(OUTCOME_TARGETS.values())
    assert OUTCOME_LABELS["defrost_event_duration_observed_minutes"] == (
        "Event duration [min]"
    )


def test_transfer_selection_uses_failures_tail_then_simple_tie_break():
    from plots.pareto_learning import select_transfer_method

    rows = []
    for method in ("d32_ridge_ch_mlp", "t32_ridge_ch_mlp", "raw_ridge_ch_mlp"):
        for seed in (0, 1):
            for cycle, loss in enumerate((0.5, 1.0, 2.0)):
                rows.append({
                    "method": method, "seed": seed, "strategy": "first_positive",
                    "heldout_experiment": f"exp_{cycle}", "cycle_name": f"cycle_{cycle}",
                    "evaluation_status": "evaluated",
                    "maximum_relative_performance_loss_percent": loss,
                })

    selected, summary = select_transfer_method(pd.DataFrame(rows))

    assert selected == "raw_ridge_ch_mlp"
    assert summary.method.nunique() == 3


def test_vcnet_factorial_interaction_is_zero_for_additive_errors():
    from defrost_event_models.ridge_models import OUTCOME_TARGETS
    from plots.pareto_learning import _vcnet_event_pairs

    offsets = {
        "multimodal_time_linear": 0.,
        "multimodal_time_linear_z64": .1,
        "multimodal_time_linear_z32_curvature": -.2,
        "multimodal_time_linear_z64_curvature": -.1,
    }
    rows = []
    for representation, offset in offsets.items():
        for experiment, baseline in (("exp_a", 1.), ("exp_b", 2.)):
            row = {
                "representation": representation, "seed": 0,
                "experiment_id": experiment,
            }
            for target in OUTCOME_TARGETS.values():
                row[f"standardized_absolute_error_{target}"] = baseline + offset
            rows.append(row)

    _, comparisons = _vcnet_event_pairs(pd.DataFrame(rows))

    interaction = comparisons.loc[comparisons.comparison.eq("capacity_x_curvature")]
    assert len(interaction) == 4
    np.testing.assert_allclose(interaction.estimate, 0., atol=1e-12)
    capacity = comparisons.loc[comparisons.comparison.eq("capacity")]
    assert capacity.mae_ratio.notna().all()
    assert set(capacity.noninferiority_5_percent) <= {
        "supported", "insufficient_evidence", "exceeds_margin"
    }


def test_vcnet_event_pairs_require_standardized_errors():
    from plots.pareto_learning import _vcnet_event_pairs

    with pytest.raises(ValueError, match="standardized event errors"):
        _vcnet_event_pairs(pd.DataFrame([{
            "representation": "multimodal_time_linear", "seed": 0,
            "experiment_id": "exp_a",
        }]))


def test_vcnet_core_event_contrasts_share_four_model_support():
    from defrost_event_models.ridge_models import OUTCOME_TARGETS
    from plots.pareto_learning import _vcnet_event_pairs

    methods = (
        "multimodal_time_linear", "multimodal_time_linear_z64",
        "multimodal_time_linear_z32_curvature",
        "multimodal_time_linear_z64_curvature",
    )
    rows = []
    for experiment in ("exp_a", "exp_b"):
        for method in methods:
            if experiment == "exp_b" and method.endswith("z32_curvature"):
                continue
            row = {"representation": method, "seed": 0, "experiment_id": experiment}
            for target in OUTCOME_TARGETS.values():
                row[f"standardized_absolute_error_{target}"] = 1.
            rows.append(row)

    _, comparisons = _vcnet_event_pairs(pd.DataFrame(rows))

    core = comparisons.loc[comparisons.comparison.isin(
        ["capacity", "curvature_z32", "capacity_x_curvature"]
    )]
    assert set(core.experiments) == {1}


def test_vcnet_contrasts_allow_an_incomplete_incremental_run():
    from defrost_event_models.ridge_models import OUTCOME_TARGETS
    from plots.pareto_learning import (
        _vcnet_consequence_pairs,
        _vcnet_event_pairs,
        _vcnet_roughness_pairs,
    )

    event = {"representation": "multimodal_time_linear", "seed": 0,
             "experiment_id": "exp_a"}
    for target in OUTCOME_TARGETS.values():
        event[f"standardized_absolute_error_{target}"] = 1.
    _, event_pairs = _vcnet_event_pairs(pd.DataFrame([event]))
    consequence_pairs = _vcnet_consequence_pairs(pd.DataFrame([{
        "representation": "multimodal_time_linear", "seed": 0,
        "heldout_experiment": "exp_a", "cycle_name": "cycle_a",
        "maximum_relative_performance_loss_percent": 1.,
    }]), replicates=10)
    roughness_pairs = _vcnet_roughness_pairs(pd.DataFrame([{
        "representation": "multimodal_time_linear", "seed": 0,
        "experiment_id": "exp_a", "cycle_name": "cycle_a", "delta_seconds": 10,
        "latent_second_difference_rms_median": 1.,
        "outcome_second_difference_rms_median": 1.,
    }]), replicates=10)

    assert event_pairs.empty
    assert consequence_pairs.empty
    assert roughness_pairs.empty


def test_vcnet_state_runs_add_only_d32_from_reference(tmp_path):
    from plots.pareto_learning import _vcnet_state_runs

    state = tmp_path / "state"
    reference = tmp_path / "reference"
    for root, method in (
        (state, "multimodal_time_linear_z64"),
        (state, "multimodal_time_varying"),
        (reference, "multimodal_time_linear"),
        (reference, "multimodal_time_varying_regularized"),
    ):
        folder = root / method / "seed_0"
        folder.mkdir(parents=True)
        (folder / "outcome_predictions.csv").write_text("representation,seed\n")

    runs = _vcnet_state_runs(state, reference)

    assert {(path.parent.name, path.name) for path in runs} == {
        ("multimodal_time_linear_z64", "seed_0"),
        ("multimodal_time_linear", "seed_0"),
    }


def test_vcnet_consequence_bootstrap_preserves_experiment_clusters():
    from plots.pareto_learning import _vcnet_consequence_pairs

    rows = []
    effects = {
        "multimodal_time_linear": 0.,
        "multimodal_time_linear_z64": 1.,
        "multimodal_time_linear_z32_curvature": -2.,
        "multimodal_time_linear_z64_curvature": -1.,
    }
    for method, effect in effects.items():
        for experiment, baseline in (("a", 2.), ("b", 4.)):
            for cycle in range(2):
                rows.append({
                    "representation": method, "seed": 0,
                    "heldout_experiment": experiment,
                    "cycle_name": f"{experiment}-{cycle}",
                    "maximum_relative_performance_loss_percent": baseline + effect,
                })

    result = _vcnet_consequence_pairs(pd.DataFrame(rows), replicates=100)

    interaction = result.loc[
        result.comparison.eq("capacity_x_curvature")
        & result.statistic.eq("median")
    ]
    assert len(interaction) == 1
    assert interaction.estimate.iloc[0] == pytest.approx(0.)


def test_vcnet_consequence_core_contrasts_share_four_model_support():
    from plots.pareto_learning import _vcnet_consequence_pairs

    methods = (
        "multimodal_time_linear", "multimodal_time_linear_z64",
        "multimodal_time_linear_z32_curvature",
        "multimodal_time_linear_z64_curvature",
    )
    rows = [{
        "representation": method, "seed": 0,
        "heldout_experiment": experiment, "cycle_name": f"{experiment}-0",
        "maximum_relative_performance_loss_percent": 1.,
    } for experiment in ("exp_a", "exp_b") for method in methods
        if not (experiment == "exp_b" and method.endswith("z32_curvature"))]

    result = _vcnet_consequence_pairs(pd.DataFrame(rows), replicates=10)

    core = result.loc[result.comparison.isin(
        ["capacity", "curvature_z32", "capacity_x_curvature"]
    )]
    assert set(core.experiments) == {1}


def test_vcnet_roughness_factorial_interaction_is_zero_for_additive_values():
    from plots.pareto_learning import _vcnet_roughness_pairs

    offsets = {
        "multimodal_time_linear": 0.,
        "multimodal_time_linear_z64": 1.,
        "multimodal_time_linear_z32_curvature": -2.,
        "multimodal_time_linear_z64_curvature": -1.,
    }
    rows = pd.DataFrame([
        {
            "representation": method, "seed": 0, "experiment_id": experiment,
            "cycle_name": f"{experiment}-{cycle}", "delta_seconds": 10,
            "latent_second_difference_rms_median": baseline + offset,
            "outcome_second_difference_rms_median": 2 * (baseline + offset),
        }
        for method, offset in offsets.items()
        for experiment, baseline in (("a", 4.), ("b", 6.))
        for cycle in range(2)
    ])

    result = _vcnet_roughness_pairs(rows, replicates=100)

    interaction = result.loc[result.comparison.eq("capacity_x_curvature")]
    assert len(interaction) == 2
    np.testing.assert_allclose(interaction.estimate, 0., atol=1e-12)


def test_vcnet_roughness_core_contrasts_share_four_model_support():
    from plots.pareto_learning import _vcnet_roughness_pairs

    methods = (
        "multimodal_time_linear", "multimodal_time_linear_z64",
        "multimodal_time_linear_z32_curvature",
        "multimodal_time_linear_z64_curvature",
    )
    rows = pd.DataFrame([{
        "representation": method, "seed": 0, "experiment_id": experiment,
        "cycle_name": f"{experiment}-0", "delta_seconds": 10,
        "latent_second_difference_rms_median": 1.,
        "outcome_second_difference_rms_median": 1.,
    } for experiment in ("exp_a", "exp_b") for method in methods
        if not (experiment == "exp_b" and method.endswith("z32_curvature"))])

    result = _vcnet_roughness_pairs(rows, replicates=10)

    core = result.loc[result.comparison.isin(
        ["capacity", "curvature_z32", "capacity_x_curvature"]
    )]
    assert set(core.experiments) == {1}


def test_state_pathway_figure_keeps_three_scales_and_source_data(tmp_path, monkeypatch):
    from plots import pareto_learning

    rows = pd.DataFrame([
        {
            "representation": "multimodal_time_linear", "seed": 0,
            "heldout_experiment": "e", "experiment_id": "e", "cycle_name": "c",
            "delta_seconds": seconds, "pathway": pathway,
            "target": "defrost_event_electricity_observed_kwh",
            "standardized_rms": value,
        }
        for seconds in (10, 60, 300)
        for pathway, value in (("rgb", 1.), ("sensor", 2.), ("ledger", 3.),
                               ("quality", 4.), ("time", 5.), ("interaction", 1.5))
    ])
    figures = []
    monkeypatch.setattr(
        pareto_learning, "_export", lambda figure, path: figures.append(path)
    )

    pareto_learning._render_state_pathways(rows, tmp_path)

    source = pd.read_csv(tmp_path / "state_pathway_source.csv")
    assert set(source.delta_seconds) == {10, 60, 300}
    assert set(source.pathway) == {
        "rgb", "sensor", "ledger", "quality", "time", "interaction"
    }
    assert figures == [tmp_path / "state_pathway_sensitivity"]


def test_state_intervention_renders_median_and_p90_consequence_columns(
    tmp_path, monkeypatch
):
    from defrost_event_models.ridge_models import OUTCOME_TARGETS
    from plots import pareto_learning

    methods = (
        "multimodal_time_linear", "multimodal_time_linear_z64",
        "multimodal_time_linear_z32_curvature",
        "multimodal_time_linear_z64_curvature",
    )
    event = pd.DataFrame([
        {"representation": method, "seed": seed, "experiment_id": "e",
         "target": target, "mae": 1.}
        for method in methods for seed in (0, 1) for target in OUTCOME_TARGETS.values()
    ])
    roughness = pd.DataFrame([
        {"representation": method, "seed": seed, "experiment_id": "e",
         "cycle_name": "c", "delta_seconds": 10,
         "latent_second_difference_rms_median": 1.,
         "outcome_second_difference_rms_median": 1.}
        for method in methods for seed in (0, 1)
    ])
    consequences = pd.DataFrame([
        {"representation": method, "seed": seed, "heldout_experiment": "e",
         "cycle_name": "c", "maximum_relative_performance_loss_percent": 1.}
        for method in methods for seed in (0, 1)
    ])
    figures = []
    monkeypatch.setattr(
        pareto_learning, "_export", lambda figure, path: figures.append(path)
    )

    pareto_learning._render_state_intervention(
        event, roughness, consequences, tmp_path
    )

    assert figures == [tmp_path / "state_intervention"]


def test_cross_input_comparisons_keep_fixed_head_direction_and_neither_is_not_same():
    times = pd.date_range("2026-01-01", periods=3, freq="min")
    logits = {
        "ridge_head_ridge_ch": [-1., 1., 1.],
        "ridge_head_neural_ch": [-1., -1., 1.],
        "neural_head_ridge_ch": [-1., -1., -1.],
        "neural_head_neural_ch": [-1., -1., -1.],
    }
    predictions = pd.concat([
        pd.DataFrame({
            "row_id": [f"r{i}" for i in range(3)], "cycle_name": "cycle",
            "heldout_experiment": "exp", "method": method,
            "image_time": times, "teacher_time": times[1], "logit": values,
        })
        for method, values in logits.items()
    ], ignore_index=True)
    trigger_times = {
        "ridge_head_ridge_ch": times[1],
        "ridge_head_neural_ch": times[2],
        "neural_head_ridge_ch": pd.NaT,
        "neural_head_neural_ch": pd.NaT,
    }
    consequences = pd.DataFrame([
        {
            "method": method, "strategy": "first_positive",
            "heldout_experiment": "exp", "cycle_name": "cycle",
            "trigger_time": trigger, "maximum_relative_performance_loss_percent": loss,
        }
        for method, trigger, loss in (
            ("ridge_head_ridge_ch", trigger_times["ridge_head_ridge_ch"], 1.),
            ("ridge_head_neural_ch", trigger_times["ridge_head_neural_ch"], 2.),
            ("neural_head_ridge_ch", pd.NaT, np.nan),
            ("neural_head_neural_ch", pd.NaT, np.nan),
        )
    ])

    cycles, frames = cross_input_comparisons(predictions, consequences)

    ridge = cycles.loc[cycles.head_source.eq("ridge")].iloc[0]
    neural = cycles.loc[cycles.head_source.eq("neural")].iloc[0]
    assert ridge.trigger_change_minutes == 1.
    assert ridge.loss_change_percent == 1.
    assert ridge.trigger_status == "both_trigger"
    assert not ridge.same_trigger_frame
    assert neural.trigger_status == "neither_trigger"
    assert not neural.same_trigger_frame
    assert frames.loc[frames.head_source.eq("ridge"), "sign_flip"].sum() == 1


def test_maximum_relative_performance_loss_only_penalizes_the_worst_drop():
    assert maximum_relative_performance_loss(2., 5., 2.1, 4.9) == pytest.approx(2.)
    assert maximum_relative_performance_loss(2., 5., 2.1, 5.1) == 0.
    assert maximum_relative_performance_loss(2., 5., 1.92, 4.9) == pytest.approx(4.)
    assert np.isnan(maximum_relative_performance_loss(0., 5., 2., 5.))


def test_performance_consequences_scores_first_trigger_on_shared_ridge_curve(tmp_path):
    time = pd.date_range("2026-01-01", periods=5, freq="min")
    predictions = pd.DataFrame({
        "row_id": [f"row-{index}" for index in range(5)],
        "cycle_name": "frost_cycle_001", "experiment_id": "exp_20260101",
        "heldout_experiment": "exp_20260101", "method": "s1_neural", "seed": 0,
        "image_time": time, "teacher_time": time[0],
        "logit": [-1., 1., -1., -1., 1.],
    })
    evaluation = pd.DataFrame({
        "row_id": predictions.row_id, "cycle_name": "frost_cycle_001",
        "experiment_id": "exp_20260101", "heldout_experiment": "exp_20260101",
        "candidate_defrost_time": time, "is_knee": [True, False, False, False, False],
        "is_teacher_candidate": [True, True, True, True, True],
        "cycle_cop": [2., 1.92, 1.94, 1.98, 1.994],
        "cycle_heating_rate_kw": [5., 4.9, 4.95, 5.01, 5.01],
        "cycle_evaporator_capacity_kw": [3., 2.9, 2.95, 3.01, 3.02],
        "cycle_cop_eligible": [True, False, True, True, True],
        "cycle_heating_rate_kw_eligible": [True, False, True, True, True],
        "cycle_evaporator_capacity_kw_eligible": True,
        "cycle_cop_measurements_valid": True,
        "cycle_heating_rate_kw_measurements_valid": True,
        "cycle_evaporator_capacity_kw_measurements_valid": True,
        "cycle_cop_physically_valid": True,
        "cycle_heating_rate_kw_physically_valid": True,
        "cycle_evaporator_capacity_kw_physically_valid": True,
        "defrost_event_electricity_in_training_domain": False,
        "defrost_event_net_heat_in_training_domain": False,
        "defrost_event_duration_in_training_domain": False,
        "cycle_cop_uses_model_extrapolation": True,
        "cycle_heating_rate_kw_uses_model_extrapolation": True,
        "cycle_evaporator_capacity_kw_uses_model_extrapolation": True,
        "pre_defrost_electricity_uses_measurement_reconstruction": False,
        "pre_defrost_heat_uses_measurement_reconstruction": False,
    })

    result = performance_consequences(
        predictions, evaluation, methods=("s1_neural",), strategies=("first_positive",)
    ).iloc[0]

    assert result.trigger_time == time[1]
    assert result.trigger_c == pytest.approx(1.92)
    assert result.delta_c_percent == pytest.approx(4.)
    assert result.delta_h_percent == pytest.approx(2.)
    assert result.maximum_relative_performance_loss_percent == pytest.approx(4.)
    assert result.delta_o == pytest.approx(-.1)
    assert result.evaluation_status == "evaluated"
    assert result.trigger_uses_model_extrapolation
    assert result.dominated_by_teacher_grid


def test_performance_coverage_retains_unknown_cycles_in_fixed_denominator(
    tmp_path, monkeypatch
):
    from plots import pareto_learning

    rows = pd.DataFrame({
        "method": ["s0", "s0", "s1", "s1"], "seed": 0,
        "method_name": [METHOD_NAMES[m].name for m in ("s0", "s0", "s1", "s1")],
        "method_label": [METHOD_NAMES[m].label for m in ("s0", "s0", "s1", "s1")],
        "strategy": "first_positive", "cycle_name": ["a", "b", "a", "b"],
        "trigger_time": [pd.Timestamp("2026-01-01"), pd.NaT] * 2,
        "trigger_error_minutes": [0., np.nan, 1., 2.],
        "maximum_relative_performance_loss_percent": [0., np.nan, 1., 2.],
        "evaluation_status": ["evaluated", "no_trigger", "evaluated", "evaluated"],
    })
    figures = []
    monkeypatch.setattr(
        pareto_learning, "_export", lambda figure, path: figures.append((figure, path))
    )

    source, summary = render_performance_coverage(
        rows, "first_positive", tmp_path, denominator=2
    )

    assert summary.set_index("method").loc["s0", "coverage_at_5_percent"] == .5
    assert summary.set_index("method").loc["s1", "coverage_at_5_percent"] == 1.
    assert source.groupby("method").coverage_fraction.max().to_dict() == {
        "s0": .5, "s1": 1.
    }
    assert all(
        group.coverage_fraction.is_monotonic_increasing
        for _, group in source.groupby("method")
    )
    assert len(figures) == 1


def test_overview_figure_contract_uses_four_main_and_five_diagnostic_outputs():
    from plots.pareto_learning import OVERVIEW_FIGURE_STEMS, OVERVIEW_METHOD_LABELS

    assert OVERVIEW_FIGURE_STEMS == (
        "01_objectives_and_timing",
        "02_ridge_observed_calibration",
        "03_timing_and_consequence",
        "04_online_policy_evaluation",
        "diagnostic_neural_event_calibration",
        "diagnostic_online_information_and_probe",
        "diagnostic_curvature_effects",
        "reference_outcome_model_shift_ecdf",
        "reference_fixed_predictor_input_swap",
    )
    assert len(set(OVERVIEW_FIGURE_STEMS)) == 9
    assert OVERVIEW_METHOD_LABELS["s1"] == "Add current Ridge C/H"
    assert OVERVIEW_METHOD_LABELS["t32_ridge_ch_mlp"] == (
        "32D curvature-regularized state"
    )
    assert OVERVIEW_METHOD_LABELS["raw_ridge_ch_mlp"] == "Pre-compression features"
    assert not any(
        label in {"S0", "S1", "S2", "S3", "S4", "N2", "D32", "T32", "Raw"}
        for label in OVERVIEW_METHOD_LABELS.values()
    )


def test_vcnet_pareto_consequence_uses_common_ridge_values_without_time_penalty():
    from plots.pareto_learning import vcnet_pareto_consequences

    rows = pd.DataFrame({
        "cycle_name": ["a", "a"], "experiment_id": ["e", "e"],
        "heldout_experiment": ["e", "e"], "representation": ["method", "method"],
        "seed": [0, 0], "row_id": ["early", "late"],
        "candidate_defrost_time": pd.to_datetime(
            ["2026-01-01", "2026-01-01 02:00"], format="mixed"
        ),
        "teacher_time": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        "is_knee": [True, False], "neural_is_knee": [False, True],
        "cycle_cop": [4., 3.8], "cycle_heating_rate_kw": [10., 10.2],
        "cycle_evaporator_capacity_kw": [8., 7.5],
        "cycle_cop_measurements_valid": [True, True],
        "cycle_cop_physically_valid": [True, True],
        "cycle_heating_rate_kw_measurements_valid": [True, True],
        "cycle_heating_rate_kw_physically_valid": [True, True],
    })

    result = vcnet_pareto_consequences(rows).iloc[0]

    assert result.delta_time_minutes == 120
    assert result.d_c_percent == pytest.approx(5)
    assert result.d_h_percent == pytest.approx(-2)
    assert result.maximum_relative_performance_loss_percent == pytest.approx(5)


def test_vcnet_trajectory_metrics_use_elapsed_time_between_adjacent_grid_points():
    from plots.pareto_learning import vcnet_trajectory_metrics

    rows = pd.DataFrame({
        "cycle_name": ["a"] * 3, "experiment_id": ["e"] * 3,
        "heldout_experiment": ["e"] * 3, "representation": ["method"] * 3,
        "seed": [0] * 3, "replay": ["observed_state_and_time"] * 3,
        "is_teacher_candidate": [True] * 3,
        "candidate_defrost_time": pd.to_datetime([
            "2026-01-01 00:00", "2026-01-01 00:01", "2026-01-01 00:03"
        ]),
        "predicted_defrost_event_electricity_observed_kwh": [1., 3., 4.],
    })

    result = vcnet_trajectory_metrics(rows).iloc[0]

    assert result.adjacent_absolute_change_median == 1.5
    assert result.absolute_change_per_minute_median == 1.25
    assert result.total_variation == 3
    assert result.trajectory_span == 3


def test_online_headroom_plot_preserves_support_and_unscored_cycles(tmp_path, monkeypatch):
    from plots import pareto_learning

    rows = pd.DataFrame({
        "model": ["RB", "model_a", "RB", "model_a", "model_b", "model_c"],
        "cycle_name": ["a", "a", "b", "b", "c", "d"],
        "status": ["scored", "scored", "scored", "no_trigger", "scored", "scored"],
        "reference_cop": [2.2, 2.2, 2.4, 2.4, 2.2, 2.1],
        "baseline_rb_cop": [2., 2., 2., 2., 2., 2.],
        "trigger_cop": [2., 2.5, 2., np.nan, 2.6, .2],
        "outside_reference_support": [True, False, "False", "False", np.nan, False],
    })
    exported = []
    monkeypatch.setattr(
        pareto_learning,
        "_export",
        lambda figure, path, formats=("png",): exported.append((figure, path, formats)),
    )

    pareto_learning._render_headroom_comparison(rows, tmp_path)

    source = pd.read_csv(tmp_path / "headroom_points.csv")
    assert len(source) == len(rows)
    assert not source.loc[source.model.eq("RB"), "plot_included"].any()
    assert source.loc[source.cycle_name.eq("a") & source.model.eq("model_a"),
                      "support_state"].item() == "outside"
    assert source.loc[source.cycle_name.eq("b") & source.model.eq("model_a"),
                      "support_state"].item() == "in"
    assert source.loc[source.model.eq("model_b"), "support_state"].item() == "unknown"
    assert source.loc[source.model.eq("model_b"), "realized_gain_percent"].item() == pytest.approx(30.)
    assert source.loc[source.model.eq("model_b"), "reference_headroom_percent"].item() == pytest.approx(10.)
    assert source.loc[source.cycle_name.eq("b") & source.model.eq("model_a"),
                      "missing_reason"].item() == "no_trigger"
    figure, path, formats = exported[0]
    assert path == tmp_path / "headroom_comparison"
    assert formats == ("png", "pdf", "svg")
    assert figure.axes[0].get_ylim()[1] >= 30.
    assert figure.axes[0].get_ylim()[0] <= -90.
    assert figure.axes[0].get_xlim()[1] < 20.
    assert {"Realized = headroom", "Zero gain"} <= set(
        figure.axes[0].get_legend_handles_labels()[1]
    )
    assert "model_a: 1 available / 1 unscored" in "\n".join(
        text.get_text() for text in figure.texts
    )


def test_development_classification_renderer_uses_summary_without_cop_claims(
    tmp_path, monkeypatch
):
    from plots import pareto_learning

    summary = pd.DataFrame({
        "method": ["sensor", "shared"], "selected_fold_count": [3, 3],
        "cycle_weighted_average_precision": [.8, .7],
        "balanced_accuracy": [.75, .65], "macro_f1": [.7, .6],
        "recall": [.8, .7], "fpr": [.2, .3],
        "proposal_2of3_count": [8, 6], "near_hit_2of3_count": [7, 5],
        "unscoreable_2of3_count": [2, 4], "scored_cycle_count": [8, 6],
        "cohort_cycle_count": [10, 10], "proposal_coverage": [.8, .6],
        "near_hit_coverage": [.7, .5], "conditional_near_hit": [.875, 5 / 6],
    })
    exported = []
    monkeypatch.setattr(
        pareto_learning, "_export",
        lambda figure, path, formats=("png",): exported.append((figure, path, formats)),
    )

    source = pareto_learning.render_development_classification(summary, tmp_path)

    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "development_classification_source.csv"), summary
    )
    pd.testing.assert_frame_equal(source, summary)
    figure, path, formats = exported[0]
    assert path == tmp_path / "development_classification"
    assert formats == ("png", "pdf", "svg")
    assert len(figure.axes) == 2
    assert "COP" not in " ".join(
        axis.get_title() + axis.get_ylabel() for axis in figure.axes
    )
