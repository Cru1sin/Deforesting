import numpy as np
import pandas as pd


def test_fixed_headroom_clips_overfrequency_and_preserves_causal_missingness():
    from image_models.control_context import native_control_context, stopping_inputs

    start = pd.Timestamp("2026-01-01")
    raw = pd.DataFrame(
        dict(
            timestamp=pd.date_range(start, periods=5, freq="30s"),
            frequency_command_hz=90.0,
            water_temperature_setpoint=50.0,
            actual_frequency_hz=[80.0, 100.0, 110.0, np.nan, 70.0],
        )
    )
    actions = pd.DataFrame(
        dict(candidate_defrost_time=start + pd.to_timedelta([1, 31, 61, 91, 120, 151], unit="s"))
    )
    result = native_control_context(raw, actions, start)
    np.testing.assert_allclose(
        result.frequency_headroom_hz, [20.0, 0.0, 0.0, np.nan, np.nan, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        result.frequency_headroom_fraction, [0.2, 0.0, 0.0, np.nan, np.nan, np.nan], equal_nan=True
    )
    raw.loc[4, "actual_frequency_hz"] = 150.0
    pd.testing.assert_frame_equal(native_control_context(raw, actions, start), result)
    headroom = stopping_inputs("headroom100", "changes")
    command = stopping_inputs("command", "changes")
    assert headroom["RGB_headroom"] == ["frequency_headroom_hz"]
    assert headroom["RGB_control"] == [
        "water_temperature_setpoint",
        "frequency_headroom_hz",
        "frequency_change_60s_hz",
        "frequency_change_300s_hz",
    ]
    assert command["RGB_command"] == ["frequency_command_hz"]
    assert stopping_inputs("headroom_fraction", "changes")["RGB_headroom"] == [
        "frequency_headroom_fraction"
    ]


def test_cop_stopping_label_uses_first_disjoint_basin_and_never_moves_missing_event():
    from image_models.labels import label_actions

    time = pd.date_range("2026-01-01", periods=7, freq="30s")
    reference = pd.DataFrame(
        dict(
            cycle_name="a",
            experiment_id="d",
            candidate_defrost_time=time,
            on_action_grid=True,
            cycle_cop_eligible=True,
            cycle_cop=[1.0, 1.99, 1.0, 2.0, 2.0, 2.0, 1.0],
            t_RB=time[-1],
            input_valid=[True, False, True, True, True, True, True],
            compressor_frequency=80.0,
        )
    )
    labelled, intervals = label_actions(reference, label_rule="stopping")
    assert intervals.iloc[0].t_label == time[1]
    assert intervals.iloc[0].L == intervals.iloc[0].U == time[1]
    assert labelled.after_optimum_target.iloc[:2].tolist() == [0.0, 1.0]
    assert labelled.after_optimum_target.iloc[2:].isna().all()
    assert labelled.train_mask.tolist() == [True, True, False, False, False, False, False]
    assert labelled.near_optimal.tolist() == [False, True, False, True, True, True, False]
    changed = reference.assign(compressor_frequency=150.0, input_valid=True)
    repeated, _ = label_actions(changed, label_rule="stopping")
    np.testing.assert_allclose(
        repeated.after_optimum_target, labelled.after_optimum_target, equal_nan=True
    )


def test_control_context_freezes_completed_baseline_and_uses_only_past_commands():
    from image_models.control_context import native_control_context

    start = pd.Timestamp("2026-01-01")
    time = pd.date_range(start, periods=22, freq="30s")
    raw = pd.DataFrame(
        dict(
            timestamp=time, frequency_command_hz=60 + np.arange(22), water_temperature_setpoint=50.0
        )
    )
    actions = pd.DataFrame(dict(candidate_defrost_time=time))
    result = native_control_context(raw, actions, start)
    assert result.frequency_baseline_hz.iloc[:10].isna().all()
    assert result.frequency_baseline_hz.iloc[10:].eq(64.5).all()
    assert result.frequency_command_hz.iloc[10] == 69.0
    assert result.frequency_delta_hz.iloc[10] == 4.5
    np.testing.assert_allclose(result.frequency_slope_hz_per_min.iloc[10:], 2.0)
    changed = raw.copy()
    changed.loc[16:, "frequency_command_hz"] = 150.0
    repeated = native_control_context(changed, actions, start)
    pd.testing.assert_frame_equal(result.iloc[:16], repeated.iloc[:16])
    stale = native_control_context(raw.iloc[:11], actions, start)
    assert np.isnan(stale.frequency_command_hz.iloc[12])


def test_net_command_changes_do_not_depend_on_retrospective_stability_boundary():
    from image_models.control_context import HISTORIES, native_control_context

    start = pd.Timestamp("2026-01-01")
    time = pd.date_range(start, periods=22, freq="30s")
    raw = pd.DataFrame(
        dict(
            timestamp=time, frequency_command_hz=60 + np.arange(22), water_temperature_setpoint=50.0
        )
    )
    actions = pd.DataFrame(dict(candidate_defrost_time=time))
    first = native_control_context(raw, actions, start)
    shifted = native_control_context(raw, actions, start + pd.Timedelta(minutes=5))
    pd.testing.assert_frame_equal(first[HISTORIES["changes"]], shifted[HISTORIES["changes"]])
    assert first.frequency_change_60s_hz.iloc[12] == 2.0
    assert first.frequency_change_300s_hz.iloc[12] == 10.0


def test_cop_action_label_remains_stop_after_first_near_optimal_interval():
    from image_models.labels import label_actions

    time = pd.date_range("2026-01-01", periods=6, freq="30s")
    reference = pd.DataFrame(
        dict(
            cycle_name="a",
            experiment_id="d",
            candidate_defrost_time=time,
            on_action_grid=True,
            cycle_cop_eligible=[True, True, True, False, True, True],
            cycle_cop=[1.0, 1.99, 1.0, 1.0, 2.0, 0.5],
            t_RB=time[-1],
        )
    )
    labelled, intervals = label_actions(reference, label_rule="stopping_latch")
    assert intervals.iloc[0].t_label == time[1]
    np.testing.assert_allclose(
        labelled.after_optimum_target, [0.0, 1.0, 1.0, np.nan, 1.0, 1.0], equal_nan=True
    )
    assert labelled.train_mask.tolist() == [True, True, True, False, True, True]


def test_connected_near_labels_bridge_both_gaps_without_inventing_rewards():
    from image_models.labels import label_actions

    time = pd.date_range("2026-01-01", periods=6, freq="30s")
    reference = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=time,
            on_action_grid=True,
            cycle_cop_eligible=[True, True, True, False, True, True],
            cycle_cop=[1.0, 1.99, 1.0, 1.0, 2.0, 0.5],
            t_RB=time[-1],
        )
    )
    original = reference.copy(deep=True)
    rows, intervals = label_actions(
        reference, label_rule="stopping_latch", near_set_rule="connected"
    )
    assert rows.near_optimal.tolist() == [False, True, True, True, True, False]
    assert rows.near_optimal_exact.tolist() == [False, True, False, False, True, False]
    assert rows.near_optimal_bridged.tolist() == [False, False, True, True, False, False]
    assert rows.train_mask.all() and not rows.teacher_valid.iloc[3]
    assert pd.isna(rows.cycle_cop.iloc[3])
    assert intervals.iloc[0].L == time[1] and intervals.iloc[0].U == time[4]
    pd.testing.assert_frame_equal(reference, original)


def test_control_only_diagnostic_does_not_shrink_primary_rgb_comparison(tmp_path):
    from image_models.comparisons import compensation_comparisons

    records = []
    for version, cop in [("rgb", 2.0), ("fusion", 3.0), ("control", np.nan)]:
        for seed in [0, 1]:
            records.append(
                dict(
                    split="test",
                    version=version,
                    seed=seed,
                    cycle_name="a",
                    experiment_id="d",
                    selected_cop=cop,
                    rb_cop=1.0,
                    optimal_cop=4.0,
                    early_trigger=False,
                    soft_hard_gap=np.nan,
                    near_optimal_hit=False,
                    no_trigger=False,
                    unsupported_trigger=not np.isfinite(cop),
                    regret_pct=25.0,
                )
            )
    compensation_comparisons(
        pd.DataFrame(records),
        tmp_path,
        [("rgb", "fusion"), ("control", "fusion")],
        main_versions=["rgb", "fusion"],
    )
    common = pd.read_csv(tmp_path / "common_summary.csv")
    assert set(common.version) == {"rgb", "fusion"}
    assert common.cohort_cycles.eq(1).all()


def test_seed_average_cop_comparison_does_not_hide_one_failed_seed(tmp_path):
    from image_models.comparisons import compensation_comparisons

    records = []
    for version in [
        "RGB_COP_basin_midpoint_label",
        "RGB_frost_frequency_label",
        "RGB_frost_frequency_label_plus_frequency",
        "RGB_frost_frequency_label_plus_water_setpoint",
        "RGB_common_frost_threshold_label",
    ]:
        for seed in [0, 1]:
            for cycle in ["common", "failure"]:
                cop = 2.0 if version == "RGB_common_frost_threshold_label" else 3.0
                if cycle == "failure":
                    cop = (
                        np.nan
                        if version == "RGB_common_frost_threshold_label" and seed == 1
                        else 100.0
                    )
                records.append(
                    dict(
                        split="test",
                        version=version,
                        seed=seed,
                        cycle_name=cycle,
                        experiment_id="day",
                        selected_cop=cop,
                        rb_cop=1.0,
                        optimal_cop=100.0,
                        early_trigger=False,
                        soft_hard_gap=np.nan,
                        near_optimal_hit=False,
                        no_trigger=False,
                        unsupported_trigger=not np.isfinite(cop),
                        regret_pct=100 - cop,
                    )
                )
    compensation_comparisons(pd.DataFrame(records), tmp_path)
    result = (
        pd.read_csv(tmp_path / "experiment_bootstrap.csv")
        .query(
            'left == "RGB_common_frost_threshold_label" and right == "RGB_frost_frequency_label_plus_frequency"'
        )
        .iloc[0]
    )
    assert result.paired_cycles == 1
    assert result.experiment_equal_delta_cop == 1.0
    assert np.isclose(result.pooled_H_delta_pp, 100 / 99)
    assert np.isclose(result.pooled_H_lower_95, result.pooled_H_upper_95)
    paired = pd.read_csv(tmp_path / "paired_common_summary.csv").query(
        'left == "RGB_common_frost_threshold_label"'
    )
    assert paired.cohort_cycles.eq(1).all()
