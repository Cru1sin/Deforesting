import numpy as np
import pandas as pd
import pytest

from image_models.cycle_selection import screen_cv_cycles


def test_published_screening_and_checkpoint_organization_survive_relocation(tmp_path, monkeypatch):
    import json
    import pickle
    import shutil

    from dataset_tools.training_data import publish_screening_inputs
    from image_models import checkpoints
    from image_models.cycle_selection import screen_published_cycles

    original = tmp_path / "original"
    training = original / "training_data"
    training.mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    rows, catalog = inputs()
    rows["water_temperature_setpoint"] = 50.0
    rows.to_parquet(training / "candidate_decisions.parquet", index=False)
    quality = pd.DataFrame(
        [
            dict(
                cycle_name="cycle",
                experiment_id="experiment",
                actions=12,
                raw_rgb_coverage=1.0,
                dino_action_coverage=1.0,
                control_coverage=1.0,
                action_coverage=1.0,
                max_internal_image_gap_s=30.0,
                leading_image_gap_s=0.0,
                trailing_image_gap_s=0.0,
                heat_coverage=1.0,
                electricity_coverage=1.0,
                ridge_prediction_coverage=1.0,
                cop_candidate_coverage=1.0,
                actual_endpoint=True,
                endpoint_reward_valid=True,
                rb_reward_valid=True,
                interpolated_fraction=0.2,
            )
        ]
    )
    quality.to_csv(source / "quality.csv", index=False)
    rows[["cycle_name", "candidate_defrost_time"]].assign(input_valid=True).to_parquet(
        source / "decision_inputs.parquet", index=False
    )
    settings = dict(
        dataset=str(original),
        decisions=str(training / "candidate_decisions.parquet"),
        quality=str(source / "quality.csv"),
        input_rule="rgb70",
        cohort_rule="observed",
        exclude_rb=False,
        limits={"maximum_interpolated_fraction": 0.1},
    )
    (training / "settings.json").write_text(json.dumps(settings))
    (original / "dataset_manifest.json").write_text(
        json.dumps(dict(dataset_schema_version=3, dataset_id="frost_cycle_dataset", experiments=[]))
    )
    (original / "channel_registry.json").write_text("{}")
    records = catalog.to_dict("records")
    records[0]["assets"] = {"parquet": "cycles/cycle.parquet"}
    (original / "cycle_catalog.json").write_text(json.dumps({"cycles": records}, default=str))
    for name in ("cycles", "cycles_original"):
        (original / name).mkdir()
    publish_screening_inputs(original, settings)
    moved = tmp_path / "moved"
    original.rename(moved)
    shutil.rmtree(source)
    assert not screen_published_cycles(moved).selected.item()  # The saved .1 limit is honored.
    settings["limits"]["maximum_interpolated_fraction"] = 0.3
    (moved / "training_data/settings.json").write_text(json.dumps(settings))
    assert screen_published_cycles(moved / "training_data").selected.item()

    run = tmp_path / "run"
    run.mkdir()
    (run / "settings.json").write_text(
        json.dumps(
            dict(
                dataset=str(original),
                data_definition=settings,
                cycle_cohort=str(source / "deleted_cohort.csv"),
                cycle_names=["cycle"],
            )
        )
    )
    model = dict(training_cycles=[], training_experiments=[])
    predicted = pd.DataFrame(
        dict(cycle_name=["cycle"], experiment_id=["experiment"], split=["test"])
    )
    (run / "model.pkl").write_bytes(pickle.dumps(dict(checkpoint=model, predictions=predicted)))

    def unexpected_index():
        pytest.fail("Organizing one run must not write the global index")

    monkeypatch.setattr("tools.result_index.write_output_index", unexpected_index)
    checkpoints.organize([run], dataset=moved)
    roles = pd.read_csv(run / "checkpoints/model/cycles.csv")
    assert roles.outer_test.item() and not roles.final_training.item()
    assert json.loads((run / "settings.json").read_text())["dataset"] == str(original)


def test_balanced_expansion_preserves_strict_cycles_and_spreads_experiments():
    frames, catalogs, quality = [], [], []
    for name, experiment, end in [("a", "e1", 1.97), ("b", "e1", 1.999), ("c", "e2", 1.999)]:
        rows, catalog = inputs()
        rows["cycle_name"] = name
        rows["water_temperature_setpoint"] = 50.0
        rows.loc[11, "pre_defrost_heat_kwh"] = end
        catalog["cycle_name"], catalog["experiment_id"] = name, experiment
        frames.append(rows)
        catalogs.append(catalog)
        quality.append(
            dict(
                cycle_name=name,
                experiment_id=experiment,
                actions=100,
                raw_rgb_coverage=1.0,
                dino_action_coverage=1.0,
                control_coverage=1.0,
                action_coverage=1.0,
                max_internal_image_gap_s=30.0,
                leading_image_gap_s=0.0,
                trailing_image_gap_s=0.0,
                near_input_gap_s=0.0,
                heat_coverage=1.0,
                electricity_coverage=1.0,
                ridge_prediction_coverage=1.0,
                ridge_domain_coverage=1.0,
                cop_candidate_coverage=1.0,
                actual_endpoint=True,
                endpoint_reward_valid=True,
                rb_reward_valid=True,
            )
        )
    kwargs = dict(
        quality=pd.DataFrame(quality),
        exclude_rb=False,
        minimum_terminal_decline=0.0001,
        cohort_rule="labels",
        maximum_per_setpoint=2,
    )
    result = screen_cv_cycles(pd.concat(frames), pd.concat(catalogs), **kwargs)
    assert set(result.loc[result.selected, "cycle_name"]) == {"a", "c"}
    assert result.admission_candidate.all()
    assert (
        result.set_index("cycle_name").loc["b", "exclusion_reasons"].endswith("balanced_cohort_cap")
    )
    # Duration at selected setpoints replaces decline, even when the cycle ends at its maximum.
    duration_rows = pd.concat(frames)
    duration_rows.loc[
        duration_rows.candidate_defrost_time.eq(duration_rows.candidate_defrost_time.max()),
        "pre_defrost_heat_kwh",
    ] = 2.0
    options = {
        **kwargs,
        "minimum_terminal_decline": 0.01,
        "maximum_per_setpoint": None,
        "post_near_minutes_by_setpoint": {"50": 5.5},
    }
    duration = screen_cv_cycles(duration_rows, pd.concat(catalogs), **options)
    assert duration.selected.all() and not duration.observed_terminal_decline_pass.any()
    assert duration.post_near_minutes.eq(5.5).all()
    options["post_near_minutes_by_setpoint"] = {"50": 5.51}
    assert not screen_cv_cycles(duration_rows, pd.concat(catalogs), **options).selected.any()


def test_rb_review_requires_quality_and_classifies_outside_near_envelope():
    from image_models.cycle_selection import rb_failure_cases

    rows, _ = inputs()
    rows["t_RB"] = rows.candidate_defrost_time.iloc[0] - pd.Timedelta(seconds=1)
    rows["rb_status"] = "triggered"
    screening = pd.DataFrame(
        [
            dict(
                cycle_name="cycle",
                baseline_cycle=True,
                gate_a=True,
                gate_b=True,
                near_points=10,
                low_extrapolation=True,
                manual_excluded=False,
            )
        ]
    )
    assert (
        rb_failure_cases(rows, screening, early_review_rule="strict").failure_type.item()
        == "早除霜"
    )
    rows["t_RB"] = rows.candidate_defrost_time.iloc[-1]
    assert rb_failure_cases(rows, screening).failure_type.item() == "迟除霜"
    # A hole between near components is neither early nor late.
    rows["t_RB"] = rows.candidate_defrost_time.iloc[5]
    assert rb_failure_cases(rows, screening).empty
    rows["t_RB"] = rows.candidate_defrost_time.iloc[-1]
    screening["gate_b"] = False
    assert rb_failure_cases(rows, screening).empty
    screening["gate_b"], screening["manual_excluded"] = True, True
    assert rb_failure_cases(rows, screening).empty


def test_relaxed_early_review_all_cycles_needs_supported_trigger_not_training_quality():
    from image_models.cycle_selection import rb_failure_cases

    rows, _ = inputs()
    rows.loc[0, "pre_defrost_heat_kwh"] = 1.5
    rows["t_RB"] = rows.candidate_defrost_time.iloc[0]
    rows["rb_status"] = "triggered"
    screening = pd.DataFrame(
        [
            dict(
                cycle_name="cycle",
                baseline_cycle=False,
                gate_a=False,
                gate_b=False,
                near_points=10,
                low_extrapolation=False,
                manual_excluded=False,
            )
        ]
    )
    case = rb_failure_cases(rows, screening).iloc[0]
    assert case.failure_type == "早除霜" and case.rb_trigger_source == "replayed_rb_in_experiment"
    assert not case.strict_review_eligible
    assert rb_failure_cases(rows, screening, early_review_rule="strict").empty
    screening["manual_excluded"] = True
    assert rb_failure_cases(rows, screening).empty
    screening["manual_excluded"] = False
    rows.loc[0, "pre_defrost_heat_measurement_valid"] = False
    assert rb_failure_cases(rows, screening).empty


def inputs():
    # Ten near-optimal points in two components; three use extrapolation.
    times = pd.date_range("2026-01-01", periods=12, freq="30s")
    heat = [2.0] * 5 + [1.0] + [2.0] * 5 + [1.97]
    rows = pd.DataFrame(
        dict(
            cycle_name="cycle",
            candidate_defrost_time=times,
            effective_heat_rule="without_trunc",
            pre_defrost_heat_kwh=heat,
            pre_defrost_electricity_kwh=0.9,
            defrost_event_electricity_kwh=0.1,
            defrost_event_net_heat_kwh=0.0,
            pre_defrost_feature_window_valid=True,
            pre_defrost_electricity_measurement_valid=True,
            pre_defrost_heat_measurement_valid=True,
            defrost_event_electricity_prediction_available=True,
            defrost_event_net_heat_prediction_available=True,
            defrost_event_electricity_in_training_domain=True,
            defrost_event_net_heat_in_training_domain=True,
            on_action_grid=True,
        )
    )
    rows.loc[[0, 6, 7], "defrost_event_electricity_in_training_domain"] = False
    rows.loc[11, "on_action_grid"] = False  # Exact endpoint must not be dropped.
    catalog = pd.DataFrame(
        [
            dict(
                cycle_name="cycle",
                experiment_id="experiment",
                defrost_preparation_start=times[-1],
                end_time=times[-1] + pd.Timedelta(minutes=5),
                front_rgb_valid=False,
                status="invalid",
            )
        ]
    )  # Neither is a CV selection gate.
    return rows, catalog


def test_extrapolation_marks_never_exclude_near_points_or_cycles():
    rows, catalog = inputs()
    before = rows.copy(deep=True)
    result = screen_cv_cycles(rows, catalog).iloc[0]
    assert result.near_points == 10
    assert result.near_ridge_extrapolation_points == 3
    assert result.near_extrapolation_fraction == 0.3
    assert result.selected and result.long_enough and result.low_extrapolation
    assert result.end_cop == 1.97
    pd.testing.assert_frame_equal(rows, before)
    rows.loc[5, "defrost_event_electricity_in_training_domain"] = False
    assert screen_cv_cycles(rows, catalog).selected.iloc[
        0
    ]  # Outside-near extrapolation is irrelevant.
    rows.loc[8, "defrost_event_electricity_in_training_domain"] = False
    assert screen_cv_cycles(rows, catalog).selected.iloc[0]
    from defrost_decision.performance_objectives import (
        add_single_objective_optima,
        calculate_cycle_cop,
    )

    before = add_single_objective_optima(
        calculate_cycle_cop(rows, effective=True), names=("cycle_cop",)
    )
    rows["defrost_event_electricity_in_training_domain"] = False
    after = add_single_objective_optima(
        calculate_cycle_cop(rows, effective=True), names=("cycle_cop",)
    )
    columns = [
        "cycle_cop",
        "cycle_cop_eligible",
        "cycle_cop_t_star",
        "cycle_cop_basin_1pct_start",
        "cycle_cop_basin_1pct_end",
    ]
    pd.testing.assert_frame_equal(before[columns], after[columns])
    assert after.loc[after.cycle_cop_eligible, "cycle_cop_uses_model_extrapolation"].all()
    assert screen_cv_cycles(rows, catalog).selected.item()


def test_saved_candidates_accept_mixed_timestamp_precision():
    from image_models.cycle_selection import legal_quality_actions

    rows, catalog = inputs()
    rows["candidate_defrost_time"] = rows.candidate_defrost_time.astype(str)
    rows.loc[1, "candidate_defrost_time"] += ".000000000"
    rows["heating_accounting_start"] = "2026-01-01 00:00:00"
    rows.loc[1, "heating_accounting_start"] += ".000000000"
    assert screen_cv_cycles(rows, catalog).selected.iloc[0]
    assert legal_quality_actions(rows).iloc[1]


def test_manual_cycle_exclusion_preserves_scientific_checks():
    rows, catalog = inputs()
    result = screen_cv_cycles(rows, catalog, excluded_cycles=["cycle"]).iloc[0]
    assert result.long_enough and result.low_extrapolation
    assert not result.selected and result.manual_excluded
    assert "manual_cycle_exclusion" in result.exclusion_reasons


@pytest.mark.parametrize("end_heat,expected", [(1.98, False), (1.979, True), (np.nan, None)])
def test_endpoint_boundary_and_unknown(end_heat, expected):
    rows, catalog = inputs()
    rows.loc[11, "pre_defrost_heat_kwh"] = end_heat
    result = screen_cv_cycles(rows, catalog).iloc[0]
    if expected is None:
        assert pd.isna(result.long_enough) and not result.selected
    else:
        assert result.long_enough == expected


def test_missing_endpoint_no_reference_and_recording_end_are_explicit():
    rows, catalog = inputs()
    missing = screen_cv_cycles(rows.iloc[:-1], catalog).iloc[0]
    assert pd.isna(missing.long_enough) and not missing.selected
    assert missing.exclusion_reasons == "exact_end_point_absent"
    rows["pre_defrost_heat_measurement_valid"] = False
    unknown = screen_cv_cycles(rows, catalog).iloc[0]
    assert pd.isna(unknown.low_extrapolation) and not unknown.selected
    rows, catalog = inputs()
    catalog["end_time"] = catalog.defrost_preparation_start
    catalog["defrost_preparation_start"] = pd.NaT
    fallback = screen_cv_cycles(rows, catalog).iloc[0]
    assert fallback.selected and fallback.end_source == "recording_end"


def test_rejects_temperature_truncated_heat():
    rows, catalog = inputs()
    rows["effective_heat_rule"] = "outlet_at_least_recovery_temperature"
    with pytest.raises(ValueError, match="without_trunc"):
        screen_cv_cycles(rows, catalog)


@pytest.mark.parametrize("end_heat,passed", [(1.90, False), (1.899, True), (1.97, False)])
def test_terminal_decline_is_additional_to_quality_and_allows_rb(end_heat, passed):
    rows, catalog = inputs()
    rows.loc[11, "pre_defrost_heat_kwh"] = end_heat
    catalog["assets"] = [{"parquet": "cycles_RB/cycle.parquet"}]
    q = pd.DataFrame(
        [
            dict(
                cycle_name="cycle",
                experiment_id="experiment",
                actions=100,
                raw_rgb_coverage=1.0,
                dino_action_coverage=1.0,
                control_coverage=1.0,
                action_coverage=1.0,
                max_internal_image_gap_s=30.0,
                leading_image_gap_s=0.0,
                trailing_image_gap_s=0.0,
                heat_coverage=1.0,
                electricity_coverage=1.0,
                ridge_prediction_coverage=1.0,
                ridge_domain_coverage=1.0,
                cop_candidate_coverage=1.0,
                actual_endpoint=True,
                endpoint_reward_valid=True,
                rb_reward_valid=True,
            )
        ]
    )
    r = screen_cv_cycles(
        rows, catalog, quality=q, exclude_rb=False, minimum_terminal_decline=0.05
    ).iloc[0]
    assert r.gate_a and r.gate_b and r.baseline_cycle
    assert r.terminal_decline_pass == passed and r.selected == passed
    q.loc[0, "raw_rgb_coverage"] = 0.5
    assert not screen_cv_cycles(
        rows, catalog, quality=q, exclude_rb=False, minimum_terminal_decline=0.05
    ).selected.item()
    from image_models.cycle_selection import quality_gates

    for column in [
        "raw_rgb_coverage",
        "dino_action_coverage",
        "control_coverage",
        "action_coverage",
    ]:
        q[column] = 0.70
    q["max_internal_image_gap_s"] = 1000.0
    q["near_input_gap_s"] = 90.0
    assert quality_gates(q, "near70").gate_a.item()
    q["near_input_gap_s"] = 120.0
    assert quality_gates(q, "near70").gate_a.item()
    q["near_input_gap_s"] = 150.0
    assert not quality_gates(q, "near70").gate_a.item()


def test_label_training_uses_observed_terminal_and_ignores_reward_gate():
    rows, catalog = inputs()
    catalog["end_time"] = catalog.defrost_preparation_start
    catalog["defrost_preparation_start"] = pd.NaT
    q = pd.DataFrame(
        [
            dict(
                cycle_name="cycle",
                experiment_id="experiment",
                actions=100,
                raw_rgb_coverage=1.0,
                dino_action_coverage=1.0,
                control_coverage=1.0,
                action_coverage=1.0,
                max_internal_image_gap_s=30.0,
                leading_image_gap_s=0.0,
                trailing_image_gap_s=0.0,
                heat_coverage=0.99,
                electricity_coverage=1.0,
                ridge_prediction_coverage=1.0,
                ridge_domain_coverage=0.6,
                cop_candidate_coverage=0.99,
                actual_endpoint=False,
                endpoint_reward_valid=False,
                rb_reward_valid=False,
            )
        ]
    )
    r = screen_cv_cycles(
        rows,
        catalog,
        quality=q,
        exclude_rb=False,
        minimum_terminal_decline=0.01,
        rb_minimum_terminal_decline=0.02,
        cohort_rule="labels",
    ).iloc[0]
    assert r.selected_train and not r.selected_evaluation and r.label_valid
    assert r.observed_terminal_cop == 1.97
    assert r.observed_terminal_time == rows.candidate_defrost_time.iloc[-1]
    rows.loc[8, "defrost_event_electricity_in_training_domain"] = False
    assert screen_cv_cycles(
        rows,
        catalog,
        quality=q,
        exclude_rb=False,
        minimum_terminal_decline=0.01,
        cohort_rule="labels",
    ).selected_train.item()


def test_near_gap_does_not_fill_cop_holes_and_rb_has_distinct_threshold():
    from image_models.cycle_selection import near_input_gaps

    times = pd.date_range("2026-01-01", periods=8, freq="30s")
    g = pd.DataFrame(
        dict(
            cycle_name="x",
            candidate_defrost_time=times,
            near_optimal=[True, True, True, False, True, True, True, False],
            input_valid=False,
        )
    )
    assert near_input_gaps(g).near_input_gap_s.item() == 90
    g.loc[3, "near_optimal"] = True
    assert near_input_gaps(g).near_input_gap_s.item() == 210
    rows, catalog = inputs()
    rows.loc[11, "pre_defrost_heat_kwh"] = 1.97
    catalog["assets"] = [{"parquet": "cycles_RB/cycle.parquet"}]
    assert not screen_cv_cycles(
        rows,
        catalog,
        exclude_rb=False,
        minimum_terminal_decline=0.01,
        rb_minimum_terminal_decline=0.02,
    ).selected.item()
    rows.loc[11, "pre_defrost_heat_kwh"] = 1.959
    assert screen_cv_cycles(
        rows,
        catalog,
        exclude_rb=False,
        minimum_terminal_decline=0.01,
        rb_minimum_terminal_decline=0.02,
    ).selected.item()


def test_quality_admission_is_independent_of_optimum_and_other_gate():
    from image_models.cycle_selection import legal_quality_actions, quality_gates

    rows, catalog = inputs()
    q = pd.DataFrame(
        [
            dict(
                cycle_name="cycle",
                experiment_id="experiment",
                actions=100,
                raw_rgb_coverage=1.0,
                dino_action_coverage=0.98,
                control_coverage=0.95,
                action_coverage=0.93,
                max_internal_image_gap_s=30.02,
                leading_image_gap_s=20.0,
                trailing_image_gap_s=15.0,
                heat_coverage=1.0,
                electricity_coverage=1.0,
                ridge_prediction_coverage=1.0,
                ridge_domain_coverage=0.8,
                cop_candidate_coverage=1.0,
                actual_endpoint=True,
                endpoint_reward_valid=True,
                rb_reward_valid=True,
            )
        ]
    )
    assert screen_cv_cycles(rows, catalog, quality=q).selected.item()
    q["ridge_domain_coverage"] = 0.0
    assert screen_cv_cycles(rows, catalog, quality=q).selected.item()
    rows["pre_defrost_heat_kwh"] = 2.0  # Endpoint now optimal; new admission must not move.
    result = screen_cv_cycles(rows, catalog, quality=q)
    assert result.selected.item() and not result.legacy_selected.item()
    catalog["assets"] = [{"parquet": "cycles_RB/cycle.parquet"}]
    assert not screen_cv_cycles(rows, catalog, quality=q).selected.item()
    assert screen_cv_cycles(rows, catalog, quality=q, exclude_rb=False).selected.item()
    catalog["assets"] = [{"parquet": "cycles/cycle.parquet"}]
    q.loc[0, "max_internal_image_gap_s"] = 600
    assert not quality_gates(q).gate_a.item() and quality_gates(q).gate_b.item()
    q.loc[0, "max_internal_image_gap_s"] = 30
    q.loc[0, "cop_candidate_coverage"] = 0.99
    assert quality_gates(q).gate_a.item() and not quality_gates(q).gate_b.item()
    rows["heating_accounting_start"] = rows.candidate_defrost_time.iloc[0]
    assert not legal_quality_actions(rows).iloc[0] and legal_quality_actions(rows).iloc[1]


def test_input_chain_does_not_require_harmony_or_reward(monkeypatch, tmp_path):
    from defrost_decision.performance_objectives import calculate_cycle_cop
    from image_models import admission

    rows, catalog = inputs()
    rows["heating_accounting_start"] = rows.candidate_defrost_time.iloc[0]
    rows["stable_heating_start"] = rows.heating_accounting_start
    rows["t_RB"] = rows.candidate_defrost_time.iloc[-1]
    rows = calculate_cycle_cop(rows, effective=True)
    rows["cycle_cop_eligible"] = False
    frames = pd.DataFrame(
        dict(
            cycle_name="cycle",
            camera_role="front",
            file_name=["a.jpg", "b.jpg"],
            image_time=rows.candidate_defrost_time.iloc[[1, 2]].to_numpy(),
        )
    )

    def features(keys, paths):
        return pd.concat(
            [
                keys.reset_index(drop=True),
                pd.DataFrame(1.0, index=range(len(keys)), columns=admission.RGB),
            ],
            axis=1,
        )

    def control(dataset, name, actions):
        return actions.assign(
            control_timestamp=actions.candidate_defrost_time - pd.Timedelta(seconds=1),
            **{field: 1.0 for field in admission.FREQUENCY},
        )

    monkeypatch.setattr(admission, "merge_cached_features", features)
    monkeypatch.setattr(admission, "load_control_context", control)
    from types import SimpleNamespace

    monkeypatch.setattr(
        admission,
        "DatasetLoader",
        lambda _: SimpleNamespace(
            load_cycle=lambda _: pd.DataFrame({"is_interpolated": [False, True, False, False]})
        ),
    )
    matched, quality, _ = admission.audit_cycle(
        "cycle", rows, frames, catalog.iloc[0], tmp_path, []
    )
    assert matched.input_valid.iloc[0] and not matched.cycle_cop_eligible.any()
    assert "visual_frost_score" not in matched
    assert not quality.complete_stopping_reward.item()
    assert quality.interpolated_fraction.item() == 0.25
    # Long image loss remains visible; it is never filled by a reward-valid point.
    assert not matched.input_valid.iloc[-1]


@pytest.mark.parametrize(
    "fraction,terminal,passed",
    [(0.30, 1.97, True), (0.301, 1.97, False), (0.1, 1.98, False), (0.1, np.nan, False)],
)
def test_observed_cohort_uses_only_images_interpolation_and_exact_decline(
    fraction, terminal, passed
):
    rows, catalog = inputs()
    rows.loc[11, "pre_defrost_heat_kwh"] = terminal
    catalog["assets"] = [{"parquet": "cycles_RB/cycle.parquet"}]
    q = pd.DataFrame(
        [
            dict(
                cycle_name="cycle",
                experiment_id="experiment",
                actions=100,
                raw_rgb_coverage=0.70,
                dino_action_coverage=0.70,
                interpolated_fraction=fraction,
                control_coverage=0.0,
                action_coverage=0.0,
                max_internal_image_gap_s=9999.0,
                leading_image_gap_s=9999.0,
                trailing_image_gap_s=9999.0,
                heat_coverage=0.0,
                electricity_coverage=0.0,
                ridge_prediction_coverage=0.0,
                cop_candidate_coverage=0.0,
                actual_endpoint=False,
                endpoint_reward_valid=False,
                rb_reward_valid=False,
            )
        ]
    )
    r = screen_cv_cycles(
        rows, catalog, quality=q, exclude_rb=False, input_rule="rgb70", cohort_rule="observed"
    ).iloc[0]
    assert r.selected == passed and r.selected_evaluation == passed and r.selected_train == passed
    assert not r.gate_b


def test_interpolation_threshold_preserves_saved_30pct_recipe():
    from image_models.cycle_selection import input_quality_limits

    assert input_quality_limits("rgb70")["maximum_interpolated_fraction"] == 0.30
    assert input_quality_limits("rgb70", 0.30)["maximum_interpolated_fraction"] == 0.30
