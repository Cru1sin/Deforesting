import numpy as np
import pandas as pd
import pytest


def test_camera_stage_is_explicit_and_index_honors_requested_camera(tmp_path, monkeypatch):
    from image_models import camera_comparison as study

    with pytest.raises(SystemExit):
        study.build_parser().parse_args([])
    calls = []
    monkeypatch.setattr(study, "image_index", lambda *args: calls.append(args))
    study.main(["--stage", "index", "--cameras", "front", "--output", str(tmp_path)])
    assert calls[0][2] == ["front"]


def test_camera_preparation_extracts_only_requested_views(tmp_path, monkeypatch):
    from image_models import camera_comparison as study
    from image_models import camera_data

    args = study.build_parser().parse_args(
        ["--stage", "prepare", "--cameras", "left", "--output", str(tmp_path)]
    )
    indexed, commands = [], []
    monkeypatch.setattr(camera_data, "image_index", lambda *a: indexed.append(a[2]))
    monkeypatch.setattr(
        camera_data.subprocess, "run", lambda command, **kw: commands.append(command)
    )

    def stop_before_publication(_):
        raise RuntimeError("stop before Dataset publication")

    monkeypatch.setattr(camera_data, "DatasetLoader", stop_before_publication)
    with pytest.raises(RuntimeError, match="stop before Dataset publication"):
        camera_data.prepare(args)
    assert indexed == [["left"]]
    command = commands[0]
    assert command[command.index("--camera-roles") + 1 : command.index("--include-rb")] == ["left"]


def test_lstm_tuning_recipe_preserves_searched_capacity_and_regularization():
    from image_models.model_selection import tuning_options

    params = dict(
        hidden_width=25,
        recurrent_layers=2,
        learning_rate=0.003,
        weight_decay=0.0001,
        visual_dropout=0.1,
        output_dropout=0.2,
    )
    options = tuning_options(params)
    assert options["hidden_width"] == 25 and options["recurrent_layers"] == 2
    assert options["learning_rate"] == 0.003
    assert options["dropout_config"] == dict(
        visual_dropout=0.1, output_dropout=0.2, visual_time_shared=True, output_time_shared=True
    )


def test_camera_summary_rejects_missing_heldout_cycles():
    from image_models.camera_comparison import validate_test_table

    table = pd.DataFrame(
        dict(cycle_name=["a"], split=["test"], outer_experiment=["e"], experiment_id=["e"])
    )
    with pytest.raises(ValueError, match="cover the requested camera cohort exactly"):
        validate_test_table(table, ["a", "b"], first=True)


def test_camera_freezes_once_before_loeo_and_excludes_second_test(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from image_models import camera_comparison as study
    from image_models import experiments, inputs, training
    from image_models import model_selection as shared

    args = SimpleNamespace(
        dataset=tmp_path,
        output=tmp_path,
        seed=42,
        n_jobs=1,
        maximum_epochs=160,
        tune_trials=18,
        standard_through="exp_20260910",
        cameras=["front", "top"],
        test_experiments=None,
    )
    rows = pd.DataFrame(
        dict(
            cycle_name=["a", "b", "later"],
            experiment_id=["e1", "e2", "external"],
            water_temperature_setpoint=50.0,
            training_eligible=True,
            evaluation_eligible=True,
        )
    )
    cohort = pd.concat(
        [
            rows.assign(camera_role=camera, data_role=["first", "first", "second"])
            for camera in args.cameras
        ]
    )
    monkeypatch.setattr(study, "study_cohorts", lambda _: cohort)
    monkeypatch.setattr(
        study, "DatasetLoader", lambda _: SimpleNamespace(camera_training_settings=lambda _: {})
    )
    # Loader owns experiment splits; this fixture has two complementary groups.
    study.DatasetLoader.inner_folds = lambda *a, **k: [["e1"], ["e2"]]
    from dataset_tools import DatasetLoader

    study.DatasetLoader.split_training_data = DatasetLoader.split_training_data
    monkeypatch.setattr(inputs, "load_data", lambda *a: (rows, rows, pd.DataFrame()))
    monkeypatch.setattr(experiments, "record_experiment", lambda *a: None)
    selections = []
    fits = []
    deployments = []
    choice = dict(
        epoch=80,
        fixed_epochs=80,
        learning_rate=0.003,
        weight_decay=0.0001,
        visual_dropout=0.2,
        output_dropout=0.2,
    )

    def candidate(model, development, reference, *a, **k):
        assert set(development.cycle_name) == set(reference.cycle_name) == {"a", "b"}
        assert len(shared.policy_numeric_columns(model)) == 7 and model.hidden_width == 25
        selections.append(model.camera_role)
        return choice

    def search(model, development, reference, *a, **k):
        candidate(model, development, reference)
        pd.DataFrame([choice]).to_csv(model.output / "frozen_configurations.csv", index=False)

    monkeypatch.setattr(shared, "fit_candidate", candidate)
    monkeypatch.setattr(shared, "search_shared_learning_rate", search)

    def fold(development, reference, model, *, test, epochs, seed, output, name, camera_role):
        assert camera_role == name
        assert set(development.cycle_name) == {"a", "b"}
        assert set(development.loc[development.split.eq("train"), "experiment_id"]) == (
            {"e1", "e2"} - {test}
        )
        assert model["architecture"] == "LSTM" and model["film"] and model["hidden_width"] == 25
        fits.append((name, test, epochs, model["learning_rate"]))

    monkeypatch.setattr(training, "fit_model", fold)

    def deployment(model, rows, reference, standard, external):
        assert standard == {"a", "b"} and external == {"later"}
        deployments.append(model.camera_role)

    monkeypatch.setattr(study, "fit_deployment", deployment)
    monkeypatch.setattr(study, "collect_camera", lambda *a: None)
    study.train(args)
    assert selections == deployments == ["front", "top"]
    assert fits == [(camera, test, 80, 0.003) for camera in args.cameras for test in ["e1", "e2"]]


def test_deployment_loader_rejects_cycle_and_experiment_overlap():
    from dataset_tools import DatasetLoader

    rows = pd.DataFrame(dict(cycle_name=["a", "b", "c"], experiment_id=["old", "old", "new"]))
    split = DatasetLoader.split_deployment_data(rows, ["a", "b"], ["c"])
    assert split.split.tolist() == ["train", "train", "test"]
    with pytest.raises(ValueError, match="cycles overlap"):
        DatasetLoader.split_deployment_data(rows, ["a"], ["a"])
    with pytest.raises(ValueError, match="experiments overlap"):
        DatasetLoader.split_deployment_data(rows, ["a"], ["b"])


def test_repeated_setpoint_folds_keep_single_experiment_condition_without_leakage():
    from dataset_tools import DatasetLoader

    conditions = pd.DataFrame(
        [
            dict(
                cycle_name=f"{temp}_{i}",
                experiment_id=f"{temp}_{i}",
                water_temperature_setpoint=temp,
            )
            for temp in [50, 55, 60]
            for i in range(4)
        ]
        + [dict(cycle_name="rare", experiment_id="rare", water_temperature_setpoint=65)]
    )
    folds = DatasetLoader.inner_folds(conditions, "50_0", coverage="repeated")
    assert set(folds[0]).isdisjoint(folds[1]) and "50_0" not in folds[0] + folds[1]
    assert "rare" in folds[0] + folds[1]
    for fold in folds:
        assert {50, 55, 60}.issubset(
            set(conditions.loc[conditions.experiment_id.isin(fold), "water_temperature_setpoint"])
        )


def test_seed_replication_uses_common_inner_only_epoch_rule(tmp_path, monkeypatch):
    import json

    from image_models import ablation

    original = tmp_path / "original"
    (original / "training").mkdir(parents=True)
    (original / "training/settings.json").write_text(
        json.dumps(
            dict(
                dataset="dataset",
                recipes=[["GRU32_film", "cdf_ramp"]],
                hidden_width=32,
                learning_rate=0.003,
                output_dropout=0.2,
                maximum_epochs=60,
                fixed_epochs=67,
                reset_history=True,
            )
        )
    )
    pd.DataFrame([{"方法": "无历史GRU32 + FiLM", "source": str(original)}]).to_csv(
        tmp_path / "所有方法比较.csv", index=False
    )
    monkeypatch.setattr(ablation, "analysis_output", lambda path: tmp_path)

    def capture(command, **kwargs):
        assert command[command.index("--maximum-epochs") + 1] == "160"
        assert command[command.index("--validation-utility") + 1] == "headroom"
        assert command[command.index("--validation-rule") + 1] == "stratified"
        assert "--fixed-epochs" not in command and "--tune-trials" not in command
        start = command.index("--seeds") + 1
        assert command[start : start + 3] == ["42", "2026", "1010"]
        assert "--reset-history" in command
        raise RuntimeError("captured replication command")

    monkeypatch.setattr(ablation.subprocess, "run", capture)
    with pytest.raises(RuntimeError, match="captured replication command"):
        ablation.replicate_typical_models(tmp_path / "baseline", tmp_path, [42, 2026, 1010], 6)


def test_shared_lr_search_freezes_one_training_side_recipe(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from image_models import model_selection as optimal_stopping

    calls = []

    def fit(args, rows, reference, test, folds, width, depth, learning_rate, **kwargs):
        calls.append((test, width, kwargs))
        return dict(
            candidate_key=str(learning_rate),
            learning_rate=learning_rate,
            epoch=80,
            parameters=42226,
            weight_decay=kwargs["weight_decay"],
            visual_dropout=kwargs["visual_dropout"],
            output_dropout=kwargs["output_dropout"],
            validation_paired=10,
            validation_headroom=100 - abs(learning_rate - 0.003) * 1000,
            validation_set_loss=1.0,
        )

    monkeypatch.setattr(optimal_stopping, "fit_candidate", fit)
    args = SimpleNamespace(
        output=tmp_path,
        development_experiment=None,
        seeds=[0],
        n_jobs=1,
        shared_lr_trials=3,
        maximum_epochs=160,
        hidden_width=25,
        recurrent_layers=1,
        weight_decay=1e-4,
        visual_dropout=0.2,
        output_dropout=0.2,
    )
    optimal_stopping.search_shared_learning_rate(args, None, None, {"outer": [["a"], ["b"]]})
    assert args.learning_rate == 0.003 and args.fixed_epochs == 80 and args.trigger_rule == "cdf"
    assert all(t == "outer" and w == 25 and k["visual_dropout"] == 0.2 for t, w, k in calls)
    optimal_stopping.search_shared_learning_rate(args, None, None, {"outer": [["a"], ["b"]]})
    assert len(calls) == 3


def test_settings_resume_compares_json_shapes(tmp_path):
    from image_models.experiments import save_settings

    path = tmp_path / "settings.json"
    save_settings(path, dict(shapes=[(25,)], layers=1))
    save_settings(path, dict(shapes=[(25,)], layers=1))
    with pytest.raises(ValueError, match="settings changed"):
        save_settings(path, dict(shapes=[(25,)], layers=2))


def test_nested_fit_keeps_requested_visual_dropout_architecture(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from image_models import training as optimal_stopping

    architectures = []
    history = pd.DataFrame(
        dict(
            epoch=[1],
            training_loss=[1.0],
            validation_cycles=[1],
            validation_cdf_evaluable=[1],
            validation_hazard_evaluable=[1],
            validation_cdf_cop=[2.0],
            validation_hazard_cop=[1.0],
            validation_set_loss=[1.0],
            validation_set_cycles=[1],
        )
    )

    def fit(train, validation, **kwargs):
        architectures.append(kwargs["architecture"])
        assert "outer" not in set(train.experiment_id)
        return dict(losses=history.copy())

    monkeypatch.setattr("image_models.policy.fit_policy", fit)
    monkeypatch.setattr(optimal_stopping, "fit_case", lambda *args, **kwargs: None)
    rows = pd.DataFrame(
        dict(
            experiment_id=["a", "b", "outer"],
            cycle_name=["a", "b", "outer"],
            input_valid=True,
            training_eligible=True,
            evaluation_eligible=True,
            near_optimal=True,
        )
    )
    args = SimpleNamespace(
        output=tmp_path,
        inner_folds=2,
        version="drop",
        architecture="GRU32_visualdrop",
        loss_name="set_stop",
        maximum_epochs=1,
        patience=2,
        cohort="training",
        tune_trials=0,
    )
    optimal_stopping.fit_nested(args, rows, pd.DataFrame(), "outer", [["a"], ["b"]], 0)
    assert architectures == ["GRU32_visualdrop"] * 3


def test_inner_experiments_keep_each_setpoint_and_ignore_outer():
    from dataset_tools import DatasetLoader

    inner_folds = DatasetLoader.inner_folds
    rows = pd.DataFrame(
        [
            dict(experiment_id=f"{temp}_{i}", water_temperature_setpoint=temp)
            for temp in [50, 55, 60]
            for i in range(4)
        ]
    )
    folds = inner_folds(rows, "50_0")
    pool = rows.loc[rows.experiment_id.ne("50_0")]
    assert set(folds[0]).isdisjoint(folds[1])
    assert set(folds[0] + folds[1]) == set(pool.experiment_id)
    for validation in folds:
        assert set(
            pool.loc[~pool.experiment_id.isin(validation), "water_temperature_setpoint"]
        ) == {50, 55, 60}
    changed = rows.copy()
    changed.loc[changed.experiment_id.eq("50_0"), "water_temperature_setpoint"] = 99
    assert inner_folds(changed, "50_0") == folds


def test_temperature_is_strictly_past_and_not_interpolated():
    from image_models.control_context import native_control_context

    times = pd.date_range("2026-01-01", periods=14, freq="30s")
    raw = pd.DataFrame(
        dict(
            timestamp=times,
            frequency_command_hz=80.0,
            actual_frequency_hz=79.0,
            water_temperature_setpoint=50.0,
            water_out_temperature=np.arange(14) + 40.0,
        )
    )
    actions = pd.DataFrame(dict(candidate_defrost_time=[times[11]], stable_heating_start=times[0]))
    a = native_control_context(raw, actions, times[0])
    assert a.water_out_temperature.item() == 50.0
    raw.loc[raw.timestamp.ge(times[11]), "water_out_temperature"] = 999.0
    assert native_control_context(raw, actions, times[0]).water_out_temperature.item() == 50.0


def test_compensation_uses_actual_frequency_and_past_temperature_error():
    from image_models.control_context import native_control_context

    times = pd.date_range("2026-01-01", periods=14, freq="30s")
    raw = pd.DataFrame(
        dict(
            timestamp=times,
            frequency_command_hz=80.0,
            actual_frequency_hz=np.arange(14) + 70.0,
            water_temperature_setpoint=50.0,
            water_out_temperature=49.0 - np.arange(14) * 0.1,
        )
    )
    actions = pd.DataFrame(dict(candidate_defrost_time=[times[11]], stable_heating_start=times[0]))
    result = native_control_context(raw, actions, times[0])
    assert result.actual_frequency_change_300s_hz.item() == 10.0
    assert np.isclose(result.temperature_error_change_300s_c.item(), 1.0)
    assert np.isclose(result.temperature_error_c.item(), 2.0)
    raw.loc[raw.timestamp.ge(times[11]), "actual_frequency_hz"] = 200.0
    assert (
        native_control_context(raw, actions, times[0]).actual_frequency_change_300s_hz.item()
        == 10.0
    )


def test_training_cohort_diagnostics_do_not_require_reviewer_or_leak_experiments():
    from dataset_tools import DatasetLoader

    rows = pd.DataFrame(
        dict(
            experiment_id=["a", "a", "b", "c", "d"],
            evaluation_eligible=[True, False, False, False, False],
            training_eligible=True,
        )
    )
    folds = DatasetLoader.training_folds(rows, cohort="training", validation_count=1)
    assert set(folds) == {"a", "b", "c", "d"}
    split = DatasetLoader.split_training_data(rows, "a", ["b"], cohort="training")
    assert split.loc[split.experiment_id.eq("a"), "split"].eq("test").all()
    assert split.loc[split.experiment_id.eq("b"), "split"].eq("validation").all()
    assert set(split.loc[split.split.eq("train"), "experiment_id"]) == {"c", "d"}


def test_cop_tie_break_uses_validation_set_loss():
    from unittest.mock import patch

    import torch

    from image_models.policy import fit_policy

    torch.set_num_threads(1)
    rows = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=pd.date_range("2026-01-01", periods=4, freq="30s"),
            x=[-1.0, 0.0, 1.0, 2.0],
            near_optimal=[False, False, True, True],
            input_valid=True,
        )
    )

    def fixed_threshold(*args, **kwargs):
        return 0.5, pd.DataFrame(dict(threshold=[0.5], mean_cycle_cop=[2.0]))

    with patch("image_models.policy.select_cop_threshold", side_effect=fixed_threshold):
        fit = fit_policy(
            rows,
            rows,
            architecture="logistic",
            loss_name="set_stop",
            numeric_columns=["x"],
            visual_columns=[],
            seed=0,
            maximum_epochs=8,
            patience=9,
            cycles_per_batch=1,
            validation_tiebreak="set_loss",
            initial_hazard=0.01,
        )
    history = fit["losses"]
    assert history.validation_mean_cycle_cop.nunique() == 1
    assert fit["selected_epoch"] == int(history.loc[history.validation_set_loss.idxmin(), "epoch"])
    assert fit["selected_epoch"] > 1


def test_joint_epoch_selection_can_choose_cdf_without_changing_training():
    from unittest.mock import patch

    from image_models.policy import fit_policy

    times = pd.date_range("2026-01-01", periods=3, freq="30s")
    rows = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=times,
            x=[0.0, 1.0, 2.0],
            near_optimal=[False, True, False],
            input_valid=True,
            cycle_cop=[1.0, 2.0, 1.0],
            cycle_cop_eligible=True,
            terminal_time=times[-1],
            optimal_time=times[1],
            optimal_cop=2.0,
            t_RB=times[0],
        )
    )

    def prediction(probabilities):
        p = np.array(probabilities)
        return rows.assign(probability=p, decision_margin=np.log(p / (1 - p)))

    fits = []
    for joint in [False, True]:
        with patch(
            "image_models.policy.predict_policy",
            side_effect=[prediction([0.6, 0.1, 0.1]), prediction([0.3, 0.3, 0.3])],
        ):
            fits.append(
                fit_policy(
                    rows,
                    rows,
                    architecture="logistic",
                    loss_name="set_stop",
                    numeric_columns=["x"],
                    visual_columns=[],
                    seed=0,
                    maximum_epochs=2,
                    patience=3,
                    cycles_per_batch=1,
                    validation_strategy="first_positive_terminal",
                    validation_complete_first=True,
                    joint_trigger_selection=joint,
                    controller_reference=rows,
                )
            )
    assert [f["selected_epoch"] for f in fits] == [1, 2]
    assert fits[1]["losses"].validation_trigger_rule.tolist() == ["hazard", "cdf"]
    np.testing.assert_array_equal(fits[0]["losses"].training_loss, fits[1]["losses"].training_loss)


def test_stopping_ensemble_averages_event_distributions_causally():
    from image_models.policy import apply_trigger_rule, mix_stopping_hazards

    times = pd.date_range("2026-01-01", periods=4, freq="30s")
    a = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=times,
            input_valid=[True, False, True, True],
            probability=[0.2, np.nan, 0.5, 0.8],
        )
    )
    b = a.assign(probability=[0.6, np.nan, 0.1, 0.2])
    mixed = mix_stopping_hazards([a, b])
    expected = (
        apply_trigger_rule(a, "cdf").probability + apply_trigger_rule(b, "cdf").probability
    ) / 2
    np.testing.assert_allclose(
        apply_trigger_rule(mixed, "cdf").probability, expected, equal_nan=True
    )
    assert np.isnan(mixed.probability.iloc[1])
    changed = b.copy()
    changed.loc[3, "probability"] = 0.99
    pd.testing.assert_frame_equal(mixed.iloc[:3], mix_stopping_hazards([a, changed]).iloc[:3])


def test_patch_pooling_does_not_reuse_cls_features():
    import io
    import tempfile
    from pathlib import Path

    import torch
    from PIL import Image

    from image_models.dinov2_features import (
        DINO_COLUMNS,
        PATCH_COLUMNS,
        _embed,
        merge_cached_features,
    )

    class Backbone(torch.nn.Module):
        def forward(self, x):
            return torch.full((len(x), 384), 2.0)

        def forward_features(self, x):
            return {
                "x_norm_patchtokens": torch.stack(
                    [torch.full((len(x), 384), 3.0), torch.full((len(x), 384), 5.0)], dim=1
                )
            }

    payload = io.BytesIO()
    Image.new("RGB", (2, 2)).save(payload, format="PNG")
    np.testing.assert_allclose(
        _embed([payload.getvalue()], Backbone(), torch.device("cpu"), "patch_mean"), 4.0
    )
    rows = pd.DataFrame(dict(cycle_name=["a"], camera_role=["front"], file_name=["a.png"]))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "cache.parquet"
        pd.concat([rows, pd.DataFrame(np.ones((1, 384)), columns=DINO_COLUMNS)], axis=1).to_parquet(
            path, index=False
        )
        assert merge_cached_features(rows, [path], PATCH_COLUMNS)[PATCH_COLUMNS].isna().all().all()


def test_visual_augmentation_has_equal_width_no_information_control():
    from types import SimpleNamespace
    from unittest.mock import patch

    from image_models.dinov2_features import PATCH_COLUMNS
    from image_models.inputs import RGB, VISUAL_POOLING
    from image_models.inputs import prepare_ablation_inputs as prepare
    from image_models.model import build_model

    original = pd.DataFrame(np.arange(2 * 768).reshape(2, 768), columns=[*RGB, *PATCH_COLUMNS])
    original["input_valid"] = [True, False]
    original["near_optimal"] = [False, True]
    for pooling in ["cls_patch_mean", "cls_repeat"]:
        with patch("image_models.inputs.DatasetLoader") as loader:
            loader.return_value.load_training_data.return_value = (original.copy(), pd.DataFrame())
            rows, _ = prepare(
                SimpleNamespace(
                    dataset="dataset",
                    loss_name="set_stop",
                    visual_pooling=pooling,
                    visual_history="none",
                    recipe="temperature",
                    candidate_set="exact",
                    late_preference="none",
                )
            )
        columns = VISUAL_POOLING[pooling]
        assert len(columns) == len(set(columns)) == 768
        np.testing.assert_array_equal(rows[columns[:384]], original[RGB])
        np.testing.assert_array_equal(
            rows[columns[384:]], original[PATCH_COLUMNS if pooling == "cls_patch_mean" else RGB]
        )
        pd.testing.assert_frame_equal(
            rows[["input_valid", "near_optimal"]], original[["input_valid", "near_optimal"]]
        )
    models = [
        build_model("GRU32", 5, len(VISUAL_POOLING[p]), 0) for p in ["cls_patch_mean", "cls_repeat"]
    ]
    assert sum(v.numel() for v in models[0].parameters()) == sum(
        v.numel() for v in models[1].parameters()
    )
    with patch("image_models.inputs.DatasetLoader") as loader:
        loader.return_value.load_training_data.return_value = (original.copy(), pd.DataFrame())
        rows, _ = prepare(
            SimpleNamespace(
                dataset="dataset",
                loss_name="set_stop",
                visual_pooling="patch_mean_repeat4",
                visual_history="none",
                recipe="temperature",
                candidate_set="exact",
                late_preference="none",
            )
        )
    assert len(VISUAL_POOLING["patch_mean_repeat4"]) == len(VISUAL_POOLING["patch_grid2"]) == 1536
    for start in range(0, 1536, 384):
        np.testing.assert_array_equal(
            rows[VISUAL_POOLING["patch_mean_repeat4"][start : start + 384]], original[PATCH_COLUMNS]
        )


def test_zero_visual_clock_keeps_teacher_and_action_support():
    from types import SimpleNamespace
    from unittest.mock import patch

    from image_models.inputs import RGB
    from image_models.inputs import prepare_ablation_inputs as prepare

    source = pd.DataFrame(np.ones((3, 384)), columns=RGB)
    source["input_valid"] = [True, False, True]
    source["near_optimal"] = [False, True, True]
    source["cycle_start"] = pd.Timestamp("2026-01-01")
    source["candidate_defrost_time"] = pd.date_range("2026-01-01 00:10", periods=3, freq="30s")
    with patch("image_models.inputs.DatasetLoader") as loader:
        loader.return_value.load_training_data.return_value = (source.copy(), pd.DataFrame())
        rows, _ = prepare(
            SimpleNamespace(
                dataset="dataset",
                loss_name="set_stop",
                elapsed_context=True,
                visual_pooling="zero",
                visual_history="none",
                recipe="temperature",
                candidate_set="exact",
                late_preference="none",
            )
        )
    assert rows[RGB].eq(0).all().all()
    np.testing.assert_allclose(rows.elapsed_cycle_minutes, [10.0, 10.5, 11.0])
    pd.testing.assert_frame_equal(
        rows[["input_valid", "near_optimal"]], source[["input_valid", "near_optimal"]]
    )


def test_spatial_pooling_preserves_quadrant_order_and_global_mean():
    import io

    import torch
    from PIL import Image

    from image_models.dinov2_features import _embed

    class Backbone(torch.nn.Module):
        def forward_features(self, x):
            grid = (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                .repeat_interleave(8, 0)
                .repeat_interleave(8, 1)
            )
            return {"x_norm_patchtokens": grid.flatten()[None, :, None].expand(len(x), 256, 384)}

    payload = io.BytesIO()
    Image.new("RGB", (2, 2)).save(payload, format="PNG")
    grid = _embed([payload.getvalue()], Backbone(), torch.device("cpu"), "patch_grid2").reshape(
        1, 4, 384
    )
    np.testing.assert_allclose(grid[0, :, 0], [1.0, 2.0, 3.0, 4.0])
    mean = _embed([payload.getvalue()], Backbone(), torch.device("cpu"), "patch_mean")
    np.testing.assert_allclose(grid.mean(axis=1), mean)


def test_temperature_trend_adds_only_published_causal_control():
    from types import SimpleNamespace
    from unittest.mock import patch

    from image_models.inputs import FREQUENCY, TEMPERATURE
    from image_models.training import fit_fold

    args = SimpleNamespace(
        recipe="temperature",
        cohort="training",
        ambient_context=False,
        version="trend",
        architecture="GRU32",
        visual_history="none",
        visual_pooling="cls",
        temperature_trend=True,
    )
    rows = pd.DataFrame()
    with (
        patch("image_models.training.DatasetLoader.split_training_data", return_value=rows),
        patch("image_models.training.fit_case") as fit,
    ):
        fit_fold(args, rows, rows, "test", ["val"], 0)
    assert fit.call_args.kwargs["numeric_columns"] == [
        *FREQUENCY,
        *TEMPERATURE,
        "temperature_error_change_300s_c",
    ]


def test_pooled_selection_weights_evaluable_cycles_and_keeps_rules_separate():
    from image_models.model_selection import pooled_validation_selection

    histories = []
    for counts, cops, losses in [
        (([1, 2], [2, 2]), ([2.0, 5.0], [4.0, 6.0]), [0.4, 0.2]),
        (([3, 2], [3, 3]), ([8.0, 2.0], [2.0, 1.0]), [0.1, 0.3]),
    ]:
        h = pd.DataFrame(
            {
                "epoch": [1, 2],
                "validation_cycles": [counts[1][0]] * 2,
                "validation_set_cycles": [counts[1][0]] * 2,
                "validation_set_loss": losses,
            }
        )
        for i, rule in enumerate(["hazard", "cdf"]):
            h[f"validation_{rule}_evaluable"] = counts[i]
            h[f"validation_{rule}_cop"] = cops[i]
        histories.append(h)
    result = pooled_validation_selection(histories)
    assert result.iloc[0].epoch == 2 and result.iloc[0].rule == "cdf"
    assert result.iloc[0].validation_evaluable == 5
    assert result.iloc[0].validation_cop == 3.0
    hazard = result.loc[result.rule.eq("hazard") & result.epoch.eq(1)].iloc[0]
    assert hazard.validation_cop == 6.5 and hazard.validation_evaluable == 4


def test_stratified_validation_preserves_conditions_and_excludes_outer():
    from dataset_tools import DatasetLoader

    rows = pd.DataFrame(
        [
            dict(
                cycle_name=f"{temp}_{i}",
                experiment_id=f"{temp}_{i}",
                water_temperature_setpoint=temp,
                evaluation_eligible=True,
                training_eligible=True,
            )
            for temp in [50, 55, 60]
            for i in range(7)
        ]
    )
    folds = DatasetLoader.training_folds(rows, cohort="training", validation_rule="stratified")
    for test, val in folds.items():
        assert test not in val
        train = rows.loc[~rows.experiment_id.isin([test, *val])]
        assert set(train.water_temperature_setpoint) == {50, 55, 60}
        assert set(rows.loc[rows.experiment_id.isin(val), "water_temperature_setpoint"]) == {
            50,
            55,
            60,
        }
    nested = DatasetLoader.training_folds(rows, nested=True, cohort="training")
    for test, folds in nested.items():
        assert set(folds[0]).isdisjoint(folds[1])
        assert set(folds[0] + folds[1]) == set(rows.experiment_id) - {test}


def test_trigger_rule_selection_uses_validation_and_preserves_model_name(monkeypatch):
    import json
    import pickle
    import tempfile
    from pathlib import Path

    from image_models.stopping_diagnostics import select_trigger_rule

    # This synthetic fixture tests rule selection, not full Dataset admission/reporting.
    monkeypatch.setattr(
        "image_models.comparisons.write_evaluation_metrics", lambda metrics, dataset, output: None
    )
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        run = root / "run"
        inner = root / "inner"
        output = root / "out"
        run.mkdir()
        inner.mkdir()
        (root / "training_data").mkdir()
        settings = dict(dataset=str(root), folds={"outer": ["validation"]})
        for path in [run, inner]:
            (path / "settings.json").write_text(json.dumps(settings))
        times = pd.date_range("2026-01-01", periods=3, freq="30s")

        def trajectory(name, experiment, split, rewards):
            return pd.DataFrame(
                dict(
                    cycle_name=name,
                    experiment_id=experiment,
                    split=split,
                    candidate_defrost_time=times,
                    input_valid=True,
                    probability=0.3,
                    cycle_cop=rewards,
                    cycle_cop_eligible=True,
                    terminal_time=times[-1],
                    optimal_time=times[np.argmax(rewards)],
                    optimal_cop=max(rewards),
                    near_optimal=np.array(rewards) >= 0.99 * max(rewards),
                    t_RB=times[0],
                )
            )

        validation = trajectory("v", "validation", "validation", [1.0, 2.0, 1.0])
        outer = trajectory("t", "outer", "test", [1.0, 1.1, 3.0])
        pd.concat([validation, outer]).to_parquet(
            root / "training_data/reference.parquet", index=False
        )
        metadata = pd.DataFrame(dict(outer_experiment=["outer"], version=["anchor"]))
        for path, predictions in [(run, outer), (inner, validation)]:
            with (path / "anchor_outer_seed0.pkl").open("wb") as stream:
                pickle.dump(
                    dict(checkpoint={"seed": 0}, predictions=predictions, cycle_metrics=metadata),
                    stream,
                )
        select_trigger_rule(run, inner, output)
        result = pd.read_csv(output / "cycle_metrics.csv")
        assert result.chosen_rule.item() == "cdf", (
            "Outer reward must not choose hazard despite its higher test COP"
        )
        assert result.selected_cop.item() == 1.1
        assert result.version.item() == "anchor_validation_rule"
        with (run / "anchor_outer_seed0.pkl").open("wb") as stream:
            pickle.dump(
                dict(
                    checkpoint={"seed": 0},
                    predictions=outer,
                    cycle_metrics=metadata,
                    validation_predictions=validation,
                ),
                stream,
            )
        select_trigger_rule(run, run, root / "embedded")
        assert pd.read_csv(root / "embedded/cycle_metrics.csv").chosen_rule.item() == "cdf"


def test_ambient_context_is_a_strictly_past_external_measurement():
    from image_models.control_context import native_control_context

    times = pd.date_range("2026-01-01", periods=14, freq="30s")
    raw = pd.DataFrame(
        dict(
            timestamp=times,
            frequency_command_hz=80.0,
            actual_frequency_hz=79.0,
            water_temperature_setpoint=50.0,
            water_out_temperature=49.0,
            control_ambient_c=np.arange(14) * 0.1 - 5.0,
        )
    )
    actions = pd.DataFrame(dict(candidate_defrost_time=[times[11]], stable_heating_start=times[0]))
    result = native_control_context(raw, actions, times[0])
    assert result.control_ambient_c.item() == -4.0
    raw.loc[raw.timestamp.ge(times[11]), "control_ambient_c"] = 20.0
    assert native_control_context(raw, actions, times[0]).control_ambient_c.item() == -4.0


def test_feature_omission_removes_only_requested_numeric_inputs():
    from types import SimpleNamespace

    from image_models.inputs import policy_numeric_columns

    args = SimpleNamespace(
        elapsed_context=True,
        compensation_progress=True,
        drop_inputs=["frequency_change_60s_hz", "frequency_change_300s_hz"],
    )
    assert policy_numeric_columns(args) == [
        "frequency_headroom_fraction",
        "control_setpoint_c",
        "control_outlet_c",
        "elapsed_cycle_minutes",
        "frequency_used_margin_fraction",
    ]
    args.drop_inputs = []
    assert len(policy_numeric_columns(args)) == 7


def test_pooled_headroom_uses_paired_sums_and_fixed_cdf():
    from image_models.model_selection import pooled_validation_selection, tuning_options

    history = pd.DataFrame(
        dict(
            epoch=[1, 2],
            validation_cycles=[2, 2],
            validation_cdf_evaluable=[2, 2],
            validation_cdf_cop=[3.0, 2.0],
            validation_set_loss=[1.0, 1.0],
            validation_set_cycles=[2, 2],
            validation_cdf_validation_paired=[2, 2],
            validation_cdf_model_sum=[4.0, 4.5],
            validation_cdf_rb_sum=[4.0, 4.0],
            validation_cdf_optimal_sum=[5.0, 5.0],
        )
    )
    result = pooled_validation_selection([history], "cdf", "headroom")
    assert result.iloc[0].epoch == 2 and result.iloc[0].validation_headroom == 50.0
    assert set(result.rule) == {"cdf"}
    assert (
        tuning_options(dict(dropout_place="none", learning_rate=0.001))["dropout_config"][
            "visual_dropout"
        ]
        == 0.0
    )


def test_pooled_cdf_loss_selects_teacher_fit_and_weights_cycles_not_folds():
    from image_models.model_selection import pooled_validation_selection

    histories = []
    for count, losses in [(1, [0.8, 0.1]), (3, [0.2, 0.3])]:
        histories.append(
            pd.DataFrame(
                dict(
                    epoch=[1, 2],
                    validation_cycles=count,
                    validation_cdf_evaluable=count,
                    validation_cdf_cop=[3.0, 2.0],
                    validation_set_loss=1.0,
                    validation_set_cycles=count,
                    validation_cdf_validation_paired=count,
                    validation_cdf_model_sum=[3.0 * count, 2.0 * count],
                    validation_cdf_rb_sum=1.0 * count,
                    validation_cdf_optimal_sum=4.0 * count,
                    validation_cdf_loss=losses,
                    validation_cdf_loss_cycles=count,
                )
            )
        )
    result = pooled_validation_selection(histories, "cdf", "cdf_loss")
    assert result.iloc[0].epoch == 2
    np.testing.assert_allclose(result.validation_cdf_loss, [0.25, 0.35])
    assert result.iloc[0].validation_headroom < result.iloc[1].validation_headroom


def test_optuna_architecture_refit_never_uses_outer_experiment(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from image_models import training as optimal_stopping

    calls = []

    def fit(train, validation, **kwargs):
        assert "outer" not in set(train.experiment_id)
        calls.append(kwargs["architecture"])
        model_sum = 4.8 if kwargs["architecture"] == "GRU32_film" else 4.4
        return dict(
            losses=pd.DataFrame(
                dict(
                    epoch=[1],
                    training_loss=[1.0],
                    validation_cycles=[2],
                    validation_cdf_evaluable=[2],
                    validation_cdf_cop=[model_sum / 2],
                    validation_set_loss=[1.0],
                    validation_set_cycles=[2],
                    validation_cdf_validation_paired=[2],
                    validation_cdf_model_sum=[model_sum],
                    validation_cdf_rb_sum=[4.0],
                    validation_cdf_optimal_sum=[5.0],
                )
            )
        )

    monkeypatch.setattr("image_models.policy.fit_policy", fit)
    monkeypatch.setattr(optimal_stopping, "fit_case", lambda *args, **kwargs: None)
    rows = pd.DataFrame(
        dict(
            experiment_id=["a", "b", "outer"],
            cycle_name=["a", "b", "outer"],
            input_valid=True,
            training_eligible=True,
            evaluation_eligible=True,
            near_optimal=True,
        )
    )
    args = SimpleNamespace(
        output=tmp_path,
        inner_folds=2,
        version="tuned",
        architecture="GRU32",
        loss_name="cdf_ramp",
        maximum_epochs=1,
        patience=2,
        cohort="training",
        tune_trials=4,
        tune_architectures=["GRU32", "GRU32_film"],
        trigger_rule="cdf",
        validation_utility="headroom",
    )
    optimal_stopping.fit_nested(args, rows, pd.DataFrame(), "outer", [["a"], ["b"]], 0)
    assert calls[-1] == "GRU32_film"
    assert len(list(tmp_path.glob("checkpoints/*/*trials.csv"))) == 1


def test_cached_inner_fit_reuses_effective_dropout_but_rejects_changed_inputs(
    tmp_path, monkeypatch
):
    import json
    import pickle
    from types import SimpleNamespace

    from image_models import model_selection as stopping
    from image_models.checkpoints import checkpoint_path

    run = tmp_path / "baseline"
    run.mkdir()
    settings = dict(
        loss_name="cdf_ramp",
        maximum_epochs=60,
        data_definition={"cohort": "frozen"},
        elapsed_context=True,
    )
    (run / "settings.json").write_text(json.dumps(settings))
    monkeypatch.setattr(
        stopping,
        "DatasetLoader",
        lambda dataset: SimpleNamespace(
            camera_training_settings=lambda camera: {"cohort": "frozen"}
        ),
    )
    args = SimpleNamespace(
        compare_run=run,
        dataset=tmp_path,
        loss_name="cdf_ramp",
        maximum_epochs=60,
        elapsed_context=True,
        validation_utility="headroom",
    )
    path = checkpoint_path(run, "baseline_outer_seed0.pkl")
    path.parent.mkdir(parents=True)
    checkpoint = dict(
        architecture="GRU32_film",
        dropout_config={"output_dropout": 0.2, "output_time_shared": True},
        learning_rate=0.002,
        inner_validation_folds=[["a"], ["b"]],
        training_experiments=["a", "b"],
        numeric_columns=stopping.policy_numeric_columns(args),
        visual_columns=stopping.policy_visual_columns(args),
    )
    path.write_bytes(pickle.dumps({"checkpoint": checkpoint}))
    history_path = path.parent / "inner_baseline_outer_fold0_seed0.csv"
    pd.DataFrame({"epoch": [1, 2], "validation_cdf_loss": [0.3, 0.2]}).to_csv(
        history_path, index=False
    )
    options = {
        "learning_rate": 0.002,
        "dropout_config": {"output_dropout": 0.2, "output_time_shared": True},
    }
    history, origin = stopping.cached_inner_history(
        args, "outer", [["a"], ["b"]], 0, "GRU32_film", options, 0
    )
    assert list(history.epoch) == [1, 2] and origin == str(history_path)
    args.drop_inputs = ["frequency_change_60s_hz"]
    assert stopping.cached_inner_history(
        args, "outer", [["a"], ["b"]], 0, "GRU32_film", options, 0
    ) == (None, None)


@pytest.mark.parametrize("architecture", ["GRU32_film", "LSTM32_film"])
def test_depth_and_history_ablation_remain_causal_and_restore(architecture):
    import torch

    from image_models.policy import (
        _model_values,
        build_model,
        fit_policy,
        predict_policy,
        restore_policy,
    )

    x = torch.randn(2, 6, 11)
    for layers in (1, 2, 3):
        model = build_model(
            architecture,
            numeric_width=3,
            visual_width=8,
            seed=0,
            hidden_width=16,
            recurrent_layers=layers,
            dropout_config={"visual_dropout": 0.0},
        )
        model.eval()
        assert model.gru.num_layers == layers
        assert torch.allclose(model(x)[:, :3], model(x[:, :3]), atol=1e-6)
        restored = restore_policy(
            dict(
                architecture=architecture,
                numeric_columns=list(range(3)),
                visual_columns=list(range(8)),
                seed=0,
                hidden_width=16,
                recurrent_layers=layers,
                dropout_config={"visual_dropout": 0.0},
                model_state_dict=model.state_dict(),
            )
        )
        assert torch.equal(model(x), restored(x))
    isolated = build_model(
        architecture,
        numeric_width=3,
        visual_width=8,
        seed=0,
        reset_history=True,
        dropout_config={"visual_dropout": 0.0},
    )
    isolated.eval()
    changed = x.clone()
    changed[:, :3] = 99
    assert torch.equal(isolated(x)[:, 3:], isolated(changed)[:, 3:])
    assert sum(p.numel() for p in isolated.parameters()) == sum(
        p.numel()
        for p in build_model(architecture, numeric_width=3, visual_width=8, seed=0).parameters()
    )
    columns = [f"x{i}" for i in range(11)]
    rows = pd.DataFrame(x[0].numpy(), columns=columns).assign(
        cycle_name="a",
        candidate_defrost_time=pd.date_range("2026-01-01", periods=6, freq="30s"),
        stopping_cdf_target=[0.0, 0.0, 0.25, 0.5, 0.75, 1.0],
    )
    fitted = fit_policy(
        rows,
        None,
        architecture=architecture,
        loss_name="cdf_ramp",
        numeric_columns=columns[:3],
        visual_columns=columns[3:],
        seed=0,
        maximum_epochs=2,
        patience=3,
        cycles_per_batch=1,
        recurrent_layers=2,
        reset_history=True,
    )
    assert restore_policy(fitted).gru.num_layers == 2
    assert restore_policy(fitted).reset_history
    assert np.isfinite(fitted["losses"].training_loss).all()
    original = predict_policy(rows, fitted)
    cached = predict_policy(
        rows, fitted, cached_values=_model_values(rows, columns, fitted["preprocessor"])
    )
    assert np.array_equal(original.decision_margin, cached.decision_margin)


@pytest.mark.parametrize("architecture", ["MLP_film", "MLP_plain", "logistic"])
def test_current_only_heads_fit_restore_and_do_not_use_previous_input(architecture):
    import torch

    from image_models.policy import build_model, fit_policy, predict_policy, restore_policy

    columns = [f"x{i}" for i in range(11)]
    values = torch.randn(1, 6, 11)
    model = build_model(
        architecture,
        numeric_width=3,
        visual_width=8,
        seed=0,
        dropout_config={"visual_dropout": 0.0, "output_dropout": 0.0},
    )
    model.eval()

    def forward(x):
        return model(x) if architecture.startswith("MLP_") else model(x.reshape(-1, 11))

    changed = values.clone()
    changed[:, :3] = 99.0
    assert torch.equal(forward(values).reshape(-1)[3:], forward(changed).reshape(-1)[3:])
    rows = pd.DataFrame(values[0].numpy(), columns=columns).assign(
        cycle_name="a",
        candidate_defrost_time=pd.date_range("2026-01-01", periods=6, freq="30s"),
        stopping_cdf_target=[0.0, 0.0, 0.25, 0.5, 0.75, 1.0],
    )
    fitted = fit_policy(
        rows,
        None,
        architecture=architecture,
        loss_name="cdf_ramp",
        numeric_columns=columns[:3],
        visual_columns=columns[3:],
        seed=0,
        maximum_epochs=2,
        patience=3,
        cycles_per_batch=1,
        dropout_config={"visual_dropout": 0.0, "output_dropout": 0.0},
    )
    assert np.isfinite(fitted["losses"].training_loss).all()
    assert np.isfinite(predict_policy(rows, fitted).decision_margin).all()
    restored = restore_policy(fitted)
    assert not hasattr(restored, "gru")
    if architecture == "MLP_film":
        assert restored.film is not None
        assert (
            sum(
                p.numel()
                for p in build_model(
                    architecture, numeric_width=7, visual_width=384, seed=0, hidden_width=84
                ).parameters()
            )
            == 41497
        )


def test_plain_mlp_removes_only_film():
    import torch

    from image_models.policy import build_model

    options = dict(
        numeric_width=7,
        visual_width=384,
        seed=0,
        hidden_width=84,
        dropout_config={"visual_dropout": 0.1, "output_dropout": 0.2, "output_time_shared": True},
    )
    film = build_model("MLP_film", **options)
    plain = build_model("MLP_plain", **options)
    assert plain.film is None
    assert sum(p.numel() for p in plain.parameters()) == 40153
    for key, value in plain.state_dict().items():
        assert torch.equal(value, film.state_dict()[key])
    assert plain.visual_dropout == film.visual_dropout
    assert plain.output_dropout.p == film.output_dropout.p


def test_pure_mlp_tuning_and_depth_restore():
    import torch

    from image_models.model_selection import tuning_options
    from image_models.policy import build_model, restore_policy

    params = dict(
        hidden_width=64,
        mlp_layers=3,
        learning_rate=0.006,
        weight_decay=0.001,
        visual_dropout=0.1,
        output_dropout=0.2,
    )
    options = tuning_options(params)
    assert options["learning_rate"] == 0.006 and options["weight_decay"] == 0.001
    for depth in [1, 2, 4]:
        model = build_model(
            "MLP_plain",
            numeric_width=7,
            visual_width=384,
            seed=0,
            hidden_width=64,
            mlp_layers=depth,
        )
        expected = 392 * 64 + (depth - 1) * 65 * 64 + 65
        assert sum(p.numel() for p in model.parameters()) == expected
        checkpoint = dict(
            architecture="MLP_plain",
            numeric_columns=list(range(7)),
            visual_columns=list(range(384)),
            seed=0,
            hidden_width=64,
            mlp_layers=depth,
            model_state_dict=model.state_dict(),
        )
        restored = restore_policy(checkpoint)
        values = torch.randn(2, 5, 391)
        torch.testing.assert_close(model(values), restored(values))
    # Old MLP weights without a depth field still restore as two hidden layers.
    old = build_model("MLP_plain", numeric_width=7, visual_width=384, seed=0, hidden_width=64)
    checkpoint.update(model_state_dict=old.state_dict())
    checkpoint.pop("mlp_layers")
    assert len(restore_policy(checkpoint).mlp) == 4


@pytest.mark.parametrize("lstm", [False, True])
def test_shared_mlp_candidate_uses_only_training_side_and_resumes(tmp_path, monkeypatch, lstm):
    from types import SimpleNamespace

    from image_models import model_selection as optimal_stopping

    calls = []

    def fit(train, validation, **kwargs):
        assert "outer" not in set(train.experiment_id) | set(validation.experiment_id)
        assert set(train.experiment_id).isdisjoint(validation.experiment_id)
        calls.append(kwargs)
        return dict(losses=pd.DataFrame(dict(epoch=[1], training_loss=[1.0])))

    monkeypatch.setattr("image_models.policy.fit_policy", fit)
    monkeypatch.setattr(
        optimal_stopping,
        "pooled_validation_selection",
        lambda histories, *args: pd.DataFrame([dict(epoch=1, validation_headroom=75.0)]),
    )
    rows = pd.DataFrame(
        dict(
            experiment_id=["a", "b", "outer"],
            input_valid=True,
            training_eligible=True,
            evaluation_eligible=True,
        )
    )
    args = SimpleNamespace(
        output=tmp_path,
        cohort="training",
        architecture="LSTM32_film" if lstm else "MLP_film",
        loss_name="cdf_ramp",
        maximum_epochs=3,
        seeds=[0],
        visual_dropout=0.1,
        output_dropout=0.2,
        visual_time_shared=True,
        output_time_shared=True,
        reset_history=True,
    )
    for _ in range(2):
        result = optimal_stopping.fit_candidate(
            args,
            rows,
            pd.DataFrame(),
            "outer",
            [["a"], ["b"]],
            64,
            2,
            0.003,
            recurrent_layers=2 if lstm else 1,
        )
        if lstm:
            from image_models.model import build_model

            assert calls[-1]["recurrent_layers"] == 2
            assert result["parameters"] == sum(
                p.numel()
                for p in build_model(
                    "LSTM32_film",
                    len(optimal_stopping.policy_numeric_columns(args)),
                    len(optimal_stopping.policy_visual_columns(args)),
                    0,
                    hidden_width=64,
                    recurrent_layers=2,
                ).parameters()
            )
            assert result["candidate_key"].endswith("_layers2")
        else:
            assert result["parameters"] == (390 * 64 + 65 * 64 + 65 + 2 * 64 * 6)
            assert (
                result["candidate_key"] is None
            )  # Legacy grid histories retain their original filenames.
    assert len(calls) == 2
    assert all(call["reset_history"] for call in calls)


@pytest.mark.parametrize("scheduled", [False, True])
def test_fixed_epochs_skip_inner_selection_and_exclude_outer(tmp_path, monkeypatch, scheduled):
    from types import SimpleNamespace

    from image_models import training as optimal_stopping

    calls = []

    def fit(train, validation, **kwargs):
        assert validation is None
        assert set(train.experiment_id) == {"a", "b"}
        assert kwargs["maximum_epochs"] == 101
        calls.append(kwargs)
        return dict(losses=pd.DataFrame(dict(epoch=[101], training_loss=[1.0])))

    def save(*args, **kwargs):
        assert kwargs["checkpoint"]["fixed_training_epochs"] == 101
        assert "pooled_validation_scores" not in kwargs["checkpoint"]

    monkeypatch.setattr("image_models.policy.fit_policy", fit)
    monkeypatch.setattr(optimal_stopping, "fit_case", save)
    rows = pd.DataFrame(
        dict(
            experiment_id=["a", "b", "outer"],
            input_valid=True,
            training_eligible=True,
            evaluation_eligible=True,
        )
    )
    args = SimpleNamespace(
        output=tmp_path,
        inner_folds=2,
        version="fixed",
        architecture="MLP_film",
        loss_name="cdf_ramp",
        fixed_epochs=101,
        maximum_epochs=101,
        patience=102,
        tune_trials=0,
        cohort="training",
        visual_time_shared=True,
        trigger_rule="cdf",
        hidden_width=86,
        learning_rate=0.003,
        visual_dropout=0.1,
        output_dropout=0.2,
    )
    if scheduled:
        args.fixed_epochs = None
        args.epoch_schedule = tmp_path / "epochs.csv"
        args.frozen_epochs = {("outer", 0): 101}
    optimal_stopping.fit_nested(args, rows, pd.DataFrame(), "outer", [["a"], ["b"]], 0)
    assert len(calls) == 1


def test_fold_tuning_reuse_keeps_outer_experiment_unseen(tmp_path, monkeypatch):
    import pickle
    from types import SimpleNamespace

    import pytest

    from image_models import training as optimal_stopping
    from image_models.checkpoints import checkpoint_path

    folds = [["a"], ["b"]]
    params = dict(
        hidden_width=64,
        mlp_layers=1,
        learning_rate=0.006,
        weight_decay=0.001,
        visual_dropout=0.1,
        output_dropout=0.2,
    )
    source = checkpoint_path(tmp_path / "source", "mlp_outer_seed0.pkl")
    source.parent.mkdir(parents=True)
    original = dict(
        training_experiments=["a", "b"],
        inner_validation_folds=folds,
        tuning=dict(parameters=params),
    )
    source.write_bytes(pickle.dumps(dict(checkpoint=original)))
    history = pd.DataFrame(
        dict(
            epoch=[1],
            training_loss=[1.0],
            validation_cycles=[1],
            validation_cdf_evaluable=[1],
            validation_hazard_evaluable=[1],
            validation_cdf_cop=[2.0],
            validation_hazard_cop=[1.0],
            validation_set_loss=[1.0],
            validation_set_cycles=[1],
        )
    )

    def fit(train, validation, **kwargs):
        assert "outer" not in set(train.experiment_id)
        assert kwargs["hidden_width"] == 64 and kwargs["mlp_layers"] == 1
        assert kwargs["dropout_config"]["visual_dropout"] == 0.1
        assert kwargs["dropout_config"]["output_dropout"] == 0.2
        return dict(losses=history.copy())

    monkeypatch.setattr("image_models.policy.fit_policy", fit)
    monkeypatch.setattr(optimal_stopping, "fit_case", lambda *args, **kwargs: None)
    rows = pd.DataFrame(
        dict(
            experiment_id=["a", "b", "outer"],
            cycle_name=["a", "b", "outer"],
            input_valid=True,
            training_eligible=True,
            evaluation_eligible=True,
            near_optimal=True,
        )
    )
    args = SimpleNamespace(
        output=tmp_path / "new",
        fold_tuning_from=tmp_path / "source",
        inner_folds=2,
        version="mlp",
        architecture="MLP_plain",
        loss_name="set_stop",
        maximum_epochs=1,
        patience=2,
        cohort="training",
        tune_trials=0,
        visual_time_shared=True,
    )
    optimal_stopping.fit_nested(args, rows, pd.DataFrame(), "outer", folds, 0)
    original["training_experiments"].append("outer")
    source.write_bytes(pickle.dumps(dict(checkpoint=original)))
    with pytest.raises(ValueError, match="independent outer experiment"):
        optimal_stopping.fit_nested(args, rows, pd.DataFrame(), "outer", folds, 0)


def test_tapered_and_expanding_mlp_restore_and_parameter_count():
    import torch

    from image_models.policy import build_model, restore_policy

    for widths in [(64,), (256, 128, 64), (64, 128, 64), (1000, 64)]:
        for architecture in ["MLP_plain", "MLP_film"]:
            model = build_model(
                architecture, numeric_width=7, visual_width=384, seed=0, mlp_widths=widths
            )
            expected = (
                sum((before + 1) * after for before, after in zip((391, *widths[:-1]), widths))
                + widths[-1]
                + 1
            )
            if architecture == "MLP_film":
                expected += 16 * widths[-1]
            assert sum(p.numel() for p in model.parameters()) == expected
            checkpoint = dict(
                architecture=architecture,
                numeric_columns=list(range(7)),
                visual_columns=list(range(384)),
                seed=0,
                mlp_widths=list(widths),
                model_state_dict=model.state_dict(),
            )
            restored = restore_policy(checkpoint)
            model.eval()
            restored.eval()
            values = torch.randn(2, 3, 391)
            assert torch.equal(model(values), restored(values))


def test_lstm_component_recipes_are_single_factor_and_seed_filtered(tmp_path, monkeypatch):
    import pickle

    from image_models import ablation

    recipes = ablation.ablation_recipes("LSTM32_film")
    assert [(dimension, label) for dimension, label, _ in recipes] == [
        ("LSTM组件", "无视觉"),
        ("LSTM组件", "仅视觉"),
        ("LSTM组件", "无FiLM"),
        ("LSTM组件", "无LSTM历史"),
    ]
    assert recipes[1][2] == ["--drop-inputs", *ablation.NUMERIC]
    assert recipes[2][2] == ["--architecture", "LSTM_plain"]
    paths = []
    for seed in (42, 2026, 1010):
        path = tmp_path / f"{seed}.pkl"
        with path.open("wb") as stream:
            pickle.dump(
                dict(
                    checkpoint=dict(seed=seed),
                    cycle_metrics=pd.DataFrame(
                        dict(
                            seed=[seed, seed], split=["train", "test"], cycle_name=["train", "test"]
                        )
                    ),
                ),
                stream,
            )
        paths.append(path)
    monkeypatch.setattr(ablation, "checkpoint_files", lambda run: paths)
    metrics = ablation.read_metrics(tmp_path, "reference", [42])
    assert metrics.seed.tolist() == [42] and metrics.split.tolist() == ["test"]


def test_no_cdf_ramp_is_pointwise_bce_with_the_same_labels():
    import torch

    from image_models.policy import _batch_loss

    rows = pd.DataFrame(
        dict(
            cycle_name=["a"] * 3,
            candidate_defrost_time=pd.date_range("2026-01-01", periods=3, freq="30s"),
            stopping_cdf_target=[0.0, 0.5, 1.0],
            stopping_cdf_weight=[1.0, 1.0, 1.0],
            stopping_cdf_instantaneous=True,
        )
    )
    logits = torch.tensor([-2.0, 0.0, 1.0], requires_grad=True)
    actual = _batch_loss(logits, rows, "cdf_ramp")
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([0.0, 0.5, 1.0])
    )
    assert torch.allclose(actual, expected)
    gradient = torch.autograd.grad(actual, logits)[0]
    assert torch.allclose(gradient, (logits.sigmoid() - torch.tensor([0.0, 0.5, 1.0])) / 3)
    cumulative = _batch_loss(logits, rows.drop(columns="stopping_cdf_instantaneous"), "cdf_ramp")
    assert torch.isfinite(cumulative) and not torch.allclose(actual, cumulative)
