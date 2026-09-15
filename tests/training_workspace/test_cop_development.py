import numpy as np
import pandas as pd
import pytest
import torch
from types import SimpleNamespace

from image_models.cop_development import (
    CHECKPOINT_EPOCHS,
    StaticNearOptimalClassifier,
    _add_visual_history_rows,
    _input_available,
    _load_refit_recipe,
    _outer_evaluation_status,
    _predict_checkpoint,
    _front_images,
    _training_targets,
    _tensors,
    candidate_clock,
    cycle_equal_weights,
    development_feature_columns,
    label_complete_curves,
    score_classifier,
    select_epoch_threshold,
)


def test_development_features_and_matched_model_contract():
    numeric, visual = development_feature_columns()
    assert len(numeric) == 191
    assert len([name for name in numeric if name.startswith("stat_")]) == 189
    assert "stat_cop_current" in numeric
    assert not any("candidate_cop" in name for name in numeric)
    assert numeric[-2:] == ["causal_elapsed_minutes", "causal_total_electricity_kwh"]
    assert len(visual) == 384

    model = StaticNearOptimalClassifier(len(numeric))
    assert [layer.out_features for layer in model.numeric if isinstance(layer, torch.nn.Linear)] == [128, 64, 32]
    assert [layer.out_features for layer in model.visual if isinstance(layer, torch.nn.Linear)] == [64, 32]
    assert [layer.out_features for layer in model.classifier if isinstance(layer, torch.nn.Linear)] == [64, 32, 1]
    assert not any(isinstance(layer, torch.nn.Dropout) for layer in model.modules())
    output = model(torch.zeros(2, len(numeric)), None)
    assert output.shape == (2,)
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in model.modules()
               if isinstance(layer, torch.nn.Linear))

    sensor_model = StaticNearOptimalClassifier(len(numeric), visual_width=0)
    assert sensor_model.visual is None
    assert not any(name.startswith("visual.") for name in sensor_model.state_dict())
    assert next(
        layer for layer in sensor_model.classifier if isinstance(layer, torch.nn.Linear)
    ).in_features == 32
    assert sensor_model(torch.zeros(2, len(numeric))).shape == (2,)

    class Identity:
        def transform(self, values):
            return values.to_numpy()

    frame = pd.DataFrame(0., index=range(2), columns=[*numeric, *visual])
    assert _tensors(frame, Identity(), Identity(), "sensor")[1] is None
    assert _tensors(frame, Identity(), Identity(), "rgb")[1].shape == (2, 384)

    delta_model = StaticNearOptimalClassifier(len(numeric), visual_width=768)
    assert next(layer for layer in delta_model.visual
                if isinstance(layer, torch.nn.Linear)).in_features == 768
    delta_columns = [*visual, *[f"delta_{name}" for name in visual]]
    delta_frame = pd.DataFrame(0., index=range(2), columns=[*numeric, *delta_columns])
    assert _tensors(
        delta_frame, Identity(), Identity(), "rgb", mechanism="delta"
    )[1].shape == (2, 768)


def test_clock_is_heating_anchored_but_reference_accounting_starts_at_recovery():
    row = pd.Series({
        "cycle_name": "c", "experiment_id": "e",
        "heating_start": "2026-01-01 00:00:07",
        "stable_heating_start": "2026-01-01 00:02:11",
        "observed_defrost_preparation_start": "2026-01-01 00:03:20",
        "observation_end": "2026-01-01 00:04:00",
    })
    clock = candidate_clock(row)
    assert clock.candidate_defrost_time.tolist() == list(pd.to_datetime([
        "2026-01-01 00:00:07", "2026-01-01 00:00:37", "2026-01-01 00:01:07",
        "2026-01-01 00:01:37", "2026-01-01 00:02:07", "2026-01-01 00:02:37",
        "2026-01-01 00:03:07",
    ]))
    assert clock.heating_accounting_start.nunique() == 1
    assert clock.heating_accounting_start.iloc[0] == pd.Timestamp("2026-01-01 00:02:11")
    assert clock.causal_elapsed_minutes.iloc[-1] == 3
    aligned = row.copy()
    aligned["observed_defrost_preparation_start"] = "2026-01-01 00:03:07"
    assert candidate_clock(aligned).candidate_defrost_time.max() == pd.Timestamp(
        "2026-01-01 00:02:37"
    )
    missing = row.copy()
    missing["observed_defrost_preparation_start"] = pd.NaT
    assert candidate_clock(missing).candidate_defrost_time.max() == pd.Timestamp(
        "2026-01-01 00:03:37"
    )


def test_labels_precede_availability_and_cycle_weights_are_equal():
    rows = pd.DataFrame({
        "cycle_name": ["a"] * 3 + ["b"] * 2,
        "cycle_cop": [1., 2., 1.98, 3., 2.],
        "cycle_cop_eligible": [True] * 5,
        "joint_input_available": [False, True, True, True, False],
    })
    labeled = label_complete_curves(rows, epsilon=.01)
    assert labeled.binary_target.tolist() == [0., 1., 1., 1., 0.]
    np.testing.assert_allclose(
        cycle_equal_weights(labeled.cycle_name),
        [5 / 6, 5 / 6, 5 / 6, 5 / 4, 5 / 4],
    )
    anchored = label_complete_curves(
        rows.loc[rows.cycle_name.eq("a")], epsilon=.01,
        reference_max_by_cycle={"a": 2.5},
    )
    assert anchored.binary_target.eq(0).all()


def test_threshold_selection_and_supported_proposals_keep_fixed_denominator():
    grid = pd.DataFrame([
        {"epoch": 1, "threshold": .4, "balanced_accuracy": .8,
         "macro_f1": .7, "validation_bce": .4},
        {"epoch": 3, "threshold": .6, "balanced_accuracy": .8,
         "macro_f1": .7, "validation_bce": .3},
        {"epoch": 3, "threshold": .7, "balanced_accuracy": .8,
         "macro_f1": .7, "validation_bce": .3},
    ])
    selected = select_epoch_threshold(grid)
    assert (selected.epoch, selected.threshold) == (3, .7)
    assert CHECKPOINT_EPOCHS == (1, 3, 5, 10, 20, 30, 50)

    times = pd.date_range("2026-01-01", periods=3, freq="30s")
    rows = pd.DataFrame({
        "cycle_name": sum(([name] * 3 for name in "abcde"), []),
        "candidate_defrost_time": list(times) * 5,
        "binary_target": [
            0., 1., 1., 0., np.nan, 1., 0., 0., 0.,
            0., 1., 1., np.nan, np.nan, np.nan,
        ],
        "probability": [
            .2, .8, .9, .8, .9, .9, .1, .2, .3,
            np.nan, np.nan, np.nan, .2, .8, .9,
        ],
    })
    metrics, cycles = score_classifier(rows, .5)
    status = cycles.set_index("cycle_name").two_of_three_status.to_dict()
    assert status == {
        "a": "supported_near_hit", "b": "unsupported_proposal",
        "c": "no_proposal", "d": "no_proposal", "e": "unsupported_proposal",
    }
    assert metrics["classified_cycle_count"] == 3
    assert metrics["both_class_cycles"] == 2
    assert metrics["single_class_cycles"] == 1
    assert metrics["no_prediction_cycles"] == 1
    assert metrics["no_supported_labels_cycles"] == 1

    undefined = pd.DataFrame([{
        "epoch": 1, "threshold": .5, "balanced_accuracy": np.nan,
        "macro_f1": np.nan, "validation_bce": np.nan,
    }])
    with pytest.raises(ValueError, match="no selectable"):
        select_epoch_threshold(undefined)


def test_label_smoothing_changes_only_training_targets_and_evaluation_stays_hard():
    hard = torch.tensor([0., 1.])
    torch.testing.assert_close(_training_targets(hard, "bce"), hard)
    torch.testing.assert_close(
        _training_targets(hard, "label-smoothing"), torch.tensor([.05, .95])
    )
    torch.testing.assert_close(hard, torch.tensor([0., 1.]))

    rows = pd.DataFrame({
        "cycle_name": ["a", "b"],
        "candidate_defrost_time": pd.to_datetime([
            "2026-01-01 00:00:00", "2026-01-01 00:00:00",
        ]),
        "binary_target": [0., 1.],
        "probability": [.2, .8],
    })
    metrics, _ = score_classifier(rows, .5)
    assert metrics["cycle_weighted_brier_score"] == pytest.approx(.04)
    assert metrics["cycle_weighted_expected_calibration_error"] == pytest.approx(.2)
    assert metrics["cycle_weighted_negative_log_likelihood"] == pytest.approx(-np.log(.8))
    assert rows.binary_target.tolist() == [0., 1.]


def test_missing_rgb_cache_preserves_complete_front_metadata(tmp_path):
    _, visual = development_feature_columns()
    metadata = pd.DataFrame({
        "camera_role": ["front", "front", "top"],
        "image_time": pd.to_datetime([
            "2026-01-01 00:00:10", "2026-01-01 00:00:40", "2026-01-01 00:00:20",
        ]),
        "file_name": ["early.jpg", "late.jpg", "top.jpg"],
    })
    images = _front_images(metadata, tmp_path / "missing.parquet", visual)
    assert images.file_name.tolist() == ["early.jpg", "late.jpg"]
    assert images[visual].isna().all().all()


def test_visual_delta_is_past_only_and_missing_history_stays_unavailable():
    visual = ["dinov2_000"]
    reference = pd.DataFrame({
        "candidate_defrost_time": pd.to_datetime([
            "2026-01-01 00:03:00", "2026-01-01 00:04:00",
        ]),
        "dinov2_000": [3., 5.],
        "joint_input_available": [True, True],
    })
    images = pd.DataFrame({
        "image_time": pd.to_datetime([
            "2026-01-01 00:00:00", "2026-01-01 00:01:30",
        ]),
        "file_name": ["past.jpg", "future_for_first.jpg"],
        "dinov2_000": [1., 100.],
    })
    original = _add_visual_history_rows(reference, images.iloc[:1], visual)
    with_future = _add_visual_history_rows(reference, images, visual)
    assert original.loc[0, "delta_dinov2_000"] == 2.
    assert with_future.loc[0, "delta_dinov2_000"] == 2.
    assert with_future.loc[0, "past_image_time"] == pd.Timestamp("2026-01-01 00:00:00")
    assert pd.isna(original.loc[1, "past_image_time"])
    assert not _input_available(original).iloc[1]

    keys = ["candidate_defrost_time", "binary_target", "development_input_available"]
    labeled = with_future.assign(binary_target=[0., 1.])
    sensor = labeled[keys].copy()
    static_rgb = labeled[keys].copy()
    delta_rgb = labeled[keys].copy()
    pd.testing.assert_frame_equal(sensor, static_rgb)
    pd.testing.assert_frame_equal(sensor, delta_rgb)


def test_refit_recipe_is_frozen_and_saved_state_reloads_identically(tmp_path):
    reference = tmp_path / "winner"
    reference.mkdir()
    (reference / "settings.json").write_text("""{
        "audit_data": "/tmp/audit", "dataset": "/tmp/dataset",
        "rgb_cache": "/tmp/rgb", "method": "sensor", "mechanism": "static",
        "development_loss": "label-smoothing", "require_visual_history": true,
        "epsilon": 0.01, "batch_size": 256, "seed": 0
    }""")
    pd.DataFrame([{"epoch": 20, "threshold": .25}]).to_csv(
        reference / "selected_configuration.csv", index=False
    )
    recipe = _load_refit_recipe(reference)
    assert recipe["selected_epoch"] == 20
    assert recipe["threshold"] == .25
    assert recipe["development_loss"] == "label-smoothing"

    class Identity:
        def transform(self, values):
            return values.to_numpy()

    torch.manual_seed(0)
    model = StaticNearOptimalClassifier(2)
    rows = pd.DataFrame({"a": [0., 1.], "b": [1., 0.]})
    checkpoint = {
        "method": "sensor", "development_mechanism": "static",
        "numeric_columns": ["a", "b"], "visual_columns": [],
        "numeric_preprocessor": Identity(), "visual_preprocessor": None,
        "model_state_dict": model.state_dict(),
    }
    with torch.no_grad():
        expected = model(torch.tensor(rows[["a", "b"]].to_numpy(), dtype=torch.float32))
        expected = expected.sigmoid().numpy()
    np.testing.assert_allclose(_predict_checkpoint(rows, checkpoint), expected)


def test_refit_reads_an_explicit_base_root_and_outer_empty_labels_are_unevaluable(
    tmp_path, monkeypatch,
):
    from image_models import relative_cop

    base_root = tmp_path / "winner" / "base"
    base_root.mkdir(parents=True)
    pd.DataFrame({"source_value": [7]}).to_parquet(base_root / "cycle.parquet")
    output = tmp_path / "final"
    (output / "ridge").mkdir(parents=True)
    monkeypatch.setattr(
        relative_cop, "ridge_parameters",
        lambda events, omitted: {"training_experiment_ids": []},
    )
    monkeypatch.setattr(
        relative_cop, "apply_reference",
        lambda base, model, rgb, include_history=True: base,
    )
    args = SimpleNamespace(
        output=output, reference_run=tmp_path / "unused", quality_filtered=True,
        require_rgb_input=False, rgb="off",
    )
    rows, _ = relative_cop.build_fold_rows(
        args,
        pd.DataFrame({"cycle_name": ["cycle"], "experiment_id": ["experiment"]}),
        pd.DataFrame(), (), include_history=False, base_root=base_root,
    )
    assert rows.source_value.tolist() == [7]

    status, supported = _outer_evaluation_status(pd.DataFrame({
        "binary_target": [np.nan, np.nan], "probability": [.2, .8],
    }))
    assert status == "unevaluable"
    assert supported.empty
    status, _ = _outer_evaluation_status(pd.DataFrame({
        "binary_target": [1., 1.], "probability": [.2, .8],
    }))
    assert status == "single_class"


def test_cli_exposes_static_development_stage():
    from train_pareto_boundary import build_parser

    args = build_parser().parse_args([
        "--task", "cop-classification", "--action", "develop",
        "--development-mechanism", "static", "--rgb", "off",
    ])
    assert args.action == "develop"
    assert args.development_mechanism == "static"
    assert args.development_loss == "bce"
    smooth = build_parser().parse_args(["--development-loss", "label-smoothing"])
    assert smooth.development_loss == "label-smoothing"
    delta = build_parser().parse_args([
        "--development-mechanism", "delta", "--rgb", "on",
    ])
    assert delta.development_mechanism == "delta"
    history = build_parser().parse_args(["--require-visual-history"])
    assert history.require_visual_history
    comparison = build_parser().parse_args([
        "--task", "cop-classification", "--action", "compare-frozen-policies",
        "--runs", "frozen", "sensor", "rgb", "chen",
    ])
    assert comparison.action == "compare-frozen-policies"
    assert len(comparison.runs) == 4
