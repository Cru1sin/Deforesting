import pickle
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


def test_stopping_loss_comparison_cli_parses_defaults_and_dispatches(monkeypatch) -> None:
    import train_pareto_boundary

    args = train_pareto_boundary.build_parser().parse_args(
        ["--action", "compare-stopping-losses", "--task", "cop-classification"]
    )
    assert args.stopping_architectures == [
        "r_sensor", "r_sensor_rgb", "chen_rgb", "new_sensor", "new_rgb_difference",
    ]
    assert args.stopping_losses == ["after_optimum", "cop_stopping"]
    assert args.seeds == [0, 1, 2]

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "image_models.stopping_loss_comparison",
        types.SimpleNamespace(run=calls.append),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_pareto_boundary.py",
            "--action", "compare-stopping-losses",
            "--task", "cop-classification",
            "--stopping-architectures", "new_sensor", "chen_rgb",
            "--stopping-losses", "cop_stopping",
            "--seeds", "5", "8",
        ],
    )

    train_pareto_boundary.main()

    [dispatched] = calls
    assert dispatched.stopping_architectures == ["new_sensor", "chen_rgb"]
    assert dispatched.stopping_losses == ["cop_stopping"]
    assert dispatched.seeds == [5, 8]
    assert dispatched.output == Path("output/image_models/stopping_loss_comparison")


def test_stopping_loss_comparison_requires_classification_task(monkeypatch) -> None:
    import train_pareto_boundary

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_pareto_boundary.py", "--action", "compare-stopping-losses"],
    )

    with pytest.raises(SystemExit):
        train_pareto_boundary.main()


def test_fold_exclusions_and_changed_run_settings(tmp_path: Path) -> None:
    from train_pareto_boundary import build_parser, fold_exclusions, save_settings

    folds = fold_exclusions(["d", "b", "a", "c"])
    assert folds == {"a": "b", "b": "c", "c": "d", "d": "a"}
    path = tmp_path / "settings.json"
    save_settings(path, {"seed": 0})
    save_settings(path, {"seed": 0})
    with pytest.raises(ValueError, match="new output"):
        save_settings(path, {"seed": 1})
    args = build_parser().parse_args(["--action", "prepare", "--allow-extrapolation"])
    assert args.allow_extrapolation
    represent = build_parser().parse_args(["--action", "represent"])
    heads = build_parser().parse_args(["--action", "heads"])
    neural = build_parser().parse_args([
        "--action", "neural", "--reference-run", "output/test/reference"
    ])
    evaluate = build_parser().parse_args([
        "--action", "evaluate", "--runs", "output/test/a", "output/test/b"
    ])
    cross = build_parser().parse_args([
        "--action", "cross-input", "--runs", "output/test/ridge", "output/test/neural"
    ])
    vcnet = build_parser().parse_args([
        "--action", "vcnet", "--event-head", "multimodal_time_linear_z64_curvature",
        "--diagnose-state",
    ])
    probe = build_parser().parse_args(["--action", "probe"])
    transfer = build_parser().parse_args([
        "--action", "state-transfer", "--runs", "output/d32", "output/t32"
    ])
    refit = build_parser().parse_args([
        "--action", "refit", "--selected-method", "raw_ridge_ch_mlp"
    ])
    overview = build_parser().parse_args([
        "--action", "overview", "--runs", *[f"output/run-{index}" for index in range(7)]
    ])
    assert represent.representations.name == "representations"
    assert heads.action == "heads"
    assert neural.reference_run.name == "reference"
    assert [path.name for path in evaluate.runs] == ["a", "b"]
    assert [path.name for path in cross.runs] == ["ridge", "neural"]
    assert vcnet.event_head == ["multimodal_time_linear_z64_curvature"]
    assert vcnet.diagnose_state
    assert probe.action == "probe"
    assert [path.name for path in transfer.runs] == ["d32", "t32"]
    assert refit.selected_method == "raw_ridge_ch_mlp"
    assert overview.action == "overview"
    assert len(overview.runs) == 7
    audit = build_parser().parse_args(["--action", "audit", "--task", "cop-classification"])
    assert audit.action == "audit"


def test_temporal_rows_can_load_without_a_fold_teacher(tmp_path: Path) -> None:
    from train_pareto_boundary import load_fold_rows

    pd.DataFrame({
        "row_id": ["row-1"], "experiment_id": ["exp_train"],
        "is_teacher_candidate": [True],
    }).to_parquet(tmp_path / "base.parquet", index=False)

    result = load_fold_rows(
        tmp_path, tmp_path, ("exp_missing",), include_rgb=False,
        include_teacher=False,
    )

    assert result.row_id.tolist() == ["row-1"]


def test_refit_keeps_ridge_validity_separate_from_event_start_features(
    tmp_path: Path,
) -> None:
    from train_pareto_boundary import _refit_event_tables

    pd.DataFrame({"energy_event_valid": [True]}).to_parquet(
        tmp_path / "events.parquet", index=False
    )
    pd.DataFrame({"dinov2_000": [1.0]}).to_parquet(
        tmp_path / "event_starts.parquet", index=False
    )

    ridge_events, event_starts = _refit_event_tables(tmp_path)

    assert ridge_events.columns.tolist() == ["energy_event_valid"]
    assert event_starts.columns.tolist() == ["dinov2_000"]


def test_state_checkpoint_reuses_inner_only_for_the_same_validation_fold(
    tmp_path: Path,
) -> None:
    from train_pareto_boundary import load_state_checkpoints

    folder = tmp_path / "seed_0" / "folds" / "exp_test"
    folder.mkdir(parents=True)
    with (folder / "checkpoint.pkl").open("wb") as stream:
        pickle.dump({
            "checkpoint": {"name": "outer"},
            "inner_checkpoint": {"name": "inner"},
            "inner_validation_experiment": "exp_matching",
        }, stream)

    outer, inner = load_state_checkpoints(
        tmp_path, 0, "exp_test", "exp_matching"
    )
    assert outer["name"] == "outer"
    assert inner["name"] == "inner"
    assert load_state_checkpoints(tmp_path, 0, "exp_test", "exp_other")[1] is None


def test_replace_economic_inputs_swaps_all_eight_fields_only() -> None:
    from train_pareto_boundary import CURRENT_ECONOMIC_INPUTS, replace_economic_inputs

    row = {"z_00": 7.0}
    for index, name in enumerate(CURRENT_ECONOMIC_INPUTS):
        row[f"online_{name}"] = float(index)
        row[f"neural_{name}"] = float(index + 10)
    rows = pd.DataFrame([row])

    ridge = replace_economic_inputs(rows, "ridge")
    neural = replace_economic_inputs(rows, "neural")

    assert ridge.equals(rows)
    assert neural.z_00.item() == 7.0
    assert [neural[f"online_{name}"].item() for name in CURRENT_ECONOMIC_INPUTS] == [
        float(index + 10) for index in range(8)
    ]
    assert [neural[f"neural_{name}"].item() for name in CURRENT_ECONOMIC_INPUTS] == [
        float(index + 10) for index in range(8)
    ]


def test_run_predictions_deduplicate_identical_methods_and_keep_readable_names(
    tmp_path: Path,
) -> None:
    from train_pareto_boundary import load_run_predictions

    common = pd.DataFrame({
        "row_id": ["a"], "cycle_name": ["cycle"],
        "heldout_experiment": ["exp"], "method": ["s0"], "seed": [0],
        "image_time": [pd.Timestamp("2026-01-01")],
        "teacher_time": [pd.Timestamp("2026-01-01")],
        "logit": [1.], "prediction": [1], "target": [1.],
    })
    runs = [tmp_path / name for name in ("one", "two")]
    for run in runs:
        run.mkdir()
        common.to_parquet(run / "predictions.parquet", index=False)

    result = load_run_predictions(runs)

    assert len(result) == 1
    assert result.method_name.iloc[0] == "multimodal_latent"
    assert result.method_label.iloc[0] == "RGB+sensor latent"


def test_run_predictions_reject_conflicting_duplicate_predictions(tmp_path: Path) -> None:
    from train_pareto_boundary import load_run_predictions

    runs = [tmp_path / name for name in ("one", "two")]
    for index, run in enumerate(runs):
        run.mkdir()
        pd.DataFrame({
            "row_id": ["a"], "cycle_name": ["cycle"],
            "heldout_experiment": ["exp"], "method": ["s0"], "seed": [0],
            "image_time": [pd.Timestamp("2026-01-01")],
            "teacher_time": [pd.Timestamp("2026-01-01")],
            "logit": [float(index)], "prediction": [index], "target": [1.],
        }).to_parquet(run / "predictions.parquet", index=False)

    with pytest.raises(ValueError, match="conflicting duplicate predictions"):
        load_run_predictions(runs)


def test_selected_inputs_reject_conflicting_fold_definitions(tmp_path: Path) -> None:
    from train_pareto_boundary import load_selected_inputs

    runs = [tmp_path / name for name in ("one", "two")]
    for method, run in zip(("s3", "s0"), runs, strict=True):
        run.mkdir()
        pd.DataFrame({
            "heldout_experiment": ["exp"], "selected_for_s4": [method]
        }).to_csv(run / "selected_inputs.csv", index=False)

    with pytest.raises(ValueError, match="conflicting S4 selected inputs"):
        load_selected_inputs(runs)


def test_extrapolation_definition_must_match_prepared_data(tmp_path: Path) -> None:
    import json

    from train_pareto_boundary import require_matching_extrapolation_setting

    (tmp_path / "settings.json").write_text(json.dumps({"allow_extrapolation": False}))
    require_matching_extrapolation_setting(tmp_path, False)
    with pytest.raises(ValueError, match="extrapolation setting differs"):
        require_matching_extrapolation_setting(tmp_path, True)


def test_native_frame_cohort_keeps_only_observed_preparation_prefixes() -> None:
    from train_pareto_boundary import native_frame_cycles

    catalog = pd.DataFrame(
        {
            "cycle_name": ["short", "normal"],
            "heating_start": ["2026-08-07 10:02:59"] * 2,
            "defrost_preparation_start": ["2026-08-07 10:03:08", "2026-08-07 11:03:08"],
        }
    )
    images = pd.DataFrame(
        {
            "cycle_name": ["short", "normal"],
            "camera_role": ["front"] * 2,
            "image_time": ["2026-08-07 10:04:00"] * 2,
        }
    )
    assert native_frame_cycles(catalog, images) == {"normal"}


def test_visual_latent_merge_keeps_the_frozen_z_coordinates_only() -> None:
    from train_pareto_boundary import merge_visual_latents

    rows = pd.DataFrame({"row_id": ["a", "b"], "value": [1, 2]})
    latents = pd.DataFrame({
        "row_id": ["a", "b"], "z_00": [3.0, 4.0], "nz_00": [5.0, 6.0]
    })

    result = merge_visual_latents(rows, latents)

    assert result.columns.tolist() == ["row_id", "value", "z_00"]


def test_pareto_statuses_require_knee_and_rgb_bracketing() -> None:
    from train_pareto_boundary import derive_pareto_cycle_statuses

    origin = pd.Timestamp("2026-08-07 10:00:00")
    rows = pd.DataFrame(
        {
            "cycle_name": ["covered"] * 3 + ["no_knee"] * 2 + ["ends_early"] * 2,
            "candidate_defrost_time": [
                origin,
                origin + pd.Timedelta(minutes=1),
                origin + pd.Timedelta(minutes=2),
                origin,
                origin + pd.Timedelta(minutes=2),
                origin,
                origin + pd.Timedelta(seconds=30),
            ],
            "is_frame": [True, False, True, True, True, True, True],
            "teacher_time": [
                origin + pd.Timedelta(minutes=1),
                origin + pd.Timedelta(minutes=1),
                origin + pd.Timedelta(minutes=1),
                pd.NaT,
                pd.NaT,
                origin + pd.Timedelta(minutes=1),
                origin + pd.Timedelta(minutes=1),
            ],
        }
    )

    result = derive_pareto_cycle_statuses(rows).set_index("cycle_name")

    assert tuple(result.loc["covered"]) == ("valid", "valid")
    assert tuple(result.loc["no_knee"]) == ("invalid", "invalid")
    assert tuple(result.loc["ends_early"]) == ("valid", "invalid")


def test_pareto_cv_ready_cycles_requires_all_three_statuses() -> None:
    from train_pareto_boundary import pareto_cv_ready_cycles

    records = [
        {
            "cycle_name": "ready",
            "status": "valid",
            "pareto_knee_status": "valid",
            "rgb_knee_coverage_status": "valid",
        },
        {
            "cycle_name": "bad_cycle",
            "status": "invalid",
            "pareto_knee_status": "valid",
            "rgb_knee_coverage_status": "valid",
        },
        {
            "cycle_name": "no_knee",
            "status": "valid",
            "pareto_knee_status": "invalid",
            "rgb_knee_coverage_status": "invalid",
        },
        {
            "cycle_name": "no_rgb_coverage",
            "status": "valid",
            "pareto_knee_status": "valid",
            "rgb_knee_coverage_status": "invalid",
        },
    ]

    assert pareto_cv_ready_cycles({"cycles": records}) == {"ready"}

    records.append(
        {
            "cycle_name": "extrapolation_ready",
            "status": "valid",
            "pareto_extrapolated_knee_status": "valid",
            "rgb_extrapolated_knee_coverage_status": "valid",
        }
    )
    assert pareto_cv_ready_cycles(
        {"cycles": records}, allow_extrapolation=True
    ) == {"extrapolation_ready"}
