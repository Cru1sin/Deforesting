from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from image_models import checkpoints, comparisons, evaluation
from tools import result_index


def test_camera_is_read_from_saved_evidence_and_mismatch_stops_before_dataset(
    tmp_path, monkeypatch
):
    import pickle

    import pandas as pd

    import output_layout

    monkeypatch.setattr(output_layout, "PROJECT", tmp_path)
    path = tmp_path / "arbitrary_filename.pkl"
    saved = dict(checkpoint={}, cycle_metrics=pd.DataFrame(dict(version=["left_deployment"])))
    path.write_bytes(pickle.dumps(saved))
    assert checkpoints.checkpoint_camera(path) == "left"
    output = tmp_path / "temp/test/replay"
    with pytest.raises(ValueError, match="Checkpoint camera"):
        evaluation.evaluate_external(
            None,
            tmp_path / "missing_dataset",
            ["new"],
            output,
            checkpoint=path,
            camera_role="front",
        )
    assert not output.exists()
    saved["checkpoint"]["camera_role"] = "top"
    path.write_bytes(pickle.dumps(saved))
    assert checkpoints.checkpoint_camera(path) == "top"
    saved["checkpoint"].clear()
    saved["cycle_metrics"]["version"] = "unknown"
    path.write_bytes(pickle.dumps(saved))
    with pytest.raises(ValueError, match="Cannot establish training camera"):
        checkpoints.checkpoint_camera(path)
    path = checkpoints.checkpoint_path(tmp_path / "run/training", "model.pkl")
    path.parent.mkdir(parents=True)
    path.write_bytes(pickle.dumps(saved))
    settings = tmp_path / "run/training/settings.json"
    settings.parent.mkdir()
    settings.write_text('{"camera_role": "top_close"}')
    assert checkpoints.checkpoint_camera(path) == "top_close"


def test_external_checkpoint_masks_missing_inputs_and_rejects_training_overlap(
    tmp_path, monkeypatch
):
    import pickle

    import pandas as pd

    from image_models import policy

    rows = pd.DataFrame(
        dict(cycle_name=["new", "new"], experiment_id="new_exp", input_valid=[True, False])
    )
    model = dict(
        training_experiments=["old_exp"],
        training_cycles=["new"],  # A different Dataset may reuse the same cycle number.
        threshold=0.5,
        seed=42,
        trigger_rule="cdf",
        reset_history=True,
        numeric_columns=[],
    )
    path = tmp_path / "frozen.pkl"
    with path.open("wb") as stream:
        pickle.dump(
            dict(checkpoint=model, cycle_metrics=pd.DataFrame(dict(version=["model"]))), stream
        )

    def predict(query, checkpoint):
        assert query.input_valid.all()
        return query.assign(probability=0.2, decision_margin=-1.4)

    def score(predicted, reference, version, seed, fold, strategy, rule):
        assert pd.isna(predicted.probability.iloc[1])
        assert rule == "cdf" and strategy == "first_positive_terminal"
        return pd.DataFrame(dict(cycle_name=["new"], selected_cop=[2.0]))

    monkeypatch.setattr(policy, "predict_policy", predict)
    monkeypatch.setattr(evaluation, "score_stopping_predictions", score)
    result = evaluation._score_external_checkpoint(path, rows, pd.DataFrame(), {"new": "60C_low"})
    assert result.operating_condition.item() == "60C_low" and result.reset_history.item()
    with pytest.raises(ValueError, match="overlaps"):
        evaluation._score_external_checkpoint(
            path, rows.assign(experiment_id="old_exp"), pd.DataFrame(), {}
        )


def test_checkpoint_location_preserves_legacy_reads_and_detects_duplicates():
    with TemporaryDirectory() as directory:
        old_root = checkpoints.ROOT
        try:
            checkpoints.ROOT = Path(directory) / "training_checkpoint"
            run = Path(directory) / "experiment"
            run.mkdir()
            target = checkpoints.checkpoint_path(run, "head_seed0.pkl")
            assert target == run / "checkpoints" / "head_seed0" / "head_seed0.pkl"
            legacy = run / "head_seed0.pkl"
            legacy.write_bytes(b"frozen model")
            assert checkpoints.checkpoint_path(run, legacy.name) == legacy
            target.parent.mkdir(parents=True)
            legacy.rename(target)
            assert checkpoints.checkpoint_files(run) == [target]
            assert checkpoints.checkpoint_path(run, target.name).read_bytes() == b"frozen model"
            legacy.write_bytes(b"ambiguous model")
            with pytest.raises(ValueError, match="Duplicate"):
                checkpoints.checkpoint_files(run)
        finally:
            checkpoints.ROOT = old_root


def test_prune_keeps_requested_models_and_historical_results():
    with TemporaryDirectory() as directory:
        old_root = checkpoints.ROOT
        try:
            checkpoints.ROOT = Path(directory) / "training_checkpoint"
            for name in ["keep", "obsolete"]:
                folder = checkpoints.ROOT / name / "checkpoints" / "fold"
                folder.mkdir(parents=True)
                (folder / "fold.pkl").write_bytes(b"model")
                (folder / "cycles.csv").write_text("cycle_name\n")
                result = Path(directory) / name
                result.mkdir()
                (result / "summary.csv").write_text("frozen results")
            with pytest.raises(ValueError):
                checkpoints.prune_checkpoints(["misspelled"])
            assert checkpoints.prune_checkpoints(["keep"]) == 1
            assert (checkpoints.ROOT / "keep/checkpoints/fold/fold.pkl").read_bytes() == b"model"
            assert not list((checkpoints.ROOT / "obsolete").rglob("*.pkl"))
            assert (Path(directory) / "obsolete/summary.csv").read_text() == "frozen results"
            result_index.write_output_index()
            assert (Path(directory) / "index.html").exists()
        finally:
            checkpoints.ROOT = old_root


def test_inner_search_records_follow_their_checkpoint(tmp_path):
    run = tmp_path / "training_checkpoint/experiment/training"
    run.mkdir(parents=True)
    model = checkpoints.checkpoint_path(run, "GRU_exp_20261007_seed0.pkl")
    model.parent.mkdir(parents=True)
    model.write_bytes(b"frozen")
    names = [
        "inner_GRU_exp_20261007_fold0_seed0.csv",
        "exp_20261007_seed0_trials.csv",
        "exp_20261007_seed0.sqlite3",
    ]
    for name in names:
        (run / name).write_bytes(b"original")
    assert checkpoints.organize_fold_history(run) == 3
    assert all((model.parent / name).read_bytes() == b"original" for name in names)


def test_ablation_tables_keep_epochs_and_separate_partial_from_full_loeo(tmp_path, monkeypatch):
    import json
    import pickle

    import pandas as pd
    import torch

    from image_models import policy

    monkeypatch.setattr(policy, "restore_policy", lambda model: torch.nn.Linear(2, 1))
    monkeypatch.setattr(policy, "build_model", lambda architecture, **kwargs: torch.nn.Linear(4, 1))
    monkeypatch.setattr(
        policy, "summarize_metrics", lambda metrics: {"cohort_cycles": len(metrics)}
    )
    runs = []
    for name, tests in [("screen", ["exp_a"]), ("full", ["exp_a", "exp_b"])]:
        run = tmp_path / name
        run.mkdir()
        (run / "settings.json").write_text(json.dumps({"folds": dict.fromkeys(tests)}))
        runs.append(run)
        for outer in tests:
            path = checkpoints.checkpoint_path(run, f"model_{outer}.pkl")
            path.parent.mkdir(parents=True)
            model = dict(
                seed=0,
                architecture="GRU32",
                loss_name="cdf_ramp",
                numeric_columns=["elapsed"],
                visual_columns=["dino"],
                selected_epoch=2,
                training_experiments=[e for e in ["exp_a", "exp_b"] if e != outer],
                losses=pd.DataFrame({"epoch": [1, 2], "loss": [0.3, 0.2]}),
                pooled_validation_scores=pd.DataFrame({"validation_headroom": [70.0]}),
            )
            metrics = pd.DataFrame({"experiment_id": [outer], "version": ["baseline"], "seed": [0]})
            path.write_bytes(pickle.dumps({"checkpoint": model, "cycle_metrics": metrics}))
            pd.DataFrame({"epoch": [1, 2], "validation_cdf_loss": [0.4, 0.3]}).to_csv(
                path.parent / "inner_baseline_fold0_seed0.csv", index=False
            )
            pd.DataFrame(
                {
                    "number": [0],
                    "params_learning_rate": [0.01],
                    "params_architecture": ["GRU32"],
                    "params_dropout_place": ["none"],
                    "user_attrs_selected_epoch": [1],
                    "state": ["COMPLETE"],
                }
            ).to_csv(path.parent / "search_trials.csv", index=False)
    output = tmp_path / "tables"
    comparisons.write_ablation_tables(runs, output)
    scores = pd.read_csv(output / "ablation_summary.csv")
    curves = pd.read_csv(output / "ablation_learning_curves.csv")
    assert set(scores.loc[scores.experiment == "screen", "evaluation_scope"]) == {
        "partial_experiment_holdout"
    }
    assert set(scores.loc[scores.experiment == "full", "evaluation_scope"]) == {
        "full_experiment_loeo"
    }
    assert len(curves) == 12
    assert set(curves.scope) == {"inner_training_curve", "refit_training_curve"}
    assert set(curves.parameters) == {3}
    trials = pd.read_csv(output / "ablation_trials.csv")
    assert len(trials) == 3
    assert set(trials.learning_rate) == {0.01}
    assert set(trials.parameters) == {5}
    assert set(trials.selected_epoch) == {1}
    assert trials.visual_time_shared.all() and not trials.output_time_shared.any()
    grid = runs[-1] / "shared_grid"
    grid.mkdir()
    pd.DataFrame(
        [
            dict(
                hidden_width=64,
                mlp_layers=2,
                learning_rate=0.003,
                parameters=30000,
                epoch=2,
                validation_cycles=999,
                development_outer_experiment="exp_a",
                selected=True,
            )
        ]
    ).to_csv(runs[-1] / "shared_mlp_grid.csv", index=False)
    for fold in range(2):
        pd.DataFrame(dict(epoch=[1, 2], validation_cycles=[1, 1])).to_csv(
            grid / f"w64_d2_lr0.003_fold{fold}.csv", index=False
        )
    comparisons.write_ablation_tables(runs, output)
    curves = pd.read_csv(output / "ablation_learning_curves.csv")
    shared = curves.loc[curves.scope.eq("shared_inner_training_curve")]
    assert len(shared) == 4 and set(shared.epoch) == {1, 2}
    assert set(shared.validation_cycles) == {1}
    assert set(shared.parameters) == {30000}
