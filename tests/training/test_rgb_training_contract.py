from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn


def _module():  # type: ignore[no-untyped-def]
    path = Path("scripts/training/train_rgb_smoke_models.py")
    spec = importlib.util.spec_from_file_location("rgb_smoke_models", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_batch_norm_stays_in_evaluation_mode() -> None:
    module = _module()
    model = nn.Sequential(nn.BatchNorm2d(3), nn.Flatten(), nn.Linear(12, 2))
    for parameter in model[0].parameters():
        parameter.requires_grad = False

    model.train()
    module._keep_frozen_batch_norm_eval(model)

    assert not model[0].training
    assert model[2].training


def test_boundary_weights_give_each_regret_target_stratum_equal_mass() -> None:
    module = _module()
    rows = pd.DataFrame(
        {
            "relative_regret": [0.005, 0.005, 0.02, 0.02, 0.02, 0.005, 0.03],
            "target": [0, 0, 0, 0, 0, 1, 1],
        }
    )

    weights = module._boundary_sample_weights(rows).numpy()
    strata = pd.MultiIndex.from_arrays(
        [rows["relative_regret"].le(0.01), rows["target"]]
    )
    masses = pd.Series(weights).groupby(strata).sum()

    assert masses.nunique() == 1


def test_stage_selection_rejects_more_than_one_point_full_f1_drop() -> None:
    module = _module()
    metrics = [
        {"stage": "head", "split": "validation", "macro_f1": 0.80},
        {"stage": "head", "split": "near_1pct_validation", "macro_f1": 0.60},
        {"stage": "finetune", "split": "validation", "macro_f1": 0.79},
        {"stage": "finetune", "split": "near_1pct_validation", "macro_f1": 0.70},
        {"stage": "boundary", "split": "validation", "macro_f1": 0.789},
        {"stage": "boundary", "split": "near_1pct_validation", "macro_f1": 0.99},
    ]

    selected = module._select_stage(metrics)

    assert selected == {
        "stage": "finetune",
        "checkpoint": "best_finetune.pt",
        "validation_macro_f1": 0.79,
        "near_1pct_validation_macro_f1": 0.70,
    }


@pytest.mark.parametrize("extra, expected", [([], 0), (["--boundary-epochs", "3"], 3)])
def test_boundary_epochs_cli_is_optional_and_forwarded(monkeypatch, extra, expected) -> None:
    module = _module()
    captured = []
    monkeypatch.setattr(module, "run", captured.append)
    monkeypatch.setattr(sys, "argv", ["train_rgb_smoke_models.py", "--run-id", "test", *extra])

    module.main()

    assert captured[0].boundary_epochs == expected


def test_adaptation_cli_is_optional_and_forwarded(monkeypatch) -> None:
    module = _module()
    captured = []
    monkeypatch.setattr(module, "run", captured.append)
    checkpoint = Path("mixed.pt")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_rgb_smoke_models.py",
            "--run-id",
            "test",
            "--init-checkpoint",
            str(checkpoint),
            "--adapt-epochs",
            "3",
            "--adapt-lr",
            "2e-5",
        ],
    )

    module.main()

    assert captured[0].init_checkpoint == checkpoint
    assert captured[0].adapt_epochs == 3
    assert captured[0].adapt_lr == 2e-5


def test_boundary_stage_reuses_finetune_learning_rate() -> None:
    module = _module()
    args = SimpleNamespace(
        init_checkpoint=None,
        head_epochs=5,
        finetune_epochs=4,
        boundary_epochs=3,
        lr=1e-3,
        finetune_lr=1e-4,
    )

    assert module._stage_plan(args) == [
        ("head", 5, 1e-3),
        ("finetune", 4, 1e-4),
        ("boundary", 3, 1e-4),
    ]


def test_adaptation_is_the_only_stage_and_unfreezes_layer4_and_classifier() -> None:
    module = _module()
    args = SimpleNamespace(
        init_checkpoint=Path("mixed.pt"),
        adapt_epochs=3,
        adapt_lr=1e-5,
        head_epochs=5,
        finetune_epochs=4,
        boundary_epochs=3,
        lr=1e-3,
        finetune_lr=1e-4,
    )
    model = nn.Module()
    model.feature_extractor = nn.Sequential(*(nn.Linear(1, 1) for _ in range(8)))
    model.classifier = nn.Linear(1, 2)

    module._set_stage(model, "adapt")

    assert module._stage_plan(args) == [("adapt", 3, 1e-5)]
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert all(parameter.requires_grad for parameter in model.feature_extractor[7].parameters())
    assert not any(
        parameter.requires_grad
        for layer in model.feature_extractor[:7]
        for parameter in layer.parameters()
    )


def test_zero_epoch_adaptation_loads_complete_checkpoint_as_fallback(
    tmp_path, monkeypatch
) -> None:
    module = _module()

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.BatchNorm2d(3), *(nn.Linear(1, 1) for _ in range(7))
            )
            self.classifier = nn.Linear(1, 2)

        def forward(self, images):  # type: ignore[no-untyped-def]
            batch = len(images)
            self.feature_extractor[0](images)
            return torch.zeros(batch, 2), torch.zeros(batch, 2048)

    rows = pd.DataFrame(
        [
            {
                "split": split,
                "target": target,
                "relative_regret": regret,
                "absolute_path": tmp_path / "unused.png",
                "image_path": f"{split}-{target}-{regret}.png",
                "cycle_name": f"cycle-{split}",
                "experiment_id": f"experiment-{split}",
                "camera_role": "front",
                "image_time": pd.Timestamp("2026-01-01"),
            }
            for split in ("train", "validation", "test")
            for target in (0, 1)
            for regret in (0.005, 0.02)
        ]
    )
    initial = TinyModel()
    for parameter in initial.parameters():
        nn.init.constant_(parameter, 0.25)
    init_checkpoint = tmp_path / "mixed.pt"
    torch.save({"model_state_dict": initial.state_dict()}, init_checkpoint)
    monkeypatch.setattr(module, "BinaryResNet50", TinyModel)
    monkeypatch.setattr(module, "preferred_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(module, "_transforms", lambda: (None, None))
    monkeypatch.setattr(module, "_load_rows", lambda *args, **kwargs: (rows, {}))

    def evaluate(model, selected, *args):  # type: ignore[no-untyped-def]
        assert all(
            torch.equal(value, initial.state_dict()[name])
            for name, value in model.state_dict().items()
        )
        probabilities = np.array(
            [[0.9, 0.1] if target == 0 else [0.1, 0.9] for target in selected["target"]]
        )
        return module._metrics(selected["target"].to_numpy(), probabilities), probabilities

    monkeypatch.setattr(module, "_evaluate", evaluate)
    args = SimpleNamespace(
        dataset=tmp_path,
        labels=tmp_path / "labels.parquet",
        candidates=tmp_path / "candidates.parquet",
        camera_group="front",
        heat_basis="water",
        head_epochs=5,
        finetune_epochs=5,
        boundary_epochs=3,
        init_checkpoint=init_checkpoint,
        adapt_epochs=0,
        batch_size=2,
        workers=0,
        lr=1e-3,
        finetune_lr=1e-4,
        adapt_lr=1e-5,
        run_id="adapt-zero",
        output=tmp_path / "output",
        limit_per_split=0,
    )

    run_dir = module.run(args)

    fallback = torch.load(run_dir / "best_adapt.pt", map_location="cpu")
    assert fallback["stage"] == "adapt"
    batch_norm_buffers = {
        name for name in initial.state_dict() if "running_" in name or "num_batches" in name
    }
    assert batch_norm_buffers
    assert all(
        torch.equal(fallback["model_state_dict"][name], initial.state_dict()[name])
        for name in batch_norm_buffers
    )
    assert all(
        torch.equal(value, initial.state_dict()[name])
        for name, value in fallback["model_state_dict"].items()
    )
    selected = json.loads((run_dir / "selected_stage.json").read_text())
    assert selected["stage"] == "adapt"


def test_architecture_probe_uses_eval_without_warm_start(tmp_path, monkeypatch) -> None:
    module = _module()

    class ProbeDone(Exception):
        pass

    class TinyModel(nn.Module):
        def forward(self, images):  # type: ignore[no-untyped-def]
            assert not self.training
            raise ProbeDone

    rows = pd.DataFrame({"split": ["train", "validation", "test"]})
    monkeypatch.setattr(module, "BinaryResNet50", TinyModel)
    monkeypatch.setattr(module, "preferred_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(module, "_transforms", lambda: (None, None))
    monkeypatch.setattr(module, "_load_rows", lambda *args, **kwargs: (rows, {}))
    args = SimpleNamespace(
        output=tmp_path,
        run_id="probe-eval",
        dataset=tmp_path,
        labels=tmp_path,
        candidates=tmp_path,
        heat_basis="water",
        limit_per_split=0,
        camera_group="all",
        init_checkpoint=None,
    )

    with pytest.raises(ProbeDone):
        module.run(args)


def test_smoke_limit_keeps_all_target_and_near_regret_strata() -> None:
    module = _module()
    rows = pd.DataFrame(
        [
            {"split": split, "target": target, "relative_regret": regret, "row": copy}
            for split in ("train", "validation", "test")
            for target in (0, 1)
            for regret in (0.005, 0.02)
            for copy in (0, 1)
        ]
    )

    limited = module._limit_rows(rows, limit_per_split=2)

    strata = limited.assign(
        near_1pct=limited["relative_regret"].le(0.01)
    ).groupby("split")[["target", "near_1pct"]].apply(
        lambda group: set(group.itertuples(index=False, name=None))
    )
    assert strata.to_dict() == {
        split: {(0, False), (0, True), (1, False), (1, True)}
        for split in ("train", "validation", "test")
    }


def test_training_camera_groups_select_exact_roles() -> None:
    module = _module()
    from frost_analysis.training.evaluation import CAMERA_GROUPS

    assert module.CAMERA_GROUPS is CAMERA_GROUPS
    rows = pd.DataFrame(
        {"camera_role": ["front", "left", "left_close", "top", "top_close", "extreme"]}
    )
    expected = {
        "front": {"front"},
        "left": {"left"},
        "left_close": {"left_close"},
        "top": {"top"},
        "top_close": {"top_close"},
        "extreme": {"extreme"},
        "left_pair": {"left", "left_close"},
        "top_pair": {"top", "top_close"},
        "all": {"front", "left", "left_close", "top", "top_close", "extreme"},
    }

    actual = {
        group: set(module._select_camera_group(rows, group)["camera_role"])
        for group in expected
    }

    assert actual == expected


def test_explicit_manifest_avoids_unrequested_cartesian_combinations(tmp_path) -> None:
    path = tmp_path / "plan.csv"
    pd.DataFrame(
        [
            {
                "camera_group": "front",
                "regret_threshold": 0.01,
                "representation": "dinov2",
                "model": "logistic",
                "modality": "rgb_time",
            },
            {
                "camera_group": "all",
                "regret_threshold": 0.01,
                "representation": "convnext_tiny",
                "model": "rbf_svm",
                "modality": "rgb",
            },
        ]
    ).to_csv(path, index=False)

    evaluate_path = Path("scripts/training/evaluate_rgb_feature_shards.py")
    spec = importlib.util.spec_from_file_location("rgb_feature_evaluation", evaluate_path)
    assert spec and spec.loader
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)

    plan = evaluate.read_experiment_manifest(path)

    assert len(plan) == 2
    assert plan.iloc[0]["modality"] == "rgb_time"


def test_three_class_manifest_has_51_actual_progress_combinations() -> None:
    evaluate_path = Path("scripts/training/evaluate_rgb_feature_shards.py")
    spec = importlib.util.spec_from_file_location("rgb_feature_progress", evaluate_path)
    assert spec and spec.loader
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    plan = evaluate.read_experiment_manifest(
        Path("configs/rgb_experiment_manifest.csv"), task="three"
    )

    stages = evaluate.combination_stages(plan)

    assert len(plan) == 55
    assert len(stages) == 51
    assert "MATRIX/PRIMARY/VIEW" in stages.values()


def test_formal_three_class_run_requires_14_unique_experiments() -> None:
    evaluate_path = Path("scripts/training/evaluate_rgb_feature_shards.py")
    spec = importlib.util.spec_from_file_location("rgb_feature_shape", evaluate_path)
    assert spec and spec.loader
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)

    with pytest.raises(SystemExit, match="14 unique experiments"):
        evaluate.validate_formal_run_shape(
            task="three",
            has_manifest=True,
            combination_count=51,
            experiments=[f"experiment_{index}" for index in range(13)],
        )


def test_resume_rejects_changed_combination_with_same_count(tmp_path) -> None:
    evaluate_path = Path("scripts/training/evaluate_rgb_feature_shards.py")
    spec = importlib.util.spec_from_file_location("rgb_combination_config", evaluate_path)
    assert spec and spec.loader
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    from frost_analysis.training.run import RunStore

    original = [("all", 0.01, "dinov3", "logistic", "rgb")]
    changed = [("all", 0.01, "dinov3", "rbf_svm", "rgb")]
    RunStore(
        tmp_path,
        "same-run",
        {"combinations": evaluate.normalized_combinations(original)},
    )

    with pytest.raises(ValueError, match="configuration"):
        RunStore(
            tmp_path,
            "same-run",
            {"combinations": evaluate.normalized_combinations(changed)},
        )


def test_staged_manifest_locks_binary_and_three_class_experiments() -> None:
    path = Path("configs/rgb_experiment_manifest.csv")

    manifest = pd.read_csv(path)
    columns = [
        "stage",
        "task",
        "camera_group",
        "regret_threshold",
        "representation",
        "model",
        "modality",
    ]
    rows = set(manifest[columns].itertuples(index=False, name=None))

    binary_a = {
        ("A", "binary", camera, 0.01, representation, model, "rgb")
        for camera in ("front", "all")
        for representation in (
            "handcrafted",
            "dinov2",
            "efficientnet",
            "mobilenet_v3_small",
            "repvit_m0_9",
            "convnext_tiny",
        )
        for model in ("logistic", "rbf_svm")
    } | {
        ("A", "binary", camera, 0.01, "dinov2", model, modality)
        for camera in ("front", "all")
        for model in ("logistic", "rbf_svm")
        for modality in ("time", "rgb_time")
    }
    binary_b = {
        ("B", "binary", camera, 0.01, representation, model, "rgb")
        for camera in (
            "top",
            "top_close",
            "left",
            "left_close",
            "front",
            "extreme",
            "top_pair",
            "left_pair",
            "all",
        )
        for representation in ("handcrafted", "mobilenet_v3_small")
        for model in (
            "logistic",
            "random_forest",
            "rbf_svm",
            "hist_gradient_boosting",
            "mlp",
        )
    }
    binary_c = {
        ("C", "binary", "front", threshold, "handcrafted", "logistic", "rgb")
        for threshold in (0.01, 0.02, 0.05, 0.10)
    }
    binary = manifest.loc[manifest["task"].eq("binary")]
    assert len(binary) == 126
    assert set(binary[columns].itertuples(index=False, name=None)) == (
        binary_a | binary_b | binary_c
    )

    representations = {
        "handcrafted",
        "dinov2",
        "efficientnet",
        "mobilenet_v3_small",
        "repvit_m0_9",
        "convnext_tiny",
        "dinov3",
        "siglip2",
    }
    models = {
        "logistic",
        "random_forest",
        "rbf_svm",
        "hist_gradient_boosting",
        "mlp",
        "window_logistic",
    }
    matrix = {
        ("MATRIX", "three", "all", 0.01, representation, model, "rgb")
        for representation in representations
        for model in models
    }
    primary = {
        ("PRIMARY", "three", "all", 0.01, representation, "logistic", modality)
        for representation, modality in (
            ("handcrafted", "time"),
            ("handcrafted", "rgb"),
            ("dinov2", "rgb"),
            ("dinov3", "rgb"),
            ("dinov3", "rgb_time"),
        )
    }
    view = {
        ("VIEW", "three", camera, 0.01, "dinov3", "logistic", "rgb") for camera in ("front", "all")
    }
    three = manifest.loc[manifest["task"].eq("three")]
    assert three["regret_threshold"].eq(0.01).all()
    assert rows - (binary_a | binary_b | binary_c) == matrix | primary | view
    assert len(matrix) == 48
    assert len(primary) == 5
    assert len(view) == 2
    for stage, expected in (("MATRIX", matrix), ("PRIMARY", primary), ("VIEW", view)):
        staged = three.loc[three["stage"].eq(stage), columns]
        assert not staged.duplicated().any()
        assert set(staged.itertuples(index=False, name=None)) == expected


def test_holdout_audit_rejects_cycle_assigned_to_multiple_experiments() -> None:
    evaluate_path = Path("scripts/training/evaluate_rgb_feature_shards.py")
    spec = importlib.util.spec_from_file_location("rgb_feature_evaluation_audit", evaluate_path)
    assert spec and spec.loader
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    frame = pd.DataFrame(
        {
            "experiment_id": ["a", "b"],
            "cycle_name": ["same", "same"],
            "camera_role": ["front", "front"],
            "image_time": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        }
    )

    with pytest.raises(ValueError, match="multiple experiments"):
        evaluate.audit_holdout_cohort(frame)


def test_rgb_time_modality_excludes_future_cycle_end_progress() -> None:
    evaluate_path = Path("scripts/training/evaluate_rgb_feature_shards.py")
    spec = importlib.util.spec_from_file_location("rgb_feature_evaluation_time", evaluate_path)
    assert spec and spec.loader
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    frame = pd.DataFrame(
        {
            "feature_001": [2.0],
            "dinov2_000": [1.0],
            "time_elapsed_minutes": [12.0],
            "time_candidate_progress": [0.75],
        }
    )

    modalities = evaluate.build_modality_frames(frame, "dinov2")

    assert "dinov2_time_elapsed_minutes" in modalities["rgb_time"]
    assert modalities["time"]["feature_000"].tolist() == [12.0]
    assert not {"feature_001", "dinov2_000"} & set(modalities["time"])
    assert not any(
        "progress" in column for column in modalities["rgb_time"] if column.startswith("dinov2_")
    )
    assert "rgb_state" not in modalities
    assert "rgb_all_sensor" not in modalities


@pytest.mark.parametrize("wandb_project", [None, "rgb-test"], ids=["offline", "wandb"])
def test_feature_evaluator_runs_offline_and_with_wandb(  # noqa: C901
    tmp_path, monkeypatch, wandb_project, capsys
) -> None:
    evaluate_path = Path("scripts/training/evaluate_rgb_feature_shards.py")
    spec = importlib.util.spec_from_file_location("rgb_feature_evaluation_no_hash", evaluate_path)
    assert spec and spec.loader
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    shards = tmp_path / "shards"
    shards.mkdir()
    rows = []
    candidates = []
    for index, experiment in enumerate(("a", "b", "c")):
        cycle = f"cycle_{experiment}"
        started = pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)
        candidates.append(
            {
                "cycle_name": cycle,
                "candidate_time": started + pd.Timedelta(minutes=30),
                "heating_hours": 0.5,
            }
        )
        for target, state, value in ((0, "pre_optimal", -1.0), (1, "post_optimal", 1.0)):
            rows.append(
                {
                    "experiment_id": experiment,
                    "cycle_name": cycle,
                    "camera_role": "front",
                    "image_time": started + pd.Timedelta(minutes=10 * target),
                    "cost_state": state,
                    "relative_regret": 0.1,
                    "feature_000": value,
                }
            )
    features = pd.DataFrame(rows)
    features.to_parquet(shards / "features.parquet")
    candidate_path = tmp_path / "candidates.parquet"
    pd.DataFrame(candidates).to_parquet(candidate_path)
    label_balance = tmp_path / "label_balance.csv"
    pd.DataFrame(
        {
            "camera_group": ["front"] * 3,
            "regret_threshold": [0.01] * 3,
            "cost_state": ["pre_optimal", "near_optimal", "post_optimal"],
            "image_count": [3, 0, 3],
        }
    ).to_csv(label_balance, index=False)
    output = tmp_path / "output"
    argv = [
        "evaluate_rgb_feature_shards.py",
        "--shards",
        str(shards),
        "--candidates",
        str(candidate_path),
        "--label-balance",
        str(label_balance),
        "--camera-groups",
        "front",
        "--regret-thresholds",
        "0.01",
        "--models",
        "logistic",
        "--modalities",
        "rgb",
        "--jobs",
        "1",
        "--run-id",
        "contract-run",
        "--output",
        str(output),
    ]
    if wandb_project:
        argv.extend(["--wandb-project", wandb_project, "--wandb-run-name", "test-run"])
    monkeypatch.setattr("sys.argv", argv)

    class FakeRun:
        def __init__(self) -> None:
            self.logs = []
            self.finished = False

        def log(self, values):  # type: ignore[no-untyped-def]
            self.logs.append(values)

        def finish(self) -> None:
            self.finished = True

        def define_metric(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

    class FakeWandb:
        def __init__(self) -> None:
            self.calls = []
            self.run = FakeRun()

        def init(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(kwargs)
            return self.run

    fake_wandb = FakeWandb()

    evaluate.main(wandb_module=fake_wandb)

    run_output = output / "runs" / "contract-run"
    predictions = pd.read_parquet(run_output / "predictions.parquet")
    assert len(predictions) == len(rows)
    progress = capsys.readouterr().out
    assert "3/3, OK=3, INVALID=0, FAILED=0" in progress
    if wandb_project:
        assert fake_wandb.calls[0]["id"] == "contract-run"
        assert fake_wandb.calls[0]["resume"] == "allow"
        assert [row["task_step"] for row in fake_wandb.run.logs] == [1, 2, 3]
        assert fake_wandb.run.finished
    else:
        assert fake_wandb.calls == []
        assert not fake_wandb.run.finished


def test_wandb_exceptions_are_noncritical(capsys) -> None:  # type: ignore[no-untyped-def]
    evaluate_path = Path("scripts/training/evaluate_rgb_feature_shards.py")
    spec = importlib.util.spec_from_file_location(
        "rgb_feature_evaluation_wandb_error", evaluate_path
    )
    assert spec and spec.loader
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)

    class BrokenRun:
        def log(self, values) -> None:  # type: ignore[no-untyped-def]
            raise RuntimeError("offline")

    assert evaluate._safe_wandb_method(BrokenRun(), "log", {"value": 1}) is None
    assert "[W&B warning] log: RuntimeError: offline" in capsys.readouterr().out


def test_three_class_evaluator_does_not_read_binary_label_balance(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    path = Path("scripts/training/evaluate_rgb_feature_shards.py")
    spec = importlib.util.spec_from_file_location("rgb_feature_evaluation_three", path)
    assert spec and spec.loader
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    shards = tmp_path / "shards"
    shards.mkdir()
    rows = []
    candidates = []
    for index, experiment in enumerate(("a", "b", "c")):
        cycle = f"cycle_{experiment}"
        started = pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)
        for minute in (0, 20):
            candidates.append(
                {
                    "cycle_name": cycle,
                    "candidate_time": started + pd.Timedelta(minutes=minute),
                    "heating_hours": 0.0,
                }
            )
        for target, state in enumerate(("pre_optimal", "near_optimal", "post_optimal")):
            rows.append(
                {
                    "experiment_id": experiment,
                    "cycle_name": cycle,
                    "camera_role": "front",
                    "image_time": started + pd.Timedelta(minutes=10 * target),
                    "cost_state": state,
                    "relative_regret": 0.1,
                    "feature_000": float(target),
                }
            )
    features = pd.DataFrame(rows)
    for cycle, cycle_rows in features.groupby("cycle_name"):
        cycle_rows.to_parquet(shards / f"{cycle}.parquet")
    candidate_path = tmp_path / "candidates.parquet"
    pd.DataFrame(candidates).to_parquet(candidate_path)
    labels = tmp_path / "labels.parquet"
    features[["cycle_name", "camera_role", "image_time", "relative_regret"]].to_parquet(labels)
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_rgb_feature_shards.py",
            "--shards",
            str(shards),
            "--candidates",
            str(candidate_path),
            "--label-balance",
            str(tmp_path / "missing.csv"),
            "--labels",
            str(labels),
            "--camera-groups",
            "front",
            "--task",
            "three",
            "--regret-thresholds",
            "0.01",
            "--models",
            "logistic",
            "--modalities",
            "rgb",
            "--run-id",
            "three-run",
            "--output",
            str(output),
        ],
    )

    evaluate.main()

    predictions = pd.read_parquet(output / "runs" / "three-run" / "predictions.parquet")
    assert len(predictions) == len(rows)


def test_three_class_evaluator_selects_only_label_target_shards(tmp_path) -> None:
    path = Path("scripts/training/evaluate_rgb_feature_shards.py")
    spec = importlib.util.spec_from_file_location("rgb_feature_evaluation_targets", path)
    assert spec and spec.loader
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    shards = tmp_path / "shards"
    shards.mkdir()
    pd.DataFrame({"value": [1]}).to_parquet(shards / "cycle_target.parquet")
    pd.DataFrame({"value": [2]}).to_parquet(shards / "cycle_history.parquet")
    labels = tmp_path / "labels.parquet"
    pd.DataFrame(
        {
            "cycle_name": ["cycle_target", "cycle_history"],
            "relative_regret": [0.1, float("nan")],
        }
    ).to_parquet(labels)

    selected = evaluate.target_shard_paths(shards, labels)

    assert selected == [shards / "cycle_target.parquet"]


def test_summary_scores_are_na_when_no_fold_is_evaluable() -> None:
    evaluate_path = Path("scripts/training/evaluate_rgb_feature_shards.py")
    spec = importlib.util.spec_from_file_location("rgb_feature_evaluation_scores", evaluate_path)
    assert spec and spec.loader
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    frame = pd.DataFrame(
        {
            "target": [0, 1],
            "predicted_target": [pd.NA, pd.NA],
            "decision_score": [float("nan"), float("nan")],
            "fold_evaluable": [False, False],
        }
    )

    scores = evaluate.score_rows(frame)

    assert all(math.isnan(value) for value in scores.values())


def test_summary_scores_sort_multiclass_columns_by_numeric_class() -> None:
    evaluate_path = Path("scripts/training/evaluate_rgb_feature_shards.py")
    spec = importlib.util.spec_from_file_location(
        "rgb_feature_evaluation_numeric_classes", evaluate_path
    )
    assert spec and spec.loader
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    frame = pd.DataFrame(
        {
            "target": [2, 2, 10, 10, 11, 11],
            "predicted_target": [2, 2, 10, 10, 11, 11],
            "decision_score_2": [0.9, 0.9, 0.05, 0.05, 0.05, 0.05],
            "decision_score_10": [0.05, 0.05, 0.9, 0.9, 0.05, 0.05],
            "decision_score_11": [0.05, 0.05, 0.05, 0.05, 0.9, 0.9],
        }
    )

    scores = evaluate.score_rows(frame)

    assert scores["auroc"] == 1.0
    assert scores["accuracy"] == 1.0
    assert scores["macro_f1"] == 1.0
    assert all(math.isnan(scores[name]) for name in ("positive_f1", "precision", "recall"))
