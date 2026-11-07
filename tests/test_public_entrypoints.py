from pathlib import Path


def test_public_entrypoints_name_their_scientific_tasks() -> None:
    expected = {
        "main_defrosting_model.py",
        "validate_dataset.py",
        "defrosting_electricity_model.py",
        "optimal_defrost_time.py",
        "calculate_rb_reference.py",
    }
    ambiguous = {
        "main_data.py",
        "main_cost.py",
        "main_labels.py",
        "main_train.py",
        "main_evaluate.py",
    }

    assert all(Path(name).is_file() for name in expected)
    assert all(not Path(name).exists() for name in ambiguous)


def test_training_entrypoint_preserves_experiment_folds_and_saved_recipe(tmp_path, monkeypatch):
    """Exercise CLI-to-output wiring without fitting models or rendering figures."""
    import json

    import pandas as pd

    from dataset_tools import DatasetLoader
    from image_models import checkpoints, experiments, optimal_stopping

    rows = pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "cycle_b", "cycle_c"],
            "experiment_id": ["a", "b", "c"],
            "evaluation_eligible": True,
            "training_eligible": True,
            "input_valid": True,
        }
    )

    class PublishedDataset(DatasetLoader):
        def __init__(self, root):
            self._registry = {"measurement_processing": "fixture"}

        def camera_training_settings(self, camera_role):
            return {"camera_role": camera_role}

    monkeypatch.setattr(experiments, "DatasetLoader", PublishedDataset)
    monkeypatch.setattr(optimal_stopping, "prepare_ablation_inputs", lambda args: (rows, rows))
    fitted = []

    def fit_fold(args, observations, reference, test, validation, seed):
        split = DatasetLoader.split_training_data(observations, test, validation)
        groups = [
            set(split.loc[split.split.eq(role), "experiment_id"])
            for role in ("train", "validation", "test")
        ]
        assert all(groups) and not any(groups[i] & groups[j] for i in range(3) for j in range(i))
        fitted.append((test, seed))

    monkeypatch.setattr(optimal_stopping, "fit_fold", fit_fold)
    monkeypatch.setattr(experiments, "checkpoint_files", lambda run: fitted)
    summaries = []
    monkeypatch.setattr(
        "image_models.comparisons.summarize",
        lambda args, versions, contrasts, previous: summaries.append((versions, contrasts)),
    )
    monkeypatch.setattr(checkpoints, "organize", lambda *args, **kwargs: None)
    monkeypatch.setattr("image_models.comparisons.write_ablation_tables", lambda *args: None)

    output = tmp_path / "training"
    optimal_stopping.main(
        [
            "--dataset",
            str(tmp_path),
            "--output",
            str(output),
            "--purpose",
            "entrypoint regression",
            "--recipe",
            "single",
            "--architecture",
            "logistic",
            "--seeds",
            "42",
            "--n-jobs",
            "1",
        ]
    )

    settings = json.loads((output / "settings.json").read_text())
    assert fitted == [("a", 42), ("b", 42), ("c", 42)]
    assert settings["recipes"] == [["logistic", "set_stop"]]
    assert settings["cycle_names"] == rows.cycle_name.tolist()
    assert settings["data_definition"] == {"camera_role": "front"}
    assert pd.read_csv(output / "folds.csv").test_experiment.tolist() == ["a", "b", "c"]
    assert summaries == [(["logistic_set_stop"], [])]


def test_main_defaults_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    import defrosting_electricity_model as ridge
    import main_defrosting_model as visual
    import optimal_defrost_time as cop
    from dataset_tools import manage_dataset
    from output_layout import OUTPUT, PROJECT, REPRODUCTION

    monkeypatch.chdir(tmp_path)
    assert ridge.build_parser().parse_args([]).dataset == PROJECT / "dataset"
    assert manage_dataset.build_parser().parse_args(["smooth"]).dataset == PROJECT / "dataset"
    assert visual.get_args(["train", "--output", "temp/test/demo"]).dataset == PROJECT / "dataset"
    defaults = cop.get_args([])
    assert defaults.dataset == PROJECT / "dataset"
    assert defaults.model_file == cop.DEFAULT_MODEL_FILE
    assert defaults.output == OUTPUT / REPRODUCTION / "03_COP候选/effective_cop"
    selected = cop.get_args(["--dataset", "packaged", "--output", "results"])
    assert selected.dataset == Path("packaged")
    assert selected.output == Path("results")
    assert selected.model_file.is_relative_to(Path("packaged"))


def test_zero_argument_acceptance_uses_frozen_recipe_and_new_output(tmp_path, monkeypatch):
    import main_defrosting_model as visual
    from output_layout import PROJECT

    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(visual, "evaluate_external", lambda *a, **kw: calls.append((a, kw)))
    visual.main([])
    visual.main([])
    for args, options in calls:
        assert args[1] == PROJECT / "dataset"
        assert args[2] == ["exp_20260920"]
        assert args[3].parent == PROJECT / "temp/test"
        assert options["checkpoint"].name == "正面_部署模型.pkl"
        assert options["camera_role"] is None
    assert calls[0][0][3] != calls[1][0][3]
    assert not calls[0][0][3].exists()


def test_plot_flags_do_not_choose_training_labels():
    from image_models.cycle_selection import build_parser

    parser = build_parser()
    for flags in ([], ["--figures"], ["--dataset-figures"]):
        args = parser.parse_args(["--decisions", "candidates.csv", *flags])
        assert args.near_set_rule == "exact"
    assert (
        parser.parse_args(
            ["--decisions", "candidates.csv", "--dataset-figures", "--near-set-rule", "connected"]
        ).near_set_rule
        == "connected"
    )


def test_loeo_ablation_rejects_fixed_train_test_before_writing(tmp_path, monkeypatch):
    import json
    import sys

    import pytest

    from image_models import ablation

    baseline = tmp_path / "fixed_model"
    (baseline / "training").mkdir(parents=True)
    (baseline / "training/settings.json").write_text(
        json.dumps({"loss": "cdf_ramp", "model": {}, "seed": 42})
    )
    output = tmp_path / "analysis"
    monkeypatch.setattr(
        sys, "argv", ["ablation", "--baseline", str(baseline), "--output", str(output)]
    )
    with pytest.raises(SystemExit) as error:
        ablation.main()
    assert error.value.code == 2
    assert not output.exists()
