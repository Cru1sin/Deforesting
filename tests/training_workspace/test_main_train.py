from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import main_train


def test_parser_defaults_to_one_primary_setting() -> None:
    args = main_train.build_parser().parse_args([])

    assert args.dataset == Path("dataset")
    assert args.labels == Path("output/labels/v1/image_cost_labels.parquet")
    assert args.output == Path("output/models/current")
    assert args.task == "binary"
    assert args.state_column == "cost_state_01pct"
    assert args.representations == ["handcrafted"]
    assert args.heads == ["rbf_svm"]
    assert args.cameras == ["front"]
    assert args.modalities == ["rgb"]
    assert args.jobs == 6
    assert args.maximum_per_group == 48
    assert main_train.expand_settings(args) == [
        main_train.Setting("handcrafted", "rbf_svm", "front", "rgb")
    ]


def test_explicit_multiple_values_expand_with_itertools_product() -> None:
    args = main_train.build_parser().parse_args(
        [
            "--representations",
            "handcrafted",
            "--heads",
            "logistic",
            "random_forest",
            "--cameras",
            "front",
            "top",
            "--modalities",
            "rgb",
            "time",
        ]
    )

    settings = main_train.expand_settings(args)

    assert len(settings) == 8
    assert main_train.Setting("handcrafted", "logistic", "front", "rgb") in settings
    assert main_train.Setting("handcrafted", "random_forest", "top", "time") in settings


@pytest.mark.parametrize(
    "arguments, message",
    [
        (
            ["--representations", "resnet50_finetune", "--heads", "logistic", "--jobs", "1"],
            "paper_mlp",
        ),
        (
            ["--representations", "resnet50_finetune", "--heads", "paper_mlp"],
            "jobs=1",
        ),
        (
            [
                "--representations",
                "resnet50_finetune",
                "--heads",
                "paper_mlp",
                "--modalities",
                "time",
                "--jobs",
                "1",
            ],
            "rgb",
        ),
        (
            ["--representations", "handcrafted", "--heads", "paper_mlp"],
            "only valid",
        ),
    ],
)
def test_invalid_setting_combinations_are_rejected(
    arguments: list[str], message: str
) -> None:
    args = main_train.build_parser().parse_args(arguments)

    with pytest.raises(ValueError, match=message):
        main_train.expand_settings(args)


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment_id": ["a", "a", "a", "b", "b", "b", "ignored"],
            "cycle_name": ["c1", "c1", "c1", "c2", "c2", "c2", "c3"],
            "camera_role": ["front", "front", "front", "front", "front", "front", "top"],
            "file_name": [f"{index}.png" for index in range(7)],
            "image_path": [f"images/c/{index}.png" for index in range(7)],
            "image_time": pd.date_range("2026-01-01", periods=7, freq="min"),
            "stable_heating_start": [pd.Timestamp("2025-12-31 23:50:00")] * 7,
            "relative_regret": [0.2, 0.0, 0.3, 0.3, 0.0, 0.2, float("nan")],
            "cost_state_01pct": [
                "pre_optimal",
                "near_optimal",
                "post_optimal",
                "pre_optimal",
                "near_optimal",
                "post_optimal",
                "pre_optimal",
            ],
        }
    )


def test_label_rows_apply_binary_and_three_class_contracts() -> None:
    labels = _labels()

    binary = main_train.label_rows(
        labels, task="binary", state_column="cost_state_01pct", camera="front"
    )
    three = main_train.label_rows(
        labels, task="three", state_column="cost_state_01pct", camera="front"
    )

    assert binary["target"].tolist() == [0, 1, 0, 1]
    assert "near_optimal" not in binary["cost_state_01pct"].tolist()
    assert three["target"].tolist() == [0, 1, 2, 0, 1, 2]
    assert binary["relative_regret"].notna().all()
    assert binary["camera_role"].eq("front").all()


def test_label_rows_requires_canonical_stable_heating_start() -> None:
    with pytest.raises(ValueError, match="stable_heating_start"):
        main_train.label_rows(
            _labels().drop(columns="stable_heating_start"),
            task="binary",
            state_column="cost_state_01pct",
            camera="front",
        )


def test_dry_run_reads_labels_and_prints_task_total_without_touching_images(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    labels_path = tmp_path / "labels.parquet"
    _labels().to_parquet(labels_path, index=False)
    args = main_train.build_parser().parse_args(
        [
            "--labels",
            str(labels_path),
            "--dataset",
            str(tmp_path / "dataset-does-not-exist"),
            "--output",
            str(tmp_path / "models"),
            "--heads",
            "logistic",
            "random_forest",
            "--dry-run",
        ]
    )

    assert main_train.run(args, command=["main_train.py", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "Settings: 2" in output
    assert "Held-out experiments: 2" in output
    assert "Total tasks: 4" in output
    assert "handcrafted / logistic / front / rgb" in output
    assert "handcrafted / random_forest / front / rgb" in output
    assert not (tmp_path / "models").exists()


def _training_labels(dataset: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment_index, experiment in enumerate(("a", "b", "c")):
        cycle = f"cycle_{experiment}"
        for target, state in ((0, "pre_optimal"), (1, "post_optimal")):
            relative = Path("images") / cycle / "front" / f"{target}.png"
            image = dataset / relative
            image.parent.mkdir(parents=True, exist_ok=True)
            color = 30 + 180 * target + experiment_index
            Image.new("RGB", (20, 12), (color, color, color)).save(image)
            rows.append(
                {
                    "experiment_id": experiment,
                    "cycle_name": cycle,
                    "camera_role": "front",
                    "file_name": image.name,
                    "image_path": str(relative),
                    "image_time": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(minutes=target),
                    "stable_heating_start": pd.Timestamp("2025-12-31 23:50:00"),
                    "relative_regret": 0.2,
                    "cost_state_01pct": state,
                }
            )
    return pd.DataFrame(rows)


def test_nonempty_output_is_rejected_before_feature_extraction(tmp_path: Path) -> None:
    labels = tmp_path / "labels.parquet"
    _labels().to_parquet(labels, index=False)
    output = tmp_path / "models"
    output.mkdir()
    (output / "keep.txt").write_text("owned by user")
    args = main_train.build_parser().parse_args(
        ["--labels", str(labels), "--output", str(output)]
    )

    with pytest.raises(FileExistsError, match="not empty"):
        main_train.run(args)


def test_parallel_handcrafted_run_writes_complete_artifacts_from_main(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "dataset"
    labels = tmp_path / "labels.parquet"
    _training_labels(dataset).to_parquet(labels, index=False)
    output = tmp_path / "trained"
    args = main_train.build_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--heads",
            "logistic",
            "random_forest",
            "--jobs",
            "2",
        ]
    )

    assert main_train.run(args, command=["main_train.py", "--jobs", "2"]) == 0

    assert (output / "command.txt").read_text().strip() == "main_train.py --jobs 2"
    saved_args = json.loads((output / "args.json").read_text())
    assert saved_args["jobs"] == 2
    assert len(pd.read_csv(output / "settings.csv")) == 2
    progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    metrics = pd.read_csv(output / "metrics.csv")
    predictions = pd.read_parquet(output / "predictions.parquet")
    assert len(progress) == 6
    assert {row["status"] for row in progress} == {"ok"}
    assert len(metrics) == 6
    assert len(predictions) == 12
    assert set(predictions["held_out_experiment"]) == {"a", "b", "c"}
    assert set(predictions["head"]) == {"logistic", "random_forest"}
    assert (tmp_path / "output/models/_cache/handcrafted/front/features.parquet").is_file()


def test_all_invalid_folds_still_write_the_prediction_schema(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    labels = _labels().loc[[0, 5]].copy()
    labels_path = tmp_path / "labels.parquet"
    labels.to_parquet(labels_path, index=False)
    output = tmp_path / "invalid"
    args = main_train.build_parser().parse_args(
        [
            "--labels",
            str(labels_path),
            "--output",
            str(output),
            "--heads",
            "logistic",
            "--modalities",
            "time",
            "--jobs",
            "1",
        ]
    )

    assert main_train.run(args) == 0

    assert set(pd.read_parquet(output / "predictions.parquet").columns) == {
        "experiment_id",
        "cycle_name",
        "camera",
        "camera_role",
        "image_time",
        "held_out_experiment",
        "representation",
        "head",
        "modality",
        "target",
        "prediction",
        "decision_score",
    }


def test_each_setting_uses_only_its_own_experiments_for_folds(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    rows = []
    for camera, experiments in (("front", ("a", "b")), ("top", ("c", "d"))):
        for experiment in experiments:
            for target, state in ((0, "pre_optimal"), (1, "post_optimal")):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "cycle_name": f"cycle_{experiment}",
                        "camera_role": camera,
                        "file_name": f"{target}.png",
                        "image_path": f"{experiment}/{target}.png",
                        "image_time": pd.Timestamp("2026-01-01")
                        + pd.Timedelta(minutes=target),
                        "stable_heating_start": pd.Timestamp("2026-01-01"),
                        "relative_regret": 0.2,
                        "cost_state_01pct": state,
                    }
                )
    labels = tmp_path / "labels.parquet"
    pd.DataFrame(rows).to_parquet(labels, index=False)
    output = tmp_path / "run"
    args = main_train.build_parser().parse_args(
        [
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--heads",
            "logistic",
            "--cameras",
            "front",
            "top",
            "--modalities",
            "time",
            "--jobs",
            "1",
        ]
    )

    assert main_train.run(args) == 0

    metrics = pd.read_csv(output / "metrics.csv")
    assert len(metrics) == 4
    assert set(zip(metrics["camera"], metrics["held_out_experiment"], strict=True)) == {
        ("front", "a"),
        ("front", "b"),
        ("top", "c"),
        ("top", "d"),
    }
    assert "Total tasks: 4" in capsys.readouterr().out
