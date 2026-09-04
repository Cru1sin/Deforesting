from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import train_image_models
from image_models.sensor_features import (
    CURRENT_SENSORS,
    SLOPE_SENSORS,
    build_past_only_sensor_features,
)


def test_parser_defaults_to_one_primary_setting() -> None:
    args = train_image_models.build_parser().parse_args([])

    assert args.dataset == Path("dataset")
    assert args.labels == Path("output/image_labels/v1_label_reference/image_timing_labels.parquet")
    assert args.output == Path("output/image_models/current")
    assert args.task == "binary"
    assert args.label_column == "binary_target_01pct"
    assert args.image_features == ["color_gradient"]
    assert args.classifiers == ["rbf_svm"]
    assert args.camera_groups == ["front"]
    assert args.input_features == ["image_only"]
    assert args.workers == 6
    assert args.max_images_per_cycle_label == 48
    assert args.seed == 0
    assert args.wandb_project is None
    assert args.wandb_run_name is None
    assert train_image_models.build_training_settings(args) == [
        train_image_models.Setting("color_gradient", "rbf_svm", "front", "image_only")
    ]


def test_explicit_multiple_values_expand_with_itertools_product() -> None:
    args = train_image_models.build_parser().parse_args(
        [
            "--image-features",
            "color_gradient",
            "--classifiers",
            "logistic_regression",
            "random_forest",
            "--camera-groups",
            "front",
            "top",
            "--input-features",
            "image_only",
            "elapsed_time_only",
        ]
    )

    settings = train_image_models.build_training_settings(args)

    assert len(settings) == 8
    assert (
        train_image_models.Setting("color_gradient", "logistic_regression", "front", "image_only")
        in settings
    )
    assert (
        train_image_models.Setting("color_gradient", "random_forest", "top", "elapsed_time_only")
        in settings
    )


@pytest.mark.parametrize(
    "arguments, message",
    [
        (
            [
                "--image-features",
                "resnet50_end_to_end",
                "--classifiers",
                "logistic_regression",
                "--workers",
                "1",
            ],
            "resnet_mlp",
        ),
        (
            ["--image-features", "resnet50_end_to_end", "--classifiers", "resnet_mlp"],
            "workers=1",
        ),
        (
            [
                "--image-features",
                "resnet50_end_to_end",
                "--classifiers",
                "resnet_mlp",
                "--input-features",
                "elapsed_time_only",
                "--workers",
                "1",
            ],
            "image_only",
        ),
        (
            ["--image-features", "color_gradient", "--classifiers", "resnet_mlp"],
            "only valid",
        ),
    ],
)
def test_invalid_setting_combinations_are_rejected(arguments: list[str], message: str) -> None:
    args = train_image_models.build_parser().parse_args(arguments)

    with pytest.raises(ValueError, match=message):
        train_image_models.build_training_settings(args)


def _labels() -> pd.DataFrame:
    labels = pd.DataFrame(
        {
            "experiment_id": ["a", "a", "a", "b", "b", "b", "ignored"],
            "cycle_name": ["c1", "c1", "c1", "c2", "c2", "c2", "c3"],
            "camera_role": ["front", "front", "front", "front", "front", "front", "top"],
            "file_name": [f"{index}.png" for index in range(7)],
            "image_path": [f"images/c/{index}.png" for index in range(7)],
            "image_time": pd.date_range("2026-01-01", periods=7, freq="min"),
            "stable_heating_start": [pd.Timestamp("2025-12-31 23:50:00")] * 7,
            "relative_regret": [0.2, 0.0, 0.3, 0.3, 0.0, 0.2, float("nan")],
            "timing_state_01pct": [
                "before_reference",
                "near_reference",
                "after_reference",
                "before_reference",
                "near_reference",
                "after_reference",
                "before_reference",
            ],
        }
    )
    labels["binary_target_01pct"] = labels["timing_state_01pct"].mask(
        labels["timing_state_01pct"].eq("near_reference")
    )
    return labels


def test_label_rows_apply_binary_and_three_class_contracts() -> None:
    labels = _labels()

    binary = train_image_models.select_labeled_images(
        labels, task="binary", label_column="timing_state_01pct", camera="front"
    )
    three = train_image_models.select_labeled_images(
        labels, task="three", label_column="timing_state_01pct", camera="front"
    )

    assert binary["target"].tolist() == [0, 1, 0, 1]
    assert "near_reference" not in binary["timing_state_01pct"].tolist()
    assert three["target"].tolist() == [0, 1, 2, 0, 1, 2]
    assert binary["relative_regret"].notna().all()
    assert binary["camera_role"].eq("front").all()


def test_label_rows_requires_canonical_stable_heating_start() -> None:
    with pytest.raises(ValueError, match="stable_heating_start"):
        train_image_models.select_labeled_images(
            _labels().drop(columns="stable_heating_start"),
            task="binary",
            label_column="timing_state_01pct",
            camera="front",
        )


def test_policy_binary_rows_do_not_require_cost_regret() -> None:
    labels = _labels().drop(columns=["relative_regret", "timing_state_01pct"])
    labels["binary_target"] = [
        "before_reference",
        "before_reference",
        "after_reference",
        "before_reference",
        "after_reference",
        "after_reference",
        "before_reference",
    ]

    rows = train_image_models.select_labeled_images(
        labels, task="binary", label_column="binary_target", camera="front"
    )

    assert rows["target"].tolist() == [0, 0, 1, 0, 1, 1]


def test_sensor_slope_uses_only_the_value_five_minutes_in_the_past() -> None:
    frame = pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "cycle_a"],
            "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:05"]),
            "evaporating_pressure": [2.0, 1.5],
            "evaporating_pressure__imputed": [False, False],
        }
    )

    result = build_past_only_sensor_features(
        frame,
        current_sensors=("evaporating_pressure",),
        slope_sensors=("evaporating_pressure",),
    )

    assert (
        result["sensor_timestamp"].tolist()
        == pd.to_datetime(["2026-01-01 00:00:10", "2026-01-01 00:05:10"]).tolist()
    )
    assert pd.isna(result.loc[0, "evaporating_pressure__slope_5min"])
    assert result.loc[1, "evaporating_pressure__slope_5min"] == pytest.approx(-0.1)


def test_dry_run_reads_labels_and_prints_task_total_without_touching_images(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    labels_path = tmp_path / "image_labels.parquet"
    _labels().to_parquet(labels_path, index=False)
    args = train_image_models.build_parser().parse_args(
        [
            "--labels",
            str(labels_path),
            "--dataset",
            str(tmp_path / "dataset-does-not-exist"),
            "--output",
            str(tmp_path / "models"),
            "--classifiers",
            "logistic_regression",
            "random_forest",
            "--dry-run",
        ]
    )

    assert train_image_models.run(args, command=["train_image_models.py", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "Settings: 2" in output
    assert "Held-out experiments: 2" in output
    assert "Total tasks: 4" in output
    assert "color_gradient / logistic_regression / front / image_only" in output
    assert "color_gradient / random_forest / front / image_only" in output
    assert not (tmp_path / "models").exists()


def _training_labels(dataset: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment_index, experiment in enumerate(("a", "b", "c")):
        cycle = f"cycle_{experiment}"
        for target, state in ((0, "before_reference"), (1, "after_reference")):
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
                    "image_time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=target),
                    "stable_heating_start": pd.Timestamp("2025-12-31 23:50:00"),
                    "relative_regret": 0.2,
                    "timing_state_01pct": state,
                    "binary_target_01pct": state,
                }
            )
    return pd.DataFrame(rows)


def test_nonempty_output_is_rejected_before_feature_extraction(tmp_path: Path) -> None:
    labels = tmp_path / "image_labels.parquet"
    _labels().to_parquet(labels, index=False)
    output = tmp_path / "models"
    output.mkdir()
    (output / "keep.txt").write_text("owned by user")
    args = train_image_models.build_parser().parse_args(
        ["--labels", str(labels), "--output", str(output)]
    )

    with pytest.raises(FileExistsError, match="not empty"):
        train_image_models.run(args)


def test_parallel_color_gradient_run_writes_complete_artifacts_from_main(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "dataset"
    labels = tmp_path / "image_labels.parquet"
    _training_labels(dataset).to_parquet(labels, index=False)
    output = tmp_path / "trained"
    args = train_image_models.build_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--classifiers",
            "logistic_regression",
            "random_forest",
            "--workers",
            "2",
        ]
    )

    assert train_image_models.run(args, command=["train_image_models.py", "--workers", "2"]) == 0

    saved_args = json.loads((output / "run_settings.json").read_text())
    assert saved_args["command"] == "uv run python train_image_models.py --workers 2"
    assert saved_args["workers"] == 2
    assert len(pd.read_csv(output / "training_settings.csv")) == 2
    assert not (output / "progress.jsonl").exists()
    task_log = [json.loads(line) for line in (output / "fold_log.jsonl").read_text().splitlines()]
    metrics = pd.read_csv(output / "fold_metrics.csv")
    predictions = pd.read_parquet(output / "predictions.parquet")
    assert len(task_log) == 6
    assert {row["status"] for row in task_log} == {"ok"}
    assert len(metrics) == 6
    assert len(predictions) == 12
    assert set(predictions["held_out_experiment"]) == {"a", "b", "c"}
    assert set(predictions["classifier"]) == {"logistic_regression", "random_forest"}
    assert (tmp_path / "output/image_models/_cache/color_gradient/front/features.parquet").is_file()


def test_sensor_slope_setting_attaches_sensor_rows_from_the_public_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels_path = tmp_path / "labels.parquet"
    labels = _training_labels(tmp_path / "unused_dataset")
    labels.to_parquet(labels_path, index=False)
    attached = 0

    def load_cache(rows: pd.DataFrame, *_: object) -> pd.DataFrame:
        return rows.assign(dinov2_0=range(len(rows)))

    def attach(rows: pd.DataFrame, _dataset: Path) -> pd.DataFrame:
        nonlocal attached
        attached += 1
        result = rows.assign(sensor_timestamp=rows["image_time"])
        for name in CURRENT_SENSORS:
            result[name] = 1.0
        for name in SLOPE_SENSORS:
            result[f"{name}__slope_5min"] = 0.1
        return result

    monkeypatch.setattr(train_image_models, "load_dinov2_feature_cache", load_cache)
    monkeypatch.setattr(train_image_models, "attach_latest_past_sensor_values", attach)
    args = train_image_models.build_parser().parse_args(
        [
            "--labels",
            str(labels_path),
            "--dataset",
            str(tmp_path / "dataset"),
            "--dinov2-feature-cache",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "trained"),
            "--image-features",
            "dinov2_cache",
            "--classifiers",
            "logistic_regression",
            "--input-features",
            "image_plus_sensor_slopes",
            "--workers",
            "1",
        ]
    )

    assert train_image_models.run(args) == 0
    assert attached == 1
    assert set(pd.read_csv(tmp_path / "trained/fold_metrics.csv")["status"]) == {"ok"}


def test_parallel_run_writes_task_log_in_completion_order(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    labels = tmp_path / "image_labels.parquet"
    _labels().to_parquet(labels, index=False)
    output = tmp_path / "trained"

    def result(index: int) -> tuple[int, dict[str, object]]:
        return index, {
            "metrics": {"status": "ok"},
            "predictions": pd.DataFrame(),
            "model": None,
        }

    class Executor:
        def __init__(self, *, max_workers: int) -> None:
            assert max_workers == 2

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def submit(self, function, fold):  # type: ignore[no-untyped-def]
            future = train_image_models.concurrent.futures.Future()
            future.set_result(result(fold[0]))
            return future

    monkeypatch.setattr(train_image_models.concurrent.futures, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(
        train_image_models.concurrent.futures,
        "as_completed",
        lambda futures: reversed(list(futures)),
    )
    args = train_image_models.build_parser().parse_args(
        [
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--classifiers",
            "logistic_regression",
            "random_forest",
            "--input-features",
            "elapsed_time_only",
            "--workers",
            "2",
        ]
    )

    assert train_image_models.run(args) == 0
    task_log = [json.loads(line) for line in (output / "fold_log.jsonl").read_text().splitlines()]
    assert [row["task_index"] for row in task_log] == [1, 0, 3, 2]


def test_main_passes_seed_unchanged_to_frozen_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = tmp_path / "image_labels.parquet"
    _labels().to_parquet(labels, index=False)
    seeds: list[int] = []

    def fake_fold(*args: object, **kwargs: object) -> dict[str, object]:
        seeds.append(kwargs["seed"])  # type: ignore[arg-type]
        return {"metrics": {"status": "ok"}, "predictions": pd.DataFrame(), "model": None}

    monkeypatch.setattr(train_image_models, "train_frozen_feature_fold", fake_fold)
    args = train_image_models.build_parser().parse_args(
        [
            "--labels",
            str(labels),
            "--output",
            str(tmp_path / "trained"),
            "--classifiers",
            "logistic_regression",
            "--input-features",
            "elapsed_time_only",
            "--workers",
            "1",
            "--seed",
            "23",
        ]
    )

    assert train_image_models.run(args) == 0
    assert seeds == [23, 23]


def test_wandb_is_optional_and_logged_only_by_main_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = tmp_path / "image_labels.parquet"
    _labels().to_parquet(labels, index=False)
    output = tmp_path / "trained"
    initialized: list[dict[str, object]] = []

    class Run:
        def __init__(self) -> None:
            self.logs: list[dict[str, object]] = []
            self.finished = False

        def log(self, values: dict[str, object]) -> None:
            self.logs.append(values)

        def finish(self) -> None:
            self.finished = True

    run = Run()

    def init(**kwargs: object) -> Run:
        initialized.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=init))
    args = train_image_models.build_parser().parse_args(
        [
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--classifiers",
            "logistic_regression",
            "--input-features",
            "elapsed_time_only",
            "--workers",
            "1",
            "--wandb-project",
            "image_only-feature-matrix",
            "--wandb-run-name",
            "review-run",
        ]
    )

    assert train_image_models.run(args) == 0

    assert initialized[0]["project"] == "image_only-feature-matrix"
    assert initialized[0]["name"] == "review-run"
    assert len(run.logs) == 2
    assert run.logs[0]["task_step"] == 1
    assert run.logs[-1]["progress/completed"] == 2
    assert run.finished


def test_all_invalid_folds_still_write_the_prediction_schema(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    labels = _labels().loc[[0, 5]].copy()
    labels_path = tmp_path / "image_labels.parquet"
    labels.to_parquet(labels_path, index=False)
    output = tmp_path / "invalid"
    args = train_image_models.build_parser().parse_args(
        [
            "--labels",
            str(labels_path),
            "--output",
            str(output),
            "--classifiers",
            "logistic_regression",
            "--input-features",
            "elapsed_time_only",
            "--workers",
            "1",
        ]
    )

    assert train_image_models.run(args) == 0

    assert set(pd.read_parquet(output / "predictions.parquet").columns) == {
        "experiment_id",
        "cycle_name",
        "camera",
        "camera_role",
        "image_time",
        "held_out_experiment",
        "image_feature",
        "classifier",
        "input_feature",
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
            for target, state in ((0, "before_reference"), (1, "after_reference")):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "cycle_name": f"cycle_{experiment}",
                        "camera_role": camera,
                        "file_name": f"{target}.png",
                        "image_path": f"{experiment}/{target}.png",
                        "image_time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=target),
                        "stable_heating_start": pd.Timestamp("2026-01-01"),
                        "relative_regret": 0.2,
                        "timing_state_01pct": state,
                        "binary_target_01pct": state,
                    }
                )
    labels = tmp_path / "image_labels.parquet"
    pd.DataFrame(rows).to_parquet(labels, index=False)
    output = tmp_path / "run"
    args = train_image_models.build_parser().parse_args(
        [
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--classifiers",
            "logistic_regression",
            "--camera-groups",
            "front",
            "top",
            "--input-features",
            "elapsed_time_only",
            "--workers",
            "1",
        ]
    )

    assert train_image_models.run(args) == 0

    metrics = pd.read_csv(output / "fold_metrics.csv")
    assert len(metrics) == 4
    assert set(zip(metrics["camera"], metrics["held_out_experiment"], strict=True)) == {
        ("front", "a"),
        ("front", "b"),
        ("top", "c"),
        ("top", "d"),
    }
    assert "Total tasks: 4" in capsys.readouterr().out


def test_parallel_fits_limit_inner_threadpools(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "dataset"
    labels = tmp_path / "image_labels.parquet"
    _training_labels(dataset).to_parquet(labels, index=False)
    active = False
    limits: list[int] = []

    class Limit:
        def __init__(self, value: int) -> None:
            limits.append(value)

        def __enter__(self) -> None:
            nonlocal active
            active = True

        def __exit__(self, *args: object) -> None:
            nonlocal active
            active = False

    original = train_image_models.train_frozen_feature_fold

    def checked(*args, **kwargs):  # type: ignore[no-untyped-def]
        assert active
        return original(*args, **kwargs)

    monkeypatch.setattr(train_image_models, "threadpool_limits", Limit, raising=False)
    monkeypatch.setattr(train_image_models, "train_frozen_feature_fold", checked)
    args = train_image_models.build_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--labels",
            str(labels),
            "--output",
            str(tmp_path / "run"),
            "--classifiers",
            "logistic_regression",
            "--input-features",
            "elapsed_time_only",
            "--workers",
            "2",
        ]
    )

    assert train_image_models.run(args) == 0
    assert limits == [1]
