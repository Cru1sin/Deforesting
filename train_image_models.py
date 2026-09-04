#!/usr/bin/env python3
"""Train fixed leave-one-experiment-out image models."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import itertools
import json
import pickle
import shlex
import sys
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from image_labels.timing import CAMERA_GROUPS
from image_models.classifiers import train_frozen_feature_fold
from image_models.image_features import (
    load_dinov2_feature_cache,
    prepare_cached_features,
    prepare_features,
)
from image_models.sensor_features import attach_latest_past_sensor_values

IMAGE_FEATURES = ("color_gradient", "dinov2_cache", "resnet50_end_to_end")
CLASSIFIERS = ("logistic_regression", "random_forest", "rbf_svm", "resnet_mlp")
INPUT_FEATURES = (
    "image_only",
    "elapsed_time_only",
    "image_plus_elapsed_time",
    "image_plus_current_sensors",
    "image_plus_sensor_slopes",
)
SENSOR_INPUT_FEATURES = {"image_plus_current_sensors", "image_plus_sensor_slopes"}
PREDICTION_COLUMNS = (
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
)


class Setting(NamedTuple):
    image_feature: str
    classifier: str
    camera: str
    input_feature: str


class FoldTask(NamedTuple):
    index: int
    heldout_experiment: str
    setting: Setting
    rows: pd.DataFrame
    feature_columns: list[str]
    task: str
    save_model: bool
    seed: int
    test_rows: pd.DataFrame | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("output/image_labels/v1_label_reference/image_timing_labels.parquet"),
    )
    parser.add_argument("--dinov2-feature-cache", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/image_models/current"))
    parser.add_argument("--task", choices=("binary", "three"), default="binary")
    parser.add_argument("--label-column", default="binary_target_01pct")
    parser.add_argument(
        "--image-features", nargs="+", choices=IMAGE_FEATURES, default=["color_gradient"]
    )
    parser.add_argument("--classifiers", nargs="+", choices=CLASSIFIERS, default=["rbf_svm"])
    parser.add_argument(
        "--camera-groups", nargs="+", choices=tuple(CAMERA_GROUPS), default=["front"]
    )
    parser.add_argument(
        "--input-features", nargs="+", choices=INPUT_FEATURES, default=["image_only"]
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images-per-cycle-label", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_setting(setting: Setting, workers: int) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    if setting.image_feature == "resnet50_end_to_end":
        if setting.classifier != "resnet_mlp":
            raise ValueError("resnet50_end_to_end requires the resnet_mlp classifier")
        if setting.input_feature != "image_only":
            raise ValueError("resnet50_end_to_end requires the image_only input_feature")
        if workers != 1:
            raise ValueError("resnet50_end_to_end requires workers=1")
    elif setting.classifier == "resnet_mlp":
        raise ValueError("resnet_mlp is only valid with resnet50_end_to_end")
    if (
        setting.input_feature in {"image_plus_current_sensors", "image_plus_sensor_slopes"}
        and setting.image_feature != "dinov2_cache"
    ):
        raise ValueError("sensor input_features require cached dinov2 features")


def build_training_settings(args: argparse.Namespace) -> list[Setting]:
    settings = [
        Setting(*values)
        for values in itertools.product(
            args.image_features, args.classifiers, args.camera_groups, args.input_features
        )
    ]
    for setting in settings:
        validate_setting(setting, args.workers)
    return settings


def select_labeled_images(
    labels: pd.DataFrame, *, task: str, label_column: str, camera: str
) -> pd.DataFrame:
    required = {
        "experiment_id",
        "cycle_name",
        "camera_role",
        "image_time",
        "image_path",
        "stable_heating_start",
        label_column,
    }
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"labels are missing required columns: {', '.join(missing)}")
    rows = labels.loc[
        labels[label_column].notna() & labels["camera_role"].isin(CAMERA_GROUPS[camera])
    ].copy()
    mapping = (
        {"before_reference": 0, "after_reference": 1}
        if task == "binary"
        else {"before_reference": 0, "near_reference": 1, "after_reference": 2}
    )
    rows["target"] = rows[label_column].map(mapping)
    rows = rows.loc[rows["target"].notna()].copy()
    rows["target"] = rows["target"].astype("int64")
    rows["image_time"] = pd.to_datetime(rows["image_time"], errors="raise", format="mixed")
    return rows.reset_index(drop=True)


def _train_frozen_task(task: FoldTask) -> tuple[int, dict[str, Any]]:
    result = train_frozen_feature_fold(
        task.rows,
        task.feature_columns,
        heldout_experiment=task.heldout_experiment,
        classifier=task.setting.classifier,
        image_feature=task.setting.image_feature,
        camera=task.setting.camera,
        input_feature=task.setting.input_feature,
        task=task.task,
        return_model=task.save_model,
        seed=task.seed,
        test_rows=task.test_rows,
    )
    return task.index, result


def train_heldout_experiments(
    folds: list[FoldTask],
    workers: int,
) -> Any:
    if workers == 1:
        yield from map(_train_frozen_task, folds)
        return
    with (
        threadpool_limits(1),
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        futures = [executor.submit(_train_frozen_task, fold) for fold in folds]
        for future in concurrent.futures.as_completed(futures):
            yield future.result()


def _write_result(
    output: Path,
    task_index: int,
    result: dict[str, Any],
    metrics: list[dict[str, Any]],
    predictions: list[pd.DataFrame],
    *,
    save_model: bool,
    total_tasks: int,
    wandb_run: Any | None,
) -> None:
    row = dict(result["metrics"])
    row["task_index"] = task_index
    metrics.append(row)
    prediction = result["predictions"]
    if not prediction.empty:
        predictions.append(prediction)
    with (output / "fold_log.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, default=str) + "\n")
    if wandb_run is not None:
        try:
            wandb_run.log(
                {
                    "task_step": len(metrics),
                    "progress/completed": len(metrics),
                    "progress/total": total_tasks,
                    **{
                        f"latest/{key}": value
                        for key, value in row.items()
                        if isinstance(value, (int, float))
                    },
                }
            )
        except Exception as exc:
            print(f"[W&B warning] {exc}", flush=True)
    if save_model and result["model"] is not None:
        model_dir = output / "models"
        model_dir.mkdir(exist_ok=True)
        with (model_dir / f"fold_{task_index:04d}.pkl").open("wb") as stream:
            pickle.dump(result["model"], stream)


def run(  # noqa: C901 - this is the explicit setting/fold orchestration view.
    args: argparse.Namespace, *, command: list[str] | None = None
) -> int:
    settings = build_training_settings(args)
    labels = pd.read_parquet(args.labels)
    heldout: set[str] = set()
    total_tasks = 0
    for setting in settings:
        selected = select_labeled_images(
            labels,
            task=args.task,
            label_column=args.label_column,
            camera=setting.camera,
        )
        experiments = selected["experiment_id"].unique()
        heldout.update(map(str, experiments))
        total_tasks += len(experiments)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    print(f"Args: {json.dumps(serializable_args, sort_keys=True)}")
    print(f"Settings: {len(settings)}")
    for setting in settings:
        print("  " + " / ".join(setting))
    print(f"Held-out experiments: {len(heldout)}")
    print(f"Total tasks: {total_tasks}")
    if args.dry_run:
        return 0
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError(f"output exists and is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    serializable_args["command"] = shlex.join(["uv", "run", "python", *(command or sys.argv)])
    (args.output / "run_settings.json").write_text(
        json.dumps(serializable_args, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([setting._asdict() for setting in settings]).to_csv(
        args.output / "training_settings.csv", index=False
    )
    (args.output / "fold_log.jsonl").touch()

    wandb_run: Any | None = None
    if args.wandb_project:
        try:
            wandb = importlib.import_module("wandb")
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    **serializable_args,
                    "settings": [setting._asdict() for setting in settings],
                    "total_tasks": total_tasks,
                },
            )
        except Exception as exc:
            print(f"[W&B warning] {exc}", flush=True)

    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    deep_features: pd.DataFrame | None = None
    if any(setting.image_feature == "dinov2_cache" for setting in settings):
        if args.dinov2_feature_cache is None:
            raise ValueError("dinov2_cache requires --dinov2-feature-cache")
        deep_features = load_dinov2_feature_cache(labels, args.dinov2_feature_cache, "dinov2")
        if any(setting.input_feature in SENSOR_INPUT_FEATURES for setting in settings):
            deep_features = attach_latest_past_sensor_values(deep_features, args.dataset)
    task_index = 0
    for setting in settings:
        selected = select_labeled_images(
            deep_features if setting.image_feature == "dinov2_cache" else labels,
            task=args.task,
            label_column=args.label_column,
            camera=setting.camera,
        )
        if setting.image_feature == "resnet50_end_to_end":
            selected["absolute_path"] = selected["image_path"].map(
                lambda value: str((args.dataset / str(value)).resolve())
            )
            feature_columns: list[str] = []
        elif setting.image_feature == "dinov2_cache":
            selected, full_stream, feature_columns = prepare_cached_features(
                selected,
                image_feature="dinov2",
                camera=setting.camera,
                input_feature=setting.input_feature,
                label_column=args.label_column,
                max_images_per_cycle_label=args.max_images_per_cycle_label,
            )
        else:
            selected, feature_columns = prepare_features(
                selected,
                dataset_root=args.dataset,
                image_feature=setting.image_feature,
                camera=setting.camera,
                input_feature=setting.input_feature,
                label_column=args.label_column,
                max_images_per_cycle_label=args.max_images_per_cycle_label,
            )
            full_stream = selected
        experiments = [str(value) for value in selected["experiment_id"].unique()]
        if setting.image_feature == "resnet50_end_to_end":
            from image_models.resnet50 import train_resnet_fold

            for experiment in experiments:
                result = train_resnet_fold(
                    selected,
                    heldout_experiment=experiment,
                    task=args.task,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    save_path=(
                        args.output / "models" / f"fold_{task_index:04d}.pt"
                        if args.save_models
                        else None
                    ),
                    camera=setting.camera,
                    seed=args.seed,
                )
                _write_result(
                    args.output,
                    task_index,
                    result,
                    metrics,
                    predictions,
                    save_model=args.save_models,
                    total_tasks=total_tasks,
                    wandb_run=wandb_run,
                )
                task_index += 1
            continue

        folds = []
        for experiment in experiments:
            folds.append(
                FoldTask(
                    index=task_index + len(folds),
                    heldout_experiment=experiment,
                    setting=setting,
                    rows=selected,
                    feature_columns=feature_columns,
                    task=args.task,
                    save_model=args.save_models,
                    seed=args.seed,
                    test_rows=full_stream,
                )
            )
        for index, result in train_heldout_experiments(folds, args.workers):
            _write_result(
                args.output,
                index,
                result,
                metrics,
                predictions,
                save_model=args.save_models,
                total_tasks=total_tasks,
                wandb_run=wandb_run,
            )
        task_index += len(folds)

    pd.DataFrame(metrics).sort_values("task_index").to_csv(
        args.output / "fold_metrics.csv", index=False
    )
    combined = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=PREDICTION_COLUMNS)
    )
    combined.to_parquet(args.output / "predictions.parquet", index=False)
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception as exc:
            print(f"[W&B warning] {exc}", flush=True)
    return 0


def main() -> None:
    run(build_parser().parse_args(), command=sys.argv)


if __name__ == "__main__":
    main()
