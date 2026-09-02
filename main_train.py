#!/usr/bin/env python3
"""Train fixed leave-one-experiment-out image models."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import pickle
import shlex
import sys
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from labels.build import CAMERA_GROUPS
from model.features import prepare_features
from model.model import train_frozen_fold

REPRESENTATIONS = ("handcrafted", "resnet50_finetune")
HEADS = ("logistic", "random_forest", "rbf_svm", "paper_mlp")
MODALITIES = ("rgb", "time", "rgb_time")
PREDICTION_COLUMNS = (
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
)


class Setting(NamedTuple):
    representation: str
    head: str
    camera: str
    modality: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("output/labels/v1/image_cost_labels.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("output/models/current"))
    parser.add_argument("--task", choices=("binary", "three"), default="binary")
    parser.add_argument("--state-column", default="cost_state_01pct")
    parser.add_argument(
        "--representations", nargs="+", choices=REPRESENTATIONS, default=["handcrafted"]
    )
    parser.add_argument("--heads", nargs="+", choices=HEADS, default=["rbf_svm"])
    parser.add_argument(
        "--cameras", nargs="+", choices=tuple(CAMERA_GROUPS), default=["front"]
    )
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=["rgb"])
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--maximum-per-group", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_setting(setting: Setting, jobs: int) -> None:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if setting.representation == "resnet50_finetune":
        if setting.head != "paper_mlp":
            raise ValueError("resnet50_finetune requires the paper_mlp head")
        if setting.modality != "rgb":
            raise ValueError("resnet50_finetune requires the rgb modality")
        if jobs != 1:
            raise ValueError("resnet50_finetune requires jobs=1")
    elif setting.head == "paper_mlp":
        raise ValueError("paper_mlp is only valid with resnet50_finetune")


def expand_settings(args: argparse.Namespace) -> list[Setting]:
    settings = [
        Setting(*values)
        for values in itertools.product(
            args.representations, args.heads, args.cameras, args.modalities
        )
    ]
    for setting in settings:
        validate_setting(setting, args.jobs)
    return settings


def label_rows(
    labels: pd.DataFrame, *, task: str, state_column: str, camera: str
) -> pd.DataFrame:
    required = {
        "experiment_id",
        "cycle_name",
        "camera_role",
        "image_time",
        "image_path",
        "stable_heating_start",
        "relative_regret",
        state_column,
    }
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"labels are missing required columns: {', '.join(missing)}")
    rows = labels.loc[
        labels["relative_regret"].notna()
        & labels["camera_role"].isin(CAMERA_GROUPS[camera])
    ].copy()
    mapping = (
        {"pre_optimal": 0, "post_optimal": 1}
        if task == "binary"
        else {"pre_optimal": 0, "near_optimal": 1, "post_optimal": 2}
    )
    rows["target"] = rows[state_column].map(mapping)
    rows = rows.loc[rows["target"].notna()].copy()
    rows["target"] = rows["target"].astype("int64")
    rows["image_time"] = pd.to_datetime(rows["image_time"], errors="raise", format="mixed")
    return rows.reset_index(drop=True)


def _train_frozen_task(
    task: tuple[int, str, Setting, pd.DataFrame, list[str], str, bool],
) -> tuple[int, dict[str, Any]]:
    index, experiment, setting, rows, feature_columns, task_name, return_model = task
    result = train_frozen_fold(
        rows,
        feature_columns,
        heldout_experiment=experiment,
        head=setting.head,
        representation=setting.representation,
        camera=setting.camera,
        modality=setting.modality,
        task=task_name,
        return_model=return_model,
    )
    return index, result


def _write_result(
    output: Path,
    task_index: int,
    result: dict[str, Any],
    metrics: list[dict[str, Any]],
    predictions: list[pd.DataFrame],
    *,
    save_model: bool,
) -> None:
    row = dict(result["metrics"])
    row["task_index"] = task_index
    metrics.append(row)
    prediction = result["predictions"]
    if not prediction.empty:
        predictions.append(prediction)
    with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, default=str) + "\n")
    if save_model and result["model"] is not None:
        model_dir = output / "models"
        model_dir.mkdir(exist_ok=True)
        with (model_dir / f"fold_{task_index:04d}.pkl").open("wb") as stream:
            pickle.dump(result["model"], stream)


def run(  # noqa: C901 - this is the explicit setting/fold orchestration view.
    args: argparse.Namespace, *, command: list[str] | None = None
) -> int:
    settings = expand_settings(args)
    labels = pd.read_parquet(args.labels)
    selected_settings = [
        (
            setting,
            label_rows(
                labels,
                task=args.task,
                state_column=args.state_column,
                camera=setting.camera,
            ),
        )
        for setting in settings
    ]
    heldout = sorted(
        {
            str(experiment)
            for _, selected in selected_settings
            for experiment in selected["experiment_id"].unique()
        }
    )
    total_tasks = sum(
        selected["experiment_id"].nunique() for _, selected in selected_settings
    )
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
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
    (args.output / "command.txt").write_text(
        shlex.join(command or sys.argv) + "\n", encoding="utf-8"
    )
    (args.output / "args.json").write_text(
        json.dumps(serializable_args, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([setting._asdict() for setting in settings]).to_csv(
        args.output / "settings.csv", index=False
    )
    (args.output / "progress.jsonl").touch()

    folds: list[tuple[int, str, Setting, pd.DataFrame, list[str], str, bool]] = []
    for setting, selected in selected_settings:
        if setting.representation == "resnet50_finetune":
            selected["absolute_path"] = selected["image_path"].map(
                lambda value: str((args.dataset / str(value)).resolve())
            )
            feature_columns: list[str] = []
        else:
            selected, feature_columns = prepare_features(
                selected,
                dataset_root=args.dataset,
                representation=setting.representation,
                camera=setting.camera,
                modality=setting.modality,
                state_column=args.state_column,
                maximum_per_group=args.maximum_per_group,
            )
        for experiment in selected["experiment_id"].unique():
            folds.append(
                (
                    len(folds),
                    str(experiment),
                    setting,
                    selected,
                    feature_columns,
                    args.task,
                    args.save_models,
                )
            )

    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    completed: list[tuple[int, dict[str, Any]]] = []
    if args.jobs > 1:
        with threadpool_limits(1), concurrent.futures.ThreadPoolExecutor(
            max_workers=args.jobs
        ) as executor:
            completed = list(executor.map(_train_frozen_task, folds))
    else:
        for fold in folds:
            index, experiment, setting, selected, _, _, _ = fold
            if setting.representation == "resnet50_finetune":
                from model.resnet import train_resnet_fold

                result = train_resnet_fold(
                    selected,
                    heldout_experiment=experiment,
                    task=args.task,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    save_path=(
                        args.output / "models" / f"fold_{index:04d}.pt"
                        if args.save_models
                        else None
                    ),
                    camera=setting.camera,
                )
                completed.append((index, result))
            else:
                completed.append(_train_frozen_task(fold))

    for index, result in completed:
        _write_result(
            args.output,
            index,
            result,
            metrics,
            predictions,
            save_model=args.save_models,
        )

    pd.DataFrame(metrics).sort_values("task_index").to_csv(
        args.output / "metrics.csv", index=False
    )
    combined = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=PREDICTION_COLUMNS)
    )
    combined.to_parquet(args.output / "predictions.parquet", index=False)
    return 0


def main() -> None:
    run(build_parser().parse_args(), command=sys.argv)


if __name__ == "__main__":
    main()
