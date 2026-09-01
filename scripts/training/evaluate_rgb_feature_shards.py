#!/usr/bin/env python3
"""Evaluate the locked RGB feature model by held-out experiment."""

from __future__ import annotations

import argparse
import importlib
import re
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
from joblib import parallel_config
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.parallel import Parallel, delayed
from threadpoolctl import threadpool_limits

from frost_analysis.labels.cost import high_confidence_coverage, map_cost_state_targets
from frost_analysis.training.evaluation import (
    CAMERA_GROUPS,
    MODEL_NAMES,
    REPRESENTATION_PREFIXES,
    REPRESENTATIONS,
    add_cycle_time_features,
    bootstrap_mean_interval,
    evaluate_holdout_task,
    experiment_prediction_metrics,
    retain_high_confidence_rows,
    three_class_eligible_image_coverage,
)
from frost_analysis.training.run import (
    COMBINATION_COLUMNS,
    RunStore,
    atomic_csv,
    atomic_parquet,
    stable_task_id,
    validate_completed_run,
)

MANIFEST_COLUMNS = (
    "camera_group",
    "regret_threshold",
    "representation",
    "model",
    "modality",
)
STATE_SENSORS = (
    "coil_temperature",
    "evaporating_pressure",
    "fan_current",
    "ambient_temperature",
)
ALL_SENSORS = (
    *STATE_SENSORS,
    "cop",
    "compressor_power",
    "evaporator_capacity__baseline_residual",
)


def read_experiment_manifest(
    path: Path,
    *,
    stages: list[str] | None = None,
    task: str | None = None,
) -> pd.DataFrame:
    """Read exact experiments, avoiding an accidental full Cartesian benchmark."""
    plan = pd.read_csv(path)
    missing = set(MANIFEST_COLUMNS) - set(plan)
    if missing:
        raise ValueError(f"experiment manifest missing columns: {sorted(missing)}")
    if "stage" not in plan:
        plan["stage"] = "custom"
    if "task" not in plan:
        plan["task"] = "binary"
    if stages:
        plan = plan.loc[plan["stage"].isin(stages)]
    if task:
        plan = plan.loc[plan["task"].eq(task)]
    return plan[["stage", "task", *MANIFEST_COLUMNS]].drop_duplicates().reset_index(drop=True)


def combination_stages(plan: pd.DataFrame) -> dict[tuple[object, ...], str]:
    """Map each actual training combination to its possibly shared stages."""
    return {
        key: "/".join(rows["stage"].astype(str).drop_duplicates())
        for key, rows in plan.groupby(list(MANIFEST_COLUMNS), sort=False)
    }


def audit_holdout_cohort(features: pd.DataFrame) -> pd.DataFrame:
    """Require one experiment per cycle and one row per camera timestamp."""
    required = {"experiment_id", "cycle_name", "camera_role", "image_time"}
    missing = required - set(features)
    if missing:
        raise ValueError(f"feature cohort missing holdout keys: {sorted(missing)}")
    if features[list(required)].isna().any().any():
        raise ValueError("feature cohort has missing holdout keys")
    cycle_experiments = features.groupby("cycle_name")["experiment_id"].nunique()
    if cycle_experiments.gt(1).any():
        raise ValueError("one or more cycles belong to multiple experiments")
    keys = ["experiment_id", "cycle_name", "camera_role", "image_time"]
    if features.duplicated(keys).any():
        raise ValueError("feature cohort contains duplicate camera timestamps")
    return (
        features.groupby("experiment_id", as_index=False)
        .agg(cycle_count=("cycle_name", "nunique"), image_count=("cycle_name", "size"))
        .sort_values("experiment_id")
    )


def target_shard_paths(shards: Path, labels_path: Path) -> list[Path]:
    """Select exactly the labeled cycle cohort, excluding historical shards."""
    labels = pd.read_parquet(labels_path, columns=["cycle_name", "relative_regret"])
    cycles = sorted(
        labels.loc[labels["relative_regret"].notna(), "cycle_name"].astype(str).unique()
    )
    paths = [shards / f"{cycle}.parquet" for cycle in cycles]
    missing = [path.stem for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing target feature shards: {', '.join(missing)}")
    return paths


def build_modality_frames(scoped: pd.DataFrame, representation: str) -> dict[str, pd.DataFrame]:
    """Build RGB and elapsed-time modalities without future cycle-end information."""
    prefix = REPRESENTATION_PREFIXES[representation]
    rgb_prefixes = tuple(REPRESENTATION_PREFIXES.values())
    rgb_columns = [column for column in scoped if column.startswith(rgb_prefixes)]
    time_only = scoped.drop(columns=rgb_columns).copy()
    time_only["feature_000"] = scoped["time_elapsed_minutes"]
    rgb_time = scoped.copy()
    rgb_time[f"{prefix}time_elapsed_minutes"] = scoped["time_elapsed_minutes"]
    def with_sensors(columns):  # type: ignore[no-untyped-def]
        result = scoped.copy()
        for column in columns:
            result[f"{prefix}sensor_{column}"] = scoped[column]
        return result

    modalities = {
        "rgb": scoped,
        "time": time_only,
        "rgb_time": rgb_time,
    }
    if set(STATE_SENSORS).issubset(scoped):
        modalities["rgb_state"] = with_sensors(STATE_SENSORS)
    if set(ALL_SENSORS).issubset(scoped):
        modalities["rgb_all_sensor"] = with_sensors(ALL_SENSORS)
    return modalities


def add_causal_sensor_features(
    images: pd.DataFrame, sensor_dir: Path, tolerance_seconds: float = 15.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match each image to the latest same-cycle sensor row, never to the future."""
    paths = [sensor_dir / f"{cycle}.parquet" for cycle in images["cycle_name"].unique()]
    missing = [path.stem for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing sensor cycles: {', '.join(missing)}")
    columns = ["cycle_name", "timestamp", *ALL_SENSORS]
    sensors = pd.concat(pd.read_parquet(path, columns=columns) for path in paths)
    sensors["sensor_timestamp"] = pd.to_datetime(sensors.pop("timestamp"), format="mixed")
    result = pd.merge_asof(
        images.assign(image_time=pd.to_datetime(images["image_time"], format="mixed")).sort_values(
            "image_time"
        ),
        sensors.sort_values("sensor_timestamp"),
        left_on="image_time",
        right_on="sensor_timestamp",
        by="cycle_name",
        direction="backward",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    ).sort_index()
    matched = result["sensor_timestamp"].notna()
    audit = pd.DataFrame(
        {
            "model_input_image_count": [len(result)],
            "sensor_matched_image_count": [int(matched.sum())],
            "sensor_match_rate": [float(matched.mean())],
            "tolerance_seconds": [tolerance_seconds],
        }
    )
    return result, audit


def score_rows(frame: pd.DataFrame) -> dict[str, float]:
    if "fold_evaluable" in frame:
        frame = frame.loc[frame["fold_evaluable"]]
    if frame.empty or frame["target"].nunique() < 2:
        return dict.fromkeys(
            (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "positive_f1",
                "precision",
                "recall",
                "auroc",
            ),
            float("nan"),
        )
    score_columns = sorted(
        (column for column in frame if column.startswith("decision_score_")),
        key=lambda column: int(column.removeprefix("decision_score_")),
    )
    try:
        auroc = (
            roc_auc_score(frame["target"], frame[score_columns], multi_class="ovr", average="macro")
            if score_columns
            else roc_auc_score(frame["target"], frame["decision_score"])
        )
    except ValueError:
        auroc = float("nan")
    binary = frame["target"].nunique() == 2
    return {
        "accuracy": accuracy_score(frame["target"], frame["predicted_target"]),
        "balanced_accuracy": balanced_accuracy_score(frame["target"], frame["predicted_target"]),
        "macro_f1": f1_score(frame["target"], frame["predicted_target"], average="macro"),
        "positive_f1": (
            f1_score(frame["target"], frame["predicted_target"], pos_label=1)
            if binary
            else float("nan")
        ),
        "precision": (
            precision_score(
                frame["target"], frame["predicted_target"], pos_label=1, zero_division=0
            )
            if binary
            else float("nan")
        ),
        "recall": (
            recall_score(
                frame["target"], frame["predicted_target"], pos_label=1, zero_division=0
            )
            if binary
            else float("nan")
        ),
        "auroc": auroc,
    }


def _experiment_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions["target"].nunique() == 2:
        return experiment_prediction_metrics(predictions)
    rows = []
    for experiment, values in predictions.groupby("experiment_id", sort=True):
        scores = score_rows(values)
        evaluable = pd.notna(scores["accuracy"])
        recalls = (
            recall_score(
                values["target"],
                values["predicted_target"],
                labels=sorted(values["target"].unique()),
                average=None,
                zero_division=0,
            )
            if evaluable
            else [float("nan")] * 3
        )
        incorrect_regret = values["relative_regret"].where(
            values["target"].ne(values["predicted_target"]), 0.0
        )
        rows.append(
            {
                "experiment_id": experiment,
                "evaluable": evaluable,
                "recall_before": recalls[0],
                "recall_within": recalls[1],
                "recall_after": recalls[-1],
                **scores,
                "balanced_misclassification_regret": incorrect_regret.groupby(
                    values["target"]
                ).mean().mean(),
                "image_count": len(values),
                "cycle_count": values["cycle_name"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def _parse_args():  # type: ignore[no-untyped-def]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards",
        type=Path,
        default=Path("output/test/model/RGB特征缓存/手工特征/cycles"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("output/test/成本函数/其他/经验经济窗口/源数据/candidate_cost_curves.parquet"),
    )
    parser.add_argument(
        "--label-balance",
        type=Path,
        default=Path("output/label/cost_function_v1_binary/label_balance.csv"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("output/label/cost_function_v1_binary/image_cost_labels.parquet"),
    )
    parser.add_argument("--camera-groups", nargs="+", choices=tuple(CAMERA_GROUPS), default=["all"])
    parser.add_argument("--task", choices=("binary", "near_binary", "three"), default="binary")
    parser.add_argument("--sensor-dir", type=Path, default=Path("dataset/cycles"))
    parser.add_argument("--stages", nargs="+")
    parser.add_argument("--experiment-manifest", type=Path)
    parser.add_argument(
        "--regret-thresholds", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.10]
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["rbf_svm"])
    parser.add_argument(
        "--representations", nargs="+", choices=REPRESENTATIONS, default=["handcrafted"]
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=("rgb", "time", "rgb_time", "rgb_state", "rgb_all_sensor"),
        default=["rgb", "time", "rgb_time"],
    )
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--backend", choices=("threading", "loky"), default="threading")
    parser.add_argument("--run-id")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--output", type=Path, default=Path("output/model/rgb_full_cohort_latest"))
    return parser.parse_args()


def _safe_wandb(label, function, *args, **kwargs):  # type: ignore[no-untyped-def]
    try:
        return function(*args, **kwargs)
    except Exception as error:
        print(f"[W&B warning] {label}: {type(error).__name__}: {error}", flush=True)
        return None


def _safe_wandb_method(run, method_name, *args, **kwargs):  # type: ignore[no-untyped-def]
    return _safe_wandb(method_name, lambda: getattr(run, method_name)(*args, **kwargs))


def _combinations(args, plan):  # type: ignore[no-untyped-def]
    if plan is not None:
        return list(combination_stages(plan))
    return [
        (camera, threshold, representation, model, modality)
        for camera in args.camera_groups
        for threshold in args.regret_thresholds
        for representation in args.representations
        for model in args.models
        for modality in args.modalities
    ]


def normalized_combinations(combinations):  # type: ignore[no-untyped-def]
    """Return ordered combination fields as JSON-comparable basic values."""
    return [
        {
            column: value.item() if hasattr(value, "item") else value
            for column, value in zip(COMBINATION_COLUMNS, combination, strict=True)
        }
        for combination in combinations
    ]


def validate_formal_run_shape(
    *, task: str, has_manifest: bool, combination_count: int, experiments: list[object]
) -> int:
    """Lock the formal three-class matrix to 51 combinations by 14 experiments."""
    experiment_count = len(set(experiments))
    if task == "three" and has_manifest:
        if combination_count != 51:
            raise SystemExit("formal three-class run requires 51 unique combinations")
        if experiment_count != 14:
            raise SystemExit("formal three-class run requires 14 unique experiments")
        return 714
    return combination_count * experiment_count


def _worker(frame, heldout, metadata, expected_classes, inner_jobs):  # type: ignore[no-untyped-def]
    result, predictions = evaluate_holdout_task(
        frame,
        heldout,
        model_name=metadata["model"],
        representation="handcrafted"
        if metadata["modality"] == "time"
        else metadata["representation"],
        expected_classes=expected_classes,
        n_jobs=inner_jobs,
    )
    return {**metadata, **result, "heldout": heldout}, predictions


def _write_summaries(store, predictions, ledger, plan, cohort_audit, input_audit):  # type: ignore[no-untyped-def]  # noqa: C901
    metrics = []
    experiment_rows = []
    summary = []
    group_columns = list(COMBINATION_COLUMNS)
    for key, values in predictions.groupby(group_columns, sort=False, dropna=False):
        metadata = dict(zip(group_columns, key, strict=True))
        protocol = (
            "pooled_views"
            if metadata["camera_group"] in {"top_pair", "left_pair", "all"}
            else "single_view"
        )
        retained = values["sample_retained_fraction"].iat[0]
        coverage = values["eligible_image_coverage"].iat[0]
        held = _experiment_scores(values)
        for column, value in metadata.items():
            held[column] = value
        experiment_rows.append(held)
        common = {
            **metadata,
            "training_protocol": protocol,
            "sample_retained_fraction": retained,
            "eligible_image_coverage": coverage,
        }
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "positive_f1",
            "precision",
            "recall",
            "auroc",
            "balanced_misclassification_regret",
        ):
            summary.append(
                {
                    **common,
                    "metric": metric,
                    **bootstrap_mean_interval(held[metric]),
                    "experiment_count": len(held),
                    "evaluable_experiment_count": int(held[metric].notna().sum()),
                }
            )
        metrics.append(
            {
                **common,
                "scope": "all_held_out_predictions",
                **score_rows(values),
                "image_count": len(values),
                "cycle_count": values["cycle_name"].nunique(),
                "experiment_count": values["experiment_id"].nunique(),
            }
        )
        for experiment, rows in values.groupby("experiment_id", sort=True):
            metrics.append(
                {
                    **common,
                    "scope": str(experiment),
                    **score_rows(rows),
                    "image_count": len(rows),
                    "cycle_count": rows["cycle_name"].nunique(),
                    "experiment_count": 1,
                }
            )
    held_out = pd.concat(experiment_rows, ignore_index=True)
    comparisons = []
    melted = held_out.melt(
        id_vars=[
            "experiment_id",
            "representation",
            "model",
            "camera_group",
            "modality",
            "regret_threshold",
        ],
        value_vars=[
            "balanced_accuracy",
            "balanced_misclassification_regret",
            "macro_f1",
            "positive_f1",
        ],
        var_name="metric",
    )
    for key, values in melted.groupby(
        ["representation", "model", "camera_group", "regret_threshold", "metric"], sort=True
    ):
        representation, model, camera, threshold, metric = key
        paired = values.pivot(index="experiment_id", columns="modality", values="value")
        for modality in ("rgb", "rgb_time"):
            if "time" in paired and modality in paired:
                differences = paired[modality] - paired["time"]
                comparisons.append(
                    {
                        "representation": representation,
                        "model": model,
                        "camera_group": camera,
                        "regret_threshold": threshold,
                        "comparison": f"{modality}_minus_time",
                        "metric": metric,
                        **bootstrap_mean_interval(differences),
                        "experiment_count": len(paired),
                        "evaluable_experiment_count": int(differences.notna().sum()),
                    }
                )
        if metric in {"macro_f1", "positive_f1"} and "rgb" in paired:
            for modality in ("rgb_state", "rgb_all_sensor"):
                if modality in paired:
                    differences = paired[modality] - paired["rgb"]
                    comparisons.append(
                        {
                            "representation": representation,
                            "model": model,
                            "camera_group": camera,
                            "regret_threshold": threshold,
                            "comparison": f"{modality}_minus_rgb",
                            "metric": metric,
                            **bootstrap_mean_interval(differences),
                            "experiment_count": len(paired),
                            "evaluable_experiment_count": int(differences.notna().sum()),
                        }
                    )
    executed = ledger[[*COMBINATION_COLUMNS]].drop_duplicates()
    if plan is not None:
        executed = plan.merge(executed, on=list(MANIFEST_COLUMNS), how="inner")
    else:
        executed.insert(0, "stage", "custom")
    atomic_csv(store.run_dir / "cohort_holdout_audit.csv", cohort_audit)
    atomic_csv(store.run_dir / "input_audit.csv", input_audit)
    atomic_csv(store.run_dir / "experiment_manifest.csv", executed)
    atomic_csv(store.run_dir / "metrics.csv", pd.DataFrame(metrics))
    atomic_csv(store.run_dir / "experiment_metrics.csv", held_out)
    atomic_csv(store.run_dir / "summary_metrics.csv", pd.DataFrame(summary))
    atomic_csv(
        store.run_dir / "modality_deltas.csv",
        pd.DataFrame(
            comparisons,
            columns=[
                "representation",
                "model",
                "camera_group",
                "regret_threshold",
                "comparison",
                "metric",
                "estimate",
                "lower",
                "upper",
                "experiment_count",
                "evaluable_experiment_count",
            ],
        ),
    )
    atomic_parquet(store.run_dir / "predictions.parquet", predictions)


def _evaluate(args, wandb_module=None) -> None:  # type: ignore[no-untyped-def]  # noqa: C901
    if args.jobs < 1:
        raise SystemExit("jobs must be at least 1")
    if args.task == "near_binary" and len(args.regret_thresholds) != 1:
        raise SystemExit("near_binary requires exactly one regret threshold")
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise SystemExit("run-id must contain only letters, numbers, dot, underscore, or dash")
    plan = (
        read_experiment_manifest(args.experiment_manifest, stages=args.stages, task=args.task)
        if args.experiment_manifest
        else None
    )
    if plan is not None and plan.empty:
        raise SystemExit("experiment manifest has no rows for the requested stage and task")
    paths = (
        target_shard_paths(args.shards, args.labels)
        if args.task in {"near_binary", "three"}
        else sorted(args.shards.glob("*.parquet"))
    )
    if not paths:
        raise SystemExit("no feature shards")
    combinations = _combinations(args, plan)
    selected_prefixes = tuple(
        REPRESENTATION_PREFIXES[combination[2]] for combination in combinations
    )
    all_prefixes = tuple(REPRESENTATION_PREFIXES.values())
    available = pd.read_parquet(paths[0]).columns
    columns = [
        column
        for column in available
        if not column.startswith(all_prefixes) or column.startswith(selected_prefixes)
    ]
    features = pd.concat(
        (pd.read_parquet(path, columns=columns) for path in paths), ignore_index=True
    )
    cohort_audit = audit_holdout_cohort(features)
    features["target"] = (
        features["relative_regret"].le(args.regret_thresholds[0]).astype(int)
        if args.task == "near_binary"
        else map_cost_state_targets(features["cost_state"], args.task)
    )
    features = features.loc[features["target"].notna()].copy()
    features["target"] = features["target"].astype(int)
    candidates = pd.read_parquet(args.candidates)
    features = add_cycle_time_features(features, candidates)
    labels = (
        pd.read_parquet(
            args.labels, columns=["cycle_name", "camera_role", "image_time", "relative_regret"]
        )
        if args.task in {"near_binary", "three"}
        else None
    )
    if args.task == "near_binary":
        features, sensor_audit = add_causal_sensor_features(features, args.sensor_dir)
        sensor_audit["labeled_image_count"] = int(labels["relative_regret"].notna().sum())
        sensor_audit["feature_shard_shortfall"] = (
            sensor_audit["labeled_image_count"] - sensor_audit["model_input_image_count"]
        )
    else:
        sensor_audit = pd.DataFrame()
    label_balance = pd.read_csv(args.label_balance) if args.task == "binary" else None
    experiments = sorted(features["experiment_id"].unique())
    total = validate_formal_run_shape(
        task=args.task,
        has_manifest=plan is not None,
        combination_count=len(combinations),
        experiments=experiments,
    )
    config = {
        "version": "rgb-matrix-v1",
        "task": args.task,
        "stages": args.stages,
        "cv": "leave-one-experiment-out",
        "combination_count": len(combinations),
        "combinations": normalized_combinations(combinations),
        "task_count": total,
        "shards": str(args.shards),
        "candidates": str(args.candidates),
        "labels": str(args.labels),
        "sensor_dir": str(args.sensor_dir) if args.task == "near_binary" else None,
        "experiment_manifest": str(args.experiment_manifest) if args.experiment_manifest else None,
    }
    execution = {"jobs": args.jobs, "backend": args.backend}
    store = RunStore(args.output, run_id, config, execution)
    stages = combination_stages(plan) if plan is not None else {}
    completed_ids = store.completed_task_ids()
    wandb_run = None
    if args.wandb_project:
        if wandb_module is None:
            wandb_module = _safe_wandb("import", importlib.import_module, "wandb")
        if wandb_module is not None:
            wandb_run = _safe_wandb_method(
                wandb_module,
                "init",
                project=args.wandb_project,
                id=run_id,
                resume="allow",
                name=args.wandb_run_name,
                config={**config, **execution},
            )
        if wandb_run is not None:
            _safe_wandb_method(wandb_run, "define_metric", "task_step")
            _safe_wandb_method(wandb_run, "define_metric", "*", step_metric="task_step")

    def task_stream():  # type: ignore[no-untyped-def]
        for combination_index, combination in enumerate(combinations, start=1):
            camera, threshold, representation, model, modality = combination
            heldouts = [
                (heldout, stable_task_id(combination_index, heldout))
                for heldout in experiments
                if stable_task_id(combination_index, heldout) not in completed_ids
            ]
            if not heldouts:
                continue
            camera_rows = features.loc[features["camera_role"].isin(CAMERA_GROUPS[camera])]
            scoped = (
                retain_high_confidence_rows(camera_rows, threshold)
                if args.task == "binary"
                else camera_rows.copy()
            )
            values = build_modality_frames(scoped, representation)[modality]
            retained = len(scoped) / len(camera_rows)
            coverage = (
                high_confidence_coverage(label_balance, camera, threshold)
                if args.task == "binary"
                else (
                    len(camera_rows) / int(labels["relative_regret"].notna().sum())
                    if args.task == "near_binary"
                    else three_class_eligible_image_coverage(
                        labels, candidates, CAMERA_GROUPS[camera]
                    )
                )
            )
            metadata = {
                "stage": stages.get(combination, "custom"),
                "camera_group": camera,
                "regret_threshold": threshold,
                "representation": representation,
                "model": model,
                "modality": modality,
                "sample_retained_fraction": retained,
                "eligible_image_coverage": coverage,
                "combination_index": combination_index,
            }
            for heldout, task_id in heldouts:
                yield delayed(_worker)(
                    values,
                    heldout,
                    {**metadata, "task_id": task_id},
                    (0, 1) if args.task in {"binary", "near_binary"} else (0, 1, 2),
                    1 if args.jobs > 1 else -1,
                )

    started = perf_counter()
    limits = threadpool_limits(limits=1) if args.backend == "threading" else nullcontext()
    configuration = (
        parallel_config(backend="loky", inner_max_num_threads=1)
        if args.backend == "loky"
        else nullcontext()
    )
    try:
        with limits, configuration:
            completed = Parallel(
                n_jobs=args.jobs,
                backend=args.backend,
                return_as="generator_unordered",
                batch_size="auto",
                pre_dispatch="2*n_jobs",
            )(task_stream())
            for result, predicted in completed:
                for key in (*COMBINATION_COLUMNS, "task_id"):
                    predicted[key] = result[key]
                predicted["sample_retained_fraction"] = result["sample_retained_fraction"]
                predicted["eligible_image_coverage"] = result["eligible_image_coverage"]
                store.record(result, predicted)
                ledger = store.ledger()
                counts = ledger["status"].value_counts()
                done = len(ledger)
                rate = done / max(perf_counter() - started, 1e-9)
                progress = (
                    f"{done}/{total}, OK={counts.get('ok', 0)}, "
                    f"INVALID={counts.get('invalid', 0)}, FAILED={counts.get('failed', 0)}, "
                    f"latest={result['task_id']} heldout={result['heldout']}, "
                    f"rate={rate:.2f}/s, elapsed={perf_counter() - started:.1f}s"
                )
                print(progress, flush=True)
                if wandb_run is not None:
                    scores = score_rows(predicted) if result["status"] == "ok" else {}
                    _safe_wandb_method(
                        wandb_run,
                        "log",
                        {
                            "task_step": done,
                            "completed": done,
                            "failed": int(counts.get("failed", 0)),
                            "success_rate": float(
                                (counts.get("ok", 0) + counts.get("invalid", 0)) / done
                            ),
                            "rate": rate,
                            "latest_elapsed_seconds": result["elapsed"],
                            **{
                                key: result[key]
                                for key in (
                                    *COMBINATION_COLUMNS,
                                    "heldout",
                                    "status",
                                    "warning_count",
                                )
                            },
                            **scores,
                        },
                    )
        ledger = store.ledger()
        prediction_frames = [
            pd.read_parquet(store.predictions_dir / f"{task_id}.parquet")
            for task_id in ledger["task_id"].astype(str)
        ]
        predictions = pd.concat(prediction_frames, ignore_index=True)
        validate_completed_run(
            ledger, predictions, expected_tasks=total, folds_per_combination=len(experiments)
        )
        if args.task == "near_binary":
            if set(ledger["modality"]) != {"rgb", "rgb_state", "rgb_all_sensor"}:
                raise ValueError("near-binary run requires exactly three sensor-fusion inputs")
            if (
                predictions.groupby(["modality", "held_out_experiment"])["target"]
                .nunique()
                .ne(2)
                .any()
            ):
                raise ValueError("each near-binary fold must contain both classes")
        _write_summaries(store, predictions, ledger, plan, cohort_audit, sensor_audit)
        store.mark_complete()
    except Exception as error:
        store.mark_failed(f"{type(error).__name__}: {error}")
        raise
    finally:
        if wandb_run is not None:
            _safe_wandb_method(wandb_run, "finish")


def main(wandb_module=None) -> None:  # type: ignore[no-untyped-def]
    _evaluate(_parse_args(), wandb_module)


if __name__ == "__main__":
    main()
