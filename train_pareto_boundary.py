"""Prepare shared causal data, fit fold teachers, and compare direct Pareto decisions."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from image_models.pareto_learning import METHOD_NAMES

METHODS = {
    "baseline": (False, False, False),
    "economic": (True, False, False),
    "relation": (False, True, False),
    "combined": (True, True, False),
    "nonvisual": (True, False, True),
}
STOP_METHODS = ("s0", "s1", "s2", "s3", "s4", "n2")
PERFORMANCE_METHODS = (*STOP_METHODS, "s1_neural")
CROSS_INPUT_METHODS = (
    "ridge_head_ridge_ch", "ridge_head_neural_ch",
    "neural_head_ridge_ch", "neural_head_neural_ch",
)
PROBE_METHODS = (
    "latent_ridge_ch_linear", "latent_ridge_ch_mlp",
    "latent_raw_ridge_ch_mlp", "raw_ridge_ch_mlp",
)
STATE_TRANSFER_METHODS = (
    "d32_ridge_ch_mlp", "t32_ridge_ch_mlp", "raw_ridge_ch_mlp",
)
CURRENT_ECONOMIC_INPUTS = tuple(
    f"{objective}_{suffix}"
    for objective in ("c", "h")
    for suffix in ("current", "current_missing", "valid_count", "age_seconds")
)
PARETO_STATUS_FIELDS = {
    False: ("status", "pareto_knee_status", "rgb_knee_coverage_status"),
    True: (
        "status",
        "pareto_extrapolated_knee_status",
        "rgb_extrapolated_knee_coverage_status",
    ),
}


def fold_exclusions(experiments: list[str]) -> dict[str, str]:
    """Rotate inner validation by date order without inspecting targets."""
    ordered = sorted(set(experiments))
    return {test: ordered[(index + 1) % len(ordered)] for index, test in enumerate(ordered)}


def save_settings(path: Path, settings: dict) -> None:
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("settings changed; use a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")


def log_wandb(project: str | None, name: str, settings: dict, **metrics: float) -> None:
    if not project:
        return
    import wandb

    with wandb.init(project=project, name=name, config=settings) as run:
        run.log(metrics)


def require_matching_extrapolation_setting(data: Path, allow_extrapolation: bool) -> None:
    settings = json.loads((data / "settings.json").read_text())
    if bool(settings.get("allow_extrapolation", False)) != allow_extrapolation:
        raise ValueError(
            "extrapolation setting differs from prepared data; use a new data directory"
        )


def native_frame_cycles(catalog: pd.DataFrame, images: pd.DataFrame) -> set[str]:
    names = ["cycle_name", "heating_start", "defrost_preparation_start"]
    rows = images.loc[images["camera_role"].eq("front"), ["cycle_name", "image_time"]].merge(
        catalog[names], on="cycle_name", validate="many_to_one"
    )
    for column in ("image_time", "heating_start", "defrost_preparation_start"):
        rows[column] = pd.to_datetime(rows[column], format="mixed")
    native = rows.image_time.ge(rows.heating_start) & rows.image_time.lt(
        rows.defrost_preparation_start
    )
    return set(rows.loc[native, "cycle_name"])


def build_parser() -> argparse.ArgumentParser:
    method_help = "\n".join(
        f"  {method}: {recipe.label}" for method, recipe in METHOD_NAMES.items()
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"Stopping methods:\n{method_help}",
    )
    parser.add_argument(
        "--action",
        choices=(
            "prepare", "fit", "train", "represent", "heads", "neural",
            "evaluate", "cross-input", "vcnet", "probe", "state-transfer", "refit",
            "overview", "screen",
            "audit", "develop", "compare-development", "refit-development",
            "evaluate-frozen-development", "compare-frozen-policies",
            "compare-stopping-losses", "render-cycles",
        ),
        default=None,
    )
    parser.add_argument(
        "--task", choices=("pareto", "relative-cop-regression", "effective-cop-binary", "cop-classification-regression", "cop-classification"),
        default="pareto",
    )
    parser.add_argument("--regression-architecture",
                        choices=("r-cop32", "d32", "pinn4soh", "state-consistency"),
                        default="r-cop32", help="Architecture for the shared relative-COP task")
    parser.add_argument("--rgb", choices=("off", "on"), default="off")
    parser.add_argument("--rgb-projection", type=int, choices=(0, 32), default=0)
    parser.add_argument("--require-rgb-input", action="store_true", help="Match sensor-only input times to the same RGB availability")
    parser.add_argument("--screen-duration-seconds", type=float, default=60.)
    parser.add_argument("--evaluation-threshold", type=float, default=None, help="fixed probability threshold for frozen classification replay")
    parser.add_argument("--evaluation-cohort", choices=("valid", "rgb-valid"), default="valid")
    parser.add_argument("--classification-label", choices=("near-optimal", "after-optimum"), default="near-optimal")
    parser.add_argument(
        "--development-mechanism", choices=("static", "delta"), default="static"
    )
    parser.add_argument("--require-visual-history", action="store_true")
    parser.add_argument(
        "--development-loss", choices=("bce", "label-smoothing"), default="bce"
    )
    parser.add_argument(
        "--stopping-architectures", nargs="+",
        choices=("r_sensor", "r_sensor_rgb", "chen_rgb", "new_sensor", "new_rgb_difference"),
        default=["r_sensor", "r_sensor_rgb", "chen_rgb", "new_sensor", "new_rgb_difference"],
    )
    parser.add_argument(
        "--stopping-losses", nargs="+", choices=("after_optimum", "cop_stopping"),
        default=["after_optimum", "cop_stopping"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--near-optimal-epsilon", type=float, default=.01)
    parser.add_argument("--regression-weight", type=float, default=1.)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pinn-alpha", type=float, default=1.)
    parser.add_argument("--pinn-beta", type=float, default=1.)
    parser.add_argument("--peak-weighted", action="store_true")
    parser.add_argument("--trigger-threshold", type=float, default=.99)
    parser.add_argument("--processing-seconds", type=int, default=30)
    parser.add_argument("--regression-suite", action="store_true",
                        help="Run four shared recipes, then weight the validation winner")
    parser.add_argument("--decision-run", type=Path,
                        default=Path("output/defrost_decisions/effective_cop_tref"))
    parser.add_argument("--event-run", type=Path,
                        default=Path("output/defrost_event_models/heating_start_zero"))
    parser.add_argument("--figure-output", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--data", type=Path, default=Path("output/image_models/_cache/pareto_boundary_v1")
    )
    parser.add_argument(
        "--rgb-cache",
        type=Path,
        default=Path("output/image_models/_cache/dinov2_vits14_r256_c224_front_v1/cycles"),
    )
    parser.add_argument("--output", type=Path, default=Path("output/test/pareto_boundary/current"))
    parser.add_argument(
        "--representations", type=Path,
        default=Path("output/image_models/_cache/outcome_pareto/representations"),
    )
    parser.add_argument(
        "--reference-run", type=Path,
        default=Path("output/test/pareto_boundary_outcome_v1"),
    )
    parser.add_argument(
        "--runs", type=Path, nargs="+",
        help=("Saved runs; cross-input expects Ridge then Neural. Probe optionally "
              "accepts the time-aware outcome root for calibration. Overview expects "
              "policy, probe, neural, cross-input, outcome, state, transfer. Frozen "
              "development evaluation expects Delta RGB then History Sensor "
              "development runs."),
    )
    parser.add_argument("--method", choices=METHODS, default="baseline")
    parser.add_argument("--selected-method", choices=STATE_TRANSFER_METHODS)
    parser.add_argument(
        "--event-head", nargs="+",
        choices=(
            "multimodal_time_linear", "multimodal_time_linear_z16",
            "multimodal_time_linear_z64", "multimodal_time_linear_z32_curvature",
            "multimodal_time_linear_z64_curvature", "multimodal_time_varying",
            "multimodal_time_varying_regularized",
        ),
        default=[
            "multimodal_time_linear", "multimodal_time_varying",
            "multimodal_time_varying_regularized",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maximum-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--heldout-experiment")
    parser.add_argument("--wandb-project")
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--diagnose-state", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def prepare(args: argparse.Namespace) -> None:
    from dataset_tools import DatasetLoader
    from defrost_decision.candidate_times import clean_anchor_cycles, metadata_eligible_cycles
    from defrost_event_models.training_data import build_defrost_event_training_table
    from image_models.pareto_data import build_cycle_base_table

    loader = DatasetLoader(args.dataset)
    metadata_cycles = metadata_eligible_cycles(loader, None, None)
    cycles, _ = clean_anchor_cycles(loader, metadata_cycles, explicit=False)
    sources = [args.dataset / name for name in ("cycle_catalog.json", "image_metadata.parquet")]
    save_settings(
        args.data / "settings.json",
        {
            "dataset": str(args.dataset.resolve()),
            "data_schema": "pareto_boundary_online_quality_v2",
            "sources": {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in sources},
            "sensor_window_minutes": 5,
            "candidate_step_seconds": 10,
            "rgb_policy": "latest_same_cycle_front_max_age_45s",
            "allow_extrapolation": args.allow_extrapolation,
            "cycles": cycles,
        },
    )
    events = args.data / "events.parquet"
    coverage = loader.list_cycles()[["cycle_name", "experiment_id", "status"]].copy()
    coverage["preparation_status"] = "catalog_or_boundary_ineligible"
    coverage.loc[coverage.cycle_name.isin(metadata_cycles), "preparation_status"] = (
        "anchor_ineligible"
    )
    coverage.loc[coverage.cycle_name.isin(cycles), "preparation_status"] = "eligible"
    native = native_frame_cycles(loader.list_cycles(), loader.load_image_metadata())
    coverage.loc[
        coverage.cycle_name.isin(cycles) & ~coverage.cycle_name.isin(native), "preparation_status"
    ] = "no_native_preparation_prefix_frames"
    cycles = [cycle for cycle in cycles if cycle in native]
    coverage.to_csv(args.data / "cycle_coverage.csv", index=False)
    if not events.exists():
        build_defrost_event_training_table(loader).to_parquet(events, index=False)
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        results = Parallel(return_as="generator_unordered")(
            delayed(build_cycle_base_table)(
                loader,
                cycle,
                cache_dir=args.data.parent
                / (
                    "pareto_measured_reconstructed_stat6_quality_v2"
                    if args.allow_extrapolation
                    else "pareto_measured_stat6_quality_v2"
                ),
                allow_measurement_reconstruction=args.allow_extrapolation,
            )
            for cycle in cycles
        )
        tables = []
        for table in results:
            tables.append(table)
            print(f"Measured/sensor cache: {len(tables)}/{len(cycles)}", flush=True)
    base = pd.concat(tables, ignore_index=True)
    base.to_parquet(args.data / "base.parquet", index=False)
    folds = fold_exclusions(base["experiment_id"].astype(str).unique().tolist())
    save_settings(args.data / "folds.json", folds)


def _fit_teacher(
    data: Path, excluded: tuple[str, ...], *, allow_model_extrapolation: bool = False
) -> None:
    from image_models.pareto_data import apply_fold_teacher, fit_fold_parameters

    stem = "__".join(excluded)
    path = data / "teachers" / f"{stem}.parquet"
    if path.exists():
        return
    base = pd.read_parquet(data / "base.parquet")
    parameter_path = path.with_suffix(".json")
    if parameter_path.exists():
        parameters = json.loads(parameter_path.read_text())
    else:
        parameters = fit_fold_parameters(pd.read_parquet(data / "events.parquet"), excluded)
        parameter_path.parent.mkdir(parents=True, exist_ok=True)
        parameter_path.write_text(json.dumps(parameters))
    tables = [
        apply_fold_teacher(
            cycle, parameters, allow_model_extrapolation=allow_model_extrapolation
        )
        for _, cycle in base.groupby("cycle_name")
    ]
    result = pd.concat(tables, ignore_index=True)
    # Only fold-dependent quantities are cached here; RGB and sensors stay shared.
    fields = [
        c
        for c in result
        if c not in base
        or c
        in {
            "row_id",
            "pareto_selection_score",
            "relation_branch",
            "relation_support_run",
            "is_knee",
            "target",
            "teacher_time",
            "economic_c",
            "economic_h",
        }
    ]
    temporary = path.with_suffix(".tmp")
    result[fields].to_parquet(temporary, index=False)
    temporary.replace(path)
    print(f"Teacher complete; excluded experiments: {excluded}", flush=True)


def derive_pareto_cycle_statuses(rows: pd.DataFrame) -> pd.DataFrame:
    """Classify knee existence and native-RGB bracketing for each cycle."""
    statuses = []
    for cycle_name, cycle in rows.groupby("cycle_name", sort=False):
        knees = pd.to_datetime(cycle["teacher_time"], errors="coerce").dropna().unique()
        if len(knees) > 1:
            raise ValueError(f"{cycle_name}: fold teacher contains multiple knee times")
        knee_valid = len(knees) == 1
        times = pd.to_datetime(
            cycle.loc[cycle["is_frame"], "candidate_defrost_time"], errors="coerce"
        ).dropna()
        covered = knee_valid and not times.empty and times.min() <= knees[0] <= times.max()
        statuses.append(
            {
                "cycle_name": str(cycle_name),
                "pareto_knee_status": "valid" if knee_valid else "invalid",
                "rgb_knee_coverage_status": "valid" if covered else "invalid",
            }
        )
    return pd.DataFrame(statuses)


def update_pareto_cycle_statuses(
    dataset: Path, data: Path, *, allow_extrapolation: bool = False
) -> None:
    """Write out-of-sample Pareto readiness beside the Dataset cycle status."""
    from dataset_tools.cycle_metadata import read_catalog, write_catalog

    catalog = read_catalog(dataset)
    base = pd.read_parquet(data / "base.parquet")
    statuses = []
    for experiment_id in json.loads((data / "folds.json").read_text()):
        teacher = pd.read_parquet(data / "teachers" / f"{experiment_id}.parquet")
        heldout = base.loc[base["experiment_id"].astype(str).eq(str(experiment_id))]
        rows = heldout.merge(
            teacher[["row_id", "teacher_time"]], on="row_id", validate="one_to_one"
        )
        statuses.append(derive_pareto_cycle_statuses(rows))
    by_cycle = (
        pd.concat(statuses, ignore_index=True).set_index("cycle_name").to_dict("index")
        if statuses
        else {}
    )
    positioned = []
    target_fields = PARETO_STATUS_FIELDS[allow_extrapolation][1:]
    source_fields = ("pareto_knee_status", "rgb_knee_coverage_status")
    for record in catalog["cycles"]:
        values = by_cycle.get(str(record.get("cycle_name")), {})
        statuses = {
            target: values.get(source, "invalid")
            for target, source in zip(target_fields, source_fields, strict=True)
        }
        ordered = {}
        for field, value in record.items():
            if field not in statuses:
                ordered[field] = value
            if field == "status_reason":
                ordered.update(statuses)
        if not statuses.keys() <= ordered.keys():
            ordered.update(statuses)
        positioned.append(ordered)
    catalog["cycles"] = positioned
    write_catalog(dataset, catalog)


def pareto_cv_ready_cycles(
    catalog: dict, *, allow_extrapolation: bool = False
) -> set[str]:
    """Return cycles admitted by the three independent validity gates."""
    return {
        str(record["cycle_name"])
        for record in catalog["cycles"]
        if all(
            record.get(field, "invalid") == "valid"
            for field in PARETO_STATUS_FIELDS[allow_extrapolation]
        )
    }


def fit(args: argparse.Namespace) -> None:
    require_matching_extrapolation_setting(args.data, args.allow_extrapolation)
    folds = json.loads((args.data / "folds.json").read_text())
    exclusions = sorted(
        {
            tuple(sorted(group))
            for test, inner in folds.items()
            for group in ((test,), (test, inner))
        }
    )
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        Parallel()(
            delayed(_fit_teacher)(
                args.data,
                excluded,
                allow_model_extrapolation=args.allow_extrapolation,
            )
            for excluded in exclusions
        )
    update_pareto_cycle_statuses(
        args.dataset, args.data, allow_extrapolation=args.allow_extrapolation
    )


def load_fold_rows(
    data: Path, rgb_cache: Path, excluded: tuple[str, ...], *, include_rgb: bool = True,
    included_experiments: tuple[str, ...] | None = None,
    include_teacher: bool = True,
) -> pd.DataFrame:
    from image_models.image_features import load_dinov2_feature_cache
    base = pd.read_parquet(data / "base.parquet")
    stem = "__".join(sorted(excluded))
    if included_experiments is not None:
        keep = base["experiment_id"].astype(str).isin(included_experiments)
        base = base.loc[keep]
    rows = base
    if include_teacher:
        teacher = pd.read_parquet(data / "teachers" / f"{stem}.parquet")
        teacher = teacher.loc[teacher["row_id"].isin(base["row_id"])]
        rows = base.drop(
            columns=base.columns.intersection(teacher.columns).difference(["row_id"])
        ).merge(teacher, on="row_id", validate="one_to_one")
    if include_teacher:
        online_path = data / "online" / f"{stem}.parquet"
        if online_path.exists():
            online = pd.read_parquet(online_path)
            online = online.loc[online["row_id"].isin(rows["row_id"])]
            overlap = online.columns.intersection(rows.columns).difference(["row_id"])
            rows = rows.merge(
                online.drop(columns=overlap), on="row_id", validate="one_to_one"
            )
        else:
            from image_models.pareto_data import add_online_economic_features

            existing = set(rows)
            rows = add_online_economic_features(rows)
            online = rows[[
                "row_id", *[
                    name for name in rows if name.startswith("online_") and name not in existing
                ],
            ]]
            online_path.parent.mkdir(parents=True, exist_ok=True)
            online.to_parquet(online_path, index=False)
    if not include_rgb:
        return rows
    keys = ["cycle_name", "camera_role", "file_name"]
    images = rows.loc[rows["file_name"].notna(), keys].drop_duplicates()
    features = load_dinov2_feature_cache(images, rgb_cache, "dinov2")
    rows = rows.merge(features, on=keys, how="left", validate="many_to_one")
    missing = rows["dinov2_000"].isna()
    if (missing & rows["is_frame"]).any():
        raise ValueError(f"{int(missing.sum())} rows lack RGB; finish cache/coverage review first")
    return rows


def fold_ready_cycles(rows: pd.DataFrame) -> set[str]:
    statuses = derive_pareto_cycle_statuses(rows)
    valid = statuses.pareto_knee_status.eq("valid") & statuses.rgb_knee_coverage_status.eq("valid")
    return set(statuses.loc[valid, "cycle_name"])


def _represent_fold(args: argparse.Namespace, test: str, inner: str) -> dict:
    import torch

    from image_models.outcome_representation import (
        encode_outcome_rows,
        fit_outcome_representation,
        outcome_event_rows,
        predict_outcomes,
    )

    folder = args.representations / "folds" / test
    checkpoint_path = folder / "checkpoint.pkl"
    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as stream:
            return pickle.load(stream)  # noqa: S301 - local run-owned checkpoint
    torch.set_num_threads(1)
    outer = load_fold_rows(args.data, args.rgb_cache, (test,))
    nested = load_fold_rows(args.data, args.rgb_cache, (test, inner))
    events = pd.read_parquet(args.data / "events.parquet")
    outer_events = outcome_event_rows(outer, events)
    nested_events = outcome_event_rows(nested, events)
    inner_train = nested_events.loc[
        nested_events.experiment_id.astype(str).ne(test)
        & nested_events.experiment_id.astype(str).ne(inner)
    ]
    inner_validation = nested_events.loc[nested_events.experiment_id.astype(str).eq(inner)]
    outer_train = outer_events.loc[outer_events.experiment_id.astype(str).ne(test)]
    outer_test = outer_events.loc[outer_events.experiment_id.astype(str).eq(test)]
    fitted = {}
    for name, use_rgb in (("visual", True), ("nonvisual", False)):
        fitted[name] = fit_outcome_representation(
            inner_train, inner_validation, outer_train, use_rgb=use_rgb, seed=args.seed,
            maximum_epochs=args.maximum_epochs, patience=args.patience,
        )
    inner_latent = encode_outcome_rows(nested, fitted["visual"]["inner_checkpoint"])
    inner_nonvisual = encode_outcome_rows(nested, fitted["nonvisual"]["inner_checkpoint"])
    outer_latent = encode_outcome_rows(outer, fitted["visual"]["checkpoint"])
    outer_nonvisual = encode_outcome_rows(outer, fitted["nonvisual"]["checkpoint"])
    for table, nonvisual in ((inner_latent, inner_nonvisual), (outer_latent, outer_nonvisual)):
        table[[name.replace("z_", "nz_") for name in nonvisual if name.startswith("z_")]] = (
            nonvisual.filter(like="z_").to_numpy()
        )
    predictions, losses = [], []
    for name, result in fitted.items():
        prediction = predict_outcomes(outer_test, result["checkpoint"])
        prediction["representation"] = name
        predictions.append(prediction)
        loss = result["losses"].copy()
        loss["representation"] = name
        losses.append(loss)
    result = {
        "checkpoints": {name: value["checkpoint"] for name, value in fitted.items()},
        "outcome_predictions": pd.concat(predictions, ignore_index=True),
        "losses": pd.concat(losses, ignore_index=True),
        "heldout_experiment": test,
        "inner_validation_experiment": inner,
    }
    folder.mkdir(parents=True, exist_ok=True)
    inner_latent.to_parquet(folder / "inner_latents.parquet", index=False)
    outer_latent.to_parquet(folder / "outer_latents.parquet", index=False)
    temporary = checkpoint_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(result, stream)
    temporary.replace(checkpoint_path)
    print(f"representation: completed {test}", flush=True)
    return result


def represent(args: argparse.Namespace) -> None:
    require_matching_extrapolation_setting(args.data, args.allow_extrapolation)
    folds = json.loads((args.data / "folds.json").read_text())
    if args.heldout_experiment:
        folds = {args.heldout_experiment: folds[args.heldout_experiment]}
    settings = {
        "data": str(args.data), "seed": args.seed, "maximum_epochs": args.maximum_epochs,
        "patience": args.patience, "folds": folds,
    }
    save_settings(args.representations / "settings.json", settings)
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        results = Parallel()(delayed(_represent_fold)(args, test, inner)
                             for test, inner in folds.items())
    pd.concat([result["outcome_predictions"] for result in results]).to_csv(
        args.representations / "outcome_predictions.csv", index=False
    )
    pd.concat([result["losses"].assign(
        heldout_experiment=result["heldout_experiment"]
    ) for result in results]).to_csv(args.representations / "losses.csv", index=False)
    log_wandb(
        args.wandb_project, args.representations.name, settings, completed_folds=len(results)
    )


def _vcnet_event_starts(args: argparse.Namespace) -> pd.DataFrame:
    """Build one causal preparation-start row per complete observed event."""
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "event_starts.parquet"
    if path.exists():
        return pd.read_parquet(path)
    reference = args.reference_run / "event_starts.parquet"
    if args.diagnose_state and reference.exists():
        return pd.read_parquet(reference)
    from dataset_tools import DatasetLoader
    from defrost_event_models.ridge_models import select_events_complete_for_all_outcomes
    from defrost_event_models.training_data import build_defrost_event_training_table
    from image_models.image_features import load_dinov2_feature_cache
    from image_models.outcome_representation import RGB, outcome_event_rows
    from image_models.pareto_data import build_cycle_base_table

    loader = DatasetLoader(args.dataset)
    events = build_defrost_event_training_table(loader)
    events.to_parquet(args.output / "events.parquet", index=False)
    complete = select_events_complete_for_all_outcomes(events)
    complete_cycles = complete["event_id"].astype(str).tolist()
    base = pd.read_parquet(args.data / "base.parquet")
    available = base.loc[
        pd.to_datetime(base["candidate_defrost_time"]).eq(
            pd.to_datetime(base["observed_defrost_preparation_start"])
        ) & base["cycle_name"].isin(complete_cycles)
    ]
    tables = [available]
    coverage = {cycle: "reused_pareto_base" for cycle in available["cycle_name"]}
    missing = [cycle for cycle in complete_cycles if cycle not in coverage]
    for index, cycle in enumerate(missing, 1):
        try:
            table = build_cycle_base_table(
                loader, cycle, cache_dir=args.output / "event_start_cache",
                allow_measurement_reconstruction=args.allow_extrapolation,
                event_start_only=True,
            )
            tables.append(table)
            coverage[cycle] = "built_event_start"
        except (FileNotFoundError, KeyError, ValueError) as error:
            coverage[cycle] = f"unavailable:{type(error).__name__}:{error}"
        print(f"Event-start input: {index}/{len(missing)}", flush=True)
    starts = outcome_event_rows(pd.concat(tables, ignore_index=True), events)
    image_keys = ["cycle_name", "camera_role", "file_name"]
    images = starts.loc[starts["file_name"].notna(), image_keys].drop_duplicates()
    if not images.empty:
        features = load_dinov2_feature_cache(images, args.rgb_cache, "dinov2")
        starts = starts.merge(features, on=image_keys, how="left", validate="many_to_one")
    for column in RGB:
        if column not in starts:
            starts[column] = np.nan
    starts.to_parquet(path, index=False)
    pd.DataFrame({
        "cycle_name": complete_cycles,
        "experiment_id": complete["experiment_id"].astype(str).tolist(),
        "event_input_status": [coverage.get(cycle, "unavailable") for cycle in complete_cycles],
        "event_input_available": [cycle in set(starts.cycle_name) for cycle in complete_cycles],
        "rgb_available": [
            bool(starts.loc[starts.cycle_name.eq(cycle), "file_name"].notna().any())
            for cycle in complete_cycles
        ],
    }).to_csv(args.output / "event_coverage.csv", index=False)
    return starts


def _outcome_replays(
    rows: pd.DataFrame, latents: pd.DataFrame, checkpoint: dict
) -> pd.DataFrame:
    """Separate candidate variation carried by heating time and by observed state."""
    from image_models.outcome_representation import TIME_COLUMN, predict_outcomes_from_latent

    meta = rows[[
        "row_id", "cycle_name", "experiment_id", "candidate_defrost_time",
        "observed_defrost_preparation_start", "is_teacher_candidate",
    ]].merge(latents, on="row_id", validate="one_to_one")
    z_columns = sorted(name for name in latents if name.startswith("z_"))
    tables = []
    for _, cycle in meta.groupby("cycle_name", sort=False):
        anchor = cycle.loc[
            pd.to_datetime(cycle.candidate_defrost_time).eq(
                pd.to_datetime(cycle.observed_defrost_preparation_start)
            )
        ]
        if len(anchor) != 1:
            continue
        for replay, values in (
            ("observed_state_and_time", cycle.copy()),
            ("fixed_state_varying_time", cycle.assign(
                **{name: anchor[name].iloc[0] for name in z_columns}
            )),
            ("varying_state_fixed_time", cycle.assign(
                **{TIME_COLUMN: anchor[TIME_COLUMN].iloc[0]}
            )),
        ):
            prediction = predict_outcomes_from_latent(values, checkpoint)
            tables.append(values[[
                "row_id", "cycle_name", "experiment_id", "candidate_defrost_time",
                "is_teacher_candidate",
            ]].merge(prediction, on="row_id", validate="one_to_one").assign(replay=replay))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _vcnet_fold(
    args: argparse.Namespace, events: pd.DataFrame, event_head: str,
    test: str, inner: str, run: Path,
) -> dict:
    import torch

    from image_models.outcome_representation import (
        CURVATURE_HEADS,
        encode_outcome_rows,
        event_nearest_neighbor_diagnostics,
        fit_latent_scale,
        fit_outcome_representation,
        predict_outcomes,
        predict_outcomes_from_latent,
        trajectory_roughness,
    )
    from image_models.pareto_data import apply_neural_pareto

    folder = run / "folds" / test
    artifact_path = folder / "checkpoint.pkl"
    if artifact_path.exists():
        with artifact_path.open("rb") as stream:
            artifact = pickle.load(stream)  # noqa: S301 - local run-owned checkpoint
        for name, filename in (
            ("candidates", "candidates.parquet"),
            ("replays", "trajectory_predictions.parquet"),
            ("roughness", "roughness.csv"),
            ("nearest_neighbors", "nearest_neighbors.csv"),
        ):
            path = folder / filename
            if name not in artifact:
                artifact[name] = (
                    pd.read_parquet(path) if path.suffix == ".parquet" and path.exists()
                    else pd.read_csv(path) if path.exists() else pd.DataFrame()
                )
        return artifact
    torch.set_num_threads(1)
    experiment = events["experiment_id"].astype(str)
    inner_train = events.loc[experiment.ne(test) & experiment.ne(inner)]
    inner_validation = events.loc[experiment.eq(inner)]
    outer_train = events.loc[experiment.ne(test)]
    outer_test = events.loc[experiment.eq(test)]
    inner_temporal = outer_temporal = None
    if event_head in CURVATURE_HEADS:
        inner_temporal = load_fold_rows(
            args.data, args.rgb_cache, (test, inner), include_teacher=False
        )
        inner_temporal = inner_temporal.loc[
            ~inner_temporal.experiment_id.astype(str).isin((test, inner))
        ]
    if event_head in CURVATURE_HEADS or args.diagnose_state:
        outer_temporal = load_fold_rows(
            args.data, args.rgb_cache, (test,), include_teacher=False
        )
        outer_temporal = outer_temporal.loc[
            outer_temporal.experiment_id.astype(str).ne(test)
        ]
    fitted = fit_outcome_representation(
        inner_train, inner_validation, outer_train, use_rgb=True,
        event_head=event_head, seed=args.seed, maximum_epochs=args.maximum_epochs,
        patience=args.patience, inner_temporal=inner_temporal,
        outer_temporal=outer_temporal,
    )
    checkpoint = fitted["checkpoint"]
    event_latents = encode_outcome_rows(outer_train, checkpoint)
    checkpoint["event_latent_scale"] = fit_latent_scale(event_latents)
    if outer_temporal is not None:
        trajectory_latents = encode_outcome_rows(outer_temporal, checkpoint)
        checkpoint["trajectory_latent_scale"] = fit_latent_scale(
            trajectory_latents, outer_temporal["cycle_name"].reset_index(drop=True)
        )
    prediction = predict_outcomes(outer_test, checkpoint).assign(
        representation=event_head, seed=args.seed, heldout_experiment=test
    )
    for index, target in enumerate(checkpoint["target_scaler"].feature_names_in_):
        prediction[f"standardized_absolute_error_{target}"] = (
            prediction[f"predicted_{target}"] - prediction[target]
        ).abs() / checkpoint["target_scaler"].scale_[index]
    losses = fitted["losses"].assign(
        representation=event_head, seed=args.seed, heldout_experiment=test
    )
    nearest_neighbors = (
        event_nearest_neighbor_diagnostics(outer_train, outer_test, checkpoint).assign(
            representation=event_head, seed=args.seed, heldout_experiment=test
        ) if args.diagnose_state else pd.DataFrame()
    )
    candidate_folds = json.loads((args.data / "folds.json").read_text())
    candidates = replays = roughness = pd.DataFrame()
    if test in candidate_folds:
        rows = load_fold_rows(
            args.data, args.rgb_cache, (test,), included_experiments=(test,)
        )
        latents = encode_outcome_rows(rows, checkpoint)
        candidate_prediction = predict_outcomes_from_latent(latents, checkpoint)
        candidates = apply_neural_pareto(rows, candidate_prediction).assign(
            representation=event_head, seed=args.seed, heldout_experiment=test
        )
        replays = _outcome_replays(rows, latents, checkpoint).assign(
            representation=event_head, seed=args.seed, heldout_experiment=test
        )
        if args.diagnose_state:
            roughness = trajectory_roughness(
                rows, latents, candidate_prediction,
                checkpoint["trajectory_latent_scale"], checkpoint["target_scaler"].scale_,
            ).assign(representation=event_head, seed=args.seed,
                     heldout_experiment=test)
        folder.mkdir(parents=True, exist_ok=True)
        latents.to_parquet(folder / "candidate_latents.parquet", index=False)
    artifact = {
        "checkpoint": checkpoint, "inner_checkpoint": fitted["inner_checkpoint"],
        "outcome_predictions": prediction, "losses": losses,
        "candidates": candidates, "replays": replays,
        "roughness": roughness, "nearest_neighbors": nearest_neighbors,
        "heldout_experiment": test, "inner_validation_experiment": inner,
    }
    folder.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(folder / "candidates.parquet", index=False)
    replays.to_parquet(folder / "trajectory_predictions.parquet", index=False)
    if not roughness.empty:
        roughness.to_csv(folder / "roughness.csv", index=False)
    if not nearest_neighbors.empty:
        nearest_neighbors.to_csv(folder / "nearest_neighbors.csv", index=False)
    temporary = artifact_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(
            {name: value for name, value in artifact.items()
             if name not in {"candidates", "replays", "roughness", "nearest_neighbors"}},
            stream,
        )
    temporary.replace(artifact_path)
    print(f"VCNet {event_head}: completed {test}", flush=True)
    return artifact


def _diagnose_d32_fold(
    args: argparse.Namespace, events: pd.DataFrame, test: str, seed: int
) -> dict[str, pd.DataFrame]:
    """Evaluate one historical D32 fold without changing its checkpoint."""
    from image_models.outcome_representation import (
        TIME_COLUMN,
        encode_outcome_rows,
        event_nearest_neighbor_diagnostics,
        fit_latent_scale,
        local_pathway_sensitivity,
        predict_outcomes_from_latent,
        trajectory_roughness,
    )

    destination = (
        args.output / "reference_diagnostics" / "multimodal_time_linear"
        / f"seed_{seed}" / "folds" / test
    )
    paths = {
        "event_errors": destination / "event_errors.csv",
        "roughness": destination / "roughness.csv",
        "pathways": destination / "pathways.csv",
        "nearest_neighbors": destination / "nearest_neighbors.csv",
    }
    complete = destination / "complete.json"
    if complete.exists():
        return {
            name: pd.read_csv(path) if path.exists() else pd.DataFrame()
            for name, path in paths.items()
        }
    source = (
        args.reference_run / "multimodal_time_linear" / f"seed_{seed}"
        / "folds" / test / "checkpoint.pkl"
    )
    if not source.exists():
        raise FileNotFoundError(f"missing frozen D32 checkpoint: {source}")
    with source.open("rb") as stream:
        artifact = pickle.load(stream)  # noqa: S301 - local frozen research artifact
    checkpoint = artifact["checkpoint"]
    experiment = events["experiment_id"].astype(str)
    outer_train = events.loc[experiment.ne(test)]
    outer_test = events.loc[experiment.eq(test)]
    event_scale = fit_latent_scale(encode_outcome_rows(outer_train, checkpoint))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "event_latent_scale.json").write_text(json.dumps(event_scale) + "\n")
    nearest = event_nearest_neighbor_diagnostics(
        outer_train, outer_test, checkpoint
    ).assign(
        representation="multimodal_time_linear", seed=seed,
        heldout_experiment=test,
    )
    event_errors = artifact["outcome_predictions"].copy()
    for index, target in enumerate(checkpoint["target_scaler"].feature_names_in_):
        event_errors[f"standardized_absolute_error_{target}"] = (
            event_errors[f"predicted_{target}"] - event_errors[target]
        ).abs() / checkpoint["target_scaler"].scale_[index]
    candidates = artifact.get("candidates", pd.DataFrame())
    latent_path = source.parent / "candidate_latents.parquet"
    if candidates.empty or not latent_path.exists():
        roughness = pathways = pd.DataFrame()
    else:
        outer_rows = load_fold_rows(
            args.data, args.rgb_cache, (test,), include_teacher=False
        )
        outer_rows = outer_rows.loc[outer_rows.experiment_id.astype(str).ne(test)]
        training_latents = encode_outcome_rows(outer_rows, checkpoint)
        trajectory_scale = fit_latent_scale(
            training_latents, outer_rows["cycle_name"].reset_index(drop=True)
        )
        (destination / "trajectory_latent_scale.json").write_text(
            json.dumps(trajectory_scale) + "\n"
        )
        latents = pd.read_parquet(latent_path)
        prediction = predict_outcomes_from_latent(latents, checkpoint)
        roughness = trajectory_roughness(
            candidates, latents, prediction, trajectory_scale,
            checkpoint["target_scaler"].scale_,
        ).assign(
            representation="multimodal_time_linear", seed=seed,
            heldout_experiment=test,
        )
        sensitivity_rows = candidates
        if TIME_COLUMN not in sensitivity_rows:
            sensitivity_rows = sensitivity_rows.merge(
                latents[["row_id", TIME_COLUMN]], on="row_id", validate="one_to_one"
            )
        pathways = local_pathway_sensitivity(sensitivity_rows, checkpoint).assign(
            representation="multimodal_time_linear", seed=seed,
            heldout_experiment=test,
        )
    for name, table in (
        ("event_errors", event_errors), ("roughness", roughness), ("pathways", pathways),
        ("nearest_neighbors", nearest),
    ):
        if not table.empty:
            table.to_csv(paths[name], index=False)
    complete.write_text(json.dumps({"source_checkpoint": str(source)}) + "\n")
    return {"event_errors": event_errors, "roughness": roughness, "pathways": pathways,
            "nearest_neighbors": nearest}


def _diagnose_d32(
    args: argparse.Namespace, events: pd.DataFrame, folds: dict[str, str]
) -> None:
    """Create state-study diagnostics that reference, but never copy, frozen D32."""
    for seed in (0, 1):
        root = (
            args.output / "reference_diagnostics" / "multimodal_time_linear"
            / f"seed_{seed}"
        )
        with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
            results = Parallel()(
                delayed(_diagnose_d32_fold)(args, events, test, seed)
                for test in folds
            )
        for name in ("event_errors", "roughness", "pathways", "nearest_neighbors"):
            tables = [result[name] for result in results if not result[name].empty]
            if tables:
                pd.concat(tables, ignore_index=True).to_csv(root / f"{name}.csv", index=False)


def vcnet(args: argparse.Namespace) -> None:
    """Fit paired time-aware event heads and replay their held-out Pareto decisions."""
    require_matching_extrapolation_setting(args.data, args.allow_extrapolation)
    events = _vcnet_event_starts(args)
    folds = fold_exclusions(events["experiment_id"].astype(str).unique().tolist())
    if args.heldout_experiment:
        folds = {args.heldout_experiment: folds[args.heldout_experiment]}
    if args.diagnose_state:
        _diagnose_d32(args, events, folds)
    for event_head in args.event_head:
        run = args.output / event_head / f"seed_{args.seed}"
        settings = {
            "action": "vcnet", "data": str(args.data), "dataset": str(args.dataset),
            "event_head": event_head, "seed": args.seed,
            "maximum_epochs": args.maximum_epochs, "patience": args.patience,
            "allow_extrapolation": args.allow_extrapolation, "folds": folds,
            "training_events": len(events), "diagnose_state": args.diagnose_state,
        }
        save_settings(run / "settings.json", settings)
        with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
            artifacts = Parallel()(
                delayed(_vcnet_fold)(args, events, event_head, test, inner, run)
                for test, inner in folds.items()
            )
        pd.concat([value["outcome_predictions"] for value in artifacts]).to_csv(
            run / "outcome_predictions.csv", index=False
        )
        pd.concat([value["losses"] for value in artifacts]).to_csv(
            run / "losses.csv", index=False
        )
        for name, filename in (
            ("candidates", "candidates.parquet"),
            ("replays", "trajectory_predictions.parquet"),
        ):
            tables = [value[name] for value in artifacts if not value[name].empty]
            if tables:
                pd.concat(tables, ignore_index=True).to_parquet(run / filename, index=False)
        for name, filename in (
            ("roughness", "roughness.csv"),
            ("nearest_neighbors", "nearest_neighbors.csv"),
        ):
            tables = [value[name] for value in artifacts if not value[name].empty]
            if tables:
                pd.concat(tables, ignore_index=True).to_csv(run / filename, index=False)
        log_wandb(
            args.wandb_project, f"{event_head}-seed-{args.seed}", settings,
            completed_folds=float(len(artifacts)),
        )
    from plots.pareto_learning import render_vcnet_figures

    render_vcnet_figures(args.output, args.data, args.reference_run)


def _head_rows(
    args: argparse.Namespace, test: str, inner: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nested = load_fold_rows(args.data, args.rgb_cache, (test, inner), include_rgb=False)
    outer = load_fold_rows(args.data, args.rgb_cache, (test,), include_rgb=False)
    folder = args.representations / "folds" / test
    nested = nested.merge(pd.read_parquet(folder / "inner_latents.parquet"), on="row_id")
    outer = outer.merge(pd.read_parquet(folder / "outer_latents.parquet"), on="row_id")
    nested = nested.loc[nested.cycle_name.isin(fold_ready_cycles(nested))]
    outer = outer.loc[outer.cycle_name.isin(fold_ready_cycles(outer))]
    return nested, outer


def merge_visual_latents(rows: pd.DataFrame, latents: pd.DataFrame) -> pd.DataFrame:
    columns = ["row_id", *sorted(name for name in latents if name.startswith("z_"))]
    return rows.merge(latents[columns], on="row_id", validate="one_to_one")


def _heads_fold(args: argparse.Namespace, test: str, inner: str) -> dict:
    import torch

    from image_models.pareto_learning import stop_feature_columns, train_stop_fold

    path = args.output / "folds" / f"{test}.pkl"
    if path.exists():
        with path.open("rb") as stream:
            return pickle.load(stream)  # noqa: S301 - local run-owned checkpoint
    torch.set_num_threads(1)
    nested, outer = _head_rows(args, test, inner)
    nested = nested.loc[nested.target.notna() & nested.experiment_id.astype(str).ne(test)]
    outer_train = outer.loc[outer.target.notna() & outer.experiment_id.astype(str).ne(test)]
    inner_train = nested.loc[nested.experiment_id.astype(str).ne(inner)]
    inner_validation = nested.loc[nested.experiment_id.astype(str).eq(inner)]
    outer_test = outer.loc[outer.experiment_id.astype(str).eq(test)]
    results = {
        method: train_stop_fold(
            inner_train, inner_validation, outer_train, outer_test, method=method,
            seed=args.seed, maximum_epochs=args.maximum_epochs, patience=args.patience,
        )
        for method in ("s0", "s1", "s2", "s3", "n2")
    }
    selected = min(
        ("s0", "s1", "s2", "s3"),
        key=lambda method: (
            results[method]["validation_loss"],
            len(stop_feature_columns(outer_train, method)), method,
        ),
    )
    results["s4"] = train_stop_fold(
        inner_train, inner_validation, outer_train, outer_test, method="s4",
        base_method=selected, seed=args.seed, maximum_epochs=args.maximum_epochs,
        patience=args.patience,
    )
    for method, result in results.items():
        for name in ("predictions", "losses", "pair_metrics"):
            result[name]["heldout_experiment"] = test
            result[name]["method"] = method
    artifact = {"results": results, "selected_for_s4": selected,
                "heldout_experiment": test, "inner_validation_experiment": inner}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(artifact, stream)
    temporary.replace(path)
    print(f"heads: completed {test}; S4 base={selected}", flush=True)
    return artifact


def _probe_rows(
    args: argparse.Namespace, test: str, inner: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the same frozen coordinates plus their pre-compression inputs."""
    nested = load_fold_rows(args.data, args.rgb_cache, (test, inner))
    outer = load_fold_rows(args.data, args.rgb_cache, (test,))
    folder = args.representations / "folds" / test
    with (folder / "checkpoint.pkl").open("rb") as stream:
        representation = pickle.load(stream)  # noqa: S301 - local frozen checkpoint
    raw_columns = representation["checkpoints"]["visual"]["feature_columns"]
    nested = merge_visual_latents(
        nested, pd.read_parquet(folder / "inner_latents.parquet")
    )
    outer = merge_visual_latents(
        outer, pd.read_parquet(folder / "outer_latents.parquet")
    )
    nested = nested.loc[nested.cycle_name.isin(fold_ready_cycles(nested))]
    outer = outer.loc[outer.cycle_name.isin(fold_ready_cycles(outer))]
    return nested, outer, raw_columns


def _probe_fold(args: argparse.Namespace, test: str, inner: str) -> dict:
    import torch

    from image_models.pareto_learning import (
        PROBE_RECIPES,
        probe_feature_columns,
        train_stop_fold,
    )

    path = args.output / "folds" / f"{test}.pkl"
    if path.exists():
        with path.open("rb") as stream:
            return pickle.load(stream)  # noqa: S301 - local run-owned checkpoint
    torch.set_num_threads(1)
    nested, outer, raw_columns = _probe_rows(args, test, inner)
    nested = nested.loc[nested.target.notna() & nested.experiment_id.astype(str).ne(test)]
    outer_train = outer.loc[outer.target.notna() & outer.experiment_id.astype(str).ne(test)]
    inner_train = nested.loc[nested.experiment_id.astype(str).ne(inner)]
    inner_validation = nested.loc[nested.experiment_id.astype(str).eq(inner)]
    outer_test = outer.loc[outer.experiment_id.astype(str).eq(test)]
    results = {}
    for method, recipe in PROBE_RECIPES.items():
        columns = probe_feature_columns(outer_train, method, raw_columns)
        result = train_stop_fold(
            inner_train, inner_validation, outer_train, outer_test,
            method=method, feature_columns=columns, hidden_width=recipe.hidden_width,
            seed=args.seed, maximum_epochs=args.maximum_epochs, patience=args.patience,
        )
        for name in ("predictions", "losses", "pair_metrics"):
            result[name]["heldout_experiment"] = test
            result[name]["method"] = method
            result[name]["seed"] = args.seed
        results[method] = result
    artifact = {
        "results": results, "heldout_experiment": test,
        "inner_validation_experiment": inner,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(artifact, stream)
    temporary.replace(path)
    print(f"probe: completed {test}", flush=True)
    return artifact


def probe(args: argparse.Namespace) -> None:
    """Compare stopping-head capacity and pre-compression feature access."""
    require_matching_extrapolation_setting(args.data, args.allow_extrapolation)
    folds = json.loads((args.data / "folds.json").read_text())
    if args.heldout_experiment:
        folds = {args.heldout_experiment: folds[args.heldout_experiment]}
    settings = {
        "action": "probe", "data": str(args.data),
        "representations": str(args.representations), "rgb_cache": str(args.rgb_cache),
        "calibration_runs": [str(run) for run in args.runs or []],
        "allow_extrapolation": args.allow_extrapolation, "seed": args.seed,
        "maximum_epochs": args.maximum_epochs, "patience": args.patience,
        "folds": folds, "methods": list(PROBE_METHODS),
    }
    save_settings(args.output / "settings.json", settings)
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        artifacts = Parallel()(
            delayed(_probe_fold)(args, test, inner) for test, inner in folds.items()
        )
    predictions = pd.concat([
        artifact["results"][method]["predictions"].assign(seed=args.seed)
        for artifact in artifacts for method in PROBE_METHODS
    ], ignore_index=True)
    losses = pd.concat([
        artifact["results"][method]["losses"].assign(seed=args.seed)
        for artifact in artifacts for method in PROBE_METHODS
    ], ignore_index=True)
    selected_epochs = pd.DataFrame([
        {
            "heldout_experiment": artifact["heldout_experiment"], "method": method,
            "selected_epoch": artifact["results"][method]["checkpoint"]["selected_epoch"],
            "validation_loss": artifact["results"][method]["validation_loss"],
        }
        for artifact in artifacts for method in PROBE_METHODS
    ])
    predictions.to_parquet(args.output / "predictions.parquet", index=False)
    losses.to_csv(args.output / "losses.csv", index=False)
    selected_epochs.to_csv(args.output / "selected_epochs.csv", index=False)

    from plots.pareto_learning import (
        performance_consequences,
        render_calibration_figures,
        render_probe_examples,
        render_probe_figures,
    )

    evaluation = load_performance_evaluation(args.data, folds)
    consequences = performance_consequences(
        predictions, evaluation, methods=PROBE_METHODS,
    )
    consequences.to_csv(args.output / "performance_consequences.csv", index=False)
    render_probe_figures(
        predictions, losses, selected_epochs, consequences, args.output,
        denominator=consequences[["heldout_experiment", "cycle_name"]].drop_duplicates().shape[0],
    )
    render_probe_examples(
        predictions, consequences, load_teacher_curves(args.data, folds), args.output
    )
    calibration, event_starts = load_probe_calibration(args)
    render_calibration_figures(calibration, event_starts, args.output)
    log_wandb(
        args.wandb_project, args.output.name, settings,
        completed_folds=float(len(artifacts)),
    )


def load_state_checkpoints(
    source: Path, seed: int, test: str, inner: str
) -> tuple[dict, dict | None]:
    """Load one frozen outer encoder and reuse its inner encoder only when matched."""
    path = source / f"seed_{seed}" / "folds" / test / "checkpoint.pkl"
    with path.open("rb") as stream:
        artifact = pickle.load(stream)  # noqa: S301 - local frozen research artifact
    matched = artifact.get("inner_validation_experiment") == inner
    return artifact["checkpoint"], artifact["inner_checkpoint"] if matched else None


def _matched_inner_state(
    args: argparse.Namespace, source: Path, seed: int, test: str, inner: str,
    outer_checkpoint: dict, events: pd.DataFrame, nested: pd.DataFrame,
) -> dict:
    """Fit only the two inner encoder combinations absent from the frozen runs."""
    event_head = outer_checkpoint["event_head"]
    path = (
        args.output / "inner_representations" / event_head / f"seed_{seed}"
        / f"{test}__{inner}.pkl"
    )
    if path.exists():
        with path.open("rb") as stream:
            return pickle.load(stream)  # noqa: S301 - local run-owned checkpoint
    from image_models.outcome_representation import (
        CURVATURE_HEADS,
        fit_outcome_representation,
    )

    settings = json.loads((source / f"seed_{seed}" / "settings.json").read_text())
    experiment = events.experiment_id.astype(str)
    train = events.loc[experiment.ne(test) & experiment.ne(inner)]
    validation = events.loc[experiment.eq(inner)]
    temporal = None
    if event_head in CURVATURE_HEADS:
        temporal = nested.loc[
            ~nested.experiment_id.astype(str).isin((test, inner))
        ]
    fitted = fit_outcome_representation(
        train, validation, train, use_rgb=True, event_head=event_head, seed=seed,
        maximum_epochs=int(settings["maximum_epochs"]),
        patience=int(settings["patience"]), inner_temporal=temporal,
        outer_temporal=temporal,
    )
    checkpoint = fitted["inner_checkpoint"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(checkpoint, stream)
    temporary.replace(path)
    return checkpoint


def _state_transfer_fold(
    args: argparse.Namespace, sources: tuple[Path, Path], events: pd.DataFrame,
    seed: int, test: str, inner: str,
) -> dict:
    """Fit the three matched stopping candidates for one outer fold."""
    import torch

    from image_models.outcome_representation import encode_outcome_rows
    from image_models.pareto_learning import (
        STATE_TRANSFER_RECIPES,
        probe_feature_columns,
        train_stop_fold,
    )

    path = args.output / f"seed_{seed}" / "folds" / f"{test}.pkl"
    if path.exists():
        with path.open("rb") as stream:
            return pickle.load(stream)  # noqa: S301 - local run-owned checkpoint
    torch.set_num_threads(1)
    nested_all = load_fold_rows(args.data, args.rgb_cache, (test, inner))
    outer = load_fold_rows(args.data, args.rgb_cache, (test,))
    nested = nested_all.loc[
        nested_all.cycle_name.isin(fold_ready_cycles(nested_all))
    ]
    outer = outer.loc[outer.cycle_name.isin(fold_ready_cycles(outer))]
    encoded = {}
    raw_columns = None
    for method, source in zip(STATE_TRANSFER_METHODS[:2], sources, strict=True):
        outer_checkpoint, inner_checkpoint = load_state_checkpoints(
            source, seed, test, inner
        )
        if inner_checkpoint is None:
            inner_checkpoint = _matched_inner_state(
                args, source, seed, test, inner, outer_checkpoint, events, nested_all
            )
        encoded[method] = (
            merge_visual_latents(nested, encode_outcome_rows(nested, inner_checkpoint)),
            merge_visual_latents(outer, encode_outcome_rows(outer, outer_checkpoint)),
        )
        raw_columns = outer_checkpoint["feature_columns"]
    assert raw_columns is not None
    encoded["raw_ridge_ch_mlp"] = (nested, outer)
    results = {}
    for method, recipe in STATE_TRANSFER_RECIPES.items():
        nested_rows, outer_rows = encoded[method]
        nested_rows = nested_rows.loc[
            nested_rows.target.notna() & nested_rows.experiment_id.astype(str).ne(test)
        ]
        outer_train = outer_rows.loc[
            outer_rows.target.notna() & outer_rows.experiment_id.astype(str).ne(test)
        ]
        inner_train = nested_rows.loc[nested_rows.experiment_id.astype(str).ne(inner)]
        inner_validation = nested_rows.loc[nested_rows.experiment_id.astype(str).eq(inner)]
        outer_test = outer_rows.loc[outer_rows.experiment_id.astype(str).eq(test)]
        columns = probe_feature_columns(outer_train, method, raw_columns)
        result = train_stop_fold(
            inner_train, inner_validation, outer_train, outer_test,
            method=method, feature_columns=columns, hidden_width=recipe.hidden_width,
            seed=seed, maximum_epochs=args.maximum_epochs, patience=args.patience,
        )
        for name in ("predictions", "losses", "pair_metrics"):
            result[name]["heldout_experiment"] = test
            result[name]["method"] = method
            result[name]["seed"] = seed
        results[method] = result
    artifact = {
        "results": results, "seed": seed, "heldout_experiment": test,
        "inner_validation_experiment": inner,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(artifact, stream)
    temporary.replace(path)
    print(f"state transfer seed {seed}: completed {test}", flush=True)
    return artifact


def state_transfer(args: argparse.Namespace) -> None:
    """Compare D32, curvature-stabilized T32 and pre-compression stopping inputs."""
    if not args.runs or len(args.runs) != 2:
        raise ValueError("--action state-transfer requires --runs D32_ROOT T32_ROOT")
    require_matching_extrapolation_setting(args.data, args.allow_extrapolation)
    sources = (args.runs[0], args.runs[1])
    folds = json.loads((args.data / "folds.json").read_text())
    if args.heldout_experiment:
        folds = {args.heldout_experiment: folds[args.heldout_experiment]}
    events = pd.read_parquet(sources[0].parent / "event_starts.parquet")
    settings = {
        "action": "state-transfer", "data": str(args.data),
        "sources": [str(source) for source in sources], "seeds": [0, 1],
        "methods": list(STATE_TRANSFER_METHODS), "folds": folds,
        "maximum_epochs": args.maximum_epochs, "patience": args.patience,
        "allow_extrapolation": args.allow_extrapolation,
    }
    save_settings(args.output / "settings.json", settings)
    tasks = [(seed, test, inner) for seed in (0, 1) for test, inner in folds.items()]
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        artifacts = Parallel()(
            delayed(_state_transfer_fold)(args, sources, events, seed, test, inner)
            for seed, test, inner in tasks
        )
    predictions = pd.concat([
        artifact["results"][method]["predictions"]
        for artifact in artifacts for method in STATE_TRANSFER_METHODS
    ], ignore_index=True)
    losses = pd.concat([
        artifact["results"][method]["losses"]
        for artifact in artifacts for method in STATE_TRANSFER_METHODS
    ], ignore_index=True)
    selected_epochs = pd.DataFrame([
        {
            "heldout_experiment": artifact["heldout_experiment"],
            "method": method, "seed": artifact["seed"],
            "selected_epoch": artifact["results"][method]["checkpoint"]["selected_epoch"],
            "validation_loss": artifact["results"][method]["validation_loss"],
        }
        for artifact in artifacts for method in STATE_TRANSFER_METHODS
    ])
    predictions.to_parquet(args.output / "predictions.parquet", index=False)
    losses.to_csv(args.output / "losses.csv", index=False)
    selected_epochs.to_csv(args.output / "selected_epochs.csv", index=False)

    from plots.pareto_learning import (
        performance_consequences,
        render_probe_examples,
        render_probe_figures,
        select_transfer_method,
        stopping_stability_metrics,
    )

    evaluation = load_performance_evaluation(args.data, folds)
    consequences = pd.concat([
        performance_consequences(
            predictions.loc[predictions.seed.eq(seed)], evaluation,
            methods=STATE_TRANSFER_METHODS,
        )
        for seed in (0, 1)
    ], ignore_index=True)
    consequences.to_csv(args.output / "performance_consequences.csv", index=False)
    stopping_stability_metrics(predictions).to_csv(
        args.output / "score_stability.csv", index=False
    )
    selected, selection = select_transfer_method(consequences)
    selection.to_csv(args.output / "selection_summary.csv", index=False)
    (args.output / "selection.json").write_text(json.dumps({
        "selected_method": selected,
        "selection_order": [
            "unevaluable_fraction", "p90_loss", "median_loss",
            "coverage_1_percent", "coverage_2_percent",
        ],
    }, indent=2) + "\n")
    comparisons = (
        ("t32_ridge_ch_mlp", "d32_ridge_ch_mlp"),
        ("raw_ridge_ch_mlp", "d32_ridge_ch_mlp"),
        ("t32_ridge_ch_mlp", "raw_ridge_ch_mlp"),
    )
    paired = []
    teacher_curves = load_teacher_curves(args.data, folds)
    denominator = consequences[["heldout_experiment", "cycle_name"]].drop_duplicates().shape[0]
    for seed in (0, 1):
        destination = args.output / f"seed_{seed}"
        render_probe_figures(
            predictions.loc[predictions.seed.eq(seed)],
            losses.loc[losses.seed.eq(seed)],
            selected_epochs.loc[selected_epochs.seed.eq(seed)],
            consequences.loc[consequences.seed.eq(seed)], destination,
            denominator=denominator, comparisons=comparisons,
        )
        table = pd.read_csv(destination / "paired_comparisons.csv").assign(seed=seed)
        paired.append(table)
        render_probe_examples(
            predictions.loc[predictions.seed.eq(seed)],
            consequences.loc[consequences.seed.eq(seed)], teacher_curves, destination,
            candidate="t32_ridge_ch_mlp", reference="d32_ridge_ch_mlp",
        )
    pd.concat(paired, ignore_index=True).to_csv(
        args.output / "paired_comparisons.csv", index=False
    )
    coverage = pd.read_csv(args.data / "cycle_coverage.csv")
    coverage["in_stopping_comparison_panel"] = coverage.cycle_name.isin(
        consequences.cycle_name
    )
    coverage.to_csv(args.output / "source_coverage.csv", index=False)
    log_wandb(
        args.wandb_project, args.output.name, settings,
        completed_folds=float(len(artifacts)),
    )


def _oof_teacher_labels(data: Path, folds: dict[str, str]) -> pd.DataFrame:
    """Keep each cycle's frozen outer-fold knee labels for final stopping refit."""
    base = pd.read_parquet(data / "base.parquet", columns=["row_id", "experiment_id"])
    tables = []
    fields = ["row_id", "target", "is_knee", "teacher_time"]
    for test in folds:
        row_ids = set(base.loc[base.experiment_id.astype(str).eq(test), "row_id"])
        teacher = pd.read_parquet(data / "teachers" / f"{test}.parquet", columns=fields)
        tables.append(teacher.loc[teacher.row_id.isin(row_ids)])
    return pd.concat(tables, ignore_index=True)


def _full_refit_rows(
    args: argparse.Namespace, parameters: dict, labels: pd.DataFrame
) -> pd.DataFrame:
    """Build full-Ridge economic inputs while retaining frozen OOF decisions."""
    from image_models.image_features import load_dinov2_feature_cache
    from image_models.pareto_data import add_online_economic_features, apply_fold_teacher

    base = pd.read_parquet(args.data / "base.parquet")
    rows = pd.concat([
        apply_fold_teacher(
            cycle, parameters, allow_model_extrapolation=args.allow_extrapolation
        )
        for _, cycle in base.groupby("cycle_name", sort=False)
    ], ignore_index=True)
    rows = rows.drop(columns=["target", "is_knee", "teacher_time"], errors="ignore")
    rows = rows.merge(labels, on="row_id", validate="one_to_one")
    rows = add_online_economic_features(rows)
    keys = ["cycle_name", "camera_role", "file_name"]
    images = rows.loc[rows.file_name.notna(), keys].drop_duplicates()
    features = load_dinov2_feature_cache(images, args.rgb_cache, "dinov2")
    return rows.merge(features, on=keys, how="left", validate="many_to_one")


def _median_event_epoch(source: Path) -> int:
    values = []
    for path in sorted((source / "seed_0" / "folds").glob("*/checkpoint.pkl")):
        with path.open("rb") as stream:
            values.append(pickle.load(stream)["checkpoint"]["selected_epoch"])  # noqa: S301
    if not values:
        raise ValueError(f"no seed-0 event folds in {source}")
    return int(np.ceil(np.median(values)))


def _refit_event_tables(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep Ridge validity rows separate from feature-rich event-start rows."""
    return (
        pd.read_parquet(source / "events.parquet"),
        pd.read_parquet(source / "event_starts.parquet"),
    )


def refit(args: argparse.Namespace) -> None:
    """Fit the preselected stopping algorithm once on all available data."""
    if not args.selected_method:
        raise ValueError("--action refit requires --selected-method")
    if not args.runs or len(args.runs) != 2:
        raise ValueError("--action refit requires --runs D32_ROOT T32_ROOT")
    selection_path = args.reference_run / "selection.json"
    selected = json.loads(selection_path.read_text())["selected_method"]
    if args.selected_method != selected:
        raise ValueError(f"selected method differs from frozen comparison: {selected}")
    from image_models.outcome_representation import (
        encode_outcome_rows,
        fit_outcome_fixed,
    )
    from image_models.pareto_data import fit_fold_parameters
    from image_models.pareto_learning import (
        STATE_TRANSFER_RECIPES,
        probe_feature_columns,
        train_stop_fixed,
    )

    require_matching_extrapolation_setting(args.data, args.allow_extrapolation)
    folds = json.loads((args.data / "folds.json").read_text())
    ridge_events, events = _refit_event_tables(args.runs[0].parent)
    ridge = fit_fold_parameters(ridge_events, ())
    rows = _full_refit_rows(args, ridge, _oof_teacher_labels(args.data, folds))
    outcome_checkpoint = None
    event_losses = pd.DataFrame()
    source = None
    if selected == "d32_ridge_ch_mlp":
        source = args.runs[0]
    elif selected == "t32_ridge_ch_mlp":
        source = args.runs[1]
    if source is not None:
        event_head = (
            "multimodal_time_linear" if selected.startswith("d32_")
            else "multimodal_time_linear_z32_curvature"
        )
        event_epoch = _median_event_epoch(source)
        outcome_checkpoint, event_losses = fit_outcome_fixed(
            events, use_rgb=True, event_head=event_head, epochs=event_epoch, seed=0,
            temporal=rows if selected.startswith("t32_") else None,
        )
        rows = merge_visual_latents(
            rows, encode_outcome_rows(rows, outcome_checkpoint)
        )
        raw_columns = outcome_checkpoint["feature_columns"]
    else:
        with (
            args.runs[0] / "seed_0" / "folds" / next(iter(folds)) / "checkpoint.pkl"
        ).open("rb") as stream:
            raw_columns = pickle.load(stream)["checkpoint"]["feature_columns"]  # noqa: S301
        event_epoch = None
    rows = rows.loc[rows.cycle_name.isin(fold_ready_cycles(rows)) & rows.target.notna()]
    selected_epochs = pd.read_csv(args.reference_run / "selected_epochs.csv")
    stop_epoch = int(np.ceil(np.median(selected_epochs.loc[
        selected_epochs.method.eq(selected) & selected_epochs.seed.eq(0),
        "selected_epoch",
    ])))
    columns = probe_feature_columns(rows, selected, raw_columns)
    stopping = train_stop_fixed(
        rows, rows, method=selected, epochs=stop_epoch, seed=0,
        feature_columns=columns,
        hidden_width=STATE_TRANSFER_RECIPES[selected].hidden_width,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "checkpoint.pkl").open("wb") as stream:
        pickle.dump({
            "selected_method": selected,
            "ridge_event_models": ridge,
            "outcome_checkpoint": outcome_checkpoint,
            "stopping_checkpoint": stopping["checkpoint"],
        }, stream)
    settings = {
        "action": "refit", "selected_method": selected, "seed": 0,
        "training_events": len(events), "stopping_cycles": rows.cycle_name.nunique(),
        "event_epoch": event_epoch, "stopping_epoch": stop_epoch,
        "feature_columns": columns, "ridge_model": "dynamic_state_8_full_data",
        "label_source": "frozen outer-fold Ridge teacher knees",
    }
    save_settings(args.output / "settings.json", settings)
    event_losses.to_csv(args.output / "event_losses.csv", index=False)
    stopping["losses"].to_csv(args.output / "stopping_losses.csv", index=False)
    stopping["predictions"].to_parquet(args.output / "training_predictions.parquet", index=False)
    log_wandb(
        args.wandb_project, args.output.name, settings,
        training_events=float(len(events)), stopping_cycles=float(rows.cycle_name.nunique()),
    )


def load_probe_calibration(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    """Load existing OOF event predictions; repeated seeds remain separate groups."""
    from defrost_event_models.ridge_models import OUTCOME_TARGETS
    from image_models.outcome_representation import outcome_event_rows

    targets = list(OUTCOME_TARGETS.values())
    frozen = pd.read_csv(args.representations / "outcome_predictions.csv")
    frozen["seed"] = 0
    frozen["panel"] = "frozen_92"
    visual = frozen.loc[frozen.representation.eq("visual")]
    ridge_columns = {f"ridge_predicted_{target}": f"predicted_{target}" for target in targets}
    ridge = visual[[
        "event_id", "cycle_name", "experiment_id", *targets, *ridge_columns,
    ]].rename(columns=ridge_columns).assign(
        representation="ridge", seed=0, panel="frozen_92"
    )
    tables = [frozen, ridge]
    starts = {
        "frozen_92": outcome_event_rows(
            pd.read_parquet(args.data / "base.parquet"),
            pd.read_parquet(args.data / "events.parquet"),
        )
    }
    for root in args.runs or []:
        path = root / "event_starts.parquet"
        run_files = sorted(root.glob("*/seed_*/outcome_predictions.csv"))
        if not path.exists() or not run_files:
            continue
        time_aware = pd.concat([pd.read_csv(run) for run in run_files], ignore_index=True)
        time_aware["panel"] = "time_aware_101"
        tables.append(time_aware)
        starts["time_aware_101"] = pd.read_parquet(path)
    return pd.concat(tables, ignore_index=True), starts


def load_teacher_curves(data: Path, heldout_experiments) -> pd.DataFrame:
    """Load the original fold-frozen teacher grid for publication examples."""
    base = pd.read_parquet(data / "base.parquet")
    tables = []
    for heldout in sorted(heldout_experiments):
        teacher = pd.read_parquet(data / "teachers" / f"{heldout}.parquet")
        overlap = base.columns.intersection(teacher.columns).difference(["row_id"])
        rows = base.drop(columns=overlap).merge(teacher, on="row_id", validate="one_to_one")
        tables.append(rows.loc[
            rows.experiment_id.astype(str).eq(heldout) & rows.is_teacher_candidate
        ].assign(heldout_experiment=heldout))
    return pd.concat(tables, ignore_index=True)


def heads(args: argparse.Namespace) -> None:
    require_matching_extrapolation_setting(args.data, args.allow_extrapolation)
    folds = json.loads((args.data / "folds.json").read_text())
    if args.heldout_experiment:
        folds = {args.heldout_experiment: folds[args.heldout_experiment]}
    settings = {
        "data": str(args.data), "representations": str(args.representations), "seed": args.seed,
        "maximum_epochs": args.maximum_epochs, "patience": args.patience, "folds": folds,
    }
    save_settings(args.output / "settings.json", settings)
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        artifacts = Parallel()(delayed(_heads_fold)(args, test, inner)
                               for test, inner in folds.items())
    for name in ("predictions", "losses", "pair_metrics"):
        tables = [artifact["results"][method][name]
                  for artifact in artifacts for method in STOP_METHODS]
        path = args.output / ("predictions.parquet" if name == "predictions" else f"{name}.csv")
        combined = pd.concat(tables, ignore_index=True)
        if name == "predictions":
            duplicated = [column for column in combined if column.endswith("_x")
                          and column[:-2] + "_y" in combined]
            for column in duplicated:
                combined[column[:-2]] = combined[column]
            combined = combined.drop(columns=[
                column for source in duplicated for column in (source, source[:-2] + "_y")
            ])
            combined.to_parquet(path, index=False)
        else:
            combined.to_csv(path, index=False)
    pd.DataFrame({
        "heldout_experiment": [artifact["heldout_experiment"] for artifact in artifacts],
        "inner_validation_experiment": [
            artifact["inner_validation_experiment"] for artifact in artifacts
        ],
        "selected_for_s4": [artifact["selected_for_s4"] for artifact in artifacts],
    }).to_csv(args.output / "selected_inputs.csv", index=False)
    log_wandb(args.wandb_project, args.output.name, settings, completed_folds=len(artifacts))


def _neural_fold(args: argparse.Namespace, test: str, inner: str) -> dict:
    import torch

    from image_models.outcome_representation import predict_outcomes_from_latent
    from image_models.pareto_data import (
        add_neural_event_predictions,
        add_online_economic_features,
        apply_neural_pareto,
    )
    from image_models.pareto_learning import train_stop_fixed

    path = args.output / "folds" / f"{test}.pkl"
    if path.exists():
        with path.open("rb") as stream:
            saved = pickle.load(stream)  # noqa: S301 - local run-owned checkpoint
        if any(name.startswith("z_") for name in saved["checkpoint"]["feature_columns"]):
            return saved
    torch.set_num_threads(1)
    rows = load_fold_rows(args.data, args.rgb_cache, (test,), include_rgb=False)
    representation = args.representations / "folds" / test
    latent = pd.read_parquet(representation / "outer_latents.parquet")
    with (representation / "checkpoint.pkl").open("rb") as stream:
        event_checkpoint = pickle.load(stream)["checkpoints"]["visual"]  # noqa: S301
    event_predictions = predict_outcomes_from_latent(latent, event_checkpoint)
    candidates = apply_neural_pareto(rows, event_predictions)

    neural_rows = add_online_economic_features(
        add_neural_event_predictions(
            merge_visual_latents(rows, latent), event_predictions
        ),
        feature_prefix="neural",
    )
    neural_rows = neural_rows.loc[neural_rows.cycle_name.isin(fold_ready_cycles(rows))]
    neural_rows = neural_rows.loc[neural_rows.target.notna()]
    input_columns = CURRENT_ECONOMIC_INPUTS
    model_rows = neural_rows.copy()
    for column in input_columns:
        model_rows[f"online_{column}"] = model_rows[f"neural_{column}"]
    train_rows = model_rows.loc[model_rows.experiment_id.astype(str).ne(test)]
    test_rows = model_rows.loc[model_rows.experiment_id.astype(str).eq(test)]
    with (args.reference_run / "folds" / f"{test}.pkl").open("rb") as stream:
        reference = pickle.load(stream)  # noqa: S301 - local run-owned checkpoint
    epochs = int(reference["results"]["s1"]["checkpoint"]["selected_epoch"])
    fitted = train_stop_fixed(train_rows, test_rows, method="s1", epochs=epochs, seed=args.seed)
    prediction = fitted["predictions"].rename(columns={
        f"online_{column}": f"neural_{column}" for column in input_columns
    })
    ridge_inputs = ["row_id", *[f"online_{column}" for column in input_columns]]
    prediction = prediction.merge(
        neural_rows.loc[
            neural_rows.experiment_id.astype(str).eq(test) & neural_rows.is_frame,
            ridge_inputs,
        ],
        on="row_id", how="left", validate="one_to_one",
    )
    prediction["method"] = "s1_neural"
    prediction["seed"] = args.seed
    prediction["heldout_experiment"] = test
    prediction["economic_source"] = "neural_event_head"
    losses = fitted["losses"].assign(
        method="s1_neural", seed=args.seed, heldout_experiment=test,
        epoch_source="frozen_s1",
    )
    artifact = {
        "candidates": candidates.loc[candidates.experiment_id.astype(str).eq(test)].assign(
            heldout_experiment=test
        ),
        "predictions": prediction,
        "losses": losses,
        "checkpoint": fitted["checkpoint"],
        "selected_epoch": epochs,
        "heldout_experiment": test,
        "inner_validation_experiment": inner,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(artifact, stream)
    temporary.replace(path)
    print(f"neural: completed {test}; S1 epoch={epochs}", flush=True)
    return artifact


def neural(args: argparse.Namespace) -> None:
    """Replay frozen neural outcomes, select Neural Pareto, and fit S1-neural."""
    require_matching_extrapolation_setting(args.data, args.allow_extrapolation)
    folds = json.loads((args.data / "folds.json").read_text())
    if args.heldout_experiment:
        folds = {args.heldout_experiment: folds[args.heldout_experiment]}
    settings = {
        "action": "neural", "data": str(args.data),
        "representations": str(args.representations),
        "reference_run": str(args.reference_run), "seed": args.seed,
        "allow_extrapolation": args.allow_extrapolation, "folds": folds,
        "stopping_label": "frozen_ridge_pareto_knee",
        "event_model": "frozen_visual_outcome_encoder_and_head",
    }
    save_settings(args.output / "settings.json", settings)
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        artifacts = Parallel()(
            delayed(_neural_fold)(args, test, inner) for test, inner in folds.items()
        )
    candidates = pd.concat([value["candidates"] for value in artifacts], ignore_index=True)
    neural_predictions = pd.concat(
        [value["predictions"] for value in artifacts], ignore_index=True
    )
    reference_predictions = pd.read_parquet(args.reference_run / "predictions.parquet")
    reference_predictions = reference_predictions.loc[
        reference_predictions.method.isin(["s0", "s1"])
        & reference_predictions.heldout_experiment.astype(str).isin(folds)
    ]
    if "seed" not in reference_predictions:
        reference_seed = json.loads((args.reference_run / "settings.json").read_text())["seed"]
        reference_predictions["seed"] = reference_seed
    predictions = pd.concat([reference_predictions, neural_predictions], ignore_index=True)
    losses = pd.concat([value["losses"] for value in artifacts], ignore_index=True)
    candidates.to_parquet(args.output / "candidates.parquet", index=False)
    predictions.to_parquet(args.output / "predictions.parquet", index=False)
    losses.to_csv(args.output / "losses.csv", index=False)
    from plots.pareto_learning import render_neural_figures

    render_neural_figures(predictions, candidates, losses, args.output / "figures")
    log_wandb(args.wandb_project, args.output.name, settings, completed_folds=len(artifacts))


def replace_economic_inputs(rows: pd.DataFrame, source: str) -> pd.DataFrame:
    """Keep one frozen representation and swap the complete current C/H input group."""
    result = rows.copy()
    if source == "ridge":
        return result
    if source != "neural":
        raise ValueError(f"unknown economic input source: {source}")
    for name in CURRENT_ECONOMIC_INPUTS:
        result[f"online_{name}"] = result[f"neural_{name}"]
    return result


def _cross_input_fold(
    args: argparse.Namespace,
    test: str,
    ridge_stream: pd.DataFrame,
    neural_stream: pd.DataFrame,
) -> pd.DataFrame:
    """Replay both frozen heads with both saved economic input sources for one fold."""
    import torch

    from image_models.pareto_learning import predict_stop_rows

    path = args.output / "folds" / f"{test}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    torch.set_num_threads(1)
    keys = ["row_id", "cycle_name", "image_time", "teacher_time"]
    left = ridge_stream[keys].sort_values("row_id", kind="stable").reset_index(drop=True)
    right = neural_stream[keys].sort_values("row_id", kind="stable").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError(f"{test}: Ridge and Neural native-frame keys differ")
    latent = pd.read_parquet(
        args.representations / "folds" / test / "outer_latents.parquet"
    )
    rows = neural_stream.merge(latent, on="row_id", validate="one_to_one")
    with (args.runs[0] / "folds" / f"{test}.pkl").open("rb") as stream:
        ridge_checkpoint = pickle.load(stream)["results"]["s1"]["checkpoint"]  # noqa: S301
    with (args.runs[1] / "folds" / f"{test}.pkl").open("rb") as stream:
        neural_checkpoint = pickle.load(stream)["checkpoint"]  # noqa: S301
    sources = {
        "ridge_head_ridge_ch": ("ridge", "ridge", ridge_checkpoint, ridge_stream),
        "ridge_head_neural_ch": ("ridge", "neural", ridge_checkpoint, None),
        "neural_head_ridge_ch": ("neural", "ridge", neural_checkpoint, None),
        "neural_head_neural_ch": ("neural", "neural", neural_checkpoint, neural_stream),
    }
    predictions = []
    for method, (head_source, economic_source, checkpoint, historical) in sources.items():
        replay = predict_stop_rows(
            replace_economic_inputs(rows, economic_source), checkpoint
        )
        replay["method"] = method
        replay["method_name"] = METHOD_NAMES[method].name
        replay["method_label"] = METHOD_NAMES[method].label
        replay["head_source"] = head_source
        replay["economic_source"] = economic_source
        replay["heldout_experiment"] = test
        replay["seed"] = args.seed
        if historical is not None:
            expected = historical[["row_id", "logit"]].rename(
                columns={"logit": "historical_logit"}
            )
            replay = replay.drop(columns="historical_logit", errors="ignore").merge(
                expected, on="row_id", validate="one_to_one"
            )
            replay["replay_logit_difference"] = replay.logit - replay.historical_logit
            if replay.replay_logit_difference.abs().max() > 1e-5:
                raise ValueError(f"{test}: {method} does not reproduce its saved logits")
        predictions.append(replay)
    result = pd.concat(predictions, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    print(f"cross-input: completed {test}", flush=True)
    return result


def cross_input(args: argparse.Namespace) -> None:
    """Replay frozen Ridge and Neural stopping heads on both saved C/H inputs."""
    if not args.runs or len(args.runs) != 2:
        raise ValueError("--action cross-input requires --runs RIDGE_RUN NEURAL_RUN")
    ridge = pd.read_parquet(args.runs[0] / "predictions.parquet").loc[
        lambda rows: rows.method.eq("s1")
    ]
    neural = pd.read_parquet(args.runs[1] / "predictions.parquet").loc[
        lambda rows: rows.method.eq("s1_neural")
    ]
    folds = sorted(set(ridge.heldout_experiment.astype(str)) & set(
        neural.heldout_experiment.astype(str)
    ))
    if args.heldout_experiment:
        folds = [args.heldout_experiment]
    settings = {
        "action": "cross-input", "data": str(args.data),
        "representations": str(args.representations),
        "ridge_run": str(args.runs[0]), "neural_run": str(args.runs[1]),
        "folds": folds, "seed": args.seed,
        "comparison": "fixed head; Ridge versus Neural current C/H inputs",
    }
    save_settings(args.output / "settings.json", settings)
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        artifacts = Parallel()(
            delayed(_cross_input_fold)(
                args, test,
                ridge.loc[ridge.heldout_experiment.astype(str).eq(test)],
                neural.loc[neural.heldout_experiment.astype(str).eq(test)],
            )
            for test in folds
        )
    predictions = pd.concat(artifacts, ignore_index=True)
    predictions.to_parquet(args.output / "predictions.parquet", index=False)
    evaluation = load_performance_evaluation(args.data, folds)
    from plots.pareto_learning import (
        cross_input_comparisons,
        performance_consequences,
        render_cross_input_figures,
        render_performance_coverage,
    )

    consequences = performance_consequences(
        predictions, evaluation, methods=CROSS_INPUT_METHODS
    )
    consequences.to_csv(args.output / "performance_consequences.csv", index=False)
    cycle_pairs, frame_pairs = cross_input_comparisons(predictions, consequences)
    cycle_pairs.to_csv(args.output / "cross_input_cycles.csv", index=False)
    frame_pairs.to_csv(args.output / "cross_input_frames.csv", index=False)
    summaries = []
    denominator = consequences[["heldout_experiment", "cycle_name"]].drop_duplicates().shape[0]
    for strategy in ("first_positive", "two_of_three"):
        _, summary = render_performance_coverage(
            consequences, strategy, args.output, denominator=denominator
        )
        summaries.append(summary)
    pd.concat(summaries, ignore_index=True).to_csv(
        args.output / "performance_summary.csv", index=False
    )
    render_cross_input_figures(cycle_pairs, frame_pairs, args.output)
    log_wandb(
        args.wandb_project, args.output.name, settings,
        folds=float(len(folds)), cycles=float(denominator),
    )


def load_run_predictions(runs: list[Path]) -> pd.DataFrame:
    """Combine saved predictions and reject conflicting copies of the same method."""
    tables = []
    for run in runs:
        rows = pd.read_parquet(run / "predictions.parquet")
        if "seed" not in rows:
            rows["seed"] = json.loads((run / "settings.json").read_text())["seed"]
        tables.append(rows)
    predictions = pd.concat(tables, ignore_index=True)
    keys = ["method", "seed", "heldout_experiment", "row_id"]
    duplicated = predictions.loc[predictions.duplicated(keys, keep=False)]
    compare = [
        "cycle_name", "image_time", "teacher_time", "logit", "prediction", "target"
    ]
    for column in compare:
        if duplicated.groupby(keys, dropna=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"conflicting duplicate predictions in {column}")
    predictions = predictions.drop_duplicates(keys, keep="first").copy()
    unknown = set(predictions.method).difference(METHOD_NAMES)
    if unknown:
        raise ValueError(f"unknown stopping methods: {sorted(unknown)}")
    predictions["method_name"] = predictions.method.map(
        {method: recipe.name for method, recipe in METHOD_NAMES.items()}
    )
    predictions["method_label"] = predictions.method.map(
        {method: recipe.label for method, recipe in METHOD_NAMES.items()}
    )
    return predictions


def load_selected_inputs(runs: list[Path]) -> pd.DataFrame:
    """Read S4's fold-level input choice and reject conflicting copies."""
    tables = [
        pd.read_csv(run / "selected_inputs.csv")
        for run in runs if (run / "selected_inputs.csv").exists()
    ]
    if not tables:
        return pd.DataFrame()
    selected = pd.concat(tables, ignore_index=True)
    if selected.groupby("heldout_experiment").selected_for_s4.nunique().gt(1).any():
        raise ValueError("conflicting S4 selected inputs")
    return selected.drop_duplicates("heldout_experiment")


def load_performance_evaluation(data: Path, heldout_experiments) -> pd.DataFrame:
    """Load each held-out fold's frozen Ridge objective curve once."""
    base_columns = [
        "row_id", "cycle_name", "experiment_id", "candidate_defrost_time",
        "is_teacher_candidate",
        "pre_defrost_electricity_uses_measurement_reconstruction",
        "pre_defrost_heat_uses_measurement_reconstruction",
    ]
    teacher_columns = [
        "row_id", "cycle_cop", "cycle_heating_rate_kw",
        "cycle_evaporator_capacity_kw",
        "cycle_cop_measurements_valid", "cycle_cop_physically_valid",
        "cycle_heating_rate_kw_measurements_valid",
        "cycle_heating_rate_kw_physically_valid",
        "cycle_evaporator_capacity_kw_measurements_valid",
        "cycle_evaporator_capacity_kw_physically_valid",
        "defrost_event_electricity_in_training_domain",
        "defrost_event_net_heat_in_training_domain",
        "defrost_event_compressor_electricity_in_training_domain",
        "defrost_event_duration_in_training_domain", "is_knee",
    ]
    base = pd.read_parquet(data / "base.parquet", columns=base_columns)
    tables = []
    for heldout in sorted(heldout_experiments):
        rows = base.loc[base.experiment_id.eq(heldout)]
        teacher = pd.read_parquet(
            data / "teachers" / f"{heldout}.parquet", columns=teacher_columns
        )
        tables.append(
            rows.merge(teacher, on="row_id", validate="one_to_one")
            .assign(heldout_experiment=heldout)
        )
    return pd.concat(tables, ignore_index=True)


def evaluate(args: argparse.Namespace) -> None:
    """Re-score saved stopping decisions by their shared Ridge C/H consequences."""
    if not args.runs:
        raise ValueError("--action evaluate requires --runs")
    predictions = load_run_predictions(args.runs)
    missing = set(PERFORMANCE_METHODS).difference(predictions.method.unique())
    if missing:
        raise ValueError(f"missing stopping methods: {sorted(missing)}")
    evaluation = load_performance_evaluation(
        args.data, predictions.heldout_experiment.astype(str).unique()
    )
    selected_inputs = load_selected_inputs(args.runs)
    from plots.pareto_learning import (
        performance_consequences,
        render_performance_coverage,
        render_performance_paired_comparison,
    )

    consequences = performance_consequences(
        predictions, evaluation, methods=PERFORMANCE_METHODS,
        selected_inputs=selected_inputs,
    )
    denominator = consequences[["heldout_experiment", "cycle_name"]].drop_duplicates().shape[0]
    settings = {
        "action": "evaluate", "data": str(args.data),
        "runs": [str(run) for run in args.runs],
        "reference": "fold-frozen Ridge C/H Pareto knee",
        "cohort_cycles": denominator,
    }
    save_settings(args.output / "settings.json", settings)
    consequences.to_csv(args.output / "performance_consequences.csv", index=False)
    coverage = pd.read_csv(args.data / "cycle_coverage.csv").loc[
        lambda rows: rows.preparation_status.eq("eligible")
    ].copy()
    panel_cycles = set(consequences.cycle_name)
    coverage["in_stopping_comparison_panel"] = coverage.cycle_name.isin(panel_cycles)
    coverage["ridge_reference_knee_available"] = coverage.cycle_name.isin(
        evaluation.loc[evaluation.is_knee, "cycle_name"]
    )
    neural_candidates = [
        pd.read_parquet(run / "candidates.parquet", columns=["cycle_name", "neural_is_knee"])
        for run in args.runs if (run / "candidates.parquet").exists()
    ]
    neural_knees = set(
        pd.concat(neural_candidates).loc[lambda rows: rows.neural_is_knee, "cycle_name"]
    ) if neural_candidates else set()
    coverage["neural_pareto_knee_available"] = coverage.cycle_name.isin(neural_knees)
    coverage.to_csv(args.output / "source_coverage.csv", index=False)
    summaries = []
    for strategy in ("first_positive", "two_of_three"):
        _, summary = render_performance_coverage(
            consequences, strategy, args.output, denominator=denominator
        )
        summaries.append(summary)
    pd.concat(summaries, ignore_index=True).to_csv(
        args.output / "performance_summary.csv", index=False
    )
    render_performance_paired_comparison(consequences, args.output)
    log_wandb(
        args.wandb_project, args.output.name, settings,
        evaluated_methods=float(len(PERFORMANCE_METHODS)), cohort_cycles=float(denominator),
    )


def _train_fold(args: argparse.Namespace, test: str, inner: str) -> dict:
    import torch

    from dataset_tools.cycle_metadata import read_catalog
    from image_models.pareto_learning import train_pareto_fold

    path = args.output / "folds" / f"{test}.pkl"
    if path.exists():
        with path.open("rb") as stream:
            return pickle.load(stream)  # noqa: S301 - local run-owned checkpoint
    torch.set_num_threads(1)
    outer = load_fold_rows(args.data, args.rgb_cache, (test,))
    nested = load_fold_rows(args.data, args.rgb_cache, (test, inner))
    ready = pareto_cv_ready_cycles(
        read_catalog(args.dataset), allow_extrapolation=args.allow_extrapolation
    )
    outer = outer.loc[outer["cycle_name"].astype(str).isin(ready)]
    nested = nested.loc[nested["cycle_name"].astype(str).isin(ready)]
    nested = nested.loc[nested["target"].notna() & nested["experiment_id"].ne(test)]
    outer_train = outer.loc[outer["target"].notna() & outer["experiment_id"].ne(test)]
    inner_train = nested.loc[nested["experiment_id"].ne(inner)]
    inner_validation = nested.loc[nested["experiment_id"].eq(inner)]
    for label, rows in (
        ("inner train", inner_train),
        ("inner validation", inner_validation),
        ("outer train", outer_train),
    ):
        if not (rows["is_frame"] | rows["is_knee"]).any():
            raise ValueError(f"{test}: no teacher-covered side labels in {label}")
    economic, relation, nonvisual = METHODS[args.method]
    result = train_pareto_fold(
        inner_train,
        inner_validation,
        outer_train,
        outer.loc[outer["experiment_id"].eq(test)],
        use_economic_context=economic,
        use_pareto_relation=relation,
        nonvisual=nonvisual,
        seed=args.seed,
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
    )
    for name in ("predictions", "losses", "pair_metrics"):
        result[name]["heldout_experiment"] = test
        result[name]["method"] = args.method
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(result, stream)
    temporary.replace(path)
    print(f"{args.method}: completed {test}", flush=True)
    return result


def train(args: argparse.Namespace) -> None:
    from dataset_tools.cycle_metadata import read_catalog

    require_matching_extrapolation_setting(args.data, args.allow_extrapolation)
    settings = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in {"n_jobs", "heldout_experiment", "dry_run", "wandb_project"}
    }
    settings["pareto_cv_cycles"] = sorted(
        pareto_cv_ready_cycles(
            read_catalog(args.dataset), allow_extrapolation=args.allow_extrapolation
        )
    )
    save_settings(args.output / "settings.json", settings)
    (args.output / "folds").mkdir(exist_ok=True)
    folds = json.loads((args.data / "folds.json").read_text())
    if args.heldout_experiment:
        folds = {args.heldout_experiment: folds[args.heldout_experiment]}
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        Parallel()(delayed(_train_fold)(args, test, inner) for test, inner in folds.items())
    results = []
    for path in sorted((args.output / "folds").glob("*.pkl")):
        with path.open("rb") as stream:
            results.append(pickle.load(stream))  # noqa: S301 - local run-owned checkpoints
    pd.concat([r["predictions"] for r in results]).to_parquet(
        args.output / "predictions.parquet", index=False
    )
    for name in ("losses", "pair_metrics"):
        pd.concat([r[name] for r in results]).to_csv(args.output / f"{name}.csv", index=False)
    log_wandb(args.wandb_project, args.output.name, settings, completed_folds=len(results))


def overview(args: argparse.Namespace) -> None:
    """Redraw the overview evidence from frozen result tables."""
    if not args.runs or len(args.runs) != 7:
        raise ValueError(
            "--action overview requires --runs POLICY PROBE NEURAL CROSS_INPUT "
            "OUTCOME STATE TRANSFER"
        )
    from plots.pareto_learning import render_overview_figures

    render_overview_figures(args.runs, args.output / "figures")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.task == "pareto" and args.action is None:
        parser.error("--action is required for the historical Pareto task")
    if args.dry_run:
        print(vars(args))
        return
    if args.action == "screen":
        from image_models.relative_cop import screen_curves

        screen_curves(args)
        return
    if args.action == "audit":
        if args.task != "cop-classification":
            parser.error("--action audit requires --task cop-classification")
        from image_models.relative_cop import audit

        audit(args)
        return
    if args.action == "develop":
        if args.task != "cop-classification":
            parser.error("--action develop requires --task cop-classification")
        if args.rgb not in ("off", "on"):
            parser.error("development requires --rgb off or on")
        from image_models.cop_development import develop

        develop(args)
        return
    if args.action == "compare-development":
        if args.task != "cop-classification":
            parser.error("--action compare-development requires --task cop-classification")
        if not args.runs:
            parser.error("--action compare-development requires --runs")
        from image_models.cop_development import compare_development

        compare_development(args.runs, args.output, seed=args.seed)
        return
    if args.action == "refit-development":
        if args.task != "cop-classification":
            parser.error("--action refit-development requires --task cop-classification")
        from image_models.cop_development import refit_development

        refit_development(args)
        return
    if args.action == "evaluate-frozen-development":
        if args.task != "cop-classification":
            parser.error(
                "--action evaluate-frozen-development requires --task cop-classification"
            )
        if not args.runs or len(args.runs) != 2:
            parser.error(
                "--action evaluate-frozen-development requires --runs "
                "DELTA_RGB_DEVELOPMENT_RUN HISTORY_SENSOR_DEVELOPMENT_RUN"
            )
        from image_models.cop_development import evaluate_frozen_development

        evaluate_frozen_development(args)
        return
    if args.action == "compare-frozen-policies":
        if args.task != "cop-classification":
            parser.error("--action compare-frozen-policies requires --task cop-classification")
        if not args.runs or len(args.runs) != 4:
            parser.error(
                "--action compare-frozen-policies requires --runs FROZEN_RETROSPECTIVE "
                "LEGACY_SENSOR LEGACY_RGB CHEN_BINARY"
            )
        if args.output == Path("output/test/pareto_boundary/current"):
            args.output = Path("output/test/cop_five_policy_comparison")
        from image_models.relative_cop import compare_frozen_policies

        compare_frozen_policies(args)
        return
    if args.action == "compare-stopping-losses":
        if args.task != "cop-classification":
            parser.error(
                "--action compare-stopping-losses requires --task cop-classification"
            )
        from image_models.stopping_loss_comparison import run

        if args.output == Path("output/test/pareto_boundary/current"):
            args.output = Path("output/image_models/stopping_loss_comparison")
        run(args)
        return
    if args.task in ("relative-cop-regression", "effective-cop-binary", "cop-classification-regression", "cop-classification"):
        from image_models.relative_cop import run

        if args.rgb_projection and args.rgb != "on":
            parser.error("--rgb-projection requires --rgb on")
        if args.task in ("cop-classification", "cop-classification-regression"):
            args.regression_architecture = args.task
            if args.classification_label == "after-optimum" and args.task != "cop-classification":
                parser.error("after-optimum labels require pure classification")
            if not 0 <= args.near_optimal_epsilon < 1 or args.regression_weight < 0:
                parser.error("epsilon must be in [0,1) and regression weight nonnegative")
            if args.regression_suite or args.peak_weighted:
                parser.error("joint task uses single-timestamp BCE plus global MSE")
        if args.task == "effective-cop-binary":
            args.regression_architecture = "dinov2-binary"
            args.rgb = "on"
            if args.action != "evaluate":
                args.trigger_threshold = .5
            args.processing_seconds = 30
            if args.regression_suite or args.peak_weighted:
                parser.error("binary baseline uses one unweighted-by-class RGB recipe")
            if args.reference_run == Path("output/test/pareto_boundary_outcome_v1"):
                args.reference_run = Path("output/image_models/relative_cop_tref")
            default_cache = Path(
                "output/image_models/_cache/dinov2_vits14_r256_c224_front_v1/cycles"
            )
            if args.rgb_cache == default_cache:
                args.rgb_cache = args.dataset.resolve().parent / args.rgb_cache
            if args.output == Path("output/test/pareto_boundary/current"):
                args.output = Path("output/image_models/dinov2_binary_tref")

        if args.output == Path("output/test/pareto_boundary/current"):
            args.output = Path("output/image_models") / (
                "pinn4soh_cop" if args.regression_suite else
                f"{args.regression_architecture}_{args.rgb}_official"
                + ("_peak" if args.peak_weighted else "")
            )
        if args.figure_output is None:
            args.figure_output = Path("output/test") / args.output.name
        if args.action == "render-cycles":
            if args.task != "effective-cop-binary":
                parser.error("--action render-cycles requires --task effective-cop-binary")
            from plots.pareto_learning import render_binary_cycle_probabilities

            render_binary_cycle_probabilities(
                args.output,
                args.dataset,
                args.decision_run,
                args.figure_output,
                n_jobs=args.n_jobs,
            )
        elif args.action == "evaluate":
            from image_models.relative_cop import evaluate_online

            evaluate_online(args)
        elif args.regression_suite:
            from image_models.relative_cop import run_suite

            run_suite(args)
        else:
            run(args)
        return
    {"prepare": prepare, "fit": fit, "train": train,
     "represent": represent, "heads": heads, "neural": neural,
     "evaluate": evaluate, "cross-input": cross_input, "vcnet": vcnet,
     "probe": probe, "state-transfer": state_transfer, "refit": refit,
     "overview": overview}[args.action](args)


if __name__ == "__main__":
    main()
