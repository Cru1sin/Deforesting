"""Prepare shared causal data, fit fold teachers, and compare direct Pareto decisions."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed, parallel_config

METHODS = {
    "baseline": (False, False, False),
    "economic": (True, False, False),
    "relation": (False, True, False),
    "combined": (True, True, False),
    "nonvisual": (True, False, True),
}


def fold_exclusions(experiments: list[str]) -> dict[str, str]:
    """Choose an inner experiment by date order only, without inspecting targets."""
    ordered = sorted(set(experiments))
    return {
        test: (train := [value for value in ordered if value != test])[len(train) // 2]
        for test in ordered
    }


def save_settings(path: Path, settings: dict) -> None:
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("settings changed; use a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "fit", "train"), required=True)
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
    parser.add_argument("--method", choices=METHODS, default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maximum-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--heldout-experiment")
    parser.add_argument("--wandb-project")
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
            "data_schema": "pareto_boundary_v1",
            "sources": {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in sources},
            "sensor_window_minutes": 5,
            "candidate_step_seconds": 10,
            "rgb_policy": "latest_same_cycle_front_max_age_45s",
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
                loader, cycle, cache_dir=args.data.parent / "pareto_measured_stat6_v1"
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


def _fit_teacher(data: Path, excluded: tuple[str, ...]) -> None:
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
    tables = [apply_fold_teacher(cycle, parameters) for _, cycle in base.groupby("cycle_name")]
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


def fit(args: argparse.Namespace) -> None:
    folds = json.loads((args.data / "folds.json").read_text())
    exclusions = sorted(
        {
            tuple(sorted(group))
            for test, inner in folds.items()
            for group in ((test,), (test, inner))
        }
    )
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        Parallel()(delayed(_fit_teacher)(args.data, excluded) for excluded in exclusions)


def load_fold_rows(data: Path, rgb_cache: Path, excluded: tuple[str, ...]) -> pd.DataFrame:
    from image_models.image_features import load_dinov2_feature_cache

    base = pd.read_parquet(data / "base.parquet")
    teacher = pd.read_parquet(data / "teachers" / ("__".join(sorted(excluded)) + ".parquet"))
    rows = base.drop(columns=base.columns.intersection(teacher.columns).difference(["row_id"]))
    rows = rows.merge(teacher, on="row_id", validate="one_to_one")
    keys = ["cycle_name", "camera_role", "file_name"]
    images = rows.loc[rows["file_name"].notna(), keys].drop_duplicates()
    features = load_dinov2_feature_cache(images, rgb_cache, "dinov2")
    rows = rows.merge(features, on=keys, how="left", validate="many_to_one")
    missing = rows["dinov2_000"].isna()
    if (missing & rows["is_frame"]).any():
        raise ValueError(f"{int(missing.sum())} rows lack RGB; finish cache/coverage review first")
    # Missing past RGB cannot supply a rank/anchor input; it never moves the teacher.
    return rows.loc[~missing].copy()


def _train_fold(args: argparse.Namespace, test: str, inner: str) -> dict:
    import torch

    from image_models.pareto_learning import train_pareto_fold

    path = args.output / "folds" / f"{test}.pkl"
    if path.exists():
        with path.open("rb") as stream:
            return pickle.load(stream)  # noqa: S301 - local run-owned checkpoint
    torch.set_num_threads(1)
    outer = load_fold_rows(args.data, args.rgb_cache, (test,))
    nested = load_fold_rows(args.data, args.rgb_cache, (test, inner))
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
    settings = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in {"n_jobs", "heldout_experiment", "dry_run", "wandb_project"}
    }
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
    if args.wandb_project:
        import wandb

        with wandb.init(project=args.wandb_project, name=args.output.name, config=settings) as run:
            run.log({"completed_folds": len(results)})


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        print(vars(args))
        return
    {"prepare": prepare, "fit": fit, "train": train}[args.action](args)


if __name__ == "__main__":
    main()
