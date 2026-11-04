"""Stopping-model storage and per-checkpoint cycle membership."""

import argparse
import json
import pickle
import shutil
from pathlib import Path

from output_layout import PROJECT, experiment_root

ROOT = Path(__file__).resolve().parents[1] / "output" / "90_复现材料" / "01_视觉模型训练"


def checkpoint_directory(run):
    return experiment_root(run) / "checkpoints"


def checkpoint_path(run, name):
    run = Path(run)
    name = Path(name).name
    target = checkpoint_directory(run) / Path(name).stem / name
    # Historical nested runs remain readable until explicitly migrated.
    for path in [target, run / name, run / "checkpoints" / name]:
        if path.exists():
            return path
    return target


def checkpoint_files(run, pattern="*.pkl"):
    run = Path(run)
    paths = sorted(
        set(
            [
                *checkpoint_directory(run).glob(f"*/{pattern}"),
                *run.glob(pattern),
                *(run / "checkpoints").glob(pattern),
            ]
        )
    )
    if len({p.name for p in paths}) != len(paths):
        raise ValueError(f"Duplicate checkpoint identities in {run}")
    return sorted(paths)


def checkpoint_camera(path):
    """Read the saved camera; historical deployment versions already name it explicitly."""
    path = Path(path)
    with path.open("rb") as stream:
        saved = pickle.load(stream)
    camera = saved["checkpoint"].get("camera_role")
    if camera is None:
        versions = saved["cycle_metrics"].version.dropna().unique()
        if len(versions) == 1 and versions[0].endswith("_deployment"):
            camera = versions[0].removesuffix("_deployment")
    if camera is None:
        root = (
            path.parent.parent.parent
            if path.parent.parent.name == "checkpoints"
            else experiment_root(path.parent)
        )
        for settings in (root / "training/settings.json", root / "settings.json"):
            if settings.is_file():
                camera = json.loads(settings.read_text()).get("camera_role")
                break
    if camera not in {"front", "left", "left_close", "top", "top_close", "extreme"}:
        raise ValueError(
            f"Cannot establish training camera for {path}; retain its training/settings.json"
        )
    return camera


def organize_fold_history(run):
    """Keep inner-fold search histories beside the model they selected."""
    moved = 0
    models = checkpoint_files(run)
    for model in models:
        if "_exp_" not in model.stem:
            continue
        version, fold = model.stem.rsplit("_exp_", 1)
        experiment, seed = fold.rsplit("_seed", 1)
        patterns = [f"inner_{version}_exp_{experiment}*_seed{seed}.csv"]
        if sum(p.stem.endswith("_exp_" + fold) for p in models) == 1:
            patterns += [
                f"exp_{experiment}_seed{seed}_trials.csv",
                f"exp_{experiment}_seed{seed}.sqlite3",
            ]
        for pattern in patterns:
            for path in Path(run).glob(pattern):
                target = model.parent / path.name
                if target.exists():
                    raise FileExistsError(target)
                path.rename(target)
                moved += 1
    return moved


def prune_checkpoints(keep_runs):
    """Explicitly retain named experiments; delete only obsolete checkpoint folders."""
    keep = {Path(run).name for run in keep_runs}
    available = {
        p.name: p for p in ROOT.iterdir() if p.is_dir() and list(p.glob("checkpoints/*/*.pkl"))
    }
    if not keep or keep - set(available):
        raise ValueError(
            f"Keep list must name existing checkpoint experiments: {sorted(keep - set(available))}"
        )
    removed = 0
    for name, run in available.items():
        if name in keep:
            continue
        for model in list(run.glob("checkpoints/*/*.pkl")):
            shutil.rmtree(model.parent)
            removed += 1
        (run / "index.html").unlink(missing_ok=True)
        if not list(run.iterdir()):
            run.rmdir()
    return removed


def organize(runs, *, dataset=None):
    """Move flat checkpoints and reconstruct historical train/validation/test roles."""
    from image_models.cycle_selection import screen_published_cycles

    groups = {}
    for run in runs:
        run = Path(run)
        settings = json.loads((run / "settings.json").read_text())
        definition = settings["data_definition"]
        root = Path(dataset) if dataset is not None else Path(settings["dataset"])
        camera = settings.get("camera_role", "front")
        key = (str(root.resolve()), camera, json.dumps(definition, sort_keys=True))
        if key not in groups:
            screening = screen_published_cycles(root, camera, definition=definition)
            groups[key] = screening.loc[screening.selected, ["cycle_name", "experiment_id"]]
        cohort = groups[key]
        if settings.get("cycle_cohort"):
            names = set(settings["cycle_names"])
            cohort = cohort.loc[cohort.cycle_name.isin(names)].copy()
        for source in checkpoint_files(run):
            saved = pickle.load(source.open("rb"))
            model = saved["checkpoint"]
            predicted = saved["predictions"]
            roles = cohort.copy()
            test = set(predicted.loc[predicted.split.eq("test"), "cycle_name"])
            validation = saved.get(
                "validation_predictions", predicted.loc[predicted.split.eq("validation")]
            )
            val = set(validation.cycle_name)
            inner = model.get("inner_validation_folds")
            if validation.empty:
                experiments = model.get("inner_validation_experiments", [])
                if experiments and isinstance(experiments[0], list):
                    inner = experiments
                    experiments = [e for fold in inner for e in fold]
                val = set(roles.loc[roles.experiment_id.isin(experiments), "cycle_name"])
            train_experiments = model.get(
                "training_experiments",
                set(roles.experiment_id)
                - set(validation.experiment_id)
                - set(predicted.loc[predicted.split.eq("test"), "experiment_id"]),
            )
            roles["final_training"] = (
                roles.cycle_name.isin(model["training_cycles"])
                if "training_cycles" in model
                else roles.experiment_id.isin(train_experiments)
            )
            roles["inner_validation"] = roles.cycle_name.isin(val)
            roles["outer_test"] = roles.cycle_name.isin(test)
            selection = model.get("selection_training_experiments")
            roles["selection_training"] = (
                roles.experiment_id.isin(selection)
                if selection is not None
                else roles.final_training & ~roles.inner_validation
            )
            if inner:
                roles["selection_training"] = roles.final_training
                for index, experiments in enumerate(inner):
                    roles[f"inner_fold_{index}"] = roles.experiment_id.map(
                        lambda e: (
                            "validation"
                            if e in experiments
                            else "training"
                            if e in train_experiments
                            else "outer_test"
                        )
                    )
            roles["membership_source"] = (
                "saved_cycle_ids"
                if "training_cycles" in model
                else "saved_experiment_partition_and_original_admission"
            )
            roles = roles.loc[
                roles[
                    ["final_training", "inner_validation", "outer_test", "selection_training"]
                ].any(axis=1)
            ]
            if (roles.final_training & roles.outer_test).any():
                raise ValueError(f"Training/test overlap: {source}")
            folder = checkpoint_directory(run) / source.stem
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / source.name
            if source != target:
                if target.exists():
                    raise FileExistsError(target)
                source.rename(target)
            for suffix in ["_history.csv", "_refit_history.csv"]:
                history = run / (source.stem + suffix)
                if history.exists():
                    history.rename(folder / history.name)
            roles.to_csv(folder / "cycles.csv", index=False)
        print(f"[organize] {run.name}: {len(checkpoint_files(run))} checkpoints", flush=True)
        organize_fold_history(run)


def main():
    from image_models.comparisons import write_ablation_tables
    from image_models.evaluation import evaluate_external
    from tools.result_index import write_output_index

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate one frozen deployment weight")
    parser.add_argument("--camera-role", default="front")
    parser.add_argument(
        "--prune-others",
        action="store_true",
        help="Delete checkpoint folders outside the explicit --runs keep list; retain experiment results",
    )
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument(
        "--ablation-output",
        type=Path,
        help="Collect explicit --runs into plot-ready score/settings and epoch tables",
    )
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument(
        "--external-evaluation",
        type=Path,
        help="Evaluate saved models on explicitly named new experiments",
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--experiments", nargs="+")
    args = parser.parse_args()
    if args.external_evaluation:
        if bool(args.runs) == bool(args.checkpoint) or not args.experiments:
            parser.error(
                "--external-evaluation requires --experiments and either --runs or --checkpoint"
            )
        for run in args.runs or [None]:
            output = args.external_evaluation
            if args.runs and len(args.runs) > 1:
                output = output / Path(run).name
            evaluate_external(
                run,
                args.dataset or (PROJECT / "dataset"),
                args.experiments,
                output,
                n_jobs=args.n_jobs,
                checkpoint=args.checkpoint,
                camera_role=args.camera_role,
            )
    elif args.ablation_output:
        if not args.runs:
            parser.error("--ablation-output requires --runs")
        write_ablation_tables(args.runs, args.ablation_output)
    elif args.index_only:
        write_output_index()
    elif args.prune_others:
        if not args.runs:
            parser.error("--prune-others requires an explicit --runs keep list")
        print(f"Deleted {prune_checkpoints(args.runs)} obsolete checkpoint folders")
        write_output_index()
    elif args.runs:
        organize(args.runs, dataset=args.dataset)
    else:
        parser.error("Provide --runs or --index-only")


if __name__ == "__main__":
    main()
