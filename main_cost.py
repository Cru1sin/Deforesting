"""Calculate and compare Dataset-native defrost cost curves."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pandas as pd
from joblib import parallel_config
from sklearn.utils.parallel import Parallel, delayed

from cost import cho, cost_function_v1, cost_function_v2_5, cost_function_v2_6_8
from cost.boundaries import catalog_exclusion_reason, clean_anchor_exclusion_reason
from cost.energy_models import load_parameters
from cost.fit_v2_6_8 import (
    MODEL_FEATURES,
    OUTCOME_TARGETS,
    assemble_target_artifact,
    fit_full_outcome,
    fit_outcome_fold,
    load_artifacts,
    mean_outcome_artifact,
)
from cost.v2_6_8_data import build_event_table, candidate_cohort
from cost.validate_v2_6_8 import bootstrap_minima, build_validation_table
from dataloader import DatasetLoader
from plots.cost import generate_cost_function_figures

COST_MODULES: dict[str, ModuleType] = {
    "v1": cost_function_v1,
    "v2.5": cost_function_v2_5,
    "v2.6.8": cost_function_v2_6_8,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", required=True, choices=("calculate", "fit", "compare", "policy")
    )
    parser.add_argument("--cost", choices=tuple(COST_MODULES), default="v1")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--cycles", nargs="*")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--results", nargs="+", type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--candidate-step-seconds", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument(
        "--allow-extrapolation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--integration-protocol", choices=("historical_reconstruction", "strict_causal")
    )
    parser.add_argument("--state-protocol", choices=("historical_interpolation", "strict_causal"))
    parser.add_argument(
        "--heating-heat-model", choices=("measured_unit_heat", "measured_water_heat")
    )
    parser.add_argument(
        "--transition-energy-model",
        choices=(
            "pe_quadratic_plus_fixed_recovery",
            "pe_quadratic",
            "experiment_mean",
            "ticket_ridge_static5",
            "ticket_ridge_physical6",
            "ticket_ridge_dynamic8",
        ),
    )
    parser.add_argument(
        "--transition-heat-model",
        choices=(
            "zero_transition_heat",
            "linear_qprep_plus_signed_quadratic_qd",
            "experiment_mean",
            "ticket_ridge_static5",
            "ticket_ridge_physical6",
            "ticket_ridge_dynamic8",
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="show the selected action only")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    arguments = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(arguments)
    if args.action == "fit":
        return _fit_v268(args, arguments)
    if args.action == "policy":
        return _run_policy(args, arguments)
    if args.action == "compare":
        if not args.results:
            raise ValueError("compare requires --results run directories")
        if args.dry_run:
            print(
                f"Compare dry-run: results={args.results}, "
                f"output={args.output_root / 'plots'}"
            )
            return 0
        generate_cost_function_figures(
            args.results,
            DatasetLoader(args.dataset),
            args.output_root / "plots",
            overwrite=args.overwrite,
        )
        print(f"Comparison written: {args.output_root / 'plots'}")
        return 0

    module = COST_MODULES[args.cost]
    recipe = _recipe(module, args)
    run = _run_directory(args.output_root, recipe)
    if run.exists() and not args.overwrite:
        raise FileExistsError(f"run directory exists; pass --overwrite: {run}")
    loader = DatasetLoader(args.dataset)
    if args.cost == "v2.6.8":
        artifact = load_artifacts()
        model_name = str(recipe["transition_energy_model"])
        heat_model_name = str(recipe["transition_heat_model"])
        if model_name not in artifact.get("models", {}) or heat_model_name not in artifact.get(
            "models", {}
        ):
            raise ValueError("V2.6.8 artifact does not contain the selected component model")
        experiment_folds = set(artifact["models"][model_name]["energy"]["folds"])
        experiment_folds &= set(artifact["models"][heat_model_name]["heat"]["folds"])
        cycles = _cycle_names(loader, args.cycles, experiment_folds)
        parameters: dict[str, Any] = {}
    else:
        parameters = load_parameters()
        if not {"pe_quadratic", "v1", "v2.5"} <= parameters.keys():
            raise ValueError("empirical parameters are incomplete")
        cycles = _cycle_names(loader, args.cycles, set(parameters["pe_quadratic"]))
    if (
        recipe["transition_energy_model"] == "pe_quadratic_plus_fixed_recovery"
        and "fixed_recovery_electricity_kwh" not in parameters["v1"]
    ):
        raise ValueError("V1 fixed recovery parameter is missing")
    if (
        recipe["transition_heat_model"] == "linear_qprep_plus_signed_quadratic_qd"
        and not {
            "preparation_heat",
            "defrost_heat",
        }
        <= parameters["v2.5"].keys()
    ):
        raise ValueError("V2.5 transition heat parameters are incomplete")
    if args.dry_run:
        print(
            "Dry-run OK: recipe, parameters, variant, output, and "
            f"{len(cycles)} metadata-eligible cycle(s) checked; "
            "raw clean-anchor gate deferred; "
            f"integration_protocol={recipe['integration_protocol']}; "
            f"state_protocol={recipe['state_protocol']}"
        )
        return 0

    cycles, anchor_excluded = _clean_anchor_cycles(loader, cycles, explicit=bool(args.cycles))
    print(f"Selected {len(cycles)} cycle(s); excluded {anchor_excluded} by raw clean-anchor gate")
    table = module.calculate(loader, cycles, recipe)
    _write_run(run, table, recipe, arguments, overwrite=args.overwrite)
    print(f"Cost result written: {run}")
    return 0


def _write_run(
    run: Path,
    table: pd.DataFrame,
    recipe: dict[str, object],
    arguments: list[str],
    *,
    overwrite: bool,
) -> None:
    run.mkdir(parents=True, exist_ok=True)
    table.to_csv(run / "cost.csv", index=False)
    cycles_dir = run / "cycles"
    cycles_dir.mkdir(exist_ok=True)
    if overwrite:
        for stale in cycles_dir.glob("*.csv"):
            stale.unlink()
    for cycle_name, cycle in table.groupby("cycle_name", sort=False):
        cycle.to_csv(cycles_dir / f"{cycle_name}.csv", index=False)
    (run / "recipe.json").write_text(
        json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run / "command.txt").write_text(
        shlex.join(["uv", "run", "python", "main_cost.py", *arguments]) + "\n",
        encoding="utf-8",
    )


def _run_policy(args: argparse.Namespace, arguments: list[str]) -> int:
    if args.parameters is None:
        raise ValueError("policy requires --parameters ARTIFACT.json")
    if not args.variant:
        raise ValueError("policy requires --variant NAME")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(args.variant)):
        raise ValueError("variant may contain only letters, numbers, dot, underscore, and hyphen")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be greater than zero")
    if args.candidate_step_seconds <= 0:
        raise ValueError("--candidate-step-seconds must be greater than zero")

    artifact = load_artifacts(args.parameters)
    model_name = str(cho.DEFAULT_POLICY_RECIPE["outcome_model"])
    model_set = artifact.get("models", {}).get(model_name)
    required = tuple(OUTCOME_TARGETS)
    if not isinstance(model_set, dict) or not set(required) <= set(model_set):
        raise ValueError(f"policy requires {model_name} with all four outcome targets")
    experiment_folds = set.intersection(
        *(set(model_set[name].get("folds", {})) for name in required)
    )
    recipe = {
        **cho.DEFAULT_POLICY_RECIPE,
        "working_mode": "offline_policy",
        "variant": str(args.variant),
        "candidate_step_seconds": args.candidate_step_seconds,
        "allow_extrapolation": args.allow_extrapolation,
    }
    run = args.output_root / "cost" / "policy" / str(args.variant)
    if run.exists() and not args.overwrite:
        raise FileExistsError(f"run directory exists; pass --overwrite: {run}")
    loader = DatasetLoader(args.dataset)
    cycles = _cycle_names(loader, args.cycles, experiment_folds)
    if args.dry_run:
        print(
            f"Policy dry-run: variant={args.variant}, candidate_step_seconds="
            f"{args.candidate_step_seconds}, n_jobs={args.n_jobs}, "
            f"allow_extrapolation={args.allow_extrapolation}; "
            f"{len(cycles)} metadata-eligible cycle(s) checked"
        )
        return 0

    cycles, anchor_excluded = _clean_anchor_cycles(loader, cycles, explicit=bool(args.cycles))
    print(f"Selected {len(cycles)} cycle(s); excluded {anchor_excluded} by raw clean-anchor gate")
    tables = []
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        results = Parallel(
            return_as="generator_unordered",
            batch_size="auto",
            pre_dispatch="2*n_jobs",
        )(
            delayed(cho.calculate_cycle)(
                loader,
                cycle_name,
                artifact,
                step_seconds=args.candidate_step_seconds,
                allow_extrapolation=args.allow_extrapolation,
            )
            for cycle_name in cycles
        )
        for completed, table in enumerate(results, start=1):
            tables.append(table)
            print(f"Policy cycles complete: {completed}/{len(cycles)}")
    combined = (
        pd.concat(tables, ignore_index=True)
        .sort_values(["cycle_name", "candidate_time"], kind="stable")
        .reset_index(drop=True)
        if tables
        else pd.DataFrame()
    )
    _write_run(run, combined, recipe, arguments, overwrite=args.overwrite)
    print(f"Policy result written: {run}")
    return 0


def _recipe(module: Any, args: argparse.Namespace) -> dict[str, object]:
    recipe = dict(module.DEFAULT_RECIPE)
    recipe["variant"] = args.variant
    for argument, key in (
        (args.integration_protocol, "integration_protocol"),
        (args.state_protocol, "state_protocol"),
        (args.heating_heat_model, "heating_heat_model"),
        (args.transition_energy_model, "transition_energy_model"),
        (args.transition_heat_model, "transition_heat_model"),
    ):
        if argument is not None:
            recipe[key] = argument
    if args.heating_heat_model is not None:
        recipe["heat_basis"] = (
            "unit" if args.heating_heat_model == "measured_unit_heat" else "water"
        )
    return cast(dict[str, object], module.validate_recipe(recipe))


def _run_directory(output_root: Path, recipe: dict[str, object]) -> Path:
    variant = recipe["variant"]
    if variant is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", str(variant)):
        raise ValueError("variant may contain only letters, numbers, dot, underscore, and hyphen")
    name = str(recipe["base_cost"]) if variant is None else f"{recipe['base_cost']}__{variant}"
    return output_root / "cost" / name


def _cycle_names(
    loader: Any, requested: list[str] | None, parameter_experiments: set[str]
) -> list[str]:
    catalog = loader.list_cycles(statuses={"valid"})
    available = [str(value) for value in catalog["cycle_name"].tolist()]
    if requested is None or not requested:
        selected = available
    else:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"unknown or invalid cycles: {', '.join(missing)}")
        selected = requested
    records = {str(row["cycle_name"]): row.to_dict() for _, row in catalog.iterrows()}
    eligible = []
    for cycle_name in selected:
        reason = catalog_exclusion_reason(records[cycle_name], parameter_experiments)
        if reason is not None:
            if requested:
                raise ValueError(f"{cycle_name} excluded: {reason}")
            continue
        eligible.append(cycle_name)
    if not eligible:
        raise ValueError("no metadata-eligible cycles selected")
    return eligible


def _clean_anchor_cycles(
    loader: Any, cycles: list[str], *, explicit: bool
) -> tuple[list[str], int]:
    selected = []
    columns = [
        "timestamp",
        "water_flow",
        "water_in_temperature",
        "water_out_temperature",
        "power_total",
    ]
    for cycle_name in cycles:
        frame = loader.load_cycle_original(cycle_name, columns=columns)
        reason = clean_anchor_exclusion_reason(frame, loader.get_cycle_record(cycle_name))
        if reason is not None:
            if explicit:
                raise ValueError(f"{cycle_name} excluded: {reason}")
            continue
        selected.append(cycle_name)
    if not selected:
        raise ValueError("no cycles pass the raw clean-anchor gate")
    return selected, len(cycles) - len(selected)


def _fit_v268(args: argparse.Namespace, arguments: list[str]) -> int:  # noqa: C901
    """Fit the review-only candidate artifact with explicit outer loops."""
    if args.cost != "v2.6.8":
        raise ValueError("fit is available only for --cost v2.6.8")
    if not args.variant:
        raise ValueError("V2.6.8 fit requires --variant NAME")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(args.variant)):
        raise ValueError("variant may contain only letters, numbers, dot, underscore, and hyphen")
    run = args.output_root / "cost" / "fit" / str(args.variant)
    if run.exists() and not args.overwrite:
        raise FileExistsError(f"fit directory exists; pass --overwrite: {run}")
    if args.dry_run:
        print(
            f"Fit V2.6.8: variant={args.variant}, models={list(MODEL_FEATURES)}, "
            f"bootstrap_replicates=200, bootstrap_seed=268, output={run}"
        )
        return 0
    loader = DatasetLoader(args.dataset)
    events = build_event_table(loader)
    valid = events.loc[events["event_valid"].fillna(False)].copy()
    if valid.empty:
        raise ValueError("V2.6.8 fit has no valid observed events")
    experiments = sorted(valid["experiment_id"].astype(str).unique())
    mean_models: dict[str, dict[str, object]] = {}
    for target_name, target in OUTCOME_TARGETS.items():
        folds = {
            heldout: mean_outcome_artifact(
                valid.loc[~valid["experiment_id"].astype(str).eq(heldout)], target
            )
            for heldout in experiments
        }
        mean_models[target_name] = {
            "artifact_version": "v2.6.8",
            "target": target,
            "feature_order": [],
            "support_policy": "all_candidates_for_experiment_balanced_mean",
            "folds": folds,
            "full_data_model": mean_outcome_artifact(valid, target),
        }
    artifact: dict[str, Any] = {
        "artifact_version": "v2.6.8",
        "fit_variant": str(args.variant),
        "bootstrap_seed": 268,
        "bootstrap_replicates": 200,
        "models": {
            "experiment_mean": mean_models,
        },
    }
    for model_name, features in MODEL_FEATURES.items():
        artifact["models"][model_name] = {}
        for target_name, target in OUTCOME_TARGETS.items():
            folds = {
                heldout: fit_outcome_fold(valid, heldout, features, target)
                for heldout in experiments
            }
            artifact["models"][model_name][target_name] = assemble_target_artifact(
                target, features, folds, fit_full_outcome(valid, features, target)
            )

    validation = build_validation_table(events, artifact)
    cohort, candidate_rows = candidate_cohort(loader, set(experiments))
    curves = pd.concat(
        [
            cost_function_v2_6_8.calculate_cycle(
                loader, cycle_name, cost_function_v2_6_8.DEFAULT_RECIPE, artifact
            )
            for cycle_name in cohort
        ],
        ignore_index=True,
    )
    dynamic = artifact["models"]["ticket_ridge_dynamic8"]
    bootstrap = bootstrap_minima(curves, events, dynamic["energy"], dynamic["heat"])
    recipe = dict(cost_function_v2_6_8.DEFAULT_RECIPE)
    run.mkdir(parents=True, exist_ok=True)
    (run / "command.txt").write_text(
        "uv run python main_cost.py " + shlex.join(arguments) + "\n", encoding="utf-8"
    )
    (run / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True), encoding="utf-8")
    events.to_csv(run / "events.csv", index=False)
    validation.to_csv(run / "validation.csv", index=False)
    bootstrap.to_csv(run / "bootstrap.csv", index=False)
    (run / "params_candidate.json").write_text(
        json.dumps(artifact, sort_keys=True, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"Fit candidate written: {run}; {len(valid)} valid event(s), "
        f"{len(events) - len(valid)} exclusion(s), {len(cohort)} candidate cycle(s), "
        f"{candidate_rows} candidate row(s); not promoted"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
