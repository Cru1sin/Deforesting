"""Select a defrost time from cycle COP and heating-rate Pareto trade-offs."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import parallel_config
from sklearn.utils.parallel import Parallel, delayed

from dataset_tools import DatasetLoader
from defrost_decision.candidate_quantities import build_candidate_quantities
from defrost_decision.candidate_times import clean_anchor_cycles, metadata_eligible_cycles
from defrost_decision.pareto_selection import select_cop_heating_rate_pareto_knee
from defrost_decision.performance_objectives import (
    add_single_objective_optima,
    calculate_performance_objectives,
)
from defrost_event_models.ridge_models import (
    OUTCOME_TARGETS,
    load_defrost_event_models,
    predict_with_event_model,
)

OUTCOME_MODEL = "ridge_dynamic_state_8"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--cycles", nargs="*")
    parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("defrost_event_models/parameters/released_ridge_models.json"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--run-name", default="current")
    parser.add_argument("--candidate-step-seconds", type=int, default=10)
    parser.add_argument(
        "--prediction-mode",
        choices=("cross-fitted", "full-model"),
        default="cross-fitted",
        help="held-out folds for retrospective replay or the full model for new experiments",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _predict_additional_outcome(
    candidates: pd.DataFrame,
    model: dict[str, Any],
    experiment_id: str,
    prediction_mode: str,
    *,
    value_column: str,
    field_name: str,
) -> None:
    prediction = predict_with_event_model(
        model, candidates, experiment_id, prediction_mode=prediction_mode
    )
    candidates[value_column] = prediction["prediction"].to_numpy()
    candidates[f"{field_name}_training_distance"] = prediction["support_distance"].to_numpy()
    candidates[f"{field_name}_prediction_available"] = np.isfinite(prediction["prediction"])
    candidates[f"{field_name}_in_training_domain"] = (
        prediction["support_distance"].le(prediction["support_threshold"]).to_numpy()
    )


def calculate_cycle(
    loader: Any,
    cycle_name: str,
    models: Mapping[str, Any],
    *,
    candidate_step_seconds: int,
    allow_extrapolation: bool,
    prediction_mode: str = "cross-fitted",
) -> pd.DataFrame:
    """Run the visible candidate quantities -> objectives -> Pareto sequence."""
    model_set = models["models"][OUTCOME_MODEL]
    candidates = build_candidate_quantities(
        loader,
        cycle_name,
        models,
        candidate_step_seconds=candidate_step_seconds,
        prediction_mode=prediction_mode,
    )
    record = loader.get_cycle_record(cycle_name)
    experiment_id = str(record["experiment_id"])
    _predict_additional_outcome(
        candidates,
        model_set["event_compressor_electricity"],
        experiment_id,
        prediction_mode,
        value_column="defrost_event_compressor_electricity_kwh",
        field_name="defrost_event_compressor_electricity",
    )
    _predict_additional_outcome(
        candidates,
        model_set["event_duration"],
        experiment_id,
        prediction_mode,
        value_column="defrost_event_duration_minutes",
        field_name="defrost_event_duration",
    )
    boundaries = record.get("boundaries")
    boundary_source = boundaries if isinstance(boundaries, Mapping) else record
    objectives = calculate_performance_objectives(
        candidates, allow_model_extrapolation=allow_extrapolation
    )
    result = select_cop_heating_rate_pareto_knee(
        add_single_objective_optima(objectives),
        minimum_time=boundary_source.get("stable_heating_start"),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        raise ValueError("run name may contain only letters, numbers, dot, underscore and hyphen")
    if args.workers < 1 or args.candidate_step_seconds < 1:
        raise ValueError("workers and candidate step must be positive")
    run = args.output_root / "defrost_decisions" / args.run_name
    if run.exists() and not args.overwrite:
        raise FileExistsError(f"run directory exists; pass --overwrite: {run}")

    models = load_defrost_event_models(args.model_file)
    model_set = models.get("models", {}).get(OUTCOME_MODEL)
    required = tuple(OUTCOME_TARGETS)
    if not isinstance(model_set, dict) or not set(required) <= set(model_set):
        raise ValueError(f"model file requires {OUTCOME_MODEL} with all four outcomes")
    experiments = (
        set.intersection(*(set(model_set[name]["folds"]) for name in required))
        if args.prediction_mode == "cross-fitted"
        else None
    )
    loader = DatasetLoader(args.dataset)
    cycles = metadata_eligible_cycles(loader, args.cycles, experiments)
    if args.dry_run:
        print(
            f"Select defrost time: run={args.run_name}, cycles={len(cycles)}, "
            f"mode={args.prediction_mode}, step={args.candidate_step_seconds}s, "
            f"workers={args.workers}, output={run}"
        )
        return 0

    cycles, excluded = clean_anchor_cycles(loader, cycles, explicit=bool(args.cycles))
    print(f"Selected {len(cycles)} cycle(s); excluded {excluded} by clean-anchor gate")
    tables = []
    with parallel_config(backend="loky", n_jobs=args.workers, inner_max_num_threads=1):
        results = Parallel(return_as="generator_unordered")(
            delayed(calculate_cycle)(
                loader,
                cycle_name,
                models,
                candidate_step_seconds=args.candidate_step_seconds,
                allow_extrapolation=args.allow_extrapolation,
                prediction_mode=args.prediction_mode,
            )
            for cycle_name in cycles
        )
        for completed, table in enumerate(results, start=1):
            tables.append(table)
            print(f"Cycles complete: {completed}/{len(cycles)}")
    decisions = pd.concat(tables, ignore_index=True).sort_values(
        ["cycle_name", "candidate_defrost_time"], kind="stable"
    )
    run.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(run / "candidate_decisions.csv", index=False)
    settings = {
        "candidate_step_seconds": args.candidate_step_seconds,
        "allow_extrapolation": args.allow_extrapolation,
        "model_file": str(args.model_file),
        "prediction_mode": args.prediction_mode,
        "model_training_scope": (
            "held_out_experiment_excluded"
            if args.prediction_mode == "cross-fitted"
            else "all_available_training_experiments"
        ),
        "objective_selection": "cycle_cop_and_heating_rate_pareto_compromise",
        "evaporator_capacity_role": "reference_only",
        "workers": args.workers,
        "command": shlex.join(["uv", "run", "python", "select_defrost_time.py", *arguments]),
    }
    (run / "run_settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.figures:
        from plots.defrost_decision import render_current_decision_figures

        render_current_decision_figures(decisions, loader, run / "figures")
    print(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
