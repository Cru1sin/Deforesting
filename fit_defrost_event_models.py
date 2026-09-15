"""Fit and validate complete defrost-event Ridge models with nested LOEO."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import parallel_config
from sklearn.utils.parallel import Parallel, delayed

from dataset_tools import DatasetLoader
from dataset_tools.builder.detect_cycles import (
    add_recovery_arguments,
    audit_recovery_cycle,
    recovery_settings,
)
from defrost_event_models.ridge_models import (
    MODEL_FEATURES,
    OUTCOME_TARGETS,
    assemble_target_model,
    fit_model_for_heldout_experiment,
    fit_model_on_all_experiments,
    mean_outcome_model,
    select_events_complete_for_all_outcomes,
    select_valid_events_for_quantity,
)
from defrost_event_models.training_data import (
    build_defrost_event_training_table,
    observed_cycle_cop_comparison,
)
from defrost_event_models.validation import build_validation_table, summarize_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--n-jobs", "--workers", dest="workers", type=int, default=6)
    parser.add_argument("--audit-boundaries", action="store_true")
    parser.add_argument(
        "--figures",
        action="store_true",
        help="render boundary audit or held-out event calibration figures",
    )
    parser.add_argument("--preparation-heat", choices=("include", "zero"), default="zero")
    add_recovery_arguments(parser)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _fit_ridge_outcome(
    model_name: str,
    outcome: str,
    target: str,
    features: tuple[str, ...],
    events: pd.DataFrame,
) -> tuple[str, str, dict[str, object]]:
    experiments = sorted(events["experiment_id"].astype(str).unique())
    folds = {
        heldout: fit_model_for_heldout_experiment(events, heldout, features, target)
        for heldout in experiments
    }
    fitted = assemble_target_model(
        target,
        features,
        folds,
        fit_model_on_all_experiments(events, features, target),
    )
    return model_name, outcome, fitted


def _fit_models(events: pd.DataFrame, workers: int, preparation_heat=None) -> dict[str, Any]:
    targets = (
        OUTCOME_TARGETS
        if preparation_heat is None
        else {
            key: OUTCOME_TARGETS[key]
            for key in (
                ("event_electricity", "event_net_heat")
                if preparation_heat == "include"
                else ("event_electricity",)
            )
        }
    )
    if preparation_heat is None:
        events = select_events_complete_for_all_outcomes(events)
    outcome_events = {name: select_valid_events_for_quantity(events, name) for name in targets}
    mean_models: dict[str, dict[str, object]] = {}
    for outcome, target in targets.items():
        target_events = outcome_events[outcome]
        experiments = sorted(target_events["experiment_id"].astype(str).unique())
        mean_models[outcome] = {
            "model_format_version": "1",
            "target": target,
            "feature_order": [],
            "support_rule": "all_candidates_for_experiment_balanced_mean",
            "folds": {
                heldout: mean_outcome_model(
                    target_events.loc[~target_events["experiment_id"].astype(str).eq(heldout)],
                    target,
                )
                for heldout in experiments
            },
            "full_data_model": mean_outcome_model(target_events, target),
        }
    models: dict[str, Any] = {"experiment_balanced_mean": mean_models}
    with parallel_config(backend="loky", n_jobs=workers, inner_max_num_threads=1):
        fitted = Parallel()(
            delayed(_fit_ridge_outcome)(
                model_name,
                outcome,
                target,
                features,
                outcome_events[outcome],
            )
            for model_name, features in MODEL_FEATURES.items()
            for outcome, target in targets.items()
        )
    for model_name, outcome, model in fitted:
        models.setdefault(model_name, {})[outcome] = model
    return models


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        raise ValueError("run name may contain only letters, numbers, dot, underscore and hyphen")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    run = (
        args.output_root
        / ("recovery" if args.audit_boundaries else "defrost_event_models")
        / args.run_name
    )
    if run.exists() and not args.overwrite:
        raise FileExistsError(f"fit directory exists; pass --overwrite: {run}")
    if args.dry_run:
        print(
            f"Fit defrost-event models: run={args.run_name}, models={list(MODEL_FEATURES)}, "
            f"workers={args.workers}, output={run}"
        )
        return 0

    loader = DatasetLoader(args.dataset)
    if args.audit_boundaries:
        run.mkdir(parents=True, exist_ok=True)
        rows, codes = [], []
        with parallel_config(backend="loky", n_jobs=args.workers, inner_max_num_threads=1):
            results = Parallel(return_as="generator_unordered")(
                delayed(audit_recovery_cycle)(
                    loader, name, recovery_settings(args), run, figures=args.figures
                )
                for name in loader.list_cycles().cycle_name
            )
            for index, (cycle_rows, cycle_codes) in enumerate(results, 1):
                rows.extend(cycle_rows)
                codes.extend(cycle_codes)
                print(f"[recovery] {index}/{len(loader.list_cycles())}", flush=True)
        pd.DataFrame(rows).sort_values(["cycle_name", "recovery_rule"]).to_csv(
            run / "boundaries.csv", index=False
        )
        if args.figures:
            from plots.publication import render_recovery_overviews

            render_recovery_overviews(
                loader, pd.DataFrame(rows).sort_values(["cycle_name", "recovery_rule"]), run
            )
        pd.DataFrame(rows).groupby(
            ["experiment_id", "recovery_rule", "recovery_status"]
        ).size().rename("n_cycles").reset_index().to_csv(
            run / "experiment_coverage.csv", index=False
        )
        pd.DataFrame(codes).to_csv(run / "controller_states.csv", index=False)
        (run / "settings.json").write_text(json.dumps(recovery_settings(args), indent=2))
        print(pd.DataFrame(rows).groupby(["recovery_rule", "recovery_status"]).size().to_string())
        return 0
    boundaries = loader.configure_recovery(recovery_settings(args))
    events = build_defrost_event_training_table(loader, preparation_heat=args.preparation_heat)
    valid_events = select_valid_events_for_quantity(events, "event_electricity")
    if valid_events.empty:
        raise ValueError("no valid observed defrost events")
    parameters: dict[str, Any] = {
        "model_format_version": "1",
        "run_name": args.run_name,
        "training_cohort_rule": "per_target_valid_events",
        "cop_definition": "refrigerant_effective_heat",
        "preparation_heat": args.preparation_heat,
        "recovery_settings": recovery_settings(args),
        "models": _fit_models(events, args.workers, args.preparation_heat),
    }
    validation = build_validation_table(events, parameters)
    run.mkdir(parents=True, exist_ok=True)
    boundaries.to_csv(run / "recovery_boundaries.csv", index=False)
    events.to_csv(run / "defrost_events.csv", index=False)
    if args.preparation_heat == "include":
        observed_comparison = observed_cycle_cop_comparison(loader, events)
        observed_comparison.to_csv(run / "observed_cycle_cop_comparison.csv", index=False)
    validation.to_csv(run / "model_validation.csv", index=False)
    summarize_validation(validation).to_csv(run / "validation_summary.csv", index=False)
    if args.figures:
        from defrost_event_models.validation import _VALIDATION_COLUMNS
        from plots.defrost_decision import render_preparation_comparison
        from plots.pareto_learning import render_calibration_figures

        if args.preparation_heat == "include":
            render_preparation_comparison(
                observed_comparison,
                args.output_root / "test" / args.run_name / "observed_preparation_effect.png",
            )

        targets = [
            OUTCOME_TARGETS[name] for name in parameters["models"]["experiment_balanced_mean"]
        ]
        predictions = (
            validation.loc[validation.model_name.ne("excluded_event")]
            .rename(
                columns={
                    prediction: "predicted_" + OUTCOME_TARGETS[name]
                    for name, (prediction, _) in _VALIDATION_COLUMNS.items()
                }
            )
            .assign(
                panel=args.preparation_heat, representation=lambda rows: rows.model_name, seed=0
            )
        )
        render_calibration_figures(
            predictions,
            None,
            args.output_root / "test" / args.run_name,
            targets=targets,
            bootstrap_replicates=0,
            labels={
                OUTCOME_TARGETS["event_electricity"]: "Total event electricity [kWh]",
                OUTCOME_TARGETS["event_net_heat"]: "Preparation refrigerant heat [kWh]",
            },
        )
    (run / "candidate_model_parameters.json").write_text(
        json.dumps(parameters, sort_keys=True, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    settings = {
        "workers": args.workers,
        "command": shlex.join(["uv", "run", "python", "fit_defrost_event_models.py", *arguments]),
        "candidate_parameters_are_not_released_automatically": True,
        "training_cohort_rule": "per_target_valid_events",
        "cop_definition": "refrigerant_effective_heat",
        "preparation_heat": args.preparation_heat,
        "recovery_settings": recovery_settings(args),
        "energy_training_event_count": len(valid_events),
    }
    (run / "run_settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{run}: {len(valid_events)} valid event(s); parameters not promoted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
