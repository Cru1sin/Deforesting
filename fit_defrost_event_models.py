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
from defrost_event_models.ridge_models import (
    MODEL_FEATURES,
    OUTCOME_TARGETS,
    TRAINING_COHORT_RULE,
    assemble_target_model,
    fit_model_for_heldout_experiment,
    fit_model_on_all_experiments,
    mean_outcome_model,
    select_events_complete_for_all_outcomes,
    select_valid_events_for_quantity,
)
from defrost_event_models.training_data import (
    build_defrost_event_training_table,
)
from defrost_event_models.validation import build_validation_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--workers", type=int, default=6)
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


def _fit_models(events: pd.DataFrame, workers: int) -> dict[str, Any]:
    events = select_events_complete_for_all_outcomes(events)
    outcome_events = {
        name: select_valid_events_for_quantity(events, name) for name in OUTCOME_TARGETS
    }
    mean_models: dict[str, dict[str, object]] = {}
    for outcome, target in OUTCOME_TARGETS.items():
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
            for outcome, target in OUTCOME_TARGETS.items()
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
    run = args.output_root / "defrost_event_models" / args.run_name
    if run.exists() and not args.overwrite:
        raise FileExistsError(f"fit directory exists; pass --overwrite: {run}")
    if args.dry_run:
        print(
            f"Fit defrost-event models: run={args.run_name}, models={list(MODEL_FEATURES)}, "
            f"workers={args.workers}, output={run}"
        )
        return 0

    loader = DatasetLoader(args.dataset)
    events = build_defrost_event_training_table(loader)
    valid_events = select_events_complete_for_all_outcomes(events)
    if valid_events.empty:
        raise ValueError("no valid observed defrost events")
    parameters: dict[str, Any] = {
        "model_format_version": "1",
        "run_name": args.run_name,
        "training_cohort_rule": TRAINING_COHORT_RULE,
        "models": _fit_models(events, args.workers),
    }
    validation = build_validation_table(events, parameters)
    run.mkdir(parents=True, exist_ok=True)
    events.to_csv(run / "defrost_events.csv", index=False)
    validation.to_csv(run / "model_validation.csv", index=False)
    (run / "candidate_model_parameters.json").write_text(
        json.dumps(parameters, sort_keys=True, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    settings = {
        "workers": args.workers,
        "command": shlex.join(["uv", "run", "python", "fit_defrost_event_models.py", *arguments]),
        "candidate_parameters_are_not_released_automatically": True,
        "training_cohort_rule": TRAINING_COHORT_RULE,
        "common_training_event_count": len(valid_events),
    }
    (run / "run_settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{run}: {len(valid_events)} valid event(s); parameters not promoted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
