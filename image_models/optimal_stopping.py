"""Run stopping-model tuning and ablations on published Dataset inputs.

Read: arguments → inputs → experiment folds → optional search → training → evaluation.
For the fixed main model, use main_defrosting_model.py train or main_defrosting_model.py evaluate.
"""

import json

import pandas as pd
from joblib import Parallel, delayed, parallel_config

from image_models.experiments import (
    build_run_settings,
    prepare_training_folds,
    recipes_for,
    record_experiment,
    summarize_training_run,
)
from image_models.inputs import prepare_ablation_inputs
from image_models.model_selection import search_shared_learning_rate
from image_models.stopping_config import build_parser, configure_training_arguments
from image_models.training import fit_fold, fit_nested


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_training_arguments(parser, args)
    rows, reference = prepare_ablation_inputs(args)
    nested, folds = prepare_training_folds(parser, args, rows)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.shared_lr_trials:
        search_shared_learning_rate(args, rows, reference, folds)
        if args.shared_search_only:
            return
    pd.DataFrame(
        [dict(test_experiment=t, validation_experiments=json.dumps(v)) for t, v in folds.items()]
    ).to_csv(args.output / "folds.csv", index=False)
    recipes = recipes_for(args.recipe, args.loss_name, args.architecture)
    settings = build_run_settings(args, rows, folds, recipes)
    record_experiment(args.output, settings, args.purpose)
    args.flat_output = True
    args.plot = False
    if not args.summarize_only:
        with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
            fit = fit_nested if nested else fit_fold
            Parallel(max_nbytes=None)(
                delayed(fit)(args, rows, reference, test, validation, seed)
                for test, validation in folds.items()
                for seed in args.seeds
            )
    summarize_training_run(args, reference, folds, recipes)


if __name__ == "__main__":
    main()
