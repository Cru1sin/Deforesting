"""Published inputs, fixed model training, and held-out evaluation.

The main model uses a frozen training recipe. Test experiments never select
epochs, model parameters, or the CDF trigger threshold.
"""

import pickle
from functools import partial
from types import SimpleNamespace

import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from dataset_tools import DatasetLoader
from image_models.checkpoints import checkpoint_path
from image_models.control_context import COMPENSATION
from image_models.experiments import recipes_for
from image_models.inputs import (
    FREQUENCY,
    TEMPERATURE,
    TEMPERATURE_RECIPES,
    policy_numeric_columns,
    policy_visual_columns,
    temperature_columns,
)
from image_models.model_selection import (
    fit_inner_histories,
    nested_fit_options,
    pooled_validation_selection,
    tune_nested,
)
from image_models.policy import fit_policy, summarize_metrics
from image_models.training_cases import fit_case


def fit_model(
    rows, reference, model, *, epochs, seed, output, name, test=None, camera_role="front"
):
    """Fit one fixed recipe and use the shared prediction/scoring/storage path.

    DatasetLoader assigns train/test roles before this call. A completed weight
    is reused only within an experiment whose saved settings were checked by
    the caller; it is never rewritten during resume.
    """
    torch.set_num_threads(1)
    tag = name if test is None else f"{name}_{test}"
    path = checkpoint_path(output, f"{tag}_seed{seed}.pkl")
    if path.exists():
        return path
    training = rows.loc[rows.split.eq("train") & rows.input_valid & rows.training_eligible]
    heldout = rows.loc[rows.split.eq("test")]
    if set(training.experiment_id) & set(heldout.experiment_id):
        raise ValueError("Training and test experiments overlap")
    if training.empty or heldout.empty:
        raise ValueError("Training and test data must both be non-empty")
    with threadpool_limits(limits=1):
        checkpoint = fit_policy(
            training,
            None,
            **model,
            loss_name="cdf_ramp",
            seed=seed,
            maximum_epochs=epochs,
            patience=epochs + 1,
            cycles_per_batch=8,
            initial_hazard=0.01,
            trigger_rule="cdf",
        )
    checkpoint["training_experiments"] = sorted(training.experiment_id.unique())
    checkpoint["fixed_training_epochs"] = epochs
    checkpoint["inner_validation_experiments"] = []
    # fit_case owns prediction, CDF triggering, exact-set scoring and serialization.
    args = SimpleNamespace(
        output=output,
        loss_name="cdf_ramp",
        refit_all=True,
        trigger_rule="cdf",
        flat_output=True,
        camera_role=camera_role,
    )
    fit_case(
        args,
        rows,
        reference,
        "DINO",
        "frequency",
        model["architecture"],
        seed,
        version=name,
        fold=test,
        numeric_columns=model["numeric_columns"],
        visual_columns=model["visual_columns"],
        checkpoint=checkpoint,
    )
    return path


def summarize(path, output):
    """Export the same paired COP/near-set metrics used by the camera study."""
    saved = pickle.load(path.open("rb"))
    metrics = saved["cycle_metrics"]
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "cycle_metrics.csv", index=False)
    summary = pd.DataFrame([summarize_metrics(metrics)])
    summary.to_csv(output / "summary.csv", index=False)
    return summary


def fit_nested(args, rows, reference, test, validation_folds, seed):
    import torch

    from image_models.policy import fit_policy, observable_stopping_cycles

    torch.set_num_threads(1)
    if getattr(args, "epoch_schedule", None):
        from copy import copy

        args = copy(args)
        args.fixed_epochs = int(args.frozen_epochs[(test, seed)])
    pool = rows.loc[rows.experiment_id.ne(test)].copy()
    pooled = getattr(args, "inner_folds", 1) == 2
    contexts = {args.version: temperature_columns(args)} if pooled else TEMPERATURE_RECIPES
    for version, extra in contexts.items():
        final_path = checkpoint_path(args.output, f"{version}_{test}_seed{seed}.pkl")
        if final_path.exists():
            if (
                getattr(args, "fixed_epochs", None)
                and getattr(args, "trigger_rule", "auto") != "auto"
            ):
                from image_models.training_cases import replay_fixed_trigger

                replay_fixed_trigger(args.output, reference, args.trigger_rule, paths=[final_path])
            continue
        final_path.parent.mkdir(parents=True, exist_ok=True)
        numeric = policy_numeric_columns(args, extra)
        visual = policy_visual_columns(args)

        inner_histories = partial(
            fit_inner_histories, args, pool, reference, test, validation_folds, seed, version, extra
        )

        shared, fit_options = nested_fit_options(args, test, seed, version, validation_folds)
        tuning = None
        if args.tune_trials:
            fit_options, tuning, histories = tune_nested(
                args, test, seed, version, final_path, numeric, visual, inner_histories
            )
        elif not getattr(args, "fixed_epochs", None):
            histories = inner_histories(fit_options)
        if getattr(args, "fixed_epochs", None):
            epoch = args.fixed_epochs
        elif pooled:
            selection = pooled_validation_selection(
                histories,
                getattr(args, "trigger_rule", "auto"),
                getattr(args, "validation_utility", "cop"),
            )
            epoch = int(selection.iloc[0].epoch)
        else:
            curves = pd.concat(histories).assign(
                cop_sum=lambda x: x.validation_mean_cycle_cop * x.validation_cycles
            )
            selection = curves.groupby("epoch").agg(
                cop_sum=("cop_sum", "sum"), cycles=("validation_cycles", "sum")
            )
            selection["mean_cop"] = selection.cop_sum / selection.cycles
            epoch = int(selection.mean_cop.idxmax())
        if not getattr(args, "fixed_epochs", None):
            selection.to_csv(final_path.parent / f"inner_{version}_{test}_selection_seed{seed}.csv")
        training = pool.loc[pool.input_valid & pool.training_eligible]
        if args.loss_name == "set_stop":
            training = observable_stopping_cycles(training)
        options = dict(fit_options)
        architecture = options.pop("architecture", args.architecture)
        final = fit_policy(
            training,
            None,
            architecture=architecture,
            loss_name=args.loss_name,
            numeric_columns=numeric,
            visual_columns=visual,
            seed=seed,
            maximum_epochs=epoch,
            patience=args.patience,
            cycles_per_batch=8,
            initial_hazard=0.01,
            trigger_rule=getattr(args, "trigger_rule", "auto"),
            **options,
        )
        final["inner_validation_experiments"] = (
            []
            if getattr(args, "fixed_epochs", None)
            else (
                sorted({e for fold in validation_folds for e in fold})
                if pooled
                else validation_folds
            )
        )
        final["training_experiments"] = sorted(pool.experiment_id.unique())
        if tuning is not None:
            final["tuning"] = tuning
        if shared:
            final["shared_tuning"] = shared
        if pooled and not getattr(args, "fixed_epochs", None):
            final["inner_validation_folds"] = validation_folds
            final["pooled_validation_scores"] = selection.loc[selection.epoch.eq(epoch)].copy()
            final["refit_losses"] = final["losses"]
            final["losses"] = (
                selection.drop_duplicates("epoch")
                .sort_values("epoch")
                .rename(columns={"validation_cop": "validation_mean_cycle_cop"})
            )
            # Diagnostic mean across inner fits; the pooled selector uses validation only.
            training_loss = pd.concat(histories).groupby("epoch").training_loss.mean()
            final["losses"]["training_loss"] = final["losses"].epoch.map(training_loss)
        if getattr(args, "fixed_epochs", None):
            final["fixed_training_epochs"] = epoch
            if getattr(args, "epoch_schedule", None):
                final["epoch_schedule"] = str(args.epoch_schedule)
        split = DatasetLoader.split_training_data(
            rows, test, [], cohort=args.cohort if pooled else "reviewer"
        )
        fit_case(
            args,
            split,
            reference,
            "DINO",
            "frequency",
            args.architecture,
            seed,
            version=version,
            fold=test,
            strategy="first_positive_terminal",
            numeric_columns=numeric,
            visual_columns=visual,
            checkpoint=final,
        )


def fit_fold(args, rows, reference, test, validation, seed):
    rows = DatasetLoader.split_training_data(rows, test, validation, cohort=args.cohort)
    if args.recipe in ("compensation", "temperature", "compensation_temperature"):
        contexts = (
            [("GRU32_setpoint_outlet", [*FREQUENCY, *temperature_columns(args)])]
            if args.recipe == "temperature"
            else [("GRU32_frequency", FREQUENCY), ("GRU32_compensation", COMPENSATION)]
        )
        if args.recipe == "compensation_temperature":
            contexts = [
                (
                    "GRU32_compensation_temperature",
                    list(dict.fromkeys([*COMPENSATION, *TEMPERATURE])),
                )
            ]
        for version, numeric in contexts:
            if getattr(args, "elapsed_context", False):
                numeric = [*numeric, "elapsed_cycle_minutes"]
            if getattr(args, "temperature_trend", False):
                numeric = [*numeric, "temperature_error_change_300s_c"]
            if args.ambient_context:
                numeric = [*numeric, "control_ambient_c"]
            if args.version:
                if len(contexts) != 1:
                    raise ValueError("Version override requires a single context")
                version = args.version
            fit_case(
                args,
                rows,
                reference,
                "DINO",
                "frequency",
                args.architecture,
                seed,
                version=version,
                fold=test,
                strategy="first_positive_terminal",
                initial_hazard=0.01,
                numeric_columns=numeric,
                visual_columns=policy_visual_columns(args),
            )
        return
    recipes = recipes_for(args.recipe, args.loss_name, args.architecture)
    for architecture, loss in recipes:
        args.loss_name = loss
        fit_case(
            args,
            rows,
            reference,
            "DINO",
            "frequency",
            architecture,
            seed,
            version=f"{architecture}_{loss}",
            fold=test,
            strategy="first_positive_terminal",
            initial_hazard=0.01,
        )
