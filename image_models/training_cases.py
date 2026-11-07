"""Train or refit one experiment, then replay and save through the shared evaluator."""

import pickle

import numpy as np
import torch

from image_models.checkpoints import checkpoint_files, checkpoint_path
from image_models.inputs import FREQUENCY, RGB
from image_models.policy import (
    fit_policy,
    predict_policy,
)

STRATEGY = "first_positive_terminal"


def fit_case(
    args,
    rows,
    reference,
    representation,
    context,
    architecture,
    seed,
    *,
    visual_columns=None,
    version=None,
    fold=None,
    strategy=STRATEGY,
    initial_hazard=None,
    numeric_columns=None,
    checkpoint=None,
):
    from image_models.evaluation import score_stopping_predictions

    torch.set_num_threads(1)
    supplied_checkpoint = checkpoint is not None
    version = version or f"{representation}_{context}_{architecture}"
    tag = f"{version}_{fold}" if fold is not None else version
    path = checkpoint_path(args.output, f"{tag}_seed{seed}.pkl")
    if path.exists():
        return
    visual = visual_columns if visual_columns is not None else RGB
    numeric = (
        numeric_columns
        if numeric_columns is not None
        else (FREQUENCY if context == "frequency" else [])
    )
    train = rows.loc[rows.split.eq("train") & rows.input_valid].copy()
    if "training_eligible" in rows:
        train = train.loc[train.training_eligible]
    if args.loss_name == "set_stop" and not getattr(args, "admission_run", None):
        from image_models.policy import observable_stopping_cycles

        train = observable_stopping_cycles(train)
    validation = rows.loc[rows.split.eq("validation")]
    if checkpoint is not None:
        pass  # A nested-CV refit uses the same prediction and metric path below.
    else:
        checkpoint = fit_policy(
            train,
            validation,
            architecture=architecture,
            loss_name=args.loss_name,
            numeric_columns=numeric,
            visual_columns=visual,
            seed=seed,
            maximum_epochs=args.maximum_epochs,
            patience=args.patience,
            cycles_per_batch=8,
            validation_metric="cop",
            validation_strategy=strategy,
            controller_reference=reference,
            validation_complete_first=True,
            initial_hazard=initial_hazard,
            validation_tiebreak=getattr(args, "validation_tiebreak", "earliest"),
            joint_trigger_selection=getattr(args, "joint_trigger_selection", False),
            trigger_rule=getattr(args, "trigger_rule", "auto"),
        )
    validation_predictions = None
    if getattr(args, "refit_all", False) and not supplied_checkpoint:
        from dataset_tools import DatasetLoader

        partition = DatasetLoader.split_training_data(rows, fold, [], cohort=args.cohort)
        refit = partition.loc[
            partition.split.eq("train") & partition.input_valid & partition.training_eligible
        ]
        if args.loss_name == "set_stop":
            from image_models.policy import observable_stopping_cycles

            refit = observable_stopping_cycles(refit)
        selection = checkpoint
        validation_predictions = validation.copy().assign(
            probability=np.nan, decision_margin=np.nan
        )
        available = validation_predictions.input_valid
        validation_predictions.loc[available, ["probability", "decision_margin"]] = predict_policy(
            validation.loc[available], selection
        )[["probability", "decision_margin"]].to_numpy()
        checkpoint = fit_policy(
            refit,
            None,
            architecture=architecture,
            loss_name=args.loss_name,
            numeric_columns=numeric,
            visual_columns=visual,
            seed=seed,
            maximum_epochs=selection["selected_epoch"],
            patience=args.patience,
            cycles_per_batch=8,
            initial_hazard=initial_hazard,
            trigger_rule=getattr(args, "trigger_rule", "auto"),
        )
        checkpoint["refit_losses"] = checkpoint["losses"]
        checkpoint["losses"] = selection["losses"]
        checkpoint["inner_validation_experiments"] = sorted(validation.experiment_id.unique())
        checkpoint["training_experiments"] = sorted(refit.experiment_id.unique())
        checkpoint["selection_training_experiments"] = sorted(train.experiment_id.unique())
        assert set(checkpoint["selection_training_experiments"]).isdisjoint(
            validation.experiment_id
        )
        assert fold not in checkpoint["training_experiments"]
    checkpoint["camera_role"] = getattr(args, "camera_role", "front")
    splits = ["test"] if getattr(args, "refit_all", False) else ["validation", "test"]
    predicted = (
        rows.loc[rows.split.isin(splits)].copy().assign(probability=np.nan, decision_margin=np.nan)
    )
    valid = predicted.input_valid
    predicted.loc[valid, ["probability", "decision_margin"]] = predict_policy(
        predicted.loc[valid], checkpoint
    )[["probability", "decision_margin"]].to_numpy()
    metrics = score_stopping_predictions(
        predicted, reference, version, seed, fold, strategy, getattr(args, "trigger_rule", "auto")
    )
    columns = [
        "cycle_name",
        "experiment_id",
        "split",
        "candidate_defrost_time",
        "probability",
        "decision_margin",
        "input_valid",
        "near_optimal",
        "cycle_cop",
        "cycle_cop_eligible",
        "optimal_cop",
        "optimal_time",
        "visual_frost_score",
        *FREQUENCY,
    ]
    columns += [c for c in ["terminal_time", "terminal_cop", "training_eligible"] if c in predicted]
    columns = list(dict.fromkeys([*columns, *numeric]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        result = dict(checkpoint=checkpoint, predictions=predicted[columns], cycle_metrics=metrics)
        if validation_predictions is not None:
            result["validation_predictions"] = validation_predictions[columns]
        pickle.dump(result, stream)
    if getattr(args, "flat_output", False):
        checkpoint["losses"].to_csv(path.with_name(path.stem + "_history.csv"), index=False)
        if "refit_losses" in checkpoint:
            checkpoint["refit_losses"].to_csv(
                path.with_name(path.stem + "_refit_history.csv"), index=False
            )
        from datetime import datetime

        with (args.output / "log.txt").open("a") as stream:
            stream.write(
                f"{datetime.now().isoformat(timespec='seconds')} completed {tag} seed={seed} epoch={checkpoint['selected_epoch']}\n"
            )
    print(
        f"[latent] {version} seed={seed} epoch={checkpoint['selected_epoch']}; "
        f"test hits={metrics.loc[metrics.split.eq('test'), 'near_optimal_hit'].eq(True).sum()}",
        flush=True,
    )


def replay_fixed_trigger(run, reference, rule, *, paths=None):
    """Explicitly update a run's deployment rule without refitting its weights."""
    from image_models.evaluation import score_stopping_predictions

    for path in checkpoint_files(run) if paths is None else paths:
        saved = pickle.load(path.open("rb"))
        old = saved["cycle_metrics"]
        outer = old.outer_experiment.iloc[0] if "outer_experiment" in old else None
        saved["cycle_metrics"] = score_stopping_predictions(
            saved["predictions"],
            reference,
            old.version.iloc[0],
            int(old.seed.iloc[0]),
            outer,
            "first_positive_terminal",
            rule,
        )
        saved["checkpoint"]["trigger_rule"] = rule
        with path.open("wb") as stream:
            pickle.dump(saved, stream)
