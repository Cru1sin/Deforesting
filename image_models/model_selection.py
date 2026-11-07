"""Choose model parameters and training duration using training-side validation only."""

import json
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed, parallel_config

from dataset_tools import DatasetLoader
from image_models.checkpoints import checkpoint_files
from image_models.experiments import record_experiment
from image_models.inputs import policy_numeric_columns, policy_visual_columns


def tuning_options(params):
    """The same dropout/optimizer parameters for searched and frozen runs."""
    if "mlp_layers" in params or "recurrent_layers" in params:
        depth = "mlp_layers" if "mlp_layers" in params else "recurrent_layers"
        return dict(
            hidden_width=params["hidden_width"],
            **{depth: params[depth]},
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            dropout_config=dict(
                visual_dropout=params["visual_dropout"],
                output_dropout=params["output_dropout"],
                visual_time_shared=True,
                output_time_shared=True,
            ),
        )
    place = params["dropout_place"]
    config = dict(visual_dropout=0.0, output_dropout=0.0)
    if place != "none":
        config.update(
            {
                place + "_dropout": params["dropout_rate"],
                place + "_time_shared": params["time_shared"],
            }
        )
    return dict(
        dropout_config=config,
        learning_rate=params["learning_rate"],
        **({"architecture": params["architecture"]} if "architecture" in params else {}),
    )


def cached_inner_history(args, test, folds, seed, architecture, options, fold):
    """Reuse an identical ablation fit; outer-test results never select a trial."""
    import pickle

    source = getattr(args, "compare_run", None)
    while source:
        source = Path(source)
        config = json.loads((source / "settings.json").read_text())
        same = (
            config["loss_name"] == args.loss_name
            and config["maximum_epochs"] == args.maximum_epochs
            and config.get("camera_role", "front") == getattr(args, "camera_role", "front")
            and config.get("cycle_cohort")
            == (str(args.cycle_cohort.resolve()) if getattr(args, "cycle_cohort", None) else None)
            and config["data_definition"]
            == DatasetLoader(args.dataset).camera_training_settings(
                getattr(args, "camera_role", "front")
            )
            and config.get("visual_pooling", "cls") == getattr(args, "visual_pooling", "cls")
            and bool(config.get("elapsed_context")) == getattr(args, "elapsed_context", False)
            and not config.get("visual_history")
            and getattr(args, "visual_history", "none") == "none"
            and config.get("cdf_regret_weight", 0.0) == 0.0
            and config.get("cdf_score", "bce") == "bce"
        )
        if same:
            for path in checkpoint_files(source, f"*_{test}_seed{seed}.pkl"):
                with path.open("rb") as stream:
                    model = pickle.load(stream)["checkpoint"]
                actual = model["architecture"]
                default = dict(
                    visual_dropout=0.1
                    if actual
                    in (
                        "MLP_film",
                        "LSTM32_film",
                        "GRU32_visualdrop",
                        "GRU32_film",
                        "GRU32_film_skip",
                    )
                    else 0.0,
                    output_dropout=0.2 if actual == "GRU32_drop" else 0.0,
                    visual_time_shared=True,
                    output_time_shared=False,
                )
                dropout = {**default, **(model.get("dropout_config") or {})}
                requested = {
                    **dict(
                        visual_dropout=0.1
                        if architecture
                        in (
                            "MLP_film",
                            "LSTM32_film",
                            "GRU32_visualdrop",
                            "GRU32_film",
                            "GRU32_film_skip",
                        )
                        else 0.0,
                        output_dropout=0.2 if architecture == "GRU32_drop" else 0.0,
                        visual_time_shared=True,
                        output_time_shared=False,
                    ),
                    **options.get("dropout_config", {}),
                }
                if (
                    actual == architecture
                    and not model.get("ema_decay", 0.0)
                    and dropout == requested
                    and model.get("learning_rate", 1e-3) == options.get("learning_rate", 1e-3)
                    and model.get("weight_decay", 1e-4) == options.get("weight_decay", 1e-4)
                    and model.get("hidden_width", 32) == options.get("hidden_width", 32)
                    and model.get("recurrent_layers", 1) == options.get("recurrent_layers", 1)
                    and model.get("reset_history", False) == options.get("reset_history", False)
                    and model.get("mlp_layers", 2) == options.get("mlp_layers", 2)
                    and model.get("mlp_widths") == options.get("mlp_widths")
                    and model["inner_validation_folds"] == folds
                    and test not in model["training_experiments"]
                    and model["numeric_columns"] == policy_numeric_columns(args)
                    and model["visual_columns"] == policy_visual_columns(args)
                ):
                    pattern = f"inner_*_{test}_fold{fold}_seed{seed}.csv"
                    for history_path in sorted(path.parent.glob(pattern)):
                        return pd.read_csv(history_path), str(history_path)
        source = config.get("compare_run")
    return None, None


def pooled_validation_selection(histories, trigger_rule="auto", utility="cop"):
    """Pool disjoint validation cycles; compare one common rule at each epoch."""
    records = []
    for history in histories:
        for rule in ["hazard", "cdf"] if trigger_rule == "auto" else [trigger_rule]:
            count = history[f"validation_{rule}_evaluable"]
            record = pd.DataFrame(
                dict(
                    epoch=history.epoch,
                    rule=rule,
                    validation_evaluable=count,
                    validation_cycles=history.validation_cycles,
                    cop_sum=(history[f"validation_{rule}_cop"] * count).fillna(0.0),
                    set_sum=history.validation_set_loss * history.validation_set_cycles,
                    validation_set_cycles=history.validation_set_cycles,
                )
            )
            if utility in ("headroom", "cdf_loss"):
                for key in ["validation_paired", "model_sum", "rb_sum", "optimal_sum"]:
                    record[key] = history[f"validation_{rule}_{key}"]
            if utility == "cdf_loss":
                record["cdf_sum"] = (
                    history.validation_cdf_loss * history.validation_cdf_loss_cycles
                ).fillna(0.0)
                record["validation_cdf_loss_cycles"] = history.validation_cdf_loss_cycles
            records.append(record)
    pooled = pd.concat(records).groupby(["epoch", "rule"], as_index=False).sum()
    pooled["validation_cop"] = pooled.cop_sum / pooled.validation_evaluable.replace(0, float("nan"))
    pooled["validation_set_loss"] = pooled.set_sum / pooled.validation_set_cycles
    if utility in ("headroom", "cdf_loss"):
        denominator = pooled.optimal_sum - pooled.rb_sum
        pooled["validation_headroom"] = (
            100 * (pooled.model_sum - pooled.rb_sum) / denominator.where(denominator.gt(0))
        )
    pooled["rule_order"] = pooled.rule.map({"hazard": 0, "cdf": 1})
    if utility == "cdf_loss":
        pooled["validation_cdf_loss"] = pooled.cdf_sum / pooled.validation_cdf_loss_cycles.replace(
            0, float("nan")
        )
        return pooled.sort_values(
            ["validation_paired", "validation_cdf_loss", "epoch", "rule_order"],
            ascending=[False, True, True, True],
            na_position="last",
            kind="stable",
        ).drop(columns=["cop_sum", "set_sum", "cdf_sum", "rule_order"])
    return pooled.sort_values(
        [
            "validation_paired" if utility == "headroom" else "validation_evaluable",
            "validation_headroom" if utility == "headroom" else "validation_cop",
            "validation_set_loss",
            "epoch",
            "rule_order",
        ],
        ascending=[False, False, True, True, True],
        na_position="last",
        kind="stable",
    ).drop(columns=["cop_sum", "set_sum", "rule_order"])


def fit_candidate(
    args,
    rows,
    reference,
    test,
    validation_folds,
    width,
    depth,
    learning_rate,
    mlp_widths=None,
    weight_decay=1e-4,
    visual_dropout=None,
    output_dropout=None,
    architecture=None,
    recurrent_layers=1,
):
    """One shared candidate: training-side histories only, no outer-test scoring."""
    import torch

    from image_models.policy import fit_policy

    torch.set_num_threads(1)
    pool = rows.loc[rows.experiment_id.ne(test)]
    numeric = policy_numeric_columns(args)
    visual = policy_visual_columns(args)
    widths = tuple(mlp_widths or [width] * depth)
    architecture = architecture or args.architecture
    recurrent = architecture.startswith(("LSTM", "GRU"))
    visual_dropout = args.visual_dropout if visual_dropout is None else visual_dropout
    output_dropout = args.output_dropout if output_dropout is None else output_dropout
    key = (
        f"{architecture}_"
        + "x".join(map(str, widths))
        + f"_lr{learning_rate:g}_wd{weight_decay:g}_vd{visual_dropout:g}_od{output_dropout:g}"
    )
    if recurrent:
        key += f"_layers{recurrent_layers}"
    histories = []
    directory = args.output / "shared_grid"
    directory.mkdir(parents=True, exist_ok=True)
    for fold, validation in enumerate(validation_folds):
        path = directory / (
            f"{key}_fold{fold}.csv"
            if mlp_widths or recurrent
            else f"w{width}_d{depth}_lr{learning_rate:g}_fold{fold}.csv"
        )
        if path.exists():
            history = pd.read_csv(path)
        else:
            training = pool.loc[
                ~pool.experiment_id.isin(validation) & pool.input_valid & pool.training_eligible
            ]
            eligible = (
                pool.training_eligible if args.cohort == "training" else pool.evaluation_eligible
            )
            heldout = pool.loc[pool.experiment_id.isin(validation) & eligible]
            assert set(training.experiment_id).isdisjoint(heldout.experiment_id)
            fitted = fit_policy(
                training,
                heldout,
                architecture=architecture,
                film=getattr(args, "film", None),
                loss_name=args.loss_name,
                numeric_columns=numeric,
                visual_columns=visual,
                seed=args.seeds[0],
                maximum_epochs=args.maximum_epochs,
                patience=args.maximum_epochs + 1,
                cycles_per_batch=8,
                controller_reference=reference,
                validation_strategy="first_positive_terminal",
                validation_complete_first=True,
                validation_tiebreak="set_loss",
                joint_trigger_selection=True,
                trigger_rule="cdf",
                initial_hazard=0.01,
                hidden_width=width,
                mlp_layers=depth,
                mlp_widths=None if recurrent else widths,
                recurrent_layers=recurrent_layers,
                weight_decay=weight_decay,
                reset_history=getattr(args, "reset_history", False),
                learning_rate=learning_rate,
                dropout_config=dict(
                    visual_dropout=visual_dropout,
                    output_dropout=output_dropout,
                    visual_time_shared=args.visual_time_shared,
                    output_time_shared=args.output_time_shared,
                ),
            )
            history = fitted["losses"]
            history.to_csv(path, index=False)
        histories.append(history)
    epochs = getattr(args, "shared_epoch_candidates", None)
    selected_histories = [h.loc[h.epoch.isin(epochs)] for h in histories] if epochs else histories
    selection = pooled_validation_selection(selected_histories, "cdf", "headroom").iloc[0].to_dict()
    # Joint-input MLP plus numeric FiLM; DINO stays frozen.
    parameters = (
        sum(
            (before + 1) * after
            for before, after in zip((len(numeric) + len(visual), *widths[:-1]), widths)
        )
        + widths[-1]
        + 1
    )
    if architecture == "MLP_film":
        parameters += 2 * widths[-1] * (len(numeric) + 1)
    if recurrent:
        from image_models.policy import build_model

        model = build_model(
            architecture,
            numeric_width=len(numeric),
            visual_width=len(visual),
            seed=args.seeds[0],
            hidden_width=width,
            recurrent_layers=recurrent_layers,
            film=getattr(args, "film", None),
        )
        parameters = sum(p.numel() for p in model.parameters())
    selection.update(
        hidden_width=widths[-1],
        mlp_layers=len(widths),
        learning_rate=learning_rate,
        mlp_widths=json.dumps(None if recurrent else widths),
        architecture=architecture,
        film=getattr(args, "film", None),
        weight_decay=weight_decay,
        recurrent_layers=recurrent_layers,
        visual_dropout=visual_dropout,
        output_dropout=output_dropout,
        candidate_key=key if mlp_widths or recurrent else None,
        parameters=parameters,
        development_outer_experiment=test,
    )
    print(
        f"[shared grid] width={width} depth={depth} lr={learning_rate:g} "
        f"epoch={selection['epoch']} validation H_RB={selection['validation_headroom']:.2f}%",
        flush=True,
    )
    return selection


def rank_shared_candidates(candidates):
    """Coverage first, then pooled development utility; one fixed configuration per row."""
    return (
        pd.DataFrame(candidates)
        .drop_duplicates("candidate_key")
        .sort_values(
            [
                "validation_paired",
                "validation_headroom",
                "validation_set_loss",
                "parameters",
                "epoch",
            ],
            ascending=[False, False, True, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def search_shared_learning_rate(args, rows, reference, folds, *, regularization=False):
    """Globally freeze LR (and optional regularization) before fixed-recipe LOEO."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    test = args.development_experiment or next(iter(folds))
    args.trigger_rule = "cdf"
    args.shared_epoch_candidates = [
        e for e in (40, 60, 80, 100, 120, 160) if e <= args.maximum_epochs
    ]
    if getattr(args, "shared_search_only", False):
        record_experiment(
            args.output,
            dict(
                dataset=str(args.dataset.resolve()),
                seeds=args.seeds,
                development_outer_experiment=test,
                inner_validation_folds=folds[test],
                hidden_width=args.hidden_width,
                recurrent_layers=args.recurrent_layers,
                architecture=args.architecture,
                maximum_epochs=args.maximum_epochs,
                epoch_candidates=args.shared_epoch_candidates,
                weight_decay=args.weight_decay,
                visual_dropout=args.visual_dropout,
                output_dropout=args.output_dropout,
                interpretation="Training-side LR search only; frozen winner before outer evaluation",
            ),
            args.purpose,
        )
    study = optuna.create_study(
        study_name="shared_regularization" if regularization else "shared_learning_rate",
        storage="sqlite:///" + str((args.output / "learning_rate_optuna.sqlite3").resolve()),
        load_if_exists=True,
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=args.seeds[0], n_startup_trials=6),
    )
    if not study.trials:
        for lr in (0.001, 0.002, 0.003, 0.004, 0.006, 0.01):
            study.enqueue_trial(
                dict(
                    learning_rate=lr,
                    **(
                        dict(
                            weight_decay=args.weight_decay,
                            visual_dropout=args.visual_dropout,
                            output_dropout=args.output_dropout,
                        )
                        if regularization
                        else {}
                    ),
                )
            )
    results = [t.user_attrs["selection"] for t in study.trials if "selection" in t.user_attrs]
    completed = len(results)
    while completed < args.shared_lr_trials:
        trials = [study.ask() for _ in range(min(args.n_jobs, args.shared_lr_trials - completed))]
        options = []
        for trial in trials:
            options.append(
                dict(
                    learning_rate=trial.suggest_float("learning_rate", 0.0005, 0.02, log=True),
                    weight_decay=trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3])
                    if regularization
                    else args.weight_decay,
                    visual_dropout=trial.suggest_categorical("visual_dropout", [0.0, 0.1, 0.2, 0.3])
                    if regularization
                    else args.visual_dropout,
                    output_dropout=trial.suggest_categorical("output_dropout", [0.0, 0.1, 0.2, 0.3])
                    if regularization
                    else args.output_dropout,
                )
            )
        with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
            found = Parallel(max_nbytes=None)(
                delayed(fit_candidate)(
                    args,
                    rows,
                    reference,
                    test,
                    folds[test],
                    args.hidden_width,
                    1,
                    recurrent_layers=args.recurrent_layers,
                    **option,
                )
                for option in options
            )
        for trial, row in zip(trials, found):
            row["search_stage"] = (
                "optuna_regularization" if regularization else "optuna_learning_rate"
            )
            row["trial"] = trial.number
            trial.set_user_attr("selection", row)
            study.tell(trial, [row["validation_paired"], row["validation_headroom"]])
            results.append(row)
        completed = len(results)
        rank_shared_candidates(results).to_csv(args.output / "shared_lstm_grid.csv", index=False)
        study.trials_dataframe().to_csv(
            args.output / "optuna_learning_rate_trials.csv", index=False
        )
        print(
            f"[Optuna LR] {completed}/{args.shared_lr_trials} training-side trials complete",
            flush=True,
        )
    frozen = rank_shared_candidates(results).head(1).copy()
    frozen["fixed_epochs"] = frozen.epoch.astype(int)
    frozen["trigger_rule"] = "cdf"
    frozen["threshold"] = 0.5
    frozen["seed"] = args.seeds[0]
    frozen.to_csv(args.output / "frozen_configurations.csv", index=False)
    winner = frozen.iloc[0]
    for key in ("learning_rate", "weight_decay", "visual_dropout", "output_dropout"):
        setattr(args, key, float(winner[key]))
    args.fixed_epochs = int(winner.epoch)
    print(
        f"[freeze Optuna LR] lr={args.learning_rate:g} epochs={args.fixed_epochs}; all LOEO folds identical",
        flush=True,
    )


def fit_inner_histories(
    args, pool, reference, test, validation_folds, seed, version, extra, options_in, suffix=""
):
    """Fit each inner fold without exposing the outer test experiment."""
    from image_models.checkpoints import checkpoint_path
    from image_models.policy import fit_policy, observable_stopping_cycles

    final_path = checkpoint_path(args.output, f"{version}_{test}_seed{seed}.pkl")
    pooled = getattr(args, "inner_folds", 1) == 2
    numeric = policy_numeric_columns(args, extra)
    visual = policy_visual_columns(args)
    histories = []
    for fold, validation in enumerate(validation_folds):
        path = final_path.parent / f"inner_{version}_{test}{suffix}_fold{fold}_seed{seed}.csv"
        if path.exists():
            history = pd.read_csv(path)
        else:
            training = pool.loc[
                ~pool.experiment_id.isin(validation) & pool.input_valid & pool.training_eligible
            ]
            if args.loss_name == "set_stop":
                training = observable_stopping_cycles(training)
            eligible = (
                pool.training_eligible
                if pooled and args.cohort == "training"
                else pool.evaluation_eligible
            )
            heldout = pool.loc[pool.experiment_id.isin(validation) & eligible]
            assert set(training.experiment_id).isdisjoint(heldout.experiment_id)
            options = dict(options_in)
            architecture = options.pop("architecture", args.architecture)
            cached, origin = cached_inner_history(
                args, test, validation_folds, seed, architecture, options, fold
            )
            if cached is not None:
                cached["reused_training_history_from"] = origin
                cached.to_csv(path, index=False)
                histories.append(cached)
                continue
            fitted = fit_policy(
                training,
                heldout,
                architecture=architecture,
                loss_name=args.loss_name,
                numeric_columns=numeric,
                visual_columns=visual,
                seed=seed,
                maximum_epochs=args.maximum_epochs,
                patience=args.maximum_epochs + 1,
                cycles_per_batch=8,
                controller_reference=reference,
                validation_strategy="first_positive_terminal",
                validation_complete_first=True,
                initial_hazard=0.01,
                joint_trigger_selection=pooled,
                validation_tiebreak="set_loss" if pooled else "earliest",
                trigger_rule=getattr(args, "trigger_rule", "auto"),
                **options,
            )
            history = fitted["losses"]
            history.to_csv(path, index=False)
        histories.append(history)
    return histories


def nested_fit_options(args, test, seed, version, validation_folds):
    """Restore a frozen training-side recipe, then apply explicit overrides."""
    from image_models.checkpoints import checkpoint_path

    shared = getattr(args, "shared_tuning", None)
    if getattr(args, "fold_tuning_from", None):
        import pickle

        tuning_seed = getattr(args, "fold_tuning_seed", None)
        source = checkpoint_path(
            args.fold_tuning_from,
            f"{version}_{test}_seed{seed if tuning_seed is None else tuning_seed}.pkl",
        )
        with source.open("rb") as stream:
            original = pickle.load(stream)["checkpoint"]
        if (
            test in original["training_experiments"]
            or original["inner_validation_folds"] != validation_folds
        ):
            raise ValueError(
                "Fold tuning must preserve the independent outer experiment and inner folds"
            )
        shared = dict(
            source=str(source),
            parameters=original["tuning"]["parameters"],
            inner_validation_only=True,
            outer_experiment=test,
        )
    fit_options = tuning_options(shared["parameters"]) if shared else {}
    fixed_drop = {
        key: getattr(args, key, None)
        for key in ("visual_dropout", "output_dropout")
        if getattr(args, key, None) is not None
    }
    if "output_dropout" in fixed_drop:
        fixed_drop["output_time_shared"] = getattr(args, "output_time_shared", True)
    if hasattr(args, "visual_time_shared"):
        fixed_drop["visual_time_shared"] = args.visual_time_shared
    if fixed_drop:
        fit_options["dropout_config"] = {**fit_options.get("dropout_config", {}), **fixed_drop}
    if getattr(args, "learning_rate", None) is not None:
        fit_options["learning_rate"] = args.learning_rate
    if getattr(args, "hidden_width", 32) != 32:
        fit_options["hidden_width"] = args.hidden_width
    if getattr(args, "recurrent_layers", 1) != 1:
        fit_options["recurrent_layers"] = args.recurrent_layers
    if getattr(args, "reset_history", False):
        fit_options["reset_history"] = True
    if getattr(args, "mlp_widths", None):
        fit_options["mlp_widths"] = args.mlp_widths
    if getattr(args, "weight_decay", None) is not None:
        fit_options["weight_decay"] = args.weight_decay
    if getattr(args, "mlp_layers", 2) != 2:
        fit_options["mlp_layers"] = args.mlp_layers
    return shared, fit_options


def tune_nested(args, test, seed, version, final_path, numeric, visual, inner_histories):
    """Search within the outer-training experiments, then freeze one trial."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        directions=["maximize", "maximize"],
        study_name=f"{version}_{test}_seed{seed}",
        storage=f"sqlite:///{final_path.parent / (test + '_seed' + str(seed) + '.sqlite3')}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=3),
    )
    mlp_search = args.architecture == "MLP_plain"
    lstm_search = args.architecture == "LSTM32_film"
    if not study.trials:
        if lstm_search:
            for width, lr, visual_drop, output_drop in [
                (25, 0.003, 0.1, 0.2),
                (32, 0.003, 0.1, 0.2),
                (25, 0.001, 0.2, 0.2),
            ]:
                study.enqueue_trial(
                    dict(
                        hidden_width=width,
                        recurrent_layers=1,
                        learning_rate=lr,
                        weight_decay=1e-4,
                        visual_dropout=visual_drop,
                        output_dropout=output_drop,
                    )
                )
        elif mlp_search:
            # Include the frozen pure-MLP reference, plus shallow and wider starts.
            for width, depth, lr in [(86, 2, 0.003), (64, 1, 0.006), (128, 3, 0.001)]:
                study.enqueue_trial(
                    dict(
                        hidden_width=width,
                        mlp_layers=depth,
                        learning_rate=lr,
                        weight_decay=1e-4,
                        visual_dropout=0.1,
                        output_dropout=0.2,
                    )
                )
        else:
            initial = (
                [("visual", 0.1, True), ("visual", 0.1, True), ("output", 0.2, False)]
                if args.tune_architectures
                else [("visual", 0.1, True), ("visual", 0.2, True), ("visual", 0.1, False)]
            )
            for index, (place, rate, shared) in enumerate(initial):
                study.enqueue_trial(
                    dict(
                        dropout_place=place,
                        dropout_rate=rate,
                        time_shared=shared,
                        learning_rate=1e-3,
                        **(
                            {
                                "architecture": args.tune_architectures[
                                    min(index, len(args.tune_architectures) - 1)
                                ]
                            }
                            if args.tune_architectures
                            else {}
                        ),
                    )
                )

    def objective(trial):
        if lstm_search:
            trial.suggest_categorical("hidden_width", [16, 25, 32, 48])
            trial.suggest_int("recurrent_layers", 1, 2)
            trial.suggest_float("learning_rate", 3e-4, 1e-2, log=True)
            trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            trial.suggest_categorical("visual_dropout", [0.0, 0.1, 0.2, 0.3])
            trial.suggest_categorical("output_dropout", [0.0, 0.1, 0.2, 0.3])
        elif mlp_search:
            trial.suggest_categorical("hidden_width", [32, 64, 86, 128, 192, 256])
            trial.suggest_int("mlp_layers", 1, 4)
            trial.suggest_float("learning_rate", 3e-4, 2e-2, log=True)
            trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            trial.suggest_categorical("visual_dropout", [0.0, 0.1, 0.2, 0.3])
            trial.suggest_categorical("output_dropout", [0.0, 0.1, 0.2, 0.3, 0.4])
            width = trial.params["hidden_width"]
            depth = trial.params["mlp_layers"]
            inputs = len(numeric) + len(visual)
            trial.set_user_attr(
                "trainable_parameters",
                (inputs + 1) * width + (depth - 1) * (width + 1) * width + width + 1,
            )
        else:
            if args.tune_architectures:
                trial.suggest_categorical("architecture", args.tune_architectures)
            place = trial.suggest_categorical("dropout_place", ["none", "visual", "output"])
            if place != "none":
                trial.suggest_categorical("dropout_rate", [0.05, 0.1, 0.2, 0.3])
                trial.suggest_categorical("time_shared", [False, True])
            trial.suggest_float("learning_rate", 3e-4, 2e-3, log=True)
        history = inner_histories(tuning_options(trial.params), f"_trial{trial.number}")
        selection = pooled_validation_selection(
            history,
            getattr(args, "trigger_rule", "auto"),
            getattr(args, "validation_utility", "cop"),
        )
        choice = selection.iloc[0]
        trial.set_user_attr("selected_epoch", int(choice.epoch))
        trial.set_user_attr("rule", choice.rule)
        trial.set_user_attr("set_loss", float(choice.validation_set_loss))
        if args.validation_utility in ("headroom", "cdf_loss"):
            trial.set_user_attr("validation_headroom", float(choice.validation_headroom))
        if args.validation_utility == "cdf_loss":
            trial.set_user_attr("validation_cdf_loss", float(choice.validation_cdf_loss))
        print(
            f"[tune] {test} trial={trial.number} coverage={choice.validation_evaluable} COP={choice.validation_cop:.6f}"
            + (
                f" H_RB={choice.validation_headroom:.2f}%"
                if args.validation_utility in ("headroom", "cdf_loss")
                else ""
            ),
            flush=True,
        )
        if args.validation_utility == "cdf_loss":
            return float(choice.validation_paired), -float(choice.validation_cdf_loss)
        return (
            (float(choice.validation_paired), float(choice.validation_headroom))
            if args.validation_utility == "headroom"
            else (float(choice.validation_evaluable), float(choice.validation_cop))
        )

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    study.optimize(objective, n_trials=max(0, args.tune_trials - len(complete)))
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = max(complete, key=lambda t: (*t.values, -t.user_attrs["set_loss"], -t.number))
    fit_options = tuning_options(best.params)
    tuning = dict(trial=best.number, parameters=best.params, inner_validation_only=True)
    study.trials_dataframe().to_csv(
        final_path.parent / f"{test}_seed{seed}_trials.csv", index=False
    )
    histories = inner_histories(fit_options, f"_trial{best.number}")
    return fit_options, tuning, histories
