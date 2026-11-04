"""Shared experiment records and result summaries."""

import json
from pathlib import Path

import pandas as pd

from dataset_tools import DatasetLoader
from image_models.checkpoints import checkpoint_files, checkpoint_path
from image_models.control_context import COMPENSATION
from image_models.inputs import (
    CHANGE,
    FREQUENCY,
    RGB,
    TEMPERATURE,
    TEMPERATURE_RECIPES,
    VISUAL_POOLING,
    policy_numeric_columns,
    temperature_columns,
)


def save_settings(path, settings):
    from output_layout import CATEGORIES, OUTPUT

    relative = (
        path.parent.resolve().relative_to(OUTPUT)
        if path.parent.resolve().is_relative_to(OUTPUT)
        else None
    )
    if relative is not None and (not relative.parts or relative.parts[0] not in CATEGORIES):
        raise ValueError(
            "Flat output is no longer allowed; route the output through output_layout first"
        )
    settings = json.loads(json.dumps(settings))
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("settings changed; use a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")


def record_experiment(output, settings, purpose):
    """Record settings inside the owning experiment stage."""
    from datetime import datetime

    output = Path(output)
    if "test" in output.parts and "temp" not in output.parts:
        raise ValueError(
            "Temporary runs belong in temp/test/<experiment>; formal runs inside output/90_复现材料/01_视觉模型训练/<experiment>/training"
        )
    settings = {**settings, "purpose": purpose}
    save_settings(output / "settings.json", settings)
    text = (
        "# " + output.name + "\n\n" + purpose + "\n\n"
        "Checkpoints and per-cycle figures: output/90_复现材料/01_视觉模型训练/<experiment>/checkpoints/<checkpoint>/. inner_*.csv = validation history; "
        "folds.csv = experiment split; cycle_metrics.csv = held-out evaluation.\n\n"
        "```json\n" + json.dumps(settings, indent=2, ensure_ascii=False) + "\n```\n"
    )
    (output / "README.md").write_text(text)
    with (output / "log.txt").open("a") as stream:
        stream.write(f"{datetime.now().isoformat(timespec='seconds')} start/resume: {purpose}\n")


def common_cycle_metrics(metrics, *, model_count=None):
    """Use only cycles scored by every compared model within the same seed."""
    scored = metrics.loc[metrics.selected_cop.notna() & metrics.rb_cop.notna()]
    coverage = scored.groupby(["seed", "cycle_name"]).version.transform("nunique")
    count = metrics.version.nunique() if model_count is None else model_count
    return scored.loc[coverage.eq(count)].copy()


def record_training(args, config, split, screening):
    """Save the actual model configuration and experiment-disjoint data roles."""
    output = args.output / "training"
    settings = dict(
        dataset=str(args.dataset.resolve()),
        camera_role=args.camera_role,
        data_definition=DatasetLoader(args.dataset).camera_training_settings(args.camera_role),
        model=config,
        epochs=args.epochs,
        seed=args.seed,
        standard_through=args.standard_through,
        training_cycles=sorted(split.loc[split.split.eq("train"), "cycle_name"].unique()),
        test_cycles=sorted(split.loc[split.split.eq("test"), "cycle_name"].unique()),
        training_experiments=sorted(split.loc[split.split.eq("train"), "experiment_id"].unique()),
        test_experiments=sorted(split.loc[split.split.eq("test"), "experiment_id"].unique()),
        loss="cdf_ramp",
        trigger_rule="cdf",
        threshold=0.5,
    )
    record_experiment(
        output, settings, "Fixed visual defrost model; later experiments excluded from selection"
    )
    screening.to_csv(output / "cycle_screening.csv", index=False)

    return output


def recipes_for(recipe, loss_name="cdf_ramp", architecture="GRU32"):
    return {
        "single": [(architecture, loss_name)],
        "nested_temperature": [("GRU32", loss_name)],
        "compensation": [(architecture, loss_name)],
        "temperature": [(architecture, loss_name)],
        "compensation_temperature": [(architecture, loss_name)],
    }[recipe]


def prepare_training_folds(parser, args, rows):
    """Build experiment-disjoint folds and verify support before fitting."""
    nested = args.recipe == "nested_temperature" or args.inner_folds == 2
    folds = DatasetLoader.training_folds(
        rows,
        nested=nested,
        cohort=args.cohort,
        validation_rule=args.validation_rule,
        inner_setpoint_coverage=args.inner_setpoint_coverage,
    )
    if args.test_experiments:
        folds = {test: folds[test] for test in args.test_experiments}
    if args.epoch_schedule:
        if args.inner_folds != 2 or args.fixed_epochs or args.tune_trials or args.fold_tuning_from:
            parser.error(
                "Epoch schedule requires two-fold refit and no other epoch/parameter selection"
            )
        schedule = pd.read_csv(args.epoch_schedule)
        if (
            schedule.duplicated(["outer_experiment", "seed"]).any()
            or not schedule.epoch.ge(1).all()
        ):
            raise ValueError("Epoch schedule must have one positive duration per fold and seed")
        schedule = schedule.loc[
            schedule.seed.isin(args.seeds) & schedule.outer_experiment.isin(folds)
        ]
        args.frozen_epochs = {
            (row.outer_experiment, int(row.seed)): int(row.epoch) for row in schedule.itertuples()
        }
        if set(args.frozen_epochs) != {(test, seed) for test in folds for seed in args.seeds}:
            raise ValueError("Epoch schedule does not match the requested fold/seed matrix")
    # Every inner split must have reviewer cycles, before any expensive fitting.
    for test, validations in folds.items():
        for validation in validations if nested else [validations]:
            split = DatasetLoader.split_training_data(rows, test, validation, cohort=args.cohort)
            if split.loc[split.split.eq("validation")].empty:
                raise ValueError(
                    f"No reviewer cycles for validation in outer {test}; publish a usable evaluation cohort"
                )
            if split.loc[
                split.split.eq("train") & split.training_eligible & split.input_valid
            ].empty:
                raise ValueError(f"No eligible training trajectories in outer {test}")
    return nested, folds


def build_run_settings(args, rows, folds, recipes):
    """Describe the selected recipe without changing its calculation or fitting."""
    settings = dict(
        dataset=str(args.dataset.resolve()),
        cohort=args.cohort,
        validation_tiebreak=args.validation_tiebreak,
        validation_rule=args.validation_rule,
        refit_all=args.refit_all,
        data_definition=DatasetLoader(args.dataset).camera_training_settings(args.camera_role),
        measurement_processing=DatasetLoader(args.dataset).registry.get("measurement_processing"),
        loss_name=args.loss_name,
        seeds=args.seeds,
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
        folds=folds,
        recipes=[list(recipe) for recipe in recipes],
        inputs=[*RGB, *FREQUENCY],
        threshold=0.5,
        initial_hazard=0.01,
        teacher="fixed without_trunc COP, recalculated with model extrapolation as in unified screening; no outer energy refit",
        terminal="exact actual preparation time/COP; recording end is censored, not a forced stop",
        training="published label trajectories; experiment-disjoint validation/test",
        history="GRU state reset per cycle; missing observations skip updates and force continuation; no future inputs",
        objective="cycle-balanced CDF ramp or exact set likelihood",
        validation=f"{args.validation_rule} experiment holdout; evaluable count then deterministic first-trigger COP; tie break {args.validation_tiebreak}; refit_all={args.refit_all}",
        preprocessing="train fold imputer/scaler; unchanged cached action input mask; no Harmony input",
    )
    settings["camera_role"] = args.camera_role
    settings["cycle_names"] = sorted(rows.cycle_name.unique())
    settings["inner_setpoint_coverage"] = args.inner_setpoint_coverage
    settings["cycle_cohort"] = str(args.cycle_cohort.resolve()) if args.cycle_cohort else None
    describe_input_recipe(args, settings)
    describe_network_recipe(args, settings)
    describe_selection_recipe(args, settings)
    finalize_recipe_settings(args, settings)
    return settings


def describe_input_recipe(args, settings):
    if args.recipe == "nested_temperature":
        settings["inputs_by_version"] = {
            v: [*RGB, *FREQUENCY, *extra] for v, extra in TEMPERATURE_RECIPES.items()
        }
        settings["validation"] = (
            "two setpoint-stratified experiment folds; select epoch by pooled inner first-trigger COP; refit all outer-training experiments"
        )
        settings["inner_epochs"] = args.maximum_epochs
        settings["temperature"] = (
            "strictly past original p1 snapshot <=30s; no teacher temperature statistics; unchanged action support"
        )
    if args.recipe == "compensation":
        settings["inputs_by_version"] = {
            "GRU32_frequency": [*RGB, *FREQUENCY],
            "GRU32_compensation": [*RGB, *COMPENSATION],
        }
    if args.recipe == "temperature":
        settings["inputs"] = [*RGB, *FREQUENCY, *temperature_columns(args)]
    if args.temperature_representation != "absolute":
        settings["temperature_representation"] = args.temperature_representation
        settings["temperature_context"] = (
            "setpoint and causal setpoint-minus-outlet error; same dimension and information, no new sensor, teacher, or support"
        )
    if args.elapsed_context:
        settings["inputs"] += ["elapsed_cycle_minutes"]
        settings["elapsed_context"] = (
            "candidate time minus recorded cycle_start; no total duration or future normalization; same published action mask"
        )
    if args.compensation_progress:
        settings["inputs"] += ["frequency_used_margin_fraction"]
        settings["compensation_progress"] = (
            "running maximum observed frequency_headroom_fraction minus current headroom; per-cycle, input-valid updates only, fixed 100 Hz cap; identifies rebound, not frost causality"
        )
    if args.drop_inputs:
        settings["inputs"] = [*RGB, *policy_numeric_columns(args)]
        settings["drop_inputs"] = args.drop_inputs
        settings["feature_ablation"] = (
            "Remove named features from recurrent input and FiLM; unchanged published input mask, teacher and cohort"
        )
    if args.temperature_trend:
        settings["inputs"] += ["temperature_error_change_300s_c"]
        settings["temperature_trend"] = (
            "(setpoint - outlet) now minus 300s earlier; published strictly past snapshots; unchanged input mask and train-fitted imputer"
        )


def describe_network_recipe(args, settings):
    settings["research_basis"] = {
        "cdf": "https://arxiv.org/abs/1801.05512; survival/CDF construction only; uniform near-window target and BCE are this experiment design, not an established heat-pump rule",
        "film": "https://arxiv.org/abs/1709.07871; feature-wise affine conditioning; no heat-pump performance claim",
    }
    if args.tune_trials:
        settings["research_basis"]["drop"] = (
            "https://arxiv.org/abs/1512.05287; NeurIPS 2016 motivates time-shared dropout for recurrent models; this input-feature-only drop is not the full variational recurrent dropout algorithm and uses one deterministic model at deployment"
        )
    if args.elapsed_context:
        settings["research_basis"]["clock"] = (
            "https://www.kdd.org/kdd2017/papers/view/patient-subtyping-via-time-aware-lstm-networks; explicit elapsed-time motivation for irregular observations only; current-cycle age here is a causal covariate, not T-LSTM memory decay"
        )
    if args.visual_history != "none":
        settings["research_basis"]["visual_change"] = (
            "https://arxiv.org/abs/1608.00859; ECCV 2016 appearance plus RGB difference; causal frozen-DINO differences are a transfer hypothesis, not TSN itself"
        )
    settings["trigger_rule"] = args.trigger_rule
    if args.architecture == "logistic":
        settings["history"] = (
            "Linear hazard from current visual/control input; CDF accumulates per-cycle survival; no learned recurrent state or future input"
        )
    if args.architecture == "GRU32_film_skip":
        settings["history"] = (
            "Baseline joint visual/control GRU32 FiLM plus a zero-initialized linear current-DINO readout; causal cycle-local history and unchanged action mask"
        )
        settings["research_basis"]["visual_skip"] = (
            "https://arxiv.org/abs/1606.07792; joint linear and deep readout motivation only; frozen current-DINO residual is our ablation"
        )
    if args.visual_dropout is not None:
        settings["visual_dropout"] = args.visual_dropout
        settings["research_basis"]["drop"] = (
            "https://arxiv.org/abs/1512.05287; shared-time input dropout motivation for GRU only; no recurrent-weight dropout or deployment sampling"
        )
    if args.output_dropout is not None:
        settings["output_dropout"] = args.output_dropout
        settings["output_time_shared"] = True
        settings["research_basis"]["drop"] = (
            "https://arxiv.org/abs/1512.05287; shared-time readout dropout motivation for GRU; one deterministic deployment model, not full variational recurrent dropout"
        )
    if args.learning_rate is not None:
        settings["learning_rate"] = args.learning_rate
    settings["hidden_width"] = args.hidden_width
    settings["recurrent_layers"] = args.recurrent_layers
    settings["reset_history"] = args.reset_history
    settings["visual_time_shared"] = args.visual_time_shared
    settings["output_time_shared"] = args.output_time_shared
    settings["recurrent_architecture"] = (
        f"{args.recurrent_layers}-layer causal {'LSTM' if args.architecture.startswith('LSTM') else 'GRU'}; hidden width {args.hidden_width}; history reset={args.reset_history}"
    )
    if args.architecture in ("MLP_plain", "MLP_film", "logistic"):
        settings["recurrent_architecture"] = (
            "No learned recurrent state; current inputs only; cumulative CDF retained"
        )
    if args.architecture == "MLP_plain":
        settings["history"] = (
            f"{args.mlp_layers} equal-width ReLU layers encode current joint visual/control inputs; direct hazard readout without FiLM; cycle-shared dropout and cumulative CDF retained"
        )
    if args.architecture == "MLP_film":
        settings["history"] = (
            f"{args.mlp_layers} equal-width ReLU layers encode current joint visual/control inputs; current numeric FiLM before hazard readout; cycle-shared dropout and cumulative CDF retained"
        )
    if args.architecture.startswith("LSTM"):
        settings["history"] = (
            "Joint visual/control LSTM resets per cycle; same causal input support and cumulative CDF; readout follows the named architecture"
        )
    if args.reset_history:
        settings["history"] = (
            "Recurrent encoder resets on every input-valid action; current visual/control only; cumulative CDF retains past hazards; unchanged action mask"
        )


def describe_selection_recipe(args, settings):
    settings["validation_utility"] = args.validation_utility
    if args.loss_name == "cdf_ramp" and args.validation_utility == "headroom":
        settings["research_basis"]["epoch_selection"] = (
            "https://ojs.aaai.org/index.php/AAAI/article/view/3982; prediction-error versus decision-quality mismatch motivates our validation-only H_RB epoch-selection ablation; CDF BCE training is unchanged, not the paper algorithm"
        )
    if args.validation_utility == "cdf_loss":
        settings["validation"] = (
            "Two experiment-disjoint inner folds: paired coverage first, then minimum cycle-mean validation CDF BCE, then earliest epoch; H_RB remains final evaluation only"
        )
    if args.loss_name == "cdf_ramp":
        settings["objective"] = (
            "cycle-mean BCE of cumulative event CDF against elapsed-time 0..1 ramp across first/last exact near points, then 1; bounds before input masking; training fills gaps, exact COP evaluation unchanged"
        )
        settings["cdf_link"] = args.cdf_link
        if args.cdf_link == "instantaneous":
            settings["objective"] = settings["objective"].replace(
                "cumulative event CDF", "current stopping probability"
            )
            settings["history"] += "; no cumulative probability at training or deployment"
    if args.joint_trigger_selection:
        settings["joint_trigger_selection"] = True
        settings["validation"] += (
            "; jointly select epoch and hazard/CDF rule on inner validation only"
        )
    if (
        args.candidate_set != "exact"
        or args.late_preference != "none"
        or args.set_aggregation != "marginal"
    ):
        settings["stopping_candidates"] = dict(
            rule=args.candidate_set,
            late_preference=args.late_preference,
            aggregation=args.set_aggregation,
            loss=(
                "-sum normalized_candidate_weight(t)*log q(t)"
                if args.set_aggregation == "distribution"
                else "-log sum within stopping support of q(t)*w(t)"
            ),
            weights="w=1 or normalized_time (singleton weight1); exact COP labels and validation selection unchanged",
        )
    if args.tune_trials:
        settings["tune_trials"] = args.tune_trials
        settings["tune_architectures"] = args.tune_architectures
        settings["tuning"] = (
            "Inner-fold-only Optuna TPE: optional architecture list, dropout none/visual/output, p .05/.1/.2/.3, shared-time boolean, lr 3e-4..2e-3; paired coverage then requested utility then exact set-loss; fixed outer experiment holdout"
        )
        if args.architecture == "LSTM32_film":
            settings["tuning"] = (
                "Each outer fold independently searches LSTM widths 16/25/32/48, layers 1/2, LR 3e-4..1e-2, weight decay 1e-6..1e-2, visual/output dropout 0/.1/.2/.3 with time-shared masks; paired coverage then inner H_RB, exact set loss and earliest epoch; no outer-test selection"
            )
    settings["mlp_layers"] = args.mlp_layers
    settings["mlp_widths"] = args.mlp_widths
    if args.mlp_widths:
        settings["history"] = settings["history"].replace(
            f"{args.mlp_layers} equal-width", str(args.mlp_widths)
        )
    settings["weight_decay"] = args.weight_decay if args.weight_decay is not None else 1e-4
    if args.architecture in ("MLP_plain", "MLP_film") and (
        args.tune_trials or args.fold_tuning_from
    ):
        settings["history"] = (
            "Inner-selected equal-width ReLU MLP on current visual/control inputs; depth and width stored per checkpoint; "
            + (
                "numeric FiLM before hazard readout; "
                if args.architecture == "MLP_film"
                else "no FiLM; "
            )
            + "no recurrent memory; cumulative CDF retained"
        )
    if args.fold_tuning_from:
        settings["fold_tuning_from"] = str(args.fold_tuning_from.resolve())
        settings["fold_tuning_seed"] = args.fold_tuning_seed
    if args.architecture == "LSTM32_film" and (args.tune_trials or args.fold_tuning_from):
        settings["recurrent_architecture"] = (
            "Causal LSTM; inner-selected width and depth stored separately for each outer checkpoint"
        )
        settings["history"] = (
            "Joint visual/control LSTM resets per cycle; numeric FiLM before the hazard readout; inner-selected capacity and regularization per outer fold; unchanged causal support and cumulative CDF"
        )
    if args.tune_trials and args.architecture == "MLP_plain":
        settings["tuning"] = (
            "Inner-only TPE; width 32/64/86/128/192/256; depth 1..4; lr 3e-4..2e-2; AdamW decay 1e-6..1e-2; independent visual/output dropout; coverage then validation H_RB; outer experiment untouched"
        )
    if args.shared_tuning:
        settings["shared_tuning"] = args.shared_tuning
    if args.inner_folds == 2:
        settings["inner_folds"] = 2
        settings["validation"] = (
            f"two complementary experiment-stratified inner folds; coverage then pooled {args.validation_utility} then exact set loss selects epoch; trigger={args.trigger_rule}; one outer-training refit"
        )
        if args.validation_utility == "cdf_loss":
            settings["validation"] = (
                f"two complementary experiment-stratified inner folds; paired coverage then minimum cycle-mean CDF BCE then earliest epoch; fixed trigger={args.trigger_rule}; one outer-training refit"
            )


def finalize_recipe_settings(args, settings):
    if args.visual_pooling != "cls":
        settings["visual_pooling"] = args.visual_pooling
        settings["inputs"] = [
            *VISUAL_POOLING[args.visual_pooling],
            *[c for c in settings["inputs"] if c not in RGB],
        ]
        if args.visual_pooling == "zero":
            settings["visual_information"] = (
                "DINO channels fixed at zero before train-fold scaling; identical parameter count and availability mask, no image information"
            )
    if args.recipe == "compensation_temperature":
        settings["inputs"] = [*RGB, *dict.fromkeys([*COMPENSATION, *TEMPERATURE])]
    if args.ambient_context:
        settings["inputs"] += ["control_ambient_c"]
        settings["ambient_context"] = (
            "original external T4 snapshot strictly before action, <=30s old; no interpolated teacher temperature"
        )
    if args.visual_history != "none":
        if args.visual_history == "anchor_only":
            settings["inputs"] = [c for c in settings["inputs"] if c not in RGB]
        settings["inputs"] += CHANGE
        settings["visual_history"] = dict(
            mode=args.visual_history,
            minutes=args.history_minutes,
            source="past image <= requested time; <=30s tolerance; first frame warmup; missing lag zero difference; original action mask",
        )
        if args.visual_history in ("anchor", "anchor_only"):
            settings["visual_history"]["source"] = (
                "difference from first observable RGB in each cycle; fixed causal reference, not assumed frost-free; original action mask"
            )
    if args.compare_run:
        previous = json.loads((args.compare_run / "settings.json").read_text())
        for key in [
            "dataset",
            "data_definition",
            "seeds",
            "maximum_epochs",
            "patience",
            "folds",
            "teacher",
            "terminal",
            "training",
            "threshold",
            "initial_hazard",
        ]:
            if key == "seeds" and set(settings[key]).issubset(previous[key]):
                continue
            if key == "folds" and args.inner_folds == 2:
                if set(previous[key]) != set(settings[key]):
                    raise ValueError("Two-fold comparison must retain every outer test experiment")
                continue
            if previous[key] != settings[key]:
                raise ValueError(f"Comparison differs in {key}")
        settings["compare_run"] = str(args.compare_run.resolve())
    if args.fixed_epochs:
        settings["fixed_epochs"] = args.fixed_epochs
        settings["validation"] = (
            "Frozen shared configuration and training duration; no per-fold inner selection; outer experiment excluded from fitting"
        )
    if args.epoch_schedule:
        settings["epoch_schedule"] = str(args.epoch_schedule.resolve())
        settings["validation"] = (
            "Frozen baseline training-side epoch per outer experiment and seed; same duration in every factorial arm; no new inner selection"
        )


def summarize_training_run(args, reference, folds, recipes):
    """Require complete checkpoints, compare requested recipes, and organize outputs."""
    versions = [f"{a}_{l}" for a, l in recipes]
    if args.recipe == "nested_temperature":
        versions = list(TEMPERATURE_RECIPES)
    if args.recipe == "compensation":
        versions = ["GRU32_frequency", "GRU32_compensation"]
    if args.recipe == "temperature":
        versions = ["GRU32_setpoint_outlet"]
    if args.recipe == "compensation_temperature":
        versions = ["GRU32_compensation_temperature"]
    if args.version:
        versions = [args.version]
    expected_fits = len(folds) * len(args.seeds) * len(versions)
    if len(checkpoint_files(args.output)) != expected_fits:
        raise ValueError(
            f"Incomplete fold matrix: expected {expected_fits} checkpoints; resume training first"
        )
    if args.summarize_only and args.trigger_rule != "auto":
        from image_models.training_cases import replay_fixed_trigger

        replay_fixed_trigger(args.output, reference, args.trigger_rule)
    if args.recipe == "compensation":
        contrasts = [(versions[0], versions[1])]
    elif args.recipe == "temperature" and args.version:
        previous_versions = (
            pd.read_csv(args.compare_run / "summary.csv").version.unique().tolist()
            if args.compare_run
            else []
        )
        # A comparison summary may also contain its own older baseline, whose
        # checkpoints are not in this directory and are not loaded by summarize.
        previous_versions = [
            v
            for v in previous_versions
            if checkpoint_path(
                args.compare_run, f"{v}_{next(iter(folds))}_seed{args.seeds[0]}.pkl"
            ).exists()
        ]
        contrasts = [(v, args.version) for v in previous_versions]
        versions = [*previous_versions, args.version]
    elif args.recipe == "temperature":
        versions = (
            ["GRU32_frequency", "GRU32_setpoint_outlet", "GRU32_compensation"]
            if args.compare_run
            else versions
        )
        contrasts = [("GRU32_setpoint_outlet", "GRU32_compensation")] if args.compare_run else []
    elif args.recipe == "compensation_temperature":
        contrasts = (
            [("GRU32_compensation", "GRU32_compensation_temperature")] if args.compare_run else []
        )
        if args.compare_run:
            versions = ["GRU32_frequency", "GRU32_compensation", *versions]
    elif args.recipe == "single":
        contrasts = []
    elif args.recipe == "nested_temperature":
        contrasts = [
            (versions[0], versions[1]),
            (versions[1], versions[2]),
            (versions[0], versions[2]),
        ]
    else:
        contrasts = [
            (versions[0], versions[1]),
            (versions[0], versions[2]),
            (versions[2], versions[3]),
            (versions[0], versions[3]),
        ]
    from image_models.comparisons import summarize

    summarize(args, versions, contrasts, args.compare_run)
    from image_models.checkpoints import organize
    from image_models.comparisons import write_ablation_tables

    organize([args.output], dataset=args.dataset)
    write_ablation_tables([args.output], args.output)


def collect_frozen_mlp_metrics(runs, dataset, output, *, references=()):
    """One shared evaluator for every frozen configuration and operating condition."""
    from image_models.ablation import read_metrics
    from image_models.comparisons import compensation_comparisons, write_evaluation_metrics

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    versions = []
    for run in [*runs, *references]:
        version = pd.read_csv(run / "training/summary.csv").version.iloc[0]
        versions.append(version)
        frames.append(read_metrics(run, version))
    metrics = pd.concat(frames, ignore_index=True)
    write_evaluation_metrics(metrics, dataset, output)
    if references:
        compensation_comparisons(
            metrics,
            output,
            [(ref, version) for ref in versions[len(runs) :] for version in versions[: len(runs)]],
            versions,
        )
    return metrics
