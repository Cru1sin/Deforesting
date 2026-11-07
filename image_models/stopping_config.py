"""Command-line recipes and scientific constraints for stopping-model ablations."""

import argparse
import json
from pathlib import Path

from dataset_tools import DatasetLoader
from image_models.checkpoints import checkpoint_path
from image_models.inputs import FREQUENCY, TEMPERATURE, VISUAL_POOLING
from output_layout import PROJECT


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    data = parser.add_argument_group("Published data and input features")
    data.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    data.add_argument(
        "--camera-role",
        default="front",
        choices=["front", "top", "top_close", "left", "left_close", "extreme"],
    )
    data.add_argument(
        "--cycle-cohort",
        type=Path,
        help="Published cycle-name subset; applied before DatasetLoader assigns folds",
    )
    data.add_argument(
        "--visual-history",
        choices=["none", "current", "delta", "anchor", "anchor_only"],
        default="none",
        help="Causal DINO difference or equal-width current-frame control",
    )
    data.add_argument("--history-minutes", type=float, default=5.0)
    data.add_argument(
        "--ambient-context",
        action="store_true",
        help="Add only the strictly past external T4 ambient measurement",
    )
    data.add_argument(
        "--drop-inputs",
        nargs="+",
        choices=[
            *FREQUENCY,
            *TEMPERATURE,
            "elapsed_cycle_minutes",
            "frequency_used_margin_fraction",
        ],
        help="Remove named numeric features from both GRU and FiLM; retain the identical published action mask",
    )
    data.add_argument("--visual-pooling", choices=list(VISUAL_POOLING), default="cls")
    data.add_argument(
        "--temperature-representation",
        choices=["absolute", "tracking_error"],
        default="absolute",
        help="Equal-width reparameterization: setpoint and setpoint minus outlet instead of setpoint and outlet",
    )
    data.add_argument(
        "--temperature-trend",
        action="store_true",
        help="Add only the published causal 5-minute tracking-error change",
    )
    data.add_argument(
        "--compensation-progress",
        action="store_true",
        help="Add causal headroom recovered then spent: past running maximum headroom minus current headroom",
    )
    data.add_argument(
        "--elapsed-context",
        action="store_true",
        help="Add causal elapsed minutes since cycle start; use visual-pooling zero for capacity-matched time/control ablation",
    )

    model = parser.add_argument_group("Network and training objective")
    model.add_argument(
        "--loss-name",
        choices=["set_stop", "cdf_ramp"],
        default="set_stop",
    )
    model.add_argument(
        "--cdf-link",
        choices=["cumulative", "instantaneous"],
        default="cumulative",
        help="Same ramp labels: cumulative CDF BCE or current-probability BCE for the no-CDF ablation",
    )
    model.add_argument(
        "--architecture",
        choices=[
            "LSTM_plain",
            "MLP_plain",
            "MLP_film",
            "LSTM32_film",
            "GRU32",
            "GRU32_skip",
            "GRU32_visualdrop",
            "GRU32_drop",
            "GRU32_film",
            "GRU32_film_skip",
            "logistic",
        ],
        default="GRU32",
    )
    model.add_argument(
        "--mlp-widths", type=int, nargs="+", help="Explicit hidden widths in input-to-output order"
    )
    model.add_argument(
        "--output-dropout",
        type=float,
        help="Fixed time-shared readout dropout rate for a single-factor two-inner-fold GRU ablation",
    )
    model.add_argument(
        "--hidden-width",
        type=int,
        default=32,
        help="Encoder hidden width; widths may differ to match trainable parameter counts, default 32",
    )
    model.add_argument(
        "--mlp-layers",
        type=int,
        choices=[1, 2, 3, 4],
        default=2,
        help="Pure MLP hidden depth; default preserves historical two-layer weights",
    )
    model.add_argument("--recurrent-layers", type=int, choices=[1, 2, 3], default=1)
    model.add_argument(
        "--reset-history",
        action="store_true",
        help="Same GRU parameters, independent one-step states; cumulative CDF remains",
    )
    model.add_argument("--visual-time-shared", action=argparse.BooleanOptionalAction, default=True)
    model.add_argument("--output-time-shared", action=argparse.BooleanOptionalAction, default=True)
    model.add_argument(
        "--visual-dropout",
        type=float,
        help="Fixed single-factor visual drop rate for the two-inner-fold GRU ablation; default keeps architecture recipe",
    )
    model.add_argument(
        "--candidate-set",
        choices=["exact", "filled"],
        default="exact",
        help="Training support only; filled joins the first and last exact 1 percent points",
    )
    model.add_argument(
        "--late-preference",
        choices=["none", "linear"],
        default="none",
        help="Linear stopping weight from zero at the left endpoint to one at the right",
    )
    model.add_argument(
        "--set-aggregation",
        choices=["marginal", "distribution"],
        default="marginal",
        help="Marginal weighted event likelihood or cross entropy against normalized candidate weights",
    )

    training = parser.add_argument_group("Optimizer, duration and parameter search")
    training.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="One seed for rapid comparisons; use --seeds 0 1 2 for final replication",
    )
    training.add_argument("--maximum-epochs", type=int, default=300)
    training.add_argument(
        "--fixed-epochs",
        type=int,
        help="Frozen training duration for every outer fold; skip inner epoch selection",
    )
    training.add_argument(
        "--epoch-schedule",
        type=Path,
        help="Pre-frozen training-side epoch per outer experiment and seed; no new inner search",
    )
    training.add_argument(
        "--shared-search-only",
        action="store_true",
        help="Freeze training-side LR search recipes without automatically running outer LOEO",
    )
    training.add_argument(
        "--shared-lr-trials",
        type=int,
        default=0,
        help="Training-side Optuna LR-only search for a fixed LSTM; globally freeze winner before LOEO",
    )
    training.add_argument("--weight-decay", type=float, help="Fixed Adam weight decay")
    training.add_argument(
        "--development-experiment",
        help="Experiment excluded from the shared grid; default first LOEO experiment; shared search is development evidence",
    )
    training.add_argument("--patience", type=int, default=60)
    training.add_argument(
        "--tune-architectures",
        nargs="+",
        choices=[
            "GRU32",
            "GRU32_film",
            "GRU32_film_skip",
        ],
        help="Optional inner-only Optuna architecture candidates",
    )
    training.add_argument(
        "--learning-rate",
        type=float,
        help="Fixed learning rate for a two-inner-fold ablation; default uses fit_policy recipe",
    )
    training.add_argument(
        "--tune-trials",
        type=int,
        default=0,
        help="Optuna trials per outer fold; two complementary inner folds only",
    )
    training.add_argument(
        "--fold-tuning-from",
        type=Path,
        help="Reuse each outer fold own inner-only searched parameters; reselect epoch and refit without another search",
    )
    training.add_argument(
        "--fold-tuning-seed",
        type=int,
        help="Reuse the specified search seed across replication seeds; each outer fold retains its own parameters",
    )
    training.add_argument(
        "--fixed-tuning-from",
        type=Path,
        help="Reuse one local searched checkpoint hyperparameter configuration; overlapping folds are development evidence only",
    )

    validation = parser.add_argument_group("Experiment folds and validation selection")
    validation.add_argument(
        "--validation-rule", choices=["chronological", "stratified"], default="chronological"
    )
    validation.add_argument(
        "--inner-setpoint-coverage",
        choices=["all", "repeated"],
        default="all",
        help="Repeated: require two-fold coverage only for setpoints present in at least two outer-training experiments",
    )
    validation.add_argument(
        "--validation-tiebreak", choices=["earliest", "set_loss"], default="earliest"
    )
    validation.add_argument(
        "--refit-all",
        action="store_true",
        help="Refit all outer-training experiments to the inner-selected epoch",
    )
    validation.add_argument(
        "--validation-utility",
        choices=["cop", "headroom", "cdf_loss"],
        default="cop",
        help="Pooled inner criterion: COP, H_RB, or cycle-mean configured CDF score",
    )
    validation.add_argument(
        "--trigger-rule",
        choices=["auto", "hazard", "cdf"],
        default="auto",
        help="Fixed deployment and validation trigger; auto preserves historical rule selection",
    )
    validation.add_argument(
        "--joint-trigger-selection",
        action="store_true",
        help="Select epoch with validation-only hazard/CDF comparison at threshold 0.5",
    )
    validation.add_argument(
        "--inner-folds",
        type=int,
        choices=[1, 2],
        default=1,
        help="Two complementary experiment folds pool epoch/rule selection before one refit",
    )
    validation.add_argument(
        "--test-experiments", nargs="+", help="Explicit fold subset for a run-isolated smoke test"
    )
    validation.add_argument(
        "--cohort",
        choices=["reviewer", "training"],
        default="reviewer",
        help="Training cohort provides held-out diagnostics without requiring full reward",
    )

    run = parser.add_argument_group("Experiment name, outputs and execution")
    run.add_argument(
        "--purpose", required=True, help="Research question and comparison for this experiment"
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--version", help="Explicit label for a single temperature-context comparison")
    run.add_argument("--n-jobs", type=int, default=6)
    run.add_argument("--summarize-only", action="store_true")
    run.add_argument(
        "--recipe",
        choices=[
            "single",
            "nested_temperature",
            "compensation",
            "temperature",
            "compensation_temperature",
        ],
        default="single",
    )
    run.add_argument("--compare-run", type=Path)
    return parser


def configure_training_arguments(parser, args):
    """Validate recipe combinations and resolve saved tuning before loading trajectories."""
    if args.shared_search_only and not args.shared_lr_trials:
        parser.error("Search-only requires a shared LR search")
    if args.mlp_widths:
        if args.architecture not in ("MLP_plain", "MLP_film") or any(
            w < 1 for w in args.mlp_widths
        ):
            parser.error("Explicit widths require a pure MLP with positive widths")
        args.mlp_layers = len(args.mlp_widths)
        args.hidden_width = args.mlp_widths[-1]
    from output_layout import prepare_output

    prepare_output(args, training=True)
    if args.fixed_epochs is not None and (
        args.fixed_epochs < 1 or args.inner_folds != 2 or args.tune_trials or args.fold_tuning_from
    ):
        parser.error(
            "Fixed epochs require a positive duration, two-fold recipe and no per-fold tuning"
        )
    if args.shared_lr_trials and (
        args.shared_lr_trials < 1
        or args.architecture != "LSTM32_film"
        or args.inner_folds != 2
        or len(args.seeds) != 1
        or args.maximum_epochs < 40
        or args.visual_dropout is None
        or args.output_dropout is None
        or args.weight_decay is None
        or args.fixed_epochs
        or args.tune_trials
        or args.fixed_tuning_from
        or args.fold_tuning_from
    ):
        parser.error(
            "Shared LR search requires fixed LSTM, one seed, two inner folds, explicit regularization and no other search"
        )
    if args.cdf_link == "instantaneous" and (
        args.loss_name != "cdf_ramp" or args.trigger_rule != "hazard"
    ):
        parser.error("No-CDF ablation requires ramp BCE and fixed instantaneous hazard triggering")
    if args.drop_inputs and (
        args.recipe != "temperature" or args.inner_folds != 2 or not args.version
    ):
        parser.error("Feature omission requires an explicit two-inner-fold temperature ablation")
    if args.mlp_layers != 2 and (
        args.architecture not in ("MLP_plain", "MLP_film") or args.inner_folds != 2
    ):
        parser.error("MLP depth requires the two-inner-fold MLP recipe")
    if args.hidden_width != 32 and (
        args.recipe != "temperature"
        or args.inner_folds != 2
        or args.tune_trials
        or args.fixed_tuning_from
    ):
        parser.error(
            "Capacity ablation requires two-inner-fold temperature GRU with fixed optimizer settings"
        )
    if (
        args.recurrent_layers != 1
        or args.reset_history
        or not args.visual_time_shared
        or not args.output_time_shared
    ) and (
        args.recipe != "temperature"
        or args.inner_folds != 2
        or not args.architecture.startswith(("GRU", "LSTM"))
    ):
        parser.error(
            "Depth/history/drop-mask ablations require the two-inner-fold temperature GRU recipe"
        )
    if args.compensation_progress and (
        args.recipe != "temperature" or not args.version or args.inner_folds != 2
    ):
        parser.error("Compensation progress requires an explicit two-fold temperature version")
    if args.temperature_representation != "absolute" and (
        args.recipe != "temperature" or not args.version
    ):
        parser.error("Temperature reparameterization requires an explicit temperature version")
    fixed_drop = [x for x in (args.visual_dropout, args.output_dropout) if x is not None]
    if (fixed_drop or args.learning_rate is not None) and (
        any(not 0 <= x < 1 for x in fixed_drop)
        or (args.learning_rate is not None and args.learning_rate <= 0)
        or args.inner_folds != 2
        or args.tune_trials
        or args.fixed_tuning_from
    ):
        parser.error(
            "Fixed optimizer/drop ablation requires two-fold GRU, drop rates in [0,1), positive learning rate and no Optuna or frozen search"
        )
    if args.validation_utility == "cdf_loss" and (
        args.loss_name != "cdf_ramp" or args.inner_folds != 2 or args.trigger_rule != "cdf"
    ):
        parser.error(
            "CDF-loss epoch selection requires cdf_ramp, two inner folds and fixed CDF triggering"
        )
    if args.elapsed_context and (args.recipe != "temperature" or not args.version):
        parser.error("Elapsed-time ablation requires an explicit temperature version")
    if (
        args.candidate_set != "exact"
        or args.late_preference != "none"
        or args.set_aggregation != "marginal"
    ) and args.loss_name != "set_stop":
        parser.error("Candidate-set ablation requires set_stop")
    if args.tune_trials and (args.tune_trials < 3 or args.inner_folds != 2):
        parser.error("Optuna requires at least 3 trials and two inner folds")
    if args.fold_tuning_from:
        from output_layout import training_output

        args.fold_tuning_from = training_output(args.fold_tuning_from)
        source = json.loads((args.fold_tuning_from / "settings.json").read_text())
        if (
            args.tune_trials
            or args.fixed_tuning_from
            or args.inner_folds != 2
            or source["data_definition"]
            != DatasetLoader(args.dataset).camera_training_settings(args.camera_role)
            or source.get("camera_role", "front") != args.camera_role
            or source.get("cycle_cohort")
            != (str(args.cycle_cohort.resolve()) if args.cycle_cohort else None)
        ):
            parser.error(
                "Fold tuning reuse requires identical published data, two inner folds and no new search"
            )
    args.shared_tuning = None
    if args.fixed_tuning_from:
        if args.tune_trials or args.inner_folds != 2:
            parser.error("Frozen tuning requires two inner folds and no new Optuna trials")
        import pickle

        if not args.fixed_tuning_from.exists():
            args.fixed_tuning_from = checkpoint_path(
                args.fixed_tuning_from.parent, args.fixed_tuning_from.name
            )
        with args.fixed_tuning_from.open("rb") as stream:
            source = pickle.load(stream)["checkpoint"]
        args.shared_tuning = dict(
            source=str(args.fixed_tuning_from.resolve()),
            parameters=source["tuning"]["parameters"],
            development_experiments=source["training_experiments"],
            interpretation="Shared hyperparameters; any overlapping outer experiment is development evidence, not independent evaluation",
        )
    if args.inner_folds == 2 and not (
        args.recipe == "temperature"
        and args.version
        and args.refit_all
        and args.joint_trigger_selection
        and args.validation_tiebreak == "set_loss"
        and args.loss_name in ("set_stop", "cdf_ramp")
        and args.architecture
        in (
            "LSTM_plain",
            "MLP_plain",
            "MLP_film",
            "LSTM32_film",
            "GRU32",
            "GRU32_visualdrop",
            "GRU32_film",
            "GRU32_film_skip",
            "logistic",
        )
        and args.visual_pooling in ("cls", "zero")
        and not args.ambient_context
        and not args.temperature_trend
    ):
        parser.error(
            "Two-fold selection requires temperature/CLS GRU32 family or logistic, refit-all, joint trigger selection and set-loss tie break"
        )
    if args.temperature_trend and (args.recipe != "temperature" or not args.version):
        parser.error("Temperature trend ablation requires temperature recipe and explicit version")
    if args.visual_pooling != "cls" and (
        args.recipe != "temperature" or not args.version or args.visual_history != "none"
    ):
        parser.error(
            "Visual pooling ablation requires temperature recipe, explicit version and no extra visual history"
        )
    if args.ambient_context and (args.recipe != "temperature" or not args.version):
        parser.error("Ambient context requires temperature recipe and explicit version")
    if args.visual_history != "none" and (
        args.recipe != "temperature" or not args.version or args.history_minutes <= 0
    ):
        parser.error(
            "Visual history requires temperature recipe, explicit version and positive history minutes"
        )
    if args.refit_all and args.recipe != "temperature":
        parser.error("--refit-all currently requires the single temperature recipe")
    if args.version and args.recipe not in ("temperature", "compensation_temperature"):
        parser.error("--version requires a single temperature recipe")
