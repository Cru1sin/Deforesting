"""One camera study: full-candidate screening, frozen DINO, fixed LSTM25 FiLM, development LOEO and independent external evaluation."""

import argparse
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from dataset_tools import DatasetLoader
from image_models.camera_data import CAMERAS, image_index, prepare, study_cohorts
from image_models.comparisons import compensation_comparisons
from image_models.policy import summarize_metrics
from output_layout import (
    OUTPUT,
    PROJECT,
    REPRODUCTION,
    analysis_output,
    camera_outputs,
    training_output,
)


def model_arguments(args, camera):
    """Explicit LSTM settings; no synthetic command line or inherited recipe flags."""

    return SimpleNamespace(
        dataset=args.dataset,
        output=camera_outputs(args.output, camera)["training"],
        camera_role=camera,
        purpose=f"{camera}: fixed LSTM with FiLM; development LOEO and later independent test",
        architecture="LSTM",
        film=True,
        hidden_width=25,
        recurrent_layers=1,
        loss_name="cdf_ramp",
        cohort="training",
        version=camera,
        elapsed_context=True,
        compensation_progress=True,
        learning_rate=0.003,
        weight_decay=0.0001,
        visual_dropout=0.2,
        output_dropout=0.2,
        visual_time_shared=True,
        output_time_shared=True,
        maximum_epochs=args.maximum_epochs,
        seeds=[args.seed],
        n_jobs=args.n_jobs,
        shared_lr_trials=args.tune_trials,
        development_experiment=None,
    )


def model_options(args):
    """Actual network settings shared by the camera folds and final weight."""
    from image_models.inputs import NUMERIC_INPUTS, RGB

    return dict(
        architecture=args.architecture,
        film=args.film,
        hidden_width=args.hidden_width,
        recurrent_layers=args.recurrent_layers,
        numeric_columns=NUMERIC_INPUTS,
        visual_columns=RGB,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_config=dict(
            visual_dropout=args.visual_dropout,
            output_dropout=args.output_dropout,
            visual_time_shared=True,
            output_time_shared=True,
        ),
    )


def fit_deployment(args, rows, reference, standard, external):
    """Fit the full-standard weight through the same path as each fixed LOEO fold."""
    from image_models.training import fit_model

    split = DatasetLoader.split_deployment_data(rows, standard, external)
    return fit_model(
        split,
        reference,
        model_options(args),
        epochs=args.fixed_epochs,
        seed=args.seeds[0],
        output=args.output,
        name=args.version + "_deployment",
        camera_role=args.camera_role,
    )


def train(args):
    from joblib import Parallel, delayed, parallel_config

    from image_models.experiments import record_experiment
    from image_models.inputs import load_data, policy_numeric_columns, policy_visual_columns
    from image_models.model_selection import fit_candidate, search_shared_learning_rate
    from image_models.training import fit_model

    cohort = study_cohorts(args)
    for camera in args.cameras:
        model = model_arguments(args, camera)
        model.output.mkdir(parents=True, exist_ok=True)
        part = cohort.loc[cohort.camera_role.eq(camera)]
        standard = set(part.loc[part.data_role.eq("first"), "cycle_name"])
        external = set(part.loc[part.data_role.eq("second"), "cycle_name"])
        rows, reference, _ = load_data(args.dataset, camera)
        rows = rows.loc[rows.cycle_name.isin(standard | external)].copy()
        reference = reference.loc[reference.cycle_name.isin(standard | external)].copy()
        development = rows.loc[rows.cycle_name.isin(standard)].copy()
        development_reference = reference.loc[reference.cycle_name.isin(standard)].copy()
        conditions = (
            development.groupby(["cycle_name", "experiment_id"])
            .water_temperature_setpoint.median()
            .reset_index()
        )
        validations = DatasetLoader.inner_folds(conditions, None, coverage="repeated")
        experiments = sorted(development.experiment_id.unique())
        record_experiment(
            model.output,
            dict(
                dataset=str(args.dataset.resolve()),
                camera_role=camera,
                data_definition=DatasetLoader(args.dataset).camera_training_settings(camera),
                inputs=[*policy_visual_columns(model), *policy_numeric_columns(model)],
                seed=args.seed,
                standard_cycles=sorted(standard),
                external_cycles=sorted(external),
                standard_through=args.standard_through,
                architecture=model.architecture,
                film=model.film,
                hidden_width=25,
                recurrent_layers=1,
                maximum_epochs=args.maximum_epochs,
                tune_trials=0 if camera == "front" else args.tune_trials,
                inner_validation_folds=validations,
                first_test_experiments=experiments,
                first_test="Fixed global recipe LOEO; development evidence because global selection overlaps these experiments",
                second_test="One full-standard deployment weight; independent later experiments excluded from all selection",
            ),
            model.purpose,
        )
        frozen = model.output / "frozen_configurations.csv"
        if not frozen.exists():
            if camera == "front":
                model.shared_epoch_candidates = [
                    e for e in (40, 60, 80, 100, 120, 160) if e <= args.maximum_epochs
                ]
                choice = fit_candidate(
                    model,
                    development,
                    development_reference,
                    None,
                    validations,
                    25,
                    1,
                    0.003,
                    weight_decay=1e-4,
                    visual_dropout=0.2,
                    output_dropout=0.2,
                )
                pd.DataFrame(
                    [
                        {
                            **choice,
                            "fixed_epochs": int(choice["epoch"]),
                            "seed": args.seed,
                            "trigger_rule": "cdf",
                            "threshold": 0.5,
                        }
                    ]
                ).to_csv(frozen, index=False)
            else:
                search_shared_learning_rate(
                    model,
                    development,
                    development_reference,
                    {None: validations},
                    regularization=True,
                )
        selected = pd.read_csv(frozen).iloc[0]
        for key in ["learning_rate", "weight_decay", "visual_dropout", "output_dropout"]:
            setattr(model, key, float(selected[key]))
        model.fixed_epochs = int(selected.fixed_epochs)
        tests = args.test_experiments or experiments
        if not set(tests).issubset(experiments):
            raise ValueError(
                "First-test experiments must belong to the standard development cohort"
            )
        print(
            f"[frozen] {camera}: epochs={model.fixed_epochs}, lr={model.learning_rate:g}, "
            f"wd={model.weight_decay:g}, visual_drop={model.visual_dropout:g}, output_drop={model.output_dropout:g}",
            flush=True,
        )
        with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
            Parallel(max_nbytes=None)(
                delayed(fit_model)(
                    DatasetLoader.split_training_data(development, test, [], cohort="training"),
                    development_reference,
                    model_options(model),
                    epochs=model.fixed_epochs,
                    seed=args.seed,
                    output=model.output,
                    name=model.version,
                    test=test,
                    camera_role=camera,
                )
                for test in tests
            )
        fit_deployment(model, rows, reference, standard, external)
        collect_camera(args, model, part, tests)


def validate_test_table(table, expected, *, first):
    if (
        set(table.cycle_name) != set(expected)
        or table.cycle_name.duplicated().any()
        or not table.split.eq("test").all()
        or (first and not table.outer_experiment.eq(table.experiment_id).all())
    ):
        raise ValueError("Held-out results do not cover the requested camera cohort exactly")


def collect_camera(args, model, cohort, tests):
    import pickle

    from image_models.checkpoints import checkpoint_path
    from output_layout import experiment_root

    first = []
    for test in tests:
        path = checkpoint_path(model.output, f"{model.version}_{test}_seed{args.seed}.pkl")
        result = pickle.load(path.open("rb"))
        first.append(result["cycle_metrics"])
    second_path = checkpoint_path(model.output, f"{model.version}_deployment_seed{args.seed}.pkl")
    second = pickle.load(second_path.open("rb"))["cycle_metrics"]
    tables = {"first": pd.concat(first, ignore_index=True), "second": second}
    owner = experiment_root(model.output)
    for role, table in tables.items():
        expected = cohort.loc[cohort.data_role.eq(role)]
        if role == "first":
            expected = expected.loc[expected.experiment_id.isin(tests)]
        validate_test_table(table, expected.cycle_name, first=role == "first")
        if not table.seed.eq(args.seed).all():
            raise ValueError("Unexpected evaluation seed")
        output = owner / "evaluation" / role
        output.mkdir(parents=True, exist_ok=True)
        table.to_csv(output / "cycle_metrics.csv", index=False)
        pd.DataFrame([summarize_metrics(table)]).to_csv(output / "summary.csv", index=False)
        score = summarize_metrics(table)
        print(
            f"[result] {model.version}/{role}: n={score['cohort_cycles']}, "
            f"H_RB={score['headroom_captured_pct']:.2f}%, hit={score['hit_rate_all_pct']:.2f}%, "
            f"COP gain={score['gain_vs_rb_pct']:.3f}%",
            flush=True,
        )
    return tables


def summarize(args):
    from output_layout import experiment_root

    cohort = study_cohorts(args)
    details, parameters = [], []
    for camera in CAMERAS:
        model = model_arguments(args, camera)
        part = cohort.loc[cohort.camera_role.eq(camera)]
        parameters.append(
            pd.read_csv(model.output / "frozen_configurations.csv").assign(camera_role=camera)
        )
        for role in ["first", "second"]:
            table = pd.read_csv(
                experiment_root(model.output) / "evaluation" / role / "cycle_metrics.csv"
            )
            expected = part.loc[part.data_role.eq(role)]
            if role == "first" and args.test_experiments:
                expected = expected.loc[expected.experiment_id.isin(args.test_experiments)]
            validate_test_table(table, expected.cycle_name, first=role == "first")
            common = set(expected.loc[expected.common_test, "cycle_name"])
            table = table.assign(version=camera, camera_role=camera, test_round=role)
            details.append(table.assign(evaluation_scope=role + "_own"))
            details.append(
                table.loc[table.cycle_name.isin(common)].assign(evaluation_scope=role + "_common")
            )
    pd.concat(parameters, ignore_index=True).to_csv(
        args.output / "selected_hyperparameters.csv", index=False
    )
    metrics = pd.concat(details, ignore_index=True)
    catalog = DatasetLoader(args.dataset).list_cycles().set_index("cycle_name")
    reference = pd.read_parquet(
        args.dataset / "training_data/reference.parquet",
        columns=["cycle_name", "water_temperature_setpoint"],
    )
    metrics["setpoint_c"] = metrics.cycle_name.map(
        reference.groupby("cycle_name").water_temperature_setpoint.median()
    )
    metrics["operating_condition"] = metrics.cycle_name.map(catalog.operating_condition).fillna(
        "unknown"
    )
    metrics.to_csv(args.output / "comparison_cycles.csv", index=False)
    for filename, groups in [
        ("comparison_summary", ["evaluation_scope", "camera_role", "seed"]),
        ("setpoint_summary", ["evaluation_scope", "camera_role", "seed", "setpoint_c"]),
        ("condition_summary", ["evaluation_scope", "camera_role", "seed", "operating_condition"]),
    ]:
        summaries = [
            dict(
                zip(groups, keys),
                experiments=group.experiment_id.nunique(),
                **summarize_metrics(group),
            )
            for keys, group in metrics.groupby(groups)
        ]
        pd.DataFrame(summaries).to_csv(args.output / f"{filename}.csv", index=False)
    for role in ["first", "second"]:
        scope = role + "_common"
        output = args.output / scope
        output.mkdir(exist_ok=True)
        table = metrics.loc[metrics.evaluation_scope.eq(scope)].copy()
        if not table.groupby(["cycle_name", "seed"]).camera_role.nunique().eq(len(CAMERAS)).all():
            raise ValueError("Common tests must contain every camera on every cycle")
        compensation_comparisons(
            table, output, contrasts=list(combinations(CAMERAS, 2)), main_versions=list(CAMERAS)
        )
    historical = (
        training_output(OUTPUT / REPRODUCTION / "01_视觉模型训练" / "网络比较_LSTM25_FiLM")
        / "cycle_metrics.csv"
    )
    if historical.exists():
        old = pd.read_csv(historical).loc[lambda t: t.seed.eq(args.seed)]
        common = set(cohort.loc[cohort.data_role.eq("first") & cohort.common_test, "cycle_name"])
        pd.DataFrame(
            [
                dict(
                    evaluation_scope=scope,
                    method="historical front, per-fold inner epoch selection",
                    seed=args.seed,
                    **summarize_metrics(part),
                )
                for scope, part in [
                    ("first_own", old),
                    ("first_common", old.loc[old.cycle_name.isin(common)]),
                ]
            ]
        ).to_csv(args.output / "historical_front_summary.csv", index=False)
    from plots.camera import render_camera_comparison

    render_camera_comparison(args.output)


def summarize_near_sets(args):
    """Describe the frozen front test cohorts, summing disconnected near-set bands."""
    import pickle

    from image_models.checkpoints import checkpoint_files
    from image_models.policy import near_set_timing

    metrics = pd.read_csv(args.output / "comparison_cycles.csv")
    metrics = metrics.loc[
        metrics.camera_role.eq("front")
        & metrics.evaluation_scope.isin(["first_common", "second_common"])
    ].copy()
    metrics.selected_time = pd.to_datetime(metrics.selected_time)
    records = []
    for path in checkpoint_files(model_arguments(args, "front").output):
        saved = pickle.load(path.open("rb"))
        predicted = saved["predictions"]
        for scope, part in metrics.loc[metrics.seed.eq(saved["checkpoint"]["seed"])].groupby(
            "evaluation_scope"
        ):
            rows = predicted.loc[
                predicted.split.eq("test") & predicted.cycle_name.isin(part.cycle_name)
            ].copy()
            if rows.empty:
                continue
            rows.candidate_defrost_time = pd.to_datetime(rows.candidate_defrost_time)
            _, bands = near_set_timing(rows, part)
            bands["near_minutes"] = (bands.end - bands.start).dt.total_seconds() / 60
            duration = bands.groupby("cycle_name").near_minutes.sum()
            available = rows.loc[rows.input_valid & rows.cycle_cop_eligible]
            fraction = available.groupby("cycle_name").near_optimal.mean()
            selected = part.loc[
                part.cycle_name.isin(rows.cycle_name),
                [
                    "evaluation_scope",
                    "camera_role",
                    "seed",
                    "cycle_name",
                    "experiment_id",
                    "near_components",
                ],
            ].copy()
            records.append(
                selected.assign(
                    near_minutes=selected.cycle_name.map(duration),
                    near_action_fraction=selected.cycle_name.map(fraction),
                )
            )
    source = pd.concat(records, ignore_index=True)
    if (
        len(source) != len(metrics)
        or source.duplicated(["evaluation_scope", "seed", "cycle_name"]).any()
    ):
        raise ValueError(
            "Frozen near-set diagnostics must cover each front test cycle exactly once"
        )
    source.to_csv(args.output / "camera_near_set_source.csv", index=False)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    parser.add_argument("--output", type=Path, default=(PROJECT / "output/01_核心结果/六机位比较"))
    parser.add_argument(
        "--stage",
        choices=["index", "prepare", "train", "summarize", "plot", "images", "all"],
        required=True,
        help="显式选择阶段；prepare/all 会读取原图并更新 Dataset，train 使用已发布特征，summarize 汇总已有六机位结果",
    )
    parser.add_argument(
        "--comparison-layout",
        choices=["overview", "grouped", "diagnostic"],
        default="overview",
        help="Figure layout for --stage plot; uses frozen result tables without fitting",
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--maximum-epochs", type=int, default=160)
    parser.add_argument("--tune-trials", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standard-through", default="exp_20260910")
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=CAMERAS,
        default=list(CAMERAS),
        help="index、prepare、train 的目标机位；summarize/plot/images 使用完整六机位研究结果",
    )
    parser.add_argument(
        "--test-experiments", nargs="+", help="Run-isolated first-test fold subset for smoke checks"
    )
    parser.add_argument(
        "--frost-progression", type=Path, help="Saved trajectory; default follows --frost-scale"
    )
    parser.add_argument("--frost-scale", choices=["native", "groups16"], default="native")
    parser.add_argument(
        "--frost-grades",
        nargs="+",
        type=float,
        help="Six increasing front reference grades; default native 0–25 or groups16 1,4,7,10,13,16",
    )
    parser.add_argument("--image-layout", choices=["full", "square"], default="full")
    parser.add_argument(
        "--image-cycle", help="Optional single cycle for the early-frost image strip"
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.tune_trials < 1 or args.maximum_epochs < 1 or args.n_jobs < 1:
        parser.error("Trials, epochs and worker count must be positive")
    if (
        args.test_experiments or args.maximum_epochs != 160
    ) and args.output.resolve().is_relative_to(OUTPUT):
        parser.error("Partial/smoke runs belong in temp/test/<experiment>")
    args.output = analysis_output(args.output)
    if args.stage == "all" and set(args.cameras) != set(CAMERAS):
        parser.error(
            "Subset cameras: choose index, prepare or train; all includes the full six-camera summary"
        )
    print(
        f"Stage: {args.stage}\nDataset: {args.dataset.resolve()}\nAnalysis: {args.output.resolve()}",
        flush=True,
    )
    if args.stage in {"prepare", "all"}:
        print(
            f"Dataset writes: {args.dataset.resolve() / 'image_DINO_feature'}, {args.dataset.resolve() / 'training_data'}",
            flush=True,
        )
    for camera in args.cameras if args.stage in {"index", "prepare", "train", "all"} else CAMERAS:
        paths = camera_outputs(args.output, camera)
        if args.stage in {"prepare", "all"}:
            print(f"{camera}: quality={paths['admission']}; archive={paths['archive']}", flush=True)
        if args.stage in {"train", "all", "summarize"}:
            print(f"{camera}: training={paths['training']}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "index":
        image_index(args.dataset, args.output, args.cameras)
    if args.stage in ["prepare", "all"]:
        prepare(args)
    if args.stage in ["train", "all"]:
        train(args)
    if args.stage in ["summarize", "all"]:
        summarize(args)
    if args.stage == "plot":
        from plots.camera import render_camera_comparison

        if args.comparison_layout == "diagnostic":
            summarize_near_sets(args)
        render_camera_comparison(args.output, layout=args.comparison_layout)
    if args.stage == "images":
        from plots.camera import render_camera_images

        render_camera_images(args)


if __name__ == "__main__":
    main()
