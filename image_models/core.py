"""Three parallel Binary models on the same inputs, reference curves and folds."""

import json
import pickle

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed, parallel_config
from torch import nn

from image_models.inputs import RGB, SENSORS, STATISTICS

PARENTS = {"F_core": None, "B0": None, "G_core": "F_core"}


def contract(name):
    if name not in PARENTS:
        raise ValueError(f"Unknown model: {name}")
    if name == "B0":
        return [], RGB.copy()
    numeric = [f"stat_{sensor}_{stat}" for sensor in SENSORS for stat in STATISTICS]
    numeric += ["causal_elapsed_minutes", "causal_total_electricity_kwh", "pre_defrost_heat_kwh"]
    return numeric, RGB.copy() if name == "G_core" else []


def model(name, numeric_width, visual_width, seed):
    torch.manual_seed(seed)
    width = numeric_width + visual_width
    if name == "B0":
        return nn.Sequential(
            nn.Linear(width, 1000), nn.ReLU(), nn.Linear(1000, 64), nn.ReLU(), nn.Linear(64, 2)
        )
    if name not in PARENTS:
        raise ValueError(f"Unknown model: {name}")
    widths = [256, 128, 64, 32] if name == "G_core" else [60, 60, 32, 32]
    layers = []
    for hidden in widths:
        layers += [nn.Linear(width, hidden), nn.SiLU()]
        width = hidden
    net = nn.Sequential(*layers, nn.Linear(width, 1), nn.Flatten(0, 1))
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    return net


def fold_exclusions(experiments):
    ordered = sorted(set(experiments))
    return {test: ordered[(index + 1) % len(ordered)] for index, test in enumerate(ordered)}


def save_settings(path, settings):
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("settings changed; use a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")


def common_actions(rows):
    """Select the clock after defining the common full valid-curve optimum."""
    from image_models.inputs import processing_rows

    rows = rows.sort_values(["cycle_name", "candidate_defrost_time"]).copy()
    time = pd.to_datetime(rows.candidate_defrost_time)
    end = pd.to_datetime(rows.observation_end)
    if "observed_defrost_preparation_start" in rows:
        end = pd.to_datetime(rows.observed_defrost_preparation_start).fillna(end)
    physical = time.ge(pd.to_datetime(rows.stable_heating_start)) & time.lt(end)
    valid = physical & rows.cycle_cop_eligible.fillna(False) & np.isfinite(rows.cycle_cop)
    peaks = rows.loc[valid].groupby("cycle_name").cycle_cop.max()
    optimum = (
        rows.loc[valid & rows.cycle_cop.eq(rows.cycle_name.map(peaks))]
        .groupby("cycle_name")
        .candidate_defrost_time.min()
    )
    rows["optimal_time"] = rows.cycle_name.map(optimum)
    rows["optimal_cop"] = rows.cycle_name.map(peaks)
    rows["after_optimum_target"] = time.ge(rows.optimal_time).astype(float)
    return rows.loc[processing_rows(rows) & valid & rows.optimal_cop.gt(0)].reset_index(drop=True)


def prepare(args, cohort):
    from dataset_tools import DatasetLoader
    from defrost_decision.baselines.electricity import integrate_heating_curve

    loader = DatasetLoader(args.dataset)
    for boundary in cohort.itertuples():
        path = args.output / "base" / f"{boundary.cycle_name}.parquet"
        if path.exists():
            continue
        rows = pd.read_parquet(args.data / "base" / path.name)
        raw = loader.load_cycle_original(boundary.cycle_name)
        times = pd.to_datetime(rows.candidate_defrost_time)
        start = pd.Timestamp(boundary.heating_start)
        causal = integrate_heating_curve(
            raw.timestamp, raw.power_total, times, start, "strict_causal", causal_candidates=True
        )
        rows["causal_elapsed_minutes"] = (times - start).dt.total_seconds() / 60
        rows["causal_total_electricity_kwh"] = causal.strict_energy_kwh.where(
            causal.strict_coverage.gt(0)
        ).to_numpy()
        rows["joint_input_available"] = rows.sensor_timestamp.notna() & rows.rgb_available
        for field in (
            "stable_heating_start",
            "observation_end",
            "observed_defrost_preparation_start",
            "t_RB",
        ):
            rows[field] = getattr(boundary, field)
        rows.to_parquet(path, index=False)
        print(f"[inputs] {boundary.cycle_name}", flush=True)


def fold(args, cohort, events, test, inner):
    from image_models.inputs import build_fold_rows, processing_rows
    from image_models.policy import fit_policy, policy_cycle_metrics, predict_policy

    parents = {name: PARENTS[name] for name in args.models}
    recipes = list(parents)
    if all(
        (args.output / name / f"seed_{seed}" / "folds" / f"{test}.pkl").exists()
        for name in parents
        for seed in args.seeds
    ):
        return
    nested = cohort.loc[~cohort.experiment_id.eq(test)]
    inner_rows, _ = build_fold_rows(args, nested, events, (test, inner))
    outer_rows, _ = build_fold_rows(args, cohort, events, (test,))
    training = common_actions(inner_rows.loc[~inner_rows.experiment_id.eq(inner)])
    validation = common_actions(inner_rows.loc[inner_rows.experiment_id.eq(inner)])
    outer_train = common_actions(outer_rows.loc[~outer_rows.experiment_id.eq(test)])
    outer_test = common_actions(outer_rows.loc[outer_rows.experiment_id.eq(test)])
    reference = outer_rows.loc[outer_rows.experiment_id.eq(test)].copy()
    reference.to_parquet(args.output / "ridge" / f"{test}_reference.parquet", index=False)
    clock = reference.loc[processing_rows(reference)]
    val_clock = inner_rows.loc[inner_rows.experiment_id.eq(inner)]
    val_clock = val_clock.loc[processing_rows(val_clock)]
    empty = [
        label
        for label, table in [
            ("training", training),
            ("validation", validation),
            ("outer_training", outer_train),
            ("test", outer_test),
        ]
        if table.empty
    ]
    if empty:
        for name in parents:
            for seed in args.seeds:
                destination = args.output / name / f"seed_{seed}" / "folds" / f"{test}.pkl"
                if destination.exists():
                    continue
                metrics = pd.DataFrame(
                    dict(
                        cycle_name=sorted(set(reference.cycle_name)),
                        selected_cop=np.nan,
                        optimal_cop=np.nan,
                        rb_cop=np.nan,
                        forced_final=False,
                        early_trigger=False,
                        soft_hard_gap=np.nan,
                        evaluation_status="no_valid_" + "_".join(empty),
                    )
                )
                result = dict(
                    cycle_metrics=metrics,
                    predictions=pd.DataFrame(),
                    losses=pd.DataFrame(),
                    heldout_experiment=test,
                    inner_experiment=inner,
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    pickle.dump(result, stream)
                temporary.replace(destination)
        print(f"[models] {test}: no valid {empty}", flush=True)
        return
    for name in recipes:
        loss_name = "after_optimum"
        for seed in args.seeds:
            path = args.output / name / f"seed_{seed}" / "folds" / f"{test}.pkl"
            if path.exists():
                continue
            numeric, visual = contract(name)
            options = dict(
                architecture=name,
                loss_name=loss_name,
                numeric_columns=numeric,
                visual_columns=visual,
                seed=seed,
                patience=args.patience,
                cycles_per_batch=args.batch_size,
                thresholds=(0.5,),
            )
            fitted = fit_policy(
                training,
                validation,
                maximum_epochs=args.maximum_epochs,
                controller_reference=val_clock,
                **options,
            )
            checkpoint = fit_policy(
                outer_train, None, maximum_epochs=fitted["selected_epoch"], **options
            )
            predictions = predict_policy(outer_test, checkpoint)
            losses = pd.concat(
                [fitted["losses"].assign(stage="inner"), checkpoint["losses"].assign(stage="outer")]
            )
            controller = "first_positive"
            metrics = (
                policy_cycle_metrics(
                    predictions,
                    reference,
                    0.5,
                    loss_name=loss_name,
                    strategy=controller,
                    controller_reference=clock,
                )
                if len(predictions)
                else pd.DataFrame(columns=["cycle_name", "selected_cop", "rb_cop"])
            )
            missing = sorted(set(reference.cycle_name) - set(metrics.cycle_name))
            if missing:
                metrics = pd.concat(
                    [
                        metrics,
                        pd.DataFrame(
                            dict(
                                cycle_name=missing,
                                selected_cop=np.nan,
                                optimal_cop=np.nan,
                                rb_cop=np.nan,
                                forced_final=False,
                                early_trigger=False,
                                strategy=controller,
                                soft_hard_gap=np.nan,
                            )
                        ),
                    ],
                    ignore_index=True,
                )
            metrics["evaluation_status"] = np.where(
                metrics.selected_cop.isna(),
                "no_valid_actions",
                np.where(metrics.rb_cop.notna(), "evaluated", "rb_unavailable"),
            )
            result = dict(
                checkpoint=checkpoint,
                predictions=predictions[
                    [
                        "cycle_name",
                        "experiment_id",
                        "candidate_defrost_time",
                        "probability",
                        "cycle_cop",
                        "optimal_cop",
                        "optimal_time",
                        "after_optimum_target",
                        "cycle_cop_uses_model_extrapolation",
                    ]
                ],
                cycle_metrics=metrics,
                heldout_experiment=test,
                inner_experiment=inner,
                losses=losses,
            )
            destination = args.output / name / f"seed_{seed}" / "folds" / f"{test}.pkl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                pickle.dump(result, stream)
            temporary.replace(destination)
            print(f"[model] {test} {name} seed={seed} epoch={fitted['selected_epoch']}", flush=True)


def summarize(output, parents=PARENTS):
    from image_models.policy import summarize_metrics

    summaries, tables = [], []
    for name in parents:
        for folder in sorted((output / name).glob("seed_*")):
            seed = int(folder.name.removeprefix("seed_"))
            results = []
            for path in sorted((folder / "folds").glob("*.pkl")):
                with path.open("rb") as stream:
                    results.append(pickle.load(stream))
            if not results:
                continue
            for key, suffix in [
                ("cycle_metrics", "csv"),
                ("predictions", "parquet"),
                ("losses", "csv"),
            ]:
                frame = pd.concat(
                    [r[key].assign(heldout_experiment=r["heldout_experiment"]) for r in results],
                    ignore_index=True,
                )
                getattr(frame, f"to_{suffix}")(folder / f"{key}.{suffix}", index=False)
                if key == "cycle_metrics":
                    tables.append(frame.assign(version=name, seed=seed))
                    summaries.append(
                        dict(
                            version=name,
                            seed=seed,
                            no_valid_actions=int(frame.selected_cop.isna().sum()),
                            forced_final_rate=float(
                                frame.loc[frame.selected_cop.notna(), "forced_final"].mean()
                            ),
                            **summarize_metrics(frame),
                        )
                    )
    summary = pd.DataFrame(summaries)
    if summary.empty:
        return
    all_cycles = pd.concat(tables, ignore_index=True)
    all_cycles.to_csv(output / "cycle_metrics.csv", index=False)
    scored = all_cycles.loc[all_cycles.selected_cop.notna() & all_cycles.rb_cop.notna()]
    coverage = scored.groupby(["seed", "cycle_name"]).version.transform("nunique")
    common = scored.loc[coverage.eq(len(parents))]
    for index, row in summary.iterrows():
        paired = common.loc[common.version.eq(row.version) & common.seed.eq(row.seed)]
        for key, value in summarize_metrics(paired).items():
            summary.loc[index, f"common_{key}"] = value
    for name, parent in parents.items():
        for reference, tag in [(parent, "parent"), ("B0", "chen")]:
            if reference is None or reference not in parents:
                continue
            baseline = summary.loc[summary.version.eq(reference)].set_index("seed")
            selected = summary.version.eq(name)
            for column, metric in [
                ("mean_cycle_cop", "cop"),
                ("headroom_captured_pct", "headroom_pp"),
            ]:
                column = f"common_{column}"
                summary.loc[selected, f"delta_{tag}_{metric}"] = summary.loc[
                    selected, column
                ] - summary.loc[selected, "seed"].map(baseline[column])
    summary.to_csv(output / "summary.csv", index=False)
    numeric = summary.select_dtypes("number").columns.drop("seed")
    summary.groupby("version", sort=False)[numeric].agg(["mean", "std"]).to_csv(
        output / "effects.csv"
    )


def run(args):
    from image_models.inputs import reliable_rgb_cohort

    args.output.mkdir(parents=True, exist_ok=True)
    for directory in ("base", "ridge"):
        (args.output / directory).mkdir(exist_ok=True)
    settings = json.loads((args.data / "settings.json").read_text())
    parents = {name: PARENTS[name] for name in args.models}
    args.rgb, args.require_rgb_input = "off", False
    args.allow_extrapolation = True
    cohort = reliable_rgb_cohort(
        args.dataset, source=pd.read_csv(args.data / "recovery_boundaries.csv")
    )
    event_settings = json.loads((args.event_run / "run_settings.json").read_text())
    if (
        event_settings["preparation_heat"] != "zero"
        or event_settings["recovery_settings"] != settings["causal_boundary_settings"]
    ):
        raise ValueError("Event and candidate recovery/heat definitions must match")
    events = pd.read_csv(args.event_run / "defrost_events.csv")
    cohort.to_csv(args.output / "cohort.csv", index=False)
    configuration = dict(
        data=str(args.data.resolve()),
        event_run=str(args.event_run.resolve()),
        parents=parents,
        seeds=args.seeds,
        model_layers={
            name: [
                [layer.in_features, layer.out_features]
                for layer in model(
                    name, len(contract(name)[0]), len(contract(name)[1]), 0
                ).modules()
                if isinstance(layer, nn.Linear)
            ]
            for name in parents
        },
        frozen_predictions_from=None,
        controller="1/1@0.5",
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
        batch_cycles=args.batch_size,
        threshold=0.5,
        clock_seconds=30,
        reference="full physically valid curve including assumed extrapolation before clock and input filtering",
        allow_model_extrapolation=True,
        missing_input="training-median imputation without row filtering",
        checkpoint="validation COP at fixed 2/3@0.5 for every recipe",
        no_trigger="last valid action fallback for every recipe",
        binary_loss="cycle and present-class equal BCE",
        initialization="Xavier normal except original Chen default",
        optimizer="AdamW lr=0.001 weight_decay=0.0001",
        causal_boundary_settings=settings["causal_boundary_settings"],
    )
    config_path = args.output / "settings.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Training protocol changed; select a new output directory")
    save_settings(config_path, configuration)
    prepare(args, cohort)
    inners = fold_exclusions(sorted(cohort.experiment_id.unique()))
    tests = [args.heldout_experiment] if args.heldout_experiment else sorted(inners)
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        Parallel()(delayed(fold)(args, cohort, events, test, inners[test]) for test in tests)
    summarize(args.output, parents)
