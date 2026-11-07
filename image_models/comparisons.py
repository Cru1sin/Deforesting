"""Summarize and compare saved model experiments through shared policy metrics."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from image_models.checkpoints import checkpoint_files
from image_models.cycle_selection import screen_published_cycles
from image_models.experiments import common_cycle_metrics
from image_models.policy import policy_timing_metrics, summarize_metrics
from output_layout import experiment_root


def write_evaluation_metrics(metrics, dataset, output, camera_role="front"):
    """One table: all/early/good RB subsets × overall/setpoint, separately per seed.

    Good RB means supported RB COP >=99% of that cycle's oracle COP. It is
    not the complement of the catalogued early subset. Missing decisions stay
    in the hit-rate denominator; COP and timing expose their own paired counts.
    """
    from dataset_tools import DatasetLoader

    dataset = DatasetLoader(dataset).dataset_root
    screening = screen_published_cycles(dataset, camera_role)
    screening.to_csv(output / "cycle_screening.csv", index=False)
    rows = metrics.loc[
        metrics.cycle_name.isin(screening.loc[screening.selected, "cycle_name"])
    ].copy()
    reference = pd.read_parquet(Path(dataset) / "training_data" / "reference.parquet")
    condition = reference.groupby("cycle_name").water_temperature_setpoint.median()
    rows["setpoint_c"] = rows.cycle_name.map(condition)
    rows["complete_evaluation_eligible"] = rows.cycle_name.map(
        screening.set_index("cycle_name").selected_evaluation
    )
    rows = pd.concat(
        [
            policy_timing_metrics(g, reference)
            for _, g in rows.groupby(["version", "seed", "split"])
        ],
        ignore_index=True,
    )
    early_path = Path(dataset) / "RB失败循环" / "早除霜" / "cohort.csv"
    early = pd.read_csv(early_path).cycle_name if early_path.exists() else []
    rows["rb_early"] = rows.cycle_name.isin(early)
    rows["rb_good"] = rows.rb_cop.ge(0.99 * rows.optimal_cop) & rows.rb_cop.notna()
    subsets = [
        ("all", rows),
        ("rb_early", rows.loc[rows.rb_early]),
        ("rb_good", rows.loc[rows.rb_good]),
    ]
    settings = json.loads((Path(dataset) / "training_data" / "settings.json").read_text())
    if settings.get("historical_strict_cohort"):
        historical = pd.read_csv(settings["historical_strict_cohort"]).cycle_name
        rows["historical_strict_cohort"] = rows.cycle_name.isin(historical)
        subsets.extend(
            [
                ("historical_strict", rows.loc[rows.historical_strict_cohort]),
                ("newly_admitted", rows.loc[~rows.historical_strict_cohort]),
            ]
        )
    records = []
    for subset, table in subsets:
        for (version, seed, split), g in table.groupby(["version", "seed", "split"]):
            groups = [
                ("all", g),
                *[
                    (value, g.loc[g.setpoint_c.eq(value)])
                    for value in sorted(rows.setpoint_c.dropna().unique())
                ],
            ]
            for setpoint, part in groups:
                records.append(
                    dict(
                        subset=subset,
                        setpoint_c=setpoint,
                        version=version,
                        seed=seed,
                        split=split,
                        **summarize_metrics(part),
                        complete_evaluation_cycles=int(
                            part.complete_evaluation_eligible.fillna(False).sum()
                        ),
                        unsupported_cop_cycles=int(part.selected_cop.isna().sum()),
                        no_trigger_cycles=int(part.no_trigger.fillna(False).sum())
                        if "no_trigger" in part
                        else 0,
                        fallback_cycles=int(part.forced_final.fillna(False).sum())
                        if "forced_final" in part
                        else 0,
                        improved_cycles=int(part.selected_cop.gt(part.rb_cop).sum()),
                        worsened_cycles=int(part.selected_cop.lt(part.rb_cop).sum()),
                    )
                )
    rows.to_csv(output / "evaluation_cycles.csv", index=False)
    summary = pd.DataFrame(records)
    summary.to_csv(output / "evaluation_summary.csv", index=False)


def paired_cycle_changes(metrics, left, right):
    """Retain every cycle, including newly unsupported triggers, in each contrast."""
    keys = ["cycle_name", "experiment_id", "seed"]
    fields = [
        c
        for c in [
            "selected_cop",
            "regret_pct",
            "selected_time",
            "trigger_status",
            "fallback_terminal",
        ]
        if c in metrics
    ]
    arms = [
        metrics.loc[metrics.version.eq(v), keys + fields].rename(
            columns={c: f"{side}_{c}" for c in fields}
        )
        for v, side in [(left, "left"), (right, "right")]
    ]
    changes = arms[0].merge(arms[1], on=keys, how="outer", validate="one_to_one")
    changes["delta_cop"] = changes.right_selected_cop - changes.left_selected_cop
    if "selected_time" in fields:
        changes["trigger_delay_minutes"] = (
            pd.to_datetime(changes.right_selected_time, format="mixed")
            - pd.to_datetime(changes.left_selected_time, format="mixed")
        ).dt.total_seconds() / 60
    a, b = changes.left_selected_cop.notna(), changes.right_selected_cop.notna()
    changes["change"] = np.select(
        [a & ~b, ~a & b, ~a & ~b, changes.delta_cop.gt(0), changes.delta_cop.lt(0)],
        ["newly_unsupported", "newly_evaluable", "both_unsupported", "improved", "worsened"],
        default="unchanged",
    )
    return changes.assign(left=left, right=right)


def compensation_comparisons(metrics, output, contrasts=None, main_versions=None):
    """Paired seed effects and experiment-block uncertainty on the fixed test set."""
    test = metrics.loc[metrics.split.eq("test")]
    pairs, uncertainty, paired_common, cycle_changes = [], [], [], []
    if contrasts is None:
        contrasts = [
            ("RGB_COP_basin_midpoint_label", "RGB_frost_frequency_label"),
            ("RGB_frost_frequency_label", "RGB_frost_frequency_label_plus_frequency"),
            (
                "RGB_frost_frequency_label_plus_water_setpoint",
                "RGB_frost_frequency_label_plus_frequency",
            ),
            ("RGB_COP_basin_midpoint_label", "RGB_frost_frequency_label_plus_frequency"),
        ]
        if "RGB_common_frost_threshold_label" in metrics.version.unique():
            contrasts += [
                ("RGB_common_frost_threshold_label", "RGB_frost_frequency_label"),
                ("RGB_common_frost_threshold_label", "RGB_frost_frequency_label_plus_frequency"),
            ]
    for left, right in contrasts:
        cycle_changes.append(paired_cycle_changes(test, left, right))
        selected = test.loc[test.version.isin([left, right])]
        wide = selected.pivot(
            index=["cycle_name", "experiment_id", "seed"], columns="version", values="selected_cop"
        )
        # Pairing is explicit; missing COP remains a failure in the all-cycle hit table.
        for seed, group in wide.groupby("seed"):
            paired = group[[left, right]].dropna()
            paired_rows = selected.loc[
                selected.seed.eq(seed)
                & selected.cycle_name.isin(paired.index.get_level_values("cycle_name"))
            ]
            captured = {
                version: summarize_metrics(g)["headroom_captured_pct"]
                for version, g in paired_rows.groupby("version")
            }
            left_capture, right_capture = captured.get(left, np.nan), captured.get(right, np.nan)
            pairs.append(
                dict(
                    left=left,
                    right=right,
                    seed=seed,
                    paired_cycles=len(paired),
                    delta_cop=(paired[right] - paired[left]).mean(),
                    relative_cop_gain_pct=100 * (paired[right].mean() / paired[left].mean() - 1),
                    left_headroom_captured_pct=left_capture,
                    right_headroom_captured_pct=right_capture,
                    headroom_capture_delta_pp=right_capture - left_capture,
                )
            )
            if "trigger_delay_minutes" in cycle_changes[-1]:
                timing = (
                    cycle_changes[-1]
                    .loc[
                        lambda x: (
                            x.seed.eq(seed)
                            & x.cycle_name.isin(paired.index.get_level_values("cycle_name"))
                        ),
                        "trigger_delay_minutes",
                    ]
                    .dropna()
                )
                pairs[-1].update(
                    timing_paired_cycles=len(timing),
                    mean_trigger_delay_minutes=timing.mean(),
                    median_trigger_delay_minutes=timing.median(),
                    later_cycles=int(timing.gt(0).sum()),
                    earlier_cycles=int(timing.lt(0).sum()),
                )
        counts = wide[[left, right]].notna().groupby(["cycle_name", "experiment_id"]).sum()
        all_seeds = counts.index[counts.eq(test.seed.nunique()).all(axis=1)]
        shared = selected.loc[selected.cycle_name.isin(all_seeds.get_level_values("cycle_name"))]
        paired_common.extend(
            dict(left=left, right=right, version=version, seed=seed, **summarize_metrics(g))
            for (version, seed), g in shared.groupby(["version", "seed"])
        )
        cycle_means = (
            wide.loc[wide.index.droplevel("seed").isin(all_seeds)]
            .groupby(["cycle_name", "experiment_id"])[[left, right]]
            .mean()
        )
        days = cycle_means.groupby("experiment_id")[[left, right]].mean()
        # Seeds repeat the same experiments. Average seeds first, then resample
        # entire experiments with equal weight; 2,000 vectorized draws, no refits.
        deltas = (days[right] - days[left]).to_numpy()
        low, high = np.nan, np.nan
        pooled_delta, pooled_low, pooled_high = np.nan, np.nan, np.nan
        if len(days):
            draws = np.random.default_rng(0).integers(len(days), size=(2000, len(days)))
            bootstrap = deltas[draws].mean(axis=1)
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            reward = shared.groupby(["cycle_name", "experiment_id"])[
                ["rb_cop", "optimal_cop"]
            ].mean()
            reward_days = reward.groupby("experiment_id").sum().reindex(days.index)
            gain_days = (
                cycle_means.groupby("experiment_id")[[left, right]].sum().reindex(days.index)
            )
            gains = (gain_days[right] - gain_days[left]).to_numpy()
            budgets = (reward_days.optimal_cop - reward_days.rb_cop).to_numpy()
            denominator = budgets[draws].sum(axis=1)
            # H_RB is a pooled ratio; resample entire experiments, never individual seeds.
            if budgets.sum() > 0:
                pooled_delta = 100 * gains.sum() / budgets.sum()
                valid = denominator > 0
                pooled_bootstrap = 100 * gains[draws][valid].sum(axis=1) / denominator[valid]
                pooled_low, pooled_high = np.quantile(pooled_bootstrap, [0.025, 0.975])
        uncertainty.append(
            dict(
                left=left,
                right=right,
                test_experiments=len(days),
                paired_cycles=len(cycle_means),
                pairing_scope="both arms evaluable in every seed",
                experiment_equal_delta_cop=deltas.mean() if len(days) else np.nan,
                bootstrap_lower_95=low,
                bootstrap_upper_95=high,
                pooled_H_delta_pp=pooled_delta,
                pooled_H_lower_95=pooled_low,
                pooled_H_upper_95=pooled_high,
                positive_experiments=int((deltas > 0).sum()),
                bootstrap_draws=2000,
            )
        )
    pd.DataFrame(pairs).to_csv(output / "paired_effects.csv", index=False)
    pd.DataFrame(uncertainty).to_csv(output / "experiment_bootstrap.csv", index=False)
    pd.DataFrame(paired_common).to_csv(output / "paired_common_summary.csv", index=False)
    if cycle_changes:
        pd.concat(cycle_changes, ignore_index=True).to_csv(
            output / "cycle_changes.csv", index=False
        )
    per_day = []
    primary = metrics if main_versions is None else metrics.loc[metrics.version.isin(main_versions)]
    common = common_cycle_metrics(primary)
    common = common.loc[
        common.groupby("cycle_name").seed.transform("nunique").eq(metrics.seed.nunique())
    ]
    common_summary = [
        dict(
            split=s, version=v, seed=k, **summarize_metrics(g), mean_regret_pct=g.regret_pct.mean()
        )
        for (s, v, k), g in common.groupby(["split", "version", "seed"])
    ]
    if "terminal_cop" in common:
        for record, (_, group) in zip(common_summary, common.groupby(["split", "version", "seed"])):
            terminal = summarize_metrics(group.assign(rb_cop=group.terminal_cop))
            record.update(
                terminal_paired_cycles=terminal["rb_paired_cycles"],
                terminal_headroom_captured_pct=terminal["headroom_captured_pct"],
                gain_vs_terminal_pct=terminal["gain_vs_rb_pct"],
            )
    pd.DataFrame(common_summary).to_csv(output / "common_summary.csv", index=False)
    for scope, table in [("available_cop", metrics), ("common_all_seeds", common)]:
        for (split, version, seed, day), group in table.groupby(
            ["split", "version", "seed", "experiment_id"]
        ):
            per_day.append(
                dict(
                    scope=scope,
                    split=split,
                    version=version,
                    seed=seed,
                    experiment_id=day,
                    **summarize_metrics(group),
                    hit_rate_all=group.near_optimal_hit.eq(True).mean(),
                    no_trigger_count=int(group.no_trigger.sum()),
                    unsupported_count=int(group.unsupported_trigger.sum()),
                    mean_regret_pct=group.regret_pct.mean(),
                )
            )
    pd.DataFrame(per_day).to_csv(output / "experiment_metrics.csv", index=False)


def write_ablation_tables(runs, output):
    """Plot-ready settings, selected scores and all saved inner/refit epoch curves."""
    from image_models.policy import build_model, restore_policy, summarize_metrics
    from output_layout import analysis_output

    output = analysis_output(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries, curves, trials = [], [], []
    for run in runs:
        run = experiment_root(run)
        settings_path = run / "training/settings.json"
        if not settings_path.exists():
            settings_path = run / "settings.json"
        if not settings_path.exists():
            continue
        settings = json.loads(settings_path.read_text())
        paths = checkpoint_files(run)
        outer_experiments = set()
        cohort_experiments = set()
        for path in paths:
            saved = pickle.load(path.open("rb"))
            metrics = saved["cycle_metrics"]
            outer_experiments.update(metrics.experiment_id.unique())
            cohort_experiments.update(saved["checkpoint"].get("training_experiments", []))
        cohort_experiments.update(outer_experiments)
        evaluation_scope = (
            "full_experiment_loeo"
            if cohort_experiments and outer_experiments == cohort_experiments
            else "partial_experiment_holdout"
        )
        tests = []
        for path in paths:
            saved = pickle.load(path.open("rb"))
            model = saved["checkpoint"]
            actual = restore_policy(model)
            metrics = saved["cycle_metrics"]
            outer = (
                metrics.outer_experiment.iloc[0]
                if "outer_experiment" in metrics
                else metrics.experiment_id.iloc[0]
            )
            fields = dict(
                experiment=run.name,
                source_run=str(run),
                checkpoint=path.stem,
                outer_experiment=outer,
                version=metrics.version.iloc[0],
                seed=model["seed"],
                architecture=model["architecture"],
                hidden_width=model.get("hidden_width", 32),
                recurrent_layers=model.get("recurrent_layers", 1),
                mlp_layers=model.get("mlp_layers", 2),
                mlp_widths=json.dumps(model.get("mlp_widths")),
                weight_decay=model.get("weight_decay", 1e-4),
                reset_history=model.get("reset_history", False),
                parameters=sum(p.numel() for p in actual.parameters()),
                loss=model["loss_name"],
                cdf_score=model.get("cdf_score", settings.get("cdf_score", "bce")),
                cdf_link=model.get("cdf_link", settings.get("cdf_link", "cumulative")),
                cdf_regret_weight=settings.get("cdf_regret_weight", 0.0),
                validation_cdf_score=model.get(
                    "validation_cdf_score", settings.get("validation_cdf_score", "training")
                ),
                learning_rate=model.get("learning_rate", settings.get("learning_rate", 0.001)),
                visual_dropout=actual.visual_dropout if hasattr(actual, "visual_dropout") else 0.0,
                output_dropout=actual.output_dropout.p
                if hasattr(actual, "output_dropout")
                else 0.0,
                trigger_rule=settings.get("trigger_rule"),
                validation_utility=settings.get("validation_utility", "cop"),
                maximum_epochs=settings.get("maximum_epochs"),
                cohort_rule=settings.get("data_definition", {}).get("selection"),
                visual_time_shared=getattr(actual, "visual_time_shared", None),
                output_time_shared=getattr(actual, "output_time_shared", None),
                inputs=";".join(model["numeric_columns"]),
                numeric_dimensions=len(model["numeric_columns"]),
                visual_dimensions=len(model["visual_columns"]),
                visual_pooling=settings.get("visual_pooling", "cls"),
                selected_epoch=model["selected_epoch"],
                completed_outer_folds=len(paths),
                expected_outer_folds=len(settings.get("folds", {})),
                evaluation_scope=evaluation_scope,
                cohort_experiments=len(cohort_experiments),
            )
            selection = model.get("pooled_validation_scores")
            if selection is not None:
                summaries.append(
                    dict(**fields, scope="inner_validation", **selection.iloc[0].to_dict())
                )
            summaries.append(dict(**fields, scope="outer_fold", **summarize_metrics(metrics)))
            tests.append(metrics)
            trial_files = list(path.parent.glob("*_trials.csv"))
            for trial_path in trial_files:
                frame = pd.read_csv(trial_path)
                frame = frame.assign(
                    **fields, scope="hyperparameter_trial", history_source=str(trial_path)
                )
                for key in [
                    "architecture",
                    "hidden_width",
                    "mlp_layers",
                    "recurrent_layers",
                    "learning_rate",
                    "weight_decay",
                    "visual_dropout",
                    "output_dropout",
                ]:
                    if "params_" + key in frame:
                        frame[key] = frame["params_" + key].fillna(fields[key])
                if "user_attrs_selected_epoch" in frame:
                    frame["selected_epoch"] = frame.user_attrs_selected_epoch
                if "params_dropout_place" in frame:
                    rate = frame.get(
                        "params_dropout_rate", pd.Series(0.0, index=frame.index)
                    ).fillna(0.0)
                    frame["visual_dropout"] = rate.where(
                        frame.params_dropout_place == "visual", 0.0
                    )
                    frame["output_dropout"] = rate.where(
                        frame.params_dropout_place == "output", 0.0
                    )
                    shared = frame.get("params_time_shared", pd.Series(True, index=frame.index))
                    frame["visual_time_shared"] = shared.where(
                        frame.params_dropout_place == "visual", True
                    )
                    frame["output_time_shared"] = shared.where(
                        frame.params_dropout_place == "output", False
                    )
                frame["parameters"] = [
                    sum(
                        p.numel()
                        for p in build_model(
                            row.architecture,
                            numeric_width=len(model["numeric_columns"]),
                            visual_width=len(model["visual_columns"]),
                            seed=model["seed"],
                            hidden_width=int(row.hidden_width),
                            mlp_layers=int(row.mlp_layers),
                            recurrent_layers=int(row.recurrent_layers),
                        ).parameters()
                    )
                    for row in frame.itertuples()
                ]
                trials.append(frame)
            for history_path in path.parent.glob("inner_*_fold*_seed*.csv"):
                frame = pd.read_csv(history_path)
                meta = {
                    **fields,
                    "scope": "inner_training_curve",
                    "history_source": str(history_path),
                }
                if "_trial" in history_path.stem and trial_files:
                    trial = int(history_path.stem.split("_trial")[1].split("_")[0])
                    parameters = pd.read_csv(trial_files[0]).set_index("number").loc[trial]
                    place = parameters.get("params_dropout_place", "none")
                    rate = 0.0 if place == "none" else float(parameters.params_dropout_rate)
                    meta.update(
                        trial=trial,
                        learning_rate=parameters.params_learning_rate,
                        dropout_place=place,
                        visual_dropout=rate if place == "visual" else 0.0,
                        output_dropout=rate if place == "output" else 0.0,
                        selected_epoch=parameters.get("user_attrs_selected_epoch"),
                        trial_dropout_rate=rate,
                        trial_time_shared=parameters.get("params_time_shared"),
                        visual_time_shared=parameters.get("params_time_shared", True)
                        if place == "visual"
                        else True,
                        output_time_shared=parameters.get("params_time_shared", False)
                        if place == "output"
                        else False,
                    )
                    if "params_mlp_layers" in parameters or "params_recurrent_layers" in parameters:
                        meta.update(visual_time_shared=True, output_time_shared=True)
                    architecture = parameters.get("params_architecture", model["architecture"])
                    width = parameters.get("params_hidden_width", model.get("hidden_width", 32))
                    trial_model = build_model(
                        architecture,
                        numeric_width=len(model["numeric_columns"]),
                        visual_width=len(model["visual_columns"]),
                        seed=model["seed"],
                        hidden_width=int(width),
                        mlp_layers=int(parameters.get("params_mlp_layers", 2)),
                        recurrent_layers=int(
                            parameters.get(
                                "params_recurrent_layers", model.get("recurrent_layers", 1)
                            )
                        ),
                    )
                    meta.update(
                        architecture=architecture,
                        hidden_width=width,
                        mlp_layers=int(parameters.get("params_mlp_layers", 2)),
                        recurrent_layers=int(
                            parameters.get(
                                "params_recurrent_layers", model.get("recurrent_layers", 1)
                            )
                        ),
                        weight_decay=parameters.get(
                            "params_weight_decay", model.get("weight_decay", 1e-4)
                        ),
                        visual_dropout=parameters.get(
                            "params_visual_dropout", meta["visual_dropout"]
                        ),
                        output_dropout=parameters.get(
                            "params_output_dropout", meta["output_dropout"]
                        ),
                        parameters=sum(p.numel() for p in trial_model.parameters()),
                        trial_state=parameters.get("state"),
                    )
                curves.append(frame.assign(**meta))
            history = model.get("refit_losses", model["losses"])
            curves.append(
                history.assign(**fields, scope="refit_training_curve", history_source=str(path))
            )
        grid_path = settings_path.parent / (
            "shared_lstm_grid.csv"
            if (settings_path.parent / "shared_lstm_grid.csv").exists()
            else "shared_mlp_grid.csv"
        )
        if grid_path.exists():
            for candidate in pd.read_csv(grid_path).to_dict("records"):
                meta = {
                    **(
                        fields
                        if tests
                        else dict(
                            experiment=run.name, source_run=str(run), seed=settings["seeds"][0]
                        )
                    ),
                    **candidate,
                    "checkpoint": "",
                    "outer_experiment": candidate["development_outer_experiment"],
                    "selected_epoch": candidate["epoch"],
                    "scope": "shared_hyperparameter_trial",
                }
                trials.append(pd.DataFrame([meta]))
                for fold in range(2):
                    key = candidate.get("candidate_key")
                    filename = (
                        f"{key}_fold{fold}.csv"
                        if isinstance(key, str)
                        else f"w{int(candidate['hidden_width'])}_d{int(candidate['mlp_layers'])}_lr{candidate['learning_rate']:g}_fold{fold}.csv"
                    )
                    path = settings_path.parent / "shared_grid" / filename
                    history_meta = {
                        **(
                            fields
                            if tests
                            else dict(
                                experiment=run.name, source_run=str(run), seed=settings["seeds"][0]
                            )
                        ),
                        **{
                            key: meta[key]
                            for key in (
                                "hidden_width",
                                "mlp_layers",
                                "learning_rate",
                                "parameters",
                                "outer_experiment",
                                "selected_epoch",
                            )
                        },
                        **{
                            key: candidate[key]
                            for key in (
                                "mlp_widths",
                                "architecture",
                                "weight_decay",
                                "visual_dropout",
                                "output_dropout",
                                "search_stage",
                                "recurrent_layers",
                            )
                            if key in candidate
                        },
                        "scope": "shared_inner_training_curve",
                        "inner_fold": fold,
                        "history_source": str(path),
                    }
                    curves.append(pd.read_csv(path).assign(**history_meta))
        if tests:
            metrics = pd.concat(tests, ignore_index=True)
            for (version, seed), g in metrics.groupby(["version", "seed"]):
                folds = pd.DataFrame(
                    [
                        row
                        for row in summaries
                        if row["experiment"] == run.name
                        and row["version"] == version
                        and row["seed"] == seed
                        and row["scope"] == "outer_fold"
                    ]
                )
                shared = {
                    key: folds[key].iloc[0]
                    for key in [
                        "architecture",
                        "hidden_width",
                        "mlp_layers",
                        "mlp_widths",
                        "weight_decay",
                        "recurrent_layers",
                        "reset_history",
                        "parameters",
                        "loss",
                        "cdf_score",
                        "cdf_regret_weight",
                        "validation_cdf_score",
                        "learning_rate",
                        "visual_dropout",
                        "output_dropout",
                        "inputs",
                        "numeric_dimensions",
                        "visual_dimensions",
                        "visual_pooling",
                        "visual_time_shared",
                        "output_time_shared",
                        "trigger_rule",
                        "validation_utility",
                        "maximum_epochs",
                        "selected_epoch",
                        "cohort_rule",
                    ]
                    if folds[key].nunique(dropna=False) == 1
                }
                summaries.append(
                    dict(
                        experiment=run.name,
                        source_run=str(run),
                        version=version,
                        seed=seed,
                        scope="outer_summary",
                        completed_outer_folds=len(paths),
                        expected_outer_folds=len(settings.get("folds", {})),
                        evaluation_scope=evaluation_scope,
                        cohort_experiments=len(cohort_experiments),
                        **shared,
                        **summarize_metrics(g),
                    )
                )
    pd.DataFrame(summaries).to_csv(output / "ablation_summary.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(
            output / "ablation_learning_curves.csv", index=False
        )
    if trials:
        pd.concat(trials, ignore_index=True).to_csv(output / "ablation_trials.csv", index=False)
    print(
        f"[ablations] {len(summaries)} score rows; {sum(len(g) for g in curves)} epoch rows; {output}",
        flush=True,
    )


def summarize(args, versions=None, contrasts=None, comparison_run=None):
    metrics, fits = [], []
    paths = checkpoint_files(args.output)
    if comparison_run is not None:
        paths += checkpoint_files(comparison_run)
    for path in sorted(paths):
        with path.open("rb") as stream:
            result = pickle.load(stream)
        m, c = result["cycle_metrics"], result["checkpoint"]
        metrics.append(m.loc[m.split.eq("test")])
        fits.append(
            dict(
                version=m.version.iloc[0],
                seed=int(m.seed.iloc[0]),
                outer_experiment=m.outer_experiment.iloc[0],
                validation_experiments=(
                    json.dumps(c["inner_validation_experiments"])
                    if "inner_validation_experiments" in c
                    else "|".join(sorted(m.loc[m.split.eq("validation"), "experiment_id"].unique()))
                ),
                selected_epoch=c["selected_epoch"],
                max_validation_evaluable=(
                    int(c["losses"].validation_evaluable.max())
                    if "validation_evaluable" in c["losses"]
                    else np.nan
                ),
                selected_training_loss=c["losses"]
                .loc[c["losses"].epoch.eq(c["selected_epoch"]), "training_loss"]
                .item(),
            )
        )
    table = pd.concat(metrics, ignore_index=True)
    table.to_csv(args.output / "cycle_metrics.csv", index=False)
    from output_layout import experiment_root

    owner = experiment_root(args.output)
    evaluation = (
        owner / "evaluation" / "policy"
        if owner.parent.name in {"01_视觉模型训练", "training_checkpoint"}
        else args.output
    )
    evaluation.mkdir(parents=True, exist_ok=True)
    write_evaluation_metrics(
        table, args.dataset, evaluation, camera_role=getattr(args, "camera_role", "front")
    )
    pd.DataFrame(fits).to_csv(args.output / "fit_summary.csv", index=False)
    records = []
    for (version, seed), g in table.groupby(["version", "seed"]):
        record = dict(
            version=version,
            seed=seed,
            **summarize_metrics(g),
            median_regret_pct=g.regret_pct.median(),
            p90_regret_pct=g.regret_pct.quantile(0.9),
            evaluable_coverage=g.evaluable.mean(),
            no_input_count=g.trigger_status.eq("no_input").sum(),
            no_trigger_count=g.trigger_status.eq("no_trigger").sum(),
            unsupported_count=g.unsupported_trigger.sum(),
            early_count=g.trigger_status.eq("early").sum(),
            hit_count=g.trigger_status.eq("hit").sum(),
            mean_set_probability=g.probability_first_in_set.mean(),
        )
        records.append(record)
    summary = pd.DataFrame(records)
    summary.to_csv(args.output / "summary.csv", index=False)
    versions = versions or table.version.unique().tolist()
    compensation_comparisons(
        table, args.output, contrasts or [(versions[0], v) for v in versions[1:]], versions
    )
    if args.plot:
        from plots.model_comparison import plot_temporal_stopping

        plot_temporal_stopping(summary, args.output)
    print(summary.to_string(index=False), flush=True)
