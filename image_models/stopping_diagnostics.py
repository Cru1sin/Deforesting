"""Frozen stopping-policy evaluation and mechanism diagnostics."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from image_models.checkpoints import checkpoint_files
from output_layout import PROJECT


def rescore_decisions(run, output, dataset):
    """Change the energy reference, keeping every saved stopping time fixed.

    Only common admitted cycles are comparable; never interpolate unsupported COP.
    This is a frozen-decision comparison, not a forward replay on a new action grid.
    """
    from dataset_tools import DatasetLoader
    from image_models.comparisons import write_evaluation_metrics
    from image_models.experiments import record_experiment
    from image_models.policy import finalize_policy_metrics

    rows, reference = DatasetLoader(dataset).load_training_data()
    old = pd.read_csv(run / "cycle_metrics.csv", parse_dates=["selected_time"])
    old = old.loc[old.cycle_name.isin(rows.cycle_name)].copy()
    keep = [
        "cycle_name",
        "experiment_id",
        "selected_time",
        "forced_final",
        "no_trigger",
        "strategy",
        "version",
        "seed",
        "split",
        "outer_experiment",
    ]
    metrics = old[[c for c in keep if c in old]].copy()
    rewards = reference[["cycle_name", "candidate_defrost_time", "cycle_cop", "cycle_cop_eligible"]]
    metrics = metrics.merge(
        rewards,
        left_on=["cycle_name", "selected_time"],
        right_on=["cycle_name", "candidate_defrost_time"],
        how="left",
        validate="many_to_one",
    )
    metrics["selected_cop"] = metrics.cycle_cop.where(metrics.cycle_cop_eligible.eq(True))
    metrics = metrics.drop(columns=["candidate_defrost_time", "cycle_cop", "cycle_cop_eligible"])
    peaks = rows[["cycle_name", "optimal_cop", "optimal_time"]].drop_duplicates()
    metrics = metrics.merge(peaks, on="cycle_name", validate="many_to_one")
    metrics = finalize_policy_metrics(metrics, reference)
    metrics["regret_pct"] = 100 * (1 - metrics.selected_cop / metrics.optimal_cop)
    metrics["near_optimal_hit"] = metrics.selected_cop.ge(0.99 * metrics.optimal_cop).where(
        metrics.selected_cop.notna()
    )
    metrics["evaluable"] = metrics.selected_cop.notna()
    metrics["unsupported_trigger"] = metrics.selected_time.notna() & ~metrics.evaluable
    metrics["early_trigger"] = metrics.selected_time.lt(metrics.optimal_time)
    record_experiment(
        output,
        dict(
            source_run=str(run),
            dataset=str(dataset),
            data_definition=DatasetLoader(dataset).training_settings,
        ),
        "Frozen saved stopping times, rescored on current published teacher; common admitted cycles only.",
    )
    metrics.to_csv(output / "cycle_metrics.csv", index=False)
    write_evaluation_metrics(metrics, dataset, output)


def diagnose_cycles(run, output, numbers):
    """Frozen-head replay and input sensitivity, restricted to non-baseline cycles.

    Probes change model inputs only; they are not simulated physical interventions.
    """
    import torch

    from dataset_tools import DatasetLoader
    from image_models.control_context import load_control_context
    from image_models.cycle_selection import screen_cv_cycles
    from image_models.inputs import RGB
    from image_models.policy import predict_policy
    from image_models.training_cases import FREQUENCY

    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    settings = json.loads((run / "settings.json").read_text())
    admission = settings["admission"]
    loader = DatasetLoader(admission["dataset"])
    screening = screen_cv_cycles(
        pd.read_csv(admission["decisions"], low_memory=False),
        loader.list_cycles(),
        quality=pd.read_csv(admission["quality"]),
        exclude_rb=admission.get("exclude_rb", False),
        minimum_terminal_decline=admission.get("minimum_terminal_decline"),
        rb_minimum_terminal_decline=admission.get("rb_minimum_terminal_decline"),
        input_rule=admission.get("input_rule", "historical"),
        cohort_rule=admission.get("cohort_rule", "reward_complete"),
        action_inputs=Path(admission["quality"]).parent / "decision_inputs.parquet",
    )
    screening.to_csv(output / "cycle_screening.csv", index=False)
    names = [f"frost_cycle_{n:06d}" for n in numbers]
    roles = screening.loc[
        screening.cycle_name.isin(names), ["cycle_name", "baseline_cycle", "selected"]
    ]
    roles.to_csv(output / "requested_cycle_roles.csv", index=False)
    metrics = pd.read_csv(run / "cycle_metrics.csv")
    if "fallback_terminal" in metrics:
        metrics.loc[metrics.fallback_terminal].to_csv(output / "fallback_cycles.csv", index=False)
    names = roles.loc[~roles.baseline_cycle, "cycle_name"].tolist()
    if not names:
        print("Requested cycles are baseline experiments; no trajectory or model audit performed.")
        return
    inputs = pd.read_parquet(Path(admission["quality"]).parent / "decision_inputs.parquet")
    inputs = inputs.loc[inputs.cycle_name.isin(names)].copy()
    temperature_path = run / "temperature_inputs.parquet"
    if temperature_path.exists():
        temperatures = pd.read_parquet(temperature_path)
        inputs = inputs.merge(
            temperatures[
                ["cycle_name", "candidate_defrost_time", "control_setpoint_c", "control_outlet_c"]
            ],
            on=["cycle_name", "candidate_defrost_time"],
            validate="one_to_one",
        )
    reference = pd.read_parquet(run / "reference.parquet")
    conditions = (
        reference.groupby(["cycle_name", "experiment_id"])
        .water_temperature_setpoint.median()
        .reset_index()
    )
    conditions = conditions.loc[
        conditions.cycle_name.isin(screening.loc[screening.selected, "cycle_name"])
    ]
    trajectories = []
    probes = []
    events = []
    segments = []
    support = []
    for name, g in inputs.groupby("cycle_name"):
        g = g.sort_values("candidate_defrost_time")
        controls = load_control_context(
            admission["dataset"],
            name,
            g[["cycle_name", "candidate_defrost_time", "stable_heating_start"]],
        )
        g = g.merge(
            controls[["candidate_defrost_time", "frequency_command_hz", "actual_frequency_hz"]],
            on="candidate_defrost_time",
            validate="one_to_one",
        )
        heat = g.pre_defrost_heat_kwh
        energy = g.pre_defrost_electricity_kwh
        cost = g.defrost_event_electricity_kwh
        g["heat_10min_kw"] = heat.diff(20) * 6
        g["power_10min_kw"] = energy.diff(20) * 6
        denominator = energy.diff(20) + cost.diff(20)
        g["marginal_cop_10min"] = heat.diff(20) / denominator.where(denominator.gt(0))
        outer = g.experiment_id.iloc[0]
        nested = isinstance(settings["folds"][outer][0], list)
        partition = conditions.assign(
            split=np.where(
                conditions.experiment_id.eq(outer),
                "test",
                np.where(
                    conditions.experiment_id.isin([] if nested else settings["folds"][outer]),
                    "validation",
                    "train",
                ),
            )
        )
        for (split, setpoint), part in partition.groupby(["split", "water_temperature_setpoint"]):
            support.append(
                dict(
                    audited_cycle=name,
                    outer_experiment=outer,
                    split=split,
                    water_temperature_setpoint=setpoint,
                    cycles=len(part),
                    experiments=part.experiment_id.nunique(),
                )
            )
        for path in checkpoint_files(run, f"*_{outer}_seed*.pkl"):
            with path.open("rb") as stream:
                result = pickle.load(stream)
            seed = result["checkpoint"]["seed"]
            version = result["cycle_metrics"].version.iloc[0]
            saved = result["predictions"].loc[lambda p: p.cycle_name.eq(name) & p.split.eq("test")]
            metric = result["cycle_metrics"].loc[lambda p: p.cycle_name.eq(name)].iloc[0]
            grid = g.loc[g.candidate_defrost_time.isin(saved.candidate_defrost_time)]
            actual = grid.loc[grid.input_valid].copy()
            replay = predict_policy(actual, result["checkpoint"])
            frozen = saved.set_index("candidate_defrost_time").reindex(
                replay.candidate_defrost_time
            )
            np.testing.assert_allclose(replay.probability, frozen.probability, atol=1e-6, rtol=1e-5)
            trace = grid.drop(columns=["near_optimal"], errors="ignore").merge(
                saved[["candidate_defrost_time", "probability", "decision_margin", "near_optimal"]],
                on="candidate_defrost_time",
                validate="one_to_one",
            )
            fields = [
                "cycle_name",
                "experiment_id",
                "candidate_defrost_time",
                "image_time",
                "input_valid",
                "cycle_cop",
                "probability",
                "decision_margin",
                "near_optimal",
                *FREQUENCY,
                "frequency_command_hz",
                "actual_frequency_hz",
                "ambient_temperature",
                "water_temperature_setpoint",
                "pre_defrost_heat_kwh",
                "pre_defrost_electricity_kwh",
                "defrost_event_electricity_kwh",
                "heat_10min_kw",
                "power_10min_kw",
                "marginal_cop_10min",
            ]
            fields += [c for c in ["control_setpoint_c", "control_outlet_c"] if c in trace]
            trajectories.append(trace[fields].assign(seed=seed, version=version))
            nearest_time = actual.iloc[
                (actual.candidate_defrost_time - pd.Timestamp(metric.optimal_time)).abs().argmin()
            ].candidate_defrost_time
            for label, time in [
                ("first_trigger", metric.selected_time),
                ("cop_optimum", metric.optimal_time),
                ("nearest_observable_optimum", nearest_time),
            ]:
                row = trace.loc[trace.candidate_defrost_time.eq(time)]
                if len(row):
                    events.append(
                        row[fields].assign(
                            seed=seed, version=version, event=label, regret_pct=metric.regret_pct
                        )
                    )
            # Explain the actual COP change between the earlier and later decision.
            full = reference.loc[reference.cycle_name.eq(name)].set_index("candidate_defrost_time")
            start, end = sorted(
                [pd.Timestamp(metric.selected_time), pd.Timestamp(metric.optimal_time)]
            )
            a, b = full.loc[start], full.loc[end]
            duration = (end - start).total_seconds() / 3600
            fixed_cost = b.pre_defrost_heat_kwh / (
                b.pre_defrost_electricity_kwh + a.defrost_event_electricity_kwh
            )
            segments.append(
                dict(
                    cycle_name=name,
                    seed=seed,
                    version=version,
                    start=start,
                    end=end,
                    minutes=duration * 60,
                    cop_start=a.cycle_cop,
                    cop_end=b.cycle_cop,
                    segment_heat_kw=(b.pre_defrost_heat_kwh - a.pre_defrost_heat_kwh) / duration
                    if duration
                    else np.nan,
                    segment_power_kw=(b.pre_defrost_electricity_kwh - a.pre_defrost_electricity_kwh)
                    / duration
                    if duration
                    else np.nan,
                    event_cost_start=a.defrost_event_electricity_kwh,
                    event_cost_end=b.defrost_event_electricity_kwh,
                    heat_electricity_cop_change=fixed_cost - a.cycle_cop,
                    ridge_cost_cop_change=b.cycle_cop - fixed_cost,
                )
            )
            temperature_columns = [
                c for c in result["checkpoint"]["numeric_columns"] if c.startswith("control_")
            ]
            probe_names = [
                "actual",
                "zero_frequency_changes",
                "hold_frequency_headroom",
                "hold_frequency_context",
                "hold_visual_state",
                "headroom_minus_5pp",
                "headroom_plus_5pp",
                "zero_headroom",
            ]
            probe_names += [
                f"{c}_{direction}"
                for c in temperature_columns
                for direction in ["minus_1c", "plus_1c"]
            ]
            for probe in probe_names:
                query = actual.copy()
                if probe == "zero_frequency_changes":
                    query[FREQUENCY[1:]] = 0.0
                elif probe == "hold_frequency_headroom":
                    query[FREQUENCY[0]] = actual[FREQUENCY[0]].iloc[0]
                elif probe == "hold_frequency_context":
                    query[FREQUENCY] = np.broadcast_to(
                        actual[FREQUENCY].iloc[0].to_numpy(), (len(actual), len(FREQUENCY))
                    )
                elif probe == "hold_visual_state":
                    query[RGB] = np.broadcast_to(
                        actual[RGB].iloc[0].to_numpy(), (len(actual), len(RGB))
                    )
                elif probe in ["headroom_minus_5pp", "headroom_plus_5pp"]:
                    query[FREQUENCY[0]] = (
                        actual[FREQUENCY[0]] + (-0.05 if "minus" in probe else 0.05)
                    ).clip(lower=0)
                elif probe == "zero_headroom":
                    query[FREQUENCY[0]] = 0.0
                elif probe.startswith("control_"):
                    column, direction = probe.rsplit("_", 2)[0], probe.rsplit("_", 2)[1]
                    query[column] = actual[column] + (-1 if direction == "minus" else 1)
                predicted = (
                    replay if probe == "actual" else predict_policy(query, result["checkpoint"])
                )
                positive = predicted.loc[predicted.probability.ge(0.5)]
                chosen = (
                    positive.candidate_defrost_time.iloc[0]
                    if len(positive)
                    else saved.terminal_time.iloc[0]
                )
                nearest = predicted.iloc[
                    (predicted.candidate_defrost_time - pd.Timestamp(metric.optimal_time))
                    .abs()
                    .argmin()
                ]
                probes.append(
                    dict(
                        cycle_name=name,
                        seed=seed,
                        version=version,
                        probe=probe,
                        selected_time=chosen,
                        shift_from_actual_min=(
                            chosen - pd.Timestamp(metric.selected_time)
                        ).total_seconds()
                        / 60,
                        autonomous=bool(len(positive)),
                        max_hazard=predicted.probability.max(),
                        probability_nearest_optimum=nearest.probability,
                    )
                )
    pd.concat(trajectories, ignore_index=True).to_csv(
        output / "cycle_trajectories.csv", index=False
    )
    pd.concat(events, ignore_index=True).to_csv(output / "decision_events.csv", index=False)
    pd.DataFrame(segments).to_csv(output / "cop_change_decomposition.csv", index=False)
    pd.DataFrame(probes).to_csv(output / "input_sensitivity.csv", index=False)
    pd.DataFrame(support).to_csv(output / "fold_condition_support.csv", index=False)
    print(pd.DataFrame(probes).to_string(index=False))


def admission_diagnostics(run, output):
    """Paired COP loss budget; input support is distinct from learned policy error."""
    from image_models.policy import summarize_metrics

    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(run / "cycle_metrics.csv")
    predictions = []
    for path in checkpoint_files(run):
        with path.open("rb") as stream:
            result = pickle.load(stream)
        p = result["predictions"]
        predictions.append(p.loc[p.split.eq("test")])
    p = pd.concat(predictions).drop_duplicates(["cycle_name", "candidate_defrost_time"])
    available = p.loc[p.input_valid & p.cycle_cop_eligible].groupby("cycle_name").cycle_cop.max()
    metrics["observable_oracle_cop"] = np.fmax(
        metrics.cycle_name.map(available), metrics.terminal_cop
    )
    metrics["input_support_loss"] = metrics.optimal_cop - metrics.observable_oracle_cop
    metrics["policy_loss"] = metrics.observable_oracle_cop - metrics.selected_cop
    metrics["remaining_loss"] = metrics.optimal_cop - metrics.selected_cop
    np.testing.assert_allclose(
        metrics.input_support_loss + metrics.policy_loss, metrics.remaining_loss, atol=1e-12
    )
    assert metrics.input_support_loss.ge(-1e-10).all() and metrics.policy_loss.ge(-1e-10).all()
    metrics.to_csv(output / "cycle_loss_budget.csv", index=False)
    records = []
    for (version, seed), g in metrics.groupby(["version", "seed"]):
        paired = g.dropna(subset=["selected_cop", "rb_cop", "optimal_cop", "observable_oracle_cop"])
        base = summarize_metrics(paired)
        oracle = summarize_metrics(paired.assign(selected_cop=paired.observable_oracle_cop))
        records.append(
            dict(
                version=version,
                seed=seed,
                **base,
                observable_oracle_H_RB_pct=oracle["headroom_captured_pct"],
                mean_observable_oracle_cop=paired.observable_oracle_cop.mean(),
                mean_input_support_loss=paired.input_support_loss.mean(),
                mean_policy_loss=paired.policy_loss.mean(),
                mean_remaining_loss=paired.remaining_loss.mean(),
                input_share_remaining_pct=100
                * paired.input_support_loss.sum()
                / paired.remaining_loss.sum(),
                policy_share_remaining_pct=100
                * paired.policy_loss.sum()
                / paired.remaining_loss.sum(),
                fallback_count=int(g.fallback_terminal.sum()),
                no_input_count=int(g.input_missing.sum()),
                evaluable_count=int(g.evaluable.sum()),
            )
        )
    summary = pd.DataFrame(records)
    summary.to_csv(output / "loss_budget_summary.csv", index=False)
    reference_path = run / "reference.parquet"
    if not reference_path.exists():
        settings = json.loads((run / "settings.json").read_text())
        reference_path = Path(settings["dataset"]) / "training_data/reference.parquet"
    reference = pd.read_parquet(reference_path)
    conditions = reference.groupby("cycle_name").water_temperature_setpoint.median()
    conditioned = metrics.assign(setpoint_c=metrics.cycle_name.map(conditions))
    pd.DataFrame(
        [
            dict(
                version=version,
                seed=seed,
                setpoint_c=setpoint,
                **summarize_metrics(g),
                fallback_count=int(g.fallback_terminal.sum()),
                evaluable_count=int(g.evaluable.sum()),
                median_regret_pct=g.regret_pct.median(),
                p90_regret_pct=g.regret_pct.quantile(0.9),
            )
            for (version, seed, setpoint), g in conditioned.groupby(
                ["version", "seed", "setpoint_c"]
            )
        ]
    ).to_csv(output / "condition_summary.csv", index=False)
    print(summary.to_string(index=False))


def audit_cdf_target(run, output):
    """Perfect ramp fit at CDF .5; future COP labels are diagnostic only."""
    from dataset_tools import DatasetLoader
    from image_models.comparisons import write_evaluation_metrics
    from image_models.cycle_selection import screen_published_cycles
    from image_models.labels import stopping_cdf_targets
    from image_models.policy import policy_cycle_metrics

    settings = json.loads((run / "settings.json").read_text())
    loader = DatasetLoader(settings["dataset"])
    if loader.training_settings != settings["data_definition"]:
        raise ValueError("Published data differs from the frozen experiment")
    screening = screen_published_cycles(settings["dataset"])
    rows, reference = loader.load_training_data(objective="cdf_ramp")
    rows = stopping_cdf_targets(
        rows.loc[rows.cycle_name.isin(screening.loc[screening.selected, "cycle_name"])]
    )
    rows["probability"] = rows.stopping_cdf_target.where(rows.input_valid)
    metrics = policy_cycle_metrics(
        rows,
        reference,
        0.5,
        loss_name="cdf_ramp",
        strategy="first_positive_terminal",
        controller_reference=reference,
    )
    metrics = metrics.assign(version="CDF_target_oracle", seed=0, split="diagnostic")
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "cycle_metrics.csv", index=False)
    write_evaluation_metrics(metrics, settings["dataset"], output)


def audit_visual_support(run, output):
    """Descriptive early-cycle embedding support; no policy or teacher changes."""
    from dataset_tools import DatasetLoader
    from image_models.cycle_selection import screen_published_cycles
    from image_models.inputs import RGB

    settings = json.loads((run / "settings.json").read_text())
    loader = DatasetLoader(settings["dataset"])
    if loader.training_settings != settings["data_definition"]:
        raise ValueError("Published data differs from the frozen experiment")
    screening = screen_published_cycles(settings["dataset"])
    rows, _ = loader.load_training_data(objective=settings["loss_name"])
    rows = rows.loc[
        rows.cycle_name.isin(screening.loc[screening.selected, "cycle_name"]) & rows.input_valid
    ].copy()
    rows["visual_norm"] = np.linalg.norm(rows[RGB].to_numpy(float), axis=1)
    summaries = []
    centroids = []
    near_centroids = []
    for name, g in rows.groupby("cycle_name", sort=True):
        g = g.sort_values("candidate_defrost_time", kind="stable")
        visual = g[RGB].to_numpy(float)
        visual = visual / np.linalg.norm(visual, axis=1, keepdims=True)
        jumps = 1 - (visual[1:] * visual[:-1]).sum(axis=1)
        start = g.candidate_defrost_time.min()
        early = g.loc[g.candidate_defrost_time.le(start + pd.Timedelta(minutes=5))]
        near = g.loc[g.near_optimal]
        summaries.append(
            dict(
                cycle_name=name,
                experiment_id=g.experiment_id.iloc[0],
                near_actions=len(near),
                setpoint_c=g.control_setpoint_c.median(),
                early_actions=len(early),
                max_adjacent_cosine_jump=jumps.max() if len(jumps) else np.nan,
                p90_adjacent_cosine_jump=np.quantile(jumps, 0.9) if len(jumps) else np.nan,
                early_norm=early.visual_norm.median(),
                all_norm=g.visual_norm.median(),
                norm_p10=g.visual_norm.quantile(0.1),
                norm_p90=g.visual_norm.quantile(0.9),
            )
        )
        vector = early[RGB].to_numpy(float).mean(axis=0)
        centroids.append(vector / np.linalg.norm(vector))
        vector = near[RGB].to_numpy(float).mean(axis=0) if len(near) else np.full(len(RGB), np.nan)
        near_centroids.append(vector / np.linalg.norm(vector))
        for column in [
            "frequency_headroom_fraction",
            "frequency_change_60s_hz",
            "frequency_change_300s_hz",
            "control_outlet_c",
        ]:
            summaries[-1]["near_" + column] = near[column].median()
    table = pd.DataFrame(summaries)
    vectors = np.asarray(centroids)
    near_vectors = np.asarray(near_centroids)
    for i, row in table.iterrows():
        train = table.experiment_id.ne(row.experiment_id) & table.setpoint_c.sub(
            row.setpoint_c
        ).abs().le(1)
        ids = np.flatnonzero(train)
        similarities = vectors[ids] @ vectors[i]
        nearest = ids[np.argmax(similarities)]
        table.loc[i, "nearest_training_cycle"] = table.loc[nearest, "cycle_name"]
        table.loc[i, "early_cosine_distance"] = 1 - similarities.max()
        table.loc[i, "early_norm_ratio_to_train"] = (
            row.early_norm / table.loc[train, "early_norm"].median()
        )
        table.loc[i, "all_norm_ratio_to_train"] = (
            row.all_norm / table.loc[train, "all_norm"].median()
        )
        ids = np.flatnonzero(train & table.near_actions.gt(0))
        if row.near_actions and len(ids):
            similarities = near_vectors[ids] @ near_vectors[i]
            nearest = ids[np.argmax(similarities)]
            table.loc[i, "nearest_near_training_cycle"] = table.loc[nearest, "cycle_name"]
            table.loc[i, "near_cosine_distance"] = 1 - similarities.max()
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "cycle_visual_support.csv", index=False)
    table.groupby("setpoint_c")[
        ["early_cosine_distance", "early_norm_ratio_to_train", "all_norm_ratio_to_train"]
    ].agg(["median", "min", "max"]).to_csv(output / "condition_visual_support.csv")
    print(
        table.sort_values("early_cosine_distance", ascending=False).head(10).to_string(index=False)
    )


def audit_cdf_fit(run, output, numbers):
    """Compare in-sample fit and held-out transfer without changing a saved model."""
    import torch

    from dataset_tools import DatasetLoader
    from image_models.cycle_selection import screen_published_cycles
    from image_models.labels import stopping_cdf_targets
    from image_models.policy import _batch_loss, apply_trigger_rule, predict_policy

    torch.set_num_threads(1)
    settings = json.loads((run / "settings.json").read_text())
    if (
        settings["loss_name"] != "cdf_ramp"
        or settings.get("cdf_regret_weight", 0)
        or settings.get("visual_history")
    ):
        raise ValueError("This fit diagnostic compares the original CDF/current-visual baseline")
    loader = DatasetLoader(settings["dataset"])
    if loader.training_settings != settings["data_definition"]:
        raise ValueError("Published data differs from the frozen experiment")
    screening = screen_published_cycles(settings["dataset"])
    rows, _ = loader.load_training_data(objective="cdf_ramp")
    if settings.get("compensation_progress"):
        from image_models.inputs import causal_compensation_progress

        rows = causal_compensation_progress(rows)
    if settings.get("temperature_representation") == "tracking_error":
        rows["temperature_error_c"] = rows.control_setpoint_c - rows.control_outlet_c
    if settings.get("elapsed_context"):
        from image_models.inputs import elapsed_cycle_minutes

        rows["elapsed_cycle_minutes"] = elapsed_cycle_minutes(rows)
    names = [f"frost_cycle_{number:06d}" for number in numbers]
    rows = stopping_cdf_targets(
        rows.loc[
            rows.cycle_name.isin(names)
            & rows.cycle_name.isin(screening.loc[screening.selected, "cycle_name"])
        ]
    )
    records = []
    trajectories = []
    for path in checkpoint_files(run):
        saved = pickle.load(path.open("rb"))
        model = saved["checkpoint"]
        test = set(saved["cycle_metrics"].cycle_name)
        training = set(model["training_cycles"])
        valid = rows.loc[rows.input_valid & rows.cycle_name.isin(test | training)]
        predicted = predict_policy(valid, model)
        cumulative = apply_trigger_rule(predicted, "cdf")
        for name, g in predicted.groupby("cycle_name"):
            scores = cumulative.loc[cumulative.cycle_name.eq(name)]
            if name in test:
                fields = [
                    "cycle_name",
                    "candidate_defrost_time",
                    "cycle_start",
                    "cycle_cop",
                    "optimal_cop",
                    "near_optimal",
                    "stopping_cdf_target",
                    "probability",
                    "decision_margin",
                    *model["numeric_columns"],
                ]
                trajectory = scores.reindex(columns=fields).rename(
                    columns={"probability": "cdf_probability"}
                )
                trajectory["hazard_probability"] = trajectory.candidate_defrost_time.map(
                    g.set_index("candidate_defrost_time").probability
                )
                trajectories.append(trajectory.assign(checkpoint=path.stem))
            last = rows.loc[
                rows.cycle_name.eq(name) & rows.near_optimal, "candidate_defrost_time"
            ].max()
            first = rows.loc[
                rows.cycle_name.eq(name) & rows.near_optimal, "candidate_defrost_time"
            ].min()
            prefix = scores.loc[scores.candidate_defrost_time.le(last), "probability"]
            triggered = scores.loc[scores.probability.ge(0.5)]
            chosen = triggered.iloc[0] if len(triggered) else None
            records.append(
                dict(
                    checkpoint=path.stem,
                    cycle_name=name,
                    phase="heldout" if name in test else "training",
                    epoch=model["selected_epoch"],
                    cdf_score="bce",
                    cdf_score_loss=float(
                        _batch_loss(torch.as_tensor(g.decision_margin.to_numpy()), g, "cdf_ramp")
                    ),
                    cdf_at_near_end=prefix.iloc[-1] if len(prefix) else np.nan,
                    first_crossing_time=chosen.candidate_defrost_time
                    if chosen is not None
                    else pd.NaT,
                    trigger_status=(
                        "no_crossing"
                        if chosen is None
                        else "hit"
                        if chosen.near_optimal
                        else "early"
                        if chosen.candidate_defrost_time < first
                        else "late"
                        if chosen.candidate_defrost_time > last
                        else "gap"
                    ),
                    first_crossing_regret_pct=100 * (1 - chosen.cycle_cop / chosen.optimal_cop)
                    if chosen is not None
                    else np.nan,
                    no_crossing=chosen is None,
                )
            )
    output.mkdir(parents=True, exist_ok=True)
    screening.to_csv(output / "cycle_screening.csv", index=False)
    pd.DataFrame(records).to_csv(output / "cdf_fit_cycles.csv", index=False)
    if trajectories:
        pd.concat(trajectories, ignore_index=True).to_csv(
            output / "heldout_cdf_trajectories.csv", index=False
        )


def audit_trigger_rules(run, output):
    """Frozen-checkpoint diagnostics; CDF crossing is a probe, never a refit or selector."""
    from image_models.policy import replay_cycles

    settings = json.loads((run / "settings.json").read_text())
    reference = pd.read_parquet(Path(settings["dataset"]) / "training_data/reference.parquet")
    records = []
    paths = checkpoint_files(run)
    for path in sorted(paths):
        saved = pickle.load(path.open("rb"))
        checkpoint = saved["checkpoint"]
        history = checkpoint["losses"]
        predicted = saved["predictions"]
        for name, g in predicted.loc[predicted.split.eq("test")].groupby("cycle_name"):
            g = g.sort_values("candidate_defrost_time").copy()
            metric = saved["cycle_metrics"].loc[saved["cycle_metrics"].cycle_name.eq(name)].iloc[0]
            hazard = g.probability.where(g.input_valid, 0.0).fillna(0.0)
            cumulative = 1.0 - (1.0 - hazard).cumprod()
            assert cumulative.diff().dropna().ge(-1e-12).all()
            assert (cumulative + 1e-12).ge(hazard).all()
            probe = g.assign(probability=cumulative.where(g.input_valid))
            replay = replay_cycles(
                probe, 0.5, strategy="first_positive_terminal", reference=reference
            ).iloc[0]
            near = g.loc[g.near_optimal]
            first = near.candidate_defrost_time.min()
            last = near.candidate_defrost_time.max()
            match = g.loc[g.candidate_defrost_time.eq(replay.selected_time)]
            at_first = g.loc[g.candidate_defrost_time.eq(first)]
            at_last = g.loc[g.candidate_defrost_time.eq(last)]
            record = dict(
                cycle_name=name,
                version=metric.version,
                seed=metric.seed,
                selected_epoch=checkpoint["selected_epoch"],
                epochs_run=len(history),
                first_training_loss=history.training_loss.iloc[0],
                last_training_loss=history.training_loss.iloc[-1],
                validation_cop_levels=history.validation_mean_cycle_cop.nunique(),
                selected_cop=metric.selected_cop,
                optimal_cop=metric.optimal_cop,
                rb_cop=metric.rb_cop,
                trigger_status=metric.trigger_status,
                fallback_terminal=metric.fallback_terminal,
                original_regret_pct=metric.regret_pct,
                near_points=len(near),
                max_hazard=hazard.max(),
                near_max_hazard=near.probability.max(),
                pre_near_max_hazard=g.loc[g.candidate_defrost_time.lt(first), "probability"].max(),
                cdf_at_near_start=cumulative.loc[at_first.index].max(),
                cdf_at_near_end=cumulative.loc[at_last.index].max(),
                selected_minus_near_end_minutes=(
                    pd.Timestamp(metric.selected_time) - last
                ).total_seconds()
                / 60,
                cdf_selected_cop=replay.selected_cop,
                cdf_fallback=replay.forced_final,
                cdf_no_trigger=replay.no_trigger,
                cdf_hit=bool(match.near_optimal.any())
                if not replay.forced_final and not replay.no_trigger
                else False,
                cdf_regret_pct=100 * (1.0 - replay.selected_cop / metric.optimal_cop),
                cdf_delta_cop=replay.selected_cop - metric.selected_cop,
            )
            for column in [
                "control_setpoint_c",
                "control_outlet_c",
                "frequency_headroom_fraction",
                "frequency_change_60s_hz",
                "frequency_change_300s_hz",
            ]:
                if column in g:
                    record["near_start_" + column] = (
                        at_first[column].iloc[0] if len(at_first) else np.nan
                    )
            records.append(record)
    output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result.to_csv(output / "trigger_rule_audit.csv", index=False)
    print(result.to_string(index=False))


def select_trigger_rule(
    run, selection_run, output, ensemble_runs=(), rules=("hazard", "cdf"), candidate_runs=()
):
    """Choose hazard/CDF on inner validation only, then replay frozen outer fits."""
    from image_models.comparisons import write_evaluation_metrics
    from image_models.evaluation import score
    from image_models.experiments import record_experiment
    from image_models.policy import (
        apply_trigger_rule,
        mix_stopping_hazards,
        near_set_timing,
        summarize_metrics,
        trigger_rule_scores,
    )

    settings = json.loads((run / "settings.json").read_text())
    reference = pd.read_parquet(Path(settings["dataset"]) / "training_data/reference.parquet")
    records, choices, cases = [], [], {}
    if ensemble_runs and candidate_runs:
        raise ValueError("Select one candidate model or ensemble seeds, never both")
    for source, selection in [
        (run, selection_run),
        *((r, r) for r in [*ensemble_runs, *candidate_runs]),
    ]:
        source_settings = json.loads((source / "settings.json").read_text())
        inner_settings = json.loads((selection / "settings.json").read_text())
        for key in [
            "folds",
            "dataset",
            "data_definition",
            "inputs",
            "recipes",
            "loss_name",
            "validation_rule",
            "validation_tiebreak",
            "visual_history",
            "joint_trigger_selection",
        ]:
            assert source_settings.get(key) == inner_settings.get(key), (
                f"Inner selection differs in {key}"
            )
            if not (candidate_runs and key == "inputs"):
                assert settings.get(key) == source_settings.get(key), f"Member differs in {key}"
        assert source_settings.get("seeds") == inner_settings.get("seeds")
        paths = checkpoint_files(source)
        if source_settings.get("folds") and source_settings.get("seeds"):
            expected = len(source_settings["folds"]) * len(source_settings["seeds"])
            if len(paths) != expected:
                raise ValueError(f"Incomplete fold matrix: {len(paths)}/{expected} in {source}")
        for path in paths:
            saved = pickle.load(path.open("rb"))
            test = saved["cycle_metrics"].outer_experiment.iloc[0]
            seed = saved["checkpoint"]["seed"]
            matches = checkpoint_files(selection, f"*_{test}_seed{seed}.pkl")
            assert len(matches) == 1
            inner = pickle.load(matches[0].open("rb"))
            pooled = inner["checkpoint"].get("pooled_validation_scores")
            validation = inner.get(
                "validation_predictions",
                inner["predictions"].loc[lambda x: x.split.eq("validation")],
            )
            if pooled is not None:
                assert not ensemble_runs and tuple(rules) in (("hazard", "cdf"), ("cdf",))
                pooled = pooled.loc[pooled.rule.isin(rules)]
                folds = inner["checkpoint"]["inner_validation_folds"]
                assert len(folds) == 2 and set(folds[0]).isdisjoint(folds[1])
                assert set(folds[0] + folds[1]) == set(inner["checkpoint"]["training_experiments"])
                assert test not in set(folds[0] + folds[1])
            else:
                assert len(validation) and test not in set(validation.experiment_id)
                assert set(validation.experiment_id).isdisjoint(
                    inner["checkpoint"].get("selection_training_experiments", [])
                )
            assert test not in saved["checkpoint"].get("training_experiments", [])
            key = (test, -1 if ensemble_runs else seed)
            cases.setdefault(key, []).append(
                (
                    seed,
                    saved["cycle_metrics"].version.iloc[0],
                    validation,
                    saved["predictions"].loc[lambda x: x.split.eq("test")],
                    pooled,
                    str(source),
                )
            )
    for (test, seed), members in cases.items():
        candidate_choices = []
        if candidate_runs:
            assert len(members) == 1 + len(candidate_runs)
            for index, member in enumerate(members):
                grid = (
                    member[4]
                    if member[4] is not None
                    else trigger_rule_scores(member[2], reference, rules)
                )
                candidate_choices.append(
                    grid.assign(
                        candidate_index=index,
                        candidate_version=member[1],
                        candidate_source=member[5],
                    )
                )
            candidates = pd.concat(candidate_choices, ignore_index=True)
            # Same inner experiments, same fixed threshold; outer rewards never select the model.
            candidates = candidates.sort_values(
                ["validation_evaluable", "validation_cop", "candidate_index"],
                ascending=[False, False, True],
                kind="stable",
            )
            winner = int(candidates.iloc[0].candidate_index)
            members = [members[winner]]
        if ensemble_runs:
            assert len(members) == 1 + len(ensemble_runs) and len({m[0] for m in members}) == len(
                members
            )
        validation = (
            mix_stopping_hazards([m[2] for m in members]) if ensemble_runs else members[0][2]
        )
        predicted = (
            mix_stopping_hazards([m[3] for m in members]) if ensemble_runs else members[0][3]
        )
        grid = (
            members[0][4]
            if members[0][4] is not None
            else trigger_rule_scores(validation, reference, rules)
        )
        choices.extend(
            (candidates if candidate_runs else grid)
            .assign(outer_experiment=test, seed=seed)
            .to_dict("records")
        )
        rule = grid.iloc[0].rule
        test_rows = apply_trigger_rule(predicted, rule)
        metric = score(test_rows, reference, strategy="first_positive_terminal")
        timing, _ = near_set_timing(test_rows, metric)
        metric = metric.merge(timing, on="cycle_name", validate="one_to_one")
        metric["fallback_terminal"] = metric.forced_final
        suffix = (
            "_validation_rule" if tuple(rules) == ("hazard", "cdf") else "_" + ("_".join(rules))
        )
        version = members[0][1] + ("_ensemble" if ensemble_runs else "") + suffix
        if candidate_runs:
            version = (
                "MLP"
                if settings.get("architecture", settings.get("recipes", [[""]])[0][0]).startswith(
                    "MLP"
                )
                else "GRU32"
            ) + "_validation_candidate_rule"
        records.append(
            metric.assign(
                version=version,
                seed=seed,
                split="test",
                outer_experiment=test,
                chosen_rule=rule,
                chosen_model=members[0][1],
                chosen_source=members[0][5],
            )
        )
    record_experiment(
        output,
        dict(
            source_run=str(run),
            selection_run=str(selection_run),
            threshold=0.5,
            shared_tuning=settings.get("shared_tuning"),
            trigger_rules=list(rules),
            candidate_runs=list(map(str, candidate_runs)),
            ensemble_runs=list(map(str, ensemble_runs)),
            ensemble="equal mixture of first-event distributions; seed=-1 denotes ensemble"
            if ensemble_runs
            else None,
            selection=(
                "inner validation evaluable count then COP; stable candidate order breaks ties; fixed CDF; no outer test selection"
                if tuple(rules) == ("cdf",)
                else "inner validation evaluable count then COP; hazard wins ties; no outer test selection"
            ),
        ),
        "Validation-only selection of one candidate model with fixed triggering."
        if candidate_runs
        else "Priority C: validation-only selection between fixed hazard and cumulative-event first crossing.",
    )
    metrics = pd.concat(records, ignore_index=True)
    metrics.to_csv(output / "cycle_metrics.csv", index=False)
    write_evaluation_metrics(metrics, settings["dataset"], output)
    choices = pd.DataFrame(choices).merge(
        metrics[["outer_experiment", "seed", "chosen_rule"]].drop_duplicates(),
        on=["outer_experiment", "seed"],
        validate="many_to_one",
    )
    choices["selected"] = choices.rule.eq(choices.chosen_rule)
    if candidate_runs:
        choices = choices.merge(
            metrics[
                ["outer_experiment", "seed", "chosen_model", "chosen_source"]
            ].drop_duplicates(),
            on=["outer_experiment", "seed"],
            validate="many_to_one",
        )
        choices["selected"] &= choices.candidate_source.eq(choices.chosen_source)
    choices.to_csv(output / "rule_selection.csv", index=False)
    summary = pd.DataFrame(
        [
            dict(
                version=version,
                **summarize_metrics(g),
                hits=int(g.near_optimal_hit.eq(True).sum()),
                fallback_count=int(g.fallback_terminal.sum()),
                unsupported_count=int(g.unsupported_trigger.sum()),
                no_trigger_count=int(g.no_trigger.sum()),
            )
            for version, g in metrics.groupby("version")
        ]
    )
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False))


def visual_state_probes(rows, checkpoint, offsets=(-300, 300), frost_bins=False):
    """Replace only the current visual observation, holding actual GRU history fixed.

    Earlier/later images are temporal proxies, not verified frost grades. Positive
    offsets are offline counterfactual probes, never deployable future inputs.
    Each returned row is an independent one-step intervention, not a new trajectory.
    """
    import torch

    from image_models.policy import _model_values, restore_policy

    model = restore_policy(checkpoint)
    if model.visual_skip is not None:
        raise ValueError("Current-state probes require the original joint GRU readout")
    if not checkpoint["visual_columns"]:
        raise ValueError("Visual probes require visual inputs")
    columns = checkpoint["numeric_columns"] + checkpoint["visual_columns"]
    numeric = len(checkpoint["numeric_columns"])
    result = []
    with torch.no_grad():
        for name, g in rows.loc[rows.input_valid].groupby("cycle_name", sort=False):
            g = g.sort_values("candidate_defrost_time").reset_index(drop=True)
            x = _model_values(g, columns, checkpoint["preprocessor"])
            hidden, _ = model.gru(x.unsqueeze(0))
            previous = torch.cat([torch.zeros_like(hidden[:, :1]), hidden[:, :-1]], dim=1)
            base = model.readout(hidden, x[None, :, :numeric]).reshape(-1)
            if "decision_margin" in g:
                np.testing.assert_allclose(
                    base.numpy(), g.decision_margin.to_numpy(), atol=5e-5, rtol=1e-5
                )
            survival = torch.cat([torch.ones(1), (1 - base.sigmoid()).cumprod(0)[:-1]])
            times = g.candidate_defrost_time.to_numpy(dtype="datetime64[ns]")
            donors = []
            for offset in () if frost_bins else offsets:
                target = times + np.timedelta64(offset, "s")
                donor = np.searchsorted(times, target, side="right") - 1
                valid = donor >= 0
                donor = np.maximum(donor, 0)
                valid &= (target - times[donor]) <= np.timedelta64(30, "s")
                donors.append((offset, donor, valid))
            if frost_bins:
                # One real same-cycle image per band; never synthesize a DINO vector.
                for lower in np.arange(10) / 10:
                    upper = lower + 0.1
                    candidates = g.index[
                        g.visual_frost_score.ge(lower)
                        & (
                            g.visual_frost_score.le(upper)
                            if lower == 0.9
                            else g.visual_frost_score.lt(upper)
                        )
                    ]
                    if len(candidates):
                        chosen = (
                            (g.loc[candidates, "visual_frost_score"] - (lower + 0.05))
                            .abs()
                            .idxmin()
                        )
                        donors.append(
                            (float(lower), np.full(len(g), chosen), np.ones(len(g), bool))
                        )
            for offset, donor, valid in donors:
                index = np.flatnonzero(valid)
                if not len(index):
                    continue
                altered = x[index].clone()
                altered[:, numeric:] = x[donor[index], numeric:]
                state, _ = model.gru(altered.unsqueeze(1), previous[:, index, :])
                changed = model.readout(state, altered[:, None, :numeric]).reshape(-1)
                frame = g.iloc[index][
                    ["cycle_name", "experiment_id", "candidate_defrost_time", "near_optimal"]
                ].copy()
                frame["donor_time"] = times[donor[index]]
                frame["offset_seconds"] = offset
                if frost_bins:
                    frame["frost_bin_lower"] = offset
                    frame["original_frost_score"] = g.visual_frost_score.iloc[index].to_numpy()
                    frame["donor_frost_score"] = g.visual_frost_score.iloc[donor[index]].to_numpy()
                    frame["offset_seconds"] = (times[donor[index]] - times[index]) / np.timedelta64(
                        1, "s"
                    )
                frame["original_margin"] = base[index].numpy()
                frame["probe_margin"] = changed.numpy()
                frame["delta_margin"] = (changed - base[index]).numpy()
                frame["original_hazard"] = base[index].sigmoid().numpy()
                frame["probe_hazard"] = changed.sigmoid().numpy()
                frame["original_cdf"] = (1 - survival[index] * (1 - base[index].sigmoid())).numpy()
                frame["probe_cdf"] = (1 - survival[index] * (1 - changed.sigmoid())).numpy()
                result.append(frame)
    return pd.concat(result, ignore_index=True)


def audit_visual_states(run, selected_run, output, frost_progression=None):
    import torch

    from dataset_tools import DatasetLoader
    from image_models.experiments import record_experiment

    torch.set_num_threads(1)
    selected = pd.read_csv(selected_run / "cycle_metrics.csv")
    settings = json.loads((run / "settings.json").read_text())
    from image_models.cycle_selection import screen_published_cycles

    screening = screen_published_cycles(settings["dataset"])
    selected = selected.loc[
        selected.cycle_name.isin(screening.loc[screening.selected, "cycle_name"])
    ]
    inputs, _ = DatasetLoader(settings["dataset"]).load_training_data()
    inputs = inputs.loc[inputs.cycle_name.isin(selected.cycle_name)]
    if settings.get("compensation_progress"):
        from image_models.inputs import causal_compensation_progress

        inputs = causal_compensation_progress(inputs)
    if settings.get("elapsed_context"):
        from image_models.inputs import elapsed_cycle_minutes

        inputs["elapsed_cycle_minutes"] = elapsed_cycle_minutes(inputs)
    if settings.get("temperature_representation") == "tracking_error":
        inputs["temperature_error_c"] = inputs.control_setpoint_c - inputs.control_outlet_c
    if "chosen_rule" not in selected:
        selected["chosen_rule"] = settings["trigger_rule"]
    if frost_progression:
        source = json.loads((frost_progression.parent / "run_settings.json").read_text())
        if source["coordinate_correction"] != "harmony":
            raise ValueError("Frost-band diagnostic requires native Harmony + Slingshot")
        frost = pd.read_csv(frost_progression)
        frost = frost.loc[
            frost.camera_role.eq("front"), ["cycle_name", "file_name", "visual_frost_score"]
        ]
        inputs = inputs.drop(columns="visual_frost_score").merge(
            frost, on=["cycle_name", "file_name"], how="left", validate="many_to_one"
        )
    records = []
    for path in checkpoint_files(run):
        saved = pickle.load(path.open("rb"))
        predictions = saved["predictions"].loc[lambda x: x.split.eq("test")]
        keys = ["cycle_name", "candidate_defrost_time"]
        rows = inputs.merge(
            predictions[keys + ["decision_margin"]], on=keys, how="inner", validate="one_to_one"
        )
        probes = visual_state_probes(rows, saved["checkpoint"], frost_bins=bool(frost_progression))
        probes["seed"] = saved["checkpoint"]["seed"]
        probes["version"] = saved["cycle_metrics"].version.iloc[0]
        records.append(probes)
    points = pd.concat(records, ignore_index=True).merge(
        selected[["cycle_name", "seed", "version", "chosen_rule", "selected_time"]],
        on=["cycle_name", "seed", "version"],
        validate="many_to_one",
    )
    points["before_or_at_actual_stop"] = points.candidate_defrost_time.le(
        pd.to_datetime(points.selected_time, format="mixed")
    )
    for prefix in ["original", "probe"]:
        points[prefix + "_score"] = np.where(
            points.chosen_rule.eq("cdf"), points[prefix + "_cdf"], points[prefix + "_hazard"]
        )
    points["continue_to_stop"] = (
        points.original_score.lt(0.5)
        & points.probe_score.ge(0.5)
        & points.candidate_defrost_time.lt(pd.to_datetime(points.selected_time, format="mixed"))
    )
    points["stop_to_continue"] = (
        points.original_score.ge(0.5) & points.probe_score.lt(0.5) & points.before_or_at_actual_stop
    )
    if frost_progression:
        # CDF carries earlier decisions: report it separately from current-frame hazard.
        points["hazard_stop"] = points.probe_hazard.ge(0.5)
        points["rule_stop"] = points.probe_score.ge(0.5)
        cycle = (
            points.loc[points.before_or_at_actual_stop]
            .groupby(["cycle_name", "experiment_id", "seed", "frost_bin_lower"])
            .agg(
                actions=("delta_margin", "size"),
                donor_frost_score=("donor_frost_score", "first"),
                mean_delta_margin=("delta_margin", "mean"),
                mean_hazard=("probe_hazard", "mean"),
                hazard_stop_fraction=("hazard_stop", "mean"),
                rule_stop_fraction=("rule_stop", "mean"),
            )
            .reset_index()
        )
        summary = (
            cycle.groupby(["seed", "frost_bin_lower"])
            .agg(
                cycles=("cycle_name", "size"),
                mean_hazard=("mean_hazard", "mean"),
                hazard_stop_fraction=("hazard_stop_fraction", "mean"),
                rule_stop_fraction=("rule_stop_fraction", "mean"),
                mean_delta_margin=("mean_delta_margin", "mean"),
            )
            .reset_index()
        )
        # Across-band comparisons need identical cycles and original action contexts.
        paired = []
        for seed, g in cycle.groupby("seed"):
            low = g.loc[g.frost_bin_lower.eq(0.1)].set_index("cycle_name")
            for lower in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                high = g.loc[np.isclose(g.frost_bin_lower, lower)].set_index("cycle_name")
                common = low.index.intersection(high.index)
                delta = high.loc[common].mean_hazard - low.loc[common].mean_hazard
                paired.append(
                    dict(
                        seed=seed,
                        low_bin=0.1,
                        high_bin=lower,
                        paired_cycles=len(common),
                        higher_hazard_cycles=int(delta.gt(0).sum()),
                        lower_hazard_cycles=int(delta.lt(0).sum()),
                        mean_low_hazard=low.loc[common].mean_hazard.mean(),
                        mean_high_hazard=high.loc[common].mean_hazard.mean(),
                    )
                )
        record_experiment(
            output,
            dict(
                source_run=str(run),
                selected_run=str(selected_run),
                frost_progression=str(frost_progression),
                coordinate_correction="harmony",
                bins="[0,.1),...,[.9,1]; same-cycle real donor nearest bin midpoint",
                interpretation="Offline proxy-state intervention, current time/control/GRU history fixed; global descriptive Slingshot is not independent frost ground truth",
            ),
            "Extreme-to-middle visual evidence audit; hazard and accumulated deployment score reported separately.",
        )
        screening.to_csv(output / "cycle_screening.csv", index=False)
        points.to_csv(output / "visual_state_action_probes.csv", index=False)
        cycle.to_csv(output / "visual_state_cycle_probes.csv", index=False)
        summary.to_csv(output / "visual_state_summary.csv", index=False)
        pd.DataFrame(paired).to_csv(output / "visual_state_paired_bands.csv", index=False)
        observed = points.drop_duplicates(["cycle_name", "seed", "candidate_defrost_time"])
        at_stop = observed.loc[
            observed.candidate_defrost_time.eq(
                pd.to_datetime(observed.selected_time, format="mixed")
            ),
            [
                "cycle_name",
                "seed",
                "candidate_defrost_time",
                "original_frost_score",
                "original_hazard",
                "original_score",
                "near_optimal",
            ],
        ]
        at_stop.to_csv(output / "visual_state_at_actual_stop.csv", index=False)
        near = inputs.loc[
            inputs.cycle_cop_eligible & inputs.cycle_cop.ge(0.99 * inputs.optimal_cop)
        ]
        evidence = (
            near.groupby("cycle_name")
            .visual_frost_score.agg(near_frost_min="min", near_frost_max="max")
            .reset_index()
        )
        evidence = selected.merge(
            evidence, on="cycle_name", how="left", validate="many_to_one"
        ).merge(
            at_stop[["cycle_name", "seed", "original_frost_score"]],
            on=["cycle_name", "seed"],
            how="left",
            validate="one_to_one",
        )
        evidence.to_csv(output / "frost_cycle_evidence.csv", index=False)
        inputs.groupby("cycle_name").visual_frost_score.agg(["count", "min", "max"]).to_csv(
            output / "frost_coordinate_coverage.csv"
        )
        print(summary.to_string(index=False))
        return
    points["direction_consistent"] = points.delta_margin.mul(np.sign(points.offset_seconds)).gt(0)
    cycle = (
        points.groupby(["cycle_name", "experiment_id", "seed", "offset_seconds"])
        .agg(
            paired_actions=("delta_margin", "size"),
            mean_delta_margin=("delta_margin", "mean"),
            direction_consistent_fraction=("direction_consistent", "mean"),
            continue_to_stop_count=("continue_to_stop", "sum"),
            stop_to_continue_count=("stop_to_continue", "sum"),
        )
        .reset_index()
    )
    summary = (
        cycle.groupby(["seed", "offset_seconds"])
        .agg(
            cycles=("cycle_name", "size"),
            mean_cycle_delta_margin=("mean_delta_margin", "mean"),
            mean_cycle_direction_consistent=("direction_consistent_fraction", "mean"),
            cycles_advance=("continue_to_stop_count", lambda x: int(x.gt(0).sum())),
            cycles_suppress=("stop_to_continue_count", lambda x: int(x.gt(0).sum())),
        )
        .reset_index()
    )
    record_experiment(
        output,
        dict(
            source_run=str(run),
            selected_run=str(selected_run),
            offset_seconds=[-300, 300],
            intervention="current visual vector only; original control/time and original GRU history fixed; <=30s donor match; no extrapolated edge frames",
            interpretation="Offline temporal-visual sensitivity, not measured frost severity or physical counterfactual COP; do not replay independently probed rows as one trajectory",
        ),
        "Does the frozen classifier respond to earlier/later visual observations at identical current controls and history?",
    )
    points.to_csv(output / "visual_state_action_probes.csv", index=False)
    cycle.to_csv(output / "visual_state_cycle_probes.csv", index=False)
    summary.to_csv(output / "visual_state_summary.csv", index=False)
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loss-budget", action="store_true")
    parser.add_argument(
        "--cycles", type=int, nargs="+", help="Non-baseline frozen-model mechanism audit"
    )
    parser.add_argument(
        "--trigger-audit",
        action="store_true",
        help="Audit hazard, CDF crossing and selection plateau in frozen fits",
    )
    parser.add_argument(
        "--visual-support-audit",
        action="store_true",
        help="Numerical early-cycle DINO magnitude and same-condition held-out experiment support",
    )
    parser.add_argument(
        "--cdf-fit-cycles",
        type=int,
        nargs="+",
        help="Frozen baseline CDF in-sample versus held-out fitting diagnostic",
    )
    parser.add_argument(
        "--cdf-target-audit",
        action="store_true",
        help="Replay a perfect ramp target at CDF .5 on the published input support; diagnostic oracle only",
    )
    parser.add_argument(
        "--selection-run",
        type=Path,
        help="Inner-fit run for validation-only hazard/CDF rule selection",
    )
    parser.add_argument(
        "--ensemble-runs",
        type=Path,
        nargs="*",
        default=[],
        help="Additional independent seeds with embedded validation predictions; equal event-distribution mixture",
    )
    parser.add_argument(
        "--candidate-runs",
        type=Path,
        nargs="*",
        default=[],
        help="Alternative feature recipes; inner validation selects one model per fold, no ensemble",
    )
    parser.add_argument(
        "--trigger-rules",
        nargs="+",
        choices=["hazard", "cdf"],
        default=["hazard", "cdf"],
    )
    parser.add_argument(
        "--visual-state-audit",
        action="store_true",
        help="Fixed-history, fixed-control current visual replacement; selection-run supplies frozen chosen rules",
    )
    parser.add_argument(
        "--frost-progression",
        type=Path,
        help="Native Harmony Slingshot progression.csv; probe real donors in ten frost-coordinate bands",
    )
    parser.add_argument(
        "--evaluation-summary",
        action="store_true",
        help="Unified overall, setpoint, RB-early and RB-good metrics for saved decisions",
    )
    parser.add_argument(
        "--rescore-decisions",
        action="store_true",
        help="Freeze saved stopping times and rescore against the current published teacher",
    )
    parser.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    args = parser.parse_args()
    from output_layout import prepare_output

    prepare_output(args, training=False)
    execute(args)
    from output_layout import OUTPUT
    from tools.result_index import write_output_index

    if args.output.resolve().is_relative_to(OUTPUT):
        write_output_index()


def execute(args):
    if args.visual_support_audit:
        audit_visual_support(args.run, args.output)
        return
    if args.cdf_target_audit:
        audit_cdf_target(args.run, args.output)
        return
    if args.cdf_fit_cycles:
        audit_cdf_fit(args.run, args.output, args.cdf_fit_cycles)
        return
    if args.rescore_decisions:
        rescore_decisions(args.run, args.output, args.dataset)
        return
    if args.evaluation_summary:
        from image_models.comparisons import write_evaluation_metrics
        from image_models.experiments import record_experiment

        settings = json.loads((args.run / "settings.json").read_text())
        source = (
            json.loads((Path(settings["source_run"]) / "settings.json").read_text())
            if "source_run" in settings
            else settings
        )
        record_experiment(
            args.output,
            dict(
                source_run=str(args.run),
                dataset=source["dataset"],
                delay_space="last supported raw 1% near-COP point minus RB; positive spaces only; ratio of sums, unclipped",
                rb_good="supported RB COP >= 99% of cycle oracle; early from fixed user-designated Dataset catalog",
                hit="raw COP >=99% optimum; all-cycle denominator includes unsupported decisions",
                filled_hit="selected time within first/last exact near points on legal action grid, inclusive; no-trigger excluded; timing metric does not certify COP",
            ),
            "Unified COP, hit, duration and delay-space evaluation by subset and operating setpoint.",
        )
        write_evaluation_metrics(
            pd.read_csv(args.run / "cycle_metrics.csv"), source["dataset"], args.output
        )
        return
    if args.visual_state_audit:
        if args.selection_run is None:
            raise ValueError("Visual-state audit requires the selected-rule run")
        audit_visual_states(args.run, args.selection_run, args.output, args.frost_progression)
        return
    if args.selection_run:
        select_trigger_rule(
            args.run,
            args.selection_run,
            args.output,
            args.ensemble_runs,
            args.trigger_rules,
            args.candidate_runs,
        )
        return
    if args.trigger_audit:
        audit_trigger_rules(args.run, args.output)
        return
    if args.cycles:
        diagnose_cycles(args.run, args.output, args.cycles)
        return
    if args.loss_budget:
        admission_diagnostics(args.run, args.output)
        return
    raise ValueError("Choose an explicit evaluation or mechanism-audit option")


if __name__ == "__main__":
    main()
