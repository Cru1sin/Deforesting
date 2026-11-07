"""Decompose frozen COP on adjacent supported actions; never change teacher selection."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from defrost_decision.performance_objectives import add_single_objective_optima, calculate_cycle_cop
from image_models.overlap import crossing_intervals


def read_models(run):
    return json.loads((run / "candidate_model_parameters.json").read_text())


def parameters(models, experiment):
    return models["models"]["ridge_dynamic_state_8"]["event_electricity"]["folds"][experiment]


def decompose(g, steps=1, step_seconds=30):
    g = g.sort_values("candidate_defrost_time").reset_index(drop=True).copy()
    nxt = g.shift(-steps)
    complete = g.cycle_cop_eligible.eq(True).rolling(steps + 1).sum().shift(-steps).eq(steps + 1)
    pair = complete & (nxt.candidate_defrost_time - g.candidate_defrost_time).dt.total_seconds().eq(
        step_seconds * steps
    )
    for short, col in [
        ("Q", "pre_defrost_heat_kwh"),
        ("E", "pre_defrost_electricity_kwh"),
        ("D", "defrost_event_electricity_kwh"),
    ]:
        g[f"d{short}"] = nxt[col] - g[col]
    g["interval_hours"] = (
        nxt.candidate_defrost_time - g.candidate_defrost_time
    ).dt.total_seconds() / 3600
    g["heat_margin_kw"] = (g.dQ - g.cycle_cop * g.dE) / g.interval_hours
    g["cost_margin_kw"] = g.cycle_cop * g.dD / g.interval_hours
    g["net_margin_kw"] = g.heat_margin_kw - g.cost_margin_kw
    g["q_kw"], g["p_kw"] = g.dQ / g.interval_hours, g.dE / g.interval_hours
    g["instant_cop"] = g.dQ / g.dE
    expected = (
        g.net_margin_kw
        * g.interval_hours
        / (nxt.pre_defrost_electricity_kwh + nxt.defrost_event_electricity_kwh)
    )
    np.testing.assert_allclose((nxt.cycle_cop - g.cycle_cop)[pair], expected[pair], atol=1e-12)
    g["supported_pair"] = pair
    return g


def frequency_onsets(grid):
    """Observed frequency onsets, not proof that the thermodynamic state was steady."""
    f, t = grid.compressor_frequency, grid.candidate_defrost_time
    before = f.rolling(7).max() - f.rolling(7).min()
    change = f.shift(-1) - f
    sign = np.sign(change)
    persistent = ((f.shift(-2) - f) * sign).ge(2) & ((f.shift(-3) - f) * sign).ge(2)
    continuous = (t - t.shift(6)).dt.total_seconds().eq(180) & (
        t.shift(-3) - t
    ).dt.total_seconds().eq(90)
    return before.le(1) & change.abs().ge(2) & persistent & continuous


def frequency_audit(args):
    """Audit frozen predictions during frequency ramps; never move any selector."""
    from dataset_tools import DatasetLoader
    from defrost_event_models.event_electricity_ridge import predict_with_model_parameters
    from image_models.cycle_selection import screen_cv_cycles

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    loader = DatasetLoader(args.dataset)
    decisions = pd.read_csv(args.decisions / "candidate_decisions.csv", low_memory=False)
    decisions.candidate_defrost_time = pd.to_datetime(
        decisions.candidate_defrost_time, format="mixed"
    )
    screening = screen_cv_cycles(decisions, loader.list_cycles(), exclude_rb=False)
    screening.to_csv(out / "cycle_screening.csv", index=False)
    names = set(screening.loc[screening.selected, "cycle_name"])
    models = read_models(args.model)
    records, intervals = [], []
    for name, g in decisions.loc[decisions.cycle_name.isin(names)].groupby("cycle_name"):
        g = g.sort_values("candidate_defrost_time").reset_index(drop=True)
        raw = loader.load_cycle_measurements(name)
        times = pd.to_datetime(raw.timestamp)
        stable = raw.loc[
            times.ge(pd.Timestamp(g.heating_accounting_start.iloc[0]))
            & times.lt(g.candidate_defrost_time.max()),
            "water_temperature_setpoint",
        ].dropna()
        task_range = stable.max() - stable.min()
        records.append(
            dict(
                cycle_name=name,
                experiment_id=g.experiment_id.iloc[0],
                stable_setpoint_range_c=task_range,
                observed_setpoint_count=len(stable),
                constant_task=bool(task_range <= 0.5),
            )
        )
        if not task_range <= 0.5:
            continue
        p = parameters(models, g.experiment_id.iloc[0])
        prediction = predict_with_model_parameters(p, g)
        np.testing.assert_allclose(
            prediction.prediction, g.defrost_event_electricity_kwh, atol=1e-10
        )
        features = p["feature_order"]
        # Reproduce the saved predictor's imputation for exact feature attribution.
        x = g[features].fillna(dict(zip(features, p["imputer_median"])))
        valid = g.cycle_cop_eligible.eq(True) & g.cycle_cop.notna()
        peak = g.loc[valid, "cycle_cop"].idxmax()
        end = g.loc[valid].index[-1]
        grid = g.loc[g.on_action_grid.eq(True)].reset_index(drop=True)
        gx = x.loc[g.on_action_grid.eq(True)].reset_index(drop=True)
        onsets = frequency_onsets(grid)
        for scope, curve, values, steps in [
            ("30s", grid, gx, 1),
            ("5min", grid, gx, 10),
            ("onset_30s", grid, gx, 1),
            ("onset_2min", grid, gx, 4),
            ("onset_5min", grid, gx, 10),
            (
                "peak_to_end",
                g.loc[[peak, end]].reset_index(drop=True),
                x.loc[[peak, end]].reset_index(drop=True),
                1,
            ),
        ]:
            part = decompose(curve, steps=steps)
            nxt = curve.shift(-steps)
            denominator = nxt.pre_defrost_electricity_kwh + nxt.defrost_event_electricity_kwh
            part["measured_cop_change"] = part.heat_margin_kw * part.interval_hours / denominator
            part["ridge_cop_change"] = -part.cost_margin_kw * part.interval_hours / denominator
            part["cop_change"] = nxt.cycle_cop - part.cycle_cop
            part["end_time"] = nxt.candidate_defrost_time
            part["frequency_change_hz"] = nxt.compressor_frequency - part.compressor_frequency
            part["after_peak"] = part.candidate_defrost_time.ge(
                g.loc[peak, "candidate_defrost_time"]
            )
            part["both_in_domain"] = part.defrost_event_electricity_in_training_domain.eq(
                True
            ) & nxt.defrost_event_electricity_in_training_domain.eq(True)
            parts = []
            for f, b, s in zip(features, p["coefficients"], p["scaler_scale"], strict=True):
                col = "ridge_delta_" + f
                part[col] = (values[f].shift(-steps) - values[f]) * b / s
                parts.append(col)
            keep = part.supported_pair if scope != "peak_to_end" else part.index.to_series().eq(0)
            if scope.startswith("onset_"):
                keep &= onsets
                part["onset_direction"] = np.where(
                    grid.compressor_frequency.shift(-1).gt(grid.compressor_frequency),
                    "increase",
                    "decrease",
                )
            part = part.loc[keep].copy()
            np.testing.assert_allclose(part[parts].sum(axis=1), part.dD, atol=1e-10)
            np.testing.assert_allclose(
                part.measured_cop_change + part.ridge_cop_change, part.cop_change, atol=1e-12
            )
            part["scope"] = scope
            part["ridge_net_decline_fraction"] = -part.ridge_cop_change / (-part.cop_change).where(
                part.cop_change.lt(-1e-12)
            )
            part["ridge_majority_decline"] = part.cop_change.lt(
                -1e-12
            ) & part.ridge_net_decline_fraction.gt(0.5)
            columns = [
                "cycle_name",
                "experiment_id",
                "scope",
                "candidate_defrost_time",
                "end_time",
                "after_peak",
                "both_in_domain",
                "compressor_frequency",
                "frequency_change_hz",
                "cycle_cop",
                "defrost_event_electricity_kwh",
                "dD",
                "measured_cop_change",
                "ridge_cop_change",
                "cop_change",
                "ridge_net_decline_fraction",
                "ridge_majority_decline",
            ] + parts
            if scope.startswith("onset_"):
                columns.append("onset_direction")
            intervals.append(part[columns])
    pd.DataFrame(records).to_csv(out / "constant_setpoint_cycles.csv", index=False)
    points = pd.concat(intervals, ignore_index=True)
    points.to_csv(out / "cop_change_attribution.csv", index=False)
    onset_points = points.loc[points.scope.str.startswith("onset_")].copy()
    keys = ["cycle_name", "candidate_defrost_time"]
    # Compare the same onset at all horizons; this does not select or move any action.
    complete = onset_points.groupby(keys).scope.transform("nunique").eq(3)
    onset_points = onset_points.loc[complete].copy()
    onset_points["absolute_cost_cop_change"] = onset_points.ridge_cop_change.abs()
    onset_points["other_feature_cost_change_kwh"] = (
        onset_points.dD - onset_points.ridge_delta_compressor_frequency
    )
    onset_cycles = onset_points.groupby(["scope", "onset_direction", "cycle_name"]).agg(
        n_onsets=("dD", "size"),
        median_cost_change_kwh=("dD", "median"),
        median_frequency_term_kwh=("ridge_delta_compressor_frequency", "median"),
        median_other_terms_kwh=("other_feature_cost_change_kwh", "median"),
        median_absolute_cost_cop_change=("absolute_cost_cop_change", "median"),
    )
    onset_cycles.to_csv(out / "frequency_onset_cycles.csv")
    onset_cycles.groupby(["scope", "onset_direction"]).agg(
        n_cycles=("n_onsets", "size"),
        n_onsets=("n_onsets", "sum"),
        median_cost_change_kwh=("median_cost_change_kwh", "median"),
        median_frequency_term_kwh=("median_frequency_term_kwh", "median"),
        median_other_terms_kwh=("median_other_terms_kwh", "median"),
        median_absolute_cost_cop_change=("median_absolute_cost_cop_change", "median"),
    ).to_csv(out / "frequency_onset_summary.csv")
    summaries = []
    for scope, frame in points.groupby("scope"):
        for threshold in [0, 2, 5]:
            for post_peak in [False, True]:
                subset = frame.loc[
                    frame.frequency_change_hz.gt(threshold)
                    if threshold == 0
                    else frame.frequency_change_hz.ge(threshold)
                ]
                if post_peak:
                    subset = subset.loc[subset.after_peak]
                falling = subset.loc[subset.cop_change.lt(-1e-12)]
                cycle_shares = falling.groupby("cycle_name").ridge_net_decline_fraction.median()
                summaries.append(
                    dict(
                        scope=scope,
                        minimum_rise_hz=threshold,
                        after_peak_only=post_peak,
                        n_intervals=len(subset),
                        n_cycles=subset.cycle_name.nunique(),
                        falling_intervals=len(falling),
                        falling_cycles=falling.cycle_name.nunique(),
                        ridge_majority_intervals=int(falling.ridge_majority_decline.sum()),
                        ridge_majority_cycles=falling.loc[
                            falling.ridge_majority_decline, "cycle_name"
                        ].nunique(),
                        cycle_median_ridge_fraction=cycle_shares.median(),
                        cycles_with_median_ridge_majority=int(cycle_shares.gt(0.5).sum()),
                        median_ridge_net_decline_fraction=falling.ridge_net_decline_fraction.median(),
                        median_measured_cop_change=falling.measured_cop_change.median(),
                        median_ridge_cop_change=falling.ridge_cop_change.median(),
                        median_cop_change=falling.cop_change.median(),
                    )
                )
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "summary.csv", index=False)
    configuration = dict(
        dataset=str(args.dataset),
        decisions=str(args.decisions),
        model=str(args.model),
        full_cycles=len(screening),
        screened_cycles=len(names),
        constant_setpoint_cycles=sum(r["constant_task"] for r in records),
        setpoint_range_limit_c=0.5,
        scope="stable heating; no RGB filtering; includes RB",
        onset_rule="preceding 180s frequency range <=1Hz; next 30s change >=2Hz, retained at 60s and 90s; continuous grid",
        onset_summary="same supported onsets at 30s, 2min and 5min; median of within-cycle medians",
        limitation="Overlapping windows are descriptive, not independent samples; frost causation is not established.",
        attribution="Exact endpoint identity; >100% means measured heat/electricity offset part of Ridge-induced decline.",
    )
    (out / "configuration.json").write_text(json.dumps(configuration, ensure_ascii=False, indent=2))
    (out / "run.log").write_text(
        "Purpose: distinguish local COP downturns from peak-to-end decline during observed frequency rise.\n"
        "Full saved cohort screened before observed constant-setpoint restriction.\n"
        "All saved Ridge predictions reconstructed; feature sums and COP changes verified numerically.\n"
        "No model refit, selector change, Dataset change, or causal frost attribution.\n"
        + summary.to_string(index=False)
        + "\n"
    )
    print(json.dumps(configuration, ensure_ascii=False), flush=True)
    print(summary.to_string(index=False), flush=True)


def synchronous_frequency_audit(args):
    """Describe simultaneous changes on saved actions, with exact COP attribution."""
    from dataset_tools import DatasetLoader
    from image_models.cycle_selection import screen_cv_cycles
    from plots.frequency import render_frequency_cop_response

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(args.decisions / "candidate_decisions.csv", low_memory=False)
    decisions.candidate_defrost_time = pd.to_datetime(
        decisions.candidate_defrost_time, format="mixed"
    )
    screening = screen_cv_cycles(
        decisions, DatasetLoader(args.dataset).list_cycles(), exclude_rb=False
    )
    screening.to_csv(out / "cycle_screening.csv", index=False)
    selected = set(screening.loc[screening.selected, "cycle_name"])
    records = []
    for name, frame in decisions.groupby("cycle_name", sort=True):
        grid = (
            frame.loc[frame.on_action_grid.eq(True)]
            .sort_values("candidate_defrost_time")
            .reset_index(drop=True)
        )
        for steps in (1, 4):
            part = decompose(grid, steps=steps)
            nxt = grid.shift(-steps)
            denominator = nxt.pre_defrost_electricity_kwh + nxt.defrost_event_electricity_kwh
            part["measured_cop_change"] = (part.dQ - part.cycle_cop * part.dE) / denominator
            part["ridge_cop_change"] = -part.cycle_cop * part.dD / denominator
            part["cop_change"] = nxt.cycle_cop - part.cycle_cop
            part["frequency_change_hz"] = nxt.compressor_frequency - part.compressor_frequency
            part["end_time"] = nxt.candidate_defrost_time
            part["window_seconds"] = steps * 30
            part["selected_cohort"] = name in selected
            part["extrapolated_window"] = (
                grid.cycle_cop_uses_model_extrapolation.eq(True)
                .rolling(steps + 1)
                .max()
                .shift(-steps)
                .eq(1)
            )
            part = part.loc[part.supported_pair & part.frequency_change_hz.notna()].copy()
            np.testing.assert_allclose(
                part.measured_cop_change + part.ridge_cop_change, part.cop_change, atol=1e-12
            )
            part["direction"] = np.select(
                [part.frequency_change_hz.gt(0), part.frequency_change_hz.lt(0)],
                ["increase", "decrease"],
                default="unchanged",
            )
            part["falling"] = part.cop_change.lt(-1e-12)
            part["falling_over_0_1pct"] = (part.cop_change / part.cycle_cop).lt(-0.001)
            part["ridge_majority_decline"] = part.falling & part.ridge_cop_change.lt(
                0.5 * part.cop_change
            )
            part["ridge_reversed_sign"] = part.falling & part.measured_cop_change.ge(0)
            columns = [
                "cycle_name",
                "experiment_id",
                "candidate_defrost_time",
                "end_time",
                "window_seconds",
                "selected_cohort",
                "extrapolated_window",
                "compressor_frequency",
                "frequency_change_hz",
                "direction",
                "cycle_cop",
                "cop_change",
                "measured_cop_change",
                "ridge_cop_change",
                "dQ",
                "dE",
                "dD",
                "q_kw",
                "p_kw",
                "falling",
                "falling_over_0_1pct",
                "ridge_majority_decline",
                "ridge_reversed_sign",
            ]
            records.append(part[columns])
    points = pd.concat(records, ignore_index=True)
    points.to_csv(out / "frequency_cop_windows.csv", index=False)
    summaries = []
    for scope, frame in [
        ("all_valid", points),
        ("selected_cohort", points.loc[points.selected_cohort]),
    ]:
        for (seconds, direction), part in frame.groupby(["window_seconds", "direction"]):
            subsets = [("any", part)]
            if direction != "unchanged":
                subsets.append(("at_least_2Hz", part.loc[part.frequency_change_hz.abs().ge(2)]))
            for threshold, sample in subsets:
                falling = sample.loc[sample.falling]
                summaries.append(
                    dict(
                        scope=scope,
                        window_seconds=seconds,
                        direction=direction,
                        threshold=threshold,
                        n_windows=len(sample),
                        n_cycles=sample.cycle_name.nunique(),
                        n_experiments=sample.experiment_id.nunique(),
                        falling_windows=int(sample.falling.sum()),
                        falling_fraction=sample.falling.mean(),
                        cycle_macro_falling_fraction=sample.groupby("cycle_name")
                        .falling.mean()
                        .mean(),
                        falling_over_0_1pct_fraction=sample.falling_over_0_1pct.mean(),
                        extrapolated_windows=int(sample.extrapolated_window.sum()),
                        median_cop_change=sample.cop_change.median(),
                        median_measured_change_when_falling=falling.measured_cop_change.median(),
                        median_ridge_change_when_falling=falling.ridge_cop_change.median(),
                        ridge_majority_falling_windows=int(falling.ridge_majority_decline.sum()),
                        ridge_reversed_sign_windows=int(falling.ridge_reversed_sign.sum()),
                    )
                )
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "frequency_cop_summary.csv", index=False)
    # Show both outcomes, choosing the multivariate median-rank example in each group.
    rising = points.loc[
        points.window_seconds.eq(30)
        & points.direction.eq("increase")
        & points.frequency_change_hz.ge(2)
    ]
    examples = []
    for falling, group in rising.groupby("falling"):
        distance = (
            (group[["frequency_change_hz", "cop_change", "ridge_cop_change"]].rank(pct=True) - 0.5)
            .abs()
            .sum(axis=1)
        )
        row = group.loc[distance.idxmin()].copy()
        row["selection"] = "median_rank_falling" if falling else "median_rank_not_falling"
        examples.append(row)
    examples = pd.DataFrame(examples)
    examples.to_csv(out / "example_selection.csv", index=False)
    curves = []
    for row in examples.itertuples():
        center = row.candidate_defrost_time
        g = decisions.loc[
            decisions.cycle_name.eq(row.cycle_name)
            & decisions.on_action_grid.eq(True)
            & decisions.candidate_defrost_time.between(
                center - pd.Timedelta(minutes=4), center + pd.Timedelta(minutes=4)
            )
        ].copy()
        g = g.sort_values("candidate_defrost_time").reset_index(drop=True)
        g["selection"] = row.selection
        g["minutes_from_event"] = (g.candidate_defrost_time - center).dt.total_seconds() / 60
        g["zero_cost_cop"] = g.pre_defrost_heat_kwh / g.pre_defrost_electricity_kwh
        d0 = g.loc[g.candidate_defrost_time.eq(center), "defrost_event_electricity_kwh"].iloc[0]
        g["fixed_cost_cop"] = g.pre_defrost_heat_kwh / (g.pre_defrost_electricity_kwh + d0)
        g[["cycle_cop", "zero_cost_cop", "fixed_cost_cop"]] = g[
            ["cycle_cop", "zero_cost_cop", "fixed_cost_cop"]
        ].where(g.cycle_cop_eligible, axis=0)
        curves.append(g)
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(out / "example_curves.csv", index=False)
    (out / "configuration.json").write_text(
        json.dumps(
            dict(
                decisions=str(args.decisions),
                dataset=str(args.dataset),
                cohort_screen="screen_cv_cycles, full without_trunc, exclude_rb=False; before any filtering",
                population="all physically eligible pairs; selected cohort reported separately; no RGB or setpoint filter",
                windows_seconds=[30, 120],
                extrapolation="annotation only",
                example_rule="30s rise >=2Hz; nearest median percentile ranks of frequency/COP/cost contributions in each outcome group",
                attribution="delta COP = (delta Q - COP0*delta E - COP0*delta D)/(E1+D1)",
                limitations="observational association, overlapping windows; 60s causal heat smoothing affects apparent timing; no causal intervention",
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    render_frequency_cop_response(points, summary, curves, examples, out)
    print(summary.to_string(index=False))


def frequency_episodes(frame, maximum_gap_seconds=30):
    """Join same-direction steps across plateaus; split at reversals or missing data."""
    f = (
        frame[["timestamp", "compressor_frequency"]]
        .dropna()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    blocks = f.timestamp.diff().dt.total_seconds().gt(maximum_gap_seconds).cumsum()
    episodes = []
    for _, block in f.groupby(blocks):
        delta = block.compressor_frequency.diff()
        changes = block.loc[delta.notna() & delta.ne(0)].copy()
        changes["direction"] = np.sign(delta.loc[changes.index]).astype(int)
        runs = changes.direction.ne(changes.direction.shift()).cumsum()
        for _, run in changes.groupby(runs):
            first, last = run.index[0], run.index[-1]
            episodes.append(
                dict(
                    start_time=f.loc[first, "timestamp"],
                    end_time=f.loc[last, "timestamp"],
                    block_end=block.timestamp.iloc[-1],
                    direction="increase" if run.direction.iloc[0] > 0 else "decrease",
                    start_frequency_hz=f.loc[first - 1, "compressor_frequency"],
                    end_frequency_hz=f.loc[last, "compressor_frequency"],
                    n_steps=len(run),
                )
            )
    return episodes


def plot_cycle_frequency_episodes(group, dataset, phase_models, output):
    from dataset_tools import DatasetLoader
    from defrost_event_models.training_data import extract_pre_defrost_features
    from plots.frequency import render_frequency_episode

    loader = DatasetLoader(dataset)
    name = group.cycle_name.iloc[0]
    record = loader.get_cycle_record(name)
    frame = loader.load_cycle_measurements(name).copy()
    frame.timestamp = pd.to_datetime(frame.timestamp, format="mixed")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    heating_start = pd.Timestamp(record["boundaries"]["heating_start"])
    origin = pd.Timestamp(record["boundaries"]["start_time"])
    accounting = pd.Timestamp(group.heating_accounting_start.iloc[0])
    last = group.candidate_defrost_time.max()
    frequency = frame.loc[frame.timestamp.ge(accounting) & frame.timestamp.lt(last)]
    episodes = frequency_episodes(frequency)
    if not episodes:
        return [], dict(cycle_name=name, status="no_observed_frequency_change", n_episodes=0)
    # Published Dataset values are authoritative; only the diagnostic anchor is predicted.
    from defrost_event_models.event_electricity_ridge import predict_with_model_parameters

    anchors = pd.DatetimeIndex([e["start_time"] - pd.Timedelta(seconds=1) for e in episodes])
    features = extract_pre_defrost_features(frame, anchors, heating_start)
    anchor_cost = np.zeros(len(anchors))
    anchor_valid = np.ones(len(anchors), dtype=bool)
    for parameters in phase_models.values():
        prediction = predict_with_model_parameters(parameters, features).prediction.to_numpy()
        channels = parameters["feature_order"]
        anchor_valid &= (
            features[channels].notna().all(axis=1).to_numpy()
            & features[[c + "_valid_second_count" for c in channels]].ge(48).all(axis=1).to_numpy()
            & np.isfinite(prediction)
            & (prediction >= 0)
        )
        anchor_cost += prediction
    g = group.sort_values("candidate_defrost_time").copy().reset_index(drop=True)
    g["cycle_minutes"] = (g.candidate_defrost_time - origin).dt.total_seconds() / 60
    frame["cycle_minutes"] = (frame.timestamp - origin).dt.total_seconds() / 60
    # Instantaneous heat/power COP has no future event-cost term under either assumption.
    frame["instantaneous_cop"] = (frame.heating_capacity_processed / frame.power_total).where(
        frame.power_total.gt(0)
    )
    raw_columns = [
        "timestamp",
        "cycle_minutes",
        "compressor_frequency",
        "heating_capacity_processed",
        "power_total",
        "instantaneous_cop",
    ]
    frame.loc[frame.timestamp.between(accounting, last), raw_columns].to_csv(
        output / f"{name}_instantaneous.csv", index=False
    )
    saved = []
    sources = []
    for index, e in enumerate(episodes, 1):
        anchor = e["start_time"] - pd.Timedelta(seconds=1)
        e.update(
            cycle_name=name,
            experiment_id=group.experiment_id.iloc[0],
            episode=index,
            anchor_time=anchor,
        )
        available = bool(anchor_valid[index - 1])
        d0 = float(anchor_cost[index - 1]) if available else np.nan
        left = max(accounting, e["start_time"] - pd.Timedelta(minutes=2))
        right = min(e["block_end"], e["end_time"] + pd.Timedelta(minutes=2))
        if index < len(episodes):
            right = min(right, episodes[index]["start_time"] - pd.Timedelta(seconds=1))
        part = g.loc[g.candidate_defrost_time.between(left, right)].copy()
        fixed = part.copy()
        # Freeze only D; the measured Q/E, feature validity and integration rules stay shared.
        fixed["defrost_event_electricity_kwh"] = d0
        fixed["defrost_event_electricity_prediction_available"] = available
        fixed = calculate_cycle_cop(fixed, effective=True)
        part["cop_current"] = part.cycle_cop.where(part.cycle_cop_eligible)
        part["cop_fixed_cost"] = fixed.cycle_cop.where(fixed.cycle_cop_eligible)
        part["fixed_event_cost_kwh"] = d0
        part["episode"] = index
        suffix = "升频" if e["direction"] == "increase" else "降频"
        filename = (
            f"{name}_{index:02d}_{suffix}_{e['start_frequency_hz']:g}至{e['end_frequency_hz']:g}Hz"
        )
        e.update(
            status="plotted" if available else "plotted_missing_fixed_cost",
            figure=filename + ".png",
            fixed_event_cost_kwh=d0,
            valid_cop_points=int(part.cop_current.notna().sum()),
            maximum_absolute_cop_difference=float(
                (part.cop_current - part.cop_fixed_cost).abs().max()
            ),
        )
        raw = frame.loc[frame.timestamp.between(left, right)]
        e["figure"] = render_frequency_episode(part, raw, e, origin, output, filename)
        sources.append(
            part[
                [
                    "episode",
                    "candidate_defrost_time",
                    "cycle_minutes",
                    "compressor_frequency",
                    "pre_defrost_heat_kwh",
                    "pre_defrost_electricity_kwh",
                    "defrost_event_electricity_kwh",
                    "fixed_event_cost_kwh",
                    "cop_current",
                    "cop_fixed_cost",
                    "cycle_cop_uses_model_extrapolation",
                ]
            ]
        )
        saved.append(e)
    if sources:
        pd.concat(sources, ignore_index=True).to_csv(output / f"{name}_cop_curves.csv", index=False)
    return saved, dict(
        cycle_name=name,
        status="processed",
        n_episodes=len(episodes),
        n_figures=sum(e["status"].startswith("plotted") for e in saved),
    )


def frequency_episode_audit(args):
    from joblib import Parallel, delayed, parallel_config

    from dataset_tools import DatasetLoader
    from image_models.cycle_selection import screen_cv_cycles

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    loader = DatasetLoader(args.dataset)
    sources = {
        loader.get_cycle_record(name)["effective_cop_without_trunc"]["source_table"]
        for name in loader.list_cycles().cycle_name
    }
    if len(sources) != 1:
        raise ValueError("Dataset COP sources differ; select a consistent published source first")
    source = Path(sources.pop())
    args.decisions = source.parent
    decisions = pd.read_csv(source, low_memory=False)
    decisions.candidate_defrost_time = pd.to_datetime(
        decisions.candidate_defrost_time, format="mixed"
    )
    screen_cv_cycles(decisions, DatasetLoader(args.dataset).list_cycles(), exclude_rb=False).to_csv(
        out / "cycle_screening.csv", index=False
    )
    settings = json.loads((args.decisions / "run_settings.json").read_text())
    model = json.loads(Path(settings["model_file"]).read_text())
    assert settings["effective_heat_rule"] == "without_trunc"
    jobs = list(decisions.groupby("cycle_name", sort=True))
    if args.cycles:
        jobs = [(name, group) for name, group in jobs if name in args.cycles]
    (out / "configuration.json").write_text(
        json.dumps(
            dict(
                source=str(args.decisions),
                model=settings["model_file"],
                purpose="Per-cycle staircase frequency episodes with pre-onset minus 1s frozen future cost",
                segmentation="all nonzero observed changes; same-direction steps joined across any plateau; reverse direction or >30s data gap splits",
                scope="all cycles, stable heating before preparation; screen reported but not used to exclude cycles",
                instantaneous_cop="heating_capacity_processed / power_total; identical under both future-cost assumptions",
                curve_sampling="Dataset published candidate times; formal Q/E/D/COP copied without recomputation; single exact pre-onset anchor prediction only",
                x_axis="minutes since Dataset cycle start_time",
                n_jobs=args.n_jobs,
                extrapolation="annotation only; no exclusion",
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        results = Parallel(verbose=10)(
            delayed(plot_cycle_frequency_episodes)(
                g, args.dataset, model["phase_models"]["folds"][g.experiment_id.iloc[0]], out
            )
            for _, g in jobs
        )
    events = pd.DataFrame([e for episodes, _ in results for e in episodes])
    coverage = pd.DataFrame([status for _, status in results])
    # A bounded rerun replaces only the explicitly requested cycles in the shared tables.
    if args.cycles and (out / "cycles.csv").exists():
        previous = pd.read_csv(out / "cycles.csv")
        coverage = pd.concat(
            [previous.loc[~previous.cycle_name.isin(args.cycles)], coverage], ignore_index=True
        )
        previous = pd.read_csv(out / "episodes.csv")
        events = pd.concat(
            [previous.loc[~previous.cycle_name.isin(args.cycles)], events], ignore_index=True
        )
    events.to_csv(out / "episodes.csv", index=False)
    coverage.to_csv(out / "cycles.csv", index=False)
    consolidate_episode_csv(out, args.cycles)
    print(coverage.to_string(index=False), flush=True)


def render_saved_frequency_cycle(episodes, curves, raw, output):
    from plots.frequency import render_frequency_episode

    raw = raw.sort_values("timestamp").copy()
    raw.timestamp = pd.to_datetime(raw.timestamp, format="mixed")
    origin = (raw.timestamp.iloc[0] - pd.to_timedelta(raw.cycle_minutes.iloc[0], unit="min")).round(
        "s"
    )
    episodes = episodes.sort_values("episode").to_dict("records")
    for index, episode in enumerate(episodes):
        for key in ["start_time", "end_time", "anchor_time", "block_end"]:
            episode[key] = pd.Timestamp(episode[key])
        episode["episode"], episode["n_steps"] = int(episode["episode"]), int(episode["n_steps"])
        curve = curves.loc[curves.episode.eq(episode["episode"])].copy()
        curve["cycle_cop_uses_model_extrapolation"] = (
            curve.cycle_cop_uses_model_extrapolation.astype(str).str.lower().eq("true")
        )
        left = episode["start_time"] - pd.Timedelta(minutes=2)
        right = min(episode["block_end"], episode["end_time"] + pd.Timedelta(minutes=2))
        if index + 1 < len(episodes):
            right = min(
                right, pd.Timestamp(episodes[index + 1]["start_time"]) - pd.Timedelta(seconds=1)
            )
        render_frequency_episode(
            curve,
            raw.loc[raw.timestamp.between(left, right)],
            episode,
            origin,
            output,
            Path(episode["figure"]).stem,
        )
    return len(episodes)


def rerender_frequency_episodes(args):
    from joblib import Parallel, delayed, parallel_config

    # Read once; retain each record type's own fields rather than a wide sparse table.
    fields = dict(
        episode=[
            "cycle_name",
            "episode",
            "n_steps",
            "start_time",
            "end_time",
            "anchor_time",
            "block_end",
            "start_frequency_hz",
            "end_frequency_hz",
            "direction",
            "fixed_event_cost_kwh",
            "maximum_absolute_cop_difference",
            "figure",
        ],
        cop_curve=[
            "cycle_name",
            "episode",
            "cycle_minutes",
            "cop_current",
            "cop_fixed_cost",
            "defrost_event_electricity_kwh",
            "fixed_event_cost_kwh",
            "cycle_cop_uses_model_extrapolation",
            "pre_defrost_electricity_kwh",
            "pre_defrost_heat_kwh",
        ],
        instantaneous=[
            "cycle_name",
            "timestamp",
            "cycle_minutes",
            "compressor_frequency",
            "instantaneous_cop",
            "power_total",
            "heating_capacity_processed",
        ],
    )
    parts = {kind: [] for kind in fields}
    columns = set(["record_type"] + [column for values in fields.values() for column in values])
    for chunk in pd.read_csv(
        args.output / "全部循环调频数据.csv",
        usecols=lambda column: column in columns,
        chunksize=100000,
        low_memory=False,
    ):
        if args.cycles:
            chunk = chunk.loc[chunk.cycle_name.isin(args.cycles)]
        for kind, names in fields.items():
            part = chunk.loc[chunk.record_type.eq(kind), names]
            if not part.empty:
                parts[kind].append(part)
    tables = {
        kind: dict(tuple(pd.concat(frames, ignore_index=True).groupby("cycle_name")))
        for kind, frames in parts.items()
    }
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        counts = Parallel(verbose=10)(
            delayed(render_saved_frequency_cycle)(
                episodes, tables["cop_curve"][name], tables["instantaneous"][name], args.output
            )
            for name, episodes in tables["episode"].items()
        )
    print(f"Rendered {sum(counts)} figures from unchanged consolidated CSV.", flush=True)


def organize_frequency_figures(output):
    import csv
    from collections import Counter

    from plots.frequency import frequency_episode_folder

    source = output / "全部循环调频数据.csv"
    temporary = source.with_suffix(".csv.tmp")
    counts = Counter()
    with source.open(newline="") as reader_file, temporary.open("w", newline="") as writer_file:
        reader = csv.DictReader(reader_file)
        writer = csv.DictWriter(writer_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["record_type"] == "episode":
                folder = frequency_episode_folder(row)
                relative = Path(folder) / Path(row["figure"]).name
                old, new = output / row["figure"], output / relative
                if old != new:
                    new.parent.mkdir(parents=True, exist_ok=True)
                    if new.exists():
                        raise FileExistsError(new)
                    old.rename(new)
                assert new.is_file()
                row["figure"] = str(relative)
                counts[folder] += 1
            writer.writerow(row)
    temporary.replace(source)
    print(dict(counts), flush=True)


def consolidate_episode_csv(output, updated_cycles=None):
    """Combine losslessly; distinguish observation, curve, episode and cycle rows."""
    import csv
    import shutil
    from collections import Counter
    from datetime import datetime

    target = output / "全部循环调频数据.csv"
    files = sorted(p for p in output.glob("*.csv") if p != target)
    if not files:
        return
    kinds = {"cycles.csv": "cycle", "episodes.csv": "episode", "cycle_screening.csv": "screening"}

    def records():
        if target.exists() and updated_cycles:
            with target.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if (
                        row["record_type"] != "screening"
                        and row["cycle_name"] not in updated_cycles
                    ):
                        yield row
        for path in files:
            kind = kinds.get(
                path.name,
                "instantaneous" if path.name.endswith("_instantaneous.csv") else "cop_curve",
            )
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if "cycle_name" not in row:
                        row["cycle_name"] = path.name[: len("frost_cycle_000000")]
                    yield dict(record_type=kind, **row)

    columns = ["record_type", "cycle_name", "episode"]
    for path in ([target] if target.exists() and updated_cycles else []) + files:
        with path.open(newline="") as handle:
            for column in next(csv.reader(handle)):
                if column not in columns:
                    columns.append(column)
    temporary = output / "全部循环调频数据.csv.tmp"
    counts = Counter()
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in records():
            writer.writerow(row)
            counts[row["record_type"]] += 1
    # Check every retained value before archiving the source files.
    with temporary.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for expected in records():
            actual = next(reader)
            assert actual == {column: expected.get(column, "") for column in columns}
        assert next(reader, None) is None
    archive = (
        Path("output/99_历史归档")
        / output.name
        / ("分散CSV_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    archive.mkdir(parents=True, exist_ok=False)
    for path in files + ([target] if target.exists() else []):
        shutil.move(str(path), archive / path.name)
    temporary.rename(target)
    print(f"Consolidated CSV: {target}; record counts: {dict(counts)}", flush=True)


def frozen_replay(original, selected_index):
    """Score an independently selected action without moving it to the original basin."""
    supported = original.cycle_cop_eligible.eq(True)
    assert supported.loc[selected_index]
    optimum = original.loc[supported, "cycle_cop"].idxmax()
    best = original.loc[optimum, "cycle_cop"]
    selected = original.loc[selected_index, "cycle_cop"]
    return dict(
        frozen_cop_at_selected=selected,
        frozen_cop_loss=best - selected,
        frozen_regret_pct=100 * (1 - selected / best),
        frozen_hit_1pct=bool(selected >= 0.99 * best),
        reference_shift=original.loc[selected_index, "visual_frost_score"]
        - original.loc[optimum, "visual_frost_score"],
        trigger_shift_minutes=(
            original.loc[selected_index, "candidate_defrost_time"]
            - original.loc[optimum, "candidate_defrost_time"]
        ).total_seconds()
        / 60,
    )


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    actions = pd.read_csv(args.source / "action_points.csv")
    groups = pd.read_csv(args.groups)
    screening = pd.read_csv(args.source / "cycle_screening.csv")
    decisions = pd.read_csv(args.decisions / "candidate_decisions.csv", low_memory=False)
    keys = ["cycle_name", "candidate_defrost_time"]
    d = actions[keys + ["visual_frost_score"]].merge(decisions, on=keys, validate="one_to_one")
    d = d.merge(groups[["cycle_name", "Ts"]], on="cycle_name", validate="many_to_one")
    d.candidate_defrost_time = pd.to_datetime(d.candidate_defrost_time)
    assert d.defrost_event_net_heat_kwh.eq(0).all()
    np.testing.assert_allclose(
        d.cycle_cop,
        d.pre_defrost_heat_kwh / (d.pre_defrost_electricity_kwh + d.defrost_event_electricity_kwh),
        atol=1e-12,
    )
    models = read_models(args.model)
    curves = []
    records = []
    for name, g in d.groupby("cycle_name", sort=True):
        g = decompose(g)
        supported = g.cycle_cop_eligible.eq(True)
        p = parameters(models, g.experiment_id.iloc[0])
        contribution = []
        for f, b, s in zip(p["feature_order"], p["coefficients"], p["scaler_scale"], strict=True):
            col = "cost_delta_" + f
            g[col] = (g[f].shift(-1) - g[f]) * b / s
            contribution.append(col)
        np.testing.assert_allclose(
            g.loc[g.supported_pair, contribution].sum(axis=1),
            g.loc[g.supported_pair, "dD"],
            atol=1e-10,
        )
        valid = g.loc[g.supported_pair & g.p_kw.gt(0)]
        p_typical = valid.p_kw.median()
        base = valid.head(10).instant_cop.median()
        g["relative_instant_cop"] = g.instant_cop / base
        g["cost_equivalent_minutes"] = g.defrost_event_electricity_kwh / p_typical * 60
        g["cost_fraction"] = g.defrost_event_electricity_kwh / (
            g.pre_defrost_electricity_kwh + g.defrost_event_electricity_kwh
        )
        fixed = g.copy()
        fixed["defrost_event_electricity_kwh"] = g.loc[
            supported, "defrost_event_electricity_kwh"
        ].median()
        fixed = calculate_cycle_cop(fixed, effective=True)
        assert fixed.cycle_cop_eligible.equals(g.cycle_cop_eligible)
        for variant, c in [("frozen", g), ("constant_cost", fixed)]:
            c = add_single_objective_optima(c, names=("cycle_cop",))
            k = c.loc[supported, "cycle_cop"].idxmax()
            c["near_optimal"] = supported & c.cycle_cop.ge(c.loc[k, "cycle_cop"] * 0.99)
            intervals = crossing_intervals(c, True)
            rec = dict(
                cycle_name=name,
                Ts=g.Ts.iloc[0],
                experiment_id=g.experiment_id.iloc[0],
                variant=variant,
                rho_star=g.loc[k, "visual_frost_score"],
                cop_star=c.loc[k, "cycle_cop"],
                near_reference_width=sum(b - a for a, b in intervals),
                peak_at_support_edge=(
                    k == 0 or k == len(g) - 1 or not supported.iloc[max(0, k - 1) : k + 2].all()
                ),
                basin_minutes=c.loc[k, "cycle_cop_basin_1pct_width_minutes"],
                peak_minutes=g.loc[k, "heating_elapsed_minutes"],
                p_typical_kw=p_typical,
                baseline_cop=base,
                ambient_temperature=g.loc[supported, "ambient_temperature"].median(),
                predicted_cost_kwh=g.loc[supported, "defrost_event_electricity_kwh"].median(),
                equivalent_minutes=g.loc[supported, "cost_equivalent_minutes"].median(),
            )
            records.append(rec)
            rec.update(frozen_replay(g, k))
            rec["selected_time"] = g.loc[k, "candidate_defrost_time"]
        curves.append(g)
    allpoints = pd.concat(curves, ignore_index=True)
    pairs = allpoints.loc[allpoints.supported_pair & allpoints.visual_frost_score.notna()].copy()
    pairs["rho_band"] = pd.cut(
        pairs.visual_frost_score,
        [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        include_lowest=True,
    ).astype(str)
    metrics = [
        "heat_margin_kw",
        "cost_margin_kw",
        "net_margin_kw",
        "q_kw",
        "p_kw",
        "instant_cop",
        "relative_instant_cop",
        "cost_fraction",
        "cost_equivalent_minutes",
        "ambient_temperature",
        "compressor_frequency",
        "evaporating_pressure",
        "water_out_temperature",
        "heating_elapsed_minutes",
    ]
    pairs["cop_rising"] = pairs.net_margin_kw.gt(0)
    band = (
        pairs.groupby(["cycle_name", "experiment_id", "Ts", "rho_band"], observed=True)[
            metrics + ["cop_rising"]
        ]
        .mean()
        .reset_index()
    )
    band.to_csv(out / "cycle_reference_bands.csv", index=False)
    bands = (
        band.groupby(["Ts", "rho_band"])
        .agg(
            n_cycles=("cycle_name", "size"),
            n_experiments=("experiment_id", "nunique"),
            **{c: (c, "median") for c in metrics + ["cop_rising"]},
        )
        .reset_index()
    )
    bands.to_csv(out / "temperature_reference_bands.csv", index=False)
    # Means preserve the additive energy identity; medians above describe typical cycles.
    band.groupby(["Ts", "rho_band"])[metrics].mean().to_csv(
        out / "temperature_reference_band_means.csv"
    )
    band.groupby(["Ts", "experiment_id", "rho_band"])[metrics].mean().to_csv(
        out / "experiment_reference_bands.csv"
    )
    cost_columns = [c for c in pairs if c.startswith("cost_delta_")]
    parts = pairs.groupby(["cycle_name", "Ts", "rho_band"])[cost_columns].mean().reset_index()
    parts.groupby(["Ts", "rho_band"])[cost_columns].mean().to_csv(
        out / "ridge_cost_contributions.csv"
    )
    pairs[
        keys
        + [
            "Ts",
            "visual_frost_score",
            "dQ",
            "dE",
            "dD",
            "heat_margin_kw",
            "cost_margin_kw",
            "net_margin_kw",
        ]
        + [c for c in pairs if c.startswith("cost_delta_")]
    ].to_csv(out / "adjacent_energy_balance.csv", index=False)
    cycle = pd.DataFrame(records).merge(
        screening[["cycle_name", "core_overlap", "cop_right_censored", "ridge_coverage"]],
        on="cycle_name",
    )
    cycle.to_csv(out / "cycle_diagnostics.csv", index=False)
    replay = cycle.loc[cycle.variant.eq("constant_cost")].copy()
    replay["abs_reference_shift"] = replay.reference_shift.abs()
    replay["abs_trigger_shift_minutes"] = replay.trigger_shift_minutes.abs()
    replay.to_csv(out / "constant_cost_frozen_replay.csv", index=False)
    replay_summaries = []
    for scope, frame in [
        ("all", replay),
        ("core", replay.loc[replay.core_overlap]),
        ("reference_shift_ge_0.1", replay.loc[replay.abs_reference_shift.ge(0.1)]),
    ]:
        for label, part in [("all", frame), *list(frame.groupby("Ts"))]:
            replay_summaries.append(
                dict(
                    scope=scope,
                    Ts=label,
                    n_cycles=len(part),
                    n_experiments=part.experiment_id.nunique(),
                    hits_1pct=int(part.frozen_hit_1pct.sum()),
                    hit_rate=part.frozen_hit_1pct.mean(),
                    regret_mean_pct=part.frozen_regret_pct.mean(),
                    regret_median_pct=part.frozen_regret_pct.median(),
                    regret_p90_pct=part.frozen_regret_pct.quantile(0.9),
                    regret_max_pct=part.frozen_regret_pct.max(),
                    cop_loss_mean=part.frozen_cop_loss.mean(),
                    abs_reference_shift_median=part.abs_reference_shift.median(),
                    abs_trigger_shift_minutes_median=part.abs_trigger_shift_minutes.median(),
                )
            )
    pd.DataFrame(replay_summaries).to_csv(out / "constant_cost_replay_summary.csv", index=False)
    numeric = [
        "rho_star",
        "near_reference_width",
        "basin_minutes",
        "peak_minutes",
        "baseline_cop",
        "ambient_temperature",
        "predicted_cost_kwh",
        "equivalent_minutes",
        "peak_at_support_edge",
        "cop_right_censored",
    ]
    summaries = []
    for scope, frame in [("all", cycle), ("core", cycle.loc[cycle.core_overlap])]:
        s = (
            frame.groupby(["Ts", "variant"])
            .agg(
                n_cycles=("cycle_name", "size"),
                n_experiments=("experiment_id", "nunique"),
                **{
                    c: (
                        c,
                        "median"
                        if c not in ["peak_at_support_edge", "cop_right_censored"]
                        else "sum",
                    )
                    for c in numeric
                },
            )
            .reset_index()
        )
        s["scope"] = scope
        summaries.append(s)
    pd.concat(summaries).to_csv(out / "temperature_summary.csv", index=False)
    groups.groupby(["experiment_id", "Ts"]).size().rename("n_cycles").to_csv(
        out / "experiment_temperature_counts.csv"
    )
    events = pd.read_csv(args.model / "defrost_events.csv")
    v = pd.read_csv(args.model / "model_validation.csv")
    v = v.loc[v.model_name.eq("ridge_dynamic_state_8")]
    ev = events.loc[events.energy_event_valid.eq(True)].merge(
        v[
            [
                "cycle_name",
                "defrost_event_electricity_prediction_kwh",
                "event_electricity_in_training_domain",
            ]
        ],
        on="cycle_name",
        validate="one_to_one",
    )
    ev = ev.merge(
        cycle.loc[cycle.variant.eq("frozen"), ["cycle_name", "p_typical_kw"]],
        on="cycle_name",
        how="left",
    )
    ev["Ts"] = ev.water_temperature_setpoint
    ev["bias_kwh"] = (
        ev.defrost_event_electricity_prediction_kwh - ev.defrost_event_electricity_observed_kwh
    )
    ev["abs_error_kwh"] = ev.bias_kwh.abs()
    ev["equivalent_minutes"] = ev.defrost_event_electricity_observed_kwh / ev.p_typical_kw * 60
    ev["in_visual_pool"] = ev.cycle_name.isin(groups.cycle_name)
    ev.to_csv(out / "observed_events.csv", index=False)
    em = [
        "E_prep_kwh",
        "E_D_kwh",
        "E_R_kwh",
        "defrost_event_electricity_observed_kwh",
        "equivalent_minutes",
    ]
    es = []
    for scope, frame in [("all_training_events", ev), ("visual_pool", ev.loc[ev.in_visual_pool])]:
        s = (
            frame.groupby("Ts")
            .agg(
                n_events=("cycle_name", "size"),
                n_experiments=("experiment_id", "nunique"),
                bias_kwh=("bias_kwh", "mean"),
                mae_kwh=("abs_error_kwh", "mean"),
                n_equivalent_minutes=("equivalent_minutes", "count"),
                **{c: (c, "median") for c in em},
            )
            .reset_index()
        )
        s["scope"] = scope
        es.append(s)
    pd.concat(es).to_csv(out / "event_temperature_summary.csv", index=False)
    print(pd.concat(summaries).to_string(index=False))
    print(pd.concat(es).to_string(index=False))


def pre_frequency_cycle(name, dataset, frost):
    """Describe pre-onset windows without using COP to choose a frequency episode."""
    from threadpoolctl import threadpool_limits

    from dataset_tools import DatasetLoader

    with threadpool_limits(limits=1):
        f = DatasetLoader(dataset).load_cycle_measurements(name)
    f = f.loc[f.cycle_stage.eq("frost_development")].sort_values("timestamp").copy()
    f["instantaneous_cop"] = f.heating_capacity_processed / f.power_total.where(f.power_total.gt(0))
    f["ambient_evaporating_difference"] = f.ambient_temperature - f.evaporating_temperature
    fields = [
        "instantaneous_cop",
        "heating_capacity_processed",
        "power_total",
        "evaporating_pressure_processed",
        "ambient_evaporating_difference",
        "compressor_frequency",
        "ambient_temperature",
        "water_temperature_setpoint",
        "water_in_temperature",
        "water_out_temperature",
    ]
    records = []
    for episode in frequency_episodes(f):
        if (
            episode["direction"] != "increase"
            or episode["end_frequency_hz"] - episode["start_frequency_hz"] <= 5
        ):
            continue
        t = episode["start_time"]
        windows = [
            f.loc[
                f.timestamp.ge(t + pd.Timedelta(minutes=a))
                & f.timestamp.lt(t + pd.Timedelta(minutes=b))
            ]
            for a, b in [(-10, -5), (-5, 0), (0, 5)]
        ]
        # 80% temporal coverage for 1-second Dataset observations, separately per quantity.
        complete = all(
            len(w) >= 240
            and w[fields].notna().sum().ge(240).all()
            and w.timestamp.diff().dt.total_seconds().max() <= 30
            for w in windows[:2]
        )
        if not complete:
            continue
        pre = pd.concat(windows[:2])
        r = dict(
            cycle_name=name,
            experiment_id=f.experiment_id.iloc[0],
            **episode,
            pre_frequency_span=pre.compressor_frequency.max() - pre.compressor_frequency.min(),
            pre_ambient_span=pre.ambient_temperature.max() - pre.ambient_temperature.min(),
            pre_setpoint_span=pre.water_temperature_setpoint.max()
            - pre.water_temperature_setpoint.min(),
        )
        r["stable_pre_controls"] = (
            r["pre_frequency_span"] <= 1
            and r["pre_ambient_span"] <= 0.5
            and r["pre_setpoint_span"] == 0
        )
        for label, w in zip(["early", "late", "after"], windows):
            for field in fields:
                r[f"{label}_{field}"] = w[field].median() if w[field].count() >= 240 else np.nan
            r[f"{label}_interpolated_fraction"] = w.is_interpolated.mean()
            a, b = {"early": (-10, -5), "late": (-5, 0), "after": (0, 5)}[label]
            images = frost.loc[
                frost.image_time.ge(t + pd.Timedelta(minutes=a))
                & frost.image_time.lt(t + pd.Timedelta(minutes=b))
            ]
            r[f"{label}_frost_frames"] = len(images)
            r[f"{label}_visual_frost_score"] = (
                images.visual_frost_score.median() if len(images) >= 3 else np.nan
            )
        for field in fields + ["visual_frost_score"]:
            r[f"pre_change_{field}"] = r[f"late_{field}"] - r[f"early_{field}"]
        r["pre_cop_change_pct"] = (
            100 * r["pre_change_instantaneous_cop"] / r["early_instantaneous_cop"]
        )
        records.append(r)
    tail = f.loc[f.timestamp.ge(f.timestamp.max() - pd.Timedelta(minutes=5))]
    terminal = dict(
        cycle_name=name,
        final_frequency=tail.compressor_frequency.median(),
        final_frequency_span=tail.compressor_frequency.max() - tail.compressor_frequency.min(),
        observed_maximum_frequency=f.compressor_frequency.max(),
    )
    return records, terminal


def pre_frequency_evidence(args):
    """Sequential evidence for deterioration, control response and decision timing."""
    from joblib import Parallel, delayed, parallel_config

    from dataset_tools import DatasetLoader
    from image_models.cycle_selection import screen_published_cycles

    source = Path("output/99_历史归档/legacy_test/slingshot_harmony_20261001")
    assert (
        json.loads((source / "run_settings.json").read_text())["coordinate_correction"] == "harmony"
    )
    screening = screen_published_cycles(args.dataset)
    frames = pd.read_csv(source / "progression.csv")
    frames.image_time = pd.to_datetime(frames.image_time)
    frost = dict(tuple(frames.groupby("cycle_name")))
    names = DatasetLoader(args.dataset).list_cycles().cycle_name.tolist()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(pre_frequency_cycle)(name, args.dataset, frost.get(name, frames.iloc[:0]))
            for name in names
        )
    episodes = pd.DataFrame([r for records, _ in results for r in records]).sort_values(
        ["cycle_name", "start_time"]
    )
    # One earliest eligible onset per cycle; outcome-independent selection.
    candidates = pd.read_parquet(args.dataset / "training_data/candidate_decisions.parquet")
    candidates.candidate_defrost_time = pd.to_datetime(candidates.candidate_defrost_time)
    candidates = candidates.loc[candidates.cycle_cop_eligible.eq(True)]
    curves = dict(tuple(candidates.groupby("cycle_name")))
    for index, row in episodes.iterrows():
        curve = curves.get(row.cycle_name)
        if curve is None:
            continue
        peaks = pd.to_datetime(curve.cycle_cop_t_star.dropna()).unique()
        if len(peaks) != 1:
            continue
        peak = pd.Timestamp(peaks[0])
        episodes.loc[index, "saved_peak_time"] = peak
        side = "left" if row.start_time < peak else "right" if row.start_time > peak else "at_peak"
        episodes.loc[index, "onset_side"] = side
        episodes.loc[index, "episode_region"] = (
            "left" if row.end_time < peak else "right" if row.start_time > peak else "crosses_peak"
        )
        episodes.loc[index, "pre_window_region"] = (
            "left"
            if row.start_time <= peak
            else "right"
            if row.start_time - pd.Timedelta(minutes=10) >= peak
            else "crosses_peak"
        )
        episodes.loc[index, "five_minute_region"] = (
            "left"
            if row.start_time + pd.Timedelta(minutes=5) < peak
            else "right"
            if row.start_time > peak
            else "crosses_peak"
        )
        for label, target in [
            ("onset", row.start_time),
            ("five_minutes_later", row.start_time + pd.Timedelta(minutes=5)),
        ]:
            prior = (
                curve.loc[curve.candidate_defrost_time.le(target)]
                .sort_values("candidate_defrost_time")
                .tail(1)
            )
            if len(prior) and (target - prior.candidate_defrost_time.iloc[0]).total_seconds() <= 30:
                episodes.loc[index, f"{label}_saved_cycle_cop"] = prior.cycle_cop.iloc[0]
                episodes.loc[index, f"{label}_extrapolated"] = (
                    prior.cycle_cop_uses_model_extrapolation.iloc[0]
                )
    episodes["saved_cycle_cop_change_5min"] = (
        episodes.five_minutes_later_saved_cycle_cop - episodes.onset_saved_cycle_cop
    )
    episodes["selected_cv"] = episodes.cycle_name.isin(
        screening.loc[screening.selected, "cycle_name"]
    )
    first = episodes.drop_duplicates("cycle_name").copy()
    # Keep left/right independently; the first overall onset would hide later right-side episodes.
    side_first = episodes.loc[
        episodes.onset_side.isin(["left", "right"])
        & episodes.pre_window_region.eq(episodes.onset_side)
    ].drop_duplicates(["cycle_name", "onset_side"])
    side_first.to_csv(out / "first_onsets_per_peak_side.csv", index=False)
    examples = (
        side_first.loc[
            side_first.stable_pre_controls
            & side_first.pre_change_visual_frost_score.gt(0)
            & side_first.pre_cop_change_pct.lt(-1)
        ]
        .groupby("onset_side", sort=False)
        .head(1)
    )
    if len(examples):
        example_data = []
        for example in examples.itertuples():
            data = DatasetLoader(args.dataset).load_cycle_measurements(example.cycle_name)
            start, end = (
                example.start_time - pd.Timedelta(minutes=10),
                example.start_time + pd.Timedelta(minutes=5),
            )
            data = data.loc[
                data.timestamp.between(start, end),
                [
                    "timestamp",
                    "compressor_frequency",
                    "heating_capacity_processed",
                    "power_total",
                    "evaporating_pressure_processed",
                ],
            ].copy()
            data["instantaneous_cop"] = data.heating_capacity_processed / data.power_total.where(
                data.power_total.gt(0)
            )
            grade = frost[example.cycle_name]
            grade = grade.loc[
                grade.image_time.between(start, end), ["image_time", "visual_frost_score"]
            ].rename(columns={"image_time": "timestamp"})
            cop = curves[example.cycle_name]
            cop = cop.loc[
                cop.candidate_defrost_time.between(start, end),
                ["candidate_defrost_time", "cycle_cop"],
            ].rename(columns={"candidate_defrost_time": "timestamp"})
            data = (
                data.merge(grade, on="timestamp", how="outer")
                .merge(cop, on="timestamp", how="outer")
                .sort_values("timestamp")
            )
            data["minutes_from_onset"] = (
                data.timestamp - example.start_time
            ).dt.total_seconds() / 60
            example_data.append(
                data.assign(cycle_name=example.cycle_name, onset_side=example.onset_side)
            )
        pd.concat(example_data).to_csv(out / "example_curve.csv", index=False)
        examples.to_csv(out / "example_selection.csv", index=False)
        from plots.frequency import render_pre_frequency_evidence

        render_pre_frequency_evidence(out)
    episodes.to_csv(out / "all_onsets.csv", index=False)
    first.to_csv(out / "first_onsets.csv", index=False)
    terminal = screening.merge(
        pd.DataFrame([r for _, r in results]), on="cycle_name", validate="one_to_one"
    )
    terminal.to_csv(out / "cycle_screening_and_terminal_frequency.csv", index=False)
    summaries = []
    scopes = [
        ("all_first_onsets", first),
        ("stable_pre_controls", first.loc[first.stable_pre_controls]),
    ]
    for side in ["left", "right"]:
        g = side_first.loc[side_first.onset_side.eq(side)]
        scopes.extend(
            [(side + "_all", g), (side + "_stable_pre_controls", g.loc[g.stable_pre_controls])]
        )
    for scope, g in scopes:
        for field in [
            "instantaneous_cop",
            "heating_capacity_processed",
            "power_total",
            "evaporating_pressure_processed",
            "ambient_evaporating_difference",
            "visual_frost_score",
            "saved_cycle_cop_change_5min",
        ]:
            key = field if field == "saved_cycle_cop_change_5min" else f"pre_change_{field}"
            eligible = (
                g.loc[g.five_minute_region.eq(g.onset_side)]
                if field == "saved_cycle_cop_change_5min"
                else g
            )
            v = eligible[key].dropna()
            summaries.append(
                dict(
                    scope=scope,
                    quantity=field,
                    cycles=len(v),
                    experiments=eligible.loc[eligible[key].notna(), "experiment_id"].nunique(),
                    median_change=v.median(),
                    decreases=int(v.lt(0).sum()),
                    increases=int(v.gt(0).sum()),
                )
            )
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    (out / "settings.json").write_text(
        json.dumps(
            dict(
                dataset=str(args.dataset),
                frost_source=str(source),
                coordinate_correction="harmony",
                windows_minutes=[[-10, -5], [-5, 0], [0, 5]],
                selection="first >5Hz same-direction rise with >=240 finite seconds in each pre-window; no COP outcome selection",
                stable_controls="pre10min frequency range <=1Hz, ambient range <=0.5C, constant water setpoint",
                peak="saved cycle_cop_t_star; descriptive retrospective boundary, unavailable prospectively",
                side_selection="first eligible onset per cycle per side with entire pre10min on that side; 5min comparisons crossing peak excluded",
                limitation="observational; unmeasured control/fan effects remain; no causal or optimal-defrost inference",
            ),
            indent=2,
        )
    )
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--groups", type=Path)
    p.add_argument("--frequency-audit", action="store_true")
    p.add_argument("--pre-frequency-evidence", action="store_true")
    p.add_argument("--synchronous-frequency", action="store_true")
    p.add_argument("--frequency-episodes", action="store_true")
    p.add_argument("--consolidate-episode-csv", action="store_true")
    p.add_argument("--rerender-frequency-episodes", action="store_true")
    p.add_argument("--organize-frequency-figures", action="store_true")
    p.add_argument("--n-jobs", type=int, default=6)
    p.add_argument("--cycles", nargs="+")
    p.add_argument("--dataset", type=Path, default=Path("dataset"))
    p.add_argument("--decisions", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.pre_frequency_evidence:
        pre_frequency_evidence(args)
    elif args.organize_frequency_figures:
        organize_frequency_figures(args.output)
    elif args.rerender_frequency_episodes:
        rerender_frequency_episodes(args)
    elif args.consolidate_episode_csv:
        consolidate_episode_csv(args.output)
    elif args.frequency_episodes:
        frequency_episode_audit(args)
    elif args.synchronous_frequency:
        synchronous_frequency_audit(args)
    elif args.frequency_audit:
        frequency_audit(args)
    else:
        if args.source is None or args.groups is None:
            p.error("--source and --groups are required for the temperature audit")
        run(args)
