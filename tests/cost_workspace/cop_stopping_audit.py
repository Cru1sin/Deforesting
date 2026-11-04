"""Audit marginal COP, native frequency rises and same-frequency temperature changes.

Uses the published cohort recipe on ALL saved without_trunc decisions first.
No diagnostic here changes the COP selector or claims frost causality.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from dataset_tools import DatasetLoader
from dataset_tools.builder.channel_mapping import CHANNELS
from defrost_decision.performance_objectives import calculate_cycle_cop
from image_models.cycle_selection import screen_cv_cycles
from output_layout import prepare_output
from tests.cost_workspace.cop_slope import frequency_rise_events, visible_frequency_rises
from tests.cost_workspace.temperature_energy_audit import decompose


def downward_crossings(points, column):
    """Only adjacent supported windows; never bridge gaps or unsupported values."""
    g = points.sort_values("candidate_defrost_time")
    y = g[column]
    valid = g.supported_pair & np.isfinite(y)
    continuous = g.candidate_defrost_time.diff().dt.total_seconds().eq(10)
    hit = valid & valid.shift(fill_value=False) & continuous & y.le(0) & y.shift().gt(0)
    left, right = g.candidate_defrost_time.shift(), g.candidate_defrost_time
    fraction = y.shift() / (y.shift() - y)
    return (left + (right - left) * fraction).loc[hit]


def marginal_points(curve, seconds):
    # Reuse exact cumulative-energy differences, including saved candidate cost changes.
    g = decompose(curve, steps=seconds // 10, step_seconds=10)
    g["window_seconds"] = seconds
    g["fixed_cost_margin"] = g.instant_cop - g.cycle_cop
    denominator = g.dE + g.dD
    g["corrected_marginal_cop"] = g.dQ / denominator.where(denominator.gt(0))
    g["corrected_margin"] = g.corrected_marginal_cop - g.cycle_cop
    g["event_cost_rate_kw"] = g.dD / g.interval_hours
    return g


def temperature_trace(loader, name, curve):
    channels = [
        "evaporating_temperature",
        "compressor_frequency",
        "ambient_temperature",
        "environment_temperature",
        "fan_speed",
        "water_in_temperature",
        "water_out_temperature",
    ]
    mapping = {c: CHANNELS[c]["source_names"][0] for c in channels}
    # Native observations only; interpolated temperatures cannot establish a temporal shift.
    raw = loader.load_cycle_original(name, columns=["timestamp", *mapping.values()])
    raw = raw.rename(columns={v: k for k, v in mapping.items()})
    raw.timestamp = pd.to_datetime(raw.timestamp)
    start = pd.Timestamp(curve.heating_accounting_start.iloc[0])
    end = curve.candidate_defrost_time.max()
    raw = raw.loc[raw.timestamp.between(start, end)].sort_values("timestamp")
    trace = raw.set_index("timestamp").resample("30s").median(numeric_only=True)
    trace["native_count"] = (
        raw.set_index("timestamp").evaporating_temperature.resample("30s").count()
    )
    trace = trace.loc[trace.native_count.ge(15)].reset_index()
    trace["progress"] = (trace.timestamp - start).dt.total_seconds() / (end - start).total_seconds()
    trace["mean_water_temperature"] = (trace.water_in_temperature + trace.water_out_temperature) / 2
    trace["te_minus_ambient"] = trace.evaporating_temperature - trace.ambient_temperature
    trace["frequency_bin_hz"] = trace.compressor_frequency.round()
    trace["cycle_name"] = name
    pairs = []
    for hz, group in trace.groupby("frequency_bin_hz"):
        early, late = group.loc[group.progress.le(1 / 3)], group.loc[group.progress.ge(2 / 3)]
        if len(early) < 2 or len(late) < 2:
            continue
        record = dict(cycle_name=name, frequency_hz=hz, early_bins=len(early), late_bins=len(late))
        for c in channels + ["mean_water_temperature", "te_minus_ambient"]:
            record[c + "_early"] = early[c].median()
            record[c + "_late"] = late[c].median()
            record[c + "_change"] = late[c].median() - early[c].median()
        record["matched_context"] = (
            abs(record["ambient_temperature_change"]) <= 0.5
            and abs(record["mean_water_temperature_change"]) <= 1.5
            and abs(record["fan_speed_change"]) <= 5
        )
        pairs.append(record)
    events = frequency_rise_events(
        raw, curve, frequency_column="compressor_frequency", merge_seconds=np.inf
    )
    events["cycle_name"] = name
    first = events.sort_values("rise_start").head(1).copy()
    first["definition"] = "first_native_positive_change_after_accounting_start"
    displayed = visible_frequency_rises(events).copy()
    displayed["definition"] = "first_displayed_stage_ge3Hz_supported_COP_ge60pct_peak"
    rises = pd.concat([first, displayed], ignore_index=True)
    rises["lag_to_peak_minutes"] = (rises.peak_time - rises.rise_start).dt.total_seconds() / 60
    rises["cop_rises_after_onset"] = rises.peak_time.gt(rises.rise_start)
    return trace, pd.DataFrame(pairs), rises


def main(args):
    prepare_output(args)
    if args.output.exists():
        raise FileExistsError(args.output)
    loader = DatasetLoader(args.dataset)
    settings = json.loads((args.dataset / "training_data/settings.json").read_text())
    full = pd.read_csv(settings["decisions"], low_memory=False)
    quality = pd.read_csv(settings["quality"])
    recipe = {
        k: settings[k]
        for k in [
            "exclude_rb",
            "input_rule",
            "cohort_rule",
            "minimum_terminal_decline",
            "rb_minimum_terminal_decline",
            "excluded_cycles",
            "maximum_per_setpoint",
            "post_near_minutes_by_setpoint",
        ]
    }
    screening = screen_cv_cycles(
        full,
        loader.list_cycles(),
        quality=quality,
        maximum_interpolated_fraction=settings["limits"]["maximum_interpolated_fraction"],
        **recipe,
    )
    cohort = screening.loc[screening.selected & screening.setpoint_c.eq(args.setpoint)].copy()
    values = calculate_cycle_cop(full.loc[full.cycle_name.isin(cohort.cycle_name)], effective=True)
    values.candidate_defrost_time = pd.to_datetime(values.candidate_defrost_time, format="mixed")
    assert values.defrost_event_net_heat_kwh.eq(0).all(), "Decomposition requires zero event heat"
    points, checks = [], []
    for name, curve in values.groupby("cycle_name"):
        curve = curve.sort_values("candidate_defrost_time").reset_index(drop=True)
        valid = curve.cycle_cop_eligible & np.isfinite(curve.cycle_cop)
        peak = curve.loc[valid].loc[curve.loc[valid, "cycle_cop"].idxmax()]
        for seconds in args.window_seconds:
            g = marginal_points(curve, seconds)
            g["peak_time"] = peak.candidate_defrost_time
            points.append(g)
            for kind in ["fixed_cost_margin", "corrected_margin"]:
                crossings = downward_crossings(g, kind)
                offsets = (crossings - peak.candidate_defrost_time).dt.total_seconds() / 60
                nearest = offsets.iloc[np.argmin(abs(offsets))] if len(offsets) else np.nan
                row = g.loc[(g.candidate_defrost_time - peak.candidate_defrost_time).abs().idxmin()]
                checks.append(
                    dict(
                        cycle_name=name,
                        experiment_id=peak.experiment_id,
                        window_seconds=seconds,
                        criterion=kind,
                        peak_time=peak.candidate_defrost_time,
                        peak_cop=peak.cycle_cop,
                        downward_crossings=len(crossings),
                        nearest_crossing_minus_peak_minutes=nearest,
                        instant_cop_at_peak=row.instant_cop,
                        corrected_marginal_cop_at_peak=row.corrected_marginal_cop,
                        event_cost_rate_at_peak_kw=row.event_cost_rate_kw,
                    )
                )
    print(f"{len(full)} full candidates -> {len(cohort)} cycles at {args.setpoint:g} C", flush=True)
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        measured = Parallel(verbose=5)(
            delayed(temperature_trace)(DatasetLoader(args.dataset), name, g)
            for name, g in values.groupby("cycle_name")
        )
    args.output.mkdir(parents=True)
    screening.to_csv(args.output / "cycle_screening.csv", index=False)
    cohort.to_csv(args.output / "cohort.csv", index=False)
    pd.concat(points).to_parquet(args.output / "marginal_cop_windows.parquet", index=False)
    checks = pd.DataFrame(checks)
    checks.to_csv(args.output / "crossing_peak_checks.csv", index=False)
    traces = pd.concat([t for t, _, _ in measured], ignore_index=True)
    pairs = pd.concat([p for _, p, _ in measured], ignore_index=True)
    rises = pd.concat([r for _, _, r in measured], ignore_index=True)
    traces.to_csv(args.output / "temperature_frequency_source.csv", index=False)
    pairs.to_csv(args.output / "same_frequency_early_late.csv", index=False)
    rises.to_csv(args.output / "first_frequency_rises.csv", index=False)
    metadata = dict(
        settings=settings,
        setpoint_c=args.setpoint,
        windows_seconds=args.window_seconds,
        definition="eta=Q/(E+D); Q starts at stable heating, saved cross-fitted D includes preparation+defrost electricity; event net heat=0",
        criterion="finite interval: deltaQ/(deltaE+deltaD)=eta_left; constant-cost comparison: deltaQ/deltaE=eta_left",
        crossing="downward sign crossings on supported overlapping windows, nearest to unchanged global peak; timestamps are window LEFT edges",
        temperature="native 30s medians with >=15 samples; first/last thirds, same rounded Hz; matched context <=0.5C ambient, <=1.5C mean water, <=5rpm fan; descriptive only",
    )
    (args.output / "settings.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    from plots.frequency import render_cop_stopping_audit

    render_cop_stopping_audit(args.output)
    print(
        checks.groupby(["window_seconds", "criterion"])
        .nearest_crossing_minus_peak_minutes.agg(["count", "median", "min", "max"])
        .to_string()
    )
    print(
        rises.groupby("definition")
        .lag_to_peak_minutes.agg(["count", "median", "min", "max"])
        .to_string()
    )
    print(
        f"Same-frequency groups: {len(pairs)}, cycles: {pairs.cycle_name.nunique()}, matched groups: {pairs.matched_context.sum()}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setpoint", type=float, default=50)
    parser.add_argument("--window-seconds", type=int, nargs="+", default=[10, 60, 120])
    parser.add_argument("--n-jobs", type=int, default=6)
    args = parser.parse_args()
    if any(s <= 0 or s % 10 for s in args.window_seconds):
        parser.error("Windows must be positive multiples of 10 s")
    main(args)
