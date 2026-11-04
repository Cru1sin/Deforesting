"""Describe COP concavity on the current published cohort, without selecting actions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_tools import DatasetLoader
from defrost_decision.performance_objectives import calculate_cycle_cop
from image_models.cycle_selection import screen_cv_cycles
from output_layout import prepare_output


def slope_windows(curve, window_minutes):
    """Full, nonoverlapping secants; never bridge an unsupported candidate interval."""
    g = curve.sort_values("candidate_defrost_time").copy()
    origin = pd.Timestamp(g.heating_accounting_start.iloc[0])
    t = (g.candidate_defrost_time - origin).dt.total_seconds().to_numpy() / 60
    valid = g.cycle_cop_eligible.eq(True).to_numpy() & np.isfinite(g.cycle_cop)
    breaks = ~valid | np.r_[True, ~valid[:-1] | (np.diff(t) > 0.502)]
    records = []
    peak = g.loc[valid, "cycle_cop"].max()
    for block, part in g.loc[valid].groupby(np.cumsum(breaks)[valid]):
        times = (part.candidate_defrost_time - origin).dt.total_seconds().to_numpy() / 60
        edges = np.arange(0.5, t[-1] + 1e-9, window_minutes)
        edges = edges[(edges >= times[0]) & (edges <= times[-1])]
        if len(edges) < 2:
            continue
        cop = np.interp(edges, times, part.cycle_cop.to_numpy()) / peak
        # Percentage points of peak COP per physical minute, not normalized time.
        slopes = 100 * np.diff(cop) / window_minutes
        for j, slope in enumerate(slopes):
            records.append(
                dict(
                    window_minutes=window_minutes,
                    block=int(block),
                    midpoint_minutes=(edges[j] + edges[j + 1]) / 2,
                    slope_pct_peak_per_minute=slope,
                    slope_cop_per_minute=slope * peak / 100,
                    slope_change=slopes[j] - slopes[j - 1] if j else np.nan,
                )
            )
    return pd.DataFrame(records)


def slope_direction_support(curves, windows, cop_minimum=0.6):
    """Locate slope comparisons and retain their two-window COP range explicitly."""
    result = windows.reset_index(drop=True).copy()
    result["minimum_relative_cop_in_comparison"] = np.nan
    result["comparison_start_progress"] = np.nan
    result["comparison_end_progress"] = np.nan
    for name, w in result.groupby("cycle_name"):
        g = curves.loc[curves.cycle_name.eq(name)].sort_values("elapsed_minutes")
        duration = (g.elapsed_minutes / g.cycle_progress).median()
        recovery = g.recovery_minutes.iloc[0]
        for index, row in w.iterrows():
            midpoint = row.midpoint_minutes + recovery
            start, end = midpoint - row.window_minutes / 2, midpoint + row.window_minutes / 2
            result.loc[index, "comparison_start_progress"] = start / duration
            result.loc[index, "comparison_end_progress"] = end / duration
            if pd.isna(row.slope_change):
                continue
            left = start - row.window_minutes
            observed = g.loc[g.elapsed_minutes.between(left, end), "relative_cop"]
            edges = np.interp([left, end], g.elapsed_minutes, g.relative_cop)
            if observed.notna().all() and np.isfinite(edges).all():
                result.loc[index, "minimum_relative_cop_in_comparison"] = min(
                    observed.min(), edges.min()
                )
    result["comparison_above_cop_minimum"] = result.minimum_relative_cop_in_comparison.ge(
        cop_minimum
    )
    return result


def frequency_cop_cells(curve, cell_seconds=30, cop_minimum=0.6):
    """Six joint directions over the same complete interval, using actual frequency."""
    g = curve.sort_values("candidate_defrost_time").copy()
    origin = pd.Timestamp(g.heating_accounting_start.iloc[0])
    heating = pd.Timestamp(g.measured_cycle_start.iloc[0])
    peak = g.loc[g.cycle_cop_eligible, "cycle_cop"].max()
    g["cycle_cop_eligible"] &= g.cycle_cop.ge(cop_minimum * peak) & np.isfinite(
        g.compressor_frequency
    )
    cells = slope_windows(g, cell_seconds / 60)
    if cells.empty:
        return cells
    t = (g.candidate_defrost_time - origin).dt.total_seconds().to_numpy() / 60
    edges = cells.midpoint_minutes.to_numpy()[:, None] + np.array([-1, 1]) * cell_seconds / 120
    frequency = np.interp(edges.ravel(), t, g.compressor_frequency).reshape(-1, 2)
    cells["frequency_start_hz"], cells["frequency_end_hz"] = frequency.T
    cells["frequency_change_hz"] = frequency[:, 1] - frequency[:, 0]
    cop = np.interp(edges.ravel(), t, g.cycle_cop).reshape(-1, 2)
    cells["cop_start_absolute"], cells["cop_end_absolute"] = cop.T
    cop = cop / peak
    cells["cop_start_relative"], cells["cop_end_relative"] = cop.T
    cells["cop_change_fraction_of_peak"] = cop[:, 1] - cop[:, 0]
    raw_cop = g.cycle_cop.to_numpy() / peak
    left = np.searchsorted(t, edges[:, 0], side="left")
    right = np.searchsorted(t, edges[:, 1], side="right")
    cells["minimum_relative_cop_in_cell"] = [
        min(cop[j].min(), raw_cop[a:b].min()) if b > a else cop[j].min()
        for j, (a, b) in enumerate(zip(left, right))
    ]
    cells["within_1pct_peak_entire_cell"] = cells.minimum_relative_cop_in_cell.ge(0.99)
    cells["cell_start_time"] = origin + pd.to_timedelta(edges[:, 0], unit="min")
    cells["cell_end_time"] = origin + pd.to_timedelta(edges[:, 1], unit="min")
    # Numerical equality, not an arbitrary control-action amplitude threshold.
    cells["frequency_direction"] = np.select(
        [cells.frequency_change_hz.gt(1e-9), cells.frequency_change_hz.lt(-1e-9)],
        ["increase", "decrease"],
        default="unchanged",
    )
    cells["cop_direction"] = np.where(
        cells.cop_change_fraction_of_peak.gt(0), "increase", "decrease"
    )
    cells = cells.loc[cells.cop_change_fraction_of_peak.abs().gt(1e-12)].copy()
    cells["category"] = cells.frequency_direction.map(
        {"increase": 0, "unchanged": 2, "decrease": 4}
    ) + cells.cop_direction.eq("decrease").astype(int)
    recovery = (origin - heating).total_seconds() / 60
    duration = (g.candidate_defrost_time.iloc[-1] - heating).total_seconds() / 60
    cells["cell_start_progress"] = (
        cells.midpoint_minutes + recovery - cell_seconds / 120
    ) / duration
    cells["cell_end_progress"] = (cells.midpoint_minutes + recovery + cell_seconds / 120) / duration
    return cells.drop(columns=["slope_change", "slope_pct_peak_per_minute", "slope_cop_per_minute"])


def frequency_rise_events(raw, curve, *, frequency_column, merge_seconds=30):
    """Native upward ramps and strictly preceding COP; never infer a ramp from COP."""
    g = curve.sort_values("candidate_defrost_time")
    start = pd.Timestamp(g.heating_accounting_start.iloc[0])
    end = g.candidate_defrost_time.max()
    trace = raw[["timestamp", frequency_column]].dropna().sort_values("timestamp").copy()
    trace.timestamp = pd.to_datetime(trace.timestamp)
    delta = trace[frequency_column].diff()
    fresh = trace.timestamp.diff().dt.total_seconds().le(30)
    trace["observation_block"] = trace.timestamp.diff().dt.total_seconds().gt(30).cumsum()
    changes = trace.loc[fresh & delta.abs().gt(1e-9)].copy()
    changes["delta_hz"] = delta.loc[changes.index]
    changes = changes.loc[changes.timestamp.between(start, end)]
    ramps = []
    for row in changes.itertuples(index=False):
        time, frequency, block, change = row
        if change > 0:
            if (
                ramps
                and ramps[-1]["last_change_positive"]
                and ramps[-1]["observation_block"] == block
                and (time - ramps[-1]["rise_end"]).total_seconds() <= merge_seconds
            ):
                ramps[-1]["rise_end"] = time
                ramps[-1]["frequency_after_hz"] = frequency
                ramps[-1]["rise_hz"] += change
            else:
                ramps.append(
                    dict(
                        rise_start=time,
                        rise_end=time,
                        frequency_before_hz=frequency - change,
                        frequency_after_hz=frequency,
                        rise_hz=change,
                        last_change_positive=True,
                        observation_block=block,
                    )
                )
        elif ramps:
            ramps[-1]["last_change_positive"] = False
    valid = g.loc[g.cycle_cop_eligible.eq(True) & np.isfinite(g.cycle_cop)]
    peak = valid.cycle_cop.max()
    peak_time = valid.loc[valid.cycle_cop.idxmax(), "candidate_defrost_time"]
    records = []
    for ramp in ramps:
        before = valid.loc[valid.candidate_defrost_time.lt(ramp["rise_start"])].tail(1)
        age = (
            (ramp["rise_start"] - before.candidate_defrost_time.iloc[0]).total_seconds()
            if len(before)
            else np.nan
        )
        supported = bool(len(before) and age <= 10.1)
        cop = before.cycle_cop.iloc[0] if supported else np.nan
        records.append(
            {k: v for k, v in ramp.items() if k != "last_change_positive"}
            | dict(
                frequency_column=frequency_column,
                pre_cop_time=before.candidate_defrost_time.iloc[0] if supported else pd.NaT,
                pre_cop_age_seconds=age,
                pre_cop=cop,
                maximum_cop=peak,
                peak_time=peak_time,
                pre_over_peak=cop / peak,
                pre_gap_pct=100 * (1 - cop / peak),
                post_peak=ramp["rise_start"] > peak_time,
                cop_supported=supported,
            )
        )
    return pd.DataFrame(records)


def collect_frequency_rises(
    dataset, cohort, values, n_jobs, *, merge_seconds=30, frequency_kinds=("actual", "command")
):
    from joblib import Parallel, delayed, parallel_config

    from image_models.control_context import RAW_CONTROL

    def cycle_events(name, curve):
        raw = DatasetLoader(dataset).load_cycle_original(
            name,
            columns=[
                "timestamp",
                RAW_CONTROL["actual_frequency_hz"],
                RAW_CONTROL["frequency_command_hz"],
            ],
        )
        events = []
        for kind, column in [
            ("actual", RAW_CONTROL["actual_frequency_hz"]),
            ("command", RAW_CONTROL["frequency_command_hz"]),
        ]:
            if kind not in frequency_kinds:
                continue
            table = frequency_rise_events(
                raw, curve, frequency_column=column, merge_seconds=merge_seconds
            )
            if table.empty:
                continue
            record = cohort.set_index("cycle_name").loc[name]
            table["frequency_kind"], table["cycle_name"] = kind, name
            table["experiment_id"], table["setpoint_c"] = record.experiment_id, record.setpoint_c
            events.append(table)
        return pd.concat(events, ignore_index=True) if events else pd.DataFrame()

    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        tables = Parallel(verbose=5)(
            delayed(cycle_events)(name, g) for name, g in values.groupby("cycle_name")
        )
    return pd.concat(tables, ignore_index=True)


def visible_frequency_rises(stages, *, first_only=True):
    """Supported nondecreasing stages in the displayed range, optionally first per cycle."""
    stages = stages.copy()
    for column in ["rise_start", "rise_end"]:
        stages[column] = pd.to_datetime(stages[column])
    # Holds belong to a stage; a fall or native observation gap splits it.
    eligible = stages.loc[
        stages.rise_hz.ge(3)
        & stages.rise_end.gt(stages.rise_start)
        & stages.cop_supported.eq(True)
        & stages.pre_over_peak.ge(0.6)
    ]
    first = eligible.sort_values("rise_start").copy()
    if first_only:
        first = first.groupby("cycle_name", sort=False).head(1).copy()
    first["rise_size_class"] = pd.cut(
        first.rise_hz, [3, 5, 10, np.inf], right=False, labels=["3–<5 Hz", "5–<10 Hz", "≥10 Hz"]
    )
    first["pre_within_1pct_peak"] = first.pre_over_peak.ge(0.99)
    return first


def audit_frequency_rises(dataset, cohort, values, output, n_jobs):
    events = collect_frequency_rises(dataset, cohort, values, n_jobs)
    output.mkdir(parents=True)
    cohort.to_csv(output / "cohort.csv", index=False)
    events.to_csv(output / "frequency_rise_events.csv", index=False)
    rows = []
    for (kind, temperature), group in events.groupby(["frequency_kind", "setpoint_c"]):
        supported = group.loc[group.cop_supported]
        relevant = supported.loc[supported.pre_over_peak.ge(0.6)]
        for scope, g in [("all_supported", supported), ("cop_at_least_06", relevant)]:
            near = g.pre_over_peak.ge(0.99)
            rows.append(
                dict(
                    frequency_kind=kind,
                    setpoint_c=temperature,
                    scope=scope,
                    detected_ramps=len(group),
                    supported_ramps=len(supported),
                    ramps=len(g),
                    cycles=g.cycle_name.nunique(),
                    experiments=g.experiment_id.nunique(),
                    near_peak=int(near.sum()),
                    outside_1pct=int((~near).sum()),
                    outside_1pct_post_peak=int((~near & g.post_peak).sum()),
                    outside_1pct_pre_peak=int((~near & ~g.post_peak).sum()),
                    median_gap_pct=g.pre_gap_pct.median(),
                    q90_gap_pct=g.pre_gap_pct.quantile(0.9),
                )
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "frequency_rise_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def main(args):
    prepare_output(args)
    if args.output.exists():
        raise FileExistsError(args.output)
    published = args.dataset / "training_data"
    settings = json.loads((published / "settings.json").read_text())
    # Full saved candidates and full quality records precede all cohort filtering.
    full = pd.read_parquet(published / "candidate_decisions.parquet")
    quality = pd.read_csv(settings["quality"])
    screening = screen_cv_cycles(
        full,
        DatasetLoader(args.dataset).list_cycles(),
        quality=quality,
        exclude_rb=settings["exclude_rb"],
        input_rule=settings["input_rule"],
        cohort_rule=settings["cohort_rule"],
        minimum_terminal_decline=settings.get("minimum_terminal_decline"),
        rb_minimum_terminal_decline=settings.get("rb_minimum_terminal_decline"),
        excluded_cycles=settings.get("excluded_cycles", []),
        maximum_per_setpoint=settings.get("maximum_per_setpoint"),
        post_near_minutes_by_setpoint=settings.get("post_near_minutes_by_setpoint"),
        maximum_interpolated_fraction=settings["limits"]["maximum_interpolated_fraction"],
    )
    cohort = screening.loc[screening.selected].copy()
    published_names = set(pd.read_csv(published / "cohort.csv").cycle_name)
    if set(cohort.cycle_name) != published_names:
        raise ValueError("Current screening differs from the published main cohort")
    values = calculate_cycle_cop(full.loc[full.cycle_name.isin(cohort.cycle_name)], effective=True)
    values.candidate_defrost_time = pd.to_datetime(values.candidate_defrost_time, format="mixed")
    if args.frequency_rise_audit:
        audit_frequency_rises(args.dataset, cohort, values, args.output, args.n_jobs)
        return
    traces, windows, summaries, cells = [], [], [], []
    for name, curve in values.groupby("cycle_name"):
        record = cohort.set_index("cycle_name").loc[name]
        origin = pd.Timestamp(curve.measured_cycle_start.iloc[0])
        heat_origin = pd.Timestamp(curve.heating_accounting_start.iloc[0])
        recovery_minutes = (heat_origin - origin).total_seconds() / 60
        duration = (record.end_time - origin).total_seconds() / 60
        curve = curve.loc[curve.candidate_defrost_time.between(heat_origin, record.end_time)].copy()
        peak = curve.loc[curve.cycle_cop_eligible, "cycle_cop"].max()
        trace = curve[
            [
                "cycle_name",
                "experiment_id",
                "candidate_defrost_time",
                "cycle_cop",
                "compressor_frequency",
                "cycle_cop_eligible",
                "cycle_cop_uses_model_extrapolation",
            ]
        ].copy()
        trace["elapsed_minutes"] = (trace.candidate_defrost_time - origin).dt.total_seconds() / 60
        trace["cycle_progress"] = trace.elapsed_minutes / duration
        trace["recovery_minutes"] = recovery_minutes
        trace["stable_heating_progress"] = recovery_minutes / duration
        trace["relative_cop"] = (trace.cycle_cop / peak).where(trace.cycle_cop_eligible)
        trace["setpoint_c"] = record.setpoint_c
        traces.append(trace)
        if args.figure == "frequency-cop-grid":
            cell = frequency_cop_cells(curve, args.cell_seconds)
            cell["cycle_name"], cell["experiment_id"], cell["setpoint_c"] = (
                name,
                record.experiment_id,
                record.setpoint_c,
            )
            cells.append(cell)
            continue
        for minutes in args.window_minutes:
            w = slope_windows(curve, minutes)
            if w.empty:
                continue
            w["cycle_name"], w["experiment_id"] = name, record.experiment_id
            w["cycle_progress"] = (w.midpoint_minutes + recovery_minutes) / duration
            w["setpoint_c"] = record.setpoint_c
            w["compressor_frequency"] = np.interp(
                w.midpoint_minutes,
                (curve.candidate_defrost_time - heat_origin).dt.total_seconds() / 60,
                curve.compressor_frequency,
            )
            windows.append(w)
            changes = w.slope_change.dropna()
            summaries.append(
                dict(
                    cycle_name=name,
                    experiment_id=record.experiment_id,
                    setpoint_c=record.setpoint_c,
                    window_minutes=minutes,
                    slope_comparisons=len(changes),
                    decreasing_slope_fraction=changes.le(0).mean(),
                    strictly_nonincreasing=bool(len(changes) and changes.le(0).all()),
                )
            )
    args.output.mkdir(parents=True)
    cohort.to_csv(args.output / "cohort.csv", index=False)
    if args.connect_rise_stages or (
        args.figure != "frequency-cop-grid" and args.rise_markers != "none"
    ):
        stages = collect_frequency_rises(
            args.dataset,
            cohort,
            values,
            args.n_jobs,
            merge_seconds=np.inf,
            frequency_kinds=("actual",),
        )
        origins = values.groupby("cycle_name").measured_cycle_start.first().map(pd.Timestamp)
        ends = cohort.set_index("cycle_name").end_time
        duration = (ends - origins).dt.total_seconds()
        for edge in ["start", "end"]:
            stages[f"stage_{edge}_progress"] = (
                stages[f"rise_{edge}"] - stages.cycle_name.map(origins)
            ).dt.total_seconds() / stages.cycle_name.map(duration)
        stages.to_csv(args.output / "frequency_rise_stages.csv", index=False)
    if args.figure == "frequency-cop-grid":
        pd.concat(cells, ignore_index=True).to_csv(
            args.output / "frequency_cop_cells.csv", index=False
        )
        from plots.frequency import render_frequency_cop_grid

        render_frequency_cop_grid(args.output, frequency_gradient=args.connect_rise_stages)
    else:
        traces, windows, summaries = pd.concat(traces), pd.concat(windows), pd.DataFrame(summaries)
        windows = slope_direction_support(traces, windows)
        traces.to_csv(args.output / "cop_curves.csv", index=False)
        windows.to_csv(args.output / "slope_windows.csv", index=False)
        summaries.to_csv(args.output / "cycle_slope_summary.csv", index=False)
        if args.rise_markers != "none":
            first = visible_frequency_rises(stages, first_only=args.rise_markers == "first")
            first["marker_progress"] = (
                pd.to_datetime(first.pre_cop_time) - first.cycle_name.map(origins)
            ).dt.total_seconds() / first.cycle_name.map(duration)
            first.to_csv(args.output / "frequency_rise_markers.csv", index=False)
        from plots.frequency import render_cop_slope

        render_cop_slope(
            args.output,
            rise_markers=args.rise_markers,
            activity_only=args.activity_only,
            include_all=args.figure == "cop-frequency-set",
            all_styles=args.figure == "cop-frequency-set",
            cop_lower=1.3 if args.figure == "cop-frequency-set" else 1.5,
        )
        print(
            summaries.groupby("window_minutes")
            .decreasing_slope_fraction.agg(["count", "median", "min", "max"])
            .to_string()
        )
    print(f"{len(cohort)} cycles, {cohort.experiment_id.nunique()} experiments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-minutes", type=int, nargs="+", default=[5, 10])
    parser.add_argument(
        "--figure",
        choices=["cop-slope", "frequency-cop-grid", "cop-frequency-set"],
        default="cop-slope",
    )
    parser.add_argument("--cell-seconds", type=int, default=30)
    parser.add_argument("--frequency-rise-audit", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--connect-rise-stages", action="store_true")
    parser.add_argument("--rise-markers", choices=["first", "all", "none"], default="first")
    parser.add_argument("--activity-only", action="store_true")
    args = parser.parse_args()
    if any(m <= 0 for m in args.window_minutes):
        parser.error("Window lengths must be positive")
    if args.cell_seconds <= 0:
        parser.error("Cell duration must be positive")
    main(args)
