"""Bounded, replayable heat-window experiment using the existing COP path.

Run from the repository with PYTHONPATH=. uv run python <this file> --check,
or supply --cycles and --output. No Dataset or model files are modified.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def smooth_heat(frame, start, window_seconds):
    """Trailing time integral of the same linear segments as window_audit.

    Short missing intervals bridge only once both observations have arrived;
    original NaNs remain NaN. Gaps > MAXIMUM_GAP_SECONDS restart history.
    The first observation is the zero-duration limit of the startup mean.
    """
    from defrost_event_models.training_data import MAXIMUM_GAP_SECONDS

    result = frame.heating_capacity.copy()
    if window_seconds == 0:
        return result
    valid = frame.loc[frame.timestamp.ge(start) & result.notna()]
    groups = valid.timestamp.diff().dt.total_seconds().gt(MAXIMUM_GAP_SECONDS).cumsum()
    for _, part in valid.groupby(groups):
        t = (part.timestamp - part.timestamp.iloc[0]).dt.total_seconds().to_numpy()
        q = part.heating_capacity.to_numpy()
        area = np.r_[0., np.cumsum(np.diff(t) * (q[1:] + q[:-1]) / 2)]
        left = np.maximum(0, t - window_seconds)
        j = np.searchsorted(t, left, side="right") - 1
        next_j = np.minimum(j + 1, len(t) - 1)
        slope = np.divide(q[next_j] - q[j], t[next_j] - t[j],
                          out=np.zeros(len(t)), where=next_j != j)
        offset = left - t[j]
        left_area = area[j] + q[j] * offset + slope * offset ** 2 / 2
        result.loc[part.index] = np.divide(area - left_area, t - left,
                                         out=q.copy(), where=t > left)
    return result


def check_smoothing():
    start = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame({
        "timestamp": start + pd.to_timedelta([0, 1, 4, 10], unit="s"),
        "heating_capacity": [20., 19., 16., 10.],
    })
    result = smooth_heat(frame, start, 6)
    np.testing.assert_allclose(result, [20, 19.5, 18, 13])
    for n in range(1, len(frame) + 1):
        np.testing.assert_allclose(smooth_heat(frame.iloc[:n], start, 6), result.iloc[:n])
    pd.testing.assert_series_equal(smooth_heat(frame, start, 0), frame.heating_capacity)
    frame = pd.DataFrame({
        "timestamp": start + pd.to_timedelta([-10, 0, 5, 10, 50, 55], unit="s"),
        "heating_capacity": [100., 6., np.nan, 6., 3., 3.],
    })
    result = smooth_heat(frame, start, 60)
    np.testing.assert_allclose(result, [100, 6, np.nan, 6, 3, 3], equal_nan=True)
    print("PASS: irregular-time linear integral, startup, prefix causality, raw identity, "
          "NaN preservation, phase boundary and long-gap reset", flush=True)


def run_cycle(dataset, model_file, name, windows, output, pulse_cycles):
    from dataset_tools.load_dataset import DatasetLoader
    from defrost_decision.performance_objectives import (
        add_single_objective_optima,
        calculate_cycle_cop,
    )
    from defrost_event_models.ridge_models import load_defrost_event_models
    from defrost_event_models.training_data import candidate_integral_table
    from select_defrost_time import calculate_cycle

    loader = DatasetLoader(dataset)
    models = load_defrost_event_models(model_file)
    raw = calculate_cycle(loader, name, models, candidate_step_seconds=10,
                          preparation_heat="zero", prediction_mode="cross-fitted")
    if not raw.cycle_cop_eligible.any():
        raw.to_csv(output / f"{name}_unavailable.csv", index=False)
        print(f"{name}: no supported raw candidate", flush=True)
        return pd.DataFrame([dict(cycle_name=name, status=raw.cycle_status.iloc[0])])
    frame = loader.load_cycle_original(name)
    frame.timestamp = pd.to_datetime(frame.timestamp)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    start = raw.stable_heating_start.iloc[0]
    times = raw.candidate_defrost_time
    tref = raw.reference_outlet_temperature.iloc[0]
    raw_best = raw.loc[raw.cycle_cop_eligible, "cycle_cop"].idxmax()
    raw_time = times.iloc[raw_best]
    rows, curves, references = [], [], {}
    traces = frame.loc[frame.timestamp.ge(start) & frame.timestamp.lt(times.max()),
                       ["timestamp", "heating_capacity", "water_out_temperature"]].copy()
    scenarios = [("reference", w, 0.) for w in windows]
    if name in pulse_cycles:
        scenarios += [(f"pulse_{sign:+.2f}kW", w, sign)
                      for sign in [-.17, .17] for w in [0, 60]]
    for scenario, window, amplitude in scenarios:
        trial = frame.copy()
        if amplitude:
            pulse = trial.timestamp.ge(raw_time - pd.Timedelta(seconds=20)) & (
                trial.timestamp.lt(raw_time - pd.Timedelta(seconds=10)))
            trial.loc[pulse, "heating_capacity"] += amplitude
        trial.heating_capacity = smooth_heat(trial, start, window)
        heat = candidate_integral_table(trial, start, times.tolist(), "heating_capacity",
                                       minimum_outlet_temperature=tref)
        if window == 0 and not amplitude:
            np.testing.assert_allclose(heat.energy, raw.pre_defrost_heat_kwh,
                                       rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(heat.valid, raw.pre_defrost_heat_measurement_valid)
        curve = raw.copy()
        curve["pre_defrost_heat_kwh"] = heat.energy.to_numpy()
        # Smoothing cannot rehabilitate an originally invalid measurement.
        curve["pre_defrost_heat_measurement_valid"] = (
            heat.valid.to_numpy() & raw.pre_defrost_heat_measurement_valid.to_numpy())
        curve = add_single_objective_optima(calculate_cycle_cop(curve, effective=True),
                                           names=("cycle_cop",))
        curve["is_selected"] = times.eq(curve.cycle_cop_t_star)
        curve["t_star"] = curve.cycle_cop_t_star
        curve["window_seconds"] = window
        curve["scenario"] = scenario
        curves.append(curve)
        np.testing.assert_array_equal(curve.defrost_event_electricity_kwh,
                                      raw.defrost_event_electricity_kwh)
        selected = curve.loc[curve.is_selected]
        if selected.empty:
            rows.append(dict(cycle_name=name, window_seconds=window, scenario=scenario,
                             status="no_supported_candidate"))
            continue
        best = selected.index[0]
        tstar = times.iloc[best]
        common = curve.cycle_cop_eligible & raw.cycle_cop_eligible
        last = common[common].index[-1]
        if not amplitude:
            references[window] = curve
            traces[f"heat_sma_{window}s_kw"] = trial.loc[traces.index, "heating_capacity"]
            prefix = frame.loc[frame.timestamp.lt(tstar)].copy()
            np.testing.assert_allclose(
                smooth_heat(prefix, start, window),
                trial.loc[prefix.index, "heating_capacity"], equal_nan=True)
        reference = references[window]
        ref_best = reference.loc[reference.cycle_cop_eligible, "cycle_cop"].idxmax()
        rows.append(dict(
            cycle_name=name, experiment_id=raw.experiment_id.iloc[0], status="supported",
            window_seconds=window, scenario=scenario, t_star=tstar,
            shift_from_raw_s=(tstar - raw_time).total_seconds(),
            pulse_shift_s=(tstar - times.iloc[ref_best]).total_seconds(),
            cop_max=curve.cycle_cop.iloc[best],
            raw_cop_loss_at_selected_pct=(100 * (1 - raw.cycle_cop.iloc[best]
                / raw.cycle_cop.iloc[raw_best]) if raw.cycle_cop_eligible.iloc[best] else np.nan),
            own_cop_loss_at_raw_selected_pct=(100 * (1 - curve.cycle_cop.iloc[raw_best]
                / curve.cycle_cop.iloc[best])
                if curve.cycle_cop_eligible.iloc[raw_best] else np.nan),
            in_raw_connected_1pct_basin=(
                raw.cycle_cop_basin_1pct_start.iloc[0] <= tstar
                <= raw.cycle_cop_basin_1pct_end.iloc[0]),
            basin_1pct_minutes=curve.cycle_cop_basin_1pct_width_minutes.iloc[0],
            heat_change_at_raw_opt_pct=100 * (curve.pre_defrost_heat_kwh.iloc[raw_best]
                / raw.pre_defrost_heat_kwh.iloc[raw_best] - 1),
            last_common_time=times.iloc[last],
            heat_change_at_last_common_pct=100 * (curve.pre_defrost_heat_kwh.iloc[last]
                / raw.pre_defrost_heat_kwh.iloc[last] - 1),
            eligible_candidates=int(curve.cycle_cop_eligible.sum()),
            eligibility_changed=int(curve.cycle_cop_eligible.ne(raw.cycle_cop_eligible).sum()),
            pulse_start=raw_time - pd.Timedelta(seconds=20) if amplitude else pd.NaT,
            pulse_end=raw_time - pd.Timedelta(seconds=10) if amplitude else pd.NaT,
        ))
    combined = pd.concat(curves, ignore_index=True)
    combined.to_csv(output / f"{name}_candidates.csv", index=False)
    traces.to_csv(output / f"{name}_heat.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / f"{name}_summary.csv", index=False)
    print(f"{name}: {len(times)} candidates; " + ", ".join(
        f"{r['window_seconds']}s: shift {r['shift_from_raw_s']:+.0f}s"
        for r in rows if r.get("scenario") == "reference" and r["status"] == "supported"),
        flush=True)
    return summary


def render_comparison(output, names, *, refined=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plots.cop import _plot_supported_curve

    fig, axes = plt.subplots(len(names), 2, figsize=(12, 3 * len(names)), squeeze=False,
                             layout="constrained")
    colors = {0: "#555555", 30: "#287D68", 60: "#C77836", 90: "#3C6E8F"}
    for row, name in enumerate(names):
        suffix = "_refined" if refined else ""
        path = output / f"{name}{suffix}_candidates.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table = table.loc[table.scenario.eq("reference")]
        origin = pd.Timestamp(table.stable_heating_start.iloc[0])
        raw = table.loc[table.window_seconds.eq(0)]
        raw_max = raw.loc[raw.cycle_cop_eligible, "cycle_cop"].max()
        selected_times = pd.to_datetime(table.loc[table.is_selected, "candidate_defrost_time"])
        margin = 1 if refined else 5
        left = max(0, (selected_times.min() - origin).total_seconds() / 60 - margin)
        right = (selected_times.max() - origin).total_seconds() / 60 + margin
        for window, curve in table.groupby("window_seconds", sort=True):
            x = (pd.to_datetime(curve.candidate_defrost_time) - origin).dt.total_seconds() / 60
            for ax in axes[row]:
                _plot_supported_curve(ax, x, 100 * curve.cycle_cop / raw_max,
                                      curve.cycle_cop_eligible, color=colors[window],
                                      label="Raw" if window == 0 else f"{window} s")
                chosen = curve.is_selected
                ax.scatter(x[chosen], 100 * curve.loc[chosen, "cycle_cop"] / raw_max,
                           color=colors[window], s=22, zorder=4)
            axes[row, 0].set_title(name + " | full curve")
            axes[row, 1].set_title("Peak detail; dots = each method's own selection")
        axes[row, 1].set_xlim(left, right)
        local = table.loc[((pd.to_datetime(table.candidate_defrost_time) - origin)
                          .dt.total_seconds() / 60).between(left, right)
                          & table.cycle_cop_eligible, "cycle_cop"] * 100 / raw_max
        if len(local):
            padding = max(.02, (local.max() - local.min()) * .08)
            axes[row, 1].set_ylim(local.min() - padding, local.max() + padding)
        for ax in axes[row]:
            ax.set_xlabel("Minutes after stable heating")
            ax.set_ylabel("COP / Raw maximum (%)")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(alpha=.15)
    fig.suptitle("Fixed denominator and temperature gate | causal heat-window sensitivity")
    fig.savefig(output / f"cop_window_comparison{'_refined' if refined else ''}.png", dpi=140)
    plt.close(fig)


def refine_cycle(dataset, model_file, name, output):
    """Add a one-second grid within 60 s of the coarse optimum; retain all old points."""
    from dataset_tools.load_dataset import DatasetLoader
    from defrost_decision.candidate_quantities import effective_candidate_cop
    from defrost_decision.performance_objectives import (
        add_single_objective_optima,
        calculate_cycle_cop,
    )
    from defrost_event_models.ridge_models import load_defrost_event_models
    from defrost_event_models.training_data import candidate_integral_table

    coarse = pd.read_csv(output / f"{name}_candidates.csv")
    coarse = coarse.loc[coarse.scenario.eq("reference")].copy()
    coarse.candidate_defrost_time = pd.to_datetime(coarse.candidate_defrost_time)
    center = coarse.loc[coarse.is_selected & coarse.window_seconds.eq(0),
                        "candidate_defrost_time"].iloc[0]
    times = pd.date_range(max(center - pd.Timedelta(seconds=60),
                             coarse.candidate_defrost_time.min()),
                          min(center + pd.Timedelta(seconds=60),
                              coarse.candidate_defrost_time.max()), freq="s")
    loader = DatasetLoader(dataset)
    bounds = loader.get_cycle_record(name)["boundaries"]
    start = pd.Timestamp(bounds["stable_heating_start"])
    frame = loader.load_cycle_original(name)
    frame.timestamp = pd.to_datetime(frame.timestamp)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    raw = effective_candidate_cop(
        frame, times, start, pd.Timestamp(bounds["heating_start"]),
        load_defrost_event_models(model_file), coarse.experiment_id.iloc[0],
        preparation_heat="zero", prediction_mode="cross-fitted")
    results, curves = [], []
    for window in sorted(coarse.window_seconds.unique()):
        trial = frame.copy()
        trial.heating_capacity = smooth_heat(trial, start, window)
        heat = candidate_integral_table(trial, start, times, "heating_capacity",
                                       minimum_outlet_temperature=raw.reference_outlet_temperature.iloc[0])
        curve = raw.copy()
        curve["pre_defrost_heat_kwh"] = heat.energy.to_numpy()
        curve["pre_defrost_heat_measurement_valid"] = (
            heat.valid.to_numpy() & raw.pre_defrost_heat_measurement_valid.to_numpy())
        curve = calculate_cycle_cop(curve, effective=True)
        curve["window_seconds"] = window
        curve = pd.concat([coarse.loc[coarse.window_seconds.eq(window)], curve],
                          ignore_index=True).drop_duplicates("candidate_defrost_time", keep="last")
        curve = curve.sort_values("candidate_defrost_time").reset_index(drop=True)
        curve = add_single_objective_optima(curve, names=("cycle_cop",))
        curve["is_selected"] = curve.candidate_defrost_time.eq(curve.cycle_cop_t_star)
        curve["t_star"] = curve.cycle_cop_t_star
        curve["selected_point_in_training_domain"] = curve.is_selected & curve.cycle_cop_eligible
        curve["t_star_model_supported"] = curve.is_selected.any()
        curve["scenario"] = "reference"
        curve["cycle_name"] = name
        curve["stable_heating_start"] = start
        curves.append(curve)
        best = curve.loc[curve.is_selected].iloc[0]
        results.append(dict(cycle_name=name, window_seconds=window,
                            t_star=best.candidate_defrost_time,
                            change_from_coarse_raw_s=(best.candidate_defrost_time-center).total_seconds(),
                            cop_max=best.cycle_cop,
                            basin_1pct_minutes=best.cycle_cop_basin_1pct_width_minutes))
    summary = pd.DataFrame(results)
    summary["shift_from_refined_raw_s"] = (
        summary.t_star - summary.loc[summary.window_seconds.eq(0), "t_star"].iloc[0]
    ).dt.total_seconds()
    summary.to_csv(output / f"{name}_refined_summary.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(
        output / f"{name}_refined_candidates.csv", index=False)
    print(f"{name}: 1 s refinement shifts {summary.shift_from_refined_raw_s.tolist()}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--model-file", type=Path, default=Path(
        "output/defrost_event_models/complete_defrost_events_zero/candidate_model_parameters.json"))
    parser.add_argument("--cycles", nargs="+", default=[])
    parser.add_argument("--windows", nargs="+", type=int, default=[0, 30, 60, 90])
    parser.add_argument("--pulse-cycles", nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--refine-only", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_smoothing()
    if args.cycles:
        from joblib import Parallel, delayed, parallel_config
        if args.output is None:
            parser.error("--output is required for a study")
        if args.refine_only:
            with parallel_config(backend="loky", inner_max_num_threads=1):
                results = Parallel(n_jobs=args.n_jobs)(delayed(refine_cycle)(
                    args.dataset, args.model_file, name, args.output) for name in args.cycles)
            pd.concat(results, ignore_index=True).to_csv(
                args.output / "refined_summary.csv", index=False)
            render_comparison(args.output, args.cycles, refined=True)
            raise SystemExit(0)
        args.output.mkdir(parents=True, exist_ok=False)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=args.n_jobs)(delayed(run_cycle)(
                args.dataset, args.model_file, name, args.windows, args.output,
                args.pulse_cycles) for name in args.cycles)
        pd.concat(results, ignore_index=True).to_csv(args.output / "summary.csv", index=False)
        render_comparison(args.output, args.cycles)
