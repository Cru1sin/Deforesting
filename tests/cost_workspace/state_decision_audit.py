"""Compare frozen state variants on shared raw-heat COP and support-boundary diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from dataset_tools import DatasetLoader
from defrost_decision.candidate_quantities import effective_candidate_cop
from defrost_decision.performance_objectives import add_single_objective_optima, calculate_cycle_cop
from defrost_event_models.training_data import catalog
from tests.cost_workspace.state_representation_audit import (
    load_raw,
    nearest_components,
    parameters,
    read_models,
)


def compare_cycle(dataset, record, variants, output):
    name = record["cycle_name"]
    destination = output / f"{name}_decisions.csv"
    if destination.exists():
        return pd.read_csv(destination)
    recovery = record["stable_heating_start"]
    if pd.isna(recovery):
        rows = pd.DataFrame(
            [
                dict(cycle_name=name, variant=key, status="missing_recovery_boundary")
                for key in variants
            ]
        )
        rows.to_csv(destination, index=False)
        return rows
    states = {key: pd.read_parquet(root / f"{name}.parquet") for key, (_, root) in variants.items()}
    baseline = next(iter(variants))
    times = pd.DatetimeIndex(states[baseline].candidate_defrost_time)
    for state in states.values():
        pd.testing.assert_index_equal(times, pd.DatetimeIndex(state.candidate_defrost_time))
    measured_path = output / f"{name}_measured.parquet"
    if measured_path.exists():
        measured = pd.read_parquet(measured_path)
    else:
        frame = load_raw(DatasetLoader(dataset), name)
        measured = effective_candidate_cop(
            frame,
            times,
            pd.Timestamp(recovery),
            pd.Timestamp(record["heating_start"]),
            variants[baseline][0],
            record["experiment_id"],
            preparation_heat="zero",
            prediction_mode="cross-fitted",
        )
        np.testing.assert_allclose(
            measured.defrost_event_electricity_kwh, states[baseline].prediction_kwh, atol=1e-12
        )
        measured.to_parquet(measured_path, index=False)
    rows, curve_output, tops = [], pd.DataFrame({"candidate_defrost_time": times}), []
    support_changes = []
    for key, (model, _) in variants.items():
        state = states[key]
        curve = measured.copy()
        curve["pre_defrost_feature_window_valid"] = state.pre_defrost_feature_window_valid
        curve["defrost_event_electricity_kwh"] = state.prediction_kwh
        curve["defrost_event_electricity_prediction_available"] = np.isfinite(state.prediction_kwh)
        curve["defrost_event_electricity_in_training_domain"] = state.support_score.le(1)
        curve = add_single_objective_optima(
            calculate_cycle_cop(curve, effective=True), names=("cycle_cop",)
        )
        possible = calculate_cycle_cop(
            curve, effective=True, allow_model_extrapolation=True
        ).cycle_cop_eligible
        eligible = curve.cycle_cop_eligible
        for column in ["cycle_cop", "cycle_cop_eligible"]:
            curve_output[f"{key}_{column}"] = curve[column]
        curve_output[f"{key}_support_score"] = state.support_score
        row = dict(
            cycle_name=name,
            experiment_id=record["experiment_id"],
            variant=key,
            status="selected" if eligible.any() else "no_supported_optimum",
            grid_candidates=len(curve),
            computable_candidates=int(possible.sum()),
            supported_candidates=int(eligible.sum()),
            supported_fraction=float(eligible.sum() / possible.sum()) if possible.any() else np.nan,
        )
        if eligible.any():
            i = int(
                curve.index[
                    eligible & times.to_series(index=curve.index).eq(curve.cycle_cop_t_star)
                ][0]
            )

            def edge(j, state=state, possible=possible, eligible=eligible):
                if j < 0 or j >= len(state):
                    return "action_boundary"
                if not state.pre_defrost_feature_window_valid.iloc[j]:
                    return "feature_invalid"
                if not possible.iloc[j]:
                    return "measurement_or_physical_invalid"
                return "model_support" if not eligible.iloc[j] else "interior"

            before, after = edge(i - 1), edge(i + 1)
            row.update(
                t_star=times[i],
                cop_max=curve.cycle_cop.iloc[i],
                edge_before=before,
                edge_after=after,
                support_boundary_optimum="model_support" in [before, after],
                basin_1pct_start=curve.cycle_cop_basin_1pct_start.iloc[i],
                basin_1pct_end=curve.cycle_cop_basin_1pct_end.iloc[i],
                basin_1pct_minutes=curve.cycle_cop_basin_1pct_width_minutes.iloc[i],
            )
            unsupported_times = times[possible & ~eligible]
            row["nearest_unsupported_seconds"] = (
                np.abs((unsupported_times - times[i]).total_seconds()).min()
                if len(unsupported_times)
                else np.nan
            )
            # Diagnostics never reselect on another variant's support.
            for other, other_state in states.items():
                row[f"selected_in_{other}_domain"] = bool(other_state.support_score.iloc[i] <= 1)
            p = parameters(model, record["experiment_id"])
            j = p["feature_order"].index("ambient_coil_difference")
            gain = p["coefficients"][j] / p["scaler_scale"][j]
            coil_contribution = -gain * state.coil_temperature.diff()
            row["peak_coil_contribution_kwh"] = coil_contribution.iloc[i]
            row["peak_coil_current_delta"] = state.current_coil_temperature.diff().iloc[i]
            near = eligible & times.to_series(index=curve.index).between(
                row["basin_1pct_start"], row["basin_1pct_end"]
            )
            for source, contribution, stable in [
                (
                    "coil_temperature",
                    coil_contribution,
                    state.current_coil_temperature.diff().eq(0)
                    & state.age_coil_temperature.le(1)
                    & state.age_coil_temperature.shift().le(1),
                ),
                *[
                    (
                        feature,
                        state[f"contribution_{feature}"],
                        state[f"raw_stable_{feature}"] & state[f"raw_observed_pair_{feature}"],
                    )
                    for feature in [
                        "ambient_temperature",
                        "mean_water_temperature",
                        "setpoint_outlet_difference",
                    ]
                ],
            ]:
                mask = near & stable & state.supported_pair
                for idx in contribution.loc[mask].abs().nlargest(3).index:
                    if abs(contribution.iloc[idx]) < 1e-12:
                        continue
                    tops.append(
                        dict(
                            cycle_name=name,
                            variant=key,
                            feature=source,
                            candidate_defrost_time=times[idx],
                            t_star=times[i],
                            contribution_kwh=contribution.iloc[idx],
                            cop=curve.cycle_cop.iloc[idx],
                            cop_max=row["cop_max"],
                        )
                    )
        if key != baseline:
            old_state = states[baseline]
            comparable = possible & old_state.pre_defrost_feature_window_valid
            old_domain, new_domain = old_state.support_score.le(1), state.support_score.le(1)
            changed = comparable & old_domain.ne(new_domain)
            if changed.any():
                old_params = parameters(variants[baseline][0], record["experiment_id"])
                new_params = parameters(model, record["experiment_id"])
                old_parts, _, _ = nearest_components(old_params, old_state.loc[changed])
                new_parts, _, _ = nearest_components(new_params, state.loc[changed])
                delta = new_parts / new_params["support_threshold"] ** 2 - (
                    old_parts / old_params["support_threshold"] ** 2
                )
                loss = old_domain.loc[changed].to_numpy()
                direction = np.where(loss, 1, -1)
                dominant = np.array(new_params["feature_order"])[
                    (delta * direction[:, None]).argmax(axis=1)
                ]
                support_changes.append(
                    pd.DataFrame(
                        dict(
                            cycle_name=name,
                            variant=key,
                            candidate_defrost_time=times[changed],
                            direction=np.where(loss, "loss", "gain"),
                            dominant_normalized_distance_change=dominant,
                            baseline_score=old_state.loc[changed, "support_score"].to_numpy(),
                            variant_score=state.loc[changed, "support_score"].to_numpy(),
                        )
                    )
                )
        rows.append(row)
    pd.DataFrame(rows).to_csv(destination, index=False)
    curve_output.to_parquet(output / f"{name}_cop.parquet", index=False)
    pd.DataFrame(
        tops,
        columns=[
            "cycle_name",
            "variant",
            "feature",
            "candidate_defrost_time",
            "t_star",
            "contribution_kwh",
            "cop",
            "cop_max",
        ],
    ).to_csv(output / f"{name}_near_optimal_jumps.csv", index=False)
    if support_changes:
        pd.concat(support_changes, ignore_index=True).to_csv(
            output / f"{name}_support_changes.csv", index=False
        )
    print(name, [(r["variant"], str(r.get("t_star"))) for r in rows], flush=True)
    return pd.DataFrame(rows)


def summarize_coil(variants, output):
    """Separate coil contribution from the unchanged ambient term in the difference."""
    chunks = {key: [] for key in variants}
    baseline_root = next(iter(variants.values()))[1]
    paths = sorted(baseline_root.glob("frost_cycle_*.parquet"))
    for path in paths:
        frames = {key: pd.read_parquet(root / path.name) for key, (_, root) in variants.items()}
        common = np.logical_and.reduce([frame.supported_pair for frame in frames.values()])
        for key, (model, _) in variants.items():
            frame = frames[key]
            p = parameters(model, frame.experiment_id.iloc[0])
            j = p["feature_order"].index("ambient_coil_difference")
            gain = p["coefficients"][j] / p["scaler_scale"][j]
            table = frame[
                [
                    "candidate_defrost_time",
                    "cycle_name",
                    "coil_temperature",
                    "current_coil_temperature",
                    "supported_pair",
                    "valid_pair",
                ]
            ].copy()
            table["contribution_kwh"] = -gain * frame.coil_temperature.diff()
            table["common_supported_pair"] = common
            table["current_stable_observed"] = (
                frame.current_coil_temperature.diff().eq(0)
                & frame.age_coil_temperature.le(1)
                & frame.age_coil_temperature.shift().le(1)
            )
            current = frame.current_coil_temperature
            frequency = frame.current_compressor_frequency
            table["single_second_return"] = (
                current.shift(-1).eq(current.shift())
                & current.diff().ne(0)
                & frame.age_coil_temperature.le(1)
                & frame.age_coil_temperature.shift().le(1)
                & frame.age_coil_temperature.shift(-1).le(1)
                & frequency.eq(frequency.shift())
                & frequency.eq(frequency.shift(-1))
                & frame.supported_pair.shift(-1, fill_value=False)
            )
            table["mismatch_degc"] = frame.coil_temperature - frame.current_coil_temperature
            chunks[key].append(table)
    summary, top = [], []
    for key, pieces in chunks.items():
        table = pd.concat(pieces, ignore_index=True)
        for scope in ["valid_pair", "supported_pair", "common_supported_pair"]:
            mask = table[scope]
            x = table.loc[mask, "contribution_kwh"].abs()
            stable = mask & table.current_stable_observed
            summary.append(
                dict(
                    variant=key,
                    scope=scope,
                    n=int(mask.sum()),
                    p95=x.quantile(0.95),
                    p99=x.quantile(0.99),
                    maximum=x.max(),
                    stable_observed_max=table.loc[stable, "contribution_kwh"].abs().max(),
                    mismatch_p95_degc=table.loc[mask, "mismatch_degc"].abs().quantile(0.95),
                    single_second_returns=int((mask & table.single_second_return).sum()),
                    return_contribution_p95_kwh=table.loc[
                        mask & table.single_second_return, "contribution_kwh"
                    ]
                    .abs()
                    .quantile(0.95),
                )
            )
            for kind, selected in [("largest", mask), ("current_stable", stable)]:
                indices = table.loc[selected, "contribution_kwh"].abs().nlargest(20).index
                rows = table.loc[indices].copy()
                rows["variant"], rows["scope"], rows["kind"] = key, scope, kind
                top.append(rows)
    pd.DataFrame(summary).to_csv(output / "coil_contribution_summary.csv", index=False)
    pd.concat(top, ignore_index=True).to_csv(output / "coil_contribution_top20.csv", index=False)


def audit_near_sources(variants, output):
    """Combine repeated underlying inputs before judging residual near-optimum jumps."""
    decisions = pd.read_csv(output / "decisions.csv")
    top, peaks = [], []
    for _, decision in decisions.loc[decisions.status.eq("selected")].iterrows():
        key, name = decision.variant, decision.cycle_name
        model, root = variants[key]
        state = pd.read_parquet(root / f"{name}.parquet")
        p = parameters(model, state.experiment_id.iloc[0])
        gain = dict(
            zip(
                p["feature_order"],
                np.array(p["coefficients"]) / np.array(p["scaler_scale"]),
                strict=True,
            )
        )
        source_gain = {
            "ambient_temperature": gain["ambient_temperature"] + gain["ambient_coil_difference"],
            "water_in_temperature": gain["mean_water_temperature"] / 2,
            "water_out_temperature": gain["mean_water_temperature"] / 2
            - gain["setpoint_outlet_difference"],
            "water_temperature_setpoint": gain["setpoint_outlet_difference"],
            "coil_temperature": -gain["ambient_coil_difference"],
            **{
                feature: gain[feature]
                for feature in [
                    "evaporating_pressure",
                    "compressor_frequency",
                    "heating_elapsed_minutes",
                    "evaporating_pressure_slope_5m",
                ]
            },
        }
        contribution = state[list(source_gain)].diff() * pd.Series(source_gain)
        mask = state.valid_pair
        np.testing.assert_allclose(
            contribution.sum(axis=1).loc[mask], state.prediction_kwh.diff().loc[mask], atol=1e-12
        )
        times = state.candidate_defrost_time
        peak = int(state.index[times.eq(pd.Timestamp(decision.t_star))][0])
        coil = state.current_coil_temperature
        one_second_return = bool(
            0 < peak < len(state) - 1
            and coil.iloc[peak - 1] == coil.iloc[peak + 1]
            and coil.iloc[peak] != coil.iloc[peak - 1]
            and state.age_coil_temperature.iloc[peak - 1 : peak + 2].le(1).all()
            and state.current_compressor_frequency.iloc[peak - 1 : peak + 2].nunique() == 1
        )
        source = contribution.iloc[peak].abs().idxmax() if state.valid_pair.iloc[peak] else None
        measured = pd.read_parquet(
            output / f"{name}_measured.parquet", columns=["pre_defrost_electricity_kwh"]
        )
        denominator = (
            measured.pre_defrost_electricity_kwh.iloc[peak] + state.prediction_kwh.iloc[peak]
        )
        peaks.append(
            dict(
                cycle_name=name,
                variant=key,
                t_star=decision.t_star,
                coil_single_second_return=one_second_return,
                peak_coil_change_cop_fraction=(
                    -contribution.loc[peak, "coil_temperature"] / denominator
                    if state.valid_pair.iloc[peak]
                    else np.nan
                ),
                largest_source=source,
                largest_source_contribution_kwh=(
                    contribution.loc[peak, source] if source is not None else np.nan
                ),
            )
        )
        costs = pd.read_parquet(
            output / f"{name}_cop.parquet",
            columns=[f"{key}_cycle_cop", f"{key}_cycle_cop_eligible"],
        )
        # Audit every near-optimal component; do not change the selected connected basin.
        near = costs[f"{key}_cycle_cop_eligible"] & costs[f"{key}_cycle_cop"].ge(
            0.99 * decision.cop_max
        )
        for source in [
            "ambient_temperature",
            "water_in_temperature",
            "water_out_temperature",
            "water_temperature_setpoint",
            "coil_temperature",
        ]:
            stable = state[f"current_{source}"].diff().eq(0) & state[f"age_{source}"].le(1)
            stable &= state[f"age_{source}"].shift().le(1)
            selected = near & state.supported_pair & stable
            for i in contribution.loc[selected, source].abs().nlargest(3).index:
                if abs(contribution.loc[i, source]) < 1e-12:
                    continue
                top.append(
                    dict(
                        cycle_name=name,
                        variant=key,
                        source=source,
                        near_scope="all_eligible_within_1pct",
                        candidate_defrost_time=times.iloc[i],
                        t_star=decision.t_star,
                        contribution_kwh=contribution.loc[i, source],
                        total_delta_E_kwh=state.prediction_kwh.diff().iloc[i],
                    )
                )
    pd.DataFrame(peaks).to_csv(output / "peak_source_audit.csv", index=False)
    table = pd.DataFrame(top)
    table["absolute_contribution_kwh"] = table.contribution_kwh.abs()
    table.sort_values("absolute_contribution_kwh", ascending=False).groupby(
        ["variant", "source"], sort=False
    ).head(20).to_csv(output / "near_source_top20.csv", index=False)


def audit_outlet_hold(variants, output):
    """One-point, one-second holds only; never replace saved model curves or support."""
    from image_models.inputs import processing_rows

    decisions = pd.read_csv(output / "decisions.csv")
    evidence, summaries = [], []
    for key, (model, root) in variants.items():
        for row in decisions.loc[decisions.variant.eq(key)].itertuples():
            if row.status != "selected":
                summaries.append(dict(variant=key, cycle_name=row.cycle_name, status=row.status))
                continue
            state = pd.read_parquet(root / f"{row.cycle_name}.parquet")
            measured = pd.read_parquet(output / f"{row.cycle_name}_measured.parquet")
            curve = pd.read_parquet(output / f"{row.cycle_name}_cop.parquet")
            pd.testing.assert_series_equal(
                state.candidate_defrost_time, measured.candidate_defrost_time
            )
            p = parameters(model, state.experiment_id.iloc[0])
            gain = dict(
                zip(
                    p["feature_order"],
                    np.array(p["coefficients"]) / np.array(p["scaler_scale"]),
                    strict=True,
                )
            )
            # Outlet occurs in two derived inputs; combine both signed effects.
            a = (
                gain["mean_water_temperature"] / 2 - gain["setpoint_outlet_difference"]
            ) * state.water_out_temperature.diff()
            q = measured.pre_defrost_heat_kwh + measured.defrost_event_net_heat_kwh
            d = measured.pre_defrost_electricity_kwh + state.prediction_kwh
            c, hold = q / d, q / (d - a)
            eligible = curve[f"{key}_cycle_cop_eligible"]
            valid = eligible & state.valid_pair & (d - a).gt(0) & np.isfinite(hold)
            np.testing.assert_allclose(
                c[eligible], curve.loc[eligible, f"{key}_cycle_cop"], atol=1e-12
            )
            np.testing.assert_allclose(((c - hold) / hold)[valid], (-a / d)[valid], atol=1e-12)
            stable = (
                state.current_water_out_temperature.diff().eq(0)
                & state.age_water_out_temperature.le(1)
                & state.age_water_out_temperature.shift().le(1)
            )
            updates = valid & stable & a.abs().gt(1e-12)
            clock = processing_rows(state.assign(cycle_name=row.cycle_name))
            grid = pd.DataFrame(
                dict(
                    candidate_defrost_time=state.candidate_defrost_time,
                    cycle_cop=c,
                    cycle_cop_eligible=eligible & clock,
                )
            )
            selected = add_single_objective_optima(grid, names=("cycle_cop",))
            if not (eligible & clock).any():
                summaries.append(
                    dict(variant=key, cycle_name=row.cycle_name, status="no_supported_action_grid")
                )
                continue
            t0 = selected.cycle_cop_t_star.iloc[0]
            i0 = int(state.index[state.candidate_defrost_time.eq(t0)][0])
            nearby = (state.candidate_defrost_time - pd.Timestamp(row.t_star)).dt.total_seconds()
            indices = state.index[updates | nearby.abs().le(60) & valid]
            changed, losses, hold_losses = 0, [], []
            for i in indices:
                record = dict(
                    variant=key,
                    cycle_name=row.cycle_name,
                    candidate_defrost_time=state.candidate_defrost_time.iloc[i],
                    seconds_from_second_peak=nearby.iloc[i],
                    on_action_grid=bool(clock.iloc[i]),
                    stable_outlet_update=bool(updates.iloc[i]),
                    outlet_current=state.current_water_out_temperature.iloc[i],
                    outlet_feature=state.water_out_temperature.iloc[i],
                    contribution_kwh=a.iloc[i],
                    heat_kwh=q.iloc[i],
                    denominator_kwh=d.iloc[i],
                    cop=c.iloc[i],
                    cop_hold=hold.iloc[i],
                    relative_effect=-a.iloc[i] / d.iloc[i],
                    grid_t_star=t0,
                )
                if clock.iloc[i] and updates.iloc[i]:
                    trial = grid.copy()
                    trial.loc[i, "cycle_cop"] = hold.iloc[i]
                    chosen = add_single_objective_optima(trial, names=("cycle_cop",))
                    t1 = chosen.cycle_cop_t_star.iloc[0]
                    j = int(state.index[state.candidate_defrost_time.eq(t1)][0])
                    loss = (c.iloc[i0] - c.iloc[j]) / c.iloc[i0]
                    hold_loss = (
                        trial.cycle_cop.iloc[j] - trial.cycle_cop.iloc[i0]
                    ) / trial.cycle_cop.iloc[j]
                    changed += int(t1 != t0)
                    losses.append(loss)
                    hold_losses.append(hold_loss)
                    record.update(
                        hold_grid_t_star=t1,
                        original_model_regret=loss,
                        hold_diagnostic_regret=hold_loss,
                    )
                evidence.append(record)
            summaries.append(
                dict(
                    variant=key,
                    cycle_name=row.cycle_name,
                    status="checked",
                    grid_t_star=t0,
                    grid_cop_max=c.iloc[i0],
                    updates_seconds=int(updates.sum()),
                    updates_on_grid=int((updates & clock).sum()),
                    changed_grid_choices=changed,
                    max_original_model_regret=max(losses, default=0),
                    max_hold_diagnostic_regret=max(hold_losses, default=0),
                    max_absolute_local_effect=(-a[updates] / d[updates]).abs().max(),
                )
            )
    pd.DataFrame(evidence).to_csv(output / "outlet_hold_points.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "outlet_hold_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        nargs=3,
        action="append",
        metavar=("NAME", "MODEL_RUN", "STATE_AUDIT"),
        required=True,
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument(
        "--outlet-hold-only",
        action="store_true",
        help="Read caches only; diagnose isolated outlet updates on the existing 30 s clock",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    variants = {key: (read_models(Path(run)), Path(audit)) for key, run, audit in args.variant}
    if args.outlet_hold_only:
        audit_outlet_hold(variants, args.output)
        return
    summarize_coil(variants, args.output)
    records = catalog(DatasetLoader(args.dataset))
    selected = records.loc[
        records.cycle_name.map(
            lambda name: all((root / f"{name}.parquet").exists() for _, root in variants.values())
        )
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(compare_cycle)(args.dataset, row, variants, args.output)
            for row in selected.to_dict("records")
        )
    pd.concat(results, ignore_index=True).to_csv(args.output / "decisions.csv", index=False)
    audit_near_sources(variants, args.output)
    near = pd.concat(
        [pd.read_csv(path) for path in args.output.glob("*_near_optimal_jumps.csv")],
        ignore_index=True,
    )
    changes = pd.concat(
        [pd.read_csv(path) for path in args.output.glob("*_support_changes.csv")], ignore_index=True
    )
    changes.groupby(["variant", "direction", "dominant_normalized_distance_change"]).size().rename(
        "candidate_count"
    ).reset_index().to_csv(args.output / "support_change_summary.csv", index=False)
    near["absolute_contribution_kwh"] = near.contribution_kwh.abs()
    near.sort_values("absolute_contribution_kwh", ascending=False).groupby(
        ["variant", "feature"], sort=False
    ).head(20).to_csv(args.output / "near_optimal_top20.csv", index=False)


if __name__ == "__main__":
    main()
