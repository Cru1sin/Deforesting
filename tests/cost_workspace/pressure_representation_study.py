"""Compare state representations using shared models and saved candidate grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_tools import DatasetLoader
from defrost_decision.performance_objectives import add_single_objective_optima, calculate_cycle_cop
from defrost_event_models.ridge_models import predict_with_event_model
from defrost_event_models.training_data import extract_pre_defrost_features


def load_curve(root, name):
    table = pd.read_csv(root / f"{name}_refined_candidates.csv")
    table = table.loc[table.window_seconds.eq(0)].copy()
    table.candidate_defrost_time = pd.to_datetime(table.candidate_defrost_time)
    return table.sort_values("candidate_defrost_time").reset_index(drop=True)


def audit_features(root, names, models, output):
    tables = []
    for name in names:
        curve = load_curve(root, name)
        params = models["models"]["ridge_dynamic_state_8"]["event_electricity"]["folds"][
            curve.experiment_id.dropna().iloc[0]
        ]
        features = params["feature_order"]
        filled = curve[features].fillna(dict(zip(features, params["imputer_median"], strict=True)))
        changes = filled.diff() * (
            np.array(params["coefficients"]) / np.array(params["scaler_scale"])
        )
        delta = curve.defrost_event_electricity_kwh.diff()
        np.testing.assert_allclose(changes.sum(axis=1).iloc[1:], delta.iloc[1:], atol=1e-12)
        table = changes.add_prefix("contribution_")
        table["cycle_name"] = name
        table["candidate_defrost_time"] = curve.candidate_defrost_time
        table["step_seconds"] = curve.candidate_defrost_time.diff().dt.total_seconds()
        table["delta_E_kwh"] = delta
        table["supported_pair"] = curve.cycle_cop_eligible & curve.cycle_cop_eligible.shift(
            fill_value=False
        )
        table["imputed_pair"] = curve[features].isna().any(axis=1) | curve[features].isna().any(
            axis=1
        ).shift(fill_value=False)
        table["dominant_feature"] = changes.iloc[1:].abs().idxmax(axis=1)
        tables.append(table.iloc[1:])
    jumps = pd.concat(tables, ignore_index=True)
    jumps.to_csv(output / "feature_jump_audit.csv", index=False)
    large = jumps.loc[jumps.delta_E_kwh.abs().gt(0.02)]
    summary = (
        large.groupby(["step_seconds", "supported_pair", "dominant_feature"])
        .size()
        .rename("count")
        .reset_index()
    )
    summary.to_csv(output / "large_jump_sources.csv", index=False)
    print("Old-model jumps >0.02 kWh (descriptive threshold; mixed grids kept separate):")
    print(summary.to_string(index=False), flush=True)


def replay_variant(frame, old, models, heating):
    """Keep measured Q/E and boundaries fixed; change only state/prediction/support."""
    result = old.copy()
    features = extract_pre_defrost_features(
        frame,
        result.candidate_defrost_time,
        heating,
        pressure_statistic=models.get("pressure_statistic", "median"),
        frequency_statistic=models.get("frequency_statistic", "median"),
        coil_statistic=models.get("coil_statistic", "median"),
    )
    for column in features:
        result[column] = features[column].to_numpy()
    prediction = predict_with_event_model(
        models["models"]["ridge_dynamic_state_8"]["event_electricity"],
        result,
        str(old.experiment_id.dropna().iloc[0]),
        prediction_mode="cross-fitted",
    )
    result["defrost_event_electricity_kwh"] = prediction.prediction.to_numpy()
    result["defrost_event_electricity_prediction_available"] = np.isfinite(prediction.prediction)
    result["defrost_event_electricity_training_distance"] = prediction.support_distance.to_numpy()
    result["defrost_event_electricity_in_training_domain"] = prediction.support_distance.le(
        prediction.support_threshold
    ).to_numpy()
    result["pressure_statistic"] = models.get("pressure_statistic", "median")
    result["frequency_statistic"] = models.get("frequency_statistic", "median")
    result["coil_statistic"] = models.get("coil_statistic", "median")
    result = add_single_objective_optima(
        calculate_cycle_cop(result, effective=True), names=("cycle_cop",)
    )
    result["is_selected"] = result.candidate_defrost_time.eq(result.cycle_cop_t_star)
    result["t_star"] = result.cycle_cop_t_star
    result["selected_point_in_training_domain"] = result.is_selected & result.cycle_cop_eligible
    result["t_star_model_supported"] = result.is_selected.any()
    return result


def compare_cycle(root, dataset, name, models, output):
    from defrost_decision.candidate_quantities import effective_candidate_cop

    baseline, challenger = models
    loader = DatasetLoader(dataset)
    frame = loader.load_cycle_original(name)
    frame.timestamp = pd.to_datetime(frame.timestamp)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    old = load_curve(root, name)
    record = loader.get_cycle_record(name)
    heating = pd.Timestamp(record["boundaries"]["heating_start"])
    recovery = pd.Timestamp(old.stable_heating_start.dropna().iloc[0])
    curves = {key: replay_variant(frame, old, model, heating) for key, model in models.items()}
    # Add a local 1 s grid around both independently selected optima, retaining the global grid.
    extra_times = []
    for curve in curves.values():
        if curve.is_selected.any():
            center = curve.loc[curve.is_selected, "candidate_defrost_time"].iloc[0]
            extra_times.extend(
                pd.date_range(
                    max(old.candidate_defrost_time.min(), center - pd.Timedelta(seconds=60)),
                    min(old.candidate_defrost_time.max(), center + pd.Timedelta(seconds=60)),
                    freq="s",
                )
            )
    extra_times = sorted(set(extra_times) - set(old.candidate_defrost_time))
    if extra_times:
        extra = effective_candidate_cop(
            frame,
            extra_times,
            recovery,
            heating,
            models[baseline],
            str(old.experiment_id.dropna().iloc[0]),
            preparation_heat="zero",
            prediction_mode="cross-fitted",
        )
        extra["experiment_id"] = old.experiment_id.dropna().iloc[0]
        extra["stable_heating_start"] = recovery
        old = (
            pd.concat([old, extra], ignore_index=True)
            .sort_values("candidate_defrost_time")
            .reset_index(drop=True)
        )
        curves = {key: replay_variant(frame, old, model, heating) for key, model in models.items()}
    current = extract_pre_defrost_features(
        frame, old.candidate_defrost_time, heating, frequency_statistic="current"
    ).compressor_frequency
    rows = []
    raw_time = curves[baseline].loc[curves[baseline].is_selected, "candidate_defrost_time"].iloc[0]
    for key, curve in curves.items():
        curve["current_frequency"] = current.to_numpy()
        curve["frequency_mismatch_hz"] = curve.compressor_frequency - current.to_numpy()
        params = models[key]["models"]["ridge_dynamic_state_8"]["event_electricity"]["folds"][
            str(old.experiment_id.dropna().iloc[0])
        ]
        features = params["feature_order"]
        filled = curve[features].fillna(dict(zip(features, params["imputer_median"], strict=True)))
        contributions = filled.diff() * (
            np.array(params["coefficients"]) / np.array(params["scaler_scale"])
        )
        np.testing.assert_allclose(
            contributions.sum(axis=1).iloc[1:],
            curve.defrost_event_electricity_kwh.diff().iloc[1:],
            atol=1e-12,
        )
        for feature in features:
            curve[f"contribution_{feature}"] = contributions[feature]
        curve["variant"] = key
        curve["cycle_name"] = name
        curve.to_csv(output / f"{name}_{key}_candidates.csv", index=False)
        best = curve.loc[curve.is_selected]
        row = dict(
            cycle_name=name,
            variant=key,
            supported_candidates=int(curve.cycle_cop_eligible.sum()),
            total_candidates=len(curve),
            frequency_mismatch_p95_hz=curve.loc[curve.cycle_cop_eligible, "frequency_mismatch_hz"]
            .abs()
            .quantile(0.95),
            frequency_mismatch_max_hz=curve.loc[curve.cycle_cop_eligible, "frequency_mismatch_hz"]
            .abs()
            .max(),
            t_star=best.candidate_defrost_time.iloc[0] if len(best) else pd.NaT,
        )
        if len(best):
            b = best.iloc[0]
            row.update(
                shift_from_baseline_seconds=(b.candidate_defrost_time - raw_time).total_seconds(),
                cop_max=b.cycle_cop,
                basin_1pct_minutes=b.cycle_cop_basin_1pct_width_minutes,
            )
            idx = best.index[0]
            row["peak_preceding_step_seconds"] = (
                (
                    curve.candidate_defrost_time.iloc[idx]
                    - curve.candidate_defrost_time.iloc[idx - 1]
                ).total_seconds()
                if idx
                else np.nan
            )
            row["peak_delta_E_kwh"] = curve.defrost_event_electricity_kwh.diff().iloc[idx]
        other = curves[challenger if key == baseline else baseline]
        if len(best):
            row["selected_also_supported_by_other"] = bool(
                other.loc[best.index[0], "cycle_cop_eligible"]
            )
            a = pd.Timestamp(best.cycle_cop_basin_1pct_start.iloc[0])
            b_end = pd.Timestamp(best.cycle_cop_basin_1pct_end.iloc[0])
            other_best = other.loc[other.is_selected]
            if len(other_best):
                c = pd.Timestamp(other_best.cycle_cop_basin_1pct_start.iloc[0])
                d = pd.Timestamp(other_best.cycle_cop_basin_1pct_end.iloc[0])
                row["own_basin_overlap_seconds"] = max(
                    0.0, (min(b_end, d) - max(a, c)).total_seconds()
                )
        rows.append(row)
    print(name, [(r["variant"], str(r["t_star"])) for r in rows], flush=True)
    return pd.DataFrame(rows), summarize_continuity(curves, name)


def summarize_continuity(curves, name):
    """Common support is diagnostic only; each selector has already acted independently."""
    baseline, challenger = curves
    common = curves[baseline].cycle_cop_eligible & curves[challenger].cycle_cop_eligible
    supported_pair = common & common.shift(fill_value=False)
    rows = []
    for key, curve in curves.items():
        dt = pd.to_datetime(curve.candidate_defrost_time).diff().dt.total_seconds()
        for scope in ["all", "common_supported"]:
            for step in [1, 10]:
                mask = dt.eq(step) & (supported_pair if scope == "common_supported" else True)
                delta = curve.defrost_event_electricity_kwh.diff().loc[mask].abs()
                rows.append(
                    dict(
                        cycle_name=name,
                        variant=key,
                        scope=scope,
                        step_seconds=step,
                        n=len(delta),
                        p50=delta.quantile(0.5),
                        p95=delta.quantile(0.95),
                        p99=delta.quantile(0.99),
                        maximum=delta.max(),
                        n_over002=int(delta.gt(0.02).sum()),
                        frequency_changed_current_unchanged=int(
                            (
                                mask
                                & curve.current_frequency.diff().eq(0)
                                & curve.compressor_frequency.diff().abs().gt(0)
                            ).sum()
                        ),
                    )
                )
    return pd.DataFrame(rows)


def compare_validation(runs, output):
    baseline, challenger = runs
    events = {
        k: pd.read_csv(p / "defrost_events.csv").set_index("event_id") for k, p in runs.items()
    }
    settings = {
        k: json.loads((p / "candidate_model_parameters.json").read_text()) for k, p in runs.items()
    }
    changed = {
        feature
        for setting, feature in [
            ("pressure_statistic", "evaporating_pressure"),
            ("frequency_statistic", "compressor_frequency"),
            ("coil_statistic", "coil_temperature"),
            ("coil_statistic", "ambient_coil_difference"),
        ]
        if settings[baseline].get(setting, "median") != settings[challenger].get(setting, "median")
    }
    common_columns = [c for c in events[baseline] if c not in changed]
    pd.testing.assert_frame_equal(
        events[baseline][common_columns], events[challenger][common_columns]
    )
    tables = {}
    for key, path in runs.items():
        t = pd.read_csv(path / "model_validation.csv")
        tables[key] = t.loc[t.model_name.eq("ridge_dynamic_state_8")].set_index("event_id")
    pd.testing.assert_index_equal(tables[baseline].index, tables[challenger].index)
    target = "defrost_event_electricity_observed_kwh"
    prediction = "defrost_event_electricity_prediction_kwh"
    paired = tables[baseline][["experiment_id", target]].copy()
    for key, table in tables.items():
        np.testing.assert_array_equal(paired[target], table[target])
        paired[f"error_{key}"] = table[prediction] - table[target]
        paired[f"supported_{key}"] = table.event_electricity_in_training_domain
    paired.to_csv(output / "paired_event_errors.csv")
    # Paired experiment-cluster bootstrap of saved LOEO errors; not refit uncertainty/equivalence.
    groups = paired.groupby("experiment_id").agg(
        n=(target, "size"),
        mae0=(f"error_{baseline}", lambda x: x.abs().sum()),
        mae1=(f"error_{challenger}", lambda x: x.abs().sum()),
    )
    draw = np.random.default_rng(20260921).integers(0, len(groups), (2000, len(groups)))
    difference = (groups.mae1.to_numpy()[draw] - groups.mae0.to_numpy()[draw]).sum(
        axis=1
    ) / groups.n.to_numpy()[draw].sum(axis=1)
    interval = np.quantile(difference, [0.025, 0.975])
    delta = paired[f"error_{challenger}"].abs().mean() - paired[f"error_{baseline}"].abs().mean()
    pd.DataFrame(
        [
            dict(
                delta_mae_kwh=delta,
                lower95=interval[0],
                upper95=interval[1],
                n_experiments=len(groups),
                bootstrap_replicates=2000,
            )
        ]
    ).to_csv(output / "paired_mae_uncertainty.csv", index=False)
    print(
        f"Paired MAE difference {challenger}-{baseline}:",
        delta,
        "cluster interval:",
        interval,
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root", type=Path, default=Path("output/test/heat_smoothing_pilot")
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=Path(
            "output/defrost_event_models/complete_defrost_events_zero/candidate_model_parameters.json"
        ),
    )
    parser.add_argument("--cycles", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--baseline-name", default="R0")
    parser.add_argument("--challenger-name", default="R1")
    parser.add_argument("--r0-run", type=Path)
    parser.add_argument("--r1-run", type=Path)
    parser.add_argument("--n-jobs", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        audit_features(
            args.candidate_root,
            args.cycles,
            json.loads(args.baseline_model.read_text()),
            args.output,
        )
        return
    from joblib import Parallel, delayed, parallel_config

    runs = {args.baseline_name: args.r0_run, args.challenger_name: args.r1_run}
    compare_validation(runs, args.output)
    models = {
        key: json.loads((path / "candidate_model_parameters.json").read_text())
        for key, path in runs.items()
    }
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(compare_cycle)(args.candidate_root, args.dataset, name, models, args.output)
            for name in args.cycles
        )
    pd.concat([r[0] for r in results], ignore_index=True).to_csv(
        args.output / "decisions.csv", index=False
    )
    pd.concat([r[1] for r in results], ignore_index=True).to_csv(
        args.output / "continuity.csv", index=False
    )


if __name__ == "__main__":
    main()
