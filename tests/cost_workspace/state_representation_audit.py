"""Read-only R2 support provenance and full-catalog, one-second feature audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from dataset_tools import DatasetLoader
from defrost_event_models.ridge_models import predict_with_model_parameters
from defrost_event_models.training_data import (
    MAXIMUM_GAP_SECONDS,
    build_candidate_boundaries,
    catalog,
    extract_pre_defrost_features,
)

BASES = {
    "ambient_temperature": ["ambient_temperature"],
    "mean_water_temperature": ["water_in_temperature", "water_out_temperature"],
    "setpoint_outlet_difference": ["water_temperature_setpoint", "water_out_temperature"],
    "ambient_coil_difference": ["ambient_temperature", "coil_temperature"],
    "evaporating_pressure": ["evaporating_pressure"],
    "compressor_frequency": ["compressor_frequency"],
    "heating_elapsed_minutes": [],
    "evaporating_pressure_slope_5m": ["evaporating_pressure"],
}


def read_models(run):
    return json.loads((run / "candidate_model_parameters.json").read_text())


def parameters(models, experiment):
    return models["models"]["ridge_dynamic_state_8"]["event_electricity"]["folds"][experiment]


def load_raw(loader, name):
    frame = loader.load_cycle_original(name).copy()
    frame.timestamp = pd.to_datetime(frame.timestamp)
    return frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def nearest_components(params, values):
    features = params["feature_order"]
    x = values[features].fillna(dict(zip(features, params["imputer_median"], strict=True)))
    z = (x.to_numpy() - np.array(params["scaler_mean"])) / np.array(params["scaler_scale"])
    references = np.array(params["training_standardized_references"])
    squared = np.square(z[:, None, :] - references[None, :, :])
    nearest = squared.sum(axis=2).argmin(axis=1)
    parts = squared[np.arange(len(values)), nearest]
    prediction = predict_with_model_parameters(params, values)
    np.testing.assert_allclose(np.sqrt(parts.sum(axis=1)), prediction.support_distance, atol=1e-12)
    return parts, nearest, prediction


def support_provenance(baseline, model, curves_root, output, dataset):
    curves = {
        k: pd.read_csv(curves_root / f"frost_cycle_000036_{k}_candidates.csv") for k in ["R1", "R2"]
    }
    lost = curves["R1"].cycle_cop_eligible & ~curves["R2"].cycle_cop_eligible
    assert lost.sum() == 229, "The requested frozen comparison must retain its 229 losses"
    rows = (
        curves["R2"].loc[lost, ["candidate_defrost_time", "experiment_id"]].reset_index(drop=True)
    )
    experiment = rows.experiment_id.iloc[0]
    params = {"R1": parameters(baseline, experiment), "R2": parameters(model, experiment)}
    features = params["R2"]["feature_order"]
    normalized = {}
    for key in ["R1", "R2"]:
        values = curves[key].loc[lost].reset_index(drop=True)
        p = params[key]
        parts, nearest, prediction = nearest_components(p, values)
        normalized[key] = parts / p["support_threshold"] ** 2
        rows[f"{key}_distance"] = prediction.support_distance
        rows[f"{key}_threshold"] = p["support_threshold"]
        rows[f"{key}_score"] = prediction.support_distance / p["support_threshold"]
        rows[f"{key}_nearest_event"] = np.array(p["training_event_ids"])[nearest]
        rows[f"{key}_dominant_distance_feature"] = np.array(features)[parts.argmax(axis=1)]
        for j, feature in enumerate(features):
            rows[f"{key}_squared_distance_{feature}"] = parts[:, j]
        rows[f"{key}_frequency"] = values.compressor_frequency
        j = features.index("compressor_frequency")
        rows[f"{key}_nearest_frequency"] = (
            np.array(p["training_standardized_references"])[nearest, j] * p["scaler_scale"][j]
            + p["scaler_mean"][j]
        )
        rows[f"{key}_frequency_scale"] = p["scaler_scale"][j]
    increase = normalized["R2"] - normalized["R1"]
    rows["dominant_normalized_distance_increase"] = np.array(features)[increase.argmax(axis=1)]
    for j, feature in enumerate(features):
        rows[f"delta_normalized_squared_distance_{feature}"] = increase[:, j]
    np.testing.assert_allclose(increase.sum(axis=1), rows.R2_score**2 - rows.R1_score**2)
    # Fixed R1 geometry: isolate candidate-frequency substitution, not a new selector.
    changed = curves["R1"].loc[lost].reset_index(drop=True).copy()
    changed["compressor_frequency"] = rows.R2_frequency
    rows["R1_geometry_current_frequency_score"] = (
        predict_with_model_parameters(params["R1"], changed).support_distance
        / params["R1"]["support_threshold"]
    )
    # Exact but order-dependent decomposition: candidate, reference, scale, threshold.
    p1, p2 = params["R1"], params["R2"]
    references_raw = np.array(p2["training_standardized_references"]) * np.array(
        p2["scaler_scale"]
    ) + np.array(p2["scaler_mean"])
    mixed = dict(p1)
    mixed["training_standardized_references"] = (
        (references_raw - np.array(p1["scaler_mean"])) / np.array(p1["scaler_scale"])
    ).tolist()
    rows["new_references_old_scale_threshold_score"] = (
        predict_with_model_parameters(mixed, changed).support_distance / p1["support_threshold"]
    )
    rows["new_geometry_old_threshold_score"] = rows.R2_distance / p1["support_threshold"]
    nearest = nearest_components(p2, curves["R2"].loc[lost])[1]
    for j, feature in enumerate(features):
        rows[f"value_{feature}"] = curves["R2"].loc[lost, feature].to_numpy()
        rows[f"training_min_{feature}"] = references_raw[:, j].min()
        rows[f"training_max_{feature}"] = references_raw[:, j].max()
        rows[f"nearest_value_{feature}"] = references_raw[nearest, j]
    loader = DatasetLoader(dataset)
    frame = load_raw(loader, "frost_cycle_000036")
    heating = pd.Timestamp(
        loader.get_cycle_record("frost_cycle_000036")["boundaries"]["heating_start"]
    )
    current, age = latest_values(
        frame, pd.DatetimeIndex(rows.candidate_defrost_time), "compressor_frequency", heating
    )
    np.testing.assert_array_equal(current, rows.R2_frequency)
    rows["frequency_observation_age_seconds"] = age
    rows.to_csv(output / "cycle36_support_loss.csv", index=False)
    print(
        "Support provenance:",
        rows.groupby("dominant_normalized_distance_increase").size().to_dict(),
        "R1/R2 thresholds",
        p1["support_threshold"],
        p2["support_threshold"],
        flush=True,
    )


def latest_values(frame, times, name, heating):
    valid = frame.loc[np.isfinite(pd.to_numeric(frame[name], errors="coerce")), ["timestamp", name]]
    stamps = pd.DatetimeIndex(valid.timestamp)
    position = stamps.searchsorted(times, side="left") - 1
    if valid.empty:
        return np.full(len(times), np.nan), np.full(len(times), np.nan)
    safe = np.maximum(position, 0)
    age = (times.view("i8") - stamps.view("i8")[safe]) / 1e9
    available = (position >= 0) & (age <= MAXIMUM_GAP_SECONDS) & (stamps[safe] >= heating)
    return np.where(available, valid[name].to_numpy()[safe], np.nan), np.where(
        available, age, np.nan
    )


def audit_cycle(dataset, record, models, output):
    name = record["cycle_name"]
    path = output / f"{name}.parquet"
    if path.exists():
        return path
    frame = load_raw(DatasetLoader(dataset), name)
    heating = pd.Timestamp(record["heating_start"])
    times = pd.DatetimeIndex(
        build_candidate_boundaries(
            name,
            record["experiment_id"],
            heating,
            pd.Timestamp(record["defrost_preparation_start"]),
            step_seconds=1,
        ).candidate_defrost_time
    )
    result = extract_pre_defrost_features(
        frame,
        times,
        heating,
        pressure_statistic=models["pressure_statistic"],
        frequency_statistic=models["frequency_statistic"],
        coil_statistic=models.get("coil_statistic", "median"),
    )
    p = parameters(models, record["experiment_id"])
    prediction = predict_with_model_parameters(p, result)
    result["candidate_defrost_time"] = times
    result["cycle_name"] = name
    result["experiment_id"] = record["experiment_id"]
    result["prediction_kwh"] = prediction.prediction
    result["support_score"] = prediction.support_distance / prediction.support_threshold
    names = sorted({name for bases in BASES.values() for name in bases})
    for raw in names:
        current, age = latest_values(frame, times, raw, heating)
        result[f"current_{raw}"] = current
        result[f"age_{raw}"] = age
    filled = result[p["feature_order"]].fillna(
        dict(zip(p["feature_order"], p["imputer_median"], strict=True))
    )
    contributions = filled.diff() * (np.array(p["coefficients"]) / np.array(p["scaler_scale"]))
    np.testing.assert_allclose(
        contributions.sum(axis=1).iloc[1:], prediction.prediction.diff().iloc[1:], atol=1e-12
    )
    for feature in p["feature_order"]:
        result[f"contribution_{feature}"] = contributions[feature]
        bases = BASES[feature]
        stable = pd.Series(bool(bases), index=result.index)
        observed = stable.copy()
        for raw in bases:
            stable &= result[f"current_{raw}"].diff().eq(0)
            observed &= result[f"age_{raw}"].le(1) & result[f"age_{raw}"].shift().le(1)
        result[f"raw_stable_{feature}"] = stable
        result[f"raw_observed_pair_{feature}"] = observed
    result["valid_pair"] = result.pre_defrost_feature_window_valid & (
        result.pre_defrost_feature_window_valid.shift(fill_value=False)
    )
    result["supported_pair"] = (
        result.valid_pair & result.support_score.le(1) & (result.support_score.shift().le(1))
    )
    result["imputed_pair"] = filled.notna().all(axis=1) & (
        result[p["feature_order"]].isna().any(axis=1)
        | result[p["feature_order"]].isna().any(axis=1).shift(fill_value=False)
    )
    result.to_parquet(path, index=False)
    print(name, len(result), "candidates", flush=True)
    return path


def summarize(paths, model, output):
    tables = [pd.read_parquet(path) for path in paths]
    all_rows = pd.concat(tables, ignore_index=True)
    features = next(
        iter(model["models"]["ridge_dynamic_state_8"]["event_electricity"]["folds"].values())
    )["feature_order"]
    summaries, top = [], []
    for feature in features:
        value = all_rows[f"contribution_{feature}"].abs()
        for scope in ["valid_pair", "supported_pair"]:
            mask = all_rows[scope]
            x = value.loc[mask]
            stable = (
                mask & all_rows[f"raw_stable_{feature}"] & all_rows[f"raw_observed_pair_{feature}"]
            )
            summaries.append(
                dict(
                    feature=feature,
                    scope=scope,
                    n=len(x),
                    p95=x.quantile(0.95),
                    p99=x.quantile(0.99),
                    maximum=x.max(),
                    raw_stable_observed_max=value.loc[stable].max(),
                    nonzero_raw_stable_observed=int((stable & value.gt(1e-12)).sum()),
                )
            )
            for kind, selected in [("largest", mask), ("raw_stable_observed", stable)]:
                indices = value.loc[selected & value.gt(1e-12)].nlargest(20).index
                rows = all_rows.loc[indices].copy()
                rows["rank_feature"] = feature
                rows["rank_scope"] = scope
                rows["rank_kind"] = kind
                rows["rank_abs_contribution_kwh"] = value.loc[indices]
                rows["previous_feature_value"] = all_rows.loc[indices - 1, feature].to_numpy()
                rows["feature_value"] = all_rows.loc[indices, feature].to_numpy()
                for raw in BASES[feature]:
                    rows[f"previous_current_{raw}"] = all_rows.loc[
                        indices - 1, f"current_{raw}"
                    ].to_numpy()
                top.append(rows)
    pd.DataFrame(summaries).to_csv(output / "feature_jump_summary.csv", index=False)
    pd.concat(top, ignore_index=True).to_csv(output / "feature_top20.csv", index=False)
    cycles = all_rows.groupby(["cycle_name", "experiment_id"]).agg(
        candidates=("candidate_defrost_time", "size"),
        valid_pairs=("valid_pair", "sum"),
        supported_pairs=("supported_pair", "sum"),
        current_frequency_min=("current_compressor_frequency", "min"),
        current_frequency_max=("current_compressor_frequency", "max"),
        water_setpoint_min=("current_water_temperature_setpoint", "min"),
        water_setpoint_max=("current_water_temperature_setpoint", "max"),
    )
    cycles.to_csv(output / "cycle_audit_summary.csv")
    # Marginal observed frequency coverage is not evidence for joint-state support.
    training_ids = set(parameters(model, "exp_20260724")["training_experiment_ids"])
    observed = all_rows.loc[
        all_rows.experiment_id.isin(training_ids)
        & all_rows.pre_defrost_feature_window_valid
        & all_rows.current_compressor_frequency.between(51, 58)
        & all_rows.age_compressor_frequency.le(1)
    ]
    observed.groupby(["cycle_name", "experiment_id"]).agg(
        observed_seconds=("candidate_defrost_time", "size"),
        frequency_min=("current_compressor_frequency", "min"),
        frequency_max=("current_compressor_frequency", "max"),
    ).to_csv(output / "training_trajectory_low_frequency.csv")
    print("Full audit complete:", len(paths), "cycles", len(all_rows), "candidates", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--curves-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--support-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = read_models(args.model_run)
    if args.baseline_run is not None:
        support_provenance(
            read_models(args.baseline_run), models, args.curves_root, args.output, args.dataset
        )
    folds = models["models"]["ridge_dynamic_state_8"]["event_electricity"]["folds"]
    rows = []
    for experiment, p in folds.items():
        assert experiment not in p["training_experiment_ids"]
        j = p["feature_order"].index("compressor_frequency")
        rows.append(
            dict(
                experiment_id=experiment,
                coefficient=p["coefficients"][j],
                scale=p["scaler_scale"][j],
                alpha=p["alpha"],
                sensitivity_kwh_per_hz=p["coefficients"][j] / p["scaler_scale"][j],
            )
        )
    pd.DataFrame(rows).to_csv(args.output / "frequency_fold_sensitivity.csv", index=False)
    if args.support_only:
        return
    cohort = catalog(DatasetLoader(args.dataset))
    cohort["audit_reason"] = np.select(
        [
            ~cohort.experiment_id.isin(folds),
            cohort.heating_start.isna() | cohort.defrost_preparation_start.isna(),
            (cohort.defrost_preparation_start - cohort.heating_start).dt.total_seconds().lt(600),
        ],
        ["no_heldout_model", "missing_action_boundary", "shorter_than_candidate_start"],
        default="included",
    )
    cohort[["cycle_name", "experiment_id", "status", "audit_reason"]].to_csv(
        args.output / "audit_cohort.csv", index=False
    )
    with parallel_config(backend="loky", inner_max_num_threads=1):
        paths = Parallel(n_jobs=args.n_jobs)(
            delayed(audit_cycle)(args.dataset, row, models, args.output)
            for row in cohort.loc[cohort.audit_reason.eq("included")].to_dict("records")
        )
    summarize(paths, models, args.output)


if __name__ == "__main__":
    main()
