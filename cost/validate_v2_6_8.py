"""Validate V2.6.8 transition models and diagnostic minima."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cost_curve import five_minute_support_runs
from .fit_v2_6_8 import (
    OUTCOME_TARGETS,
    OUTCOME_VALIDITY,
    fit_weighted_ridge,
    predict_from_artifact,
    valid_outcome_events,
)

_VALIDATION_COLUMNS = {
    "energy": ("E_T_prediction_kwh", "E_support_distance"),
    "heat": ("Q_T_prediction_kwh", "Q_support_distance"),
    "compressor_energy": ("E_comp_T_prediction_kwh", "Ecomp_support_distance"),
    "duration": ("D_T_prediction_minutes", "D_support_distance"),
}


def build_validation_table(events: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment, group in events.groupby("experiment_id", sort=False):
        for model_name, model_set in artifacts["models"].items():
            event_rows: dict[str, dict[str, object]] = {}
            for name in OUTCOME_TARGETS:
                if name not in model_set:
                    continue
                selected = valid_outcome_events(group, name)
                if selected.empty:
                    continue
                prediction = predict_from_artifact(model_set[name], selected, str(experiment))
                for position, (_, event) in enumerate(selected.iterrows()):
                    event_id = str(event["event_id"])
                    row = event_rows.setdefault(
                        event_id,
                        {
                            "event_id": event_id,
                            "cycle_name": event["cycle_name"],
                            "experiment_id": str(experiment),
                            "model_name": model_name,
                            "event_valid": bool(event.get("event_valid", False)),
                            "event_invalid_reason": event.get("event_invalid_reason", ""),
                            **{
                                column: bool(event.get(column, event.get("event_valid", False)))
                                for column in OUTCOME_VALIDITY.values()
                            },
                        },
                    )
                    target = OUTCOME_TARGETS[name]
                    prediction_column, support_column = _VALIDATION_COLUMNS[name]
                    row[target] = event.get(target, np.nan)
                    row[prediction_column] = prediction.iloc[position]["prediction"]
                    row[support_column] = prediction.iloc[position]["support_distance"]
            rows.extend(event_rows.values())
    any_valid = pd.Series(False, index=events.index)
    for name, target in OUTCOME_TARGETS.items():
        if target in events:
            any_valid.loc[valid_outcome_events(events, name).index] = True
    for _, event in events.loc[~any_valid].iterrows():
        rows.append(
            {
                "event_id": event.get("event_id", event["cycle_name"]),
                "cycle_name": event["cycle_name"],
                "experiment_id": event["experiment_id"],
                "model_name": "excluded_event",
                "event_valid": False,
                "event_invalid_reason": event.get("event_invalid_reason", ""),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_minima(
    curves: pd.DataFrame,
    events: pd.DataFrame,
    energy_artifact: dict[str, Any],
    heat_artifact: dict[str, Any],
    *,
    replicates: int = 200,
    seed: int = 268,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    valid = events.loc[events["event_valid"].fillna(False)].copy()
    features = tuple(str(value) for value in energy_artifact["feature_order"])
    minima: dict[str, list[pd.Timestamp]] = {
        str(name): [] for name in curves["cycle_name"].astype(str).unique()
    }
    for _ in range(replicates):
        for heldout, heldout_curves in curves.groupby("experiment_id", sort=False):
            heldout = str(heldout)
            available = sorted(set(valid["experiment_id"].astype(str)) - {heldout})
            if len(available) < 2 or heldout not in energy_artifact["folds"]:
                continue
            sampled = rng.choice(available, size=len(available), replace=True)
            parts = []
            for draw, experiment in enumerate(sampled):
                part = valid.loc[valid["experiment_id"].astype(str).eq(str(experiment))].copy()
                part["experiment_id"] = f"draw_{draw}"
                parts.append(part)
            training = pd.concat(parts, ignore_index=True)
            energy_model = fit_weighted_ridge(
                training,
                features,
                "E_T_observed_kwh",
                alpha=float(energy_artifact["folds"][heldout]["alpha"]),
            )
            heat_model = fit_weighted_ridge(
                training,
                features,
                "Q_T_observed_kwh",
                alpha=float(heat_artifact["folds"][heldout]["alpha"]),
            )
            candidates = heldout_curves.copy()
            energy = energy_model.predict(candidates)
            heat = heat_model.predict(candidates)
            e_supported = (
                energy_model.support_distance(candidates) <= energy_model.support_threshold
            )
            q_supported = heat_model.support_distance(candidates) <= heat_model.support_threshold
            numerator = candidates["heating_energy_kwh"].to_numpy(dtype=float) + energy
            denominator = candidates["heating_heat_kwh"].to_numpy(dtype=float) + heat
            inverse = np.divide(
                numerator,
                denominator,
                out=np.full(len(candidates), np.nan),
                where=(numerator > 0) & (denominator > 0.01),
            )
            candidates["bootstrap_inverse_cop"] = inverse
            candidates["bootstrap_base"] = (
                candidates["heating_measurement_valid"].fillna(False).to_numpy()
                & candidates["pre_action_window_valid"].fillna(False).to_numpy()
                & e_supported
                & q_supported
                & np.isfinite(inverse)
            )
            for cycle_name, curve in candidates.groupby("cycle_name", sort=False):
                eligible = five_minute_support_runs(
                    curve["candidate_time"], curve["bootstrap_base"]
                )
                if eligible.any():
                    position = curve.index[
                        eligible
                        & curve["bootstrap_inverse_cop"].eq(
                            curve.loc[eligible, "bootstrap_inverse_cop"].min()
                        )
                    ][0]
                    minima[str(cycle_name)].append(
                        pd.Timestamp(curve.loc[position, "candidate_time"])
                    )
    rows = []
    for cycle_name, curve in curves.groupby("cycle_name", sort=False):
        values = minima[str(cycle_name)]
        numeric = np.asarray([value.value for value in values], dtype=np.int64)
        basin_start = pd.to_datetime(curve["basin_5pct_start"].iloc[0], errors="coerce")
        basin_end = pd.to_datetime(curve["basin_5pct_end"].iloc[0], errors="coerce")
        in_basin = (
            float(np.mean([basin_start <= value <= basin_end for value in values]))
            if values and pd.notna(basin_start) and pd.notna(basin_end)
            else np.nan
        )
        rows.append(
            {
                "cycle_name": cycle_name,
                "experiment_id": curve["experiment_id"].iloc[0],
                "repeat_count": replicates,
                "seed": seed,
                "valid_minimum_count": len(values),
                "valid_minimum_fraction": len(values) / replicates if replicates else np.nan,
                "argmin_median_time": pd.Timestamp(int(np.median(numeric)))
                if numeric.size
                else pd.NaT,
                "argmin_q25_time": pd.Timestamp(int(np.quantile(numeric, 0.25)))
                if numeric.size
                else pd.NaT,
                "argmin_q75_time": pd.Timestamp(int(np.quantile(numeric, 0.75)))
                if numeric.size
                else pd.NaT,
                "argmin_in_original_5pct_basin_fraction": in_basin,
                "support_policy": "refit_95pct_cross_experiment_distance_each_replicate",
            }
        )
    return pd.DataFrame(rows)
