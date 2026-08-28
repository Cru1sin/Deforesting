#!/usr/bin/env python3
"""Audit empirical defrost tickets and summarize optimal windows, COP and RGB."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from frost_analysis.cost.core import (
    build_partial_pool_curves,
    integrate_energy_kwh,
    leave_one_event_out_partial_pool,
    water_side_heating_kw,
)
from frost_analysis.dataset.core import render_publication_asset
from frost_analysis.dataset.images import (
    materialize_cycle_image_members,
    scan_cycle_images,
)
from frost_analysis.dataset.loader import DatasetLoader
from frost_analysis.dataset.metadata import following_cycle_names
from frost_analysis.figures.visualization import (
    match_decision_rgb_images,
    render_decision_publication,
)
from frost_analysis.labels.cost import complete_observed_cycle_names

RAW_COLUMNS = [
    "timestamp",
    "water_flow",
    "water_in_temperature",
    "water_out_temperature",
    "water_temperature_setpoint",
    "power_total",
    "heating_capacity",
    "ambient_temperature",
    "environment_relative_humidity",
    "compressor_frequency",
    "compressor_frequency_setpoint",
    "compressor_power",
    "fan_speed",
    "fan_current",
    "exv_opening",
    "evaporating_temperature",
    "evaporating_pressure",
    "coil_temperature",
    "suction_temperature",
    "discharge_temperature",
    "condensing_pressure",
    "p1__PIStep<1_00>",
    "p1__PowPI<1_00>",
    "p1__StepMax<1_00>",
    "p1__限频代号'1_00",
    "p1__频率状态<1_00>",
]
PI_STATE_COLUMNS = {
    "p1__PIStep<1_00>": "pi_step",
    "p1__PowPI<1_00>": "pi_power",
    "p1__StepMax<1_00>": "pi_step_limit",
    "p1__限频代号'1_00": "compressor_limit_code",
    "p1__频率状态<1_00>": "compressor_frequency_state",
}
WINDOW_FEATURES = [
    "q_heating_kw",
    "cop",
    "power_total",
    "heating_capacity",
    "water_flow",
    "water_delta_temperature",
    "water_temperature_setpoint",
    "ambient_temperature",
    "environment_relative_humidity",
    "water_in_temperature",
    "water_out_temperature",
    "compressor_frequency",
    "compressor_frequency_setpoint",
    "compressor_power",
    "fan_speed",
    "fan_current",
    "exv_opening",
    "evaporating_temperature",
    "evaporating_pressure",
    "coil_temperature",
    "suction_temperature",
    "discharge_temperature",
    "condensing_pressure",
    *PI_STATE_COLUMNS.values(),
]
RECOVERY_STATE_FEATURES = [
    "water_temperature_setpoint",
    "water_in_temperature",
    "water_out_temperature",
    "water_flow",
    "coil_temperature",
    "evaporating_temperature",
    "suction_temperature",
    "discharge_temperature",
    "evaporating_pressure",
    "condensing_pressure",
    "ambient_temperature",
    "environment_relative_humidity",
    "compressor_frequency",
    "compressor_frequency_setpoint",
    "compressor_power",
    *PI_STATE_COLUMNS.values(),
]
RECOVERY_OUTCOMES = [
    "recovery_electricity_kwh",
    "recovery_water_heat_kwh",
    "recovery_duration_minutes",
]
RECOVERY_RATE_OUTCOMES = [
    "recovery_mean_power_kw",
    "recovery_mean_water_heat_kw",
]
STATE_FEATURES = [
    "minutes_from_stable",
    "cop",
    "ambient_temperature",
    "environment_relative_humidity",
    "water_in_temperature",
    "compressor_frequency",
    "evaporating_temperature",
    "coil_temperature",
]
DYNAMIC_BASE_FEATURES = [
    "q_heating_kw",
    "cop",
    "power_total",
    "water_flow",
    "water_delta_temperature",
    "ambient_temperature",
    "environment_relative_humidity",
    "water_in_temperature",
    "water_out_temperature",
    "compressor_frequency",
    "evaporating_temperature",
    "coil_temperature",
]
DYNAMIC_FEATURES = [
    "minutes_from_stable",
    *DYNAMIC_BASE_FEATURES,
    *(f"{column}_slope_per_min" for column in DYNAMIC_BASE_FEATURES),
    *(f"{column}_iqr" for column in DYNAMIC_BASE_FEATURES),
]
OUTCOMES = {
    "cost": "equivalent_cost_kwh",
    "duration": "duration_minutes",
    "electricity": "electricity_kwh",
    "thermal_shortfall": "thermal_shortfall_kwh",
}
STRATEGIES = (
    "mean",
    "time",
    "state",
    "dynamic",
    "nonlinear",
    "component",
    "partial_pool",
)
MARKUS_POWER_COEFFICIENTS_KW = np.array(
    [480.0, -40.0, -7.9, -5.9, 308.0, -45.0]
) / 1000.0
PREDEFROST_SENSOR_BASES = [
    "ambient_temperature",
    "environment_relative_humidity",
    "water_in_temperature",
    "water_out_temperature",
    "water_flow",
    "water_delta_temperature",
    "compressor_frequency",
    "evaporating_temperature",
    "power_total",
    "q_heating_kw",
    "cop",
]
PREDEFROST_SENSOR_FEATURES = [
    *PREDEFROST_SENSOR_BASES,
    *(
        f"{feature}_{suffix}"
        for feature in PREDEFROST_SENSOR_BASES
        for suffix in ("iqr", "slope_per_min")
    ),
]
PREPARATION_SENSOR_FEATURES = [
    "water_flow_slope_per_min",
    "compressor_frequency_slope_per_min",
    "compressor_frequency_iqr",
    "evaporating_temperature_slope_per_min",
    "power_total_slope_per_min",
]
LITERATURE_SENSOR_FEATURES = [
    "coil_temperature_slope_per_min",
    "coil_temperature_iqr",
    "evaporator_capacity_ratio_clean",
    "q_heating_kw",
    "q_heating_ratio_clean",
    "q_heating_kw_slope_per_min",
    "evaporating_pressure",
    "cop",
    "cop_ratio_clean",
    "cop_slope_per_min",
    "fan_current",
    "fan_current_iqr",
    "fan_current_slope_per_min",
    "compressor_frequency",
    "compressor_frequency_slope_per_min",
    "compressor_power",
    "compressor_power_slope_per_min",
    "power_total",
    "power_total_slope_per_min",
    "ambient_temperature",
    "environment_relative_humidity",
]
LITERATURE_AUDIT_FEATURES = [
    "evaporating_temperature",
    "evaporating_temperature_slope_per_min",
]
FIXED_LITERATURE_FEATURES = ["evaporating_pressure", "cop", "power_total"]
FIXED_LITERATURE_COMBINATION = "__fixed_pe_cop_power_total__"
FIXED_SCREEN_MIN_EVENT_IMPROVEMENT_PCT = 5.0
FIXED_SCREEN_MIN_MACRO_IMPROVEMENT_PCT = 5.0
FIXED_SCREEN_MIN_IMPROVED_EXPERIMENTS = 8
PREPARATION_NETWORK_FEATURES = [
    "t3_prepreparation_c",
    "evaporating_pressure",
    "cop",
    "compressor_power",
    "evaporator_capacity_ratio_clean",
    "fan_current",
    "ambient_temperature",
]
PREPARATION_NETWORK_HIDDEN_LAYERS = (4,)
PREPARATION_NETWORK_RIDGE_ALPHA = 1.0
PREPARATION_NETWORK_MLP_ALPHA = 1.0
PREPARATION_NETWORK_SEED = 20260821

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def preceding_features(
    frame: pd.DataFrame,
    end: pd.Timestamp,
    *,
    seconds: int = 60,
    include_dynamics: bool = False,
) -> dict[str, float]:
    """Return raw levels, and optionally dynamics, before one candidate action."""
    values = (
        frame
        if {"q_heating_kw", "cop", "water_delta_temperature"} <= set(frame)
        else _prepare_raw_features(frame)
    )
    return window_features(
        values,
        end - pd.Timedelta(seconds=seconds),
        end,
        WINDOW_FEATURES,
        include_dynamics=include_dynamics,
    )


def window_features(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    columns: list[str],
    *,
    include_dynamics: bool = False,
) -> dict[str, float]:
    """Summarize sensor levels and optional slopes within one time window."""
    window = frame.loc[frame["timestamp"].ge(start) & frame["timestamp"].lt(end)]
    result = {}
    for column in columns:
        if column not in window:
            result[column] = float("nan")
            if include_dynamics:
                result[f"{column}_iqr"] = float("nan")
                result[f"{column}_slope_per_min"] = float("nan")
            continue
        numeric = pd.to_numeric(window[column], errors="coerce")
        valid = numeric.notna()
        observed = numeric.loc[valid]
        result[column] = float(observed.median()) if not observed.empty else float("nan")
        if not include_dynamics:
            continue
        result[f"{column}_iqr"] = (
            float(observed.quantile(0.75) - observed.quantile(0.25))
            if not observed.empty
            else float("nan")
        )
        elapsed = (
            (window.loc[valid, "timestamp"] - window.loc[valid, "timestamp"].min())
            .dt.total_seconds()
            .to_numpy(dtype=float)
            / 60.0
        )
        result[f"{column}_slope_per_min"] = (
            float(np.polyfit(elapsed, observed.to_numpy(dtype=float), 1)[0])
            if len(observed) >= 2 and np.ptp(elapsed) > 0
            else float("nan")
        )
    return result


def _prepare_raw_features(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.copy()
    values["timestamp"] = pd.to_datetime(values["timestamp"], errors="coerce")
    values["q_heating_kw"] = water_side_heating_kw(values)
    power = pd.to_numeric(values["power_total"], errors="coerce")
    values["cop"] = values["q_heating_kw"] / power.where(power.gt(0))
    values["water_delta_temperature"] = (
        pd.to_numeric(values["water_out_temperature"], errors="coerce")
        - pd.to_numeric(values["water_in_temperature"], errors="coerce")
    )
    for source, target in PI_STATE_COLUMNS.items():
        values[target] = (
            pd.to_numeric(values[source], errors="coerce")
            if source in values
            else np.nan
        )
    return values


def _ridge_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    outcome: str,
) -> np.ndarray:
    usable = [column for column in features if train[column].notna().any()]
    if not usable:
        return np.repeat(train[outcome].mean(), len(test))
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[usable], train[outcome])
    return model.predict(test[usable])


def _nonlinear_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    outcome: str,
) -> np.ndarray:
    usable = [column for column in features if train[column].notna().any()]
    if not usable:
        return np.repeat(train[outcome].mean(), len(test))
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=8,
            l2_regularization=1.0,
            random_state=0,
        ),
    )
    model.fit(train[usable], train[outcome])
    return model.predict(test[usable])


def leave_one_experiment_out_ticket_predictions(
    events: pd.DataFrame,
    state_features: list[str],
    dynamic_features: list[str] | None = None,
) -> pd.DataFrame:
    """Predict held-out ticket outcomes using only other experiments."""
    predictions = []
    dynamic_features = dynamic_features or state_features
    for experiment in sorted(events["experiment_id"].unique()):
        train = events.loc[~events["experiment_id"].eq(experiment)]
        test = events.loc[events["experiment_id"].eq(experiment)].copy()
        test["training_event_count"] = len(train)
        for label, outcome in OUTCOMES.items():
            test[f"predicted_mean_{label}"] = train[outcome].mean()
            test[f"predicted_time_{label}"] = _ridge_predict(
                train, test, ["minutes_from_stable"], outcome
            )
            test[f"predicted_state_{label}"] = _ridge_predict(
                train, test, state_features, outcome
            )
            test[f"predicted_dynamic_{label}"] = _ridge_predict(
                train, test, dynamic_features, outcome
            )
            test[f"predicted_nonlinear_{label}"] = _nonlinear_predict(
                train, test, dynamic_features, outcome
            )
        thermal_weight = (
            (train["equivalent_cost_kwh"] - train["electricity_kwh"])
            / train["thermal_shortfall_kwh"].where(train["thermal_shortfall_kwh"].gt(0))
        ).median()
        test["predicted_component_cost"] = (
            test["predicted_dynamic_electricity"]
            + thermal_weight * test["predicted_dynamic_thermal_shortfall"]
        )
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True)


def leave_one_experiment_out_defrost_predictions(events: pd.DataFrame) -> pd.DataFrame:
    """Predict defrost electricity while leaving each complete experiment out."""
    events = events.copy()
    heat_base_features = [
        "water_in_temperature",
        "water_out_temperature",
        "rule_defrost_duration_minutes",
        "coil_temperature",
        "evaporating_pressure",
    ]
    if "defrost_absorbed_heat_kwh" in events:
        for feature in heat_base_features:
            events[f"{feature}_squared"] = events[feature].pow(2)
    predictions = []
    for experiment in sorted(events["experiment_id"].unique()):
        train = events.loc[
            ~events["experiment_id"].eq(experiment)
            & events["defrost_duration_minutes"].gt(0)
            & events["defrost_electricity_kwh"].notna()
        ]
        test = events.loc[events["experiment_id"].eq(experiment)].copy()
        duration = train["defrost_duration_minutes"]
        mean_duration = float(duration.mean())
        mean_power = float(
            train["defrost_electricity_kwh"].sum() / (duration.sum() / 60.0)
        )
        test["predicted_fixed_defrost_electricity"] = train[
            "defrost_electricity_kwh"
        ].mean()
        test["predicted_known_duration_defrost_electricity"] = (
            mean_power * test["defrost_duration_minutes"] / 60.0
        )
        predicted_duration = _ridge_predict(
            train, test, ["coil_temperature"], "defrost_duration_minutes"
        )
        predicted_duration = np.where(
            test["coil_temperature"].notna(), predicted_duration, mean_duration
        )
        test["predicted_t3_duration_minutes"] = np.clip(
            predicted_duration, duration.min(), duration.max()
        )
        test["predicted_t3_rule_defrost_electricity"] = (
            mean_power * test["predicted_t3_duration_minutes"] / 60.0
        )
        test["training_mean_defrost_power_kw"] = mean_power
        if "defrost_absorbed_heat_kwh" in events:
            heat_features = {
                "mean": [],
                "water": ["water_in_temperature", "water_out_temperature"],
                "water_duration": [
                    "water_in_temperature",
                    "water_out_temperature",
                    "rule_defrost_duration_minutes",
                ],
                "water_duration_t3": [
                    "water_in_temperature",
                    "water_out_temperature",
                    "rule_defrost_duration_minutes",
                    "coil_temperature",
                ],
                "water_duration_pe": [
                    "water_in_temperature",
                    "water_out_temperature",
                    "rule_defrost_duration_minutes",
                    "evaporating_pressure",
                ],
                "water_duration_t3_pe": [
                    *heat_base_features,
                ],
                "water_duration_t3_pe_squared": [
                    *heat_base_features,
                    *(f"{feature}_squared" for feature in heat_base_features),
                ],
            }
            for label, features in heat_features.items():
                test[f"predicted_{label}_defrost_absorbed_heat"] = _ridge_predict(
                    train, test, features, "defrost_absorbed_heat_kwh"
                )
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True)


def defrost_model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize held-out defrost electricity and absorbed-heat errors."""
    columns = {
        "fixed": "predicted_fixed_defrost_electricity",
        "known_duration": "predicted_known_duration_defrost_electricity",
        "t3_rule": "predicted_t3_rule_defrost_electricity",
    }
    rows = []
    for experiment, values in predictions.groupby("experiment_id", sort=True):
        for strategy, prediction in columns.items():
            rows.append(
                {
                    "experiment_id": experiment,
                    "strategy": strategy,
                    "outcome": "defrost_electricity",
                    "protocol": "leave_one_experiment_out",
                    "mae": (values[prediction] - values["defrost_electricity_kwh"])
                    .abs()
                    .mean(),
                    "mse": (values[prediction] - values["defrost_electricity_kwh"])
                    .pow(2)
                    .mean(),
                    "rmse": float(
                        np.sqrt(
                            (values[prediction] - values["defrost_electricity_kwh"])
                            .pow(2)
                            .mean()
                        )
                    ),
                    "event_count": len(values),
                }
            )
    per_experiment = pd.DataFrame(rows)
    fixed = per_experiment.loc[per_experiment["strategy"].eq("fixed")].set_index(
        "experiment_id"
    )["mae"]
    overall = []
    fixed_macro = float(fixed.mean())
    for strategy, prediction in columns.items():
        selected = per_experiment.loc[per_experiment["strategy"].eq(strategy)]
        macro = float(selected["mae"].mean())
        improvement = 100.0 * (fixed_macro - macro) / fixed_macro
        improved_fraction = float(
            selected.set_index("experiment_id")["mae"].lt(fixed).mean()
        )
        overall.append(
            {
                "experiment_id": "__overall__",
                "strategy": strategy,
                "outcome": "defrost_electricity",
                "protocol": "leave_one_experiment_out",
                "mae": (
                    predictions[prediction] - predictions["defrost_electricity_kwh"]
                )
                .abs()
                .mean(),
                "event_count": len(predictions),
                "event_weighted_mae": (
                    predictions[prediction] - predictions["defrost_electricity_kwh"]
                )
                .abs()
                .mean(),
                "mse": (
                    predictions[prediction] - predictions["defrost_electricity_kwh"]
                )
                .pow(2)
                .mean(),
                "rmse": float(
                    np.sqrt(
                        (
                            predictions[prediction]
                            - predictions["defrost_electricity_kwh"]
                        )
                        .pow(2)
                        .mean()
                    )
                ),
                "macro_mae": macro,
                "improvement_vs_fixed_pct": improvement,
                "improved_experiment_fraction": improved_fraction,
                "recommend_for_primary_cost": bool(
                    strategy == "t3_rule" and improvement >= 10.0 and improved_fraction > 0.5
                ),
            }
        )
    metrics = [per_experiment, pd.DataFrame(overall)]
    heat_columns = {
        label.removeprefix("predicted_").removesuffix("_defrost_absorbed_heat"): label
        for label in predictions
        if label.startswith("predicted_")
        and label.endswith("_defrost_absorbed_heat")
    }
    if heat_columns:
        heat_rows = []
        for experiment, values in predictions.groupby("experiment_id", sort=True):
            for strategy, prediction in heat_columns.items():
                error = values[prediction] - values[
                    "defrost_absorbed_heat_kwh"
                ]
                heat_rows.append(
                    {
                        "experiment_id": experiment,
                        "strategy": strategy,
                        "outcome": "defrost_absorbed_heat",
                        "protocol": "leave_one_experiment_out",
                        "mae": error.abs().mean(),
                        "mse": error.pow(2).mean(),
                        "rmse": float(np.sqrt(error.pow(2).mean())),
                        "event_count": len(values),
                    }
                )
        heat_metrics = pd.DataFrame(heat_rows)
        for strategy, prediction in heat_columns.items():
            error = predictions[prediction] - predictions["defrost_absorbed_heat_kwh"]
            heat_metrics.loc[len(heat_metrics)] = {
                "experiment_id": "__overall__",
                "strategy": strategy,
                "outcome": "defrost_absorbed_heat",
                "protocol": "leave_one_experiment_out",
                "mae": error.abs().mean(),
                "mse": error.pow(2).mean(),
                "rmse": float(np.sqrt(error.pow(2).mean())),
                "event_count": len(predictions),
            }
        metrics.append(heat_metrics)
    return pd.concat(metrics, ignore_index=True)


def predict_candidate_tickets(
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    state_features: list[str],
) -> pd.DataFrame:
    """Apply an experiment-held-out state model at every candidate time."""
    predictions = []
    for experiment, test in candidates.groupby("experiment_id", sort=True):
        train = events.loc[~events["experiment_id"].eq(experiment)]
        if train.empty:
            train = events
        test = test.copy()
        for label, outcome in OUTCOMES.items():
            predicted = _ridge_predict(train, test, state_features, outcome)
            test[f"predicted_ticket_{label}"] = np.clip(
                predicted, train[outcome].min(), train[outcome].max()
            )
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True)


def conditional_optimal_points(
    curves: pd.DataFrame, *, threshold: float = 0.01
) -> pd.DataFrame:
    """Return conditional minima and near-optimal windows for every cycle."""
    rows = []
    for cycle, values in curves.groupby("cycle_name", sort=True):
        values = values.sort_values("candidate_time", kind="stable")
        minimum = values["renewal_cost_conditional"].min()
        optimum = values.loc[values["renewal_cost_conditional"].eq(minimum)].iloc[0]
        near = values.loc[
            values["renewal_cost_conditional"].le((1 + threshold) * minimum)
        ]
        position = values.index.get_loc(optimum.name)
        rows.append(
            {
                "cycle_name": cycle,
                "t_star_conditional": pd.Timestamp(optimum["candidate_time"]),
                "rho_min_conditional": minimum,
                "near_opt_start_conditional": pd.to_datetime(near["candidate_time"]).min(),
                "near_opt_end_conditional": pd.to_datetime(near["candidate_time"]).max(),
                "minimum_location_conditional": "left_boundary"
                if position == 0
                else "right_boundary"
                if position == len(values) - 1
                else "interior",
            }
        )
    return pd.DataFrame(rows)


def _load_raw(loader: DatasetLoader, cycle: str) -> pd.DataFrame:
    frame = loader.load_cycle_original(cycle, columns=RAW_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame.sort_values("timestamp", kind="stable").drop_duplicates("timestamp")


def build_defrost_power_rows(loader: DatasetLoader, events: pd.DataFrame) -> pd.DataFrame:
    """Return interpolated 1 s defrost power rows with the observed T3 stage split."""
    rows = []
    for event in events.itertuples(index=False):
        start = pd.Timestamp(event.catalog_defrost_start)
        end = pd.Timestamp(event.catalog_defrost_end)
        duration_actual_s = int((end - start).total_seconds())
        frame = loader.load_cycle_original(
            event.cycle_name,
            columns=["timestamp", "power_total", "coil_temperature"],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = (
            frame.loc[frame["timestamp"].ge(start) & frame["timestamp"].lt(end)]
            .sort_values("timestamp", kind="stable")
            .drop_duplicates("timestamp")
            .set_index("timestamp")
            .reindex(pd.date_range(start, periods=duration_actual_s, freq="s"))
        )
        frame[["power_total", "coil_temperature"]] = frame[
            ["power_total", "coil_temperature"]
        ].apply(pd.to_numeric, errors="coerce").interpolate(
            method="time", limit_area="inside"
        )
        if frame["coil_temperature"].isna().any():
            raise ValueError(f"incomplete defrost trace: {event.cycle_name}")
        reached = np.flatnonzero(frame["coil_temperature"].to_numpy() >= 20.0)
        if not len(reached):
            raise ValueError(f"T3 never reaches 20 C: {event.cycle_name}")
        cross20_s = int(reached[0])
        elapsed_s = np.arange(duration_actual_s)
        stage = np.where(elapsed_s < cross20_s, 1, 2)
        local_s = np.where(stage == 1, elapsed_s, elapsed_s - cross20_s)
        values = pd.DataFrame(
            {
                "cycle_name": event.cycle_name,
                "experiment_id": event.experiment_id,
                "elapsed_s": elapsed_s,
                "elapsed_min": elapsed_s / 60.0,
                "stage": stage,
                "u_min": local_s / 60.0,
                "power_kw": frame["power_total"].to_numpy(dtype=float),
                "cross20_s": cross20_s,
                "duration_rule_s": min(cross20_s + 40, 350),
                "duration_actual_s": duration_actual_s,
                "actual_energy_kwh": event.defrost_electricity_kwh,
            }
        ).dropna(subset=["power_kw"])
        values["stage_weight"] = 1.0 / values.groupby("stage")["stage"].transform(
            "size"
        )
        rows.append(values)
    return pd.concat(rows, ignore_index=True)


def _defrost_power_design(duration_min: float, u_min: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(u_min)),
            np.repeat(duration_min, len(u_min)),
            u_min,
            duration_min * u_min,
            u_min**2,
            duration_min * u_min**2,
        ]
    )


def _integrate_power_kwh(power_kw: np.ndarray) -> float:
    return float(np.asarray(power_kw).sum() / 3600.0)


def _fit_defrost_power(
    rows: pd.DataFrame, duration_column: str, time_column: str, weights: pd.Series
) -> tuple[np.ndarray, float]:
    duration = rows[duration_column].to_numpy(dtype=float) / 60.0
    u = rows[time_column].to_numpy(dtype=float)
    design = np.column_stack(
        [np.ones(len(rows)), duration, u, duration * u, u**2, duration * u**2]
    )
    root_weight = np.sqrt(weights.to_numpy(dtype=float))
    weighted_design = design * root_weight[:, None]
    weighted_power = rows["power_kw"].to_numpy(dtype=float) * root_weight
    if not np.isfinite(weighted_design).all() or not np.isfinite(weighted_power).all():
        raise ValueError("non-finite defrost power least-squares data")
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        weighted_design, weighted_power, rcond=None
    )
    if rank != design.shape[1]:
        raise ValueError(f"rank-deficient defrost power design: {rank}/6")
    if not np.isfinite(coefficients).all() or not np.isfinite(singular_values).all():
        raise ValueError("non-finite defrost power least-squares result")
    return coefficients, float(singular_values[0] / singular_values[-1])


def leave_one_experiment_out_defrost_power(
    rows: pd.DataFrame, duration_mode: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit power on other experiments and integrate held-out event predictions."""
    duration_column = f"duration_{duration_mode}_s"
    predictions = []
    coefficient_rows = []
    for experiment in sorted(rows["experiment_id"].unique()):
        train = rows.loc[~rows["experiment_id"].eq(experiment)].copy()
        test = rows.loc[rows["experiment_id"].eq(experiment)]
        event_weight = 1.0 / train.groupby("cycle_name")["cycle_name"].transform("size")
        one_stage, one_stage_condition = _fit_defrost_power(
            train, duration_column, "elapsed_min", event_weight
        )
        fitted_stages = {
            stage: _fit_defrost_power(
                train.loc[train["stage"].eq(stage)],
                duration_column,
                "u_min",
                train.loc[train["stage"].eq(stage), "stage_weight"],
            )
            for stage in (1, 2)
        }
        two_stage = {stage: fitted_stages[stage][0] for stage in (1, 2)}
        for model, stage, coefficients, condition in [
            ("markus_original", "all", MARKUS_POWER_COEFFICIENTS_KW, np.nan),
            ("one_stage_ls", "all", one_stage, one_stage_condition),
            ("two_stage_ls", "1", *fitted_stages[1]),
            ("two_stage_ls", "2", *fitted_stages[2]),
        ]:
            coefficient_rows.append(
                {
                    "held_out_experiment": experiment,
                    "duration_mode": duration_mode,
                    "model": model,
                    "stage": stage,
                    "design_condition_number": condition,
                    **{f"b{i}": value for i, value in enumerate(coefficients)},
                }
            )
        for cycle_name, event in test.groupby("cycle_name", sort=False):
            meta = event.iloc[0]
            duration_s = int(meta[duration_column])
            elapsed_s = np.arange(duration_s)
            duration_min = duration_s / 60.0
            markus = (
                _defrost_power_design(duration_min, elapsed_s / 60.0)
                @ MARKUS_POWER_COEFFICIENTS_KW
            )
            fitted_one = _defrost_power_design(duration_min, elapsed_s / 60.0) @ one_stage
            stage = np.where(elapsed_s < meta["cross20_s"], 1, 2)
            local_s = np.where(stage == 1, elapsed_s, elapsed_s - meta["cross20_s"])
            fitted_two = np.empty(duration_s)
            for phase in (1, 2):
                selected = stage == phase
                fitted_two[selected] = (
                    _defrost_power_design(duration_min, local_s[selected] / 60.0)
                    @ two_stage[phase]
                )
            for model, predicted_power in (
                ("markus_original", markus),
                ("one_stage_ls", fitted_one),
                ("two_stage_ls", fitted_two),
            ):
                predictions.append(
                    {
                        "cycle_name": cycle_name,
                        "experiment_id": experiment,
                        "duration_mode": duration_mode,
                        "model": model,
                        "cross20_s": int(meta["cross20_s"]),
                        "prediction_duration_s": duration_s,
                        "actual_energy_kwh": float(meta["actual_energy_kwh"]),
                        "predicted_energy_kwh": _integrate_power_kwh(predicted_power),
                        "negative_power_count": int((predicted_power < 0).sum()),
                        "predicted_second_count": duration_s,
                        "min_predicted_power_kw": float(predicted_power.min()),
                        "max_predicted_power_kw": float(predicted_power.max()),
                    }
                )
    return pd.DataFrame(predictions), pd.DataFrame(coefficient_rows)


def defrost_power_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize event energy error and per-second prediction diagnostics."""
    rows = []
    for (duration_mode, model), values in predictions.groupby(
        ["duration_mode", "model"], sort=True
    ):
        error = values["predicted_energy_kwh"] - values["actual_energy_kwh"]
        experiment_mse = error.pow(2).groupby(values["experiment_id"]).mean()
        rows.append(
            {
                "duration_mode": duration_mode,
                "model": model,
                "event_count": len(values),
                "experiment_count": values["experiment_id"].nunique(),
                "mse_kwh2": error.pow(2).mean(),
                "rmse_kwh": float(np.sqrt(error.pow(2).mean())),
                "mae_kwh": error.abs().mean(),
                "bias_kwh": error.mean(),
                "macro_experiment_mse_kwh2": experiment_mse.mean(),
                "negative_power_fraction": values["negative_power_count"].sum()
                / values["predicted_second_count"].sum(),
                "min_predicted_power_kw": values["min_predicted_power_kw"].min(),
                "max_predicted_power_kw": values["max_predicted_power_kw"].max(),
            }
        )
    return pd.DataFrame(rows)


def write_defrost_power_evidence(dataset: Path, output: Path) -> pd.DataFrame:
    """Write the held-out one- and two-stage power-model evidence."""
    loader = DatasetLoader(dataset)
    events = pd.read_csv(output / "ticket_event_features_and_predictions.csv")
    power_rows = build_defrost_power_rows(loader, events)
    results = [
        leave_one_experiment_out_defrost_power(power_rows, mode)
        for mode in ("rule", "actual")
    ]
    predictions = pd.concat([result[0] for result in results], ignore_index=True)
    coefficients = pd.concat([result[1] for result in results], ignore_index=True)
    metrics = defrost_power_metrics(predictions)
    predictions.to_csv(output / "defrost_power_predictions.csv", index=False)
    coefficients.to_csv(output / "defrost_power_coefficients.csv", index=False)
    metrics.to_csv(output / "defrost_power_metrics.csv", index=False)
    return metrics


def _fit_duration_t3_energy(
    duration_s: pd.Series, t3_pre60: pd.Series, energy_kwh: pd.Series
) -> tuple[np.ndarray, float]:
    duration_hours = duration_s.to_numpy(dtype=float) / 3600.0
    t3 = t3_pre60.to_numpy(dtype=float)
    design = np.column_stack([duration_hours, duration_hours * t3])
    target = energy_kwh.to_numpy(dtype=float)
    if not np.isfinite(design).all() or not np.isfinite(target).all():
        raise ValueError("non-finite duration-T3 energy least-squares data")
    beta, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    if rank != 2:
        raise ValueError(f"rank-deficient duration-T3 energy design: {rank}/2")
    if not np.isfinite(beta).all() or not np.isfinite(singular_values).all():
        raise ValueError("non-finite duration-T3 energy least-squares result")
    return beta, float(singular_values[0] / singular_values[-1])


def _fit_additive_duration_t3_energy(
    duration_s: pd.Series, t3_pre60: pd.Series, energy_kwh: pd.Series
) -> tuple[np.ndarray, float]:
    design = np.column_stack(
        [
            np.ones(len(duration_s)),
            duration_s.to_numpy(dtype=float) / 60.0,
            t3_pre60.to_numpy(dtype=float),
        ]
    )
    target = energy_kwh.to_numpy(dtype=float)
    if not np.isfinite(design).all() or not np.isfinite(target).all():
        raise ValueError("non-finite additive duration-T3 least-squares data")
    beta, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    if rank != 3:
        raise ValueError(f"rank-deficient additive duration-T3 design: {rank}/3")
    if not np.isfinite(beta).all() or not np.isfinite(singular_values).all():
        raise ValueError("non-finite additive duration-T3 least-squares result")
    return beta, float(singular_values[0] / singular_values[-1])


def leave_one_experiment_out_duration_t3_energy(
    events: pd.DataFrame, duration_rows: pd.DataFrame, duration_mode: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare held-out energy models using pre-defrost T3 and known duration."""
    duration_column = f"duration_{duration_mode}_s"
    data = events[
        [
            "cycle_name",
            "experiment_id",
            "coil_temperature",
            "defrost_electricity_kwh",
            "predicted_t3_rule_defrost_electricity",
        ]
    ].merge(
        duration_rows[["cycle_name", duration_column]].drop_duplicates("cycle_name"),
        on="cycle_name",
        validate="one_to_one",
    )
    predictions = []
    coefficient_rows = []
    for experiment in sorted(data["experiment_id"].unique()):
        train = data.loc[~data["experiment_id"].eq(experiment)].copy()
        test = data.loc[data["experiment_id"].eq(experiment)].copy()
        duration_hours = test[duration_column].to_numpy(dtype=float) / 3600.0
        train_duration_hours = train[duration_column].to_numpy(dtype=float) / 3600.0
        mean_energy = float(train["defrost_electricity_kwh"].mean())
        mean_power = float(
            train["defrost_electricity_kwh"].sum() / train_duration_hours.sum()
        )
        beta, condition = _fit_duration_t3_energy(
            train[duration_column],
            train["coil_temperature"],
            train["defrost_electricity_kwh"],
        )
        additive_beta, additive_condition = _fit_additive_duration_t3_energy(
            train[duration_column],
            train["coil_temperature"],
            train["defrost_electricity_kwh"],
        )
        t3 = test["coil_temperature"].to_numpy(dtype=float)
        model_predictions = {
            "additive_sensitivity": (
                additive_beta[0]
                + additive_beta[1] * test[duration_column].to_numpy(dtype=float) / 60.0
                + additive_beta[2] * t3
            ),
            "fixed_mean_energy": np.repeat(mean_energy, len(test)),
            "duration_only": mean_power * duration_hours,
            "old_t3_duration": test[
                "predicted_t3_rule_defrost_electricity"
            ].to_numpy(dtype=float),
            "duration_t3_physical": duration_hours * (beta[0] + beta[1] * t3),
        }
        coefficient_rows.append(
            {
                "held_out_experiment": experiment,
                "duration_mode": duration_mode,
                "model": "duration_t3_physical",
                "beta0_kw": beta[0],
                "beta1_kw_per_c": beta[1],
                "design_condition_number": condition,
                "training_mean_power_kw": mean_power,
            }
        )
        coefficient_rows.append(
            {
                "held_out_experiment": experiment,
                "duration_mode": duration_mode,
                "model": "additive_sensitivity",
                "additive_b0_kwh": additive_beta[0],
                "additive_b1_kwh_per_min": additive_beta[1],
                "additive_b2_kwh_per_c": additive_beta[2],
                "design_condition_number": additive_condition,
                "training_mean_power_kw": mean_power,
            }
        )
        for row_index, (_, event) in enumerate(test.iterrows()):
            for model, values in model_predictions.items():
                predicted_energy = float(values[row_index])
                predictions.append(
                    {
                        "cycle_name": event["cycle_name"],
                        "experiment_id": experiment,
                        "duration_mode": duration_mode,
                        "model": model,
                        "t3_pre60_c": event["coil_temperature"],
                        "known_duration_s": event[duration_column],
                        "actual_energy_kwh": event["defrost_electricity_kwh"],
                        "predicted_energy_kwh": predicted_energy,
                        "predicted_mean_power_kw": predicted_energy
                        / duration_hours[row_index],
                    }
                )
    return pd.DataFrame(predictions), pd.DataFrame(coefficient_rows)


def duration_t3_energy_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (duration_mode, model), values in predictions.groupby(
        ["duration_mode", "model"], sort=True
    ):
        error = values["predicted_energy_kwh"] - values["actual_energy_kwh"]
        experiment_mse = error.pow(2).groupby(values["experiment_id"]).mean()
        rows.append(
            {
                "duration_mode": duration_mode,
                "model": model,
                "event_count": len(values),
                "experiment_count": values["experiment_id"].nunique(),
                "mse_kwh2": error.pow(2).mean(),
                "rmse_kwh": float(np.sqrt(error.pow(2).mean())),
                "mae_kwh": error.abs().mean(),
                "bias_kwh": error.mean(),
                "macro_experiment_mse_kwh2": experiment_mse.mean(),
                "negative_energy_fraction": values["predicted_energy_kwh"].lt(0).mean(),
                "negative_mean_power_fraction": values["predicted_mean_power_kw"]
                .lt(0)
                .mean(),
                "min_predicted_energy_kwh": values["predicted_energy_kwh"].min(),
                "max_predicted_energy_kwh": values["predicted_energy_kwh"].max(),
                "min_predicted_mean_power_kw": values[
                    "predicted_mean_power_kw"
                ].min(),
                "max_predicted_mean_power_kw": values[
                    "predicted_mean_power_kw"
                ].max(),
            }
        )
    metrics = pd.DataFrame(rows)
    for duration_mode in metrics["duration_mode"].unique():
        selected = predictions.loc[predictions["duration_mode"].eq(duration_mode)]
        baseline = metrics.loc[
            metrics["duration_mode"].eq(duration_mode)
            & metrics["model"].eq("duration_only"),
            "mse_kwh2",
        ].item()
        wide = selected.pivot(
            index=["experiment_id", "cycle_name"],
            columns="model",
            values=["predicted_energy_kwh", "actual_energy_kwh"],
        )
        actual = wide["actual_energy_kwh"].iloc[:, 0]
        duration_mse = (
            wide["predicted_energy_kwh"]["duration_only"] - actual
        ).pow(2).groupby(level="experiment_id").mean()
        for model in (
            "old_t3_duration",
            "duration_t3_physical",
            "additive_sensitivity",
        ):
            model_row = metrics["duration_mode"].eq(duration_mode) & metrics[
                "model"
            ].eq(model)
            metrics.loc[model_row, "improvement_vs_duration_only_pct"] = 100.0 * (
                baseline - metrics.loc[model_row, "mse_kwh2"]
            ) / baseline
            model_mse = (
                wide["predicted_energy_kwh"][model] - actual
            ).pow(2).groupby(level="experiment_id").mean()
            metrics.loc[model_row, "improved_experiment_count"] = int(
                model_mse.lt(duration_mse).sum()
            )
    return metrics


def write_duration_t3_energy_evidence(dataset: Path, output: Path) -> pd.DataFrame:
    loader = DatasetLoader(dataset)
    events = pd.read_csv(output / "ticket_event_features_and_predictions.csv")
    duration_rows = build_defrost_power_rows(loader, events)
    results = [
        leave_one_experiment_out_duration_t3_energy(events, duration_rows, mode)
        for mode in ("rule", "actual")
    ]
    predictions = pd.concat([result[0] for result in results], ignore_index=True)
    coefficients = pd.concat([result[1] for result in results], ignore_index=True)
    metrics = duration_t3_energy_metrics(predictions)
    predictions.to_csv(output / "duration_t3_energy_predictions.csv", index=False)
    coefficients.to_csv(output / "duration_t3_energy_coefficients.csv", index=False)
    metrics.to_csv(output / "duration_t3_energy_metrics.csv", index=False)
    return metrics


def _fit_predefrost_energy_features(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    train_columns = [
        np.ones(len(train)),
        train["defrost_duration_minutes"].to_numpy(dtype=float),
        train["coil_temperature"].to_numpy(dtype=float),
    ]
    test_columns = [
        np.ones(len(test)),
        test["defrost_duration_minutes"].to_numpy(dtype=float),
        test["coil_temperature"].to_numpy(dtype=float),
    ]
    used = {}
    for feature in features:
        median = float(pd.to_numeric(train[feature], errors="coerce").median())
        train_values = pd.to_numeric(train[feature], errors="coerce").fillna(median)
        test_values = pd.to_numeric(test[feature], errors="coerce").fillna(median)
        scale = float(train_values.std(ddof=0))
        if not np.isfinite(median) or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"unusable predefrost feature: {feature}")
        train_columns.append((train_values.to_numpy(dtype=float) - median) / scale)
        test_columns.append((test_values.to_numpy(dtype=float) - median) / scale)
        used[feature] = test_values.to_numpy(dtype=float)
    design = np.column_stack(train_columns)
    test_design = np.column_stack(test_columns)
    target = train["defrost_electricity_kwh"].to_numpy(dtype=float)
    if not all(
        np.isfinite(values).all() for values in (design, test_design, target)
    ):
        raise ValueError("non-finite predefrost feature model data")
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, target, rcond=None
    )
    if rank != design.shape[1]:
        raise ValueError("rank-deficient predefrost feature model")
    if not np.isfinite(coefficients).all() or not np.isfinite(singular_values).all():
        raise ValueError("non-finite predefrost feature model result")
    return (
        test_design @ coefficients,
        float(singular_values[0] / singular_values[-1]),
        used,
    )


def _loeo_feature_mse(events: pd.DataFrame, feature: str) -> float:
    errors = []
    for experiment in sorted(events["experiment_id"].unique()):
        train = events.loc[~events["experiment_id"].eq(experiment)]
        test = events.loc[events["experiment_id"].eq(experiment)]
        predicted, _, _ = _fit_predefrost_energy_features(train, test, [feature])
        errors.extend(predicted - test["defrost_electricity_kwh"].to_numpy(dtype=float))
    return float(np.mean(np.square(errors)))


def _held_out_feature_residual(
    train: pd.DataFrame, test: pd.DataFrame, feature: str
) -> np.ndarray:
    median = float(pd.to_numeric(train[feature], errors="coerce").median())
    train_feature = pd.to_numeric(train[feature], errors="coerce").fillna(median)
    test_feature = pd.to_numeric(test[feature], errors="coerce").fillna(median)
    train_design = np.column_stack(
        [
            np.ones(len(train)),
            train["defrost_duration_minutes"],
            train["coil_temperature"],
        ]
    )
    test_design = np.column_stack(
        [
            np.ones(len(test)),
            test["defrost_duration_minutes"],
            test["coil_temperature"],
        ]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(
        train_design, train_feature.to_numpy(dtype=float), rcond=None
    )
    if rank != 3 or not np.isfinite(coefficients).all():
        raise ValueError(f"rank-deficient feature residual model: {feature}")
    return test_feature.to_numpy(dtype=float) - test_design @ coefficients


def select_fixed_literature_features(summary: pd.DataFrame) -> list[str]:
    """Apply the fixed screen and collapse deterministic/physical duplicates."""
    baseline_macro = float(
        summary.loc[
            summary["feature"].eq("__baseline__"),
            "macro_experiment_mse_kwh2",
        ].iloc[0]
    )
    candidates = summary.loc[
        summary["feature"].isin(LITERATURE_SENSOR_FEATURES)
    ].copy()
    candidates["macro_improvement_pct"] = 100.0 * (
        baseline_macro - candidates["macro_experiment_mse_kwh2"]
    ) / baseline_macro
    eligible = candidates.loc[
        candidates["improvement_vs_baseline_pct"].ge(
            FIXED_SCREEN_MIN_EVENT_IMPROVEMENT_PCT
        )
        & candidates["macro_improvement_pct"].ge(
            FIXED_SCREEN_MIN_MACRO_IMPROVEMENT_PCT
        )
        & candidates["improved_experiment_count"].ge(
            FIXED_SCREEN_MIN_IMPROVED_EXPERIMENTS
        )
    ]
    selected = []
    for family in (
        ["evaporating_pressure"],
        ["cop"],
        ["power_total", "compressor_power", "compressor_frequency"],
    ):
        available = eligible.loc[eligible["feature"].isin(family)]
        if len(available):
            selected.append(str(available.sort_values("mse_kwh2").iloc[0]["feature"]))
    return selected


def evaluate_predefrost_sensor_increment(  # noqa: C901
    events: pd.DataFrame,
    candidates: list[str],
    *,
    include_nested: bool = True,
    fixed_combinations: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate pre-defrost sensors beyond actual duration and pre-defrost T3."""
    prediction_rows = []
    failed: set[str] = set()
    for experiment in sorted(events["experiment_id"].unique()):
        train = events.loc[~events["experiment_id"].eq(experiment)]
        test = events.loc[events["experiment_id"].eq(experiment)]
        baseline, condition, _ = _fit_predefrost_energy_features(train, test, [])
        for row_index, (_, event) in enumerate(test.iterrows()):
            prediction_rows.append(
                {
                    "cycle_name": event["cycle_name"],
                    "experiment_id": experiment,
                    "feature": "__baseline__",
                    "selected_features": "",
                    "feature_value_used": np.nan,
                    "feature_residual_oof": np.nan,
                    "actual_energy_kwh": event["defrost_electricity_kwh"],
                    "predicted_energy_kwh": baseline[row_index],
                    "design_condition_number": condition,
                }
            )
        for feature in candidates:
            try:
                predicted, condition, used = _fit_predefrost_energy_features(
                    train, test, [feature]
                )
                feature_residual = _held_out_feature_residual(train, test, feature)
            except ValueError:
                failed.add(feature)
                continue
            for row_index, (_, event) in enumerate(test.iterrows()):
                prediction_rows.append(
                    {
                        "cycle_name": event["cycle_name"],
                        "experiment_id": experiment,
                        "feature": feature,
                        "selected_features": feature,
                        "feature_value_used": used[feature][row_index],
                        "feature_residual_oof": feature_residual[row_index],
                        "actual_energy_kwh": event["defrost_electricity_kwh"],
                        "predicted_energy_kwh": predicted[row_index],
                        "design_condition_number": condition,
                    }
                )
        for label, features in (fixed_combinations or {}).items():
            try:
                predicted, condition, _ = _fit_predefrost_energy_features(
                    train, test, features
                )
            except ValueError:
                failed.add(label)
                continue
            for row_index, (_, event) in enumerate(test.iterrows()):
                prediction_rows.append(
                    {
                        "cycle_name": event["cycle_name"],
                        "experiment_id": experiment,
                        "feature": label,
                        "selected_features": ";".join(features),
                        "feature_value_used": np.nan,
                        "feature_residual_oof": np.nan,
                        "actual_energy_kwh": event["defrost_electricity_kwh"],
                        "predicted_energy_kwh": predicted[row_index],
                        "design_condition_number": condition,
                    }
                )
    predictions = pd.DataFrame(prediction_rows)
    for feature in failed:
        predictions = predictions.loc[~predictions["feature"].eq(feature)]

    if include_nested:
        nested_rows = []
        usable = [
            feature
            for feature in candidates
            if feature not in failed
            and not feature.startswith(("q_heating_kw", "cop"))
        ]
        for experiment in sorted(events["experiment_id"].unique()):
            train = events.loc[~events["experiment_id"].eq(experiment)]
            test = events.loc[events["experiment_id"].eq(experiment)]
            ranked = []
            for feature in usable:
                try:
                    ranked.append((_loeo_feature_mse(train, feature), feature))
                except ValueError:
                    continue
            selected = [feature for _, feature in sorted(ranked)[:2]]
            if not selected:
                continue
            predicted, condition, _ = _fit_predefrost_energy_features(
                train, test, selected
            )
            label = ";".join(selected)
            for row_index, (_, event) in enumerate(test.iterrows()):
                nested_rows.append(
                    {
                        "cycle_name": event["cycle_name"],
                        "experiment_id": experiment,
                        "feature": "__nested_top2__",
                        "selected_features": label,
                        "feature_value_used": np.nan,
                        "feature_residual_oof": np.nan,
                        "actual_energy_kwh": event["defrost_electricity_kwh"],
                        "predicted_energy_kwh": predicted[row_index],
                        "design_condition_number": condition,
                    }
                )
        predictions = pd.concat(
            [predictions, pd.DataFrame(nested_rows)], ignore_index=True
        )

    baseline = predictions.loc[predictions["feature"].eq("__baseline__")].set_index(
        "cycle_name"
    )
    baseline_error = (
        baseline["predicted_energy_kwh"] - baseline["actual_energy_kwh"]
    )
    baseline_experiment_mse = baseline_error.pow(2).groupby(
        baseline["experiment_id"]
    ).mean()
    correlation = events.set_index("cycle_name")
    numeric_candidates = [feature for feature in candidates if feature not in failed]
    feature_correlations = correlation[numeric_candidates].corr().abs()
    np.fill_diagonal(feature_correlations.values, np.nan)
    summary_rows = []
    for feature, values in predictions.groupby("feature", sort=True):
        error = values["predicted_energy_kwh"] - values["actual_energy_kwh"]
        experiment_mse = error.pow(2).groupby(values["experiment_id"]).mean()
        raw_feature = correlation.get(feature, None)
        if raw_feature is None:
            valid_n = len(events)
            missing_fraction = 0.0
            pearson = spearman = residual_pearson = residual_spearman = np.nan
            correlated_feature = ""
            max_feature_correlation = np.nan
        else:
            valid = raw_feature.notna() & correlation["defrost_electricity_kwh"].notna()
            valid_n = int(valid.sum())
            missing_fraction = float(1 - valid_n / len(events))
            pearson = raw_feature.loc[valid].corr(
                correlation.loc[valid, "defrost_electricity_kwh"], method="pearson"
            )
            spearman = raw_feature.loc[valid].corr(
                correlation.loc[valid, "defrost_electricity_kwh"], method="spearman"
            )
            feature_residual = values.set_index("cycle_name")["feature_residual_oof"]
            energy_residual = -baseline_error.loc[feature_residual.index]
            residual_pearson = feature_residual.corr(
                energy_residual, method="pearson"
            )
            residual_spearman = feature_residual.corr(
                energy_residual, method="spearman"
            )
            correlations = feature_correlations[feature].dropna()
            correlated_feature = str(correlations.idxmax()) if len(correlations) else ""
            max_feature_correlation = (
                float(correlations.max()) if len(correlations) else np.nan
            )
        baseline_mse = float(baseline_error.pow(2).mean())
        summary_rows.append(
            {
                "feature": feature,
                "status": "ok",
                "valid_n": valid_n,
                "missing_fraction": missing_fraction,
                "pearson_energy": pearson,
                "spearman_energy": spearman,
                "oof_residual_pearson": residual_pearson,
                "oof_residual_spearman": residual_spearman,
                "mse_kwh2": error.pow(2).mean(),
                "rmse_kwh": float(np.sqrt(error.pow(2).mean())),
                "mae_kwh": error.abs().mean(),
                "bias_kwh": error.mean(),
                "macro_experiment_mse_kwh2": experiment_mse.mean(),
                "improvement_vs_baseline_pct": 100
                * (baseline_mse - error.pow(2).mean())
                / baseline_mse,
                "improved_experiment_count": int(
                    experiment_mse.lt(baseline_experiment_mse).sum()
                ),
                "negative_prediction_fraction": values[
                    "predicted_energy_kwh"
                ].lt(0).mean(),
                "min_predicted_energy_kwh": values["predicted_energy_kwh"].min(),
                "max_predicted_energy_kwh": values["predicted_energy_kwh"].max(),
                "max_condition_number": values["design_condition_number"].max(),
                "most_correlated_feature": correlated_feature,
                "max_abs_feature_correlation": max_feature_correlation,
                "derived_heating_feature": feature.startswith(
                    ("q_heating_kw", "cop", "evaporator_capacity")
                ),
            }
        )
    for feature in sorted(failed):
        summary_rows.append(
            {
                "feature": feature,
                "status": "unusable_in_at_least_one_fold",
                "valid_n": int(events[feature].notna().sum()),
                "missing_fraction": float(events[feature].isna().mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["status", "mse_kwh2", "feature"], kind="stable", na_position="last"
    )
    return summary.reset_index(drop=True), predictions.reset_index(drop=True)


def write_predefrost_sensor_increment_evidence(output: Path) -> pd.DataFrame:
    events = pd.read_csv(output / "ticket_event_features_and_predictions.csv")
    summary, predictions = evaluate_predefrost_sensor_increment(
        events, PREDEFROST_SENSOR_FEATURES
    )
    summary.to_csv(output / "predefrost_sensor_increment_summary.csv", index=False)
    predictions.to_csv(
        output / "predefrost_sensor_increment_predictions.csv", index=False
    )
    return summary


def build_preparation_inclusive_events(  # noqa: C901
    loader: DatasetLoader, tickets: pd.DataFrame, catalog: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild pre-preparation features and preparation-inclusive energy."""
    boundaries = tickets[["cycle_name", "experiment_id"]].merge(
        catalog[
            [
                "cycle_name",
                "defrost_preparation_start",
                "defrost_start",
                "defrost_end",
                "baseline_start",
                "baseline_end",
            ]
        ],
        on="cycle_name",
        how="left",
        validate="one_to_one",
    )
    event_rows = []
    audit_rows = []
    raw_columns = [
        "timestamp",
        "power_total",
        "coil_temperature",
        "water_flow",
        "water_in_temperature",
        "water_out_temperature",
        "compressor_frequency",
        "compressor_power",
        "evaporating_temperature",
        "evaporating_pressure",
        "fan_current",
        "ambient_temperature",
        "environment_relative_humidity",
    ]
    for event in boundaries.itertuples(index=False):
        preparation_start = pd.to_datetime(
            event.defrost_preparation_start, errors="coerce"
        )
        defrost_start = pd.to_datetime(event.defrost_start, errors="coerce")
        end = pd.to_datetime(event.defrost_end, errors="coerce")
        baseline_start = pd.to_datetime(event.baseline_start, errors="coerce")
        baseline_end = pd.to_datetime(event.baseline_end, errors="coerce")
        reason = ""
        if pd.isna(preparation_start) or pd.isna(defrost_start) or pd.isna(end):
            reason = "missing_preparation_or_defrost_boundary"
        elif not preparation_start < defrost_start < end:
            reason = "invalid_preparation_or_defrost_boundary_order"
        if reason:
            audit_rows.append(
                {"cycle_name": event.cycle_name, "status": "excluded", "reason": reason}
            )
            continue
        frame = loader.load_cycle_original(event.cycle_name, columns=raw_columns)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = (
            frame.sort_values("timestamp", kind="stable")
            .drop_duplicates("timestamp")
            .set_index("timestamp")
        )
        duration_s = int((end - preparation_start).total_seconds())
        preparation_duration_s = int(
            (defrost_start - preparation_start).total_seconds()
        )
        target = frame.reindex(
            pd.date_range(preparation_start, periods=duration_s, freq="s")
        )[["power_total"]].apply(pd.to_numeric, errors="coerce").interpolate(
            method="time", limit_area="inside"
        )
        window = frame.reindex(
            pd.date_range(
                preparation_start - pd.Timedelta(seconds=60), periods=60, freq="s"
            )
        )[
            [
                "power_total",
                "coil_temperature",
                "water_flow",
                "water_in_temperature",
                "water_out_temperature",
                "compressor_frequency",
                "compressor_power",
                "evaporating_temperature",
                "evaporating_pressure",
                "fan_current",
                "ambient_temperature",
                "environment_relative_humidity",
            ]
        ].apply(pd.to_numeric, errors="coerce").interpolate(
            method="time", limit_area="inside"
        )
        if pd.notna(baseline_start) and pd.notna(baseline_end):
            clean = frame.reindex(
                pd.date_range(baseline_start, baseline_end, inclusive="left", freq="s")
            )[window.columns].apply(pd.to_numeric, errors="coerce").interpolate(
                method="time", limit_area="inside"
            )
        else:
            clean = pd.DataFrame(columns=window.columns, dtype=float)
        if target["power_total"].isna().any():
            reason = "incomplete_preparation_inclusive_power"
        elif window["coil_temperature"].isna().any():
            reason = "incomplete_prepreparation_t3_window"
        if reason:
            audit_rows.append(
                {"cycle_name": event.cycle_name, "status": "excluded", "reason": reason}
            )
            continue

        def slope(values: pd.Series) -> float:
            valid = values.notna()
            return (
                float(np.polyfit(np.arange(60)[valid] / 60.0, values[valid], 1)[0])
                if valid.sum() >= 2
                else np.nan
            )

        def add_thermal_features(values: pd.DataFrame) -> pd.DataFrame:
            values = values.copy()
            values["q_heating_kw"] = water_side_heating_kw(values)
            values["evaporator_capacity_kw"] = (
                values["q_heating_kw"] - values["compressor_power"]
            )
            values["cop"] = values["q_heating_kw"].div(
                values["power_total"].where(values["power_total"].gt(0))
            )
            return values

        window = add_thermal_features(window)
        clean = add_thermal_features(clean)

        def median(name: str, values: pd.DataFrame = window) -> float:
            observed = values[name].dropna()
            return float(observed.median()) if len(observed) else np.nan

        def ratio_clean(
            name: str,
            current: pd.DataFrame = window,
            reference_values: pd.DataFrame = clean,
        ) -> float:
            reference = median(name, reference_values)
            return (
                median(name, current) / reference
                if np.isfinite(reference) and reference
                else np.nan
            )

        power = target["power_total"].to_numpy(dtype=float)
        event_rows.append(
            {
                "cycle_name": event.cycle_name,
                "experiment_id": event.experiment_id,
                "defrost_preparation_start": preparation_start,
                "defrost_start": defrost_start,
                "defrost_end": end,
                "inclusive_duration_minutes": duration_s / 60.0,
                "preparation_duration_s": preparation_duration_s,
                "inclusive_energy_kwh": power.sum() / 3600.0,
                "preparation_energy_kwh": power[:preparation_duration_s].sum()
                / 3600.0,
                "t3_prepreparation_c": window["coil_temperature"].median(),
                "coil_temperature_slope_per_min": slope(
                    window["coil_temperature"]
                ),
                "coil_temperature_iqr": window["coil_temperature"].quantile(0.75)
                - window["coil_temperature"].quantile(0.25),
                "evaporator_capacity_ratio_clean": ratio_clean(
                    "evaporator_capacity_kw"
                ),
                "q_heating_kw": median("q_heating_kw"),
                "q_heating_ratio_clean": ratio_clean("q_heating_kw"),
                "q_heating_kw_slope_per_min": slope(window["q_heating_kw"]),
                "evaporating_temperature": median("evaporating_temperature"),
                "evaporating_pressure": median("evaporating_pressure"),
                "water_flow_slope_per_min": slope(window["water_flow"]),
                "cop": median("cop"),
                "cop_ratio_clean": ratio_clean("cop"),
                "cop_slope_per_min": slope(window["cop"]),
                "fan_current": median("fan_current"),
                "fan_current_iqr": window["fan_current"].quantile(0.75)
                - window["fan_current"].quantile(0.25),
                "fan_current_slope_per_min": slope(window["fan_current"]),
                "compressor_frequency": median("compressor_frequency"),
                "compressor_frequency_slope_per_min": slope(
                    window["compressor_frequency"]
                ),
                "compressor_frequency_iqr": window[
                    "compressor_frequency"
                ].quantile(0.75)
                - window["compressor_frequency"].quantile(0.25),
                "evaporating_temperature_slope_per_min": slope(
                    window["evaporating_temperature"]
                ),
                "compressor_power": median("compressor_power"),
                "compressor_power_slope_per_min": slope(
                    window["compressor_power"]
                ),
                "power_total": median("power_total"),
                "power_total_slope_per_min": slope(window["power_total"]),
                "ambient_temperature": median("ambient_temperature"),
                "environment_relative_humidity": median(
                    "environment_relative_humidity"
                ),
            }
        )
        audit_rows.append(
            {"cycle_name": event.cycle_name, "status": "included", "reason": ""}
        )
    return pd.DataFrame(event_rows), pd.DataFrame(audit_rows)


def build_preparation_network_cohort(
    loader: DatasetLoader, catalog: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the catalog-valid complete-case cohort for the fixed network."""
    boundaries = catalog[["defrost_preparation_start", "defrost_end"]].apply(
        pd.to_datetime, errors="coerce", format="mixed"
    )
    candidates = catalog.loc[
        catalog["status"].eq("valid") & boundaries.notna().all(axis=1),
        ["cycle_name", "experiment_id"],
    ]
    events, audit = build_preparation_inclusive_events(loader, candidates, catalog)
    missing_clean = events["evaporator_capacity_ratio_clean"].isna()
    other_features = [
        feature
        for feature in PREPARATION_NETWORK_FEATURES
        if feature != "evaporator_capacity_ratio_clean"
    ]
    missing_input = events[other_features].isna().any(axis=1)
    for mask, reason in (
        (missing_input, "incomplete_network_input_window"),
        (missing_clean, "incomplete_network_clean_baseline"),
    ):
        selected = audit["cycle_name"].isin(events.loc[mask, "cycle_name"])
        audit.loc[selected, ["status", "reason"]] = [
            "excluded",
            reason,
        ]
    return events.loc[~missing_input & ~missing_clean].reset_index(drop=True), audit


def evaluate_preparation_network(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare fixed complete-case models by leaving each experiment out."""
    required = [*PREPARATION_NETWORK_FEATURES, "inclusive_energy_kwh"]
    numeric = events[required].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("preparation network requires complete-case inputs")
    rows = []
    pe_index = PREPARATION_NETWORK_FEATURES.index("evaporating_pressure")
    for experiment in sorted(events["experiment_id"].unique()):
        train = events.loc[~events["experiment_id"].eq(experiment)]
        test = events.loc[events["experiment_id"].eq(experiment)]
        scaler = StandardScaler().fit(train[PREPARATION_NETWORK_FEATURES])
        train_x = scaler.transform(train[PREPARATION_NETWORK_FEATURES])
        test_x = scaler.transform(test[PREPARATION_NETWORK_FEATURES])
        train_y = train["inclusive_energy_kwh"].to_numpy(dtype=float)
        target_mean = float(train_y.mean())
        target_scale = float(train_y.std(ddof=0))
        if not np.isfinite(target_scale) or target_scale <= 0:
            raise ValueError("preparation network training target has zero variance")
        scaled_y = (train_y - target_mean) / target_scale

        pe_design = np.column_stack([np.ones(len(train)), train_x[:, pe_index]])
        pe_test_design = np.column_stack([np.ones(len(test)), test_x[:, pe_index]])
        pe_coefficients = np.linalg.lstsq(pe_design, scaled_y, rcond=None)[0]
        pe_quadratic_raw = np.column_stack(
            [
                train["evaporating_pressure"],
                train["evaporating_pressure"].pow(2),
            ]
        )
        pe_quadratic_test_raw = np.column_stack(
            [
                test["evaporating_pressure"],
                test["evaporating_pressure"].pow(2),
            ]
        )
        pe_quadratic_scaler = StandardScaler().fit(pe_quadratic_raw)
        pe_quadratic_x = pe_quadratic_scaler.transform(pe_quadratic_raw)
        pe_quadratic_test_x = pe_quadratic_scaler.transform(pe_quadratic_test_raw)
        pe_quadratic_ridge = Ridge(alpha=PREPARATION_NETWORK_RIDGE_ALPHA).fit(
            pe_quadratic_x, scaled_y
        )
        ridge = Ridge(alpha=PREPARATION_NETWORK_RIDGE_ALPHA).fit(train_x, scaled_y)
        squared_scaler = StandardScaler().fit(
            np.column_stack(
                [
                    train[PREPARATION_NETWORK_FEATURES],
                    train[PREPARATION_NETWORK_FEATURES].pow(2),
                ]
            )
        )
        train_squared_x = squared_scaler.transform(
            np.column_stack(
                [
                    train[PREPARATION_NETWORK_FEATURES],
                    train[PREPARATION_NETWORK_FEATURES].pow(2),
                ]
            )
        )
        test_squared_x = squared_scaler.transform(
            np.column_stack(
                [
                    test[PREPARATION_NETWORK_FEATURES],
                    test[PREPARATION_NETWORK_FEATURES].pow(2),
                ]
            )
        )
        squared_ridge = Ridge(alpha=PREPARATION_NETWORK_RIDGE_ALPHA).fit(
            train_squared_x, scaled_y
        )
        mlp = MLPRegressor(
            hidden_layer_sizes=PREPARATION_NETWORK_HIDDEN_LAYERS,
            solver="lbfgs",
            alpha=PREPARATION_NETWORK_MLP_ALPHA,
            random_state=PREPARATION_NETWORK_SEED,
            max_iter=2000,
        ).fit(train_x, scaled_y)
        predictions = {
            "train_mean": np.full(len(test), target_mean),
            "pe_linear": target_mean
            + target_scale * (pe_test_design @ pe_coefficients),
            "pe_quadratic_ridge": target_mean
            + target_scale
            * pe_quadratic_ridge.predict(pe_quadratic_test_x),
            "ridge_7": target_mean + target_scale * ridge.predict(test_x),
            "ridge_7_squared": target_mean
            + target_scale * squared_ridge.predict(test_squared_x),
            "mlp_7_4_1": target_mean + target_scale * mlp.predict(test_x),
        }
        conditions = {
            "train_mean": np.nan,
            "pe_linear": float(np.linalg.cond(pe_design)),
            "pe_quadratic_ridge": float(
                np.linalg.cond(
                    np.column_stack([np.ones(len(train)), pe_quadratic_x])
                )
            ),
            "ridge_7": float(
                np.linalg.cond(np.column_stack([np.ones(len(train)), train_x]))
            ),
            "ridge_7_squared": float(
                np.linalg.cond(
                    np.column_stack([np.ones(len(train)), train_squared_x])
                )
            ),
            "mlp_7_4_1": np.nan,
        }
        feature_labels = {
            "train_mean": "",
            "pe_linear": "evaporating_pressure",
            "pe_quadratic_ridge": (
                "evaporating_pressure;evaporating_pressure_squared"
            ),
            "ridge_7": ";".join(PREPARATION_NETWORK_FEATURES),
            "ridge_7_squared": ";".join(
                [
                    *PREPARATION_NETWORK_FEATURES,
                    *(f"{feature}_squared" for feature in PREPARATION_NETWORK_FEATURES),
                ]
            ),
            "mlp_7_4_1": ";".join(PREPARATION_NETWORK_FEATURES),
        }
        for model, predicted in predictions.items():
            for row_index, (_, event) in enumerate(test.iterrows()):
                rows.append(
                    {
                        "cycle_name": event["cycle_name"],
                        "experiment_id": experiment,
                        "model": model,
                        "selected_features": feature_labels[model],
                        "actual_energy_kwh": event["inclusive_energy_kwh"],
                        "predicted_energy_kwh": predicted[row_index],
                        "design_condition_number": conditions[model],
                    }
                )
    predictions = pd.DataFrame(rows)
    baseline = predictions.loc[predictions["model"].eq("train_mean")]
    baseline_mse = float(
        (baseline["predicted_energy_kwh"] - baseline["actual_energy_kwh"])
        .pow(2)
        .mean()
    )
    baseline_experiment_mse = (
        baseline["predicted_energy_kwh"] - baseline["actual_energy_kwh"]
    ).pow(2).groupby(baseline["experiment_id"]).mean()
    summary_rows = []
    for model, values in predictions.groupby("model", sort=False):
        error = values["predicted_energy_kwh"] - values["actual_energy_kwh"]
        experiment_mse = error.pow(2).groupby(values["experiment_id"]).mean()
        summary_rows.append(
            {
                "model": model,
                "event_count": len(values),
                "experiment_count": values["experiment_id"].nunique(),
                "mse_kwh2": error.pow(2).mean(),
                "rmse_kwh": float(np.sqrt(error.pow(2).mean())),
                "mae_kwh": error.abs().mean(),
                "bias_kwh": error.mean(),
                "macro_experiment_mse_kwh2": experiment_mse.mean(),
                "improvement_vs_train_mean_pct": 100.0
                * (baseline_mse - error.pow(2).mean())
                / baseline_mse,
                "improved_experiment_count": int(
                    experiment_mse.lt(baseline_experiment_mse).sum()
                ),
                "negative_prediction_count": int(
                    values["predicted_energy_kwh"].lt(0).sum()
                ),
                "max_condition_number": values["design_condition_number"].max(),
            }
        )
    return pd.DataFrame(summary_rows), predictions


def write_preparation_inclusive_sensor_evidence(
    dataset: Path, output: Path
) -> pd.DataFrame:
    loader = DatasetLoader(dataset)
    tickets = pd.read_csv(output / "ticket_event_features_and_predictions.csv")
    catalog = loader.list_cycles()
    events, audit = build_preparation_inclusive_events(
        loader, tickets, catalog
    )
    model_events = events.rename(
        columns={
            "inclusive_duration_minutes": "defrost_duration_minutes",
            "t3_prepreparation_c": "coil_temperature",
            "inclusive_energy_kwh": "defrost_electricity_kwh",
        }
    )
    summary, predictions = evaluate_predefrost_sensor_increment(
        model_events, PREPARATION_SENSOR_FEATURES, include_nested=False
    )
    literature_candidates = [
        *LITERATURE_SENSOR_FEATURES,
        *LITERATURE_AUDIT_FEATURES,
    ]
    literature_summary, literature_predictions = evaluate_predefrost_sensor_increment(
        model_events, literature_candidates, include_nested=False
    )
    selected = select_fixed_literature_features(literature_summary)
    if selected != FIXED_LITERATURE_FEATURES:
        raise ValueError(f"fixed literature screen changed: {selected}")
    fixed_summary, fixed_predictions = evaluate_predefrost_sensor_increment(
        model_events,
        [],
        include_nested=False,
        fixed_combinations={FIXED_LITERATURE_COMBINATION: selected},
    )
    literature_summary = pd.concat(
        [
            literature_summary,
            fixed_summary.loc[
                fixed_summary["feature"].eq(FIXED_LITERATURE_COMBINATION)
            ],
        ],
        ignore_index=True,
    )
    literature_predictions = pd.concat(
        [
            literature_predictions,
            fixed_predictions.loc[
                fixed_predictions["feature"].eq(FIXED_LITERATURE_COMBINATION)
            ],
        ],
        ignore_index=True,
    )
    baseline_macro = float(
        literature_summary.loc[
            literature_summary["feature"].eq("__baseline__"),
            "macro_experiment_mse_kwh2",
        ].iloc[0]
    )
    literature_summary["macro_improvement_vs_baseline_pct"] = 100.0 * (
        baseline_macro - literature_summary["macro_experiment_mse_kwh2"]
    ) / baseline_macro
    literature_summary["passes_fixed_screen"] = (
        literature_summary["feature"].isin(LITERATURE_SENSOR_FEATURES)
        & literature_summary["improvement_vs_baseline_pct"].ge(
            FIXED_SCREEN_MIN_EVENT_IMPROVEMENT_PCT
        )
        & literature_summary["macro_improvement_vs_baseline_pct"].ge(
            FIXED_SCREEN_MIN_MACRO_IMPROVEMENT_PCT
        )
        & literature_summary["improved_experiment_count"].ge(
            FIXED_SCREEN_MIN_IMPROVED_EXPERIMENTS
        )
    )
    literature_summary["candidate_role"] = np.select(
        [
            literature_summary["feature"].eq("__baseline__"),
            literature_summary["feature"].isin(LITERATURE_AUDIT_FEATURES),
            literature_summary["feature"].eq(FIXED_LITERATURE_COMBINATION),
            literature_summary["feature"].isin(FIXED_LITERATURE_FEATURES),
        ],
        [
            "baseline",
            "deterministic_te_audit_only",
            "fixed_combination",
            "fixed_combination_member",
        ],
        default="screened_out",
    )
    literature_predictions["candidate_role"] = literature_predictions[
        "feature"
    ].map(literature_summary.set_index("feature")["candidate_role"])
    old = pd.read_csv(output / "predefrost_sensor_increment_summary.csv").set_index(
        "feature"
    )
    candidate = summary["feature"].isin(PREPARATION_SENSOR_FEATURES)
    summary.loc[candidate, "new_rank"] = summary.loc[candidate, "mse_kwh2"].rank(
        method="first"
    )
    for feature in PREPARATION_SENSOR_FEATURES:
        selected = summary["feature"].eq(feature)
        summary.loc[selected, "old_rank"] = old.loc[
            PREPARATION_SENSOR_FEATURES, "mse_kwh2"
        ].rank(method="first").loc[feature]
        for column in (
            "mse_kwh2",
            "macro_experiment_mse_kwh2",
            "improvement_vs_baseline_pct",
            "oof_residual_pearson",
            "oof_residual_spearman",
        ):
            summary.loc[selected, f"old_{column}"] = old.loc[feature, column]
        for outcome in ("preparation_duration_s", "preparation_energy_kwh"):
            summary.loc[selected, f"pearson_{outcome}"] = events[feature].corr(
                events[outcome], method="pearson"
            )
            summary.loc[selected, f"spearman_{outcome}"] = events[feature].corr(
                events[outcome], method="spearman"
            )
    predictions = predictions.merge(
        events[
            [
                "cycle_name",
                "inclusive_duration_minutes",
                "preparation_duration_s",
                "inclusive_energy_kwh",
                "preparation_energy_kwh",
                "t3_prepreparation_c",
            ]
        ],
        on="cycle_name",
        how="left",
        validate="many_to_one",
    )
    audit.merge(events, on="cycle_name", how="left").to_csv(
        output / "preparation_inclusive_events.csv", index=False
    )
    summary.to_csv(
        output / "preparation_inclusive_sensor_summary.csv", index=False
    )
    predictions.to_csv(
        output / "preparation_inclusive_sensor_predictions.csv", index=False
    )
    literature_summary.to_csv(
        output / "preparation_inclusive_literature_summary.csv", index=False
    )
    literature_predictions.to_csv(
        output / "preparation_inclusive_literature_predictions.csv", index=False
    )
    network_events, network_audit = build_preparation_network_cohort(loader, catalog)
    network_summary, network_predictions = evaluate_preparation_network(network_events)
    network_audit.merge(network_events, on="cycle_name", how="left").to_csv(
        output / "preparation_inclusive_network_events.csv", index=False
    )
    network_summary.to_csv(
        output / "preparation_inclusive_network_summary.csv", index=False
    )
    network_predictions.to_csv(
        output / "preparation_inclusive_network_predictions.csv", index=False
    )
    return summary


def build_recovery_events(
    loader: DatasetLoader, tickets: pd.DataFrame, catalog: pd.DataFrame
) -> pd.DataFrame:
    """Join each defrost to the next cycle and measure its complete recovery."""
    records = catalog.set_index("cycle_name")
    ticket_records = tickets.set_index("cycle_name")
    following = following_cycle_names(catalog)
    rows = []
    for cycle, next_cycle in following.items():
        current_record = records.loc[cycle]
        if pd.isna(current_record["defrost_end"]):
            continue
        next_record = records.loc[next_cycle]
        recovery_start = pd.Timestamp(next_record["heating_start"])
        defrost_end = pd.Timestamp(current_record["defrost_end"])
        gap_seconds = (recovery_start - defrost_end).total_seconds()
        if abs(gap_seconds) > 1:
            raise ValueError(f"non-adjacent defrost/recovery pair: {cycle} -> {next_cycle}")

        current = _prepare_raw_features(_load_raw(loader, cycle))
        recovered = _prepare_raw_features(_load_raw(loader, next_cycle))
        state = window_features(
            recovered,
            recovery_start,
            recovery_start + pd.Timedelta(seconds=30),
            RECOVERY_STATE_FEATURES,
        )
        preparation_start = pd.to_datetime(
            current_record["defrost_preparation_start"], errors="coerce"
        )
        predictors = preceding_features(
            current,
            pd.Timestamp(preparation_start)
            if pd.notna(preparation_start)
            else current["timestamp"].min(),
            include_dynamics=True,
        )
        start_error = (
            state["water_temperature_setpoint"] - state["water_out_temperature"]
        )
        pre_error = (
            predictors["water_temperature_setpoint"]
            - predictors["water_out_temperature"]
        )
        ticket = ticket_records.loc[cycle]
        valid = bool(ticket["valid"])
        row = {
            "cycle_name": cycle,
            "next_cycle_name": next_cycle,
            "experiment_id": current_record["experiment_id"],
            "defrost_end": defrost_end,
            "recovery_start": recovery_start,
            "adjacent_gap_seconds": gap_seconds,
            "recovery_valid": valid,
            "recovery_invalid_reason": "" if valid else ticket["invalid_reason"],
            "recovery_start_water_temperature_error": start_error,
            "pre_water_temperature_error": pre_error,
            **{f"recovery_start_{key}": value for key, value in state.items()},
            **{f"pre_{key}": value for key, value in predictors.items()},
        }
        if valid:
            recovery_stable = pd.Timestamp(ticket["recovery_stable"])
            interval = recovered.loc[
                recovered["timestamp"].between(recovery_start, recovery_stable)
            ]
            electricity, electricity_coverage = integrate_energy_kwh(
                interval["timestamp"], interval["power_total"]
            )
            water_heat, water_heat_coverage = integrate_energy_kwh(
                interval["timestamp"], interval["q_heating_kw"]
            )
            early = window_features(
                recovered,
                recovery_start,
                min(recovery_stable, recovery_start + pd.Timedelta(seconds=120)),
                [
                    "compressor_frequency_setpoint",
                    "compressor_frequency",
                    "water_out_temperature",
                    "q_heating_kw",
                    *PI_STATE_COLUMNS.values(),
                ],
                include_dynamics=True,
            )
            if not np.isclose(electricity, ticket["recovery_electricity_kwh"]):
                raise ValueError(f"recovery electricity mismatch: {cycle}")
            duration_minutes = (recovery_stable - recovery_start).total_seconds() / 60
            duration_hours = duration_minutes / 60
            row.update(
                {
                    "recovery_stable": recovery_stable,
                    "recovery_duration_minutes": duration_minutes,
                    "recovery_electricity_kwh": electricity,
                    "recovery_water_heat_kwh": water_heat,
                    "recovery_mean_power_kw": electricity / duration_hours,
                    "recovery_mean_water_heat_kw": water_heat / duration_hours,
                    "recovery_electricity_coverage": electricity_coverage,
                    "recovery_water_heat_coverage": water_heat_coverage,
                    **{f"recovery_early_{key}": value for key, value in early.items()},
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_recovery_start_state(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize between-event variation in the first 30 s of recovery."""
    rows = []
    for feature in RECOVERY_STATE_FEATURES:
        column = f"recovery_start_{feature}"
        values = pd.to_numeric(events[column], errors="coerce").dropna()
        grouped = events.loc[values.index, ["experiment_id", column]].dropna()
        total = ((values - values.mean()) ** 2).sum()
        between = sum(
            len(group) * (group[column].mean() - values.mean()) ** 2
            for _, group in grouped.groupby("experiment_id")
        )
        rows.append(
            {
                "feature": feature,
                "event_count": len(values),
                "mean": values.mean(),
                "sd": values.std(),
                "median": values.median(),
                "iqr": values.quantile(0.75) - values.quantile(0.25),
                "minimum": values.min(),
                "maximum": values.max(),
                "between_experiment_variance_fraction": between / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_recovery_outcomes(events: pd.DataFrame) -> pd.DataFrame:
    """Describe complete recovery duration, electricity and delivered heat."""
    rows = []
    for outcome in [*RECOVERY_OUTCOMES, *RECOVERY_RATE_OUTCOMES]:
        coverage_column = outcome.replace("_kwh", "_coverage") if outcome.endswith("_kwh") else None
        usable = events[outcome].notna()
        if coverage_column and coverage_column in events:
            usable &= events[coverage_column].ge(0.95)
        values = events.loc[usable, outcome]
        rows.append(
            {
                "outcome": outcome,
                "event_count": len(values),
                "mean": values.mean(),
                "sd": values.std(),
                "cv": values.std() / abs(values.mean()),
                "median": values.median(),
                "iqr": values.quantile(0.75) - values.quantile(0.25),
                "minimum": values.min(),
                "maximum": values.max(),
            }
        )
    return pd.DataFrame(rows)


def summarize_recovery_drivers(events: pd.DataFrame) -> pd.DataFrame:
    """Rank physically interpretable recovery drivers within each water setpoint."""
    features = {
        "pre_defrost": [
            "pre_water_temperature_error",
            "pre_water_in_temperature",
            "pre_water_out_temperature",
            "pre_q_heating_kw",
            "pre_ambient_temperature",
            "pre_coil_temperature",
            "pre_evaporating_pressure",
            "pre_condensing_pressure",
            "pre_compressor_frequency_setpoint",
            "pre_compressor_frequency",
            "pre_compressor_power",
            *(f"pre_{feature}" for feature in PI_STATE_COLUMNS.values()),
        ],
        "recovery_start": [
            "recovery_start_water_temperature_error",
            "recovery_start_water_in_temperature",
            "recovery_start_water_out_temperature",
            "recovery_start_coil_temperature",
            "recovery_start_evaporating_pressure",
            "recovery_start_condensing_pressure",
            "recovery_start_compressor_frequency_setpoint",
            "recovery_start_compressor_frequency",
            "recovery_start_compressor_power",
            *(f"recovery_start_{feature}" for feature in PI_STATE_COLUMNS.values()),
        ],
        "first_120_seconds": [
            "recovery_early_compressor_frequency_setpoint_slope_per_min",
            "recovery_early_compressor_frequency_slope_per_min",
            "recovery_early_water_out_temperature_slope_per_min",
            "recovery_early_q_heating_kw_slope_per_min",
            "recovery_early_pi_step",
            "recovery_early_pi_power",
            "recovery_early_pi_step_limit",
            "recovery_early_compressor_limit_code",
            "recovery_early_compressor_frequency_state",
        ],
    }
    rows = []
    setpoint = "recovery_start_water_temperature_setpoint"
    valid = events.loc[events["recovery_valid"]].copy()
    for availability, columns in features.items():
        for feature in columns:
            if feature not in valid:
                continue
            for outcome in RECOVERY_OUTCOMES:
                values = valid[[setpoint, feature, outcome]].apply(
                    pd.to_numeric, errors="coerce"
                ).dropna()
                if len(values) < 3:
                    continue
                residual = values[[feature, outcome]] - values.groupby(setpoint)[
                    [feature, outcome]
                ].transform("mean")
                raw_correlation = (
                    values[feature].corr(values[outcome], method="spearman")
                    if values[feature].nunique() > 1
                    else float("nan")
                )
                within_correlation = (
                    residual[feature].corr(residual[outcome], method="spearman")
                    if residual[feature].nunique() > 1
                    else float("nan")
                )
                rows.append(
                    {
                        "availability": availability,
                        "feature": feature,
                        "outcome": outcome,
                        "event_count": len(values),
                        "spearman": raw_correlation,
                        "within_setpoint_spearman": within_correlation,
                    }
                )
    return pd.DataFrame(rows)


def evaluate_recovery_predictors(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a small, literature-driven set of pre-defrost recovery models."""
    ts = ["pre_water_temperature_setpoint"]
    thermal_deficit = [
        *ts,
        "pre_water_temperature_error",
        "pre_water_in_temperature",
        "pre_water_flow",
        "pre_q_heating_kw",
    ]
    capacity_margin = [
        *ts,
        "pre_ambient_temperature",
        "pre_environment_relative_humidity",
        "pre_coil_temperature",
        "pre_evaporating_pressure",
        "pre_condensing_pressure",
        "pre_compressor_frequency_setpoint",
        "pre_compressor_frequency",
        "pre_compressor_power",
    ]
    controller_state = [
        *ts,
        "pre_compressor_frequency_setpoint",
        "pre_compressor_frequency",
        "pre_pi_step",
        "pre_pi_power",
        "pre_pi_step_limit",
        "pre_compressor_limit_code",
        "pre_compressor_frequency_state",
    ]
    models = [
        ("ts_only", ts),
        (
            "thermal_deficit",
            thermal_deficit,
        ),
        ("capacity_margin", capacity_margin),
        ("controller_state", controller_state),
        ("physical_prestate", list(dict.fromkeys([*thermal_deficit, *capacity_margin]))),
    ]
    rows = []
    for experiment in sorted(events["experiment_id"].unique()):
        for outcome in RECOVERY_OUTCOMES:
            coverage_column = (
                outcome.replace("_kwh", "_coverage")
                if outcome.endswith("_kwh")
                else None
            )
            usable = events[outcome].notna()
            if coverage_column and coverage_column in events:
                usable &= events[coverage_column].ge(0.95)
            train = events.loc[usable & ~events["experiment_id"].eq(experiment)]
            test = events.loc[usable & events["experiment_id"].eq(experiment)]
            if train.empty or test.empty:
                continue
            for model, features in models:
                predicted = _ridge_predict(train, test, features, outcome)
                for index, (_, event) in enumerate(test.iterrows()):
                    rows.append(
                        {
                            "cycle_name": event["cycle_name"],
                            "experiment_id": experiment,
                            "outcome": outcome,
                            "model": model,
                            "selected_features": ";".join(features),
                            "actual": event[outcome],
                            "predicted": predicted[index],
                        }
                    )
    predictions = pd.DataFrame(rows).dropna(subset=["actual", "predicted"])
    summaries = []
    for outcome, outcome_rows in predictions.groupby("outcome", sort=False):
        baseline = outcome_rows.loc[outcome_rows["model"].eq("ts_only")]
        baseline_rmse = float(
            np.sqrt((baseline["predicted"] - baseline["actual"]).pow(2).mean())
        )
        baseline_by_experiment = (
            (baseline["predicted"] - baseline["actual"])
            .pow(2)
            .groupby(baseline["experiment_id"])
            .mean()
            .pow(0.5)
        )
        for model, values in outcome_rows.groupby("model", sort=False):
            error = values["predicted"] - values["actual"]
            by_experiment = error.pow(2).groupby(values["experiment_id"]).mean().pow(0.5)
            summaries.append(
                {
                    "outcome": outcome,
                    "model": model,
                    "selected_features": values["selected_features"].iloc[0],
                    "event_count": len(values),
                    "experiment_count": values["experiment_id"].nunique(),
                    "event_rmse": np.sqrt(error.pow(2).mean()),
                    "event_mae": error.abs().mean(),
                    "macro_rmse": by_experiment.mean(),
                    "improvement_vs_ts_pct": 100
                    * (baseline_rmse - np.sqrt(error.pow(2).mean()))
                    / baseline_rmse,
                    "improved_experiment_count": int(
                        by_experiment.lt(baseline_by_experiment).sum()
                    ),
                }
            )
    return pd.DataFrame(summaries), predictions


def build_ticket_features(
    loader: DatasetLoader,
    tickets: pd.DataFrame,
    points: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    valid = tickets.loc[tickets["valid"]].merge(
        points[["cycle_name", "actual_minutes_from_stable"]], on="cycle_name", how="left"
    ).merge(
        catalog[["cycle_name", "experiment_id", "defrost_start", "defrost_end"]].rename(
            columns={
                "defrost_start": "catalog_defrost_start",
                "defrost_end": "catalog_defrost_end",
            }
        ),
        on="cycle_name",
        how="left",
    )
    valid = valid.rename(columns={"actual_minutes_from_stable": "minutes_from_stable"})
    start = pd.to_datetime(valid["catalog_defrost_start"], errors="coerce")
    end = pd.to_datetime(valid["catalog_defrost_end"], errors="coerce")
    duration = (end - start).dt.total_seconds() / 60.0
    valid["defrost_duration_minutes"] = duration.where(duration.gt(0))
    valid["mean_defrost_power_kw"] = (
        valid["defrost_electricity_kwh"] / (valid["defrost_duration_minutes"] / 60.0)
    )
    rows = []
    for event in valid.itertuples(index=False):
        frame = _prepare_raw_features(_load_raw(loader, event.cycle_name))
        active_defrost = frame.loc[
            frame["timestamp"].ge(pd.Timestamp(event.catalog_defrost_start))
            & frame["timestamp"].lt(pd.Timestamp(event.catalog_defrost_end))
        ]
        signed_heat, signed_heat_coverage = integrate_energy_kwh(
            active_defrost["timestamp"], active_defrost["q_heating_kw"]
        )
        rows.append(
            {
                "cycle_name": event.cycle_name,
                "defrost_absorbed_heat_kwh": -signed_heat,
                "defrost_signed_heat_coverage": signed_heat_coverage,
                **preceding_features(
                    frame,
                    pd.Timestamp(event.catalog_defrost_start),
                    include_dynamics=True,
                ),
            }
        )
    return valid.merge(pd.DataFrame(rows), on="cycle_name")


def build_candidate_features(
    loader: DatasetLoader,
    curves: pd.DataFrame,
    points: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    stable = points.set_index("cycle_name")["t_heating_stable"]
    curves = curves.loc[curves["cycle_name"].isin(stable.index)].copy()
    for cycle, candidates in curves.groupby("cycle_name", sort=True):
        frame = _prepare_raw_features(_load_raw(loader, cycle))
        start = pd.Timestamp(stable.loc[cycle])
        for candidate in candidates.itertuples(index=False):
            time = pd.Timestamp(candidate.candidate_time)
            rows.append(
                {
                    "cycle_name": cycle,
                    "candidate_time": time,
                    "minutes_from_stable": (time - start).total_seconds() / 60,
                    **preceding_features(frame, time),
                }
            )
    result = curves.merge(pd.DataFrame(rows), on=["cycle_name", "candidate_time"])
    if "experiment_id" not in result:
        result = result.merge(
            catalog[["cycle_name", "experiment_id"]], on="cycle_name", how="left"
        )
    return result


def ticket_model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, values in predictions.groupby("experiment_id", sort=True):
        for strategy in STRATEGIES:
            for label, outcome in OUTCOMES.items():
                prediction = f"predicted_{strategy}_{label}"
                if prediction not in values:
                    continue
                error = (
                    values[prediction] - values[outcome]
                ).abs()
                rows.append(
                    {
                        "experiment_id": experiment,
                        "strategy": strategy,
                        "outcome": label,
                        "protocol": (
                            "leave_one_event_out_within_experiment"
                            if strategy == "partial_pool"
                            else "leave_one_experiment_out"
                        ),
                        "mae": error.mean(),
                        "event_count": len(values),
                    }
                )
    return pd.DataFrame(rows)


def build_ticket_evidence(
    loader: DatasetLoader,
    tickets: pd.DataFrame,
    points: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the existing ticket table plus the held-out defrost rule comparison."""
    features = build_ticket_features(loader, tickets, points, catalog)
    rule_duration = (
        build_defrost_power_rows(loader, features)
        .groupby("cycle_name", as_index=False)["duration_rule_s"]
        .first()
    )
    rule_duration["rule_defrost_duration_minutes"] = (
        rule_duration.pop("duration_rule_s") / 60.0
    )
    features = features.merge(rule_duration, on="cycle_name", validate="one_to_one")
    predictions = leave_one_experiment_out_ticket_predictions(
        features, STATE_FEATURES, DYNAMIC_FEATURES
    ).merge(leave_one_event_out_partial_pool(features), on="cycle_name", how="left")
    defrost = leave_one_experiment_out_defrost_predictions(features)
    added = [
        "cycle_name",
        "predicted_fixed_defrost_electricity",
        "predicted_known_duration_defrost_electricity",
        "predicted_t3_duration_minutes",
        "predicted_t3_rule_defrost_electricity",
        "training_mean_defrost_power_kw",
        *(
            column
            for column in defrost
            if column.startswith("predicted_")
            and column.endswith("_defrost_absorbed_heat")
        ),
    ]
    predictions = predictions.merge(defrost[added], on="cycle_name", how="left")
    metrics = pd.concat(
        [ticket_model_metrics(predictions), defrost_model_metrics(predictions)],
        ignore_index=True,
        sort=False,
    )
    return predictions, metrics


def write_defrost_rule_evidence(dataset: Path, source: Path, output: Path) -> None:
    """Update only the two existing ticket evidence CSV files."""
    loader = DatasetLoader(dataset)
    tickets = pd.read_csv(source / "defrost_ticket_events.csv")
    points = pd.read_csv(source / "cycle_optimal_points.csv")
    predictions, metrics = build_ticket_evidence(loader, tickets, points, loader.list_cycles())
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "ticket_event_features_and_predictions.csv", index=False)
    metrics.to_csv(output / "ticket_model_metrics_by_experiment.csv", index=False)


def write_recovery_evidence(dataset: Path, source: Path, output: Path) -> pd.DataFrame:
    """Write the reusable cross-cycle recovery audit and held-out models."""
    loader = DatasetLoader(dataset)
    events = build_recovery_events(
        loader,
        pd.read_csv(source / "defrost_ticket_events.csv"),
        loader.list_cycles(),
    )
    state = summarize_recovery_start_state(events)
    outcomes = summarize_recovery_outcomes(events)
    drivers = summarize_recovery_drivers(events)
    metrics, predictions = evaluate_recovery_predictors(events)
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output / "recovery_events.csv", index=False)
    state.to_csv(output / "recovery_start_state_summary.csv", index=False)
    outcomes.to_csv(output / "recovery_outcome_summary.csv", index=False)
    drivers.to_csv(output / "recovery_driver_correlations.csv", index=False)
    events[RECOVERY_OUTCOMES].corr().rename_axis("outcome").reset_index().to_csv(
        output / "recovery_outcome_correlation.csv", index=False
    )
    metrics.to_csv(output / "recovery_predictor_metrics.csv", index=False)
    predictions.to_csv(output / "recovery_predictor_predictions.csv", index=False)
    plot_recovery_start_states(
        events, output.parent / "图表" / "figure_recovery_start_states"
    )
    plot_recovery_outcome_by_cycle(
        events,
        "recovery_electricity_kwh",
        "Recovery electricity [kWh]",
        "Electricity consumed during post-defrost recovery",
        output.parent / "图表" / "figure_recovery_electricity_by_cycle",
    )
    plot_recovery_outcome_by_cycle(
        events,
        "recovery_water_heat_kwh",
        "Water-side heat during recovery [kWh]",
        "Water-side heat delivered during post-defrost recovery",
        output.parent / "图表" / "figure_recovery_water_heat_by_cycle",
    )
    plot_recovery_predictors(
        metrics, output.parent / "图表" / "figure_recovery_predictors"
    )
    return metrics


def partial_pool_optimal_points(curves: pd.DataFrame) -> pd.DataFrame:
    renamed = curves.rename(
        columns={"renewal_cost_partial_pool": "renewal_cost_conditional"}
    )
    return conditional_optimal_points(renamed).rename(
        columns={
            "t_star_conditional": "t_star_partial_pool",
            "rho_min_conditional": "rho_min_partial_pool",
            "near_opt_start_conditional": "near_opt_start_partial_pool",
            "near_opt_end_conditional": "near_opt_end_partial_pool",
            "minimum_location_conditional": "minimum_location_partial_pool",
        }
    )


def merge_rb_points(curves: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """Attach the first causal RB trigger to each publication cost curve."""
    return curves.merge(
        points[["cycle_name", "t_RB", "rb_status", "trigger_type"]],
        on="cycle_name",
        how="left",
        validate="many_to_one",
    )


def render_cost_publication(
    loader: DatasetLoader,
    cycle_name: str,
    curves: pd.DataFrame,
    output: Path | None = None,
    *,
    unit_heat: bool = False,
) -> pd.DataFrame:
    """Render one cycle with causal RB/optimal RGB evidence and three panels."""
    options = {"unit_heat": True} if unit_heat else {}
    prepared = _prepare_cost_publication(loader, cycle_name, curves, **options)
    return _render_prepared_cost_publication(
        prepared, output, loader.dataset_root, **options
    )


def _prepare_cost_publication(
    loader: DatasetLoader,
    cycle_name: str,
    curves: pd.DataFrame,
    cycle_dir: Path | None = None,
    *,
    unit_heat: bool = False,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame | None,
    dict[str, Mapping[str, object]],
    pd.DataFrame,
]:
    """Load one cycle and resolve both decision frames once for all output formats."""
    record = loader.get_cycle_record(cycle_name)
    curve_columns = [
        "candidate_time",
        "inverse_cop_unit" if unit_heat else "inverse_cop",
        "relative_regret_unit" if unit_heat else "relative_regret",
        "optimization_eligible",
        "candidate_in_interpolated_gap",
        "candidate_in_extrapolated_endpoint",
        "support_status",
        "actual_preparation_time",
        "t_RB",
        "rb_status",
        "trigger_type",
    ]
    curve = curves.loc[
        curves["cycle_name"].eq(cycle_name),
        [column for column in curve_columns if column in curves],
    ].rename(
        columns={
            "inverse_cop_unit": "inverse_cop",
            "relative_regret_unit": "relative_regret",
        }
    )
    curve["cycle_cop"] = 1 / pd.to_numeric(curve["inverse_cop"], errors="coerce")
    if not unit_heat and "minimum_location" in curves:
        curve["minimum_location"] = curves.loc[
            curves["cycle_name"].eq(cycle_name), "minimum_location"
        ].to_numpy()
    # Keep the light-weight test/dry-run loader contract backward compatible.
    if not hasattr(loader, "load_cycle"):
        return record, curve, None, {}, pd.DataFrame()

    frame = loader.load_cycle(cycle_name)
    metadata = loader.load_image_metadata(cycle_name)
    images = (
        loader.load_cycle_images(cycle_name)
        if cycle_dir is None
        else scan_cycle_images(loader.dataset_root, cycle_name, metadata, cycle_dir=cycle_dir)
    )
    frame = frame.sort_values("timestamp", kind="stable").copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    eligible = curve["optimization_eligible"].fillna(False) & pd.to_numeric(
        curve["inverse_cop"], errors="coerce"
    ).notna()
    t_star = (
        pd.Timestamp(curve.loc[curve.loc[eligible, "inverse_cop"].idxmin(), "candidate_time"])
        if eligible.any()
        else None
    )
    rb_time = pd.NaT
    status = curve.get("rb_status", pd.Series(dtype=str)).dropna().astype(str)
    rb_times = pd.to_datetime(curve.get("t_RB", pd.Series(dtype=object)), errors="coerce").dropna()
    if not status.empty and status.iloc[0] == "triggered" and not rb_times.empty:
        rb_time = pd.Timestamp(rb_times.iloc[0])
    matches = match_decision_rgb_images(
        metadata,
        images,
        {"rb": rb_time, "optimal": t_star},
    )
    matches.insert(0, "cycle_name", cycle_name)
    matches["source_relative_path"] = matches["file_name"].map(
        lambda name: f"images/{cycle_name}/front/{name}" if name else ""
    )
    matches["available_at_render"] = matches["available"]
    decision_images = {
        str(row["target_type"]): row.to_dict()
        for _, row in matches.iterrows()
    }
    return record, curve, frame, decision_images, matches


def _render_prepared_cost_publication(
    prepared: tuple[
        dict[str, object],
        pd.DataFrame,
        pd.DataFrame | None,
        dict[str, Mapping[str, object]],
        pd.DataFrame,
    ],
    output: Path | None,
    dataset_root: Path | None = None,
    *,
    unit_heat: bool = False,
) -> pd.DataFrame:
    if output is None:
        raise ValueError("report output path is required; Dataset assets are read-only")
    record, curve, frame, decision_images, matches = prepared
    if frame is None:
        render_publication_asset(
            Path(".") if dataset_root is None else dataset_root,
            record,
            output_path=output,
            cost_curve=curve,
        )
    else:
        labels = (
            {
                "optimal_label": "Unit-heat optimum",
                "cost_label": "Unit-reported heat inverse COP [-]",
            }
            if unit_heat
            else {}
        )
        render_decision_publication(
            frame,
            record,
            curve,
            decision_images,
            output,
            full_candidate_domain=True,
            **labels,
        )
    return matches


def render_all_cost_publications(  # noqa: C901
    loader: DatasetLoader,
    curves: pd.DataFrame,
    output: Path,
    *,
    match_output: Path | None = None,
    fetch_cloud: bool = False,
    cloud_root: Path | None = None,
    cleanup_downloaded: bool = False,
    minimum_free_gib: float = 50,
    unit_heat: bool = False,
) -> list[str]:
    """Write one report publication per complete observed cycle."""
    cycles = complete_observed_cycle_names(loader.list_cycles(), curves)
    if unit_heat:
        eligible_unit = curves["optimization_eligible"].fillna(False) & pd.to_numeric(
            curves["inverse_cop_unit"], errors="coerce"
        ).notna()
        unit_cycles = set(curves.loc[eligible_unit, "cycle_name"].astype(str))
        cycles = [cycle for cycle in cycles if cycle in unit_cycles]
    matches: list[pd.DataFrame] = []
    options = {"unit_heat": True} if unit_heat else {}
    for cycle_name in cycles:
        if hasattr(loader, "load_cycle"):
            prepared = _prepare_cost_publication(loader, cycle_name, curves, **options)
            _record, _curve, _frame, _decision_images, initial_matches = prepared
            missing_names: list[str] = []
            if "file_name" in initial_matches and "status" in initial_matches:
                missing_names = sorted(
                    {
                        str(name)
                        for name in initial_matches.loc[
                            initial_matches["status"].eq("physical_image_missing"), "file_name"
                        ]
                        if str(name)
                    }
                )
            if missing_names and fetch_cloud:
                with materialize_cycle_image_members(
                    loader.dataset_root,
                    cycle_name,
                    missing_names,
                    fetch_cloud=True,
                    cloud_root=cloud_root,
                    minimum_free_gib=minimum_free_gib,
                ) as cycle_dir:
                    prepared = _prepare_cost_publication(
                        loader, cycle_name, curves, cycle_dir, **options
                    )
                    suffixes = ("png",) if unit_heat else ("svg", "png")
                    for index, suffix in enumerate(suffixes):
                        rendered = _render_prepared_cost_publication(
                            prepared,
                            output / f"{cycle_name}{'_J_unit' if unit_heat else ''}.{suffix}",
                            **options,
                        )
                        if index == 0:
                            matches.append(rendered)
            elif initial_matches["status"].eq("physical_image_missing").any():
                from frost_analysis.dataset.images import materialize_cycle_images

                with materialize_cycle_images(
                    loader.dataset_root,
                    cycle_name,
                    fetch_cloud=fetch_cloud,
                    cloud_root=cloud_root,
                    cleanup_downloaded=cleanup_downloaded,
                    minimum_free_gib=minimum_free_gib,
                ):
                    prepared = _prepare_cost_publication(
                        loader, cycle_name, curves, **options
                    )
                    suffixes = ("png",) if unit_heat else ("svg", "png")
                    for index, suffix in enumerate(suffixes):
                        rendered = _render_prepared_cost_publication(
                            prepared,
                            output / f"{cycle_name}{'_J_unit' if unit_heat else ''}.{suffix}",
                            **options,
                        )
                        if index == 0:
                            matches.append(rendered)
            else:
                suffixes = ("png",) if unit_heat else ("svg", "png")
                for index, suffix in enumerate(suffixes):
                    rendered = _render_prepared_cost_publication(
                        prepared,
                        output / f"{cycle_name}{'_J_unit' if unit_heat else ''}.{suffix}",
                        **options,
                    )
                    if index == 0:
                        matches.append(rendered)
        else:
            matches.append(
                render_cost_publication(
                    loader,
                    cycle_name,
                    curves,
                    output / f"{cycle_name}{'_J_unit.png' if unit_heat else '.svg'}",
                    **options,
                )
            )
    _remove_stale_cycle_figures(
        output,
        {f"{cycle}{'_J_unit' if unit_heat else ''}" for cycle in cycles},
        filename_suffix="_J_unit" if unit_heat else "",
    )
    if match_output is not None and matches:
        match_output.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(matches, ignore_index=True).to_csv(match_output, index=False)
    return cycles


def render_representative_cost_publication(
    loader: DatasetLoader,
    points: pd.DataFrame,
    curves: pd.DataFrame,
    output: Path,
) -> str:
    primary = points.loc[points["valid"] & points["primary_analysis"]].copy()
    interior = primary.loc[primary["minimum_location"].eq("interior")]
    pool = interior if not interior.empty else primary
    median_advance = pool["minutes_earlier_than_actual"].median()
    representative = pool.iloc[
        (pool["minutes_earlier_than_actual"] - median_advance).abs().argmin()
    ]
    cycle_name = str(representative["cycle_name"])
    paths = (output.with_suffix(".svg"), output.with_suffix(".pdf"), output.with_suffix(".png"))
    for path in paths:
        render_cost_publication(loader, cycle_name, curves, path)
    return cycle_name


def build_window_overview(
    loader: DatasetLoader,
    points: pd.DataFrame,
    image_labels: pd.DataFrame,
    dataset: Path,
) -> pd.DataFrame:
    rows = []
    images = image_labels.loc[image_labels["camera_role"].eq("front")].copy()
    images["image_time"] = pd.to_datetime(images["image_time"], errors="coerce")
    for point in points.loc[points["valid"]].itertuples(index=False):
        t_star = pd.Timestamp(point.t_star)
        features = preceding_features(_load_raw(loader, point.cycle_name), t_star)
        available = images.loc[images["cycle_name"].eq(point.cycle_name)].copy()
        if available.empty:
            image_time = pd.NaT
            image_path = ""
        else:
            available["distance"] = (available["image_time"] - t_star).abs()
            nearest = available.loc[available["distance"].idxmin()]
            image_time = nearest["image_time"]
            candidate = dataset / str(nearest["image_path"])
            image_path = str(candidate) if candidate.is_file() else ""
        rows.append(
            {
                "cycle_name": point.cycle_name,
                "cohort_tier": point.cohort_tier,
                "minimum_location": point.minimum_location,
                "is_censored": point.is_censored,
                "minutes_from_stable": point.minutes_from_stable,
                "near_opt_start_minutes": (
                    pd.Timestamp(point.near_opt_start) - pd.Timestamp(point.t_heating_stable)
                ).total_seconds()
                / 60,
                "near_opt_end_minutes": (
                    pd.Timestamp(point.near_opt_end) - pd.Timestamp(point.t_heating_stable)
                ).total_seconds()
                / 60,
                "t_star": t_star,
                "cop_at_t_star_60s": features["cop"],
                "q_at_t_star_60s_kw": features["q_heating_kw"],
                "power_at_t_star_60s_kw": features["power_total"],
                "front_image_time": image_time,
                "front_image_path": image_path,
                "front_image_available": bool(image_path),
                "front_image_offset_seconds": abs((image_time - t_star).total_seconds())
                if pd.notna(image_time)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["minutes_from_stable", "cycle_name"], kind="stable"
    ).reset_index(drop=True)


def build_defrost_energy_method_source(
    power_metrics: pd.DataFrame,
    duration_metrics: pd.DataFrame,
    literature_summary: pd.DataFrame,
    network_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build the prespecified relative-MSE comparison table."""
    groups = [
        {
            "group_id": "A",
            "group_title": "Can power trajectories reconstruct defrost energy?",
            "target": "energy during defrost",
            "group_information_timing": "observed defrost duration used",
            "source_file": "defrost_power_metrics.csv",
            "source_filter": "duration_mode=rule",
            "key": "model",
            "frame": power_metrics.loc[power_metrics["duration_mode"].eq("rule")],
            "experiment_count": None,
            "methods": [
                (
                    "markus_original",
                    "Observed duration\nPublished power curve (reference)",
                    "baseline",
                    "post-event/process; rule D",
                ),
                (
                    "one_stage_ls",
                    "Observed duration\nOne-stage least squares",
                    "comparison",
                    "post-event/process; rule D",
                ),
                (
                    "two_stage_ls",
                    "Observed duration\nTwo-stage least squares",
                    "comparison",
                    "post-event/process; rule D",
                ),
            ],
            "baseline": "markus_original",
        },
        {
            "group_id": "B",
            "group_title": "Which duration-temperature formulation best predicts defrost energy?",
            "target": "energy during defrost",
            "group_information_timing": (
                "observed duration unless the label says predicted"
            ),
            "source_file": "duration_t3_energy_metrics.csv",
            "source_filter": "duration_mode=rule",
            "key": "model",
            "frame": duration_metrics.loc[
                duration_metrics["duration_mode"].eq("rule")
            ],
            "experiment_count": None,
            "methods": [
                (
                    "fixed_mean_energy",
                    "No event-level inputs\nHistorical mean",
                    "comparison",
                    "pre-action; training mean",
                ),
                (
                    "duration_only",
                    "Observed duration\nAverage defrost power (reference)",
                    "baseline",
                    "during/post-event; rule-derived D",
                ),
                (
                    "old_t3_duration",
                    "Coil temperature\nDuration model × average power",
                    "comparison",
                    "pre-action; T3-predicted D",
                ),
                (
                    "duration_t3_physical",
                    "Duration + coil temperature\nTemperature-adjusted power",
                    "comparison",
                    "during/post-event; rule-derived D + pre-start T3",
                ),
                (
                    "additive_sensitivity",
                    "Duration + coil temperature\nAdditive linear model",
                    "comparison",
                    "during/post-event; rule-derived D + pre-start T3",
                ),
            ],
            "baseline": "duration_only",
        },
        {
            "group_id": "C",
            "group_title": "Do additional physical predictors improve generalization?",
            "target": "energy from preparation start through defrost",
            "group_information_timing": (
                "observed duration used; one predictor added at a time"
            ),
            "source_file": "preparation_inclusive_literature_summary.csv",
            "source_filter": "prespecified main-comparison features",
            "key": "feature",
            "frame": literature_summary,
            "experiment_count": 15,
            "methods": [
                (
                    "__baseline__",
                    "Duration + coil temperature\nLinear regression (reference)",
                    "baseline",
                    "post-event; inclusive D + pre-action T3",
                ),
                (
                    "evaporating_pressure",
                    "Duration + coil temperature + Pe\nLinear regression",
                    "comparison",
                    "post-event; inclusive D + pre-action Pe",
                ),
                (
                    "cop",
                    "Duration + coil temperature + COP\nLinear regression",
                    "comparison",
                    "post-event; inclusive D + pre-action COP",
                ),
                (
                    "power_total",
                    "Duration + coil temperature + total power\nLinear regression",
                    "comparison",
                    "post-event; inclusive D + pre-action total power",
                ),
                (
                    "__fixed_pe_cop_power_total__",
                    "Duration + coil temperature + Pe + COP + power\nLinear regression",
                    "rejected_complex",
                    "post-event; inclusive D + pre-action features",
                ),
            ],
            "baseline": "__baseline__",
        },
        {
            "group_id": "D",
            "group_title": "A minimal pre-action model achieves the best generalization",
            "target": "energy from preparation start through defrost",
            "group_information_timing": "pre-action inputs only; no observed duration",
            "source_file": "preparation_inclusive_network_summary.csv",
            "source_filter": "six prespecified complete-case models",
            "key": "model",
            "frame": network_summary,
            "experiment_count": None,
            "methods": [
                (
                    "train_mean",
                    "No event-level inputs\nHistorical mean (reference)",
                    "baseline",
                    "pre-action; no true D",
                ),
                (
                    "pe_linear",
                    "Pe only\nLinear regression",
                    "comparison",
                    "pre-action; no true D",
                ),
                (
                    "pe_quadratic_ridge",
                    "Pe + Pe²\nQuadratic Ridge regression",
                    "selected_deployable",
                    "pre-action; no true D",
                ),
                (
                    "ridge_7",
                    "7 physical variables\nRidge regression",
                    "comparison",
                    "pre-action; no true D",
                ),
                (
                    "ridge_7_squared",
                    "7 variables + individual squares\nRidge regression",
                    "comparison",
                    "pre-action; no true D",
                ),
                (
                    "mlp_7_4_1",
                    "7 physical variables\nMLP (7-4-1)",
                    "rejected_complex",
                    "pre-action; no true D",
                ),
            ],
            "baseline": "train_mean",
        },
    ]
    rows = []
    for group in groups:
        method_ids = [method[0] for method in group["methods"]]
        selected = group["frame"].loc[group["frame"][group["key"]].isin(method_ids)]
        if len(selected) != len(method_ids) or selected[group["key"]].duplicated().any():
            raise ValueError(
                f"{group['group_id']} comparison requires one row for each of {method_ids}"
            )
        selected = selected.set_index(group["key"])
        baseline = selected.loc[group["baseline"]]
        for method_id, label, role, timing in group["methods"]:
            values = selected.loc[method_id]
            event_count_column = "event_count" if "event_count" in values else "valid_n"
            experiment_count = group["experiment_count"]
            if experiment_count is None:
                experiment_count = int(values["experiment_count"])
            rows.append(
                {
                    "group_id": group["group_id"],
                    "group_title": group["group_title"],
                    "target": group["target"],
                    "event_count": int(values[event_count_column]),
                    "experiment_count": experiment_count,
                    "group_information_timing": group["group_information_timing"],
                    "information_timing": timing,
                    "method_id": method_id,
                    "method_label": label,
                    "method_role": role,
                    "source_file": group["source_file"],
                    "source_filter": group["source_filter"],
                    "event_mse_kwh2": float(values["mse_kwh2"]),
                    "macro_mse_kwh2": float(
                        values["macro_experiment_mse_kwh2"]
                    ),
                    "event_relative_mse_pct": 100.0
                    * float(values["mse_kwh2"])
                    / float(baseline["mse_kwh2"]),
                    "macro_relative_mse_pct": 100.0
                    * float(values["macro_experiment_mse_kwh2"])
                    / float(baseline["macro_experiment_mse_kwh2"]),
                }
            )
    source = pd.DataFrame(rows)
    baseline = source["method_role"].eq("baseline")
    source.loc[
        baseline, ["event_relative_mse_pct", "macro_relative_mse_pct"]
    ] = 100.0
    return source


def plot_defrost_energy_method_comparison(
    source: pd.DataFrame,
    output: Path,
    *,
    title: str = (
        "A regularized pressure curve generalizes better than multivariable "
        "pre-defrost models"
    ),
    height_mm: float = 170,
) -> None:
    """Render one paired-dot comparison of all prespecified energy methods."""
    colors = {
        "baseline": "#8C8C8C",
        "selected_deployable": "#174A75",
        "rejected_complex": "#D7A0A0",
        "comparison": "#718697",
    }
    fig, axis = plt.subplots(figsize=(183 / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.48, right=0.98, top=0.84, bottom=0.09)
    fig.text(
        0.03,
        0.975,
        title,
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    method_y = {}
    header_y = {}
    separators = []
    cursor = 0.2
    for group_id, values in source.groupby("group_id", sort=False):
        header_y[group_id] = cursor
        cursor += 1.15
        for row in values.itertuples(index=False):
            method_y[(group_id, row.method_id)] = cursor
            cursor += 1.08
        separators.append(cursor - 0.25)
        cursor += 0.65

    for row in source.itertuples(index=False):
        y = method_y[(row.group_id, row.method_id)]
        color = colors[row.method_role]
        if row.method_role == "selected_deployable":
            axis.axhspan(y - 0.44, y + 0.44, color="#EDF4F8", zorder=0)
        axis.plot(
            [row.event_mse_kwh2, row.macro_mse_kwh2],
            [y - 0.08, y + 0.08],
            color=color,
            lw=0.8,
            zorder=2,
        )
        axis.scatter(
            row.event_mse_kwh2,
            y - 0.08,
            s=23,
            marker="o",
            facecolor=color,
            edgecolor=color,
            linewidth=0.7,
            zorder=3,
        )
        axis.scatter(
            row.macro_mse_kwh2,
            y + 0.08,
            s=29,
            marker="D",
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            zorder=4,
        )
    for group_id, values in source.groupby("group_id", sort=False):
        first = values.iloc[0]
        y = header_y[group_id]
        axis.text(
            -0.90,
            y,
            f"{group_id.lower()}  {first['group_title']}",
            transform=axis.get_yaxis_transform(),
            clip_on=False,
            fontsize=6.9,
            fontweight="bold",
            va="center",
        )
        axis.text(
            -0.90,
            y + 0.38,
            (
                f"Target: {first['target']}  ·  "
                f"n = {first['event_count']} events / {first['experiment_count']} experiments"
            ),
            transform=axis.get_yaxis_transform(),
            clip_on=False,
            color="#4D4D4D",
            fontsize=5.9,
            va="center",
        )

    for separator in separators[:-1]:
        axis.axhline(separator, color="#D9DEE2", lw=0.6, zorder=0)
    maximum = source[["event_mse_kwh2", "macro_mse_kwh2"]].max().max()
    axis.set_xlim(0, float(maximum) * 1.08)
    axis.set_ylim(cursor - 0.2, -0.55)
    axis.set_xlabel("Held-out MSE (kWh²; lower is better)")
    axis.ticklabel_format(
        axis="x", style="sci", scilimits=(0, 0), useMathText=True
    )
    axis.grid(axis="x", color="#E6EAED", lw=0.5)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", length=0, pad=5)
    axis.set_yticks(list(method_y.values()))
    axis.set_yticklabels(source["method_label"], fontsize=6.2)
    axis.spines["left"].set_visible(False)
    for tick, role in zip(axis.get_yticklabels(), source["method_role"], strict=True):
        tick.set_linespacing(1.05)
        if role == "selected_deployable":
            tick.set_color(colors[role])
            tick.set_fontweight("bold")
        elif role == "rejected_complex":
            tick.set_color("#8D4F4F")
    legend_color = colors["comparison"]
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=4.5,
                markerfacecolor=legend_color,
                markeredgecolor=legend_color,
                label="All-event MSE",
            ),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                markersize=4.8,
                markerfacecolor="white",
                markeredgecolor=legend_color,
                label="Experiment-balanced MSE",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.07),
        ncol=2,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    _save_figure(fig, output, bbox_inches=None, tiff=True)


def write_defrost_energy_method_comparison(output: Path) -> pd.DataFrame:
    source = build_defrost_energy_method_source(
        pd.read_csv(output / "defrost_power_metrics.csv"),
        pd.read_csv(output / "duration_t3_energy_metrics.csv"),
        pd.read_csv(output / "preparation_inclusive_literature_summary.csv"),
        pd.read_csv(output / "preparation_inclusive_network_summary.csv"),
    )
    source.to_csv(output / "defrost_energy_method_comparison_source.csv", index=False)
    plot_defrost_energy_method_comparison(
        source,
        output.parent / "图表" / "figure_defrost_energy_method_comparison",
    )
    return source


def build_pe_linear_cycle_fit_source(
    events: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    """Build cycle-level Pe-quadratic Ridge LOEO data and raw-unit coefficients."""
    if "status" in events:
        events = events.loc[events["status"].eq("included")]
    source = events[
        [
            "cycle_name",
            "experiment_id",
            "evaporating_pressure",
            "inclusive_energy_kwh",
        ]
    ].rename(
        columns={
            "evaporating_pressure": "evaporating_pressure_mpa",
            "inclusive_energy_kwh": "actual_energy_kwh",
        }
    )
    pe_predictions = predictions.loc[
        predictions["model"].eq("pe_quadratic_ridge"),
        ["cycle_name", "predicted_energy_kwh"],
    ].rename(columns={"predicted_energy_kwh": "loeo_predicted_energy_kwh"})
    source = source.merge(
        pe_predictions, on="cycle_name", how="inner", validate="one_to_one"
    )
    if len(source) != len(events):
        raise ValueError(
            "Pe quadratic cycle fit requires one LOEO prediction per included event"
        )
    for experiment in source["experiment_id"].unique():
        train = source.loc[source["experiment_id"].ne(experiment)]
        design = np.column_stack(
            [
                train["evaporating_pressure_mpa"],
                train["evaporating_pressure_mpa"].pow(2),
            ]
        )
        scaler = StandardScaler().fit(design)
        model = Ridge(alpha=PREPARATION_NETWORK_RIDGE_ALPHA).fit(
            scaler.transform(design),
            train["actual_energy_kwh"],
        )
        coefficients = model.coef_ / scaler.scale_
        intercept = model.intercept_ - np.dot(coefficients, scaler.mean_)
        held_out = source["experiment_id"].eq(experiment)
        source.loc[held_out, "fold_intercept_kwh"] = intercept
        source.loc[held_out, "fold_linear_kwh_per_mpa"] = coefficients[0]
        source.loc[held_out, "fold_quadratic_kwh_per_mpa2"] = coefficients[1]
        source.loc[held_out, "fold_train_pe_min_mpa"] = train[
            "evaporating_pressure_mpa"
        ].min()
        source.loc[held_out, "fold_train_pe_max_mpa"] = train[
            "evaporating_pressure_mpa"
        ].max()
        source.loc[held_out, "fold_train_event_count"] = len(train)
        source.loc[held_out, "fold_train_experiment_count"] = train[
            "experiment_id"
        ].nunique()
    source["loeo_residual_kwh"] = (
        source["loeo_predicted_energy_kwh"] - source["actual_energy_kwh"]
    )
    source["absolute_loeo_residual_kwh"] = source["loeo_residual_kwh"].abs()
    source["label_largest_residual"] = False
    source.loc[
        source["absolute_loeo_residual_kwh"].nlargest(5).index,
        "label_largest_residual",
    ] = True
    return source


def plot_pe_linear_cycle_fit(
    source: pd.DataFrame,
    output: Path,
    *,
    parity: bool = False,
    prediction_label: str = "Pe-quadratic Ridge LOEO prediction",
    title: str | None = None,
) -> None:
    """Plot paired observed and LOEO predictions against Pe or an identity line."""
    observed_color = "#777777"
    selected_color = "#174A75"
    fold_color = "#8CB4CD"
    fig, axis = plt.subplots(figsize=(183 / 25.4, 118 / 25.4))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.81, bottom=0.17)
    x_column = "actual_energy_kwh" if parity else "evaporating_pressure_mpa"
    if parity:
        limits = np.array(
            [
                source[["actual_energy_kwh", "loeo_predicted_energy_kwh"]]
                .min()
                .min(),
                source[["actual_energy_kwh", "loeo_predicted_energy_kwh"]]
                .max()
                .max(),
            ]
        )
        pad = np.ptp(limits) * 0.06
        limits += np.array([-pad, pad])
        axis.plot(limits, limits, color=selected_color, lw=1.2, zorder=1)
        axis.set(xlim=limits, ylim=limits)
    else:
        for _, fold in source.groupby("experiment_id", sort=True):
            row = fold.iloc[0]
            x = np.linspace(
                row["fold_train_pe_min_mpa"], row["fold_train_pe_max_mpa"], 100
            )
            axis.plot(
                x,
                row["fold_intercept_kwh"]
                + row["fold_linear_kwh_per_mpa"] * x
                + row["fold_quadratic_kwh_per_mpa2"] * x**2,
                color=fold_color,
                lw=0.8,
                alpha=0.30,
                zorder=1,
            )
        full_design = np.column_stack(
            [
                source["evaporating_pressure_mpa"],
                source["evaporating_pressure_mpa"].pow(2),
            ]
        )
        full_scaler = StandardScaler().fit(full_design)
        full_model = Ridge(alpha=PREPARATION_NETWORK_RIDGE_ALPHA).fit(
            full_scaler.transform(full_design), source["actual_energy_kwh"]
        )
        full_coefficients = full_model.coef_ / full_scaler.scale_
        full_intercept = full_model.intercept_ - np.dot(
            full_coefficients, full_scaler.mean_
        )
        full_x = np.linspace(
            source["evaporating_pressure_mpa"].min(),
            source["evaporating_pressure_mpa"].max(),
            200,
        )
        axis.plot(
            full_x,
            full_intercept
            + full_coefficients[0] * full_x
            + full_coefficients[1] * full_x**2,
            color=selected_color,
            lw=1.6,
            zorder=2,
        )
    axis.vlines(
        source[x_column],
        source["actual_energy_kwh"],
        source["loeo_predicted_energy_kwh"],
        color="#AAB0B5",
        lw=0.55,
        alpha=0.75,
        zorder=2,
    )
    axis.scatter(
        source[x_column],
        source["actual_energy_kwh"],
        s=20,
        marker="o",
        facecolor=observed_color,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    axis.scatter(
        source[x_column],
        source["loeo_predicted_energy_kwh"],
        s=28,
        marker="D",
        facecolor="white",
        edgecolor=selected_color,
        linewidth=0.9,
        zorder=4,
    )
    labeled = source.loc[source["label_largest_residual"]].sort_values(
        "actual_energy_kwh"
    )
    spacing = max(np.ptp(source["actual_energy_kwh"]) * 0.025, 0.0024)
    label_x = labeled[x_column].max() + max(np.ptp(source[x_column]) * 0.03, 0.012)
    previous_label_y = -np.inf
    for _, row in labeled.iterrows():
        label_y = max(row["actual_energy_kwh"], previous_label_y + spacing)
        previous_label_y = label_y
        axis.annotate(
            f"Cycle {int(str(row['cycle_name']).rsplit('_', maxsplit=1)[-1])}",
            (row[x_column], row["actual_energy_kwh"]),
            xytext=(label_x, label_y),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=5.4,
            color="#404040",
            arrowprops={"arrowstyle": "-", "color": "#808080", "lw": 0.45},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.2},
            zorder=5,
        )
    event_rmse = float(np.sqrt(source["loeo_residual_kwh"].pow(2).mean()))
    macro_rmse = float(
        np.sqrt(
            source["loeo_residual_kwh"]
            .pow(2)
            .groupby(source["experiment_id"])
            .mean()
            .mean()
        )
    )
    axis.text(
        0.02 if parity else 0.98,
        0.96,
        (
            f"n = {len(source)} events / {source['experiment_id'].nunique()} experiments\n"
            f"LOEO event RMSE = {event_rmse:.5f} kWh\n"
            f"LOEO macro RMSE = {macro_rmse:.5f} kWh"
        ),
        transform=axis.transAxes,
        ha="left" if parity else "right",
        va="top",
        fontsize=6.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.8},
        zorder=6,
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.3,
            markerfacecolor=observed_color,
            markeredgecolor=observed_color,
            label="Observed cycle",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markersize=4.7,
            markerfacecolor="white",
            markeredgecolor=selected_color,
            label=prediction_label,
        ),
    ]
    if parity:
        handles.append(Line2D([], [], color=selected_color, lw=1.2, label="Identity line"))
    else:
        handles.extend(
            [
                Line2D(
                    [],
                    [],
                    color=fold_color,
                    lw=1.0,
                    alpha=0.45,
                    label=f"{source['experiment_id'].nunique()} LOEO fold fits",
                ),
                Line2D(
                    [],
                    [],
                    color=selected_color,
                    lw=1.6,
                    label="Full-cohort quadratic Ridge (descriptive only)",
                ),
            ]
        )
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=len(handles),
        columnspacing=1.2,
        handletextpad=0.5,
    )
    if title:
        fig.text(0.02, 0.98, title, fontsize=9, fontweight="bold", va="top")
    axis.set_xlabel(
        "Observed active-defrost absorbed heat [kWh]"
        if parity
        else "Median Pe in 60 s before preparation [MPa]"
    )
    axis.set_ylabel(
        "Predicted active-defrost absorbed heat [kWh]"
        if parity
        else "Preparation-inclusive energy [kWh]"
    )
    axis.grid(color="#E5E9EC", lw=0.5)
    axis.set_axisbelow(True)
    axis.margins(x=0.04, y=0.10)
    _save_figure(fig, output, bbox_inches=None, tiff=True)


def write_pe_linear_cycle_fit(output: Path) -> pd.DataFrame:
    source = build_pe_linear_cycle_fit_source(
        pd.read_csv(output / "preparation_inclusive_network_events.csv"),
        pd.read_csv(output / "preparation_inclusive_network_predictions.csv"),
    )
    source.to_csv(output / "pe_linear_cycle_fit_source.csv", index=False)
    source.groupby("experiment_id", sort=True).first().reset_index()[
        [
            "experiment_id",
            "fold_intercept_kwh",
            "fold_linear_kwh_per_mpa",
            "fold_quadratic_kwh_per_mpa2",
            "fold_train_pe_min_mpa",
            "fold_train_pe_max_mpa",
            "fold_train_event_count",
            "fold_train_experiment_count",
        ]
    ].to_csv(output / "pe_quadratic_ridge_fold_coefficients.csv", index=False)
    plot_pe_linear_cycle_fit(
        source,
        output.parent / "图表" / "figure_pe_linear_cycle_fit",
    )
    return source


def write_defrost_absorbed_heat_figures(
    events: pd.DataFrame, metrics: pd.DataFrame, output: Path
) -> None:
    """Reuse the established method-comparison and cycle-fit figure interfaces."""
    feature_metadata = [
        ("water_in_temperature", "T_in", "kWh/°C"),
        ("water_out_temperature", "T_out", "kWh/°C"),
        ("rule_defrost_duration_minutes", "D_rule", "kWh/min"),
        ("coil_temperature", "T3", "kWh/°C"),
        ("evaporating_pressure", "Pe", "kWh/MPa"),
    ]
    feature_columns = [column for column, _, _ in feature_metadata]
    squared_metadata = [
        (f"{column}_squared", f"{symbol}^2", f"{unit}²")
        for column, symbol, unit in feature_metadata
    ]
    model_events = events[feature_columns].copy()
    for column in feature_columns:
        model_events[f"{column}_squared"] = model_events[column].pow(2)
    model_metadata = [*feature_metadata, *squared_metadata]
    model_columns = [column for column, _, _ in model_metadata]
    full_model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)
    ).fit(model_events[model_columns], events["defrost_absorbed_heat_kwh"])
    scaler = full_model.named_steps["standardscaler"]
    ridge = full_model.named_steps["ridge"]
    raw_coefficients = ridge.coef_ / scaler.scale_
    raw_intercept = ridge.intercept_ - np.dot(raw_coefficients, scaler.mean_)
    pd.DataFrame(
        [
            {
                "term": "intercept",
                "symbol": "b0",
                "coefficient": raw_intercept,
                "coefficient_unit": "kWh",
                "training_min": np.nan,
                "training_max": np.nan,
            },
            *(
                {
                    "term": column,
                    "symbol": symbol,
                    "coefficient": coefficient,
                    "coefficient_unit": unit,
                    "training_min": model_events[column].min(),
                    "training_max": model_events[column].max(),
                }
                for (column, symbol, unit), coefficient in zip(
                    model_metadata, raw_coefficients, strict=True
                )
            ),
        ]
    ).to_csv(output / "defrost_absorbed_heat_model_coefficients.csv", index=False)

    models = [
        ("mean", "Training-fold mean", "baseline"),
        ("water", "Inlet + outlet water temperature", "comparison"),
        ("water_duration", "Water temperatures + rule duration", "comparison"),
        ("water_duration_t3", "Water temperatures + rule duration + T3", "comparison"),
        ("water_duration_pe", "Water temperatures + rule duration + Pe", "comparison"),
        (
            "water_duration_t3_pe",
            "Water temperatures + rule duration + T3 + Pe",
            "comparison",
        ),
        (
            "water_duration_t3_pe_squared",
            "Five inputs + individual squared terms",
            "selected_deployable",
        ),
    ]
    selected = metrics.loc[metrics["outcome"].eq("defrost_absorbed_heat")]
    overall = selected.loc[selected["experiment_id"].eq("__overall__")].set_index(
        "strategy"
    )
    per_experiment = selected.loc[~selected["experiment_id"].eq("__overall__")]
    baseline_mse = float(overall.loc["mean", "mse"])
    baseline_macro = float(
        per_experiment.loc[per_experiment["strategy"].eq("mean"), "mse"].mean()
    )
    rows = []
    for method, label, role in models:
        values = overall.loc[method]
        macro_mse = float(
            per_experiment.loc[per_experiment["strategy"].eq(method), "mse"].mean()
        )
        rows.append(
            {
                "group_id": "A",
                "group_title": "Which inputs predict active-defrost absorbed heat?",
                "target": "active-defrost absorbed heat",
                "event_count": int(values["event_count"]),
                "experiment_count": events["experiment_id"].nunique(),
                "method_id": method,
                "method_label": label,
                "method_role": role,
                "event_mse_kwh2": float(values["mse"]),
                "macro_mse_kwh2": macro_mse,
                "event_relative_mse_pct": 100 * float(values["mse"]) / baseline_mse,
                "macro_relative_mse_pct": 100 * macro_mse / baseline_macro,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        output / "defrost_absorbed_heat_method_comparison_source.csv", index=False
    )
    plot_defrost_energy_method_comparison(
        comparison,
        output.parent / "图表" / "figure_defrost_absorbed_heat_method_comparison",
        title="Individual squared terms improve held-out absorbed-heat prediction",
        height_mm=100,
    )

    fit = events[
        [
            "cycle_name",
            "experiment_id",
            "water_in_temperature",
            "water_out_temperature",
            "rule_defrost_duration_minutes",
            "coil_temperature",
            "evaporating_pressure",
            "defrost_absorbed_heat_kwh",
            "predicted_water_duration_t3_pe_squared_defrost_absorbed_heat",
        ]
    ].rename(
        columns={
            "defrost_absorbed_heat_kwh": "actual_energy_kwh",
            "predicted_water_duration_t3_pe_squared_defrost_absorbed_heat": (
                "loeo_predicted_energy_kwh"
            ),
        }
    )
    fit["loeo_residual_kwh"] = (
        fit["loeo_predicted_energy_kwh"] - fit["actual_energy_kwh"]
    )
    fit["absolute_loeo_residual_kwh"] = fit["loeo_residual_kwh"].abs()
    fit["label_largest_residual"] = False
    fit.loc[
        fit["absolute_loeo_residual_kwh"].nlargest(5).index,
        "label_largest_residual",
    ] = True
    fit.to_csv(output / "defrost_absorbed_heat_cycle_fit_source.csv", index=False)
    plot_pe_linear_cycle_fit(
        fit,
        output.parent / "图表" / "figure_defrost_absorbed_heat_cycle_fit",
        parity=True,
        prediction_label="Five inputs + individual squared terms",
        title="The selected model reproduces active-defrost absorbed heat",
    )


def _save_figure(
    fig: plt.Figure,
    base: Path,
    *,
    bbox_inches: str | None = "tight",
    tiff: bool = False,
) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches=bbox_inches, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches=bbox_inches, facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches=bbox_inches, facecolor="white")
    if tiff:
        fig.savefig(
            base.with_suffix(".tiff"),
            dpi=600,
            bbox_inches=bbox_inches,
            facecolor="white",
        )
    plt.close(fig)


def plot_recovery_start_states(events: pd.DataFrame, output: Path) -> None:
    """Show the cross-event concentration of recovery-start thermal states."""
    labels = {
        "water_temperature_setpoint": r"Water setpoint $T_s$",
        "water_in_temperature": r"Water inlet $T_{w,in}$",
        "water_out_temperature": r"Water outlet $T_{w,out}$",
        "water_flow": "Water flow",
        "coil_temperature": r"Coil $T_3$",
        "evaporating_temperature": r"Evaporating $T_e$",
        "suction_temperature": r"Suction $T_h$",
        "discharge_temperature": r"Discharge $T_p$",
        "evaporating_pressure": r"Evaporating $P_e$",
        "condensing_pressure": r"Condensing $P_c$",
        "ambient_temperature": r"Ambient $T_4$",
        "environment_relative_humidity": "Ambient RH",
        "compressor_frequency": "Compressor frequency",
        "compressor_frequency_setpoint": "Frequency command",
        "compressor_power": "Compressor power",
        "pi_step": "PI step",
        "pi_power": "PI output",
        "pi_step_limit": "PI step limit",
        "compressor_limit_code": "Compressor limit code",
        "compressor_frequency_state": "Frequency-control state",
    }
    ordered = events.sort_values(
        ["recovery_start_water_temperature_setpoint", "experiment_id", "cycle_name"],
        kind="stable",
    )
    columns = [f"recovery_start_{feature}" for feature in RECOVERY_STATE_FEATURES]
    values = ordered[columns].apply(pd.to_numeric, errors="coerce")
    center = values.median()
    scale = values.quantile(0.75) - values.quantile(0.25)
    scale = scale.where(scale.gt(0), values.std()).replace(0, 1)
    normalized = ((values - center) / scale).clip(-2.5, 2.5).T
    fig, axis = plt.subplots(figsize=(183 / 25.4, 88 / 25.4))
    image = axis.imshow(normalized, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    boundaries = (
        ordered["experiment_id"].ne(ordered["experiment_id"].shift()).to_numpy().nonzero()[0]
    )
    axis.vlines(boundaries[1:] - 0.5, -0.5, len(columns) - 0.5, color="white", lw=0.35)
    axis.set(
        xlabel=f"Consecutive recovery events (n = {len(events)}; ordered by $T_s$ and experiment)",
        ylabel="State during first 30 s after defrost",
        xticks=[],
        yticks=np.arange(len(columns)),
        yticklabels=[labels[feature] for feature in RECOVERY_STATE_FEATURES],
    )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.02, pad=0.02)
    colorbar.set_label("Deviation from event median (IQR)")
    fig.tight_layout()
    _save_figure(fig, output)


def plot_recovery_outcome_by_cycle(
    events: pd.DataFrame,
    outcome: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    """Plot each complete post-defrost recovery with experiment-date bands."""
    source = events.loc[
        events["recovery_valid"], ["cycle_name", "experiment_id", outcome]
    ].copy()
    source["cycle_id"] = pd.to_numeric(
        source["cycle_name"].str.extract(r"(\d+)$", expand=False), errors="coerce"
    )
    source[outcome] = pd.to_numeric(source[outcome], errors="coerce")
    source = source.dropna(subset=["cycle_id", outcome]).sort_values("cycle_id")
    source["cycle_id"] = source["cycle_id"].astype(int)

    groups = list(source.groupby("experiment_id", sort=False))
    starts = [float(group["cycle_id"].min()) for _, group in groups]
    ends = [float(group["cycle_id"].max()) for _, group in groups]
    boundaries = [starts[0] - 0.75]
    boundaries.extend(
        (ends[index] + starts[index + 1]) / 2 for index in range(len(groups) - 1)
    )
    boundaries.append(ends[-1] + 0.75)

    fig, axis = plt.subplots(figsize=(183 / 25.4, 82 / 25.4))
    band_colors = ("#F3F5F8", "#EAF3F3")
    for index, ((experiment, group), left, right) in enumerate(
        zip(groups, boundaries[:-1], boundaries[1:], strict=True)
    ):
        axis.axvspan(left, right, color=band_colors[index % 2], zorder=0)
        axis.axvline(left, color="#A8A8A8", lw=0.45, zorder=1)
        axis.plot(
            group["cycle_id"],
            group[outcome],
            color="#7884B4",
            lw=0.7,
            alpha=0.75,
            zorder=2,
        )
        axis.hlines(
            group[outcome].mean(),
            group["cycle_id"].min() - 0.25,
            group["cycle_id"].max() + 0.25,
            color="#E28E2C",
            lw=1.4,
            zorder=3,
        )
        date = pd.to_datetime(str(experiment).removeprefix("exp_"), format="%Y%m%d")
        axis.text(
            (left + right) / 2,
            1.015,
            date.strftime("%m-%d"),
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=5.5,
            color="#606060",
        )
    axis.axvline(boundaries[-1], color="#A8A8A8", lw=0.45, zorder=1)
    axis.scatter(
        source["cycle_id"],
        source[outcome],
        s=17,
        color="#0F4D92",
        edgecolor="white",
        linewidth=0.35,
        zorder=4,
    )
    tick_start = 5 * int(np.floor(source["cycle_id"].min() / 5))
    tick_end = 5 * int(np.ceil(source["cycle_id"].max() / 5))
    axis.set(
        xlim=(boundaries[0], boundaries[-1]),
        xticks=np.arange(tick_start, tick_end + 1, 5),
        xlabel="Defrost cycle ID (recovery measured in the following cycle)",
        ylabel=ylabel,
    )
    axis.set_title(title, loc="left", y=1.10, pad=0)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#D8D8D8", lw=0.45, alpha=0.8)
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                color="#7884B4",
                markerfacecolor="#0F4D92",
                lw=0.7,
                label=f"Measured recovery (n = {len(source)})",
            ),
            Line2D(
                [], [], color="#E28E2C", lw=1.4, label="Experiment-date mean"
            ),
        ],
        loc="upper left",
        fontsize=5.5,
    )
    fig.tight_layout()
    _save_figure(fig, output)


def plot_recovery_predictors(metrics: pd.DataFrame, output: Path) -> None:
    """Compare physical pre-defrost models against the Ts-only baseline."""
    source = metrics.loc[~metrics["model"].eq("ts_only")].copy()
    table = source.pivot(
        index="model", columns="outcome", values="improvement_vs_ts_pct"
    ).reindex(columns=RECOVERY_OUTCOMES)
    table = table.loc[table.max(axis=1).sort_values(ascending=False).index]
    display = [
        {
            "thermal_deficit": "water-side thermal deficit",
            "capacity_margin": "source and capacity margin",
            "controller_state": "PI and compressor controller state",
            "physical_prestate": "combined physical pre-state",
        }.get(
            name,
            name.removeprefix("pre_")
            .replace("_slope_per_min", " slope")
            .replace("_", " "),
        )
        for name in table.index
    ]
    limit = max(10.0, float(np.nanmax(np.abs(table.to_numpy()))))
    fig, axis = plt.subplots(figsize=(183 / 25.4, max(80, 7 * len(table)) / 25.4))
    image = axis.imshow(table, aspect="auto", cmap="RdBu", vmin=-limit, vmax=limit)
    for row in range(len(table)):
        for column in range(len(table.columns)):
            value = table.iat[row, column]
            axis.text(column, row, f"{value:+.0f}%", ha="center", va="center", fontsize=5.5)
    axis.set(
        xlabel=r"LOEO event-RMSE improvement over $T_s$-only",
        xticks=np.arange(3),
        xticklabels=["Electricity", "Water-side heat", "Duration"],
        yticks=np.arange(len(table)),
        yticklabels=display,
    )
    axis.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("RMSE improvement (%)")
    fig.tight_layout()
    _save_figure(fig, output)


def _remove_stale_cycle_figures(
    output: Path, expected_cycles: set[str], *, filename_suffix: str = ""
) -> list[Path]:
    """Remove only obsolete generated cycle image formats from one exact directory."""
    removed = []
    for suffix in (".svg", ".pdf", ".png"):
        for path in output.glob(f"frost_cycle_*{suffix}"):
            if filename_suffix and not path.stem.endswith(filename_suffix):
                continue
            if not filename_suffix and path.stem.endswith("_J_unit"):
                continue
            if path.stem not in expected_cycles:
                path.unlink()
                removed.append(path)
    return removed


def plot_inverse_cop_curves(
    loader: DatasetLoader, curves: pd.DataFrame, output: Path
) -> list[str]:
    """Export one standalone inverse-COP candidate curve per complete cycle."""
    cycle_table = loader.list_cycles()
    cycles = complete_observed_cycle_names(cycle_table, curves)
    stable_starts = cycle_table.set_index("cycle_name")["stable_heating_start"]
    exported = []
    for cycle_name in cycles:
        values = curves.loc[curves["cycle_name"].eq(cycle_name)].copy()
        values["candidate_time"] = pd.to_datetime(values["candidate_time"], errors="coerce")
        values["inverse_cop"] = pd.to_numeric(values["inverse_cop"], errors="coerce")
        values = values.dropna(subset=["candidate_time", "inverse_cop"]).sort_values(
            "candidate_time", kind="stable"
        )
        eligible = values.get(
            "optimization_eligible", pd.Series(True, index=values.index)
        ).fillna(False)
        if not eligible.any():
            continue
        stable_start = pd.to_datetime(
            stable_starts.loc[cycle_name], errors="coerce", format="mixed"
        )
        values["minutes"] = (
            values["candidate_time"] - stable_start
        ).dt.total_seconds() / 60
        eligible_values = values.loc[eligible.astype(bool)]
        minimum = eligible_values.loc[eligible_values["inverse_cop"].idxmin()]
        fig, axis = plt.subplots(figsize=(3.5, 2.4))
        axis.plot(values["minutes"], values["inverse_cop"], color="#3775BA", lw=1.3)
        near = eligible_values.loc[
            pd.to_numeric(eligible_values["relative_regret"], errors="coerce").le(0.01)
        ]
        axis.scatter(
            near["minutes"],
            near["inverse_cop"],
            s=12,
            color="#E28E2C",
            label="1% near-optimal",
        )
        axis.scatter(
            [minimum["minutes"]],
            [minimum["inverse_cop"]],
            s=24,
            marker="*",
            color="#B64342",
            zorder=3,
            label="Minimum",
        )
        axis.set(
            xlabel="Time from stable heating start [min]",
            ylabel="Cycle inverse COP [-]",
            title=str(cycle_name),
        )
        axis.legend(fontsize=6)
        _save_figure(fig, output / str(cycle_name))
        exported.append(cycle_name)
    _remove_stale_cycle_figures(output, set(exported))
    return exported


def plot_ticket_audit(
    events: pd.DataFrame,
    metrics: pd.DataFrame,
    shifts: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), gridspec_kw={"hspace": 0.42, "wspace": 0.34})
    valid = events.sort_values("equivalent_cost_kwh").reset_index(drop=True)
    thermal_equivalent = valid["equivalent_cost_kwh"] - valid["electricity_kwh"]
    axes[0, 0].bar(range(len(valid)), valid["electricity_kwh"], color="#72B7B2", width=0.9)
    axes[0, 0].bar(
        range(len(valid)),
        thermal_equivalent,
        bottom=valid["electricity_kwh"],
        color="#A6C8E0",
        width=0.9,
    )
    axes[0, 0].axhline(valid["equivalent_cost_kwh"].mean(), color="#D95F02", lw=1)
    axes[0, 0].set(xlabel="Defrost/recovery event (sorted)", ylabel="Ticket cost (kWh-eq.)")
    axes[0, 0].legend(
        handles=[
            Patch(color="#72B7B2", label="Electricity"),
            Patch(color="#A6C8E0", label="Thermal shortfall equivalent"),
            Line2D([], [], color="#D95F02", lw=1, label="Mean"),
        ],
        fontsize=5.5,
    )

    axes[0, 1].scatter(
        valid["minutes_from_stable"], valid["equivalent_cost_kwh"], s=15, color="#4C78A8", alpha=0.8
    )
    rho = valid[["minutes_from_stable", "equivalent_cost_kwh"]].corr(method="spearman").iloc[0, 1]
    axes[0, 1].text(0.04, 0.94, f"Spearman ρ = {rho:.2f}", transform=axes[0, 1].transAxes, va="top")
    axes[0, 1].set(xlabel="Observed defrost time (min)", ylabel="Ticket cost (kWh-eq.)")

    labels = {"mean": "Mean", "time": "Time linear", "state": "State ridge"}
    positions = np.arange(3)
    for offset, outcome in ((-0.13, "cost"), (0.13, "duration")):
        raw = [
            metrics.loc[
                metrics["strategy"].eq(strategy) & metrics["outcome"].eq(outcome), "mae"
            ].mean()
            for strategy in labels
        ]
        values = np.asarray(raw) / raw[0]
        axes[1, 0].bar(
            positions + offset,
            values,
            width=0.24,
            color="#4C78A8" if outcome == "cost" else "#F2A65A",
            label="Ticket cost" if outcome == "cost" else "Duration",
        )
    axes[1, 0].axhline(1, color="#555555", ls="--", lw=0.8)
    axes[1, 0].set(
        xticks=positions,
        xticklabels=list(labels.values()),
        ylabel="Held-out MAE / mean-baseline MAE",
    )
    axes[1, 0].legend(fontsize=5.5)

    shift_data = [
        shifts["median_ticket_shift_minutes"].abs().dropna(),
        shifts["conditional_shift_minutes"].abs().dropna(),
    ]
    axes[1, 1].boxplot(
        shift_data,
        tick_labels=["Mean → median", "Mean → state ridge"],
        patch_artist=True,
        boxprops={"facecolor": "#C6DBEF", "edgecolor": "#4C78A8"},
        medianprops={"color": "#D95F02"},
        whiskerprops={"color": "#4C78A8"},
        capprops={"color": "#4C78A8"},
        flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "#777777"},
    )
    rng = np.random.default_rng(0)
    for index, values in enumerate(shift_data, start=1):
        axes[1, 1].scatter(
            index + rng.uniform(-0.07, 0.07, len(values)),
            values,
            s=7,
            color="#4C78A8",
            alpha=0.35,
        )
    axes[1, 1].set(ylabel="Absolute optimum shift (min)")

    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontsize=9, fontweight="bold")
    _save_figure(fig, output)


def plot_window_cop_rgb(overview: pd.DataFrame, output: Path) -> None:
    colors = overview["minimum_location"].map(
        {
            "interior": "#4C78A8",
            "left_boundary": "#E6A34A",
            "right_boundary": "#8C8C8C",
            "right_observed": "#8C8C8C",
            "right_support_limited": "#8C8C8C",
            "right_integration_limited": "#8C8C8C",
        }
    ).to_numpy()
    x = np.arange(len(overview))
    fig = plt.figure(figsize=(7.2, 8.9))
    grid = fig.add_gridspec(
        4, 2, height_ratios=[1.15, 1.0, 0.18, 5.0], hspace=0.28, wspace=0.28
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, :])
    ax_a.vlines(
        x,
        overview["near_opt_start_minutes"],
        overview["near_opt_end_minutes"],
        color=colors,
        alpha=0.55,
        lw=1.2,
    )
    ax_a.scatter(x, overview["minutes_from_stable"], color=colors, s=10, zorder=3)
    ax_a.set(ylabel="Time from stable heating (min)", xticks=[])
    ax_a.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color="#4C78A8", label="Interior minimum"),
            Line2D([], [], marker="o", ls="", color="#8C8C8C", label="Right boundary"),
        ],
        loc="upper left",
        ncol=2,
        fontsize=5.5,
    )
    ax_b.scatter(x, overview["cop_at_t_star_60s"], color=colors, s=12)
    ax_b.axhline(overview["cop_at_t_star_60s"].median(), color="#D95F02", lw=0.9)
    ax_b.set(ylabel="COP in preceding 60 s", xticks=[])
    ax_a.text(-0.04, 1.05, "a", transform=ax_a.transAxes, fontsize=9, fontweight="bold")
    ax_b.text(-0.04, 1.05, "b", transform=ax_b.transAxes, fontsize=9, fontweight="bold")

    ax_c = fig.add_subplot(grid[2, :])
    ax_c.axis("off")
    ax_c.text(-0.04, 0.5, "c", transform=ax_c.transAxes, fontsize=9, fontweight="bold")
    ax_c.text(
        0,
        0.5,
        "Front-view RGB available for "
        f"{int(overview['front_image_available'].sum())}/{len(overview)} cycles; "
        "grey tiles await cloud retrieval",
        va="center",
        fontsize=5.5,
    )

    image_grid = grid[3, :].subgridspec(9, 7, hspace=0.08, wspace=0.04)
    for index in range(63):
        axis = fig.add_subplot(image_grid[index // 7, index % 7])
        axis.set_xticks([])
        axis.set_yticks([])
        if index >= len(overview):
            axis.axis("off")
            continue
        row = overview.iloc[index]
        image_path = row["front_image_path"]
        path = Path(image_path) if isinstance(image_path, str) and image_path else None
        if path and path.is_file():
            with Image.open(path) as image:
                axis.imshow(np.asarray(image.convert("RGB")))
        else:
            axis.set_facecolor("#ECECEC")
            axis.text(
                0.5,
                0.52,
                "cloud\npending",
                ha="center",
                va="center",
                fontsize=4.2,
                color="#666666",
            )
        cycle = int(str(row["cycle_name"]).rsplit("_", 1)[-1])
        axis.text(
            0.02,
            0.98,
            f"C{cycle:03d} · {row['minutes_from_stable']:.0f}m\nCOP {row['cop_at_t_star_60s']:.2f}",
            transform=axis.transAxes,
            va="top",
            fontsize=3.9,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 0.5},
        )
        color = colors[index]
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(0.8)
    fig.text(0.5, 0.01, "Cycles ordered by empirical optimum time", ha="center", fontsize=7)
    _save_figure(fig, output)


def analyze(
    dataset: Path,
    source: Path,
    output: Path,
    *,
    cloud_root: Path | None = None,
) -> None:
    # Stage 1: load the raw-data cost baseline.
    loader = DatasetLoader(dataset)
    catalog = loader.list_cycles()
    tickets = pd.read_csv(source / "defrost_ticket_events.csv")
    points = pd.read_csv(source / "cycle_optimal_points.csv")
    curves = pd.read_parquet(source / "candidate_cost_curves.parquet")
    for column in ("t_heating_stable", "t_star", "near_opt_start", "near_opt_end"):
        points[column] = pd.to_datetime(points[column], errors="coerce")
    curves["candidate_time"] = pd.to_datetime(curves["candidate_time"], errors="coerce")

    # Stage 2: audit observed defrost-ticket cost models.
    event_predictions, metrics = build_ticket_evidence(loader, tickets, points, catalog)
    event_features = event_predictions
    # Stage 3: recover empirical economic windows under each ticket model.
    partial_pool_curves = build_partial_pool_curves(curves, event_features, catalog)
    partial_pool_points = partial_pool_optimal_points(partial_pool_curves)
    candidate_features = build_candidate_features(
        loader, curves, points.loc[points["valid"]], catalog
    )
    conditional_curves = predict_candidate_tickets(
        event_features, candidate_features, STATE_FEATURES
    )
    conditional_curves["renewal_cost_conditional"] = (
        conditional_curves["heating_cost_kwh"] + conditional_curves["predicted_ticket_cost"]
    ) / (
        conditional_curves["heating_hours"]
        + conditional_curves["predicted_ticket_duration"] / 60
    )
    conditional = conditional_optimal_points(conditional_curves)
    shifts = points.loc[
        points["valid"], ["cycle_name", "t_star", "median_ticket_shift_minutes"]
    ].merge(
        conditional, on="cycle_name", how="left"
    )
    shifts["conditional_shift_minutes"] = (
        shifts["t_star_conditional"] - shifts["t_star"]
    ).dt.total_seconds() / 60
    # Stage 4: connect windows to COP and RGB evidence.
    labels = pd.read_parquet("output/label/cost_function_v1_binary/image_cost_labels.parquet")
    overview = build_window_overview(loader, points, labels, dataset)

    # Stage 5: write publication evidence from the same tables.
    output.mkdir(parents=True, exist_ok=True)
    event_predictions.to_csv(output / "ticket_event_features_and_predictions.csv", index=False)
    metrics.to_csv(output / "ticket_model_metrics_by_experiment.csv", index=False)
    write_defrost_absorbed_heat_figures(event_predictions, metrics, output)
    partial_pool_curves.to_parquet(output / "partial_pool_candidate_costs.parquet", index=False)
    partial_pool_points.to_csv(output / "partial_pool_optimal_points.csv", index=False)
    conditional_curves.to_parquet(output / "conditional_candidate_costs.parquet", index=False)
    shifts.to_csv(output / "conditional_optimal_points.csv", index=False)
    overview.to_csv(output / "optimal_window_cop_rgb.csv", index=False)
    plot_ticket_audit(
        event_predictions,
        metrics,
        shifts,
        output.parent / "图表" / "figure_ticket_cost_audit",
    )
    plot_window_cop_rgb(
        overview,
        output.parent / "图表" / "figure_window_cop_rgb_overview",
    )
    publication_curves = merge_rb_points(curves, points)
    render_all_cost_publications(
        loader,
        publication_curves,
        output.parent / "图表" / "循环图" / "全部循环",
        match_output=source / "rb_optimal_rgb_matches.csv",
        fetch_cloud=True,
        cloud_root=cloud_root,
        cleanup_downloaded=True,
        minimum_free_gib=35,
    )
    render_all_cost_publications(
        loader,
        publication_curves,
        output.parent / "图表" / "循环图" / "全部循环",
        match_output=source / "rb_optimal_rgb_matches_unit_heat.csv",
        fetch_cloud=True,
        cloud_root=cloud_root,
        cleanup_downloaded=False,
        minimum_free_gib=35,
        unit_heat=True,
    )
    plot_inverse_cop_curves(
        loader,
        curves,
        output.parent / "图表" / "单位有效供热电耗曲线",
    )
    render_representative_cost_publication(
        loader,
        points,
        publication_curves,
        output.parent / "图表" / "循环图" / "representative_publication_cost.png",
    )


def render_unit_cost_publications(
    dataset: Path, source: Path, *, cloud_root: Path | None = None
) -> list[str]:
    """Regenerate only the per-cycle canonical-unit-heat PNG publications."""
    loader = DatasetLoader(dataset)
    points = pd.read_csv(source / "cycle_optimal_points.csv")
    curves = pd.read_parquet(source / "candidate_cost_curves.parquet")
    return render_all_cost_publications(
        loader,
        merge_rb_points(curves, points),
        source.parent / "图表" / "循环图" / "全部循环",
        match_output=source / "rb_optimal_rgb_matches_unit_heat.csv",
        fetch_cloud=True,
        cloud_root=cloud_root,
        cleanup_downloaded=False,
        minimum_free_gib=35,
        unit_heat=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--source", type=Path, default=Path("output/test/成本函数/其他/经验经济窗口/源数据")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/test/成本函数/其他/经验经济窗口/证据")
    )
    parser.add_argument("--cloud-root", type=Path, default=None)
    only = parser.add_mutually_exclusive_group()
    only.add_argument("--defrost-rule-only", action="store_true")
    only.add_argument("--recovery-only", action="store_true")
    only.add_argument("--defrost-power-only", action="store_true")
    only.add_argument("--duration-t3-energy-only", action="store_true")
    only.add_argument("--predefrost-sensor-only", action="store_true")
    only.add_argument("--preparation-inclusive-sensor-only", action="store_true")
    only.add_argument("--energy-method-comparison-only", action="store_true")
    only.add_argument("--pe-linear-cycle-fit-only", action="store_true")
    only.add_argument("--unit-publications-only", action="store_true")
    args = parser.parse_args()
    if args.recovery_only:
        write_recovery_evidence(args.dataset, args.source, args.output)
    elif args.unit_publications_only:
        render_unit_cost_publications(
            args.dataset, args.source, cloud_root=args.cloud_root
        )
    elif args.pe_linear_cycle_fit_only:
        write_pe_linear_cycle_fit(args.output)
    elif args.energy_method_comparison_only:
        write_defrost_energy_method_comparison(args.output)
    elif args.preparation_inclusive_sensor_only:
        write_preparation_inclusive_sensor_evidence(args.dataset, args.output)
    elif args.predefrost_sensor_only:
        write_predefrost_sensor_increment_evidence(args.output)
    elif args.duration_t3_energy_only:
        write_duration_t3_energy_evidence(args.dataset, args.output)
    elif args.defrost_power_only:
        write_defrost_power_evidence(args.dataset, args.output)
    elif args.defrost_rule_only:
        write_defrost_rule_evidence(args.dataset, args.source, args.output)
    else:
        analyze(args.dataset, args.source, args.output, cloud_root=args.cloud_root)


if __name__ == "__main__":
    main()
