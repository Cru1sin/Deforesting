"""Execute the frozen 59-cycle replication and train the minimal sensor model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frost_analysis.dataset_loader import DatasetLoader
from frost_analysis.degradation_law import fit_hinge
from frost_analysis.sensor_model import (
    ReferenceModel,
    RidgeModel,
    add_cycle_future,
    apply_reference_model,
    fit_reference_model,
    fit_weighted_ridge,
    shared_complete_cases,
    split_replication_cohort,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "sensor_model_59"
FIGURES = OUT / "figures"
SOURCE = OUT / "source_data"

EARLY_MINUTES = 10
OLD_HINGE_THRESHOLD = 0.10168269890562005
OLD_HINGE_SLOPE = 0.019033817438837038
OLD_TIME_SLOPE = 0.00138602

CONTEXT = [
    "ambient_temperature",
    "water_in_temperature",
    "water_flow",
    "compressor_frequency",
    "fan_speed",
    "exv_opening",
]
REFERENCE_COLUMNS = [
    "evaporating_temperature",
    "coil_temperature",
    "suction_temperature",
    "heating_capacity",
    "cop",
    "evaporator_capacity",
    "water_delta_temperature",
]
LOAD_COLUMNS = [
    "timestamp",
    "cycle_stage",
    "ambient_temperature",
    "environment_temperature",
    "environment_relative_humidity",
    "water_in_temperature",
    "water_out_temperature",
    "water_temperature_setpoint",
    "water_flow",
    "evaporating_pressure",
    "evaporating_temperature",
    "coil_temperature",
    "suction_temperature",
    "superheat",
    "compressor_frequency",
    "compressor_frequency_setpoint",
    "compressor_power",
    "fan_speed",
    "fan_current",
    "exv_opening",
    "heating_capacity",
    "power_total",
    "cop",
    "evaporator_capacity",
    "water_delta_temperature",
    "water_delta_temperature__imputed",
]


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    loader = DatasetLoader(ROOT / "dataset")
    catalog = loader.list_cycles()
    old_cycles, new_cycles, stress_cycle = split_replication_cohort(catalog)
    all_cycles = pd.concat([old_cycles, new_cycles], ignore_index=True)
    raw = load_minute_cycles(loader, all_cycles)

    old_raw = raw.loc[raw["cohort"].eq("old")].copy()
    new_raw = raw.loc[raw["cohort"].eq("new")].copy()
    old_crossfit = crossfit_references(old_raw)
    new_frozen = frozen_new_references(old_raw, new_raw)
    replication, prospective_dates = prospective_replication(new_frozen)
    state_comparison = prospective_state_comparison(old_crossfit, new_frozen)
    rate_replication = prospective_rate_replication(old_crossfit, new_frozen)

    current_mapping, dynamics = nested_valid_cycle_analysis(raw)
    repeatability = empirical_repeatability(raw)
    lag_events = water_lag_audit(loader, all_cycles)
    stress_timeline = cycle11_timeline(loader, stress_cycle, old_raw)
    model_payload = train_final_state_models(raw)

    cohort = all_cycles.assign(
        cohort=np.where(all_cycles["cycle_name"].astype(str).le("frost_cycle_000049"), "old", "new")
    )
    eligible = set(model_payload["eligible_cycle_ids"])
    model_eligibility = cohort[["cycle_name", "status", "cohort"]].copy()
    model_eligibility["sensor_model_eligible"] = model_eligibility["cycle_name"].isin(eligible)
    model_eligibility["reason"] = np.where(
        model_eligibility["sensor_model_eligible"],
        "eligible",
        "missing first-10-min state calibration",
    )
    cohort.to_csv(SOURCE / "cohort_59.csv", index=False)
    model_eligibility.to_csv(SOURCE / "model_eligibility.csv", index=False)
    replication.to_csv(SOURCE / "prospective_cycle_errors.csv", index=False)
    prospective_dates.to_csv(SOURCE / "prospective_date_errors.csv", index=False)
    state_comparison.to_csv(SOURCE / "prospective_state_comparison.csv", index=False)
    rate_replication.to_csv(SOURCE / "prospective_rate_replication.csv", index=False)
    current_mapping.to_csv(SOURCE / "current_mapping_cv.csv", index=False)
    dynamics.to_csv(SOURCE / "dynamics_cv.csv", index=False)
    repeatability.to_csv(SOURCE / "empirical_repeatability.csv", index=False)
    lag_events.to_csv(SOURCE / "water_lag_events.csv", index=False)
    stress_timeline.to_csv(SOURCE / "cycle11_timeline.csv", index=False)
    (OUT / "sensor_state_models.json").write_text(
        json.dumps(model_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = build_summary(
        cohort,
        replication,
        prospective_dates,
        state_comparison,
        rate_replication,
        current_mapping,
        dynamics,
        repeatability,
        lag_events,
        stress_timeline,
        model_payload,
        catalog.iloc[59:]["cycle_name"].astype(str).tolist(),
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    configure_plotting()
    figure_replication(new_frozen, replication, prospective_dates)
    figure_proxy_boundary(repeatability, lag_events)
    figure_state_selection(state_comparison, current_mapping)
    figure_cycle11(stress_timeline)
    figure_model_selection(dynamics)


def load_minute_cycles(loader: DatasetLoader, cycles: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for record in cycles.to_dict(orient="records"):
        frame = loader.load_cycle(str(record["cycle_name"]), columns=LOAD_COLUMNS)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        start = pd.to_datetime(record.get("baseline_end"), errors="coerce")
        if pd.isna(start):
            start = pd.to_datetime(record["stable_heating_start"]) + pd.Timedelta(minutes=1)
        end = pd.to_datetime(record.get("defrost_start"), errors="coerce")
        if pd.isna(end):
            end = pd.to_datetime(record["end_time"])
        scoped = frame.loc[frame["timestamp"].between(start, end)].copy()
        stage = scoped.set_index("timestamp")["cycle_stage"].resample("1min").agg(mode_text)
        numeric = (
            scoped.set_index("timestamp").select_dtypes(include="number").resample("1min").median()
        )
        reduced = numeric.join(stage).reset_index()
        cycle = str(record["cycle_name"])
        reduced["cycle"] = cycle
        reduced["date"] = str(record["experiment_date"])[:10]
        reduced["cohort"] = "old" if cycle <= "frost_cycle_000049" else "new"
        reduced["minute"] = (reduced["timestamp"] - start).dt.total_seconds() / 60.0
        reduced["early"] = reduced["minute"].between(0, EARLY_MINUTES)
        rows.append(reduced)
    return (
        pd.concat(rows, ignore_index=True).sort_values(["cycle", "minute"]).reset_index(drop=True)
    )


def mode_text(values: pd.Series) -> str:
    modes = values.dropna().astype(str).mode()
    return "" if modes.empty else str(modes.iloc[0])


def fit_and_apply_references(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, ReferenceModel]]:
    prepared_train = train.copy()
    prepared_test = test.copy()
    models: dict[str, ReferenceModel] = {}
    for column in REFERENCE_COLUMNS:
        model = fit_reference_model(
            prepared_train,
            target=column,
            features=CONTEXT,
            early="early",
        )
        models[column] = model
        prepared_train[f"{column}_normal"] = apply_reference_model(
            model,
            prepared_train,
            observed=column,
            cycle="cycle",
            early="early",
        )
        prepared_test[f"{column}_normal"] = apply_reference_model(
            model,
            prepared_test,
            observed=column,
            cycle="cycle",
            early="early",
        )
    return add_states(prepared_train), add_states(prepared_test), models


def add_states(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["z_r"] = result["evaporating_temperature_normal"] - result["evaporating_temperature"]
    result["z_a"] = result["ambient_temperature"] - result["evaporating_temperature"]
    result["z_coil"] = result["coil_temperature_normal"] - result["coil_temperature"]
    result["z_suction"] = result["suction_temperature_normal"] - result["suction_temperature"]
    result["D_Q"] = 1.0 - result["heating_capacity"] / result["heating_capacity_normal"]
    result["D_COP"] = 1.0 - result["cop"] / result["cop_normal"]
    result["D_Qe_proxy"] = (
        1.0 - result["evaporator_capacity"] / result["evaporator_capacity_normal"]
    )
    result["D_deltaT"] = (
        1.0 - result["water_delta_temperature"] / result["water_delta_temperature_normal"]
    )
    return add_history(result)


def add_history(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["cycle", "minute"]).copy()
    grouped = result.groupby("cycle", sort=False)["z_r"]
    for window in [1, 3, 5]:
        result[f"rate_{window}m"] = (result["z_r"] - grouped.shift(window)) / window
    result["z_mean_5m"] = grouped.transform(lambda values: values.rolling(6, min_periods=4).mean())
    result["z_std_5m"] = grouped.transform(lambda values: values.rolling(6, min_periods=4).std())
    for lag in range(1, 6):
        result[f"z_lag_{lag}m"] = grouped.shift(lag)
    return result


def analysis_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["minute"].gt(EARLY_MINUTES) & frame["compressor_frequency"].gt(5)].copy()


def crossfit_references(frame: pd.DataFrame) -> pd.DataFrame:
    folds = []
    for date in sorted(frame["date"].unique()):
        _, test, _ = fit_and_apply_references(
            frame.loc[frame["date"].ne(date)], frame.loc[frame["date"].eq(date)]
        )
        folds.append(test)
    return pd.concat(folds, ignore_index=True).sort_values(["cycle", "minute"])


def frozen_new_references(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    folds = []
    for date in sorted(new["date"].unique()):
        training = old.loc[old["date"].ne(date)] if date in set(old["date"]) else old
        _, test, _ = fit_and_apply_references(training, new.loc[new["date"].eq(date)])
        folds.append(test)
    return pd.concat(folds, ignore_index=True).sort_values(["cycle", "minute"])


def metric_rows(frame: pd.DataFrame, error: pd.Series, label: str) -> list[dict[str, object]]:
    rows = []
    for cycle, group in frame.assign(error=error).groupby("cycle", sort=False):
        values = group["error"].dropna()
        rows.append(
            {
                "cycle": cycle,
                "date": str(group["date"].iloc[0]),
                "model": label,
                "rmse": float(np.sqrt(np.mean(np.square(values)))),
                "mae": float(np.mean(np.abs(values))),
                "observations": len(values),
            }
        )
    return rows


def prospective_replication(new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = analysis_rows(new).dropna(subset=["z_r", "D_COP", "minute"])
    state_prediction = OLD_HINGE_SLOPE * np.maximum(selected["z_r"] - OLD_HINGE_THRESHOLD, 0.0)
    time_prediction = OLD_TIME_SLOPE * selected["minute"]
    rows = metric_rows(selected, selected["D_COP"] - state_prediction, "frozen state law")
    rows.extend(metric_rows(selected, selected["D_COP"] - time_prediction, "frozen time law"))
    cycle = pd.DataFrame(rows)
    date = (
        cycle.groupby(["date", "model"], as_index=False)
        .agg(rmse=("rmse", "mean"), mae=("mae", "mean"), cycles=("cycle", "nunique"))
        .sort_values(["date", "model"])
    )
    return cycle, date


def prospective_state_comparison(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    train = analysis_rows(old)
    test = analysis_rows(new)
    candidates = {
        "residual state": ["z_r"],
        "approach temperature": ["z_a"],
        "residual + coil": ["z_r", "z_coil"],
        "residual + suction": ["z_r", "z_suction"],
    }
    train_shared = shared_complete_cases(train, target="D_COP", models=candidates)
    test_shared = shared_complete_cases(test, target="D_COP", models=candidates)
    rows = []
    for name, features in candidates.items():
        model = fit_weighted_ridge(
            train_shared,
            target="D_COP",
            features=features,
            cycle="cycle",
        )
        prediction = model.predict(test_shared)
        rows.extend(metric_rows(test_shared, test_shared["D_COP"] - prediction, name))
    hinge = fit_hinge(test["z_r"], test["D_COP"])
    rows.append(
        {
            "cycle": "all_new_diagnostic",
            "date": "all_new",
            "model": "refit hinge threshold",
            "rmse": hinge.rmse,
            "mae": np.nan,
            "observations": int(test[["z_r", "D_COP"]].dropna().shape[0]),
            "threshold": hinge.threshold,
            "slope": hinge.slope,
        }
    )
    return pd.DataFrame(rows)


def prospective_rate_replication(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in [5, 10, 20]:
        old = old.copy()
        new = new.copy()
        old["future_z"] = add_cycle_future(
            old,
            column="z_r",
            horizon_steps=horizon,
            cycle="cycle",
            time="minute",
        )
        new["future_z"] = add_cycle_future(
            new,
            column="z_r",
            horizon_steps=horizon,
            cycle="cycle",
            time="minute",
        )
        models = {
            "current state": ["z_r"],
            "state + rate": ["z_r", "rate_5m"],
        }
        train_shared = shared_complete_cases(
            analysis_rows(old), target="future_z", models=models
        )
        test_shared = shared_complete_cases(
            analysis_rows(new), target="future_z", models=models
        )
        for label, features in models.items():
            model = fit_weighted_ridge(
                train_shared,
                target="future_z",
                features=features,
                cycle="cycle",
            )
            prediction = model.predict(test_shared)
            for row in metric_rows(
                test_shared, test_shared["future_z"] - prediction, label
            ):
                row["horizon_minutes"] = horizon
                rows.append(row)
    return pd.DataFrame(rows)


def nested_valid_cycle_analysis(  # noqa: C901
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_rows: list[dict[str, object]] = []
    dynamic_rows: list[dict[str, object]] = []
    mapping_models = {
        "residual state": ["z_r"],
        "approach temperature": ["z_a"],
        "residual + coil": ["z_r", "z_coil"],
        "residual + suction": ["z_r", "z_suction"],
        "residual + context": [
            "z_r",
            "ambient_temperature",
            "compressor_frequency",
            "fan_speed",
            "exv_opening",
        ],
    }
    dynamic_models = {
        "current state": ["z_r"],
        "state + rate": ["z_r", "rate_5m"],
        "short statistics": [
            "z_r",
            "rate_1m",
            "rate_3m",
            "rate_5m",
            "z_mean_5m",
            "z_std_5m",
        ],
        "statistics + context": [
            "z_r",
            "rate_1m",
            "rate_3m",
            "rate_5m",
            "z_mean_5m",
            "z_std_5m",
            "ambient_temperature",
            "compressor_frequency",
            "fan_speed",
            "exv_opening",
            "water_in_temperature",
            "water_flow",
        ],
        "raw 5-min history": ["z_r", *[f"z_lag_{lag}m" for lag in range(1, 6)]],
    }
    for date in sorted(raw["date"].unique()):
        train, test, _ = fit_and_apply_references(
            raw.loc[raw["date"].ne(date)], raw.loc[raw["date"].eq(date)]
        )
        train_eval = analysis_rows(train)
        test_eval = analysis_rows(test)
        for target in ["D_Qe_proxy", "D_Q", "D_COP"]:
            train_shared = shared_complete_cases(
                train_eval, target=target, models=mapping_models
            )
            test_shared = shared_complete_cases(
                test_eval, target=target, models=mapping_models
            )
            for label, features in mapping_models.items():
                model = fit_weighted_ridge(
                    train_shared,
                    target=target,
                    features=features,
                    cycle="cycle",
                )
                prediction = model.predict(test_shared)
                for row in metric_rows(test_shared, test_shared[target] - prediction, label):
                    row["target"] = target
                    mapping_rows.append(row)

        for horizon in [5, 10, 20, 30]:
            for prepared in [train, test]:
                prepared["future_z"] = add_cycle_future(
                    prepared,
                    column="z_r",
                    horizon_steps=horizon,
                    cycle="cycle",
                    time="minute",
                )
                prepared["future_D_Qe"] = add_cycle_future(
                    prepared,
                    column="D_Qe_proxy",
                    horizon_steps=horizon,
                    cycle="cycle",
                    time="minute",
                )
            train_eval = analysis_rows(train)
            test_eval = analysis_rows(test)
            train_shared = shared_complete_cases(
                train_eval, target="future_z", models=dynamic_models
            )
            test_shared = shared_complete_cases(
                test_eval, target="future_z", models=dynamic_models
            )
            for row in metric_rows(
                test_shared,
                test_shared["future_z"] - test_shared["z_r"],
                "persistence",
            ):
                row.update(horizon_minutes=horizon, target="z")
                dynamic_rows.append(row)
            for row in metric_rows(
                test_shared,
                test_shared["future_z"]
                - (test_shared["z_r"] + horizon * test_shared["rate_5m"]),
                "linear extrapolation",
            ):
                row.update(horizon_minutes=horizon, target="z")
                dynamic_rows.append(row)
            state_predictions: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
            for label, features in dynamic_models.items():
                model = fit_weighted_ridge(
                    train_shared,
                    target="future_z",
                    features=features,
                    cycle="cycle",
                )
                prediction = model.predict(test_shared)
                state_predictions[label] = (test_shared, prediction)
                for row in metric_rows(
                    test_shared, test_shared["future_z"] - prediction, label
                ):
                    row.update(horizon_minutes=horizon, target="z")
                    dynamic_rows.append(row)

            features = dynamic_models["short statistics"]
            direct_train = train_shared.dropna(subset=["future_D_Qe"])
            direct_test = test_shared.dropna(subset=["future_D_Qe"])
            direct = fit_weighted_ridge(
                direct_train,
                target="future_D_Qe",
                features=features,
                cycle="cycle",
            )
            direct_prediction = direct.predict(direct_test)
            for row in metric_rows(
                direct_test,
                direct_test["future_D_Qe"] - direct_prediction,
                "direct short statistics",
            ):
                row.update(horizon_minutes=horizon, target="D_Qe_proxy")
                dynamic_rows.append(row)

            state_test, predicted_state = state_predictions["short statistics"]
            performance_map = fit_weighted_ridge(
                train_eval.dropna(subset=["D_Qe_proxy", "z_r"]),
                target="D_Qe_proxy",
                features=["z_r"],
                cycle="cycle",
            )
            structured_prediction = performance_map.predict(pd.DataFrame({"z_r": predicted_state}))
            structured_test = state_test.dropna(subset=["future_D_Qe"]).copy()
            valid = np.isfinite(state_test["future_D_Qe"].to_numpy())
            structured_prediction = structured_prediction[valid]
            for row in metric_rows(
                structured_test,
                structured_test["future_D_Qe"] - structured_prediction,
                "structured short statistics",
            ):
                row.update(horizon_minutes=horizon, target="D_Qe_proxy")
                dynamic_rows.append(row)
    return pd.DataFrame(mapping_rows), pd.DataFrame(dynamic_rows)


def empirical_repeatability(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle, group in raw.loc[raw["early"]].groupby("cycle", sort=False):
        for quantity in [
            "heating_capacity",
            "cop",
            "evaporator_capacity",
            "water_delta_temperature",
        ]:
            values = group[quantity].dropna()
            center = float(values.median())
            if len(values) < 4 or not np.isfinite(center) or center == 0.0:
                continue
            differences = (values / center).diff().dropna()
            median = float(differences.median())
            sigma = 1.4826 * float((differences - median).abs().median()) / np.sqrt(2.0)
            rows.append(
                {
                    "cycle": cycle,
                    "quantity": quantity,
                    "repeatability_95_relative": 1.96 * sigma,
                }
            )
    return pd.DataFrame(rows)


def water_lag_audit(loader: DatasetLoader, cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = [
        "timestamp",
        "compressor_frequency",
        "water_delta_temperature",
        "water_delta_temperature__imputed",
    ]
    for cycle in cycles["cycle_name"].astype(str):
        frame = loader.load_cycle(cycle, columns=columns)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        minute = (
            frame.set_index("timestamp")
            .resample("1min")
            .agg(
                compressor_frequency=("compressor_frequency", "median"),
                water_delta_temperature=("water_delta_temperature", "median"),
                imputed=("water_delta_temperature__imputed", "max"),
            )
        )
        frequency = minute["compressor_frequency"]
        last_event: pd.Timestamp | None = None
        for index in range(3, len(minute) - 10):
            before = frequency.iloc[index - 3 : index]
            after = frequency.iloc[index : index + 3]
            if before.isna().any() or after.isna().any() or before.std() >= 2:
                continue
            change = float(after.median() - before.median())
            if abs(change) < 10:
                continue
            event_time = minute.index[index]
            if last_event is not None and (event_time - last_event).total_seconds() < 600:
                continue
            last_event = event_time
            response = minute["water_delta_temperature"]
            baseline_window = response.iloc[index - 3 : index].dropna()
            if baseline_window.empty:
                baseline = np.nan
                threshold = np.nan
            else:
                baseline = float(baseline_window.median())
                noise = float((baseline_window - baseline).abs().median())
                threshold = max(3 * noise, 0.1)
            direction = float(np.sign(change))
            lag = np.nan
            for future in range(index, index + 10):
                values = response.iloc[future : future + 2]
                observed = ~minute["imputed"].iloc[future : future + 2].fillna(True).astype(bool)
                if (
                    values.notna().all()
                    and observed.all()
                    and np.isfinite(threshold)
                    and (((values - baseline) * direction) > threshold).all()
                ):
                    lag = float(future - index)
                    break
            rows.append(
                {
                    "cycle": cycle,
                    "event_time": event_time,
                    "frequency_change_hz": change,
                    "water_response_lag_minutes": lag,
                    "baseline_deltaT_degC": baseline,
                    "response_threshold_degC": threshold,
                }
            )
    return pd.DataFrame(rows)


def cycle11_timeline(
    loader: DatasetLoader, stress_cycle: pd.DataFrame, old_raw: pd.DataFrame
) -> pd.DataFrame:
    record = stress_cycle.iloc[0]
    frame = loader.load_cycle(str(record["cycle_name"]), columns=LOAD_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    start = pd.Timestamp("2026-07-16 11:30:00")
    end = pd.Timestamp("2026-07-16 14:10:00")
    frame = (
        frame.loc[frame["timestamp"].between(start, end)]
        .set_index("timestamp")
        .select_dtypes(include="number")
        .resample("1min")
        .median()
        .reset_index()
    )
    frame["cycle"] = str(record["cycle_name"])
    frame["date"] = str(record["experiment_date"])[:10]
    frame["minute"] = (frame["timestamp"] - start).dt.total_seconds() / 60.0
    frame["early"] = frame["minute"].between(0, EARLY_MINUTES)
    training = old_raw.loc[old_raw["date"].ne(frame["date"].iloc[0])]
    _, prepared, _ = fit_and_apply_references(training, frame)
    prepared["phase"] = pd.cut(
        prepared["timestamp"],
        bins=[
            pd.Timestamp("2026-07-16 11:29:59"),
            pd.Timestamp("2026-07-16 11:58:59"),
            pd.Timestamp("2026-07-16 12:06:30"),
            pd.Timestamp("2026-07-16 12:52:59"),
            pd.Timestamp("2026-07-16 14:10:01"),
        ],
        labels=["cold candidate", "warm intrusion", "unstable recovery", "renewed degradation"],
    ).astype(str)
    return prepared


def model_to_dict(model: RidgeModel | ReferenceModel) -> dict[str, Any]:
    return {
        "features": list(model.features),
        "center": model.center.tolist(),
        "scale": model.scale.tolist(),
        "coefficients": model.coefficients.tolist(),
    }


def train_final_state_models(raw: pd.DataFrame) -> dict[str, object]:
    prepared, _, references = fit_and_apply_references(raw, raw.iloc[0:0].copy())
    features = ["z_r", "rate_5m"]
    eligible = sorted(
        analysis_rows(prepared).dropna(subset=features)["cycle"].astype(str).unique().tolist()
    )
    models = {}
    eligible_by_horizon: dict[str, int] = {}
    for horizon in [5, 10, 20]:
        prepared["future_z"] = add_cycle_future(
            prepared,
            column="z_r",
            horizon_steps=horizon,
            cycle="cycle",
            time="minute",
        )
        evaluation = analysis_rows(prepared).dropna(subset=["future_z", *features])
        model = fit_weighted_ridge(
            evaluation,
            target="future_z",
            features=features,
            cycle="cycle",
        )
        models[str(horizon)] = model_to_dict(model)
        eligible_by_horizon[str(horizon)] = int(evaluation["cycle"].nunique())
    return {
        "scope": "frost_cycle_000001-frost_cycle_000059; status=valid",
        "status_valid_cycles": int(raw["cycle"].nunique()),
        "eligible_cycles": len(eligible),
        "eligible_cycle_ids": eligible,
        "eligible_cycles_by_horizon": eligible_by_horizon,
        "calibration": "first 10 min after early stable proxy; inference starts after minute 10",
        "primary_target": "future z_r, not evaporator_capacity",
        "state_definition": "z_r = Te_normal(context) - Te",
        "reference_model": model_to_dict(references["evaporating_temperature"]),
        "dynamics_models": models,
    }


def build_summary(
    cohort: pd.DataFrame,
    replication: pd.DataFrame,
    prospective_dates: pd.DataFrame,
    state_comparison: pd.DataFrame,
    rate_replication: pd.DataFrame,
    current_mapping: pd.DataFrame,
    dynamics: pd.DataFrame,
    repeatability: pd.DataFrame,
    lag_events: pd.DataFrame,
    stress: pd.DataFrame,
    model_payload: dict[str, object],
    excluded_later_cycles: list[str],
) -> dict[str, object]:
    prospective = replication.groupby("model")[["rmse", "mae"]].mean()
    state = (
        state_comparison.loc[state_comparison["cycle"].ne("all_new_diagnostic")]
        .groupby("model")[["rmse", "mae"]]
        .mean()
    )
    rate = rate_replication.groupby(["horizon_minutes", "model"])[["rmse", "mae"]].mean()
    mapping = current_mapping.groupby(["target", "model"])[["rmse", "mae"]].mean()
    dynamic = dynamics.groupby(["target", "horizon_minutes", "model"])[["rmse", "mae"]].mean()
    repeat = repeatability.groupby("quantity")["repeatability_95_relative"].median()
    diagnostic = state_comparison.loc[state_comparison["cycle"].eq("all_new_diagnostic")].iloc[0]
    valid_lags = lag_events["water_response_lag_minutes"].dropna()
    estimable_lags = lag_events["response_threshold_degC"].notna()
    return {
        "scope": {
            "cycle_ids": "frost_cycle_000001-frost_cycle_000059",
            "valid_cycles": int(len(cohort)),
            "old_valid": int(cohort["cohort"].eq("old").sum()),
            "new_valid": int(cohort["cohort"].eq("new").sum()),
            "dates": int(cohort["experiment_date"].astype(str).str[:10].nunique()),
            "excluded_later_cycles": excluded_later_cycles,
        },
        "prospective_replication": {
            "cycle_balanced_rmse": prospective["rmse"].to_dict(),
            "cycle_balanced_mae": prospective["mae"].to_dict(),
            "dates": prospective_dates.to_dict(orient="records"),
            "cycles_where_state_beats_time": int(
                replication.pivot(index="cycle", columns="model", values="rmse")
                .eval("`frozen state law` < `frozen time law`")
                .sum()
            ),
        },
        "state_selection": {
            "new_cycle_errors": state.to_dict(orient="index"),
            "new_refit_hinge_threshold_degC": float(diagnostic.get("threshold", np.nan)),
            "new_refit_hinge_slope": float(diagnostic.get("slope", np.nan)),
            "rate_replication": {
                f"{horizon}_{model}": values.to_dict()
                for (horizon, model), values in rate.iterrows()
            },
        },
        "proxy_boundary": {
            "evaporator_capacity_lineage": "heating_capacity - compressor_power",
            "heating_capacity_internal_lineage_known": False,
            "traceable_sensor_accuracy_available": False,
            "empirical_repeatability_95_median": repeat.to_dict(),
            "detectable_water_step_events": int(len(valid_lags)),
            "candidate_water_step_events": int(len(lag_events)),
            "estimable_water_step_events": int(estimable_lags.sum()),
            "not_estimable_water_step_events": int((~estimable_lags).sum()),
            "water_step_lags_minutes": valid_lags.tolist(),
        },
        "all_cycle_mapping": {
            f"{target}_{model}": values.to_dict() for (target, model), values in mapping.iterrows()
        },
        "all_cycle_dynamics": {
            f"{target}_{horizon}_{model}": values.to_dict()
            for (target, horizon, model), values in dynamic.iterrows()
        },
        "cycle11": {
            "rh_observed_fraction": float(stress["environment_relative_humidity"].notna().mean()),
            "coil_minutes_above_zero": int(stress["coil_temperature"].gt(0).sum()),
            "claim": "elapsed time fails; reversible z and hysteresis remain unproven",
        },
        "formal_model": model_payload,
    }


def configure_plotting() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    for extension, dpi in [("svg", 300), ("pdf", 300), ("png", 300), ("tiff", 600)]:
        fig.savefig(
            FIGURES / f"{name}.{extension}",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def panel_label(axis: plt.Axes, label: str, *, x: float = -0.12) -> None:
    axis.text(
        x,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def figure_replication(
    new: pd.DataFrame, cycle_errors: pd.DataFrame, date_errors: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1.6, 0.9, 0.9]}
    )
    colors = {"2026-07-28": "#7884B4", "2026-07-29": "#B64342"}
    selected = analysis_rows(new)
    for date, group in selected.groupby("date"):
        axes[0].scatter(
            group["z_r"],
            group["D_COP"],
            s=7,
            alpha=0.20,
            color=colors[date],
            label=date[5:],
            rasterized=True,
        )
    x = np.linspace(0, max(20, float(selected["z_r"].quantile(0.995))), 200)
    axes[0].plot(
        x,
        OLD_HINGE_SLOPE * np.maximum(x - OLD_HINGE_THRESHOLD, 0),
        color="#272727",
        lw=1.8,
        label="frozen 40-cycle law",
    )
    axes[0].set(xlabel=r"Residual state $z_r$ (°C)", ylabel=r"Relative COP loss $D_{COP}$")
    axes[0].set_title("The old law predicts the ten later cycles")
    axes[0].legend(fontsize=6.2)
    panel_label(axes[0], "a")

    pivot = cycle_errors.pivot(index="cycle", columns="model", values="rmse").sort_index()
    y = np.arange(len(pivot))
    axes[1].plot(pivot["frozen time law"], y, "o", color="#A8A8A8", label="time")
    axes[1].plot(pivot["frozen state law"], y, "o", color="#0F4D92", label="state")
    for index in y:
        axes[1].plot(
            [pivot["frozen state law"].iloc[index], pivot["frozen time law"].iloc[index]],
            [index, index],
            color="#D8D8D8",
            lw=0.7,
        )
    axes[1].set_yticks(y, [name[-3:] for name in pivot.index])
    axes[1].set_xlabel("Cycle RMSE")
    axes[1].set_title("10/10 cycles improve")
    axes[1].legend(fontsize=6.2)
    panel_label(axes[1], "b")

    date_pivot = date_errors.pivot(index="date", columns="model", values="rmse")
    positions = np.arange(len(date_pivot))
    width = 0.34
    axes[2].bar(
        positions - width / 2,
        date_pivot["frozen time law"],
        width,
        color="#A8A8A8",
        label="time",
    )
    axes[2].bar(
        positions + width / 2,
        date_pivot["frozen state law"],
        width,
        color="#0F4D92",
        label="state",
    )
    axes[2].set_xticks(positions, [value[5:] for value in date_pivot.index])
    axes[2].set_ylabel("Cycle-balanced RMSE")
    axes[2].set_title("New date and later\nsame-date cycles agree")
    panel_label(axes[2], "c")
    fig.suptitle("Replication on later cycles: state, not time, organizes degradation", y=1.03)
    fig.tight_layout()
    save_figure(fig, "figure_1_prospective_replication")


def figure_proxy_boundary(repeatability: pd.DataFrame, lag_events: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.3, 0.9, 0.8]}
    )
    axes[0].axis("off")
    box = dict(boxstyle="round,pad=0.35", fc="#E0F0F0", ec="#42949E", lw=0.8)
    axes[0].text(0.04, 0.76, "controller heating\ncapacity proxy", bbox=box, ha="left")
    axes[0].text(0.04, 0.28, "compressor electric\npower", bbox=box, ha="left")
    axes[0].text(
        0.62,
        0.55,
        r"$Q_{e,proxy}=Q_{controller}-P_{comp}$",
        bbox=dict(boxstyle="round,pad=0.4", fc="#F0E0D0", ec="#B8792D", lw=0.8),
        ha="center",
    )
    axes[0].annotate("", xy=(0.51, 0.64), xytext=(0.36, 0.76), arrowprops={"arrowstyle": "->"})
    axes[0].annotate("", xy=(0.51, 0.50), xytext=(0.36, 0.34), arrowprops={"arrowstyle": "->"})
    axes[0].text(
        0.5,
        0.06,
        "not a measured refrigerant-side capacity\n"
        "controller lineage and sensor accuracy unavailable",
        ha="center",
        color="#B64342",
        fontsize=6.5,
    )
    axes[0].set_title("The apparent evaporator capacity\nis a balance proxy", pad=7)
    panel_label(axes[0], "a")

    summary = (
        repeatability.groupby("quantity")["repeatability_95_relative"]
        .median()
        .reindex(["cop", "heating_capacity", "evaporator_capacity", "water_delta_temperature"])
    )
    labels = ["COP", "capacity", r"$Q_{e,proxy}$", r"$\Delta T_w$"]
    axes[1].bar(labels, 100 * summary, color=["#7884B4", "#7884B4", "#B8792D", "#A8A8A8"])
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("Empirical 95% repeatability floor (%)")
    axes[1].set_title("Small changes are\nnot distinguishable", pad=7)
    panel_label(axes[1], "b")

    lags = lag_events["water_response_lag_minutes"]
    estimable = lag_events["response_threshold_degC"].notna()
    detected = estimable & lags.notna()
    not_detected = estimable & lags.isna()
    not_estimable = ~estimable
    positions = np.arange(len(lags))
    axes[2].scatter(positions[detected], lags[detected], s=30, color="#0F4D92", label="detected")
    axes[2].scatter(
        positions[not_detected],
        np.full(not_detected.sum(), 10.0),
        marker="^",
        facecolors="none",
        edgecolors="#A8A8A8",
        s=28,
        label="not detected by 10 min",
    )
    axes[2].scatter(
        positions[not_estimable],
        np.full(not_estimable.sum(), 10.55),
        marker="x",
        color="#B64342",
        s=24,
        label="not evaluable",
    )
    axes[2].set(
        xlabel="Candidate control event",
        ylabel="Water response lag (min)",
        xticks=positions,
        ylim=(-0.25, 11.2),
    )
    axes[2].set_title(
        f"{int(detected.sum())}/{int(estimable.sum())} evaluable\nresponses detected", pad=7
    )
    axes[2].legend(fontsize=5.3, loc="center right")
    panel_label(axes[2], "c")
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, "figure_2_proxy_and_water_boundary")


def figure_state_selection(state: pd.DataFrame, mapping: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), gridspec_kw={"width_ratios": [1, 1.25]})
    prospective = (
        state.loc[state["cycle"].ne("all_new_diagnostic")]
        .groupby("model")["rmse"]
        .mean()
        .sort_values()
    )
    axes[0].barh(np.arange(len(prospective)), prospective, color="#527E8D")
    axes[0].set_yticks(np.arange(len(prospective)), prospective.index)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("New-cycle RMSE")
    axes[0].set_title("A simpler approach coordinate does not\nreplace the residual state", pad=7)
    panel_label(axes[0], "a", x=-0.16)

    selected = mapping.loc[mapping["target"].isin(["D_COP", "D_Qe_proxy"])]
    summary = selected.groupby(["target", "model"])["rmse"].mean().unstack(0)
    keep = ["residual state", "residual + coil", "residual + suction", "residual + context"]
    summary = summary.reindex(keep)
    y = np.arange(len(summary))
    axes[1].plot(summary["D_COP"], y - 0.10, "o", color="#0F4D92", label="COP proxy")
    axes[1].plot(summary["D_Qe_proxy"], y + 0.10, "s", color="#B8792D", label=r"$Q_{e,proxy}$")
    axes[1].set_yticks(y, summary.index)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Leave-one-date-out cycle RMSE")
    axes[1].set_title("Extra variables do not close\nthe proxy uncertainty", pad=7)
    axes[1].legend(fontsize=6.3)
    panel_label(axes[1], "b")
    fig.tight_layout(w_pad=1.4)
    save_figure(fig, "figure_3_state_selection")


def figure_cycle11(stress: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.5), sharex=True)
    phase_colors = {
        "cold candidate": "#E0F0F0",
        "warm intrusion": "#F6CFCB",
        "unstable recovery": "#F0E0D0",
        "renewed degradation": "#E0E0F0",
    }
    for axis in axes:
        for phase, group in stress.groupby("phase", sort=False):
            axis.axvspan(
                group["minute"].min(),
                group["minute"].max(),
                color=phase_colors.get(str(phase), "#EEEEEE"),
                alpha=0.45,
                lw=0,
            )
    axes[0].plot(stress["minute"], stress["ambient_temperature"], color="#B64342", label=r"$T_a$")
    axes[0].plot(stress["minute"], stress["coil_temperature"], color="#7884B4", label="coil")
    axes[0].axhline(0, color="#767676", ls="--", lw=0.7)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(ncol=2, fontsize=6.3)
    axes[0].text(
        0.99,
        0.92,
        "RH missing throughout",
        transform=axes[0].transAxes,
        ha="right",
        color="#B64342",
    )
    axes[0].set_title("Time advances while the measured thermal condition recovers")
    panel_label(axes[0], "a")

    axes[1].plot(
        stress["minute"], stress["evaporating_temperature"], color="#0F4D92", label=r"$T_e$"
    )
    axes[1].plot(
        stress["minute"],
        stress["evaporating_temperature_normal"],
        color="#767676",
        label=r"$T_{e,normal}$",
    )
    twin = axes[1].twinx()
    twin.plot(stress["minute"], stress["z_r"], color="#B8792D", lw=1.2, label=r"$z_r$")
    axes[1].set_ylabel("Evaporating temperature (°C)")
    twin.set_ylabel(r"$z_r$ (°C)")
    handles = axes[1].get_lines() + twin.get_lines()
    axes[1].legend(handles, [line.get_label() for line in handles], ncol=3, fontsize=6.3)
    axes[1].set_title("The frozen residual state does not show a clean reversible path")
    panel_label(axes[1], "b")

    baseline = stress.loc[stress["timestamp"].between("2026-07-16 11:55", "2026-07-16 11:58")]
    for column, color, label in [
        ("heating_capacity", "#0F4D92", "capacity proxy"),
        ("cop", "#42949E", "COP proxy"),
        ("power_total", "#767676", "total power"),
    ]:
        center = float(baseline[column].median())
        axes[2].plot(stress["minute"], stress[column] / center, color=color, label=label)
    axes[2].axhline(1, color="#767676", ls="--", lw=0.7)
    axes[2].set(xlabel="Minutes after 11:30", ylabel="Relative to 11:55–11:58")
    axes[2].legend(ncol=3, fontsize=6.3)
    axes[2].set_title("Performance recovery is confounded by simultaneous control changes")
    panel_label(axes[2], "c")
    fig.tight_layout()
    save_figure(fig, "figure_4_cycle11_stress_test")


def figure_model_selection(dynamics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.2, 1.1, 0.9]}
    )
    state = dynamics.loc[dynamics["target"].eq("z")]
    summary = state.groupby(["horizon_minutes", "model"])["mae"].mean().unstack()
    colors = {
        "persistence": "#A8A8A8",
        "current state": "#7884B4",
        "state + rate": "#0F4D92",
        "short statistics": "#42949E",
        "statistics + context": "#B8792D",
        "raw 5-min history": "#9A4D8E",
    }
    for model in colors:
        if model in summary:
            axes[0].plot(
                summary.index, summary[model], marker="o", color=colors[model], label=model
            )
    axes[0].set(xlabel="Forecast horizon (min)", ylabel="Cycle-balanced MAE (°C)")
    axes[0].set_title("Rate is useful; longer\nrepresentations add little", pad=7)
    axes[0].legend(fontsize=5.6, ncol=2)
    panel_label(axes[0], "a")

    by_date = (
        state.loc[state["horizon_minutes"].eq(10)]
        .groupby(["date", "model"])["mae"]
        .mean()
        .unstack()
    )
    delta = by_date["state + rate"] - by_date["short statistics"]
    axes[1].bar(np.arange(len(delta)), delta, color=np.where(delta > 0, "#2E9E44", "#A8A8A8"))
    axes[1].axhline(0, color="#272727", lw=0.7)
    axes[1].set_xticks(np.arange(len(delta)), [value[5:] for value in delta.index], rotation=45)
    axes[1].set_ylabel("MAE(state+rate) − MAE(short stats), °C")
    axes[1].set_title("Short statistics do not\nwin consistently", pad=7)
    panel_label(axes[1], "b")

    performance = dynamics.loc[
        dynamics["target"].eq("D_Qe_proxy") & dynamics["horizon_minutes"].isin([5, 10, 20])
    ]
    perf_summary = performance.groupby(["horizon_minutes", "model"])["mae"].mean().unstack()
    axes[2].plot(
        perf_summary.index,
        100 * perf_summary["structured short statistics"],
        "o-",
        color="#0F4D92",
        label="structured",
    )
    axes[2].plot(
        perf_summary.index,
        100 * perf_summary["direct short statistics"],
        "s--",
        color="#B8792D",
        label="direct",
    )
    axes[2].set(xlabel="Forecast horizon (min)", ylabel=r"$Q_{e,proxy}$ MAE (percentage points)")
    axes[2].set_title("Direct prediction adds\nno practical gain", pad=7)
    axes[2].legend(fontsize=6.2)
    panel_label(axes[2], "c")
    fig.tight_layout(w_pad=1.3)
    save_figure(fig, "figure_5_model_selection")


if __name__ == "__main__":
    main()
