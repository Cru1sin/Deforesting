"""V2.6.6 experiment-LOEO identification curves."""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EVIDENCE_ROOT = Path("output/test/成本函数")
STATE_COLUMNS = (
    "water_in_temperature",
    "water_out_temperature",
    "coil_temperature",
    "evaporating_pressure",
    "water_temperature_setpoint",
)
QD_FEATURES = (
    "water_in_temperature",
    "water_out_temperature",
    "rule_defrost_duration_minutes",
    "coil_temperature",
    "evaporating_pressure",
)


def _timestamp(value: object) -> pd.Timestamp | None:
    result = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(result) else pd.Timestamp(result)


def _first_timestamp(*values: object) -> pd.Timestamp | None:
    for value in values:
        result = _timestamp(value)
        if result is not None:
            return result
    return None


def _first_reason(*values: object) -> str:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value)
    return "catalog_not_in_curve_cohort"


def _is_true(value: object) -> bool:
    return bool(value) if pd.notna(value) else False


def _catalog(loader: Any) -> pd.DataFrame:
    if hasattr(loader, "list_cycles"):
        return loader.list_cycles().copy()
    records = loader.catalog["cycles"]
    rows = []
    for record in records:
        row = dict(record)
        for nested in ("boundaries", "data"):
            if isinstance(row.get(nested), dict):
                row.update(row[nested])
        rows.append(row)
    return pd.DataFrame(rows)


def _default_sources() -> dict[str, object]:
    network = pd.read_csv(
        EVIDENCE_ROOT / "ED模型/经验经济窗口/证据/preparation_inclusive_network_events.csv"
    )
    return {
        "preparation": pd.read_csv(
            EVIDENCE_ROOT / "准备阶段供热量/经验经济窗口/证据/preparation_heat_events.csv"
        ),
        "preparation_network_cycle_names": network.loc[
            network["status"].eq("included"), "cycle_name"
        ],
        "tickets": pd.read_csv(
            EVIDENCE_ROOT / "ED模型/经验经济窗口/证据/ticket_event_features_and_predictions.csv"
        ),
        "recovery": pd.read_csv(EVIDENCE_ROOT / "其他/经验经济窗口/证据/recovery_events.csv"),
    }


def _frame(
    loader: Any, cycle_name: str, cache: dict[str, pd.DataFrame] | None = None
) -> pd.DataFrame:
    if cache is not None and cycle_name in cache:
        return cache[cycle_name]
    result = loader.load_cycle_original(
        cycle_name, columns=["timestamp", "power_total", "heating_capacity", *STATE_COLUMNS]
    ).copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    for column in result.columns.drop("timestamp"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = (
        result.dropna(subset=["timestamp"])
        .sort_values("timestamp", kind="stable")
        .drop_duplicates("timestamp")
    )
    if cache is not None:
        cache[cycle_name] = result
    return result


def _raw_integral(
    frame: pd.DataFrame, column: str, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, object]:
    values = frame.loc[frame["timestamp"].between(start, end), ["timestamp", column]].copy()
    values[column] = pd.to_numeric(values[column], errors="coerce")
    elapsed = max((end - start).total_seconds(), 0.0)
    if values.empty or elapsed <= 0:
        return {"energy": np.nan, "coverage": 0.0, "start_fresh": False, "end_fresh": False}
    seconds = values["timestamp"].diff().dt.total_seconds()
    pairs = values[column].notna() & values[column].shift().notna() & seconds.gt(0) & seconds.le(5)
    energy = ((values[column] + values[column].shift()) / 2 * seconds).where(pairs, 0).sum() / 3600
    valid = values.loc[values[column].notna(), "timestamp"]
    start_fresh = not valid.empty and abs((valid.iloc[0] - start).total_seconds()) <= 5
    end_fresh = not valid.empty and 0 <= (end - valid.iloc[-1]).total_seconds() <= 5
    return {
        "energy": float(energy),
        "coverage": float(seconds.where(pairs, 0).sum() / elapsed),
        "start_fresh": bool(start_fresh),
        "end_fresh": bool(end_fresh),
    }


def _gap_metrics(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[float, float, bool]:
    timestamps = frame.loc[frame["timestamp"].between(start, end), "timestamp"]
    gaps = timestamps.diff().dt.total_seconds().dropna()
    missing = gaps.loc[gaps.gt(5)]
    previous = frame.loc[frame["timestamp"].le(end), "timestamp"].max()
    following = frame.loc[frame["timestamp"].ge(end), "timestamp"].min()
    in_gap = (
        pd.notna(previous)
        and pd.notna(following)
        and previous < end < following
        and (following - previous).total_seconds() > 5
    )
    return float(missing.sum()), float(missing.max()) if not missing.empty else 0.0, bool(in_gap)


def _state(frame: pd.DataFrame, candidate: pd.Timestamp) -> dict[str, object]:
    window = frame.loc[
        frame["timestamp"].ge(candidate - pd.Timedelta(seconds=60))
        & frame["timestamp"].lt(candidate)
    ]
    result: dict[str, object] = {}
    sufficient = True
    seconds = window["timestamp"].dt.floor("s")
    for column in STATE_COLUMNS:
        valid = window[column].notna()
        count = int(seconds.loc[valid].nunique())
        result[column] = float(window.loc[valid, column].median()) if valid.any() else np.nan
        result[f"{column}_valid_second_count"] = count
        sufficient &= count >= 48
    result["state_window_eligible"] = sufficient
    return result


def _setpoint_for_event(
    loader: Any, cycle_name: str, boundary: object, cache: dict[str, pd.DataFrame]
) -> float:
    frame = _frame(loader, cycle_name, cache)
    end = _timestamp(boundary)
    if end is None:
        end = frame["timestamp"].max()
    values = frame.loc[
        frame["timestamp"].lt(end) & frame["timestamp"].ge(end - pd.Timedelta(seconds=60)),
        "water_temperature_setpoint",
    ].dropna()
    return float(values.median()) if not values.empty else np.nan


def _with_preparation_setpoints(
    preparation: pd.DataFrame, loader: Any, cache: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    result = preparation.copy()
    if "water_temperature_setpoint" not in result:
        result["water_temperature_setpoint"] = np.nan
    missing = pd.to_numeric(result["water_temperature_setpoint"], errors="coerce").isna()
    for index, row in result.loc[missing].iterrows():
        with suppress(KeyError, ValueError, FileNotFoundError):
            result.loc[index, "water_temperature_setpoint"] = _setpoint_for_event(
                loader,
                str(row["cycle_name"]),
                row.get("defrost_preparation_start"),
                cache,
            )
    return result


def _design(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    return pd.concat([numeric, numeric.pow(2).add_suffix("_squared")], axis=1)


def _ridge_parameters(model: Any) -> dict[str, object]:
    ridge = model[-1]
    return {
        "intercept": float(ridge.intercept_),
        "coefficients": np.asarray(ridge.coef_, dtype=float).tolist(),
    }


def _training_audit(frame: pd.DataFrame) -> dict[str, object]:
    ids = sorted(frame["experiment_id"].dropna().astype(str).unique())
    return {
        "training_event_count": len(frame),
        "training_experiment_count": len(ids),
        "training_experiment_ids": ids,
    }


def _models(  # noqa: C901
    experiment: str,
    setpoint: float,
    loader: Any,
    catalog: pd.DataFrame,
    sources: dict[str, object],
    frame_cache: dict[str, pd.DataFrame],
) -> dict[str, object]:
    preparation_all = _with_preparation_setpoints(
        pd.DataFrame(sources["preparation"]), loader, frame_cache
    )
    preparation_all = preparation_all.loc[
        preparation_all["status"].eq("included")
        & pd.to_numeric(preparation_all["preparation_signed_heat_kwh"], errors="coerce").notna()
    ].copy()
    names = set(pd.Series(sources["preparation_network_cycle_names"]).dropna().astype(str))
    ed_preparation = preparation_all.loc[
        preparation_all["cycle_name"].astype(str).isin(names)
    ].copy()
    tickets = pd.DataFrame(sources["tickets"]).loc[lambda x: x["valid"].fillna(False)].copy()
    recovery = pd.DataFrame(sources["recovery"]).copy()
    train_prep = preparation_all.loc[~preparation_all["experiment_id"].astype(str).eq(experiment)]
    ed_train = ed_preparation.loc[
        ~ed_preparation["experiment_id"].astype(str).eq(experiment)
    ].dropna(subset=["evaporating_pressure", "inclusive_energy_kwh"])
    train_ticket = tickets.loc[~tickets["experiment_id"].astype(str).eq(experiment)]
    train_recovery = recovery.loc[
        ~recovery["experiment_id"].astype(str).eq(experiment)
        & recovery["recovery_valid"].fillna(False)
        & pd.to_numeric(recovery["recovery_electricity_coverage"], errors="coerce").ge(0.95)
        & pd.to_numeric(recovery["recovery_water_heat_coverage"], errors="coerce").ge(0.95)
    ]
    training_ids = sorted(
        set(train_prep["experiment_id"].dropna().astype(str))
        | set(train_ticket["experiment_id"].dropna().astype(str))
        | set(train_recovery["experiment_id"].dropna().astype(str))
    )
    result: dict[str, object] = {"training_experiment_ids": ",".join(training_ids)}

    if not ed_train.empty:
        result["ed_model"] = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(
            _design(ed_train, ("evaporating_pressure",)), ed_train["inclusive_energy_kwh"]
        )
    duration_train = train_ticket.dropna(
        subset=["coil_temperature", "rule_defrost_duration_minutes"]
    )
    if not duration_train.empty:
        result["duration_model"] = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(
            duration_train[["coil_temperature"]], duration_train["rule_defrost_duration_minutes"]
        )
        result["duration_t3_range"] = (
            float(duration_train["coil_temperature"].min()),
            float(duration_train["coil_temperature"].max()),
        )
        result["duration_range"] = (
            float(duration_train["rule_defrost_duration_minutes"].min()),
            float(duration_train["rule_defrost_duration_minutes"].max()),
        )
    qd_train = train_ticket.dropna(subset=["defrost_absorbed_heat_kwh"])
    if not qd_train.empty:
        result["qd_model"] = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1)
        ).fit(_design(qd_train, QD_FEATURES), qd_train["defrost_absorbed_heat_kwh"])

    same_prep = train_prep.loc[
        pd.to_numeric(train_prep["water_temperature_setpoint"], errors="coerce").eq(setpoint)
    ]
    same_recovery = train_recovery.loc[
        pd.to_numeric(train_recovery["pre_water_temperature_setpoint"], errors="coerce").eq(
            setpoint
        )
    ]
    if same_prep["experiment_id"].nunique() >= 3:
        result["Qprep"] = float(
            pd.to_numeric(same_prep["preparation_signed_heat_kwh"], errors="coerce").median()
        )
    if same_recovery["experiment_id"].nunique() >= 3:
        result["ER"] = float(
            pd.to_numeric(same_recovery["recovery_electricity_kwh"], errors="coerce").median()
        )
        result["QR"] = float(
            pd.to_numeric(same_recovery["recovery_water_heat_kwh"], errors="coerce").median()
        )

    lambdas = []
    for _, record in catalog.loc[~catalog["experiment_id"].astype(str).eq(experiment)].iterrows():
        start = _timestamp(record.get("stable_heating_start"))
        if start is None:
            continue
        try:
            raw = _frame(loader, str(record["cycle_name"]), frame_cache)
        except (KeyError, ValueError, FileNotFoundError):
            continue
        event_setpoint = pd.to_numeric(
            raw.loc[
                raw["timestamp"].between(start, start + pd.Timedelta(seconds=60)),
                "water_temperature_setpoint",
            ],
            errors="coerce",
        ).median()
        electricity = _raw_integral(raw, "power_total", start, start + pd.Timedelta(seconds=60))
        heat = _raw_integral(raw, "heating_capacity", start, start + pd.Timedelta(seconds=60))
        if (
            event_setpoint == setpoint
            and min(electricity["coverage"], heat["coverage"]) >= 0.95
            and heat["energy"] > 0
        ):
            lambdas.append(
                {
                    "experiment_id": str(record["experiment_id"]),
                    "cycle_name": str(record["cycle_name"]),
                    "lambda0": electricity["energy"] / heat["energy"],
                }
            )
    lambda_frame = pd.DataFrame(lambdas)
    if not lambda_frame.empty and lambda_frame["experiment_id"].nunique() >= 3:
        result["lambda0"] = float(lambda_frame["lambda0"].median())
    empty_audit = {
        "training_event_count": 0,
        "training_experiment_count": 0,
        "training_experiment_ids": [],
    }
    component_audits = {
        "ED": _training_audit(ed_train),
        "duration": _training_audit(duration_train),
        "QD": _training_audit(qd_train),
        "Qprep": _training_audit(same_prep),
        "recovery": _training_audit(same_recovery),
        "lambda0": _training_audit(lambda_frame) if not lambda_frame.empty else empty_audit,
    }
    component_audits["lambda0"]["anchor_cycle_count"] = len(lambda_frame)
    component_audits["lambda0"]["anchor_cycle_names"] = (
        sorted(lambda_frame["cycle_name"].astype(str)) if not lambda_frame.empty else []
    )
    recovery_cohort = recovery.loc[
        recovery["recovery_valid"].fillna(False)
        & pd.to_numeric(recovery["recovery_electricity_coverage"], errors="coerce").ge(0.95)
        & pd.to_numeric(recovery["recovery_water_heat_coverage"], errors="coerce").ge(0.95)
    ]
    provenance = {
        "protocol": "experiment-LOEO",
        "heldout_experiment_id": experiment,
        "training_experiment_ids": training_ids,
        "ed_cohort_event_count": len(ed_preparation),
        "ed_cohort_experiment_count": ed_preparation["experiment_id"].nunique(),
        "qprep_cohort_event_count": len(preparation_all),
        "qprep_cohort_experiment_count": preparation_all["experiment_id"].nunique(),
        "ticket_cohort_event_count": len(tickets),
        "ticket_cohort_experiment_count": tickets["experiment_id"].nunique(),
        "recovery_cohort_event_count": len(recovery_cohort),
        "recovery_cohort_experiment_count": recovery_cohort["experiment_id"].nunique(),
        "ed_training_event_count": len(ed_train),
        "ed_training_experiment_count": ed_train["experiment_id"].nunique(),
        "components": component_audits,
    }
    for name in ("ed", "duration", "qd"):
        model = result.get(f"{name}_model")
        if model is not None:
            provenance[f"{name}_ridge"] = _ridge_parameters(model)
    result["component_provenance"] = json.dumps(provenance, sort_keys=True)
    return result


def _component_predictions(
    state: dict[str, object], models: dict[str, object]
) -> dict[str, object]:
    pe = float(state["evaporating_pressure"])
    t3 = float(state["coil_temperature"])
    result = {
        key: models.get(key, np.nan)
        for key in (
            "Qprep",
            "ER",
            "QR",
            "lambda0",
            "training_experiment_ids",
            "component_provenance",
        )
    }
    try:
        result["ED"] = float(
            models["ed_model"].predict(
                pd.DataFrame(
                    {"evaporating_pressure": [pe], "evaporating_pressure_squared": [pe**2]}
                )
            )[0]
        )
        duration = float(
            models["duration_model"].predict(pd.DataFrame({"coil_temperature": [t3]}))[0]
        )
        result["predicted_defrost_duration_minutes"] = duration
        qd_row = {
            column: state[column]
            for column in QD_FEATURES
            if column != "rule_defrost_duration_minutes"
        }
        qd_row["rule_defrost_duration_minutes"] = duration
        result["QD"] = float(
            models["qd_model"].predict(_design(pd.DataFrame([qd_row]), QD_FEATURES))[0]
        )
        t3_range = models["duration_t3_range"]
        duration_range = models["duration_range"]
        result["component_extrapolated_duration"] = not (
            t3_range[0] <= t3 <= t3_range[1] and duration_range[0] <= duration <= duration_range[1]
        )
    except (KeyError, ValueError):
        result.update(
            {
                "ED": np.nan,
                "predicted_defrost_duration_minutes": np.nan,
                "QD": np.nan,
                "component_extrapolated_duration": False,
            }
        )
    return result


def _basin(
    curve: pd.DataFrame, fraction: float, optimum_index: int
) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    threshold = float(curve.loc[optimum_index, "J"]) * (1 + fraction)
    mask = curve["optimization_eligible"].fillna(False) & curve["J"].le(threshold)
    position = curve.index.get_loc(optimum_index)
    left = right = position
    while left > 0 and bool(mask.iloc[left - 1]):
        left -= 1
    while right + 1 < len(curve) and bool(mask.iloc[right + 1]):
        right += 1
    start, end = curve.iloc[left]["candidate_time"], curve.iloc[right]["candidate_time"]
    return pd.Timestamp(start), pd.Timestamp(end), (end - start).total_seconds() / 60


def _classify_cycle(curve: pd.DataFrame, optimum_index: int, basin_indices: set[int]) -> str:
    eligible_indices = curve.index[curve["optimization_eligible"].fillna(False)].tolist()
    if len(eligible_indices) < 2:
        if len(curve) <= 1 or len(eligible_indices) == 1:
            return "unidentifiable_boundary"
        return (
            "measurement_limited"
            if not curve["measurement_eligible"].fillna(False).any()
            else "unidentifiable_component"
        )
    if curve.loc[list(basin_indices), "component_extrapolated_duration"].fillna(False).any():
        return "component_extrapolated"
    if optimum_index == eligible_indices[0]:
        return "left_boundary_limited"
    if optimum_index == eligible_indices[-1]:
        if optimum_index == curve.index[-1]:
            return "right_censored"
        trailing = curve.loc[curve.index > optimum_index]
        return (
            "measurement_limited"
            if not trailing["measurement_eligible"].fillna(False).all()
            else "unidentifiable_component"
        )
    return "identified_curve"


def build_v266_table(  # noqa: C901
    points: pd.DataFrame, loader: Any, sources: dict[str, object] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build candidate curves and one audit row for every catalog cycle."""
    production = sources is None
    evidence = _default_sources() if sources is None else sources
    catalog = _catalog(loader)
    point_by_cycle = points.set_index("cycle_name", drop=False)
    tables: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    frame_cache: dict[str, pd.DataFrame] = {}
    for _, record in catalog.iterrows():
        cycle_name = str(record["cycle_name"])
        if cycle_name not in point_by_cycle.index or not _is_true(
            point_by_cycle.loc[cycle_name].get("valid", False)
        ):
            point = point_by_cycle.loc[cycle_name] if cycle_name in point_by_cycle.index else {}
            reason = _first_reason(point.get("failure_reason"), record.get("status_reason"))
            audit_rows.append(
                {
                    "cycle_name": cycle_name,
                    "experiment_id": record.get("experiment_id"),
                    "cycle_status": "unidentifiable_boundary",
                    "failure_reason": reason,
                    "eligible_candidate_count": 0,
                }
            )
            continue
        point = point_by_cycle.loc[cycle_name]
        stable = _first_timestamp(record.get("stable_heating_start"), point.get("t_heating_stable"))
        end = _first_timestamp(
            point.get("candidate_end"),
            point.get("t_actual_preparation"),
            record.get("defrost_preparation_start"),
        )
        if stable is None or end is None or end < stable + pd.Timedelta(minutes=1):
            audit_rows.append(
                {
                    "cycle_name": cycle_name,
                    "experiment_id": record.get("experiment_id"),
                    "cycle_status": "unidentifiable_boundary",
                    "failure_reason": "invalid_candidate_boundary",
                    "eligible_candidate_count": 0,
                }
            )
            continue
        raw = _frame(loader, cycle_name, frame_cache)
        experiment = str(record.get("experiment_id", point.get("experiment_id")))
        initial_state = _state(raw, stable + pd.Timedelta(minutes=1))
        setpoint = float(initial_state["water_temperature_setpoint"])
        models = _models(experiment, setpoint, loader, catalog, evidence, frame_cache)
        candidates = list(pd.date_range(stable + pd.Timedelta(minutes=1), end, freq="min"))
        if candidates and candidates[-1] != end:
            candidates.append(end)
        rows = []
        for candidate in candidates:
            electricity = _raw_integral(raw, "power_total", stable, candidate)
            heat = _raw_integral(raw, "heating_capacity", stable, candidate)
            gap_total, max_gap, candidate_in_gap = _gap_metrics(raw, stable, candidate)
            state = _state(raw, candidate)
            components = _component_predictions(state, models)
            endpoint_fresh = (
                electricity["start_fresh"]
                and electricity["end_fresh"]
                and heat["start_fresh"]
                and heat["end_fresh"]
            )
            measurement = (
                min(electricity["coverage"], heat["coverage"]) >= 0.95
                and endpoint_fresh
                and not candidate_in_gap
                and bool(state["state_window_eligible"])
                and heat["energy"] > 0
            )
            finite_components = np.isfinite(
                [components[key] for key in ("ED", "ER", "Qprep", "QD", "QR", "lambda0")]
            ).all()
            eligible = bool(measurement and finite_components)
            eh, qh, lambda0 = electricity["energy"], heat["energy"], components["lambda0"]
            loss = float(max(eh - lambda0 * qh, 0)) if eligible else np.nan
            ticket = (
                float(
                    max(
                        components["ED"]
                        + components["ER"]
                        - lambda0 * (components["Qprep"] - components["QD"] + components["QR"]),
                        0,
                    )
                )
                if eligible
                else np.nan
            )
            rows.append(
                {
                    "cycle_name": cycle_name,
                    "experiment_id": experiment,
                    "candidate_time": candidate,
                    "cycle_start": record.get("start_time"),
                    "t_heating_stable": stable,
                    "actual_preparation_time": _first_timestamp(
                        point.get("t_actual_preparation"),
                        record.get("defrost_preparation_start"),
                    ),
                    "t_RB": _timestamp(point.get("t_RB")),
                    "rb_status": point.get("rb_status"),
                    "heating_electricity_kwh": eh,
                    "unit_heating_kwh": qh,
                    "power_coverage": electricity["coverage"],
                    "unit_coverage": heat["coverage"],
                    "integration_coverage": min(electricity["coverage"], heat["coverage"]),
                    "gap_seconds_total": gap_total,
                    "max_gap_seconds": max_gap,
                    "endpoint_fresh": endpoint_fresh,
                    "candidate_in_gap": candidate_in_gap,
                    "endpoint_extrapolated": False,
                    **state,
                    **components,
                    "L": loss,
                    "K": ticket,
                    "J": lambda0 + (loss + ticket) / qh if eligible else np.nan,
                    "inverse_cop": lambda0 + (loss + ticket) / qh if eligible else np.nan,
                    "mixed_heat_basis": True,
                    "measurement_eligible": measurement,
                    "optimization_eligible": eligible,
                    "valid": eligible,
                    "failure_reason": ""
                    if eligible
                    else ("measurement_limited" if not measurement else "unidentifiable_component"),
                    "recommended_time": pd.NaT,
                    "hard_label_eligible": False,
                    "decision_status": "abstain_v266_identification_only",
                    "t_star_semantics": "diagnostic_raw_argmin_not_label",
                    "algorithm": "v2.6.6",
                }
            )
        curve = pd.DataFrame(rows)
        eligible_count = int(curve["optimization_eligible"].sum())
        if eligible_count >= 2:
            optimum_index = int(curve["J"].idxmin())
            optimum = pd.Timestamp(curve.loc[optimum_index, "candidate_time"])
            curve["raw_t_star"] = optimum
            curve["t_star"] = optimum
            basin_indices: set[int] = {optimum_index}
            for fraction in (0.01, 0.05):
                start, basin_end, width = _basin(curve, fraction, optimum_index)
                prefix = f"basin_{int(fraction * 100)}pct"
                curve[f"{prefix}_start"] = start
                curve[f"{prefix}_end"] = basin_end
                curve[f"{prefix}_width_minutes"] = width
                if fraction == 0.01:
                    basin_indices = set(
                        curve.index[curve["candidate_time"].between(start, basin_end)]
                    )
            status = _classify_cycle(curve, optimum_index, basin_indices)
        else:
            curve["raw_t_star"] = pd.NaT
            curve["t_star"] = pd.NaT
            for fraction in (1, 5):
                curve[f"basin_{fraction}pct_start"] = pd.NaT
                curve[f"basin_{fraction}pct_end"] = pd.NaT
                curve[f"basin_{fraction}pct_width_minutes"] = np.nan
            status = _classify_cycle(curve, 0, set())
        eligible = curve["optimization_eligible"].fillna(False)
        minimum = pd.to_numeric(curve.loc[eligible, "inverse_cop"], errors="coerce").min()
        curve["cycle_cop"] = 1 / pd.to_numeric(curve["inverse_cop"], errors="coerce")
        curve["relative_regret"] = (
            pd.to_numeric(curve["inverse_cop"], errors="coerce") / minimum - 1
        ).where(eligible)
        curve["near_optimal_1pct"] = eligible & curve["relative_regret"].le(0.01)
        curve["near_optimal_5pct"] = eligible & curve["relative_regret"].le(0.05)
        curve["t_star_model_supported"] = (
            not bool(curve.loc[optimum_index, "component_extrapolated_duration"])
            if eligible_count >= 2
            else False
        )
        failure = "" if status == "identified_curve" else status
        curve["cycle_status"] = status
        if failure:
            curve.loc[curve["failure_reason"].eq(""), "failure_reason"] = failure
        audit_rows.append(
            {
                "cycle_name": cycle_name,
                "experiment_id": experiment,
                "cycle_status": status,
                "failure_reason": failure,
                "eligible_candidate_count": eligible_count,
                "training_experiment_ids": models["training_experiment_ids"],
                "component_provenance": models["component_provenance"],
            }
        )
        tables.append(curve)
    table = pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame()
    audit = pd.DataFrame(audit_rows)
    if production:
        if len(audit) != 101:
            raise RuntimeError(f"v2.6.6 audit must cover 101 catalog cycles, got {len(audit)}")
        sufficient = int(audit["eligible_candidate_count"].ge(2).sum())
        if sufficient < 60:
            raise RuntimeError(
                f"v2.6.6 only has {sufficient}/69 curves with at least two eligible candidates"
            )
    table.attrs["cycle_audit"] = audit
    return table, audit
