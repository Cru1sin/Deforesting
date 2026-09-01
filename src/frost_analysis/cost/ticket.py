"""V2.6.7 LOEO terminal-ticket identification curves."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .identification import (
    EVIDENCE_ROOT,
    STATE_COLUMNS,
    _basin,
    _catalog,
    _first_reason,
    _first_timestamp,
    _frame,
    _gap_metrics,
    _is_true,
    _raw_integral,
    _state,
    _timestamp,
)

TARGET_COLUMNS = {
    "E_T": "E_T_observed_kwh",
    "Q_T": "Q_T_observed_kwh",
}
MIN_TRAINING_EVENTS = 40
MIN_TRAINING_EXPERIMENTS = 10


def ticket_cost(eh: float, qh: float, e_ticket: float, q_ticket: float) -> float:
    """Return the frozen V2.6.7 ratio, or NaN outside its physical domain."""
    numerator, denominator = eh + e_ticket, qh + q_ticket
    return (
        float(numerator / denominator)
        if np.isfinite([numerator, denominator]).all()
        and numerator > 0
        and denominator > 0
        and e_ticket >= 0
        else np.nan
    )


def prediction_in_support(row: pd.Series, support: dict[str, tuple[float, float]]) -> bool:
    values = pd.to_numeric(row[list(STATE_COLUMNS)], errors="coerce")
    return bool(
        values.notna().all()
        and all(
            support[column][0] <= values[column] <= support[column][1]
            for column in STATE_COLUMNS
        )
    )


def fit_ticket_fold(events: pd.DataFrame, heldout_experiment: str) -> dict[str, dict[str, object]]:
    """Fit the two independent frozen pipelines without the held-out experiment."""
    result: dict[str, dict[str, object]] = {}
    for target, observed in TARGET_COLUMNS.items():
        train = events.loc[
            events[observed].notna()
            & ~events["experiment_id"].astype(str).eq(str(heldout_experiment))
        ].copy()
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1)
        ).fit(train[list(STATE_COLUMNS)], train[observed])
        support = {
            column: (
                float(pd.to_numeric(train[column], errors="coerce").min()),
                float(pd.to_numeric(train[column], errors="coerce").max()),
            )
            for column in STATE_COLUMNS
        }
        result[target] = {
            "model": model,
            "support": support,
            "training_event_count": len(train),
            "training_experiment_count": train["experiment_id"].nunique(),
            "training_experiment_ids": sorted(train["experiment_id"].astype(str).unique()),
            "training_mean": float(train[observed].mean()),
        }
    return result


def classify_cycle(  # noqa: C901
    curve: pd.DataFrame, optimum_index: int | None
) -> str:
    eligible = curve["optimization_eligible"].fillna(False)
    eligible_indices = curve.index[eligible].tolist()
    measurement = curve["measurement_eligible"].fillna(False)
    supported = curve["model_supported"].fillna(False)
    component = curve.get("component_eligible", pd.Series(True, index=curve.index)).fillna(False)
    if len(curve) < 2:
        return "unidentifiable_boundary"
    if len(eligible_indices) < 2:
        if measurement.sum() < 2:
            return "measurement_limited"
        if (measurement & component).sum() < 2:
            return "unidentifiable_component"
        if (measurement & component & supported).sum() < 2:
            return "model_support_limited"
        return "unidentifiable_component"
    assert optimum_index is not None
    first_grid, last_grid = curve.index[0], curve.index[-1]
    if optimum_index == first_grid:
        return "left_boundary_limited"
    if optimum_index == last_grid:
        if not component.all():
            return "unidentifiable_component"
        if bool((measurement & component & supported).all()):
            return "right_censored"
    if optimum_index in (eligible_indices[0], eligible_indices[-1]):
        before = optimum_index == eligible_indices[0]
        outside = curve.loc[curve.index < optimum_index if before else curve.index > optimum_index]
        if not outside.empty and not outside["measurement_eligible"].fillna(False).all():
            return "measurement_limited"
        if not outside.empty and not outside["model_supported"].fillna(False).all():
            return "model_support_limited"
    return "identified_curve"


def finalize_curve(curve: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Attach diagnostic argmin/basins and the identification-only decision contract."""
    result = curve.sort_values("candidate_time", kind="stable").reset_index(drop=True).copy()
    eligible = result["optimization_eligible"].fillna(False) & result["J"].notna()
    optimum_index: int | None = None
    if int(eligible.sum()) >= 2:
        minimum = result.loc[eligible, "J"].min()
        optimum_index = int(result.index[eligible & result["J"].eq(minimum)][0])
        optimum = pd.Timestamp(result.loc[optimum_index, "candidate_time"])
        result["raw_t_star"] = optimum
        result["t_star"] = optimum
        for fraction in (0.01, 0.05):
            start, end, width = _basin(result, fraction, optimum_index)
            prefix = f"basin_{int(100 * fraction)}pct"
            result[f"{prefix}_start"] = start
            result[f"{prefix}_end"] = end
            result[f"{prefix}_width_minutes"] = width
    else:
        result["raw_t_star"] = pd.NaT
        result["t_star"] = pd.NaT
        for percent in (1, 5):
            result[f"basin_{percent}pct_start"] = pd.NaT
            result[f"basin_{percent}pct_end"] = pd.NaT
            result[f"basin_{percent}pct_width_minutes"] = np.nan
    status = classify_cycle(result, optimum_index)
    result["cycle_status"] = status
    result["recommended_time"] = pd.NaT
    result["hard_label_eligible"] = False
    result["decision_status"] = "abstain_v267_identification_only"
    result["t_star_semantics"] = "diagnostic_raw_argmin_not_label"
    return result, status


def _default_sources() -> dict[str, pd.DataFrame]:
    return {
        "tickets": pd.read_csv(
            EVIDENCE_ROOT / "ED模型/经验经济窗口/证据/ticket_event_features_and_predictions.csv"
        ),
        "recovery": pd.read_csv(
            EVIDENCE_ROOT / "其他/经验经济窗口/证据/recovery_events.csv"
        ),
    }


def _ticket_events(
    loader: Any,
    sources: dict[str, pd.DataFrame],
    frame_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    keys = ["cycle_name", "experiment_id"]
    events = _catalog(loader)[[*keys, "defrost_preparation_start", "defrost_start"]].copy()
    if events["cycle_name"].duplicated().any():
        raise ValueError("catalog must contain one row per cycle_name")
    tickets = sources["tickets"][[
        "cycle_name",
        "experiment_id",
        "defrost_electricity_kwh",
        "defrost_electricity_coverage",
        "defrost_absorbed_heat_kwh",
        "defrost_signed_heat_coverage",
    ]]
    recovery = sources["recovery"][[
        "cycle_name",
        "experiment_id",
        "recovery_valid",
        "recovery_electricity_kwh",
        "recovery_electricity_coverage",
        "recovery_water_heat_kwh",
        "recovery_water_heat_coverage",
    ]]
    catalog_experiment = events.set_index("cycle_name")["experiment_id"].astype(str)
    for name, component in (("tickets", tickets), ("recovery", recovery)):
        if component["cycle_name"].duplicated().any():
            raise ValueError(f"{name} must contain one row per cycle_name")
        unknown = ~component["cycle_name"].isin(catalog_experiment.index)
        mismatch = component["experiment_id"].astype(str).ne(
            component["cycle_name"].map(catalog_experiment)
        )
        if unknown.any() or mismatch.any():
            raise ValueError(f"{name} experiment ownership does not match catalog")
    events = events.merge(
        tickets.rename(columns={"experiment_id": "ticket_experiment_id"}),
        on="cycle_name",
        how="left",
        validate="one_to_one",
    ).merge(
        recovery.rename(columns={"experiment_id": "recovery_experiment_id"}),
        on="cycle_name",
        how="left",
        validate="one_to_one",
    )
    states = []
    for row in events.itertuples(index=False):
        boundary = _timestamp(row.defrost_preparation_start)
        defrost = _timestamp(row.defrost_start)
        qprep = {"energy": np.nan, "coverage": 0.0, "start_fresh": False, "end_fresh": False}
        try:
            raw = _frame(loader, str(row.cycle_name), frame_cache)
            state = _state(raw, boundary)
            if boundary is not None and defrost is not None:
                qprep = _raw_integral(raw, "heating_capacity", boundary, defrost)
                _, _, end_in_gap = _gap_metrics(raw, boundary, defrost)
            else:
                end_in_gap = True
        except (KeyError, ValueError, FileNotFoundError, TypeError):
            state = {column: np.nan for column in STATE_COLUMNS}
            state["state_window_eligible"] = False
            end_in_gap = True
        state_ok = bool(state.get("state_window_eligible", False))
        qprep_ok = bool(
            qprep["coverage"] >= 0.95
            and qprep["start_fresh"]
            and qprep["end_fresh"]
            and not end_in_gap
            and np.isfinite(qprep["energy"])
        )
        ed_ok = bool(
            pd.to_numeric(row.defrost_electricity_coverage, errors="coerce") >= 0.95
            and np.isfinite(pd.to_numeric(row.defrost_electricity_kwh, errors="coerce"))
        )
        qd_ok = bool(
            pd.to_numeric(row.defrost_signed_heat_coverage, errors="coerce") >= 0.95
            and np.isfinite(pd.to_numeric(row.defrost_absorbed_heat_kwh, errors="coerce"))
        )
        recovery_ok = bool(row.recovery_valid) if pd.notna(row.recovery_valid) else False
        er_ok = bool(
            recovery_ok
            and pd.to_numeric(row.recovery_electricity_coverage, errors="coerce") >= 0.95
            and np.isfinite(pd.to_numeric(row.recovery_electricity_kwh, errors="coerce"))
        )
        qr_ok = bool(
            recovery_ok
            and pd.to_numeric(row.recovery_water_heat_coverage, errors="coerce") >= 0.95
            and np.isfinite(pd.to_numeric(row.recovery_water_heat_kwh, errors="coerce"))
        )
        e_ticket = (
            float(row.defrost_electricity_kwh + row.recovery_electricity_kwh)
            if ed_ok and er_ok
            else np.nan
        )
        q_ticket = (
            float(qprep["energy"] - row.defrost_absorbed_heat_kwh + row.recovery_water_heat_kwh)
            if state_ok and qprep_ok and qd_ok and qr_ok
            else np.nan
        )
        states.append(
            {
                **{column: state[column] for column in STATE_COLUMNS},
                "training_state_window_eligible": state_ok,
                "Qprep_observed_kwh": qprep["energy"] if qprep_ok else np.nan,
                "Qprep_coverage": qprep["coverage"],
                "E_T_observed_kwh": e_ticket,
                "Q_T_observed_kwh": q_ticket,
            }
        )
    return pd.concat([events.reset_index(drop=True), pd.DataFrame(states)], axis=1)


def _fold_provenance(fold: dict[str, object]) -> str:
    model = fold["model"]
    imputer, scaler, ridge = model
    return json.dumps(
        {
            "protocol": "experiment-LOEO",
            "features": list(STATE_COLUMNS),
            "pipeline": "SimpleImputer(median)->StandardScaler()->Ridge(alpha=1)",
            "training_event_count": fold["training_event_count"],
            "training_experiment_count": fold["training_experiment_count"],
            "training_experiment_ids": fold["training_experiment_ids"],
            "imputer_medians": np.asarray(imputer.statistics_, dtype=float).tolist(),
            "scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
            "scaler_scale": np.asarray(scaler.scale_, dtype=float).tolist(),
            "ridge_intercept": float(ridge.intercept_),
            "ridge_coefficients": np.asarray(ridge.coef_, dtype=float).tolist(),
            "support_minmax": fold["support"],
        },
        sort_keys=True,
    )


def _loeo_audit(
    events: pd.DataFrame, folds: dict[str, dict[str, dict[str, object]]]
) -> pd.DataFrame:
    rows = []
    for experiment, heldout in events.groupby("experiment_id", sort=False):
        for target, observed in TARGET_COLUMNS.items():
            fold = folds[str(experiment)][target]
            available = heldout[observed].notna()
            if not available.any():
                continue
            values = heldout.loc[available]
            predictions = fold["model"].predict(values[list(STATE_COLUMNS)])
            for (_, event), prediction in zip(values.iterrows(), predictions, strict=True):
                observed_value = float(event[observed])
                rows.append(
                    {
                        "cycle_name": event["cycle_name"],
                        "experiment_id": experiment,
                        "target": target,
                        "observed_kwh": observed_value,
                        "loeo_prediction_kwh": float(prediction),
                        "training_mean_kwh": fold["training_mean"],
                        "residual_kwh": observed_value - float(prediction),
                        "baseline_residual_kwh": observed_value - float(fold["training_mean"]),
                        "supported": prediction_in_support(event, fold["support"]),
                        "heldout_experiment_id": experiment,
                        "training_event_count": fold["training_event_count"],
                        "training_experiment_count": fold["training_experiment_count"],
                        "training_experiment_ids": ",".join(fold["training_experiment_ids"]),
                        "model_provenance": _fold_provenance(fold),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_audit(
    table: pd.DataFrame,
    events: pd.DataFrame,
    *,
    replicates: int = 200,
    seed: int = 267,
) -> pd.DataFrame:
    """Experiment-bootstrap argmin stability for originally identified curves."""
    rng = np.random.default_rng(seed)
    identified = table.loc[table["cycle_status"].eq("identified_curve")]
    cycles = {name: curve.copy() for name, curve in identified.groupby("cycle_name", sort=False)}
    experiments = sorted(events["experiment_id"].dropna().astype(str).unique())
    counts = {name: [0, 0] for name in cycles}
    for _ in range(replicates):
        for heldout in experiments:
            names = [name for name in experiments if name != heldout]
            sampled = rng.choice(names, size=len(names), replace=True)
            train = pd.concat(
                [events.loc[events["experiment_id"].astype(str).eq(name)] for name in sampled],
                ignore_index=True,
            )
            fold = fit_ticket_fold(train, "__bootstrap_no_holdout__")
            for cycle_name, curve in cycles.items():
                if str(curve["experiment_id"].iloc[0]) != heldout:
                    continue
                supported = pd.Series(True, index=curve.index)
                predictions: dict[str, np.ndarray] = {}
                for target in TARGET_COLUMNS:
                    predictions[target] = fold[target]["model"].predict(curve[list(STATE_COLUMNS)])
                    support = fold[target]["support"]
                    states = curve[list(STATE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
                    lower = pd.Series({column: support[column][0] for column in STATE_COLUMNS})
                    upper = pd.Series({column: support[column][1] for column in STATE_COLUMNS})
                    supported &= states.notna().all(axis=1) & states.ge(lower).all(
                        axis=1
                    ) & states.le(upper).all(axis=1)
                values = np.array(
                    [
                        ticket_cost(eh, qh, et, qt)
                        for eh, qh, et, qt in zip(
                            curve["heating_electricity_kwh"],
                            curve["unit_heating_kwh"],
                            predictions["E_T"],
                            predictions["Q_T"],
                            strict=True,
                        )
                    ]
                )
                eligible = (
                    curve["measurement_eligible"].fillna(False).to_numpy()
                    & supported.to_numpy()
                    & np.isfinite(values)
                )
                if eligible.sum() < 2:
                    continue
                counts[cycle_name][0] += 1
                position = np.flatnonzero(eligible)[np.argmin(values[eligible])]
                candidate = pd.Timestamp(curve.iloc[position]["candidate_time"])
                start = pd.Timestamp(curve["basin_5pct_start"].iloc[0])
                end = pd.Timestamp(curve["basin_5pct_end"].iloc[0])
                counts[cycle_name][1] += int(start <= candidate <= end)
    return pd.DataFrame(
        [
            {
                "cycle_name": name,
                "repeat_count": replicates,
                "two_candidate_repeat_count": eligible_count,
                "two_candidate_repeat_fraction": eligible_count / replicates,
                "argmin_in_original_5pct_basin_count": hit_count,
                "argmin_in_original_5pct_basin_fraction": (
                    hit_count / eligible_count if eligible_count else 0.0
                ),
            }
            for name, (eligible_count, hit_count) in counts.items()
        ]
    )


def _oracle_curve(curve: pd.DataFrame, event: pd.Series | None) -> pd.DataFrame:
    oracle = curve.copy()
    for column in oracle.columns:
        if column.startswith(("E_T_", "Q_T_")) and any(
            token in column
            for token in ("hat", "model", "training", "support_minmax")
        ):
            oracle[column] = np.nan
    oracle["formula"] = "(EH+E_T_observed)/(QH+Q_T_observed)"
    oracle["E_T_model_supported"] = np.nan
    oracle["Q_T_model_supported"] = np.nan
    if event is None:
        oracle["E_T_observed_kwh"] = np.nan
        oracle["Q_T_observed_kwh"] = np.nan
        oracle["J"] = np.nan
        oracle["optimization_eligible"] = False
        oracle["component_eligible"] = False
        oracle["model_supported"] = False
        oracle["cycle_electricity_kwh"] = np.nan
        oracle["cycle_net_heat_kwh"] = np.nan
    else:
        e_ticket = float(event["E_T_observed_kwh"])
        q_ticket = float(event["Q_T_observed_kwh"])
        oracle["E_T_observed_kwh"] = e_ticket
        oracle["Q_T_observed_kwh"] = q_ticket
        oracle["cycle_electricity_kwh"] = oracle["heating_electricity_kwh"] + e_ticket
        oracle["cycle_net_heat_kwh"] = oracle["unit_heating_kwh"] + q_ticket
        oracle["J"] = [
            ticket_cost(eh, qh, e_ticket, q_ticket)
            for eh, qh in zip(
                oracle["heating_electricity_kwh"], oracle["unit_heating_kwh"], strict=True
            )
        ]
        oracle["optimization_eligible"] = (
            oracle["measurement_eligible"].fillna(False) & oracle["J"].notna()
        )
        oracle["component_eligible"] = oracle["J"].notna()
        oracle["component_prediction_valid"] = oracle["component_eligible"]
        oracle["model_supported"] = oracle["component_eligible"]
    oracle["inverse_cop"] = oracle["J"]
    oracle, _ = finalize_curve(oracle)
    oracle["valid"] = oracle["optimization_eligible"]
    oracle["component_prediction_valid"] = oracle["component_eligible"]
    oracle["failure_reason"] = np.where(
        ~oracle["measurement_eligible"].fillna(False),
        "measurement_limited",
        np.where(~oracle["component_eligible"].fillna(False), "unidentifiable_component", ""),
    )
    not_identified = ~oracle["cycle_status"].eq("identified_curve")
    oracle.loc[not_identified & oracle["failure_reason"].eq(""), "failure_reason"] = oracle.loc[
        not_identified, "cycle_status"
    ]
    oracle["t_star_model_supported"] = bool(oracle["optimization_eligible"].sum() >= 2)
    oracle["algorithm"] = "v1-r"
    oracle["oracle_only"] = True
    oracle["available_at_candidate_time"] = False
    return oracle


def build_v267_table(  # noqa: C901
    points: pd.DataFrame,
    loader: Any,
    sources: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build V2.6.7 curves plus the frozen audit artifacts."""
    evidence = _default_sources() if sources is None else sources
    catalog = _catalog(loader)
    point_by_cycle = points.set_index("cycle_name", drop=False)
    frame_cache: dict[str, pd.DataFrame] = {}
    events = _ticket_events(loader, evidence, frame_cache)
    folds = {
        str(experiment): fit_ticket_fold(events, str(experiment))
        for experiment in events["experiment_id"].dropna().astype(str).unique()
    }
    event_by_cycle = events.set_index("cycle_name", drop=False)
    tables: list[pd.DataFrame] = []
    oracle_tables: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for _, record in catalog.iterrows():
        cycle_name = str(record["cycle_name"])
        point = point_by_cycle.loc[cycle_name] if cycle_name in point_by_cycle.index else {}
        if cycle_name not in point_by_cycle.index or not _is_true(point.get("valid", False)):
            audit_rows.append(
                {
                    "cycle_name": cycle_name,
                    "experiment_id": record.get("experiment_id"),
                    "cycle_status": "unidentifiable_boundary",
                    "failure_reason": _first_reason(
                        point.get("failure_reason"), record.get("status_reason")
                    ),
                    "eligible_candidate_count": 0,
                }
            )
            continue
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
        experiment = str(record.get("experiment_id", point.get("experiment_id")))
        raw = _frame(loader, cycle_name, frame_cache)
        fold = folds.get(experiment)
        candidates = list(pd.date_range(stable + pd.Timedelta(minutes=1), end, freq="min"))
        if candidates and candidates[-1] != end:
            candidates.append(end)
        rows = []
        for candidate in candidates:
            electricity = _raw_integral(raw, "power_total", stable, candidate)
            heat = _raw_integral(raw, "heating_capacity", stable, candidate)
            gap_total, max_gap, candidate_in_gap = _gap_metrics(raw, stable, candidate)
            state = _state(raw, candidate)
            endpoint_fresh = all(
                (
                    electricity["start_fresh"],
                    electricity["end_fresh"],
                    heat["start_fresh"],
                    heat["end_fresh"],
                )
            )
            measurement = bool(
                min(electricity["coverage"], heat["coverage"]) >= 0.95
                and endpoint_fresh
                and not candidate_in_gap
                and state["state_window_eligible"]
                and heat["energy"] > 0
            )
            predictions: dict[str, float] = {"E_T": np.nan, "Q_T": np.nan}
            support = {"E_T": False, "Q_T": False}
            training_ok = False
            if fold is not None:
                for target in TARGET_COLUMNS:
                    predictions[target] = float(
                        fold[target]["model"].predict(pd.DataFrame([state])[list(STATE_COLUMNS)])[0]
                    )
                    support[target] = prediction_in_support(
                        pd.Series(state), fold[target]["support"]
                    )
                training_ok = all(
                    fold[target]["training_event_count"] >= MIN_TRAINING_EVENTS
                    and fold[target]["training_experiment_count"] >= MIN_TRAINING_EXPERIMENTS
                    for target in TARGET_COLUMNS
                )
            model_supported = bool(training_ok and all(support.values()))
            value = ticket_cost(
                float(electricity["energy"]),
                float(heat["energy"]),
                predictions["E_T"],
                predictions["Q_T"],
            )
            component = bool(np.isfinite(value))
            cycle_electricity = float(electricity["energy"] + predictions["E_T"])
            cycle_net_heat = float(heat["energy"] + predictions["Q_T"])
            eligible = bool(measurement and model_supported and component)
            if not measurement:
                failure = "measurement_limited"
            elif not model_supported:
                failure = "model_support_limited"
            elif not component:
                failure = "unidentifiable_component"
            else:
                failure = ""
            row = {
                "cycle_name": cycle_name,
                "experiment_id": experiment,
                "candidate_time": candidate,
                "cycle_start": record.get("start_time"),
                "t_heating_stable": stable,
                "actual_preparation_time": _first_timestamp(
                    point.get("t_actual_preparation"), record.get("defrost_preparation_start")
                ),
                "t_RB": _timestamp(point.get("t_RB")),
                "rb_status": point.get("rb_status"),
                "candidate_end": end,
                "heating_electricity_kwh": electricity["energy"],
                "unit_heating_kwh": heat["energy"],
                "power_coverage": electricity["coverage"],
                "unit_coverage": heat["coverage"],
                "integration_coverage": min(electricity["coverage"], heat["coverage"]),
                "gap_seconds_total": gap_total,
                "max_gap_seconds": max_gap,
                "endpoint_fresh": endpoint_fresh,
                "candidate_in_gap": candidate_in_gap,
                "endpoint_extrapolated": False,
                **state,
                "E_T_hat_kwh": predictions["E_T"],
                "Q_T_hat_kwh": predictions["Q_T"],
                "E_T_model_supported": support["E_T"],
                "Q_T_model_supported": support["Q_T"],
                "model_supported": model_supported,
                "component_eligible": component,
                "component_prediction_valid": component,
                "cycle_electricity_kwh": cycle_electricity,
                "cycle_net_heat_kwh": cycle_net_heat,
                "mixed_heat_basis": True,
                "J": value if eligible else np.nan,
                "inverse_cop": value if eligible else np.nan,
                "measurement_eligible": measurement,
                "optimization_eligible": eligible,
                "valid": eligible,
                "failure_reason": failure,
                "algorithm": "v2.6.7",
                "formula": "(EH+E_T_hat)/(QH+Q_T_hat)",
                "prediction_clipped": False,
                "interpolated": False,
            }
            if fold is not None:
                for target in TARGET_COLUMNS:
                    prefix = target
                    row[f"{prefix}_training_event_count"] = fold[target]["training_event_count"]
                    row[f"{prefix}_training_experiment_count"] = fold[target][
                        "training_experiment_count"
                    ]
                    row[f"{prefix}_training_experiment_ids"] = ",".join(
                        fold[target]["training_experiment_ids"]
                    )
                    row[f"{prefix}_model_provenance"] = _fold_provenance(fold[target])
                    row[f"{prefix}_support_minmax_json"] = json.dumps(
                        fold[target]["support"], sort_keys=True
                    )
            rows.append(row)
        curve, status = finalize_curve(pd.DataFrame(rows))
        eligible = curve["optimization_eligible"].fillna(False)
        minimum = pd.to_numeric(curve.loc[eligible, "J"], errors="coerce").min()
        curve["cycle_cop"] = 1 / pd.to_numeric(curve["inverse_cop"], errors="coerce")
        curve["relative_regret"] = (
            pd.to_numeric(curve["inverse_cop"], errors="coerce") / minimum - 1
        ).where(eligible)
        curve["near_optimal_1pct"] = eligible & curve["relative_regret"].le(0.01)
        curve["near_optimal_5pct"] = eligible & curve["relative_regret"].le(0.05)
        curve["t_star_model_supported"] = bool(eligible.sum() >= 2)
        curve.loc[curve["failure_reason"].eq(""), "failure_reason"] = (
            "" if status == "identified_curve" else status
        )
        event = event_by_cycle.loc[cycle_name] if cycle_name in event_by_cycle.index else None
        if isinstance(event, pd.DataFrame):
            event = event.iloc[0]
        if event is not None and pd.isna(event["Q_T_observed_kwh"]):
            event = None
        oracle = _oracle_curve(curve, event)
        oracle_tables.append(oracle)
        oracle_eligible = oracle["optimization_eligible"].fillna(False)
        main_t = pd.to_datetime(curve["raw_t_star"].iloc[0], errors="coerce")
        oracle_t = pd.to_datetime(oracle["raw_t_star"].iloc[0], errors="coerce")
        oracle_min = pd.to_numeric(oracle.loc[oracle_eligible, "J"], errors="coerce").min()
        at_main = oracle.loc[oracle["candidate_time"].eq(main_t), "J"]
        basin_start = pd.to_datetime(oracle["basin_5pct_start"].iloc[0], errors="coerce")
        basin_end = pd.to_datetime(oracle["basin_5pct_end"].iloc[0], errors="coerce")
        comparison_rows.append(
            {
                "cycle_name": cycle_name,
                "experiment_id": experiment,
                "candidate_intersection_consistent": curve["candidate_time"].equals(
                    oracle["candidate_time"]
                ),
                "main_raw_t_star": main_t,
                "oracle_raw_t_star": oracle_t,
                "main_t_star_in_oracle_5pct_basin": bool(
                    pd.notna(main_t)
                    and pd.notna(basin_start)
                    and basin_start <= main_t <= basin_end
                ),
                "oracle_regret_at_main_t_star": (
                    float(at_main.iloc[0] / oracle_min - 1)
                    if len(at_main) and pd.notna(oracle_min) and oracle_min > 0
                    else np.nan
                ),
                "within_cycle_spearman": curve.loc[
                    curve["candidate_time"].isin(oracle.loc[oracle_eligible, "candidate_time"]), "J"
                ].corr(oracle.loc[oracle_eligible, "J"], method="spearman"),
                "comparable": bool(eligible.sum() >= 2 and oracle_eligible.sum() >= 2),
            }
        )
        audit_rows.append(
            {
                "cycle_name": cycle_name,
                "experiment_id": experiment,
                "cycle_status": status,
                "failure_reason": "" if status == "identified_curve" else status,
                "eligible_candidate_count": int(eligible.sum()),
                "measurement_eligible_candidate_count": int(
                    curve["measurement_eligible"].sum()
                ),
                "joint_supported_candidate_count": int(curve["model_supported"].sum()),
            }
        )
        tables.append(curve)
    table = pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame()
    artifacts = {
        "cycle_audit": pd.DataFrame(audit_rows),
        "ticket_loeo": _loeo_audit(events, folds),
        "v1r": pd.concat(oracle_tables, ignore_index=True, sort=False)
        if oracle_tables
        else pd.DataFrame(),
        "v1r_audit": pd.DataFrame(comparison_rows),
        "bootstrap_audit": bootstrap_audit(table, events, replicates=200, seed=267),
    }
    return table, artifacts
