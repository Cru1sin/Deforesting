"""V2.7 literature-inspired evaluation metrics on the V2.6.8 data boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_config
from sklearn.utils.parallel import Parallel, delayed

from .core import water_side_heating_kw
from .outcome import (
    DYNAMIC_8,
    Q_MIN_KWH,
    RidgeOutcomeModel,
    _catalog,
    _fit_fold_set,
    _load_frame,
    _long_support_runs,
    _model_provenance,
    _prediction_support,
    _robust_observation_cadence_seconds,
    _sorted_time_slice,
    build_v268_table,
    fit_outcome_fold,
    fit_weighted_ridge,
)

HEALTHY_FEATURES = (
    "ambient_temperature",
    "water_in_temperature",
    "water_flow",
    "compressor_frequency",
    "water_temperature_setpoint",
)

DYNAMIC_LOSS_TARGET_PROVENANCE = (
    "cross_fitted_healthy_water_heat_reference_minus_observed_Q_T"
)


@dataclass(frozen=True)
class MetricDefinition:
    algorithm: str
    direction: str
    source_idea: str
    project_definition: str
    label: str
    formula: str
    unit: str = "-"


METRICS = {
    "cop_cyc_evt": MetricDefinition(
        "v2.7.4",
        "max",
        "complete event-cycle COP",
        "fixed-9 stable-to-stable water-side heat divided by total electricity",
        "Complete-cycle COP",
        "(water_heating_kwh+Q_T_hat_kwh)/(heating_electricity_kwh+E_T_hat_kwh)",
    ),
    "eta_h_cyc": MetricDefinition(
        "v2.7.4",
        "max",
        "algebraically closed heating-service retention",
        "actual complete-cycle water heat over the dynamic healthy counterfactual",
        "Heating-service retention",
        "(water_heating_kwh+Q_T_hat_kwh)/"
        "(healthy_water_heating_kwh+healthy_water_heat_kw*D_T_hat_minutes/60)",
    ),
    "eta_e_cyc": MetricDefinition(
        "v2.7.0",
        "max",
        "evaporator/air-source cycle capacity",
        "water-side inferred air-source heat over a cross-fitted healthy air-source reference",
        "Evaporator cycle capacity efficiency",
        "(evaporator_heating_kwh+Q_T_hat_kwh-E_comp_T_hat_kwh)/"
        "(healthy_evaporator_heating_kwh+(D_T_hat_minutes/60)*healthy_evaporator_power_kw)",
    ),
    "cop_e": MetricDefinition(
        "v2.7.0",
        "max",
        "evaporator/air-source cycle COP",
        "water-side inferred air-source heat divided by total electrical energy",
        "Evaporator-side cycle COP",
        "(evaporator_heating_kwh+Q_T_hat_kwh-E_comp_T_hat_kwh)/(heating_electricity_kwh+E_T_hat_kwh)",
    ),
    "epsilon_hl": MetricDefinition(
        "v2.7.1",
        "min",
        "Tan normalized heating-loss coefficient",
        "signed heating loss relative to a cross-fitted dynamic healthy water-side reference",
        "Normalized heating-loss coefficient",
        "(healthy_water_heating_kwh-water_heating_kwh+L_T_dynamic_hat_kwh)/"
        "(healthy_water_heating_kwh+Q_T_hat_kwh+L_T_dynamic_hat_kwh)",
    ),
    "epsilon_hl_t0_proxy": MetricDefinition(
        "v2.7.1",
        "min",
        "Tan normalized heating-loss coefficient",
        "project sensitivity using the earliest stable 120-s raw water-side capacity as t0 proxy",
        "Heating-loss coefficient (t0 proxy)",
        "(t0_proxy_healthy_heating_kwh-water_heating_kwh+L_T_t0_hat_kwh)/"
        "(t0_proxy_healthy_heating_kwh+Q_T_hat_kwh+L_T_t0_hat_kwh)",
    ),
    "cop_cyc_k": MetricDefinition(
        "v2.7.2",
        "max",
        "Klingebiel heating-plus-defrost cycle COP",
        "leading recovery plus heating and prep/defrost outcome; "
        "only post-candidate future recovery excluded",
        "Heating + defrost cycle COP",
        "(rr_water_heating_kwh+Q_PD_hat_kwh)/(rr_heating_electricity_kwh+E_PD_hat_kwh)",
    ),
    "epsilon_hl_2a": MetricDefinition(
        "v2.7.3",
        "min",
        "Tan double-cycle terminal-loss interpolation",
        "project within-experiment leave-cycle-out 5%/35% two-anchor heating-loss interpolation",
        "Two-anchor heating-loss coefficient",
        "(t0_proxy_healthy_heating_kwh-water_heating_kwh+L_T_two_anchor_hat_kwh)/"
        "(t0_proxy_healthy_heating_kwh+Q_T_hat_kwh+L_T_two_anchor_hat_kwh)",
    ),
}

FINAL_MODEL_REQUIREMENTS = {
    "cop_cyc_evt": ("E_T", "Q_T"),
    "eta_h_cyc": ("Q_T", "Qw0", "D_T"),
    "eta_e_cyc": ("Q_T", "Qw0", "Pcomp0", "E_comp_T", "D_T"),
}


def _required_model_support(
    metric_id: str, supports: dict[str, pd.Series | np.ndarray]
) -> np.ndarray:
    """Combine only the prediction supports used by one final objective."""
    return np.logical_and.reduce(
        [np.asarray(supports[name], dtype=bool) for name in FINAL_MODEL_REQUIREMENTS[metric_id]]
    )


def _metric_measurement_support(frame: pd.DataFrame, metric_id: str) -> pd.Series:
    """Return the raw-integral gate required by one final objective."""
    water = frame["water_heating_measurement_eligible"].fillna(False)
    if metric_id == "eta_h_cyc":
        return water
    if metric_id == "eta_e_cyc":
        return water & frame["heating_compressor_measurement_eligible"].fillna(False)
    if metric_id == "cop_cyc_evt":
        return water & frame["heating_electricity_measurement_eligible"].fillna(False)
    raise ValueError(f"no native measurement support for {metric_id}")




def earliest_t0_proxy(
    frame: pd.DataFrame,
    heating_start: pd.Timestamp,
    *,
    duration_seconds: int = 120,
    tolerance: float = 0.02,
) -> dict[str, object]:
    """Find the earliest raw 120-s water-capacity window stable within ±2%."""
    values = frame.copy()
    values["timestamp"] = pd.to_datetime(values["timestamp"], errors="coerce")
    values["water_heat_kw"] = water_side_heating_kw(values)
    values = (
        values.dropna(subset=["timestamp"])
        .sort_values("timestamp", kind="stable")
        .drop_duplicates("timestamp")
    )
    starts = values.loc[values["timestamp"].ge(heating_start), "timestamp"]
    for start in starts:
        end = pd.Timestamp(start) + pd.Timedelta(seconds=duration_seconds)
        window = values.loc[
            values["timestamp"].ge(start) & values["timestamp"].lt(end),
            ["timestamp", "water_heat_kw"],
        ].dropna()
        if window.empty:
            continue
        gaps = window["timestamp"].diff().dt.total_seconds().dropna()
        valid_gaps = gaps.gt(0) & gaps.le(30)
        cadence = _robust_observation_cadence_seconds(window["timestamp"])
        trailing_seconds = max(
            (end - pd.Timestamp(window["timestamp"].iloc[-1])).total_seconds(), 0.0
        )
        hold_seconds = (
            min(trailing_seconds, cadence)
            if np.isfinite(cadence) and cadence > 0
            else 0.0
        )
        covered = float(gaps.where(valid_gaps, 0.0).sum() + hold_seconds)
        coverage = covered / duration_seconds
        maximum_gap = (
            float(max(gaps.max(), trailing_seconds))
            if not gaps.empty
            else float(trailing_seconds)
        )
        fresh = (
            abs((window["timestamp"].iloc[0] - start).total_seconds()) <= 30
            and abs((end - window["timestamp"].iloc[-1]).total_seconds()) <= 30
        )
        median = float(window["water_heat_kw"].median())
        scale = abs(median)
        stable = scale > 0 and window["water_heat_kw"].sub(median).abs().le(tolerance * scale).all()
        if coverage >= 0.95 and maximum_gap <= 30 and fresh and stable:
            return {
                "valid": True,
                "start": pd.Timestamp(start),
                "end": end,
                "water_heat_kw": median,
                "coverage": min(coverage, 1.0),
                "maximum_gap_seconds": maximum_gap,
            }
    return {
        "valid": False,
        "start": pd.NaT,
        "end": pd.NaT,
        "water_heat_kw": np.nan,
        "coverage": 0.0,
        "maximum_gap_seconds": np.nan,
    }


def t0_proxy_reference_kwh(
    candidate_time: pd.Timestamp,
    proxy_start: pd.Timestamp,
    proxy_end: pd.Timestamp,
    water_heat_kw: float,
) -> float:
    """Return the proxy healthy heat from the proxy start to an identifiable candidate.

    The stability window is only a pre-action reference.  A candidate before its
    end would use information from the future, so it is deliberately returned as
    unavailable instead of being extrapolated.
    """
    candidate = pd.Timestamp(candidate_time)
    start = pd.Timestamp(proxy_start)
    end = pd.Timestamp(proxy_end)
    if (
        pd.isna(candidate)
        or pd.isna(start)
        or pd.isna(end)
        or candidate < end
        or not np.isfinite(water_heat_kw)
    ):
        return np.nan
    return float(water_heat_kw * (candidate - start).total_seconds() / 3600.0)


def select_two_anchors(
    events: pd.DataFrame,
    target_cycle: str,
    experiment_id: str,
) -> dict[str, object]:
    """Select distinct within-experiment events nearest 5% and 35% attenuation."""
    # Bootstrap model rows use ``draw_###`` as their fitting group so repeated
    # experiments receive separate sample weights.  That identifier is not the
    # physical experiment from which an anchor was observed.  Keep the source
    # identifier as the selection key whenever it is present; otherwise retain
    # the historical ``experiment_id`` behaviour for the main (non-bootstrap)
    # table.
    source_experiment = (
        events["source_experiment_id"]
        if "source_experiment_id" in events.columns
        else events["experiment_id"]
    )
    source_experiment = source_experiment.where(
        source_experiment.notna(), events["experiment_id"]
    )
    candidates = events.loc[
        source_experiment.astype(str).eq(str(experiment_id))
        & events["cycle_name"].astype(str).ne(str(target_cycle))
        & events["event_valid"].fillna(False)
    ].copy()
    # A with-replacement anchor draw can contain the same sibling more than
    # once.  Such rows are one physical sibling, not two independent anchors.
    # Deduplicating here makes the ``two different siblings`` requirement
    # explicit and prevents one cycle from supplying both interpolation ends.
    candidates = candidates.drop_duplicates(subset=["cycle_name"], keep="first")
    numeric = candidates[["attenuation_fraction", "heating_elapsed_minutes", "L_T_t0_kwh"]].apply(
        pd.to_numeric, errors="coerce"
    )
    candidates = candidates.loc[numeric.notna().all(axis=1)].copy()
    if len(candidates) < 2:
        return _invalid_anchors("fewer_than_two_sibling_events")
    mild_index = (candidates["attenuation_fraction"] - 0.05).abs().idxmin()
    severe_pool = candidates.drop(index=mild_index)
    severe_index = (severe_pool["attenuation_fraction"] - 0.35).abs().idxmin()
    mild = candidates.loc[mild_index]
    severe = candidates.loc[severe_index]
    t_mild = float(mild["heating_elapsed_minutes"])
    t_severe = float(severe["heating_elapsed_minutes"])
    if not t_mild < t_severe:
        return _invalid_anchors("anchor_times_not_ordered")
    return {
        "valid": True,
        "reason": "",
        "anchor_5_cycle": str(mild["cycle_name"]),
        "anchor_35_cycle": str(severe["cycle_name"]),
        "anchor_5_attenuation": float(mild["attenuation_fraction"]),
        "anchor_35_attenuation": float(severe["attenuation_fraction"]),
        "anchor_5_deviation": abs(float(mild["attenuation_fraction"]) - 0.05),
        "anchor_35_deviation": abs(float(severe["attenuation_fraction"]) - 0.35),
        "anchor_5_time_minutes": t_mild,
        "anchor_35_time_minutes": t_severe,
        "anchor_5_loss_kwh": float(mild["L_T_t0_kwh"]),
        "anchor_35_loss_kwh": float(severe["L_T_t0_kwh"]),
    }


def _invalid_anchors(reason: str) -> dict[str, object]:
    return {
        "valid": False,
        "reason": reason,
        "anchor_5_cycle": "",
        "anchor_35_cycle": "",
        "anchor_5_attenuation": np.nan,
        "anchor_35_attenuation": np.nan,
        "anchor_5_deviation": np.nan,
        "anchor_35_deviation": np.nan,
        "anchor_5_time_minutes": np.nan,
        "anchor_35_time_minutes": np.nan,
        "anchor_5_loss_kwh": np.nan,
        "anchor_35_loss_kwh": np.nan,
    }


def project_two_anchor_loss(time_minutes: float, anchors: dict[str, object]) -> float:
    if not bool(anchors.get("valid")):
        return np.nan
    t1 = float(anchors["anchor_5_time_minutes"])
    t2 = float(anchors["anchor_35_time_minutes"])
    loss1 = float(anchors["anchor_5_loss_kwh"])
    loss2 = float(anchors["anchor_35_loss_kwh"])
    return float(loss2 + (time_minutes - t2) * (loss1 - loss2) / (t1 - t2))


def finalize_metric_curve(
    curve: pd.DataFrame,
    direction: str,
    measurement_eligible: pd.Series,
) -> pd.DataFrame:
    """Preserve the full curve and select an extreme only on long valid support."""
    if direction not in {"min", "max"}:
        raise ValueError("direction must be 'min' or 'max'")
    result = curve.copy()
    result = result.drop(
        columns=[
            "_metric_measurement_eligible",
            "continuous_support",
            "optimization_eligible",
            "optimization_direction",
            "diagnostic_extreme",
            "t_star",
            "relative_optimality_gap",
            "near_optimal_1pct",
            "near_optimal_2pct",
            "near_optimal_5pct",
            "extreme_location",
            "basin_1pct_start",
            "basin_1pct_end",
            "basin_1pct_width_minutes",
            "basin_2pct_start",
            "basin_2pct_end",
            "basin_2pct_width_minutes",
            "basin_5pct_start",
            "basin_5pct_end",
            "basin_5pct_width_minutes",
            "valid",
            "t_star_model_supported",
        ],
        errors="ignore",
    )
    defaults = pd.DataFrame(
        {
            "_metric_measurement_eligible": measurement_eligible.to_numpy(),
            "continuous_support": False,
            "optimization_eligible": False,
            "optimization_direction": direction,
            "diagnostic_extreme": pd.NaT,
            "t_star": pd.NaT,
            "relative_optimality_gap": np.nan,
            "near_optimal_1pct": False,
            "near_optimal_2pct": False,
            "near_optimal_5pct": False,
            "extreme_location": "unidentified",
            "basin_1pct_start": pd.NaT,
            "basin_1pct_end": pd.NaT,
            "basin_1pct_width_minutes": np.nan,
            "basin_2pct_start": pd.NaT,
            "basin_2pct_end": pd.NaT,
            "basin_2pct_width_minutes": np.nan,
            "basin_5pct_start": pd.NaT,
            "basin_5pct_end": pd.NaT,
            "basin_5pct_width_minutes": np.nan,
            "valid": False,
            "t_star_model_supported": False,
        },
        index=result.index,
    )
    result = pd.concat([result, defaults], axis=1)
    result = result.sort_values("candidate_time", kind="stable").reset_index(drop=True)
    objective = pd.to_numeric(result["objective_value"], errors="coerce")
    metric_measurement = result.pop("_metric_measurement_eligible").fillna(False)
    base = (
        result["supported"].fillna(False)
        & result["pre_action_window_valid"].fillna(False)
        & metric_measurement
        & result["physical_valid"].fillna(False)
        & result.get("identifiable", pd.Series(True, index=result.index)).fillna(False)
        & objective.notna()
    )
    result["continuous_support"] = _long_support_runs(result["candidate_time"], base)
    eligible = base & result["continuous_support"]
    result["optimization_eligible"] = eligible
    if eligible.any():
        extreme = (
            objective.loc[eligible].min() if direction == "min" else objective.loc[eligible].max()
        )
        optimum_index = int(result.index[eligible & objective.eq(extreme)][0])
        optimum_time = pd.Timestamp(result.loc[optimum_index, "candidate_time"])
        result[["diagnostic_extreme", "t_star"]] = optimum_time
        scale = abs(float(extreme)) or max(float(objective.loc[eligible].abs().max()), 1.0)
        gap = (objective - extreme) / scale if direction == "min" else (extreme - objective) / scale
        result["relative_optimality_gap"] = gap.where(eligible)
        eligible_indices = result.index[eligible]
        result["extreme_location"] = (
            "left_boundary"
            if optimum_index == int(eligible_indices.min())
            else "right_boundary"
            if optimum_index == int(eligible_indices.max())
            else "interior"
        )
        for percent in (1, 2, 5):
            near = eligible & gap.le(percent / 100)
            result[f"near_optimal_{percent}pct"] = near
            left = right = optimum_index
            while left > 0 and bool(near.iloc[left - 1]):
                left -= 1
            while right + 1 < len(near) and bool(near.iloc[right + 1]):
                right += 1
            start = pd.Timestamp(result.loc[left, "candidate_time"])
            end = pd.Timestamp(result.loc[right, "candidate_time"])
            result[f"basin_{percent}pct_start"] = start
            result[f"basin_{percent}pct_end"] = end
            result[f"basin_{percent}pct_width_minutes"] = (end - start).total_seconds() / 60
    result["valid"] = eligible
    result["t_star_model_supported"] = result["t_star"].notna()
    return result


def _healthy_reference_samples(
    loader: object,
    catalog: pd.DataFrame,
    cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, record in catalog.iterrows():
        heating_start = pd.to_datetime(record.get("heating_start"), errors="coerce")
        if pd.isna(heating_start):
            continue
        stable = pd.Timestamp(heating_start) + pd.Timedelta(minutes=9)
        frame = _load_frame(loader, str(record["cycle_name"]), cache)
        window = _sorted_time_slice(
            frame,
            stable,
            stable + pd.Timedelta(minutes=5),
            end_inclusive=False,
        ).copy()
        if window.empty:
            continue
        window["water_heat_kw"] = water_side_heating_kw(window)
        window["minute"] = ((window["timestamp"] - stable).dt.total_seconds() // 60).astype(int)
        for minute, values in window.groupby("minute", sort=True):
            if not 0 <= minute < 5:
                continue
            columns = [*HEALTHY_FEATURES, "water_heat_kw", "compressor_power"]
            numeric = values[columns].apply(pd.to_numeric, errors="coerce")
            if numeric.notna().sum().min() < 48:
                continue
            medians = numeric.median()
            rows.append(
                {
                    "cycle_name": str(record["cycle_name"]),
                    "experiment_id": str(record["experiment_id"]),
                    "sample_time": stable + pd.Timedelta(minutes=int(minute), seconds=30),
                    **{feature: float(medians[feature]) for feature in HEALTHY_FEATURES},
                    "healthy_water_heat_kw": float(medians["water_heat_kw"]),
                    "healthy_compressor_power_kw": float(medians["compressor_power"]),
                }
            )
    return pd.DataFrame(rows)


def _preaction_water_heat(frame: pd.DataFrame, action: pd.Timestamp) -> float:
    values = _sorted_time_slice(
        frame,
        action - pd.Timedelta(seconds=60),
        action,
        end_inclusive=False,
    ).copy()
    return float(water_side_heating_kw(values).median()) if not values.empty else np.nan


def _augment_event_references(
    events: pd.DataFrame,
    loader: object,
    cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    result = events.copy()
    result["E_comp_T_observed_kwh"] = pd.to_numeric(
        result.get("E_comp_T_kwh"), errors="coerce"
    ).where(result.get("compressor_event_valid", False))
    result["event_duration_observed_minutes"] = pd.to_numeric(
        result.get("event_duration_minutes"), errors="coerce"
    )
    result["t0_proxy_valid"] = False
    result["t0_proxy_start"] = pd.NaT
    result["t0_proxy_end"] = pd.NaT
    for column in (
        "t0_proxy_water_heat_kw",
        "preaction_water_heat_kw",
        "attenuation_fraction",
        "L_T_t0_kwh",
    ):
        result[column] = np.nan
    for index, event in result.loc[result["event_valid"].fillna(False)].iterrows():
        heating_start = pd.to_datetime(event.get("heating_start"), errors="coerce")
        action = pd.to_datetime(event.get("defrost_preparation_start"), errors="coerce")
        if pd.isna(heating_start) or pd.isna(action):
            continue
        frame = _load_frame(loader, str(event["cycle_name"]), cache)
        proxy = earliest_t0_proxy(frame, pd.Timestamp(heating_start))
        preaction = _preaction_water_heat(frame, pd.Timestamp(action))
        result.loc[index, "t0_proxy_valid"] = bool(proxy["valid"])
        result.loc[index, "t0_proxy_start"] = proxy["start"]
        result.loc[index, "t0_proxy_end"] = proxy["end"]
        result.loc[index, "t0_proxy_water_heat_kw"] = proxy["water_heat_kw"]
        result.loc[index, "preaction_water_heat_kw"] = preaction
        if bool(proxy["valid"]) and float(proxy["water_heat_kw"]) != 0:
            result.loc[index, "attenuation_fraction"] = 1 - preaction / float(
                proxy["water_heat_kw"]
            )
            duration = float(result.loc[index, "event_duration_observed_minutes"])
            result.loc[index, "L_T_t0_kwh"] = float(proxy["water_heat_kw"]) * duration / 60 - float(
                result.loc[index, "Q_T_observed_kwh"]
            )
    return result


def _predict(
    model: RidgeOutcomeModel, frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = model.predict(frame)
    supported, distance = _prediction_support(model, frame)
    return prediction, supported, distance


def _model_audit_columns(
    models: dict[str, RidgeOutcomeModel],
) -> dict[str, object]:
    """Expose fit provenance beside every V2.7 prediction column."""
    columns: dict[str, object] = {}
    provenance: dict[str, object] = {}
    for name, model in models.items():
        columns[f"{name}_alpha"] = model.alpha
        columns[f"{name}_support_threshold"] = model.support_threshold
        columns[f"{name}_model_training_experiment_ids"] = ",".join(
            sorted(set(model.training_groups.astype(str)))
        )
        serialized = _model_provenance(model)
        columns[f"{name}_model_provenance"] = serialized
        provenance[name] = json.loads(serialized)
    columns["v27_model_provenance"] = json.dumps(provenance, sort_keys=True)
    return columns


def _cumulative_reference(
    candidate_time: pd.Series,
    values_kw: pd.Series,
    start: pd.Timestamp,
) -> np.ndarray:
    times = pd.to_datetime(candidate_time, errors="coerce")
    values = pd.to_numeric(values_kw, errors="coerce").to_numpy(dtype=float)
    elapsed_hours = (times - start).dt.total_seconds().to_numpy(dtype=float) / 3600
    result = np.full(len(values), np.nan)
    total = 0.0
    previous_time = 0.0
    previous_value = np.nan
    for index, (time, value) in enumerate(zip(elapsed_hours, values, strict=True)):
        if not np.isfinite([time, value]).all() or time < previous_time:
            continue
        width = time - previous_time
        total += width * (
            value if not np.isfinite(previous_value) else (previous_value + value) / 2
        )
        result[index] = total
        previous_time = time
        previous_value = value
    return result


def _metric_copy(
    common: pd.DataFrame,
    metric_id: str,
    objective: pd.Series,
    supported: pd.Series,
    physical: pd.Series,
    identifiable: pd.Series | bool = True,
    *,
    measurement_eligible: pd.Series,
) -> pd.DataFrame:
    definition = METRICS[metric_id]
    result = common.copy()
    # The shared V2.6.8 table contains its own inverse-COP optimum fields.  Do
    # not leave those names in a V2.7 metric table where a reader could mistake
    # them for this metric's native extreme; preserve them only as provenance.
    for column in (
        "diagnostic_minimum",
        "raw_t_star",
        "relative_regret",
        "J",
        "cycle_cop",
        "inverse_cop",
    ):
        if column in result:
            result[f"legacy_v268_{column}"] = result.pop(column)
    identified = (
        pd.Series(bool(identifiable), index=result.index)
        if isinstance(identifiable, (bool, np.bool_))
        else identifiable.fillna(False).astype(bool)
    )
    objective = pd.to_numeric(objective, errors="coerce")
    supported = supported.fillna(False).astype(bool)
    metadata = pd.DataFrame(
        {
            "algorithm": definition.algorithm,
            "metric_id": metric_id,
            "source_idea": definition.source_idea,
            "project_definition": definition.project_definition,
            "formula": definition.formula,
            "optimization_direction": definition.direction,
            "objective_label": definition.label,
            "objective_unit": definition.unit,
            "objective_value": objective,
            "display_only_objective": objective,
            "supported": supported,
            "model_supported": supported,
            "physical_valid": physical.fillna(False).astype(bool),
            "identifiable": identified,
        },
        index=result.index,
    )
    result = result.drop(columns=metadata.columns, errors="ignore")
    result = pd.concat([result, metadata], axis=1)
    pieces = [
        finalize_metric_curve(
            group,
            definition.direction,
            measurement_eligible.loc[group.index],
        )
        for _, group in result.groupby("cycle_name", sort=False)
    ]
    result = pd.concat(pieces, ignore_index=True, sort=False)
    result["display_only_objective"] = pd.to_numeric(
        result["objective_value"], errors="coerce"
    ).where(~result["optimization_eligible"].fillna(False))
    result["cycle_status"] = np.where(
        result["t_star"].notna(), "identified_curve", "support_or_identifiability_limited"
    )
    result["decision_status"] = "diagnostic_observational_v27"
    result["t_star_semantics"] = "historical_policy_model_implied_extreme"
    result["recommended_time"] = pd.NaT
    result["hard_label_eligible"] = False
    return result


def _validation_rows(
    events: pd.DataFrame,
    healthy_folds: dict[str, dict[str, RidgeOutcomeModel]],
    event_folds: dict[str, dict[str, RidgeOutcomeModel]],
    dynamic_loss_folds: dict[str, RidgeOutcomeModel],
    v268_validation: pd.DataFrame,
) -> pd.DataFrame:
    # V2.6.8 already owns the four cross-fitted E_T/Q_T rows.  Keep those rows
    # intact and only join the additional V2.7 event/reference outcomes once per
    # event; fitting another copy of the four base models would change the audit.
    extras: list[dict[str, object]] = []
    for _, event in events.loc[events["event_valid"].fillna(False)].iterrows():
        experiment = str(event["experiment_id"])
        frame = pd.DataFrame([event])
        q0_model = healthy_folds[experiment]["healthy_water_heat_kw"]
        p0_model = healthy_folds[experiment]["healthy_compressor_power_kw"]
        q0 = float(q0_model.predict(frame)[0])
        p0 = float(p0_model.predict(frame)[0])
        q0_supported, q0_distance = _prediction_support(q0_model, frame)
        p0_supported, p0_distance = _prediction_support(p0_model, frame)
        duration = float(event["event_duration_observed_minutes"])
        l_dynamic_observed = q0 * duration / 60 - float(event["Q_T_observed_kwh"])
        e_comp_model = event_folds[experiment]["E_comp_T_observed_kwh"]
        duration_model = event_folds[experiment]["event_duration_observed_minutes"]
        l_t0_model = event_folds[experiment]["L_T_t0_kwh"]
        dynamic_loss_model = dynamic_loss_folds[experiment]
        e_comp = float(e_comp_model.predict(frame)[0])
        duration_prediction = float(duration_model.predict(frame)[0])
        dynamic_loss_prediction = float(dynamic_loss_model.predict(frame)[0])
        t0_loss_prediction = float(l_t0_model.predict(frame)[0])
        e_comp_supported, e_comp_distance = _prediction_support(e_comp_model, frame)
        duration_supported, duration_distance = _prediction_support(duration_model, frame)
        dynamic_supported, dynamic_distance = _prediction_support(dynamic_loss_model, frame)
        t0_supported, t0_distance = _prediction_support(l_t0_model, frame)
        row: dict[str, object] = {
            "cycle_name": str(event["cycle_name"]),
            "E_comp_T_observed_kwh": float(event["E_comp_T_observed_kwh"]),
            "event_duration_observed_minutes": duration,
            "L_T_dynamic_observed_kwh": l_dynamic_observed,
            # Kept as a compatibility column above, but this target is a
            # cross-fitted reference residual, not an independent measurement.
            "L_T_dynamic_reference_derived_kwh": l_dynamic_observed,
            "L_T_dynamic_observed_provenance": DYNAMIC_LOSS_TARGET_PROVENANCE,
            "L_T_dynamic_observed_is_direct_measurement": False,
            "L_T_t0_observed_kwh": float(event["L_T_t0_kwh"]),
            "v27_healthy_water_heat_prediction_kw": q0,
            "v27_healthy_compressor_power_prediction_kw": p0,
            "v27_healthy_water_heat_supported": bool(q0_supported[0]),
            "v27_healthy_compressor_power_supported": bool(p0_supported[0]),
            "v27_healthy_water_heat_support_distance": float(q0_distance[0]),
            "v27_healthy_compressor_power_support_distance": float(p0_distance[0]),
            "E_comp_T_prediction_kwh": e_comp,
            "event_duration_prediction_minutes": duration_prediction,
            "L_T_dynamic_prediction_kwh": dynamic_loss_prediction,
            "L_T_t0_prediction_kwh": t0_loss_prediction,
            "v27_E_comp_T_supported": bool(e_comp_supported[0]),
            "v27_event_duration_supported": bool(duration_supported[0]),
            "v27_L_T_dynamic_supported": bool(dynamic_supported[0]),
            "v27_L_T_t0_supported": bool(t0_supported[0]),
            "v27_E_comp_T_support_distance": float(e_comp_distance[0]),
            "v27_event_duration_support_distance": float(duration_distance[0]),
            "v27_L_T_dynamic_support_distance": float(dynamic_distance[0]),
            "v27_L_T_t0_support_distance": float(t0_distance[0]),
            "v27_model_training_experiment_ids": ",".join(
                sorted(set(e_comp_model.training_groups.astype(str)))
            ),
        }
        row.update(
            _model_audit_columns(
                {
                    "healthy_water_heat": q0_model,
                    "healthy_compressor_power": p0_model,
                    "E_comp_T": e_comp_model,
                    "D_T": duration_model,
                    "L_dynamic": dynamic_loss_model,
                    "L_t0": l_t0_model,
                }
            )
        )
        row["event_duration_alpha"] = row["D_T_alpha"]
        row["event_duration_support_threshold"] = row["D_T_support_threshold"]
        row["event_duration_model_training_experiment_ids"] = row[
            "D_T_model_training_experiment_ids"
        ]
        row["event_duration_model_provenance"] = row["D_T_model_provenance"]
        for observed_value, predicted, residual in (
            (
                float(event["E_comp_T_observed_kwh"]),
                e_comp,
                "E_comp_T_residual_kwh",
            ),
            (
                float(event["event_duration_observed_minutes"]),
                duration_prediction,
                "event_duration_residual_minutes",
            ),
            (
                l_dynamic_observed,
                dynamic_loss_prediction,
                "L_T_dynamic_residual_kwh",
            ),
            (
                float(event["L_T_t0_kwh"]),
                t0_loss_prediction,
                "L_T_t0_residual_kwh",
            ),
        ):
            row[residual] = observed_value - float(predicted)
        extras.append(row)
    if not extras:
        return v268_validation.copy()
    extra_frame = pd.DataFrame(extras)
    validation = v268_validation.merge(
        extra_frame, on="cycle_name", how="left", validate="many_to_one"
    )
    # J_w columns are generated by the shared V2.6.8 validation table.  These
    # row-level summaries make experiment-macro error and calibration plots
    # computable without refitting or a second validation table.
    j_prediction = pd.to_numeric(validation.get("J_w_prediction"), errors="coerce")
    j_observed = pd.to_numeric(validation.get("J_w_observed"), errors="coerce")
    validation["J_w_evaluable"] = j_observed.notna() & j_prediction.notna()
    validation["J_w_abs_error"] = (j_observed - j_prediction).abs().where(
        validation["J_w_evaluable"]
    )
    validation["J_w_squared_error"] = (j_observed - j_prediction).pow(2).where(
        validation["J_w_evaluable"]
    )
    validation["J_w_calibration_observed"] = j_observed.where(validation["J_w_evaluable"])
    validation["J_w_calibration_prediction"] = j_prediction.where(validation["J_w_evaluable"])
    return validation


def _identifiability_table(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "audit_id": "strict_tan_qhc1",
            "available": False,
            "reason": "no condition-matched non-frosting instantaneous-capacity reference",
        },
        {
            "audit_id": "wang_nominal_capacity",
            "available": False,
            "reason": "no verified nominal heating capacity q_nc",
        },
        {
            "audit_id": "da_silva_frost_mass_efficiency",
            "available": False,
            "reason": "frost mass and defrost efficiency are unavailable",
        },
    ]
    anchor = curves["v2.7.3"]
    for cycle_name, curve in anchor.groupby("cycle_name", sort=False):
        first = curve.iloc[0]
        rows.append(
            {
                "audit_id": "project_two_anchor",
                "cycle_name": cycle_name,
                "experiment_id": first["experiment_id"],
                "available": bool(first.get("anchor_valid", False)),
                "reason": first.get("anchor_reason", ""),
                "anchor_5_cycle": first.get("anchor_5_cycle", ""),
                "anchor_35_cycle": first.get("anchor_35_cycle", ""),
                "anchor_5_attenuation": first.get("anchor_5_attenuation", np.nan),
                "anchor_35_attenuation": first.get("anchor_35_attenuation", np.nan),
            }
        )
    return pd.DataFrame(rows)


def build_v27_tables(
    points: pd.DataFrame,
    loader: object,
    *,
    bootstrap_replicates: int = 200,
    bootstrap_trajectory: Path | None = None,
    bootstrap_final_only: bool = False,
    n_jobs: int = 1,
    candidate_step_seconds: int = 60,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Build all V2.7 project metrics on one shared V2.6.8 candidate/model pass."""
    base, v268_artifacts = build_v268_table(
        points,
        loader,
        bootstrap_replicates=0,
        candidate_step_seconds=candidate_step_seconds,
    )
    events = v268_artifacts["events"]
    catalog = _catalog(loader)
    cache: dict[str, pd.DataFrame] = {}
    events = _augment_event_references(events, loader, cache)
    valid_events = events.loc[events["event_valid"].fillna(False)].copy()
    experiments = sorted(
        set(base["experiment_id"].dropna().astype(str))
        | set(valid_events["experiment_id"].dropna().astype(str))
    )
    healthy_samples = _healthy_reference_samples(loader, catalog, cache)
    healthy_folds = _fit_fold_set(
        healthy_samples,
        HEALTHY_FEATURES,
        ("healthy_water_heat_kw", "healthy_compressor_power_kw"),
        experiments,
    )
    event_folds = _fit_fold_set(
        valid_events,
        DYNAMIC_8,
        (
            "E_comp_T_observed_kwh",
            "event_duration_observed_minutes",
            "L_T_t0_kwh",
        ),
    )
    base_event_folds = _fit_fold_set(
        valid_events,
        DYNAMIC_8,
        (
            "E_T_observed_kwh",
            "Q_T_observed_kwh",
            "E_PD_kwh",
            "Q_PD_kwh",
        ),
    )
    dynamic_loss_folds: dict[str, RidgeOutcomeModel] = {}
    for experiment in experiments:
        training = valid_events.loc[
            ~valid_events["experiment_id"].astype(str).eq(experiment)
        ].copy()
        q0_model = healthy_folds[experiment]["healthy_water_heat_kw"]
        q0 = q0_model.predict(training)
        training["L_T_dynamic_kwh"] = q0 * training["event_duration_observed_minutes"].to_numpy(
            dtype=float
        ) / 60 - training["Q_T_observed_kwh"].to_numpy(dtype=float)
        dynamic_loss_folds[experiment] = fit_outcome_fold(
            training,
            experiment,
            DYNAMIC_8,
            "L_T_dynamic_kwh",
        )

    prepared: list[pd.DataFrame] = []
    for (cycle_name, experiment), source in base.groupby(
        ["cycle_name", "experiment_id"], sort=False
    ):
        curve = source.sort_values("candidate_time", kind="stable").reset_index(drop=True).copy()
        experiment = str(experiment)
        q0_model = healthy_folds[experiment]["healthy_water_heat_kw"]
        p0_model = healthy_folds[experiment]["healthy_compressor_power_kw"]
        e_comp_model = event_folds[experiment]["E_comp_T_observed_kwh"]
        duration_model = event_folds[experiment]["event_duration_observed_minutes"]
        l_t0_model = event_folds[experiment]["L_T_t0_kwh"]
        l_dynamic_model = dynamic_loss_folds[experiment]
        q0, q0_support, q0_distance = _predict(q0_model, curve)
        p0, p0_support, p0_distance = _predict(p0_model, curve)
        e_comp, e_comp_support, e_comp_distance = _predict(e_comp_model, curve)
        duration, duration_support, duration_distance = _predict(duration_model, curve)
        l_dynamic, l_dynamic_support, l_dynamic_distance = _predict(l_dynamic_model, curve)
        l_t0, l_t0_support, l_t0_distance = _predict(l_t0_model, curve)
        stable = pd.Timestamp(curve["stable_start_fixed9"].iloc[0])
        curve["healthy_water_heat_kw"] = q0
        curve["healthy_compressor_power_kw"] = p0
        curve["healthy_evaporator_power_kw"] = q0 - p0
        curve["instant_water_heat_kw"] = (
            1.161
            * pd.to_numeric(curve["water_flow"], errors="coerce")
            * (
                pd.to_numeric(curve["water_out_temperature"], errors="coerce")
                - pd.to_numeric(curve["water_in_temperature"], errors="coerce")
            )
        )
        curve["instant_evaporator_capacity_kw"] = curve["instant_water_heat_kw"] - pd.to_numeric(
            curve["compressor_power"], errors="coerce"
        )
        curve["instant_water_cop"] = curve["instant_water_heat_kw"] / pd.to_numeric(
            curve["power_total"], errors="coerce"
        )
        curve["instant_unit_cop"] = pd.to_numeric(
            curve["heating_capacity"], errors="coerce"
        ) / pd.to_numeric(curve["power_total"], errors="coerce")
        curve["heating_attenuation_fraction"] = (
            1 - curve["instant_water_heat_kw"] / curve["healthy_water_heat_kw"]
        )
        curve["healthy_water_heating_kwh"] = _cumulative_reference(
            curve["candidate_time"], curve["healthy_water_heat_kw"], stable
        )
        curve["healthy_compressor_electricity_kwh"] = _cumulative_reference(
            curve["candidate_time"], curve["healthy_compressor_power_kw"], stable
        )
        curve["healthy_evaporator_heating_kwh"] = (
            curve["healthy_water_heating_kwh"] - curve["healthy_compressor_electricity_kwh"]
        )
        curve["evaporator_heating_kwh"] = (
            curve["water_heating_kwh"] - curve["heating_compressor_electricity_kwh"]
        )
        curve["E_comp_T_hat_kwh"] = e_comp
        curve["D_T_hat_minutes"] = duration
        curve["L_T_dynamic_hat_kwh"] = l_dynamic
        curve["L_T_t0_hat_kwh"] = l_t0
        curve["healthy_water_heat_supported"] = q0_support
        curve["healthy_compressor_power_supported"] = p0_support
        curve["E_comp_T_supported"] = e_comp_support
        curve["D_T_supported"] = duration_support
        curve["healthy_supported"] = q0_support & p0_support
        curve["event_extension_supported"] = e_comp_support & duration_support
        curve["dynamic_loss_supported"] = l_dynamic_support
        curve["t0_loss_supported"] = l_t0_support
        curve["Qw0_support_distance"] = q0_distance
        curve["Pcomp0_support_distance"] = p0_distance
        curve["E_comp_support_distance"] = e_comp_distance
        curve["D_T_support_distance"] = duration_distance
        curve["L_dynamic_support_distance"] = l_dynamic_distance
        curve["L_t0_support_distance"] = l_t0_distance
        curve["cycle_elapsed_minutes"] = (
            pd.to_datetime(curve["candidate_time"]) - pd.Timestamp(curve["heating_start"].iloc[0])
        ).dt.total_seconds() / 60
        current = _load_frame(loader, str(cycle_name), cache)
        proxy = earliest_t0_proxy(current, pd.Timestamp(curve["heating_start"].iloc[0]))
        curve["t0_proxy_valid"] = bool(proxy["valid"])
        curve["t0_proxy_water_heat_kw"] = float(proxy["water_heat_kw"])
        curve["t0_proxy_start"] = proxy["start"]
        curve["t0_proxy_end"] = proxy["end"]
        candidate_times = pd.to_datetime(curve["candidate_time"], errors="coerce")
        curve["t0_proxy_candidate_valid"] = (
            bool(proxy["valid"])
            & candidate_times.ge(pd.Timestamp(proxy["end"]))
            & curve["pre_action_window_valid"].fillna(False)
        )
        curve["t0_proxy_identifiable"] = curve["t0_proxy_candidate_valid"]
        curve["t0_proxy_healthy_heating_kwh"] = [
            t0_proxy_reference_kwh(
                candidate,
                proxy["start"],
                proxy["end"],
                float(proxy["water_heat_kw"]),
            )
            if bool(valid)
            else np.nan
            for candidate, valid in zip(
                candidate_times, curve["t0_proxy_candidate_valid"], strict=True
            )
        ]
        audit_models = _model_audit_columns(
            {
                "healthy_water_heat": q0_model,
                "healthy_compressor_power": p0_model,
                "E_comp_T": e_comp_model,
                "D_T": duration_model,
                "L_dynamic": l_dynamic_model,
                "L_t0": l_t0_model,
            }
        )
        curve = curve.assign(**audit_models)
        for field in (
            "alpha",
            "support_threshold",
            "model_training_experiment_ids",
            "model_provenance",
        ):
            curve[f"event_duration_{field}"] = curve[f"D_T_{field}"]
            curve[f"L_T_dynamic_{field}"] = curve[f"L_dynamic_{field}"]
            curve[f"L_T_t0_{field}"] = curve[f"L_t0_{field}"]
        curve["model_training_experiment_ids"] = ",".join(
            sorted(set(e_comp_model.training_groups.astype(str)))
        )
        prepared.append(curve)
    common = pd.concat(prepared, ignore_index=True, sort=False)

    numerator_evap = (
        common["evaporator_heating_kwh"] + common["Q_T_hat_kwh"] - common["E_comp_T_hat_kwh"]
    )
    eta_denominator = common["healthy_evaporator_heating_kwh"] + (
        common["D_T_hat_minutes"] / 60 * common["healthy_evaporator_power_kw"]
    )
    cop_e_denominator = common["heating_electricity_kwh"] + common["E_T_hat_kwh"]
    eta = numerator_evap / eta_denominator
    cop_e = numerator_evap / cop_e_denominator
    native_supports = {
        "E_T": common["E_T_supported"],
        "Q_T": common["Q_T_supported"],
        "Qw0": common["healthy_water_heat_supported"],
        "Pcomp0": common["healthy_compressor_power_supported"],
        "E_comp_T": common["E_comp_T_supported"],
        "D_T": common["D_T_supported"],
    }
    support_h = _required_model_support("eta_h_cyc", native_supports)
    support_o = _required_model_support("eta_e_cyc", native_supports)
    support_270 = (
        common["supported"] & common["healthy_supported"] & common["event_extension_supported"]
    )
    v270 = pd.concat(
        [
            _metric_copy(
                common,
                "eta_e_cyc",
                eta,
                pd.Series(support_o, index=common.index),
                eta_denominator.gt(0.01) & numerator_evap.notna(),
                measurement_eligible=_metric_measurement_support(common, "eta_e_cyc"),
            ),
            _metric_copy(
                common,
                "cop_e",
                cop_e,
                support_270,
                cop_e_denominator.gt(0) & numerator_evap.notna(),
                measurement_eligible=common["measurement_eligible"],
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    dynamic_heating_loss = common["healthy_water_heating_kwh"] - common["water_heating_kwh"]
    dynamic_denominator = (
        common["healthy_water_heating_kwh"] + common["Q_T_hat_kwh"] + common["L_T_dynamic_hat_kwh"]
    )
    epsilon = (dynamic_heating_loss + common["L_T_dynamic_hat_kwh"]) / dynamic_denominator
    t0_heating_loss = common["t0_proxy_healthy_heating_kwh"] - common["water_heating_kwh"]
    t0_denominator = (
        common["t0_proxy_healthy_heating_kwh"] + common["Q_T_hat_kwh"] + common["L_T_t0_hat_kwh"]
    )
    epsilon_t0 = (t0_heating_loss + common["L_T_t0_hat_kwh"]) / t0_denominator
    v271 = pd.concat(
        [
            _metric_copy(
                common.assign(heating_loss_kwh=dynamic_heating_loss),
                "epsilon_hl",
                epsilon,
                common["supported"]
                & common["healthy_supported"]
                & common["dynamic_loss_supported"],
                dynamic_denominator.gt(0.01),
                measurement_eligible=common["measurement_eligible"],
            ),
            _metric_copy(
                common.assign(heating_loss_kwh=t0_heating_loss),
                "epsilon_hl_t0_proxy",
                epsilon_t0,
                common["supported"] & common["t0_loss_supported"],
                t0_denominator.gt(0.01),
                common["t0_proxy_candidate_valid"],
                measurement_eligible=common["measurement_eligible"],
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    cop_k = 1 / pd.to_numeric(common["J_rr_model"], errors="coerce")
    v272 = _metric_copy(
        common,
        "cop_cyc_k",
        cop_k,
        common["rr_supported"],
        common["rr_physical_valid"],
        measurement_eligible=common["rr_measurement_eligible"],
    )

    anchor_curves: list[pd.DataFrame] = []
    for (cycle_name, experiment), curve in common.groupby(
        ["cycle_name", "experiment_id"], sort=False
    ):
        anchors = select_two_anchors(valid_events, str(cycle_name), str(experiment))
        values = curve.copy()
        for key, value in anchors.items():
            values[f"anchor_{key.removeprefix('anchor_')}"] = value
        values["anchor_valid"] = bool(anchors["valid"])
        values["anchor_reason"] = anchors["reason"]
        values["L_T_two_anchor_hat_kwh"] = values["cycle_elapsed_minutes"].map(
            lambda time, selected=anchors: project_two_anchor_loss(float(time), selected)
        )
        inside = (
            values["cycle_elapsed_minutes"].between(
                float(anchors["anchor_5_time_minutes"]),
                float(anchors["anchor_35_time_minutes"]),
            )
            if bool(anchors["valid"])
            else pd.Series(False, index=values.index)
        )
        denominator = (
            values["t0_proxy_healthy_heating_kwh"]
            + values["Q_T_hat_kwh"]
            + values["L_T_two_anchor_hat_kwh"]
        )
        objective = (
            values["t0_proxy_healthy_heating_kwh"]
            - values["water_heating_kwh"]
            + values["L_T_two_anchor_hat_kwh"]
        ) / denominator
        anchor_curves.append(
            _metric_copy(
                values,
                "epsilon_hl_2a",
                objective,
                values["supported"],
                denominator.gt(0.01),
                inside & values["t0_proxy_candidate_valid"],
                measurement_eligible=values["measurement_eligible"],
            )
        )
    v273 = pd.concat(anchor_curves, ignore_index=True, sort=False)

    complete_heat = common["water_heating_kwh"] + common["Q_T_hat_kwh"]
    healthy_complete_heat = common["healthy_water_heating_kwh"] + (
        common["healthy_water_heat_kw"] * common["D_T_hat_minutes"] / 60
    )
    common = common.copy()
    common = common.assign(
        healthy_complete_heat_kwh=healthy_complete_heat,
        heating_loss_closed_kwh=healthy_complete_heat - complete_heat,
        epsilon_hl_closed=1 - complete_heat / healthy_complete_heat,
    )
    v274 = _metric_copy(
        common,
        "eta_h_cyc",
        complete_heat / healthy_complete_heat,
        pd.Series(support_h, index=common.index),
        healthy_complete_heat.gt(Q_MIN_KWH) & complete_heat.notna(),
        measurement_eligible=_metric_measurement_support(common, "eta_h_cyc"),
    )

    curves = {
        "v2.7.0": v270,
        "v2.7.1": v271,
        "v2.7.2": v272,
        "v2.7.3": v273,
        "v2.7.4": v274,
    }
    validation = _validation_rows(
        events,
        healthy_folds,
        event_folds,
        dynamic_loss_folds,
        v268_artifacts["validation"],
    )
    bootstrap = _bootstrap_refit_summary(
        common,
        curves,
        valid_events,
        healthy_samples,
        healthy_folds,
        event_folds,
        base_event_folds,
        dynamic_loss_folds,
        bootstrap_replicates,
        trajectory_path=bootstrap_trajectory,
        final_only=bootstrap_final_only,
        n_jobs=n_jobs,
    )
    for algorithm, table in curves.items():
        curves[algorithm] = table.merge(
            bootstrap.loc[bootstrap["algorithm"].eq(algorithm)],
            on=["cycle_name", "experiment_id", "algorithm", "metric_id"],
            how="left",
            validate="many_to_one",
        )
    return curves, {
        "validation": validation,
        "identifiability": _identifiability_table(curves),
        "bootstrap": bootstrap,
        "bootstrap_draws": bootstrap.attrs.get("draws", pd.DataFrame()),
        "healthy_samples": healthy_samples,
    }


def _resample_experiments(frame: pd.DataFrame, sampled: np.ndarray) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for draw, experiment in enumerate(sampled):
        part = frame.loc[frame["experiment_id"].astype(str).eq(str(experiment))].copy()
        if part.empty:
            continue
        # ``experiment_id`` is intentionally replaced below for Ridge fitting,
        # but anchor selection must retain the physical source identity.  This
        # column also lets tests/audits prove that the two bootstrap pools are
        # disjoint without relying on the synthetic draw labels.
        if "source_experiment_id" in part.columns:
            part["source_experiment_id"] = part["source_experiment_id"].where(
                part["source_experiment_id"].notna(), part["experiment_id"]
            )
        else:
            part["source_experiment_id"] = part["experiment_id"]
        part["experiment_id"] = f"draw_{draw:03d}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _resample_anchor_events(
    events: pd.DataFrame,
    heldout_experiment: str,
    target_cycle: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw held-out sibling events for one target cycle.

    Anchor rows are deliberately sampled from the held-out experiment, while
    all Ridge training rows are sampled from the complementary experiments in
    ``_bootstrap_refit_summary``.  The returned ``experiment_id`` values are
    synthetic draw groups, but ``source_experiment_id`` remains the held-out
    physical experiment so ``select_two_anchors`` can recover the intended
    within-experiment pool.
    """
    source_experiment = (
        events["source_experiment_id"]
        if "source_experiment_id" in events.columns
        else events["experiment_id"]
    )
    source_experiment = source_experiment.where(
        source_experiment.notna(), events["experiment_id"]
    )
    siblings = events.loc[
        source_experiment.astype(str).eq(str(heldout_experiment))
        & events["cycle_name"].astype(str).ne(str(target_cycle))
        & events["event_valid"].fillna(False)
    ].copy()
    # The event table is normally one row per cycle.  Deduplicate defensively
    # before drawing so ``n_distinct_siblings`` has its physical meaning.
    siblings = siblings.drop_duplicates(subset=["cycle_name"], keep="first")
    sibling_cycles = siblings["cycle_name"].astype(str).dropna().unique()
    if sibling_cycles.size < 2:
        columns = list(events.columns)
        if "source_experiment_id" not in columns:
            columns.append("source_experiment_id")
        return pd.DataFrame(columns=columns)
    sampled_cycles = rng.choice(sibling_cycles, size=sibling_cycles.size, replace=True)
    parts: list[pd.DataFrame] = []
    for draw, cycle_name in enumerate(sampled_cycles):
        part = siblings.loc[siblings["cycle_name"].astype(str).eq(str(cycle_name))].copy()
        if part.empty:
            continue
        if "source_experiment_id" in part.columns:
            part["source_experiment_id"] = part["source_experiment_id"].where(
                part["source_experiment_id"].notna(), part["experiment_id"]
            )
        else:
            part["source_experiment_id"] = part["experiment_id"]
        part["experiment_id"] = f"draw_{draw:03d}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _fit_bootstrap_model(
    training: pd.DataFrame,
    target: str,
    alpha: float,
    features: tuple[str, ...] = DYNAMIC_8,
) -> RidgeOutcomeModel | None:
    try:
        return fit_weighted_ridge(training, features, target, alpha=alpha)
    except ValueError:
        return None


def _bootstrap_prediction_frame(
    candidates: pd.DataFrame,
    models: dict[str, RidgeOutcomeModel | None],
) -> pd.DataFrame | None:
    """Predict one bootstrap replicate on the held-out candidate curves."""
    required = ("E_T", "Q_T", "E_comp_T", "D_T", "Qw0", "Pcomp0")
    if any(models[name] is None for name in required):
        return None
    result = candidates.copy()
    for name in required:
        model = models[name]
        assert model is not None
        result[name] = model.predict(result)
    for name in ("L_dynamic", "L_t0", "E_PD", "Q_PD"):
        model = models.get(name)
        result[name] = model.predict(result) if model is not None else np.nan

    integrated: list[pd.DataFrame] = []
    for _, curve in result.groupby("cycle_name", sort=False):
        values = curve.sort_values("candidate_time", kind="stable").copy()
        stable = pd.Timestamp(values["stable_start_fixed9"].iloc[0])
        values["Qw0_H"] = _cumulative_reference(values["candidate_time"], values["Qw0"], stable)
        values["Pcomp0_H"] = _cumulative_reference(
            values["candidate_time"], values["Pcomp0"], stable
        )
        integrated.append(values)
    return pd.concat(integrated, ignore_index=True, sort=False).sort_values(
        ["cycle_name", "candidate_time"], kind="stable"
    )


def _bootstrap_objectives(candidates: pd.DataFrame) -> dict[str, np.ndarray]:
    result = candidates
    qe_h = result["water_heating_kwh"] - result["heating_compressor_electricity_kwh"]
    numerator = qe_h + result["Q_T"] - result["E_comp_T"]
    qe0_h = result["Qw0_H"] - result["Pcomp0_H"]
    qe0 = result["Qw0"] - result["Pcomp0"]
    eta_denominator = qe0_h + result["D_T"] / 60 * qe0
    cop_e_denominator = result["heating_electricity_kwh"] + result["E_T"]
    complete_heat = result["water_heating_kwh"] + result["Q_T"]
    complete_electricity = result["heating_electricity_kwh"] + result["E_T"]
    healthy_complete_heat = result["Qw0_H"] + result["Qw0"] * result["D_T"] / 60
    dynamic_denominator = result["Qw0_H"] + result["Q_T"] + result["L_dynamic"]
    t0_denominator = result["t0_proxy_healthy_heating_kwh"] + result["Q_T"] + result["L_t0"]
    two_anchor_loss = result.get(
        "L_T_two_anchor_hat_kwh", pd.Series(np.nan, index=result.index)
    )
    two_anchor_denominator = (
        result["t0_proxy_healthy_heating_kwh"] + result["Q_T"] + two_anchor_loss
    )
    return {
        "cop_cyc_evt": np.where(
            (complete_electricity > 0) & (complete_heat > Q_MIN_KWH),
            complete_heat / complete_electricity,
            np.nan,
        ),
        "eta_h_cyc": np.where(
            healthy_complete_heat > Q_MIN_KWH,
            complete_heat / healthy_complete_heat,
            np.nan,
        ),
        "eta_e_cyc": np.where(eta_denominator > 0.01, numerator / eta_denominator, np.nan),
        "cop_e": np.where(cop_e_denominator > 0, numerator / cop_e_denominator, np.nan),
        "epsilon_hl": np.where(
            dynamic_denominator > 0.01,
            (result["Qw0_H"] - result["water_heating_kwh"] + result["L_dynamic"])
            / dynamic_denominator,
            np.nan,
        ),
        "epsilon_hl_t0_proxy": np.where(
            t0_denominator > 0.01,
            (result["t0_proxy_healthy_heating_kwh"] - result["water_heating_kwh"] + result["L_t0"])
            / t0_denominator,
            np.nan,
        ),
        "cop_cyc_k": np.where(
            (result["rr_heating_electricity_kwh"] + result["E_PD"] > 0)
            & (result["rr_water_heating_kwh"] + result["Q_PD"] > Q_MIN_KWH),
            (result["rr_water_heating_kwh"] + result["Q_PD"])
            / (result["rr_heating_electricity_kwh"] + result["E_PD"]),
            np.nan,
        ),
        "epsilon_hl_2a": np.where(
            two_anchor_denominator > 0.01,
            (
                result["t0_proxy_healthy_heating_kwh"]
                - result["water_heating_kwh"]
                + two_anchor_loss
            )
            / two_anchor_denominator,
            np.nan,
        ),
    }


def _bootstrap_component_supports(
    candidates: pd.DataFrame,
    models: dict[str, RidgeOutcomeModel | None],
) -> dict[str, np.ndarray]:
    result = candidates
    supports: dict[str, np.ndarray] = {}
    for name in (
        "E_T",
        "Q_T",
        "Qw0",
        "Pcomp0",
        "E_comp_T",
        "D_T",
        "L_dynamic",
        "L_t0",
        "E_PD",
        "Q_PD",
    ):
        model = models.get(name)
        if model is None:
            supports[name] = np.zeros(len(result), dtype=bool)
        else:
            supports[name], _ = _prediction_support(model, result)
    return supports


def _bootstrap_metric_status(
    candidates: pd.DataFrame,
    models: dict[str, RidgeOutcomeModel | None],
    supports: dict[str, np.ndarray] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Recompute support, physical validity, and identifiability per replicate."""
    result = candidates
    supports = supports or _bootstrap_component_supports(result, models)

    base_support = _required_model_support("cop_cyc_evt", supports)
    healthy_support = supports["Qw0"] & supports["Pcomp0"]
    event_support = supports["E_comp_T"] & supports["D_T"]
    dynamic_support = supports["L_dynamic"]
    t0_support = supports["L_t0"]
    rr_support = supports["E_PD"] & supports["Q_PD"]
    qe_h = result["water_heating_kwh"] - result["heating_compressor_electricity_kwh"]
    numerator = qe_h + result["Q_T"] - result["E_comp_T"]
    qe0_h = result["Qw0_H"] - result["Pcomp0_H"]
    qe0 = result["Qw0"] - result["Pcomp0"]
    eta_denominator = qe0_h + result["D_T"] / 60 * qe0
    cop_e_denominator = result["heating_electricity_kwh"] + result["E_T"]
    complete_heat = result["water_heating_kwh"] + result["Q_T"]
    complete_electricity = result["heating_electricity_kwh"] + result["E_T"]
    healthy_complete_heat = result["Qw0_H"] + result["Qw0"] * result["D_T"] / 60
    dynamic_denominator = result["Qw0_H"] + result["Q_T"] + result["L_dynamic"]
    t0_denominator = result["t0_proxy_healthy_heating_kwh"] + result["Q_T"] + result["L_t0"]
    rr_e_total = result["rr_heating_electricity_kwh"] + result["E_PD"]
    rr_q_total = result["rr_water_heating_kwh"] + result["Q_PD"]
    two_anchor_loss = result.get(
        "L_T_two_anchor_hat_kwh", pd.Series(np.nan, index=result.index)
    )
    two_anchor_denominator = (
        result["t0_proxy_healthy_heating_kwh"] + result["Q_T"] + two_anchor_loss
    )
    t0_identifiable = result.get(
        "t0_proxy_candidate_valid",
        result.get("t0_proxy_valid", pd.Series(False, index=result.index)),
    ).fillna(False).to_numpy(dtype=bool)
    anchor_identifiable = result.get(
        "anchor_identifiable", pd.Series(False, index=result.index)
    ).fillna(False).to_numpy(dtype=bool)
    def finite(values: pd.Series) -> np.ndarray:
        return np.isfinite(values.to_numpy(dtype=float))
    return {
        "cop_cyc_evt": (
            base_support,
            (complete_electricity > 0) & (complete_heat > Q_MIN_KWH),
            np.ones(len(result), dtype=bool),
        ),
        "eta_h_cyc": (
            _required_model_support("eta_h_cyc", supports),
            (healthy_complete_heat > Q_MIN_KWH) & finite(complete_heat),
            np.ones(len(result), dtype=bool),
        ),
        "eta_e_cyc": (
            _required_model_support("eta_e_cyc", supports),
            (eta_denominator > 0.01) & finite(numerator),
            np.ones(len(result), dtype=bool),
        ),
        "cop_e": (
            base_support & healthy_support & event_support,
            (cop_e_denominator > 0) & finite(numerator),
            np.ones(len(result), dtype=bool),
        ),
        "epsilon_hl": (
            base_support & healthy_support & dynamic_support,
            dynamic_denominator > 0.01,
            np.ones(len(result), dtype=bool),
        ),
        "epsilon_hl_t0_proxy": (
            base_support & t0_support,
            (t0_denominator > 0.01) & finite(t0_denominator),
            t0_identifiable,
        ),
        "cop_cyc_k": (
            rr_support,
            (rr_e_total > 0) & (rr_q_total > Q_MIN_KWH),
            np.ones(len(result), dtype=bool),
        ),
        "epsilon_hl_2a": (
            base_support,
            (two_anchor_denominator > 0.01) & finite(two_anchor_denominator),
            anchor_identifiable & t0_identifiable,
        ),
    }


def _bootstrap_two_anchor_curve(
    candidates: pd.DataFrame,
    events: pd.DataFrame,
    experiment_id: str,
    q0_model: RidgeOutcomeModel,
) -> pd.DataFrame:
    """Re-select project anchors and recompute their losses for one replicate."""
    result = candidates.copy()
    result["L_T_two_anchor_hat_kwh"] = np.nan
    result["anchor_identifiable"] = False
    for cycle_name, indices in result.groupby("cycle_name", sort=False).groups.items():
        anchors = select_two_anchors(events, str(cycle_name), str(experiment_id))
        if not bool(anchors["valid"]):
            continue
        anchor_losses: dict[str, float] = {}
        for label in ("5", "35"):
            anchor_cycle = anchors[f"anchor_{label}_cycle"]
            anchor_source = (
                events["source_experiment_id"]
                if "source_experiment_id" in events.columns
                else events["experiment_id"]
            )
            anchor_source = anchor_source.where(
                anchor_source.notna(), events["experiment_id"]
            )
            anchor_event = events.loc[
                events["cycle_name"].astype(str).eq(str(anchor_cycle))
                & anchor_source.astype(str).eq(str(experiment_id))
            ]
            if anchor_event.empty:
                break
            predicted_q0 = float(q0_model.predict(anchor_event.iloc[[0]])[0])
            duration = float(anchor_event.iloc[0]["event_duration_observed_minutes"])
            observed_q = float(anchor_event.iloc[0]["Q_T_observed_kwh"])
            anchor_losses[label] = predicted_q0 * duration / 60 - observed_q
        if len(anchor_losses) != 2:
            continue
        selected = dict(anchors)
        selected["anchor_5_loss_kwh"] = anchor_losses["5"]
        selected["anchor_35_loss_kwh"] = anchor_losses["35"]
        local = result.loc[indices, "cycle_elapsed_minutes"]
        result.loc[indices, "L_T_two_anchor_hat_kwh"] = [
            project_two_anchor_loss(float(value), selected) for value in local
        ]
        result.loc[indices, "anchor_identifiable"] = local.between(
            float(anchors["anchor_5_time_minutes"]),
            float(anchors["anchor_35_time_minutes"]),
        ).to_numpy()
    return result


def _bootstrap_refit_summary(  # noqa: C901
    common: pd.DataFrame,
    curves: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    healthy_samples: pd.DataFrame,
    healthy_folds: dict[str, dict[str, RidgeOutcomeModel]],
    event_folds: dict[str, dict[str, RidgeOutcomeModel]],
    base_event_folds: dict[str, dict[str, RidgeOutcomeModel]],
    dynamic_loss_folds: dict[str, RidgeOutcomeModel],
    replicates: int,
    *,
    seed: int = 270,
    trajectory_path: Path | None = None,
    final_only: bool = False,
    n_jobs: int = 1,
    _replicate_specs: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Refit all candidate-dependent models after experiment-level resampling."""
    specs = _replicate_specs or [
        (index, int(value))
        for index, value in enumerate(np.random.SeedSequence(seed).generate_state(replicates))
    ]
    workers = min(max(int(n_jobs), 1), len(specs)) if specs else 1
    if workers > 1:
        batches = [specs[index::workers] for index in range(workers)]
        part_paths = (
            [
                trajectory_path.with_name(
                    f".{trajectory_path.stem}.part-{index}{trajectory_path.suffix}"
                )
                for index in range(workers)
            ]
            if trajectory_path is not None
            else [None] * workers
        )
        with parallel_config(backend="loky", n_jobs=workers, inner_max_num_threads=1):
            pieces = Parallel()(delayed(_bootstrap_refit_summary)(
                common,
                curves,
                events,
                healthy_samples,
                healthy_folds,
                event_folds,
                base_event_folds,
                dynamic_loss_folds,
                len(batch),
                seed=seed,
                trajectory_path=part_path,
                final_only=final_only,
                n_jobs=1,
                _replicate_specs=batch,
            ) for batch, part_path in zip(batches, part_paths, strict=True))
        minima: dict[tuple[str, str, str], list[pd.Timestamp]] = {}
        for piece in pieces:
            for key, values in piece.attrs["minima"].items():
                minima.setdefault(key, []).extend(values)
        if trajectory_path is not None:
            _merge_bootstrap_trajectories(
                [path for path in part_paths if path is not None], trajectory_path
            )
        result = _bootstrap_refit_rows(curves, minima, replicates)
        result.attrs["minima"] = minima
        result.attrs["draws"] = pd.concat(
            [piece.attrs["draws"] for piece in pieces], ignore_index=True
        )
        return result

    writer = None
    temporary_trajectory = (
        trajectory_path.with_suffix(".tmp.parquet") if trajectory_path is not None else None
    )
    minima: dict[tuple[str, str, str], list[pd.Timestamp]] = {}
    draw_rows: list[dict[str, object]] = []
    for algorithm, table in curves.items():
        for (cycle_name, metric_id), _curve in table.groupby(
            ["cycle_name", "metric_id"], sort=False
        ):
            key = (algorithm, str(metric_id), str(cycle_name))
            minima[key] = []

    event_experiments = set(events["experiment_id"].dropna().astype(str))
    healthy_experiments = set(healthy_samples["experiment_id"].dropna().astype(str))
    for replicate_id, replicate_seed in specs:
        rng = np.random.default_rng(replicate_seed)
        for heldout, heldout_common in common.groupby("experiment_id", sort=False):
            heldout = str(heldout)
            available = sorted((event_experiments & healthy_experiments) - {heldout})
            if len(available) < 2:
                continue
            sampled = rng.choice(available, size=len(available), replace=True)
            draw_counts = pd.Series(sampled).value_counts()
            draw_rows.extend(
                {
                    "replicate_id": replicate_id,
                    "heldout_experiment_id": heldout,
                    "source_experiment_id": source,
                    "draw_count": int(draw_counts.get(source, 0)),
                }
                for source in available
            )
            sampled_events = _resample_experiments(events, sampled)
            sampled_healthy = _resample_experiments(healthy_samples, sampled)
            if sampled_events.empty or sampled_healthy.empty:
                continue
            q0_model = _fit_bootstrap_model(
                sampled_healthy,
                "healthy_water_heat_kw",
                healthy_folds[heldout]["healthy_water_heat_kw"].alpha,
                HEALTHY_FEATURES,
            )
            p0_model = _fit_bootstrap_model(
                sampled_healthy,
                "healthy_compressor_power_kw",
                healthy_folds[heldout]["healthy_compressor_power_kw"].alpha,
                HEALTHY_FEATURES,
            )
            if q0_model is None or p0_model is None:
                continue
            sampled_events["L_T_dynamic_kwh"] = q0_model.predict(sampled_events) * sampled_events[
                "event_duration_observed_minutes"
            ].to_numpy(dtype=float) / 60 - sampled_events["Q_T_observed_kwh"].to_numpy(dtype=float)
            models: dict[str, RidgeOutcomeModel | None] = {
                "Qw0": q0_model,
                "Pcomp0": p0_model,
                "E_T": _fit_bootstrap_model(
                    sampled_events,
                    "E_T_observed_kwh",
                    base_event_folds[heldout]["E_T_observed_kwh"].alpha,
                ),
                "Q_T": _fit_bootstrap_model(
                    sampled_events,
                    "Q_T_observed_kwh",
                    base_event_folds[heldout]["Q_T_observed_kwh"].alpha,
                ),
                "E_PD": None if final_only else _fit_bootstrap_model(
                    sampled_events,
                    "E_PD_kwh",
                    base_event_folds[heldout]["E_PD_kwh"].alpha,
                ),
                "Q_PD": None if final_only else _fit_bootstrap_model(
                    sampled_events,
                    "Q_PD_kwh",
                    base_event_folds[heldout]["Q_PD_kwh"].alpha,
                ),
                "E_comp_T": _fit_bootstrap_model(
                    sampled_events,
                    "E_comp_T_observed_kwh",
                    event_folds[heldout]["E_comp_T_observed_kwh"].alpha,
                ),
                "D_T": _fit_bootstrap_model(
                    sampled_events,
                    "event_duration_observed_minutes",
                    event_folds[heldout]["event_duration_observed_minutes"].alpha,
                ),
                "L_t0": None if final_only else _fit_bootstrap_model(
                    sampled_events,
                    "L_T_t0_kwh",
                    event_folds[heldout]["L_T_t0_kwh"].alpha,
                ),
                "L_dynamic": None if final_only else _fit_bootstrap_model(
                    sampled_events,
                    "L_T_dynamic_kwh",
                    dynamic_loss_folds[heldout].alpha,
                ),
            }
            candidate = (
                heldout_common.sort_values(["cycle_name", "candidate_time"], kind="stable")
                .reset_index(drop=True)
                .copy()
            )
            candidate["candidate_time"] = pd.to_datetime(candidate["candidate_time"])
            # Two-anchor interpolation is a within-experiment construction,
            # not another model-training target.  Build a separate
            # with-replacement held-out sibling pool for each target cycle;
            # this both excludes that cycle from its own anchors and allows
            # anchor-selection uncertainty to propagate across replicates.
            if not final_only:
                candidate = pd.concat(
                    [
                        _bootstrap_two_anchor_curve(
                            cycle_candidates,
                            _resample_anchor_events(events, heldout, str(cycle_name), rng),
                            heldout,
                            q0_model,
                        )
                        for cycle_name, cycle_candidates in candidate.groupby(
                            "cycle_name", sort=False
                        )
                    ],
                    ignore_index=True,
                    sort=False,
                )
            prediction = _bootstrap_prediction_frame(candidate, models)
            if prediction is None:
                continue
            candidate = prediction
            objectives = _bootstrap_objectives(candidate)
            if not objectives:
                continue
            component_supports = _bootstrap_component_supports(candidate, models)
            statuses = _bootstrap_metric_status(candidate, models, component_supports)
            trajectory_eligible = {
                metric_id: np.zeros(len(candidate), dtype=bool)
                for metric_id in ("cop_cyc_evt", "eta_h_cyc", "eta_e_cyc")
            }
            trajectory_base = {
                metric_id: np.zeros(len(candidate), dtype=bool)
                for metric_id in trajectory_eligible
            }
            trajectory_supported = {
                metric_id: np.zeros(len(candidate), dtype=bool)
                for metric_id in trajectory_eligible
            }
            trajectory_physical = {
                metric_id: np.zeros(len(candidate), dtype=bool)
                for metric_id in trajectory_eligible
            }
            trajectory_measurement = {
                metric_id: np.zeros(len(candidate), dtype=bool)
                for metric_id in trajectory_eligible
            }
            selected_objectives = (
                {metric: objectives[metric] for metric in trajectory_eligible}
                if final_only
                else objectives
            )
            for metric_id, objective in selected_objectives.items():
                algorithm = METRICS[metric_id].algorithm
                if metric_id not in statuses:
                    continue
                supported, physical, identifiable = statuses[metric_id]
                for cycle_name, indices in candidate.groupby(
                    "cycle_name", sort=False
                ).groups.items():
                    positions = np.asarray(list(indices), dtype=int)
                    values = np.asarray(objective)[positions]
                    local_supported = supported[positions]
                    local_physical = physical[positions]
                    local_identifiable = identifiable[positions]
                    local = candidate.iloc[positions]
                    measurement = (
                        local["rr_measurement_eligible"]
                        if metric_id == "cop_cyc_k"
                        else _metric_measurement_support(local, metric_id)
                        if metric_id in FINAL_MODEL_REQUIREMENTS
                        else local["measurement_eligible"]
                    )
                    base_eligible = (
                        local_supported
                        & local_physical
                        & local_identifiable
                        & local["pre_action_window_valid"].fillna(False).to_numpy(dtype=bool)
                        & measurement.fillna(False).to_numpy(dtype=bool)
                        & np.isfinite(values)
                    )
                    eligible = base_eligible
                    eligible = _long_support_runs(
                        local["candidate_time"], pd.Series(eligible)
                    ).to_numpy()
                    if metric_id in trajectory_eligible:
                        trajectory_base[metric_id][positions] = base_eligible
                        trajectory_supported[metric_id][positions] = local_supported
                        trajectory_physical[metric_id][positions] = local_physical
                        trajectory_measurement[metric_id][positions] = measurement.fillna(
                            False
                        ).to_numpy(dtype=bool)
                        trajectory_eligible[metric_id][positions] = eligible
                    valid = eligible
                    if not valid.any():
                        continue
                    valid_positions = np.flatnonzero(valid)
                    valid_values = values[valid]
                    selected = (
                        valid_positions[int(np.argmin(valid_values))]
                        if METRICS[metric_id].direction == "min"
                        else valid_positions[int(np.argmax(valid_values))]
                    )
                    key = (algorithm, metric_id, str(cycle_name))
                    if key in minima:
                        minima[key].append(pd.Timestamp(local.iloc[selected]["candidate_time"]))
            if trajectory_path is not None:
                import pyarrow as pa
                import pyarrow.parquet as pq

                columns = [
                    "cycle_name",
                    "experiment_id",
                    "candidate_time",
                    "stable_start_fixed9",
                    "pre_action_window_valid",
                    "heating_electricity_kwh",
                    "water_heating_kwh",
                    "heating_compressor_electricity_kwh",
                    "E_T",
                    "Q_T",
                    "E_comp_T",
                    "D_T",
                    "Qw0_H",
                    "Qw0",
                    "Pcomp0_H",
                    "Pcomp0",
                ]
                trajectory = candidate[columns].copy()
                trajectory.insert(0, "replicate_id", replicate_id)
                for metric_id in trajectory_eligible:
                    trajectory[metric_id] = objectives[metric_id]
                    trajectory[f"{metric_id}_model_supported"] = trajectory_supported[
                        metric_id
                    ]
                    trajectory[f"{metric_id}_physical_valid"] = trajectory_physical[metric_id]
                    trajectory[f"{metric_id}_measurement_eligible"] = trajectory_measurement[
                        metric_id
                    ]
                    trajectory[f"{metric_id}_base_eligible"] = trajectory_base[metric_id]
                    trajectory[f"{metric_id}_eligible"] = trajectory_eligible[metric_id]
                for name in ("Q_T", "Qw0", "D_T", "Pcomp0", "E_comp_T"):
                    trajectory[f"support_{name}"] = component_supports[name]
                arrow = pa.Table.from_pandas(trajectory, preserve_index=False)
                if writer is None:
                    assert temporary_trajectory is not None
                    temporary_trajectory.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(temporary_trajectory, arrow.schema)
                writer.write_table(arrow)

    if writer is not None:
        writer.close()
        assert temporary_trajectory is not None and trajectory_path is not None
        temporary_trajectory.replace(trajectory_path)
    result = _bootstrap_refit_rows(curves, minima, replicates)
    result.attrs["minima"] = minima
    result.attrs["draws"] = pd.DataFrame(draw_rows)
    return result


def _bootstrap_refit_rows(
    curves: dict[str, pd.DataFrame],
    minima: dict[tuple[str, str, str], list[pd.Timestamp]],
    replicates: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for algorithm, table in curves.items():
        for (cycle_name, experiment, metric_id), curve in table.groupby(
            ["cycle_name", "experiment_id", "metric_id"], sort=False
        ):
            values = minima.get((algorithm, str(metric_id), str(cycle_name)), [])
            numeric = np.array([value.value for value in values], dtype=np.int64)
            basin_start = pd.to_datetime(curve["basin_5pct_start"].iloc[0], errors="coerce")
            basin_end = pd.to_datetime(curve["basin_5pct_end"].iloc[0], errors="coerce")
            rows.append(
                {
                    "cycle_name": cycle_name,
                    "experiment_id": experiment,
                    "algorithm": algorithm,
                    "metric_id": metric_id,
                    "repeat_count": replicates,
                    "bootstrap_valid_extreme_count": len(values),
                    "bootstrap_extreme_median_time": (
                        pd.Timestamp(int(np.median(numeric))) if numeric.size else pd.NaT
                    ),
                    "bootstrap_extreme_q25_time": (
                        pd.Timestamp(int(np.quantile(numeric, 0.25))) if numeric.size else pd.NaT
                    ),
                    "bootstrap_extreme_q75_time": (
                        pd.Timestamp(int(np.quantile(numeric, 0.75))) if numeric.size else pd.NaT
                    ),
                    "bootstrap_in_original_5pct_basin_fraction": (
                        float(np.mean([basin_start <= value <= basin_end for value in values]))
                        if values and pd.notna(basin_start) and pd.notna(basin_end)
                        else np.nan
                    ),
                    "bootstrap_method": "experiment_refit_excluding_target",
                }
            )
    return pd.DataFrame(rows)


def _merge_bootstrap_trajectories(parts: list[Path], output: Path) -> None:
    import pyarrow.parquet as pq

    available = [path for path in parts if path.exists()]
    if not available:
        return
    temporary = output.with_suffix(".tmp.parquet")
    writer = None
    try:
        for path in available:
            source = pq.ParquetFile(path)
            writer = writer or pq.ParquetWriter(temporary, source.schema_arrow)
            for row_group in range(source.num_row_groups):
                writer.write_table(source.read_row_group(row_group))
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output)
    for path in available:
        path.unlink()
