"""Small post-processing steps for selected cost-function iterations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .core import optimize_cycle_cop_cost


def close_stable_cycle(
    table: pd.DataFrame,
    recovery_by_setpoint: Mapping[float, tuple[float, float, float]],
) -> pd.DataFrame:
    """Convert a V2.6 current-cycle table to a stable-to-stable cycle."""
    tables = []
    for _cycle_name, source in table.groupby("cycle_name", sort=False):
        values = source.copy()
        setpoint = float(
            pd.to_numeric(values["water_temperature_setpoint"], errors="coerce").median()
        )
        if setpoint not in recovery_by_setpoint:
            raise ValueError("v2.6.2 requires a 50 or 55 degC setpoint")
        duration, recovery_electricity, recovery_heat = recovery_by_setpoint[setpoint]

        prefix_electricity = pd.to_numeric(
            values["heating_boundary_electricity_adjustment_kwh"], errors="coerce"
        )
        prefix_unit_heat = pd.to_numeric(
            values["heating_boundary_unit_heat_adjustment_kwh"], errors="coerce"
        )
        values["observed_prefix_recovery_electricity_kwh"] = prefix_electricity
        values["observed_prefix_recovery_unit_heat_kwh"] = prefix_unit_heat
        values["stable_heating_electricity_kwh"] = (
            pd.to_numeric(values["heating_electricity_kwh"], errors="coerce") - prefix_electricity
        )
        values["stable_unit_heating_kwh"] = (
            pd.to_numeric(values["unit_heating_kwh"], errors="coerce") - prefix_unit_heat
        )
        values["projected_recovery_electricity_kwh"] = recovery_electricity
        values["projected_recovery_heat_kwh"] = recovery_heat
        values["recovery_duration_minutes"] = duration
        values["recovery_electricity_kwh"] = recovery_electricity
        values["recovery_heat_kwh"] = recovery_heat
        values["transition_electricity_kwh"] = (
            values["defrost_electricity_kwh"] + recovery_electricity
        )
        values["transition_service_heat_kwh"] = (
            values["preparation_heat_kwh"] - values["defrost_absorbed_heat_kwh"] + recovery_heat
        )
        values["user_heating_kwh"] = (
            values["stable_unit_heating_kwh"]
            + values["preparation_heat_kwh"]
            - values["defrost_absorbed_heat_kwh"]
        )
        denominator = values["user_heating_kwh"] + recovery_heat
        denominator_eligible = np.isfinite(denominator) & denominator.gt(0)
        values["heat_balance_eligible"] = denominator_eligible
        values["optimization_eligible"] = (
            values["optimization_eligible"].fillna(False) & denominator_eligible
        )
        values["heating_electricity_kwh"] = values["stable_heating_electricity_kwh"]
        values["algorithm"] = "v2.6.2"
        values["model_protocol"] = "stable_to_stable_projected_post_defrost_recovery"

        curve, optimum = optimize_cycle_cop_cost(
            values,
            defrost_recovery_electricity_kwh=values["transition_electricity_kwh"],
            defrost_recovery_heat_kwh=recovery_heat,
        )
        eligible = curve["optimization_eligible"].fillna(False)
        minimum = float(optimum["inverse_cop"])
        curve["relative_regret"] = (curve["inverse_cop"] / minimum - 1.0).where(eligible)
        curve["near_optimal_1pct"] = eligible & curve["relative_regret"].le(0.01)
        curve["near_optimal_5pct"] = eligible & curve["relative_regret"].le(0.05)
        curve["t_star"] = optimum["candidate_time"]
        curve["t_star_model_supported"] = bool(
            curve.loc[
                curve["candidate_time"].eq(optimum["candidate_time"]),
                "model_supported",
            ].iloc[0]
        )
        curve["minimum_location"] = optimum["minimum_location"]
        tables.append(curve)

    return pd.concat(tables, ignore_index=True, sort=False) if tables else table.copy()


def normalize_degradation(table: pd.DataFrame) -> pd.DataFrame:
    """Express closed-cycle cost as baseline plus avoidable excess electricity."""
    tables = []
    for _cycle_name, source in table.groupby("cycle_name", sort=False):
        values = source.sort_values("candidate_time", kind="stable").reset_index(drop=True)
        eligible = values["optimization_eligible"].fillna(False)
        baseline_rows = values.loc[
            eligible
            & np.isfinite(values["stable_heating_electricity_kwh"])
            & np.isfinite(values["stable_unit_heating_kwh"])
            & values["stable_unit_heating_kwh"].gt(0)
        ].head(5)
        if baseline_rows.empty:
            raise ValueError("v2.6.3 requires an early stable-heating baseline")
        baseline = float(
            baseline_rows["stable_heating_electricity_kwh"].iloc[-1]
            / baseline_rows["stable_unit_heating_kwh"].iloc[-1]
        )
        stable_heat = pd.to_numeric(values["stable_unit_heating_kwh"], errors="coerce")
        values["baseline_inverse_cop"] = baseline
        values["baseline_candidate_count"] = len(baseline_rows)
        values["heating_degradation_electricity_kwh"] = (
            values["stable_heating_electricity_kwh"] - baseline * stable_heat
        ).clip(lower=0)
        values["transition_excess_electricity_kwh"] = (
            values["transition_electricity_kwh"] - baseline * values["transition_service_heat_kwh"]
        ).clip(lower=0)
        values["total_excess_electricity_kwh"] = (
            values["heating_degradation_electricity_kwh"]
            + values["transition_excess_electricity_kwh"]
        )
        values["algorithm"] = "v2.6.3"
        values["model_protocol"] = "baseline_normalized_degradation"

        objective = values.copy()
        objective["heating_electricity_kwh"] = (
            baseline * stable_heat + values["heating_degradation_electricity_kwh"]
        )
        objective["user_heating_kwh"] = stable_heat
        curve, optimum = optimize_cycle_cop_cost(
            objective,
            defrost_recovery_electricity_kwh=values["transition_excess_electricity_kwh"],
        )
        values["inverse_cop"] = curve["inverse_cop"]
        values["cycle_cop"] = curve["cycle_cop"]
        minimum = float(optimum["inverse_cop"])
        values["relative_regret"] = (values["inverse_cop"] / minimum - 1.0).where(eligible)
        values["near_optimal_1pct"] = eligible & values["relative_regret"].le(0.01)
        values["near_optimal_5pct"] = eligible & values["relative_regret"].le(0.05)
        values["t_star"] = optimum["candidate_time"]
        values["t_star_model_supported"] = bool(
            values.loc[
                values["candidate_time"].eq(optimum["candidate_time"]),
                "model_supported",
            ].iloc[0]
        )
        values["minimum_location"] = optimum["minimum_location"]
        tables.append(values)

    return pd.concat(tables, ignore_index=True, sort=False) if tables else table.copy()


def marginal_dinkelbach(table: pd.DataFrame, window_points: int = 5) -> pd.DataFrame:
    """Find the candidate closest to LOEO average-cost marginal indifference."""
    values = table.copy()
    eligible = values["optimization_eligible"].fillna(False) & pd.to_numeric(
        values["stable_unit_heating_kwh"], errors="coerce"
    ).gt(0)
    values["excess_per_heating_kwh"] = (
        values["total_excess_electricity_kwh"] / values["stable_unit_heating_kwh"]
    ).where(eligible)
    cycle_minimum = (
        values.loc[eligible]
        .groupby(["experiment_id", "cycle_name"], sort=False)["excess_per_heating_kwh"]
        .min()
        .rename("cycle_minimum_excess_per_heating_kwh")
        .reset_index()
    )
    shadows = {
        experiment: float(
            cycle_minimum.loc[
                cycle_minimum["experiment_id"].ne(experiment),
                "cycle_minimum_excess_per_heating_kwh",
            ].median()
        )
        for experiment in cycle_minimum["experiment_id"].unique()
    }
    values["shadow_excess_per_heating_kwh"] = values["experiment_id"].map(shadows)

    tables = []
    for _cycle_name, source in values.groupby("cycle_name", sort=False):
        curve = source.sort_values("candidate_time", kind="stable").reset_index(drop=True)
        previous_excess = curve["total_excess_electricity_kwh"].shift(window_points)
        previous_heat = curve["stable_unit_heating_kwh"].shift(window_points)
        previous_time = pd.to_datetime(curve["candidate_time"]).shift(window_points)
        origin_time = pd.to_datetime(curve["t_heating_stable"], errors="coerce")
        curve["marginal_window_minutes"] = (
            pd.to_datetime(curve["candidate_time"])
            - previous_time.where(previous_time.notna(), origin_time)
        ).dt.total_seconds() / 60
        curve["marginal_delta_excess_electricity_kwh"] = curve[
            "total_excess_electricity_kwh"
        ] - previous_excess.fillna(0)
        curve["marginal_delta_heating_kwh"] = curve[
            "stable_unit_heating_kwh"
        ] - previous_heat.fillna(0)
        curve["marginal_delta_g_kwh"] = (
            curve["marginal_delta_excess_electricity_kwh"]
            - curve["shadow_excess_per_heating_kwh"] * curve["marginal_delta_heating_kwh"]
        )
        marginal_eligible = (
            curve["optimization_eligible"].fillna(False)
            & np.isfinite(curve["marginal_delta_g_kwh"])
            & curve["marginal_delta_heating_kwh"].gt(0)
        )
        curve["marginal_eligible"] = marginal_eligible
        curve["marginal_window_points"] = window_points
        curve["inverse_cop"] = (
            curve["baseline_inverse_cop"]
            + curve["marginal_delta_g_kwh"].abs() / curve["marginal_delta_heating_kwh"]
        )
        curve["cycle_cop"] = 1 / curve["inverse_cop"]
        if not marginal_eligible.any():
            raise ValueError("v2.6.4 requires a finite LOEO shadow price")
        best_index = curve["inverse_cop"].where(marginal_eligible).idxmin()
        minimum = float(curve.loc[best_index, "inverse_cop"])
        curve["relative_regret"] = (curve["inverse_cop"] / minimum - 1).where(marginal_eligible)
        curve["near_optimal_1pct"] = marginal_eligible & curve["relative_regret"].le(0.01)
        curve["near_optimal_5pct"] = marginal_eligible & curve["relative_regret"].le(0.05)
        curve["t_star"] = curve.loc[best_index, "candidate_time"]
        curve["t_star_model_supported"] = bool(curve.loc[best_index, "model_supported"])
        positions = np.flatnonzero(marginal_eligible.to_numpy())
        best_position = int(best_index)
        curve["minimum_location"] = (
            "left_observed"
            if best_position == positions[0]
            else "right_observed"
            if best_position == positions[-1]
            else "interior"
        )
        curve["algorithm"] = "v2.6.4"
        curve["model_protocol"] = "loeo_dinkelbach_five_minute_marginal_balance"
        tables.append(curve)

    if not tables:
        return table.copy()
    result = pd.concat(tables, ignore_index=True, sort=False)
    result["t_star"] = pd.to_datetime(result["t_star"], errors="coerce")
    return result


def select_final_basin(
    curve: pd.DataFrame, raw_t_star: pd.Timestamp, minimum_location: str
) -> dict[str, object]:
    """Choose the latest supported, marginally confirmed point in the raw basin."""
    values = curve.sort_values("candidate_time", kind="stable").reset_index(drop=True)
    raw = values["candidate_time"].eq(pd.Timestamp(raw_t_star))
    near = values["near_optimal_1pct"].fillna(False)
    segments = near.ne(near.shift(fill_value=False)).cumsum()
    basin = near & segments.eq(segments.loc[raw].iloc[0])
    supported = basin & values["model_supported"].fillna(False)
    confirmed = supported & values["marginal_delta_g_kwh"].ge(0)
    if confirmed.any():
        chosen = values.loc[confirmed, "candidate_time"].max()
        status = "supported_optimal"
        hard = True
    elif supported.any():
        chosen = values.loc[supported, "candidate_time"].max()
        status = "supported_basin_no_marginal_confirmation"
        hard = False
    else:
        chosen = pd.Timestamp(raw_t_star)
        status = "extrapolated_raw_optimum"
        hard = False
    if str(minimum_location).startswith("right"):
        status, hard = "right_censored_lower_bound", False
    elif str(minimum_location).startswith("left"):
        status, hard = "left_censored_upper_bound", False
    return {"t_star": pd.Timestamp(chosen), "decision_status": status, "hard_label_eligible": hard}


def finalize_supported_basin(table: pd.DataFrame) -> pd.DataFrame:
    """Keep the average curve and use five-minute marginal evidence only to decide."""
    marginal = marginal_dinkelbach(table)
    columns = [
        column
        for column in marginal
        if column.startswith("marginal_")
        or column in {"shadow_excess_per_heating_kwh", "excess_per_heating_kwh"}
    ]
    values = table.copy()
    values[columns] = marginal[columns]
    tables = []
    for _cycle_name, source in values.groupby("cycle_name", sort=False):
        curve = source.sort_values("candidate_time", kind="stable").reset_index(drop=True)
        eligible = curve["optimization_eligible"].fillna(False)
        raw_index = curve["inverse_cop"].where(eligible).idxmin()
        raw_time = pd.Timestamp(curve.loc[raw_index, "candidate_time"])
        positions = np.flatnonzero(eligible.to_numpy())
        location = (
            "left_boundary"
            if raw_index == positions[0]
            else "right_observed"
            if raw_index == positions[-1]
            else "interior"
        )
        decision = select_final_basin(curve, raw_time, location)
        chosen = curve["candidate_time"].eq(decision["t_star"])
        raw_cost = float(curve.loc[raw_index, "inverse_cop"])
        curve["raw_t_star"] = raw_time
        curve["t_star"] = decision["t_star"]
        curve["decision_regret"] = float(curve.loc[chosen, "inverse_cop"].iloc[0] / raw_cost - 1)
        curve["t_star_model_supported"] = bool(curve.loc[chosen, "model_supported"].iloc[0])
        curve["decision_status"] = decision["decision_status"]
        curve["hard_label_eligible"] = decision["hard_label_eligible"]
        curve["minimum_location"] = location
        curve["algorithm"] = "v2.6.5"
        curve["model_protocol"] = "causal_duration_closed_average_supported_marginal_basin"
        tables.append(curve)
    return pd.concat(tables, ignore_index=True, sort=False) if tables else table.copy()
