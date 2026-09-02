"""Minimal empirical defrost-cost calculations on unsmoothed measurements."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from cost.energy_models import integrate_energy_curve_kwh as integrate_energy_curve_kwh
from cost.energy_models import integrate_energy_kwh as integrate_energy_kwh
from cost.energy_models import water_side_heating_kw as water_side_heating_kw


def candidate_domain_end(
    observed_defrost: pd.Timestamp | None,
    sensor_record_end: pd.Timestamp | None,
) -> tuple[pd.Timestamp, str, bool]:
    """Return the observed boundary, or an explicitly right-censored sensor boundary."""
    if observed_defrost is not None:
        return pd.Timestamp(observed_defrost), "observed_defrost", False
    if sensor_record_end is not None:
        return pd.Timestamp(sensor_record_end), "sensor_record_end", True
    raise ValueError("candidate end is unavailable")


def count_true_runs(values: list[bool] | pd.Series) -> int:
    """Count disconnected True regions in an ordered Boolean sequence."""
    previous = False
    runs = 0
    for value in values:
        current = bool(value)
        runs += int(current and not previous)
        previous = current
    return runs


def find_recovery_time(
    timestamps: pd.Series | pd.DatetimeIndex,
    heating_kw: pd.Series,
    *,
    reference_kw: float,
    threshold_fraction: float = 0.9,
    continuous_seconds: float = 30.0,
) -> pd.Timestamp | None:
    """Return the first raw point starting a continuous recovered run."""
    time = pd.Series(pd.to_datetime(timestamps, errors="coerce")).reset_index(drop=True)
    heat = pd.to_numeric(heating_kw, errors="coerce").reset_index(drop=True)
    good = heat.ge(threshold_fraction * reference_kw) & time.notna()
    run = good.ne(good.shift(fill_value=False)).cumsum()
    nominal_dt = time.diff().dt.total_seconds().median()
    for _, index in time[good].groupby(run[good]).groups.items():
        points = list(index)
        duration = (time.iloc[points[-1]] - time.iloc[points[0]]).total_seconds() + nominal_dt
        if duration >= continuous_seconds:
            return pd.Timestamp(time.iloc[points[0]])
    return None


def optimize_renewal_cost(
    candidates: pd.DataFrame,
    *,
    ticket_cost_kwh: float,
    ticket_duration_hours: float,
    near_optimal_fraction: float = 0.05,
    required_end_time: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate renewal-average cost and return its earliest minimizer."""
    curve = candidates.copy()
    if required_end_time is not None and (
        curve.empty
        or pd.to_datetime(curve["candidate_time"]).max() < pd.Timestamp(required_end_time)
    ):
        raise ValueError("candidate domain does not reach the observed defrost boundary")
    curve["renewal_cost_kw"] = (curve["heating_cost_kwh"] + ticket_cost_kwh) / (
        curve["heating_hours"] + ticket_duration_hours
    )
    best_index = curve["renewal_cost_kw"].idxmin()
    best_position = curve.index.get_loc(best_index)
    location = (
        "left_boundary"
        if best_position == 0
        else "right_boundary"
        if best_position == len(curve) - 1
        else "interior"
    )
    minimum = float(curve.loc[best_index, "renewal_cost_kw"])
    near = curve.loc[curve["renewal_cost_kw"].le((1 + near_optimal_fraction) * minimum)]
    optimum = curve.loc[best_index].to_dict()
    optimum.update(
        {
            "minimum_location": location,
            "near_opt_start": near["candidate_time"].min(),
            "near_opt_end": near["candidate_time"].max(),
        }
    )
    return curve, optimum


def _candidate_eligibility_masks(
    curve: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, bool]:
    """Resolve Pe support, integration, and joint eligibility with legacy defaults."""
    has_pe_support = "pe_supported" in curve
    if has_pe_support:
        pe_supported = curve["pe_supported"].fillna(False).astype(bool)
    elif "optimization_eligible" in curve:
        pe_supported = curve["optimization_eligible"].fillna(False).astype(bool)
    else:
        pe_supported = pd.Series(True, index=curve.index)
    if "integration_eligible" in curve:
        integration_eligible = curve["integration_eligible"].fillna(False).astype(bool)
    elif has_pe_support and "optimization_eligible" in curve:
        integration_eligible = (
            curve["optimization_eligible"].fillna(False).astype(bool) | ~pe_supported
        )
    else:
        integration_eligible = pd.Series(True, index=curve.index)
    eligible = (
        curve["optimization_eligible"].fillna(False).astype(bool)
        if "optimization_eligible" in curve
        else pe_supported & integration_eligible
    )
    return pe_supported, integration_eligible, eligible, has_pe_support


def optimize_cycle_cop_cost(
    candidates: pd.DataFrame,
    *,
    defrost_recovery_electricity_kwh: float | pd.Series,
    defrost_recovery_heat_kwh: float | pd.Series = 0.0,
    near_optimal_fraction: float = 0.05,
    required_end_time: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Minimize full-cycle electricity per delivered heating energy."""
    curve = candidates.copy()
    curve["_transient_ticket_kwh"] = defrost_recovery_electricity_kwh
    curve["_transient_recovery_heat_kwh"] = defrost_recovery_heat_kwh
    curve = curve.sort_values("candidate_time", kind="stable").reset_index(drop=True)
    if required_end_time is not None and (
        curve.empty
        or pd.to_datetime(curve["candidate_time"]).max() < pd.Timestamp(required_end_time)
    ):
        raise ValueError("candidate domain does not reach the observed defrost boundary")
    pe_supported, integration_eligible, eligible, has_pe_support = (
        _candidate_eligibility_masks(curve)
    )
    if not eligible.any():
        raise ValueError("no_supported_candidates")
    heating_electricity = curve.loc[eligible, "heating_electricity_kwh"]
    tickets = pd.to_numeric(curve["_transient_ticket_kwh"], errors="coerce")
    recovery_heat = pd.to_numeric(
        curve["_transient_recovery_heat_kwh"], errors="coerce"
    )
    curve["cycle_user_heating_kwh"] = curve["user_heating_kwh"] + recovery_heat
    cycle_user_heating = curve.loc[eligible, "cycle_user_heating_kwh"]
    if (
        not np.isfinite(heating_electricity).all()
        or not np.isfinite(tickets.loc[eligible]).all()
        or not np.isfinite(cycle_user_heating).all()
    ):
        raise ValueError("cycle COP requires finite energy values")
    if cycle_user_heating.le(0).any():
        raise ValueError("cycle COP requires positive user heating")
    curve["cycle_electricity_kwh"] = curve["heating_electricity_kwh"] + tickets
    curve["inverse_cop"] = (
        curve["cycle_electricity_kwh"] / curve["cycle_user_heating_kwh"]
    )
    if not np.isfinite(curve.loc[eligible, "inverse_cop"]).all():
        raise ValueError("cycle COP requires finite energy values")
    curve["cycle_cop"] = 1 / curve["inverse_cop"]
    curve = curve.drop(
        columns=["_transient_ticket_kwh", "_transient_recovery_heat_kwh"]
    )
    best_index = curve["inverse_cop"].where(eligible).idxmin()
    best_position = curve.index.get_loc(best_index)
    eligible_positions = np.flatnonzero(eligible.to_numpy())
    if best_position == eligible_positions[-1]:
        if best_position == len(curve) - 1:
            location = "right_observed"
        elif pe_supported.iloc[best_position + 1 :].any():
            location = "right_integration_limited"
        else:
            location = "right_support_limited"
    elif best_position == eligible_positions[0]:
        location = "left_boundary"
    else:
        location = "interior"
    minimum = float(curve.loc[best_index, "inverse_cop"])
    near = curve.loc[
        eligible & curve["inverse_cop"].le((1 + near_optimal_fraction) * minimum)
    ]
    optimum = curve.loc[best_index].to_dict()
    optimum.update(
        {
            "minimum_location": location,
            "left_support_removed": bool(
                (~pe_supported.iloc[: eligible_positions[0]]).any()
            ),
            "left_integration_removed": bool(
                (
                    pe_supported.iloc[: eligible_positions[0]]
                    & ~integration_eligible.iloc[: eligible_positions[0]]
                ).any()
            ),
            "near_opt_start": near["candidate_time"].min(),
            "near_opt_end": near["candidate_time"].max(),
        }
    )
    return curve, optimum


def _partial_pool_prior_strength(events: pd.DataFrame, outcome: str) -> float:
    groups = events.groupby("experiment_id", sort=False)[outcome]
    counts = groups.size()
    if len(counts) < 2 or len(events) <= len(counts):
        return math.inf
    within_sum = sum(float(((values - values.mean()) ** 2).sum()) for _, values in groups)
    within_variance = within_sum / (len(events) - len(counts))
    between_variance = max(
        float(groups.mean().var(ddof=1) - within_variance * (1 / counts).mean()),
        0.0,
    )
    return within_variance / between_variance if between_variance > 0 else math.inf


def partial_pool_group_estimates(events: pd.DataFrame) -> pd.DataFrame:
    """Shrink each experiment's observed ticket mean toward the cohort mean."""
    rows = []
    for experiment, values in events.groupby("experiment_id", sort=True):
        row: dict[str, object] = {
            "experiment_id": experiment,
            "ticket_event_count": len(values),
        }
        for label, outcome in (("cost", "equivalent_cost_kwh"), ("duration", "duration_minutes")):
            cohort_mean = float(events[outcome].mean())
            strength = _partial_pool_prior_strength(events, outcome)
            estimate = (
                cohort_mean
                if not math.isfinite(strength)
                else (values[outcome].sum() + strength * cohort_mean) / (len(values) + strength)
            )
            row[f"partial_pool_{label}"] = float(estimate)
            row[f"prior_strength_{label}"] = strength
        rows.append(row)
    return pd.DataFrame(rows)


def leave_one_event_out_partial_pool(events: pd.DataFrame) -> pd.DataFrame:
    """Audit partial pooling without using the held-out event's ticket outcome."""
    rows = []
    for index, event in events.iterrows():
        train = events.drop(index=index)
        estimates = partial_pool_group_estimates(train)
        matched = estimates.loc[estimates["experiment_id"].eq(event["experiment_id"])]
        rows.append(
            {
                "cycle_name": event["cycle_name"],
                "predicted_partial_pool_cost": (
                    float(matched.iloc[0]["partial_pool_cost"])
                    if not matched.empty
                    else float(train["equivalent_cost_kwh"].mean())
                ),
                "predicted_partial_pool_duration": (
                    float(matched.iloc[0]["partial_pool_duration"])
                    if not matched.empty
                    else float(train["duration_minutes"].mean())
                ),
            }
        )
    return pd.DataFrame(rows)


def build_partial_pool_curves(
    curves: pd.DataFrame,
    events: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute historical candidate costs with experiment-calibrated ticket terms."""
    estimates = partial_pool_group_estimates(events)
    result = curves.copy()
    if "experiment_id" not in result:
        result = result.merge(
            catalog[["cycle_name", "experiment_id"]], on="cycle_name", how="left"
        )
    result = result.merge(estimates, on="experiment_id", how="left")
    result["partial_pool_cost"] = result["partial_pool_cost"].fillna(
        events["equivalent_cost_kwh"].mean()
    )
    result["partial_pool_duration"] = result["partial_pool_duration"].fillna(
        events["duration_minutes"].mean()
    )
    result["renewal_cost_partial_pool"] = (
        result["heating_cost_kwh"] + result["partial_pool_cost"]
    ) / (result["heating_hours"] + result["partial_pool_duration"] / 60.0)
    result["relative_regret_partial_pool"] = result.groupby("cycle_name")[
        "renewal_cost_partial_pool"
    ].transform(lambda values: values / values.min() - 1.0)
    return result
