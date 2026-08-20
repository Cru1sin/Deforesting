"""Minimal empirical defrost-cost calculations on unsmoothed measurements."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


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


def water_side_heating_kw(frame: pd.DataFrame) -> pd.Series:
    """Return raw water-side heating capacity in kW."""
    return (
        1.161
        * pd.to_numeric(frame["water_flow"], errors="coerce")
        * (
            pd.to_numeric(frame["water_out_temperature"], errors="coerce")
            - pd.to_numeric(frame["water_in_temperature"], errors="coerce")
        )
    )


def integrate_energy_kwh(
    timestamps: pd.Series | pd.DatetimeIndex,
    power_kw: pd.Series,
    *,
    maximum_gap_seconds: float = 5.0,
) -> tuple[float, float]:
    """Trapezoid-integrate valid adjacent raw points without bridging gaps."""
    raw = pd.DataFrame(
        {
            "time": pd.to_datetime(timestamps, errors="coerce"),
            "power": pd.to_numeric(power_kw, errors="coerce"),
        }
    )
    span = (
        (raw["time"].max() - raw["time"].min()).total_seconds()
        if raw["time"].notna().any()
        else 0.0
    )
    observed = raw.dropna().sort_values("time").drop_duplicates("time")
    dt = observed["time"].diff().dt.total_seconds()
    valid = dt.gt(0) & dt.le(maximum_gap_seconds)
    energy = (
        ((observed["power"] + observed["power"].shift()) / 2 * dt / 3600).where(valid, 0.0).sum()
    )
    coverage = float(dt.where(valid, 0.0).sum() / span) if span > 0 else 0.0
    return float(energy), coverage


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
    result = curves.merge(
        catalog[["cycle_name", "experiment_id"]], on="cycle_name", how="left"
    ).merge(estimates, on="experiment_id", how="left")
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
