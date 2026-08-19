"""Minimal empirical defrost-cost calculations on unsmoothed measurements."""

from __future__ import annotations

from typing import Any

import pandas as pd


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
