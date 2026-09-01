"""Measured heating electricity and frozen empirical transition electricity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.frost_analysis.cost.core import integrate_energy_curve_kwh

MINIMUM_COVERAGE = 0.95


def load_parameters() -> dict[str, Any]:
    """Load the checked-in empirical parameters."""
    path = Path(__file__).with_name("params") / "empirical_models.json"
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("empirical_models.json must contain an object")
    return value


def heating_energy(frame: pd.DataFrame, boundaries: pd.DataFrame) -> pd.DataFrame:
    """Integrate measured total power from the recipe's heating boundary."""
    start = pd.Timestamp(boundaries["integration_start"].iloc[0])
    end = pd.Timestamp(boundaries["candidate_time"].max())
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    source = frame.loc[timestamps.ge(start) & timestamps.le(end)]
    curve = integrate_energy_curve_kwh(
        source["timestamp"],
        pd.to_numeric(source["power_total"], errors="coerce"),
        boundaries["candidate_time"],
        bridge_internal_gaps=True,
        extrapolate_endpoints=True,
    )
    coverage = _anchored_coverage(
        source["timestamp"], start, boundaries["candidate_time"], curve["coverage"]
    )
    supported = coverage.ge(MINIMUM_COVERAGE)
    return pd.DataFrame(
        {
            "heating_energy_kwh": curve["energy_kwh"].to_numpy(),
            "heating_energy_coverage": coverage.to_numpy(),
            "heating_energy_supported": supported.to_numpy(),
            "heating_energy_model": "measured_total_power",
            "heating_energy_rule": str(boundaries["integration_start"].iloc[0]),
            "heating_energy_status": np.where(supported, "supported", "incomplete"),
        }
    )


def _anchored_coverage(
    timestamps: pd.Series,
    start: pd.Timestamp,
    candidates: pd.Series,
    coverage: pd.Series,
) -> pd.Series:
    valid = pd.to_datetime(timestamps, errors="coerce").dropna()
    if valid.empty:
        return pd.Series(0.0, index=coverage.index)
    first = max(pd.Timestamp(valid.min()), start)
    candidate_times = pd.to_datetime(candidates, errors="coerce")
    required = (candidate_times - start).dt.total_seconds()
    observed = (candidate_times - first).dt.total_seconds().clip(lower=0)
    fraction = observed.div(required).where(required.gt(0), 0.0).clip(upper=1)
    return coverage.reset_index(drop=True) * fraction.reset_index(drop=True)


def _strict_pressure(frame: pd.DataFrame, end: pd.Timestamp) -> float:
    values = frame[["timestamp", "evaporating_pressure"]].copy()
    values["timestamp"] = pd.to_datetime(values["timestamp"], errors="coerce")
    values["evaporating_pressure"] = pd.to_numeric(values["evaporating_pressure"], errors="coerce")
    start = end - pd.Timedelta(seconds=60)
    values = values.loc[values["timestamp"].ge(start) & values["timestamp"].lt(end)]
    values = values.dropna(subset=["timestamp"]).drop_duplicates("timestamp").set_index("timestamp")
    grid = pd.date_range(start, periods=60, freq="s")
    interpolated = values["evaporating_pressure"].reindex(values.index.union(grid).sort_values())
    interpolated = interpolated.interpolate(method="time", limit_area="inside").reindex(grid)
    return float(interpolated.median()) if interpolated.notna().any() else float("nan")


def transition_energy(
    frame: pd.DataFrame,
    boundaries: pd.DataFrame,
    experiment_id: str,
    *,
    include_fixed_recovery: bool,
) -> pd.DataFrame:
    """Predict ED from strict-window Pe and optionally add V1 fixed recovery."""
    parameters = load_parameters()
    try:
        model = parameters["pe_quadratic"][experiment_id]
    except KeyError as exc:
        raise ValueError(f"no Pe quadratic parameters for {experiment_id}") from exc
    coefficients = [float(value) for value in model["coefficients"]]
    lower, upper = (float(value) for value in model["support"])
    pe = pd.Series(
        [_strict_pressure(frame, pd.Timestamp(value)) for value in boundaries["candidate_time"]]
    )
    defrost = coefficients[0] + coefficients[1] * pe + coefficients[2] * pe.pow(2)
    recovery = (
        float(parameters["v1"]["fixed_recovery_electricity_kwh"]) if include_fixed_recovery else 0.0
    )
    status = np.select(
        [pe.isna(), pe.lt(lower), pe.gt(upper)],
        ["missing", "below_support", "above_support"],
        default="supported",
    )
    return pd.DataFrame(
        {
            "transition_energy_kwh": defrost + recovery,
            "defrost_electricity_kwh": defrost,
            "recovery_electricity_kwh": recovery,
            "evaporating_pressure_mpa": pe,
            "pe_quadratic_intercept_kwh": coefficients[0],
            "pe_quadratic_linear_kwh_per_mpa": coefficients[1],
            "pe_quadratic_squared_kwh_per_mpa2": coefficients[2],
            "pe_support_min_mpa": lower,
            "pe_support_max_mpa": upper,
            "ET_supported": pe.between(lower, upper),
            "transition_energy_model": (
                "pe_quadratic_plus_fixed_recovery" if include_fixed_recovery else "pe_quadratic"
            ),
            "transition_energy_rule": "strict_pre_action_window_[tau-60s,tau)",
            "transition_energy_status": status,
        }
    )
