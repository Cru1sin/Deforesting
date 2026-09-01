"""Measured heating electricity and frozen empirical transition electricity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    curve = integrate_heating_curve(
        frame["timestamp"],
        pd.to_numeric(frame["power_total"], errors="coerce"),
        boundaries["candidate_time"],
        start,
    )
    coverage = curve["coverage"]
    supported = coverage.ge(MINIMUM_COVERAGE)
    return pd.DataFrame(
        {
            "heating_energy_kwh": curve["energy_kwh"].to_numpy(),
            "heating_energy_legacy_bridged_kwh": curve["legacy_bridged_energy_kwh"].to_numpy(),
            "heating_energy_coverage": coverage.to_numpy(),
            "heating_energy_supported": supported.to_numpy(),
            "heating_energy_model": "measured_total_power",
            "heating_energy_rule": str(boundaries["integration_start_rule"].iloc[0]),
            "heating_energy_status": np.where(supported, "supported", "incomplete"),
        }
    )


def integrate_heating_curve(
    timestamps: pd.Series,
    power_kw: pd.Series,
    candidates: pd.Series,
    start: pd.Timestamp,
) -> pd.DataFrame:
    """Causally integrate each candidate using only valid signal observations through tau."""
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, errors="coerce"),
            "power_kw": pd.to_numeric(power_kw, errors="coerce"),
        }
    ).sort_values("timestamp", kind="stable")
    raw = raw.drop_duplicates("timestamp", keep="last")
    rows = []
    for candidate in pd.to_datetime(candidates, errors="coerce"):
        values = raw.loc[raw["timestamp"].ge(start) & raw["timestamp"].le(candidate)].dropna()
        dt = values["timestamp"].diff().dt.total_seconds()
        valid = dt.gt(0) & dt.le(5)
        segments = (values["power_kw"] + values["power_kw"].shift()) / 2 * dt / 3600
        energy = segments.where(valid, 0.0).sum()
        legacy_bridged = segments.where(dt.gt(0), 0.0).sum()
        required = (pd.Timestamp(candidate) - start).total_seconds()
        covered = dt.where(valid, 0.0).sum()
        rows.append(
            {
                "energy_kwh": float(energy),
                "legacy_bridged_energy_kwh": float(legacy_bridged),
                "coverage": float(covered / required) if required > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _strict_pressure(frame: pd.DataFrame, end: pd.Timestamp) -> tuple[float, int]:
    values = frame[["timestamp", "evaporating_pressure"]].copy()
    values["timestamp"] = pd.to_datetime(values["timestamp"], errors="coerce")
    values["evaporating_pressure"] = pd.to_numeric(values["evaporating_pressure"], errors="coerce")
    start = end - pd.Timedelta(seconds=60)
    values = values.loc[values["timestamp"].ge(start) & values["timestamp"].lt(end)]
    complete = values.dropna().copy()
    complete["timestamp"] = complete["timestamp"].dt.floor("s")
    complete_seconds = len(complete.drop_duplicates("timestamp"))
    values = values.dropna(subset=["timestamp"]).drop_duplicates("timestamp").set_index("timestamp")
    grid = pd.date_range(start, periods=60, freq="s")
    interpolated = values["evaporating_pressure"].reindex(values.index.union(grid).sort_values())
    interpolated = interpolated.interpolate(method="time", limit_area="inside").reindex(grid)
    median = float(interpolated.median()) if interpolated.notna().any() else float("nan")
    return median, complete_seconds


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
    features = [
        _strict_pressure(frame, pd.Timestamp(value)) for value in boundaries["candidate_time"]
    ]
    pe = pd.Series([value for value, _ in features])
    complete_seconds = pd.Series([count for _, count in features])
    defrost = coefficients[0] + coefficients[1] * pe + coefficients[2] * pe.pow(2)
    recovery = (
        float(parameters["v1"]["fixed_recovery_electricity_kwh"]) if include_fixed_recovery else 0.0
    )
    status = np.select(
        [pe.isna(), complete_seconds.lt(48), pe.lt(lower), pe.gt(upper)],
        ["missing", "incomplete", "below_support", "above_support"],
        default="supported",
    )
    return pd.DataFrame(
        {
            "transition_energy_kwh": defrost + recovery,
            "preparation_energy_kwh": 0.0,
            "defrost_energy_kwh": defrost,
            "recovery_energy_kwh": recovery,
            "evaporating_pressure_mpa": pe,
            "pe_complete_seconds": complete_seconds,
            "pe_quadratic_intercept_kwh": coefficients[0],
            "pe_quadratic_linear_kwh_per_mpa": coefficients[1],
            "pe_quadratic_squared_kwh_per_mpa2": coefficients[2],
            "pe_support_min_mpa": lower,
            "pe_support_max_mpa": upper,
            "ET_supported": pe.between(lower, upper) & complete_seconds.ge(48),
            "transition_energy_model": (
                "pe_quadratic_plus_fixed_recovery" if include_fixed_recovery else "pe_quadratic"
            ),
            "transition_energy_rule": "strict_pre_action_window_[tau-60s,tau)",
            "transition_energy_status": status,
        }
    )
