"""Measured heating electricity and frozen empirical transition electricity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.frost_analysis.cost.core import integrate_energy_curve_kwh, integrate_energy_kwh

MINIMUM_COVERAGE = 0.95


def load_parameters() -> dict[str, Any]:
    """Load the checked-in empirical parameters."""
    path = Path(__file__).with_name("params") / "empirical_models.json"
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("empirical_models.json must contain an object")
    return value


def heating_energy(
    frame: pd.DataFrame,
    boundaries: pd.DataFrame,
    integration_protocol: str = "historical_reconstruction",
    *,
    historical_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Integrate measured total power from the recipe's heating boundary."""
    start = pd.Timestamp(boundaries["integration_start"].iloc[0])
    curve = integrate_heating_curve(
        frame["timestamp"],
        pd.to_numeric(frame["power_total"], errors="coerce"),
        boundaries["candidate_time"],
        start,
        integration_protocol,
        historical_start,
    )
    coverage = curve["coverage"]
    supported = coverage.ge(MINIMUM_COVERAGE)
    strict_supported = curve["strict_coverage"].ge(MINIMUM_COVERAGE)
    start_rule = str(boundaries["integration_start_rule"].iloc[0])
    if integration_protocol == "historical_reconstruction":
        rule = (
            "offline_historical_reconstruction_stable_block_bridged_internal_gaps_"
            "endpoint_extrapolation_plus_bridged_observed_heating_start_prefix"
            if historical_start is not None and historical_start != start
            else "offline_historical_reconstruction_bridged_internal_gaps_"
            f"endpoint_extrapolation_from_{start_rule}"
        )
    else:
        rule = f"strict_causal_gap_aware_5s_from_{start_rule}"
    return pd.DataFrame(
        {
            "heating_energy_kwh": curve["energy_kwh"].to_numpy(),
            "heating_energy_coverage": coverage.to_numpy(),
            "heating_energy_supported": supported.to_numpy(),
            "strict_heating_energy_kwh": curve["strict_energy_kwh"].to_numpy(),
            "strict_heating_energy_coverage": curve["strict_coverage"].to_numpy(),
            "strict_heating_energy_supported": strict_supported.to_numpy(),
            "heating_energy_model": "measured_total_power",
            "heating_energy_rule": rule,
            "heating_energy_status": np.where(supported, "supported", "incomplete"),
            "strict_heating_energy_status": np.where(strict_supported, "supported", "incomplete"),
        }
    )


def integrate_heating_curve(
    timestamps: pd.Series,
    power_kw: pd.Series,
    candidates: pd.Series,
    start: pd.Timestamp,
    integration_protocol: str = "historical_reconstruction",
    historical_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return the selected heating integral plus the causal strict diagnostic.

    A distinct historical start reproduces the formal mixed V2.5 rule: its stable block
    extrapolates endpoints, while the observed heating-start prefix only bridges gaps.
    """
    strict_curve = _strict_heating_curve(timestamps, power_kw, candidates, start)
    if integration_protocol == "strict_causal":
        selected = strict_curve
    elif integration_protocol == "historical_reconstruction":
        base_start = start if historical_start is None else historical_start
        parsed = pd.to_datetime(timestamps, errors="coerce")
        end = pd.Timestamp(pd.to_datetime(candidates, errors="coerce").max())
        source = parsed.ge(base_start) & parsed.le(end)
        selected = integrate_energy_curve_kwh(
            parsed.loc[source],
            pd.to_numeric(power_kw.loc[source], errors="coerce"),
            candidates,
            bridge_internal_gaps=True,
            extrapolate_endpoints=True,
        )[["energy_kwh", "coverage"]]
        if base_start != start:
            prefix = parsed.between(min(start, base_start), max(start, base_start))
            adjustment, coverage = integrate_energy_kwh(
                parsed.loc[prefix],
                pd.to_numeric(power_kw.loc[prefix], errors="coerce"),
                maximum_gap_seconds=np.inf,
            )
            if coverage < MINIMUM_COVERAGE:
                raise ValueError("historical heating prefix is incomplete")
            selected["energy_kwh"] += adjustment if start < base_start else -adjustment
    else:
        raise ValueError("integration protocol must be historical_reconstruction or strict_causal")
    return selected.assign(
        strict_energy_kwh=strict_curve["energy_kwh"].to_numpy(),
        strict_coverage=strict_curve["coverage"].to_numpy(),
    )


def _strict_heating_curve(
    timestamps: pd.Series,
    power_kw: pd.Series,
    candidates: pd.Series,
    start: pd.Timestamp,
) -> pd.DataFrame:
    """Causally integrate valid adjacent observations no more than five seconds apart."""
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
        required = (pd.Timestamp(candidate) - start).total_seconds()
        covered = dt.where(valid, 0.0).sum()
        rows.append(
            {
                "energy_kwh": float(energy),
                "coverage": float(covered / required) if required > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _historical_pressure(frame: pd.DataFrame, end: pd.Timestamp) -> float:
    """Reproduce the formal whole-cycle interpolation on the pre-action grid."""
    values = frame[["timestamp", "evaporating_pressure"]].copy()
    values["timestamp"] = pd.to_datetime(values["timestamp"], errors="coerce")
    values["evaporating_pressure"] = pd.to_numeric(values["evaporating_pressure"], errors="coerce")
    values = (
        values.dropna(subset=["timestamp"])
        .sort_values("timestamp", kind="stable")
        .drop_duplicates("timestamp")
        .set_index("timestamp")
    )
    grid = pd.date_range(end - pd.Timedelta(seconds=60), periods=60, freq="s")
    interpolated = values["evaporating_pressure"].reindex(values.index.union(grid).sort_values())
    interpolated = interpolated.interpolate(method="time", limit_area="inside").reindex(grid)
    finite = values["evaporating_pressure"].dropna().sort_index()
    if len(finite) >= 2:
        finite_time = finite.index
        finite_values = finite.to_numpy(dtype=float)
        left_slope = (finite_values[1] - finite_values[0]) / (
            finite_time[1] - finite_time[0]
        ).total_seconds()
        right_slope = (finite_values[-1] - finite_values[-2]) / (
            finite_time[-1] - finite_time[-2]
        ).total_seconds()
        left = grid < finite_time[0]
        right = grid > finite_time[-1]
        grid_ns: Any = grid.view("i8").astype(float)
        interpolated.loc[left] = (
            finite_values[0] + left_slope * (grid_ns[left] - finite_time[0].value) / 1e9
        )
        interpolated.loc[right] = (
            finite_values[-1] + right_slope * (grid_ns[right] - finite_time[-1].value) / 1e9
        )
    return float(interpolated.median()) if interpolated.notna().any() else float("nan")


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
    state_protocol: str = "historical_interpolation",
) -> pd.DataFrame:
    """Predict ED from the selected Pe protocol and emit strict causal diagnostics."""
    parameters = load_parameters()
    try:
        model = parameters["pe_quadratic"][experiment_id]
    except KeyError as exc:
        raise ValueError(f"no Pe quadratic parameters for {experiment_id}") from exc
    coefficients = [float(value) for value in model["coefficients"]]
    lower, upper = (float(value) for value in model["support"])
    strict_features = [
        _strict_pressure(frame, pd.Timestamp(value)) for value in boundaries["candidate_time"]
    ]
    strict_pe = pd.Series([value for value, _ in strict_features])
    complete_seconds = pd.Series([count for _, count in strict_features])
    if state_protocol == "historical_interpolation":
        pe = pd.Series(
            [
                _historical_pressure(frame, pd.Timestamp(value))
                for value in boundaries["candidate_time"]
            ]
        )
    elif state_protocol == "strict_causal":
        pe = strict_pe
    else:
        raise ValueError("state protocol must be historical_interpolation or strict_causal")
    defrost = coefficients[0] + coefficients[1] * pe + coefficients[2] * pe.pow(2)
    strict_defrost = (
        coefficients[0] + coefficients[1] * strict_pe + coefficients[2] * strict_pe.pow(2)
    )
    recovery = (
        float(parameters["v1"]["fixed_recovery_electricity_kwh"]) if include_fixed_recovery else 0.0
    )
    selected_supported = (
        pe.notna()
        if state_protocol == "historical_interpolation"
        else pe.notna() & complete_seconds.ge(48)
    )
    status = np.select(
        [pe.isna(), ~selected_supported, pe.lt(lower), pe.gt(upper)],
        ["missing", "incomplete", "below_support", "above_support"],
        default="supported",
    )
    strict_supported = strict_pe.notna() & complete_seconds.ge(48)
    strict_status = np.select(
        [strict_pe.isna(), ~strict_supported, strict_pe.lt(lower), strict_pe.gt(upper)],
        ["missing", "incomplete", "below_support", "above_support"],
        default="supported",
    )
    rule = (
        "offline_historical_interpolation_[tau-60s,tau)"
        if state_protocol == "historical_interpolation"
        else "strict_causal_[tau-60s,tau)"
    )
    return pd.DataFrame(
        {
            "transition_energy_kwh": defrost + recovery,
            "preparation_energy_kwh": 0.0,
            "defrost_energy_kwh": defrost,
            "recovery_energy_kwh": recovery,
            "evaporating_pressure_mpa": pe,
            "strict_transition_energy_kwh": strict_defrost + recovery,
            "strict_defrost_energy_kwh": strict_defrost,
            "strict_evaporating_pressure_mpa": strict_pe,
            "strict_pe_complete_seconds": complete_seconds,
            "pe_quadratic_intercept_kwh": coefficients[0],
            "pe_quadratic_linear_kwh_per_mpa": coefficients[1],
            "pe_quadratic_squared_kwh_per_mpa2": coefficients[2],
            "pe_support_min_mpa": lower,
            "pe_support_max_mpa": upper,
            "ET_supported": selected_supported,
            "strict_ET_supported": strict_supported,
            "transition_energy_model": (
                "pe_quadratic_plus_fixed_recovery" if include_fixed_recovery else "pe_quadratic"
            ),
            "transition_energy_rule": rule,
            "transition_energy_status": status,
            "strict_transition_energy_status": strict_status,
        }
    )
