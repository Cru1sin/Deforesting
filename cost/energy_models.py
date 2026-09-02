"""Measured heating electricity and frozen empirical transition electricity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

MINIMUM_COVERAGE = 0.95


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


def integrate_energy_curve_kwh(
    timestamps: pd.Series | pd.DatetimeIndex,
    power_kw: pd.Series,
    candidate_times: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
    *,
    maximum_gap_seconds: float = 5.0,
    bridge_internal_gaps: bool = False,
    extrapolate_endpoints: bool = False,
) -> pd.DataFrame:
    """Return gap-aware cumulative energy and coverage at many candidate times."""
    raw = pd.DataFrame(
        {
            "time": pd.to_datetime(timestamps, errors="coerce"),
            "power": pd.to_numeric(power_kw, errors="coerce"),
        }
    )
    raw_time = pd.DatetimeIndex(raw["time"].dropna().sort_values().drop_duplicates())
    observed = raw.dropna().sort_values("time").drop_duplicates("time")
    candidates = pd.DatetimeIndex(pd.to_datetime(candidate_times, errors="coerce"))
    if observed.empty or raw_time.empty:
        return pd.DataFrame(
            {
                "energy_kwh": 0.0,
                "coverage": 0.0,
                "bridged_internal_gap": False,
                "extrapolated_endpoint": False,
            },
            index=range(len(candidates)),
        )
    endpoint_rows: NDArray[np.bool_] = np.zeros(len(raw), dtype=bool)
    endpoint_ranges: list[tuple[pd.Timestamp, pd.Timestamp, bool]] = []
    if extrapolate_endpoints and len(observed) >= 2:
        first_observed_time = pd.Timestamp(observed["time"].iloc[0])
        last_observed_time = pd.Timestamp(observed["time"].iloc[-1])
        observed_time = observed["time"].astype("int64").to_numpy(dtype=float)
        observed_power = observed["power"].to_numpy(dtype=float)
        left_slope = (observed_power[1] - observed_power[0]) / (
            observed_time[1] - observed_time[0]
        )
        right_slope = (observed_power[-1] - observed_power[-2]) / (
            observed_time[-1] - observed_time[-2]
        )
        raw_time_ns = raw["time"].astype("int64").to_numpy(dtype=float)
        left = raw["time"].lt(observed["time"].iloc[0]) & raw["power"].isna()
        right = raw["time"].gt(observed["time"].iloc[-1]) & raw["power"].isna()
        raw.loc[left, "power"] = observed_power[0] + left_slope * (
            raw_time_ns[left.to_numpy()] - observed_time[0]
        )
        raw.loc[right, "power"] = observed_power[-1] + right_slope * (
            raw_time_ns[right.to_numpy()] - observed_time[-1]
        )
        endpoint_rows = (left | right).to_numpy()
        if left.any():
            endpoint_ranges.append((pd.Timestamp(raw_time[0]), first_observed_time, False))
        if right.any():
            endpoint_ranges.append((last_observed_time, pd.Timestamp(raw_time[-1]), True))
        observed = raw.dropna().sort_values("time").drop_duplicates("time")
    dt = observed["time"].diff().dt.total_seconds()
    short = dt.gt(0) & dt.le(maximum_gap_seconds)
    bridged = bridge_internal_gaps & dt.gt(maximum_gap_seconds)
    valid = short | bridged
    increments = (
        (observed["power"] + observed["power"].shift()) / 2 * dt / 3600
    ).where(valid, 0.0)
    energy = increments.cumsum().to_numpy()
    covered_seconds = dt.where(valid, 0.0).cumsum().to_numpy()
    observed_index = pd.DatetimeIndex(observed["time"])
    positions: NDArray[np.intp] = observed_index.searchsorted(candidates, side="right") - 1
    raw_positions: NDArray[np.intp] = raw_time.searchsorted(candidates, side="right") - 1
    safe_positions = np.maximum(positions, 0)
    spans = np.where(
        raw_positions >= 0,
        (raw_time[np.maximum(raw_positions, 0)] - raw_time[0]).total_seconds(),
        0.0,
    )
    cumulative = np.where(positions >= 0, energy[safe_positions], 0.0)
    covered = np.where(positions >= 0, covered_seconds[safe_positions], 0.0)
    bridged_candidates: NDArray[np.bool_] = np.zeros(len(candidates), dtype=bool)
    extrapolated_candidates: NDArray[np.bool_] = np.zeros(len(candidates), dtype=bool)
    if bridge_internal_gaps:
        candidate_ns = candidates.view("i8")
        observed_ns = observed_index.view("i8")
        next_positions: NDArray[np.intp] = positions + 1
        bridged_segments = bridged.to_numpy()
        inside = (
            (positions >= 0)
            & (next_positions < len(observed_index))
            & (candidate_ns > observed_ns[np.maximum(positions, 0)])
            & (candidate_ns < observed_ns[np.minimum(next_positions, len(observed_index) - 1)])
            & bridged_segments[np.minimum(next_positions, len(observed_index) - 1)]
        )
        left = np.maximum(positions, 0)
        right = np.minimum(next_positions, len(observed_index) - 1)
        partial_seconds = np.where(
            inside,
            (candidate_ns - observed_ns[left]) / 1e9,
            0.0,
        )
        segment_seconds = dt.to_numpy()[right]
        left_power = observed["power"].to_numpy()[left]
        right_power = observed["power"].to_numpy()[right]
        fraction = np.divide(
            partial_seconds,
            segment_seconds,
            out=np.zeros_like(partial_seconds),
            where=segment_seconds > 0,
        )
        partial_power = left_power + (right_power - left_power) * fraction
        cumulative += np.where(
            inside,
            (left_power + partial_power) / 2 * partial_seconds / 3600,
            0.0,
        )
        covered += partial_seconds
        bridged_candidates = inside.copy()
        for gap_index in np.flatnonzero(bridged_segments):
            bridged_candidates |= (
                (candidate_ns > observed_ns[gap_index - 1])
                & (candidate_ns < observed_ns[gap_index])
            )
        spans = np.where(
            candidates >= raw_time[0],
            (candidates - raw_time[0]).total_seconds(),
            0.0,
        )
    if extrapolate_endpoints and endpoint_rows.any():
        for endpoint_left, endpoint_right, right_inclusive in endpoint_ranges:
            if right_inclusive:
                extrapolated_candidates |= (candidates > endpoint_left) & (
                    candidates <= endpoint_right
                )
            else:
                extrapolated_candidates |= (candidates >= endpoint_left) & (
                    candidates < endpoint_right
                )
    coverage = np.divide(covered, spans, out=np.zeros_like(covered), where=spans > 0)
    return pd.DataFrame(
        {
            "energy_kwh": cumulative,
            "coverage": coverage,
            "bridged_internal_gap": bridged_candidates,
            "extrapolated_endpoint": extrapolated_candidates,
        }
    )


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
