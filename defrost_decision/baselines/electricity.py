"""Measured heating electricity integration with explicit gap handling."""
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
    candidate_defrost_times: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
    *,
    maximum_gap_seconds: float = 5.0,
    bridge_internal_gaps: bool = False,
    extrapolate_endpoints: bool = False,
    causal_candidates: bool = False,
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
    candidates = pd.DatetimeIndex(pd.to_datetime(candidate_defrost_times, errors="coerce"))
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
        left_slope = (observed_power[1] - observed_power[0]) / (observed_time[1] - observed_time[0])
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
    increments = ((observed["power"] + observed["power"].shift()) / 2 * dt / 3600).where(valid, 0.0)
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
        if not causal_candidates:
            cumulative += np.where(
                inside,
                (left_power + partial_power) / 2 * partial_seconds / 3600,
                0.0,
            )
            covered += partial_seconds
        bridged_candidates = (
            inside.copy() if not causal_candidates else np.zeros(len(candidates), bool)
        )
        for gap_index in np.flatnonzero(bridged_segments):
            bridged_candidates |= (
                candidate_ns >= observed_ns[gap_index]
                if causal_candidates
                else (candidate_ns > observed_ns[gap_index - 1])
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

def integrate_heating_curve(
    timestamps: pd.Series,
    power_kw: pd.Series,
    candidates: pd.Series,
    start: pd.Timestamp,
    integration_protocol: str = "historical_reconstruction",
    historical_start: pd.Timestamp | None = None,
    *,
    causal_candidates: bool = False,
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
            extrapolate_endpoints=not causal_candidates,
            causal_candidates=causal_candidates,
        )[["energy_kwh", "coverage", "bridged_internal_gap"]]
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
