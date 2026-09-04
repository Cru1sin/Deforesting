"""Measured heating heat and frozen V2.5 transition heat."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from defrost_decision.candidate_times import cycle_boundaries

from .electricity import (
    MINIMUM_COVERAGE,
    _historical_pressure,
    _strict_pressure,
    integrate_heating_curve,
    load_parameters,
)


def heating_heat(
    frame: pd.DataFrame,
    boundaries: pd.DataFrame,
    basis: str,
    integration_protocol: str = "historical_reconstruction",
    *,
    historical_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Integrate clipped unit heat or signed water heat from the recipe boundary."""
    if basis == "unit":
        power = pd.to_numeric(frame["heating_capacity"], errors="coerce").clip(lower=0)
        model = "measured_unit_heat"
    elif basis == "water":
        power = (
            1.161
            * pd.to_numeric(frame["water_flow"], errors="coerce")
            * (
                pd.to_numeric(frame["water_out_temperature"], errors="coerce")
                - pd.to_numeric(frame["water_in_temperature"], errors="coerce")
            )
        )
        model = "measured_water_heat"
    else:
        raise ValueError("heat basis must be unit or water")
    start = pd.Timestamp(boundaries["heating_accounting_start"].iloc[0])
    curve = integrate_heating_curve(
        frame["timestamp"],
        power,
        boundaries["candidate_defrost_time"],
        start,
        integration_protocol,
        historical_start,
    )
    coverage = curve["coverage"]
    supported = coverage.ge(MINIMUM_COVERAGE)
    strict_supported = curve["strict_coverage"].ge(MINIMUM_COVERAGE)
    start_rule = str(boundaries["heating_accounting_start_rule"].iloc[0])
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
            "pre_defrost_heat_kwh": curve["energy_kwh"].to_numpy(),
            "heating_heat_coverage": coverage.to_numpy(),
            "heating_heat_supported": supported.to_numpy(),
            "strict_pre_defrost_heat_kwh": curve["strict_energy_kwh"].to_numpy(),
            "strict_heating_heat_coverage": curve["strict_coverage"].to_numpy(),
            "strict_heating_heat_supported": strict_supported.to_numpy(),
            "heating_heat_model": model,
            "heating_heat_rule": rule,
            "heating_heat_status": np.where(supported, "supported", "incomplete"),
            "strict_heating_heat_status": np.where(strict_supported, "supported", "incomplete"),
        }
    )


def zero_transition_heat(count: int) -> pd.DataFrame:
    """Return the V1 QT=0 block."""
    return pd.DataFrame(
        {
            "defrost_event_net_heat_kwh": np.zeros(count),
            "preparation_heat_kwh": np.zeros(count),
            "defrost_heat_kwh": np.zeros(count),
            "recovery_heat_kwh": np.zeros(count),
            "defrost_event_net_heat_evaluable": np.ones(count, dtype=bool),
            "defrost_event_net_heat_in_training_domain": np.ones(count, dtype=bool),
            "QT_physical_valid": np.ones(count, dtype=bool),
            "strict_QT_supported": np.ones(count, dtype=bool),
            "strict_defrost_event_net_heat_kwh": np.zeros(count),
            "strict_preparation_heat_kwh": np.zeros(count),
            "strict_defrost_heat_kwh": np.zeros(count),
            "transition_heat_model": "zero_transition_heat",
            "transition_heat_rule": "none",
            "transition_heat_status": "supported",
        }
    )


def _candidate_states(
    frame: pd.DataFrame, end: pd.Timestamp, state_protocol: str
) -> tuple[dict[str, float], dict[str, int]]:
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    window = frame.loc[timestamps.ge(end - pd.Timedelta(seconds=60)) & timestamps.lt(end)]
    result = {}
    complete_seconds = {}
    for column in (
        "water_in_temperature",
        "water_out_temperature",
        "coil_temperature",
        "evaporating_pressure",
    ):
        values = pd.to_numeric(window[column], errors="coerce")
        finite = pd.DataFrame(
            {"timestamp": timestamps.loc[window.index].dt.floor("s"), column: values}
        )
        complete_seconds[column] = len(finite.dropna().drop_duplicates("timestamp"))
        values = values.dropna()
        result[column] = float(values.median()) if not values.empty else float("nan")
    if state_protocol == "historical_interpolation":
        result["evaporating_pressure"] = _historical_pressure(frame, end)
    elif state_protocol == "strict_causal":
        result["evaporating_pressure"] = _strict_pressure(frame, end)[0]
    else:
        raise ValueError("state protocol must be historical_interpolation or strict_causal")
    return result, complete_seconds


def _linear_prediction(states: pd.DataFrame, model: Mapping[str, object]) -> pd.Series:
    raw_coefficients = model["coefficients"]
    feature_order = model["feature_order"]
    if not isinstance(raw_coefficients, list) or not isinstance(feature_order, list):
        raise ValueError("linear model parameters must be lists")
    coefficients = [float(value) for value in raw_coefficients]
    prediction = pd.Series(coefficients[0], index=states.index)
    for name, coefficient in zip(feature_order, coefficients[1:], strict=True):
        prediction += coefficient * states[str(name)]
    return prediction


def _quadratic_prediction(states: pd.DataFrame, model: Mapping[str, object]) -> pd.Series:
    coefficients = model["coefficients"]
    feature_order = model["feature_order"]
    if not isinstance(coefficients, Mapping) or not isinstance(feature_order, list):
        raise ValueError("quadratic model parameters are malformed")
    linear = coefficients.get("linear")
    quadratic = coefficients.get("quadratic")
    if not isinstance(linear, list) or not isinstance(quadratic, list):
        raise ValueError("quadratic model coefficients must be lists")
    prediction = pd.Series(float(coefficients["intercept"]), index=states.index)
    for name, linear_coefficient, quadratic_coefficient in zip(
        feature_order,
        linear,
        quadratic,
        strict=True,
    ):
        values = states[str(name)]
        prediction += float(linear_coefficient) * values + float(
            quadratic_coefficient
        ) * values.pow(2)
    return prediction


def _rule_duration(frame: pd.DataFrame, record: Mapping[str, object]) -> float:
    events = cycle_boundaries(record)
    start, end = events["defrost_start"], events["defrost_end"]
    raw = frame[["timestamp", "coil_temperature"]].copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce").dt.floor("s")
    raw["coil_temperature"] = pd.to_numeric(raw["coil_temperature"], errors="coerce")
    raw = raw.loc[raw["timestamp"].ge(start) & raw["timestamp"].lt(end)]
    raw = raw.dropna(subset=["timestamp"]).drop_duplicates("timestamp").set_index("timestamp")
    grid = pd.date_range(start.ceil("s"), end.ceil("s"), freq="s", inclusive="left")
    values = raw["coil_temperature"].reindex(raw.index.union(grid).sort_values())
    reached = np.flatnonzero(
        values.interpolate(method="time", limit_area="inside").reindex(grid).ge(20).to_numpy()
    )
    if not len(reached):
        raise ValueError("cannot reconstruct observed defrost rule duration")
    return min(int(reached[0]) + 40, 350) / 60


def transition_heat_v2_5(
    frame: pd.DataFrame,
    boundaries: pd.DataFrame,
    record: Mapping[str, object],
    state_protocol: str = "historical_interpolation",
) -> pd.DataFrame:
    """Predict linear Qprep plus signed-negative quadratic QD, with QR=0."""
    parameters = load_parameters()["v2.5"]
    qprep_model = parameters["preparation_heat"]
    qd_model = parameters["defrost_heat"]
    events = cycle_boundaries(record)
    preparation_duration = (
        events["defrost_start"] - events["defrost_preparation_start"]
    ).total_seconds() / 60
    duration = _rule_duration(frame, record)
    candidate_defrost_times = [
        pd.Timestamp(value) for value in boundaries["candidate_defrost_time"]
    ]
    state_rows = [
        _candidate_states(frame, value, state_protocol) for value in candidate_defrost_times
    ]
    strict_rows = (
        state_rows
        if state_protocol == "strict_causal"
        else [_candidate_states(frame, value, "strict_causal") for value in candidate_defrost_times]
    )
    states = pd.DataFrame([values for values, _ in state_rows])
    strict_states = pd.DataFrame([values for values, _ in strict_rows])
    strict_counts = pd.DataFrame([counts for _, counts in strict_rows])
    strict_window_supported = strict_counts.min(axis=1).ge(48)
    states["preparation_duration_minutes"] = preparation_duration
    states["rule_defrost_duration_minutes"] = duration
    strict_states["preparation_duration_minutes"] = preparation_duration
    strict_states["rule_defrost_duration_minutes"] = duration
    qprep = _linear_prediction(states, qprep_model)
    qd = _quadratic_prediction(states, qd_model)
    strict_qprep = _linear_prediction(strict_states, qprep_model)
    strict_qd = _quadratic_prediction(strict_states, qd_model)
    qprep_supported = pd.Series(True, index=states.index)
    for name, limits in qprep_model["support"].items():
        qprep_supported &= states[name].between(float(limits[0]), float(limits[1]))
    qd_supported = pd.Series(True, index=states.index)
    for name, limits in qd_model["support"].items():
        qd_supported &= states[name].between(float(limits[0]), float(limits[1]))
    signed_qd = -qd
    physical = qprep.gt(0) & signed_qd.lt(0)
    evaluable = (
        qprep.notna() & qd.notna()
        if state_protocol == "historical_interpolation"
        else qprep.notna() & qd.notna() & strict_window_supported
    )
    in_support = evaluable & qprep_supported & qd_supported
    strict_signed_qd = -strict_qd
    strict_supported = strict_window_supported & strict_qprep.gt(0) & strict_signed_qd.lt(0)
    result = states.rename(columns={"evaporating_pressure": "qd_evaporating_pressure_mpa"})
    result["defrost_event_net_heat_kwh"] = qprep + signed_qd
    result["preparation_heat_kwh"] = qprep
    result["defrost_heat_kwh"] = signed_qd
    result["recovery_heat_kwh"] = 0.0
    result["qprep_supported"] = qprep_supported
    result["qd_supported"] = qd_supported
    result["defrost_event_net_heat_evaluable"] = evaluable
    result["defrost_event_net_heat_in_training_domain"] = in_support
    result["QT_physical_valid"] = physical
    result["strict_defrost_event_net_heat_kwh"] = strict_qprep + strict_signed_qd
    result["strict_preparation_heat_kwh"] = strict_qprep
    result["strict_defrost_heat_kwh"] = strict_signed_qd
    for column in strict_counts:
        result[f"strict_{column}_complete_seconds"] = strict_counts[column]
    result["strict_state_window_supported"] = strict_window_supported
    result["strict_QT_supported"] = strict_supported
    result["transition_heat_model"] = "linear_qprep_plus_signed_quadratic_qd"
    result["transition_heat_rule"] = (
        "offline_historical_interpolation_[tau-60s,tau)"
        if state_protocol == "historical_interpolation"
        else "strict_causal_[tau-60s,tau)"
    )
    result["transition_heat_status"] = np.select(
        [~evaluable, ~physical, ~in_support],
        ["incomplete", "physical_invalid", "outside_empirical_support"],
        default="supported",
    )
    result["strict_transition_heat_status"] = np.where(strict_supported, "supported", "incomplete")
    return result
