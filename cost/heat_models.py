"""Measured heating heat and frozen V2.5 transition heat."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .boundaries import cycle_boundaries
from .energy_models import MINIMUM_COVERAGE, integrate_heating_curve, load_parameters


def heating_heat(frame: pd.DataFrame, boundaries: pd.DataFrame, basis: str) -> pd.DataFrame:
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
    start = pd.Timestamp(boundaries["integration_start"].iloc[0])
    curve = integrate_heating_curve(
        frame["timestamp"],
        power,
        boundaries["candidate_time"],
        start,
    )
    coverage = curve["coverage"]
    supported = coverage.ge(MINIMUM_COVERAGE)
    return pd.DataFrame(
        {
            "heating_heat_kwh": curve["energy_kwh"].to_numpy(),
            "heating_heat_legacy_bridged_kwh": curve["legacy_bridged_energy_kwh"].to_numpy(),
            "heating_heat_coverage": coverage.to_numpy(),
            "heating_heat_supported": supported.to_numpy(),
            "heating_heat_model": model,
            "heating_heat_rule": str(boundaries["integration_start_rule"].iloc[0]),
            "heating_heat_status": np.where(supported, "supported", "incomplete"),
        }
    )


def zero_transition_heat(count: int) -> pd.DataFrame:
    """Return the V1 QT=0 block."""
    return pd.DataFrame(
        {
            "transition_heat_kwh": np.zeros(count),
            "preparation_heat_kwh": np.zeros(count),
            "defrost_heat_kwh": np.zeros(count),
            "recovery_heat_kwh": np.zeros(count),
            "QT_supported": np.ones(count, dtype=bool),
            "transition_heat_model": "zero_transition_heat",
            "transition_heat_rule": "none",
            "transition_heat_status": "supported",
        }
    )


def _strict_states(frame: pd.DataFrame, end: pd.Timestamp) -> tuple[dict[str, float], bool]:
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    window = frame.loc[timestamps.ge(end - pd.Timedelta(seconds=60)) & timestamps.lt(end)]
    result = {}
    complete_seconds = []
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
        complete_seconds.append(len(finite.dropna().drop_duplicates("timestamp")))
        values = values.dropna()
        result[column] = float(values.median()) if not values.empty else float("nan")
    return result, min(complete_seconds) >= 48


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
    state_rows = [
        _strict_states(frame, pd.Timestamp(value)) for value in boundaries["candidate_time"]
    ]
    states = pd.DataFrame([values for values, _ in state_rows])
    state_window_supported = pd.Series([supported for _, supported in state_rows])
    states["preparation_duration_minutes"] = preparation_duration
    states["rule_defrost_duration_minutes"] = duration
    qprep_features = qprep_model["feature_order"]
    qprep_coefficients = [float(value) for value in qprep_model["coefficients"]]
    qprep = pd.Series(qprep_coefficients[0], index=states.index)
    for name, coefficient in zip(qprep_features, qprep_coefficients[1:], strict=True):
        qprep += coefficient * states[name]
    qd_features = qd_model["feature_order"]
    qd = pd.Series(float(qd_model["coefficients"]["intercept"]), index=states.index)
    for name, linear, quadratic in zip(
        qd_features,
        qd_model["coefficients"]["linear"],
        qd_model["coefficients"]["quadratic"],
        strict=True,
    ):
        qd += float(linear) * states[name] + float(quadratic) * states[name].pow(2)
    qprep_supported = pd.Series(True, index=states.index)
    for name, limits in qprep_model["support"].items():
        qprep_supported &= states[name].between(float(limits[0]), float(limits[1]))
    qd_supported = pd.Series(True, index=states.index)
    for name, limits in qd_model["support"].items():
        qd_supported &= states[name].between(float(limits[0]), float(limits[1]))
    signed_qd = -qd
    supported = (
        state_window_supported & qprep_supported & qd_supported & qprep.gt(0) & signed_qd.le(0)
    )
    result = states.rename(columns={"evaporating_pressure": "qd_evaporating_pressure_mpa"})
    result["transition_heat_kwh"] = qprep + signed_qd
    result["preparation_heat_kwh"] = qprep
    result["defrost_heat_kwh"] = signed_qd
    result["recovery_heat_kwh"] = 0.0
    result["qprep_supported"] = qprep_supported
    result["qd_supported"] = qd_supported
    result["state_window_supported"] = state_window_supported
    result["QT_supported"] = supported
    result["transition_heat_model"] = "linear_qprep_plus_signed_quadratic_qd"
    result["transition_heat_rule"] = "strict_pre_action_window_[tau-60s,tau)"
    result["transition_heat_status"] = np.select(
        [~state_window_supported, supported],
        ["incomplete", "supported"],
        default="outside_support",
    )
    return result
