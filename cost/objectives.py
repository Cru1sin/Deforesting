"""Independent C, H, and O objective construction for V2.6.8 candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cost_function_v2_6_8 import five_minute_support_runs


def _connected_basin(
    result: pd.DataFrame, prefix: str, optimum: int, eligible: pd.Series, percent: int
) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    values = result[prefix]
    threshold = float(values.iloc[optimum]) - abs(float(values.iloc[optimum])) * percent / 100
    within = eligible & values.ge(threshold)
    left = right = optimum
    while left and bool(within.iloc[left - 1]):
        left -= 1
    while right + 1 < len(result) and bool(within.iloc[right + 1]):
        right += 1
    start, end = result.loc[[left, right], "candidate_time"].map(pd.Timestamp)
    return start, end, (end - start).total_seconds() / 60


def build_objectives(candidate_table: pd.DataFrame) -> pd.DataFrame:
    """Build three mathematically separate maximization objectives and native gates."""
    result = (
        candidate_table.sort_values("candidate_time", kind="stable").reset_index(drop=True).copy()
    )
    elapsed_hours = (
        pd.to_datetime(result["candidate_time"]) - pd.to_datetime(result["integration_start"])
    ).dt.total_seconds() / 3600
    duration_hours = elapsed_hours + pd.to_numeric(
        result["transition_duration_minutes"], errors="coerce"
    ) / 60
    total_heat = result["heating_heat_kwh"] + result["transition_heat_kwh"]
    total_energy = result["heating_energy_kwh"] + result["transition_energy_kwh"]
    net_output = (
        result["heating_heat_kwh"]
        - result["heating_compressor_energy_kwh"]
        + result["transition_heat_kwh"]
        - result["transition_compressor_energy_kwh"]
    )
    result["C"] = total_heat / total_energy
    result["H"] = total_heat / duration_hours
    result["O"] = net_output / duration_hours

    definitions = {
        "C": (("ET", "QT"), ("heating_energy_measurement_valid", "heating_heat_measurement_valid")),
        "H": (("QT", "DT"), ("heating_heat_measurement_valid",)),
        "O": (
            ("QT", "DT", "EcompT"),
            ("heating_heat_measurement_valid", "heating_compressor_measurement_valid"),
        ),
    }
    finite_duration = np.isfinite(duration_hours) & duration_hours.gt(0)
    for prefix, (models, measurements) in definitions.items():
        model = pd.Series(True, index=result.index)
        for name in models:
            model &= result[f"{name}_evaluable"].fillna(False) & result[
                f"{name}_in_support"
            ].fillna(False)
        measurement = pd.Series(True, index=result.index)
        for name in measurements:
            measurement &= result[name].fillna(False)
        if prefix == "C":
            physical = (
                np.isfinite(total_energy)
                & total_energy.gt(0)
                & np.isfinite(total_heat)
                & total_heat.gt(0.01)
            )
        else:
            numerator = total_heat if prefix == "H" else net_output
            physical = finite_duration & np.isfinite(numerator)
        finite = np.isfinite(result[prefix])
        base = (
            model
            & measurement
            & physical
            & result["pre_action_window_valid"].fillna(False)
            & finite
        )
        result[f"{prefix}_model_supported"] = model
        result[f"{prefix}_measurement_eligible"] = measurement
        result[f"{prefix}_physical_valid"] = physical
        result[f"{prefix}_continuous_support"] = five_minute_support_runs(
            result["candidate_time"], base
        )
        result[f"{prefix}_native_eligible"] = base & result[f"{prefix}_continuous_support"]
        result[f"{prefix}_t_star"] = pd.NaT
        for percent in (1, 2, 5):
            result[f"{prefix}_basin_{percent}pct_start"] = pd.NaT
            result[f"{prefix}_basin_{percent}pct_end"] = pd.NaT
            result[f"{prefix}_basin_{percent}pct_width_minutes"] = np.nan
        eligible = result[f"{prefix}_native_eligible"]
        if not eligible.any():
            continue
        optimum = int(
            result.index[eligible & result[prefix].eq(result.loc[eligible, prefix].max())][0]
        )
        result[f"{prefix}_t_star"] = result.loc[optimum, "candidate_time"]
        for percent in (1, 2, 5):
            start, end, width = _connected_basin(result, prefix, optimum, eligible, percent)
            result[f"{prefix}_basin_{percent}pct_start"] = start
            result[f"{prefix}_basin_{percent}pct_end"] = end
            result[f"{prefix}_basin_{percent}pct_width_minutes"] = width
    return result
