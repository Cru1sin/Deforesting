"""Replay the frozen historical rule-based defrost controller."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

RAW_COLUMNS = (
    "timestamp",
    "coil_temperature",
    "ambient_temperature",
    "water_out_temperature",
    "p1__T3o'2_20",
    "p1__DefTim1'2_20",
    "p1__DefTim2'2_20",
)


def _case(ambient_temperature_c: float) -> int:
    if ambient_temperature_c >= -2:
        return 1
    if ambient_temperature_c >= -5:
        return 2
    if ambient_temperature_c >= -8:
        return 3
    if ambient_temperature_c >= -10:
        return 4
    return 5


def limits(
    ambient_temperature_c: float,
    water_out_temperature_c: float,
    initial_coil_temperature_c: float,
) -> tuple[float, float]:
    """Return the historical Case 1--5 T1 and coil-temperature limits."""
    case = _case(ambient_temperature_c)
    time_limits = {
        1: (40, 35, 30),
        2: (40, 38, 33),
        3: (80, 60, 40),
        4: (90, 70, 50),
        5: (150, 120, 90),
    }
    water_bin = 0 if water_out_temperature_c >= 35 else 1 if water_out_temperature_c >= 25 else 2
    coil_limit = (
        initial_coil_temperature_c - (3 if case == 1 else 5)
        if case <= 3
        else ambient_temperature_c - 5
    )
    return time_limits[case][water_bin], coil_limit


def first_trigger(
    frame: pd.DataFrame,
    heating_start: pd.Timestamp,
    observation_end: pd.Timestamp,
    *,
    return_trace: bool = False,
) -> dict[str, object] | pd.DataFrame:
    """Return the first causal RB trigger on the historical unfilled one-second grid."""
    columns = {
        "coil_temperature": "T3_C",
        "ambient_temperature": "T4_C",
        "water_out_temperature": "Twout_C",
        "p1__T3o'2_20": "T3o_C",
        "p1__DefTim1'2_20": "T1_min",
        "p1__DefTim2'2_20": "T2_min",
    }
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    values = frame.loc[
        timestamps.ge(heating_start) & timestamps.lt(observation_end),
        ["timestamp", *columns],
    ].copy()
    values["timestamp"] = pd.to_datetime(values["timestamp"], errors="coerce").dt.floor("s")
    values = values.dropna(subset=["timestamp"]).drop_duplicates("timestamp").set_index("timestamp")
    grid = pd.date_range(
        pd.Timestamp(heating_start).ceil("s"),
        pd.Timestamp(observation_end).ceil("s"),
        freq="s",
        inclusive="left",
    )
    values = values.reindex(grid).rename(columns=columns).apply(pd.to_numeric, errors="coerce")
    values["T3o_C"] /= 10
    values[["T1_min", "T2_min"]] /= 60

    case = values["T4_C"].map(lambda value: _case(value) if pd.notna(value) else np.nan)
    thresholds = [
        limits(t4, twout, t3o)
        if pd.notna(t4) and pd.notna(twout) and pd.notna(t3o)
        else (np.nan, np.nan)
        for t4, twout, t3o in values[["T4_C", "Twout_C", "T3o_C"]].itertuples(
            index=False, name=None
        )
    ]
    t1_limit = pd.Series([value[0] for value in thresholds], index=grid)
    t3_limit = pd.Series([value[1] for value in thresholds], index=grid)

    historical_t3_max = values["T3_C"].shift(50).rolling(551, min_periods=1).max()
    condition1 = (
        values["T1_min"].gt(35)
        & values["T2_min"].ge(6)
        & values["T3_C"].le(-1)
        & historical_t3_max.sub(values["T3_C"]).ge(1)
    )
    case_temperature = values["T3_C"].lt(t3_limit)
    case_confirmed = case_temperature.rolling(20, min_periods=20).sum().eq(20)
    condition2 = values["T2_min"].ge(6) & values["T1_min"].ge(t1_limit) & case_confirmed
    case7_temperature = values["T3_C"].le(-10) & values["T3_C"].le(
        0.8 * values["T4_C"] - 12
    )
    condition7 = (
        values["T1_min"].ge(30)
        & case7_temperature.rolling(20, min_periods=20).sum().eq(20)
    )
    condition8 = values["T1_min"].ge(150)

    triggered = condition1 | condition2 | condition7 | condition8
    if return_trace:
        return values.assign(case=case, triggered=triggered)
    if not triggered.any():
        return {
            "t_RB": pd.NaT,
            "rb_status": "right_censored",
            "trigger_type": "",
            "case": np.nan,
            "T4_C": np.nan,
            "Twout_C": np.nan,
            "T1_min": np.nan,
            "T2_min": np.nan,
            "T3_C": np.nan,
            "T3o_C": np.nan,
            "t_observation_end": observation_end,
        }

    timestamp = triggered[triggered].index[0]
    trigger_type = (
        "Condition1"
        if condition1.loc[timestamp]
        else f"Case{int(case.loc[timestamp])}"
        if condition2.loc[timestamp]
        else "Case7"
        if condition7.loc[timestamp]
        else "Case8"
    )
    return {
        "t_RB": timestamp,
        "rb_status": "triggered",
        "trigger_type": trigger_type,
        "case": int(case.loc[timestamp]) if pd.notna(case.loc[timestamp]) else np.nan,
        **values.loc[
            timestamp, ["T4_C", "Twout_C", "T1_min", "T2_min", "T3_C", "T3o_C"]
        ].to_dict(),
        "t_observation_end": observation_end,
    }


def calculate_cycle(loader: Any, cycle_name: str) -> dict[str, object]:
    """Replay RB for one Dataset cycle, including incomplete observed cycles."""
    record = loader.get_cycle_record(cycle_name)
    nested = record.get("boundaries")
    boundaries = nested if isinstance(nested, Mapping) else record
    frame = loader.load_cycle_original(cycle_name, columns=list(RAW_COLUMNS))
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if not timestamps.notna().any():
        raise ValueError(f"cycle has no valid raw timestamps: {cycle_name}")
    heating_start = pd.to_datetime(boundaries.get("heating_start"), errors="coerce")
    if pd.isna(heating_start):
        heating_start = timestamps.min()
    observation_end = pd.to_datetime(
        boundaries.get("defrost_preparation_start"), errors="coerce"
    )
    observation_end_source = "actual_preparation"
    if pd.isna(observation_end):
        observation_end = timestamps.max().floor("s") + pd.Timedelta(seconds=1)
        observation_end_source = "raw_cycle_end"
    return {
        "cycle_name": cycle_name,
        "cycle_id": record.get("cycle_id"),
        "experiment_id": record.get("experiment_id"),
        "observation_end_source": observation_end_source,
        **first_trigger(frame, pd.Timestamp(heating_start), pd.Timestamp(observation_end)),
    }
