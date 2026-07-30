"""Cycle- and stage-isolated missing-data handling for Pipeline 2."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def handle_missing_data(
    frame: pd.DataFrame,
    continuous_columns: Iterable[str],
    control_columns: Iterable[str],
    *,
    continuous_max_gap_seconds: float,
    control_max_gap_seconds: float,
    target_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Fill only bounded gaps within a cycle and stage; leave targets untouched."""
    result = frame.copy()
    targets = set(target_columns)
    time_column = "timestamp" if "timestamp" in result else "sensor_time"
    if time_column != "timestamp":
        result = result.rename(columns={time_column: "timestamp"})
    result = result.sort_values("timestamp", kind="stable").reset_index(drop=True)
    groups = [column for column in ("cycle_id", "cycle_stage") if column in result]
    if not groups:
        groups = ["timestamp"]
    for column in continuous_columns:
        if column not in result or column in targets:
            continue
        _fill_numeric_column(result, column, groups, continuous_max_gap_seconds)
    for column in control_columns:
        if column not in result or column in targets:
            continue
        _fill_numeric_column(result, column, groups, control_max_gap_seconds, forward_only=True)
    if time_column != "timestamp":
        result = result.rename(columns={"timestamp": time_column})
    return result


def _fill_numeric_column(
    frame: pd.DataFrame,
    column: str,
    groups: list[str],
    maximum_gap_seconds: float,
    *,
    forward_only: bool = False,
) -> None:
    for _, group in frame.groupby(groups, sort=False, dropna=False):
        values = pd.to_numeric(group[column], errors="coerce").reset_index(drop=True)
        times = pd.to_datetime(group["timestamp"], errors="coerce").reset_index(drop=True)
        missing = values.isna().to_numpy()
        positions = np.flatnonzero(missing)
        for position in positions:
            previous = position - 1
            following = position + 1
            if previous < 0 or pd.isna(values.iloc[int(previous)]):
                continue
            if forward_only:
                elapsed = (
                    times.iloc[int(position)] - times.iloc[int(previous)]
                ).total_seconds()
                if elapsed <= maximum_gap_seconds:
                    frame.loc[group.index[int(position)], column] = float(
                        values.iloc[int(previous)]
                    )
                continue
            if following >= len(group) or pd.isna(values.iloc[int(following)]):
                continue
            elapsed = (
                times.iloc[int(following)] - times.iloc[int(previous)]
            ).total_seconds()
            if elapsed <= maximum_gap_seconds:
                fraction = (
                    times.iloc[int(position)] - times.iloc[int(previous)]
                ).total_seconds() / elapsed
                value = values.iloc[int(previous)] + fraction * (
                    values.iloc[int(following)] - values.iloc[int(previous)]
                )
                frame.loc[group.index[int(position)], column] = float(value)
