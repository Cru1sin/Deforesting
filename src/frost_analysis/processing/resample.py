"""Role-aware, cycle-isolated resampling for processed datasets."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import pandas as pd


def resample_data(
    frame: pd.DataFrame,
    *,
    interval_seconds: int,
    numeric_columns: Iterable[str],
    control_columns: Iterable[str],
    state_columns: Iterable[str],
) -> pd.DataFrame:
    """Resample each cycle/stage independently without inventing cross-boundary rows."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if frame.empty:
        return frame.copy()
    result_groups: list[pd.DataFrame] = []
    group_columns = [column for column in ("cycle_id", "cycle_stage") if column in frame]
    if not group_columns:
        group_columns = ["cycle_id"] if "cycle_id" in frame else []
    grouped = (
        frame.groupby(group_columns, sort=False, dropna=False)
        if group_columns
        else [(None, frame)]
    )
    numeric = set(numeric_columns)
    controls = set(control_columns)
    states = set(state_columns)
    for _, group in grouped:
        current = group.copy().sort_values("timestamp", kind="stable")
        current = current.set_index(pd.to_datetime(current.pop("timestamp"), errors="raise"))
        aggregations: dict[str, str] = {}
        for column in current.columns:
            if column in {"cycle_id", "cycle_stage"}:
                aggregations[column] = "last"
                continue
            if column in numeric:
                aggregations[column] = "mean"
            elif column in controls or column in states or current[column].dtype == "object":
                aggregations[column] = "last"
            else:
                aggregations[column] = "last"
        sampled = current.resample(
            f"{interval_seconds}s", label="left", closed="left"
        ).agg(cast(Any, aggregations))
        sampled = sampled.loc[sampled.notna().any(axis=1)]
        sampled.insert(0, "timestamp", sampled.index)
        result_groups.append(sampled.reset_index(drop=True))
    if not result_groups:
        return frame.iloc[0:0].copy()
    return pd.concat(result_groups, ignore_index=True, sort=False).sort_values(
        "timestamp", kind="stable"
    ).reset_index(drop=True)
