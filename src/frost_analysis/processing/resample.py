"""Cycle- and stage-isolated fixed-interval observation resampling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any

import pandas as pd


def resample_observations(  # noqa: C901 - column contracts are intentionally explicit
    frame: pd.DataFrame,
    registry_specs: Mapping[str, Any],
    *,
    interval_seconds: int,
    state_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Resample each cycle/stage from its first timestamp and keep empty bins."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if frame.empty:
        return frame.copy()
    group_columns = [column for column in ("cycle_id", "cycle_stage") if column in frame]
    if not group_columns:
        group_columns = ["cycle_id"] if "cycle_id" in frame else []
    grouped = (
        frame.groupby(group_columns, sort=False, dropna=False)
        if group_columns
        else [(None, frame)]
    )
    groups: list[pd.DataFrame] = []
    for _, group in grouped:
        current = group.copy().sort_values("timestamp", kind="stable")
        current["timestamp"] = pd.to_datetime(current["timestamp"], errors="raise")
        start = current["timestamp"].iloc[0]
        offsets = (current["timestamp"] - start).dt.total_seconds()
        current["_bin_time"] = start + pd.to_timedelta(
            (offsets // interval_seconds) * interval_seconds, unit="s"
        )
        end = current["_bin_time"].iloc[-1]
        bins = pd.date_range(start, end, freq=f"{interval_seconds}s")
        sampled = pd.DataFrame(index=bins)
        sampled.index.name = "timestamp"
        for column in current.columns:
            if column in {"timestamp", "_bin_time"}:
                continue
            spec = registry_specs.get(column)
            if spec is not None and str(getattr(spec, "data_kind", "")) == "derived":
                # Derived values are rebuilt after source missing-data policy.
                continue
            if column in group_columns or column in set(state_columns) or column in {
                "cycle_status",
                "cycle_progress",
                "cycle_elapsed_seconds",
                "is_heating",
            }:
                values = current[column].dropna()
                sampled[column] = values.iloc[0] if not values.empty else pd.NA
                continue
            if column.endswith("__source_state"):
                states = current[column].astype("string").groupby(current["_bin_time"])
                sampled[column] = states.agg(_aggregate_source_state).reindex(bins).fillna(
                    "not_sampled"
                )
                continue
            if column.endswith(("__missing", "__invalid")):
                flags = current[column].astype("boolean").groupby(current["_bin_time"])
                sampled[column] = flags.any().reindex(bins).fillna(False).astype("boolean")
                continue
            method = _resample_method(spec)
            if method in {"none", "last"}:
                sampled[column] = current[column].groupby(current["_bin_time"]).last().reindex(bins)
            else:
                numeric = pd.to_numeric(current[column], errors="coerce")
                if not numeric.notna().any() and not pd.api.types.is_numeric_dtype(
                    current[column]
                ):
                    sampled[column] = (
                        current[column]
                        .groupby(current["_bin_time"])
                        .last()
                        .reindex(bins)
                    )
                else:
                    sampled[column] = numeric.groupby(current["_bin_time"]).mean().reindex(bins)
        sampled.insert(0, "timestamp", sampled.index)
        sampled["source_sample_count"] = (
            current.groupby("_bin_time").size().reindex(bins).fillna(0).astype(int).to_numpy()
        )
        sampled = sampled.reset_index(drop=True)
        groups.append(sampled)
    all_columns = list(dict.fromkeys(column for group in groups for column in group.columns))
    concat_groups = [
        group.drop(
            columns=[
                column
                for column in group.columns
                if bool(group[column].isna().all())
            ]
        )
        for group in groups
    ]
    combined = pd.concat(concat_groups, ignore_index=True, sort=False)
    for column in all_columns:
        if column not in combined:
            combined[column] = pd.Series(pd.NA, index=combined.index)
    return combined.loc[:, all_columns].sort_values("timestamp", kind="stable").reset_index(
        drop=True
    )


def resample_data(
    frame: pd.DataFrame,
    *,
    interval_seconds: int,
    numeric_columns: Iterable[str],
    control_columns: Iterable[str],
    state_columns: Iterable[str],
) -> pd.DataFrame:
    """Backward-compatible wrapper using the role lists from the old API."""
    specs: dict[str, Any] = {
        column: SimpleNamespace(resample_method="mean") for column in numeric_columns
    }
    specs.update(
        {column: SimpleNamespace(resample_method="last") for column in control_columns}
    )
    return resample_observations(
        frame,
        specs,
        interval_seconds=interval_seconds,
        state_columns=state_columns,
    )


def _resample_method(spec: Any) -> str:
    method = getattr(spec, "resample_method", "mean") if spec is not None else "mean"
    return str(method)


def _aggregate_source_state(values: pd.Series) -> str:
    states = values.dropna().astype("string")
    if states.eq("observed").any():
        return "observed"
    if states.eq("invalid").any():
        return "invalid"
    if states.eq("missing").any():
        return "missing"
    return "not_sampled"
