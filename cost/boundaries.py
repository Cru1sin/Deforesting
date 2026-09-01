"""Candidate boundaries reconstructed directly from the Dataset catalog."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

REQUIRED_BOUNDARIES = (
    "heating_start",
    "stable_heating_start",
    "defrost_preparation_start",
    "defrost_start",
    "defrost_end",
)


def catalog_exclusion_reason(
    record: Mapping[str, object], parameter_experiments: set[str]
) -> str | None:
    """Return why catalog metadata cannot enter the frozen science cohort."""
    nested = record.get("boundaries")
    source = nested if isinstance(nested, Mapping) else record
    values: list[pd.Timestamp] = []
    for name in REQUIRED_BOUNDARIES:
        raw = source.get(name)
        if raw is None or str(raw) in {"nan", "NaT"}:
            return f"missing required boundary: {name}"
        try:
            values.append(pd.Timestamp(str(raw)))
        except (TypeError, ValueError):
            return f"missing required boundary: {name}"
    if not (values[0] <= values[1] < values[2] <= values[3] < values[4]):
        return "required boundaries are not ordered"
    experiment = str(record.get("experiment_id"))
    if experiment not in parameter_experiments:
        return f"missing Pe parameters for experiment {experiment}"
    return None


def clean_anchor_exclusion_reason(frame: pd.DataFrame, record: Mapping[str, object]) -> str | None:
    """Return why the current cycle fails its raw 60-second clean anchor."""
    events = cycle_boundaries(record)
    start = events["stable_heating_start"]
    columns = [
        "timestamp",
        "water_flow",
        "water_in_temperature",
        "water_out_temperature",
        "power_total",
    ]
    values = frame.loc[:, columns].copy()
    values["timestamp"] = pd.to_datetime(values["timestamp"], errors="coerce")
    for column in columns[1:]:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values.loc[
        values["timestamp"].ge(start) & values["timestamp"].lt(start + pd.Timedelta(seconds=60))
    ]
    values = values.dropna()
    if len(values) < 48:
        return f"clean anchor has {len(values)} complete rows; requires 48"
    water_heat = (
        1.161
        * values["water_flow"]
        * (values["water_out_temperature"] - values["water_in_temperature"])
    )
    if water_heat.median() <= 0:
        return "clean anchor median water heat is not positive"
    if values["power_total"].median() <= 0:
        return "clean anchor median power is not positive"
    return None


def cycle_boundaries(record: Mapping[str, object]) -> dict[str, pd.Timestamp]:
    """Return the four required event timestamps from a Dataset cycle record."""
    nested = record.get("boundaries")
    source = nested if isinstance(nested, Mapping) else record
    names = REQUIRED_BOUNDARIES
    result = {
        name: pd.Timestamp(str(source[name])) for name in names if source.get(name) is not None
    }
    missing = set(names) - result.keys()
    if missing:
        raise ValueError(f"cycle boundaries are missing: {sorted(missing)}")
    return result


def build_candidate_boundaries(loader: Any, cycle_name: str, start_rule: str) -> pd.DataFrame:
    """Build one-minute candidates from stable+10 min through exact preparation start."""
    record = loader.get_cycle_record(cycle_name)
    events = cycle_boundaries(record)
    if start_rule not in {"heating_start", "stable_heating_start"}:
        raise ValueError("start rule must be heating_start or stable_heating_start")
    first = events["stable_heating_start"] + pd.Timedelta(minutes=10)
    end = events["defrost_preparation_start"]
    if end < first:
        raise ValueError(f"candidate interval is empty for {cycle_name}")
    candidates = list(pd.date_range(first, end, freq="min"))
    if not candidates or candidates[-1] != end:
        candidates.append(end)
    integration_start = events[start_rule]
    return pd.DataFrame(
        {
            "cycle_name": cycle_name,
            "experiment_id": str(record["experiment_id"]),
            "candidate_time": candidates,
            "candidate_elapsed_minutes": [
                (candidate - integration_start).total_seconds() / 60 for candidate in candidates
            ],
            "integration_start": integration_start,
            "integration_start_rule": start_rule,
            "actual_preparation_time": end,
        }
    )
