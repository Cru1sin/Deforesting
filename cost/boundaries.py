"""Candidate boundaries reconstructed directly from the Dataset catalog."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def cycle_boundaries(record: Mapping[str, object]) -> dict[str, pd.Timestamp]:
    """Return the four required event timestamps from a Dataset cycle record."""
    nested = record.get("boundaries")
    source = nested if isinstance(nested, Mapping) else record
    names = (
        "heating_start",
        "stable_heating_start",
        "defrost_preparation_start",
        "defrost_start",
        "defrost_end",
    )
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
