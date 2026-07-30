"""Auditable missing-data assessment and cycle-local policy application."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MissingDataReport:
    channel_summary: pd.DataFrame
    cycle_summary: pd.DataFrame
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MissingDataResult:
    data: pd.DataFrame
    cycle_summary_updates: pd.DataFrame
    warnings: tuple[str, ...]
    metrics: Mapping[str, int | float]


def assess_missing_data(
    data: pd.DataFrame,
    registry_specs: Mapping[str, Any],
    config: Mapping[str, Any],
) -> MissingDataReport:
    """Measure source missingness without modifying the input frame."""
    group_columns = _group_columns(data, config)
    rows: list[dict[str, object]] = []
    for channel, spec in registry_specs.items():
        if channel not in data:
            continue
        for group_key, group in _groups(data, group_columns):
            cycle_id = _cycle_id(group_key, group)
            values = pd.to_numeric(group[channel], errors="coerce")
            observed = _observed_mask(group, channel, values)
            available = values.notna()
            rows.append(
                {
                    "cycle_id": cycle_id,
                    "channel": channel,
                    "missing_policy": str(getattr(spec, "missing_policy", "none")),
                    "observed_count": int(observed.sum()),
                    "available_count": int(available.sum()),
                    "row_count": int(len(group)),
                    "observed_coverage": float(observed.mean()) if len(group) else 0.0,
                    "available_coverage": float(available.mean()) if len(group) else 0.0,
                    "max_raw_gap_seconds": _maximum_gap(group, observed),
                }
            )
    channel_summary = pd.DataFrame(
        rows,
        columns=[
            "cycle_id",
            "channel",
            "missing_policy",
            "observed_count",
            "available_count",
            "row_count",
            "observed_coverage",
            "available_coverage",
            "max_raw_gap_seconds",
        ],
    )
    cycle_summary = _summarize_cycles(channel_summary)
    return MissingDataReport(channel_summary, cycle_summary)


def apply_missing_policy(
    data: pd.DataFrame,
    report: MissingDataReport,
    registry_specs: Mapping[str, Any],
    config: Mapping[str, Any],
) -> MissingDataResult:
    """Apply configured policies within each cycle and stage only."""
    del report
    result = data.copy()
    group_columns = _group_columns(result, config)
    continuous_config = dict(config.get("continuous", {}))
    control_config = dict(config.get("control", {}))
    imputed_total = 0
    for channel, spec in registry_specs.items():
        if channel not in result:
            continue
        policy = str(getattr(spec, "missing_policy", "none"))
        values = pd.to_numeric(result[channel], errors="coerce")
        observed = _observed_mask(result, channel, values)
        result[f"{channel}__observed"] = observed.astype(bool)
        imputed = pd.Series(False, index=result.index, dtype=bool)
        if policy == "linear" and continuous_config.get("method", "time_linear") != "none":
            maximum = float(
                continuous_config.get("maximum_bracketing_gap_seconds", 60)
            )
            for _, group in _groups(result, group_columns):
                indexes = group.index
                filled, flags = _linear_fill(
                    values.loc[indexes].copy(),
                    pd.to_datetime(group["timestamp"], errors="raise"),
                    _state_values(group, channel),
                    maximum,
                    bool(continuous_config.get("require_both_sides", True)),
                )
                values.loc[indexes] = filled.to_numpy()
                imputed.loc[indexes] = flags.to_numpy(dtype=bool)
        elif policy == "forward_fill" and control_config.get("method", "forward_fill") != "none":
            maximum = float(control_config.get("maximum_age_seconds", 30))
            for _, group in _groups(result, group_columns):
                indexes = group.index
                filled, flags = _forward_fill(
                    result.loc[indexes, channel].copy(),
                    pd.to_datetime(group["timestamp"], errors="raise"),
                    _state_values(group, channel),
                    maximum,
                )
                result.loc[indexes, channel] = filled.to_numpy()
                imputed.loc[indexes] = flags.to_numpy(dtype=bool)
        elif policy == "none":
            pass
        elif policy not in {"linear", "forward_fill"}:
            raise ValueError(f"unsupported missing policy for {channel}: {policy}")
        if policy != "forward_fill":
            result[channel] = values
        result[f"{channel}__imputed"] = imputed
        imputed_total += int(imputed.sum())
    updates = _build_cycle_updates(result, registry_specs, group_columns)
    metrics: dict[str, int | float] = {
        "imputed_value_count": imputed_total,
        "channel_count": len([name for name in registry_specs if name in result]),
    }
    return MissingDataResult(result, updates, (), metrics)


def handle_missing_data(
    frame: pd.DataFrame,
    continuous_columns: Iterable[str],
    control_columns: Iterable[str],
    *,
    continuous_max_gap_seconds: float,
    control_max_gap_seconds: float,
    target_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Compatibility wrapper for callers using the pre-Registry interface."""
    targets = set(target_columns)
    specs: dict[str, Any] = {}
    for column in continuous_columns:
        specs[column] = SimpleNamespace(
            missing_policy="none" if column in targets else "linear"
        )
    for column in control_columns:
        specs[column] = SimpleNamespace(missing_policy="forward_fill")
    config = {
        "group_columns": ["cycle_id", "cycle_stage"],
        "continuous": {
            "method": "time_linear",
            "maximum_bracketing_gap_seconds": continuous_max_gap_seconds,
            "require_both_sides": True,
        },
        "control": {
            "method": "forward_fill",
            "maximum_age_seconds": control_max_gap_seconds,
        },
    }
    report = assess_missing_data(frame, specs, config)
    return apply_missing_policy(frame, report, specs, config).data


def _group_columns(frame: pd.DataFrame, config: Mapping[str, Any]) -> list[str]:
    configured = config.get("group_columns", ["cycle_id", "cycle_stage"])
    return [str(column) for column in configured if str(column) in frame]


def _groups(frame: pd.DataFrame, columns: list[str]) -> Iterable[tuple[object, pd.DataFrame]]:
    if columns:
        return frame.groupby(columns, sort=False, dropna=False)
    return [(None, frame)]


def _cycle_id(group_key: object, group: pd.DataFrame) -> str:
    if isinstance(group_key, tuple) and group_key:
        return str(group_key[0])
    if group_key is not None:
        return str(group_key)
    return str(group["cycle_id"].iloc[0]) if "cycle_id" in group and not group.empty else ""


def _observed_mask(group: pd.DataFrame, channel: str, values: pd.Series) -> pd.Series:
    state_column = f"{channel}__source_state"
    if state_column in group:
        return group[state_column].astype("string").eq("observed")
    return values.notna()


def _state_values(group: pd.DataFrame, channel: str) -> pd.Series:
    state_column = f"{channel}__source_state"
    if state_column in group:
        return group[state_column].astype("string").reset_index(drop=True)
    return pd.Series("observed", index=range(len(group)), dtype="string")


def _linear_fill(
    values: pd.Series,
    times: pd.Series,
    states: pd.Series,
    maximum_gap_seconds: float,
    require_both_sides: bool,
) -> tuple[pd.Series, pd.Series]:
    values = values.reset_index(drop=True).astype(float)
    times = times.reset_index(drop=True)
    states = states.reset_index(drop=True)
    original = values.copy()
    imputed = pd.Series(False, index=values.index, dtype=bool)
    fillable = original.isna() & ~states.eq("invalid")
    starts = fillable.ne(fillable.shift(fill_value=False))
    run_ids = starts.cumsum()
    for _, run in fillable[fillable].groupby(run_ids[fillable]):
        positions = run.index.to_list()
        previous = positions[0] - 1
        following = positions[-1] + 1
        if previous < 0 or following >= len(values):
            continue
        if require_both_sides and (
            pd.isna(original.iloc[previous]) or pd.isna(original.iloc[following])
        ):
            continue
        elapsed = (times.iloc[following] - times.iloc[previous]).total_seconds()
        if elapsed <= 0 or elapsed > maximum_gap_seconds:
            continue
        for position in positions:
            fraction = (times.iloc[position] - times.iloc[previous]).total_seconds() / elapsed
            values.iloc[position] = original.iloc[previous] + fraction * (
                original.iloc[following] - original.iloc[previous]
            )
            imputed.iloc[position] = True
    return values, imputed


def _forward_fill(
    values: pd.Series,
    times: pd.Series,
    states: pd.Series,
    maximum_age_seconds: float,
) -> tuple[pd.Series, pd.Series]:
    result = values.reset_index(drop=True).copy()
    times = times.reset_index(drop=True)
    states = states.reset_index(drop=True)
    original = result.copy()
    imputed = pd.Series(False, index=result.index, dtype=bool)
    last_valid: int | None = None
    for position, value in enumerate(original):
        if pd.notna(value) and states.iloc[position] != "invalid":
            last_valid = position
            continue
        if pd.notna(value) or states.iloc[position] == "invalid" or last_valid is None:
            continue
        age = (times.iloc[position] - times.iloc[last_valid]).total_seconds()
        if 0 <= age <= maximum_age_seconds:
            result.iloc[position] = original.iloc[last_valid]
            imputed.iloc[position] = True
    return result, imputed


def _maximum_gap(group: pd.DataFrame, observed: pd.Series) -> float:
    times = pd.to_datetime(group.loc[observed, "timestamp"], errors="coerce").dropna()
    deltas = times.sort_values().diff().dt.total_seconds().dropna()
    return float(deltas.max()) if not deltas.empty else 0.0


def _summarize_cycles(channel_summary: pd.DataFrame) -> pd.DataFrame:
    if channel_summary.empty:
        return pd.DataFrame(columns=["cycle_id"])
    return (
        channel_summary.groupby("cycle_id", sort=False)
        .agg(
            observed_coverage=("observed_coverage", "min"),
            available_coverage=("available_coverage", "min"),
            maximum_raw_gap_seconds=("max_raw_gap_seconds", "max"),
        )
        .reset_index()
    )


def _build_cycle_updates(
    frame: pd.DataFrame,
    specs: Mapping[str, Any],
    group_columns: list[str],
) -> pd.DataFrame:
    if "cycle_id" not in frame:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for cycle_id, group in frame.groupby("cycle_id", sort=False, dropna=False):
        row: dict[str, object] = {"cycle_id": cycle_id}
        for channel in specs:
            if channel not in group:
                continue
            values = pd.to_numeric(group[channel], errors="coerce")
            observed = group.get(
                f"{channel}__observed", values.notna()
            ).astype(bool)
            imputed = group.get(
                f"{channel}__imputed", pd.Series(False, index=group.index)
            ).astype(bool)
            row[f"{channel}__observed_coverage"] = float(observed.mean())
            row[f"{channel}__available_coverage"] = float(values.notna().mean())
            row[f"{channel}__imputed_fraction"] = float(imputed.mean())
        rows.append(row)
    del group_columns
    return pd.DataFrame(rows)
