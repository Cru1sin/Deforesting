"""Cycle-isolated, backward-looking multiscale feature engineering."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureResult:
    frame: pd.DataFrame
    dictionary: pd.DataFrame
    quality: pd.DataFrame


@dataclass(frozen=True)
class AnalysisResolutionResult:
    """Normal-heating observations retained without crossing physical boundaries."""

    frame: pd.DataFrame
    sensor_specs: dict[str, dict[str, str]]


def resample_for_analysis(  # noqa: C901 - aggregation rules are intentionally explicit
    frame: pd.DataFrame,
    sensor_specs: dict[str, dict[str, str]],
    *,
    interval_seconds: int,
    state_columns: list[str] | None = None,
    passthrough_columns: list[str] | None = None,
) -> AnalysisResolutionResult:
    """Filter normal heating observations while preserving their timestamps.

    Earlier versions rounded observations into fixed calendar bins. That made
    sparse samples appear regularly observed and moved values across stage
    boundaries. The analysis-resolution contract now keeps one row per source
    observation and exposes bin-shaped metadata solely for downstream
    compatibility and auditability.
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    required = {
        "sensor_time",
        "cycle_id",
        "cycle_quality",
        "stage",
        "cycle_phase",
        "cycle_time_s",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"analysis resampling missing required columns: {missing}")

    normal = frame.loc[
        frame["cycle_quality"].eq("complete")
        & frame["stage"].isin(["stable_clean", "frost_development"])
        & frame["cycle_id"].notna()
    ].copy()
    if normal.empty:
        columns = [
            "sensor_time",
            "cycle_id",
            "cycle_quality",
            "stage",
            "cycle_phase",
            "cycle_time_s",
            "source_latest_time",
            "analysis_interval_s",
            "analysis_bin_end",
            "source_sample_count",
            "analysis_bin_expected_count",
            "analysis_bin_coverage",
            "analysis_bin_available",
        ]
        columns.extend(name for name in sensor_specs if name in frame.columns)
        columns.extend(
            column
            for column in frame.columns
            if column in (state_columns or [])
            or column in (passthrough_columns or [])
            or column in _PASSTHROUGH_COLUMNS
        )
        registry_names = set(sensor_specs)
        columns.extend(
            column
            for column in frame.columns
            if "__" in column
            and column.split("__", 1)[0] in registry_names
            and column.split("__", 1)[1] in _BASELINE_SUFFIXES
        )
        columns = list(dict.fromkeys(columns))
        return AnalysisResolutionResult(pd.DataFrame(columns=columns), dict(sensor_specs))

    normal["sensor_time"] = pd.to_datetime(normal["sensor_time"], errors="raise")
    normal = normal.sort_values(["cycle_id", "sensor_time"], kind="stable")

    # Keep the registry channels, their baseline provenance, state columns, and
    # explicitly requested targets. Other source columns are intentionally left
    # out so feature engineering cannot accidentally consume an unregistered
    # signal. The source timestamps are intentionally preserved: rolling
    # windows below are time-based and must see real gaps rather than a
    # fabricated regular grid.
    selected = [
        column
        for column in normal.columns
        if column in required
        or column in sensor_specs
        or column in (state_columns or [])
        or column in (passthrough_columns or [])
        or column in _PASSTHROUGH_COLUMNS
    ]
    registry_names = set(sensor_specs)
    for column in normal.columns:
        root, separator, suffix = column.partition("__")
        if separator and root in registry_names and suffix in _BASELINE_SUFFIXES:
            selected.append(column)
    selected = list(dict.fromkeys(column for column in selected if column in normal))
    result = normal.loc[:, selected].copy()
    result["source_latest_time"] = result["sensor_time"]
    result["source_sample_count"] = 1
    result["analysis_interval_s"] = np.nan
    for _, group in result.groupby("cycle_id", sort=False):
        nominal = _nominal_seconds(
            pd.DatetimeIndex(pd.to_datetime(group["sensor_time"], errors="coerce"))
        )
        result.loc[group.index, "analysis_interval_s"] = nominal
    result["analysis_bin_end"] = result["sensor_time"]
    result["analysis_bin_expected_count"] = 1
    result["analysis_bin_coverage"] = 1.0
    result["analysis_bin_available"] = True
    result = result.sort_values(["cycle_id", "sensor_time"], kind="stable")
    result = result.reset_index(drop=True)
    return AnalysisResolutionResult(
        result, {name: dict(spec) for name, spec in sensor_specs.items()}
    )


def _transition_count(values: pd.Series) -> int:
    observed = values.dropna()
    if len(observed) < 2:
        return 0
    return int(observed.ne(observed.shift()).sum() - 1)


def engineer_features(
    frame: pd.DataFrame,
    sensor_specs: Mapping[str, Any] | Any,
    *,
    windows_minutes: list[int],
    minimum_coverage: float,
) -> FeatureResult:
    """Build a compact, cycle-isolated matrix from registry channels."""
    if not windows_minutes or any(int(value) <= 0 for value in windows_minutes):
        raise ValueError("windows_minutes must contain positive values")
    windows = list(dict.fromkeys(int(value) for value in windows_minutes))
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    specs = _normalise_specs(sensor_specs)
    base = _feature_frame(frame, specs)
    base, specs, relationship_records = _add_physical_relationships(base, specs)
    generated: dict[str, pd.Series] = {}
    records: list[dict[str, object]] = relationship_records
    channels = [
        name
        for name, spec in specs.items()
        if name in base
        and bool(spec.get("analysis_enabled", True))
        and str(spec.get("data_role", "X")) in {"X", "C"}
    ]
    for column in channels:
        spec = specs[column]
        base[column] = pd.to_numeric(base[column], errors="coerce")
        records.append(
            _record(
                column,
                str(spec.get("formula") or column),
                [column],
                spec,
                "current value",
                "",
                "current_or_past_only",
            )
        )
        offset = f"{column}__baseline_offset"
        relative = f"{column}__baseline_relative"
        if offset in base:
            baseline_spec = {
                **spec,
                "baseline_availability_column": f"{column}__baseline_available",
                "baseline_source_latest_time_column": (f"{column}__baseline_source_latest_time"),
                "missing_handling": (
                    "NaN until the cycle-local clean baseline is known; no cross-cycle fill"
                ),
            }
            if relative in base:
                records.append(
                    _record(
                        relative,
                        relative,
                        [column],
                        baseline_spec,
                        "cycle-local clean baseline relative offset",
                        "",
                        "available_at_or_after_clean_end; current_or_past_only",
                    )
                )
            records.append(
                _record(
                    offset,
                    offset,
                    [column],
                    baseline_spec,
                    "cycle-local clean baseline offset",
                    "",
                    "available_at_or_after_clean_end; current_or_past_only",
                )
            )
            auc = _cumulative_auc(base, offset)
            auc_name = f"{column}__auc_signed"
            generated[auc_name] = auc["signed"]
            records.append(
                _record(
                    auc_name,
                    f"trapezoidal cumulative signed({column} - clean_mean) over elapsed seconds",
                    [column],
                    {**baseline_spec, "unit": f"{spec.get('unit', 'unknown')}*s"},
                    "signed cumulative baseline deviation",
                    "cumulative",
                    "available_at_or_after_clean_end; current_or_past_only",
                )
            )

        diff_name = f"{column}__diff1"
        generated[diff_name] = _cycle_difference(base, column)
        records.append(
            _record(
                diff_name,
                f"{column}[t] - {column}[previous valid sample]",
                [column],
                spec,
                "cycle-local first difference",
                "previous sample",
                "current_or_past_only",
            )
        )
        for minutes in windows:
            window_features, support = _rolling_features(
                base,
                column,
                minutes,
                minimum_coverage,
                include_action_count=str(spec.get("role")) == "control",
            )
            support_prefix = f"{column}__window_{minutes}m"
            for support_name in ("coverage", "available"):
                generated[f"{support_prefix}__{support_name}"] = support[support_name]
            for kind, values in window_features.items():
                suffix = f"slope_{minutes}m_per_s" if kind == "slope" else f"{kind}_{minutes}m"
                name = f"{column}__{suffix}"
                generated[name] = values
                record = _record(
                    name,
                    _rolling_formula(column, kind, minutes),
                    [column],
                    spec,
                    _rolling_interpretation(kind),
                    f"{minutes} min backward elapsed-time window",
                    "current_or_past_only",
                )
                record.update(
                    {
                        "support_prefix": support_prefix,
                        "availability_column": f"{support_prefix}__available",
                        "coverage_column": f"{support_prefix}__coverage",
                        "support_reason_column": f"{support_prefix}__reason",
                    }
                )
                records.append(record)
    result = pd.concat([base, pd.DataFrame(generated, index=base.index)], axis=1)
    numeric = result.select_dtypes(include=["number"]).columns
    result.loc[:, numeric] = result.loc[:, numeric].replace([np.inf, -np.inf], np.nan)
    dictionary = pd.DataFrame(records)
    if dictionary.empty:
        dictionary = pd.DataFrame(
            columns=[
                "feature",
                "feature_id",
                "canonical_name",
                "raw_source",
                "meaning_zh",
                "availability",
                "confidence",
                "notes",
            ]
        )
    else:
        dictionary = dictionary.drop_duplicates("feature", keep="first").reset_index(drop=True)
    quality = _feature_quality(result, dictionary)
    return FeatureResult(result, dictionary, quality)


_BASELINE_SUFFIXES = (
    "baseline_offset",
    "baseline_relative",
    "relative_valid",
    "baseline_available",
    "baseline_source_latest_time",
)
_PASSTHROUGH_COLUMNS = {
    "cycle_gap_contaminated",
    "source_latest_time",
    "analysis_interval_s",
    "analysis_bin_end",
    "analysis_bin_expected_count",
    "analysis_bin_coverage",
    "analysis_bin_available",
    "source_sample_count",
    "is_heating",
    "operating_mode",
    "defrost_flag",
    "system_state",
    "compressor_state",
    "heating_capacity",
    "power_total",
    "power",
    "cop",
    "water_heating_capacity",
    "water_cop",
}


def _feature_frame(frame: pd.DataFrame, specs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Retain channel/provenance columns and stable pipeline context only."""
    required = [
        "sensor_time",
        "cycle_id",
        "cycle_quality",
        "stage",
        "cycle_phase",
        "cycle_time_s",
        "analysis_bin_available",
        "analysis_bin_coverage",
        "cycle_gap_contaminated",
    ]
    names = set(specs)
    retained = [column for column in required if column in frame]
    for column in frame.columns:
        if column in retained:
            continue
        root, separator, suffix = column.partition("__")
        if (
            column in names
            or separator
            and root in names
            and suffix in _BASELINE_SUFFIXES
            or column in _PASSTHROUGH_COLUMNS
        ):
            retained.append(column)
    if "sensor_time" not in retained:
        raise ValueError("feature engineering requires sensor_time")
    result = frame.loc[:, list(dict.fromkeys(retained))].copy()
    result["sensor_time"] = pd.to_datetime(result["sensor_time"], errors="raise")
    if "cycle_id" in result:
        result = result.sort_values(["cycle_id", "sensor_time"], kind="stable").reset_index(
            drop=True
        )
    else:
        result = result.sort_values("sensor_time", kind="stable").reset_index(drop=True)
    return result


def _normalise_specs(sensor_specs: Mapping[str, Any] | Any) -> dict[str, dict[str, Any]]:
    registry_mode = hasattr(sensor_specs, "specs") or isinstance(sensor_specs, pd.DataFrame)
    if hasattr(sensor_specs, "specs"):
        sensor_specs = sensor_specs.specs
    if isinstance(sensor_specs, pd.DataFrame):
        rows = sensor_specs.to_dict(orient="records")
        sensor_specs = {
            str(row.get("canonical_name", row.get("feature_id", ""))): row for row in rows
        }
    if not isinstance(sensor_specs, Mapping):
        raise TypeError("sensor_specs must be a mapping, DataFrame, or registry result")
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in sensor_specs.items():
        if hasattr(value, "feature_id") or (
            isinstance(value, Mapping)
            and {"feature_id", "canonical_name", "data_role"}.intersection(value)
        ):
            registry_mode = True
        name = str(getattr(value, "canonical_name", key))
        if hasattr(value, "__dataclass_fields__"):
            spec = {field: getattr(value, field) for field in value.__dataclass_fields__}
        elif isinstance(value, Mapping):
            spec = dict(value)
        else:
            spec = {"canonical_name": name}
        spec.setdefault("feature_id", str(getattr(value, "feature_id", name)))
        spec.setdefault("canonical_name", name)
        spec.setdefault("physical_group", spec.get("physical_family", "unclassified"))
        spec.setdefault("physical_family", spec.get("physical_group", "unclassified"))
        spec.setdefault("data_role", spec.get("category", "X"))
        source_type = str(spec.get("source_type", "measured"))
        spec.setdefault(
            "role",
            "control" if source_type in {"setpoint", "controller_value", "control"} else "sensor",
        )
        spec.setdefault("unit", "unknown")
        spec.setdefault("source", spec.get("raw_source", name))
        spec.setdefault("status", spec.get("deployment_status", "present"))
        spec.setdefault("confidence", "confirmed")
        spec.setdefault("unit_confidence", "high")
        spec.setdefault("analysis_enabled", True)
        spec["_registry_mode"] = registry_mode
        normalized[name] = spec
    return normalized


def _cycle_difference(frame: pd.DataFrame, column: str) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby("cycle_id", sort=False, dropna=False):
        ordered = group.sort_values("sensor_time", kind="stable")
        values = pd.to_numeric(ordered[column], errors="coerce").to_numpy(dtype=float)
        times = pd.DatetimeIndex(pd.to_datetime(ordered["sensor_time"], errors="coerce"))
        nominal = _nominal_seconds(times)
        delta = np.diff(times.view("int64"), prepend=np.iinfo(np.int64).min) / 1e9
        valid = np.isfinite(values)
        difference = np.full(len(ordered), np.nan, dtype=float)
        for position in range(1, len(ordered)):
            if valid[position] and valid[position - 1] and 0 < delta[position] <= 3 * nominal:
                difference[position] = values[position] - values[position - 1]
        result.loc[ordered.index] = difference
    return result


def _nominal_seconds(times: pd.DatetimeIndex) -> float:
    if len(times) < 2:
        return 1.0
    deltas = np.diff(times.view("int64")) / 1e9
    positive = deltas[np.isfinite(deltas) & (deltas > 0)]
    return float(np.median(positive)) if len(positive) else 1.0


def _rolling_features(
    frame: pd.DataFrame,
    column: str,
    minutes: int,
    minimum_coverage: float,
    *,
    include_action_count: bool = False,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    # Kept as a compatibility argument for older callers. The registry surface
    # intentionally exposes only compact level, change, and trend summaries.
    del include_action_count
    names = ["mean", "change", "slope"]
    output = {name: pd.Series(np.nan, index=frame.index, dtype=float) for name in names}
    support: dict[str, pd.Series] = {
        "observed_count": pd.Series(0, index=frame.index, dtype=int),
        "expected_count": pd.Series(0, index=frame.index, dtype=int),
        "elapsed_span_s": pd.Series(np.nan, index=frame.index, dtype=float),
        "coverage": pd.Series(0.0, index=frame.index, dtype=float),
        "maximum_gap_s": pd.Series(np.nan, index=frame.index, dtype=float),
        "available": pd.Series(False, index=frame.index, dtype=bool),
        "reason": pd.Series("outside_complete_cycle", index=frame.index, dtype="string"),
    }
    window_seconds = minutes * 60.0
    for _, group in frame.loc[frame["cycle_id"].notna()].groupby("cycle_id", sort=False):
        ordered = group.sort_values("sensor_time", kind="stable")
        times = pd.DatetimeIndex(pd.to_datetime(ordered["sensor_time"], errors="coerce"))
        ordered_values = pd.to_numeric(ordered[column], errors="coerce")
        if "analysis_bin_available" in ordered:
            ordered_values = ordered_values.where(ordered["analysis_bin_available"].astype(bool))
        values = ordered_values.to_numpy(dtype=float)
        nominal = _nominal_seconds(times)
        expected_count = max(2, int(np.floor(window_seconds / max(nominal, 1e-9))) + 1)
        min_periods = max(2, int(np.ceil(expected_count * minimum_coverage)))
        by_time = {
            "mean": np.full(len(ordered), np.nan, dtype=float),
            "change": _backward_change(times, values, window_seconds),
            "slope": _rolling_slope(
                times,
                pd.Series(values, index=ordered.index),
                minutes,
                min_periods,
            ),
        }
        nanoseconds = times.view("int64")
        starts = np.searchsorted(nanoseconds, nanoseconds - int(window_seconds * 1e9), side="left")
        finite = np.isfinite(values)
        for position, start in enumerate(starts):
            valid_positions = np.flatnonzero(finite[start : position + 1]) + start
            if len(valid_positions):
                by_time["mean"][position] = float(np.mean(values[valid_positions]))
        support_values = _window_support(
            times,
            values,
            window_seconds,
            nominal,
            minimum_coverage,
        )
        available = support_values["available"]
        for name, array in by_time.items():
            by_time[name] = np.where(available, array, np.nan)
        ordered_indices = ordered.index.to_numpy()
        for name, array in by_time.items():
            output[name].loc[ordered_indices] = array
        for name, array in support_values.items():
            support[name].loc[ordered_indices] = array
    return output, support


def _window_support(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    window_seconds: float,
    nominal_seconds: float,
    minimum_coverage: float,
) -> dict[str, np.ndarray]:
    nanoseconds = times.view("int64")
    if len(times) != len(values):
        raise ValueError("times and values must have equal length")
    starts = np.searchsorted(nanoseconds, nanoseconds - int(window_seconds * 1e9), side="left")
    expected_value = max(2, int(np.floor(window_seconds / max(nominal_seconds, 1e-9))) + 1)
    observed = np.zeros(len(times), dtype=int)
    expected = np.full(len(times), expected_value, dtype=int)
    span = np.full(len(times), np.nan)
    coverage = np.zeros(len(times), dtype=float)
    maximum_gap = np.full(len(times), np.nan)
    available = np.zeros(len(times), dtype=bool)
    reason = np.full(len(times), "no_valid_observations", dtype=object)
    finite = np.isfinite(values)
    for position, start in enumerate(starts):
        valid_positions = np.flatnonzero(finite[start : position + 1]) + start
        observed[position] = len(valid_positions)
        if not len(valid_positions):
            continue
        span[position] = float((nanoseconds[position] - nanoseconds[valid_positions[0]]) / 1e9)
        observed_times = nanoseconds[valid_positions]
        maximum_gap[position] = (
            float(np.diff(observed_times).max() / 1e9) if len(observed_times) > 1 else 0.0
        )
        count_coverage = min(1.0, observed[position] / expected_value)
        span_coverage = min(1.0, span[position] / window_seconds)
        coverage[position] = min(count_coverage, span_coverage)
        if maximum_gap[position] > 3 * max(nominal_seconds, 1e-9):
            reason[position] = "internal_gap"
        elif count_coverage < minimum_coverage:
            reason[position] = "insufficient_count"
        elif span_coverage < minimum_coverage:
            reason[position] = "insufficient_elapsed_span"
        else:
            available[position] = True
            reason[position] = "available"
    return {
        "observed_count": observed,
        "expected_count": expected,
        "elapsed_span_s": span,
        "coverage": coverage,
        "maximum_gap_s": maximum_gap,
        "available": available,
        "reason": reason,
    }


def _rolling_slope(
    times: pd.DatetimeIndex, values: pd.Series, minutes: int, min_periods: int
) -> np.ndarray:
    timestamps = pd.DatetimeIndex(pd.to_datetime(times, errors="coerce"))
    nanoseconds = timestamps.view("int64")
    values_array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values_array), np.nan, dtype=float)
    window_seconds = float(minutes) * 60.0
    starts = np.searchsorted(nanoseconds, nanoseconds - int(window_seconds * 1e9), side="left")
    finite = np.isfinite(values_array)
    for position, start in enumerate(starts):
        valid_positions = np.flatnonzero(finite[start : position + 1]) + start
        if len(valid_positions) < min_periods:
            continue
        seconds = (nanoseconds[valid_positions] - nanoseconds[position]) / 1e9
        observed = values_array[valid_positions]
        centered = seconds - float(np.mean(seconds))
        denominator = float(np.dot(centered, centered))
        if denominator <= 1e-12:
            continue
        output[position] = float(np.dot(centered, observed - np.mean(observed)) / denominator)
    return output


def _backward_change(
    times: pd.DatetimeIndex, values: np.ndarray, window_seconds: float
) -> np.ndarray:
    nanoseconds = times.view("int64")
    targets = nanoseconds - int(window_seconds * 1e9)
    starts = np.searchsorted(nanoseconds, targets, side="left")
    result = np.full(len(values), np.nan, dtype=float)
    finite = np.isfinite(values)
    for position, start in enumerate(starts):
        if not finite[position] or position <= start:
            continue
        previous = np.flatnonzero(finite[start:position]) + start
        if len(previous):
            result[position] = values[position] - values[previous[0]]
    return result


def _cumulative_auc(frame: pd.DataFrame, offset_column: str) -> dict[str, pd.Series]:
    output = {
        kind: pd.Series(np.nan, index=frame.index, dtype=float)
        for kind in ("signed", "positive", "negative", "absolute")
    }
    for _, group in frame.loc[frame["cycle_id"].notna()].groupby("cycle_id", sort=False):
        ordered = group.sort_values("sensor_time", kind="stable")
        values = pd.to_numeric(ordered[offset_column], errors="coerce").to_numpy(dtype=float)
        times = pd.DatetimeIndex(pd.to_datetime(ordered["sensor_time"], errors="coerce"))
        dt = np.diff(times.view("int64"), prepend=np.iinfo(np.int64).min) / 1e9
        nominal = _nominal_seconds(times)
        finite = np.isfinite(values)
        accumulators = {kind: 0.0 for kind in output}
        arrays = {kind: np.full(len(ordered), np.nan) for kind in output}
        previous_finite = False
        for position in range(len(ordered)):
            if not finite[position]:
                # A missing baseline offset breaks the causal integral. A new
                # segment starts only after two consecutive finite points.
                accumulators = {kind: 0.0 for kind in output}
                previous_finite = False
                continue
            if (
                previous_finite
                and np.isfinite(dt[position])
                and 0 < dt[position] <= 3 * max(nominal, 1e-9)
            ):
                signed_step = 0.5 * (values[position] + values[position - 1]) * dt[position]
                increments = {
                    "signed": signed_step,
                    "positive": max(signed_step, 0.0),
                    "negative": max(-signed_step, 0.0),
                    "absolute": abs(signed_step),
                }
                for kind, increment in increments.items():
                    accumulators[kind] += float(increment)
                for kind in output:
                    arrays[kind][position] = accumulators[kind]
            else:
                # First point of a segment, or a gap beyond the nominal
                # cadence, is not an integral observation yet.
                accumulators = {kind: 0.0 for kind in output}
            previous_finite = True
        for kind, values_out in arrays.items():
            output[kind].loc[ordered.index] = values_out
    return output


def _add_physical_relationships(
    frame: pd.DataFrame, specs: dict[str, dict[str, str]]
) -> tuple[pd.DataFrame, dict[str, dict[str, str]], list[dict[str, object]]]:
    result = frame.copy()
    expanded = {key: dict(value) for key, value in specs.items()}
    registry_mode = any(bool(value.get("_registry_mode")) for value in expanded.values())
    records: list[dict[str, object]] = []
    relationships = [
        (
            "ambient_evaporating_delta",
            "ambient_temperature",
            "evaporating_temperature",
            "subtract",
            "degC",
            "evaporator_response",
            "air-to-evaporating temperature approach",
        ),
        (
            "water_delta_temperature",
            "water_out_temperature",
            "water_in_temperature",
            "subtract",
            "degC",
            "system_performance",
            "water-side temperature rise",
        ),
        (
            "superheat_calculated",
            "suction_temperature",
            "evaporating_temperature",
            "subtract",
            "degC",
            "evaporator_response",
            "suction superheat proxy",
        ),
        (
            "pressure_lift",
            "condensing_pressure",
            "evaporating_pressure",
            "subtract",
            "MPa",
            "condenser_cycle_response",
            "condensing-to-evaporating pressure lift",
        ),
        (
            "ambient_minus_evaporating_temperature",
            "ambient_temperature",
            "evaporating_temperature",
            "subtract",
            "degC",
            "evaporator_thermodynamics",
            "air-to-evaporating temperature approach",
        ),
        (
            "suction_superheat",
            "suction_temperature",
            "evaporating_temperature",
            "subtract",
            "degC",
            "evaporator_thermodynamics",
            "suction superheat proxy",
        ),
        (
            "temperature_lift",
            "condensing_temperature",
            "evaporating_temperature",
            "subtract",
            "degC",
            "evaporator_thermodynamics",
            "condensing-to-evaporating temperature lift",
        ),
        (
            "condensing_minus_evaporating_pressure",
            "condensing_pressure",
            "evaporating_pressure",
            "subtract",
            "MPa",
            "evaporator_thermodynamics",
            "condensing-to-evaporating pressure lift",
        ),
        (
            "pressure_ratio",
            "condensing_pressure",
            "evaporating_pressure",
            "divide",
            "dimensionless",
            "evaporator_thermodynamics",
            "absolute condensing-to-evaporating pressure ratio",
        ),
        (
            "water_out_minus_in_temperature",
            "water_out_temperature",
            "water_in_temperature",
            "subtract",
            "degC",
            "system_performance",
            "water-side temperature rise",
        ),
        (
            "fan_current_per_speed",
            "fan_current",
            "fan_speed",
            "divide",
            "A/speed",
            "air_side",
            "fan loading proxy",
        ),
        (
            "compressor_current_per_frequency",
            "compressor_current",
            "compressor_frequency",
            "divide",
            "A/Hz",
            "control_response",
            "compressor effort per frequency",
        ),
        (
            "calculated_cop",
            "heating_capacity",
            "power",
            "divide",
            "dimensionless",
            "system_performance",
            "heating capacity divided by input power",
        ),
    ]
    registry_relationships = {
        "ambient_evaporating_delta",
        "water_delta_temperature",
        "superheat_calculated",
        "pressure_lift",
    }
    for name, left, right, operation, unit, group, meaning in relationships:
        # In registry mode, only explicitly declared derived channels are
        # eligible. This prevents legacy aliases from silently expanding the
        # feature surface and competing with canonical registry names.
        if registry_mode and name not in expanded:
            continue
        if not registry_mode and name in registry_relationships:
            continue
        if name in result:
            continue
        if (
            left not in result
            or right not in result
            or left not in expanded
            or right not in expanded
        ):
            continue
        left_spec = expanded[left]
        right_spec = expanded[right]
        confidence_score = min(
            _confidence_score(left_spec.get("confidence", "confirmed")),
            _confidence_score(right_spec.get("confidence", "confirmed")),
        )
        if confidence_score < _confidence_score("medium"):
            continue
        left_values = pd.to_numeric(result[left], errors="coerce")
        right_values = pd.to_numeric(result[right], errors="coerce")
        if operation == "subtract":
            result[name] = left_values - right_values
            formula = str(expanded.get(name, {}).get("formula") or f"{left} - {right}")
        else:
            result[name] = left_values / right_values.where(right_values.abs().gt(1e-12))
            formula = str(expanded.get(name, {}).get("formula") or f"{left} / {right}")
        confidence = _confidence_label(confidence_score)
        spec = {
            **expanded.get(name, {}),
            "unit": unit,
            "physical_group": group,
            "source": f"{left_spec.get('source', left)};{right_spec.get('source', right)}",
            "status": "derived",
            "confidence": confidence,
            "unit_confidence": _confidence_label(
                min(
                    _confidence_score(left_spec.get("unit_confidence", "high")),
                    _confidence_score(right_spec.get("unit_confidence", "high")),
                )
            ),
            "rationale": meaning,
            "input_sources": ";".join(
                [str(left_spec.get("source", left)), str(right_spec.get("source", right))]
            ),
            "input_statuses": ";".join(
                [str(left_spec.get("status", "present")), str(right_spec.get("status", "present"))]
            ),
            "input_confidences": ";".join(
                [
                    str(left_spec.get("confidence", "confirmed")),
                    str(right_spec.get("confidence", "confirmed")),
                ]
            ),
            "input_units": (
                f"{left_spec.get('unit', 'unknown')};{right_spec.get('unit', 'unknown')}"
            ),
            "input_rationales": ";".join(
                [str(left_spec.get("rationale", "")), str(right_spec.get("rationale", ""))]
            ),
        }
        expanded[name] = spec
        records.append(
            _record(name, formula, [left, right], spec, meaning, "", "current_or_past_only")
        )
    return result, expanded, records


def _record(
    feature: str,
    formula: str,
    inputs: list[str],
    spec: Mapping[str, Any],
    interpretation: str,
    window: str,
    causality: str,
) -> dict[str, object]:
    confidence = str(spec.get("confidence", "confirmed"))
    return {
        "feature": feature,
        "feature_id": str(spec.get("feature_id", feature)),
        "canonical_name": str(spec.get("canonical_name", inputs[0] if inputs else feature)),
        "formula": formula,
        "input_fields": ";".join(inputs),
        "unit": spec.get("unit", "unknown"),
        "window": window,
        "causality": causality,
        "missing_handling": spec.get("missing_handling", "retain NaN; no cross-cycle fill"),
        "role": spec.get("role", "derived"),
        "physical_group": spec.get("physical_group", "unclassified"),
        "physical_family": spec.get("physical_family", spec.get("physical_group", "unclassified")),
        "physical_interpretation": interpretation,
        "input_sources": spec.get("input_sources", spec.get("source", ";".join(inputs))),
        "input_statuses": spec.get("input_statuses", spec.get("status", "present")),
        "input_confidences": spec.get("input_confidences", confidence),
        "input_units": spec.get("input_units", spec.get("unit", "unknown")),
        "input_rationales": spec.get("input_rationales", spec.get("rationale", "")),
        "raw_source": spec.get("raw_source", spec.get("source", "")),
        "meaning_zh": spec.get("meaning_zh", ""),
        "availability": spec.get("availability", "current_history"),
        "confidence": confidence,
        "notes": spec.get("notes", ""),
        "semantic_confidence": confidence,
        "semantic_confidence_score": _confidence_score(confidence),
        "unit_confidence": spec.get("unit_confidence", "high"),
        "source_type": spec.get("source_type", "measured"),
        "data_role": spec.get("data_role", "X"),
        "deployment_status": spec.get("deployment_status", spec.get("status", "present")),
        "primary_or_validation": spec.get("primary_or_validation", "primary"),
        "analysis_enabled": bool(spec.get("analysis_enabled", True)),
        "support_prefix": "",
        "availability_column": "",
        "coverage_column": "",
        "support_reason_column": "",
        "baseline_availability_column": spec.get("baseline_availability_column", ""),
        "baseline_source_latest_time_column": spec.get("baseline_source_latest_time_column", ""),
    }


def _confidence_score(value: str) -> float:
    return {
        "confirmed": 1.0,
        "high": 1.0,
        "medium": 0.7,
        "low": 0.4,
        "unknown": 0.4,
        "pending": 0.7,
        "missing": 0.0,
    }.get(str(value).lower(), 0.4)


def _confidence_label(score: float) -> str:
    if score >= 1.0:
        return "high"
    if score >= 0.7:
        return "medium"
    return "low"


def _rolling_formula(column: str, kind: str, minutes: int) -> str:
    if kind in {"slope", "slope_per_s"}:
        return f"OLS slope of {column} versus elapsed seconds over past {minutes} min"
    if kind == "change":
        return f"{column}[t] - earliest {column} within past {minutes} min"
    return f"rolling {kind}({column}) over past {minutes} min"


def _rolling_interpretation(kind: str) -> str:
    return {
        "mean": "smoothed level",
        "slope": "elapsed-time trend",
        "slope_per_s": "elapsed-time trend",
        "change": "backward change",
        "std": "short-term variability",
        "range": "short-term range",
        "iqr": "robust short-term variability",
        "cv": "relative variability",
        "action_count": "control or signal action frequency",
    }[kind]


def _feature_quality(frame: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metadata in dictionary.itertuples(index=False):
        feature = str(metadata.feature)
        if feature not in frame:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        availability_column = str(getattr(metadata, "availability_column", ""))
        coverage_column = str(getattr(metadata, "coverage_column", ""))
        reason_column = str(getattr(metadata, "support_reason_column", ""))
        availability = (
            frame[availability_column].fillna(False).astype(bool)
            if availability_column and availability_column in frame
            else pd.Series(True, index=frame.index)
        )
        cycle_support_raw = (
            frame.assign(_available=availability)
            .groupby("cycle_id", dropna=False)["_available"]
            .mean()
            .to_dict()
            if "cycle_id" in frame
            else {}
        )
        cycle_support = {str(key): value for key, value in cycle_support_raw.items()}
        reason_counts = (
            frame[reason_column].value_counts(dropna=False).to_dict()
            if reason_column and reason_column in frame
            else {}
        )
        reason_counts = {str(key): value for key, value in reason_counts.items()}
        rows.append(
            {
                "feature": feature,
                "row_count": len(frame),
                "valid_count": int(values.notna().sum()),
                "missing_rate": float(values.isna().mean()),
                "infinite_count": int(np.isinf(values.to_numpy(dtype=float)).sum()),
                "valid_cycle_count": int(frame.loc[values.notna(), "cycle_id"].nunique()),
                "constant": bool(values.dropna().nunique() <= 1),
                "window_availability_rate": float(availability.mean()),
                "minimum_window_coverage": float(
                    pd.to_numeric(frame[coverage_column], errors="coerce").min()
                )
                if coverage_column and coverage_column in frame
                else np.nan,
                "support_reason_counts": json.dumps(
                    reason_counts, default=str
                )
                if reason_column and reason_column in frame
                else "{}",
                "cycle_availability_json": json.dumps(cycle_support, default=str),
            }
        )
    return pd.DataFrame(rows)
