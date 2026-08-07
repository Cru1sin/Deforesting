"""Target, lead-time, and incremental-model readiness evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import theilslopes

from .contracts import READINESS_SPLIT_COLUMNS, READINESS_SUMMARY_COLUMNS
from .metrics import observed_mask
from .settings import EvidenceSettings
from .summary import date_balanced_median

FloatArray = NDArray[np.float64]
BoolArray = NDArray[Any]
CycleInput = tuple[Mapping[str, object], pd.DataFrame]


@dataclass(frozen=True)
class _ModelData:
    expected: int
    predictors: FloatArray
    response: FloatArray


def audit_performance_target(
    record: Mapping[str, object],
    frame: pd.DataFrame,
    target: str,
    settings: EvidenceSettings,
) -> dict[str, object]:
    """Audit one performance target without reinterpreting cycle eligibility."""
    row = _target_row(record, target)
    stage = _frost_frame(frame)
    if stage.empty:
        return _unavailable(row, "missing_frost_stage")
    required = ("timestamp", target, f"{target}__imputed", f"{target}__baseline")
    if any(column not in stage for column in required):
        return _unavailable(row, "target_unavailable")

    baseline, reason = _cycle_baseline(stage[f"{target}__baseline"])
    if reason:
        row["primary_event_status"] = "baseline_unavailable"
        return _unavailable(row, reason)

    timestamps = pd.to_datetime(stage["timestamp"], errors="coerce")
    frost_start = timestamps.min()
    censor = _defrost_start(record)
    analysis_start = _baseline_end(record)
    if pd.isna(frost_start) or pd.isna(censor) or pd.isna(analysis_start):
        return _unavailable(row, "target_unavailable")
    analysis_period = timestamps.ge(analysis_start) & timestamps.lt(censor)
    stage = stage.loc[analysis_period].reset_index(drop=True)
    timestamps = timestamps.loc[analysis_period].reset_index(drop=True)
    observed = observed_mask(stage, target).to_numpy(dtype=bool)
    values = _numeric(stage[target])
    row["baseline_value"] = baseline
    row["target_observed_fraction"] = float(observed.sum() / len(stage)) if len(stage) else 0.0
    row["censor_elapsed_minutes"] = float((censor - frost_start).total_seconds() / 60.0)
    if observed.sum() == 0:
        row["primary_event_status"] = "target_unavailable"
        return _unavailable(row, "target_unavailable")

    degradation = (baseline - values) / abs(baseline)
    events: dict[float, float] = {}
    elapsed = cast(
        FloatArray,
        (timestamps - frost_start).dt.total_seconds().to_numpy(dtype=float) / 60.0,
    )
    for threshold in settings.event_thresholds:
        event = _persistent_start(
            timestamps,
            elapsed,
            observed & np.isfinite(degradation) & (degradation >= threshold),
            settings.event_persistence_seconds,
        )
        events[threshold] = event
        field = f"event_{int(round(threshold * 100))}_elapsed_minutes"
        if field in row:
            row[field] = event

    primary = events[settings.primary_event_threshold]
    row["primary_event_elapsed_minutes"] = primary
    row["primary_event_status"] = (
        "event_observed" if np.isfinite(primary) else "right_censored_at_legacy_defrost"
    )
    for horizon in (5, 10, 20):
        row[f"valid_pairs_{horizon}min"] = _target_pair_count(
            timestamps, observed, horizon
        )
    row["metric_status"] = "available"
    return row


def compute_signal_lead(
    frame: pd.DataFrame,
    feature: str,
    direction: str,
    performance_event_elapsed: float,
    performance_event_status: str,
    settings: EvidenceSettings,
) -> dict[str, object]:
    """Return statistical signal onset and descriptive lead for one cycle."""
    result: dict[str, object] = {
        "signal_onset_elapsed_minutes": np.nan,
        "performance_event_elapsed_minutes": performance_event_elapsed,
        "lead_minutes": np.nan,
        "lead_status": "signal_not_observed",
    }
    stage = _frost_frame(frame)
    residual = f"{feature}__baseline_residual"
    if stage.empty:
        result["lead_status"] = "missing_frost_stage"
        return result
    if residual not in stage or f"{feature}__imputed" not in stage or "timestamp" not in stage:
        result["lead_status"] = "signal_unavailable"
        return result

    timestamps = pd.to_datetime(stage["timestamp"], errors="coerce")
    start = timestamps.min()
    elapsed_seconds = (timestamps - start).dt.total_seconds().to_numpy(dtype=float)
    values = _numeric(stage[residual])
    observed = observed_mask(stage, residual).to_numpy(dtype=bool)
    reference_end = settings.signal_reference_minutes * 60.0
    reference = observed & np.isfinite(elapsed_seconds) & (elapsed_seconds <= reference_end)
    reference_values = values[reference]
    if not len(reference_values):
        result["lead_status"] = "invalid_initial_scale"
        return result
    reference_median = float(np.median(reference_values))
    mad = float(np.median(np.abs(reference_values - reference_median)))
    if not np.isfinite(mad) or mad <= 0:
        result["lead_status"] = "invalid_initial_scale"
        return result

    aligned = values - reference_median
    if direction == "decrease":
        aligned = -aligned
    rolling = _past_rolling_median(
        timestamps, aligned, observed, settings.signal_smoothing_seconds
    )
    qualifying = (
        observed
        & (elapsed_seconds > reference_end)
        & np.isfinite(rolling)
        & (rolling > settings.signal_mad_multiplier * mad)
    )
    elapsed_minutes = cast(FloatArray, elapsed_seconds / 60.0)
    onset = _persistent_start(
        timestamps, elapsed_minutes, qualifying, settings.signal_persistence_seconds
    )
    result["signal_onset_elapsed_minutes"] = onset
    if performance_event_status != "event_observed":
        result["lead_status"] = "performance_event_censored"
    elif np.isfinite(onset) and np.isfinite(performance_event_elapsed):
        result["lead_minutes"] = float(performance_event_elapsed - onset)
        result["lead_status"] = "available"
    return result


def compare_incremental_models(
    cycles: list[CycleInput],
    features: list[tuple[str, str]],
    target_audit: pd.DataFrame,
    settings: EvidenceSettings,
) -> pd.DataFrame:
    """Compare M0-M3 on common anchors and independent held-out cycles."""
    rows: list[dict[str, object]] = []
    for split_id, held_out, training in _leave_out_splits(cycles):
        for held_record, held_frame in held_out:
            rows.extend(
                _compare_held_cycle(
                    split_id,
                    held_record,
                    held_frame,
                    training,
                    features,
                    target_audit,
                    settings,
                )
            )
    return pd.DataFrame(rows, columns=READINESS_SPLIT_COLUMNS)


def _compare_held_cycle(
    split_id: str,
    held_record: Mapping[str, object],
    held_frame: pd.DataFrame,
    training: list[CycleInput],
    features: list[tuple[str, str]],
    target_audit: pd.DataFrame,
    settings: EvidenceSettings,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cycle_name = str(held_record.get("cycle_name", ""))
    held_date = str(held_record.get("experiment_date", ""))[:10]
    for feature, direction in features:
        for target in settings.targets:
            audit = _audit_row(target_audit, cycle_name, target)
            event_elapsed = _as_float(
                audit.get("primary_event_elapsed_minutes", np.nan)
            )
            event_status = str(audit.get("primary_event_status", "target_unavailable"))
            lead = compute_signal_lead(
                held_frame,
                feature,
                direction,
                event_elapsed,
                event_status,
                settings,
            )
            for horizon in settings.horizons_minutes:
                test_data = _model_data(
                    held_record, held_frame, feature, target, horizon, settings
                )
                row = _split_row(
                    split_id,
                    cycle_name,
                    held_date,
                    feature,
                    target,
                    horizon,
                    lead,
                    test_data,
                )
                reason = _sample_reason(test_data, settings)
                if reason:
                    row["exclusion_reason"] = reason
                    rows.append(row)
                    continue
                _fit_split_models(
                    row,
                    test_data,
                    training,
                    feature,
                    target,
                    horizon,
                    settings,
                )
                rows.append(row)
    return rows


def _fit_split_models(
    row: dict[str, object],
    test_data: _ModelData,
    training: list[CycleInput],
    feature: str,
    target: str,
    horizon: int,
    settings: EvidenceSettings,
) -> None:
    train_parts: list[_ModelData] = []
    train_dates: set[str] = set()
    for train_record, train_frame in training:
        part = _model_data(
            train_record, train_frame, feature, target, horizon, settings
        )
        if not _sample_reason(part, settings):
            train_parts.append(part)
            train_dates.add(str(train_record.get("experiment_date", ""))[:10])
    row["train_cycle_count"] = len(train_parts)
    row["train_date_count"] = len(train_dates)
    if not train_parts:
        row["exclusion_reason"] = (
            "no_training_cycles_after_holdout"
            if not training
            else "insufficient_training_cycles"
        )
        return

    train_x = np.concatenate([part.predictors for part in train_parts], axis=0)
    train_y = np.concatenate([part.response for part in train_parts])
    maes = [
        _ridge_mae(
            train_x[:, columns],
            train_y,
            test_data.predictors[:, columns],
            test_data.response,
            settings.ridge_alpha,
        )
        for columns in _model_columns(len(settings.context_features))
    ]
    row.update({f"mae_m{index}": value for index, value in enumerate(maes)})
    skills = (
        _skill(maes[1], maes[0]),
        _skill(maes[2], maes[1]),
        _skill(maes[3], maes[2]),
    )
    row.update(
        {
            "skill_context_vs_time": skills[0],
            "skill_level_vs_context": skills[1],
            "skill_dynamic_vs_level": skills[2],
        }
    )
    if all(np.isfinite(skills)):
        row["metric_status"] = "available"
    else:
        row["exclusion_reason"] = "invalid_skill_denominator"


def summarize_readiness(
    split_rows: pd.DataFrame,
    target_audit: pd.DataFrame,
    feature_metrics: pd.DataFrame,
    features: list[tuple[str, str]],
    settings: EvidenceSettings,
) -> pd.DataFrame:
    """Summarize independent validation units and assign readiness status."""
    rows: list[dict[str, object]] = []
    for feature, _ in features:
        trends = feature_metrics.loc[
            feature_metrics["feature"].eq(feature)
            & feature_metrics["metric_status"].eq("available")
        ]
        trend_effect, trend_cycles, trend_dates = date_balanced_median(
            trends, "signed_effect"
        )
        trend_consistency = _trend_consistency(trends)
        for target in settings.targets:
            for horizon in settings.horizons_minutes:
                selected = split_rows.loc[
                    split_rows["feature"].eq(feature)
                    & split_rows["target"].eq(target)
                    & split_rows["horizon_minutes"].eq(horizon)
                ]
                leads = selected.loc[selected["lead_status"].eq("available")]
                lead_units = _independent_units(leads, "lead_minutes")
                level_units = _independent_units(
                    selected.loc[selected["metric_status"].eq("available")],
                    "skill_level_vs_context",
                )
                dynamic_units = _independent_units(
                    selected.loc[selected["metric_status"].eq("available")],
                    "skill_dynamic_vs_level",
                )
                lead_values = lead_units.to_numpy(dtype=float)
                level_values = level_units.to_numpy(dtype=float)
                dynamic_values = dynamic_units.to_numpy(dtype=float)
                row: dict[str, object] = {
                    "feature": feature,
                    "target": target,
                    "horizon_minutes": horizon,
                    "trend_valid_cycle_count": trend_cycles,
                    "trend_valid_date_count": trend_dates,
                    "trend_effect": trend_effect,
                    "trend_direction_consistency": trend_consistency,
                    "lead_valid_cycle_count": int(leads["held_out_cycle"].nunique()),
                    "lead_median_minutes": _median(lead_values),
                    "lead_q25_minutes": _quantile(lead_values, 0.25),
                    "positive_lead_fraction": _positive_fraction(lead_values),
                    "level_skill_median": _median(level_values),
                    "level_improvement_fraction": _positive_fraction(level_values),
                    "dynamic_skill_median": _median(dynamic_values),
                    "dynamic_improvement_fraction": _positive_fraction(dynamic_values),
                    "readiness_status": "insufficient_validation_data",
                    "readiness_reason": "",
                }
                status, reason = _readiness_status(
                    target_audit,
                    selected,
                    row,
                    target,
                )
                row["readiness_status"] = status
                row["readiness_reason"] = reason
                rows.append(row)
    return pd.DataFrame(rows, columns=READINESS_SUMMARY_COLUMNS)


def _target_row(record: Mapping[str, object], target: str) -> dict[str, object]:
    return {
        "cycle_name": str(record.get("cycle_name", "")),
        "experiment_date": str(record.get("experiment_date", ""))[:10],
        "target": target,
        "baseline_value": np.nan,
        "target_observed_fraction": np.nan,
        "event_5_elapsed_minutes": np.nan,
        "event_10_elapsed_minutes": np.nan,
        "event_15_elapsed_minutes": np.nan,
        "primary_event_elapsed_minutes": np.nan,
        "primary_event_status": "target_unavailable",
        "censor_elapsed_minutes": np.nan,
        "valid_pairs_5min": 0,
        "valid_pairs_10min": 0,
        "valid_pairs_20min": 0,
        "metric_status": "unavailable",
        "exclusion_reason": "",
    }


def _unavailable(row: dict[str, object], reason: str) -> dict[str, object]:
    row["exclusion_reason"] = reason
    return row


def _frost_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "cycle_stage" not in frame:
        return frame.iloc[0:0].copy()
    return frame.loc[frame["cycle_stage"].eq("frost_development")].reset_index(drop=True)


def _cycle_baseline(series: pd.Series[Any]) -> tuple[float, str]:
    values = _numeric(series)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan, "baseline_unavailable"
    baseline = float(finite[0])
    if baseline <= 0:
        return np.nan, "baseline_nonpositive_or_zero"
    if not np.all(np.isclose(finite, baseline, rtol=1e-9, atol=1e-12)):
        return np.nan, "baseline_inconsistent"
    return baseline, ""


def _defrost_start(record: Mapping[str, object]) -> pd.Timestamp:
    return _boundary(record, "defrost_start")


def _baseline_end(record: Mapping[str, object]) -> pd.Timestamp:
    return _boundary(record, "baseline_end")


def _boundary(record: Mapping[str, object], name: str) -> pd.Timestamp:
    boundaries = record.get("boundaries")
    value = boundaries.get(name) if isinstance(boundaries, Mapping) else None
    parsed = pd.to_datetime(str(value), errors="coerce") if value is not None else pd.NaT
    return cast(pd.Timestamp, parsed)


def _persistent_start(
    timestamps: pd.Series[Any],
    elapsed_minutes: FloatArray,
    qualifying: BoolArray,
    duration_seconds: int,
) -> float:
    valid_times = pd.to_datetime(timestamps, errors="coerce")
    differences = valid_times.diff().dt.total_seconds().to_numpy(dtype=float)
    expected = float(np.nanmedian(differences[1:])) if len(differences) > 1 else np.nan
    run_start: int | None = None
    for index, value in enumerate(qualifying):
        contiguous = index == 0 or (
            np.isfinite(expected)
            and np.isfinite(differences[index])
            and np.isclose(differences[index], expected)
        )
        if not value or not contiguous:
            run_start = None
            if not value:
                continue
        if value and run_start is None:
            run_start = index
        if run_start is not None:
            span = (valid_times.iloc[index] - valid_times.iloc[run_start]).total_seconds()
            if span >= duration_seconds:
                return float(elapsed_minutes[run_start])
    return np.nan


def _target_pair_count(
    timestamps: pd.Series[Any], observed: BoolArray, horizon: int
) -> int:
    positions = {timestamp: index for index, timestamp in enumerate(timestamps)}
    delta = pd.Timedelta(minutes=horizon)
    return sum(
        bool(observed[index] and observed[positions[timestamp + delta]])
        for index, timestamp in enumerate(timestamps)
        if timestamp + delta in positions
    )


def _past_rolling_median(
    timestamps: pd.Series[Any],
    values: FloatArray,
    observed: BoolArray,
    window_seconds: int,
) -> FloatArray:
    result = np.full(len(values), np.nan)
    for index, timestamp in enumerate(timestamps):
        within = (
            timestamps.ge(timestamp - pd.Timedelta(seconds=window_seconds))
            & timestamps.le(timestamp)
        ).to_numpy(dtype=bool)
        selected = values[within & observed]
        if len(selected):
            result[index] = float(np.median(selected))
    return result


def _numeric(series: pd.Series[Any]) -> FloatArray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _leave_out_splits(
    cycles: list[CycleInput],
) -> list[tuple[str, list[CycleInput], list[CycleInput]]]:
    dates = {str(record.get("experiment_date", ""))[:10] for record, _ in cycles}
    splits: list[tuple[str, list[CycleInput], list[CycleInput]]] = []
    if len(dates) > 1:
        for date in sorted(dates):
            held = [item for item in cycles if str(item[0].get("experiment_date", ""))[:10] == date]
            training = [
                item
                for item in cycles
                if str(item[0].get("experiment_date", ""))[:10] != date
            ]
            splits.append((f"date:{date}", held, training))
    else:
        for item in cycles:
            name = str(item[0].get("cycle_name", ""))
            training = [other for other in cycles if other is not item]
            splits.append((f"cycle:{name}", [item], training))
    return splits


def _audit_row(table: pd.DataFrame, cycle: str, target: str) -> dict[str, object]:
    selected = table.loc[table["cycle_name"].eq(cycle) & table["target"].eq(target)]
    if selected.empty:
        return {}
    return {str(key): value for key, value in selected.iloc[0].to_dict().items()}


def _model_data(
    record: Mapping[str, object],
    frame: pd.DataFrame,
    feature: str,
    target: str,
    horizon: int,
    settings: EvidenceSettings,
) -> _ModelData:
    stage = _frost_frame(frame)
    empty = np.empty((0, len(settings.context_features) + 4), dtype=float)
    if stage.empty or "timestamp" not in stage:
        return _ModelData(0, empty, np.empty(0, dtype=float))
    timestamps = pd.to_datetime(stage["timestamp"], errors="coerce")
    frost_start = timestamps.min()
    analysis_start = _baseline_end(record)
    if pd.isna(frost_start) or pd.isna(analysis_start):
        return _ModelData(0, empty, np.empty(0, dtype=float))
    analysis_period = timestamps.ge(analysis_start)
    stage = stage.loc[analysis_period].reset_index(drop=True)
    timestamps = timestamps.loc[analysis_period].reset_index(drop=True)
    positions = {
        timestamp: index
        for index, timestamp in enumerate(timestamps)
        if not pd.isna(timestamp)
    }
    delta = pd.Timedelta(minutes=horizon)
    anchors = [
        (index, positions[timestamp + delta])
        for index, timestamp in enumerate(timestamps)
        if not pd.isna(timestamp) and timestamp + delta in positions
    ]
    required = [
        target,
        f"{target}__baseline",
        f"{target}__baseline_residual",
        f"{target}__imputed",
        f"{feature}__baseline_residual",
        f"{feature}__imputed",
    ]
    required.extend(settings.context_features)
    required.extend(f"{name}__imputed" for name in settings.context_features)
    if any(column not in stage for column in required):
        return _ModelData(len(anchors), empty, np.empty(0, dtype=float))
    baseline, reason = _cycle_baseline(stage[f"{target}__baseline"])
    if reason:
        return _ModelData(len(anchors), empty, np.empty(0, dtype=float))
    elapsed = (timestamps - frost_start).dt.total_seconds().to_numpy(dtype=float) / 60.0
    target_raw = _numeric(stage[target])
    target_residual = _numeric(stage[f"{target}__baseline_residual"])
    target_observed = observed_mask(stage, target).to_numpy(dtype=bool)
    feature_column = f"{feature}__baseline_residual"
    feature_values = _numeric(stage[feature_column])
    feature_observed = observed_mask(stage, feature_column).to_numpy(dtype=bool)
    contexts = [_numeric(stage[name]) for name in settings.context_features]
    context_observed = [
        observed_mask(stage, name).to_numpy(dtype=bool) for name in settings.context_features
    ]
    slopes = _past_slopes(
        timestamps,
        feature_values,
        feature_observed,
        settings.dynamic_window_minutes,
    )

    predictors: list[list[float]] = []
    response: list[float] = []
    for current, future in anchors:
        complete = (
            np.isfinite(elapsed[current])
            and feature_observed[current]
            and target_observed[current]
            and target_observed[future]
            and np.isfinite(target_residual[current])
            and np.isfinite(slopes[current])
            and all(mask[current] for mask in context_observed)
        )
        if not complete:
            continue
        future_degradation = (target_raw[current] - target_raw[future]) / abs(baseline)
        values = [
            elapsed[current],
            *(context[current] for context in contexts),
            target_residual[current],
            feature_values[current],
            slopes[current],
        ]
        if np.isfinite(future_degradation) and np.all(np.isfinite(values)):
            predictors.append([float(value) for value in values])
            response.append(float(future_degradation))
    return _ModelData(
        len(anchors),
        np.asarray(predictors, dtype=float).reshape((-1, empty.shape[1])),
        np.asarray(response, dtype=float),
    )


def _past_slopes(
    timestamps: pd.Series[Any],
    values: FloatArray,
    observed: BoolArray,
    window_minutes: int,
) -> FloatArray:
    result = np.full(len(values), np.nan)
    for index, timestamp in enumerate(timestamps):
        within = (
            timestamps.ge(timestamp - pd.Timedelta(minutes=window_minutes))
            & timestamps.le(timestamp)
        ).to_numpy(dtype=bool)
        selected = within & observed
        selected_times = (
            (timestamps.loc[selected] - timestamps.loc[selected].min())
            .dt.total_seconds()
            .to_numpy(dtype=float)
            / 60.0
        )
        selected_values = values[selected]
        if len(selected_values) >= 2 and np.unique(selected_times).size >= 2:
            result[index] = float(theilslopes(selected_values, selected_times).slope)
    return result


def _sample_reason(data: _ModelData, settings: EvidenceSettings) -> str:
    if data.expected == 0:
        return "insufficient_pair_coverage"
    if len(data.response) < settings.minimum_valid_pairs:
        return "insufficient_valid_pairs"
    if len(data.response) / data.expected < settings.minimum_pair_coverage:
        return "insufficient_pair_coverage"
    return ""


def _split_row(
    split_id: str,
    cycle: str,
    date: str,
    feature: str,
    target: str,
    horizon: int,
    lead: Mapping[str, object],
    data: _ModelData,
) -> dict[str, object]:
    return {
        "split_id": split_id,
        "held_out_cycle": cycle,
        "held_out_date": date,
        "feature": feature,
        "target": target,
        "horizon_minutes": horizon,
        **lead,
        "expected_anchor_count": data.expected,
        "valid_anchor_count": len(data.response),
        "anchor_coverage": len(data.response) / data.expected if data.expected else 0.0,
        "train_cycle_count": 0,
        "train_date_count": 0,
        "mae_m0": np.nan,
        "mae_m1": np.nan,
        "mae_m2": np.nan,
        "mae_m3": np.nan,
        "skill_context_vs_time": np.nan,
        "skill_level_vs_context": np.nan,
        "skill_dynamic_vs_level": np.nan,
        "metric_status": "unavailable",
        "exclusion_reason": "",
    }


def _model_columns(context_count: int) -> tuple[list[int], list[int], list[int], list[int]]:
    target_index = context_count + 1
    feature_index = target_index + 1
    slope_index = feature_index + 1
    return (
        [0],
        list(range(target_index + 1)),
        list(range(feature_index + 1)),
        list(range(slope_index + 1)),
    )


def _ridge_mae(
    train_x: FloatArray,
    train_y: FloatArray,
    test_x: FloatArray,
    test_y: FloatArray,
    alpha: float,
) -> float:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    standardized_train = (train_x - mean) / scale
    standardized_test = (test_x - mean) / scale
    target_mean = float(train_y.mean())
    centred_target = train_y - target_mean
    identity = np.eye(standardized_train.shape[1], dtype=float)
    coefficients = np.linalg.solve(
        standardized_train.T @ standardized_train + alpha * identity,
        standardized_train.T @ centred_target,
    )
    predictions = target_mean + standardized_test @ coefficients
    return float(np.mean(np.abs(predictions - test_y)))


def _skill(candidate_mae: float, reference_mae: float) -> float:
    if not np.isfinite(reference_mae) or reference_mae <= 0:
        return np.nan
    return float(1.0 - candidate_mae / reference_mae)


def _independent_units(frame: pd.DataFrame, value_column: str) -> pd.Series[Any]:
    if frame.empty or value_column not in frame:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[value_column], errors="coerce")
    selected = frame.loc[np.isfinite(values.to_numpy(dtype=float)), [
        "held_out_cycle",
        "held_out_date",
        value_column,
    ]].copy()
    if selected.empty:
        return pd.Series(dtype=float)
    selected[value_column] = pd.to_numeric(selected[value_column], errors="coerce")
    cycle_values = selected.groupby(
        ["held_out_date", "held_out_cycle"], sort=False
    )[value_column].median()
    if cycle_values.index.get_level_values("held_out_date").nunique() > 1:
        grouped = cycle_values.groupby(level="held_out_date", sort=False).median()
        return pd.Series(grouped.to_numpy(dtype=float), index=grouped.index, dtype=float)
    return pd.Series(cycle_values.to_numpy(dtype=float), dtype=float)


def _trend_consistency(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    cycle_values = frame.groupby(
        ["experiment_date", "cycle_name"], sort=False
    )["signed_effect"].median()
    date_values = cycle_values.groupby(level="experiment_date", sort=False).median()
    return float((date_values > 0).mean())


def _readiness_status(
    audits: pd.DataFrame,
    selected: pd.DataFrame,
    row: Mapping[str, object],
    target: str,
) -> tuple[str, str]:
    target_available = not audits.loc[
        audits["target"].eq(target) & audits["metric_status"].eq("available")
    ].empty
    anchors_available = (
        not selected.empty
        and pd.to_numeric(selected["expected_anchor_count"], errors="coerce").max() > 0
    )
    if not target_available or not anchors_available:
        return "target_not_evaluable", "target_or_horizon_unavailable"
    if (
        selected.loc[selected["metric_status"].eq("available")].empty
        or _as_int(row["trend_valid_cycle_count"]) == 0
        or _as_int(row["lead_valid_cycle_count"]) == 0
    ):
        reason = (
            "no_training_cycles_after_holdout"
            if selected["exclusion_reason"].eq("no_training_cycles_after_holdout").any()
            else "insufficient_independent_validation"
        )
        return "insufficient_validation_data", reason
    lead_q25 = _as_float(row["lead_q25_minutes"])
    if lead_q25 <= 0:
        return "state_candidate", "lead_not_stably_positive"
    level_stable = (
        _as_float(row["level_skill_median"]) > 0
        and _as_float(row["level_improvement_fraction"]) > 0.5
    )
    if not level_stable:
        return "no_incremental_prediction", "level_increment_not_stable"
    dynamic_stable = (
        _as_float(row["dynamic_skill_median"]) > 0
        and _as_float(row["dynamic_improvement_fraction"]) > 0.5
    )
    if dynamic_stable:
        return "dynamic_prediction_candidate", ""
    return "static_prediction_candidate", "dynamic_increment_not_stable"


def _median(values: FloatArray) -> float:
    return float(np.median(values)) if len(values) else np.nan


def _quantile(values: FloatArray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if len(values) else np.nan


def _positive_fraction(values: FloatArray) -> float:
    return float((values > 0).mean()) if len(values) else np.nan


def _as_float(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _as_int(value: object) -> int:
    return int(_as_float(value))
