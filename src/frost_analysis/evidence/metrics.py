"""Per-cycle Evidence metrics and the single quality-mask contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr, theilslopes

from .settings import EvidenceSettings

FloatArray = NDArray[np.float64]
BoolArray = NDArray[Any]
FeatureSpec = Sequence[tuple[str, str]]


def observed_mask(frame: pd.DataFrame, value_column: str) -> pd.Series[Any]:
    """Return finite, non-imputed observations for one Dataset value column."""
    source = value_column.removesuffix("__baseline_residual")
    quality_column = f"{source}__imputed"
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    finite = np.isfinite(values)
    imputed = frame[quality_column].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return pd.Series(finite & ~imputed, index=frame.index, dtype=bool)


def _feature_cycle_rows(
    frame: pd.DataFrame,
    metadata: Mapping[str, object],
    features: FeatureSpec,
    settings: EvidenceSettings,
) -> list[dict[str, object]]:
    stage = _frost_frame(frame)
    rows: list[dict[str, object]] = []
    for feature, direction in features:
        row: dict[str, object] = {
            **metadata,
            "feature": feature,
            "observed_fraction": np.nan,
            "spearman": np.nan,
            "signed_effect": np.nan,
            "trend_slope_per_min": np.nan,
            "onset_minutes": np.nan,
            "metric_status": "unavailable",
            "exclusion_reason": "",
        }
        if stage is None or stage.empty:
            row["exclusion_reason"] = "missing_frost_stage"
            rows.append(row)
            continue
        if "cycle_elapsed_seconds" not in stage:
            row["exclusion_reason"] = "missing_frost_elapsed"
            rows.append(row)
            continue
        residual_name = f"{feature}__baseline_residual"
        quality_name = f"{feature}__imputed"
        if residual_name not in stage:
            row["exclusion_reason"] = "missing_feature"
            rows.append(row)
            continue
        if quality_name not in stage:
            row["exclusion_reason"] = "missing_quality_column"
            rows.append(row)
            continue

        elapsed = _numeric(stage["cycle_elapsed_seconds"])
        residual = _numeric(stage[residual_name])
        observed = observed_mask(stage, residual_name).to_numpy(dtype=bool)
        observed_fraction = float(observed.sum() / len(stage))
        row["observed_fraction"] = observed_fraction
        metric_mask = observed & np.isfinite(elapsed)
        points = int(metric_mask.sum())
        if observed_fraction < settings.minimum_feature_coverage:
            row["exclusion_reason"] = "insufficient_feature_coverage"
            rows.append(row)
            continue
        if points < settings.minimum_feature_points:
            row["exclusion_reason"] = "insufficient_points"
            rows.append(row)
            continue
        values = residual[metric_mask]
        x_seconds = elapsed[metric_mask]
        if _is_constant(values):
            row["exclusion_reason"] = "constant_feature"
            rows.append(row)
            continue
        correlation = _spearman(x_seconds / 60.0, values)
        slope = _theil_sen(x_seconds / 60.0, values)
        if not np.isfinite(correlation) or not np.isfinite(slope):
            row["exclusion_reason"] = "insufficient_points"
            rows.append(row)
            continue
        signed = correlation if direction == "increase" else -correlation
        row.update(
            {
                "spearman": correlation,
                "signed_effect": signed,
                "trend_slope_per_min": slope,
                "onset_minutes": _onset_minutes(x_seconds, values, settings),
                "metric_status": "available",
            }
        )
        rows.append(row)
    return rows


def _future_rows(
    frame: pd.DataFrame,
    metadata: Mapping[str, object],
    features: FeatureSpec,
    settings: EvidenceSettings,
) -> list[dict[str, object]]:
    stage = _frost_frame(frame)
    rows: list[dict[str, object]] = []
    for feature, _ in features:
        for target in settings.targets:
            for horizon in settings.horizons_minutes:
                row: dict[str, object] = {
                    **metadata,
                    "feature": feature,
                    "target": target,
                    "horizon_minutes": horizon,
                    "effect": np.nan,
                    "valid_pairs": 0,
                    "pair_coverage": 0.0,
                    "metric_status": "unavailable",
                    "exclusion_reason": "",
                }
                reason = _future_column_error(stage, feature, target)
                if reason is not None:
                    row["exclusion_reason"] = reason
                    rows.append(row)
                    continue
                assert stage is not None
                if "cycle_elapsed_seconds" not in stage:
                    row["exclusion_reason"] = "missing_frost_elapsed"
                    rows.append(row)
                    continue
                effect, valid_pairs, coverage, reason = _future_metric(
                    stage, feature, target, horizon, settings
                )
                row.update(
                    {
                        "effect": effect,
                        "valid_pairs": valid_pairs,
                        "pair_coverage": coverage,
                        "exclusion_reason": reason,
                    }
                )
                if not reason:
                    row["metric_status"] = "available"
                rows.append(row)
    return rows


def _future_column_error(
    stage: pd.DataFrame | None, feature: str, target: str
) -> str | None:
    if stage is None or stage.empty:
        return "missing_frost_stage"
    feature_residual = f"{feature}__baseline_residual"
    feature_quality = f"{feature}__imputed"
    target_residual = f"{target}__baseline_residual"
    target_quality = f"{target}__imputed"
    if feature_residual not in stage:
        return "missing_feature"
    if feature_quality not in stage:
        return "missing_quality_column"
    if target_residual not in stage:
        return "missing_target"
    if target_quality not in stage:
        return "missing_target_quality"
    return None


def _future_metric(
    stage: pd.DataFrame,
    feature: str,
    target: str,
    horizon_minutes: int,
    settings: EvidenceSettings,
) -> tuple[float, int, float, str]:
    elapsed = _numeric(stage["cycle_elapsed_seconds"])
    anchors = _future_anchors(elapsed, horizon_minutes)
    theoretical = len(anchors)
    if theoretical == 0:
        return np.nan, 0, 0.0, "insufficient_pair_coverage"

    feature_values = _numeric(stage[f"{feature}__baseline_residual"])
    target_values = _numeric(stage[f"{target}__baseline_residual"])
    feature_observed = observed_mask(
        stage, f"{feature}__baseline_residual"
    ).to_numpy(dtype=bool)
    target_observed = observed_mask(
        stage, f"{target}__baseline_residual"
    ).to_numpy(dtype=bool)
    xs, changes = _valid_future_pairs(
        anchors, feature_values, target_values, feature_observed, target_observed
    )
    valid_pairs = len(xs)
    coverage = valid_pairs / theoretical
    if valid_pairs < settings.minimum_valid_pairs:
        return np.nan, valid_pairs, coverage, "insufficient_valid_pairs"
    if coverage < settings.minimum_pair_coverage:
        return np.nan, valid_pairs, coverage, "insufficient_pair_coverage"
    x_values = np.asarray(xs, dtype=float)
    change_values = np.asarray(changes, dtype=float)
    if _is_constant(x_values):
        return np.nan, valid_pairs, coverage, "constant_feature"
    if _is_constant(change_values):
        return np.nan, valid_pairs, coverage, "constant_target_change"
    effect = _spearman(x_values, change_values)
    if not np.isfinite(effect):
        return np.nan, valid_pairs, coverage, "constant_target_change"
    return effect, valid_pairs, coverage, ""


def _future_anchors(elapsed: FloatArray, horizon_minutes: int) -> list[tuple[int, int]]:
    finite_elapsed = np.isfinite(elapsed)
    elapsed_to_index: dict[float, int] = {}
    for index, value in enumerate(elapsed):
        if finite_elapsed[index] and float(value) not in elapsed_to_index:
            elapsed_to_index[float(value)] = index
    horizon_seconds = float(horizon_minutes * 60)
    anchors: list[tuple[int, int]] = []
    for index, value in enumerate(elapsed):
        if not finite_elapsed[index]:
            continue
        future_index = elapsed_to_index.get(float(value + horizon_seconds))
        if future_index is not None:
            anchors.append((index, future_index))
    return anchors


def _valid_future_pairs(
    anchors: Sequence[tuple[int, int]],
    feature_values: FloatArray,
    target_values: FloatArray,
    feature_observed: BoolArray,
    target_observed: BoolArray,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    changes: list[float] = []
    for index, future_index in anchors:
        if not feature_observed[index] or not target_observed[index]:
            continue
        if not target_observed[future_index]:
            continue
        change = target_values[future_index] - target_values[index]
        if np.isfinite(change):
            xs.append(float(feature_values[index]))
            changes.append(float(change))
    return xs, changes


def _pair_input(frame: pd.DataFrame, features: FeatureSpec) -> dict[str, dict[float, float]]:
    stage = _frost_frame(frame)
    if stage is None or stage.empty or "cycle_elapsed_seconds" not in stage:
        return {}
    elapsed = _numeric(stage["cycle_elapsed_seconds"])
    finite_elapsed = np.isfinite(elapsed)
    result: dict[str, dict[float, float]] = {}
    for feature, _ in features:
        residual_name = f"{feature}__baseline_residual"
        quality_name = f"{feature}__imputed"
        if residual_name not in stage or quality_name not in stage:
            continue
        values = _numeric(stage[residual_name])
        observed = observed_mask(stage, residual_name).to_numpy(dtype=bool)
        observed &= finite_elapsed
        result[feature] = {
            float(elapsed[index]): float(values[index])
            for index in range(len(stage))
            if observed[index]
        }
    return result


def _frost_frame(frame: pd.DataFrame) -> pd.DataFrame | None:
    if "cycle_stage" not in frame:
        return None
    return frame.loc[frame["cycle_stage"].eq("frost_development")].reset_index(drop=True)


def _numeric(series: pd.Series[Any]) -> FloatArray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return cast(FloatArray, values)


def _spearman(first: FloatArray, second: FloatArray) -> float:
    if len(first) < 2 or _is_constant(first) or _is_constant(second):
        return np.nan
    result = spearmanr(first, second, nan_policy="omit")
    return float(result.statistic)


def _theil_sen(x_values: FloatArray, y_values: FloatArray) -> float:
    if len(x_values) < 2 or _is_constant(x_values) or _is_constant(y_values):
        return np.nan
    return float(theilslopes(y_values, x_values).slope)


def _onset_minutes(
    elapsed_seconds: FloatArray,
    values: FloatArray,
    settings: EvidenceSettings,
) -> float:
    """Compute initial_frost_window_mad_onset_v1, not the legacy onset contract."""
    order = np.argsort(elapsed_seconds, kind="stable")
    elapsed = elapsed_seconds[order]
    ordered_values = values[order]
    baseline = ordered_values[elapsed <= settings.onset_window_seconds]
    if len(baseline) == 0:
        return np.nan
    center = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - center)))
    threshold = settings.onset_mad_multiplier * mad
    beyond = np.abs(ordered_values - center) > threshold
    for index, start in enumerate(elapsed):
        if start <= settings.onset_window_seconds or not beyond[index]:
            continue
        end = start + settings.onset_persistence_seconds
        window = (elapsed >= start) & (elapsed <= end)
        if (
            window.any()
            and elapsed[window].max() - start >= settings.onset_persistence_seconds
            and bool(np.all(beyond[window]))
        ):
            return float(start / 60.0)
    return np.nan


def _is_constant(values: FloatArray) -> bool:
    return len(values) == 0 or len(np.unique(values)) <= 1


__all__ = ["observed_mask"]
