"""Cycle-local stable clean-window selection and baseline offsets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineResult:
    frame: pd.DataFrame
    cycles: pd.DataFrame
    baselines: pd.DataFrame
    candidates: pd.DataFrame


def select_clean_baselines(  # noqa: C901 - selection and auditable fallback are coupled
    frame: pd.DataFrame,
    cycles: pd.DataFrame,
    sensor_columns: list[str],
    anchor_columns: list[str],
    config: dict[str, Any],
) -> BaselineResult:
    """Choose an early stable window per eligible cycle, with transparent fallback."""
    labeled = frame.copy()
    updated_cycles = cycles.copy()
    baseline_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    epsilon = float(config.get("relative_epsilon", 0.001))
    initial_columns: dict[str, pd.Series] = {}
    for column in sensor_columns:
        initial_columns[f"{column}__baseline_offset"] = pd.Series(
            np.nan, index=labeled.index, dtype=float
        )
        initial_columns[f"{column}__baseline_relative"] = pd.Series(
            np.nan, index=labeled.index, dtype=float
        )
        initial_columns[f"{column}__relative_valid"] = pd.Series(
            False, index=labeled.index, dtype=bool
        )
        initial_columns[f"{column}__baseline_available"] = pd.Series(
            False, index=labeled.index, dtype=bool
        )
        initial_columns[f"{column}__baseline_source_latest_time"] = pd.Series(
            pd.NaT, index=labeled.index, dtype="datetime64[ns]"
        )
    labeled = pd.concat([labeled, pd.DataFrame(initial_columns)], axis=1)

    for cycle_index, cycle in updated_cycles.iterrows():
        if cycle.get("quality_flag") != "complete":
            continue
        cycle_id = str(cycle["cycle_id"])
        heating_start = pd.Timestamp(cycle["heating_start"])
        stable_start = pd.Timestamp(cycle["stable_heating_start"])
        defrost_start = pd.Timestamp(cycle["defrost_start"])
        search_start = max(
            stable_start,
            heating_start
            + pd.Timedelta(seconds=float(config.get("recovery_exclusion_seconds", 180))),
        )
        fraction = float(config.get("search_end_fraction", 0.35))
        search_end = search_start + (defrost_start - search_start) * fraction
        duration = float(config.get("window_seconds", 300))
        step = float(config.get("step_seconds", 60))
        candidates = _candidate_windows(
            labeled,
            cycle_id,
            search_start,
            search_end,
            duration,
            step,
            anchor_columns,
            config,
        )
        for candidate in candidates:
            summary = {key: value for key, value in candidate.items() if key != "anchor_evidence"}
            for anchor_evidence in cast(list[dict[str, object]], candidate["anchor_evidence"]):
                candidate_rows.append({"cycle_id": cycle_id, **summary, **anchor_evidence})
        if not candidates:
            reason = (
                "anchor_values_missing"
                if _anchors_missing(labeled, cycle_id, anchor_columns, search_start, search_end)
                else "insufficient_search_duration"
            )
            for column in sensor_columns:
                baseline_rows.append(_failed_baseline(cycle_id, column, reason))
            continue
        qualified = [candidate for candidate in candidates if bool(candidate["qualified"])]
        selected = (
            qualified[0]
            if qualified
            else min(candidates, key=lambda item: cast(float, item["stability_score"]))
        )
        selection_status = "selected" if qualified else "low_confidence_fallback"
        quality = "good" if qualified else "low_confidence"
        clean_start = cast(pd.Timestamp, selected["candidate_start"])
        clean_end = cast(pd.Timestamp, selected["candidate_end"])
        updated_cycles.loc[cycle_index, "clean_start"] = clean_start
        updated_cycles.loc[cycle_index, "clean_end"] = clean_end
        selected_mask = labeled["cycle_id"].eq(cycle_id) & labeled["sensor_time"].between(
            clean_start, clean_end
        )
        labeled.loc[selected_mask, "stage"] = "stable_clean"
        cycle_mask = labeled["cycle_id"].eq(cycle_id)
        for column in sensor_columns:
            values = pd.to_numeric(labeled.loc[selected_mask, column], errors="coerce").dropna()
            if values.empty:
                baseline_rows.append(
                    _failed_baseline(
                        cycle_id, column, "baseline_values_missing", clean_start, clean_end
                    )
                )
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            median = float(values.median())
            mad = float((values - median).abs().median() * 1.4826)
            scale = max(mad, std, epsilon)
            available_mask = cycle_mask & labeled["sensor_time"].ge(clean_end)
            labeled.loc[cycle_mask, f"{column}__baseline_source_latest_time"] = clean_end
            labeled.loc[available_mask, f"{column}__baseline_available"] = True
            labeled.loc[available_mask, f"{column}__baseline_offset"] = (
                pd.to_numeric(labeled.loc[available_mask, column], errors="coerce") - mean
            )
            valid_relative = abs(mean) >= epsilon
            if valid_relative:
                labeled.loc[available_mask, f"{column}__baseline_relative"] = labeled.loc[
                    available_mask, f"{column}__baseline_offset"
                ] / abs(mean)
                labeled.loc[available_mask, f"{column}__relative_valid"] = True
            baseline_rows.append(
                {
                    "cycle_id": cycle_id,
                    "variable": column,
                    "clean_start": clean_start,
                    "clean_end": clean_end,
                    "baseline_mean": mean,
                    "baseline_std": std,
                    "baseline_robust_scale": scale,
                    "stability_score": cast(float, selected["stability_score"]),
                    "anchor_pass_fraction": cast(float, selected["anchor_pass_fraction"]),
                    "failed_anchor_count": cast(int, selected["failed_anchor_count"]),
                    "selection_method": "earliest_stable_elapsed_window"
                    if qualified
                    else "minimum_score_fallback",
                    "selection_status": selection_status,
                    "quality_flag": quality,
                    "failure_reason": "" if qualified else "anchor_stability_not_met",
                    "relative_offset_valid": valid_relative,
                }
            )
    return BaselineResult(
        labeled,
        updated_cycles,
        pd.DataFrame(baseline_rows, columns=_baseline_columns()),
        pd.DataFrame(candidate_rows),
    )


def _candidate_windows(
    frame: pd.DataFrame,
    cycle_id: str,
    search_start: pd.Timestamp,
    search_end: pd.Timestamp,
    duration_s: float,
    step_s: float,
    anchors: list[str],
    config: dict[str, Any],
) -> list[dict[str, object]]:
    if search_end - search_start < pd.Timedelta(seconds=duration_s):
        return []
    candidates: list[dict[str, object]] = []
    current = search_start
    while current + pd.Timedelta(seconds=duration_s) <= search_end:
        end = current + pd.Timedelta(seconds=duration_s)
        window = frame.loc[
            frame["cycle_id"].eq(cycle_id) & frame["sensor_time"].between(current, end)
        ]
        evidence: list[dict[str, object]] = []
        minimum_coverage = float(config.get("minimum_coverage", 0.8))
        maximum_score = float(
            config.get(
                "maximum_anchor_stability_score",
                config.get("maximum_stability_score", 0.12),
            )
        )
        scale_floors = {
            str(key): float(value)
            for key, value in dict(config.get("anchor_scale_floors", {})).items()
        }
        change_limits = {
            str(key): float(value)
            for key, value in dict(config.get("anchor_maximum_absolute_change", {})).items()
        }
        required = {str(column) for column in config.get("required_anchor_columns", anchors)}
        for column in anchors:
            values = (
                pd.to_numeric(window[column], errors="coerce")
                if column in window
                else pd.Series(np.nan, index=window.index)
            )
            elapsed = window["sensor_time"].diff().dt.total_seconds().dropna()
            nominal = float(elapsed[elapsed.gt(0)].median()) if elapsed.gt(0).any() else 1.0
            expected = max(2.0, duration_s / nominal + 1)
            coverage = min(1.0, float(values.notna().sum() / expected))
            valid = values.notna()
            if valid.sum() < 2:
                evidence.append(
                    _anchor_evidence(
                        column,
                        coverage,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        False,
                        "insufficient_observations",
                    )
                )
                continue
            seconds = (
                (window.loc[valid, "sensor_time"] - window.loc[valid, "sensor_time"].iloc[0])
                .dt.total_seconds()
                .to_numpy()
            )
            observed = values.loc[valid].to_numpy(dtype=float)
            slope = float(np.polyfit(seconds, observed, 1)[0]) if np.ptp(seconds) > 0 else 0.0
            fitted = observed[0] + slope * seconds
            residual = observed - fitted
            residual_median = float(np.median(residual))
            variation = float(np.median(np.abs(residual - residual_median)) * 1.4826)
            scale = max(
                scale_floors.get(column, float(config.get("default_anchor_scale_floor", 1.0))),
                1e-12,
            )
            total_change = slope * duration_s
            score = abs(total_change) / scale + variation / scale
            change_limit = change_limits.get(column, np.inf)
            change_ok = abs(total_change) <= change_limit
            passed = coverage >= minimum_coverage and score <= maximum_score and change_ok
            reason = (
                ""
                if passed
                else "insufficient_coverage"
                if coverage < minimum_coverage
                else "absolute_change_exceeds_threshold"
                if not change_ok
                else "drift_or_variation_exceeds_threshold"
            )
            evidence.append(
                _anchor_evidence(
                    column,
                    coverage,
                    slope,
                    total_change,
                    variation,
                    score,
                    passed,
                    reason,
                    change_limit,
                )
            )
        observed_evidence = [
            row for row in evidence if np.isfinite(float(str(row["anchor_stability_score"])))
        ]
        if observed_evidence:
            passed_count = sum(bool(row["anchor_pass"]) for row in evidence)
            pass_fraction = passed_count / max(1, len(anchors))
            required_failures = sum(
                row["anchor"] in required and not bool(row["anchor_pass"]) for row in evidence
            )
            minimum_pass_fraction = float(config.get("minimum_anchor_pass_fraction", 1.0))
            candidates.append(
                {
                    "candidate_start": current,
                    "candidate_end": end,
                    "stability_score": float(
                        max(cast(float, row["anchor_stability_score"]) for row in observed_evidence)
                    ),
                    "minimum_anchor_coverage": float(
                        min(cast(float, row["anchor_coverage"]) for row in evidence)
                    ),
                    "anchor_count": len(observed_evidence),
                    "anchor_pass_fraction": pass_fraction,
                    "failed_anchor_count": len(anchors) - passed_count,
                    "required_anchor_failures": required_failures,
                    "qualified": bool(
                        pass_fraction >= minimum_pass_fraction and required_failures == 0
                    ),
                    "anchor_evidence": evidence,
                }
            )
        current += pd.Timedelta(seconds=step_s)
    return candidates


def _anchor_evidence(
    anchor: str,
    coverage: float,
    slope: float,
    total_change: float,
    variation: float,
    score: float,
    passed: bool,
    reason: str,
    change_limit: float = np.inf,
) -> dict[str, object]:
    return {
        "anchor": anchor,
        "anchor_coverage": coverage,
        "anchor_slope_per_s": slope,
        "anchor_total_change": total_change,
        "anchor_change_limit": change_limit,
        "anchor_variation": variation,
        "anchor_stability_score": score,
        "anchor_pass": passed,
        "anchor_failure_reason": reason,
    }


def _anchors_missing(
    frame: pd.DataFrame, cycle_id: str, anchors: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> bool:
    window = frame.loc[frame["cycle_id"].eq(cycle_id) & frame["sensor_time"].between(start, end)]
    return not any(
        column in window and pd.to_numeric(window[column], errors="coerce").notna().any()
        for column in anchors
    )


def _failed_baseline(
    cycle_id: str,
    variable: str,
    reason: str,
    clean_start: pd.Timestamp | None = None,
    clean_end: pd.Timestamp | None = None,
) -> dict[str, object]:
    return {
        "cycle_id": cycle_id,
        "variable": variable,
        "clean_start": clean_start,
        "clean_end": clean_end,
        "baseline_mean": np.nan,
        "baseline_std": np.nan,
        "baseline_robust_scale": np.nan,
        "stability_score": np.nan,
        "anchor_pass_fraction": np.nan,
        "failed_anchor_count": np.nan,
        "selection_method": "none",
        "selection_status": "failed",
        "quality_flag": "failed",
        "failure_reason": reason,
        "relative_offset_valid": False,
    }


def _baseline_columns() -> list[str]:
    return [
        "cycle_id",
        "variable",
        "clean_start",
        "clean_end",
        "baseline_mean",
        "baseline_std",
        "baseline_robust_scale",
        "stability_score",
        "anchor_pass_fraction",
        "failed_anchor_count",
        "selection_method",
        "selection_status",
        "quality_flag",
        "failure_reason",
        "relative_offset_valid",
    ]
