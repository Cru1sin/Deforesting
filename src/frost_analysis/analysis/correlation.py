"""Replaceable correlation task for the processed candidate-channel dataset."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ..config import load_app_config
from ..data.registry import FeatureSpec, load_feature_registry
from ..schemas import AppConfig
from .screening import screen_candidate_channels


def run_correlation_analysis(config: AppConfig) -> Path:
    """Read only processed artifacts and publish one channel-level result table."""
    if not config.paths.processed_data.is_file():
        raise FileNotFoundError(f"processed data does not exist: {config.paths.processed_data}")
    if not config.paths.cycle_summary.is_file():
        raise FileNotFoundError(f"cycle summary does not exist: {config.paths.cycle_summary}")
    processed = pd.read_parquet(config.paths.processed_data)
    summary = pd.read_csv(config.paths.cycle_summary)
    specs = load_feature_registry(config.paths.registry)
    results = build_correlation_results(
        processed,
        summary,
        specs,
        methods=config.analysis.methods,
        lags_minutes=config.analysis.lags_minutes,
        targets=config.analysis.targets,
        minimum_cycles=config.analysis.minimum_cycles,
        features=config.analysis.features,
        modalities=config.analysis.modalities,
    )
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.paths.correlation_results, index=False)
    _write_state(config, results)
    return config.paths.correlation_results


def build_correlation_results(
    processed: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    specs: dict[str, FeatureSpec],
    *,
    methods: list[str],
    lags_minutes: list[int],
    targets: list[str],
    minimum_cycles: int,
    features: list[str] | None = None,
    modalities: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Combine four evidence layers without producing a composite rank."""
    internal = _screening_frame(processed)
    cycles = _screening_cycles(cycle_summary)
    registry = pd.DataFrame([vars(spec) for spec in specs.values()])
    if features:
        requested = {str(value) for value in features}
        registry = registry.loc[registry["canonical_name"].isin(requested)].copy()
    screen_config: dict[str, Any] = {
        "minimum_valid_cycles": minimum_cycles,
        "minimum_coverage": 0.7,
        "lead_horizons_minutes": [value for value in lags_minutes if value > 0],
    }
    # RGB qualification is intentionally not consulted by this sensor-only
    # task unless a future task explicitly adds a modality-aware analyzer.
    del modalities
    evidence = screen_candidate_channels(internal, cycles, registry, screen_config)
    if evidence.empty:
        return evidence
    trend = _trend_method_summary(internal, evidence["canonical_name"].tolist(), methods)
    lagged = _lagged_summary(
        internal,
        evidence["canonical_name"].tolist(),
        targets,
        methods,
        lags_minutes,
    )
    lagged_compact = _compact_lagged(lagged)
    evidence = evidence.drop(
        columns=[
            column
            for column in evidence
            if str(column).startswith("lag_") and column != "lag_valid_cycle_count"
        ],
        errors="ignore",
    )
    result = evidence.merge(trend, on="canonical_name", how="left", validate="one_to_one")
    result = result.merge(
        lagged_compact, on="canonical_name", how="left", validate="one_to_one"
    )
    result["correlation_methods"] = ",".join(methods)
    result["correlation_lags_minutes"] = ",".join(str(value) for value in lags_minutes)
    return result.sort_values(
        ["candidate_status", "physical_family", "canonical_name"], kind="stable"
    ).reset_index(drop=True)


def _screening_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result = result.rename(
        columns={
            "timestamp": "sensor_time",
            "cycle_status": "cycle_quality",
            "cycle_stage": "stage",
            "cycle_progress": "cycle_phase",
            "cycle_elapsed_seconds": "cycle_time_s",
        }
    )
    if "cycle_quality" not in result:
        result["cycle_quality"] = "partial"
    if "stage" not in result:
        result["stage"] = "partial"
    if "is_heating" not in result:
        result["is_heating"] = True
    result["cycle_quality"] = result["cycle_quality"].map(
        {"valid": "complete", "invalid": "abnormal", "incomplete": "partial"}
    ).fillna(result["cycle_quality"])
    return result


def _screening_cycles(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["quality_flag"] = result["cycle_status"].map(
        {"valid": "complete", "invalid": "abnormal", "incomplete": "partial"}
    ).fillna(result["cycle_status"])
    if "maximum_gap_seconds" not in result and "max_sensor_gap_seconds" in result:
        result["maximum_gap_seconds"] = result["max_sensor_gap_seconds"]
    for column in ("heating_start", "defrost_start", "defrost_end", "clean_end"):
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def _trend_method_summary(
    frame: pd.DataFrame, channels: list[str], methods: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for channel in channels:
        row: dict[str, object] = {"canonical_name": channel}
        signal = _signal(frame, channel)
        for method in methods:
            name = method.lower()
            if name not in {"pearson", "spearman"}:
                continue
            coefficients: list[float] = []
            p_values: list[float] = []
            samples = 0
            for _, group in _eligible(frame, required_variables=[channel]).groupby(
                "cycle_id", sort=False
            ):
                phase_source = group.get(
                    "cycle_phase", pd.Series(np.nan, index=group.index)
                )
                phase = pd.to_numeric(phase_source, errors="coerce")
                values = signal.loc[group.index]
                valid = phase.notna() & values.notna()
                if int(valid.sum()) < 3 or values.loc[valid].nunique() < 2:
                    continue
                x = phase.loc[valid].to_numpy(dtype=float)
                y = values.loc[valid].to_numpy(dtype=float)
                coefficient, p_value = _correlation(x, y, name)
                if np.isfinite(coefficient):
                    coefficients.append(coefficient)
                    p_values.append(p_value)
                    samples += int(valid.sum())
            row[f"{name}_trend_coefficient"] = _median_or_nan(coefficients)
            row[f"{name}_trend_p_value"] = _median_or_nan(p_values)
            row[f"{name}_trend_cycle_count"] = len(coefficients)
            row[f"{name}_trend_sample_count"] = samples
        rows.append(row)
    return pd.DataFrame(rows)


def _lagged_summary(
    frame: pd.DataFrame,
    channels: list[str],
    targets: list[str],
    methods: list[str],
    lags_minutes: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for channel in channels:
        row: dict[str, object] = {"canonical_name": channel}
        for target in targets:
            if target not in frame:
                continue
            for lag in lags_minutes:
                for method in methods:
                    name = method.lower()
                    if name not in {"pearson", "spearman"}:
                        continue
                    coefficients: list[float] = []
                    p_values: list[float] = []
                    sample_count = 0
                    eligible = _eligible(frame, required_variables=[channel, target])
                    for _, group in eligible.groupby("cycle_id", sort=False):
                        paired = _future_pairs(group, channel, target, lag)
                        if len(paired) < 3:
                            continue
                        coefficient, p_value = _correlation(
                            paired["source"].to_numpy(dtype=float),
                            paired["target"].to_numpy(dtype=float),
                            name,
                        )
                        if np.isfinite(coefficient):
                            coefficients.append(coefficient)
                            p_values.append(p_value)
                            sample_count += len(paired)
                    prefix = f"lag_{target}_{lag}m_{name}"
                    row[f"{prefix}_coefficient"] = _median_or_nan(coefficients)
                    row[f"{prefix}_p_value"] = _median_or_nan(p_values)
                    row[f"{prefix}_cycle_count"] = len(coefficients)
                    row[f"{prefix}_sample_count"] = sample_count
        rows.append(row)
    return pd.DataFrame(rows)


def _compact_lagged(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["canonical_name", "lagged_correlation_json"])
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        values = {
            str(column): _json_value(value)
            for column, value in row.items()
            if column != "canonical_name"
        }
        rows.append(
            {
                "canonical_name": row["canonical_name"],
                "lagged_correlation_json": json.dumps(values, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def _json_value(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return value


def _eligible(
    frame: pd.DataFrame,
    *,
    required_variables: list[str] | None = None,
) -> pd.DataFrame:
    quality = frame.get("cycle_quality", pd.Series("partial", index=frame.index))
    stage = frame.get("stage", pd.Series("partial", index=frame.index))
    mask = quality.eq("complete") & stage.isin(["stable_clean", "frost_development"])
    if "analysis_bin_available" in frame:
        mask &= frame["analysis_bin_available"].fillna(False).astype(bool)
    result = frame.loc[mask].sort_values(["cycle_id", "sensor_time"], kind="stable")
    for variable in required_variables or []:
        candidates = [f"{variable}__baseline_offset", variable]
        signal_name = next((column for column in candidates if column in result), "")
        if not signal_name:
            return result.iloc[0:0]
        available_cycles = set(
            result.loc[pd.to_numeric(result[signal_name], errors="coerce").notna(), "cycle_id"]
            .astype(str)
        )
        result = result.loc[result["cycle_id"].astype(str).isin(available_cycles)]
    return result


def _signal(frame: pd.DataFrame, channel: str) -> pd.Series:
    offset = f"{channel}__baseline_offset"
    if offset in frame and pd.to_numeric(frame[offset], errors="coerce").notna().any():
        return pd.to_numeric(frame[offset], errors="coerce")
    return pd.to_numeric(frame.get(channel, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _future_pairs(group: pd.DataFrame, channel: str, target: str, lag: int) -> pd.DataFrame:
    signal = _signal(group, channel)
    source = pd.DataFrame({"sensor_time": group["sensor_time"], "source": signal}).dropna()
    future = pd.DataFrame(
        {
            "future_time": group["sensor_time"] - pd.Timedelta(minutes=lag),
            "target": pd.to_numeric(group[target], errors="coerce"),
        }
    ).dropna()
    if source.empty or future.empty:
        return pd.DataFrame(columns=["source", "target"])
    matched = pd.merge_asof(
        source.sort_values("sensor_time"),
        future.sort_values("future_time"),
        left_on="sensor_time",
        right_on="future_time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=20),
    )
    return matched[["source", "target"]].dropna()


def _correlation(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan
    result = pearsonr(x, y) if method == "pearson" else spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _median_or_nan(values: list[float]) -> float:
    return float(np.median(values)) if values else np.nan


def _write_state(config: AppConfig, results: pd.DataFrame) -> None:
    config.paths.state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "analyze",
        "task": config.analysis.task,
        "date": config.date,
        "rows": len(results),
        "weighted_ranking": False,
        "created_at": time.time(),
    }
    (config.paths.state_dir / "analyze.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_from_config(path: Path) -> Path:
    """Small importable helper used by smoke tests and future task runners."""
    return run_correlation_analysis(load_app_config(path))
