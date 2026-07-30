"""Pipeline 2: turn prepared observations into one reusable analysis dataset."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.artifacts import write_dataframe
from ..data.registry import FeatureSpec, load_feature_registry
from ..processing.baseline import select_clean_baselines
from ..processing.features import engineer_features
from ..processing.missing import handle_missing_data
from ..processing.resample import resample_data
from ..schemas import AppConfig


def process_dataset(config: AppConfig) -> Path:
    """Read only prepared artifacts and publish ``processed_data.parquet``."""
    prepared = load_prepared_data(config.paths.prepared_data)
    cycle_summary = load_cycle_summary(config.paths.cycle_summary)
    specs = load_feature_registry(config.paths.registry)
    processed, updated_summary = build_processed_dataset(prepared, cycle_summary, specs, config)
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(processed, config.paths.processed_data)
    updated_summary.to_csv(config.paths.cycle_summary, index=False)
    _write_state(config, processed, updated_summary)
    return config.paths.processed_data


def load_prepared_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"prepared data does not exist: {path}")
    frame = pd.read_parquet(path)
    required = {"timestamp", "cycle_id", "cycle_stage", "cycle_status"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"prepared data missing required columns: {missing}")
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("prepared timestamps must be sorted")
    return frame


def load_cycle_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"cycle summary does not exist: {path}")
    result = pd.read_csv(path)
    required = {"cycle_id", "cycle_status", "heating_start", "defrost_start"}
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"cycle summary missing required columns: {missing}")
    for column in ("heating_start", "defrost_start", "defrost_end", "clean_start", "clean_end"):
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def build_processed_dataset(
    prepared: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    specs: dict[str, FeatureSpec],
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    internal = _internal_cycle_columns(prepared)
    active = [
        spec.canonical_name
        for spec in specs.values()
        if spec.data_role in {"X", "C"}
        and spec.analysis_enabled
        and spec.canonical_name in internal
    ]
    control = [
        spec.canonical_name
        for spec in specs.values()
        if spec.data_role == "C" and spec.canonical_name in internal
    ]
    target_columns = {
        "heating_capacity",
        "power_total",
        "cop",
        "water_heating_capacity",
        "water_cop",
    }
    filled = handle_missing_data(
        internal,
        active,
        control,
        continuous_max_gap_seconds=config.process.continuous_max_gap_seconds,
        control_max_gap_seconds=config.process.control_max_gap_seconds,
        target_columns=target_columns,
    )
    anchors = [
        name
        for name in (
            "evaporating_temperature",
            "compressor_frequency",
            "water_in_temperature",
            "water_out_temperature",
        )
        if name in active
    ]
    baseline = select_clean_baselines(
        filled,
        _internal_cycle_summary(cycle_summary),
        active,
        list(anchors or active[:4]),
        config.process.baseline_settings,
    )
    sampled = resample_data(
        baseline.frame.rename(columns={"sensor_time": "timestamp"}),
        interval_seconds=config.process.resample_interval_seconds,
        numeric_columns=active,
        control_columns=control,
        state_columns=["cycle_status", "cycle_stage", "heating_mode"],
    )
    sampled = sampled.rename(columns={"timestamp": "sensor_time"})
    feature_result = engineer_features(
        sampled,
        {name: specs[name] for name in active},
        windows_minutes=config.process.windows_minutes,
        minimum_coverage=config.process.minimum_coverage,
    )
    processed = feature_result.frame.rename(
        columns={
            "sensor_time": "timestamp",
            "cycle_quality": "cycle_status",
            "stage": "cycle_stage",
            "cycle_phase": "cycle_progress",
            "cycle_time_s": "cycle_elapsed_seconds",
        }
    )
    image_columns = [column for column in sampled if str(column).startswith("image_")]
    if image_columns:
        image_frame = sampled[["sensor_time", "cycle_id", *image_columns]].rename(
            columns={"sensor_time": "timestamp"}
        ).drop_duplicates(["timestamp", "cycle_id"], keep="last")
        processed = processed.merge(
            image_frame, on=["timestamp", "cycle_id"], how="left", validate="many_to_one"
        )
    updated_summary = _update_cycle_summary(cycle_summary, baseline.cycles, processed)
    return processed.sort_values("timestamp", kind="stable").reset_index(drop=True), updated_summary


def _internal_cycle_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().rename(columns={"timestamp": "sensor_time"})
    result["cycle_quality"] = result["cycle_status"].map(
        {
            "valid": "complete",
            "long_gap": "contaminated",
            "incomplete": "partial",
            "invalid_mode": "abnormal",
        }
    )
    result["stage"] = result["cycle_stage"]
    cycle_progress = result.get("cycle_progress", pd.Series(np.nan, index=result.index))
    result["cycle_phase"] = pd.to_numeric(cycle_progress, errors="coerce")
    elapsed = result.get("cycle_elapsed_seconds", pd.Series(np.nan, index=result.index))
    result["cycle_time_s"] = pd.to_numeric(elapsed, errors="coerce")
    result["cycle_gap_contaminated"] = result["cycle_status"].eq("long_gap")
    return result


def _internal_cycle_summary(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["quality_flag"] = result["cycle_status"].map(
        {
            "valid": "complete",
            "long_gap": "contaminated",
            "incomplete": "partial",
            "invalid_mode": "abnormal",
        }
    )
    result["exclusion_reason"] = result.get(
        "cycle_status_reason", pd.Series("", index=result.index)
    )
    if "max_sensor_gap_seconds" in result and "maximum_gap_seconds" not in result:
        result["maximum_gap_seconds"] = result["max_sensor_gap_seconds"]
    for column in ("stable_heating_start", "cycle_duration"):
        if column not in result:
            result[column] = pd.NaT if column == "stable_heating_start" else 0.0
    return result


def _update_cycle_summary(
    summary: pd.DataFrame, baseline_cycles: pd.DataFrame, processed: pd.DataFrame
) -> pd.DataFrame:
    result = summary.copy()
    baseline = baseline_cycles.set_index("cycle_id")
    for column in ("clean_start", "clean_end"):
        if column in baseline:
            result[column] = result["cycle_id"].map(baseline[column])
    result["baseline_start"] = result["clean_start"]
    result["baseline_end"] = result["clean_end"]
    rates = _cycle_missing_rate(processed)
    result["missing_rate"] = result["cycle_id"].map(rates)
    result["is_processable"] = result["cycle_status"].eq("valid") & result["clean_start"].notna()
    return result


def _cycle_missing_rate(frame: pd.DataFrame) -> pd.Series:
    numeric = frame.select_dtypes(include=["number"])
    if numeric.empty or "cycle_id" not in frame:
        return pd.Series(dtype=float)
    rates = 1.0 - numeric.notna().groupby(frame["cycle_id"]).mean().mean(axis=1)
    return rates


def _write_state(config: AppConfig, processed: pd.DataFrame, summary: pd.DataFrame) -> None:
    config.paths.state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "process",
        "date": config.date,
        "processed_rows": len(processed),
        "cycle_rows": len(summary),
        "created_at": time.time(),
    }
    (config.paths.state_dir / "process.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
