"""Pure metadata helpers for the final self-contained Dataset contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import DATASET_ID, DATASET_SCHEMA_VERSION
from .dataset_io import write_json

CATALOG_FILENAME = "cycle_catalog.json"
MANIFEST_FILENAME = "dataset_manifest.json"


def cycle_assets(cycle_name: str) -> dict[str, str]:
    return {
        "parquet": f"cycles/{cycle_name}.parquet",
        "csv": f"cycles/{cycle_name}.csv",
        "original_csv": f"cycles_original/{cycle_name}.csv",
        "publication": f"cycles/{cycle_name}.png",
        "rgb_panel": f"cycles/{cycle_name}_rgb_panel.png",
    }


def read_manifest(dataset_dir: Path) -> dict[str, Any]:
    """Read and validate the small Dataset-level manifest."""
    payload = _read_object(dataset_dir / MANIFEST_FILENAME)
    expected = {
        "dataset_schema_version",
        "dataset_id",
        "experiments",
    }
    missing = expected - set(payload)
    if missing:
        raise ValueError(
            "dataset_manifest.json is missing required fields: "
            f"{sorted(missing)}"
        )
    if payload.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("Dataset manifest is not schema version 3")
    if payload.get("dataset_id") != DATASET_ID:
        raise ValueError("Dataset manifest has an invalid dataset_id")
    if not isinstance(payload.get("experiments"), list):
        raise ValueError("Dataset manifest experiments must be a list")
    expected_experiment = {"experiment_id", "experiment_date"}
    for item in payload["experiments"]:
        if not isinstance(item, Mapping) or not expected_experiment <= set(item):
            raise ValueError("Dataset experiment records are missing identity fields")
    return payload


def write_manifest(dataset_dir: Path, manifest: Mapping[str, Any]) -> None:
    """Write only the final Dataset-level manifest fields."""
    write_json(dict(manifest), dataset_dir / MANIFEST_FILENAME)


def image_root(dataset_dir: Path, manifest: Mapping[str, Any] | None = None) -> Path:
    """Resolve the single Dataset image-location entry."""
    payload = (
        read_manifest(dataset_dir)
        if manifest is None and (dataset_dir / MANIFEST_FILENAME).is_file()
        else (manifest or {})
    )
    configured = Path(str(payload.get("images_root", "images"))).expanduser()
    return (
        configured.resolve()
        if configured.is_absolute()
        else (dataset_dir / configured).resolve()
    )


def read_catalog(dataset_dir: Path) -> dict[str, Any]:
    """Read the human-readable cycle catalog."""
    payload = _read_object(dataset_dir / CATALOG_FILENAME)
    if "cycles" not in payload or not isinstance(payload["cycles"], list):
        raise ValueError("cycle_catalog.json must contain a cycles list")
    return payload


def write_catalog(dataset_dir: Path, catalog: Mapping[str, Any]) -> None:
    """Write the cycle catalog directly."""
    if "cycles" not in catalog or not isinstance(catalog["cycles"], list):
        raise ValueError("cycle catalog must contain a cycles list")
    write_json(dict(catalog), dataset_dir / CATALOG_FILENAME)


def experiment_record(
    experiment_id: str,
    experiment_date: str,
) -> dict[str, object]:
    return {
        "experiment_id": str(experiment_id),
        "experiment_date": str(experiment_date)[:10],
    }


def build_cycle_record(
    summary_row: Mapping[str, Any],
    *,
    cycle_name: str,
    cycle_uid: str,
    processed: pd.DataFrame,
    original: pd.DataFrame,
    image_summary: Mapping[str, Any],
    assets: Mapping[str, str],
) -> dict[str, Any]:
    """Build one complete, human-readable cycle record."""
    pipeline_status = _clean(summary_row.get("cycle_status")) or "invalid"
    pipeline_reason = _clean(summary_row.get("cycle_status_reason"))
    processed_timestamps = (
        processed["timestamp"] if "timestamp" in processed else pd.Series(dtype=object)
    )
    original_timestamps = (
        original["timestamp"] if "timestamp" in original else pd.Series(dtype=object)
    )
    timestamps = pd.to_datetime(processed_timestamps, errors="coerce").dropna()
    original_times = pd.to_datetime(original_timestamps, errors="coerce").dropna()
    boundaries = {
        name: _iso(summary_row.get(name))
        for name in (
            "start_time",
            "end_time",
            "heating_start",
            "stable_heating_start",
            "defrost_preparation_start",
            "defrost_start",
            "defrost_end",
            "baseline_start",
            "baseline_end",
        )
    }
    if boundaries["start_time"] is None and not timestamps.empty:
        boundaries["start_time"] = pd.Timestamp(timestamps.min()).isoformat()
    if boundaries["end_time"] is None and not timestamps.empty:
        boundaries["end_time"] = pd.Timestamp(timestamps.max()).isoformat()
    interval = original_times.sort_values().diff().dt.total_seconds().dropna()
    return {
        "cycle_name": cycle_name,
        "cycle_uid": cycle_uid,
        "experiment_id": str(summary_row["experiment_id"]),
        "experiment_date": str(summary_row["experiment_date"])[:10],
        "cycle_id": str(summary_row["cycle_id"]),
        "pipeline_status": pipeline_status,
        "pipeline_status_reason": pipeline_reason,
        "status": pipeline_status,
        "status_reason": pipeline_reason,
        "boundaries": boundaries,
        "data": {
            "processed_row_count": int(len(processed)),
            "original_row_count": int(len(original)),
            "median_original_interval_seconds": (
                float(interval.median()) if not interval.empty else None
            ),
        },
        "image": dict(image_summary),
        "assets": dict(assets),
    }


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset is missing {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _clean(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _iso(value: Any) -> str | None:
    cleaned = _clean(value)
    if cleaned is None:
        return None
    timestamp = pd.to_datetime(cleaned, errors="coerce")
    return None if pd.isna(timestamp) else pd.Timestamp(timestamp).isoformat()
