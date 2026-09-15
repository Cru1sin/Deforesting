"""Pure metadata helpers for the final self-contained Dataset format."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset_paths import write_json

DATASET_SCHEMA_VERSION = 3
DATASET_ID = "frost_cycle_dataset"
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


def following_cycle_names(catalog: pd.DataFrame) -> dict[str, str]:
    """Map each cycle to the immediately following cycle in the same experiment."""
    ordered = catalog.sort_values(["experiment_id", "start_time"], kind="stable")
    names = ordered["cycle_name"].astype(str).tolist()
    experiments = ordered.set_index("cycle_name")["experiment_id"]
    return {
        current: following
        for current, following in zip(names, names[1:], strict=False)
        if experiments.loc[current] == experiments.loc[following]
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
        "pareto_knee_status": "invalid",
        "rgb_knee_coverage_status": "invalid",
        "pareto_extrapolated_knee_status": "invalid",
        "rgb_extrapolated_knee_coverage_status": "invalid",
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


def complete_peak_screen(rows, epsilon=.01, minimum_seconds=60, maximum_gap_seconds=30):
    """Offline curve eligibility, independent of RGB and every model's predictions."""
    records = []
    for name, curve in rows.groupby("cycle_name", sort=True):
        curve = curve.sort_values("candidate_defrost_time")
        time = curve.candidate_defrost_time
        supported = curve.cycle_cop_eligible & np.isfinite(curve.cycle_cop)
        peak = curve.loc[supported, "cycle_cop"].max()
        record = dict(cycle_name=name, experiment_id=curve.experiment_id.iloc[0],
                      selected=False, reason="no_supported_reference", reference_time=pd.NaT,
                      before_seconds=0., after_seconds=0.)
        if pd.notna(peak) and peak > 0:
            boundary = time.loc[supported & curve.cycle_cop.eq(peak)].min()
            below = supported & curve.cycle_cop.le((1 - epsilon) * peak)
            for side, mask in (("before", time.lt(boundary)), ("after", time.gt(boundary))):
                valid = below & mask
                # Unsupported samples and time gaps break evidence; never bridge them.
                segment = ((~valid) | time.diff().dt.total_seconds().gt(maximum_gap_seconds)).cumsum()
                spans = time.loc[valid].groupby(segment.loc[valid]).agg(["min", "max"])
                record[side + "_seconds"] = float((spans["max"] - spans["min"]).dt.total_seconds().max()) if len(spans) else 0.
            missing = [side for side in ("before", "after") if record[side + "_seconds"] < minimum_seconds]
            record.update(reference_time=boundary, selected=not missing,
                          reason="complete_peak" if not missing else "insufficient_" + "_and_".join(missing))
        records.append(record)
    return pd.DataFrame(records)


def update_effective_cop_quality(dataset_dir, rows, *, cycle_names=None, source=None):
    """Publish COP shape quality after calculation, retaining prior human/pipeline status."""
    catalog = read_catalog(dataset_dir)
    quality = complete_peak_screen(rows).set_index("cycle_name")
    scope = set(cycle_names) if cycle_names is not None else set(quality.index)
    for record in catalog["cycles"]:
        name = record["cycle_name"]
        if name not in scope:
            continue
        record.setdefault("pre_cop_status", {
            "status": record["status"], "status_reason": record.get("status_reason")})
        row = quality.loc[name] if name in quality.index else None
        complete = bool(row.selected) if row is not None else False
        reason = str(row.reason) if row is not None else "no_supported_reference"
        record["cop_peak_quality"] = dict(
            complete=complete, reason=reason, epsilon=.01, minimum_seconds=60,
            maximum_gap_seconds=30, source=str(source) if source is not None else None,
            before_seconds=float(row.before_seconds) if row is not None else 0.,
            after_seconds=float(row.after_seconds) if row is not None else 0.)
    write_catalog(dataset_dir, catalog)
    return quality.reset_index()


def update_rgb_validity(dataset_dir, rows, *, source=None):
    """Image presence is Dataset quality; model-specific view coverage is separate."""
    catalog = read_catalog(dataset_dir)
    metadata = Path(dataset_dir) / "image_metadata.parquet"
    counts = (pd.read_parquet(metadata).groupby("cycle_name").size()
              if metadata.exists() else pd.Series(dtype=int))
    available = rows.groupby("cycle_name").rgb_available.any()
    evidence = {}
    for record in catalog["cycles"]:
        name = record["cycle_name"]
        present = bool(counts.get(name, 0) > 0 or available.get(name, False))
        evidence[name] = dict(rgb_valid=present,
            rgb_valid_reason="images_present" if present else "no_image_records")
        record.pop("rgb_positive_slots", None)
    results = []
    for record in catalog["cycles"]:
        quality = evidence.get(record["cycle_name"], dict(
            rgb_valid=False, rgb_valid_reason="not_assessed"))
        record.update(quality)
        record["rgb_valid_source"] = str(source) if source is not None else None
        results.append(dict(cycle_name=record["cycle_name"], status=record["status"], **quality))
    write_catalog(dataset_dir, catalog)
    return pd.DataFrame(results)
