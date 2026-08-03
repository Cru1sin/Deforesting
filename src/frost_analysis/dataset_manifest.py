"""Creation and refresh of the single-assessment Dataset manifest."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import DATASET_V2_SCHEMA_VERSION
from .dataset_coverage import image_summary
from .dataset_images import scan_cycle_images
from .dataset_io import write_atomic_json, write_atomic_parquet
from .io import sha256_file

ASSESSMENT_STATUSES = {"valid", "partial", "incomplete", "invalid"}


def make_assessment(
    status: str,
    reasons: list[str] | None = None,
    note: str | None = None,
    *,
    updated_at: str | None = None,
) -> dict[str, object]:
    """Create the one editable assessment object stored for a cycle."""
    if status not in ASSESSMENT_STATUSES:
        raise ValueError(f"invalid cycle assessment status: {status}")
    return {
        "status": status,
        "reasons": list(reasons or []),
        "note": note,
        "updated_at": updated_at or datetime.now(UTC).isoformat(),
    }


def automatic_assessment(cycle_status: object, reason: object) -> dict[str, object]:
    """Translate the existing scientific cycle status into the editable record."""
    status = str(cycle_status)
    if status not in ASSESSMENT_STATUSES:
        status = "invalid"
    missing_reason = reason is None or (isinstance(reason, float) and pd.isna(reason))
    reasons = [] if missing_reason or not str(reason) else [str(reason)]
    return make_assessment(status, reasons)


def build_manifest(
    *,
    dataset_id: str,
    source_schema: list[dict[str, Any]],
    source_runs: list[dict[str, object]],
    cycles: list[dict[str, object]],
    cycle_index: pd.DataFrame,
    image_metadata: pd.DataFrame,
    created_at: str | None = None,
) -> dict[str, object]:
    """Build a v2 manifest from already materialized Dataset facts."""
    return {
        "dataset_schema_version": DATASET_V2_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "source_processed_schema": source_schema,
        "source_runs": source_runs,
        "cycles": cycles,
        "summary_cycle_count": int(len(cycle_index)),
        "image_count": int(len(image_metadata)),
        "cycle_index": {
            "path": "cycle_index.parquet",
            "row_count": int(len(cycle_index)),
        },
        "image_metadata": {
            "path": "image_metadata.parquet",
            "row_count": int(len(image_metadata)),
        },
    }


def write_manifest(dataset_dir: Path, manifest: dict[str, object]) -> None:
    """Atomically write the manifest JSON."""
    write_atomic_json(manifest, dataset_dir / "dataset_manifest.json")


def refresh_cycle_asset_hashes(dataset_dir: Path, cycle_name: str) -> None:
    """Record the current hashes after regenerating one cycle's assets."""
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = _read_manifest(manifest_path)
    cycles = manifest.get("cycles")
    if not isinstance(cycles, list):
        raise ValueError("dataset manifest is missing cycles")
    record = next(
        (
            item
            for item in cycles
            if isinstance(item, dict) and item.get("cycle_name") == cycle_name
        ),
        None,
    )
    if not isinstance(record, dict):
        raise KeyError(f"unknown cycle: {cycle_name}")
    asset_hashes = record.get("asset_sha256")
    if not isinstance(asset_hashes, dict):
        raise ValueError(f"cycle asset hashes are missing: {cycle_name}")
    asset_paths = {
        "parquet": dataset_dir / "cycles" / f"{cycle_name}.parquet",
        "csv": dataset_dir / "cycles" / f"{cycle_name}.csv",
        "publication": dataset_dir / "cycles" / f"{cycle_name}.png",
        "rgb_coverage": dataset_dir / "cycles" / f"{cycle_name}_rgb_coverage.png",
    }
    for key, path in asset_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"cycle asset does not exist: {path}")
        asset_hashes[key] = sha256_file(path)
    record["asset_sha256"] = asset_hashes
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    write_manifest(dataset_dir, manifest)


def review_cycle(
    dataset_dir: Path,
    cycle_name: str,
    *,
    status: str,
    note: str | None = None,
) -> None:
    """Update only one cycle's assessment, preserving all factual fields."""
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = _read_manifest(manifest_path)
    cycles = manifest.get("cycles")
    if not isinstance(cycles, list):
        raise ValueError("dataset manifest is missing cycles")
    for record in cycles:
        if isinstance(record, dict) and record.get("cycle_name") == cycle_name:
            record["assessment"] = make_assessment(status, note=note)
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            write_manifest(dataset_dir, manifest)
            return
    raise KeyError(f"unknown cycle: {cycle_name}")


def refresh_manifest(dataset_dir: Path) -> Path:
    """Refresh factual image/cycle summaries without touching assessments."""
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = _read_manifest(manifest_path)
    cycles = manifest.get("cycles")
    if not isinstance(cycles, list):
        raise ValueError("dataset manifest is missing cycles")
    image_metadata_path = dataset_dir / "image_metadata.parquet"
    cycle_index_path = dataset_dir / "cycle_index.parquet"
    image_metadata = pd.read_parquet(image_metadata_path)
    cycle_index = pd.read_parquet(cycle_index_path)
    for record in cycles:
        if not isinstance(record, dict):
            raise ValueError("dataset manifest contains an invalid cycle record")
        cycle_name = str(record["cycle_name"])
        frame = pd.read_parquet(dataset_dir / "cycles" / f"{cycle_name}.parquet")
        images = scan_cycle_images(dataset_dir, cycle_name, image_metadata)
        record["row_count"] = int(len(frame))
        record["image_summary"] = image_summary(frame, images)
        record["assessment"] = _preserve_assessment(record.get("assessment"), cycle_name)
        mask = cycle_index["cycle_name"].eq(cycle_name)
        cycle_index.loc[mask, "row_count"] = len(frame)
        cycle_index.loc[mask, "image_count"] = len(images)
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    manifest["image_count"] = int(len(image_metadata))
    write_atomic_parquet(cycle_index, cycle_index_path)
    files = manifest.get("files")
    if isinstance(files, dict):
        cycle_file = files.get("cycle_index")
        if isinstance(cycle_file, dict):
            cycle_file["sha256"] = sha256_file(cycle_index_path)
            cycle_file["row_count"] = len(cycle_index)
    write_manifest(dataset_dir, manifest)
    return dataset_dir


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"dataset manifest is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest must be an object")
    return payload


def _preserve_assessment(value: object, cycle_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"cycle assessment is missing: {cycle_name}")
    required = {"status", "reasons", "note", "updated_at"}
    if set(value) != required or str(value["status"]) not in ASSESSMENT_STATUSES:
        raise ValueError(f"cycle assessment is invalid: {cycle_name}")
    if not isinstance(value["reasons"], list):
        raise ValueError(f"cycle assessment reasons must be a list: {cycle_name}")
    return value
