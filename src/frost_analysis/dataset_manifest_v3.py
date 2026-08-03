"""Manifest editing and refresh for Dataset v3."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset_coverage_v3 import coverage_ratio
from .dataset_io import write_atomic_json, write_atomic_parquet
from .dataset_loader import DatasetLoader
from .dataset_v3 import V3_DATASET_SCHEMA_VERSION
from .io import sha256_file

ASSESSMENT_STATUSES = {"valid", "partial", "incomplete", "invalid"}


def review_cycle(
    dataset_dir: Path, cycle_name: str, *, status: str, note: str | None = None
) -> None:
    if status not in ASSESSMENT_STATUSES:
        raise ValueError(f"invalid cycle assessment status: {status}")
    path = dataset_dir / "dataset_manifest.json"
    manifest = _read_json(path)
    for record in manifest.get("cycles", []):
        if isinstance(record, dict) and record.get("cycle_name") == cycle_name:
            previous = record.get("assessment")
            reasons = previous.get("reasons", []) if isinstance(previous, dict) else []
            record["assessment"] = {
                "status": status,
                "reasons": list(reasons) if isinstance(reasons, list) else [],
                "note": note,
                "updated_at": datetime.now(UTC).isoformat(),
            }
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            write_atomic_json(manifest, path)
            return
    raise KeyError(f"unknown cycle: {cycle_name}")


def refresh_manifest(dataset_dir: Path) -> Path:
    """Refresh factual asset and role summaries without changing assessment."""
    loader = DatasetLoader(dataset_dir)
    manifest = loader.manifest
    cycle_index = loader.cycle_index
    role_accumulator: dict[str, dict[str, Any]] = {}
    total_cycles = len(cycle_index)
    records = manifest.get("cycles", [])
    if not isinstance(records, list):
        raise ValueError("dataset manifest cycles must be a list")
    for record in records:
        if not isinstance(record, dict):
            continue
        cycle_name = str(record["cycle_name"])
        frame = loader.load_cycle(cycle_name)
        images = loader.load_cycle_images(cycle_name)
        record["row_count"] = len(frame)
        record["image_count"] = len(images)
        record["image_summary"] = _cycle_image_summary(frame, images)
        for role, group in images.groupby("camera_role", sort=True):
            entry = role_accumulator.setdefault(
                str(role),
                {"image_count": 0, "cycle_count": 0, "ratios": [], "full": 0, "gaps": 0},
            )
            ratio = coverage_ratio(frame, group)
            entry["image_count"] += len(group)
            entry["cycle_count"] += 1
            entry["ratios"].append(ratio)
            entry["full"] += int(ratio == 1.0)
            entry["gaps"] += int(ratio < 1.0)
        mask = cycle_index["cycle_name"].eq(cycle_name)
        cycle_index.loc[mask, "processed_row_count"] = len(frame)
        cycle_index.loc[mask, "image_count"] = len(images)
    by_role: dict[str, dict[str, Any]] = {}
    for role, entry in role_accumulator.items():
        ratios = list(entry["ratios"])
        by_role[role] = {
            "image_count": entry["image_count"],
            "cycle_count": entry["cycle_count"],
            "missing_role_cycle_count": total_cycles - entry["cycle_count"],
            "mean_coverage_ratio": float(sum(ratios) / len(ratios)) if ratios else 0.0,
            "minimum_coverage_ratio": float(min(ratios)) if ratios else 0.0,
            "fully_covered_cycle_count": entry["full"],
            "has_gap_cycle_count": entry["gaps"],
        }
    manifest["image_summary"] = {"by_camera_role": by_role}
    cycle_index_path = dataset_dir / "cycle_index.parquet"
    write_atomic_parquet(cycle_index, cycle_index_path)
    manifest["cycle_index"] = {
        "path": "cycle_index.parquet",
        "row_count": len(cycle_index),
        "sha256": sha256_file(cycle_index_path),
    }
    manifest["summary_cycle_count"] = len(cycle_index)
    manifest["image_count"] = len(loader.load_image_metadata())
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    write_atomic_json(manifest, dataset_dir / "dataset_manifest.json")
    return dataset_dir


def _cycle_image_summary(frame: pd.DataFrame, images: pd.DataFrame) -> dict[str, Any]:
    by_role: dict[str, Any] = {}
    for role, group in images.groupby("camera_role", sort=True):
        by_role[str(role)] = {
            "image_count": len(group),
            "coverage_ratio": coverage_ratio(frame, group),
        }
    return {"camera_role_count": len(by_role), "by_camera_role": by_role}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    if payload.get("dataset_schema_version") != V3_DATASET_SCHEMA_VERSION:
        raise ValueError("Dataset manifest is not schema version 3")
    return payload
