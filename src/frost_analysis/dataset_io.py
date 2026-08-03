"""Dataset-specific staging and atomic metadata writes."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


def write_atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a parquet file through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_atomic_json(payload: Any, path: Path) -> None:
    """Write JSON through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_build_staging(output_dir: Path) -> tuple[Path, Path]:
    """Create a build staging root containing the final dataset basename."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_dir.parent / f".dataset-build-{uuid4().hex}"
    staging_dataset = staging_root / output_dir.name
    staging_dataset.mkdir(parents=True, exist_ok=False)
    (staging_dataset / "cycles").mkdir()
    (staging_dataset / "images").mkdir()
    return staging_root, staging_dataset


def publish_build(staging_root: Path, staging_dataset: Path, output_dir: Path) -> None:
    """Publish a validated build into a path that must not already exist."""
    if output_dir.exists():
        raise FileExistsError(f"dataset output already exists: {output_dir}")
    os.replace(staging_dataset, output_dir)
    shutil.rmtree(staging_root, ignore_errors=True)


def create_append_staging(dataset_dir: Path) -> tuple[Path, Path]:
    """Create sibling append staging with the final dataset basename."""
    staging_root = dataset_dir.parent / f".{dataset_dir.name}.append-{uuid4().hex}"
    staging_dataset = staging_root / dataset_dir.name
    (staging_dataset / "cycles").mkdir(parents=True, exist_ok=False)
    (staging_dataset / "images").mkdir(parents=True, exist_ok=False)
    return staging_root, staging_dataset


def backup_append_metadata(dataset_dir: Path, staging_root: Path) -> Path:
    """Copy the three commit-marker metadata files into append staging."""
    backup_dir = staging_root / "backups"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name in ("cycle_index.parquet", "image_index.parquet", "dataset_manifest.json"):
        source = dataset_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"dataset metadata is missing: {source}")
        shutil.copy2(source, backup_dir / name)
    return backup_dir


def commit_append_files(
    staging_dataset: Path,
    dataset_dir: Path,
    relative_paths: list[str],
    moved_files: list[Path] | None = None,
) -> list[Path]:
    """Move new data files, then atomically replace metadata in commit order."""
    moved = moved_files if moved_files is not None else []
    for relative in relative_paths:
        source = staging_dataset / relative
        target = dataset_dir / relative
        if target.exists():
            raise FileExistsError(f"append target already exists: {target}")
        if not source.is_file():
            raise FileNotFoundError(f"append staging file is missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        moved.append(target)
    for name in ("cycle_index.parquet", "image_index.parquet", "dataset_manifest.json"):
        source = staging_dataset / name
        if not source.is_file():
            raise FileNotFoundError(f"append staging metadata is missing: {source}")
        os.replace(source, dataset_dir / name)
    return moved


def rollback_append(
    dataset_dir: Path,
    backup_dir: Path | None,
    moved_files: list[Path],
) -> None:
    """Restore metadata and remove only files moved by the failed append."""
    for path in moved_files:
        path.unlink(missing_ok=True)
    if backup_dir is None:
        return
    for name in ("cycle_index.parquet", "image_index.parquet", "dataset_manifest.json"):
        backup = backup_dir / name
        if backup.is_file():
            shutil.copy2(backup, dataset_dir / name)


def cleanup_append_staging(staging_root: Path) -> None:
    """Remove an append staging tree after success or rollback."""
    shutil.rmtree(staging_root, ignore_errors=True)


def create_dataset_staging(dataset_dir: Path, *, kind: str) -> tuple[Path, Path]:
    """Create a v2 staging tree with the final dataset basename."""
    if kind not in {"build", "append"}:
        raise ValueError(f"unsupported dataset staging kind: {kind}")
    prefix = ".dataset-build-" if kind == "build" else f".{dataset_dir.name}.add-"
    staging_root = dataset_dir.parent / f"{prefix}{uuid4().hex}"
    staging_dataset = staging_root / dataset_dir.name
    (staging_dataset / "cycles").mkdir(parents=True, exist_ok=False)
    (staging_dataset / "images").mkdir(parents=True, exist_ok=False)
    return staging_root, staging_dataset


def commit_v2_append_files(
    staging_dataset: Path,
    dataset_dir: Path,
    relative_paths: list[str],
    *,
    moved_files: list[Path] | None = None,
) -> list[Path]:
    """Move v2 cycle/image assets, then replace the three metadata files."""
    moved = moved_files if moved_files is not None else []
    for relative in relative_paths:
        source = staging_dataset / relative
        target = dataset_dir / relative
        if target.exists():
            raise FileExistsError(f"append target already exists: {target}")
        if not source.is_file():
            raise FileNotFoundError(f"append staging file is missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        moved.append(target)
    for name in ("cycle_index.parquet", "image_metadata.parquet", "dataset_manifest.json"):
        source = staging_dataset / name
        if not source.is_file():
            raise FileNotFoundError(f"append staging metadata is missing: {source}")
        os.replace(source, dataset_dir / name)
    return moved


def backup_v2_metadata(dataset_dir: Path, staging_root: Path) -> Path:
    """Back up v2 commit-marker metadata before an append."""
    backup_dir = staging_root / "backups"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name in ("cycle_index.parquet", "image_metadata.parquet", "dataset_manifest.json"):
        source = dataset_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"dataset metadata is missing: {source}")
        shutil.copy2(source, backup_dir / name)
    return backup_dir


def rollback_v2_append(
    dataset_dir: Path,
    backup_dir: Path | None,
    moved_files: list[Path],
) -> None:
    """Restore v2 metadata and remove only files moved by this append."""
    for path in moved_files:
        path.unlink(missing_ok=True)
    if backup_dir is None:
        return
    for name in ("cycle_index.parquet", "image_metadata.parquet", "dataset_manifest.json"):
        backup = backup_dir / name
        if backup.is_file():
            shutil.copy2(backup, dataset_dir / name)
