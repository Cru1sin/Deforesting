"""Atomic Dataset writes and one directory-level staging transaction."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


def write_atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_atomic_json(payload: Any, path: Path) -> None:
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


def write_atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def clone_with_hardlinks(dataset_dir: Path) -> tuple[Path, Path]:
    """Clone a Dataset tree without copying unchanged file contents."""
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    staging_root = dataset_dir.parent / f".{dataset_dir.name}.staging-{uuid4().hex}"
    staging_dataset = staging_root / dataset_dir.name
    try:
        staging_root.mkdir(parents=True, exist_ok=False)
        shutil.copytree(
            dataset_dir,
            staging_dataset,
            copy_function=os.link,
            dirs_exist_ok=False,
        )
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return staging_root, staging_dataset


def create_empty_final_staging(dataset_dir: Path) -> tuple[Path, Path]:
    """Create an empty sibling Dataset tree for add/rebuild."""
    dataset_dir = dataset_dir.resolve()
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = dataset_dir.parent / f".{dataset_dir.name}.staging-{uuid4().hex}"
    staging_dataset = staging_root / dataset_dir.name
    try:
        (staging_dataset / "cycles").mkdir(parents=True, exist_ok=False)
        (staging_dataset / "cycles_original").mkdir(parents=True, exist_ok=False)
        (staging_dataset / "images").mkdir(parents=True, exist_ok=False)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return staging_root, staging_dataset


def publish_with_rollback(
    staging_root: Path,
    staging_dataset: Path,
    dataset_dir: Path,
) -> None:
    """Swap a complete sibling staging tree into place atomically."""
    dataset_dir = dataset_dir.resolve()
    rollback: Path | None = None
    try:
        if dataset_dir.exists():
            rollback = dataset_dir.parent / f".{dataset_dir.name}.rollback-{uuid4().hex}"
            dataset_dir.rename(rollback)
        staging_dataset.rename(dataset_dir)
    except Exception:
        if dataset_dir.exists() and dataset_dir != staging_dataset:
            shutil.rmtree(dataset_dir, ignore_errors=True)
        if rollback is not None and rollback.exists() and not dataset_dir.exists():
            rollback.rename(dataset_dir)
        raise
    else:
        if rollback is not None:
            shutil.rmtree(rollback, ignore_errors=True)
        shutil.rmtree(staging_root, ignore_errors=True)


def mutate_dataset(
    dataset_dir: Path,
    operation: Any,
    *,
    validate: Any,
    rebuild: bool = False,
) -> Path:
    """Run every Dataset write through staging, validation, and directory swap."""
    dataset_dir = dataset_dir.resolve()
    if rebuild or not dataset_dir.exists():
        staging_root, staging_dataset = create_empty_final_staging(dataset_dir)
    else:
        staging_root, staging_dataset = clone_with_hardlinks(dataset_dir)
    published = False
    try:
        operation(staging_dataset)
        validate(staging_dataset)
        publish_with_rollback(staging_root, staging_dataset, dataset_dir)
        published = True
        return dataset_dir
    finally:
        if not published:
            shutil.rmtree(staging_root, ignore_errors=True)
