"""Checks that a Dataset is usable for scientific analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dataset_loader import DatasetLoader


def validate_dataset(dataset_dir: Path) -> None:
    loader = DatasetLoader(dataset_dir)
    columns = loader.registry.get("columns")
    if not isinstance(columns, list):
        raise ValueError("channel_registry columns are missing")
    expected = ["cycle_name", "cycle_uid", *map(str, columns)]

    metadata = loader.load_image_metadata()
    keys = ["cycle_name", "source_camera_id", "file_name"]
    missing = [name for name in keys if name not in metadata]
    if missing:
        raise ValueError(f"image metadata is missing connection keys: {missing}")
    if metadata.duplicated(keys).any():
        raise ValueError("image metadata contains duplicate connection keys")

    for cycle_name in loader.list_cycles()["cycle_name"].astype(str):
        processed = loader.load_cycle(cycle_name)
        original = loader.load_cycle_original(cycle_name)
        if processed.empty or original.empty:
            raise ValueError(f"{cycle_name}: cycle data must be non-empty")
        if processed.columns.tolist() != expected:
            raise ValueError(f"{cycle_name}: processed schema differs from registry")
        _check_time(processed, cycle_name, unique=True)
        _check_time(original, cycle_name, unique=False)
        for name in ("cycle_name", "cycle_uid", "experiment_id", "cycle_id"):
            if name in processed and processed[name].nunique(dropna=False) != 1:
                raise ValueError(f"{cycle_name}: inconsistent {name}")


def _check_time(frame: pd.DataFrame, cycle_name: str, *, unique: bool) -> None:
    if "timestamp" not in frame:
        raise ValueError(f"{cycle_name}: timestamp is missing")
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"{cycle_name}: invalid timestamps")
    if unique and not timestamps.is_unique:
        raise ValueError(f"{cycle_name}: duplicate processed timestamps")
