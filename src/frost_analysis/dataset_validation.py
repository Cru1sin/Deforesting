"""Scientific usability checks for the self-contained Dataset."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .dataset import make_cycle_uid
from .dataset_loader import DatasetLoader
from .dataset_registry import is_image_column


def validate_dataset(dataset_dir: Path) -> None:
    """Check the scientific files needed by Dataset consumers."""
    loader = DatasetLoader(dataset_dir)
    fields = loader.registry.get("fields")
    if not isinstance(fields, list):
        raise ValueError("channel_registry fields are missing")
    expected_columns = [
        "dataset_id",
        "dataset_schema_version",
        "dataset_cycle_index",
        "cycle_name",
        "cycle_uid",
        *[
            str(item["name"])
            for item in fields
            if isinstance(item, Mapping) and "name" in item
        ],
    ]

    _validate_image_metadata(loader.load_image_metadata())
    for cycle_name in loader.list_cycles()["cycle_name"].astype(str):
        _validate_cycle(loader, cycle_name, expected_columns)


def _validate_image_metadata(metadata: pd.DataFrame) -> None:
    key_columns = ["cycle_name", "source_camera_id", "file_name"]
    missing = [column for column in key_columns if column not in metadata]
    if missing:
        raise ValueError(f"image metadata is missing connection keys: {missing}")
    if metadata.duplicated(key_columns).any():
        raise ValueError("image metadata contains duplicate source/file keys")


def _validate_cycle(
    loader: DatasetLoader, cycle_name: str, expected_columns: list[str]
) -> None:
    record = loader.get_cycle_record(cycle_name)
    processed = loader.load_cycle(cycle_name)
    original = loader.load_cycle_original(cycle_name)
    if processed.empty or original.empty:
        raise ValueError(f"{cycle_name}: cycle data must be non-empty")
    if processed.columns.tolist() != expected_columns:
        raise ValueError(f"{cycle_name}: processed schema differs from registry")
    if any(is_image_column(column) for column in processed.columns):
        raise ValueError(f"{cycle_name}: processed data contains image columns")

    _validate_cycle_identity(processed, record, f"processed {cycle_name}")
    _validate_cycle_identity(
        original,
        record,
        f"original {cycle_name}",
        fields=("experiment_id", "cycle_id"),
    )
    _validate_timestamps(processed, original, cycle_name)
    _validate_original_columns(original, cycle_name)


def _validate_timestamps(
    processed: pd.DataFrame, original: pd.DataFrame, cycle_name: str
) -> None:
    timestamps = pd.to_datetime(processed["timestamp"], errors="coerce")
    original_timestamps = pd.to_datetime(original["timestamp"], errors="coerce")
    if timestamps.isna().any() or not timestamps.is_unique:
        raise ValueError(f"{cycle_name}: invalid processed timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"{cycle_name}: unordered processed timestamps")
    if original_timestamps.isna().any() or not original_timestamps.is_monotonic_increasing:
        raise ValueError(f"{cycle_name}: invalid original timestamps")


def _validate_original_columns(original: pd.DataFrame, cycle_name: str) -> None:
    if any(
        str(column).endswith(("__missing", "__invalid", "__duplicate", "__conflict"))
        for column in original.columns
    ):
        raise ValueError(f"{cycle_name}: original contains source quality columns")


def _validate_cycle_identity(
    frame: pd.DataFrame,
    record: Mapping[str, object],
    context: str,
    *,
    fields: tuple[str, ...] = (
        "cycle_name",
        "cycle_uid",
        "experiment_id",
        "cycle_id",
    ),
) -> None:
    for field in fields:
        if field not in frame:
            raise ValueError(f"{context}: missing identity column {field}")
    expected_cycle_name = str(record["cycle_name"])
    expected_uid = str(record["cycle_uid"])
    expected_experiment = str(record["experiment_id"])
    expected_cycle_id = str(record["cycle_id"])
    expected: dict[str, str] = {
        "cycle_name": expected_cycle_name,
        "cycle_uid": expected_uid,
        "experiment_id": expected_experiment,
        "cycle_id": expected_cycle_id,
    }
    for field, value in expected.items():
        if field not in fields:
            continue
        observed = frame[field]
        if not observed.eq(value).all():
            raise ValueError(f"{context}: {field} does not match Catalog")
    if expected_uid != make_cycle_uid(expected_experiment, expected_cycle_id):
        raise ValueError(f"{context}: cycle_uid is inconsistent")
