"""Structural and archive validation for Dataset schema 3."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .io import sha256_file


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset is missing {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _validate_cycle_identity(
    frame: pd.DataFrame,
    record: Mapping[str, Any],
    label: str,
    fields: tuple[str, ...] = (
        "cycle_name",
        "cycle_uid",
        "experiment_id",
        "cycle_id",
    ),
) -> None:
    """Ensure a persisted cycle table still belongs to its Catalog record."""
    for field in fields:
        if field not in frame:
            raise ValueError(f"{label} is missing identity column: {field}")
        values = frame[field].astype("string").dropna().unique()
        expected = str(record.get(field, ""))
        if len(values) != 1 or str(values[0]) != expected:
            raise ValueError(f"{label} {field} identity mismatch")


def _safe_relative(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe Dataset relative path: {value}")
    return path.as_posix()


def _safe_source_relative(value: str) -> str:
    return _safe_relative(value)


def validate_staging_structure(dataset_dir: Path) -> None:  # noqa: C901
    """Validate only the files and keys needed immediately before publish."""
    from .dataset import _cycle_assets, parse_cycle_name
    from .dataset_images import _parse_role_directory
    from .dataset_metadata import read_catalog, read_manifest

    root = dataset_dir.resolve()
    manifest = read_manifest(root)
    catalog = read_catalog(root)
    registry_path = root / "channel_registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError("Dataset is missing channel_registry.json")
    _read_json_object(registry_path)
    for directory in ("cycles", "cycles_original", "images"):
        if not (root / directory).is_dir():
            raise FileNotFoundError(f"Dataset is missing {directory}/")

    experiments = manifest["experiments"]
    identities: list[tuple[str, str]] = []
    experiment_ids: set[str] = set()
    previous_date = ""
    for item in experiments:
        if not isinstance(item, Mapping):
            raise ValueError("manifest experiment record is invalid")
        identity = (str(item.get("experiment_id", "")), str(item.get("experiment_date", "")))
        if not all(identity) or identity in identities:
            raise ValueError("manifest experiment identity is missing or duplicated")
        if identity[1] < previous_date:
            raise ValueError("manifest experiments are not date ordered")
        previous_date = identity[1]
        identities.append(identity)
        experiment_ids.add(identity[0])

    records = catalog["cycles"]
    names: set[str] = set()
    uids: set[str] = set()
    expected_cycles: set[str] = set()
    cycle_indices: list[int] = []
    allowed_statuses = {"valid", "partial", "incomplete", "invalid"}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("cycle catalog record is invalid")
        cycle_name = str(record.get("cycle_name", ""))
        cycle_uid = str(record.get("cycle_uid", ""))
        if not cycle_name or not cycle_uid or cycle_name in names or cycle_uid in uids:
            raise ValueError("cycle catalog contains duplicate or empty identity")
        cycle_index = parse_cycle_name(cycle_name)
        cycle_indices.append(cycle_index)
        experiment_id = str(record.get("experiment_id", ""))
        cycle_id = str(record.get("cycle_id", ""))
        if experiment_id not in experiment_ids:
            raise ValueError(f"cycle references an unknown experiment: {cycle_name}")
        if cycle_uid != f"{experiment_id}::{cycle_id}":
            raise ValueError(f"cycle_uid disagrees with cycle identity: {cycle_name}")
        if str(record.get("pipeline_status", "")) not in allowed_statuses:
            raise ValueError(f"invalid pipeline_status: {cycle_name}")
        if str(record.get("status", "")) not in allowed_statuses:
            raise ValueError(f"invalid Dataset status: {cycle_name}")
        names.add(cycle_name)
        uids.add(cycle_uid)
        expected_cycles.add(cycle_name)
        assets = record.get("assets")
        if not isinstance(assets, Mapping):
            raise ValueError(f"cycle assets are missing: {cycle_name}")
        required_assets = {"parquet", "csv", "original_csv", "publication", "rgb_coverage"}
        if set(assets) != required_assets:
            raise ValueError(f"cycle assets are incomplete: {cycle_name}")
        if dict(assets) != _cycle_assets(cycle_name):
            raise ValueError(f"cycle assets disagree with cycle_name: {cycle_name}")
        for relative in assets.values():
            if not isinstance(relative, str) or _safe_relative(relative) != relative:
                raise ValueError(f"unsafe cycle asset path: {cycle_name}")
            if not (root / relative).is_file():
                raise FileNotFoundError(f"cycle asset is missing: {root / relative}")

    if sorted(cycle_indices) != list(range(1, len(cycle_indices) + 1)):
        raise ValueError("cycle_name values must be continuous from frost_cycle_000001")

    metadata_path = root / "image_metadata.parquet"
    if not metadata_path.is_file():
        raise FileNotFoundError("Dataset is missing image_metadata.parquet")
    metadata = pd.read_parquet(metadata_path)
    required_metadata = {
        "image_id",
        "cycle_uid",
        "cycle_name",
        "source_camera_id",
        "file_name",
        "frame_index",
        "initial_camera_slot",
        "image_time",
        "matched_timestamp",
        "offset_seconds",
        "cycle_stage",
        "source_relative_path",
        "file_size_bytes",
    }
    _require_columns(metadata, required_metadata, "image_metadata")
    if "sha256" in metadata.columns:
        raise ValueError("image_metadata must not contain image sha256")
    if metadata["image_id"].duplicated().any():
        raise ValueError("image_metadata image_id is not unique")
    if metadata.duplicated(["cycle_name", "source_camera_id", "file_name"]).any():
        raise ValueError("image_metadata source/file key is not unique")
    if not set(metadata["cycle_name"].astype(str)) <= expected_cycles:
        raise ValueError("image_metadata references an unknown cycle")
    for value in metadata["source_relative_path"].astype(str):
        _safe_source_relative(value)
    for cycle_root in (root / "images").iterdir():
        if not cycle_root.is_dir():
            continue
        if cycle_root.name not in expected_cycles:
            raise ValueError(f"image directory references an unknown cycle: {cycle_root}")
        source_roles: dict[str, str] = {}
        for role_dir in cycle_root.iterdir():
            if not role_dir.is_dir():
                continue
            source_camera_id, current_role = _parse_role_directory(role_dir.name)
            previous_role = source_roles.get(source_camera_id)
            if previous_role is not None and previous_role != current_role:
                raise ValueError(
                    "source camera is assigned to multiple current roles: "
                    f"{cycle_root.name}/{source_camera_id}"
                )
            source_roles[source_camera_id] = current_role


def validate_dataset(dataset_dir: Path) -> None:  # noqa: C901
    """Perform the explicit, complete non-image archive validation."""
    from .dataset_metadata import read_catalog, read_manifest
    from .dataset_registry import canonical_registry_hash, is_image_column

    root = dataset_dir.resolve()
    validate_staging_structure(root)
    catalog = read_catalog(root)
    manifest = read_manifest(root)
    registry = _read_json_object(root / "channel_registry.json")
    if registry.get("canonical_hash") != canonical_registry_hash(registry):
        raise ValueError("channel_registry canonical hash mismatch")
    fields = registry.get("fields")
    if not isinstance(fields, list):
        raise ValueError("channel_registry fields are missing")
    expected_columns = [
        "dataset_id",
        "dataset_schema_version",
        "dataset_cycle_index",
        "cycle_name",
        "cycle_uid",
        *[str(item["name"]) for item in fields if isinstance(item, Mapping)],
    ]
    for record in catalog["cycles"]:
        if not isinstance(record, Mapping):
            raise ValueError("cycle catalog record is invalid")
        cycle_name = str(record["cycle_name"])
        assets = record["assets"]
        hashes = record.get("asset_sha256")
        if not isinstance(hashes, Mapping):
            raise ValueError(f"cycle asset SHA record is missing: {cycle_name}")
        for key in ("parquet", "csv", "original_csv", "publication", "rgb_coverage"):
            relative = str(assets[key])
            expected_hash = str(hashes.get(key, ""))
            if not expected_hash or sha256_file(root / relative) != expected_hash:
                raise ValueError(f"cycle asset SHA mismatch: {cycle_name}/{key}")
        parquet = root / str(assets["parquet"])
        csv_path = root / str(assets["csv"])
        original_path = root / str(assets["original_csv"])
        frame = pd.read_parquet(parquet)
        csv_frame = pd.read_csv(csv_path)
        original = pd.read_csv(original_path)
        if frame.empty or csv_frame.empty or original.empty:
            raise ValueError(f"cycle assets must be non-empty: {cycle_name}")
        if (
            frame.columns.tolist() != expected_columns
            or csv_frame.columns.tolist() != expected_columns
        ):
            raise ValueError(f"cycle schema disagrees with channel registry: {cycle_name}")
        if any(is_image_column(column) for column in frame.columns):
            raise ValueError(f"cycle contains image columns: {cycle_name}")
        _validate_cycle_identity(frame, record, f"processed {cycle_name}")
        _validate_cycle_identity(
            csv_frame,
            record,
            f"CSV {cycle_name}",
        )
        _validate_cycle_identity(
            original,
            record,
            f"original {cycle_name}",
            fields=("experiment_id", "cycle_id"),
        )
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        original_timestamps = pd.to_datetime(original["timestamp"], errors="coerce")
        if (
            timestamps.isna().any()
            or not timestamps.is_unique
            or not timestamps.is_monotonic_increasing
        ):
            raise ValueError(f"processed timestamps are invalid: {cycle_name}")
        if original_timestamps.isna().any() or not original_timestamps.is_monotonic_increasing:
            raise ValueError(f"original timestamps are invalid: {cycle_name}")
        if any(
            str(column).endswith(
                ("__missing", "__invalid", "__duplicate", "__conflict")
            )
            for column in original.columns
        ):
            raise ValueError(f"original cycle contains source quality columns: {cycle_name}")
        data = record.get("data")
        if not isinstance(data, Mapping):
            raise ValueError(f"cycle data summary is missing: {cycle_name}")
        if int(data.get("processed_row_count", -1)) != len(frame):
            raise ValueError(f"processed row count mismatch: {cycle_name}")
        if int(data.get("original_row_count", -1)) != len(original):
            raise ValueError(f"original row count mismatch: {cycle_name}")
    _ = manifest
