"""Standalone validator for Cycle Dataset v3."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .dataset_registry import canonical_registry_hash, is_image_column
from .dataset_v3 import V3_DATASET_FIELDS, V3_DATASET_ID, V3_DATASET_SCHEMA_VERSION
from .io import sha256_file

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_IGNORED_NAMES = {".DS_Store"}
_ASSESSMENT_STATUSES = {"valid", "partial", "incomplete", "invalid"}
_NATURAL_CYCLE_RE = re.compile(r"(?:^|_)cycle_(\d+)$")


def validate_v3_dataset(
    dataset_dir: Path,
    *,
    require_assigned: bool = False,
    verify_image_hashes: bool = True,
    verify_asset_hashes: bool = True,
    selected_assets: set[str] | None = None,
) -> None:
    """Validate Dataset files, metadata closure, content hashes and image folders."""
    root = dataset_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")
    manifest = _read_json(root / "dataset_manifest.json")
    if manifest.get("dataset_schema_version") != V3_DATASET_SCHEMA_VERSION:
        raise ValueError("Dataset is not schema version 3")
    if manifest.get("dataset_id") != V3_DATASET_ID:
        raise ValueError("Dataset manifest has invalid dataset_id")
    for directory in (root / "cycles", root / "images"):
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset is missing {directory.name}/")
    registry = _read_json(root / "channel_registry.json")
    _validate_registry_hash(root, manifest, registry)
    cycle_index = _read_parquet(root / "cycle_index.parquet")
    image_metadata = _read_parquet(root / "image_metadata.parquet")
    _validate_index_hash(
        root,
        manifest,
        "cycle_index",
        "cycle_index.parquet",
        row_count=len(cycle_index),
    )
    _validate_index_hash(
        root,
        manifest,
        "image_metadata",
        "image_metadata.parquet",
        row_count=len(image_metadata),
    )
    _validate_cycle_index(cycle_index)
    _validate_cycle_assessments(manifest, cycle_index)
    _validate_cycle_assets(
        root,
        cycle_index,
        manifest,
        registry,
        verify_asset_hashes=verify_asset_hashes,
        selected_assets=selected_assets,
    )
    _validate_image_metadata(image_metadata, cycle_index)
    _validate_image_files(
        root,
        image_metadata,
        require_assigned=require_assigned,
        verify_hashes=verify_image_hashes,
        selected_assets=selected_assets,
        expected_cycle_names=set(cycle_index["cycle_name"].astype(str)),
    )
    _validate_counts(manifest, cycle_index, image_metadata)


def _validate_cycle_index(frame: pd.DataFrame) -> None:
    required = {
        "dataset_cycle_index",
        "cycle_name",
        "cycle_uid",
        "experiment_id",
        "experiment_date",
        "cycle_id",
        "data_path",
        "csv_path",
        "publication_path",
        "rgb_coverage_path",
        "processed_row_count",
        "image_count",
        "published",
    }
    _require_columns(frame, required, "cycle_index")
    if frame.empty:
        return
    if frame["cycle_uid"].duplicated().any() or frame["cycle_name"].duplicated().any():
        raise ValueError("cycle_index contains duplicate cycle identity")
    indices = pd.to_numeric(frame["dataset_cycle_index"], errors="raise").astype(int).tolist()
    if indices != list(range(1, len(indices) + 1)):
        raise ValueError("published cycle indices must be consecutive")
    ordered = frame.assign(
        _date=pd.to_datetime(frame["experiment_date"], errors="raise"),
        _segment_start=pd.to_datetime(
            frame.get("segment_start", frame["experiment_date"]), errors="coerce"
        ),
        _cycle_number=frame["cycle_id"].astype(str).map(_natural_cycle_number),
        _row_order=range(len(frame)),
    ).sort_values(
        ["_date", "experiment_id", "_segment_start", "_row_order", "_cycle_number", "cycle_id"],
        kind="stable",
        na_position="last",
    )
    if ordered["cycle_uid"].astype(str).tolist() != frame["cycle_uid"].astype(str).tolist():
        raise ValueError("cycle_index is not in canonical chronological order")
    for row in frame.to_dict(orient="records"):
        cycle_name = str(row["cycle_name"])
        expected_index = int(row["dataset_cycle_index"])
        if cycle_name != f"frost_cycle_{expected_index:06d}":
            raise ValueError(f"invalid cycle_name: {cycle_name}")
        if str(row["cycle_uid"]) != f"{row['experiment_id']}::{row['cycle_id']}":
            raise ValueError(f"invalid cycle_uid: {row['cycle_uid']}")
        if not bool(row["published"]):
            raise ValueError("v3 does not publish unpublished or empty cycles")


def _natural_cycle_number(value: str) -> int:
    match = _NATURAL_CYCLE_RE.search(value)
    return int(match.group(1)) if match else 2**31 - 1


def _validate_cycle_assessments(  # noqa: C901
    manifest: Mapping[str, Any], cycle_index: pd.DataFrame
) -> None:
    records = manifest.get("cycles")
    if not isinstance(records, list):
        raise ValueError("manifest cycles must be a list")
    by_name: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or "cycle_name" not in record:
            raise ValueError("manifest contains an invalid cycle record")
        cycle_name = str(record["cycle_name"])
        if cycle_name in by_name:
            raise ValueError(f"manifest contains duplicate cycle record: {cycle_name}")
        by_name[cycle_name] = record
        assessment = record.get("assessment")
        if not isinstance(assessment, Mapping):
            raise ValueError(f"cycle assessment is missing: {cycle_name}")
        if set(assessment) != {"status", "reasons", "note", "updated_at"}:
            raise ValueError(f"cycle assessment fields are invalid: {cycle_name}")
        if str(assessment.get("status")) not in _ASSESSMENT_STATUSES:
            raise ValueError(f"cycle assessment has invalid status: {cycle_name}")
        reasons = assessment.get("reasons")
        if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
            raise ValueError(f"cycle assessment reasons are invalid: {cycle_name}")
        note = assessment.get("note")
        if note is not None and not isinstance(note, str):
            raise ValueError(f"cycle assessment note is invalid: {cycle_name}")
        updated_at = assessment.get("updated_at")
        if not isinstance(updated_at, str) or not updated_at:
            raise ValueError(f"cycle assessment timestamp is invalid: {cycle_name}")
    expected = set(cycle_index["cycle_name"].astype(str))
    if set(by_name) != expected:
        raise ValueError("manifest cycle assessments do not cover cycle_index")


def _validate_cycle_assets(  # noqa: C901
    root: Path,
    cycle_index: pd.DataFrame,
    manifest: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    verify_asset_hashes: bool,
    selected_assets: set[str] | None,
) -> None:
    registry_fields = registry.get("fields")
    if not isinstance(registry_fields, list):
        raise ValueError("channel registry fields are missing")
    expected_schema = [
        str(field["name"])
        for field in registry_fields
        if isinstance(field, Mapping) and "name" in field
    ]
    if len(expected_schema) != len(registry_fields):
        raise ValueError("channel registry fields are invalid")
    expected_schema.extend(V3_DATASET_FIELDS)
    cycle_records = {
        str(record["cycle_name"]): record
        for record in manifest.get("cycles", [])
        if isinstance(record, Mapping) and "cycle_name" in record
    }
    expected_files: set[str] = set()
    for row in cycle_index.to_dict(orient="records"):
        cycle_name = str(row["cycle_name"])
        record = cycle_records.get(cycle_name)
        if record is None:
            raise ValueError(f"manifest is missing cycle record: {cycle_name}")
        for field in (
            "cycle_name",
            "cycle_uid",
            "experiment_id",
            "experiment_date",
            "cycle_id",
        ):
            if str(record.get(field)) != str(row[field]):
                raise ValueError(f"manifest cycle identity disagrees: {cycle_name}/{field}")
        try:
            record_row_count = int(record["row_count"])
            record_image_count = int(record["image_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"manifest cycle counts are invalid: {cycle_name}") from error
        if record_row_count != int(row["processed_row_count"]):
            raise ValueError(f"manifest cycle row count disagrees: {cycle_name}")
        if record_image_count != int(row["image_count"]):
            raise ValueError(f"manifest cycle image count disagrees: {cycle_name}")
        for column in ("data_path", "csv_path", "publication_path", "rgb_coverage_path"):
            relative = _safe_relative(str(row[column]))
            record_relative = record.get(column)
            if not isinstance(record_relative, str) or _safe_relative(record_relative) != relative:
                raise ValueError(f"manifest cycle asset path disagrees: {cycle_name}/{column}")
            expected_files.add(relative)
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"cycle asset is missing: {path}")
            if verify_asset_hashes and (
                selected_assets is None or relative in selected_assets
            ):
                _validate_asset_hash(record, column, path)
        parquet = root / _safe_relative(str(row["data_path"]))
        csv = root / _safe_relative(str(row["csv_path"]))
        frame = pd.read_parquet(parquet)
        csv_frame = pd.read_csv(csv)
        if len(frame) != int(row["processed_row_count"]):
            raise ValueError(f"cycle row count disagrees: {cycle_name}")
        if frame.columns.tolist() != csv_frame.columns.tolist():
            raise ValueError(f"cycle Parquet/CSV columns disagree: {cycle_name}")
        if frame.columns.tolist() != expected_schema:
            raise ValueError(
                "cycle registry schema mismatch: "
                f"{cycle_name}: expected {expected_schema}, got {frame.columns.tolist()}"
            )
        if frame.empty:
            raise ValueError(f"cycle file has no Processed rows: {cycle_name}")
        schema = pq.read_schema(parquet)
        for field_spec, field in zip(
            registry_fields, list(schema)[: len(registry_fields)], strict=True
        ):
            expected_type = str(field_spec["logical_type"])
            actual_type = str(field.type)
            if actual_type != expected_type:
                raise ValueError(
                    "cycle registry dtype mismatch: "
                    f"{cycle_name}/{field.name}: expected {expected_type}, got {actual_type}"
                )
        image_columns = [name for name in frame.columns if is_image_column(name)]
        if image_columns:
            raise ValueError(f"cycle contains image triple columns: {image_columns}")
        if list(frame.columns[-5:]) != list(V3_DATASET_FIELDS):
            raise ValueError(f"cycle Dataset fields are not the final five columns: {cycle_name}")
        for field in V3_DATASET_FIELDS:
            if frame[field].nunique(dropna=False) != 1:
                raise ValueError(f"cycle Dataset field is not constant: {cycle_name}/{field}")
        expected_identity: dict[str, object] = {
            "dataset_id": V3_DATASET_ID,
            "dataset_schema_version": V3_DATASET_SCHEMA_VERSION,
            "dataset_cycle_index": int(cast(Any, row["dataset_cycle_index"])),
            "cycle_name": cycle_name,
            "cycle_uid": str(row["cycle_uid"]),
        }
        for field, expected in expected_identity.items():
            observed = frame[field].iloc[0]
            if field == "dataset_cycle_index":
                try:
                    matches = int(cast(Any, observed)) == int(cast(Any, expected))
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = str(observed) == str(expected)
            if not matches:
                raise ValueError(f"cycle file identity disagrees: {cycle_name}/{field}")
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        if (
            timestamps.isna().any()
            or not timestamps.is_unique
            or not timestamps.is_monotonic_increasing
        ):
            raise ValueError(f"cycle timestamps must be unique and increasing: {cycle_name}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in (root / "cycles").rglob("*")
        if path.is_file() and path.name not in _IGNORED_NAMES and not path.name.startswith(".tmp-")
    }
    if actual_files != expected_files:
        raise ValueError(f"cycle orphan files: {sorted(actual_files - expected_files)}")


def _validate_image_metadata(  # noqa: C901
    frame: pd.DataFrame, cycle_index: pd.DataFrame
) -> None:
    required = {
        "image_id",
        "cycle_uid",
        "cycle_name",
        "frame_index",
        "source_camera_id",
        "initial_camera_slot",
        "image_time",
        "matched_timestamp",
        "offset_seconds",
        "source_relative_path",
        "file_size_bytes",
        "sha256",
    }
    _require_columns(frame, required, "image_metadata")
    if frame["image_id"].duplicated().any():
        raise ValueError("image_metadata contains duplicate image_id")
    from .dataset_v3 import make_image_id

    cycle_names = dict(
        zip(
            cycle_index["cycle_uid"].astype(str),
            cycle_index["cycle_name"].astype(str),
            strict=True,
        )
    )
    metadata_counts = (
        frame["cycle_name"].astype(str).value_counts().to_dict()
        if not frame.empty
        else {}
    )
    for index_row in cycle_index.to_dict(orient="records"):
        cycle_name = str(index_row["cycle_name"])
        if int(metadata_counts.get(cycle_name, 0)) != int(index_row["image_count"]):
            raise ValueError(f"per-cycle image metadata count disagrees: {cycle_name}")
    for row in frame.to_dict(orient="records"):
        value = str(row["source_relative_path"])
        _safe_source_relative(value)
        cycle_uid = str(row["cycle_uid"])
        cycle_name = str(row["cycle_name"])
        if cycle_names.get(cycle_uid) != cycle_name:
            raise ValueError(f"image metadata cycle identity mismatch: {value}")
        uid_parts = cycle_uid.split("::")
        if len(uid_parts) != 2 or not all(uid_parts):
            raise ValueError(f"invalid cycle_uid in image metadata: {cycle_uid}")
        if not cycle_name.startswith("frost_cycle_"):
            raise ValueError(f"invalid cycle_name in image metadata: {cycle_name}")
        if pd.isna(row["image_time"]) or pd.isna(row["matched_timestamp"]):
            raise ValueError(f"image metadata timestamps are missing: {value}")
        if pd.isna(row["offset_seconds"]):
            raise ValueError(f"image metadata offset is missing: {value}")
        if str(row["image_id"]) != make_image_id(
            str(row["cycle_uid"]),
            str(row["source_camera_id"]),
            value,
        ):
            raise ValueError(f"image_id does not match immutable source identity: {value}")
        if value.split("/", 1)[0] != str(row["source_camera_id"]):
            raise ValueError(f"source camera identity disagrees with source path: {value}")
    if not frame.empty:
        for _, group in frame.groupby(["cycle_uid", "source_camera_id"], sort=False):
            ordered = group.assign(
                _image_time=pd.to_datetime(group["image_time"], errors="coerce")
            ).sort_values(
                ["_image_time", "source_relative_path", "image_id"],
                kind="stable",
            )
            frame_indices = pd.to_numeric(group["frame_index"], errors="coerce")
            if frame_indices.isna().any() or not frame_indices.is_unique:
                raise ValueError("image_metadata frame_index is invalid")
            if sorted(frame_indices.astype(int).tolist()) != list(range(1, len(group) + 1)):
                raise ValueError("image_metadata frame_index is not consecutive")
            if ordered["image_id"].tolist() != group.sort_values(
                "frame_index", kind="stable"
            )["image_id"].tolist():
                raise ValueError("image_metadata image order is not stable")


def _validate_image_files(  # noqa: C901
    root: Path,
    metadata: pd.DataFrame,
    *,
    require_assigned: bool = False,
    verify_hashes: bool = True,
    selected_assets: set[str] | None = None,
    expected_cycle_names: set[str] | None = None,
) -> None:
    scanned: dict[str, Path] = {}
    expected_cycles = (
        set(metadata["cycle_name"].astype(str))
        if expected_cycle_names is None
        else expected_cycle_names
    )
    for cycle_root in (root / "images").iterdir():
        if not cycle_root.is_dir():
            if cycle_root.name not in _IGNORED_NAMES and not cycle_root.name.startswith(".tmp-"):
                raise ValueError(f"image orphan path: {cycle_root}")
            continue
        if cycle_root.name not in expected_cycles and any(cycle_root.iterdir()):
            raise ValueError(f"image orphan path: {cycle_root}")
        for role_root in cycle_root.iterdir():
            if not role_root.is_dir():
                if role_root.name not in _IGNORED_NAMES and not role_root.name.startswith(".tmp-"):
                    raise ValueError(f"image orphan path: {role_root}")
                continue
            if require_assigned and role_root.name.startswith("unassigned_"):
                raise ValueError(f"unassigned camera role remains: {role_root}")
            for image_path in role_root.iterdir():
                if image_path.name in _IGNORED_NAMES or image_path.name.startswith(".tmp-"):
                    continue
                if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_SUFFIXES:
                    raise ValueError(f"invalid Dataset image file: {image_path}")
                if image_path.stem in scanned:
                    raise ValueError(f"duplicate Dataset image stem: {image_path.stem}")
                scanned[image_path.stem] = image_path
    metadata_by_id = metadata.set_index("image_id") if not metadata.empty else metadata
    if set(scanned) != set(metadata["image_id"].astype(str)):
        raise ValueError("Dataset image files and image_metadata are not a closed set")
    for image_id, image_path in scanned.items():
        row = metadata_by_id.loc[image_id]
        relative = image_path.relative_to(root).as_posix()
        if (
            verify_hashes
            and (selected_assets is None or relative in selected_assets)
            and sha256_file(image_path) != str(row["sha256"])
        ):
            raise ValueError(f"image SHA mismatch: {image_path}")
        size_value = row["file_size_bytes"]
        if isinstance(size_value, pd.Series):
            raise ValueError(f"image metadata is not unique: {image_id}")
        if image_path.stat().st_size != int(cast(Any, size_value)):
            raise ValueError(f"image size mismatch: {image_path}")
        if image_path.parent.parent.name != str(row["cycle_name"]):
            raise ValueError(f"image cycle path mismatch: {image_path}")


def _validate_counts(
    manifest: Mapping[str, Any], cycle_index: pd.DataFrame, image_metadata: pd.DataFrame
) -> None:
    if int(manifest.get("summary_cycle_count", -1)) != len(cycle_index):
        raise ValueError("manifest summary_cycle_count disagrees")
    if int(manifest.get("image_count", -1)) != len(image_metadata):
        raise ValueError("manifest image_count disagrees")
    image_total = int(cycle_index["image_count"].sum()) if not cycle_index.empty else 0
    if image_total != len(image_metadata):
        raise ValueError("cycle image_count total disagrees")


def _validate_registry_hash(
    root: Path, manifest: Mapping[str, Any], registry: Mapping[str, Any]
) -> None:
    expected = manifest.get("channel_registry", {})
    if not isinstance(expected, Mapping):
        raise ValueError("manifest channel_registry is invalid")
    if str(expected.get("path")) != "channel_registry.json":
        raise ValueError("manifest channel_registry path disagrees")
    canonical = canonical_registry_hash(registry)
    if str(registry.get("canonical_hash")) != canonical:
        raise ValueError("channel_registry stored canonical hash mismatch")
    if str(expected.get("sha256")) != sha256_file(root / "channel_registry.json"):
        raise ValueError("channel_registry SHA mismatch")
    if str(expected.get("canonical_hash")) != canonical:
        raise ValueError("channel_registry canonical hash mismatch")


def _validate_index_hash(
    root: Path,
    manifest: Mapping[str, Any],
    key: str,
    filename: str,
    *,
    row_count: int,
) -> None:
    entry = manifest.get(key)
    if not isinstance(entry, Mapping):
        raise ValueError(f"manifest {key} metadata is missing")
    if str(entry.get("path")) != filename:
        raise ValueError(f"manifest {key} path disagrees")
    try:
        recorded_row_count = int(entry.get("row_count", -1))
    except (TypeError, ValueError) as error:
        raise ValueError(f"manifest {key} row count is invalid") from error
    if recorded_row_count != row_count:
        raise ValueError(f"manifest {key} row count disagrees")
    path = root / filename
    if str(entry.get("sha256")) != sha256_file(path):
        raise ValueError(f"{key} SHA mismatch")


def _validate_asset_hash(record: Mapping[str, Any], column: str, path: Path) -> None:
    hashes = record.get("asset_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError(f"cycle asset hashes are missing: {path}")
    expected = hashes.get(_asset_key(column))
    if not isinstance(expected, str) or not expected:
        raise ValueError(f"cycle asset hash is missing: {path}")
    if expected != sha256_file(path):
        raise ValueError(f"cycle asset SHA mismatch: {path}")


def _asset_key(column: str) -> str:
    return {
        "data_path": "parquet",
        "csv_path": "csv",
        "publication_path": "publication",
        "rgb_coverage_path": "rgb_coverage",
    }[column]


def _safe_relative(value: str) -> str:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"Dataset path must be relative POSIX: {value}")
    return value


def _safe_source_relative(value: str) -> None:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"source_relative_path is unsafe: {value}")


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(str(column) for column in frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset is missing {path.name}")
    return pd.read_parquet(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset is missing {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload
