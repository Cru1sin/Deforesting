"""Structural validation for published cycle datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .dataset import (
    CYCLE_NAME_WIDTH,
    DATASET_SCHEMA_VERSION,
    DATASET_V2_SCHEMA_VERSION,
    format_cycle_name,
    logical_schema_compatible,
    make_cycle_uid,
    make_v2_cycle_uid,
    parse_cycle_name,
    validate_dataset_id,
)
from .images import image_columns, image_roles
from .io import sha256_file

_CYCLE_INDEX_COLUMNS = {
    "dataset_cycle_index",
    "cycle_name",
    "cycle_uid",
    "experiment_id",
    "experiment_date",
    "cycle_id",
    "cycle_status",
    "cycle_status_reason",
    "baseline_status",
    "baseline_failure_reason",
    "published",
    "data_path",
    "data_sha256",
    "data_size_bytes",
    "processed_row_count",
    "image_count",
    "recommended_for_analysis",
    "dataset_exclusion_reason",
}
_IMAGE_INDEX_COLUMNS = {
    "image_id",
    "cycle_uid",
    "cycle_name",
    "camera_role",
    "image_time",
    "matched_timestamp",
    "offset_seconds",
    "cycle_stage",
    "image_path",
    "source_relative_path",
    "sha256",
    "file_size_bytes",
}
_IGNORED_NAMES = {".DS_Store"}


def validate_dataset(dataset_dir: Path) -> None:
    """Validate a complete published dataset, including all file hashes."""
    manifest_path = dataset_dir / "dataset_manifest.json"
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
        if (
            isinstance(payload, dict)
            and payload.get("dataset_schema_version") == DATASET_V2_SCHEMA_VERSION
        ):
            _validate_v2_dataset(dataset_dir, payload)
            return
    manifest, cycle_index, image_index = _validate_dataset_structure(dataset_dir)
    _validate_cycle_files(dataset_dir, manifest, cycle_index, image_index)
    _validate_image_files(dataset_dir, image_index)
    _validate_orphans(dataset_dir, cycle_index, image_index)


def _validate_append_candidate(
    staging_dataset: Path,
    manifest: dict[str, Any],
    cycle_index: pd.DataFrame,
    image_index: pd.DataFrame,
    new_cycle_index: pd.DataFrame,
    new_records: list[dict[str, object]],
) -> None:
    """Validate merged metadata and only the files present in append staging."""
    _validate_dataset_structure(staging_dataset)
    _validate_new_files(
        staging_dataset,
        manifest,
        new_cycle_index,
        image_index,
        new_records=new_records,
    )


def _validate_new_files(
    dataset_dir: Path,
    manifest: dict[str, Any],
    new_cycle_index: pd.DataFrame,
    image_index: pd.DataFrame,
    *,
    new_records: list[dict[str, object]] | None = None,
) -> None:
    """Validate hashes and content of newly written cycle/image files only."""
    if new_records is None:
        new_paths = set(image_index["image_path"].astype(str))
        new_image_index = image_index.loc[
            image_index["image_path"].astype(str).isin(new_paths)
        ]
    else:
        new_paths = {str(record["image_path"]) for record in new_records}
        new_image_index = image_index.loc[
            image_index["image_path"].astype(str).isin(new_paths)
        ]
    _validate_cycle_files(dataset_dir, manifest, new_cycle_index, image_index)
    _validate_image_files(dataset_dir, new_image_index)


def _validate_dataset_structure(
    dataset_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_dir}")
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset is missing dataset_manifest.json: {dataset_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"dataset manifest is not valid JSON: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest must be an object")
    dataset_id = validate_dataset_id(str(manifest.get("dataset_id", "")))
    if dataset_dir.name != dataset_id:
        raise ValueError("dataset directory name must equal manifest.dataset_id")
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported dataset schema version")
    if manifest.get("cycle_name_width") != CYCLE_NAME_WIDTH:
        raise ValueError("unexpected cycle name width")
    source_schema = manifest.get("source_processed_schema")
    if not isinstance(source_schema, list):
        raise ValueError("manifest is missing source_processed_schema")

    cycle_path = dataset_dir / "cycle_index.parquet"
    image_path = dataset_dir / "image_index.parquet"
    if not cycle_path.is_file() or not image_path.is_file():
        raise FileNotFoundError("dataset is missing cycle_index.parquet or image_index.parquet")
    cycle_index = pd.read_parquet(cycle_path)
    image_index = pd.read_parquet(image_path)
    _require_columns(cycle_index, _CYCLE_INDEX_COLUMNS, "cycle_index")
    _require_columns(image_index, _IMAGE_INDEX_COLUMNS, "image_index")
    _validate_index_hash(manifest, "cycle_index", cycle_path, len(cycle_index))
    _validate_index_hash(manifest, "image_index", image_path, len(image_index))
    _validate_cycle_index(cycle_index, manifest)
    _validate_image_index(image_index, cycle_index)
    _validate_manifest_counts(manifest, cycle_index, image_index)
    _validate_source_runs(manifest, cycle_index)
    return manifest, cycle_index, image_index


def _validate_existing_file_paths(
    dataset_dir: Path,
    cycle_index: pd.DataFrame,
    image_index: pd.DataFrame,
) -> None:
    """Check existing data paths without reading or hashing their contents."""
    for value in cycle_index.loc[cycle_index["published"].astype(bool), "data_path"].tolist():
        path = dataset_dir / _safe_dataset_relative_path(str(value))
        if not path.is_file():
            raise FileNotFoundError(f"cycle data file does not exist: {path}")
    for value in image_index["image_path"].tolist():
        path = dataset_dir / _safe_dataset_relative_path(str(value))
        if not path.is_file():
            raise FileNotFoundError(f"dataset image does not exist: {path}")


def _validate_cycle_index(frame: pd.DataFrame, manifest: dict[str, Any]) -> None:  # noqa: C901
    if frame.duplicated(["cycle_uid"]).any():
        raise ValueError("cycle_index cycle_uid must be unique")
    if frame.duplicated(["cycle_name"]).any():
        published_names = frame.loc[frame["published"].astype(bool), "cycle_name"]
        if published_names.duplicated().any():
            raise ValueError("published cycle_name must be unique")
    if frame.duplicated(["experiment_id", "cycle_id"]).any():
        raise ValueError("cycle_index experiment_id/cycle_id must be unique")
    published = frame.loc[frame["published"].astype(bool)].copy()
    indices = sorted(int(value) for value in published["dataset_cycle_index"].dropna())
    if indices != list(range(1, len(indices) + 1)):
        raise ValueError("published dataset_cycle_index must be consecutive")
    for row in frame.to_dict(orient="records"):
        expected_uid = make_cycle_uid(str(row["experiment_id"]), str(row["cycle_id"]))
        if str(row["cycle_uid"]) != expected_uid:
            raise ValueError(f"cycle_uid disagrees with source identity: {expected_uid}")
        is_published = bool(row["published"])
        if is_published:
            name = str(row["cycle_name"])
            if parse_cycle_name(name) != int(row["dataset_cycle_index"]):
                raise ValueError("cycle_name disagrees with dataset_cycle_index")
            if int(row["processed_row_count"]) <= 0:
                raise ValueError("published cycle must have Processed rows")
            expected_recommended = (
                str(row["cycle_status"]) == "valid"
                and str(row["baseline_status"]) == "available"
            )
            if bool(row["recommended_for_analysis"]) != expected_recommended:
                raise ValueError("recommended_for_analysis disagrees with cycle facts")
            if pd.isna(row["data_path"]) or pd.isna(row["data_sha256"]):
                raise ValueError("published cycle must have a data file record")
            if pd.isna(row["data_size_bytes"]):
                raise ValueError("published cycle must have a data size record")
        else:
            if not pd.isna(row["dataset_cycle_index"]) or not pd.isna(row["cycle_name"]):
                raise ValueError("unpublished cycle must not have a dataset index or name")
            if int(row["processed_row_count"]) != 0:
                raise ValueError("unpublished cycle must have zero Processed rows")
            if row["dataset_exclusion_reason"] != "no_processed_rows":
                raise ValueError("unpublished cycle has an invalid exclusion reason")
            if any(
                not pd.isna(row[column])
                for column in ("data_path", "data_sha256", "data_size_bytes")
            ):
                raise ValueError("unpublished cycle must not have a data file record")


def _validate_image_index(frame: pd.DataFrame, cycle_index: pd.DataFrame) -> None:
    if frame.duplicated(["image_id"]).any() or frame.duplicated(["image_path"]).any():
        raise ValueError("image_index image_id and image_path must be unique")
    published = cycle_index.loc[cycle_index["published"].astype(bool)]
    published_uids = set(published["cycle_uid"].astype(str))
    if not set(frame["cycle_uid"].astype(str)).issubset(published_uids):
        raise ValueError("image_index references an unpublished cycle")
    cycle_names = dict(zip(published["cycle_uid"], published["cycle_name"], strict=True))
    image_counts = frame.groupby("cycle_uid", dropna=False).size().to_dict()
    for row in cycle_index.loc[cycle_index["published"].astype(bool)].to_dict(orient="records"):
        if int(row["image_count"]) != int(image_counts.get(row["cycle_uid"], 0)):
            raise ValueError("cycle_index image_count disagrees with image_index")
    for row in frame.to_dict(orient="records"):
        if str(row["cycle_name"]) != str(cycle_names[str(row["cycle_uid"])]):
            raise ValueError("image_index cycle_name disagrees with cycle_index")
        image_path = _safe_dataset_relative_path(str(row["image_path"]))
        if not image_path.startswith("images/"):
            raise ValueError("image_path must be inside images/")
        if Path(image_path).stem != str(row["image_id"]):
            raise ValueError("image_id must equal image_path stem")
        source_path = str(row["source_relative_path"])
        if (
            not source_path
            or "\\" in source_path
            or Path(source_path).is_absolute()
            or ".." in Path(source_path).parts
        ):
            raise ValueError("source_relative_path must be safe and relative")


def _validate_cycle_files(  # noqa: C901
    dataset_dir: Path,
    manifest: dict[str, Any],
    cycle_index: pd.DataFrame,
    image_index: pd.DataFrame,
) -> None:
    source_schema = manifest["source_processed_schema"]
    expected_fixed = [
        ("dataset_id", "string"),
        ("dataset_schema_version", "int64"),
        ("dataset_cycle_index", "int64"),
        ("cycle_name", "string"),
        ("cycle_uid", "string"),
    ]
    referenced_images = set(image_index["image_path"].astype(str))
    for row in cycle_index.loc[cycle_index["published"].astype(bool)].to_dict(orient="records"):
        relative = _safe_dataset_relative_path(str(row["data_path"]))
        if not relative.startswith("cycles/"):
            raise ValueError("cycle data_path must be inside cycles/")
        path = dataset_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"cycle data file does not exist: {path}")
        if sha256_file(path) != str(row["data_sha256"]):
            raise ValueError(f"cycle data SHA mismatch: {path}")
        if path.stat().st_size != int(row["data_size_bytes"]):
            raise ValueError(f"cycle data size mismatch: {path}")
        frame = pd.read_parquet(path)
        if len(frame) != int(row["processed_row_count"]):
            raise ValueError(f"cycle row count mismatch: {path}")
        schema = pq.read_schema(path)
        expected_names = [field["name"] for field in source_schema]
        actual_names = [field.name for field in schema]
        fixed_names = [name for name, _ in expected_fixed]
        if actual_names != expected_names + fixed_names:
            raise ValueError(f"cycle schema columns differ from source schema: {path}")
        actual_source = [
            {
                "name": field.name,
                "logical_type": str(field.type),
                "nullable": bool(field.nullable),
            }
            for field in list(schema)[: len(source_schema)]
        ]
        logical_schema_compatible(source_schema, actual_source)
        for field, (name, logical_type) in zip(list(schema)[-5:], expected_fixed, strict=True):
            if field.name != name or str(field.type) != logical_type:
                raise ValueError(f"cycle Dataset field has wrong type: {path}")
        for name, expected in {
            "dataset_id": str(manifest["dataset_id"]),
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "dataset_cycle_index": int(row["dataset_cycle_index"]),
            "cycle_name": str(row["cycle_name"]),
            "cycle_uid": str(row["cycle_uid"]),
        }.items():
            values = frame[name].drop_duplicates().tolist()
            if len(values) != 1 or values[0] != expected:
                raise ValueError(f"cycle Dataset field is inconsistent: {path} {name}")
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        if (
            timestamps.isna().any()
            or timestamps.duplicated().any()
            or not timestamps.is_monotonic_increasing
        ):
            raise ValueError(f"cycle timestamps must be unique and increasing: {path}")
        for role in image_roles(frame):
            missing = set(image_columns(role)) - set(frame.columns)
            if missing:
                raise ValueError(f"cycle image columns are incomplete: {path}")
            path_column, _, _ = image_columns(role)
            image_values = frame[path_column].dropna().astype(str)
            for value in image_values:
                if not _safe_dataset_relative_path(value).startswith("images/"):
                    raise ValueError(f"cycle image path is outside images/: {path}")
            if not set(image_values).issubset(referenced_images):
                raise ValueError(f"cycle image path is missing from image_index: {path}")


def _validate_image_files(dataset_dir: Path, image_index: pd.DataFrame) -> None:
    for row in image_index.to_dict(orient="records"):
        path = dataset_dir / _safe_dataset_relative_path(str(row["image_path"]))
        if not path.is_file():
            raise FileNotFoundError(f"dataset image does not exist: {path}")
        if sha256_file(path) != str(row["sha256"]):
            raise ValueError(f"dataset image SHA mismatch: {path}")
        if path.stat().st_size != int(row["file_size_bytes"]):
            raise ValueError(f"dataset image size mismatch: {path}")


def _validate_orphans(
    dataset_dir: Path,
    cycle_index: pd.DataFrame,
    image_index: pd.DataFrame,
) -> None:
    expected_cycles = {str(value) for value in cycle_index["data_path"].dropna()}
    expected_images = {str(value) for value in image_index["image_path"].dropna()}
    for directory, expected, suffixes in (
        (dataset_dir / "cycles", expected_cycles, {".parquet"}),
        (dataset_dir / "images", expected_images, {".jpg", ".jpeg", ".png", ".bmp", ".webp"}),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {directory}")
        for path in directory.rglob("*"):
            if (
                not path.is_file()
                or path.name in _IGNORED_NAMES
                or path.name.startswith(".tmp-")
                or path.name.endswith(".tmp")
            ):
                continue
            relative = path.relative_to(dataset_dir).as_posix()
            if path.suffix.lower() in suffixes and relative not in expected:
                raise ValueError(f"orphan dataset file: {path}")


def _validate_index_hash(
    manifest: dict[str, Any], name: str, path: Path, row_count: int
) -> None:
    files = manifest.get("files")
    entry = files.get(name) if isinstance(files, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"manifest is missing file entry: {name}")
    if entry.get("path") != path.name or entry.get("row_count") != row_count:
        raise ValueError(f"manifest file entry disagrees with {name}")
    if entry.get("sha256") != sha256_file(path):
        raise ValueError(f"manifest SHA mismatch: {name}")


def _validate_manifest_counts(
    manifest: dict[str, Any], cycle_index: pd.DataFrame, image_index: pd.DataFrame
) -> None:
    expected = {
        "summary_cycle_count": len(cycle_index),
        "published_cycle_count": int(cycle_index["published"].sum()),
        "excluded_cycle_count": int((~cycle_index["published"]).sum()),
        "image_count": len(image_index),
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"manifest count disagrees: {name}")


def _validate_source_runs(
    manifest: dict[str, Any], cycle_index: pd.DataFrame
) -> None:
    source_runs = manifest.get("source_runs")
    if not isinstance(source_runs, list):
        raise ValueError("manifest is missing source_runs")
    if not all(isinstance(item, dict) for item in source_runs):
        raise ValueError("manifest source_runs entries must be objects")
    for item in source_runs:
        source_path = str(item["source_run_path"])
        if not source_path or Path(source_path).is_absolute() or ".." in Path(source_path).parts:
            raise ValueError("manifest source_run_path must be relative and safe")
    ids = [str(item["experiment_id"]) for item in source_runs]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest source_runs experiment_id must be unique")
    cycle_ids = set(cycle_index["experiment_id"].astype(str))
    if set(ids) != cycle_ids:
        raise ValueError("manifest source_runs must cover cycle_index experiments")


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _safe_dataset_relative_path(value: str) -> str:
    path = Path(value)
    if "\\" in value or path.is_absolute() or ".." in path.parts or not value:
        raise ValueError(f"dataset path must be relative and safe: {value!r}")
    return path.as_posix()


_V2_CYCLE_COLUMNS = {
    "cycle_name",
    "cycle_uid",
    "experiment_id",
    "experiment_date",
    "source_cycle_id",
    "start_time",
    "end_time",
    "duration_seconds",
    "row_count",
    "data_path",
    "csv_path",
    "publication_path",
    "rgb_coverage_path",
    "image_count",
    "cycle_status",
    "cycle_status_reason",
    "baseline_status",
}
_V2_IMAGE_COLUMNS = {
    "image_id",
    "cycle_name",
    "cycle_uid",
    "source_camera_id",
    "initial_camera_slot",
    "source_role",
    "image_time",
    "matched_timestamp",
    "offset_seconds",
    "cycle_stage",
    "source_relative_path",
    "sha256",
    "file_size_bytes",
}
_V2_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _validate_v2_dataset(dataset_dir: Path, manifest: dict[str, Any]) -> None:  # noqa: C901
    _validate_v2_structure(dataset_dir, manifest=manifest)
    cycle_index = pd.read_parquet(dataset_dir / "cycle_index.parquet")
    image_metadata = pd.read_parquet(dataset_dir / "image_metadata.parquet")
    source_schema = manifest.get("source_processed_schema")
    if not isinstance(source_schema, list):
        raise ValueError("v2 manifest is missing source_processed_schema")
    scanned_images: list[pd.DataFrame] = []
    for row in cycle_index.to_dict(orient="records"):
        cycle_name = str(row["cycle_name"])
        parquet_path = dataset_dir / str(row["data_path"])
        csv_path = dataset_dir / str(row["csv_path"])
        publication_path = dataset_dir / str(row["publication_path"])
        coverage_path = dataset_dir / str(row["rgb_coverage_path"])
        for path in (parquet_path, csv_path, publication_path, coverage_path):
            if not path.is_file():
                raise FileNotFoundError(f"cycle asset does not exist: {path}")
        manifest_cycle = _manifest_cycle(manifest, cycle_name)
        if manifest_cycle is None:
            raise ValueError(f"manifest is missing cycle asset record: {cycle_name}")
        asset_hashes = manifest_cycle.get("asset_sha256")
        if not isinstance(asset_hashes, dict):
            raise ValueError(f"manifest is missing cycle asset hashes: {cycle_name}")
        for key, path in {
            "parquet": parquet_path,
            "csv": csv_path,
            "publication": publication_path,
            "rgb_coverage": coverage_path,
        }.items():
            if str(asset_hashes.get(key)) != sha256_file(path):
                raise ValueError(f"cycle asset SHA mismatch: {path}")
        frame = pd.read_parquet(parquet_path)
        if len(frame) != int(row["row_count"]):
            raise ValueError(f"cycle row count mismatch: {parquet_path}")
        _validate_v2_cycle_schema(parquet_path, source_schema)
        csv_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
        if csv_columns != frame.columns.tolist():
            raise ValueError(f"cycle CSV columns differ from Parquet: {csv_path}")
        image_root = dataset_dir / "images" / cycle_name
        actual_image_ids = {
            path.stem
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in _V2_IMAGE_SUFFIXES
        }
        expected_image_ids = set(
            image_metadata.loc[image_metadata["cycle_name"].eq(cycle_name), "image_id"].astype(str)
        )
        if actual_image_ids - expected_image_ids:
            raise ValueError(f"orphan image file in {cycle_name}")
        scanned_images.append(
            _scan_v2_cycle_images(dataset_dir, cycle_name, image_metadata)
        )
        if len(scanned_images[-1]) != int(row["image_count"]):
            raise ValueError(f"cycle image count mismatch: {cycle_name}")
        if manifest_cycle:
            summary = manifest_cycle.get("image_summary")
            if (
                isinstance(summary, dict)
                and int(summary.get("total_image_count", -1)) != int(row["image_count"])
            ):
                raise ValueError(f"manifest image count mismatch: {cycle_name}")
    all_scanned = (
        pd.concat(scanned_images, ignore_index=True)
        if scanned_images
        else pd.DataFrame(columns=["image_id"])
    )
    if len(all_scanned) != len(image_metadata):
        raise ValueError("image_metadata does not match current image directories")
    if set(all_scanned["image_id"].astype(str)) != set(image_metadata["image_id"].astype(str)):
        raise ValueError("image_metadata image IDs do not match current image directories")
    for row in all_scanned.to_dict(orient="records"):
        path = Path(str(row["path"]))
        metadata_row = image_metadata.loc[
            image_metadata["image_id"].eq(row["image_id"])
        ].iloc[0]
        expected_sha = str(metadata_row["sha256"])
        expected_size = int(metadata_row["file_size_bytes"])
        if sha256_file(path) != expected_sha:
            raise ValueError(f"image SHA mismatch: {path}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"image size mismatch: {path}")
    _validate_v2_orphans(dataset_dir, cycle_index, all_scanned)


def _validate_v2_structure(  # noqa: C901
    dataset_dir: Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_dir}")
    if manifest is None:
        path = dataset_dir / "dataset_manifest.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"dataset manifest is not valid JSON: {path}") from error
        if not isinstance(payload, dict):
            raise ValueError("dataset manifest must be an object")
        manifest = payload
    if manifest.get("dataset_schema_version") != DATASET_V2_SCHEMA_VERSION:
        raise ValueError("unsupported Dataset v2 schema version")
    dataset_id = validate_dataset_id(str(manifest.get("dataset_id", "")))
    if dataset_dir.name != dataset_id:
        raise ValueError("dataset directory name must equal manifest.dataset_id")
    for directory in (dataset_dir / "cycles", dataset_dir / "images"):
        if not directory.is_dir():
            raise FileNotFoundError(f"dataset is missing {directory.name}/")
    for name in ("README.md", "cycle_index.parquet", "image_metadata.parquet"):
        if not (dataset_dir / name).is_file():
            raise FileNotFoundError(f"dataset is missing {name}")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("manifest is missing top-level file hashes")
    for key, name in (
        ("cycle_index", "cycle_index.parquet"),
        ("image_metadata", "image_metadata.parquet"),
    ):
        entry = files.get(key)
        if not isinstance(entry, dict) or str(entry.get("path")) != name:
            raise ValueError(f"manifest file record is invalid: {key}")
        if str(entry.get("sha256")) != sha256_file(dataset_dir / name):
            raise ValueError(f"manifest file SHA mismatch: {name}")
    cycle_index = pd.read_parquet(dataset_dir / "cycle_index.parquet")
    image_metadata = pd.read_parquet(dataset_dir / "image_metadata.parquet")
    expected_manifest_counts = {
        "summary_cycle_count": len(cycle_index),
        "image_count": len(image_metadata),
    }
    for name, expected in expected_manifest_counts.items():
        if manifest.get(name) != expected:
            raise ValueError(f"manifest count disagrees: {name}")
    for name, expected in (
        ("cycle_index", len(cycle_index)),
        ("image_metadata", len(image_metadata)),
    ):
        entry = manifest.get(name)
        if not isinstance(entry, dict) or entry.get("row_count") != expected:
            raise ValueError(f"manifest row count disagrees: {name}")
    _require_columns(cycle_index, _V2_CYCLE_COLUMNS, "cycle_index")
    _require_columns(image_metadata, _V2_IMAGE_COLUMNS, "image_metadata")
    if cycle_index.duplicated(["cycle_name"]).any() or cycle_index.duplicated(["cycle_uid"]).any():
        raise ValueError("cycle_index cycle identities must be unique")
    if image_metadata.duplicated(["image_id"]).any():
        raise ValueError("image_metadata image_id must be unique")
    cycle_names = [str(value) for value in cycle_index["cycle_name"]]
    expected_names = [format_cycle_name(index) for index in range(1, len(cycle_names) + 1)]
    if cycle_names != expected_names:
        raise ValueError("cycle_name indices must be consecutive")
    cycles = manifest.get("cycles")
    if not isinstance(cycles, list):
        raise ValueError("manifest is missing cycles")
    manifest_names = {
        str(item["cycle_name"])
        for item in cycles
        if isinstance(item, dict) and "cycle_name" in item
    }
    index_names = set(cycle_index["cycle_name"].astype(str))
    if manifest_names != index_names:
        raise ValueError("manifest cycles and cycle_index disagree")
    for record in cycles:
        if not isinstance(record, dict):
            raise ValueError("manifest contains an invalid cycle record")
        _validate_v2_assessment(record)
    for row in cycle_index.to_dict(orient="records"):
        expected_uid = make_v2_cycle_uid(
            str(row["experiment_id"]), str(row["source_cycle_id"])
        )
        if str(row["cycle_uid"]) != expected_uid:
            raise ValueError("cycle_uid disagrees with source identity")
        for column in ("data_path", "csv_path", "publication_path", "rgb_coverage_path"):
            relative = _safe_dataset_relative_path(str(row[column]))
            if not relative.startswith("cycles/"):
                raise ValueError(f"{column} must be inside cycles/")
    for row in image_metadata.to_dict(orient="records"):
        source = str(row["source_relative_path"])
        path = Path(source)
        if not source or path.is_absolute() or ".." in path.parts or "\\" in source:
            raise ValueError("source_relative_path must be safe and relative")
        if str(row["cycle_uid"]) not in set(cycle_index["cycle_uid"].astype(str)):
            raise ValueError("image_metadata references unknown cycle")
    return manifest, cycle_index, image_metadata


def _validate_v2_new_assets(  # noqa: C901
    dataset_dir: Path,
    cycle_index: pd.DataFrame,
    image_metadata: pd.DataFrame,
) -> None:
    """Validate only newly materialized v2 assets in build/append staging."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset staging does not exist: {dataset_dir}")
    payload = json.loads(
        (dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest must be an object")
    for row in cycle_index.to_dict(orient="records"):
        for column in ("data_path", "csv_path", "publication_path", "rgb_coverage_path"):
            path = dataset_dir / _safe_dataset_relative_path(str(row[column]))
            if not path.is_file():
                raise FileNotFoundError(f"new cycle asset does not exist: {path}")
        parquet = dataset_dir / _safe_dataset_relative_path(str(row["data_path"]))
        if len(pd.read_parquet(parquet)) != int(row["row_count"]):
            raise ValueError(f"new cycle row count mismatch: {parquet}")
        record = _manifest_cycle(payload, str(row["cycle_name"]))
        if record is None or not isinstance(record.get("asset_sha256"), dict):
            raise ValueError(f"new cycle asset hashes are missing: {row['cycle_name']}")
        for key, column in {
            "parquet": "data_path",
            "csv": "csv_path",
            "publication": "publication_path",
            "rgb_coverage": "rgb_coverage_path",
        }.items():
            path = dataset_dir / _safe_dataset_relative_path(str(row[column]))
            if str(record["asset_sha256"].get(key)) != sha256_file(path):
                raise ValueError(f"new cycle asset SHA mismatch: {path}")
    for cycle_name in image_metadata["cycle_name"].astype(str).unique():
        scanned = _scan_v2_cycle_images(dataset_dir, cycle_name, image_metadata)
        expected = image_metadata.loc[image_metadata["cycle_name"].eq(cycle_name)]
        if set(scanned["image_id"].astype(str)) != set(expected["image_id"].astype(str)):
            raise ValueError(f"new image metadata does not match files: {cycle_name}")
        expected_by_id = {
            str(row["image_id"]): row for row in expected.to_dict(orient="records")
        }
        for row in scanned.to_dict(orient="records"):
            image_id = str(row["image_id"])
            path = Path(str(row["path"]))
            metadata = expected_by_id[image_id]
            if sha256_file(path) != str(metadata["sha256"]):
                raise ValueError(f"image SHA mismatch: {path}")
            if path.stat().st_size != int(metadata["file_size_bytes"]):
                raise ValueError(f"image size mismatch: {path}")


def _validate_v2_cycle_schema(path: Path, source_schema: list[object]) -> None:
    schema = pq.read_schema(path)
    actual_names = [field.name for field in schema]
    expected_names = [str(field["name"]) for field in source_schema if isinstance(field, dict)]
    if actual_names != expected_names:
        raise ValueError(f"cycle schema columns differ from source schema: {path}")
    actual = [
        {"name": field.name, "logical_type": str(field.type), "nullable": bool(field.nullable)}
        for field in schema
    ]
    logical_schema_compatible(
        [field for field in source_schema if isinstance(field, dict)],
        actual,
    )


def _scan_v2_cycle_images(
    dataset_dir: Path,
    cycle_name: str,
    image_metadata: pd.DataFrame,
) -> pd.DataFrame:
    from .dataset_images import scan_cycle_images

    return scan_cycle_images(dataset_dir, cycle_name, image_metadata)


def _manifest_cycle(manifest: dict[str, Any], cycle_name: str) -> dict[str, Any] | None:
    cycles = manifest.get("cycles")
    if not isinstance(cycles, list):
        return None
    for record in cycles:
        if isinstance(record, dict) and str(record.get("cycle_name")) == cycle_name:
            return record
    return None


def _validate_v2_assessment(record: dict[str, Any]) -> None:
    assessment = record.get("assessment")
    if not isinstance(assessment, dict):
        raise ValueError(f"cycle assessment is missing: {record.get('cycle_name')}")
    required = {"status", "reasons", "note", "updated_at"}
    if set(assessment) != required:
        raise ValueError(f"cycle assessment fields are invalid: {record.get('cycle_name')}")
    if str(assessment["status"]) not in {"valid", "partial", "incomplete", "invalid"}:
        raise ValueError(f"cycle assessment status is invalid: {record.get('cycle_name')}")
    if not isinstance(assessment["reasons"], list):
        raise ValueError(f"cycle assessment reasons are invalid: {record.get('cycle_name')}")


def _validate_v2_orphans(
    dataset_dir: Path,
    cycle_index: pd.DataFrame,
    scanned_images: pd.DataFrame,
) -> None:
    expected_cycle_assets = {
        f"{row['cycle_name']}.parquet"
        for row in cycle_index.to_dict(orient="records")
    } | {
        f"{row['cycle_name']}.csv"
        for row in cycle_index.to_dict(orient="records")
    } | {
        f"{row['cycle_name']}.png"
        for row in cycle_index.to_dict(orient="records")
    } | {
        f"{row['cycle_name']}_rgb_coverage.png"
        for row in cycle_index.to_dict(orient="records")
    }
    actual_cycle_assets = {
        path.name
        for path in (dataset_dir / "cycles").iterdir()
        if path.is_file() and not _ignored_v2_name(path.name)
    }
    if actual_cycle_assets != expected_cycle_assets:
        raise ValueError("cycles/ contains orphan or missing ordinary assets")
    expected_image_paths = {
        Path(str(path)).relative_to(dataset_dir).as_posix()
        for path in scanned_images.get("path", pd.Series(dtype=object)).tolist()
    }
    actual_image_paths = {
        path.relative_to(dataset_dir).as_posix()
        for path in (dataset_dir / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in _V2_IMAGE_SUFFIXES
    }
    if actual_image_paths != expected_image_paths:
        raise ValueError("images/ contains orphan image files")


def _ignored_v2_name(name: str) -> bool:
    return name == ".DS_Store" or name.startswith(".tmp-") or name.endswith(".tmp")
