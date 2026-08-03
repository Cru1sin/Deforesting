"""Publish Prepared image matches into the flat dataset image directory."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .dataset import make_cycle_uid, make_v2_cycle_uid
from .images import image_columns, image_roles
from .io import relative_posix_path, sha256_file

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_matched_images(  # noqa: C901
    prepared: pd.DataFrame,
    *,
    input_dir: Path,
    cycle_names: Mapping[tuple[str, str], str],
) -> tuple[list[dict[str, object]], str]:
    """Collect unique Prepared image matches for published cycles."""
    roles = image_roles(prepared)
    for role in roles:
        missing = set(image_columns(role)) - set(prepared.columns)
        if missing:
            raise ValueError(f"incomplete image columns for {role}: {sorted(missing)}")

    candidates: dict[tuple[str, str, str], dict[str, object]] = {}
    source_metadata: dict[Path, tuple[str, int]] = {}
    source_matches: dict[str, tuple[str, pd.Timestamp]] = {}
    for values in prepared.to_dict(orient="records"):
        experiment_id = str(values["experiment_id"])
        cycle_id = str(values["cycle_id"])
        cycle_name = cycle_names.get((experiment_id, cycle_id))
        if cycle_name is None:
            continue
        cycle_uid = make_cycle_uid(experiment_id, cycle_id)
        timestamp = pd.Timestamp(values["timestamp"])
        for role in roles:
            path_column, time_column, offset_column = image_columns(role)
            raw_path = values[path_column]
            raw_time = values[time_column]
            raw_offset = values[offset_column]
            if pd.isna(raw_path):
                if not pd.isna(raw_time) or not pd.isna(raw_offset):
                    raise ValueError(f"incomplete image match for {path_column}")
                continue
            if pd.isna(raw_time) or pd.isna(raw_offset):
                raise ValueError(f"incomplete image match for {path_column}")
            raw_path_text = str(raw_path)
            raw_path_value = Path(raw_path_text)
            if raw_path_value.is_absolute() or ".." in raw_path_value.parts:
                raise ValueError(
                    f"source image path must be safe and relative: {raw_path_text}"
                )
            relative = relative_posix_path(input_dir / raw_path_value, input_dir)
            source_path = (input_dir / relative).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"matched source image does not exist: {source_path}")
            image_time = pd.Timestamp(raw_time)
            key = (cycle_uid, role, relative)
            existing_source = source_matches.get(relative)
            if existing_source is not None and existing_source != (cycle_uid, image_time):
                raise ValueError(
                    f"source image is matched to multiple cycles or times: {relative}"
                )
            source_matches[relative] = (cycle_uid, image_time)
            record = {
                "cycle_uid": cycle_uid,
                "cycle_name": cycle_name,
                "camera_role": role,
                "image_time": image_time,
                "matched_timestamp": timestamp,
                "offset_seconds": float(raw_offset),
                "cycle_stage": str(values["cycle_stage"]),
                "source_relative_path": relative,
                "source_path": source_path,
            }
            previous = candidates.get(key)
            if previous is not None:
                for field in ("image_time", "matched_timestamp", "offset_seconds", "cycle_stage"):
                    if previous[field] != record[field]:
                        raise ValueError(f"inconsistent image match for {relative}")
                continue
            if source_path not in source_metadata:
                source_metadata[source_path] = (
                    sha256_file(source_path),
                    source_path.stat().st_size,
                )
            sha256, size = source_metadata[source_path]
            record["sha256"] = sha256
            record["file_size_bytes"] = size
            candidates[key] = record

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in candidates.values():
        grouped.setdefault(
            (str(record["cycle_uid"]), str(record["camera_role"])), []
        ).append(record)

    records: list[dict[str, object]] = []
    for group_key in sorted(grouped):
        group = sorted(
            grouped[group_key],
            key=lambda item: (
                cast(pd.Timestamp, item["image_time"]),
                str(item["source_relative_path"]),
            ),
        )
        for index, record in enumerate(group, start=1):
            image_time = cast(pd.Timestamp, record["image_time"])
            source_path = Path(str(record["source_path"]))
            timestamp_token = image_time.strftime("%Y%m%dT%H%M%S%f")[:-3]
            image_id = (
                f"{record['cycle_name']}__{record['camera_role']}__"
                f"{index:06d}__{timestamp_token}"
            )
            record["image_id"] = image_id
            record["image_path"] = f"images/{image_id}{source_path.suffix.lower()}"
            records.append(record)

    inventory_records = sorted(
        (relative, sha256, size)
        for path, (sha256, size) in source_metadata.items()
        for relative in [relative_posix_path(path, input_dir)]
    )
    payload = json.dumps(
        inventory_records,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    inventory_hash = hashlib.sha256(payload).hexdigest()
    return records, inventory_hash


def rewrite_processed_image_paths(
    processed: pd.DataFrame,
    records: list[dict[str, object]],
) -> pd.DataFrame:
    """Rewrite source image paths using records from one source run."""
    result = processed.copy()
    mapping = {
        (str(record["cycle_uid"]), str(record["camera_role"]), str(record["source_relative_path"])):
        str(record["image_path"])
        for record in records
    }
    roles = image_roles(result)
    for row_index, row in result.iterrows():
        cycle_uid = make_cycle_uid(str(row["experiment_id"]), str(row["cycle_id"]))
        for role in roles:
            path_column, _, _ = image_columns(role)
            raw_path = row[path_column]
            if pd.isna(raw_path):
                continue
            key = (cycle_uid, role, str(raw_path))
            if key not in mapping:
                raise ValueError(f"Processed image path was not exported: {key}")
            result.at[row_index, path_column] = mapping[key]
    return result


def copy_dataset_image(record: Mapping[str, object], dataset_dir: Path) -> None:
    """Copy one source image and verify its bytes using the recorded SHA."""
    source_path = Path(str(record["source_path"]))
    destination = dataset_dir / str(record["image_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    expected = str(record["sha256"])
    if sha256_file(destination) != expected:
        raise ValueError(f"copied image SHA mismatch: {destination}")


def collect_cycle_images(  # noqa: C901
    prepared: pd.DataFrame,
    *,
    input_dir: Path,
    cycle_names: Mapping[tuple[str, str], str],
) -> tuple[list[dict[str, object]], str]:  # noqa: C901
    """Collect all Prepared matches and place them under initial slot directories.

    The initial directory is deliberately a neutral slot name.  After publication,
    the directory name is the authority for the current camera role; this function
    only records the source slot and never consults the YAML camera-role mapping.
    """
    roles = image_roles(prepared)
    for role in roles:
        missing = set(image_columns(role)) - set(prepared.columns)
        if missing:
            raise ValueError(f"incomplete image columns for {role}: {sorted(missing)}")

    source_camera_ids = {
        _source_camera_id(str(value))
        for row in prepared.to_dict(orient="records")
        for role in roles
        for value in [row.get(image_columns(role)[0])]
        if not pd.isna(value)
    }
    slot_by_camera = {
        camera_id: f"unassigned_{index:02d}"
        for index, camera_id in enumerate(sorted(source_camera_ids), start=1)
    }
    candidates: dict[tuple[str, str, str], dict[str, object]] = {}
    source_metadata: dict[Path, tuple[str, int]] = {}
    source_matches: dict[str, tuple[str, pd.Timestamp]] = {}
    for row in prepared.to_dict(orient="records"):
        key = (str(row["experiment_id"]), str(row["cycle_id"]))
        cycle_name = cycle_names.get(key)
        if cycle_name is None:
            raise ValueError(f"missing dataset cycle name for {key}")
        cycle_uid = make_v2_cycle_uid(*key)
        timestamp = pd.Timestamp(row["timestamp"])
        for source_role in roles:
            path_column, time_column, offset_column = image_columns(source_role)
            raw_path = row[path_column]
            raw_time = row[time_column]
            raw_offset = row[offset_column]
            if pd.isna(raw_path):
                if not pd.isna(raw_time) or not pd.isna(raw_offset):
                    raise ValueError(f"incomplete image match for {path_column}")
                continue
            if pd.isna(raw_time) or pd.isna(raw_offset):
                raise ValueError(f"incomplete image match for {path_column}")
            relative = _safe_source_relative(str(raw_path), input_dir)
            source_path = input_dir / relative
            if not source_path.is_file():
                raise FileNotFoundError(f"matched source image does not exist: {source_path}")
            source_role = str(source_role)
            camera_id = _source_camera_id(relative)
            slot = slot_by_camera[camera_id]
            image_time = pd.Timestamp(raw_time)
            previous_match = source_matches.get(relative)
            if previous_match is not None and previous_match != (cycle_uid, image_time):
                raise ValueError(f"source image is matched to multiple cycles or times: {relative}")
            source_matches[relative] = (cycle_uid, image_time)
            candidate_key = (cycle_uid, slot, relative)
            record = {
                "cycle_uid": cycle_uid,
                "cycle_name": cycle_name,
                "source_role": source_role,
                "camera_role": slot,
                "initial_camera_slot": slot,
                "source_camera_id": camera_id,
                "image_time": image_time,
                "matched_timestamp": timestamp,
                "offset_seconds": float(raw_offset),
                "cycle_stage": str(row["cycle_stage"]),
                "source_relative_path": relative,
                "source_path": source_path,
            }
            previous = candidates.get(candidate_key)
            if previous is not None:
                for field in ("image_time", "matched_timestamp", "offset_seconds", "cycle_stage"):
                    if previous[field] != record[field]:
                        raise ValueError(f"inconsistent image match for {relative}")
                continue
            if source_path not in source_metadata:
                source_metadata[source_path] = (
                    sha256_file(source_path),
                    source_path.stat().st_size,
                )
            sha256, size = source_metadata[source_path]
            record["sha256"] = sha256
            record["file_size_bytes"] = size
            candidates[candidate_key] = record

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in candidates.values():
        group_key = (str(record["cycle_name"]), str(record["camera_role"]))
        grouped.setdefault(group_key, []).append(record)

    records: list[dict[str, object]] = []
    for group_key in sorted(grouped):
        group = sorted(
            grouped[group_key],
            key=lambda item: (
            pd.Timestamp(cast(Any, item["image_time"])),
                str(item["source_relative_path"]),
            ),
        )
        for frame_index, record in enumerate(group, start=1):
            image_time = pd.Timestamp(cast(Any, record["image_time"]))
            source_path = Path(str(record["source_path"]))
            timestamp_token = image_time.strftime("%Y%m%dT%H%M%S%f")[:-3]
            image_id = (
                f"{record['cycle_name']}__{record['initial_camera_slot']}__"
                f"{frame_index:06d}__{timestamp_token}"
            )
            record["image_id"] = image_id
            record["image_path"] = (
                f"images/{record['cycle_name']}/{record['camera_role']}/"
                f"{image_id}{source_path.suffix.lower()}"
            )
            records.append(record)

    inventory_records = sorted(
        (relative, sha256, size)
        for path, (sha256, size) in source_metadata.items()
        for relative in [_safe_source_relative_from_path(path, input_dir)]
    )
    payload = json.dumps(inventory_records, ensure_ascii=False, separators=(",", ":"))
    inventory_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return records, inventory_hash


def rewrite_processed_image_paths_v2(
    processed: pd.DataFrame,
    records: list[dict[str, object]],
) -> pd.DataFrame:
    """Rewrite only Processed image path columns using source-role scoped records."""
    mapping = {
        (
            str(record["cycle_uid"]),
            str(record["source_role"]),
            str(record["source_relative_path"]),
        ): str(record["image_path"])
        for record in records
    }
    result = processed.copy()
    for index, row in result.iterrows():
        cycle_uid = make_v2_cycle_uid(str(row["experiment_id"]), str(row["cycle_id"]))
        for role in image_roles(result):
            path_column, _, _ = image_columns(role)
            raw_path = row[path_column]
            if pd.isna(raw_path):
                continue
            key = (cycle_uid, role, str(raw_path).replace("\\", "/"))
            if key not in mapping:
                raise ValueError(f"Processed image path was not exported: {key}")
            result.at[index, path_column] = mapping[key]
    return result


def _source_camera_id(relative: str) -> str:
    return relative.split("/", 1)[0]


def _safe_source_relative(raw_path: str, input_dir: Path) -> str:
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts or "\\" in raw_path:
        raise ValueError(f"source image path must be safe and relative: {raw_path}")
    candidate = input_dir / path
    try:
        relative = candidate.relative_to(input_dir)
    except ValueError as error:
        raise ValueError(f"source image path must be inside input_dir: {raw_path}") from error
    return relative.as_posix()


def _safe_source_relative_from_path(path: Path, input_dir: Path) -> str:
    try:
        return path.relative_to(input_dir).as_posix()
    except ValueError:
        return relative_posix_path(path, input_dir)


def scan_cycle_images(
    dataset_root: Path,
    cycle_name: str,
    image_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Scan the current cycle image folders and join immutable image metadata."""
    columns = [
        "image_id",
        "cycle_name",
        "camera_role",
        "path",
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
    ]
    root = dataset_root / "images" / cycle_name
    rows: list[dict[str, object]] = []
    if root.is_dir():
        for role_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for image_path in sorted(path for path in role_dir.iterdir() if path.is_file()):
                if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                rows.append(
                    {
                        "image_id": image_path.stem,
                        "cycle_name": cycle_name,
                        "camera_role": role_dir.name,
                        "path": image_path,
                    }
                )
    scanned = pd.DataFrame(rows)
    if scanned.empty:
        return pd.DataFrame(columns=columns)
    metadata = image_metadata.loc[image_metadata["cycle_name"].eq(cycle_name)].copy()
    if metadata["image_id"].duplicated().any():
        raise ValueError(f"image metadata has duplicate image_id in {cycle_name}")
    joined = scanned.merge(metadata, on=["image_id", "cycle_name"], how="left", indicator=True)
    if joined["_merge"].ne("both").any():
        missing = joined.loc[joined["_merge"].ne("both"), "image_id"].tolist()
        raise ValueError(f"image metadata is missing scanned images: {missing}")
    return joined.drop(columns="_merge")[[column for column in columns if column in joined]]
