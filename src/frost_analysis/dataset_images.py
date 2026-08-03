"""Publish Prepared image matches into the flat dataset image directory."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pandas as pd

from .dataset import make_cycle_uid
from .images import image_columns, image_roles
from .io import relative_posix_path, sha256_file


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
