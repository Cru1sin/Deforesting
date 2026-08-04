"""Collect and scan Dataset images and compute merged RGB coverage intervals."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .dataset import make_cycle_uid
from .images import image_columns, image_roles

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def stable_logical_image_id(
    cycle_uid: str, source_camera_id: str, source_relative_path: str
) -> str:
    """Return an identity hash for source metadata, never for image bytes."""
    payload = (
        f"{cycle_uid}\0{source_camera_id}\0{source_relative_path}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def collect_final_images(  # noqa: C901
    prepared: pd.DataFrame,
    *,
    input_dir: Path,
    cycle_names: Mapping[tuple[str, str], str],
) -> list[dict[str, object]]:
    """Collect Prepared image matches without re-matching or hashing bytes."""
    roles = image_roles(prepared)
    candidates: dict[tuple[str, str, str], dict[str, object]] = {}
    source_matches: dict[str, tuple[str, pd.Timestamp]] = {}
    source_camera_roles: dict[str, str] = {}
    for values in prepared.to_dict(orient="records"):
        key = (str(values["experiment_id"]), str(values["cycle_id"]))
        cycle_name = cycle_names.get(key)
        if cycle_name is None:
            continue
        cycle_uid = make_cycle_uid(*key)
        for role in roles:
            path_column, time_column, offset_column = image_columns(role)
            raw_path = values.get(path_column)
            raw_time = values.get(time_column)
            raw_offset = values.get(offset_column)
            if pd.isna(raw_path):
                if not pd.isna(raw_time) or not pd.isna(raw_offset):
                    raise ValueError(f"incomplete image match for {path_column}")
                continue
            if pd.isna(raw_time) or pd.isna(raw_offset):
                raise ValueError(f"incomplete image match for {path_column}")
            raw_relative = str(raw_path).replace("\\", "/")
            relative_path = Path(raw_relative)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or "\\" in raw_relative
            ):
                raise ValueError(f"source image path must be safe and relative: {raw_path}")
            relative = relative_path.as_posix()
            lexical_source_path = input_dir / relative
            try:
                lexical_source_path.relative_to(input_dir)
            except ValueError as error:
                raise ValueError(f"source image path escapes input_dir: {raw_path}") from error
            source_path = lexical_source_path.resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"matched source image does not exist: {source_path}")
            source_camera_id = relative.split("/", 1)[0]
            if not source_camera_id or "__" in source_camera_id:
                raise ValueError(f"invalid source_camera_id: {source_camera_id!r}")
            previous_role = source_camera_roles.get(source_camera_id)
            if previous_role is not None and previous_role != role:
                raise ValueError(
                    "one source camera was matched to multiple roles: "
                    f"{source_camera_id}: {previous_role!r}, {role!r}"
                )
            source_camera_roles[source_camera_id] = role
            image_time = pd.Timestamp(raw_time)
            source_match = source_matches.get(relative)
            if source_match is not None and source_match != (cycle_uid, image_time):
                raise ValueError(f"source image is matched to multiple cycles: {relative}")
            source_matches[relative] = (cycle_uid, image_time)
            record = {
                "cycle_uid": cycle_uid,
                "cycle_name": cycle_name,
                "source_camera_id": source_camera_id,
                "image_time": image_time,
                "matched_timestamp": pd.Timestamp(values["timestamp"]),
                "offset_seconds": float(raw_offset),
                "cycle_stage": str(values.get("cycle_stage", "")),
                "source_relative_path": relative,
                "file_name": Path(relative).name,
                "source_path": source_path,
            }
            candidate_key = (cycle_uid, source_camera_id, relative)
            previous = candidates.get(candidate_key)
            if previous is not None:
                for field in (
                    "image_time",
                    "matched_timestamp",
                    "offset_seconds",
                    "cycle_stage",
                ):
                    if previous[field] != record[field]:
                        raise ValueError(f"inconsistent image match for {relative}")
                continue
            candidates[candidate_key] = record

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in candidates.values():
        key = (str(record["cycle_name"]), str(record["source_camera_id"]))
        grouped.setdefault(key, []).append(record)

    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_filenames: set[tuple[str, str, str]] = set()
    for group_key in sorted(grouped):
        group = sorted(
            grouped[group_key],
            key=lambda item: (
                pd.Timestamp(cast(Any, item["image_time"])),
                str(item["source_relative_path"]),
            ),
        )
        cycle_name, source_camera_id = group_key
        current_role = source_camera_roles[source_camera_id]
        for frame_index, record in enumerate(group, start=1):
            file_name = str(record["file_name"])
            filename_key = (cycle_name, source_camera_id, file_name)
            if filename_key in seen_filenames:
                raise ValueError(
                    "duplicate source basename within camera: "
                    f"{cycle_name}/{source_camera_id}/{file_name}"
                )
            seen_filenames.add(filename_key)
            image_id = stable_logical_image_id(
                str(record["cycle_uid"]), source_camera_id, str(record["source_relative_path"])
            )
            if image_id in seen_ids:
                raise ValueError(f"duplicate generated image_id: {image_id}")
            seen_ids.add(image_id)
            record.update(
                {
                    "image_id": image_id,
                    "frame_index": frame_index,
                    "initial_camera_slot": current_role,
                    "current_role": current_role,
                    "image_path": (
                        f"images/{cycle_name}/{source_camera_id}__{current_role}/"
                        f"{file_name}"
                    ),
                    "file_size_bytes": Path(str(record["source_path"])).stat().st_size,
                }
            )
            records.append(record)
    return records


def image_metadata_frame_final(records: list[dict[str, object]]) -> pd.DataFrame:
    """Build the final metadata table without current role or image SHA."""
    columns = [
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
    ]
    return pd.DataFrame(
        [{column: record[column] for column in columns} for record in records],
        columns=columns,
    )


def copy_final_image(record: Mapping[str, object], dataset_dir: Path) -> None:
    """Copy a source image and perform only an immediate size/existence check."""
    source = Path(str(record["source_path"]))
    target = dataset_dir / str(record["image_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if not target.is_file() or target.stat().st_size != source.stat().st_size:
        raise ValueError(f"copied image is incomplete: {target}")


def _parse_role_directory(name: str) -> tuple[str, str]:
    parts = name.split("__")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            "camera directory must be <source_camera_id>__<current_role>: "
            f"{name!r}"
        )
    if any(separator in part for part in parts for separator in ("/", "\\")):
        raise ValueError(f"camera directory contains a path separator: {name!r}")
    return parts[0], parts[1]


def scan_final_cycle_images(
    dataset_root: Path,
    cycle_name: str,
    image_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Scan current role folders and join only currently available metadata."""
    columns = [
        "image_id",
        "cycle_name",
        "camera_role",
        "source_camera_id",
        "file_name",
        "path",
        "frame_index",
        "initial_camera_slot",
        "image_time",
        "matched_timestamp",
        "offset_seconds",
        "cycle_stage",
        "source_relative_path",
        "file_size_bytes",
    ]
    root = dataset_root / "images" / cycle_name
    if not root.is_dir():
        return pd.DataFrame(columns=columns)
    scoped = image_metadata.loc[
        image_metadata["cycle_name"].astype(str).eq(cycle_name)
    ].copy()
    key_columns = ["cycle_name", "source_camera_id", "file_name"]
    if scoped.duplicated(key_columns).any():
        raise ValueError(f"image metadata has duplicate source/file key: {cycle_name}")
    lookup = {
        tuple(str(row[column]) for column in key_columns): row
        for row in scoped.to_dict(orient="records")
    }
    rows: list[dict[str, object]] = []
    source_roles: dict[str, str] = {}
    for role_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        source_camera_id, current_role = _parse_role_directory(role_dir.name)
        previous_role = source_roles.get(source_camera_id)
        if previous_role is not None and previous_role != current_role:
            raise ValueError(
                "source camera is assigned to multiple current roles: "
                f"{cycle_name}/{source_camera_id}"
            )
        source_roles[source_camera_id] = current_role
        for image_path in sorted(path for path in role_dir.iterdir() if path.is_file()):
            if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            key = (cycle_name, source_camera_id, image_path.name)
            metadata = lookup.get(key)
            if metadata is None:
                continue
            row = dict(metadata)
            row.update(
                {
                    "camera_role": current_role,
                    "path": image_path,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def build_rgb_coverage_intervals(
    cycle_start: pd.Timestamp,
    cycle_end: pd.Timestamp,
    image_times: pd.Series,
    *,
    max_image_gap_seconds: float,
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Build one merged available/missing interval set for one camera role."""
    start = pd.Timestamp(cycle_start)
    end = pd.Timestamp(cycle_end)
    if end <= start:
        return {"available": [], "missing": []}
    times = pd.to_datetime(image_times, errors="coerce").dropna().sort_values().unique()
    available: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for value in times:
        image_time = pd.Timestamp(value)
        available_start = max(start, image_time)
        available_end = min(end, image_time + pd.Timedelta(seconds=max_image_gap_seconds))
        if available_end > available_start:
            available.append((available_start, available_end))
    available = _merge_intervals(available)
    missing: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    for available_start, available_end in available:
        if available_start > cursor:
            missing.append((cursor, available_start))
        cursor = max(cursor, available_end)
    if cursor < end:
        missing.append((cursor, end))
    return {"available": available, "missing": _merge_intervals(missing)}


def summarize_rgb_coverage(
    cycle_start: pd.Timestamp,
    cycle_end: pd.Timestamp,
    intervals: Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
) -> float:
    """Return the ratio represented by the exact intervals used for drawing."""
    total = (pd.Timestamp(cycle_end) - pd.Timestamp(cycle_start)).total_seconds()
    available = sum(
        max(0.0, (end - start).total_seconds())
        for start, end in intervals.get("available", [])
    )
    return 0.0 if total <= 0 else min(1.0, max(0.0, available / total))


def _merge_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ordered = sorted(
        (pd.Timestamp(start), pd.Timestamp(end))
        for start, end in intervals
        if end > start
    )
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


# The final contract deliberately replaces the legacy image-id/stem join.
scan_cycle_images = scan_final_cycle_images
