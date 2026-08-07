"""Collect and scan Dataset images and compute merged RGB coverage intervals."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .dataset import make_cycle_uid
from .images import image_columns, image_roles


def collect_images(  # noqa: C901
    prepared: pd.DataFrame,
    *,
    input_dir: Path,
    cycle_names: Mapping[tuple[str, str], str],
) -> list[dict[str, object]]:
    """Collect Prepared image matches without re-matching or hashing bytes."""
    roles = image_roles(prepared)
    candidates: dict[tuple[str, str, str], dict[str, object]] = {}
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
            relative = str(raw_path).replace("\\", "/")
            source_path = input_dir / relative
            if not source_path.is_file():
                raise FileNotFoundError(f"matched source image does not exist: {source_path}")
            source_camera_id = relative.split("/", 1)[0]
            image_time = pd.Timestamp(raw_time)
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
            candidates.setdefault(candidate_key, record)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in candidates.values():
        key = (str(record["cycle_name"]), str(record["source_camera_id"]))
        grouped.setdefault(key, []).append(record)

    records: list[dict[str, object]] = []
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
        for frame_index, record in enumerate(group, start=1):
            file_name = str(record["file_name"])
            filename_key = (cycle_name, source_camera_id, file_name)
            if filename_key in seen_filenames:
                raise ValueError(
                    "duplicate source basename within camera: "
                    f"{cycle_name}/{source_camera_id}/{file_name}"
                )
            seen_filenames.add(filename_key)
            record.update(
                {
                    "frame_index": frame_index,
                    "image_path": f"images/{cycle_name}/{source_camera_id}/{file_name}",
                }
            )
            records.append(record)
    return records


def image_metadata_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    """Build the final metadata table without current role or image SHA."""
    columns = [
        "cycle_uid",
        "cycle_name",
        "source_camera_id",
        "file_name",
        "frame_index",
        "image_time",
        "matched_timestamp",
        "offset_seconds",
        "cycle_stage",
        "source_relative_path",
    ]
    return pd.DataFrame(
        [{column: record[column] for column in columns} for record in records],
        columns=columns,
    )


def copy_image(record: Mapping[str, object], dataset_dir: Path) -> None:
    """Copy one matched source image into its cycle/camera directory."""
    source = Path(str(record["source_path"]))
    target = dataset_dir / str(record["image_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def scan_cycle_images(
    dataset_root: Path,
    cycle_name: str,
    image_metadata: pd.DataFrame,
    camera_roles: Mapping[str, str],
) -> pd.DataFrame:
    """Join available source-camera files to metadata and Manifest roles."""
    columns = [
        "cycle_name",
        "camera_role",
        "source_camera_id",
        "file_name",
        "path",
        "frame_index",
        "image_time",
        "matched_timestamp",
        "offset_seconds",
        "cycle_stage",
        "source_relative_path",
    ]
    root = dataset_root / "images" / cycle_name
    if not root.is_dir():
        return pd.DataFrame(columns=columns)
    scoped = image_metadata.loc[image_metadata["cycle_name"].astype(str).eq(cycle_name)].copy()
    key_columns = ["cycle_name", "source_camera_id", "file_name"]
    if scoped.duplicated(key_columns).any():
        raise ValueError(f"image metadata has duplicate source/file key: {cycle_name}")
    lookup = {
        tuple(str(row[column]) for column in key_columns): row
        for row in scoped.to_dict(orient="records")
    }
    rows: list[dict[str, object]] = []
    for camera_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        source_camera_id = camera_dir.name
        camera_role = camera_roles.get(source_camera_id, source_camera_id)
        for image_path in sorted(path for path in camera_dir.iterdir() if path.is_file()):
            key = (cycle_name, source_camera_id, image_path.name)
            metadata = lookup.get(key)
            if metadata is None:
                continue
            row = {str(key): value for key, value in metadata.items()}
            row.update(
                {
                    "camera_role": camera_role,
                    "path": image_path,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _cycle_window(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce").dropna().sort_values()
    if timestamps.empty:
        raise ValueError("cycle has no valid timestamp")
    intervals = timestamps.diff().dropna().dt.total_seconds()
    positive = intervals.loc[intervals > 0]
    step = float(positive.median()) if not positive.empty else 1.0
    return (
        pd.Timestamp(timestamps.iloc[0]),
        pd.Timestamp(timestamps.iloc[-1]) + pd.Timedelta(seconds=step),
    )


def _cycle_image_summary(
    dataset_dir: Path,
    cycle_name: str,
    frame: pd.DataFrame,
    image_metadata: pd.DataFrame,
    registry: Mapping[str, Any],
    camera_roles: Mapping[str, str],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]],
]:
    start, end = _cycle_window(frame)
    images = scan_cycle_images(dataset_dir, cycle_name, image_metadata, camera_roles)
    settings = registry.get("image_coverage", {})
    max_gap = float(
        settings.get("max_image_gap_seconds", 40.0) if isinstance(settings, Mapping) else 40.0
    )
    intervals: dict[str, dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]] = {}
    if not images.empty:
        for role, group in images.groupby("camera_role", sort=True):
            role_intervals = build_rgb_coverage_intervals(
                start,
                end,
                group["image_time"],
                max_image_gap_seconds=max_gap,
            )
            intervals[str(role)] = role_intervals
    return {"image_count": int(len(images))}, intervals


def _sensor_coverage_intervals(  # noqa: C901
    frame: pd.DataFrame,
    registry: Mapping[str, Any],
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Build sensor availability from the same Processed rows used for drawing."""
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    valid = timestamps.notna()
    ordered = timestamps.loc[valid].sort_values(kind="stable")
    if ordered.empty:
        return {"available": [], "missing": []}

    diffs = ordered.diff().dropna().dt.total_seconds()
    positive = diffs.loc[diffs > 0]
    step = float(positive.median()) if not positive.empty else 10.0
    channel_settings = registry.get("channels", {})
    required_names = (
        [
            str(name)
            for name, settings in channel_settings.items()
            if isinstance(settings, Mapping) and bool(settings.get("coverage_required", False))
        ]
        if isinstance(channel_settings, Mapping)
        else []
    )
    observed_names = required_names
    if not observed_names:
        observed_names = [
            str(name)
            for name in registry.get("columns", [])
            if str(name) in frame
            and str(name) not in {"timestamp", "cycle_stage"}
            and not str(name).endswith("__imputed")
        ]

    availability = pd.Series(True, index=frame.index, dtype=bool)
    for name in observed_names:
        if name not in frame:
            availability &= False
            continue
        values = pd.to_numeric(frame[name], errors="coerce").notna()
        imputed = frame.get(f"{name}__imputed")
        if imputed is not None:
            values &= ~imputed.fillna(False).astype(bool)
        availability &= values

    available_rows = frame.loc[valid & availability].sort_values("timestamp", kind="stable")
    start = pd.Timestamp(ordered.iloc[0])
    end = pd.Timestamp(ordered.iloc[-1]) + pd.Timedelta(seconds=step)
    available: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if not available_rows.empty:
        current_start = pd.Timestamp(available_rows.iloc[0]["timestamp"])
        previous = current_start
        for raw in available_rows["timestamp"].iloc[1:]:
            current = pd.Timestamp(raw)
            if (current - previous).total_seconds() > step * 1.5:
                available.append((current_start, previous + pd.Timedelta(seconds=step)))
                current_start = current
            previous = current
        available.append((current_start, previous + pd.Timedelta(seconds=step)))

    missing: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    for available_start, available_end in available:
        if cursor < available_start:
            missing.append((cursor, available_start))
        cursor = max(cursor, available_end)
    if cursor < end:
        missing.append((cursor, end))
    return {"available": available, "missing": missing}


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
        max(0.0, (end - start).total_seconds()) for start, end in intervals.get("available", [])
    )
    return 0.0 if total <= 0 else min(1.0, max(0.0, available / total))


def _merge_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ordered = sorted(
        (pd.Timestamp(start), pd.Timestamp(end)) for start, end in intervals if end > start
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
