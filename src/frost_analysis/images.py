"""Explicit camera-role loading and per-role image matching."""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

_TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{17})(?!\d)")


def validate_camera_directories(
    image_files: Iterable[Path], camera_roles: Mapping[str, str]
) -> tuple[str, ...]:
    """Reject image directories absent from the explicit date mapping."""
    discovered = sorted({path.parent.name for path in image_files})
    unknown = tuple(camera for camera in discovered if camera not in camera_roles)
    if unknown:
        raise ValueError(f"unmapped camera directories: {list(unknown)}")
    return tuple(discovered)


def match_images(
    timestamps: Iterable[pd.Timestamp],
    image_files: Iterable[Path],
    *,
    camera_roles: Mapping[str, str],
    tolerance_seconds: float = 2.0,
) -> pd.DataFrame:
    """Match each image at most once within its own camera role."""
    if tolerance_seconds < 0:
        raise ValueError("image tolerance must be nonnegative")
    files = sorted(image_files)
    validate_camera_directories(files, camera_roles)
    sensor_times = pd.to_datetime(pd.Series(list(timestamps)), errors="coerce")
    roles = sorted(set(camera_roles.values()))
    records_by_role: dict[str, list[tuple[Path, pd.Timestamp]]] = {role: [] for role in roles}
    for path in files:
        image_time = _image_timestamp(path)
        if image_time is None:
            continue
        records_by_role[camera_roles[path.parent.name]].append((path, image_time))

    columns: dict[str, list[Any]] = {}
    for role in roles:
        records = sorted(records_by_role[role], key=lambda item: (item[1], str(item[0])))
        image_times = [record[1] for record in records]
        used: set[Path] = set()
        paths: list[object] = []
        image_times_out: list[object] = []
        offsets: list[object] = []
        for timestamp in sensor_times:
            selected = _select_image(timestamp, records, image_times, used, tolerance_seconds)
            if selected is None:
                paths.append(pd.NA)
                image_times_out.append(pd.NaT)
                offsets.append(float("nan"))
                continue
            path, image_time = selected
            paths.append(str(path))
            image_times_out.append(image_time)
            offsets.append((image_time - timestamp).total_seconds())
        columns[f"image_{role}_path"] = paths
        columns[f"image_{role}_time"] = image_times_out
        columns[f"image_{role}_offset_seconds"] = offsets
    return pd.DataFrame(columns, index=sensor_times.index)


def _select_image(
    timestamp: pd.Timestamp,
    records: list[tuple[Path, pd.Timestamp]],
    image_times: list[pd.Timestamp],
    used: set[Path],
    tolerance_seconds: float,
) -> tuple[Path, pd.Timestamp] | None:
    if pd.isna(timestamp):
        return None
    lower = timestamp - pd.Timedelta(seconds=tolerance_seconds)
    upper = timestamp + pd.Timedelta(seconds=tolerance_seconds)
    left = bisect_left(image_times, lower)
    right = bisect_right(image_times, upper)
    candidates = [record for record in records[left:right] if record[0] not in used]
    if not candidates:
        return None
    selected = min(
        candidates,
        key=lambda record: (abs((record[1] - timestamp).total_seconds()), str(record[0])),
    )
    used.add(selected[0])
    return selected


def _image_timestamp(path: Path) -> pd.Timestamp | None:
    match = _TIMESTAMP_RE.search(path.stem)
    if match is None:
        return None
    value = pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S%f", errors="coerce")
    return None if pd.isna(value) else pd.Timestamp(value)
