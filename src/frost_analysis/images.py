"""Filename-based image matching performed only during Prepare."""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

_TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{17})(?!\d)")


def match_images(
    timestamps: Iterable[pd.Timestamp],
    image_files: Iterable[Path],
    *,
    tolerance_seconds: float = 2.0,
) -> pd.DataFrame:
    """Match each image at most once to the nearest sensor timestamp."""
    sensor_times = pd.to_datetime(pd.Series(list(timestamps)), errors="coerce")
    records = [
        (path, _image_timestamp(path), path.parent.name)
        for path in sorted(image_files)
    ]
    usable: list[tuple[Path, pd.Timestamp, str]] = []
    for path, image_time, camera_id in records:
        if image_time is not None:
            usable.append((path, image_time, camera_id))
    usable.sort(key=lambda record: (record[1], str(record[0])))
    image_times = [record[1] for record in usable]
    used: set[Path] = set()
    rows: list[dict[str, object]] = []
    for timestamp in sensor_times:
        if pd.isna(timestamp):
            rows.append({"image_path": pd.NA, "image_time": pd.NaT, "image_camera_id": pd.NA})
            continue
        lower = timestamp - pd.Timedelta(seconds=tolerance_seconds)
        upper = timestamp + pd.Timedelta(seconds=tolerance_seconds)
        left = bisect_left(image_times, lower)
        right = bisect_right(image_times, upper)
        candidates = [record for record in usable[left:right] if record[0] not in used]
        if not candidates:
            rows.append({"image_path": pd.NA, "image_time": pd.NaT, "image_camera_id": pd.NA})
            continue
        selected = min(
            candidates,
            key=lambda record: (abs((record[1] - timestamp).total_seconds()), str(record[0])),
        )
        used.add(selected[0])
        rows.append(
            {
                "image_path": str(selected[0]),
                "image_time": selected[1],
                "image_camera_id": selected[2],
            }
        )
    return pd.DataFrame(rows, index=sensor_times.index)


def _image_timestamp(path: Path) -> pd.Timestamp | None:
    match = _TIMESTAMP_RE.search(path.stem)
    if match is None:
        return None
    value = pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S%f", errors="coerce")
    return None if pd.isna(value) else pd.Timestamp(value)
