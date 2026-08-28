"""Per-role image matching from camera folder names."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from .alignment import match_nearest_one_to_one

_TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{17})(?!\d)")


def image_roles(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return image roles represented by complete path-column names."""
    prefix = "image_"
    suffix = "_path"
    roles = {
        str(column)[len(prefix) : -len(suffix)]
        for column in frame.columns
        if str(column).startswith(prefix) and str(column).endswith(suffix)
    }
    return tuple(sorted(roles))


def image_columns(role: str) -> tuple[str, str, str]:
    """Return the path, timestamp, and offset columns for one image role."""
    return (
        f"image_{role}_path",
        f"image_{role}_time",
        f"image_{role}_offset_seconds",
    )


def match_images(
    timestamps: Iterable[pd.Timestamp],
    image_files: Iterable[Path],
    *,
    tolerance_seconds: float = 2.0,
) -> pd.DataFrame:
    """Match each image at most once within its own camera role."""
    if tolerance_seconds < 0:
        raise ValueError("image tolerance must be nonnegative")
    files = sorted(image_files)
    sensor_times = pd.to_datetime(pd.Series(list(timestamps)), errors="coerce")
    roles = sorted({path.parent.name for path in files})
    records_by_role: dict[str, list[tuple[Path, pd.Timestamp]]] = {role: [] for role in roles}
    for path in files:
        image_time = _image_timestamp(path)
        if image_time is None:
            continue
        records_by_role[path.parent.name].append((path, image_time))

    columns: dict[str, list[Any]] = {}
    for role in roles:
        records = sorted(records_by_role[role], key=lambda item: (item[1], str(item[0])))
        paths: list[object] = [pd.NA] * len(sensor_times)
        image_times_out: list[object] = [pd.NaT] * len(sensor_times)
        offsets: list[object] = [float("nan")] * len(sensor_times)
        pairs = match_nearest_one_to_one(
            sensor_times,
            pd.Series([record[1] for record in records]),
            pd.to_timedelta(tolerance_seconds, unit="s"),
        )
        for sensor_position, image_position in pairs:
            path, image_time = records[image_position]
            sensor_time = sensor_times.iloc[sensor_position]
            paths[sensor_position] = str(path)
            image_times_out[sensor_position] = image_time
            offsets[sensor_position] = (image_time - sensor_time).total_seconds()
        columns[f"image_{role}_path"] = paths
        columns[f"image_{role}_time"] = image_times_out
        columns[f"image_{role}_offset_seconds"] = offsets
    return pd.DataFrame(columns, index=sensor_times.index)


def _image_timestamp(path: Path) -> pd.Timestamp | None:
    match = _TIMESTAMP_RE.search(path.stem)
    if match is None:
        return None
    value = pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S%f", errors="coerce")
    return None if pd.isna(value) else pd.Timestamp(value)
