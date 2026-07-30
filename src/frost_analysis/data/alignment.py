"""Narrow image-to-sensor matching and non-reusing multiview grouping."""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd

_ImageRecord = dict[str, Any]
_LABEL_COLUMNS = ["cycle_id", "cycle_quality", "stage", "cycle_time_s", "cycle_phase"]
_EMPTY_MULTIVIEW_COLUMNS = [
    "group_id",
    "group_time",
    "camera_count",
    "all_cameras_present",
]


class _MultiviewGroup(TypedDict):
    seed_time: pd.Timestamp
    cameras: dict[str, _ImageRecord]
    members: list[_ImageRecord]
    active: bool


def match_images_to_sensors(
    image_frame: pd.DataFrame, sensor_frame: pd.DataFrame, *, tolerance_s: float
) -> pd.DataFrame:
    """Match nearest sensor times, preferring the earlier sample on exact ties."""
    if tolerance_s < 0:
        raise ValueError("sensor matching tolerance must be non-negative")
    images = image_frame.reset_index(drop=True)
    matching = pd.DataFrame(
        {
            "candidate_sensor_time": pd.Series(pd.NaT, index=images.index, dtype="datetime64[ns]"),
            "time_delta_s": np.nan,
            "matched": False,
            "sensor_time": pd.Series(pd.NaT, index=images.index, dtype="datetime64[ns]"),
        }
    )
    result = pd.concat([images, matching], axis=1)
    if sensor_frame.empty or result.empty:
        return result
    if "sensor_time" not in sensor_frame:
        raise ValueError("sensor frame must contain sensor_time")
    sensors = (
        sensor_frame[["sensor_time"]]
        .dropna()
        .drop_duplicates()
        .sort_values("sensor_time", ignore_index=True)
    )
    if sensors.empty:
        return result
    sensor_times = pd.to_datetime(sensors["sensor_time"]).astype("int64").to_numpy()
    valid_mask = result["image_time"].notna().to_numpy()
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        return result
    image_times = pd.to_datetime(result.loc[valid_mask, "image_time"]).astype("int64").to_numpy()
    insertion = np.searchsorted(sensor_times, image_times, side="left")
    left = np.clip(insertion - 1, 0, len(sensor_times) - 1)
    right = np.clip(insertion, 0, len(sensor_times) - 1)
    left_distance = np.abs(sensor_times[left] - image_times)
    right_distance = np.abs(sensor_times[right] - image_times)
    left_distance[insertion == 0] = np.iinfo(np.int64).max
    right_distance[insertion == len(sensor_times)] = np.iinfo(np.int64).max
    chosen = np.where(left_distance <= right_distance, left, right)
    delta_s = (sensor_times[chosen] - image_times) / 1_000_000_000
    within_tolerance = np.abs(delta_s) <= tolerance_s
    candidate_times = sensors.iloc[chosen]["sensor_time"].to_numpy()
    result.loc[valid_indices, "candidate_sensor_time"] = candidate_times
    result.loc[valid_indices, "time_delta_s"] = delta_s
    matched_indices = valid_indices[within_tolerance]
    result.loc[matched_indices, "matched"] = True
    result.loc[matched_indices, "sensor_time"] = candidate_times[within_tolerance]
    return result


def attach_cycle_labels(alignment: pd.DataFrame, sensor_frame: pd.DataFrame) -> pd.DataFrame:
    """Attach only cycle labels and aggregate sensor-quality evidence to matches."""
    if "sensor_time" not in alignment or "sensor_time" not in sensor_frame:
        raise ValueError("alignment and sensor frames must contain sensor_time")
    label_columns = [column for column in _LABEL_COLUMNS if column in sensor_frame]
    conflict_columns = [
        column for column in sensor_frame if str(column).endswith("__duplicate_conflict")
    ]
    labels = sensor_frame[["sensor_time", *label_columns]].copy()
    if conflict_columns:
        labels["sensor_duplicate_conflict"] = (
            sensor_frame[conflict_columns].astype("boolean").fillna(False).any(axis=1)
        )
    else:
        labels["sensor_duplicate_conflict"] = False
    labels["sensor_quality_flag"] = np.where(
        labels["sensor_duplicate_conflict"], "duplicate_conflict", "ok"
    )
    drop_existing = [
        column
        for column in [*_LABEL_COLUMNS, "sensor_duplicate_conflict", "sensor_quality_flag"]
        if column in alignment
    ]
    result = alignment.drop(columns=drop_existing).merge(
        labels.drop_duplicates("sensor_time"), how="left", on="sensor_time", validate="many_to_one"
    )
    unmatched = ~result.get("matched", pd.Series(False, index=result.index)).fillna(False)
    for column in [*label_columns, "sensor_duplicate_conflict", "sensor_quality_flag"]:
        result.loc[unmatched, column] = pd.NA
    return result


def build_multiview(image_frame: pd.DataFrame, *, tolerance_ms: float) -> pd.DataFrame:
    """Group images once, with at most one image per camera in each group."""
    if tolerance_ms < 0:
        raise ValueError("multiview tolerance must be non-negative")
    valid = image_frame.loc[image_frame["image_time"].notna()].sort_values(
        ["image_time", "camera_id", "sample_id"], kind="stable"
    )
    if valid.empty:
        return pd.DataFrame(columns=_EMPTY_MULTIVIEW_COLUMNS)
    tolerance = pd.Timedelta(milliseconds=tolerance_ms)
    camera_ids = sorted(valid["camera_id"].astype(str).unique())
    completed: list[_MultiviewGroup] = []
    active: deque[_MultiviewGroup] = deque()
    available = {camera_id: deque[_MultiviewGroup]() for camera_id in camera_ids}
    for image in cast(list[_ImageRecord], valid.to_dict(orient="records")):
        image_time = pd.Timestamp(image["image_time"])
        while active and image_time - active[0]["seed_time"] > tolerance:
            expired = active.popleft()
            expired["active"] = False
            completed.append(expired)
        camera_id = str(image["camera_id"])
        queue = available[camera_id]
        while queue and not queue[0]["active"]:
            queue.popleft()
        if queue:
            target = queue.popleft()
        else:
            target = _MultiviewGroup(seed_time=image_time, cameras={}, members=[], active=True)
            active.append(target)
            for other in camera_ids:
                if other != camera_id:
                    available[other].append(target)
        target["cameras"][camera_id] = image
        target["members"].append(image)
    for group in active:
        group["active"] = False
        completed.append(group)
    rows = [_multiview_row(group, camera_ids) for group in completed]
    return pd.DataFrame.from_records(rows).sort_values("group_time", ignore_index=True)


def _multiview_row(group: _MultiviewGroup, camera_ids: list[str]) -> dict[str, object]:
    members = group["members"]
    times = np.sort(
        np.array([pd.Timestamp(member["image_time"]).value for member in members], dtype=np.int64)
    )
    middle = len(times) // 2
    median_ns = int(times[middle])
    if len(times) % 2 == 0:
        lower = int(times[middle - 1])
        median_ns = lower + (median_ns - lower) // 2
    group_time = pd.Timestamp(median_ns, unit="ns")
    sample_ids = sorted(str(member["sample_id"]) for member in members)
    digest = hashlib.sha1("|".join(sample_ids).encode()).hexdigest()[:8]
    row: dict[str, object] = {
        "group_id": f"mv_{group['seed_time'].strftime('%Y%m%d%H%M%S%f')[:17]}_{digest}",
        "group_time": group_time,
        "camera_count": len(members),
        "all_cameras_present": len(members) == len(camera_ids),
    }
    if members and "experiment_id" in members[0]:
        row["experiment_id"] = members[0]["experiment_id"]
    for camera_id in camera_ids:
        member = group["cameras"].get(camera_id)
        row[f"{camera_id}__sample_id"] = member.get("sample_id") if member else None
        row[f"{camera_id}__image_path"] = member.get("image_path") if member else None
        row[f"{camera_id}__image_time"] = member.get("image_time") if member else pd.NaT
        row[f"{camera_id}__delta_s"] = (
            (pd.Timestamp(member["image_time"]).value - median_ns) / 1_000_000_000
            if member
            else np.nan
        )
        row[f"{camera_id}__image_ok"] = member.get("image_ok") if member else None
        row[f"{camera_id}__camera_role"] = member.get("camera_role") if member else None
    return row
