"""Narrow image-to-sensor matching and non-reusing multiview grouping."""

from __future__ import annotations

import hashlib
import re
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


def attach_image_paths(prepared: pd.DataFrame, alignment: pd.DataFrame) -> pd.DataFrame:
    """Merge one nearest matched image path per timestamp and camera role."""
    # These columns are the contract produced by match_images_to_sensors().
    required_columns = {
        "matched",
        "camera_role",
        "timestamp",
        "image_path",
        "time_delta_s",
    }
    missing_columns = required_columns - set(alignment.columns)
    if missing_columns:
        raise ValueError(f"Image alignment is missing columns: {sorted(missing_columns)}")

    # Unmatched image candidates must not create empty wide-table columns.
    matched_images = alignment.loc[
        alignment["matched"].fillna(False).astype(bool)
    ].copy()
    if matched_images.empty:
        return prepared.copy()
    if matched_images["camera_role"].isna().any():
        raise ValueError("Image alignment contains missing camera_role")

    # Keep the closest image when several files map to one sensor row and role.
    matched_images["camera_role"] = matched_images["camera_role"].astype(str).str.strip()
    matched_images["absolute_time_delta"] = pd.to_numeric(
        matched_images["time_delta_s"], errors="coerce"
    ).abs()
    matched_images = matched_images.sort_values(
        ["absolute_time_delta", "camera_role", "image_path"],
        kind="stable",
        na_position="last",
    ).drop_duplicates(["timestamp", "camera_role"], keep="first")

    result = prepared.copy()  # 传感器行是主键，图片只补充路径和时间偏移。
    for camera_role, group in matched_images.groupby("camera_role", sort=True):
        role_key = _camera_role_key(camera_role)
        # camera_role, rather than camera_id, keeps output names stable across days.
        image_columns = group[["timestamp", "image_path", "time_delta_s"]].rename(
            columns={
                "image_path": f"image_{role_key}_path",
                "time_delta_s": f"image_{role_key}_offset_seconds",
            }
        )
        result = result.merge(
            image_columns,
            on="timestamp",
            how="left",
            validate="one_to_one",
        )
    return result


def _camera_role_key(camera_role: object) -> str:
    """Turn a configured role into a stable output suffix, independent of IP."""
    role_text = str(camera_role).strip()
    safe_role = re.sub(r"[^A-Za-z0-9]+", "_", role_text).strip("_").lower()
    if safe_role:
        return safe_role
    digest = hashlib.sha1(role_text.encode("utf-8")).hexdigest()[:8]
    return f"role_{digest}"


class _MultiviewGroup(TypedDict):
    seed_time: pd.Timestamp
    cameras: dict[str, _ImageRecord]
    members: list[_ImageRecord]
    active: bool


def match_images_to_sensors(
    image_frame: pd.DataFrame, sensor_frame: pd.DataFrame, *, tolerance_s: float
) -> pd.DataFrame:
    """Match image times to canonical ``timestamp`` rows without copying data."""
    if tolerance_s < 0:
        raise ValueError("sensor matching tolerance must be non-negative")
    images = image_frame.reset_index(drop=True)
    matching = pd.DataFrame(
        {
            "candidate_timestamp": pd.Series(pd.NaT, index=images.index, dtype="datetime64[ns]"),
            "time_delta_s": np.nan,
            "matched": False,
            "timestamp": pd.Series(pd.NaT, index=images.index, dtype="datetime64[ns]"),
        }
    )
    result = pd.concat([images, matching], axis=1)
    if sensor_frame.empty or result.empty:
        return result
    if "timestamp" not in sensor_frame:
        raise ValueError("sensor frame must contain timestamp")
    sensors = (
        sensor_frame[["timestamp"]]
        .dropna()
        .drop_duplicates()
        .sort_values("timestamp", ignore_index=True)
    )
    if sensors.empty:
        return result
    sensor_times = pd.to_datetime(sensors["timestamp"]).astype("int64").to_numpy()
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
    candidate_times = sensors.iloc[chosen]["timestamp"].to_numpy()
    result.loc[valid_indices, "candidate_timestamp"] = candidate_times
    result.loc[valid_indices, "time_delta_s"] = delta_s
    matched_indices = valid_indices[within_tolerance]
    result.loc[matched_indices, "matched"] = True
    result.loc[matched_indices, "timestamp"] = candidate_times[within_tolerance]
    return result


def attach_cycle_labels(alignment: pd.DataFrame, sensor_frame: pd.DataFrame) -> pd.DataFrame:
    """Attach cycle labels to matched rows using the canonical timestamp key."""
    if "timestamp" not in alignment or "timestamp" not in sensor_frame:
        raise ValueError("alignment and sensor frames must contain timestamp")
    label_columns = [column for column in _LABEL_COLUMNS if column in sensor_frame]
    conflict_columns = [
        column for column in sensor_frame if str(column).endswith("__duplicate_conflict")
    ]
    labels = sensor_frame[["timestamp", *label_columns]].copy()
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
        labels.drop_duplicates("timestamp"), how="left", on="timestamp", validate="many_to_one"
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
