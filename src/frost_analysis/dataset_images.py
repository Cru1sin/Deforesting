"""Collect and scan Dataset images and compute merged RGB coverage intervals."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, cast
from zipfile import ZipFile

import pandas as pd

from .images import _image_timestamp

DEFAULT_CLOUD_IMAGES_ROOT = Path(
    "/Users/cruisin/Library/CloudStorage/OneDrive-Personal/"
    "HKUST/Project/Defrost/dataset/images"
)
DEFAULT_CLOUD_IMAGES_REMOTE = "onedrive_hkust:HKUST/Project/Defrost/dataset/images"
MIN_FREE_BYTES = 50 * 1024**3


def _require_free_space(path: Path, required: int, action: str) -> None:
    if required + MIN_FREE_BYTES > shutil.disk_usage(path).free:
        raise OSError(
            f"{action} would cross the 50 GiB safety floor; "
            "review local cycle images before deleting anything"
        )


@contextmanager
def materialize_cycle_images(  # noqa: C901
    dataset_dir: Path,
    cycle_name: str,
    *,
    fetch_cloud: bool = False,
    cloud_root: Path | None = None,
) -> Iterator[Path]:
    """Copy one exact cloud ZIP into the local Dataset image tree and retain it."""
    if not cycle_name.startswith("frost_cycle_") or not cycle_name[12:].isdigit():
        raise ValueError(f"invalid cycle name: {cycle_name}")
    images_root = Path(dataset_dir).resolve() / "images"
    cycle_dir = images_root / cycle_name
    if cycle_dir.is_dir() or not fetch_cloud:
        yield cycle_dir
        return

    default_cloud = cloud_root is None
    archive = Path(cloud_root or DEFAULT_CLOUD_IMAGES_ROOT) / f"{cycle_name}.zip"
    if not archive.is_file():
        print(f"[images] no local directory or cloud ZIP: {cycle_name}", flush=True)
        yield cycle_dir
        return

    images_root.mkdir(parents=True, exist_ok=True)
    _require_free_space(images_root, archive.stat().st_size, f"downloading {archive.name}")

    with TemporaryDirectory(prefix=f".{cycle_name}-", dir=images_root) as temporary:
        work = Path(temporary)
        local_archive = work / archive.name
        print(f"[images] copying cloud ZIP: {archive.name}", flush=True)
        if default_cloud:
            subprocess.run(
                [
                    "rclone",
                    "copyto",
                    f"{DEFAULT_CLOUD_IMAGES_REMOTE}/{archive.name}",
                    str(local_archive),
                    "--progress",
                    "--stats",
                    "1s",
                    "--multi-thread-streams",
                    "8",
                    "--multi-thread-cutoff",
                    "256M",
                    "--timeout",
                    "2m",
                    "--retries",
                    "10",
                    "--retries-sleep",
                    "10s",
                ],
                check=True,
            )
        else:
            shutil.copyfile(archive, local_archive)
        with ZipFile(local_archive) as bundle:
            members = bundle.infolist()
            for member in members:
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or path.parts[:1] != (cycle_name,):
                    raise ValueError(f"unsafe cycle ZIP member: {member.filename}")
            required = sum(member.file_size for member in members)
            _require_free_space(images_root, required, f"extracting {archive.name}")
            print(f"[images] extracting local copy: {cycle_name}", flush=True)
            bundle.extractall(work)

        extracted = work / cycle_name
        if not extracted.is_dir():
            raise ValueError(f"cycle ZIP has no {cycle_name} directory")
        if cycle_dir.exists():
            yield cycle_dir
            return
        extracted.replace(cycle_dir)
        print(f"[images] retained local copy: {cycle_name}", flush=True)
        yield cycle_dir


def collect_cycle_images(  # noqa: C901
    image_files: list[Path],
    *,
    input_dir: Path,
    cycles: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Assign every parseable image in a half-open cycle time window."""
    windows = [
        (
            cycle,
            pd.to_datetime(cycle.get("start_time"), errors="coerce"),
            pd.to_datetime(cycle.get("end_time"), errors="coerce"),
            pd.to_datetime(cycle.get("stable_heating_start"), errors="coerce"),
            pd.to_datetime(cycle.get("defrost_preparation_start"), errors="coerce"),
            pd.to_datetime(cycle.get("defrost_start"), errors="coerce"),
            pd.to_datetime(cycle.get("defrost_end"), errors="coerce"),
        )
        for cycle in cycles
    ]
    records: list[dict[str, object]] = []
    for path in image_files:
        source_path = path if path.is_absolute() else input_dir / path
        image_time = _image_timestamp(source_path)
        if image_time is None:
            continue
        for cycle, start, end, stable, preparation_start, defrost_start, defrost_end in windows:
            if pd.isna(start) or pd.isna(end) or not (start <= image_time < end):
                continue
            stage = "partial"
            if not pd.isna(defrost_start) and image_time >= defrost_start:
                if pd.isna(defrost_end) or image_time < defrost_end:
                    stage = "defrost"
            elif not pd.isna(preparation_start) and image_time >= preparation_start:
                stage = "defrost_preparation"
            elif not pd.isna(stable):
                stage = "recovery" if image_time < stable else "frost_development"
            elif not pd.isna(defrost_start):
                stage = "recovery"
            records.append(
                {
                    "cycle_name": str(cycle["cycle_name"]),
                    "camera_role": source_path.parent.name,
                    "file_name": source_path.name,
                    "image_time": image_time,
                    "cycle_stage": stage,
                    "source_path": source_path,
                    "image_path": (
                        f"images/{cycle['cycle_name']}/{source_path.parent.name}/"
                        f"{source_path.name}"
                    ),
                }
            )
            break

    records.sort(
        key=lambda item: (
            str(item["cycle_name"]),
            str(item["camera_role"]),
            pd.Timestamp(cast(Any, item["image_time"])),
            str(item["file_name"]),
        )
    )
    seen: set[tuple[str, str, str]] = set()
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (
            str(record["cycle_name"]),
            str(record["camera_role"]),
            str(record["file_name"]),
        )
        if key in seen:
            raise ValueError(
                "duplicate source basename within camera: " + "/".join(key)
            )
        seen.add(key)
        group = key[:2]
        counts[group] = counts.get(group, 0) + 1
        record["frame_index"] = counts[group]
    return records


def image_metadata_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    """Build the final metadata table without current role or image SHA."""
    columns = [
        "cycle_name",
        "camera_role",
        "file_name",
        "frame_index",
        "image_time",
        "cycle_stage",
    ]
    return pd.DataFrame(
        [{column: record[column] for column in columns} for record in records],
        columns=columns,
    )


def copy_image(record: Mapping[str, object], dataset_dir: Path) -> None:
    """Copy one cycle-owned source image into its cycle/camera directory."""
    from .dataset_metadata import image_root

    source = Path(str(record["source_path"]))
    target = image_root(dataset_dir) / Path(str(record["image_path"])).relative_to("images")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def scan_cycle_images(
    dataset_root: Path,
    cycle_name: str,
    image_metadata: pd.DataFrame,
    *,
    cycle_dir: Path | None = None,
) -> pd.DataFrame:
    """Join Dataset image files to their metadata."""
    from .dataset_metadata import image_root

    columns = [
        "cycle_name",
        "camera_role",
        "file_name",
        "path",
        "frame_index",
        "image_time",
        "cycle_stage",
    ]
    root = Path(cycle_dir) if cycle_dir is not None else image_root(dataset_root) / cycle_name
    if not root.is_dir():
        return pd.DataFrame(columns=columns)
    scoped = image_metadata.loc[image_metadata["cycle_name"].astype(str).eq(cycle_name)].copy()
    key_columns = ["cycle_name", "camera_role", "file_name"]
    if scoped.duplicated(key_columns).any():
        raise ValueError(f"image metadata has duplicate source/file key: {cycle_name}")
    lookup = {
        tuple(str(row[column]) for column in key_columns): row
        for row in scoped.to_dict(orient="records")
    }
    rows: list[dict[str, object]] = []
    for camera_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        camera_role = camera_dir.name
        for image_path in sorted(path for path in camera_dir.iterdir() if path.is_file()):
            key = (cycle_name, camera_role, image_path.name)
            metadata = lookup.get(key)
            if metadata is None:
                continue
            row = {str(key): value for key, value in metadata.items()}
            row.update(
                {
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
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]],
]:
    start, end = _cycle_window(frame)
    images = scan_cycle_images(dataset_dir, cycle_name, image_metadata)
    intervals = image_coverage_intervals(frame, images, registry)
    return {"image_count": int(len(images))}, intervals


def image_coverage_intervals(
    frame: pd.DataFrame,
    images: pd.DataFrame,
    registry: Mapping[str, Any],
) -> dict[str, dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]]:
    """Compute per-role intervals from already-selected image facts."""
    start, end = _cycle_window(frame)
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
    return intervals


def rgb_stage_metrics(
    frame: pd.DataFrame,
    intervals: Mapping[str, Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]]],
    expected_roles: tuple[str, ...],
) -> dict[str, object]:
    """Compute cycle-level RGB coverage from the intersection of all roles."""
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    stages = frame.get("cycle_stage", pd.Series(index=frame.index, dtype="string")).astype(
        "string"
    )
    result: dict[str, object] = {}
    for output_name, stage_name in (
        ("frost", "frost_development"),
        ("defrost", "defrost"),
    ):
        stage_times = timestamps.loc[stages.eq(stage_name) & timestamps.notna()]
        if stage_times.empty:
            coverage: float | None = None
            status = "not_applicable"
        else:
            available = pd.Series(True, index=stage_times.index)
            for role in expected_roles:
                role_available = pd.Series(False, index=stage_times.index)
                for start, end in intervals.get(role, {}).get("available", []):
                    role_available |= stage_times.ge(start) & stage_times.lt(end)
                available &= role_available
            if not expected_roles:
                available &= False
            coverage = float(available.mean())
            status = "valid" if coverage >= 0.8 else "invalid"
        result[f"rgb_{output_name}_coverage"] = coverage
        result[f"rgb_{output_name}_auto_status"] = status
    return result


def rgb_overall_intervals(
    frame: pd.DataFrame,
    intervals: Mapping[str, Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]]],
    expected_roles: tuple[str, ...],
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Return continuous availability where every expected camera is present."""
    start, end = _cycle_window(frame)
    available = [(start, end)] if expected_roles else []
    for role in expected_roles:
        available = _merge_intervals(
            [
                (max(left, role_left), min(right, role_right))
                for left, right in available
                for role_left, role_right in intervals.get(role, {}).get("available", [])
                if min(right, role_right) > max(left, role_left)
            ]
        )
    missing: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    for left, right in available:
        if cursor < left:
            missing.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        missing.append((cursor, end))
    return {"available": available, "missing": missing}


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
