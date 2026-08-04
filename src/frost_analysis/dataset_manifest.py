"""Creation and refresh of the single-assessment Dataset manifest."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import DATASET_V2_SCHEMA_VERSION
from .dataset_coverage import image_summary
from .dataset_images import scan_cycle_images
from .dataset_io import write_atomic_json, write_atomic_parquet
from .io import sha256_file

ASSESSMENT_STATUSES = {"valid", "partial", "incomplete", "invalid"}


def make_assessment(
    status: str,
    reasons: list[str] | None = None,
    note: str | None = None,
    *,
    updated_at: str | None = None,
) -> dict[str, object]:
    """Create the one editable assessment object stored for a cycle."""
    if status not in ASSESSMENT_STATUSES:
        raise ValueError(f"invalid cycle assessment status: {status}")
    return {
        "status": status,
        "reasons": list(reasons or []),
        "note": note,
        "updated_at": updated_at or datetime.now(UTC).isoformat(),
    }


def automatic_assessment(cycle_status: object, reason: object) -> dict[str, object]:
    """Translate the existing scientific cycle status into the editable record."""
    status = str(cycle_status)
    if status not in ASSESSMENT_STATUSES:
        status = "invalid"
    missing_reason = reason is None or (isinstance(reason, float) and pd.isna(reason))
    reasons = [] if missing_reason or not str(reason) else [str(reason)]
    return make_assessment(status, reasons)


def build_manifest(
    *,
    dataset_id: str,
    source_schema: list[dict[str, Any]],
    source_runs: list[dict[str, object]],
    cycles: list[dict[str, object]],
    cycle_index: pd.DataFrame,
    image_metadata: pd.DataFrame,
    created_at: str | None = None,
) -> dict[str, object]:
    """Build a v2 manifest from already materialized Dataset facts."""
    return {
        "dataset_schema_version": DATASET_V2_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "source_processed_schema": source_schema,
        "source_runs": source_runs,
        "cycles": cycles,
        "summary_cycle_count": int(len(cycle_index)),
        "image_count": int(len(image_metadata)),
        "cycle_index": {
            "path": "cycle_index.parquet",
            "row_count": int(len(cycle_index)),
        },
        "image_metadata": {
            "path": "image_metadata.parquet",
            "row_count": int(len(image_metadata)),
        },
    }


def write_manifest(dataset_dir: Path, manifest: dict[str, object]) -> None:
    """Atomically write the manifest JSON."""
    write_atomic_json(manifest, dataset_dir / "dataset_manifest.json")


def _review_legacy_cycle(
    dataset_dir: Path,
    cycle_name: str,
    *,
    status: str,
    note: str | None = None,
) -> None:
    """Update only one cycle's assessment, preserving all factual fields."""
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = _read_manifest(manifest_path)
    cycles = manifest.get("cycles")
    if not isinstance(cycles, list):
        raise ValueError("dataset manifest is missing cycles")
    for record in cycles:
        if isinstance(record, dict) and record.get("cycle_name") == cycle_name:
            record["assessment"] = make_assessment(status, note=note)
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            write_manifest(dataset_dir, manifest)
            return
    raise KeyError(f"unknown cycle: {cycle_name}")


def _refresh_legacy_manifest(dataset_dir: Path) -> Path:
    """Refresh factual image/cycle summaries without touching assessments."""
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = _read_manifest(manifest_path)
    cycles = manifest.get("cycles")
    if not isinstance(cycles, list):
        raise ValueError("dataset manifest is missing cycles")
    image_metadata_path = dataset_dir / "image_metadata.parquet"
    cycle_index_path = dataset_dir / "cycle_index.parquet"
    image_metadata = pd.read_parquet(image_metadata_path)
    cycle_index = pd.read_parquet(cycle_index_path)
    for record in cycles:
        if not isinstance(record, dict):
            raise ValueError("dataset manifest contains an invalid cycle record")
        cycle_name = str(record["cycle_name"])
        frame = pd.read_parquet(dataset_dir / "cycles" / f"{cycle_name}.parquet")
        images = scan_cycle_images(dataset_dir, cycle_name, image_metadata)
        record["row_count"] = int(len(frame))
        record["image_summary"] = image_summary(frame, images)
        record["assessment"] = _preserve_assessment(record.get("assessment"), cycle_name)
        mask = cycle_index["cycle_name"].eq(cycle_name)
        cycle_index.loc[mask, "row_count"] = len(frame)
        cycle_index.loc[mask, "image_count"] = len(images)
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    manifest["image_count"] = int(len(image_metadata))
    write_atomic_parquet(cycle_index, cycle_index_path)
    files = manifest.get("files")
    if isinstance(files, dict):
        cycle_file = files.get("cycle_index")
        if isinstance(cycle_file, dict):
            cycle_file["sha256"] = sha256_file(cycle_index_path)
            cycle_file["row_count"] = len(cycle_index)
    write_manifest(dataset_dir, manifest)
    return dataset_dir


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"dataset manifest is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest must be an object")
    return payload


def _preserve_assessment(value: object, cycle_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"cycle assessment is missing: {cycle_name}")
    required = {"status", "reasons", "note", "updated_at"}
    if set(value) != required or str(value["status"]) not in ASSESSMENT_STATUSES:
        raise ValueError(f"cycle assessment is invalid: {cycle_name}")
    if not isinstance(value["reasons"], list):
        raise ValueError(f"cycle assessment reasons must be a list: {cycle_name}")
    return value


def review_cycle(
    dataset_dir: Path,
    cycle_name: str,
    *,
    status: str,
    note: str | None = None,
) -> None:
    """Edit the single canonical cycle status, or legacy assessment if needed."""
    if not _is_canonical_manifest(dataset_dir):
        _review_legacy_cycle(dataset_dir, cycle_name, status=status, note=note)
        return
    if status not in ASSESSMENT_STATUSES:
        raise ValueError(f"invalid cycle status: {status}")
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = _read_manifest(manifest_path)
    record = _find_cycle(manifest, cycle_name)
    record["cycle_status"] = status
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    write_atomic_json(manifest, manifest_path)
    _refresh_canonical_manifest(
        dataset_dir,
        render_publication=True,
        render_coverage=False,
    )


def refresh_manifest(dataset_dir: Path) -> Path:
    """Refresh image facts, coverage, titles and asset hashes without changing status."""
    if not _is_canonical_manifest(dataset_dir):
        return _refresh_legacy_manifest(dataset_dir)
    _refresh_canonical_manifest(dataset_dir, render_publication=True)
    return dataset_dir


def refresh_cycle_asset_hashes(dataset_dir: Path, cycle_name: str) -> None:
    """Refresh one cycle's asset hashes in cycle_index for both contracts."""
    if not _is_canonical_manifest(dataset_dir):
        _legacy_refresh_cycle_asset_hashes(dataset_dir, cycle_name)
        return
    index_path = dataset_dir / "cycle_index.parquet"
    index = pd.read_parquet(index_path)
    mask = index["cycle_name"].astype(str).eq(cycle_name)
    if not mask.any():
        raise KeyError(f"unknown cycle: {cycle_name}")
    for column, relative in (
        ("parquet_sha256", f"cycles/{cycle_name}.parquet"),
        ("csv_sha256", f"cycles/{cycle_name}.csv"),
        ("publication_sha256", f"cycles/{cycle_name}.png"),
        ("rgb_coverage_sha256", f"cycles/{cycle_name}_rgb_coverage.png"),
    ):
        path = dataset_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"cycle asset does not exist: {path}")
        index.loc[mask, column] = sha256_file(path)
    write_atomic_parquet(index, index_path)


def _legacy_refresh_cycle_asset_hashes(dataset_dir: Path, cycle_name: str) -> None:
    """Preserve the original v2 asset-hash update behavior."""
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = _read_manifest(manifest_path)
    cycles = manifest.get("cycles")
    if not isinstance(cycles, list):
        raise ValueError("dataset manifest is missing cycles")
    record = next(
        (
            item
            for item in cycles
            if isinstance(item, dict) and item.get("cycle_name") == cycle_name
        ),
        None,
    )
    if not isinstance(record, dict):
        raise KeyError(f"unknown cycle: {cycle_name}")
    asset_hashes = record.get("asset_sha256")
    if not isinstance(asset_hashes, dict):
        raise ValueError(f"cycle asset hashes are missing: {cycle_name}")
    asset_paths = {
        "parquet": dataset_dir / "cycles" / f"{cycle_name}.parquet",
        "csv": dataset_dir / "cycles" / f"{cycle_name}.csv",
        "publication": dataset_dir / "cycles" / f"{cycle_name}.png",
        "rgb_coverage": dataset_dir / "cycles" / f"{cycle_name}_rgb_coverage.png",
    }
    for key, path in asset_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"cycle asset does not exist: {path}")
        asset_hashes[key] = sha256_file(path)
    record["asset_sha256"] = asset_hashes
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    write_manifest(dataset_dir, manifest)


def _is_canonical_manifest(dataset_dir: Path) -> bool:
    path = dataset_dir / "dataset_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = _read_manifest(path)
    except Exception:
        return False
    return set(manifest) == {
        "dataset_schema_version",
        "dataset_id",
        "created_at",
        "updated_at",
        "source_experiments",
        "cycles",
    } and manifest.get("dataset_schema_version") == 3


def _find_cycle(manifest: dict[str, Any], cycle_name: str) -> dict[str, Any]:
    cycles = manifest.get("cycles")
    if not isinstance(cycles, list):
        raise ValueError("dataset manifest cycles must be a list")
    for record in cycles:
        if isinstance(record, dict) and record.get("cycle_name") == cycle_name:
            return record
    raise KeyError(f"unknown cycle: {cycle_name}")


def _refresh_canonical_manifest(
    dataset_dir: Path,
    *,
    render_publication: bool,
    render_coverage: bool = True,
) -> None:  # noqa: C901
    from .dataset import (
        _canonical_role_summary,
        _registry_image_gap_seconds,
        _scan_current_cycle_images,
    )
    from .dataset_coverage import render_rgb_coverage
    from .visualization import render_cycle_publication

    manifest = _read_manifest(dataset_dir / "dataset_manifest.json")
    registry = json.loads((dataset_dir / "channel_registry.json").read_text(encoding="utf-8"))
    metadata = pd.read_parquet(dataset_dir / "image_metadata.parquet")
    index = pd.read_parquet(dataset_dir / "cycle_index.parquet")
    cycles = manifest.get("cycles")
    if not isinstance(cycles, list):
        raise ValueError("dataset manifest cycles must be a list")
    for record in cycles:
        if not isinstance(record, dict):
            raise ValueError("dataset manifest contains an invalid cycle record")
        name = str(record["cycle_name"])
        frame = pd.read_parquet(dataset_dir / "cycles" / f"{name}.parquet")
        images = _scan_current_cycle_images(dataset_dir, name, metadata)
        record["image"] = {
            "by_camera_role": _canonical_role_summary(
                frame,
                images,
                max_image_gap_seconds=_registry_image_gap_seconds(registry),
            )
        }
        original_path = dataset_dir / "cycles_original" / f"{name}.csv"
        original = pd.read_csv(original_path)
        original_timestamps = pd.to_datetime(original["timestamp"], errors="coerce").dropna()
        original_intervals = (
            original_timestamps.sort_values().diff().dropna().dt.total_seconds()
        )
        record["original_data"] = {
            "row_count": int(len(original)),
            "start_time": (
                original_timestamps.min().isoformat()
                if not original_timestamps.empty
                else None
            ),
            "end_time": (
                original_timestamps.max().isoformat()
                if not original_timestamps.empty
                else None
            ),
            "median_interval_seconds": (
                float(original_intervals.median())
                if not original_intervals.empty
                else None
            ),
            "sha256": sha256_file(original_path),
        }
        row_mask = index["cycle_name"].astype(str).eq(name)
        index.loc[row_mask, "processed_row_count"] = len(frame)
        index.loc[row_mask, "original_row_count"] = int(record["original_data"]["row_count"])
        if render_publication:
            render_cycle_publication(frame, record, dataset_dir / "cycles" / f"{name}.png")
            if render_coverage:
                render_rgb_coverage(
                    frame,
                    images,
                    record,
                    dataset_dir / "cycles" / f"{name}_rgb_coverage.png",
                    registry=registry,
                )
        for column, relative in (
            ("parquet_sha256", f"cycles/{name}.parquet"),
            ("csv_sha256", f"cycles/{name}.csv"),
            ("publication_sha256", f"cycles/{name}.png"),
            ("rgb_coverage_sha256", f"cycles/{name}_rgb_coverage.png"),
        ):
            index.loc[row_mask, column] = sha256_file(dataset_dir / relative)
    write_atomic_parquet(index, dataset_dir / "cycle_index.parquet")
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    write_atomic_json(manifest, dataset_dir / "dataset_manifest.json")


def edit_dataset(  # noqa: C901
    dataset_dir: Path,
    *,
    baseline_seconds: int | None = None,
    recovery_seconds: int | None = None,
    recovery_end_by: str | None = None,
    statuses: list[str] | None = None,
    camera_renames: list[str] | None = None,
) -> Path:
    """Apply explicit Dataset edits with the smallest necessary recomputation."""
    if not _is_canonical_manifest(dataset_dir):
        raise ValueError("dataset edit requires a canonical schema-3 Dataset")
    if recovery_seconds is not None and recovery_end_by is not None:
        raise ValueError("--recovery-seconds and --recovery-end-by are mutually exclusive")
    if baseline_seconds is not None and baseline_seconds <= 0:
        raise ValueError("baseline_seconds must be positive")
    if recovery_seconds is not None and recovery_seconds < 0:
        raise ValueError("recovery_seconds must be nonnegative")
    renames = _parse_camera_renames(camera_renames or [])
    manifest = _read_manifest(dataset_dir / "dataset_manifest.json")
    records = manifest.get("cycles")
    if not isinstance(records, list):
        raise ValueError("dataset manifest cycles must be a list")
    if statuses:
        for item in statuses:
            cycle_name, status = _parse_cycle_status(item)
            record = _find_cycle(manifest, cycle_name)
            if status not in ASSESSMENT_STATUSES:
                raise ValueError(f"invalid cycle status: {status}")
            record["cycle_status"] = status
    if renames:
        _rename_camera_directories(dataset_dir, renames)
    if recovery_seconds is not None or recovery_end_by is not None:
        _edit_recovery_boundaries(
            dataset_dir,
            records,
            fixed_seconds=recovery_seconds,
            use_ts_minus=recovery_end_by == "ts-minus",
        )
    if baseline_seconds is not None:
        _edit_baseline_windows(dataset_dir, records, baseline_seconds)
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    write_atomic_json(manifest, dataset_dir / "dataset_manifest.json")
    scientific_edit = recovery_seconds is not None or baseline_seconds is not None
    image_edit = bool(renames)
    status_edit = bool(statuses)
    _refresh_canonical_manifest(
        dataset_dir,
        render_publication=scientific_edit or status_edit,
        render_coverage=scientific_edit or image_edit,
    )
    return dataset_dir


def _parse_cycle_status(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("--status must use CYCLE=STATUS")
    cycle, status = value.split("=", 1)
    if not cycle or not status:
        raise ValueError("--status must use CYCLE=STATUS")
    return cycle, status


def _parse_camera_renames(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--rename-camera must use OLD=NEW")
        old, new = value.split("=", 1)
        if not old or not new or "/" in new or "\\" in new or new in {".", ".."}:
            raise ValueError("camera role names must be non-empty safe folder names")
        if old in result:
            raise ValueError(f"camera role is renamed more than once: {old}")
        result[old] = new
    return result


def _rename_camera_directories(dataset_dir: Path, renames: Mapping[str, str]) -> None:
    for cycle_root in (dataset_dir / "images").iterdir():
        if not cycle_root.is_dir():
            continue
        for old, new in renames.items():
            source = cycle_root / old
            target = cycle_root / new
            if not source.exists():
                continue
            if target.exists():
                raise ValueError(f"camera role target already exists: {target}")
            source.rename(target)


def _edit_recovery_boundaries(
    dataset_dir: Path,
    records: list[object],
    *,
    fixed_seconds: int | None,
    use_ts_minus: bool,
) -> None:
    """Update stage labels in both original and 10-s cycle files."""
    for raw_record in records:
        if not isinstance(raw_record, dict):
            continue
        name = str(raw_record["cycle_name"])
        original_path = dataset_dir / "cycles_original" / f"{name}.csv"
        original = pd.read_csv(original_path)
        timestamps = pd.to_datetime(original["timestamp"], errors="coerce")
        if timestamps.empty:
            continue
        heating_start = timestamps.min()
        old_stage = original.get(
            "cycle_stage", pd.Series("partial", index=original.index)
        ).astype(str)
        defrost_mask = old_stage.eq("defrost")
        stable: pd.Timestamp | None
        if fixed_seconds is not None:
            stable = heating_start + pd.Timedelta(seconds=fixed_seconds)
        else:
            stable = _first_ts_minus_crossing(original, timestamps)
        if stable is None:
            original["cycle_stage"] = "partial"
        else:
            original["cycle_stage"] = "partial"
            original.loc[timestamps.lt(stable), "cycle_stage"] = "recovery"
            existing_defrost = timestamps.loc[defrost_mask]
            defrost_start = existing_defrost.min() if not existing_defrost.empty else None
            if defrost_start is None:
                original.loc[timestamps.ge(stable), "cycle_stage"] = "frost_development"
            else:
                original.loc[
                    timestamps.ge(stable) & timestamps.lt(defrost_start), "cycle_stage"
                ] = "frost_development"
                original.loc[defrost_mask, "cycle_stage"] = "defrost"
        original.to_csv(original_path, index=False)
        processed_path = dataset_dir / "cycles" / f"{name}.parquet"
        processed = pd.read_parquet(processed_path)
        processed_times = pd.to_datetime(processed["timestamp"], errors="coerce")
        if stable is None:
            processed["cycle_stage"] = "partial"
        else:
            processed["cycle_stage"] = "partial"
            processed.loc[processed_times.lt(stable), "cycle_stage"] = "recovery"
            processed.loc[processed_times.ge(stable), "cycle_stage"] = "frost_development"
            if defrost_start is not None:
                processed.loc[processed_times.ge(defrost_start), "cycle_stage"] = "defrost"
        processed.to_parquet(processed_path, index=False)
        processed.to_csv(dataset_dir / "cycles" / f"{name}.csv", index=False)


def _first_ts_minus_crossing(
    frame: pd.DataFrame, timestamps: pd.Series
) -> pd.Timestamp | None:
    if "water_out_temperature" not in frame or "water_temperature_setpoint" not in frame:
        return None
    water_out = pd.to_numeric(frame["water_out_temperature"], errors="coerce")
    setpoint = pd.to_numeric(frame["water_temperature_setpoint"], errors="coerce")
    valid = timestamps.notna() & water_out.notna() & setpoint.notna()
    paired = pd.DataFrame(
        {"timestamp": timestamps, "water_out": water_out, "setpoint": setpoint}
    ).loc[valid].sort_values("timestamp", kind="stable")
    for offset in (2.0, 3.0, 4.0):
        crossing = paired.loc[paired["water_out"].ge(paired["setpoint"] - offset), "timestamp"]
        if not crossing.empty:
            return pd.Timestamp(crossing.iloc[0])
    return None


def _edit_baseline_windows(  # noqa: C901
    dataset_dir: Path, records: list[object], baseline_seconds: int
) -> None:
    for raw_record in records:
        if not isinstance(raw_record, dict):
            continue
        name = str(raw_record["cycle_name"])
        path = dataset_dir / "cycles" / f"{name}.parquet"
        frame = pd.read_parquet(path)
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        if timestamps.empty:
            continue
        stage = frame.get("cycle_stage", pd.Series("partial", index=frame.index)).astype(str)
        stable_values = timestamps.loc[stage.eq("frost_development")]
        start = stable_values.min() if not stable_values.empty else None
        end = start + pd.Timedelta(seconds=baseline_seconds) if start is not None else None
        window = (
            frame.loc[timestamps.ge(start) & timestamps.lt(end)]
            if start is not None and end is not None
            else frame.iloc[0:0]
        )
        usable = not window.empty and timestamps.max() >= end if end is not None else False
        registry = _read_manifest(dataset_dir / "channel_registry.json")
        channels = registry.get("channels", {})
        if not isinstance(channels, Mapping):
            raise ValueError("channel_registry channels must be a mapping")
        for channel, settings in channels.items():
            if not isinstance(settings, Mapping):
                continue
            if not bool(settings.get("analysis_candidate", False)) and settings.get(
                "role"
            ) != "performance":
                continue
            baseline_column = f"{channel}__baseline"
            residual_column = f"{channel}__baseline_residual"
            if baseline_column not in frame.columns or residual_column not in frame.columns:
                continue
            if usable and channel in window:
                values = pd.to_numeric(window[channel], errors="coerce")
                baseline = values.dropna().median()
                if pd.notna(baseline):
                    frame[baseline_column] = float(baseline)
                    frame[residual_column] = pd.to_numeric(
                        frame[channel], errors="coerce"
                    ) - float(baseline)
                    continue
            frame[baseline_column] = np.nan
            frame[residual_column] = np.nan
        frame.to_parquet(path, index=False)
        frame.to_csv(dataset_dir / "cycles" / f"{name}.csv", index=False)
