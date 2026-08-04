"""Build and manage the self-contained Cycle Dataset schema 3.

The module owns orchestration only; scientific transforms remain in Prepare/Process
and image, metadata, IO, and validation concerns live in their focused modules.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .config import find_project_root
from .io import sha256_file

DATASET_SCHEMA_VERSION = 3
DATASET_ID = "frost_cycle_dataset"
CYCLE_NAME_WIDTH = 6
SOURCE_QUALITY_SUFFIXES = (
    "__missing",
    "__invalid",
    "__duplicate",
    "__conflict",
)
CycleKey = tuple[str, str]


def validate_dataset_id(value: str) -> str:
    if value != DATASET_ID:
        raise ValueError(f"invalid dataset_id: {value!r}")
    return value


def make_cycle_uid(experiment_id: str, cycle_id: str) -> str:
    return f"{experiment_id}::{cycle_id}"


def format_cycle_name(index: int) -> str:
    if index < 1:
        raise ValueError("dataset cycle index must be positive")
    return f"frost_cycle_{index:0{CYCLE_NAME_WIDTH}d}"


def parse_cycle_name(name: str) -> int:
    match = re.fullmatch(rf"frost_cycle_(\d{{{CYCLE_NAME_WIDTH},}})", name)
    if match is None or int(match.group(1)) < 1:
        raise ValueError(f"invalid cycle_name: {name!r}")
    return int(match.group(1))


def _prepared_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    drop = {"cycle_elapsed_seconds", "cycle_progress"}
    image_pattern = re.compile(r"^image_.+_(?:path|time|offset_seconds)$")
    columns = [
        str(column)
        for column in frame.columns
        if (
            str(column) not in drop
            and not image_pattern.fullmatch(str(column))
            and not str(column).endswith(SOURCE_QUALITY_SUFFIXES)
            and not str(column).endswith("__baseline")
            and not str(column).endswith("__baseline_residual")
        )
    ]
    if "timestamp" not in columns:
        raise ValueError("Prepared data has no timestamp column")
    return frame.loc[:, columns].sort_values("timestamp", kind="stable").reset_index(
        drop=True
    )


@dataclass(frozen=True)
class _DirectDatePipeline:
    input_dir: Path
    config: Any
    channels: Mapping[str, Mapping[str, Any]]
    prepared: pd.DataFrame
    summary: pd.DataFrame
    processed: pd.DataFrame
    source_fingerprint: str


def add_dataset(input_dir: Path, dataset_dir: Path | None = None) -> Path:
    """Build or append one date directly from raw input."""
    input_path = Path(input_dir).resolve()
    project_root = _resolve_project_root()
    target = (
        Path(dataset_dir).resolve()
        if dataset_dir is not None
        else project_root / "dataset"
    )
    _validate_date_input(input_path, project_root)

    from .dataset_io import mutate_dataset
    from .dataset_metadata import read_manifest
    from .dataset_validation import validate_staging_structure

    experiment_id, experiment_date, fingerprint = _input_source_identity(
        input_path, project_root
    )
    if target.exists():
        manifest = read_manifest(target)
        experiments = manifest["experiments"]
        existing = next(
            (
                item
                for item in experiments
                if isinstance(item, Mapping)
                and str(item.get("experiment_id")) == experiment_id
            ),
            None,
        )
        if existing is not None:
            if str(existing.get("source_fingerprint", "")) == fingerprint:
                return target
            raise ValueError(f"source fingerprint conflict for {experiment_id}")
        if experiments:
            last_date = max(str(item["experiment_date"])[:10] for item in experiments)
            if experiment_date <= last_date:
                raise ValueError(
                    "historical or same-date append is not supported; use dataset rebuild"
                )

        def operation(staging: Path) -> None:
            pipeline = _run_direct_pipeline(input_path, project_root)
            _append_direct_pipeline(staging, pipeline, fingerprint)

        return mutate_dataset(target, operation, validate=validate_staging_structure)

    def operation(staging: Path) -> None:
        pipeline = _run_direct_pipeline(input_path, project_root)
        _materialize_direct_pipelines(staging, [pipeline])

    return mutate_dataset(
        target,
        operation,
        validate=validate_staging_structure,
        rebuild=True,
    )


def rebuild_dataset(
    input_dirs: Sequence[Path], dataset_dir: Path | None = None
) -> Path:
    """Rebuild a Dataset from raw dates without reading the previous Dataset."""
    if not input_dirs:
        raise ValueError("dataset rebuild requires at least one INPUT_DIR")
    project_root = _resolve_project_root()
    inputs = sorted(
        {Path(value).resolve() for value in input_dirs},
        key=lambda path: path.name,
    )
    for input_path in inputs:
        _validate_date_input(input_path, project_root)
    target = (
        Path(dataset_dir).resolve()
        if dataset_dir is not None
        else project_root / "dataset"
    )

    from .dataset_io import mutate_dataset
    from .dataset_validation import validate_staging_structure

    def operation(staging: Path) -> None:
        pipelines = [_run_direct_pipeline(path, project_root) for path in inputs]
        _materialize_direct_pipelines(staging, pipelines)

    return mutate_dataset(
        target,
        operation,
        validate=validate_staging_structure,
        rebuild=True,
    )


def assign_final_cycle_names_by_time(
    summary: pd.DataFrame,
    *,
    prepared: pd.DataFrame | None = None,
    start_index: int = 1,
) -> dict[CycleKey, str]:
    """Assign names by segment start and retain only Prepared cycles."""
    required = {"experiment_id", "experiment_date", "cycle_id"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"cycle summary missing columns: {missing}")

    allowed: set[CycleKey] | None = None
    if prepared is not None:
        _require_columns(prepared, ["experiment_id", "cycle_id"], "Prepared")
        allowed = {
            (str(values[0]), str(values[1]))
            for values in prepared[["experiment_id", "cycle_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }

    prepared_starts: dict[CycleKey, pd.Timestamp] = {}
    if prepared is not None and "timestamp" in prepared:
        prepared_times = prepared.copy()
        prepared_times["timestamp"] = pd.to_datetime(
            prepared_times["timestamp"], errors="coerce"
        )
        for key, group in prepared_times.groupby(
            ["experiment_id", "cycle_id"], sort=False, dropna=False
        ):
            values = group["timestamp"].dropna()
            if not values.empty:
                prepared_starts[(str(key[0]), str(key[1]))] = pd.Timestamp(values.min())

    rows: list[tuple[tuple[str, int, str], CycleKey]] = []
    for values in summary.to_dict(orient="records"):
        key = (str(values["experiment_id"]), str(values["cycle_id"]))
        if allowed is not None and key not in allowed:
            continue
        raw_start = pd.to_datetime(values.get("segment_start"), errors="coerce")
        start = (
            pd.Timestamp(raw_start)
            if not pd.isna(raw_start)
            else prepared_starts.get(key)
        )
        start_value = int(start.value) if start is not None else pd.Timestamp.max.value
        rows.append(
            (
                (
                    str(values["experiment_date"])[:10],
                    start_value,
                    str(values["cycle_id"]),
                ),
                key,
            )
        )

    rows.sort(key=lambda item: item[0])
    names: dict[CycleKey, str] = {}
    for offset, (_sort_key, key) in enumerate(rows, start=start_index):
        if key in names:
            raise ValueError(f"duplicate cycle identity in summary: {key}")
        names[key] = format_cycle_name(offset)
    return names


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _resolve_project_root() -> Path:
    root = find_project_root(Path(__file__))
    if root is None:
        raise FileNotFoundError("could not find project root containing pyproject.toml")
    return root


def _validate_date_input(input_path: Path, project_root: Path) -> None:
    if not input_path.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_path}")
    if re.fullmatch(r"\d{4}", input_path.name) is None:
        raise ValueError("dataset input basename must be a four-digit MMDD date")
    config_path = project_root / "configs" / f"{input_path.name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"date configuration does not exist: {config_path}")


def _load_config_for_input(input_path: Path, project_root: Path) -> Any:
    from .config import load_config

    config = load_config(project_root / "configs" / f"{input_path.name}.yaml")
    if not str(config.experiment_date).startswith("2026-"):
        raise ValueError("Dataset currently accepts only 2026 experiment dates")
    object.__setattr__(config, "input_dir", input_path.resolve())
    return config


def _input_source_identity(
    input_path: Path, project_root: Path
) -> tuple[str, str, str]:
    """Build the add/no-op fingerprint without reading image contents."""
    from .channels import load_channels
    from .config import resolved_config_sha256
    from .io import discover_inputs

    config = _load_config_for_input(input_path, project_root)
    channels = load_channels(config.channels_path)
    inputs = discover_inputs(config)
    inventory: list[tuple[str, int, int]] = []
    for path in [*inputs.sensor_files, *inputs.image_files]:
        relative = path.relative_to(input_path).as_posix()
        stat = path.stat()
        inventory.append((relative, int(stat.st_size), int(stat.st_mtime_ns)))
    payload = {
        "experiment_id": str(config.experiment_id),
        "experiment_date": str(config.experiment_date)[:10],
        "config": resolved_config_sha256(config),
        "channels": sorted(str(name) for name in channels),
        "inventory": sorted(inventory),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        str(config.experiment_id),
        str(config.experiment_date)[:10],
        hashlib.sha256(encoded).hexdigest(),
    )


def _run_direct_pipeline(input_path: Path, project_root: Path) -> _DirectDatePipeline:
    from .channels import load_channels
    from .prepare import prepare
    from .process import process
    from .validation import validate_prepared, validate_processed

    config = _load_config_for_input(input_path, project_root)
    channels = load_channels(config.channels_path)
    prepared, initial_summary, prepare_summary = prepare(config, channels)
    validate_prepared(prepared, initial_summary)
    processed, final_summary = process(prepared, initial_summary, config, channels)
    validate_processed(processed, final_summary)
    _ = prepare_summary
    _, _, fingerprint = _input_source_identity(input_path, project_root)
    return _DirectDatePipeline(
        input_dir=input_path.resolve(),
        config=config,
        channels=channels,
        prepared=prepared,
        summary=final_summary,
        processed=processed,
        source_fingerprint=fingerprint,
    )


def _settings_mapping(value: Any) -> Mapping[str, Any]:
    if is_dataclass(value):
        return cast(Mapping[str, Any], asdict(value))
    if isinstance(value, Mapping):
        return value
    return {}


def _build_direct_registry(
    pipelines: Sequence[_DirectDatePipeline],
) -> dict[str, Any]:
    from .dataset_registry import (
        canonical_registry_hash,
        merge_registries,
        registry_from_frame,
    )

    registry: dict[str, Any] | None = None
    for pipeline in pipelines:
        candidate = registry_from_frame(
            pipeline.processed,
            pipeline.channels,
            analysis_settings=_settings_mapping(pipeline.config.analysis),
            resample_interval_seconds=int(
                pipeline.config.process.resample_interval_seconds
            ),
        )
        registry = candidate if registry is None else merge_registries(registry, candidate)
    if registry is None:
        raise ValueError("Dataset requires at least one processed pipeline")
    registry["image_coverage"] = {"max_image_gap_seconds": 40.0}
    registry.pop("canonical_hash", None)
    registry["canonical_hash"] = canonical_registry_hash(registry)
    return registry


def _canonical_original_columns(
    pipelines: Sequence[_DirectDatePipeline],
) -> list[str]:
    ordered: list[str] = []
    for pipeline in pipelines:
        for column in _prepared_export_frame(pipeline.prepared).columns:
            if str(column) not in ordered:
                ordered.append(str(column))
    if "timestamp" not in ordered:
        raise ValueError("Prepared data has no timestamp column")
    return ordered


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
    staging: Path,
    cycle_name: str,
    frame: pd.DataFrame,
    image_metadata: pd.DataFrame,
    registry: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]],
]:
    from .dataset_images import (
        build_rgb_coverage_intervals,
        scan_final_cycle_images,
        summarize_rgb_coverage,
    )

    start, end = _cycle_window(frame)
    images = scan_final_cycle_images(staging, cycle_name, image_metadata)
    settings = registry.get("image_coverage", {})
    max_gap = float(
        settings.get("max_image_gap_seconds", 40.0)
        if isinstance(settings, Mapping)
        else 40.0
    )
    by_role: dict[str, Any] = {}
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
            by_role[str(role)] = {
                "image_count": int(len(group)),
                "coverage_ratio": summarize_rgb_coverage(start, end, role_intervals),
            }
    return {"image_count": int(len(images)), "by_camera_role": by_role}, intervals


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
            if isinstance(settings, Mapping)
            and bool(settings.get("coverage_required", False))
        ]
        if isinstance(channel_settings, Mapping)
        else []
    )
    observed_names = required_names
    if not observed_names:
        observed_names = [
            str(field["name"])
            for field in registry.get("fields", [])
            if isinstance(field, Mapping)
            and str(field.get("name")) in frame
            and str(field.get("name")) not in {"timestamp", "cycle_stage"}
            and not str(field.get("name")).endswith("__imputed")
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

    available_rows = frame.loc[valid & availability].sort_values(
        "timestamp", kind="stable"
    )
    start = pd.Timestamp(ordered.iloc[0])
    end = pd.Timestamp(ordered.iloc[-1]) + pd.Timedelta(seconds=step)
    available: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if not available_rows.empty:
        current_start = pd.Timestamp(available_rows.iloc[0]["timestamp"])
        previous = current_start
        for raw in available_rows["timestamp"].iloc[1:]:
            current = pd.Timestamp(raw)
            if (current - previous).total_seconds() > step * 1.5:
                available.append(
                    (current_start, previous + pd.Timedelta(seconds=step))
                )
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


def _dataset_frame(
    frame: pd.DataFrame,
    registry: Mapping[str, Any],
    *,
    cycle_name: str,
    cycle_uid: str,
) -> pd.DataFrame:
    from .dataset_registry import canonical_frame

    result = canonical_frame(frame, registry)
    result.insert(0, "cycle_uid", cycle_uid)
    result.insert(0, "cycle_name", cycle_name)
    result.insert(0, "dataset_cycle_index", parse_cycle_name(cycle_name))
    result.insert(0, "dataset_schema_version", DATASET_SCHEMA_VERSION)
    result.insert(0, "dataset_id", DATASET_ID)
    return result


def _cycle_assets(cycle_name: str) -> dict[str, str]:
    return {
        "parquet": f"cycles/{cycle_name}.parquet",
        "csv": f"cycles/{cycle_name}.csv",
        "original_csv": f"cycles_original/{cycle_name}.csv",
        "publication": f"cycles/{cycle_name}.png",
        "rgb_coverage": f"cycles/{cycle_name}_rgb_coverage.png",
    }


def _asset_hashes(root: Path, assets: Mapping[str, str]) -> dict[str, str]:
    return {name: sha256_file(root / path) for name, path in assets.items()}


def _materialize_direct_pipelines(
    staging: Path,
    pipelines: Sequence[_DirectDatePipeline],
) -> None:
    from .dataset_images import (
        collect_final_images,
        copy_final_image,
        image_metadata_frame_final,
    )
    from .dataset_io import write_atomic_csv, write_atomic_json, write_atomic_parquet
    from .dataset_metadata import build_cycle_record, experiment_record, now_iso
    from .visualization import render_cycle_publication, render_rgb_coverage_intervals

    if not pipelines:
        raise ValueError("Dataset requires at least one date")
    summary = pd.concat([pipeline.summary for pipeline in pipelines], ignore_index=True)
    prepared = pd.concat([pipeline.prepared for pipeline in pipelines], ignore_index=True)
    names = assign_final_cycle_names_by_time(summary, prepared=prepared)
    if not names:
        raise ValueError("Dataset contains no cycles with Prepared rows")

    summary_keys = {
        (str(row["experiment_id"]), str(row["cycle_id"]))
        for row in summary.to_dict(orient="records")
    }
    prepared_keys = {
        (str(row["experiment_id"]), str(row["cycle_id"]))
        for row in prepared[["experiment_id", "cycle_id"]]
        .drop_duplicates()
        .to_dict(orient="records")
    }
    if not prepared_keys <= summary_keys:
        raise ValueError("Prepared cycle is missing from cycle summary")

    processed_counts = processed_counts_by_key(pipelines)
    for key in names:
        if processed_counts.get(key, 0) <= 0:
            raise ValueError(f"cycle has Prepared rows but no Processed rows: {key}")

    registry = _build_direct_registry(pipelines)
    original_columns = _canonical_original_columns(pipelines)
    all_images: list[dict[str, object]] = []
    for pipeline in pipelines:
        all_images.extend(
            collect_final_images(
                pipeline.prepared,
                input_dir=pipeline.input_dir,
                cycle_names=names,
            )
        )
    image_metadata = image_metadata_frame_final(all_images)
    write_atomic_parquet(image_metadata, staging / "image_metadata.parquet")
    for record in all_images:
        copy_final_image(record, staging)

    summary_lookup = {
        (str(row["experiment_id"]), str(row["cycle_id"])): row
        for row in summary.to_dict(orient="records")
    }
    records: list[dict[str, Any]] = []
    for key, cycle_name in sorted(
        names.items(), key=lambda item: parse_cycle_name(item[1])
    ):
        pipeline = next(
            item for item in pipelines if str(item.config.experiment_id) == key[0]
        )
        original = _prepared_export_frame(
            pipeline.prepared.loc[
                pipeline.prepared["experiment_id"].astype(str).eq(key[0])
                & pipeline.prepared["cycle_id"].astype(str).eq(key[1])
            ]
        ).reindex(columns=original_columns)
        raw = pipeline.processed.loc[
            pipeline.processed["experiment_id"].astype(str).eq(key[0])
            & pipeline.processed["cycle_id"].astype(str).eq(key[1])
        ].copy()
        canonical = _dataset_frame(
            raw,
            registry,
            cycle_name=cycle_name,
            cycle_uid=make_cycle_uid(*key),
        )
        assets = _cycle_assets(cycle_name)
        write_atomic_parquet(canonical, staging / assets["parquet"])
        write_atomic_csv(canonical, staging / assets["csv"])
        write_atomic_csv(original, staging / assets["original_csv"])

        image_summary, intervals = _cycle_image_summary(
            staging, cycle_name, canonical, image_metadata, registry
        )
        record = build_cycle_record(
            summary_lookup[key],
            cycle_name=cycle_name,
            cycle_uid=make_cycle_uid(*key),
            processed=canonical,
            original=original,
            image_summary=image_summary,
            assets=assets,
            asset_sha256={},
        )
        render_cycle_publication(canonical, record, staging / assets["publication"])
        start, end = _cycle_window(canonical)
        render_rgb_coverage_intervals(
            cycle_name,
            start,
            end,
            intervals,
            staging / assets["rgb_coverage"],
            sensor_intervals=_sensor_coverage_intervals(canonical, registry),
        )
        record["asset_sha256"] = _asset_hashes(staging, assets)
        records.append(record)

    registry["canonical_hash"] = _registry_hash(registry)
    write_atomic_json(registry, staging / "channel_registry.json")
    experiments: list[dict[str, Any]] = []
    for pipeline in sorted(pipelines, key=lambda item: str(item.config.experiment_date)):
        provenance = experiment_record(
            str(pipeline.config.experiment_id),
            str(pipeline.config.experiment_date),
            pipeline.input_dir,
            pipeline.config.project_root,
        )
        provenance["source_fingerprint"] = pipeline.source_fingerprint
        experiments.append(provenance)
    now = now_iso()
    write_atomic_json(
        {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "created_at": now,
            "updated_at": now,
            "experiments": experiments,
        },
        staging / "dataset_manifest.json",
    )
    write_atomic_json({"cycles": records}, staging / "cycle_catalog.json")
    (staging / "README.md").write_text(
        "# frost_cycle_dataset\n\nSelf-contained Cycle Dataset schema 3.\n",
        encoding="utf-8",
    )


def processed_counts_by_key(
    pipelines: Sequence[_DirectDatePipeline],
) -> dict[CycleKey, int]:
    counts: dict[CycleKey, int] = {}
    for pipeline in pipelines:
        for values in pipeline.processed[["experiment_id", "cycle_id"]].itertuples(
            index=False, name=None
        ):
            key = (str(values[0]), str(values[1]))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _registry_hash(registry: Mapping[str, Any]) -> str:
    from .dataset_registry import canonical_registry_hash

    return canonical_registry_hash(registry)


def _append_direct_pipeline(
    staging: Path,
    pipeline: _DirectDatePipeline,
    fingerprint: str,
) -> None:
    from .dataset_images import (
        collect_final_images,
        copy_final_image,
        image_metadata_frame_final,
    )
    from .dataset_io import write_atomic_csv, write_atomic_json, write_atomic_parquet
    from .dataset_metadata import (
        build_cycle_record,
        experiment_record,
        now_iso,
        read_catalog,
        read_manifest,
    )
    from .dataset_registry import merge_registries
    from .visualization import render_cycle_publication, render_rgb_coverage_intervals

    manifest = read_manifest(staging)
    catalog = read_catalog(staging)
    old_records = [record for record in catalog["cycles"] if isinstance(record, dict)]
    names = assign_final_cycle_names_by_time(
        pipeline.summary,
        prepared=pipeline.prepared,
        start_index=len(old_records) + 1,
    )
    old_registry = json.loads(
        (staging / "channel_registry.json").read_text(encoding="utf-8")
    )
    candidate = _build_direct_registry([pipeline])
    merged_registry = merge_registries(old_registry, candidate)
    merged_registry["image_coverage"] = old_registry.get(
        "image_coverage", {"max_image_gap_seconds": 40.0}
    )

    old_images = pd.read_parquet(staging / "image_metadata.parquet")
    new_images = collect_final_images(
        pipeline.prepared,
        input_dir=pipeline.input_dir,
        cycle_names=names,
    )
    for image in new_images:
        copy_final_image(image, staging)
    merged_images = pd.concat(
        [old_images, image_metadata_frame_final(new_images)],
        ignore_index=True,
    )
    write_atomic_parquet(merged_images, staging / "image_metadata.parquet")

    original_columns: list[str] = []
    for record in old_records:
        old_path = staging / str(record["assets"]["original_csv"])
        for column in pd.read_csv(old_path, nrows=0).columns:
            if str(column) not in original_columns:
                original_columns.append(str(column))
    for column in _canonical_original_columns([pipeline]):
        if column not in original_columns:
            original_columns.append(column)

    summary_lookup = {
        (str(row["experiment_id"]), str(row["cycle_id"])): row
        for row in pipeline.summary.to_dict(orient="records")
    }
    new_records: list[dict[str, Any]] = []
    for key, cycle_name in sorted(
        names.items(), key=lambda item: parse_cycle_name(item[1])
    ):
        original = _prepared_export_frame(
            pipeline.prepared.loc[
                pipeline.prepared["experiment_id"].astype(str).eq(key[0])
                & pipeline.prepared["cycle_id"].astype(str).eq(key[1])
            ]
        ).reindex(columns=original_columns)
        raw = pipeline.processed.loc[
            pipeline.processed["experiment_id"].astype(str).eq(key[0])
            & pipeline.processed["cycle_id"].astype(str).eq(key[1])
        ]
        canonical = _dataset_frame(
            raw,
            merged_registry,
            cycle_name=cycle_name,
            cycle_uid=make_cycle_uid(*key),
        )
        assets = _cycle_assets(cycle_name)
        write_atomic_parquet(canonical, staging / assets["parquet"])
        write_atomic_csv(canonical, staging / assets["csv"])
        write_atomic_csv(original, staging / assets["original_csv"])
        image_summary, intervals = _cycle_image_summary(
            staging, cycle_name, canonical, merged_images, merged_registry
        )
        record = build_cycle_record(
            summary_lookup[key],
            cycle_name=cycle_name,
            cycle_uid=make_cycle_uid(*key),
            processed=canonical,
            original=original,
            image_summary=image_summary,
            assets=assets,
            asset_sha256={},
        )
        render_cycle_publication(canonical, record, staging / assets["publication"])
        start, end = _cycle_window(canonical)
        render_rgb_coverage_intervals(
            cycle_name,
            start,
            end,
            intervals,
            staging / assets["rgb_coverage"],
            sensor_intervals=_sensor_coverage_intervals(canonical, merged_registry),
        )
        record["asset_sha256"] = _asset_hashes(staging, assets)
        new_records.append(record)

    old_field_names = [
        str(item["name"])
        for item in old_registry.get("fields", [])
        if isinstance(item, Mapping)
    ]
    merged_field_names = [
        str(item["name"])
        for item in merged_registry.get("fields", [])
        if isinstance(item, Mapping)
    ]
    if merged_field_names != old_field_names:
        from .dataset_registry import canonical_frame

        for record in old_records:
            name = str(record["cycle_name"])
            assets = record["assets"]
            old_frame = pd.read_parquet(staging / str(assets["parquet"]))
            scientific = old_frame.drop(
                columns=[
                    "dataset_id",
                    "dataset_schema_version",
                    "dataset_cycle_index",
                    "cycle_name",
                    "cycle_uid",
                ],
                errors="ignore",
            )
            rewritten = _dataset_frame(
                canonical_frame(scientific, merged_registry),
                merged_registry,
                cycle_name=name,
                cycle_uid=str(record["cycle_uid"]),
            )
            write_atomic_parquet(rewritten, staging / str(assets["parquet"]))
            write_atomic_csv(rewritten, staging / str(assets["csv"]))
            record.setdefault("data", {})["processed_row_count"] = int(len(rewritten))
            hashes = record.setdefault("asset_sha256", {})
            hashes["parquet"] = sha256_file(staging / str(assets["parquet"]))
            hashes["csv"] = sha256_file(staging / str(assets["csv"]))

    merged_registry["canonical_hash"] = _registry_hash(merged_registry)
    write_atomic_json(merged_registry, staging / "channel_registry.json")
    manifest["experiments"] = [
        *manifest["experiments"],
        {
            **experiment_record(
                str(pipeline.config.experiment_id),
                str(pipeline.config.experiment_date),
                pipeline.input_dir,
                pipeline.config.project_root,
            ),
            "source_fingerprint": fingerprint,
        },
    ]
    manifest["experiments"].sort(key=lambda value: str(value["experiment_date"]))
    manifest["updated_at"] = now_iso()
    write_atomic_json(manifest, staging / "dataset_manifest.json")
    catalog["cycles"] = [*old_records, *new_records]
    write_atomic_json(catalog, staging / "cycle_catalog.json")


def _refresh_cycles(
    staging: Path,
    cycle_names: Sequence[str] | None = None,
    *,
    render_publication: bool = True,
    render_coverage: bool = True,
) -> None:
    """Synchronize image statistics and requested figures inside staging."""
    from .dataset_io import write_atomic_json
    from .dataset_metadata import now_iso, read_catalog, read_manifest
    from .visualization import render_cycle_publication, render_rgb_coverage_intervals

    catalog = read_catalog(staging)
    registry = json.loads((staging / "channel_registry.json").read_text(encoding="utf-8"))
    metadata = pd.read_parquet(staging / "image_metadata.parquet")
    selected = set(cycle_names) if cycle_names is not None else None
    for record in catalog["cycles"]:
        if not isinstance(record, dict):
            continue
        cycle_name = str(record["cycle_name"])
        if selected is not None and cycle_name not in selected:
            continue
        assets = record.get("assets")
        if not isinstance(assets, Mapping):
            raise ValueError(f"cycle assets are missing: {cycle_name}")
        frame = pd.read_parquet(staging / str(assets["parquet"]))
        image_summary, intervals = _cycle_image_summary(
            staging, cycle_name, frame, metadata, registry
        )
        record["image"] = image_summary
        record.setdefault("data", {})["processed_row_count"] = int(len(frame))
        if render_publication:
            render_cycle_publication(
                frame, record, staging / str(assets["publication"])
            )
        if render_coverage:
            start, end = _cycle_window(frame)
            render_rgb_coverage_intervals(
                cycle_name,
                start,
                end,
                intervals,
                staging / str(assets["rgb_coverage"]),
                sensor_intervals=_sensor_coverage_intervals(frame, registry),
            )
        hashes = record.setdefault("asset_sha256", {})
        if render_publication:
            hashes["publication"] = sha256_file(
                staging / str(assets["publication"])
            )
        if render_coverage:
            hashes["rgb_coverage"] = sha256_file(
                staging / str(assets["rgb_coverage"])
            )
        if render_publication or render_coverage:
            hashes["parquet"] = sha256_file(staging / str(assets["parquet"]))
            hashes["csv"] = sha256_file(staging / str(assets["csv"]))
            hashes["original_csv"] = sha256_file(
                staging / str(assets["original_csv"])
            )
    write_atomic_json(catalog, staging / "cycle_catalog.json")
    manifest = read_manifest(staging)
    manifest["updated_at"] = now_iso()
    write_atomic_json(manifest, staging / "dataset_manifest.json")


def review_cycle(
    dataset_dir: Path,
    cycle_name: str,
    *,
    status: str,
    reason: str | None = None,
) -> Path:
    """Update the user-controlled Dataset status through one transaction."""
    allowed = {"valid", "partial", "incomplete", "invalid"}
    if status not in allowed:
        raise ValueError(f"invalid Dataset status: {status}")

    from .dataset_io import mutate_dataset, write_atomic_json
    from .dataset_metadata import read_catalog
    from .dataset_validation import validate_staging_structure

    def operation(staging: Path) -> None:
        catalog = read_catalog(staging)
        for record in catalog["cycles"]:
            if isinstance(record, dict) and record.get("cycle_name") == cycle_name:
                record["status"] = status
                record["status_reason"] = reason
                write_atomic_json(catalog, staging / "cycle_catalog.json")
                _refresh_cycles(
                    staging,
                    [cycle_name],
                    render_publication=True,
                    render_coverage=False,
                )
                return
        raise KeyError(f"unknown cycle: {cycle_name}")

    return mutate_dataset(dataset_dir, operation, validate=validate_staging_structure)


def edit_dataset(
    dataset_dir: Path,
    *,
    baseline_seconds: int | None = None,
    recovery_seconds: int | None = None,
    recovery_end_by: str | None = None,
    camera_renames: Sequence[str] = (),
) -> Path:
    """Apply baseline, recovery, or camera-role edits atomically."""
    if recovery_seconds is not None and recovery_end_by is not None:
        raise ValueError("--recovery-seconds and --recovery-end-by are mutually exclusive")
    if (
        baseline_seconds is None
        and recovery_seconds is None
        and recovery_end_by is None
        and not camera_renames
    ):
        raise ValueError("dataset edit requires at least one edit")

    from .dataset_edit import apply_baseline_edit, apply_recovery_edit, rename_camera_role
    from .dataset_io import mutate_dataset, write_atomic_json
    from .dataset_metadata import read_catalog
    from .dataset_registry import canonical_registry_hash
    from .dataset_validation import validate_staging_structure

    def operation(staging: Path) -> None:
        catalog = read_catalog(staging)
        registry = json.loads(
            (staging / "channel_registry.json").read_text(encoding="utf-8")
        )
        render_publication = False
        render_coverage = False
        if baseline_seconds is not None:
            apply_baseline_edit(
                staging,
                catalog,
                baseline_seconds=baseline_seconds,
                registry=registry,
            )
            render_publication = True
        if recovery_seconds is not None or recovery_end_by is not None:
            apply_recovery_edit(
                staging,
                catalog,
                mode="seconds" if recovery_seconds is not None else "ts-minus",
                recovery_seconds=recovery_seconds,
            )
            render_publication = True
            render_coverage = True

        renames: dict[str, str] = {}
        for expression in camera_renames:
            if "=" not in expression:
                raise ValueError(f"camera rename must be OLD=NEW: {expression}")
            old, new = expression.split("=", 1)
            renames[old] = new
        if renames:
            rename_camera_role(staging, renames)
            render_coverage = True

        registry.pop("canonical_hash", None)
        registry["canonical_hash"] = canonical_registry_hash(registry)
        write_atomic_json(registry, staging / "channel_registry.json")
        write_atomic_json(catalog, staging / "cycle_catalog.json")
        _refresh_cycles(
            staging,
            render_publication=render_publication,
            render_coverage=render_coverage,
        )

    return mutate_dataset(dataset_dir, operation, validate=validate_staging_structure)


def refresh_dataset(dataset_dir: Path) -> Path:
    """Refresh current images, statistics, and both figure families atomically."""
    from .dataset_io import mutate_dataset
    from .dataset_validation import validate_staging_structure

    return mutate_dataset(
        dataset_dir,
        lambda staging: _refresh_cycles(staging),
        validate=validate_staging_structure,
    )


def render_dataset(
    dataset_dir: Path,
    cycle_name: str,
    *,
    publication: bool = True,
    coverage: bool = True,
) -> Path:
    """Render selected final assets without reading any source directory."""
    from .dataset_io import mutate_dataset
    from .dataset_metadata import read_catalog
    from .dataset_validation import validate_staging_structure

    def operation(staging: Path) -> None:
        catalog = read_catalog(staging)
        if not any(
            isinstance(record, Mapping)
            and str(record.get("cycle_name")) == cycle_name
            for record in catalog["cycles"]
        ):
            raise KeyError(f"unknown cycle: {cycle_name}")
        _refresh_cycles(
            staging,
            [cycle_name],
            render_publication=publication,
            render_coverage=coverage,
        )

    return mutate_dataset(dataset_dir, operation, validate=validate_staging_structure)
