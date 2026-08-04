"""Build and manage the self-contained Cycle Dataset schema 3.

The module owns orchestration only; scientific transforms remain in Prepare/Process
and image, metadata, IO, and validation concerns live in their focused modules.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping, Sequence
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

    experiment_id, experiment_date = _input_experiment_identity(input_path, project_root)
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
            return target
        if experiments:
            last_date = max(str(item["experiment_date"])[:10] for item in experiments)
            if experiment_date <= last_date:
                raise ValueError(
                    "historical or same-date append is not supported; use dataset rebuild"
                )

        def append_operation(staging: Path) -> None:
            pipeline = _run_direct_pipeline(input_path, project_root)
            _append_direct_pipeline(staging, pipeline)

        return mutate_dataset(
            target,
            append_operation,
            validate=validate_staging_structure,
        )

    def build_operation(staging: Path) -> None:
        pipeline = _run_direct_pipeline(input_path, project_root)
        _materialize_direct_pipelines(staging, [pipeline])

    return mutate_dataset(
        target,
        build_operation,
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
    for row in summary.to_dict(orient="records"):
        key = (str(row["experiment_id"]), str(row["cycle_id"]))
        if allowed is not None and key not in allowed:
            continue
        raw_start = pd.to_datetime(
            cast(Any, row.get("segment_start")), errors="coerce"
        )
        start = (
            pd.Timestamp(raw_start)
            if not pd.isna(raw_start)
            else prepared_starts.get(key)
        )
        start_value = int(start.value) if start is not None else pd.Timestamp.max.value
        rows.append(
            (
                (
                    str(row["experiment_date"])[:10],
                    start_value,
                    str(row["cycle_id"]),
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


def _input_experiment_identity(input_path: Path, project_root: Path) -> tuple[str, str]:
    """Resolve only the stable date identity used by Dataset add."""
    config = _load_config_for_input(input_path, project_root)
    return str(config.experiment_id), str(config.experiment_date)[:10]


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
    return _DirectDatePipeline(
        input_dir=input_path.resolve(),
        config=config,
        channels=channels,
        prepared=prepared,
        summary=final_summary,
        processed=processed,
    )


def _settings_mapping(value: Any) -> Mapping[str, Any]:
    if is_dataclass(value):
        return cast(Mapping[str, Any], asdict(cast(Any, value)))
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
    processing_settings = _processing_settings(pipelines[0])
    for pipeline in pipelines[1:]:
        if _processing_settings(pipeline) != processing_settings:
            raise ValueError("Dataset processing settings changed across experiments")
    registry["processing_settings"] = {
        "feature_windows_minutes": processing_settings["feature_windows_minutes"],
    }
    registry["baseline_seconds"] = processing_settings["baseline_seconds"]
    registry["baseline_managed"] = processing_settings["baseline_managed"]
    registry["recovery_edit"] = processing_settings["recovery_edit"]
    registry["image_coverage"] = {"max_image_gap_seconds": 40.0}
    registry.pop("canonical_hash", None)
    registry["canonical_hash"] = canonical_registry_hash(registry)
    return registry


def _processing_settings(pipeline: _DirectDatePipeline) -> dict[str, Any]:
    process = pipeline.config.process
    baseline = getattr(process, "baseline", None)
    windows = getattr(process, "feature_windows_minutes", ())
    return {
        "feature_windows_minutes": [int(value) for value in windows],
        "baseline_seconds": int(getattr(baseline, "baseline_seconds", 60)),
        "baseline_managed": False,
        "recovery_edit": {
            # Normal cycle segmentation uses the fixed-priority Ts−2/Ts−3/Ts−4
            # crossing rule.  Persist that rule until a user explicitly edits it.
            "mode": "ts-minus",
            "seconds": None,
            "fallback_seconds": float(
                getattr(pipeline.config.cycles, "stable_heating_seconds", 180)
            ),
            "managed": False,
        },
    }


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


def _rewrite_original_schema(
    staging: Path,
    records: Sequence[dict[str, Any]],
    original_columns: Sequence[str],
) -> set[str]:
    """Fill newly introduced original columns in historical cycle files."""
    from .dataset_io import write_atomic_csv

    expected = [str(column) for column in original_columns]
    changed: set[str] = set()
    for record in records:
        assets = record.get("assets")
        if not isinstance(assets, Mapping):
            raise ValueError(f"cycle assets are missing: {record.get('cycle_name')}")
        relative = assets.get("original_csv")
        if not isinstance(relative, str):
            raise ValueError(f"original CSV asset is missing: {record.get('cycle_name')}")
        path = staging / relative
        columns = [str(column) for column in pd.read_csv(path, nrows=0).columns]
        if columns == expected:
            continue
        frame = pd.read_csv(path).reindex(columns=expected)
        write_atomic_csv(frame, path)
        hashes = record.setdefault("asset_sha256", {})
        if not isinstance(hashes, dict):
            raise ValueError(f"cycle asset hashes are invalid: {record.get('cycle_name')}")
        hashes["original_csv"] = sha256_file(path)
        changed.add(relative)
    return changed


def _materialize_cycle(
    staging: Path,
    pipeline: _DirectDatePipeline,
    key: CycleKey,
    cycle_name: str,
    registry: Mapping[str, Any],
    original_columns: Sequence[str],
    image_metadata: pd.DataFrame,
    summary_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one complete cycle and return its Catalog record."""
    from .dataset_images import _cycle_image_summary, _cycle_window, _sensor_coverage_intervals
    from .dataset_io import write_atomic_csv, write_atomic_parquet
    from .dataset_metadata import build_cycle_record
    from .visualization import render_cycle_publication, render_rgb_coverage_intervals

    original = _prepared_export_frame(
        pipeline.prepared.loc[
            pipeline.prepared["experiment_id"].astype(str).eq(key[0])
            & pipeline.prepared["cycle_id"].astype(str).eq(key[1])
        ]
    ).reindex(columns=list(original_columns))
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
        summary_row,
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
    return record


def _materialize_direct_pipelines(
    staging: Path,
    pipelines: Sequence[_DirectDatePipeline],
) -> None:
    from .dataset_images import collect_final_images, copy_final_image, image_metadata_frame_final
    from .dataset_io import write_atomic_json, write_atomic_parquet
    from .dataset_metadata import experiment_record, now_iso

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
    for image in all_images:
        copy_final_image(image, staging)

    summary_lookup: dict[CycleKey, dict[str, Any]] = {
        (str(row["experiment_id"]), str(row["cycle_id"])): {
            str(key): value for key, value in row.items()
        }
        for row in summary.to_dict(orient="records")
    }
    pipelines_by_experiment = {
        str(pipeline.config.experiment_id): pipeline for pipeline in pipelines
    }
    records = [
        _materialize_cycle(
            staging,
            pipelines_by_experiment[key[0]],
            key,
            cycle_name,
            registry,
            original_columns,
            image_metadata,
            summary_lookup[key],
        )
        for key, cycle_name in sorted(
            names.items(), key=lambda item: parse_cycle_name(item[1])
        )
    ]

    registry["canonical_hash"] = _registry_hash(registry)
    write_atomic_json(registry, staging / "channel_registry.json")
    experiments = [
        experiment_record(
            str(pipeline.config.experiment_id),
            str(pipeline.config.experiment_date),
            pipeline.input_dir,
            pipeline.config.project_root,
        )
        for pipeline in sorted(pipelines, key=lambda item: str(item.config.experiment_date))
    ]
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


def _append_direct_pipeline(  # noqa: C901
    staging: Path,
    pipeline: _DirectDatePipeline,
) -> None:
    from .dataset_images import collect_final_images, copy_final_image, image_metadata_frame_final
    from .dataset_io import write_atomic_csv, write_atomic_json, write_atomic_parquet
    from .dataset_metadata import experiment_record, now_iso, read_catalog, read_manifest
    from .dataset_registry import merge_registries

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
    for key in (
        "image_coverage",
        "processing_settings",
        "baseline_seconds",
        "baseline_managed",
        "recovery_edit",
    ):
        if key in old_registry:
            merged_registry[key] = old_registry[key]
        elif key in candidate:
            merged_registry[key] = candidate[key]

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

    summary_lookup: dict[CycleKey, dict[str, Any]] = {
        (str(row["experiment_id"]), str(row["cycle_id"])): {
            str(key): value for key, value in row.items()
        }
        for row in pipeline.summary.to_dict(orient="records")
    }
    new_records = [
        _materialize_cycle(
            staging,
            pipeline,
            key,
            cycle_name,
            merged_registry,
            original_columns,
            merged_images,
            summary_lookup[key],
        )
        for key, cycle_name in sorted(
            names.items(), key=lambda item: parse_cycle_name(item[1])
        )
    ]

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
                scientific,
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

    all_records = [*old_records, *new_records]
    catalog["cycles"] = all_records
    _rewrite_original_schema(staging, all_records, original_columns)

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
        },
    ]
    manifest["experiments"].sort(key=lambda value: str(value["experiment_date"]))
    manifest["updated_at"] = now_iso()
    write_atomic_json(manifest, staging / "dataset_manifest.json")
    write_atomic_json(catalog, staging / "cycle_catalog.json")
    _apply_saved_dataset_settings(staging, catalog, merged_registry, set(names.values()))
    write_atomic_json(catalog, staging / "cycle_catalog.json")
    merged_registry["canonical_hash"] = _registry_hash(merged_registry)
    write_atomic_json(merged_registry, staging / "channel_registry.json")
    _refresh_cycles(
        staging,
        sorted(names.values()),
        asset_updates={
            name: {"parquet", "csv", "original_csv", "publication", "rgb_coverage"}
            for name in names.values()
        },
    )


def _apply_saved_dataset_settings(
    staging: Path,
    catalog: dict[str, Any],
    registry: Mapping[str, Any],
    cycle_names: set[str],
) -> None:
    """Apply Dataset-managed scientific settings to newly materialized cycles."""
    from .dataset_edit import apply_baseline_edit, apply_recovery_edit

    recovery = registry.get("recovery_edit")
    if isinstance(recovery, Mapping) and bool(recovery.get("managed", False)):
        mode = str(recovery.get("mode", ""))
        if mode in {"seconds", "ts-minus"}:
            raw_seconds = recovery.get("seconds")
            seconds = int(raw_seconds) if raw_seconds is not None else None
            apply_recovery_edit(
                staging,
                catalog,
                mode=mode,
                recovery_seconds=seconds,
                registry=registry,
                cycle_names=cycle_names,
            )
    baseline_seconds = registry.get("baseline_seconds")
    if bool(registry.get("baseline_managed", False)) and baseline_seconds is not None:
        apply_baseline_edit(
            staging,
            catalog,
            baseline_seconds=int(baseline_seconds),
            registry=registry,
            cycle_names=cycle_names,
        )


def _refresh_cycles(  # noqa: C901
    staging: Path,
    cycle_names: Sequence[str] | None = None,
    *,
    render_publication: bool = True,
    render_coverage: bool = True,
    asset_updates: Mapping[str, Collection[str]] | None = None,
) -> None:
    """Synchronize image statistics and requested figures inside staging."""
    from .dataset_images import _cycle_image_summary, _cycle_window, _sensor_coverage_intervals
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
        if asset_updates is None:
            changed_assets = set()
            if render_publication:
                changed_assets.add("publication")
            if render_coverage:
                changed_assets.add("rgb_coverage")
        else:
            changed_assets = {
                str(asset)
                for asset in asset_updates.get(cycle_name, ())
            }
        hashes = record.setdefault("asset_sha256", {})
        for asset in changed_assets:
            if asset not in assets:
                raise ValueError(f"unknown cycle asset: {asset}")
            hashes[asset] = sha256_file(staging / str(assets[asset]))
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
                    asset_updates={cycle_name: {"publication"}},
                )
                return
        raise KeyError(f"unknown cycle: {cycle_name}")

    return mutate_dataset(dataset_dir, operation, validate=validate_staging_structure)


def edit_dataset(  # noqa: C901
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
        changed_assets: dict[str, set[str]] = {}
        if baseline_seconds is not None:
            baseline_changed = apply_baseline_edit(
                staging,
                catalog,
                baseline_seconds=baseline_seconds,
                registry=registry,
            )
            render_publication = True
            for name in baseline_changed:
                changed_assets.setdefault(name, set()).update(
                    {"parquet", "csv", "publication"}
                )
        if recovery_seconds is not None or recovery_end_by is not None:
            recovery_changed = apply_recovery_edit(
                staging,
                catalog,
                mode="seconds" if recovery_seconds is not None else "ts-minus",
                recovery_seconds=recovery_seconds,
                registry=registry,
            )
            render_publication = True
            render_coverage = True
            for name in recovery_changed:
                changed_assets.setdefault(name, set()).update(
                    {"parquet", "csv", "original_csv", "publication", "rgb_coverage"}
                )

        renames: dict[str, str] = {}
        for expression in camera_renames:
            if "=" not in expression:
                raise ValueError(f"camera rename must be OLD=NEW: {expression}")
            old, new = expression.split("=", 1)
            renames[old] = new
        if renames:
            camera_changed = rename_camera_role(staging, renames)
            render_coverage = True
            for name in camera_changed:
                changed_assets.setdefault(name, set()).add("rgb_coverage")

        registry.pop("canonical_hash", None)
        registry["canonical_hash"] = canonical_registry_hash(registry)
        write_atomic_json(registry, staging / "channel_registry.json")
        write_atomic_json(catalog, staging / "cycle_catalog.json")
        _refresh_cycles(
            staging,
            sorted(changed_assets),
            render_publication=render_publication,
            render_coverage=render_coverage,
            asset_updates=changed_assets,
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
