"""Build and manage the self-contained Cycle Dataset schema 3.

The module owns orchestration only; scientific transforms remain in Prepare/Process
and image, metadata, IO, and validation concerns live in their focused modules.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .config import find_project_root

DATASET_SCHEMA_VERSION = 3
DATASET_ID = "frost_cycle_dataset"
CYCLE_NAME_WIDTH = 6
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
    _validate_date_input(input_path)
    config = _load_config_for_input(input_path, project_root)

    from .dataset_metadata import read_manifest

    experiment_id = str(config.experiment_id)
    experiment_date = str(config.experiment_date)[:10]
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

        pipeline = _run_direct_pipeline(input_path, config)
        _append_direct_pipeline(target, pipeline)
        return target

    pipeline = _run_direct_pipeline(input_path, config)
    _materialize_direct_pipelines(target, [pipeline])
    return target


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
        _validate_date_input(input_path)
    target = (
        Path(dataset_dir).resolve()
        if dataset_dir is not None
        else project_root / "dataset"
    )

    if target.exists():
        import shutil

        shutil.rmtree(target)
    pipelines = [
        _run_direct_pipeline(path, _load_config_for_input(path, project_root))
        for path in inputs
    ]
    _materialize_direct_pipelines(target, pipelines)
    return target


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


def _validate_date_input(input_path: Path) -> None:
    if not input_path.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_path}")
    if re.fullmatch(r"\d{4}", input_path.name) is None:
        raise ValueError("dataset input basename must be a four-digit MMDD date")
def _load_config_for_input(input_path: Path, project_root: Path) -> Any:
    from .config import load_config

    config = load_config(project_root / "configs" / f"{input_path.name}.yaml")
    if not str(config.experiment_date).startswith("2026-"):
        raise ValueError("Dataset currently accepts only 2026 experiment dates")
    object.__setattr__(config, "input_dir", input_path.resolve())
    return config


def _run_direct_pipeline(input_path: Path, config: Any) -> _DirectDatePipeline:
    from .channels import load_channels
    from .prepare import prepare
    from .process import process
    from .validation import validate_prepared, validate_processed

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


def _materialize_cycle(
    dataset_dir: Path,
    pipeline: _DirectDatePipeline,
    key: CycleKey,
    cycle_name: str,
    registry: Mapping[str, Any],
    original_columns: Sequence[str],
    image_metadata: pd.DataFrame,
    summary_row: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Write one complete cycle and return its Catalog record."""
    from .dataset_edit import apply_baseline, apply_recovery
    from .dataset_images import (
        _cycle_image_summary,
        _cycle_window,
        _sensor_coverage_intervals,
    )
    from .dataset_io import write_csv, write_parquet
    from .dataset_metadata import build_cycle_record
    from .dataset_registry import build_processed_frame, export_original_frame
    from .visualization import render_cycle_publication, render_rgb_coverage_intervals

    original = export_original_frame(
        pipeline.prepared.loc[
            pipeline.prepared["experiment_id"].astype(str).eq(key[0])
            & pipeline.prepared["cycle_id"].astype(str).eq(key[1])
        ]
    ).reindex(columns=list(original_columns))
    raw = pipeline.processed.loc[
        pipeline.processed["experiment_id"].astype(str).eq(key[0])
        & pipeline.processed["cycle_id"].astype(str).eq(key[1])
    ].copy()
    canonical = build_processed_frame(
        raw,
        registry,
        cycle_name=cycle_name,
        cycle_uid=make_cycle_uid(*key),
    )
    from .dataset_metadata import cycle_assets

    assets = cycle_assets(cycle_name)
    record = build_cycle_record(
        summary_row,
        cycle_name=cycle_name,
        cycle_uid=make_cycle_uid(*key),
        processed=canonical,
        original=original,
        image_summary={"image_count": 0, "by_camera_role": {}},
        assets=assets,
    )
    metadata_result = image_metadata
    recovery = registry.get("recovery_edit")
    if isinstance(recovery, Mapping) and bool(recovery.get("managed", False)):
        mode = str(recovery.get("mode", ""))
        if mode in {"seconds", "ts-minus"}:
            raw_seconds = recovery.get("seconds")
            original, canonical, metadata_result = apply_recovery(
                original,
                canonical,
                metadata_result,
                record,
                dict(registry),
                mode=mode,
                seconds=int(raw_seconds) if raw_seconds is not None else None,
            )
    if bool(registry.get("baseline_managed", False)):
        baseline_seconds = registry.get("baseline_seconds")
        if baseline_seconds is not None:
            baseline_registry = dict(registry)
            canonical = apply_baseline(
                canonical,
                record,
                baseline_registry,
                seconds=int(baseline_seconds),
            )
    write_parquet(canonical, dataset_dir / assets["parquet"])
    write_csv(canonical, dataset_dir / assets["csv"])
    write_csv(original, dataset_dir / assets["original_csv"])
    image_summary, intervals = _cycle_image_summary(
        dataset_dir, cycle_name, canonical, metadata_result, registry
    )
    final_record = build_cycle_record(
        summary_row,
        cycle_name=cycle_name,
        cycle_uid=make_cycle_uid(*key),
        processed=canonical,
        original=original,
        image_summary=image_summary,
        assets=assets,
    )
    final_record["boundaries"] = record["boundaries"]
    render_cycle_publication(canonical, final_record, dataset_dir / assets["publication"])
    start, end = _cycle_window(canonical)
    render_rgb_coverage_intervals(
        cycle_name,
        start,
        end,
        intervals,
        dataset_dir / assets["rgb_coverage"],
        sensor_intervals=_sensor_coverage_intervals(canonical, registry),
    )
    return final_record, metadata_result


def _materialize_direct_pipelines(
    dataset_dir: Path,
    pipelines: Sequence[_DirectDatePipeline],
) -> None:
    from .dataset_images import collect_final_images, copy_final_image, image_metadata_frame_final
    from .dataset_io import write_json, write_parquet
    from .dataset_metadata import experiment_record, now_iso
    from .dataset_registry import build_registry, merge_original_columns

    if not pipelines:
        raise ValueError("Dataset requires at least one date")
    for directory in ("cycles", "cycles_original", "images"):
        (dataset_dir / directory).mkdir(parents=True, exist_ok=True)
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

    registry = build_registry(pipelines)
    original_columns = merge_original_columns(pipelines)
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
    for image in all_images:
        copy_final_image(image, dataset_dir)

    summary_lookup: dict[CycleKey, dict[str, Any]] = {
        (str(row["experiment_id"]), str(row["cycle_id"])): {
            str(key): value for key, value in row.items()
        }
        for row in summary.to_dict(orient="records")
    }
    pipelines_by_experiment = {
        str(pipeline.config.experiment_id): pipeline for pipeline in pipelines
    }
    records: list[dict[str, Any]] = []
    for key, cycle_name in sorted(
        names.items(), key=lambda item: parse_cycle_name(item[1])
    ):
        record, image_metadata = _materialize_cycle(
            dataset_dir,
            pipelines_by_experiment[key[0]],
            key,
            cycle_name,
            registry,
            original_columns,
            image_metadata,
            summary_lookup[key],
        )
        records.append(record)

    write_parquet(image_metadata, dataset_dir / "image_metadata.parquet")

    write_json(registry, dataset_dir / "channel_registry.json")
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
    write_json(
        {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "created_at": now,
            "updated_at": now,
            "experiments": experiments,
        },
        dataset_dir / "dataset_manifest.json",
    )
    write_json({"cycles": records}, dataset_dir / "cycle_catalog.json")
    (dataset_dir / "README.md").write_text(
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


def _append_direct_pipeline(  # noqa: C901
    dataset_dir: Path,
    pipeline: _DirectDatePipeline,
) -> None:
    from .dataset_images import collect_final_images, copy_final_image, image_metadata_frame_final
    from .dataset_io import read_json, write_csv, write_json, write_parquet
    from .dataset_metadata import experiment_record, now_iso, read_catalog, read_manifest
    from .dataset_registry import (
        align_original_schema,
        build_processed_frame,
        build_registry,
        merge_original_columns,
        merge_registries,
    )

    manifest = read_manifest(dataset_dir)
    catalog = read_catalog(dataset_dir)
    old_records = [record for record in catalog["cycles"] if isinstance(record, dict)]
    names = assign_final_cycle_names_by_time(
        pipeline.summary,
        prepared=pipeline.prepared,
        start_index=len(old_records) + 1,
    )
    old_registry = read_json(dataset_dir / "channel_registry.json")
    candidate = build_registry([pipeline])
    merged_registry = merge_registries(old_registry, candidate)
    for setting_name in (
        "image_coverage",
        "processing_settings",
        "baseline_seconds",
        "baseline_managed",
        "recovery_edit",
    ):
        if setting_name in old_registry:
            merged_registry[setting_name] = old_registry[setting_name]
        elif setting_name in candidate:
            merged_registry[setting_name] = candidate[setting_name]

    old_images = pd.read_parquet(dataset_dir / "image_metadata.parquet")
    new_images = collect_final_images(
        pipeline.prepared,
        input_dir=pipeline.input_dir,
        cycle_names=names,
    )
    for image in new_images:
        copy_final_image(image, dataset_dir)
    merged_images = pd.concat(
        [old_images, image_metadata_frame_final(new_images)],
        ignore_index=True,
    )
    original_columns: list[str] = []
    for record in old_records:
        old_path = dataset_dir / str(record["assets"]["original_csv"])
        for column in pd.read_csv(old_path, nrows=0).columns:
            if str(column) not in original_columns:
                original_columns.append(str(column))
    for column in merge_original_columns([pipeline]):
        if column not in original_columns:
            original_columns.append(column)

    summary_lookup: dict[CycleKey, dict[str, Any]] = {
        (str(row["experiment_id"]), str(row["cycle_id"])): {
            str(key): value for key, value in row.items()
        }
        for row in pipeline.summary.to_dict(orient="records")
    }
    new_records: list[dict[str, Any]] = []
    for key, cycle_name in sorted(
        names.items(), key=lambda item: parse_cycle_name(item[1])
    ):
        record, merged_images = _materialize_cycle(
            dataset_dir,
            pipeline,
            key,
            cycle_name,
            merged_registry,
            original_columns,
            merged_images,
            summary_lookup[key],
        )
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
        for record in old_records:
            name = str(record["cycle_name"])
            assets = record["assets"]
            old_frame = pd.read_parquet(dataset_dir / str(assets["parquet"]))
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
            rewritten = build_processed_frame(
                scientific,
                merged_registry,
                cycle_name=name,
                cycle_uid=str(record["cycle_uid"]),
            )
            write_parquet(rewritten, dataset_dir / str(assets["parquet"]))
            write_csv(rewritten, dataset_dir / str(assets["csv"]))
            record.setdefault("data", {})["processed_row_count"] = int(len(rewritten))

    all_records = [*old_records, *new_records]
    catalog["cycles"] = all_records
    align_original_schema(dataset_dir, all_records, original_columns)
    write_parquet(merged_images, dataset_dir / "image_metadata.parquet")

    write_json(merged_registry, dataset_dir / "channel_registry.json")
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
    write_json(manifest, dataset_dir / "dataset_manifest.json")
    write_json(catalog, dataset_dir / "cycle_catalog.json")


def _refresh_cycles(
    dataset_dir: Path,
    cycle_names: Sequence[str] | None = None,
    *,
    render_publication: bool = True,
    render_coverage: bool = True,
) -> None:
    """Refresh only the selected cycle summaries and figures."""
    from .dataset_images import (
        _cycle_image_summary,
        _cycle_window,
        _sensor_coverage_intervals,
    )
    from .dataset_io import read_json, write_json
    from .dataset_metadata import now_iso, read_catalog, read_manifest
    from .visualization import render_cycle_publication, render_rgb_coverage_intervals

    catalog = read_catalog(dataset_dir)
    registry = read_json(dataset_dir / "channel_registry.json")
    if not isinstance(registry, dict):
        raise ValueError("channel_registry.json must contain an object")
    metadata = pd.read_parquet(dataset_dir / "image_metadata.parquet")
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
        frame = pd.read_parquet(dataset_dir / str(assets["parquet"]))
        image_summary, intervals = _cycle_image_summary(
            dataset_dir, cycle_name, frame, metadata, registry
        )
        record["image"] = image_summary
        record.setdefault("data", {})["processed_row_count"] = int(len(frame))
        if render_publication:
            render_cycle_publication(
                frame, record, dataset_dir / str(assets["publication"])
            )
        if render_coverage:
            start, end = _cycle_window(frame)
            render_rgb_coverage_intervals(
                cycle_name,
                start,
                end,
                intervals,
                dataset_dir / str(assets["rgb_coverage"]),
                sensor_intervals=_sensor_coverage_intervals(frame, registry),
            )
    write_json(catalog, dataset_dir / "cycle_catalog.json")
    manifest = read_manifest(dataset_dir)
    manifest["updated_at"] = now_iso()
    write_json(manifest, dataset_dir / "dataset_manifest.json")


def review_cycle(
    dataset_dir: Path,
    cycle_name: str,
    *,
    status: str,
    reason: str | None = None,
) -> Path:
    """Update one user-controlled Dataset status and redraw that cycle."""
    allowed = {"valid", "partial", "incomplete", "invalid"}
    if status not in allowed:
        raise ValueError(f"invalid Dataset status: {status}")
    from .dataset_io import write_json
    from .dataset_metadata import read_catalog

    catalog = read_catalog(dataset_dir)
    for record in catalog["cycles"]:
        if isinstance(record, dict) and record.get("cycle_name") == cycle_name:
            record["status"] = status
            record["status_reason"] = reason
            write_json(catalog, dataset_dir / "cycle_catalog.json")
            _refresh_cycles(
                dataset_dir,
                [cycle_name],
                render_publication=True,
                render_coverage=False,
            )
            return dataset_dir
    raise KeyError(f"unknown cycle: {cycle_name}")


def edit_dataset(
    dataset_dir: Path,
    *,
    baseline_seconds: int | None = None,
    recovery_seconds: int | None = None,
    recovery_end_by: str | None = None,
    camera_renames: Sequence[str] = (),
) -> Path:
    """Apply scientific or camera-role edits directly to the Dataset."""
    if recovery_seconds is not None and recovery_end_by is not None:
        raise ValueError("--recovery-seconds and --recovery-end-by are mutually exclusive")
    if (
        baseline_seconds is None
        and recovery_seconds is None
        and recovery_end_by is None
        and not camera_renames
    ):
        raise ValueError("dataset edit requires at least one edit")

    from .dataset_edit import rename_camera_role
    from .dataset_metadata import read_catalog

    renames = _camera_rename_mapping(camera_renames)
    catalog = read_catalog(dataset_dir)
    changed_cycles: set[str] = set()
    render_publication = False
    render_coverage = False
    if _has_scientific_edit(baseline_seconds, recovery_seconds, recovery_end_by):
        changed_cycles, render_publication, render_coverage = _apply_scientific_edits(
            dataset_dir,
            catalog,
            baseline_seconds=baseline_seconds,
            recovery_seconds=recovery_seconds,
            recovery_end_by=recovery_end_by,
        )

    if renames:
        camera_changed = rename_camera_role(dataset_dir, renames)
        changed_cycles.update(camera_changed)
        render_coverage = True

    if changed_cycles:
        _refresh_cycles(
            dataset_dir,
            sorted(changed_cycles),
            render_publication=render_publication,
            render_coverage=render_coverage,
        )
    return dataset_dir


def _camera_rename_mapping(expressions: Sequence[str]) -> dict[str, str]:
    renames: dict[str, str] = {}
    for expression in expressions:
        if "=" not in expression:
            raise ValueError(f"camera rename must be OLD=NEW: {expression}")
        old, new = expression.split("=", 1)
        renames[old] = new
    return renames


def _has_scientific_edit(
    baseline_seconds: int | None,
    recovery_seconds: int | None,
    recovery_end_by: str | None,
) -> bool:
    return (
        baseline_seconds is not None
        or recovery_seconds is not None
        or recovery_end_by is not None
    )


def _apply_scientific_edits(
    dataset_dir: Path,
    catalog: dict[str, Any],
    *,
    baseline_seconds: int | None,
    recovery_seconds: int | None,
    recovery_end_by: str | None,
) -> tuple[set[str], bool, bool]:
    from .dataset_io import read_json, write_json, write_parquet

    registry_value = read_json(dataset_dir / "channel_registry.json")
    if not isinstance(registry_value, dict):
        raise ValueError("channel_registry.json must contain an object")
    registry = registry_value
    recovery_edit = recovery_seconds is not None or recovery_end_by is not None
    metadata = (
        pd.read_parquet(dataset_dir / "image_metadata.parquet")
        if recovery_edit
        else None
    )
    changed_cycles: set[str] = set()
    for record in catalog["cycles"]:
        if not isinstance(record, dict):
            continue
        cycle_name = str(record["cycle_name"])
        _apply_scientific_edit_to_cycle(
            dataset_dir,
            record,
            registry,
            metadata,
            baseline_seconds=baseline_seconds,
            recovery_seconds=recovery_seconds,
            recovery_edit=recovery_edit,
        )
        changed_cycles.add(cycle_name)

    if metadata is not None:
        write_parquet(metadata, dataset_dir / "image_metadata.parquet")
    write_json(registry, dataset_dir / "channel_registry.json")
    write_json(catalog, dataset_dir / "cycle_catalog.json")
    return changed_cycles, True, recovery_edit


def _apply_scientific_edit_to_cycle(
    dataset_dir: Path,
    record: dict[str, Any],
    registry: dict[str, Any],
    metadata: pd.DataFrame | None,
    *,
    baseline_seconds: int | None,
    recovery_seconds: int | None,
    recovery_edit: bool,
) -> None:
    from .dataset_edit import apply_baseline, apply_recovery
    from .dataset_io import write_csv, write_parquet

    cycle_name = str(record["cycle_name"])
    assets = record.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError(f"cycle assets are missing: {cycle_name}")
    processed = pd.read_parquet(dataset_dir / str(assets["parquet"]))
    if recovery_edit:
        if metadata is None:
            raise ValueError("recovery edit requires image metadata")
        original = pd.read_csv(dataset_dir / str(assets["original_csv"]))
        metadata_cycle = metadata.loc[
            metadata["cycle_name"].astype(str).eq(cycle_name)
        ].copy()
        original, processed, metadata_cycle = apply_recovery(
            original,
            processed,
            metadata_cycle,
            record,
            registry,
            mode="seconds" if recovery_seconds is not None else "ts-minus",
            seconds=recovery_seconds,
        )
        mask = metadata["cycle_name"].astype(str).eq(cycle_name)
        metadata.loc[mask, :] = metadata_cycle.reindex(
            columns=metadata.columns
        ).to_numpy()
        write_csv(original, dataset_dir / str(assets["original_csv"]))

    if baseline_seconds is not None:
        processed = apply_baseline(
            processed,
            record,
            registry,
            seconds=baseline_seconds,
        )

    write_parquet(processed, dataset_dir / str(assets["parquet"]))
    write_csv(processed, dataset_dir / str(assets["csv"]))


def refresh_dataset(dataset_dir: Path) -> Path:
    """Refresh current images, statistics, and both figure families."""
    _refresh_cycles(dataset_dir)
    return dataset_dir


def render_dataset(
    dataset_dir: Path,
    cycle_name: str,
    *,
    publication: bool = True,
    coverage: bool = True,
) -> Path:
    """Render selected final assets without reading any source directory."""
    from .dataset_metadata import read_catalog

    catalog = read_catalog(dataset_dir)
    if not any(
        isinstance(record, Mapping)
        and str(record.get("cycle_name")) == cycle_name
        for record in catalog["cycles"]
    ):
        raise KeyError(f"unknown cycle: {cycle_name}")
    _refresh_cycles(
        dataset_dir,
        [cycle_name],
        render_publication=publication,
        render_coverage=coverage,
    )
    return dataset_dir
