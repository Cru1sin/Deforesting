"""Build and manage the self-contained Cycle Dataset schema 3.

The module owns orchestration only; scientific transforms remain in Prepare/Process
and image, metadata, IO, and validation concerns live in their focused modules.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .config import find_project_root

DATASET_SCHEMA_VERSION = 3
DATASET_ID = "frost_cycle_dataset"
CYCLE_NAME_WIDTH = 6
CycleKey = tuple[str, str]


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
class _DateBuild:
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
    target = Path(dataset_dir).resolve() if dataset_dir is not None else project_root / "dataset"
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
                if isinstance(item, Mapping) and str(item.get("experiment_id")) == experiment_id
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

        build = _build_date(input_path, config)
        _append_build(target, build)
        return target

    build = _build_date(input_path, config)
    _materialize_builds(target, [build])
    return target


def rebuild_dataset(input_dirs: Sequence[Path], dataset_dir: Path | None = None) -> Path:
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
    target = Path(dataset_dir).resolve() if dataset_dir is not None else project_root / "dataset"

    if target.exists():
        import shutil

        shutil.rmtree(target)
    builds = [
        _build_date(path, _load_config_for_input(path, project_root)) for path in inputs
    ]
    _materialize_builds(target, builds)
    return target


def update_cycle_columns(
    dataset_dir: Path, updates: Mapping[str, pd.DataFrame]
) -> None:
    """Add or replace timestamp-aligned columns in existing Processed cycles."""
    from .dataset_io import write_csv, write_parquet

    for cycle_name, update in updates.items():
        parquet_path = dataset_dir / "cycles" / f"{cycle_name}.parquet"
        csv_path = parquet_path.with_suffix(".csv")
        frame = pd.read_parquet(parquet_path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        aligned = update.copy()
        aligned["timestamp"] = pd.to_datetime(aligned["timestamp"])
        frame = frame.set_index("timestamp")
        aligned = aligned.set_index("timestamp")
        for column in aligned:
            frame[column] = aligned[column].reindex(frame.index)
        result = frame.reset_index()
        write_parquet(result, parquet_path)
        write_csv(result, csv_path)


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
        prepared_times["timestamp"] = pd.to_datetime(prepared_times["timestamp"], errors="coerce")
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
        raw_start = pd.to_datetime(cast(Any, row.get("segment_start")), errors="coerce")
        start = pd.Timestamp(raw_start) if not pd.isna(raw_start) else prepared_starts.get(key)
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
    return replace(config, input_dir=input_path.resolve())


def _build_date(input_path: Path, config: Any) -> _DateBuild:
    from .channels import load_channels
    from .prepare import prepare
    from .process import process
    from .validation import validate_prepared, validate_processed

    channels = load_channels(config.channels_path)
    prepared, initial_summary = prepare(config, channels)
    validate_prepared(prepared, initial_summary)
    processed, final_summary = process(prepared, initial_summary, config, channels)
    validate_processed(processed, final_summary)
    return _DateBuild(
        input_dir=input_path.resolve(),
        config=config,
        channels=channels,
        prepared=prepared,
        summary=final_summary,
        processed=processed,
    )


def _materialize_cycle(
    dataset_dir: Path,
    build: _DateBuild,
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
    from .dataset_schema import build_processed_frame, export_original_frame
    from .visualization import render_cycle_publication, render_rgb_coverage_intervals

    original = export_original_frame(
        build.prepared.loc[
            build.prepared["experiment_id"].astype(str).eq(key[0])
            & build.prepared["cycle_id"].astype(str).eq(key[1])
        ]
    ).reindex(columns=list(original_columns))
    raw = build.processed.loc[
        build.processed["experiment_id"].astype(str).eq(key[0])
        & build.processed["cycle_id"].astype(str).eq(key[1])
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
        image_summary={"image_count": 0},
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
        dataset_dir,
        cycle_name,
        canonical,
        metadata_result,
        registry,
        getattr(build.config, "camera_roles", {}),
    )
    record["image"] = image_summary
    render_cycle_publication(canonical, record, dataset_dir / assets["publication"])
    start, end = _cycle_window(canonical)
    render_rgb_coverage_intervals(
        cycle_name,
        start,
        end,
        intervals,
        dataset_dir / assets["rgb_coverage"],
        sensor_intervals=_sensor_coverage_intervals(canonical, registry),
    )
    return record, metadata_result


def _materialize_builds(
    dataset_dir: Path,
    builds: Sequence[_DateBuild],
) -> None:
    from .dataset_images import collect_images, copy_image, image_metadata_frame
    from .dataset_io import write_json, write_parquet
    from .dataset_metadata import experiment_record
    from .dataset_schema import build_registry, merge_original_columns

    if not builds:
        raise ValueError("Dataset requires at least one date")
    for directory in ("cycles", "cycles_original", "images"):
        (dataset_dir / directory).mkdir(parents=True, exist_ok=True)
    summary = pd.concat([build.summary for build in builds], ignore_index=True)
    prepared = pd.concat([build.prepared for build in builds], ignore_index=True)
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
    processed_counts = processed_counts_by_key(builds)
    for key in names:
        if processed_counts.get(key, 0) <= 0:
            raise ValueError(f"cycle has Prepared rows but no Processed rows: {key}")

    registry = build_registry(builds)
    original_columns = merge_original_columns(builds)
    all_images: list[dict[str, object]] = []
    for build in builds:
        all_images.extend(
            collect_images(
                build.prepared,
                input_dir=build.input_dir,
                cycle_names=names,
            )
        )
    image_metadata = image_metadata_frame(all_images)
    for image in all_images:
        copy_image(image, dataset_dir)

    summary_lookup: dict[CycleKey, dict[str, Any]] = {
        (str(row["experiment_id"]), str(row["cycle_id"])): {
            str(key): value for key, value in row.items()
        }
        for row in summary.to_dict(orient="records")
    }
    builds_by_experiment = {
        str(build.config.experiment_id): build for build in builds
    }
    records: list[dict[str, Any]] = []
    for key, cycle_name in sorted(names.items(), key=lambda item: parse_cycle_name(item[1])):
        record, image_metadata = _materialize_cycle(
            dataset_dir,
            builds_by_experiment[key[0]],
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
            str(build.config.experiment_id),
            str(build.config.experiment_date),
            getattr(build.config, "camera_roles", {}),
        )
        for build in sorted(builds, key=lambda item: str(item.config.experiment_date))
    ]
    write_json(
        {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
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
    builds: Sequence[_DateBuild],
) -> dict[CycleKey, int]:
    counts: dict[CycleKey, int] = {}
    for build in builds:
        for values in build.processed[["experiment_id", "cycle_id"]].itertuples(
            index=False, name=None
        ):
            key = (str(values[0]), str(values[1]))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _append_build(  # noqa: C901
    dataset_dir: Path,
    build: _DateBuild,
) -> None:
    from .dataset_images import collect_images, copy_image, image_metadata_frame
    from .dataset_io import read_json, write_csv, write_json, write_parquet
    from .dataset_metadata import experiment_record, read_catalog, read_manifest
    from .dataset_schema import (
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
        build.summary,
        prepared=build.prepared,
        start_index=len(old_records) + 1,
    )
    old_registry = read_json(dataset_dir / "channel_registry.json")
    candidate = build_registry([build])
    merged_registry = merge_registries(old_registry, candidate)
    for setting_name in (
        "image_coverage",
        "baseline_seconds",
        "baseline_managed",
        "recovery_edit",
    ):
        if setting_name in old_registry:
            merged_registry[setting_name] = old_registry[setting_name]
        elif setting_name in candidate:
            merged_registry[setting_name] = candidate[setting_name]

    old_images = pd.read_parquet(dataset_dir / "image_metadata.parquet")
    new_images = collect_images(
        build.prepared,
        input_dir=build.input_dir,
        cycle_names=names,
    )
    for image in new_images:
        copy_image(image, dataset_dir)
    merged_images = pd.concat(
        [old_images, image_metadata_frame(new_images)],
        ignore_index=True,
    )
    original_columns: list[str] = []
    for record in old_records:
        old_path = dataset_dir / str(record["assets"]["original_csv"])
        for column in pd.read_csv(old_path, nrows=0).columns:
            if str(column) not in original_columns:
                original_columns.append(str(column))
    for column in merge_original_columns([build]):
        if column not in original_columns:
            original_columns.append(column)

    summary_lookup: dict[CycleKey, dict[str, Any]] = {
        (str(row["experiment_id"]), str(row["cycle_id"])): {
            str(key): value for key, value in row.items()
        }
        for row in build.summary.to_dict(orient="records")
    }
    new_records: list[dict[str, Any]] = []
    for key, cycle_name in sorted(names.items(), key=lambda item: parse_cycle_name(item[1])):
        record, merged_images = _materialize_cycle(
            dataset_dir,
            build,
            key,
            cycle_name,
            merged_registry,
            original_columns,
            merged_images,
            summary_lookup[key],
        )
        new_records.append(record)

    old_columns = [str(name) for name in old_registry.get("columns", [])]
    merged_columns = [str(name) for name in merged_registry.get("columns", [])]
    if merged_columns != old_columns:
        for record in old_records:
            name = str(record["cycle_name"])
            assets = record["assets"]
            old_frame = pd.read_parquet(dataset_dir / str(assets["parquet"]))
            scientific = old_frame.drop(
                columns=[
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
                str(build.config.experiment_id),
                str(build.config.experiment_date),
                getattr(build.config, "camera_roles", {}),
            ),
        },
    ]
    manifest["experiments"].sort(key=lambda value: str(value["experiment_date"]))
    write_json(manifest, dataset_dir / "dataset_manifest.json")
    write_json(catalog, dataset_dir / "cycle_catalog.json")


def _render_publication(dataset_dir: Path, record: Mapping[str, Any]) -> None:
    from .visualization import render_cycle_publication

    cycle_name = str(record["cycle_name"])
    assets = record.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError(f"cycle assets are missing: {cycle_name}")
    frame = pd.read_parquet(dataset_dir / str(assets["parquet"]))
    render_cycle_publication(
        frame,
        record,
        dataset_dir / str(assets["publication"]),
    )


def _render_coverage(
    dataset_dir: Path,
    record: Mapping[str, Any],
    metadata: pd.DataFrame,
    registry: Mapping[str, Any],
    camera_roles: Mapping[str, str],
) -> None:
    from .dataset_images import (
        _cycle_image_summary,
        _cycle_window,
        _sensor_coverage_intervals,
    )
    from .visualization import render_rgb_coverage_intervals

    cycle_name = str(record["cycle_name"])
    assets = record.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError(f"cycle assets are missing: {cycle_name}")
    frame = pd.read_parquet(dataset_dir / str(assets["parquet"]))
    _, intervals = _cycle_image_summary(
        dataset_dir, cycle_name, frame, metadata, registry, camera_roles
    )
    start, end = _cycle_window(frame)
    render_rgb_coverage_intervals(
        cycle_name,
        start,
        end,
        intervals,
        dataset_dir / str(assets["rgb_coverage"]),
        sensor_intervals=_sensor_coverage_intervals(frame, registry),
    )


def _refresh_cycle_record(
    dataset_dir: Path,
    record: dict[str, Any],
    metadata: pd.DataFrame,
    registry: Mapping[str, Any],
    camera_roles: Mapping[str, str],
) -> None:
    from .dataset_images import (
        _cycle_image_summary,
        _cycle_window,
        _sensor_coverage_intervals,
    )
    from .visualization import render_cycle_publication, render_rgb_coverage_intervals

    cycle_name = str(record["cycle_name"])
    assets = record.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError(f"cycle assets are missing: {cycle_name}")
    frame = pd.read_parquet(dataset_dir / str(assets["parquet"]))
    image_summary, intervals = _cycle_image_summary(
        dataset_dir, cycle_name, frame, metadata, registry, camera_roles
    )
    record["image"] = image_summary
    record.setdefault("data", {})["processed_row_count"] = int(len(frame))
    render_cycle_publication(
        frame,
        record,
        dataset_dir / str(assets["publication"]),
    )
    start, end = _cycle_window(frame)
    render_rgb_coverage_intervals(
        cycle_name,
        start,
        end,
        intervals,
        dataset_dir / str(assets["rgb_coverage"]),
        sensor_intervals=_sensor_coverage_intervals(frame, registry),
    )


def _refresh_all_cycles(dataset_dir: Path) -> None:
    """Refresh image facts and both figure families."""
    from .dataset_io import read_json
    from .dataset_metadata import read_catalog, read_manifest, write_catalog

    catalog = read_catalog(dataset_dir)
    registry = read_json(dataset_dir / "channel_registry.json")
    if not isinstance(registry, dict):
        raise ValueError("channel_registry.json must contain an object")
    metadata = pd.read_parquet(dataset_dir / "image_metadata.parquet")
    manifest = read_manifest(dataset_dir)
    for record in catalog["cycles"]:
        if isinstance(record, dict):
            _refresh_cycle_record(
                dataset_dir,
                record,
                metadata,
                registry,
                _experiment_camera_roles(manifest, str(record["experiment_id"])),
            )
    write_catalog(dataset_dir, catalog)


def review_cycle(
    dataset_dir: Path,
    cycle_name: str,
    *,
    status: str,
    reason: str | None = None,
) -> Path:
    """Update only the user-controlled status and its publication title."""
    allowed = {"valid", "partial", "incomplete", "invalid"}
    if status not in allowed:
        raise ValueError(f"invalid Dataset status: {status}")
    from .dataset_metadata import read_catalog, write_catalog

    catalog = read_catalog(dataset_dir)
    for record in catalog["cycles"]:
        if isinstance(record, dict) and record.get("cycle_name") == cycle_name:
            record["status"] = status
            record["status_reason"] = reason
            _render_publication(dataset_dir, record)
            write_catalog(dataset_dir, catalog)
            return dataset_dir
    raise KeyError(f"unknown cycle: {cycle_name}")


def edit_dataset(  # noqa: C901
    dataset_dir: Path,
    *,
    baseline_seconds: int | None = None,
    recovery_seconds: int | None = None,
    recovery_end_by: str | None = None,
) -> Path:
    """Apply baseline or recovery edits directly to the Dataset."""
    if recovery_seconds is not None and recovery_end_by is not None:
        raise ValueError("--recovery-seconds and --recovery-end-by are mutually exclusive")
    if (
        baseline_seconds is None
        and recovery_seconds is None
        and recovery_end_by is None
    ):
        raise ValueError("dataset edit requires at least one edit")

    from .dataset_edit import apply_baseline, apply_recovery
    from .dataset_images import (
        _cycle_image_summary,
        _cycle_window,
        _sensor_coverage_intervals,
    )
    from .dataset_io import read_json, write_csv, write_json, write_parquet
    from .dataset_metadata import read_catalog, read_manifest, write_catalog
    from .visualization import render_cycle_publication, render_rgb_coverage_intervals

    catalog = read_catalog(dataset_dir)
    registry = read_json(dataset_dir / "channel_registry.json")
    if not isinstance(registry, dict):
        raise ValueError("channel_registry.json must contain an object")
    recovery_edit = recovery_seconds is not None or recovery_end_by is not None
    metadata = pd.read_parquet(dataset_dir / "image_metadata.parquet") if recovery_edit else None
    manifest = read_manifest(dataset_dir) if recovery_edit else None

    for record in catalog["cycles"]:
        if not isinstance(record, dict):
            continue
        cycle_name = str(record["cycle_name"])
        assets = record["assets"]
        processed = pd.read_parquet(dataset_dir / str(assets["parquet"]))
        if recovery_edit:
            assert metadata is not None and manifest is not None
            original = pd.read_csv(dataset_dir / str(assets["original_csv"]))
            mask = metadata["cycle_name"].astype(str).eq(cycle_name)
            original, processed, cycle_metadata = apply_recovery(
                original,
                processed,
                metadata.loc[mask].copy(),
                record,
                registry,
                mode="seconds" if recovery_seconds is not None else "ts-minus",
                seconds=recovery_seconds,
            )
            if mask.any():
                metadata.loc[mask, "cycle_stage"] = cycle_metadata["cycle_stage"].to_numpy()
            write_csv(original, dataset_dir / str(assets["original_csv"]))
        if baseline_seconds is not None:
            processed = apply_baseline(processed, record, registry, seconds=baseline_seconds)
        write_parquet(processed, dataset_dir / str(assets["parquet"]))
        write_csv(processed, dataset_dir / str(assets["csv"]))
        render_cycle_publication(
            processed, record, dataset_dir / str(assets["publication"])
        )
        if recovery_edit:
            assert metadata is not None and manifest is not None
            roles = _experiment_camera_roles(manifest, str(record["experiment_id"]))
            image_summary, intervals = _cycle_image_summary(
                dataset_dir, cycle_name, processed, metadata, registry, roles
            )
            record["image"] = image_summary
            start, end = _cycle_window(processed)
            render_rgb_coverage_intervals(
                cycle_name,
                start,
                end,
                intervals,
                dataset_dir / str(assets["rgb_coverage"]),
                sensor_intervals=_sensor_coverage_intervals(processed, registry),
            )

    if metadata is not None and recovery_edit:
        write_parquet(metadata, dataset_dir / "image_metadata.parquet")
    write_json(registry, dataset_dir / "channel_registry.json")
    write_catalog(dataset_dir, catalog)
    return dataset_dir


def refresh_dataset(dataset_dir: Path) -> Path:
    """Refresh current images, statistics, and both figure families."""
    _refresh_all_cycles(dataset_dir)
    return dataset_dir


def render_dataset(
    dataset_dir: Path,
    cycle_name: str,
    *,
    publication: bool = True,
    coverage: bool = True,
) -> Path:
    """Render selected final assets without reading any source directory."""
    from .dataset_io import read_json
    from .dataset_metadata import read_catalog, read_manifest

    catalog = read_catalog(dataset_dir)
    if not any(
        isinstance(record, Mapping) and str(record.get("cycle_name")) == cycle_name
        for record in catalog["cycles"]
    ):
        raise KeyError(f"unknown cycle: {cycle_name}")
    record = next(
        record
        for record in catalog["cycles"]
        if isinstance(record, Mapping) and str(record.get("cycle_name")) == cycle_name
    )
    if publication:
        _render_publication(dataset_dir, record)
    if coverage:
        registry = read_json(dataset_dir / "channel_registry.json")
        if not isinstance(registry, dict):
            raise ValueError("channel_registry.json must contain an object")
        metadata = pd.read_parquet(dataset_dir / "image_metadata.parquet")
        roles = _experiment_camera_roles(
            read_manifest(dataset_dir), str(record["experiment_id"])
        )
        _render_coverage(dataset_dir, record, metadata, registry, roles)
    return dataset_dir


def _experiment_camera_roles(
    manifest: Mapping[str, Any], experiment_id: str
) -> dict[str, str]:
    experiment = next(
        item
        for item in manifest["experiments"]
        if isinstance(item, Mapping) and str(item.get("experiment_id")) == experiment_id
    )
    roles = experiment.get("camera_roles", {})
    if not isinstance(roles, Mapping):
        raise ValueError(f"camera_roles must be an object: {experiment_id}")
    return {str(key): str(value) for key, value in roles.items()}
