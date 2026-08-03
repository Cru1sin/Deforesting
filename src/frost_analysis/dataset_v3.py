"""Self-contained Cycle Dataset v3 publication and append workflow."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .channels import load_channels
from .config import load_config, resolved_config_sha256
from .dataset_io import write_atomic_json, write_atomic_parquet
from .dataset_registry import (
    canonical_frame,
    canonical_registry_hash,
    merge_registries,
    registry_from_frame,
)
from .io import (
    ensure_output_outside_input,
    optional_sha256,
    relative_posix_path,
    sha256_file,
)

V3_DATASET_SCHEMA_VERSION = 3
V3_DATASET_ID = "frost_cycle_dataset"
V3_CYCLE_NAME_WIDTH = 6
# Public v3 names.  The legacy ``dataset`` module keeps its v1 compatibility
# constants until the real-data v3 acceptance gate is complete.
DATASET_SCHEMA_VERSION = V3_DATASET_SCHEMA_VERSION
DATASET_ID = V3_DATASET_ID
CYCLE_NAME_WIDTH = V3_CYCLE_NAME_WIDTH
V3_DATASET_FIELDS = (
    "dataset_id",
    "dataset_schema_version",
    "dataset_cycle_index",
    "cycle_name",
    "cycle_uid",
)
_DATE_KEY_RE = re.compile(r"^\d{4}$")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class V3SourceRun:
    path: Path
    project_root: Path
    experiment_id: str
    experiment_date: str
    input_dir: Path
    prepared_path: Path
    processed_path: Path
    summary_path: Path
    manifest_path: Path
    manifest_sha256: str
    resolved_config_sha256: str
    input_inventory_sha256: str
    pipeline_commit: str | None
    manifest: dict[str, Any]


def resolve_project_root(start: Path | None = None) -> Path:
    """Resolve the repository root independently of the caller's cwd."""
    candidates = [(start or Path.cwd()).resolve(), Path(__file__).resolve().parent]
    for candidate in candidates:
        if candidate.is_file():
            candidate = candidate.parent
        for parent in (candidate, *candidate.parents):
            if (parent / "pyproject.toml").is_file():
                return parent
    raise FileNotFoundError("could not find project root containing pyproject.toml")


def make_image_id(
    cycle_uid: str, source_camera_id: str, source_relative_path: str
) -> str:
    """Create an image identity from immutable source facts, never frame order."""
    payload = "\0".join((cycle_uid, source_camera_id, source_relative_path)).encode("utf-8")
    return f"img_{hashlib.sha256(payload).hexdigest()[:16]}"


def source_fingerprint(
    experiment_id: str,
    experiment_date: str,
    run_manifest_sha256: str,
    input_inventory_sha256: str,
    canonical_registry_sha256: str,
) -> str:
    """Return the immutable source identity used for add no-op/conflict checks."""
    payload = {
        "experiment_id": experiment_id,
        "experiment_date": experiment_date,
        "run_manifest_sha256": run_manifest_sha256,
        "input_inventory_sha256": input_inventory_sha256,
        "canonical_registry_sha256": canonical_registry_sha256,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def add_dataset(input_dir: Path, dataset_dir: Path | None = None) -> Path:
    """Run or reuse one formal run and publish it into a v3 Dataset."""
    project_root = resolve_project_root()
    input_path = (input_dir if input_dir.is_absolute() else Path.cwd() / input_dir).resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_path}")
    date_key = input_path.name
    if _DATE_KEY_RE.fullmatch(date_key) is None:
        raise ValueError("dataset add input basename must be an MMDD date key")
    config_path = project_root / "configs" / f"{date_key}.yaml"
    run_dir = project_root / "outputs" / "runs" / date_key
    config = load_config(config_path)
    if not str(config.experiment_date).startswith("2026-"):
        raise ValueError("Dataset v3 currently accepts only 2026 experiment dates")
    if config.input_dir.resolve() != input_path:
        raise ValueError(
            "config input_dir does not match dataset add INPUT_DIR: "
            f"{config.input_dir} != {input_path}"
        )
    if not run_dir.exists():
        ensure_output_outside_input(run_dir, input_path)
        from .pipeline import run_pipeline

        run_pipeline(config_path, run_dir, overwrite=False)
    source = load_v3_source_run(run_dir)
    if source.input_dir != input_path:
        raise ValueError("formal run source input_dir does not match dataset add INPUT_DIR")
    current_config_hash = resolved_config_sha256(config)
    if current_config_hash != source.resolved_config_sha256:
        raise ValueError(
            "formal run resolved config differs from current config; "
            "rebuild the formal run explicitly"
        )
    return add_formal_run(source, dataset_dir or project_root / "dataset")


def add_formal_run(source: V3SourceRun | Path, dataset_dir: Path) -> Path:
    """Publish an already-created formal run; kept internal for tests and migration."""
    source_run = load_v3_source_run(source) if isinstance(source, Path) else source
    dataset_dir = dataset_dir.resolve()
    ensure_output_outside_input(dataset_dir, source_run.input_dir)
    if not dataset_dir.exists():
        return _build_dataset(source_run, dataset_dir)
    from .dataset_validation_v3 import validate_v3_dataset

    validate_v3_dataset(
        dataset_dir,
        verify_image_hashes=False,
        verify_asset_hashes=False,
    )
    return _append_dataset(source_run, dataset_dir)


def load_v3_source_run(run_path: Path) -> V3SourceRun:  # noqa: C901
    """Read and verify a frozen formal run without scanning its raw input tree."""
    run_dir = run_path.resolve()
    manifest_path = run_dir / "manifest.json"
    if not run_dir.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"formal run is missing manifest.json: {run_dir}")
    manifest = _read_json(manifest_path)
    project_root = resolve_project_root(run_dir)
    experiment_id = str(manifest.get("experiment_id", ""))
    experiment_date = str(manifest.get("experiment_date", ""))[:10]
    if not experiment_id or not re.fullmatch(r"2026-\d{2}-\d{2}", experiment_date):
        raise ValueError(f"formal run has invalid experiment identity: {run_dir}")
    resolved_config = manifest.get("resolved_config")
    provenance = manifest.get("config_provenance")
    if not isinstance(resolved_config, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError(f"formal run lacks configuration provenance: {run_dir}")
    input_value = resolved_config.get("input_dir")
    if not isinstance(input_value, str):
        raise ValueError(f"formal run lacks resolved input_dir: {run_dir}")
    input_dir = Path(input_value)
    if not input_dir.is_absolute():
        input_dir = project_root / input_dir
    input_dir = input_dir.resolve()
    resolved_hash = str(provenance.get("resolved_config_sha256", ""))
    inventory_hash = str(
        manifest.get("input_inventory_sha256")
        or manifest.get("prepare_summary", {}).get("input_inventory_sha256", "")
    )
    if not resolved_hash or not inventory_hash:
        raise ValueError("formal run must record resolved config and input inventory hashes")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"formal run lacks outputs: {run_dir}")
    required = {
        "prepared_data": "prepared_data.parquet",
        "processed_data": "processed_data.parquet",
        "cycle_summary": "cycle_summary.csv",
        "candidate_channel_evidence": "candidate_channel_evidence.csv",
    }
    paths: dict[str, Path] = {}
    for key, expected_name in required.items():
        value = outputs.get(key)
        if not isinstance(value, str) or Path(value).name != expected_name:
            raise ValueError(f"formal run output {key} is invalid: {run_dir}")
        path = (run_dir / value).resolve()
        if not path.is_file() or run_dir not in path.parents:
            raise FileNotFoundError(f"formal run is missing {expected_name}: {run_dir}")
        paths[key] = path
    expected_hashes = manifest.get("output_sha256")
    if not isinstance(expected_hashes, Mapping):
        raise ValueError("formal run must record output hashes")
    for key, path in paths.items():
        expected = expected_hashes.get(key)
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"formal run output hash is missing: {key}")
        if expected != sha256_file(path):
            raise ValueError(f"formal run output SHA mismatch: {path}")
    git_value = manifest.get("git_commit")
    return V3SourceRun(
        path=run_dir,
        project_root=project_root,
        experiment_id=experiment_id,
        experiment_date=experiment_date,
        input_dir=input_dir,
        prepared_path=paths["prepared_data"],
        processed_path=paths["processed_data"],
        summary_path=paths["cycle_summary"],
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        resolved_config_sha256=resolved_hash,
        input_inventory_sha256=inventory_hash,
        pipeline_commit=None if git_value is None else str(git_value),
        manifest=manifest,
    )


def _build_dataset(source: V3SourceRun, dataset_dir: Path) -> Path:
    from .dataset_io import create_dataset_staging, publish_build
    from .dataset_validation_v3 import validate_v3_dataset

    staging_root, staging_dataset = create_dataset_staging(dataset_dir, kind="build")
    published = False
    try:
        prepared, processed, summary = _load_and_validate_source(source)
        candidate_registry = _source_registry(source, processed)
        registry_hash = canonical_registry_hash(candidate_registry)
        names = _assign_cycle_names(summary, start_index=1)
        materialized = _materialize_source(
            source,
            prepared,
            processed,
            summary,
            candidate_registry,
            names,
            staging_dataset,
        )
        _write_dataset_metadata(
            staging_dataset,
            registry=candidate_registry,
            registry_hash=registry_hash,
            source=source,
            cycle_records=materialized["cycle_records"],
            cycle_index=materialized["cycle_index"],
            image_metadata=materialized["image_metadata"],
            source_fingerprint_value=source_fingerprint(
                source.experiment_id,
                source.experiment_date,
                source.manifest_sha256,
                source.input_inventory_sha256,
                registry_hash,
            ),
        )
        _write_readme(staging_dataset)
        validate_v3_dataset(staging_dataset)
        from .dataset_manifest_v3 import refresh_manifest

        refresh_manifest(staging_dataset)
        validate_v3_dataset(staging_dataset)
        publish_build(staging_root, staging_dataset, dataset_dir)
        published = True
        validate_v3_dataset(dataset_dir)
        return dataset_dir
    except Exception:
        if published:
            shutil.rmtree(dataset_dir, ignore_errors=True)
        raise
    finally:
        if not published:
            shutil.rmtree(staging_root, ignore_errors=True)


def _append_dataset(source: V3SourceRun, dataset_dir: Path) -> Path:
    from .dataset_io import (
        backup_v3_metadata,
        cleanup_append_staging,
        commit_v3_append_files,
        create_dataset_staging,
        rollback_v3_append,
    )
    from .dataset_validation_v3 import validate_v3_dataset

    manifest = _read_json(dataset_dir / "dataset_manifest.json")
    old_index = pd.read_parquet(dataset_dir / "cycle_index.parquet")
    old_images = pd.read_parquet(dataset_dir / "image_metadata.parquet")
    old_registry = _read_json(dataset_dir / "channel_registry.json")
    prepared, processed, summary = _load_and_validate_source(source)
    candidate = _source_registry(source, processed)
    source_registry_hash = canonical_registry_hash(candidate)
    merged_registry = merge_registries(old_registry, candidate)
    merged_hash = canonical_registry_hash(merged_registry)
    fingerprint = source_fingerprint(
        source.experiment_id,
        source.experiment_date,
        source.manifest_sha256,
        source.input_inventory_sha256,
        source_registry_hash,
    )
    source_records = manifest.get("source_experiments", [])
    if not isinstance(source_records, list):
        raise ValueError("dataset manifest source_experiments is invalid")
    existing = next(
        (
            item
            for item in source_records
            if isinstance(item, Mapping)
            and item.get("experiment_id") == source.experiment_id
        ),
        None,
    )
    if existing is not None:
        if str(existing.get("source_fingerprint")) == fingerprint:
            return dataset_dir
        raise ValueError(f"source fingerprint conflict for {source.experiment_id}")
    if old_index.empty:
        last_date = None
    else:
        last_date = str(old_index["experiment_date"].astype(str).max())[:10]
    if last_date is not None and source.experiment_date <= last_date:
        raise ValueError(
            "historical or same-date append is not supported; rebuild a new Dataset "
            "with dataset add in date order"
        )
    names = _assign_cycle_names(summary, start_index=_next_cycle_index(old_index))
    staging_root, staging_dataset = create_dataset_staging(dataset_dir, kind="append")
    moved: list[Path] = []
    backup: Path | None = None
    committed = False
    try:
        materialized = _materialize_source(
            source,
            prepared,
            processed,
            summary,
            merged_registry,
            names,
            staging_dataset,
        )
        merged_index = pd.concat([old_index, materialized["cycle_index"]], ignore_index=True)
        merged_images = pd.concat([old_images, materialized["image_metadata"]], ignore_index=True)
        # A registry expansion must materialize historical cycle files before
        # Manifest hashes are written.
        replaced = _rewrite_historical_cycles_if_needed(
            dataset_dir,
            staging_dataset,
            old_registry,
            merged_registry,
            old_index,
        )
        _write_dataset_metadata(
            staging_dataset,
            registry=merged_registry,
            registry_hash=merged_hash,
            source=source,
            cycle_records=materialized["cycle_records"],
            cycle_index=merged_index,
            image_metadata=merged_images,
            source_fingerprint_value=fingerprint,
            existing_manifest=manifest,
        )
        _write_readme(staging_dataset)
        relative_new = _new_asset_paths(
            staging_dataset, materialized["cycle_index"], materialized["image_metadata"]
        )
        backup = backup_v3_metadata(dataset_dir, staging_root, replaced)
        moved = commit_v3_append_files(
            staging_dataset,
            dataset_dir,
            relative_new,
            replaced,
            moved_files=moved,
        )
        committed = True
        selected_assets = set(relative_new) | set(replaced)
        validate_v3_dataset(
            dataset_dir,
            verify_image_hashes=True,
            verify_asset_hashes=True,
            selected_assets=selected_assets,
        )
        from .dataset_manifest_v3 import refresh_manifest

        refresh_manifest(dataset_dir)
        validate_v3_dataset(
            dataset_dir,
            verify_image_hashes=True,
            verify_asset_hashes=True,
            selected_assets=selected_assets,
        )
        return dataset_dir
    except Exception as append_error:
        if committed or moved or backup is not None:
            try:
                rollback_v3_append(dataset_dir, backup, moved)
                validate_v3_dataset(
                    dataset_dir,
                    verify_image_hashes=False,
                    verify_asset_hashes=False,
                )
            except Exception as rollback_error:
                raise RuntimeError(
                    "append failed and rollback validation also failed: "
                    f"{rollback_error}"
                ) from append_error
        raise
    finally:
        cleanup_append_staging(staging_root)


def _load_and_validate_source(  # noqa: C901
    source: V3SourceRun,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from .validation import validate_prepared, validate_processed

    prepared = pd.read_parquet(source.prepared_path)
    processed = pd.read_parquet(source.processed_path)
    summary = _read_summary(source.summary_path)
    evidence_path = source.path / "candidate_channel_evidence.csv"
    evidence = pd.read_csv(evidence_path)
    validate_prepared(prepared, summary)
    validate_processed(processed, summary)
    counts = source.manifest.get("output_row_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("formal run must record output row counts")
    expected_counts = {
        "prepared_data": len(prepared),
        "processed_data": len(processed),
        "cycle_summary": len(summary),
        "candidate_channel_evidence": len(evidence),
    }
    for key, actual in expected_counts.items():
        expected = counts.get(key)
        if expected is None or int(expected) != actual:
            raise ValueError(f"formal run output row count mismatch: {key}")
    from .validation import validate_analysis

    validate_analysis(evidence)
    prepared_roles = _validate_image_triples(prepared, "Prepared")
    processed_roles = _validate_image_triples(processed, "Processed")
    if prepared_roles != processed_roles:
        raise ValueError("Prepared and Processed image roles do not match")
    for frame, name in ((prepared, "Prepared"), (processed, "Processed")):
        if not _identity_matches(frame, source):
            raise ValueError(f"{name} identity disagrees with formal run manifest")
    for raw_row in summary.to_dict(orient="records"):
        row = {str(key): value for key, value in raw_row.items()}
        if (
            str(row.get("experiment_id", "")) != source.experiment_id
            or str(row.get("experiment_date", ""))[:10] != source.experiment_date
        ):
            raise ValueError("Summary identity disagrees with formal run manifest")
        cycle_key = (str(row["experiment_id"]), str(row["cycle_id"]))
        prepared_count = int(
            (
                (prepared["experiment_id"].astype(str) == cycle_key[0])
                & (prepared["cycle_id"].astype(str) == cycle_key[1])
            ).sum()
        )
        processed_count = int(
            (
                (processed["experiment_id"].astype(str) == cycle_key[0])
                & (processed["cycle_id"].astype(str) == cycle_key[1])
            ).sum()
        )
        if prepared_count == 0:
            raise ValueError(f"Summary cycle has no Prepared rows: {cycle_key}")
        if processed_count == 0:
            raise ValueError(f"Summary cycle has no Processed rows: {cycle_key}")
    return prepared, processed, _sort_summary(summary)


def _source_registry(source: V3SourceRun, processed: pd.DataFrame) -> dict[str, Any]:
    channels: dict[str, dict[str, Any]] = {}
    analysis_settings: Mapping[str, Any] = {}
    config_path = _manifest_config_path(source)
    if config_path is not None and config_path.is_file():
        config = load_config(config_path)
        if resolved_config_sha256(config) != source.resolved_config_sha256:
            raise ValueError(
                "formal run resolved config differs from current config; "
                "rebuild the formal run explicitly"
            )
        provenance = source.manifest.get("config_provenance")
        expected_channels_hash = (
            provenance.get("channels_sha256")
            if isinstance(provenance, Mapping)
            else None
        )
        actual_channels_hash = optional_sha256(config.channels_path)
        if (
            isinstance(expected_channels_hash, str)
            and expected_channels_hash
            and expected_channels_hash != actual_channels_hash
        ):
            raise ValueError(
                "formal run channel configuration differs from current channels; "
                "rebuild the formal run explicitly"
            )
        channels = load_channels(config.channels_path)
        analysis_settings = dataclasses.asdict(config.analysis)
    registry = registry_from_frame(
        processed,
        channels,
        analysis_settings=analysis_settings,
        resample_interval_seconds=(
            int(config.process.resample_interval_seconds)
            if config_path is not None and config_path.is_file()
            else 10
        ),
    )
    registry["canonical_hash"] = canonical_registry_hash(registry)
    return registry


def _manifest_config_path(source: V3SourceRun) -> Path | None:
    provenance = source.manifest.get("config_provenance")
    if not isinstance(provenance, Mapping):
        return None
    value = provenance.get("experiment_config_path")
    if not isinstance(value, str):
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else source.project_root / path


def _materialize_source(
    source: V3SourceRun,
    prepared: pd.DataFrame,
    processed: pd.DataFrame,
    summary: pd.DataFrame,
    registry: Mapping[str, Any],
    names: Mapping[tuple[str, str], str],
    staging_dataset: Path,
) -> dict[str, Any]:
    from .dataset_coverage_v3 import render_rgb_coverage
    from .visualization import render_cycle_publication

    records = _collect_images(prepared, source, names)
    for record in records:
        _copy_image(record, staging_dataset)
    metadata = _image_metadata_frame(records)
    canonical = canonical_frame(processed, registry)
    cycle_records: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for raw_row in summary.to_dict(orient="records"):
        row = {str(key): value for key, value in raw_row.items()}
        key = (str(row["experiment_id"]), str(row["cycle_id"]))
        cycle_name = names[key]
        cycle_uid = _cycle_uid(*key)
        cycle_frame = canonical.loc[
            canonical["experiment_id"].astype(str).eq(key[0])
            & canonical["cycle_id"].astype(str).eq(key[1])
        ].copy()
        if cycle_frame.empty:
            raise ValueError(f"cannot publish empty Processed cycle: {key}")
        cycle_frame = _add_dataset_fields(
            cycle_frame,
            cycle_index=_cycle_index_from_name(cycle_name),
            cycle_name=cycle_name,
            cycle_uid=cycle_uid,
        )
        parquet_path = staging_dataset / "cycles" / f"{cycle_name}.parquet"
        csv_path = staging_dataset / "cycles" / f"{cycle_name}.csv"
        write_atomic_parquet(cycle_frame, parquet_path)
        cycle_frame.to_csv(csv_path, index=False)
        cycle_images = [item for item in records if item["cycle_uid"] == cycle_uid]
        record = _cycle_record(row, cycle_name, cycle_uid, cycle_frame, cycle_images)
        publication_path = staging_dataset / "cycles" / f"{cycle_name}.png"
        coverage_path = staging_dataset / "cycles" / f"{cycle_name}_rgb_coverage.png"
        render_cycle_publication(cycle_frame, record, publication_path)
        render_rgb_coverage(
            cycle_frame,
            _image_records_frame(cycle_images),
            record,
            coverage_path,
            registry=registry,
        )
        record.update(
            {
                "data_path": f"cycles/{cycle_name}.parquet",
                "csv_path": f"cycles/{cycle_name}.csv",
                "publication_path": f"cycles/{cycle_name}.png",
                "rgb_coverage_path": f"cycles/{cycle_name}_rgb_coverage.png",
                "asset_sha256": {
                    "parquet": sha256_file(parquet_path),
                    "csv": sha256_file(csv_path),
                    "publication": sha256_file(publication_path),
                    "rgb_coverage": sha256_file(coverage_path),
                },
            }
        )
        cycle_records.append(record)
        index_row = {
                "dataset_cycle_index": _cycle_index_from_name(cycle_name),
                "cycle_name": cycle_name,
                "cycle_uid": cycle_uid,
                "experiment_id": key[0],
                "experiment_date": str(row["experiment_date"])[:10],
                "cycle_id": key[1],
                "cycle_status": row.get("cycle_status"),
                "cycle_status_reason": row.get("cycle_status_reason"),
                "baseline_status": row.get("baseline_status"),
                "baseline_failure_reason": row.get("baseline_failure_reason"),
                "published": True,
                "data_path": record["data_path"],
                "csv_path": record["csv_path"],
                "publication_path": record["publication_path"],
                "rgb_coverage_path": record["rgb_coverage_path"],
                "processed_row_count": len(cycle_frame),
                "image_count": len(cycle_images),
            }
        for summary_field in (
            "heating_start",
            "stable_heating_start",
            "defrost_start",
            "defrost_end",
            "baseline_start",
            "baseline_end",
        ):
            if summary_field in row:
                index_row[summary_field] = row.get(summary_field)
        index_rows.append(index_row)
    return {
        "cycle_records": cycle_records,
        "cycle_index": pd.DataFrame(index_rows),
        "image_metadata": metadata,
    }


def _collect_images(  # noqa: C901
    prepared: pd.DataFrame, source: V3SourceRun, names: Mapping[tuple[str, str], str]
) -> list[dict[str, Any]]:
    from .images import image_columns, image_roles

    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_metadata: dict[Path, tuple[str, int]] = {}
    for values in prepared.to_dict(orient="records"):
        key = (str(values["experiment_id"]), str(values["cycle_id"]))
        cycle_name = names.get(key)
        if cycle_name is None:
            continue
        cycle_uid = _cycle_uid(*key)
        for role in image_roles(prepared):
            path_column, time_column, offset_column = image_columns(role)
            raw_path = values.get(path_column)
            if pd.isna(raw_path):
                if not pd.isna(values.get(time_column)) or not pd.isna(values.get(offset_column)):
                    raise ValueError(f"incomplete image match for {path_column}")
                continue
            raw_time = values.get(time_column)
            raw_offset = values.get(offset_column)
            if pd.isna(raw_time) or pd.isna(raw_offset):
                raise ValueError(f"incomplete image match for {path_column}")
            relative = _safe_source_relative(str(raw_path))
            source_path = (source.input_dir / relative).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"matched source image does not exist: {source_path}")
            source_camera_id = relative.split("/", 1)[0]
            candidate_key = (cycle_uid, source_camera_id, relative)
            record = {
                "cycle_uid": cycle_uid,
                "cycle_name": cycle_name,
                "source_camera_id": source_camera_id,
                "initial_camera_slot": _slot_name(source_camera_id),
                "image_time": pd.Timestamp(raw_time),
                "matched_timestamp": pd.Timestamp(values["timestamp"]),
                "offset_seconds": float(raw_offset),
                "cycle_stage": str(values.get("cycle_stage", "")),
                "source_relative_path": relative,
                "source_path": source_path,
            }
            existing = candidates.get(candidate_key)
            if existing is not None:
                for field in ("image_time", "matched_timestamp", "offset_seconds"):
                    if existing[field] != record[field]:
                        raise ValueError(f"inconsistent image match for {relative}")
                continue
            if source_path not in source_metadata:
                source_metadata[source_path] = (
                    sha256_file(source_path),
                    source_path.stat().st_size,
                )
            record["sha256"], record["file_size_bytes"] = source_metadata[source_path]
            candidates[candidate_key] = record
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in candidates.values():
        grouped.setdefault(
            (str(record["cycle_uid"]), str(record["source_camera_id"])), []
        ).append(record)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for group_key in sorted(grouped):
        group = sorted(
            grouped[group_key],
            key=lambda item: (item["image_time"], item["source_relative_path"]),
        )
        for frame_index, record in enumerate(group, start=1):
            image_id = make_image_id(
                str(record["cycle_uid"]),
                str(record["source_camera_id"]),
                str(record["source_relative_path"]),
            )
            if image_id in seen_ids:
                raise ValueError(f"duplicate generated image_id: {image_id}")
            seen_ids.add(image_id)
            suffix = Path(str(record["source_relative_path"])).suffix.lower()
            record.update(
                {
                    "image_id": image_id,
                    "frame_index": frame_index,
                    "image_path": (
                        f"images/{record['cycle_name']}/"
                        f"{record['initial_camera_slot']}/{image_id}{suffix}"
                    ),
                }
            )
            records.append(record)
    return records


def _image_metadata_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "image_id",
        "cycle_uid",
        "cycle_name",
        "frame_index",
        "source_camera_id",
        "initial_camera_slot",
        "image_time",
        "matched_timestamp",
        "offset_seconds",
        "cycle_stage",
        "source_relative_path",
        "file_size_bytes",
        "sha256",
    ]
    return pd.DataFrame(
        [{name: record[name] for name in columns} for record in records],
        columns=columns,
    )


def _copy_image(record: Mapping[str, Any], staging_dataset: Path) -> None:
    destination = staging_dataset / str(record["image_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(str(record["source_path"])), destination)
    if sha256_file(destination) != str(record["sha256"]):
        raise ValueError(f"copied image SHA mismatch: {destination}")


def _image_records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    columns = ["image_id", "camera_role", "image_time", "matched_timestamp", "offset_seconds"]
    rows = [{**record, "camera_role": str(record["initial_camera_slot"])} for record in records]
    return pd.DataFrame(rows, columns=columns)


def _cycle_record(
    summary_row: Mapping[str, Any],
    cycle_name: str,
    cycle_uid: str,
    cycle_frame: pd.DataFrame,
    images: list[dict[str, Any]],
) -> dict[str, Any]:
    timestamps = pd.to_datetime(cycle_frame["timestamp"], errors="coerce").dropna()
    record = {
        "cycle_name": cycle_name,
        "cycle_uid": cycle_uid,
        "experiment_id": str(summary_row["experiment_id"]),
        "experiment_date": str(summary_row["experiment_date"])[:10],
        "cycle_id": str(summary_row["cycle_id"]),
        "cycle_status": summary_row.get("cycle_status"),
        "cycle_status_reason": _none_if_na(summary_row.get("cycle_status_reason")),
        "baseline_status": _none_if_na(summary_row.get("baseline_status")),
        "baseline_failure_reason": _none_if_na(summary_row.get("baseline_failure_reason")),
        "start_time": timestamps.min().isoformat() if not timestamps.empty else None,
        "end_time": timestamps.max().isoformat() if not timestamps.empty else None,
        "row_count": int(len(cycle_frame)),
        "image_count": int(len(images)),
        "assessment": _assessment(summary_row),
    }
    for summary_field in (
        "heating_start",
        "stable_heating_start",
        "defrost_start",
        "defrost_end",
        "baseline_start",
        "baseline_end",
    ):
        if summary_field in summary_row:
            record[summary_field] = _json_value(summary_row.get(summary_field))
    return record


def _assessment(row: Mapping[str, Any]) -> dict[str, Any]:
    status = str(row.get("cycle_status", "invalid"))
    if status not in {"valid", "partial", "incomplete", "invalid"}:
        status = "invalid"
    reason = _none_if_na(row.get("cycle_status_reason"))
    return {
        "status": status,
        "reasons": [] if reason is None else [str(reason)],
        "note": None,
        "updated_at": datetime.now(UTC).isoformat(),
    }


def _write_dataset_metadata(
    staging_dataset: Path,
    *,
    registry: Mapping[str, Any],
    registry_hash: str,
    source: V3SourceRun,
    cycle_records: list[dict[str, Any]],
    cycle_index: pd.DataFrame,
    image_metadata: pd.DataFrame,
    source_fingerprint_value: str,
    existing_manifest: Mapping[str, Any] | None = None,
) -> None:
    write_atomic_parquet(cycle_index, staging_dataset / "cycle_index.parquet")
    write_atomic_parquet(image_metadata, staging_dataset / "image_metadata.parquet")
    registry_payload = dict(registry)
    registry_payload["canonical_hash"] = registry_hash
    write_atomic_json(registry_payload, staging_dataset / "channel_registry.json")
    old_sources = (
        []
        if existing_manifest is None
        else list(existing_manifest.get("source_experiments", []))
    )
    old_cycles = [] if existing_manifest is None else list(existing_manifest.get("cycles", []))
    all_cycles = old_cycles + cycle_records
    all_sources = old_sources + [_source_provenance(source, source_fingerprint_value)]
    for record in all_cycles:
        if not isinstance(record, dict):
            continue
        hashes = dict(record.get("asset_sha256", {}))
        for field, hash_key in (
            ("data_path", "parquet"),
            ("csv_path", "csv"),
            ("publication_path", "publication"),
            ("rgb_coverage_path", "rgb_coverage"),
        ):
            relative = record.get(field)
            if isinstance(relative, str):
                staged = staging_dataset / relative
                if staged.is_file():
                    hashes[hash_key] = sha256_file(staged)
        if hashes:
            record["asset_sha256"] = hashes
    manifest = {
        "dataset_schema_version": V3_DATASET_SCHEMA_VERSION,
        "dataset_id": V3_DATASET_ID,
        "created_at": (existing_manifest or {}).get("created_at", datetime.now(UTC).isoformat()),
        "updated_at": datetime.now(UTC).isoformat(),
        "cycle_name_width": V3_CYCLE_NAME_WIDTH,
        "cycles": all_cycles,
        "source_experiments": all_sources,
        "cycle_index": {
            "path": "cycle_index.parquet",
            "row_count": len(cycle_index),
            "sha256": sha256_file(staging_dataset / "cycle_index.parquet"),
        },
        "image_metadata": {
            "path": "image_metadata.parquet",
            "row_count": len(image_metadata),
            "sha256": sha256_file(staging_dataset / "image_metadata.parquet"),
        },
        "channel_registry": {
            "path": "channel_registry.json",
            "sha256": sha256_file(staging_dataset / "channel_registry.json"),
            "canonical_hash": registry_hash,
        },
        "summary_cycle_count": len(cycle_index),
        "image_count": len(image_metadata),
    }
    write_atomic_json(manifest, staging_dataset / "dataset_manifest.json")


def _source_provenance(source: V3SourceRun, fingerprint: str) -> dict[str, Any]:
    return {
        "experiment_id": source.experiment_id,
        "experiment_date": source.experiment_date,
        "input_dir": str(source.input_dir),
        "formal_run": relative_posix_path(source.path, source.project_root),
        "resolved_config_sha256": source.resolved_config_sha256,
        "input_inventory_sha256": source.input_inventory_sha256,
        "run_manifest_sha256": source.manifest_sha256,
        "pipeline_commit": source.pipeline_commit,
        "source_fingerprint": fingerprint,
    }


def _rewrite_historical_cycles_if_needed(
    dataset_dir: Path,
    staging_dataset: Path,
    old_registry: Mapping[str, Any],
    new_registry: Mapping[str, Any],
    old_index: pd.DataFrame,
) -> list[str]:
    fields_changed = old_registry.get("fields") != new_registry.get("fields")
    coverage_changed = _coverage_required_channels(old_registry) != _coverage_required_channels(
        new_registry
    )
    if not fields_changed and not coverage_changed:
        return []
    loader = None
    cycle_records: dict[str, Mapping[str, Any]] = {}
    if coverage_changed:
        from .dataset_loader import DatasetLoader

        loader = DatasetLoader(dataset_dir)
        raw_cycle_records = loader.manifest.get("cycles")
        if not isinstance(raw_cycle_records, list):
            raise ValueError("dataset manifest cycles must be a list")
        cycle_records = {
            str(record["cycle_name"]): record
            for record in raw_cycle_records
            if isinstance(record, Mapping) and "cycle_name" in record
        }
    replaced: list[str] = []
    for row in old_index.to_dict(orient="records"):
        path = str(row["data_path"])
        source_path = dataset_dir / path
        frame = pd.read_parquet(source_path)
        frame = frame.drop(columns=list(V3_DATASET_FIELDS), errors="ignore")
        canonical = frame
        if fields_changed:
            canonical = _add_dataset_fields(
                canonical_frame(frame, new_registry),
                cycle_index=int(row["dataset_cycle_index"]),
                cycle_name=str(row["cycle_name"]),
                cycle_uid=str(row["cycle_uid"]),
            )
            target = staging_dataset / path
            write_atomic_parquet(canonical, target)
            csv_path = str(row["csv_path"])
            canonical.to_csv(staging_dataset / csv_path, index=False)
            replaced.extend([path, csv_path])
        if coverage_changed:
            assert loader is not None
            from .dataset_coverage_v3 import render_rgb_coverage

            cycle_name = str(row["cycle_name"])
            cycle_record = cycle_records.get(cycle_name)
            if cycle_record is None:
                raise ValueError(f"dataset manifest is missing historical cycle: {cycle_name}")
            coverage_path = str(row["rgb_coverage_path"])
            render_rgb_coverage(
                canonical,
                loader.load_cycle_images(cycle_name),
                cycle_record,
                staging_dataset / coverage_path,
                registry=new_registry,
            )
            replaced.append(coverage_path)
    return replaced


def _coverage_required_channels(registry: Mapping[str, Any]) -> frozenset[str]:
    channels = registry.get("channels")
    if not isinstance(channels, Mapping):
        return frozenset()
    return frozenset(
        str(name)
        for name, settings in channels.items()
        if isinstance(settings, Mapping) and bool(settings.get("coverage_required", False))
    )


def _new_asset_paths(
    staging_dataset: Path, index: pd.DataFrame, images: pd.DataFrame
) -> list[str]:
    paths: list[str] = []
    for row in index.to_dict(orient="records"):
        paths.extend(
            [
                str(row["data_path"]),
                str(row["csv_path"]),
                str(row["publication_path"]),
                str(row["rgb_coverage_path"]),
            ]
        )
    for row in images.to_dict(orient="records"):
        matches = list(
            (staging_dataset / "images" / str(row["cycle_name"])).rglob(
                f"{row['image_id']}.*"
            )
        )
        if matches:
            paths.append(matches[0].relative_to(staging_dataset).as_posix())
        else:
            raise FileNotFoundError(f"staged image is missing: {row['image_id']}")
    return paths


def _write_readme(dataset_dir: Path) -> None:
    (dataset_dir / "README.md").write_text(
        "# frost_cycle_dataset\n\nSelf-contained Cycle Dataset v3.\n",
        encoding="utf-8",
    )


def _add_dataset_fields(
    frame: pd.DataFrame, *, cycle_index: int, cycle_name: str, cycle_uid: str
) -> pd.DataFrame:
    result = frame.copy()
    result["dataset_id"] = V3_DATASET_ID
    result["dataset_schema_version"] = V3_DATASET_SCHEMA_VERSION
    result["dataset_cycle_index"] = cycle_index
    result["cycle_name"] = cycle_name
    result["cycle_uid"] = cycle_uid
    return result


def _assign_cycle_names(summary: pd.DataFrame, *, start_index: int) -> dict[tuple[str, str], str]:
    ordered = _sort_summary(summary)
    names: dict[tuple[str, str], str] = {}
    for offset, row in enumerate(ordered.to_dict(orient="records"), start=start_index):
        key = (str(row["experiment_id"]), str(row["cycle_id"]))
        if key in names:
            raise ValueError(f"duplicate cycle identity: {key}")
        names[key] = f"frost_cycle_{offset:0{V3_CYCLE_NAME_WIDTH}d}"
    return names


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["_date"] = pd.to_datetime(result["experiment_date"], errors="raise")
    result["_cycle_number"] = result["cycle_id"].astype(str).map(_natural_cycle_number)
    return result.sort_values(
        ["_date", "experiment_id", "_cycle_number", "cycle_id"], kind="stable"
    ).drop(columns=["_date", "_cycle_number"]).reset_index(drop=True)


def _natural_cycle_number(value: str) -> int:
    match = re.search(r"(?:^|_)cycle_(\d+)$", value)
    return int(match.group(1)) if match else 2**31 - 1


def _next_cycle_index(index: pd.DataFrame) -> int:
    if index.empty:
        return 1
    return int(pd.to_numeric(index["dataset_cycle_index"], errors="raise").max()) + 1


def _cycle_index_from_name(name: str) -> int:
    match = re.fullmatch(r"frost_cycle_(\d+)", name)
    if match is None:
        raise ValueError(f"invalid cycle name: {name}")
    return int(match.group(1))


def _cycle_uid(experiment_id: str, cycle_id: str) -> str:
    return f"{experiment_id}::{cycle_id}"


def _slot_name(source_camera_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", source_camera_id).strip("_").lower()
    return f"unassigned_{safe or 'unknown'}"


def _safe_source_relative(raw_path: str) -> str:
    path = Path(raw_path.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or "\\" in raw_path:
        raise ValueError(f"source image path must be safe and relative: {raw_path}")
    return path.as_posix()


def _identity_matches(frame: pd.DataFrame, source: V3SourceRun) -> bool:
    if frame.empty:
        return True
    return bool(
        frame["experiment_id"].astype(str).eq(source.experiment_id).all()
        and frame["experiment_date"].astype(str).str[:10].eq(source.experiment_date).all()
    )


def _validate_image_triples(frame: pd.DataFrame, frame_name: str) -> tuple[str, ...]:
    roles: dict[str, set[str]] = {}
    pattern = re.compile(r"^image_(.+)_(path|time|offset_seconds)$")
    for column in frame.columns:
        match = pattern.fullmatch(str(column))
        if match is not None:
            roles.setdefault(match.group(1), set()).add(match.group(2))
    for role, suffixes in roles.items():
        if suffixes != {"path", "time", "offset_seconds"}:
            raise ValueError(f"{frame_name} image triple is incomplete for {role}")
    return tuple(sorted(roles))


def _read_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in (
        "heating_start",
        "stable_heating_start",
        "defrost_start",
        "defrost_end",
        "baseline_start",
        "baseline_end",
    ):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _none_if_na(value: Any) -> Any:
    return None if value is None or pd.isna(value) else value


def _json_value(value: Any) -> Any:
    value = _none_if_na(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value
