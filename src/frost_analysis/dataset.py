"""Cycle-level dataset publication helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, TypedDict, cast

import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .config import find_project_root, is_iso_date
from .io import ensure_output_outside_input, relative_posix_path, sha256_file

DATASET_SCHEMA_VERSION = 1
CYCLE_NAME_WIDTH = 6
_DATASET_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_CYCLE_NUMBER_RE = re.compile(r"(?:^|_)cycle_(\d+)$")
CycleKey = tuple[str, str]


@dataclass(frozen=True)
class SourceRun:
    """Static metadata for one formal experiment run."""

    path: Path
    experiment_id: str
    experiment_date: str
    input_dir: Path
    prepared_path: Path
    processed_path: Path
    summary_path: Path
    resolved_config_sha256: str
    prepared_sha256: str
    processed_sha256: str
    summary_sha256: str
    manifest_sha256: str
    git_commit: str | None


class SourceDescriptor(TypedDict):
    source: SourceRun
    project_root: Path
    summary: pd.DataFrame
    processed_counts: dict[CycleKey, int]
    schema: list[dict[str, Any]]


class SourceExport(TypedDict):
    source: SourceRun
    summary: pd.DataFrame
    processed_counts: dict[CycleKey, int]
    records: list[dict[str, object]]
    inventory_hash: str
    rewritten: pd.DataFrame
    cycle_names: dict[CycleKey, str]


def validate_dataset_id(value: str) -> str:
    """Validate and return a dataset identifier suitable for a directory name."""
    if not _DATASET_ID_RE.fullmatch(value):
        raise ValueError(f"invalid dataset_id: {value!r}")
    return value


def make_cycle_uid(experiment_id: str, cycle_id: str) -> str:
    """Build the stable source identity for one cycle."""
    return f"{experiment_id}__{cycle_id}"


def make_v2_cycle_uid(experiment_id: str, cycle_id: str) -> str:
    """Build the v2 self-contained Dataset cycle identity."""
    return f"{experiment_id}::{cycle_id}"


def format_cycle_name(index: int) -> str:
    """Build the human-readable global cycle filename stem."""
    if index < 1:
        raise ValueError("dataset cycle index must be positive")
    return f"frost_cycle_{index:0{CYCLE_NAME_WIDTH}d}"


def parse_cycle_name(name: str) -> int:
    """Parse the positive global index from a dataset cycle name."""
    match = re.fullmatch(rf"frost_cycle_(\d{{{CYCLE_NAME_WIDTH},}})", name)
    if match is None or int(match.group(1)) < 1:
        raise ValueError(f"invalid cycle_name: {name!r}")
    return int(match.group(1))


def cycle_sort_key(
    experiment_date: str,
    experiment_id: str,
    cycle_id: str,
) -> tuple[date, str, int, str]:
    """Return a deterministic date/experiment/natural-cycle sort key."""
    match = _CYCLE_NUMBER_RE.search(cycle_id)
    number = int(match.group(1)) if match else 2**31 - 1
    return date.fromisoformat(experiment_date), experiment_id, number, cycle_id


def assign_cycle_names(
    summary: pd.DataFrame,
    processed_counts: Mapping[CycleKey, int],
) -> dict[CycleKey, str]:
    """Assign consecutive names to cycles that have Processed rows."""
    required = {"experiment_id", "experiment_date", "cycle_id"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"cycle summary missing columns: {missing}")
    keys = {
        (str(row.experiment_id), str(row.cycle_id)): str(row.experiment_date)[:10]
        for row in summary[["experiment_id", "experiment_date", "cycle_id"]].itertuples(
            index=False
        )
    }
    published = [key for key, count in processed_counts.items() if count > 0]
    published.sort(key=lambda key: cycle_sort_key(keys[key], key[0], key[1]))
    return {key: format_cycle_name(index) for index, key in enumerate(published, start=1)}


def build_cycle_index(
    summary: pd.DataFrame,
    *,
    processed_counts: Mapping[CycleKey, int],
    cycle_names: Mapping[CycleKey, str],
    cycle_files: Mapping[CycleKey, Mapping[str, object]],
) -> pd.DataFrame:
    """Build one index row for every Summary cycle."""
    rows: list[dict[str, object]] = []
    for row in summary.to_dict(orient="records"):
        experiment_id = str(row["experiment_id"])
        cycle_id = str(row["cycle_id"])
        key = (experiment_id, cycle_id)
        processed_row_count = int(processed_counts.get(key, 0))
        published = processed_row_count > 0
        cycle_name = cycle_names.get(key)
        file_info = cycle_files.get(key, {})
        baseline_status = row.get("baseline_status")
        cycle_status = row.get("cycle_status")
        rows.append(
            {
                "dataset_cycle_index": parse_cycle_name(cycle_name) if cycle_name else pd.NA,
                "cycle_name": cycle_name,
                "cycle_uid": make_cycle_uid(experiment_id, cycle_id),
                "experiment_id": experiment_id,
                "experiment_date": str(row["experiment_date"]),
                "cycle_id": cycle_id,
                "cycle_status": cycle_status,
                "cycle_status_reason": row.get("cycle_status_reason"),
                "baseline_status": baseline_status,
                "baseline_failure_reason": row.get("baseline_failure_reason"),
                "published": published,
                "data_path": file_info.get("data_path"),
                "data_sha256": file_info.get("data_sha256"),
                "data_size_bytes": file_info.get("data_size_bytes"),
                "processed_row_count": processed_row_count,
                "image_count": int(cast(int, file_info.get("image_count", 0))),
                "recommended_for_analysis": bool(
                    published
                    and cycle_status == "valid"
                    and baseline_status == "available"
                ),
                "dataset_exclusion_reason": None if published else "no_processed_rows",
            }
        )
    result = pd.DataFrame(rows)
    if "dataset_cycle_index" in result:
        result["dataset_cycle_index"] = result["dataset_cycle_index"].astype("Int64")
    if "data_size_bytes" in result:
        result["data_size_bytes"] = result["data_size_bytes"].astype("Int64")
    return result


def scientific_fingerprint(source_run: SourceRun, image_inventory_sha256: str) -> str:
    """Hash only the inputs consumed by the cycle dataset exporter."""
    payload = {
        "experiment_id": source_run.experiment_id,
        "experiment_date": source_run.experiment_date,
        "resolved_config_sha256": source_run.resolved_config_sha256,
        "prepared_data_sha256": source_run.prepared_sha256,
        "processed_data_sha256": source_run.processed_sha256,
        "cycle_summary_sha256": source_run.summary_sha256,
        "matched_image_inventory_sha256": image_inventory_sha256,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def logical_schema_compatible(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge two ordered Arrow schemas, allowing null to promote once."""
    if len(left) != len(right) or [field["name"] for field in left] != [
        field["name"] for field in right
    ]:
        raise ValueError(_schema_mismatch_message(left, right))
    merged: list[dict[str, Any]] = []
    for left_field, right_field in zip(left, right, strict=True):
        left_type = str(left_field["logical_type"])
        right_type = str(right_field["logical_type"])
        if left_type == "null":
            logical_type = right_type
        elif right_type == "null":
            logical_type = left_type
        elif left_type != right_type:
            raise ValueError(_schema_mismatch_message(left, right))
        else:
            logical_type = left_type
        merged.append(
            {
                "name": left_field["name"],
                "logical_type": logical_type,
                "nullable": bool(left_field["nullable"] or right_field["nullable"]),
            }
        )
    return merged


def _schema_mismatch_message(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]]
) -> str:
    """Describe schema differences without attempting to repair them."""
    expected_names = [str(field["name"]) for field in expected]
    actual_names = [str(field["name"]) for field in actual]
    expected_set = set(expected_names)
    actual_set = set(actual_names)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    order_changed = not missing and not extra and expected_names != actual_names

    expected_by_name = {str(field["name"]): field for field in expected}
    actual_by_name = {str(field["name"]): field for field in actual}
    type_changes = []
    for name in sorted(expected_set & actual_set):
        expected_type = str(expected_by_name[name]["logical_type"])
        actual_type = str(actual_by_name[name]["logical_type"])
        if expected_type == "null" or actual_type == "null":
            continue
        if expected_type != actual_type:
            type_changes.append(f"{name}: {expected_type} -> {actual_type}")

    return (
        "Processed schema mismatch:\n"
        f"missing columns: {missing}\n"
        f"extra columns: {extra}\n"
        f"order changed: {str(order_changed).lower()}\n"
        f"type changes: {type_changes}"
    )


def source_processed_schema(path: Path) -> list[dict[str, Any]]:
    """Read the logical Arrow schema of a source Processed parquet file."""
    schema = pq.read_schema(path)
    return [
        {
            "name": field.name,
            "logical_type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


def load_source_run(run_path: Path) -> SourceRun:  # noqa: C901
    """Load and preflight the static contract of one formal run directory."""
    run_dir = run_path.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"source run directory does not exist: {run_path}")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source run is missing manifest.json: {run_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"source manifest is not valid JSON: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"source manifest must be an object: {manifest_path}")

    experiment_id = str(manifest.get("experiment_id", ""))
    experiment_date = str(manifest.get("experiment_date", ""))
    if not experiment_id or not is_iso_date(experiment_date):
        raise ValueError(f"source manifest has invalid experiment identity: {run_dir}")

    project_root = find_project_root(manifest_path)
    if project_root is None:
        raise FileNotFoundError(f"could not find project root for source run: {run_dir}")
    relative_posix_path(run_dir, project_root)

    provenance = manifest.get("config_provenance")
    resolved_config = manifest.get("resolved_config")
    outputs = manifest.get("outputs")
    if not isinstance(provenance, dict) or not isinstance(resolved_config, dict):
        raise ValueError(f"source manifest lacks configuration provenance: {run_dir}")
    if not isinstance(outputs, dict):
        raise ValueError(f"source manifest lacks outputs: {run_dir}")
    resolved_hash = str(provenance.get("resolved_config_sha256", ""))
    input_value = resolved_config.get("input_dir")
    if not resolved_hash or not isinstance(input_value, str) or Path(input_value).is_absolute():
        raise ValueError(f"source manifest has invalid resolved input contract: {run_dir}")
    input_dir = (project_root / input_value).resolve()
    relative_posix_path(input_dir, project_root)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"source input directory does not exist: {input_dir}")

    required_outputs = {
        "prepared_data": "prepared_data.parquet",
        "processed_data": "processed_data.parquet",
        "cycle_summary": "cycle_summary.csv",
        "candidate_channel_evidence": "candidate_channel_evidence.csv",
    }
    output_paths: dict[str, Path] = {}
    for key, expected_name in required_outputs.items():
        declared = outputs.get(key)
        if (
            not isinstance(declared, str)
            or Path(declared).is_absolute()
            or ".." in Path(declared).parts
            or Path(declared).name != expected_name
        ):
            raise ValueError(f"source manifest output {key!r} is invalid: {run_dir}")
        output_path = (run_dir / declared).resolve()
        relative_posix_path(output_path, run_dir)
        if not output_path.is_file():
            raise FileNotFoundError(f"source run is missing {expected_name}: {run_dir}")
        output_paths[key] = output_path

    git_value = manifest.get("git_commit")
    git_commit = None if git_value is None else str(git_value)
    return SourceRun(
        path=run_dir,
        experiment_id=experiment_id,
        experiment_date=experiment_date,
        input_dir=input_dir,
        prepared_path=output_paths["prepared_data"],
        processed_path=output_paths["processed_data"],
        summary_path=output_paths["cycle_summary"],
        resolved_config_sha256=resolved_hash,
        prepared_sha256=sha256_file(output_paths["prepared_data"]),
        processed_sha256=sha256_file(output_paths["processed_data"]),
        summary_sha256=sha256_file(output_paths["cycle_summary"]),
        manifest_sha256=sha256_file(manifest_path),
        git_commit=git_commit,
    )


def build_dataset(run_paths: Sequence[Path], output_dir: Path) -> Path:  # noqa: C901
    """Build a complete cycle dataset from explicit formal run directories."""
    from .dataset_io import (
        create_build_staging,
        publish_build,
        write_atomic_json,
        write_atomic_parquet,
    )
    from .dataset_validation import validate_dataset

    output_dir = output_dir.resolve()
    dataset_id = validate_dataset_id(output_dir.name)
    if output_dir.exists():
        raise FileExistsError(f"dataset output already exists: {output_dir}")
    if not run_paths:
        raise ValueError("dataset build requires at least one --run")
    resolved_paths = [path.resolve() for path in run_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("duplicate source run path")

    descriptors = [_source_descriptor(path) for path in resolved_paths]
    sources = [descriptor["source"] for descriptor in descriptors]
    experiment_ids = [source.experiment_id for source in sources]
    if len(set(experiment_ids)) != len(experiment_ids):
        duplicates = sorted({value for value in experiment_ids if experiment_ids.count(value) > 1})
        raise ValueError(f"duplicate experiment_id in build inputs: {duplicates}")
    project_roots = {descriptor["project_root"] for descriptor in descriptors}
    if len(project_roots) != 1:
        raise ValueError("source runs must share one project root")
    for source in sources:
        ensure_output_outside_input(output_dir, source.input_dir)

    summary_frames = [descriptor["summary"] for descriptor in descriptors]
    summary = pd.concat(summary_frames, ignore_index=True)
    summary = _sort_summary(summary)
    processed_counts: dict[CycleKey, int] = {}
    for descriptor in descriptors:
        processed_counts.update(descriptor["processed_counts"])
    cycle_names = assign_cycle_names(summary, processed_counts)
    merged_schema = descriptors[0]["schema"]
    for descriptor in descriptors[1:]:
        merged_schema = logical_schema_compatible(merged_schema, descriptor["schema"])

    staging_root, staging_dataset = create_build_staging(output_dir)
    published = False
    try:
        cycle_files: dict[tuple[str, str], dict[str, object]] = {}
        image_records: list[dict[str, object]] = []
        source_manifests: list[dict[str, object]] = []
        for descriptor in sorted(
            descriptors,
            key=lambda item: (
                item["source"].experiment_date,
                item["source"].experiment_id,
            ),
        ):
            export = _load_source_export(descriptor, cycle_names)
            source = export["source"]
            cycle_files.update(
                _write_source_export(
                    export,
                    dataset_id=dataset_id,
                    staging_dataset=staging_dataset,
                )
            )
            records = export["records"]
            image_records.extend(records)
            source_manifests.append(
                _source_manifest_record(source, str(export["inventory_hash"]))
            )

        cycle_index = build_cycle_index(
            summary,
            processed_counts=processed_counts,
            cycle_names=cycle_names,
            cycle_files=cycle_files,
        )
        image_index = _image_index_frame(image_records)
        write_atomic_parquet(cycle_index, staging_dataset / "cycle_index.parquet")
        write_atomic_parquet(image_index, staging_dataset / "image_index.parquet")
        manifest = _dataset_manifest(
            dataset_id=dataset_id,
            source_schema=merged_schema,
            source_runs=source_manifests,
            cycle_index=cycle_index,
            image_index=image_index,
            staging_dataset=staging_dataset,
        )
        write_atomic_json(manifest, staging_dataset / "dataset_manifest.json")
        (staging_dataset / "README.md").write_text(
            _dataset_readme(dataset_id), encoding="utf-8"
        )
        validate_dataset(staging_dataset)
        publish_build(staging_root, staging_dataset, output_dir)
        published = True
        return output_dir
    finally:
        if not published:
            import shutil

            shutil.rmtree(staging_root, ignore_errors=True)


def append_dataset(run_path: Path, dataset_dir: Path) -> Path:  # noqa: C901
    """Append one formal run using a metadata-first, recoverable transaction."""
    from .dataset_io import (
        backup_append_metadata,
        cleanup_append_staging,
        commit_append_files,
        create_append_staging,
        rollback_append,
        write_atomic_json,
        write_atomic_parquet,
    )
    from .dataset_validation import (
        _validate_append_candidate,
        _validate_dataset_structure,
        _validate_existing_file_paths,
        _validate_new_files,
    )

    dataset_dir = dataset_dir.resolve()
    manifest, old_cycle_index, old_image_index = _validate_dataset_structure(dataset_dir)
    _validate_existing_file_paths(dataset_dir, old_cycle_index, old_image_index)
    dataset_id = validate_dataset_id(str(manifest["dataset_id"]))
    descriptor = _source_descriptor(run_path)
    source = descriptor["source"]
    dataset_root = find_project_root(dataset_dir)
    source_root = descriptor["project_root"]
    if dataset_root is not None and source_root != dataset_root:
        raise ValueError("source run and dataset must share one project root")
    ensure_output_outside_input(dataset_dir, source.input_dir)

    source_id = source.experiment_id
    existing_runs = manifest.get("source_runs")
    if not isinstance(existing_runs, list):
        raise ValueError("dataset manifest is missing source_runs")
    existing_record = next(
        (
            item
            for item in existing_runs
            if isinstance(item, dict) and item.get("experiment_id") == source_id
        ),
        None,
    )
    if existing_record is not None and not isinstance(existing_record, dict):
        raise ValueError(f"invalid source run record for {source_id}")

    existing_names = {
        (str(row["experiment_id"]), str(row["cycle_id"])): str(row["cycle_name"])
        for row in old_cycle_index.loc[old_cycle_index["published"].astype(bool)].to_dict(
            orient="records"
        )
    }
    processed_counts = descriptor["processed_counts"]
    new_keys = [key for key, count in processed_counts.items() if count > 0]
    if existing_record is None:
        if any(str(value) == source_id for value in old_cycle_index["experiment_id"]):
            raise ValueError(f"dataset already contains experiment_id: {source_id}")
        next_index = int(old_cycle_index["dataset_cycle_index"].max()) + 1 if old_cycle_index[
            "dataset_cycle_index"
        ].notna().any() else 1
        new_keys.sort(
            key=lambda key: cycle_sort_key(source.experiment_date, key[0], key[1])
        )
        cycle_names = {
            key: format_cycle_name(next_index + offset)
            for offset, key in enumerate(new_keys)
        }
    else:
        cycle_names = {key: existing_names[key] for key in new_keys if key in existing_names}
        missing_names = [key for key in new_keys if key not in cycle_names]
        if missing_names:
            raise ValueError(
                f"existing experiment has new cycle identities: {missing_names}"
            )

    export = _load_source_export(descriptor, {**existing_names, **cycle_names})
    new_source_record = _source_manifest_record(source, str(export["inventory_hash"]))
    if existing_record is not None:
        old_fingerprint = str(existing_record.get("fingerprint", ""))
        new_fingerprint = str(new_source_record["fingerprint"])
        if old_fingerprint == new_fingerprint:
            return dataset_dir
        changed = [
            field
            for field in (
                "resolved_config_sha256",
                "prepared_data_sha256",
                "processed_data_sha256",
                "cycle_summary_sha256",
                "matched_image_inventory_sha256",
            )
            if str(existing_record.get(field)) != str(new_source_record[field])
        ]
        details = ", ".join(changed) if changed else "fingerprint"
        raise ValueError(f"fingerprint conflict for {source_id}; changed: {details}")

    merged_schema = logical_schema_compatible(
        cast(list[dict[str, Any]], manifest["source_processed_schema"]),
        descriptor["schema"],
    )
    staging_root, staging_dataset = create_append_staging(dataset_dir)
    moved_files: list[Path] = []
    backup_dir: Path | None = None
    committed = False
    try:
        new_cycle_files = _write_source_export(
            export,
            dataset_id=dataset_id,
            staging_dataset=staging_dataset,
        )
        new_records = export["records"]
        new_cycle_index = build_cycle_index(
            export["summary"],
            processed_counts=processed_counts,
            cycle_names={**existing_names, **cycle_names},
            cycle_files=new_cycle_files,
        )
        new_cycle_index = new_cycle_index.loc[
            new_cycle_index["experiment_id"].eq(source_id)
        ].reset_index(drop=True)
        merged_cycle_index = pd.concat(
            [old_cycle_index, new_cycle_index], ignore_index=True
        )
        merged_image_index = pd.concat(
            [old_image_index, _image_index_frame(new_records)], ignore_index=True
        )
        write_atomic_parquet(merged_cycle_index, staging_dataset / "cycle_index.parquet")
        write_atomic_parquet(merged_image_index, staging_dataset / "image_index.parquet")
        source_runs = [
            item for item in existing_runs if isinstance(item, dict)
        ] + [new_source_record]
        manifest_payload = _dataset_manifest(
            dataset_id=dataset_id,
            source_schema=merged_schema,
            source_runs=source_runs,
            cycle_index=merged_cycle_index,
            image_index=merged_image_index,
            staging_dataset=staging_dataset,
        )
        write_atomic_json(manifest_payload, staging_dataset / "dataset_manifest.json")
        (staging_dataset / "README.md").write_text(
            _dataset_readme(dataset_id), encoding="utf-8"
        )
        _validate_append_candidate(
            staging_dataset,
            manifest_payload,
            merged_cycle_index,
            merged_image_index,
            new_cycle_index,
            new_records,
        )
        backup_dir = backup_append_metadata(dataset_dir, staging_root)
        moved_files = commit_append_files(
            staging_dataset,
            dataset_dir,
            [
                *[str(item["data_path"]) for item in new_cycle_files.values()],
                *[str(item["image_path"]) for item in new_records],
            ],
            moved_files=moved_files,
        )
        committed = True
        manifest_after, cycle_after, image_after = _validate_dataset_structure(dataset_dir)
        _validate_new_files(
            dataset_dir,
            manifest_after,
            cycle_after.loc[cycle_after["experiment_id"].eq(source_id)],
            image_after,
            new_records=new_records,
        )
        return dataset_dir
    except Exception as append_error:
        if committed or moved_files or backup_dir is not None:
            try:
                rollback_append(dataset_dir, backup_dir, moved_files)
                _validate_dataset_structure(dataset_dir)
                _validate_existing_file_paths(
                    dataset_dir,
                    old_cycle_index,
                    old_image_index,
                )
            except Exception as rollback_error:
                raise RuntimeError(
                    "append failed and rollback validation also failed: "
                    f"{rollback_error}"
                ) from append_error
        raise
    finally:
        cleanup_append_staging(staging_root)


def _source_descriptor(run_path: Path) -> SourceDescriptor:
    source = load_source_run(run_path)
    project_root = find_project_root(source.path / "manifest.json")
    if project_root is None:
        raise FileNotFoundError(f"could not find project root for source run: {run_path}")
    summary = _sort_summary(_read_cycle_summary(source.summary_path))
    processed_keys = pd.read_parquet(
        source.processed_path,
        columns=["experiment_id", "cycle_id"],
    )
    counts = {
        (str(experiment_id), str(cycle_id)): int(len(group))
        for (experiment_id, cycle_id), group in processed_keys.groupby(
            ["experiment_id", "cycle_id"], dropna=False, sort=False
        )
    }
    return {
        "source": source,
        "project_root": project_root,
        "summary": summary,
        "processed_counts": counts,
        "schema": source_processed_schema(source.processed_path),
    }


def _load_source_export(
    descriptor: SourceDescriptor,
    cycle_names: Mapping[CycleKey, str],
) -> SourceExport:
    """Load, scientifically validate, and transform one source run in memory."""
    from .dataset_images import collect_matched_images, rewrite_processed_image_paths
    from .validation import validate_prepared, validate_processed

    source = descriptor["source"]
    summary = descriptor["summary"]
    prepared = pd.read_parquet(source.prepared_path)
    processed = pd.read_parquet(source.processed_path)
    validate_prepared(prepared, summary)
    validate_processed(processed, summary)
    if _identity_mismatch(prepared, source) or _identity_mismatch(processed, source):
        raise ValueError(f"source data identity disagrees with manifest: {source.path}")
    records, inventory_hash = collect_matched_images(
        prepared,
        input_dir=source.input_dir,
        cycle_names=cycle_names,
    )
    rewritten = rewrite_processed_image_paths(processed, records)
    return {
        "source": source,
        "summary": summary,
        "processed_counts": descriptor["processed_counts"],
        "records": records,
        "inventory_hash": inventory_hash,
        "rewritten": rewritten,
        "cycle_names": dict(cycle_names),
    }


def _write_source_export(
    export: SourceExport,
    *,
    dataset_id: str,
    staging_dataset: Path,
) -> dict[tuple[str, str], dict[str, object]]:
    """Write one transformed source run into a build or append staging tree."""
    from .dataset_images import copy_dataset_image
    from .dataset_io import write_atomic_parquet

    records = export["records"]
    rewritten = export["rewritten"]
    processed_counts = export["processed_counts"]
    cycle_names = export["cycle_names"]
    cycle_files: dict[CycleKey, dict[str, object]] = {}
    for record in records:
        destination = staging_dataset / str(record["image_path"])
        if destination.exists():
            raise FileExistsError(f"dataset image path collision: {destination}")
        copy_dataset_image(record, staging_dataset)
    for key, count in processed_counts.items():
        if count <= 0:
            continue
        cycle_name = cycle_names.get(key)
        if cycle_name is None:
            raise ValueError(f"missing dataset cycle name for {key}")
        cycle_uid = make_cycle_uid(*key)
        cycle_frame = rewritten.loc[
            rewritten["experiment_id"].astype(str).eq(key[0])
            & rewritten["cycle_id"].astype(str).eq(key[1])
        ].copy()
        if len(cycle_frame) != count:
            raise ValueError(f"Processed row count changed while exporting {key}")
        cycle_frame = _append_dataset_fields(
            cycle_frame,
            dataset_id=dataset_id,
            cycle_index=parse_cycle_name(cycle_name),
            cycle_name=cycle_name,
            cycle_uid=cycle_uid,
        )
        data_path = f"cycles/{cycle_name}.parquet"
        output_path = staging_dataset / data_path
        write_atomic_parquet(cycle_frame, output_path)
        cycle_files[key] = {
            "data_path": data_path,
            "data_sha256": sha256_file(output_path),
            "data_size_bytes": output_path.stat().st_size,
            "image_count": sum(1 for record in records if record["cycle_uid"] == cycle_uid),
        }
    return cycle_files


def _read_cycle_summary(path: Path) -> pd.DataFrame:
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


def _identity_mismatch(frame: pd.DataFrame, source: SourceRun) -> bool:
    return (
        "experiment_id" not in frame
        or bool(frame["experiment_id"].astype(str).ne(source.experiment_id).any())
        or (
            "experiment_date" in frame
            and bool(
                frame["experiment_date"].astype(str).str[:10].ne(source.experiment_date).any()
            )
        )
    )


def _append_dataset_fields(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    cycle_index: int,
    cycle_name: str,
    cycle_uid: str,
) -> pd.DataFrame:
    result = frame.reset_index(drop=True).copy()
    result["dataset_id"] = pd.Series(
        [dataset_id] * len(result), index=result.index, dtype="string"
    )
    result["dataset_schema_version"] = pd.Series(
        [DATASET_SCHEMA_VERSION] * len(result), index=result.index, dtype="int64"
    )
    result["dataset_cycle_index"] = pd.Series(
        [cycle_index] * len(result), index=result.index, dtype="int64"
    )
    result["cycle_name"] = pd.Series(
        [cycle_name] * len(result), index=result.index, dtype="string"
    )
    result["cycle_uid"] = pd.Series(
        [cycle_uid] * len(result), index=result.index, dtype="string"
    )
    return result


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return Summary rows in the same deterministic order as global cycles."""
    required = {"experiment_id", "experiment_date", "cycle_id"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"cycle summary missing columns: {missing}")
    result = summary.copy()
    result["_dataset_sort_key"] = [
        cycle_sort_key(str(row.experiment_date)[:10], str(row.experiment_id), str(row.cycle_id))
        for row in result[["experiment_date", "experiment_id", "cycle_id"]].itertuples(
            index=False
        )
    ]
    result = result.sort_values("_dataset_sort_key", kind="stable").drop(
        columns="_dataset_sort_key"
    )
    return result.reset_index(drop=True)


def _image_index_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "image_id",
        "cycle_uid",
        "cycle_name",
        "camera_role",
        "image_time",
        "matched_timestamp",
        "offset_seconds",
        "cycle_stage",
        "image_path",
        "source_relative_path",
        "sha256",
        "file_size_bytes",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([{column: record[column] for column in columns} for record in records])


def _source_manifest_record(source: SourceRun, image_inventory_hash: str) -> dict[str, object]:
    project_root = find_project_root(source.path / "manifest.json")
    if project_root is None:
        raise FileNotFoundError(f"could not find project root for source run: {source.path}")
    return {
        "experiment_id": source.experiment_id,
        "experiment_date": source.experiment_date,
        "source_run_path": relative_posix_path(source.path, project_root),
        "resolved_config_sha256": source.resolved_config_sha256,
        "prepared_data_sha256": source.prepared_sha256,
        "processed_data_sha256": source.processed_sha256,
        "cycle_summary_sha256": source.summary_sha256,
        "matched_image_inventory_sha256": image_inventory_hash,
        "fingerprint": scientific_fingerprint(source, image_inventory_hash),
        "manifest_sha256": source.manifest_sha256,
        "git_commit": source.git_commit,
    }


def _dataset_manifest(
    *,
    dataset_id: str,
    source_schema: list[dict[str, Any]],
    source_runs: list[dict[str, object]],
    cycle_index: pd.DataFrame,
    image_index: pd.DataFrame,
    staging_dataset: Path,
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "cycle_name_width": CYCLE_NAME_WIDTH,
        "summary_cycle_count": len(cycle_index),
        "published_cycle_count": int(cycle_index["published"].sum()),
        "excluded_cycle_count": int((~cycle_index["published"]).sum()),
        "image_count": len(image_index),
        "source_processed_schema": source_schema,
        "source_runs": sorted(
            source_runs,
            key=lambda item: (str(item["experiment_date"]), str(item["experiment_id"])),
        ),
        "files": {
            "cycle_index": {
                "path": "cycle_index.parquet",
                "sha256": sha256_file(staging_dataset / "cycle_index.parquet"),
                "row_count": len(cycle_index),
            },
            "image_index": {
                "path": "image_index.parquet",
                "sha256": sha256_file(staging_dataset / "image_index.parquet"),
                "row_count": len(image_index),
            },
        },
    }


def _dataset_readme(dataset_id: str) -> str:
    return (
        f"# {dataset_id}\n\n"
        "This dataset contains Processed cycle files and Prepared matched RGB images.\n"
        "The source run paths in dataset_manifest.json are audit metadata only.\n"
        "Use cycle_index.parquet and image_index.parquet for relationships.\n"
    )


# ---------------------------------------------------------------------------
# Dataset v2: self-contained cycle publication
# ---------------------------------------------------------------------------

DATASET_V2_SCHEMA_VERSION = 2


def add_dataset(run_path: Path, dataset_dir: Path) -> Path:  # noqa: C901
    """Create a Dataset from one run, or append one new run to an existing Dataset."""
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.exists():
        return _build_v2_dataset(run_path, dataset_dir)
    from .dataset_validation import _validate_v2_structure

    manifest, cycle_index, image_metadata = _validate_v2_structure(dataset_dir)
    return _append_v2_dataset(run_path, dataset_dir, manifest, cycle_index, image_metadata)


def _build_v2_dataset(run_path: Path, dataset_dir: Path) -> Path:  # noqa: C901
    from .dataset_io import (
        create_dataset_staging,
        publish_build,
        write_atomic_json,
        write_atomic_parquet,
    )
    from .dataset_manifest import build_manifest
    from .dataset_validation import validate_dataset

    dataset_id = validate_dataset_id(dataset_dir.name)
    descriptor = _load_v2_source(run_path)
    source = descriptor["source"]
    ensure_output_outside_input(dataset_dir, source.input_dir)
    cycle_names = _assign_v2_cycle_names(descriptor["summary"], start_index=1)
    staging_root, staging_dataset = create_dataset_staging(dataset_dir, kind="build")
    published = False
    try:
        cycle_index, image_metadata, cycles, inventory_hash = _write_v2_source_assets(
            descriptor,
            cycle_names=cycle_names,
            dataset_id=dataset_id,
            staging_dataset=staging_dataset,
        )
        write_atomic_parquet(cycle_index, staging_dataset / "cycle_index.parquet")
        write_atomic_parquet(image_metadata, staging_dataset / "image_metadata.parquet")
        source_schema = descriptor["schema"]
        source_record = _source_manifest_record(source, inventory_hash)
        manifest = build_manifest(
            dataset_id=dataset_id,
            source_schema=source_schema,
            source_runs=[source_record],
            cycles=cycles,
            cycle_index=cycle_index,
            image_metadata=image_metadata,
        )
        _add_v2_file_info(manifest, staging_dataset)
        write_atomic_json(manifest, staging_dataset / "dataset_manifest.json")
        (staging_dataset / "README.md").write_text(_v2_readme(dataset_id), encoding="utf-8")
        validate_dataset(staging_dataset)
        publish_build(staging_root, staging_dataset, dataset_dir)
        published = True
        return dataset_dir
    finally:
        if not published:
            import shutil

            shutil.rmtree(staging_root, ignore_errors=True)


def _append_v2_dataset(
    run_path: Path,
    dataset_dir: Path,
    manifest: dict[str, Any],
    old_cycle_index: pd.DataFrame,
    old_image_metadata: pd.DataFrame,
) -> Path:  # noqa: C901
    from .dataset_io import (
        backup_v2_metadata,
        cleanup_append_staging,
        commit_v2_append_files,
        create_dataset_staging,
        rollback_v2_append,
        write_atomic_json,
        write_atomic_parquet,
    )
    from .dataset_manifest import build_manifest
    from .dataset_validation import (
        _validate_v2_new_assets,
        _validate_v2_structure,
    )

    descriptor = _load_v2_source(run_path)
    source = descriptor["source"]
    ensure_output_outside_input(dataset_dir, source.input_dir)
    source_runs = manifest.get("source_runs")
    if not isinstance(source_runs, list):
        raise ValueError("dataset manifest is missing source_runs")
    source_record = _source_manifest_record(
        source,
        _collect_v2_inventory_hash(descriptor),
    )
    existing_record = next(
        (
            item
            for item in source_runs
            if isinstance(item, dict) and item.get("experiment_id") == source.experiment_id
        ),
        None,
    )
    if existing_record is not None:
        if str(existing_record.get("fingerprint")) == str(source_record["fingerprint"]):
            return dataset_dir
        changed = [
            field
            for field in (
                "resolved_config_sha256",
                "prepared_data_sha256",
                "processed_data_sha256",
                "cycle_summary_sha256",
                "matched_image_inventory_sha256",
            )
            if str(existing_record.get(field)) != str(source_record.get(field))
        ]
        detail = ", ".join(changed) if changed else "fingerprint"
        raise ValueError(f"fingerprint conflict for {source.experiment_id}; changed: {detail}")

    existing_ids = set(old_cycle_index["experiment_id"].astype(str))
    if source.experiment_id in existing_ids:
        raise ValueError(f"dataset already contains experiment_id: {source.experiment_id}")
    next_index = _next_v2_cycle_index(old_cycle_index)
    cycle_names = _assign_v2_cycle_names(descriptor["summary"], start_index=next_index)
    existing_keys = set(
        zip(
            old_cycle_index["experiment_id"].astype(str),
            old_cycle_index["source_cycle_id"].astype(str),
            strict=True,
        )
    )
    if existing_keys and set(cycle_names).intersection(existing_keys):
        raise ValueError("appended source cycle identity collides with existing Dataset")
    merged_schema = logical_schema_compatible(
        cast(list[dict[str, Any]], manifest.get("source_processed_schema", [])),
        descriptor["schema"],
    )

    staging_root, staging_dataset = create_dataset_staging(dataset_dir, kind="append")
    moved_files: list[Path] = []
    backup_dir: Path | None = None
    committed = False
    try:
        new_cycle_index, new_image_metadata, new_cycles, _ = _write_v2_source_assets(
            descriptor,
            cycle_names=cycle_names,
            dataset_id=str(manifest["dataset_id"]),
            staging_dataset=staging_dataset,
        )
        merged_cycle_index = pd.concat([old_cycle_index, new_cycle_index], ignore_index=True)
        merged_image_metadata = pd.concat(
            [old_image_metadata, new_image_metadata], ignore_index=True
        )
        write_atomic_parquet(merged_cycle_index, staging_dataset / "cycle_index.parquet")
        write_atomic_parquet(
            merged_image_metadata,
            staging_dataset / "image_metadata.parquet",
        )
        existing_cycles = [item for item in manifest.get("cycles", []) if isinstance(item, dict)]
        full_manifest = build_manifest(
            dataset_id=str(manifest["dataset_id"]),
            source_schema=merged_schema,
            source_runs=[item for item in source_runs if isinstance(item, dict)] + [source_record],
            cycles=existing_cycles + new_cycles,
            cycle_index=merged_cycle_index,
            image_metadata=merged_image_metadata,
            created_at=str(manifest.get("created_at")),
        )
        _add_v2_file_info(full_manifest, staging_dataset)
        write_atomic_json(full_manifest, staging_dataset / "dataset_manifest.json")
        _write_v2_asset_readme(staging_dataset, str(manifest["dataset_id"]))
        _validate_v2_new_assets(staging_dataset, new_cycle_index, new_image_metadata)
        backup_dir = backup_v2_metadata(dataset_dir, staging_root)
        relative_assets = _v2_asset_paths(new_cycle_index, new_image_metadata, staging_dataset)
        moved_files = commit_v2_append_files(
            staging_dataset,
            dataset_dir,
            relative_assets,
            moved_files=moved_files,
        )
        committed = True
        after_manifest, after_index, after_images = _validate_v2_structure(dataset_dir)
        _validate_v2_new_assets(
            dataset_dir,
            after_index.loc[after_index["experiment_id"].eq(source.experiment_id)],
            after_images.loc[after_images["cycle_name"].isin(new_image_metadata["cycle_name"])],
        )
        _ = after_manifest
        return dataset_dir
    except Exception as append_error:
        if committed or moved_files or backup_dir is not None:
            try:
                rollback_v2_append(dataset_dir, backup_dir, moved_files)
                _validate_v2_structure(dataset_dir)
            except Exception as rollback_error:
                raise RuntimeError(
                    "append failed and rollback validation also failed: "
                    f"{rollback_error}"
                ) from append_error
        raise
    finally:
        cleanup_append_staging(staging_root)


def _load_v2_source(run_path: Path) -> dict[str, Any]:
    from .validation import validate_prepared, validate_processed

    source = load_source_run(run_path)
    summary = _sort_summary(_read_cycle_summary(source.summary_path))
    prepared = pd.read_parquet(source.prepared_path)
    processed = pd.read_parquet(source.processed_path)
    validate_prepared(prepared, summary)
    validate_processed(processed, summary)
    if _identity_mismatch(prepared, source) or _identity_mismatch(processed, source):
        raise ValueError(f"source data identity disagrees with manifest: {source.path}")
    return {
        "source": source,
        "project_root": find_project_root(source.path / "manifest.json"),
        "summary": summary,
        "prepared": prepared,
        "processed": processed,
        "schema": source_processed_schema(source.processed_path),
    }


def _assign_v2_cycle_names(summary: pd.DataFrame, *, start_index: int) -> dict[CycleKey, str]:
    names: dict[CycleKey, str] = {}
    for offset, row in enumerate(summary.to_dict(orient="records")):
        key = (str(row["experiment_id"]), str(row["cycle_id"]))
        if key in names:
            raise ValueError(f"duplicate cycle identity: {key}")
        names[key] = format_cycle_name(start_index + offset)
    return names


def _next_v2_cycle_index(index: pd.DataFrame) -> int:
    values = [
        parse_cycle_name(str(value))
        for value in index.get("cycle_name", pd.Series(dtype="string")).dropna()
    ]
    return max(values) + 1 if values else 1


def _write_v2_source_assets(
    descriptor: Mapping[str, Any],
    *,
    cycle_names: Mapping[CycleKey, str],
    dataset_id: str,
    staging_dataset: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], str]:  # noqa: C901
    from .dataset_coverage import render_rgb_coverage
    from .dataset_images import (
        collect_cycle_images,
        copy_dataset_image,
        rewrite_processed_image_paths_v2,
    )
    from .dataset_io import write_atomic_parquet
    from .visualization import render_cycle_publication

    source = descriptor["source"]
    summary = descriptor["summary"]
    prepared = descriptor["prepared"]
    processed = descriptor["processed"]
    records, inventory_hash = collect_cycle_images(
        prepared,
        input_dir=source.input_dir,
        cycle_names=cycle_names,
    )
    rewritten = rewrite_processed_image_paths_v2(processed, records)
    for record in records:
        copy_dataset_image(record, staging_dataset)
    image_metadata = _v2_image_metadata_frame(records)
    write_atomic_parquet(image_metadata, staging_dataset / "image_metadata.parquet")
    cycle_rows: list[dict[str, object]] = []
    cycle_records: list[dict[str, object]] = []
    for row in summary.to_dict(orient="records"):
        key = (str(row["experiment_id"]), str(row["cycle_id"]))
        cycle_name = cycle_names[key]
        cycle_uid = make_v2_cycle_uid(*key)
        cycle_frame = rewritten.loc[
            rewritten["experiment_id"].astype(str).eq(key[0])
            & rewritten["cycle_id"].astype(str).eq(key[1])
        ].copy()
        if cycle_frame.empty:
            cycle_frame = processed.iloc[0:0].copy()
        parquet_path = staging_dataset / "cycles" / f"{cycle_name}.parquet"
        csv_path = staging_dataset / "cycles" / f"{cycle_name}.csv"
        write_atomic_parquet(cycle_frame, parquet_path)
        cycle_frame.to_csv(csv_path, index=False)
        cycle_images = _records_frame(
            [record for record in records if str(record["cycle_uid"]) == cycle_uid]
        )
        prepared_cycle = prepared.loc[
            prepared["experiment_id"].astype(str).eq(key[0])
            & prepared["cycle_id"].astype(str).eq(key[1])
        ]
        record = _v2_cycle_record(
            row,
            cycle_name=cycle_name,
            cycle_uid=cycle_uid,
            cycle_frame=cycle_frame,
            prepared_cycle=prepared_cycle,
            cycle_images=cycle_images,
            data_path=f"cycles/{cycle_name}.parquet",
            csv_path=f"cycles/{cycle_name}.csv",
        )
        publication_path = staging_dataset / "cycles" / f"{cycle_name}.png"
        coverage_path = staging_dataset / "cycles" / f"{cycle_name}_rgb_coverage.png"
        render_cycle_publication(cycle_frame, record, publication_path)
        render_rgb_coverage(
            cycle_frame if not cycle_frame.empty else prepared_cycle,
            cycle_images,
            record,
            coverage_path,
        )
        record["publication_path"] = f"cycles/{cycle_name}.png"
        record["rgb_coverage_path"] = f"cycles/{cycle_name}_rgb_coverage.png"
        record["asset_sha256"] = {
            "parquet": sha256_file(parquet_path),
            "csv": sha256_file(csv_path),
            "publication": sha256_file(publication_path),
            "rgb_coverage": sha256_file(coverage_path),
        }
        cycle_records.append(record)
        cycle_rows.append(
            {
                "cycle_name": cycle_name,
                "cycle_uid": cycle_uid,
                "experiment_id": key[0],
                "experiment_date": str(row["experiment_date"])[:10],
                "source_cycle_id": key[1],
                "start_time": record["start_time"],
                "end_time": record["end_time"],
                "duration_seconds": record["duration_seconds"],
                "row_count": int(len(cycle_frame)),
                "data_path": record["data_path"],
                "csv_path": record["csv_path"],
                "publication_path": record["publication_path"],
                "rgb_coverage_path": record["rgb_coverage_path"],
                "image_count": int(len(cycle_images)),
                "cycle_status": row.get("cycle_status"),
                "cycle_status_reason": row.get("cycle_status_reason"),
                "baseline_status": row.get("baseline_status"),
            }
        )
    return pd.DataFrame(cycle_rows), image_metadata, cycle_records, inventory_hash


def _v2_cycle_record(
    row: Mapping[str, object],
    *,
    cycle_name: str,
    cycle_uid: str,
    cycle_frame: pd.DataFrame,
    prepared_cycle: pd.DataFrame,
    cycle_images: pd.DataFrame,
    data_path: str,
    csv_path: str,
) -> dict[str, object]:
    from .dataset_coverage import image_summary
    from .dataset_manifest import automatic_assessment

    timestamp_values = cycle_frame.get(
        "timestamp",
        prepared_cycle.get("timestamp", pd.Series(dtype="datetime64[ns]")),
    )
    timestamps = pd.to_datetime(timestamp_values, errors="coerce").dropna()
    if timestamps.empty:
        timestamps = pd.to_datetime(
            pd.Series(
                [row.get("heating_start"), row.get("defrost_end")], dtype="object"
            ),
            errors="coerce",
        ).dropna()
    start_time = timestamps.min().isoformat() if not timestamps.empty else None
    end_time = timestamps.max().isoformat() if not timestamps.empty else None
    duration = (
        float((timestamps.max() - timestamps.min()).total_seconds())
        if len(timestamps) > 1
        else 0.0
    )
    stages = set(cycle_frame.get("cycle_stage", pd.Series(dtype=str)).dropna().astype(str))
    data_summary = {
        "has_sensor_data": bool(len(cycle_frame)),
        "has_heating_stage": bool({"recovery", "frost_development"} & stages),
        "has_frosting_stage": "frost_development" in stages,
        "has_defrost_stage": "defrost" in stages,
        "has_recovery_stage": "recovery" in stages,
        "long_gap_count": _long_gap_count(timestamps),
        "maximum_gap_seconds": _maximum_gap(timestamps),
    }
    return {
        "cycle_name": cycle_name,
        "cycle_uid": cycle_uid,
        "experiment_id": str(row["experiment_id"]),
        "experiment_date": str(row["experiment_date"])[:10],
        "source_cycle_id": str(row["cycle_id"]),
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration,
        "row_count": int(len(cycle_frame)),
        "data_summary": data_summary,
        "image_summary": image_summary(
            cycle_frame if not cycle_frame.empty else prepared_cycle,
            cycle_images,
        ),
        "assessment": automatic_assessment(
            row.get("cycle_status"), row.get("cycle_status_reason")
        ),
        "data_path": data_path,
        "csv_path": csv_path,
    }


def _v2_image_metadata_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "image_id",
        "cycle_name",
        "cycle_uid",
        "source_camera_id",
        "initial_camera_slot",
        "source_role",
        "image_time",
        "matched_timestamp",
        "offset_seconds",
        "cycle_stage",
        "source_relative_path",
        "sha256",
        "file_size_bytes",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        [{column: record[column] for column in columns} for record in records],
        columns=columns,
    )


def _records_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["image_id", "camera_role", "image_time"])
    rows = []
    for record in records:
        rows.append({**record, "path": record["image_path"]})
    return pd.DataFrame(rows)


def _collect_v2_inventory_hash(descriptor: Mapping[str, Any]) -> str:
    from .dataset_images import collect_cycle_images

    records, inventory_hash = collect_cycle_images(
        descriptor["prepared"],
        input_dir=descriptor["source"].input_dir,
        cycle_names=_assign_v2_cycle_names(descriptor["summary"], start_index=1),
    )
    _ = records
    return inventory_hash


def _v2_asset_paths(
    cycle_index: pd.DataFrame,
    image_metadata: pd.DataFrame,
    staging_dataset: Path,
) -> list[str]:
    paths: list[str] = []
    for row in cycle_index.to_dict(orient="records"):
        for column in ("data_path", "csv_path", "publication_path", "rgb_coverage_path"):
            paths.append(str(row[column]))
    for cycle_name in image_metadata["cycle_name"].astype(str).unique():
        image_root = staging_dataset / "images" / cycle_name
        if image_root.is_dir():
            paths.extend(
                path.relative_to(staging_dataset).as_posix()
                for path in image_root.rglob("*")
                if path.is_file()
            )
    return paths


def _add_v2_file_info(manifest: dict[str, object], dataset_dir: Path) -> None:
    manifest["files"] = {
        "cycle_index": {
            "path": "cycle_index.parquet",
            "sha256": sha256_file(dataset_dir / "cycle_index.parquet"),
        },
        "image_metadata": {
            "path": "image_metadata.parquet",
            "sha256": sha256_file(dataset_dir / "image_metadata.parquet"),
        },
    }


def _long_gap_count(times: pd.Series) -> int:
    if len(times) < 2:
        return 0
    gaps = times.sort_values().diff().dropna().dt.total_seconds()
    return int((gaps > 30.0).sum())


def _maximum_gap(times: pd.Series) -> float:
    if len(times) < 2:
        return 0.0
    gaps = times.sort_values().diff().dropna().dt.total_seconds()
    return float(gaps.max()) if not gaps.empty else 0.0


def _write_v2_asset_readme(staging_dataset: Path, dataset_id: str) -> None:
    (staging_dataset / "README.md").write_text(_v2_readme(dataset_id), encoding="utf-8")


def _v2_readme(dataset_id: str) -> str:
    return (
        f"# {dataset_id}\n\n"
        "This self-contained Dataset stores every source cycle, its Processed values, "
        "publication figures, RGB coverage figure, and matched image metadata.\n\n"
        "Use frost_analysis.dataset_loader.DatasetLoader as the read-only downstream "
        "entry point.\n"
        "The current image parent directory is the camera role; image metadata records "
        "source facts.\n"
    )
