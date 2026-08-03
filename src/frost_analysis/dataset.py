"""Cycle-level dataset publication helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .config import find_project_root, is_iso_date
from .io import ensure_output_outside_input, relative_posix_path, sha256_file

DATASET_SCHEMA_VERSION = 1
CYCLE_NAME_WIDTH = 6
_DATASET_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_CYCLE_NUMBER_RE = re.compile(r"(?:^|_)cycle_(\d+)$")


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


def validate_dataset_id(value: str) -> str:
    """Validate and return a dataset identifier suitable for a directory name."""
    if not _DATASET_ID_RE.fullmatch(value):
        raise ValueError(f"invalid dataset_id: {value!r}")
    return value


def make_cycle_uid(experiment_id: str, cycle_id: str) -> str:
    """Build the stable source identity for one cycle."""
    return f"{experiment_id}__{cycle_id}"


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
    processed_counts: Mapping[tuple[str, str], int],
) -> dict[tuple[str, str], str]:
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
    processed_counts: Mapping[tuple[str, str], int],
    cycle_names: Mapping[tuple[str, str], str],
    cycle_files: Mapping[tuple[str, str], Mapping[str, object]],
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
    if len(left) != len(right):
        raise ValueError("schema column count differs")
    merged: list[dict[str, Any]] = []
    for left_field, right_field in zip(left, right, strict=True):
        if left_field["name"] != right_field["name"]:
            raise ValueError("schema column names or order differ")
        left_type = str(left_field["logical_type"])
        right_type = str(right_field["logical_type"])
        if left_type == "null":
            logical_type = right_type
        elif right_type == "null":
            logical_type = left_type
        elif left_type != right_type:
            raise ValueError(
                f"schema type differs for {left_field['name']}: {left_type} vs {right_type}"
            )
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
    sources = [cast(SourceRun, descriptor["source"]) for descriptor in descriptors]
    experiment_ids = [source.experiment_id for source in sources]
    if len(set(experiment_ids)) != len(experiment_ids):
        duplicates = sorted({value for value in experiment_ids if experiment_ids.count(value) > 1})
        raise ValueError(f"duplicate experiment_id in build inputs: {duplicates}")
    project_roots = {descriptor["project_root"] for descriptor in descriptors}
    if len(project_roots) != 1:
        raise ValueError("source runs must share one project root")
    for source in sources:
        ensure_output_outside_input(output_dir, source.input_dir)

    summary_frames = [cast(pd.DataFrame, descriptor["summary"]) for descriptor in descriptors]
    summary = pd.concat(summary_frames, ignore_index=True)
    summary = _sort_summary(summary)
    processed_counts: dict[tuple[str, str], int] = {}
    for descriptor in descriptors:
        processed_counts.update(
            cast(dict[tuple[str, str], int], descriptor["processed_counts"])
        )
    cycle_names = assign_cycle_names(summary, processed_counts)
    merged_schema = cast(list[dict[str, Any]], descriptors[0]["schema"])
    for descriptor in descriptors[1:]:
        merged_schema = logical_schema_compatible(
            merged_schema, cast(list[dict[str, Any]], descriptor["schema"])
        )

    staging_root, staging_dataset = create_build_staging(output_dir)
    published = False
    try:
        cycle_files: dict[tuple[str, str], dict[str, object]] = {}
        image_records: list[dict[str, object]] = []
        source_manifests: list[dict[str, object]] = []
        for descriptor in sorted(
            descriptors,
            key=lambda item: (
                cast(SourceRun, item["source"]).experiment_date,
                cast(SourceRun, item["source"]).experiment_id,
            ),
        ):
            export = _load_source_export(descriptor, cycle_names)
            source = cast(SourceRun, export["source"])
            cycle_files.update(
                _write_source_export(
                    export,
                    dataset_id=dataset_id,
                    staging_dataset=staging_dataset,
                )
            )
            records = cast(list[dict[str, object]], export["records"])
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
    source = cast(SourceRun, descriptor["source"])
    dataset_root = find_project_root(dataset_dir)
    source_root = cast(Path | None, descriptor["project_root"])
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
    processed_counts = cast(dict[tuple[str, str], int], descriptor["processed_counts"])
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
        cast(list[dict[str, Any]], descriptor["schema"]),
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
        new_records = cast(list[dict[str, object]], export["records"])
        new_cycle_index = build_cycle_index(
            cast(pd.DataFrame, export["summary"]),
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
    except Exception:
        if committed or moved_files or backup_dir is not None:
            rollback_append(dataset_dir, backup_dir, moved_files)
        raise
    finally:
        cleanup_append_staging(staging_root)


def _source_descriptor(run_path: Path) -> dict[str, object]:
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
    descriptor: Mapping[str, object],
    cycle_names: Mapping[tuple[str, str], str],
) -> dict[str, object]:
    """Load, scientifically validate, and transform one source run in memory."""
    from .dataset_images import collect_matched_images, rewrite_processed_image_paths
    from .validation import validate_prepared, validate_processed

    source = cast(SourceRun, descriptor["source"])
    summary = cast(pd.DataFrame, descriptor["summary"])
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
    export: Mapping[str, object],
    *,
    dataset_id: str,
    staging_dataset: Path,
) -> dict[tuple[str, str], dict[str, object]]:
    """Write one transformed source run into a build or append staging tree."""
    from .dataset_images import copy_dataset_image
    from .dataset_io import write_atomic_parquet

    records = cast(list[dict[str, object]], export["records"])
    rewritten = cast(pd.DataFrame, export["rewritten"])
    processed_counts = cast(dict[tuple[str, str], int], export["processed_counts"])
    cycle_names = cast(Mapping[tuple[str, str], str], export["cycle_names"])
    cycle_files: dict[tuple[str, str], dict[str, object]] = {}
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
