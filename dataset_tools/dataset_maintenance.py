"""Build and manage the self-contained Cycle Dataset schema 3.

The module owns orchestration only; scientific transforms remain in Prepare/Process
and image, metadata, IO, and validation concerns live in their focused modules.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .builder.build_experiment import ExperimentBuild, build_experiment
from .builder.dataset_settings import find_project_root
from .cycle_metadata import DATASET_ID, DATASET_SCHEMA_VERSION, image_root
from .dataset_paths import require_external_storage

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


def add_dataset(input_dir: Path, dataset_dir: Path | None = None) -> Path:
    """Build or append one date directly from raw input."""
    input_path = Path(input_dir).resolve()
    require_external_storage(input_path)
    project_root = _resolve_project_root()
    target = Path(dataset_dir).resolve() if dataset_dir is not None else project_root / "dataset"
    _validate_date_input(input_path)
    config = _load_config_for_input(input_path, project_root, target)
    print(
        f"[add] input={input_path}\n[add] dataset={target}\n[add] images={image_root(target)}\n[add] writes measurements, metadata, images and cycle figures; new channels may update existing tables",
        flush=True,
    )

    from .cycle_metadata import read_manifest

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
            print(f"[add] experiment already exists: {experiment_id}")
            return target
        if experiments:
            last_date = max(str(item["experiment_date"])[:10] for item in experiments)
            if experiment_date <= last_date:
                raise ValueError(
                    "historical or same-date append is not supported; remove that date first"
                )

        print("[add] preparing and processing")
        build = build_experiment(input_path, config)
        _append_build(target, build)
        print("[add] done")
        return target

    print("[add] preparing and processing")
    build = build_experiment(input_path, config)
    _materialize_builds(target, [build])
    print("[add] done")
    return target


def replace_dataset(input_dir: Path, dataset_dir: Path | None = None) -> Path:  # noqa: C901
    """Rebuild one published experiment and renumber everything after it."""
    import shutil
    import tempfile

    from .builder.raw import discover_inputs
    from .builder.schema import (
        align_original_schema,
        build_processed_frame,
        build_registry,
        merge_original_columns,
        merge_registries,
    )
    from .cycle_metadata import read_catalog
    from .dataset_paths import read_json, write_csv, write_json, write_parquet
    from .local_images import collect_cycle_images, copy_image, image_metadata_frame
    from .validate_dataset import validate_dataset

    input_path = Path(input_dir).resolve()
    require_external_storage(input_path)
    project_root = _resolve_project_root()
    root = Path(dataset_dir).resolve() if dataset_dir is not None else project_root / "dataset"
    _validate_date_input(input_path)
    config = _load_config_for_input(input_path, project_root, root)
    experiment_id = str(config.experiment_id)
    catalog = read_catalog(root)
    records = [record for record in catalog["cycles"] if isinstance(record, dict)]
    replaced = [record for record in records if str(record["experiment_id"]) == experiment_id]
    if not replaced:
        raise ValueError(f"experiment is not published: {experiment_id}")

    print(f"[replace] rebuilding {experiment_id}")
    build = build_experiment(input_path, config)
    start = min(parse_cycle_name(str(record["cycle_name"])) for record in replaced)
    old_end = max(parse_cycle_name(str(record["cycle_name"])) for record in replaced)
    names = assign_final_cycle_names_by_time(
        build.summary,
        prepared=build.prepared,
        start_index=start,
    )
    delta = len(names) - len(replaced)
    later = [record for record in records if parse_cycle_name(str(record["cycle_name"])) > old_end]
    earlier = [record for record in records if parse_cycle_name(str(record["cycle_name"])) < start]

    old_registry = read_json(root / "channel_registry.json")
    registry = merge_registries(old_registry, build_registry([build]))
    for key in (
        "image_coverage",
        "heating_branches",
        "baseline_seconds",
        "baseline_managed",
        "recovery_edit",
    ):
        if key in old_registry:
            registry[key] = old_registry[key]
    original_columns: list[str] = []
    for record in records:
        for column in pd.read_csv(root / str(record["assets"]["original_csv"]), nrows=0):
            if str(column) not in original_columns:
                original_columns.append(str(column))
    for column in merge_original_columns([build]):
        if column not in original_columns:
            original_columns.append(column)

    images = collect_cycle_images(
        list(discover_inputs(config).image_files),
        input_dir=input_path,
        cycles=_image_cycle_windows(build, names),
    )
    new_metadata = image_metadata_frame(images)
    old_metadata = pd.read_parquet(image_root(root) / "image_metadata.parquet")
    replaced_names = {str(record["cycle_name"]) for record in replaced}
    later_names = {
        str(record["cycle_name"]): format_cycle_name(
            parse_cycle_name(str(record["cycle_name"])) + delta
        )
        for record in later
    }
    kept_metadata = old_metadata.loc[
        ~old_metadata["cycle_name"].astype(str).isin(replaced_names)
    ].copy()
    kept_metadata["cycle_name"] = kept_metadata["cycle_name"].replace(later_names)

    reusable_images: dict[tuple[str, str], Path] = {}
    for record in replaced:
        cycle_root = root / "images" / str(record["cycle_name"])
        if not cycle_root.is_dir():
            continue
        for role in cycle_root.iterdir():
            if role.is_dir():
                for path in role.iterdir():
                    if path.is_file():
                        reusable_images.setdefault((role.name, path.name), path.relative_to(root))

    backup = Path(tempfile.mkdtemp(prefix="frost-replace-", dir=root.parent))
    moved: list[tuple[Path, Path]] = []
    created: list[Path] = []

    def move(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        moved.append((source, target))

    for name in ("cycle_catalog.json", "channel_registry.json"):
        shutil.copy2(root / name, backup / name)
    shutil.copy2(image_root(root) / "image_metadata.parquet", backup / "image_metadata.parquet")

    try:
        # Move affected files aside first; image directories move without reading contents.
        for record in [*replaced, *later]:
            for relative in record["assets"].values():
                source = root / str(relative)
                if source.is_file():
                    move(source, backup / source.relative_to(root))
            source = root / "images" / str(record["cycle_name"])
            if source.is_dir():
                move(source, backup / source.relative_to(root))

        # Restore later cycles under their shifted Dataset names.
        shifted: list[dict[str, Any]] = []
        from .cycle_metadata import cycle_assets

        for record in later:
            old_name = str(record["cycle_name"])
            new_name = later_names[old_name]
            directory = Path(str(record["assets"]["csv"])).parent.name
            shifted_record = {
                **record,
                "cycle_name": new_name,
                "assets": cycle_assets(new_name, rb=directory == "cycles_RB"),
            }
            control = record.get("defrost_control", {})
            if control.get("inherited_from") in later_names:
                shifted_record["defrost_control"] = {
                    **control,
                    "inherited_from": later_names[control["inherited_from"]],
                }
            for key, target_relative in shifted_record["assets"].items():
                source_relative = record["assets"].get(key)
                if source_relative is None:
                    continue
                source = backup / str(source_relative)
                if source.is_file():
                    move(source, root / target_relative)
            image_source = backup / "images" / old_name
            if image_source.is_dir():
                move(image_source, root / "images" / new_name)
            frame = pd.read_parquet(root / shifted_record["assets"]["parquet"])
            frame["cycle_name"] = new_name
            write_parquet(frame, root / shifted_record["assets"]["parquet"])
            write_csv(frame, root / shifted_record["assets"]["csv"])
            shifted.append(shifted_record)

        copied = 0
        reused = 0
        print(f"[replace] publishing images: 0/{len(images)}")
        for index, image in enumerate(images, start=1):
            path = root / str(image["image_path"])
            reusable = reusable_images.get((str(image["camera_role"]), str(image["file_name"])))
            reusable_path = backup / reusable if reusable is not None else None
            if reusable_path is not None and reusable_path.is_file():
                move(reusable_path, path)
                reused += 1
            else:
                created.append(path)
                copy_image(image, root)
                copied += 1
            if index == len(images) or index % 500 == 0:
                print(f"[replace] publishing images: {index}/{len(images)}", flush=True)

        summary_lookup = {
            (str(row["experiment_id"]), str(row["cycle_id"])): row
            for row in build.summary.to_dict(orient="records")
        }
        new_records: list[dict[str, Any]] = []
        old_by_uid = {str(record["cycle_uid"]): record for record in replaced}
        for index, (key, cycle_name) in enumerate(
            sorted(names.items(), key=lambda item: parse_cycle_name(item[1])), start=1
        ):
            print(f"[replace] rendering {index}/{len(names)} {cycle_name}", flush=True)
            expected_assets = cycle_assets(cycle_name)
            created.extend(root / relative for relative in expected_assets.values())
            record, new_metadata = _materialize_cycle(
                root,
                build,
                key,
                cycle_name,
                registry,
                original_columns,
                new_metadata,
                summary_lookup[key],
                previous=new_records[-1]
                if new_records
                else max(
                    earlier,
                    key=lambda row: row["cycle_name"],
                    default=None,
                ),
            )
            reviewed = old_by_uid.get(str(record["cycle_uid"]))
            same_boundaries = reviewed is not None and all(
                reviewed.get("boundaries", {}).get(name) == record.get("boundaries", {}).get(name)
                for name in ("heating_start", "defrost_start", "defrost_end")
            )
            if (
                same_boundaries
                and reviewed is not None
                and (reviewed.get("status"), reviewed.get("status_reason"))
                != (
                    reviewed.get("pipeline_status"),
                    reviewed.get("pipeline_status_reason"),
                )
            ):
                record["status"] = reviewed.get("status")
                record["status_reason"] = reviewed.get("status_reason")
            for stage in ("frost", "defrost"):
                field = f"rgb_{stage}_status"
                automatic = f"rgb_{stage}_auto_status"
                if (
                    same_boundaries
                    and reviewed is not None
                    and reviewed.get(field) != reviewed.get(automatic)
                ):
                    record[field] = reviewed.get(field)
            new_records.append(record)

        final_records = [*earlier, *new_records, *shifted]
        final_metadata = pd.concat([kept_metadata, new_metadata], ignore_index=True)
        catalog["cycles"] = final_records
        write_parquet(final_metadata, image_root(root) / "image_metadata.parquet")
        write_json(registry, root / "channel_registry.json")
        write_json(catalog, root / "cycle_catalog.json")
        align_original_schema(root, final_records, original_columns)

        if list(registry.get("columns", [])) != list(old_registry.get("columns", [])):
            for record in [*earlier, *shifted]:
                path = root / str(record["assets"]["parquet"])
                frame = pd.read_parquet(path).drop(
                    columns=["cycle_name", "cycle_uid"], errors="ignore"
                )
                frame = build_processed_frame(
                    frame,
                    registry,
                    cycle_name=str(record["cycle_name"]),
                    cycle_uid=str(record["cycle_uid"]),
                )
                write_parquet(frame, path)
                write_csv(frame, root / str(record["assets"]["csv"]))

        validate_dataset(root)
    except Exception:
        for path in reversed(created):
            if path.is_file():
                path.unlink()
        for source, target in reversed(moved):
            if target.exists():
                if source.is_dir():
                    shutil.rmtree(source)
                elif source.is_file():
                    source.unlink()
                source.parent.mkdir(parents=True, exist_ok=True)
                target.rename(source)
        image_parents = {
            parent
            for path in created
            if "images" in path.parts
            for parent in (path.parent, path.parent.parent)
        }
        for directory in sorted(image_parents, key=lambda path: len(path.parts), reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        for name in ("cycle_catalog.json", "channel_registry.json"):
            shutil.copy2(backup / name, root / name)
        shutil.copy2(backup / "image_metadata.parquet", image_root(root) / "image_metadata.parquet")
        shutil.rmtree(backup, ignore_errors=True)
        raise

    shutil.rmtree(backup)
    print(f"[replace] cycles={len(replaced)}->{len(names)}, shift={delta:+d}")
    print(f"[replace] images reused={reused}, copied={copied}")
    print("[replace] done")
    return root


def remove_dataset(dataset_dir: Path, date: str) -> Path:  # noqa: C901
    """Remove one experiment without renumbering or rewriting other cycles."""
    import shutil

    from .cycle_metadata import image_root, read_catalog, read_manifest, write_catalog
    from .dataset_paths import write_json, write_parquet
    from .validate_dataset import validate_dataset

    root = Path(dataset_dir).resolve()
    manifest = read_manifest(root)
    selector = str(date).strip()
    matches = [
        item
        for item in manifest["experiments"]
        if isinstance(item, Mapping)
        and (
            str(item.get("experiment_id")) == selector
            or str(item.get("experiment_date")) == selector
            or str(item.get("experiment_date", "")).replace("-", "")[4:] == selector
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"Dataset remove requires one matching experiment: {selector!r}")
    experiment_id = str(matches[0]["experiment_id"])
    catalog = read_catalog(root)
    removed = [
        record
        for record in catalog["cycles"]
        if isinstance(record, dict) and str(record.get("experiment_id")) == experiment_id
    ]
    if not removed:
        raise ValueError(f"experiment has no cycles: {experiment_id}")

    print(f"[remove] date={selector}")
    print(f"[remove] experiment={experiment_id}")
    print(f"[remove] cycles={len(removed)}")
    for record in removed:
        cycle_name = str(record["cycle_name"])
        print(f"[remove] deleting {cycle_name}")
        assets = record.get("assets", {})
        if isinstance(assets, Mapping):
            for relative in assets.values():
                target = (root / str(relative)).resolve()
                if not target.is_relative_to(root):
                    raise ValueError(f"cycle asset escapes Dataset: {relative}")
                if target.is_file():
                    target.unlink()
        images = image_root(root, manifest)
        image_dir = (images / cycle_name).resolve()
        if not image_dir.is_relative_to(images):
            raise ValueError(f"invalid cycle image directory: {cycle_name}")
        if image_dir.is_dir():
            shutil.rmtree(image_dir)

    removed_names = {str(record["cycle_name"]) for record in removed}
    metadata = pd.read_parquet(image_root(root) / "image_metadata.parquet")
    metadata = metadata.loc[~metadata["cycle_name"].astype(str).isin(removed_names)].copy()
    print("[remove] updating image_metadata.parquet")
    write_parquet(metadata, image_root(root) / "image_metadata.parquet")
    catalog["cycles"] = [record for record in catalog["cycles"] if record not in removed]
    write_catalog(root, catalog)
    manifest["experiments"] = [item for item in manifest["experiments"] if item not in matches]
    for experiment in manifest["experiments"]:
        if isinstance(experiment, dict):
            experiment.pop("camera_roles", None)
    manifest.setdefault("images_root", "images")
    write_json(manifest, root / "dataset_manifest.json")
    print("[remove] validating Dataset")
    validate_dataset(root)
    print("[remove] done")
    return root


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


def _load_config_for_input(
    input_path: Path, project_root: Path, dataset_dir: Path | None = None
) -> Any:
    from .builder.dataset_settings import load_config

    # EDF 可能分段或补录；实验日期只认控制器导出文件名。
    dates: set[str] = set()
    for pattern in ("*.xls", "*.xlsx"):
        for path in input_path.glob(pattern):
            if not path.is_file() or path.name.startswith((".", "~$")):
                continue
            expression = r"\d{4}-\d{2}-\d{2}" if pattern == "*.xls" else r"(?<=DataExport)\d{8}"
            match = re.search(expression, path.name)
            if match is None:
                raise ValueError("sensor filenames must contain one shared date")
            value = match.group()
            dates.add(value if "-" in value else f"{value[:4]}-{value[4:6]}-{value[6:8]}")
    if len(dates) != 1:
        raise ValueError("sensor filenames must contain one shared date")
    config = load_config(
        project_root=project_root,
        experiment_date=dates.pop(),
        input_dir=input_path,
    )
    return config


def _materialize_cycle(
    dataset_dir: Path,
    build: ExperimentBuild,
    key: CycleKey,
    cycle_name: str,
    registry: Mapping[str, Any],
    original_columns: Sequence[str],
    image_metadata: pd.DataFrame,
    summary_row: Mapping[str, Any],
    previous: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Write one complete cycle and return its Catalog record."""
    from plots.cycle import render_cycle_publication, render_rgb_panel

    from .builder.schema import build_processed_frame, export_original_frame
    from .cycle_metadata import build_cycle_record, cycle_assets, defrost_control
    from .dataset_paths import write_csv, write_parquet
    from .edit_cycle_metadata import apply_recovery
    from .local_images import (
        _cycle_image_summary,
        _sensor_coverage_intervals,
        rgb_overall_intervals,
        rgb_stage_metrics,
        scan_cycle_images,
    )

    original_source = build.original if build.original is not None else build.prepared
    original = export_original_frame(
        original_source.loc[
            original_source["experiment_id"].astype(str).eq(key[0])
            & original_source["cycle_id"].astype(str).eq(key[1])
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
    control = defrost_control(original, summary_row, previous)
    rb = control["mode"] == "rb"
    assets = cycle_assets(cycle_name, rb=rb)
    record = build_cycle_record(
        summary_row,
        cycle_name=cycle_name,
        cycle_uid=make_cycle_uid(*key),
        processed=canonical,
        original=original,
        image_summary={"image_count": 0},
        assets=assets,
    )
    record["defrost_control"] = control
    metadata_result = image_metadata
    recovery = registry.get("recovery_edit")
    if isinstance(recovery, Mapping) and bool(recovery.get("managed", False)):
        mode = str(recovery.get("mode", ""))
        if mode in {"seconds", "ts-minus", "frequency-setpoint", "frequency-actual"}:
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
    from heat_smoothing.methods import process_measurements

    original["timestamp"] = pd.to_datetime(original.timestamp)
    canonical = process_measurements(original, canonical)
    write_parquet(canonical, dataset_dir / assets["parquet"])
    write_csv(canonical, dataset_dir / assets["csv"])
    write_csv(original, dataset_dir / assets["original_csv"])
    image_summary, intervals = _cycle_image_summary(
        dataset_dir,
        cycle_name,
        canonical,
        metadata_result,
        registry,
    )
    roles = tuple(
        sorted(
            metadata_result.loc[
                metadata_result["cycle_name"].astype(str).eq(cycle_name), "camera_role"
            ]
            .dropna()
            .astype(str)
            .unique()
        )
    )
    record["image"] = image_summary
    _update_rgb_record(record, rgb_stage_metrics(canonical, intervals, roles))
    render_cycle_publication(
        canonical,
        record,
        dataset_dir / assets["publication"],
        sensor_intervals=_sensor_coverage_intervals(canonical, registry),
        rgb_intervals=rgb_overall_intervals(canonical, intervals, roles),
    )
    render_rgb_panel(
        record,
        canonical,
        scan_cycle_images(dataset_dir, cycle_name, metadata_result),
        intervals,
        roles,
        dataset_dir / assets["rgb_panel"],
    )
    return record, metadata_result


def _materialize_builds(  # noqa: C901
    dataset_dir: Path,
    builds: Sequence[ExperimentBuild],
) -> None:
    from .builder.raw import discover_inputs
    from .builder.schema import build_registry, merge_original_columns
    from .cycle_metadata import experiment_record
    from .dataset_paths import write_json, write_parquet
    from .local_images import collect_cycle_images, copy_image, image_metadata_frame

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
            collect_cycle_images(
                list(discover_inputs(build.config).image_files),
                input_dir=build.input_dir,
                cycles=_image_cycle_windows(build, names),
            )
        )
    image_metadata = image_metadata_frame(all_images)
    print(f"[add] copying images: 0/{len(all_images)}")
    for index, image in enumerate(all_images, start=1):
        copy_image(image, dataset_dir)
        if index == len(all_images) or index % 500 == 0:
            print(f"[add] copying images: {index}/{len(all_images)}", flush=True)

    summary_lookup: dict[CycleKey, dict[str, Any]] = {
        (str(row["experiment_id"]), str(row["cycle_id"])): {
            str(key): value for key, value in row.items()
        }
        for row in summary.to_dict(orient="records")
    }
    builds_by_experiment = {str(build.config.experiment_id): build for build in builds}
    records: list[dict[str, Any]] = []
    ordered_names = sorted(names.items(), key=lambda item: parse_cycle_name(item[1]))
    for index, (key, cycle_name) in enumerate(ordered_names, start=1):
        print(
            f"[add] rendering cycles: {index}/{len(ordered_names)} {cycle_name}",
            flush=True,
        )
        record, image_metadata = _materialize_cycle(
            dataset_dir,
            builds_by_experiment[key[0]],
            key,
            cycle_name,
            registry,
            original_columns,
            image_metadata,
            summary_lookup[key],
            previous=records[-1] if records else None,
        )
        records.append(record)

    write_parquet(image_metadata, image_root(dataset_dir) / "image_metadata.parquet")

    write_json(registry, dataset_dir / "channel_registry.json")
    experiments = [
        experiment_record(
            str(build.config.experiment_id),
            str(build.config.experiment_date),
        )
        for build in sorted(builds, key=lambda item: str(item.config.experiment_date))
    ]
    write_json(
        {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "images_root": "images",
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
    builds: Sequence[ExperimentBuild],
) -> dict[CycleKey, int]:
    counts: dict[CycleKey, int] = {}
    for build in builds:
        for values in build.processed[["experiment_id", "cycle_id"]].itertuples(
            index=False, name=None
        ):
            key = (str(values[0]), str(values[1]))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _image_cycle_windows(
    build: ExperimentBuild,
    names: Mapping[CycleKey, str],
) -> list[dict[str, object]]:
    """Build the minimal cycle boundaries used to preserve Raw images."""
    summary = {
        (str(row["experiment_id"]), str(row["cycle_id"])): row
        for row in build.summary.to_dict(orient="records")
    }
    windows: list[dict[str, object]] = []
    for key, cycle_name in names.items():
        row = summary[key]
        scoped = build.processed.loc[
            build.processed["experiment_id"].astype(str).eq(key[0])
            & build.processed["cycle_id"].astype(str).eq(key[1])
        ]
        timestamps = pd.to_datetime(scoped["timestamp"], errors="coerce").dropna()
        start = pd.to_datetime(row.get("start_time"), errors="coerce")
        end = pd.to_datetime(row.get("end_time"), errors="coerce")
        windows.append(
            {
                "cycle_name": cycle_name,
                "start_time": timestamps.min() if pd.isna(start) else start,
                "end_time": timestamps.max() if pd.isna(end) else end,
                "stable_heating_start": row.get("stable_heating_start"),
                "defrost_preparation_start": row.get("defrost_preparation_start"),
                "defrost_start": row.get("defrost_start"),
                "defrost_end": row.get("defrost_end"),
            }
        )
    return sorted(windows, key=lambda item: pd.Timestamp(item["start_time"]))


def _append_build(  # noqa: C901
    dataset_dir: Path,
    build: ExperimentBuild,
) -> None:
    from .builder.raw import discover_inputs
    from .builder.schema import (
        align_original_schema,
        build_processed_frame,
        build_registry,
        merge_original_columns,
        merge_registries,
    )
    from .cycle_metadata import experiment_record, read_catalog, read_manifest
    from .dataset_paths import read_json, write_csv, write_json, write_parquet
    from .local_images import collect_cycle_images, copy_image, image_metadata_frame

    manifest = read_manifest(dataset_dir)
    catalog = read_catalog(dataset_dir)
    old_records = [record for record in catalog["cycles"] if isinstance(record, dict)]
    names = assign_final_cycle_names_by_time(
        build.summary,
        prepared=build.prepared,
        start_index=max(
            (parse_cycle_name(str(record["cycle_name"])) for record in old_records),
            default=0,
        )
        + 1,
    )
    old_registry = read_json(dataset_dir / "channel_registry.json")
    candidate = build_registry([build])
    merged_registry = merge_registries(old_registry, candidate)
    for setting_name in (
        "image_coverage",
        "heating_branches",
        "baseline_seconds",
        "baseline_managed",
        "recovery_edit",
    ):
        if setting_name in old_registry:
            merged_registry[setting_name] = old_registry[setting_name]
        elif setting_name in candidate:
            merged_registry[setting_name] = candidate[setting_name]

    old_images = pd.read_parquet(image_root(dataset_dir) / "image_metadata.parquet")
    new_images = collect_cycle_images(
        list(discover_inputs(build.config).image_files),
        input_dir=build.input_dir,
        cycles=_image_cycle_windows(build, names),
    )
    print(f"[add] copying images: 0/{len(new_images)}")
    for index, image in enumerate(new_images, start=1):
        copy_image(image, dataset_dir)
        if index == len(new_images) or index % 500 == 0:
            print(f"[add] copying images: {index}/{len(new_images)}", flush=True)
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
    ordered_names = sorted(names.items(), key=lambda item: parse_cycle_name(item[1]))
    for index, (key, cycle_name) in enumerate(ordered_names, start=1):
        print(
            f"[add] rendering cycles: {index}/{len(ordered_names)} {cycle_name}",
            flush=True,
        )
        record, merged_images = _materialize_cycle(
            dataset_dir,
            build,
            key,
            cycle_name,
            merged_registry,
            original_columns,
            merged_images,
            summary_lookup[key],
            previous=new_records[-1]
            if new_records
            else max(
                old_records,
                key=lambda row: row["cycle_name"],
                default=None,
            ),
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
    write_parquet(merged_images, image_root(dataset_dir) / "image_metadata.parquet")

    write_json(merged_registry, dataset_dir / "channel_registry.json")
    manifest["experiments"] = [
        *manifest["experiments"],
        {
            **experiment_record(
                str(build.config.experiment_id),
                str(build.config.experiment_date),
            ),
        },
    ]
    manifest["experiments"].sort(key=lambda value: str(value["experiment_date"]))
    manifest.setdefault("images_root", "images")
    write_json(manifest, dataset_dir / "dataset_manifest.json")
    write_json(catalog, dataset_dir / "cycle_catalog.json")


def _refresh_cycle_record(
    dataset_dir: Path,
    record: dict[str, Any],
    metadata: pd.DataFrame,
    registry: Mapping[str, Any],
    camera_roles: tuple[str, ...],
) -> None:
    from plots.cycle import render_cycle_publication, render_rgb_panel

    from .local_images import (
        _cycle_image_summary,
        _sensor_coverage_intervals,
        rgb_overall_intervals,
        rgb_stage_metrics,
        scan_cycle_images,
    )

    cycle_name = str(record["cycle_name"])
    assets = record.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError(f"cycle assets are missing: {cycle_name}")
    frame = pd.read_parquet(dataset_dir / str(assets["parquet"]))
    image_summary, intervals = _cycle_image_summary(
        dataset_dir, cycle_name, frame, metadata, registry
    )
    record["image"] = image_summary
    _update_rgb_record(
        record,
        rgb_stage_metrics(frame, intervals, camera_roles),
    )
    record.setdefault("data", {})["processed_row_count"] = int(len(frame))
    render_cycle_publication(
        frame,
        record,
        dataset_dir / str(assets["publication"]),
        sensor_intervals=_sensor_coverage_intervals(frame, registry),
        rgb_intervals=rgb_overall_intervals(frame, intervals, camera_roles),
    )
    render_rgb_panel(
        record,
        frame,
        scan_cycle_images(dataset_dir, cycle_name, metadata),
        intervals,
        camera_roles,
        dataset_dir / str(assets["rgb_panel"]),
    )


def _update_rgb_record(record: dict[str, Any], metrics: Mapping[str, object]) -> None:
    for stage in ("frost", "defrost"):
        coverage = f"rgb_{stage}_coverage"
        auto_status = f"rgb_{stage}_auto_status"
        human_status = f"rgb_{stage}_status"
        record[coverage] = metrics[coverage]
        record[auto_status] = metrics[auto_status]
        record.setdefault(human_status, metrics[auto_status])


def review_cycle(
    dataset_dir: Path,
    cycle_name: str,
    *,
    status: str,
    reason: str | None = None,
    rgb_frost: str | None = None,
    rgb_defrost: str | None = None,
) -> Path:
    """Update only the user-controlled status and its publication title."""
    from plots.dataset import render_publication_asset

    allowed = {"valid", "invalid"}
    if status not in allowed:
        raise ValueError(f"invalid Dataset status: {status}")
    rgb_allowed = {"valid", "invalid", "not_applicable"}
    for stage, value in (("frost", rgb_frost), ("defrost", rgb_defrost)):
        if value is not None and value not in rgb_allowed:
            raise ValueError(f"invalid RGB {stage} status: {value}")
    from .cycle_metadata import read_catalog, write_catalog

    catalog = read_catalog(dataset_dir)
    for record in catalog["cycles"]:
        if isinstance(record, dict) and record.get("cycle_name") == cycle_name:
            record["status"] = status
            record["review_status"] = status
            record["status_reason"] = reason
            if rgb_frost is not None:
                record["rgb_frost_status"] = rgb_frost
            if rgb_defrost is not None:
                record["rgb_defrost_status"] = rgb_defrost
            render_publication_asset(dataset_dir, record)
            write_catalog(dataset_dir, catalog)
            return dataset_dir
    raise KeyError(f"unknown cycle: {cycle_name}")


def edit_dataset(  # noqa: C901
    dataset_dir: Path,
    *,
    baseline_seconds: int | None = None,
    recovery_seconds: int | None = None,
    recovery_end_by: str | None = None,
    defrost_preparation: bool = False,
    render_rgb_panels: bool = True,
    render_figures: bool = True,
) -> Path:
    """Apply baseline or recovery edits directly to the Dataset."""
    from plots.dataset import _experiment_camera_roles, _render_rgb_panel

    if recovery_seconds is not None and recovery_end_by is not None:
        raise ValueError("--recovery-seconds and --recovery-end-by are mutually exclusive")
    if (
        baseline_seconds is None
        and recovery_seconds is None
        and recovery_end_by is None
        and not defrost_preparation
    ):
        raise ValueError("dataset edit requires at least one edit")

    if render_figures:
        from plots.cycle import render_cycle_publication

    from .cycle_metadata import read_catalog, write_catalog
    from .dataset_paths import read_json, write_csv, write_json, write_parquet
    from .edit_cycle_metadata import apply_baseline, apply_defrost_preparation, apply_recovery
    from .local_images import (
        _sensor_coverage_intervals,
        image_coverage_intervals,
        rgb_overall_intervals,
        rgb_stage_metrics,
        scan_cycle_images,
    )

    catalog = read_catalog(dataset_dir)
    registry = read_json(dataset_dir / "channel_registry.json")
    if not isinstance(registry, dict):
        raise ValueError("channel_registry.json must contain an object")
    stage_edit = recovery_seconds is not None or recovery_end_by is not None or defrost_preparation
    metadata = (
        pd.read_parquet(image_root(dataset_dir) / "image_metadata.parquet") if stage_edit else None
    )

    records = catalog["cycles"]
    published_columns = {}
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        cycle_name = str(record["cycle_name"])
        print(f"[edit] updating cycles: {index}/{len(records)} {cycle_name}", flush=True)
        assets = record["assets"]
        processed = pd.read_parquet(dataset_dir / str(assets["parquet"]))
        if recovery_seconds is not None or recovery_end_by is not None:
            assert metadata is not None
            original = pd.read_csv(dataset_dir / str(assets["original_csv"]))
            mask = metadata["cycle_name"].astype(str).eq(cycle_name)
            original, processed, cycle_metadata = apply_recovery(
                original,
                processed,
                metadata.loc[mask].copy(),
                record,
                registry,
                mode="seconds" if recovery_seconds is not None else str(recovery_end_by),
                seconds=recovery_seconds,
            )
            if mask.any():
                metadata.loc[mask, "cycle_stage"] = cycle_metadata["cycle_stage"].to_numpy()
            write_csv(original, dataset_dir / str(assets["original_csv"]))
        if defrost_preparation:
            assert metadata is not None
            original = pd.read_csv(dataset_dir / str(assets["original_csv"]))
            mask = metadata["cycle_name"].astype(str).eq(cycle_name)
            original, processed, cycle_metadata = apply_defrost_preparation(
                original,
                processed,
                metadata.loc[mask].copy(),
                record,
                registry,
            )
            if mask.any():
                metadata.loc[mask, "cycle_stage"] = cycle_metadata["cycle_stage"].to_numpy()
            write_csv(original, dataset_dir / str(assets["original_csv"]))
        if stage_edit:
            from heat_smoothing.methods import process_measurements

            original["timestamp"] = pd.to_datetime(original.timestamp)
            processed = process_measurements(original, processed)
        if baseline_seconds is not None:
            processed = apply_baseline(processed, record, registry, seconds=baseline_seconds)
        if stage_edit:
            from heat_smoothing.methods import MEASUREMENT_PROCESSING

            columns = [name for name in processed if name not in {"cycle_name", "cycle_uid"}]
            published_columns[cycle_name] = columns
            registry["columns"] = list(
                dict.fromkeys(name for group in published_columns.values() for name in group)
            )
            registry["measurement_processing"] = MEASUREMENT_PROCESSING
        write_parquet(processed, dataset_dir / str(assets["parquet"]))
        write_csv(processed, dataset_dir / str(assets["csv"]))
        if stage_edit:
            assert metadata is not None
            roles = _experiment_camera_roles(catalog, metadata, str(record["experiment_id"]))
            cycle_images = scan_cycle_images(dataset_dir, cycle_name, metadata)
            intervals = image_coverage_intervals(processed, cycle_images, registry)
            record["image"] = {"image_count": int(len(cycle_images))}
            _update_rgb_record(
                record,
                rgb_stage_metrics(processed, intervals, roles),
            )
        if render_figures:
            publication_metadata = (
                metadata.loc[metadata["cycle_name"].astype(str).eq(cycle_name)]
                if metadata is not None
                else pd.DataFrame(columns=["camera_role", "image_time"])
            )
            publication_roles = (
                _experiment_camera_roles(catalog, metadata, str(record["experiment_id"]))
                if metadata is not None
                else ()
            )
            publication_intervals = image_coverage_intervals(
                processed, publication_metadata, registry
            )
            render_cycle_publication(
                processed,
                record,
                dataset_dir / str(assets["publication"]),
                sensor_intervals=_sensor_coverage_intervals(processed, registry),
                rgb_intervals=rgb_overall_intervals(
                    processed, publication_intervals, publication_roles
                ),
            )
            if stage_edit and metadata is not None and render_rgb_panels:
                _render_rgb_panel(
                    dataset_dir,
                    record,
                    metadata,
                    registry,
                    publication_roles,
                    processed,
                    cycle_images,
                    intervals,
                )

    # Reprocessing can introduce interpolated/filled columns; all cycles share
    # the published schema, including a channel absent from one Original file.
    for record in records if stage_edit else ():
        if published_columns[record["cycle_name"]] == registry["columns"]:
            continue
        assets = record["assets"]
        frame = pd.read_parquet(dataset_dir / assets["parquet"])
        frame = frame.reindex(columns=["cycle_name", "cycle_uid", *registry["columns"]])
        write_parquet(frame, dataset_dir / assets["parquet"])
        write_csv(frame, dataset_dir / assets["csv"])
    if metadata is not None and stage_edit:
        write_parquet(metadata, image_root(dataset_dir) / "image_metadata.parquet")
    write_json(registry, dataset_dir / "channel_registry.json")
    write_catalog(dataset_dir, catalog)
    return dataset_dir


def refresh_dataset(dataset_dir: Path, mode: str) -> Path:  # noqa: C901
    """Refresh only the Dataset layer named by the user's physical edit."""
    from plots.dataset import _experiment_camera_roles, _render_rgb_panel, render_publication_asset

    from .cycle_metadata import image_root, read_catalog, read_manifest, write_catalog
    from .dataset_paths import read_json, write_json, write_parquet
    from .local_images import _image_timestamp, collect_cycle_images, image_metadata_frame
    from .validate_dataset import validate_dataset

    if mode not in {"roles", "images", "figures", "all"}:
        raise ValueError(f"invalid refresh mode: {mode}")
    root = Path(dataset_dir).resolve()
    print(f"[refresh] mode={mode}")
    catalog = read_catalog(root)
    registry = read_json(root / "channel_registry.json")
    if not isinstance(registry, dict):
        raise ValueError("channel_registry.json must contain an object")
    metadata = pd.read_parquet(image_root(root) / "image_metadata.parquet")
    manifest = read_manifest(root)
    images_root = image_root(root, manifest)

    if mode == "roles":
        print("[refresh] scanning camera folders")
        rows: list[dict[str, object]] = []
        for record in catalog["cycles"]:
            if not isinstance(record, Mapping):
                continue
            cycle_name = str(record["cycle_name"])
            scoped = metadata.loc[metadata["cycle_name"].astype(str).eq(cycle_name)]
            cycle_image_root = images_root / cycle_name
            files = sorted(
                path
                for role_dir in (cycle_image_root.iterdir() if cycle_image_root.is_dir() else [])
                if role_dir.is_dir()
                for path in role_dir.iterdir()
                if path.is_file()
            )
            if len(files) != len(scoped):
                raise ValueError(f"{cycle_name}: image set changed; use refresh images")
            old_roles = set(scoped["camera_role"].dropna().astype(str))
            new_roles = {path.parent.name for path in files}
            removed_roles = sorted(old_roles - new_roles)
            added_roles = sorted(new_roles - old_roles)
            if len(removed_roles) == len(added_roles):
                for old_role, new_role in zip(removed_roles, added_roles, strict=True):
                    print(f"[roles] {cycle_name}: {old_role} -> {new_role}")
            for path in files:
                candidates = scoped.loc[scoped["file_name"].astype(str).eq(path.name)]
                image_time = _image_timestamp(path)
                if image_time is not None:
                    parsed = pd.to_datetime(candidates["image_time"], errors="coerce")
                    candidates = candidates.loc[parsed.eq(image_time)]
                if candidates.empty:
                    raise ValueError(f"{cycle_name}: image set changed; use refresh images")
                row = {str(key): value for key, value in candidates.iloc[0].to_dict().items()}
                row["camera_role"] = path.parent.name
                rows.append(row)
        metadata = pd.DataFrame(rows, columns=metadata.columns)
        metadata = metadata.sort_values(
            ["cycle_name", "camera_role", "image_time", "file_name"], kind="stable"
        ).reset_index(drop=True)
        metadata["frame_index"] = (
            metadata.groupby(["cycle_name", "camera_role"], sort=False).cumcount() + 1
        )
        print("[refresh] updating image metadata")
        write_parquet(metadata, image_root(root) / "image_metadata.parquet")

    if mode in {"images", "all"}:
        print("[refresh] scanning images")
        images: list[dict[str, object]] = []
        found = 0
        for record in catalog["cycles"]:
            if not isinstance(record, Mapping):
                continue
            cycle_name = str(record["cycle_name"])
            cycle_image_root = images_root / cycle_name
            files = sorted(
                path.relative_to(cycle_image_root)
                for role_dir in (cycle_image_root.iterdir() if cycle_image_root.is_dir() else [])
                if role_dir.is_dir()
                for path in role_dir.iterdir()
                if path.is_file()
            )
            found += len(files)
            assets = record.get("assets", {})
            if not isinstance(assets, Mapping):
                raise ValueError(f"cycle assets are missing: {cycle_name}")
            frame = pd.read_parquet(root / str(assets["parquet"]), columns=["timestamp"])
            timestamps = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
            boundaries = record.get("boundaries", {})
            if not isinstance(boundaries, Mapping):
                boundaries = {}
            window = {
                "cycle_name": cycle_name,
                "start_time": boundaries.get("start_time") or timestamps.min(),
                "end_time": boundaries.get("end_time") or timestamps.max(),
                "stable_heating_start": boundaries.get("stable_heating_start"),
                "defrost_preparation_start": boundaries.get("defrost_preparation_start"),
                "defrost_start": boundaries.get("defrost_start"),
                "defrost_end": boundaries.get("defrost_end"),
            }
            images.extend(
                collect_cycle_images(
                    [cycle_image_root / path for path in files],
                    input_dir=cycle_image_root,
                    cycles=[window],
                )
            )
        metadata = image_metadata_frame(images)
        print(f"[refresh] found {found:,} images; in cycle {len(metadata):,}")
        if len(metadata) != found:
            raise ValueError(
                f"{found - len(metadata)} images fall outside their cycle; metadata unchanged"
            )
        print("[refresh] rebuilding image metadata")
        write_parquet(metadata, image_root(root) / "image_metadata.parquet")

    if mode != "figures":
        for experiment in manifest["experiments"]:
            if isinstance(experiment, dict):
                experiment.pop("camera_roles", None)
        manifest.setdefault("images_root", "images")
        write_json(manifest, root / "dataset_manifest.json")

    records = [record for record in catalog["cycles"] if isinstance(record, dict)]
    if mode == "figures":
        for index, record in enumerate(records, start=1):
            print(
                f"[refresh] rendering figures: {index}/{len(records)} {record['cycle_name']}",
                flush=True,
            )
            render_publication_asset(root, record)
            _render_rgb_panel(
                root,
                record,
                metadata,
                registry,
                _experiment_camera_roles(catalog, metadata, str(record["experiment_id"])),
            )
    else:
        for index, record in enumerate(records, start=1):
            print(
                f"[refresh] updating cycle statistics: {index}/{len(records)} "
                f"{record['cycle_name']}",
                flush=True,
            )
            _refresh_cycle_record(
                root,
                record,
                metadata,
                registry,
                _experiment_camera_roles(catalog, metadata, str(record["experiment_id"])),
            )
            legacy = record.get("assets", {}).pop("rgb_coverage", None)
            if legacy is not None:
                legacy_path = (root / str(legacy)).resolve()
                if legacy_path.is_relative_to(root) and legacy_path.is_file():
                    legacy_path.unlink()
        write_catalog(root, catalog)
    print("[refresh] validating dataset")
    validate_dataset(root)
    print("[refresh] done")
    return root
