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
    original: pd.DataFrame | None = None


def add_dataset(input_dir: Path, dataset_dir: Path | None = None) -> Path:
    """Build or append one date directly from raw input."""
    input_path = Path(input_dir).resolve()
    project_root = _resolve_project_root()
    target = Path(dataset_dir).resolve() if dataset_dir is not None else project_root / "dataset"
    _validate_date_input(input_path)
    config = _load_config_for_input(input_path, project_root)
    print(f"[add] input={input_path.name}")

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
            print(f"[add] experiment already exists: {experiment_id}")
            return target
        if experiments:
            last_date = max(str(item["experiment_date"])[:10] for item in experiments)
            if experiment_date <= last_date:
                raise ValueError(
                    "historical or same-date append is not supported; remove that date first"
                )

        print("[add] preparing and processing")
        build = _build_date(input_path, config)
        _append_build(target, build)
        print("[add] done")
        return target

    print("[add] preparing and processing")
    build = _build_date(input_path, config)
    _materialize_builds(target, [build])
    print("[add] done")
    return target


def replace_dataset(input_dir: Path, dataset_dir: Path | None = None) -> Path:  # noqa: C901
    """Rebuild one published experiment and renumber everything after it."""
    import shutil
    import tempfile

    from .dataset_images import collect_cycle_images, copy_image, image_metadata_frame
    from .dataset_io import read_json, write_csv, write_json, write_parquet
    from .dataset_metadata import read_catalog
    from .dataset_schema import (
        align_original_schema,
        build_processed_frame,
        build_registry,
        merge_original_columns,
        merge_registries,
    )
    from .dataset_validation import validate_dataset
    from .io import discover_inputs

    input_path = Path(input_dir).resolve()
    project_root = _resolve_project_root()
    root = Path(dataset_dir).resolve() if dataset_dir is not None else project_root / "dataset"
    _validate_date_input(input_path)
    config = _load_config_for_input(input_path, project_root)
    experiment_id = str(config.experiment_id)
    catalog = read_catalog(root)
    records = [record for record in catalog["cycles"] if isinstance(record, dict)]
    replaced = [record for record in records if str(record["experiment_id"]) == experiment_id]
    if not replaced:
        raise ValueError(f"experiment is not published: {experiment_id}")

    print(f"[replace] rebuilding {experiment_id}")
    build = _build_date(input_path, config)
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
    for key in ("image_coverage", "baseline_seconds", "baseline_managed", "recovery_edit"):
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
    old_metadata = pd.read_parquet(root / "image_metadata.parquet")
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

    for name in ("cycle_catalog.json", "image_metadata.parquet", "channel_registry.json"):
        shutil.copy2(root / name, backup / name)

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
        from .dataset_metadata import cycle_assets

        for record in later:
            old_name = str(record["cycle_name"])
            new_name = later_names[old_name]
            shifted_record = {**record, "cycle_name": new_name, "assets": cycle_assets(new_name)}
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
            reusable = reusable_images.get(
                (str(image["camera_role"]), str(image["file_name"]))
            )
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
            )
            reviewed = old_by_uid.get(str(record["cycle_uid"]))
            same_boundaries = reviewed is not None and all(
                reviewed.get("boundaries", {}).get(name)
                == record.get("boundaries", {}).get(name)
                for name in ("heating_start", "defrost_start", "defrost_end")
            )
            if same_boundaries and reviewed is not None and (
                reviewed.get("status"), reviewed.get("status_reason")
            ) != (
                reviewed.get("pipeline_status"),
                reviewed.get("pipeline_status_reason"),
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
        write_parquet(final_metadata, root / "image_metadata.parquet")
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
        for name in ("cycle_catalog.json", "image_metadata.parquet", "channel_registry.json"):
            shutil.copy2(backup / name, root / name)
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

    from .dataset_io import write_json, write_parquet
    from .dataset_metadata import image_root, read_catalog, read_manifest, write_catalog
    from .dataset_validation import validate_dataset

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
    metadata = pd.read_parquet(root / "image_metadata.parquet")
    metadata = metadata.loc[~metadata["cycle_name"].astype(str).isin(removed_names)].copy()
    print("[remove] updating image_metadata.parquet")
    write_parquet(metadata, root / "image_metadata.parquet")
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


def aggregate_original(dataset_dir: Path, *, seconds: int = 10) -> Path:
    """Rebuild Processed cycles from the published Original sensor rows."""
    from .channels import load_channels
    from .config import load_config
    from .dataset_edit import apply_baseline
    from .dataset_io import read_json, write_csv, write_json, write_parquet
    from .dataset_schema import build_processed_frame, merge_registries, registry_from_frame
    from .process import process

    if seconds <= 0:
        raise ValueError("aggregate seconds must be positive")
    root = Path(dataset_dir).resolve()
    project_root = _resolve_project_root()
    registry = read_json(root / "channel_registry.json")
    config_path = project_root / "configs/config.yaml"
    configured = load_channels(config_path)
    saved = registry.get("channels", {})
    channels = {
        name: {
            **settings,
            **{key: value for key, value in saved.get(name, {}).items() if value is not None},
        }
        for name, settings in configured.items()
    }
    records = read_json(root / "cycle_catalog.json")["cycles"]
    folder = "cycles" if seconds == registry["resample_interval_seconds"] else f"cycles_{seconds}s"
    target = root / folder
    output_registry: dict[str, Any] | None = None
    configs: dict[str, Any] = {}

    for record in records:
        date = str(record["experiment_date"])
        if date not in configs:
            config = load_config(
                config_path,
                experiment_date=date,
                input_dir=root / "cycles_original",
            )
            configs[date] = replace(
                config,
                process=replace(config.process, resample_interval_seconds=seconds),
            )
        original = pd.read_csv(
            root / str(record["assets"]["original_csv"]), parse_dates=["timestamp"]
        )
        prepared = _prepared_from_original(original, channels)
        processed, _ = process(
            prepared,
            pd.DataFrame([_summary_from_original(record, prepared)]),
            configs[date],
            channels,
        )
        candidate = registry_from_frame(processed, channels, resample_interval_seconds=seconds)
        if output_registry is None:
            output_registry = (
                {**registry, **merge_registries(registry, candidate)}
                if target.name == "cycles"
                else candidate
            )
        canonical = build_processed_frame(
            processed,
            output_registry,
            cycle_name=str(record["cycle_name"]),
            cycle_uid=str(record["cycle_uid"]),
        )
        if bool(registry.get("baseline_managed", False)):
            canonical = apply_baseline(
                canonical,
                dict(record),
                dict(output_registry),
                seconds=int(registry["baseline_seconds"]),
            )
        write_csv(canonical, target / f"{record['cycle_name']}.csv")
        write_parquet(canonical, target / f"{record['cycle_name']}.parquet")

    if output_registry is None:
        raise ValueError("Dataset has no cycles")
    if target.name == "cycles":
        write_json(output_registry, root / "channel_registry.json")
    return target


def _prepared_from_original(
    original: pd.DataFrame, channels: Mapping[str, Mapping[str, Any]]
) -> pd.DataFrame:
    from .prepare import _combine_channel

    identity = [
        "experiment_id",
        "experiment_date",
        "timestamp",
        "cycle_id",
        "cycle_stage",
        "cycle_status",
        "cycle_status_reason",
    ]
    prepared = original.sort_values("timestamp", kind="stable").drop_duplicates("timestamp")
    prepared = prepared.loc[:, identity].reset_index(drop=True)
    timestamps = prepared["timestamp"]
    channel_columns: dict[str, Any] = {}
    for name, settings in channels.items():
        if str(settings.get("kind")) == "derived":
            continue
        if name in original:
            values = original.drop_duplicates("timestamp").set_index("timestamp")[name]
            channel_columns[name] = values.reindex(timestamps).reset_index(drop=True)
            channel_columns[f"{name}__missing"] = channel_columns[name].isna()
            for suffix in ("__invalid", "__duplicate", "__conflict"):
                channel_columns[f"{name}{suffix}"] = False
            continue
        frames = [
            original[["timestamp", source]].rename(columns={source: "raw"})
            for source in settings.get("source_names", [])
            if source in original
        ]
        values = _combine_channel(frames, settings, timestamps)
        channel_columns[name] = values["value"]
        for suffix in ("__missing", "__invalid", "__duplicate", "__conflict"):
            channel_columns[f"{name}{suffix}"] = values[suffix]
    return pd.concat([prepared, pd.DataFrame(channel_columns)], axis=1)


def _summary_from_original(
    record: Mapping[str, Any], prepared: pd.DataFrame
) -> dict[str, Any]:
    boundaries = record.get("boundaries", {})
    partial = str(record["cycle_id"]).startswith("partial_")
    return {
        "experiment_id": record["experiment_id"],
        "experiment_date": record["experiment_date"],
        "cycle_id": record["cycle_id"],
        "cycle_status": prepared["cycle_status"].iloc[0],
        "cycle_status_reason": prepared["cycle_status_reason"].iloc[0],
        **{
            name: None if partial else boundaries.get(name)
            for name in (
                "heating_start",
                "stable_heating_start",
                "defrost_preparation_start",
                "defrost_start",
                "defrost_end",
            )
        },
    }


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


def _load_config_for_input(input_path: Path, project_root: Path) -> Any:
    from .config import load_config

    # EDF 可能分段或补录；实验日期只认 XLS 参数文件名。
    matches = [re.search(r"\d{4}-\d{2}-\d{2}", path.name) for path in input_path.glob("*.xls")]
    dates = {match.group() for match in matches if match}
    if not matches or len(dates) != 1 or any(match is None for match in matches):
        raise ValueError("XLS filenames must contain one shared date")
    return load_config(
        project_root / "configs/config.yaml",
        experiment_date=dates.pop(),
        input_dir=input_path,
    )


def _build_date(input_path: Path, config: Any) -> _DateBuild:
    from .channels import load_channels
    from .prepare import prepare, prepare_original
    from .process import process
    from .validation import validate_prepared, validate_processed

    channels = load_channels(config.channels_path)
    print("[add] prepare sensors", flush=True)
    prepared, initial_summary = prepare(config, channels)
    print("[add] validate prepared", flush=True)
    validate_prepared(prepared, initial_summary)
    print("[add] process cycles", flush=True)
    processed, final_summary = process(prepared, initial_summary, config, channels)
    print("[add] validate processed", flush=True)
    validate_processed(processed, final_summary)
    print("[add] preserve original sensors", flush=True)
    original = prepare_original(config, prepared)
    print(f"[add] cycles={final_summary['cycle_id'].nunique()}", flush=True)
    return _DateBuild(
        input_dir=input_path.resolve(),
        config=config,
        channels=channels,
        prepared=prepared,
        summary=final_summary,
        processed=processed,
        original=original,
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
        _sensor_coverage_intervals,
        rgb_overall_intervals,
        rgb_stage_metrics,
        scan_cycle_images,
    )
    from .dataset_io import write_csv, write_parquet
    from .dataset_metadata import build_cycle_record
    from .dataset_schema import build_processed_frame, export_original_frame
    from .visualization import (
        render_cycle_publication,
        render_rgb_panel,
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
    builds: Sequence[_DateBuild],
) -> None:
    from .dataset_images import collect_cycle_images, copy_image, image_metadata_frame
    from .dataset_io import write_json, write_parquet
    from .dataset_metadata import experiment_record
    from .dataset_schema import build_registry, merge_original_columns
    from .io import discover_inputs

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
    builds_by_experiment = {
        str(build.config.experiment_id): build for build in builds
    }
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
        )
        records.append(record)

    write_parquet(image_metadata, dataset_dir / "image_metadata.parquet")

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


def _image_cycle_windows(
    build: _DateBuild,
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
    build: _DateBuild,
) -> None:
    from .dataset_images import collect_cycle_images, copy_image, image_metadata_frame
    from .dataset_io import read_json, write_csv, write_json, write_parquet
    from .dataset_metadata import experiment_record, read_catalog, read_manifest
    from .dataset_schema import (
        align_original_schema,
        build_processed_frame,
        build_registry,
        merge_original_columns,
        merge_registries,
    )
    from .io import discover_inputs

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
        "baseline_seconds",
        "baseline_managed",
        "recovery_edit",
    ):
        if setting_name in old_registry:
            merged_registry[setting_name] = old_registry[setting_name]
        elif setting_name in candidate:
            merged_registry[setting_name] = candidate[setting_name]

    old_images = pd.read_parquet(dataset_dir / "image_metadata.parquet")
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
            ),
        },
    ]
    manifest["experiments"].sort(key=lambda value: str(value["experiment_date"]))
    manifest.setdefault("images_root", "images")
    write_json(manifest, dataset_dir / "dataset_manifest.json")
    write_json(catalog, dataset_dir / "cycle_catalog.json")


def render_publication_asset(
    dataset_dir: Path,
    record: Mapping[str, Any],
    *,
    cost_curve: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> None:
    """Render one Dataset publication, optionally with an analysis cost curve."""
    from .dataset_images import (
        _sensor_coverage_intervals,
        image_coverage_intervals,
        rgb_overall_intervals,
    )
    from .dataset_io import read_json
    from .dataset_metadata import read_catalog
    from .visualization import render_cycle_publication

    cycle_name = str(record["cycle_name"])
    assets = record.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError(f"cycle assets are missing: {cycle_name}")
    frame = pd.read_parquet(dataset_dir / str(assets["parquet"]))
    registry = read_json(dataset_dir / "channel_registry.json")
    if not isinstance(registry, dict):
        raise ValueError("channel_registry.json must contain an object")
    metadata = pd.read_parquet(dataset_dir / "image_metadata.parquet")
    catalog = read_catalog(dataset_dir)
    roles = _experiment_camera_roles(catalog, metadata, str(record["experiment_id"]))
    scoped = metadata.loc[metadata["cycle_name"].astype(str).eq(cycle_name)]
    intervals = image_coverage_intervals(frame, scoped, registry)
    render_cycle_publication(
        frame,
        record,
        output_path or dataset_dir / str(assets["publication"]),
        sensor_intervals=_sensor_coverage_intervals(frame, registry),
        rgb_intervals=rgb_overall_intervals(frame, intervals, roles),
        cost_curve=cost_curve,
    )


def _render_rgb_panel(
    dataset_dir: Path,
    record: Mapping[str, Any],
    metadata: pd.DataFrame,
    registry: Mapping[str, Any],
    camera_roles: tuple[str, ...],
    frame: pd.DataFrame | None = None,
    images: pd.DataFrame | None = None,
    intervals: (
        Mapping[str, Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]]] | None
    ) = None,
) -> None:
    from .dataset_images import scan_cycle_images
    from .visualization import render_rgb_panel

    cycle_name = str(record["cycle_name"])
    assets = record.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError(f"cycle assets are missing: {cycle_name}")
    if frame is None:
        frame = pd.read_parquet(dataset_dir / str(assets["parquet"]))
    if images is None:
        images = scan_cycle_images(dataset_dir, cycle_name, metadata)
    if intervals is None:
        from .dataset_images import image_coverage_intervals

        intervals = image_coverage_intervals(frame, images, registry)
    render_rgb_panel(
        record,
        frame,
        images,
        intervals,
        camera_roles,
        dataset_dir
        / str(assets["rgb_panel"]),
    )


def _refresh_cycle_record(
    dataset_dir: Path,
    record: dict[str, Any],
    metadata: pd.DataFrame,
    registry: Mapping[str, Any],
    camera_roles: tuple[str, ...],
) -> None:
    from .dataset_images import (
        _cycle_image_summary,
        _sensor_coverage_intervals,
        rgb_overall_intervals,
        rgb_stage_metrics,
        scan_cycle_images,
    )
    from .visualization import (
        render_cycle_publication,
        render_rgb_panel,
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
    allowed = {"valid", "invalid"}
    if status not in allowed:
        raise ValueError(f"invalid Dataset status: {status}")
    rgb_allowed = {"valid", "invalid", "not_applicable"}
    for stage, value in (("frost", rgb_frost), ("defrost", rgb_defrost)):
        if value is not None and value not in rgb_allowed:
            raise ValueError(f"invalid RGB {stage} status: {value}")
    from .dataset_metadata import read_catalog, write_catalog

    catalog = read_catalog(dataset_dir)
    for record in catalog["cycles"]:
        if isinstance(record, dict) and record.get("cycle_name") == cycle_name:
            record["status"] = status
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
) -> Path:
    """Apply baseline or recovery edits directly to the Dataset."""
    if recovery_seconds is not None and recovery_end_by is not None:
        raise ValueError("--recovery-seconds and --recovery-end-by are mutually exclusive")
    if (
        baseline_seconds is None
        and recovery_seconds is None
        and recovery_end_by is None
        and not defrost_preparation
    ):
        raise ValueError("dataset edit requires at least one edit")

    from .dataset_edit import apply_baseline, apply_defrost_preparation, apply_recovery
    from .dataset_images import (
        _sensor_coverage_intervals,
        image_coverage_intervals,
        rgb_overall_intervals,
        rgb_stage_metrics,
        scan_cycle_images,
    )
    from .dataset_io import read_json, write_csv, write_json, write_parquet
    from .dataset_metadata import read_catalog, write_catalog
    from .visualization import render_cycle_publication

    catalog = read_catalog(dataset_dir)
    registry = read_json(dataset_dir / "channel_registry.json")
    if not isinstance(registry, dict):
        raise ValueError("channel_registry.json must contain an object")
    stage_edit = recovery_seconds is not None or recovery_end_by is not None or defrost_preparation
    metadata = pd.read_parquet(dataset_dir / "image_metadata.parquet") if stage_edit else None

    records = catalog["cycles"]
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
                mode="seconds" if recovery_seconds is not None else "ts-minus",
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
        if baseline_seconds is not None:
            processed = apply_baseline(processed, record, registry, seconds=baseline_seconds)
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

    if metadata is not None and stage_edit:
        write_parquet(metadata, dataset_dir / "image_metadata.parquet")
    write_json(registry, dataset_dir / "channel_registry.json")
    write_catalog(dataset_dir, catalog)
    return dataset_dir


def refresh_dataset(dataset_dir: Path, mode: str) -> Path:  # noqa: C901
    """Refresh only the Dataset layer named by the user's physical edit."""
    from .dataset_images import collect_cycle_images, image_metadata_frame
    from .dataset_io import read_json, write_json, write_parquet
    from .dataset_metadata import image_root, read_catalog, read_manifest, write_catalog
    from .dataset_validation import validate_dataset
    from .images import _image_timestamp

    if mode not in {"roles", "images", "figures", "all"}:
        raise ValueError(f"invalid refresh mode: {mode}")
    root = Path(dataset_dir).resolve()
    print(f"[refresh] mode={mode}")
    catalog = read_catalog(root)
    registry = read_json(root / "channel_registry.json")
    if not isinstance(registry, dict):
        raise ValueError("channel_registry.json must contain an object")
    metadata = pd.read_parquet(root / "image_metadata.parquet")
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
        metadata["frame_index"] = metadata.groupby(
            ["cycle_name", "camera_role"], sort=False
        ).cumcount() + 1
        print("[refresh] updating image metadata")
        write_parquet(metadata, root / "image_metadata.parquet")

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
        write_parquet(metadata, root / "image_metadata.parquet")

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


def render_dataset(
    dataset_dir: Path,
    cycle_name: str,
    *,
    publication: bool = True,
    panel: bool = True,
    fetch_cloud_images: bool = False,
) -> Path:
    """Render selected final assets without reading any source directory."""
    from .dataset_io import read_json
    from .dataset_metadata import read_catalog

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
        render_publication_asset(dataset_dir, record)
    if panel:
        from .dataset_images import materialize_cycle_images, scan_cycle_images

        registry = read_json(dataset_dir / "channel_registry.json")
        if not isinstance(registry, dict):
            raise ValueError("channel_registry.json must contain an object")
        metadata = pd.read_parquet(dataset_dir / "image_metadata.parquet")
        roles = _experiment_camera_roles(catalog, metadata, str(record["experiment_id"]))
        with materialize_cycle_images(
            dataset_dir, cycle_name, fetch_cloud=fetch_cloud_images
        ) as cycle_dir:
            images = scan_cycle_images(
                dataset_dir, cycle_name, metadata, cycle_dir=cycle_dir
            )
            _render_rgb_panel(
                dataset_dir, record, metadata, registry, roles, images=images
            )
    return dataset_dir


def _experiment_camera_roles(
    catalog: Mapping[str, Any], image_metadata: pd.DataFrame, experiment_id: str
) -> tuple[str, ...]:
    cycle_names = {
        str(record["cycle_name"])
        for record in catalog["cycles"]
        if isinstance(record, Mapping) and str(record.get("experiment_id")) == experiment_id
    }
    scoped = image_metadata.loc[image_metadata["cycle_name"].astype(str).isin(cycle_names)]
    return tuple(sorted(scoped["camera_role"].dropna().astype(str).unique()))
