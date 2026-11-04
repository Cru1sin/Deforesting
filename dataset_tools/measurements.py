"""Rebuild and update the published cycle measurement tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from dataset_tools.dataset_maintenance import _resolve_project_root


def update_cycle_columns(dataset_dir: Path, updates: Mapping[str, pd.DataFrame]) -> None:
    """Add or replace timestamp-aligned columns in existing Processed cycles."""
    from .cycle_metadata import read_catalog
    from .dataset_paths import write_csv, write_parquet

    assets = {r["cycle_name"]: r["assets"] for r in read_catalog(dataset_dir)["cycles"]}
    for cycle_name, update in updates.items():
        parquet_path = dataset_dir / assets[cycle_name]["parquet"]
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


def add_missing_channel(dataset_dir: Path, name: str, *, cycles=None) -> Path:
    """Add a channel or campaign annotation without rebuilding any cycle."""
    from .builder.channel_mapping import load_channels
    from .builder.schema import merge_registries, registry_from_frame
    from .dataset_paths import read_json, write_csv, write_json, write_parquet

    root = Path(dataset_dir).resolve()
    settings = load_channels()[name]
    registry = read_json(root / "channel_registry.json")
    candidate = registry_from_frame(
        pd.DataFrame(columns=[name]),
        {name: settings},
        resample_interval_seconds=registry["resample_interval_seconds"],
    )
    merged = {**registry, **merge_registries(registry, candidate)}
    catalog = read_json(root / "cycle_catalog.json")
    records = catalog["cycles"]
    targets = records if cycles is None else [r for r in records if r["cycle_name"] in cycles]
    catalog_changed = False
    for index, record in enumerate(targets, 1):
        value = None
        if name == "user_side_fan_setting":
            from .cycle_metadata import user_side_fan_setting

            value = user_side_fan_setting(record["experiment_date"])
        original_path = root / record["assets"]["original_csv"]
        if value is not None or name not in pd.read_csv(original_path, nrows=0).columns:
            original = pd.read_csv(original_path, dtype=str, keep_default_na=False)
            if name == "user_side_fan_setting":
                value = user_side_fan_setting(
                    record["experiment_date"], original.get("water_temperature_setpoint")
                )
            if name not in original or not original[name].eq(value).all():
                original[name] = "" if value is None else value
                write_csv(original, original_path)
        frame = pd.read_parquet(root / record["assets"]["parquet"])
        if name == "user_side_fan_setting":
            value = user_side_fan_setting(
                record["experiment_date"], frame.get("water_temperature_setpoint")
            )
        columns = list(
            dict.fromkeys(
                column
                for column in ["cycle_name", "cycle_uid", *merged["columns"], *frame.columns]
                if column in frame or column == name
            )
        )
        annotation_changed = value is not None and (
            name not in frame or not frame[name].eq(value).fillna(False).all()
        )
        if frame.columns.tolist() != columns or annotation_changed:
            frame = frame.reindex(columns=columns)
            if value is not None:
                frame[name] = value
            write_parquet(frame, root / record["assets"]["parquet"])
            write_csv(frame, root / record["assets"]["csv"])
        if name == "user_side_fan_setting":
            from .cycle_metadata import cycle_operating_condition

            condition = cycle_operating_condition(frame)
            if record.get("operating_condition") != condition:
                record["operating_condition"] = condition
                catalog_changed = True
        if index % 25 == 0 or index == len(targets):
            print(f"[add-missing-channel] {name}: {index}/{len(targets)}", flush=True)
    if catalog_changed:
        write_json(catalog, root / "cycle_catalog.json")
    if merged != registry:
        write_json(merged, root / "channel_registry.json")
    return root


def refresh_processed_measurements(dataset_dir: Path, *, n_jobs: int = 6) -> Path:
    """Publish offline interpolation and trailing means; preserve Original assets."""
    from joblib import Parallel, delayed, parallel_config

    from heat_smoothing.methods import MEASUREMENT_PROCESSING, process_measurements

    from .builder.channel_mapping import load_channels
    from .builder.detect_cycles import recovery_stages
    from .builder.schema import registry_from_frame
    from .cycle_metadata import read_catalog
    from .dataset_paths import read_json, write_csv, write_json, write_parquet

    root = Path(dataset_dir).resolve()
    records = read_catalog(root)["cycles"]

    def update(record):
        assets = record["assets"]
        original = pd.read_csv(root / assets["original_csv"], parse_dates=["timestamp"])
        original["cycle_stage"] = recovery_stages(original, record["boundaries"])
        frame = pd.read_parquet(root / assets["parquet"])
        frame["timestamp"] = pd.to_datetime(frame.timestamp)
        result = process_measurements(original, frame)
        write_parquet(result, root / assets["parquet"])
        write_csv(result, root / assets["csv"])
        print(
            f"[offline] {record['cycle_name']}: {len(result)} rows, {int(result.is_interpolated.sum())} interpolated",
            flush=True,
        )
        return result.columns.tolist()

    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        columns = Parallel()(delayed(update)(record) for record in records)
    registry = read_json(root / "channel_registry.json")
    definitions = load_channels()
    registry["columns"] = list(
        dict.fromkeys(
            name for group in columns for name in group if name not in {"cycle_name", "cycle_uid"}
        )
    )
    candidate = registry_from_frame(
        pd.DataFrame(columns=registry["columns"]),
        definitions,
        resample_interval_seconds=registry["resample_interval_seconds"],
    )
    registry["channels"] = {
        name: candidate["channels"].get(name, registry["channels"].get(name, {}))
        for name in registry["columns"]
        if name in candidate["channels"] or name in registry["channels"]
    }
    registry.pop("heating_branches", None)
    registry["measurement_processing"] = MEASUREMENT_PROCESSING
    write_json(registry, root / "channel_registry.json")
    return root


def aggregate_original(
    dataset_dir: Path, *, seconds: int = 1, publish: bool = False, n_jobs: int = 6
) -> Path:
    """Rebuild Processed cycles from the published Original sensor rows."""
    from .builder.build_cycle_tables import process
    from .builder.channel_mapping import load_channels
    from .builder.dataset_settings import load_config
    from .builder.schema import build_processed_frame, merge_registries, registry_from_frame
    from .dataset_paths import read_json, write_csv, write_json, write_parquet

    if seconds <= 0:
        raise ValueError("aggregate seconds must be positive")
    root = Path(dataset_dir).resolve()
    project_root = _resolve_project_root()
    registry = read_json(root / "channel_registry.json")
    configured = load_channels()
    saved = registry.get("channels", {})
    channels = {
        name: {
            **settings,
            **{key: value for key, value in saved.get(name, {}).items() if value is not None},
        }
        for name, settings in configured.items()
    }
    records = read_json(root / "cycle_catalog.json")["cycles"]
    definition_changed = publish and seconds != registry["resample_interval_seconds"]
    folder = (
        "cycles"
        if publish or seconds == registry["resample_interval_seconds"]
        else f"cycles_{seconds}s"
    )
    target = root / folder
    print(
        f"[aggregate] dataset={root}\n[aggregate] target={target}\n[aggregate] replace formal measurements={folder == 'cycles'}",
        flush=True,
    )
    registry = {**registry, "resample_interval_seconds": seconds}
    from joblib import Parallel, delayed, parallel_config

    from .edit_cycle_metadata import apply_baseline

    def rebuild(record):
        config = load_config(
            project_root=project_root,
            experiment_date=str(record["experiment_date"]),
            input_dir=root / "cycles_original",
        )
        config = replace(
            config,
            process=replace(
                config.process,
                resample_interval_seconds=seconds,
            ),
        )
        original = pd.read_csv(
            root / str(record["assets"]["original_csv"]), parse_dates=["timestamp"]
        )
        prepared = _prepared_from_original(original, channels)
        # Catalog owns the build status; historical Original may say "incomplete".
        status = str(record.get("pipeline_status", prepared.cycle_status.iloc[0]))
        prepared["cycle_status"] = status if status in {"valid", "invalid"} else "invalid"
        prepared["cycle_status_reason"] = record.get("pipeline_status_reason", "")
        processed, _ = process(
            prepared, pd.DataFrame([_summary_from_original(record, prepared)]), config, channels
        )
        if registry.get("baseline_managed"):
            processed = apply_baseline(
                processed, record, dict(registry), seconds=registry["baseline_seconds"]
            )
        candidate = registry_from_frame(processed, channels, resample_interval_seconds=seconds)
        output_registry = (
            {**registry, **merge_registries(registry, candidate)}
            if folder == "cycles"
            else candidate
        )
        output_registry["resample_interval_seconds"] = seconds
        canonical = build_processed_frame(
            processed,
            output_registry,
            cycle_name=str(record["cycle_name"]),
            cycle_uid=str(record["cycle_uid"]),
        )
        directory = root / Path(record["assets"]["csv"]).parent if folder == "cycles" else target
        write_csv(canonical, directory / f"{record['cycle_name']}.csv")
        write_parquet(canonical, directory / f"{record['cycle_name']}.parquet")
        record.setdefault("data", {})["processed_row_count"] = len(canonical)
        if definition_changed:
            for field in (
                "pareto_knee_status",
                "rgb_knee_coverage_status",
                "pareto_extrapolated_knee_status",
                "rgb_extrapolated_knee_coverage_status",
            ):
                if field in record:
                    record[field] = "invalid"
            for field in (
                "effective_cop",
                "effective_cop_without_trunc",
                "cop_peak_quality",
                "pre_cop_status",
            ):
                record.pop(field, None)
        print(f"[aggregate] {record['cycle_name']}: {len(canonical)} rows, {seconds} s", flush=True)
        return record, candidate

    if not records:
        raise ValueError("Dataset has no cycles")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs)(delayed(rebuild)(record) for record in records)
    if folder == "cycles":
        output_registry = dict(registry)
        for _, candidate in results:
            output_registry = {**output_registry, **merge_registries(output_registry, candidate)}
        output_registry["resample_interval_seconds"] = seconds
        catalog = read_json(root / "cycle_catalog.json")
        catalog["cycles"] = [record for record, _ in results]
        write_json(catalog, root / "cycle_catalog.json")
        write_json(output_registry, root / "channel_registry.json")
    return target


def _prepared_from_original(
    original: pd.DataFrame, channels: Mapping[str, Mapping[str, Any]]
) -> pd.DataFrame:
    from .builder.prepare_measurements import _combine_channel

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


def _summary_from_original(record: Mapping[str, Any], prepared: pd.DataFrame) -> dict[str, Any]:
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
