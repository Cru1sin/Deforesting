"""Small, Dataset-local scientific and camera transformations."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .cycles import _stable_start
from .dataset_io import write_atomic_csv, write_atomic_parquet


def apply_recovery_edit(  # noqa: C901
    dataset_dir: Path,
    catalog: dict[str, Any],
    *,
    mode: str,
    recovery_seconds: int | None = None,
    registry: Mapping[str, Any] | None = None,
    cycle_names: Collection[str] | None = None,
) -> set[str]:
    """Update recovery facts and all Dataset-local stage-dependent fields."""
    if mode not in {"seconds", "ts-minus"}:
        raise ValueError(f"unsupported recovery edit mode: {mode}")
    if mode == "seconds" and (recovery_seconds is None or recovery_seconds < 0):
        raise ValueError("recovery_seconds must be a nonnegative integer")
    selected = set(cycle_names) if cycle_names is not None else None
    changed: set[str] = set()
    for record in _cycle_records(catalog):
        cycle_name = str(record["cycle_name"])
        if selected is not None and cycle_name not in selected:
            continue
        assets = _assets(record)
        original_path = dataset_dir / str(assets["original_csv"])
        processed_path = dataset_dir / str(assets["parquet"])
        original = pd.read_csv(original_path)
        original["timestamp"] = pd.to_datetime(original["timestamp"], errors="raise")
        processed = pd.read_parquet(processed_path)
        processed["timestamp"] = pd.to_datetime(processed["timestamp"], errors="raise")
        boundaries = record.setdefault("boundaries", {})
        heating_start = _timestamp(boundaries.get("heating_start"))
        defrost_start = _timestamp(boundaries.get("defrost_start"))
        defrost_end = _timestamp(boundaries.get("defrost_end"))
        if heating_start is None and not original.empty:
            heating_start = pd.Timestamp(original["timestamp"].min())
        stable: pd.Timestamp | None
        if mode == "seconds":
            candidate = (
                heating_start + pd.Timedelta(seconds=int(recovery_seconds or 0))
                if heating_start is not None
                else None
            )
            stable = _within_data(candidate, original)
        else:
            recovery_settings = registry.get("recovery_edit", {}) if registry else {}
            fallback_seconds = (
                recovery_settings.get("fallback_seconds", 180)
                if isinstance(recovery_settings, Mapping)
                else 180
            )
            stable = _stable_start(
                original,
                heating_start,
                defrost_start,
                {"stable_heating_seconds": fallback_seconds},
            )
        boundaries["heating_start"] = _iso(heating_start)
        boundaries["stable_heating_start"] = _iso(stable)
        _rewrite_stage_column(original, stable, defrost_start, defrost_end)
        _rewrite_stage_column(processed, stable, defrost_start, defrost_end)
        processed = _recompute_stage_dependent_fields(
            processed,
            record,
            registry or {},
        )
        write_atomic_csv(original, original_path)
        write_atomic_parquet(processed, processed_path)
        csv_asset = assets.get("csv")
        if isinstance(csv_asset, str):
            write_atomic_csv(processed, dataset_dir / csv_asset)
        changed.add(cycle_name)
    metadata_path = dataset_dir / "image_metadata.parquet"
    if changed and metadata_path.is_file():
        metadata = pd.read_parquet(metadata_path)
        for record in _cycle_records(catalog):
            cycle_name = str(record["cycle_name"])
            if cycle_name not in changed:
                continue
            assets = _assets(record)
            processed = pd.read_parquet(dataset_dir / str(assets["parquet"]))
            cycle_mask = metadata["cycle_name"].astype(str).eq(cycle_name)
            if not cycle_mask.any():
                continue
            timestamps = pd.to_datetime(
                metadata.loc[cycle_mask, "matched_timestamp"], errors="coerce"
            )
            image_times = pd.to_datetime(
                metadata.loc[cycle_mask, "image_time"], errors="coerce"
            )
            timestamps = timestamps.fillna(image_times)
            lookup = (
                processed.loc[:, ["timestamp", "cycle_stage"]]
                .assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"]))
                .drop_duplicates("timestamp")
                .set_index("timestamp")["cycle_stage"]
            )
            metadata.loc[cycle_mask, "cycle_stage"] = timestamps.map(lookup).fillna(
                metadata.loc[cycle_mask, "cycle_stage"]
            )
        write_atomic_parquet(metadata, metadata_path)
    if isinstance(registry, dict):
        previous = registry.get("recovery_edit", {})
        registry["recovery_edit"] = {
            "mode": mode,
            "seconds": (
                int(recovery_seconds)
                if mode == "seconds" and recovery_seconds is not None
                else None
            ),
            "fallback_seconds": (
                previous.get("fallback_seconds", 180)
                if isinstance(previous, Mapping)
                else 180
            ),
            "managed": True,
        }
    return changed


def apply_baseline_edit(  # noqa: C901
    dataset_dir: Path,
    catalog: dict[str, Any],
    *,
    baseline_seconds: int,
    registry: Mapping[str, Any],
    cycle_names: Collection[str] | None = None,
) -> set[str]:
    """Apply the fixed stable-start baseline window without searching."""
    if baseline_seconds < 0:
        raise ValueError("baseline_seconds must be nonnegative")
    channels = registry.get("channels", {})
    if not isinstance(channels, Mapping):
        raise ValueError("channel registry channels must be a mapping")
    eligible = [
        str(name)
        for name, settings in channels.items()
        if isinstance(settings, Mapping)
        and str(settings.get("kind")) in {"continuous", "step", "derived"}
        and (bool(settings.get("analysis_candidate")) or settings.get("role") == "performance")
    ]
    selected = set(cycle_names) if cycle_names is not None else None
    changed: set[str] = set()
    for record in _cycle_records(catalog):
        cycle_name = str(record["cycle_name"])
        if selected is not None and cycle_name not in selected:
            continue
        assets = _assets(record)
        processed_path = dataset_dir / str(assets["parquet"])
        frame = pd.read_parquet(processed_path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        boundaries = record.setdefault("boundaries", {})
        stable = _timestamp(boundaries.get("stable_heating_start"))
        end = stable + pd.Timedelta(seconds=baseline_seconds) if stable is not None else None
        enough = end is not None and not frame.empty and frame["timestamp"].max() >= end
        for name in eligible:
            baseline_column = f"{name}__baseline"
            residual_column = f"{name}__baseline_residual"
            frame[baseline_column] = pd.NA
            frame[residual_column] = pd.NA
        if enough and stable is not None and end is not None:
            window = frame.loc[
                frame["timestamp"].ge(stable)
                & frame["timestamp"].lt(end)
                & frame.get("cycle_stage", pd.Series("", index=frame.index)).eq(
                    "frost_development"
                )
            ]
            for name in eligible:
                if name not in window:
                    continue
                values = pd.to_numeric(window[name], errors="coerce")
                imputed = window.get(f"{name}__imputed")
                observed = values.notna()
                if imputed is not None:
                    observed &= ~imputed.fillna(False).astype(bool)
                finite = values.loc[observed]
                if finite.empty:
                    continue
                baseline = float(finite.median())
                frame.loc[:, f"{name}__baseline"] = baseline
                frame.loc[:, f"{name}__baseline_residual"] = (
                    pd.to_numeric(frame[name], errors="coerce") - baseline
                )
            boundaries["baseline_start"] = _iso(stable)
            boundaries["baseline_end"] = _iso(end)
        else:
            boundaries["baseline_start"] = None
            boundaries["baseline_end"] = None
        write_atomic_parquet(frame, processed_path)
        csv_asset = assets.get("csv")
        if isinstance(csv_asset, str):
            write_atomic_csv(frame, dataset_dir / csv_asset)
        changed.add(cycle_name)
    registry["baseline_seconds"] = int(baseline_seconds)  # type: ignore[index]
    registry["baseline_managed"] = True  # type: ignore[index]
    return changed


def rename_camera_role(dataset_dir: Path, renames: Mapping[str, str]) -> set[str]:
    """Rename only the mutable role suffix of source/role image directories."""
    changed: set[str] = set()
    for cycle_root in sorted((dataset_dir / "images").iterdir()):
        if not cycle_root.is_dir():
            continue
        existing = {path.name for path in cycle_root.iterdir() if path.is_dir()}
        for old, new in renames.items():
            if not old or not new or "__" in old or "__" in new:
                raise ValueError("camera roles must be non-empty and cannot contain __")
            for role_dir in sorted(path for path in cycle_root.iterdir() if path.is_dir()):
                parts = role_dir.name.split("__")
                if len(parts) != 2 or parts[1] != old:
                    continue
                target = role_dir.with_name(f"{parts[0]}__{new}")
                if target.name in existing and target != role_dir:
                    raise ValueError(f"camera role target already exists: {target}")
                role_dir.rename(target)
                existing.discard(role_dir.name)
                existing.add(target.name)
                changed.add(cycle_root.name)
    return changed


def _cycle_records(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = catalog.get("cycles")
    if not isinstance(records, list):
        raise ValueError("cycle catalog cycles must be a list")
    return [record for record in records if isinstance(record, dict)]


def _assets(record: Mapping[str, Any]) -> Mapping[str, Any]:
    assets = record.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError(f"cycle assets are missing: {record.get('cycle_name')}")
    return assets


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _iso(value: pd.Timestamp | None) -> str | None:
    return None if value is None else pd.Timestamp(value).isoformat()


def _within_data(value: pd.Timestamp | None, frame: pd.DataFrame) -> pd.Timestamp | None:
    if value is None or frame.empty:
        return None
    return value if pd.Timestamp(frame["timestamp"].max()) >= value else None


def _rewrite_stage_column(
    frame: pd.DataFrame,
    stable: pd.Timestamp | None,
    defrost_start: pd.Timestamp | None,
    defrost_end: pd.Timestamp | None,
) -> None:
    if "cycle_stage" not in frame:
        return
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    stages = pd.Series("partial", index=frame.index, dtype="string")
    before_defrost = (
        timestamps.notna()
        if defrost_start is None
        else timestamps.lt(defrost_start)
    )
    if stable is not None:
        stages.loc[before_defrost & timestamps.lt(stable)] = "recovery"
        stages.loc[before_defrost & timestamps.ge(stable)] = "frost_development"
    elif defrost_start is not None:
        stages.loc[before_defrost] = "recovery"
    if defrost_start is not None:
        active = timestamps.ge(defrost_start)
        if defrost_end is not None:
            active &= timestamps.lt(defrost_end)
        stages.loc[active] = "defrost"
    frame.loc[:, "cycle_stage"] = stages


def _recompute_stage_dependent_fields(
    processed: pd.DataFrame,
    record: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> pd.DataFrame:
    """Recompute Process fields whose definitions depend on recovery stage."""
    from .features import add_dynamic_features
    from .process import _recompute_cycle_coordinates

    boundaries = record.get("boundaries", {})
    if not isinstance(boundaries, Mapping):
        boundaries = {}
    summary = pd.DataFrame(
        [
            {
                "experiment_id": record.get("experiment_id"),
                "cycle_id": record.get("cycle_id"),
                "stable_heating_start": _timestamp(
                    boundaries.get("stable_heating_start")
                ),
                "defrost_start": _timestamp(boundaries.get("defrost_start")),
            }
        ]
    )
    result = _recompute_cycle_coordinates(processed, summary)
    channels = registry.get("channels", {})
    if not isinstance(channels, Mapping):
        return result
    windows = _feature_windows(result, registry)
    if not windows:
        return result
    interval = int(registry.get("resample_interval_seconds", 10))
    updated = add_dynamic_features(
        result,
        channels,
        interval_seconds=interval,
        windows_minutes=windows,
    )
    return updated.sort_values(
        ["experiment_id", "timestamp"], kind="stable"
    ).reset_index(drop=True)


def _feature_windows(
    frame: pd.DataFrame, registry: Mapping[str, Any]
) -> list[int]:
    settings = registry.get("processing_settings")
    if isinstance(settings, Mapping):
        configured = settings.get("feature_windows_minutes")
        if isinstance(configured, (list, tuple)):
            return sorted({int(value) for value in configured if int(value) > 0})
    found: set[int] = set()
    pattern = re.compile(r"__(?:lag|delta|rolling_mean)_(\d+)min$")
    for column in frame.columns:
        match = pattern.search(str(column))
        if match is not None:
            found.add(int(match.group(1)))
    return sorted(found)
