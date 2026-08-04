"""Small, Dataset-local scientific and camera transformations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .cycles import _stable_start
from .dataset_io import write_atomic_csv, write_atomic_parquet


def apply_recovery_edit(
    dataset_dir: Path,
    catalog: dict[str, Any],
    *,
    mode: str,
    recovery_seconds: int | None = None,
) -> set[str]:
    """Update stable-heating boundaries and stage labels in both resolutions."""
    if mode not in {"seconds", "ts-minus"}:
        raise ValueError(f"unsupported recovery edit mode: {mode}")
    if mode == "seconds" and (recovery_seconds is None or recovery_seconds < 0):
        raise ValueError("recovery_seconds must be a nonnegative integer")
    changed: set[str] = set()
    for record in _cycle_records(catalog):
        cycle_name = str(record["cycle_name"])
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
            stable = _stable_start(
                original,
                heating_start,
                defrost_start,
                {"stable_heating_seconds": 180},
            )
        boundaries["heating_start"] = _iso(heating_start)
        boundaries["stable_heating_start"] = _iso(stable)
        _rewrite_stage_column(original, stable, defrost_start, defrost_end)
        _rewrite_stage_column(processed, stable, defrost_start, defrost_end)
        write_atomic_csv(original, original_path)
        write_atomic_parquet(processed, processed_path)
        csv_asset = assets.get("csv")
        if isinstance(csv_asset, str):
            write_atomic_csv(processed, dataset_dir / csv_asset)
        changed.add(cycle_name)
    return changed


def apply_baseline_edit(  # noqa: C901
    dataset_dir: Path,
    catalog: dict[str, Any],
    *,
    baseline_seconds: int,
    registry: Mapping[str, Any],
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
    changed: set[str] = set()
    for record in _cycle_records(catalog):
        cycle_name = str(record["cycle_name"])
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
        if enough:
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
        timestamps.lt(defrost_start) if defrost_start is not None else timestamps.notna()
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
