"""Dataset-local scientific transforms and camera-role edits."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .cycles import find_stable_heating_start


def apply_recovery(
    original: pd.DataFrame,
    processed: pd.DataFrame,
    image_metadata: pd.DataFrame,
    record: dict[str, Any],
    registry: dict[str, Any],
    *,
    mode: str,
    seconds: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply a recovery rule to in-memory Dataset frames."""
    if mode not in {"seconds", "ts-minus"}:
        raise ValueError(f"unsupported recovery edit mode: {mode}")
    if mode == "seconds" and (seconds is None or seconds < 0):
        raise ValueError("recovery seconds must be a nonnegative integer")

    original_result = original.copy()
    processed_result = processed.copy()
    metadata_result = image_metadata.copy()
    original_result["timestamp"] = pd.to_datetime(
        original_result["timestamp"], errors="raise"
    )
    processed_result["timestamp"] = pd.to_datetime(
        processed_result["timestamp"], errors="raise"
    )

    boundaries = record.setdefault("boundaries", {})
    if not isinstance(boundaries, dict):
        raise ValueError("cycle boundaries must be a mapping")
    heating_start = _timestamp(boundaries.get("heating_start"))
    defrost_start = _timestamp(boundaries.get("defrost_start"))
    defrost_end = _timestamp(boundaries.get("defrost_end"))
    if heating_start is None and not original_result.empty:
        heating_start = pd.Timestamp(original_result["timestamp"].min())

    recovery_settings = registry.get("recovery_edit", {})
    if mode == "seconds":
        candidate = (
            heating_start + pd.Timedelta(seconds=int(seconds or 0))
            if heating_start is not None
            else None
        )
        stable = _within_data(candidate, original_result)
    else:
        fallback_seconds = (
            recovery_settings.get("fallback_seconds", 180)
            if isinstance(recovery_settings, Mapping)
            else 180
        )
        stable = find_stable_heating_start(
            original_result,
            heating_start,
            defrost_start,
            {"stable_heating_seconds": fallback_seconds},
        )

    boundaries["heating_start"] = _iso(heating_start)
    boundaries["stable_heating_start"] = _iso(stable)
    _rewrite_stage_column(original_result, stable, defrost_start, defrost_end)
    _rewrite_stage_column(processed_result, stable, defrost_start, defrost_end)
    processed_result = _recompute_stage_dependent_fields(
        processed_result, record, registry
    )
    metadata_result = _update_image_stages(metadata_result, processed_result, record)

    registry["recovery_edit"] = {
        "mode": mode,
        "seconds": int(seconds) if mode == "seconds" and seconds is not None else None,
        "fallback_seconds": (
            float(recovery_settings.get("fallback_seconds", 180))
            if isinstance(recovery_settings, Mapping)
            else 180.0
        ),
        "managed": True,
    }
    return original_result, processed_result, metadata_result


def apply_baseline(
    processed: pd.DataFrame,
    record: dict[str, Any],
    registry: dict[str, Any],
    *,
    seconds: int,
) -> pd.DataFrame:
    """Apply the fixed stable-start baseline window to one frame."""
    if seconds < 0:
        raise ValueError("baseline seconds must be nonnegative")
    channels = registry.get("channels", {})
    if not isinstance(channels, Mapping):
        raise ValueError("channel registry channels must be a mapping")
    eligible = [
        str(name)
        for name, settings in channels.items()
        if isinstance(settings, Mapping)
        and str(settings.get("kind")) in {"continuous", "step", "derived"}
        and (
            bool(settings.get("analysis_candidate"))
            or settings.get("role") == "performance"
        )
    ]

    result = processed.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    boundaries = record.setdefault("boundaries", {})
    if not isinstance(boundaries, dict):
        raise ValueError("cycle boundaries must be a mapping")
    stable = _timestamp(boundaries.get("stable_heating_start"))
    end = stable + pd.Timedelta(seconds=seconds) if stable is not None else None
    enough = end is not None and not result.empty and result["timestamp"].max() >= end

    for name in eligible:
        result[f"{name}__baseline"] = pd.NA
        result[f"{name}__baseline_residual"] = pd.NA

    if enough and stable is not None and end is not None:
        stage = result.get("cycle_stage", pd.Series("", index=result.index))
        window = result.loc[
            result["timestamp"].ge(stable)
            & result["timestamp"].lt(end)
            & stage.eq("frost_development")
        ]
        for name in eligible:
            if name not in window:
                continue
            values = pd.to_numeric(window[name], errors="coerce")
            observed = values.notna()
            imputed = window.get(f"{name}__imputed")
            if imputed is not None:
                observed &= ~imputed.fillna(False).astype(bool)
            finite = values.loc[observed]
            if finite.empty:
                continue
            baseline = float(finite.median())
            result.loc[:, f"{name}__baseline"] = baseline
            result.loc[:, f"{name}__baseline_residual"] = (
                pd.to_numeric(result[name], errors="coerce") - baseline
            )
        boundaries["baseline_start"] = _iso(stable)
        boundaries["baseline_end"] = _iso(end)
    else:
        boundaries["baseline_start"] = None
        boundaries["baseline_end"] = None

    registry["baseline_seconds"] = int(seconds)
    registry["baseline_managed"] = True
    return result


def rename_camera_role(dataset_dir: Path, renames: Mapping[str, str]) -> set[str]:
    """Rename only the mutable role suffix of source/role image directories."""
    changed: set[str] = set()
    images_root = dataset_dir / "images"
    if not images_root.is_dir():
        return changed
    for cycle_root in sorted(path for path in images_root.iterdir() if path.is_dir()):
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
    before_defrost = timestamps.notna() if defrost_start is None else timestamps.lt(defrost_start)
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
    from .features import recompute_dynamic_features
    from .process import recompute_cycle_coordinates

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
    result = recompute_cycle_coordinates(processed, summary)
    channels = registry.get("channels", {})
    if not isinstance(channels, Mapping):
        return result
    windows = _feature_windows(result, registry)
    if not windows:
        return result
    interval = int(registry.get("resample_interval_seconds", 10))
    return recompute_dynamic_features(
        result,
        channels,
        interval_seconds=interval,
        windows_minutes=windows,
    ).sort_values(["experiment_id", "timestamp"], kind="stable").reset_index(drop=True)


def _update_image_stages(
    metadata: pd.DataFrame,
    processed: pd.DataFrame,
    record: Mapping[str, Any],
) -> pd.DataFrame:
    result = metadata.copy()
    cycle_name = str(record.get("cycle_name"))
    if result.empty or "cycle_name" not in result or "cycle_stage" not in result:
        return result
    mask = result["cycle_name"].astype(str).eq(cycle_name)
    if not mask.any():
        return result
    timestamps = pd.to_datetime(result.loc[mask, "matched_timestamp"], errors="coerce")
    if "image_time" in result:
        timestamps = timestamps.fillna(
            pd.to_datetime(result.loc[mask, "image_time"], errors="coerce")
        )
    lookup = (
        processed[["timestamp", "cycle_stage"]]
        .assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"]))
        .drop_duplicates("timestamp")
        .set_index("timestamp")["cycle_stage"]
    )
    result.loc[mask, "cycle_stage"] = timestamps.map(lookup).fillna(
        result.loc[mask, "cycle_stage"]
    )
    return result


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


def _feature_windows(frame: pd.DataFrame, registry: Mapping[str, Any]) -> list[int]:
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
