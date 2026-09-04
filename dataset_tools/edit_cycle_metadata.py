"""Dataset-local scientific transforms and camera-role edits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .builder.detect_cycles import find_defrost_preparation_start, resolve_stable_heating_start


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
    preparation_start = _timestamp(boundaries.get("defrost_preparation_start"))
    defrost_end = _timestamp(boundaries.get("defrost_end"))
    if heating_start is None and not original_result.empty:
        heating_start = pd.Timestamp(original_result["timestamp"].min())

    recovery_settings = registry.get("recovery_edit", {})
    if mode == "seconds":
        stable = resolve_stable_heating_start(
            original_result,
            heating_start,
            defrost_start,
            recovery_settings if isinstance(recovery_settings, Mapping) else {},
            mode="seconds",
            seconds=int(seconds or 0),
        )
    else:
        fallback_seconds = (
            recovery_settings.get("fallback_seconds", 180)
            if isinstance(recovery_settings, Mapping)
            else 180
        )
        stable = resolve_stable_heating_start(
            original_result,
            heating_start,
            defrost_start,
            {"stable_heating_seconds": fallback_seconds},
        )

    boundaries["heating_start"] = _iso(heating_start)
    boundaries["stable_heating_start"] = _iso(stable)
    _rewrite_stage_column(
        original_result, stable, preparation_start, defrost_start, defrost_end
    )
    _rewrite_stage_column(
        processed_result, stable, preparation_start, defrost_start, defrost_end
    )
    processed_result = _recompute_stage_dependent_fields(
        processed_result, record, registry
    )
    metadata_result = _update_image_stages(metadata_result, record)

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


def apply_defrost_preparation(
    original: pd.DataFrame,
    processed: pd.DataFrame,
    image_metadata: pd.DataFrame,
    record: dict[str, Any],
    registry: dict[str, Any],
    *,
    setpoint_drop_hz: float = 10.0,
    lookback_seconds: float = 120.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive the pre-defrost unloading interval from Dataset sensor facts."""
    original_result = original.copy()
    processed_result = processed.copy()
    metadata_result = image_metadata.copy()
    original_result["timestamp"] = pd.to_datetime(original_result["timestamp"], errors="raise")
    processed_result["timestamp"] = pd.to_datetime(processed_result["timestamp"], errors="raise")
    boundaries = record.setdefault("boundaries", {})
    if not isinstance(boundaries, dict):
        raise ValueError("cycle boundaries must be a mapping")
    stable = _timestamp(boundaries.get("stable_heating_start"))
    defrost_start = _timestamp(boundaries.get("defrost_start"))
    defrost_end = _timestamp(boundaries.get("defrost_end"))
    preparation_start = find_defrost_preparation_start(
        original_result,
        stable,
        defrost_start,
        {
            "defrost_preparation_setpoint_drop_hz": setpoint_drop_hz,
            "defrost_preparation_lookback_seconds": lookback_seconds,
        },
    )
    boundaries["defrost_preparation_start"] = _iso(preparation_start)
    _rewrite_stage_column(
        original_result, stable, preparation_start, defrost_start, defrost_end
    )
    _rewrite_stage_column(
        processed_result, stable, preparation_start, defrost_start, defrost_end
    )
    processed_result = _recompute_stage_dependent_fields(
        processed_result, record, registry
    )
    metadata_result = _update_image_stages(metadata_result, record)
    registry["defrost_preparation"] = {
        "setpoint_drop_hz": float(setpoint_drop_hz),
        "lookback_seconds": float(lookback_seconds),
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
    from .builder.baseline import apply_fixed_baseline

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
        result, _unavailable = apply_fixed_baseline(
            result,
            eligible,
            start=stable,
            end=end,
            minimum_observed_coverage=0.0,
            stage="frost_development",
        )
        boundaries["baseline_start"] = _iso(stable)
        boundaries["baseline_end"] = _iso(end)
    else:
        boundaries["baseline_start"] = None
        boundaries["baseline_end"] = None

    registry["baseline_seconds"] = int(seconds)
    registry["baseline_managed"] = True
    return result


def _rewrite_stage_column(
    frame: pd.DataFrame,
    stable: pd.Timestamp | None,
    preparation_start: pd.Timestamp | None,
    defrost_start: pd.Timestamp | None,
    defrost_end: pd.Timestamp | None,
) -> None:
    if "cycle_stage" not in frame:
        return
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    stages = pd.Series("partial", index=frame.index, dtype="string")
    before_defrost = timestamps.notna() if defrost_start is None else timestamps.lt(defrost_start)
    if stable is not None:
        frost_end = preparation_start or defrost_start
        stages.loc[before_defrost & timestamps.lt(stable)] = "recovery"
        stages.loc[
            before_defrost
            & timestamps.ge(stable)
            & (timestamps.lt(frost_end) if frost_end is not None else True)
        ] = "frost_development"
    elif defrost_start is not None:
        stages.loc[before_defrost] = "recovery"
    if preparation_start is not None and defrost_start is not None:
        stages.loc[
            timestamps.ge(preparation_start) & timestamps.lt(defrost_start)
        ] = "defrost_preparation"
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
    from .builder.build_cycle_tables import recompute_cycle_coordinates

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
                "defrost_preparation_start": _timestamp(
                    boundaries.get("defrost_preparation_start")
                ),
                "defrost_start": _timestamp(boundaries.get("defrost_start")),
            }
        ]
    )
    return recompute_cycle_coordinates(processed, summary).sort_values(
        ["experiment_id", "timestamp"], kind="stable"
    ).reset_index(drop=True)


def _update_image_stages(
    metadata: pd.DataFrame,
    record: Mapping[str, Any],
) -> pd.DataFrame:
    result = metadata.copy()
    cycle_name = str(record.get("cycle_name"))
    if result.empty or "cycle_name" not in result or "cycle_stage" not in result:
        return result
    mask = result["cycle_name"].astype(str).eq(cycle_name)
    if not mask.any():
        return result
    boundaries = record.get("boundaries", {})
    if not isinstance(boundaries, Mapping):
        boundaries = {}
    images = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(result.loc[mask, "image_time"], errors="coerce"),
            "cycle_stage": result.loc[mask, "cycle_stage"].to_numpy(),
        },
        index=result.index[mask],
    )
    _rewrite_stage_column(
        images,
        _timestamp(boundaries.get("stable_heating_start")),
        _timestamp(boundaries.get("defrost_preparation_start")),
        _timestamp(boundaries.get("defrost_start")),
        _timestamp(boundaries.get("defrost_end")),
    )
    result.loc[mask, "cycle_stage"] = images["cycle_stage"]
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
