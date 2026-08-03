"""Canonical scientific schema for the self-contained Dataset v3."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]

IMAGE_COLUMNS = ("path", "time", "offset_seconds")
_IMAGE_COLUMN_RE = re.compile(r"^image_.+_(?:path|time|offset_seconds)$")


def is_image_column(name: object) -> bool:
    """Return whether a Processed column is one of the source image triples."""
    return _IMAGE_COLUMN_RE.fullmatch(str(name)) is not None


def drop_image_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove all source image triples without changing other column order."""
    return frame.drop(columns=[name for name in frame.columns if is_image_column(name)])


def registry_from_frame(
    frame: pd.DataFrame,
    channels: Mapping[str, Mapping[str, Any]],
    *,
    analysis_settings: Mapping[str, Any] | None = None,
    resample_interval_seconds: int = 10,
) -> dict[str, Any]:
    """Create a deterministic registry snapshot from one Processed frame."""
    scientific = drop_image_columns(frame)
    fields = [_field_spec(scientific, str(name)) for name in scientific.columns]
    channel_specs = {
        str(name): _channel_spec(str(name), settings, fields)
        for name, settings in channels.items()
    }
    return {
        "registry_version": 1,
        "resample_interval_seconds": int(resample_interval_seconds),
        "channels": channel_specs,
        "fields": fields,
        "analysis_settings": _jsonable(dict(analysis_settings or {})),
    }


def merge_registries(
    existing: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge registries, rejecting semantic changes and allowing new fields."""
    if int(existing.get("registry_version", 1)) != int(candidate.get("registry_version", 1)):
        raise ValueError("channel registry version changed")
    if int(existing.get("resample_interval_seconds", 10)) != int(
        candidate.get("resample_interval_seconds", 10)
    ):
        raise ValueError("channel registry resample interval changed")

    old_channels = _mapping(existing.get("channels"), "existing channels")
    new_channels = _mapping(candidate.get("channels"), "candidate channels")
    merged_channels: dict[str, Any] = {
        str(name): _jsonable(value) for name, value in old_channels.items()
    }
    for name, value in new_channels.items():
        channel_name = str(name)
        normalized = _jsonable(value)
        if channel_name in merged_channels:
            merged_channels[channel_name] = _merge_channel(
                channel_name, merged_channels[channel_name], normalized
            )
        else:
            merged_channels[channel_name] = normalized

    old_fields = _field_mapping(existing.get("fields"), "existing fields")
    new_fields = _field_mapping(candidate.get("fields"), "candidate fields")
    merged_fields: list[dict[str, Any]] = []
    for name, old_field in old_fields.items():
        candidate_field = new_fields.get(name)
        if candidate_field is None:
            merged_fields.append(dict(old_field))
            continue
        merged_fields.append(_merge_field(old_field, candidate_field))
    for name, field in new_fields.items():
        if name not in old_fields:
            merged_fields.append(dict(field))

    old_analysis = _jsonable(existing.get("analysis_settings", {}))
    new_analysis = _jsonable(candidate.get("analysis_settings", {}))
    if old_analysis != new_analysis:
        raise ValueError("channel registry analysis settings changed")
    return {
        "registry_version": int(existing.get("registry_version", 1)),
        "resample_interval_seconds": int(existing.get("resample_interval_seconds", 10)),
        "channels": merged_channels,
        "fields": merged_fields,
        "analysis_settings": old_analysis,
    }


def canonical_registry_hash(registry: Mapping[str, Any]) -> str:
    """Hash only the scientific registry content, excluding a stored hash."""
    payload = {str(key): value for key, value in registry.items() if key != "canonical_hash"}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_frame(frame: pd.DataFrame, registry: Mapping[str, Any]) -> pd.DataFrame:
    """Project a source frame into the registry's nullable ordered field schema."""
    source = drop_image_columns(frame).copy()
    fields = _field_mapping(registry.get("fields"), "registry fields")
    unknown = [str(name) for name in source.columns if str(name) not in fields]
    if unknown:
        raise ValueError(f"Processed columns are not in channel registry: {unknown}")
    for name, field in fields.items():
        if name not in source or source[name].isna().all():
            source[name] = _null_series(
                len(source), str(field.get("logical_type", "null")), index=source.index
            )
    return source.loc[:, list(fields)]


def _field_spec(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    if frame[name].isna().all():
        return {
            "name": name,
            "logical_type": "null",
            "nullable": True,
        }
    arrow_field = pa.Schema.from_pandas(frame[[name]], preserve_index=False).field(name)
    return {
        "name": name,
        "logical_type": str(arrow_field.type),
        "nullable": True,
    }


def _channel_spec(
    name: str, settings: Mapping[str, Any], fields: list[dict[str, Any]]
) -> dict[str, Any]:
    field = next((value for value in fields if value["name"] == name), None)
    return {
        "kind": str(settings.get("kind", "continuous")),
        "role": settings.get("role"),
        "unit": settings.get("unit"),
        "dtype": (field or {}).get("logical_type", settings.get("dtype", "null")),
        "resample": settings.get("resample", "mean"),
        "missing": settings.get("missing", "none"),
        "source_names": _jsonable(settings.get("source_names", [])),
        "valid_range": _jsonable(settings.get("valid_range")),
        "scale": settings.get("scale"),
        "offset": settings.get("offset"),
        "allowed_values": _jsonable(settings.get("allowed_values")),
        "formula": settings.get("formula"),
        "dependencies": _jsonable(settings.get("dependencies", [])),
        "analysis_candidate": bool(settings.get("analysis_candidate", False)),
        "expected_frost_direction": settings.get("expected_frost_direction"),
        "coverage_required": bool(settings.get("coverage_required", False)),
    }


def _merge_field(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    old_type = str(old.get("logical_type", "null"))
    new_type = str(new.get("logical_type", "null"))
    if old_type != "null" and new_type != "null" and old_type != new_type:
        raise ValueError(
            f"channel registry field type changed: {old.get('name')}: "
            f"{old_type} -> {new_type}"
        )
    return {
        "name": str(old["name"]),
        "logical_type": new_type if old_type == "null" else old_type,
        "nullable": bool(old.get("nullable", True) or new.get("nullable", True)),
    }


def _merge_channel(
    name: str, old: Mapping[str, Any], new: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge one channel, treating an unavailable date dtype as nullable."""
    old_value = {str(key): _jsonable(value) for key, value in old.items()}
    new_value = {str(key): _jsonable(value) for key, value in new.items()}
    old_dtype = str(old_value.get("dtype", "null"))
    new_dtype = str(new_value.get("dtype", "null"))
    for key in set(old_value) | set(new_value):
        if key == "dtype":
            continue
        if _canonical_json(old_value.get(key)) != _canonical_json(new_value.get(key)):
            raise ValueError(f"channel registry conflict: {name}")
    if old_dtype != "null" and new_dtype != "null" and old_dtype != new_dtype:
        raise ValueError(f"channel registry conflict: {name}")
    merged = dict(old_value)
    if old_dtype == "null" and new_dtype != "null":
        merged["dtype"] = new_dtype
    return merged


def _field_mapping(value: object, name: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, Mapping) or "name" not in item:
            raise ValueError(f"{name} contains an invalid field")
        field = {str(key): _jsonable(raw) for key, raw in item.items()}
        field_name = str(field["name"])
        if field_name in result:
            raise ValueError(f"{name} contains duplicate field: {field_name}")
        result[field_name] = field
    return result


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _null_series(
    length: int, logical_type: str, *, index: pd.Index | None = None
) -> pd.Series:
    series_index = range(length) if index is None else index
    if logical_type == "null":
        return pd.Series([None] * length, index=series_index, dtype=object)
    if logical_type.startswith("timestamp"):
        return pd.Series(pd.NaT, index=series_index, dtype="datetime64[ns]")
    if logical_type in {"double", "float", "float32", "float64"}:
        return pd.Series(pd.NA, index=series_index, dtype="Float64")
    if logical_type.startswith("int") or logical_type.startswith("uint"):
        return pd.Series(pd.NA, index=series_index, dtype="Int64")
    if logical_type == "bool":
        return pd.Series(pd.NA, index=series_index, dtype="boolean")
    return pd.Series(pd.NA, index=series_index, dtype="string")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
