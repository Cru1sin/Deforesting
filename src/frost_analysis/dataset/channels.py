"""Simple, explicit channel facts and processing policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_KINDS = {"continuous", "step", "event", "categorical", "protected", "derived"}
_CANDIDATE_KINDS = {"continuous", "step", "derived"}
_DIRECTIONS = {"increase", "decrease"}
_FORMULAS = {
    "cop",
    "evaporator_capacity",
    "pressure_ratio",
    "water_delta_temperature",
    "superheat_calculated",
}
_REQUIRED_KEYS = {"unit", "kind", "role", "resample", "missing", "analysis_candidate"}


def load_channels(path: Path) -> dict[str, dict[str, Any]]:
    """Load a small mapping of channel facts and validate its public contract."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict) or not isinstance(loaded.get("channels"), dict):
        raise ValueError("channels file must contain a 'channels' mapping")

    result: dict[str, dict[str, Any]] = {}
    for name, raw in loaded["channels"].items():
        if not isinstance(raw, dict):
            raise ValueError(f"channel {name} must be a mapping")
        channel = {str(key): value for key, value in raw.items()}
        _validate_channel(str(name), channel)
        result[str(name)] = channel
    if not result:
        raise ValueError("channels must not be empty")
    return result


def _validate_channel(name: str, channel: dict[str, Any]) -> None:
    missing = sorted(_REQUIRED_KEYS - set(channel))
    if missing:
        raise ValueError(f"channel {name} missing keys: {missing}")
    kind = str(channel["kind"])
    if kind not in _KINDS:
        raise ValueError(f"invalid kind for {name}: {kind}")
    if kind == "derived":
        _validate_derived(name, channel)
    else:
        _validate_source(name, channel, kind)
    candidate = bool(channel["analysis_candidate"])
    if candidate and kind not in _CANDIDATE_KINDS:
        raise ValueError(f"analysis_candidate is not allowed for {name}")
    if candidate and channel.get("expected_frost_direction") not in _DIRECTIONS:
        raise ValueError(f"analysis_candidate requires expected_frost_direction for {name}")
    if not candidate and channel.get("expected_frost_direction") not in (None, *_DIRECTIONS):
        raise ValueError(f"invalid expected_frost_direction for {name}")


def _validate_source(name: str, channel: dict[str, Any], kind: str) -> None:
    source_names = channel.get("source_names")
    if not isinstance(source_names, list) or not source_names or any(
        not isinstance(value, str) or not value for value in source_names
    ):
        raise ValueError(f"source channel requires exact source_names: {name}")
    if "formula" in channel or "dependencies" in channel:
        raise ValueError(f"non-derived channel cannot define formula or dependencies: {name}")
    for key in ("scale", "offset"):
        if key in channel and not isinstance(channel[key], (int, float)):
            raise ValueError(f"{key} must be numeric for {name}")
    if kind in {"event", "categorical", "protected"} and any(
        key in channel for key in ("scale", "offset")
    ):
        raise ValueError(f"{kind} channel cannot define scale or offset: {name}")


def _validate_derived(name: str, channel: dict[str, Any]) -> None:
    if "source_names" in channel:
        raise ValueError(f"derived channel cannot define source_names: {name}")
    formula = str(channel.get("formula", ""))
    if formula not in _FORMULAS:
        raise ValueError(f"unsupported derived formula for {name}: {formula}")
    dependencies = channel.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies or any(
        not isinstance(value, str) or not value for value in dependencies
    ):
        raise ValueError(f"derived channel requires dependencies: {name}")
