"""Minimal channel facts and processing-policy loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_KINDS = {"continuous", "step", "event", "categorical", "protected", "derived"}
_CANDIDATE_KINDS = {"continuous", "step", "derived"}
_DIRECTIONS = {"increase", "decrease"}
_REQUIRED_KEYS = {"unit", "kind", "role", "resample", "missing", "analysis_candidate"}


def load_channels(path: Path) -> dict[str, dict[str, Any]]:
    """Load and minimally validate the declarative channel facts."""
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
    candidate = bool(channel["analysis_candidate"])
    if candidate and kind not in _CANDIDATE_KINDS:
        raise ValueError(f"analysis_candidate is not allowed for {name}")
    if candidate and channel.get("expected_frost_direction") not in _DIRECTIONS:
        raise ValueError(f"analysis_candidate requires expected_frost_direction for {name}")
    if kind == "derived":
        if not str(channel.get("formula", "")).strip():
            raise ValueError(f"derived channel requires formula: {name}")
        dependencies = channel.get("dependencies")
        if not isinstance(dependencies, list) or not dependencies:
            raise ValueError(f"derived channel requires dependencies: {name}")
    elif "formula" in channel:
        raise ValueError(f"non-derived channel cannot define formula: {name}")
