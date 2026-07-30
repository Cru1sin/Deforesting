"""Strict loading for the small flat configuration contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    """Configuration shared by Prepare, Process, Analyze, and the CLI."""

    project_root: Path
    experiment_id: str
    experiment_date: str
    input_dir: Path
    channels_path: Path
    sensor_globs: tuple[str, ...]
    image_extensions: tuple[str, ...]
    camera_mapping_file: str
    cycles: dict[str, Any]
    process: dict[str, Any]
    analysis: dict[str, Any]
    config_path: Path | None = None


def load_config(path: Path) -> Config:
    """Load one flat YAML config and resolve paths relative to the repository."""
    config_path = path.resolve()
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config must be a YAML mapping")
    required = {
        "experiment_id",
        "experiment_date",
        "input_dir",
        "channels_path",
        "sensor_globs",
        "image_extensions",
        "camera_mapping_file",
        "cycles",
        "process",
        "analysis",
    }
    missing = sorted(required - set(loaded))
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    project_root = _find_project_root(config_path)
    experiment_date = str(loaded["experiment_date"])
    if not _is_iso_date(experiment_date):
        raise ValueError("experiment_date must use ISO YYYY-MM-DD format")
    return Config(
        project_root=project_root,
        experiment_id=str(loaded["experiment_id"]),
        experiment_date=experiment_date,
        input_dir=_resolve_path(project_root, loaded["input_dir"]),
        channels_path=_resolve_path(project_root, loaded["channels_path"]),
        sensor_globs=_tuple_strings(loaded["sensor_globs"], "sensor_globs"),
        image_extensions=_image_extensions(loaded["image_extensions"]),
        camera_mapping_file=str(loaded["camera_mapping_file"]),
        cycles=_mapping(loaded["cycles"], "cycles"),
        process=_mapping(loaded["process"], "process"),
        analysis=_mapping(loaded["analysis"], "analysis"),
        config_path=config_path,
    )


def _find_project_root(config_path: Path) -> Path:
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("could not find project root containing pyproject.toml")


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _tuple_strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not str(item).strip() for item in value):
        raise ValueError(f"{name} must be a non-empty list of strings")
    return tuple(str(item) for item in value)


def _image_extensions(value: Any) -> tuple[str, ...]:
    return tuple(
        item.lower() if item.startswith(".") else f".{item.lower()}"
        for item in _tuple_strings(value, "image_extensions")
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return {str(key): item for key, item in value.items()}


def _is_iso_date(value: str) -> bool:
    return (
        len(value) == 10
        and value[4] == "-"
        and value[7] == "-"
        and value.replace("-", "").isdigit()
    )
