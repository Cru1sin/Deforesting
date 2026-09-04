"""Raw experiment input discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dataset_settings import Config


@dataclass(frozen=True)
class InputFiles:
    sensor_files: tuple[Path, ...]
    image_files: tuple[Path, ...]


def discover_inputs(config: Config) -> InputFiles:
    input_dir = config.input_dir
    if not input_dir.is_dir():
        raise FileNotFoundError(f"raw input directory does not exist: {input_dir}")

    sensor_files = tuple(
        sorted(
            {
                path
                for pattern in config.sensor_globs
                for path in input_dir.glob(pattern)
                if path.is_file()
            }
        )
    )
    extensions = set(config.image_extensions)
    image_files = tuple(
        sorted(
            path
            for directory in input_dir.iterdir()
            if directory.is_dir()
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
    )
    return InputFiles(sensor_files, image_files)
