"""Small, explicit input discovery and pipeline output writers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config

_PREPARE_FILES = {
    "prepared_data.parquet",
    "cycle_summary.csv",
    "prepare_summary.json",
}
_RUN_FILES = {
    "prepared_data.parquet",
    "processed_data.parquet",
    "cycle_summary.csv",
    "candidate_channel_evidence.csv",
    "manifest.json",
}


@dataclass(frozen=True)
class InputFiles:
    """Files deliberately discovered from one raw experiment directory."""

    sensor_files: tuple[Path, ...]
    image_files: tuple[Path, ...]
    camera_mapping_file: Path


def discover_inputs(config: Config) -> InputFiles:
    """Discover sensor files at the raw root and images one camera level below it."""
    input_dir = config.input_dir
    if not input_dir.is_dir():
        raise FileNotFoundError(f"raw input directory does not exist: {input_dir}")

    sensor_paths = {
        path
        for pattern in config.sensor_globs
        for path in input_dir.glob(pattern)
        if path.is_file()
    }
    image_extensions = set(config.image_extensions)
    image_paths = {
        path
        for camera_dir in input_dir.iterdir()
        if camera_dir.is_dir()
        for path in camera_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    }
    camera_mapping = input_dir / config.camera_mapping_file
    return InputFiles(
        sensor_files=tuple(sorted(sensor_paths)),
        image_files=tuple(sorted(image_paths)),
        camera_mapping_file=camera_mapping,
    )


def write_prepare_outputs(
    prepared: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    prepare_summary: dict[str, Any],
    output_dir: Path,
    input_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write a reusable Prepared snapshot without creating a formal manifest."""
    _prepare_output_dir(output_dir, input_dir, _PREPARE_FILES, overwrite)
    prepared.to_parquet(output_dir / "prepared_data.parquet", index=False)
    cycle_summary.to_csv(output_dir / "cycle_summary.csv", index=False)
    summary = dict(prepare_summary)
    summary.setdefault("prepared_rows", len(prepared))
    summary.setdefault("cycle_count", len(cycle_summary))
    (output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_run_outputs(
    prepared: pd.DataFrame,
    processed: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    evidence: pd.DataFrame,
    prepare_summary: dict[str, Any],
    config: Config,
    config_path: Path,
    output_dir: Path,
    input_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write the four formal result files, then create the success manifest."""
    _prepare_output_dir(output_dir, input_dir, _RUN_FILES, overwrite)
    prepared.to_parquet(output_dir / "prepared_data.parquet", index=False)
    processed.to_parquet(output_dir / "processed_data.parquet", index=False)
    cycle_summary.to_csv(output_dir / "cycle_summary.csv", index=False)
    evidence.to_csv(output_dir / "candidate_channel_evidence.csv", index=False)

    manifest = {
        "experiment_id": config.experiment_id,
        "experiment_date": config.experiment_date,
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "config_sha256": _optional_sha256(config_path),
        "channels_sha256": _optional_sha256(config.channels_path),
        "git_commit": _git_commit(config.project_root),
        "prepare_summary": prepare_summary,
        "outputs": {
            "prepared_data": "prepared_data.parquet",
            "processed_data": "processed_data.parquet",
            "cycle_summary": "cycle_summary.csv",
            "candidate_channel_evidence": "candidate_channel_evidence.csv",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def ensure_output_outside_input(output_dir: Path, input_dir: Path) -> None:
    """Reject any derived output path equal to or below the raw input path."""
    input_resolved = input_dir.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == input_resolved or input_resolved in output_resolved.parents:
        raise ValueError("output directory must not be inside the raw input directory")


def _prepare_output_dir(
    output_dir: Path,
    input_dir: Path,
    known_files: set[str],
    overwrite: bool,
) -> None:
    ensure_output_outside_input(output_dir, input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_dir.iterdir() if path.name in known_files}
    if existing and not overwrite:
        raise FileExistsError(
            f"pipeline output files already exist: {sorted(existing)}; use overwrite=True"
        )
    if overwrite:
        for name in known_files:
            (output_dir / name).unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def optional_sha256(path: Path | None) -> str | None:
    """Return a file hash when a source file is available."""
    return None if path is None else _optional_sha256(path)


def git_commit(project_root: Path) -> str | None:
    """Return the repository commit used for a prepared snapshot."""
    return _git_commit(project_root)


def _git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )
    commit = result.stdout.strip()
    return commit or None


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__}")
