"""Small, explicit input discovery and pipeline output writers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .config import Config, resolved_config_mapping, resolved_config_sha256
from .validation import validate_processed

_PREPARE_FILES = {
    "prepared_data.parquet",
    "cycle_summary.csv",
    "prepare_summary.json",
}
_PROCESS_FILES = {"processed_data.parquet", "cycle_summary.csv"}
_ANALYZE_FILES = {"candidate_channel_evidence.csv"}
_EVIDENCE_FILES = {
    "cycle_eligibility.csv",
    "feature_cycle_metrics.csv",
    "future_association.csv",
    "feature_profile.csv",
    "feature_pair_similarity.csv",
    "evidence_manifest.json",
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


@dataclass(frozen=True)
class EvidenceRunContract:
    run_dir: Path
    experiment_id: str
    experiment_date: str
    grid_interval_seconds: int
    channels_sha256: str
    baseline_policy_sha256: str
    baseline_reference_type: str
    manifest_sha256: str


@dataclass(frozen=True)
class EvidenceLoadResult:
    processed: pd.DataFrame
    cycle_summary: pd.DataFrame
    run_contracts: tuple[EvidenceRunContract, ...]
    grid_interval_seconds: int


def load_evidence_runs(
    run_dirs: Sequence[Path],
    *,
    registry_hash: str,
) -> EvidenceLoadResult:
    """Load and validate immutable Processed snapshots for Evidence."""
    resolved_dirs = [path.resolve() for path in run_dirs]
    if not resolved_dirs:
        raise ValueError("at least one --run-dir is required")
    if len(set(resolved_dirs)) != len(resolved_dirs):
        raise ValueError("duplicate resolved run directory")
    processed_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    contracts: list[EvidenceRunContract] = []
    for run_dir in resolved_dirs:
        processed, summary, contract = _load_evidence_run(run_dir, registry_hash)
        processed_frames.append(processed)
        summary_frames.append(summary)
        contracts.append(contract)
    processed = pd.concat(processed_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    _reject_duplicate_keys(processed, ["experiment_id", "experiment_date", "cycle_id", "timestamp"])
    _reject_duplicate_keys(summary, ["experiment_id", "experiment_date", "cycle_id"])
    intervals = {item.grid_interval_seconds for item in contracts}
    if len(intervals) != 1:
        raise ValueError("inconsistent grid interval across evidence runs")
    policies = {item.baseline_policy_sha256 for item in contracts}
    if len(policies) != 1:
        raise ValueError("inconsistent baseline policy across evidence runs")
    references = {item.baseline_reference_type for item in contracts}
    if len(references) != 1:
        raise ValueError("inconsistent baseline reference type across evidence runs")
    return EvidenceLoadResult(processed, summary, tuple(contracts), intervals.pop())


def _load_evidence_run(
    run_dir: Path,
    registry_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame, EvidenceRunContract]:
    processed_path = run_dir / "processed_data.parquet"
    summary_path = run_dir / "cycle_summary.csv"
    manifest_path = run_dir / "manifest.json"
    if not processed_path.is_file() or not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"evidence run must contain manifest.json, processed_data.parquet and cycle_summary.csv: {run_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a JSON object: {manifest_path}")
    experiment_id = str(manifest.get("experiment_id", ""))
    experiment_date = str(manifest.get("experiment_date", ""))
    if not experiment_id or not _is_iso_date(experiment_date):
        raise ValueError(f"manifest must contain experiment_id and ISO experiment_date: {run_dir}")
    provenance = manifest.get("config_provenance")
    resolved = manifest.get("resolved_config")
    if not isinstance(provenance, dict) or not isinstance(resolved, dict):
        raise ValueError(f"manifest lacks config provenance or resolved config: {run_dir}")
    channels_sha256 = str(provenance.get("channels_sha256", ""))
    if channels_sha256 != registry_hash:
        raise ValueError(f"run channels hash disagrees with current registry: {run_dir}")
    process = resolved.get("process")
    if not isinstance(process, dict) or "resample_interval_seconds" not in process:
        raise ValueError(f"manifest lacks process resample interval: {run_dir}")
    interval = int(process["resample_interval_seconds"])
    baseline = process.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError(f"manifest lacks resolved baseline policy: {run_dir}")
    baseline_payload = json.dumps(
        baseline, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    baseline_hash = hashlib.sha256(baseline_payload).hexdigest()
    processed = pd.read_parquet(processed_path)
    summary = pd.read_csv(summary_path)
    for column in (
        "heating_start",
        "stable_heating_start",
        "defrost_start",
        "defrost_end",
        "baseline_start",
        "baseline_end",
    ):
        if column in summary:
            summary[column] = pd.to_datetime(summary[column], errors="coerce")
    for frame, name in ((processed, "processed"), (summary, "cycle summary")):
        for column, expected in (("experiment_id", experiment_id), ("experiment_date", experiment_date)):
            if column not in frame:
                frame[column] = expected
            if frame[column].astype(str).ne(expected).any():
                raise ValueError(f"{name} {column} disagrees with manifest in {run_dir}")
        if "cycle_id" not in frame:
            raise ValueError(f"{name} requires cycle_id in {run_dir}")
    validate_processed(processed, summary)
    types = (
        summary.get("baseline_reference_type", pd.Series(dtype="string"))
        .dropna()
        .astype(str)
        .str.strip()
    )
    types = types[types.ne("")]
    if types.nunique() != 1:
        raise ValueError(f"inconsistent_baseline_reference_type_within_run: {run_dir}")
    contract = EvidenceRunContract(
        run_dir=run_dir,
        experiment_id=experiment_id,
        experiment_date=experiment_date,
        grid_interval_seconds=interval,
        channels_sha256=channels_sha256,
        baseline_policy_sha256=baseline_hash,
        baseline_reference_type=str(types.iloc[0]),
        manifest_sha256=_sha256(manifest_path),
    )
    return processed, summary, contract


def _reject_duplicate_keys(frame: pd.DataFrame, keys: list[str]) -> None:
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise ValueError(f"evidence input missing key columns: {missing}")
    if frame.duplicated(keys).any():
        raise ValueError(f"duplicate evidence key: {keys}")


def is_iso_date(value: str) -> bool:
    try:
        from datetime import date

        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _is_iso_date(value: str) -> bool:
    return is_iso_date(value)


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
    return InputFiles(
        sensor_files=tuple(sorted(sensor_paths)),
        image_files=tuple(sorted(image_paths)),
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
    summary.setdefault("prepared_row_count", len(prepared))
    summary.setdefault("cycle_count", len(cycle_summary))
    (output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_process_outputs(
    processed: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    output_dir: Path,
    input_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write standalone Process results without creating a manifest."""
    _prepare_output_dir(output_dir, input_dir, _PROCESS_FILES, overwrite)
    processed.to_parquet(output_dir / "processed_data.parquet", index=False)
    cycle_summary.to_csv(output_dir / "cycle_summary.csv", index=False)


def write_analysis_outputs(
    evidence: pd.DataFrame,
    output_dir: Path,
    input_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write standalone Analysis results without creating a manifest."""
    _prepare_output_dir(output_dir, input_dir, _ANALYZE_FILES, overwrite)
    evidence.to_csv(output_dir / "candidate_channel_evidence.csv", index=False)


def write_evidence_outputs(
    bundle: Any,
    output_dir: Path,
    input_run_dirs: Path | Sequence[Path],
    *,
    load_result: EvidenceLoadResult | None = None,
    settings: Any | None = None,
    candidate_registry_path: Path | None = None,
    project_root: Path | None = None,
    legacy_evidence: pd.DataFrame | None = None,
    overwrite: bool = False,
) -> None:
    """Write the five batch evidence tables and their reproducibility manifest."""
    run_dirs = [input_run_dirs] if isinstance(input_run_dirs, Path) else list(input_run_dirs)
    if not run_dirs:
        raise ValueError("evidence output requires at least one input run directory")
    for run_dir in run_dirs:
        ensure_output_outside_input(output_dir, run_dir)
    known_files = set(_EVIDENCE_FILES) | {"candidate_channel_evidence.csv"}
    _prepare_output_dir(output_dir, run_dirs[0], known_files, overwrite)
    tables = {
        "cycle_eligibility.csv": bundle.cycle_eligibility,
        "feature_cycle_metrics.csv": bundle.feature_cycle_metrics,
        "future_association.csv": bundle.future_association,
        "feature_profile.csv": bundle.feature_profile,
        "feature_pair_similarity.csv": bundle.feature_pair_similarity,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)
    if legacy_evidence is not None:
        legacy_evidence.to_csv(output_dir / "candidate_channel_evidence.csv", index=False)
    if load_result is not None:
        manifest_hashes: dict[str, str | None] = {
            str(contract.run_dir): contract.manifest_sha256
            for contract in load_result.run_contracts
        }
    else:
        manifest_hashes = {
            str(path.resolve()): optional_sha256(path / "manifest.json") for path in run_dirs
        }
    output_files = {name: name for name in (*tables, "evidence_manifest.json")}
    output_row_counts = {name: len(table) for name, table in tables.items()}
    if legacy_evidence is not None:
        output_files["candidate_channel_evidence.csv"] = "candidate_channel_evidence.csv"
        output_row_counts["candidate_channel_evidence.csv"] = len(legacy_evidence)
    manifest = {
        "analysis_version": "frost-cycle-evidence-v1",
        "git_commit": _git_commit(project_root) if project_root is not None else None,
        "created_at": datetime.now(UTC).isoformat(),
        "input_run_dirs": [str(path.resolve()) for path in run_dirs],
        "input_manifest_hashes": manifest_hashes,
        "analysis_settings": _serializable_settings(settings),
        "candidate_registry_hash": optional_sha256(candidate_registry_path),
        "output_files": output_files,
        "output_row_counts": output_row_counts,
    }
    (output_dir / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def remove_manifest_for_overwrite(output_dir: Path, input_dir: Path, *, overwrite: bool) -> None:
    """Remove only a prior formal manifest before an overwrite run starts."""
    ensure_output_outside_input(output_dir, input_dir)
    if overwrite:
        (output_dir / "manifest.json").unlink(missing_ok=True)


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

    experiment_config = config.config_path or config_path.resolve()
    if config.defaults_path is None:
        raise ValueError("schema-v2 config must define defaults_path")
    manifest = {
        "experiment_id": config.experiment_id,
        "experiment_date": config.experiment_date,
        "created_at": datetime.now(UTC).isoformat(),
        "config_provenance": {
            "schema_version": 2,
            "defaults_path": _relative_path(config.defaults_path, config.project_root),
            "defaults_sha256": _optional_sha256(config.defaults_path),
            "experiment_config_path": _relative_path(experiment_config, config.project_root),
            "experiment_config_sha256": _optional_sha256(experiment_config),
            "channels_path": _relative_path(config.channels_path, config.project_root),
            "channels_sha256": _optional_sha256(config.channels_path),
            "resolved_config_sha256": resolved_config_sha256(config),
        },
        "resolved_config": resolved_config_mapping(config),
        "git_commit": _git_commit(config.project_root),
        "prepare_summary": prepare_summary,
        "outputs": {
            "prepared_data": "prepared_data.parquet",
            "processed_data": "processed_data.parquet",
            "cycle_summary": "cycle_summary.csv",
            "candidate_channel_evidence": "candidate_channel_evidence.csv",
        },
        "output_row_counts": {
            "prepared_data": len(prepared),
            "processed_data": len(processed),
            "cycle_summary": len(cycle_summary),
            "candidate_channel_evidence": len(evidence),
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


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def optional_sha256(path: Path | None) -> str | None:
    """Return a file hash when a source file is available."""
    return None if path is None else _optional_sha256(path)


def source_file_metadata(path: Path, root: Path) -> dict[str, Any]:
    """Return stable metadata for one raw source file."""
    return {
        "relative_path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


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
        return str(value.isoformat())
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _serializable_settings(settings: Any) -> Any:
    return asdict(cast(Any, settings)) if is_dataclass(settings) else settings
