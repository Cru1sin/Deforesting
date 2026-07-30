"""Strict configuration loading for the reusable dataset stages."""

from __future__ import annotations

from ipaddress import ip_address
from pathlib import Path
from typing import Any

import yaml

from .schemas import AnalysisOptions, AppConfig, DatasetPaths, PrepareOptions, ProcessOptions


def load_camera_mapping(path: Path) -> tuple[dict[str, str], str]:
    """Load one YAML camera map and validate its IP-to-role contract."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    roles, unknown = _camera_roles(loaded.get("camera_roles"), loaded.get("unknown_role"))
    return roles, unknown


def load_date_camera_mapping(
    raw_dir: Path,
    *,
    fallback_roles: dict[str, str],
    fallback_unknown_role: str,
) -> tuple[dict[str, str], str]:
    """Prefer the date-local map and retain the old config as a fallback.

    ``raw_dir`` identifies one experiment date. Keeping the fallback here makes
    old data folders readable while ensuring a present ``IPlocation.yaml`` is
    always authoritative for that date.
    """
    local_mapping = raw_dir / "IPlocation.yaml"
    if not local_mapping.is_file():
        return dict(fallback_roles), fallback_unknown_role
    return load_camera_mapping(local_mapping)


def load_app_config(path: Path) -> AppConfig:
    """Load one stage configuration and reject unknown sections or keys."""
    config_path = path.resolve()
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("application config must be a YAML mapping")
    _require_keys(loaded, {"project", "paths", "prepare", "process", "analysis"}, "top level")
    root = config_path.parents[1]
    project = _mapping(loaded["project"], "project")
    paths = _mapping(loaded["paths"], "paths")
    prepare = _mapping(loaded["prepare"], "prepare")
    process = _mapping(loaded["process"], "process")
    analysis = _mapping(loaded["analysis"], "analysis")
    date = str(project.get("date", "")).strip()
    if not date.isdigit() or len(date) != 4:
        raise ValueError("project.date must be a four-digit MMDD token")
    experiment_id = str(project.get("experiment_id", f"exp_{date}"))
    raw_dir = _path(root, paths.get("raw_dir", f"data/{date}"))
    output_dir = _path(root, paths.get("output_dir", f"outputs/{date}"))
    registry = _path(root, paths.get("registry", "configs/feature_registry.yaml"))
    dataset_paths = DatasetPaths(
        project_root=root,
        raw_dir=raw_dir,
        output_dir=output_dir,
        registry=registry,
        prepared_data=output_dir / "prepared_data.parquet",
        processed_data=output_dir / "processed_data.parquet",
        cycle_summary=output_dir / "cycle_summary.csv",
        correlation_results=output_dir / "correlation_results.csv",
        state_dir=output_dir / ".pipeline",
    )
    _require_keys(
        prepare,
        {
            "timestamp_column",
            "duplicate_policy",
            "heating_mode_value",
            "images",
            "image_tolerance_seconds",
            "multiview_tolerance_milliseconds",
            "camera_roles",
            "unknown_camera_role",
            "cycles",
            "cycle_validation",
            "gap_warning_factor",
        },
        "prepare",
    )
    camera_roles, unknown_role = _camera_roles(
        prepare["camera_roles"], prepare["unknown_camera_role"]
    )
    cycle_settings = _mapping(prepare["cycles"], "prepare.cycles")
    cycle_validation = _mapping(prepare["cycle_validation"], "prepare.cycle_validation")
    image_settings = _mapping(prepare["images"], "prepare.images")
    _require_keys(image_settings, {"required"}, "prepare.images")
    prepare_options = PrepareOptions(
        timestamp_column=str(prepare["timestamp_column"]),
        duplicate_policy=str(prepare["duplicate_policy"]),
        heating_mode_value=int(prepare["heating_mode_value"]),
        images_required=bool(image_settings["required"]),
        image_tolerance_seconds=_positive_float(
            prepare["image_tolerance_seconds"], "prepare.image_tolerance_seconds"
        ),
        multiview_tolerance_milliseconds=_positive_float(
            prepare["multiview_tolerance_milliseconds"],
            "prepare.multiview_tolerance_milliseconds",
        ),
        camera_roles=camera_roles,
        unknown_camera_role=unknown_role,
        cycle_settings=cycle_settings,
        cycle_validation=cycle_validation,
        gap_warning_factor=_positive_float(
            prepare["gap_warning_factor"], "prepare.gap_warning_factor"
        ),
    )
    _require_keys(
        process,
        {
            "missing",
            "baseline",
            "resample_interval_seconds",
            "windows_minutes",
            "minimum_coverage",
        },
        "process",
    )
    missing = _mapping(process["missing"], "process.missing")
    baseline = _mapping(process["baseline"], "process.baseline")
    process_options = ProcessOptions(
        continuous_max_gap_seconds=_positive_float(
            missing.get("continuous_max_gap_seconds", 60),
            "process.missing.continuous_max_gap_seconds",
        ),
        control_max_gap_seconds=_positive_float(
            missing.get("control_max_gap_seconds", 30),
            "process.missing.control_max_gap_seconds",
        ),
        resample_interval_seconds=int(
            _positive_float(
                process["resample_interval_seconds"], "process.resample_interval_seconds"
            )
        ),
        windows_minutes=_positive_int_list(process["windows_minutes"], "process.windows_minutes"),
        minimum_coverage=_fraction(process["minimum_coverage"], "process.minimum_coverage"),
        baseline_settings=baseline,
    )
    _require_keys(
        analysis,
        {"task", "targets", "methods", "lags_minutes", "minimum_cycles", "save_figures"},
        "analysis",
    )
    analysis_options = AnalysisOptions(
        task=str(analysis["task"]),
        targets=_string_list(analysis["targets"], "analysis.targets"),
        methods=_string_list(analysis["methods"], "analysis.methods"),
        lags_minutes=_nonnegative_int_list(analysis["lags_minutes"], "analysis.lags_minutes"),
        minimum_cycles=int(
            _positive_float(analysis["minimum_cycles"], "analysis.minimum_cycles")
        ),
        save_figures=bool(analysis["save_figures"]),
    )
    if not dataset_paths.registry.is_file():
        raise FileNotFoundError(f"feature registry does not exist: {dataset_paths.registry}")
    return AppConfig(
        date,
        experiment_id,
        dataset_paths,
        prepare_options,
        process_options,
        analysis_options,
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return {str(key): item for key, item in value.items()}


def _require_keys(mapping: dict[str, Any], expected: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - expected)
    missing = sorted(expected - set(mapping))
    if unknown:
        raise ValueError(f"unknown {name} config keys: {unknown}")
    if missing:
        raise ValueError(f"missing {name} config keys: {missing}")


def _path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _camera_roles(value: Any, unknown: Any) -> tuple[dict[str, str], str]:
    roles = _mapping(value, "camera_roles")
    if not isinstance(unknown, str) or not unknown.strip():
        raise ValueError("unknown_camera_role must be non-empty")
    result: dict[str, str] = {}
    for address, role in roles.items():
        try:
            parsed = ip_address(address)
        except ValueError as error:
            raise ValueError(f"invalid camera IP: {address}") from error
        if parsed.version != 4 or not isinstance(role, str) or not role.strip():
            raise ValueError(f"invalid camera role for {address}")
        result[address] = role
    if not result:
        raise ValueError("camera_roles must not be empty")
    return result, unknown.strip()


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _fraction(value: Any, name: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return number


def _positive_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value or any(int(item) <= 0 for item in value):
        raise ValueError(f"{name} must be a non-empty list of positive integers")
    result = [int(item) for item in value]
    if result != sorted(set(result)):
        raise ValueError(f"{name} must be unique and increasing")
    return result


def _nonnegative_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value or any(int(item) < 0 for item in value):
        raise ValueError(f"{name} must be a non-empty list of non-negative integers")
    result = [int(item) for item in value]
    if result != sorted(set(result)):
        raise ValueError(f"{name} must be unique and increasing")
    return result


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not str(item).strip() for item in value):
        raise ValueError(f"{name} must be a non-empty list of strings")
    return [str(item) for item in value]
