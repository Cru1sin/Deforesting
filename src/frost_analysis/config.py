"""Strict configuration loading for the reusable dataset stages."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any

import yaml

from .schemas import AnalysisOptions, AppConfig, DatasetPaths, PrepareOptions, ProcessOptions


@dataclass(frozen=True)
class Config:
    """Small configuration contract used by the new flat pipeline."""

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


def load_config(path: Path) -> Config:
    """Load the new flat configuration without changing the legacy loader."""
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
    date = str(loaded["experiment_date"])
    if len(date) != 10 or date[4] != "-" or date[7] != "-":
        raise ValueError("experiment_date must use ISO YYYY-MM-DD format")
    sensor_globs = _tuple_strings(loaded["sensor_globs"], "sensor_globs")
    image_extensions = tuple(
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in _tuple_strings(loaded["image_extensions"], "image_extensions")
    )
    return Config(
        project_root=project_root,
        experiment_id=str(loaded["experiment_id"]),
        experiment_date=date,
        input_dir=_resolve_config_path(project_root, loaded["input_dir"]),
        channels_path=_resolve_config_path(project_root, loaded["channels_path"]),
        sensor_globs=sensor_globs,
        image_extensions=image_extensions,
        camera_mapping_file=str(loaded["camera_mapping_file"]),
        cycles=_mapping(loaded["cycles"], "cycles"),
        process=_mapping(loaded["process"], "process"),
        analysis=_mapping(loaded["analysis"], "analysis"),
    )


def _find_project_root(config_path: Path) -> Path:
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("could not find project root containing pyproject.toml")


def _resolve_config_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _tuple_strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not str(item).strip() for item in value):
        raise ValueError(f"{name} must be a non-empty list of strings")
    return tuple(str(item) for item in value)


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
            "duplicate_timestamps",
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
    duplicate_timestamps = _mapping(prepare["duplicate_timestamps"], "prepare.duplicate_timestamps")
    _require_keys(
        duplicate_timestamps,
        {"exact_duplicate", "conflicting_duplicate"},
        "prepare.duplicate_timestamps",
    )
    if str(duplicate_timestamps["exact_duplicate"]) != "drop":
        raise ValueError("prepare.duplicate_timestamps.exact_duplicate must be 'drop'")
    duplicate_conflict_policy = str(duplicate_timestamps["conflicting_duplicate"])
    if duplicate_conflict_policy not in {"warn_keep_stable", "error"}:
        raise ValueError(
            "prepare.duplicate_timestamps.conflicting_duplicate must be "
            "'warn_keep_stable' or 'error'"
        )
    cycle_settings = _mapping(prepare["cycles"], "prepare.cycles")
    cycle_validation = _mapping(prepare["cycle_validation"], "prepare.cycle_validation")
    image_settings = _mapping(prepare["images"], "prepare.images")
    _require_keys(image_settings, {"required"}, "prepare.images")
    prepare_options = PrepareOptions(
        timestamp_column=str(prepare["timestamp_column"]),
        duplicate_conflict_policy=duplicate_conflict_policy,
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
            "resample",
            "baseline",
            "features",
        },
        "process",
    )
    missing = _mapping(process["missing"], "process.missing")
    _require_keys(
        missing,
        {"group_columns", "continuous", "control", "protected", "audit"},
        "process.missing",
    )
    continuous = _mapping(missing["continuous"], "process.missing.continuous")
    control = _mapping(missing["control"], "process.missing.control")
    protected = _mapping(missing["protected"], "process.missing.protected")
    audit = _mapping(missing["audit"], "process.missing.audit")
    _require_keys(
        continuous,
        {"method", "maximum_bracketing_gap_seconds", "require_both_sides"},
        "process.missing.continuous",
    )
    _require_keys(
        control,
        {"method", "maximum_age_seconds"},
        "process.missing.control",
    )
    _require_keys(protected, {"method"}, "process.missing.protected")
    _require_keys(
        audit,
        {"keep_imputed_flag", "keep_imputation_method"},
        "process.missing.audit",
    )
    resample = _mapping(process["resample"], "process.resample")
    _require_keys(resample, {"interval_seconds"}, "process.resample")
    feature_settings = _mapping(process["features"], "process.features")
    _require_keys(
        feature_settings,
        {
            "windows_minutes",
            "minimum_observed_coverage",
            "minimum_available_coverage",
            "maximum_imputed_fraction",
            "maximum_raw_gap_seconds",
        },
        "process.features",
    )
    baseline = _mapping(process["baseline"], "process.baseline")
    process_options = ProcessOptions(
        continuous_max_gap_seconds=_positive_float(
            continuous["maximum_bracketing_gap_seconds"],
            "process.missing.continuous_max_gap_seconds",
        ),
        control_max_gap_seconds=_positive_float(
            control["maximum_age_seconds"],
            "process.missing.control_max_gap_seconds",
        ),
        resample_interval_seconds=int(
            _positive_float(
                resample["interval_seconds"], "process.resample.interval_seconds"
            )
        ),
        windows_minutes=_positive_int_list(
            feature_settings["windows_minutes"], "process.features.windows_minutes"
        ),
        minimum_coverage=_fraction(
            feature_settings["minimum_available_coverage"],
            "process.features.minimum_available_coverage",
        ),
        minimum_observed_coverage=_fraction(
            feature_settings["minimum_observed_coverage"],
            "process.features.minimum_observed_coverage",
        ),
        minimum_available_coverage=_fraction(
            feature_settings["minimum_available_coverage"],
            "process.features.minimum_available_coverage",
        ),
        maximum_imputed_fraction=_fraction(
            feature_settings["maximum_imputed_fraction"],
            "process.features.maximum_imputed_fraction",
        ),
        maximum_raw_gap_seconds=_positive_float(
            feature_settings["maximum_raw_gap_seconds"],
            "process.features.maximum_raw_gap_seconds",
        ),
        missing_settings=missing,
        baseline_settings=baseline,
    )
    _require_keys(
        analysis,
        {
            "task",
            "features",
            "targets",
            "methods",
            "lags_minutes",
            "minimum_cycles",
            "save_figures",
            "modalities",
        },
        "analysis",
    )
    modalities = _mapping(analysis["modalities"], "analysis.modalities")
    _require_keys(modalities, {"sensor", "rgb"}, "analysis.modalities")
    sensor_modality = _mapping(modalities["sensor"], "analysis.modalities.sensor")
    rgb_modality = _mapping(modalities["rgb"], "analysis.modalities.rgb")
    _require_keys(sensor_modality, {"required"}, "analysis.modalities.sensor")
    _require_keys(
        rgb_modality,
        {"required", "required_camera_roles"},
        "analysis.modalities.rgb",
    )
    required_camera_roles = rgb_modality["required_camera_roles"]
    if not isinstance(required_camera_roles, list) or any(
        not isinstance(role, str) or not role.strip() for role in required_camera_roles
    ):
        raise ValueError("analysis.modalities.rgb.required_camera_roles must be a list of strings")
    if bool(rgb_modality["required"]) and not required_camera_roles:
        raise ValueError(
            "analysis.modalities.rgb.required_camera_roles must not be empty when RGB is required"
        )
    analysis_options = AnalysisOptions(
        task=str(analysis["task"]),
        features=_string_list(analysis["features"], "analysis.features"),
        targets=_string_list(analysis["targets"], "analysis.targets"),
        methods=_string_list(analysis["methods"], "analysis.methods"),
        lags_minutes=_nonnegative_int_list(analysis["lags_minutes"], "analysis.lags_minutes"),
        minimum_cycles=int(
            _positive_float(analysis["minimum_cycles"], "analysis.minimum_cycles")
        ),
        save_figures=bool(analysis["save_figures"]),
        modalities={
            "sensor": dict(sensor_modality),
            "rgb": {
                **dict(rgb_modality),
                "required_camera_roles": [str(role).strip() for role in required_camera_roles],
            },
        },
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
