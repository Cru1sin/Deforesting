"""Small typed configuration contract for the research pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CycleSettings:
    """Cycle segmentation thresholds expressed in physical seconds."""

    defrost_channel: str = "defrost_active"
    maximum_state_gap_seconds: float = 5.0
    debounce_seconds: float = 20.0
    minimum_defrost_seconds: float = 60.0
    maximum_defrost_seconds: float = 1200.0
    minimum_heating_seconds: float = 1800.0
    maximum_heating_seconds: float = 21600.0
    stable_heating_seconds: float = 180.0
    operating_mode_channel: str = ""
    required_operating_mode: str = "3"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CycleSettings:
        mapping = _mapping(values, "cycles")
        _validate_dataclass_keys(mapping, cls, "cycles")
        result = cls(
            defrost_channel=str(mapping.get("defrost_channel", cls.defrost_channel)),
            maximum_state_gap_seconds=float(mapping.get("maximum_state_gap_seconds", 5)),
            debounce_seconds=float(mapping.get("debounce_seconds", 20)),
            minimum_defrost_seconds=float(mapping.get("minimum_defrost_seconds", 60)),
            maximum_defrost_seconds=float(mapping.get("maximum_defrost_seconds", 1200)),
            minimum_heating_seconds=float(mapping.get("minimum_heating_seconds", 1800)),
            maximum_heating_seconds=float(mapping.get("maximum_heating_seconds", 21600)),
            stable_heating_seconds=float(mapping.get("stable_heating_seconds", 180)),
            operating_mode_channel=str(mapping.get("operating_mode_channel", "")),
            required_operating_mode=str(mapping.get("required_operating_mode", "3")),
        )
        _validate_nonnegative("maximum_state_gap_seconds", result.maximum_state_gap_seconds)
        _validate_positive("debounce_seconds", result.debounce_seconds)
        _validate_positive("minimum_defrost_seconds", result.minimum_defrost_seconds)
        _validate_positive("maximum_defrost_seconds", result.maximum_defrost_seconds)
        _validate_positive("minimum_heating_seconds", result.minimum_heating_seconds)
        _validate_positive("maximum_heating_seconds", result.maximum_heating_seconds)
        _validate_nonnegative("stable_heating_seconds", result.stable_heating_seconds)
        if result.minimum_defrost_seconds > result.maximum_defrost_seconds:
            raise ValueError("minimum_defrost_seconds must not exceed maximum_defrost_seconds")
        if result.minimum_heating_seconds > result.maximum_heating_seconds:
            raise ValueError("minimum_heating_seconds must not exceed maximum_heating_seconds")
        return result


@dataclass(frozen=True)
class BaselineSettings:
    """Rules for one common, cycle-local baseline window."""

    stage: str = "frost_development"
    search_start_minutes: int = 0
    search_end_minutes: int = 20
    window_minutes: int = 5
    window_step_minutes: int = 1
    minimum_observed_coverage: float = 0.8
    required_anchor_channels: tuple[str, ...] = (
        "ambient_temperature",
        "water_in_temperature",
        "water_out_temperature",
        "compressor_frequency",
    )
    anchor_maximum_std: dict[str, float] = field(
        default_factory=lambda: {
            "ambient_temperature": 1.0,
            "water_in_temperature": 1.0,
            "water_out_temperature": 1.0,
            "compressor_frequency": 5.0,
        }
    )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> BaselineSettings:
        mapping = _mapping(values, "baseline")
        _validate_dataclass_keys(mapping, cls, "baseline")
        anchors = mapping.get("required_anchor_channels", cls().required_anchor_channels)
        maximum_std = mapping.get("anchor_maximum_std", cls().anchor_maximum_std)
        result = cls(
            stage=str(mapping.get("stage", "frost_development")),
            search_start_minutes=int(mapping.get("search_start_minutes", 0)),
            search_end_minutes=int(mapping.get("search_end_minutes", 20)),
            window_minutes=int(mapping.get("window_minutes", 5)),
            window_step_minutes=int(mapping.get("window_step_minutes", 1)),
            minimum_observed_coverage=float(mapping.get("minimum_observed_coverage", 0.8)),
            required_anchor_channels=tuple(str(value) for value in anchors),
            anchor_maximum_std={str(key): float(value) for key, value in maximum_std.items()},
        )
        if result.stage != "frost_development":
            raise ValueError("baseline stage must be frost_development")
        if result.search_start_minutes < 0:
            raise ValueError("baseline search_start_minutes must be nonnegative")
        if result.search_end_minutes <= result.search_start_minutes:
            raise ValueError("baseline search_end_minutes must be later than search_start_minutes")
        if result.window_minutes <= 0 or result.window_step_minutes <= 0:
            raise ValueError("baseline window and step must be positive")
        if result.window_minutes > result.search_end_minutes - result.search_start_minutes:
            raise ValueError("baseline window_minutes must fit within the search range")
        _validate_fraction("minimum_observed_coverage", result.minimum_observed_coverage)
        if any(value < 0 for value in result.anchor_maximum_std.values()):
            raise ValueError("anchor_maximum_std values must be nonnegative")
        return result


@dataclass(frozen=True)
class ProcessSettings:
    """Resampling, imputation, baseline, and dynamic-feature settings."""

    resample_interval_seconds: int = 10
    minimum_continuous_bucket_coverage: float = 0.8
    continuous_max_gap_seconds: float = 60.0
    control_max_gap_seconds: float = 30.0
    baseline: BaselineSettings = field(default_factory=BaselineSettings)
    feature_windows_minutes: tuple[int, ...] = (5, 10, 30)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ProcessSettings:
        mapping = _mapping(values, "process")
        _validate_keys(
            mapping,
            {
                "resample_interval_seconds",
                "minimum_continuous_bucket_coverage",
                "continuous_max_gap_seconds",
                "control_max_gap_seconds",
                "baseline",
                "features",
            },
            "process",
        )
        baseline_values = mapping.get("baseline", {})
        feature_values = mapping.get("features", {})
        if not isinstance(baseline_values, Mapping) or not isinstance(feature_values, Mapping):
            raise ValueError("process.baseline and process.features must be mappings")
        _validate_keys(feature_values, {"windows_minutes"}, "process.features")
        windows = tuple(int(value) for value in feature_values.get("windows_minutes", [5, 10, 30]))
        result = cls(
            resample_interval_seconds=int(mapping.get("resample_interval_seconds", 10)),
            minimum_continuous_bucket_coverage=float(
                mapping.get("minimum_continuous_bucket_coverage", 0.8)
            ),
            continuous_max_gap_seconds=float(mapping.get("continuous_max_gap_seconds", 60)),
            control_max_gap_seconds=float(mapping.get("control_max_gap_seconds", 30)),
            baseline=BaselineSettings.from_mapping(baseline_values),
            feature_windows_minutes=windows,
        )
        if result.resample_interval_seconds <= 0:
            raise ValueError("resample_interval_seconds must be positive")
        _validate_fraction(
            "minimum_continuous_bucket_coverage", result.minimum_continuous_bucket_coverage
        )
        _validate_nonnegative("continuous_max_gap_seconds", result.continuous_max_gap_seconds)
        _validate_nonnegative("control_max_gap_seconds", result.control_max_gap_seconds)
        if not result.feature_windows_minutes or any(
            value <= 0 for value in result.feature_windows_minutes
        ):
            raise ValueError("feature windows must be positive")
        return result


@dataclass(frozen=True)
class AnalysisSettings:
    """Transparent candidate-evidence thresholds."""

    performance_target: str = "heating_capacity__baseline_residual"
    future_horizon_minutes: int = 10
    minimum_valid_cycles: int = 3
    minimum_trend_effect: float = 0.3
    minimum_direction_consistency: float = 0.7
    minimum_points_per_cycle: int = 6

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AnalysisSettings:
        mapping = _mapping(values, "analysis")
        _validate_dataclass_keys(mapping, cls, "analysis")
        result = cls(
            performance_target=str(
                mapping.get("performance_target", "heating_capacity__baseline_residual")
            ),
            future_horizon_minutes=int(mapping.get("future_horizon_minutes", 10)),
            minimum_valid_cycles=int(mapping.get("minimum_valid_cycles", 3)),
            minimum_trend_effect=float(mapping.get("minimum_trend_effect", 0.3)),
            minimum_direction_consistency=float(
                mapping.get("minimum_direction_consistency", 0.7)
            ),
            minimum_points_per_cycle=int(mapping.get("minimum_points_per_cycle", 6)),
        )
        if result.future_horizon_minutes <= 0:
            raise ValueError("future_horizon_minutes must be positive")
        if result.minimum_valid_cycles <= 0 or result.minimum_points_per_cycle < 2:
            raise ValueError("analysis minimum counts are too small")
        _validate_fraction("minimum_direction_consistency", result.minimum_direction_consistency)
        if result.minimum_trend_effect < 0 or result.minimum_trend_effect > 1:
            raise ValueError("minimum_trend_effect must be within [0, 1]")
        return result


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
    timestamp_column: str
    expected_sensor_interval_seconds: int
    image_match_tolerance_seconds: float
    edf_pair_tolerance_seconds: float
    cycles: CycleSettings
    process: ProcessSettings
    analysis: AnalysisSettings
    config_path: Path | None = None
    defaults_path: Path | None = None
    camera_roles: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw_cycles: Any = self.cycles
        raw_process: Any = self.process
        raw_analysis: Any = self.analysis
        if isinstance(raw_cycles, Mapping):
            object.__setattr__(self, "cycles", CycleSettings.from_mapping(raw_cycles))
        if isinstance(raw_process, Mapping):
            object.__setattr__(self, "process", ProcessSettings.from_mapping(raw_process))
        if isinstance(raw_analysis, Mapping):
            object.__setattr__(self, "analysis", AnalysisSettings.from_mapping(raw_analysis))
        if self.process.resample_interval_seconds % self.expected_sensor_interval_seconds != 0:
            raise ValueError(
                "resample_interval_seconds must be divisible by "
                "expected_sensor_interval_seconds"
            )
        _validate_positive(
            "edf_pair_tolerance_seconds",
            self.edf_pair_tolerance_seconds,
        )


def load_config(path: Path) -> Config:
    """Load one schema-v2 date file and resolve its shared defaults."""
    config_path = path.resolve()
    loaded = _load_yaml_mapping(config_path, "experiment config")
    _validate_keys(
        loaded,
        {
            "schema_version",
            "defaults_path",
            "experiment_id",
            "experiment_date",
            "input_dir",
            "expected_sensor_interval_seconds",
            "camera_roles",
            "overrides",
        },
        "experiment config",
    )
    if loaded.get("schema_version") != 2:
        raise ValueError("only config schema_version 2 is supported")
    required = {
        "schema_version",
        "defaults_path",
        "experiment_id",
        "experiment_date",
        "input_dir",
        "expected_sensor_interval_seconds",
        "camera_roles",
    }
    missing = sorted(required - set(loaded))
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    defaults_path = (config_path.parent / str(loaded["defaults_path"])).resolve()
    defaults = _load_yaml_mapping(defaults_path, "defaults")
    _validate_keys(
        defaults,
        {
            "channels_path",
            "input_format",
            "image_match_tolerance_seconds",
            "cycles",
            "process",
            "analysis",
        },
        "defaults",
    )
    overrides = loaded.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("overrides must be a YAML mapping")
    resolved = _merge_existing_keys(defaults, overrides)
    input_format = _mapping(resolved["input_format"], "input_format")
    _validate_keys(
        input_format,
        {"sensor_globs", "image_extensions", "timestamp_column", "edf"},
        "input_format",
    )
    if "edf" not in input_format:
        raise ValueError("input_format missing keys: ['edf']")
    edf_input = _mapping(input_format["edf"], "input_format.edf")
    _validate_keys(
        edf_input,
        {"pair_tolerance_seconds"},
        "input_format.edf",
    )
    if "pair_tolerance_seconds" not in edf_input:
        raise ValueError("input_format.edf missing keys: ['pair_tolerance_seconds']")
    edf_pair_tolerance = float(edf_input["pair_tolerance_seconds"])
    experiment_date = str(loaded["experiment_date"])
    if not _is_iso_date(experiment_date):
        raise ValueError("experiment_date must use ISO YYYY-MM-DD format")
    project_root = _find_project_root(config_path)
    process = ProcessSettings.from_mapping(_mapping(resolved["process"], "process"))
    analysis = AnalysisSettings.from_mapping(_mapping(resolved["analysis"], "analysis"))
    if analysis.future_horizon_minutes * 60 % process.resample_interval_seconds != 0:
        raise ValueError("future_horizon_minutes must align with the resample interval")
    expected_interval = int(loaded["expected_sensor_interval_seconds"])
    image_tolerance = float(resolved["image_match_tolerance_seconds"])
    _validate_positive("expected_sensor_interval_seconds", expected_interval)
    if process.resample_interval_seconds % expected_interval != 0:
        raise ValueError(
            "resample_interval_seconds must be divisible by expected_sensor_interval_seconds"
        )
    _validate_nonnegative("image_match_tolerance_seconds", image_tolerance)
    _validate_positive("edf_pair_tolerance_seconds", edf_pair_tolerance)
    if not str(input_format["timestamp_column"]).strip():
        raise ValueError("timestamp_column must not be empty")
    camera_roles = _camera_roles(loaded["camera_roles"])
    return Config(
        project_root=project_root,
        experiment_id=str(loaded["experiment_id"]),
        experiment_date=experiment_date,
        input_dir=_resolve_path(project_root, loaded["input_dir"]),
        channels_path=_resolve_path(defaults_path.parent, resolved["channels_path"]),
        sensor_globs=_tuple_strings(input_format["sensor_globs"], "sensor_globs"),
        image_extensions=_image_extensions(input_format["image_extensions"]),
        timestamp_column=str(input_format["timestamp_column"]),
        expected_sensor_interval_seconds=expected_interval,
        image_match_tolerance_seconds=image_tolerance,
        edf_pair_tolerance_seconds=edf_pair_tolerance,
        cycles=CycleSettings.from_mapping(_mapping(resolved["cycles"], "cycles")),
        process=process,
        analysis=analysis,
        config_path=config_path,
        defaults_path=defaults_path,
        camera_roles=camera_roles,
    )


def resolved_config_mapping(config: Config) -> dict[str, Any]:
    """Return the effective configuration without source-file provenance."""
    return {
        "experiment_id": config.experiment_id,
        "experiment_date": config.experiment_date,
        "input_dir": _relative_path(config.input_dir, config.project_root),
        "expected_sensor_interval_seconds": config.expected_sensor_interval_seconds,
        "camera_roles": dict(config.camera_roles),
        "input_format": {
            "sensor_globs": list(config.sensor_globs),
            "image_extensions": list(config.image_extensions),
            "timestamp_column": config.timestamp_column,
            "edf": {
                "pair_tolerance_seconds": config.edf_pair_tolerance_seconds,
            },
        },
        "image_match_tolerance_seconds": config.image_match_tolerance_seconds,
        "cycles": asdict(config.cycles),
        "process": {
            "resample_interval_seconds": config.process.resample_interval_seconds,
            "minimum_continuous_bucket_coverage": (
                config.process.minimum_continuous_bucket_coverage
            ),
            "continuous_max_gap_seconds": config.process.continuous_max_gap_seconds,
            "control_max_gap_seconds": config.process.control_max_gap_seconds,
            "baseline": asdict(config.process.baseline),
            "features": {"windows_minutes": list(config.process.feature_windows_minutes)},
        },
        "analysis": asdict(config.analysis),
    }


def resolved_config_sha256(config: Config) -> str:
    """Hash the effective configuration with stable JSON serialization."""
    payload = json.dumps(
        resolved_config_mapping(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _find_project_root(config_path: Path) -> Path:
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("could not find project root containing pyproject.toml")


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_yaml_mapping(path: Path, name: str) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return {str(key): value for key, value in loaded.items()}


def _validate_keys(values: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unknown keys: {unknown}")


def _validate_dataclass_keys(
    values: Mapping[str, Any], cls: type[Any], name: str
) -> None:
    _validate_keys(values, {item.name for item in fields(cls)}, name)


def _merge_existing_keys(
    base: Mapping[str, Any], overrides: Mapping[str, Any], prefix: str = ""
) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if key not in result:
            raise ValueError(f"unknown override key: {name}")
        existing = result[key]
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = _merge_existing_keys(existing, value, name)
        elif isinstance(existing, Mapping) != isinstance(value, Mapping):
            raise ValueError(f"override mapping shape does not match: {name}")
        else:
            result[key] = copy.deepcopy(value)
    return result


def _camera_roles(value: Any) -> dict[str, str]:
    roles = _mapping(value, "camera_roles")
    result = {str(camera): str(role) for camera, role in roles.items()}
    if any(not camera.strip() or not role.strip() for camera, role in result.items()):
        raise ValueError("camera_roles contains an empty camera ID or role")
    if len(result.values()) != len(set(result.values())):
        raise ValueError("two camera IDs cannot map to the same role")
    return result


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


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
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return {str(key): item for key, item in value.items()}


def _is_iso_date(value: str) -> bool:
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _validate_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_fraction(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be within [0, 1]")
