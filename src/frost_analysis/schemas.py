"""Typed contracts shared by the prepare, process, and task layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetPaths:
    project_root: Path
    raw_dir: Path
    output_dir: Path
    registry: Path
    prepared_data: Path
    processed_data: Path
    cycle_summary: Path
    correlation_results: Path
    state_dir: Path


@dataclass(frozen=True)
class PrepareOptions:
    timestamp_column: str
    duplicate_policy: str
    heating_mode_value: int
    images_required: bool
    image_tolerance_seconds: float
    multiview_tolerance_milliseconds: float
    camera_roles: dict[str, str]
    unknown_camera_role: str
    cycle_settings: dict[str, Any]
    cycle_validation: dict[str, Any]
    gap_warning_factor: float


@dataclass(frozen=True)
class ProcessOptions:
    continuous_max_gap_seconds: float
    control_max_gap_seconds: float
    resample_interval_seconds: int
    windows_minutes: list[int]
    minimum_coverage: float
    baseline_settings: dict[str, Any]


@dataclass(frozen=True)
class AnalysisOptions:
    task: str
    targets: list[str]
    methods: list[str]
    lags_minutes: list[int]
    minimum_cycles: int
    save_figures: bool


@dataclass(frozen=True)
class AppConfig:
    date: str
    experiment_id: str
    paths: DatasetPaths
    prepare: PrepareOptions
    process: ProcessOptions
    analysis: AnalysisOptions
