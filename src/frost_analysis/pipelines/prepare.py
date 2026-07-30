"""
Pipeline 1: organize raw sensors and RGB records without numerical processing.
数据准备：整理和清点目录内的数据。
把原始传感器文件和多机位图片，整理成一张字段统一、时间有序、带循环标签、带图片路径的数据表，但不进行插值、重采样、无霜基准和特征工程。
主要生成
prepared_data.parquet: 供 Pipeline 2 继续加工
cycle_summary.csv: 让人快速判断每个循环是否可用
.pipeline/prepare.json: 记录本次 prepare 的运行状态
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from ..config import load_date_camera_mapping
from ..core.artifacts import write_dataframe  # 保存主表
from ..data.alignment import (
    attach_image_paths,
    build_multiview,
    match_images_to_sensors,
)
from ..data.cycles import (
    CycleValidationResult,
    build_cycle_summary,
    normalize_cycle_status,
    segment_cycles,
    validate_cycles,
)
from ..data.images import build_image_manifest  # 扫描原始图片，建立图片记录表
from ..data.inventory import inventory_directory  # 扫描原始目录，识别文件和字段
from ..data.registry import apply_feature_registry, load_feature_registry
from ..data.sensors import preprocess_directory  # 加载并拼接传感器文件
from ..schemas import AppConfig  # 阶段配置和路径合同


@dataclass(frozen=True)
class PrepareResult:
    """In-memory outputs passed from prepare validation to publication."""

    prepared_data: pd.DataFrame  # 时间点级标准传感器表，附带紧凑循环和图片字段。
    cycle_summary: pd.DataFrame  # 循环级质量表，保存完整度和详细原因。
    warnings: tuple[str, ...]  # 不阻止发布、但需要人工关注的问题。
    metrics: dict[str, int]  # 本次输入清点和图片分组的计数指标。


def prepare_dataset(config: AppConfig) -> Path:
    """Build, validate, and publish the two prepare-stage artifacts."""
    result = build_prepared_dataset(config)  # 构建内存结果
    validate_prepare_result(result, config)  # 发布前检查合同
    return publish_prepare_result(result, config)  # 原子发布文件


def validate_prepare_result(result: PrepareResult, config: AppConfig) -> None:
    """Reject malformed prepare output before any public file is replaced."""
    prepared = result.prepared_data  # 时间点级输出。
    required_prepared = {
        "timestamp",
        "cycle_id",
        "cycle_stage",
        "cycle_status",
        "cycle_elapsed_seconds",
        "cycle_progress",
    }
    missing_prepared = sorted(required_prepared - set(prepared.columns))
    if missing_prepared:
        raise ValueError(f"prepared data missing required columns: {missing_prepared}")
    # Prepare may carry missing values, but it must never carry reconstructed values.
    interpolation_columns = [
        str(column) for column in prepared.columns if str(column).endswith("__interpolated")
    ]
    if interpolation_columns:
        raise RuntimeError(
            "Prepare stage produced interpolated columns: "
            + ", ".join(interpolation_columns)
        )
    timestamps = pd.to_datetime(prepared["timestamp"], errors="raise")
    if not timestamps.is_monotonic_increasing or not timestamps.is_unique:
        raise ValueError("prepared timestamps must be sorted and unique")
    if "cycle_status_reason" in prepared:
        raise ValueError("cycle_status_reason belongs only in cycle_summary")

    cycle_summary = result.cycle_summary  # 循环级输出，原因和完整度只在这里保存。
    required_summary = {
        "cycle_id",
        "cycle_status",
        "cycle_status_reason",
        "sensor_coverage_fraction",
        "rgb_coverage_fraction",
        "multimodal_coverage_fraction",
    }
    missing_summary = sorted(required_summary - set(cycle_summary.columns))
    if missing_summary:
        raise ValueError(f"cycle summary missing required columns: {missing_summary}")
    allowed_statuses = {"valid", "incomplete", "invalid"}
    observed_statuses = set(cycle_summary["cycle_status"].dropna().astype(str))
    unexpected_statuses = sorted(observed_statuses - allowed_statuses)
    if unexpected_statuses:
        raise ValueError(f"cycle summary has unknown statuses: {unexpected_statuses}")
    for column in (
        "sensor_coverage_fraction",
        "rgb_coverage_fraction",
        "multimodal_coverage_fraction",
    ):
        coverage = pd.to_numeric(cycle_summary[column], errors="coerce").dropna()
        if not coverage.between(0.0, 1.0).all():
            raise ValueError(f"cycle summary coverage outside [0, 1]: {column}")
    if config.prepare.images_required and result.metrics.get("rgb_image_count", 0) == 0:
        raise RuntimeError("No RGB images were found")


def publish_prepare_result(result: PrepareResult, config: AppConfig) -> Path:
    """Write validated prepare artifacts through temporary-file replacement."""
    # The parquet and CSV writers replace sibling temporary files atomically.
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(result.prepared_data, config.paths.prepared_data)
    _write_csv_atomic(result.cycle_summary, config.paths.cycle_summary)
    _write_state(config, result)
    return config.paths.prepared_data


def build_prepared_dataset(config: AppConfig) -> PrepareResult:
    """Build prepared observations in the readable sensor-to-RGB pipeline order."""
    # Fail before inventorying when the configured date directory is absent.
    if not config.paths.raw_dir.is_dir():
        raise FileNotFoundError(f"raw data directory does not exist: {config.paths.raw_dir}")

    # file_inventory 记录有哪些文件；source_field_inventory 记录原始表有哪些字段。
    file_inventory, source_field_inventory = inventory_directory(config.paths.raw_dir)

    # 日期目录中的相机位置优先于旧配置，避免不同日期共用错误映射。
    camera_roles, unknown_camera_role = load_date_camera_mapping(
        config.paths.raw_dir,
        fallback_roles=config.prepare.camera_roles,
        fallback_unknown_role=config.prepare.unknown_camera_role,
    )
    camera_mapping_warning = (
        "camera_mapping_fallback"
        if not (config.paths.raw_dir / "IPlocation.yaml").is_file()
        else ""
    )

    # image_records 是逐张图片的记录表，不是图片内容本身。
    image_records = build_image_manifest(
        config.paths.raw_dir,
        file_inventory,
        experiment_id=config.experiment_id,
        camera_roles=camera_roles,
        unknown_role=unknown_camera_role,
    )
    image_requirement_warning = validate_image_requirement(
        image_records,
        required=config.prepare.images_required,
    )

    # multiview_index 按时间把不同相机图片分组，不保存图像像素。
    multiview_index = build_multiview(
        image_records,
        tolerance_ms=config.prepare.multiview_tolerance_milliseconds,
    )

    # sensor_load_result 已完成读取、拼接和时间解析，但没有数值插值。
    sensor_load_result = preprocess_directory(
        config.paths.raw_dir,
        short_gap_max_seconds=0,
        transition_guard_seconds=0,
    )

    # registry_specs 描述原始点位到标准字段的映射关系。
    registry_specs = load_feature_registry(config.paths.registry)
    registry_result = apply_feature_registry(
        sensor_load_result.frame,
        registry_specs,
        heating_mode_value=config.prepare.heating_mode_value,
    )

    # prepared_data 只保留标准字段，并把内部 sensor_time 统一为 timestamp。
    prepared_data = build_prepared_sensor_table(
        registry_result.frame,
        registry_result.metadata,
    )
    defrost_column = registry_specs["defrost_flag"].canonical_name
    cycle_input = prepared_data  # 循环函数只接收标准字段，不接收原始列名。
    cycle_segmentation = segment_cycles(
        cycle_input,
        defrost_column,
        dict(config.prepare.cycle_settings),
    )
    cycle_validation = validate_cycles(
        cycle_segmentation,
        config.prepare.cycle_validation,
    )
    validated_cycles = cycle_validation.cycles
    prepared_data = attach_cycle_fields(
        prepared_data,
        cycle_validation,
    )
    image_alignment = match_images_to_sensors(
        image_records,
        prepared_data[["timestamp"]],
        tolerance_s=config.prepare.image_tolerance_seconds,
    )
    prepared_data = attach_image_paths(prepared_data, image_alignment)
    cycle_summary = build_cycle_summary(
        validated_cycles,
        prepared_data,
        multiview_index,
        date=config.date,
        gap_warning_factor=float(
            config.prepare.cycle_validation.get(
                "gap_warning_factor", config.prepare.gap_warning_factor
            )
        ),
    )
    prepared_data = select_prepared_output_columns(
        prepared_data,
        registered_columns=[
            str(name)
            for name in registry_result.metadata.get(
                "canonical_name", pd.Series(dtype=str)
            ).tolist()
        ],
    )
    registry_warnings = [
        f"registry source unavailable: {spec.feature_id}:{spec.raw_source}"
        for spec in registry_specs.values()
        if spec.raw_source and spec.raw_source not in sensor_load_result.frame
    ]
    warnings = [
        *cycle_validation.warnings,
        *([camera_mapping_warning] if camera_mapping_warning else []),
        *registry_warnings,
    ]
    if image_requirement_warning:
        warnings.append(image_requirement_warning)
    metrics = summarize_prepare_metrics(
        source_field_inventory=source_field_inventory,
        image_records=image_records,
        multiview_index=multiview_index,
    )
    return PrepareResult(
        prepared_data=prepared_data,
        cycle_summary=cycle_summary,
        warnings=tuple(warnings),
        metrics=metrics,
    )


def validate_image_requirement(image_records: pd.DataFrame, *, required: bool) -> str:
    """Enforce the experiment's RGB policy and return a warning when optional."""
    if not image_records.empty:
        return ""
    if required:
        raise RuntimeError("No RGB images were found")
    return "no_images_found"


def select_prepared_output_columns(
    frame: pd.DataFrame,
    *,
    registered_columns: list[str],
) -> pd.DataFrame:
    """Select public fields explicitly and reject interpolation markers."""
    interpolated_columns = [
        str(column) for column in frame.columns if str(column).endswith("__interpolated")
    ]
    if interpolated_columns:
        raise RuntimeError(
            "Prepare stage produced interpolated columns: " + ", ".join(interpolated_columns)
        )
    cycle_columns = [
        "cycle_id",
        "cycle_stage",
        "cycle_status",
        "cycle_elapsed_seconds",
        "cycle_progress",
    ]
    # Keep registry fields first, then compact cycle labels, then stable image columns.
    requested_columns = [
        "timestamp",
        *registered_columns,
        "operating_mode",
        "is_heating",
        *cycle_columns,
    ]
    requested_columns.extend(
        column for column in frame.columns if str(column).startswith("image_")
    )
    output_columns = list(dict.fromkeys(column for column in requested_columns if column in frame))
    return frame.loc[:, output_columns].copy()


def summarize_prepare_metrics(
    *,
    source_field_inventory: pd.DataFrame,
    image_records: pd.DataFrame,
    multiview_index: pd.DataFrame,
) -> dict[str, int]:
    """Summarize input inventory and image grouping without inspecting pixels."""
    return {
        "raw_inventory_fields": int(len(source_field_inventory)),
        "raw_inventory_unique_fields": int(
            source_field_inventory.get("canonical_column", pd.Series(dtype=str)).nunique()
        ),
        "rgb_image_count": int(len(image_records)),
        "rgb_group_count": int(len(multiview_index)),
        "rgb_complete_group_count": int(
            multiview_index.get("all_cameras_present", pd.Series(dtype=bool)).sum()
        ),
    }


def build_prepared_sensor_table(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Select registered fields, parse timestamps, and return sorted data."""
    # metadata names are the only sensor fields allowed to cross the prepare boundary.
    names = [
        str(name)
        for name in metadata.get("canonical_name", pd.Series(dtype=str)).tolist()
    ]
    keep = ["sensor_time", *names, "operating_mode", "is_heating"]
    keep = list(dict.fromkeys(column for column in keep if column in frame))
    # Copy before parsing and sorting so callers retain the Registry result unchanged.
    result = frame.loc[:, keep].copy()
    result["sensor_time"] = pd.to_datetime(result["sensor_time"], errors="raise")
    result = result.sort_values("sensor_time", kind="stable").reset_index(drop=True)
    return result.rename(columns={"sensor_time": "timestamp"})


def attach_cycle_fields(
    prepared: pd.DataFrame,
    cycle_validation: CycleValidationResult,
) -> pd.DataFrame:
    """Merge compact cycle labels while keeping detailed reasons in the summary."""
    labeled = cycle_validation.frame
    labels = labeled[
        ["timestamp", "cycle_id", "cycle_quality", "stage", "cycle_time_s", "cycle_phase"]
    ].copy()
    labels = labels.rename(
        columns={
            "cycle_quality": "cycle_status_raw",
            "stage": "cycle_stage",
            "cycle_time_s": "cycle_elapsed_seconds",
            "cycle_phase": "cycle_progress",
        }
    )
    labels["cycle_status"] = labels["cycle_status_raw"].map(normalize_cycle_status)
    labels = labels.drop(columns=["cycle_status_raw"])
    result = prepared.merge(labels, on="timestamp", how="left", validate="one_to_one")
    return result


def _write_state(config: AppConfig, result: PrepareResult) -> None:
    """Persist a compact, human-readable record of the published prepare run."""
    config.paths.state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "prepare",
        "date": config.date,
        "prepared_rows": len(result.prepared_data),
        "cycle_rows": len(result.cycle_summary),
        "metrics": result.metrics,
        "warnings": list(result.warnings),
        "prepared_data_path": str(config.paths.prepared_data),
        "cycle_summary_path": str(config.paths.cycle_summary),
        "config_fingerprint": _config_fingerprint(config),
        "registry_fingerprint": _file_fingerprint(config.paths.registry),
        "created_at": datetime.now(UTC).isoformat(),
    }
    state_path = config.paths.state_dir / "prepare.json"
    temporary_path = state_path.with_name(f".{state_path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(state_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write one CSV via a sibling temporary file so readers see whole files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _file_fingerprint(path: Path) -> str:
    """Hash one configuration file for state reproducibility."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_fingerprint(config: AppConfig) -> str:
    """Hash the loaded application contract without storing its full contents."""
    serialized = json.dumps(asdict(config), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
