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

import json
import re # 正则处理，将相机名称转换成安全的列名
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.artifacts import write_dataframe # 负责保存 DataFrame
from ..data.alignment import build_multiview, match_images_to_sensors # 把时间相近的多机位图片分为一组，给每张图片寻找最近的传感器时间点
from ..data.cycles import _sensor_gap_evidence, build_cycle_summary, segment_cycles # 检查单个循环中的传感器中断，划分结霜—除霜循环，汇总每个循环的数据质量
from ..data.images import build_image_manifest # 扫描原始图片，建立图片记录表
from ..data.inventory import inventory_directory # 扫描原始目录识别文件等信息
from ..data.registry import apply_feature_registry, load_feature_registry # 加载标准字段定义，把原始点位映射成标准变量
from ..data.sensors import preprocess_directory # 加载传感器文件，清洗数据
from ..schemas import AppConfig # 数据配置：路径、编号、映射、阈值等


@dataclass(frozen=True)
class PrepareResult: # 准备结果类，包含准备后的数据、循环总结、警告信息和指标
    prepared_data: pd.DataFrame
    cycle_summary: pd.DataFrame
    warnings: list[str]
    metrics: dict[str, int]


def prepare_dataset(config: AppConfig) -> Path: # 正式的外部入口
    """
    把“如何组织数据”和“如何写文件”分开
    创建 prepared_data.parquet 和规范化的循环总结，并保存到指定路径。
    """
    result = build_prepared_dataset(config) # 构建数据
    config.paths.output_dir.mkdir(parents=True, exist_ok=True) # 创建输出目录
    write_dataframe(result.prepared_data, config.paths.prepared_data) # 保存主数据
    result.cycle_summary.to_csv(config.paths.cycle_summary, index=False)  # 保存循环摘要
    _write_state(config, result) # 保存运行状态
    return config.paths.prepared_data # 返回主输出路径


def build_prepared_dataset(config: AppConfig) -> PrepareResult: # 核心的处理函数

    if not config.paths.raw_dir.is_dir(): # 确认原始数据目录存在
        raise FileNotFoundError(f"raw data directory does not exist: {config.paths.raw_dir}")

    # 盘点原始目录，需要知道哪些文件是图片，图片所在目录，原始数据的字段
    inventory, inventory_columns = inventory_directory(config.paths.raw_dir) 

    # 把文件夹中的所有图片整理成一张标准表，结构化记录
    image_manifest = build_image_manifest(
        config.paths.raw_dir,
        inventory,
        experiment_id=config.experiment_id,
        camera_roles=config.prepare.camera_roles,
        unknown_role=config.prepare.unknown_camera_role,
    )

    # 
    multiview = build_multiview(
        image_manifest,
        tolerance_ms=config.prepare.multiview_tolerance_milliseconds,
    )
    raw = preprocess_directory(
        config.paths.raw_dir,
        short_gap_max_seconds=0,
        transition_guard_seconds=0,
    )
    specs = load_feature_registry(config.paths.registry)
    registry = apply_feature_registry(raw.frame, specs)
    prepared = _standardize_schema(registry.frame, registry.metadata)
    internal = prepared.rename(columns={"timestamp": "sensor_time"})
    defrost = specs["defrost_flag"].canonical_name
    segmentation = segment_cycles(internal, defrost, dict(config.prepare.cycle_settings))
    labeled, audited, mode_warnings = _enforce_heating_mode(
        segmentation.frame, segmentation.cycles
    )
    labeled, audited, gap_warnings = _mark_long_gap_cycles(
        labeled,
        audited,
        nominal_seconds=_nominal_seconds(labeled),
        factor=config.prepare.gap_warning_factor,
    )
    prepared = _attach_cycle_fields(prepared, labeled, audited)
    alignment = match_images_to_sensors(
        image_manifest,
        prepared[["timestamp"]].rename(columns={"timestamp": "sensor_time"}),
        tolerance_s=config.prepare.image_tolerance_seconds,
    )
    prepared = _attach_image_columns(prepared, alignment)
    cycle_summary = build_cycle_summary(
        audited,
        prepared.rename(columns={"timestamp": "sensor_time"}),
        multiview,
        date=config.date,
        gap_warning_factor=config.prepare.gap_warning_factor,
    )
    prepared = _drop_preparation_artifacts(prepared)
    warnings = [
        *mode_warnings,
        *gap_warnings,
        *[
            f"registry source unavailable: {spec.feature_id}:{spec.raw_source}"
            for spec in specs.values()
            if spec.raw_source and spec.raw_source not in raw.frame
        ],
    ]
    if image_manifest.empty:
        warnings.append("no_images_found")
    metrics = {
        "raw_inventory_fields": int(len(inventory_columns)),
        "raw_inventory_unique_fields": int(
            inventory_columns.get("canonical_column", pd.Series(dtype=str)).nunique()
        ),
        "rgb_image_count": int(len(image_manifest)),
        "rgb_group_count": int(len(multiview)),
        "rgb_complete_group_count": int(
            multiview.get("all_cameras_present", pd.Series(dtype=bool)).sum()
        ),
    }
    return PrepareResult(prepared, cycle_summary, warnings, metrics)


def _standardize_schema(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    names = [
        str(name)
        for name in metadata.get("canonical_name", pd.Series(dtype=str)).tolist()
    ]
    keep = ["sensor_time", *names, "heating_mode"]
    keep = list(dict.fromkeys(column for column in keep if column in frame))
    result = frame.loc[:, keep].copy()
    result["sensor_time"] = pd.to_datetime(result["sensor_time"], errors="raise")
    result = result.sort_values("sensor_time", kind="stable").reset_index(drop=True)
    return result.rename(columns={"sensor_time": "timestamp"})


def _attach_cycle_fields(
    prepared: pd.DataFrame, labeled: pd.DataFrame, cycles: pd.DataFrame
) -> pd.DataFrame:
    labels = labeled[
        ["sensor_time", "cycle_id", "cycle_quality", "stage", "cycle_time_s", "cycle_phase"]
    ].copy()
    labels = labels.rename(
        columns={
            "sensor_time": "timestamp",
            "cycle_quality": "cycle_status_raw",
            "stage": "cycle_stage",
            "cycle_time_s": "cycle_elapsed_seconds",
            "cycle_phase": "cycle_progress",
        }
    )
    reasons = cycles.set_index("cycle_id")["exclusion_reason"].to_dict()
    labels["cycle_status"] = labels["cycle_status_raw"].map(_cycle_status)
    labels["cycle_status_reason"] = labels["cycle_id"].map(reasons).fillna("")
    labels = labels.drop(columns=["cycle_status_raw"])
    result = prepared.merge(labels, on="timestamp", how="left", validate="one_to_one")
    return result


def _attach_image_columns(prepared: pd.DataFrame, alignment: pd.DataFrame) -> pd.DataFrame:
    matched = alignment.loc[
        alignment.get("matched", pd.Series(False, index=alignment.index)).fillna(False).astype(bool)
    ].copy()
    if matched.empty:
        return prepared
    matched["camera_id"] = matched["camera_id"].astype(str)
    matched["abs_offset"] = pd.to_numeric(matched["time_delta_s"], errors="coerce").abs()
    matched = matched.sort_values("abs_offset", kind="stable").drop_duplicates(
        ["sensor_time", "camera_id"], keep="first"
    )
    pieces: list[pd.DataFrame] = []
    for camera_id, group in matched.groupby("camera_id", sort=True):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", str(camera_id)).strip("_").lower()
        piece = group[["sensor_time", "image_path", "time_delta_s"]].rename(
            columns={
                "image_path": f"image_{safe}_path",
                "time_delta_s": f"image_{safe}_offset_seconds",
            }
        )
        pieces.append(piece)
    result = prepared.copy()
    for piece in pieces:
        result = result.merge(
            piece.rename(columns={"sensor_time": "timestamp"}),
            on="timestamp",
            how="left",
            validate="one_to_one",
        )
    return result


def _drop_preparation_artifacts(frame: pd.DataFrame) -> pd.DataFrame:
    drop = [
        column
        for column in frame
        if str(column).endswith(("__raw", "__invalid", "__missing", "__interpolated"))
        or column in {"cycle_status_raw"}
    ]
    return frame.drop(columns=drop, errors="ignore")


def _cycle_status(quality: object) -> str:
    value = str(quality)
    if value == "complete":
        return "valid"
    if value == "contaminated":
        return "long_gap"
    if value in {"abnormal", "excluded"}:
        return "invalid_mode"
    return "incomplete"


def _enforce_heating_mode(
    frame: pd.DataFrame, cycles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if "heating_mode" not in frame or cycles.empty:
        return frame, cycles, []
    labeled = frame.copy()
    audited = cycles.copy()
    warnings: list[str] = []
    for index, cycle in audited.iterrows():
        if cycle.get("quality_flag") != "complete":
            continue
        mask = labeled["cycle_id"].eq(cycle["cycle_id"]) & labeled["stage"].isin(
            ["stable_clean", "frost_development"]
        )
        observed = labeled.loc[mask, "heating_mode"].astype("boolean").dropna()
        if not observed.empty and not observed.all():
            audited.loc[index, "quality_flag"] = "abnormal"
            audited.loc[index, "exclusion_reason"] = "nonheating_mode_inside_cycle"
            labeled.loc[labeled["cycle_id"].eq(cycle["cycle_id"]), "cycle_quality"] = "abnormal"
            warnings.append(f"{cycle['cycle_id']}:nonheating_mode_inside_cycle")
    return labeled, audited, warnings


def _mark_long_gap_cycles(
    frame: pd.DataFrame,
    cycles: pd.DataFrame,
    *,
    nominal_seconds: float,
    factor: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    labeled = frame.copy()
    audited = cycles.copy()
    warnings: list[str] = []
    limit = nominal_seconds * factor
    for index, row in audited.iterrows():
        gap = pd.to_numeric(pd.Series([row.get("maximum_gap_seconds")]), errors="coerce").iloc[0]
        cycle_id = str(row.get("cycle_id", ""))
        if pd.notna(row.get("heating_start")) and pd.notna(row.get("defrost_end")):
            channel_gap, _ = _sensor_gap_evidence(
                labeled,
                cycle_id,
                pd.Timestamp(row["heating_start"]),
                pd.Timestamp(row["defrost_end"]),
                nominal_seconds,
            )
            gap = max(float(gap) if pd.notna(gap) else 0.0, channel_gap)
            audited.loc[index, "maximum_gap_seconds"] = gap
        if row.get("quality_flag") == "complete" and pd.notna(gap) and float(gap) > limit:
            audited.loc[index, "quality_flag"] = "contaminated"
            old = str(row.get("exclusion_reason", ""))
            audited.loc[index, "exclusion_reason"] = ";".join(
                item for item in (old, "long_gap") if item
            )
            mask = labeled["cycle_id"].eq(cycle_id)
            labeled.loc[mask, "cycle_quality"] = "contaminated"
            labeled.loc[mask, "cycle_phase"] = np.nan
            warnings.append(f"{cycle_id}:long_gap:{float(gap):.1f}s>{limit:.1f}s")
    return labeled, audited, warnings


def _nominal_seconds(frame: pd.DataFrame) -> float:
    times = pd.to_datetime(frame.get("sensor_time", pd.Series(dtype=object)), errors="coerce")
    deltas = times.sort_values().diff().dt.total_seconds()
    positive = deltas[deltas.gt(0)]
    value = float(positive.median()) if not positive.empty else 1.0
    return value if np.isfinite(value) and value > 0 else 1.0


def _write_state(config: AppConfig, result: PrepareResult) -> None:
    config.paths.state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "prepare",
        "date": config.date,
        "prepared_rows": len(result.prepared_data),
        "cycle_rows": len(result.cycle_summary),
        "metrics": result.metrics,
        "warnings": result.warnings,
        "created_at": time.time(),
    }
    (config.paths.state_dir / "prepare.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
