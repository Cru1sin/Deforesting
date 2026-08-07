"""Prepare raw sensor and image records without reconstructing measurements."""

# =============================================================================
# Prepare 阶段的职责边界
# =============================================================================
#
# 这个模块负责把“原始、分散、字段名不统一”的传感器文件和图片目录，
# 整理成一张按时间戳组织的 Prepared 表，并生成循环级摘要。
#
# 主流程可以概括为：
#
# 原始传感器文件
#     ↓
# 读取文本、解析时间戳
#     ↓
# 原始列名映射为标准通道名
#     ↓
# 解析数值并保留 missing / invalid / duplicate / conflict 标记
#     ↓
# 汇总所有有效时间戳，形成 Prepared 时间轴
#     ↓
# 调用 cycles.py 切分循环和阶段
#     ↓
# 调用 images.py 将图片一次性匹配到传感器时间戳
#     ↓
# 生成 Prepared 数据和 cycle summary
#
# Prepare 明确“不做”的事情：
# - 不重采样到 10 秒网格；
# - 不对缺失值插值或前向填充；
# - 不计算派生热力学量；
# - 不寻找 baseline；
# - 不计算动态特征；
# - 不筛选候选通道。
#
# 因此这里的原则是：
# “忠实整理原始观测并显式记录质量问题，而不是重构或美化测量值。”

from __future__ import annotations

# csv 用于自动识别文本传感器文件的分隔符。
import csv

# re 用于从文件名中提取“参数1、参数2……”中的参数组编号。
import re

# defaultdict 允许一个原始来源名称对应多个标准通道，
# 也允许同一个标准通道收集来自多个文件或列的数据片段。
from collections import defaultdict

# Mapping 表示字典式只读接口，兼容普通 dict 和其他 mapping 对象。
from collections.abc import Mapping

# Path 统一处理输入文件、图片路径和相对路径。
from pathlib import Path

# Any 用于尚未通过 channels 合同完全收敛类型的配置值。
from typing import Any

# NumPy 主要用于 NaN 和部分数组操作。
import numpy as np

# pandas 是本模块整理时间序列和表格数据的核心工具。
import pandas as pd

# match_nearest_one_to_one：将 EDF 融合观测按有序一对一规则对齐到主时间轴。
from .alignment import match_nearest_one_to_one

# Config：已经解析完成的一次实验配置。
from .config import Config

# label_cycles：根据除霜状态和循环阈值，为 Prepared 表添加 cycle_id、
# cycle_stage、cycle_progress 等标签，并返回循环摘要。
from .cycles import label_cycles

# match_images：把图片按时间戳和相机角色匹配到 Prepared 时间轴。
from .images import match_images

# discover_inputs：发现原始传感器文件和图片。
from .io import discover_inputs
from .sensors import read_edf_environment

# =============================================================================
# 1. Prepare 对外主入口
# =============================================================================


def prepare(
    config: Config, channels: Mapping[str, Mapping[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw observations, segment cycles, and attach one-shot image matches."""

    # 返回值由两部分组成：
    #
    # 1. prepared:
    #    时间戳级数据表。包含原始标准化通道、质量标记、循环标签和图片匹配。
    #
    # 2. cycle_summary:
    #    每个循环一行。包含循环边界、状态和 Prepare 阶段的覆盖质量指标。
    #
    # -------------------------------------------------------------------------
    # 第 1 步：发现输入文件
    # -------------------------------------------------------------------------

    # discover_inputs() 按 config 中的 sensor_globs、image_extensions 和目录结构，
    # 查找本次实验的原始传感器文件与图片文件。
    inputs = discover_inputs(config)

    # 传感器文件是 Prepare 的必要输入。
    # 图片可以缺失，但没有任何传感器文件时无法建立实验时间轴。
    if not inputs.sensor_files:
        raise ValueError(f"no sensor files found in {config.input_dir}")

    # 相机 ID → 物理角色映射，例如：
    # {"camera01": "front", "camera02": "top"}。
    # 下游列名将按 role 生成，而不是直接使用相机目录名。
    camera_roles = config.camera_roles

    # -------------------------------------------------------------------------
    # 第 2 步：读取原始传感器文件，并按标准通道收集数据片段
    # -------------------------------------------------------------------------

    # channel_frames 的结构近似：
    #
    # {
    #     "ambient_temperature": [DataFrame(timestamp, raw), ...],
    #     "defrost_active": [DataFrame(timestamp, raw), ...],
    # }
    #
    # 一个标准通道可能对应多个原始文件或多个原始列，因此 value 是 list。
    #
    # invalid_timestamp_rows 统计所有传感器文件中无法解析时间戳的原始行数。
    channel_frames = _load_prepare_channel_frames(inputs.sensor_files, config, channels)

    # 这里的“available”表示：
    # 至少在某个输入文件中发现了该标准通道对应的原始来源列。
    #
    # 它不表示每一行都有有效数值，也不表示通道没有重复、冲突或非法值。
    available_source_channels = set(channel_frames)

    # -------------------------------------------------------------------------
    # 第 3 步：建立 Prepared 的统一原始时间轴
    # -------------------------------------------------------------------------

    # 收集所有已映射标准通道中出现过的有效时间戳，去重并排序。
    # 此处不是固定 1 秒或 10 秒完整网格，只是原始观测时间戳的并集。
    timestamps = _all_timestamps(channel_frames)

    # 如果文件存在，但所有时间戳都无效，或没有任何原始列能映射到 channels 合同，
    # 就无法继续形成 Prepared 表。
    if timestamps.empty:
        raise ValueError("sensor files contain no valid timestamps")

    # 先用时间戳建立 Prepared 主表。
    prepared = pd.DataFrame({"timestamp": timestamps})

    # insert(0, ...) 每次都插到第一列。
    # 连续两次插入后，最终顺序为：
    # experiment_id, experiment_date, timestamp。
    prepared.insert(0, "experiment_date", config.experiment_date)
    prepared.insert(0, "experiment_id", config.experiment_id)

    # 先在字典中集中构建所有通道列，最后一次 concat 到 prepared。
    # 这样比在循环中反复向 DataFrame 插列更清楚，也通常更高效。
    channel_columns: dict[str, pd.Series] = {}

    # -------------------------------------------------------------------------
    # 第 4 步：逐个标准通道解析并对齐到统一时间轴
    # -------------------------------------------------------------------------

    for name, settings in channels.items():
        # derived 通道依赖其他通道计算，不属于原始观测。
        # Prepare 只整理 source channel，因此在这里跳过。
        if str(settings.get("kind")) == "derived":
            continue

        # 将该通道来自不同文件/列的记录合并、解析并对齐到 timestamps。
        #
        # 返回列：
        # value：唯一且合法时保留的标准值；
        # __missing：该时间戳没有非空原始值；
        # __invalid：至少存在一个无法解析或超范围的非空值；
        # __duplicate：同一时间戳出现多于一条原始记录；
        # __conflict：同一时间戳出现多个不同的合法值。
        values = _combine_channel(channel_frames.get(name, []), settings, timestamps)

        # 主通道列使用 canonical channel 名。
        channel_columns[name] = values["value"]

        # Prepare 保留四种来源质量证据，不在这里自动修复。
        for suffix in ("__missing", "__invalid", "__duplicate", "__conflict"):
            channel_columns[f"{name}{suffix}"] = values[suffix]

    # 将所有标准通道及质量标记一次性拼接到 Prepared 主表。
    # index 已与 prepared 的 RangeIndex 对齐。
    prepared = pd.concat(
        [prepared, pd.DataFrame(channel_columns, index=prepared.index)], axis=1
    )

    # -------------------------------------------------------------------------
    # 第 5 步：确保循环切分所需的除霜通道一定存在
    # -------------------------------------------------------------------------

    # 从 CycleSettings 中读取循环识别使用的标准除霜状态通道。
    defrost_channel = config.cycles.defrost_channel

    # 正常情况下，除霜通道应在 channels 合同中声明，因此前面已经建列。
    # 这个分支是防御性处理：即便合同中完全没有该列，也生成一列全缺失状态，
    # 让 cycles.py 能按“不知道除霜状态”处理，而不是直接 KeyError。
    if defrost_channel not in prepared:
        # pandas nullable boolean 可以同时表达 True / False / <NA>。
        prepared[defrost_channel] = pd.Series(pd.NA, index=prepared.index, dtype="boolean")

        # 整列不存在来源，因此所有时间戳都标记为 missing；
        # 但它不是“有原始值却解析失败”，也没有重复或冲突。
        prepared[f"{defrost_channel}__missing"] = True
        prepared[f"{defrost_channel}__invalid"] = False
        prepared[f"{defrost_channel}__duplicate"] = False
        prepared[f"{defrost_channel}__conflict"] = False

    # -------------------------------------------------------------------------
    # 第 6 步：调用 cycles.py 切分循环和阶段
    # -------------------------------------------------------------------------

    # label_cycles() 会在 prepared 中增加循环相关列，并生成 cycle_summary。
    #
    # 典型新增信息包括：
    # cycle_id、cycle_status、cycle_stage、
    # heating_start、defrost_start、defrost_end、
    # elapsed / progress 等。
    #
    # Prepare 只调用循环模块，不在本文件里重复实现状态机。
    prepared, cycle_summary = label_cycles(
        prepared,
        defrost_channel,
        config.cycles,
        experiment_id=config.experiment_id,
        experiment_date=config.experiment_date,
    )

    # -------------------------------------------------------------------------
    # 第 7 步：将图片一次性匹配到 Prepared 时间戳
    # -------------------------------------------------------------------------

    # 给 match_images() 传入：
    # 1. Prepared 时间戳；
    # 2. 相对于 input_dir 的图片路径；
    # 3. 相机 ID → role 映射；
    # 4. 最大匹配容差。
    #
    # 使用相对路径可避免 Prepared 表依赖某台机器的绝对目录。
    image_matches = match_images(
        prepared["timestamp"],
        [path.relative_to(config.input_dir) for path in inputs.image_files],
        camera_roles=camera_roles,
        tolerance_seconds=config.image_match_tolerance_seconds,
    )

    # match_images() 可能生成例如：
    # image_front_path
    # image_front_time
    # image_front_offset_seconds
    #
    # to_numpy() 按位置赋值，避免 image_matches 自身 index 与 prepared index
    # 不一致时发生基于标签的意外对齐。
    for column in image_matches.columns:
        prepared[column] = image_matches[column].to_numpy()

    # 按实验和时间稳定排序，并重建连续 RangeIndex。
    # kind="stable" 保证相同键值记录仍保持原相对顺序。
    prepared = prepared.sort_values(["experiment_id", "timestamp"], kind="stable").reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # 第 8 步：给循环摘要补充 Prepare 阶段的质量指标
    # -------------------------------------------------------------------------

    cycle_summary = _add_cycle_summary_metrics(
        prepared,
        cycle_summary,
        channels,
        camera_roles,
        config.expected_sensor_interval_seconds,
        available_source_channels,
    )

    return prepared, cycle_summary


# =============================================================================
# 2. 将原始文件列映射为标准通道的数据片段
# =============================================================================


def _load_prepare_channel_frames(
    sensor_files: tuple[Path, ...],
    config: Config,
    channels: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[pd.DataFrame]]:
    """Load main files and append EDF environment frames when configured."""
    edf_paths = tuple(path for path in sensor_files if path.suffix.lower() == ".edf")
    main_paths = tuple(path for path in sensor_files if path.suffix.lower() != ".edf")
    if not main_paths:
        raise ValueError(f"no non-EDF sensor files found in {config.input_dir}")

    channel_frames = _load_channel_frames(main_paths, channels, config.timestamp_column)
    main_timestamps = _all_timestamps(channel_frames)
    if main_timestamps.empty:
        raise ValueError("sensor files contain no valid timestamps")

    if not edf_paths:
        return channel_frames

    environment = read_edf_environment(
        edf_paths,
        pd.Timestamp(main_timestamps.min()),
        pd.Timestamp(main_timestamps.max()),
        pd.to_timedelta(config.edf_pair_tolerance_seconds, unit="s"),
    )
    environment = _align_environment_to_main_timestamps(
        environment,
        main_timestamps,
        pd.to_timedelta(config.expected_sensor_interval_seconds / 2, unit="s"),
    )
    for name in ("environment_temperature", "environment_relative_humidity"):
        channel_frames[name] = [
            environment[["timestamp", name]].rename(columns={name: "raw"})
        ]
    return channel_frames


def _align_environment_to_main_timestamps(
    environment: pd.DataFrame,
    main_timestamps: pd.Series,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    """Map fused EDF observations onto the authoritative main time axis."""
    if environment.empty:
        return environment.copy()

    main = pd.Series(main_timestamps).reset_index(drop=True)
    environment = environment.reset_index(drop=True)
    pairs = match_nearest_one_to_one(
        main,
        environment["timestamp"],
        tolerance,
    )
    if not pairs:
        return environment.iloc[0:0].copy()

    main_positions = [left for left, _ in pairs]
    environment_positions = [right for _, right in pairs]
    aligned = environment.iloc[environment_positions].reset_index(drop=True)
    aligned["timestamp"] = pd.Series(
        [main.iloc[position] for position in main_positions], dtype=main.dtype
    )
    return aligned[["timestamp", "environment_temperature", "environment_relative_humidity"]]


def _load_channel_frames(
    paths: tuple[Path, ...],
    channels: Mapping[str, Mapping[str, Any]],
    timestamp_column: str,
) -> dict[str, list[pd.DataFrame]]:

    # source_to_channels 建立：
    # 原始来源名称 → 一个或多个 canonical channel 名。
    #
    # 一个来源可以被多个标准通道引用，因此 value 使用 set。
    source_to_channels: dict[str, set[str]] = defaultdict(set)

    # 先反向建立 source name 查找表。
    for name, settings in channels.items():
        # derived 通道没有原始 source_names，不在 Prepare 读取。
        if str(settings.get("kind")) == "derived":
            continue

        # channels.yaml 中每个标准通道可以声明多个可能的 source_names。
        for source in settings.get("source_names", []):
            source_to_channels[str(source)].add(name)

    # frames 的结构：
    # canonical channel → 多个 DataFrame(timestamp, raw)。
    frames: dict[str, list[pd.DataFrame]] = defaultdict(list)

    # 逐个读取传感器源文件。
    for path in paths:
        table = _read_sensor_table(path, timestamp_column)

        # 文件可能只有无效时间戳；过滤后为空则跳过。
        if table.empty:
            continue

        # 从文件名提取参数组，例如：
        # “参数1_20260715.txt” → "1"。
        group = _parameter_group(path)

        # 遍历该文件除 timestamp 外的每个原始数据列。
        for raw_column in table.columns:
            if raw_column == "timestamp":
                continue

            # 把文件参数组和原始列名组合成 channels.yaml 使用的 source name。
            # 例如 group="1", raw_column="Te" → "p1__Te"。
            canonical = _canonical_name(group, str(raw_column))

            # 找出所有引用该原始来源的标准通道。
            # 未在 channels 合同中声明的原始列会被忽略，不进入 Prepared。
            channel_names = source_to_channels.get(canonical, set())

            for name in channel_names:
                # 这里只保留 timestamp 和尚未解析的 raw 字符串。
                # 数值解析、范围检查和重复冲突判断统一交给 _combine_channel()。
                frames[name].append(
                    pd.DataFrame({"timestamp": table["timestamp"], "raw": table[raw_column]})
                )

    return frames


# =============================================================================
# 3. 读取单个传感器文本文件
# =============================================================================


def _read_sensor_table(path: Path, timestamp_column: str) -> pd.DataFrame:
    # 只读取文件前 128 KiB 用于：
    # 1. 判断是否为二进制 Excel；
    # 2. 尝试编码；
    # 3. 推断分隔符。
    #
    # 实际完整表格仍由 pd.read_csv() 从 path 读取。
    sample = path.read_bytes()[:131_072]

    # 识别两类常见二进制 Excel 文件头：
    # D0 CF 11 E0：旧式 .xls OLE 文件；
    # PK 03 04：.xlsx 等 ZIP 容器格式。
    #
    # 即使文件扩展名伪装成 .txt/.csv，也会被拒绝。
    if sample.startswith((b"\xd0\xcf\x11\xe0", b"PK\x03\x04")):
        raise ValueError(f"binary Excel workbook is not supported: {path}")

    # 默认严格解码；只有两种编码都无法识别时才降级为 replace。
    encoding_errors = "strict"

    # 优先尝试 UTF-8 with BOM，再尝试兼容中文 Windows 文件的 gb18030。
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            decoded = sample.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        # 极端情况下使用 gb18030 并替换无法解码字节，
        # 使文件仍可被读取，同时避免整个流程因少数字节中断。
        encoding = "gb18030"
        encoding_errors = "replace"
        decoded = sample.decode(encoding, errors=encoding_errors)

    # 在制表符、逗号和分号中识别文件分隔符。
    delimiter = _detect_delimiter(decoded)

    # 全部先按字符串读取：
    # - 防止 pandas 自动改变设备编号或分类值；
    # - 统一由 channels 合同决定后续解析方式；
    # - keep_default_na=False 保留空字符串，便于区分缺失与非法非空值。
    table = pd.read_csv(
        path,
        sep=delimiter,
        encoding=encoding,
        dtype=str,
        keep_default_na=False,
        engine="python",
        encoding_errors=encoding_errors,
    )

    # 清理表头两侧空格，避免 " Time " 与 "Time" 被视为不同列。
    table.columns = [str(column).strip() for column in table.columns]

    # 时间戳列是建立统一时间轴的必要字段。
    if timestamp_column not in table.columns:
        raise ValueError(f"timestamp column {timestamp_column!r} not found in {path}")

    # 从原表移出时间戳列，并将无法解析的值转为 NaT。
    timestamps = pd.to_datetime(table.pop(timestamp_column), errors="coerce")

    # 将规范化后的 timestamp 放回第一列。
    table = pd.concat([timestamps.rename("timestamp"), table], axis=1)

    # 删除无效时间戳行并重建连续索引。
    # 注意：这里只删除无法定位到时间轴的行，不删除通道值为空或非法的行。
    return table.loc[table["timestamp"].notna()].reset_index(drop=True)


# =============================================================================
# 4. 合并一个标准通道的全部原始来源
# =============================================================================


def _combine_channel(
    frames: list[pd.DataFrame], settings: Mapping[str, Any], timestamps: pd.Series
) -> pd.DataFrame:
    # 目标：
    # 将同一 canonical channel 来自不同文件和原始列的数据，
    # 对齐到统一 timestamps，并生成 value + 四种来源质量标记。

    # -------------------------------------------------------------------------
    # 情况 A：本次输入中完全没有发现这个通道的来源列
    # -------------------------------------------------------------------------
    if not frames:
        empty = pd.DataFrame(index=timestamps.index)

        # 没有来源时，所有标准值均为 NaN。
        empty["value"] = np.nan

        # 每个时间戳都属于“缺少原始观测”。
        empty["__missing"] = True

        # 没有非空原始值，因此不属于解析非法。
        empty["__invalid"] = False

        # 没有记录，因此不存在重复和冲突。
        empty["__duplicate"] = False
        empty["__conflict"] = False
        return empty

    # -------------------------------------------------------------------------
    # 情况 B：存在一个或多个来源片段
    # -------------------------------------------------------------------------

    # 合并全部来源记录并按时间稳定排序。
    # 这里暂不去重，也不平均重复值。
    records = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="stable")

    # kind 决定值的解析方式：
    # event / categorical / continuous / step / protected 等。
    kind = str(settings.get("kind"))

    # 原始值统一使用 pandas string，并去掉两侧空格。
    raw_values = records["raw"].astype("string").str.strip()

    # 根据 kind 和 channels 配置完成：
    # - 事件映射；
    # - 数值解析；
    # - scale / offset；
    # - valid_range 检查。
    #
    # cleaned：规范化后的值；
    # invalid：非空原始值是否非法。
    cleaned, invalid = _parse_values(raw_values, kind, settings)

    # 构造后续按 timestamp 分组的工作表。
    work = pd.DataFrame(
        {
            "timestamp": records["timestamp"].to_numpy(),
            "value": cleaned.to_numpy(),
            "invalid": invalid.to_numpy(),

            # nonempty 记录原始单元格是否真正包含内容。
            # 它用于区分：
            # - missing：原始值为空；
            # - invalid：原始值非空，但无法解析或超范围。
            "nonempty": (raw_values.ne("") & raw_values.notna()).to_numpy(),
        }
    )

    # 同一标准通道、同一 timestamp 的所有来源记录放到一组。
    grouped = work.groupby("timestamp", sort=True)

    # 每个时间戳共有多少条来源记录。
    counts = grouped.size()

    # 同一时间戳是否至少出现一条非法记录。
    invalid_any = grouped["invalid"].any()

    # 同一时间戳是否至少出现一条非空原始记录。
    nonempty_any = grouped["nonempty"].any()

    # 取分组中的第一个非空规范值。
    # 但后面只有 counts == 1 且不存在 invalid 时才允许它进入最终 value，
    # 因此这里不会用“第一个值”解决重复记录。
    first_value = grouped["value"].first()

    # 只保留合法且非空的值，用于判断多个来源之间是否互相冲突。
    valid_values = work.loc[~work["invalid"] & work["value"].notna(), ["timestamp", "value"]]

    # 转成 string key 后统计每个时间戳有多少个不同合法值。
    # 两条相同合法值：duplicate=True, conflict=False；
    # 两条不同合法值：duplicate=True, conflict=True。
    valid_values = valid_values.assign(value_key=valid_values["value"].astype("string"))
    unique_counts = valid_values.groupby("timestamp")["value_key"].nunique()

    # 统一时间轴转为 DatetimeIndex，便于各统计量 reindex。
    index = pd.DatetimeIndex(timestamps)

    # 最终结果先以完整原始时间戳并集为索引。
    result = pd.DataFrame(index=index)

    # 对没有该通道记录的时间戳：
    # count=0、invalid=False、nonempty=False、unique_count=0。
    counts_aligned = counts.reindex(index, fill_value=0)
    invalid_aligned = invalid_any.reindex(index, fill_value=False)
    nonempty_aligned = nonempty_any.reindex(index, fill_value=False)
    unique_counts_aligned = unique_counts.reindex(index, fill_value=0)

    # Prepare 的保守合同：
    # 只有“该时间戳恰好一条来源记录，并且该记录不非法”时才保留 value。
    #
    # 即使两条重复记录的合法数值完全相同，也不会在 Prepare 中擅自选取或平均；
    # 它们仍被置为 NaN，并通过 __duplicate 明确记录。
    result["value"] = first_value.reindex(index).where(
        counts_aligned.eq(1) & ~invalid_aligned, np.nan
    )

    # missing 表示不存在任何非空原始记录。
    # 非空但无法解析的值属于 invalid，不属于 missing。
    result["__missing"] = ~nonempty_aligned

    # invalid 表示至少一条非空来源记录无法解析或超出有效范围。
    result["__invalid"] = invalid_aligned

    # duplicate 表示同一时间戳出现多于一条来源记录，
    # 不关心这些记录的值是否相同。
    result["__duplicate"] = counts_aligned.gt(1)

    # conflict 表示同一时间戳至少出现两个不同的合法规范值。
    result["__conflict"] = unique_counts_aligned.gt(1)

    # 将质量列稳定收敛为普通 bool，避免 object/nullable 类型漂移。
    result["__missing"] = result["__missing"].astype(bool)
    for column in ("__invalid", "__duplicate", "__conflict"):
        result[column] = result[column].astype(bool)

    # 去掉 DatetimeIndex，恢复与 Prepared 主表相同的 RangeIndex。
    return result.reset_index(drop=True)


# =============================================================================
# 5. 按通道类型解析原始值
# =============================================================================


def _parse_values(
    raw_values: pd.Series, kind: str, settings: Mapping[str, Any]
) -> tuple[pd.Series, pd.Series]:
    # nonempty 用于区分：
    # 空字符串/NA → missing；
    # 非空但无法解释 → invalid。
    nonempty = raw_values.ne("") & raw_values.notna()

    # -------------------------------------------------------------------------
    # Event：把设备原始事件编码映射为 True / False
    # -------------------------------------------------------------------------
    if kind == "event":
        # allowed_values 来自 channels 合同，例如：
        # {"0": false, "1": true, "OFF": false, "ON": true}。
        allowed = settings.get("allowed_values", {})

        # 原始键去空格并转大写，使 on / ON / " ON " 使用同一映射。
        # bool(value) 预期 value 已由 channels 配置校验为真正布尔值。
        mapping = {str(key).strip().upper(): bool(value) for key, value in allowed.items()}

        # 未在 allowed_values 中出现的非空编码映射为 NaN。
        values = raw_values.map(lambda value: mapping.get(str(value).strip().upper(), np.nan))

        # 空值属于 missing；只有非空且映射失败才属于 invalid。
        invalid = nonempty & values.isna()

        # 使用 object 以同时容纳 bool 和 NaN。
        return values.astype("object"), invalid

    # -------------------------------------------------------------------------
    # Categorical：保留非空文本，不进行数值转换和范围判断
    # -------------------------------------------------------------------------
    if kind == "categorical":
        # 空值转为 pd.NA；非空文本原样保留。
        # 当前 categorical 分支不定义 allowed category，因此不会产生 invalid。
        return raw_values.where(nonempty, pd.NA).astype("object"), pd.Series(
            False, index=raw_values.index, dtype=bool
        )

    # -------------------------------------------------------------------------
    # 其他通道：按数值处理
    # -------------------------------------------------------------------------

    # 空字符串先转 pd.NA，再将其余文本转为数值。
    # 无法解析的非空文本会得到 NaN。
    numeric = pd.to_numeric(raw_values.replace("", pd.NA), errors="coerce")

    # 非空但无法转成数值的记录属于 invalid。
    invalid = nonempty & numeric.isna()

    # 按 channels 合同进行单位缩放和偏移：
    # standardized_value = raw_value * scale + offset。
    scale = float(settings.get("scale", 1.0))
    offset = float(settings.get("offset", 0.0))
    numeric = numeric * scale + offset

    # valid_range 是标准化之后数值的允许范围。
    valid_range = settings.get("valid_range")

    # 只有恰好两个元素的 list 才启用上下界检查。
    if isinstance(valid_range, list) and len(valid_range) == 2:
        lower, upper = float(valid_range[0]), float(valid_range[1])

        # between() 默认包含上下界。
        out_of_range = numeric.notna() & ~numeric.between(lower, upper)

        # 超范围值属于 invalid，并从标准值列中屏蔽为 NaN。
        invalid = invalid | out_of_range
        numeric = numeric.mask(out_of_range)

    # 数值通道统一返回 float，另返回逐行 invalid bool。
    return numeric.astype(float), invalid


# =============================================================================
# 6. 建立所有已映射通道的原始时间戳并集
# =============================================================================


def _all_timestamps(channel_frames: Mapping[str, list[pd.DataFrame]]) -> pd.Series:
    # 展开所有标准通道的全部来源片段，只取 timestamp 列。
    values = [frame["timestamp"] for frames in channel_frames.values() for frame in frames]

    # 没有任何已映射来源时，返回明确的空 datetime Series。
    if not values:
        return pd.Series(dtype="datetime64[ns]")

    # 合并、去重、排序。
    # 这只是“观测时间戳并集”，不是补全后的规则时间网格。
    unique = pd.concat(values, ignore_index=True).drop_duplicates().sort_values()

    # 使用底层数组重新构造 Series，得到从 0 开始的连续索引。
    return pd.Series(unique.to_numpy())


# =============================================================================
# 7. 为 cycle_summary 增加 Prepare 阶段质量指标
# =============================================================================


def _add_cycle_summary_metrics(
    prepared: pd.DataFrame,
    summary: pd.DataFrame,
    channels: Mapping[str, Mapping[str, Any]],
    camera_roles: Mapping[str, str],
    expected_interval_seconds: int,
    available_source_channels: set[str],
) -> pd.DataFrame:
    # raw_channels 包含全部非 derived 标准通道。
    # 这些通道可能在本次输入中不可用，后续 observed_fraction 会再次过滤。
    raw_channels = [
        name for name, settings in channels.items() if str(settings.get("kind")) != "derived"
    ]

    # 根据配置的物理角色构造图片路径列名。
    # set() 去除重复角色；config.py 已原则上禁止角色重复。
    role_path_columns = [f"image_{role}_path" for role in sorted(set(camera_roles.values()))]

    # 对应构造每个角色的图片实际拍摄时间列名。
    role_time_columns = [f"image_{role}_time" for role in sorted(set(camera_roles.values()))]

    # 不原地修改 cycles.py 返回的 summary。
    result = summary.copy()

    # 每个循环生成一个指标字典，最后一次性转 DataFrame。
    records: list[dict[str, object]] = []

    # cycle_summary 每行代表一个循环，包括 valid / incomplete / invalid 等状态。
    for _, row in result.iterrows():
        cycle_id = row["cycle_id"]

        # 取出 Prepared 中属于该 cycle_id 的所有时间戳行。
        # 当前 prepare() 一次只处理一个 experiment，因此这里只按 cycle_id 过滤。
        group = prepared.loc[prepared["cycle_id"].eq(cycle_id)]

        # 该循环实际出现在 Prepared 原始时间轴中的行数。
        raw_count = len(group)

        # 根据 heating_start、defrost_end 和理论原始采样间隔估算应有行数。
        expected = _expected_row_count(row, expected_interval_seconds)

        # 计算“实际存在来源的原始通道”在该循环中的平均非空单元格比例。
        observed_fraction = _observed_fraction(
            group, raw_channels, available_source_channels
        )

        # Prepared 原始时间轴中相邻时间戳的最大间隔。
        maximum_gap = _maximum_gap(group["timestamp"])

        # 统计至少一个原始通道带 duplicate 标记的时间戳行数。
        duplicate_count = _quality_row_count(group, raw_channels, "__duplicate")

        # 统计至少一个原始通道带 conflict 标记的时间戳行数。
        conflict_count = _quality_row_count(group, raw_channels, "__conflict")

        # 将所有角色图片路径单元格展平，用于统计唯一图片文件数。
        image_values = group[role_path_columns].to_numpy().ravel()

        # 同一图片理论上可能因匹配逻辑出现在多个位置；
        # nunique() 统计唯一图片路径，而不是非空单元格总数。
        image_count = int(pd.Series(image_values).dropna().nunique()) if image_values.size else 0

        # role_count 表示该循环中至少出现过一张图片的角色数量。
        # 它不是每个时间戳的相机完整性，也不是图片总数。
        role_count = sum(int(group[column].notna().any()) for column in role_path_columns)

        # 配置角色中，在该循环至少出现过一次图片的比例。
        complete_fraction = role_count / len(role_path_columns) if role_path_columns else 0.0

        # 分别计算每个角色连续图片拍摄时间的最大间隔。
        image_gaps = [
            _maximum_image_gap(group[column].dropna())
            for column in role_time_columns
            if column in group
        ]

        # 单张或零张图片时最大间隔为 NaN，需要排除。
        image_gaps = [value for value in image_gaps if pd.notna(value)]

        # 循环级 maximum_rgb_gap 取所有角色最大间隔中的最大值。
        image_gap = max(image_gaps) if image_gaps else np.nan

        records.append(
            {
                # Prepared 中该循环实际存在的原始时间戳行数。
                "raw_row_count": raw_count,

                # 按理论采样间隔估算的循环期望行数。
                "expected_row_count": expected,

                # 可用来源通道的平均观测单元格比例。
                "sensor_observed_fraction": observed_fraction,

                # 原始时间轴最大断点。
                "maximum_timeline_gap_seconds": maximum_gap,

                # 出现任一通道重复记录的时间戳行数。
                "duplicate_observation_count": duplicate_count,

                # 出现任一通道冲突记录的时间戳行数。
                "conflict_observation_count": conflict_count,

                # 循环中匹配到的唯一 RGB 图片数量。
                "rgb_image_count": image_count,

                # 循环中至少出现一张图的相机角色数量。
                "rgb_role_count": role_count,

                # 配置相机角色的循环级出现比例。
                "rgb_role_presence_fraction": complete_fraction,

                # 所有角色中最大的相邻图片时间间隔。
                "maximum_rgb_gap_seconds": image_gap,
            }
        )

    # 保持 cycles.py 原有摘要列在前，Prepare 质量指标追加在后。
    return pd.concat([result.reset_index(drop=True), pd.DataFrame(records)], axis=1)


# =============================================================================
# 8. 循环级质量指标辅助函数
# =============================================================================


def _expected_row_count(row: pd.Series, interval_seconds: int) -> object:
    # 完整循环期望范围使用 heating_start → defrost_end。
    start = row.get("heating_start")
    end = row.get("defrost_end")

    # incomplete/invalid 循环可能缺少明确边界，此时期望行数不可定义。
    if not isinstance(start, pd.Timestamp) or not isinstance(end, pd.Timestamp):
        return np.nan

    # 使用 ceil 是因为持续时间未必正好是 interval 的整数倍。
    # 该值是理论期望行数，不会在 Prepare 中据此补行。
    return int(np.ceil((end - start).total_seconds() / interval_seconds))


def _observed_fraction(
    group: pd.DataFrame, channels: list[str], available_source_channels: set[str]
) -> float:
    # 只统计：
    # 1. 本次文件中确实发现过来源列；
    # 2. 当前 Prepared 表中确实存在的非 derived 通道。
    #
    # 完全 unavailable 的通道不进入分母，否则大量未部署传感器会系统性压低覆盖率。
    available = [
        name for name in channels if name in available_source_channels and name in group
    ]

    # 没有循环数据或没有任何可用来源通道时，定义为 0。
    if group.empty or not available:
        return 0.0

    # group[available].notna()：
    # 每个单元格是否有最终保留的 value。
    #
    # 第一个 mean()：每个通道在时间维度的非空比例；
    # 第二个 mean()：再对全部可用通道取平均。
    #
    # 注意：重复、冲突和非法记录的 value 在 _combine_channel() 中已置 NaN，
    # 因此它们会降低该指标。
    return float(group[available].notna().mean().mean())


def _maximum_gap(timestamps: pd.Series) -> float:
    # 再次防御性解析时间戳，删除 NaT 并排序。
    parsed = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()

    # 少于两个时间戳无法定义“相邻最大间隔”。
    if len(parsed) < 2:
        return np.nan

    # diff() 计算相邻时间差，转换为秒后取最大值。
    return float(parsed.diff().dt.total_seconds().dropna().max())


def _maximum_image_gap(timestamps: pd.Series) -> float:
    # 该实现与 _maximum_gap() 当前完全相同，
    # 只是函数名明确表达调用对象是图片时间。
    #
    # 后续可以考虑合并成一个通用函数，但若保留两个语义化名称，
    # 阅读调用点会更直接；不要仅为减少几行代码而仓促改动。
    parsed = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()

    if len(parsed) < 2:
        return np.nan

    return float(parsed.diff().dt.total_seconds().dropna().max())


def _quality_row_count(group: pd.DataFrame, channels: list[str], suffix: str) -> int:
    # 找出当前表中实际存在的质量列，例如：
    # ambient_temperature__duplicate。
    columns = [f"{name}{suffix}" for name in channels if f"{name}{suffix}" in group]

    # 没有数据或没有对应质量列时，计数为 0。
    if group.empty or not columns:
        return 0

    # any(axis=1) 表示：
    # 只要某个时间戳上任一通道出现该质量问题，就把该行计数一次。
    #
    # 因此结果是“问题时间戳行数”，不是问题单元格数，也不是重复源记录条数。
    return int(group[columns].astype(bool).any(axis=1).sum())


# =============================================================================
# 9. 原始列名标准化辅助函数
# =============================================================================


def _parameter_group(path: Path) -> str | None:
    # 从文件名 stem 中寻找“参数 + 数字”，例如：
    # 参数1_0715 → "1"
    # 参数12 → "12"
    match = re.search(r"参数(?P<group>\d+)", path.stem)

    # 文件名没有参数组时返回 None。
    return None if match is None else match.group("group")


def _canonical_name(group: str | None, raw_column: str) -> str:
    # 有参数组：
    # group="1", raw_column="Te" → "p1__Te"
    #
    # 无参数组：
    # raw_column="Te" → "Te"
    #
    # 该 source name 必须与 channels.yaml 中的 source_names 完全匹配。
    return f"p{group}__{raw_column}" if group else raw_column


# =============================================================================
# 10. 文本分隔符识别
# =============================================================================


def _detect_delimiter(sample: str) -> str:
    # 优先让 csv.Sniffer 在 tab、逗号、分号中推断结构。
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        # Sniffer 失败时，退化为统计三种候选分隔符在样本中的出现次数。
        counts = {delimiter: sample.count(delimiter) for delimiter in ("\t", ",", ";")}

        # 选择出现次数最多的候选分隔符。
        delimiter, count = max(counts.items(), key=lambda item: item[1])

        # 三种分隔符都没有出现，无法将文件解释为受支持的表格文本。
        if count == 0:
            raise ValueError("no supported delimiter") from None

        return delimiter
