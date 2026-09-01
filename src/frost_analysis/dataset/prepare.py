"""Prepare raw sensor and image records without reconstructing measurements."""

from __future__ import annotations

import codecs
import csv
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .alignment import match_nearest_one_to_one
from .config import Config
from .cycles import label_cycles
from .matching import image_roles, match_images
from .raw import discover_inputs
from .sensors import read_edf_environment


def prepare(
    config: Config, channels: Mapping[str, Mapping[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw observations, segment cycles, and attach one-shot image matches."""
    inputs = discover_inputs(config)

    if not inputs.sensor_files:
        raise ValueError(f"no sensor files found in {config.input_dir}")

    channel_frames = _load_prepare_channel_frames(inputs.sensor_files, config, channels)
    available_source_channels = set(channel_frames)

    # 此处不是固定 1 秒或 10 秒完整网格，只是原始观测时间戳的并集。
    timestamps = _all_timestamps(channel_frames)

    if timestamps.empty:
        raise ValueError("sensor files contain no valid timestamps")

    prepared = pd.DataFrame({"timestamp": timestamps})
    prepared.insert(0, "experiment_date", config.experiment_date)
    prepared.insert(0, "experiment_id", config.experiment_id)

    channel_columns: dict[str, pd.Series] = {}

    for name, settings in channels.items():
        if str(settings.get("kind")) == "derived":
            continue
        values = _combine_channel(channel_frames.get(name, []), settings, timestamps)
        channel_columns[name] = values["value"]

        # Prepare 保留四种来源质量证据，不在这里自动修复。
        for suffix in ("__missing", "__invalid", "__duplicate", "__conflict"):
            channel_columns[f"{name}{suffix}"] = values[suffix]

    prepared = pd.concat(
        [prepared, pd.DataFrame(channel_columns, index=prepared.index)], axis=1
    )

    defrost_channel = config.cycles.defrost_channel

    if defrost_channel not in prepared:
        raise ValueError(
            f"configured defrost channel is not declared in Prepared channels: {defrost_channel}"
        )

    prepared, cycle_summary = label_cycles(
        prepared,
        defrost_channel,
        config.cycles,
        experiment_id=config.experiment_id,
        experiment_date=config.experiment_date,
        shutdown_gap_seconds=config.process.continuous_max_gap_seconds,
    )

    # 使用相对路径可避免 Prepared 表依赖某台机器的绝对目录。
    image_matches = match_images(
        prepared["timestamp"],
        [path.relative_to(config.input_dir) for path in inputs.image_files],
        tolerance_seconds=config.image_match_tolerance_seconds,
    )

    for column in image_matches.columns:
        prepared[column] = image_matches[column].to_numpy()

    prepared = prepared.sort_values(["experiment_id", "timestamp"], kind="stable").reset_index(
        drop=True
    )

    cycle_summary = _add_cycle_summary_metrics(
        prepared,
        cycle_summary,
        channels,
        image_roles(prepared),
        config.expected_sensor_interval_seconds,
        available_source_channels,
    )

    return prepared, cycle_summary


def prepare_original(config: Config, prepared: pd.DataFrame) -> pd.DataFrame:
    """Attach every raw controller point to standardized Prepared observations."""
    from .schema import export_original_frame

    inputs = discover_inputs(config)
    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for path in inputs.sensor_files:
        if path.suffix.lower() == ".edf":
            continue
        grouped_paths[_parameter_group(path) or path.stem].append(path)

    occurrence = "__raw_timestamp_occurrence"
    raw: pd.DataFrame | None = None
    for group, paths in grouped_paths.items():
        tables = [
            _read_sensor_table(path, config.timestamp_column, infer_types=True)
            for path in paths
        ]
        combined = pd.concat(tables, ignore_index=True, sort=False)
        combined = combined.rename(
            columns={
                name: _canonical_name(group, str(name))
                for name in combined.columns
                if name != "timestamp"
            }
        )
        combined[occurrence] = combined.groupby("timestamp", sort=False).cumcount()
        raw = (
            combined
            if raw is None
            else raw.merge(
                combined,
                on=["timestamp", occurrence],
                how="outer",
                sort=False,
                validate="one_to_one",
            )
        )

    original = export_original_frame(prepared)
    result = original if raw is None else original.merge(
        raw, on="timestamp", how="outer", sort=False, validate="one_to_many"
    )
    edf_paths = tuple(path for path in inputs.sensor_files if path.suffix.lower() == ".edf")
    if edf_paths:
        environment = read_edf_environment(
            edf_paths,
            pd.Timestamp(original["timestamp"].min()),
            pd.Timestamp(original["timestamp"].max()),
            pd.to_timedelta(config.edf_pair_tolerance_seconds, unit="s"),
        )
        environment = _align_environment_to_main_timestamps(
            environment,
            original["timestamp"],
            pd.to_timedelta(config.expected_sensor_interval_seconds / 2, unit="s"),
        ).rename(columns=lambda name: f"edf__{name}" if str(name).startswith("sensor_") else name)
        edf_columns = [name for name in environment if str(name).startswith("edf__")]
        result = result.merge(
            environment[["timestamp", *edf_columns]],
            on="timestamp",
            how="left",
            validate="many_to_one",
        )
    return result.sort_values(["timestamp", occurrence], kind="stable").drop(
        columns=occurrence
    ).reset_index(drop=True)


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
    return aligned


def _load_channel_frames(
    paths: tuple[Path, ...],
    channels: Mapping[str, Mapping[str, Any]],
    timestamp_column: str,
) -> dict[str, list[pd.DataFrame]]:

    source_to_channels: dict[str, set[str]] = defaultdict(set)
    for name, settings in channels.items():
        if str(settings.get("kind")) == "derived":
            continue
        for source in settings.get("source_names", []):
            source_to_channels[str(source)].add(name)

    frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for path in paths:
        table = _read_sensor_table(path, timestamp_column)

        if table.empty:
            continue

        group = _parameter_group(path)
        for raw_column in table.columns:
            if raw_column == "timestamp":
                continue

            canonical = _canonical_name(group, str(raw_column))
            # 未在 channels 合同中声明的原始列会被忽略，不进入 Prepared。
            channel_names = source_to_channels.get(canonical, set())

            for name in channel_names:
                frames[name].append(
                    pd.DataFrame({"timestamp": table["timestamp"], "raw": table[raw_column]})
                )

    return frames


def _read_sensor_table(
    path: Path, timestamp_column: str, *, infer_types: bool = False
) -> pd.DataFrame:
    sample = path.read_bytes()[:131_072]

    # 即使文件扩展名伪装成 .txt/.csv，也会被拒绝。
    if sample.startswith((b"\xd0\xcf\x11\xe0", b"PK\x03\x04")):
        raise ValueError(f"binary Excel workbook is not supported: {path}")

    # 优先尝试 UTF-8 with BOM，再尝试兼容中文 Windows 文件的 gb18030。
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            decoded = codecs.getincrementaldecoder(encoding)().decode(
                sample, final=len(sample) < 131_072
            )
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError(
            "utf-8-sig",
            sample,
            0,
            len(sample),
            "sensor file is not valid UTF-8 or GB18030 text",
        )

    delimiter = _detect_delimiter(decoded)

    # 全部先按字符串读取：
    # - 防止 pandas 自动改变设备编号或分类值；
    # - 统一由 channels 合同决定后续解析方式；
    # - keep_default_na=False 保留空字符串，便于区分缺失与非法非空值。
    options = (
        {"keep_default_na": True, "low_memory": False}
        if infer_types
        else {"dtype": str, "keep_default_na": False, "engine": "python"}
    )
    table = pd.read_csv(path, sep=delimiter, encoding=encoding, **options)

    table.columns = [str(column).strip() for column in table.columns]

    if timestamp_column not in table.columns:
        raise ValueError(f"timestamp column {timestamp_column!r} not found in {path}")

    timestamps = pd.to_datetime(table.pop(timestamp_column), errors="coerce")
    table = pd.concat([timestamps.rename("timestamp"), table], axis=1)

    # 删除无效时间戳行并重建连续索引。
    # 注意：这里只删除无法定位到时间轴的行，不删除通道值为空或非法的行。
    return table.loc[table["timestamp"].notna()].reset_index(drop=True)
def _combine_channel(
    frames: list[pd.DataFrame], settings: Mapping[str, Any], timestamps: pd.Series
) -> pd.DataFrame:
    if not frames:
        empty = pd.DataFrame(index=timestamps.index)
        empty["value"] = np.nan
        empty["__missing"] = True
        empty["__invalid"] = False
        empty["__duplicate"] = False
        empty["__conflict"] = False
        return empty

    # 这里暂不去重，也不平均重复值。
    records = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="stable")

    kind = str(settings.get("kind"))
    raw_values = records["raw"].astype("string").str.strip()
    cleaned, invalid = _parse_values(raw_values, kind, settings)

    # 构造后续按 timestamp 分组的工作表。
    work = pd.DataFrame(
        {
            "timestamp": records["timestamp"].to_numpy(),
            "value": cleaned.to_numpy(),
            "invalid": invalid.to_numpy(),
            "nonempty": (raw_values.ne("") & raw_values.notna()).to_numpy(),
        }
    )

    grouped = work.groupby("timestamp", sort=True)
    counts = grouped.size()
    invalid_any = grouped["invalid"].any()
    nonempty_any = grouped["nonempty"].any()

    # 取分组中的第一个非空规范值。
    # 但后面只有 counts == 1 且不存在 invalid 时才允许它进入最终 value，
    # 因此这里不会用“第一个值”解决重复记录。
    first_value = grouped["value"].first()

    valid_values = work.loc[~work["invalid"] & work["value"].notna(), ["timestamp", "value"]]

    # 转成 string key 后统计每个时间戳有多少个不同合法值。
    # 两条相同合法值：duplicate=True, conflict=False；
    # 两条不同合法值：duplicate=True, conflict=True。
    valid_values = valid_values.assign(value_key=valid_values["value"].astype("string"))
    unique_counts = valid_values.groupby("timestamp")["value_key"].nunique()

    index = pd.DatetimeIndex(timestamps)
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

    result["__missing"] = ~nonempty_aligned
    result["__invalid"] = invalid_aligned
    result["__duplicate"] = counts_aligned.gt(1)
    result["__conflict"] = unique_counts_aligned.gt(1)

    # 将质量列稳定收敛为普通 bool，避免 object/nullable 类型漂移。
    result["__missing"] = result["__missing"].astype(bool)
    for column in ("__invalid", "__duplicate", "__conflict"):
        result[column] = result[column].astype(bool)

    return result.reset_index(drop=True)


def _parse_values(
    raw_values: pd.Series, kind: str, settings: Mapping[str, Any]
) -> tuple[pd.Series, pd.Series]:
    nonempty = raw_values.ne("") & raw_values.notna()
    if kind == "event":
        allowed = settings.get("allowed_values", {})

        # 原始键去空格并转大写，使 on / ON / " ON " 使用同一映射。
        # bool(value) 预期 value 已由 channels 配置校验为真正布尔值。
        mapping = {str(key).strip().upper(): bool(value) for key, value in allowed.items()}

        values = raw_values.map(lambda value: mapping.get(str(value).strip().upper(), np.nan))
        invalid = nonempty & values.isna()
        return values.astype("object"), invalid
    if kind == "categorical":
        # 空值转为 pd.NA；非空文本原样保留。
        # 当前 categorical 分支不定义 allowed category，因此不会产生 invalid。
        return raw_values.where(nonempty, pd.NA).astype("object"), pd.Series(
            False, index=raw_values.index, dtype=bool
        )

    numeric = pd.to_numeric(raw_values.replace("", pd.NA), errors="coerce")
    invalid = nonempty & numeric.isna()

    # 按 channels 合同进行单位缩放和偏移：
    # standardized_value = raw_value * scale + offset。
    scale = float(settings.get("scale", 1.0))
    offset = float(settings.get("offset", 0.0))
    numeric = numeric * scale + offset

    valid_range = settings.get("valid_range")
    if isinstance(valid_range, list) and len(valid_range) == 2:
        lower, upper = float(valid_range[0]), float(valid_range[1])
        out_of_range = numeric.notna() & ~numeric.between(lower, upper)
        invalid = invalid | out_of_range
        numeric = numeric.mask(out_of_range)

    return numeric.astype(float), invalid


def _all_timestamps(channel_frames: Mapping[str, list[pd.DataFrame]]) -> pd.Series:
    values = [frame["timestamp"] for frames in channel_frames.values() for frame in frames]
    if not values:
        return pd.Series(dtype="datetime64[ns]")

    # 这只是“观测时间戳并集”，不是补全后的规则时间网格。
    unique = pd.concat(values, ignore_index=True).drop_duplicates().sort_values()

    return pd.Series(unique.to_numpy())


# =============================================================================
# 7. 为 cycle_summary 增加 Prepare 阶段质量指标
# =============================================================================


def _add_cycle_summary_metrics(
    prepared: pd.DataFrame,
    summary: pd.DataFrame,
    channels: Mapping[str, Mapping[str, Any]],
    roles: tuple[str, ...],
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
    role_path_columns = [f"image_{role}_path" for role in roles]

    # 对应构造每个角色的图片实际拍摄时间列名。
    role_time_columns = [f"image_{role}_time" for role in roles]

    # 不原地修改 cycles.py 返回的 summary。
    result = summary.copy()

    # 每个循环生成一个指标字典，最后一次性转 DataFrame。
    records: list[dict[str, object]] = []

    # cycle_summary 每行代表一个循环，状态只分 valid / invalid。
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

    # 开放边界或 invalid 循环可能缺少明确边界，此时期望行数不可定义。
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
