"""科研分析 Pipeline 的轻量、强类型配置合同。

本模块只负责：
1. 从 schema 的日期配置和共享 defaults 中读取配置；
2. 将 YAML 数据转换为不可变的类型化对象；
3. 在 Pipeline 启动前拒绝未知字段、非法数值和不一致配置；
4. 生成与配置文件位置无关的“最终生效配置”及其稳定哈希。

本模块不负责：
- 读取实验传感器数据；
- 切分结霜/除霜循环；
- 重采样、插值、Baseline 或候选通道分析。

真正使用这些参数的模块主要是：
- cycles.py：使用 CycleSettings；
- process.py / baseline.py / features.py：使用 ProcessSettings；
- analysis.py：使用 AnalysisSettings。
"""

from __future__ import annotations

# copy 用于深拷贝 defaults 和 overrides，避免合并时修改原始字典。
import copy

# hashlib 和 json 用于生成最终生效配置的稳定 SHA-256。
import hashlib
import json
import math

# Mapping 表示“字典式对象”，兼容普通 dict 和其他 Mapping 实现。
from collections.abc import Mapping

# asdict：将 dataclass 递归转为普通字典。
# dataclass：声明不可变的类型化配置对象。
# field：为可变字段提供安全的 default_factory。
# fields：读取 dataclass 的字段名，用于拒绝未知配置键。
from dataclasses import asdict, dataclass, field, fields

# date 用于严格检查 experiment_date 是否为 ISO YYYY-MM-DD。
from datetime import date

# Path 统一处理跨平台文件路径。
from pathlib import Path

# Any 用于 YAML 刚加载后尚未完成类型收敛的数据。
from typing import Any

# PyYAML：读取日期配置和共享 defaults.yaml。
import yaml

# =============================================================================
# 1. 循环切分配置
# =============================================================================


@dataclass(frozen=True)
class CycleSettings:
    """保存 cycles.py 切分制热—结霜—除霜循环时使用的阈值。

    frozen=True 表示对象构造完成后不可原地修改，防止同一次运行中参数被意外改变。
    本类只保存并校验阈值；真正的状态去抖、除霜事件识别和循环切分在 cycles.py 中完成。
    """

    # cycles.py 用来识别“是否正在除霜”的标准化事件通道名。
    # 这里保存 canonical channel 名，而不是原始 Excel 列名。
    defrost_channel: str = "defrost_active"

    # 允许自动桥接的除霜状态缺失区间上限（秒）。
    # 只有缺失段两端状态一致且总间隔不超过该值时，cycles.py 才允许补齐状态。
    # 0.0 表示任何除霜状态缺失均不自动推断。
    maximum_state_gap_seconds: float = 0.0

    # 除霜状态去抖阈值（秒）。
    # 用于抑制持续时间过短的 ON/OFF 状态抖动，避免将瞬时跳变识别为真实除霜事件。
    debounce_seconds: float = 20.0

    # 一次已识别除霜事件被判为有效时允许的持续时间范围（秒）。
    # 它们用于质量判定，不是控制机组实际除霜持续时间。
    minimum_defrost_seconds: float = 60.0
    maximum_defrost_seconds: float = 1200.0

    # 相邻两次除霜之间制热阶段的有效持续时间范围（秒）。
    # 具体区间是：前一次除霜结束 heating_start 到下一次除霜开始 defrost_start。
    minimum_heating_seconds: float = 1800.0
    maximum_heating_seconds: float = 21600.0

    # 制热开始后保留为 recovery 阶段的固定时长（秒）。
    # stable_heating_start = heating_start + stable_heating_seconds；
    # 此后才标记为 frost_development。
    # “stable”是固定阶段划分假设，不代表代码通过数据检测到系统已经稳定。
    stable_heating_seconds: float = 180.0

    # 用于检查制热阶段运行模式的 canonical channel 名。
    # 空字符串表示不启用运行模式检查。
    operating_mode_channel: str = ""

    # 有效制热循环要求的运行模式值。
    # 使用字符串是为了与源数据中的文本/分类值保持一致。
    required_operating_mode: str = "3"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CycleSettings:
        """将 YAML 中的 cycles mapping 转换为经过校验的 CycleSettings。

        执行顺序：
        1. 确认输入是 YAML mapping；
        2. 拒绝 dataclass 中不存在的未知键；
        3. 将 YAML 值显式转换为固定 Python 类型；
        4. 检查非负、正数和最小值不大于最大值等约束。
        """

        # 将任意 Mapping 规范化为 str key 的普通字典。
        mapping = _mapping(values, "cycles")

        # 只允许出现 CycleSettings 已声明的字段。
        # 例如 debounce_second 的拼写错误不会被静默忽略。
        _validate_dataclass_keys(mapping, cls, "cycles")

        # 将 YAML 中可能为 int、float 或字符串形式的值收敛到明确类型。
        result = cls(
            defrost_channel=str(
                mapping.get("defrost_channel", cls.defrost_channel)
            ),

            maximum_state_gap_seconds=float(
                mapping.get(
                    "maximum_state_gap_seconds",
                    cls.maximum_state_gap_seconds,
                )
            ),
            debounce_seconds=float(
                mapping.get("debounce_seconds", cls.debounce_seconds)
            ),
            minimum_defrost_seconds=float(
                mapping.get(
                    "minimum_defrost_seconds",
                    cls.minimum_defrost_seconds,
                )
            ),
            maximum_defrost_seconds=float(
                mapping.get(
                    "maximum_defrost_seconds",
                    cls.maximum_defrost_seconds,
                )
            ),
            minimum_heating_seconds=float(
                mapping.get(
                    "minimum_heating_seconds",
                    cls.minimum_heating_seconds,
                )
            ),
            maximum_heating_seconds=float(
                mapping.get(
                    "maximum_heating_seconds",
                    cls.maximum_heating_seconds,
                )
            ),
            stable_heating_seconds=float(
                mapping.get(
                    "stable_heating_seconds",
                    cls.stable_heating_seconds,
                )
            ),
            operating_mode_channel=str(
                mapping.get(
                    "operating_mode_channel",
                    cls.operating_mode_channel,
                )
            ),
            required_operating_mode=str(
                mapping.get(
                    "required_operating_mode",
                    cls.required_operating_mode,
                )
            ),
        )

        # 缺失桥接阈值允许为 0，但不能为负数。
        _validate_nonnegative(
            "maximum_state_gap_seconds",
            result.maximum_state_gap_seconds,
        )

        # 去抖时间和有效持续时间范围必须严格大于 0。
        _validate_positive("debounce_seconds", result.debounce_seconds)
        _validate_positive(
            "minimum_defrost_seconds",
            result.minimum_defrost_seconds,
        )
        _validate_positive(
            "maximum_defrost_seconds",
            result.maximum_defrost_seconds,
        )
        _validate_positive(
            "minimum_heating_seconds",
            result.minimum_heating_seconds,
        )
        _validate_positive(
            "maximum_heating_seconds",
            result.maximum_heating_seconds,
        )

        # recovery 持续时间允许设为 0，表示不保留固定 recovery 阶段。
        _validate_nonnegative(
            "stable_heating_seconds",
            result.stable_heating_seconds,
        )

        # 最小有效除霜时长不能大于最大有效除霜时长。
        if result.minimum_defrost_seconds > result.maximum_defrost_seconds:
            raise ValueError(
                "minimum_defrost_seconds must not exceed "
                "maximum_defrost_seconds"
            )

        # 最小有效制热时长不能大于最大有效制热时长。
        if result.minimum_heating_seconds > result.maximum_heating_seconds:
            raise ValueError(
                "minimum_heating_seconds must not exceed "
                "maximum_heating_seconds"
            )

        return result


# =============================================================================
# 2. Baseline 配置
# =============================================================================
@dataclass(frozen=True)
class BaselineSettings:
    """定义每个循环共同、局部早期 Baseline 窗口的搜索规则。

    本类只定义搜索范围和合格阈值。
    真正的窗口滑动、稳定性检查和 baseline/residual 计算在 baseline.py 中完成。

    当前 baseline 的科学定位是：
    cycle_local_early_stable_proxy

    它是每个循环早期的稳定参考代理，不是人工或图像确认的绝对无霜真值。
    """

    # Baseline 只能在 frost_development 阶段搜索。
    # from_mapping() 会拒绝其他阶段，避免 recovery 或 defrost 被用作参考。
    stage: str = "frost_development"

    # The baseline is the first fixed window after recovery.  The older
    # search_* fields remain readable for existing configs, but are no longer
    # used to slide the window forward.
    baseline_seconds: int = 60

    # Baseline 搜索区间相对于 stable_heating_start 的起点和终点（分钟）。
    # 默认在结霜发展开始后的第 0～20 分钟内寻找候选窗口。
    search_start_minutes: int = 0
    search_end_minutes: int = 20

    # 每个候选 Baseline 窗口的长度（分钟）。
    window_minutes: int = 5

    # 滑动搜索候选窗口时，每次向后移动的步长（分钟）。
    # 默认每隔 1 分钟检查一个 5 分钟窗口。
    window_step_minutes: int = 1

    # 每个 required anchor 在候选窗口内必须达到的最低原始观测覆盖率。
    # 0.8 表示至少 80% 的时间点存在非空观测。
    minimum_observed_coverage: float = 0.8

    # 共同窗口必须同时满足稳定性要求的锚点通道。
    # 这些通道用于判断当前环境、负荷和控制是否足够稳定，
    # 而不是规定只有这些通道才能计算 baseline。
    required_anchor_channels: tuple[str, ...] = (
        "ambient_temperature",
        "water_in_temperature",
        "water_out_temperature",
        "compressor_frequency",
    )

    # 每个锚点在候选窗口内允许的最大标准差。
    # 使用 default_factory 是为了让每个 BaselineSettings 实例拥有独立字典，
    # 避免多个实例共享同一个可变对象。
    anchor_maximum_std: dict[str, float] = field(
        default_factory=lambda: {
            "ambient_temperature": 1.0,
            "water_in_temperature": 1.0,
            "water_out_temperature": 1.0,
            "compressor_frequency": 5.0,
        }
    )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> BaselineSettings:
        """将 YAML baseline mapping 转成经过校验的 BaselineSettings。"""

        # 确认 baseline 是 mapping，并将 key 统一转成字符串。
        mapping = _mapping(values, "baseline")

        # BaselineSettings 的 YAML 结构与 dataclass 字段一一对应，
        # 因此可以直接通过 dataclass 字段列表拒绝未知键。
        _validate_dataclass_keys(mapping, cls, "baseline")

        # tuple 和 dict 是复合字段，先取出原始值，再分别完成类型转换。
        anchors = mapping.get(
            "required_anchor_channels",
            cls().required_anchor_channels,
        )
        maximum_std = mapping.get(
            "anchor_maximum_std",
            cls().anchor_maximum_std,
        )

        result = cls(
            stage=str(mapping.get("stage", cls.stage)),
            baseline_seconds=int(
                mapping.get("baseline_seconds", cls.baseline_seconds)
            ),
            search_start_minutes=int(
                mapping.get(
                    "search_start_minutes",
                    cls.search_start_minutes,
                )
            ),
            search_end_minutes=int(
                mapping.get(
                    "search_end_minutes",
                    cls.search_end_minutes,
                )
            ),
            window_minutes=int(
                mapping.get("window_minutes", cls.window_minutes)
            ),
            window_step_minutes=int(
                mapping.get(
                    "window_step_minutes",
                    cls.window_step_minutes,
                )
            ),
            minimum_observed_coverage=float(
                mapping.get(
                    "minimum_observed_coverage",
                    cls.minimum_observed_coverage,
                )
            ),

            # 将 YAML list 转为不可变 tuple[str, ...]。
            required_anchor_channels=tuple(
                str(value) for value in anchors
            ),

            # 将 YAML mapping 的 key 标准化为 str，阈值标准化为 float。
            anchor_maximum_std={
                str(key): float(value)
                for key, value in maximum_std.items()
            },
        )

        # 当前 Baseline 算法只允许在 frost_development 中搜索。
        if result.stage != "frost_development":
            raise ValueError(
                "baseline stage must be frost_development"
            )

        if result.baseline_seconds <= 0:
            raise ValueError("baseline_seconds must be positive")

        # 搜索起点相对于 stable_heating_start，不能是负数。
        if result.search_start_minutes < 0:
            raise ValueError(
                "baseline search_start_minutes must be nonnegative"
            )

        # 搜索终点必须严格晚于搜索起点。
        if result.search_end_minutes <= result.search_start_minutes:
            raise ValueError(
                "baseline search_end_minutes must be later than "
                "search_start_minutes"
            )

        # 窗口长度和滑动步长必须严格为正数。
        if (
            result.window_minutes <= 0
            or result.window_step_minutes <= 0
        ):
            raise ValueError(
                "baseline window and step must be positive"
            )

        # 候选窗口必须能够完整放入搜索区间。
        if (
            result.window_minutes
            > result.search_end_minutes
            - result.search_start_minutes
        ):
            raise ValueError(
                "baseline window_minutes must fit within "
                "the search range"
            )

        # 覆盖率必须在闭区间 [0, 1] 中。
        _validate_fraction(
            "minimum_observed_coverage",
            result.minimum_observed_coverage,
        )

        # 标准差阈值可以为 0，但不能为负数。
        if any(
            value < 0
            for value in result.anchor_maximum_std.values()
        ):
            raise ValueError(
                "anchor_maximum_std values must be nonnegative"
            )

        return result


# =============================================================================
# 3. Process 配置
# =============================================================================


@dataclass(frozen=True)
class ProcessSettings:
    """定义 Prepared → Processed 阶段的处理规则。

    对应 process.py 的主要顺序：
    公共时间网格 → bucket coverage → bounded 缺失处理
    → 派生量 → Baseline/残差 → past-only 动态特征。
    """

    # Processed 数据的统一时间网格间隔（秒）。
    # 默认将原生 1 秒数据聚合为 10 秒 bucket。
    resample_interval_seconds: int = 10

    # continuous 通道在一个重采样 bucket 中的最低观测覆盖率。
    # 低于该比例时，该 bucket 的聚合值先置为 NaN，
    # 再由后续 bounded 缺失策略判断能否重建。
    minimum_continuous_bucket_coverage: float = 0.8

    # continuous 通道允许线性插值的最大实际缺失间隔（秒）。
    # 只有完整缺失段两端都有 observed 值，且两端时间差不超过该阈值时才插值。
    continuous_max_gap_seconds: float = 60.0

    # step/control 通道允许向前保持最后观测值的最长时间（秒）。
    # 每个缺失 bucket 都按其距最后 observed 值的真实时间差独立判断。
    control_max_gap_seconds: float = 30.0

    # 嵌套的共同 Baseline 搜索规则。
    # default_factory 保证每个 ProcessSettings 实例拥有独立 BaselineSettings。
    baseline: BaselineSettings = field(
        default_factory=BaselineSettings
    )

    # 为 analysis_candidate 通道生成动态特征的时间窗口（分钟）。
    # 对每个窗口生成 lag、delta 和 past-only rolling mean。
    feature_windows_minutes: tuple[int, ...] = (5, 10, 30)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> ProcessSettings:
        """将 YAML process mapping 转成经过校验的 ProcessSettings。"""

        mapping = _mapping(values, "process")

        # 这里不能直接使用 _validate_dataclass_keys()：
        # YAML 公共结构使用 process.features.windows_minutes，
        # 但 dataclass 内部直接保存为 feature_windows_minutes。
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

        # baseline 和 features 都是 process 下的嵌套 mapping。
        baseline_values = mapping.get("baseline", {})
        feature_values = mapping.get("features", {})

        # 防止用户将嵌套对象错误写成数字、字符串或列表。
        if (
            not isinstance(baseline_values, Mapping)
            or not isinstance(feature_values, Mapping)
        ):
            raise ValueError(
                "process.baseline and process.features "
                "must be mappings"
            )

        # features 当前只允许 windows_minutes，拒绝其他未知功能键。
        _validate_keys(
            feature_values,
            {"windows_minutes"},
            "process.features",
        )

        # 将 YAML list 转为不可变的分钟窗口 tuple。
        windows = tuple(
            int(value)
            for value in feature_values.get(
                "windows_minutes",
                list(cls.feature_windows_minutes),
            )
        )

        result = cls(
            resample_interval_seconds=int(
                mapping.get(
                    "resample_interval_seconds",
                    cls.resample_interval_seconds,
                )
            ),
            minimum_continuous_bucket_coverage=float(
                mapping.get(
                    "minimum_continuous_bucket_coverage",
                    cls.minimum_continuous_bucket_coverage,
                )
            ),
            continuous_max_gap_seconds=float(
                mapping.get(
                    "continuous_max_gap_seconds",
                    cls.continuous_max_gap_seconds,
                )
            ),
            control_max_gap_seconds=float(
                mapping.get(
                    "control_max_gap_seconds",
                    cls.control_max_gap_seconds,
                )
            ),

            # BaselineSettings 自己负责其内部字段和范围校验。
            baseline=BaselineSettings.from_mapping(
                baseline_values
            ),
            feature_windows_minutes=windows,
        )

        # 重采样时间间隔必须严格大于 0。
        if result.resample_interval_seconds <= 0:
            raise ValueError(
                "resample_interval_seconds must be positive"
            )

        # coverage 阈值必须在 [0, 1] 内。
        _validate_fraction(
            "minimum_continuous_bucket_coverage",
            result.minimum_continuous_bucket_coverage,
        )

        # 缺失处理阈值允许为 0，表示不允许对应类型的自动重建。
        _validate_nonnegative(
            "continuous_max_gap_seconds",
            result.continuous_max_gap_seconds,
        )
        _validate_nonnegative(
            "control_max_gap_seconds",
            result.control_max_gap_seconds,
        )

        # 至少需要一个动态特征窗口，并且每个窗口必须是正整数分钟。
        if (
            not result.feature_windows_minutes
            or any(
                value <= 0
                for value in result.feature_windows_minutes
            )
        ):
            raise ValueError(
                "feature windows must be positive"
            )

        return result


# =============================================================================
# 4. Analyze 配置
# =============================================================================


@dataclass(frozen=True)
class AnalysisSettings:
    """定义候选通道证据分析使用的目标、样本量和阈值。

    本类不训练机器学习模型。
    当前 analysis.py 生成透明的循环级趋势、未来性能关联和工况关联证据；
    最终 decision 目前只使用趋势循环数、趋势强度和方向一致性。
    """

    # 用于计算 future association 的性能目标列。
    # 默认目标是制热量相对本循环 Baseline 的残差。
    # 当前候选 decision 不直接使用该未来关联值。
    performance_target: str = (
        "heating_capacity__baseline_residual"
    )

    # future association 的预测时间跨度（分钟）。
    # analysis.py 使用同一循环和阶段内 timestamp + horizon 的精确匹配。
    future_horizon_minutes: int = 10

    # 一个候选通道获得正式趋势判断所需的最少有效循环数。
    minimum_valid_cycles: int = 3

    # 方向对齐后的循环趋势 Spearman 中位数阈值。
    # 正值表示符合该通道配置的 expected_frost_direction。
    minimum_trend_effect: float = 0.3

    # 出现正向对齐趋势的循环比例阈值。
    # 0.7 表示至少 70% 的有效循环方向符合预期。
    minimum_direction_consistency: float = 0.7

    # 单个循环内计算 Spearman 相关所需的最少有限数据点数。
    minimum_points_per_cycle: int = 6
    evidence: EvidencePolicy = field(default_factory=lambda: EvidencePolicy())

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> AnalysisSettings:
        """将 YAML analysis mapping 转成经过校验的 AnalysisSettings。"""

        mapping = _mapping(values, "analysis")

        # AnalysisSettings 的 YAML 键与 dataclass 字段一一对应。
        _validate_dataclass_keys(mapping, cls, "analysis")

        result = cls(
            performance_target=str(
                mapping.get(
                    "performance_target",
                    cls.performance_target,
                )
            ),
            future_horizon_minutes=int(
                mapping.get(
                    "future_horizon_minutes",
                    cls.future_horizon_minutes,
                )
            ),
            minimum_valid_cycles=int(
                mapping.get(
                    "minimum_valid_cycles",
                    cls.minimum_valid_cycles,
                )
            ),
            minimum_trend_effect=float(
                mapping.get(
                    "minimum_trend_effect",
                    cls.minimum_trend_effect,
                )
            ),
            minimum_direction_consistency=float(
                mapping.get(
                    "minimum_direction_consistency",
                    cls.minimum_direction_consistency,
                )
            ),
            minimum_points_per_cycle=int(
                mapping.get(
                    "minimum_points_per_cycle",
                    cls.minimum_points_per_cycle,
                )
            ),
            evidence=EvidencePolicy.from_mapping(mapping.get("evidence", {})),
        )

        # future horizon 必须严格大于 0。
        if result.future_horizon_minutes <= 0:
            raise ValueError(
                "future_horizon_minutes must be positive"
            )

        # 至少需要一个有效循环；相关性至少需要两个点才有定义。
        if (
            result.minimum_valid_cycles <= 0
            or result.minimum_points_per_cycle < 2
        ):
            raise ValueError(
                "analysis minimum counts are too small"
            )

        # 方向一致性是比例，必须位于 [0, 1]。
        _validate_fraction(
            "minimum_direction_consistency",
            result.minimum_direction_consistency,
        )

        # Spearman 绝对范围是 [-1, 1]。
        # 这里使用方向对齐后的正阈值，因此限制在 [0, 1]。
        if (
            result.minimum_trend_effect < 0
            or result.minimum_trend_effect > 1
        ):
            raise ValueError(
                "minimum_trend_effect must be within [0, 1]"
            )

        return result


# =============================================================================
# 5. Pipeline 最终配置对象
# =============================================================================


@dataclass(frozen=True)
class EvidencePolicy:
    """Date-independent rules for the cycle evidence analysis."""

    min_segment_coverage: float = 0.8
    min_segment_points: int = 12
    min_pair_coverage: float = 0.8
    min_valid_pairs: int = 30
    min_valid_cycles: int = 3
    horizons_minutes: tuple[int, ...] = (5, 10, 20)
    targets: tuple[str, ...] = ("heating_capacity", "cop")
    primary_target: str = "heating_capacity"
    primary_target_type: str = "future_change"
    primary_horizon_minutes: int = 10
    primary_feature_variant: str = "residual_level"
    lead_target: str = "heating_capacity"
    auto_reference_window_minutes: int = 5
    auto_reference_min_observed_fraction: float = 0.8
    auto_reference_max_gap_seconds: float = 60.0
    onset_window_seconds: int = 60
    onset_mad_multiplier: float = 3.0
    onset_persistence_seconds: int = 60
    similarity_threshold: float = 0.85

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> EvidencePolicy:
        mapping = _mapping(values, "analysis.evidence")
        _validate_dataclass_keys(mapping, cls, "analysis.evidence")
        horizons = tuple(
            int(value) for value in mapping.get("horizons_minutes", cls.horizons_minutes)
        )
        targets = tuple(str(value) for value in mapping.get("targets", cls.targets))
        result = cls(
            min_segment_coverage=float(
                mapping.get("min_segment_coverage", cls.min_segment_coverage)
            ),
            min_segment_points=int(mapping.get("min_segment_points", cls.min_segment_points)),
            min_pair_coverage=float(mapping.get("min_pair_coverage", cls.min_pair_coverage)),
            min_valid_pairs=int(mapping.get("min_valid_pairs", cls.min_valid_pairs)),
            min_valid_cycles=int(mapping.get("min_valid_cycles", cls.min_valid_cycles)),
            horizons_minutes=horizons,
            targets=targets,
            primary_target=str(mapping.get("primary_target", cls.primary_target)),
            primary_target_type=str(mapping.get("primary_target_type", cls.primary_target_type)),
            primary_horizon_minutes=int(
                mapping.get("primary_horizon_minutes", cls.primary_horizon_minutes)
            ),
            primary_feature_variant=str(
                mapping.get("primary_feature_variant", cls.primary_feature_variant)
            ),
            lead_target=str(mapping.get("lead_target", cls.lead_target)),
            auto_reference_window_minutes=int(
                mapping.get(
                    "auto_reference_window_minutes",
                    cls.auto_reference_window_minutes,
                )
            ),
            auto_reference_min_observed_fraction=float(
                mapping.get(
                    "auto_reference_min_observed_fraction",
                    cls.auto_reference_min_observed_fraction,
                )
            ),
            auto_reference_max_gap_seconds=float(
                mapping.get(
                    "auto_reference_max_gap_seconds",
                    cls.auto_reference_max_gap_seconds,
                )
            ),
            onset_window_seconds=int(mapping.get("onset_window_seconds", cls.onset_window_seconds)),
            onset_mad_multiplier=float(
                mapping.get("onset_mad_multiplier", cls.onset_mad_multiplier)
            ),
            onset_persistence_seconds=int(
                mapping.get("onset_persistence_seconds", cls.onset_persistence_seconds)
            ),
            similarity_threshold=float(
                mapping.get("similarity_threshold", cls.similarity_threshold)
            ),
        )
        _validate_evidence_policy(result, horizons, targets)
        return result


def _validate_evidence_policy(
    result: EvidencePolicy, horizons: tuple[int, ...], targets: tuple[str, ...]
) -> None:
    _validate_evidence_thresholds(result)
    _validate_evidence_targets(result, horizons, targets)
    _validate_evidence_windows(result)


def _validate_evidence_thresholds(result: EvidencePolicy) -> None:
    _validate_fraction("analysis.evidence.min_segment_coverage", result.min_segment_coverage)
    _validate_fraction("analysis.evidence.min_pair_coverage", result.min_pair_coverage)
    _validate_fraction(
        "analysis.evidence.auto_reference_min_observed_fraction",
        result.auto_reference_min_observed_fraction,
    )
    _validate_fraction("analysis.evidence.similarity_threshold", result.similarity_threshold)
    if result.min_segment_points < 2 or result.min_valid_pairs < 1:
        raise ValueError("analysis.evidence minimum counts are too small")
    if result.min_valid_cycles < 1:
        raise ValueError("analysis.evidence.min_valid_cycles must be positive")


def _validate_evidence_targets(
    result: EvidencePolicy, horizons: tuple[int, ...], targets: tuple[str, ...]
) -> None:
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("analysis.evidence horizons must be positive")
    if len(set(horizons)) != len(horizons):
        raise ValueError("analysis.evidence horizons must be unique")
    if not targets or any(not value for value in targets):
        raise ValueError("analysis.evidence targets must not be empty")
    if result.primary_target not in targets:
        raise ValueError("analysis.evidence.primary_target must be a configured target")
    if result.lead_target not in targets:
        raise ValueError("analysis.evidence.lead_target must be a configured target")
    if result.primary_horizon_minutes not in horizons:
        raise ValueError("analysis.evidence.primary_horizon_minutes must be configured")


def _validate_evidence_windows(result: EvidencePolicy) -> None:
    if result.auto_reference_window_minutes <= 0:
        raise ValueError("analysis.evidence auto reference window must be positive")
    if result.auto_reference_max_gap_seconds < 0:
        raise ValueError("analysis.evidence auto reference gap must be nonnegative")
    if result.onset_window_seconds <= 0 or result.onset_persistence_seconds <= 0:
        raise ValueError("analysis.evidence onset windows must be positive")
    if result.onset_mad_multiplier <= 0:
        raise ValueError("analysis.evidence onset MAD multiplier must be positive")


@dataclass(frozen=True)
class EvidenceSettings:
    """Evidence policy plus the candidate registry path used by Analyze."""

    channels_path: Path
    policy: EvidencePolicy


def validate_evidence_timing(policy: EvidencePolicy, grid_interval_seconds: int) -> None:
    """Validate every Evidence duration against the run's actual grid."""
    from .evidence_cycle import duration_buckets

    durations = [
        policy.auto_reference_window_minutes * 60,
        5 * 60,
        policy.onset_window_seconds,
        policy.onset_persistence_seconds,
        *(horizon * 60 for horizon in policy.horizons_minutes),
    ]
    for duration in durations:
        duration_buckets(duration, grid_interval_seconds)


def load_evidence_settings(path: Path, *, allow_date_config: bool) -> EvidenceSettings:
    """Load date-independent evidence settings or project them from one date config."""
    config_path = path.resolve()
    loaded = _load_yaml_mapping(config_path, "evidence config")
    date_keys = {
        "experiment_id",
        "experiment_date",
        "input_dir",
        "camera_roles",
        "overrides",
    }
    if not allow_date_config and date_keys.intersection(loaded):
        raise ValueError("date-specific facts are not allowed in batch evidence config")
    if "schema_version" in loaded:
        if not allow_date_config:
            raise ValueError("date-specific config is not allowed in batch evidence config")
        config = load_config(config_path)
        return EvidenceSettings(config.channels_path, config.analysis.evidence)

    channels_value = loaded.get("channels_path")
    analysis_value = _mapping(loaded.get("analysis", {}), "analysis")
    policy = EvidencePolicy.from_mapping(analysis_value.get("evidence", {}))
    if channels_value is None:
        raise ValueError("evidence config requires channels_path")
    channels_path = _resolve_path(config_path.parent, channels_value)
    return EvidenceSettings(channels_path, policy)


@dataclass(frozen=True)
class Config:
    """一次 Pipeline 运行最终使用的完整、不可变配置。

    字段来源分为四类：
    1. 日期实验事实：experiment_id、experiment_date、input_dir、
       expected_sensor_interval_seconds、camera_roles；
    2. 共享输入格式与通道事实：channels_path、sensor_globs、
       image_extensions、timestamp_column；
    3. 科学处理规则：cycles、process、analysis；
    4. 配置来源追踪：config_path、defaults_path。

    load_config() 是正式构造入口。
    """

    # 仓库根目录，用于解析仓库相对路径和记录运行来源。
    project_root: Path

    # 一次实验的稳定唯一标识，例如 exp_20260715。
    experiment_id: str

    # 实验日期，正式格式必须为 ISO YYYY-MM-DD。
    experiment_date: str

    # 原始实验数据目录；Pipeline 将其视为只读输入。
    input_dir: Path

    # 共享 channels.yaml 的绝对路径。
    channels_path: Path

    # 在 input_dir 根目录中发现传感器文本文件的 glob 规则。
    sensor_globs: tuple[str, ...]

    # 允许发现的图片扩展名，统一为小写且带前导点。
    image_extensions: tuple[str, ...]

    # 原始传感器文件中的时间戳列名。
    timestamp_column: str

    # 原始传感器理论采样间隔（秒），用于估算每个重采样桶的期望点数。
    expected_sensor_interval_seconds: int

    # 图片时间戳与传感器时间戳匹配时允许的最大偏移（秒）。
    image_match_tolerance_seconds: float

    # EDF 双 SHT40 顺序配对允许的最大时间差（秒）。
    edf_pair_tolerance_seconds: float

    # 循环切分规则。
    cycles: CycleSettings

    # 重采样、缺失处理、Baseline 和动态特征规则。
    process: ProcessSettings

    # 候选通道证据分析规则。
    analysis: AnalysisSettings

    # 当前日期配置文件的绝对路径，用于 provenance 和 hash。
    config_path: Path | None = None

    # 当前共享 defaults.yaml 的绝对路径，用于 provenance 和 hash。
    defaults_path: Path | None = None

    # 原始相机目录 ID 到物理角色的映射。
    # 例如 {"camera_01": "front"}。
    camera_roles: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """兼容测试或内部代码直接传入 Mapping，并检查跨字段约束。

        正式 load_config() 通常已经生成三个 Settings 对象；
        这里仍接受 Mapping，便于测试构造和保持 Config 自身边界稳定。
        """

        # 先保留原始类型，便于 mypy 理解后续 Mapping 分支。
        raw_cycles: Any = self.cycles
        raw_process: Any = self.process
        raw_analysis: Any = self.analysis

        # frozen dataclass 不能普通赋值。
        # object.__setattr__ 是 __post_init__ 中完成类型规范化的受控方式。
        if isinstance(raw_cycles, Mapping):
            object.__setattr__(
                self,
                "cycles",
                CycleSettings.from_mapping(raw_cycles),
            )

        if isinstance(raw_process, Mapping):
            object.__setattr__(
                self,
                "process",
                ProcessSettings.from_mapping(raw_process),
            )

        if isinstance(raw_analysis, Mapping):
            object.__setattr__(
                self,
                "analysis",
                AnalysisSettings.from_mapping(raw_analysis),
            )

        # 公共重采样间隔必须是原生采样间隔的整数倍，
        # 否则每个 bucket 的期望原始点数无法稳定定义。
        #
        # 注意：load_config() 会先检查 expected_sensor_interval_seconds > 0。
        # 若项目要求直接 Config(...) 构造也完全安全，后续可将正数检查
        # 一并移入此处，避免 0 导致 ZeroDivisionError。
        if (
            self.process.resample_interval_seconds
            % self.expected_sensor_interval_seconds
            != 0
        ):
            raise ValueError(
                "resample_interval_seconds must be divisible by "
                "expected_sensor_interval_seconds"
            )

        _validate_positive(
            "edf_pair_tolerance_seconds",
            self.edf_pair_tolerance_seconds,
        )


# =============================================================================
# 6. schema v2 配置加载入口
# =============================================================================


def load_config(path: Path) -> Config:
    """读取一个 schema v2 日期配置，并解析其共享 defaults。

    配置解析顺序：
    日期实验事实
        → 共享 defaults
        → 日期 overrides
        → 嵌套结构和数值校验
        → 类型化 Config

    日期 YAML 只保存日期事实和少量例外；
    defaults.yaml 保存跨日期共享的输入格式和科学处理规则。
    """

    # 将调用者传入的相对路径立即规范化为绝对路径。
    config_path = path.resolve()

    # 读取日期 YAML，并确保其顶层是 mapping。
    loaded = _load_yaml_mapping(
        config_path,
        "experiment config",
    )

    # schema v2 日期文件只允许这些顶层键。
    # 未知键通常意味着拼写错误或旧 schema 残留，因此立即拒绝。
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

    # 最终代码只支持 schema v2，不保留 legacy loader。
    if loaded.get("schema_version") != 2:
        raise ValueError(
            "only config schema_version 2 is supported"
        )

    # overrides 是可选项；其余字段是每个日期文件必须明确提供的实验事实。
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
        raise ValueError(
            f"config missing keys: {missing}"
        )

    # defaults_path 相对于“日期配置文件所在目录”解析，
    # 而不是相对于当前工作目录或仓库根目录。
    defaults_path = (
        config_path.parent
        / str(loaded["defaults_path"])
    ).resolve()

    # 读取共享 defaults.yaml。
    defaults = _load_yaml_mapping(
        defaults_path,
        "defaults",
    )

    # defaults.yaml 只允许保存跨日期共享的事实和规则。
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

    # 日期文件可选地提供 overrides；没有时使用空 mapping。
    overrides = loaded.get("overrides", {})

    # overrides 必须保持 YAML mapping 结构。
    if not isinstance(overrides, Mapping):
        raise ValueError(
            "overrides must be a YAML mapping"
        )

    # 将日期 overrides 递归覆盖到 defaults。
    # 只能替换 defaults 中已存在的键，不能通过 overrides 创建新参数。
    resolved = _merge_existing_keys(
        defaults,
        overrides,
    )

    # input_format 是共享 defaults 中的一个嵌套 mapping。
    input_format = _mapping(
        resolved["input_format"],
        "input_format",
    )

    # 输入格式目前固定由三项组成。
    _validate_keys(
        input_format,
        {
            "sensor_globs",
            "image_extensions",
            "timestamp_column",
            "edf",
        },
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
    # 日期值统一转为字符串后，再严格检查 ISO 格式。
    experiment_date = str(
        loaded["experiment_date"]
    )
    if not _is_iso_date(experiment_date):
        raise ValueError(
            "experiment_date must use ISO YYYY-MM-DD format"
        )

    # 从日期配置向父目录逐级寻找 pyproject.toml，
    # 以此确定仓库根目录，而不是依赖命令执行时的 cwd。
    project_root = _find_project_root(config_path)

    # 将共享 defaults + 日期 overrides 中的 process 规则转换为类型化对象。
    process = ProcessSettings.from_mapping(
        _mapping(
            resolved["process"],
            "process",
        )
    )

    # 将 analysis 规则转换为类型化对象。
    analysis = AnalysisSettings.from_mapping(
        _mapping(
            resolved["analysis"],
            "analysis",
        )
    )

    # future horizon 必须恰好落在 Processed 时间网格上。
    # 例如 10 分钟 horizon 和 10 秒网格可精确对齐。
    if (
        analysis.future_horizon_minutes * 60
        % process.resample_interval_seconds
        != 0
    ):
        raise ValueError(
            "future_horizon_minutes must align with "
            "the resample interval"
        )

    # 原始采样间隔是日期实验事实，因此从日期 YAML 读取。
    expected_interval = int(
        loaded["expected_sensor_interval_seconds"]
    )

    # 图片匹配容差属于共享处理规则，但可由日期 overrides 覆盖。
    image_tolerance = float(
        resolved["image_match_tolerance_seconds"]
    )

    # 原始采样间隔必须严格大于 0。
    _validate_positive(
        "expected_sensor_interval_seconds",
        expected_interval,
    )

    # 重采样间隔必须是原始采样间隔的整数倍。
    # 例如 1 秒原始采样可聚合为 10 秒；3 秒不能完整划入 10 秒期望点数。
    if (
        process.resample_interval_seconds
        % expected_interval
        != 0
    ):
        raise ValueError(
            "resample_interval_seconds must be divisible by "
            "expected_sensor_interval_seconds"
        )

    # 图片匹配容差可以为 0，但不能为负数。
    _validate_nonnegative(
        "image_match_tolerance_seconds",
        image_tolerance,
    )
    _validate_positive(
        "edf_pair_tolerance_seconds",
        edf_pair_tolerance,
    )

    # 时间戳列名不能是空字符串或纯空白。
    if not str(input_format["timestamp_column"]).strip():
        raise ValueError(
            "timestamp_column must not be empty"
        )

    # 验证并标准化日期文件中的相机 ID → 物理角色映射。
    camera_roles = _camera_roles(
        loaded["camera_roles"]
    )

    # 构造一次运行唯一使用的最终 Config。
    return Config(
        project_root=project_root,
        experiment_id=str(
            loaded["experiment_id"]
        ),
        experiment_date=experiment_date,

        # input_dir 相对于仓库根目录解析。
        input_dir=_resolve_path(
            project_root,
            loaded["input_dir"],
        ),

        # channels_path 相对于 defaults.yaml 所在目录解析。
        channels_path=_resolve_path(
            defaults_path.parent,
            resolved["channels_path"],
        ),

        # 将 YAML list 转为不可变 tuple[str, ...]。
        sensor_globs=_tuple_strings(
            input_format["sensor_globs"],
            "sensor_globs",
        ),

        # 同时将扩展名规范化为小写且带前导点。
        image_extensions=_image_extensions(
            input_format["image_extensions"]
        ),

        timestamp_column=str(
            input_format["timestamp_column"]
        ),
        expected_sensor_interval_seconds=expected_interval,
        image_match_tolerance_seconds=image_tolerance,
        edf_pair_tolerance_seconds=edf_pair_tolerance,

        # cycles 同样使用共享 defaults + 日期 overrides 的最终结果。
        cycles=CycleSettings.from_mapping(
            _mapping(
                resolved["cycles"],
                "cycles",
            )
        ),
        process=process,
        analysis=analysis,

        # 保留两份源配置文件路径，用于 provenance 和 manifest。
        config_path=config_path,
        defaults_path=defaults_path,
        camera_roles=camera_roles,
    )


# =============================================================================
# 7. 最终生效配置的规范化表示与哈希
# =============================================================================


def resolved_config_mapping(
    config: Config,
) -> dict[str, Any]:
    """返回真正影响运行结果的最终生效配置。

    该表示主动排除 config_path、defaults_path 等“来源位置”信息，
    使配置哈希只反映有效参数，而不因仓库被移动到不同绝对路径而变化。

    input_dir 会尽量写成仓库相对路径；若它位于仓库外，则保留绝对路径。
    """

    return {
        # 日期实验身份。
        "experiment_id": config.experiment_id,
        "experiment_date": config.experiment_date,

        # 尽可能使用仓库相对路径，避免机器绝对路径影响配置哈希。
        "input_dir": _relative_path(
            config.input_dir,
            config.project_root,
        ),

        # 日期事实。
        "expected_sensor_interval_seconds": (
            config.expected_sensor_interval_seconds
        ),
        "camera_roles": dict(config.camera_roles),

        # 将内部扁平字段恢复为 YAML 对外结构。
        "input_format": {
            "sensor_globs": list(config.sensor_globs),
            "image_extensions": list(
                config.image_extensions
            ),
            "timestamp_column": config.timestamp_column,
            "edf": {
                "pair_tolerance_seconds": config.edf_pair_tolerance_seconds,
            },
        },

        # 共享但可能被日期 overrides 覆盖的规则。
        "image_match_tolerance_seconds": (
            config.image_match_tolerance_seconds
        ),

        # asdict() 递归将 dataclass 转为普通字典。
        "cycles": asdict(config.cycles),

        # ProcessSettings 内部字段 feature_windows_minutes
        # 在有效配置表示中恢复为 features.windows_minutes。
        "process": {
            "resample_interval_seconds": (
                config.process.resample_interval_seconds
            ),
            "minimum_continuous_bucket_coverage": (
                config.process.minimum_continuous_bucket_coverage
            ),
            "continuous_max_gap_seconds": (
                config.process.continuous_max_gap_seconds
            ),
            "control_max_gap_seconds": (
                config.process.control_max_gap_seconds
            ),
            "baseline": asdict(
                config.process.baseline
            ),
            "features": {
                "windows_minutes": list(
                    config.process.feature_windows_minutes
                )
            },
        },

        # AnalysisSettings 可直接递归转换。
        "analysis": asdict(config.analysis),
    }


def resolved_config_sha256(config: Config) -> str:
    """对最终生效配置生成稳定 SHA-256。

    只要有效参数相同，即使：
    - YAML 键顺序不同；
    - 配置文件位于不同绝对目录；
    - JSON 默认空格格式不同；
    也应得到相同哈希。
    """

    # 将最终配置以稳定 JSON 规则序列化。
    payload = json.dumps(
        resolved_config_mapping(config),

        # 对所有 mapping key 排序，消除 YAML/字典插入顺序影响。
        sort_keys=True,

        # 使用紧凑分隔符，消除无意义空格差异。
        separators=(",", ":"),

        # 保留中文字符本身，不转义为 \uXXXX。
        ensure_ascii=False,

        # 配置中不允许 NaN/Infinity，避免生成非标准 JSON。
        allow_nan=False,
    ).encode("utf-8")

    # 返回十六进制 SHA-256 字符串。
    return hashlib.sha256(payload).hexdigest()


# =============================================================================
# 8. 路径、YAML、键名和合并辅助函数
# =============================================================================


def _find_project_root(config_path: Path) -> Path:
    """从配置文件目录向上寻找包含 pyproject.toml 的仓库根目录。"""

    # 先检查配置文件所在目录，再依次检查所有父目录。
    for parent in (
        config_path.parent,
        *config_path.parents,
    ):
        if (parent / "pyproject.toml").is_file():
            return parent

    # 不依赖当前工作目录；找不到明确仓库根目录就立即失败。
    raise FileNotFoundError(
        "could not find project root containing pyproject.toml"
    )


def find_project_root(config_path: Path) -> Path | None:
    """Return the nearest project root, or None when no repository root exists."""
    resolved = config_path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def _resolve_path(root: Path, value: Any) -> Path:
    """将配置路径规范化为绝对路径。

    若 value 已经是绝对路径，直接 resolve；
    否则将其相对于调用者指定的 root 解析。
    """

    path = Path(str(value))
    return (
        path.resolve()
        if path.is_absolute()
        else (root / path).resolve()
    )


def _load_yaml_mapping(
    path: Path,
    name: str,
) -> dict[str, Any]:
    """读取一个 YAML 文件，并要求其顶层必须是 mapping。"""

    # 空 YAML 文件通过“or {}”规范化为空字典，
    # 随后由 required key 检查给出更明确的错误。
    loaded = (
        yaml.safe_load(
            path.read_text(encoding="utf-8")
        )
        or {}
    )

    if not isinstance(loaded, Mapping):
        raise ValueError(
            f"{name} must be a YAML mapping"
        )

    # YAML key 理论上可以不是字符串；配置合同统一将 key 转为 str。
    return {
        str(key): value
        for key, value in loaded.items()
    }


def _validate_keys(
    values: Mapping[str, Any],
    allowed: set[str],
    name: str,
) -> None:
    """拒绝指定 mapping 中所有不在 allowed 集合里的未知键。"""

    # 这里只检查“额外键”；必填键由调用者单独检查。
    unknown = sorted(set(values) - allowed)

    if unknown:
        raise ValueError(
            f"{name} contains unknown keys: {unknown}"
        )


def _validate_dataclass_keys(
    values: Mapping[str, Any],
    cls: type[Any],
    name: str,
) -> None:
    """以 dataclass 字段名作为 allowed key 集合。"""

    # fields(cls) 返回 dataclass 的字段元数据；
    # item.name 提取每个公开配置字段名。
    _validate_keys(
        values,
        {item.name for item in fields(cls)},
        name,
    )


def _merge_existing_keys(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """递归覆盖 defaults 中已经存在的键。

    规则：
    1. overrides 不允许新增 defaults 中不存在的键；
    2. mapping 只能被 mapping 覆盖，不能被标量替换，反之亦然；
    3. list/tuple 等非 mapping 值整体替换，不执行追加或逐项合并；
    4. 使用深拷贝，避免修改调用者传入的 defaults 或 overrides。
    """

    # 先深拷贝 base，确保后续递归写入不污染原始 defaults。
    result = copy.deepcopy(dict(base))

    for key, value in overrides.items():
        # prefix 用于产生完整错误路径，例如 process.baseline.window_minutes。
        name = (
            f"{prefix}.{key}"
            if prefix
            else str(key)
        )

        # overrides 只能覆盖已经存在的正式配置键。
        if key not in result:
            raise ValueError(
                f"unknown override key: {name}"
            )

        existing = result[key]

        # 两侧都是 mapping 时递归合并其中已有键。
        if (
            isinstance(existing, Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _merge_existing_keys(
                existing,
                value,
                name,
            )

        # 一侧是 mapping、另一侧不是，说明用户改变了配置结构。
        elif (
            isinstance(existing, Mapping)
            != isinstance(value, Mapping)
        ):
            raise ValueError(
                "override mapping shape does not match: "
                f"{name}"
            )

        # 标量、字符串、列表和 tuple 等值都按整体替换处理。
        else:
            result[key] = copy.deepcopy(value)

    return result


def _camera_roles(value: Any) -> dict[str, str]:
    """验证并标准化 camera ID → physical role 映射。"""

    # camera_roles 必须是 mapping。
    roles = _mapping(value, "camera_roles")

    # 将 camera ID 和 role 都统一转成字符串。
    result = {
        str(camera): str(role)
        for camera, role in roles.items()
    }

    # 不允许空 camera ID 或空物理角色。
    if any(
        not camera.strip() or not role.strip()
        for camera, role in result.items()
    ):
        raise ValueError(
            "camera_roles contains an empty camera ID or role"
        )

    # 当前下游按 role 生成 image_<role>_* 三列，
    # 因此一个 role 只能由一个 camera ID 提供，避免列语义冲突。
    if len(result.values()) != len(set(result.values())):
        raise ValueError(
            "two camera IDs cannot map to the same role"
        )

    return result


def _relative_path(path: Path, root: Path) -> str:
    """尽可能返回 path 相对于 root 的 POSIX 路径。"""

    try:
        # 仓库内部路径写成相对路径，使有效配置不依赖机器绝对目录。
        return (
            path.resolve()
            .relative_to(root.resolve())
            .as_posix()
        )
    except ValueError:
        # path 位于 root 外部时无法相对化，保留规范化绝对路径。
        return path.resolve().as_posix()


def _tuple_strings(
    value: Any,
    name: str,
) -> tuple[str, ...]:
    """将非空 YAML 字符串列表转换为 tuple[str, ...]。"""

    # 必须是非空 list，且每一项转成字符串后不能是空白。
    if (
        not isinstance(value, list)
        or not value
        or any(
            not str(item).strip()
            for item in value
        )
    ):
        raise ValueError(
            f"{name} must be a non-empty list of strings"
        )

    # tuple 不可变，更适合 frozen 配置对象。
    return tuple(str(item) for item in value)


def _image_extensions(value: Any) -> tuple[str, ...]:
    """规范化图片扩展名。

    例如：
    ["JPG", ".Png"] → (".jpg", ".png")
    """

    return tuple(
        # 已有前导点则只转小写；没有则自动补 "."。
        item.lower()
        if item.startswith(".")
        else f".{item.lower()}"
        for item in _tuple_strings(
            value,
            "image_extensions",
        )
    )


def _mapping(
    value: Any,
    name: str,
) -> dict[str, Any]:
    """检查一个已经加载的嵌套配置值是否为 mapping。"""

    if not isinstance(value, Mapping):
        raise ValueError(
            f"{name} must be a YAML mapping"
        )

    # 与顶层 YAML loader 一致，统一将 key 转为字符串。
    return {
        str(key): item
        for key, item in value.items()
    }


def _is_iso_date(value: str) -> bool:
    """严格判断字符串是否为规范 ISO 日期 YYYY-MM-DD。"""
    try:
        # fromisoformat() 负责解析；
        # 再转回 isoformat()，拒绝非规范但可被宽松解析的表示。
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def is_iso_date(value: str) -> bool:
    """公开兼容入口，供 IO 和批量 evidence 配置校验复用。"""
    return _is_iso_date(value)


def _validate_positive(
    name: str,
    value: float,
) -> None:
    """要求数值严格大于 0。"""

    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name} must be positive"
        )


def _validate_nonnegative(
    name: str,
    value: float,
) -> None:
    """要求数值大于或等于 0。"""

    if value < 0:
        raise ValueError(
            f"{name} must be nonnegative"
        )


def _validate_fraction(
    name: str,
    value: float,
) -> None:
    """要求比例值位于闭区间 [0, 1]。"""

    if not 0 <= value <= 1:
        raise ValueError(
            f"{name} must be within [0, 1]"
        )
