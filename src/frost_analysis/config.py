"""Dataset construction configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CycleSettings:
    """保存 cycles.py 切分制热—结霜—除霜循环时使用的阈值。

    本类只保存阈值；状态去抖、事件识别和循环切分由 cycles.py 完成。
    """

    defrost_channel: str = "defrost_active"

    # 允许自动桥接的除霜状态缺失区间上限（秒）。
    # 只有缺失段两端状态一致且总间隔不超过该值时，cycles.py 才允许补齐状态。
    # 0.0 表示任何除霜状态缺失均不自动推断。
    maximum_state_gap_seconds: float = 0.0

    # 抑制短暂 ON/OFF 抖动，避免将瞬时跳变识别为除霜事件。
    debounce_seconds: float = 20.0

    # 以下持续时间只用于构建质量判定，不定义机组控制行为。
    minimum_defrost_seconds: float = 60.0
    maximum_defrost_seconds: float = 1200.0

    minimum_heating_seconds: float = 1800.0
    maximum_heating_seconds: float = 21600.0

    # “stable”是固定阶段划分假设，不代表代码通过数据检测到系统已经稳定。
    stable_heating_seconds: float = 180.0

    # 除霜信号前，压机频率出现该幅度的单步下降即视为除霜准备开始。
    defrost_preparation_setpoint_drop_hz: float = 10.0
    defrost_preparation_lookback_seconds: float = 120.0

    operating_mode_channel: str = ""

    required_operating_mode: str = "3"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CycleSettings:
        """Parse and validate cycle construction thresholds."""
        mapping = _mapping(values, "cycles")
        _validate_dataclass_keys(mapping, cls, "cycles")
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
            defrost_preparation_setpoint_drop_hz=float(
                mapping.get(
                    "defrost_preparation_setpoint_drop_hz",
                    cls.defrost_preparation_setpoint_drop_hz,
                )
            ),
            defrost_preparation_lookback_seconds=float(
                mapping.get(
                    "defrost_preparation_lookback_seconds",
                    cls.defrost_preparation_lookback_seconds,
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

        _validate_nonnegative(
            "maximum_state_gap_seconds",
            result.maximum_state_gap_seconds,
        )

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
        _validate_positive(
            "defrost_preparation_setpoint_drop_hz",
            result.defrost_preparation_setpoint_drop_hz,
        )
        _validate_positive(
            "defrost_preparation_lookback_seconds",
            result.defrost_preparation_lookback_seconds,
        )

        _validate_nonnegative(
            "stable_heating_seconds",
            result.stable_heating_seconds,
        )

        if result.minimum_defrost_seconds > result.maximum_defrost_seconds:
            raise ValueError(
                "minimum_defrost_seconds must not exceed "
                "maximum_defrost_seconds"
            )

        if result.minimum_heating_seconds > result.maximum_heating_seconds:
            raise ValueError(
                "minimum_heating_seconds must not exceed "
                "maximum_heating_seconds"
            )

        return result


@dataclass(frozen=True)
class BaselineSettings:
    """Cycle-local early stable proxy, not verified frost-free ground truth."""

    # Recovery and defrost observations cannot define the reference window.
    stage: str = "frost_development"

    baseline_seconds: int = 60

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
        """Parse and validate the active baseline method."""
        mapping = _mapping(values, "baseline")
        _validate_dataclass_keys(mapping, cls, "baseline")
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
            minimum_observed_coverage=float(
                mapping.get(
                    "minimum_observed_coverage",
                    cls.minimum_observed_coverage,
                )
            ),
            required_anchor_channels=tuple(
                str(value) for value in anchors
            ),
            anchor_maximum_std={
                str(key): float(value)
                for key, value in maximum_std.items()
            },
        )

        if result.stage != "frost_development":
            raise ValueError(
                "baseline stage must be frost_development"
            )

        if result.baseline_seconds <= 0:
            raise ValueError("baseline_seconds must be positive")

        _validate_fraction(
            "minimum_observed_coverage",
            result.minimum_observed_coverage,
        )

        if any(
            value < 0
            for value in result.anchor_maximum_std.values()
        ):
            raise ValueError(
                "anchor_maximum_std values must be nonnegative"
            )

        return result


@dataclass(frozen=True)
class ProcessSettings:
    """定义 Prepared → Processed 阶段的处理规则。

    顺序为公共时间网格、bounded 缺失处理、派生量、Baseline/残差。
    """

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

    baseline: BaselineSettings = field(
        default_factory=BaselineSettings
    )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> ProcessSettings:
        """Parse and validate Process settings."""

        mapping = _mapping(values, "process")

        _validate_keys(
            mapping,
            {
                "resample_interval_seconds",
                "minimum_continuous_bucket_coverage",
                "continuous_max_gap_seconds",
                "control_max_gap_seconds",
                "baseline",
            },
            "process",
        )

        baseline_values = mapping.get("baseline", {})

        if not isinstance(baseline_values, Mapping):
            raise ValueError("process.baseline must be a mapping")

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
            baseline=BaselineSettings.from_mapping(
                baseline_values
            ),
        )

        if result.resample_interval_seconds <= 0:
            raise ValueError(
                "resample_interval_seconds must be positive"
            )

        _validate_fraction(
            "minimum_continuous_bucket_coverage",
            result.minimum_continuous_bucket_coverage,
        )

        _validate_nonnegative(
            "continuous_max_gap_seconds",
            result.continuous_max_gap_seconds,
        )
        _validate_nonnegative(
            "control_max_gap_seconds",
            result.control_max_gap_seconds,
        )

        return result


@dataclass(frozen=True)
class Config:
    """One experiment's Raw-to-Dataset settings."""

    project_root: Path
    experiment_id: str
    experiment_date: str
    input_dir: Path
    channels_path: Path
    sensor_globs: tuple[str, ...]
    image_extensions: tuple[str, ...]
    timestamp_column: str

    # 原始传感器理论采样间隔（秒），用于估算每个重采样桶的期望点数。
    expected_sensor_interval_seconds: int

    # 图片时间戳与传感器时间戳匹配时允许的最大偏移（秒）。
    image_match_tolerance_seconds: float

    # EDF 双 SHT40 顺序配对允许的最大时间差（秒）。
    edf_pair_tolerance_seconds: float

    # 循环切分规则。
    cycles: CycleSettings

    # 重采样、bounded 缺失处理、派生量和 Baseline 规则。
    process: ProcessSettings

    def __post_init__(self) -> None:
        """兼容测试或内部代码直接传入 Mapping，并检查跨字段约束。

        正式 load_config() 通常已经生成两个 Settings 对象；
        这里仍接受 Mapping，便于测试构造和保持 Config 自身边界稳定。
        """

        raw_cycles: Any = self.cycles
        raw_process: Any = self.process

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
# 6. Shared configuration loader
# =============================================================================


def load_config(
    path: Path, *, experiment_date: str, input_dir: Path
) -> Config:
    """Build one experiment from the shared channel and processing config."""
    config_path = path.resolve()
    loaded = _load_yaml_mapping(config_path, "config")
    required = {
        "channels",
        "input_format",
        "expected_sensor_interval_seconds",
        "image_match_tolerance_seconds",
        "cycles",
        "process",
    }
    _validate_keys(loaded, required, "config")
    if missing := sorted(required - loaded.keys()):
        raise ValueError(f"config missing keys: {missing}")

    # 日期和输入目录属于本次导入，不属于所有日期共用的处理规则。
    if not _is_iso_date(experiment_date):
        raise ValueError("experiment_date must use ISO YYYY-MM-DD format")

    input_format = _mapping(loaded["input_format"], "input_format")
    _validate_keys(
        input_format,
        {"sensor_globs", "image_extensions", "timestamp_column", "edf"},
        "input_format",
    )
    edf_input = _mapping(input_format["edf"], "input_format.edf")
    _validate_keys(edf_input, {"pair_tolerance_seconds"}, "input_format.edf")

    expected_interval = int(loaded["expected_sensor_interval_seconds"])
    image_tolerance = float(loaded["image_match_tolerance_seconds"])
    edf_pair_tolerance = float(edf_input["pair_tolerance_seconds"])
    _validate_positive("expected_sensor_interval_seconds", expected_interval)
    _validate_nonnegative("image_match_tolerance_seconds", image_tolerance)
    _validate_positive("edf_pair_tolerance_seconds", edf_pair_tolerance)
    if not str(input_format["timestamp_column"]).strip():
        raise ValueError("timestamp_column must not be empty")

    return Config(
        project_root=_find_project_root(config_path),
        experiment_id=f"exp_{experiment_date.replace('-', '')}",
        experiment_date=experiment_date,
        input_dir=Path(input_dir).resolve(),
        channels_path=config_path,
        sensor_globs=_tuple_strings(input_format["sensor_globs"], "sensor_globs"),
        image_extensions=_image_extensions(input_format["image_extensions"]),
        timestamp_column=str(input_format["timestamp_column"]),
        expected_sensor_interval_seconds=expected_interval,
        image_match_tolerance_seconds=image_tolerance,
        edf_pair_tolerance_seconds=edf_pair_tolerance,
        cycles=CycleSettings.from_mapping(_mapping(loaded["cycles"], "cycles")),
        process=ProcessSettings.from_mapping(_mapping(loaded["process"], "process")),
    )


# =============================================================================
# 7. YAML and validation helpers
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
