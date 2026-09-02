"""Dataset construction configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class CycleSettings:
    """保存 cycles.py 切分制热—结霜—除霜循环时使用的阈值。

    本类只保存阈值；状态去抖、事件识别和循环切分由 cycles.py 完成。
    """

    defrost_channel: str = "defrost_active"

    # 允许自动桥接的除霜状态缺失区间上限（秒）。
    # 只有缺失段两端状态一致且总间隔不超过该值时，cycles.py 才允许补齐状态。
    # 0.0 表示任何除霜状态缺失均不自动推断。
    maximum_state_gap_seconds: float = 5.0

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

    operating_mode_channel: str = "operating_mode"

    required_operating_mode: str = "3"

    def __post_init__(self) -> None:
        _validate_nonnegative("maximum_state_gap_seconds", self.maximum_state_gap_seconds)
        _validate_positive("debounce_seconds", self.debounce_seconds)
        _validate_positive("minimum_defrost_seconds", self.minimum_defrost_seconds)
        _validate_positive("maximum_defrost_seconds", self.maximum_defrost_seconds)
        _validate_positive("minimum_heating_seconds", self.minimum_heating_seconds)
        _validate_positive("maximum_heating_seconds", self.maximum_heating_seconds)
        _validate_positive(
            "defrost_preparation_setpoint_drop_hz",
            self.defrost_preparation_setpoint_drop_hz,
        )
        _validate_positive(
            "defrost_preparation_lookback_seconds",
            self.defrost_preparation_lookback_seconds,
        )
        _validate_nonnegative("stable_heating_seconds", self.stable_heating_seconds)
        if self.minimum_defrost_seconds > self.maximum_defrost_seconds:
            raise ValueError(
                "minimum_defrost_seconds must not exceed "
                "maximum_defrost_seconds"
            )
        if self.minimum_heating_seconds > self.maximum_heating_seconds:
            raise ValueError(
                "minimum_heating_seconds must not exceed "
                "maximum_heating_seconds"
            )


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

    def __post_init__(self) -> None:
        if self.stage != "frost_development":
            raise ValueError("baseline stage must be frost_development")
        if self.baseline_seconds <= 0:
            raise ValueError("baseline_seconds must be positive")
        _validate_fraction("minimum_observed_coverage", self.minimum_observed_coverage)
        if any(value < 0 for value in self.anchor_maximum_std.values()):
            raise ValueError("anchor_maximum_std values must be nonnegative")


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

    def __post_init__(self) -> None:
        if self.resample_interval_seconds <= 0:
            raise ValueError("resample_interval_seconds must be positive")
        _validate_fraction(
            "minimum_continuous_bucket_coverage",
            self.minimum_continuous_bucket_coverage,
        )
        _validate_nonnegative("continuous_max_gap_seconds", self.continuous_max_gap_seconds)
        _validate_nonnegative("control_max_gap_seconds", self.control_max_gap_seconds)


@dataclass(frozen=True)
class Config:
    """One experiment's Raw-to-Dataset settings."""

    project_root: Path
    experiment_id: str
    experiment_date: str
    input_dir: Path
    sensor_globs: tuple[str, ...] = ("*.xls", "*.edf")
    image_extensions: tuple[str, ...] = (".jpg", ".png")
    timestamp_column: str = "时间"

    # 原始传感器理论采样间隔（秒），用于估算每个重采样桶的期望点数。
    expected_sensor_interval_seconds: int = 1

    # 图片时间戳与传感器时间戳匹配时允许的最大偏移（秒）。
    image_match_tolerance_seconds: float = 2.0

    # EDF 双 SHT40 顺序配对允许的最大时间差（秒）。
    edf_pair_tolerance_seconds: float = 1.0

    # 循环切分规则。
    cycles: CycleSettings = field(default_factory=CycleSettings)

    # 重采样、bounded 缺失处理、派生量和 Baseline 规则。
    process: ProcessSettings = field(default_factory=ProcessSettings)

    def __post_init__(self) -> None:
        # 公共重采样间隔必须是原生采样间隔的整数倍，
        # 否则每个 bucket 的期望原始点数无法稳定定义。
        _validate_positive(
            "expected_sensor_interval_seconds", self.expected_sensor_interval_seconds
        )
        _validate_nonnegative(
            "image_match_tolerance_seconds", self.image_match_tolerance_seconds
        )
        if self.process.resample_interval_seconds % self.expected_sensor_interval_seconds != 0:
            raise ValueError(
                "resample_interval_seconds must be divisible by "
                "expected_sensor_interval_seconds"
            )
        _validate_positive("edf_pair_tolerance_seconds", self.edf_pair_tolerance_seconds)


# =============================================================================
# 6. Shared configuration loader
# =============================================================================


def load_config(*, project_root: Path, experiment_date: str, input_dir: Path) -> Config:
    """Build one experiment from the explicit Raw-to-Dataset defaults."""
    if not _is_iso_date(experiment_date):
        raise ValueError("experiment_date must use ISO YYYY-MM-DD format")
    return Config(
        project_root=Path(project_root).resolve(),
        experiment_id=f"exp_{experiment_date.replace('-', '')}",
        experiment_date=experiment_date,
        input_dir=Path(input_dir).resolve(),
    )


# =============================================================================
# 7. Validation helpers
# =============================================================================


def find_project_root(config_path: Path) -> Path | None:
    """Return the nearest project root, or None when no repository root exists."""
    resolved = config_path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def _is_iso_date(value: str) -> bool:
    """严格判断字符串是否为规范 ISO 日期 YYYY-MM-DD。"""
    try:
        # fromisoformat() 负责解析；
        # 再转回 isoformat()，拒绝非规范但可被宽松解析的表示。
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


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
