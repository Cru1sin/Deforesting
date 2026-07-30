# Frost Sensor Candidate Validation

这是第一阶段的传感器侧候选通道验证管线。它不训练最终结霜隐状态、不训练监督网络，也不输出除霜策略。

```text
原始传感器 -> Unified Feature Registry -> 循环级无霜基准残差
            -> 趋势/复位/滞后/工况证据 -> 候选通道
```

## 运行

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m frost_analysis prepare --config configs/0715.yaml
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m frost_analysis process --config configs/0715.yaml
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m frost_analysis analyze --task correlation --config configs/0715.yaml
```

输入位于 `data/<MMDD>`，结果发布到 `outputs/<MMDD>`。`.DS_Store` 只作为输入盘点文件，不参与 Registry 或候选分析。

## 活动契约

活动字段只由 `configs/feature_registry.yaml` 定义。参数表 5 的 backup、同名、平均、修正和标定量不会进入活动科学层；制热量只使用 `QComp10W -> heating_capacity`，`CCQ_Comp` 不进入 Registry。压比统一为 `PR = Pc_abs / Pe_abs`，`Pr` 只作 `Pr ≈ 100 × PR` 的一致性核查。

用户可见目录只发布四个核心文件：

- `prepared_data.parquet`：字段已统一、图像已对齐、循环已标记，但不插值、不重采样。
- `processed_data.parquet`：限定缺口处理、循环内 clean baseline、重采样和动态特征。
- `cycle_summary.csv`：传感器与 RGB 覆盖、最大缺口、具体中断区间、基准和可处理状态。
- `correlation_results.csv`：一行一个物理通道的趋势、工况、滞后和复位证据；不包含固定权重排名。

运行状态只写入 `outputs/<MMDD>/.pipeline/`。旧单体入口、旧多表输出和旧报告已移动到 `archive/feature_registry_transition_20260724/`。

## 代码结构

代码按职责分层，CLI 和公共合同留在根目录：

```text
src/frost_analysis/
├── cli.py, config.py, schemas.py
├── pipelines/       prepare.py, process.py
├── data/            sensors.py, images.py, cycles.py, alignment.py, registry.py, inventory.py
├── processing/      missing.py, baseline.py, resample.py, features.py
├── analysis/        correlation.py, screening.py
└── core/            artifacts.py, validation.py
```

`pipelines` 只编排阶段顺序；`data` 处理原始数据和实验结构；`processing` 做缺失、基准、重采样和特征；`analysis` 是可替换的科研任务；`core` 只负责保存和合同校验。当前不创建神经网络专用 `dataloader.py`。

## 验证边界

当前统计单位是循环和日期，不是每个原始采样点。阶段一保留源时间戳和真实缺口，`analysis_interval_seconds` 仅是目标分辨率提示，实际源间隔按循环记录，不人为填充规则网格。长缺口循环会被标记为污染，不作为完整证据。当前图像只完成文件清单、时间对齐和质量掩码，尚未用像素验证霜层。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check --no-cache src/frost_analysis tests
.venv/bin/python -m mypy --strict --cache-dir=/tmp/frost-analysis-mypy src/frost_analysis
```
