# Frost Sensor 科研分析 Pipeline

本项目把热泵除霜实验的文本传感器和 RGB 图片整理为可追溯的候选通道证据。主流程只有三阶段：

```text
data/0715（只读）
   ↓ prepare
prepared_data.parquet + cycle_summary.csv
   ↓ process
processed_data.parquet + cycle_summary.csv
   ↓ analyze
candidate_channel_evidence.csv
```

## 最短运行路径

```bash
python -m frost_analysis run \
  --config configs/0715.yaml \
  --output outputs/runs/0715_example
```

完整运行成功后，目录中还会有 `manifest.json`。阶段命令 `prepare`、`process`、`analyze` 都要求显式输入；它们不自动寻找最新结果、不缓存、不恢复，也不生成阶段 manifest。

Prepared 快照可以独立复用：

```bash
python -m frost_analysis process \
  --config configs/0715.yaml \
  --input outputs/prepared/exp_20260715/prepare_01/prepared_data.parquet \
  --cycles outputs/prepared/exp_20260715/prepare_01/cycle_summary.csv \
  --output outputs/runs/0715_process
```

原始 `data/<MMDD>` 永远只读；所有派生结果写入 `outputs/prepared/` 或 `outputs/runs/`。默认拒绝覆盖，确认后使用 `--overwrite`。

## 科学边界

Prepare 只解析原始 `.xls` 文本、应用显式单位换算、切分循环和独立匹配相机图片；不重采样、不填补、不计算 baseline 或动态特征。Process 在每个 `experiment_id × cycle_id` 内建立一次公共 10 秒网格，按精确阶段边界的重叠时长确定唯一阶段；然后仅在 `experiment_id × cycle_id × cycle_stage` 内执行 bounded 缺失处理、派生公式、共同 baseline 和 past-only 特征。等长 transition bucket 被排除，partial 行不进入 Process。Analyze 只使用 valid 且 baseline 可用的 `frost_development` 行。

通道、阈值和每日期相机映射均在 `configs/` 中显式记录。baseline 是 `cycle_local_early_stable_proxy`，不是人工或图像证明的绝对无霜真值。Reset evidence 当前固定为 `not_evaluated`，不使用下一循环自身 baseline 自证恢复。

## 测试

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check --no-cache src tests
.venv/bin/python -m mypy --strict --cache-dir=/tmp/frost-analysis-mypy src
```

字段、公式、质量标记和 decision 规则见 [`docs/pipeline_contract.md`](docs/pipeline_contract.md)。
