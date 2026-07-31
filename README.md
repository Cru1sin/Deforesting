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

科研结果写出后，可单独生成只读 QA 报告；Report 不是第四个科学处理阶段。
```

## 最短运行路径

```bash
python -m frost_analysis run \
  --config configs/0715.yaml \
  --output outputs/runs/0715_example
```

完整运行成功后，目录中还会有 `manifest.json`。阶段命令 `prepare`、`process`、`analyze` 都要求显式输入；它们不自动寻找最新结果、不缓存、不恢复，也不生成阶段 manifest。

为一个已经完成的运行生成 QA 图片：

```bash
python -m frost_analysis report \
  --input outputs/runs/0715_example \
  --output outputs/qa/0715_example
```

也可以在完整运行后请求 QA；如果 QA 失败，四个科研输出和 manifest 仍保留，但命令返回非零：

```bash
python -m frost_analysis run \
  --config configs/0715.yaml \
  --output outputs/runs/0715_example \
  --report --overwrite
```

## 模块地图

需要修改的内容与主要文件：

| 内容 | 文件 |
| --- | --- |
| 日期路径和阈值 | `configs/<date>.yaml` |
| 原始字段、单位、角色 | `configs/channels.yaml` |
| 相机物理角色 | `configs/camera_mappings/<date>.yaml` |
| 原始数据整理 | `src/frost_analysis/prepare.py` |
| 循环边界 | `src/frost_analysis/cycles.py` |
| 重采样和缺失处理 | `src/frost_analysis/process.py` |
| Baseline | `src/frost_analysis/baseline.py` |
| 候选证据 | `src/frost_analysis/analysis.py` |
| 科研 QA 图片 | `src/frost_analysis/report.py` |
| 阶段编排 | `src/frost_analysis/pipeline.py` |
| 合同检查 | `src/frost_analysis/validation.py` |

建议阅读顺序：

```text
README → docs/pipeline_contract.md → pipeline.py → config.py / channels.py
→ prepare.py / cycles.py → process.py / baseline.py → analysis.py → validation.py
→ report.py
```

Prepared 快照可以独立复用：

```bash
python -m frost_analysis process \
  --config configs/0715.yaml \
  --input outputs/prepared/exp_20260715/prepare_01/prepared_data.parquet \
  --cycles outputs/prepared/exp_20260715/prepare_01/cycle_summary.csv \
  --output outputs/runs/0715_process
```

原始 `data/<MMDD>` 永远只读；所有派生结果写入 `outputs/prepared/` 或 `outputs/runs/`。默认拒绝覆盖，确认后使用 `--overwrite`。

## How to validate one experiment

先完成科研运行，再打开对应的 QA 目录：

```text
outputs/qa/<date>/
├── cycles/cycle_001_overview.png
├── coverage.png
├── baseline.png
├── candidate.png
└── report_summary.json
```

建议顺序：

1. 查看首个、中间和最后一个 cycle overview：确认起止边界、阶段、baseline 窗口、图片数量，以及传感器断线是否保持为空白。
2. 查看 `coverage.png`：确认观测时间带、缺失区间和图片时间是否符合当天记录。
3. 查看 `baseline.png`：只核对已保存窗口和固定诊断通道的 observed-only 曲线，不把它当作搜索区间证明。
4. 查看 `candidate.png`：核对 Evidence 已保存的数值、decision 和 reason；Report 不重新拟合或计算统计量。
5. 最后查看四个科研 CSV/Parquet 和 `report_summary.json`。

Report 只读取正式输出，允许为展示进行筛选、分组、遮罩、计数、排序、布局和 hash；不重新执行循环切分、重采样、coverage 阈值、插补、派生公式、baseline/residual、相关性、未来/context evidence、方向一致性或 decision。

## 科学边界

Prepare 只解析原始 `.xls` 文本、应用显式单位换算、切分循环和独立匹配相机图片；不重采样、不填补、不计算 baseline 或动态特征。Process 在每个 `experiment_id × cycle_id` 内按 summary 边界建立完整 10 秒网格；任何阶段或循环边界严格落在桶内部时排除该桶，不使用 overlap winner，也不从输入行 fallback 推断阶段；然后仅在 `experiment_id × cycle_id × cycle_stage` 内执行 bounded 缺失处理、派生公式、共同 baseline 和 past-only 特征，Step 缺失桶逐点按其距最后 observed 值的时间差执行 bounded forward fill。partial 行不进入 Process。Analyze 只使用 valid 且 baseline 可用的 `frost_development` 行。

通道、阈值和每日期相机映射均在 `configs/` 中显式记录。baseline 是 `cycle_local_early_stable_proxy`，不是人工或图像证明的绝对无霜真值。Reset evidence 当前固定为 `not_evaluated`，不使用下一循环自身 baseline 自证恢复。

## 测试

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check --no-cache src tests
.venv/bin/python -m mypy --strict --cache-dir=/tmp/frost-analysis-mypy src
```

字段、公式、质量标记和 decision 规则见 [`docs/pipeline_contract.md`](docs/pipeline_contract.md)。
