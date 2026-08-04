# Frost Sensor 科研分析 Pipeline

本项目把热泵除霜实验的文本传感器和 RGB 图片整理为可追溯的候选通道证据。主流程只有三阶段：

```text
data/0715-0731（只读）
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

### Config：先定义科学规则

#### 日期配置

每个实验日期只有一个日期事实文件

`configs/0715.yaml` 定义实验编号和日期、原始数据目录、通道配置路径、相机映射路径、传感器采样间隔、图片匹配容差、循环切分阈值、重采样和缺失处理规则、baseline 窗口、分析阈值

`config.py` 使用不可变 dataclass 将 YAML 转成：

```
Config
├── CycleSettings
├── ProcessSettings
│   └── BaselineSettings
└── AnalysisSettings
```

加载时会立即检查日期格式、时间间隔整除关系、阈值范围、未来 horizon 是否和重采样网格对齐。

#### 通道配置

`configs/channels.yaml` 定义每一个变量的：

```
unit
kind
role
source_names
scale / offset
valid_range
resample
missing
analysis_candidate
expected_frost_direction
```

通道类型只有：

```
continuous
step
event
categorical
protected
derived
```

派生量只允许白名单公式：

```
cop
pressure_ratio
water_delta_temperature
superheat_calculated
```

这意味着公式不能任意写在 YAML 中执行，避免配置文件变成不可控代码。

共同方法参数集中在 `configs/defaults.yaml`；。日期文件使用 schema v2，并将相机物理角色直接写在
`camera_roles` 中：

```yaml
schema_version: 2
defaults_path: defaults.yaml
experiment_id: frost_0715
experiment_date: "2026-07-15"
input_dir: data/0715
expected_sensor_interval_seconds: 1
camera_roles: {}
overrides: {}
```

`defaults.yaml` 保存共同的 `input_format`、图像匹配容差、`cycles`、`process` 和
`analysis` 方法设置。其中 Process 的设置保持嵌套结构：
`process.baseline` 和 `process.features`。日期文件只保存实验身份、输入目录、当天的
`camera_roles` 以及确有需要的 `overrides`；不再使用外部 camera mapping 文件。

运行阶段会在 `manifest.json` 中保存最终的 `resolved_config` 和配置 provenance，包括
defaults、日期配置、channels 文件的路径与 hash，以及 resolved config 的 hash。这样一次
运行实际采用的有效配置可以随运行产物复现；Report 读取 manifest 和正式科研输出，不重新
读取当前 YAML 配置。

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
| 共同方法参数 | `configs/defaults.yaml` |
| 日期事实、输入路径和必要 overrides | `configs/<date>.yaml` |
| 原始字段、单位、角色 | `configs/channels.yaml` |
| 相机物理角色 | `configs/<date>.yaml` 中的 `camera_roles` |
| 一对一时间匹配 | `src/frost_analysis/alignment.py` |
| 原始数据整理 | `src/frost_analysis/prepare.py` |
| 循环边界 | `src/frost_analysis/cycles.py` |
| 重采样和缺失处理 | `src/frost_analysis/process.py` |
| Baseline | `src/frost_analysis/baseline.py` |
| 候选证据 | `src/frost_analysis/analysis.py` |
| 科研 QA 图片 | `src/frost_analysis/report.py` |
| 阶段编排 | `src/frost_analysis/pipeline.py` |
| 合同检查 | `src/frost_analysis/validation.py` |
| Cycle Dataset 发布与读取 | `src/frost_analysis/dataset.py`、`dataset_metadata.py`、`dataset_images.py`、`dataset_loader.py`、`dataset_validation.py` |

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

## Cycle Dataset 自包含发布层

Dataset 直接从日期原始目录构建，不持久化 Run、不经过中间 Dataset 或 canonicalize。每个
cycle 保存 10 秒 Parquet/CSV、Original CSV、publication PNG 和 RGB coverage PNG；图片保留
原始 basename，并按 `images/<cycle>/<source_camera_id>__<current_role>/` 组织。

```bash
python -m frost_analysis dataset add data/0714
python -m frost_analysis dataset add data/0715
python -m frost_analysis dataset validate --dataset dataset
python -m frost_analysis dataset refresh --dataset dataset
python -m frost_analysis dataset review-cycle frost_cycle_000001 \
  --status valid --reason manual_review_confirmed
python -m frost_analysis analysis --dataset dataset --status valid \
  --output outputs/analysis/frost_dataset
```

Dataset 的目录、Manifest、Catalog、图片角色、Loader、追加、科学 edit、refresh 和验证
合同见 [`docs/dataset_contract.md`](docs/dataset_contract.md)。`source_directory` 只用于
人工 provenance；Dataset 下游不会重新读取 Raw 或配置。

## 科学边界

Prepare 解析原始 `.xls` 文本，并将 EDF 中双 SHT40 的温度/RH 原始信号按主数据时间范围对齐融合；随后应用显式单位换算、切分循环和独立匹配相机图片。不重采样、不填补、不计算 baseline 或动态特征。Process 在每个 `experiment_id × cycle_id` 内按 summary 边界建立完整 10 秒网格；任何阶段或循环边界严格落在桶内部时排除该桶，不使用 overlap winner，也不从输入行 fallback 推断阶段；然后仅在 `experiment_id × cycle_id × cycle_stage` 内执行 bounded 缺失处理、派生公式、共同 baseline 和 past-only 特征，Step 缺失桶逐点按其距最后 observed 值的时间差执行 bounded forward fill。partial 行不进入 Process。Analyze 只使用 valid 且 baseline 可用的 `frost_development` 行。

共同通道方法和阈值在 `configs/defaults.yaml` 中显式记录；日期事实和每日期相机角色在各自的
`configs/<date>.yaml` 中记录。baseline 是 `cycle_local_early_stable_proxy`，不是人工或图像证明的绝对无霜真值。Reset evidence 当前固定为 `not_evaluated`，不使用下一循环自身 baseline 自证恢复。

## 测试

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check --no-cache src tests
.venv/bin/python -m mypy --strict --cache-dir=/tmp/frost-analysis-mypy src
```

字段、公式、质量标记和 decision 规则见 [`docs/pipeline_contract.md`](docs/pipeline_contract.md)。
