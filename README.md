# Frost Sensor 科研分析 Pipeline

这是面向论文实验数据的科研分析代码，不是通用数据平台。主流程只有三步：

```text
data/0715                 原始实验记录（只读）
   ↓ prepare              整理字段、切分循环、匹配图片
prepared_data.parquet
   ↓ process              10 秒重采样、限定缺失处理、物理量和 baseline
processed_data.parquet
   ↓ analyze              循环级候选通道证据
candidate_channel_evidence.csv
```

## 最短上手路径

完整运行：

```bash
python -m frost_analysis run \
  --config configs/0715.yaml \
  --output outputs/runs/0715_example
```

如果只想先生成可复用的 Prepared 快照：

```bash
python -m frost_analysis prepare \
  --config configs/0715.yaml \
  --output outputs/prepared/exp_20260715/prepare_01
```

之后可通过 `process --input ... --cycles ...` 显式复用快照。Pipeline 不自动搜索或恢复旧结果。

## 目录和阶段职责

```text
src/frost_analysis/
├── config.py       扁平 YAML 配置
├── channels.py     通道事实和最小合同
├── io.py           输入发现、Parquet/CSV、manifest
├── prepare.py      原始传感器整理和阶段编排
├── cycles.py       循环边界、状态和坐标
├── images.py       文件名时间戳匹配
├── process.py      重采样、缺失处理和阶段编排
├── baseline.py     严格无霜基准
├── features.py     物理量和过去窗口特征
├── analysis.py     候选通道证据
├── validation.py   结构性不变量
└── pipeline.py     run_pipeline() 单一完整入口
```

`data/<MMDD>` 永远只读。派生输出只能写入 `outputs/prepared/` 或 `outputs/runs/`。正式运行的四个结果文件是 `prepared_data.parquet`、`processed_data.parquet`、`cycle_summary.csv` 和 `candidate_channel_evidence.csv`；四者成功写完后才生成 `manifest.json`。

## 关键科学边界

Prepare 不插值、不重采样、不计算 baseline、rolling、slope 或 residual。Prepared 的时间键是 `experiment_id + timestamp`，循环摘要的键是 `experiment_id + cycle_id`。

循环坐标只在 `frost_development` 阶段有值：

```text
cycle_elapsed_seconds = timestamp - stable_heating_start
cycle_progress = (timestamp - stable_heating_start)
                  / (defrost_start - stable_heating_start)
```

`cycle_progress` 被限制在 `[0, 1]`；其他阶段为 NaN。Process 重采样后重新计算它。

源内重复不选择、不平均：受影响通道值置为 NaN，并保留 `__duplicate`、`__conflict` 事实；同一时间其他正常通道不被整行删除。除霜状态缺口超过 5 秒不推断，并将受影响循环设为 `cycle_status=incomplete`。缺失压缩机频率表示未知，不等于停机。

Process 只在 `experiment_id × cycle_id × cycle_stage` 内处理缺失。连续量按配置线性插值，阶跃量只允许前值保持，事件量和 protected 量默认不填补。派生量的 `__imputed` 使用依赖量布尔 OR。baseline 失败保持 NaN，不 fallback。

## 候选证据

`candidate_channel_evidence.csv` 的统计单位是循环和实验日期，而不是原始采样点。每个候选通道一行，字段和公式见 [docs/pipeline_contract.md](docs/pipeline_contract.md)。当前 decision 只使用趋势证据、方向一致率和工况关联；复位证据和未来性能证据先作为展示字段，不参与综合评分。

## 验证

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/ruff check --no-cache src tests
.venv/bin/python -m mypy --strict --cache-dir=/tmp/frost-analysis-mypy src
```

所有显式日期可运行 Prepare smoke test；完整 Analysis 先选择至少三个代表日期。跨日期科学稳定性分析另行进行，不能把所有采样点直接拼接成一个相关系数。
