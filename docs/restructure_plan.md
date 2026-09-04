# PINN4SOH 式可读性重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把仓库重构为适合长期私人研究的清晰工作台：README顺序运行五个直接入口，`argparse`暴露全部关键选择，普通函数字典切换成本函数，同时保留一个不污染主线的扁平分析区。

**Architecture:** 仓库不再作为可安装Python包，而作为可直接运行和持续扩展的科研代码。五个`main_*.py`构成稳定主线；`analysis/`只放按研究问题命名的可运行脚本，单向调用主线函数，主线永远不依赖analysis。成本函数“插件”只是统一签名的普通函数和一个显式字典，不使用抽象类、动态发现、协议或配置框架。

**Tech Stack:** Python 3.11、uv、pandas、NumPy、scikit-learn、PyTorch、matplotlib、PyYAML、pytest、ruff。

---

## 1. 本轮明确不做什么

- 不保留 `src/` 布局，不提供 `python -m frost_analysis dataset`一类包入口。
- 不建立“正式包”“探索包”“证据包”或插件框架；`analysis/`只是可运行脚本目录。
- 不增加 workflow、factory、protocol、repository、service 等抽象层。
- 不为可重新生成的输出实现原子写入、回滚事务、checksum 或多级 manifest。
- 不把被删除的历史代码搬入仓库内的新 `archive/`；Git 历史已经是归档。
- 不追求覆盖每个异常分支，只保留会改变科学结论的检查。
- 不重新设计成本函数数学内容；本轮只改变代码组织和入口。

必须保留的科学约束只有：

1. 按 experiment/cycle 分组，训练集与测试集不交叉；
2. 候选时刻只使用该时刻之前的传感器和图像；
3. 能量积分不跨越长缺口；
4. 成本函数分母必须为正；
5. V2.6.7超出训练支持域时明确标记 `supported=False`；
6. 长时间训练保留最小断点续跑记录。

## 2. 目标目录

```text
README.md
pyproject.toml
uv.lock

main_data.py
main_cost.py
main_labels.py
main_train.py
main_evaluate.py

dataloader/
├── load.py             # 读取循环、metadata和图像索引
├── cycles.py           # 从原始日期构建循环、阶段和候选时间
├── images.py           # 图像匹配、下载、RGB panel
└── channels.py         # 传感器列名与少量派生量

cost/
├── energy.py           # 功率、制热量和覆盖率积分
├── candidates.py       # EH/QH累计量及候选时刻表
├── legacy.py           # V1–V2.6.6、V3和renewal_water的短函数
├── ticket.py           # V2.6.7 ticket训练与预测
└── functions.py        # COST_FUNCTIONS显式注册表

labels/
├── build.py            # 成本曲线到二/三分类标签
└── images.py           # 标签与各机位图片匹配、导出

model/
├── features.py         # RGB和传感器特征
├── models.py           # 分类头和融合模型
├── train.py            # 单次训练、按实验留出、断点续跑
└── evaluate.py         # 指标、汇总和预测表

plotter/
├── style.py
├── figure_1_cost.py
├── figure_2_labels.py
├── figure_3_rgb_increment.py
├── figure_4_failure.py
├── figure_5_models.py
└── figure_6_concentration.py

analysis/                # 不是包；主线不得import这里
├── cost_comparison.py
├── recovery_energy.py
├── ticket_stability.py
├── sensor_correlations.py
├── camera_ablation.py
└── sensor_fusion.py

configs/
└── experiments.yaml      # 仅保存批量实验预设；不是第二套运行参数

model_artifacts/
└── cost/
    ├── v1.json
    ├── legacy.json
    ├── v267.json
    └── v267_validation.csv

demo/
├── frost_cycle_000070/
└── expected/

paper/
tests/
output/                 # 保持Git忽略；沿用现有成本函数/label/model结构
```

不创建 `__init__.py`。从仓库根目录运行脚本时，Python可直接导入这些目录中的模块。

### 2.1 `data/`和`dataset/`保持原样

本轮只重构读取和计算代码，不迁移、不重命名、不重新生成现有数据：

- `data/`继续保存当前原始日期数据，目录名和文件内容不变；
- `dataset/`继续使用当前结构：`cycle_catalog.json`、`channel_registry.json`、
  `image_metadata.parquet`和`cycles/`；
- 新代码目录命名为`dataloader/`，避免与现有`dataset/`冲突；
- 重构过程中不运行`main_data.py --action add`，所有测试只读取demo副本；
- `main_data.py`默认动作是`validate`，只有用户显式指定`--action add`时才写入Dataset；
- import模块、计算成本、生成标签、训练和画图都不得写入`data/`或`dataset/`。

迁移开始和结束各保存一次路径、大小、修改时间清单到`/private/tmp`，只作本次迁移核对，
不把checksum或manifest加入项目：

```bash
find data dataset -type f -exec stat -f '%N|%z|%m' {} + | sort \
  > /private/tmp/defrost_data_before.txt
find data dataset -type f -exec stat -f '%N|%z|%m' {} + | sort \
  > /private/tmp/defrost_data_after.txt
cmp /private/tmp/defrost_data_before.txt /private/tmp/defrost_data_after.txt
```

最终`cmp`必须无输出并返回0。这个一次性核对不进入运行时代码。

### 2.2 内部接口只保留七个动词

所有主线以pandas表和普通路径连接，不创建Bundle、Contract、Result类：

| 入口 | 唯一公共函数 | 输入 | 返回 |
|---|---|---|---|
| data | `list_cycles(dataset)`、`load_cycle(dataset, cycle)` | 当前Dataset路径 | list或DataFrame |
| cost | `calculate_cost(name, candidates, artifact_path=None, fit=False)` | 候选表 | DataFrame |
| labels | `build_labels(cost_table, images, task, regret)` | 成本表、图像索引 | DataFrame |
| train | `train_experiment(labels, args)` | 标签表、argparse参数 | run目录Path |
| evaluate | `evaluate_run(run_dir)` | run目录 | metrics DataFrame |
| figures | `plot_figure_N(table, output)` | 结果表、输出路径 | 输出Path |

命名只使用`load_`、`build_`、`calculate_`、`train_`、`evaluate_`、`plot_`、`save_`。
主入口负责解析参数和选择输出路径，领域函数负责计算；不在main和领域文件各实现一遍业务逻辑。

### 2.3 输出目录和文件名固定

```text
output/
├── 成本函数/
│   ├── cost_function_<cost>.csv
│   ├── cost_function_<cost>_validation.csv
│   ├── cost_function_<cost>_cycles/
│   │   └── frost_cycle_000070.png
│   └── comparison_<costs>.png
├── label/
│   └── <cost>_<task>/
│       ├── labels.parquet
│       ├── summary.csv
│       └── label_distribution.png
├── model/
│   ├── <run_id>/
│       ├── README.md
│       ├── args.json
│       ├── progress.jsonl
│       ├── metrics.csv
│       ├── predictions.parquet
│       ├── model_front.pth
│       └── training_loss.png
│   └── _cache/           # 内部RGB特征缓存，用户无需直接读取
├── figures/
    ├── figure_1_cost.png
    ├── figure_1_cost.pdf
    └── figure_2_labels.png
└── analysis/
    ├── cost_comparison/
    ├── recovery_energy/
    ├── ticket_stability/
    ├── sensor_correlations/
    ├── camera_ablation/
    └── sensor_fusion/
```

- 同一次运行的模型、日志、表格和训练图放在同一run目录，不再增加`logs/`、`tables/`、
  `figures/`等下级目录；
- 成本CSV必须包含`cycle_name`、`experiment_id`、`candidate_time`、成本组成项、`J`、
  `supported`和最优时间，以后画图只读这一张表；
- 文件名使用研究对象，不使用A/B/C、MATRIX、hash、final或new等上下文名称；
- `output/test/`不再作为新流程输出；其中仍有用的正式表和图片分别迁入上述主线目录，
  仍在扩展的研究结果迁入`output/analysis/<study>`，RGB特征缓存迁入`output/model/_cache`，
  失败smoke和可重新生成的半成品删除；
- Dataset中的已下载图片不属于输出整理范围，不移动也不删除。

### 2.4 函数和文件可读性规则

- 每个文件开头用三行以内说明“研究问题、输入、输出”；
- 只有被其他文件调用的公共函数写简短docstring，内部一眼可懂的函数不写模板化注释；
- 一个文件只回答一个问题；不按行数机械拆文件，读者需要同时打开两个文件才能理解一个公式时就不拆；
- 函数只在同时承担两个独立计算步骤时拆分，不用行数、类型或复杂度规则强迫增加包装函数；
- 变量使用领域名称：`candidate_time`、`heating_energy`、`defrost_cost`、`macro_f1`；
- 删除`bundle`、`contract`、`protocol`、`shard`、`cohort gate`等新人必须学习的内部术语；
- 注释解释物理或实验原因，不复述代码动作。

### 2.5 私人研究扩展规则

`analysis/`对应PINN4SOH中的`data analysis/`和`results analysis/`，但命名和依赖更清楚：

- 它是普通脚本目录，不创建`__init__.py`，不被任何主线文件import；
- 一个文件回答一个研究问题，文件名直接写问题对象，不使用版本号或临时编号；
- 脚本只调用`dataloader/`、`cost/`、`labels/`、`model/`的现有函数，不复制积分、标签或训练逻辑；
- 每个脚本都有自己的`get_args()`，默认输出到同名`output/analysis/<study>/`；
- 新成本公式不留在analysis中：确认要全循环比较后，作为函数加入`COST_FUNCTIONS`；
- 分析结论失效或已经进入主线后，删除脚本或缩成一张仍有价值的比较图，不保留两套实现；
- analysis脚本不要求单独测试；只有当代码进入主线时才增加一个最小测试。

`configs/experiments.yaml`只用于表达需要批量展开的实验矩阵。`main_train.py`中的
`parser.add_argument`始终是完整、可直接运行的配置面；YAML中的每个键必须对应一个同名CLI参数，
CLI显式值覆盖预设。单次训练不要求读者先打开YAML。

添加一次新分析的固定模板是：

```python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from dataloader.load import load_cycle_table


def compare_recovery(cycles):
    columns = ["cycle_name", "recovery_electricity_kwh", "recovery_water_heat_kwh"]
    return cycles.loc[:, columns].dropna()


def plot_recovery(result, output):
    figure, axis = plt.subplots(figsize=(7, 3))
    axis.plot(result["cycle_name"], result["recovery_electricity_kwh"], label="Electricity")
    axis.plot(result["cycle_name"], result["recovery_water_heat_kwh"], label="Water heat")
    axis.legend()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def get_args():
    parser = argparse.ArgumentParser(
        "Compare recovery electricity and heat across defrost cycles."
    )
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output", default="output/analysis/recovery_energy")
    return parser.parse_args()


def main():
    args = get_args()
    cycles = load_cycle_table(args.dataset)
    result = compare_recovery(cycles)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "recovery_by_cycle.csv", index=False)
    plot_recovery(result, output / "recovery_by_cycle.png")


if __name__ == "__main__":
    main()
```

模板只规定阅读顺序和输出位置，不抽象成通用runner。

## 3. 统一入口设计

每个 `main_*.py` 必须按相同顺序组织。以 `main_cost.py` 为例：

```python
import argparse
from pathlib import Path

from cost.functions import COST_FUNCTIONS, calculate_cost
from dataloader.load import load_cycle_table
from plotter.figure_1_cost import plot_cost


def get_args():
    parser = argparse.ArgumentParser(
        "Calculate the cost of defrosting at every candidate time."
    )
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--cost", choices=COST_FUNCTIONS, default="v2.6.7")
    parser.add_argument("--artifact")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", default="output/成本函数")
    return parser.parse_args()


def main():
    args = get_args()
    cycles = load_cycle_table(args.dataset)
    result = calculate_cost(args.cost, cycles, args.artifact, fit=args.fit)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / f"cost_function_{args.cost}.csv", index=False)
    if args.plot:
        plot_cost(result, output / f"cost_function_{args.cost}.png")


if __name__ == "__main__":
    main()
```

入口文件允许有少量直接循环和路径拼接，不为三五行代码增加 helper。每个入口目标不超过180行，读者从上到下能看见完整执行路径。

## 4. 成本函数插件约定

“插件”只是一组同签名函数：

```python
# cost/functions.py
import json
from pathlib import Path

from cost.legacy import (
    cost_renewal_water,
    cost_v1,
    cost_v2,
    cost_v21,
    cost_v22,
    cost_v23,
    cost_v24,
    cost_v25,
    cost_v26,
    cost_v261,
    cost_v262,
    cost_v263,
    cost_v264,
    cost_v265,
    cost_v266,
    cost_v3,
)
from cost.ticket import cost_v267

COST_FUNCTIONS = {
    "v1": cost_v1,
    "v2": cost_v2,
    "v2.1": cost_v21,
    "v2.2": cost_v22,
    "v2.3": cost_v23,
    "v2.4": cost_v24,
    "v2.5": cost_v25,
    "v2.6": cost_v26,
    "v2.6.1": cost_v261,
    "v2.6.2": cost_v262,
    "v2.6.3": cost_v263,
    "v2.6.4": cost_v264,
    "v2.6.5": cost_v265,
    "v2.6.6": cost_v266,
    "v2.6.7": cost_v267,
    "v3": cost_v3,
    "renewal_water": cost_renewal_water,
}

COST_ARTIFACTS = {
    "v1": "model_artifacts/cost/v1.json",
    "v2.6.7": "model_artifacts/cost/v267.json",
}


def calculate_cost(name, candidates, artifact_path, fit=False):
    path = artifact_path or COST_ARTIFACTS.get(name, "model_artifacts/cost/legacy.json")
    artifact = json.loads(Path(path).read_text())
    return COST_FUNCTIONS[name](candidates, artifact, fit)
```

`main_cost.py`直接将注册表暴露给读者：

```python
parser.add_argument(
    "--cost",
    choices=COST_FUNCTIONS,
    default="v2.6.7",
    help="Cost function used to evaluate every candidate defrost time.",
)
```

禁止加入 `BaseCostFunction`、`CostProtocol`、entry points、decorator registration或自动扫描目录。增加新算法时只写一个函数并在字典增加一行。

## 5. 模型参数与代码分离

`selected.py`中的系数和支持域移入JSON。每个JSON只使用一层可读结构：

```json
{
  "name": "v2.6.7",
  "features": [
    "water_in_temperature",
    "water_out_temperature",
    "coil_temperature",
    "evaporating_pressure",
    "water_temperature_setpoint"
  ],
  "state_window_seconds": 60,
  "targets": {
    "E_T": {"intercept": 0.0, "coefficients": [], "support": {}},
    "Q_T": {"intercept": 0.0, "coefficients": [], "support": {}}
  },
  "training": {
    "split": "leave-one-experiment-out",
    "event_count": 68,
    "experiment_count": 15
  }
}
```

实际数值由现有V2.6.7拟合结果导出，不手工复制。详细LOEO结果继续保存在一个 `v267_validation.csv`；不创建独立checksum、manifest和model-card文件。

---

## Task 1：冻结一个最小可比样例

**Files:**
- Create: `demo/frost_cycle_000070/`
- Create: `demo/expected/cost.csv`
- Create: `tests/test_demo.py`

- [ ] 在任何迁移前生成`/private/tmp/defrost_data_before.txt`，以后不对`data/`和`dataset/`
  执行写操作。
- [ ] 从现有Dataset复制 `frost_cycle_000070` 所需的传感器表、metadata和少量front图像；不复制整套Dataset。
- [ ] 从现有 `cost_function_v1.csv`、`cost_function_v2.5.csv`、`cost_function_v2.6.7.csv`提取该循环的候选时间、成本、最优点和1%区间，合并写入 `demo/expected/cost.csv`。
- [ ] 新增一个测试，只检查样例可读且三种成本版本都有候选点：

```python
def test_demo_contains_reference_costs():
    expected = pd.read_csv("demo/expected/cost.csv")
    assert set(expected["cost"]) == {"v1", "v2.5", "v2.6.7"}
    assert expected.groupby("cost")["candidate_time"].size().min() > 2
```

- [ ] 运行：`uv run pytest tests/test_demo.py -q`
- [ ] 提交：`test: add one readable defrost demo`

这不是建立黄金文件体系，只为大规模删除代码时保留一个肉眼和数值都能核对的样例。

## Task 2：建立五个argparse入口

**Files:**
- Create: `main_data.py`
- Create: `main_cost.py`
- Create: `main_labels.py`
- Create: `main_train.py`
- Create: `main_evaluate.py`
- Create: `tests/test_main_help.py`

- [ ] 为五个入口分别编写 `get_args()`；暂时调用现有实现，不复制业务逻辑。
- [ ] `main_data.py`公开：`--action add|validate|render`、`--input`、`--dataset`、`--cycle`、`--fetch-images`。
- [ ] `main_cost.py`公开：`--cost`、`--dataset`、`--artifact`、`--output`、`--plot`。
- [ ] `main_labels.py`公开：`--cost`、`--dataset`、`--cost-csv`、`--task binary|three-class`、`--regret`、`--output`。
- [ ] `main_train.py`公开：`--experiment`、`--representation`、`--camera`、`--modality`、`--model`、`--jobs`、`--wandb`、`--output`。
- [ ] `main_evaluate.py`公开：`--run`、`--split experiment`、`--figures`、`--output`。
- [ ] 只测试五个 `--help` 都能退出0，不测试帮助文字：

```python
@pytest.mark.parametrize("script", [
    "main_data.py", "main_cost.py", "main_labels.py",
    "main_train.py", "main_evaluate.py",
])
def test_main_help(script):
    result = subprocess.run([sys.executable, script, "--help"])
    assert result.returncode == 0
```

- [ ] 运行：`uv run pytest tests/test_main_help.py -q`
- [ ] 提交：`feat: add five readable experiment entrypoints`

## Task 3：扁平化Dataset

**Files:**
- Create: `dataloader/load.py`
- Create: `dataloader/cycles.py`
- Create: `dataloader/images.py`
- Create: `dataloader/channels.py`
- Modify: `main_data.py`
- Delete after migration: `src/frost_analysis/dataset/`
- Delete after migration: dataset相关 `scripts/`
- Test: `tests/test_data.py`

- [ ] 将当前Loader收敛为四个直接函数：

```python
import json
from pathlib import Path

import pandas as pd


def list_cycles(dataset):
    catalog = json.loads((Path(dataset) / "cycle_catalog.json").read_text())
    return [row["cycle_name"] for row in catalog["cycles"]]


def load_cycle(dataset, cycle):
    return pd.read_parquet(Path(dataset) / "cycles" / f"{cycle}.parquet")


def load_cycle_table(dataset):
    catalog = json.loads((Path(dataset) / "cycle_catalog.json").read_text())
    return pd.DataFrame(catalog["cycles"])


def load_images(dataset, cycle=None):
    images = pd.read_parquet(Path(dataset) / "image_metadata.parquet")
    return images if cycle is None else images.loc[images["cycle_name"].eq(cycle)]
```

- [ ] `cycles.py`只保留：添加日期、阶段划分、候选周期生成和简单有效性列。它继续写出当前
  `cycle_catalog.json`、`image_metadata.parquet`和`cycles/<cycle>.parquet`结构；本次迁移不调用
  写入路径，只在demo副本验证。
- [ ] `images.py`只保留云端直连下载、时间匹配和panel生成；删除多层provider/contract封装。
- [ ] `channels.py`保存原始列到论文变量的唯一映射，以及水侧制热量、COP、变化率等少量派生函数。
- [ ] 删除代码中的schema升级状态机，但不修改现有Dataset文件和字段；Loader只读取当前格式，
  缺少实际计算所需列时一次性报出列名。
- [ ] 用demo做两个测试：能加载循环；阶段按时间单调排列。
- [ ] 运行：`uv run pytest tests/test_data.py -q`
- [ ] 提交：`refactor: flatten dataset workflow`

## Task 4：扁平化成本函数并外置参数

**Files:**
- Create: `cost/energy.py`
- Create: `cost/candidates.py`
- Create: `cost/legacy.py`
- Create: `cost/ticket.py`
- Create: `cost/functions.py`
- Create: `model_artifacts/cost/v1.json`
- Create: `model_artifacts/cost/legacy.json`
- Create: `model_artifacts/cost/v267.json`
- Create: `model_artifacts/cost/v267_validation.csv`
- Modify: `main_cost.py`
- Delete after migration: `src/frost_analysis/cost/`
- Delete after migration: `src/frost_analysis/evidence/`
- Delete after migration: `src/frost_analysis/exploration/`
- Delete after migration: `scripts/cost/`
- Delete after migration: `scripts/exploration/`
- Test: `tests/test_cost.py`

- [ ] `energy.py`只保留相邻有效点积分、水侧制热量、unit-side制热量；最长有效间隔为函数参数，默认5 s。
- [ ] `candidates.py`从稳定制热起点到实际准备开始按分钟生成候选点，并计算每个点的 (E_H,Q_H)。
- [ ] `legacy.py`将V1、V2–V2.6.6、V3和renewal_water改成短函数；共同计算只调用 `energy.py`，版本差异在函数正文中显式可见。
- [ ] `ticket.py`保留V2.6.7的五个输入、两个Ridge目标和公式：

```python
def cost_v267(candidates, artifact):
    e_ticket = predict(candidates, artifact["targets"]["E_T"])
    q_ticket = predict(candidates, artifact["targets"]["Q_T"])
    result = candidates.copy()
    result["J"] = (result["E_H"] + e_ticket) / (result["Q_H"] + q_ticket)
    result["supported"] = in_support(result, artifact)
    return result
```

- [ ] LOEO和bootstrap作为 `main_cost.py --fit` 时执行的模型评估，不再建立Evidence工作流。输出只保留cost CSV、validation CSV和图。
- [ ] 从Python常量导出真实系数到JSON，然后删除 `selected.py`中的数字表。
- [ ] `tests/test_cost.py`只保留四项：注册表能切换；分母公式数值正确；demo三版最优点与冻结表一致；LOEO训练实验不包含留出实验。
- [ ] 运行：`uv run pytest tests/test_cost.py -q`
- [ ] 提交：`refactor: expose cost functions as argparse plugins`

## Task 5：扁平化标签生成

**Files:**
- Create: `labels/build.py`
- Create: `labels/images.py`
- Modify: `main_labels.py`
- Delete after migration: `src/frost_analysis/labels/`
- Delete after migration: `scripts/labels/`
- Test: `tests/test_labels.py`

- [ ] `build.py`输入一个成本CSV和任务定义，输出逐图标签；不再读取候选曲线、manifest和其他隐式中间文件。
- [ ] 二分类规则直接写在函数附近：最优点前为0，最优点后为1；1%近优区间的排除/并入方式由 `--regret` 和一个清楚的parser参数控制。
- [ ] `images.py`根据Dataset图像索引完成六机位匹配和最优点图片导出。
- [ ] 只测试同一循环中三张图片：最优点前、区间内、最优点后。
- [ ] 运行：`uv run pytest tests/test_labels.py -q`
- [ ] 提交：`refactor: make label generation a direct step`

## Task 6：扁平化模型训练与语义实验矩阵

**Files:**
- Create: `model/features.py`
- Create: `model/models.py`
- Create: `model/train.py`
- Create: `model/evaluate.py`
- Create: `configs/experiments.yaml`
- Modify: `main_train.py`
- Modify: `main_evaluate.py`
- Delete after migration: `src/frost_analysis/training/`
- Delete after migration: `scripts/training/`
- Test: `tests/test_model.py`

- [ ] `features.py`只负责加载图片、调用预训练backbone和拼接传感器；删除feature shard作为用户必须理解的概念。缓存存在时内部复用，不在README中暴露。
- [ ] `models.py`放全部可选分类头和RGB+sensor融合网络；parser通过 `choices=MODELS`切换。
- [ ] `train.py`只保留按experiment留出、训练、每任务追加一行 `progress.jsonl`、保存模型和预测。W&B只是可选观察器。
- [ ] `evaluate.py`读取一个run目录，输出 `metrics.csv`、`predictions.parquet`和按机位/实验汇总。
- [ ] 将 `rgb_experiment_manifest.csv`改为以下语义配置，不保留A/B/C/MATRIX内部编号：

```yaml
primary_model:
  question: Can RGB images identify whether the optimal defrost time has passed?
  task: binary
  primary_metric: macro_f1
  fixed:
    representation: resnet50
    camera: all
    modality: rgb
    model: logistic

camera_view_ablation:
  question: Which camera view is sufficient?
  task: binary
  primary_metric: macro_f1
  fixed:
    representation: resnet50
    modality: rgb
  grid:
    camera: [top, left, front, all]
    model: [logistic, random_forest, rbf_svm]
```

- [ ] `main_train.py --experiment primary_model`加载一组fixed参数；grid只用 `itertools.product`展开，不建立配置类。
- [ ] 不传`--experiment`时，所有训练选择都直接来自parser默认值或显式CLI参数；YAML只减少批量实验的命令长度，不能增加CLI中不存在的隐式选项。
- [ ] 只测试：按experiment拆分无交集；一个demo fold能完成；断点文件能跳过已完成task。
- [ ] 运行：`uv run pytest tests/test_model.py -q`
- [ ] 提交：`refactor: simplify rgb training and experiment definitions`

## Task 7：论文图与结果直接映射

**Files:**
- Create: `plotter/style.py`
- Create: `plotter/figure_1_cost.py`
- Create: `plotter/figure_2_labels.py`
- Create: `plotter/figure_3_rgb_increment.py`
- Create: `plotter/figure_4_failure.py`
- Create: `plotter/figure_5_models.py`
- Create: `plotter/figure_6_concentration.py`
- Modify: `main_evaluate.py`
- Delete after migration: `src/frost_analysis/figures/`
- Delete after migration: `scripts/figures/`
- Test: `tests/test_plotter.py`

- [ ] 每个figure文件第一段docstring写清论文问题、输入CSV和输出PNG/PDF。
- [ ] 每个figure文件包含可直接运行的 `get_args()`；默认读取正式output路径。
- [ ] `style.py`只放颜色、字号和单栏/双栏尺寸，不放数据处理。
- [ ] `main_evaluate.py --figures all`按1到6顺序调用绘图函数。
- [ ] 只测试Figure 1使用demo数据能生成一张非空PNG。
- [ ] 运行：`uv run pytest tests/test_plotter.py -q`
- [ ] 提交：`refactor: map plotting code directly to paper figures`

## Task 8：建立单一研究分析区

**Files:**
- Create: `analysis/cost_comparison.py`
- Create: `analysis/recovery_energy.py`
- Create: `analysis/ticket_stability.py`
- Create: `analysis/sensor_correlations.py`
- Create: `analysis/camera_ablation.py`
- Create: `analysis/sensor_fusion.py`

- [ ] 将现有成本函数版本比较和早除霜归因保留到`cost_comparison.py`，统一通过
  `COST_FUNCTIONS`运行，不复制任何成本公式。
- [ ] 将recovery能耗/制热量跨循环比较保留到`recovery_energy.py`，只使用
  `load_cycle_table()`和共享画图风格。
- [ ] 将V2.6.7 LOEO、bootstrap和支持域图放到`ticket_stability.py`，读取
  `model_artifacts/cost/v267_validation.csv`，不重新实现训练。
- [ ] 将T3、Pe、COP、压机频率及变化率分析收敛到`sensor_correlations.py`。
- [ ] 将机位消融和RGB+传感器融合分别放到`camera_ablation.py`和`sensor_fusion.py`，
  直接调用`train_experiment()`和`evaluate_run()`。
- [ ] 每个脚本的parser只暴露该研究真正变化的参数，默认输出到对应
  `output/analysis/<study>/`。
- [ ] 分别运行六个脚本的`--help`；再用demo运行`recovery_energy.py`，确认产生一个CSV和一张PNG。
- [ ] 提交：`refactor: keep research analyses outside the main workflow`

## Task 9：删除旧架构和工程防御

**Files:**
- Delete: `src/`
- Delete: `scripts/`
- Delete: `paper_workflow/`
- Delete: 已被论文或README吸收的重复 `docs/`、`report/`材料
- Reorganize: `output/test/`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [ ] 删除 `src/frost_analysis`、旧CLI和镜像脚本层；先用 `rg "frost_analysis|scripts/"`确认五个入口不存在残余导入。
- [ ] 已经被Task 8六个问题脚本吸收的历史分析代码直接删除，不迁移到新archive；未被继续使用的
  分析从Git历史查看。
- [ ] `pyproject.toml`移除setuptools build配置、mypy、type stubs、coverage和复杂度检查；保留运行依赖、pytest、ruff。
- [ ] ruff只检查 `E,F,I,B`；不以类型和圈复杂度规则迫使代码增加包装。
- [ ] 删除以下测试类型：帮助文字、路径命名、异常消息、schema版本迁移、原子写入、rollback、hash、重复警告传播、内部函数调用次数。
- [ ] 将`output/test/成本函数`中被最终方法引用的CSV和图迁入`output/成本函数`或
  `output/figures`，仍用于扩展研究的证据迁入对应`output/analysis/<study>`；将已复用的RGB特征
  移入`output/model/_cache`；删除失败smoke、旧partial CSV和已被正式结果替代的重复图片。
  不要移动或删除`dataset/`中的图片。
- [ ] 最终tests只保留：`test_demo.py`、`test_main_help.py`、`test_data.py`、`test_cost.py`、`test_labels.py`、`test_model.py`、`test_plotter.py`。
- [ ] 运行：`uv run pytest tests -q`，目标约10–15个测试，不追求覆盖率数字。
- [ ] 提交：`refactor: remove package and workflow scaffolding`

## Task 10：重写README为唯一入口

**Files:**
- Modify: `README.md`
- Create or rename: `paper/`
- Delete: 与README和论文重复的仓库级报告入口

- [ ] 第一行改为具体研究任务：

```markdown
# Image-guided optimal defrost timing for air-source heat pumps
```

- [ ] 紧接着用一张流程图说明：

```text
Cycle sensor data -> Cost function -> Image labels -> RGB/sensor model -> Evaluation
```

- [ ] README只给以下顺序，不介绍内部历史：

```bash
uv sync --extra ml
uv run python main_data.py --action validate
uv run python main_cost.py --cost v2.6.7 --plot
uv run python main_labels.py --cost v1 --task binary
uv run python main_train.py --experiment primary_model --jobs 6
uv run python main_evaluate.py --run output/model/latest --figures all
```

- [ ] 明确解释当前V2.6.7只用于成本窗口识别，bootstrap未通过前不作为硬标签；因此README标签示例暂用V1。
- [ ] 给出一张“参数怎么换”表：成本函数、机位、特征、分类头、传感器融合都对应哪个 `parser.add_argument`。
- [ ] 给出Figure 1–6与 `plotter/figure_*.py`的一一映射。
- [ ] 在README末尾增加“继续研究”一节，只列`analysis/`六个按问题命名的脚本；明确它们
  不是主流程，也不能被主线import。
- [ ] 提交：`docs: make README the only onboarding path`

## Task 11：最终最小验收

- [ ] 运行五个 `--help`，确认不用理解内部术语即可选择算法。
- [ ] 使用demo顺序运行data、cost、labels、train、evaluate。
- [ ] 比较 `frost_cycle_000070` 的V1、V2.5、V2.6.7最优点与迁移前冻结表一致。
- [ ] 生成`/private/tmp/defrost_data_after.txt`并用`cmp`确认现有`data/`、`dataset/`路径、
  文件大小和修改时间完全未变。
- [ ] 检查所有新产物只位于`output/成本函数`、`output/label`、`output/model`、
  `output/figures`和`output/analysis`，且不存在四层以上目录。
- [ ] 运行全部约10–15个测试和ruff。
- [ ] 用 `rg`确认不存在 `src/frost_analysis`、`python -m frost_analysis`、旧scripts路径和嵌入的大段拟合系数。
- [ ] 确认工作树只包含计划内变更，然后提交：`refactor: finish readable research workflow`

---

## 6. 代码审阅标准

完成后按以下标准判断，而不是按“是否保留了旧架构”判断：

| 标准 | 目标 |
|---|---:|
| README到第一张demo成本图 | 3条命令以内 |
| 对外入口 | 5个 |
| 成本函数切换 | 1个 `--cost` 参数 |
| 主入口文件 | 从上到下能看完参数、调用顺序和输出，不把业务逻辑藏进wrapper |
| 普通领域文件 | 一个文件回答一个问题；不以行数迫使拆分 |
| 稳定主线Python总量 | 删除重复实现和历史防御；不以目标行数代替可读性审阅 |
| analysis扩展 | 一个问题一个脚本，结论失效或进入主线后删除重复实现 |
| 测试 | 约10–15个科学/主流程测试 |
| 必须阅读的内部术语 | 不再出现feature shard、Evidence bundle、contract version等 |
| 论文图映射 | Figure 1–6逐文件对应 |

如果某段代码只能解释“未来也许需要”，直接删除。若某项检查不能防止数据泄漏、错误成本公式、错误标签或训练无法恢复，也直接删除。
