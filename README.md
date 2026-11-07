# 基于视觉的热泵除霜时机优化

数据预处理 → 除霜电耗模型 → COP 与最优时机 → 视觉决策模型。四个阶段可使用已有输入独立验收。

**GitHub 仓库仅含源码、测试与说明，不含 Dataset、DINO 特征或模型权重。** 下述演示需要另行放好交付数据；只有源码时可先阅读各阶段或运行不依赖真实数据的测试。

## 先运行一次冻结演示

在项目根目录安装环境，然后运行：

```bash
uv sync --extra ml --extra dev
uv run --no-sync python main_defrosting_model.py
```

默认读取 `dataset/`、正面冻结权重和 `exp_20260920`，不训练、不下载照片。结果直接显示在终端，并保存到本次 `temp/test/frozen_model_<时间>/`，可重复运行。

预期：循环 `frost_cycle_000327` 在 **16:46:14** 触发，策略 COP **1.794667**、RB COP **1.781829**，达到冻结参考最优 COP 的 99%。这是单循环运行验收；整体性能看完整测试集。

需随源码提供数值 Dataset、DINO 特征和[正面权重](output/02_模型与消融/部署模型_六机位/正面_部署模型.pkl)。交付清单见下文；已有环境时跳过安装。

## 分别验收四个阶段

以下命令均在项目根目录执行，默认读取 `dataset/`。阶段 2、3 的命令更新各自测试目录，阶段 4 每次创建新目录。

### 1. 数据预处理

输入是原始运行数据和通道定义；输出是循环、阶段及 **1 s** 测量表。先根据运行状态识别循环，再按通道聚合、线性插值，并对供热功率、Pe、T3 作过去 **60 s** 平滑。插值使用后续观测，整条处理链属于离线处理。

先验收已有数据：

```bash
uv run --no-sync python validate_dataset.py
```

终端显示 `dataset valid` 表示数值结构检查通过。打开[示例循环](dataset/cycles_RB/frost_cycle_000327.csv)：

| 列 | 看什么 |
| --- | --- |
| `timestamp`、`cycle_stage` | 1 s 时间网格、阶段变化；边界与运行状态是否对应 |
| 原通道、`_interpolated`、`_processed` | 原始值、插值值、平滑值的变化 |
| `is_interpolated` | 该行是否包含插值，避免把补值当作直接观测 |

结构检查不替代阶段边界的科学核验。[数据入口与导入方法](dataset_tools/README.md) → [循环识别与通道处理](dataset_tools/builder/README.md)。

### 2. 除霜电耗模型

输入：压缩机频率、环境温度、盘管温度。三输入 Ridge 输出**准备＋除霜电耗（kWh）**。

先看[验证汇总](dataset/除霜预测模型/三输入Ridge-两阶段-287-1-统一平滑/validation_summary.csv)和[逐事件预测](dataset/除霜预测模型/三输入Ridge-两阶段-287-1-统一平滑/model_validation.csv)。当前 **287 事件、47 实验**，按实验留出预测 RMSE 约 **0.004155 kWh**。

需要重新拟合时：

```bash
uv run --no-sync python defrosting_electricity_model.py --output temp/test/ridge_acceptance
```

| 输出：`temp/test/ridge_acceptance/` | 看什么 |
| --- | --- |
| `validation_summary.csv` | `mae`、`rmse` 是预测误差，`bias` 是平均偏差；单位均为 kWh |
| `model_validation.csv` | 比较实测 `E_prep_D_kwh` 与预测 `predicted_E_prep_D_kwh` |
| `candidate_model_parameters.json` | COP 阶段读取的模型参数 |

[事件筛选、Ridge 与验证实现](defrost_event_models/README.md)。

### 3. COP 与最优时机

输入：循环测量、冻结 Ridge 和 [RB 时刻先验](dataset/priors/rb_reference.csv)。先算各候选 COP，再选峰值及其所在的连续 **1% 近优区间**。

```bash
uv run --no-sync python optimal_defrost_time.py --cycles frost_cycle_000327 \
  --output temp/test/cop_acceptance --overwrite
```

先打开 `temp/test/cop_acceptance/cycle_comparison.csv`：

| 字段 | 含义及当前示例结果 |
| --- | --- |
| `t_star`、`COP_eff_selected` | 最优时刻 **16:36:04**，COP **1.800238** |
| `t_RB`、`COP_eff_RB` | RB 时刻 **17:27:41**，COP **1.781830** |
| `COP_relative_change_pct` | 相对 RB 的 COP 提升，约 **1.0331%** |
| `cycle_cop_basin_1pct_start/end` | 连续近优区间 **16:24:24–17:16:54** |

再看同目录 `candidate_decisions.csv`：每行一个候选时刻。用 `cycle_cop_eligible` 与 `on_action_grid` 判断可选点，`is_selected` 标记最终选点。

默认使用已冻结 Ridge。要接入阶段 2 的新模型，增加 `--model-file temp/test/ridge_acceptance/candidate_model_parameters.json`；新模型结果不要求等于上表。[COP 公式与选点实现](defrost_decision/README.md)。

### 4. 视觉决策模型

输入：已发布动作与标签、**384 维 DINO＋7 项运行特征**，以及冻结权重。直接运行首部命令即可验收；无需原图。

| 输出：终端打印的 `temp/test/frozen_model_<时间>/` | 看什么 |
| --- | --- |
| `cycle_metrics.csv` | `selected_time` 为预测时刻；`selected_cop`、`rb_cop` 比较策略；`gain_vs_rb_pct` 为提升；`near_optimal_hit` 表示是否达到冻结参考最优 COP 的 99% |
| `checkpoint_summary.csv` | 当前权重的评价汇总 |
| `cycle_screening.csv` | 请求实验中各循环是否准入及原因 |

当前示例相对 RB 提升约 **0.7205%**。第三阶段给出参考最优时刻，第四阶段给出模型预测时刻，二者不必相同。视觉 CDF 在首末近优点间过渡，与 COP 峰值连续区间分开定义。

更换权重或实验：`python main_defrosting_model.py evaluate --checkpoint <权重> --experiments <实验>`。机位自动读取，训练实验重叠或显式机位错配会拒绝评价。

重新训练才执行 `uv run --no-sync python main_defrosting_model.py train`，默认输出 `output/90_复现材料/01_视觉模型训练/主模型/`。[模型、默认配置与支线](image_models/README.md)。

## 交付与路径

- `dataset/`：三个根级元数据文件、`cycles/`、`cycles_RB/`、`cycles_original/`、完整 `training_data/`、`image_DINO_feature/`、`priors/rb_reference.csv`、所需 Ridge 模型。
- 权重：保留首部链接中的相对位置。Dataset 和权重需另附，Git 源码不包含完整数据。
- 结霜图像只交付 DINO 特征；不打包 `images/`、照片索引、原图链接或外接盘内容。不能只复制一个演示循环，Loader 会读取该机位完整发布输入。
- 默认路径以项目根目录为准；显式相对路径以当前工作目录为准。主入口打印实际目标，`--output` 指定本次目录。

## 继续读哪里

| 目录 | 职责 |
| --- | --- |
| [dataset_tools](dataset_tools/README.md) → [builder](dataset_tools/builder/README.md) | 导入、循环识别、测量处理、读取与输入发布 |
| [heat_smoothing](heat_smoothing/README.md) | 共享插值与平滑 |
| [defrost_event_models](defrost_event_models/README.md) | 事件电耗 Ridge |
| [defrost_decision](defrost_decision/README.md) → [baselines](defrost_decision/baselines/README.md) | 候选 COP、选点与 RB |
| [image_models](image_models/README.md) | 标签、时序模型、训练、评价和研究支线 |
| [plots](plots/README.md) | 共享出图方法 |
| [tests](tests/README.md)、[tools](tools/README.md) | 自动测试、研究核验与结果维护 |
| [Dataset](dataset/README.md)、[数据格式](DATASET_FORMAT.md)、[科学定义](docs/README.md) | 文件含义、发布步骤与公式 |

霜态与最优时机的关系属于补充分析，不是主线。`research/` 保存论文和汇报，`archive/` 保存旧工程材料；两者及 `temp/` 均不属于演示必需内容。
