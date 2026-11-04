# 视觉除霜决策模型

[返回阶段 4 验收](../README.md#4-视觉决策模型)

先运行根目录 [main_defrosting_model.py](../main_defrosting_model.py)：不带参数回放冻结模型，`train` 才重新训练。输入是已发布动作、标签、冻结 DINO 和运行特征；训练与评价都不准备原始数据。

## 主流程

| 顺序 | 文件／关键函数 | 输入 → 输出 |
| --- | --- | --- |
| 1 | [inputs.py](inputs.py) `load_data` → [cycle_selection.py](cycle_selection.py) `screen_cv_cycles` | 完整候选与 Catalog → 准入循环、动作和参考曲线 |
| 2 | [DatasetLoader](../dataset_tools/load_dataset.py) `split_deployment_data` | 开发期／后期实验 → 互斥训练与测试集合 |
| 3 | [training.py](training.py) `fit_model` → [policy.py](policy.py) `fit_policy` | 训练输入 → 冻结模型 |
| 4 | [model.py](model.py) | DINO＋运行序列 → 模型输出 |
| 5 | [training_cases.py](training_cases.py) `fit_case` | 预测、评分、保存检查点 |
| 6 | [evaluation.py](evaluation.py) `evaluate_external` | 冻结权重＋未见实验 → 触发时刻、COP、RB 对比 |

辅助职责：[labels.py](labels.py) 定义监督目标；[control_context.py](control_context.py) 读取过去运行状态；[experiments.py](experiments.py) 保存配置；[checkpoints.py](checkpoints.py) 定位权重及机位；[comparisons.py](comparisons.py) 汇总对比。

## 固定主模型

| 设置 | 默认值 |
| --- | --- |
| 输入 | DINOv2 384 维＋`inputs.py::NUMERIC_INPUTS` 的 7 项运行特征 |
| 网络 | 单层 LSTM，宽度 25，FiLM 开启 |
| 训练 | 60 轮，学习率 0.003，权重衰减 0.0001，种子 42 |
| Dropout | 视觉／输出均 0.2，循环内共享掩码 |
| 划分 | 截止 `exp_20260910` 为开发期，后期实验只测试 |
| 触发 | 循环等权 CDF 损失；累计停止概率首次达到 0.5 |

当前正面开发期为 101 循环／29 实验，后期为 12 循环／6 实验；数量由完整候选的实际筛选决定。规则要求 RGB/DINO 覆盖各≥70%、整循环插值比例≤30%、有效终点 COP 低于峰值99%；所有工况和 RB 同一规则。训练与评价先筛完整候选，不复用历史名单。

CDF 在首末近优点之间从 0 过渡到 1，不等同于 COP 峰值连续区间。`near_optimal_hit` 判断预测时刻 COP 是否达到冻结参考最优 COP 的 99%；保存的逐候选标签另用于 `trigger_status` 的时序分类。未触发时按保存的实际准备时刻回退，评价表保留回退和输入缺失标记。霜态等级不作为主模型输入。

60 轮来自既有训练配置，不根据后期测试选轮。开发期含过往调参数据，独立评价指后期实验。改变模型或训练设置属于新实验。

## 保存与复用

默认训练目录：`output/90_复现材料/01_视觉模型训练/主模型/`。

- `training/`：配置与日志；同配置可续跑，更换配置使用新目录。
- `checkpoints/LSTM_seed42/`：权重、训练记录和循环信息。
- `evaluation/external/`：后期实验的逐循环结果与汇总。

评价自动读取权重机位；历史部署权重从其保存的版本标识读取，不凭文件名猜测。机位不符或训练实验重叠时拒绝评价。路径由 [output_layout.py](../output_layout.py) 管理。

## 支线：按研究问题进入

| 研究问题 | 入口与边界 |
| --- | --- |
| 调参／消融 | [optimal_stopping.py](optimal_stopping.py)、[ablation.py](ablation.py)；共享模型和评分；`ablation --baseline` 仅接受历史 LOEO 基准，不接受主模型固定训练／后期测试结果 |
| 选参／选轮 | [model_selection.py](model_selection.py)；只用训练侧验证，测试不参与选择 |
| 六机位比较 | [camera_comparison.py](camera_comparison.py)；必须指定 `--stage`，见下文 |
| 输入核验／DINO | [admission.py](admission.py)、[dinov2_features.py](dinov2_features.py)；研究端准备，照片核验与重提取需要原图 |
| 霜态与最优时机 | [slingshot.py](slingshot.py)、[frost_progression.py](frost_progression.py)；补充分析采用原生 Harmony＋Slingshot，标明来源方法 |
| 表征／决策诊断 | [representation_audit.py](representation_audit.py)、[stopping_diagnostics.py](stopping_diagnostics.py)、[overlap.py](overlap.py) |

六机位命令形式：`uv run --no-sync python -m image_models.camera_comparison --stage <阶段>`。
`index/prepare/train` 可用 `--cameras` 选择机位；`prepare` 会读取原图并更新 Dataset，`train` 使用已发布特征。`summarize/plot/images` 面向完整六机位结果；`all` 包含准备、训练和汇总，不是验收捷径。执行前打印分析、训练、质量与归档目标。

六机位研究以正面开发期为范围，其他机位使用该范围内的有效子集。开发期统一选参后逐实验留一回放属于开发期评价；全开发期训练的部署权重在后期实验独立评价。共同集只限制测试样本，不另训模型。完整配置以各实验 `training/settings.json` 为准。
