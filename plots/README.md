# 科研图表

[返回项目验收](../README.md)

图表读取共享计算或已保存结果。先核对数值表，再选择相应 renderer；冻结模型验收不要求生成图片。

| 文件 | 图表内容 |
| --- | --- |
| [publication.py](publication.py) | 统一样式、保存与公共测量面板 |
| [dataset.py](dataset.py)、[cycle.py](cycle.py)、[data_overview.py](data_overview.py) | Dataset 图表发布、循环测量与数据总览 |
| [defrost_decision.py](defrost_decision.py)、[cop.py](cop.py) | 候选 COP、最优时刻和连续近优区间 |
| [event_calibration.py](event_calibration.py)、[rb_rule_explainer.py](rb_rule_explainer.py) | 电耗预测校准与 RB 规则 |
| [model_comparison.py](model_comparison.py)、[external_policy.py](external_policy.py)、[camera.py](camera.py) | 模型、外部实验和机位比较 |
| [frost_state.py](frost_state.py)、[frost_trajectory.py](frost_trajectory.py) | 霜态与轨迹 |
| [frost_cop.py](frost_cop.py)、[frost_comparison.py](frost_comparison.py)、[frequency.py](frequency.py) | 霜态、COP 与运行特征关系 |

Dataset 图经 `manage_dataset render` 发布，COP 图经 `optimal_defrost_time.py --figures` 生成；参数见各入口 `--help`。含照片的图需要原图，数值验收不依赖照片。

同类指标复用 renderer 与源表，图形选项不改变选点或标签。原生 Harmony＋Slingshot 图与其他映射分别标注；COP 峰值连续区间与冻结视觉标签分别解释。
