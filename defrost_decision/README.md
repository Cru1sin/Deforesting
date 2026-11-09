# 候选 COP 与最优除霜时机

[返回阶段 3 验收](../README.md#3-cop-与最优时机)

入口是 [optimal_defrost_time.py](../optimal_defrost_time.py)。读取循环测量、冻结 Ridge 与 RB 时刻，输出候选表和逐循环汇总。

## 计算与阅读顺序

| 文件／函数 | 职责 |
| --- | --- |
| [candidate_quantities.py](candidate_quantities.py) `calculate_cycle` | 构建候选时刻，取得累计量与事件电耗预测 |
| [performance_objectives.py](performance_objectives.py) `calculate_cycle_cop` | 计算 COP 并标记测量、物理及预测支持 |
| 同文件 `add_single_objective_optima`、`select_cycle_optimum` | 在有效动作网格上选峰值及其连续近优区间 |
| [selection_results.py](selection_results.py) `summarize_decisions` | 输出逐循环最优／RB 对比 |
| [baselines](baselines/README.md) | 冻结 RB 规则与电量积分 |

当前 COP = 累计有效供热量 ÷（本循环恢复及制热实测电耗＋预测准备及除霜电耗）。供热不按出水温度截断，事件有效供热按零计。公式和边界详见 [科学定义](../docs/README.md)。

连续 1% 近优区间是覆盖所有 COP 不低于峰值 99% 的有效可选时刻的最小时间区间；内部允许低于阈值，不因局部下降而拆分。近优命中仍按具体时刻的 COP 判断，无效候选不参与选点。新结果以 `near_optimal_interval_rule=cover_all_near_optimal_actions` 记录该定义，冻结历史结果保持原样。模型外推只作标记，不单独排除候选；测量和物理支持仍须成立。

## 输入与结果

Dataset 测量为 1 s；候选默认间隔 10 s，动作网格按运行设置确定。先看 `cycle_comparison.csv`，再用 `candidate_decisions.csv` 的 `cycle_cop_eligible`、`on_action_grid`、`is_selected` 核对选点。

RB 时刻直接读取 `dataset/priors/rb_reference.csv`，对应 COP 使用本次模型计算。视觉冻结输入保留其发布时的模型来源，不随新 COP 自动改写。

普通计算只写 `--output`。只有显式 `--figures --publish-dataset` 才发布图和更新 Dataset；研究端操作见 [Dataset 说明](../dataset/README.md)。并列目标各自选点，交叉评价不反过来修改选点。
