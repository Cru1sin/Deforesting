# 原始数据经验最优除霜点：论文初稿级 demo

## 结论边界

本阶段直接使用约 1 s 原始水侧与电功率数据，不进行平滑。结果是“当前固定时长除霜策略门票假设下的经验等效能耗最优启动时刻”，不是包含电价、真实热舒适与任意反事实动作的因果全局经济最优点。

## 固定方法

- 水侧制热量：`1.161 × water_flow × (water_out_temperature - water_in_temperature)`，单位 kW。
- clean reference：每循环稳定制热开始后的 60 s 中位数，与下一循环 clean anchor 线性连接。
- clean COP：2.487；热量缺口等效电量系数 `lambda_Q = 1/COP = 0.402`。
- 恢复：除霜结束后原始制热量连续 30 s 达到下一 clean anchor 的 90%。
- 经验门票：50 个有效事件；均值成本 1.018 kWh-eq.，均值时长 13.60 min。
- 候选：稳定制热后 10 min 起，以 1 min 网格搜索，并包含实际除霜时刻。
- 目标：`rho(tau) = [C_H(tau) + mean(K_D)] / [T_H(tau) + mean(T_D)]`。

## 当前结果

- catalog 循环：77；得到有效经验最优点：47。
- 未给出点估计的 30 个循环包括：无实际除霜边界 12 个、catalog 无效 9 个、候选域被长缺口截断 5 个、clean anchor 无效 4 个。
- 内部最小值：37；左边界：0；右边界：10。
- 相对实际除霜的提前量中位数：58.8 min。
- 5% near-optimal 区间宽度中位数：95.0 min。
- 均值门票改为中位数门票后，最优点绝对移动量中位数：0.0 min；90% 分位：36.2 min。
- 双 clean anchor 改为仅用当前 clean anchor 后，最优点绝对移动量中位数：1.0 min；90% 分位：50.4 min；超过 30 min 的循环占比：12.8%。

若右边界最小值占比高，含义是观察区间尚未跨过最优点，不能强制制造内部 optimum。若左边界占比高，需优先检查固定门票是否过低或最小运行时长是否设置过晚。下一阶段是否增加工况条件化门票，只由门票残差诊断决定。

本阶段的 `lambda_Q × thermal_shortfall` 是供热服务缺口的等效电量代理。由于当前数据没有室内空气温度、PMV/PPD、占用人数或暴露时长，它不能被表述为直接测得的热舒适损失。

## 可追溯输出

- `source_data/cycle_optimal_points.csv`：全部循环结果与无效原因。
- `source_data/defrost_ticket_events.csv`：经验除霜门票分解。
- `source_data/candidate_cost_curves.parquet`：每个候选时刻的成本曲线。
- `source_data/clean_anchor_summary.csv`：clean anchor 与 COP。
- `source_data/cohort_audit.csv`：队列纳入审计。
- `source_data/empirical_policy_summary.csv`：经验门票和换算系数。
- `source_data/near_optimal_band_sensitivity.csv`：1%、2%、5%、10% regret 阈值对应的区间宽度。
- `figures/figure_1_empirical_optimal_defrost.*`：PNG/SVG/PDF/TIFF 主图。
- `figures/cycles/`：每个有效循环一张原始曲线与成本曲线图。
