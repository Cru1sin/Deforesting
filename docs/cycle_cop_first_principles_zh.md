# 循环有效 COP 的物理定义

[返回项目入口](../README.md) · [完整技术文档](technical_report.tex)

## 计算对象与边界

循环分为恢复、稳定制热、准备与除霜。记循环制热起点为 $t_0$，稳定制热起点为 $t_r$，候选准备时刻为 $t$。当前方法从 $t_r$ 累计有效供热，从 $t_0$ 累计实测电耗，再加入该候选状态下预测的准备及除霜电耗：

$$
Q_{\mathrm{use}}(t)=\frac{1}{3600}\int_{t_r}^{t}\dot Q_{\mathrm{processed}}(\tau)\,d\tau,
$$

$$
E_{\mathrm{meas}}(t)=\frac{1}{3600}\int_{t_0}^{t}P_{\mathrm{total}}(\tau)\,d\tau,
\qquad
COP_{\mathrm{eff}}(t)=\frac{Q_{\mathrm{use}}(t)}{E_{\mathrm{meas}}(t)+\widehat E_{\mathrm{prep+D}}(t)}.
$$

功率单位为 kW，时间为秒，积分为 kWh，COP 无量纲。供热读取 `heating_capacity_processed`，不按出水温度截断。恢复阶段不累计有效供热；准备与除霜有效供热按零计，这是本指标的定义。当前恢复电耗已经进入实测项，不再额外加入预测恢复电耗。

## 准备与除霜电耗

三输入 Ridge 使用压缩机频率、环境温度和盘管温度，标准化为 $z(t)$ 后预测：

$$
\widehat E_{\mathrm{prep+D}}(t)=\beta_0+\boldsymbol\beta^\top z(t).
$$

该预测随候选运行状态变化。已有拟合实验使用排除该实验的留出模型，新实验使用全量模型。单次准备与除霜电耗由累计有效供热分摊，提前除霜的电耗已计入 COP 分母，无需另加人为的除霜次数惩罚。

## 最优时刻与近优区间

在测量、特征和预测有效的动作集合上选择最大 COP；并列取最早时刻。连续 1% 近优区间是覆盖所有 COP 不低于峰值 99% 的有效可选时刻的最小时间区间，内部允许低于阈值，不因局部下降而拆分。近优命中仍按具体时刻的 COP 判断；区间不改变候选有效性。

测量为 1 s；默认 COP 候选为 10 s，视觉发布输入使用 30 s 动作。RB 时刻直接读取 `dataset/priors/rb_reference.csv`，其 COP 用本次模型计算。当前计算只在已有制热记录支持的范围内搜索，候选之后的准备与除霜电耗由 Ridge 估计。

视觉训练读取发布时保存的参考和标签，在视觉动作网格的连续近优区间内构造线性增加的 CDF 目标。重算 COP 不自动更新冻结输入和模型；候选网格与视觉动作网格的区间边界按各自有效可选时刻计算。

## 对应实现

- [candidate_quantities.py](../defrost_decision/candidate_quantities.py)：累计供热、实测电耗和候选事件电耗。
- [performance_objectives.py](../defrost_decision/performance_objectives.py)：计算 COP、有效性、峰值和连续近优区间。
- [event_electricity_ridge.py](../defrost_event_models/event_electricity_ridge.py)：实验隔离的 Ridge 拟合与预测。
- [training_data.py](../defrost_event_models/training_data.py)：共享测量积分、事件监督与动作前特征。
