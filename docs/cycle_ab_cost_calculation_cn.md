# 第一版：只用现有长 cycle 计算最优启动除霜点 \(a\)

> **本版唯一目标：**用已有系统点位，为每个长 Heating cycle 计算一个“在当前固定 5 min 除霜策略下的经验经济最优启动点”。本版不计算最优停止除霜点 \(b\)，不截取 Defrost 尾段构造反事实，也不训练 RGB 模型。

---

## 0. 先锁定符号和结论

本文统一使用：

\[
\boxed{
a=\text{Heating}\rightarrow\text{Defrost 的最优启动边界}
}
\]

\[
\boxed{
b=\text{Defrost}\rightarrow\text{Heating 的最优停止边界}
}
\]

若后续视觉状态定义为：

\[
x=0\text{ 表示 clean},\qquad x=1\text{ 表示严重结霜},
\]

那么通常应有：

\[
\boxed{a>b.}
\]

旧版本中的 \(a<b\) 不再适用。

当前数据能够支持的结论是：

\[
\boxed{
t_{a,i}^{*}
=
\text{cycle }i\text{ 在当前固定时长除霜策略下的经验最优启动时间}
}
\]

以及在经济最优时间算完以后，读取对应 RGB 状态：

\[
\boxed{
a_i^*=x_i(t_{a,i}^{*}).
}
\]

当前数据**不能**支持：

\[
\boxed{b_i^*=\text{严格的经济最优停止边界}.}
\]

因为所有真实 cycle 都只在固定 300 s 后切回 Heating。数据没有告诉我们：“如果在第 120、180 或 240 s 提前停止，残霜、残水和恢复供热会怎样。”固定 300 s 终点是控制器的实际动作，不是最优 \(b\)。

因此第一版故意不追求形式上的双边界对称：

\[
\boxed{
\text{先可靠计算 }a;\qquad b\text{ 暂不估计。}
}
\]

---

## 1. 为什么只用现有长 cycle 就能先估计 \(a\)

每条长 Heating trajectory 已经真实经历：

\[
\text{clean heating}
\rightarrow
\text{轻度结霜}
\rightarrow
\text{严重结霜}.
\]

所以对任意候选启动时刻 \(\tau\)，从稳定 Heating 起点到 \(\tau\) 的运行成本都是**已经发生、可以直接积分的真实数据**。

如果在 \(\tau\) 启动除霜，会支付一次除霜代价。第一版不尝试预测“不同霜量对应的除霜轨迹”，只用历史完整 cycle 估计一张固定门票：

\[
\boxed{
\text{一次典型固定 5 min Defrost + 恢复供热的平均代价}.
}
\]

于是候选 \(\tau\) 的选择变成一句话：

> 除霜太早，会频繁支付固定除霜门票；除霜太晚，带霜 Heating 的供热损失越来越大。使长期平均代价最低的时刻，就是第一版的最优启动点。

这正是现有数据最容易闭合的部分。它不要求额外实验，也不要求先有 RGB 状态。

---

## 2. 成本函数：电功率 + 供热服务缺口

### 2.1 为什么不能只用电功率

当前数据的阶段统计已经给出直接反例：

| 阶段 | 水侧 \(Q_h\) 中位数 | 整机电功率 \(P_{el}\) 中位数 |
|---|---:|---:|
| Frost development | \(+6.62\) kW | \(2.93\) kW |
| Defrost | \(-11.68\) kW | \(0.85\) kW |

若只最小化 \(P_{el}\)，Defrost 看起来比 Heating 更“便宜”。但 Defrost 时水侧 \(Q_h<0\)，机组正在停止供热，甚至从用户水侧取热。

因此只看能耗会得出错误方向。第一版必须同时计算：

1. 机器实际用了多少电；
2. 用户少获得了多少热。

### 2.2 水侧实际供热量

对每个 10 s 采样点计算：

\[
\boxed{
Q_h(t)=
1000\times4.186\times
\frac{\dot V_w(t)}{3600}
\left[T_{w,out}(t)-T_{w,in}(t)\right]
}
\]

当：

- \(\dot V_w\) 的单位为 m³/h；
- 水温单位为 °C；

则 \(Q_h\) 的单位为 kW。

需要使用的现有点位为：

| 符号 | 物理意义 | 现有 channel | 单位/处理 |
|---|---|---|---|
| \(P_{el}\) | 整机实际电功率 | `power_total` | kW |
| \(\dot V_w\) | 水侧体积流量 | `water_flow` | 先确认原始单位为 m³/h |
| \(T_{w,in}\) | 水侧入口温度 | `water_in_temperature` | °C |
| \(T_{w,out}\) | 水侧出口温度 | `water_out_temperature` | °C |
| mode | 当前 Heating/Defrost 状态 | `defrost_active`, `operating_mode` | 用于切段 |
| \(Q_h\) | 用户侧实际得到的热功率 | 由上述水侧点位计算 | Defrost 中允许为负 |

不要用控制器内部的 `heating_capacity` 替代 Defrost 中的 \(Q_h\)。它可以在 Heating 段做交叉核验，但不能代表反向除霜时用户侧真实热流。

### 2.3 最简 \(Q_{ref}(t)\)：每个 cycle 两个 clean anchors

\(Q_{ref}(t)\) 表示：

> 在相近工况下，如果盘管是 clean 的，用户本来应该得到多少供热功率。

长 cycle 内水温和环境可能漂移，所以不建议整条轨迹只使用开头一个常数。第一版也不拟合大型 baseline 模型，只使用每个完整 cycle 自带的两个 clean anchors：

1. Heating 开始、启动瞬态结束后的稳定 60 s；
2. Defrost 结束并恢复稳定后的 clean Heating 60 s。

分别取水侧供热量中位数：

\[
Q_{0,i}=\operatorname{median}(Q_h\text{ in start-clean 60 s}),
\]

\[
Q_{1,i}=\operatorname{median}(Q_h\text{ in recovered-clean 60 s}).
\]

设两个窗口中心时刻为 \(t_{0,i}^{ref}\) 和 \(t_{1,i}^{ref}\)，则在两者之间做线性插值：

\[
\boxed{
Q_{ref,i}(t)=
Q_{0,i}
+
\frac{t-t_{0,i}^{ref}}
{t_{1,i}^{ref}-t_{0,i}^{ref}}
(Q_{1,i}-Q_{0,i}).
}
\]

目的只有一个：避免把长时间实验中的缓慢工况漂移全部误认为结霜损失。

第一轮只使用具有两个可靠 clean anchors 的完整 cycles。不要为了多保留几个 cycle，立即增加回归模型和特殊补值规则。若后续发现两点线性参考误差明显，再升级 condition-matched baseline。

### 2.4 用一个数值代表用户侧供热服务损失

定义全数据共用的 clean-state COP：

\[
\boxed{
COP_0=
\operatorname{median}
\left(
\frac{Q_h}{P_{el}}
\right)_{\text{all clean anchors}}.
}
\]

然后只使用一个换算系数：

\[
\boxed{
\lambda_Q=\frac{1}{COP_0}.
}
\]

其单位和解释为：

\[
\lambda_Q:
\quad
\frac{\mathrm{kWh_e}}{\mathrm{kWh_{th}}},
\]

即：少供 1 kWh 热量，按 clean heat pump 的效率折算，相当于多少 kWh 电量。

根据当前 clean anchors 的初步统计：

\[
COP_0\approx2.56,
\qquad
\lambda_Q\approx0.39.
\]

正式计算必须从清洗后的数据重算，不要把 0.39 硬编码。

这个 \(\lambda_Q\) 应准确称为：

\[
\boxed{\text{供热服务缺口的等效电量换算系数}.}
\]

它不是 PMV、室温偏差或真实人体不满意度模型。当前没有建筑热惯性、室内温度响应和用户偏好数据，就不应声称已经计算真实 thermal comfort。第一版只把“少供热”作为用户侧代理量。

### 2.5 唯一的瞬时成本

先定义供热服务缺口：

\[
\boxed{
s_{Q,i}(t)=
[Q_{ref,i}(t)-Q_{h,i}(t)]_+,
}
\]

其中：

\[
[y]_+=\max(y,0).
\]

然后定义：

\[
\boxed{
g_i(t)=
P_{el,i}(t)
+
\lambda_Q s_{Q,i}(t).
}
\]

单位为 kW 等效电功率。

这条式子可直接通俗地读成：

> 当前真实电功率，加上当前少供热量按 clean COP 折算的等效电功率。

采用 positive part 的原因是：超过参考需求的热不用于无限抵消其他时刻的供热中断。Defrost 中若 \(Q_h<0\)，则 \(Q_{ref}-Q_h\) 会自然变大，无需再增加人为的 `defrost penalty`。

这借用了两块已有结构：

- Zhe Wang 参与的建筑 Economic MPC：经济量与服务/舒适违约共同进入目标；
- Wei Wang 等的 ASHP optimal initiation：以 nominal heating-energy loss 表达结霜和除霜损失。

这里是针对现有点位的最小改写，不声称照搬任一论文的原始目标函数。

### 2.6 为什么第一版不再加入 \(P_{ref}\)

建筑 Economic MPC 通常直接计算设备能耗，再对服务违约加罚。为保持这一结构，第一版使用实际 \(P_{el}\)，不再增加 \(P_{el}-P_{ref}\) 的第二套 baseline。

这样只需要建立一个 \(Q_{ref}\)，少一个模型、少一个误差来源。对候选启动点而言，健康 Heating 的正常功率会成为共同背景；真正改变 optimum 的是固定除霜门票与后期持续上升的供热缺口。

若后续需要专门报告“相对 clean operation 的额外损失”，可另画 \(g-P_{ref}\) 作为解释曲线，但第一版 optimizer 不需要它。

---

## 3. 从历史 cycle 计算一张固定除霜门票

### 3.1 门票包含什么

对历史 cycle \(j\)，定义事件窗口：

\[
\Omega_{D,j}
=
[\text{Defrost start},\ \text{Heating recovery stable}].
\]

窗口从 `defrost_active` 上升沿开始，经过完整固定 300 s Defrost，并在切回 Heating 后达到恢复稳定时结束。

第一版可把“恢复稳定”定义为：

\[
Q_h(t)\ge0.9Q_{ref}(t)
\]

连续满足 3 个 10 s 点，同时 mode 已回到 Heating。

使用完整 `Defrost + immediate recovery` 而不是只积分 300 s，是为了避免漏掉固定控制策略真实产生的恢复代价。这里不拟合 recovery 与停止状态的关系，只把实际观察到的恢复成本一起压进固定门票。

### 3.2 每个历史事件的两笔账

实际电能：

\[
\boxed{
E_{D,j}=
\int_{\Omega_{D,j}}P_{el,j}(t)\,dt.
}
\]

用户侧少收到的热量：

\[
\boxed{
H_{D,j}=
\int_{\Omega_{D,j}}
[Q_{ref,j}(t)-Q_{h,j}(t)]_+\,dt.
}
\]

一次事件的等效成本：

\[
\boxed{
K_{D,j}=E_{D,j}+\lambda_QH_{D,j}.
}
\]

其中：

- \(E_{D,j}\)：kWh 电；
- \(H_{D,j}\)：kWh 热；
- \(\lambda_QH_{D,j}\)：kWh 等效电；
- \(K_{D,j}\)：kWh 等效电。

事件时长为：

\[
T_{D,j}=|\Omega_{D,j}|.
\]

### 3.3 全局固定门票

对通过质量检查的 \(N_D\) 个历史 Defrost 事件，主分析使用算术平均：

\[
\boxed{
\bar K_D=
\frac{1}{N_D}\sum_{j=1}^{N_D}K_{D,j},
\qquad
\bar T_D=
\frac{1}{N_D}\sum_{j=1}^{N_D}T_{D,j}.
}
\]

使用平均值的原因是：\(\bar K_D\) 与 \(\bar T_D\) 要代表“下一次事件的期望门票和期望占用时间”。

同时报告中位数版本作为抗异常值检查：

\[
K_D^{med}=\operatorname{median}_jK_{D,j},
\qquad
T_D^{med}=\operatorname{median}_jT_{D,j}.
\]

若平均值和中位数差异很大，应先找坏 cycle，而不是立刻发明更复杂的 robust model。

### 3.4 这张门票代表什么、又不代表什么

它准确表示：

> 如果现在开始除霜，假定仍执行当前固定 5 min Defrost 策略，则支付一个由历史事件估计的典型经济成本和占用时间。

它没有表示：

> 不同霜量下真正最短、最优的 Defrost trajectory。

固定 300 s 很可能包含过度除霜，因此 \(\bar K_D\) 可能偏高，并把第一版最优启动时间推迟。这个结果应理解为：

\[
\boxed{
\text{current fixed-duration defrost policy 下的 empirical optimum}.
}
\]

这不是缺陷被隐藏起来，而是第一版有意选择的可计算边界。将来一旦获得可靠的最优停止边界 \(b\)，只需替换这张门票，不必推翻启动点计算框架。

---

## 4. 对每个长 Heating cycle 计算 \(a_i^*\)

### 4.1 先在时间上优化，不先用 RGB 阈值

对 cycle \(i\)：

- \(t_{0,i}\)：启动瞬态结束后的稳定 clean Heating 起点；
- \(t_{D,i}^{actual}\)：实验中实际开始 Defrost 的时刻；
- \(\tau\)：一个候选“如果现在开始 Defrost”的时间。

候选集合第一版固定为：

\[
\boxed{
\tau\in
\{t_{0,i}+10\text{ min},
t_{0,i}+11\text{ min},\ldots,
t_{D,i}^{actual}\}.
}
\]

即每 60 s 检查一次。前 10 min 只用于排除刚进入稳定段后的不现实高频切换；如果机组已有明确最小 Heating dwell time，应直接用设备约束替换 10 min。

先搜索 \(\tau\) 有两个好处：

1. Heating 的 \(t_{0,i}\rightarrow\tau\) 是完全真实发生的数据；
2. RGB 不参与“最优”的定义，避免用 RGB 生成标签、再让 RGB 学自己。

### 4.2 Heating 累计成本

对候选 \(\tau\)：

\[
\boxed{
C_{H,i}(\tau)=
\int_{t_{0,i}}^{\tau}g_i(t)\,dt.
}
\]

\(C_{H,i}\) 的单位为 kWh 等效电。

Heating 时长为：

\[
\boxed{
T_{H,i}(\tau)=\tau-t_{0,i}.
}
\]

### 4.3 候选启动时刻的长期平均成本

如果系统反复执行：

\[
\text{clean Heating}
\rightarrow
\tau\text{ 启动 Defrost}
\rightarrow
\text{固定 5 min Defrost + recovery}
\rightarrow
\text{clean Heating},
\]

则一个更新周期的平均成本为：

\[
\boxed{
\rho_i(\tau)=
\frac{
C_{H,i}(\tau)+\bar K_D
}{
T_{H,i}(\tau)+\bar T_D
}.
}
\]

其中：

- 分子：一次 Heating 加一次固定除霜事件共付出多少；
- 分母：这个完整过程共持续多久；
- \(\rho_i\)：长期平均等效电功率，单位 kW。

必须除以总时长。若只比较累计成本，较长 Heating 几乎必然累计更多，不能公平比较不同启动时刻。

### 4.4 最优启动时间与启动边界

第一版最优启动时间：

\[
\boxed{
t_{a,i}^*=\arg\min_{\tau}\rho_i(\tau).
}
\]

只有这一步完成后，才从同步 RGB 状态读取：

\[
\boxed{
a_i^*=x_i(t_{a,i}^*).
}
\]

即：

\[
\boxed{
\text{system economic cost}
\rightarrow
t_{a,i}^*
\rightarrow
\text{RGB state }a_i^*.
}
\]

而不是：

\[
\text{RGB threshold}\rightarrow\text{宣称经济最优}.
\]

即使当前还没有可靠的 \(x(t)\)，也完全可以先得到 \(t_{a,i}^*\)。RGB 是后续监督学习的观察量，不是第一版成本计算的前提。

### 4.5 为什么它会出现内部 minimum

在 \(\bar K_D,\bar T_D\) 固定且数据连续时，对 \(\rho_i(\tau)\) 求导，符号由下式决定：

\[
\boxed{
g_i(\tau)-\rho_i(\tau).
}
\]

因此内部 optimum 满足：

\[
\boxed{
g_i(t_{a,i}^*)=\rho_i(t_{a,i}^*).
}
\]

通俗解释：

> 当前继续 Heating 的损失速度，刚好等于整个“Heating + 一次固定除霜”的平均损失速度。此前继续 Heating 可以摊薄固定除霜门票；此后结霜损失太快，再拖就不划算。

若晚期 \(g_i(t)\) 总体上升且只穿过 \(\rho_i(t)\) 一次，\(\rho_i\) 就有唯一内部 minimum。若成本始终下降、minimum 总贴在实际 Defrost 时刻，说明现有 cycle 还没有跨过 optimum，或成本定义/基准存在问题；不能强行给标签。

### 4.6 不把一个 60 s 网格点伪装成精确真值

除最小点外，同时输出 5% near-optimal band：

\[
\boxed{
\mathcal B_i=
\{\tau:\rho_i(\tau)\le1.05\rho_i(t_{a,i}^*)\}.
}
\]

它告诉我们：最优点附近是否是一条宽谷。若 \(\mathcal B_i\) 很宽，结论应是“在这一段内启动近似等价”，而不是报告到 10 s 的伪精度。

---

## 5. 10 s 数据的直接计算步骤

### Step 1：只保留可用 cycles

一个 cycle 要进入第一轮计算，至少满足：

1. 有完整稳定 Heating 到实际 Defrost start；
2. 有 `power_total`、水流量、进出水温；
3. 有 start-clean 和 recovered-clean 两个可靠 60 s anchors；
4. mode 标记可识别；
5. Defrost 后能找到 recovery stable。

缺少关键 passage 的 cycle 直接排除，不做复杂插补。单点或不超过 20 s 的同 mode 缺口可以线性插值，但不能跨 mode 边界插值。

### Step 2：计算每个时刻的物理量

逐 10 s 计算：

\[
Q_h(t),\qquad Q_{ref}(t),\qquad s_Q(t),\qquad g(t).
\]

积分时：

\[
\Delta t=\frac{10}{3600}\ \mathrm{h}.
\]

例如：

\[
C_{H,i}(\tau)
\approx
\sum_{k:t_{0,i}\le t_k\le\tau}
g_i(t_k)\frac{10}{3600}.
\]

梯形和或矩形和都可以；10 s 分辨率下不需要复杂积分器。

### Step 3：先从全部历史事件得到固定门票

逐事件计算：

\[
E_{D,j},\quad H_{D,j},\quad K_{D,j},\quad T_{D,j},
\]

然后计算：

\[
\bar K_D,\quad\bar T_D.
\]

### Step 4：对每个 Heating cycle 扫描候选 \(\tau\)

每 60 s 计算一次：

\[
C_{H,i}(\tau),\quad
T_{H,i}(\tau),\quad
\rho_i(\tau).
\]

取最小值得：

\[
t_{a,i}^*,\quad \mathcal B_i.
\]

### Step 5：最后才读取 RGB 状态

若已有全数据共用尺度的 \(x(t)\)，读取：

\[
a_i^*=x_i(t_{a,i}^*).
\]

禁止每个 cycle 单独 min-max 到 \([0,1]\)。否则所有轨迹都会被人为拉成相同范围，跨 cycle 的 \(a_i^*\) 集中没有意义。

### Step 6：每个 cycle 输出一行结果

建议结果表至少包含：

| 字段 | 含义 |
|---|---|
| `cycle_id` | cycle 编号 |
| `t_heating_stable` | \(t_{0,i}\) |
| `t_actual_defrost` | 实验实际启动 Defrost 时间 |
| `t_a_star` | 计算得到的 \(t_{a,i}^*\) |
| `a_star_rgb` | 可选，\(a_i^*=x_i(t_{a,i}^*)\) |
| `rho_min` | \(\min\rho_i\) |
| `near_opt_start`, `near_opt_end` | 5% near-optimal band |
| `minimum_location` | `interior` 或 `left/right boundary` |
| `valid` | 是否通过质量检查 |
| `invalid_reason` | 失败原因，不硬补 |

`b_star` 字段第一版应留空或明确写 `not_identifiable`，不能填实际 300 s 终点。

---

## 6. 80 个 cycle 之间怎样互相提供信息

跨 cycle 第一版只共享三个东西：

1. 全局 \(COP_0\)，从所有 clean anchors 估计；
2. 固定除霜门票 \((\bar K_D,\bar T_D)\)，从所有有效历史 Defrost 事件估计；
3. 后续使用的统一 RGB 状态尺度 \(x\)。

每个 Heating trajectory 自己提供自己的 \(C_{H,i}(\tau)\)，因此仍能得到每个 cycle 的 \(t_{a,i}^*\)。

第一轮先画所有：

\[
\{t_{a,i}^*\},\qquad\{a_i^*\}.
\]

真正希望看到：

\[
CV(a_i^*)<CV(t_{a,i}^*),
\]

即“最优时间随工况变化较大，但对应的视觉状态更集中”。只有出现这个结果，才有理由进一步定义全局 RGB 启动边界。

若后续要在共同候选状态 \(u\) 上求全局经济边界，应使用 pooled ratio：

\[
\boxed{
\rho_{pool}(u)=
\frac{
\sum_i\left[C_{H,i}(t_i^H(u))+\bar K_D\right]
}{
\sum_i\left[(t_i^H(u)-t_{0,i})+\bar T_D\right]
}.
}
\]

然后：

\[
a_{global}^*=\arg\min_u\rho_{pool}(u).
\]

这一步只有在统一 \(x\) 已验证、多个 cycles 真实穿过共同 \(u\) 时才做。不能用 \(\operatorname{median}_i\rho_i\) 冒充严格的全局 renewal objective；median curve 只能做抗异常值的描述图。

在后续机器学习 train/test 划分时，\(COP_0\)、固定门票和全局边界应只用训练 cycles 估计，避免测试信息泄漏。第一轮物理可行性诊断可先使用全部有效 cycles。

---

## 7. 对另一份意见的取舍

### 现在采纳

| 建议 | 是否采纳 | 原因 |
|---|---|---|
| 先搜索候选时间 \(\tau\)，再读取 RGB 状态 | 采纳 | 经济成本产生标签，RGB 不参与定义最优性 |
| 用 renewal 平均成本 \(C/T\) | 采纳 | 不同候选 Heating 时长不同，必须按时间公平比较 |
| 全局 objective 用 ratio of sums | 采纳 | 对应长期总成本/总时间，而不是 median of ratios |
| 把结果称为 empirical optimum | 采纳 | 固定门票是历史近似，不是每个候选动作的直接实验 |
| 检查 \(\rho_i(\tau)\) 是否存在内部 minimum | 采纳 | 没有内部 minimum 就没有数据证据支持最优启动点 |

### 第一版不采纳

| 建议 | 暂不采纳原因 |
|---|---|
| 用 RGB 状态 \(x_\tau\) 匹配 Defrost trajectory | 相同可见霜状态不代表相同盘管温度、冷媒分布和切换瞬态 |
| 从实际 Defrost 中截取 \(x_\tau\rightarrow\)clearance 的 tail | 这是一条没有真实执行过的反事实；当前既然选择固定历史门票，就不要同时保留第二套估计 |
| 拆分复杂的 \(K(a)\)、entry model、recovery model | 当前数据不识别这些函数，增加参数只会制造伪精确 |
| 用 clearance boundary 代替 \(b^*\) | “看起来清霜”不等于经济最优停止，尤其未观测提前停止后的残水和恢复代价 |
| 现在直接给出全局 RGB 阈值 | 应先看每个 cycle 的 \(a_i^*\) 是否在统一视觉尺度上集中 |

最重要的删减是：

\[
\boxed{
\text{固定历史除霜门票}
\quad\text{与}\quad
\text{Defrost tail matching}
\text{ 二选一。}
}
\]

第一版选择前者，因为它假设更少、实现更短、结果更容易解释。

---

## 8. 第一轮必须画的四张图

### 图 1：原始供热与瞬时成本

对 5 个代表 cycles 画：

\[
Q_h(t),\quad Q_{ref}(t),\quad P_{el}(t),\quad g(t).
\]

应看到：clean Heating 的供热接近 reference；晚期结霜时服务缺口总体增加；Defrost 时 \(g\) 显著升高。

目的：先确认成本函数没有单位、符号或 baseline 错误。

### 图 2：历史除霜门票分布

画：

\[
E_{D,j},\quad
\lambda_QH_{D,j},\quad
K_{D,j},\quad
T_{D,j}.
\]

同时标出 mean 和 median。

目的：确认“固定门票”是否是合理近似。若离散极大，先按工况分组检查，不要立即拟合高维函数。

### 图 3：每个 cycle 的 \(\rho_i(\tau)\)

标出：

\[
t_{a,i}^*,\quad
\mathcal B_i,\quad
t_{D,i}^{actual}.
\]

目的：这是方案是否成立的核心图。多数曲线应存在不贴候选区边界的最低区域。

### 图 4：最优时间和事后视觉状态的跨-cycle 分布

分别画：

\[
t_{a,i}^*,\qquad a_i^*.
\]

目的：检查是否出现“时间分散、状态集中”。若 \(a_i^*\) 并不比时间稳定，就暂时没有证据支持单一 RGB 阈值。

---

## 9. 第一版只做三组敏感性检查

不要全面调参，只检查会改变结论的三个简化：

1. 用户侧换算系数：
   \[
   \lambda_Q\in\{0.5,1,2\}\times\frac1{COP_0};
   \]
2. 除霜门票：mean 与 median 两个版本；
3. 候选时间网格：30、60、120 s。

若 \(t_{a,i}^*\) 在这些设置下只小幅移动，第一版标签才可信。若 optimum 大量跳到左右边界或变化几十分钟，应先回查 \(Q_{ref}\)、水侧热量和 Defrost ticket，而不是训练 CNN。

---

## 10. 第一版参数锁定表

| 项目 | 主分析固定值 | 目的 |
|---|---:|---|
| 原始采样间隔 | 10 s | 使用现有分辨率 |
| clean anchor | 稳定 60 s 中位数 | 压低传感器噪声 |
| recovery stable | \(Q_h\ge0.9Q_{ref}\) 连续 30 s | 结束固定事件窗口 |
| 候选启动网格 | 60 s | 避免 10 s 噪声制造假 minimum |
| 最小 Heating dwell | 10 min 或设备真实约束 | 排除不现实频繁切换 |
| 用户侧系数 | \(\lambda_Q=1/COP_0\) | 唯一等效换算数值 |
| 固定 Defrost 门票 | 有效事件算术平均 | 代表期望事件成本 |
| near-optimal band | 最低值的 5% | 不报告伪精度 |

这些值先锁住跑完第一轮。没有结果证据前，不增加更多参数。

---

## 11. Go / No-Go 判断

满足以下条件，才进入后续 RGB 学习：

1. 大多数 \(\rho_i(\tau)\) 有内部 minimum，而不是贴在候选区左右边界；
2. 最低区域在 \(\lambda_Q\)、门票统计量和时间网格敏感性下基本稳定；
3. 计算得到的 \(t_{a,i}^*\) 明显早于被故意拖长的实际 Defrost 时刻；
4. 使用全数据统一 RGB 尺度后，\(a_i^*\) 的跨-cycle 离散小于 \(t_{a,i}^*\)；
5. 成本曲线的变化能由真实 \(Q_h\) 和 \(P_{el}\) 解释，而不是由 baseline 漂移或坏点产生。

若这些条件不成立，应停在物理成本层查原因。神经网络不能把一个没有稳定经济 minimum 的问题变正确。

---

## 12. 本版所有符号

| 符号 | 含义 | 单位/范围 | 来源 |
|---|---|---|---|
| \(i\) | Heating cycle 编号 | — | 数据索引 |
| \(j\) | 历史 Defrost 事件编号 | — | 数据索引 |
| \(a\) | 最优启动除霜状态边界 | 与 \(x\) 相同 | 由经济最优时间事后映射 |
| \(b\) | 最优停止除霜状态边界 | 与 \(x\) 相同 | 第一版不可识别 |
| \(x(t)\) | 全数据统一的视觉/结霜状态 | 通常 \([0,1]\) | RGB，后续使用 |
| \(t_{0,i}\) | cycle \(i\) 稳定 Heating 起点 | h 或 timestamp | mode 与稳定性判断 |
| \(t_{D,i}^{actual}\) | 实验实际启动 Defrost 时刻 | h 或 timestamp | mode 上升沿 |
| \(\tau\) | 候选启动 Defrost 时刻 | h 或 timestamp | 一维遍历 |
| \(t_{a,i}^*\) | cycle \(i\) 的经验最优启动时间 | h 或 timestamp | \(\arg\min\rho_i\) |
| \(a_i^*\) | \(t_{a,i}^*\) 对应的 RGB 状态 | 与 \(x\) 相同 | \(x_i(t_{a,i}^*)\) |
| \(P_{el}\) | 整机实际电功率 | kW | `power_total` |
| \(\dot V_w\) | 水侧体积流量 | m³/h | `water_flow` |
| \(T_{w,in},T_{w,out}\) | 水侧进、出水温度 | °C | 温度点位 |
| \(Q_h\) | 水侧实际供热功率 | kW | 由流量与温差计算 |
| \(Q_{ref}\) | clean-coil 参考供热功率 | kW | 两个 clean anchors 插值 |
| \(s_Q\) | 未满足的供热功率 | kW | \([Q_{ref}-Q_h]_+\) |
| \(COP_0\) | 全局 clean-state COP | 无量纲 | 所有 clean anchors |
| \(\lambda_Q\) | 供热缺口的等效电量换算系数 | kWh_e/kWh_th | \(1/COP_0\) |
| \(g(t)\) | 瞬时经济成本率 | kW 等效电 | \(P_{el}+\lambda_Qs_Q\) |
| \(\Omega_{D,j}\) | 第 \(j\) 个 Defrost + recovery 窗口 | 时间区间 | mode 与 recovery 条件 |
| \(E_{D,j}\) | 第 \(j\) 个事件实际电能 | kWh | \(\int P_{el}dt\) |
| \(H_{D,j}\) | 第 \(j\) 个事件供热缺口 | kWh_th | \(\int s_Qdt\) |
| \(K_{D,j}\) | 第 \(j\) 个事件总等效成本 | kWh 等效电 | \(E_D+\lambda_QH_D\) |
| \(T_{D,j}\) | 第 \(j\) 个事件占用时间 | h | \(|\Omega_D|\) |
| \(\bar K_D,\bar T_D\) | 历史平均固定门票和时长 | kWh、h | 有效事件算术平均 |
| \(C_{H,i}(\tau)\) | Heating 到候选 \(\tau\) 的累计成本 | kWh 等效电 | \(\int gdt\) |
| \(T_{H,i}(\tau)\) | Heating 到候选 \(\tau\) 的时长 | h | \(\tau-t_0\) |
| \(\rho_i(\tau)\) | 一个更新周期的平均成本 | kW 等效电 | 总成本/总时长 |
| \(\mathcal B_i\) | 5% near-optimal 时间区间 | 时间区间 | 成本曲线 |

---

## 13. 最后压成四条公式

每个时刻：

\[
\boxed{
g_i(t)=
P_{el,i}(t)
+
\frac{[Q_{ref,i}(t)-Q_{h,i}(t)]_+}{COP_0}.
}
\]

历史固定除霜门票：

\[
\boxed{
\bar K_D=
\operatorname{mean}_j
\int_{\Omega_{D,j}}g_j(t)\,dt,
\qquad
\bar T_D=
\operatorname{mean}_j|\Omega_{D,j}|.
}
\]

每个候选启动时间：

\[
\boxed{
\rho_i(\tau)=
\frac{
\int_{t_{0,i}}^{\tau}g_i(t)\,dt+\bar K_D
}{
(\tau-t_{0,i})+\bar T_D
}.
}
\]

第一版最终输出：

\[
\boxed{
t_{a,i}^*=\arg\min_\tau\rho_i(\tau),
\qquad
a_i^*=x_i(t_{a,i}^*),
\qquad
b_i^*=\text{not identifiable in Stage 1}.
}
\]

这就是第一版全部主线。没有二维优化，没有 Defrost tail matching，没有 latent dynamics，也没有 CNN。

---

## 参考文献拼图

1. [Blum, D., Wang, Z., et al. Field demonstration and implementation analysis of model predictive control in an office HVAC system. *Applied Energy* 318, 119104 (2022).](https://doi.org/10.1016/j.apenergy.2022.119104)  
   借用：Economic MPC 中将设备经济量与服务/舒适约束违约共同纳入目标的结构。

2. [Wang, W., Zhang, S., Li, Z., et al. Determination of the optimal defrosting initiating time point for an ASHP unit based on the minimum loss coefficient in the nominal output heating energy. *Energy* 191, 116505 (2020).](https://doi.org/10.1016/j.energy.2019.116505)  
   借用：启动除霜应由 frosting heating loss 与 defrost heating loss 的权衡决定，而不是只看瞬时 COP 或电耗。

> 本文成本函数是对上述两类结构的最小组合，不是任一论文公式的原样复制。
