# 空气源热泵结霜—除霜控制：从“双时间点”收敛为一个经济滞回策略

> 结论先行：不要分别拟合“最佳开始时间”和“最佳结束时间”。先用 RGB 构造一把能贯穿结霜与融霜的一维“霜负荷尺子” \(x\)，再用真实电耗和真实供热损失求两个状态阈值 \(a^*<b^*\)。每个 cycle 的最佳时间，只是轨迹穿过这两个阈值的时刻。

本文遵循一个原则：**能由一条曲线完成的工作，不交给复杂模型。**

---

## 1. 核心判断

这个 idea 应该 pivot 成：

\[
\boxed{\text{共享霜负荷状态 }x\text{ 上的、经济代价驱动的 renewal hysteresis policy}}
\]

控制规则只有两句：

\[
\begin{aligned}
&x\uparrow b^*:\quad Heating\rightarrow Defrost,\\
&x\downarrow a^*:\quad Defrost\rightarrow Heating,
\end{aligned}
\qquad a^*<b^*.
\]

这里的 \((a^*,b^*)\) 是**状态边界**，不是分钟数；某个 cycle 的 \(t^*_{\mathrm{start},i}\) 和 \(t^*_{\mathrm{end},i}\) 才是**穿越边界的时间**。

真正决定 idea 能否成立的只有两个 bottleneck。

### Bottleneck 1：长轨迹没有直接观测反事实

长 Heating 段只观测了“继续制热会怎样”，没有观测“如果在更早时刻立刻除霜会怎样”。固定 300 s 的 Defrost 段只观测了“继续除霜会怎样”，没有观测“如果提前恢复制热会怎样”。因此：

- 单条长 frosting trajectory 不能无条件给出 \(t^*_{\mathrm{start}}\)；
- 单条长 defrost trajectory 也不能无条件给出 \(t^*_{\mathrm{end}}\)；
- 从一条经验曲线上找 minimum，不等于解决了 optimal switching。

现有 cycle 能互相补齐这些反事实，必须满足一个可检验的条件：存在一个 H/D 共用、近似 Markov 的霜负荷状态 \(x\)。换句话说，在相同 \(x\)、相同 mode 和相近工况下，当前的成本和霜变化速度不应再明显依赖“来自哪个 cycle、之前结过多厚的霜、已经除霜多久”。

这也是本研究的第一道 go/no-go：**若一个标量 \(x\) 不能做到这一点，就不能声称现有数据已经识别出双边界。**

### Bottleneck 2：必须真的量到“供热服务”

经济最优不能只用盘管温度、蒸发压力和图像白度拼一个分数。所幸当前数据合同已经包含：

- `power_total`：整机电功率；
- `water_flow`、`water_in_temperature`、`water_out_temperature`：可计算水侧供热量；
- `ambient_temperature`、环境湿度、水温和压缩机频率：可匹配 clean-coil 参考工况；
- `defrost_active` 与 `operating_mode`：可切分 H/D mode。

因此第二个 bottleneck 在通道层面已经基本解决。Defrost 时尤其应从水侧计算允许为负的 (Q_h)，而不是直接相信可能截断负值的控制器 `heating_capacity`。

本地 Dataset 当前登记 62 个 cycle，其中 53 个为 `valid`，36 个同时具有有效 frosting 与 defrost RGB。后续新数据可以沿同一方法加入，但当前分析不应把尚未进入 Dataset 的“80 个”当成已核验样本量。

---

## 2. 通俗解释

### 为什么要开始除霜？

霜刚出现时，它对制热的影响很小。此时除霜反而会停掉正常供热、让四通阀换向，还要消耗电能，所以太早除霜很亏。

霜继续变厚后，风量和换热越来越差。机组每多坚持一分钟 Heating，少送出的热和浪费的电越来越多。到了某个状态，继续带霜运行的损失，已经大于现在付一次除霜代价。这个状态就是上边界 \(b^*\)。

所以 \(b^*\) 不是“霜出现了没有”，而是：

> **霜造成的下一小段运行损失，是否已经大到值得支付一次切换和除霜代价。**

### 为什么要结束除霜？

除霜开始时，盘管上霜多，继续除霜一分钟可以清掉很多霜，这一分钟很值。

到尾部时，大部分霜已经清掉，继续输入的热更多是在加热盘管金属、空气和残余水；与此同时，室内仍没有正常供热，甚至可能从水侧拿走热量。于是继续除霜的收益越来越小，代价仍在累积。两者相等的位置就是下边界 \(a^*\)。

太早停在 \(x>a^*\)，残霜会立刻拖累下一轮 Heating；太晚停在 \(x<a^*\)，是在用停止供热的代价追求几乎没有价值的“更干净”。

### 为什么不是同一个阈值？

每次 H/D 换向都有真实代价。如果刚切到 Defrost 就在同一点切回 Heating，系统会抖动，换向损失会无限累积。最优策略必然允许状态在一个区间内来回走：

\[
a^*\;\xleftarrow{\quad Defrost\quad}\;x\;\xrightarrow{\quad Heating\quad}\;b^*.
\]

因此两条边界只有**控制逻辑对称**，物理来源并不对称：

- \(b^*\) 主要由严重结霜时的 Heating 性能恶化决定；
- \(a^*\) 主要由 Defrost 尾部收益衰减和下一轮残霜损失决定。

整个框架可以用一句话解释：

> **什么时候继续待在当前 mode，已经不如付出切换代价进入另一个 mode 划算？**

---

## 3. 文献拼图

下面只保留真正闭合数学链条的六块。没有一篇论文直接给出本文完整方案；价值在于它们分别解决了链条中的一段。

| Paper | 已有方法或结论 | 在本研究中的对应模块与最小修改 |
|---|---|---|
| [Wang et al., *Energy* 191, 116505 (2020)](https://doi.org/10.1016/j.energy.2019.116505) | 用 nominal output heating energy 的损失系数寻找 optimal defrost initiating point，说明“固定除霜损失”与“继续结霜损失”之间确实会形成最优启动点。 | 保留其 full-cycle loss 思想；不把最优点写成固定时间，而是改写为共享霜状态上的上边界 (b^*)，并把终止边界纳入同一个 renewal cycle。 |
| [Klingebiel et al., *Energy* 324, 135871 (2025)](https://doi.org/10.1016/j.energy.2025.135871) | 比较常见 controller 与 optimal initiation，显示最优启动会随环境和 heating capacity 改变，固定时间或固定传感器阈值存在明显损失。 | 证明不能直接学习一个“第几分钟除霜”；先用 clean-coil 参考量归一化，再检查同一状态边界在不同工况下是否稳定。 |
| [Song et al., *Energy Procedia* 105, 335–342 (2017)](https://doi.org/10.1016/j.egypro.2017.03.323) | 实验展示终止温度过高会延长除霜并加热环境，过低则残留更多水、影响下一 cycle；其设备得到约 20–25 °C 的合适范围。 | 采用“尾部边际收益下降、下一 cycle 有残留代价”的物理结构；不照搬 22 °C，因为它是设备相关经验阈值，不是普适最优边界。 |
| [Zheng et al., *Sustainable Cities and Society* 51, 101667 (2019)](https://doi.org/10.1016/j.scs.2019.101667) | 经典图像处理得到 frosting coefficient，并用两个经验图像阈值分别启动和终止除霜。 | 直接借用“RGB 可成为 H/D 共用霜状态尺子”这一成熟结构；把经验阈值替换成由电耗与供热损失求得的 (a^*,b^*)。 |
| [Drgoňa et al., *Annual Reviews in Control* 50, 190–232 (2020)](https://doi.org/10.1016/j.arcontrol.2020.09.001) | 建筑 MPC 中常把 energy 与 thermal-service/comfort violation 放进同一 stage cost。 | 只借用经济目标的计账原则；本研究不实现完整 MPC，因为一维单调过程可以直接化为一个 renewal quotient。 |
| [Liu et al., *European Journal of Operational Research* 263, 879–887 (2017)](https://doi.org/10.1016/j.ejor.2017.05.006) | 在退化系统中，随状态恶化而上升的 operating cost 会导出 condition-based control-limit policy。 | 作为跨领域结构类比：结霜是退化、除霜是维护。本文仍对热泵的一维双向 dynamics 给出自己的阈值存在与唯一性证明。 |

文献拼出的最短链条是：

\[
\text{全周期供热损失}
\rightarrow
\text{状态化而非定时化}
\rightarrow
\text{除霜尾部收益递减}
\rightarrow
\text{RGB 共享状态}
\rightarrow
\text{经济 stage cost}
\rightarrow
\text{双 control limits}.
\]

---

## 4. 最终数学框架

### 4.1 最小 two-mode system

系统状态只保留两部分：

\[
s_t=(x_t,m_t),\qquad
m_t\in\{H,D\}.
\]

- \(x_t\in[x_L,x_U]\)：归一化霜负荷，越大表示盘管上的霜越严重。它来自固定 coil ROI 的 RGB frost coefficient，而不是由成本反推；
- \(m_t=H\)：Heating；\(m_t=D\)：Defrost，来自 `defrost_active` 和 `operating_mode`。

在一个近似不变的 operating envelope 内，写成：

\[
\dot x=
\begin{cases}
f_H(x)>0, & m=H,\\
-f_D(x)<0, & m=D.
\end{cases}
\]

\(f_H\) 是结霜速度，\(f_D\) 是除霜速度。二者不要求相等，也不要求线性。证明只要求在实际使用的状态区间内连续且为正。若接近完全清霜时 \(f_D\to0\)，就把 \(x_L\) 设在图像仍能分辨变化的最低位置；不在不可辨识区间硬求阈值。

### 4.2 一个不需要任意权重的 stage cost

先从现有水侧通道计算真实供热量：

\[
Q_h(t)=\rho_w c_p\frac{\dot V_w(t)}{3600}
\left[T_{w,out}(t)-T_{w,in}(t)\right].
\]

- \(\dot V_w\)：`water_flow`，单位 m³/h；
- \(T_{w,out},T_{w,in}\)：`water_out_temperature` 与 `water_in_temperature`；
- \(\rho_w,c_p\)：该水温范围内的水密度和比热，可首轮取常数；
- \(Q_h\) 单位 kW，Defrost 时允许为零或负。

令：

- \(P_{el}(t)\)：`power_total`，整机电功率；
- \(Q_{ref}(t)\)：相同环境温湿度、水温、流量和压缩机频率下，clean-coil Heating 应提供的热量；
- \(P_{ref}(t)\)：同一 clean-coil 参考状态下的电功率；
- \(COP_{ref}=Q_{ref}/P_{ref}\)。

\(Q_{ref},P_{ref}\) 不需要新模型。用各 cycle 的 early-stable clean 段，按上述工况做分箱或最近邻平均即可。

定义“等效电功率”成本：

\[
\boxed{
\ell_m(t)
=P_{el,m}(t)
+\frac{[Q_{ref}(t)-Q_{h,m}(t)]_+}{COP_{ref}(t)}
}
\]

其中 \([z]_+=\max(z,0)\)。第一项是真实用电；第二项把“少送出的热”换算成一台 clean heat pump 为补回这些热所需的等效电功率。两项单位都是 kW，因此不需要主观地调 \(w_E,w_Q\)。若乘上电价，只是改变单位，不改变 optimum。

模式切换还有一个有限正成本：

\[
\kappa_{HD}>0,\qquad \kappa_{DH}>0,
\qquad K=\kappa_{HD}+\kappa_{DH}.
\]

它表示四通阀换向、短时无有效供热和恢复过程造成的等效能量损失。数据处理中把短切换窗口从连续 H/D 曲线拟合中拿出，单独积分成 \(\kappa\)，避免重复计账。数学上把这几秒折叠成一次 impulse cost；这是一项明确的近似，不把机械磨损再人为加权。

长期目标是单位时间平均等效电耗：

\[
\boxed{
J_\pi=
\limsup_{T\to\infty}
\frac{
\int_0^T \ell_{m_t}(x_t)\,dt
+\sum_k \kappa_{m_{k^-}m_{k^+}}
}{T}
}
\]

这已经是一个完整的 optimal switching problem。对应的平均成本 Bellman 条件可写成：

\[
\min\{\ell_H-\rho+f_HV'_H,\ \kappa_{HD}+V_D-V_H\}=0,
\]

\[
\min\{\ell_D-\rho-f_DV'_D,\ \kappa_{DH}+V_H-V_D\}=0.
\]

但本问题不需要数值解这组方程。因为 \(x\) 在 H 中只向上走、在 D 中只向下走，可以直接化为一个 renewal cycle。

### 4.3 从局部 dynamics 到一个可画出来的函数

若策略在低状态 \(a\) 从 D 切回 H，在高状态 \(b\) 从 H 切到 D，一个完整 cycle 的平均成本为：

\[
\boxed{
\rho(a,b)=
\frac{
K+
\displaystyle\int_a^b\frac{\ell_H(x)}{f_H(x)}dx+
\displaystyle\int_a^b\frac{\ell_D(x)}{f_D(x)}dx
}{
\displaystyle\int_a^b\frac{1}{f_H(x)}dx+
\displaystyle\int_a^b\frac{1}{f_D(x)}dx
}}
\]

定义：

\[
g(x)=\frac{\ell_H(x)}{f_H(x)}+\frac{\ell_D(x)}{f_D(x)},
\qquad
q(x)=\frac{1}{f_H(x)}+\frac{1}{f_D(x)},
\]

\[
\boxed{r(x)=\frac{g(x)}{q(x)}}.
\]

\(r(x)\) 的直觉很简单：在霜负荷 \(x\) 附近多走一个极小的“结一点霜、再融回去”的往返，平均每秒要付出多少等效电功率。

### 4.4 先说清楚：弱假设本身不够

“H 中霜增加、D 中霜减少、成本连续、切换有代价”只足以让问题定义良好，**不能自动证明唯一双阈值**。连续函数完全可能有两个低谷；例如让 \(r(x)\) 具有两个谷底，所有 dynamics 和成本仍可连续，却会出现多个候选 operating bands。

因此必须增加一个最弱、且能由现有数据直接检验的结构假设：

> **Single-well 条件：**存在唯一 \(x_0\)，使 \(r(x)\) 在 \([x_L,x_0]\) 上严格下降，在 \([x_0,x_U]\) 上严格上升。

它的物理含义是：

- 靠近清霜端，继续 Defrost 的清霜速度越来越慢，但无供热成本仍存在，所以 (r) 高；
- 严重结霜端，继续 Heating 的供热损失很大，所以 (r) 也高；
- 中间状态两边都没那么差，于是只有一个谷底。

如果数据不支持这个 U 形，本文就不能先拟合一个 minimum 再称它为定理；正确结论应是“双阈值结构尚未被现有状态定义支持”。

### 4.5 阈值结构、存在性、唯一性与 hysteresis 的证明

下面的结论是**相对于上述假设完备的**。它不是声称任何热泵在任何状态定义下都必然有双阈值。

#### 命题 1：为什么 stationary policy 可化为两个边界

考虑只依赖当前 \((x,m)\)、不允许无限快抖动的 recurrent policy。

- 在 H 中，\(x\) 严格增加。无论“切到 D”的状态集合写得多复杂，轨迹实际只会碰到从当前低点向上遇到的第一个点，记作 \(b\)；更高的切换点永远到不了。
- 在 D 中，\(x\) 严格减少。同理，实际只会碰到向下遇到的第一个切换点，记作 \(a\)。
- 一个可重复 cycle 必须满足 \(a<b\)。正切换成本排除了在同一点来回切换。

所以任何有效的 stationary recurrent controller，在实际轨迹上都等价于一对 first-hitting boundaries \((a,b)\)。在一维 Markov 状态成立时，历史不提供额外信息；多个重复 excursion 的长期平均又是各 excursion 平均成本的时间加权平均，不会低于其中成本最低的重复 excursion。因此只需最小化 \(\rho(a,b)\)。

#### 命题 2：最优解存在

假设 \(f_H,f_D>0\) 且连续，\(\ell_H,\ell_D\) 连续，状态区间紧致。则 \(g,q\) 连续且 \(q>0\)，所以 \(\rho(a,b)\) 在 \(x_L\le a<b\le x_U\) 上连续。

当 \(b-a\to0\) 时，cycle 时间趋于 0，而 \(K>0\)，因此 \(K/T_{cycle}\to\infty\)。最优解不会塌缩成无限快切换。把一个足够小的对角邻域排除后，可行域是紧集；由连续函数在紧集上取到最小值，\((a^*,b^*)\) 存在。

#### 命题 3：内部最优边界满足同一条水平线

记 \(\rho=\rho(a,b)\)。直接求偏导：

\[
\frac{\partial\rho}{\partial a}
=\frac{q(a)}{T_{cycle}}[\rho-r(a)],
\]

\[
\frac{\partial\rho}{\partial b}
=\frac{q(b)}{T_{cycle}}[r(b)-\rho].
\]

若 optimum 不贴物理端点，一阶条件给出：

\[
\boxed{r(a^*)=r(b^*)=\rho^*.}
\]

再由“总成本 \(=\rho^*\times\) 总时间”得到面积条件：

\[
\boxed{
K=\int_{a^*}^{b^*}q(x)[\rho^*-r(x)]dx.
}
\]

因此两个边界就是 U 形曲线 \(r(x)\) 与一条水平线 \(r=\rho^*\) 的左右交点；两交点之间的加权面积恰好支付一次 H→D→H 的切换成本。

#### 命题 4：什么条件下边界唯一

在 single-well 条件下，对任意 \(r(x_0)<\rho<\min\{r(x_L),r(x_U)\}\)，水平线 \(r=\rho\) 在左右两侧各有唯一交点 \(a(\rho),b(\rho)\)。定义：

\[
A(\rho)=\int_{a(\rho)}^{b(\rho)}q(x)[\rho-r(x)]dx.
\]

端点处 integrand 为零，因此 Leibniz 求导时端点项消失：

\[
A'(\rho)=\int_{a(\rho)}^{b(\rho)}q(x)dx>0.
\]

所以 \(A(\rho)=K\) 至多有一个解。若 \(K\) 小于水平线到达物理端点前可形成的最大面积，就存在唯一内部解 \((a^*,b^*)\)。若 \(K\) 太大，最优边界会贴到 \(x_L\) 或 \(x_U\)，相应的一阶等式变成单侧不等式；不能假装仍有两个内部交点。

这个内部解还是全局最优。因为对任意其他区间 ([a,b])：

\[
N(a,b)-\rho^*T(a,b)
=K+\int_a^b q(x)[r(x)-\rho^*]dx.
\]

在 \([a^*,b^*]\) 内，\(r-\rho^*<0\)，其全部负面积恰好为 \(-K\)；区间外 \(r-\rho^*\ge0\)。任何其他 \([a,b]\) 要么没有包含全部负面积，要么额外包含正面积，因此上式严格大于 0。故 \(\rho(a,b)>\rho^*\)，唯一性得证。

#### 命题 5：为什么必然形成 hysteresis

若 \(K>0\)，面积条件右侧必须严格为正，于是 \(b^*>a^*\)。若 \(K\to0\)，所需面积趋于 0，两边界趋向 U 形谷底 \(x_0\)，才可能重合。

因此 hysteresis 不是为了控制方便额外加的 debounce，而是正切换成本的数学结果。并且在 single-well 条件下，(K) 越大，水平线越高，左右交点越远，滞回区越宽。

### 4.6 状态边界与 cycle 时间必须分开

全局状态边界是：

\[
(a^*,b^*)=\arg\min_{a<b}\rho(a,b).
\]

第 \(i\) 个真实 Heating 段的启动时刻是：

\[
\boxed{
t^*_{\mathrm{start},i}
=\inf\{t:x_i^H(t)\ge b^*\}.
}
\]

第 \(i\) 个真实 Defrost 段相对其实际除霜起点的最佳终止延时是：

\[
\boxed{
\tau^*_{\mathrm{end},i}
=\inf\{s:x_i^D(s)\le a^*\},
\qquad
t^*_{\mathrm{end},i}
=t^{actual}_{D,start,i}+\tau^*_{\mathrm{end},i}.
}
\]

不同 cycle 的结霜/融霜速度不同，所以穿越时间可以相差很大，即使边界相同。

还要诚实区分：现有数据中的 D 段是在“实际较晚的启动点”之后发生的。上式给的是该真实 D trajectory 何时应停止；它和同一条记录中更早的 \(t^*_{\mathrm{start},i}\) 并不是一条真正执行过的 counterfactual optimal trajectory。未来在线 controller 同时采用 \((a^*,b^*)\) 时，才会执行完整最优 cycle。现有数据对它的支持来自共享状态与 cross-cycle Markov 假设，而不是直接观测。

### 4.7 多工况怎样保持简单

本方案首轮求一对**面向当前实验工况分布的全局边界**，以保留最终的 `RGB + mode` controller。对不同工况 \(\theta\)，先由 clean reference 归一化 stage cost，再令：

\[
\bar g(x)=\mathbb E_\theta[g(x;\theta)],
\qquad
\bar q(x)=\mathbb E_\theta[q(x;\theta)],
\qquad
\bar r(x)=\frac{\bar g(x)}{\bar q(x)}.
\]

把上面的 \(g,q,r,K\) 换成 cycle 等权平均后的 \(\bar g,\bar q,\bar r,\bar K\)，证明完全不变。这样得到的是对当前目标工况分布长期平均最优的一对边界，而不是声称每个温湿度下都有相同的逐工况 optimum。

如果按环境温湿度分层后，\(\bar r(x)\) 的 U 形位置明显漂移且一个全局边界产生很大 regret，那么“只用 RGB + mode”这一研究命题本身就不成立；此时不能让图像网络偷偷吸收工况差异。这个检验应先于建模。

---

## 5. cycle 怎么用

### 5.1 输入、输出与数据划分

**输入**只有四类：

1. H/D mode 与时间边界：`defrost_active`、`operating_mode`、catalog boundaries；
2. 成本：`power_total`，以及由水流量和进出水温计算的 (Q_h)；
3. clean reference 工况：环境温湿度、水温、流量、压缩机频率；
4. 固定 camera ROI 的 RGB。

**中间量**只有一个状态 \(x\)、一个 stage cost \(\ell\)、一条核心曲线 \(r(x)\)。

**输出**是全局 \((a^*,b^*)\)，以及每个可穿越 cycle 的 \(t^*_{\mathrm{start},i}\)、\(t^*_{\mathrm{end},i}\)。

划分必须按完整 cycle，并以 `experiment_date` 分组。当前只有 11 个实验日期、36 个 H/D RGB 都有效的 cycle，不应随机拆帧。首轮采用 5-fold grouped cross-fitting：每次用四折日期估计 \((a^*,b^*)\)，只给留出日期的 cycle 生成 crossing time 与 RGB 标签。最终部署前再用全部有效 cycle 重估一次边界。

### 5.2 先做一把 H/D 共用的霜状态尺子

对每个 camera role 固定 coil ROI、曝光处理和 clean reference。用经典图像处理得到一个 whiteness-weighted frosting coefficient：每个像素同时满足“亮度相对 clean reference 增加、饱和度下降”时贡献 0–1 的 frost score，ROI 内取平均并用 clean 与 heavily-frosted prototypes 归一化到 \([0,1]\)。

这一步不训练网络。它只回答“盘管目前有多少可见霜”，不参与定义“此刻是否经济”。单纯 coverage 容易在霜完全覆盖后过早饱和，因此需要保留像素白度/不透明度的连续权重，但最终仍只输出一个标量 \(x\)。

先看 raw \(x(t)\) 是否在 H 中大体非降、在 D 中大体非升；只有原始趋势成立，才用 isotonic smoothing 降噪。不能先强制单调，再把强制后的结果当成物理证据。

### 5.3 每条长 Heating trajectory 直接提供什么

从 \(t_0\) 到实际除霜开始，直接观测：

\[
\{x_i^H(t),\ P_{el,i}^H(t),\ Q_{h,i}^H(t),\ \ell_i^H(t)\}.
\]

它直接告诉我们：霜从低到高经过每个状态区间时，花了多少时间、付了多少经济成本。它**没有**直接告诉我们在任意更早状态切入 Defrost 后会发生什么。

### 5.4 每条 300 s Defrost trajectory 直接提供什么

从实际 Defrost 起点到 300 s，直接观测：

\[
\{x_i^D(s),\ P_{el,i}^D(s),\ Q_{h,i}^D(s),\ \ell_i^D(s)\}.
\]

它直接告诉我们：霜从高到低经过每个状态区间时，继续除霜花了多少时间、付了多少成本。它**没有**直接告诉我们在残霜状态切回 H 后的湿盘管恢复过程。

cross-cycle pooling 的作用正是：用其他 H trajectory 在相同 \(x\) 处的运行表现，补上“残霜下继续 Heating”的局部信息；用 D trajectory 在相同 \(x\) 处的表现，补上“从该霜状态进入 Defrost”的局部信息。这个补法只在 \(x\) 近似 Markov 时成立。

### 5.5 不拟合 dynamics，直接按状态分箱积分

理论中写 \(f_H,f_D\) 是为了证明；数据上不必先拟合两条微分方程。把 H/D 都穿越的共同状态范围分成小 bin \([x_j,x_{j+1}]\)。对 cycle \(i\) 直接累计：

\[
\Delta t^{H/D}_{ij}=\int_{x\in bin_j}dt,
\qquad
\Delta C^{H/D}_{ij}=\int_{x\in bin_j}\ell(t)dt.
\]

然后 cycle 等权平均：

\[
\hat q_j=
\frac{1}{n_j}\sum_i
\frac{\Delta t^H_{ij}+\Delta t^D_{ij}}{\Delta x_j},
\]

\[
\hat g_j=
\frac{1}{n_j}\sum_i
\frac{\Delta C^H_{ij}+\Delta C^D_{ij}}{\Delta x_j},
\qquad
\boxed{\hat r_j=\hat g_j/\hat q_j.}
\]

这与理论中的 \(g,q,r\) 完全对应，却避免对 noisy \(x(t)\) 求导。长 cycle 不会因为帧数更多而自动拥有更大权重；有效样本单位始终是 cycle。

切换短窗口单独积分得到每条 cycle 的 \(K_i\)，在按既定质量规则剔除无效 cycle 后，用 cycle 等权均值估计 \(\bar K\)。用整条 cycle bootstrap，而不是 frame bootstrap，得到 \(\hat r(x)\) 和边界的置信区间。

### 5.6 怎样得到两个边界和每条 cycle 的时间

1. 只在 H 与 D 都有充分穿越的 common-support 区间上估计 \(\hat r(x)\)；
2. 检验它是否具有稳定 single-well 结构，而不是用 U 形约束强行拟合；
3. 解
   \[
   \hat r(a)=\hat r(b)=\hat\rho,
   \qquad
   \bar K=\int_a^b\hat q(x)[\hat\rho-\hat r(x)]dx
   \]
   得到 \((a^*,b^*)\)；
4. 在每条留出 cycle 的 H 段找第一次 \(x\ge b^*\)，在 D 段找第一次 \(x\le a^*\)；
5. 若长 H 没到 \(b^*\)，记为 right-censored；若 300 s D 没到 \(a^*\)，记为 not-cleared/censored。不能外推一条未发生的复杂 future trajectory，再制造一个精确标签。

最后检验状态是否够用：在相同 \(x\)、mode 和工况下，检查 \(\ell\) 与 bin passage time 的 residual 是否还能被“cycle id、此前最大霜负荷、距切换时间”明显解释。若可以，一维状态不闭合；这比继续调神经网络更早地否定方案。

---

## 6. RGB 怎么学

在当前固定相机、样本 cycle 不多的条件下，最容易真正成功的 RGB 模型不是 CNN，而是：

\[
RGB_t\xrightarrow{\text{固定 ROI + 经典图像处理}}x_t
\xrightarrow{\text{mode-specific threshold}}P(\mathrm{switch}).
\]

### 输入

- 当前单帧 coil ROI；
- 当前 mode \(m_t\in\{H,D\}\)。

### 标签

用 grouped cross-fitting 得到的边界产生 out-of-fold 标签：

\[
y_t=
\begin{cases}
0, & H\text{ 且 }x_t<b^*,\\
1, & H\text{ 且 }x_t\ge b^*,\\
0, & D\text{ 且 }x_t>a^*,\\
1, & D\text{ 且 }x_t\le a^*.
\end{cases}
\]

把 bootstrap 边界置信区间对应的帧设为 uncertain，不参加训练；不要把估计误差伪装成硬标签。

### 输出与最小模型

两个单调 sigmoid 已经足够：

\[
P(\mathrm{switch}\mid x,H)=\sigma[\alpha_H(x-b^*)],
\quad \alpha_H>0,
\]

\[
P(\mathrm{switch}\mid x,D)=\sigma[\alpha_D(a^*-x)],
\quad \alpha_D>0.
\]

实际上直接比较阈值也可以；sigmoid 只提供概率和置信度。在线沿用现有 20 s debounce：连续 20 s 超过相应阈值才切换。

首轮不使用 Transformer、时序网络、RL，也不需要 ResNet。若经典 \(x\) 本身不能在 H/D 中保持物理单调和跨 cycle 一致，换一个更大的图像网络并不会补回缺失的 Markov state，反而会把问题藏起来。

RGB 部分真正的创新因此不是 architecture，而是：

\[
\boxed{
\text{真实 energy/service optimum}
\rightarrow
\text{自动生成视觉切换监督}
}
\]

评估也应按 cycle，而不是按帧 accuracy：报告 boundary crossing time error、过早/过晚切换比例，以及由预测边界造成的经济 regret。

---

## 7. 下一步马上做什么

现在不要训练网络。先用当前有效 cycle 画四张图：

1. **共享状态图**：所有 cycle 的 raw \(x(t)\)；H 从低到高对齐，D 从高到低对齐，同时标出单调违例率和 H/D common support。它判断“这把尺子能否同时量结霜和融霜”。
2. **物理组成图**：在相同 \(x\) bin 上画 \(\ell_H(x),\ell_D(x)\) 与 passage time density \(1/f_H,1/f_D\)，并在旁边画同一 \(x\) 下对 previous maximum frost、time-since-switch 的 residual。它判断一维状态是否近似 Markov。
3. **决定性曲线**：画 cycle 等权的 \(\hat r(x)=\hat g(x)/\hat q(x)\) 及整 cycle bootstrap 置信带，再用不同实验日期/温湿度分层叠加。只有稳定 U 形且左右趋势一致，双边界结构才开始成立。
4. **边界图**：在 \(\hat r(x)\) 上画 \(r=\hat\rho^*\) 水平线、面积 \(\bar K\)、交点 \(a^*,b^*\)，下方只画各 cycle 的 crossing-time 分布与 censored 数量。

第一轮的判据只有一句：

\[
\boxed{
\text{共享状态近似成立}
+\text{核心 }r(x)\text{ 稳定单谷}
+\text{面积方程有内部解}
\Rightarrow
\text{再生成 RGB 标签。}
}
\]

若其中任一项失败，先停止“双边界最优”的 claim；不要用一个更复杂的模型把失败遮住。
