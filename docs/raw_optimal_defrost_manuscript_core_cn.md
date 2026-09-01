# From thermal service cost to visual frost states for heat-pump defrost timing

> 论文核心初稿 v0.1。当前已实证完成“原始传感器数据 → 经验成本 → 最优除霜区间”；RGB 分类结果尚未产生，相关位置明确标为待完成，不能提前写成结论。

## One-sentence argument

In an air-source heat-pump system operated under a fixed-duration defrost policy, we show that unsmoothed water-side and electrical measurements can define auditable cycle-specific empirical defrost-cost minima and near-optimal intervals, providing a physically grounded—but not yet causally optimal—supervision target for subsequent frost-image classification.

## Abstract

Defrost control in air-source heat pumps requires a decision boundary that connects progressive frost accumulation to the energetic consequence of continued heating. Historical operation records, however, observe only the defrost action selected by the existing controller and contain substantial short-timescale variation in measured heating capacity. Here we develop an empirical renewal-cost framework that operates directly on approximately 1-s, unsmoothed water-side and electrical measurements. Water-side heating capacity was calculated from flow rate and inlet–outlet temperature difference, while a cycle-specific clean reference was defined from 60-s stable-heating anchors. Observed defrost and recovery periods from adjacent cycle files were combined to estimate a fixed-policy defrost ticket comprising electricity use and an equivalent thermal-service shortfall. Among 77 catalogued cycles, 50 provided valid defrost tickets and 47 retained complete candidate domains for timing optimization. Thirty-seven of these 47 cycles exhibited an interior cost minimum, whereas 10 remained minimized at the observed defrost boundary. The median empirical optimum preceded the observed action by 58.8 min, but the median envelope spanning the 5% near-optimal set was 95.0 min wide; three cycles contained disconnected near-optimal segments. Replacing the mean defrost ticket with its median left the optimum unchanged in 26 cycles and shifted it by no more than 5 min in 83.0% of valid cycles, while a small subset remained sensitive. These findings establish an auditable route from system-level operation to regret-valued image labels, but they do not identify a causal global optimum under unobserved counterfactual defrost actions. Image-based classification and prospective control validation remain necessary to complete the framework.

## Introduction

Air-source heat-pump defrost control must balance two competing losses: continuing to operate under frost reduces useful heating service, whereas initiating defrost consumes electricity and temporarily interrupts heat delivery. A controller that acts too late accumulates frost-related degradation; one that acts too early pays the defrost penalty too frequently. The scientifically relevant target is therefore not a visually convenient monotonic heating-capacity curve, but the time at which the expected cost of continued operation overtakes the cost of resetting the evaporator state.

Existing experimental records do not directly reveal this boundary. They contain the action taken by the installed controller, not the outcomes of all alternative defrost times. Moreover, total heating capacity responds to water-side inertia, refrigerant dynamics and control actions in addition to frost. Consequently, forcing measured heating capacity to decrease monotonically would convert a modelling assumption into artificial evidence. A more defensible first step is to preserve the raw measured signal and formulate an explicitly bounded empirical decision problem under the observed defrost policy.

We therefore separate the study into two linked tasks. First, system measurements define an empirical renewal cost for every feasible defrost start time within each complete heating cycle. This stage produces both a point minimum and a near-optimal interval, retaining information about whether the decision boundary is sharp or weakly identified. Second, these cost-derived states will supervise RGB models that infer whether the visible frost condition lies before, within or after the economically near-optimal region. The present analysis establishes the first task; the image-learning task is deliberately not claimed until cycle-level train, validation and test experiments are complete.

> 文献缺口：正式投稿前需补充并核验三组引用——空气源热泵结霜/除霜代价、renewal-reward 或维护时机优化、区间/不确定标签视觉学习。此稿不虚构引用。

## Results

### Raw measurements yielded an auditable water-side heating signal

The analysis used the original approximately 1-s cycle files without rolling, low-pass, wavelet or monotonic smoothing. For each timestamp, water-side heating capacity was calculated as

\[
Q_h(t)=1.161\,\dot V_w(t)\,[T_{out}(t)-T_{in}(t)],
\]

where volumetric flow is expressed in m³ h⁻¹ and heating capacity in kW. A 60-s median was used only to estimate each stable clean anchor; the cost integral itself retained the raw pointwise measurements. The median clean-anchor coefficient of performance across 74 valid anchors was 2.487, giving a thermal-service conversion coefficient \(\lambda_Q=1/COP_0=0.402\).

### Cross-cycle reconstruction recovered the observed defrost ticket

The end of each defrost event occurred at the beginning of the following cycle file. Treating files independently would therefore truncate every defrost/recovery event. We linked temporally adjacent cycles from the same experiment and defined recovery as the first raw interval in which heating capacity remained above 90% of the following clean anchor for 30 s. This reconstruction yielded 50 valid events. Their mean fixed-policy equivalent cost was 1.018 kWh-eq. and their mean duration was 13.60 min. The equivalent cost comprised 0.328 kWh mean electrical use and a thermal-service shortfall converted using \(\lambda_Q\); this shortfall is not a direct PMV, PPD or occupant-comfort measurement.

### Renewal cost identified internal minima in most complete cycles

For candidate defrost time \(\tau\), the heating-stage cost was

\[
C_H(\tau)=\int_{t_0}^{\tau}
\left(P_{el}(t)+\lambda_Q[Q_{ref}(t)-Q_h(t)]_+\right)dt,
\]

and the empirical renewal-average objective was

\[
\rho_i(\tau)=
\frac{C_{H,i}(\tau)+\bar K_D}
{T_{H,i}(\tau)+\bar T_D}.
\]

Candidate starts were evaluated at 1-min intervals from 10 min after stable heating to the observed defrost start, which was always included. A cycle was excluded if its valid raw-data domain did not reach the observed defrost boundary. Of 77 catalogued cycles, 47 met the complete-domain contract. Thirty-seven had an interior minimum and 10 were minimized at the right boundary; none was minimized at the 10-min left boundary. The median point estimate occurred 58.8 min before the observed defrost action.

### Broad cost valleys limit single-time supervision

Point minima alone overstated timing precision. Defining the near-optimal set as candidate times within 5% of the minimum cost produced a median 95.0-min envelope from the earliest to latest included candidate. Three cycles contained two disconnected low-regret segments, so this envelope must not be interpreted as one continuous interval. Figure 1b shows the representative pattern: the cost falls toward an interior minimum but remains nearly flat over a long period. This indicates that several candidate defrost times can be equivalent in the present objective even when the mathematical argmin is unique.

The empirical ticket choice was stable for most, but not all, cycles. Replacing the mean ticket cost and duration with their medians produced zero shift in 26 of 47 valid cycles, a shift no larger than 1 min in 63.8%, and a shift no larger than 5 min in 83.0%. The 90th percentile absolute shift was nevertheless 36.2 min. These sensitive cycles should not contribute a sharp binary image boundary without additional conditioning or uncertainty handling.

The clean-reference definition produced a similar mixed result. Replacing the two-anchor interpolation with a constant reference based only on the current clean anchor yielded a median absolute optimum shift of 1.0 min, and 76.6% of cycles remained within 5 min. However, 12.8% shifted by more than 30 min and the 90th percentile shift was 50.4 min. Thus, most point estimates were insensitive to future-anchor information, but a clearly identifiable subset was not.

The envelope width also depended strongly on the allowed relative regret. Median widths were 33.0, 52.0, 95.0 and 102.8 min at 1%, 2%, 5% and 10% thresholds, respectively. The 5% set is therefore retained as an exploratory reporting convention, not as a validated ±time label. The image stage must assign each frame from its interpolated candidate regret and compare thresholds on cycle-held-out validation data rather than assume that a universal 10-min interval exists.

## Methods

### Study object and cohort contract

The unit of analysis was one heating-to-defrost cycle. All 77 catalogued cycles remained in the audit table. A cycle received a point estimate only when its catalog status was valid, its stable-heating and observed-defrost boundaries were available, both clean anchors were valid, and at least 95% of the candidate-domain duration was covered by adjacent valid raw observations. Twelve cycles lacked an observed defrost boundary, nine were catalogued as invalid, five contained long gaps that truncated the candidate domain, and four lacked a valid clean anchor.

### Clean reference and thermal-service proxy

The clean reference joined the stable 60-s anchor of the current cycle to the stable anchor of the following cycle by linear interpolation. The pointwise thermal-service shortfall was

\[
s_Q(t)=[Q_{ref}(t)-Q_h(t)]_+.
\]

We converted this shortfall to an equivalent electrical quantity using the reciprocal of the cohort median clean COP. This conversion keeps electricity and service loss in a common kW-equivalent objective. It does not estimate indoor thermal comfort because the available dataset contains no direct indoor-state, occupancy or exposure measurement.

### Integration of irregular raw observations

Energy was integrated by the trapezoidal rule over adjacent valid raw samples. Invalid observation rows were removed before integration, allowing irregular intervals of at most 5 s. Longer gaps were never bridged. Coverage was defined as the sum of accepted adjacent intervals divided by the full candidate duration. This rule retains raw samples without smoothing while preventing a single missing timestamp from being counted twice as lost coverage.

### Empirical defrost and recovery ticket

For each valid observed event, the event window began at defrost start and ended at stable recovery. Electrical energy, thermal-service shortfall and duration were integrated over this window. The primary analysis used the arithmetic mean equivalent cost \(\bar K_D\) and mean duration \(\bar T_D\) across valid observed events. A median-ticket sensitivity analysis tested dependence on this fixed empirical choice. No environmental regression was fitted at this stage because the minimum feasible experiment was intended to establish the end-to-end cost chain before adding condition-dependent terms.

### Timing optimization and uncertainty set

The empirical optimum was the earliest candidate minimizing \(\rho_i(\tau)\). We recorded whether the minimum was internal or located at a search boundary. The 5% near-optimal set was

\[
\mathcal B_i=\{\tau:\rho_i(\tau)\le1.05\rho_i(t_i^*)\}.
\]

Both \(t_i^*\) and \(\mathcal B_i\) are exported. The interval is required for the subsequent visual-learning stage because it distinguishes a numerically unique argmin from a practically flat decision region.

We additionally exported the relative regret of every candidate and repeated the interval calculation at 1%, 2%, 5% and 10%. These values define label candidates; they do not select the final classification threshold without held-out image evidence.

## Discussion

The principal contribution of this stage is methodological: it turns historical heat-pump measurements into an auditable empirical timing target without forcing raw heating capacity to be monotonic. The internal minima observed in 37 complete cycles indicate that the measured deterioration signal can, under a fixed observed defrost ticket, outweigh the benefit of extending the heating interval before the installed controller acts. However, the broad near-optimal sets show that the data support an interval more strongly than an exact second or minute.

This distinction changes the image-learning problem. A binary label placed exactly at \(t_i^*\) would treat economically near-equivalent images on opposite sides of the argmin as contradictory classes. The next experiment should therefore compare two label contracts: a high-confidence binary task that excludes near-optimal images, and a three-state task consisting of pre-optimal, near-optimal and post-optimal states. The width must come from each cycle's cost curve rather than an assumed ±10-min window. Cycle identity, not individual frames, must define train, validation and test splits to prevent near-duplicate image leakage.

Three limitations bound the present result. First, historical observations do not reveal counterfactual recovery cost under earlier defrost actions, so the optimum is policy-conditional rather than causal. Second, the thermal term measures service shortfall, not direct occupant comfort. Third, a single cohort-level ticket ignores possible dependence on ambient temperature, humidity and operating state. These limitations motivate, respectively, prospective intervention experiments, indoor-state measurement and residual-guided ticket conditioning; none can be repaired by stronger smoothing alone.

The future clean anchor is an additional offline-information boundary. It is useful for reconstructing a post-defrost clean state, but it is not available to a real-time controller at the candidate time. The observed sensitivity subset should therefore be excluded from high-confidence image labels or re-analysed with a reference model that uses only contemporaneously available operating variables.

## Conclusion

Unsmoothed water-side and electrical measurements can support a complete, reproducible chain from observed heat-pump cycles to empirical defrost-cost minima and candidate-level regret. Among 47 cycles with complete candidate domains, 37 exhibited internal minima, while the 95.0-min median near-optimal envelope showed that timing uncertainty is substantial. Pointwise regret provides a more defensible supervision target for frost-image learning than a forced monotonic heating-capacity curve or an arbitrary time window. The next decisive evidence must demonstrate cycle-independent image classification and, ultimately, prospective energy and service benefits under controlled changes to defrost timing.

## Figure 1 draft legend

**Figure 1 | Raw-data empirical defrost timing under the observed fixed-duration policy.** **a**, Unsmoothed water-side heating capacity and the linearly interpolated clean reference for a representative complete cycle; orange and dashed black lines indicate the empirical optimum and observed defrost start, respectively. **b**, Renewal-average cost for candidate defrost starts in the same cycle; shading denotes candidates within 5% of the minimum. **c**, Observed versus empirical-optimum minutes from stable heating across 47 complete cycles. Blue points denote internal minima and grey points right-boundary minima; the dashed line is equality. **d**, Distribution of minutes by which the empirical optimum precedes the observed action. The box indicates the median and interquartile range, whiskers extend to 1.5 times the interquartile range, and points show individual cycles. Fifty observed defrost/recovery events contributed to the fixed empirical ticket.

## Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| 原始数据可形成可审计的经验成本链 | 代码、6 个核心测试、77 行 cohort audit、候选曲线源表 | supported |
| 多数完整循环存在内部经验最优点 | 47 个有效循环中 37 个内部 minimum | supported under fixed-policy ticket |
| 当前策略通常晚于经验最优点 | 提前量中位数 58.8 min | supported descriptively, not causally |
| 精确单点标签可靠 | near-optimal 宽度中位数 95.0 min | contradicted for many cycles |
| 区间标签有望服务 RGB 分类 | 成本区间已建立 | needs image experiments |
| 新策略可真实节能并改善舒适 | 尚无前瞻干预与室内舒适测量 | needs prospective evidence |

## 下一阶段最小实验

1. 仅下载已获得有效成本区间的 cycle 图像，按 cycle 流式处理，避免磁盘峰值。
2. 先比较“排除 near-optimal 的高置信二分类”和“pre/near/post 三分类”，不预设 ±10 min。
3. 以 cycle 为单位固定 train/validation/test，分别评估单机位、近远机位合并和全部机位。
4. 最小模型梯度：冻结视觉特征 + 线性分类器、传统纹理/颜色特征模型、小型 CNN、预训练 CNN 微调、预训练视觉 Transformer 微调。
5. 主指标除准确率外至少报告 balanced accuracy、macro-F1、AUROC/one-vs-rest AUROC 和 cycle-level bootstrap interval；若类别区间高度不平衡，先调整采样而不是增加模型复杂度。
