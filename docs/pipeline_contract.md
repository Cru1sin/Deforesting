# Pipeline 数据合同

本文档记录会改变科研含义的稳定约定。阅读顺序是 `config → prepare → process → analyze`。

## 1. 阶段边界和唯一键

| 阶段 | 输入 | 输出 | 主要职责 |
| --- | --- | --- | --- |
| Prepare | `data/<MMDD>` 原始目录 | Prepared、初始 `cycle_summary`、`prepare_summary.json` | 解析原始文本、单位换算、循环切分、图片匹配 |
| Process | Prepared、初始 summary | Processed、最终 `cycle_summary` | 10 秒重采样、bounded 缺失、派生量、共同 baseline、动态特征 |
| Analyze | Processed、最终 summary | `candidate_channel_evidence.csv` | 循环级趋势、未来关联和工况关联证据 |

原始目录只读。Prepared 和 Processed 的唯一键是 `experiment_id + timestamp`；循环摘要的唯一键是 `experiment_id + cycle_id`；候选证据的唯一键是 `experiment_id + channel`。

## 2. 通道和重采样

每个通道在 `configs/channels.yaml` 中显式声明 `unit`、`kind`、`role`、`resample`、`missing` 和 `analysis_candidate`。允许的 `kind` 为：

```text
continuous, step, event, categorical, protected, derived
```

`analysis_candidate=true` 只允许用于数值语义和单位已确认、具有 `expected_frost_direction` 的 `continuous`、`step` 或 `derived` 通道。事件、分类和 protected 通道不进入候选分析。

非派生通道先解析原始值，再执行：

```text
canonical_value = raw_value * scale + offset
```

最后才按 canonical 单位执行 `valid_range` 检查。派生公式不是 YAML 表达式，只能使用 Python 白名单：`cop`、`pressure_ratio`、`water_delta_temperature`、`superheat_calculated`。除数不为正或依赖缺失时结果为 NaN。

Process 的重采样规则如下：

| kind | 规则 |
| --- | --- |
| continuous | 配置指定的 mean 等连续聚合 |
| step | last observed |
| event | last observed |
| categorical | last observed |
| protected | 配置明确指定，否则 last observed |
| derived | 不直接重采样，填补后重新计算 |

重采样产生的空桶不是 imputation；只有后续缺失策略真正重建的值才标记 `__imputed`。

每个 `experiment_id × cycle_id` 只建立一次公共 10 秒时间网格，不按
`cycle_stage` 分别建网格。每个输出桶代表左闭右开区间 `[timestamp,
timestamp + 10 秒)`。使用 `cycle_summary` 的 `heating_start`、
`stable_heating_start`、`defrost_start` 和 `defrost_end` 计算精确阶段区间。
阶段或循环边界严格位于桶内部时，整个 transition bucket 被排除；边界恰好位于
桶起点或终点时保留。桶内只聚合所属阶段的原始观测，不混合其他阶段，也不通过
overlap winner 或 fallback 推断阶段。被排除的桶不会被后续缺失处理补回。最终仍
保证 `experiment_id + timestamp` 唯一；`cycle_summary.excluded_transition_bucket_count`
按循环内唯一 bucket timestamp 计数。Continuous 通道只在 Prepared 中
`channel__missing` 不全为 True 时进入 coverage 分母；coverage 分子是桶内
canonical 非空行数，低于配置阈值的值先置为 NaN，再执行 bounded fill。
`low_coverage_channel_bucket_count` 和 `eligible_continuous_channel_bucket_count`
均在插值前记录；未进入 Process 的 incomplete cycle 使用 NaN 诊断值。

## 3. 质量标记、循环和图片

质量后缀固定为：

| 后缀 | 含义 |
| --- | --- |
| `__missing` | 没有任何非空原始记录 |
| `__invalid` | 原始值无法解析或超出范围 |
| `__duplicate` | 同一源通道同一时间有多条记录 |
| `__conflict` | 重复记录的有效值不一致 |
| `__imputed` | 当前值或派生依赖包含缺失重建值 |

重复记录不选择、不平均；受影响 canonical 值置为 NaN，同一时间其他正常通道保留。Process 先屏蔽 duplicate/conflict，再重采样，不机械复制 Prepared 源质量标记。

`cycle_status` 只有 `valid`、`incomplete`、`invalid`；`cycle_stage` 只有 `recovery`、`frost_development`、`defrost`、`partial`。长除霜状态缺口不推断；影响边界或阶段识别时使用 `cycle_status=incomplete` 和 `cycle_status_reason=defrost_state_gap`。持续时间超出日期配置时使用 `invalid`，并分别记录 `preceding_defrost_duration_seconds` 和 `terminal_defrost_duration_seconds`；两侧除霜异常分别使用 `preceding_defrost_duration_out_of_range` 和 `terminal_defrost_duration_out_of_range`。Operating mode 只检查 `[heating_start, defrost_start)` 内的 observed、非 duplicate/conflict 值，不检查 terminal defrost。partial 连续区间分别编号，且不进入 Process；没有完整相邻除霜事件时不创建 phantom cycle。

循环坐标只有在 `frost_development` 有值：

```text
cycle_elapsed_seconds = timestamp - stable_heating_start
cycle_progress = (timestamp - stable_heating_start)
                  / (defrost_start - stable_heating_start)
```

`cycle_progress` 限制到 `[0, 1]`，其他阶段为 NaN。Prepare 按原始时间戳计算一次；Process 重采样后按同一循环边界重新计算，Processed 值为最终权威值。缺失压缩机频率表示未知，不解释为停机。

相机映射只来自配置指定的 `camera_mapping_path`。相机目录名必须精确匹配映射 key；未映射目录直接失败，映射存在但当天缺失的角色只记录。每个角色独立匹配且一张图片最多使用一次，输出：

```text
image_<role>_path
image_<role>_time
image_<role>_offset_seconds
```

图片不 forward fill；同一 10 秒桶内每个角色最多保留距离桶时间最近的一张。

## 4. Bounded 缺失和共同 baseline

重采样完成后，所有填补限制在 `experiment_id × cycle_id × cycle_stage` 内。连续量只有在完整 NaN run 两侧都有 observed 值、两侧实际时间间隔不超过 `continuous_max_gap_seconds` 时才整段线性插值；超长 run 整段保持 NaN。step 只在前值 observed 且整个缺失段距前值不超过 `control_max_gap_seconds` 时限时前值保持。event、categorical 和 protected 默认不填补。

派生通道的 `__imputed` 是全部依赖 `__imputed` 的布尔 OR。过去窗口特征只为候选通道生成：

```text
channel__lag_Nmin = N 分钟前值
channel__delta_Nmin = current - lag
channel__rolling_mean_Nmin = shift(1) 后的完整窗口均值
```

rolling 不包含当前值，窗口不足保持 NaN，不跨循环、阶段或实验。

baseline 是每个 valid cycle 在 early `frost_development` 中寻找的“循环局部早期稳定参考代理”，正式名称为：

```text
cycle_local_early_stable_proxy
```

它不是人工或图像确认的绝对无霜状态。所有 required anchors 必须在同一个候选窗口内同时满足：observed coverage 达标、没有任何 imputed 值、标准差不超过各自阈值。选择时间最早的合格共同窗口；没有共同窗口时整个循环 baseline `unavailable`。incomplete 和 invalid 循环为 `not_applicable`。

通道 baseline 只使用共同窗口内 non-imputed 的有限值并取中位数。Process 输出：

```text
channel__baseline = cycle-local reference median
channel__baseline_residual = current - baseline
```

负 residual 表示当前值低于该循环参考。通道数据不足只使该通道 baseline/residual 为 NaN，不改变共同 baseline 状态。失败原因只能是 `no_candidate_window`、`missing_required_anchor`、`insufficient_observed_coverage`、`too_much_imputation` 或 `unstable_anchor`，不允许 fallback。

## 5. Candidate evidence

`candidate_channel_evidence.csv` 每个实验和候选通道一行，固定字段为：

```text
experiment_id, experiment_date, channel,
trend_cycle_count, reset_pair_count, future_cycle_count, context_cycle_count,
trend_effect, direction_consistency, reset_effect,
reset_evidence_status, reset_evidence_reason,
future_performance_association, median_max_abs_context_spearman,
decision, reason
```

四个样本量字段分别表示产生有限对应证据的循环数，或（reset）循环对数：

* `trend_cycle_count`：产生有限趋势 Spearman 的 valid、baseline available 循环数；
* `future_cycle_count`：产生有限未来关联的循环数；
* `context_cycle_count`：至少有一个 context 关联的循环数；
* `reset_pair_count`：本轮固定为 0。

### Trend

每个 valid 且 baseline available 循环只使用 `frost_development`，并要求至少 `minimum_points_per_cycle` 个有限点。先计算候选 residual 与 `cycle_progress` 的 Spearman，再按预期方向对齐：

```text
expected_frost_direction = increase → aligned = raw
expected_frost_direction = decrease → aligned = -raw
```

`trend_effect` 是 aligned effect 的循环中位数；`direction_consistency` 是 aligned effect 大于零的循环比例。因此正 trend effect 表示符合配置的结霜方向。

### Future association

`future_performance_association` 是候选 residual 与配置 target 在未来 horizon 的 Spearman 关联中位数，不称为预测能力。当前点只在同一 `experiment_id × cycle_id × frost_development` 内匹配精确 `timestamp + horizon`；不使用最近邻，不跨循环或阶段。

### Context association

每循环分别计算候选 residual 与各 context 通道的绝对 Spearman，取该循环最大值，再对循环最大值取中位数，输出 `median_max_abs_context_spearman`。它只是工况关联提示，不是因果混杂证明。Trend、future 和 context 证据均排除对应基础通道 `__imputed=true` 的点；residual 的质量列由去除 `__baseline_residual` 后的基础通道名确定，缺少必需质量列属于输入合同错误。

### Reset 和 decision

Reset evidence 本轮正式禁用，固定输出：

```text
reset_pair_count = 0
reset_effect = NaN
reset_evidence_status = not_evaluated
reset_evidence_reason = independent_reference_unavailable
```

不使用下一循环自身 baseline 自动归零，也不把 reset 放入 decision。Decision 规则按以下顺序执行：

```text
trend_cycle_count < minimum_valid_cycles
→ insufficient_coverage

trend_effect >= minimum_trend_effect
且 direction_consistency >= minimum_direction_consistency
→ trend_supported_candidate

其他
→ partial_evidence
```

当前不使用绝对趋势、不使用 reset 或 future evidence，不生成 rank、weighted score 或综合 candidate score。
