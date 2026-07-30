# Pipeline 数据合同

这份文档只记录会改变科研含义的稳定约定。实现入口按 `prepare → process → analyze` 阅读。

## 阶段

| 阶段 | 输入 | 输出 | 禁止内容 |
| --- | --- | --- | --- |
| Prepare | `data/<MMDD>` 原始目录 | Prepared、初始 `cycle_summary`、`prepare_summary.json` | 插值、重采样、baseline、动态特征 |
| Process | Prepared、初始 `cycle_summary` | Processed、最终 `cycle_summary` | 跨实验/循环/阶段填补、baseline fallback |
| Analyze | Processed、最终 `cycle_summary` | `candidate_channel_evidence.csv` | 固定权重综合排名、因果混杂结论 |

正式时间点表的唯一键为 `experiment_id + timestamp`；循环摘要的唯一键为 `experiment_id + cycle_id`；候选证据的唯一键为 `experiment_id + channel`。

## 循环字段

`cycle_status` 只有 `valid`、`incomplete`、`invalid`。`cycle_stage` 只有 `recovery`、`frost_development`、`defrost`、`partial`。

| 字段 | 定义 | 有效阶段 | 单位 |
| --- | --- | --- | --- |
| `cycle_elapsed_seconds` | `timestamp - stable_heating_start` | `frost_development` | s |
| `cycle_progress` | `(timestamp - stable_heating_start) / (defrost_start - stable_heating_start)`，限制到 `[0, 1]` | `frost_development` | 0–1 |

Prepare 根据原始时间戳计算一次，Process 重采样后再次计算；Processed 值是最终权威值。除霜状态有效观测之间超过 5 秒时不保持状态；影响边界或阶段识别时记录 `cycle_status_reason=defrost_state_gap`。

## 质量后缀

| 后缀 | 含义 |
| --- | --- |
| `__missing` | 原始记录缺失 |
| `__invalid` | 原始值无法解析或超出范围 |
| `__duplicate` | 同一源通道同一时间有多条记录 |
| `__conflict` | 重复记录的有效值不一致 |
| `__imputed` | 当前值或派生依赖包含缺失重建值 |

Prepared 保留源质量事实。Process 屏蔽 duplicate/conflict 后，不机械转发源标记，只新增具有当前处理语义的 `__imputed`。

## Baseline

baseline residual 统一为：

```text
channel__baseline_residual = current - baseline
```

负值表示当前值低于无霜基准。没有合格窗口时只能使用：

```text
no_candidate_window
missing_required_anchor
insufficient_observed_coverage
too_much_imputation
unstable_anchor
```

失败不允许 fallback，也不改变 `cycle_status`。

## Candidate evidence

固定字段为：

```text
experiment_id, experiment_date, channel,
trend_cycle_count, reset_pair_count, future_cycle_count, context_cycle_count,
trend_effect, direction_consistency, reset_effect,
future_performance_effect, max_abs_context_spearman,
decision, reason
```

### 样本量和趋势

`trend_cycle_count` 是产生有限趋势相关系数的循环数；`reset_pair_count` 是相邻循环对数；`future_cycle_count` 是产生有限未来性能效应的循环数；`context_cycle_count` 是产生工况关联的循环数。

每个有效循环计算候选通道 residual 与 `cycle_progress` 的 Spearman 相关，`trend_effect` 为这些循环相关系数的中位数。`direction_consistency` 是循环趋势方向与总体趋势方向一致的比例。

### Reset

只在同一实验中按 `heating_start` 排序后配对相邻循环 `i` 和 `i+1`，不跳过失败中间循环。循环 `i` 的除霜前值是 `defrost_start` 前 5 分钟、`frost_development` residual 的中位数；循环 `i+1` 的除霜后值是已接受 baseline 的 `recovery` 窗口 residual 中位数。两者均可用才形成 pair。

原始效应为：

```text
post_defrost_residual - pre_defrost_residual
```

按 `expected_frost_direction` 转换符号，使正值表示按预期方向复位。

### Future

`future_performance_effect` 使用配置中的正式 target 和 horizon。10 秒网格上，当前点只匹配同一 `experiment_id × cycle_id × frost_development` 内精确的 `timestamp + 10 min` 目标值。不存在精确点时不参与；不允许最近邻、跨循环或跨阶段匹配。

### Context 和 decision

`max_abs_context_spearman` 是候选 residual 与 context 通道的最大绝对 Spearman 关联，只是关联提示，不是因果混杂证明。

阈值来自配置：

```text
trend_cycle_count < minimum_valid_cycles → insufficient_coverage
max_abs_context_spearman >= maximum_context_association → high_context_association
abs(trend_effect) >= minimum_absolute_trend_effect
且 direction_consistency >= minimum_direction_consistency
→ trend_supported_candidate
其他 → partial_evidence
```

`reset_effect` 和 `future_performance_effect` 当前只作为证据展示，不参与 decision。
