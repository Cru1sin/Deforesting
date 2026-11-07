# RB-20260923：冻结规则

冻结日期：2026-09-23。冻结的是本轮选定的回放方法与参数组合，不代表完整还原实机固件。
后续方法变更应另命名、另存结果，不覆盖本版本。未完成独立实机验证。

## 1. 冻结对象与复现

共享实现：`defrost_decision/baselines/rule_based.py`。
参考结果：`output/90_复现材料/02_循环筛选/RB触发时间核验/`，含308个周期段。
该目录的 `run_settings.json` 为本版本参数依据；`rb_actual_comparison.csv` 为实机核对结果。

| 参数 | 冻结值 |
|---|---:|
| condition1_drop_c | 2 |
| sample_tolerance_seconds | 1 |
| confirmation_missing_seconds | 3 |

当前实现只提供本冻结规则，不再提供旧参数组合入口。复现命令：

```bash
uv run --no-sync python calculate_rb_reference.py --audit-actual --n-jobs 6 --run-name rb_20260923_replay
```

逐秒回放在供热起点至准备除霜前进行；T1/T2 从秒换为分钟，T3o 从十分之一摄氏度换为摄氏度。
环境温度恰好落在边界时归较暖档；Condition1、Case1–5 与 Case7 的先后及确认逻辑保持冻结。
同一时刻按 Condition1、Case1–5、Case7 的顺序记录首次满足条件。20秒窗口必须完整覆盖，
所有实测值满足条件，当前值也满足；允许缺测最多3秒。另允许连续19点满足条件，两条容差不叠加。
Condition1 使用2℃温降。未满足规则时保留 right_censored，不用实机时间反填规则触发时间。

完整实现及实机核对结果仍由本说明开头的路径提供。
