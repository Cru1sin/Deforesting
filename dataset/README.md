# frost_cycle_dataset

### 下载数传数据

```bash
caffeinate -ims uv run python -m dataset_tools.manage_dataset dts-download '数传链接'

caffeinate -ims uv run python -m dataset_tools.manage_dataset dts-download \
  '数传链接' --start-no 13 --end-no 30
```

密码按提示输入；本地图像仅在显式指定时删除。



### 实验日期数据切分循环进入数据库

```bash
uv run python -m dataset_tools.manage_dataset add data/0819
uv run python -m dataset_tools.manage_dataset replace data/0819
uv run python -m dataset_tools.manage_dataset aggregate-original
```



### 检查与维护

```bash
uv run python validate_dataset.py
uv run python -m dataset_tools.manage_dataset refresh all
uv run python -m dataset_tools.manage_dataset render
uv run python -m dataset_tools.manage_dataset render --panel --fetch-cloud-images
uv run python -m dataset_tools.manage_dataset render --publication --n-jobs 6
uv run python -m dataset_tools.manage_dataset review-cycle frost_cycle_000001 --status valid
uv run python -m dataset_tools.manage_dataset edit --recovery-seconds 540
uv run python -m dataset_tools.manage_dataset remove 0819
```

当前决策采用冷媒侧有效供热循环 COP，准备、正式除霜及恢复阶段的有效热量全部为 0；
这些阶段的总电耗均保留。每个循环仅发布 `cycles/frost_cycle_000xxx_effective_cop.png`，
Catalog 的资产键为 `effective_cop`。此前两种口径的图和 Catalog 已归档至 worktree 的
`output/test/effective_cop_zero/previous_dataset_decisions/`。

图中绿色虚线／圆点表示 RB 触发，暖橙色点划线／菱形表示 COP 最优触发；
绝对 COP 为蓝色，归一化 COP 为黑色，并保留阈值、近优区间和阶段图例。
内部缺测仅在线性插值后的显示曲线中补齐，不改变积分、支持状态和最优时刻。

没有实际除霜的开放制热循环可在观测结束前评估假想除霜，保留原人工有效性状态。
`partial` 的未闭合循环可以参与候选评估；缺失恢复边界、人工无效和模型不支持仍有明确状态。
实际已发生除霜但准备边界缺失的记录仍按边界质量问题保留，不伪造实际阶段。

在 `pareto-boundary-learning` worktree 中从保存结果重绘并发布：

```bash
uv run python select_defrost_time.py --dataset ../../dataset \
  --model-file output/defrost_event_models/heating_start_zero/candidate_model_parameters.json \
  --run-name effective_cop_zero --preparation-heat zero \
  --figures-only --fetch-cloud-images --publish-dataset --n-jobs 6
```

时间／循环 ID 总对比图和逐循环源数据为 worktree 下的
`output/test/effective_cop_zero/timing_vs_RB.png` 和 `timing_vs_RB.csv`。
图上的提升为有受支持 RB／最优 COP 的共同循环中，逐循环相对提升的算术平均；
属于模型理想化估计，并非实际试验收益。

这些命令不重新拟合 Ridge 或训练 RGB。main 中旧的 Pareto `render --decision` 流程
对应历史方法，不用于当前有效 COP 图的重绘。
物理总览仍由 `--publication` 单独更新；RGB 拼图仍由 `--panel` 更新。
front 照片复用本地缓存；`--fetch-cloud-images` 按需提取缺少的 RB／最优时刻照片，
沿用当前网络代理，保留下载缓存。匹配偏差最大为 ±120 秒，网络失败与真实缺图分别处理。




### 上传

```bash
uv run python -m dataset_tools.manage_dataset upload-data all
uv run python -m dataset_tools.manage_dataset upload-data 0806 0807
uv run python -m dataset_tools.manage_dataset upload-images all
uv run python -m dataset_tools.manage_dataset upload-images all --cleanup-uploaded-images
```
