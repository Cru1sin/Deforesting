# DefrostingSensor

在主目录、`main` 分支工作。主线只有：

```text
Dataset → 完整除霜事件 Ridge → 有效供热 COP 参考曲线
        → F_core / Chen B0 / G_core → 统一触发评价与循环图
```

## 环境与数据

```bash
uv sync --extra ml --extra dev
uv run python validate_dataset.py --dataset dataset
```

原始数据默认保存在 `/Volumes/Proust/DefrostingSensor/data/`，图片通过 Dataset 的
`images_root` 指向 `/Volumes/Proust/DefrostingSensor/dataset/images/`。下载和外接盘读写前检查 Proust 已挂载。
项目的 `data/` 和 `dataset/images/` 是外接盘目录链接，`add data/0831` 从外接盘读取；表格和元数据仍保存在本地 `dataset/`。
数据维护命令见 [dataset/README.md](dataset/README.md)。

## 当前定义

事件从除霜准备开始，到下一循环重新进入有效供热状态；恢复边界由频率设定值阶段判据识别，
不使用固定 9 min。事件有效热量为零，成本为整个事件的**总电耗**。
制热阶段只积分达到恢复末端参考出水温度的有效热量。

`COP_eff(t) = Q_eff(t) / [E_heating(t) + Ridge_event_electricity(t)]`。

历史 Ridge106 为 8 输入、106 有效事件、21 实验；按实验等权，标准化及 α 选择仅使用训练实验。
事件资格不依赖 RGB 或 pre-COP 状态。分类队列另行要求当前状态、pre-COP、front RGB 有效，
正式结果为 80 循环、18 实验。模型外推不排除候选、RB 或触发评分；图上外推部分用虚线。
非有限值、负事件电耗或无效实测账本仍不能作为物理有效 COP。

| 模型 | 输入 | 分类头 |
|---|---|---|
| F_core | 189 运行统计 + 因果时间、电量、有效热量 | 192→60→60→32→32→1，SiLU |
| Chen B0 | 冻结 DINOv2 当前正面图像 | 384→1000→64→2，ReLU |
| G_core | F_core 输入 + 当前 DINOv2 | 576→256→128→64→32→1，SiLU |

三个模型共享循环及循环内现有类别等权 Binary loss、AdamW、外层 LOEO 和训练折预处理。
正式协议保留：内部验证以固定 2/3@0.5 的 COP 选择轮数；最终评价为 30 s 网格首次 p≥0.5，
未触发时取最后有效动作并标记兜底。验证选轮与最终控制的差别是已有实验定义，不是可搜索接口。

## 运行

```bash
# 复用现有输入与106事件；已有折会跳过
uv run python train_defrost_models.py train --n-jobs 6
# 新训练必须写新目录，避免混用协议和断点
uv run python train_defrost_models.py train --output output/image_models/new_run --n-jobs 6
# 从已存折重新汇总
uv run python train_defrost_models.py summary
# 三模型全部循环，只画 seed 0
uv run python train_defrost_models.py figures --figure-seed 0 --n-jobs 6
# 重新提取和拟合完整事件；新名称保护当前成果
uv run python fit_defrost_event_models.py --run-name ridge_refit --preparation-heat zero --n-jobs 6 --figures
```

当前模型输入在正式运行的 `base/`，恢复边界在 `recovery_boundaries.csv`；重新训练不依赖旧审计结果。
DINOv2 缓存在 `output/image_models/_cache/dinov2_vits14_r256_c224_front_v1/cycles/`，无需重新提取。

## 当前成果

- `output/image_models/core_comparison_ridge106/`：三模型、三个 seed、162 个折 checkpoint、预测、参考曲线和汇总。
- `output/defrost_event_models/complete_defrost_events_zero/`：106 事件 Ridge、边界、验证和参数。
- `output/cycles/core_comparison_ridge106/`：seed 0 的 240 张循环 PNG。
- `output/test/complete_defrost_events_zero/ridge_calibration/`：事件电耗和有效 COP 校准图及源数据。

核心代码：`image_models/inputs.py` 构造参考，`core.py` 定义三模型并调度折，
`policy.py` 训练与评价；`plots/publication.py` 统一循环绘图，`plots/event_calibration.py` 绘制 Ridge 校准。
旧实验代码保存在 Git 历史，不再作为主目录入口。

```bash
uv run python -m pytest
```

## 当前状态表示选择（R3-M，开发阶段）

本轮选定蒸发压力与盘管温度为 60 s 因果时间加权均值，运行频率为动作前最新有效状态，
沿用既有新鲜度限制。其他状态、派生量、趋势、Raw 供热、温度门槛、积分与支持硬约束不变。
选定结果位于 `output/test/coil_representation_r3/defrost_event_models/R3_M/`；
R2 与 R3-C 保留作历史对照，不改旧方法默认值或覆盖已有模型、标签。
盘管表示搜索到此结束。最后一次出水温度更新诊断复用缓存，使用既有 30 s 动作时钟，
每次只暂缓单个时刻的一次来源更新；1% 是工程近优容忍尺度，不是置信区间。
这些多轮 LOEO 结果属于开发证据，不能代替独立实机确认。

最后的固定模型局部检查已通过本轮 1% 工程尺度：110 个可计算循环中，12,769 次当前出水温度
稳定时的历史更新，404 次落在 30 s 动作时钟；每次只暂缓一个点的更新，仅循环 46、116
选择后移 30 s，原模型下最大 COP 损失为 0.053%，局部诊断下反向损失最大 0.060%。
据此结束本轮状态表示修正，采用 R3-M + Raw 作为选定工作方法，不继续搜索温度表示。
诊断证据：`output/test/coil_representation_r3/decisions/outlet_hold_summary.csv` 与
`outlet_hold_points.csv`。这是单点贡献敏感性结论，不是永久删除更新或替代表示的全轨迹实验；
支持边界限制、多近优区域及独立实机确认要求仍保留，既有训练标签没有被覆盖。

截至 0916 的数据扩充已完成：308 个周期段，其中 `cycles` 176 个、`cycles_RB` 132 个。
首个 RB 为 `frost_cycle_000177`（0827 16:58:32 开始供热，17:46:57 除霜）；
后续按本轮 RB 实验安排分目录，原始表与图片仍共享。CV 队列及 DINOv2 图片提取排除 RB。
新 R3-M Ridge 位于 `output/defrost_event_models/r3m_20260922_through0916/`，
使用两类循环中的 231 个有效事件、40 个实验；历史模型和标签保留，未自动替换默认参数。
复现命令：

```bash
uv run python fit_defrost_event_models.py --run-name r3m_refit --preparation-heat zero --pressure-statistic time-weighted-mean --frequency-statistic current --coil-statistic time-weighted-mean --n-jobs 6
```

## R3-M 图表与 RGB 状态对照

当前每循环 effective COP 图来自 `output/defrost_decisions/r3m_rb_frozen_20260923/`，
使用新 Ridge、30 s 最优动作网格；RB 独立逐秒回放，未按 30 s 取整。
当前选定的RB方法冻结为 **RB-20260923**，参数、完整规则、相对最初版本的改动及剩余偏差见
[冻结说明](RB_RULES_FROZEN.md)。参考结果位于 `output/defrost_decisions/rb_sampling_time_audit/`。
冻结的是显式参数组合；CLI历史默认值、旧结果和已有COP图保持不变，不代表已完整复现实机固件。

霜层图与保留缺失状态的 CSV 位于 `output/RGB结霜状态/r3m_20260922/`。
同图旧/新 DINO 特征已核对；只补提取选点所需的本地图片，缺失或不可读图片不下载。

```bash
uv run python -m plots.rgb_frost_state --point optimal --cohort regular --decisions output/defrost_decisions/r3m_through0916_rb2 --feature-cache output/image_models/_cache/dinov2_frost_points_r3m/cycles --output output/RGB结霜状态/r3m_20260922/optimal_regular.png
```

`--point rb` 为规则回放点，`--point observed --cohort rb` 为 RB 循环实测除霜开始；
`--point cv --model B0` 使用旧 CV 选点（另支持 `F_core`、`G_core`，`--seed` 默认 0）。
`--cohort` 可选 `regular/rb/all`。选图取动作时刻及之前最近的正面帧，最大间隔 120 s，
缺失样本留在 CSV、不参与 PCA 和余弦统计；旧缓存可用于原图片不在盘上的状态分析，图中图片仍留空。


## 历史旧CV覆盖子集的霜层状态一致性（53循环）

结果：`output/RGB结霜状态/rb_frozen_20260923/`。最优点、RB和B0/F_core/G_core使用同一批53个普通长循环、seed=0。
筛选复用 `reliable_rgb_cohort` 与 `common_cycle_metrics`，再要求完整COP峰形、各方法选点的本地图像和完整DINO向量。
`common_cycles.csv` 固定交集；各方法CSV保留完整cycle_name、cycle_id、动作时间、图片时间及相似度。
`*_availability.csv` 保留73个初筛循环的图像/特征缺失原因。没有下载图片，也没有把RB实验混入CV队列。

`optimal.png`、`rb.png` 显示各自队列内的一致性；右图逐行标注cycle id并按相似度排序。
`rb_vs_optimal.png`、`B0.png`、`F_core.png`、`G_core.png` 使用 `optimal.csv` 固定的PCA和最优点中心。
队列内相似度与最优点参考相似度分别保存；更集中不等于COP决策更好。
旧CV模型仅重新评价霜层状态，没有在本轮重新训练。

后续模型沿用同一入口。`--model`接受 `cycle_metrics.csv` 中任意模型名；
将下面的`MODEL_RUN`与`MODEL_NAME`替换为新模型结果路径和名称即可：

```bash
uv run python -m plots.rgb_frost_state \
  --root MODEL_RUN --model MODEL_NAME --point cv --cohort regular --seed 0 \
  --cycle-list output/RGB结霜状态/rb_frozen_20260923/common_cycles.csv \
  --reference-states output/RGB结霜状态/rb_frozen_20260923/optimal.csv \
  --feature-cache output/image_models/_cache/dinov2_frost_points_r3m/cycles output/image_models/_cache/dinov2_vits14_r256_c224_front_v1/cycles output/image_models/_cache/dinov2_vits14_through0916_local/cycles \
  --output output/RGB结霜状态/MODEL_NAME.png
```

若新选点缺少向量，使用现有 `image_models.dinov2_features --image-list ... --local-only` 补提取，
然后将增量缓存加入 `--feature-cache`。缺图或缺特征会记录在CSV中，不能静默视为完整的53循环对照。


## 当前霜层状态一致性：从308全量筛选

当前结果在 `output/RGB结霜状态/rb_frozen_20260923_all308/`，不再以旧CV结果限制最优点和RB队列。
308个周期段中，96个通过当前有效状态、原始审计有效、完整COP峰形及RGB审计/索引筛选。
较新循环没有旧式front_rgb_valid字段时，使用当前前视图索引与rgb_frost_status=valid；明确无效的旧标记仍排除。
再检查动作点、120秒内的因果本地图像及完整DINO特征：最优点81个，RB点81个，分别含75个普通长循环和6个RB循环。
两种方法可用循环并不完全相同，因此配对比较为79个（73个普通长循环、6个RB循环）。

- `optimal.png` / `rb.png`：全量筛选后各方法全部81个可用状态。
- `optimal_regular.png` / `rb_regular.png`：各自75个普通长循环，符合原来的长循环分析要求。
- `optimal_paired.png` / `rb_vs_optimal.png`：同一批79个循环，RB使用配对最优点固定参考。
- `all_308_screening.csv`：全部308循环的质量字段、队列类别、入选情况、选点及图像/特征状态。
- `paired_cycles.csv`：全量配对交集；不再使用旧53循环列表作为总体名单。

最优点/RB图使用 `--cohort all` 从全量筛选，或用 `--cohort regular` 仅保留普通长循环。
只有 `--point cv` 才使用当前给定模型结果中的共同评价循环。后续模型对照可用本目录的
`paired_cycles.csv` 和 `optimal_regular.csv` 作为普通长循环的名单与固定参考，并报告模型实际覆盖数量。
增量特征缓存为 `output/image_models/_cache/dinov2_rb_frozen_all308/cycles`；本轮仅从本地图像补提取2帧，未下载。
之前53循环图属于旧模型覆盖子集，不能称为308全量数据的合格循环总数。
