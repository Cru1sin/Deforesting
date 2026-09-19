# Effective heating COP defrost control

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

原始传感器和图像保存在 `data/`、`dataset/`，不进入 Git。
数据维护命令见 [dataset/README.md](dataset/README.md)。

## 当前定义

事件从除霜准备开始，到下一循环重新进入有效供热状态；恢复边界由频率设定值阶段判据识别，
不使用固定 9 min。事件有效热量为零，成本为整个事件的**总电耗**。
制热阶段只积分达到恢复末端参考出水温度的有效热量。

`COP_eff(t) = Q_eff(t) / [E_heating(t) + Ridge_event_electricity(t)]`。

当前 Ridge 为 8 输入、106 有效事件、21 实验；按实验等权，标准化及 α 选择仅使用训练实验。
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
