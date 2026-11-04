# Dataset 文件与发布

[返回四阶段验收](../README.md) · [数据格式](../DATASET_FORMAT.md) · [数据处理代码](../dataset_tools/README.md)

已有交付数据直接运行根目录 `main_defrosting_model.py`。本页用于查文件，或在新增数据后准备模型输入。

## 文件放在哪里

| 位置 | 内容 |
| --- | --- |
| `dataset_manifest.json`、`cycle_catalog.json`、`channel_registry.json` | 实验清单、循环边界与资产路径、通道及单位 |
| `cycles/`、`cycles_RB/` | 处理后的循环测量；实际路径以 Catalog 为准 |
| `cycles_original/` | 原始观测，保留设备列和重复时间戳 |
| `image_DINO_feature/` | 每循环一个特征 Parquet，以 `camera_role` 区分机位 |
| `training_data/` | 动作、标签、完整候选、参考曲线与筛选证据 |
| `priors/rb_reference.csv` | RB 首次触发时刻；未触发保留空值和 `right_censored` |
| `除霜预测模型/<模型名称>/` | Ridge 参数、事件清单和验证结果 |

交付保留上述数值内容和独立的视觉权重。结霜图像只放 DINO 特征，不放 `images/`、照片索引或外接盘链接。特征中的文件名只是关联键；训练和冻结评价不打开照片。

`image_DINO_feature/` 与 `training_data/` 均平铺。不能仅复制演示循环：Loader 会读取目标机位的完整发布输入。

## 处理表怎么看

`timestamp` 为 1 s 时间网格，`cycle_stage` 记录阶段。连续通道保留原始列与 `_interpolated` 列，离散控制量另存 `_filled`。供热功率、Pe、T3 的 `_processed` 列是过去 60 s 平滑结果。

`is_interpolated` 表示该行任一连续传感器发生插值，整循环比例为 True 行数／总行数。内部缺口离线线性插值，端点缺失不外推；平滑在循环和阶段边界重置。边界来自运行状态，不能由平滑曲线反推。[处理算法](../heat_smoothing/README.md)。

Catalog 区分构建事实 `pipeline_status` 与人工状态 `status`。本次除霜条件码 10/20/23/26/29 归入 RB，50 归入定时除霜；无除霜事件的连续片段可继承前段 RB 分类，但不补造触发时间。分类依据见 `defrost_control`。

## 已发布模型输入

| `training_data/` 文件 | 用途 |
| --- | --- |
| `cohort.csv` | `(cycle_name, camera_role)` 成员及训练／评价资格 |
| `frost_cycle_*.parquet` | 逐动作输入与监督标签 |
| `reference.parquet` | 已发布参考曲线 |
| `candidate_decisions.parquet` | 全部循环、全部候选，供权威规则重新筛选 |
| `quality_metrics.csv`、`action_validity.parquet` | 全候选循环的质量证据及逐动作有效性，二者均必需 |
| `cycle_screening.csv`、`settings.json` | 筛选结果与来源定义；不替代原质量证据 |

正面保留上述文件名；其他机位的动作、质量、筛选和设置文件带 `__<camera_role>` 后缀，共享候选与参考曲线保持单份。`complete_reward` 是诊断支持，实际角色由实验划分决定。

模型只读这里的已发布内容，不打开配置中记录的历史来源路径。当前标签保留其原模型来源；重新拟合 Ridge 或计算 COP 不会自动更新标签。[筛选及模型定义](../image_models/README.md)。

## 新数据发布：研究端操作

顺序：完整候选 → 输入质量核验 → 权威筛选 → 发布。例子使用独立 `temp/test/new_data/dataset`，假定已导入数值数据、提取 DINO、准备 RB 先验，并放好冻结 Ridge 参数。质量核验还需照片索引和原图，不是特征交付包的验收步骤。

下面的实验应未参与 Ridge 拟合，因此使用 `full-model`；已有拟合实验的留出评价使用 `cross-fitted`。

```bash
# 全量候选；冻结 Ridge 位于该 Dataset 的默认模型目录
uv run --no-sync python optimal_defrost_time.py \
  --dataset temp/test/new_data/dataset --prediction-mode full-model \
  --output temp/test/new_data/cop

# 默认使用该 Dataset 的 image_DINO_feature
uv run --no-sync python -m image_models.cycle_selection \
  --dataset temp/test/new_data/dataset \
  --decisions temp/test/new_data/cop/candidate_decisions.csv \
  --audit-inputs --output temp/test/new_data/quality

# 当前正面配方；由完整候选重新筛选
uv run --no-sync python -m image_models.cycle_selection \
  --dataset temp/test/new_data/dataset \
  --decisions temp/test/new_data/cop/candidate_decisions.csv \
  --quality temp/test/new_data/quality/quality_metrics.csv \
  --include-rb --input-rule rgb70 --cohort-rule observed --near-set-rule exact \
  --output temp/test/new_data/screening

# 首次发布时，目标 training_data 尚不存在
uv run --no-sync python -m dataset_tools.training_data training-data \
  --dataset temp/test/new_data/dataset --admission-run temp/test/new_data/screening
```

已有发布不能直接覆盖：先保留旧 Dataset 快照，再发布新版本；`--append` 仅用于追加符合条件的新循环。绘图不改变标签规则。`connected` 是桥接首末近优点空隙的历史方法，当前配方保持 `exact`。

旧 Dataset 缺少质量证据时，可在原来源仍可访问的情况下运行 `python -m dataset_tools.training_data screening-inputs --camera-role front`；其他机位分别指定。它只补齐证据，不改标签定义。

## 维护边界

数据命令默认修改项目 `dataset/`，试用时始终指定独立 `--dataset`。

| 操作 | 命令入口／影响 |
| --- | --- |
| 导入与替换 | `manage_dataset add/replace`；写循环、元数据、照片及图，replace 可调整后续编号 |
| 更新测量 | `manage_dataset smooth/aggregate-original`；原地更新处理表及定义 |
| 人工复核 | `manage_dataset review-cycle`；记录状态与原因 |
| 删除实验 | `manage_dataset remove`；删除指定实验记录，仅用于明确删除 |
| 更新 RB 先验 | `calculate_rb_reference.py --publish-prior --output <本次目录>`；回放规则并替换先验 |
| 发布 COP 图 | `optimal_defrost_time.py --figures --publish-dataset`；更新图及 Catalog 质量字段 |
| 照片／云端 | `dataset_tools.cloud_images/cloud_upload/dts_downloader`；需要原图或网络，不属于冻结验收 |

完整参数看相应子命令 `--help`。冻结历史结果保持原定义；新增结果使用新目录，不能通过改名把历史标签当作新模型结果。
