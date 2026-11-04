# Dataset 数据格式

Dataset 保存循环测量、科学边界、人工质量标记和已发布模型输入。主模型通过 `DatasetLoader` 读取这些内容，不在训练时重新生成 COP、标签或 DINO 特征。

## 交付目录

```text
dataset/
  dataset_manifest.json         数据集身份及实验列表
  cycle_catalog.json            循环身份、阶段边界、质量和资产位置
  channel_registry.json         通道、单位及测量处理定义
  cycles/                       常规定时除霜循环的处理表
  cycles_RB/                    RB 控制循环的处理表
  cycles_original/              各循环原始观测表
  training_data/                动作、标签、参考曲线、完整候选及筛选证据
  image_DINO_feature/            按循环保存的冻结 DINO 特征
  priors/rb_reference.csv        冻结 RB 触发时刻
  除霜预测模型/<模型名称>/         Ridge 参数与验证结果
```

`training_data/` 和 `image_DINO_feature/` 都保持平铺。结霜图像交付只需 DINO 特征，无需原图、照片索引或图片软链接。特征中的 `file_name` 用作关联键，不要求存在对应照片。

`cycles/` 和 `cycles_RB/` 中 CSV 与 Parquet 表达相同的处理测量；研究端还可保留循环图。循环具体位置由 Catalog 的 `assets` 给出，读取时不要根据循环编号自行猜路径。RB 分类及继承规则见 [Dataset 使用说明](dataset/README.md)。

## 测量与阶段

原始数据先根据运行状态识别循环，划分恢复、稳定制热、准备与除霜阶段；随后构建 **1 s 时间网格，按通道聚合**。

- 连续传感器保留原始值，对内部缺口线性插值，保存 `_interpolated` 列；两端缺失不外推。
- `is_interpolated` 记录该行是否有任一连续传感器经过插值；它不是数据有效性的替代字段。
- 供热功率、蒸发压力 Pe 和盘管温度 T3 再按过去 60 s 时间加权平滑，保存 `_processed` 列。窗口在循环和阶段边界重置，不足 60 s 时按已有时长归一化。
- 离散控制量以前值保持生成 `_filled` 列。

离线插值使用后续观测，不能把整个处理链称为在线因果处理。具体算法见 [测量处理说明](heat_smoothing/README.md)。Ridge 直接读取盘管温度处理值，不重复平滑。

`cycles_original/` 保留标准通道、循环身份和阶段，以及带参数组前缀的设备原始列；重复时间戳观测不折叠。新增通道可能为已有循环补充空列，使各实验表结构一致。

循环停机缺口识别使用 `shutdown_gap_seconds`：足够长的传感器缺口且两侧运行模式满足停机条件时才划分片段。这个设置不限制连续测量的内部线性插值。

当前恢复边界使用共享 `recovery_control_trace` 的 `frequency-setpoint` 算法。未观察到可靠控制转变时保留空边界，不自动回退到温度规则。已保存的 Original、Processed 与阶段记录使用同一边界；历史模型保留其原定义。

## 身份、质量与训练资格

Manifest 记录 schema 版本、数据集身份和实验列表。研究端可另外配置 `images_root`，默认是 Dataset 内的 `images/`；模型读取不访问该位置。

Catalog 中 `pipeline_status` 表示构建事实，`status` 表示当前人工使用状态：`valid`、`invalid`、`partial` 或 `reference`。人工复核通过 `review-cycle` 修改。历史 Pareto 状态字段仅用于相应历史方法，不是当前视觉主模型的训练入口。

当前循环筛选以完整 `training_data/candidate_decisions.parquet` 和 Catalog 为输入，先调用 `screen_cv_cycles`，再检查已发布的 RGB/DINO 覆盖和插值质量。不能拿历史循环名单代替重新筛选。

`training_data/cohort.csv` 按 `(cycle_name, camera_role)` 标记成员及 `evaluation_eligible`。前机位文件为 `<cycle_name>.parquet`，其他机位为 `<cycle_name>__<camera_role>.parquet`。DINO 特征每循环一个 Parquet，包含机位、时间和文件关联字段。原始质量证据随 `training_data/` 一起保留，因此没有原图也可以重做已发布证据上的筛选。

COP 的近优区间是峰值所在的连续 99% 区间。冻结视觉模型沿用其已发布近优标签，以首末近优点生成 CDF 训练目标；两者用途不同，不重新解释或修改冻结标签。训练、验证与测试按实验隔离，由 `DatasetLoader` 分配。

## 使用与维护

四个主入口默认读取项目根目录的 `dataset/`。显式相对路径按当前工作目录解释。首次导入使用独立 Dataset：

```bash
uv run python -m dataset_tools.manage_dataset add data/0714 --dataset temp/test/import_demo/dataset
uv run python validate_dataset.py --dataset dataset
uv run python -m dataset_tools.manage_dataset aggregate-original --dataset dataset --n-jobs 6
uv run python -m dataset_tools.manage_dataset smooth --dataset dataset
uv run python -m dataset_tools.manage_dataset review-cycle frost_cycle_000001 --status valid --reason manual_review_confirmed
```

`validate_dataset.py` 只读检查循环测量、列结构、时间和身份；`--check-images` 才额外检查照片索引。`aggregate-original` 从 Original 重建正式 1 s 测量表；`smooth` 更新插值和平滑列。两者都会修改指定 Dataset，执行前打印目标。

导入、替换、删除和科学边界修改属于研究端维护，不是冻结模型演示的必要步骤。照片重新索引、照片质量复核、DINO 重提取和含照片出图需要原图；云端照片获取须显式请求。下载照片默认保留，只有显式清理参数才删除本次下载内容。完整维护选项以 `python -m dataset_tools.manage_dataset <子命令> --help` 为准。

数据构建直接写入指定 Dataset；操作失败时按当前文件状态修正后重新运行。冻结历史模型与结果不因数据维护自动更新。输入发布和 RB 先验维护见 [Dataset 使用说明](dataset/README.md)，四阶段命令及输出目录见 [项目入口](README.md)。
