# 从原始实验到循环测量

[返回数据入口](../README.md)

从 [build_experiment.py](build_experiment.py) 的 `build_experiment` 开始：读取通道 → 识别循环 → 1 s 处理 → 校验 → 保留原始观测。由 `manage_dataset add` 调用，无需逐个运行文件。

| 步骤 | 文件／关键函数 | 核验重点 |
| --- | --- | --- |
| 读原始输入 | [raw.py](raw.py)、[sensors.py](sensors.py) | 设备表格与环境传感器的时间、单位 |
| 匹配通道 | [channel_mapping.py](channel_mapping.py) `load_channels`、[alignment.py](alignment.py) | 设备列映射；多源记录按既定一对一规则配对 |
| 识别循环 | [prepare_measurements.py](prepare_measurements.py) `prepare` → [detect_cycles.py](detect_cycles.py) `label_cycles` | 根据运行状态划分循环与阶段，先保留原始时间戳并集 |
| 构建测量表 | [build_cycle_tables.py](build_cycle_tables.py) `process` | 按通道聚合到 1 s，再调用共享插值和平滑 |
| 保存数据 | [schema.py](schema.py) | 列、单位、原始值和处理值的发布格式 |
| 核验边界 | [validate_prepared_measurements.py](validate_prepared_measurements.py) | 处理前后身份、阶段及测量约束 |

恢复边界由运行控制确定；不能从平滑后的曲线反推循环。平滑窗口在循环和阶段处重置，离线插值使用后续观测，方法见 [测量处理](../../heat_smoothing/README.md)。

`dataset_settings.py` 定义配置；`features.py` 计算派生量；`baseline.py` 保存基线差值；`match_camera_images.py` 关联照片。照片关联不参与冻结 DINO 模型验收。

输入／输出表的含义见 [数据格式](../../DATASET_FORMAT.md)，不要仅凭循环编号推测文件位置。
