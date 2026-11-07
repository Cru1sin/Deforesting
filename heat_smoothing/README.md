# 插值与平滑

[返回数据入口](../dataset_tools/README.md)

共享实现位于 [methods.py](methods.py)，Dataset 构建和测量更新复用这一条路径。

| 函数 | 输入 → 输出 |
| --- | --- |
| `process_measurements` | 循环原始观测、目标时间网格 → 插值／前值保持／平滑列 |
| `smooth_heat` | 指定通道与阶段起点 → 过去窗口的时间加权均值 |
| `measurement_values` | 已发布处理表 → 供计算读取的通道值 |

连续通道在循环内部按实际时间线性插值，端点缺失不外推；离散量前值保持。供热功率、Pe、T3 再计算过去 60 s 均值，窗口在循环及阶段处重置，不足 60 s 时使用已有时长。

原始列保留，插值列为 `_interpolated`，平滑列为 `_processed`，共享 `is_interpolated` 标记该行是否发生插值。插值使用后续观测，因此整条处理链属于离线处理。

从 [示例循环](../dataset/cycles_RB/frost_cycle_000327.csv) 对照这些列；阶段边界核验见 [builder](../dataset_tools/builder/README.md)。
