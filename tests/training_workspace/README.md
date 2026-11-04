# 视觉模型与评价测试

[返回测试目录](../README.md) · [对应实现](../../image_models/README.md)

| 文件 | 主要风险 |
| --- | --- |
| `test_cycle_selection.py` | 完整候选筛选与循环资格 |
| `test_policy.py`、`test_temporal_stopping.py` | 标签、概率、触发和评分 |
| `test_main_model.py`、`test_nested_stopping.py` | 配置等价、实验隔离、训练侧选参 |
| `test_checkpoints.py`、`test_output_layout.py` | 权重读取、机位核对和输出范围 |
| `test_slingshot.py`、`test_frost_progression.py` | 补充分析的轨迹和时间匹配 |

入口修改后可运行：`uv run --no-sync python -m pytest tests/training_workspace/test_main_model.py tests/training_workspace/test_checkpoints.py -q`。

`reference_ranking.py` 是研究核验程序；这些小型测试不替代完整后期实验评价。
