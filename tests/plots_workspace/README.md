# 图表测试

[返回测试目录](../README.md) · [对应实现](../../plots/README.md)

`test_defrost_decision.py` 检查决策图、机位对比及源表；`test_event_calibration.py` 检查事件校准；`test_rgb_frost_state.py` 检查霜态图的数据关系。

```bash
uv run --no-sync python -m pytest tests/plots_workspace -q
```

测试在隔离目录生成必要产物，核对计算和绘图接口；通过测试不等于已人工查看图形。
