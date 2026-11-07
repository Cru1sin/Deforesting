# 测试与研究核验

[返回项目验收](../README.md)

验收人员先运行根 README 的四阶段命令；这里供修改代码后验证。所有命令在项目根目录执行。

```bash
uv run --no-sync python -m pytest tests/test_public_entrypoints.py -q
```

| 目录 | 检查内容 |
| --- | --- |
| [dataset](dataset/README.md) | 通道、循环、边界、插值、Dataset 发布 |
| [cost_workspace](cost_workspace/README.md) | 电耗、COP、RB 与独立选点 |
| [training_workspace](training_workspace/README.md) | 筛选、标签、模型、分折与冻结评价 |
| [plots_workspace](plots_workspace/README.md) | renderer 与源表对应关系 |

`test_*.py` 是自动测试；其他脚本是研究核验程序，可能写出分析结果，先看其参数。根级测试覆盖入口及跨模块研究检查。

全部测试还覆盖霜态支线，需要其可选依赖：

```bash
uv sync --extra ml --extra dev --extra slingshot
uv run --no-sync python -m pytest -q
```

默认排除 `slow`。完整 Dataset 测试需显式 `-m slow` 及对应本地数据；通过小型测试不等于全量模型性能验证。
