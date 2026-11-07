# COP 与电耗核验

[返回测试目录](../README.md) · [对应实现](../../defrost_decision/README.md)

先看 `test_effective_cop.py`：能量、有效支持、动作网格与峰值区间。`test_rule_based.py` 检查冻结 RB；`test_public_decision_cli.py` 检查 COP 入口。

```bash
uv run --no-sync python -m pytest tests/cost_workspace/test_effective_cop.py -q
```

`test_v268_*` 沿用历史文件名，检查共享 Ridge 拟合和科学约束，并非独立版本入口。其余脚本研究 COP 斜率、停止决策、阈值及温度电耗；对应根级 `test_*.py` 检查其行为。研究目标各自选点，不把多个目标合成新成本函数。
