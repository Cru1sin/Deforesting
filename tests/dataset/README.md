# Dataset 测试

[返回测试目录](../README.md) · [对应实现](../../dataset_tools/README.md)

- `test_alignment.py`、`test_prepare_contract.py`：多源时间配对与原始观测保留。
- `test_recovery_control.py`、`test_process_contract.py`、`test_processed_channels.py`：阶段边界、1 s 处理、插值与平滑。
- `test_io_contract.py`、`test_dataset_schema.py`、`test_training_bridge.py`：文件读取、格式、模型输入发布。
- 下载和上传测试使用隔离输入；`audit_control_fields.py` 是研究核验脚本。

最小发布检查：`uv run --no-sync python -m pytest tests/dataset/test_training_bridge.py -q`。缺少特定原始批次时，相应数据测试可能跳过；跳过不表示已核验。
