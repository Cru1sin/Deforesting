# 数据导入、处理与读取

[返回项目验收](../README.md) · [数据文件与发布](../dataset/README.md)

本目录把原始实验整理为 Dataset，并提供统一读取接口。训练读取已发布输入；导入与维护会修改指定 Dataset。

## 从哪里开始

已有数据只检查：`uv run --no-sync python validate_dataset.py`。

验收原始导入时，将下面的目录换成实际实验目录；命令在项目根目录执行：

```bash
uv run --no-sync python -m dataset_tools.manage_dataset add data/0819 \
  --dataset temp/test/import_demo/dataset
```

输入为原始测量、设备控制记录及可用照片。输出在指定 Dataset：`cycles/`、`cycles_RB/` 保存处理表，`cycles_original/` 保存原观测，Catalog 保存阶段边界和质量。导入还会复制照片并生成循环图；同名实验跳过。

## 顺着数据流读

| 文件／函数 | 职责 |
| --- | --- |
| [manage_dataset.py](manage_dataset.py) `main` | 选择导入或维护操作，打印目标 |
| [dataset_maintenance.py](dataset_maintenance.py) `add_dataset` | 组织单次导入和 Dataset 更新 |
| [builder](builder/README.md) `build_experiment` | 读通道、识别循环、处理测量、保存原始观测 |
| [measurements.py](measurements.py) `refresh_processed_measurements` | 重建已有循环的插值和平滑列 |
| [load_dataset.py](load_dataset.py) `DatasetLoader` | 读取循环、已发布模型输入，按实验划分训练／验证／测试 |
| [training_data.py](training_data.py) `publish_training_data` | 从完整候选重新筛选并发布模型输入 |
| [cycle_metadata.py](cycle_metadata.py) | 循环身份、阶段、分类和资产路径 |

通道匹配、循环识别与 1 s 处理详见 [builder](builder/README.md)；共享平滑详见 [heat_smoothing](../heat_smoothing/README.md)。

## 有限维护

`manage_dataset <子命令> --help` 说明实际读写范围。`smooth` 更新处理列；`review-cycle` 记录人工复核；`replace/remove` 替换或删除指定实验。已有冻结结果不会自动更新。

`dts_downloader.py` 下载原始实验；`local_images.py`／`cloud_images.py` 管理照片；`cloud_upload.py` 上传。它们属于研究端维护，冻结模型验收不经过这些步骤。更多操作见 [Dataset 说明](../dataset/README.md)。
