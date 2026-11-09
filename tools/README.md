# 结果维护工具

[返回项目验收](../README.md)

这里的命令不参与四阶段运行。迁移工具可先看 `--help`；索引工具没有参数解析，直接运行（即使附 `--help`）也会写入索引文件。

| 文件 | 用途 |
| --- | --- |
| [result_index.py](result_index.py) | 为已有结果生成 HTML 索引；不自动随训练执行 |
| [output_migration.py](output_migration.py) | 显式迁移或归档历史结果，会移动文件 |
| [package_release.py](package_release.py) | 整理源码、正面模型与霜层等级分析的复现包，默认写入新建 `dist/` |

日常输出规则在根目录 [output_layout.py](../output_layout.py)：正式视觉实验放 `output/90_复现材料/01_视觉模型训练/<名称>/`，临时运行放 `temp/test/<名称>/`，共享候选、质量和轨迹各归其类。

历史配置中的来源路径只供追溯；模型输入由 DatasetLoader 定位。不要为让旧路径可读而重新计算或改写冻结历史结果。

准备交付文件可运行 `uv run --no-sync python tools/package_release.py --n-jobs 6`。输出位置需要尚未存在，也可用 `--output` 指定新目录。包内保留完整数值目录、全候选筛选证据、正面 113 循环输入、两类 DINO 特征、论文模型权重、霜层轨迹、统计结果与文档插图。脚本只整理本地文件。

`github/` 与 `source.zip` 只含源码和说明，可用于 GitHub；`reproduction.zip` 含实验材料，单独交付，不上传 GitHub。解压复现包至源码根目录后，按技术报告运行。
