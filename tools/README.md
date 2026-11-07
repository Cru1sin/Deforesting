# 结果维护工具

[返回项目验收](../README.md)

这里的命令不参与四阶段运行。迁移工具可先看 `--help`；索引工具没有参数解析，直接运行（即使附 `--help`）也会写入索引文件。

| 文件 | 用途 |
| --- | --- |
| [result_index.py](result_index.py) | 为已有结果生成 HTML 索引；不自动随训练执行 |
| [output_migration.py](output_migration.py) | 显式迁移或归档历史结果，会移动文件 |

日常输出规则在根目录 [output_layout.py](../output_layout.py)：正式视觉实验放 `output/90_复现材料/01_视觉模型训练/<名称>/`，临时运行放 `temp/test/<名称>/`，共享候选、质量和轨迹各归其类。

历史配置中的来源路径只供追溯；模型输入由 DatasetLoader 定位。不要为让旧路径可读而重新计算或改写冻结历史结果。
