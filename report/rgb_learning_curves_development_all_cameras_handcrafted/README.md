# 九机位组学习曲线（开发性结果）

本结果使用现有手工 40D 图像特征，按独立实验留一验证；训练规模为 2、4、6、8、10 个实验，每个规模使用 3 个嵌套重复。九个组包括六个原始机位、`top_pair`、`left_pair` 和全部机位 pooled。pair/all 是 pooled training，不是同步 multi-view fusion。

满数据下每个机位组的最佳模型：

| 机位组 | 最佳模型 | 10 实验 BA | 距满数据 BA 0.02 内所需实验数 |
|---|---|---:|---:|
| front | RBF-SVM | 0.965 | 6 |
| all | RBF-SVM | 0.951 | 2 |
| left | RBF-SVM | 0.949 | 4 |
| top_pair | MLP | 0.950 | 4 |
| left_pair | Logistic | 0.943 | 6 |
| left_close | Random forest | 0.938 | 4 |
| top | MLP | 0.912 | 4 |
| top_close | Random forest | 0.910 | 8 |
| extreme | Random forest | 0.907 | 10 |

这里的“所需实验数”仅表示当前学习曲线上，最小训练规模的平均 BA 已进入该模型满数据 BA 的 0.02 范围；它不是统计功效分析，也不能外推为正式样本量要求。全量云端 cycle、DINOv2 与 EfficientNet 表征完成后必须重新计算。
