# 9 个机位组 × 5 个模型：统一 RGB 特征比较

## 锁定协议

- 输入：同一套冻结的 40 维单帧 RGB 颜色/梯度特征。
- 标签：逐图 pointwise relative regret，排除不超过 1% 的 near-optimal 模糊区。
- 划分：leave-one-experiment-out；同一实验及其循环不会同时进入训练和测试。
- 指标：balanced accuracy、macro-F1、AUROC、class-balanced misclassification regret；区间按独立实验 bootstrap。
- `top_pair`、`left_pair`、`all` 是不同机位图片的 pooled training，不是同一时刻的 multi-view fusion。

## 主要结果

- 最佳组合是 `front + rbf_svm`：balanced accuracy 0.9653，95% CI 0.9420–0.9867。
- `front` 上 logistic、MLP、random forest、histogram gradient boosting 分别为 0.9616、0.9613、0.9565、0.9533；模型复杂度没有形成决定性优势。
- `all + rbf_svm` 为 0.9507（0.9302–0.9708），没有超过最佳单机位 `front`。
- 这组实验只支持“冻结低维视觉特征具有跨实验判别信息”；不能据此声称同步多视角融合有效，也不能把 retrospective 标签结果表述为已实现在线节能。

## OOD 状态

`frost_cycle_000011` 已从 catalog 锁定为环境密闭条件中途被破坏的 OOD 循环。它有 2,055 条图像元数据，但当前没有本地 feature shard 或本地图像，因此未进入任何训练、选模或主均值。本阶段不临时下载并把它包装成预注册结果；后续只能在主模型锁定后流式下载、预测并删除本地临时副本。
