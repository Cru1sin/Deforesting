# 论文图表

[返回文档](../README.md)

论文引用 15 张 PNG 和 2 张矢量 PDF；代码使用手册仅引用循环 000043 的 3 张 PNG。图像随本地文档交付，不上传 GitHub；RB 表写在 `paper.tex` 中。

| 文件或图表 | 用途与来源 |
| --- | --- |
| `experiment_setup.png` | 所附实验平台图 |
| `cop_workflow.png` | 用户提供的最终数据处理与 COP 构建图；完整 60 s 窗口及边界归一化在图旁正文说明 |
| `cop_example.png` | 所附典型 COP 图，区别于循环 000043 |
| `model_flow.png` | 用户指定的模型总体流程截图；训练目标与触发规则见正文公式 |
| `ridge_validation.png` | 由共享 `render_calibration_figures` 重绘，统计文字置于坐标轴外并去除白底；数值来自 `output/01_核心结果/除霜电耗预测_287事件/` |
| `model_comparison_history.png` | 所附全部循环历史比较，位于正文多模型性能比较；不将瀑布增量解释为组件贡献 |
| `rb_early_comparison_history.png` | 所附早触发子集历史图，仅保留历史文件，不用于当前论文和技术报告 |
| `operation_results.pdf` | 共享绘图模块 `plots/external_policy.py` 的 `operations` 视图，从三种子逐循环结果重绘 a–d 四子图；各子图绘图区宽高比约 1.2∶1；坐标轴及文字均为黑色，数据点保留统一配色 |
| `model_evidence.png` | 正文完整 a–d 四子图的模型结构与组件对照；第 12 张附件的更新版；由共享 `plots/model_comparison.py::render_stopping_evidence` 读取现存表重绘 |
| `camera_comparison.png` | 所附六机位图；对应 `output/01_核心结果/六机位比较/comparison_summary.csv` 的 `first_common`、`second_common`；数据集 96 准入、极端视角 95 可评价 |
| `cop_curves.png` | 所附 101 循环候选曲线，颜色用于描述运行频率关联 |
| LaTeX：RB 规则表（附录 C） | 对应第 15 张附件；与用户截图逐项核对阈值、边界及持续要求；表中列出完整规则，正文说明本研究未启用 Case 8 |
| `rb_headroom.png` | 所附 RB 改善空间图；来源 `output/03_数据与物理依据/RB提升空间与触发时机/` |
| `lstm_film.png` | 用户提供的最终 LSTM 与 FiLM 结构图；运行特征在前、视觉特征在后，各时刻下标一致 |
| `frost_workflow.png` | 用户本轮第一张附图，原样保留；仅说明图像、近优霜态范围与阈值标签的宏观关系，不将图内旧示例作为当前 16 级结果 |
| `frost_method.png` | 用户本轮第二张附图，原样保留；DINOv2、PCA、Harmony 与 Slingshot MST 霜层分级方法 |
| `frost_coverage.png` | 用户本轮第三张附图，原样保留；16 级离线序列估计的分工况近优覆盖率与代表照片 |
| `frost_cop.png` | 用户本轮第四张附图，原样保留；对应 `output/03_数据与物理依据/离线连续霜态与COP_16级/` 的各等级循环等权平均 COP，未混用直接投影版本的均值 |
| `frost_data_overview.pdf` | 复用 `output/03_数据与物理依据/历史队列构成_101循环/data_overview.pdf`；位于正文实验设置，展示循环筛选及 23/48/30 的工况组成；霜层附录引用此图 |
| `frost_cycle_000043.png` | 技术报告演示附图：运行记录与阶段边界 |
| `frost_cycle_000043_effective_cop.png` | 技术报告演示附图：RB 与离线参考最优 |
| `frost_cycle_000043_rgb_panel.png` | 技术报告演示附图：多机位照片与覆盖记录 |

更新的四面板图使用 `output/02_模型与消融/模型组件与学习曲线/` 下 `case_selection.csv`、`panel_a_predictions.csv`、`cycle_results.csv`、`panel_b_models.csv`、`panel_b_width_screen.csv`、`panel_c_components.csv`、`panel_d_learning.csv`。清零记忆采用现存三种子结果 83.08±1.03%，不沿用附件中未对应到同协议来源的 81.16%。重绘在 `temp/test/report_figures_20261109/` 运行，未覆盖冻结结果。

论文区分当前默认 60 轮、Dropout 0.2 的训练配方与历史内部验证选轮/宽度筛选配方。误差范围按各图来源解释为种子极差、种子标准差或验证损失四分位范围，不统一称作置信区间。

运行量综合图使用 `output/01_核心结果/制热与电耗_101循环/cycle_results.csv`，通过 `render_stopping_results(..., decision_view="operations")` 在同一画布绘制 2×2 子图。复绘结果及四张定量源表位于 `temp/test/paper_operation_results/`；原实验数据保持不变。论文全部采用纵向页面。
