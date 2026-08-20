# Cycle Optimum and Model Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用未平滑的约 1 s 原始数据扩展并审计逐循环经验最优除霜点，完成 9 个机位组 × 5 个模型、cycle11 OOD、视觉集中性检验与中文 LaTeX/PDF 初稿。

**Architecture:** 保留现有 `defrost_cost.py`、逐循环成本曲线和 40D RGB 特征作为唯一事实源。主分析只使用完整观察边界的 Tier A；缺实际除霜边界但有足够传感器记录的 Tier B 仅作右截尾敏感性分析；cycle11 完全隔离为扰动 OOD。模型比较统一使用 experiment-held-out 协议，pooled camera groups 明确不称为 multi-view fusion。

**Tech Stack:** Python 3.11、pandas、NumPy、scikit-learn、matplotlib、pytest、XeLaTeX/ctexart。

---

### Task 1: 固化 cohort 与候选域语义

**Files:**
- Modify: `src/frost_analysis/defrost_cost.py`
- Modify: `scripts/analyze_raw_optimal_defrost.py`
- Test: `tests/test_defrost_cost.py`
- Test: `tests/test_plot_raw_optimal_defrost.py`

- [ ] **Step 1: 写失败测试**：覆盖 observed boundary、sensor-record censored end、候选域不足和 minimum location 四种状态。
- [ ] **Step 2: 运行** `.venv/bin/pytest tests/test_defrost_cost.py tests/test_plot_raw_optimal_defrost.py -q`，确认新断言先失败。
- [ ] **Step 3: 最小实现**：沿用 `_candidate_costs`，仅增加显式 `cohort_tier`、`candidate_end_source`、`is_censored`；不插补未观察到的 defrost ticket。
- [ ] **Step 4: 重跑测试并生成结果**：运行 `.venv/bin/python scripts/analyze_raw_optimal_defrost.py`；输出 interior、left boundary、right boundary/censored、invalid 的完整计数。
- [ ] **Step 5: 提交并推送**：`analysis: audit extended optimal-point cohort`。

### Task 2: 逐循环 publication atlas

**Files:**
- Modify: `scripts/analyze_raw_optimal_defrost.py`
- Test: `tests/test_plot_raw_optimal_defrost.py`
- Create: `report/raw_optimal_defrost/figures/cycle_atlas.pdf`

- [ ] **Step 1: 写失败测试**：每个可估计循环必须同时有 raw-Q、成本曲线、最优竖线、近最优背景带和边界/截尾标签。
- [ ] **Step 2: 最小绘图改动**：复用 `_plot_cycle`，用连续背景时间带代替仅成本轴布尔填色；每循环单图并合成 atlas PDF。
- [ ] **Step 3: 导出 PNG/PDF/SVG/TIFF 与 source CSV**，在 100% 缩放下检查标签、颜色和截尾声明。
- [ ] **Step 4: 提交并推送**：`figures: show per-cycle empirical optimum windows`。

### Task 3: 统一 9×5 模型协议

**Files:**
- Modify: `src/frost_analysis/rgb_evaluation.py`
- Modify: `scripts/evaluate_rgb_feature_shards.py`
- Test: `tests/test_rgb_evaluation.py`
- Create: `report/rgb_model_comparison/`

- [ ] **Step 1: 写失败测试**：模型注册表固定为 logistic regression、random forest、RBF-SVM、histogram gradient boosting、MLP；所有模型返回 prediction、score 和 held-out experiment。
- [ ] **Step 2: 最小实现**：复用现有 40D feature columns 和 LOEO 循环，不新增 GitHub 依赖；随机种子固定为 0。
- [ ] **Step 3: 运行 9 个 camera groups × 5 models × 锁定 1% regret threshold**；输出 balanced accuracy、macro-F1、AUROC、balanced misclassification regret 和 experiment bootstrap CI。
- [ ] **Step 4: 明确 `top_pair`、`left_pair`、`all` 是 pooled training，不是同步 multi-view fusion；检查每折无 experiment/frame leakage。
- [ ] **Step 5: 提交并推送**：`analysis: compare nine camera groups across five models`。

### Task 4: cycle11 OOD 压力测试

**Files:**
- Modify: `scripts/evaluate_rgb_feature_shards.py`
- Test: `tests/test_rgb_evaluation.py`
- Create: `report/rgb_model_comparison/cycle11_ood_metrics.csv`

- [ ] **Step 1: 从 catalog 锁定** `frost_cycle_000011`，记录 invalid 原因“中间密闭环境被破坏”。
- [ ] **Step 2: 验证 cycle11 不在任何训练/选模折中**；若现有本地特征不存在，报告 unavailable，不下载新图像来掩盖缺失。
- [ ] **Step 3: 若特征可用，仅用主分析已锁定模型预测并报告 OOD 指标与覆盖率，不把结果并入主均值。
- [ ] **Step 4: 提交并推送**：`analysis: isolate cycle11 as ood stress test`。

### Task 5: 检验“时间分散、视觉集中”

**Files:**
- Create: `scripts/analyze_visual_state_concentration.py`
- Create: `tests/test_visual_state_concentration.py`
- Create: `report/visual_state_concentration/`

- [ ] **Step 1: 写失败测试**：按 cycle 取 1% low-regret 邻域的冻结 40D feature centroid，计算标准化跨 cycle dispersion；固定时间对照使用同 cycle 的候选域中位进度。
- [ ] **Step 2: 最小实现**：时间离散度报告 `IQR(t*)/median(t*)`；视觉集中度比较 optimum-neighbourhood 与 fixed-time centroid distance，并按 experiment bootstrap 差值 CI。
- [ ] **Step 3: 防循环论证**：集中性检验不使用分类标签训练或最优分类器准确率，只使用预先提取的冻结特征距离。
- [ ] **Step 4: 若 CI 不支持视觉更集中，如实触发下一阶段 RGB+当前传感器；否则不开发序列模型。
- [ ] **Step 5: 提交并推送**：`analysis: test temporal dispersion against visual concentration`。

### Task 6: Nature-style 结果图

**Files:**
- Create: `scripts/plot_expanded_paper_figures.py`
- Create: `tests/test_plot_expanded_paper_figures.py`
- Create: `report/paper_expansion/figures/`

- [ ] **Step 1: Figure 5 合同**：结论为“统一协议下模型和机位改变泛化与控制 regret”；panel a 比较同机位不同模型，panel b 比较同模型不同机位，panel c 显示 OOD/不可用边界。
- [ ] **Step 2: Figure 6 合同**：结论为“最优时间是否比对应视觉状态更分散”；报告配对点、bootstrap CI 和样本量。
- [ ] **Step 3: Python 独占导出** SVG/PDF/600-dpi TIFF/PNG 和逐 panel source CSV；检查字体、颜色、裁切、图例与 pooled-training 注释。
- [ ] **Step 4: 提交并推送**：`figures: compare model and visual-state evidence`。

### Task 7: 中文 LaTeX 初稿与 PDF

**Files:**
- Create: `paper_zh/main.tex`
- Create: `paper_zh/references.bib`
- Create: `paper_zh/figures/`
- Create: `paper_zh/main.pdf`

- [ ] **Step 1: 建立 `ctexart` 单一事实源**，正文按摘要、引言、结果、讨论、方法、数据与代码可用性组织，不声称使用 Nature 官方模板。
- [ ] **Step 2: 从 source CSV 自动填入全部数字**；每个定量主张映射到脚本输出，负结果和 right-censored 结果保留。
- [ ] **Step 3: 嵌入 Figure 1–6**；正文明确“离线经验最优”“thermal-service proxy”“pooled training”“exploratory camera finding”和非因果边界。
- [ ] **Step 4: 用 `latexmk -xelatex -interaction=nonstopmode main.tex` 编译；检查 undefined reference、overfull box 和 PDF 页面渲染。
- [ ] **Step 5: 提交并推送**：`paper: build Chinese LaTeX proof-of-concept`。

### Task 8: 对抗评审、验证与收口

**Files:**
- Modify: `paper_workflow/06_review/editorial_decision.md`
- Modify: `paper_workflow/06_review/review_matrix.csv`
- Create: `paper_zh/verification_report.md`

- [ ] **Step 1: 按 Nature reviewer 门控审查新颖性、候选域截尾、ticket 外推、视觉集中性、模型公平性和 cycle11 OOD。
- [ ] **Step 2: 修复 critical/major 问题；不通过增加未授权复杂模型来掩盖负结果。
- [ ] **Step 3: 运行** `.venv/bin/pytest -q`、`.venv/bin/ruff check .`、`git diff --check` 和 LaTeX 编译检查，记录真实结果。
- [ ] **Step 4: 最终提交并推送**：`docs: review expanded manuscript evidence`。

### Task 9: 全量云端循环与模型数据量曲线

**Files:**
- Modify: `scripts/extract_rgb_feature_shards.py`
- Create: `scripts/evaluate_rgb_learning_curves.py`
- Create: `tests/test_rgb_learning_curves.py`
- Create: `report/rgb_full_features/`
- Create: `report/rgb_learning_curves/`

- [ ] **Step 1: 逐cycle流式物化全部cost-valid云端图像**：提取全部pre/near/post冻结特征；每个临时cycle处理后仅删除本地副本，云端对象不修改、不删除。
- [ ] **Step 2: 用全量特征重跑9机位组×5模型LOEO**，保持1% pointwise-regret协议和experiment隔离。
- [ ] **Step 3: 数据量以训练experiment/cycle数量定义**，在每个held-out experiment内使用嵌套训练子集；禁止通过随机抽帧制造虚假的样本量曲线。
- [ ] **Step 4: front与all pooled先比较五模型的学习曲线、方差和饱和点**；只有机位排序改变时才扩展到其余7组。
- [ ] **Step 5: 主模型锁定后单独流式提取cycle11并做扰动OOD**，不进入训练、选模或学习曲线。
- [ ] **Step 6: 用全量结果替换demo数字、重画Figure 5--6并重新编译中文PDF；提交并推送**：`analysis: validate models on full cloud cohort`。

## 自审结果

- 需求覆盖：候选域扩展、逐循环背景带、9×5、cycle11 OOD、pooled/fusion 区分、主命题检验、全云端循环、按experiment/cycle定义的数据量曲线、升级门控、中文 LaTeX/PDF 均有对应任务。
- 刻意不做：当前不下载图像、不引入 timm/Swin/YOLO、不实现同步 multi-view、不强制所有循环产生内部最小值。
- 科学边界：Tier B 只作右截尾敏感性；cycle11 只作 OOD；所有“最优”均是固定经验 ticket 下的离线点估计。
