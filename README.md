# Image-guided heat-pump defrost timing

This repository is a Dataset-ready research workflow for selecting heat-pump defrost times
and learning the resulting timing labels from RGB images. The processed experimental Dataset
is not included; prepare it according to [DATASET_FORMAT.md](DATASET_FORMAT.md) first.

The code has two deliberately separate routes.

## Route A — Refrigerant effective-heating COP

Dataset → offline recovery boundary → event Ridge → supported candidate COP maximum.
The closed cycle runs from confirmed normal heating to the next confirmed normal heating.
`cycle_cop = (normal refrigerant heat + predicted preparation heat) /
(normal electricity + predicted complete-event electricity)`.
Formal defrost and pressure rebuilding contribute zero effective heat and retain all electricity.
`--preparation-heat include|zero` selects the explicit preparation convention; both must be
compared before replacing Dataset decision assets. This is different from measured net water-side
COP. Historical results and released parameters remain frozen; the current selector uses no H,
Pareto knee, evaporator-capacity objective, or shared multi-objective support domain.

## Route B — Stable image-classification baseline

```text
frozen V1 inverse-COP reference
→ image timing labels
→ color-gradient image features + RBF SVM
→ leave-one-experiment-out evaluation
```

V1 is retained only as the frozen label reference. It is not mixed with Route A.

## 1. Install

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

Install end-to-end ResNet50 or optional W&B support only when needed:

```bash
uv sync --extra ml
uv sync --extra tracking
```

## 2. Validate the Dataset

Place the processed Dataset at `dataset/`, then run:

```bash
uv run python validate_dataset.py --dataset dataset
```

Raw-to-Dataset construction and infrequent maintenance remain available as advanced commands:

```bash
uv run python -m dataset_tools.manage_dataset --help
```

Raw files and `dataset/` are data, not source code, and are never moved by these workflows.

## 3A. Audit recovery and refit event Ridge

From the existing `pareto-boundary-learning` worktree, the shared Dataset is `../../dataset`.
The audit writes boundary tables, numeric signal/state trajectories and experiment coverage to
`output/test/recovery/<run>/`; add `--figures` only when display figures are needed.
All new stages are applied in memory by `DatasetLoader.configure_recovery`; raw data and manual
validity labels are retained. Audit setpoint and actual-frequency states through the same detector:

```bash
uv run python fit_defrost_event_models.py --dataset ../../dataset \
  --audit-boundaries --output-root output/test --run-name recovery_final --n-jobs 6
```

Stage detection was corrected after an early-boundary bug was found in cycles
120, 133–135 and 138. The `offline_effective_include` and `offline_effective_zero` Ridge
runs used those earlier boundaries; their errors cannot establish current model accuracy.
They remain historical diagnostics. Current refits are `heating_start_include` and
`heating_start_zero` under `output/defrost_event_models/`, using the corrected boundaries.

Initial heating is separated from post-defrost recovery. When an adjacent preceding
record has a defrost end within 60 seconds of the recorded heating start, the complete
post-defrost interval is retained, including temporary zero-frequency readings. Otherwise,
an observed initial 0→positive setpoint transition defines the cold-start heating origin;
pre-command standby is excluded from the configured Dataset view and overview time axis.
The shared `heating_episode_bounds` function supplies this origin to the loader and audit.
Recording prefixes already at positive frequency remain explicitly unobserved startups;
no earlier start is invented. Raw files and historical Catalog fields are preserved.
Cold starts are labelled in overview figures. The stable-heating timestamp is unchanged.

The selected offline rule is `--recovery-rule frequency-setpoint`; `frequency-actual` is a
numeric cross-check. Pc − Pe in MPa and the raw controller-state codes remain supporting
observations. Pressure stabilization alone was rejected as the control-mode boundary.

The detector compresses repeated frequency observations into states, preserving each state's
first timestamp. States need three observed samples to exclude isolated transients. After the
observed startup band (≤42 Hz), the median rate of all preceding command intervals supplies
the ramp reference, requiring at least `--control-intervals` (default 3) intervals. A candidate
must have both a slow current interval and a slow median over the next three command
intervals, at ≤25% of the preceding reference. This rejects an isolated small frequency step
inside continuing acceleration, including a faster initial ramp followed by a slower ramp.
The boundary remains the **starting command timestamp**, without confirmation padding.
Rate comparisons account for measured sampling uncertainty. Evidence is bounded locally
when an unchanged state lasts four preceding median command intervals; remote later gaps
do not invalidate an already observed hold. Required evidence crossing a gap is rejected.
A truncated final hold must still cover the observed fast-ramp command cadence.

There is no 15 s or 120 s confirmation offset, trailing slope window, temperature knee, or
piecewise regression in the final rule. It uses subsequent observations offline and does not
claim online causality. Parameters are `--startup-frequency-max`, `--minimum-state-observations`,
`--frequency-ramp`, `--slow-fraction`, `--control-intervals`, and `--gap-seconds`. These define an operational boundary
from measured control changes; undecoded controller codes are not claimed as ground truth.

The earlier online experiments and pressure-only alternatives remain identifiable under
`output/test/recovery/`. Current offline outputs are under
`output/test/recovery/offline_heating_start_verified/`. The earlier Ridge results are provisional and superseded by this stage correction.
Ridge now fits event total electricity and, for `include`, preparation refrigerant heat,
with independent target cohorts and nested leave-one-experiment-out validation. `zero`
fits electricity only. `validation_summary.csv` reports MSE (kWh²), RMSE/MAE (kWh),
experiment-macro MSE and support coverage. Relative RMSE divides by mean absolute
observed energy. The include run also writes `observed_cycle_cop_comparison.csv`: both
heat definitions at the same actual defrost time, using measured complete-cycle energy.
With `--figures`, the shared renderers save calibration and fixed-time heat sensitivity
under `output/test/<run>/`. Candidate comparisons remain separate from this measured effect.

Update only the cycle overview PNGs with these boundaries and the Pc − Pe panel:

```bash
uv run python -m dataset_tools.manage_dataset render --dataset ../../dataset --publication \
  --recovery-boundaries output/test/recovery/offline_heating_start_verified/boundaries.csv --n-jobs 6
```

The boundary table changes plotting stages in memory; the Catalog, sensor files and RGB
panels are preserved. Unidentified recovery boundaries are marked in the figure title.
Current preprocessing does not generate baseline values or residuals, and validation does
not require baseline fields. Cycle and decision renderers no longer shade baseline windows.
Historical baseline functions, explicit legacy edits and stored fields are retained; recovery
configuration does not move those historical windows.


```bash
uv run python fit_defrost_event_models.py --dataset ../../dataset \
  --preparation-heat include --run-name effective_include --n-jobs 6
uv run python fit_defrost_event_models.py --dataset ../../dataset \
  --preparation-heat zero --run-name effective_zero --n-jobs 6
```

Each run under `output/defrost_event_models/` contains remeasured events, boundaries,
`candidate_model_parameters.json`, `model_validation.csv`, and `validation_summary.csv`.
Energy training uses every valid energy event; preparation heat uses its own valid cohort.
Zero mode fits no heat model. Standardization, equal experiment weights, nested Ridge
regularization, held-out-experiment prediction and support checks use the existing interfaces.
The old fixed nine-minute recovery and four-target complete-case model are not used here.

## 4A. Compare decisions and publication figures

```bash
uv run python select_defrost_time.py --dataset ../../dataset \
  --model-file output/defrost_event_models/effective_include/candidate_model_parameters.json \
  --preparation-heat include --run-name effective_include --n-jobs 6 --figures
uv run python select_defrost_time.py --dataset ../../dataset \
  --model-file output/defrost_event_models/effective_zero/candidate_model_parameters.json \
  --preparation-heat zero --run-name effective_zero --n-jobs 6 --figures \
  --compare-run output/defrost_decisions/effective_include
```

Candidate tables, recovery boundaries, and per-cycle RB comparisons are saved under
`output/defrost_decisions/<run>/`. Both conventions retain all cycles and explicit no-decision
statuses. Comparison figures are saved under `output/test/<run>/`. The shared publication renderer shows two front RGB frames, instantaneous physical COP,
water temperature, and one merged panel for absolute and normalized effective cycle COP.
The absolute axis retains its 75–102% display range; the black right axis spans 90–100.8%
of the eligible maximum, with 1%, 2%, and 5% loss thresholds. Unsupported curve sections
are dashed at full opacity. Interior display gaps use linear interpolation without
changing measured integrals, support or selected times. RB is green; unsupported RB COP is not interpolated.
Preparation sensitivity source data and its COP/time/coverage figure are under
`output/test/<run>_preparation_heat/`. The chosen definition is now preparation heat zero
(the CLI default); include remains an explicit historical sensitivity calculation.
Use `--figures-only --publish-dataset` on `effective_cop_zero` to publish only
`cycles/frost_cycle_000xxx_effective_cop.png` with Catalog key `effective_cop`.
Add `--fetch-cloud-images` to fetch only missing RB/optimum front frames through the
existing archive reader. It inherits the configured network proxy and reports network
failures separately from absent archives; downloaded frames remain cached.
Earlier include/zero decision assets and their Catalog are archived under
`output/test/effective_cop_zero/previous_dataset_decisions/`.

Without an observed defrost, the candidate window ends at the observed recording end.
A `partial` open-heating record can support hypothetical actions while retaining its manual
status; invalid/reference records and unidentified recovery remain excluded. An observed
defrost with an unknown preparation boundary is still a boundary-quality failure.
No future preparation or recovery samples are invented. The unchanged Ridge model predicts
the entire hypothetical event electricity, with all event effective heat set to zero.
`timing_vs_RB.png` displays cycle IDs against elapsed time, and annotates the mean per-cycle
relative COP improvement only over paired supported RB/optimum points. Its source is
`timing_vs_RB.csv`; the annotation describes a model-based idealization, not observed gains.

The default `--prediction-mode cross-fitted` uses models excluding the current experiment.
`full-model` is for new experiments. All candidate features use observations strictly before
the action time; offline selection is the retrospective maximum, not a future-known online
stopping policy. Exact ties select the earliest supported candidate.

The current recovery definition uses later observations to locate an earlier transition.
It therefore does not support a confirmed online boundary or prefix-invariant COP. The
existing `--current-csv` interface and earlier model outputs are deferred; they are not a
validated realtime implementation of this offline stage definition.

## 3B. Calculate the frozen V1 label reference

```bash
uv run python calculate_v1_label_reference.py \
  --dataset dataset \
  --output-root output
```

This writes `output/defrost_decisions/v1_label_reference/candidate_decisions.csv` and
`run_settings.json`.

## 4B. Build image labels

```bash
uv run python build_image_labels.py \
  --dataset dataset \
  --label-source cost-optimum \
  --source-table output/defrost_decisions/v1_label_reference/candidate_decisions.csv \
  --output output/image_labels/v1_label_reference \
  --figures
```

The training table is
`output/image_labels/v1_label_reference/image_timing_labels.parquet`.

## 5B. Train a fast CPU baseline

The first runnable model uses image-derived colour/gradient features and needs no external
feature cache:

```bash
uv run python train_image_models.py \
  --dataset dataset \
  --labels output/image_labels/v1_label_reference/image_timing_labels.parquet \
  --output output/image_models/color_gradient_rbf \
  --task binary \
  --image-features color_gradient \
  --classifiers rbf_svm \
  --camera-groups front \
  --input-features image_only \
  --workers 6 \
  --seed 0
```

Training writes `training_settings.csv`, `fold_metrics.csv`, `predictions.parquet`,
`fold_log.jsonl`, and `run_settings.json` into the selected run directory.

## 6B. Evaluate image models

```bash
uv run python evaluate_image_models.py \
  --results output/image_models/color_gradient_rbf \
  --output output/evaluations/color_gradient_rbf \
  --task binary \
  --figures
```

Evaluation is leave-one-experiment-out (LOEO), not a random image split.

## End-to-end ResNet50

```bash
uv run python train_image_models.py \
  --dataset dataset \
  --labels output/image_labels/v1_label_reference/image_timing_labels.parquet \
  --output output/image_models/resnet50_front \
  --task binary \
  --image-features resnet50_end_to_end \
  --classifiers resnet_mlp \
  --camera-groups front \
  --input-features image_only \
  --workers 1 \
  --epochs 5 \
  --seed 0
```

ResNet50 uses one worker because each fold trains on one accelerator. Frozen CPU matrices use
six outer workers, while each worker's model internals remain single-threaded.

## DINOv2 and sensor inputs

DINOv2 is advanced usage because this repository currently consumes a precomputed feature
cache. Provide it explicitly with `--dinov2-feature-cache`. Sensor options are
`image_plus_current_sensors` and `image_plus_sensor_slopes`; they are joined causally to the
image latent features before classification.

## Pareto boundary learning: reusable experiment entry

This workflow learns the frozen Pareto-knee boundary from native front-camera frames.
Run from the repository root with `dataset/` prepared. Build the teacher-independent
RGB cache once; subsequent runs extract only missing image keys:

```bash
uv run python -m image_models.dinov2_features --dataset dataset \
  --output output/image_models/_cache/dinov2_vits14_r256_c224_front_v1 --n-jobs 6
```

To repair selected cycles, repeat this command with `--cycles frost_cycle_000006`.
Compatible existing vectors can be imported with `--source-cache PATH`.
Use `--device mps` on Apple Silicon or `--device cuda` on a CUDA GPU.
Changing the teacher, method, seed or training run does not invalidate RGB vectors.

Prepare measured cumulative quantities and causal sensor statistics independently of G:

```bash
uv run python train_pareto_boundary.py --action prepare --dataset dataset --n-jobs 6
```

Review `output/image_models/_cache/pareto_boundary_v1/cycle_coverage.csv` and RGB coverage
before fitting the fold-specific G models and teachers:

```bash
uv run python train_pareto_boundary.py --action fit --n-jobs 6
```

Review teacher coverage and experiment exclusions before training the first method:

```bash
uv run python train_pareto_boundary.py --action train --method baseline --seed 0 \
  --output output/test/pareto_boundary/baseline_seed0 --n-jobs 6
```

Review each stage and each method before proceeding; do not automate the complete
comparison as one unattended loop. Reuse the same prepared data for `economic`
(economic input only), `relation` (Pareto relation supervision only), and `combined`
(both); `baseline` enables neither. `nonvisual` is the matched economic-only control
with the visual branch zeroed. Each method/seed uses its own `--output` directory.

The outcome-representation experiment uses the same prepared teacher. `represent`
learns one visual and one independent sensor-only encoder per outer fold from the four
observed outcomes at the actual preparation start; `heads` freezes those encoders and
fits S0–S4/N2 stopping probes:

```bash
uv run python train_pareto_boundary.py --action represent \
  --data output/image_models/_cache/pareto_boundary_outcome_v1 \
  --representations output/image_models/_cache/outcome_pareto_v1 \
  --allow-extrapolation --n-jobs 6

uv run python train_pareto_boundary.py --action heads \
  --data output/image_models/_cache/pareto_boundary_outcome_v1 \
  --representations output/image_models/_cache/outcome_pareto_v1 \
  --output output/test/pareto_boundary_outcome_v1 \
  --allow-extrapolation --n-jobs 6

uv run python -m plots.pareto_learning \
  --runs output/test/pareto_boundary_outcome_v1 \
  --data output/image_models/_cache/pareto_boundary_outcome_v1 \
  --representations output/image_models/_cache/outcome_pareto_v1 \
  --output output/test/pareto_boundary_outcome_v1/figures
```

S0 uses the learned outcome representation; S1 adds current C/H; S2 adds one causal
five-minute C/H trajectory; S3 additionally adds O; S4 applies the revised signed
relation loss to the input selected inside each fold; N2 is the matched sensor-only
outcome representation plus S2 economics. These probes imitate the frozen extrapolated
Pareto teacher and do not redefine its knee or establish a validated economic optimum.

Replay triggers at the first native frame with `logit >= 0`. Observation gaps remain
missing, with no interpolated frames; plots break lines across gaps longer than 45 s.
This is offline imitation and retrospective replay, not validation of actual online control.

After adding raw data, use a new `--data output/image_models/_cache/SNAPSHOT_NAME`
consistently for prepare, fit and train. Keep snapshots under the same cache parent to
reuse unchanged per-cycle `pareto_measured_stat6_v1` tables, and retain the RGB cache
(the training default points to its `cycles/` directory). Preserve previous snapshots
and run directories; choose new paths rather than overwriting historical results.

## Weights & Biases

W&B remains optional and is controlled only at the public training entry:

```bash
uv run python train_image_models.py ... \
  --wandb-project PROJECT \
  --wandb-run-name RUN_NAME
```

Training remains valid without W&B; tracking failures do not change fold calculations.
The Pareto entry accepts the same optional `--wandb-project PROJECT` argument for
`train`, `represent`, and `heads`.

## Source layout

```text
dataset_tools/          Dataset loading, validation and Raw→Dataset operations
defrost_event_models/   observed defrost events, Ridge fitting and LOEO validation
defrost_decision/       candidate quantities, objectives and Pareto selection
image_labels/           image-time label construction
image_models/           image/sensor features, classifiers, ResNet50 and evaluation
plots/                  one shared publication rendering path
tests/                  scientific, interface and parity checks
```

Historical inverse-COP definitions are isolated under `defrost_decision/baselines/`; the
current Pareto route does not import them.

## Release scope

This is a **Dataset-ready** code release, not a clone-and-run data release. The processed
experimental Dataset is not included. Do not describe the repository as fully reproducible
until the Dataset or a public download route is released. Repository renaming, license choice,
and citation metadata remain release-owner decisions and are intentionally not fabricated by
this refactor.

### R-COP32：相对最优有效 COP 回归

模型命名为 **R-COP32**（Relative Effective COP Regression，32 维表征）。
Tref 口径下已完成的 AdamW 基线冻结在 `output/image_models/relative_cop_tref/` 与
`output/image_models/d32_cop_tref/`，不代表新训练入口的默认训练流程。

当前使用 PINN4SOH 的 Adam、预热／余弦调度和批次训练；PINN 使用官方三项损失，
普通回归只使用配对数据 MSE。最多 200 epoch、耐心 25、seed 0 是本项目预算。
主体 `Solution_u` 输出相对最优有效 COP；`--regression-architecture pinn4soh`
增加动态网络 `[x,t,u,u_x,u_t] → G`，不增加传感器编码器或 RGB 投影。
导数采用标准化坐标，包含 RGB 特征的偏导不解释为已知热力学定律。

```bash
uv run python train_pareto_boundary.py --task relative-cop-regression \
  --regression-suite --dataset ../../dataset \
  --decision-run output/defrost_decisions/effective_cop_tref \
  --event-run output/defrost_event_models/heating_start_zero \
  --reference-run output/image_models/relative_cop_tref \
  --rgb-cache ../../output/image_models/_cache/dinov2_vits14_r256_c224_front_v1/cycles \
  --output output/image_models/pinn4soh_cop --figure-output output/test/pinn4soh_cop \
  --n-jobs 6
```

四组依次为 R-COP32-Sensor、PINN4SOH-COP-Sensor、PINN4SOH-COP-RGB、R-COP32-RGB。
随后仅根据内部验证选点 COP 损失、2% 命中率、验证 MSE 选择一组增加峰值加权训练；
外层测试不参与配置选择。该补充结果属于经模型筛选后的探索结果。
不使用套组时，通过 `--regression-architecture r-cop32|pinn4soh|d32`、`--rgb off|on`
与 `--peak-weighted` 单独运行相同路径。D32 保留 RGB 输入要求。

输入为去湿度后的 27 个通道与候选 COP 的 196 项当前／历史特征、累计有效热量、
总电量及最后一列时间，共 199 维；RGB 开启时直接增加冻结的 384 维特征至 583 维。
整列缺失特征在训练折删除，填补和标准化仅从训练实验拟合。质量标记不进入网络。
RGB 关闭时不访问照片；开启时只向后匹配 45 秒内的照片，缺图不伪造特征。
候选 COP 与标签均由按折排除实验后的事件电耗 Ridge 生成；完整参考曲线最大值
只作监督分母，不进入输入，也不因缺图改变。全部阶段、Tref、特征规则保持一致。

`--reference-run` 复用已核对定义的基础表、Ridge 参数及固定实验划分，不复用旧预测。
每个实验折完成后保存 checkpoint，可通过同一命令续跑；`--heldout-experiment`
可以先检查单折。`--batch-size` 默认 256，每批次更新，不使用整 epoch 梯度累计。
动态损失系数由 `--pinn-alpha`、`--pinn-beta` 指定，默认均为 1。

共享 renderer 输出原始 COP Fig.2、相对 COP 留出预测图、各组自身覆盖与共同可用时刻
的比较图及源数据。主要选点指标为 COP 损失、1%／2%／5% 命中率、相对近优点集合的
提前／延后分钟数；全轨迹 MSE 保留。未识别／缺输入循环保留状态，参考峰值始终固定。
Dataset 的解析决策图不被神经网络预测图替换。

当前有效供热口径：压差重建结束、稳定制热开始、有效供热开始共用
`stable_heating_start`。固定 `Tref = water_out_temperature(stable_heating_start)`，
此后出水温度低于 Tref 的区间热量置零，电耗仍完整计入；等于 Tref 时计热。
边界温度只在不超过 30 秒的有效观测间线性插值，无法确定时不提供有效 COP。
相邻观测跨越 Tref 时按线性交点划分积分区间。旧 `relative_cop` 结果属于未施加
温度门槛的历史口径，予以保留。标签来自 Ridge 参考曲线，不是实测真值；
预测最大点是离线比较结果，不是实时触发规则。

在线触发回放沿用留出预测，无需重新训练：上述入口使用 `--action evaluate --runs`
列出各模型目录，配合 `--trigger-threshold 0.99 --processing-seconds 30` 与新的输出目录。
从首个候选时刻开始按固定时钟检查；单次阳性立即触发，三帧内两次在确认时触发，
缺输入保留为未命中时隙。RB 保留原有一秒信号历史和内部持续条件，仅外层按相同
30 秒时钟检查。阶段边界固定为已确认的离线结果，未来参考峰值只用于事后评分。
输出逐时刻轨迹、触发时刻、未触发／不可评分状态、各自及共同可评分循环统计；
不把未触发自动补成 RB 或离线最优点，不在观测结束之后外推。

第二轮状态一致性采用 `--regression-architecture state-consistency --rgb on
--pinn-alpha 1 --pinn-beta 0`，模型命名为 PSDC-COP32-RGB。Solution_u 和 Data-only
保持相同；G 输入 `[p_current, p_previous, time_current, u_previous]` 为 34 维，
预测当前输出，采用数据 MSE 加状态一致性 MSE，联合更新两个网络。不使用偏导、
方向惩罚、上界惩罚或 latent dynamics。16 个状态直接取现有 14 项当前运行量及
两项累计量；只对间隔恰好 10 秒的相邻样本施加一致性，不改变数据损失的样本。
参考 `output/image_models/pinn4soh_cop/r-cop32_on` 复用折划分、基础表和 Ridge，
从相同随机初始化、以相同预算训练新模型，已有 Data-only 权重保留作比较而非冻结
预测器后只拟合 G。`state_diagnostics.csv` 对比 G 与上一时刻预测直接保持的误差。
时间和有效热量均以离线恢复边界及 Tref 为给定条件；固定这些条件后，热量积分
只使用候选之前的样本。这里不声称完整在线因果，也不称其为已知物理控制方程。

### DINOv2-MLP：有效 COP 边界二分类 baseline

复用上述 Tref 有效供热定义、按折排除实验的事件电耗 Ridge、固定实验划分与
384 维 DINOv2 正面机位缓存。最优点取完整受支持曲线的最早最大值，缺图不移动
参考点；时间早于该点为 0，否则为 1。输入仅为 RGB 特征，分类头为
384→Linear(1000)→ReLU→Linear(64)→ReLU→Linear(2)，不使用传感器或时间。
损失为循环等权交叉熵；内部验证选轮数，外层重新训练并留一实验测试。

```bash
uv run python train_pareto_boundary.py --task effective-cop-binary \
  --dataset ../../dataset --n-jobs 6
```

默认复用 `output/image_models/relative_cop_tref` 的基础表、Ridge 参数与实验划分；
结果写入 `output/image_models/dinov2_binary_tref/`，图表写入
`output/test/dinov2_binary_tref/`。只用已缓存特征，不补提取或下载图片。
`--heldout-experiment exp_20260715` 可验证单折；同一完整命令跳过已完成折并续跑。

在线决策统一每 30 秒检查，图像仅向后匹配 45 秒，二分类阈值固定为 0.5；
所有有效 COP 模型只使用最近 3 次中至少 2 次阳性的首次确认时刻。
缺输入不能计为阳性，不再输出首次阳性策略。回归模型保留自身相对 COP 判据，
不将回归分数解释为分类概率；事后 COP 支持范围只影响评分，不屏蔽在线模型输出。
`cycle_metrics.csv` 保留缺输入、未触发、无参考及支持范围外的状态；`metrics.csv`
报告逐循环等权分类指标、时间误差、COP 损失、覆盖及以全部循环为分母的命中率。
所有 COP 损失是现有参考模型下的估计决策损失，不是实测节能收益。

### RGB32 投影的匹配输入对照

`--rgb-projection 32` 在 R-COP32 前增加 `Linear(384,32) → Sin`，再与原有199维
数值和时间拼接；默认0保留直接拼接。三组采用相同的 after-optimum BCE、内部验证
阈值选择及30秒三帧两次确认。`--evaluation-cohort rgb-valid` 也可用于训练，固定
catalog 中 COP valid 且 RGB valid 的循环；`--require-rgb-input` 使 Sensor 使用
相同照片可用时刻，但不将RGB特征输入Sensor。标签峰值仍在完整支持曲线上计算。
事件电耗Ridge仅使用COP valid事件并保持嵌套实验隔离，图片覆盖不筛除Ridge事件。

训练前先运行标签与参考支持的初步 gate 审计：

```bash
uv run python train_pareto_boundary.py --task cop-classification --action audit \
  --dataset ../../dataset \
  --event-run output/defrost_event_models/heating_start_zero \
  --decision-run output/defrost_decisions/effective_cop_tref \
  --reference-run output/image_models/cop_after_optimum_rgb \
  --rgb-cache ../../output/image_models/_cache/dinov2_vits14_r256_c224_front_v1/cycles \
  --output output/test/causal_cop_development_audit_zero_prep_fix --n-jobs 6
```

后续 support decomposition 已修正上述初步审计的分母和零准备阶段处理，结果保存在
`output/test/cop_support_decomposition/`。95个COP-valid循环中，82个具有可信参考曲线及
30秒网格。独立于照片的1% perfect-label 2/3兼容率为79/82（96.34%），2%为81/82
（98.78%）；因此保留1%标签。共享Sensor/front输入下分别为72/82、76/82，输入覆盖
与标签兼容性单独报告。93个RGB-valid仅表示有图片记录，133缺front不影响参考有效性。

RB受支持时刻在3折实验OOF下为1/95，全开发拟合下为0/95；全开发拟合仅用于覆盖诊断。
83个完整实际准备事件在全开发拟合下均受支持。固定RB规则回放早于87个实际准备时刻，
中位差68.42分钟，不能把RB回放当成已观测除霜事件。支持距离是覆盖线索，不是校准不确定性。
参考不足不阻止在受支持标签上开发分类器，但阻止可信的总体COP收益优化及独立控制资格判定。
开发干预需补控制器可能访问的动作—后果范围，实际动作和安全范围由机组负责人确认；
开发干预与最终实机确认数据永久分开。

以下循环仅记录门槛建立前的既有匹配输入实验，不是当前建议执行的下一步：

```bash
for recipe in sensor direct projection32; do
  rgb=on; projection=0
  if [ "$recipe" = sensor ]; then rgb=off; fi
  if [ "$recipe" = projection32 ]; then projection=32; fi
  uv run python train_pareto_boundary.py --task cop-classification \
    --classification-label after-optimum --rgb "$rgb" --rgb-projection "$projection" \
    --require-rgb-input --evaluation-cohort rgb-valid --dataset ../../dataset \
    --reference-run output/image_models/cop_after_optimum_rgb \
    --output "output/image_models/cop_rgb_valid_$recipe" \
    --figure-output "output/test/cop_rgb_valid_$recipe" --n-jobs 6 || break
done
```

三个输出独立保留，重复命令可续跑已完成折。固定离线恢复边界和Tref的回放限制不变；
这组实验检验融合结构，不能与原119循环训练结果混作单因素消融。

本次 seed0 匹配输入实验完成87循环、19折处理，其中18折可训练；
`exp_20260723` 的内部验证无受支持样本，保留不可校准状态，不更换验证实验。
三组参考曲线、标签、输入时刻、内部验证及Ridge参数已核对一致。Sensor／直接RGB／
RGB32投影分别有12／8／17个可评分循环，1%近优命中为8／4／8个（分母均87）。
本次按COP valid事件重新拟合Ridge后，原始RB的87个时刻均在支持域外；
三模型共同可评分仅4循环，不能据此宣称投影改善相对RB的COP收益。
比较图与逐循环源数据：`output/test/cop_rgb_valid_fusion_comparison/`。

RGB可用性与固定阈值回放：`rgb_valid` 仅表示该循环有图像记录（或已有RGB输入），
不再要求峰值之后有2/3确认机会。模型的front视角缺失仍单独记录为缺输入。
`--action evaluate --evaluation-threshold 0.5 --allow-extrapolation` 在共享评价入口中
重新回放冻结概率：所有分类模型使用固定0.5及30秒2/3确认；域外但测量、能耗和
特征质量有效的COP保留评分，并记录 `outside_reference_support`。参考最优点仍为
原受支持曲线的最大值，不由域外值改写。缺失或无效COP仍不可评分；新增valid循环
没有历史预测时记录 `no_frozen_prediction`，不从总队列静默删除。

### 因果近优分类开发（当前三轮）

当前分类开发使用95个COP-valid循环的固定3折实验分组，标签仅在交叉拟合Ridge的
可信参考域内定义为有效COP不低于该折峰值的99%。不受支持的时刻保留为未知，不能当作负类。
30秒输入从已核验制热启动事件计时，使用191维数值量；删除候选有效COP历史和Tref累计热量。
当前front向后匹配45秒；第三轮另要求150秒前、同循环45秒窗口内的历史front。
Sensor和RGB使用相同输入可用时刻，未输入时隙保留，不压缩2/3确认时间。

数值分支191→128→64→32，视觉分支384→64→32，分类头64→64→32→1。
Sensor以零向量代替视觉分支输出。视觉差分模型使用当前DINO与当前减历史DINO的768维拼接；
三组有效参数量分别为41,185、67,905、92,481，并非参数完全匹配。
隐藏层SiLU、Xavier初始化、零偏置；AdamW 1e-3、weight decay 1e-4、batch256、50epoch。
第一轮循环等权BCE；第二轮固定标签平滑 .9d+.05；第三轮检验视觉变化，并增加同历史可用域的
Sensor和静态RGB对照。评估始终使用原硬标签，不开展超参数扫描。

```bash
uv run python train_pareto_boundary.py --task cop-classification --action develop \
  --data output/test/causal_cop_development_audit_zero_prep_fix --dataset ../../dataset \
  --rgb-cache ../../output/image_models/_cache/dinov2_vits14_r256_c224_front_v1/cycles \
  --rgb on --development-mechanism delta --development-loss label-smoothing \
  --maximum-epochs 50 --batch-size 256 --seed 0 --n-jobs 6 \
  --output output/test/cop_development_delta_rgb
```

七组冻结预测统一到5,946个有标签共享时刻、76循环，所有95循环状态仍保留。
共同范围Balanced Accuracy：原BCE Sensor 81.81%、原BCE RGB 81.96%、平滑Sensor 81.32%、
平滑RGB 82.15%、历史匹配Sensor 81.59%、历史匹配静态RGB 82.25%、差分RGB 82.62%。
按预定BA→macro-F1→BCE→更少epoch→更高阈值选择差分RGB（50epoch、阈值0.20）。
其相对历史匹配Sensor的BA增量为1.03个百分点，实验聚类95%区间−2.24至4.57个百分点；
相对历史匹配静态RGB为0.37个百分点，区间−2.49至3.21个百分点。
这些是经过选择的开发数据描述性区间，不是独立确认，不能证明稳定RGB增益。
图及源数据：`output/test/cop_development_comparison/`。

差分RGB在95循环上有91次首次2/3提议，其中89次不受支持，仅2次可评分。
这不是实际执行策略或COP收益证据；当前分类能力仅针对可信参考域中的Ridge代理标签。
研究用全开发重拟合权重不构成独立控制资格，也不把训练成绩当作泛化结果。

研究分类器重拟合：

```bash
uv run python train_pareto_boundary.py --task cop-classification --action refit-development \
  --reference-run output/test/cop_development_delta_rgb \
  --output output/image_models/cop_near_optimal_final --n-jobs 6
```

`selected.pkl`包含模型、训练折预处理、列顺序及冻结研究阈值。重拟合使用自身实验被排除的
Ridge标签，共6,030时刻、80循环；12循环无可训练支持标签，3循环无共同输入。
重新加载后预测差异为0。该文件没有部署gate，阈值仅供研究回放，不能直接接入机组控制。

冻结后的回顾性稳定性评价使用以下独立入口：

```bash
uv run python train_pareto_boundary.py --task cop-classification \
  --action evaluate-frozen-development \
  --runs output/test/cop_development_delta_rgb output/test/cop_development_history_sensor \
  --output output/test/cop_frozen_retrospective --n-jobs 6
```

固定视觉差分方案及历史匹配Sensor对照，运行seed 0、1、2和冻结队列全部实验留一。
每个外层训练集内部沿用固定3个实验组（移除外层实验），仅以内部OOF选择epoch和阈值；
外层实验永久排除于该折所有Ridge及预处理拟合。各折参考支持域由该折Ridge确定，
不能要求其与原3折开发支持数量相同。不存在可信标签时保留预测与不可评价状态。
只并行外层任务，worker内部单线程。完成的外层任务可续跑；所有阈值和参数独立保存。

这不是未接触过的确认数据：模型方案曾用同一批实验开发。评价结束后不据此修改方案，
三种seed不作为独立实验重复；先在同循环内平均seed差值，再按实验聚类计算区间。
分类稳定性与实际控制COP增益分别解释，后者仍需补齐动作后果数据及独立随机交叉实机确认。

冻结回顾性评价已完成两组×3seed×19实验的114个外层任务（含内部验证及重拟合共456次拟合）。
每个模型/seed保留全部95循环：80可评价、12无受支持标签、3无输入；80中73具备两类，
7为单类。循环BA按共享实现对实际出现的类别求平均，并单列both-class状态。

| 冻结方案 | seed0 BA | seed1 BA | seed2 BA | seed平均BA | seed平均macro-F1 |
|---|---:|---:|---:|---:|---:|
| 历史匹配Sensor |79.59%|80.36%|79.50%|79.81%|0.7448|
| 视觉差分RGB |79.43%|78.90%|78.51%|78.95%|0.7405|

80个共同可评价循环、19实验上，先平均同循环的3seed差值后聚类bootstrap，RGB−Sensor
的BA差值为−0.87个百分点，95%区间−2.63至0.73个百分点。开发阶段的微小优势没有在此
回顾性稳定性评价中重现，不能宣称RGB有稳定增量，也不据此继续修改或重新选择冻结方案。
每个RGB seed有91次首次2/3提议，其中90次不受支持，1次受支持近优；Sensor分别有
91/92/91次提议，90/91/90次不受支持，均仅1次受支持近优。未提议及不可评分不从分母删除。
这仍没有建立可靠的COP收益比较或独立控制资格，不把受支持的1次成功解释成100%总体命中。
原始逐折参数/阈值、预测及全部状态：`output/test/cop_frozen_retrospective/`；
图与配对区间源数据：其`figures/`目录。

### 五个冻结策略相对原始RB的COP比较

```bash
uv run python train_pareto_boundary.py --task cop-classification \
  --action compare-frozen-policies \
  --runs output/test/cop_frozen_retrospective \
    output/image_models/cop_after_optimum_sensor \
    output/image_models/cop_after_optimum_rgb \
    output/image_models/dinov2_binary_tref \
  --output output/test/cop_five_policy_comparison
```

不重训。比较93个RGB-valid循环，新模型保留各外层内部选择的阈值及全部3seed，
旧R-Sensor、R-Sensor+RGB及Chen保留0.5；各自保存的30秒时钟与首次2/3规则不变。
这是整套冻结策略比较，不是网络架构单因素消融。固定RB原始时刻不增加确认延迟。
五个方法使用同一逐实验留出的Ridge和old10s∪new30s支持域峰值；域外评分不移动峰值。
逐循环增益为100×(COP_trigger/COP_RB−1)，再对配对循环等权平均。

当前RB的正式支持为0/93，因此正式收益与参考改善空间全部不可评价。下面只作为
通过测量/物理有效性检查的Ridge外推估计，不能当作实测控制收益：

| 策略 | Seed | 各自可评分循环 | 相对RB平均COP变化 |
|---|---:|---:|---:|
| 新Sensor |0|15|−80.64%|
| 新Sensor |1|10|−76.91%|
| 新Sensor |2|13|−81.99%|
| 新视觉差分RGB |0|3|−55.44%|
| 新视觉差分RGB |1|14|−83.83%|
| 新视觉差分RGB |2|12|−76.98%|
| 旧R-Sensor |原冻结|88|−10.32%|
| 旧R-Sensor+RGB |原冻结|85|−14.19%|
| Chen |原冻结|85|+1.61%|

以上各行覆盖不同，不直接排名。五方法且全部3seed共同可评分的交集仅036、095、134三循环：
新Sensor/新RGB均−55.44%，旧R-Sensor−28.29%，旧R-Sensor+RGB−1.13%，Chen+3.46%。
其中仅095、134存在受支持参考峰值；该两循环平均参考空间3.18%，不可直接减去三循环平均收益。
全队列84个有支持峰值循环的参考峰值相对外推RB平均差值为+1.46%（中位2.34%），
65个超过1%、30个超过3%，15个为负。由于RB处于支持域外，这不是全局物理上限或可保证提升。

新Sensor的76/82/78循环、新RGB的88/77/78循环在有效供热核算起点之前提出首次确认，
单独记为before_reference_accounting_start，不推迟时刻、不填零COP、不从队列删除。
旧模型缺011/012/037留出预测也明确保留。原始930行策略/seed/循环状态与派生图表源数据
保存在`output/test/cop_five_policy_comparison/`，所有历史结果保持原样。

### 五架构的成对 stopping loss 比较

历史分类与策略结果保持冻结。新的正式比较在每个架构自己的输入可用域和历史候选时钟内，
只保留可信有效COP点作为合法动作；同一架构的前/后二分类与COP optimal-stopping使用相同
动作行、seed、优化器、cycle batch和训练预算。验证及测试均使用首次1/1触发，未触发时在
最后合法点强制动作；阈值只按inner-validation回放后的平均循环COP选择。

```bash
uv run python train_pareto_boundary.py --task cop-classification \
  --action compare-stopping-losses \
  --data output/test/causal_cop_development_audit_zero_prep_fix \
  --maximum-epochs 200 --patience 25 --batch-size 8 --n-jobs 6 \
  --output output/image_models/stopping_loss_comparison
```

默认运行五个架构、两个loss及seed 0/1/2。`--stopping-architectures`、
`--stopping-losses`、`--seeds`及`--heldout-experiment`可用于最小烟雾检查和续跑。
`--runs`可按R-Sensor、R-Sensor+RGB、Chen RGB、New Sensor、New RGB difference顺序
显式替换五个冻结基础表目录。主输出为逐时刻预测、逐循环COP/RB/headroom/早触发与
soft-hard gap、阈值网格及汇总表；无合法动作或RB不在可信支持域的循环保留状态而不填值。
