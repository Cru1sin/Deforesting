# Image-guided heat-pump defrost timing

This repository is a Dataset-ready research workflow for selecting heat-pump defrost times
and learning the resulting timing labels from RGB images. The processed experimental Dataset
is not included; prepare it according to [DATASET_FORMAT.md](DATASET_FORMAT.md) first.

The code has two deliberately separate routes.

## Route A — Current multi-objective defrost decision

```text
Dataset
→ fit complete defrost-event models
→ calculate candidate quantities
→ cycle COP / heating rate / evaporator capacity
→ cycle-COP–heating-rate Pareto front
→ selected defrost time
```

Cycle COP and cycle heating rate select the Pareto compromise. Evaporator capacity is an
independent reference objective: it is reported and plotted but never changes the selected
time.

For a candidate defrost time \(\tau\), the three reported objectives are

\[
C(\tau)=\frac{Q_H(\tau)+\hat Q_T(\tau)}{E_H(\tau)+\hat E_T(\tau)},
\]

\[
H(\tau)=\frac{Q_H(\tau)+\hat Q_T(\tau)}{t_H(\tau)+\hat D_T(\tau)},
\]

\[
O(\tau)=\frac{Q_H(\tau)-E_{\mathrm{comp},H}(\tau)
+\hat Q_T(\tau)-\hat E_{\mathrm{comp},T}(\tau)}
{t_H(\tau)+\hat D_T(\tau)}.
\]

The symbols remain useful in the paper; CSV files and Python interfaces use complete names:
`cycle_cop`, `cycle_heating_rate_kw`, and `cycle_evaporator_capacity_kw`.

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

## 3A. Fit complete defrost-event models

The released model file is
`defrost_event_models/parameters/released_ridge_models.json`. Refit after adding experiments:

```bash
uv run python fit_defrost_event_models.py \
  --dataset dataset \
  --run-name new_data_refit \
  --workers 6 \
  --output-root output
```

```text
output/defrost_event_models/new_data_refit/
├── defrost_events.csv
├── model_validation.csv
├── candidate_model_parameters.json
└── run_settings.json
```

`candidate_model_parameters.json` is review-only and never overwrites the released parameters
automatically. Promote it only after scientific review and full-cycle parity checks.

All four event outcomes currently use the same complete-case training cohort. This keeps their
empirical support domain comparable, but discards events missing any one outcome;
`model_validation.csv` records both each outcome's available count and the final common count.

## 4A. Select defrost times and render publication figures

For leakage-free retrospective evaluation of experiments already represented by saved LOEO
folds, use `cross-fitted`:

```bash
uv run python select_defrost_time.py \
  --dataset dataset \
  --model-file defrost_event_models/parameters/released_ridge_models.json \
  --prediction-mode cross-fitted \
  --run-name current \
  --workers 6 \
  --figures \
  --output-root output
```

After refitting, point the same command at
`output/defrost_event_models/new_data_refit/candidate_model_parameters.json` to evaluate the
new experimental batch without training on each held-out experiment.

For application to a genuinely new experiment whose ID has no saved retrospective fold, use
the model fitted on all available training experiments:

```bash
uv run python select_defrost_time.py \
  --dataset dataset \
  --model-file defrost_event_models/parameters/released_ridge_models.json \
  --prediction-mode full-model \
  --run-name new_experiment \
  --workers 6 \
  --figures \
  --output-root output
```

Both `candidate_decisions.csv` and `run_settings.json` record the prediction mode and model
training scope; cross-fitted and full-model results must not be presented as the same estimate.

```text
output/defrost_decisions/current/
├── candidate_decisions.csv
├── run_settings.json
└── figures/
```

`candidate_decisions.csv` contains every candidate quantity, each objective's own eligibility and
near-optimal basin, the C–H Pareto front, and `selected_defrost_time`. The Pareto panel colours
points by the reference-only evaporator-capacity objective.

To train against the current Pareto-selected boundary rather than the frozen V1 reference:

```bash
uv run python build_image_labels.py \
  --dataset dataset \
  --label-source selected-time \
  --source-table output/defrost_decisions/current/candidate_decisions.csv \
  --output output/image_labels/current_pareto
```

Use `--labels output/image_labels/current_pareto/image_timing_labels.parquet` and
`--label-column binary_target` in the training command below.

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
