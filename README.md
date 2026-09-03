# Image-guided defrost timing for air-source heat pumps

This repository implements one Dataset-native scientific workflow:

```text
Data -> Cost -> Labels -> Train -> Evaluate
```

Current scientific status:

- Canonical V1 remains the only source of hard RGB training labels.
- V2.5 and V2.6.8 are diagnostic cost definitions and are not label-eligible.
- V2.6.8 evaluates a corrected fixed-nine-minute post-defrost formulation with
  empirical-support gating; its minimum is diagnostic only.
- The current offline policy fits four V2.6.8 transition outcomes, constructs
  independent C/H/O objectives, and selects a knee from the C-H Pareto front.
  O is reference-only and cannot move the selected time.
- Image models are evaluated with leave-one-experiment-out (LOEO) validation.

The project-specific method is:

```text
Observed transition events -> measured ET and QT
Candidate pre-action state -> nested-LOEO Ridge -> predicted ET(tau), QT(tau)
Measured EH(tau), QH(tau) + predicted transition -> support gate -> diagnostic minimum

V2.6.8 candidates + predicted ET, QT, compressor ET and transition duration
-> independent C/H/O objectives -> C-H Pareto front -> knee policy

Candidate cost curve -> interpolate regret at image times
-> before optimum | near optimum | after optimum
-> binary labels omit near-optimal images

For each representation x head x camera x modality
-> hold out one experiment -> fit on the others -> predict the held-out experiment
-> repeat for every experiment -> experiment-weighted summary
```

Run commands from the repository root. An existing Dataset is expected at
`dataset/`.

```bash
uv sync
```

Add `--extra ml` only when training `resnet50_finetune`.

## Data

Daily Dataset reads use `dataloader/`. Raw-data construction and processing live
under `dataloader/builder/`.

```bash
uv run python main_data.py validate --dataset dataset
```

`main_data.py` also provides `add`, `replace`, `aggregate-original`, `remove`,
`refresh`, `review-cycle`, `edit`, and `render`. See the complete arguments with
`uv run python main_data.py --help`.

## Cost

All three versions minimize inverse cycle COP,

\[
J(\tau)=\frac{E_H(\tau)+E_T(\tau)}{Q_H(\tau)+Q_T(\tau)},
\]

but preserve separate scientific definitions and decision policies:

| Version | Formula components | Decision policy | Labels |
| --- | --- | --- | --- |
| V1 | measured unit heat; quadratic transition electricity plus fixed recovery; \(Q_T=0\) | historical extrapolation allowed; eligible argmin | canonical |
| V2.5 | measured water heat; quadratic transition electricity; linear preparation plus signed quadratic defrost heat | historical extrapolation allowed; eligible argmin | diagnostic |
| V2.6.8 | measured water heat from fixed post-defrost minute 9; independently fitted full-transition \(E_T\) and \(Q_T\) | empirical support and continuous five-minute eligibility required | diagnostic |

Calculate the canonical label source:

```bash
uv run python main_cost.py --action calculate --cost v1 --dataset dataset --output-root output
```

Select `v2.5` or `v2.6.8` with `--cost` for diagnostic calculations. Named
variants require `--variant` plus explicit supported recipe overrides.

Refit the diagnostic V2.6.8 transition models when its training data or feature
definition changes:

```bash
uv run python main_cost.py \
  --action fit \
  --cost v2.6.8 \
  --dataset dataset \
  --variant dynamic8_refit \
  --output-root output
```

This writes `events.csv`, `validation.csv`, `bootstrap.csv`, and
`params_candidate.json` under `output/cost/fit/dynamic8_refit/`. The candidate
parameters never replace the frozen model automatically.

Apply a reviewed four-outcome artifact to the offline C-H Pareto policy:

```bash
uv run python main_cost.py \
  --action policy \
  --dataset dataset \
  --parameters output/cost/fit/dynamic8_refit/params_candidate.json \
  --variant dynamic8_refit \
  --output-root output \
  --n-jobs 6
```

For each candidate time, the policy calculates

\[
C=\frac{Q_H+\hat Q_T}{E_H+\hat E_T},\qquad
H=\frac{Q_H+\hat Q_T}{t_H+\hat D_T},\qquad
O=\frac{Q_H-E_{comp,H}+\hat Q_T-\hat E_{comp,T}}{t_H+\hat D_T}.
\]

It maximizes C and H, constructs their Pareto front, and selects its
normalized-chord knee (or the relative ideal-point fallback for a degenerate
front). O is plotted for physical interpretation only. The policy writes one
shared `cost.csv` plus per-cycle tables under `output/cost/policy/<variant>/`;
it is an offline working policy and cannot yet generate hard RGB labels.

Compare completed runs through the shared renderer:

```bash
uv run python main_cost.py --action compare --results RUN_A RUN_B --dataset dataset --output-root output
```

## Labels

```bash
uv run python main_labels.py \
  --dataset dataset \
  --cost-csv output/cost/v1/cost.csv \
  --output output/labels/v1 \
  --figures
```

The standard outputs are `image_cost_labels.parquet`, `label_balance.csv`, and
`cycle_audit.csv`. Default relative-cost thresholds are 0.01, 0.02, 0.05, and
0.10.

## Train

### Fast CPU baseline

```bash
uv run python main_train.py \
  --dataset dataset \
  --labels output/labels/v1/image_cost_labels.parquet \
  --output output/models/current \
  --task binary \
  --representations handcrafted \
  --heads rbf_svm \
  --cameras front \
  --modalities rgb \
  --jobs 6 \
  --seed 0
```

The trainer writes `settings.csv`, `metrics.csv`, `predictions.parquet`, and a
fold-completion `task_log.jsonl`. Available representations, heads, cameras,
modalities, and compatibility rules are listed by `--help`.

### End-to-end ResNet50

```bash
uv sync --extra ml
uv run python main_train.py \
  --dataset dataset \
  --labels output/labels/v1/image_cost_labels.parquet \
  --output output/models/resnet50_front \
  --task binary \
  --representations resnet50_finetune \
  --heads paper_mlp \
  --cameras front \
  --modalities rgb \
  --jobs 1 \
  --epochs 5 \
  --seed 0
```

W&B tracking is optional. Install and authenticate it only when needed:

```bash
uv sync --extra tracking
uv run wandb login
uv run python main_train.py ... --seed 0 --wandb-project PROJECT --wandb-run-name NAME
```

Without `--wandb-project`, W&B is not imported; local outputs remain
authoritative.

## Evaluate

```bash
uv run python main_evaluate.py \
  --results output/models/current \
  --output output/models/evaluation \
  --task binary \
  --figures
```

Evaluation recomputes `experiment_metrics.csv` and `summary.csv` from frozen
LOEO predictions. Label and evaluation figures default to PNG and accept
`--figure-format svg` or `--figure-format pdf`.

## Code map

- `main_data.py`, `main_cost.py`, `main_labels.py`, `main_train.py`,
  `main_evaluate.py`: primary command-line entry points.
- `dataloader/`: Dataset reading, validation, maintenance, and raw-data builders.
- `cost/`: versioned cost definitions, boundaries, component models, and fitting.
- `labels/`: canonical cost-to-image label construction.
- `model/`: feature extraction, model fitting, and evaluation.
- `plots/`: shared publication rendering.
- `docs/`: Dataset and first-principles scientific contracts.
- `paper_zh/`: Chinese manuscript and publication figures.
- `tests/`: focused scientific and interface checks.

| Question | Reading order |
| --- | --- |
| V1 or V2.5 cost | `main_cost.py` -> matching `cost_function_*.py` -> component model |
| V2.6.8 candidate cost | `main_cost.py` -> `cost/cost_function_v2_6_8.py` |
| V2.6.8 targets and Ridge validation | `cost/v2_6_8_data.py` -> `cost/fit_v2_6_8.py` -> `cost/validate_v2_6_8.py` |
| C/H/O Pareto policy | `main_cost.py` -> `cost/cho.py` -> `cost/objectives.py` -> `cost/policy.py` |
| RGB labels | `main_labels.py` -> `labels/build.py` |
| Frozen-feature training | `main_train.py` -> `model/features.py` -> `model/model.py` |
| ResNet50 fine-tuning | `main_train.py` -> `model/resnet.py` |
