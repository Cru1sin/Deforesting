# PINN4SOH Frost Sensor Workspace

One Dataset-native path from frost-cycle measurements to reproducible image-model
evaluation:

```text
Data -> Cost -> Labels -> Train -> Evaluate
```

## Current scientific status

- **Current cost under study:** V2.6.8
- **Current label source:** canonical V1
- **V2.6.8 status:** diagnostic-only; it must not generate training labels
- **Validation:** leave-one-experiment-out (LOEO)

V2.6.8 studies a corrected, fixed-nine-minute transition formulation. Its
minimum is a diagnostic minimum, not a promoted decision label. Training and
evaluation therefore remain anchored to the frozen canonical V1 labels.

Run all commands from the repository root with the project `uv` environment.

## 1. Data

Validate the self-contained Dataset before downstream work:

```bash
uv run python main_data.py validate --dataset dataset
```

`main_data.py` also owns explicit Dataset maintenance actions: `add`, `replace`,
`aggregate-original`, `remove`, `refresh`, `review-cycle`, `edit`, and `render`.

## 2. Cost

Calculate the current V2.6.8 diagnostic curves:

```bash
uv run python main_cost.py --action calculate --cost v2.6.8 --dataset dataset --output-root output
```

The result is written under `output/cost/v2.6.8/` as one standard `cost.csv`,
per-cycle tables, and its recipe. This diagnostic output does not replace
`output/cost/v1/cost.csv` as the label source.

Cost versions are selected with `--cost` (`v1`, `v2.5`, or `v2.6.8`). Named
variants use `--variant` together with the explicit supported recipe arguments.
`--action compare --results ...` sends comparable runs through the shared cost
plotter in `plots/cost.py`.

## 3. Labels

Build hard RGB labels from canonical V1 and render the default PNG figures:

```bash
uv run python main_labels.py --dataset dataset --cost-csv output/cost/v1/cost.csv --output output/labels/v1 --figures
```

The standard label tables are `image_cost_labels.parquet`, `label_balance.csv`,
and `cycle_audit.csv`. The default thresholds are `0.01`, `0.02`, `0.05`, and
`0.10`; override them with `--thresholds` when the analysis requires it.

## 4. Train

Train the baseline image model with LOEO folds:

```bash
uv run python main_train.py --dataset dataset --labels output/labels/v1/image_cost_labels.parquet --output output/models/current --task binary --representations handcrafted --heads rbf_svm --cameras front --modalities rgb --jobs 6
```

Parallel representations, heads, cameras, and modalities are selected
explicitly through their matching arguments. Available values are listed by
`--help`, and setting compatibility is validated before training; accepted
settings share the same fold orchestration and standard `settings.csv`,
`metrics.csv`, and `predictions.parquet` outputs.

## 5. Evaluate

Recompute summaries from the frozen LOEO predictions and render figures:

```bash
uv run python main_evaluate.py --results output/models/current --output output/models/evaluation --task binary --figures
```

Evaluation writes the standard `experiment_metrics.csv` and `summary.csv`
tables and uses the shared model plotter in `plots/model.py`. Multiple runs can
be compared by passing more directories to `--results`.

## Figure formats

Label and evaluation figures default to PNG. Request alternatives explicitly
with `--figure-format svg`, `--figure-format pdf`, or multiple values. Cost
comparisons use the shared publication renderer in `plots/cost.py`.

## Code map

Core:

- `main_data.py`, `main_cost.py`, `main_labels.py`, `main_train.py`,
  `main_evaluate.py`: the five primary workspace entry points.
- `dataloader/`, `cost/`, `labels/`, `model/`: loading and scientific
  calculations.
- `plots/`: shared publication rendering for comparable outputs.

Supporting areas:

- `src/frost_analysis/cli.py` and the module-entry source remain as
  legacy/compatibility code, but are not the current workspace CLI.
- `src/frost_analysis/` still contains migrating or reused cost, labels,
  training, evidence, and figures code.
- `scripts/`, `configs/`, `docs/`, `archive/`: exploration, supporting material,
  and historical utilities; not the primary workflow.
- `tests/`: focused scientific and interface checks.
- `output/`: generated results; exploratory evidence belongs under
  `output/test/`, while formal cost, label, and model artifacts stay in their
  named top-level output directories.

Each stage exposes its authoritative interface through its `--help` flag.
Run a retained script from the repository root as a module, for example
`uv run python -m scripts.labels.audit_rgb_cycle_assets --help`.
