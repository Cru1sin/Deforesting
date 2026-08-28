# Frost Sensor Analysis

The repository has one production workflow:

```text
Raw data -> Dataset -> Cost v1/v2 -> RGB labels -> Training/evaluation
```

Exploration, evidence analysis, and figure rendering support this path but do
not define production models. `Prepare` organizes raw observations and stage
labels; `Process` produces the 10-second cycle time series used downstream.

## Code map

- `src/frost_analysis/dataset/`: raw inputs, cycle construction, validation, and loaders.
- `src/frost_analysis/cost/`: shared energy integration and the selected v1/v2 models.
- `src/frost_analysis/labels/`: image cost-state labels and target construction.
- `src/frost_analysis/training/`: feature extraction, classifiers, and run state.
- `src/frost_analysis/exploration/`: correlation and model-selection calculations.
- `src/frost_analysis/figures/`: final publication plotting helpers.
- `scripts/`: the same workflow layout; `tests/extra/` holds exploratory and figure checks.
- `output/成本函数/`: published cost CSVs, cycle plots, and comparisons.
- `output/label/` and `output/model/`: formal RGB labels and model runs.
- `output/test/`: exploratory cost analyses, historical model experiments, and caches.
- `report/`: human-readable Markdown reports only.

Read the project in this order:

1. **Dataset:** start with `src/frost_analysis/__main__.py` and `src/frost_analysis/dataset/`.
2. **Economic defrost:** read `src/frost_analysis/cost/selected.py`, then `scripts/cost/`.
3. **Labels and training:** continue with `src/frost_analysis/labels/`,
   `src/frost_analysis/training/`, and their matching `scripts/` directories.
4. **Exploration:** inspect `scripts/exploration/` only when revisiting model selection.
5. **Figures:** finish with `scripts/figures/`.

Run commands from the repository root, for example:

```bash
uv run python scripts/exploration/analyze_raw_optimal_defrost.py --output output/test/成本函数/其他/经验经济窗口
uv run python scripts/cost/build.py --algorithm v1 v2 --output output/成本函数
uv run python scripts/cost/plot.py --cost v1=output/成本函数/cost_function_v1.csv v2=output/成本函数/cost_function_v2.csv --output output/成本函数
uv run python scripts/labels/build_rgb_cost_labels.py --cost-source output/成本函数/cost_function_v2.csv --output output/label/cost_function_v2_binary
uv run python scripts/training/extract_rgb_feature_shards.py \
  --labels output/label/cost_function_v2_binary/image_cost_labels.parquet \
  --output output/test/model/RGB特征缓存/cost_function_v2
uv run python scripts/training/evaluate_rgb_feature_shards.py \
  --shards output/test/model/RGB特征缓存/cost_function_v2/cycles \
  --candidates output/test/成本函数/其他/经验经济窗口/源数据/candidate_cost_curves.parquet \
  --label-balance output/label/cost_function_v2_binary/label_balance.csv \
  --labels output/label/cost_function_v2_binary/image_cost_labels.parquet \
  --task binary --jobs 6 --run-id 20260828_v2_binary \
  --output output/model/20260828_v2_binary
uv run pytest tests/cost/test_core.py
```

The exploration command refreshes the shared candidate curves; `cost/build.py`
writes the two comprehensive CSVs, and `cost/plot.py` renders the three
69-cycle PNG sets plus v1/v2/RB comparisons under `output/成本函数/`.
Existing completed RGB runs use historical v1 labels. Generate v2 labels and
independent v2 feature shards before starting a v2 run. The evaluator's
candidate parquet supplies only the shared cycle-time grid; targets and regret
come from the v2 labels/shards. Report documents remain Markdown-only.

## Dataset

Add experiment dates in order:

```bash
python -m frost_analysis dataset add data/0714 --dataset dataset
python -m frost_analysis dataset add data/0715 --dataset dataset
```

The Dataset CLI supports `add`, `remove`, `refresh`, `review-cycle`, `edit`,
`validate`, and `render`. It deliberately has no destructive rebuild command.
Use `python -m frost_analysis dataset --help` for arguments.

`prepare()` and `process()` are internal transformations used while building a
Dataset. They are not public commands and do not write standalone artifacts.

## Evidence

Analyze a schema v3 Dataset:

```bash
python -m frost_analysis evidence \
  --dataset dataset \
  --config configs/evidence.yaml \
  --output output/test/成本函数/其他/历史证据/frost_cycle_evidence_v2_3
```

Evidence reads cycles only through `DatasetLoader` and never writes inside the
Dataset. Dataset status is authoritative: human-reviewed `status` controls cycle
eligibility, while metric availability is recorded locally inside Evidence.

The current exploratory configuration includes `valid` cycles strictly longer
than 30 minutes. The Evidence eligibility table records this inclusion and its
reason. Use a frozen, stricter admission protocol before treating the future
100+ cycle run as confirmatory.

Each run writes nine auditable CSV tables and five Python/matplotlib figures in
editable SVG, PDF, PNG, and 600-dpi TIFF. The scientific gates and interpretation
rules are documented in
[`docs/evidence_analysis_framework_cn.md`](docs/evidence_analysis_framework_cn.md).

Configuration has one owner per workflow:

- `configs/config.yaml`: channels and shared Raw-to-Dataset rules.
- `configs/evidence.yaml`: Dataset-to-Evidence scientific analysis.

The formal data contracts are documented in
[`docs/pipeline_contract.md`](docs/pipeline_contract.md) and
[`docs/dataset_contract.md`](docs/dataset_contract.md).
