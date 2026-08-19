# Frost Sensor Analysis

The repository has one scientific workflow:

```text
Raw
-> Prepare
-> Process
-> Cycle Dataset
-> Evidence
```

`Prepare` only organizes raw observations, source quality information, and
cycle/stage labels. `Process` only produces the 10-second scientific time
series: resampled values, bounded missing handling, derived physical
quantities, and baseline residuals.

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
  --output outputs/evidence/frost_cycle_evidence_v2_3
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
