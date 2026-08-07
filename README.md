# Frost Sensor Analysis

The repository has one scientific workflow:

```text
Raw experiment directories
-> Dataset build / review / edit
-> Self-contained Cycle Dataset
-> DatasetLoader
-> Dataset-native Evidence
-> Evidence tables and figures
```

## Dataset

Build all experiment dates:

```bash
python -m frost_analysis dataset rebuild \
  data/0714 data/0715 data/0716 data/0717 data/0720 data/0721 data/0722 \
  --dataset dataset
```

The Dataset CLI also supports `add`, `validate`, `refresh`, `review-cycle`,
`edit`, and `render`. Use `python -m frost_analysis dataset --help` for their
arguments.

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
Dataset. Dataset status is authoritative: valid cycles enter analysis; metric
availability is recorded locally in the corresponding Evidence table.

Configuration has one owner per workflow:

- `configs/defaults.yaml`: Raw-to-Dataset scientific transformations.
- `configs/evidence.yaml`: Dataset-to-Evidence scientific analysis.
- `configs/channels.yaml`: channel names, units, roles, formulas, and quality rules.

The formal data contracts are documented in
[`docs/pipeline_contract.md`](docs/pipeline_contract.md) and
[`docs/dataset_contract.md`](docs/dataset_contract.md).
