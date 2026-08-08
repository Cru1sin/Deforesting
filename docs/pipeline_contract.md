# Scientific Workflow Contract

## Scope

```text
Raw experiment directories
-> Prepare
-> Process
-> Cycle Dataset schema v3
-> DatasetLoader
-> Evidence
```

Prepare and Process are internal Dataset construction functions. The only public
CLI products are `dataset` and `evidence`.

## Prepare

`prepare(config, channels)` returns:

```python
prepared, cycle_summary
```

Prepare reads raw sensor files, maps source columns to canonical channels,
preserves source quality flags, labels cycle stages, and aligns image paths. It
does not resample, interpolate, calculate derived quantities, fit baselines, or
select Evidence features.

Prepared rows use observed source timestamps. Every source channel has explicit
`__missing`, `__invalid`, `__duplicate`, and `__conflict` flags. Cycle labels come
from the configured defrost state and operating-mode rules.

## Process

`process(prepared, cycle_summary, config, channels)` returns:

```python
processed, cycle_summary
```

Process applies one fixed order:

```text
10-second resampling
-> bounded within-stage missing handling
-> derived physical quantities
-> baseline and residuals
```

Filling never crosses experiment, cycle, stage, or operating-mode boundaries.
`<channel>__imputed` identifies reconstructed Processed values. Baselines and
residuals are calculated only by the configured baseline method. Dynamic trend,
lag, rolling, lead, and prediction features are Evidence-side analysis, not
Process outputs.

Prepared and Processed validators protect these scientific contracts while the
Dataset is built.

## Dataset

Dataset construction writes self-contained cycle files, Original observations,
image metadata, publication figures, a cycle catalog, a channel registry, and a
Dataset manifest. Downstream code must use `DatasetLoader`; it must not revisit
Raw inputs or configuration files.

Dataset status is the sole cycle eligibility decision. Review and edit operations
update Dataset-owned status or managed scientific fields, then refresh affected
assets and metadata.

See [`dataset_contract.md`](dataset_contract.md) for the persisted schema.

## Evidence

Evidence accepts one schema v3 `DatasetLoader` and `configs/evidence.yaml`.
Calculations use Dataset values as stored; Evidence does not interpolate,
resample, recompute baselines, or reinterpret cycle status.

Formal cycle metrics use only `cycle_stage == frost_development` and finite,
non-imputed observations. Future associations require exact within-stage time
anchors. Cross-cycle summaries first take a cycle median within each date, then
give dates equal weight.

Evidence writes formal CSV tables, publication figures, and an analysis manifest
outside the Dataset directory.
