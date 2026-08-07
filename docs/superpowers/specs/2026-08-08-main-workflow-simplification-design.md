# Main Workflow Simplification Design

## Goal

Reduce `main` to one readable scientific workflow without changing current
Dataset or Dataset-native Evidence outputs:

```text
Raw experiment directories
-> Dataset build and review
-> DatasetLoader
-> Evidence tables and figures
```

The public CLI contains only `dataset` and `evidence`.

This is a structural cleanup. Dataset files, Evidence CSV schemas, numerical
definitions, status semantics, figures, and manifests remain unchanged.

## Ponytail Rules

- Delete before moving, moving before rewriting.
- Keep one implementation for each behavior.
- Do not add adapters, compatibility commands, service layers, factories,
  protocols, feature flags, or speculative extension points.
- Keep validation only where it protects scientific meaning, user data, or a
  persisted public contract.
- Prefer direct names that state the scientific object being calculated.
- A helper exists only when it removes real duplication or names a nontrivial
  scientific definition.
- A test exists only when it protects a public operation, persisted output, or
  scientific calculation.

## Final Public Workflow

### Dataset

```bash
python -m frost_analysis dataset rebuild data/0714 data/0715
python -m frost_analysis dataset add data/0716
python -m frost_analysis dataset validate --dataset dataset
python -m frost_analysis dataset review-cycle frost_cycle_000001 --status valid
python -m frost_analysis dataset edit --dataset dataset --baseline-seconds 60
python -m frost_analysis dataset render --dataset dataset frost_cycle_000001
```

`prepare()` and `process()` remain internal Dataset construction functions.
They are not CLI stages and do not write standalone run artifacts.

### Evidence

```bash
python -m frost_analysis evidence \
  --dataset dataset \
  --config configs/evidence.yaml \
  --output outputs/evidence/frost_cycle_evidence_v2_3
```

Evidence reads only through `DatasetLoader` and writes only outside Dataset.

## Removed Runtime

Delete the complete obsolete runtime closure:

- `src/frost_analysis/pipeline.py`
- `src/frost_analysis/analysis.py`
- `src/frost_analysis/evidence_cycle.py`
- old run-level and QA-only code in `src/frost_analysis/report.py`
- public `run`, `prepare`, `process`, and `report` CLI commands
- old run/stage/evidence writers and loaders in `src/frost_analysis/io.py`
- old analysis validation in `src/frost_analysis/validation.py`
- old package exports from `src/frost_analysis/__init__.py`
- documentation describing Run, stage artifacts, old Analyze, or QA Report

No deprecation aliases remain. Removed commands fail through normal argparse
unknown-command behavior.

## Active Module Responsibilities

### CLI

`cli.py` only parses `dataset` and `evidence`, creates their direct inputs, calls
one function, and prints the returned output path. Dataset subcommands keep their
current names and arguments.

### Dataset construction

`dataset.py` owns Dataset operation orchestration. It calls existing Raw parsing,
cycle labeling, Process, Dataset writing, and Dataset rendering functions.

The implementation may be split or merged only when the result gives each file
one concrete responsibility and reduces total code. File count is not a goal.

### Source I/O

The surviving part of `io.py` contains only source discovery, source metadata,
inventory hashes required by current Dataset output, and output-path checks used
by Dataset construction. Rename it to `source_io.py` only if the final contents
are exclusively source-facing; otherwise keep `io.py`. Do not create wrappers
around moved functions.

### Configuration

`config.py` keeps only values consumed by Dataset construction or serialized into
the current Dataset registry/manifest. Old Evidence runtime loaders, policy
wrappers, timing adapters, ranking thresholds, and decision-only validation are
removed.

If an old analysis mapping must remain to preserve current registry bytes, keep it
as a plain normalized mapping instead of retaining an unused runtime class graph.
The serialized keys and values must not change in this cleanup.

### Validation

`validation.py` keeps Prepared and Processed scientific contracts used while
building Dataset. Dataset publication validation remains in
`dataset_validation.py`. Evidence availability stays local to Evidence tables.

### Visualization

Dataset currently reaches its publication renderer through private functions in
the obsolete `report.py`. Move only the transitive function and constant closure
needed by `render_cycle_publication()` into `visualization.py` or one focused
Dataset figure module. Delete QA overview, run report, warning ledger, report
manifest, baseline report, candidate report, and publish-report code.

Publication PNG and RGB coverage output remain visually and structurally
unchanged.

### Evidence

Keep the public API:

```python
EvidenceSettings.from_yaml(path)
build_evidence(loader, settings)
write_evidence(bundle, output_dir, loader=loader, settings=settings)
```

Simplify the active package without changing its calculations:

- one finite and non-imputed observation mask;
- one date-balanced aggregation implementation;
- one row constructor per formal table where it reduces repeated status fields;
- no alternate loaders, fallbacks, dynamic schemas, or compatibility paths;
- descriptive names for time anchors, observed values, target changes, held-out
  cycles, and date-level effects;
- no ranking, sensor decision, recovery conclusion, or new statistical method.

`readiness.py` may be reorganized into fewer or smaller units only when total code
and cross-file coupling decrease. Moving code without simplification is rejected.

## Dependency Cleanup

After obsolete modules are removed, search all imports and delete dependencies
with no active caller. Expected candidates are `seaborn` and `tabulate`; remove
them only when repository search and the full test suite confirm they are unused.

No new dependency is allowed.

## Test Cleanup

Delete tests for removed behavior:

- all old analysis tests;
- all run pipeline tests;
- all QA/run Report tests;
- stage/run writer and manifest tests;
- compatibility-command tests;
- tests whose only assertion is a private helper's internal shape.

Reduce surviving tests by these rules:

1. Keep one CLI test proving only `dataset` and `evidence` are accepted.
2. Keep Dataset tests for each user operation and each persisted Dataset contract.
3. Keep Prepared/Processed tests only for scientific transformations used by
   Dataset construction.
4. Keep Evidence tests for cohort authority, observation masks, exact future
   anchors, event timing, model comparison, date-balanced aggregation, formal
   table columns, output location, and figure data sources.
5. Delete duplicate tests that exercise the same behavior with different private
   call paths.
6. Delete defensive tests for impossible states that valid Dataset construction
   cannot produce, unless the state crosses a user-controlled file boundary.
7. Keep plotting tests about required panels and source data; delete tests tied to
   artist counts, private layout coordinates, or incidental implementation.

The target is not maximum coverage. The target is the smallest suite that would
catch a changed scientific result or broken public workflow.

## Behavior Lock

Before deleting production code, add the smallest failing CLI contract test:

```text
top-level commands == {dataset, evidence}
```

Existing Dataset and Evidence fixtures are the output lock. During cleanup:

- Dataset manifest, catalog, registry, cycle frames, Original frames, and image
  metadata retain their schemas and values;
- Evidence formal tables retain columns, row identities, statuses, and numbers;
- publication and coverage figures retain their current required content;
- no test baseline is updated merely to accept changed output.

## Execution Order

1. Lock the two-command CLI and remove old public entry points.
2. Remove old pipeline, analysis, run I/O, analysis validation, and their tests.
3. isolate Dataset publication rendering and delete the remaining Report closure.
4. Remove old Evidence runtime and simplify config while preserving serialized
   Dataset settings.
5. Simplify active Dataset, source I/O, validation, and Evidence implementations
   in small behavior-preserving changes.
6. Remove unused dependencies and rewrite README/contracts around the sole flow.
7. Run the full verification suite and inspect the final diff for code that only
   exists to support deleted behavior.

## Verification

Every batch must pass:

```bash
python -m pytest -q -p no:cacheprovider
python -m ruff check --no-cache src tests
python -m compileall -q src
git diff --check
```

Strict mypy diagnostics may decrease as dead code disappears. No new diagnostic
is accepted in surviving Dataset, Evidence, or CLI code.

## Separate Scientific Redesign Deliverable

This cleanup does not change Evidence science. The final implementation report
will include a separate, non-implemented proposal for a future schema centered
on four questions:

1. when the performance event occurs;
2. which signals lead it consistently;
3. whether a signal adds information beyond elapsed time and performance history;
4. whether current data justify the next algorithm stage.

That proposal must identify tables and statistics to delete, retain, rename, or
replace, but it is not part of this behavior-preserving refactor.
