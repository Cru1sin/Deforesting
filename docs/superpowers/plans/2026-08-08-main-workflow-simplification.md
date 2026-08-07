# Main Workflow Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the obsolete Run, stage, Report, and legacy Evidence runtime so the repository exposes only Raw -> Dataset -> DatasetLoader -> Evidence without changing scientific Dataset values or Evidence numerical outputs.

**Architecture:** Keep `prepare()` and `process()` as direct internal Dataset transformations. Keep source discovery in `io.py`, Dataset publication rendering in `visualization.py`, Dataset-native analysis in `evidence/`, and remove every compatibility or provenance closure with no remaining consumer.

**Tech Stack:** Python 3.11, pandas, matplotlib, pytest, ruff, mypy.

---

### Task 1: Lock And Reduce The Public CLI

**Files:**
- Modify: `tests/evidence/test_cli.py`
- Modify: `src/frost_analysis/cli.py`
- Modify: `src/frost_analysis/__init__.py`
- Delete: `src/frost_analysis/pipeline.py`
- Delete: `src/frost_analysis/analysis.py`
- Delete: `tests/analysis/test_flat_analysis_contract.py`
- Delete: `tests/test_pipeline_contract.py`

- [ ] Add a parser test that obtains the top-level argparse choices and asserts exactly `{"dataset", "evidence"}`.
- [ ] Run `python -m pytest -q -p no:cacheprovider tests/evidence/test_cli.py` and confirm it fails because old commands remain.
- [ ] Reduce `cli.py` to direct Dataset and Evidence dispatch, and reduce `__init__.py` to package metadata only.
- [ ] Delete the unreachable pipeline and old analysis implementation with their tests.
- [ ] Run `python -m pytest -q -p no:cacheprovider tests/evidence/test_cli.py tests/evidence` and confirm the active CLI and Evidence suite pass.
- [ ] Commit only this batch.

### Task 2: Delete Run I/O And Prepare Provenance

**Files:**
- Modify: `src/frost_analysis/io.py`
- Modify: `src/frost_analysis/prepare.py`
- Modify: `src/frost_analysis/dataset.py`
- Modify: `tests/test_io_contract.py`
- Modify: `tests/test_prepare_contract.py`

- [ ] Reduce `tests/test_io_contract.py` to one source-discovery contract and change surviving Prepare tests to unpack exactly two return values.
- [ ] Run the focused tests and confirm they fail against the old three-value `prepare()` and broad I/O module.
- [ ] Keep only `InputFiles` and `discover_inputs()` in `io.py`.
- [ ] Change `prepare()` to return `(prepared, cycle_summary)` and delete `prepare_summary`, inventory hashes, source hashes, config hashes, git metadata, camera summaries, unavailable-channel summaries, and helpers used only to produce them.
- [ ] Update Dataset construction to consume the two-value result.
- [ ] Run `python -m pytest -q -p no:cacheprovider tests/test_io_contract.py tests/test_prepare_contract.py tests/test_dataset_final_contract.py`.
- [ ] Commit only this batch.

### Task 3: Replace Dataset's Report Dependency And Delete Report

**Files:**
- Modify: `src/frost_analysis/visualization.py`
- Modify: `tests/test_dataset_final_contract.py`
- Delete: `src/frost_analysis/report.py`
- Delete: `tests/test_report.py`

- [ ] Add or retain one Dataset publication test that calls `render_cycle_publication()` and checks the required target/stage content is produced without importing `report`.
- [ ] Run the focused test and confirm it fails while `visualization.py` imports Report internals.
- [ ] Implement the smallest direct publication renderer in `visualization.py`, reusing existing matplotlib helpers only where they express required figure meaning.
- [ ] Delete `report.py` and its QA tests; do not move Report manifests, warning ledgers, QA panels, or publish logic.
- [ ] Run `python -m pytest -q -p no:cacheprovider tests/test_dataset_final_contract.py`.
- [ ] Commit only this batch.

### Task 4: Delete Legacy Evidence Configuration And Registry Metadata

**Files:**
- Modify: `src/frost_analysis/config.py`
- Modify: `configs/defaults.yaml`
- Modify: `src/frost_analysis/dataset_schema.py`
- Modify: `tests/test_config_contract.py`
- Modify: `tests/test_dataset_schema.py`
- Delete: `src/frost_analysis/evidence_cycle.py`

- [ ] Remove tests for `AnalysisSettings`, `EvidencePolicy`, resolved-config hashes, and registry `analysis_settings`; keep tests for active Dataset construction settings.
- [ ] Run the focused config and schema tests and confirm expected failures before production deletion.
- [ ] Remove the old analysis/evidence policy classes, loaders, timing adapter, and resolved provenance mapping/hash from `config.py`.
- [ ] Remove `analysis:` from `configs/defaults.yaml` and remove `analysis_settings` from registry construction and merge logic.
- [ ] Delete `evidence_cycle.py` without replacement or re-export.
- [ ] Run `python -m pytest -q -p no:cacheprovider tests/test_config_contract.py tests/test_dataset_schema.py tests/evidence`.
- [ ] Commit only this batch.

### Task 5: Remove Residual Vocabulary, Tests, And Dependencies

**Files:**
- Modify: `src/frost_analysis/validation.py`
- Modify: `README.md`
- Modify: `docs/pipeline_contract.md`
- Modify: `pyproject.toml`
- Modify or delete: tests that only protect removed runtime behavior

- [ ] Search `src`, `tests`, `configs`, `README.md`, and active docs for Run artifacts, QA Report, stage writers, old Evidence, and compatibility APIs.
- [ ] Delete only code and tests whose last consumer was removed; retain Prepared and Processed scientific validators.
- [ ] Remove `seaborn` and `tabulate` only if repository-wide import search finds no active use.
- [ ] Rewrite workflow documentation around `dataset` and `evidence` only.
- [ ] Run the complete verification suite:

```bash
python -m pytest -q -p no:cacheprovider
python -m ruff check --no-cache src tests
python -m compileall -q src
git diff --check
```

- [ ] Run strict mypy on surviving CLI, Dataset, and Evidence modules and confirm no new diagnostics.
- [ ] Compare retained Dataset/Evidence contract tests and inspect the final diff for accidental scientific changes.
- [ ] Commit and push `main`, leaving `docs/frost_defrost_optimal_timing_analysis_cn.md` untracked and untouched.

