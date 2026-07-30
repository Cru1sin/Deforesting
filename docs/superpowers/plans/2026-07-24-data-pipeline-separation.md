# Data Pipeline Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate raw-data preparation, reusable data processing, and replaceable analysis tasks while preserving the candidate-channel scientific boundaries.

**Architecture:** `prepare_dataset` creates one minimally transformed `prepared_data.parquet` plus the canonical `cycle_summary.csv`. `process_dataset` reads only those artifacts and creates one `processed_data.parquet` with missing-data handling, baselines, resampling, and features. Analysis tasks read `processed_data.parquet` and `cycle_summary.csv`; the first task is correlation/candidate evidence and can later be replaced by forecasting or multimodal training.

**Tech Stack:** Python 3.11, pandas, PyArrow, PyYAML, SciPy, pytest, Ruff, strict mypy.

---

### Task 1: Define the stage contracts and configuration

**Files:**
- Create: `configs/0715.yaml`
- Create: `src/frost_analysis/schemas.py`
- Modify: `src/frost_analysis/config.py`
- Modify: `tests/test_output_contract.py`

- [ ] **Step 1: Add failing contract tests** for `prepared_data.parquet`, `processed_data.parquet`, `cycle_summary.csv`, and `correlation_results.csv`, including the rule that prepared data contains no baseline/rolling columns.
- [ ] **Step 2: Add typed stage path/config objects** with strict rejection of unknown YAML sections and explicit `prepare`, `process`, and `analysis` sections.
- [ ] **Step 3: Define canonical cycle fields**: `cycle_id`, `cycle_stage`, `cycle_status`, and `cycle_status_reason`; keep compatibility aliases only at read boundaries.
- [ ] **Step 4: Run the focused contract tests and verify they fail before stage implementations exist.**

### Task 2: Implement `prepare_dataset`

**Files:**
- Create: `src/frost_analysis/prepare.py`
- Modify: `src/frost_analysis/sensors.py`
- Modify: `src/frost_analysis/images.py`
- Modify: `src/frost_analysis/cycles.py`
- Create: `tests/test_prepare.py`

- [ ] **Step 1: Test that preparation preserves raw timestamps and missing values** and does not emit rolling, baseline, interpolated, or resampled columns.
- [ ] **Step 2: Add `standardize_schema` and `clean_timestamps`** around the existing Registry mapping; deterministic unit conversions are allowed, but interpolation is disabled.
- [ ] **Step 3: Add image-to-sensor attachment** that stores per-camera image path and offset columns in the prepared frame; retain image/multiview computation in memory only.
- [ ] **Step 4: Add cycle preparation quality fields** and merge sensor/RGB coverage and interruption intervals into the single cycle summary.
- [ ] **Step 5: Implement atomic `prepare` publication** with only `prepared_data.parquet` and `cycle_summary.csv` visible in the stage output; detailed provenance goes under `.pipeline/`.

### Task 3: Implement `process_dataset`

**Files:**
- Create: `src/frost_analysis/process.py`
- Create: `src/frost_analysis/missing.py`
- Create: `src/frost_analysis/resample.py`
- Modify: `src/frost_analysis/baseline.py`
- Modify: `src/frost_analysis/features.py`
- Create: `tests/test_process.py`

- [ ] **Step 1: Test that process reads only prepared artifacts** and rejects missing or incompatible prepared schema.
- [ ] **Step 2: Implement role-aware missing handling** with no cross-cycle or cross-stage fill, bounded continuous interpolation only when configured, and images/targets left missing by default.
- [ ] **Step 3: Reuse the existing clean-baseline selector** and write baseline columns into `processed_data.parquet`; update the same cycle summary rather than creating another summary.
- [ ] **Step 4: Implement role-aware resampling** with configurable aggregations for continuous, control, event, and image-path columns.
- [ ] **Step 5: Reuse feature engineering with explicit long names** and publish only `processed_data.parquet` plus the updated cycle summary.

### Task 4: Implement replaceable analysis tasks

**Files:**
- Create: `src/frost_analysis/correlation.py`
- Modify: `src/frost_analysis/screening.py`
- Create: `tests/test_correlation.py`

- [ ] **Step 1: Test cycle-level Spearman/Pearson and lag summaries** without treating raw timestamps as independent experiments.
- [ ] **Step 2: Move candidate evidence orchestration into `run_correlation_analysis`**, reading only `processed_data.parquet` and `cycle_summary.csv`.
- [ ] **Step 3: Remove analysis-only screening, plotting, and report generation from prepare/process; make figure output opt-in.**
- [ ] **Step 4: Emit one human-readable `correlation_results.csv` and structured task metadata.**

### Task 5: Replace the CLI and retire the monolith

**Files:**
- Create: `src/frost_analysis/cli.py`
- Modify: `src/frost_analysis/__main__.py`
- Modify: `src/frost_analysis/artifact_io.py`
- Modify: `src/frost_analysis/validation.py`
- Modify: `README.md`
- Create: `tests/test_cli.py`
- Archive: `src/frost_analysis/pipeline.py` and superseded activity outputs under `archive/feature_registry_transition_20260724/`

- [ ] **Step 1: Add `prepare`, `process`, and `analyze --task correlation` commands** with separate stage paths and no implicit rerun of earlier stages.
- [ ] **Step 2: Add stage-specific manifest/state under `outputs/<date>/.pipeline/`** while keeping the user-visible output to the four main artifacts.
- [ ] **Step 3: Update validation and README to the new contract.**
- [ ] **Step 4: Move the old monolithic entry and obsolete output trees to the dated archive; do not touch `.DS_Store`.**

### Task 6: End-to-end verification

**Files:**
- Modify: `tests/test_*` as required by failing contract tests.

- [ ] **Step 1: Run `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`.**
- [ ] **Step 2: Run Ruff, strict mypy, and compileall.**
- [ ] **Step 3: Run `prepare`, `process`, and `analyze --task correlation` for 0715.**
- [ ] **Step 4: Verify exact visible outputs, cycle/RGB quality numbers, no raw-input reread in process/analyze, and no final model or defrost policy training.**
- [ ] **Step 5: Run one Critic review and one final Evaluator review; repeat Critic only for a high-severity defect.**
