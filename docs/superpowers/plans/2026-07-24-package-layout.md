# Package Layout Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Organize the existing frost-analysis package by responsibility without changing the three CLI stages or scientific outputs.

**Architecture:** Keep the package root for public entry points and contracts. Move raw-data modules to `data/`, numerical transformations to `processing/`, the replaceable correlation task to `analysis/`, and artifact/contract helpers to `core/`. Update all imports and tests in one migration so no duplicate compatibility modules remain active.

**Tech Stack:** Python 3.11, pandas, PyYAML, pytest, Ruff, strict mypy.

---

### Task 1: Establish the target package layout

**Files:**
- Create: `src/frost_analysis/pipelines/__init__.py`
- Create: `src/frost_analysis/data/__init__.py`
- Create: `src/frost_analysis/processing/__init__.py`
- Create: `src/frost_analysis/analysis/__init__.py`
- Create: `src/frost_analysis/core/__init__.py`
- Modify: `.gitignore`

- [ ] Create the five package marker files with short responsibility docstrings.
- [ ] Add `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, and `.DS_Store` to `.gitignore` without deleting existing user files.
- [ ] Confirm the new packages import without importing business modules.

### Task 2: Move modules by responsibility

**Files:**
- Move: `prepare.py` -> `pipelines/prepare.py`
- Move: `process.py` -> `pipelines/process.py`
- Move: `inventory.py`, `sensors.py`, `images.py`, `registry.py`, `alignment.py`, `cycles.py` -> `data/`
- Move: `missing.py`, `baseline.py`, `resample.py`, `features.py` -> `processing/`
- Move: `correlation.py`, `screening.py` -> `analysis/`
- Move: `artifact_io.py` -> `core/artifacts.py`
- Move: `validation.py` -> `core/validation.py`

- [ ] Move each active module exactly once; do not leave root-level compatibility copies.
- [ ] Rename only `artifact_io.py` to `core/artifacts.py`; keep public function behavior unchanged.
- [ ] Keep `screening.py` because `analysis/correlation.py` imports and uses it.

### Task 3: Repair imports and public entry points

**Files:**
- Modify: `src/frost_analysis/cli.py`
- Modify: `src/frost_analysis/__main__.py`
- Modify: all moved modules under `src/frost_analysis/`
- Modify: `tests/conftest.py` and tests importing moved modules

- [ ] Enforce imports from `cli -> pipelines -> data/processing/analysis -> core`.
- [ ] Change `correlation.py` to import `screen_candidate_channels` from `.screening` within `analysis`.
- [ ] Change moved modules to import `artifacts`, `validation`, and sibling packages through their new absolute package paths.
- [ ] Preserve these commands exactly:
  `python -m frost_analysis prepare --config configs/0715.yaml`,
  `python -m frost_analysis process --config configs/0715.yaml`, and
  `python -m frost_analysis analyze --task correlation --config configs/0715.yaml`.

### Task 4: Mirror the package layout in tests

**Files:**
- Move sensor/image/cycle/inventory/alignment tests to `tests/data/`
- Move missing/baseline/resample/features tests to `tests/processing/`
- Move correlation/candidate tests to `tests/analysis/`
- Move prepare/process tests to `tests/pipelines/`
- Keep: `tests/test_output_contract.py`, `tests/conftest.py`, `tests/test_registry_contract.py`

- [ ] Move tests without changing assertions except import paths.
- [ ] Add package marker files only if pytest/package discovery requires them.
- [ ] Ensure no test imports `frost_analysis.prepare`, `frost_analysis.process`, or other old root-level modules.

### Task 5: Documentation and structural checks

**Files:**
- Modify: `README.md`
- Create: `tests/test_package_layout.py`

- [ ] Document the root/data/processing/analysis/core responsibilities and unchanged CLI.
- [ ] Add structural tests asserting moved modules exist, old root-level modules do not, and `screening.py` remains active under `analysis/`.
- [ ] Confirm `core` modules do not import `pipelines`, `data`, `processing`, or `analysis`.

### Task 6: Verify the migration with real data

- [ ] Run `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`.
- [ ] Run `.venv/bin/ruff check --no-cache src/frost_analysis tests`.
- [ ] Run `.venv/bin/python -m mypy --strict --cache-dir=/tmp/frost-analysis-mypy src/frost_analysis`.
- [ ] Run `prepare`, `process`, and `analyze --task correlation` for 0715.
- [ ] Verify `validate_stage_outputs` remains empty and the four user-visible artifacts are unchanged in name and schema.
