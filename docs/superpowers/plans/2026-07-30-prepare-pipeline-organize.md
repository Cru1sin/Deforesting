# Prepare Pipeline Organize Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the prepare pipeline on `organize` into an explicit, timestamp-based, non-interpolating sensor/image preparation flow with per-date camera mapping and cycle coverage metrics.

**Architecture:** Keep `prepare.py` as a small orchestration layer. Move cycle validation into `data/cycles.py` and image path attachment into `data/alignment.py`; keep compatibility only at existing stage boundaries. Use explicit output columns, stable camera roles, and keyword-based result construction.

**Tech Stack:** Python 3.11, pandas, PyYAML, pytest, ruff, mypy, Parquet/CSV artifacts.

---

### Task 1: Establish explicit prepare and registry contracts

**Files:**
- Modify: `src/frost_analysis/pipelines/prepare.py`
- Modify: `src/frost_analysis/schemas.py`
- Modify: `src/frost_analysis/data/registry.py`
- Modify: `configs/0715.yaml`
- Modify: `configs/feature_registry.yaml`
- Test: `tests/pipelines/test_prepare.py`
- Test: `tests/test_registry_contract.py`

- [ ] **Step 1: Write failing contract tests**

Assert that registry application emits numeric `operating_mode` and nullable
boolean `is_heating`, and that `PrepareResult` is constructed with keyword
arguments and tuple warnings.

```python
def _mode_spec(raw_source: str) -> FeatureSpec:
    return FeatureSpec(
        feature_id="mode",
        canonical_name="operating_mode",
        raw_source=raw_source,
        meaning_zh="mode",
        physical_family="event_quality",
        source_type="event",
        unit="code",
        formula="",
        data_role="M",
        availability="current_history",
        deployment_status="confirmed",
        confidence="high",
        primary_or_validation="primary",
        analysis_enabled=False,
        notes="",
    )

def test_registry_exposes_numeric_mode_and_boolean_heating_flag() -> None:
    frame = pd.DataFrame({"mode_source": [3, 2, None]})
    result = apply_feature_registry(frame, {"mode": _mode_spec("mode_source")})
    assert result.frame["operating_mode"].tolist() == [3, 2, None]
    assert result.frame["is_heating"].tolist() == [True, False, None]
```

- [ ] **Step 2: Run the focused tests and verify the expected failure**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/pipelines/test_prepare.py tests/test_registry_contract.py
```

Expected: failure because the current registry emits `mode` and
`heating_mode`.

- [ ] **Step 3: Implement the minimum contract changes**

Rename orchestration locals to `file_inventory`, `source_field_inventory`,
`image_records`, `multiview_index`, `sensor_load_result`, `registry_specs`,
`registry_result`, and `prepared_data`. Rename the mode canonical field and
derived flag, then construct `PrepareResult` with keyword arguments.

- [ ] **Step 4: Run focused tests and the full suite**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
```

Expected: all tests pass after downstream references use the new mode names.

### Task 2: Add date-local camera mapping and explicit image policy

**Files:**
- Modify: `src/frost_analysis/config.py`
- Modify: `src/frost_analysis/schemas.py`
- Modify: `src/frost_analysis/data/images.py`
- Modify: `src/frost_analysis/pipelines/prepare.py`
- Modify: `configs/0715.yaml`
- Add: `configs/IPlocation.example.yaml`
- Test: `tests/data/test_images.py`
- Test: `tests/data/test_inventory.py`

- [ ] **Step 1: Write failing mapping and policy tests**

Test that `data/<date>/IPlocation.yaml` overrides the legacy mapping and that
`prepare.images.required` raises `RuntimeError("No RGB images were found")`
when the manifest is empty.

- [ ] **Step 2: Run the focused tests and verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/data/test_images.py tests/data/test_inventory.py
```

Expected: failure because the local loader and image requirement field do not
exist.

- [ ] **Step 3: Implement local mapping and image policy**

Load and validate the date-local YAML when present, fall back to the legacy
configured mapping otherwise, and add `images_required` to `PrepareOptions`.
Use that option to turn an empty manifest into a warning or an error.

- [ ] **Step 4: Verify mapping and policy**

Run the focused tests and the full suite. Confirm no IP-derived image columns
are created.

### Task 3: Make timestamp the prepare data contract

**Files:**
- Modify: `src/frost_analysis/data/cycles.py`
- Modify: `src/frost_analysis/data/alignment.py`
- Modify: `src/frost_analysis/pipelines/prepare.py`
- Modify: `tests/data/test_cycles.py`
- Modify: `tests/data/test_alignment.py`
- Modify: `tests/test_output_contract.py`

- [ ] **Step 1: Write failing timestamp tests**

Update fixtures to use `timestamp` and assert that segmentation, image
matching, and cycle summary accept and return that column.

- [ ] **Step 2: Run focused tests and verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/data/test_cycles.py tests/data/test_alignment.py tests/test_output_contract.py
```

Expected: failure because the current data functions require `sensor_time`.

- [ ] **Step 3: Migrate data functions and callers**

Use `timestamp` in cycle segmentation, cycle summary, sensor gap evidence,
image matching, and cycle-label attachment. Keep any legacy conversion only
inside `pipelines/process.py` where its internal algorithms still require it.

- [ ] **Step 4: Verify the migration**

Run focused and full tests, then confirm no prepare-side
`rename(columns={"timestamp": "sensor_time"})` remains.

### Task 4: Move cycle validation and calculate coverage metrics

**Files:**
- Modify: `src/frost_analysis/data/cycles.py`
- Modify: `src/frost_analysis/pipelines/prepare.py`
- Modify: `src/frost_analysis/schemas.py`
- Modify: `configs/0715.yaml`
- Test: `tests/data/test_cycles.py`
- Test: `tests/test_output_contract.py`

- [ ] **Step 1: Write failing cycle-validation tests**

Cover three-state status normalization, NaN-safe reason appending, configured
sampling fallback, and sensor/RGB/joint time-span fractions.

```python
def test_cycle_status_normalizes_quality_to_three_states() -> None:
    assert normalize_cycle_status("complete") == "valid"
    assert normalize_cycle_status("contaminated") == "invalid"
    assert normalize_cycle_status("excluded") == "invalid"
    assert normalize_cycle_status("partial") == "incomplete"
```

- [ ] **Step 2: Run focused tests and verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/data/test_cycles.py tests/test_output_contract.py
```

Expected: failure because the public validation interface and coverage fields
do not yet exist.

- [ ] **Step 3: Implement `validate_cycles()`**

Move heating-mode enforcement, long-gap marking, status normalization, and
sampling interval inference into `data/cycles.py`. Accept
`Mapping[str, object]`, use `expected_sampling_interval_seconds` only when a
positive median cannot be inferred, and return labeled rows, cycles, interval,
and warnings.

- [ ] **Step 4: Implement coverage summaries**

Compute observed non-NaN time span divided by cycle span, keep interruption
intervals and counts, and set joint coverage to the smaller of sensor and RGB
coverage.

- [ ] **Step 5: Verify cycle quality**

Run focused and full tests. Confirm the prepared table carries `cycle_status`
but not `cycle_status_reason`.

### Task 5: Move image wide-table attachment to alignment

**Files:**
- Modify: `src/frost_analysis/data/alignment.py`
- Modify: `src/frost_analysis/pipelines/prepare.py`
- Test: `tests/data/test_alignment.py`
- Test: `tests/pipelines/test_prepare.py`

- [ ] **Step 1: Write failing image-attachment tests**

Assert that missing required alignment columns raises `ValueError`, matched
images are grouped by `camera_role`, and output names are stable role names.

```python
def test_attach_image_paths_requires_alignment_contract() -> None:
    with pytest.raises(ValueError, match="camera_role"):
        attach_image_paths(prepared_frame(), pd.DataFrame({"matched": [True]}))
```

- [ ] **Step 2: Run focused tests and verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/data/test_alignment.py tests/pipelines/test_prepare.py
```

Expected: failure because the current function is private to `prepare.py` and
silently treats a missing `matched` column as all unmatched.

- [ ] **Step 3: Implement `attach_image_paths()`**

Validate all required columns, filter `matched` explicitly, deduplicate by
`timestamp` and `camera_role` using the smallest absolute delta, and merge
one stable path/offset pair per role.

- [ ] **Step 4: Verify image attachment**

Run focused and full tests. Confirm IP and device IDs never appear in output
column names.

### Task 6: Make output selection, state, validation, and publication explicit

**Files:**
- Modify: `src/frost_analysis/pipelines/prepare.py`
- Modify: `src/frost_analysis/config.py`
- Modify: `tests/pipelines/test_prepare.py`
- Add: `tests/pipelines/test_prepare_integration.py`

- [ ] **Step 1: Write failing output-safety tests**

Test interpolation rejection, explicit output selection, UTC ISO state time,
keyword-based `PrepareResult`, and atomic publication of both output files.

```python
def test_prepare_rejects_interpolation_artifacts() -> None:
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-07-15"]), "temperature__interpolated": [False]})
    config = make_test_config()
    with pytest.raises(RuntimeError, match="interpolated columns"):
        validate_prepare_result(frame, config)
```

Define `make_test_config()` in the test module with the existing `AppConfig`
fixture values and a temporary output directory; it is a test-only constructor,
not a production fallback.

- [ ] **Step 2: Run focused tests and verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/pipelines
```

Expected: failure because preparation currently drops interpolation markers
and writes state using a Unix timestamp.

- [ ] **Step 3: Implement explicit selection and validation**

Add `select_prepared_output_columns()`, `summarize_prepare_metrics()`,
`validate_prepare_result()`, and `publish_prepare_result()`. Return warning
tuples and preserve only documented cycle and image fields.

- [ ] **Step 4: Implement state fingerprints and atomic publication**

Write UTC ISO timestamps and fingerprints derived from config and registry file
bytes. Publish only after validation passes and retain the current path return
contract.

- [ ] **Step 5: Run complete verification**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check --no-cache src/frost_analysis tests
.venv/bin/python -m mypy --strict --cache-dir=/tmp/frost-analysis-mypy src/frost_analysis
```

Expected: all tests pass, ruff reports no errors, and mypy exits with code 0.

- [ ] **Step 6: Review the final diff**

```bash
git diff --check
git diff --stat
git status --short --branch
```

Confirm every changed line serves the prepare contract, timestamp migration,
cycle quality, image-role mapping, or output validation request.
