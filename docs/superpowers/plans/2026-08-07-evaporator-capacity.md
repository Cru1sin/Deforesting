# Evaporator Capacity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add P5 compressor power and calculate evaporator capacity as heating capacity minus compressor power.

**Architecture:** Reuse the existing source-channel normalization and named derived-formula dispatch. No Dataset-specific code or new abstraction is needed.

**Tech Stack:** Python, pandas, YAML, pytest.

---

### Task 1: Add the two scientific channels

**Files:**
- Modify: `configs/channels.yaml`
- Modify: `src/frost_analysis/channels.py`
- Modify: `src/frost_analysis/features.py`
- Test: `tests/test_process_contract.py`

- [x] **Step 1: Write the failing formula test**

Add a derived channel with dependencies `heating_capacity` and `compressor_power`; assert `10.0 - 2.428 == 7.572` and a missing compressor value yields null.

- [x] **Step 2: Verify the test fails**

Run `python -m pytest -q -p no:cacheprovider tests/test_process_contract.py -k evaporator_capacity` and expect rejection of the unsupported formula.

- [x] **Step 3: Implement the minimum formula and channel configuration**

Map `p5__压机功率` to `compressor_power` with `scale: 0.001`, define derived `evaporator_capacity`, add its formula name to the two existing allowlists, and return `values[0] - values[1]` in the existing dispatcher.

- [x] **Step 4: Verify focused and full tests**

Run the focused test, then the complete pytest, Ruff, and strict mypy commands.

- [x] **Step 5: Commit and push**

Commit only the plan, test, YAML, and two formula files, then push `codex/dataset-review-fixes`.
