# Evidence v2.3 Model Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable target, lead-time, and incremental Ridge readiness chain to DatasetLoader-native Evidence without changing the existing six scientific tables.

**Architecture:** Keep all new scientific calculations in one `readiness.py` module. Reuse DatasetLoader streaming, `observed_mask`, Theil-Sen, existing date-balanced conventions, bundle/output wiring, and the existing evidence CLI. Add only stable table contracts, flat settings, fixture data, and focused scientific tests.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, pytest, existing DatasetLoader and Evidence package.

---

### Task 1: Lock settings and table contracts

**Files:**
- Modify: `src/frost_analysis/evidence/settings.py`
- Modify: `src/frost_analysis/evidence/contracts.py`
- Modify: `configs/evidence.yaml`
- Modify: `tests/evidence/conftest.py`
- Modify: `tests/evidence/test_settings.py`
- Modify: `tests/evidence/test_core.py`

- [ ] **Step 1: Write failing settings and schema tests**

Assert that the new flat settings round-trip through `from_yaml()`,
`normalized()`, and `sha256`, and that `EvidenceBundle` contains exactly the
existing six plus `target_audit`, `readiness_split`, and `readiness_summary`.
Assert exact column lists from the approved design.

- [ ] **Step 2: Run RED tests**

Run:

```bash
python -m pytest -q -p no:cacheprovider tests/evidence/test_settings.py tests/evidence/test_core.py
```

Expected: failures for missing settings and readiness contracts.

- [ ] **Step 3: Add minimal immutable contracts**

Add flat dataclass fields with direct validation for positive durations,
positive Ridge alpha, primary threshold membership, and non-empty contexts.
Define the three column constants and extend the frozen bundle. Extend the test
fixture factory with production defaults while preserving its short synthetic
horizons.

- [ ] **Step 4: Run GREEN tests**

Run the same command and expect all selected tests to pass.

### Task 2: Implement target and signal audit by TDD

**Files:**
- Create: `src/frost_analysis/evidence/readiness.py`
- Create: `tests/evidence/test_readiness.py`

- [ ] **Step 1: Add target audit failure tests**

Construct canonical 10-second frost frames with raw target, baseline, residual,
quality, timestamp, and `defrost_start`. Test current degradation direction,
5/10/15 percent persistent events, exact 120-second persistence, interruption
by missing/imputed points, baseline unavailable/nonpositive/inconsistent
reasons, target observed fraction, exact 5/10/20-minute pair counts, and right
censoring.

- [ ] **Step 2: Run target tests and confirm RED**

```bash
python -m pytest -q -p no:cacheprovider tests/evidence/test_readiness.py -k target
```

- [ ] **Step 3: Implement target audit helpers**

Use one frost-stage selector, one strict observed-mask wrapper, one baseline
validator, one exact timestamp anchor builder, and one persistence detector.
Return rows in `TARGET_AUDIT_COLUMNS`; represent scientific absence in status
and reason fields rather than exceptions.

- [ ] **Step 4: Add signal onset failure tests**

Test increase/decrease alignment, reference-window exclusion, closed past-only
rolling median, 60-second persistence, missing-grid interruption, invalid MAD,
right-censored lead, and positive lead calculation.

- [ ] **Step 5: Run signal tests and confirm RED**

```bash
python -m pytest -q -p no:cacheprovider tests/evidence/test_readiness.py -k signal
```

- [ ] **Step 6: Implement signal onset and lead helpers**

Use residual values and existing `observed_mask`; do not add fallback quality
logic. Search starts after the five-minute reference. Return no numeric lead
unless both signal and primary event are observed.

- [ ] **Step 7: Run target and signal tests GREEN**

```bash
python -m pytest -q -p no:cacheprovider tests/evidence/test_readiness.py -k 'target or signal or lead'
```

### Task 3: Implement common-anchor Ridge comparison by TDD

**Files:**
- Modify: `src/frost_analysis/evidence/readiness.py`
- Modify: `tests/evidence/test_readiness.py`

- [ ] **Step 1: Add split/model failure tests**

Test exact same-cycle/same-stage horizons, anchor-level complete-case removal,
identical M0-M3 anchors, leave-one-date-out with per-cycle MAE, one-date
leave-one-cycle-out, no training-cycle degradation, train-only scaling, fixed
Ridge alpha, and M2-vs-M1/M3-vs-M2 skills. Include a synthetic useful signal
and a pure elapsed-time signal.

- [ ] **Step 2: Run model tests and confirm RED**

```bash
python -m pytest -q -p no:cacheprovider tests/evidence/test_readiness.py -k 'anchor or ridge or split or skill'
```

- [ ] **Step 3: Implement the smallest model pipeline**

Build complete M3 rows once per cycle/feature/target/horizon. Derive M0-M2 by
column selection. Fit means/stds and NumPy Ridge on training rows only. Produce
held-out-cycle MAEs independently even when cycles share a date split. Reject
only anchors with incomplete inputs; reject a split only after count/coverage
or training-cycle checks.

- [ ] **Step 4: Run model tests GREEN**

Run the same command and expect all selected tests to pass.

### Task 4: Implement date-balanced readiness summary

**Files:**
- Modify: `src/frost_analysis/evidence/readiness.py`
- Modify: `tests/evidence/test_readiness.py`

- [ ] **Step 1: Add summary/status tests**

Test cycle-to-date-to-overall medians, date-balanced positive fractions,
dynamic skill relative to M2, target-not-evaluable precedence, one-cycle
insufficient-validation precedence, state/no-increment/static/dynamic statuses,
and absence of magnitude thresholds.

- [ ] **Step 2: Run summary tests RED**

```bash
python -m pytest -q -p no:cacheprovider tests/evidence/test_readiness.py -k 'summary or status or date_balanced'
```

- [ ] **Step 3: Implement summary directly from formal rows**

Join trend evidence by feature, summarize independent units without pooling
anchors, and apply the six-state ordered decision contract exactly once.

- [ ] **Step 4: Run all readiness tests GREEN**

```bash
python -m pytest -q -p no:cacheprovider tests/evidence/test_readiness.py
```

### Task 5: Wire build, output, and manifest

**Files:**
- Modify: `src/frost_analysis/evidence/core.py`
- Modify: `src/frost_analysis/evidence/output.py`
- Modify: `tests/evidence/test_output.py`
- Modify: `tests/evidence/test_integration.py`

- [ ] **Step 1: Add failing integration tests**

Assert that one Evidence build streams valid cycles once, preserves the six
existing table schemas, returns all three readiness tables, writes nine CSVs,
records their row counts, leaves Dataset files unchanged, and succeeds with one
valid cycle while reporting `insufficient_validation_data`.

- [ ] **Step 2: Run integration tests RED**

```bash
python -m pytest -q -p no:cacheprovider tests/evidence/test_output.py tests/evidence/test_integration.py
```

- [ ] **Step 3: Add direct wiring only**

Collect valid cycle record/frame pairs during the existing stream, pass them to
the three readiness operations, extend the bundle constructor, and add the
three table mappings to `write_evidence()`. Do not add a service layer, cache,
new CLI command, or model dependency.

- [ ] **Step 4: Run Evidence tests GREEN**

```bash
python -m pytest -q -p no:cacheprovider tests/evidence
```

### Task 6: Verify the repository and real Dataset

**Files:**
- No new files unless a scientific defect requires a scoped correction.

- [ ] **Step 1: Run full automated verification**

```bash
python -m pytest -q -p no:cacheprovider
python -m ruff check --no-cache src tests
python -m compileall -q src
python -m mypy --strict src/frost_analysis/evidence src/frost_analysis/cli.py
git diff --check
```

Record pre-existing strict mypy diagnostics separately; do not modify unrelated
modules to clear repository typing debt.

- [ ] **Step 2: Run the real Dataset**

Use the schema v3 Dataset in the sibling Dataset review worktree and a fresh
temporary output directory. Confirm nine CSVs, existing figures, target/lead
rows, one-cycle `insufficient_validation_data`, no crash, and no Dataset writes.

- [ ] **Step 3: Review the final diff**

Confirm one new production module, no new dependency, no Dataset/upstream
changes, no duplicated quality policy, unchanged old table contracts, and no
unrequested ranking or model recommendation.

- [ ] **Step 4: Commit and push**

After all attributable failures are fixed:

```bash
git add docs/superpowers src/frost_analysis/evidence configs/evidence.yaml tests/evidence
git commit -m "feat(evidence): add model readiness audit"
git push origin codex/evidence-clean-v2-ideal
```
