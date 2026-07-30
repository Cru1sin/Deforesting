# Candidate Registry Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the first-stage pipeline limited to auditable candidate-channel validation and correct its remaining registry, time-resolution, and artifact-provenance defects.

**Architecture:** The Unified Feature Registry remains the single scientific contract. `X` response channels are audited as candidates, `C` channels are used only as conditioning context, and `M` channels are used for segmentation, masks, and reset evaluation. The pipeline will publish one clean output tree and will not train `z_t`, a neural network, or a defrost policy.

**Tech Stack:** Python 3.11, pandas, PyYAML, pytest, Parquet/CSV fallback.

---

### Task 1: Correct registry semantics and pressure/performance contracts

**Files:**
- Modify: `configs/feature_registry.yaml`
- Modify: `src/frost_analysis/registry.py`
- Modify: `src/frost_analysis/reporting.py`
- Modify: `src/frost_analysis/validation.py`
- Modify: `tests/test_registry_contract.py`

- [ ] Add `CCQ_Comp` as a validation performance-response channel mapped to the source field, while keeping `QComp10W` as the primary capacity channel.
- [ ] Preserve event fields as string-capable registry channels and ensure coverage counts use non-empty values for both numeric and event data.
- [ ] Document and test `PR = Pc_abs / Pe_abs` with `Pr ~= 100 * PR`; do not describe Pc/Pe as gauge pressure unless the consistency check disproves the absolute-pressure interpretation.
- [ ] Update reports and output validation so CCQ_Comp is allowed as a registry validation field and cannot become a duplicate primary model channel.

### Task 2: Make analysis resolution honest and auditable

**Files:**
- Modify: `src/frost_analysis/features.py`
- Modify: `src/frost_analysis/pipeline.py`
- Modify: `tests/test_features.py`
- Modify: `tests/test_output_contract.py`

- [ ] Aggregate numeric registry channels into fixed per-cycle 10-second bins using the latest observation within each bin and retain source counts, expected counts, coverage, and bin end time.
- [ ] Carry event/state and context values without averaging; keep baseline provenance and cycle labels attached to each bin.
- [ ] Mark bins unavailable when their source gap exceeds the configured coverage requirement so candidate screening cannot treat missing intervals as observed.
- [ ] Keep the output contract to the three active analysis CSVs plus the documented processed/report/log artifacts.

### Task 3: Remove staging-path leakage and document scope boundaries

**Files:**
- Modify: `src/frost_analysis/reporting.py`
- Modify: `src/frost_analysis/pipeline.py`
- Modify: `README.md`
- Modify: `tests/test_output_contract.py`

- [ ] Ensure every manifest artifact path is relative to the published date directory or uses the final output path; no temporary staging directory may remain.
- [ ] State explicitly that parameter-table-5 backup/mirror/calibration fields remain only in raw processed provenance and are ignored by the Registry and candidate evidence.
- [ ] Record candidate-validation-only status and skipped future-target analysis in the manifest and report.

### Task 4: Run Builder verification and independent Critic/Evaluator review

**Files:**
- Create: `review/critic_round_20260723.md`
- Create: `review/evaluator_round_20260723.md`

- [ ] Run the full test suite and the dated pipeline for `0715`.
- [ ] Check that the registry, candidate evidence, cycle summary, manifest, reports, and processed artifacts agree.
- [ ] Have an independent critic review requirements, logic, edge cases, code quality, test coverage, and runtime evidence; fix every actionable issue.
- [ ] Have an evaluator decide whether the output is candidate-validation-only and whether the evidence supports moving to cross-date second-stage analysis.

