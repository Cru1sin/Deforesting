# Cost-to-RGB Paper Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete a reproducible demo from raw-cycle empirical defrost cost to cycle-safe RGB classification and a claim-bounded Nature-style manuscript draft.

**Architecture:** Keep the existing raw-cost and image-label modules unchanged unless a failing check exposes a defect. Use the existing fixed experiment split and pointwise 1% regret labels, select one model family on validation balanced accuracy, then compare the required camera groups without consulting test metrics. Store source tables beside publication figures and write only claims supported by the locked test set.

**Tech Stack:** Python, pandas, NumPy, scikit-learn, PyTorch/torchvision, matplotlib, pytest, Git.

---

### Task 1: Finish and audit the five-model smoke test

**Files:**
- Modify only if a check fails: `src/frost_analysis/rgb_smoke.py`
- Modify only if a check fails: `scripts/train_rgb_smoke_models.py`
- Test: `tests/test_rgb_smoke.py`
- Generate: `report/rgb_model_smoke/three_all/`

- [ ] **Step 1: Let the active three-class run finish**

Run the already active command and require `metrics.csv`, `confusion_matrices.csv`, `predictions.parquet`, `sample_manifest.parquet`, and `excluded_images.csv`.

- [ ] **Step 2: Verify common-sample fairness and split isolation**

```python
manifest = pd.read_parquet("report/rgb_model_smoke/three_all/sample_manifest.parquet")
assert not manifest.groupby("cycle_name")["split"].nunique().gt(1).any()
assert set(manifest["split"]) == {"train", "validation", "test"}
```

- [ ] **Step 3: Run the minimal unit checks**

Run: `.venv/bin/pytest -q tests/test_rgb_smoke.py tests/test_rgb_cost_labels.py tests/test_defrost_cost.py`

Expected: all selected tests pass.

- [ ] **Step 4: Commit the reusable smoke-test implementation**

Run: `git add pyproject.toml uv.lock src/frost_analysis/rgb_smoke.py scripts/train_rgb_smoke_models.py tests/test_rgb_smoke.py report/rgb_model_smoke/three_all && git commit -m "feat: compare cycle-safe RGB baselines" && git push`

### Task 2: Lock the task contract and select one model family

**Files:**
- Generate: `report/rgb_model_selection/model_selection.csv`
- Generate: `report/rgb_model_selection/README.md`

- [ ] **Step 1: Compare three-class and high-confidence binary tasks**

Run: `.venv/bin/python scripts/train_rgb_smoke_models.py --task binary --camera-group all --maximum-per-group 12 --output report/rgb_model_smoke/binary_all`

- [ ] **Step 2: Select using validation only**

Rank model-task pairs by validation balanced accuracy, break ties by validation macro-F1, and record the selected family before reading its test result. Do not tune on the test split.

- [ ] **Step 3: Reject a task contract that lacks cycle coverage**

Require every retained class in validation and test and report class-specific cycle counts. If a class is supported by only one validation cycle, label the run an engineering smoke test rather than model evidence.

- [ ] **Step 4: Commit the locked selection record**

Run: `git add -f report/rgb_model_smoke/binary_all report/rgb_model_selection && git commit -m "analysis: lock RGB task and baseline" && git push`

### Task 3: Compare required camera groups with the locked winner

**Files:**
- Modify: `scripts/train_rgb_smoke_models.py`
- Test: `tests/test_rgb_smoke.py`
- Generate: `report/rgb_camera_comparison/`

- [ ] **Step 1: Add one model selector instead of a new training framework**

Expose a `--models` argument whose default remains all five names and reuse the existing estimator/training branches. The camera comparison calls only the locked winner.

- [ ] **Step 2: Run required groups**

Run the locked task and model for `top`, `top_close`, `left`, `left_close`, `front`, `extreme`, `top_pair`, `left_pair`, and `all`, retaining the same experiment split and sample budget.

- [ ] **Step 3: Choose camera configuration using validation only**

Rank camera groups by validation balanced accuracy and macro-F1; reveal the chosen group's locked test metrics once.

- [ ] **Step 4: Commit camera evidence**

Run: `git add scripts/train_rgb_smoke_models.py tests/test_rgb_smoke.py report/rgb_camera_comparison && git commit -m "analysis: compare frost camera groups" && git push`

### Task 4: Complete missing cycles without exceeding local storage

**Files:**
- Reuse: `src/frost_analysis/dataset_images.py`
- Generate: `report/rgb_full_cohort/`

- [ ] **Step 1: Materialize exactly one required cycle**

Call `materialize_cycle_images` for the next missing labelled cycle, validate image decode and metadata alignment, extract only the locked model's features or predictions, then remove only the local temporary copy in its existing `finally` path.

- [ ] **Step 2: Persist cycle-level outputs before the next download**

Write one manifest/prediction shard per cycle so an interrupted download does not invalidate completed work.

- [ ] **Step 3: Refit once after all eligible train cycles and evaluate once**

Use all completed training-cycle shards, select settings from validation experiments, and calculate test metrics only after the configuration is frozen.

- [ ] **Step 4: Commit full-cohort evidence**

Run: `git add report/rgb_full_cohort && git commit -m "analysis: evaluate full RGB cohort" && git push`

### Task 5: Produce the publication evidence package

**Files:**
- Create: `scripts/plot_cost_to_rgb_paper.py`
- Create: `tests/test_paper_source_tables.py`
- Generate: `report/paper_figures/`
- Modify: `docs/raw_optimal_defrost_manuscript_core_cn.md`

- [ ] **Step 1: Define one claim per figure**

Figure 1 establishes empirical cost minima from unsmoothed data; Figure 2 establishes pointwise-regret image labels without frame leakage; Figure 3 compares locked models and cameras; Figure 4 reports test generalization, calibration, and cycle-level failures.

- [ ] **Step 2: Export source tables before drawing**

Each panel reads a CSV or Parquet source table stored under `report/paper_figures/source_data/`; tests assert that displayed sample counts and summary values reproduce from those tables.

- [ ] **Step 3: Export with the selected Python backend**

Generate editable SVG and PDF plus 600-dpi TIFF and PNG previews. Visually inspect every panel, label, legend, image crop, and argmin marker.

- [ ] **Step 4: Rewrite the manuscript from the claim register**

State the contribution as a retrospective, policy-conditional, equivalent-energy decision label plus cycle-held-out visual prediction. Keep prospective savings, occupant comfort, and causal optimality as future validation unless intervention data exist.

- [ ] **Step 5: Run adversarial review and final verification**

Require passing tests, citation verification, claim-to-source alignment, figure QA, and a clean `git diff --check` before the paper-demo commit and push.

## Self-review

- Coverage: raw cost, regret labels, binary/three-class choice, five models, all requested camera combinations, cycle-level splits, streamed cloud cycles, figures, and manuscript are represented.
- Deliberate exclusions: no smoothing, no arbitrary ±10-min class, no frame-random split, no prospective savings claim, and no new model architecture before the baseline gate.
- Main unresolved scientific gate: a Nature-level causal control claim still requires prospective interventions and direct comfort measurements; this demo can support the retrospective method and visual-label proof of concept only.
