# Pareto boundary experiment implementation plan

**Goal:** Test economic inputs and Pareto relation supervision as a fixed 2×2 comparison, using all current eligible cycles and reusable front-camera data.

**Architecture:** Existing Dataset → independent RGB/statistics/measured-accounting caches → train-experiment-only G → unchanged Pareto selector → one conditional network → first nonnegative score. No future predictor, new cost, near-optimal window, or controller tuning.

**Stack:** Project uv, pandas/parquet, existing Ridge/DINOv2, torch, loky (6 workers, inner threads 1), existing publication renderers.

## Frozen decisions

- Use current Dataset metadata, not the old 101-cycle or 69-teacher subset. Retain explicit exclusions.
- Front camera; frozen DINOv2 ViT-S/14, Resize256/CenterCrop224/ImageNet. Cache exact image keys independently of labels. Current image only; no new history factor.
- Last-five-minute sensor statistics: mean/std/skew/kurtosis/minute slope/10-bin histogram entropy. Resampled buckets become available at their end; imputed values are masked. No direct elapsed-time input. Retain measured cumulative heat/electricity.
- Existing G dynamic8 definition, four complete-event targets, signed water accounting, unchanged knee selector and 10-second candidate grid. Numerical teachers are fold-specific because G excludes the outer test experiment.
- Inner validation excludes both outer and inner experiments from G. G and teacher tables are shared across all methods and seeds.
- Actual image-time inputs and teacher candidate-time inputs coexist in one row index, but only the historical candidate grid may move the knee. The most recent same-cycle RGB is used at a candidate time, never a future frame.
- All four methods share the Sin60/60/32 sensor encoder and normalized conditional visual readout. Two economic slots are zeroed after scaling when off. Nonvisual control zeros the visual branch.
- BCE is native-frame mean plus an exact-knee anchor if absent from native frames. Relation loss is equal-weight mean softplus on adjacent distinct valid K ranks, same cycle/side/selector branch/support run. No fabricated K for dominated points.
- Max200 epochs, patience25, AdamW lr0.001/wd0.001; inner BCE chooses epoch; outer refit uses fixed selected epoch. No threshold search. All native pre-preparation frames are replayed, including recovery.
- Early undefined economics remain missing and are imputed from training only. Future-derived support/eligibility is evaluation metadata, never online input or a trigger gate.

## Work and acceptance

- [x] Reuse label-independent RGB cache and verify exact-key coverage: all 23,443 experimental frames present; 30,396 cached vectors overall. Cycle 62 has a source-corrupt JPEG and is outside the experimental cohort; its cache remains incomplete.
- [x] Cache measured account/stat rows once. Test causal truncation and original candidate-grid numerical parity. Save events and source/config metadata once.
- [x] Fit/cache fold G and teachers. Confirm test experiment absent from fitted references; preserve teacher abstentions and unavailable RGB explicitly.
- [x] Test shared model's input masks, same-side rank direction, grouped preprocessing and checkpoint replay on small synthetic data.
- [x] Run Baseline; review complete-stream triggers and losses before next stage.
- [x] Run economic-only addition; review against Baseline.
- [x] Run relation-only addition; review whether valid pairs cover enough cycles and improve triggers, not merely pair order.
- [x] Run combined method and matched nonvisual control; no new experiment family.
- [x] Render one shared figure suite: design matrix, loss, signed cycle/date errors, C/H deviations, representative full score trajectories and failure modes. Retain quantitative source tables.
- [x] Review actual results, report unsupported ideas as unsupported. No automatic merge/push or release-model replacement.

## Ownership

`image_models/dinov2_features.py`: existing extraction functions adapted to teacher-independent cache.
`image_models/pareto_data.py` + small existing sensor/accounting edits: reusable measured inputs and fold teachers.
`image_models/pareto_learning.py`: one model, relation construction and fold trainer.
`train_pareto_boundary.py`: explicit data/fit/train/plot actions and resume; no formulas.
Existing `plots/image_models.py` plus one shared comparison entry only if required: no method-specific renderers.

Shared data: `output/image_models/_cache/`; isolated results: `output/test/pareto_boundary_20260906/`.
