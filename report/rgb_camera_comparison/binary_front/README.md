# Local RGB model smoke test

- Task: binary; classes: pre_optimal, post_optimal.
- Camera group: front; roles: front.
- Label: 1% pointwise empirical-cost regret.
- Sampling: at most 12 evenly spaced frames per split × cycle × state × camera role.
- Decode QA: 0 unreadable sampled images were excluded from every model and recorded in `excluded_images.csv`; source files were not modified or deleted.
- Split: fixed experiment-level split from `report/rgb_cost_labels/cycle_splits.csv`.
- Models: color_rbf_svm.
- Scope: local-image engineering smoke test only. No hyperparameter search, repeated seeds, confidence intervals or cloud-cycle completion; do not use these metrics as publication evidence.
