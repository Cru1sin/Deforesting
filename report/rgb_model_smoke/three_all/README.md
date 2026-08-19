# Local RGB model smoke test

- Task: three; classes: pre_optimal, near_optimal, post_optimal.
- Camera group: all; roles: top, top_close, left, left_close, front, extreme.
- Label: 1% pointwise empirical-cost regret.
- Sampling: at most 12 evenly spaced frames per split × cycle × state × camera role.
- Decode QA: 1 unreadable sampled images were excluded from every model and recorded in `excluded_images.csv`; source files were not modified or deleted.
- Split: fixed experiment-level split from `report/rgb_cost_labels/cycle_splits.csv`.
- Models: color logistic regression, color random forest, color RBF-SVM, small CNN and pretrained ResNet18 linear probe.
- Scope: local-image engineering smoke test only. No hyperparameter search, repeated seeds, confidence intervals or cloud-cycle completion; do not use these metrics as publication evidence.
