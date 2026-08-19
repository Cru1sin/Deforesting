# RGB task and model selection record

## Locked development decision

The 1% pointwise-regret **binary** task is retained for the next demo stage. Images in the near-optimal set are excluded rather than forced into a visually ambiguous third class. Among five baselines on the fixed local validation subset, `color_rbf_svm` had the highest balanced accuracy for this task (0.816). It is therefore the only model used in the camera smoke comparison.

The provisional camera leader is the pooled `all` group (balanced accuracy 0.816), but several groups scored 0.813. With only two local validation cycles and one post-optimal validation cycle, these differences are not scientifically resolved. `all` is retained only as the maximum-coverage engineering default until the missing cohort is streamed.

`top_pair`, `left_pair` and `all` pool independently labelled images from their included camera roles; they are not synchronized multi-view fusion models.

## Integrity boundary

The local test metrics were inspected during smoke development. These six local test cycles are therefore not a blinded final test and must not support a confirmatory paper claim. Publication evidence requires experiment-level cross-validation and/or a new prospectively locked experiment cohort. Model, task and camera selection must not be revised in response to the already observed local test values.

## Source trace

- Five-model three-class results: `report/rgb_model_smoke/three_all/metrics.csv`.
- Five-model binary results: `report/rgb_model_smoke/binary_all/metrics.csv`.
- Camera results and manifests: `report/rgb_camera_comparison/binary_*/`.
- Labels: 1% pointwise empirical-cost regret from `report/rgb_cost_labels/image_cost_labels.parquet`.
- Splits: fixed experiment-level assignment from `report/rgb_cost_labels/cycle_splits.csv`.
