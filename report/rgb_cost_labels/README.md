# RGB cost-label audit

This stage reads image metadata only; it does not download, alter or delete cloud images.

- Cost-valid cycles with frost-development images: 45.
- Eligible image records across six camera roles: 57213.
- Locally available eligible records: 17153.
- Split unit: whole experiment (and therefore whole cycle), assigned deterministically in a 3 train : 1 validation : 1 test pattern over sorted experiments; no frame-level random split and no hash.
- Labels: pointwise interpolated relative regret. Images outside the candidate domain are excluded from model fitting. Images at or below each regret threshold are `near_optimal`; other images are `pre_optimal` or `post_optimal` relative to the earliest argmin.
- Audited thresholds: 1%, 2%, 5% and 10%. No final threshold is selected before held-out model and decision-regret evaluation.
- The 1% threshold has post-optimal images from 17/5/7 train/validation/test cycles and is the primary model-demo candidate; 2% is retained as label sensitivity.
- A no-download local demo can use 17153 eligible images, but its post-optimal class spans only 3/1/4 train/validation/test cycles. It is an engineering smoke test, not publication evidence.
