# Project brief

- Working title: Cost-derived visual frost states for heat-pump defrost timing
- Target journal: Nature Communications
- Manuscript type: Article; retrospective method and proof-of-concept visual validation.
- Research question: Can unsmoothed operational measurements define auditable, policy-conditional defrost-regret states, and can RGB frost images identify those states in experiments and cycles withheld from model training?
- Available data/results: 77 catalogued heating/defrost cycles; 47 complete empirical-cost candidate domains; 105,178 image metadata rows across six camera roles; raw-cost optimization, sensitivity analyses and pointwise-regret image labels; local images for cycles 60–77; five-model smoke test in progress.
- Required outputs: reproducible cost and label code; cycle/experiment-safe model comparison; single and paired camera comparison; streamed full-cohort evaluation; publication source data and figures; Chinese Nature-style manuscript draft; scientific review and revision record.
- Constraints: do not smooth cost inputs; do not random-split frames; do not infer occupant comfort from thermal-service shortfall; do not claim causal or globally optimal control from historical fixed-policy tickets; do not choose thresholds, models or cameras on the test split; delete no source data and archive superseded project artifacts.

## Current evidence boundary

The present optimization estimates an empirical equivalent-energy minimum conditional on the observed fixed-duration defrost ticket. The complete-paper claim requires locked cycle-held-out RGB performance. Claims of prospective energy savings, comfort improvement or causal optimality require controlled timing interventions and direct indoor-state measurements beyond the current demo.
