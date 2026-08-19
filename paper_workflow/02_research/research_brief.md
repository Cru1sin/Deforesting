# Research brief

## Research question

Can unsmoothed operational measurements define auditable, policy-conditional defrost-regret states, and can RGB frost images identify those states in experiments and cycles withheld from model training?

## Scope

- Unit of inference: one complete heating-to-defrost cycle; experiment/date remains the split group.
- Cost target: empirical equivalent-energy renewal cost under the observed fixed-duration defrost ticket.
- Visual target: pointwise relative-regret state, not manually judged frost severity and not an arbitrary time window.
- Primary comparison: high-confidence binary classification versus pre/near/post classification, followed by camera-group comparison with a model selected on validation data only.
- Out of scope for the current demo: causal global optimality, prospective savings, occupant comfort, and a deployable online clean-reference model.

## Search strategy

Initial verified search used title/keyword combinations covering `air-source heat pump`, `optimal defrost initiation`, `energy loss coefficient`, `frost image recognition`, `on-demand defrosting`, `frost accumulation tracking`, `classification with reject option`, and `cycle/group leakage`. Records are admitted to the source index only after DOI or official repository verification. Author-provided notes seed cross-domain terms but are not themselves citable evidence.

## Evidence synthesis

The literature already establishes three separate components: cycle-level optimal defrost timing from performance-loss objectives (HP-001, HP-002, HP-006), direct frost-state tracking (HP-003), and image-based demand defrosting (HP-004, HP-005). The present candidate contribution is therefore not “first optimal timing” or “first image defrost control”. It is the explicit bridge from a raw-data, cycle-specific cost surface to pointwise regret-valued RGB supervision, evaluated with experiment-held-out splits and ambiguity-aware labels.

The current dataset supports the cost and label construction retrospectively. It does not observe counterfactual recovery tickets for defrost actions taken earlier than the installed controller, so the resulting optimum remains conditional on the observed policy. The RGB smoke test can establish implementation feasibility but becomes paper evidence only after validation and test classes cover multiple independent experiments/cycles.

## Gaps and contradictions

1. The fixed empirical ticket may mechanically favour early candidate times and does not model action-dependent recovery.
2. A future-cycle clean anchor is valid for offline reconstruction but unavailable online and materially changes a minority of cycle optima.
3. Thermal-service shortfall is not a direct comfort metric.
4. Existing image-control papers report high accuracy or field savings, but their label construction and random-versus-group split contracts require full-text audit before comparison.
5. The current local validation subset has only two cycles and one post-optimal cycle; model ranking is provisional until the cloud cohort is streamed and evaluated.
