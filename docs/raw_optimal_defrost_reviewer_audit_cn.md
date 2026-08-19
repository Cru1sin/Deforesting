# Nature-style adversarial review: raw-cost defrost manuscript core v0.1

Assumption: the target is a Nature-family, broad-interest interdisciplinary paper. This review evaluates the current manuscript core, not the still-unfinished RGB study.

## Findings ordered by severity

### R1 — Critical — The fixed observed ticket is not a counterfactual defrost model

- Location: Methods, “Empirical defrost and recovery ticket”; Abstract and Discussion boundaries.
- Affected claim: the computed start time is the true optimal defrost action.
- Evidence/reasoning: every candidate time receives the mean cost and duration measured when the installed controller actually defrosted. Earlier frost states may require different defrost energy and recovery duration.
- Why it matters: without \(K_D(\tau,x)\) and \(T_D(\tau,x)\), the argmin is conditional on a fixed-policy ticket and cannot establish causal savings.
- Required action: retain “empirical equivalent-energy optimum under the observed fixed-duration policy”; fit a simple condition-dependent ticket only if cycle-held-out residual diagnostics support it; ultimately run prospective timing interventions.
- New evidence needed: yes for any causal or deployment claim.
- Verification: candidate-specific ticket predictions generalize by held-out cycle, followed by randomized or controlled timing tests.

### R2 — Major — Full-paper title and visual-learning promise exceed current evidence

- Location: title, Abstract final sentence, proposed pipeline.
- Affected claim: visual frost states can identify the decision boundary.
- Evidence/reasoning: no RGB train/validation/test result exists yet.
- Why it matters: the full Nature claim depends on image generalization, not only retrospective cost calculation.
- Required action: keep image statements prospective until cycle-held-out results exist.
- New evidence needed: yes.
- Verification: locked cycle-level splits, multiple camera-group settings, strong baselines, uncertainty and failure analysis.

### R3 — Major — Future clean-anchor information changes a sensitive subset

- Location: clean-reference method.
- Affected claim: cycle labels are available at decision time.
- Evidence/reasoning: replacing the two-anchor reference with the current anchor alone shifted the optimum by more than 30 min in 12.8% of valid cycles; the 90th percentile shift was 50.4 min.
- Why it matters: the following-cycle anchor is valid for offline reconstruction but unavailable online and potentially affected by the observed defrost.
- Required action: export this sensitivity, flag unstable cycles, and develop a contemporaneous reference model before deployment claims.
- New evidence needed: analysis now; prospective validation later.
- Verification: a current-information reference reproduces stable labels on held-out cycles.

### R4 — Major — The “thermal comfort” term is not a comfort measurement

- Location: cost definition and interpretation.
- Affected claim: improved thermal comfort.
- Evidence/reasoning: the dataset has water-side thermal shortfall but no indoor air state, PMV/PPD, occupancy or exposure duration.
- Why it matters: service shortfall and occupant comfort are not interchangeable outcomes.
- Required action: use “thermal-service shortfall” or “equivalent-energy proxy”; add indoor measurements before comfort claims.
- New evidence needed: yes for comfort.
- Verification: a predefined comfort metric measured during controlled timing experiments.

### R5 — Major — The near-optimal classification threshold is not identified

- Location: Results, 5% band; proposed RGB labels.
- Affected claim: a fixed ±10-min or 5% band is the correct third class.
- Evidence/reasoning: median band width changes from 33.0 min at 1% regret to 102.8 min at 10% regret.
- Why it matters: class definition can dominate classifier performance and scientific interpretation.
- Required action: export candidate regret; select the threshold using held-out cycle performance and decision regret, not frame-level accuracy alone.
- New evidence needed: yes, during RGB experiments.
- Verification: threshold chosen without test-set access and supported by downstream regret/calibration.

### R6 — Major — Repeated cycles and uncertainty are not yet modelled

- Location: cross-cycle result summary and Figure 1.
- Affected claim: generality across operating conditions.
- Evidence/reasoning: descriptive medians treat cycles as the unit but do not quantify experiment/date clustering or confidence intervals.
- Why it matters: adjacent cycles can share environmental and equipment state.
- Required action: report experiment-level grouping; use cluster bootstrap or leave-one-experiment-out sensitivity when inferential claims are added.
- New evidence needed: re-analysis, not new experiments.
- Verification: effect and interval remain interpretable under experiment-level resampling.

### R7 — Editorial — “Economic optimum” overstates the current objective

- Location: Chinese report terminology.
- Affected claim: monetary or complete economic optimization.
- Evidence/reasoning: the objective is kW-equivalent energy/service cost with no tariff, demand charge or direct comfort valuation.
- Required action: use “empirical equivalent-energy optimum” until economic weights are measured.
- New evidence needed: no for renaming; yes for monetary claims.
- Verification: terminology and units agree throughout text, tables and figures.

## Strengths that survive scrutiny

- Raw pointwise heating and power data are retained; no monotonic or smoothing assumption manufactures the optimum.
- Cross-cycle reconstruction correctly resolves the defrost/recovery window split across files.
- The complete-domain rule prevents truncated raw records from generating false right-boundary optima.
- All 77 cycles remain auditable, including 30 without a valid point estimate.
- Mean-versus-median ticket and alternative-reference sensitivities expose instability instead of suppressing it.
- Candidate-level regret and multiple near-optimal thresholds provide a defensible bridge to interval labels.

## Editorial decision

- Current cost-stage methods paper: **Major revision**.
- Claimed complete Nature paper: **Reject/rescope at present**, because the image and prospective-control evidence is absent.

## Prioritized revision roadmap

1. Preserve the policy-conditional claim and equivalent-energy terminology.
2. Use candidate regret to define high-confidence/near-optimal states; do not hard-code ±10 min.
3. Audit whether ticket cost/duration require a minimal condition-dependent model using cycle-held-out residuals.
4. Train image models with cycle-level splits and report camera-group generalization.
5. Add experiment-level uncertainty and failure-mode analysis.
6. Test timing interventions before claiming energy or comfort improvement.

## Competing hypotheses

| Hypothesis | Current evidence fit | Falsifiable test |
|---|---|---|
| Frost-related service loss creates the interior cost minimum | Plausible; raw \(Q_h\) declines in representative cycles | Relate candidate regret to independent frost/air-side indicators and RGB state |
| Controller and water-loop drift, not frost, creates the minimum | Also plausible; reference choice changes a subset | Current-information baseline and matched-condition analysis |
| A constant defrost ticket mechanically creates early minima | Plausible for sensitive cycles | Condition-dependent ticket fitted and evaluated by held-out cycle |
| Broad valleys indicate no practically unique optimum | Strongly supported by regret bands | Prospective outcomes across several times inside/outside the band |

## Review limitations

No literature novelty audit, RGB dataset, intervention experiment, indoor comfort measurement or external test cohort was available. The review therefore cannot assess priority, image generalization or real-world savings.
