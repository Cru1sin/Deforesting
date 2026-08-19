# Nature-style adversarial review

## Editorial decision

**Decision: Major revision — retrospective proof-of-concept, not yet submission-ready for Nature Communications.**

The manuscript presents a coherent and unusually auditable bridge from raw heat-pump operation to image supervision. Its strongest elements survive scrutiny: cost is integrated from unsmoothed observations; the output is explicitly policy conditional; pointwise regret replaces an unjustifiably sharp timestamp; experiments rather than frames define evaluation units; and RGB is tested against a strong retrospective time control. The complete-cohort result is also reported honestly: the pre-locked all-view increment crosses zero, whereas the front-camera increment is labelled exploratory.

The present evidence does not yet establish that the reconstructed minima transport to actions different from those observed, that RGB adds information prospectively, or that either component improves energy or thermal-service outcomes. Those are not wording problems. They require a prospective intervention and an independent confirmation cohort. The manuscript can stand now as a methods proof-of-concept, but a Nature Communications-level operational claim requires the experiments listed below.

## Findings, ordered by severity

### REV-M01 — Counterfactual action validity

- **Severity:** Major
- **Location:** Results, “Unsmoothed measurements…”; Methods, “Observed defrost ticket…”; Discussion limitations
- **Affected claim:** cycle-specific empirical optimal defrost timing
- **Evidence:** heating history before each candidate is observed, but the defrost/recovery consequence of an earlier action is replaced by a cohort-average ticket. The following clean anchor is also used offline.
- **Why it matters:** the argmin is a policy-conditional reconstruction, not evidence that initiating defrost at that time would minimize realized cost.
- **Required action:** retain the current conditional wording and run a prospective experiment assigning cycles to candidate regions inside and outside the low-regret set. Measure defrost electricity, recovery, delivered heat and indoor conditions.
- **New data needed:** yes.
- **Verification:** estimated low-regret regions predict realized renewal cost under changed actions, with uncertainty reported by experiment.

### REV-M02 — Independent confirmation of the camera-specific increment

- **Severity:** Major
- **Location:** Abstract; Results, “RGB was tested…”; Discussion
- **Affected claim:** the front view adds decision information beyond cycle time
- **Evidence:** front was the only RGB-alone group whose balanced-accuracy and regret-difference intervals excluded zero, but it was identified after complete-cohort inspection across nine camera groups.
- **Why it matters:** post-cohort selection can convert camera multiplicity into an apparently stable discovery.
- **Required action:** pre-specify front as the primary view, freeze the 1% task and model, and test once on new experiments not used in any model or threshold decision. Treat the current interval as hypothesis-generating.
- **New data needed:** yes.
- **Verification:** the independent paired RGB-minus-online-time interval excludes zero in the beneficial direction for both balanced accuracy and misclassification regret.

### REV-M03 — Retrospective time is not an online comparator

- **Severity:** Major
- **Location:** Methods, “Temporal control…”; Results and Discussion
- **Affected claim:** visual information beyond operational timing
- **Evidence:** normalized progress uses the future observed candidate boundary.
- **Why it matters:** this is an excellent leakage diagnostic but cannot quantify improvement over a deployable controller.
- **Required action:** add a control using only variables available at decision time, such as elapsed heating time and contemporaneous operating sensors, and preserve the retrospective-progress model as a separate diagnostic upper bound.
- **New data needed:** existing data suffice for an offline baseline; prospective deployment validation still needs new data.
- **Verification:** identical experiment-held-out folds compare RGB against an explicitly online-available baseline with paired intervals.

### REV-M04 — Apparatus, season and experiment support are narrow

- **Severity:** Major
- **Location:** Study object and cohort; limitations
- **Affected claim:** general transfer across experiments
- **Evidence:** 45 evaluable cycles arise from 11 experiments on one installation over a short observation period; strict thresholds leave only seven or two evaluable experiments.
- **Why it matters:** held-out dates do not establish transfer across hardware, weather regimes, camera ageing or sites.
- **Required action:** describe the environmental and control envelope in full and add a temporally separated or external apparatus cohort before making general claims.
- **New data needed:** yes for external validity.
- **Verification:** performance and paired increments are reported for a pre-defined external cohort without re-tuning.

### REV-M05 — Illumination and camera geometry remain alternative explanations

- **Severity:** Major
- **Location:** Image processing; camera comparison; Fig. 3
- **Affected claim:** RGB captures visible frost state
- **Evidence:** compact colour/gradient features may encode exposure, background or view stability. The strongest result coming from one view is equally compatible with better illumination control.
- **Why it matters:** camera-dependent performance does not by itself identify frost morphology as the causal visual signal.
- **Required action:** audit illumination/exposure metadata, test global colour normalization or exposure perturbations using the existing images, and add saliency or region-restricted ablations tied to the heat-exchanger surface.
- **New data needed:** not necessarily for the first audit; controlled illumination data would provide stronger confirmation.
- **Verification:** the front-view increment persists when nuisance appearance is perturbed or removed and disappears when the coil region is withheld.

### REV-M06 — Model-selection chronology and multiplicity need a reproducible lock

- **Severity:** Major
- **Location:** Model development; Streamed image processing and models; Statistical analysis
- **Affected claim:** developmental leave-one-experiment-out estimates reflect a frozen pipeline
- **Evidence:** five models, binary versus three-class tasks, four regret thresholds and nine camera groups were inspected during development. The manuscript states this, but the exact lock chronology is not tabulated.
- **Why it matters:** readers cannot distinguish pre-specified analysis from exploratory reuse of the same experiments.
- **Required action:** add a one-page analysis chronology listing every decision, data visible at that decision and the first untouched cohort to which it will be applied. Do not multiplicity-correct the exploratory screen into a confirmatory claim; confirm it independently.
- **New data needed:** no for documentation; yes for confirmation.
- **Verification:** every headline result is tagged pre-specified, development or exploratory in the manuscript and source table.

### REV-M07 — Economic objective sensitivity is incomplete

- **Severity:** Major
- **Location:** Cost definition; ticket sensitivity; Discussion
- **Affected claim:** pointwise regret represents economically relevant ambiguity
- **Evidence:** thermal-service shortfall is converted using cohort median clean COP, and the ticket uses a cohort mean. Ticket and anchor sensitivities are reported, but the service/electricity trade-off is not varied over a plausible range.
- **Why it matters:** optimal regions may depend on the implicit value assigned to unserved heat.
- **Required action:** add a compact sensitivity surface over plausible service weights and ticket cost/duration, reporting optimum shifts and low-regret overlap rather than selecting a favourable weight.
- **New data needed:** no.
- **Verification:** the stable and sensitive cycle subsets remain explicit across the declared range.

### REV-M08 — Literature and novelty audit is unfinished

- **Severity:** Major
- **Location:** Introduction; References; draft integrity notes
- **Affected claim:** the methodological gap is unresolved by prior optimal-timing and image-control studies
- **Evidence:** DOI and abstract records are verified, but detailed comparative wording has not undergone full-text audit.
- **Why it matters:** the contribution is a new integration and validation contract, so the novelty boundary depends on precise comparison with the closest integrated studies.
- **Required action:** complete full-text extraction for the closest energy-optimum, image-defrost and frost-tracking papers; add a comparison table of target, labels, split unit, counterfactual treatment and deployment evidence.
- **New data needed:** no.
- **Verification:** each novelty sentence maps to a verified full-text passage and no “first” claim is introduced.

### REV-M09 — Submission reproducibility is not complete

- **Severity:** Major
- **Location:** Data availability; Code availability
- **Affected claim:** the framework is auditable and reproducible
- **Evidence:** source tables and code exist, but the manuscript still lacks an archival DOI, frozen environment and public or controlled-access data record.
- **Why it matters:** local traceability is not equivalent to reviewer-accessible reproducibility.
- **Required action:** freeze a release, environment specification, figure source-data package and data-access record before submission.
- **New data needed:** no.
- **Verification:** a clean checkout reproduces the reported tables and Figs. 1–4 from documented inputs.

### REV-m01 — Figure and terminology cleanup

- **Severity:** Minor
- **Location:** Fig. 3 labels; figure legends
- **Affected claim:** camera-group definitions
- **Evidence:** labels such as “top + pair” can be read as three views rather than the pooled top/top-close group; complete standalone legends are not yet included in the manuscript draft.
- **Why it matters:** camera comparisons must be interpretable without source code.
- **Required action:** rename groups in reader-facing figures and add legends defining metrics, intervals, `n`, sampling and the exploratory status of front.
- **New data needed:** no.
- **Verification:** a reader can reconstruct each panel without consulting the methods code.

## Co-scientist challenge

| Candidate explanation | Evidence fit | Falsifiability | Parsimony | Test that separates it |
|---|---|---|---|---|
| Frost appearance contains incremental state information | Moderate; supported only for exploratory front | High | High | independent front-camera cohort versus online-only control |
| RGB primarily encodes elapsed time or systematic exposure drift | Moderate to high; all-view increment crosses zero | High | High | exposure perturbation and online time/sensor controls |
| Labels primarily reflect the fixed ticket and future clean anchor | High for the cost target, not necessarily for image separability | High | High | prospective actions and current-information-only reconstruction |
| Front succeeds because its geometry and illumination are more stable, not because it sees more frost | Moderate | High | High | coil-region masking, background-only and illumination-controlled ablations |
| High accuracy mainly reflects easy extreme states created by selective exclusion | High; error rate falls from 15.9% near the boundary to 0.6% above 5% regret | High | High | calibrated risk-coverage curves on untouched experiments |

The original explanation is plausible but not uniquely identified. The strongest counter-hypothesis is that the full pipeline learns a time- and acquisition-correlated proxy for a cost label whose counterfactual component has not been experimentally observed.

## Reviewer-panel synthesis

- **Editorial fit:** the cost-to-regret-to-image bridge is conceptually clear and potentially broad, but current evidence is one-apparatus retrospective development.
- **Methods/statistics:** experiment-level splitting, paired intervals and negative controls are strong; action validity, post-selection and external confirmation are the principal weaknesses.
- **Domain:** the policy-conditional framing is correct; claims about realised optimum, comfort or savings must remain blocked.
- **Cross-disciplinary perspective:** the work is best framed as decision-aware selective supervision, not as a new image architecture.
- **Devil's advocate:** the same results are compatible with temporal/exposure proxies and label-model artefacts.
- **Disagreement:** a methods journal could consider the present proof-of-concept after bounded revision; Nature Communications-level operational significance requires prospective confirmation.

## Strengths that survive scrutiny

1. Raw cost integration avoids using smoothing to manufacture an optimum.
2. Pointwise regret represents label ambiguity more honestly than a universal ±10 min window.
3. Experiment-held-out evaluation prevents the most obvious frame leakage.
4. The retrospective time control exposes, rather than hides, a strong competing explanation.
5. Failure cycles, null all-view increments and shrinking threshold coverage are reported rather than suppressed.

## Prioritized revision roadmap

1. **Prospective confirmation:** intervene at pre-defined candidate regions and measure realized outcomes.
2. **Frozen external visual test:** front camera, 1% threshold, locked model, online-available baseline.
3. **Existing-data analyses:** illumination/region ablations and economic-weight sensitivity.
4. **Reporting lock:** analysis chronology, full figure legends, model/preprocessing details and complete literature comparison.
5. **Reproducibility:** archival data/code release and clean-run verification.

## Review limitations

The review used the manuscript, committed result tables, source data and figures available in the project. It did not independently recalibrate physical sensors, inspect every raw RGB frame, reproduce cloud downloads, or verify all cited papers from full text. It therefore evaluates internal claim–evidence alignment, not external replication.
