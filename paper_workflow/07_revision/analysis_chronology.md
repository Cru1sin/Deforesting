# Analysis decision chronology

This chronology separates pre-defined, developmental and exploratory decisions. It is a reporting lock, not a claim of prospective preregistration.

| Stage | Data visible | Decision | Status |
|---|---|---|---|
| Raw cost reconstruction | Operational cycle files and event catalogue | Use approximately 1-s unsmoothed observations; define the observed fixed-duration ticket; require at least 95% candidate-domain coverage | Pre-defined for the cost analysis |
| Cost ambiguity audit | Complete candidate curves for 47 cycles | Retain pointwise relative regret and audit 1%, 2%, 5% and 10%; do not impose a universal ±10 min interval | Pre-defined before final RGB evaluation |
| RGB metadata construction | Cost curves and image timestamps | Join regret by cycle and timestamp; keep whole experiments together; exclude images outside candidate domains | Pre-defined leakage control |
| Task and model development | Fixed 7/2/2 experiment split; historical test outputs were subsequently inspected | Compare five model families and binary versus three-class targets; select the 1% high-confidence binary task and compact RBF-SVM | Development; not a blind test |
| Complete-cohort extraction | All 45 cost-valid imaged cycles | Sample at most 12 images per cycle × state × camera role; retain 5,289 images; verify zero duplicate image keys | Frozen extraction contract |
| Developmental LOEO | 11 completed experiments | Evaluate RGB, retrospective time and RGB plus time on identical held-out experiments; resample experiments for paired 95% intervals | Developmental cross-validation |
| Camera and threshold audit | Complete LOEO outputs | Treat all views at 1% regret as the pre-locked primary summary; report nine camera groups and 1/2/5/10% thresholds as comparative or sensitivity analyses | All-view primary; camera findings exploratory |
| Post-cohort interpretation | Complete result tables and failure audit | Retain the all-view null increment; identify front as a prospective hypothesis; do not promote 2/5/10% thresholds despite higher apparent accuracy | Exploratory interpretation |

The next confirmatory analysis must freeze the front camera, 1% exclusion threshold, compact feature pipeline, classifier parameters and an online-available time/sensor comparator before any new experiment is inspected.
