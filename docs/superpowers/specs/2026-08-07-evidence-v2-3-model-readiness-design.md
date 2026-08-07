# Evidence v2.3 Model Readiness Design

## Purpose

Evidence v2.3 adds the shortest auditable chain needed to decide whether the
current sensor Dataset is ready for predictive modelling:

1. determine whether each performance target is measurable;
2. determine whether a candidate signal changes before a performance event;
3. determine whether candidate level and recent dynamics add information beyond
   elapsed time, operating context, and current performance.

The existing six Evidence v2.2 tables and their scientific meaning remain
unchanged. This work adds three tables and one module. It does not alter Dataset,
DatasetLoader, Process, pipeline, report, release, or freeze behavior.

The analysis version becomes `frost-cycle-evidence-v2.3`.

## Boundaries

- Dataset schema v3 and `DatasetLoader` remain the only input contract.
- Only cycles with Dataset status `valid` are analysed.
- All metrics use `cycle_stage == frost_development` unless this document
  explicitly names the Dataset `defrost_start` boundary.
- Dataset cycle status is authoritative and is never reinterpreted.
- Raw target values, baseline residuals, baseline columns, and `__imputed`
  columns are read from Dataset. Evidence does not interpolate, resample, or
  recompute a baseline.
- Existing Evidence trend, future association, profile, and similarity
  calculations are not redesigned.
- Recovery, image features, feature selection, hyperparameter search, neural
  models, optimal stopping, and control policy analysis are out of scope.

## Architecture

Add `src/frost_analysis/evidence/readiness.py` with three public analysis
operations:

```python
audit_performance_target(...)
compute_signal_lead(...)
compare_incremental_models(...)
```

Private helpers may implement event persistence, complete-case anchors,
leave-out splits, standardization, Ridge fitting, and date-balanced summaries.
No additional readiness submodules or CLI commands are introduced.

`build_evidence()` continues to stream valid cycles through `DatasetLoader` and
adds the readiness analysis after the existing Evidence calculations.
`EvidenceBundle` gains exactly three DataFrames:

```text
target_audit
readiness_split
readiness_summary
```

`write_evidence()` writes the three additional CSVs and includes their filenames
and row counts in the existing compact manifest. No output hashes are added.

## Settings

The following immutable, flat fields are added to `EvidenceSettings` and
`configs/evidence.yaml`:

```yaml
event_thresholds: [0.05, 0.10, 0.15]
primary_event_threshold: 0.10
event_persistence_seconds: 120

signal_reference_minutes: 5
signal_smoothing_seconds: 60
signal_mad_multiplier: 3.0
signal_persistence_seconds: 60

dynamic_window_minutes: 5
ridge_alpha: 1.0

context_features:
  - ambient_temperature
  - environment_relative_humidity
  - water_in_temperature
  - water_flow
  - compressor_frequency
```

These fields are included in the normalized settings hash. Existing
`minimum_valid_pairs` and `minimum_pair_coverage` govern readiness anchors; no
new sample threshold is introduced. The production configuration fixes the
event thresholds and horizons shown above. `EvidenceSettings` requires those
exact production tuples because the audit CSV columns are fixed to them.

## Target Audit

For target `y`, Evidence reads `<target>__baseline`. A cycle baseline is valid
only when the column exists, all non-null values are finite and strictly
positive, and all values are numerically consistent (`numpy.isclose` with
`rtol=1e-9`, `atol=1e-12`). It is never selected by taking an arbitrary first
row.

Baseline failures use these reasons:

```text
baseline_unavailable
baseline_nonpositive_or_zero
baseline_inconsistent
```

Three distinct quantities are used:

```text
current_degradation(t) = (baseline - y(t)) / abs(baseline)
future_degradation(t,h) = (y(t) - y(t+h)) / abs(baseline)
performance_event(gamma) = first persistent current_degradation >= gamma
```

`future_degradation` is the M0-M3 regression target. The performance event is
used only for lead-time analysis.

Target observations must be finite and have `<target>__imputed != true`.
Persistence is evaluated on exact canonical Dataset timestamps. A missing,
non-finite, or imputed point breaks persistence. A qualifying run begins at its
first threshold-satisfying timestamp and triggers only when every expected grid
timestamp is present and qualifying through a timestamp whose elapsed distance
from the run start is at least the configured duration. This elapsed-span rule
avoids deriving persistence from an ambiguous endpoint point count.

The observation boundary is `boundaries.defrost_start` from
`DatasetLoader.get_cycle_record()`. If a valid target does not trigger before
that boundary, it is right censored rather than treated as a negative event.
Primary event status is one of:

```text
event_observed
right_censored_at_legacy_defrost
target_unavailable
baseline_unavailable
```

The audit table has one row per `cycle_name x target`:

```text
cycle_name
experiment_date
target
baseline_value
target_observed_fraction
event_5_elapsed_minutes
event_10_elapsed_minutes
event_15_elapsed_minutes
primary_event_elapsed_minutes
primary_event_status
censor_elapsed_minutes
valid_pairs_5min
valid_pairs_10min
valid_pairs_20min
metric_status
exclusion_reason
```

Event and censor elapsed time are measured from the first frost-development
timestamp. Horizon pair counts require exact `t+h` timestamps in the same
cycle and frost-development stage, with both target observations valid and a
valid cycle baseline. `metric_status` is `available` when the target and
baseline can be audited and `unavailable` otherwise. `exclusion_reason` stores
the specific target or baseline failure reason rather than collapsing all
baseline failures into `primary_event_status`.

## Statistical Signal Onset

Signal onset is a statistical onset, not physical first frost, visual first
frost, performance degradation onset, or an optimal defrost time.

For each candidate residual, direction is aligned before detection:

```text
increase: z(t) = residual(t) - median(reference residual)
decrease: z(t) = -(residual(t) - median(reference residual))
```

The reference comprises finite, non-imputed observations in the first five
minutes starting at the first frost-development timestamp. Its MAD must be
finite and greater than zero; otherwise onset is unavailable with
`invalid_initial_scale`. No epsilon fallback is permitted.

The online statistic is a past-only rolling median over the closed interval
`[t - signal_smoothing_seconds, t]`. Alarm search starts after the reference
window and cannot reuse the reference window as an alarm period. Signal onset
is the first timestamp for which the rolling median exceeds
`signal_mad_multiplier * reference_MAD` for the full configured persistence
elapsed span, using the same complete-grid rule as performance events. Missing,
non-finite, imputed, or absent canonical timestamps break persistence.

Lead is computed only when both the primary performance event and signal onset
are observed:

```text
lead_minutes = performance_event_elapsed_minutes
               - signal_onset_elapsed_minutes
```

Right-censored performance events produce no numeric lead and use
`lead_status = performance_event_censored`. They are not filled with zero.

## Incremental Models

All four models use fixed Ridge regression with `alpha = ridge_alpha` and no
hyperparameter search:

```text
M0: frost-development elapsed minutes
M1: M0 + configured contexts + current target baseline residual
M2: M1 + current candidate baseline residual
M3: M2 + candidate Theil-Sen slope over the past dynamic_window_minutes
```

Context values and candidate values must be finite and non-imputed. Evidence
does not fill missing context. The dynamic slope uses only past and current
finite, non-imputed observations and requires at least two distinct timestamps.

The implementation uses NumPy rather than adding a model dependency. Predictor
means and standard deviations are fitted on training anchors only. A zero or
non-finite training standard deviation is treated as one after centering, so a
constant predictor contributes zero. Ridge is fitted on centred predictors
with an unpenalized intercept. Test data use training transformations only.

### Common Anchor Contract

For each `feature x target x horizon x split`, an anchor is valid only when all
M3 inputs and `future_degradation` are available. A missing value excludes that
anchor, not the whole split. M0, M1, M2, and M3 use exactly this same
complete-case anchor set.

`expected_anchor_count` is the number of held-out-cycle frost-development
anchors for which exact `t+h` exists in the same frost-development stage.
`valid_anchor_count` is the complete-case subset. A held-out cycle is
unavailable when:

```text
valid_anchor_count < minimum_valid_pairs
or anchor_coverage < minimum_pair_coverage
```

Training cycles are retained only when they independently meet the same anchor
requirements. At least one qualifying training cycle is required. Otherwise
the row is unavailable with `insufficient_training_cycles`.

### Splits and Evaluation

- With more than one experiment date, use leave-one-date-out.
- With one date and more than one cycle, use leave-one-cycle-out.
- With one valid cycle, target audit and signal/lead descriptions still run,
  but model fields are unavailable with
  `no_training_cycles_after_holdout`. The Evidence command succeeds.
- Random row splitting is forbidden.

`readiness_split.csv` has grain:

```text
split_id x held_out_cycle x feature x target x horizon
```

For leave-one-date-out, cycles from the same held-out date share a training set,
but each held-out cycle receives its own predictions and MAE. A date-level MAE
must never be copied to cycle rows.

Skills are:

```text
skill_context_vs_time = 1 - MAE(M1) / MAE(M0)
skill_level_vs_context = 1 - MAE(M2) / MAE(M1)
skill_dynamic_vs_level = 1 - MAE(M3) / MAE(M2)
```

If a denominator is zero or non-finite, the corresponding skill is unavailable
with `invalid_skill_denominator`; it is not replaced with zero.

The split table fields are:

```text
split_id
held_out_cycle
held_out_date
feature
target
horizon_minutes
signal_onset_elapsed_minutes
performance_event_elapsed_minutes
lead_minutes
lead_status
expected_anchor_count
valid_anchor_count
anchor_coverage
train_cycle_count
train_date_count
mae_m0
mae_m1
mae_m2
mae_m3
skill_context_vs_time
skill_level_vs_context
skill_dynamic_vs_level
metric_status
exclusion_reason
```

## Summary and Readiness Decision

Cycle-level results are summarized without pooling anchors. With multiple
dates, each date first takes the median of its held-out cycles and dates are
then equally weighted. With one date and multiple leave-one-cycle-out splits,
held-out cycles are the independent validation units.

Lead and improvement fractions use the same independent units: date-level
cycle medians when multiple dates exist, otherwise held-out cycles. This avoids
weighting dates by cycle count.

`readiness_summary.csv` has one row per `feature x target x horizon`:

```text
feature
target
horizon_minutes
trend_valid_cycle_count
trend_valid_date_count
trend_effect
trend_direction_consistency
lead_valid_cycle_count
lead_median_minutes
lead_q25_minutes
positive_lead_fraction
level_skill_median
level_improvement_fraction
dynamic_skill_median
dynamic_improvement_fraction
readiness_status
readiness_reason
```

No effect-size threshold or p-value gate is added. Status is assigned in this
order:

1. `target_not_evaluable`: no cycle has a valid baseline, target observations,
   and exact horizon response anchors.
2. `insufficient_validation_data`: no available independent model split, no
   valid trend evidence, or no calculable lead. This includes the current
   one-valid-cycle Dataset and takes precedence over descriptive single-cycle
   results.
3. `state_candidate`: lead Q1 is less than or equal to zero.
4. `no_incremental_prediction`: lead Q1 is positive, but either the
   date-balanced level skill median is not positive or no more than half of
   independent validation units have positive level skill.
5. `static_prediction_candidate`: stable positive level increment is present,
   but dynamic increment is not stable under the same sign-consistency rule.
6. `dynamic_prediction_candidate`: level and dynamic medians are positive and
   more than half of independent validation units are positive for each.

The sign-consistency rule adds no arbitrary effect magnitude threshold:

```text
stable positive increment = median skill > 0
                            and positive-unit fraction > 0.5
```

## Output and Failure Behavior

Scientific unavailability is represented in rows and does not fail the Evidence
command. Missing feature, target, context, baseline, or quality columns produce
the corresponding local unavailable rows. Structural contract violations still
fail fast, including invalid settings, unsupported Dataset schema, malformed
registry direction, and absence of core cycle columns such as `timestamp` or
`cycle_stage`.

The existing output-directory protection remains: Evidence cannot write inside
the Dataset. No Dataset file is modified.

## Testing

`tests/evidence/test_readiness.py` covers the scientific failure modes:

1. performance decline direction and distinct current/future degradation;
2. baseline unavailable, nonpositive, and inconsistent states;
3. event persistence, missing-point interruption, and right censoring;
4. signal direction alignment, past-only smoothing, reference exclusion,
   missing-point interruption, and invalid MAD;
5. exact same-cycle, same-stage horizon anchors;
6. anchor-level complete-case filtering and M0-M3 anchor identity;
7. cycle-level MAE under leave-one-date-out;
8. train-only standardization and no random row split;
9. level skill M2 versus M1 and dynamic skill M3 versus M2;
10. date-balanced medians and positive independent-unit fractions;
11. one-valid-cycle graceful degradation;
12. unchanged schemas and values for the existing six Evidence tables.

The implementation adds no new CLI command and no new modelling dependency.
