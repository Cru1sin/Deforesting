"""First-frame decision evidence for the shared economic × relation experiment.

Cycle-level timing/C/H remain separate; unknown economic points are not safe points.
Figures use observed held-out streams, with no smoothing or threshold selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from defrost_event_models.ridge_models import OUTCOME_TARGETS
from image_models.evaluation import classification_metrics
from image_models.pareto_learning import METHOD_NAMES
from plots.defrost_decision import _shade_experiment_dates
from plots.image_models import (
    _export,
    plot_trigger_error_figures,
    trigger_error_table,
    two_of_three_trigger,
)
from plots.pareto_selection import clip_point_to_axes, plot_cop_heating_rate_pareto

STYLES = {
    "experiment_balanced_mean": ("#777777", "o"),
    "ridge_basic_state_5": ("#3B7E9B", "s"),
    "ridge_physical_state_6": ("#2E7D5B", "^"),
    "ridge_dynamic_state_8": ("#C29565", "D"),
    "baseline": ("#666666", "o"), "economic": ("#3B7E9B", "s"),
    "relation": ("#A37CAA", "^"), "combined": ("#365A83", "D"),
    "nonvisual": ("#C29565", "v"),
    "s0": ("#777777", "o"), "s1": ("#3B7E9B", "s"),
    "s2": ("#2E7D5B", "^"), "s3": ("#C29565", "D"),
    "s4": ("#A34A42", "P"), "n2": ("#8A6FA8", "v"),
    "s1_neural": ("#76A9C2", "D"),
    "ridge_head_ridge_ch": ("#3B7E9B", "s"),
    "ridge_head_neural_ch": ("#3B7E9B", "D"),
    "neural_head_ridge_ch": ("#76A9C2", "s"),
    "neural_head_neural_ch": ("#76A9C2", "D"),
    "latent_ridge_ch_linear": ("#777777", "o"),
    "latent_ridge_ch_mlp": ("#3B7E9B", "s"),
    "latent_raw_ridge_ch_mlp": ("#2E7D5B", "D"),
    "raw_ridge_ch_mlp": ("#8A6FA8", "^"),
    "d32_ridge_ch_mlp": ("#777777", "o"),
    "t32_ridge_ch_mlp": ("#D28E4B", "^"),
}
VCNET_STYLES = {
    "multimodal_time_linear": ("#777777", "o"),
    "multimodal_time_linear_z16": ("#A0A0A0", "v"),
    "multimodal_time_linear_z64": ("#4C78A8", "s"),
    "multimodal_time_linear_z32_curvature": ("#D28E4B", "^"),
    "multimodal_time_linear_z64_curvature": ("#59A14F", "D"),
    "multimodal_time_varying": ("#4C78A8", "s"),
    "multimodal_time_varying_regularized": ("#59A14F", "D"),
}
VCNET_LABELS = {
    "multimodal_time_linear": "Time-linear event head",
    "multimodal_time_linear_z16": "D16: 16-d time-linear",
    "multimodal_time_linear_z64": "D64: 64-d time-linear",
    "multimodal_time_linear_z32_curvature": "T32: 32-d + curvature",
    "multimodal_time_linear_z64_curvature": "T64: 64-d + curvature",
    "multimodal_time_varying": "Time-varying event head",
    "multimodal_time_varying_regularized": (
        "Time-varying + coefficient regularization"
    ),
}
STATE_RECIPES = {
    "multimodal_time_linear",
    "multimodal_time_linear_z16",
    "multimodal_time_linear_z64",
    "multimodal_time_linear_z32_curvature",
    "multimodal_time_linear_z64_curvature",
}
FACTORIAL_RECIPES = (
    "multimodal_time_linear",
    "multimodal_time_linear_z64",
    "multimodal_time_linear_z32_curvature",
    "multimodal_time_linear_z64_curvature",
)
VCNET_CONTRASTS = {
    "capacity": {"multimodal_time_linear_z64": 1, "multimodal_time_linear": -1},
    "curvature_z32": {
        "multimodal_time_linear_z32_curvature": 1,
        "multimodal_time_linear": -1,
    },
    "curvature_z64": {
        "multimodal_time_linear_z64_curvature": 1,
        "multimodal_time_linear_z64": -1,
    },
    "capacity_given_curvature": {
        "multimodal_time_linear_z64_curvature": 1,
        "multimodal_time_linear_z32_curvature": -1,
    },
    "capacity_x_curvature": {
        "multimodal_time_linear_z64_curvature": 1,
        "multimodal_time_linear_z64": -1,
        "multimodal_time_linear_z32_curvature": -1,
        "multimodal_time_linear": 1,
    },
    "compression": {"multimodal_time_linear_z16": 1, "multimodal_time_linear": -1},
}
OUTCOME_LABELS = {
    "defrost_event_electricity_observed_kwh": "Event electricity [kWh]",
    "defrost_event_net_heat_observed_kwh": "Event net heat [kWh]",
    "defrost_event_compressor_electricity_observed_kwh": (
        "Compressor electricity [kWh]"
    ),
    "defrost_event_duration_observed_minutes": "Event duration [min]",
}
LINESTYLES = {
    "ridge_head_ridge_ch": "-", "ridge_head_neural_ch": "--",
    "neural_head_ridge_ch": "-", "neural_head_neural_ch": "--",
}
IDENTITY = ["method", "seed"]

OVERVIEW_FIGURE_STEMS = (
    "01_objectives_and_timing",
    "02_ridge_observed_calibration",
    "03_timing_and_consequence",
    "04_online_policy_evaluation",
    "diagnostic_neural_event_calibration",
    "diagnostic_online_information_and_probe",
    "diagnostic_curvature_effects",
    "reference_outcome_model_shift_ecdf",
    "reference_fixed_predictor_input_swap",
)
OVERVIEW_METHOD_LABELS = {
    "s0": "Outcome-supervised state only",
    "s1": "Add current Ridge C/H",
    "s2": "Add 5-min Ridge C/H history",
    "s3": "Add 5-min Ridge O history",
    "s4": "Add within-cycle ranking supervision",
    "n2": "RGB + sensor state vs sensor-only state",
    "latent_ridge_ch_linear": "Linear predictor",
    "latent_ridge_ch_mlp": "16-unit MLP",
    "latent_raw_ridge_ch_mlp": "Add pre-compression features",
    "raw_ridge_ch_mlp": "Pre-compression features",
    "d32_ridge_ch_mlp": "32D state",
    "t32_ridge_ch_mlp": "32D curvature-regularized state",
}
OVERVIEW_POLICY_LABELS = {
    "C": "C-only",
    "H": "H-only",
    "O": "O-only",
    "CH knee": "C-H reference",
    "CO knee": "C-O reference",
    "HO knee": "H-O reference",
    "CH → max O": "Best O on C-H front",
    "CO → max H": "Best H on C-O front",
    "HO → max C": "Best C on H-O front",
}


def _label(method, seed):
    return METHOD_NAMES[method].label if method in METHOD_NAMES else method.capitalize()


def _groups(frame):
    return sorted(frame.groupby(IDENTITY, sort=False),
                  key=lambda item: (list(STYLES).index(item[0][0]), item[0][1]))


def _supported(row, objective):
    value = row[f"{objective}_eligible_without_extrapolation"]
    return pd.notna(value) and bool(value)


def _median(values):
    finite = values.dropna()
    return finite.median() if len(finite) else np.nan


def _domain(row):
    if not np.isfinite([row.economic_c, row.economic_h]).all():
        return "unknown"
    names = [
        f"defrost_event_{quantity}_in_training_domain"
        for quantity in ("electricity", "net_heat", "duration")
    ]
    if any(name in row and pd.notna(row[name]) and not bool(row[name]) for name in names):
        return "outside_training_domain"
    if all(_supported(row, name) for name in ("cycle_cop", "cycle_heating_rate_kw")):
        return "in_training_domain"
    return "unknown"


def maximum_relative_performance_loss(reference_c, reference_h, trigger_c, trigger_h):
    values = np.asarray([reference_c, reference_h, trigger_c, trigger_h], dtype=float)
    if not np.isfinite(values).all() or reference_c <= 0 or reference_h <= 0:
        return np.nan
    return 100 * max(
        0., (reference_c - trigger_c) / reference_c,
        (reference_h - trigger_h) / reference_h,
    )


def vcnet_pareto_consequences(candidates):
    """Evaluate each independent neural knee with the same fold-frozen Ridge C/H."""
    records = []
    keys = ["representation", "seed", "heldout_experiment", "cycle_name"]
    for identity, cycle in candidates.groupby(keys, sort=False):
        reference = cycle.loc[cycle.is_knee.eq(True)]
        selected = cycle.loc[cycle.neural_is_knee.eq(True)]
        row = dict(zip(keys, identity, strict=True))
        row.update({
            "reference_row_id": reference.row_id.iloc[0] if len(reference) == 1 else None,
            "selected_row_id": selected.row_id.iloc[0] if len(selected) == 1 else None,
            "reference_time": reference.candidate_defrost_time.iloc[0]
            if len(reference) == 1 else pd.NaT,
            "selected_time": selected.candidate_defrost_time.iloc[0]
            if len(selected) == 1 else pd.NaT,
            "evaluation_reason": "valid",
        })
        if len(reference) != 1 or len(selected) != 1:
            row["evaluation_reason"] = (
                "missing_reference" if len(reference) != 1 else "missing_neural_knee"
            )
        else:
            valid = all(
                bool(reference[name].iloc[0]) and bool(selected[name].iloc[0])
                for name in (
                    "cycle_cop_measurements_valid", "cycle_cop_physically_valid",
                    "cycle_heating_rate_kw_measurements_valid",
                    "cycle_heating_rate_kw_physically_valid",
                )
            )
            values = [
                reference.cycle_cop.iloc[0], reference.cycle_heating_rate_kw.iloc[0],
                selected.cycle_cop.iloc[0], selected.cycle_heating_rate_kw.iloc[0],
            ]
            if not valid or not np.isfinite(values).all() or values[0] <= 0 or values[1] <= 0:
                row["evaluation_reason"] = "invalid_common_ridge_evaluation"
            else:
                row["reference_c"] = values[0]
                row["reference_h"] = values[1]
                row["selected_c"] = values[2]
                row["selected_h"] = values[3]
                row["d_c_percent"] = 100 * (values[0] - values[2]) / values[0]
                row["d_h_percent"] = 100 * (values[1] - values[3]) / values[1]
                row["maximum_relative_performance_loss_percent"] = max(
                    0., row["d_c_percent"], row["d_h_percent"]
                )
                row["delta_o"] = (
                    selected.cycle_evaporator_capacity_kw.iloc[0]
                    - reference.cycle_evaporator_capacity_kw.iloc[0]
                )
        row["delta_time_minutes"] = (
            (row["selected_time"] - row["reference_time"]).total_seconds() / 60
            if pd.notna(row["selected_time"]) and pd.notna(row["reference_time"]) else np.nan
        )
        records.append(row)
    return pd.DataFrame(records)


def vcnet_trajectory_metrics(replays):
    """Per-cycle variation on the unchanged ten-second teacher grid."""
    target_columns = [
        f"predicted_{target}" for target in OUTCOME_TARGETS.values()
        if f"predicted_{target}" in replays
    ]
    keys = [
        "representation", "seed", "replay", "heldout_experiment",
        "experiment_id", "cycle_name",
    ]
    records = []
    for identity, cycle in replays.loc[replays.is_teacher_candidate].groupby(keys, sort=False):
        cycle = cycle.sort_values("candidate_defrost_time", kind="stable")
        minutes = pd.to_datetime(cycle.candidate_defrost_time).diff().dt.total_seconds() / 60
        for target in target_columns:
            values = pd.to_numeric(cycle[target], errors="coerce")
            change = values.diff().abs()
            finite = change.notna() & minutes.gt(0)
            records.append({
                **dict(zip(keys, identity, strict=True)),
                "target": target.removeprefix("predicted_"),
                "adjacent_pairs": int(finite.sum()),
                "adjacent_absolute_change_median": change.loc[finite].median(),
                "absolute_change_per_minute_median": (
                    change.loc[finite] / minutes.loc[finite]
                ).median(),
                "total_variation": change.loc[finite].sum(min_count=1),
                "trajectory_span": values.max() - values.min(),
            })
    return pd.DataFrame(records)


def _objective_valid(row, name):
    return (
        bool(row[f"{name}_measurements_valid"])
        and bool(row[f"{name}_physically_valid"])
        and np.isfinite(row[name])
    )


def _uses_extrapolation(row, outcomes):
    return not all(bool(row[f"defrost_event_{name}_in_training_domain"]) for name in outcomes)


def trigger_is_dominated(trigger_c, trigger_h, teacher_grid):
    """Return whether an eligible frozen-teacher candidate dominates the trigger."""
    if not np.isfinite(trigger_c) or not np.isfinite(trigger_h):
        return pd.NA
    grid = teacher_grid.loc[teacher_grid["is_teacher_candidate"]].dropna(
        subset=["cycle_cop", "cycle_heating_rate_kw"]
    )
    no_worse = grid.cycle_cop.ge(trigger_c) & grid.cycle_heating_rate_kw.ge(trigger_h)
    better = grid.cycle_cop.gt(trigger_c) | grid.cycle_heating_rate_kw.gt(trigger_h)
    return bool((no_worse & better).any())


def event_calibration_metrics(predictions, targets, *, bootstrap_replicates=0):
    """Summarize OOF calibration without pooling repeated seeds as new events."""
    group_columns = [
        *( ["panel"] if "panel" in predictions else []),
        "representation", *( ["seed"] if "seed" in predictions else []),
    ]
    records = []
    for keys, rows in predictions.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        group = dict(zip(group_columns, keys, strict=True))
        for target in targets:
            observed = rows[target].to_numpy(dtype=float)
            predicted = rows[f"predicted_{target}"].to_numpy(dtype=float)
            valid = np.isfinite(observed) & np.isfinite(predicted)
            observed, predicted = observed[valid], predicted[valid]
            def summarize(observed_values, predicted_values):
                slope, intercept = (
                    np.polyfit(predicted_values, observed_values, 1)
                    if len(predicted_values) > 1 and np.ptp(predicted_values) > 0
                    else (np.nan, np.nan)
                )
                residual = predicted_values - observed_values
                return np.array([
                    residual.mean(), np.abs(residual).mean(),
                    np.sqrt(np.square(residual).mean()), intercept, slope,
                ])

            estimates = summarize(observed, predicted)
            intervals = np.full((2, 5), np.nan)
            if bootstrap_replicates and len(observed):
                valid_rows = rows.loc[valid].reset_index(drop=True)
                experiments = valid_rows.experiment_id.drop_duplicates().to_numpy()
                rng = np.random.default_rng(0)
                draws = []
                for _ in range(bootstrap_replicates):
                    sampled = rng.choice(experiments, len(experiments), replace=True)
                    sampled_rows = pd.concat([
                        valid_rows.loc[valid_rows.experiment_id.eq(experiment)]
                        for experiment in sampled
                    ], ignore_index=True)
                    draws.append(summarize(
                        sampled_rows[target].to_numpy(dtype=float),
                        sampled_rows[f"predicted_{target}"].to_numpy(dtype=float),
                    ))
                intervals = np.nanquantile(np.asarray(draws), [.025, .975], axis=0)
            records.append({
                **group, "target": target, "events": len(observed),
                "experiments": rows.loc[valid, "experiment_id"].nunique(),
                **{
                    name: estimates[index]
                    for index, name in enumerate((
                        "bias", "mae", "rmse", "calibration_intercept",
                        "calibration_slope",
                    ))
                },
                **{
                    f"{name}_ci_{bound}": intervals[index, position]
                    for index, bound in enumerate(("low", "high"))
                    for position, name in enumerate((
                        "bias", "mae", "rmse", "calibration_intercept",
                        "calibration_slope",
                    ))
                },
            })
    return pd.DataFrame(records)


def actual_action_ch_calibration(predictions, event_starts):
    """Evaluate event predictions in C/H at the observed preparation start."""
    from defrost_decision.performance_objectives import calculate_performance_objectives

    targets = list(OUTCOME_TARGETS.values())
    prediction_columns = [f"predicted_{target}" for target in targets]
    metadata = [
        name for name in ("panel", "representation", "seed") if name in predictions
    ]
    keys = ["event_id", "cycle_name", "experiment_id"]
    joined = event_starts.merge(
        predictions[[*keys, *metadata, *prediction_columns]],
        on=keys, how="inner", validate="one_to_many",
    )

    def objectives(rows, predicted):
        values = rows.copy()
        for outcome, target in OUTCOME_TARGETS.items():
            unit = "minutes" if outcome == "event_duration" else "kwh"
            values[f"defrost_{outcome}_{unit}"] = values[
                f"predicted_{target}" if predicted else target
            ]
            values[f"defrost_{outcome}_prediction_available"] = True
            values[f"defrost_{outcome}_in_training_domain"] = True
        return calculate_performance_objectives(
            values, allow_model_extrapolation=True
        )[["cycle_cop", "cycle_heating_rate_kw"]]

    observed = objectives(joined, False)
    predicted = objectives(joined, True)
    result = joined[[*keys, *metadata]].reset_index(drop=True)
    result["observed_c"] = observed.cycle_cop.to_numpy()
    result["observed_h"] = observed.cycle_heating_rate_kw.to_numpy()
    result["predicted_c"] = predicted.cycle_cop.to_numpy()
    result["predicted_h"] = predicted.cycle_heating_rate_kw.to_numpy()
    return result


def performance_consequences(
    predictions,
    evaluation,
    *,
    methods=tuple(METHOD_NAMES),
    strategies=("first_positive", "two_of_three"),
    selected_inputs=None,
):
    """Score the first executed trigger against one shared Ridge C/H reference."""
    predictions = predictions.copy()
    predictions["image_time"] = pd.to_datetime(predictions.image_time, format="mixed")
    evaluation = evaluation.copy()
    evaluation["candidate_defrost_time"] = pd.to_datetime(
        evaluation.candidate_defrost_time, format="mixed"
    )
    cohort = predictions[["heldout_experiment", "cycle_name"]].drop_duplicates()
    references = evaluation.loc[evaluation.is_knee].set_index(
        ["heldout_experiment", "cycle_name"]
    )
    selected = (
        selected_inputs.set_index("heldout_experiment").selected_for_s4.to_dict()
        if selected_inputs is not None and len(selected_inputs) else {}
    )
    rows = []
    for method in methods:
        method_rows = predictions.loc[predictions.method.eq(method)]
        seeds = method_rows.seed.drop_duplicates().tolist()
        seed = seeds[0] if len(seeds) == 1 else np.nan
        recipe = METHOD_NAMES[method]
        for heldout, cycle_name in cohort.itertuples(index=False):
            stream = method_rows.loc[
                method_rows.heldout_experiment.eq(heldout)
                & method_rows.cycle_name.eq(cycle_name)
            ].sort_values("image_time", kind="stable")
            reference_key = (heldout, cycle_name)
            reference = references.loc[reference_key] if reference_key in references.index else None
            if isinstance(reference, pd.DataFrame):
                reference = reference.iloc[0]
            for strategy in strategies:
                trigger_time = pd.NaT
                if len(stream):
                    if strategy == "first_positive":
                        positive = stream.logit.ge(0)
                        if positive.any():
                            trigger_time = stream.loc[positive, "image_time"].iloc[0]
                    elif strategy == "two_of_three":
                        trigger_time, _ = two_of_three_trigger(
                            stream.image_time, stream.logit, threshold=0.
                        )
                    else:
                        raise ValueError(f"unknown trigger strategy: {strategy}")
                input_id = selected.get(heldout) if method == "s4" else method
                input_recipe = METHOD_NAMES.get(input_id)
                row = {
                    "method": method, "method_name": recipe.name,
                    "method_label": recipe.label, "seed": seed,
                    "heldout_experiment": heldout, "cycle_name": cycle_name,
                    "strategy": strategy, "trigger_time": trigger_time,
                    "selected_input_method": input_id,
                    "selected_input_name": input_recipe.name if input_recipe else pd.NA,
                    "selected_input_label": input_recipe.label if input_recipe else pd.NA,
                    "reference_time": pd.NaT, "trigger_error_minutes": np.nan,
                    "reference_c": np.nan, "reference_h": np.nan, "reference_o": np.nan,
                    "trigger_c": np.nan, "trigger_h": np.nan, "trigger_o": np.nan,
                    "delta_c_percent": np.nan, "delta_h_percent": np.nan,
                    "delta_o": np.nan,
                    "maximum_relative_performance_loss_percent": np.nan,
                    "reference_uses_model_extrapolation": pd.NA,
                    "trigger_uses_model_extrapolation": pd.NA,
                    "reference_uses_measurement_reconstruction": pd.NA,
                    "trigger_uses_measurement_reconstruction": pd.NA,
                    "dominated_by_teacher_grid": pd.NA,
                    "evaluation_status": "evaluated",
                }
                if not len(stream):
                    row["evaluation_status"] = "missing_prediction"
                elif reference is None:
                    row["evaluation_status"] = "missing_reference"
                else:
                    row.update(
                        reference_time=reference.candidate_defrost_time,
                        reference_c=reference.cycle_cop,
                        reference_h=reference.cycle_heating_rate_kw,
                        reference_o=reference.cycle_evaporator_capacity_kw,
                    )
                    if pd.isna(trigger_time):
                        row["evaluation_status"] = "no_trigger"
                    else:
                        trigger_prediction = stream.loc[stream.image_time.eq(trigger_time)].iloc[0]
                        matches = evaluation.loc[
                            evaluation.heldout_experiment.eq(heldout)
                            & evaluation.row_id.eq(trigger_prediction.row_id)
                        ]
                        if not len(matches):
                            row["evaluation_status"] = "missing_trigger_evaluation"
                        else:
                            trigger = matches.iloc[0]
                            row.update(
                                trigger_error_minutes=(
                                    trigger_time - reference.candidate_defrost_time
                                ).total_seconds() / 60,
                                trigger_c=trigger.cycle_cop,
                                trigger_h=trigger.cycle_heating_rate_kw,
                                trigger_o=trigger.cycle_evaporator_capacity_kw,
                                reference_uses_model_extrapolation=(
                                    _uses_extrapolation(reference, ("electricity", "net_heat"))
                                    or _uses_extrapolation(reference, ("net_heat", "duration"))
                                ),
                                trigger_uses_model_extrapolation=(
                                    _uses_extrapolation(trigger, ("electricity", "net_heat"))
                                    or _uses_extrapolation(trigger, ("net_heat", "duration"))
                                ),
                                reference_uses_measurement_reconstruction=bool(
                                    reference.pre_defrost_electricity_uses_measurement_reconstruction
                                    or reference.pre_defrost_heat_uses_measurement_reconstruction
                                ),
                                trigger_uses_measurement_reconstruction=bool(
                                    trigger.pre_defrost_electricity_uses_measurement_reconstruction
                                    or trigger.pre_defrost_heat_uses_measurement_reconstruction
                                ),
                            )
                            reference_valid = (
                                _objective_valid(reference, "cycle_cop")
                                and _objective_valid(reference, "cycle_heating_rate_kw")
                                and reference.cycle_cop > 0 and reference.cycle_heating_rate_kw > 0
                            )
                            trigger_valid = (
                                _objective_valid(trigger, "cycle_cop")
                                and _objective_valid(trigger, "cycle_heating_rate_kw")
                            )
                            if not reference_valid:
                                row["evaluation_status"] = "invalid_reference"
                            elif not trigger_valid:
                                row["evaluation_status"] = "invalid_trigger"
                            else:
                                row["delta_c_percent"] = 100 * (
                                    reference.cycle_cop - trigger.cycle_cop
                                ) / reference.cycle_cop
                                row["delta_h_percent"] = 100 * (
                                    reference.cycle_heating_rate_kw
                                    - trigger.cycle_heating_rate_kw
                                ) / reference.cycle_heating_rate_kw
                                if (
                                    _objective_valid(reference, "cycle_evaporator_capacity_kw")
                                    and _objective_valid(trigger, "cycle_evaporator_capacity_kw")
                                ):
                                    row["delta_o"] = (
                                        trigger.cycle_evaporator_capacity_kw
                                        - reference.cycle_evaporator_capacity_kw
                                    )
                                row["maximum_relative_performance_loss_percent"] = (
                                    maximum_relative_performance_loss(
                                        reference.cycle_cop,
                                        reference.cycle_heating_rate_kw,
                                        trigger.cycle_cop,
                                        trigger.cycle_heating_rate_kw,
                                    )
                                )
                                grid = evaluation.loc[
                                    evaluation.heldout_experiment.eq(heldout)
                                    & evaluation.cycle_name.eq(cycle_name)
                                ]
                                row["dominated_by_teacher_grid"] = trigger_is_dominated(
                                    trigger.cycle_cop, trigger.cycle_heating_rate_kw, grid
                                )
                rows.append(row)
    return pd.DataFrame(rows)


def _probe_paired_comparisons(consequences, output, comparisons=None):
    comparisons = comparisons or (
        ("latent_ridge_ch_mlp", "latent_ridge_ch_linear"),
        ("latent_raw_ridge_ch_mlp", "latent_ridge_ch_mlp"),
        ("latent_raw_ridge_ch_mlp", "raw_ridge_ch_mlp"),
    )
    records, rng = [], np.random.default_rng(0)
    keys = ["strategy", "heldout_experiment", "cycle_name"]
    for candidate, reference in comparisons:
        paired = consequences.loc[consequences.method.eq(candidate)].merge(
            consequences.loc[consequences.method.eq(reference)],
            on=keys, suffixes=("_candidate", "_reference"), validate="one_to_one",
        )
        for strategy, rows in paired.groupby("strategy", sort=False):
            for field in (
                "maximum_relative_performance_loss_percent", "trigger_error_minutes"
            ):
                left, right = f"{field}_candidate", f"{field}_reference"
                valid = rows[left].notna() & rows[right].notna()
                values = rows.loc[valid, ["heldout_experiment", left, right]].copy()
                values["difference"] = (
                    values[left] - values[right]
                    if field.startswith("maximum")
                    else values[left].abs() - values[right].abs()
                )
                experiments = values.groupby("heldout_experiment").difference.mean().to_numpy()
                draws = (
                    rng.choice(experiments, (2000, len(experiments)), replace=True).mean(axis=1)
                    if len(experiments) else np.array([np.nan])
                )
                records.append({
                    "candidate": candidate, "reference": reference,
                    "strategy": strategy, "metric": field,
                    "paired_cycles": len(values), "experiments": len(experiments),
                    "mean_difference": experiments.mean() if len(experiments) else np.nan,
                    "ci_low": np.quantile(draws, .025) if len(experiments) else np.nan,
                    "ci_high": np.quantile(draws, .975) if len(experiments) else np.nan,
                })
    result = pd.DataFrame(records)
    result.to_csv(Path(output) / "paired_comparisons.csv", index=False)
    return result


def select_transfer_method(consequences):
    """Select one deployment candidate by the frozen consequence ordering."""
    rows = consequences.loc[consequences.strategy.eq("first_positive")]
    records = []
    for (method, seed), group in rows.groupby(["method", "seed"], sort=False):
        loss = group.maximum_relative_performance_loss_percent.dropna()
        records.append({
            "method": method,
            "seed": seed,
            "unevaluable_fraction": 1 - group.evaluation_status.eq("evaluated").mean(),
            "p90_loss": loss.quantile(.9),
            "median_loss": loss.median(),
            "coverage_1_percent": loss.le(1).sum() / len(group),
            "coverage_2_percent": loss.le(2).sum() / len(group),
        })
    by_seed = pd.DataFrame(records)
    summary = by_seed.groupby("method", as_index=False).mean(numeric_only=True)
    priority = {"raw_ridge_ch_mlp": 0, "d32_ridge_ch_mlp": 1, "t32_ridge_ch_mlp": 2}
    summary["tie_break_priority"] = summary.method.map(priority)
    summary = summary.sort_values([
        "unevaluable_fraction", "p90_loss", "median_loss",
        "coverage_1_percent", "coverage_2_percent", "tie_break_priority",
    ], ascending=[True, True, True, False, False, True], kind="stable")
    return summary.method.iloc[0], summary.reset_index(drop=True)


def _loss_plot_limits(losses: pd.Series) -> tuple[float, float, int]:
    """Keep one divergent point from flattening every finite loss trajectory."""
    values = pd.to_numeric(losses, errors="coerce")
    values = values.loc[np.isfinite(values) & values.gt(0)]
    robust_upper = float(values.quantile(.999))
    maximum = float(values.max())
    upper = robust_upper * 1.2 if maximum > 10 * robust_upper else maximum * 1.1
    return float(values.min()) * .8, upper, int(values.gt(upper).sum())


def render_probe_figures(
    predictions, losses, selected_epochs, consequences, output, *, denominator,
    comparisons=None,
):
    """Render the matched frozen-representation probe through shared styles."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    methods = list(selected_epochs.method.drop_duplicates())
    classification = []
    for (method, heldout), rows in predictions.groupby(
        ["method", "heldout_experiment"], sort=False
    ):
        valid = rows.target.isin([0, 1])
        classification.append({
            "method": method, "heldout_experiment": heldout,
            "frames": int(valid.sum()),
            **classification_metrics(
                rows.loc[valid, "target"], rows.loc[valid, "prediction"], "binary"
            ),
        })
    pd.DataFrame(classification).to_csv(output / "classification_by_fold.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for method in methods:
        rows = losses.loc[losses.method.eq(method)]
        for fold in rows.heldout_experiment.drop_duplicates():
            values = rows.loc[rows.heldout_experiment.eq(fold)]
            axes[0].plot(
                values.loc[values.split.eq("train_side"), "epoch"],
                values.loc[values.split.eq("train_side"), "loss"],
                color=STYLES[method][0], alpha=.15, lw=.6,
            )
            axes[0].plot(
                values.loc[values.split.eq("val_side"), "epoch"],
                values.loc[values.split.eq("val_side"), "loss"],
                color=STYLES[method][0], alpha=.35, lw=.7,
            )
        epochs = selected_epochs.loc[selected_epochs.method.eq(method), "selected_epoch"]
        axes[1].scatter(
            np.full(len(epochs), methods.index(method)), epochs,
            color=STYLES[method][0], marker=STYLES[method][1], s=15, alpha=.65,
        )
    axes[0].set(
        xlabel="Epoch", ylabel="BCE (log scale)", title="Inner train and validation loss"
    )
    axes[0].set_yscale("log")
    lower, upper, clipped = _loss_plot_limits(losses.loss)
    axes[0].set_ylim(lower, upper)
    if clipped:
        noun = "point" if clipped == 1 else "points"
        axes[0].text(
            .98, .98, f"{clipped} {noun} above axis\nmax={losses.loss.max():.2g}",
            transform=axes[0].transAxes, ha="right", va="top", fontsize=6,
        )
    axes[1].set(
        ylabel="Selected epoch", title="Chronological inner validation",
        xticks=range(len(methods)),
        xticklabels=[METHOD_NAMES[method].label for method in methods],
    )
    axes[1].tick_params(axis="x", labelrotation=25, labelsize=6)
    figure.tight_layout()
    _export(figure, output / "probe_training")
    for strategy in ("first_positive", "two_of_three"):
        render_performance_coverage(
            consequences, strategy, output, denominator=denominator
        )
    paired = _probe_paired_comparisons(consequences, output, comparisons)
    primary = paired.loc[
        paired.strategy.eq("first_positive")
        & paired.metric.eq("maximum_relative_performance_loss_percent")
    ]
    figure, axis = plt.subplots(figsize=(7.2, 2.8))
    positions = np.arange(len(primary))
    axis.errorbar(
        primary.mean_difference, positions,
        xerr=[primary.mean_difference - primary.ci_low,
              primary.ci_high - primary.mean_difference],
        fmt="o", color="#365A83", ecolor="#777777", capsize=2,
    )
    axis.axvline(0, color="#333333", lw=.7, ls="--")
    axis.set(
        yticks=positions,
        yticklabels=[
            f"{METHOD_NAMES[row.candidate].label}\n− {METHOD_NAMES[row.reference].label}"
            for row in primary.itertuples(index=False)
        ],
        xlabel="Paired change in maximum C/H loss [percentage points]",
        title="Held-out experiment bootstrap · first positive trigger",
    )
    figure.tight_layout()
    _export(figure, output / "probe_paired_effects")
    adapted = _adapt(predictions)
    decisions = predictions[["cycle_name", "teacher_time"]].drop_duplicates().rename(
        columns={"teacher_time": "selected_defrost_time"}
    ).assign(is_selected=True)
    styles = {METHOD_NAMES[method].label: STYLES[method] for method in methods}
    plot_trigger_error_figures(
        predictions=adapted, decisions=decisions, output=output, source_output=output,
        image_feature="dinov2", classifier="pareto_boundary", continuous_stream=True,
        method_styles=styles, policies=("first_positive", "two_of_three"),
        threshold=0., flat_output=True, error_quantile=.90,
    )


def render_probe_examples(
    predictions, consequences, teacher_curves, output, *,
    candidate="latent_raw_ridge_ch_mlp", reference="latent_ridge_ch_mlp",
):
    """Reuse the cycle renderer for fixed and largest paired probe changes."""
    primary = consequences.loc[consequences.strategy.eq("first_positive")]
    candidate_rows = primary.loc[
        primary.method.eq(candidate),
        ["heldout_experiment", "cycle_name", "maximum_relative_performance_loss_percent"],
    ]
    reference_rows = primary.loc[
        primary.method.eq(reference),
        ["heldout_experiment", "cycle_name", "maximum_relative_performance_loss_percent"],
    ]
    paired = candidate_rows.merge(
        reference_rows, on=["heldout_experiment", "cycle_name"],
        suffixes=("_candidate", "_reference"), validate="one_to_one",
    ).dropna()
    paired["difference"] = (
        paired.maximum_relative_performance_loss_percent_candidate
        - paired.maximum_relative_performance_loss_percent_reference
    )
    extremes = []
    if len(paired):
        extremes.extend((paired.difference.idxmin(), paired.difference.idxmax()))
        extremes.extend(
            index for index in paired.difference.abs().sort_values(ascending=False).index
            if index not in extremes
        )
    names = {
        "frost_cycle_000036", "frost_cycle_000040", "frost_cycle_000090",
        *paired.loc[extremes[:3], "cycle_name"].tolist(),
    }
    selected = predictions.loc[predictions.cycle_name.isin(names)]
    curves = teacher_curves.loc[teacher_curves.cycle_name.isin(names)]
    teachers = curves.loc[curves.is_knee]
    cycles, _, _ = evaluate_predictions(selected, teachers)
    _cycle_figures(
        selected, cycles, Path(output), teacher_curves=curves,
        repeat_trigger_legends=False,
    )


def stopping_stability_metrics(predictions, local_window_minutes=15):
    """Summarize the existing knee-local score stability measures per cycle."""
    records = []
    for (method, seed, heldout, cycle), rows in predictions.groupby(
        ["method", "seed", "heldout_experiment", "cycle_name"], sort=False
    ):
        records.append({
            "method": method, "seed": seed, "heldout_experiment": heldout,
            "cycle_name": cycle,
            **_stream_stability(rows, local_window_minutes),
        })
    return pd.DataFrame(records)


def render_calibration_figures(
    predictions, event_starts, output, *, targets=None, labels=None, bootstrap_replicates=2000
):
    """Render OOF event and observed-action C/H calibration with true denominators."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    targets = list(OUTCOME_TARGETS.values()) if targets is None else targets
    labels = OUTCOME_LABELS if labels is None else labels
    metrics = event_calibration_metrics(
        predictions, targets, bootstrap_replicates=bootstrap_replicates
    )
    metrics.to_csv(output / "event_calibration_metrics.csv", index=False)
    predictions.to_csv(output / "event_calibration_source.csv", index=False)

    for panel, panel_rows in predictions.groupby("panel", sort=False):
        figure, axes = plt.subplots(
            (len(targets) + 1) // 2, 2,
            figsize=(7.2, 3.05 * ((len(targets) + 1) // 2)), squeeze=False,
        )
        for axis, target in zip(axes.flat, targets):
            low = min(panel_rows[target].min(), panel_rows[f"predicted_{target}"].min())
            high = max(panel_rows[target].max(), panel_rows[f"predicted_{target}"].max())
            for (method, seed), rows in panel_rows.groupby(
                ["representation", "seed"], sort=False
            ):
                color, marker = (
                    VCNET_STYLES.get(method, STYLES.get(method, ("#777777", "o")))
                )
                axis.scatter(
                    rows[target], rows[f"predicted_{target}"], s=10, alpha=.42,
                    color=color, marker=marker,
                    label=f"{VCNET_LABELS.get(method, method)} · seed {seed}",
                )
            axis.plot([low, high], [low, high], color="#333333", lw=.7, ls="--")
            axis.set(
                xlabel="Observed", ylabel="OOF predicted", title=labels[target]
            )
        for axis in list(axes.flat)[len(targets):]:
            axis.set_visible(False)
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, legend_labels, loc="lower center", ncol=2, fontsize=6.2, frameon=False)
        figure.suptitle(
            f"Actual-event calibration · {panel_rows.event_id.nunique()} events · "
            f"{panel_rows.experiment_id.nunique()} experiments",
            fontsize=10,
        )
        figure.tight_layout(rect=(0, .1, 1, .96))
        _export(figure, output / f"event_calibration_{panel}")

    if event_starts is None:
        return

    action = pd.concat([
        actual_action_ch_calibration(
            panel_rows, event_starts[panel]
        ).assign(panel=panel)
        for panel, panel_rows in predictions.groupby("panel", sort=False)
        if panel in event_starts
    ], ignore_index=True)
    action.to_csv(output / "actual_action_ch_calibration.csv", index=False)
    action_metrics = event_calibration_metrics(
        action.assign(
            c=action.observed_c, predicted_c=action.predicted_c,
            h=action.observed_h, predicted_h=action.predicted_h,
        ),
        ["c", "h"], bootstrap_replicates=2000,
    )
    action_metrics.to_csv(output / "actual_action_ch_metrics.csv", index=False)
    for panel, rows in action.groupby("panel", sort=False):
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
        for axis, objective in zip(axes, ("c", "h"), strict=True):
            low = min(rows[f"observed_{objective}"].min(), rows[f"predicted_{objective}"].min())
            high = max(rows[f"observed_{objective}"].max(), rows[f"predicted_{objective}"].max())
            for (method, seed), values in rows.groupby(
                ["representation", "seed"], sort=False
            ):
                color, marker = VCNET_STYLES.get(
                    method, STYLES.get(method, ("#777777", "o"))
                )
                axis.scatter(
                    values[f"observed_{objective}"], values[f"predicted_{objective}"],
                    s=10, alpha=.42, color=color, marker=marker,
                    label=f"{VCNET_LABELS.get(method, method)} · seed {seed}",
                )
            axis.plot([low, high], [low, high], color="#333333", lw=.7, ls="--")
            axis.set(
                xlabel=f"Observed-action {objective.upper()}",
                ylabel=f"OOF-predicted {objective.upper()}",
                title="Cycle COP" if objective == "c" else "Cycle heating rate",
            )
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=6.2, frameon=False)
        figure.suptitle("Observed preparation-start economic calibration", fontsize=10)
        figure.tight_layout(rect=(0, .14, 1, .95))
        _export(figure, output / f"actual_action_ch_calibration_{panel}")
    return metrics, action_metrics


def render_performance_coverage(consequences, strategy, output, *, denominator):
    """Plot fixed-denominator C/H loss coverage for all available methods."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    values = consequences.loc[consequences.strategy.eq(strategy)]
    full_max = max(5., float(values.maximum_relative_performance_loss_percent.max()))
    source, summaries = [], []
    methods = [method for method in METHOD_NAMES if method in set(values.method)]
    for method in methods:
        rows = values.loc[values.method.eq(method)]
        if not len(rows):
            continue
        loss = rows.maximum_relative_performance_loss_percent.dropna().sort_values()
        thresholds = np.unique(np.r_[0., loss.to_numpy(), full_max])
        recipe = METHOD_NAMES[method]
        source.append(pd.DataFrame({
            "method": method, "method_name": recipe.name, "method_label": recipe.label,
            "strategy": strategy, "loss_threshold_percent": thresholds,
            "coverage_fraction": [(loss <= value).sum() / denominator for value in thresholds],
        }))
        summaries.append({
            "method": method, "method_name": recipe.name, "method_label": recipe.label,
            "strategy": strategy, "cycles": denominator,
            "triggered_cycles": int(rows.trigger_time.notna().sum()),
            "no_trigger_cycles": int(rows.trigger_time.isna().sum()),
            "evaluated_cycles": int(loss.size),
            "median_signed_trigger_error_minutes": rows.trigger_error_minutes.median(),
            "median_absolute_trigger_error_minutes": rows.trigger_error_minutes.abs().median(),
            "p90_absolute_trigger_error_minutes": rows.trigger_error_minutes.abs().quantile(.9),
            "median_loss_percent": loss.median() if len(loss) else np.nan,
            "p90_loss_percent": loss.quantile(.9) if len(loss) else np.nan,
            **{
                f"coverage_at_{threshold}_percent": float((loss <= threshold).sum() / denominator)
                for threshold in (1, 2, 5)
            },
        })
    source = pd.concat(source, ignore_index=True)
    summary = pd.DataFrame(summaries)
    source.to_csv(output / f"performance_coverage_source_{strategy}.csv", index=False)
    summary.to_csv(output / f"performance_summary_{strategy}.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    for method in methods:
        curve = source.loc[source.method.eq(method)]
        if not len(curve):
            continue
        label = (
            f"{METHOD_NAMES[method].label} "
            f"({int(summary.loc[summary.method.eq(method), 'evaluated_cycles'].iloc[0])}"
            f"/{denominator}; no trigger "
            f"{int(summary.loc[summary.method.eq(method), 'no_trigger_cycles'].iloc[0])})"
        )
        for axis in axes:
            axis.step(
                curve.loss_threshold_percent, curve.coverage_fraction,
                where="post", color=STYLES[method][0],
                ls=LINESTYLES.get(method, "-"), lw=1.25, label=label,
            )
    axes[0].set_xlim(0, full_max * 1.02)
    axes[1].set_xlim(0, 5)
    axes[0].set_title("Complete observed loss range")
    axes[1].set_title("Low-loss detail")
    for axis in axes:
        axis.set(
            ylim=(0, 1.02), xlabel="Maximum relative C/H performance loss (%)",
            ylabel=f"Evaluated low-loss cycles / {denominator}",
        )
        axis.grid(color="#E4E7EB", lw=.5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=6.4)
    figure.suptitle(
        "First positive frame" if strategy == "first_positive" else "Two positive frames in three",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, .22, 1, .95))
    _export(figure, output / f"performance_coverage_{strategy}")
    return source, summary


def cross_input_comparisons(predictions, consequences):
    """Pair Ridge and Neural C/H while holding each fitted stopping head fixed."""
    predictions = predictions.copy()
    predictions["image_time"] = pd.to_datetime(predictions.image_time, format="mixed")
    predictions["teacher_time"] = pd.to_datetime(predictions.teacher_time, format="mixed")
    consequences = consequences.copy()
    consequences["trigger_time"] = pd.to_datetime(
        consequences.trigger_time, format="mixed"
    )
    pairs = (
        ("ridge", "ridge_head_ridge_ch", "ridge_head_neural_ch"),
        ("neural", "neural_head_ridge_ch", "neural_head_neural_ch"),
    )
    cycle_tables, frame_tables = [], []
    frame_keys = [
        "row_id", "heldout_experiment", "cycle_name", "image_time", "teacher_time"
    ]
    cycle_keys = ["strategy", "heldout_experiment", "cycle_name"]
    for head_source, ridge_method, neural_method in pairs:
        ridge_frames = predictions.loc[
            predictions.method.eq(ridge_method), [*frame_keys, "logit"]
        ]
        neural_frames = predictions.loc[
            predictions.method.eq(neural_method), [*frame_keys, "logit"]
        ]
        frames = ridge_frames.merge(
            neural_frames, on=frame_keys, suffixes=("_ridge", "_neural"),
            validate="one_to_one",
        )
        frames["head_source"] = head_source
        frames["score_change"] = frames.logit_neural - frames.logit_ridge
        frames["absolute_score_change"] = frames.score_change.abs()
        frames["sign_flip"] = frames.logit_neural.ge(0).ne(frames.logit_ridge.ge(0))
        frames["within_15min_of_knee"] = (
            frames.image_time - frames.teacher_time
        ).abs().le(pd.Timedelta(minutes=15))
        frame_tables.append(frames)

        ridge_cycles = consequences.loc[
            consequences.method.eq(ridge_method),
            [*cycle_keys, "trigger_time", "maximum_relative_performance_loss_percent"],
        ]
        neural_cycles = consequences.loc[
            consequences.method.eq(neural_method),
            [*cycle_keys, "trigger_time", "maximum_relative_performance_loss_percent"],
        ]
        cycles = ridge_cycles.merge(
            neural_cycles, on=cycle_keys, suffixes=("_ridge", "_neural"),
            validate="one_to_one",
        )
        ridge_trigger = cycles.trigger_time_ridge.notna()
        neural_trigger = cycles.trigger_time_neural.notna()
        cycles["head_source"] = head_source
        cycles["trigger_status"] = np.select(
            [ridge_trigger & neural_trigger, ridge_trigger, neural_trigger],
            ["both_trigger", "ridge_only", "neural_only"], default="neither_trigger",
        )
        cycles["same_trigger_frame"] = (
            ridge_trigger & neural_trigger
            & cycles.trigger_time_ridge.eq(cycles.trigger_time_neural)
        )
        cycles["trigger_change_minutes"] = (
            cycles.trigger_time_neural - cycles.trigger_time_ridge
        ).dt.total_seconds() / 60
        cycles["absolute_trigger_change_minutes"] = cycles.trigger_change_minutes.abs()
        cycles["loss_change_percent"] = (
            cycles.maximum_relative_performance_loss_percent_neural
            - cycles.maximum_relative_performance_loss_percent_ridge
        )
        cycle_tables.append(cycles)
    return (
        pd.concat(cycle_tables, ignore_index=True),
        pd.concat(frame_tables, ignore_index=True),
    )


def _experiment_interval(rows, field, rng):
    experiments = rows.groupby("heldout_experiment")[field].mean().dropna().to_numpy()
    draws = rng.choice(experiments, (2000, len(experiments)), replace=True).mean(axis=1)
    return experiments.mean(), np.quantile(draws, .025), np.quantile(draws, .975)


def render_cross_input_figures(cycles, frames, output):
    """Render fixed-head input perturbation timing and paired evidence."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for (head, strategy), rows in cycles.groupby(["head_source", "strategy"], sort=False):
        status = rows.trigger_status.value_counts()
        both = rows.loc[rows.trigger_status.eq("both_trigger")]
        summaries.append({
            "head_source": head, "strategy": strategy, "cycles": len(rows),
            "both_trigger": int(status.get("both_trigger", 0)),
            "ridge_only": int(status.get("ridge_only", 0)),
            "neural_only": int(status.get("neural_only", 0)),
            "neither_trigger": int(status.get("neither_trigger", 0)),
            "same_trigger_frame": int(both.same_trigger_frame.sum()),
            "median_trigger_change_minutes": both.trigger_change_minutes.median(),
            "median_absolute_trigger_change_minutes": (
                both.absolute_trigger_change_minutes.median()
            ),
            "p90_absolute_trigger_change_minutes": (
                both.absolute_trigger_change_minutes.quantile(.9)
            ),
            "median_loss_change_percent": both.loss_change_percent.median(),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "cross_input_summary.csv", index=False)

    for head, rows in cycles.groupby("head_source", sort=False):
        figure, axes = plt.subplots(2, 1, figsize=(7.2, 4.7), sharex=True)
        for axis, strategy in zip(axes, ("first_positive", "two_of_three"), strict=True):
            values = rows.loc[rows.strategy.eq(strategy)].copy()
            values["cycle_id"] = pd.to_numeric(
                values.cycle_name.str.extract(r"(\d+)$", expand=False)
            )
            values = values.sort_values("cycle_id", kind="stable").reset_index(drop=True)
            finite = values.trigger_change_minutes.dropna()
            limit = max(1., float(finite.abs().quantile(.9))) if len(finite) else 1.
            shown = values.trigger_change_minutes.clip(-.96 * limit, .96 * limit)
            x = np.arange(len(values))
            _shade_experiment_dates(axis, values.heldout_experiment.astype(str).tolist())
            valid = shown.notna()
            axis.scatter(x[valid], shown[valid], s=15, color=STYLES[
                f"{head}_head_neural_ch"
            ][0], zorder=3)
            for index in np.flatnonzero(
                values.trigger_change_minutes.abs().gt(limit).fillna(False)
            ):
                axis.annotate(
                    "↑" if values.trigger_change_minutes.iloc[index] > 0 else "↓",
                    (x[index], shown.iloc[index]), ha="center", va="center", fontsize=7,
                )
            axis.axhline(0, color="#333333", lw=.7)
            counts = values.trigger_status.value_counts()
            axis.set(
                ylim=(-limit, limit), ylabel="Neural − Ridge input trigger [min]",
            )
            axis.set_title(
                f"{'First positive' if strategy == 'first_positive' else 'Two of three'} · "
                f"both {counts.get('both_trigger', 0)} · Ridge only "
                f"{counts.get('ridge_only', 0)} · Neural only "
                f"{counts.get('neural_only', 0)} · neither "
                f"{counts.get('neither_trigger', 0)}",
                pad=18,
            )
            axis.grid(axis="y", color="#E4E7EB", lw=.5)
        axes[-1].set(
            xlabel="Cycle ID", xticks=np.arange(len(values)),
            xticklabels=values.cycle_id.astype(int).astype(str),
        )
        axes[-1].tick_params(axis="x", labelrotation=90, labelsize=5)
        figure.suptitle(f"Fixed {head} head: economic-input perturbation", fontsize=10)
        figure.tight_layout()
        _export(figure, output / f"cross_input_trigger_change_{head}_head")

    frame_cycles = frames.groupby(
        ["head_source", "heldout_experiment", "cycle_name"], sort=False
    ).agg(
        median_absolute_score_change=("absolute_score_change", "median"),
        sign_flip_fraction=("sign_flip", "mean"),
        local_median_absolute_score_change=(
            "absolute_score_change",
            lambda values: values[frames.loc[values.index, "within_15min_of_knee"]].median(),
        ),
        local_sign_flip_fraction=(
            "sign_flip",
            lambda values: values[frames.loc[values.index, "within_15min_of_knee"]].mean(),
        ),
    ).reset_index()
    frame_cycles.to_csv(output / "cross_input_score_by_cycle.csv", index=False)
    rng = np.random.default_rng(0)
    records = []
    for (head, strategy), rows in cycles.groupby(["head_source", "strategy"], sort=False):
        for field in (
            "trigger_change_minutes", "absolute_trigger_change_minutes", "loss_change_percent"
        ):
            valid = rows.loc[rows[field].notna()]
            mean, low, high = _experiment_interval(valid, field, rng)
            records.append({
                "head_source": head, "strategy": strategy, "metric": field,
                "paired_cycles": len(valid), "experiments": valid.heldout_experiment.nunique(),
                "mean_difference": mean, "ci_low": low, "ci_high": high,
            })
    for head, rows in frame_cycles.groupby("head_source", sort=False):
        for field in (
            "median_absolute_score_change", "sign_flip_fraction",
            "local_median_absolute_score_change", "local_sign_flip_fraction",
        ):
            mean, low, high = _experiment_interval(rows, field, rng)
            records.append({
                "head_source": head, "strategy": "frames", "metric": field,
                "paired_cycles": rows[field].notna().sum(),
                "experiments": rows.loc[rows[field].notna(), "heldout_experiment"].nunique(),
                "mean_difference": mean, "ci_low": low, "ci_high": high,
            })
    bootstrap = pd.DataFrame(records)
    bootstrap.to_csv(output / "cross_input_paired_bootstrap.csv", index=False)

    metrics = (
        ("trigger_change_minutes", "Trigger change [min]"),
        ("absolute_trigger_change_minutes", "Absolute trigger change [min]"),
        ("loss_change_percent", "Maximum C/H loss change [%]"),
        ("local_median_absolute_score_change", "Local median |score change|"),
        ("local_sign_flip_fraction", "Local sign-flip fraction"),
    )
    figure, axes = plt.subplots(2, len(metrics), figsize=(8.2, 3.7), squeeze=False)
    for axes_row, head in zip(axes, ("ridge", "neural"), strict=True):
        for axis, (field, title) in zip(axes_row, metrics, strict=True):
            strategy = "frames" if field.startswith("local_") else "first_positive"
            row = bootstrap.loc[
                bootstrap.head_source.eq(head)
                & bootstrap.strategy.eq(strategy)
                & bootstrap.metric.eq(field)
            ].iloc[0]
            axis.errorbar(
                row.mean_difference, 0,
                xerr=[[row.mean_difference - row.ci_low], [row.ci_high - row.mean_difference]],
                fmt="D", ms=3, color=STYLES[f"{head}_head_neural_ch"][0],
                ecolor="#777777", capsize=2,
            )
            axis.axvline(0, color="#333333", lw=.7, ls="--")
            axis.set(yticks=[], title=title)
            axis.text(.03, .08, f"n={row.paired_cycles}", transform=axis.transAxes, fontsize=6)
        axes_row[0].set_ylabel(f"Fixed {head} head")
    figure.suptitle("Ridge → Neural economic-input perturbation", fontsize=10)
    figure.tight_layout()
    _export(figure, output / "cross_input_paired_evidence")
    return summary, bootstrap


def render_performance_paired_comparison(consequences, output):
    """Compare Neural C/H with matched Ridge C/H using saved held-out decisions."""
    output = Path(output)
    ridge = consequences.loc[consequences.method.eq("s1")]
    neural = consequences.loc[consequences.method.eq("s1_neural")]
    keys = ["strategy", "heldout_experiment", "cycle_name"]
    paired = neural.merge(ridge, on=keys, suffixes=("_neural", "_ridge"), validate="one_to_one")
    metrics = (
        ("maximum_relative_performance_loss_percent", "Maximum C/H loss [%]"),
        ("delta_c_percent", "Signed C change [%]"),
        ("delta_h_percent", "Signed H change [%]"),
    )
    records = []
    rng = np.random.default_rng(0)
    for strategy, rows in paired.groupby("strategy", sort=False):
        for field, label in metrics:
            left, right = f"{field}_neural", f"{field}_ridge"
            values = rows.loc[rows[left].notna() & rows[right].notna()].copy()
            values["difference"] = values[left] - values[right]
            experiments = values.groupby("heldout_experiment").difference.mean().to_numpy()
            draws = rng.choice(experiments, (2000, len(experiments)), replace=True).mean(axis=1)
            records.append({
                "strategy": strategy, "metric": field, "metric_label": label,
                "paired_cycles": len(values), "experiments": len(experiments),
                "mean_difference": experiments.mean(),
                "ci_low": np.quantile(draws, .025),
                "ci_high": np.quantile(draws, .975),
            })
        values = rows.assign(
            difference=rows.evaluation_status_neural.eq("no_trigger").astype(float)
            - rows.evaluation_status_ridge.eq("no_trigger").astype(float)
        )
        experiments = values.groupby("heldout_experiment").difference.mean().to_numpy()
        draws = rng.choice(experiments, (2000, len(experiments)), replace=True).mean(axis=1)
        records.append({
            "strategy": strategy, "metric": "no_trigger_fraction",
            "metric_label": "No-trigger fraction", "paired_cycles": len(values),
            "experiments": len(experiments), "mean_difference": experiments.mean(),
            "ci_low": np.quantile(draws, .025), "ci_high": np.quantile(draws, .975),
        })
    results = pd.DataFrame(records)
    results.to_csv(output / "performance_paired_s1_neural_vs_ridge.csv", index=False)
    figure, axes = plt.subplots(2, 4, figsize=(7.2, 3.7), squeeze=False)
    for axes_row, strategy in zip(axes, ("first_positive", "two_of_three"), strict=True):
        subset = results.loc[results.strategy.eq(strategy)]
        for axis, row in zip(axes_row, subset.itertuples(index=False), strict=True):
            axis.errorbar(
                row.mean_difference, 0,
                xerr=[[row.mean_difference - row.ci_low], [row.ci_high - row.mean_difference]],
                fmt="D", ms=3, color=STYLES["s1_neural"][0],
                ecolor="#777777", capsize=2,
            )
            axis.axvline(0, color="#333333", lw=.7, ls="--")
            axis.set(yticks=[], title=row.metric_label)
            axis.text(.03, .08, f"n={row.paired_cycles}", transform=axis.transAxes, fontsize=6)
        axes_row[0].set_ylabel(
            "First positive" if strategy == "first_positive" else "Two of three"
        )
    figure.supxlabel("RGB+sensor + Neural C/H minus RGB+sensor + Ridge C/H")
    figure.suptitle("Experiment-level paired bootstrap (95% interval)", fontsize=10)
    figure.tight_layout()
    _export(figure, output / "performance_paired_s1_neural_vs_ridge")
    return results


def _adapt(predictions):
    return predictions.assign(
        camera="front", image_feature="dinov2", classifier="pareto_boundary",
        input_feature=[_label(m, s) for m, s in predictions[IDENTITY].itertuples(index=False)],
        decision_score=predictions.logit,
    )


def evaluate_predictions(predictions, teachers, strategy="first_positive"):
    """Return cycle decisions, grouped frame metrics, and per-method/seed summaries."""
    cycle_rows, frame_rows = [], []
    for (method, seed, heldout), stream in predictions.groupby(
        [*IDENTITY, "heldout_experiment"], sort=True
    ):
        reference = teachers.loc[teachers.heldout_experiment.eq(heldout)]
        decisions = stream[["cycle_name", "teacher_time"]].drop_duplicates().rename(
            columns={"teacher_time": "selected_defrost_time"}
        ).assign(is_selected=True)
        errors = trigger_error_table(_adapt(stream), decisions, threshold=0.)
        errors = errors.loc[errors.strategy.eq(strategy)]
        reference = reference.set_index("cycle_name")
        evaluable_names = []
        for error in errors.to_dict("records"):
            cycle = stream.loc[stream.cycle_name.eq(error["cycle_name"])].sort_values("image_time")
            teacher_time = pd.to_datetime(cycle.teacher_time.iloc[0])
            native_times = pd.to_datetime(cycle.image_time)
            first = (native_times.iloc[0] - teacher_time).total_seconds() / 60
            last = (native_times.iloc[-1] - teacher_time).total_seconds() / 60
            relative = (native_times - teacher_time).dt.total_seconds() / 60
            nearest_pre = relative.loc[relative.le(0)].max()
            nearest_post = relative.loc[relative.ge(0)].min()
            observation_status = (
                "teacher_absent" if pd.isna(teacher_time) else
                "starts_after_knee" if first > 0 else
                "ends_before_knee" if last < 0 else "brackets_knee"
            )
            evaluation_included = observation_status == "brackets_knee"
            if evaluation_included:
                evaluable_names.append(error["cycle_name"])
            else:
                error.update(trigger_time=pd.NaT, trigger_error_minutes=np.nan)
            row = {
                **error, "method": method, "seed": seed, "heldout_experiment": heldout,
                "evaluation_included": evaluation_included,
                "persistent_high": bool(cycle.logit.ge(0).all()),
                "persistent_low": bool(cycle.logit.lt(0).all()),
                "trigger_domain": "no_trigger", "delta_c": np.nan, "delta_h": np.nan,
                "first_native_relative_teacher_minutes": first,
                "last_native_relative_teacher_minutes": last,
                "observation_status": observation_status,
                "observed_post_knee": bool(last > 0),
                "nearest_pre_knee_relative_minutes": nearest_pre,
                "nearest_post_knee_relative_minutes": nearest_post,
                "trigger_extra_observed_delay_minutes": (
                    error["trigger_error_minutes"] - nearest_post
                    if evaluation_included and error["trigger_error_minutes"] >= 0 else np.nan
                ),
            }
            covered = error["cycle_name"] in reference.index
            row["teacher_covered"] = covered
            knee = reference.loc[error["cycle_name"]] if covered else pd.Series({
                "economic_c": np.nan, "economic_h": np.nan,
                "cycle_cop_eligible_without_extrapolation": False,
                "cycle_heating_rate_kw_eligible_without_extrapolation": False,
            })
            row.update(teacher_c=knee.economic_c, teacher_h=knee.economic_h)
            if pd.notna(error["trigger_time"]):
                trigger = cycle.loc[cycle.image_time.eq(error["trigger_time"])].iloc[0]
                row.update(trigger_domain=_domain(trigger), trigger_c=trigger.economic_c,
                           trigger_h=trigger.economic_h)
                for short, objective in (("c", "cycle_cop"), ("h", "cycle_heating_rate_kw")):
                    quantities = [trigger[f"economic_{short}"], knee[f"economic_{short}"]]
                    if (_supported(trigger, objective) and _supported(knee, objective)
                            and np.isfinite(quantities).all()):
                        row[f"delta_{short}"] = quantities[0] - quantities[1]
            cycle_rows.append(row)
        hard = stream.loc[
            stream.cycle_name.isin(evaluable_names) & stream.target.isin([0, 1])
        ]
        frame_rows.append({
            "method": method, "seed": seed, "heldout_experiment": heldout,
            "frame_count": len(hard),
            **(classification_metrics(
                hard.target.astype(int), hard.prediction.astype(int), "binary"
            ) if len(hard) else {key: np.nan for key in (
                "accuracy", "balanced_accuracy", "macro_f1"
            )}),
        })
    cycles = pd.DataFrame(cycle_rows)
    summaries = []
    for (method, seed), group in cycles.groupby(IDENTITY, sort=True):
        evaluated = group.loc[group.evaluation_included]
        errors = evaluated.trigger_error_minutes
        triggered = evaluated.trigger_time.notna()
        trigger_domains = evaluated.loc[triggered, "trigger_domain"]
        bracketed_errors = evaluated.loc[triggered, "trigger_error_minutes"]
        summaries.append({
            "method": method, "seed": seed, "cycles": len(evaluated),
            "audited_cycles": len(group),
            "evaluation_cycles": len(evaluated),
            "teacher_covered_cycles": int(group.teacher_covered.sum()),
            "triggered_cycles": int(triggered.sum()),
            "no_trigger_fraction": float((~triggered).mean()),
            "median_error_minutes": _median(errors),
            "median_absolute_error_minutes": _median(errors.abs()),
            "p10_error_minutes": errors.quantile(.1), "p90_error_minutes": errors.quantile(.9),
            "outside_domain_trigger_fraction": trigger_domains.eq("outside_training_domain").mean(),
            "unknown_domain_trigger_fraction": trigger_domains.eq("unknown").mean(),
            "persistent_high_fraction": evaluated.persistent_high.mean(),
            "persistent_low_fraction": evaluated.persistent_low.mean(),
            "delta_c_evaluable_cycles": int(evaluated.delta_c.notna().sum()),
            "delta_h_evaluable_cycles": int(evaluated.delta_h.notna().sum()),
            "median_delta_c": _median(evaluated.delta_c),
            "median_delta_h": _median(evaluated.delta_h),
            **{f"{status}_cycles": int(group.observation_status.eq(status).sum()) for status in (
                "teacher_absent", "starts_after_knee", "ends_before_knee", "brackets_knee"
            )},
            "brackets_knee_trigger_fraction": triggered.mean(),
            "brackets_knee_median_error_minutes": _median(bracketed_errors),
            "brackets_knee_median_absolute_error_minutes": _median(bracketed_errors.abs()),
        })
    return cycles, pd.DataFrame(frame_rows), pd.DataFrame(summaries)


def _design(output, methods):
    figure, axis = plt.subplots(figsize=(7.2, 3.2))
    axis.axis("off")
    if set(methods) & {"s0", "s1", "s1_neural", "s2", "s3", "s4", "n2"}:
        recipes = (
            ("s0", .03, .57, "Outcome z\nBCE"),
            ("s1", .35, .57, "z + current C,H\nBCE"),
            ("s2", .67, .57, "z + C,H trajectories\nBCE"),
            ("s3", .03, .15, "z + C,H,O trajectories\nBCE"),
            ("s4", .35, .15, "Inner-selected input\nBCE + signed relation"),
            ("n2", .67, .15, "Sensor-only z + C,H trajectories\nBCE"),
            ("s1_neural", .35, .15, "z + neural current C,H\nBCE"),
        )
        title = "Frozen outcome representation → lightweight stopping heads"
        footer = "First s ≥ 0 is primary; two positives within three frames is secondary."
    else:
        recipes = (
            ("baseline", .05, .56, "Economic OFF · relation OFF\nBCE"),
            ("economic", .54, .56, "Economic ON · relation OFF\nBCE"),
            ("relation", .05, .15, "Economic OFF · relation ON\nBCE + relation rank"),
            ("combined", .54, .15, "Economic ON · relation ON\nBCE + relation rank"),
        )
        title = "Two factors, one model"
        footer = "First s ≥ 0 is primary; two positives within three frames is secondary."
    for method, x, y, text in recipes:
        if method not in methods:
            continue
        axis.text(x, y, f"{method.capitalize()}\n{text}", transform=axis.transAxes,
                  fontsize=9, va="bottom", color=STYLES[method][0],
                  bbox={"facecolor": "#F5F5F5", "edgecolor": "none", "pad": 10})
    axis.set_title(title, fontsize=11)
    figure.text(.5, .01, footer + "\nG, encoder and preprocessing are experiment-isolated.",
                ha="center", fontsize=7)
    _export(figure, output / "experiment_design")


def _loss_plot(losses, output):
    settings = _groups(losses)
    stopping = {"s0", "s1", "s1_neural", "s2", "s3", "s4", "n2"}
    if set(losses.method) & stopping:
        columns = min(3, len(settings))
        row_count = int(np.ceil(len(settings) / columns))
        figure, axes = plt.subplots(
            row_count, columns, figsize=(7.2, 2.5 * row_count), squeeze=False
        )
        for axis, ((method, _seed), rows) in zip(axes.flat, settings, strict=False):
            for split, color in (("train_side", "#999999"),
                                 ("val_side", STYLES[method][0])):
                for index, (_, fold) in enumerate(
                    rows.loc[rows.split.eq(split)].groupby("heldout_experiment")
                ):
                    axis.plot(fold.epoch, fold.loss, color=color, lw=.7, alpha=.4,
                              label=split.replace("_side", "") if index == 0 else "_nolegend_")
            axis.set(title=method.upper(), xlabel="Epoch", ylabel="BCE")
            axis.legend(frameon=False, fontsize=6)
        for axis in axes.flat[len(settings):]:
            axis.axis("off")
        figure.suptitle("Stopping-head inner training and held-out validation", fontsize=10)
        figure.tight_layout()
        _export(figure, output / "losses")
        return
    figure, axes = plt.subplots(len(settings), 3, figsize=(7.2, 2 * len(settings)), squeeze=False)
    for axes_row, ((method, seed), rows) in zip(axes, settings, strict=True):
        for axis, split in zip(axes_row, ("train_side", "val_side", "train_rank"), strict=True):
            for _, fold in rows.loc[rows.split.eq(split)].groupby("heldout_experiment"):
                axis.plot(fold.epoch, fold.loss, color=STYLES[method][0], lw=.7, alpha=.45)
            title = split.replace("_", " ")
            if split == "train_rank":
                title += "\n" + ("optimized" if method in {"relation", "combined", "s4"}
                                  else "diagnostic only")
            axis.set(xlabel="Epoch", title=title)
        axes_row[0].set_ylabel(_label(method, seed))
    figure.suptitle("Grouped inner loss: each line is one held-out-experiment run", fontsize=10)
    figure.tight_layout()
    _export(figure, output / "losses")


def render_neural_sensitivity(candidates, output):
    """Show how one frozen neural event surrogate propagates into C/H and knees."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grid = candidates.loc[candidates.is_teacher_candidate].copy()
    comparisons = (
        ("defrost_event_electricity_kwh", "neural_defrost_event_electricity_kwh",
         "Event electricity [kWh]"),
        ("defrost_event_net_heat_kwh", "neural_defrost_event_net_heat_kwh",
         "Event net heat [kWh]"),
        ("defrost_event_compressor_electricity_kwh",
         "neural_defrost_event_compressor_electricity_kwh", "Compressor electricity [kWh]"),
        ("defrost_event_duration_minutes", "neural_defrost_event_duration_minutes",
         "Event duration [min]"),
        ("cycle_cop", "neural_cycle_cop", "Cycle COP [−]"),
        ("cycle_heating_rate_kw", "neural_cycle_heating_rate_kw", "Heating rate [kW]"),
    )
    source_columns = [
        "row_id", "cycle_name", "experiment_id", "heldout_experiment",
        "candidate_defrost_time", "is_teacher_candidate",
        *[name for pair in comparisons for name in pair[:2]],
    ]
    grid[source_columns].to_csv(output / "neural_sensitivity_source.csv", index=False)
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    for axis, (ridge_name, neural_name, label) in zip(
        axes.flat, comparisons, strict=True
    ):
        pair = grid[[ridge_name, neural_name]].apply(pd.to_numeric, errors="coerce").dropna()
        if pair.empty:
            axis.text(.5, .5, "No common finite points", ha="center", va="center",
                      transform=axis.transAxes, fontsize=7)
        else:
            low = float(pair.min().min())
            high = float(pair.max().max())
            axis.scatter(
                pair[ridge_name], pair[neural_name], s=1.2, alpha=.18,
                color="#3B7E9B", linewidths=0, rasterized=True,
            )
            axis.plot([low, high], [low, high], color="#777777", lw=.7, ls="--")
            difference = pair[neural_name] - pair[ridge_name]
            axis.text(
                .03, .97,
                f"n={len(pair):,}\nmedian |Δ|={difference.abs().median():.3g}",
                transform=axis.transAxes, va="top", fontsize=6.5,
            )
        axis.set(xlabel=f"Ridge {label}", ylabel=f"Neural {label}")
    figure.suptitle(
        "Frozen event surrogate: numerical differences before Pareto selection", fontsize=10
    )
    figure.tight_layout()
    _export(figure, output / "neural_outcome_propagation")

    keys = ["cycle_name", "experiment_id", "heldout_experiment"]
    knees = grid.groupby(keys, sort=True).agg(
        ridge_knee=("teacher_time", "first"),
        neural_knee=("neural_teacher_time", "first"),
    ).reset_index()
    for name in ("ridge_knee", "neural_knee"):
        knees[name] = pd.to_datetime(knees[name], errors="coerce")
    ridge_exists, neural_exists = knees.ridge_knee.notna(), knees.neural_knee.notna()
    knees["selection_status"] = np.select(
        [ridge_exists & neural_exists, ridge_exists, neural_exists],
        ["both", "ridge_only", "neural_only"], default="neither",
    )
    knees["knee_difference_minutes"] = (
        knees.neural_knee - knees.ridge_knee
    ).dt.total_seconds() / 60
    knees["cycle_id"] = pd.to_numeric(
        knees.cycle_name.astype(str).str.extract(r"(\d+)$", expand=False), errors="coerce"
    )
    knees = knees.sort_values(["cycle_id", "cycle_name"], kind="stable").reset_index(drop=True)
    knees.to_csv(output / "neural_knees_by_cycle.csv", index=False)
    figure, axis = plt.subplots(figsize=(7.2, 3.2))
    _shade_experiment_dates(axis, knees.experiment_id.astype(str).tolist())
    difference = knees.knee_difference_minutes
    finite = difference.dropna()
    limit = max(5.0, float(finite.abs().quantile(.9))) if len(finite) else 5.0
    shown = difference.clip(-.96 * limit, .96 * limit)
    selected = difference.notna()
    axis.scatter(
        np.arange(len(knees))[selected], shown[selected], s=18,
        color="#3B7E9B", marker="D", zorder=3,
    )
    for index in np.flatnonzero(selected & difference.abs().gt(limit)):
        axis.annotate(
            "↑" if difference.iloc[index] > 0 else "↓",
            (index, shown.iloc[index]), ha="center", va="center", fontsize=7,
            color="#3B7E9B", clip_on=False,
        )
    counts = knees.selection_status.value_counts()
    axis.axhline(0, color="#333333", lw=.8)
    axis.set(
        ylim=(-limit, limit), xlabel="Cycle ID", ylabel="Neural − Ridge knee [min]",
        xticks=np.arange(len(knees)),
        xticklabels=[str(int(value)) if pd.notna(value) else name
                     for value, name in knees[["cycle_id", "cycle_name"]].itertuples(index=False)],
        title=(f"Independent selectors · both {counts.get('both', 0)} · "
               f"Ridge only {counts.get('ridge_only', 0)} · "
               f"Neural only {counts.get('neural_only', 0)} · neither {counts.get('neither', 0)}"),
    )
    axis.tick_params(axis="x", labelrotation=90, labelsize=5)
    axis.grid(axis="y", color="#DDDDDD", lw=.45)
    figure.tight_layout()
    _export(figure, output / "neural_pareto_knee_difference")
    return knees


def render_neural_paired_comparison(cycles, output):
    """Summarize S1-neural minus matched S1 from saved held-out predictions."""
    output = Path(output)
    reference = cycles.loc[cycles.method.eq("s1")]
    neural = cycles.loc[cycles.method.eq("s1_neural")]
    paired = neural.merge(
        reference, on=["heldout_experiment", "cycle_name", "seed"],
        suffixes=("_neural", "_ridge"), validate="one_to_one",
    )
    methods = []
    for method, rows in cycles.groupby("method", sort=False):
        evaluated = rows.loc[rows.evaluation_included]
        error = evaluated.trigger_error_minutes
        methods.append({
            "method": method, "cycles": len(evaluated),
            "triggered_cycles": int(error.notna().sum()),
            "median_signed_error_minutes": error.median(),
            "median_absolute_error_minutes": error.abs().median(),
            "p90_absolute_error_minutes": error.abs().quantile(.9),
            "no_trigger_fraction": error.isna().mean(),
        })
    pd.DataFrame(methods).to_csv(output / "neural_method_summary.csv", index=False)

    definitions = (
        ("absolute_timing", "trigger_error_minutes", True, "Absolute timing error [min]"),
        ("signed_timing", "trigger_error_minutes", False, "Signed timing error [min]"),
        ("absolute_delta_c", "delta_c", True, "Absolute Δ cycle COP [−]"),
        ("absolute_delta_h", "delta_h", True, "Absolute Δ heating rate [kW]"),
    )
    records, rng = [], np.random.default_rng(0)
    for metric, field, absolute, _label_text in definitions:
        left, right = f"{field}_neural", f"{field}_ridge"
        valid = paired[left].notna() & paired[right].notna()
        values = paired.loc[valid, ["heldout_experiment", left, right]].copy()
        new = values[left].abs() if absolute else values[left]
        old = values[right].abs() if absolute else values[right]
        values["difference"] = new - old
        experiments = values.groupby("heldout_experiment").difference.mean().to_numpy()
        draws = (
            rng.choice(experiments, (2000, len(experiments)), replace=True).mean(axis=1)
            if len(experiments) else np.array([np.nan])
        )
        records.append({
            "metric": metric, "paired_cycles": len(values), "experiments": len(experiments),
            "mean_difference": experiments.mean() if len(experiments) else np.nan,
            "ci_low": np.nanquantile(draws, .025), "ci_high": np.nanquantile(draws, .975),
        })
    valid = paired.evaluation_included_neural & paired.evaluation_included_ridge
    misses = paired.loc[valid, ["heldout_experiment", "trigger_time_neural",
                                "trigger_time_ridge"]].copy()
    misses["difference"] = (
        misses.trigger_time_neural.isna().astype(float)
        - misses.trigger_time_ridge.isna().astype(float)
    )
    experiments = misses.groupby("heldout_experiment").difference.mean().to_numpy()
    draws = rng.choice(experiments, (2000, len(experiments)), replace=True).mean(axis=1)
    records.append({
        "metric": "no_trigger_fraction", "paired_cycles": len(misses),
        "experiments": len(experiments), "mean_difference": experiments.mean(),
        "ci_low": np.quantile(draws, .025), "ci_high": np.quantile(draws, .975),
    })
    results = pd.DataFrame(records)
    results.to_csv(output / "neural_paired_comparison.csv", index=False)
    labels = {name: label for name, _, _, label in definitions}
    labels["no_trigger_fraction"] = "No-trigger fraction [−]"
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    for axis, row in zip(axes.flat, results.itertuples(index=False), strict=False):
        axis.errorbar(
            row.mean_difference, 0,
            xerr=[[row.mean_difference - row.ci_low], [row.ci_high - row.mean_difference]],
            fmt="D", color=STYLES["s1_neural"][0], ecolor="#777777", capsize=2,
        )
        axis.axvline(0, color="#333333", lw=.7, ls="--")
        axis.set(yticks=[], xlabel="S1-neural − S1", title=labels[row.metric])
        if row.metric in {"absolute_delta_c", "absolute_delta_h"}:
            axis.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=True)
        axis.text(.03, .08, f"cycles {row.paired_cycles} · experiments {row.experiments}",
                  transform=axis.transAxes, fontsize=6.5)
    for axis in axes.flat[len(results):]:
        axis.axis("off")
    figure.suptitle(
        "Matched held-out effects; experiment-level bootstrap intervals", fontsize=10
    )
    figure.tight_layout()
    _export(figure, output / "neural_paired_comparison")
    return results


def _summary_plot(cycles, output, strategy="first_positive", stem="decision_summary"):
    settings = _groups(cycles)
    figure, axes = plt.subplots(1, 3, figsize=(9, 3.7))
    for axis, field, title in zip(
        axes, ("trigger_error_minutes", "delta_c", "delta_h"),
        ("Trigger − knee [min]", "Δ cycle COP [−]", "Δ heating rate [kW]"), strict=True
    ):
        values_by_method = [rows[field].dropna().to_numpy() for _, rows in settings]
        axis.boxplot(
            values_by_method, positions=range(len(settings)), widths=.46, showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": "#777777", "linewidth": .8},
            medianprops={"color": "#222222", "linewidth": 1.2},
            whiskerprops={"color": "#777777", "linewidth": .7},
            capprops={"color": "#777777", "linewidth": .7},
        )
        for index, ((method, _seed), rows) in enumerate(settings):
            values = rows[field].dropna().to_numpy()
            x = index + np.linspace(-.14, .14, len(values))
            axis.scatter(x, values, s=12, color=STYLES[method][0], alpha=.65)
        axis.axhline(0, color="#777777", lw=.7)
        labels = []
        for key, group in settings:
            evaluated = group.loc[group.evaluation_included]
            labels.append(
                _label(*key) + f"\nmiss {evaluated.trigger_time.isna().sum()}/{len(evaluated)}"
                + f"; n {len(evaluated)}"
            )
        axis.set(title=title, xticks=range(len(settings)), xticklabels=labels)
        axis.tick_params(axis="x", rotation=35, labelsize=6)
    controller = (
        "First-positive native-frame activation" if strategy == "first_positive"
        else "Two positives within three native frames"
    )
    figure.suptitle(f"{controller}; C and H are evaluated separately", fontsize=10)
    figure.text(
        .5, .01,
        "Boxes: IQR and median; dots: evaluable cycles. Incomplete streams are excluded.\n"
        "C/H differences require evaluable, in-domain trigger and exact knee.",
        ha="center", fontsize=7,
    )
    figure.tight_layout(rect=(0, .10, 1, .95))
    _export(figure, output / stem)


def _outside_legend(axis, columns):
    if axis.get_legend_handles_labels()[0]:
        axis.legend(
            loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0,
            frameon=False, fontsize=6, ncol=columns,
        )


def _cycle_figures(
    predictions,
    cycles,
    output,
    teacher_curves=None,
    rb_triggers=None,
    local_window_minutes=15,
    strategy="first_positive",
    comparison_curves=None,
    repeat_trigger_legends=True,
):
    names = sorted(predictions.cycle_name.dropna().unique())
    columns = 5 if comparison_curves is not None else 3 if teacher_curves is not None else 1
    cycle_output = output / "cycles"
    cycle_output.mkdir(parents=True, exist_ok=True)
    for path in cycle_output.glob("*.png"):
        path.unlink()
    predictions.to_csv(output / "cycle_streams.csv", index=False)
    pareto_sources = []
    for name in names:
        if columns == 5:
            figure = plt.figure(figsize=(7.2, 7.4))
            layout = figure.add_gridspec(3, 2, height_ratios=(.75, 1, 1), hspace=.72,
                                         wspace=.48)
            axes = np.array([[
                figure.add_subplot(layout[0, :]),
                figure.add_subplot(layout[1, 0]), figure.add_subplot(layout[1, 1]),
                figure.add_subplot(layout[2, 0]), figure.add_subplot(layout[2, 1]),
            ]])
        elif columns == 3:
            figure = plt.figure(figsize=(7.2, 5.0))
            layout = figure.add_gridspec(2, 2, height_ratios=(.85, 1), hspace=.68, wspace=.48)
            axes = np.array([[
                figure.add_subplot(layout[0, :]),
                figure.add_subplot(layout[1, 0]),
                figure.add_subplot(layout[1, 1]),
            ]])
        else:
            figure, axes = plt.subplots(1, 1, figsize=(7.2, 2.7), squeeze=False)
        axis = axes[0, 0]
        selected = predictions.loc[predictions.cycle_name.eq(name)]
        cycle_rows = cycles.loc[cycles.cycle_name.eq(name)]
        included = bool(cycle_rows.evaluation_included.any())
        status = str(cycle_rows.observation_status.iloc[0])
        origin = pd.Timestamp(selected.heating_start.iloc[0])
        stable = (pd.Timestamp(selected.stable_heating_start.iloc[0]) - origin).total_seconds() / 60
        end = (pd.to_datetime(selected.image_time).max() - origin).total_seconds() / 60
        axis.axvspan(0, stable, color="#78A6BC", alpha=.15)
        axis.axvspan(stable, end, color="#F2A35E", alpha=.10)
        triggers = {}
        for (method, seed), rows in _groups(selected):
            rows = rows.sort_values("image_time")
            times = pd.to_datetime(rows.image_time)
            minute = (times - origin).dt.total_seconds() / 60
            segments = times.diff().gt(pd.Timedelta(seconds=45)).cumsum()
            for segment, chunk in rows.groupby(segments, sort=False):
                axis.plot(minute.loc[chunk.index], chunk.logit, color=STYLES[method][0],
                          lw=.85, marker=".", markersize=2,
                          label=_label(method, seed) if segment == 0 else "_nolegend_")
            if strategy == "first_positive":
                positive = rows.logit.ge(0)
                trigger = times.loc[positive].iloc[0] if positive.any() else pd.NaT
            else:
                trigger, _ = two_of_three_trigger(times, rows.logit, threshold=0.)
            triggers[(method, seed)] = trigger if included else pd.NaT
            if included and pd.notna(trigger):
                axis.axvline(
                    (trigger - origin).total_seconds() / 60,
                    color=STYLES[method][0], lw=.8, ls=":"
                )
        rb_time = pd.NaT
        if rb_triggers is not None:
            rb = rb_triggers.loc[rb_triggers.cycle_name.eq(name)]
            if not rb.empty and ("rb_status" not in rb or str(rb.rb_status.iloc[0]) == "triggered"):
                rb_time = pd.to_datetime(rb.t_RB.iloc[0], errors="coerce")
        if pd.notna(rb_time):
            axis.axvline(
                (rb_time - origin).total_seconds() / 60,
                color="#2E7D5B", lw=1, ls="-.", label="RB trigger",
            )
        if included and pd.notna(selected.teacher_time.iloc[0]):
            knee = (pd.Timestamp(selected.teacher_time.iloc[0]) - origin).total_seconds() / 60
            axis.axvline(knee, color="black", ls="--", lw=1, label="Teacher knee")
        elif not included:
            axis.text(
                .99, .04, f"Excluded from evaluation: {status.replace('_', ' ')}",
                transform=axis.transAxes, ha="right", va="bottom", fontsize=6.5,
                color="#9A3412",
            )
        axis.axhline(0, color="#777777", lw=.7)
        axis.set(xlabel="Time from cycle start [min]", ylabel="Raw decision logit")
        axis.set_title("First-positive decision score", loc="left", pad=34, fontsize=8)
        if teacher_curves is not None:
            grid = teacher_curves.loc[
                teacher_curves.cycle_name.eq(name)
                & teacher_curves.heldout_experiment.eq(selected.heldout_experiment.iloc[0])
                & teacher_curves.is_teacher_candidate
            ].copy()
            grid["is_cop_heating_rate_pareto_point"] = grid.pareto_selection_score.notna()
            grid["is_selected_pareto_point"] = grid.is_knee & included
            fields = ["cycle_name", "heldout_experiment", "candidate_defrost_time",
                      "cycle_cop", "cycle_heating_rate_kw", "cycle_evaporator_capacity_kw",
                      "cycle_cop_eligible", "cycle_heating_rate_kw_eligible",
                      "pareto_selection_score", "is_knee",
                      "is_cop_heating_rate_pareto_point", "is_selected_pareto_point"]
            grid = grid.loc[:, grid.columns.intersection(fields)].copy()
            pareto_sources.append(
                grid.loc[:, grid.columns.intersection(fields)].assign(surrogate="ridge")
            )
            for panel, local in ((axes[0, 1], False), (axes[0, 2], True)):
                plot_cop_heating_rate_pareto(
                    panel,
                    grid,
                    origin,
                    rb_time=rb_time,
                    local=local,
                    local_window_minutes=local_window_minutes,
                )
                panel.set_box_aspect(None)
                title = panel.get_title(loc="left")
                panel.set_title(title, loc="left", pad=34, fontsize=7)
                for index, ((method, seed), stream) in enumerate(_groups(selected)):
                    trigger = triggers[(method, seed)]
                    hit = stream.loc[pd.to_datetime(stream.image_time).eq(trigger)].head(1)
                    if not hit.empty and np.isfinite(hit[["economic_c", "economic_h"]]).all().all():
                        x, y = hit[["economic_c", "economic_h"]].iloc[0]
                        shown_x, shown_y, direction = (
                            clip_point_to_axes(
                                panel, x, y, pad_fraction=.05 + .025 * index
                            )
                            if local else (x, y, "")
                        )
                        panel.scatter(
                            shown_x, shown_y, color=STYLES[method][0], marker=STYLES[method][1],
                            s=45, edgecolors="white", linewidths=.6, zorder=6,
                            label=_label(method, seed) if repeat_trigger_legends else "_nolegend_",
                        )
                        if local and direction:
                            panel.annotate(
                                direction, (shown_x, shown_y), xytext=(0, 7),
                                textcoords="offset points", ha="center", fontsize=7,
                                fontweight="bold", color=STYLES[method][0], clip_on=False,
                            )
                        if not local:
                            for value, getter, setter in (
                                (x, panel.get_xlim, panel.set_xlim),
                                (y, panel.get_ylim, panel.set_ylim),
                            ):
                                low, high = getter()
                                span = max(high, value) - min(low, value)
                                setter(min(low, value - .03 * span),
                                       max(high, value + .03 * span))
                _outside_legend(panel, 4)
        if comparison_curves is not None:
            neural = comparison_curves.loc[
                comparison_curves.cycle_name.eq(name)
                & comparison_curves.heldout_experiment.eq(selected.heldout_experiment.iloc[0])
                & comparison_curves.is_teacher_candidate
            ].copy()
            neural = neural.drop(columns=neural.columns.intersection([
                "cycle_cop", "cycle_heating_rate_kw", "cycle_evaporator_capacity_kw",
                "cycle_cop_eligible", "cycle_heating_rate_kw_eligible",
                "is_cop_heating_rate_pareto_point", "is_selected_pareto_point",
                "pareto_selection_score",
            ]))
            neural = neural.rename(columns={
                "neural_cycle_cop": "cycle_cop",
                "neural_cycle_heating_rate_kw": "cycle_heating_rate_kw",
                "neural_cycle_evaporator_capacity_kw": "cycle_evaporator_capacity_kw",
                "neural_cycle_cop_eligible": "cycle_cop_eligible",
                "neural_cycle_heating_rate_kw_eligible": "cycle_heating_rate_kw_eligible",
                "neural_is_pareto": "is_cop_heating_rate_pareto_point",
                "neural_is_knee": "is_selected_pareto_point",
                "neural_pareto_selection_score": "pareto_selection_score",
            })
            neural = neural.loc[:, neural.columns.intersection([
                *fields, "is_cop_heating_rate_pareto_point", "is_selected_pareto_point",
            ])].copy()
            pareto_sources.append(neural.assign(surrogate="neural"))
            center = pd.to_datetime(selected.teacher_time.iloc[0], errors="coerce")
            for panel, local in ((axes[0, 3], False), (axes[0, 4], True)):
                plot_cop_heating_rate_pareto(
                    panel, neural, origin, local=local,
                    local_window_minutes=local_window_minutes,
                    local_center_time=center if local else None,
                )
                panel.set_box_aspect(None)
                title = panel.get_title(loc="left")
                panel.set_title("Neural · " + title, loc="left", pad=34, fontsize=7)
                for index, ((method, seed), _stream) in enumerate(_groups(selected)):
                    trigger = triggers[(method, seed)]
                    hit = neural.loc[
                        pd.to_datetime(neural.candidate_defrost_time).eq(trigger)
                    ].head(1)
                    if not hit.empty and np.isfinite(
                        hit[["cycle_cop", "cycle_heating_rate_kw"]]
                    ).all().all():
                        x, y = hit[["cycle_cop", "cycle_heating_rate_kw"]].iloc[0]
                        shown_x, shown_y, direction = (
                            clip_point_to_axes(panel, x, y, pad_fraction=.05 + .025 * index)
                            if local else (x, y, "")
                        )
                        panel.scatter(
                            shown_x, shown_y, color=STYLES[method][0],
                            marker=STYLES[method][1], s=45, edgecolors="white",
                            linewidths=.6, zorder=6, label=_label(method, seed),
                        )
                        if local and direction:
                            panel.annotate(
                                direction, (shown_x, shown_y), xytext=(0, 7),
                                textcoords="offset points", ha="center", fontsize=7,
                                fontweight="bold", color=STYLES[method][0], clip_on=False,
                            )
                _outside_legend(panel, 4)
        _outside_legend(axis, 4)
        for label, panel in zip("abcde"[:columns], axes[0], strict=True):
            panel.text(-.1, 1, label, transform=panel.transAxes,
                       fontsize=8, fontweight="bold", va="bottom")
        figure.suptitle(name, fontsize=8, fontweight="bold", y=.98)
        if columns == 5:
            figure.subplots_adjust(left=.09, right=.91, bottom=.07, top=.89)
        elif columns == 3:
            figure.subplots_adjust(left=.09, right=.91, bottom=.09, top=.84)
        else:
            figure.tight_layout(rect=(0, 0, 1, .86), pad=.4)
        _export(figure, cycle_output / name)
    (pd.concat(pareto_sources, ignore_index=True) if pareto_sources else pd.DataFrame()).to_csv(
        output / "pareto_candidates.csv", index=False
    )
    trigger_fields = [
        "method", "seed", "heldout_experiment", "cycle_name", "trigger_time",
        "trigger_c", "trigger_h", "trigger_domain", "evaluation_included",
        "observation_status",
    ]
    cycles.loc[:, cycles.columns.intersection(trigger_fields)].to_csv(
        output / "pareto_triggers.csv", index=False
    )


def _pair_order_plot(pairs, output):
    selected = pairs.loc[pairs.split.isin(["outer_train", "outer_test"])
                         & pairs.pairs.gt(0)].dropna(subset=["pair_order_accuracy"])
    if selected.empty:
        return
    figure, axis = plt.subplots(figsize=(7.2, 3.5))
    settings = _groups(selected)
    for index, ((method, _seed), rows) in enumerate(settings):
        for offset, split, marker in ((-.15, "outer_train", "o"), (.15, "outer_test", "^")):
            values = rows.loc[rows.split.eq(split)].groupby(
                "heldout_experiment").pair_order_accuracy.mean()
            axis.scatter(np.full(len(values), index + offset), values, marker=marker,
                         color=STYLES[method][0], s=18, alpha=.65)
            axis.plot([index + offset - .08, index + offset + .08], [values.mean()] * 2,
                      color="black", lw=1.4)
            axis.text(index + offset, 1.03, f"n={len(values)}", ha="center", fontsize=6)
    axis.axhline(.5, color="#999999", ls="--", lw=.8)
    axis.set(ylim=(0, 1.1), ylabel="Same-side pair-order accuracy", xticks=range(len(settings)),
             xticklabels=[_label(*key) for key, _ in settings],
             title="Does improved training order generalize to unseen experiments?")
    axis.tick_params(axis="x", labelsize=7)
    figure.text(.5, .01, "Left circles: outer train; right triangles: outer test. "
                "Black: fold-equal mean (test: held-out experiments); 0.5: chance reference.",
                ha="center", fontsize=7)
    figure.tight_layout(rect=(0, .06, 1, 1))
    _export(figure, output / "pair_order_diagnostics")


def render_outcome_figures(predictions, losses, output):
    """Compare fold-specific outcome representations with the frozen Ridge teacher."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    targets = list(OUTCOME_TARGETS.values())
    records = []
    grouping = ["representation", *( ["seed"] if "seed" in predictions else [])]
    for identity, rows in predictions.groupby(grouping):
        identity = identity if isinstance(identity, tuple) else (identity,)
        representation = identity[0]
        seed = identity[1] if len(identity) > 1 else None
        for target in targets:
            residual = rows[f"predicted_{target}"] - rows[target]
            by_experiment = residual.groupby(rows.experiment_id)
            records.append({
                "model": representation, "target": target, "events": len(residual),
                **({"seed": seed} if seed is not None else {}),
                "mae": residual.abs().mean(), "rmse": np.sqrt(np.square(residual).mean()),
                "experiment_macro_mae": by_experiment.apply(lambda x: x.abs().mean()).mean(),
                "experiment_macro_rmse": by_experiment.apply(
                    lambda x: np.sqrt(np.square(x).mean())
                ).mean(),
            })
    ridge_rows = predictions.loc[predictions.representation.eq("visual")]
    for target in targets:
        field = f"ridge_predicted_{target}"
        if field not in ridge_rows:
            continue
        residual = ridge_rows[field] - ridge_rows[target]
        by_experiment = residual.groupby(ridge_rows.experiment_id)
        records.append({
            "model": "ridge", "target": target, "events": len(residual),
            "mae": residual.abs().mean(), "rmse": np.sqrt(np.square(residual).mean()),
            "experiment_macro_mae": by_experiment.apply(lambda x: x.abs().mean()).mean(),
            "experiment_macro_rmse": by_experiment.apply(
                lambda x: np.sqrt(np.square(x).mean())
            ).mean(),
        })
    pd.DataFrame(records).to_csv(output / "outcome_metrics.csv", index=False)

    colors = {
        "visual": "#2E7D5B", "nonvisual": "#8A6FA8", "ridge": "#777777",
        **{name: style[0] for name, style in VCNET_STYLES.items()},
    }
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    for axis, target in zip(axes.flat, targets, strict=True):
        observed = predictions[target]
        bounds = [observed.min(), observed.max()]
        for identity, rows in predictions.groupby(grouping):
            identity = identity if isinstance(identity, tuple) else (identity,)
            representation, *seed = identity
            color = colors.get(representation, "#777777")
            label = VCNET_LABELS.get(representation, representation)
            if seed:
                label = f"{label} · seed {seed[0]}"
            axis.scatter(
                rows[target], rows[f"predicted_{target}"], s=11, alpha=.5,
                facecolors=color if not seed or seed[0] == 0 else "white",
                edgecolors=color, linewidths=.5, label=label,
            )
        if f"ridge_predicted_{target}" in ridge_rows:
            axis.scatter(ridge_rows[target], ridge_rows[f"ridge_predicted_{target}"],
                         s=11, alpha=.5, color=colors["ridge"], label="ridge")
        axis.plot(bounds, bounds, color="black", lw=.7, ls="--")
        axis.set(xlabel="Observed", ylabel="Predicted", title=OUTCOME_LABELS[target])
    axes.flat[0].legend(frameon=False, fontsize=7)
    seed_text = ""
    if "seed" in predictions:
        seed_text = f" · {predictions.seed.nunique()} fixed seeds"
    figure.suptitle(
        f"Actual-event outcome prediction: {predictions.event_id.nunique()} events · "
        f"{predictions.experiment_id.nunique()} held-out experiments{seed_text}", fontsize=10,
    )
    figure.tight_layout()
    _export(figure, output / "outcome_prediction")

    settings = losses[["representation", *( ["seed"] if "seed" in losses else [])]]
    settings = list(settings.drop_duplicates().itertuples(index=False, name=None))
    columns = min(5, len(settings))
    rows = int(np.ceil(len(settings) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.35 * columns, 3 * rows),
                                squeeze=False)
    for axis, setting in zip(axes.flat, settings, strict=True):
        representation, *seed = setting
        selected = losses.loc[losses.representation.eq(representation)]
        if seed:
            selected = selected.loc[selected.seed.eq(seed[0])]
        for split, color in (
            ("train", "#777777"),
            ("validation", colors.get(representation, "#3B7E9B")),
        ):
            for index, (_, fold) in enumerate(
                selected.loc[selected.split.eq(split)].groupby("heldout_experiment")
            ):
                axis.plot(fold.epoch, fold.loss, color=color, lw=.7, alpha=.4,
                          label=split if index == 0 else "_nolegend_")
        title = VCNET_LABELS.get(representation, representation)
        if seed:
            title = f"{title}\nseed {seed[0]}"
        axis.set(title=title, xlabel="Epoch",
                 ylabel="Standardized four-target MSE")
        axis.legend(frameon=False, fontsize=7)
    for axis in axes.flat[len(settings):]:
        axis.remove()
    figure.suptitle("Outcome representation learning", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, .97))
    _export(figure, output / "outcome_training_loss")


def render_paired_comparisons(cycles, selected_inputs, output):
    """Experiment-paired intervals from saved predictions; negative favors the candidate."""
    references = {"s1": "s0", "s2": "s1", "s3": "s2", "s2_n2": "n2"}
    records = []
    rng = np.random.default_rng(0)
    for candidate, key in (("s1", "s1"), ("s2", "s2"), ("s3", "s3"), ("s4", "s4"),
                           ("s2", "s2_n2")):
        candidate_rows = cycles.loc[cycles.method.eq(candidate)]
        if candidate == "s4":
            candidate_rows = candidate_rows.merge(
                selected_inputs[["heldout_experiment", "selected_for_s4"]],
                on="heldout_experiment", validate="many_to_one",
            )
            reference_rows = pd.concat([
                cycles.loc[
                    cycles.heldout_experiment.eq(heldout) & cycles.method.eq(reference)
                ]
                for heldout, reference in selected_inputs[
                    ["heldout_experiment", "selected_for_s4"]
                ].itertuples(index=False)
            ], ignore_index=True)
            label = "S4 − inner-selected"
        else:
            reference = references[key]
            reference_rows = cycles.loc[cycles.method.eq(reference)]
            label = f"{candidate.upper()} − {reference.upper()}"
        paired = candidate_rows.merge(
            reference_rows,
            on=["heldout_experiment", "cycle_name", "seed"], suffixes=("_new", "_ref"),
            validate="one_to_one",
        )
        for metric, field in (
            ("absolute_trigger_error_minutes", "trigger_error_minutes"),
            ("absolute_delta_c", "delta_c"), ("absolute_delta_h", "delta_h"),
        ):
            valid = paired[f"{field}_new"].notna() & paired[f"{field}_ref"].notna()
            values = paired.loc[valid].copy()
            values["difference"] = (
                values[f"{field}_new"].abs() - values[f"{field}_ref"].abs()
            )
            experiment = values.groupby("heldout_experiment").difference.mean().to_numpy()
            if len(experiment):
                draws = rng.choice(experiment, (2000, len(experiment)), replace=True).mean(axis=1)
                low, high = np.quantile(draws, [.025, .975])
            else:
                low = high = np.nan
            records.append({
                "comparison": label, "metric": metric, "paired_cycles": len(values),
                "experiments": len(experiment), "mean_difference": experiment.mean()
                if len(experiment) else np.nan, "ci_low": low, "ci_high": high,
            })
    results = pd.DataFrame(records)
    results.to_csv(Path(output) / "paired_comparisons.csv", index=False)
    timing = results.loc[results.metric.eq("absolute_trigger_error_minutes")]
    figure, axis = plt.subplots(figsize=(7.2, 3.2))
    positions = np.arange(len(timing))
    axis.errorbar(
        timing.mean_difference, positions,
        xerr=[timing.mean_difference - timing.ci_low,
              timing.ci_high - timing.mean_difference],
        fmt="o", color="#365A83", ecolor="#777777", capsize=2,
    )
    axis.axvline(0, color="black", lw=.7, ls="--")
    axis.set(yticks=positions, yticklabels=timing.comparison,
             xlabel="Paired change in absolute trigger error [min]",
             title="Held-out experiment bootstrap; negative indicates improvement")
    figure.tight_layout()
    _export(figure, Path(output) / "paired_comparisons")


def _stream_stability(stream, local_window_minutes):
    stream = stream.sort_values("image_time", kind="stable")
    time = pd.to_datetime(stream.image_time)
    teacher = pd.Timestamp(stream.teacher_time.iloc[0])
    local = time.between(
        teacher - pd.Timedelta(minutes=local_window_minutes),
        teacher + pd.Timedelta(minutes=local_window_minutes),
    )
    paired = local & local.shift(fill_value=False)
    steps = stream.logit.diff().abs().loc[paired]
    if steps.empty:
        return {name: np.nan for name in (
            "median_absolute_step", "zero_crossings", "repeated_crossings",
            "local_total_variation",
        )}
    state = stream.logit.ge(0)
    flips = state.ne(state.shift()).loc[paired]
    positive = np.flatnonzero(state.to_numpy())
    positions = np.arange(len(stream))[paired.to_numpy()]
    repeated = flips.to_numpy() & (positions > positive[0]) if len(positive) else []
    return {
        "median_absolute_step": steps.median(),
        "zero_crossings": int(flips.sum()),
        "repeated_crossings": int(np.sum(repeated)),
        "local_total_variation": steps.sum(),
    }


def relation_stability_metrics(predictions, selected_inputs, local_window_minutes=15):
    """Compare S4 with its fold-matched BCE score on the fixed knee-local window."""
    records = []
    for heldout, reference in selected_inputs[
        ["heldout_experiment", "selected_for_s4"]
    ].itertuples(index=False):
        fold = predictions.loc[predictions.heldout_experiment.eq(heldout)]
        for (cycle_name, seed), relation in fold.loc[fold.method.eq("s4")].groupby(
            ["cycle_name", "seed"], sort=True
        ):
            matched = fold.loc[
                fold.method.eq(reference) & fold.cycle_name.eq(cycle_name) & fold.seed.eq(seed)
            ]
            values = {
                "matched": _stream_stability(matched, local_window_minutes),
                "s4": _stream_stability(relation, local_window_minutes),
            }
            record = {
                "heldout_experiment": heldout, "cycle_name": cycle_name, "seed": seed,
                "reference_method": reference,
            }
            for metric in values["s4"]:
                record[f"{metric}_matched"] = values["matched"][metric]
                record[f"{metric}_s4"] = values["s4"][metric]
                record[f"{metric}_difference"] = (
                    values["s4"][metric] - values["matched"][metric]
                )
            records.append(record)
    return pd.DataFrame(records)


def render_relation_stability(predictions, selected_inputs, output, local_window_minutes=15):
    """Plot cycle-paired S4 stability without retraining or choosing a new threshold."""
    paired = relation_stability_metrics(predictions, selected_inputs, local_window_minutes)
    output = Path(output)
    paired.to_csv(output / "relation_stability_by_cycle.csv", index=False)
    labels = {
        "median_absolute_step": r"Median frame step $|s_t-s_{t-1}|$",
        "zero_crossings": "Zero-boundary crossings",
        "repeated_crossings": "Repeated crossings after first trigger",
        "local_total_variation": r"Local total variation $\sum|s_t-s_{t-1}|$",
    }
    rng, summaries = np.random.default_rng(0), []
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    for axis, (metric, label) in zip(axes.flat, labels.items(), strict=True):
        selected = paired.dropna(subset=[f"{metric}_matched", f"{metric}_s4"])
        difference = selected[f"{metric}_difference"]
        experiment = difference.groupby(selected.heldout_experiment).mean().to_numpy()
        draws = rng.choice(experiment, (2000, len(experiment)), replace=True).mean(axis=1)
        low, high = np.quantile(draws, [.025, .975])
        summaries.append({
            "metric": metric, "paired_cycles": len(selected),
            "experiments": len(experiment), "mean_difference": experiment.mean(),
            "ci_low": low, "ci_high": high,
            "cycles_lower_fraction": difference.lt(0).mean(),
            "cycles_equal_fraction": difference.eq(0).mean(),
        })
        x, y = selected[f"{metric}_matched"], selected[f"{metric}_s4"]
        lower, upper = min(x.min(), y.min()), max(x.max(), y.max())
        axis.scatter(x, y, s=16, alpha=.48, color=STYLES["s4"][0], linewidths=0)
        axis.plot([lower, upper], [lower, upper], color="#777777", lw=.8, ls="--")
        axis.set(
            xlabel="Matched BCE", ylabel="S4 relation", title=label,
        )
        axis.text(
            .03, .97, f"Δ={experiment.mean():.3g} [{low:.3g}, {high:.3g}]\n"
            f"S4 lower/equal: {difference.lt(0).mean():.0%}/{difference.eq(0).mean():.0%}",
            transform=axis.transAxes, va="top", fontsize=7,
        )
    pd.DataFrame(summaries).to_csv(output / "relation_stability_summary.csv", index=False)
    figure.suptitle(
        f"Relation stability around the frozen knee (±{local_window_minutes:g} min)",
        fontsize=10,
    )
    figure.tight_layout()
    _export(figure, output / "relation_stability")


def render_data_availability(coverage, predictions, output):
    """Show the complete cohort funnel and causal cold-start delay."""
    first_method = predictions.method.iloc[0]
    stream = predictions.loc[predictions.method.eq(first_method)].copy()
    rows = []
    for cycle_name, cycle in stream.groupby("cycle_name"):
        origin = pd.Timestamp(cycle.heating_start.iloc[0])
        first_frame = pd.to_datetime(cycle.image_time).min()
        valid = cycle.loc[cycle.get("online_pointwise_valid", False).fillna(False)]
        rows.append({
            "cycle_name": cycle_name,
            "first_frame_minutes": (first_frame - origin).total_seconds() / 60,
            "first_economic_valid_minutes": (
                (pd.to_datetime(valid.image_time).min() - origin).total_seconds() / 60
                if len(valid) else np.nan
            ),
        })
    cold_start = pd.DataFrame(rows)
    cold_start.to_csv(Path(output) / "data_availability_by_cycle.csv", index=False)
    counts = pd.DataFrame({
        "stage": ["Dataset catalog", "Causal prefix prepared", "Knee bracketed by RGB"],
        "cycles": [len(coverage), coverage.preparation_status.eq("eligible").sum(),
                   stream.cycle_name.nunique()],
    })
    counts.to_csv(Path(output) / "data_availability_counts.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    axes[0].barh(counts.stage, counts.cycles, color=("#B9C2C9", "#78A6BC", "#365A83"))
    for index, value in enumerate(counts.cycles):
        axes[0].text(value + 1, index, str(value), va="center", fontsize=7)
    axes[0].set(xlabel="Cycles", title="Evaluation cohort is explicit")
    axes[0].invert_yaxis()
    axes[1].hist(cold_start.first_frame_minutes, bins=15, color="#78A6BC", alpha=.7,
                 label="First RGB frame")
    axes[1].hist(cold_start.first_economic_valid_minutes.dropna(), bins=15,
                 color="#F2A35E", alpha=.55, label="First valid C/H/O")
    axes[1].set(xlabel="Minutes from cycle start", ylabel="Cycles",
                title="Causal cold-start availability")
    axes[1].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    _export(figure, Path(output) / "data_availability")


def render_figures(
    predictions,
    teachers,
    losses,
    pairs,
    output,
    teacher_curves=None,
    rb_triggers=None,
    local_window_minutes=15,
    selected_inputs=None,
    comparison_curves=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cycles, frames, summary = evaluate_predictions(predictions, teachers, "first_positive")
    secondary_cycles, _, secondary_summary = evaluate_predictions(
        predictions, teachers, "two_of_three"
    )
    for name, table in (("cycle_decisions", cycles), ("frame_metrics", frames),
                        ("summary", summary), ("losses", losses), ("pair_diagnostics", pairs)):
        table.to_csv(output / f"{name}.csv", index=False)
    secondary_cycles.to_csv(output / "cycle_decisions_two_of_three.csv", index=False)
    secondary_summary.to_csv(output / "summary_two_of_three.csv", index=False)
    _design(output, set(predictions.method))
    if not losses.empty:
        _loss_plot(losses, output)
    _summary_plot(cycles, output)
    _summary_plot(
        secondary_cycles, output, "two_of_three", "decision_summary_two_of_three"
    )
    included = cycles.loc[cycles.evaluation_included, "cycle_name"].unique()
    _cycle_figures(
        predictions.loc[predictions.cycle_name.isin(included)],
        cycles.loc[cycles.cycle_name.isin(included)],
        output,
        teacher_curves,
        rb_triggers,
        local_window_minutes,
        comparison_curves=comparison_curves,
    )
    if not pairs.empty:
        _pair_order_plot(pairs, output)
    if selected_inputs is not None and {"s0", "s1", "s2", "s3", "s4", "n2"} <= set(
        predictions.method
    ):
        render_paired_comparisons(cycles, selected_inputs, output)
        render_relation_stability(
            predictions, selected_inputs, output, local_window_minutes
        )
    evaluated = predictions.loc[predictions.cycle_name.isin(included)]
    adapted = _adapt(evaluated)
    decisions = evaluated[["cycle_name", "teacher_time"]].drop_duplicates().rename(
        columns={"teacher_time": "selected_defrost_time"}
    ).assign(is_selected=True)
    styles = {_label(method, seed): STYLES[method] for method, seed in
              predictions[IDENTITY].drop_duplicates().itertuples(index=False)}
    plot_trigger_error_figures(
        predictions=adapted, decisions=decisions, output=output, source_output=output,
        image_feature="dinov2", classifier="pareto_boundary", continuous_stream=True,
        method_styles=styles, policies=("first_positive", "two_of_three"), threshold=0.,
        flat_output=True,
        error_quantile=.90,
    )
    return cycles, frames, summary


def render_neural_figures(predictions, candidates, losses, output):
    """Reuse the stopping renderer, then add frozen-surrogate sensitivity evidence."""
    output = Path(output)
    teachers = candidates.loc[candidates.is_knee].copy()
    curves = candidates.loc[candidates.is_teacher_candidate].copy()
    cycles, _, _ = render_figures(
        predictions, teachers, losses, pd.DataFrame(), output,
        teacher_curves=curves, comparison_curves=candidates,
    )
    render_neural_sensitivity(candidates, output)
    render_neural_paired_comparison(cycles, output)


def _vcnet_event_pairs(predictions):
    targets = list(OUTCOME_TARGETS.values())
    standardized = [f"standardized_absolute_error_{target}" for target in targets]
    missing = set(standardized) - set(predictions)
    if missing:
        raise ValueError(f"state study requires standardized event errors: {sorted(missing)}")
    by_experiment = []
    for keys, rows in predictions.groupby(
        ["representation", "seed", "experiment_id"], sort=False
    ):
        for target in targets:
            error = rows[f"standardized_absolute_error_{target}"]
            by_experiment.append({
                "representation": keys[0], "seed": keys[1], "experiment_id": keys[2],
                "target": target,
                "mae": error.mean(),
            })
    values = pd.DataFrame(by_experiment)
    records = []
    for (seed, target), group in values.groupby(["seed", "target"], sort=False):
        pivot = group.pivot(index="experiment_id", columns="representation", values="mae")
        factorial = (
            pivot[list(FACTORIAL_RECIPES)].dropna().sort_index()
            if set(FACTORIAL_RECIPES) <= set(pivot)
            else pivot.iloc[:0].reindex(columns=FACTORIAL_RECIPES)
        )
        factorial_indices = np.random.default_rng(0).integers(
            0, len(factorial), size=(2000, len(factorial))
        ) if len(factorial) else None
        for comparison, coefficients in VCNET_CONTRASTS.items():
            if not set(coefficients) <= set(pivot):
                continue
            if comparison == "compression":
                complete = pivot[list(coefficients)].dropna().sort_index()
                indices = np.random.default_rng(0).integers(
                    0, len(complete), size=(2000, len(complete))
                ) if len(complete) else None
            else:
                complete = factorial
                indices = factorial_indices
            if complete.empty:
                continue
            contrast = sum(
                coefficient * complete[method]
                for method, coefficient in coefficients.items()
            ).to_numpy()
            draws = contrast[indices].mean(axis=1)
            ratio = ratio_low = ratio_high = np.nan
            noninferiority = "not_applicable"
            positive = [name for name, value in coefficients.items() if value == 1]
            negative = [name for name, value in coefficients.items() if value == -1]
            if len(coefficients) == 2 and len(positive) == len(negative) == 1:
                candidate = complete[positive[0]].to_numpy()
                reference = complete[negative[0]].to_numpy()
                ratio = candidate.mean() / reference.mean()
                ratio_draws = candidate[indices].mean(axis=1) / reference[indices].mean(axis=1)
                ratio_low, ratio_high = np.quantile(ratio_draws, [.025, .975])
                noninferiority = (
                    "supported" if ratio_high <= 1.05
                    else "exceeds_margin" if ratio_low > 1.05
                    else "insufficient_evidence"
                )
            method = next(name for name, coefficient in coefficients.items()
                          if coefficient == 1)
            records.append({
                "comparison": comparison, "representation": method,
                "seed": seed, "target": target, "experiments": len(contrast),
                "estimate": contrast.mean(), "mean_mae_difference": contrast.mean(),
                "ci_low": np.quantile(draws, .025), "ci_high": np.quantile(draws, .975),
                "mae_ratio": ratio, "mae_ratio_ci_low": ratio_low,
                "mae_ratio_ci_high": ratio_high,
                "noninferiority_5_percent": noninferiority,
            })
    return values, pd.DataFrame(records)


def _vcnet_state_runs(root, reference_run=None):
    """Find only the pre-specified state recipes; append frozen D32 explicitly."""
    root = Path(root)
    runs = [
        path.parent for path in root.glob("*/seed_*/outcome_predictions.csv")
        if path.parent.parent.name in STATE_RECIPES - {"multimodal_time_linear"}
    ]
    if reference_run is not None:
        runs.extend(
            path.parent for path in Path(reference_run).glob(
                "multimodal_time_linear/seed_*/outcome_predictions.csv"
            )
        )
    return sorted(set(runs), key=lambda path: (path.parent.name, path.name))


def _vcnet_consequence_pairs(consequences, replicates=2000):
    """Cluster-bootstrap paired Median/P90 Ridge loss contrasts by experiment."""
    rows = consequences.dropna(
        subset=["maximum_relative_performance_loss_percent"]
    ).copy()
    experiment = "heldout_experiment" if "heldout_experiment" in rows else "experiment_id"
    records = []
    for seed, group in rows.groupby("seed", sort=False):
        pivot = group.pivot_table(
            index=[experiment, "cycle_name"], columns="representation",
            values="maximum_relative_performance_loss_percent", aggfunc="first",
        )
        factorial = (
            pivot[list(FACTORIAL_RECIPES)]
            if set(FACTORIAL_RECIPES) <= set(pivot)
            else pivot.iloc[:0].reindex(columns=FACTORIAL_RECIPES)
        )
        supported = factorial.notna().all(axis=1).groupby(level=experiment).all()
        factorial_experiments = supported.loc[supported].index.to_numpy()
        factorial = factorial.loc[
            factorial.index.get_level_values(experiment).isin(factorial_experiments)
        ]
        factorial_sampled = np.random.default_rng(0).choice(
            factorial_experiments,
            (replicates, len(factorial_experiments)), replace=True,
        ) if len(factorial_experiments) else None
        for comparison, coefficients in VCNET_CONTRASTS.items():
            if not set(coefficients) <= set(pivot):
                continue
            if comparison == "compression":
                complete = pivot[list(coefficients)].dropna()
                experiments = complete.index.get_level_values(experiment).unique().to_numpy()
                sampled = np.random.default_rng(0).choice(
                    experiments, (replicates, len(experiments)), replace=True,
                ) if len(experiments) else None
            else:
                complete = factorial
                experiments = factorial_experiments
                sampled = factorial_sampled
            if not len(experiments):
                continue
            for statistic, function in (
                ("median", np.median), ("p90", lambda x: np.quantile(x, .9))
            ):
                estimate = sum(
                    coefficient * function(complete[method].to_numpy())
                    for method, coefficient in coefficients.items()
                )
                draws = []
                for draw in sampled:
                    selected = pd.concat([complete.xs(value, level=experiment) for value in draw])
                    draws.append(sum(
                        coefficient * function(selected[method].to_numpy())
                        for method, coefficient in coefficients.items()
                    ))
                records.append({
                    "comparison": comparison, "seed": seed, "statistic": statistic,
                    "experiments": len(experiments), "estimate": estimate,
                    "ci_low": np.quantile(draws, .025),
                    "ci_high": np.quantile(draws, .975),
                })
    return pd.DataFrame(records)


def _vcnet_roughness_pairs(roughness, replicates=2000):
    """Experiment-cluster bootstrap of the pre-specified roughness contrasts."""
    metrics = (
        "latent_second_difference_rms_median",
        "outcome_second_difference_rms_median",
    )
    experiment = roughness.groupby(
        ["representation", "seed", "experiment_id", "delta_seconds"], as_index=False
    )[list(metrics)].mean()
    records = []
    for (seed, seconds), group in experiment.groupby(["seed", "delta_seconds"], sort=False):
        for metric in metrics:
            pivot = group.pivot(
                index="experiment_id", columns="representation", values=metric
            )
            factorial = (
                pivot[list(FACTORIAL_RECIPES)].dropna().sort_index()
                if set(FACTORIAL_RECIPES) <= set(pivot)
                else pivot.iloc[:0].reindex(columns=FACTORIAL_RECIPES)
            )
            factorial_indices = np.random.default_rng(0).integers(
                0, len(factorial), size=(replicates, len(factorial))
            ) if len(factorial) else None
            for comparison, coefficients in VCNET_CONTRASTS.items():
                if not set(coefficients) <= set(pivot):
                    continue
                if comparison == "compression":
                    complete = pivot[list(coefficients)].dropna().sort_index()
                    indices = np.random.default_rng(0).integers(
                        0, len(complete), size=(replicates, len(complete))
                    ) if len(complete) else None
                else:
                    complete = factorial
                    indices = factorial_indices
                if complete.empty:
                    continue
                contrast = sum(
                    coefficient * complete[method]
                    for method, coefficient in coefficients.items()
                ).to_numpy()
                draws = contrast[indices].mean(axis=1)
                records.append({
                    "comparison": comparison, "seed": seed,
                    "delta_seconds": seconds, "metric": metric,
                    "experiments": len(contrast), "estimate": contrast.mean(),
                    "ci_low": np.quantile(draws, .025),
                    "ci_high": np.quantile(draws, .975),
                })
    return pd.DataFrame(records)


def _render_state_pathways(pathways, output):
    """Plot frozen-model local sensitivity; these are not causal contributions."""
    if pathways.empty:
        return
    output = Path(output)
    order = ["rgb", "sensor", "ledger", "quality", "time", "interaction"]
    cycle = pathways.groupby(
        ["representation", "seed", "experiment_id", "cycle_name", "delta_seconds",
         "pathway", "target"], as_index=False, dropna=False,
    ).standardized_rms.mean()
    source = cycle.groupby(
        ["representation", "seed", "experiment_id", "delta_seconds", "pathway", "target"],
        as_index=False, dropna=False,
    ).standardized_rms.mean()
    source.to_csv(output / "state_pathway_source.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    markers = ("o", "s", "^", "D")
    for axis, seconds in zip(axes, (10, 60, 300), strict=True):
        selected = source.loc[source.delta_seconds.eq(seconds)]
        for offset, marker, target in zip(
            np.linspace(-.18, .18, len(markers)), markers, OUTCOME_TARGETS.values(),
            strict=True,
        ):
            values = (
                selected.loc[selected.target.eq(target)]
                .groupby("pathway").standardized_rms.mean()
            )
            axis.scatter(
                np.arange(len(order)) + offset, values.reindex(order), marker=marker, s=18,
                label=OUTCOME_LABELS[target].replace(" [kWh]", "").replace(" [min]", ""),
            )
        axis.set(
            title=f"{seconds // 60} min" if seconds >= 60 else "10 s · primary",
            xticks=range(len(order)), xticklabels=[name.capitalize() for name in order],
            ylabel="Standardized output perturbation RMS" if seconds == 10 else None,
        )
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=.15)
    axes[-1].legend(frameon=False, fontsize=5.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.suptitle("Frozen D32 pathway sensitivity (not causal attribution)", fontsize=9)
    figure.tight_layout()
    _export(figure, output / "state_pathway_sensitivity")


def _render_state_intervention(experiment_errors, roughness, consequences, output):
    """Join event fidelity, 10-second roughness and common-Ridge consequences."""
    output = Path(output)
    methods = [
        "multimodal_time_linear", "multimodal_time_linear_z64",
        "multimodal_time_linear_z32_curvature",
        "multimodal_time_linear_z64_curvature",
    ]
    short = dict(zip(methods, ("D32", "D64", "T32", "T64"), strict=True))
    event_summary = experiment_errors.groupby(
        ["representation", "seed", "target"], as_index=False
    ).mae.mean()
    event_summary.to_csv(output / "state_event_mae_source.csv", index=False)
    rough_cycle = roughness.loc[roughness.delta_seconds.eq(10)].copy()
    rough_summary = rough_cycle.groupby(
        ["representation", "seed", "experiment_id"], as_index=False
    )[[
        "latent_second_difference_rms_median",
        "outcome_second_difference_rms_median",
    ]].mean().groupby(["representation", "seed"], as_index=False).mean(numeric_only=True)
    rough_summary.to_csv(output / "state_roughness_source.csv", index=False)
    consequence_summary = consequences.groupby(
        ["representation", "seed"], as_index=False
    ).maximum_relative_performance_loss_percent.agg(
        evaluated="count", median="median", p90=lambda x: x.quantile(.9)
    )
    consequence_summary.to_csv(output / "state_consequence_source.csv", index=False)

    figure, axes = plt.subplots(2, 4, figsize=(7.2, 5.2))
    x = np.arange(len(methods))
    target_titles = ("Electricity", "Net heat", "Compressor E", "Duration")
    for axis, target, title in zip(
        axes[0], OUTCOME_TARGETS.values(), target_titles, strict=True
    ):
        selected = event_summary.loc[event_summary.target.eq(target)]
        for seed, marker in ((0, "o"), (1, "s")):
            values = selected.loc[selected.seed.eq(seed)].set_index("representation").mae
            axis.plot(x, values.reindex(methods), marker=marker, lw=.8, label=f"seed {seed}")
        axis.set(
            title=title,
            xticks=x, xticklabels=["D32", "D64", "T32", "T64"],
            ylabel="Experiment-equal standardized MAE",
        )
        axis.set_title(title, fontsize=7.5)
    for axis, metric, title in zip(
        axes[1, :2],
        ("latent_second_difference_rms_median", "outcome_second_difference_rms_median"),
        (r"10-s latent roughness $R_z$", r"10-s output roughness $R_Y$"),
        strict=True,
    ):
        for seed, marker in ((0, "o"), (1, "s")):
            values = rough_summary.loc[rough_summary.seed.eq(seed)].set_index(
                "representation"
            )[metric]
            axis.plot(x, values.reindex(methods), marker=marker, lw=.8, label=f"seed {seed}")
        axis.set(xticks=x, xticklabels=["D32", "D64", "T32", "T64"])
        axis.set_title(title, fontsize=7.5)
    for seed, marker in ((0, "o"), (1, "s")):
        selected = consequence_summary.loc[consequence_summary.seed.eq(seed)].set_index(
            "representation"
        )
        axes[1, 2].plot(x, selected["median"].reindex(methods), marker=marker, lw=.8,
                        label=f"Median · seed {seed}")
        axes[1, 2].plot(x, selected["p90"].reindex(methods), marker=marker, lw=.8, ls="--",
                        label=f"P90 · seed {seed}")
    axes[1, 2].set(
        title="Decision loss L", xticks=x,
        xticklabels=["D32", "D64", "T32", "T64"], ylabel="Maximum C/H loss [%]",
    )
    axes[1, 2].set_title("Decision loss L", fontsize=7.5)
    denominator = 97
    coverage_records = []
    for (method, seed), rows in consequences.groupby(["representation", "seed"], sort=False):
        values = rows.maximum_relative_performance_loss_percent.dropna().sort_values()
        if values.empty:
            continue
        coverage = np.arange(1, len(values) + 1) / denominator
        coverage_records.extend({
            "representation": method, "seed": seed, "loss_percent": value,
            "coverage_fraction": fraction, "denominator": denominator,
        } for value, fraction in zip(values, coverage, strict=True))
        if method in methods:
            color, _ = VCNET_STYLES[method]
            axes[1, 3].step(values, coverage, where="post", color=color,
                            ls="-" if seed == 0 else "--",
                            label=f"{short[method]} · seed {seed}")
    pd.DataFrame(coverage_records).to_csv(output / "state_coverage_source.csv", index=False)
    axes[1, 3].set(
        title="Coverage (97 cycles)", xlim=(0, 5), ylim=(0, 1.02),
        xlabel="Maximum C/H loss [%]", ylabel="Cycles / 97",
    )
    axes[1, 3].set_title("Coverage (97 cycles)", fontsize=7.5)
    axes[0, 0].legend(frameon=False, fontsize=5.5)
    axes[1, 2].legend(frameon=False, fontsize=5)
    axes[1, 3].legend(frameon=False, fontsize=4.8)
    for axis in axes.flat:
        axis.grid(alpha=.15)
    figure.suptitle(
        "State representation intervention: fidelity, roughness and consequence",
        fontsize=9,
    )
    figure.tight_layout(h_pad=2.0, w_pad=1.0)
    _export(figure, output / "state_intervention")


def _render_state_nearest_neighbors(nearest, output):
    if nearest.empty:
        return
    output = Path(output)
    nearest.to_csv(output / "nearest_neighbor_source.csv", index=False)
    summary = nearest.groupby(
        ["representation", "seed", "space", "target"], as_index=False
    ).standardized_outcome_discrepancy_median.mean()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for axis, space in zip(axes, ("input", "latent"), strict=True):
        selected = summary.loc[summary.space.eq(space)]
        for method in STATE_RECIPES:
            rows = selected.loc[selected.representation.eq(method)]
            if rows.empty:
                continue
            values = rows.groupby("target").standardized_outcome_discrepancy_median.mean()
            axis.plot(
                range(len(OUTCOME_TARGETS)), values.reindex(OUTCOME_TARGETS.values()),
                marker=VCNET_STYLES[method][1], color=VCNET_STYLES[method][0], lw=.8,
                label=VCNET_LABELS[method],
            )
        axis.set(
            title=f"{space.capitalize()}-space k=3 neighbours",
            xticks=range(len(OUTCOME_TARGETS)), xticklabels=["E", "Q", "Ecomp", "D"],
            ylabel="Standardized outcome discrepancy" if space == "input" else None,
        )
    axes[-1].legend(frameon=False, fontsize=5.2, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    _export(figure, output / "state_nearest_neighbors")


def _render_vcnet_pareto_examples(candidates, consequences, output):
    reference_method = "multimodal_time_linear"
    reference = consequences.loc[
        consequences.representation.eq(reference_method) & consequences.seed.eq(0)
        & consequences.maximum_relative_performance_loss_percent.notna()
    ].sort_values("maximum_relative_performance_loss_percent")
    if reference.empty:
        return
    indices = sorted(set([0, len(reference) // 2, len(reference) - 1]))
    fields = {
        "neural_cycle_cop": "cycle_cop",
        "neural_cycle_heating_rate_kw": "cycle_heating_rate_kw",
        "neural_cycle_evaporator_capacity_kw": "cycle_evaporator_capacity_kw",
        "neural_cycle_cop_eligible": "cycle_cop_eligible",
        "neural_cycle_heating_rate_kw_eligible": "cycle_heating_rate_kw_eligible",
        "neural_is_pareto": "is_cop_heating_rate_pareto_point",
        "neural_is_knee": "is_selected_pareto_point",
    }
    for row in reference.iloc[indices].itertuples(index=False):
        cycle = candidates.loc[
            candidates.cycle_name.eq(row.cycle_name) & candidates.seed.eq(0)
        ]
        methods = [name for name in VCNET_STYLES if name in set(cycle.representation)]
        figure, axes = plt.subplots(len(methods), 2, figsize=(7.2, 2.45 * len(methods)),
                                   squeeze=False)
        for axes_row, method in zip(axes, methods, strict=True):
            values = cycle.loc[cycle.representation.eq(method)].drop(
                columns=list(fields.values()), errors="ignore"
            ).rename(columns=fields).copy()
            origin = pd.Timestamp(values.heating_start.iloc[0])
            center = pd.Timestamp(values.teacher_time.iloc[0])
            for axis, local in zip(axes_row, (False, True), strict=True):
                plot_cop_heating_rate_pareto(
                    axis, values, origin, local=local, local_center_time=center,
                    local_window_minutes=15, selection_label="Selected Pareto knee",
                )
            axes_row[0].set_title(
                f"{VCNET_LABELS[method]}\nFull candidate domain", loc="left", fontsize=7
            )
            axes_row[1].set_title("Ridge-knee ±15 min", loc="left", fontsize=7)
        figure.suptitle(f"{row.cycle_name}: independently selected neural Pareto", fontsize=9)
        figure.tight_layout()
        _export(figure, output / f"pareto_{row.cycle_name}")


def render_vcnet_figures(root, data, reference_run=None):
    """One shared figure path for paired event, trajectory and Pareto evidence."""
    root, output = Path(root), Path(root) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    runs = _vcnet_state_runs(root, reference_run)
    if not runs:
        return
    prediction_tables, loss_tables, roughness_tables, nearest_tables = [], [], [], []
    for run in runs:
        representation, seed = run.parent.name, int(run.name.removeprefix("seed_"))
        diagnostic = (
            root / "reference_diagnostics" / representation / run.name
            / "event_errors.csv"
        )
        prediction_tables.append(pd.read_csv(
            diagnostic if representation == "multimodal_time_linear" and diagnostic.exists()
            else run / "outcome_predictions.csv"
        ).assign(representation=representation, seed=seed))
        if (run / "losses.csv").exists():
            loss_tables.append(pd.read_csv(run / "losses.csv"))
        roughness = (
            root / "reference_diagnostics" / representation / run.name / "roughness.csv"
            if representation == "multimodal_time_linear" else run / "roughness.csv"
        )
        nearest = (
            root / "reference_diagnostics" / representation / run.name
            / "nearest_neighbors.csv"
            if representation == "multimodal_time_linear" else run / "nearest_neighbors.csv"
        )
        if roughness.exists():
            roughness_tables.append(pd.read_csv(roughness).assign(
                representation=representation, seed=seed
            ))
        if nearest.exists():
            nearest_tables.append(pd.read_csv(nearest).assign(
                representation=representation, seed=seed
            ))
    predictions = pd.concat(prediction_tables, ignore_index=True)
    losses = pd.concat(loss_tables, ignore_index=True) if loss_tables else pd.DataFrame()
    candidates = pd.concat([
        pd.read_parquet(run / "candidates.parquet")
        for run in runs if (run / "candidates.parquet").exists()
    ], ignore_index=True)
    render_outcome_figures(predictions, losses, output)

    experiment_errors, paired = _vcnet_event_pairs(predictions)
    experiment_errors.to_csv(output / "event_error_by_experiment.csv", index=False)
    paired.to_csv(output / "event_error_paired_bootstrap.csv", index=False)
    target_order = list(OUTCOME_TARGETS.values())
    if not paired.empty:
        figure, axis = plt.subplots(figsize=(7.2, 3.6))
        positions = np.arange(len(target_order))
        display = paired.loc[paired.comparison.isin(
            ["capacity", "curvature_z32", "capacity_x_curvature"]
        )]
        groups = list(display.groupby(["comparison", "seed"], sort=False))
        for offset, ((comparison, seed), rows) in zip(
            np.linspace(-.18, .18, len(groups)), groups, strict=True
        ):
            ordered = rows.set_index("target").reindex(target_order)
            method = rows.representation.iloc[0]
            axis.errorbar(
                ordered.mean_mae_difference, positions + offset,
                xerr=[ordered.mean_mae_difference - ordered.ci_low,
                      ordered.ci_high - ordered.mean_mae_difference],
                fmt=VCNET_STYLES[method][1], color=VCNET_STYLES[method][0], capsize=2,
                markerfacecolor=VCNET_STYLES[method][0] if seed == 0 else "white",
                label=f"{comparison.replace('_', ' ')} · seed {seed}",
            )
        axis.axvline(0, color="#333333", lw=.7, ls="--")
        axis.set(
            yticks=positions,
            yticklabels=[OUTCOME_LABELS[name] for name in target_order],
            xlabel="Pre-specified experiment-paired MAE contrast",
        )
        axis.legend(frameon=False, fontsize=6.5)
        figure.tight_layout()
        _export(figure, output / "event_error_paired")

    consequences = vcnet_pareto_consequences(candidates)
    consequences.to_csv(output / "pareto_consequences.csv", index=False)
    consequence_pairs = _vcnet_consequence_pairs(consequences)
    consequence_pairs.to_csv(output / "pareto_consequence_paired_bootstrap.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    valid_loss = consequences.maximum_relative_performance_loss_percent.dropna()
    full_limit = max(5., float(valid_loss.max())) if len(valid_loss) else 5.
    denominator = candidates.cycle_name.nunique()
    for method, style in VCNET_STYLES.items():
        for seed, line in ((0, "-"), (1, "--")):
            values = consequences.loc[
                consequences.representation.eq(method) & consequences.seed.eq(seed),
                "maximum_relative_performance_loss_percent",
            ].dropna().sort_values()
            if values.empty:
                continue
            y = np.arange(1, len(values) + 1) / denominator
            for axis in axes:
                axis.step(values, y, where="post", color=style[0], ls=line,
                          label=f"{VCNET_LABELS[method]} · seed {seed}")
    axes[0].set(xlim=(0, full_limit), title="Full observed tail")
    axes[1].set(xlim=(0, 5), title="0–5% decision-relevant detail")
    for axis in axes:
        axis.set(ylim=(0, 1.02), xlabel="Maximum relative Ridge C/H loss [%]",
                 ylabel=f"Cycles / {denominator}")
        axis.grid(alpha=.15)
    axes[1].legend(frameon=False, fontsize=5.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    _export(figure, output / "pareto_performance_coverage")
    roughness = (
        pd.concat(roughness_tables, ignore_index=True)
        if roughness_tables else pd.DataFrame()
    )
    nearest = (
        pd.concat(nearest_tables, ignore_index=True)
        if nearest_tables else pd.DataFrame()
    )
    pathways = []
    for seed in (0, 1):
        path = (
            root / "reference_diagnostics" / "multimodal_time_linear"
            / f"seed_{seed}" / "pathways.csv"
        )
        if path.exists():
            pathways.append(pd.read_csv(path))
    if pathways:
        _render_state_pathways(pd.concat(pathways, ignore_index=True), output)
    if not roughness.empty:
        roughness.to_csv(output / "state_roughness_by_cycle.csv", index=False)
        _vcnet_roughness_pairs(roughness).to_csv(
            output / "state_roughness_paired_bootstrap.csv", index=False
        )
        _render_state_intervention(
            experiment_errors, roughness, consequences, output
        )
    _render_state_nearest_neighbors(nearest, output)
    _render_vcnet_pareto_examples(candidates, consequences, output)


def _overview_export(figure, output, stem):
    figure.align_labels()
    _export(figure, Path(output) / stem, formats=("png", "svg", "pdf"))


def _overview_interval_axis(axis, rows, labels, xlabel, title):
    rows = rows.copy()
    rows["label"] = labels
    positions = np.arange(len(rows))[::-1]
    axis.errorbar(
        rows.mean_difference,
        positions,
        xerr=[
            rows.mean_difference - rows.ci_low,
            rows.ci_high - rows.mean_difference,
        ],
        fmt="o",
        color="#3B6F8F",
        ecolor="#7B8188",
        capsize=2.5,
        lw=1.2,
    )
    axis.axvline(0, color="#333333", lw=.8, ls="--")
    axis.set(
        yticks=positions,
        yticklabels=rows.label,
        xlabel=xlabel,
    )
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=28)
    axis.grid(axis="x", color="#E2E5E9", lw=.5)
    for position, (_, row) in zip(positions, rows.iterrows(), strict=True):
        axis.text(
            max(row.ci_high, row.mean_difference) + .02 * max(1., rows.ci_high.max()),
            position,
            f"n={int(row.paired_cycles)}",
            va="center",
            fontsize=6,
            color="#61666D",
        )


def _overview_objective_tradeoffs(policy_root, output):
    regrets = pd.read_csv(policy_root / "objective_regret_source.csv").set_index("method")
    similarity = pd.read_csv(policy_root / "selection_similarity_source.csv")
    similarity = similarity.rename(columns={similarity.columns[0]: "method"}).set_index("method")
    order = ["C", "H", "O", "CH knee", "CO knee", "HO knee"]
    labels = [OVERVIEW_POLICY_LABELS[name] for name in order]
    values = regrets.reindex(order)[["C", "H", "O"]].to_numpy(dtype=float)
    agreement = similarity.reindex(index=order, columns=order).to_numpy(dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.45), gridspec_kw={"width_ratios": [1, 1.35]})
    axis = axes[0]
    image = axis.imshow(values, cmap="YlOrRd", vmin=0, vmax=max(2.2, values.max()), aspect="auto")
    for row in range(len(order)):
        for column in range(3):
            value = values[row, column]
            axis.text(
                column,
                row,
                "0" if abs(value) < 5e-4 else f"{value:.2f}%",
                ha="center",
                va="center",
                color="white" if value > 1.15 else "#25292E",
                fontweight="bold" if order[row] in {"C", "CH knee"} else "normal",
            )
    axis.set_xticks(range(3), ["C", "H", "O"], fontsize=9)
    axis.xaxis.tick_top()
    axis.set_yticks(range(len(order)), labels)
    for index, tick in enumerate(axis.get_yticklabels()):
        tick.set_fontweight("bold" if order[index] in {"C", "CH knee"} else "normal")
    axis.set_title("a  Performance consequence", loc="left", fontsize=9, fontweight="bold", pad=42)
    axis.text(.5, 1.08, "Median regret (%) | lower is better", transform=axis.transAxes,
              ha="center", color="#5E646B", fontsize=6.5)
    figure.colorbar(image, ax=axis, fraction=.05, pad=.04)

    axis = axes[1]
    shown = np.ma.array(agreement, mask=np.triu(np.ones_like(agreement, dtype=bool), 1))
    image = axis.imshow(shown, cmap="Blues", vmin=0, vmax=100)
    for row in range(len(order)):
        for column in range(row + 1):
            value = agreement[row, column]
            axis.text(column, row, f"{value:.0f}%", ha="center", va="center",
                      color="white" if value >= 65 else "#25292E", fontsize=6.5,
                      fontweight="bold" if value >= 95 else "normal")
    axis.set_xticks(range(len(order)), labels, rotation=35, ha="left", fontsize=7)
    axis.xaxis.tick_top()
    axis.set_yticks(range(len(order)), labels, fontsize=7)
    axis.tick_params(length=0)
    axis.add_patch(plt.Rectangle((.5, 1.5), 1, 1, fill=False, edgecolor="#C05A50", lw=1.5))
    axis.annotate("H and O: 30% same time", xy=(1, 2), xytext=(3.8, .8),
                  color="#A14B43", fontsize=6.5, fontweight="bold", ha="center",
                  arrowprops=dict(arrowstyle="->", color="#C05A50", lw=.8))
    axis.set_title("b  Exact selected-time agreement (96 cycles)", loc="left", fontsize=9,
                   fontweight="bold", pad=42)
    colorbar = figure.colorbar(image, ax=axis, fraction=.04, pad=.03)
    colorbar.set_label("Same time (%)", fontsize=7)
    figure.suptitle(
        "Why retain a C-H Pareto reference?",
        fontsize=11,
        fontweight="bold",
        y=.99,
    )
    figure.text(
        .5,
        .925,
        "C: cycle efficiency | H: heating throughput | O: outdoor-side extraction",
        ha="center",
        color="#5E646B",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0, 1, .89), w_pad=2.2)
    _overview_export(figure, output, OVERVIEW_FIGURE_STEMS[0])


def _overview_event_calibration(probe_root, output):
    rows = pd.read_csv(probe_root / "event_calibration_source.csv")
    rows = rows.loc[rows.panel.eq("frozen_92")]
    metrics = pd.read_csv(probe_root / "event_calibration_metrics.csv")
    metrics = metrics.loc[metrics.panel.eq("frozen_92")].set_index(["target", "representation"])
    targets = list(OUTCOME_TARGETS.values())
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))
    styles = {
        "RGB + sensor": ("visual", "#3B6F8F", "o"),
        "Sensor only": ("nonvisual", "#D28E4B", "^"),
        "Ridge": ("ridge", "#777777", "s"),
    }
    for axis, target in zip(axes.flat, targets, strict=True):
        groups = []
        for label, (representation, color, marker) in styles.items():
            group = rows.drop_duplicates("event_id") if representation == "ridge" else rows.loc[
                rows.representation.eq(representation)
            ]
            prediction = f"ridge_predicted_{target}" if representation == "ridge" else f"predicted_{target}"
            groups.append((label, representation, color, marker, group, prediction))
        low = min(
            float(rows[target].min()),
            *(float(group[prediction].min()) for *_, group, prediction in groups),
        )
        high = max(
            float(rows[target].max()),
            *(float(group[prediction].max()) for *_, group, prediction in groups),
        )
        padding = .04 * max(high - low, 1e-9)
        low, high = low - padding, high + padding
        metric_lines = []
        for label, representation, color, marker, group, prediction in groups:
            axis.scatter(
                group[target],
                group[prediction],
                s=18,
                alpha=.55,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=.25,
                label=label,
            )
            metric = metrics.loc[(target, representation)]
            metric_lines.append(
                f"{label}: MAE {metric.mae:.3g}; slope {metric.calibration_slope:.2f} "
                f"[{metric.calibration_slope_ci_low:.2f}, {metric.calibration_slope_ci_high:.2f}]"
            )
        axis.plot(
            [low, high], [low, high], color="#222222", lw=.8, ls="--",
            label="Ideal agreement",
        )
        axis.set(
            xlim=(low, high), ylim=(low, high),
            xlabel="Observed outcome",
            ylabel="Held-out prediction",
            title=OUTCOME_LABELS[target],
        )
        axis.set_aspect("equal", adjustable="box")
        axis.text(.02, .98, "\n".join(metric_lines), transform=axis.transAxes,
                  va="top", fontsize=5.4, color="#30343A",
                  bbox=dict(facecolor="white", alpha=.82, edgecolor="#D7DBDF", pad=2.5))
        axis.grid(color="#E5E7EA", lw=.45)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle(
        "Prediction of observed defrost-event outcomes",
        fontsize=11,
        fontweight="bold",
    )
    figure.text(
        .5, .935,
        "92 actual events | 19 held-out experiments | calibration: observed = intercept + slope × predicted",
        ha="center",
        color="#5E646B",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, .07, 1, .92), h_pad=2.0)
    _overview_export(figure, output, OVERVIEW_FIGURE_STEMS[4])


def _overview_ridge_calibration(probe_root, output):
    event_rows = pd.read_csv(probe_root / "event_calibration_source.csv")
    event_rows = event_rows.loc[event_rows.panel.eq("frozen_92")].drop_duplicates(
        "event_id"
    )
    event_metrics = pd.read_csv(probe_root / "event_calibration_metrics.csv")
    event_metrics = event_metrics.loc[
        event_metrics.panel.eq("frozen_92")
        & event_metrics.representation.eq("ridge")
    ].set_index("target")
    action_rows = pd.read_csv(probe_root / "actual_action_ch_calibration.csv")
    action_rows = action_rows.loc[
        action_rows.panel.eq("frozen_92")
        & action_rows.representation.eq("ridge")
    ]
    action_metrics = pd.read_csv(probe_root / "actual_action_ch_metrics.csv")
    action_metrics = action_metrics.loc[
        action_metrics.panel.eq("frozen_92")
        & action_metrics.representation.eq("ridge")
    ].set_index("target")
    panels = [
        (
            event_rows,
            target,
            f"ridge_predicted_{target}",
            event_metrics.loc[target],
            OUTCOME_LABELS[target],
        )
        for target in OUTCOME_TARGETS.values()
    ] + [
        (action_rows, "observed_c", "predicted_c", action_metrics.loc["c"],
         "Actual-action cycle efficiency C [-]"),
        (action_rows, "observed_h", "predicted_h", action_metrics.loc["h"],
         "Actual-action heating throughput H [kW]"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 7.1))
    for axis, (rows, observed, predicted, metric, title) in zip(
        axes.flat, panels, strict=True
    ):
        low = float(min(rows[observed].min(), rows[predicted].min()))
        high = float(max(rows[observed].max(), rows[predicted].max()))
        padding = .04 * max(high - low, 1e-9)
        low, high = low - padding, high + padding
        axis.scatter(
            rows[observed], rows[predicted], s=20, alpha=.62,
            color="#3B6F8F", edgecolor="white", linewidth=.3,
        )
        axis.plot([low, high], [low, high], color="#333333", lw=.8, ls="--")
        axis.set(
            xlim=(low, high), ylim=(low, high),
            xlabel="Observed", ylabel="Held-out Ridge prediction", title=title,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.text(
            .03, .97,
            f"n={int(metric.events)}\n"
            f"MAE {metric.mae:.3g} | Bias {metric.bias:+.3g}\n"
            f"Slope {metric.calibration_slope:.2f} "
            f"[{metric.calibration_slope_ci_low:.2f}, "
            f"{metric.calibration_slope_ci_high:.2f}]",
            transform=axis.transAxes, va="top", fontsize=6.3,
            bbox=dict(facecolor="white", alpha=.86, edgecolor="#D7DBDF", pad=2.5),
        )
        axis.grid(color="#E5E7EA", lw=.45)
    figure.suptitle(
        "Observed evidence for the Ridge decision-reference model",
        fontsize=11, fontweight="bold",
    )
    figure.text(
        .5, .936,
        "92 actual defrost events | 19 leave-one-experiment-out tests | dashed line: ideal agreement",
        ha="center", color="#5E646B", fontsize=7,
    )
    figure.text(
        .5, .018,
        "Calibration: observed = intercept + slope x predicted | intervals are retained experiment-bootstrap results",
        ha="center", color="#5E646B", fontsize=6.5,
    )
    figure.tight_layout(rect=(0, .045, 1, .91), h_pad=1.8, w_pad=1.4)
    _overview_export(figure, output, OVERVIEW_FIGURE_STEMS[1])


def _overview_reference_shift(neural_root, output):
    rows = pd.read_csv(neural_root / "figures" / "neural_knees_by_cycle.csv")
    shifts = rows.loc[rows.selection_status.eq("both"), "knee_difference_minutes"].sort_values().to_numpy()
    ecdf = np.arange(1, len(shifts) + 1) / len(shifts)
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    axis.step(shifts, ecdf, where="post", color="#3B7E9B", lw=1.8)
    axis.axvline(0, color="#333333", lw=.8, ls="--")
    median_abs = np.median(np.abs(shifts))
    median_signed = np.median(shifts)
    axis.set(
        xlabel="Neural reference - Ridge reference (min)",
        ylabel="Cumulative fraction of cycles",
        ylim=(0, 1.02),
    )
    axis.set_title(
        "Distribution of reference-time shifts after replacing the outcome model",
        loc="left", fontsize=11, fontweight="bold", pad=28,
    )
    axis.text(
        0, 1.04,
        f"{len(shifts)} cycles | 19 held-out experiments | median signed shift = {median_signed:.2f} min | median absolute shift = {median_abs:.2f} min",
        transform=axis.transAxes,
        color="#5E646B",
        fontsize=7,
    )
    axis.grid(color="#E1E4E8", lw=.5)
    figure.tight_layout()
    _overview_export(figure, output, OVERVIEW_FIGURE_STEMS[7])


def _overview_fixed_predictor_swap(cross_root, output):
    intervals = pd.read_csv(cross_root / "cross_input_paired_bootstrap.csv")
    summary = pd.read_csv(cross_root / "cross_input_summary.csv")
    specifications = (
        ("trigger_change_minutes", "first_positive", "Signed trigger shift\n(min)"),
        ("absolute_trigger_change_minutes", "first_positive", "Absolute trigger shift\n(min)"),
        ("loss_change_percent", "first_positive", "Maximum C/H loss shift\n(percentage points)"),
        ("local_median_absolute_score_change", "frames", "Local median absolute\nscore change"),
        ("local_sign_flip_fraction", "frames", "Local sign-flip\nfraction"),
    )
    heads = ("ridge", "neural")
    row_labels = (
        "Ridge-trained predictor\nRidge C/H to neural C/H",
        "Neural-trained predictor\nRidge C/H to neural C/H",
    )
    figure, axes = plt.subplots(2, 5, figsize=(11.2, 4.6))
    for column, (metric, strategy, title) in enumerate(specifications):
        selected = intervals.loc[
            intervals.metric.eq(metric) & intervals.strategy.eq(strategy)
        ].set_index("head_source").reindex(heads)
        for row, head in enumerate(heads):
            axis = axes[row, column]
            value = selected.loc[head]
            axis.errorbar(
                value.mean_difference,
                0,
                xerr=[[value.mean_difference - value.ci_low],
                      [value.ci_high - value.mean_difference]],
                fmt="D",
                color="#4B88A2",
                ecolor="#7B8188",
                capsize=2.5,
            )
            axis.axvline(0, color="#333333", lw=.7, ls="--")
            axis.set_yticks([])
            axis.grid(axis="x", color="#E4E7EA", lw=.45)
            axis.text(
                .02, .06, f"n={int(value.paired_cycles)}",
                transform=axis.transAxes, fontsize=6, color="#5E646B",
            )
            if column == 0:
                same = summary.loc[
                    summary.head_source.eq(head)
                    & summary.strategy.eq("first_positive"),
                    "same_trigger_frame",
                ].iloc[0]
                both = summary.loc[
                    summary.head_source.eq(head)
                    & summary.strategy.eq("first_positive"),
                    "both_trigger",
                ].iloc[0]
                axis.set_ylabel(
                    f"{row_labels[row]}\n{int(same)}/{int(both)} same first frame",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=8,
                )
            if row == 0:
                axis.set_title(title, fontsize=8)
    figure.suptitle(
        "Does replacing Ridge C/H with neural C/H change a fixed predictor's behavior?",
        fontsize=11,
        fontweight="bold",
        y=.99,
    )
    figure.text(
        .5, .895,
        "Frozen imputer, scaler and predictor | only the 8D economic input changes | first non-negative score\n"
        "Local score panels: Ridge reference ±15 min | 95% experiment bootstrap intervals",
        ha="center",
        color="#5E646B",
        fontsize=7,
    )
    figure.tight_layout(rect=(.12, 0, 1, .80))
    _overview_export(figure, output, OVERVIEW_FIGURE_STEMS[8])


def _overview_timing_consequence(transfer_root, output):
    candidates = pd.read_csv(
        transfer_root / "seed_0" / "pareto_candidates.csv",
        parse_dates=["candidate_defrost_time"],
    )
    consequences = pd.read_csv(transfer_root / "performance_consequences.csv")
    consequences["trigger_time"] = pd.to_datetime(consequences.trigger_time, format="mixed")
    consequences["reference_time"] = pd.to_datetime(consequences.reference_time, format="mixed")
    cycles = ("frost_cycle_000040", "frost_cycle_000090")
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.8))
    for row_index, cycle_name in enumerate(cycles):
        cycle = candidates.loc[candidates.cycle_name.eq(cycle_name)].sort_values(
            "candidate_defrost_time", kind="stable"
        )
        result = consequences.loc[
            consequences.method.eq("t32_ridge_ch_mlp")
            & consequences.seed.eq(0)
            & consequences.strategy.eq("first_positive")
            & consequences.cycle_name.eq(cycle_name)
        ].iloc[0]
        minutes = (
            cycle.candidate_defrost_time - result.reference_time
        ).dt.total_seconds() / 60
        trigger_minute = result.trigger_error_minutes
        reference = cycle.loc[cycle.is_knee].iloc[0]
        for column, (field, ylabel, trigger_value) in enumerate((
            ("cycle_cop", "Cycle efficiency C [-]", result.trigger_c),
            ("cycle_heating_rate_kw", "Heating throughput H [kW]", result.trigger_h),
        )):
            axis = axes[row_index, column]
            axis.plot(minutes, cycle[field], color="#9BA3AA", lw=1.1)
            axis.scatter(0, reference[field], marker="*", s=90, color="#D28E4B",
                         edgecolor="white", linewidth=.4, zorder=4, label="C-H reference")
            axis.scatter(trigger_minute, trigger_value, marker="X", s=55, color="#3B6F8F",
                         edgecolor="white", linewidth=.4, zorder=4, label="First trigger")
            axis.axvline(0, color="#D28E4B", lw=.7, ls="--")
            axis.axvline(trigger_minute, color="#3B6F8F", lw=.7, ls=":")
            axis.set(xlabel="Time from Ridge C-H reference (min)", ylabel=ylabel)
            axis.grid(color="#E3E6E9", lw=.45)
            if row_index == 0:
                axis.set_title("C over candidate time" if column == 0 else "H over candidate time",
                               fontsize=8, fontweight="bold")
        axis = axes[row_index, 2]
        front = cycle.loc[cycle.is_cop_heating_rate_pareto_point].sort_values("cycle_cop")
        axis.scatter(cycle.cycle_cop, cycle.cycle_heating_rate_kw, s=10,
                     color="#C9CED3", label="Eligible candidates")
        axis.plot(front.cycle_cop, front.cycle_heating_rate_kw, color="#575E65", lw=1.0,
                  marker="o", ms=2.8, label="Pareto front")
        axis.scatter(result.reference_c, result.reference_h, marker="*", s=95,
                     color="#D28E4B", edgecolor="white", linewidth=.4, zorder=4,
                     label="C-H reference")
        axis.scatter(result.trigger_c, result.trigger_h, marker="X", s=60,
                     color="#3B6F8F", edgecolor="white", linewidth=.4, zorder=4,
                     label="First trigger")
        axis.set(xlabel="Cycle efficiency C [-]", ylabel="Heating throughput H [kW]")
        axis.grid(color="#E3E6E9", lw=.45)
        if row_index == 0:
            axis.set_title("Decision geometry in C-H space", fontsize=8, fontweight="bold")
        cycle_id = int(cycle_name.rsplit("_", 1)[-1])
        sign = "+" if trigger_minute >= 0 else ""
        axes[row_index, 0].text(
            -.30, .5,
            f"Cycle {cycle_id}\nDelta t = {sign}{trigger_minute:.2f} min\nL = {result.maximum_relative_performance_loss_percent:.3f}%",
            transform=axes[row_index, 0].transAxes, ha="right", va="center",
            fontsize=7.5, fontweight="bold",
        )
    handles, labels = axes[0, 2].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7)
    figure.suptitle("Timing distance does not determine first-trigger consequence",
                    fontsize=11, fontweight="bold")
    figure.text(.5, .935,
                "Curvature-regularized 32D state | seed 0 | exact first non-negative score | frozen Ridge evaluation",
                ha="center", color="#5E646B", fontsize=7)
    figure.tight_layout(rect=(.08, .07, 1, .91), h_pad=2.0, w_pad=1.5)
    _overview_export(figure, output, OVERVIEW_FIGURE_STEMS[2])


def _overview_online_inputs(outcome_root, probe_root, output):
    rows = pd.read_csv(outcome_root / "figures" / "paired_comparisons.csv")
    rows = rows.loc[rows.metric.eq("absolute_trigger_error_minutes")].copy()
    order = ["S1 − S0", "S2 − S1", "S3 − S2", "S4 − inner-selected", "S2 − N2"]
    rows = rows.set_index("comparison").reindex(order).reset_index()
    labels = [
        "Add current Ridge C/H",
        "Add 5-min Ridge C/H history",
        "Add 5-min Ridge O history",
        "Add signed within-cycle ranking supervision\n(vs inner-selected baseline)",
        "RGB + sensor state vs sensor-only state",
    ]
    probe = pd.read_csv(probe_root / "paired_comparisons.csv")
    probe = probe.loc[
        probe.strategy.eq("first_positive")
        & probe.metric.eq("maximum_relative_performance_loss_percent")
    ].copy()
    probe_order = [
        ("latent_ridge_ch_mlp", "latent_ridge_ch_linear"),
        ("latent_raw_ridge_ch_mlp", "latent_ridge_ch_mlp"),
        ("latent_raw_ridge_ch_mlp", "raw_ridge_ch_mlp"),
    ]
    probe["_order"] = [
        probe_order.index((candidate, reference))
        for candidate, reference in zip(probe.candidate, probe.reference, strict=True)
    ]
    probe = probe.sort_values("_order")
    figure, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), gridspec_kw={"height_ratios": [1.25, 1]})
    axis = axes[0]
    _overview_interval_axis(
        axis,
        rows,
        labels,
        "Change in absolute trigger-time error (min)\nNegative = improvement",
        "a  Localization diagnostic",
    )
    axis.text(
        0, 1.04,
        "First non-negative score | mean paired change and 95% experiment bootstrap interval",
        transform=axis.transAxes,
        color="#5E646B",
        fontsize=7,
    )
    axis = axes[1]
    _overview_interval_axis(
        axis,
        probe,
        [
            "MLP vs linear\n(same state and C/H)",
            "Add pre-compression features\n(to state + MLP)",
            "Add outcome-supervised state\n(to pre-compression features)",
        ],
        "Change in maximum one-sided C/H loss (percentage points)\nNegative = improvement",
        "b  Exact first-trigger consequence",
    )
    axis.text(
        0, 1.04,
        "Separate matched probe | first non-negative score | mean paired change and 95% experiment bootstrap interval",
        transform=axis.transAxes, color="#5E646B", fontsize=7,
    )
    figure.suptitle("What information and predictor choices affect online triggering?",
                    fontsize=11, fontweight="bold", y=.995)
    figure.tight_layout(rect=(0, 0, 1, .95), h_pad=3.0)
    _overview_export(figure, output, OVERVIEW_FIGURE_STEMS[5])


def _overview_state_intervention(state_root, output):
    roughness = pd.read_csv(state_root / "figures" / "state_roughness_paired_bootstrap.csv")
    event = pd.read_csv(state_root / "figures" / "event_error_paired_bootstrap.csv")
    roughness = roughness.loc[
        roughness.comparison.eq("curvature_z32") & roughness.delta_seconds.eq(10)
    ]
    event = event.loc[event.comparison.eq("curvature_z32")]
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 4.8),
                               gridspec_kw={"width_ratios": [1.45, 1, 1]})

    target_labels = {
        "defrost_event_electricity_observed_kwh": "Electricity",
        "defrost_event_net_heat_observed_kwh": "Net heat",
        "defrost_event_compressor_electricity_observed_kwh": "Compressor electricity",
        "defrost_event_duration_observed_minutes": "Duration",
    }
    ordered = []
    for target in OUTCOME_TARGETS.values():
        for seed in (0, 1):
            ordered.append(event.loc[event.target.eq(target) & event.seed.eq(seed)].iloc[0])
    axis = axes[0]
    positions = np.arange(len(ordered))[::-1]
    highs = np.array([row.ci_high for row in ordered])
    colors = ["#3B6F8F" if row.seed == 0 else "#D28E4B" for row in ordered]
    for position, row, color in zip(positions, ordered, colors, strict=True):
        axis.errorbar(row.mean_mae_difference, position,
                      xerr=[[row.mean_mae_difference - row.ci_low], [row.ci_high - row.mean_mae_difference]],
                      fmt="o", color=color, ecolor="#8A9096", capsize=2, lw=1)
        status = "supported" if row.noninferiority_5_percent == "supported" else "insufficient"
        axis.text(max(highs) + .015, position, status, va="center", fontsize=5.8,
                  color="#2E7D5B" if status == "supported" else "#6A7076")
    axis.axvline(0, color="#333333", lw=.8, ls="--")
    axis.set_yticks(positions, [f"{target_labels[row.target]} · seed {int(row.seed)}" for row in ordered], fontsize=6.5)
    axis.set_xlabel("Curvature-regularized - baseline 32D state\nstandardized MAE")
    axis.set_title("a  Event prediction fidelity", loc="left", fontsize=9, fontweight="bold")
    axis.grid(axis="x", color="#E3E6E9", lw=.45)

    for axis, metric, title in zip(
        axes[1:],
        ("latent_second_difference_rms_median", "outcome_second_difference_rms_median"),
        ("b  State trajectory", "c  Predicted outcomes"),
        strict=True,
    ):
        selected = roughness.loc[roughness.metric.eq(metric)].sort_values("seed")
        positions = np.array([1, 0])
        for position, (_, row) in zip(positions, selected.iterrows(), strict=True):
            color = "#3B6F8F" if row.seed == 0 else "#D28E4B"
            axis.errorbar(row.estimate, position,
                          xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
                          fmt="o", color=color, ecolor="#8A9096", capsize=3, lw=1.1)
        axis.axvline(0, color="#333333", lw=.8, ls="--")
        axis.set_yticks(positions, ["Seed 0", "Seed 1"])
        axis.set_xlabel(
            "Curvature-regularized - baseline 32D state\n"
            "10-s second-difference RMS"
        )
        axis.set_title(title, loc="left", fontsize=9, fontweight="bold")
        axis.grid(axis="x", color="#E3E6E9", lw=.45)
    figure.suptitle("What changes when curvature regularization is added to the 32D state?",
                    fontsize=11, fontweight="bold")
    figure.text(.5, .91,
                "Discrete paired contrasts | 101 events, 21 experiments | 19 stopping experiments | negative = lower error or variation",
                ha="center", color="#5E646B", fontsize=7)
    figure.text(.5, .025,
                "5% event-MAE noninferiority: seed 0 supports electricity, compressor electricity and duration; other target-seed cases are insufficient evidence.",
                ha="center", color="#5E646B", fontsize=6.5)
    figure.tight_layout(rect=(0, .07, 1, .88), w_pad=2.0)
    _overview_export(figure, output, OVERVIEW_FIGURE_STEMS[6])


def _overview_transfer_coverage(transfer_root, output):
    average = pd.read_csv(transfer_root / "selection_summary.csv").set_index("method")
    method_order = ["d32_ridge_ch_mlp", "t32_ridge_ch_mlp", "raw_ridge_ch_mlp"]
    styles = {
        "raw_ridge_ch_mlp": ("#8A6FA8", "-"),
        "d32_ridge_ch_mlp": ("#777777", "-"),
        "t32_ridge_ch_mlp": ("#D28E4B", "-"),
    }
    sources = {
        seed: pd.read_csv(
            transfer_root / f"seed_{seed}" / "performance_coverage_source_first_positive.csv"
        )
        for seed in (0, 1)
    }
    summaries = {
        seed: pd.read_csv(
            transfer_root / f"seed_{seed}" / "performance_summary_first_positive.csv"
        ).set_index("method")
        for seed in (0, 1)
    }
    full_limit = max(
        5.,
        *(float(source.loss_threshold_percent.max()) for source in sources.values()),
    )
    figure = plt.figure(figsize=(10.5, 8.0))
    grid = figure.add_gridspec(3, 2, height_ratios=[1, 1, .42], hspace=.42, wspace=.24)
    axes = np.array([[figure.add_subplot(grid[row, column]) for column in range(2)] for row in range(2)])
    for seed in (0, 1):
        for method in method_order:
            curve = sources[seed].loc[sources[seed].method.eq(method)]
            summary = summaries[seed].loc[method]
            other = int(summary.cycles - summary.no_trigger_cycles - summary.evaluated_cycles)
            label = (
                f"{OVERVIEW_METHOD_LABELS[method]} "
                f"({int(summary.evaluated_cycles)}/91 evaluated; "
                f"{int(summary.no_trigger_cycles)} no trigger; {other} other)"
            )
            for axis in axes[seed]:
                axis.step(
                    curve.loss_threshold_percent,
                    curve.coverage_fraction,
                    where="post",
                    color=styles[method][0],
                    ls=styles[method][1],
                    lw=1.5,
                    label=label,
                )
        axes[seed, 0].set_xlim(0, full_limit * 1.02)
        axes[seed, 1].set_xlim(0, 5)
        for axis in axes[seed]:
            axis.set(
                ylim=(0, 1.02),
                xlabel="Maximum one-sided relative C/H loss (%)",
                ylabel="Evaluable low-loss cycles / 91",
            )
            axis.grid(color="#E3E6E9", lw=.5)
        axes[seed, 0].text(-.20, .5, f"Seed {seed}", transform=axes[seed, 0].transAxes,
                           rotation=90, va="center", ha="center", fontsize=8, fontweight="bold")
        handles, labels = axes[seed, 0].get_legend_handles_labels()
        axes[seed, 0].legend(handles, labels, loc="lower right", fontsize=5.8, frameon=False)
    axes[0, 0].set_title("a  Complete observed loss range", fontsize=9, fontweight="bold")
    axes[0, 1].set_title("b  0-5% loss detail", fontsize=9, fontweight="bold")
    summary_axis = figure.add_subplot(grid[2, :])
    summary_axis.axis("off")
    table_values = []
    for method in method_order:
        row = average.loc[method]
        table_values.append([
            OVERVIEW_METHOD_LABELS[method],
            f"{100 * (1 - row.unevaluable_fraction):.2f}%",
            f"{row.p90_loss:.2f}%",
            f"{row.median_loss:.2f}%",
        ])
    table = summary_axis.table(
        cellText=table_values,
        colLabels=["Policy input", "Evaluability", "Mean P90 L", "Mean median L"],
        cellLoc="center", colLoc="center", loc="upper center",
        colWidths=[.42, .18, .18, .18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.35)
    for column in range(4):
        table[(0, column)].set_text_props(fontweight="bold")
        table[(0, column)].set_facecolor("#EEF2F5")
    figure.suptitle(
        "Complete-policy comparison at the first non-negative score",
        fontsize=11, fontweight="bold", y=.99,
    )
    figure.text(.5, .955,
                "Same Ridge C/H + 16-unit MLP | fixed denominator: 91 cycles | metrics computed within seed, then averaged",
                ha="center", color="#5E646B", fontsize=7)
    figure.text(.5, .018,
                "The two 32D states tie on evaluability; the curvature-regularized state is retained by the prespecified next criterion, mean P90 L.",
                ha="center", color="#5E646B", fontsize=6.5)
    figure.subplots_adjust(left=.10, right=.98, bottom=.08, top=.91)
    _overview_export(figure, output, OVERVIEW_FIGURE_STEMS[3])
    expected = {
        "d32_ridge_ch_mlp": 2.553322,
        "t32_ridge_ch_mlp": 2.518558,
        "raw_ridge_ch_mlp": 2.107985,
    }
    for method, value in expected.items():
        if not np.isclose(float(average.loc[method, "p90_loss"]), value, atol=5e-6):
            raise ValueError(f"unexpected transfer P90 for {method}")


def render_overview_figures(runs, output):
    """Render four main figures and five diagnostic figures from frozen tables."""
    if len(runs) != 7:
        raise ValueError(
            "overview requires seven runs: policy, probe, neural, cross-input, "
            "outcome, state, transfer"
        )
    policy, probe, neural, cross, outcome, state, transfer = map(Path, runs)
    output = Path(output)
    _overview_objective_tradeoffs(policy, output)
    _overview_ridge_calibration(probe, output)
    _overview_event_calibration(probe, output)
    _overview_timing_consequence(transfer, output)
    _overview_online_inputs(outcome, probe, output)
    _overview_state_intervention(state, output)
    _overview_transfer_coverage(transfer, output)
    _overview_reference_shift(neural, output)
    _overview_fixed_predictor_swap(cross, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representations", type=Path)
    parser.add_argument("--rb-triggers", type=Path)
    parser.add_argument("--local-window-minutes", type=float, default=15)
    args = parser.parse_args()
    tables = {name: [] for name in ("predictions", "losses", "pair_metrics")}
    selected_inputs = []
    for run in args.runs:
        seed = json.loads((run / "settings.json").read_text())["seed"]
        for name in tables:
            path = run / (f"{name}.parquet" if name == "predictions" else f"{name}.csv")
            frame = pd.read_parquet(path) if name == "predictions" else pd.read_csv(path)
            tables[name].append(frame.assign(seed=seed))
        selected = run / "selected_inputs.csv"
        if selected.exists():
            selected_inputs.append(pd.read_csv(selected))
    predictions, losses, pairs = [pd.concat(tables[name], ignore_index=True) for name in tables]
    base = pd.read_parquet(args.data / "base.parquet")
    teachers, teacher_curves = [], []
    for heldout in predictions.heldout_experiment.unique():
        fold = pd.read_parquet(args.data / "teachers" / f"{heldout}.parquet")
        overlap = base.columns.intersection(fold.columns).difference(["row_id"])
        values = base.drop(columns=overlap).merge(fold, on="row_id", validate="one_to_one")
        selected = values.loc[values.is_knee & values.experiment_id.eq(heldout)]
        teachers.append(selected.assign(heldout_experiment=heldout))
        teacher_curves.append(values.loc[
            values.is_teacher_candidate & values.experiment_id.eq(heldout)
        ].assign(heldout_experiment=heldout))
    rb_triggers = pd.read_csv(args.rb_triggers) if args.rb_triggers else None
    render_figures(
        predictions, pd.concat(teachers), losses, pairs, args.output,
        teacher_curves=pd.concat(teacher_curves), rb_triggers=rb_triggers,
        local_window_minutes=args.local_window_minutes,
        selected_inputs=pd.concat(selected_inputs, ignore_index=True)
        if selected_inputs else None,
    )
    if args.representations:
        render_outcome_figures(
            pd.read_csv(args.representations / "outcome_predictions.csv"),
            pd.read_csv(args.representations / "losses.csv"),
            args.output,
        )
    render_data_availability(
        pd.read_csv(args.data / "cycle_coverage.csv"), predictions, args.output
    )


if __name__ == "__main__":
    main()


def render_cop_reference(predictions, output):
    """Absolute reference COP overview, also usable before neural refitting."""
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / 'fig2_source.csv', index=False)
    setpoints = predictions.groupby('cycle_name').water_temperature_setpoint.median()
    if not setpoints.isin([50, 55, 60]).all():
        raise ValueError('Fig.2 requires verified cycle setpoints of 50, 55 or 60 °C')
    supported = predictions.loc[
        predictions.cycle_cop_eligible & np.isfinite(predictions.cycle_cop)
    ].sort_values('candidate_defrost_time').reset_index(drop=True)
    optima = supported.loc[supported.groupby('cycle_name').cycle_cop.idxmax()].copy()
    optima['water_temperature_setpoint'] = optima.cycle_name.map(setpoints)
    optima = optima.sort_values(['water_temperature_setpoint', 'elapsed_minutes', 'cycle_name'])
    optima['at_supported_end'] = optima.candidate_defrost_time.eq(
        optima.cycle_name.map(supported.groupby('cycle_name').candidate_defrost_time.max())
    )
    optima.to_csv(output / 'fig2_optima.csv', index=False)
    # PINN4SOH-style trajectories, with room for the denser heat-pump cohort.
    with plt.rc_context({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 10, 'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.spines.bottom': True, 'axes.spines.left': True,
        'axes.linewidth': .7, 'axes.grid': False, 'text.usetex': False,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'xtick.major.width': .5, 'ytick.major.width': .5,
        'xtick.minor.width': .5, 'ytick.minor.width': .5,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    }):
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
        handles = []
        for ts, color, marker in zip((50, 55, 60),
                                     ('#4477AA', '#228833', '#CC6677'), ('o', 'v', 'D')):
            subset = predictions.loc[predictions.cycle_name.isin(setpoints.index[setpoints.eq(ts)])]
            for _, curve in subset.groupby('cycle_name'):
                curve = curve.sort_values('candidate_defrost_time')
                for supported, style in ((True, '-'), (False, '--')):
                    ax.plot(curve.elapsed_minutes,
                            curve.cycle_cop.where(curve.cycle_cop_eligible.eq(supported)),
                            color=color, alpha=.23, lw=.45, ls=style,
                            marker=marker, markersize=1.5, markeredgewidth=0, markevery=100)
            peaks = optima.loc[optima.water_temperature_setpoint.eq(ts)]
            ax.plot(peaks.elapsed_minutes, peaks.cycle_cop, color=color, lw=1.2,
                    marker=marker, markersize=4, markeredgecolor='white',
                    markeredgewidth=.4, zorder=5)
            handles.append(plt.Line2D([0], [0], color=color, lw=1,
                                     marker=marker, markersize=2.5,
                                     label=f'$T_s$ = {ts} °C'))
        ax.set(xlabel='Time since recovery (min)', ylabel='Effective COP')
        ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, 1.025),
                  frameon=False, ncol=3, fontsize=11, columnspacing=3, handlelength=2.5)
        ax.text(.01, .02, 'Markers: supported optima; connected in time order',
                transform=ax.transAxes, fontsize=8, color='.3')
        fig.tight_layout()
        _export(fig, output / 'fig2_effective_cop', formats=('png','pdf','svg'))


def render_relative_cop(predictions, metrics, output):
    """Reference trajectories and genuinely held-out regression through one renderer."""
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / 'figure_source.csv', index=False)
    metrics.to_csv(output / 'cycle_metrics.csv', index=False)
    render_cop_reference(predictions, output)
    for name, curve in predictions.groupby('cycle_name'):
        curve = curve.sort_values('candidate_defrost_time')
        record = metrics.loc[metrics.cycle_name.eq(name)].iloc[0]
        fig, ax = plt.subplots(figsize=(7, 4))
        t = curve.elapsed_minutes
        ax.plot(t, curve.relative_reference.where(~curve.cycle_cop_eligible),
                color='#3C6E8F', ls='--', lw=1, label='Reference outside support')
        ax.plot(t, curve.target, color='#3C6E8F', lw=1.1, label='Reference relative COP')
        ax.plot(t, curve.prediction, color='#C77836', lw=1, label='Held-out prediction')
        for level in (1., .99, .98, .95):
            ax.axhline(level, color='.65', ls=':', lw=.6)
        origin = pd.Timestamp(curve.candidate_defrost_time.iloc[0])-pd.Timedelta(minutes=t.iloc[0])
        for field, label, color, style in (
            ('reference_time', 'Reference optimum', '#3C6E8F', ':'),
            ('predicted_time', 'Predicted optimum', '#C77836', '-.'),
            ('t_RB', 'RB defrost trigger', '#287D68', '--'),
        ):
            value = curve.t_RB.iloc[0] if field == 't_RB' else record.get(field)
            if pd.notna(value):
                ax.axvline((pd.Timestamp(value)-origin).total_seconds()/60,
                           color=color, ls=style, lw=1, label=label)
        ax.set(xlabel='Time since recovery completion [min]',
               ylabel='Effective cycle COP / supported optimum',
               title=f'{name} · {record["status"]}')
        ax.legend(fontsize=6, loc='best')
        fig.tight_layout()
        _export(fig, output / f'{name}_relative_cop')


def _render_binary_cycle_probability_job(
    loader,
    cycle_name,
    decisions,
    traces,
    metrics,
    output,
    name_suffix,
    formats,
):
    from plots.publication import render_effective_cop_probability

    curve = decisions.loc[decisions.cycle_name.eq(cycle_name)].copy()
    trace = traces.loc[traces.cycle_name.eq(cycle_name)].copy()
    metric = metrics.loc[metrics.cycle_name.eq(cycle_name)]
    if curve.empty or trace.empty or len(metric) != 1:
        raise ValueError(
            f"{cycle_name}: expected one metric and non-empty decision/trace curves"
        )
    render_effective_cop_probability(
        loader.load_cycle(cycle_name),
        curve,
        trace,
        metric.iloc[0],
        output / f"{cycle_name}_chen_probability{name_suffix}",
        formats=formats,
    )
    return cycle_name


def render_binary_cycle_probabilities(
    run,
    dataset,
    decision_run,
    output,
    n_jobs=6,
    *,
    valid_only=False,
    formats=("png", "pdf"),
    name_suffix="",
    save_sources=True,
):
    """Render every frozen Chen-inspired binary cycle through publication styling."""
    from joblib import Parallel, delayed, parallel_config

    from dataset_tools import DatasetLoader

    run, dataset, decision_run, output = map(Path, (run, dataset, decision_run, output))
    settings = json.loads((run / "settings.json").read_text())
    expected = {
        "task": "effective-cop-binary",
        "prediction_kind": "positive_probability",
        "trigger_threshold": 0.5,
        "confirmation": "two_of_three",
        "processing_seconds": 30,
    }
    mismatched = {
        key: settings.get(key) for key, value in expected.items() if settings.get(key) != value
    }
    if mismatched:
        raise ValueError(f"binary run does not match the frozen trigger contract: {mismatched}")

    traces = pd.read_parquet(run / "online_trace.parquet")
    metrics = pd.read_csv(run / "cycle_metrics.csv")
    decisions = pd.read_csv(decision_run / "candidate_decisions.csv", low_memory=False)
    for frame in (traces, decisions):
        frame["candidate_defrost_time"] = pd.to_datetime(
            frame["candidate_defrost_time"], errors="coerce"
        )
    for field in ("trigger_time", "reference_time"):
        if field in metrics:
            metrics[field] = pd.to_datetime(metrics[field], errors="coerce")
    for field in ("t_star", "t_RB"):
        if field in decisions:
            decisions[field] = pd.to_datetime(decisions[field], errors="coerce")

    names = set(traces.cycle_name.astype(str).unique())
    if valid_only:
        valid = DatasetLoader(dataset).list_valid_cycles(require_rgb=True)
        names &= set(valid.loc[valid.front_rgb_valid.eq(True), "cycle_name"])
    names = sorted(names)
    decisions = decisions.loc[decisions.cycle_name.astype(str).isin(names)].copy()
    metrics = metrics.loc[metrics.cycle_name.astype(str).isin(names)].copy()
    if set(names) != set(decisions.cycle_name.astype(str)) or set(names) != set(
        metrics.cycle_name.astype(str)
    ):
        raise ValueError("trace, effective-COP decisions, and cycle metrics have different cohorts")
    output.mkdir(parents=True, exist_ok=True)
    if save_sources:
        traces.to_parquet(output / "probability_source.parquet", index=False)
        decisions.to_parquet(output / "effective_cop_source.parquet", index=False)
        metrics.to_csv(output / "cycle_status.csv", index=False)

    loader = DatasetLoader(dataset)
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        rendered = list(
            Parallel(return_as="generator_unordered")(
                delayed(_render_binary_cycle_probability_job)(
                    loader,
                    name,
                    decisions,
                    traces,
                    metrics,
                    output,
                    name_suffix,
                    formats,
                )
                for name in names
            )
        )
    print(f"Rendered {len(rendered)} Chen-inspired cycle probability figures to {output}")
    return rendered


def render_relative_cop_comparison(runs, output):
    """Own-coverage and common-timestamp comparisons without changing reference peaks."""
    from image_models.relative_cop import cycle_metrics

    output.mkdir(parents=True, exist_ok=True)
    names, curves, settings_list, reference = [], [], [], None
    for run in runs:
        settings = json.loads((run / "settings.json").read_text())
        names.append(settings["model_name"])
        settings_list.append(settings)
        curve = pd.read_parquet(run / "predictions.parquet").sort_values(
            ["cycle_name", "candidate_defrost_time"]).reset_index(drop=True)
        shared = curve[["cycle_name", "candidate_defrost_time", "target", "cycle_cop",
                        "cycle_cop_eligible"]]
        if reference is not None:
            pd.testing.assert_frame_equal(reference, shared)
        reference = shared
        for previous, configuration in zip(curves, settings_list[:-1], strict=True):
            if configuration.get("rgb", "on") == settings.get("rgb", "on"):
                pd.testing.assert_series_equal(previous.input_available, curve.input_available)
        curves.append(curve)
    if len(set(names)) != len(names):
        raise ValueError("comparison needs distinct model names")
    common = np.logical_and.reduce([c.prediction.notna() & c.target.notna() for c in curves])
    tables = []
    for name, curve in zip(names, curves, strict=True):
        for domain, rows in (("own_inputs", curve),
                             ("common_timestamps", curve.assign(prediction=curve.prediction.where(common)))):
            tables.append(cycle_metrics(rows).assign(model=name, domain=domain))
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(output / "model_comparison_cycles.csv", index=False)
    columns = ["mse", "rmse", "mae", "relative_cop_loss", "network_cop_loss",
               "input_coverage_loss", "within_1pct", "within_2pct", "within_5pct",
               "absolute_time_error_minutes", "chosen_vs_rb_pct",
               *[f"near_{p}pct_{d}_minutes" for p in (1, 2, 5) for d in ("early", "late")]]
    summary = metrics.groupby(["model", "domain"], sort=False)[columns].mean()
    summary["cohort_cycles"] = metrics.groupby(["model", "domain"]).size()
    summary["evaluated_cycles"] = metrics.assign(ok=metrics.status.eq("evaluated")).groupby(
        ["model", "domain"]).ok.sum()
    summary.to_csv(output / "model_comparison.csv")
    reference.assign(common_input=common).to_csv(output / "comparison_support.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    colors = ("#7699B2", "#C49474")
    for ax, metric, label, scale in zip(
        axes, ("mse", "relative_cop_loss", "within_2pct"),
        ("Cycle MSE", "Selected COP loss [%]", "Within 2% of optimum [%]"), (1, 100, 100), strict=True
    ):
        for offset, domain, color in zip((-.18, .18), ("own_inputs", "common_timestamps"), colors, strict=True):
            values = summary.xs(domain, level="domain").reindex(names)[metric] * scale
            ax.bar(np.arange(len(names)) + offset, values, width=.35, color=color,
                   label=domain.replace("_", " "))
        ax.set(xticks=np.arange(len(names)), xticklabels=names, ylabel=label)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle("Held-out COP prediction and decision accuracy")
    fig.tight_layout()
    _export(fig, output / "model_comparison", formats=("png", "pdf", "svg"))
    return summary


def _render_headroom_comparison(metrics, output):
    """Compare realizable Ridge-reference headroom with each online decision."""
    source = metrics.copy().reset_index(drop=True)
    for column in ("reference_cop", "baseline_rb_cop", "trigger_cop"):
        if column not in source:
            source[column] = np.nan
    if "outside_reference_support" not in source:
        source["outside_reference_support"] = pd.NA

    def optional_bool(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, str):
            return {"true": True, "false": False}.get(value.strip().lower(), pd.NA)
        return bool(value)

    def support_value(values):
        known = values.map(optional_bool).dropna().astype(bool)
        if known.any():
            return True
        return False if len(known) == len(values) and len(values) else pd.NA

    rb_outside = (
        source.loc[source.model.eq("RB")]
        .groupby("cycle_name").outside_reference_support.apply(support_value)
    )
    source["rb_outside_reference_support"] = source.cycle_name.map(rb_outside)

    def support_state(row):
        values = tuple(map(optional_bool, (
            row.outside_reference_support, row.rb_outside_reference_support
        )))
        if any(pd.notna(value) and value for value in values):
            return "outside"
        return "in" if all(pd.notna(value) for value in values) else "unknown"

    source["support_state"] = source.apply(support_state, axis=1)
    baseline = pd.to_numeric(source.baseline_rb_cop, errors="coerce")
    reference = pd.to_numeric(source.reference_cop, errors="coerce")
    trigger = pd.to_numeric(source.trigger_cop, errors="coerce")
    valid_baseline = np.isfinite(baseline) & baseline.gt(0)
    source["reference_headroom_percent"] = np.where(
        valid_baseline & np.isfinite(reference),
        100 * (reference - baseline) / baseline,
        np.nan,
    )
    source["realized_gain_percent"] = np.where(
        valid_baseline & np.isfinite(trigger),
        100 * (trigger - baseline) / baseline,
        np.nan,
    )

    reasons = []
    for index, row in source.iterrows():
        if row.model == "RB":
            reason = "baseline_row"
        elif row.status not in {"scored", "trigger_outside_reference_support"}:
            reason = str(row.status)
        elif not valid_baseline.loc[index]:
            reason = "baseline_rb_cop_unavailable"
        elif not np.isfinite(reference.loc[index]):
            reason = "reference_cop_unavailable"
        elif not np.isfinite(trigger.loc[index]):
            reason = "trigger_cop_unavailable"
        else:
            reason = ""
        reasons.append(reason)
    source["missing_reason"] = reasons
    source["plot_included"] = source.missing_reason.eq("")
    source.to_csv(Path(output) / "headroom_points.csv", index=False)

    plotted = source.loc[source.plot_included]
    model_order = source.loc[source.model.ne("RB"), "model"].drop_duplicates().tolist()
    fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {
        model: STYLES.get(model, (fallback[index % len(fallback)], "o"))[0]
        for index, model in enumerate(model_order)
    }
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for model in model_order:
        rows = plotted.loc[plotted.model.eq(model)]
        for state, marker in (("in", "o"), ("outside", "x"), ("unknown", "o")):
            points = rows.loc[rows.support_state.eq(state)]
            if points.empty:
                continue
            kwargs = {"s": 28, "marker": marker, "color": colors[model], "alpha": .85}
            if state == "unknown":
                kwargs.update(facecolors="none", edgecolors=colors[model])
                kwargs.pop("color")
            ax.scatter(points.reference_headroom_percent, points.realized_gain_percent,
                       **kwargs)

    def axis_limits(values):
        values = values.dropna()
        if values.empty:
            return (-1., 1.)
        lower, upper = min(0., values.min()), max(0., values.max())
        padding = max(1., .06 * (upper - lower))
        return (lower - padding, upper + padding)

    x_limits = axis_limits(plotted.reference_headroom_percent)
    y_limits = axis_limits(plotted.realized_gain_percent)
    identity_line, = ax.plot(
        x_limits, x_limits, color=".45", ls="--", lw=.8,
        label="Realized = headroom",
    )
    zero_line = ax.axhline(0, color=".65", ls=":", lw=.8, label="Zero gain")
    ax.set(
        xlim=x_limits, ylim=y_limits,
        xlabel="Ridge-supported reference headroom vs RB [%]",
        ylabel="Selected-trigger Ridge COP gain vs RB [%]",
        title="Reference-estimated headroom and selected-trigger gain",
    )
    model_handles = [
        plt.Line2D([0], [0], marker="o", lw=0, color=colors[model], label=model,
                   markersize=5)
        for model in model_order
    ]
    support_handles = [
        plt.Line2D([0], [0], marker="o", lw=0, color=".35",
                   label="In support (filled)", markersize=5),
        plt.Line2D([0], [0], marker="x", lw=0, color=".35",
                   label="Outside support", markersize=5),
        plt.Line2D([0], [0], marker="o", lw=0, color=".35",
                   label="Support unknown (open)", markerfacecolor="none", markersize=5),
    ]
    ax.legend(
        handles=model_handles + support_handles + [identity_line, zero_line],
        fontsize=7, loc="best",
    )
    counts = []
    for model in model_order:
        rows = source.loc[source.model.eq(model)]
        available = int(rows.plot_included.sum())
        counts.append(f"{model}: {available} available / {len(rows) - available} unscored")
    fig.text(.5, .035, "\n".join(counts) if counts else "No non-RB model rows",
             ha="center", va="bottom", fontsize=7, color=".25")
    fig.text(.5, .005,
             "Headroom is the Ridge support-domain reference, not a physical upper limit; "
             "values are Ridge estimates for sensitivity analysis; open circles have unknown support.",
             ha="center", va="bottom", fontsize=6.5, color=".35")
    fig.tight_layout(rect=(0, .1, 1, 1))
    _export(fig, Path(output) / "headroom_comparison", formats=("png", "pdf", "svg"))
    return source


def render_online_cop(metrics, output, processing_seconds=30, *, traces=None):
    """One confirmation rule and one renderer for all online model decisions."""
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "online_cycle_metrics.csv", index=False)
    _render_headroom_comparison(metrics, output)
    common = metrics.groupby("cycle_name").status.apply(lambda s: s.eq("scored").all())
    common_names = set(common.index[common])
    summaries = []
    for model, rows in metrics.groupby("model", sort=False):
        for scope in ("own_scored", "common_scored"):
            selected = rows.loc[rows.status.eq("scored")]
            if scope == "common_scored":
                selected = selected.loc[selected.cycle_name.isin(common_names)]
            record = dict(
                model=model,
                scope=scope,
                cohort=len(rows),
                scored=len(selected),
                triggered=int(rows.trigger_time.notna().sum()),
                no_trigger_rate=float(rows.trigger_time.isna().mean()),
                no_trigger=int(rows.status.eq("no_trigger").sum()),
                no_input=int(rows.status.eq("no_available_input").sum()),
                no_reference=int(rows.status.eq("no_supported_reference").sum()),
                no_frozen_prediction=int(rows.status.eq("no_frozen_prediction").sum()),
                trigger_cop_unavailable=int(rows.status.eq("trigger_cop_unavailable").sum()),
                outside_support=int(rows.get("outside_reference_support", rows.status.eq("trigger_outside_reference_support")).fillna(False).sum()),
                mean_cop_loss=selected.relative_cop_loss.mean(),
                median_cop_loss=selected.relative_cop_loss.median(),
                p90_cop_loss=selected.relative_cop_loss.quantile(0.9),
                median_time_error_minutes=selected.time_error_minutes.median(),
                median_absolute_time_error_minutes=selected.absolute_time_error_minutes.median(),
                p90_absolute_time_error_minutes=selected.absolute_time_error_minutes.quantile(0.9),
                **{f"within_{p}pct": selected[f"within_{p}pct"].mean() for p in (1, 2, 5)},
                **{
                    f"hit_{p}pct_of_cohort": selected[f"within_{p}pct"].fillna(0).sum() / len(rows)
                    for p in (1, 2, 5)
                },
            )
            for column in ("balanced_accuracy", "macro_f1", "fnr", "fpr", "precision", "recall", "mse", "rmse", "mae"):
                if column in rows:
                    record[column] = rows[column].mean()
            if "regression_count" in rows:
                count = rows.regression_count.sum()
                record["global_mse"] = (rows.mse * rows.regression_count).sum() / count if count else np.nan
                record["global_rmse"] = np.sqrt(record["global_mse"])
                record["global_mae"] = (rows.mae * rows.regression_count).sum() / count if count else np.nan
                for name, numerator, denominator in (
                    ("pooled_fnr", rows.fn.sum(), rows.fn.sum() + rows.tp.sum()),
                    ("pooled_fpr", rows.fp.sum(), rows.fp.sum() + rows.tn.sum()),
                    ("pooled_precision", rows.tp.sum(), rows.tp.sum() + rows.fp.sum()),
                    ("pooled_recall", rows.tp.sum(), rows.tp.sum() + rows.fn.sum()),
                ):
                    record[name] = numerator / denominator if denominator else np.nan
            if "cop_gain_vs_rb_pct" in selected:
                record["rb_paired_cycles"] = int(selected.cop_gain_vs_rb_pct.notna().sum())
                record["mean_cop_gain_vs_rb_pct"] = selected.cop_gain_vs_rb_pct.mean()
                paired = selected.loc[selected.cop_gain_vs_rb_pct.notna()]
                record["paired_mean_model_cop"] = paired.trigger_cop.mean()
                record["paired_mean_rb_cop"] = paired.baseline_rb_cop.mean()
                record["paired_mean_cop_difference"] = (paired.trigger_cop - paired.baseline_rb_cop).mean()
            if "headroom_captured_pct" in selected:
                paired = selected.loc[selected.headroom_captured_pct.notna()]
                denominator = paired.reference_cop.mean() - paired.baseline_rb_cop.mean()
                captured = (
                    100
                    * (paired.trigger_cop.mean() - paired.baseline_rb_cop.mean())
                    / denominator
                    if len(paired) and denominator > 0
                    else np.nan
                )
                record["ideal_headroom_paired_cycles"] = len(paired)
                record["headroom_captured_pct"] = captured
                record["headroom_remaining_pct"] = 100 - captured
                record["median_cycle_headroom_captured_pct"] = (
                    paired.headroom_captured_pct.median()
                )
            record["uncalibrated"] = int(rows.status.eq("threshold_uncalibrated").sum())
            summaries.append(record)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "online_summary.csv", index=False)
    table = summary.loc[summary.scope.eq("own_scored")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, column, label, scale in zip(
        axes,
        (
            "mean_cop_loss",
            "hit_2pct_of_cohort",
            "p90_absolute_time_error_minutes",
            "no_trigger_rate",
        ),
        (
            "Mean scored COP loss [%]",
            "Within 2% / all cycles [%]",
            "P90 absolute timing error [min]",
            "Not triggered / all cycles [%]",
        ),
        (100, 100, 1, 100),
        strict=True,
    ):
        ax.bar(np.arange(len(table)), table[column] * scale, color="#7799B3")
        ax.set(xticks=np.arange(len(table)), xticklabels=table.model, ylabel=label)
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
    fig.suptitle(f"{processing_seconds}-second replay · " + ("policy-specific confirmation" if "strategy" in metrics else "two of three confirmation"))
    fig.tight_layout()
    _export(fig, output / "online_comparison", formats=("png", "pdf", "svg"))
    if "cop_gain_vs_rb_pct" in metrics:
        metrics.loc[metrics.model.ne("RB")].groupby("model").agg(
            paired_cycles=("cop_gain_vs_rb_pct", "count"),
            mean_cop_gain_pct=("cop_gain_vs_rb_pct", "mean"),
        ).to_csv(output / "rb_baseline_summary.csv")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for offset, scope, color in ((-.2, "own_scored", "#3C6E8F"), (.2, "common_scored", "#C77836")):
            paired = summary.loc[summary.scope.eq(scope)]
            bars = ax.bar(np.arange(len(paired)) + offset, paired.mean_cop_gain_vs_rb_pct,
                          width=.38, color=color, label=scope.replace("_", " "))
            ax.bar_label(bars, labels=[f"n={n}" for n in paired.rb_paired_cycles], fontsize=7)
        ax.axhline(0, color="black", lw=.7)
        ax.set(xticks=np.arange(len(table)), xticklabels=table.model,
               ylabel="Paired COP gain versus original RB [%]")
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)
        ax.legend(fontsize=8)
        fig.tight_layout()
        _export(fig, output / "rb_baseline_comparison", formats=("png", "pdf", "svg"))
    if traces is not None:
        traces.to_parquet(output / "online_trace.parquet", index=False)
        for name, group in traces.groupby("cycle_name", sort=True):
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            origin = group.candidate_defrost_time.min()
            for model, curve in group.groupby("model", sort=False):
                curve = curve.sort_values("candidate_defrost_time")
                minutes = (curve.candidate_defrost_time - origin).dt.total_seconds() / 60
                (line,) = axes[0].plot(minutes, curve.score, label=model, lw=1)
                if "threshold" in curve and np.isfinite(curve.threshold.iloc[0]):
                    axes[0].axhline(curve.threshold.iloc[0], color=line.get_color(), ls=":", lw=0.7)
                record = metrics.loc[metrics.cycle_name.eq(name) & metrics.model.eq(model)].iloc[0]
                if pd.notna(record.trigger_time):
                    at = (pd.Timestamp(record.trigger_time) - origin).total_seconds() / 60
                    for ax in axes:
                        ax.axvline(at, color=line.get_color(), ls="--", lw=1,
                                   label=f"{model} trigger")
                if "target" in curve:
                    axes[1].plot(minutes, curve.target, color="#3C6E8F", lw=1, label="Reference relative COP")
                if "probability" in curve:
                    axes[0].fill_between(minutes, 0, 1, where=curve.binary_target.eq(1),
                                         color="#287D68", alpha=.12, label="True positive region")
                    if curve.prediction.notna().any():
                        axes[1].plot(minutes, curve.prediction, color="#C77836", lw=1, label="Predicted relative COP")
            if "probability" in group:
                rb = group.t_RB.iloc[0]
                if pd.notna(rb):
                    for ax in axes:
                        ax.axvline((rb-origin).total_seconds()/60, color="#287D68", ls="--", label="Original RB trigger")
                for ax in axes:
                    handles, labels = ax.get_legend_handles_labels()
                    unique = dict(zip(labels, handles))
                    ax.legend(unique.values(), unique.keys(), fontsize=6)
            reference = metrics.loc[metrics.cycle_name.eq(name), "reference_time"].iloc[0]
            if pd.notna(reference):
                for ax in axes:
                    ax.axvline(
                        (pd.Timestamp(reference) - origin).total_seconds() / 60,
                        color="black",
                        ls="-.",
                        label="COP optimum",
                    )
            axes[0].set(ylabel="Model score (model-specific threshold)", title=name)
            axes[0].legend(fontsize=7)
            axes[1].set(xlabel="Time since first candidate [min]", ylabel="Effective COP / optimum")
            fig.tight_layout()
            _export(fig, output / f"{name}_trigger", formats=("png", "pdf"))
    return summary


def render_development_classification(summary, output):
    """Render held-out classification selection and 2/3 proposal coverage."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = summary.reset_index(drop=True).copy()
    source.to_csv(output / "development_classification_source.csv", index=False)

    x = np.arange(len(source))
    width = .24
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for offset, column, label in (
        (-width, "balanced_accuracy", "Balanced accuracy"),
        (0, "macro_f1", "Macro F1"),
        (width, "cycle_weighted_average_precision", "Cycle-weighted AP"),
    ):
        axes[0].bar(x + offset, source[column], width=width, label=label)
    axes[0].set(
        xticks=x, xticklabels=source.method,
        ylabel="Held-out classification metric", ylim=(0, 1),
        title="Fixed-fold classification selection",
    )
    axes[0].legend(fontsize=7)

    unscored = source.unscoreable_2of3_count / source.cohort_cycle_count
    for offset, values, label in (
        (-width, source.proposal_coverage, "2/3 proposal / cohort"),
        (0, source.near_hit_coverage, "Supported near hit / cohort"),
        (width, unscored, "Unscoreable / cohort"),
    ):
        axes[1].bar(x + offset, values, width=width, label=label)
    axes[1].set(
        xticks=x, xticklabels=source.method,
        ylabel="Cycle fraction", ylim=(0, 1),
        title="First 2/3 proposal and reference coverage",
    )
    axes[1].legend(fontsize=7)
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=25, labelsize=7)
    fig.tight_layout()
    _export(
        fig, output / "development_classification", formats=("png", "pdf", "svg")
    )
    return source


def render_development_common_comparison(summary, bootstrap, output):
    """Render frozen development recipes on their exact common rows."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = summary.reset_index(drop=True).copy()
    intervals = bootstrap.reset_index(drop=True).copy()
    source.to_csv(output / "development_common_comparison_source.csv", index=False)
    intervals.to_csv(
        output / "development_common_paired_bootstrap.csv", index=False
    )

    x = np.arange(len(source))
    width = .24
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for offset, column, label in (
        (-width, "balanced_accuracy", "Balanced accuracy"),
        (0, "macro_f1", "Macro F1"),
        (width, "cycle_weighted_average_precision", "Cycle-weighted AP"),
    ):
        axes[0].bar(x + offset, source[column], width=width, label=label)
    axes[0].set(ylabel="Score", ylim=(0, 1), title="Common-label classification")
    axes[0].legend(fontsize=6)

    for offset, column, label in (
        (-width / 2, "cycle_weighted_brier_score", "Brier"),
        (width / 2, "cycle_weighted_expected_calibration_error", "ECE"),
    ):
        axes[1].bar(x + offset, source[column], width=width, label=label)
    axes[1].set(ylabel="Error", title="Common-label calibration")
    axes[1].legend(fontsize=6)

    y = np.arange(len(intervals))
    axes[2].errorbar(
        intervals.mean_difference, y,
        xerr=np.vstack([
            intervals.mean_difference - intervals.ci_low,
            intervals.ci_high - intervals.mean_difference,
        ]),
        fmt="D", ms=4, color="#3C6E8F", ecolor="#777777", capsize=2,
    )
    axes[2].axvline(0, color="#333333", lw=.7, ls="--")
    axes[2].set(
        yticks=y,
        yticklabels=intervals.comparison.str.replace("_", " "),
        xlabel="RGB − Sensor balanced accuracy",
        title="Paired experiment bootstrap (95% interval)",
    )
    labels = source.recipe_name.str.replace(" · ", "\n")
    for axis in axes[:2]:
        axis.set_xticks(x, labels, rotation=30, ha="right", fontsize=6)
    figure.suptitle("Frozen classifiers on one common development clock", fontsize=10)
    figure.tight_layout()
    _export(
        figure, output / "development_common_comparison",
        formats=("png", "pdf", "svg"),
    )
    return source, intervals


def render_development_retrospective(
    cycles, output, bootstrap_replicates=2000, seed=0
):
    """Render frozen outer-fold results without treating seeds as new cycles."""
    required = {
        "recipe_id", "method", "seed", "experiment_id", "cycle_name",
        "status", "balanced_accuracy", "macro_f1", "both_classes",
    }
    missing = required - set(cycles)
    if missing:
        raise ValueError(f"retrospective cycle metrics missing: {sorted(missing)}")
    sizes = cycles.groupby(["recipe_id", "seed"]).cycle_name.nunique()
    duplicates = cycles.duplicated(["recipe_id", "seed", "cycle_name"])
    if duplicates.any() or sizes.nunique() != 1:
        raise ValueError(
            "retrospective requires one shared cycle count per recipe and seed"
        )

    rows = []
    for (recipe, method, run_seed), group in cycles.groupby(
        ["recipe_id", "method", "seed"], sort=False
    ):
        rows.append({
            "recipe_id": recipe, "method": method, "scope": "seed",
            "seed": run_seed, "cohort_cycles": group.cycle_name.nunique(),
            "evaluated_cycles": int(group.status.eq("evaluated").sum()),
            "no_supported_labels_cycles": int(
                group.status.eq("no_supported_labels").sum()
            ),
            "no_prediction_cycles": int(group.status.eq("no_prediction").sum()),
            "both_class_cycles": int(group.both_classes.fillna(False).sum()),
            "balanced_accuracy": group.balanced_accuracy.mean(),
            "macro_f1": group.macro_f1.mean(),
        })
    for (recipe, method), group in cycles.groupby(["recipe_id", "method"], sort=False):
        averaged = group.groupby("cycle_name")[[
            "balanced_accuracy", "macro_f1"
        ]].mean()
        rows.append({
            "recipe_id": recipe, "method": method, "scope": "pooled",
            "seed": np.nan, "cohort_cycles": len(averaged),
            "evaluated_cycles": int(averaged.balanced_accuracy.notna().sum()),
            "no_supported_labels_cycles": np.nan, "no_prediction_cycles": np.nan,
            "both_class_cycles": np.nan,
            "balanced_accuracy": averaged.balanced_accuracy.mean(),
            "macro_f1": averaged.macro_f1.mean(),
        })
    summary = pd.DataFrame(rows)

    pivot = cycles.pivot(
        index=["seed", "experiment_id", "cycle_name"], columns="recipe_id",
        values="balanced_accuracy",
    )
    needed = ["history_sensor", "delta_rgb"]
    if not set(needed) <= set(pivot):
        raise ValueError("retrospective requires history_sensor and delta_rgb")
    paired = pivot[needed].dropna()
    per_cycle = (
        (paired.delta_rgb - paired.history_sensor)
        .groupby(level=["experiment_id", "cycle_name"])
        .agg(["mean", "count"])
    )
    difference = per_cycle.loc[
        per_cycle["count"].eq(cycles.seed.nunique()), "mean"
    ]
    clusters = difference.groupby(level="experiment_id").agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(clusters), size=(bootstrap_replicates, len(clusters))
    )
    draws = (
        clusters["sum"].to_numpy()[sampled].sum(axis=1)
        / clusters["count"].to_numpy()[sampled].sum(axis=1)
    )
    bootstrap = pd.DataFrame([{
        "comparison": "delta_rgb_minus_history_sensor",
        "metric": "balanced_accuracy",
        "mean_difference": difference.mean(),
        "ci_low": np.quantile(draws, .025),
        "ci_high": np.quantile(draws, .975),
        "paired_cycles": len(difference),
        "experiments": len(clusters),
        "seeds_per_cycle": cycles.seed.nunique(),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "scope": "descriptive_frozen_retrospective_uncertainty",
    }])

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "development_frozen_retrospective_source.csv", index=False)
    bootstrap.to_csv(
        output / "development_frozen_retrospective_paired_bootstrap.csv", index=False
    )
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    seeded = summary.loc[summary.scope.eq("seed")]
    for (recipe, metric), group in seeded.melt(
        id_vars=["recipe_id", "seed"],
        value_vars=["balanced_accuracy", "macro_f1"],
        var_name="metric", value_name="value",
    ).groupby(["recipe_id", "metric"], sort=False):
        axes[0].plot(
            group.seed, group.value, marker="o",
            label=f"{recipe.replace('_', ' ')} · {metric.replace('_', ' ')}",
        )
    axes[0].set(
        xlabel="Seed", ylabel="Cycle-average score", ylim=(0, 1),
        xticks=sorted(seeded.seed.unique()), title="Per-seed outer-fold scores",
    )
    axes[0].legend(fontsize=5)
    def count_text(column, label):
        values = seeded[column].dropna().astype(int)
        count = str(values.iloc[0]) if values.nunique() == 1 else f"{values.min()}–{values.max()}"
        return f"{count} {label}"
    axes[0].text(
        .02, .03,
        " · ".join([
            f"{int(sizes.iloc[0])} cycles/recipe–seed",
            count_text("evaluated_cycles", "evaluated"),
            count_text("no_supported_labels_cycles", "no labels"),
            count_text("no_prediction_cycles", "no prediction"),
        ]),
        transform=axes[0].transAxes, fontsize=6,
    )

    pooled = summary.loc[summary.scope.eq("pooled")].reset_index(drop=True)
    x = np.arange(len(pooled))
    axes[1].bar(x - .18, pooled.balanced_accuracy, width=.36, label="Balanced accuracy")
    axes[1].bar(x + .18, pooled.macro_f1, width=.36, label="Macro F1")
    axes[1].set(
        xticks=x, xticklabels=pooled.recipe_id.str.replace("_", " "),
        ylabel="Seed-averaged cycle score", ylim=(0, 1),
        title="Pooled frozen retrospective",
    )
    axes[1].tick_params(axis="x", labelrotation=20, labelsize=7)
    axes[1].legend(fontsize=6)

    row = bootstrap.iloc[0]
    axes[2].errorbar(
        row.mean_difference, 0,
        xerr=[[row.mean_difference - row.ci_low], [row.ci_high - row.mean_difference]],
        fmt="D", ms=4, color="#3C6E8F", ecolor="#777777", capsize=2,
    )
    axes[2].axvline(0, color="#333333", lw=.7, ls="--")
    axes[2].set(
        yticks=[], xlabel="Delta RGB − History Sensor BA",
        title="Experiment-cluster bootstrap\n(seed-averaged cycles; 95% interval)",
    )
    axes[2].text(
        .03, .05,
        f"{int(row.paired_cycles)} paired cycles · {int(row.experiments)} experiments · "
        f"{int(row.seeds_per_cycle)} seeds/cycle",
        transform=axes[2].transAxes, fontsize=6,
    )
    figure.suptitle("Frozen outer-fold classifier retrospective", fontsize=10)
    figure.tight_layout()
    _export(
        figure, output / "development_frozen_retrospective",
        formats=("png", "pdf", "svg"),
    )
    return summary, bootstrap


def render_frozen_policy_cop_comparison(metrics, output):
    """Compare five frozen policies using one peak and explicit point-support scopes."""
    required = {
        "policy_id", "policy_name", "policy_family", "seed", "cycle_name",
        "experiment_id", "status", "estimate_scope", "cop_gain_vs_rb_pct",
        "formal_reference_headroom_vs_rb_pct",
        "sensitivity_cop_gain_vs_rb_pct",
        "sensitivity_reference_headroom_vs_rb_pct",
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"frozen policy metrics missing: {sorted(missing)}")
    source = metrics.copy()
    source["_seed"] = pd.to_numeric(source.seed, errors="coerce").fillna(-1).astype(int)
    if source.duplicated(["policy_id", "_seed", "cycle_name"]).any():
        raise ValueError("frozen policy metrics contain duplicate policy-seed cycles")
    cohorts = source.groupby(["policy_id", "_seed"]).cycle_name.apply(frozenset)
    if len(set(cohorts)) != 1:
        raise ValueError("frozen policies must use the same cycle cohort for every seed")
    policy_order = source.loc[
        source.policy_family.ne("baseline"), "policy_id"
    ].drop_duplicates().tolist()
    names = source.groupby("policy_id").policy_name.first()

    status_seed = source.loc[source.policy_id.isin(policy_order)].assign(
        before=lambda x: x.status.eq("before_reference_accounting_start"),
        other_unscored=lambda x: x.estimate_scope.eq("unscored") & ~x.before,
    ).groupby(["policy_id", "_seed"]).agg(
        cohort_cycles=("cycle_name", "size"),
        before_accounting_cycles=("before", "sum"),
        other_unscored_cycles=("other_unscored", "sum"),
    ).reset_index()

    scopes = (
        (
            "formal_supported_only", "ridge_domain_supported_points",
            "cop_gain_vs_rb_pct", "formal_reference_headroom_vs_rb_pct",
        ),
        (
            "reliable_extrapolation_sensitivity",
            "reliable_points_with_ridge_domain_relaxed",
            "sensitivity_cop_gain_vs_rb_pct",
            "sensitivity_reference_headroom_vs_rb_pct",
        ),
    )
    rows, complete_values, complete_counts = [], {}, {}

    def add_row(policy, scope, surface, aggregation, values, counts):
        statuses = status_seed.loc[status_seed.policy_id.eq(policy)]
        gain, headroom = values.columns
        gain_ok, headroom_ok = values[gain].notna(), values[headroom].notna()
        joint = gain_ok & headroom_ok
        gap = joint & values[headroom].gt(1)
        gain_seeds = counts.loc[gain_ok, gain]
        headroom_seeds = counts.loc[headroom_ok, headroom]
        rows.append({
            "policy_id": policy, "policy_name": names[policy],
            "policy_family": source.loc[
                source.policy_id.eq(policy), "policy_family"
            ].iloc[0],
            "evaluation_scope": scope, "point_support_scope": surface,
            "aggregation": aggregation, "seed_count": len(statuses),
            "cohort_cycles": int(statuses.cohort_cycles.iloc[0]),
            "before_accounting_cycles_per_seed": statuses.before_accounting_cycles.mean(),
            "other_unscored_cycles_per_seed": statuses.other_unscored_cycles.mean(),
            "gain_paired_cycles": int(gain_ok.sum()),
            "headroom_paired_cycles": int(headroom_ok.sum()),
            "gain_and_headroom_paired_cycles": int(joint.sum()),
            "gap_eligible_cycles": int(gap.sum()),
            "gain_available_seeds_min": int(gain_seeds.min()) if len(gain_seeds) else 0,
            "gain_available_seeds_max": int(gain_seeds.max()) if len(gain_seeds) else 0,
            "headroom_available_seeds_min": (
                int(headroom_seeds.min()) if len(headroom_seeds) else 0
            ),
            "headroom_available_seeds_max": (
                int(headroom_seeds.max()) if len(headroom_seeds) else 0
            ),
            "mean_cop_gain_pct": values.loc[gain_ok, gain].mean(),
            "mean_reference_headroom_pct": values.loc[headroom_ok, headroom].mean(),
            "mean_remaining_gap_on_rb_denominator_pp": (
                values.loc[gap, headroom] - values.loc[gap, gain]
            ).mean(),
        })

    policy_rows = source.loc[source.policy_id.isin(policy_order)]
    for policy in policy_order:
        raw_policy = policy_rows.loc[policy_rows.policy_id.eq(policy)]
        expected_seeds = raw_policy._seed.nunique()
        for scope, surface, gain, headroom in scopes:
            for run_seed, seed_rows in raw_policy.groupby("_seed", sort=True):
                values = seed_rows.set_index("cycle_name")[[gain, headroom]]
                add_row(
                    policy, scope, surface,
                    f"own_seed_{int(run_seed)}" if run_seed >= 0 else "own_single_seed",
                    values, values.notna().astype(int),
                )
            grouped = raw_policy.groupby("cycle_name")[[gain, headroom]]
            values, counts = grouped.mean(), grouped.count()
            values = values.where(counts.eq(expected_seeds))
            complete_values[scope, policy] = values
            complete_counts[scope, policy] = counts
            add_row(
                policy, scope, surface, "own_complete_seeds", values, counts
            )

    current_seeds = sorted(
        source.loc[source.policy_family.eq("current_frozen"), "seed"]
        .dropna().astype(int).unique()
    )
    common_cycles = {}
    for scope, surface, gain, headroom in scopes:
        for run_seed in current_seeds:
            by_policy = {}
            for policy in policy_order:
                raw_policy = policy_rows.loc[policy_rows.policy_id.eq(policy)]
                seed_rows = raw_policy.loc[
                    raw_policy._seed.eq(
                        run_seed if raw_policy.seed.notna().any() else -1
                    )
                ]
                by_policy[policy] = seed_rows.set_index("cycle_name")[[gain, headroom]]
            common = set.intersection(*[
                set(values.index[values[gain].notna()])
                for values in by_policy.values()
            ])
            for policy, values in by_policy.items():
                selected = values.loc[sorted(common)]
                add_row(
                    policy, scope, surface, f"common_seed_{run_seed}",
                    selected, selected.notna().astype(int),
                )
        common = set.intersection(*[
            set(complete_values[scope, policy].index[
                complete_values[scope, policy][gain].notna()
            ]) for policy in policy_order
        ])
        common_cycles[scope] = sorted(common)
        for policy in policy_order:
            values = complete_values[scope, policy].loc[common_cycles[scope]]
            counts = complete_counts[scope, policy].loc[common_cycles[scope]]
            add_row(
                policy, scope, surface, "common_complete_seeds", values, counts
            )
    summary = pd.DataFrame(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "frozen_policy_cop_comparison_source.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    formal = summary.loc[
        summary.evaluation_scope.eq("formal_supported_only")
        & summary.aggregation.eq("common_complete_seeds")
    ].set_index("policy_id").reindex(policy_order)
    sensitivity = summary.loc[
        summary.evaluation_scope.eq("reliable_extrapolation_sensitivity")
        & summary.aggregation.eq("common_complete_seeds")
    ].set_index("policy_id").reindex(policy_order)
    status_rows = summary.loc[
        summary.evaluation_scope.eq("formal_supported_only")
        & summary.aggregation.eq("own_complete_seeds")
    ].set_index("policy_id").reindex(policy_order)
    x = np.arange(len(policy_order))
    axes[0].bar(
        x - .18, status_rows.before_accounting_cycles_per_seed, width=.36,
        color="#B45F4D", label="Before reference accounting start",
    )
    axes[0].bar(
        x + .18, status_rows.other_unscored_cycles_per_seed, width=.36,
        color="#999999", label="Other unscored",
    )
    axes[0].set(ylabel="Mean cycles per seed", title="Unscoreable policy triggers")
    axes[0].legend(fontsize=6)

    formal_bars = axes[1].bar(
        x - .18, formal.mean_cop_gain_pct, width=.36,
        color="#3C6E8F", label="Formal supported only",
    )
    sensitivity_bars = axes[1].bar(
        x + .18, sensitivity.mean_cop_gain_pct, width=.36,
        color="#C77836", label="Reliable extrapolation sensitivity",
    )
    axes[1].axhline(0, color="black", lw=.7)
    axes[1].set(
        ylabel="COP gain vs fixed RB [%]",
        title="All-policy common complete cycles",
    )
    axes[1].bar_label(
        formal_bars, labels=[f"n={value}" for value in formal.gain_paired_cycles], fontsize=6
    )
    axes[1].bar_label(
        sensitivity_bars,
        labels=[f"n={value}" for value in sensitivity.gain_paired_cycles], fontsize=6,
    )
    axes[1].legend(fontsize=6)
    axes[1].text(
        .02, .98,
        f"Formal n={int(formal.gain_paired_cycles.min())}; sensitivity "
        f"n={int(sensitivity.gain_paired_cycles.min())}",
        transform=axes[1].transAxes, va="top", fontsize=6,
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, policy in enumerate(policy_order):
        for scope, _, gain_column, headroom_column, marker in (
            (*scopes[0], "o"), (*scopes[1], "x"),
        ):
            values = complete_values[scope, policy].loc[common_cycles[scope]]
            paired = values[gain_column].notna() & values[headroom_column].notna()
            axes[2].scatter(
                values.loc[paired, headroom_column], values.loc[paired, gain_column],
                s=20, marker=marker, alpha=.7, color=colors[index % len(colors)],
            )
    limits = axes[2].get_xlim()
    axes[2].plot(limits, limits, color=".45", ls="--", lw=.8)
    axes[2].axhline(0, color=".65", ls=":", lw=.8)
    axes[2].set(
        xlabel="Candidate-domain reference headroom vs RB [%]",
        ylabel="Policy COP gain vs RB [%]",
        title="Same-cycle headroom and realized gain",
    )
    axes[2].text(
        .02, .98,
        "Common sensitivity: "
        f"gain n={int(sensitivity.gain_paired_cycles.min())}; "
        f"gain/headroom joint n={int(sensitivity.gain_and_headroom_paired_cycles.min())}",
        transform=axes[2].transAxes, va="top", fontsize=6,
    )
    policy_handles = [
        plt.Line2D([0], [0], marker="o", lw=0, color=colors[i % len(colors)],
                   label=names[policy], markersize=5)
        for i, policy in enumerate(policy_order)
    ]
    surface_handles = [
        plt.Line2D([0], [0], marker="o", lw=0, color=".3",
                   label="Formal supported", markersize=5),
        plt.Line2D([0], [0], marker="x", lw=0, color=".3",
                   label="Reliable extrapolation sensitivity", markersize=5),
    ]
    axes[2].legend(handles=policy_handles + surface_handles, fontsize=5)
    labels = [names[policy] for policy in policy_order]
    for axis in axes[:2]:
        axis.set_xticks(x, labels, rotation=28, ha="right", fontsize=6)
    figure.suptitle("Frozen policy COP comparison against fixed RB", fontsize=10)
    figure.text(
        .5, .01,
        "COP* is the same frozen supported candidate-domain maximum in both scopes. "
        "Sensitivity allows extrapolated RB/trigger points; negative headroom is retained.",
        ha="center", fontsize=6.5, color=".3",
    )
    figure.tight_layout(rect=(0, .055, 1, 1))
    _export(
        figure, output / "frozen_policy_cop_comparison",
        formats=("png", "pdf", "svg"),
    )
    return summary
