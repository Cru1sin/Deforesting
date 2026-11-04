"""Authoritative CV cohort: historical performance screen or independent quality gates.

Default calls preserve the historical COP-based screen. Supplying measured quality
uses Gates A/B instead, retaining historical selections only as comparison columns.
An explicit terminal-decline requirement adds observation-length eligibility after
the independent quality gates; it never changes their definitions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_tools import DatasetLoader
from defrost_decision.performance_objectives import calculate_cycle_cop
from output_layout import PROJECT

NEAR_OPTIMAL_FRACTION = 0.99
MAX_NEAR_EXTRAPOLATION = 0.30
NEAR_INPUT_GAP_SECONDS = 120.1  # Four missing 30 s action points, with timestamp tolerance.

# Frozen from 30 s sampling (99th percentile 30.014 s), before model refits.
QUALITY_LIMITS = dict(
    rgb_coverage=0.95,
    dino_coverage=0.95,
    control_coverage=0.90,
    action_coverage=0.90,
    image_gap_seconds=90.1,
)


def input_quality_limits(rule, maximum_interpolated_fraction=0.30):
    if rule == "rgb70":
        return dict(
            rgb_coverage=0.70,
            dino_coverage=0.70,
            maximum_interpolated_fraction=maximum_interpolated_fraction,
        )
    if rule == "near70":
        return dict(
            rgb_coverage=0.70,
            dino_coverage=0.70,
            control_coverage=0.70,
            action_coverage=0.70,
            near_input_gap_seconds=NEAR_INPUT_GAP_SECONDS,
        )
    return dict(QUALITY_LIMITS)


def legal_quality_actions(reference):
    """A stopping action needs one complete 30 s heat-accounting window."""
    elapsed = (
        pd.to_datetime(reference.candidate_defrost_time, format="mixed")
        - pd.to_datetime(reference.heating_accounting_start, format="mixed")
    ).dt.total_seconds()
    return reference.on_action_grid.eq(True) & elapsed.ge(30)


def near_input_gaps(actions):
    """Longest missing-input run inside each exact near-set component, on a 30 s grid."""
    records = []
    for name, g in actions.groupby("cycle_name"):
        g = g.sort_values("candidate_defrost_time")
        missing = g.near_optimal & ~g.input_valid.fillna(False)
        breaks = missing.ne(
            missing.shift()
        ) | g.candidate_defrost_time.diff().dt.total_seconds().gt(30.1)
        lengths = missing.groupby(breaks.cumsum()).sum()
        records.append(
            dict(
                cycle_name=name,
                near_input_gap_s=float(lengths.max() * 30) if g.near_optimal.any() else np.nan,
            )
        )
    return pd.DataFrame(records)


def quality_gates(quality, input_rule="historical", maximum_interpolated_fraction=0.30):
    """Reward checks stay fixed; near70 restricts the input-gap check to the COP set."""
    q = quality.copy()
    checks_a = dict(
        no_actions=q.actions.gt(0),
        rgb_coverage=q.raw_rgb_coverage.ge(QUALITY_LIMITS["rgb_coverage"]),
        dino_coverage=q.dino_action_coverage.ge(QUALITY_LIMITS["dino_coverage"]),
        control_coverage=q.control_coverage.ge(QUALITY_LIMITS["control_coverage"]),
        action_coverage=q.action_coverage.ge(QUALITY_LIMITS["action_coverage"]),
        internal_image_gap=q.max_internal_image_gap_s.le(QUALITY_LIMITS["image_gap_seconds"]),
        leading_image_gap=q.leading_image_gap_s.le(QUALITY_LIMITS["image_gap_seconds"]),
        trailing_image_gap=q.trailing_image_gap_s.le(QUALITY_LIMITS["image_gap_seconds"]),
    )
    if input_rule == "near70":
        checks_a = dict(
            no_actions=q.actions.gt(0),
            rgb_coverage=q.raw_rgb_coverage.ge(0.70),
            dino_coverage=q.dino_action_coverage.ge(0.70),
            control_coverage=q.control_coverage.ge(0.70),
            action_coverage=q.action_coverage.ge(0.70),
            near_input_gap=q.near_input_gap_s.le(NEAR_INPUT_GAP_SECONDS),
        )
    elif input_rule == "rgb70":
        checks_a = dict(
            no_actions=q.actions.gt(0),
            rgb_coverage=q.raw_rgb_coverage.ge(0.70),
            dino_coverage=q.dino_action_coverage.ge(0.70),
            interpolation=q.interpolated_fraction.le(maximum_interpolated_fraction),
        )
    elif input_rule != "historical":
        raise ValueError("Unknown input rule")
    checks_b = dict(
        heat_integral=q.heat_coverage.eq(1),
        electricity_integral=q.electricity_coverage.eq(1),
        ridge_prediction=q.ridge_prediction_coverage.eq(1),
        complete_grid_reward=q.cop_candidate_coverage.eq(1),
        actual_endpoint=q.actual_endpoint.eq(True),
        terminal_reward=q.endpoint_reward_valid.eq(True),
        rb_reward=q.rb_reward_valid.eq(True),
    )
    for gate, checks in [("a", checks_a), ("b", checks_b)]:
        passed = pd.DataFrame(checks, index=q.index).fillna(False)
        q["gate_" + gate] = passed.all(axis=1)
        q["gate_" + gate + "_reasons"] = passed.apply(lambda row: ";".join(row.index[~row]), axis=1)
    q["selected"] = q.gate_a & q.gate_b
    return q


def screen_cv_cycles(
    decisions: pd.DataFrame,
    cycles: pd.DataFrame,
    *,
    quality=None,
    exclude_rb=True,
    minimum_terminal_decline=None,
    rb_minimum_terminal_decline=None,
    input_rule="historical",
    action_inputs=None,
    cohort_rule="reward_complete",
    excluded_cycles=(),
    maximum_per_setpoint=None,
    post_near_minutes_by_setpoint=None,
    maximum_interpolated_fraction=0.30,
) -> pd.DataFrame:
    """Return one eligibility record per cycle present in the supplied decisions.

    `cycles` is DatasetLoader.list_cycles(), with no prior quality/RGB filtering.
    All saved candidate points, including off-grid exact endpoints, have equal
    weight. The full 1% set includes every disconnected component and valid Ridge
    extrapolation. Its extrapolated-point fraction is diagnostic only; it never gates admission.
    The exact preparation-start COP must be < 0.99 * maximum COP. If preparation
    is absent, the catalog recording end is used and explicitly marked.
    Missing endpoint/reference values are unknown, never a passing condition.
    Input frames and historical COP selections are not modified.
    In label cohorts, per-setpoint post-near durations replace the decline gate:
    elapsed observation end minus first supported near point, including gaps.
    The observed cohort uses RGB/DINO >=70%, full-cycle interpolation below the configured limit,
    and a valid exact terminal below 99% of peak for every setpoint and RB role.
    Its training and held-out replay membership are identical.
    """
    if not 0 <= maximum_interpolated_fraction <= 1:
        raise ValueError("Interpolation fraction must be in [0, 1]")
    if decisions.empty or not decisions.effective_heat_rule.eq("without_trunc").all():
        raise ValueError("Provide nonempty without_trunc candidate decisions")
    if minimum_terminal_decline is not None and not 0 <= minimum_terminal_decline < 1:
        raise ValueError("minimum_terminal_decline must be in [0, 1)")
    if cohort_rule not in ("reward_complete", "labels", "observed"):
        raise ValueError("Unknown cohort rule")
    post_near_minutes_by_setpoint = {
        float(k): float(v) for k, v in (post_near_minutes_by_setpoint or {}).items()
    }
    if any(v < 0 for v in post_near_minutes_by_setpoint.values()):
        raise ValueError("Post-near observation minutes must be nonnegative")
    if cohort_rule == "labels" and (quality is None or minimum_terminal_decline is None):
        raise ValueError(
            "Label cohort requires input quality and a default decline threshold; per-setpoint durations override it"
        )
    if cohort_rule == "observed" and (quality is None or input_rule != "rgb70"):
        raise ValueError("Observed cohort requires RGB/DINO and interpolation quality")
    values = calculate_cycle_cop(decisions, effective=True)
    values["candidate_defrost_time"] = pd.to_datetime(values.candidate_defrost_time, format="mixed")
    catalog = cycles.set_index("cycle_name")
    records = []
    for name, curve in values.groupby("cycle_name", sort=True):
        record = catalog.loc[name]
        valid = curve.cycle_cop_eligible & np.isfinite(curve.cycle_cop)
        peak = curve.loc[valid, "cycle_cop"].max()
        near = valid & curve.cycle_cop.ge(NEAR_OPTIMAL_FRACTION * peak)
        near_points = int(near.sum())
        near_start = curve.loc[near, "candidate_defrost_time"].min()
        near_end = curve.loc[near, "candidate_defrost_time"].max()
        near_span_minutes = (near_end - near_start).total_seconds() / 60 if near_points else np.nan
        extrapolated = int(
            (near & ~curve.defrost_event_electricity_in_training_domain.eq(True)).sum()
        )
        fraction = extrapolated / near_points if near_points else np.nan
        low_extrapolation = bool(fraction <= MAX_NEAR_EXTRAPOLATION) if near_points else None

        end = pd.to_datetime(record.defrost_preparation_start)
        end_source = "actual_preparation" if pd.notna(end) else "recording_end"
        if pd.isna(end):
            end = pd.to_datetime(record.end_time)
        observed = curve.loc[valid & curve.candidate_defrost_time.le(end)].sort_values(
            "candidate_defrost_time"
        )
        observed_time = observed.candidate_defrost_time.iloc[-1] if len(observed) else pd.NaT
        observed_cop = observed.cycle_cop.iloc[-1] if len(observed) else np.nan
        endpoint = curve.loc[curve.candidate_defrost_time.eq(end)]
        end_cop = endpoint.cycle_cop.iloc[0] if len(endpoint) else np.nan
        end_valid = bool(
            len(endpoint) and endpoint.cycle_cop_eligible.iloc[0] and np.isfinite(end_cop)
        )
        long_enough = (
            bool(end_cop < NEAR_OPTIMAL_FRACTION * peak) if end_valid and near_points else None
        )
        reasons = []
        if not near_points:
            reasons.append("no_valid_cop_reference")
        if endpoint.empty:
            reasons.append("exact_end_point_absent")
        elif not end_valid:
            reasons.append("exact_end_cop_invalid")
        if long_enough is False:
            reasons.append("end_in_1pct_near_set")
        records.append(
            dict(
                cycle_name=name,
                experiment_id=record.experiment_id,
                effective_heat_rule="without_trunc",
                end_source=end_source,
                end_time=end,
                end_cop=end_cop,
                maximum_cop_including_extrapolation=peak,
                end_over_max=end_cop / peak if end_valid and near_points else np.nan,
                observed_terminal_time=observed_time,
                observed_terminal_cop=observed_cop,
                observed_terminal_over_max=observed_cop / peak
                if near_points and peak > 0
                else np.nan,
                near_points=near_points,
                near_start=near_start,
                near_end=near_end,
                near_span_minutes=near_span_minutes,
                setpoint_c=float(curve.water_temperature_setpoint.median())
                if "water_temperature_setpoint" in curve
                and curve.water_temperature_setpoint.notna().any()
                else np.nan,
                post_near_minutes=(end - near_start).total_seconds() / 60
                if near_points
                else np.nan,
                near_ridge_extrapolation_points=extrapolated,
                near_extrapolation_fraction=fraction,
                long_enough=long_enough,
                low_extrapolation=low_extrapolation,
                selected=long_enough is True,
                exclusion_reasons=";".join(reasons),
            )
        )
    result = pd.DataFrame(records)
    result["baseline_cycle"] = result.cycle_name.map(
        lambda name: str(catalog.loc[name].get("assets", {}).get("parquet", "")).startswith(
            "cycles_RB/"
        )
    )
    result[["long_enough", "low_extrapolation"]] = result[
        ["long_enough", "low_extrapolation"]
    ].astype("boolean")
    if quality is not None:
        if set(quality.cycle_name) != set(result.cycle_name):
            raise ValueError("Quality audit must cover every candidate cycle before admission")
        if input_rule == "near70":
            if action_inputs is None:
                raise ValueError("near70 requires saved action inputs")
            if not isinstance(action_inputs, pd.DataFrame):
                action_inputs = pd.read_parquet(
                    action_inputs, columns=["cycle_name", "candidate_defrost_time", "input_valid"]
                )
            keys = ["cycle_name", "candidate_defrost_time"]
            grid = values.loc[legal_quality_actions(values)].merge(
                result[["cycle_name", "end_time", "maximum_cop_including_extrapolation"]],
                on="cycle_name",
            )
            grid = grid.loc[grid.candidate_defrost_time.lt(grid.end_time)].copy()
            grid["near_optimal"] = grid.cycle_cop_eligible & grid.cycle_cop.ge(
                0.99 * grid.maximum_cop_including_extrapolation
            )
            grid = grid.merge(
                action_inputs[keys + ["input_valid"]], on=keys, how="left", validate="one_to_one"
            )
            quality = quality.merge(
                near_input_gaps(grid), on="cycle_name", how="left", validate="one_to_one"
            )
        result = result.rename(
            columns={"selected": "legacy_selected", "exclusion_reasons": "legacy_exclusion_reasons"}
        )
        result = result.merge(
            quality_gates(quality, input_rule, maximum_interpolated_fraction).drop(
                columns="experiment_id"
            ),
            on="cycle_name",
            validate="one_to_one",
        )
        result["exclusion_reasons"] = result.gate_a_reasons + "|" + result.gate_b_reasons
        # Experiment role is independent of acquisition quality and reward support.
        if exclude_rb:
            result["selected"] &= ~result.baseline_cycle
            result.loc[result.baseline_cycle, "exclusion_reasons"] += "|baseline_cycle"
    if minimum_terminal_decline is not None:
        rb_decline = (
            minimum_terminal_decline
            if rb_minimum_terminal_decline is None
            else rb_minimum_terminal_decline
        )
        if not 0 <= rb_decline < 1:
            raise ValueError("RB terminal decline must be in [0, 1)")
        result["required_terminal_decline"] = np.where(
            result.baseline_cycle, rb_decline, minimum_terminal_decline
        )
        # Strictly outside the requested near-optimal range, not at its boundary.
        result["terminal_decline_pct"] = 100 * (1 - result.end_over_max)
        result["terminal_decline_pass"] = (
            result.end_source.eq("actual_preparation")
            & result.end_over_max.notna()
            & result.end_cop.lt(
                (1 - result.required_terminal_decline) * result.maximum_cop_including_extrapolation
            )
        )
        result["selected"] &= result.terminal_decline_pass
        result.loc[~result.terminal_decline_pass, "exclusion_reasons"] += (
            "|terminal_decline_insufficient_or_unknown"
        )
    if cohort_rule == "labels":
        result["label_valid"] = result.near_points.gt(0)
        result["observed_terminal_decline_pass"] = result.observed_terminal_cop.lt(
            (1 - result.required_terminal_decline) * result.maximum_cop_including_extrapolation
        )
        result["required_post_near_minutes"] = result.setpoint_c.map(post_near_minutes_by_setpoint)
        uses_duration = result.required_post_near_minutes.notna()
        result["length_pass"] = result.observed_terminal_decline_pass.where(
            ~uses_duration, result.post_near_minutes.ge(result.required_post_near_minutes)
        )
        result["length_rule"] = np.where(
            uses_duration, "near_start_to_observation_end", "observed_terminal_decline"
        )
        length_pass = result.length_pass
        result["selected_train"] = result.gate_a & result.label_valid & length_pass
        if exclude_rb:
            result["selected_train"] &= ~result.baseline_cycle
        result["selected_evaluation"] = result.selected_train & result.gate_b
        result["selected"] = result.selected_train
        result["exclusion_reasons"] = result.gate_a_reasons
        result.loc[~result.label_valid, "exclusion_reasons"] += "|near_label_missing"
        result.loc[~length_pass & uses_duration, "exclusion_reasons"] += (
            "|post_near_observation_too_short_or_unknown"
        )
        result.loc[~length_pass & ~uses_duration, "exclusion_reasons"] += (
            "|observed_terminal_decline_insufficient_or_unknown"
        )
        if exclude_rb:
            result.loc[result.baseline_cycle, "exclusion_reasons"] += "|baseline_cycle"
    if cohort_rule == "observed":
        # One cohort for fitting and held-out replay. Reward and label diagnostics do not gate it.
        result["selected_train"] = result.gate_a & result.long_enough.fillna(False)
        if exclude_rb:
            result["selected_train"] &= ~result.baseline_cycle
        result["selected_evaluation"] = result.selected_train
        result["selected"] = result.selected_train
        result["exclusion_reasons"] = result.gate_a_reasons
        result.loc[~result.long_enough.fillna(False), "exclusion_reasons"] += (
            "|terminal_not_below_99pct_or_unknown"
        )
        if exclude_rb:
            result.loc[result.baseline_cycle, "exclusion_reasons"] += "|baseline_cycle"
    if excluded_cycles:
        result["manual_excluded"] = result.cycle_name.isin(excluded_cycles)
        for column in ["selected", "selected_train", "selected_evaluation"]:
            if column in result:
                result.loc[result.manual_excluded, column] = False
        result.loc[result.manual_excluded, "exclusion_reasons"] += "|manual_cycle_exclusion"
    if maximum_per_setpoint is not None:
        if maximum_per_setpoint < 1 or quality is None or cohort_rule != "labels":
            raise ValueError(
                "Balanced admission requires a positive cap and measured input/label quality"
            )
        result["setpoint_c"] = result.cycle_name.map(
            values.groupby("cycle_name").water_temperature_setpoint.median()
        )
        result["admission_candidate"] = result.selected
        result["strict_decline_supported"] = result.observed_terminal_over_max.lt(0.99)
        result["input_quality_min"] = result[
            ["raw_rgb_coverage", "dino_action_coverage", "control_coverage", "action_coverage"]
        ].min(axis=1)
        selected = []
        for _, group in result.loc[result.selected].groupby("setpoint_c"):
            strict = group.loc[group.strict_decline_supported].sort_values("cycle_name")
            if len(strict) > maximum_per_setpoint:
                raise ValueError("Cap would remove strict-admission cycles; increase it")
            selected.extend(strict.cycle_name)
            remaining = maximum_per_setpoint - len(strict)
            extra = group.loc[~group.strict_decline_supported].sort_values(
                ["input_quality_min", "near_input_gap_s", "cycle_name"],
                ascending=[False, True, True],
            )
            # Spread additions across experiments; prioritize underrepresented days.
            counts = strict.experiment_id.value_counts().to_dict()
            while remaining and len(extra):
                first = extra.groupby("experiment_id", sort=False).head(1).copy()
                first["already_selected"] = first.experiment_id.map(counts).fillna(0)
                choice = first.sort_values(
                    ["already_selected", "input_quality_min", "near_input_gap_s", "cycle_name"],
                    ascending=[True, False, True, True],
                ).iloc[0]
                selected.append(choice.cycle_name)
                counts[choice.experiment_id] = counts.get(choice.experiment_id, 0) + 1
                extra = extra.loc[extra.cycle_name.ne(choice.cycle_name)]
                remaining -= 1
        retained = result.cycle_name.isin(selected)
        result.loc[result.selected & ~retained, "exclusion_reasons"] += "|balanced_cohort_cap"
        for column in ["selected", "selected_train", "selected_evaluation"]:
            result[column] &= retained
    return result


def rb_failure_cases(decisions, screening, *, early_review_rule="all_cycles"):
    """Frozen RB triggers outside the full supported 1% near envelope.

    all_cycles admits early review candidates with valid COP at the exact trigger,
    without training quality/domain gates. strict preserves the original review.
    Late review stays strict. No unobserved continuation is inferred.
    """
    if early_review_rule not in ("strict", "all_cycles"):
        raise ValueError("Unknown early review rule")
    label_valid = screening.near_points.gt(0)
    strict = (
        screening.baseline_cycle
        & screening.gate_a
        & screening.gate_b
        & label_valid.fillna(False)
        & ~screening.get("manual_excluded", False)
    )
    evidence = screening.assign(strict_review_eligible=strict).set_index("cycle_name")
    values = calculate_cycle_cop(decisions, effective=True)
    values["candidate_defrost_time"] = pd.to_datetime(values.candidate_defrost_time, format="mixed")
    records = []
    for name, curve in values.groupby("cycle_name"):
        record = evidence.loc[name]
        if record.get("manual_excluded", False):
            continue
        trigger = pd.to_datetime(curve.t_RB.iloc[0])
        if curve.rb_status.iloc[0] != "triggered" or pd.isna(trigger):
            continue
        valid = curve.cycle_cop_eligible & np.isfinite(curve.cycle_cop)
        peak = curve.loc[valid, "cycle_cop"].max()
        near = pd.to_datetime(
            curve.loc[
                valid & curve.cycle_cop.ge(NEAR_OPTIMAL_FRACTION * peak), "candidate_defrost_time"
            ],
            format="mixed",
        )
        if near.empty:
            continue
        kind = "早除霜" if trigger < near.min() else "迟除霜" if trigger > near.max() else None
        at_trigger = curve.loc[curve.candidate_defrost_time.eq(trigger) & valid]
        strict_pass = bool(record.strict_review_eligible)
        relaxed_pass = (
            early_review_rule == "all_cycles" and kind == "早除霜" and not at_trigger.empty
        )
        if not (strict_pass or relaxed_pass):
            continue
        if kind:
            records.append(
                dict(
                    cycle_name=name,
                    failure_type=kind,
                    t_RB=trigger,
                    near_start=near.min(),
                    near_end=near.max(),
                    review_rule=early_review_rule if kind == "早除霜" else "strict",
                    strict_review_eligible=strict_pass,
                    rb_trigger_source="rb_acquired_cycle"
                    if record.baseline_cycle
                    else "replayed_rb_in_experiment",
                    trigger_over_max=at_trigger.cycle_cop.iloc[0] / peak
                    if len(at_trigger)
                    else np.nan,
                    outside_near_seconds=(near.min() - trigger).total_seconds()
                    if kind == "早除霜"
                    else (trigger - near.max()).total_seconds(),
                )
            )
    return screening.merge(
        pd.DataFrame(
            records,
            columns=[
                "cycle_name",
                "failure_type",
                "t_RB",
                "near_start",
                "near_end",
                "review_rule",
                "strict_review_eligible",
                "rb_trigger_source",
                "trigger_over_max",
                "outside_near_seconds",
            ],
        ),
        on="cycle_name",
        validate="one_to_one",
    )


def screen_admission(settings_path, *, dataset=None, camera_role="front", definition=None):
    """Recompute saved admission rules on full candidates and the current Dataset catalog."""
    settings_path = Path(settings_path)
    if dataset is None and settings_path.parent.name == DatasetLoader.training_directory:
        dataset = settings_path.parent.parent
        camera_role = (
            settings_path.stem.removeprefix("settings__") if "__" in settings_path.stem else "front"
        )
    admission = definition if definition is not None else json.loads(settings_path.read_text())
    if dataset is not None:
        loader = DatasetLoader(dataset)
        decisions, quality, inputs = loader.load_screening_inputs(camera_role)
    else:
        loader = DatasetLoader(Path(admission["dataset"]))
        decisions_path = Path(admission["decisions"])
        decisions = (
            pd.read_parquet(decisions_path)
            if decisions_path.suffix == ".parquet"
            else pd.read_csv(decisions_path, low_memory=False)
        )
        quality = pd.read_csv(admission["quality"])
        inputs = Path(admission["quality"]).parent / "decision_inputs.parquet"
    options = {
        key: admission[key]
        for key in (
            "exclude_rb",
            "minimum_terminal_decline",
            "rb_minimum_terminal_decline",
            "input_rule",
            "cohort_rule",
            "excluded_cycles",
            "maximum_per_setpoint",
            "post_near_minutes_by_setpoint",
        )
        if key in admission
    }
    options["maximum_interpolated_fraction"] = admission.get("limits", {}).get(
        "maximum_interpolated_fraction", 0.30
    )
    if admission["input_rule"] == "near70":
        options["action_inputs"] = inputs
    return screen_cv_cycles(
        decisions,
        loader.list_cycles(),
        quality=quality,
        **options,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    parser.add_argument(
        "--decisions",
        type=Path,
        required=True,
        help="Full saved without_trunc candidate decisions (CSV or parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(PROJECT / "output/90_复现材料/02_循环筛选/统一队列规则与名单"),
    )
    parser.add_argument("--audit-inputs", action="store_true")
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Render admitted cycles with the shared effective COP renderer",
    )
    parser.add_argument(
        "--dataset-figures",
        action="store_true",
        help="Publish paired training and reviewable RB failure figures in Dataset",
    )
    parser.add_argument(
        "--redraw-figures",
        action="store_true",
        help="Refresh published COP figures; retain existing cycle overviews",
    )
    parser.add_argument(
        "--rb-early-review-rule", choices=["strict", "all_cycles"], default="all_cycles"
    )
    parser.add_argument(
        "--exclude-cycles",
        nargs="+",
        default=[],
        help="Explicitly exclude named cycles without changing their data",
    )
    parser.add_argument(
        "--near-set-rule",
        choices=["exact", "connected"],
        default="exact",
        help="exact: 离散近优点（默认）；connected: 桥接首末近优点间空隙的历史标签规则；绘图不改变此选择",
    )
    parser.add_argument("--feature-caches", type=Path, nargs="+")
    parser.add_argument("--legacy-run", type=Path)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument(
        "--quality",
        type=Path,
        help="Audited quality_metrics.csv; independent Gates A/B replace legacy admission",
    )
    parser.add_argument(
        "--include-rb", action="store_true", help="Apply the same eligibility criteria to RB cycles"
    )
    parser.add_argument(
        "--minimum-terminal-decline",
        type=float,
        help="Additional strict decline from peak at actual preparation, e.g. 0.05",
    )
    parser.add_argument("--rb-minimum-terminal-decline", type=float)
    parser.add_argument("--maximum-interpolated-fraction", type=float, default=0.30)
    parser.add_argument(
        "--input-rule", choices=["historical", "near70", "rgb70"], default="historical"
    )
    parser.add_argument(
        "--cohort-rule",
        choices=["reward_complete", "labels", "observed"],
        default="reward_complete",
    )
    parser.add_argument(
        "--maximum-per-setpoint",
        type=int,
        help="Retain strict cycles, then balance additions across experiments up to this cap",
    )
    parser.add_argument(
        "--post-near-minutes-by-setpoint",
        type=json.loads,
        help='JSON map, e.g. {"50":5,"60":18}; observation-end minus first near point replaces decline at these setpoints',
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.near_set_rule == "connected" and not args.quality:
        parser.error("--near-set-rule connected requires --quality to rebuild labels")
    from output_layout import prepare_output

    prepare_output(args, training=False)
    decisions = (
        pd.read_parquet(args.decisions)
        if args.decisions.suffix == ".parquet"
        else pd.read_csv(args.decisions, low_memory=False)
    )
    loader = DatasetLoader(args.dataset)
    catalog = loader.list_cycles()
    screening = screen_cv_cycles(
        decisions,
        catalog,
        quality=pd.read_csv(args.quality) if args.quality else None,
        exclude_rb=not args.include_rb,
        minimum_terminal_decline=args.minimum_terminal_decline,
        rb_minimum_terminal_decline=args.rb_minimum_terminal_decline,
        input_rule=args.input_rule,
        maximum_per_setpoint=args.maximum_per_setpoint,
        post_near_minutes_by_setpoint=args.post_near_minutes_by_setpoint,
        cohort_rule=args.cohort_rule,
        maximum_interpolated_fraction=args.maximum_interpolated_fraction,
        excluded_cycles=args.exclude_cycles,
        action_inputs=args.quality.parent / "decision_inputs.parquet" if args.quality else None,
    )
    if args.dataset_figures:
        from dataset_tools.training_data import publish_cycle_figures

        publish_cycle_figures(
            args.dataset,
            decisions,
            screening,
            n_jobs=args.n_jobs,
            near_set_rule=args.near_set_rule,
            early_review_rule=args.rb_early_review_rule,
            redraw=args.redraw_figures,
        )
    screening["setpoint_c"] = screening.cycle_name.map(
        decisions.groupby("cycle_name").water_temperature_setpoint.median()
    )
    screening["source"] = str(args.decisions.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    screening.to_csv(args.output / "cycle_screening.csv", index=False)
    if args.audit_inputs:
        from image_models.admission import audit_inputs

        for key in ["candidate_defrost_time", "t_RB", "stable_heating_start"]:
            decisions[key] = pd.to_datetime(decisions[key], format="mixed")
        reference = calculate_cycle_cop(decisions, effective=True)
        audit_inputs(
            reference,
            catalog,
            args.dataset,
            args.feature_caches or [args.dataset / "image_DINO_feature"],
            args.output,
            args.n_jobs,
            args.legacy_run,
        )
        return
    screening.loc[screening.selected].to_csv(args.output / "cohort.csv", index=False)
    if args.cohort_rule == "labels":
        screening.loc[screening.selected_evaluation].to_csv(
            args.output / "evaluation_cohort.csv", index=False
        )
    if args.near_set_rule == "connected":
        from image_models.labels import label_actions

        names = set(screening.loc[screening.selected, "cycle_name"])
        reference = calculate_cycle_cop(
            decisions.loc[decisions.cycle_name.isin(names)], effective=True
        )
        reference["candidate_defrost_time"] = pd.to_datetime(
            reference.candidate_defrost_time, format="mixed"
        )
        ends = screening.set_index("cycle_name").end_time
        reference["on_action_grid"] = legal_quality_actions(
            reference
        ) & reference.candidate_defrost_time.lt(reference.cycle_name.map(ends))
        labels, intervals = label_actions(
            reference, label_rule="stopping_latch", near_set_rule="connected"
        )
        inputs = pd.read_parquet(args.quality.parent / "decision_inputs.parquet")
        keys = ["cycle_name", "candidate_defrost_time"]
        # Input validity remains independent of the inferred connected labels.
        overlap = [name for name in inputs if name in labels and name not in keys]
        labels = labels.merge(
            inputs.drop(columns=overlap), on=keys, how="left", validate="one_to_one"
        )
        labels["input_valid"] = labels.input_valid.fillna(False)
        labels["train_mask"] &= labels.input_valid
        labels.to_parquet(args.output / "decision_inputs.parquet", index=False)
        reference.to_parquet(args.output / "reference.parquet", index=False)
        intervals.to_csv(args.output / "near_optimal_intervals.csv", index=False)
    if args.figures or (args.output / "candidate_decisions.csv").is_file():
        names = set(screening.loc[screening.selected, "cycle_name"])
        plotted = decisions.loc[decisions.cycle_name.isin(names)].copy()
        for column in plotted:
            if column.endswith(("_time", "_start", "_end")) or column in {"t_star", "t_RB"}:
                plotted[column] = pd.to_datetime(plotted[column], format="mixed")
        plotted.to_csv(args.output / "candidate_decisions.csv", index=False)
    for name in args.exclude_cycles:
        if name in set(screening.cycle_name):
            (args.output / f"cycle_{int(name.rsplit('_', 1)[-1]):03d}.png").unlink(missing_ok=True)
    if args.figures:
        from joblib import Parallel, delayed, parallel_config

        from plots.defrost_decision import render_current_decision_cycle

        with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
            Parallel(verbose=5)(
                delayed(render_current_decision_cycle)(
                    curve,
                    loader,
                    args.output / f"cycle_{int(name.rsplit('_', 1)[-1]):03d}.png",
                    near_set_rule=args.near_set_rule,
                )
                for name, curve in plotted.groupby("cycle_name", sort=True)
            )
    if args.quality:
        (args.output / "admission_settings.json").write_text(
            json.dumps(
                dict(
                    limits=input_quality_limits(
                        args.input_rule, args.maximum_interpolated_fraction
                    ),
                    input_rule=args.input_rule,
                    exclude_rb=not args.include_rb,
                    minimum_terminal_decline=args.minimum_terminal_decline,
                    rb_minimum_terminal_decline=args.rb_minimum_terminal_decline,
                    maximum_per_setpoint=args.maximum_per_setpoint,
                    post_near_minutes_by_setpoint=args.post_near_minutes_by_setpoint,
                    cohort_rule=args.cohort_rule,
                    excluded_cycles=args.exclude_cycles,
                    near_set_rule=args.near_set_rule,
                    near_set_definition=(
                        "连片近优: connect first-to-last legal 1% near point; bridged labels are inferred, COP validity unchanged"
                        if args.near_set_rule == "connected"
                        else "exact supported 1% near set"
                    ),
                    action_start="accounting start + 30 s; existing grid; before preparation or recording end when preparation is absent",
                    selection=(
                        f"RGB/DINO >=70%; full-cycle is_interpolated <={100 * args.maximum_interpolated_fraction:g}%; exact terminal COP <99% peak; same cohort for held-out replay"
                        if args.cohort_rule == "observed"
                        else "train: input + near-label support + last valid observed COP decline; evaluation: train + full reward gates"
                        if args.cohort_rule == "labels"
                        else "input + full reward gates + requested terminal decline"
                    ),
                    input="front RGB -> frozen DINO -> backward <=30s -> unchanged frequency features; no Harmony/reward mask",
                    reward=(
                        "diagnostic only; actual selected COP and RB support determine metric coverage"
                        if args.cohort_rule == "observed"
                        else "all legal grid heat/electricity/COP valid; actual endpoint and RB valid; model extrapolation is annotation only"
                    ),
                    decisions=str(args.decisions.resolve()),
                    dataset=str(args.dataset.resolve()),
                    quality=str(args.quality.resolve()),
                ),
                indent=2,
            )
        )
        print(screening.groupby(["gate_a", "gate_b"]).size().to_string())
        print(
            "Admitted:",
            int(screening.selected.sum()),
            "cycles;",
            screening.loc[screening.selected, "experiment_id"].nunique(),
            "experiments",
        )
        condition_counts = (
            screening.assign(quality_pass=screening.gate_a & screening.gate_b)
            .groupby(["setpoint_c", "baseline_cycle"], dropna=False)
            .agg(
                total=("cycle_name", "size"),
                quality_pass=("quality_pass", "sum"),
                selected=("selected", "sum"),
            )
        )
        if args.minimum_terminal_decline is not None:
            condition_counts["terminal_decline_pass"] = screening.groupby(
                ["setpoint_c", "baseline_cycle"], dropna=False
            ).terminal_decline_pass.sum()
        if args.cohort_rule == "labels":
            condition_counts = screening.groupby(
                ["setpoint_c", "baseline_cycle"], dropna=False
            ).agg(
                total=("cycle_name", "size"),
                input_pass=("gate_a", "sum"),
                label_pass=("label_valid", "sum"),
                observed_decline_pass=("observed_terminal_decline_pass", "sum"),
                train=("selected_train", "sum"),
                evaluation=("selected_evaluation", "sum"),
            )
        elif args.cohort_rule == "observed":
            condition_counts = screening.groupby(
                ["setpoint_c", "baseline_cycle"], dropna=False
            ).agg(
                total=("cycle_name", "size"),
                input_pass=("gate_a", "sum"),
                terminal_decline_pass=("long_enough", "sum"),
                selected=("selected", "sum"),
            )
        condition_counts.to_csv(args.output / "condition_counts.csv")
        if args.cohort_rule == "observed":
            print(condition_counts.to_string())
            return
        screening.loc[~screening.gate_b].to_csv(args.output / "reward_failures.csv", index=False)
        failures = (
            screening.loc[~screening.gate_b, ["cycle_name", "gate_b_reasons"]]
            .assign(reason=lambda x: x.gate_b_reasons.str.split(";"))
            .explode("reason")
        )
        failures.groupby("reason").agg(
            count=("cycle_name", "size"),
            cycles=("cycle_name", lambda x: ",".join(str(int(n.rsplit("_", 1)[1])) for n in x)),
        ).to_csv(args.output / "reward_failure_reasons.csv")
        print(condition_counts.to_string())
        return
    counts = []
    for rule in ["long_enough"]:
        column = screening[rule]
        counts.append(
            dict(
                rule=rule,
                total=len(screening),
                passed=int(column.eq(True).sum()),
                failed=int(column.eq(False).sum()),
                unknown=int(column.isna().sum()),
            )
        )
    combined = screening.long_enough
    counts.append(
        dict(
            rule="selected",
            total=len(screening),
            passed=int(combined.eq(True).sum()),
            failed=int(combined.eq(False).sum()),
            unknown=int(combined.isna().sum()),
        )
    )
    counts = pd.DataFrame(counts)
    counts.to_csv(args.output / "counts.csv", index=False)
    print(counts.to_string(index=False))
    print(
        "Selected endpoint sources:",
        screening.loc[screening.selected, "end_source"].value_counts().to_dict(),
    )


def screen_published_cycles(dataset, camera_role="front", *, definition=None):
    """Reapply the authoritative admission before any new analysis subsets."""
    from dataset_tools import DatasetLoader

    loader = DatasetLoader(dataset)
    training = loader.dataset_root / loader.training_directory
    suffix = "" if camera_role == "front" else f"__{camera_role}"
    return screen_admission(
        training / f"settings{suffix}.json",
        dataset=loader.dataset_root,
        camera_role=camera_role,
        definition=definition,
    )


if __name__ == "__main__":
    main()
