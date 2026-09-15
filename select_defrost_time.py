"""Evaluate refrigerant effective-heating COP and select its supported maximum."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from dataset_tools import DatasetLoader
from dataset_tools.builder.detect_cycles import add_recovery_arguments, recovery_settings
from defrost_decision.baselines import rule_based
from defrost_decision.candidate_quantities import current_effective_cop, effective_candidate_cop
from defrost_decision.performance_objectives import add_single_objective_optima
from defrost_event_models.ridge_models import load_defrost_event_models
from defrost_event_models.training_data import timestamp


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--cycles", nargs="*")
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--run-name", default="effective_cop")
    parser.add_argument("--candidate-step-seconds", type=int, default=10)
    parser.add_argument(
        "--prediction-mode", choices=("cross-fitted", "full-model"), default="cross-fitted"
    )
    parser.add_argument("--preparation-heat", choices=("include", "zero"), default="zero")
    parser.add_argument("--n-jobs", "--workers", dest="n_jobs", type=int, default=6)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--fetch-cloud-images", action="store_true")
    parser.add_argument(
        "--figures-only", action="store_true", help="render saved candidate results"
    )
    parser.add_argument("--publish-dataset", action="store_true",
                        help="publish this named heat convention as a Dataset decision asset")
    parser.add_argument("--compare-run", type=Path, help="other preparation-heat decision run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--current-csv", type=Path, help="raw observations of one new heating-cycle prefix"
    )
    parser.add_argument("--experiment-id", default="new_experiment")
    add_recovery_arguments(parser)
    return parser


def calculate_cycle(
    loader,
    cycle_name,
    models,
    *,
    candidate_step_seconds=10,
    prediction_mode="cross-fitted",
    preparation_heat="zero",
):
    record = loader.get_cycle_record(cycle_name)
    bounds = record["boundaries"]
    heating = timestamp(bounds.get("heating_start"))
    recovery = timestamp(bounds.get("stable_heating_start"))
    end = timestamp(bounds.get("defrost_preparation_start"))
    frame = loader.load_cycle_original(cycle_name)
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    observation_end = min(
        timestamp(bounds.get("end_time")) or frame.timestamp.max(), frame.timestamp.max()
    )
    open_heating = end is None and timestamp(bounds.get("defrost_start")) is None
    if open_heating:
        end = observation_end
    rb = rule_based.calculate_cycle(loader, cycle_name)
    reason = (
        str(record.get("recovery_status", "recovery_not_identified"))
        if recovery is None
        else "missing_preparation_boundary"
        if end is None
        else "catalog_" + str(record["status"])
        if record["status"] != "valid" and not (open_heating and record["status"] == "partial")
        else ""
    )
    if not reason:
        first = max(recovery + pd.Timedelta(seconds=1), heating + pd.Timedelta(minutes=5))
        times = list(pd.date_range(first, end, freq=pd.Timedelta(seconds=candidate_step_seconds)))
        if end >= first:
            times.append(end)
        rb_time = timestamp(rb.get("t_RB"))
        if rb_time is not None and first <= rb_time <= end:
            times.append(rb_time)
        times = sorted(set(times))
        if not times:
            reason = "insufficient_heating_history"
    if reason:
        result = pd.DataFrame(
            {
                "candidate_defrost_time": [end or frame.timestamp.max()],
                "cycle_cop": [np.nan],
                "cycle_cop_eligible": [False],
            }
        )
    else:
        result = effective_candidate_cop(
            frame,
            times,
            recovery,
            heating,
            models,
            str(record["experiment_id"]),
            preparation_heat=preparation_heat,
            prediction_mode=prediction_mode,
        )
    result = add_single_objective_optima(result, names=("cycle_cop",))
    result["is_selected"] = result.candidate_defrost_time.eq(result.cycle_cop_t_star)
    result["selected_point_in_training_domain"] = result.is_selected & result.cycle_cop_eligible
    result["t_star"] = result.cycle_cop_t_star
    result["t_star_model_supported"] = result.is_selected.any()
    result["cycle_status"] = (
        "identified_curve" if result.is_selected.any() else reason or "no_supported_candidate"
    )
    result["cycle_name"] = cycle_name
    result["experiment_id"] = record["experiment_id"]
    result["cycle_start"] = (
        heating
        if bounds.get("heating_origin") == "cold_start"
        else timestamp(bounds.get("start_time")) or heating
    )
    result["observation_end"] = observation_end
    result["candidate_end_source"] = "recording_end" if open_heating else "actual_preparation"
    result["catalog_status"] = record["status"]
    result["stable_heating_start"] = recovery
    result["t_RB"] = rb.get("t_RB")
    result["rb_status"] = rb["rb_status"]
    result["algorithm"] = "effective_cop"
    result["preparation_heat"] = preparation_heat
    return result


def summarize_decisions(decisions):
    rows = []
    for name, curve in decisions.groupby("cycle_name", sort=True):
        first = curve.iloc[0]
        selected = curve.loc[curve.is_selected]
        at_rb = curve.loc[
            curve.candidate_defrost_time.eq(pd.to_datetime(first.t_RB)) & curve.cycle_cop_eligible
        ]
        best = selected.cycle_cop.iloc[0] if len(selected) else np.nan
        rb_cop = at_rb.cycle_cop.iloc[0] if len(at_rb) else np.nan
        rows.append(
            {
                "cycle_name": name,
                "experiment_id": first.experiment_id,
                "cycle_status": first.cycle_status,
                "candidate_end_source": first.get("candidate_end_source", "actual_preparation"),
                "observation_end": first.get("observation_end", pd.NaT),
                "preparation_heat": first.preparation_heat,
                "cycle_start": first.cycle_start,
                "stable_heating_start": first.stable_heating_start,
                "t_star": first.t_star,
                "t_RB": first.t_RB,
                "rb_status": first.rb_status,
                "COP_eff_selected": best,
                "COP_eff_RB": rb_cop,
                "COP_relative_change_pct": 100 * (best / rb_cop - 1) if rb_cop > 0 else np.nan,
                "time_difference_minutes": (
                    pd.to_datetime(first.t_star) - pd.to_datetime(first.t_RB)
                ).total_seconds()
                / 60,
                **{col: first[col] for col in curve if col.startswith("cycle_cop_basin_")},
            }
        )
    return pd.DataFrame(rows)


def compare_preparation_runs(run, other_run, output):
    from plots.defrost_decision import render_preparation_comparison

    modes = {
        table.preparation_heat.iloc[0]: table
        for table in (
            pd.read_csv(run / "cycle_comparison.csv"),
            pd.read_csv(other_run / "cycle_comparison.csv"),
        )
    }
    if set(modes) != {"include", "zero"}:
        raise ValueError("comparison requires one include run and one zero run")
    paired = modes["include"].merge(
        modes["zero"], on="cycle_name", suffixes=("_include", "_zero"), how="outer"
    )
    paired["common_supported"] = (
        paired.COP_eff_selected_include.notna() & paired.COP_eff_selected_zero.notna()
    )
    paired["COP_difference"] = (
        paired.COP_eff_selected_include - paired.COP_eff_selected_zero
    ).where(paired.common_supported)
    paired["time_difference_minutes"] = (
        (
            pd.to_datetime(paired.t_star_include) - pd.to_datetime(paired.t_star_zero)
        ).dt.total_seconds()
        / 60
    ).where(paired.common_supported)
    paired["COP_relative_difference_pct"] = (
        100 * paired.COP_difference / paired.COP_eff_selected_zero
    )
    curves = {}
    for folder in (run, other_run):
        curve = pd.read_csv(folder / "candidate_decisions.csv")
        curves[curve.preparation_heat.iloc[0]] = curve[
            ["cycle_name", "candidate_defrost_time", "cycle_cop", "cycle_cop_eligible"]
        ]
    common = curves["include"].merge(
        curves["zero"], on=["cycle_name", "candidate_defrost_time"], suffixes=("_include", "_zero")
    )
    common = common.loc[common.cycle_cop_eligible_include & common.cycle_cop_eligible_zero]
    fixed = common.merge(paired[["cycle_name", "t_star_zero"]], on="cycle_name")
    fixed = fixed.loc[
        pd.to_datetime(fixed.candidate_defrost_time).eq(pd.to_datetime(fixed.t_star_zero))
    ].copy()
    fixed["COP_same_time_difference"] = fixed.cycle_cop_include - fixed.cycle_cop_zero
    fixed["COP_same_time_relative_pct"] = (
        100 * fixed.COP_same_time_difference / fixed.cycle_cop_zero
    )
    paired = paired.merge(
        fixed[["cycle_name", "COP_same_time_difference", "COP_same_time_relative_pct"]],
        on="cycle_name",
        how="left",
    )
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "paired_cycles.csv", index=False)
    render_preparation_comparison(paired, output / "preparation_heat_comparison.png")
    return paired


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_jobs < 1 or args.candidate_step_seconds < 1:
        raise ValueError("workers and candidate step must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        raise ValueError("invalid run name")
    if args.publish_dataset and not (args.figures or args.figures_only):
        raise ValueError("--publish-dataset requires --figures or --figures-only")
    models = load_defrost_event_models(args.model_file)
    if models.get("cop_definition") != "refrigerant_effective_heat":
        raise ValueError("refit effective COP event models first")
    if args.current_csv:
        result = current_effective_cop(
            pd.read_csv(args.current_csv),
            models,
            args.experiment_id,
            preparation_heat=args.preparation_heat,
        )
        print(pd.Series(result).to_json(date_format="iso", force_ascii=False))
        return 0
    if models["recovery_settings"] != recovery_settings(args):
        raise ValueError(
            "recovery arguments differ from fitted boundaries; refit or use matching arguments"
        )
    run = args.output_root / "defrost_decisions" / args.run_name
    if run.exists() and not args.overwrite and not args.figures_only:
        raise FileExistsError(f"run exists: {run}")
    loader = DatasetLoader(args.dataset)
    valid_names = set(loader.list_cycles(statuses={"valid"}).cycle_name)
    cycles = [name for name in (args.cycles or sorted(valid_names)) if name in valid_names]
    if not cycles:
        raise ValueError("no valid cycles selected")
    if args.dry_run:
        print(f"{len(cycles)} cycles | {args.preparation_heat} | {run}")
        return 0
    boundaries = loader.configure_recovery(recovery_settings(args))
    if args.figures_only:
        from plots.defrost_decision import render_current_decision_figures

        decisions = pd.read_csv(run / "candidate_decisions.csv", low_memory=False)
        for column in decisions:
            if column.endswith(("_time", "_start", "_end", "_t_star")) or column in {
                "t_star",
                "t_RB",
            }:
                decisions[column] = pd.to_datetime(decisions[column])
        render_current_decision_figures(
            decisions, loader, args.output_root / "test" / args.run_name, n_jobs=args.n_jobs,
            fetch_cloud_images=args.fetch_cloud_images
        )
        if args.publish_dataset:
            from dataset_tools.dataset_maintenance import publish_effective_decision_assets

            publish_effective_decision_assets(
                args.dataset, run, args.output_root / "test" / args.run_name
            )
        if args.compare_run:
            compare_preparation_runs(
                run,
                args.compare_run,
                args.output_root / "test" / f"{args.run_name}_preparation_heat",
            )
        return 0
    tables = []
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        results = Parallel(return_as="generator_unordered")(
            delayed(calculate_cycle)(
                loader,
                name,
                models,
                candidate_step_seconds=args.candidate_step_seconds,
                prediction_mode=args.prediction_mode,
                preparation_heat=args.preparation_heat,
            )
            for name in cycles
        )
        for index, table in enumerate(results, 1):
            tables.append(table)
            print(f"[effective COP] {index}/{len(cycles)}", flush=True)
    decisions = pd.concat(tables, ignore_index=True).sort_values(
        ["cycle_name", "candidate_defrost_time"]
    )
    run.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(run / "candidate_decisions.csv", index=False)
    if args.preparation_heat == "zero":
        from dataset_tools.cycle_metadata import update_effective_cop_quality

        update_effective_cop_quality(args.dataset, decisions, cycle_names=cycles,
                                     source=run / "candidate_decisions.csv")
    summary = summarize_decisions(decisions)
    summary.to_csv(run / "cycle_comparison.csv", index=False)
    boundaries.to_csv(run / "recovery_boundaries.csv", index=False)
    (run / "run_settings.json").write_text(
        json.dumps(
            {**vars(args), "cop_definition": models["cop_definition"],
             "effective_heat_rule": "outlet_at_least_recovery_temperature"},
            default=str, indent=2
        )
    )
    if args.figures:
        from plots.defrost_decision import render_current_decision_figures

        render_current_decision_figures(
            decisions, loader, args.output_root / "test" / args.run_name, n_jobs=args.n_jobs,
            fetch_cloud_images=args.fetch_cloud_images
        )
    if args.publish_dataset:
        from dataset_tools.dataset_maintenance import publish_effective_decision_assets

        publish_effective_decision_assets(
            args.dataset, run, args.output_root / "test" / args.run_name
        )
    if args.compare_run:
        compare_preparation_runs(
            run, args.compare_run, args.output_root / "test" / f"{args.run_name}_preparation_heat"
        )
    print(summary.cycle_status.value_counts().to_string())
    print(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
