"""循环有效 COP 与最优除霜时机。

输入：处理后的 Dataset 循环、已保存的除霜电耗 Ridge、RB 触发时刻先验 CSV。
计算：累计有效供热量 /（本循环恢复及制热电耗 + 预测准备及除霜电耗）。
输出：候选 COP、最优时刻、包含峰值的连续 1% 近优区间及 RB 比较。
近优区间内 COP≥峰值的 99%；不并入其他分离的高 COP 时段。
RB 时刻直接读表；该时刻的 COP 使用本次电耗模型计算，不重新回放 RB 规则。
仅写本次输出目录；显式 --publish-dataset 才回写 Dataset。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed, parallel_config

from dataset_tools import DatasetLoader
from dataset_tools.builder.detect_cycles import add_recovery_arguments, recovery_settings
from defrost_decision.candidate_quantities import calculate_cycle, current_effective_cop
from defrost_decision.selection_results import replay_saved_run, summarize_decisions
from defrost_event_models.event_electricity_ridge import (
    DEFAULT_MODEL_FILE,
    load_defrost_event_models,
)
from output_layout import OUTPUT, PROJECT, REPRODUCTION


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    common = parser.add_argument_group("常用：选择输入与结果目录")
    scientific = parser.add_argument_group("计算设置：默认即可复现当前方法")
    maintenance = parser.add_argument_group("维护：重选、绘图与发布")
    common.add_argument(
        "--dataset", type=Path, default=PROJECT / "dataset", help="处理后的 Dataset"
    )
    common.add_argument("--cycles", nargs="*", help="循环名；省略时计算全部有效循环")
    scientific.add_argument(
        "--all-cycles",
        action="store_true",
        help="also evaluate excluded cycles, retaining candidate-level quality checks",
    )
    scientific.add_argument("--action-step-seconds", type=int, default=None)
    common.add_argument(
        "--model-file",
        type=Path,
        help="已保存的 Ridge 参数 JSON；默认读取 <dataset>/除霜预测模型/三输入Ridge-两阶段-287-1-统一平滑/candidate_model_parameters.json",
    )
    common.add_argument(
        "--rb-reference", type=Path, help="RB 触发时刻 CSV；默认 <dataset>/priors/rb_reference.csv"
    )
    common.add_argument(
        "--output",
        type=Path,
        default=OUTPUT / REPRODUCTION / "03_COP候选/effective_cop",
        help="本次结果目录；直接写入候选表和汇总，不追加目录",
    )
    scientific.add_argument(
        "--candidate-step-seconds", type=int, default=10, help="候选时刻间隔；Dataset 测量仍为 1 s"
    )
    scientific.add_argument(
        "--prediction-mode",
        choices=("cross-fitted", "full-model"),
        default="cross-fitted",
        help="cross-fitted 使用对应实验的留出模型；新实验用 full-model",
    )
    common.add_argument("--n-jobs", type=int, default=6, help="按循环并行计算的进程数")
    maintenance.add_argument("--figures", action="store_true")
    maintenance.add_argument(
        "--reselect-run",
        type=Path,
        help="recalculate eligibility and selection from saved quantities into a new run; no model refit",
    )
    maintenance.add_argument("--fetch-cloud-images", action="store_true")
    maintenance.add_argument(
        "--figures-only", action="store_true", help="render saved candidate results"
    )
    maintenance.add_argument(
        "--publish-dataset",
        action="store_true",
        help="publish decision figures and update COP quality fields in the Dataset catalog",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="检查输入与模型折，打印实际路径；不计算候选、不写文件",
    )
    maintenance.add_argument("--overwrite", action="store_true", help="允许更新本次同名结果目录")
    scientific.add_argument(
        "--missing-preparation", choices=("exclude", "defrost-start"), default="exclude"
    )
    scientific.add_argument(
        "--initial-heating-start", choices=("recovery", "observed"), default="recovery"
    )
    maintenance.add_argument(
        "--reuse-run", type=Path, help="retain saved decisions for cycles not recalculated"
    )
    maintenance.add_argument(
        "--peak-after-seconds",
        type=float,
        default=None,
        help="0 requires one supported point below 99%% after the peak; omitted preserves historical screening",
    )
    scientific.add_argument(
        "--current-csv", type=Path, help="prepared measurement CSV of one heating-cycle prefix"
    )
    scientific.add_argument(
        "--current-registry",
        type=Path,
        help="channel_registry.json belonging to --current-csv (required with CSV input)",
    )
    scientific.add_argument("--experiment-id", default="new_experiment")
    add_recovery_arguments(scientific, dataset=True)
    return parser


def get_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.model_file is None:
        args.model_file = args.dataset / DEFAULT_MODEL_FILE.relative_to(PROJECT / "dataset")
    if (
        args.n_jobs < 1
        or args.candidate_step_seconds < 1
        or (
            args.action_step_seconds is not None
            and (
                args.action_step_seconds < 1
                or args.action_step_seconds % args.candidate_step_seconds
            )
        )
    ):
        raise ValueError("workers and candidate step must be positive")
    if args.publish_dataset and not (args.figures or args.figures_only):
        raise ValueError("--publish-dataset requires --figures or --figures-only")
    return args


def evaluate_current_csv(args, models):
    """Evaluate one supplied, already processed cycle prefix."""
    if args.current_registry is None:
        raise ValueError(
            "--current-csv requires its --current-registry; raw CSV is not prepared model input"
        )
    registry = json.loads(args.current_registry.read_text())
    processing = registry.get("measurement_processing")
    if not processing or processing != models.get("measurement_processing"):
        raise ValueError("CSV registry measurement processing must match the fitted model")
    frame = pd.read_csv(args.current_csv)
    if processing == "dataset_offline_linear_60s":
        from heat_smoothing.methods import measurement_values

        frame = measurement_values(frame)
    else:
        frame.attrs["measurement_processing"] = processing
    if args.dry_run:
        print(
            f"Measurements: {args.current_csv.resolve()}\nRegistry: {args.current_registry.resolve()}\nModel: {args.model_file.resolve()}\nProcessing: {processing}\nOutput: terminal only\nPreview only; no writes"
        )
        return 0
    result = current_effective_cop(
        frame,
        models,
        args.experiment_id,
    )
    print(pd.Series(result).to_json(date_format="iso", force_ascii=False))
    return 0


def calculate_candidates(args, loader, models, cycles, rb_reference):
    """Calculate selected cycles through the shared candidate calculation."""
    tables = []
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        results = Parallel(return_as="generator_unordered")(
            delayed(calculate_cycle)(
                loader,
                name,
                models,
                rb=rb_reference.loc[name],
                candidate_step_seconds=args.candidate_step_seconds,
                action_step_seconds=args.action_step_seconds,
                prediction_mode=args.prediction_mode,
                missing_preparation=args.missing_preparation,
                initial_heating_start=args.initial_heating_start,
            )
            for name in cycles
        )
        for index, table in enumerate(results, 1):
            tables.append(table)
            print(f"[effective COP] {index}/{len(cycles)}", flush=True)
    decisions = pd.concat(tables, ignore_index=True).sort_values(
        ["cycle_name", "candidate_defrost_time"]
    )
    if args.reuse_run:
        previous = pd.read_csv(args.reuse_run / "candidate_decisions.csv", low_memory=False)
        previous["candidate_defrost_time"] = pd.to_datetime(
            previous.candidate_defrost_time, format="mixed"
        )
        decisions = pd.concat(
            [previous.loc[~previous.cycle_name.isin(cycles)], decisions], ignore_index=True
        )
        decisions = decisions.sort_values(["cycle_name", "candidate_defrost_time"])
    return decisions


def save_results(args, loader, models, decisions, boundaries, run):
    """Save this run; Dataset publication is a separate explicit action."""
    run.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(run / "candidate_decisions.csv", index=False)
    summary = summarize_decisions(decisions)
    summary.to_csv(run / "cycle_comparison.csv", index=False)
    boundaries.to_csv(run / "recovery_boundaries.csv", index=False)
    (run / "run_settings.json").write_text(
        json.dumps(
            {
                **vars(args),
                "cop_definition": models["cop_definition"],
                "electricity_accounting": models.get(
                    "electricity_accounting", "stable_heating_plus_future_recovery"
                ),
                "effective_heat_rule": "without_trunc",
                "measurement_processing": loader.registry.get(
                    "measurement_processing", "dataset_causal_60s"
                ),
            },
            default=str,
            indent=2,
        )
    )
    return summary


def main(argv=None):
    args = get_args(argv)
    if args.reselect_run:
        return replay_saved_run(
            args.reselect_run,
            args.output.resolve(),
            args.dataset,
            args.n_jobs,
            args.publish_dataset,
            figures=args.figures or args.figures_only,
            dry_run=args.dry_run,
        )
    models = load_defrost_event_models(args.model_file)
    if models.get("cop_definition") != "refrigerant_effective_heat":
        raise ValueError("refit effective COP event models first")
    if args.current_csv:
        return evaluate_current_csv(args, models)
    if models["recovery_settings"] != recovery_settings(args):
        raise ValueError(
            "recovery arguments differ from fitted boundaries; refit or use matching arguments"
        )
    run = args.output.resolve()
    if run.exists() and not args.overwrite and not args.figures_only and not args.dry_run:
        raise FileExistsError(f"run exists: {run}")
    loader = DatasetLoader(args.dataset)
    if (
        models.get("dataset_resample_interval_seconds") is not None
        and models["dataset_resample_interval_seconds"]
        != loader.registry["resample_interval_seconds"]
    ):
        raise ValueError(
            "Dataset interval differs from the fitted cost model; refit before recomputing COP"
        )
    valid_names = set(
        loader.list_cycles(statuses=None if args.all_cycles else {"valid"}).cycle_name
    )
    cycles = [name for name in (args.cycles or sorted(valid_names)) if name in valid_names]
    if not cycles:
        raise ValueError("no valid cycles selected")
    rb_reference = None
    if args.figures_only:
        pd.read_csv(run / "candidate_decisions.csv", nrows=0)
    else:
        rb_reference = loader.load_rb_reference(cycles, path=args.rb_reference)
        if args.prediction_mode == "cross-fitted":
            catalog = loader.list_cycles().set_index("cycle_name")
            experiments = set(catalog.loc[cycles, "experiment_id"])
            missing = experiments - set(models["phase_models"]["folds"])
            if missing:
                raise ValueError(
                    f"No retrospective model folds: {sorted(missing)}; use --prediction-mode full-model for new experiments"
                )
        elif "E_prep_D_kwh" not in models["phase_models"]["full_data_model"]:
            raise ValueError("Ridge file has no full-data electricity model")
    if args.dry_run:
        print(
            f"Dataset: {args.dataset.resolve()}\nModel: {args.model_file.resolve()}\nRB: {(args.rb_reference or args.dataset / 'priors/rb_reference.csv').resolve()}\nProcessing: {loader.registry.get('measurement_processing')}\nPrediction: {args.prediction_mode}\nCycles: {len(cycles)}\nOutput: {run.resolve()}\nFigures: {args.figures or args.figures_only}\nPublish Dataset: {args.publish_dataset}\nPreview only; no writes"
        )
        return 0
    boundaries = loader.configure_recovery(recovery_settings(args))
    if args.figures_only:
        decisions = pd.read_csv(run / "candidate_decisions.csv", low_memory=False)
        for column in decisions:
            if column.endswith(("_time", "_start", "_end", "_t_star")) or column in {
                "t_star",
                "t_RB",
            }:
                decisions[column] = pd.to_datetime(decisions[column])
    else:
        decisions = calculate_candidates(args, loader, models, cycles, rb_reference)
        summary = save_results(args, loader, models, decisions, boundaries, run)
        print(summary.cycle_status.value_counts().to_string())
    if args.figures or args.figures_only:
        from plots.defrost_decision import render_current_decision_figures

        render_current_decision_figures(
            decisions,
            loader,
            run / "figures",
            n_jobs=args.n_jobs,
            fetch_cloud_images=args.fetch_cloud_images,
        )
    if args.publish_dataset:
        from dataset_tools.cycle_metadata import update_effective_cop_quality
        from plots.dataset import publish_effective_decision_assets

        if not args.figures_only:
            update_effective_cop_quality(
                args.dataset,
                decisions,
                cycle_names=cycles,
                source=run / "candidate_decisions.csv",
                minimum_after_seconds=args.peak_after_seconds,
            )
        publish_effective_decision_assets(args.dataset, run, run / "figures")
    print(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
