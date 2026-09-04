#!/usr/bin/env python3
"""Recompute and summarize leave-one-experiment-out model results."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from pathlib import Path

import pandas as pd

from image_models.evaluation import evaluate_run
from plots.image_models import (
    plot_model_figures,
    plot_probability_curves,
    plot_trigger_error_figures,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        type=Path,
        default=[Path("output/image_models/current")],
        help="one or more training run directories",
    )
    parser.add_argument("--output", type=Path, default=Path("output/evaluations/current"))
    parser.add_argument("--task", choices=("binary", "three"), default="binary")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--figure-output", type=Path)
    parser.add_argument(
        "--figure-format",
        nargs="+",
        choices=("png", "svg", "pdf"),
        default=["png"],
    )
    parser.add_argument("--optima", type=Path)
    parser.add_argument("--concentration", type=Path)
    parser.add_argument("--decision-table", type=Path)
    parser.add_argument("--probability-curves", action="store_true")
    parser.add_argument("--curve-image-feature", default="dinov2_cache")
    parser.add_argument("--curve-classifier", default="logistic_regression")
    parser.add_argument("--curve-window-minutes", type=float, default=10)
    parser.add_argument(
        "--continuous-stream",
        action="store_true",
        help="apply the 2-of-3 rule only when predictions cover consecutive source frames",
    )
    return parser


def run(args: argparse.Namespace, *, command: list[str] | None = None) -> int:
    if (args.optima is None) != (args.concentration is None):
        raise ValueError("--optima and --concentration must be provided together")
    if args.probability_curves and args.decision_table is None:
        raise ValueError("--probability-curves requires --decision-table")
    if args.output.exists() and not args.output.is_dir():
        raise FileExistsError(f"output exists and is not a directory: {args.output}")
    if args.overwrite and args.output.is_dir():
        shutil.rmtree(args.output)
    elif args.output.is_dir() and any(args.output.iterdir()):
        raise FileExistsError(f"output exists and is not empty: {args.output}")

    experiment_tables: list[pd.DataFrame] = []
    summary_tables: list[pd.DataFrame] = []
    for results_dir in args.results:
        metrics = pd.read_csv(results_dir / "fold_metrics.csv")
        predictions = pd.read_parquet(results_dir / "predictions.parquet")
        experiment_metrics, summary = evaluate_run(metrics, predictions, task=args.task)
        source = str(results_dir)
        experiment_metrics.insert(0, "source", source)
        summary.insert(0, "source", source)
        experiment_tables.append(experiment_metrics)
        summary_tables.append(summary)

    args.output.mkdir(parents=True, exist_ok=True)
    recorded_args = (
        dict(vars(args))
        if args.figures
        else {name: getattr(args, name) for name in ("results", "output", "task", "overwrite")}
    )
    recorded_args["command"] = shlex.join(["uv", "run", "python", *(command or sys.argv)])
    (args.output / "run_settings.json").write_text(
        json.dumps(recorded_args, default=str, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    experiment_metrics = pd.concat(experiment_tables, ignore_index=True)
    summary = pd.concat(summary_tables, ignore_index=True)
    experiment_metrics.to_csv(args.output / "experiment_metrics.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    if args.figures:
        plot_model_figures(
            summary=summary,
            output=args.figure_output or args.output / "figures",
            source_output=args.output / "figure_source_data",
            figure_formats=tuple(args.figure_format),
            optima=pd.read_csv(args.optima) if args.optima is not None else None,
            concentration=(
                pd.read_csv(args.concentration) if args.concentration is not None else None
            ),
        )
        if args.probability_curves:
            predictions = pd.concat(
                [pd.read_parquet(path / "predictions.parquet") for path in args.results],
                ignore_index=True,
            )
            decisions = pd.read_csv(args.decision_table, low_memory=False)
            plot_probability_curves(
                predictions=predictions,
                decisions=decisions,
                output=args.figure_output or args.output / "figures",
                source_output=args.output / "figure_source_data",
                image_feature=args.curve_image_feature,
                classifier=args.curve_classifier,
                window_minutes=args.curve_window_minutes,
                figure_formats=tuple(args.figure_format),
                continuous_stream=args.continuous_stream,
            )
            plot_trigger_error_figures(
                predictions=predictions,
                decisions=decisions,
                output=args.figure_output or args.output / "figures",
                source_output=args.output / "figure_source_data",
                image_feature=args.curve_image_feature,
                classifier=args.curve_classifier,
                continuous_stream=args.continuous_stream,
                figure_formats=tuple(args.figure_format),
            )
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
