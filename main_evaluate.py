#!/usr/bin/env python3
"""Recompute and summarize leave-one-experiment-out model results."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from model.evaluate import evaluate_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        type=Path,
        default=[Path("output/models/current")],
        help="one or more training run directories",
    )
    parser.add_argument("--output", type=Path, default=Path("output/models/evaluation"))
    parser.add_argument("--task", choices=("binary", "three"), default="binary")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in vars(args).items()}


def run(args: argparse.Namespace, *, command: list[str] | None = None) -> int:
    if args.output.exists() and not args.output.is_dir():
        raise FileExistsError(f"output exists and is not a directory: {args.output}")
    if args.output.is_dir() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists and is not empty: {args.output}")

    experiment_tables: list[pd.DataFrame] = []
    summary_tables: list[pd.DataFrame] = []
    for results_dir in args.results:
        metrics = pd.read_csv(results_dir / "metrics.csv")
        predictions = pd.read_parquet(results_dir / "predictions.parquet")
        experiment_metrics, summary = evaluate_run(metrics, predictions, task=args.task)
        source = str(results_dir)
        experiment_metrics.insert(0, "source", source)
        summary.insert(0, "source", source)
        experiment_tables.append(experiment_metrics)
        summary_tables.append(summary)

    args.output.mkdir(parents=True, exist_ok=True)
    serializable_args = _serializable_args(args)
    (args.output / "command.txt").write_text(
        shlex.join(command or sys.argv) + "\n", encoding="utf-8"
    )
    (args.output / "args.json").write_text(
        json.dumps(serializable_args, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.concat(experiment_tables, ignore_index=True).to_csv(
        args.output / "experiment_metrics.csv", index=False
    )
    pd.concat(summary_tables, ignore_index=True).to_csv(args.output / "summary.csv", index=False)
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
