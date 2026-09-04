"""Build RGB timing labels from a cost optimum or selected defrost time."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from image_labels.timing import (
    build_labels,
    build_selected_time_labels,
    validate_cost,
    validate_selected_times,
)
from plots.image_labels import plot_label_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--label-source",
        choices=("cost-optimum", "selected-time"),
        required=True,
    )
    parser.add_argument(
        "--source-table",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--near-optimal-thresholds",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.05, 0.10],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--figure-output", type=Path)
    parser.add_argument(
        "--figure-format",
        nargs="+",
        choices=("png", "svg", "pdf"),
        default=["png"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    recorded = {
        "label_source": args.label_source,
        "source_table": str(args.source_table),
        "dataset": str(args.dataset),
        "output": str(args.output),
        "overwrite": args.overwrite,
        "near_optimal_thresholds": args.near_optimal_thresholds,
    }
    print(json.dumps(recorded, indent=2, sort_keys=True))
    source = pd.read_csv(args.source_table, low_memory=False)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"label output exists; pass --overwrite: {args.output}")
    if args.label_source == "cost-optimum":
        validate_cost(source)
        labels, balance, audit = build_labels(args.dataset, source, args.near_optimal_thresholds)
    else:
        validate_selected_times(source)
        if args.figures:
            raise ValueError("--figures currently applies only to cost-optimum labels")
        labels, balance, audit = build_selected_time_labels(args.dataset, source)
    args.output.mkdir(parents=True, exist_ok=args.overwrite)
    labels.to_parquet(args.output / "image_timing_labels.parquet", index=False)
    balance.to_csv(args.output / "label_balance.csv", index=False)
    audit.to_csv(args.output / "cycle_audit.csv", index=False)
    if args.figures:
        plot_label_figures(
            cost=source,
            labels=labels,
            balance=balance,
            thresholds=args.near_optimal_thresholds,
            output=args.figure_output or args.output / "figures",
            source_output=args.output / "figure_source_data",
            figure_formats=tuple(args.figure_format),
        )
    recorded["command"] = shlex.join(["uv", "run", "python", "build_image_labels.py", *arguments])
    (args.output / "run_settings.json").write_text(
        json.dumps(recorded, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
