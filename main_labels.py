"""Build hard RGB labels from a canonical V1 cost curve."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from labels.build import build_labels, build_policy_labels, validate_cost, validate_policy
from plots.labels import plot_label_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--mode", choices=("cost", "policy"), default="cost")
    parser.add_argument("--cost-csv", type=Path, default=Path("output/cost/v1/cost.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/labels/v1"))
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.10])
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
        "mode": args.mode,
        "cost_csv": str(args.cost_csv),
        "dataset": str(args.dataset),
        "output": str(args.output),
        "overwrite": args.overwrite,
        "thresholds": args.thresholds,
    }
    print(json.dumps(recorded, indent=2, sort_keys=True))
    cost = pd.read_csv(args.cost_csv, low_memory=False)
    (validate_cost if args.mode == "cost" else validate_policy)(cost)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"label output exists; pass --overwrite: {args.output}")
    labels, balance, audit = (
        build_labels(args.dataset, cost, args.thresholds)
        if args.mode == "cost"
        else build_policy_labels(args.dataset, cost)
    )
    args.output.mkdir(parents=True, exist_ok=args.overwrite)
    labels.to_parquet(args.output / "image_cost_labels.parquet", index=False)
    balance.to_csv(args.output / "label_balance.csv", index=False)
    audit.to_csv(args.output / "cycle_audit.csv", index=False)
    if args.figures:
        if args.mode != "cost":
            raise ValueError("policy labels are reviewed through model probability figures")
        plot_label_figures(
            cost=cost,
            labels=labels,
            balance=balance,
            thresholds=args.thresholds,
            output=args.figure_output or args.output / "figures",
            source_output=args.output / "figure_source_data",
            figure_formats=tuple(args.figure_format),
        )
    (args.output / "command.txt").write_text(
        shlex.join(["uv", "run", "python", "main_labels.py", *arguments]) + "\n",
        encoding="utf-8",
    )
    (args.output / "args.json").write_text(
        json.dumps(recorded, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
