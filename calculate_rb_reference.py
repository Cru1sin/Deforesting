"""Replay the frozen historical rule-based defrost baseline."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from joblib import parallel_config
from sklearn.utils.parallel import Parallel, delayed

from dataset_tools import DatasetLoader
from defrost_decision.baselines import rule_based


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--cycles", nargs="*")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--run-name", default="rb_reference")
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        raise ValueError("run name may contain only letters, numbers, dot, underscore and hyphen")
    if args.n_jobs < 1:
        raise ValueError("n-jobs must be positive")
    run = args.output_root / "defrost_decisions" / args.run_name
    if run.exists() and not args.overwrite:
        raise FileExistsError(f"run directory exists; pass --overwrite: {run}")

    loader = DatasetLoader(args.dataset)
    available = loader.list_cycles()["cycle_name"].astype(str).tolist()
    cycles = available if not args.cycles else args.cycles
    missing = sorted(set(cycles) - set(available))
    if missing:
        raise ValueError(f"unknown cycles: {', '.join(missing)}")
    if args.dry_run:
        print(
            f"RB reference: run={args.run_name}, cycles={len(cycles)}, "
            f"n_jobs={args.n_jobs}, output={run}"
        )
        return 0

    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        rows = Parallel(return_as="list")(
            delayed(rule_based.calculate_cycle)(loader, cycle_name) for cycle_name in cycles
        )
    results = pd.DataFrame(rows).sort_values("cycle_name", kind="stable")
    run.mkdir(parents=True, exist_ok=True)
    results.to_csv(run / "rb_defrost_triggers.csv", index=False)
    settings = {
        "method": "frozen_historical_rule_based_reference",
        "cycles": len(cycles),
        "n_jobs": args.n_jobs,
        "command": shlex.join(["uv", "run", "python", "calculate_rb_reference.py", *arguments]),
    }
    (run / "run_settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
