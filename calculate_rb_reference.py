"""次级维护：回放或核验 RB 规则；四阶段主流程直接读取已保存的 RB 先验。

输入：Dataset 原始控制观测及冻结规则 RB-20260923。
输出：本次目录内 rb_defrost_triggers.csv；--publish-prior 显式更新 Dataset 先验。
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from joblib import parallel_config
from sklearn.utils.parallel import Parallel, delayed

from dataset_tools import DatasetLoader
from defrost_decision.baselines import rule_based
from defrost_decision.baselines.audit import audit_actual, run_table_audit
from output_layout import PROJECT, analysis_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    parser.add_argument("--cycles", nargs="*")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "output/90_复现材料/03_COP候选/rb_reference",
        help="本次实际结果目录",
    )
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--publish-prior",
        action="store_true",
        help="完整回放后更新 <dataset>/priors/rb_reference.csv",
    )
    parser.add_argument(
        "--audit-actual",
        action="store_true",
        help="compare replay with recorded RB controller events",
    )
    parser.add_argument(
        "--table-audit",
        action="store_true",
        help="Audit the supplied PPT rule definition separately from frozen RB",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        help="完整候选 COP 表，用于重新筛选循环",
    )
    parser.add_argument("--figures", action="store_true")
    parser.add_argument(
        "--no-case8", action="store_true", help="Disable the 150-minute timeout in table replay"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    if args.n_jobs < 1:
        raise ValueError("n-jobs must be positive")
    if args.publish_prior and (args.cycles or args.table_audit):
        raise ValueError("--publish-prior requires all cycles and the frozen RB rule")
    run = analysis_output(args.output.resolve())
    print(f"Dataset: {args.dataset.resolve()}\nOutput: {run}", flush=True)
    if run.exists() and not args.overwrite:
        raise FileExistsError(f"run directory exists; pass --overwrite: {run}")

    loader = DatasetLoader(args.dataset)
    available = loader.list_cycles()["cycle_name"].astype(str).tolist()
    cycles = available if not args.cycles else args.cycles
    missing = sorted(set(cycles) - set(available))
    if missing:
        raise ValueError(f"unknown cycles: {', '.join(missing)}")
    if args.table_audit:
        return run_table_audit(args, loader, cycles)
    if args.dry_run:
        print(f"RB reference: cycles={len(cycles)}, n_jobs={args.n_jobs}, output={run}")
        return 0

    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        rows = Parallel(return_as="list")(
            delayed(rule_based.calculate_cycle)(loader, cycle_name) for cycle_name in cycles
        )
    results = pd.DataFrame(rows).sort_values("cycle_name", kind="stable")
    run.mkdir(parents=True, exist_ok=True)
    results.to_csv(run / "rb_defrost_triggers.csv", index=False)
    if args.publish_prior:
        prior = args.dataset / "priors/rb_reference.csv"
        prior.parent.mkdir(parents=True, exist_ok=True)
        results[["cycle_name", "experiment_id", "t_RB", "rb_status"]].assign(
            rule="RB-20260923"
        ).to_csv(prior, index=False)
    if args.audit_actual:
        audit_actual(loader, results).to_csv(run / "rb_actual_comparison.csv", index=False)
    settings = {
        "method": "rule_based_replay",
        "snapshot_policy": "last_controller_snapshot_per_second",
        "rule": "RB-20260923",
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
