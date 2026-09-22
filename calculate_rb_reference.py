"""Replay and audit the rule-based defrost baseline."""

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
    parser.add_argument("--condition1-drop-c", type=int, default=1)
    parser.add_argument("--forced-timeout-min", type=int, default=150,
                        help="historical timeout; 0 excludes this unverified branch")
    parser.add_argument("--sample-tolerance-seconds", type=int, choices=(0, 1), default=0)
    parser.add_argument("--confirmation-missing-seconds", type=int, choices=range(4), default=0)
    parser.add_argument("--condition1-tolerance-c", type=float, choices=(0, 0.1), default=0)
    parser.add_argument("--case-boundary", choices=("warmer", "colder"), default="warmer")
    parser.add_argument("--audit-actual", action="store_true", help="compare replay with recorded RB controller events")
    return parser


def audit_actual(loader, results, **rb_settings):
    """Keep replay separate from observed preparation/defrost and counter resets."""
    rows = []
    code = "p1__化霜条件'1_00"
    for replay in results.to_dict("records"):
        record = loader.get_cycle_record(replay["cycle_name"])
        if Path(record["assets"]["csv"]).parent.name != "cycles_RB":
            continue
        bounds = record["boundaries"]
        preparation = pd.to_datetime(bounds.get("defrost_preparation_start") or pd.NaT)
        defrost = pd.to_datetime(bounds.get("defrost_start") or pd.NaT)
        frame = loader.load_cycle_original(replay["cycle_name"], columns=[*rule_based.RAW_COLUMNS, code])
        frame["timestamp"] = pd.to_datetime(frame.timestamp)
        frame = rule_based.controller_snapshots(frame)
        # At trigger, the firmware clears the counters; do not replace them with inferred values.
        before = frame.loc[frame.timestamp.lt(preparation)].dropna(subset=["p1__DefTim1'2_20"])
        after = frame.loc[frame.timestamp.ge(preparation) & frame.timestamp.le(defrost)]
        codes = pd.to_numeric(after[code], errors="coerce")
        codes = codes.loc[codes.gt(0)]
        row = {**replay, "actual_preparation": preparation, "actual_defrost": defrost,
               "recorded_condition_code": codes.iloc[0] if len(codes) else None}
        # These codes are observed in this dataset; other firmware branches remain unknown.
        row["recorded_branch"] = {10: "Condition1", 20: "Case1", 23: "Case2",
                                  26: "Case3", 29: "Case4"}.get(row["recorded_condition_code"], "")
        row["branch_matches"] = bool(row["recorded_branch"]) and row["recorded_branch"] == replay["trigger_type"]
        for key, time in (("preparation", preparation), ("defrost", defrost)):
            row[f"replay_minus_{key}_seconds"] = (
                (pd.to_datetime(replay["t_RB"]) - time).total_seconds()
                if pd.notna(replay["t_RB"]) and pd.notna(time) else float("nan")
            )
        if len(before):
            row.update({"last_preparation_sample": before.timestamp.iloc[-1],
                        **{f"last_{column}": before[column].iloc[-1]
                           for column in rule_based.RAW_COLUMNS if column != "timestamp"}})
        heating = pd.to_datetime(bounds.get("heating_start") or frame.timestamp.min())
        # Ignore the previous event's latched code at the start of the next cycle.
        observed = frame.loc[frame.timestamp.ge(heating) & frame.timestamp.le(defrost)].dropna(subset=[code])
        onset = observed.loc[observed[code].gt(0) & observed[code].shift().eq(0), "timestamp"]
        row["recorded_trigger_time"] = onset.iloc[0] if len(onset) else pd.NaT
        if pd.notna(preparation):
            trace = rule_based.first_trigger(frame, heating, preparation, return_trace=True,
                                             **rb_settings)
            row["max_T1_minutes"] = trace.T1_min.max()
            for branch in ("condition1", "condition2", "condition7", "condition8"):
                hits = trace.index[trace[branch]]
                row[f"{branch}_first"] = hits[0] if len(hits) else pd.NaT
            for case in range(1, 6):
                hits = trace.index[trace.condition2 & trace.case.eq(case)]
                row[f"case{case}_first"] = hits[0] if len(hits) else pd.NaT
        if row["recorded_branch"] == "Condition1" and pd.notna(preparation):
            # Post-action temperatures can diagnose logging delay, never repair a causal trigger.
            stop = min(preparation + pd.Timedelta(seconds=60), frame.timestamp.max())
            post = rule_based.first_trigger(frame, heating, stop + pd.Timedelta(seconds=1),
                                             return_trace=True, **rb_settings)
            reached = post.index[(post.index >= preparation) & post.historical_drop_c.ge(
                rb_settings.get("condition1_drop_c", 1) - 1e-9)]
            row["condition1_post_trigger_drop_delay_seconds"] = (
                (reached[0] - preparation).total_seconds() if len(reached) else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


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

    rb_settings = {key: getattr(args, key) for key in (
        "condition1_drop_c", "forced_timeout_min", "sample_tolerance_seconds",
        "confirmation_missing_seconds", "condition1_tolerance_c", "case_boundary")}
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        rows = Parallel(return_as="list")(
            delayed(rule_based.calculate_cycle)(loader, cycle_name, **rb_settings) for cycle_name in cycles
        )
    results = pd.DataFrame(rows).sort_values("cycle_name", kind="stable")
    run.mkdir(parents=True, exist_ok=True)
    results.to_csv(run / "rb_defrost_triggers.csv", index=False)
    if args.audit_actual:
        audit_actual(loader, results, **rb_settings).to_csv(run / "rb_actual_comparison.csv", index=False)
    settings = {
        "method": "rule_based_replay",
        "snapshot_policy": "last_controller_snapshot_per_second",
        **rb_settings,
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
