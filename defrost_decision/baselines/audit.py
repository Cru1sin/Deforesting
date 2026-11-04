"""Compare frozen RB replay with observed controller events and the supplied rule table."""

from pathlib import Path

import pandas as pd
from joblib import parallel_config
from sklearn.utils.parallel import Parallel, delayed

from defrost_decision.baselines import rule_based


def audit_actual(loader, results):
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
        frame = loader.load_cycle_original(
            replay["cycle_name"], columns=[*rule_based.RAW_COLUMNS, code]
        )
        frame["timestamp"] = pd.to_datetime(frame.timestamp)
        frame = rule_based.controller_snapshots(frame)
        # At trigger, the firmware clears the counters; do not replace them with inferred values.
        before = frame.loc[frame.timestamp.lt(preparation)].dropna(subset=["p1__DefTim1'2_20"])
        after = frame.loc[frame.timestamp.ge(preparation) & frame.timestamp.le(defrost)]
        codes = pd.to_numeric(after[code], errors="coerce")
        codes = codes.loc[codes.gt(0)]
        row = {
            **replay,
            "actual_preparation": preparation,
            "actual_defrost": defrost,
            "recorded_condition_code": codes.iloc[0] if len(codes) else None,
        }
        # These codes are observed in this dataset; other firmware branches remain unknown.
        row["recorded_branch"] = {
            10: "Condition1",
            20: "Case1",
            23: "Case2",
            26: "Case3",
            29: "Case4",
        }.get(row["recorded_condition_code"], "")
        row["branch_matches"] = (
            bool(row["recorded_branch"]) and row["recorded_branch"] == replay["trigger_type"]
        )
        for key, time in (("preparation", preparation), ("defrost", defrost)):
            row[f"replay_minus_{key}_seconds"] = (
                (pd.to_datetime(replay["t_RB"]) - time).total_seconds()
                if pd.notna(replay["t_RB"]) and pd.notna(time)
                else float("nan")
            )
        if len(before):
            row.update(
                {
                    "last_preparation_sample": before.timestamp.iloc[-1],
                    **{
                        f"last_{column}": before[column].iloc[-1]
                        for column in rule_based.RAW_COLUMNS
                        if column != "timestamp"
                    },
                }
            )
        heating = pd.to_datetime(bounds.get("heating_start") or frame.timestamp.min())
        # Ignore the previous event's latched code at the start of the next cycle.
        observed = frame.loc[frame.timestamp.ge(heating) & frame.timestamp.le(defrost)].dropna(
            subset=[code]
        )
        onset = observed.loc[observed[code].gt(0) & observed[code].shift().eq(0), "timestamp"]
        row["recorded_trigger_time"] = onset.iloc[0] if len(onset) else pd.NaT
        at_trigger = frame.loc[
            frame.timestamp.eq(row["recorded_trigger_time"]), "ambient_temperature"
        ]
        t4 = pd.to_numeric(at_trigger, errors="coerce").iloc[0] if len(at_trigger) else float("nan")
        row["T4_at_recorded_trigger_C"] = t4
        row["ambient_case_at_trigger"] = f"Case{rule_based._case(t4)}" if pd.notna(t4) else ""
        if pd.notna(preparation):
            trace = rule_based.first_trigger(frame, heating, preparation, return_trace=True)
            row["max_T1_minutes"] = trace.T1_min.max()
            for branch in ("condition1", "condition2", "condition7"):
                hits = trace.index[trace[branch]]
                row[f"{branch}_first"] = hits[0] if len(hits) else pd.NaT
            for case in range(1, 6):
                hits = trace.index[trace.condition2 & trace.case.eq(case)]
                row[f"case{case}_first"] = hits[0] if len(hits) else pd.NaT
        if row["recorded_branch"] == "Condition1" and pd.notna(preparation):
            # Post-action temperatures can diagnose logging delay, never repair a causal trigger.
            stop = min(preparation + pd.Timedelta(seconds=60), frame.timestamp.max())
            post = rule_based.first_trigger(
                frame, heating, stop + pd.Timedelta(seconds=1), return_trace=True
            )
            reached = post.index[(post.index >= preparation) & post.historical_drop_c.ge(2 - 1e-9)]
            row["condition1_post_trigger_drop_delay_seconds"] = (
                (reached[0] - preparation).total_seconds() if len(reached) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def audit_table_cycle(loader, name, *, include_case8=True):
    """Compare the sampled table recipe and frozen replay on one boundary."""
    frozen = rule_based.calculate_cycle(loader, name)
    record = loader.get_cycle_record(name)
    rb = Path(record["assets"]["csv"]).parent.name == "cycles_RB"
    row = {
        "cycle_name": name,
        "experiment_id": record["experiment_id"],
        "group": "cycle_RB" if rb else "cycle",
        "frozen_branch": frozen["trigger_type"],
        "frozen_time": frozen["t_RB"],
        "observation_end": frozen["t_observation_end"],
    }
    frame = loader.load_cycle_original(name, columns=list(rule_based.RAW_COLUMNS))
    start = pd.to_datetime(record["boundaries"].get("heating_start"))
    end = pd.to_datetime(frozen["t_observation_end"])
    defrost = pd.to_datetime(record["boundaries"].get("defrost_start"))
    if pd.isna(pd.to_datetime(record["boundaries"].get("defrost_preparation_start"))) and pd.notna(
        defrost
    ):
        end = defrost
    row["table_observation_end"] = end
    trace = rule_based.first_trigger(
        frame,
        start,
        end,
        return_trace=True,
        recipe="ppt_sample_tolerant",
        include_case8=include_case8,
    )
    branch_hits = {
        "Condition1": trace.condition1,
        **{f"Case{i}": trace.condition2 & trace.case.eq(i) for i in range(1, 6)},
        "Case7": trace.condition7,
        "Case8": trace.condition8,
    }
    hits = trace.index[trace.triggered]
    first = hits[0] if len(hits) else pd.NaT
    row["table_time"] = first
    row["table_branch"] = next(
        (key for key, hit in branch_hits.items() if pd.notna(first) and hit.loc[first]), "未触发"
    )
    for key, hit in branch_hits.items():
        row["ever_" + key] = bool(hit.any())
    observed = trace.loc[trace.T1_min.notna()]
    last = observed.iloc[-1] if len(observed) else trace.iloc[-1]
    row["last_sample"] = last.name
    for key in (
        "T1_min",
        "T2_min",
        "T3_C",
        "T3o_C",
        "T4_C",
        "Twout_C",
        "historical_drop_c",
        "t1_limit_min",
        "t3_limit_c",
    ):
        row["last_" + key] = last[key]
    row["last_temperature_run_seconds"] = int(
        trace.loc[: last.name].case_temperature.iloc[::-1].cumprod().sum()
    )
    row["table_rule"] = "ppt_sampling_1s_20s_missing3_or_19points"
    row["case8_enabled"] = include_case8
    row["last_temperature_confirmed"] = bool(trace.loc[last.name, "case_confirmed"])
    window = trace.loc[: last.name].tail(20)
    known = window.T3_C.notna() & window.t3_limit_c.notna()
    row["last_temperature_window_missing_seconds"] = int((~known).sum())
    row["last_temperature_window_failed_seconds"] = int((known & ~window.case_temperature).sum())
    if rb:
        actual = audit_actual(loader, pd.DataFrame([frozen])).iloc[0]
        for key in (
            "recorded_condition_code",
            "recorded_branch",
            "recorded_trigger_time",
            "actual_preparation",
            "actual_defrost",
            "T4_at_recorded_trigger_C",
            "ambient_case_at_trigger",
        ):
            row[key] = actual[key]
        if pd.isna(row["recorded_condition_code"]) and pd.notna(defrost):
            code = "p1__化霜条件'1_00"
            observed = loader.load_cycle_original(name, columns=["timestamp", code])
            codes = observed.loc[pd.to_datetime(observed.timestamp).ge(defrost), code]
            codes = codes.loc[codes.gt(0)]
            if len(codes):
                row["recorded_condition_code"] = codes.iloc[0]
                row["recorded_branch"] = {
                    10: "Condition1",
                    20: "Case1",
                    23: "Case2",
                    26: "Case3",
                    29: "Case4",
                }.get(codes.iloc[0], "")
        branch = row["recorded_branch"]
        row["table_branch_matches"] = branch == row["table_branch"] if branch else None
        row["frozen_branch_matches"] = branch == row["frozen_branch"] if branch else None
        row["table_minus_preparation_s"] = (
            row["table_time"] - actual.actual_preparation
        ).total_seconds()
        if branch == "Condition1":
            checks = {
                "T1未超过35min": row["last_T1_min"] + 1 / 60 > 35,
                "T2不足6min": row["last_T2_min"] + 1 / 60 >= 6,
                "T3未降至−1°C": row["last_T3_C"] <= -1,
                "50–600s温降不足2°C": row["last_historical_drop_c"] >= 2 - 1e-9,
            }
        elif branch in {f"Case{i}" for i in range(1, 6)}:
            checks = {
                "T1未到表中门槛": row["last_T1_min"] + 1 / 60 >= row["last_t1_limit_min"] - 1e-9,
                "T2不足6min": row["last_T2_min"] + 1 / 60 >= 6,
                "当前T3未低于门槛": row["last_T3_C"] < row["last_t3_limit_c"],
                "温度未连续满足20s": row["last_temperature_confirmed"],
            }
        else:
            checks = {"实机条件码未映射": False}
        failures = [key for key, passed in checks.items() if not passed]
        row["last_snapshot_failures"] = ";".join(failures)
        row["audit_category"] = (
            "触发记录或边界不完整"
            if not branch or pd.isna(actual.actual_preparation)
            else "工况分类不一致"
            if branch in {f"Case{i}" for i in range(1, 6)}
            and row["ambient_case_at_trigger"]
            and branch != row["ambient_case_at_trigger"]
            else "规则一致"
            if row["table_branch_matches"]
            else "分支不一致"
        )
    return row


def run_table_audit(args, loader, cycles):
    from image_models.cycle_selection import screen_cv_cycles
    from output_layout import analysis_output

    if args.decisions is None:
        raise ValueError("--table-audit requires full saved --decisions")
    output = analysis_output(args.output.resolve())
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    decisions = (
        pd.read_parquet(args.decisions)
        if args.decisions.suffix == ".parquet"
        else pd.read_csv(args.decisions, low_memory=False)
    )
    screening = screen_cv_cycles(decisions, loader.list_cycles(), exclude_rb=False)
    output.mkdir(parents=True, exist_ok=True)
    screening.to_csv(output / "cycle_screening.csv", index=False)
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        rows = Parallel(verbose=5)(
            delayed(audit_table_cycle)(loader, name, include_case8=not args.no_case8)
            for name in cycles
        )
    source = pd.DataFrame(rows).sort_values("cycle_name")
    source["benchmark_selected"] = source.cycle_name.map(screening.set_index("cycle_name").selected)
    source.to_csv(output / "cycle_trigger_audit.csv", index=False)
    source.loc[source.group.eq("cycle_RB")].to_csv(
        output / "rb_table_discrepancies.csv", index=False
    )
    summaries = []
    branches = ["Condition1", *[f"Case{i}" for i in range(1, 6)], "Case7", "Case8", "未触发"]
    for group, data in source.groupby("group"):
        for branch in branches:
            count = int(data.table_branch.eq(branch).sum())
            summaries.append(
                dict(
                    group=group,
                    branch=branch,
                    n=count,
                    total=len(data),
                    percent=100 * count / len(data),
                    ever_n=int(data["ever_" + branch].sum()) if branch != "未触发" else count,
                    ever_percent=100
                    * (int(data["ever_" + branch].sum()) if branch != "未触发" else count)
                    / len(data),
                )
            )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "trigger_proportions.csv", index=False)
    rb = source.loc[source.group.eq("cycle_RB")]
    rb.groupby("audit_category").size().rename("n").to_csv(output / "rb_discrepancy_counts.csv")
    print(summary.to_string(index=False))
    print(rb.groupby("audit_category").size().to_string())
    if args.figures:
        from plots.rb_rule_explainer import render_rb_table_audit

        render_rb_table_audit(output)
    print(output)
    return 0
