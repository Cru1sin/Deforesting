"""LOEO shared/group thresholds on frozen first crossings and one COP objective."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from image_models.overlap import teacher_crossing


def replay_curve(curve, thresholds):
    rows = []
    curve = curve.sort_values("candidate_defrost_time")
    for threshold in thresholds:
        result = teacher_crossing(curve, threshold)
        selected = curve.loc[curve.candidate_defrost_time.eq(result["crossing_time"])]
        cop = selected.cycle_cop.iloc[0] if result["status"] == "evaluated" else np.nan
        rows.append(
            dict(
                cycle_name=curve.cycle_name.iloc[0],
                experiment_id=curve.experiment_id.iloc[0],
                Ts=curve.Ts.iloc[0],
                threshold=threshold,
                selected_cop=cop,
                **result,
            )
        )
    return pd.DataFrame(rows)


def choose_threshold(replays, training_support="complete"):
    scored = replays.groupby("threshold").agg(
        mean_cop=("selected_cop", "mean"),
        scored_cycles=("selected_cop", "count"),
        cycles=("cycle_name", "size"),
    )
    # Unknown COP cannot silently improve a threshold by dropping its failures.
    valid = scored.loc[
        scored.scored_cycles.eq(scored.cycles)
        if training_support == "complete"
        else scored.scored_cycles.gt(0)
    ]
    if valid.empty:
        return None, scored
    threshold = (
        valid.reset_index()
        .sort_values(["mean_cop", "threshold"], ascending=[False, False], kind="stable")
        .threshold.iloc[0]
    )
    return float(threshold), scored


def loeo_replay(grid, members, cohort, training_support="complete"):
    rows = []
    folds = []
    pool = grid.loc[grid.cycle_name.isin(members)]
    for day, held in pool.groupby("experiment_id"):
        training = pool.loc[pool.experiment_id.ne(day)]
        for policy in ["shared", "by_temperature"]:
            for ts, query in held.groupby("Ts"):
                train = training if policy == "shared" else training.loc[training.Ts.eq(ts)]
                threshold, scan = choose_threshold(train, training_support)
                folds.append(
                    dict(
                        cohort=cohort,
                        experiment_id=day,
                        policy=policy,
                        Ts=ts,
                        threshold=threshold,
                        training_cycles=train.cycle_name.nunique(),
                        training_days=train.experiment_id.nunique(),
                        training_support=training_support,
                        scored_at_selected=int(scan.loc[threshold, "scored_cycles"])
                        if threshold is not None
                        else 0,
                        fully_scored_thresholds=int(scan.scored_cycles.eq(scan.cycles).sum()),
                    )
                )
                if threshold is None:
                    selected = query.drop_duplicates("cycle_name").copy()
                    selected["status"] = "no_fully_scored_training_threshold"
                    selected["threshold"] = np.nan
                    selected["selected_cop"] = np.nan
                    selected["cop_regret"] = np.nan
                    selected["hit_1pct"] = False
                    selected["crossing_time"] = pd.NaT
                else:
                    selected = query.loc[query.threshold.eq(threshold)].copy()
                rows.append(selected.assign(cohort=cohort, policy=policy))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(folds)


def summarize_decisions(results):
    valid = results.loc[results.status.eq("evaluated")]
    common = valid.groupby(["cohort", "cycle_name"]).policy.nunique()
    keys = common.loc[common.eq(2)].reset_index()[["cohort", "cycle_name"]]
    summaries = []
    for population, table in [
        ("all_selected", results),
        ("common_evaluated", results.merge(keys, on=["cohort", "cycle_name"])),
    ]:
        for (cohort, policy), g in table.groupby(["cohort", "policy"]):
            for ts, part in [("all", g), *list(g.groupby("Ts"))]:
                evaluated = part.loc[part.status.eq("evaluated")]
                summaries.append(
                    dict(
                        population=population,
                        cohort=cohort,
                        policy=policy,
                        Ts=ts,
                        cycles=len(part),
                        days=part.experiment_id.nunique(),
                        evaluated=len(evaluated),
                        hits_1pct=int(evaluated.hit_1pct.eq(True).sum()),
                        hit_fraction_all=evaluated.hit_1pct.eq(True).sum() / len(part),
                        mean_evaluated_cop=evaluated.selected_cop.mean(),
                        mean_regret_pct=100 * evaluated.cop_regret.mean(),
                    )
                )
    return pd.DataFrame(summaries)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    actions = pd.read_csv(args.source / "action_points.csv", parse_dates=["candidate_defrost_time"])
    groups = pd.read_csv(args.groups)
    actions = actions.merge(groups[["cycle_name", "Ts"]], on="cycle_name", validate="many_to_one")
    actions = actions.loc[actions.Ts.isin([50.0, 55.0, 60.0])]
    frozen = pd.read_csv(args.decisions, low_memory=False, parse_dates=["candidate_defrost_time"])
    keys = ["cycle_name", "candidate_defrost_time"]
    checked = actions.merge(
        frozen[keys + ["cycle_cop", "cycle_cop_eligible"]],
        on=keys,
        suffixes=("", "_frozen"),
        validate="one_to_one",
    )
    np.testing.assert_allclose(checked.cycle_cop, checked.cycle_cop_frozen, equal_nan=True)
    assert checked.cycle_cop_eligible.equals(checked.cycle_cop_eligible_frozen)
    thresholds = np.arange(1, 200) / 200
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.n_jobs, max_nbytes=None)(
            delayed(replay_curve)(curve, thresholds) for _, curve in actions.groupby("cycle_name")
        )
    grid = pd.concat(results, ignore_index=True)
    grid.to_csv(args.output / "first_crossing_grid.csv", index=False)
    screening = pd.read_csv(args.source / "cycle_screening.csv")
    output = []
    folds = []
    for cohort, members in [
        ("frozen_core", screening.loc[screening.core_overlap, "cycle_name"]),
        ("all", screening.cycle_name),
    ]:
        result, fold = loeo_replay(grid, members, cohort, args.training_support)
        output.append(result)
        folds.append(fold)
    results = pd.concat(output, ignore_index=True)
    results.to_csv(args.output / "held_out_decisions.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(args.output / "selected_thresholds.csv", index=False)
    summary = summarize_decisions(results)
    summary.to_csv(args.output / "decision_summary.csv", index=False)
    wide = results.pivot(
        index=["cohort", "cycle_name", "experiment_id", "Ts"],
        columns="policy",
        values="selected_cop",
    ).dropna()
    wide["conditional_minus_shared_cop"] = wide.by_temperature - wide.shared
    wide.to_csv(args.output / "paired_cop_changes.csv")
    print(summary.to_string(index=False), flush=True)
    print(results.groupby(["cohort", "policy", "status"]).size().to_string(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=Path("output/99_历史归档/legacy_test/rgb_without_trunc_screening_20260924"),
    )
    p.add_argument(
        "--groups",
        type=Path,
        default=Path(
            "output/99_历史归档/legacy_test/threshold_overlap_by_ts_day_without_trunc_20260924/group_assignments.csv"
        ),
    )
    p.add_argument(
        "--decisions",
        type=Path,
        default=Path(
            "output/90_复现材料/03_COP候选/历史统一队列_无截断候选/candidate_decisions.csv"
        ),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-jobs", type=int, default=6)
    p.add_argument(
        "--training-support",
        choices=["complete", "available"],
        default="complete",
        help="Available-case training is exploratory: threshold-specific training populations differ.",
    )
    main(p.parse_args())
