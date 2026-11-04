"""Frozen trajectory ranking against an external DTW reference, without fitting or plots.

Reference ties get temporal order only within a cycle. This is proxy agreement,
not frost-mass accuracy. Thresholds define reference regions, not calibrated mass.
Run with python -m tests.training_workspace.reference_ranking --help.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score, roc_curve


def pair_counts(reference, score):
    """Exact [ordered pairs, twice concordance credit, prediction ties], O(n log n)."""
    x, y = np.asarray(reference), np.asarray(score)
    total = len(x) * (len(x) - 1) // 2
    ties = []
    for values in [x, y, np.column_stack([x, y])]:
        counts = np.unique(values, axis=0, return_counts=True)[1]
        ties.append(int(np.sum(counts * (counts - 1) // 2)))
    comparable = total - ties[0]
    signed = 0
    if comparable and total > ties[1]:
        tau = kendalltau(x, y).statistic
        signed = round(tau * np.sqrt(float(comparable) * (total - ties[1])))
    return np.array([comparable, comparable + signed, ties[1] - ties[2]], dtype=np.int64)


def group_counts(frames, fields, x="reference"):
    return sum(
        (pair_counts(g[x], g.score) for _, g in frames.groupby(fields)),
        start=np.zeros(3, dtype=np.int64),
    )


def ranking_counts(frames):
    total = pair_counts(frames.reference, frames.score)
    within = group_counts(frames, "cycle_name")
    timed = frames.assign(time_order=pd.to_datetime(frames.image_time).astype("int64"))
    tied = group_counts(timed, ["cycle_name", "reference"], "time_order")
    return dict(
        within_cycle_reference=within,
        cross_cycle_reference=total - within,
        cross_experiment_reference=total - group_counts(frames, "experiment_id"),
        within_cycle_time_ties=tied,
        within_cycle_combined=within + tied,
    )


def metrics(counts):
    n, twice_credit, ties = map(int, counts)
    return dict(
        pairs=n,
        concordance=twice_credit / (2 * n) if n else np.nan,
        prediction_tie_fraction=ties / n if n else np.nan,
    )


def plateau_gap_counts(frames, minutes):
    counts = np.zeros(3, dtype=np.int64)
    for _, g in frames.groupby(["cycle_name", "reference"]):
        g = g.sort_values("image_time")
        i, j = np.triu_indices(len(g), k=1)
        time = g.time_order.to_numpy()
        keep = time[j] - time[i] >= minutes * 60 * 1e9
        delta = g.score.to_numpy()[j[keep]] - g.score.to_numpy()[i[keep]]
        counts += [len(delta), int((1 + np.sign(delta)).sum()), int((delta == 0).sum())]
    return counts


def experiment_summary(pairs, bootstrap=2000):
    """Resample experiments, retaining both endpoints of each cross-experiment pair.

    Copies of the same original experiment never become cross-experiment evidence.
    Interval describes this frozen cohort, not uncertainty from refitting the models.
    """
    rows = []
    for low, g in pairs.groupby("reference_min"):
        g = g.loc[g.pairs.gt(0)]
        names = pd.Index(sorted(set(g.experiment_a) | set(g.experiment_b)))
        if len(names) < 2:
            continue
        a, b = names.get_indexer(g.experiment_a), names.get_indexer(g.experiment_b)
        multiplicities = np.random.default_rng(0).multinomial(
            len(names), np.full(len(names), 1 / len(names)), size=bootstrap
        )
        weight = multiplicities[:, a] * multiplicities[:, b]
        for weighting, scale in [
            ("experiment_pair_equal", np.ones(len(g))),
            ("frame_pair_equal", g.pairs.to_numpy()),
        ]:
            weighted = weight * scale
            denominator = weighted.sum(axis=1)
            valid = denominator > 0
            estimates = (weighted[valid] @ g.concordance.to_numpy()) / denominator[valid]
            rows.append(
                dict(
                    reference_min=low,
                    weighting=weighting,
                    concordance=np.average(g.concordance, weights=scale),
                    ci_low=np.quantile(estimates, 0.025),
                    ci_high=np.quantile(estimates, 0.975),
                    experiment_pairs=len(g),
                    experiments=len(names),
                    bootstrap_valid=int(valid.sum()),
                )
            )
    return pd.DataFrame(rows)


def evaluate_source(source, reference, thresholds, bootstrap=2000):
    source = Path(source)
    predicted = pd.read_csv(source / "progression.csv")
    if not predicted.frost_only.all():
        raise ValueError(f"{source}: comparison requires frost-only predictions")
    columns = ["cycle_name", "file_name", "visual_frost_score"]
    if "projection_distance" in predicted:
        columns.append("projection_distance")
    frames = reference.merge(
        predicted[columns].rename(columns={"visual_frost_score": "score"}),
        on=["cycle_name", "file_name"],
        how="left",
        validate="one_to_one",
    )
    coverage = dict(
        method=source.name,
        reference_frames=len(frames),
        reference_unscored=int(frames.reference.isna().sum()),
        prediction_unscored=int(frames.score.isna().sum()),
        evaluated_frames=int((frames.reference.notna() & frames.score.notna()).sum()),
    )
    valid = frames.loc[np.isfinite(frames.reference) & np.isfinite(frames.score)].copy()
    summary, cycles, experiment_pairs, detection = [], [], [], []
    for low in [0.0, *thresholds]:
        region = valid.loc[valid.reference.ge(low)]
        for scope, counts in ranking_counts(region).items():
            summary.append(
                dict(
                    reference_min=low,
                    scope=scope,
                    frames=len(region),
                    cycles=region.cycle_name.nunique(),
                    experiments=region.experiment_id.nunique(),
                    **metrics(counts),
                )
            )
        for gap in [5, 10]:
            summary.append(
                dict(
                    reference_min=low,
                    scope=f"within_cycle_time_ties_gap{gap}min",
                    frames=len(region),
                    **metrics(plateau_gap_counts(region, gap)),
                )
            )
        grouped = list(region.groupby("experiment_id"))
        counts_by_exp = {name: pair_counts(g.reference, g.score) for name, g in grouped}
        for (a, ga), (b, gb) in combinations(grouped, 2):
            pooled = pd.concat([ga, gb])
            counts = (
                pair_counts(pooled.reference, pooled.score) - counts_by_exp[a] - counts_by_exp[b]
            )
            experiment_pairs.append(
                dict(reference_min=low, experiment_a=a, experiment_b=b, **metrics(counts))
            )
        for cycle, g in region.groupby("cycle_name"):
            g = g.sort_values("image_time")
            strict = pair_counts(g.reference, g.score)
            tied = group_counts(g, "reference", "time_order")
            for scope, counts in [
                ("reference", strict),
                ("time_ties", tied),
                ("combined", strict + tied),
            ]:
                cycles.append(
                    dict(
                        cycle_name=cycle,
                        experiment_id=g.experiment_id.iloc[0],
                        reference_min=low,
                        scope=scope,
                        frames=len(g),
                        score_min=g.score.min(),
                        score_max=g.score.max(),
                        score_first=g.score.iloc[0],
                        score_last=g.score.iloc[-1],
                        max_drawdown=float((g.score.cummax() - g.score).max()),
                        **metrics(counts),
                    )
                )
    weights = 1 / valid.groupby("cycle_name").score.transform("size")
    for low in thresholds:
        target = valid.reference.ge(low)
        if target.nunique() == 2:
            fpr, tpr, cut = roc_curve(target, valid.score, sample_weight=weights)
            best = np.flatnonzero(fpr <= 0.05)[-1]
            detection.append(
                dict(
                    reference_min=low,
                    positive_frames=int(target.sum()),
                    positive_cycles=valid.loc[target, "cycle_name"].nunique(),
                    cycle_weighted_auc=roc_auc_score(target, valid.score, sample_weight=weights),
                    descriptive_tpr_at_fpr05=tpr[best],
                    descriptive_fpr=fpr[best],
                    descriptive_score_threshold=cut[best],
                )
            )
    valid["reference_bin"] = np.minimum(np.floor(valid.reference * 10).astype(int), 9)
    bins = valid.groupby("reference_bin").agg(
        frames=("score", "size"),
        cycles=("cycle_name", "nunique"),
        experiments=("experiment_id", "nunique"),
        score_median=("score", "median"),
        score_q10=("score", lambda s: s.quantile(0.1)),
        score_q90=("score", lambda s: s.quantile(0.9)),
    )
    if "projection_distance" in valid:
        bins["projection_distance_median"] = valid.groupby(
            "reference_bin"
        ).projection_distance.median()
    atlas = pd.read_csv(source / "reference.csv")[["cycle_name", "file_name", "reference_phase"]]
    atlas = atlas.rename(columns={"reference_phase": "atlas_phase"}).merge(
        frames[["cycle_name", "file_name", "reference", "score"]],
        on=["cycle_name", "file_name"],
        how="left",
        validate="many_to_one",
    )
    pairs = pd.DataFrame(experiment_pairs)
    print(f"[ranking] {source.name}: {len(valid)} matched scored frames", flush=True)
    return dict(
        coverage=pd.DataFrame([coverage]),
        ranking=pd.DataFrame(summary),
        cycle_ranking=pd.DataFrame(cycles),
        experiment_pairs=pairs,
        cross_experiment_summary=experiment_summary(pairs, bootstrap),
        atlas_reference=atlas,
        heavy_detection=pd.DataFrame(detection),
        reference_bins=bins.reset_index(),
        matched_frames=frames,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heavy-thresholds", type=float, nargs="+", default=[0.7, 0.8, 0.9])
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    reference = pd.read_csv(args.reference / "progression.csv")
    if not reference.frost_only.all() or not reference.reference_method.eq("extended").all():
        raise ValueError("Use the frozen frost-only extended DTW reference")
    reference = reference.rename(columns={"visual_frost_score": "reference"})
    reference = reference[
        [
            "cycle_name",
            "file_name",
            "experiment_id",
            "image_time",
            "reference",
            "reference_index",
            "is_beyond_reference",
        ]
    ].copy()
    reference["image_time"] = pd.to_datetime(reference.image_time, format="mixed")
    reference["time_order"] = pd.to_datetime(reference.image_time).astype("int64")
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        tables = Parallel()(
            delayed(evaluate_source)(p, reference, args.heavy_thresholds, args.bootstrap)
            for p in args.sources
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for name in tables[0]:
        pd.concat(
            [
                table[name].assign(method=source.name)
                for source, table in zip(args.sources, tables, strict=True)
            ],
            ignore_index=True,
        ).to_csv(args.output / f"{name}.csv", index=False)
    (args.output / "run_settings.json").write_text(
        json.dumps(
            {
                **vars(args),
                "target": "external DTW order; time breaks within-cycle reference ties only",
                "prediction_ties": "half credit",
                "evaluation": "descriptive frozen cohort; no refitting",
            },
            default=str,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
