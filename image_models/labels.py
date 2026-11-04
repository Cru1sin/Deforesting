"""COP action labels and frozen CDF training targets."""

import numpy as np
import pandas as pd


def stopping_cdf_targets(rows):
    """Uniform-time CDF between full teacher bounds, before input masking.

    Training fills interior holes; exact near_optimal remains the COP metric.
    """
    rows = rows.copy()
    rows["stopping_cdf_target"] = np.nan
    for _, g in rows.groupby("cycle_name", sort=False):
        near = g.loc[g.near_optimal, "candidate_defrost_time"]
        if near.empty:
            raise ValueError("No exact near-optimal candidate for CDF target")
        left, right = near.min(), near.max()
        duration = (right - left).total_seconds()
        target = (
            ((g.candidate_defrost_time - left).dt.total_seconds() / duration).clip(0, 1)
            if duration
            else g.candidate_defrost_time.ge(left).astype(float)
        )
        rows.loc[g.index, "stopping_cdf_target"] = target
    return rows


def stopping_candidates(
    rows, candidate_set="exact", late_preference="none", aggregation="marginal"
):
    """Training-only stopping support; keep the exact COP labels for evaluation."""
    if candidate_set not in ("exact", "filled") or late_preference not in ("none", "linear"):
        raise ValueError("Use exact/filled candidates and none/linear late preference")
    if aggregation not in ("marginal", "distribution"):
        raise ValueError("Use marginal likelihood or target-distribution cross entropy")
    rows = rows.copy()
    rows["stopping_candidate"] = rows.near_optimal.astype(bool)
    rows["stopping_log_weight"] = 0.0
    rows["stopping_distribution"] = aggregation == "distribution"
    for _, g in rows.groupby("cycle_name", sort=False):
        near = g.loc[g.near_optimal, "candidate_defrost_time"]
        if near.empty:
            raise ValueError("No exact near-optimal candidate")
        left, right = near.min(), near.max()
        if candidate_set == "filled":
            rows.loc[g.index, "stopping_candidate"] = g.candidate_defrost_time.between(left, right)
        duration = (right - left).total_seconds()
        if duration and late_preference == "linear":
            position = (g.candidate_defrost_time - left).dt.total_seconds() / duration
            with np.errstate(divide="ignore"):
                rows.loc[g.index, "stopping_log_weight"] = np.log(position.clip(0, 1))
    return rows


def label_actions(reference, action_domain="full", label_rule="midpoint", near_set_rule="exact"):
    """Historical longest-basin midpoint, or COP-only first-near-optimal action/event."""
    if near_set_rule not in ("exact", "connected"):
        raise ValueError("Unknown near-set rule")
    records, intervals = [], []
    for name, full in reference.groupby("cycle_name", sort=True):
        legal = full.on_action_grid.eq(True)
        if action_domain == "rb":
            legal &= full.candidate_defrost_time.le(full.t_RB)
        rows = full.loc[legal].sort_values("candidate_defrost_time").copy()
        rows["teacher_valid"] = rows.cycle_cop_eligible.eq(True) & np.isfinite(rows.cycle_cop)
        peak = rows.loc[rows.teacher_valid, "cycle_cop"].max()
        if near_set_rule == "connected":
            peak = full.loc[full.cycle_cop_eligible.eq(True), "cycle_cop"].max()
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError(f"No supported positive legal grid COP: {name}")
        near = rows.teacher_valid & rows.cycle_cop.ge(0.99 * peak)
        if near_set_rule == "connected":
            rows["near_optimal_exact"] = near
            left, right = rows.loc[near, "candidate_defrost_time"].iloc[[0, -1]]
            near = rows.candidate_defrost_time.between(left, right)
            rows["near_optimal_bridged"] = near & ~rows.near_optimal_exact
        starts = ~near.shift(fill_value=False) | rows.candidate_defrost_time.diff().ne(
            pd.Timedelta(seconds=30)
        )
        groups = rows.loc[near].groupby(starts.cumsum()[near], sort=True)
        first_stop = label_rule in ("stopping", "stopping_latch")
        basin = next(iter(groups))[1] if first_stop else max((g for _, g in groups), key=len)
        left, right = basin.candidate_defrost_time.iloc[[0, -1]]
        middle = (
            rows.loc[near, "candidate_defrost_time"].iloc[0]
            if first_stop
            else basin.candidate_defrost_time.iloc[(len(basin) - 1) // 2]
        )
        optimum_source = full if near_set_rule == "connected" else rows
        optimum = optimum_source.loc[
            optimum_source.cycle_cop_eligible.eq(True) & optimum_source.cycle_cop.eq(peak),
            "candidate_defrost_time",
        ].iloc[0]
        rows["optimal_cop"], rows["optimal_time"] = peak, optimum
        rows["after_optimum_target"] = (
            rows.candidate_defrost_time.ge(middle).astype(float).where(rows.teacher_valid)
        )
        rows["train_mask"] = rows.candidate_defrost_time.le(right) & rows.teacher_valid
        if label_rule == "stopping":
            # The earliest suffix-max 1% stopping point is also the earliest
            # supported global 1% point: until then the future includes the peak.
            rows["train_mask"] = rows.candidate_defrost_time.le(middle) & rows.teacher_valid
            rows["after_optimum_target"] = (
                rows.candidate_defrost_time.eq(middle).astype(float).where(rows.train_mask)
            )
        elif label_rule == "stopping_latch":
            rows["train_mask"] = rows.teacher_valid
        if near_set_rule == "connected":
            rows["train_mask"] |= rows.near_optimal_bridged
            rows["after_optimum_target"] = (
                rows.candidate_defrost_time.ge(middle)
                .astype(float)
                .where(rows.teacher_valid | rows.near_optimal_bridged)
            )
        rows["near_optimal"] = near
        rows["cycle_cop"] = rows.cycle_cop.where(rows.teacher_valid)
        intervals.append(
            dict(
                cycle_name=name,
                L=left,
                U=right,
                t_label=middle,
                optimal_time=optimum,
                optimal_cop=peak,
                basin_actions=len(basin),
            )
        )
        records.append(rows)
    return pd.concat(records, ignore_index=True), pd.DataFrame(intervals)
