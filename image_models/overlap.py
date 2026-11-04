"""Exact first-crossing intervals and coverage on explicitly supplied curves."""

import numpy as np
import pandas as pd


def vertex_grades(scores, vertex_positions):
    """Ordered grades from arc-length midpoints, preserving missing observations."""
    scores = np.asarray(scores, dtype=float)
    vertices = np.asarray(vertex_positions, dtype=float)
    if len(vertices) < 2 or not np.all(np.diff(vertices) > 0):
        raise ValueError("Vertex arc positions must be strictly increasing")
    boundaries = (vertices[:-1] + vertices[1:]) / 2
    return np.where(
        np.isfinite(scores), np.searchsorted(boundaries, scores, side="right") + 1, np.nan
    )


def observed_grade_intervals(curves):
    """Exact discrete 1% COP-near sets; never fill unobserved intermediate grades."""
    records = []
    for name, group in curves.groupby("cycle_name"):
        near = group.raw_effective_cop.ge(0.99 * group.raw_effective_cop.max())
        grades = np.sort(group.loc[near, "frost_grade"].dropna().unique())
        for block in np.split(grades, np.flatnonzero(np.diff(grades) > 1) + 1):
            if len(block):
                records.append(
                    dict(cycle_name=name, Ts=group.Ts.median(), left=block[0], right=block[-1])
                )
    return pd.DataFrame(records, columns=["cycle_name", "Ts", "left", "right"])


def crossing_intervals(curve, near_only=False):
    """First observed crossings (previous running maximum, right]; retain raw reversals.

    Missing adjacent scores are never bridged. The running maximum defines crossing
    membership only; it does not replace or repair the visual coordinate.
    """
    return crossing_intervals_values(
        curve.visual_frost_score.to_numpy(float),
        curve.near_optimal.to_numpy(bool) if near_only else None,
    )


def crossing_intervals_values(r, near=None):
    """Array path for the same first-crossing sets during boundary optimization."""
    r = np.asarray(r, dtype=float)
    history = np.maximum.accumulate(np.where(np.isfinite(r), r, -np.inf))
    intervals = []
    valid = np.isfinite(r[:-1]) & np.isfinite(r[1:]) & (r[1:] > history[:-1])
    if near is not None:
        valid &= np.asarray(near, dtype=bool)[1:]
    for k in np.flatnonzero(valid) + 1:
        left, right = history[k - 1], r[k]
        if intervals and left <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], right)
        else:
            intervals.append((left, right))
    return intervals


def interval_cycle_counts(intervals, points, *, closed=False):
    """One vote per cycle, even when several of its intervals overlap."""
    points = np.asarray(points)
    counts = np.zeros(len(points), dtype=int)
    for _, group in intervals.groupby("cycle_name"):
        member = np.zeros(len(points), dtype=bool)
        for row in group.itertuples():
            member |= ((points >= row.left) if closed else (points > row.left)) & (
                points <= row.right
            )
        counts += member
    return counts


def coverage_table(observed, feasible, *, common_cycles=None):
    """Conditional coverage, or fixed-cohort coverage only on identified common support."""
    if common_cycles is not None:
        common_cycles = set(common_cycles)
        if not common_cycles:
            raise ValueError("Common support requires a nonempty frozen cohort")
        observed = observed.loc[observed.cycle_name.isin(common_cycles)]
        feasible = feasible.loc[feasible.cycle_name.isin(common_cycles)]
    edges = np.unique(
        np.r_[
            0.0,
            1.0,
            observed[["left", "right"]].to_numpy().ravel(),
            feasible[["left", "right"]].to_numpy().ravel(),
        ]
    )
    result = pd.DataFrame({"left": edges[:-1], "right": edges[1:]})
    for name, source in [("n_observed", observed), ("n_near_optimal", feasible)]:
        result[name] = interval_cycle_counts(source, edges[1:])
    result["coverage"] = result.n_near_optimal / result.n_observed.replace(0, np.nan)
    if common_cycles is not None:
        # Intersect identified crossing intervals, not min–max envelopes across gaps.
        result["common_support"] = result.n_observed.eq(len(common_cycles))
        result["coverage"] = (result.n_near_optimal / len(common_cycles)).where(
            result.common_support
        )
    return result


def action_intervals(actions):
    """Observed and feasible first-crossing intervals without bridging reference gaps."""
    records = []
    for name, curve in actions.groupby("cycle_name", sort=True):
        curve = curve.sort_values("candidate_defrost_time")
        for kind, near in [("observed", False), ("feasible", True)]:
            records.extend(
                dict(cycle_name=name, kind=kind, left=a, right=b)
                for a, b in crossing_intervals(curve, near)
            )
    return pd.DataFrame(records, columns=["cycle_name", "kind", "left", "right"])


def maximum_intervals(table):
    """Merge adjacent maximum atoms; retain every disconnected tied component."""
    peaks = []
    for row in table.loc[table.coverage.eq(table.coverage.max())].itertuples():
        if peaks and row.left == peaks[-1]["right"]:
            peaks[-1].update(
                right=row.right,
                n_observed_min=min(peaks[-1]["n_observed_min"], row.n_observed),
                n_observed_max=max(peaks[-1]["n_observed_max"], row.n_observed),
            )
        else:
            peaks.append(
                dict(
                    left=row.left,
                    right=row.right,
                    coverage=row.coverage,
                    n_observed_min=row.n_observed,
                    n_observed_max=row.n_observed,
                )
            )
    return peaks


def teacher_crossing(curve, rho):
    """First observed crossing; never skip unsupported actions to improve regret."""
    curve = curve.sort_values("candidate_defrost_time").reset_index(drop=True)
    r = curve.visual_frost_score.to_numpy(float)
    indices = np.flatnonzero(np.isfinite(r) & (r >= rho))
    result = dict(
        crossing_time=pd.NaT, no_crossing=not len(indices), hit_1pct=np.nan, cop_regret=np.nan
    )
    if not len(indices):
        return dict(result, status="no_crossing")
    k = indices[0]
    result["crossing_time"] = curve.candidate_defrost_time.iloc[k]
    if k == 0 or not np.isfinite(r[k - 1]):
        return dict(result, status="left_censored" if k == 0 else "reference_gap")
    norm = curve.cop_norm.iloc[k]
    if not np.isfinite(norm):
        return dict(result, hit_1pct=False, status="unsupported_cop")
    return dict(result, hit_1pct=bool(norm >= 0.99), cop_regret=1 - norm, status="evaluated")


def normalized_cop_by_grade(curves, grades, *, mapping="first-crossing"):
    """Own-maximum COP with explicit grade mapping and no bridging of temporal gaps."""
    if mapping in ["observed-grade", "interpolated-grade"]:
        values = curves.assign(
            normalized_cop=curves.raw_effective_cop
            / curves.groupby("cycle_name").raw_effective_cop.transform("max")
        )
        observed = (
            values.dropna(subset=["frost_grade", "normalized_cop"])
            .groupby(["cycle_name", "frost_grade"])
            .normalized_cop.agg(normalized_cop="mean", n_actions="count")
            .reset_index()
            .rename(columns={"frost_grade": "grade"})
        )
        if mapping == "observed-grade":
            return observed
        records = []
        grades = np.asarray(grades)
        for name, g in values.groupby("cycle_name", sort=True):
            g = g.sort_values("candidate_defrost_time")
            measured = set(observed.loc[observed.cycle_name.eq(name), "grade"])
            for left, right in zip(g.itertuples(), g.iloc[1:].itertuples()):
                if not np.isfinite(
                    [left.frost_grade, right.frost_grade, left.normalized_cop, right.normalized_cop]
                ).all():
                    continue
                # Existing actions are at most 30 s apart; native RGB sampling is 30 s (15 s jitter allowance).
                dt = (right.candidate_defrost_time - left.candidate_defrost_time).total_seconds()
                image_dt = (
                    pd.Timestamp(right.image_time) - pd.Timestamp(left.image_time)
                ).total_seconds()
                if not (0 < dt <= 30 and 0 <= image_dt <= 45):
                    continue
                low, high = sorted([left.frost_grade, right.frost_grade])
                for grade in grades[(grades > low) & (grades < high)]:
                    if grade in measured:
                        continue
                    fraction = (grade - left.frost_grade) / (right.frost_grade - left.frost_grade)
                    records.append(
                        dict(
                            cycle_name=name,
                            grade=grade,
                            normalized_cop=left.normalized_cop
                            + fraction * (right.normalized_cop - left.normalized_cop),
                        )
                    )
        interpolated = pd.DataFrame(records, columns=["cycle_name", "grade", "normalized_cop"])
        interpolated = (
            interpolated.groupby(["cycle_name", "grade"])
            .normalized_cop.agg(normalized_cop="mean", n_interpolated_segments="count")
            .reset_index()
            .assign(n_actions=0)
        )
        observed = observed.assign(n_interpolated_segments=0)
        return pd.concat([observed, interpolated], ignore_index=True).sort_values(
            ["cycle_name", "grade"]
        )
    records = []
    for name, curve in curves.groupby("cycle_name", sort=True):
        curve = curve.assign(
            cop_norm=curve.raw_effective_cop / curve.raw_effective_cop.max(),
            visual_frost_score=curve.frost_grade,
        )
        for grade in grades:
            crossing = teacher_crossing(curve, grade)
            records.append(
                dict(
                    cycle_name=name,
                    grade=grade,
                    crossing_time=crossing["crossing_time"],
                    status=crossing["status"],
                    normalized_cop=1 - crossing["cop_regret"],
                )
            )
    return pd.DataFrame(records)


def reference_basin_intervals(curves, *, objectives=None, coordinates=None, near_column=None):
    """Closed reference ranges of adjacent observed 1% points; keep gaps separate."""
    records = []
    for name, g in curves.groupby("cycle_name"):
        g = g.sort_values("candidate_defrost_time").reset_index(drop=True)
        for method, column in (
            objectives
            if objectives is not None
            else [
                ("effective_cop", "raw_effective_cop"),
                ("frontier_ratio", "relative_effective_cop"),
            ]
        ):
            near = (
                g[near_column].eq(True)
                if near_column is not None
                else g[column].ge(0.99 * g[column].max())
            )
            for coordinate in (
                coordinates
                if coordinates is not None
                else ["visual_frost_score", "slingshot_grade"]
            ):
                valid = near & g[coordinate].notna()
                starts = ~valid.shift(fill_value=False) | g.candidate_defrost_time.diff().gt(
                    pd.Timedelta(seconds=30)
                )
                for _, part in g.loc[valid].groupby(starts.cumsum()[valid]):
                    records.append(
                        dict(
                            cycle_name=name,
                            Ts=g.Ts.iloc[0],
                            method=method,
                            coordinate=coordinate,
                            left=part[coordinate].min(),
                            right=part[coordinate].max(),
                            start=part.candidate_defrost_time.iloc[0],
                            end=part.candidate_defrost_time.iloc[-1],
                            actions=len(part),
                        )
                    )
    return pd.DataFrame(records)
