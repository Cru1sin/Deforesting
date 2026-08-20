#!/usr/bin/env python3
"""Test whether low-regret visual states cluster more tightly than fixed-time states."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from frost_analysis.rgb_evaluation import (
    CAMERA_GROUPS,
    add_cycle_time_features,
    bootstrap_mean_interval,
)


def centroid_distances(
    frame: pd.DataFrame,
    *,
    regret_threshold: float,
    fixed_elapsed_minutes: float,
) -> pd.DataFrame:
    """Return paired per-cycle distances for optimum-neighbourhood and fixed-time states."""
    feature_columns = [column for column in frame if column.startswith("feature_")]
    values = frame.copy()
    values[feature_columns] = StandardScaler().fit_transform(values[feature_columns])
    centroids = []
    for cycle_name, cycle in values.groupby("cycle_name", sort=True):
        optimum = cycle.loc[cycle["relative_regret"].le(regret_threshold)]
        if optimum.empty or not (
            cycle["time_elapsed_minutes"].min()
            <= fixed_elapsed_minutes
            <= cycle["time_elapsed_minutes"].max()
        ):
            continue
        fixed = cycle.loc[
            (cycle["time_elapsed_minutes"] - fixed_elapsed_minutes)
            .abs()
            .sort_values(kind="stable")
            .index[: len(optimum)]
        ]
        for state, rows in (("optimum", optimum), ("fixed_time", fixed)):
            centroids.append(
                {
                    "cycle_name": cycle_name,
                    "experiment_id": str(cycle["experiment_id"].iloc[0]),
                    "state": state,
                    "image_count": len(rows),
                    **{column: rows[column].mean() for column in feature_columns},
                }
            )
    result = pd.DataFrame(centroids)
    if result.empty:
        return result
    for _, rows in result.groupby("state"):
        center = rows[feature_columns].mean().to_numpy()
        result.loc[rows.index, "centroid_distance"] = np.sqrt(
            np.mean((rows[feature_columns].to_numpy() - center) ** 2, axis=1)
        )
    return result


def analyze(
    shards: Path,
    candidates_path: Path,
    optima_path: Path,
    output: Path,
    regret_threshold: float = 0.01,
) -> None:
    paths = sorted(shards.glob("frost_cycle_*.parquet"))
    features = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    candidates = pd.read_parquet(candidates_path)
    features = add_cycle_time_features(features, candidates)
    optima = pd.read_csv(optima_path)
    primary = optima.loc[optima["cohort_tier"].eq("A_observed_policy")].copy()
    fixed_elapsed_minutes = float(primary["minutes_from_stable"].median() - 10.0)
    time_dispersion = float(
        primary["minutes_from_stable"].quantile(0.75)
        - primary["minutes_from_stable"].quantile(0.25)
    ) / float(primary["minutes_from_stable"].median())

    distance_rows = []
    summary_rows = []
    for group_name, roles in CAMERA_GROUPS.items():
        scoped = features.loc[features["camera_role"].isin(roles)]
        distances = centroid_distances(
            scoped,
            regret_threshold=regret_threshold,
            fixed_elapsed_minutes=fixed_elapsed_minutes,
        )
        if distances.empty:
            continue
        distances["camera_group"] = group_name
        distance_rows.append(distances)
        paired = distances.pivot(
            index=["cycle_name", "experiment_id"], columns="state", values="centroid_distance"
        ).dropna()
        paired["difference"] = paired["optimum"] - paired["fixed_time"]
        experiment_difference = paired["difference"].groupby("experiment_id").mean()
        summary_rows.append(
            {
                "camera_group": group_name,
                "regret_threshold": regret_threshold,
                "fixed_elapsed_minutes_from_candidate_start": fixed_elapsed_minutes,
                "time_optimum_iqr_over_median": time_dispersion,
                "cycle_count": len(paired),
                "experiment_count": paired.index.get_level_values("experiment_id").nunique(),
                **bootstrap_mean_interval(experiment_difference),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(distance_rows, ignore_index=True).to_csv(output / "cycle_distances.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, default=Path("outputs/RGB特征缓存/手工特征/cycles"))
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("report/02_经济除霜窗口/经验经济窗口/源数据/candidate_cost_curves.parquet"),
    )
    parser.add_argument(
        "--optima",
        type=Path,
        default=Path("report/02_经济除霜窗口/经验经济窗口/源数据/cycle_optimal_points.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("report/03_RGB标签与模型/视觉状态集中性")
    )
    args = parser.parse_args()
    analyze(args.shards, args.candidates, args.optima, args.output)


if __name__ == "__main__":
    main()
