#!/usr/bin/env python3
"""Stream one image cycle at a time into compact RGB feature shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from frost_analysis.dataset_images import materialize_cycle_images
from frost_analysis.rgb_smoke import cycle_feature_shard

ROLE_ORDER = ("top", "top_close", "left", "left_close", "front", "extreme")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("report/rgb_cost_labels/image_cost_labels.parquet"),
    )
    parser.add_argument("--state-column", default="cost_state_01pct")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"))
    parser.add_argument("--cycles", nargs="+")
    parser.add_argument("--maximum-per-group", type=int, default=12)
    parser.add_argument(
        "--cost-states",
        nargs="+",
        choices=("pre_optimal", "near_optimal", "post_optimal"),
        default=["pre_optimal", "post_optimal"],
    )
    parser.add_argument("--fetch-cloud", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("report/rgb_feature_shards"))
    args = parser.parse_args()

    labels = pd.read_parquet(args.labels)
    labels["cost_state"] = labels[args.state_column]
    labels = labels.loc[
        labels["relative_regret"].notna()
        & labels["cost_state"].isin(args.cost_states)
        & labels["camera_role"].isin(ROLE_ORDER)
    ].copy()
    if args.splits:
        labels = labels.loc[labels["split"].isin(args.splits)]
    if args.cycles:
        labels = labels.loc[labels["cycle_name"].isin(args.cycles)]

    shard_root = args.output / "cycles"
    shard_root.mkdir(parents=True, exist_ok=True)
    for cycle_name, rows in labels.groupby("cycle_name", sort=True):
        target = shard_root / f"{cycle_name}.parquet"
        if target.is_file():
            print(f"[skip] {cycle_name}", flush=True)
            continue
        with materialize_cycle_images(
            args.dataset,
            str(cycle_name),
            fetch_cloud=args.fetch_cloud,
        ) as cycle_dir:
            if not cycle_dir.is_dir():
                print(f"[missing] {cycle_name}", flush=True)
                continue
            print(f"[extract] {cycle_name}", flush=True)
            shard, excluded = cycle_feature_shard(
                rows,
                cycle_dir,
                ROLE_ORDER,
                maximum_per_group=args.maximum_per_group,
            )
            shard["target"] = shard["cost_state"].map(
                {"pre_optimal": 0, "post_optimal": 1}
            )
            shard.to_parquet(target, index=False)
            if not excluded.empty:
                excluded.to_csv(shard_root / f"{cycle_name}_excluded.csv", index=False)
            print(f"[saved] {cycle_name} rows={len(shard)}", flush=True)


if __name__ == "__main__":
    main()
