#!/usr/bin/env python3
"""Stream one image cycle at a time into compact RGB feature shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from frost_analysis.dataset.images import RGB_CAMERA_ORDER, materialize_cycle_images
from frost_analysis.labels.cost import map_cost_state_targets
from frost_analysis.training.features import (
    DEEP_REPRESENTATIONS,
    add_deep_features,
    preferred_device,
)
from frost_analysis.training.smoke import DEFAULT_MAXIMUM_PER_GROUP, cycle_feature_shard

ROLE_ORDER = RGB_CAMERA_ORDER


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("output/label/cost_function_v1_binary/image_cost_labels.parquet"),
    )
    parser.add_argument("--state-column", default="cost_state_01pct")
    parser.add_argument("--task", choices=("binary", "three"), default="binary")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"))
    parser.add_argument("--cycles", nargs="+")
    parser.add_argument(
        "--maximum-per-group", type=int, default=DEFAULT_MAXIMUM_PER_GROUP
    )
    parser.add_argument(
        "--cost-states",
        nargs="+",
        choices=("pre_optimal", "near_optimal", "post_optimal"),
        default=None,
    )
    parser.add_argument("--fetch-cloud", action="store_true")
    parser.add_argument("--cleanup-downloaded", action="store_true")
    parser.add_argument("--cloud-root", type=Path)
    parser.add_argument("--minimum-free-gib", type=float, default=21)
    parser.add_argument(
        "--deep-representations",
        nargs="*",
        choices=DEEP_REPRESENTATIONS,
        default=[],
    )
    parser.add_argument("--deep-batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("output/test/model/RGB特征缓存/手工特征")
    )
    args = parser.parse_args()

    device = preferred_device()
    cost_states = args.cost_states or (
        ["pre_optimal", "post_optimal"]
        if args.task == "binary"
        else ["pre_optimal", "near_optimal", "post_optimal"]
    )
    if args.deep_representations:
        print(
            f"[deep] device={device} representations={','.join(args.deep_representations)}",
            flush=True,
        )

    labels = pd.read_parquet(args.labels)
    labels["cost_state"] = labels[args.state_column]
    labels = labels.loc[
        labels["relative_regret"].notna()
        & labels["cost_state"].isin(cost_states)
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
        if target.is_file() and not args.overwrite:
            print(f"[skip] {cycle_name}", flush=True)
            continue
        with materialize_cycle_images(
            args.dataset,
            str(cycle_name),
            fetch_cloud=args.fetch_cloud,
            cloud_root=args.cloud_root,
            cleanup_downloaded=args.cleanup_downloaded,
            minimum_free_gib=args.minimum_free_gib,
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
            if args.deep_representations:
                paths = [
                    cycle_dir / row.camera_role / row.file_name
                    for row in shard.itertuples(index=False)
                ]
                shard = add_deep_features(
                    shard,
                    paths,
                    args.deep_representations,
                    device=device,
                    batch_size=args.deep_batch_size,
                )
            shard["target"] = map_cost_state_targets(shard["cost_state"], args.task)
            shard = shard.loc[shard["target"].notna()].copy()
            shard["target"] = shard["target"].astype(int)
            shard.to_parquet(target, index=False)
            if not excluded.empty:
                excluded.to_csv(shard_root / f"{cycle_name}_excluded.csv", index=False)
            print(f"[saved] {cycle_name} rows={len(shard)}", flush=True)


if __name__ == "__main__":
    main()
