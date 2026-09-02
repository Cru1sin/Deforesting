#!/usr/bin/env python3
"""Produce every RGB asset for selected cycles in one image transaction."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from dataloader.images import RGB_CAMERA_ORDER, materialize_cycle_images
from dataloader.loader import DatasetLoader
from dataloader.metadata import read_catalog
from dataloader.operations import render_dataset
from frost_analysis.labels.assets import build_optimal_view_manifest
from frost_analysis.labels.cost import map_cost_state_targets
from frost_analysis.training.features import (
    DEEP_REPRESENTATIONS,
    add_deep_features,
    preferred_device,
)
from frost_analysis.training.smoke import DEFAULT_MAXIMUM_PER_GROUP, cycle_feature_shard

ROLE_ORDER = tuple(RGB_CAMERA_ORDER)


def write_task_shards(shard: pd.DataFrame, output: Path, cycle_name: str) -> dict[str, Path]:
    """Write binary and three-class views of one shared feature matrix."""
    paths = {}
    for task in ("binary", "three"):
        values = shard.copy()
        values["target"] = map_cost_state_targets(values["cost_state"], task)
        values = values.loc[values["target"].notna()].copy()
        values["target"] = values["target"].astype(int)
        target = output / "features" / task / "cycles" / f"{cycle_name}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        values.to_parquet(target, index=False)
        paths[task] = target
    return paths


def copy_optimal_views(
    cycle_dir: Path, manifest: pd.DataFrame, output: Path
) -> Path:
    """Copy available nearest t-star views and retain all six audit rows."""
    expected = set(ROLE_ORDER)
    if len(manifest) != len(ROLE_ORDER) or set(manifest["camera_role"]) != expected:
        raise ValueError("optimal view export requires exactly six roles")
    available = manifest.loc[manifest["available"]]
    manifest = manifest.copy()
    manifest["exported"] = False
    for index, row in available.iterrows():
        source = cycle_dir / str(row["camera_role"]) / str(row["file_name"])
        target = output / str(row["relative_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        with Image.open(target) as image:
            image.verify()
        manifest.loc[index, "exported"] = True
    cycle_name = str(manifest["cycle_name"].iloc[0])
    target = output / cycle_name / "optimal_rgb_views_manifest.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(target, index=False)
    pd.read_csv(target)
    return target


def _verify_panel(dataset: Path, record: dict[str, object]) -> Path:
    assets = record.get("assets")
    if not isinstance(assets, dict) or "rgb_panel" not in assets:
        raise ValueError(f"cycle RGB panel asset is missing: {record.get('cycle_name')}")
    path = dataset / str(assets["rgb_panel"])
    with Image.open(path) as image:
        image.verify()
    return path


def process_cycle_assets(  # noqa: C901
    dataset: Path,
    labels_path: Path,
    curves_path: Path,
    output: Path,
    cycles: list[str],
    *,
    backbones: list[str],
    maximum_per_group: int = DEFAULT_MAXIMUM_PER_GROUP,
    batch_size: int = 32,
    fetch_cloud: bool = False,
    cloud_root: Path | None = None,
    cleanup_downloaded: bool = False,
    minimum_free_gib: float = 21,
) -> None:
    """Process each cycle completely before its optional downloaded-image cleanup."""
    loader = DatasetLoader(dataset)
    metadata = loader.load_image_metadata()
    labels = pd.read_parquet(labels_path)
    curves = pd.read_parquet(curves_path)
    records = {
        str(record["cycle_name"]): record for record in read_catalog(dataset)["cycles"]
    }
    device = preferred_device()
    for cycle_name in cycles:
        if cycle_name not in records:
            raise KeyError(f"unknown cycle: {cycle_name}")
        cycle_labels = labels.loc[labels["cycle_name"].astype(str).eq(cycle_name)].copy()
        cycle_labels["cost_state"] = cycle_labels["three_class_state_01pct"]
        cycle_labels = cycle_labels.loc[
            cycle_labels["relative_regret"].notna()
            & cycle_labels["cost_state"].isin(
                ("pre_optimal", "near_optimal", "post_optimal")
            )
            & cycle_labels["camera_role"].isin(ROLE_ORDER)
        ]
        if cycle_labels.empty:
            raise ValueError(f"cycle has no labeled RGB rows: {cycle_name}")
        with materialize_cycle_images(
            dataset,
            cycle_name,
            fetch_cloud=fetch_cloud,
            cloud_root=cloud_root,
            cleanup_downloaded=cleanup_downloaded,
            minimum_free_gib=minimum_free_gib,
        ) as cycle_dir:
            if not cycle_dir.is_dir():
                raise FileNotFoundError(cycle_dir)
            render_dataset(dataset, cycle_name, publication=False, panel=True)
            _verify_panel(dataset, records[cycle_name])
            optimal = build_optimal_view_manifest(
                curves.loc[curves["cycle_name"].astype(str).eq(cycle_name)],
                metadata.loc[metadata["cycle_name"].astype(str).eq(cycle_name)],
            )
            copy_optimal_views(cycle_dir, optimal, output / "optimal")
            shard, excluded = cycle_feature_shard(
                cycle_labels,
                cycle_dir,
                ROLE_ORDER,
                maximum_per_group=maximum_per_group,
            )
            if shard.empty:
                raise ValueError(f"cycle produced no readable RGB features: {cycle_name}")
            image_paths = [
                cycle_dir / row.camera_role / row.file_name
                for row in shard.itertuples(index=False)
            ]
            shard = add_deep_features(
                shard,
                image_paths,
                backbones,
                device=device,
                batch_size=batch_size,
            )
            outputs = write_task_shards(shard, output, cycle_name)
            if not excluded.empty:
                excluded_path = output / "features" / "excluded" / f"{cycle_name}.csv"
                excluded_path.parent.mkdir(parents=True, exist_ok=True)
                excluded.to_csv(excluded_path, index=False)
                pd.read_csv(excluded_path)
            for task, path in outputs.items():
                values = pd.read_parquet(path)
                if values.empty or values["target"].isna().any():
                    raise ValueError(f"invalid {task} shard: {cycle_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("output/label/cost_function_v1_binary/image_cost_labels.parquet"),
    )
    parser.add_argument(
        "--curves",
        type=Path,
        default=Path(
            "output/test/成本函数/其他/经验经济窗口/源数据/candidate_cost_curves.parquet"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("output/test/model/RGB循环事务"))
    parser.add_argument("--cycles", nargs="+", required=True)
    parser.add_argument(
        "--backbones",
        nargs="*",
        choices=DEEP_REPRESENTATIONS,
        default=DEEP_REPRESENTATIONS,
        help="all 7 pretrained representations by default; bare --backbones disables them",
    )
    parser.add_argument(
        "--maximum-per-group", type=int, default=DEFAULT_MAXIMUM_PER_GROUP
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fetch-cloud", action="store_true")
    parser.add_argument("--cloud-root", type=Path)
    parser.add_argument("--cleanup-downloaded", action="store_true")
    parser.add_argument("--minimum-free-gib", type=float, default=21)
    args = parser.parse_args()
    process_cycle_assets(
        args.dataset,
        args.labels,
        args.curves,
        args.output,
        args.cycles,
        backbones=args.backbones,
        maximum_per_group=args.maximum_per_group,
        batch_size=args.batch_size,
        fetch_cloud=args.fetch_cloud,
        cloud_root=args.cloud_root,
        cleanup_downloaded=args.cleanup_downloaded,
        minimum_free_gib=args.minimum_free_gib,
    )


if __name__ == "__main__":
    main()
