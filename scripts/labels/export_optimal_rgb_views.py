#!/usr/bin/env python3
"""Export one nearest image per camera at each complete cycle's inverse-COP minimum."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from dataloader.dataloader import DatasetLoader
from dataloader.images import materialize_cycle_images
from frost_analysis.labels.assets import build_optimal_view_manifest, optimal_view_report
from frost_analysis.labels.cost import complete_observed_cycle_names


def export_optimal_views(
    dataset: Path,
    curves: pd.DataFrame,
    output: Path,
    *,
    fetch_cloud: bool = False,
    cloud_root: Path | None = None,
) -> pd.DataFrame:
    """Copy selected views to the report and write a long-form manifest."""
    loader = DatasetLoader(dataset)
    complete = complete_observed_cycle_names(loader.list_cycles(), curves)
    scoped = curves.loc[curves["cycle_name"].isin(complete)]
    manifest = build_optimal_view_manifest(scoped, loader.load_image_metadata())
    manifest["exported"] = False
    for cycle_name, group in manifest.groupby("cycle_name", sort=True):
        with materialize_cycle_images(
            dataset,
            str(cycle_name),
            fetch_cloud=fetch_cloud,
            cloud_root=cloud_root,
        ) as cycle_dir:
            for index, row in group.loc[group["available"]].iterrows():
                source = cycle_dir / str(row["camera_role"]) / str(row["file_name"])
                if not source.is_file():
                    continue
                target = output / str(row["relative_path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                manifest.loc[index, "exported"] = True
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "optimal_rgb_views_manifest.csv", index=False)
    (output / "报告.md").write_text(optimal_view_report(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--curves",
        type=Path,
        default=Path("output/test/成本函数/其他/经验经济窗口/源数据/candidate_cost_curves.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/label/历史最优点六机位"),
    )
    parser.add_argument("--fetch-cloud", action="store_true")
    parser.add_argument("--cloud-root", type=Path)
    args = parser.parse_args()
    export_optimal_views(
        args.dataset,
        pd.read_parquet(args.curves),
        args.output,
        fetch_cloud=args.fetch_cloud,
        cloud_root=args.cloud_root,
    )


if __name__ == "__main__":
    main()
