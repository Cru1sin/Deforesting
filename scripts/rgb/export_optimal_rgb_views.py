#!/usr/bin/env python3
"""Export one nearest image per camera at each complete cycle's inverse-COP minimum."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from frost_analysis.dataset_images import materialize_cycle_images
from frost_analysis.dataset_loader import DatasetLoader
from frost_analysis.rgb_cost_labels import complete_observed_cycle_names
from frost_analysis.visualization import RGB_CAMERA_ORDER, RGB_PANEL_MAX_OFFSET


def build_optimal_view_manifest(curves: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Return one long-form row per cycle and expected camera role."""
    rows: list[dict[str, object]] = []
    for cycle_name, curve in curves.groupby("cycle_name", sort=True):
        ordered = curve.assign(
            candidate_time=pd.to_datetime(curve["candidate_time"], errors="coerce"),
            inverse_cop=pd.to_numeric(curve["inverse_cop"], errors="coerce"),
        ).dropna(subset=["candidate_time", "inverse_cop"])
        optimum = ordered.sort_values("candidate_time", kind="stable").loc[
            lambda values: values["inverse_cop"].eq(values["inverse_cop"].min())
        ].iloc[0]
        target = pd.Timestamp(optimum["candidate_time"])
        scoped = metadata.loc[metadata["cycle_name"].astype(str).eq(str(cycle_name))].copy()
        scoped["image_time"] = pd.to_datetime(scoped["image_time"], errors="coerce")
        for role in RGB_CAMERA_ORDER:
            camera = scoped.loc[scoped["camera_role"].astype(str).eq(role)].dropna(
                subset=["image_time"]
            )
            selected = None
            offset = pd.NaT
            if not camera.empty:
                offsets = (camera["image_time"] - target).abs()
                nearest = offsets.idxmin()
                if offsets.loc[nearest] <= RGB_PANEL_MAX_OFFSET:
                    selected = camera.loc[nearest]
                    offset = offsets.loc[nearest]
            file_name = str(selected["file_name"]) if selected is not None else ""
            rows.append(
                {
                    "cycle_name": str(cycle_name),
                    "camera_role": role,
                    "target_time": target,
                    "inverse_cop": float(optimum["inverse_cop"]),
                    "cycle_cop": float(optimum.get("cycle_cop", 1 / optimum["inverse_cop"])),
                    "relative_regret": float(
                        optimum["inverse_cop"] / ordered["inverse_cop"].min() - 1
                    ),
                    "image_time": selected["image_time"] if selected is not None else pd.NaT,
                    "offset_seconds": (
                        float(offset.total_seconds()) if selected is not None else float("nan")
                    ),
                    "file_name": file_name,
                    "available": selected is not None,
                    "relative_path": (
                        f"{cycle_name}/{role}/{file_name}" if selected is not None else ""
                    ),
                    "source_relative_path": (
                        f"images/{cycle_name}/{role}/{file_name}" if selected is not None else ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def optimal_view_report(manifest: pd.DataFrame) -> str:
    """Summarize six-view selection and export completeness."""
    selected = int(manifest["available"].sum())
    exported = int(manifest.get("exported", pd.Series(False, index=manifest.index)).sum())
    return f"""# 最优除霜点六机位导出

- 完整观测循环：{manifest["cycle_name"].nunique()}。
- 目标机位图像：{selected}/{len(manifest)}（最近图像须位于最优点前后 2 min 内）。
- 已导出图像：{exported}/{selected}。
- 每行清单对应一个循环—机位组合；未找到合格近邻图像时仍保留该行，便于审计缺失。
"""


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
        default=Path("report/02_经济除霜窗口/经验经济窗口/源数据/candidate_cost_curves.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report/03_RGB标签与模型/最优点六机位"),
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
