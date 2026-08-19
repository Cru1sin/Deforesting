#!/usr/bin/env python3
# ruff: noqa: E501
"""Build cycle-safe RGB labels from empirical candidate regret without downloading images."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from frost_analysis.dataset_loader import DatasetLoader
from frost_analysis.rgb_cost_labels import assign_image_cost_states

THRESHOLDS = (0.01, 0.02, 0.05, 0.10)
CAMERA_GROUPS = {
    "top": ("top",),
    "top_close": ("top_close",),
    "left": ("left",),
    "left_close": ("left_close",),
    "front": ("front",),
    "extreme": ("extreme",),
    "top_pair": ("top", "top_close"),
    "left_pair": ("left", "left_close"),
    "all": ("top", "top_close", "left", "left_close", "front", "extreme"),
}


def _experiment_splits(experiments: list[str]) -> dict[str, str]:
    pattern = ("train", "train", "train", "validation", "test")
    return {name: pattern[index % len(pattern)] for index, name in enumerate(sorted(experiments))}


def build_labels(dataset_root: Path, cost_root: Path, output_root: Path) -> None:
    loader = DatasetLoader(dataset_root)
    metadata = loader.load_image_metadata()
    catalog = loader.list_cycles()[["cycle_name", "experiment_id"]]
    curves = pd.read_parquet(cost_root / "candidate_cost_curves.parquet")
    valid_cycles = sorted(curves["cycle_name"].astype(str).unique())
    metadata = metadata.loc[
        metadata["cycle_name"].isin(valid_cycles) & metadata["cycle_stage"].eq("frost_development")
    ].merge(catalog, on="cycle_name", how="left", validate="many_to_one")
    split_map = _experiment_splits(metadata["experiment_id"].dropna().astype(str).unique().tolist())
    metadata["split"] = metadata["experiment_id"].map(split_map)
    metadata["image_path"] = (
        "images/"
        + metadata["cycle_name"].astype(str)
        + "/"
        + metadata["camera_role"].astype(str)
        + "/"
        + metadata["file_name"].astype(str)
    )
    metadata["local_available"] = metadata["image_path"].map(
        lambda value: (dataset_root / value).is_file()
    )

    labeled: list[pd.DataFrame] = []
    for cycle_name, images in metadata.groupby("cycle_name", sort=True):
        curve = curves.loc[curves["cycle_name"].eq(cycle_name)]
        base = assign_image_cost_states(images["image_time"], curve, regret_threshold=THRESHOLDS[0])
        result = images.reset_index(drop=True).copy()
        result["relative_regret"] = base["relative_regret"]
        for threshold in THRESHOLDS:
            state = assign_image_cost_states(
                images["image_time"], curve, regret_threshold=threshold
            )["cost_state"]
            result[f"cost_state_{int(threshold * 100):02d}pct"] = state
        labeled.append(result)
    labels = pd.concat(labeled, ignore_index=True)

    split_table = (
        labels[["experiment_id", "cycle_name", "split"]]
        .drop_duplicates()
        .sort_values(["split", "experiment_id", "cycle_name"])
    )
    balance_rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        state_column = f"cost_state_{int(threshold * 100):02d}pct"
        for group_name, roles in CAMERA_GROUPS.items():
            selected = labels.loc[labels["camera_role"].isin(roles)]
            counts = selected.groupby(["split", state_column], observed=True)
            for (split, state), rows in counts:
                balance_rows.append(
                    {
                        "regret_threshold": threshold,
                        "camera_group": group_name,
                        "split": split,
                        "cost_state": state,
                        "image_count": len(rows),
                        "cycle_count": rows["cycle_name"].nunique(),
                        "local_image_count": int(rows["local_available"].sum()),
                    }
                )
    balance = pd.DataFrame(balance_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output_root / "image_cost_labels.parquet", index=False)
    split_table.to_csv(output_root / "cycle_splits.csv", index=False)
    balance.to_csv(output_root / "label_balance.csv", index=False)
    eligible = labels.loc[labels["relative_regret"].notna()]
    one_percent = labels.loc[
        labels["relative_regret"].notna() & labels["cost_state_01pct"].eq("post_optimal")
    ]
    post_cycles = one_percent.groupby("split")["cycle_name"].nunique().to_dict()
    local = eligible.loc[eligible["local_available"]]
    local_post_cycles = (
        local.loc[local["cost_state_01pct"].eq("post_optimal")]
        .groupby("split")["cycle_name"]
        .nunique()
        .to_dict()
    )
    summary = f"""# RGB cost-label audit

This stage reads image metadata only; it does not download, alter or delete cloud images.

- Cost-valid cycles with frost-development images: {eligible["cycle_name"].nunique()}.
- Eligible image records across six camera roles: {len(eligible)}.
- Locally available eligible records: {int(eligible["local_available"].sum())}.
- Split unit: whole experiment (and therefore whole cycle), assigned deterministically in a 3 train : 1 validation : 1 test pattern over sorted experiments; no frame-level random split and no hash.
- Labels: pointwise interpolated relative regret. Images outside the candidate domain are excluded from model fitting. Images at or below each regret threshold are `near_optimal`; other images are `pre_optimal` or `post_optimal` relative to the earliest argmin.
- Audited thresholds: 1%, 2%, 5% and 10%. No final threshold is selected before held-out model and decision-regret evaluation.
- The 1% threshold has post-optimal images from {post_cycles.get("train", 0)}/{post_cycles.get("validation", 0)}/{post_cycles.get("test", 0)} train/validation/test cycles and is the primary model-demo candidate; 2% is retained as label sensitivity.
- A no-download local demo can use {len(local)} eligible images, but its post-optimal class spans only {local_post_cycles.get("train", 0)}/{local_post_cycles.get("validation", 0)}/{local_post_cycles.get("test", 0)} train/validation/test cycles. It is an engineering smoke test, not publication evidence.
"""
    (output_root / "README.md").write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--cost-source",
        type=Path,
        default=Path("report/raw_optimal_defrost/source_data"),
    )
    parser.add_argument("--output", type=Path, default=Path("report/rgb_cost_labels"))
    args = parser.parse_args()
    build_labels(args.dataset, args.cost_source, args.output)


if __name__ == "__main__":
    main()
