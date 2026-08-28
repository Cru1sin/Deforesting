#!/usr/bin/env python3
# ruff: noqa: E501
"""Build cycle-safe RGB labels from empirical candidate regret without downloading images."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from frost_analysis.dataset.images import RGB_CAMERA_ORDER
from frost_analysis.dataset.loader import DatasetLoader
from frost_analysis.labels.cost import (
    assign_image_cost_states,
    complete_catalog_cycle_names,
    complete_observed_cycle_names,
    curve_label_exclusion_reason,
)

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
    "all": RGB_CAMERA_ORDER,
}


def _write_labels_and_provenance(
    labels: pd.DataFrame,
    output_root: Path,
    cost_source: Path,
    audit: list[dict[str, object]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    labels_path = output_root / "image_cost_labels.parquet"
    labels.to_parquet(labels_path, index=False)

    repository = Path(__file__).resolve().parents[2]
    included = [row["cycle_name"] for row in audit if row["included"]]
    excluded = [row["cycle_name"] for row in audit if not row["included"]]
    provenance = {
        "cost_source": str(cost_source),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "thresholds": list(THRESHOLDS),
        "cycles": {
            "included_count": len(included),
            "excluded_count": len(excluded),
            "included": included,
            "excluded": excluded,
            "reason_counts": dict(Counter(str(row["reason"]) for row in audit)),
            "labeled_image_count": sum(int(row["labeled_image_count"]) for row in audit),
            "records": audit,
        },
    }
    (output_root / "label_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _experiment_splits(experiments: list[str]) -> dict[str, str]:
    pattern = ("train", "train", "train", "validation", "test")
    return {name: pattern[index % len(pattern)] for index, name in enumerate(sorted(experiments))}


def build_labels(dataset_root: Path, cost_source: Path, output_root: Path) -> None:
    loader = DatasetLoader(dataset_root)
    metadata = loader.load_image_metadata()
    catalog = loader.list_cycles()
    curves = pd.read_csv(cost_source) if cost_source.suffix.lower() == ".csv" else pd.read_parquet(cost_source)
    complete_cycles = complete_catalog_cycle_names(catalog)
    valid_cycles = complete_observed_cycle_names(catalog, curves)
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
    current_curve_cycles = set(curves["cycle_name"].astype(str))
    valid_cycle_set = set(valid_cycles)
    audit: list[dict[str, object]] = [
        {
            "cycle_name": cycle_name,
            "included": False,
            "reason": (
                "no_current_curve"
                if cycle_name not in current_curve_cycles
                else "censored_curve"
            ),
            "labeled_image_count": 0,
        }
        for cycle_name in complete_cycles
        if cycle_name not in valid_cycle_set
    ]
    image_groups = {
        str(cycle_name): images
        for cycle_name, images in metadata.groupby("cycle_name", sort=True)
    }
    for cycle_name in valid_cycles:
        curve = curves.loc[curves["cycle_name"].eq(cycle_name)]
        images = image_groups.get(cycle_name)
        base = (
            None
            if images is None
            else assign_image_cost_states(
                images["image_time"], curve, regret_threshold=THRESHOLDS[0]
            )
        )
        reason = curve_label_exclusion_reason(curve)
        if reason is None:
            reason = (
                "no_interpolatable_image_times"
                if base is None or not base["relative_regret"].notna().any()
                else "labeled"
            )
        audit.append(
            {
                "cycle_name": str(cycle_name),
                "included": reason == "labeled",
                "reason": reason,
                "labeled_image_count": (
                    0 if base is None else int(base["relative_regret"].notna().sum())
                ),
            }
        )
        if base is None:
            continue
        result = images.reset_index(drop=True).copy()
        result["relative_regret"] = base["relative_regret"]
        for threshold in THRESHOLDS:
            states = assign_image_cost_states(
                images["image_time"], curve, regret_threshold=threshold
            )
            suffix = f"{int(threshold * 100):02d}pct"
            result[f"cost_state_{suffix}"] = states["three_class_state"]
            result[f"three_class_state_{suffix}"] = states["three_class_state"]
            result[f"binary_state_{suffix}"] = states["binary_state"]
        labeled.append(result)
    audit.sort(key=lambda row: str(row["cycle_name"]))
    if not labeled:
        raise RuntimeError("no supported RGB labels")
    labels = pd.concat(labeled, ignore_index=True)

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

    _write_labels_and_provenance(labels, output_root, cost_source, audit)
    balance.to_csv(output_root / "label_balance.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--cost-source",
        type=Path,
        default=Path("output/成本函数/cost_function_v1.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("output/label/cost_function_v1_binary"))
    args = parser.parse_args()
    build_labels(args.dataset, args.cost_source, args.output)


if __name__ == "__main__":
    main()
