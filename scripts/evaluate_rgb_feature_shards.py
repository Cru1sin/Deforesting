#!/usr/bin/env python3
"""Evaluate the locked RGB feature model by held-out experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from frost_analysis.rgb_evaluation import leave_one_experiment_out_predictions

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


def score_rows(frame: pd.DataFrame) -> dict[str, float]:
    if frame["target"].nunique() < 2:
        return {"balanced_accuracy": float("nan"), "macro_f1": float("nan"), "auroc": float("nan")}
    return {
        "balanced_accuracy": balanced_accuracy_score(frame["target"], frame["predicted_target"]),
        "macro_f1": f1_score(frame["target"], frame["predicted_target"], average="macro"),
        "auroc": roc_auc_score(frame["target"], frame["decision_score"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, default=Path("report/rgb_feature_shards/cycles"))
    parser.add_argument("--camera-groups", nargs="+", choices=tuple(CAMERA_GROUPS), default=["all"])
    parser.add_argument("--output", type=Path, default=Path("report/rgb_full_cohort"))
    args = parser.parse_args()

    paths = sorted(args.shards.glob("frost_cycle_*.parquet"))
    if not paths:
        raise SystemExit("no feature shards")
    features = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    predictions = []
    metrics = []
    for group_name in args.camera_groups:
        scoped = features.loc[features["camera_role"].isin(CAMERA_GROUPS[group_name])]
        predicted = leave_one_experiment_out_predictions(scoped)
        predicted["camera_group"] = group_name
        predictions.append(predicted)
        metrics.append(
            {
                "camera_group": group_name,
                "scope": "all_held_out_predictions",
                **score_rows(predicted),
                "image_count": len(predicted),
                "cycle_count": predicted["cycle_name"].nunique(),
                "experiment_count": predicted["experiment_id"].nunique(),
            }
        )
        for experiment, rows in predicted.groupby("experiment_id", sort=True):
            metrics.append(
                {
                    "camera_group": group_name,
                    "scope": str(experiment),
                    **score_rows(rows),
                    "image_count": len(rows),
                    "cycle_count": rows["cycle_name"].nunique(),
                    "experiment_count": 1,
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.output / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        args.output / "predictions.parquet", index=False
    )


if __name__ == "__main__":
    main()
