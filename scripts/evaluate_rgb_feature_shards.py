#!/usr/bin/env python3
"""Evaluate the locked RGB feature model by held-out experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from frost_analysis.rgb_evaluation import (
    add_cycle_time_features,
    bootstrap_mean_interval,
    experiment_prediction_metrics,
    high_confidence_coverage,
    leave_one_experiment_out_predictions,
    retain_high_confidence_rows,
)

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
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("report/raw_optimal_defrost/source_data/candidate_cost_curves.parquet"),
    )
    parser.add_argument(
        "--label-balance",
        type=Path,
        default=Path("report/rgb_cost_labels/label_balance.csv"),
    )
    parser.add_argument("--camera-groups", nargs="+", choices=tuple(CAMERA_GROUPS), default=["all"])
    parser.add_argument(
        "--regret-thresholds", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.10]
    )
    parser.add_argument("--output", type=Path, default=Path("report/rgb_full_cohort"))
    args = parser.parse_args()

    paths = sorted(args.shards.glob("frost_cycle_*.parquet"))
    if not paths:
        raise SystemExit("no feature shards")
    features = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    features = add_cycle_time_features(features, pd.read_parquet(args.candidates))
    label_balance = pd.read_csv(args.label_balance)
    predictions = []
    metrics = []
    experiment_metrics = []
    summary_metrics = []
    for group_name in args.camera_groups:
        camera_rows = features.loc[features["camera_role"].isin(CAMERA_GROUPS[group_name])]
        for regret_threshold in args.regret_thresholds:
            scoped = retain_high_confidence_rows(camera_rows, regret_threshold)
            sample_retained_fraction = len(scoped) / len(camera_rows)
            eligible_image_coverage = high_confidence_coverage(
                label_balance, group_name, regret_threshold
            )
            rgb_columns = [column for column in scoped if column.startswith("feature_")]
            time_only = scoped.drop(columns=rgb_columns).copy()
            time_only["feature_000"] = scoped["time_elapsed_minutes"]
            time_only["feature_001"] = scoped["time_candidate_progress"]
            rgb_time = scoped.copy()
            rgb_time["feature_040"] = scoped["time_elapsed_minutes"]
            rgb_time["feature_041"] = scoped["time_candidate_progress"]
            for modality, values in (("rgb", scoped), ("time", time_only), ("rgb_time", rgb_time)):
                predicted = leave_one_experiment_out_predictions(values)
                predicted["camera_group"] = group_name
                predicted["modality"] = modality
                predicted["regret_threshold"] = regret_threshold
                predictions.append(predicted)
                held_out = experiment_prediction_metrics(predicted)
                held_out["camera_group"] = group_name
                held_out["modality"] = modality
                held_out["regret_threshold"] = regret_threshold
                experiment_metrics.append(held_out)
                for metric in (
                    "balanced_accuracy",
                    "macro_f1",
                    "auroc",
                    "balanced_misclassification_regret",
                ):
                    interval = bootstrap_mean_interval(held_out[metric])
                    summary_metrics.append(
                        {
                            "camera_group": group_name,
                            "modality": modality,
                            "regret_threshold": regret_threshold,
                            "sample_retained_fraction": sample_retained_fraction,
                            "eligible_image_coverage": eligible_image_coverage,
                            "metric": metric,
                            **interval,
                            "experiment_count": len(held_out),
                            "evaluable_experiment_count": int(held_out[metric].notna().sum()),
                        }
                    )
                metrics.append(
                    {
                        "camera_group": group_name,
                        "modality": modality,
                        "regret_threshold": regret_threshold,
                        "sample_retained_fraction": sample_retained_fraction,
                        "eligible_image_coverage": eligible_image_coverage,
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
                            "modality": modality,
                            "regret_threshold": regret_threshold,
                            "sample_retained_fraction": sample_retained_fraction,
                            "eligible_image_coverage": eligible_image_coverage,
                            "scope": str(experiment),
                            **score_rows(rows),
                            "image_count": len(rows),
                            "cycle_count": rows["cycle_name"].nunique(),
                            "experiment_count": 1,
                        }
                    )

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.output / "metrics.csv", index=False)
    held_out = pd.concat(experiment_metrics, ignore_index=True)
    held_out.to_csv(args.output / "experiment_metrics.csv", index=False)
    pd.DataFrame(summary_metrics).to_csv(args.output / "summary_metrics.csv", index=False)

    comparisons = []
    for (camera_group, regret_threshold, metric), values in held_out.melt(
        id_vars=["experiment_id", "camera_group", "modality", "regret_threshold"],
        value_vars=["balanced_accuracy", "balanced_misclassification_regret"],
        var_name="metric",
    ).groupby(["camera_group", "regret_threshold", "metric"], sort=True):
        paired = values.pivot(index="experiment_id", columns="modality", values="value")
        for modality in ("rgb", "rgb_time"):
            differences = paired[modality] - paired["time"]
            interval = bootstrap_mean_interval(differences)
            comparisons.append(
                {
                    "camera_group": camera_group,
                    "regret_threshold": regret_threshold,
                    "comparison": f"{modality}_minus_time",
                    "metric": metric,
                    **interval,
                    "experiment_count": len(paired),
                    "evaluable_experiment_count": int(differences.notna().sum()),
                }
            )
    pd.DataFrame(comparisons).to_csv(args.output / "modality_deltas.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        args.output / "predictions.parquet", index=False
    )


if __name__ == "__main__":
    main()
