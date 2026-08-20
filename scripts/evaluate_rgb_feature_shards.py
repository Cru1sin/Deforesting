#!/usr/bin/env python3
"""Evaluate the locked RGB feature model by held-out experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from frost_analysis.rgb_evaluation import (
    CAMERA_GROUPS,
    MODEL_NAMES,
    REPRESENTATION_PREFIXES,
    REPRESENTATIONS,
    add_cycle_time_features,
    bootstrap_mean_interval,
    experiment_prediction_metrics,
    high_confidence_coverage,
    leave_one_experiment_out_predictions,
    retain_high_confidence_rows,
)


def score_rows(frame: pd.DataFrame) -> dict[str, float]:
    if frame["target"].nunique() < 2:
        return {"balanced_accuracy": float("nan"), "macro_f1": float("nan"), "auroc": float("nan")}
    return {
        "balanced_accuracy": balanced_accuracy_score(frame["target"], frame["predicted_target"]),
        "macro_f1": f1_score(frame["target"], frame["predicted_target"], average="macro"),
        "auroc": roc_auc_score(frame["target"], frame["decision_score"]),
    }


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, default=Path("outputs/RGB特征缓存/手工特征/cycles"))
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("report/02_经济除霜窗口/经验经济窗口/源数据/candidate_cost_curves.parquet"),
    )
    parser.add_argument(
        "--label-balance",
        type=Path,
        default=Path("report/03_RGB标签与模型/成本标签/label_balance.csv"),
    )
    parser.add_argument("--camera-groups", nargs="+", choices=tuple(CAMERA_GROUPS), default=["all"])
    parser.add_argument(
        "--regret-thresholds", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.10]
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["rbf_svm"])
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=["handcrafted"],
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=("rgb", "time", "rgb_time"),
        default=["rgb", "time", "rgb_time"],
    )
    parser.add_argument("--output", type=Path, default=Path("report/03_RGB标签与模型/全量模态比较"))
    args = parser.parse_args()
    if any(name != "handcrafted" for name in args.representations) and set(args.modalities) != {
        "rgb"
    }:
        raise SystemExit("deep representations currently support --modalities rgb only")

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
        training_protocol = (
            "pooled_views" if group_name in {"top_pair", "left_pair", "all"} else "single_view"
        )
        camera_rows = features.loc[features["camera_role"].isin(CAMERA_GROUPS[group_name])]
        for regret_threshold in args.regret_thresholds:
            scoped = retain_high_confidence_rows(camera_rows, regret_threshold)
            sample_retained_fraction = len(scoped) / len(camera_rows)
            eligible_image_coverage = high_confidence_coverage(
                label_balance, group_name, regret_threshold
            )
            for representation in args.representations:
                prefix = REPRESENTATION_PREFIXES[representation]
                rgb_columns = [column for column in scoped if column.startswith(prefix)]
                time_only = scoped.drop(columns=rgb_columns).copy()
                time_only["feature_000"] = scoped["time_elapsed_minutes"]
                time_only["feature_001"] = scoped["time_candidate_progress"]
                rgb_time = scoped.copy()
                rgb_time[f"{prefix}time_0"] = scoped["time_elapsed_minutes"]
                rgb_time[f"{prefix}time_1"] = scoped["time_candidate_progress"]
                modality_frames = {"rgb": scoped, "time": time_only, "rgb_time": rgb_time}
                for model_name in args.models:
                    for modality in args.modalities:
                        values = modality_frames[modality]
                        used_representation = (
                            "handcrafted" if modality == "time" else representation
                        )
                        predicted = leave_one_experiment_out_predictions(
                            values,
                            model_name=model_name,
                            representation=used_representation,
                        )
                        predicted["representation"] = representation
                        predicted["training_protocol"] = training_protocol
                        predicted["camera_group"] = group_name
                        predicted["modality"] = modality
                        predicted["regret_threshold"] = regret_threshold
                        predictions.append(predicted)
                        held_out = experiment_prediction_metrics(predicted)
                        held_out["representation"] = representation
                        held_out["model"] = model_name
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
                                    "representation": representation,
                                    "model": model_name,
                                    "camera_group": group_name,
                                    "training_protocol": training_protocol,
                                    "modality": modality,
                                    "regret_threshold": regret_threshold,
                                    "sample_retained_fraction": sample_retained_fraction,
                                    "eligible_image_coverage": eligible_image_coverage,
                                    "metric": metric,
                                    **interval,
                                    "experiment_count": len(held_out),
                                    "evaluable_experiment_count": int(
                                        held_out[metric].notna().sum()
                                    ),
                                }
                            )
                        metrics.append(
                            {
                                "representation": representation,
                                "model": model_name,
                                "camera_group": group_name,
                                "training_protocol": training_protocol,
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
                                    "representation": representation,
                                    "model": model_name,
                                    "camera_group": group_name,
                                    "training_protocol": training_protocol,
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
    for (
        representation,
        model_name,
        camera_group,
        regret_threshold,
        metric,
    ), values in held_out.melt(
        id_vars=[
            "experiment_id",
            "representation",
            "model",
            "camera_group",
            "modality",
            "regret_threshold",
        ],
        value_vars=["balanced_accuracy", "balanced_misclassification_regret"],
        var_name="metric",
    ).groupby(
        ["representation", "model", "camera_group", "regret_threshold", "metric"],
        sort=True,
    ):
        paired = values.pivot(index="experiment_id", columns="modality", values="value")
        for modality in ("rgb", "rgb_time"):
            if "time" not in paired or modality not in paired:
                continue
            differences = paired[modality] - paired["time"]
            interval = bootstrap_mean_interval(differences)
            comparisons.append(
                {
                    "representation": representation,
                    "model": model_name,
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
