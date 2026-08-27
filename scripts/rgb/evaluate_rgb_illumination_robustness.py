#!/usr/bin/env python3
"""Stress-test frozen RGB representations under deterministic light shifts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from frost_analysis.rgb_deep_features import (
    DEEP_REPRESENTATIONS,
    add_embedding_columns,
    cosine_similarity_rows,
    extract_representation_matrices,
    illumination_transform,
    load_frozen_extractor,
    preferred_device,
)
from frost_analysis.rgb_evaluation import (
    CAMERA_GROUPS,
    bootstrap_mean_interval,
    make_rgb_model,
    representation_columns,
)
from frost_analysis.rgb_smoke import even_sample_groups

CONDITIONS = (
    "native",
    "dark_60pct",
    "gamma_1p8",
    "gamma_2p2",
    "gamma_2p2_sensor_noise",
    "gamma_1p8_vignette",
)
AUGMENTATION_CONDITIONS = ("gamma_1p8", "gamma_2p2_sensor_noise", "gamma_1p8_vignette")
TRAINING_STRATEGIES = ("native_only", "low_light_augmented")


def evaluate_conditions(
    cohort: pd.DataFrame,
    condition_frames: dict[str, pd.DataFrame],
    camera_groups: list[str],
    representations: list[str] | tuple[str, ...] = DEEP_REPRESENTATIONS,
) -> pd.DataFrame:
    """Fit native embeddings and score each held-out experiment under light shifts."""
    rows = []
    for representation in representations:
        columns = representation_columns(cohort, representation)
        if not columns:
            continue
        for camera_group in camera_groups:
            roles = CAMERA_GROUPS[camera_group]
            scoped = cohort.loc[cohort["camera_role"].isin(roles)]
            for held_out in sorted(cohort["experiment_id"].unique()):
                native_train = scoped.loc[~scoped["experiment_id"].eq(held_out)]
                augmented_train = pd.concat(
                    [
                        native_train,
                        *(
                            condition_frames[condition].loc[
                                ~condition_frames[condition]["experiment_id"].eq(held_out)
                                & condition_frames[condition]["camera_role"].isin(roles)
                            ]
                            for condition in AUGMENTATION_CONDITIONS
                        ),
                    ],
                    ignore_index=True,
                )
                for strategy, train in zip(
                    TRAINING_STRATEGIES, (native_train, augmented_train), strict=True
                ):
                    model = make_rgb_model("logistic")
                    model.fit(train[columns], train["target"])
                    for condition, frame in condition_frames.items():
                        test = frame.loc[
                            frame["experiment_id"].eq(held_out)
                            & frame["camera_role"].isin(roles)
                        ]
                        if test["target"].nunique() < 2:
                            continue
                        predicted = model.predict(test[columns])
                        regret = test["relative_regret"].where(
                            test["target"].ne(predicted), 0.0
                        )
                        rows.append(
                            {
                                "representation": representation,
                                "training_strategy": strategy,
                                "camera_group": camera_group,
                                "condition": condition,
                                "held_out_experiment": held_out,
                                "balanced_accuracy": balanced_accuracy_score(
                                    test["target"], predicted
                                ),
                                "balanced_misclassification_regret": regret.groupby(
                                    test["target"]
                                )
                                .mean()
                                .mean(),
                                "test_image_count": len(test),
                                "train_image_count": len(train),
                            }
                        )
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, values in metrics.groupby(
        ["representation", "training_strategy", "camera_group", "condition"], sort=True
    ):
        representation, training_strategy, camera_group, condition = keys
        for metric in ("balanced_accuracy", "balanced_misclassification_regret"):
            rows.append(
                {
                    "representation": representation,
                    "training_strategy": training_strategy,
                    "camera_group": camera_group,
                    "condition": condition,
                    "metric": metric,
                    **bootstrap_mean_interval(values[metric]),
                    "experiment_count": len(values),
                }
            )
    experiment_stability = stability.groupby(
        ["representation", "condition", "experiment_id"], as_index=False
    )["cosine_similarity"].mean()
    for keys, values in experiment_stability.groupby(["representation", "condition"], sort=True):
        representation, condition = keys
        rows.append(
            {
                "representation": representation,
                "training_strategy": "not_applicable",
                "camera_group": "all",
                "condition": condition,
                "metric": "cosine_similarity",
                **bootstrap_mean_interval(values["cosine_similarity"]),
                "experiment_count": len(values),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    palette = ("#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756")
    representations = summary["representation"].dropna().unique().tolist()
    colors = dict(zip(representations, palette, strict=False))
    linestyles = {"native_only": "--", "low_light_augmented": "-"}
    labels = ("Native", "0.6×", "γ1.8", "γ2.2", "γ2.2 + noise", "γ1.8 + vignette")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"wspace": 0.34})
    for representation in representations:
        for strategy in TRAINING_STRATEGIES:
            accuracy = (
                summary.loc[
                    summary["representation"].eq(representation)
                    & summary["training_strategy"].eq(strategy)
                    & summary["camera_group"].eq("all")
                    & summary["metric"].eq("balanced_accuracy")
                ]
                .set_index("condition")
                .reindex(CONDITIONS)
            )
            axes[0].errorbar(
                range(len(CONDITIONS)),
                accuracy["estimate"],
                yerr=[
                    accuracy["estimate"] - accuracy["lower"],
                    accuracy["upper"] - accuracy["estimate"],
                ],
                marker="o",
                linestyle=linestyles[strategy],
                color=colors[representation],
                linewidth=1,
                markersize=3.5,
                capsize=2,
                label=f"{representation}, {'augmented' if strategy != 'native_only' else 'native'}",
            )
        stability = (
            summary.loc[
                summary["representation"].eq(representation)
                & summary["metric"].eq("cosine_similarity")
            ]
            .set_index("condition")
            .reindex(CONDITIONS[1:])
        )
        axes[1].plot(
            range(len(CONDITIONS) - 1),
            stability["estimate"],
            "o-",
            color=colors[representation],
            linewidth=1,
            markersize=3.5,
            label=representation,
        )
    axes[0].set(xticks=range(len(CONDITIONS)), xticklabels=labels, ylabel="Balanced accuracy")
    axes[1].set(
        xticks=range(len(CONDITIONS) - 1),
        xticklabels=labels[1:],
        ylabel="Native-to-shift cosine similarity",
    )
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=28, labelsize=6)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
        axis.tick_params(axis="y", labelsize=7)
        axis.set_xlabel("Synthetic illumination condition", fontsize=7)
    for label, axis in zip("ab", axes, strict=True):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=5.8, ncol=2, loc="lower left")
    fig.suptitle("Synthetic low-light stress test (not real low-light validation)", fontsize=8)
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 300},
    }.items():
        fig.savefig(
            output / f"illumination_robustness{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards",
        type=Path,
        default=Path("output/test/model/RGB特征缓存/多表征开发集/cycles"),
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--maximum-per-group", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--representation", choices=DEEP_REPRESENTATIONS, default="dinov2"
    )
    parser.add_argument(
        "--camera-groups", nargs="+", choices=CAMERA_GROUPS, default=["front", "all"]
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/test/model/低照度压力测试")
    )
    args = parser.parse_args()

    paths = sorted(args.shards.glob("frost_cycle_*.parquet"))
    cohort = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    cohort = cohort.loc[
        cohort["relative_regret"].gt(0.01)
        & cohort["cost_state"].isin(("pre_optimal", "post_optimal"))
    ].copy()
    cohort["target"] = cohort["cost_state"].map({"pre_optimal": 0, "post_optimal": 1})
    sampled = even_sample_groups(
        cohort,
        ["experiment_id", "cycle_name", "cost_state", "camera_role"],
        maximum_per_group=args.maximum_per_group,
    )
    sampled["absolute_path"] = sampled.apply(
        lambda row: str(
            args.dataset / "images" / row["cycle_name"] / row["camera_role"] / row["file_name"]
        ),
        axis=1,
    )
    sampled = sampled.loc[sampled["absolute_path"].map(lambda value: Path(value).is_file())]
    device = preferred_device()
    base_extractors = {
        args.representation: load_frozen_extractor(args.representation)
    }
    condition_frames = {"native": sampled.copy()}
    metadata = sampled.drop(
        columns=[
            column
            for column in sampled
            if any(column.startswith(f"{name}_") for name in DEEP_REPRESENTATIONS)
        ]
    )
    image_paths = metadata["absolute_path"].map(Path).tolist()
    stability_rows = []
    for condition in CONDITIONS[1:]:
        extractors = {
            name: (model, illumination_transform(condition))
            for name, (model, _) in base_extractors.items()
        }
        matrices = extract_representation_matrices(
            image_paths, extractors, device=device, batch_size=args.batch_size
        )
        shifted = add_embedding_columns(metadata, matrices)
        condition_frames[condition] = shifted
        for representation, matrix in matrices.items():
            native = sampled[representation_columns(sampled, representation)].to_numpy()
            similarity = cosine_similarity_rows(native, matrix)
            for row, score in zip(sampled.itertuples(index=False), similarity, strict=True):
                stability_rows.append(
                    {
                        "representation": representation,
                        "condition": condition,
                        "experiment_id": row.experiment_id,
                        "cycle_name": row.cycle_name,
                        "camera_role": row.camera_role,
                        "cosine_similarity": score,
                    }
                )
    metrics = evaluate_conditions(
        cohort,
        condition_frames,
        args.camera_groups,
        representations=[args.representation],
    )
    stability = pd.DataFrame(stability_rows)
    summary = summarize(metrics, stability)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "experiment_metrics.csv", index=False)
    stability.to_csv(args.output / "embedding_stability.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    plot_summary(summary, args.output)


if __name__ == "__main__":
    main()
