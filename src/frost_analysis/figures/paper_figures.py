"""Small source-table transforms for the cost-to-RGB paper figures."""

from __future__ import annotations

import pandas as pd

from ..labels.cost import high_confidence_coverage


def regret_threshold_summary(
    bands: pd.DataFrame, label_balance: pd.DataFrame
) -> pd.DataFrame:
    """Summarize timing ambiguity and retained image coverage by regret threshold."""
    summary = (
        bands.groupby("relative_regret_threshold", as_index=False)["band_width_minutes"]
        .median()
        .rename(
            columns={
                "relative_regret_threshold": "regret_threshold",
                "band_width_minutes": "median_width_minutes",
            }
        )
    )
    summary["eligible_image_coverage"] = [
        high_confidence_coverage(label_balance, "all", threshold)
        for threshold in summary["regret_threshold"]
    ]
    return summary


def full_cohort_figure_3_sources(
    summary: pd.DataFrame, deltas: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Return the locked comparison tables used by the cohort figure."""
    metric = summary["metric"].eq("balanced_accuracy")
    primary = summary["regret_threshold"].eq(0.01)
    return {
        "camera_performance": summary.loc[metric & primary].copy(),
        "camera_deltas": deltas.loc[
            deltas["metric"].eq("balanced_accuracy")
            & deltas["regret_threshold"].eq(0.01)
        ].copy(),
        "threshold_tradeoff": summary.loc[
            metric & summary["camera_group"].eq("all")
        ].sort_values(["regret_threshold", "modality"]),
    }


def full_cohort_figure_4_sources(
    experiment_metrics: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Return primary held-out experiment metrics and cycle-level error costs."""
    primary_experiments = experiment_metrics.loc[
        experiment_metrics["camera_group"].eq("all")
        & experiment_metrics["regret_threshold"].eq(0.01)
    ].copy()
    primary_predictions = predictions.loc[
        predictions["camera_group"].eq("all")
        & predictions["modality"].eq("rgb")
        & predictions["regret_threshold"].eq(0.01)
    ].copy()
    primary_predictions["incorrect"] = primary_predictions["target"].ne(
        primary_predictions["predicted_target"]
    )
    primary_predictions["misclassification_regret"] = primary_predictions[
        "relative_regret"
    ].where(primary_predictions["incorrect"], 0.0)
    cycle_failures = (
        primary_predictions.groupby(["experiment_id", "cycle_name"], as_index=False)
        .agg(
            image_count=("target", "size"),
            error_rate=("incorrect", "mean"),
            mean_misclassification_regret=("misclassification_regret", "mean"),
            maximum_error_regret=("misclassification_regret", "max"),
        )
        .sort_values("mean_misclassification_regret", ascending=False)
    )
    return {
        "experiment_metrics": primary_experiments,
        "cycle_failures": cycle_failures,
    }
