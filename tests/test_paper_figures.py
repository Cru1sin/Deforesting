from __future__ import annotations

import pandas as pd

from frost_analysis import paper_figures
from frost_analysis.paper_figures import regret_threshold_summary


def test_regret_threshold_summary_joins_width_and_absolute_coverage() -> None:
    bands = pd.DataFrame(
        {
            "relative_regret_threshold": [0.01, 0.01, 0.02, 0.02],
            "band_width_minutes": [10.0, 30.0, 20.0, 40.0],
        }
    )
    balance = pd.DataFrame(
        {
            "regret_threshold": [0.01] * 3 + [0.02] * 3,
            "camera_group": ["all"] * 6,
            "cost_state": ["pre_optimal", "near_optimal", "post_optimal"] * 2,
            "image_count": [30, 40, 30, 20, 60, 20],
        }
    )

    summary = regret_threshold_summary(bands, balance)

    assert summary["median_width_minutes"].tolist() == [20.0, 30.0]
    assert summary["eligible_image_coverage"].tolist() == [0.6, 0.4]


def test_full_cohort_figure_3_sources_lock_primary_threshold_and_group() -> None:
    summary = pd.DataFrame(
        {
            "camera_group": ["top", "all", "all", "all"],
            "modality": ["rgb", "rgb", "rgb", "rgb"],
            "regret_threshold": [0.01, 0.01, 0.02, 0.01],
            "metric": ["balanced_accuracy"] * 3 + ["auroc"],
            "estimate": [0.7, 0.8, 0.9, 0.85],
        }
    )
    deltas = pd.DataFrame(
        {
            "camera_group": ["top", "all", "all"],
            "regret_threshold": [0.01, 0.01, 0.02],
            "metric": ["balanced_accuracy"] * 3,
            "comparison": ["rgb_minus_time"] * 3,
            "estimate": [0.1, 0.2, 0.3],
        }
    )

    sources = paper_figures.full_cohort_figure_3_sources(summary, deltas)

    assert sources["camera_performance"]["camera_group"].tolist() == ["top", "all"]
    assert sources["camera_deltas"]["camera_group"].tolist() == ["top", "all"]
    assert sources["threshold_tradeoff"]["regret_threshold"].tolist() == [0.01, 0.02]


def test_full_cohort_figure_4_sources_summarize_cycle_error_cost() -> None:
    experiment_metrics = pd.DataFrame(
        {
            "experiment_id": ["a", "a"],
            "camera_group": ["all", "top"],
            "modality": ["rgb", "rgb"],
            "regret_threshold": [0.01, 0.01],
            "balanced_accuracy": [0.8, 0.7],
        }
    )
    predictions = pd.DataFrame(
        {
            "experiment_id": ["a", "a", "a"],
            "cycle_name": ["c1", "c1", "c2"],
            "camera_group": ["all", "all", "top"],
            "modality": ["rgb", "rgb", "rgb"],
            "regret_threshold": [0.01, 0.01, 0.01],
            "target": [0, 1, 0],
            "predicted_target": [0, 0, 1],
            "relative_regret": [0.1, 0.3, 0.9],
        }
    )

    sources = paper_figures.full_cohort_figure_4_sources(
        experiment_metrics, predictions
    )

    assert sources["experiment_metrics"]["experiment_id"].tolist() == ["a"]
    cycle = sources["cycle_failures"].iloc[0]
    assert cycle["cycle_name"] == "c1"
    assert cycle["image_count"] == 2
    assert cycle["error_rate"] == 0.5
    assert cycle["mean_misclassification_regret"] == 0.15
