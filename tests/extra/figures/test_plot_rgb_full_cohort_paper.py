from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def test_render_figures_exports_two_complete_bundles(tmp_path) -> None:
    path = Path("scripts/figures/plot_rgb_full_cohort_paper.py")
    spec = importlib.util.spec_from_file_location("plot_rgb_full_cohort_paper", path)
    assert spec and spec.loader
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)
    summary = pd.DataFrame(
        [
            {
                "camera_group": group,
                "modality": modality,
                "regret_threshold": threshold,
                "metric": "balanced_accuracy",
                "estimate": 0.8,
                "lower": 0.7,
                "upper": 0.9,
                "eligible_image_coverage": 0.6,
                "evaluable_experiment_count": 2,
            }
            for group in ("all", "front")
            for modality in ("rgb", "time", "rgb_time")
            for threshold in (0.01, 0.02)
        ]
    )
    deltas = pd.DataFrame(
        [
            {
                "camera_group": group,
                "regret_threshold": 0.01,
                "metric": "balanced_accuracy",
                "comparison": comparison,
                "estimate": 0.05,
                "lower": 0.01,
                "upper": 0.09,
            }
            for group in ("all", "front")
            for comparison in ("rgb_minus_time", "rgb_time_minus_time")
        ]
    )
    experiment_metrics = pd.DataFrame(
        [
            {
                "experiment_id": experiment,
                "camera_group": "all",
                "modality": modality,
                "regret_threshold": 0.01,
                "balanced_accuracy": 0.8,
                "balanced_misclassification_regret": 0.01,
            }
            for experiment in ("a", "b")
            for modality in ("rgb", "time", "rgb_time")
        ]
    )
    predictions = pd.DataFrame(
        {
            "experiment_id": ["a", "b"],
            "cycle_name": ["c1", "c2"],
            "camera_group": ["all", "all"],
            "modality": ["rgb", "rgb"],
            "regret_threshold": [0.01, 0.01],
            "target": [0, 1],
            "predicted_target": [0, 0],
            "relative_regret": [0.1, 0.2],
        }
    )

    figures = tmp_path / "图表"
    plotter.render_figures(summary, deltas, experiment_metrics, predictions, figures)

    for figure in ("figure_3_rgb_increment", "figure_4_failure_audit"):
        for suffix in (".svg", ".pdf", ".png"):
            assert (figures / f"{figure}{suffix}").is_file()
        assert not (figures / f"{figure}.tiff").exists()
    figure_3_svg = (figures / "figure_3_rgb_increment.svg").read_text()
    assert "n=2" in figure_3_svg
    assert "all views" in figure_3_svg
    assert (tmp_path / "源数据" / "figure_4_cycle_failures.csv").is_file()
