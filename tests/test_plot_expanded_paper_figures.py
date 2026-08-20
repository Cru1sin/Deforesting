from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def test_expanded_figures_export_editable_bundles(tmp_path) -> None:
    path = Path("scripts/plot_expanded_paper_figures.py")
    spec = importlib.util.spec_from_file_location("expanded_figures", path)
    assert spec and spec.loader
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)
    summary = pd.DataFrame(
        [
            {
                "camera_group": camera,
                "model": model,
                "metric": "balanced_accuracy",
                "estimate": 0.9,
                "lower": 0.8,
                "upper": 0.95,
            }
            for camera in plotter.CAMERA_ORDER
            for model in plotter.MODEL_ORDER
        ]
    )
    optima = pd.DataFrame(
        {
            "cycle_name": ["a", "b"],
            "cohort_tier": ["A_observed_policy"] * 2,
            "minutes_from_stable": [30.0, 60.0],
            "minimum_location": ["interior", "right_boundary"],
        }
    )
    concentration = pd.DataFrame(
        {
            "camera_group": plotter.CAMERA_ORDER,
            "estimate": [0.1] * 9,
            "lower": [0.05] * 9,
            "upper": [0.15] * 9,
            "time_optimum_iqr_over_median": [0.5] * 9,
        }
    )

    figures = tmp_path / "图表"
    plotter.plot_model_comparison(summary, figures)
    plotter.plot_concentration(optima, concentration, figures)

    for name in ("figure_5_model_camera_comparison", "figure_6_time_visual_concentration"):
        for suffix in (".svg", ".pdf", ".png"):
            assert (figures / f"{name}{suffix}").is_file()
        assert not (figures / f"{name}.tiff").exists()
    assert (tmp_path / "源数据" / "figure_5_model_comparison.csv").is_file()
