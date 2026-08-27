from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _module():  # type: ignore[no-untyped-def]
    path = Path("scripts/figures/plot_cost_to_rgb_paper.py")
    spec = importlib.util.spec_from_file_location("plot_cost_to_rgb_paper", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cost_to_rgb_conclusions_export_as_four_single_axis_figures(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    module = _module()
    curves = pd.DataFrame(
        {
            "cycle_name": ["cycle"] * 3,
            "candidate_time": pd.date_range("2026-01-01", periods=3, freq="10min"),
            "inverse_cop": [0.6, 0.5, 0.55],
            "cycle_cop": [1.67, 2.0, 1.82],
            "relative_regret": [0.2, 0.0, 0.1],
        }
    )
    bands = pd.DataFrame(
        {
            "relative_regret_threshold": [0.01, 0.02],
            "band_width_minutes": [10.0, 20.0],
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
    splits = pd.DataFrame(
        {
            "experiment_id": ["a", "b", "c"],
            "cycle_name": ["c1", "c2", "c3"],
            "split": ["train", "validation", "test"],
        }
    )
    seen: list[tuple[str, int]] = []

    def capture(fig: plt.Figure, stem: Path) -> None:
        seen.append((stem.name, len(fig.axes)))
        plt.close(fig)

    monkeypatch.setattr(module, "export", capture)

    module.plot_cost_to_rgb_evidence(curves, bands, balance, splits, tmp_path)

    assert seen == [
        ("figure_2_inverse_cop_example", 1),
        ("figure_3_near_optimal_width", 1),
        ("figure_4_label_coverage", 1),
        ("figure_5_split_independence", 1),
    ]
