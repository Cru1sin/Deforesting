from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def test_save_figure_strips_svg_trailing_whitespace(tmp_path) -> None:
    path = Path("scripts/analyze_raw_optimal_defrost.py")
    spec = importlib.util.spec_from_file_location("analyze_raw_optimal_defrost", path)
    assert spec and spec.loader
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])

    analysis._save_figure(figure, tmp_path / "figure")

    lines = (tmp_path / "figure.svg").read_text().splitlines()
    assert all(line == line.rstrip() for line in lines)


def test_near_optimal_segments_preserve_disconnected_time_bands() -> None:
    path = Path("scripts/analyze_raw_optimal_defrost.py")
    spec = importlib.util.spec_from_file_location("analyze_raw_optimal_defrost", path)
    assert spec and spec.loader
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    curve = pd.DataFrame(
        {
            "minutes": [10, 11, 12, 13, 14],
            "relative_regret": [0.01, 0.02, 0.08, 0.03, 0.09],
        }
    )

    assert analysis._near_optimal_segments(curve) == [(10.0, 11.0), (13.0, 13.0)]
