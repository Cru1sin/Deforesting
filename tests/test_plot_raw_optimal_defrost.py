from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt


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
