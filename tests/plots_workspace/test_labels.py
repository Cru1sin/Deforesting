from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import build_image_labels
from plots import image_labels as label_plots


def _cost() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cycle_name": ["cycle"] * 5,
            "candidate_defrost_time": pd.date_range("2026-01-01", periods=5, freq="10min"),
            "inverse_cop": [0.6, 0.5, 0.7, 0.51, 0.8],
            "relative_regret": [0.01, 0.0, 0.2, 0.005, 0.0],
            "optimization_eligible": [True, True, False, True, False],
            "is_censored": [False] * 5,
            "label_eligible": [True] * 5,
        }
    )


def _balance() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regret_threshold": [0.01] * 3 + [0.02] * 3,
            "camera_group": ["all"] * 6,
            "timing_state": ["before_reference", "near_reference", "after_reference"] * 2,
            "image_count": [30, 40, 30, 20, 60, 20],
        }
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment_id": ["a", "a", "b", "c"],
            "cycle_name": ["c1", "c1", "c2", "c3"],
        }
    )


def test_regret_threshold_summary_joins_width_and_absolute_coverage() -> None:
    bands = pd.DataFrame(
        {
            "relative_regret_threshold": [0.01, 0.01, 0.02, 0.02],
            "band_width_minutes": [10.0, 30.0, 20.0, 40.0],
        }
    )

    summary = label_plots.regret_threshold_summary(bands, _balance())

    assert summary["median_width_minutes"].tolist() == [20.0, 30.0]
    assert summary["eligible_image_coverage"].tolist() == [0.6, 0.4]


def test_label_figures_reuse_four_single_axis_layouts_and_span_band_widths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, int, tuple[str, ...]]] = []

    def capture(fig: plt.Figure, stem: Path, formats: tuple[str, ...]) -> None:
        seen.append((stem.name, len(fig.axes), formats))
        plt.close(fig)

    monkeypatch.setattr(label_plots, "_export", capture)
    label_plots.plot_label_figures(
        cost=_cost(),
        labels=_labels(),
        balance=_balance(),
        thresholds=[0.01, 0.02],
        output=tmp_path / "figures",
        source_output=tmp_path / "source_data",
        figure_formats=("png", "svg"),
    )

    assert seen == [
        ("figure_2_inverse_cop_example", 1, ("png", "svg")),
        ("figure_3_near_optimal_width", 1, ("png", "svg")),
        ("figure_4_label_coverage", 1, ("png", "svg")),
        ("figure_5_dataset_scope", 1, ("png", "svg")),
    ]
    bands = pd.read_csv(tmp_path / "source_data" / "near_optimal_band_widths.csv")
    assert bands["band_width_minutes"].tolist() == [30.0, 30.0]
    scope = pd.read_csv(tmp_path / "source_data" / "dataset_scope.csv")
    assert scope.set_index("metric")["count"].to_dict() == {
        "Experiments": 3,
        "Cycles": 3,
        "Images": 4,
    }


def test_export_defaults_to_png_and_explicit_formats_share_one_figure(
    tmp_path: Path,
) -> None:
    label_plots._export(plt.figure(), tmp_path / "default")
    assert sorted(path.suffix for path in tmp_path.glob("default.*")) == [".png"]

    label_plots._export(
        plt.figure(),
        tmp_path / "all",
        ("png", "svg", "pdf"),
    )
    assert sorted(path.suffix for path in tmp_path.glob("all.*")) == [".pdf", ".png", ".svg"]


def test_main_routes_explicit_figure_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cost = _cost()
    labels = _labels()
    balance = _balance()
    output = tmp_path / "labels"
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(build_image_labels.pd, "read_csv", lambda _path, **_kwargs: cost)
    monkeypatch.setattr(
        build_image_labels,
        "build_labels",
        lambda _dataset, _cost, _thresholds: (labels, balance, pd.DataFrame()),
    )
    monkeypatch.setattr(
        build_image_labels, "plot_label_figures", lambda **kwargs: calls.append(kwargs)
    )

    assert (
        build_image_labels.main(
            [
                "--label-source",
                "cost-optimum",
                "--source-table",
                str(tmp_path / "cost.csv"),
                "--output",
                str(output),
                "--figures",
                "--figure-output",
                str(tmp_path / "paper"),
                "--figure-format",
                "svg",
                "pdf",
            ]
        )
        == 0
    )

    assert calls == [
        {
            "cost": cost,
            "labels": labels,
            "balance": balance,
            "thresholds": [0.01, 0.02, 0.05, 0.10],
            "output": tmp_path / "paper",
            "source_output": output / "figure_source_data",
            "figure_formats": ("svg", "pdf"),
        }
    ]
