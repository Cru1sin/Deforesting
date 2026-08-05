from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from frost_analysis.evidence import build_evidence
from frost_analysis.evidence.figures import (
    plot_availability_audit,
    plot_cycle_progress,
    plot_future_horizon_summary,
)

from .conftest import frame_for, settings, write_dataset


def test_figure_one_keeps_unavailable_target_panel(tmp_path: Path) -> None:
    frame = frame_for()
    frame = frame.drop(columns=["cop__baseline_residual"])
    loader = write_dataset(tmp_path / "dataset", [("c1", "2026-07-01", "valid", frame)])
    evidence_settings = settings(targets=("heating_capacity", "cop"), horizons=(1,))

    figure = plot_cycle_progress(loader, evidence_settings)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert any(text.get_text() == "Unavailable" for text in figure.axes[1].texts)


def test_figure_three_consumes_only_horizon_summary(tmp_path: Path) -> None:
    loader = write_dataset(
        tmp_path / "dataset",
        [("c1", "2026-07-01", "valid", frame_for())],
    )
    evidence_settings = settings(targets=("heating_capacity",), horizons=(1,))
    bundle = build_evidence(loader, evidence_settings)

    figure = plot_future_horizon_summary(bundle.future_horizon_summary, evidence_settings)

    assert isinstance(figure, Figure)
    assert len(figure.axes) >= 1


def test_s2_is_two_panel_local_availability_audit(tmp_path: Path) -> None:
    loader = write_dataset(
        tmp_path / "dataset",
        [("c1", "2026-07-01", "valid", frame_for())],
    )
    evidence_settings = settings(targets=("heating_capacity",), horizons=(1,))
    bundle = build_evidence(loader, evidence_settings)

    figure = plot_availability_audit(bundle)

    assert len(figure.axes) == 2
    assert "feature availability" in figure.axes[0].get_title()
    assert "future pair coverage" in figure.axes[1].get_title()
