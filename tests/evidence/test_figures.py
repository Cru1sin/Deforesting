from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from frost_analysis.evidence import EvidenceBundle, build_evidence
from frost_analysis.evidence.figures import (
    plot_availability_audit,
    plot_cycle_progress,
    plot_feature_profiles,
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
    assert figure.axes[0].images[0].get_clim() == (-1.0, 1.0)


def test_s2_is_two_panel_local_availability_audit(tmp_path: Path) -> None:
    loader = write_dataset(
        tmp_path / "dataset",
        [("c1", "2026-07-01", "valid", frame_for())],
    )
    evidence_settings = settings(targets=("heating_capacity",), horizons=(1,))
    bundle = build_evidence(loader, evidence_settings)

    figure = plot_availability_audit(bundle)

    assert len(figure.axes) == 2
    assert "availability" in figure.axes[0].get_title()
    assert "pair coverage" in figure.axes[1].get_title()


def test_s2_pair_panel_uses_only_primary_target_and_horizon(tmp_path: Path) -> None:
    loader = write_dataset(
        tmp_path / "dataset",
        [("c1", "2026-07-01", "valid", frame_for())],
    )
    evidence_settings = settings(
        targets=("heating_capacity", "cop"),
        horizons=(1, 2),
    )
    bundle = build_evidence(loader, evidence_settings)

    figure = plot_availability_audit(bundle, evidence_settings)

    labels = [label.get_text() for label in figure.axes[1].get_xticklabels()]
    assert labels == ["feature_a", "feature_b"]


def test_figure_one_uses_fixed_bins_and_excludes_imputed_nonfinite_values(
    tmp_path: Path,
) -> None:
    frame = frame_for()
    frame.loc[2, "heating_capacity__imputed"] = True
    frame.loc[3, "heating_capacity__baseline_residual"] = np.inf
    loader = write_dataset(tmp_path / "dataset", [("c1", "2026-07-01", "valid", frame)])

    figure = plot_cycle_progress(
        loader,
        settings(targets=("heating_capacity",), horizons=(1,)),
    )

    assert len(figure.axes[0].lines) == 2
    assert len(figure.axes[0].lines[0].get_xdata()) == 100
    assert int(np.isfinite(figure.axes[0].lines[0].get_ydata()).sum()) == 4


def test_figure_two_has_three_metrics_and_date_level_lines(tmp_path: Path) -> None:
    loader = write_dataset(
        tmp_path / "dataset",
        [
            ("c1", "2026-07-01", "valid", frame_for()),
            ("c2", "2026-07-02", "valid", frame_for(feature_a=(2, 3, 4, 5, 6, 7))),
        ],
    )
    evidence_settings = settings(targets=("heating_capacity",), horizons=(1,))
    bundle = build_evidence(loader, evidence_settings)

    figure = plot_feature_profiles(bundle, evidence_settings)

    assert len(figure.axes) == 6
    assert [axis.get_title() for axis in figure.axes[:3]] == [
        "signed_effect",
        "trend_slope_per_min",
        "primary_future_degradation_support",
    ]


def test_figure_two_uses_feature_profile_order(tmp_path: Path) -> None:
    loader = write_dataset(
        tmp_path / "dataset",
        [("c1", "2026-07-01", "valid", frame_for())],
    )
    evidence_settings = settings(targets=("heating_capacity",), horizons=(1,))
    bundle = build_evidence(loader, evidence_settings)
    reordered = EvidenceBundle(
        cycle_eligibility=bundle.cycle_eligibility,
        feature_cycle_metrics=bundle.feature_cycle_metrics,
        future_association=bundle.future_association,
        future_horizon_summary=bundle.future_horizon_summary,
        feature_profile=bundle.feature_profile.iloc[::-1].reset_index(drop=True),
        feature_pair_similarity=bundle.feature_pair_similarity,
    )

    figure = plot_feature_profiles(reordered, evidence_settings)

    assert figure.axes[0].get_ylabel() == "feature_b"
    assert figure.axes[3].get_ylabel() == "feature_a"


def test_figure_three_labels_summary_counts_and_support() -> None:
    summary = pd.DataFrame(
        {
            "feature": ["feature_a"],
            "target": ["heating_capacity"],
            "horizon_minutes": [1],
            "effect": [-0.4],
            "degradation_support": [0.4],
            "valid_cycle_count": [3],
            "valid_date_count": [2],
            "aggregation_method": ["date_balanced_median_of_cycle_medians_v1"],
            "metric_status": ["available"],
            "exclusion_reason": [""],
        }
    )

    figure = plot_future_horizon_summary(
        summary,
        settings(targets=("heating_capacity",), horizons=(1,)),
    )

    labels = " ".join(text.get_text() for text in figure.axes[0].texts)
    assert "cycle=3" in labels
    assert "date=2" in labels
