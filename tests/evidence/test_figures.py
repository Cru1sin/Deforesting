from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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
    evidence_settings = settings(targets=("heating_capacity", "cop"))

    figure = plot_cycle_progress(loader, evidence_settings)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert any(text.get_text() == "Unavailable" for text in figure.axes[1].texts)


def test_figure_three_consumes_only_horizon_summary(tmp_path: Path) -> None:
    loader = write_dataset(
        tmp_path / "dataset",
        [("c1", "2026-07-01", "valid", frame_for())],
    )
    evidence_settings = settings(targets=("heating_capacity",))
    bundle = build_evidence(loader, evidence_settings)

    figure = plot_future_horizon_summary(bundle.future_horizon_summary, evidence_settings)

    assert isinstance(figure, Figure)
    assert len(figure.axes) >= 1
    assert figure.axes[0].images[0].get_clim() == (-1.0, 1.0)


def test_s2_requires_settings(tmp_path: Path) -> None:
    loader = write_dataset(
        tmp_path / "dataset",
        [("c1", "2026-07-01", "valid", frame_for())],
    )
    evidence_settings = settings(targets=("heating_capacity",))
    bundle = build_evidence(loader, evidence_settings)

    with pytest.raises(TypeError, match="settings"):
        plot_availability_audit(bundle)


def test_s2_pair_panel_uses_only_primary_target_and_horizon(tmp_path: Path) -> None:
    loader = write_dataset(
        tmp_path / "dataset",
        [("c1", "2026-07-01", "valid", frame_for())],
    )
    evidence_settings = settings(
        targets=("heating_capacity", "cop"),
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
        settings(targets=("heating_capacity",)),
    )

    assert len(figure.axes[0].lines) == 2
    assert len(figure.axes[0].lines[0].get_xdata()) == 100
    assert int(np.isfinite(figure.axes[0].lines[0].get_ydata()).sum()) == 4


def test_figure_two_has_three_metrics_and_unconnected_date_values(
    tmp_path: Path,
) -> None:
    loader = write_dataset(
        tmp_path / "dataset",
        [
            (
                "c1",
                "2026-07-01",
                "valid",
                frame_for(
                    elapsed=(0, 300, 600, 900, 1200, 1500),
                    heating_capacity=(10, 9, 7, 4, 0, -5),
                ),
            ),
            (
                "c2",
                "2026-07-02",
                "valid",
                frame_for(
                    elapsed=(0, 300, 600, 900, 1200, 1500),
                    feature_a=(2, 3, 4, 5, 6, 7),
                    heating_capacity=(10, 9, 7, 4, 0, -5),
                ),
            ),
        ],
    )
    evidence_settings = settings(targets=("heating_capacity",))
    bundle = build_evidence(loader, evidence_settings)

    figure = plot_feature_profiles(bundle, evidence_settings)

    assert len(figure.axes) == 6
    assert [axis.get_title() for axis in figure.axes[:3]] == [
        "signed_effect",
        "trend_slope_per_min",
        "primary_future_degradation_support",
    ]

    signed_axis, slope_axis, support_axis = figure.axes[:3]
    assert signed_axis.get_ylim() == (-1.0, 1.0)
    assert support_axis.get_ylim() == (-1.0, 1.0)
    assert len(signed_axis.lines) == 1
    assert np.all(np.asarray(signed_axis.lines[0].get_ydata()) == 0)
    assert len(support_axis.lines) == 1
    assert np.all(np.asarray(support_axis.lines[0].get_ydata()) == 0)
    assert len(slope_axis.lines) == 0
    assert [len(collection.get_offsets()) for collection in signed_axis.collections] == [
        2,
        1,
    ]
    assert [len(collection.get_offsets()) for collection in support_axis.collections] == [
        2,
        1,
    ]


def test_figure_two_uses_feature_profile_order(tmp_path: Path) -> None:
    loader = write_dataset(
        tmp_path / "dataset",
        [("c1", "2026-07-01", "valid", frame_for())],
    )
    evidence_settings = settings(targets=("heating_capacity",))
    bundle = build_evidence(loader, evidence_settings)
    reordered = EvidenceBundle(
        cycle_eligibility=bundle.cycle_eligibility,
        feature_cycle_metrics=bundle.feature_cycle_metrics,
        future_association=bundle.future_association,
        future_horizon_summary=bundle.future_horizon_summary,
        feature_profile=bundle.feature_profile.iloc[::-1].reset_index(drop=True),
        feature_pair_similarity=bundle.feature_pair_similarity,
        target_audit=bundle.target_audit,
        readiness_split=bundle.readiness_split,
        readiness_summary=bundle.readiness_summary,
    )

    figure = plot_feature_profiles(reordered, evidence_settings)

    assert figure.axes[0].get_ylabel() == "feature_b"
    assert figure.axes[3].get_ylabel() == "feature_a"


def test_figure_two_keeps_long_labels_inside_figure(tmp_path: Path) -> None:
    loader = write_dataset(
        tmp_path / "dataset",
        [
            (
                "c1",
                "2026-07-01",
                "valid",
                frame_for(heating_capacity=(10, 9, 7, 4, 0, -5)),
            )
        ],
    )
    evidence_settings = settings(targets=("heating_capacity",))
    bundle = build_evidence(loader, evidence_settings)
    feature_names = (
        "temperature_operating_pressure",
        "coil_temperature",
        "fin_temperature",
        "surface_temperature",
    )
    profile_rows = []
    metric_rows = []
    future_rows = []
    for index, feature_name in enumerate(feature_names):
        template = "feature_a" if index % 2 == 0 else "feature_b"
        profile_rows.append(
            bundle.feature_profile.loc[
                bundle.feature_profile["feature"].eq(template)
            ].assign(feature=feature_name)
        )
        metric_rows.append(
            bundle.feature_cycle_metrics.loc[
                bundle.feature_cycle_metrics["feature"].eq(template)
            ].assign(feature=feature_name)
        )
        future_rows.append(
            bundle.future_association.loc[
                bundle.future_association["feature"].eq(template)
            ].assign(feature=feature_name)
        )
    long_bundle = EvidenceBundle(
        cycle_eligibility=bundle.cycle_eligibility,
        feature_cycle_metrics=pd.concat(metric_rows, ignore_index=True),
        future_association=pd.concat(future_rows, ignore_index=True),
        future_horizon_summary=bundle.future_horizon_summary,
        feature_profile=pd.concat(profile_rows, ignore_index=True),
        feature_pair_similarity=bundle.feature_pair_similarity,
        target_audit=bundle.target_audit,
        readiness_split=bundle.readiness_split,
        readiness_summary=bundle.readiness_summary,
    )

    figure = plot_feature_profiles(long_bundle, evidence_settings)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width = figure.bbox.width
    texts = [axis.title for axis in figure.axes[:3]] + [
        axis.yaxis.label for axis in figure.axes[::3]
    ]

    for text in texts:
        bounds = text.get_window_extent(renderer)
        assert bounds.x0 >= 0, text.get_text()
        assert bounds.x1 <= width, text.get_text()

    label_bounds = [
        axis.yaxis.label.get_window_extent(renderer) for axis in figure.axes[::3]
    ]
    for upper, lower in zip(label_bounds, label_bounds[1:], strict=False):
        assert lower.y1 <= upper.y0


def test_figure_three_keeps_long_feature_labels_inside_figure() -> None:
    feature_names = [
        "temperature_operating_pressure",
        "coil_temperature",
        "fin_temperature",
        "surface_temperature",
    ]
    summary = pd.DataFrame(
        {
            "feature": feature_names,
            "target": ["heating_capacity"] * len(feature_names),
            "horizon_minutes": [5] * len(feature_names),
            "effect": [-0.4] * len(feature_names),
            "degradation_support": [0.4] * len(feature_names),
            "valid_cycle_count": [3] * len(feature_names),
            "valid_date_count": [2] * len(feature_names),
            "aggregation_method": [
                "date_balanced_median_of_cycle_medians_v1"
            ]
            * len(feature_names),
            "metric_status": ["available"] * len(feature_names),
            "exclusion_reason": [""] * len(feature_names),
        }
    )

    figure = plot_future_horizon_summary(
        summary,
        settings(targets=("heating_capacity",)),
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width = figure.bbox.width

    for label in figure.axes[0].get_yticklabels():
        bounds = label.get_window_extent(renderer)
        assert bounds.x0 >= 0, label.get_text()
        assert bounds.x1 <= width, label.get_text()


def test_figure_three_labels_summary_counts_and_support() -> None:
    summary = pd.DataFrame(
        {
            "feature": ["feature_a"],
            "target": ["heating_capacity"],
            "horizon_minutes": [5],
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
        settings(targets=("heating_capacity",)),
    )

    labels = " ".join(text.get_text() for text in figure.axes[0].texts)
    assert "cycle=3" in labels
    assert "date=2" in labels
