from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from frost_analysis.config import EvidencePolicy
from frost_analysis.evidence_figures import (
    EvidenceFigureSettings,
    _progress_bins,
    _reason_code,
    plot_cycle_evolution,
    plot_evidence_coverage,
    plot_evidence_profiles,
    plot_future_horizon_map,
    plot_pair_similarity,
)

FEATURES = ("signal_a", "signal_b")


def _settings() -> EvidenceFigureSettings:
    return EvidenceFigureSettings(
        feature_groups=(("Sensors", FEATURES),),
        cycle_channels=("signal_a", "signal_b"),
        display_modes={"signal_a": "raw", "signal_b": "reference_normalized"},
        date_order=("2026-07-15",),
    )


def _policy() -> EvidencePolicy:
    return EvidencePolicy(
        min_segment_points=2,
        min_valid_pairs=2,
        min_valid_cycles=1,
        horizons_minutes=(5, 10),
        primary_horizon_minutes=10,
        targets=("heating_capacity", "cop"),
        primary_target="heating_capacity",
        lead_target="heating_capacity",
    )


def _profile() -> pd.DataFrame:
    rows = []
    for feature, trend, future in (("signal_a", 0.6, 0.4), ("signal_b", -0.4, -0.2)):
        rows.append(
            {
                "feature": feature,
                "registry_role": "sensor",
                "reference_scope": "auto_only",
                "trend_valid_cycle_count": 2,
                "trend_valid_date_count": 1,
                "global_spearman_median": trend,
                "global_spearman_iqr": 0.2,
                "signed_sensitivity_median": trend,
                "signed_sensitivity_iqr": 0.1,
                "primary_future_valid_cycle_count": 2,
                "primary_future_valid_date_count": 1,
                "primary_future_effect_median": future,
                "primary_future_effect_iqr": 0.2,
                "trend_evidence_status": "within_date_exploratory",
                "primary_future_evidence_status": "within_date_exploratory",
                "configured_baseline_cycle_count": 0,
                "auto_reference_cycle_count": 2,
            }
        )
    return pd.DataFrame(rows)


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment_id": ["exp", "exp"],
            "experiment_date": ["2026-07-15", "2026-07-15"],
            "cycle_id": ["c1", "c2"],
            "feature": ["signal_a", "signal_b"],
            "global_spearman": [0.6, -0.4],
            "signed_sensitivity": [0.5, -0.3],
            "reference_source": ["auto_cycle_initial_reference"] * 2,
            "reference_valid_from": pd.to_datetime(
                ["2026-07-15 00:05:00", "2026-07-15 00:05:00"]
            ),
            "onset_elapsed_minutes": [6.0, 7.0],
        }
    )


def _future() -> pd.DataFrame:
    rows = []
    for feature, effect in (("signal_a", 0.4), ("signal_b", -0.2)):
        for target in ("heating_capacity", "cop"):
            for horizon in (5, 10):
                rows.append(
                    {
                        "experiment_id": "exp",
                        "experiment_date": "2026-07-15",
                        "cycle_id": "c1",
                        "feature": feature,
                        "feature_variant": "residual_level",
                        "feature_reference_source": "auto_cycle_initial_reference",
                        "target": target,
                        "target_reference_source": "auto_cycle_initial_reference",
                        "horizon_minutes": horizon,
                        "target_type": "future_change",
                        "effect": effect,
                        "pair_coverage": 0.9,
                        "valid_pairs": 4,
                        "expected_anchor_count": 5,
                        "metric_status": "available",
                        "exclusion_reason": "",
                    }
                )
    return pd.DataFrame(rows)


def _eligibility() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment_id": ["exp"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["c1"],
            "eligibility_status": ["eligible"],
            "frost_development_grid_coverage": [1.0],
            "eligible_feature_count": [2],
            "total_candidate_count": [2],
            "exclusion_reason": [""],
        }
    )


def _pair() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": ["signal_a"],
            "feature_b": ["signal_b"],
            "dynamic_spearman_median": [0.9],
            "dynamic_spearman_iqr": [0.1],
            "similarity_sign_agreement": [1.0],
            "evaluated_cycle_count": [2],
            "valid_cycle_count": [1],
            "valid_date_count": [1],
            "pair_coverage_median": [0.7],
            "definition_dependency": [False],
            "similarity_status": ["insufficient_cycles"],
            "similarity_reason": ["valid_cycles_below_minimum"],
        }
    )


def test_settings_validate_and_flatten_display_contract() -> None:
    settings = _settings()

    assert settings.feature_order == FEATURES

    with pytest.raises(ValueError, match="duplicate feature"):
        EvidenceFigureSettings(
            feature_groups=(("A", ("signal_a",)), ("B", ("signal_a",))),
            cycle_channels=("signal_a",),
            display_modes={"signal_a": "raw"},
        )

    with pytest.raises(ValueError, match="feature_groups must not be empty"):
        EvidenceFigureSettings(
            feature_groups=(),
            cycle_channels=("signal_a",),
            display_modes={"signal_a": "raw"},
        )

    with pytest.raises(ValueError, match="cycle_channels must not be empty"):
        EvidenceFigureSettings(
            feature_groups=(("Sensors", ("signal_a",)),),
            cycle_channels=(),
            display_modes={},
        )


def test_progress_bin_uses_median_of_all_points_in_the_bin() -> None:
    progress = pd.Series([0.001, 0.002, 0.003])
    values = pd.Series([1.0, 100.0, 100.0])

    binned = _progress_bins(progress, values)

    assert binned[0] == pytest.approx(100.0)
    assert len(binned) == 100


def test_public_figures_return_figures_and_do_not_mutate_inputs() -> None:
    profile = _profile()
    metrics = _metrics()
    future = _future()
    eligibility = _eligibility()
    pair = _pair()
    before = [frame.copy(deep=True) for frame in (profile, metrics, future, eligibility, pair)]
    settings = _settings()
    policy = _policy()

    figures = [
        plot_evidence_profiles(profile, metrics, future, policy, settings),
        plot_future_horizon_map(future, policy, settings),
        plot_pair_similarity(pair, settings),
        plot_evidence_coverage(eligibility, metrics, future, policy, settings),
    ]

    assert all(isinstance(figure, Figure) for figure in figures)
    reliability_text = " ".join(
        text.get_text() for text in figures[0].axes[3].texts
    )
    assert "future n=2/d=1" in reliability_text
    assert "within_date_exploratory" in reliability_text
    for frame, original in zip(
        (profile, metrics, future, eligibility, pair), before, strict=True
    ):
        pd.testing.assert_frame_equal(frame, original)
    for figure in figures:
        figure.savefig("/tmp/evidence-figure-test.png")
    figures[0].savefig("/tmp/evidence-figure-test.pdf")
    figures[0].savefig("/tmp/evidence-figure-test.svg")
    for figure in figures:
        plt.close(figure)


def test_missing_requested_feature_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="requested figure features are missing"):
        plot_evidence_profiles(
            _profile().loc[lambda frame: frame["feature"].eq("signal_a")],
            _metrics(),
            _future(),
            _policy(),
            _settings(),
        )


def test_cycle_evolution_uses_formal_progress_and_returns_figure() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=72, freq="10s")
    processed = pd.DataFrame(
        {
            "experiment_id": "exp",
            "experiment_date": "2026-07-15",
            "cycle_id": "c1",
            "cycle_stage": "frost_development",
            "cycle_status": "valid",
            "timestamp": timestamps,
            "signal_a": np.linspace(1.0, 2.0, len(timestamps)),
            "signal_b": np.linspace(2.0, 1.0, len(timestamps)),
            "heating_capacity": np.linspace(5.0, 4.0, len(timestamps)),
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["c1"],
            "cycle_status": ["valid"],
            "stable_heating_start": [start],
            "defrost_start": [start + pd.Timedelta(minutes=12)],
        }
    )
    channels = {
        "signal_a": {"unit": "degC"},
        "signal_b": {"unit": "degC"},
    }
    processed_before = processed.copy(deep=True)
    summary_before = summary.copy(deep=True)
    eligibility = _eligibility()
    eligibility_before = eligibility.copy(deep=True)

    figure = plot_cycle_evolution(
        processed,
        summary,
        eligibility,
        channels,
        _policy(),
        _settings(),
        10,
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert any(len(line.get_xdata()) == 100 for line in figure.axes[0].lines)
    assert "Reference-normalized residual" in figure.axes[1].get_ylabel()
    assert "degC" in figure.axes[0].get_ylabel()
    assert any("Vertical ticks indicate" in text.get_text() for text in figure.texts)
    pd.testing.assert_frame_equal(processed, processed_before)
    pd.testing.assert_frame_equal(summary, summary_before)
    pd.testing.assert_frame_equal(eligibility, eligibility_before)
    plt.close(figure)


def test_future_map_masks_unavailable_values_and_uses_policy_targets() -> None:
    figure = plot_future_horizon_map(_future(), _policy(), _settings())

    assert len(figure.axes) >= 4
    residual = figure.axes[0].images[0].get_array()
    slope = figure.axes[1].images[0].get_array()
    assert residual.shape == (2, 2)
    assert float(residual[0, 0]) == pytest.approx(0.4)
    assert float(residual[1, 0]) == pytest.approx(-0.2)
    assert np.ma.isMaskedArray(slope)
    assert bool(np.ma.getmaskarray(slope).all())
    assert tuple(figure.axes[0].get_yticklabels()[index].get_text() for index in range(2)) == (
        "signal_a",
        "signal_b",
    )
    assert len(figure.axes[0].patches) == 1
    plt.close(figure)


def test_future_map_keeps_target_variant_horizon_and_feature_dimensions() -> None:
    future = pd.concat(
        [
            _future(),
            _future().assign(feature_variant="past_slope_5min", effect=0.25),
        ],
        ignore_index=True,
    )
    figure = plot_future_horizon_map(future, _policy(), _settings())

    heating_residual = figure.axes[0].images[0].get_array()
    heating_slope = figure.axes[1].images[0].get_array()
    cop_residual = figure.axes[2].images[0].get_array()
    assert float(heating_residual[0, 0]) == pytest.approx(0.4)
    assert float(heating_residual[1, 0]) == pytest.approx(-0.2)
    assert float(heating_slope[0, 0]) == pytest.approx(0.25)
    assert float(cop_residual[0, 0]) == pytest.approx(0.4)
    assert tuple(figure.axes[0].get_xticklabels()[index].get_text() for index in range(2)) == (
        "5 min",
        "10 min",
    )
    assert figure.axes[0].patches[0].get_x() == pytest.approx(0.5)
    plt.close(figure)


def test_pair_insufficient_cycles_keeps_finite_effect_visible() -> None:
    figure = plot_pair_similarity(_pair(), _settings())

    image = figure.axes[0].images[0].get_array()
    assert float(image[1, 0]) == pytest.approx(0.9)
    assert "D=definition dependency" in figure.axes[0].get_title()
    annotations = [text.get_text() for text in figure.axes[0].texts]
    assert any(
        "e=2/v=1/d=1" in text
        and "reason: valid cycles below minimum" in text.replace("\n", " ")
        for text in annotations
    )
    assert len(figure.axes[0].patches) == 1
    plt.close(figure)


def test_pair_no_valid_evidence_is_masked() -> None:
    pair = _pair().assign(
        dynamic_spearman_median=np.nan,
        similarity_status="no_valid_evidence",
        similarity_reason="no_valid_pairs",
    )
    figure = plot_pair_similarity(pair, _settings())

    image = figure.axes[0].images[0].get_array()
    assert bool(np.ma.getmaskarray(image)[1, 0])
    plt.close(figure)


@pytest.mark.parametrize(
    ("reason", "code"),
    (
        ("feature_reference_unavailable", "R"),
        ("pair_coverage_below_minimum", "C"),
        ("valid_pairs_below_minimum", "P"),
        ("zero_variability", "V"),
        ("no_structural_anchors", "N"),
        ("cycle_invalid", "X"),
    ),
)
def test_future_map_preserves_existing_exclusion_reason_codes(
    reason: str, code: str
) -> None:
    assert _reason_code([reason]) == code


def test_coverage_uses_full_cycle_key_and_candidate_reference_denominator() -> None:
    eligibility = pd.DataFrame(
        {
            "experiment_id": ["exp_a", "exp_b"],
            "experiment_date": ["2026-07-15", "2026-07-15"],
            "cycle_id": ["c1", "c1"],
            "eligibility_status": ["eligible", "excluded"],
            "frost_development_grid_coverage": [1.0, 0.0],
            "eligible_feature_count": [2, 0],
            "total_candidate_count": [2, 2],
            "exclusion_reason": ["", "cycle_invalid"],
        }
    )
    metrics = pd.DataFrame(
        {
            "experiment_id": ["exp_a", "exp_a", "exp_b", "exp_b"],
            "experiment_date": ["2026-07-15"] * 4,
            "cycle_id": ["c1"] * 4,
            "feature": ["signal_a", "signal_b", "signal_a", "signal_b"],
            "reference_source": [
                "configured_baseline",
                "auto_cycle_initial_reference",
                "unavailable",
                "unavailable",
            ],
        }
    )
    future = _future().assign(experiment_id="exp_a")
    future = pd.concat([future, future.assign(experiment_id="exp_b")], ignore_index=True)

    figure = plot_evidence_coverage(
        eligibility, metrics, future, _policy(), _settings()
    )

    details = [text.get_text() for text in figure.axes[1].texts]
    assert len(details) == 2
    assert "cfg=0.50 auto=0.50 unavailable=0.00" in details[0]
    assert "cfg=0.00 auto=0.00 unavailable=1.00" in details[1]
    labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
    assert labels == ["exp_a · 0715 · c1", "exp_b · 0715 · c1"]
    plt.close(figure)
