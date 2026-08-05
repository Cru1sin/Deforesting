from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frost_analysis.evidence.core import _candidate_features
from frost_analysis.evidence.figures import plot_cycle_progress
from frost_analysis.evidence.summary import _date_balanced

from .conftest import frame_for, settings, write_dataset


@pytest.mark.parametrize(
    "channel",
    [
        {
            "analysis_candidate": True,
            "expected_frost_direction": "increase",
        },
        {
            "analysis_candidate": True,
            "expected_frost_direction": "decrease",
            "role": "performance",
        },
    ],
)
def test_candidate_feature_rejects_target_or_performance_candidate(
    channel: dict[str, object],
) -> None:
    registry = {"channels": {"heating_capacity": channel}}

    with pytest.raises(ValueError, match="candidate"):
        _candidate_features(
            registry,
            settings(targets=("heating_capacity",), horizons=(1,)),
        )


def test_observed_mask_is_the_single_quality_rule() -> None:
    from frost_analysis.evidence.metrics import observed_mask

    frame = pd.DataFrame(
        {
            "signal__baseline_residual": [1.0, np.nan, np.inf, -np.inf, 5.0],
            "signal__imputed": [False, False, False, False, pd.NA],
        }
    )

    result = observed_mask(frame, "signal__baseline_residual")

    assert result.tolist() == [True, False, False, False, True]


def test_date_balanced_counts_unique_cycles_not_rows() -> None:
    frame = pd.DataFrame(
        {
            "cycle_name": ["c1", "c1", "c2"],
            "experiment_date": ["2026-07-01"] * 3,
            "metric_status": ["available"] * 3,
            "value": [0.2, 0.4, 0.8],
        }
    )

    effect, cycle_count, date_count = _date_balanced(frame, "value")

    assert effect == pytest.approx(0.55)
    assert cycle_count == 2
    assert date_count == 1


def test_figure_one_excludes_imputed_and_nonfinite_target_values(tmp_path: Path) -> None:
    frame = frame_for()
    frame.loc[2, "heating_capacity__imputed"] = True
    frame.loc[3, "heating_capacity__baseline_residual"] = np.inf
    loader = write_dataset(tmp_path / "dataset", [("c1", "2026-07-01", "valid", frame)])

    figure = plot_cycle_progress(
        loader,
        settings(targets=("heating_capacity",), horizons=(1,)),
    )

    assert len(figure.axes[0].lines) == 1
    assert len(figure.axes[0].lines[0].get_xdata()) == 4


def test_figure_three_labels_valid_cycle_and_date_counts() -> None:
    from frost_analysis.evidence.figures import plot_future_horizon_summary

    summary = pd.DataFrame(
        {
            "feature": ["feature_a"],
            "target": ["heating_capacity"],
            "horizon_minutes": [1],
            "effect": [0.4],
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
