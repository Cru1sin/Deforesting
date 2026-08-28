from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frost_analysis.evidence import build_evidence

from .conftest import frame_for, settings, write_dataset


def test_future_summary_is_date_balanced_median_of_cycle_medians(tmp_path: Path) -> None:
    positive = frame_for(
        elapsed=(0, 300, 600, 900, 1200),
        heating_capacity=(0, 1, 3, 6, 10),
    )
    negative = frame_for(
        elapsed=(0, 300, 600, 900, 1200),
        heating_capacity=(0, 4, 7, 9, 10),
    )
    loader = write_dataset(
        tmp_path / "dataset",
        [
            ("c1", "2026-07-01", "valid", positive),
            ("c2", "2026-07-01", "valid", negative),
            ("c3", "2026-07-02", "valid", positive),
        ],
    )

    bundle = build_evidence(
        loader,
        settings(
            targets=("heating_capacity",),
            minimum_valid_pairs=2,
            minimum_pair_coverage=1.0,
        ),
    )

    summary = bundle.future_horizon_summary.loc[
        bundle.future_horizon_summary["horizon_minutes"].eq(5)
    ].iloc[0]
    assert summary["effect"] == 0.5
    assert summary["degradation_support"] == -0.5
    assert summary["valid_cycle_count"] == 3
    assert summary["valid_date_count"] == 2
    assert summary["aggregation_method"] == "date_balanced_median_of_cycle_medians_v1"
    profile = bundle.feature_profile.iloc[0]
    assert profile["primary_future_degradation_support"] == -0.5
    assert profile["primary_future_valid_cycle_count"] == 3
    assert profile["primary_future_valid_date_count"] == 2


def test_summary_has_explicit_zero_valid_dates_status(tmp_path: Path) -> None:
    frame = frame_for(
        elapsed=(0, 60, 120, 180),
        heating_capacity=(np.nan, np.nan, np.nan, np.nan),
    )
    loader = write_dataset(tmp_path / "dataset", [("c1", "2026-07-01", "valid", frame)])

    bundle = build_evidence(
        loader,
        settings(
            targets=("heating_capacity",),
            minimum_valid_pairs=2,
            minimum_pair_coverage=1.0,
        ),
    )

    summary = bundle.future_horizon_summary.loc[
        bundle.future_horizon_summary["horizon_minutes"].eq(5)
    ].iloc[0]
    assert summary["metric_status"] == "unavailable"
    assert summary["exclusion_reason"] == "no_valid_dates"
    assert summary["valid_cycle_count"] == 0
    assert summary["valid_date_count"] == 0


def test_date_balanced_counts_unique_cycles_not_rows() -> None:
    from frost_analysis.evidence.summary import date_balanced_median

    frame = pd.DataFrame(
        {
            "cycle_name": ["c1", "c1", "c2"],
            "experiment_date": ["2026-07-01"] * 3,
            "metric_status": ["available"] * 3,
            "value": [0.2, 0.4, 0.8],
        }
    )

    effect, cycle_count, date_count = date_balanced_median(frame, "value")

    assert effect == pytest.approx(0.55)
    assert cycle_count == 2
    assert date_count == 1


def test_pair_similarity_requires_minimum_common_true_time_points() -> None:
    from frost_analysis.evidence.summary import feature_pair_similarity

    pair_inputs = [
        (
            "c1",
            "2026-07-01",
            {
                "feature_a": {0.0: 1.0, 60.0: 2.0},
                "feature_b": {0.0: 3.0, 60.0: 4.0},
            },
        )
    ]

    similarity = feature_pair_similarity(
        pair_inputs,
        (("feature_a", "increase"), ("feature_b", "decrease")),
        settings(minimum_feature_points=3),
    )

    row = similarity.iloc[0]
    assert row["metric_status"] == "unavailable"
    assert row["valid_cycle_count"] == 0
    assert row["valid_date_count"] == 0
