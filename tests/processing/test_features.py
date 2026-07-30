from __future__ import annotations

import pandas as pd

from frost_analysis.data.registry import FeatureSpec
from frost_analysis.processing.features import engineer_features


def test_window_support_keeps_observed_and_available_coverage_separate() -> None:
    spec = FeatureSpec(
        feature_id="signal",
        canonical_name="signal",
        raw_source=None,
        meaning_zh="",
        physical_family="test",
        source_type="measured",
        unit="u",
        formula="",
        data_role="X",
        availability="current_history",
        deployment_status="confirmed",
        confidence="high",
        primary_or_validation="primary",
        analysis_enabled=True,
        notes="",
    )
    frame = pd.DataFrame(
        {
            "sensor_time": pd.date_range("2026-07-15", periods=4, freq="30s"),
            "cycle_id": "cycle_001",
            "cycle_quality": "complete",
            "stage": "frost_development",
            "cycle_phase": [0.0, 0.33, 0.66, 1.0],
            "cycle_time_s": [0.0, 30.0, 60.0, 90.0],
            "analysis_bin_available": True,
            "signal": [0.0, 1.0, 2.0, 3.0],
            "signal__observed": [True, False, True, True],
            "signal__imputed": [False, True, False, False],
        }
    )

    result = engineer_features(
        frame,
        {"signal": spec},
        windows_minutes=[1],
        minimum_coverage=0.7,
        minimum_observed_coverage=0.5,
        minimum_available_coverage=0.95,
        maximum_imputed_fraction=0.4,
        maximum_raw_gap_seconds=60,
    ).frame

    row = result.iloc[-1]
    assert row["signal__window_1m__observed_coverage"] == 2 / 3
    assert row["signal__window_1m__available_coverage"] == 1.0
    assert row["signal__window_1m__imputed_fraction"] == 1 / 3
    assert bool(row["signal__window_1m__valid"])
