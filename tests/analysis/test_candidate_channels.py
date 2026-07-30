from __future__ import annotations

import pandas as pd

from frost_analysis.analysis.screening import screen_candidate_channels


def test_candidate_screening_returns_one_row_per_channel_without_weighted_score() -> None:
    times = pd.date_range("2026-07-15", periods=12, freq="min")
    frame = pd.DataFrame(
        {
            "sensor_time": list(times) * 2,
            "cycle_id": ["cycle_001"] * 12 + ["cycle_002"] * 12,
            "cycle_quality": ["complete"] * 24,
            "stage": ["frost_development"] * 24,
            "cycle_phase": list(range(12)) * 2,
            "heating_mode": [True] * 24,
            "channel_a": list(range(12)) * 2,
            "channel_a__baseline_offset": list(range(12)) * 2,
            "heating_capacity": [10 - i * 0.1 for i in range(12)] * 2,
            "power_total": [3.0] * 24,
            "cop": [3.3 - i * 0.03 for i in range(12)] * 2,
        }
    )
    cycles = pd.DataFrame(
        {
            "cycle_id": ["cycle_001", "cycle_002"],
            "quality_flag": ["complete", "complete"],
            "defrost_start": [times[10], times[10]],
            "clean_end": [times[1], times[1]],
            "heating_start": [times[0], times[0]],
        }
    )
    registry = pd.DataFrame(
        [
            {
                "feature_id": "channel_a",
                "canonical_name": "channel_a",
                "physical_family": "evaporator_response",
                "analysis_enabled": True,
                "source_type": "measured",
            }
        ]
    )
    evidence = screen_candidate_channels(frame, cycles, registry, {})
    assert list(evidence["canonical_name"]) == ["channel_a"]
    assert "candidate_score" not in evidence
    assert "trend_direction" in evidence
