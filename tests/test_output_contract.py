from __future__ import annotations

from pathlib import Path

import pandas as pd

from frost_analysis.data.cycles import build_cycle_summary


def test_stage_outputs_are_four_top_level_artifacts() -> None:
    root = Path(__file__).resolve().parents[1] / "outputs" / "0715"
    assert {
        "prepared_data.parquet",
        "processed_data.parquet",
        "cycle_summary.csv",
        "correlation_results.csv",
    } <= {path.name for path in root.iterdir()}


def test_cycle_summary_reports_rgb_gaps_and_multimodal_quality() -> None:
    cycles = pd.DataFrame(
        [
            {
                "cycle_id": "cycle_001",
                "quality_flag": "complete",
                "heating_start": pd.Timestamp("2026-07-15 10:00:00"),
                "stable_heating_start": pd.Timestamp("2026-07-15 10:00:30"),
                "defrost_start": pd.Timestamp("2026-07-15 10:03:00"),
                "defrost_end": pd.Timestamp("2026-07-15 10:03:30"),
                "clean_start": pd.Timestamp("2026-07-15 10:00:00"),
                "clean_end": pd.Timestamp("2026-07-15 10:00:30"),
                "maximum_gap_seconds": 1.0,
                "exclusion_reason": "",
            }
        ]
    )
    frame = pd.DataFrame(
        {
            "sensor_time": pd.to_datetime(
                ["2026-07-15 10:00:00", "2026-07-15 10:03:30"]
            ),
            "cycle_id": ["cycle_001", "cycle_001"],
        }
    )
    multiview = pd.DataFrame(
        {
            "group_time": pd.to_datetime(
                [
                    "2026-07-15 10:00:00",
                    "2026-07-15 10:00:30",
                    "2026-07-15 10:02:00",
                ]
            ),
            "camera_count": [6, 2, 6],
            "all_cameras_present": [True, False, True],
        }
    )
    result = build_cycle_summary(
        cycles, frame, multiview, date="0715", gap_warning_factor=3.0
    )
    row = result.iloc[0]
    assert row["rgb_group_count"] == 3
    assert row["rgb_complete_group_count"] == 2
    assert row["rgb_max_gap_seconds"] == 90.0
    assert "10:00:30" in row["rgb_interruption_intervals"]
    assert row["multimodal_quality"] == "sensor_valid_rgb_incomplete"
