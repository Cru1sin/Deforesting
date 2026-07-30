from __future__ import annotations

import pandas as pd

from frost_analysis.data.registry import FeatureSpec
from frost_analysis.processing.resample import resample_observations


def _spec(
    name: str,
    *,
    missing_policy: str = "linear",
    resample_method: str = "mean",
) -> FeatureSpec:
    return FeatureSpec(
        feature_id=name,
        canonical_name=name,
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
        missing_policy=missing_policy,
        resample_method=resample_method,
    )


def test_resample_aligns_to_each_cycle_start_and_keeps_empty_bins() -> None:
    start = pd.Timestamp("2026-07-15 00:00:05")
    frame = pd.DataFrame(
        {
            "timestamp": [
                start,
                start + pd.Timedelta(seconds=10),
                start + pd.Timedelta(seconds=30),
            ],
            "cycle_id": "cycle_001",
            "cycle_stage": "frost_development",
            "cycle_status": "valid",
            "signal": [1.0, 2.0, 4.0],
            "signal__source_state": ["observed", "observed", "observed"],
        }
    )

    result = resample_observations(
        frame,
        {"signal": _spec("signal")},
        interval_seconds=10,
    )

    assert result["timestamp"].tolist() == [
        start,
        start + pd.Timedelta(seconds=10),
        start + pd.Timedelta(seconds=20),
        start + pd.Timedelta(seconds=30),
    ]
    assert pd.isna(result.loc[2, "signal"])
    assert result.loc[2, "source_sample_count"] == 0
    assert result.loc[2, "signal__source_state"] == "not_sampled"


def test_resample_does_not_cross_cycle_or_stage_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-07-15 00:00:00",
                    "2026-07-15 00:00:10",
                    "2026-07-15 00:01:00",
                ]
            ),
            "cycle_id": ["cycle_001", "cycle_001", "cycle_002"],
            "cycle_stage": ["frost_development", "frost_development", "frost_development"],
            "cycle_status": "valid",
            "signal": [1.0, 2.0, 9.0],
        }
    )

    result = resample_observations(
        frame,
        {"signal": _spec("signal")},
        interval_seconds=10,
    )

    assert result.groupby(["cycle_id", "cycle_stage"]).size().to_dict() == {
        ("cycle_001", "frost_development"): 2,
        ("cycle_002", "frost_development"): 1,
    }


def test_resample_does_not_process_derived_channels() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-15", periods=2, freq="10s"),
            "cycle_id": "cycle_001",
            "cycle_stage": "frost_development",
            "cycle_status": "valid",
            "source": [2.0, 4.0],
            "derived": [1.0, 2.0],
        }
    )
    derived = _spec("derived")
    derived = FeatureSpec(**{**derived.__dict__, "data_kind": "derived", "resample_method": "none"})

    result = resample_observations(
        frame,
        {"source": _spec("source"), "derived": derived},
        interval_seconds=10,
    )

    assert "source" in result
    assert "derived" not in result


def test_resample_preserves_non_numeric_image_paths() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-15", periods=2, freq="10s"),
            "cycle_id": "cycle_001",
            "cycle_stage": "frost_development",
            "cycle_status": "valid",
            "image_front_path": ["front-a.jpg", None],
        }
    )

    result = resample_observations(frame, {}, interval_seconds=10)

    assert result.loc[0, "image_front_path"] == "front-a.jpg"
