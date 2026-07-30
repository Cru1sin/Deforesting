from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frost_analysis.analysis import analyze
from frost_analysis.config import Config
from frost_analysis.validation import validate_analysis, validate_prepared, validate_processed


def _config(root: Path) -> Config:
    return Config(
        project_root=root,
        experiment_id="exp_test",
        experiment_date="2026-07-15",
        input_dir=root / "data",
        channels_path=root / "channels.yaml",
        sensor_globs=("*.xls",),
        image_extensions=(".jpg",),
        camera_mapping_file="IPlocation.yaml",
        cycles={},
        process={"resample_interval_seconds": 10},
        analysis={
            "performance_target": "heating_capacity__baseline_residual",
            "future_horizon_minutes": 10,
            "reset_pre_window_minutes": 5,
            "minimum_valid_cycles": 3,
            "minimum_absolute_trend_effect": 0.3,
            "minimum_direction_consistency": 0.7,
            "maximum_context_association": 0.8,
        },
    )


def _channels() -> dict[str, dict[str, object]]:
    return {
        "signal": {
            "kind": "continuous",
            "role": "sensor",
            "analysis_candidate": True,
            "expected_frost_direction": "decrease",
        },
        "ambient_temperature": {
            "kind": "continuous",
            "role": "context",
            "analysis_candidate": False,
        },
    }


def _analysis_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    cycles: list[dict[str, object]] = []
    for number in range(3):
        start = pd.Timestamp("2026-07-15") + pd.Timedelta(hours=number)
        cycle_id = f"cycle_{number + 1:03d}"
        for point, progress in enumerate((0.0, 1 / 3, 2 / 3, 1.0)):
            timestamp = start + pd.Timedelta(minutes=10 + point * 10)
            rows.append(
                {
                    "experiment_id": "exp_test",
                    "experiment_date": "2026-07-15",
                    "timestamp": timestamp,
                    "cycle_id": cycle_id,
                    "cycle_stage": "frost_development",
                    "cycle_status": "valid",
                    "cycle_progress": progress,
                    "signal__baseline_residual": -float(point),
                    "ambient_temperature": 0.0,
                    "heating_capacity__baseline_residual": -float(point + 1),
                    "signal__baseline_status": "accepted",
                }
            )
        rows.append(
            {
                "experiment_id": "exp_test",
                "experiment_date": "2026-07-15",
                "timestamp": start + pd.Timedelta(minutes=5),
                "cycle_id": cycle_id,
                "cycle_stage": "recovery",
                "cycle_status": "valid",
                "cycle_progress": np.nan,
                "signal__baseline_residual": 0.0,
                "ambient_temperature": 0.0,
                "heating_capacity__baseline_residual": 0.0,
                "signal__baseline_status": "accepted",
            }
        )
        cycles.append(
            {
                "experiment_id": "exp_test",
                "experiment_date": "2026-07-15",
                "cycle_id": cycle_id,
                "cycle_status": "valid",
                "heating_start": start,
                "stable_heating_start": start + pd.Timedelta(minutes=10),
                "defrost_start": start + pd.Timedelta(minutes=50),
                "defrost_end": start + pd.Timedelta(minutes=55),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(cycles)


def test_analysis_emits_one_evidence_row_per_experiment_and_candidate() -> None:
    frame, cycles = _analysis_frame()

    evidence = analyze(frame, cycles, _config(Path("/tmp")), _channels())

    assert list(evidence.columns) == [
        "experiment_id",
        "experiment_date",
        "channel",
        "trend_cycle_count",
        "reset_pair_count",
        "future_cycle_count",
        "context_cycle_count",
        "trend_effect",
        "direction_consistency",
        "reset_effect",
        "future_performance_effect",
        "max_abs_context_spearman",
        "decision",
        "reason",
    ]
    assert len(evidence) == 1
    assert evidence.loc[0, "trend_cycle_count"] == 3
    assert evidence.loc[0, "decision"] == "trend_supported_candidate"
    assert "valid_cycle_count" not in evidence
    assert "rank" not in evidence
    assert "weighted_score" not in evidence


def test_future_matching_never_crosses_cycle_boundary() -> None:
    frame = pd.DataFrame(
        {
            "experiment_id": ["exp_test", "exp_test"],
            "experiment_date": ["2026-07-15"] * 2,
            "timestamp": pd.to_datetime(["2026-07-15 00:00:00", "2026-07-15 00:10:00"]),
            "cycle_id": ["cycle_001", "cycle_002"],
            "cycle_stage": ["frost_development"] * 2,
            "cycle_progress": [0.2, 0.2],
            "signal__baseline_residual": [1.0, 2.0],
            "heating_capacity__baseline_residual": [np.nan, 3.0],
        }
    )
    cycles = pd.DataFrame(
        {
            "experiment_id": ["exp_test", "exp_test"],
            "experiment_date": ["2026-07-15"] * 2,
            "cycle_id": ["cycle_001", "cycle_002"],
            "cycle_status": ["valid", "valid"],
            "heating_start": pd.to_datetime(
                ["2026-07-15 00:00:00", "2026-07-15 00:10:00"]
            ),
            "stable_heating_start": pd.to_datetime(
                ["2026-07-15 00:00:00", "2026-07-15 00:10:00"]
            ),
            "defrost_start": pd.to_datetime(["2026-07-15 00:05:00", "2026-07-15 00:15:00"]),
            "defrost_end": pd.to_datetime(["2026-07-15 00:06:00", "2026-07-15 00:16:00"]),
        }
    )

    evidence = analyze(frame, cycles, _config(Path("/tmp")), _channels())

    assert evidence.loc[0, "future_cycle_count"] == 0


def test_structural_validators_reject_duplicate_or_forbidden_contracts() -> None:
    prepared = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "timestamp": pd.to_datetime(["2026-07-15"]),
            "cycle_id": ["cycle_001"],
            "cycle_stage": ["frost_development"],
            "cycle_progress": [0.0],
        }
    )
    summary = pd.DataFrame({"experiment_id": ["exp_test"], "cycle_id": ["cycle_001"]})
    validate_prepared(prepared, summary)

    processed = prepared.assign(
        cycle_status="valid",
        channel__imputed=pd.Series([False], dtype=bool),
        channel__baseline_status="accepted",
        channel__baseline=1.0,
    )
    validate_processed(processed, summary)

    evidence = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "channel": ["signal"],
            "trend_cycle_count": [3],
            "reset_pair_count": [0],
            "future_cycle_count": [0],
            "context_cycle_count": [0],
            "trend_effect": [-1.0],
            "direction_consistency": [1.0],
            "reset_effect": [np.nan],
            "future_performance_effect": [np.nan],
            "max_abs_context_spearman": [np.nan],
            "decision": ["trend_supported_candidate"],
            "reason": ["trend_evidence_meets_threshold"],
        }
    )
    validate_analysis(evidence)

    with pytest.raises(ValueError, match="weighted score"):
        validate_analysis(evidence.assign(weighted_score=1.0))
