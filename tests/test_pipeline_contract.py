from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frost_analysis.analysis import analyze
from frost_analysis.config import Config
from frost_analysis.validation import validate_analysis, validate_prepared, validate_processed


def _config(root: Path, **analysis: object) -> Config:
    values = {
        "performance_target": "heating_capacity__baseline_residual",
        "future_horizon_minutes": 10,
        "minimum_valid_cycles": 2,
        "minimum_trend_effect": 0.3,
        "minimum_direction_consistency": 0.7,
        "minimum_points_per_cycle": 4,
        **analysis,
    }
    return Config(
        project_root=root,
        experiment_id="exp_test",
        experiment_date="2026-07-15",
        input_dir=root / "data",
        channels_path=root / "channels.yaml",
        sensor_globs=("*.xls",),
        image_extensions=(".jpg",),
        timestamp_column="时间",
        expected_sensor_interval_seconds=1,
        image_match_tolerance_seconds=2,
        edf_pair_tolerance_seconds=1.0,
        cycles={},
        process={"resample_interval_seconds": 10},
        analysis=values,
        camera_roles={},
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
    for number in range(2):
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
                    "signal__imputed": False,
                    "ambient_temperature": 0.0,
                    "ambient_temperature__imputed": False,
                    "heating_capacity__baseline_residual": -float(point + 1),
                    "heating_capacity__imputed": False,
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
            }
        )
        cycles.append(
            {
                "experiment_id": "exp_test",
                "experiment_date": "2026-07-15",
                "cycle_id": cycle_id,
                "cycle_status": "valid",
                "baseline_status": "available",
                "heating_start": start,
                "stable_heating_start": start + pd.Timedelta(minutes=10),
                "defrost_start": start + pd.Timedelta(minutes=50),
                "defrost_end": start + pd.Timedelta(minutes=55),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(cycles)


def test_analysis_emits_explicit_evidence_and_disabled_reset_fields() -> None:
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
        "reset_evidence_status",
        "reset_evidence_reason",
        "future_performance_association",
        "median_max_abs_context_spearman",
        "decision",
        "reason",
    ]
    assert len(evidence) == 1
    assert evidence.loc[0, "trend_cycle_count"] == 2
    assert evidence.loc[0, "reset_pair_count"] == 0
    assert pd.isna(evidence.loc[0, "reset_effect"])
    assert evidence.loc[0, "reset_evidence_status"] == "not_evaluated"
    assert evidence.loc[0, "reset_evidence_reason"] == "independent_reference_unavailable"
    assert evidence.loc[0, "decision"] == "trend_supported_candidate"
    assert "valid_cycle_count" not in evidence
    assert "rank" not in evidence


def test_future_matching_never_crosses_cycle_or_stage_boundary() -> None:
    frame = pd.DataFrame(
        {
            "experiment_id": ["exp_test"] * 4,
            "experiment_date": ["2026-07-15"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2026-07-15 00:00:00",
                    "2026-07-15 00:10:00",
                    "2026-07-15 00:20:00",
                    "2026-07-15 00:30:00",
                ]
            ),
            "cycle_id": ["cycle_001", "cycle_002", "cycle_001", "cycle_002"],
            "cycle_stage": ["frost_development"] * 4,
            "cycle_status": ["valid"] * 4,
            "cycle_progress": [0.2] * 4,
                "signal__baseline_residual": [1.0, 2.0, 3.0, 4.0],
                "signal__imputed": [False] * 4,
                "heating_capacity__baseline_residual": [np.nan, 3.0, 5.0, 7.0],
                "heating_capacity__imputed": [False] * 4,
        }
    )
    cycles = pd.DataFrame(
        {
            "experiment_id": ["exp_test", "exp_test"],
            "experiment_date": ["2026-07-15"] * 2,
            "cycle_id": ["cycle_001", "cycle_002"],
            "cycle_status": ["valid", "valid"],
            "baseline_status": ["available", "available"],
        }
    )

    evidence = analyze(frame, cycles, _config(Path("/tmp")), _channels())

    assert evidence.loc[0, "future_cycle_count"] == 0


def test_decrease_candidate_with_wrong_raw_direction_is_not_supported() -> None:
    frame, cycles = _analysis_frame()
    frame["signal__baseline_residual"] = frame["cycle_progress"]

    evidence = analyze(frame, cycles, _config(Path("/tmp")), _channels())

    assert evidence.loc[0, "trend_effect"] < 0
    assert evidence.loc[0, "direction_consistency"] == 0
    assert evidence.loc[0, "decision"] == "partial_evidence"


def test_context_uses_per_cycle_maximum_then_median() -> None:
    frame, cycles = _analysis_frame()
    frame.loc[frame["cycle_id"].eq("cycle_001"), "ambient_temperature"] = [
        0.0,
        1.0,
        0.0,
        1.0,
        0.5,
    ]
    frame.loc[frame["cycle_id"].eq("cycle_002"), "ambient_temperature"] = [0.0, 1.0, 0.0, 1.0, 0.5]
    evidence = analyze(frame, cycles, _config(Path("/tmp")), _channels())

    assert evidence.loc[0, "context_cycle_count"] == 2
    assert 0.4 < evidence.loc[0, "median_max_abs_context_spearman"] < 0.9


def test_analysis_excludes_imputed_candidate_future_and_context_points() -> None:
    frame, cycles = _analysis_frame()
    frame.loc[
        frame["cycle_id"].eq("cycle_002") & frame["cycle_progress"].notna(), "signal__imputed"
    ] = True
    cycle_2_frost = frame["cycle_id"].eq("cycle_002") & frame["cycle_progress"].notna()
    frame.loc[cycle_2_frost, "heating_capacity__imputed"] = True
    frame.loc[cycle_2_frost, "ambient_temperature__imputed"] = True
    frame.loc[frame["cycle_id"].eq("cycle_001"), "ambient_temperature"] = frame.loc[
        frame["cycle_id"].eq("cycle_001"), "cycle_progress"
    ].fillna(0.0)

    evidence = analyze(
        frame,
        cycles,
        _config(Path("/tmp"), minimum_points_per_cycle=3),
        _channels(),
    )

    assert evidence.loc[0, "trend_cycle_count"] == 1
    assert evidence.loc[0, "future_cycle_count"] == 1
    assert evidence.loc[0, "context_cycle_count"] == 1


def test_analysis_requires_quality_columns_for_used_values() -> None:
    frame, cycles = _analysis_frame()

    bad_config = _config(Path("/tmp"), performance_target="missing_target__baseline_residual")
    with pytest.raises(ValueError, match="missing_target__baseline_residual"):
        analyze(frame, cycles, bad_config, _channels())
    with pytest.raises(ValueError, match="missing_target__baseline_residual"):
        analyze(frame, cycles, bad_config, {})

    with pytest.raises(ValueError, match="signal__baseline_residual"):
        analyze(
            frame.drop(columns=["signal__baseline_residual"]),
            cycles,
            _config(Path("/tmp")),
            _channels(),
        )

    with pytest.raises(ValueError, match="signal__imputed"):
        analyze(frame.drop(columns=["signal__imputed"]), cycles, _config(Path("/tmp")), _channels())

    with pytest.raises(ValueError, match="heating_capacity__imputed"):
        analyze(
            frame.drop(columns=["heating_capacity__imputed"]),
            cycles,
            _config(Path("/tmp")),
            _channels(),
        )


def test_context_association_does_not_override_supported_trend() -> None:
    frame, cycles = _analysis_frame()
    frame["ambient_temperature"] = frame["cycle_progress"].fillna(0.0)

    evidence = analyze(frame, cycles, _config(Path("/tmp")), _channels())

    assert evidence.loc[0, "median_max_abs_context_spearman"] == 1.0
    assert evidence.loc[0, "decision"] == "trend_supported_candidate"


def test_analysis_excludes_incomplete_cycles_from_trend_evidence() -> None:
    frame, cycles = _analysis_frame()
    frame.loc[frame["cycle_id"].eq("cycle_002"), "cycle_status"] = "incomplete"
    cycles.loc[cycles["cycle_id"].eq("cycle_002"), "cycle_status"] = "incomplete"

    evidence = analyze(frame, cycles, _config(Path("/tmp")), _channels())

    assert evidence.loc[0, "trend_cycle_count"] == 1


def test_dataset_analysis_can_use_explicitly_selected_cycle_with_nonvalid_source_status() -> None:
    frame, cycles = _analysis_frame()
    frame.loc[frame["cycle_id"].eq("cycle_002"), "cycle_status"] = "incomplete"
    cycles.loc[cycles["cycle_id"].eq("cycle_002"), "cycle_status"] = "incomplete"

    evidence = analyze(
        frame,
        cycles,
        _config(Path("/tmp")),
        _channels(),
        respect_cycle_status=False,
    )

    assert evidence.loc[0, "trend_cycle_count"] == 2


def test_structural_validators_reject_invalid_fields() -> None:
    prepared = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "timestamp": pd.to_datetime(["2026-07-15"]),
            "cycle_id": ["cycle_001"],
            "cycle_stage": ["frost_development"],
            "cycle_status": ["not_a_status"],
            "cycle_progress": [0.0],
        }
    )
    summary = pd.DataFrame({"experiment_id": ["exp_test"], "cycle_id": ["cycle_001"]})
    with pytest.raises(ValueError, match="cycle_status"):
        validate_prepared(prepared, summary)

    valid_prepared = prepared.assign(cycle_status="valid")
    extra_summary = pd.concat(
        [summary, pd.DataFrame({"experiment_id": ["exp_test"], "cycle_id": ["cycle_002"]})],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="cycle keys"):
        validate_prepared(valid_prepared, extra_summary)

    incomplete_summary = pd.concat(
        [
            summary.assign(cycle_status="valid"),
            pd.DataFrame(
                {
                    "experiment_id": ["exp_test"],
                    "cycle_id": ["cycle_002"],
                    "cycle_status": ["incomplete"],
                }
            ),
        ],
        ignore_index=True,
    )
    validate_prepared(valid_prepared, incomplete_summary)

    invalid_summary = incomplete_summary.copy()
    invalid_summary.loc[invalid_summary["cycle_id"].eq("cycle_002"), "cycle_status"] = "invalid"
    with pytest.raises(ValueError, match="cycle keys"):
        validate_prepared(valid_prepared, invalid_summary)

    missing_summary = valid_prepared.assign(cycle_id="cycle_002")
    with pytest.raises(ValueError, match="cycle keys"):
        validate_prepared(missing_summary, summary)

    null_key = valid_prepared.assign(cycle_id=pd.NA)
    with pytest.raises(ValueError, match="cycle keys"):
        validate_prepared(null_key, summary)

    processed = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "timestamp": pd.to_datetime(["2026-07-15"]),
            "cycle_id": ["cycle_001"],
            "cycle_stage": ["frost_development"],
            "cycle_status": ["valid"],
            "cycle_progress": [0.0],
            "cycle_elapsed_seconds": [0.0],
        }
    )
    processed_summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test", "exp_test"],
            "cycle_id": ["cycle_001", "partial_001"],
            "baseline_status": ["not_applicable", "not_applicable"],
            "baseline_failure_reason": ["cycle_not_valid", "cycle_not_valid"],
        }
    )
    validate_processed(processed, processed_summary)
    with pytest.raises(ValueError, match="Processed cycle keys"):
        validate_processed(processed.assign(cycle_id="cycle_002"), processed_summary)

    negative_elapsed = valid_prepared.assign(cycle_elapsed_seconds=-1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        validate_prepared(negative_elapsed, summary)

    evidence = pd.DataFrame(columns=[
        "experiment_id", "experiment_date", "channel", "trend_cycle_count",
        "reset_pair_count", "future_cycle_count", "context_cycle_count", "trend_effect",
        "direction_consistency", "reset_effect", "reset_evidence_status",
        "reset_evidence_reason", "future_performance_association",
        "median_max_abs_context_spearman", "decision", "reason",
    ])
    validate_analysis(evidence)
    with pytest.raises(ValueError, match="weighted score"):
        validate_analysis(evidence.assign(weighted_score=1.0))


def test_processed_validator_accepts_partial_fallback_rows() -> None:
    processed = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "timestamp": pd.to_datetime(["2026-07-15"]),
            "cycle_id": ["partial_001"],
            "cycle_stage": ["partial"],
            "cycle_status": ["incomplete"],
            "cycle_progress": [np.nan],
            "cycle_elapsed_seconds": [np.nan],
            "temperature": [1.0],
            "temperature__imputed": pd.Series([False], dtype=bool),
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "cycle_id": ["partial_001"],
            "baseline_status": ["not_applicable"],
            "baseline_failure_reason": ["cycle_not_valid"],
        }
    )

    validate_processed(processed, summary)


def test_structural_validators_accept_explicit_partial_cycle_status() -> None:
    prepared = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "timestamp": pd.to_datetime(["2026-07-15"]),
            "cycle_id": ["partial_001"],
            "cycle_stage": ["partial"],
            "cycle_status": ["partial"],
            "cycle_progress": [np.nan],
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "cycle_id": ["partial_001"],
            "cycle_status": ["partial"],
            "baseline_status": ["not_applicable"],
            "baseline_failure_reason": ["cycle_not_valid"],
        }
    )

    validate_prepared(prepared, summary)
    validate_processed(
        prepared.assign(value=[1.0], value__imputed=pd.Series([False], dtype=bool)),
        summary,
    )
