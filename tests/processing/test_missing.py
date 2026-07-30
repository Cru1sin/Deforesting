from __future__ import annotations

import pandas as pd

from frost_analysis.data.registry import FeatureSpec
from frost_analysis.processing.missing import apply_missing_policy, assess_missing_data


def _spec(name: str, policy: str) -> FeatureSpec:
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
        missing_policy=policy,
        resample_method="mean",
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-15", periods=5, freq="10s"),
            "cycle_id": "cycle_001",
            "cycle_stage": "frost_development",
            "signal": [0.0, float("nan"), float("nan"), 30.0, 40.0],
            "signal__source_state": ["observed", "missing", "missing", "observed", "observed"],
            "control": [1.0, float("nan"), float("nan"), float("nan"), 2.0],
            "control__source_state": ["observed", "missing", "missing", "missing", "observed"],
            "protected": [10.0, float("nan"), float("nan"), 40.0, 50.0],
            "protected__source_state": ["observed", "missing", "missing", "observed", "observed"],
        }
    )


def test_assess_missing_data_does_not_modify_values() -> None:
    frame = _frame()
    original = frame.copy(deep=True)

    report = assess_missing_data(
        frame,
        {
            "signal": _spec("signal", "linear"),
            "control": _spec("control", "forward_fill"),
            "protected": _spec("protected", "none"),
        },
        {"group_columns": ["cycle_id", "cycle_stage"]},
    )

    pd.testing.assert_frame_equal(frame, original)
    assert (
        report.channel_summary.loc[
            report.channel_summary["channel"].eq("signal"), "observed_count"
        ].item()
        == 3
    )


def test_apply_missing_policy_uses_real_time_and_keeps_protected_nan() -> None:
    frame = _frame()
    specs = {
        "signal": _spec("signal", "linear"),
        "control": _spec("control", "forward_fill"),
        "protected": _spec("protected", "none"),
    }
    report = assess_missing_data(frame, specs, {"group_columns": ["cycle_id", "cycle_stage"]})

    result = apply_missing_policy(
        frame,
        report,
        specs,
        {
            "group_columns": ["cycle_id", "cycle_stage"],
            "continuous": {
                "method": "time_linear",
                "maximum_bracketing_gap_seconds": 30,
                "require_both_sides": True,
            },
            "control": {"method": "forward_fill", "maximum_age_seconds": 25},
            "protected": {"method": "none"},
        },
    )

    assert result.data.loc[1, "signal"] == 10.0
    assert result.data.loc[2, "signal"] == 20.0
    assert result.data.loc[1, "signal__imputed"]
    assert result.data.loc[2, "signal__imputed"]
    assert not bool(result.data.loc[1, "signal__observed"])
    assert result.data.loc[1, "control"] == 1.0
    assert result.data.loc[2, "control"] == 1.0
    assert pd.isna(result.data.loc[3, "control"])
    assert pd.isna(result.data.loc[1, "protected"])
    assert result.cycle_summary_updates.loc[0, "signal__observed_coverage"] == 0.6
    assert result.cycle_summary_updates.loc[0, "signal__available_coverage"] == 1.0


def test_missing_policy_preserves_alignment_for_interleaved_cycle_groups() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-15", periods=5, freq="10s"),
            "cycle_id": "cycle_001",
            "cycle_stage": ["stage_a", "stage_b", "stage_a", "stage_b", "stage_a"],
            "signal": [0.0, 100.0, float("nan"), 200.0, 10.0],
            "signal__source_state": [
                "observed",
                "observed",
                "missing",
                "observed",
                "observed",
            ],
        }
    )
    specs = {"signal": _spec("signal", "linear")}
    config = {
        "group_columns": ["cycle_id", "cycle_stage"],
        "continuous": {
            "method": "time_linear",
            "maximum_bracketing_gap_seconds": 60,
            "require_both_sides": True,
        },
        "control": {"method": "forward_fill", "maximum_age_seconds": 30},
    }

    result = apply_missing_policy(frame, assess_missing_data(frame, specs, config), specs, config)

    assert result.data.loc[2, "signal"] == 5.0
    assert bool(result.data.loc[2, "signal__imputed"])
    assert pd.isna(result.data.loc[1, "signal__imputed"]) is False
