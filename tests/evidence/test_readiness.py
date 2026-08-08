from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frost_analysis.evidence.readiness import (
    audit_performance_target,
    compare_incremental_models,
    compute_signal_lead,
    summarize_readiness,
)

from .conftest import settings


def _readiness_frame(seconds: int = 600) -> pd.DataFrame:
    elapsed = np.arange(0, seconds + 10, 10, dtype=float)
    timestamp = pd.Timestamp("2026-07-01T00:00:00") + pd.to_timedelta(elapsed, unit="s")
    target = np.full(len(elapsed), 100.0)
    feature = np.resize(np.array([-1.0, 0.0, 1.0]), len(elapsed))
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "cycle_stage": "frost_development",
            "cycle_elapsed_seconds": elapsed,
            "heating_capacity": target,
            "heating_capacity__baseline": 100.0,
            "heating_capacity__baseline_residual": target - 100.0,
            "heating_capacity__imputed": False,
            "feature_a__baseline_residual": feature,
            "feature_a__imputed": False,
            "ambient_temperature": 5.0,
            "ambient_temperature__imputed": False,
            "environment_relative_humidity": 80.0,
            "environment_relative_humidity__imputed": False,
            "water_in_temperature": 35.0,
            "water_in_temperature__imputed": False,
            "water_flow": 1.0,
            "water_flow__imputed": False,
            "compressor_frequency": 50.0,
            "compressor_frequency__imputed": False,
        }
    )


def _record(frame: pd.DataFrame) -> dict[str, object]:
    start = pd.to_datetime(frame["timestamp"]).min()
    return {
        "cycle_name": "cycle_001",
        "experiment_date": "2026-07-01",
        "boundaries": {
            "baseline_end": (start + pd.Timedelta(seconds=60)).isoformat(),
            "defrost_start": (
                pd.to_datetime(frame["timestamp"]).max() + pd.Timedelta(seconds=10)
            ).isoformat()
        },
    }


def test_target_event_uses_current_degradation_and_elapsed_persistence() -> None:
    frame = _readiness_frame()
    frame.loc[frame["cycle_elapsed_seconds"] >= 60, "heating_capacity"] = 89.0

    row = audit_performance_target(
        _record(frame), frame, "heating_capacity", settings()
    )

    assert row["event_5_elapsed_minutes"] == pytest.approx(1.0)
    assert row["event_10_elapsed_minutes"] == pytest.approx(1.0)
    assert np.isnan(row["event_15_elapsed_minutes"])
    assert row["primary_event_status"] == "event_observed"
    assert row["metric_status"] == "available"


@pytest.mark.parametrize(
    ("baseline", "reason"),
    [
        (np.nan, "baseline_unavailable"),
        (0.0, "baseline_nonpositive_or_zero"),
    ],
)
def test_target_audit_rejects_invalid_baseline(baseline: float, reason: str) -> None:
    frame = _readiness_frame()
    frame["heating_capacity__baseline"] = baseline

    row = audit_performance_target(
        _record(frame), frame, "heating_capacity", settings()
    )

    assert row["metric_status"] == "unavailable"
    assert row["exclusion_reason"] == reason


def test_target_audit_rejects_inconsistent_baseline() -> None:
    frame = _readiness_frame()
    frame.loc[1, "heating_capacity__baseline"] = 101.0

    row = audit_performance_target(
        _record(frame), frame, "heating_capacity", settings()
    )

    assert row["exclusion_reason"] == "baseline_inconsistent"


def test_target_event_is_right_censored_and_missing_breaks_persistence() -> None:
    frame = _readiness_frame(seconds=230)
    frame.loc[frame["cycle_elapsed_seconds"] >= 60, "heating_capacity"] = 89.0
    frame.loc[frame["cycle_elapsed_seconds"] == 120, "heating_capacity__imputed"] = True

    row = audit_performance_target(
        _record(frame), frame, "heating_capacity", settings()
    )

    assert row["primary_event_status"] == "right_censored_at_defrost_start"
    assert np.isnan(row["primary_event_elapsed_minutes"])


@pytest.mark.parametrize("mode", ["nan", "imputed"])
def test_target_without_observations_is_unavailable(mode: str) -> None:
    frame = _readiness_frame()
    if mode == "nan":
        frame["heating_capacity"] = np.nan
    else:
        frame["heating_capacity__imputed"] = True

    row = audit_performance_target(
        _record(frame), frame, "heating_capacity", settings()
    )

    assert row["primary_event_status"] == "target_unavailable"
    assert row["metric_status"] == "unavailable"
    assert row["exclusion_reason"] == "target_unavailable"


def test_baseline_window_cannot_trigger_event_or_supply_model_anchor() -> None:
    frame = _readiness_frame(seconds=1200)
    frame.loc[frame["cycle_elapsed_seconds"] < 60, "heating_capacity"] = 80.0
    record = _record(frame)

    audit = audit_performance_target(
        record, frame, "heating_capacity", settings()
    )
    rows = compare_incremental_models(
        [(record, frame)],
        [("feature_a", "increase")],
        pd.DataFrame([audit]),
        settings(targets=("heating_capacity",)),
    )

    assert audit["primary_event_status"] == "right_censored_at_defrost_start"
    assert audit["valid_pairs_5min"] == 85
    five_minute = rows.loc[rows["horizon_minutes"].eq(5)].iloc[0]
    assert five_minute["expected_anchor_count"] == 85


def test_signal_onset_aligns_direction_and_excludes_reference_window() -> None:
    frame = _readiness_frame()
    frame.loc[frame["cycle_elapsed_seconds"] > 300, "feature_a__baseline_residual"] = 8.0
    event_elapsed = 9.0

    increase = compute_signal_lead(
        frame, "feature_a", "increase", event_elapsed, "event_observed", settings()
    )
    frame["feature_a__baseline_residual"] *= -1
    decrease = compute_signal_lead(
        frame, "feature_a", "decrease", event_elapsed, "event_observed", settings()
    )

    assert increase["signal_onset_elapsed_minutes"] > 5.0
    assert decrease["signal_onset_elapsed_minutes"] == pytest.approx(
        increase["signal_onset_elapsed_minutes"]
    )
    assert increase["lead_minutes"] == pytest.approx(
        event_elapsed - increase["signal_onset_elapsed_minutes"]
    )


def test_signal_onset_has_no_epsilon_fallback_for_zero_mad() -> None:
    frame = _readiness_frame()
    frame.loc[frame["cycle_elapsed_seconds"] <= 300, "feature_a__baseline_residual"] = 0.0

    result = compute_signal_lead(
        frame, "feature_a", "increase", 9.0, "event_observed", settings()
    )

    assert result["lead_status"] == "invalid_initial_scale"
    assert np.isnan(result["signal_onset_elapsed_minutes"])


def test_one_cycle_keeps_descriptive_lead_but_models_are_unavailable() -> None:
    frame = _readiness_frame(seconds=1200)
    frame.loc[frame["cycle_elapsed_seconds"] > 300, "feature_a__baseline_residual"] += 8.0
    frame["heating_capacity"] = 100.0 - frame["cycle_elapsed_seconds"] / 300.0
    frame["heating_capacity__baseline_residual"] = frame["heating_capacity"] - 100.0
    record = _record(frame)
    audits = pd.DataFrame(
        [audit_performance_target(record, frame, "heating_capacity", settings())]
    )

    rows = compare_incremental_models(
        [(record, frame)],
        [("feature_a", "increase")],
        audits,
        settings(targets=("heating_capacity",)),
    )

    row = rows.loc[rows["horizon_minutes"].eq(5)].iloc[0]
    assert row["expected_anchor_count"] > 0
    assert row["exclusion_reason"] == "no_training_cycles_after_holdout"
    assert row["metric_status"] == "unavailable"


def test_missing_context_removes_only_its_complete_case_anchor() -> None:
    first = _readiness_frame(seconds=1200)
    second = _readiness_frame(seconds=1200)
    second.loc[second["cycle_elapsed_seconds"] == 600, "water_flow__imputed"] = True
    records = [_record(first), {**_record(second), "cycle_name": "cycle_002"}]
    audits = pd.DataFrame(
        [
            audit_performance_target(record, frame, "heating_capacity", settings())
            for record, frame in zip(records, (first, second), strict=True)
        ]
    )

    rows = compare_incremental_models(
        list(zip(records, (first, second), strict=True)),
        [("feature_a", "increase")],
        audits,
        settings(
            targets=("heating_capacity",),
            minimum_valid_pairs=2,
            minimum_pair_coverage=0.5,
        ),
    )

    second_row = rows.loc[rows["held_out_cycle"].eq("cycle_002")].iloc[0]
    first_row = rows.loc[rows["held_out_cycle"].eq("cycle_001")].iloc[0]
    assert second_row["expected_anchor_count"] == first_row["expected_anchor_count"]
    assert second_row["valid_anchor_count"] == first_row["valid_anchor_count"] - 1


def test_missing_context_column_preserves_theoretical_anchor_denominator() -> None:
    frame = _readiness_frame(seconds=1200).drop(columns="water_flow")
    record = _record(frame)
    audits = pd.DataFrame(
        [audit_performance_target(record, frame, "heating_capacity", settings())]
    )

    row = compare_incremental_models(
        [(record, frame)],
        [("feature_a", "increase")],
        audits,
        settings(targets=("heating_capacity",)),
    ).iloc[0]

    assert row["expected_anchor_count"] > 0
    assert row["valid_anchor_count"] == 0
    assert row["anchor_coverage"] == 0.0


def test_model_skills_use_m1_for_level_and_m2_for_dynamic() -> None:
    frames: list[pd.DataFrame] = []
    records: list[dict[str, object]] = []
    for index, date in enumerate(("2026-07-01", "2026-07-02", "2026-07-03")):
        frame = _readiness_frame(seconds=1200)
        elapsed = frame["cycle_elapsed_seconds"].to_numpy(dtype=float)
        feature = np.sin(elapsed / 80.0) + elapsed / 600.0 + index * 0.05
        frame["feature_a__baseline_residual"] = feature
        frame["heating_capacity"] = 100.0 - 2.0 * feature - elapsed / 1200.0
        frame["heating_capacity__baseline_residual"] = frame["heating_capacity"] - 100.0
        frame["timestamp"] = pd.Timestamp(f"{date}T00:00:00") + pd.to_timedelta(
            elapsed, unit="s"
        )
        record = _record(frame)
        record["cycle_name"] = f"cycle_{index + 1:03d}"
        record["experiment_date"] = date
        frames.append(frame)
        records.append(record)
    cycles = list(zip(records, frames, strict=True))
    audits = pd.DataFrame(
        [
            audit_performance_target(record, frame, "heating_capacity", settings())
            for record, frame in cycles
        ]
    )

    rows = compare_incremental_models(
        cycles,
        [("feature_a", "increase")],
        audits,
        settings(
            targets=("heating_capacity",),
            minimum_valid_pairs=20,
            minimum_pair_coverage=0.5,
        ),
    )

    available = rows.loc[
        rows["metric_status"].eq("available") & rows["horizon_minutes"].eq(5)
    ]
    assert len(available) == 3
    assert np.allclose(
        available["skill_level_vs_context"],
        1.0 - available["mae_m2"] / available["mae_m1"],
    )
    assert np.allclose(
        available["skill_dynamic_vs_level"],
        1.0 - available["mae_m3"] / available["mae_m2"],
    )


def test_readiness_summary_uses_date_units_and_dynamic_increment() -> None:
    split_rows = pd.DataFrame(
        [
            {
                "held_out_cycle": "a1",
                "held_out_date": "2026-07-01",
                "feature": "feature_a",
                "target": "heating_capacity",
                "horizon_minutes": 5,
                "lead_minutes": 4.0,
                "lead_status": "available",
                "skill_level_vs_context": 0.2,
                "skill_dynamic_vs_level": 0.1,
                "expected_anchor_count": 10,
                "metric_status": "available",
            },
            {
                "held_out_cycle": "a2",
                "held_out_date": "2026-07-01",
                "feature": "feature_a",
                "target": "heating_capacity",
                "horizon_minutes": 5,
                "lead_minutes": 6.0,
                "lead_status": "available",
                "skill_level_vs_context": 0.4,
                "skill_dynamic_vs_level": 0.2,
                "expected_anchor_count": 10,
                "metric_status": "available",
            },
            {
                "held_out_cycle": "b1",
                "held_out_date": "2026-07-02",
                "feature": "feature_a",
                "target": "heating_capacity",
                "horizon_minutes": 5,
                "lead_minutes": 2.0,
                "lead_status": "available",
                "skill_level_vs_context": 0.1,
                "skill_dynamic_vs_level": 0.05,
                "expected_anchor_count": 10,
                "metric_status": "available",
            },
        ]
    )
    audits = pd.DataFrame(
        [{"target": "heating_capacity", "metric_status": "available"}]
    )
    trends = pd.DataFrame(
        [
            {
                "cycle_name": "a1",
                "experiment_date": "2026-07-01",
                "feature": "feature_a",
                "signed_effect": 0.8,
                "metric_status": "available",
            },
            {
                "cycle_name": "b1",
                "experiment_date": "2026-07-02",
                "feature": "feature_a",
                "signed_effect": 0.6,
                "metric_status": "available",
            },
        ]
    )

    summary = summarize_readiness(
        split_rows,
        audits,
        trends,
        [("feature_a", "increase")],
        settings(targets=("heating_capacity",)),
    ).iloc[0]

    assert summary["lead_median_minutes"] == pytest.approx(3.5)
    assert summary["level_skill_median"] == pytest.approx(0.2)
    assert summary["dynamic_skill_median"] == pytest.approx(0.1)
    assert summary["readiness_status"] == "dynamic_increment_supported"
