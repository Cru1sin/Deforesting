from __future__ import annotations

import numpy as np
import pandas as pd

from frost_analysis.processing.baseline import select_clean_baselines


def _inputs(
    values: list[float], *, start: int = 0, stop: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stop = stop if stop is not None else len(values) - 1
    frame = pd.DataFrame(
        {
            "sensor_time": pd.date_range("2026-07-15", periods=len(values), freq="s"),
            "cycle_id": "cycle_001",
            "cycle_quality": "complete",
            "stage": "frost_development",
            "Te": values,
        }
    )
    cycles = pd.DataFrame(
        [
            {
                "cycle_id": "cycle_001",
                "heating_start": frame.sensor_time.iloc[start],
                "stable_heating_start": frame.sensor_time.iloc[start + 2],
                "defrost_start": frame.sensor_time.iloc[stop],
                "defrost_end": frame.sensor_time.iloc[min(stop + 1, len(frame) - 1)],
                "quality_flag": "complete",
            }
        ]
    )
    return frame, cycles


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "recovery_exclusion_seconds": 2,
        "search_end_fraction": 0.6,
        "window_seconds": 4,
        "step_seconds": 1,
        "minimum_coverage": 0.75,
        "maximum_stability_score": 0.15,
        "relative_epsilon": 0.01,
    }
    config.update(overrides)
    return config


def test_selects_earliest_stable_window_and_adds_cycle_local_offsets() -> None:
    frame, cycles = _inputs([10.0] * 9 + [11.0, 12.0, 13.0, 14.0])
    result = select_clean_baselines(frame, cycles, ["Te"], ["Te"], _config())
    row = result.baselines.iloc[0]
    assert row["selection_status"] == "selected"
    assert row["baseline_mean"] == 10.0
    selected = result.frame["stage"].eq("stable_clean")
    assert selected.any()
    assert result.frame.loc[result.frame["Te"].eq(14.0), "Te__baseline_offset"].item() == 4.0


def test_no_stable_window_is_retained_as_low_confidence_fallback() -> None:
    frame, cycles = _inputs(list(np.arange(15, dtype=float)))
    result = select_clean_baselines(
        frame, cycles, ["Te"], ["Te"], _config(maximum_stability_score=0.001)
    )
    assert result.baselines.iloc[0]["selection_status"] == "low_confidence_fallback"
    assert result.baselines.iloc[0]["quality_flag"] == "low_confidence"


def test_long_recovery_exclusion_moves_search_start() -> None:
    frame, cycles = _inputs([100.0] * 6 + [10.0] * 12)
    result = select_clean_baselines(
        frame, cycles, ["Te"], ["Te"], _config(recovery_exclusion_seconds=6)
    )
    assert result.baselines.iloc[0]["baseline_mean"] == 10.0
    assert result.baselines.iloc[0]["clean_start"] >= frame.sensor_time.iloc[6]


def test_insufficient_cycle_duration_has_structured_failure() -> None:
    frame, cycles = _inputs([1.0] * 5, stop=4)
    result = select_clean_baselines(frame, cycles, ["Te"], ["Te"], _config(window_seconds=10))
    assert result.baselines.iloc[0]["selection_status"] == "failed"
    assert result.baselines.iloc[0]["failure_reason"] == "insufficient_search_duration"


def test_near_zero_baseline_does_not_create_infinite_relative_offset() -> None:
    frame, cycles = _inputs([0.0] * 9 + [1.0] * 4)
    result = select_clean_baselines(frame, cycles, ["Te"], ["Te"], _config())
    relative = result.frame["Te__baseline_relative"]
    assert np.isinf(relative.dropna()).sum() == 0
    assert relative.isna().all()


def test_missing_baseline_values_are_flagged_per_variable() -> None:
    frame, cycles = _inputs([np.nan] * 13)
    result = select_clean_baselines(frame, cycles, ["Te"], ["Te"], _config())
    assert result.baselines.iloc[0]["selection_status"] == "failed"
    assert "missing" in result.baselines.iloc[0]["failure_reason"]


def test_baseline_offsets_are_unavailable_until_clean_end_and_future_safe() -> None:
    frame, cycles = _inputs([10.0] * 13, stop=12)
    changed = frame.copy()
    changed.loc[6, "Te"] = 12.0

    original = select_clean_baselines(
        frame, cycles, ["Te"], ["Te"], _config(maximum_stability_score=100.0)
    )
    perturbed = select_clean_baselines(
        changed, cycles, ["Te"], ["Te"], _config(maximum_stability_score=100.0)
    )

    clean_end = pd.Timestamp(original.baselines.iloc[0]["clean_end"])
    assert pd.Timestamp(perturbed.baselines.iloc[0]["clean_end"]) == clean_end
    assert (
        original.baselines.iloc[0]["baseline_mean"] != perturbed.baselines.iloc[0]["baseline_mean"]
    )

    for result in (original, perturbed):
        before = result.frame["sensor_time"].lt(clean_end)
        available = result.frame["Te__baseline_available"]
        source_time = pd.to_datetime(result.frame["Te__baseline_source_latest_time"])
        assert available.equals(result.frame["sensor_time"].ge(clean_end))
        assert source_time.eq(clean_end).all()
        assert result.frame.loc[before, "Te__baseline_offset"].isna().all()
        assert result.frame.loc[before, "Te__baseline_relative"].isna().all()

    before = original.frame["sensor_time"].lt(clean_end)
    pd.testing.assert_series_equal(
        original.frame.loc[before, "Te__baseline_offset"],
        perturbed.frame.loc[before, "Te__baseline_offset"],
    )


def test_required_drifting_anchor_forces_transparent_fallback_with_anchor_evidence() -> None:
    seconds = np.arange(30, dtype=float)
    frame, cycles = _inputs([10.0] * len(seconds), stop=29)
    frame["steady_anchor"] = 5.0
    frame["critical_anchor"] = seconds * 0.2
    result = select_clean_baselines(
        frame,
        cycles,
        ["Te", "steady_anchor", "critical_anchor"],
        ["Te", "steady_anchor", "critical_anchor"],
        _config(
            window_seconds=8,
            search_end_fraction=0.9,
            maximum_anchor_stability_score=1.0,
            anchor_scale_floors={"Te": 0.2, "steady_anchor": 0.2, "critical_anchor": 0.2},
            required_anchor_columns=["critical_anchor"],
            minimum_anchor_pass_fraction=0.67,
        ),
    )
    assert result.baselines.iloc[0]["selection_status"] == "low_confidence_fallback"
    assert result.baselines.iloc[0]["quality_flag"] == "low_confidence"
    evidence = result.candidates.loc[result.candidates["anchor"].eq("critical_anchor")]
    assert {
        "anchor_slope_per_s",
        "anchor_total_change",
        "anchor_variation",
        "anchor_coverage",
        "anchor_stability_score",
        "anchor_pass",
        "anchor_failure_reason",
    } <= set(evidence.columns)
    assert not evidence["anchor_pass"].any()


def test_anchor_absolute_change_limit_prevents_borderline_recovery_window() -> None:
    frame, cycles = _inputs(list(np.linspace(0.0, 1.0, 30)), stop=29)
    result = select_clean_baselines(
        frame,
        cycles,
        ["Te"],
        ["Te"],
        _config(
            window_seconds=8,
            search_end_fraction=0.9,
            maximum_anchor_stability_score=100.0,
            anchor_maximum_absolute_change={"Te": 0.1},
            required_anchor_columns=["Te"],
        ),
    )
    assert result.baselines.iloc[0]["selection_status"] == "low_confidence_fallback"
    assert result.candidates["anchor_change_limit"].eq(0.1).all()
    assert result.candidates["anchor_failure_reason"].eq("absolute_change_exceeds_threshold").all()
