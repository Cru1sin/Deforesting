from __future__ import annotations

import numpy as np
import pandas as pd

from frost_analysis.data.cycles import (
    append_issue,
    assess_sensor_quality,
    infer_sampling_interval_seconds,
    normalize_cycle_status,
    segment_cycles,
    validate_cycles,
)


def _frame(states: list[str], *, frequency: list[float] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-15", periods=len(states), freq="s"),
            "p1__Deforst": states,
            "compressor_frequency": frequency or [50.0] * len(states),
            "evidence_temperature": np.linspace(0, 1, len(states)),
        }
    )


def _config(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "debounce_seconds": 2,
        "recovery_seconds": 2,
        "min_heating_seconds": 3,
        "max_heating_seconds": 100,
        "min_defrost_seconds": 2,
        "max_defrost_seconds": 20,
        "manual_overrides": [],
        "defrost_on_values": ["ON", "1"],
        "defrost_off_values": ["OFF", "0"],
        "corroboration_columns": ["evidence_temperature"],
        "corroboration_expected_directions": {"evidence_temperature": "positive"},
        "corroboration_min_normalized_change": 1.0,
        "gap_warning_factor": 3.0,
    }
    base.update(overrides)
    return base


def test_one_complete_cycle_is_between_two_defrost_events() -> None:
    result = segment_cycles(
        _frame(["OFF"] * 4 + ["ON"] * 3 + ["OFF"] * 8 + ["ON"] * 3 + ["OFF"] * 4),
        "p1__Deforst",
        _config(),
    )
    assert int(result.cycles["quality_flag"].eq("complete").sum()) == 1
    cycle = result.cycles.loc[result.cycles["quality_flag"].eq("complete")].iloc[0]
    assert cycle["heating_start"] < cycle["defrost_start"] < cycle["defrost_end"]


def test_multiple_cycles_and_leading_trailing_partials_are_retained() -> None:
    states = (
        ["OFF"] * 3 + ["ON"] * 3 + ["OFF"] * 7 + ["ON"] * 3 + ["OFF"] * 7 + ["ON"] * 3 + ["OFF"] * 3
    )
    result = segment_cycles(_frame(states), "p1__Deforst", _config())
    assert int(result.cycles["quality_flag"].eq("complete").sum()) == 2
    assert int(result.cycles["quality_flag"].eq("partial").sum()) == 2


def test_flag_chatter_is_debounced() -> None:
    states = ["OFF"] * 3 + ["ON", "OFF"] + ["ON"] * 3 + ["OFF"] * 8 + ["ON"] * 3 + ["OFF"] * 3
    result = segment_cycles(_frame(states), "p1__Deforst", _config())
    assert int(result.cycles["quality_flag"].eq("complete").sum()) == 1
    assert result.debounce_events >= 1


def test_cycle_phase_is_monotone_bounded_and_only_in_development() -> None:
    states = ["ON"] * 3 + ["OFF"] * 10 + ["ON"] * 3 + ["OFF"] * 2
    result = segment_cycles(_frame(states), "p1__Deforst", _config())
    valid = result.frame.loc[result.frame["cycle_phase"].notna()]
    assert valid["cycle_phase"].between(0, 1).all()
    assert valid.groupby("cycle_id")["cycle_phase"].apply(lambda s: s.is_monotonic_increasing).all()
    assert valid["stage"].eq("frost_development").all()


def test_shutdown_marks_cycle_abnormal() -> None:
    states = ["ON"] * 3 + ["OFF"] * 10 + ["ON"] * 3 + ["OFF"] * 2
    frequency = [50.0] * len(states)
    frequency[7:11] = [0.0] * 4
    result = segment_cycles(_frame(states, frequency=frequency), "p1__Deforst", _config())
    assert result.cycles.iloc[0]["quality_flag"] == "abnormal"
    assert "shutdown" in result.cycles.iloc[0]["exclusion_reason"]


def test_no_complete_cycle_degrades_to_partial() -> None:
    result = segment_cycles(_frame(["OFF"] * 20), "p1__Deforst", _config())
    assert result.cycles["quality_flag"].eq("complete").sum() == 0
    assert result.cycles["quality_flag"].eq("partial").all()


def test_abnormally_short_interdefrost_cycle_is_excluded() -> None:
    states = ["ON"] * 3 + ["OFF"] * 2 + ["ON"] * 3 + ["OFF"] * 2
    result = segment_cycles(_frame(states), "p1__Deforst", _config(min_heating_seconds=4))
    assert result.cycles.iloc[0]["quality_flag"] == "excluded"
    assert "heating_duration" in result.cycles.iloc[0]["exclusion_reason"]


def test_manual_override_takes_precedence() -> None:
    override = {
        "cycle_id": "manual_001",
        "heating_start": "2026-07-15 00:00:02",
        "stable_heating_start": "2026-07-15 00:00:04",
        "defrost_start": "2026-07-15 00:00:10",
        "defrost_end": "2026-07-15 00:00:13",
    }
    result = segment_cycles(
        _frame(["OFF"] * 16), "p1__Deforst", _config(manual_overrides=[override])
    )
    assert result.cycles.iloc[0]["segmentation_method"] == "manual_override"
    assert result.cycles.iloc[0]["cycle_id"] == "manual_001"


def test_configured_numeric_and_mixed_case_states_match_on_off_boundaries() -> None:
    canonical = ["ON"] * 3 + ["OFF"] * 10 + ["ON"] * 3 + ["OFF"] * 2
    numeric = ["1"] * 3 + ["0"] * 10 + ["1"] * 3 + ["0"] * 2
    mixed = ["on"] * 3 + ["off"] * 10 + ["On"] * 3 + ["oFf"] * 2
    expected = segment_cycles(_frame(canonical), "p1__Deforst", _config()).cycles
    for states in (numeric, mixed):
        actual = segment_cycles(_frame(states), "p1__Deforst", _config()).cycles
        pd.testing.assert_series_equal(actual["defrost_start"], expected["defrost_start"])
        pd.testing.assert_series_equal(actual["defrost_end"], expected["defrost_end"])


def test_all_unknown_states_fail_fast() -> None:
    with np.testing.assert_raises_regex(ValueError, "no recognized defrost states"):
        segment_cycles(_frame(["UNKNOWN"] * 20), "p1__Deforst", _config())


def test_boundary_confidence_orders_corroboration_and_long_gap_evidence() -> None:
    states = ["ON"] * 3 + ["OFF"] * 10 + ["ON"] * 3 + ["OFF"] * 2
    supported = _frame(states)
    supported["evidence_temperature"] = [0.0] * 10 + list(np.arange(8, dtype=float))
    no_change = supported.copy()
    no_change["evidence_temperature"] = 0.0
    conflict = supported.copy()
    conflict["evidence_temperature"] = -supported["evidence_temperature"]
    long_gap = supported.copy()
    long_gap.loc[8:, "timestamp"] += pd.Timedelta(seconds=20)

    def confidence(frame: pd.DataFrame) -> float:
        result = segment_cycles(frame, "p1__Deforst", _config())
        return float(
            result.cycles.loc[result.cycles["quality_flag"].eq("complete")].iloc[0][
                "segmentation_confidence"
            ]
        )

    assert confidence(supported) > confidence(no_change) > confidence(conflict)
    gap_result = segment_cycles(long_gap, "p1__Deforst", _config(max_heating_seconds=200))
    gap_cycle = gap_result.cycles.loc[gap_result.cycles["quality_flag"].eq("complete")].iloc[0]
    assert float(gap_cycle["maximum_gap_seconds"]) >= 20
    assert confidence(supported) > float(gap_cycle["segmentation_confidence"])
    assert any("long_gap" in warning for warning in gap_result.warnings)


def test_cycle_status_normalizes_quality_to_three_states() -> None:
    """Expose one stable status vocabulary to downstream stages."""
    assert normalize_cycle_status("complete") == "valid"
    assert normalize_cycle_status("contaminated") == "invalid"
    assert normalize_cycle_status("excluded") == "invalid"
    assert normalize_cycle_status("partial") == "incomplete"


def test_validation_keeps_structurally_valid_cycle_with_long_gap() -> None:
    states = ["ON"] * 3 + ["OFF"] * 10 + ["ON"] * 3 + ["OFF"] * 2
    frame = _frame(states)
    frame.loc[8:, "timestamp"] += pd.Timedelta(seconds=20)
    segmentation = segment_cycles(frame, "p1__Deforst", _config(max_heating_seconds=200))

    result = validate_cycles(
        segmentation,
        {"expected_sampling_interval_seconds": 1.0, "gap_warning_factor": 3.0},
    )

    cycle = result.cycles.loc[result.cycles["quality_flag"].eq("complete")].iloc[0]
    assert cycle["quality_flag"] == "complete"
    assert result.frame.loc[result.frame["cycle_id"].eq(cycle["cycle_id"]), "cycle_phase"].notna().any()
    assert any("long_gap" in warning for warning in result.warnings)


def test_append_issue_does_not_turn_nan_into_literal_text() -> None:
    """Quality reasons must remain readable when the source cell is missing."""
    assert append_issue(np.nan, "long_gap") == "long_gap"
    assert append_issue("existing", "long_gap") == "existing;long_gap"


def test_sampling_interval_uses_configured_fallback_when_no_positive_delta_exists() -> None:
    """A configured interval is explicit; a hidden one-second default is not."""
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-07-15"] * 2)})
    assert infer_sampling_interval_seconds(
        frame,
        expected_sampling_interval_seconds=2.0,
    ) == 2.0


def test_sensor_quality_uses_required_channels_without_invalidating_cycle() -> None:
    timestamps = pd.date_range("2026-07-15", periods=4, freq="s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cycle_id": "cycle_001",
            "required_temperature": [1.0, np.nan, 3.0, 4.0],
            "optional_temperature": [1.0, np.nan, 3.0, 4.0],
        }
    )
    cycles = pd.DataFrame(
        [
            {
                "cycle_id": "cycle_001",
                "heating_start": timestamps[0],
                "defrost_end": timestamps[-1],
            }
        ]
    )

    result = assess_sensor_quality(frame, cycles, required_channels=["required_temperature"])

    row = result.iloc[0]
    assert row["sensor_quality"] == "partial"
    assert row["sensor_min_coverage"] < 1.0
    assert row["sensor_low_coverage_channels"] == "required_temperature"
    assert row["sensor_quality_reason"] == "required_channel_low_coverage"


def test_sensor_quality_does_not_require_unconfigured_optional_channels() -> None:
    timestamps = pd.date_range("2026-07-15", periods=4, freq="s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cycle_id": "cycle_001",
            "required_temperature": [1.0, 2.0, 3.0, 4.0],
            "optional_temperature": [1.0, np.nan, 3.0, 4.0],
        }
    )
    cycles = pd.DataFrame(
        [{"cycle_id": "cycle_001", "heating_start": timestamps[0], "defrost_end": timestamps[-1]}]
    )

    result = assess_sensor_quality(frame, cycles, required_channels=["required_temperature"])

    assert result.iloc[0]["sensor_quality"] == "complete"
