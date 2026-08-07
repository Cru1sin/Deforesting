from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frost_analysis.channels import load_channels
from frost_analysis.config import Config, load_config
from frost_analysis.cycles import (
    _build_cycles,
    _debounce_state,
    _defrost_runs,
    _fill_short_state_gaps,
    _normalize_state,
    find_stable_heating_start,
    label_cycles,
)
from frost_analysis.images import match_images
from frost_analysis.prepare import _expected_row_count, _maximum_gap, _observed_fraction, prepare
from frost_analysis.sensors import read_edf_environment

prepare_module = import_module("frost_analysis.prepare")
ROOT = Path(__file__).resolve().parents[1]


def _config(root: Path, raw: Path, *, sensor_globs: tuple[str, ...] = ("*.xls",)) -> Config:
    return Config(
        project_root=root,
        experiment_id="exp_test",
        experiment_date="2026-07-15",
        input_dir=raw,
        channels_path=root / "channels.yaml",
        sensor_globs=sensor_globs,
        image_extensions=(".jpg",),
        timestamp_column="时间",
        expected_sensor_interval_seconds=1,
        image_match_tolerance_seconds=2,
        edf_pair_tolerance_seconds=1.0,
        cycles={
            "defrost_channel": "defrost_active",
            "maximum_state_gap_seconds": 5,
            "debounce_seconds": 20,
            "minimum_defrost_seconds": 1,
            "maximum_defrost_seconds": 100,
            "minimum_heating_seconds": 1,
            "maximum_heating_seconds": 100,
            "stable_heating_seconds": 2,
        },
        process={"resample_interval_seconds": 10},
        camera_roles={},
    )


def _write_edf(path: Path, rows: list[tuple[str, str, str, str, str, str]]) -> Path:
    header = (
        "# EdfVersion=4.0\n"
        "# Date=2026-07-20T09:00:00+08:00\n"
        "Type=float64,Format=.3f,Unit=s\n"
        "Epoch_UTC\tLocal_Date_Time\tT_SHT40_111\tRH_SHT40_111\t"
        "T_SHT40_222\tRH_SHT40_222\n"
    )
    path.write_text(
        header + "\n".join("\t".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_read_edf_environment_preserves_local_clock_pairs_once_and_clips(tmp_path: Path) -> None:
    first = _write_edf(
        tmp_path / "first.edf",
        [
            ("0", "2026-07-19T23:59:59+08:00", "20", "101", "", ""),
            ("1", "2026-07-20T09:00:00+08:00", "20", "101", "", ""),
            ("2", "2026-07-20T09:00:00.200000+08:00", "", "", "22", "103"),
            ("3", "2026-07-20T09:00:01+08:00", "24", "102", "", ""),
            ("4", "2026-07-20T09:00:01.200000+08:00", "", "", "26", "104"),
        ],
    )
    second = _write_edf(
        tmp_path / "second.edf",
        [
            ("5", "2026-07-20T09:00:01+08:00", "24", "102", "", ""),
            ("6", "2026-07-20T09:00:01.200000+08:00", "", "", "26", "104"),
            ("7", "2026-07-20T09:00:03+08:00", "30", "105", "", ""),
            ("8", "2026-07-20T09:00:03.200000+08:00", "", "", "32", "107"),
            ("9", "2026-07-20T09:00:04+08:00", "34", "108", "", ""),
            ("10", "2026-07-20T09:00:06.500000+08:00", "", "", "36", "110"),
            ("11", "2026-07-21T00:00:00+08:00", "40", "111", "", ""),
            ("12", "2026-07-21T00:00:00.200000+08:00", "", "", "42", "113"),
        ],
    )

    result = read_edf_environment(
        [first, second],
        pd.Timestamp("2026-07-20 09:00:00"),
        pd.Timestamp("2026-07-20 09:00:02"),
        pd.Timedelta(seconds=1),
    )

    assert result["timestamp"].tolist() == [
        pd.Timestamp("2026-07-20 09:00:00.100000"),
        pd.Timestamp("2026-07-20 09:00:01.100000"),
    ]
    assert result["environment_temperature"].tolist() == [21.0, 25.0]
    assert result["environment_relative_humidity"].tolist() == [102.0, 103.0]
    assert result["timestamp"].dt.tz is None


def test_read_edf_environment_chooses_nearest_pair_without_reuse(tmp_path: Path) -> None:
    path = _write_edf(
        tmp_path / "environment.edf",
        [
            ("0", "2026-07-20T09:00:00+08:00", "20", "101", "", ""),
            ("1", "2026-07-20T09:00:00.900000+08:00", "", "", "22", "103"),
            ("2", "2026-07-20T09:00:01+08:00", "24", "105", "", ""),
        ],
    )

    result = read_edf_environment(
        [path],
        pd.Timestamp("2026-07-20 09:00:00"),
        pd.Timestamp("2026-07-20 09:00:02"),
        pd.Timedelta(seconds=1),
    )

    assert result["timestamp"].tolist() == [
        pd.Timestamp("2026-07-20 09:00:00.950000")
    ]
    assert result["environment_temperature"].tolist() == [23.0]
    assert result["environment_relative_humidity"].tolist() == [104.0]


def test_read_edf_environment_rejects_nat_pair_tolerance(tmp_path: Path) -> None:
    path = _write_edf(
        tmp_path / "environment.edf",
        [("0", "2026-07-20T09:00:00+08:00", "20", "101", "22", "103")],
    )

    with pytest.raises(ValueError, match="pair_tolerance"):
        read_edf_environment(
            [path],
            pd.Timestamp("2026-07-20 09:00:00"),
            pd.Timestamp("2026-07-20 09:00:01"),
            pd.NaT,
        )


def test_prepare_attaches_edf_channels_without_replacing_t4(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "0715"
    raw.mkdir(parents=True)
    (raw / "sample参数1.xls").write_text(
        "时间\tT4\tDefrost\n"
        "2026-07-15 09:00:00\t10\tOFF\n"
        "2026-07-15 09:00:01\t11\tOFF\n"
        "2026-07-15 09:00:02\t12\tOFF\n",
        encoding="utf-8",
    )
    _write_edf(
        raw / "environment.edf",
        [
            ("0", "2026-07-15T08:59:59+08:00", "1", "101", "", ""),
            ("1", "2026-07-15T09:00:00+08:00", "20", "101", "", ""),
            ("2", "2026-07-15T09:00:00.200000+08:00", "", "", "22", "103"),
            ("3", "2026-07-15T09:00:01+08:00", "24", "102", "", ""),
            ("4", "2026-07-15T09:00:01.200000+08:00", "", "", "26", "104"),
            ("5", "2026-07-15T09:00:03+08:00", "30", "105", "", ""),
            ("6", "2026-07-15T09:00:03.200000+08:00", "", "", "32", "107"),
        ],
    )
    config = _config(tmp_path, raw, sensor_globs=("*.xls", "*.edf"))
    channels = {
        "ambient_temperature": {
            "source_names": ["p1__T4"],
            "unit": "degC",
            "kind": "continuous",
            "role": "context",
            "resample": "mean",
            "missing": "interpolate",
            "analysis_candidate": False,
        },
        "defrost_active": {
            "source_names": ["p1__Defrost"],
            "unit": None,
            "kind": "event",
            "role": "event",
            "resample": "last",
            "missing": "none",
            "analysis_candidate": False,
            "allowed_values": {"ON": True, "OFF": False},
        },
        "environment_temperature": {
            "source_names": ["environment_temperature"],
            "unit": "degC",
            "kind": "continuous",
            "role": "context",
            "resample": "mean",
            "missing": "interpolate",
            "analysis_candidate": False,
        },
        "environment_relative_humidity": {
            "source_names": ["environment_relative_humidity"],
            "unit": "%RH",
            "kind": "continuous",
            "role": "context",
            "resample": "mean",
            "missing": "interpolate",
            "analysis_candidate": False,
        },
    }

    prepared, _ = prepare(config, channels)
    without_edf, _ = prepare(replace(config, sensor_globs=("*.xls",)), channels)

    assert prepared["timestamp"].tolist() == pd.to_datetime(
        [
            "2026-07-15 09:00:00",
            "2026-07-15 09:00:01",
            "2026-07-15 09:00:02",
        ]
    ).tolist()
    main_rows = prepared.loc[prepared["timestamp"].isin(pd.to_datetime(
        ["2026-07-15 09:00:00", "2026-07-15 09:00:01", "2026-07-15 09:00:02"]
    ))]
    assert main_rows["ambient_temperature"].tolist() == [10.0, 11.0, 12.0]
    environment = prepared.loc[prepared["environment_temperature"].notna()]
    assert environment["environment_temperature"].tolist() == [21.0, 25.0]
    assert environment["environment_relative_humidity"].tolist() == [102.0, 103.0]
    assert environment["environment_relative_humidity__invalid"].eq(False).all()
    old_columns = [
        column
        for column in without_edf.columns
        if not column.startswith("environment_")
    ]
    pd.testing.assert_frame_equal(
        prepared[old_columns].reset_index(drop=True),
        without_edf[old_columns].reset_index(drop=True),
    )


def test_prepare_without_edf_keeps_new_channels_missing(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "0715"
    raw.mkdir(parents=True)
    (raw / "sample参数1.xls").write_text(
        "时间\tT4\tDefrost\n2026-07-15 09:00:00\t10\tOFF\n",
        encoding="utf-8",
    )
    config = _config(tmp_path, raw, sensor_globs=("*.xls", "*.edf"))
    channels = {
        "ambient_temperature": {
            "source_names": ["p1__T4"],
            "unit": "degC",
            "kind": "continuous",
            "role": "context",
            "resample": "mean",
            "missing": "interpolate",
            "analysis_candidate": False,
        },
        "defrost_active": {
            "source_names": ["p1__Defrost"],
            "unit": None,
            "kind": "event",
            "role": "event",
            "resample": "last",
            "missing": "none",
            "analysis_candidate": False,
            "allowed_values": {"ON": True, "OFF": False},
        },
        "environment_temperature": {
            "source_names": ["environment_temperature"],
            "unit": "degC",
            "kind": "continuous",
            "role": "context",
            "resample": "mean",
            "missing": "interpolate",
            "analysis_candidate": False,
        },
        "environment_relative_humidity": {
            "source_names": ["environment_relative_humidity"],
            "unit": "%RH",
            "kind": "continuous",
            "role": "context",
            "resample": "mean",
            "missing": "interpolate",
            "analysis_candidate": False,
        },
    }

    prepared, _ = prepare(config, channels)

    assert prepared["ambient_temperature"].tolist() == [10.0]
    assert prepared["environment_temperature"].isna().all()
    assert prepared["environment_relative_humidity"].isna().all()


def test_label_cycles_defines_progress_only_during_frost_development() -> None:
    timestamps = pd.date_range("2026-07-15 00:00:00", periods=9, freq="s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "defrost_active": [True, True, False, False, False, False, True, True, False],
        }
    )

    labeled, summary = label_cycles(
        frame,
        "defrost_active",
        {
            "maximum_state_gap_seconds": 5,
            "debounce_seconds": 0.5,
            "minimum_defrost_seconds": 1,
            "maximum_defrost_seconds": 100,
            "minimum_heating_seconds": 1,
            "maximum_heating_seconds": 100,
            "stable_heating_seconds": 2,
        },
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    cycle = summary.loc[summary["cycle_status"].eq("valid")].iloc[0]
    development = labeled.loc[labeled["cycle_stage"].eq("frost_development")]
    assert cycle["cycle_status"] == "valid"
    assert development["cycle_elapsed_seconds"].notna().all()
    assert development["cycle_progress"].between(0, 1).all()
    assert labeled.loc[labeled["cycle_stage"] != "frost_development", "cycle_progress"].isna().all()


def _cycle_frame_with_mode(
    states: list[object], modes: list[object] | None = None
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-15", periods=len(states), freq="s"),
            "defrost_active": states,
        }
    )
    if modes is not None:
        frame["operating_mode"] = modes
    return frame


def _short_cycle_settings() -> dict[str, object]:
    return {
        "maximum_state_gap_seconds": 5,
        "debounce_seconds": 0.5,
        "minimum_defrost_seconds": 2,
        "maximum_defrost_seconds": 100,
        "minimum_heating_seconds": 1,
        "maximum_heating_seconds": 100,
        "stable_heating_seconds": 2,
        "operating_mode_channel": "operating_mode",
        "required_operating_mode": "3",
    }


def test_cycle_summary_keeps_preceding_and_terminal_defrost_durations() -> None:
    frame = _cycle_frame_with_mode(
        [True, True, False, False, False, False, False, True, True, False, False],
        ["3"] * 11,
    )

    _, summary = label_cycles(
        frame,
        "defrost_active",
        _short_cycle_settings(),
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    cycle = summary.loc[summary["cycle_id"].eq("cycle_001")].iloc[0]
    assert cycle["preceding_defrost_duration_seconds"] == 2
    assert cycle["terminal_defrost_duration_seconds"] == 2
    assert "defrost_duration_seconds" not in summary


@pytest.mark.parametrize(
    ("states", "reason"),
    [
        (
            [True, False, False, False, False, False, False, True, True, False, False],
            "preceding_defrost_duration_out_of_range",
        ),
        (
            [True, True, False, False, False, False, False, True, False, False, False],
            "terminal_defrost_duration_out_of_range",
        ),
    ],
)
def test_cycle_summary_reports_each_defrost_duration_failure_reason(
    states: list[object], reason: str
) -> None:
    _, summary = label_cycles(
        _cycle_frame_with_mode(states, ["3"] * len(states)),
        "defrost_active",
        _short_cycle_settings(),
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    cycle = summary.loc[summary["cycle_id"].eq("cycle_001")].iloc[0]
    assert cycle["cycle_status"] == "invalid"
    assert cycle["cycle_status_reason"] == reason


@pytest.mark.parametrize(
    ("modes", "status", "reason"),
    [
        (
            ["3", "3", "3", "2", "3", "3", "3", "2", "2", "3", "3"],
            "invalid",
            "non_heating_mode_present",
        ),
        (
            [None, None, None, None, None, None, None, "2", "2", "3", "3"],
            "incomplete",
            "missing_operating_mode",
        ),
        (
            ["3", "3", "3", "3", "3", "3", "3", "2", "2", "3", "3"],
            "valid",
            "",
        ),
    ],
)
def test_operating_mode_only_checks_heating_interval(
    modes: list[object], status: str, reason: str
) -> None:
    _, summary = label_cycles(
        _cycle_frame_with_mode(
            [True, True, False, False, False, False, False, True, True, False, False],
            modes,
        ),
        "defrost_active",
        _short_cycle_settings(),
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    cycle = summary.loc[summary["cycle_id"].eq("cycle_001")].iloc[0]
    assert cycle["cycle_status"] == status
    assert cycle["cycle_status_reason"] == reason


def test_single_defrost_event_preserves_known_defrost_stage() -> None:
    frame = _cycle_frame_with_mode([False, False, True, True, False, False])

    labeled, summary = label_cycles(
        frame,
        "defrost_active",
        _short_cycle_settings(),
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    assert not summary["cycle_id"].eq("cycle_001").any()
    assert summary["cycle_id"].tolist() == ["partial_001"]
    assert labeled["cycle_stage"].tolist() == [
        "partial",
        "partial",
        "defrost",
        "defrost",
        "partial",
        "partial",
    ]


def test_partial_cycle_uses_water_setpoint_and_defrost_to_label_known_stages() -> None:
    timestamps = pd.date_range("2026-07-15 00:00:00", periods=5, freq="s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "defrost_active": [False, False, False, True, True],
            "water_out_temperature": [40.0, 45.0, 50.0, 50.0, 50.0],
            "water_temperature_setpoint": [50.0] * 5,
        }
    )

    labeled, summary = label_cycles(
        frame,
        "defrost_active",
        _short_cycle_settings(),
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    assert summary["cycle_id"].tolist() == ["partial_001"]
    assert labeled["cycle_stage"].tolist() == [
        "recovery",
        "recovery",
        "frost_development",
        "defrost",
        "defrost",
    ]


def test_partial_cycle_without_defrost_is_frost_after_starting_at_setpoint() -> None:
    timestamps = pd.date_range("2026-07-15 00:00:00", periods=4, freq="s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "defrost_active": [False] * 4,
            "water_out_temperature": [50.0] * 4,
            "water_temperature_setpoint": [50.0] * 4,
        }
    )

    labeled, _ = label_cycles(
        frame,
        "defrost_active",
        _short_cycle_settings(),
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    assert labeled["cycle_stage"].eq("frost_development").all()


def test_partial_cycle_without_setpoint_crossing_still_labels_defrost_boundary() -> None:
    timestamps = pd.date_range("2026-07-15 00:00:00", periods=4, freq="s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "defrost_active": [False, False, True, True],
            "water_out_temperature": [40.0, 41.0, 42.0, 43.0],
            "water_temperature_setpoint": [50.0] * 4,
        }
    )

    labeled, summary = label_cycles(
        frame,
        "defrost_active",
        _short_cycle_settings(),
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    assert labeled["cycle_stage"].tolist() == [
        "recovery",
        "recovery",
        "defrost",
        "defrost",
    ]
    assert pd.isna(summary.iloc[0]["stable_heating_start"])
    assert summary.iloc[0]["defrost_start"] == timestamps[2]


def test_long_defrost_state_gap_is_not_filled_and_marks_cycle_incomplete() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-07-15 00:00:00",
            "2026-07-15 00:00:01",
            "2026-07-15 00:00:10",
            "2026-07-15 00:00:11",
        ]
    )
    frame = pd.DataFrame(
        {"timestamp": timestamps, "defrost_active": [True, None, None, False]}
    )

    labeled, summary = label_cycles(
        frame,
        "defrost_active",
        {
            "maximum_state_gap_seconds": 5,
            "debounce_seconds": 0.5,
            "minimum_defrost_seconds": 1,
            "maximum_defrost_seconds": 100,
            "minimum_heating_seconds": 1,
            "maximum_heating_seconds": 100,
            "stable_heating_seconds": 1,
        },
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    assert labeled["defrost_active"].isna().any()
    assert not summary["cycle_id"].eq("cycle_001").any()
    assert summary["cycle_status"].eq("incomplete").all()


def test_long_gap_only_marks_the_intersecting_cycle_incomplete() -> None:
    timestamps = pd.date_range("2026-07-15", periods=23, freq="s")
    states: list[object] = [
        True,
        True,
        False,
        False,
        False,
        False,
        None,
        None,
        None,
        None,
        None,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    frame = pd.DataFrame({"timestamp": timestamps, "defrost_active": states})

    _, summary = label_cycles(
        frame,
        "defrost_active",
        {
            "maximum_state_gap_seconds": 5,
            "debounce_seconds": 0.5,
            "minimum_defrost_seconds": 1,
            "maximum_defrost_seconds": 100,
            "minimum_heating_seconds": 1,
            "maximum_heating_seconds": 100,
            "stable_heating_seconds": 1,
        },
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    statuses = summary.set_index("cycle_id")["cycle_status"]
    assert statuses["cycle_001"] == "incomplete"
    assert statuses["cycle_002"] == "valid"


def test_nan_inside_defrost_run_does_not_split_event() -> None:
    timestamps = pd.date_range("2026-07-15", periods=6, freq="s")
    state = pd.Series([True, True, np.nan, True, True, False], dtype="object")

    events = _defrost_runs(timestamps.to_series(index=range(6)), state, ())

    assert len(events) == 1
    assert events[0]["start"] == timestamps[0]
    assert events[0]["end"] == timestamps[5]


def test_short_nan_between_on_and_off_ends_at_explicit_off() -> None:
    timestamps = pd.date_range("2026-07-15", periods=3, freq="s")
    state = pd.Series([True, np.nan, False], dtype="object")

    events = _defrost_runs(timestamps.to_series(index=range(3)), state, ())

    assert len(events) == 1
    assert events[0]["end"] == timestamps[2]


def test_long_off_state_gap_keeps_cycle_but_marks_it_incomplete() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-07-15 00:00:00",
            "2026-07-15 00:00:01",
            "2026-07-15 00:00:02",
            "2026-07-15 00:00:03",
            "2026-07-15 00:00:10",
            "2026-07-15 00:00:11",
            "2026-07-15 00:00:12",
            "2026-07-15 00:00:13",
            "2026-07-15 00:00:14",
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "defrost_active": [True, True, False, None, None, False, True, True, False],
        }
    )

    labeled, summary = label_cycles(
        frame,
        "defrost_active",
        _short_cycle_settings(),
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    cycle = summary.loc[summary["cycle_id"].eq("cycle_001")].iloc[0]
    assert labeled["defrost_active"].isna().any()
    assert cycle["cycle_status"] == "incomplete"
    assert cycle["cycle_status_reason"] == "defrost_state_gap"


def test_long_on_state_gap_is_not_merged_into_a_complete_event() -> None:
    timestamps = pd.date_range("2026-07-15", periods=5, freq="s")
    state = pd.Series([True, np.nan, np.nan, True, False], dtype="object")
    long_gaps = ((timestamps[0], timestamps[3]),)

    events = _defrost_runs(timestamps.to_series(index=range(5)), state, long_gaps)

    assert len(events) == 2
    assert events[0]["end"] is None
    assert events[0]["boundary_uncertain"] is True
    assert events[1]["start"] == timestamps[3]
    assert events[1]["boundary_uncertain"] is True


def test_cycle_pairs_without_heating_start_are_skipped_and_numbered_continuously() -> None:
    timestamps = pd.date_range("2026-07-15", periods=24, freq="s")
    events = [
        {"start": timestamps[0], "end": None, "duration": None},
        {"start": timestamps[10], "end": timestamps[12], "duration": 2.0},
        {"start": timestamps[20], "end": timestamps[22], "duration": 2.0},
    ]
    labeled = pd.DataFrame({"timestamp": timestamps})

    cycles, ranges = _build_cycles(
        events,
        (),
        _short_cycle_settings(),
        labeled,
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    assert [row["cycle_id"] for row in cycles] == ["cycle_001"]
    assert len(ranges) == 1


def test_data_starting_on_does_not_create_a_phantom_cycle() -> None:
    frame = _cycle_frame_with_mode([True, True, False, False])

    labeled, summary = label_cycles(
        frame,
        "defrost_active",
        _short_cycle_settings(),
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    assert not summary["cycle_id"].str.startswith("cycle_").any()
    assert labeled["cycle_id"].eq("partial_001").all()


def test_data_ending_on_keeps_only_an_incomplete_cycle_with_data() -> None:
    frame = _cycle_frame_with_mode(
        [False, False, True, True, False, False, False, True, True]
    )

    labeled, summary = label_cycles(
        frame,
        "defrost_active",
        _short_cycle_settings(),
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    formal = summary.loc[summary["cycle_id"].eq("cycle_001")].iloc[0]
    assert formal["cycle_status"] == "incomplete"
    assert formal["cycle_status_reason"] == "defrost_end_not_observed"
    assert labeled.loc[labeled["cycle_id"].eq("cycle_001")].shape[0] > 0
    assert not summary["cycle_id"].eq("cycle_002").any()


def test_stable_start_uses_ts_minus_two_before_looser_thresholds() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=5, freq="s"),
            "water_out_temperature": [45.0, 47.5, 47.8, 48.2, 48.5],
            "water_temperature_setpoint": [50.0] * 5,
        }
    )

    stable = find_stable_heating_start(frame, start, start + pd.Timedelta(seconds=5), {})

    assert stable == start + pd.Timedelta(seconds=3)


def test_stable_start_falls_back_to_ts_minus_three_only_if_minus_two_is_absent() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=4, freq="s"),
            "water_out_temperature": [45.0, 46.5, 47.1, 47.2],
            "water_temperature_setpoint": [50.0] * 4,
        }
    )

    stable = find_stable_heating_start(frame, start, start + pd.Timedelta(seconds=4), {})

    assert stable == start + pd.Timedelta(seconds=2)


def test_stable_start_is_missing_when_all_priority_thresholds_have_no_observation() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=3, freq="s"),
            "water_out_temperature": [40.0, 44.0, 45.0],
            "water_temperature_setpoint": [50.0] * 3,
        }
    )

    stable = find_stable_heating_start(frame, start, start + pd.Timedelta(seconds=3), {})

    assert stable is None


def test_short_state_gap_with_irregular_timestamps_does_not_split_defrost_event() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-07-15 00:00:00",
            "2026-07-15 00:00:01",
            "2026-07-15 00:00:03",
            "2026-07-15 00:00:04",
            "2026-07-15 00:00:05",
            "2026-07-15 00:00:06",
        ]
    )
    state = pd.Series([False, True, None, True, True, False], dtype="object")

    filled, long_gaps = _fill_short_state_gaps(timestamps.to_series(), state, 5)
    events = _defrost_runs(timestamps.to_series(), filled, long_gaps)

    assert len(events) == 1
    assert events[0]["start"] == timestamps[1]
    assert events[0]["end"] == timestamps[5]
    assert not long_gaps


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "0715").is_dir(),
    reason="0715 raw data is not available",
)
def test_0715_raw_data_has_four_defrost_events_and_three_formal_cycles() -> None:
    config = load_config(ROOT / "configs" / "0715.yaml")
    channels = load_channels(config.channels_path)
    prepared, summary = prepare(config, channels)

    raw_state = prepared["defrost_active"].map(_normalize_state).astype("object")
    filled_state, long_gaps = _fill_short_state_gaps(
        prepared["timestamp"],
        raw_state,
        config.cycles.maximum_state_gap_seconds,
    )
    events = _defrost_runs(
        prepared["timestamp"],
        _debounce_state(
            prepared["timestamp"],
            filled_state,
            config.cycles.debounce_seconds,
        ),
        long_gaps,
    )
    formal = summary.loc[summary["cycle_id"].astype(str).str.startswith("cycle_")]

    assert len(events) == 4
    assert formal["cycle_id"].tolist() == ["cycle_001", "cycle_002", "cycle_003"]
    formal_counts = prepared.loc[
        prepared["cycle_id"].astype(str).str.startswith("cycle_"), "cycle_id"
    ].value_counts()
    assert set(formal_counts.index) == set(formal["cycle_id"])
    assert (formal_counts > 0).all()


def test_prepare_duplicate_only_masks_affected_channel(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "0715"
    raw.mkdir(parents=True)
    (raw / "sample参数1.xls").write_text(
        "时间\tT4\n"
        "2026-07-15 00:00:00\t10\n"
        "2026-07-15 00:00:00\t11\n"
        "2026-07-15 00:00:01\t12\n",
        encoding="utf-8",
    )
    (raw / "events参数1.xls").write_text(
        "时间\tDefrost\n"
        "2026-07-15 00:00:00\tON\n"
        "2026-07-15 00:00:01\tOFF\n",
        encoding="utf-8",
    )
    config = _config(tmp_path, raw)
    channels = {
        "ambient_temperature": {
            "source_names": ["p1__T4"],
            "unit": "degC",
            "kind": "continuous",
            "role": "context",
            "resample": "mean",
            "missing": "interpolate",
            "analysis_candidate": False,
        },
        "defrost_active": {
            "source_names": ["p1__Defrost"],
            "unit": None,
            "kind": "event",
            "role": "event",
            "resample": "last",
            "missing": "none",
            "analysis_candidate": False,
            "allowed_values": {"ON": True, "OFF": False},
        },
    }

    prepared, _ = prepare(config, channels)

    duplicate = prepared.loc[prepared["timestamp"].eq(pd.Timestamp("2026-07-15"))].iloc[0]
    assert pd.isna(duplicate["ambient_temperature"])
    assert bool(duplicate["ambient_temperature__duplicate"])
    assert pd.isna(duplicate["defrost_active"]) is False
    assert prepared[["experiment_id", "timestamp"]].duplicated().sum() == 0
    assert not any(
        any(token in str(column) for token in ("baseline", "rolling", "slope", "__imputed"))
        for column in prepared.columns
    )


def test_match_images_does_not_reuse_one_image() -> None:
    timestamps = pd.to_datetime(
        ["2026-07-15 00:00:00", "2026-07-15 00:00:01", "2026-07-15 00:00:02"]
    )
    image_files = [
        Path("192.168.1.1_1/20260715000000100.jpg"),
        Path("192.168.1.1_1/20260715000001000.jpg"),
    ]

    matched = match_images(
        timestamps,
        image_files,
        camera_roles={"192.168.1.1_1": "front_center"},
        tolerance_seconds=2,
    )

    assert matched["image_front_center_path"].dropna().is_unique
    assert matched["image_front_center_path"].notna().sum() == 2


def test_match_images_assigns_image_to_closest_sensor_timestamp() -> None:
    timestamps = pd.to_datetime(
        ["2026-07-15 00:00:00", "2026-07-15 00:00:01"]
    )
    image_files = [Path("192.168.1.1_1/20260715000000900.jpg")]

    matched = match_images(
        timestamps,
        image_files,
        camera_roles={"192.168.1.1_1": "front_center"},
        tolerance_seconds=2,
    )

    assert pd.isna(matched.loc[0, "image_front_center_path"])
    assert matched.loc[1, "image_front_center_path"].endswith(
        "20260715000000900.jpg"
    )
    assert matched.loc[1, "image_front_center_offset_seconds"] == pytest.approx(-0.1)


def test_match_images_keeps_same_time_images_for_separate_roles() -> None:
    timestamps = pd.to_datetime(["2026-07-15 00:00:00"])
    image_files = [
        Path("192.168.1.1_1/20260715000000000.jpg"),
        Path("192.168.1.2_1/20260715000000000.jpg"),
    ]

    matched = match_images(
        timestamps,
        image_files,
        camera_roles={
            "192.168.1.1_1": "front_center",
            "192.168.1.2_1": "left_near",
        },
        tolerance_seconds=2,
    )

    assert pd.notna(matched.loc[0, "image_front_center_path"])
    assert pd.notna(matched.loc[0, "image_left_near_path"])


def test_environment_alignment_uses_each_observation_once_and_main_timestamp() -> None:
    environment = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-15 00:00:00.500"]),
            "environment_temperature": [20.0],
            "environment_relative_humidity": [50.0],
        }
    )
    main_timestamps = pd.Series(
        pd.to_datetime(["2026-07-15 00:00:00", "2026-07-15 00:00:01"])
    )

    aligned = prepare_module._align_environment_to_main_timestamps(
        environment,
        main_timestamps,
        pd.Timedelta(seconds=0.5),
    )

    assert aligned["timestamp"].tolist() == [pd.Timestamp("2026-07-15 00:00:00")]
    assert aligned["environment_temperature"].tolist() == [20.0]
    assert aligned["environment_relative_humidity"].tolist() == [50.0]


def test_partial_regions_receive_separate_cycle_ids() -> None:
    timestamps = pd.date_range("2026-07-15", periods=12, freq="s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "defrost_active": [
                None,
                None,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                None,
                None,
                None,
            ],
        }
    )

    labeled, summary = label_cycles(
        frame,
        "defrost_active",
        {
            "maximum_state_gap_seconds": 1,
            "debounce_seconds": 0.5,
            "minimum_defrost_seconds": 1,
            "maximum_defrost_seconds": 10,
            "minimum_heating_seconds": 1,
            "maximum_heating_seconds": 10,
            "stable_heating_seconds": 0,
        },
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    partial_ids = sorted(labeled.loc[labeled["cycle_stage"].eq("partial"), "cycle_id"].unique())
    assert partial_ids == ["partial_001", "partial_002"]
    assert summary["cycle_id"].isin(partial_ids).sum() == 2


def test_prepare_enforces_configured_valid_range(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "0715"
    raw.mkdir(parents=True)
    (raw / "sample参数1.xls").write_text(
        "时间\tT4\n2026-07-15 00:00:00\t99\n",
        encoding="utf-8",
    )
    config = _config(tmp_path, raw)
    channels = {
        "ambient_temperature": {
            "source_names": ["p1__T4"],
            "unit": "degC",
            "kind": "continuous",
            "role": "context",
            "resample": "mean",
            "missing": "none",
            "analysis_candidate": False,
            "valid_range": [0, 10],
        },
        "defrost_active": {
            "source_names": ["p1__Defrost"],
            "unit": None,
            "kind": "event",
            "role": "event",
            "resample": "last",
            "missing": "none",
            "analysis_candidate": False,
            "allowed_values": {"ON": True, "OFF": False},
        },
    }

    prepared, _ = prepare(config, channels)

    assert pd.isna(prepared.loc[0, "ambient_temperature"])
    assert bool(prepared.loc[0, "ambient_temperature__invalid"])


def test_summary_expected_row_count_uses_left_closed_interval() -> None:
    row = pd.Series(
        {
            "heating_start": pd.Timestamp("2026-07-15 00:00:00"),
            "defrost_end": pd.Timestamp("2026-07-15 00:00:10"),
        }
    )

    assert _expected_row_count(row, 1) == 10


def test_image_gap_is_nan_when_fewer_than_two_images_exist() -> None:
    assert pd.isna(prepare_module._maximum_image_gap(pd.Series([pd.Timestamp("2026-07-15")])))
    assert pd.isna(prepare_module._maximum_image_gap(pd.Series(dtype="datetime64[ns]")))
    assert pd.isna(_maximum_gap(pd.Series([pd.Timestamp("2026-07-15")])))


def test_observed_fraction_uses_source_discovery_denominator() -> None:
    group = pd.DataFrame({"valid": [1.0, 2.0], "invalid": [float("nan"), float("nan")]})

    assert _observed_fraction(group, ["valid", "invalid"], {"valid", "invalid"}) == 0.5
    assert _observed_fraction(group, ["valid", "invalid"], {"valid"}) == 1.0


def test_config_rejects_impossible_iso_date(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
schema_version: 2
defaults_path: defaults.yaml
experiment_id: exp_test
experiment_date: "2026-02-31"
input_dir: data/0715
expected_sensor_interval_seconds: 1
camera_roles: {}
""",
        encoding="utf-8",
    )
    (tmp_path / "defaults.yaml").write_text(
        """
channels_path: channels.yaml
input_format:
  sensor_globs: ["*.xls"]
  image_extensions: [".jpg"]
  timestamp_column: 时间
  edf:
    pair_tolerance_seconds: 1.0
image_match_tolerance_seconds: 2
cycles: {}
process: {}
""",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ISO"):
        load_config(path)
