from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from frost_analysis.config import Config, load_config
from frost_analysis.cycles import label_cycles
from frost_analysis.images import match_images
from frost_analysis.prepare import prepare


def _config(root: Path, raw: Path) -> Config:
    (root / "camera.yaml").write_text("camera_roles: {}\n", encoding="utf-8")
    return Config(
        project_root=root,
        experiment_id="exp_test",
        experiment_date="2026-07-15",
        input_dir=raw,
        channels_path=root / "channels.yaml",
        camera_mapping_path=root / "camera.yaml",
        sensor_globs=("*.xls",),
        image_extensions=(".jpg",),
        timestamp_column="时间",
        expected_sensor_interval_seconds=1,
        image_match_tolerance_seconds=2,
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
        analysis={},
    )


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


def test_single_defrost_event_creates_only_partial_cycles() -> None:
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
    assert labeled["cycle_stage"].eq("partial").all()


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
    timestamps = pd.date_range("2026-07-15", periods=22, freq="s")
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

    prepared, _, _ = prepare(config, channels)

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

    prepared, _, _ = prepare(config, channels)

    assert pd.isna(prepared.loc[0, "ambient_temperature"])
    assert bool(prepared.loc[0, "ambient_temperature__invalid"])


def test_config_rejects_impossible_iso_date(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
experiment_id: exp_test
experiment_date: "2026-02-31"
input_dir: data/0715
channels_path: configs/channels.yaml
sensor_globs: ["*.xls"]
image_extensions: [".jpg"]
camera_mapping_path: configs/camera_mappings/0715.yaml
timestamp_column: 时间
expected_sensor_interval_seconds: 1
image_match_tolerance_seconds: 2
cycles: {}
process: {}
analysis: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ISO"):
        load_config(path)
