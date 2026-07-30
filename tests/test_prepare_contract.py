from __future__ import annotations

from pathlib import Path

import pandas as pd

from frost_analysis.config import Config
from frost_analysis.cycles import label_cycles
from frost_analysis.images import match_images
from frost_analysis.prepare import prepare


def _config(root: Path, raw: Path) -> Config:
    return Config(
        project_root=root,
        experiment_id="exp_test",
        experiment_date="2026-07-15",
        input_dir=raw,
        channels_path=root / "channels.yaml",
        sensor_globs=("*.xls",),
        image_extensions=(".jpg",),
        camera_mapping_file="IPlocation.yaml",
        cycles={
            "defrost_channel": "defrost_active",
            "maximum_state_gap_seconds": 5,
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
        {"maximum_state_gap_seconds": 5, "stable_heating_seconds": 2},
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    cycle = summary.loc[summary["cycle_status"].eq("valid")].iloc[0]
    development = labeled.loc[labeled["cycle_stage"].eq("frost_development")]
    assert cycle["cycle_status"] == "valid"
    assert development["cycle_elapsed_seconds"].notna().all()
    assert development["cycle_progress"].between(0, 1).all()
    assert labeled.loc[labeled["cycle_stage"] != "frost_development", "cycle_progress"].isna().all()


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
        {"maximum_state_gap_seconds": 5, "stable_heating_seconds": 1},
        experiment_id="exp_test",
        experiment_date="2026-07-15",
    )

    assert labeled["defrost_active"].isna().any()
    assert "defrost_state_gap" in set(summary["cycle_status_reason"])
    assert summary["cycle_status"].eq("incomplete").any()


def test_prepare_duplicate_only_masks_affected_channel(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "0715"
    raw.mkdir(parents=True)
    (raw / "sample.xls").write_text(
        "时间\tT4\n"
        "2026-07-15 00:00:00\t10\n"
        "2026-07-15 00:00:00\t11\n"
        "2026-07-15 00:00:01\t12\n",
        encoding="utf-8",
    )
    (raw / "events.xls").write_text(
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

    matched = match_images(timestamps, image_files, tolerance_seconds=2)

    assert matched["image_path"].dropna().is_unique
    assert matched["image_path"].notna().sum() == 2
