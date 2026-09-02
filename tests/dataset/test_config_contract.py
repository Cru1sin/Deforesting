from __future__ import annotations

from pathlib import Path

import pytest

from dataloader.channels import load_channels
from dataloader.config import CycleSettings, load_config


def test_python_defaults_match_the_frozen_raw_dataset_contract(tmp_path: Path) -> None:
    input_dir = tmp_path / "data" / "0715"
    config = load_config(
        project_root=tmp_path,
        experiment_date="2026-07-15",
        input_dir=input_dir,
    )

    assert config.experiment_id == "exp_20260715"
    assert config.input_dir == input_dir
    assert config.sensor_globs == ("*.xls", "*.edf")
    assert config.image_extensions == (".jpg", ".png")
    assert config.timestamp_column == "时间"
    assert config.expected_sensor_interval_seconds == 1
    assert config.image_match_tolerance_seconds == 2
    assert config.edf_pair_tolerance_seconds == 1
    assert config.cycles.maximum_state_gap_seconds == 5
    assert config.cycles.operating_mode_channel == "operating_mode"
    assert config.cycles.defrost_preparation_setpoint_drop_hz == 10
    assert config.cycles.defrost_preparation_lookback_seconds == 120
    assert config.process.resample_interval_seconds == 10


def test_python_channel_table_preserves_active_sensor_contract() -> None:
    channels = load_channels()

    assert len(channels) == 36
    assert channels["compressor_frequency_setpoint"]["source_names"] == [
        "p1__设定频率<1_00>"
    ]
    assert channels["plate_heat_exchanger_outlet_temperature"]["source_names"] == [
        "p1__T2(18.1)'2_20"
    ]
    assert channels["evaporator_inlet_temperature"]["source_names"] == [
        "p1__TL(28.1)'2_31'"
    ]
    assert channels["plate_heat_exchanger_inlet_temperature"]["source_names"] == [
        "p1__T2b(20.1)'2_20"
    ]
    assert channels["power_total"]["scale"] == pytest.approx(0.001)
    assert channels["defrost_active"]["allowed_values"] == {
        "ON": True,
        "OFF": False,
        "1": True,
        "0": False,
        "TRUE": True,
        "FALSE": False,
    }
    assert channels["cop"]["dependencies"] == ["heating_capacity", "power_total"]


def test_load_channels_returns_an_independent_table() -> None:
    first = load_channels()
    first["power_total"]["scale"] = 1

    assert load_channels()["power_total"]["scale"] == pytest.approx(0.001)


def test_sensor_coverage_and_analysis_candidates_remain_explicit() -> None:
    channels = load_channels()
    required = {
        name for name, settings in channels.items() if settings.get("coverage_required")
    }
    assert required == {
        name
        for name, settings in channels.items()
        if settings.get("role") == "sensor" and settings.get("kind") != "derived"
    } - {"fin_temperature"}
    assert channels["fin_temperature"]["coverage_required"] is False
    assert {
        name for name, settings in channels.items() if settings["analysis_candidate"]
    } == {
        "evaporating_pressure",
        "evaporating_temperature",
        "coil_temperature",
        "fin_temperature",
    }


def test_config_rejects_invalid_date(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        load_config(project_root=tmp_path, experiment_date="0724", input_dir=tmp_path)


def test_cycle_settings_still_validate_explicit_overrides() -> None:
    with pytest.raises(ValueError, match="minimum_defrost_seconds"):
        CycleSettings(minimum_defrost_seconds=120, maximum_defrost_seconds=60)

    settings = CycleSettings(maximum_state_gap_seconds=0)
    assert settings.maximum_state_gap_seconds == 0
