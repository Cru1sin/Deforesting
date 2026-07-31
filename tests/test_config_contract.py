from __future__ import annotations

from pathlib import Path

import pytest

from frost_analysis.channels import load_channels
from frost_analysis.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_formal_config_uses_flat_contract_and_ten_second_resampling() -> None:
    config = load_config(ROOT / "configs" / "0715.yaml")

    assert config.experiment_id == "exp_20260715"
    assert config.input_dir == ROOT / "data" / "0715"
    assert config.process.resample_interval_seconds == 10
    assert config.cycles.maximum_state_gap_seconds == 0
    assert config.timestamp_column == "时间"
    assert config.camera_mapping_path == ROOT / "configs" / "camera_mappings" / "0715.yaml"


def test_new_config_loads_relative_paths_and_10_second_process_settings() -> None:
    config = load_config(ROOT / "tests" / "fixtures" / "configs" / "minimal.yaml")

    assert config.experiment_id == "exp_minimal"
    assert config.experiment_date == "2026-07-15"
    assert config.input_dir == ROOT / "data" / "0715"
    assert config.channels_path == ROOT / "configs" / "channels.yaml"
    assert config.sensor_globs == ("*.xls",)
    assert config.image_extensions == (".jpg", ".png")
    assert config.process.resample_interval_seconds == 10
    assert config.expected_sensor_interval_seconds == 1


def test_channels_reject_candidate_event_and_invalid_derived_contracts(tmp_path: Path) -> None:
    valid = tmp_path / "valid.yaml"
    valid.write_text(
        """
channels:
  temperature:
    source_names: ["p1__T4"]
    unit: degC
    kind: continuous
    role: context
    resample: mean
    missing: interpolate
    analysis_candidate: true
    expected_frost_direction: decrease
""",
        encoding="utf-8",
    )
    channels = load_channels(valid)
    assert channels["temperature"]["kind"] == "continuous"

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        """
channels:
  defrost_active:
    source_names: ["p1__Defrost"]
    unit: null
    kind: event
    role: event
    resample: last
    missing: none
    analysis_candidate: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="analysis_candidate"):
        load_channels(invalid)


def test_channels_require_exact_sources_and_validate_scale_and_formula(tmp_path: Path) -> None:
    path = tmp_path / "channels.yaml"
    path.write_text(
        """
channels:
  power_total:
    source_names: ["p2__总功率"]
    unit: kW
    kind: continuous
    role: performance
    resample: mean
    missing: none
    analysis_candidate: false
    scale: 0.001
    offset: 0.0
  cop:
    unit: dimensionless
    kind: derived
    role: performance
    resample: mean
    missing: none
    analysis_candidate: false
    formula: cop
    dependencies: [heating_capacity, power_total]
""",
        encoding="utf-8",
    )

    channels = load_channels(path)

    assert channels["power_total"]["scale"] == 0.001
    assert channels["cop"]["dependencies"] == ["heating_capacity", "power_total"]


def test_config_rejects_invalid_window_relationships(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    (tmp_path / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    path.write_text(
        """
experiment_id: exp_test
experiment_date: "2026-07-15"
input_dir: data/0715
channels_path: configs/channels.yaml
camera_mapping_path: configs/camera_mappings/0715.yaml
sensor_globs: ["*.xls"]
image_extensions: [".jpg"]
timestamp_column: 时间
expected_sensor_interval_seconds: 1
image_match_tolerance_seconds: 2
cycles:
  maximum_state_gap_seconds: 5
  debounce_seconds: 20
  minimum_defrost_seconds: 120
  maximum_defrost_seconds: 60
process:
  resample_interval_seconds: 10
analysis: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="minimum_defrost_seconds"):
        load_config(path)


def test_config_rejects_nonpositive_input_interval(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    (tmp_path / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    path.write_text(
        """
experiment_id: exp_test
experiment_date: "2026-07-15"
input_dir: data/0715
channels_path: configs/channels.yaml
camera_mapping_path: configs/camera_mappings/0715.yaml
sensor_globs: ["*.xls"]
image_extensions: [".jpg"]
timestamp_column: 时间
expected_sensor_interval_seconds: 0
image_match_tolerance_seconds: 2
cycles: {}
process: {resample_interval_seconds: 10}
analysis: {future_horizon_minutes: 10}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_sensor_interval_seconds"):
        load_config(path)


def test_config_accepts_zero_maximum_state_gap(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    (tmp_path / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    path.write_text(
        """
experiment_id: exp_test
experiment_date: "2026-07-15"
input_dir: data/0715
channels_path: configs/channels.yaml
camera_mapping_path: configs/camera_mappings/0715.yaml
sensor_globs: ["*.xls"]
image_extensions: [".jpg"]
timestamp_column: 时间
expected_sensor_interval_seconds: 1
image_match_tolerance_seconds: 2
cycles:
  maximum_state_gap_seconds: 0
process: {resample_interval_seconds: 10}
analysis: {future_horizon_minutes: 10}
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.cycles.maximum_state_gap_seconds == 0
