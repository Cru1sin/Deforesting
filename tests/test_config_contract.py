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
    assert config.process["resample_interval_seconds"] == 10
    assert config.cycles["maximum_state_gap_seconds"] == 5


def test_new_config_loads_relative_paths_and_10_second_process_settings() -> None:
    config = load_config(ROOT / "tests" / "fixtures" / "configs" / "minimal.yaml")

    assert config.experiment_id == "exp_minimal"
    assert config.experiment_date == "2026-07-15"
    assert config.input_dir == ROOT / "data" / "0715"
    assert config.channels_path == ROOT / "configs" / "channels.yaml"
    assert config.sensor_globs == ("*.xls", "*.xlsx")
    assert config.image_extensions == (".jpg", ".png")
    assert config.process["resample_interval_seconds"] == 10


def test_channels_reject_candidate_event_and_invalid_derived_contracts(tmp_path: Path) -> None:
    valid = tmp_path / "valid.yaml"
    valid.write_text(
        """
channels:
  temperature:
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
