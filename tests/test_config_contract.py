from __future__ import annotations

from pathlib import Path

from frost_analysis.config import load_app_config

ROOT = Path(__file__).resolve().parents[1]


def test_config_exposes_nested_missing_resample_and_window_policies() -> None:
    config = load_app_config(ROOT / "configs" / "0715.yaml")

    assert config.prepare.duplicate_conflict_policy == "warn_keep_stable"
    assert config.process.resample_interval_seconds == 30
    assert config.process.continuous_max_gap_seconds == 60
    assert config.process.control_max_gap_seconds == 30
    assert config.process.minimum_observed_coverage == 0.8
    assert config.process.minimum_available_coverage == 0.95
    assert config.process.maximum_imputed_fraction == 0.15
    assert config.process.maximum_raw_gap_seconds == 60
    assert "evaporating_temperature" in config.analysis.features
    assert config.analysis.modalities["sensor"]["required"] is True
    assert config.analysis.modalities["rgb"]["required_camera_roles"] == []
