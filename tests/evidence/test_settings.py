from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from frost_analysis.evidence import EvidenceSettings


def _payload() -> dict[str, object]:
    return {
        "targets": ["heating_capacity", "cop"],
        "primary_target": "heating_capacity",
        "primary_horizon_minutes": 10,
        "horizons_minutes": [5, 10, 20],
        "minimum_feature_points": 12,
        "minimum_feature_coverage": 0.8,
        "minimum_valid_pairs": 30,
        "minimum_pair_coverage": 0.8,
        "event_thresholds": [0.05, 0.1, 0.15],
        "primary_event_threshold": 0.1,
        "event_persistence_seconds": 120,
        "signal_reference_minutes": 5,
        "signal_smoothing_seconds": 60,
        "signal_mad_multiplier": 3.0,
        "signal_persistence_seconds": 60,
        "dynamic_window_minutes": 5,
        "ridge_alpha": 1.0,
        "context_features": [
            "ambient_temperature",
            "environment_relative_humidity",
            "water_in_temperature",
            "water_flow",
            "compressor_frequency",
        ],
    }


def test_settings_from_yaml_loads_only_scientific_contract(tmp_path: Path) -> None:
    path = tmp_path / "evidence.yaml"
    path.write_text(yaml.safe_dump(_payload()), encoding="utf-8")

    settings = EvidenceSettings.from_yaml(path)

    assert settings.targets == ("heating_capacity", "cop")
    assert settings.primary_horizon_minutes == 10
    assert settings.minimum_feature_coverage == pytest.approx(0.8)
    assert settings.event_thresholds == (0.05, 0.1, 0.15)
    assert settings.context_features[-1] == "compressor_frequency"
    assert settings.normalized() == _payload()


def test_settings_ignores_legacy_non_scientific_fields(tmp_path: Path) -> None:
    payload = {
        **_payload(),
        "analysis_version": "legacy",
        "aggregation_method": "legacy",
    }
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    settings = EvidenceSettings.from_yaml(path)

    assert not hasattr(settings, "analysis_version")
    assert not hasattr(settings, "aggregation_method")


def test_settings_post_init_checks_only_required_relationships() -> None:
    with pytest.raises(ValueError, match="primary_target"):
        EvidenceSettings(
            targets=("heating_capacity",),
            primary_target="cop",
            primary_horizon_minutes=10,
            horizons_minutes=(10,),
            minimum_feature_points=1,
            minimum_feature_coverage=0.8,
            minimum_valid_pairs=1,
            minimum_pair_coverage=0.8,
        )
    with pytest.raises(ValueError, match="primary_horizon"):
        EvidenceSettings(
            targets=("heating_capacity",),
            primary_target="heating_capacity",
            primary_horizon_minutes=20,
            horizons_minutes=(10,),
            minimum_feature_points=1,
            minimum_feature_coverage=0.8,
            minimum_valid_pairs=1,
            minimum_pair_coverage=0.8,
        )
    with pytest.raises(ValueError, match="feature_coverage"):
        EvidenceSettings(
            targets=("heating_capacity",),
            primary_target="heating_capacity",
            primary_horizon_minutes=10,
            horizons_minutes=(10,),
            minimum_feature_points=1,
            minimum_feature_coverage=1.1,
            minimum_valid_pairs=1,
            minimum_pair_coverage=0.8,
        )


def test_settings_constructor_contains_only_scientific_parameters() -> None:
    evidence_settings = EvidenceSettings(
        targets=("heating_capacity", "cop"),
        primary_target="heating_capacity",
        primary_horizon_minutes=10,
        horizons_minutes=(5, 10, 20),
        minimum_feature_points=12,
        minimum_feature_coverage=0.8,
        minimum_valid_pairs=30,
        minimum_pair_coverage=0.8,
    )

    assert evidence_settings.primary_target == "heating_capacity"
    assert not hasattr(evidence_settings, "analysis_version")
