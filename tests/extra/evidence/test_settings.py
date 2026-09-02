from __future__ import annotations

import pytest

from frost_analysis.evidence import EvidenceSettings


def _payload() -> dict[str, object]:
    return {
        "targets": ["heating_capacity", "cop"],
        "primary_target": "heating_capacity",
        "minimum_feature_points": 12,
        "minimum_feature_coverage": 0.8,
        "minimum_valid_pairs": 30,
        "minimum_pair_coverage": 0.8,
        "eligible_statuses": ["valid"],
        "minimum_cycle_minutes": 30.0,
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


def test_default_settings_preserve_the_frozen_scientific_contract() -> None:
    settings = EvidenceSettings()

    assert settings.targets == ("heating_capacity", "cop")
    assert settings.primary_horizon_minutes == 10
    assert settings.minimum_feature_coverage == pytest.approx(0.8)
    assert settings.event_thresholds == (0.05, 0.1, 0.15)
    assert settings.context_features[-1] == "compressor_frequency"
    assert settings.eligible_statuses == ("valid",)
    assert settings.minimum_cycle_minutes == pytest.approx(30.0)
    assert settings.normalized() == {
        **_payload(),
        "primary_horizon_minutes": 10,
        "horizons_minutes": [5, 10, 20],
        "event_thresholds": [0.05, 0.1, 0.15],
        "primary_event_threshold": 0.1,
    }


def test_settings_post_init_checks_only_required_relationships() -> None:
    with pytest.raises(ValueError, match="primary_target"):
        EvidenceSettings(
            targets=("heating_capacity",),
            primary_target="cop",
            minimum_feature_points=1,
            minimum_feature_coverage=0.8,
            minimum_valid_pairs=1,
            minimum_pair_coverage=0.8,
        )
    with pytest.raises(ValueError, match="eligible_statuses"):
        EvidenceSettings(
            targets=("heating_capacity",),
            primary_target="heating_capacity",
            minimum_feature_points=1,
            minimum_feature_coverage=0.8,
            minimum_valid_pairs=1,
            minimum_pair_coverage=0.8,
            eligible_statuses=(),
        )
    with pytest.raises(ValueError, match="minimum_cycle_minutes"):
        EvidenceSettings(
            targets=("heating_capacity",),
            primary_target="heating_capacity",
            minimum_feature_points=1,
            minimum_feature_coverage=0.8,
            minimum_valid_pairs=1,
            minimum_pair_coverage=0.8,
            minimum_cycle_minutes=-1.0,
        )
    with pytest.raises(ValueError, match="feature_coverage"):
        EvidenceSettings(
            targets=("heating_capacity",),
            primary_target="heating_capacity",
            minimum_feature_points=1,
            minimum_feature_coverage=1.1,
            minimum_valid_pairs=1,
            minimum_pair_coverage=0.8,
        )


def test_settings_constructor_contains_only_scientific_parameters() -> None:
    evidence_settings = EvidenceSettings(
        targets=("heating_capacity", "cop"),
        primary_target="heating_capacity",
        minimum_feature_points=12,
        minimum_feature_coverage=0.8,
        minimum_valid_pairs=30,
        minimum_pair_coverage=0.8,
    )

    assert evidence_settings.primary_target == "heating_capacity"
    assert not hasattr(evidence_settings, "analysis_version")


def test_settings_protocol_fields_are_not_constructor_parameters() -> None:
    common = {
        "targets": ("heating_capacity", "cop"),
        "primary_target": "heating_capacity",
        "minimum_feature_points": 12,
        "minimum_feature_coverage": 0.8,
        "minimum_valid_pairs": 30,
        "minimum_pair_coverage": 0.8,
    }
    with pytest.raises(TypeError):
        EvidenceSettings(horizons_minutes=(5, 15, 30), **common)
    with pytest.raises(TypeError):
        EvidenceSettings(
            primary_horizon_minutes=20,
            event_thresholds=(0.08, 0.12),
            primary_event_threshold=0.08,
            **common,
        )
