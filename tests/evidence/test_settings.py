from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from frost_analysis.evidence import EvidenceSettings


def _payload() -> dict[str, object]:
    return {
        "analysis_version": "frost-cycle-evidence-v2.2",
        "targets": ["heating_capacity", "cop"],
        "primary_target": "heating_capacity",
        "primary_horizon_minutes": 10,
        "horizons_minutes": [5, 10, 20],
        "minimum_feature_points": 12,
        "minimum_feature_coverage": 0.8,
        "minimum_valid_pairs": 30,
        "minimum_pair_coverage": 0.8,
        "onset_window_seconds": 60,
        "onset_mad_multiplier": 3.0,
        "onset_persistence_seconds": 60,
        "aggregation_method": "date_balanced_median_of_cycle_medians_v1",
    }


def test_settings_from_yaml_strictly_loads_scientific_contract(tmp_path: Path) -> None:
    path = tmp_path / "evidence.yaml"
    path.write_text(yaml.safe_dump(_payload()), encoding="utf-8")

    settings = EvidenceSettings.from_yaml(path)

    assert settings.targets == ("heating_capacity", "cop")
    assert settings.primary_horizon_minutes == 10
    assert settings.minimum_feature_coverage == pytest.approx(0.8)


def test_settings_rejects_unknown_or_missing_fields(tmp_path: Path) -> None:
    unknown = _payload()
    unknown["output"] = "not-scientific"
    unknown_path = tmp_path / "unknown.yaml"
    unknown_path.write_text(yaml.safe_dump(unknown), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        EvidenceSettings.from_yaml(unknown_path)

    missing = _payload()
    del missing["targets"]
    missing_path = tmp_path / "missing.yaml"
    missing_path.write_text(yaml.safe_dump(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="targets"):
        EvidenceSettings.from_yaml(missing_path)
