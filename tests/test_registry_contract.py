from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from frost_analysis.data.registry import (
    FeatureSpec,
    apply_feature_registry,
    load_feature_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def _mode_spec(raw_source: str) -> FeatureSpec:
    """Build the smallest mode specification needed by the contract test."""
    return FeatureSpec(
        feature_id="mode",
        canonical_name="operating_mode",
        raw_source=raw_source,
        meaning_zh="运行模式",
        physical_family="event_quality",
        source_type="event",
        unit="code",
        formula="",
        data_role="M",
        availability="current_history",
        deployment_status="confirmed",
        confidence="high",
        primary_or_validation="primary",
        analysis_enabled=False,
        notes="",
    )


def test_registry_exposes_numeric_mode_and_boolean_heating_flag() -> None:
    """Keep the raw mode code separate from its derived heating predicate."""
    frame = pd.DataFrame({"mode_source": [3, 2, None]})
    result = apply_feature_registry(frame, {"mode": _mode_spec("mode_source")})

    assert result.frame["operating_mode"].iloc[:2].tolist() == [3, 2]
    assert pd.isna(result.frame["operating_mode"].iloc[2])
    assert result.frame["is_heating"].iloc[:2].tolist() == [True, False]
    assert pd.isna(result.frame["is_heating"].iloc[2])


def test_registry_uses_qcomp_as_the_only_heating_capacity_channel() -> None:
    specs = load_feature_registry(ROOT / "configs" / "feature_registry.yaml")
    assert specs["heating_capacity"].raw_source == "p1__QComp10W'2_32"
    assert "refrigerant_heating_capacity_check" not in specs
    assert all("CCQ_Comp" not in (spec.raw_source or "") for spec in specs.values())


def test_registry_derives_absolute_pressure_ratio_and_keeps_pr_as_a_check() -> None:
    specs = load_feature_registry(ROOT / "configs" / "feature_registry.yaml")
    frame = pd.DataFrame(
        {
            "sensor_time": pd.date_range("2026-07-15", periods=2, freq="s"),
            "p1__Pc(31.2)'2_12'": [1.2, 1.3],
            "p1__Pe(34.2)'2_12'": [0.3, 0.325],
            "p1__Pr'2_20": [400.0, 400.0],
        }
    )
    result = apply_feature_registry(frame, specs)
    assert np.allclose(result.frame["pressure_ratio"], [4.0, 4.0])
    assert result.specs["controller_pressure_ratio"].primary_or_validation == "validation"


def test_registry_keeps_pending_channels_in_metadata_when_source_is_present() -> None:
    specs = load_feature_registry(ROOT / "configs" / "feature_registry.yaml")
    frame = pd.DataFrame(
        {
            "sensor_time": pd.date_range("2026-07-15", periods=3, freq="s"),
            "p1__TL(28.1)'2_31'": [1.0, 1.1, 1.2],
        }
    )
    result = apply_feature_registry(frame, specs)
    assert "evaporator_inlet_temperature" in result.specs
    assert result.specs["evaporator_inlet_temperature"].deployment_status == "pending"
    assert "evaporator_inlet_temperature" in result.frame


def test_registry_counts_string_event_states_as_observed() -> None:
    specs = load_feature_registry(ROOT / "configs" / "feature_registry.yaml")
    frame = pd.DataFrame(
        {
            "p1__Deforst": ["OFF", "ON", None],
            "sensor_time": pd.date_range("2026-07-15", periods=3, freq="s"),
        }
    )
    result = apply_feature_registry(frame, specs)
    row = result.metadata.loc[result.metadata["canonical_name"].eq("defrost_flag")].iloc[0]
    assert int(row["observed_count"]) == 2
