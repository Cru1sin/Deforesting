from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frost_analysis.data.registry import (
    FeatureSpec,
    apply_feature_registry,
    load_feature_registry,
    recompute_derived_features,
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


def test_registry_projects_source_states_to_standard_channel_names() -> None:
    spec = _mode_spec("mode_source")
    frame = pd.DataFrame(
        {
            "mode_source": [3, None],
            "mode_source__missing": [False, True],
            "mode_source__invalid": [False, False],
            "mode_source__source_state": ["observed", "missing"],
        }
    )

    result = apply_feature_registry(frame, {"mode": spec})

    assert result.frame["operating_mode__missing"].tolist() == [False, True]
    assert result.frame["operating_mode__invalid"].tolist() == [False, False]
    assert result.frame["operating_mode__source_state"].tolist() == [
        "observed",
        "missing",
    ]


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


def test_registry_loads_missing_policy_and_required_quality_metadata(tmp_path: Path) -> None:
    path = tmp_path / "registry.yaml"
    path.write_text(
        """
registry_version: 1
features:
  - feature_id: temperature
    canonical_name: temperature
    raw_source: p1__temperature
    data_kind: continuous
    missing_policy: linear
    resample_method: mean
    required_for_sensor_quality: true
""",
        encoding="utf-8",
    )

    spec = load_feature_registry(path)["temperature"]

    assert spec.data_kind == "continuous"
    assert spec.missing_policy == "linear"
    assert spec.resample_method == "mean"
    assert spec.required_for_sensor_quality is True


def test_derived_features_are_recomputed_from_processed_source_values() -> None:
    frame = pd.DataFrame(
        {
            "heating_capacity": [10.0, np.nan],
            "power_total": [2.0, 2.0],
            "water_in_temperature": [10.0, 10.0],
            "water_out_temperature": [12.0, 13.0],
            "water_flow": [1.0, 1.0],
        }
    )

    initial = recompute_derived_features(frame)
    assert pd.isna(initial.loc[1, "cop"])

    processed = frame.copy()
    processed.loc[1, "heating_capacity"] = 12.0
    result = recompute_derived_features(processed)

    assert result["cop"].tolist() == [5.0, 6.0]
    assert result["water_delta_temperature"].tolist() == [2.0, 3.0]
    assert result["water_heating_capacity"].tolist() == [
        pytest.approx(2.32556),
        pytest.approx(3.48834),
    ]
