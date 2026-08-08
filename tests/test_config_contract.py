from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from frost_analysis.channels import load_channels
from frost_analysis.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _write_v2_config(
    root: Path,
    *,
    schema_version: int | None = 2,
    defaults: dict[str, object] | None = None,
    overrides: dict[str, object] | None = None,
    camera_roles: dict[str, str] | None = None,
) -> Path:
    configs = root / "configs"
    configs.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    defaults_mapping: dict[str, object] = {
        "channels_path": "channels.yaml",
        "input_format": {
            "sensor_globs": ["*.xls"],
            "image_extensions": [".jpg", ".png"],
            "timestamp_column": "时间",
            "edf": {"pair_tolerance_seconds": 1.0},
        },
        "image_match_tolerance_seconds": 2,
        "cycles": {
            "defrost_channel": "defrost_active",
            "maximum_state_gap_seconds": 0,
            "debounce_seconds": 20,
            "minimum_defrost_seconds": 60,
            "maximum_defrost_seconds": 1200,
            "minimum_heating_seconds": 1800,
            "maximum_heating_seconds": 21600,
            "stable_heating_seconds": 180,
            "operating_mode_channel": "operating_mode",
            "required_operating_mode": "3",
        },
        "process": {
            "resample_interval_seconds": 10,
            "minimum_continuous_bucket_coverage": 0.8,
            "continuous_max_gap_seconds": 60,
            "control_max_gap_seconds": 30,
            "baseline": {
                "stage": "frost_development",
                "baseline_seconds": 60,
                "minimum_observed_coverage": 0.8,
                "required_anchor_channels": [
                    "ambient_temperature",
                    "water_in_temperature",
                    "water_out_temperature",
                    "compressor_frequency",
                ],
                "anchor_maximum_std": {
                    "ambient_temperature": 1.0,
                    "water_in_temperature": 1.0,
                    "water_out_temperature": 1.0,
                    "compressor_frequency": 5.0,
                },
            },
        },
    }
    if defaults:
        defaults_mapping.update(defaults)
    (configs / "defaults.yaml").write_text(
        yaml.safe_dump(defaults_mapping, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (configs / "channels.yaml").write_text("channels: {}\n", encoding="utf-8")
    date_mapping: dict[str, object] = {
        "defaults_path": "defaults.yaml",
        "experiment_id": "exp_test",
        "experiment_date": "2026-07-15",
        "input_dir": "data/0715",
        "expected_sensor_interval_seconds": 1,
        "camera_roles": camera_roles or {},
    }
    if schema_version is not None:
        date_mapping["schema_version"] = schema_version
    if overrides is not None:
        date_mapping["overrides"] = overrides
    config_path = configs / "0715.yaml"
    config_path.write_text(
        yaml.safe_dump(date_mapping, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def test_formal_config_uses_v2_contract_and_ten_second_resampling() -> None:
    config = load_config(ROOT / "configs" / "0715.yaml")

    assert config.experiment_id == "exp_20260715"
    assert config.input_dir == ROOT / "data" / "0715"
    assert config.process.resample_interval_seconds == 10
    assert config.cycles.maximum_state_gap_seconds == 5
    assert config.timestamp_column == "时间"
    assert not hasattr(config, "analysis")


def test_all_formal_date_configs_allow_short_defrost_state_gaps() -> None:
    for date in ("0714", "0715", "0716", "0717", "0720", "0721", "0722"):
        config = load_config(ROOT / "configs" / f"{date}.yaml")
        assert config.cycles.maximum_state_gap_seconds == 5


def test_formal_channels_include_frequency_setpoint_control() -> None:
    channels = load_channels(ROOT / "configs" / "channels.yaml")

    assert channels["compressor_frequency_setpoint"] == {
        "source_names": ["p1__设定频率<1_00>"],
        "unit": "Hz",
        "kind": "step",
        "role": "control",
        "resample": "last",
        "missing": "forward_fill",
        "analysis_candidate": False,
        "valid_range": [0, 200],
    }


def test_v2_config_loads_defaults_and_inline_camera_roles(tmp_path: Path) -> None:
    path = _write_v2_config(tmp_path, camera_roles={"camera_01": "front"})

    config = load_config(path)

    assert config.channels_path == tmp_path / "configs" / "channels.yaml"
    assert config.camera_roles == {"camera_01": "front"}
    assert config.process.baseline.stage == "frost_development"
    assert config.edf_pair_tolerance_seconds == 1.0


@pytest.mark.parametrize("value", [0, -0.1, float("nan"), float("inf"), float("-inf")])
def test_config_rejects_nonpositive_edf_pair_tolerance(tmp_path: Path, value: float) -> None:
    path = _write_v2_config(
        tmp_path,
        overrides={"input_format": {"edf": {"pair_tolerance_seconds": value}}},
    )

    with pytest.raises(ValueError, match="edf_pair_tolerance_seconds"):
        load_config(path)


def test_formal_config_discovers_edf_inputs() -> None:
    config = load_config(ROOT / "configs" / "0720.yaml")

    assert config.sensor_globs == ("*.xls", "*.edf")


@pytest.mark.parametrize("schema_version", [None, 1, 3])
def test_v2_loader_rejects_missing_or_unsupported_schema_version(
    tmp_path: Path, schema_version: int | None
) -> None:
    path = _write_v2_config(tmp_path, schema_version=schema_version)

    with pytest.raises(ValueError, match="schema_version"):
        load_config(path)


def test_v2_loader_merges_existing_nested_overrides(tmp_path: Path) -> None:
    path = _write_v2_config(
        tmp_path,
        overrides={
            "cycles": {"required_operating_mode": "4"},
            "process": {"baseline": {"baseline_seconds": 90}},
        },
    )

    config = load_config(path)

    assert config.cycles.required_operating_mode == "4"
    assert config.process.baseline.baseline_seconds == 90


def test_v2_loader_rejects_removed_process_features_override(tmp_path: Path) -> None:
    path = _write_v2_config(
        tmp_path,
        overrides={"process": {"features": {"windows_minutes": [1, 5, 15]}}},
    )

    with pytest.raises(ValueError, match="unknown override"):
        load_config(path)


def test_v2_loader_applies_state_gap_override_without_changing_defaults(tmp_path: Path) -> None:
    path = _write_v2_config(
        tmp_path,
        overrides={"cycles": {"maximum_state_gap_seconds": 5}},
    )

    config = load_config(path)
    defaults = yaml.safe_load((tmp_path / "configs" / "defaults.yaml").read_text())

    assert config.cycles.maximum_state_gap_seconds == 5
    assert defaults["cycles"]["maximum_state_gap_seconds"] == 0


def test_v2_loader_rejects_unknown_override_path(tmp_path: Path) -> None:
    path = _write_v2_config(
        tmp_path,
        overrides={"cycles": {"unknown_parameter": 123}},
    )

    with pytest.raises(ValueError, match="unknown override"):
        load_config(path)


def test_v2_loader_rejects_unknown_dataclass_field(tmp_path: Path) -> None:
    path = _write_v2_config(
        tmp_path,
        defaults={"process": {"unknown_parameter": 123}},
    )

    with pytest.raises(ValueError, match="process"):
        load_config(path)


def test_v2_loader_preserves_zero_and_rejects_negative_state_gap(tmp_path: Path) -> None:
    zero_path = _write_v2_config(tmp_path / "zero")
    assert load_config(zero_path).cycles.maximum_state_gap_seconds == 0

    negative_path = _write_v2_config(
        tmp_path / "negative",
        defaults={"cycles": {"maximum_state_gap_seconds": -1}},
    )
    with pytest.raises(ValueError, match="maximum_state_gap_seconds"):
        load_config(negative_path)


def test_v2_loader_accepts_sensor_only_camera_roles(tmp_path: Path) -> None:
    path = _write_v2_config(tmp_path, camera_roles={})

    assert load_config(path).camera_roles == {}


def test_new_config_loads_relative_paths_and_10_second_process_settings(
    tmp_path: Path,
) -> None:
    config = load_config(_write_v2_config(tmp_path))

    assert config.experiment_id == "exp_test"
    assert config.experiment_date == "2026-07-15"
    assert config.input_dir == tmp_path / "data" / "0715"
    assert config.channels_path == tmp_path / "configs" / "channels.yaml"
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


def test_production_channels_declare_required_sensor_coverage() -> None:
    channels = load_channels(ROOT / "configs" / "channels.yaml")

    required = {
        name for name, settings in channels.items() if settings.get("coverage_required")
    }
    assert required == {
        name
        for name, settings in channels.items()
        if settings.get("role") == "sensor" and settings.get("kind") != "derived"
    } - {"fin_temperature"}
    assert channels["fin_temperature"]["coverage_required"] is False
    assert required


def test_config_rejects_invalid_window_relationships(tmp_path: Path) -> None:
    path = _write_v2_config(
        tmp_path,
        defaults={
            "cycles": {
                "minimum_defrost_seconds": 120,
                "maximum_defrost_seconds": 60,
            }
        },
    )

    with pytest.raises(ValueError, match="minimum_defrost_seconds"):
        load_config(path)


def test_config_rejects_nonpositive_input_interval(tmp_path: Path) -> None:
    path = _write_v2_config(tmp_path)
    date_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    date_config["expected_sensor_interval_seconds"] = 0
    path.write_text(yaml.safe_dump(date_config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="expected_sensor_interval_seconds"):
        load_config(path)


def test_config_accepts_zero_maximum_state_gap(tmp_path: Path) -> None:
    path = _write_v2_config(tmp_path)

    config = load_config(path)

    assert config.cycles.maximum_state_gap_seconds == 0
