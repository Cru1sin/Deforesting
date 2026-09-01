from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from frost_analysis.dataset.channels import load_channels
from frost_analysis.dataset.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def _write_config(root: Path, **updates: object) -> Path:
    config_path = root / "configs/config.yaml"
    config_path.parent.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    values = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    values.update(updates)
    config_path.write_text(
        yaml.safe_dump(values, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return config_path


def _load(path: Path, *, date: str = "2026-07-15"):
    return load_config(path, experiment_date=date, input_dir=path.parent / "input")


def test_shared_config_builds_date_facts_and_ten_second_resampling() -> None:
    input_dir = ROOT / "data/0715"
    config = load_config(
        ROOT / "configs/config.yaml",
        experiment_date="2026-07-15",
        input_dir=input_dir,
    )

    assert config.experiment_id == "exp_20260715"
    assert config.input_dir == input_dir
    assert config.channels_path == ROOT / "configs/config.yaml"
    assert config.process.resample_interval_seconds == 10
    assert config.cycles.maximum_state_gap_seconds == 5
    assert config.cycles.defrost_preparation_setpoint_drop_hz == 10
    assert config.cycles.defrost_preparation_lookback_seconds == 120
    assert config.sensor_globs == ("*.xls", "*.edf")
    assert load_channels(config.channels_path)["defrost_active"]["kind"] == "event"


def test_shared_config_needs_no_date_yaml() -> None:
    config = load_config(
        ROOT / "configs/config.yaml",
        experiment_date="2026-07-24",
        input_dir=ROOT / "data/0724",
    )

    assert config.experiment_id == "exp_20260724"
    assert list((ROOT / "configs").glob("07*.yaml")) == []


def test_config_rejects_invalid_date(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        _load(_write_config(tmp_path), date="0724")


@pytest.mark.parametrize("value", [0, -0.1, float("nan"), float("inf")])
def test_config_rejects_nonpositive_edf_pair_tolerance(
    tmp_path: Path, value: float
) -> None:
    input_format = yaml.safe_load(
        (ROOT / "configs/config.yaml").read_text(encoding="utf-8")
    )["input_format"]
    input_format["edf"]["pair_tolerance_seconds"] = value

    with pytest.raises(ValueError, match="edf_pair_tolerance_seconds"):
        _load(_write_config(tmp_path, input_format=input_format))


def test_formal_channels_include_frequency_setpoint_control() -> None:
    channels = load_channels(ROOT / "configs/config.yaml")

    assert channels["compressor_frequency_setpoint"]["source_names"] == [
        "p1__设定频率<1_00>"
    ]


def test_formal_channels_include_diagram_temperature_points() -> None:
    channels = load_channels(ROOT / "configs/config.yaml")

    assert channels["plate_heat_exchanger_outlet_temperature"]["source_names"] == [
        "p1__T2(18.1)'2_20"
    ]
    assert channels["evaporator_inlet_temperature"]["source_names"] == [
        "p1__TL(28.1)'2_31'"
    ]
    assert channels["plate_heat_exchanger_inlet_temperature"]["source_names"] == [
        "p1__T2b(20.1)'2_20"
    ]




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
    channels = load_channels(ROOT / "configs/config.yaml")

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
    cycles = yaml.safe_load(
        (ROOT / "configs/config.yaml").read_text(encoding="utf-8")
    )["cycles"]
    cycles.update(minimum_defrost_seconds=120, maximum_defrost_seconds=60)

    with pytest.raises(ValueError, match="minimum_defrost_seconds"):
        _load(_write_config(tmp_path, cycles=cycles))


def test_config_rejects_nonpositive_input_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected_sensor_interval_seconds"):
        _load(_write_config(tmp_path, expected_sensor_interval_seconds=0))


def test_config_accepts_zero_maximum_state_gap(tmp_path: Path) -> None:
    cycles = yaml.safe_load(
        (ROOT / "configs/config.yaml").read_text(encoding="utf-8")
    )["cycles"]
    cycles["maximum_state_gap_seconds"] = 0

    config = _load(_write_config(tmp_path, cycles=cycles))

    assert config.cycles.maximum_state_gap_seconds == 0
