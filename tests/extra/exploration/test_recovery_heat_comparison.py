import importlib.util
import warnings
from pathlib import Path

import pandas as pd


def _module():  # type: ignore[no-untyped-def]
    path = Path("scripts/exploration/plot_recovery_heat_comparison.py")
    spec = importlib.util.spec_from_file_location("recovery_heat_comparison", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_recovery_heat_comparison_writes_one_png(tmp_path: Path) -> None:
    module = _module()
    events = pd.DataFrame(
        {
            "cycle_id": [1, 2, 4, 5],
            "experiment_date": ["2026-07-14", "2026-07-14", "2026-07-15", "2026-07-15"],
            "water_heat_kwh": [0.8, 0.9, 1.0, 1.1],
            "unit_heat_kwh": [1.0, 1.1, 1.2, 1.3],
        }
    )

    output = tmp_path / "recovery_heat_comparison.png"
    module.plot_recovery_heat_comparison(events, output)

    assert output.is_file()


def test_build_recovery_heat_comparison_uses_all_complete_recoveries() -> None:
    module = _module()

    events = module.build_recovery_heat_comparison(
        Path("dataset"),
        Path("output/test/成本函数/ED模型/经验经济窗口/源数据/defrost_ticket_events.csv"),
    )

    assert len(events) == 68
    assert events[["water_heat_kwh", "unit_heat_kwh", "water_cop", "unit_cop"]].notna().all().all()
    assert (
        events["stable_end"] - events["recovery_end"]
    ).eq(pd.Timedelta(minutes=5)).all()


def test_build_heating_stage_heat_uses_two_stage_boundary() -> None:
    module = _module()

    events = module.build_heating_stage_heat_comparison(Path("dataset"))

    assert not events.empty
    assert pd.to_datetime(events["start"]).eq(
        pd.to_datetime(events["heating_start"])
    ).all()
    assert pd.to_datetime(events["end"]).eq(
        pd.to_datetime(events["defrost_preparation_start"])
    ).all()
    assert events["duration_minutes"].gt(0).all()
    assert events[["water_heat_kwh", "unit_heat_kwh"]].notna().all().all()
    assert events["eligible"].eq(
        events[["water_heat_coverage", "unit_heat_coverage"]].min(axis=1).ge(0.95)
    ).all()


def test_plot_heating_stage_heat_writes_png_and_svg(tmp_path: Path) -> None:
    module = _module()
    events = pd.DataFrame(
        {
            "cycle_id": [1, 2, 4, 5],
            "experiment_date": ["2026-07-14", "2026-07-14", "2026-07-15", "2026-07-15"],
            "water_heat_kwh": [0.8, 0.9, 1.0, 1.1],
            "unit_heat_kwh": [1.0, 1.1, 1.2, 1.3],
            "eligible": True,
        }
    )

    output = tmp_path / "heating_stage_heat_comparison.png"
    module.plot_heating_stage_heat_comparison(events, output)

    assert output.is_file()
    assert output.with_suffix(".svg").is_file()


def test_plot_defrost_recovery_cycle_curve_writes_one_png(tmp_path: Path) -> None:
    module = _module()
    timestamps = pd.date_range("2026-01-01", periods=421, freq="s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "stage": ["defrost"] * 60 + ["recovery"] * 60 + ["stable"] * 301,
            "water_flow": 1.0,
            "water_in_temperature": 40.0,
            "water_out_temperature": 45.0,
            "heating_capacity": 6.5,
            "power_total": 2.0,
        }
    )
    event = pd.Series(
        {
            "cycle_name": "frost_cycle_000001",
            "experiment_date": "2026-01-01",
            "defrost_start": timestamps.min(),
            "recovery_start": timestamps[60],
            "recovery_end": timestamps[120],
            "stable_end": timestamps.max(),
            "setpoint_c": 50.0,
            "duration_minutes": 1.0,
            "water_heat_kwh": 0.2,
            "unit_heat_kwh": 0.25,
            "water_cop": 2.3,
            "unit_cop": 2.9,
        }
    )

    output = tmp_path / "cycle.png"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        module.plot_defrost_recovery_cycle_curve(frame, event, output)

    assert output.is_file()


def test_cycle_curves_do_not_smooth_measurements() -> None:
    source = Path("scripts/exploration/plot_recovery_heat_comparison.py").read_text()

    assert ".rolling(" not in source


def test_cycle_curves_reuse_gap_aware_energy_integration_for_cop() -> None:
    source = Path("scripts/exploration/plot_recovery_heat_comparison.py").read_text()

    assert "integrate_energy_curve_kwh" in source
