from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _raw_frame(start: pd.Timestamp, periods: int = 601) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq="s")
    elapsed = np.arange(periods) / 60
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "power_total": 2.0,
            "compressor_power": 1.0,
            "heating_capacity": 6.0,
            "water_flow": 1.0,
            "water_in_temperature": 35.0,
            "water_out_temperature": 40.0,
            "coil_temperature": -8.0,
            "evaporating_pressure": 0.4 + 0.01 * elapsed,
            "water_temperature_setpoint": 50.0,
            "ambient_temperature": 2.0,
            "compressor_frequency": 70.0,
        }
    )

    frame["evaporating_pressure_processed"] = frame.evaporating_pressure
    frame["coil_temperature_processed"] = frame.coil_temperature
    frame["heating_capacity_processed"] = frame.heating_capacity
    return frame


def test_half_open_integral_excludes_right_boundary_and_does_not_bridge_gap() -> None:
    from defrost_event_models.training_data import window_audit

    start = pd.Timestamp("2026-01-01")
    frame = _raw_frame(start, 61)
    frame.loc[60, "power_total"] = 100.0
    end = start + pd.Timedelta(seconds=60)
    audit = window_audit(frame, start, end, "power_total")
    assert audit["energy"] == pytest.approx(2 * 60 / 3600)

    gapped = frame.drop(index=range(11, 50))
    audit = window_audit(gapped, start, end, "power_total")
    assert audit["energy"] == pytest.approx(2 * 20 / 3600)
    assert not audit["valid"]


def test_window_maximum_gap_includes_leading_window_gap() -> None:
    from defrost_event_models.training_data import window_audit

    start = pd.Timestamp("2026-01-01")
    frame = _raw_frame(start, 61).iloc[25:]

    result = window_audit(frame, start, start + pd.Timedelta(seconds=60), "power_total")

    assert result["maximum_gap_seconds"] == 25
