import numpy as np
import pandas as pd
import pytest

from plots.event_calibration import event_calibration_metrics


def test_event_calibration_keeps_seed_repeats_separate():
    rows = pd.DataFrame(
        {
            "event_id": ["a", "b", "a", "b"],
            "experiment_id": ["e1", "e2", "e1", "e2"],
            "representation": ["model"] * 4,
            "seed": [0, 0, 1, 1],
            "target": [1.0, 2.0, 1.0, 2.0],
            "predicted_target": [1.1, 1.9, 1.2, 1.8],
        }
    )

    metrics = event_calibration_metrics(rows, ["target"])

    assert metrics.events.tolist() == [2, 2]
    assert metrics.seed.tolist() == [0, 1]
    np.testing.assert_allclose(metrics.mse, [0.01, 0.04])
    np.testing.assert_allclose(metrics.r2, [0.96, 0.84])


def test_total_electricity_includes_recovery_to_preparation_start():
    from types import SimpleNamespace

    from plots.event_calibration import measured_preparation_start_electricity

    start = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame(
        {"timestamp": pd.date_range(start, periods=121, freq="s"), "power_total": 3.0}
    )
    loader = SimpleNamespace(load_cycle_measurements=lambda _: frame)
    event = SimpleNamespace(
        cycle_name="cycle",
        event_id="event",
        heating_start=start,
        defrost_preparation_start=start + pd.Timedelta(seconds=90),
    )
    result = measured_preparation_start_electricity(loader, event)
    assert result["total_electricity_measurement_valid"]
    assert result["measured_pre_defrost_electricity_kwh"] == pytest.approx(0.075)
