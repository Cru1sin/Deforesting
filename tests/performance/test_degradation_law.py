from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from frost_analysis.degradation_law import (
    fit_hinge,
    leave_group_out_reference,
    monotonicity_metrics,
    relative_degradation,
    select_valid_catalog_positions,
)


def test_select_valid_catalog_positions_uses_zero_based_inclusive_scope() -> None:
    cycles = pd.DataFrame(
        {
            "cycle_name": [f"cycle_{index}" for index in range(52)],
            "status": ["valid" if index % 2 == 0 else "invalid" for index in range(52)],
        }
    )

    selected = select_valid_catalog_positions(cycles, last_position=48)

    assert selected["cycle_name"].iloc[-1] == "cycle_48"
    assert len(selected) == 25
    assert selected["status"].eq("valid").all()


def test_hinge_fit_recovers_simple_threshold_law() -> None:
    state = np.linspace(-1.0, 5.0, 241)
    loss = 0.08 * np.maximum(state - 1.5, 0.0)

    fitted = fit_hinge(state, loss)

    assert abs(fitted.threshold - 1.5) < 0.08
    assert abs(fitted.slope - 0.08) < 0.005
    assert fitted.rmse < 0.002


def test_leave_group_out_reference_never_trains_on_held_out_group() -> None:
    frame = pd.DataFrame(
        {
            "date": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
            "cycle": ["a1"] * 4 + ["b1"] * 4 + ["c1"] * 4,
            "early": [True, True, False, False] * 3,
            "context": [0.0, 1.0, 2.0, 3.0] * 3,
            "target": [1.0, 3.0, 5.0, 7.0, 11.0, 13.0, 15.0, 17.0, 21.0, 23.0, 25.0, 27.0],
        }
    )

    result = leave_group_out_reference(
        frame,
        target="target",
        features=["context"],
        group="date",
        early="early",
        cycle="cycle",
    )

    # The early-cycle calibration removes group-specific offsets, while the
    # context slope is learned exclusively from the other groups.
    assert np.allclose(result.to_numpy(), frame["target"].to_numpy(), atol=1e-5)


def test_relative_degradation_is_zero_when_observed_equals_healthy() -> None:
    result = relative_degradation([8.0, 6.0], [8.0, 8.0])

    np.testing.assert_allclose(result, [0.0, 0.25])


def test_monotonicity_metrics_accept_increasing_degradation_and_detect_recovery() -> None:
    increasing = monotonicity_metrics([0.0, 0.1, 0.2, 0.3])
    recovering = monotonicity_metrics([0.0, 0.2, 0.1, 0.3])

    assert increasing["violation_fraction"] == 0.0
    assert increasing["violating_steps"] == 0
    assert recovering["violation_fraction"] > 0.0
    assert recovering["violating_steps"] == 1


def test_condition_normalized_analysis_writes_cycle_level_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "cycles"
    output_dir = tmp_path / "report"
    input_dir.mkdir()
    for cycle_index, date in enumerate(("2026-01-01", "2026-01-02"), start=1):
        count = 80
        elapsed = np.arange(count) * 10.0
        context = np.linspace(0.0, 1.0, count)
        healthy = 8.0 + 0.2 * context + cycle_index
        degradation = np.maximum(elapsed - 600.0, 0.0) / 6000.0
        frame = pd.DataFrame(
            {
                "cycle_name": f"cycle_{cycle_index}",
                "experiment_date": date,
                "timestamp": pd.date_range(date, periods=count, freq="10s"),
                "cycle_stage": "frost_development",
                "cycle_elapsed_seconds": elapsed,
                "heating_capacity": healthy * (1.0 - degradation),
                "ambient_temperature": context,
                "water_in_temperature": 40.0 + context,
                "water_flow": 1.5 + 0.1 * context,
                "compressor_frequency": 70.0 + context,
                "fan_speed": 400.0 + context,
                "exv_opening": 180.0 + context,
                "operating_mode": "heating" if cycle_index == 1 else 1.0,
            }
        )
        frame.to_parquet(input_dir / f"cycle_{cycle_index}.parquet", index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/performance/analyze_condition_normalized_degradation.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert completed.returncode == 0, completed.stderr
    assert not (output_dir / "源数据" / "normalized_degradation.parquet").exists()
    assert (output_dir / "源数据" / "method_metrics.csv").is_file()
    assert (output_dir / "源数据" / "monotonic_prior_ablation.csv").is_file()
    assert (output_dir / "源数据" / "reference_ridge_ablation.csv").is_file()
    assert (output_dir / "源数据" / "reference_support_audit.csv").is_file()
    metrics = pd.read_csv(output_dir / "源数据" / "method_metrics.csv")
    assert "future_loss_increment_error_ratio" in metrics
    assert (output_dir / "图表" / "循环图" / "cycle_1.png").is_file()
    assert (output_dir / "报告.md").is_file()
