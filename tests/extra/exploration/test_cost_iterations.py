from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _module():  # type: ignore[no-untyped-def]
    path = Path("scripts/exploration/analyze_cost_iterations.py")
    spec = importlib.util.spec_from_file_location("analyze_cost_iterations", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_connected_basin_width_uses_segment_containing_minimum() -> None:
    module = _module()
    times = pd.date_range("2026-01-01", periods=7, freq="min")
    curve = pd.DataFrame(
        {
            "candidate_time": times,
            "minutes": range(7),
            "relative_regret": [0.02, 0.0, 0.005, 0.02, 0.005, 0.005, 0.02],
        }
    )

    assert module.connected_basin_width(curve, times[1], 0.01) == 1.0


def test_version_summary_has_five_rows() -> None:
    module = _module()
    cycles = pd.DataFrame(
        [
            {
                "version": version,
                "cycle_name": f"cycle_{cycle}",
                "t_star_minutes": 30 + cycle,
                "minimum_location": "interior",
                "supported": True,
                "hard_label": True,
                "basin_1pct_minutes": 2.0,
                "basin_5pct_minutes": 5.0,
            }
            for version in module.VERSIONS
            for cycle in range(3)
        ]
    )

    assert len(module.build_version_summary(cycles)) == 5
