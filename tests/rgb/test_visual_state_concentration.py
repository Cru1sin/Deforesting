from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def test_centroid_distances_returns_matched_optimum_and_fixed_time_states() -> None:
    path = Path("scripts/rgb/analyze_visual_state_concentration.py")
    spec = importlib.util.spec_from_file_location("visual_concentration", path)
    assert spec and spec.loader
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    rows = []
    for cycle, experiment, offset in (("a", "e1", 0.0), ("b", "e2", 1.0)):
        for minute, regret in ((0.0, 0.1), (5.0, 0.0), (10.0, 0.1)):
            rows.append(
                {
                    "cycle_name": cycle,
                    "experiment_id": experiment,
                    "time_elapsed_minutes": minute,
                    "relative_regret": regret,
                    "feature_000": offset + minute,
                    "feature_001": offset - minute,
                }
            )

    result = analysis.centroid_distances(
        pd.DataFrame(rows), regret_threshold=0.01, fixed_elapsed_minutes=0.0
    )

    assert len(result) == 4
    assert set(result["state"]) == {"optimum", "fixed_time"}
    assert result.groupby("cycle_name")["image_count"].nunique().eq(1).all()
    assert result["centroid_distance"].notna().all()
