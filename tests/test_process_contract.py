from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from frost_analysis.config import Config
from frost_analysis.process import process


def _config(root: Path) -> Config:
    return Config(
        project_root=root,
        experiment_id="exp_test",
        experiment_date="2026-07-15",
        input_dir=root / "data",
        channels_path=root / "channels.yaml",
        sensor_globs=("*.xls",),
        image_extensions=(".jpg",),
        camera_mapping_file="IPlocation.yaml",
        cycles={},
        process={
            "resample_interval_seconds": 10,
            "continuous_max_gap_seconds": 30,
            "control_max_gap_seconds": 30,
            "baseline": {
                "stage": "recovery",
                "minimum_observed_coverage": 0.5,
                "maximum_imputed_fraction": 0.0,
            },
            "features": {"windows_minutes": [1]},
        },
        analysis={},
    )


def _channels() -> dict[str, dict[str, object]]:
    return {
        "temperature": {
            "kind": "continuous",
            "role": "sensor",
            "resample": "mean",
            "missing": "none",
            "analysis_candidate": True,
            "expected_frost_direction": "decrease",
        },
        "unavailable_candidate": {
            "kind": "continuous",
            "role": "sensor",
            "resample": "mean",
            "missing": "none",
            "analysis_candidate": True,
            "expected_frost_direction": "decrease",
        },
        "compressor_frequency": {
            "kind": "step",
            "role": "control",
            "resample": "last",
            "missing": "none",
            "analysis_candidate": False,
        },
        "heating_capacity": {
            "kind": "continuous",
            "role": "performance",
            "resample": "mean",
            "missing": "interpolate",
            "analysis_candidate": False,
        },
        "power_total": {
            "kind": "continuous",
            "role": "performance",
            "resample": "mean",
            "missing": "interpolate",
            "analysis_candidate": False,
        },
        "defrost_active": {
            "kind": "event",
            "role": "event",
            "resample": "last",
            "missing": "none",
            "analysis_candidate": False,
        },
        "cop": {
            "kind": "derived",
            "role": "performance",
            "resample": "mean",
            "missing": "none",
            "analysis_candidate": False,
            "formula": "cop",
            "dependencies": ["heating_capacity", "power_total"],
        },
    }


def _prepared() -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = pd.date_range("2026-07-15", periods=6, freq="10s")
    prepared = pd.DataFrame(
        {
            "experiment_id": "exp_test",
            "experiment_date": "2026-07-15",
            "timestamp": timestamps,
            "cycle_id": "cycle_001",
            "cycle_stage": [
                "recovery",
                "recovery",
                "frost_development",
                "frost_development",
                "frost_development",
                "defrost",
            ],
            "cycle_status": "valid",
            "cycle_status_reason": "",
            "cycle_elapsed_seconds": [np.nan] * 6,
            "cycle_progress": [0.0] * 6,
            "temperature": [1.0, 1.0, 2.0, 99.0, 4.0, np.nan],
            "temperature__duplicate": [False, False, False, True, False, False],
            "temperature__conflict": [False] * 6,
            "unavailable_candidate": [np.nan] * 6,
            "compressor_frequency": [50.0, np.nan, 60.0, np.nan, 60.0, np.nan],
            "heating_capacity": [10.0, 10.0, 9.0, np.nan, 8.0, np.nan],
            "power_total": [2.0, 2.0, 2.0, np.nan, 2.0, np.nan],
            "defrost_active": [False, False, False, False, False, True],
            "image_path": ["a.jpg", np.nan, np.nan, np.nan, "b.jpg", np.nan],
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["cycle_001"],
            "cycle_status": ["valid"],
            "cycle_status_reason": [""],
            "heating_start": [timestamps[0]],
            "stable_heating_start": [timestamps[2]],
            "defrost_start": [timestamps[5]],
            "defrost_end": [timestamps[5]],
        }
    )
    return prepared, summary


def test_process_order_preserves_missing_meanings_and_recomputes_progress(tmp_path: Path) -> None:
    prepared, summary = _prepared()

    processed, final_summary = process(prepared, summary, _config(tmp_path), _channels())

    at_30 = processed["timestamp"].eq(pd.Timestamp("2026-07-15 00:00:30"))
    row = processed.loc[at_30].iloc[0]
    progress = processed.loc[at_30, "cycle_progress"].iloc[0]
    assert pd.isna(row["temperature"])
    at_10 = processed["timestamp"].eq(pd.Timestamp("2026-07-15 00:00:10"))
    assert pd.isna(processed.loc[at_10, "compressor_frequency"].iloc[0])
    assert progress == 1 / 3
    assert bool(row["cop__imputed"])
    assert row["heating_capacity__baseline_residual"] < 0
    assert processed["unavailable_candidate__baseline_residual"].isna().all()
    assert processed["unavailable_candidate__baseline_status"].eq("no_candidate_window").all()
    assert final_summary.loc[0, "cycle_status"] == "valid"


def test_process_does_not_forward_fill_images_or_cross_stage_values(tmp_path: Path) -> None:
    prepared, summary = _prepared()

    processed, _ = process(prepared, summary, _config(tmp_path), _channels())

    defrost = processed.loc[processed["cycle_stage"].eq("defrost")]
    assert defrost["image_path"].isna().all()
    assert processed.loc[processed["cycle_stage"].eq("defrost"), "temperature"].isna().all()
