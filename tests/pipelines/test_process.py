from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from frost_analysis.pipelines.process import _internal_cycle_columns, load_prepared_data


def test_process_rejects_prepared_data_without_stage_contract(tmp_path: Path) -> None:
    path = tmp_path / "prepared.parquet"
    pd.DataFrame({"timestamp": pd.date_range("2026-07-15", periods=1)}).to_parquet(path)
    with pytest.raises(ValueError, match="missing required columns"):
        load_prepared_data(path)


def test_process_maps_invalid_prepare_status_to_legacy_internal_quality() -> None:
    """Processing may use its old internal flags without weakening the public contract."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-15"]),
            "cycle_id": ["cycle_001"],
            "cycle_status": ["invalid"],
            "cycle_stage": ["frost_development"],
            "cycle_progress": [0.5],
            "cycle_elapsed_seconds": [1.0],
            "is_heating": [False],
        }
    )

    result = _internal_cycle_columns(frame)

    assert result.loc[0, "cycle_quality"] == "abnormal"
    assert bool(result.loc[0, "cycle_gap_contaminated"])
