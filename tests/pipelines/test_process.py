from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from frost_analysis.pipelines.process import load_prepared_data


def test_process_rejects_prepared_data_without_stage_contract(tmp_path: Path) -> None:
    path = tmp_path / "prepared.parquet"
    pd.DataFrame({"timestamp": pd.date_range("2026-07-15", periods=1)}).to_parquet(path)
    with pytest.raises(ValueError, match="missing required columns"):
        load_prepared_data(path)
