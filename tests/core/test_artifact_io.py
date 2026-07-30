from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from frost_analysis.core.artifacts import write_dataframe
from frost_analysis.core.validation import validate_outputs


def test_parquet_failure_falls_back_atomically_to_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise ImportError("no parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", unavailable)
    result = write_dataframe(pd.DataFrame({"x": [1, 2]}), tmp_path / "frame.parquet")
    assert result.storage_format == "csv"
    assert result.actual_path == tmp_path / "frame.csv"
    assert result.actual_path.read_text(encoding="utf-8") == "x\n1\n2\n"
    assert not (tmp_path / "frame.parquet").exists()
    assert not list(tmp_path.glob(".*.tmp"))
    assert (
        validate_outputs(tmp_path, ["frame.parquet|frame.csv"], pd.DataFrame({"cycle_phase": []}))
        == []
    )
