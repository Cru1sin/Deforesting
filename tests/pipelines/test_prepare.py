from __future__ import annotations

import pandas as pd

from frost_analysis.pipelines.prepare import _drop_preparation_artifacts, _standardize_schema


def test_prepared_schema_drops_internal_cleaning_artifacts() -> None:
    frame = pd.DataFrame(
        {
            "sensor_time": pd.to_datetime(["2026-07-15 00:00:01"]),
            "signal": [1.0],
            "signal__raw": ["1"],
            "signal__missing": [False],
            "signal__interpolated": [False],
            "heating_mode": [True],
        }
    )
    metadata = pd.DataFrame({"canonical_name": ["signal"]})
    result = _standardize_schema(frame, metadata)
    result = _drop_preparation_artifacts(result)
    assert list(result.columns) == ["timestamp", "signal", "heating_mode"]
    assert "baseline" not in " ".join(result.columns)
