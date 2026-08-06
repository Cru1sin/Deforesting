from __future__ import annotations

import pandas as pd
import pytest

from frost_analysis.dataset_schema import (
    build_processed_frame,
    drop_image_columns,
    merge_registries,
    registry_from_frame,
)


def _registry() -> dict[str, object]:
    return {
        "resample_interval_seconds": 10,
        "channels": {
            "signal": {
                "kind": "continuous",
                "unit": "C",
                "role": "performance",
                "resample": "mean",
                "formula": None,
                "dependencies": None,
                "scale": None,
                "offset": None,
                "analysis_candidate": True,
                "expected_frost_direction": None,
                "coverage_required": True,
            }
        },
        "columns": ["timestamp", "experiment_id", "cycle_id", "signal"],
        "analysis_settings": {},
    }


def test_drop_image_columns_removes_all_image_triples() -> None:
    frame = pd.DataFrame(
        columns=[
            "timestamp",
            "image_front_path",
            "image_front_time",
            "image_front_offset_seconds",
            "signal",
        ]
    )
    assert drop_image_columns(frame).columns.tolist() == ["timestamp", "signal"]


def test_registry_is_explicit_and_merge_adds_new_columns() -> None:
    candidate = registry_from_frame(
        pd.DataFrame(columns=["timestamp", "experiment_id", "cycle_id", "humidity"]),
        {
            "humidity": {
                "kind": "continuous",
                "unit": "%",
                "resample": "mean",
            }
        },
    )
    merged = merge_registries(_registry(), candidate)
    assert list(merged["channels"]) == ["signal", "humidity"]
    assert merged["columns"] == [
        "timestamp",
        "experiment_id",
        "cycle_id",
        "signal",
        "humidity",
    ]


def test_registry_rejects_changed_scientific_definition() -> None:
    changed = _registry()
    changed["channels"] = {"signal": {**changed["channels"]["signal"], "unit": "K"}}
    with pytest.raises(ValueError, match="channel definition changed"):
        merge_registries(_registry(), changed)


def test_processed_frame_has_only_cycle_and_scientific_identity() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-14 10:00:00"]),
            "experiment_id": ["exp"],
            "cycle_id": ["cycle_001"],
            "signal": [1.0],
        }
    )
    result = build_processed_frame(
        frame,
        _registry(),
        cycle_name="frost_cycle_000001",
        cycle_uid="exp::cycle_001",
    )
    assert result.columns.tolist() == [
        "cycle_name",
        "cycle_uid",
        "timestamp",
        "experiment_id",
        "cycle_id",
        "signal",
    ]
    assert not {
        "dataset_id",
        "dataset_schema_version",
        "dataset_cycle_index",
    } & set(result)
