from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from frost_analysis.config import load_app_config
from frost_analysis.pipelines.prepare import (
    PrepareResult,
    build_prepared_sensor_table,
    publish_prepare_result,
    select_prepared_output_columns,
    validate_image_requirement,
    validate_prepare_result,
)

ROOT = Path(__file__).resolve().parents[2]


def test_prepared_schema_drops_internal_cleaning_artifacts() -> None:
    frame = pd.DataFrame(
        {
            "sensor_time": pd.to_datetime(["2026-07-15 00:00:01"]),
            "signal": [1.0],
            "signal__raw": ["1"],
            "signal__missing": [False],
            "signal__interpolated": [False],
            "operating_mode": [3],
            "is_heating": [True],
        }
    )
    metadata = pd.DataFrame({"canonical_name": ["signal"]})
    result = build_prepared_sensor_table(frame, metadata)
    result = select_prepared_output_columns(result, registered_columns=["signal"])
    assert list(result.columns) == ["timestamp", "signal", "operating_mode", "is_heating"]
    assert "baseline" not in " ".join(result.columns)


def test_required_images_fail_when_no_image_records_exist() -> None:
    """RGB-required experiments must fail instead of publishing sensor-only data."""
    with pytest.raises(RuntimeError, match="No RGB images were found"):
        validate_image_requirement(pd.DataFrame(), required=True)


def test_prepared_output_selection_rejects_interpolation_artifacts() -> None:
    """Prepare must prove that no numerical reconstruction crossed its boundary."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-15"]),
            "signal": [1.0],
            "signal__interpolated": [False],
        }
    )

    with pytest.raises(RuntimeError, match="interpolated columns"):
        select_prepared_output_columns(frame, registered_columns=["signal"])


def test_publish_writes_iso_state_and_validated_outputs(tmp_path: Path) -> None:
    """The state file should explain the artifact paths without Unix timestamps."""
    loaded = load_app_config(ROOT / "configs" / "0715.yaml")
    paths = replace(
        loaded.paths,
        output_dir=tmp_path,
        prepared_data=tmp_path / "prepared_data.parquet",
        cycle_summary=tmp_path / "cycle_summary.csv",
        state_dir=tmp_path / ".pipeline",
    )
    config = replace(loaded, paths=paths)
    prepared = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-15"]),
            "cycle_id": ["cycle_001"],
            "cycle_stage": ["frost_development"],
            "cycle_status": ["valid"],
            "cycle_elapsed_seconds": [0.0],
            "cycle_progress": [0.0],
        }
    )
    summary = pd.DataFrame(
        {
            "cycle_id": ["cycle_001"],
            "cycle_status": ["valid"],
            "cycle_status_reason": [""],
            "sensor_coverage_fraction": [1.0],
            "rgb_coverage_fraction": [0.0],
            "multimodal_coverage_fraction": [0.0],
        }
    )
    result = PrepareResult(
        prepared_data=prepared,
        cycle_summary=summary,
        warnings=(),
        metrics={"rgb_image_count": 0},
    )

    validate_prepare_result(result, config)
    publish_prepare_result(result, config)

    state = json.loads((tmp_path / ".pipeline" / "prepare.json").read_text(encoding="utf-8"))
    assert datetime.fromisoformat(state["created_at"]).tzinfo is not None
    assert state["prepared_data_path"].endswith("prepared_data.parquet")
    assert state["config_fingerprint"]
    assert state["registry_fingerprint"]
