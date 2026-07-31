from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from frost_analysis.config import Config
from frost_analysis.io import (
    discover_inputs,
    ensure_output_outside_input,
    remove_manifest_for_overwrite,
    write_analysis_outputs,
    write_prepare_outputs,
    write_process_outputs,
    write_run_outputs,
)


def _config(root: Path, input_dir: Path) -> Config:
    return Config(
        project_root=root,
        experiment_id="exp_test",
        experiment_date="2026-07-15",
        input_dir=input_dir,
        channels_path=root / "channels.yaml",
        camera_mapping_path=root / "camera.yaml",
        sensor_globs=("*.xls",),
        image_extensions=(".jpg", ".png"),
        timestamp_column="时间",
        expected_sensor_interval_seconds=1,
        image_match_tolerance_seconds=2,
        cycles={"defrost_channel": "defrost_active"},
        process={"resample_interval_seconds": 10},
        analysis={"future_horizon_minutes": 10},
    )


def test_discover_inputs_reads_root_sensors_and_one_level_images(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "0715"
    camera = raw / "192.168.1.1_1"
    camera.mkdir(parents=True)
    (raw / "参数1.xls").write_text("sensor", encoding="utf-8")
    mapping = tmp_path / "camera.yaml"
    mapping.write_text("camera_roles: {}\n", encoding="utf-8")
    (camera / "20260715080000000.jpg").write_bytes(b"image")
    (camera / "nested").mkdir()
    (camera / "nested" / "ignored.jpg").write_bytes(b"image")

    result = discover_inputs(_config(tmp_path, raw))

    assert result.sensor_files == (raw / "参数1.xls",)
    assert result.image_files == (camera / "20260715080000000.jpg",)
    assert result.camera_mapping_path == mapping


def test_output_inside_raw_input_is_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "0715"
    raw.mkdir(parents=True)

    with pytest.raises(ValueError, match="raw input directory"):
        write_prepare_outputs(
            pd.DataFrame(),
            pd.DataFrame(),
            {},
            raw / "outputs",
            raw,
        )


def test_standalone_stage_output_guard_rejects_raw_input(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "0715"
    raw.mkdir(parents=True)

    with pytest.raises(ValueError, match="raw input directory"):
        ensure_output_outside_input(raw / "stage", raw)


def test_prepare_outputs_are_separate_from_formal_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "0715"
    raw.mkdir(parents=True)
    prepared_dir = tmp_path / "outputs" / "prepared" / "exp_test" / "prepare-1"
    prepared = pd.DataFrame(
        {"experiment_id": ["exp_test"], "timestamp": [pd.Timestamp("2026-07-15")]}
    )
    summary = pd.DataFrame({"experiment_id": ["exp_test"], "cycle_id": ["cycle_001"]})

    write_prepare_outputs(prepared, summary, {"prepared_row_count": 1}, prepared_dir, raw)

    assert (prepared_dir / "prepared_data.parquet").is_file()
    assert (prepared_dir / "cycle_summary.csv").is_file()
    assert (prepared_dir / "prepare_summary.json").is_file()
    assert not (prepared_dir / "manifest.json").exists()


def test_formal_manifest_is_written_after_four_outputs(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "0715"
    raw.mkdir(parents=True)
    config = _config(tmp_path, raw)
    output_dir = tmp_path / "outputs" / "runs" / "run-1"
    frame = pd.DataFrame({"experiment_id": ["exp_test"], "timestamp": [pd.Timestamp("2026-07-15")]})
    summary = pd.DataFrame({"experiment_id": ["exp_test"], "cycle_id": ["cycle_001"]})
    evidence = pd.DataFrame({"experiment_id": ["exp_test"], "channel": ["temperature"]})

    write_run_outputs(
        frame,
        frame,
        summary,
        evidence,
        {"sensor_file_count": 1},
        config,
        tmp_path / "configs" / "0715.yaml",
        output_dir,
        raw,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"] == {
        "prepared_data": "prepared_data.parquet",
        "processed_data": "processed_data.parquet",
        "cycle_summary": "cycle_summary.csv",
        "candidate_channel_evidence": "candidate_channel_evidence.csv",
    }


def test_standalone_writers_protect_known_files_and_keep_unknown_files(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "0715"
    raw.mkdir(parents=True)
    output = tmp_path / "outputs" / "stage"
    output.mkdir(parents=True)
    (output / "notes.txt").write_text("keep", encoding="utf-8")
    processed = pd.DataFrame({"experiment_id": ["exp_test"]})
    summary = pd.DataFrame({"experiment_id": ["exp_test"], "cycle_id": ["cycle_001"]})
    evidence = pd.DataFrame({"experiment_id": ["exp_test"], "channel": ["signal"]})

    write_process_outputs(processed, summary, output, raw)
    write_analysis_outputs(evidence, output, raw)

    with pytest.raises(FileExistsError):
        write_analysis_outputs(evidence, output, raw)
    write_analysis_outputs(evidence, output, raw, overwrite=True)
    assert (output / "notes.txt").read_text(encoding="utf-8") == "keep"
    assert not (output / "manifest.json").exists()


def test_overwrite_removes_old_manifest_before_pipeline_work(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "0715"
    raw.mkdir(parents=True)
    output = tmp_path / "outputs" / "run"
    output.mkdir(parents=True)
    (output / "manifest.json").write_text("old", encoding="utf-8")

    remove_manifest_for_overwrite(output, raw, overwrite=True)

    assert not (output / "manifest.json").exists()
