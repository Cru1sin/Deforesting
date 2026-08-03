from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from frost_analysis.dataset import append_dataset, build_dataset
from frost_analysis.dataset_validation import validate_dataset


def _write_run(
    root: Path,
    experiment_id: str,
    experiment_date: str,
    *,
    image_bytes: bytes = b"image",
    include_images: bool = True,
    include_processed: bool = True,
) -> Path:
    if not (root / "pyproject.toml").exists():
        (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    date_token = experiment_date.replace("-", "")
    input_dir = root / "data" / date_token[-4:]
    input_dir.mkdir(parents=True, exist_ok=True)
    image_relative = f"camera01/{date_token}100000000.jpg"
    image_path = input_dir / image_relative
    if include_images:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)

    run_dir = root / "outputs" / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp(f"{experiment_date} 10:00:00")
    prepared = pd.DataFrame(
        {
            "experiment_id": [experiment_id],
            "experiment_date": [experiment_date],
            "timestamp": [timestamp],
            "cycle_id": ["cycle_1"],
            "cycle_stage": ["frost_development"],
            "cycle_status": ["valid"],
            "cycle_status_reason": [pd.NA],
            "cycle_elapsed_seconds": [0.0],
            "cycle_progress": [0.0],
            "image_front_path": [image_relative if include_images else pd.NA],
            "image_front_time": [timestamp if include_images else pd.NaT],
            "image_front_offset_seconds": [0.0 if include_images else float("nan")],
        }
    )
    processed = prepared.copy() if include_processed else prepared.iloc[0:0].copy()
    summary = pd.DataFrame(
        {
            "experiment_id": [experiment_id],
            "experiment_date": [experiment_date],
            "cycle_id": ["cycle_1"],
            "cycle_status": ["valid"],
            "cycle_status_reason": [pd.NA],
            "baseline_status": ["available"],
            "baseline_failure_reason": [pd.NA],
            "baseline_start": [timestamp],
            "baseline_end": [timestamp + pd.Timedelta(minutes=5)],
        }
    )
    prepared.to_parquet(run_dir / "prepared_data.parquet", index=False)
    processed.to_parquet(run_dir / "processed_data.parquet", index=False)
    summary.to_csv(run_dir / "cycle_summary.csv", index=False)
    (run_dir / "candidate_channel_evidence.csv").write_text(
        "experiment_id,channel\n", encoding="utf-8"
    )
    manifest = {
        "experiment_id": experiment_id,
        "experiment_date": experiment_date,
        "config_provenance": {"resolved_config_sha256": f"config-{experiment_id}"},
        "resolved_config": {"input_dir": f"data/{date_token[-4:]}"},
        "outputs": {
            "prepared_data": "prepared_data.parquet",
            "processed_data": "processed_data.parquet",
            "cycle_summary": "cycle_summary.csv",
            "candidate_channel_evidence": "candidate_channel_evidence.csv",
        },
        "git_commit": "fixture-commit",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return run_dir


def test_build_append_and_standalone_validation(tmp_path: Path) -> None:
    first = _write_run(tmp_path, "frost_0715", "2026-07-15")
    second = _write_run(tmp_path, "frost_0716", "2026-07-16")
    dataset_dir = tmp_path / "outputs" / "datasets" / "frost_cycles_v1"

    build_dataset([second, first], dataset_dir)

    index = pd.read_parquet(dataset_dir / "cycle_index.parquet")
    assert index["cycle_name"].dropna().tolist() == [
        "frost_cycle_000001",
        "frost_cycle_000002",
    ]
    assert len(list((dataset_dir / "cycles").glob("*.parquet"))) == 2
    assert len(list((dataset_dir / "images").glob("*.jpg"))) == 2
    validate_dataset(dataset_dir)

    third = _write_run(tmp_path, "frost_0717", "2026-07-17")
    append_dataset(third, dataset_dir)
    index = pd.read_parquet(dataset_dir / "cycle_index.parquet")
    assert index["cycle_name"].dropna().tolist() == [
        "frost_cycle_000001",
        "frost_cycle_000002",
        "frost_cycle_000003",
    ]
    validate_dataset(dataset_dir)

    manifest_before = (dataset_dir / "dataset_manifest.json").read_bytes()
    append_dataset(third, dataset_dir)
    assert (dataset_dir / "dataset_manifest.json").read_bytes() == manifest_before

    shutil.rmtree(first)
    shutil.rmtree(second)
    shutil.rmtree(third)
    shutil.rmtree(tmp_path / "data")
    validate_dataset(dataset_dir)


def test_dataset_round_trip_preserves_processed_content_exactly(tmp_path: Path) -> None:
    first = _write_run(tmp_path, "frost_0715", "2026-07-15")
    second = _write_run(tmp_path, "frost_0716", "2026-07-16")
    dataset_dir = tmp_path / "outputs" / "datasets" / "frost_cycles_v1"

    build_dataset([second, first], dataset_dir)

    source_frames = [
        pd.read_parquet(run / "processed_data.parquet") for run in (first, second)
    ]
    expected = pd.concat(source_frames, ignore_index=True)
    exported_frames = [
        pd.read_parquet(path)
        for path in sorted((dataset_dir / "cycles").glob("*.parquet"))
    ]
    actual = pd.concat(exported_frames, ignore_index=True)
    image_index = pd.read_parquet(dataset_dir / "image_index.parquet")
    dataset_to_source_path = dict(
        zip(
            image_index["image_path"],
            image_index["source_relative_path"],
            strict=True,
        )
    )

    actual = actual.drop(
        columns=[
            "dataset_id",
            "dataset_schema_version",
            "dataset_cycle_index",
            "cycle_name",
            "cycle_uid",
        ]
    )
    for column in actual.columns:
        if column.startswith("image_") and column.endswith("_path"):
            present = actual[column].notna()
            actual.loc[present, column] = actual.loc[present, column].map(
                dataset_to_source_path
            )

    sort_columns = ["experiment_id", "cycle_id", "timestamp"]
    expected = expected.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    actual = actual.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    assert len(actual) == len(expected)
    assert actual.columns.tolist() == expected.columns.tolist()
    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=True,
        check_like=False,
        check_exact=True,
    )


def test_build_publishes_sensor_cycle_without_images(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "frost_sensor_only",
        "2026-07-18",
        include_images=False,
    )
    dataset_dir = tmp_path / "outputs" / "datasets" / "frost_cycles_v1"

    build_dataset([run], dataset_dir)

    cycle_index = pd.read_parquet(dataset_dir / "cycle_index.parquet")
    image_index = pd.read_parquet(dataset_dir / "image_index.parquet")
    assert bool(cycle_index.loc[0, "published"])
    assert int(cycle_index.loc[0, "image_count"]) == 0
    assert image_index.empty
    assert (dataset_dir / "cycles").is_dir()
    assert (dataset_dir / "images").is_dir()
    validate_dataset(dataset_dir)


def test_build_keeps_summary_cycle_when_processed_is_empty(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "frost_no_processed",
        "2026-07-19",
        include_images=False,
        include_processed=False,
    )
    dataset_dir = tmp_path / "outputs" / "datasets" / "frost_cycles_v1"

    build_dataset([run], dataset_dir)

    cycle_index = pd.read_parquet(dataset_dir / "cycle_index.parquet")
    assert len(cycle_index) == 1
    assert not bool(cycle_index.loc[0, "published"])
    assert int(cycle_index.loc[0, "processed_row_count"]) == 0
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    assert int(manifest["published_cycle_count"]) == 0
    assert not list((dataset_dir / "cycles").glob("*.parquet"))
    assert not list((dataset_dir / "images").glob("*"))
    validate_dataset(dataset_dir)


def test_append_rejects_changed_existing_experiment(tmp_path: Path) -> None:
    first = _write_run(tmp_path, "frost_0715", "2026-07-15")
    dataset_dir = tmp_path / "outputs" / "datasets" / "frost_cycles_v1"
    build_dataset([first], dataset_dir)
    changed = _write_run(
        tmp_path,
        "frost_0715",
        "2026-07-15",
        image_bytes=b"changed",
    )

    with pytest.raises(ValueError, match="fingerprint"):
        append_dataset(changed, dataset_dir)


@pytest.mark.parametrize("failure_name", ["cycle_file", "image_index"])
def test_append_rolls_back_after_commit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_name: str
) -> None:
    first = _write_run(tmp_path, "frost_0715", "2026-07-15")
    dataset_dir = tmp_path / "outputs" / "datasets" / "frost_cycles_v1"
    build_dataset([first], dataset_dir)
    old_manifest = (dataset_dir / "dataset_manifest.json").read_bytes()
    old_cycles = sorted(path.name for path in (dataset_dir / "cycles").glob("*.parquet"))
    third = _write_run(tmp_path, "frost_0717", "2026-07-17")

    import frost_analysis.dataset_io as dataset_io

    original_replace = dataset_io.os.replace

    def fail_replace(source: object, target: object) -> None:
        source_path = Path(source)
        target_path = Path(target)
        if failure_name == "cycle_file" and target_path.parent == dataset_dir / "cycles":
            raise RuntimeError("injected cycle move failure")
        if failure_name == "image_index" and source_path.name == "image_index.parquet":
            raise RuntimeError("injected image index failure")
        original_replace(source, target)

    monkeypatch.setattr(dataset_io.os, "replace", fail_replace)
    with pytest.raises(RuntimeError):
        append_dataset(third, dataset_dir)

    assert (dataset_dir / "dataset_manifest.json").read_bytes() == old_manifest
    assert sorted(path.name for path in (dataset_dir / "cycles").glob("*.parquet")) == old_cycles
    validate_dataset(dataset_dir)
