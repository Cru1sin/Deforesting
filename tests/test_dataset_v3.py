from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from frost_analysis.analysis import EVIDENCE_COLUMNS
from frost_analysis.dataset_loader import DatasetLoader
from frost_analysis.dataset_v3 import add_formal_run, load_v3_source_run
from frost_analysis.dataset_validation_v3 import validate_v3_dataset


def _write_run(
    root: Path,
    token: str,
    date: str,
    *,
    signal_name: str = "signal",
    include_image: bool = False,
) -> Path:
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    input_dir = root / "data" / token
    input_dir.mkdir(parents=True, exist_ok=True)
    image_relative = "camera01/frame.jpg"
    if include_image:
        (input_dir / image_relative).parent.mkdir(parents=True, exist_ok=True)
        (input_dir / image_relative).write_bytes(b"image")
    run_dir = root / "outputs" / "runs" / token
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range(f"{date} 10:00:00", periods=3, freq="10s")
    prepared = pd.DataFrame(
        {
            "experiment_id": [f"exp_{token}"] * 3,
            "experiment_date": [date] * 3,
            "timestamp": timestamps,
            "cycle_id": ["cycle_1"] * 3,
            "cycle_stage": ["frost_development"] * 3,
            "cycle_status": ["valid"] * 3,
            "cycle_status_reason": [pd.NA] * 3,
            "cycle_elapsed_seconds": [0.0, 10.0, 20.0],
            "cycle_progress": [0.0, 0.5, 1.0],
            signal_name: [1.0, 2.0, 3.0],
        }
    )
    if include_image:
        prepared["image_source_path"] = [image_relative, pd.NA, pd.NA]
        prepared["image_source_time"] = [timestamps[0], pd.NaT, pd.NaT]
        prepared["image_source_offset_seconds"] = [0.0, float("nan"), float("nan")]
    processed = prepared.copy()
    processed[f"{signal_name}__imputed"] = False
    summary = pd.DataFrame(
        {
            "experiment_id": [f"exp_{token}"],
            "experiment_date": [date],
            "cycle_id": ["cycle_1"],
            "cycle_status": ["valid"],
            "cycle_status_reason": [pd.NA],
            "baseline_status": ["available"],
            "baseline_failure_reason": [pd.NA],
            "baseline_start": [timestamps[0]],
            "baseline_end": [timestamps[1]],
        }
    )
    prepared.to_parquet(run_dir / "prepared_data.parquet", index=False)
    processed.to_parquet(run_dir / "processed_data.parquet", index=False)
    summary.to_csv(run_dir / "cycle_summary.csv", index=False)
    (run_dir / "candidate_channel_evidence.csv").write_text(
        ",".join(EVIDENCE_COLUMNS) + "\n", encoding="utf-8"
    )
    manifest = {
        "experiment_id": f"exp_{token}",
        "experiment_date": date,
        "config_provenance": {"resolved_config_sha256": f"config-{token}"},
        "resolved_config": {"input_dir": f"data/{token}"},
        "input_inventory_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "outputs": {
            "prepared_data": "prepared_data.parquet",
            "processed_data": "processed_data.parquet",
            "cycle_summary": "cycle_summary.csv",
            "candidate_channel_evidence": "candidate_channel_evidence.csv",
        },
        "output_row_counts": {
            "prepared_data": len(prepared),
            "processed_data": len(processed),
            "cycle_summary": len(summary),
            "candidate_channel_evidence": 0,
        },
        "output_sha256": {
            key: hashlib.sha256((run_dir / filename).read_bytes()).hexdigest()
            for key, filename in {
                "prepared_data": "prepared_data.parquet",
                "processed_data": "processed_data.parquet",
                "cycle_summary": "cycle_summary.csv",
                "candidate_channel_evidence": "candidate_channel_evidence.csv",
            }.items()
        },
        "git_commit": "fixture",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_v3_add_removes_image_contract_from_cycle_and_is_self_contained(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14")
    dataset = tmp_path / "dataset"

    add_formal_run(run, dataset)

    validate_v3_dataset(dataset)
    loader = DatasetLoader(dataset)
    frame = loader.load_cycle("frost_cycle_000001")
    assert not any(str(column).startswith("image_") for column in frame.columns)
    assert list(frame.columns[-5:]) == [
        "dataset_id",
        "dataset_schema_version",
        "dataset_cycle_index",
        "cycle_name",
        "cycle_uid",
    ]


def test_v3_build_removes_published_dataset_when_final_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14")
    dataset = tmp_path / "dataset"

    import frost_analysis.dataset_validation_v3 as validation_v3

    original_validate = validation_v3.validate_v3_dataset

    def fail_published_dataset(path: Path, **kwargs: object) -> None:
        if path.resolve() == dataset.resolve():
            raise RuntimeError("injected final validation failure")
        original_validate(path, **kwargs)

    monkeypatch.setattr(validation_v3, "validate_v3_dataset", fail_published_dataset)
    with pytest.raises(RuntimeError, match="final validation"):
        add_formal_run(run, dataset)

    assert not dataset.exists()


def test_v3_append_requires_strictly_later_date_and_same_source_is_noop(tmp_path: Path) -> None:
    first = _write_run(tmp_path, "0714", "2026-07-14")
    second = _write_run(tmp_path, "0715", "2026-07-15")
    historical = _write_run(tmp_path, "0713", "2026-07-13")
    dataset = tmp_path / "dataset"
    add_formal_run(first, dataset)
    add_formal_run(second, dataset)
    before = (dataset / "dataset_manifest.json").read_bytes()
    add_formal_run(second, dataset)
    assert (dataset / "dataset_manifest.json").read_bytes() == before

    with pytest.raises(ValueError, match="historical or same-date"):
        add_formal_run(historical, dataset)


def test_v3_source_run_loader_does_not_require_raw_directory_to_exist(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14")
    source = load_v3_source_run(run)
    source.input_dir.rmdir()

    assert source.input_dir.name == "0714"


def test_v3_source_run_requires_formal_output_hashes(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14")
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("output_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="output hashes"):
        load_v3_source_run(run)


def test_v3_rejects_summary_identity_different_from_formal_run(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14")
    summary_path = run / "cycle_summary.csv"
    summary = pd.read_csv(summary_path)
    summary["experiment_date"] = "2026-07-31"
    summary.to_csv(summary_path, index=False)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"]["cycle_summary"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Summary identity"):
        add_formal_run(run, tmp_path / "dataset")


def test_v3_dataset_is_self_contained_after_formal_run_is_removed(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14", include_image=True)
    dataset = tmp_path / "dataset"
    add_formal_run(run, dataset)

    import shutil

    shutil.rmtree(run)
    loader = DatasetLoader(dataset)
    assert not loader.load_cycle("frost_cycle_000001").empty
    validate_v3_dataset(dataset)


def test_v3_image_stem_is_stable_and_role_folder_is_loader_authority(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14", include_image=True)
    dataset = tmp_path / "dataset"

    add_formal_run(run, dataset)
    image = next((dataset / "images" / "frost_cycle_000001").rglob("*.jpg"))
    metadata = pd.read_parquet(dataset / "image_metadata.parquet")
    assert image.stem == str(metadata.loc[0, "image_id"])
    image.parent.rename(image.parent.parent / "top_center")

    loader = DatasetLoader(dataset)
    images = loader.load_cycle_images("frost_cycle_000001")
    assert images.loc[0, "camera_role"] == "top_center"
    validate_v3_dataset(dataset)


def test_v3_rejects_incomplete_source_image_triple(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14", include_image=True)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("prepared_data.parquet", "processed_data.parquet"):
        path = run / name
        frame = pd.read_parquet(path).drop(columns=["image_source_path"])
        frame.to_parquet(path, index=False)
        manifest["output_sha256"][name.removesuffix(".parquet")] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="image triple"):
        add_formal_run(run, tmp_path / "dataset")


def test_v3_validator_rejects_image_cycle_identity_mismatch(tmp_path: Path) -> None:
    from frost_analysis.dataset_v3 import make_image_id

    run = _write_run(tmp_path, "0714", "2026-07-14", include_image=True)
    dataset = tmp_path / "dataset"
    add_formal_run(run, dataset)
    metadata_path = dataset / "image_metadata.parquet"
    metadata = pd.read_parquet(metadata_path)
    old_image = next((dataset / "images" / "frost_cycle_000001").rglob("*.jpg"))
    metadata.loc[0, "cycle_uid"] = "other_experiment::cycle_1"
    new_id = make_image_id(
        str(metadata.loc[0, "cycle_uid"]),
        str(metadata.loc[0, "source_camera_id"]),
        str(metadata.loc[0, "source_relative_path"]),
    )
    metadata.loc[0, "image_id"] = new_id
    new_image = old_image.with_name(f"{new_id}{old_image.suffix}")
    old_image.rename(new_image)
    metadata.to_parquet(metadata_path, index=False)
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["image_metadata"]["sha256"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="cycle identity"):
        validate_v3_dataset(dataset)


def test_v3_append_new_channel_rewrites_historical_parquet_and_csv(tmp_path: Path) -> None:
    first = _write_run(tmp_path, "0714", "2026-07-14", signal_name="signal")
    second = _write_run(tmp_path, "0715", "2026-07-15", signal_name="humidity")
    dataset = tmp_path / "dataset"

    add_formal_run(first, dataset)
    add_formal_run(second, dataset)

    old_frame = pd.read_parquet(dataset / "cycles" / "frost_cycle_000001.parquet")
    new_frame = pd.read_parquet(dataset / "cycles" / "frost_cycle_000002.parquet")
    assert "humidity" in old_frame
    assert old_frame["humidity"].isna().all()
    assert "signal" in new_frame
    assert new_frame["signal"].isna().all()
    assert old_frame.columns.tolist() == new_frame.columns.tolist()
    assert (
        pd.read_csv(dataset / "cycles" / "frost_cycle_000001.csv").columns.tolist()
        == old_frame.columns.tolist()
    )
    validate_v3_dataset(dataset)


def test_v3_registry_coverage_channel_rewrites_historical_rgb_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _write_run(tmp_path, "0714", "2026-07-14", signal_name="signal")
    second = _write_run(tmp_path, "0715", "2026-07-15", signal_name="humidity")
    dataset = tmp_path / "dataset"

    add_formal_run(first, dataset)
    coverage_path = dataset / "cycles" / "frost_cycle_000001_rgb_coverage.png"
    before = coverage_path.read_bytes()

    from frost_analysis import dataset_v3
    from frost_analysis.dataset_registry import registry_from_frame

    def candidate_registry(source: object, frame: pd.DataFrame) -> dict[str, object]:
        del source
        return registry_from_frame(
            frame,
            {"humidity": {"coverage_required": True}},
            resample_interval_seconds=10,
        )

    monkeypatch.setattr(dataset_v3, "_source_registry", candidate_registry)
    add_formal_run(second, dataset)

    assert coverage_path.read_bytes() != before
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cycles"][0]["asset_sha256"]["rgb_coverage"] == hashlib.sha256(
        coverage_path.read_bytes()
    ).hexdigest()
    validate_v3_dataset(dataset)


def test_v3_registry_only_coverage_change_rewrites_historical_rgb_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _write_run(tmp_path, "0714", "2026-07-14", signal_name="signal")
    second = _write_run(tmp_path, "0715", "2026-07-15", signal_name="signal")
    dataset = tmp_path / "dataset"

    add_formal_run(first, dataset)
    coverage_path = dataset / "cycles" / "frost_cycle_000001_rgb_coverage.png"
    before = coverage_path.read_bytes()

    from frost_analysis import dataset_v3
    from frost_analysis.dataset_registry import registry_from_frame

    def candidate_registry(source: object, frame: pd.DataFrame) -> dict[str, object]:
        del source
        return registry_from_frame(
            frame,
            {"humidity": {"coverage_required": True}},
            resample_interval_seconds=10,
        )

    monkeypatch.setattr(dataset_v3, "_source_registry", candidate_registry)
    add_formal_run(second, dataset)

    assert coverage_path.read_bytes() != before
    validate_v3_dataset(dataset)


def test_v3_duplicate_source_remains_noop_after_registry_expansion(tmp_path: Path) -> None:
    first = _write_run(tmp_path, "0714", "2026-07-14", signal_name="signal")
    second = _write_run(tmp_path, "0715", "2026-07-15", signal_name="humidity")
    third = _write_run(tmp_path, "0716", "2026-07-16", signal_name="pressure")
    dataset = tmp_path / "dataset"

    add_formal_run(first, dataset)
    add_formal_run(second, dataset)
    add_formal_run(third, dataset)
    before = (dataset / "dataset_manifest.json").read_bytes()

    add_formal_run(second, dataset)

    assert (dataset / "dataset_manifest.json").read_bytes() == before


def test_v3_role_summary_uses_missing_cycle_count_for_absent_role(
    tmp_path: Path,
) -> None:
    first = _write_run(tmp_path, "0714", "2026-07-14", include_image=True)
    second = _write_run(tmp_path, "0715", "2026-07-15", include_image=False)
    dataset = tmp_path / "dataset"

    add_formal_run(first, dataset)
    add_formal_run(second, dataset)

    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    role_summary = manifest["image_summary"]["by_camera_role"]["unassigned_camera01"]
    assert role_summary["cycle_count"] == 1
    assert role_summary["missing_role_cycle_count"] == 1


def test_v3_role_summary_uses_each_cycle_as_the_denominator() -> None:
    from frost_analysis.dataset_coverage_v3 import summarize_cycle_roles

    timestamps = pd.date_range("2026-07-14 10:00:00", periods=2, freq="10s")
    frame = pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "cycle_b"],
            "timestamp": timestamps,
        }
    )
    images = pd.DataFrame(
        {
            "cycle_name": ["cycle_a"],
            "camera_role": ["top"],
            "matched_timestamp": [timestamps[0]],
        }
    )

    summary = summarize_cycle_roles(frame, images)

    assert summary["top"]["cycle_count"] == 1
    assert summary["top"]["missing_role_cycle_count"] == 1


def test_v3_validator_binds_cycle_columns_to_registry(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14")
    dataset = tmp_path / "dataset"
    add_formal_run(run, dataset)
    cycle_path = dataset / "cycles" / "frost_cycle_000001.parquet"
    frame = pd.read_parquet(cycle_path).drop(columns=["signal"])
    frame.to_parquet(cycle_path, index=False)
    csv_path = dataset / "cycles" / "frost_cycle_000001.csv"
    frame.to_csv(csv_path, index=False)
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cycles"][0]["asset_sha256"]["parquet"] = hashlib.sha256(
        cycle_path.read_bytes()
    ).hexdigest()
    manifest["cycles"][0]["asset_sha256"]["csv"] = hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="registry schema"):
        validate_v3_dataset(dataset)


def test_v3_validator_requires_all_cycle_asset_hashes(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14")
    dataset = tmp_path / "dataset"
    add_formal_run(run, dataset)
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cycles"][0].pop("asset_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="asset hashes"):
        validate_v3_dataset(dataset)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cycle_uid", "wrong::cycle_1", "manifest cycle identity"),
        ("row_count", 999, "manifest cycle row count"),
        ("image_count", 999, "manifest cycle image count"),
        ("data_path", "cycles/wrong.parquet", "manifest cycle asset path"),
    ],
)
def test_v3_validator_closes_manifest_cycle_record_to_cycle_index(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14")
    dataset = tmp_path / "dataset"
    add_formal_run(run, dataset)
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cycles"][0][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_v3_dataset(dataset)


def test_v3_validator_rejects_nested_cycle_orphan_file(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14")
    dataset = tmp_path / "dataset"
    add_formal_run(run, dataset)
    orphan = dataset / "cycles" / "orphan" / "nested.parquet"
    orphan.parent.mkdir()
    orphan.write_bytes(b"orphan")

    with pytest.raises(ValueError, match="cycle orphan files"):
        validate_v3_dataset(dataset)


def test_v3_validator_checks_manifest_index_row_count(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14")
    dataset = tmp_path / "dataset"
    add_formal_run(run, dataset)
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cycle_index"]["row_count"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="cycle_index row count"):
        validate_v3_dataset(dataset)


def test_v3_validator_closes_per_cycle_image_count(tmp_path: Path) -> None:
    first = _write_run(tmp_path, "0714", "2026-07-14", include_image=True)
    second = _write_run(tmp_path, "0715", "2026-07-15", include_image=True)
    dataset = tmp_path / "dataset"
    add_formal_run(first, dataset)
    add_formal_run(second, dataset)
    index_path = dataset / "cycle_index.parquet"
    cycle_index = pd.read_parquet(index_path)
    cycle_index.loc[0, "image_count"] = 2
    cycle_index.loc[1, "image_count"] = 0
    cycle_index.to_parquet(index_path, index=False)
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cycle_index"]["sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest["cycles"][0]["image_count"] = 2
    manifest["cycles"][1]["image_count"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="per-cycle image metadata count"):
        validate_v3_dataset(dataset)


def test_v3_validator_closes_cycle_identity_against_cycle_index(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14")
    dataset = tmp_path / "dataset"
    add_formal_run(run, dataset)
    cycle_path = dataset / "cycles" / "frost_cycle_000001.parquet"
    csv_path = dataset / "cycles" / "frost_cycle_000001.csv"
    frame = pd.read_parquet(cycle_path)
    frame["cycle_uid"] = "wrong::cycle_1"
    frame.to_parquet(cycle_path, index=False)
    frame.to_csv(csv_path, index=False)
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cycles"][0]["asset_sha256"]["parquet"] = hashlib.sha256(
        cycle_path.read_bytes()
    ).hexdigest()
    manifest["cycles"][0]["asset_sha256"]["csv"] = hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        validate_v3_dataset(dataset)


def test_v3_validator_rejects_cycle_dtype_change(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14")
    dataset = tmp_path / "dataset"
    add_formal_run(run, dataset)
    cycle_path = dataset / "cycles" / "frost_cycle_000001.parquet"
    frame = pd.read_parquet(cycle_path)
    frame["signal"] = frame["signal"].astype("string")
    frame.to_parquet(cycle_path, index=False)
    csv_path = dataset / "cycles" / "frost_cycle_000001.csv"
    frame.to_csv(csv_path, index=False)
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cycles"][0]["asset_sha256"]["parquet"] = hashlib.sha256(
        cycle_path.read_bytes()
    ).hexdigest()
    manifest["cycles"][0]["asset_sha256"]["csv"] = hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="dtype"):
        validate_v3_dataset(dataset)


def test_v3_refresh_updates_role_summary_without_overwriting_assessment(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14", include_image=True)
    dataset = tmp_path / "dataset"
    add_formal_run(run, dataset)
    cycle_name = "frost_cycle_000001"
    role_dir = dataset / "images" / cycle_name / "unassigned_camera01"
    role_dir.rename(role_dir.parent / "top")

    from frost_analysis.dataset_manifest_v3 import refresh_manifest, review_cycle

    review_cycle(dataset, cycle_name, status="partial", note="manual")
    before = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    refresh_manifest(dataset)
    after = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))

    assert after["cycles"][0]["assessment"] == before["cycles"][0]["assessment"]
    assert "top" in after["image_summary"]["by_camera_role"]
    validate_v3_dataset(dataset)


def test_v3_validator_requires_one_valid_assessment_per_cycle(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "0714", "2026-07-14")
    dataset = tmp_path / "dataset"
    add_formal_run(run, dataset)
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cycles"][0]["assessment"]["status"] = "unknown"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="assessment"):
        validate_v3_dataset(dataset)


def test_v3_coverage_uses_unique_matched_timestamps_and_sensor_quality_mask() -> None:
    from frost_analysis.dataset_coverage_v3 import coverage_ratio, sensor_overall_mask

    timestamps = pd.date_range("2026-07-14 10:00:00", periods=3, freq="10s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": [1.0, 2.0, 3.0],
            "temperature__imputed": [False, True, False],
            "humidity": [40.0, 41.0, None],
            "humidity__imputed": [False, False, False],
        }
    )
    metadata = pd.DataFrame(
        {
            "matched_timestamp": [timestamps[0], timestamps[0], timestamps[2]],
        }
    )

    assert coverage_ratio(frame, metadata) == pytest.approx(2 / 3)
    mask = sensor_overall_mask(
        frame,
        {
            "channels": {
                "temperature": {"coverage_required": True},
                "humidity": {"coverage_required": True},
            }
        },
    )
    assert mask.tolist() == [True, False, False]


def test_v3_sensor_overall_requires_explicit_false_imputation_flag() -> None:
    from frost_analysis.dataset_coverage_v3 import sensor_overall_mask

    frame = pd.DataFrame(
        {
            "signal": [1.0, 2.0],
            "signal__imputed": [False, pd.NA],
        }
    )

    mask = sensor_overall_mask(
        frame,
        {"channels": {"signal": {"coverage_required": True}}},
    )

    assert mask.tolist() == [True, False]


def test_v3_rgb_coverage_reuses_publication_time_and_gap_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis import report
    from frost_analysis.dataset_coverage_v3 import render_rgb_coverage

    calls: dict[str, object] = {}
    origin = pd.Timestamp("2026-07-14 10:00:00")

    def capture_origin(frame: pd.DataFrame, cycle: pd.Series) -> pd.Timestamp:
        calls["origin_frame"] = frame
        calls["origin_cycle"] = cycle
        return origin

    def capture_stage(axes: list[object], cycle: pd.Series, value: pd.Timestamp) -> None:
        calls["stage"] = (axes, cycle, value)

    def capture_gap(axes: list[object], gaps: list[object], value: pd.Timestamp) -> None:
        calls["gap"] = (axes, gaps, value)

    monkeypatch.setattr(report, "_cycle_time_origin", capture_origin)
    monkeypatch.setattr(report, "_shade_cycle_stages", capture_stage)
    monkeypatch.setattr(report, "_shade_defrost_state_gaps", capture_gap)

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(origin, periods=2, freq="10s"),
            "temperature": [1.0, 2.0],
            "temperature__imputed": [False, False],
        }
    )
    record = {
        "cycle_name": "frost_cycle_000001",
        "heating_start": origin,
        "stable_heating_start": origin + pd.Timedelta(seconds=10),
        "defrost_start": origin + pd.Timedelta(seconds=20),
        "defrost_end": origin + pd.Timedelta(seconds=30),
    }

    render_rgb_coverage(
        frame,
        pd.DataFrame(columns=["camera_role", "matched_timestamp"]),
        record,
        tmp_path / "coverage.png",
    )

    assert calls["origin_cycle"].to_dict() == record
    assert calls["stage"][2] == origin
    assert calls["gap"][2] == origin


@pytest.mark.parametrize("failure", ["asset", "metadata"])
def test_v3_append_rolls_back_metadata_and_new_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    first = _write_run(tmp_path, "0714", "2026-07-14", include_image=True)
    second = _write_run(tmp_path, "0715", "2026-07-15", include_image=True)
    dataset = tmp_path / "dataset"
    add_formal_run(first, dataset)
    old_manifest = (dataset / "dataset_manifest.json").read_bytes()
    old_cycle_index = (dataset / "cycle_index.parquet").read_bytes()
    old_images = sorted(
        path.relative_to(dataset).as_posix()
        for path in (dataset / "images").rglob("*")
        if path.is_file()
    )

    import frost_analysis.dataset_io as dataset_io

    original_replace = dataset_io.os.replace

    def fail_replace(source: object, target: object) -> None:
        source_path = Path(source)
        target_path = Path(target)
        if failure == "asset" and target_path.name == "frost_cycle_000002.parquet":
            raise RuntimeError("injected v3 asset failure")
        if failure == "metadata" and source_path.name == "image_metadata.parquet":
            raise RuntimeError("injected v3 metadata failure")
        original_replace(source, target)

    monkeypatch.setattr(dataset_io.os, "replace", fail_replace)
    with pytest.raises(RuntimeError):
        add_formal_run(second, dataset)

    assert (dataset / "dataset_manifest.json").read_bytes() == old_manifest
    assert (dataset / "cycle_index.parquet").read_bytes() == old_cycle_index
    assert not (dataset / "cycles" / "frost_cycle_000002.parquet").exists()
    assert sorted(
        path.relative_to(dataset).as_posix()
        for path in (dataset / "images").rglob("*")
        if path.is_file()
    ) == old_images
    validate_v3_dataset(dataset)
