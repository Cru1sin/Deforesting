from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from test_dataset_integration import _write_run

from frost_analysis.dataset import add_dataset
from frost_analysis.dataset_loader import DatasetLoader
from frost_analysis.dataset_validation import validate_dataset


def test_add_publishes_all_cycle_assets_and_single_assessment(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "frost_0714", "2026-07-14")
    dataset = tmp_path / "outputs" / "datasets" / "frost_dataset"

    add_dataset(run, dataset)

    index = pd.read_parquet(dataset / "cycle_index.parquet")
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    cycle_name = str(index.loc[0, "cycle_name"])
    cycle_dir = dataset / "cycles"
    assert (cycle_dir / f"{cycle_name}.parquet").is_file()
    assert (cycle_dir / f"{cycle_name}.csv").is_file()
    assert (cycle_dir / f"{cycle_name}.png").is_file()
    assert (cycle_dir / f"{cycle_name}_rgb_coverage.png").is_file()
    assert (dataset / "image_metadata.parquet").is_file()
    assert manifest["cycles"][0]["assessment"]["status"] == "valid"
    assert set(manifest["cycles"][0]["assessment"]) == {
        "status",
        "reasons",
        "note",
        "updated_at",
    }
    validate_dataset(dataset)


def test_unprocessed_cycle_still_has_empty_scientific_file_and_four_assets(
    tmp_path: Path,
) -> None:
    run = _write_run(
        tmp_path,
        "frost_incomplete",
        "2026-07-15",
        include_images=False,
        include_processed=False,
    )
    dataset = tmp_path / "outputs" / "datasets" / "frost_dataset"

    add_dataset(run, dataset)

    index = pd.read_parquet(dataset / "cycle_index.parquet")
    cycle_name = str(index.loc[0, "cycle_name"])
    assert int(index.loc[0, "row_count"]) == 0
    assert pd.read_parquet(dataset / "cycles" / f"{cycle_name}.parquet").empty
    assert (dataset / "cycles" / f"{cycle_name}.csv").is_file()
    assert (dataset / "cycles" / f"{cycle_name}.png").is_file()
    assert (dataset / "cycles" / f"{cycle_name}_rgb_coverage.png").is_file()
    validate_dataset(dataset)


def test_loader_uses_current_image_parent_directory_as_camera_role(
    tmp_path: Path,
) -> None:
    run = _write_run(tmp_path, "frost_0714", "2026-07-14")
    dataset = tmp_path / "outputs" / "datasets" / "frost_dataset"
    add_dataset(run, dataset)
    loader = DatasetLoader(dataset)
    cycle_name = str(loader.cycle_index.loc[0, "cycle_name"])

    image_path = next((dataset / "images" / cycle_name).glob("*/*"))
    image_path.parent.rename(image_path.parent.parent / "top_center")

    renamed = DatasetLoader(dataset).load_cycle_images(cycle_name)

    assert renamed.loc[0, "camera_role"] == "top_center"
    assert renamed.loc[0, "image_id"] == image_path.stem
    frame = DatasetLoader(dataset).load_cycle(cycle_name)
    assert "top_center" in str(frame.loc[0, "image_front_path"])
    from frost_analysis.dataset_manifest import refresh_manifest

    refresh_manifest(dataset)
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert "top_center" in manifest["cycles"][0]["image_summary"]["by_camera_role"]
    validate_dataset(dataset)


def test_refresh_preserves_manually_reviewed_assessment(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "frost_0714", "2026-07-14")
    dataset = tmp_path / "outputs" / "datasets" / "frost_dataset"
    add_dataset(run, dataset)

    from frost_analysis.dataset_manifest import refresh_manifest, review_cycle

    cycle_name = str(pd.read_parquet(dataset / "cycle_index.parquet").loc[0, "cycle_name"])
    review_cycle(dataset, cycle_name, status="partial", note="manual review")
    before = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))

    refresh_manifest(dataset)

    after = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert after["cycles"][0]["assessment"] == before["cycles"][0]["assessment"]


def test_rendered_assets_keep_dataset_manifest_hashes_valid(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "frost_0714", "2026-07-14")
    dataset = tmp_path / "outputs" / "datasets" / "frost_dataset"
    add_dataset(run, dataset)

    cycle_name = str(pd.read_parquet(dataset / "cycle_index.parquet").loc[0, "cycle_name"])
    role_dir = dataset / "images" / cycle_name / "unassigned_01"
    role_dir.rename(role_dir.parent / "top_center")

    from frost_analysis.visualization import (
        generate_cycle_publication,
        generate_rgb_coverage,
    )

    loader = DatasetLoader(dataset)
    generate_cycle_publication(loader, cycle_name)
    generate_rgb_coverage(loader, cycle_name)

    validate_dataset(dataset)


def test_add_appends_cycles_and_same_source_is_a_noop(tmp_path: Path) -> None:
    first = _write_run(tmp_path, "frost_0714", "2026-07-14")
    second = _write_run(tmp_path, "frost_0715", "2026-07-15")
    dataset = tmp_path / "outputs" / "datasets" / "frost_dataset"
    add_dataset(first, dataset)
    old_cycle = (dataset / "cycles" / "frost_cycle_000001.parquet").read_bytes()
    add_dataset(second, dataset)

    index = pd.read_parquet(dataset / "cycle_index.parquet")
    assert index["cycle_name"].tolist() == [
        "frost_cycle_000001",
        "frost_cycle_000002",
    ]
    assert (dataset / "cycles" / "frost_cycle_000001.parquet").read_bytes() == old_cycle
    manifest_before = (dataset / "dataset_manifest.json").read_bytes()
    add_dataset(second, dataset)
    assert (dataset / "dataset_manifest.json").read_bytes() == manifest_before
    validate_dataset(dataset)


def test_loader_filters_current_assessment_and_analysis_uses_only_loader(
    tmp_path: Path,
) -> None:
    run = _write_run(tmp_path, "frost_0714", "2026-07-14")
    dataset = tmp_path / "outputs" / "datasets" / "frost_dataset"
    add_dataset(run, dataset)
    from frost_analysis.analysis import run_analysis
    from frost_analysis.dataset_manifest import review_cycle

    cycle_name = str(pd.read_parquet(dataset / "cycle_index.parquet").loc[0, "cycle_name"])
    review_cycle(dataset, cycle_name, status="partial", note="manual")
    loader = DatasetLoader(dataset)
    assert loader.list_cycles(statuses={"partial"})["cycle_name"].tolist() == [cycle_name]
    output = tmp_path / "outputs" / "analysis"
    run_analysis(loader, statuses={"partial"}, output_dir=output)
    assert (output / "cycle_statistics.csv").is_file()
    assert (output / "image_sensor_alignment.csv").is_file()


def test_v2_cycle_files_round_trip_source_processed_exactly(tmp_path: Path) -> None:
    first = _write_run(tmp_path, "frost_0714", "2026-07-14")
    second = _write_run(tmp_path, "frost_0715", "2026-07-15")
    dataset = tmp_path / "outputs" / "datasets" / "frost_dataset"
    add_dataset(first, dataset)
    add_dataset(second, dataset)
    loader = DatasetLoader(dataset)

    expected = pd.concat(
        [
            pd.read_parquet(first / "processed_data.parquet"),
            pd.read_parquet(second / "processed_data.parquet"),
        ],
        ignore_index=True,
    )
    exported = [
        loader.load_cycle(name)
        for name in loader.cycle_index["cycle_name"].astype(str)
        if len(loader.load_cycle(name))
    ]
    actual = pd.concat(exported, ignore_index=True)
    path_to_source: dict[str, str] = {}
    for cycle_name in loader.cycle_index["cycle_name"].astype(str):
        for row in loader.load_cycle_images(cycle_name).to_dict(orient="records"):
            path = Path(str(row["path"])).relative_to(dataset).as_posix()
            path_to_source[path] = str(row["source_relative_path"])
    for column in actual.columns:
        if str(column).startswith("image_") and str(column).endswith("_path"):
            present = actual[column].notna()
            actual.loc[present, column] = actual.loc[present, column].map(path_to_source)
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


@pytest.mark.parametrize("failure", ["asset", "metadata"])
def test_v2_add_rolls_back_after_asset_or_metadata_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    first = _write_run(tmp_path, "frost_0714", "2026-07-14")
    second = _write_run(tmp_path, "frost_0715", "2026-07-15")
    dataset = tmp_path / "outputs" / "datasets" / "frost_dataset"
    add_dataset(first, dataset)
    old_manifest = (dataset / "dataset_manifest.json").read_bytes()
    old_index = (dataset / "cycle_index.parquet").read_bytes()

    import frost_analysis.dataset_io as dataset_io

    original_replace = dataset_io.os.replace

    def fail_replace(source: object, target: object) -> None:
        source_path = Path(source)
        target_path = Path(target)
        if (
            failure == "asset"
            and target_path.parent == dataset / "cycles"
            and target_path.name.endswith("000002.parquet")
        ):
            raise RuntimeError("injected asset failure")
        if (
            failure == "metadata"
            and source_path.name == "image_metadata.parquet"
            and target_path == dataset / source_path.name
        ):
            raise RuntimeError("injected metadata failure")
        original_replace(source, target)

    monkeypatch.setattr(dataset_io.os, "replace", fail_replace)
    with pytest.raises(RuntimeError):
        add_dataset(second, dataset)

    assert (dataset / "dataset_manifest.json").read_bytes() == old_manifest
    assert (dataset / "cycle_index.parquet").read_bytes() == old_index
    assert not (dataset / "cycles" / "frost_cycle_000002.parquet").exists()
    validate_dataset(dataset)


def test_v2_validator_finds_asset_hash_and_orphan_errors(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "frost_0714", "2026-07-14")
    dataset = tmp_path / "outputs" / "datasets" / "frost_dataset"
    add_dataset(run, dataset)
    publication = dataset / "cycles" / "frost_cycle_000001.png"
    original = publication.read_bytes()
    publication.write_bytes(b"changed")
    with pytest.raises(ValueError, match="cycle asset SHA mismatch"):
        validate_dataset(dataset)
    publication.write_bytes(original)
    image_root = dataset / "images" / "frost_cycle_000001" / "unassigned_01"
    (image_root / "orphan.jpg").write_bytes(b"orphan")
    with pytest.raises(ValueError, match="orphan image"):
        validate_dataset(dataset)
