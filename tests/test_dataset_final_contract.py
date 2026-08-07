from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from frost_analysis.dataset import assign_final_cycle_names_by_time


def _write_renderable_dataset(dataset_dir: Path) -> tuple[str, dict[str, str]]:
    from frost_analysis.dataset_io import write_json

    cycle_name = "frost_cycle_000001"
    assets = {
        "parquet": f"cycles/{cycle_name}.parquet",
        "csv": f"cycles/{cycle_name}.csv",
        "original_csv": f"cycles_original/{cycle_name}.csv",
        "publication": f"cycles/{cycle_name}.png",
        "rgb_coverage": f"cycles/{cycle_name}_rgb_coverage.png",
    }
    (dataset_dir / "cycles").mkdir(parents=True)
    (dataset_dir / "cycles_original").mkdir()
    (dataset_dir / "images" / cycle_name).mkdir(parents=True)
    timestamps = pd.date_range("2026-07-14 10:00:00", periods=2, freq="10s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cycle_stage": ["recovery", "frost_development"],
        }
    )
    frame.to_parquet(dataset_dir / assets["parquet"], index=False)
    frame.to_csv(dataset_dir / assets["csv"], index=False)
    frame.to_csv(dataset_dir / assets["original_csv"], index=False)
    (dataset_dir / assets["publication"]).write_bytes(b"publication")
    (dataset_dir / assets["rgb_coverage"]).write_bytes(b"coverage")
    metadata_columns = [
        "cycle_uid",
        "cycle_name",
        "source_camera_id",
        "file_name",
        "frame_index",
        "image_time",
        "matched_timestamp",
        "offset_seconds",
        "cycle_stage",
        "source_relative_path",
    ]
    pd.DataFrame(columns=metadata_columns).to_parquet(
        dataset_dir / "image_metadata.parquet", index=False
    )
    write_json(
        {
            "dataset_schema_version": 3,
            "dataset_id": "frost_cycle_dataset",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "experiments": [
                {
                    "experiment_id": "exp",
                    "experiment_date": "2026-07-14",
                    "source_directory": "data/0714",
                    "camera_roles": {},
                }
            ],
        },
        dataset_dir / "dataset_manifest.json",
    )
    write_json(
        {
            "cycles": [
                {
                    "cycle_name": cycle_name,
                    "cycle_uid": "exp::cycle_001",
                    "experiment_id": "exp",
                    "experiment_date": "2026-07-14",
                    "cycle_id": "cycle_001",
                    "pipeline_status": "partial",
                    "pipeline_status_reason": "source_gap",
                    "status": "partial",
                    "status_reason": "source_gap",
                    "boundaries": {},
                    "data": {},
                    "image": {"image_count": 0, "by_camera_role": {}},
                    "assets": assets,
                }
            ]
        },
        dataset_dir / "cycle_catalog.json",
    )
    write_json(
        {
            "fields": [],
            "channels": {},
            "image_coverage": {"max_image_gap_seconds": 40},
        },
        dataset_dir / "channel_registry.json",
    )
    return cycle_name, assets


def test_materialize_cycle_builds_one_catalog_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import frost_analysis.dataset as dataset_module
    from frost_analysis import dataset_metadata
    from frost_analysis.dataset import _DirectDatePipeline, add_dataset

    input_dir = tmp_path / "0714"
    input_dir.mkdir()
    timestamps = pd.date_range("2026-07-14 10:00:00", periods=2, freq="10s")
    frame = pd.DataFrame(
        {
            "experiment_id": ["exp_0714"] * 2,
            "experiment_date": ["2026-07-14"] * 2,
            "cycle_id": ["cycle_001"] * 2,
            "cycle_status": ["partial"] * 2,
            "cycle_status_reason": ["source_gap"] * 2,
            "timestamp": timestamps,
            "cycle_stage": ["recovery", "frost_development"],
            "signal": [1.0, 2.0],
        }
    )
    summary = pd.DataFrame(
        [
            {
                "experiment_id": "exp_0714",
                "experiment_date": "2026-07-14",
                "cycle_id": "cycle_001",
                "segment_start": timestamps[0],
                "cycle_status": "partial",
                "cycle_status_reason": "source_gap",
                "heating_start": timestamps[0],
                "stable_heating_start": timestamps[1],
            }
        ]
    )
    pipeline = _DirectDatePipeline(
        input_dir=input_dir,
        config=SimpleNamespace(
            experiment_id="exp_0714",
            experiment_date="2026-07-14",
            project_root=tmp_path,
            cycles=SimpleNamespace(stable_heating_seconds=180),
            process=SimpleNamespace(
                resample_interval_seconds=10,
                baseline=SimpleNamespace(baseline_seconds=60),
                feature_windows_minutes=(),
            ),
            analysis=SimpleNamespace(feature_windows_minutes=[]),
        ),
        channels={
            "signal": {
                "kind": "continuous",
                "resample": "mean",
                "analysis_candidate": False,
                "role": "performance",
                "unit": "1",
            }
        },
        prepared=frame,
        summary=summary,
        processed=frame.copy(),
    )
    monkeypatch.setattr(dataset_module, "_validate_date_input", lambda *_args: None)
    monkeypatch.setattr(
        dataset_module,
        "_load_config_for_input",
        lambda *_args: pipeline.config,
    )
    monkeypatch.setattr(dataset_module, "_run_direct_pipeline", lambda *_args: pipeline)
    original_builder = dataset_metadata.build_cycle_record
    call_count = 0

    def counting_builder(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(dataset_metadata, "build_cycle_record", counting_builder)

    add_dataset(input_dir, tmp_path / "dataset")

    assert call_count == 1
    dataset_dir = tmp_path / "dataset"
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    assert set(manifest) == {"dataset_schema_version", "dataset_id", "experiments"}
    processed = pd.read_parquet(dataset_dir / "cycles/frost_cycle_000001.parquet")
    assert not {
        "dataset_id",
        "dataset_schema_version",
        "dataset_cycle_index",
    } & set(processed)
    metadata = pd.read_parquet(dataset_dir / "image_metadata.parquet")
    assert metadata.columns.tolist() == [
        "cycle_uid",
        "cycle_name",
        "source_camera_id",
        "file_name",
        "frame_index",
        "image_time",
        "matched_timestamp",
        "offset_seconds",
        "cycle_stage",
        "source_relative_path",
    ]
    catalog = json.loads((dataset_dir / "cycle_catalog.json").read_text())
    assert catalog["cycles"][0]["image"] == {"image_count": 0}


def test_metadata_readers_allow_extra_review_fields(tmp_path: Path) -> None:
    from frost_analysis.dataset_io import write_json
    from frost_analysis.dataset_metadata import read_catalog, read_manifest, write_catalog

    write_json(
        {
            "dataset_schema_version": 3,
            "dataset_id": "frost_cycle_dataset",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "experiments": [
                {
                    "experiment_id": "exp",
                    "experiment_date": "2026-07-14",
                    "source_directory": "data/0714",
                    "experiment_note": "manual note",
                }
            ],
            "comment": "dataset note",
        },
        tmp_path / "dataset_manifest.json",
    )
    catalog = {
        "catalog_note": "review later",
        "cycles": [{"cycle_name": "frost_cycle_000001", "review_note": "check"}],
    }
    write_json(catalog, tmp_path / "cycle_catalog.json")

    manifest = read_manifest(tmp_path)
    loaded_catalog = read_catalog(tmp_path)
    write_catalog(tmp_path, loaded_catalog)

    assert manifest["comment"] == "dataset note"
    assert manifest["experiments"][0]["experiment_note"] == "manual note"
    assert loaded_catalog["catalog_note"] == "review later"
    assert loaded_catalog["cycles"][0]["review_note"] == "check"


def test_review_cycle_does_not_scan_images_or_update_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis import dataset_images, visualization
    from frost_analysis.dataset import review_cycle

    cycle_name, _ = _write_renderable_dataset(tmp_path)
    manifest_before = (tmp_path / "dataset_manifest.json").read_bytes()
    render_calls: list[str] = []

    def fail_scan(*args: object, **kwargs: object) -> pd.DataFrame:
        pytest.fail("review-cycle must not scan current image directories")

    monkeypatch.setattr(dataset_images, "scan_cycle_images", fail_scan)
    monkeypatch.setattr(
        visualization,
        "render_cycle_publication",
        lambda _frame, _record, _path: render_calls.append(cycle_name),
    )

    review_cycle(tmp_path, cycle_name, status="valid", reason="reviewed")

    catalog = json.loads((tmp_path / "cycle_catalog.json").read_text())
    assert catalog["cycles"][0]["status"] == "valid"
    assert catalog["cycles"][0]["status_reason"] == "reviewed"
    assert render_calls == [cycle_name]
    assert (tmp_path / "dataset_manifest.json").read_bytes() == manifest_before


def test_render_only_draws_without_writing_catalog_or_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis import dataset_io, visualization
    from frost_analysis.dataset import render_dataset

    cycle_name, _ = _write_renderable_dataset(tmp_path)
    catalog_before = (tmp_path / "cycle_catalog.json").read_bytes()
    manifest_before = (tmp_path / "dataset_manifest.json").read_bytes()
    writes: list[Path] = []
    publication_calls: list[str] = []
    coverage_calls: list[str] = []

    monkeypatch.setattr(
        dataset_io,
        "write_json",
        lambda _data, path: writes.append(path),
    )
    monkeypatch.setattr(
        visualization,
        "render_cycle_publication",
        lambda _frame, _record, _path: publication_calls.append(cycle_name),
    )
    monkeypatch.setattr(
        visualization,
        "render_rgb_coverage_intervals",
        lambda _name, _start, _end, _intervals, _path, **_kwargs: coverage_calls.append(cycle_name),
    )

    render_dataset(tmp_path, cycle_name, publication=True, coverage=True)

    assert writes == []
    assert publication_calls == [cycle_name]
    assert coverage_calls == [cycle_name]
    assert (tmp_path / "cycle_catalog.json").read_bytes() == catalog_before
    assert (tmp_path / "dataset_manifest.json").read_bytes() == manifest_before


def test_validate_scientific_schema_ignores_catalog_counts(
    tmp_path: Path,
) -> None:
    from frost_analysis.dataset import make_cycle_uid
    from frost_analysis.dataset_io import write_json
    from frost_analysis.dataset_validation import validate_dataset

    cycle_name = "frost_cycle_000001"
    cycle_uid = make_cycle_uid("exp", "cycle_001")
    processed = pd.DataFrame(
        {
            "cycle_name": [cycle_name] * 2,
            "cycle_uid": [cycle_uid] * 2,
            "experiment_id": ["exp"] * 2,
            "cycle_id": ["cycle_001"] * 2,
            "timestamp": pd.date_range("2026-07-14 10:00:00", periods=2, freq="10s"),
            "signal": [1.0, 2.0],
        }
    )
    original = processed[["experiment_id", "cycle_id", "timestamp", "signal"]].copy()
    (tmp_path / "cycles").mkdir()
    (tmp_path / "cycles_original").mkdir()
    (tmp_path / "images").mkdir()
    processed.to_parquet(tmp_path / f"cycles/{cycle_name}.parquet", index=False)
    processed.to_csv(tmp_path / f"cycles/{cycle_name}.csv", index=False)
    original.to_csv(tmp_path / f"cycles_original/{cycle_name}.csv", index=False)
    for filename in (f"{cycle_name}.png", f"{cycle_name}_rgb_coverage.png"):
        (tmp_path / "cycles" / filename).write_bytes(b"figure")
    metadata_columns = [
        "cycle_uid",
        "cycle_name",
        "source_camera_id",
        "file_name",
        "frame_index",
        "image_time",
        "matched_timestamp",
        "offset_seconds",
        "cycle_stage",
        "source_relative_path",
    ]
    pd.DataFrame(columns=metadata_columns).to_parquet(
        tmp_path / "image_metadata.parquet", index=False
    )
    write_json(
        {
            "dataset_schema_version": 3,
            "dataset_id": "frost_cycle_dataset",
            "experiments": [
                {
                    "experiment_id": "exp",
                    "experiment_date": "2026-07-14",
                    "source_directory": "data/0714",
                    "camera_roles": {},
                }
            ],
        },
        tmp_path / "dataset_manifest.json",
    )
    write_json(
        {
            "cycles": [
                {
                    "cycle_name": cycle_name,
                    "cycle_uid": cycle_uid,
                    "experiment_id": "exp",
                    "cycle_id": "cycle_001",
                    "data": {
                        "processed_row_count": 999,
                        "original_row_count": 999,
                    },
                    "assets": {
                        "parquet": f"cycles/{cycle_name}.parquet",
                        "csv": f"cycles/{cycle_name}.csv",
                        "original_csv": f"cycles_original/{cycle_name}.csv",
                        "publication": f"cycles/{cycle_name}.png",
                        "rgb_coverage": f"cycles/{cycle_name}_rgb_coverage.png",
                    },
                }
            ]
        },
        tmp_path / "cycle_catalog.json",
    )
    write_json(
        {
            "columns": ["experiment_id", "cycle_id", "timestamp", "signal"],
            "channels": {},
        },
        tmp_path / "channel_registry.json",
    )

    validate_dataset(tmp_path)


def test_final_cycle_names_follow_segment_start_not_cycle_id_order() -> None:
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_20260714"] * 4,
            "experiment_date": ["2026-07-14"] * 4,
            "cycle_id": ["cycle_001", "partial_001", "cycle_002", "partial_002"],
            "segment_start": pd.to_datetime(
                [
                    "2026-07-14 17:41:16",
                    "2026-07-14 17:23:01",
                    "2026-07-14 18:01:04",
                    "2026-07-14 21:11:42",
                ]
            ),
        }
    )

    names = assign_final_cycle_names_by_time(summary)

    assert names == {
        ("exp_20260714", "partial_001"): "frost_cycle_000001",
        ("exp_20260714", "cycle_001"): "frost_cycle_000002",
        ("exp_20260714", "cycle_002"): "frost_cycle_000003",
        ("exp_20260714", "partial_002"): "frost_cycle_000004",
    }


def test_final_cycle_names_skip_summary_only_cycles() -> None:
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_20260714", "exp_20260714"],
            "experiment_date": ["2026-07-14", "2026-07-14"],
            "cycle_id": ["cycle_001", "cycle_002"],
            "segment_start": pd.to_datetime(["2026-07-14 17:41:16", "2026-07-14 18:01:04"]),
        }
    )
    prepared = pd.DataFrame(
        {
            "experiment_id": ["exp_20260714"],
            "cycle_id": ["cycle_001"],
        }
    )

    names = assign_final_cycle_names_by_time(summary, prepared=prepared)

    assert names == {("exp_20260714", "cycle_001"): "frost_cycle_000001"}


def test_cli_default_dataset_paths_use_project_root(monkeypatch: object, tmp_path: Path) -> None:
    from frost_analysis import cli

    calls: list[tuple[str, Path]] = []
    dataset = tmp_path / "dataset"
    monkeypatch.setattr(cli, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        cli,
        "run_analysis",
        lambda loader, **kwargs: calls.append(("analysis", loader.dataset_root)) or tmp_path,
    )
    monkeypatch.setattr(
        cli,
        "DatasetLoader",
        lambda path: type(
            "Loader",
            (),
            {
                "dataset_root": Path(path),
                "list_cycles": lambda self: pd.DataFrame(),
                "load_image_metadata": lambda self: pd.DataFrame(),
            },
        )(),
    )
    monkeypatch.setattr(
        cli,
        "edit_dataset",
        lambda path, **kwargs: calls.append(("edit", path)) or path,
    )

    assert cli.main(["analysis", "--output", str(tmp_path / "analysis")]) == 0
    assert cli.main(["dataset", "edit", "--baseline-seconds", "60"]) == 0
    assert calls == [("analysis", dataset), ("edit", dataset)]


def test_recovery_transform_recomputes_stage_coordinates_features_and_images() -> None:
    from frost_analysis.dataset_edit import apply_recovery

    timestamps = pd.date_range("2026-07-14 10:00:00", periods=12, freq="10s")
    stages = [
        "recovery",
        "frost_development",
        *["frost_development"] * 8,
        "defrost",
        "defrost",
    ]
    original = pd.DataFrame(
        {
            "experiment_id": ["exp"] * len(timestamps),
            "cycle_id": ["cycle_001"] * len(timestamps),
            "timestamp": timestamps,
            "cycle_stage": stages,
        }
    )
    processed = original.assign(
        signal=list(range(len(timestamps))),
        cycle_elapsed_seconds=0.0,
        cycle_progress=0.0,
        signal__lag_1min=100.0,
    )
    metadata = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000001"] * 2,
            "matched_timestamp": [timestamps[1], timestamps[3]],
            "image_time": [timestamps[1], timestamps[3]],
            "cycle_stage": ["frost_development"] * 2,
        }
    )
    record = {
        "cycle_name": "frost_cycle_000001",
        "experiment_id": "exp",
        "cycle_id": "cycle_001",
        "boundaries": {
            "heating_start": timestamps[0].isoformat(),
            "stable_heating_start": timestamps[1].isoformat(),
            "defrost_start": timestamps[10].isoformat(),
            "defrost_end": None,
        },
    }
    registry = {
        "resample_interval_seconds": 10,
        "channels": {"signal": {"analysis_candidate": True, "kind": "continuous"}},
        "processing_settings": {"feature_windows_minutes": [1]},
    }

    new_original, new_processed, new_metadata = apply_recovery(
        original,
        processed,
        metadata,
        record,
        registry,
        mode="seconds",
        seconds=20,
    )

    assert new_original.loc[1, "cycle_stage"] == "recovery"
    assert new_original.loc[2, "cycle_stage"] == "frost_development"
    assert pd.isna(new_processed.loc[1, "cycle_elapsed_seconds"])
    assert new_processed.loc[2, "cycle_elapsed_seconds"] == 0.0
    assert new_processed.loc[5, "cycle_progress"] == pytest.approx(0.375)
    assert pd.isna(new_processed.loc[7, "signal__lag_1min"])
    assert new_metadata["cycle_stage"].tolist() == ["recovery", "frost_development"]


def test_partial_stage_context_keeps_known_defrost_without_temperature_evidence() -> None:
    from frost_analysis.cycles import _partial_stage_context

    timestamps = pd.date_range("2026-07-14 10:00:00", periods=5, freq="10s")
    segment = pd.DataFrame(
        {
            "timestamp": timestamps,
            "defrost_active": [False, False, True, True, False],
        }
    )

    stages, heating_start, stable_start, defrost_start, defrost_end = _partial_stage_context(
        segment, "defrost_active", {}
    )

    assert heating_start == timestamps[0]
    assert stable_start is None
    assert defrost_start == timestamps[2]
    assert defrost_end == timestamps[4]
    assert stages.tolist() == ["partial", "partial", "defrost", "defrost", "partial"]


def test_sensor_coverage_keeps_required_channels_in_the_mask() -> None:
    from frost_analysis.dataset_images import _sensor_coverage_intervals

    timestamps = pd.date_range("2026-07-14 10:00:00", periods=4, freq="10s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "required_a": [1.0, 1.0, 1.0, 1.0],
            "required_b": [None, None, None, None],
            "required_a__imputed": [False] * 4,
            "required_b__imputed": [False] * 4,
        }
    )
    registry = {
        "channels": {
            "required_a": {"coverage_required": True},
            "required_b": {"coverage_required": True},
        }
    }

    intervals = _sensor_coverage_intervals(frame, registry)

    assert intervals["available"] == []
    assert intervals["missing"] == [
        (timestamps[0], timestamps[-1] + pd.Timedelta(seconds=10)),
    ]


def test_refresh_dataset_does_not_modify_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis import visualization
    from frost_analysis.dataset import refresh_dataset
    from frost_analysis.dataset_io import write_json

    (tmp_path / "cycles").mkdir()
    (tmp_path / "cycles_original").mkdir()
    (tmp_path / "images").mkdir()
    timestamps = pd.date_range("2026-07-14 10:00:00", periods=2, freq="10s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cycle_stage": ["recovery", "frost_development"],
        }
    )
    frame.to_parquet(tmp_path / "cycles/frost_cycle_000001.parquet", index=False)
    frame.to_csv(tmp_path / "cycles/frost_cycle_000001.csv", index=False)
    frame.to_csv(tmp_path / "cycles_original/frost_cycle_000001.csv", index=False)
    (tmp_path / "cycles/frost_cycle_000001.png").write_bytes(b"publication")
    (tmp_path / "cycles/frost_cycle_000001_rgb_coverage.png").write_bytes(b"coverage")
    write_json(
        {
            "dataset_schema_version": 3,
            "dataset_id": "frost_cycle_dataset",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "experiments": [
                {
                    "experiment_id": "exp",
                    "experiment_date": "2026-07-14",
                    "source_directory": "data/0714",
                    "camera_roles": {},
                }
            ],
        },
        tmp_path / "dataset_manifest.json",
    )
    write_json(
        {
            "cycles": [
                    {
                    "cycle_name": "frost_cycle_000001",
                    "experiment_id": "exp",
                    "assets": {
                        "parquet": "cycles/frost_cycle_000001.parquet",
                        "csv": "cycles/frost_cycle_000001.csv",
                        "original_csv": "cycles_original/frost_cycle_000001.csv",
                        "publication": "cycles/frost_cycle_000001.png",
                        "rgb_coverage": "cycles/frost_cycle_000001_rgb_coverage.png",
                    },
                    "data": {},
                    "image": {},
                }
            ]
        },
        tmp_path / "cycle_catalog.json",
    )
    write_json({"fields": [], "channels": []}, tmp_path / "channel_registry.json")
    pd.DataFrame(
        columns=[
            "cycle_uid",
            "cycle_name",
            "source_camera_id",
            "file_name",
            "frame_index",
            "image_time",
            "matched_timestamp",
            "offset_seconds",
            "cycle_stage",
            "source_relative_path",
        ]
    ).to_parquet(tmp_path / "image_metadata.parquet", index=False)

    monkeypatch.setattr(
        visualization,
        "render_cycle_publication",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        visualization,
        "render_rgb_coverage_intervals",
        lambda *_args, **_kwargs: None,
    )

    manifest_before = (tmp_path / "dataset_manifest.json").read_bytes()
    refresh_dataset(tmp_path)

    assert (tmp_path / "dataset_manifest.json").read_bytes() == manifest_before


def test_dataset_add_append_edit_refresh_loader_validate_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import frost_analysis.dataset as dataset_module
    from frost_analysis.dataset import _DirectDatePipeline, add_dataset
    from frost_analysis.dataset_loader import DatasetLoader
    from frost_analysis.dataset_validation import validate_dataset

    def make_pipeline(input_dir: Path) -> _DirectDatePipeline:
        date = {
            "0714": "2026-07-14",
            "0715": "2026-07-15",
            "0716": "2026-07-16",
        }[input_dir.name]
        experiment_id = f"exp_{input_dir.name}"
        timestamps = pd.date_range(f"{date} 10:00:00", periods=12, freq="10s")
        stages = pd.Series(
            ["recovery", "recovery"] + ["frost_development"] * 8 + ["defrost", "defrost"],
            dtype="string",
        )
        common = {
            "experiment_id": [experiment_id] * len(timestamps),
            "experiment_date": [date] * len(timestamps),
            "cycle_id": ["cycle_001"] * len(timestamps),
            "cycle_status": ["valid"] * len(timestamps),
            "cycle_status_reason": [""] * len(timestamps),
            "timestamp": timestamps,
            "cycle_stage": stages,
            "signal": [float(value) for value in range(len(timestamps))],
            "cycle_elapsed_seconds": [
                None,
                None,
                *[float(value * 10) for value in range(8)],
                None,
                None,
            ],
            "cycle_progress": [
                None,
                None,
                *[value / 8 for value in range(8)],
                None,
                None,
            ],
            "signal__baseline": [None] * len(timestamps),
            "signal__baseline_residual": [None] * len(timestamps),
        }
        prepared = pd.DataFrame(common)
        if input_dir.name == "0716":
            prepared["humidity"] = [None] * len(timestamps)
        processed = prepared.drop(columns=["humidity"], errors="ignore").copy()
        processed["signal__lag_1min"] = [None] * len(timestamps)
        processed["signal__delta_1min"] = [None] * len(timestamps)
        processed["signal__rolling_mean_1min"] = [None] * len(timestamps)
        summary = pd.DataFrame(
            [
                {
                    "experiment_id": experiment_id,
                    "experiment_date": date,
                    "cycle_id": "cycle_001",
                    "segment_start": timestamps[0],
                    "cycle_status": "valid",
                    "cycle_status_reason": "",
                    "heating_start": timestamps[0],
                    "stable_heating_start": timestamps[2],
                    "defrost_start": timestamps[10],
                    "defrost_end": timestamps[-1] + pd.Timedelta(seconds=10),
                }
            ]
        )
        config = SimpleNamespace(
            experiment_id=experiment_id,
            experiment_date=date,
            project_root=tmp_path,
            cycles=SimpleNamespace(stable_heating_seconds=20),
            process=SimpleNamespace(
                resample_interval_seconds=10,
                baseline=SimpleNamespace(baseline_seconds=60),
                feature_windows_minutes=(1,),
            ),
            analysis=SimpleNamespace(feature_windows_minutes=[1]),
        )
        channels = {
            "signal": {
                "kind": "continuous",
                "resample": "mean",
                "analysis_candidate": True,
                "role": "performance",
                "unit": "1",
            }
        }
        return _DirectDatePipeline(
            input_dir=input_dir,
            config=config,
            channels=channels,
            prepared=prepared,
            summary=summary,
            processed=processed,
        )

    first_input = tmp_path / "0714"
    second_input = tmp_path / "0715"
    first_input.mkdir()
    second_input.mkdir()
    monkeypatch.setattr(dataset_module, "_validate_date_input", lambda *_args: None)
    monkeypatch.setattr(
        dataset_module,
        "_run_direct_pipeline",
        lambda input_dir, _project_root: make_pipeline(input_dir),
    )
    dataset_dir = tmp_path / "dataset"

    add_dataset(first_input, dataset_dir)
    initial_registry = json.loads((dataset_dir / "channel_registry.json").read_text())
    initial_catalog = json.loads((dataset_dir / "cycle_catalog.json").read_text())
    assert initial_registry["baseline_managed"] is False
    assert initial_registry["recovery_edit"]["managed"] is False
    assert "canonical_hash" not in initial_registry
    assert all("asset_sha256" not in record for record in initial_catalog["cycles"])
    add_dataset(second_input, dataset_dir)
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    assert all("source_fingerprint" not in item for item in manifest["experiments"])

    from frost_analysis import dataset_images
    from frost_analysis.dataset import edit_dataset, refresh_dataset

    read_parquet = dataset_module.pd.read_parquet
    scan_cycle_images = dataset_images.scan_cycle_images
    processed_reads = 0
    image_scans = 0

    def counting_read_parquet(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal processed_reads
        if "/cycles/" in str(path) and str(path).endswith(".parquet"):
            processed_reads += 1
        return read_parquet(path, *args, **kwargs)

    def counting_scan(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal image_scans
        image_scans += 1
        return scan_cycle_images(*args, **kwargs)

    monkeypatch.setattr(dataset_module.pd, "read_parquet", counting_read_parquet)
    monkeypatch.setattr(dataset_images, "scan_cycle_images", counting_scan)

    edit_dataset(dataset_dir, baseline_seconds=30, recovery_seconds=30)
    assert processed_reads == 2
    assert image_scans == 2
    edited_registry = json.loads((dataset_dir / "channel_registry.json").read_text())
    assert edited_registry["baseline_managed"] is True
    assert edited_registry["recovery_edit"]["managed"] is True
    third_input = tmp_path / "0716"
    third_input.mkdir()
    add_dataset(third_input, dataset_dir)
    refresh_dataset(dataset_dir)
    loader = DatasetLoader(dataset_dir)
    validate_dataset(dataset_dir)

    assert len(loader.list_cycles()) == 3
    assert loader.registry["baseline_seconds"] == 30
    assert loader.registry["recovery_edit"]["mode"] == "seconds"
    assert "humidity" in loader.load_cycle_original("frost_cycle_000001").columns
    assert loader.load_cycle_original("frost_cycle_000001")["humidity"].isna().all()
    new_record = loader.get_cycle_record("frost_cycle_000003")
    assert new_record["boundaries"]["stable_heating_start"].endswith("10:00:30")
    assert new_record["boundaries"]["baseline_start"].endswith("10:00:30")
    assert new_record["boundaries"]["baseline_end"].endswith("10:01:00")


def test_dataset_io_exposes_direct_writes_without_transactions() -> None:
    import frost_analysis.dataset_io as dataset_io

    assert hasattr(dataset_io, "write_json")
    assert hasattr(dataset_io, "write_csv")
    assert hasattr(dataset_io, "write_parquet")
    assert not hasattr(dataset_io, "mutate_dataset")
    assert not hasattr(dataset_io, "publish_with_rollback")


def test_baseline_edit_does_not_require_original_or_image_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis.dataset import edit_dataset
    from frost_analysis.dataset_io import write_json

    cycle_name = "frost_cycle_000001"
    cycles_dir = tmp_path / "cycles"
    cycles_dir.mkdir()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-14 10:00:00", periods=2, freq="10s"),
            "cycle_stage": ["frost_development", "frost_development"],
        }
    )
    frame.to_parquet(cycles_dir / f"{cycle_name}.parquet", index=False)
    frame.to_csv(cycles_dir / f"{cycle_name}.csv", index=False)
    write_json(
        {
            "dataset_schema_version": 3,
            "dataset_id": "frost_cycle_dataset",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "experiments": [],
        },
        tmp_path / "dataset_manifest.json",
    )
    write_json(
        {
            "cycles": [
                {
                    "cycle_name": cycle_name,
                    "cycle_uid": "exp::cycle_001",
                    "experiment_id": "exp",
                    "cycle_id": "cycle_001",
                    "boundaries": {
                        "stable_heating_start": "2026-07-14T10:00:00",
                    },
                    "assets": {
                        "parquet": f"cycles/{cycle_name}.parquet",
                        "csv": f"cycles/{cycle_name}.csv",
                        "publication": f"cycles/{cycle_name}.png",
                    },
                }
            ]
        },
        tmp_path / "cycle_catalog.json",
    )
    write_json(
        {
            "channels": {},
            "columns": ["timestamp", "cycle_stage"],
            "processing_settings": {"feature_windows_minutes": []},
        },
        tmp_path / "channel_registry.json",
    )
    monkeypatch.setattr(
        "frost_analysis.visualization.render_cycle_publication",
        lambda *_args, **_kwargs: None,
    )

    edit_dataset(tmp_path, baseline_seconds=60)


def test_loader_resolves_camera_role_from_manifest_without_renaming_directory(
    tmp_path: Path,
) -> None:
    from frost_analysis.dataset_io import write_json
    from frost_analysis.dataset_loader import DatasetLoader

    cycle_name = "frost_cycle_000001"
    camera_dir = tmp_path / "images" / cycle_name / "camera01"
    camera_dir.mkdir(parents=True)
    (camera_dir / "frame_0001.jpg").write_bytes(b"image")
    cycles_dir = tmp_path / "cycles"
    cycles_dir.mkdir()
    (tmp_path / "cycles_original").mkdir()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-14 10:00:00", periods=2, freq="10s"),
            "cycle_stage": ["frost_development", "frost_development"],
        }
    )
    parquet_path = cycles_dir / f"{cycle_name}.parquet"
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(cycles_dir / f"{cycle_name}.csv", index=False)
    frame.to_csv(tmp_path / "cycles_original" / f"{cycle_name}.csv", index=False)
    (cycles_dir / f"{cycle_name}.png").write_bytes(b"publication")
    (cycles_dir / f"{cycle_name}_rgb_coverage.png").write_bytes(b"coverage")
    write_json(
        {
            "dataset_schema_version": 3,
            "dataset_id": "frost_cycle_dataset",
            "experiments": [
                {
                    "experiment_id": "exp",
                    "experiment_date": "2026-07-14",
                    "source_directory": "data/0714",
                    "camera_roles": {"camera01": "front"},
                }
            ],
        },
        tmp_path / "dataset_manifest.json",
    )
    write_json(
        {
            "cycles": [
                {
                    "cycle_name": cycle_name,
                    "cycle_uid": "exp::cycle_001",
                    "experiment_id": "exp",
                    "cycle_id": "cycle_001",
                    "image": {},
                    "assets": {
                        "parquet": f"cycles/{cycle_name}.parquet",
                        "csv": f"cycles/{cycle_name}.csv",
                        "original_csv": f"cycles_original/{cycle_name}.csv",
                        "publication": f"cycles/{cycle_name}.png",
                        "rgb_coverage": f"cycles/{cycle_name}_rgb_coverage.png",
                    },
                }
            ]
        },
        tmp_path / "cycle_catalog.json",
    )
    write_json(
        {
            "channels": {},
            "columns": ["timestamp", "cycle_stage"],
            "image_coverage": {"max_image_gap_seconds": 40},
        },
        tmp_path / "channel_registry.json",
    )
    pd.DataFrame(
        [
            {
                "cycle_uid": "exp::cycle_001",
                "cycle_name": cycle_name,
                "source_camera_id": "camera01",
                "file_name": "frame_0001.jpg",
                "frame_index": 1,
                "image_time": frame["timestamp"].iloc[0],
                "matched_timestamp": frame["timestamp"].iloc[0],
                "offset_seconds": 0.0,
                "cycle_stage": "frost_development",
                "source_relative_path": "camera01/frame_0001.jpg",
            }
        ]
    ).to_parquet(tmp_path / "image_metadata.parquet", index=False)

    images = DatasetLoader(tmp_path).load_cycle_images(cycle_name)

    assert images["camera_role"].tolist() == ["front"]
    assert images["path"].tolist() == [camera_dir / "frame_0001.jpg"]

    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    manifest["experiments"][0]["camera_roles"]["camera01"] = "top"
    write_json(manifest, tmp_path / "dataset_manifest.json")

    renamed = DatasetLoader(tmp_path).load_cycle_images(cycle_name)
    assert renamed["camera_role"].tolist() == ["top"]
    assert camera_dir.is_dir()


def test_dataset_edit_cli_rejects_removed_camera_rename_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import frost_analysis.cli as cli

    monkeypatch.setattr(cli, "edit_dataset", lambda *_args, **_kwargs: Path("dataset"))

    with pytest.raises(SystemExit):
        cli.main(["dataset", "edit", "--rename-camera", "old=new"])


def test_update_cycle_columns_writes_parquet_and_csv_by_timestamp(tmp_path: Path) -> None:
    from frost_analysis.dataset import update_cycle_columns

    cycle_name = "frost_cycle_000001"
    cycles = tmp_path / "cycles"
    cycles.mkdir()
    timestamps = pd.date_range("2026-07-14 10:00:00", periods=2, freq="10s")
    frame = pd.DataFrame({"timestamp": timestamps, "heating_capacity": [10.0, 12.0]})
    frame.to_parquet(cycles / f"{cycle_name}.parquet", index=False)
    frame.to_csv(cycles / f"{cycle_name}.csv", index=False)

    update_cycle_columns(
        tmp_path,
        {
            cycle_name: pd.DataFrame(
                {
                    "timestamp": timestamps[::-1].astype(str),
                    "compressor_power": [2.0, 1.0],
                    "evaporator_capacity": [10.0, 9.0],
                }
            )
        },
    )

    expected = pd.DataFrame(
        {
            "timestamp": timestamps,
            "heating_capacity": [10.0, 12.0],
            "compressor_power": [1.0, 2.0],
            "evaporator_capacity": [9.0, 10.0],
        }
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(cycles / f"{cycle_name}.parquet"), expected
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(cycles / f"{cycle_name}.csv", parse_dates=["timestamp"]), expected
    )


def test_dataset_add_same_experiment_identity_is_a_noop_without_source_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import frost_analysis.dataset as dataset_module
    from frost_analysis.dataset import add_dataset
    from frost_analysis.dataset_io import write_json

    input_dir = tmp_path / "0714"
    input_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    write_json(
        {
            "dataset_schema_version": 3,
            "dataset_id": "frost_cycle_dataset",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "experiments": [
                {
                    "experiment_id": "exp_0714",
                    "experiment_date": "2026-07-14",
                    "source_directory": "data/0714",
                }
            ],
        },
        dataset_dir / "dataset_manifest.json",
    )
    monkeypatch.setattr(dataset_module, "_validate_date_input", lambda *_args: None)
    monkeypatch.setattr(
        dataset_module,
        "_load_config_for_input",
        lambda *_args: type(
            "ConfigStub",
            (),
            {"experiment_id": "exp_0714", "experiment_date": "2026-07-14"},
        )(),
    )
    monkeypatch.setattr(
        dataset_module,
        "_run_direct_pipeline",
        lambda *_args: pytest.fail("same experiment must not rerun Prepare/Process"),
    )

    assert add_dataset(input_dir, dataset_dir) == dataset_dir.resolve()
