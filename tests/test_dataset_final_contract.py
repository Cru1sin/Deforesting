from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from frost_analysis.dataset import assign_final_cycle_names_by_time


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
            "segment_start": pd.to_datetime(
                ["2026-07-14 17:41:16", "2026-07-14 18:01:04"]
            ),
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
        lambda path: type("Loader", (), {
            "dataset_root": Path(path),
            "list_cycles": lambda self: pd.DataFrame(),
            "load_image_metadata": lambda self: pd.DataFrame(),
        })(),
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

    stages, heating_start, stable_start, defrost_start, defrost_end = (
        _partial_stage_context(segment, "defrost_active", {})
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


def test_refresh_cycles_updates_manifest_timestamp(tmp_path: Path) -> None:
    import json

    from frost_analysis.dataset import _refresh_cycles
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
            "experiments": [],
        },
        tmp_path / "dataset_manifest.json",
    )
    write_json(
        {
            "cycles": [
                {
                    "cycle_name": "frost_cycle_000001",
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
            "image_id",
            "cycle_uid",
            "cycle_name",
            "source_camera_id",
            "file_name",
            "frame_index",
            "initial_camera_slot",
            "image_time",
            "matched_timestamp",
            "offset_seconds",
            "cycle_stage",
            "source_relative_path",
            "file_size_bytes",
        ]
    ).to_parquet(tmp_path / "image_metadata.parquet", index=False)

    _refresh_cycles(
        tmp_path,
        render_publication=False,
        render_coverage=False,
    )

    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    assert manifest["updated_at"] != "2026-01-01T00:00:00+00:00"


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
    initial_registry = json.loads(
        (dataset_dir / "channel_registry.json").read_text()
    )
    initial_catalog = json.loads(
        (dataset_dir / "cycle_catalog.json").read_text()
    )
    assert initial_registry["baseline_managed"] is False
    assert initial_registry["recovery_edit"]["managed"] is False
    assert "canonical_hash" not in initial_registry
    assert all("asset_sha256" not in record for record in initial_catalog["cycles"])
    add_dataset(second_input, dataset_dir)
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    assert all("source_fingerprint" not in item for item in manifest["experiments"])

    from frost_analysis.dataset import edit_dataset, refresh_dataset

    edit_dataset(dataset_dir, baseline_seconds=30, recovery_seconds=30)
    edited_registry = json.loads(
        (dataset_dir / "channel_registry.json").read_text()
    )
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
                    },
                }
            ]
        },
        tmp_path / "cycle_catalog.json",
    )
    write_json(
        {
            "channels": {},
            "fields": [],
            "processing_settings": {"feature_windows_minutes": []},
        },
        tmp_path / "channel_registry.json",
    )
    monkeypatch.setattr(
        "frost_analysis.dataset._refresh_cycles",
        lambda *_args, **_kwargs: None,
    )

    edit_dataset(tmp_path, baseline_seconds=60)


def test_camera_rename_does_not_read_scientific_cycle_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis.dataset import edit_dataset
    from frost_analysis.dataset_io import write_json

    cycle_name = "frost_cycle_000001"
    role_dir = tmp_path / "images" / cycle_name / "camera01__unassigned_01"
    role_dir.mkdir(parents=True)
    (role_dir / "frame_0001.jpg").write_bytes(b"image")
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
                    "assets": {},
                }
            ]
        },
        tmp_path / "cycle_catalog.json",
    )
    write_json({"channels": {}, "fields": []}, tmp_path / "channel_registry.json")
    monkeypatch.setattr(
        "frost_analysis.dataset._refresh_cycles",
        lambda *_args, **_kwargs: None,
    )

    edit_dataset(tmp_path, camera_renames=["unassigned_01=front"])

    assert (tmp_path / "images" / cycle_name / "camera01__front").is_dir()


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
