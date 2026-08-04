from __future__ import annotations

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


def test_mutate_dataset_uses_hardlinks_for_unchanged_files(tmp_path: Path) -> None:
    from frost_analysis.dataset_io import mutate_dataset

    dataset = tmp_path / "dataset"
    (dataset / "cycles").mkdir(parents=True)
    (dataset / "cycles_original").mkdir()
    (dataset / "images").mkdir()
    (dataset / "untouched.bin").write_bytes(b"stable")

    def operation(staging: Path) -> None:
        assert (staging / "untouched.bin").stat().st_ino == (
            dataset / "untouched.bin"
        ).stat().st_ino

    # The structure callback is intentionally not used for this transaction test.
    mutate_dataset(dataset, operation, validate=lambda _path: None)

    assert (dataset / "untouched.bin").read_bytes() == b"stable"


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


def test_recovery_edit_uses_temperature_crossing_before_stage_rewrite(tmp_path: Path) -> None:
    from frost_analysis.dataset_edit import apply_recovery_edit

    original_root = tmp_path / "cycles_original"
    processed_root = tmp_path / "cycles"
    original_root.mkdir()
    processed_root.mkdir()
    timestamps = pd.date_range("2026-07-14 10:00:00", periods=5, freq="10s")
    original = pd.DataFrame(
        {
            "experiment_id": ["exp"] * 5,
            "cycle_id": ["cycle_001"] * 5,
            "timestamp": timestamps,
            "water_out_temperature": [0.0, 1.0, 8.0, 9.0, 9.0],
            "water_temperature_setpoint": [10.0] * 5,
            "defrost_active": [False, False, False, True, True],
            "cycle_stage": ["recovery"] * 5,
        }
    )
    original.to_csv(original_root / "frost_cycle_000001.csv", index=False)
    original.assign(cycle_stage=original["cycle_stage"]).to_parquet(
        processed_root / "frost_cycle_000001.parquet", index=False
    )
    catalog = {
        "cycles": [
            {
                "cycle_name": "frost_cycle_000001",
                "boundaries": {
                    "heating_start": timestamps[0].isoformat(),
                    "stable_heating_start": None,
                    "defrost_start": timestamps[3].isoformat(),
                    "defrost_end": None,
                },
                "assets": {
                    "parquet": "cycles/frost_cycle_000001.parquet",
                    "original_csv": "cycles_original/frost_cycle_000001.csv",
                },
            }
        ]
    }

    changed = apply_recovery_edit(tmp_path, catalog, mode="ts-minus")

    assert changed == {"frost_cycle_000001"}
    updated = pd.read_csv(original_root / "frost_cycle_000001.csv")
    assert updated.loc[2, "cycle_stage"] == "frost_development"


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


def test_image_scanner_rejects_duplicate_source_camera_roles(tmp_path: Path) -> None:
    from frost_analysis.dataset_images import scan_cycle_images

    cycle_root = tmp_path / "images" / "frost_cycle_000001"
    (cycle_root / "camera01__front").mkdir(parents=True)
    (cycle_root / "camera01__top").mkdir()
    metadata = pd.DataFrame(
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
    )

    with pytest.raises(ValueError, match="source camera"):
        scan_cycle_images(tmp_path, "frost_cycle_000001", metadata)


def test_dataset_identity_validator_rejects_mismatched_cycle_file() -> None:
    from frost_analysis.dataset_validation import _validate_cycle_identity

    frame = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000002"],
            "cycle_uid": ["exp_20260714::cycle_001"],
            "experiment_id": ["exp_20260714"],
            "cycle_id": ["cycle_001"],
        }
    )

    with pytest.raises(ValueError, match="cycle_name"):
        _validate_cycle_identity(
            frame,
            {
                "cycle_name": "frost_cycle_000001",
                "cycle_uid": "exp_20260714::cycle_001",
                "experiment_id": "exp_20260714",
                "cycle_id": "cycle_001",
            },
            "processed",
        )


def test_sensor_coverage_keeps_required_channels_in_the_mask() -> None:
    from frost_analysis.dataset import _sensor_coverage_intervals

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
    from frost_analysis.dataset_io import write_atomic_json

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
    write_atomic_json(
        {
            "dataset_schema_version": 3,
            "dataset_id": "frost_cycle_dataset",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "experiments": [],
        },
        tmp_path / "dataset_manifest.json",
    )
    write_atomic_json(
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
    write_atomic_json({"fields": [], "channels": []}, tmp_path / "channel_registry.json")
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
