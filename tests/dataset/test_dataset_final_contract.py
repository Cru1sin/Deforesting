from __future__ import annotations

import json
import shutil
import subprocess
from collections import namedtuple
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from dataloader.builder.config import Config, ProcessSettings
from dataloader.operations import assign_final_cycle_names_by_time


def _allow_image_download(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataloader import cloud_images

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        cloud_images.shutil,
        "disk_usage",
        lambda _path: usage(200 * 1024**3, 100 * 1024**3, 100 * 1024**3),
    )


def test_collect_cycle_images_uses_cycle_window_without_sensor_match(
    tmp_path: Path,
) -> None:
    from dataloader.images import collect_cycle_images

    camera = tmp_path / "front_center"
    camera.mkdir()
    before = camera / "20260714095959999.jpg"
    recovery = camera / "20260714100001000.jpg"
    frost = camera / "20260714100500000.jpg"
    preparation = camera / "20260714100830000.jpg"
    defrost = camera / "20260714101000000.jpg"
    at_end = camera / "20260714101500000.jpg"
    for path in (before, recovery, frost, preparation, defrost, at_end):
        path.write_bytes(b"image")

    records = collect_cycle_images(
        [before, recovery, frost, preparation, defrost, at_end],
        input_dir=tmp_path,
        cycles=[
            {
                "cycle_name": "frost_cycle_000001",
                "start_time": pd.Timestamp("2026-07-14 10:00:00"),
                "end_time": pd.Timestamp("2026-07-14 10:15:00"),
                "stable_heating_start": pd.Timestamp("2026-07-14 10:02:00"),
                "defrost_preparation_start": pd.Timestamp("2026-07-14 10:08:00"),
                "defrost_start": pd.Timestamp("2026-07-14 10:09:00"),
                "defrost_end": pd.Timestamp("2026-07-14 10:12:00"),
            }
        ],
    )

    assert [record["file_name"] for record in records] == [
        recovery.name,
        frost.name,
        preparation.name,
        defrost.name,
    ]
    assert [record["cycle_stage"] for record in records] == [
        "recovery",
        "frost_development",
        "defrost_preparation",
        "defrost",
    ]


def test_image_metadata_has_no_sensor_alignment_fields() -> None:
    from dataloader.images import image_metadata_frame

    metadata = image_metadata_frame(
        [
            {
                "cycle_name": "frost_cycle_000001",
                "camera_role": "front_center",
                "file_name": "20260714100001000.jpg",
                "frame_index": 1,
                "image_time": pd.Timestamp("2026-07-14 10:00:01"),
                "cycle_stage": "recovery",
            }
        ]
    )

    assert metadata.columns.tolist() == [
        "cycle_name",
        "camera_role",
        "file_name",
        "frame_index",
        "image_time",
        "cycle_stage",
    ]


def test_materialize_cycle_images_prefers_and_preserves_local_images(
    tmp_path: Path,
) -> None:
    from dataloader.cloud_images import materialize_cycle_images

    cycle_name = "frost_cycle_000020"
    local = tmp_path / "dataset" / "images" / cycle_name
    local.mkdir(parents=True)
    (local / "local.jpg").write_bytes(b"local")
    cloud = tmp_path / "cloud"
    cloud.mkdir()
    (cloud / f"{cycle_name}.zip").write_bytes(b"must not be read")

    with materialize_cycle_images(
        tmp_path / "dataset", cycle_name, fetch_cloud=True, cloud_root=cloud
    ) as available:
        assert available == local
        assert (available / "local.jpg").read_bytes() == b"local"

    assert (local / "local.jpg").read_bytes() == b"local"


def test_materialize_cycle_images_treats_missing_cloud_zip_as_no_images(
    tmp_path: Path,
) -> None:
    from dataloader.cloud_images import materialize_cycle_images

    cycle_name = "frost_cycle_000020"
    cloud = tmp_path / "cloud"
    cloud.mkdir()

    with materialize_cycle_images(
        tmp_path / "dataset", cycle_name, fetch_cloud=True, cloud_root=cloud
    ) as available:
        assert not available.exists()


def test_materialize_cycle_images_copies_extracts_and_preserves_local_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zipfile import ZipFile

    from dataloader.cloud_images import materialize_cycle_images

    cycle_name = "frost_cycle_000020"
    dataset = tmp_path / "dataset"
    cloud = tmp_path / "OneDrive" / "images"
    cloud.mkdir(parents=True)
    archive = cloud / f"{cycle_name}.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(f"{cycle_name}/front_center/frame.jpg", b"rgb")
    cloud_bytes = archive.read_bytes()
    _allow_image_download(monkeypatch)

    with materialize_cycle_images(
        dataset, cycle_name, fetch_cloud=True, cloud_root=cloud
    ) as available:
        assert available == dataset / "images" / cycle_name
        assert (available / "front_center" / "frame.jpg").read_bytes() == b"rgb"
        assert archive.read_bytes() == cloud_bytes

    assert (dataset / "images" / cycle_name / "front_center" / "frame.jpg").read_bytes() == b"rgb"
    assert archive.read_bytes() == cloud_bytes


def test_materialize_cycle_images_cleans_only_fresh_download_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zipfile import ZipFile

    from dataloader import cloud_images

    cycle_name = "frost_cycle_000021"
    dataset = tmp_path / "dataset"
    cloud = tmp_path / "cloud"
    cloud.mkdir()
    archive = cloud / f"{cycle_name}.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(f"{cycle_name}/front/frame.jpg", b"rgb")
    _allow_image_download(monkeypatch)

    with cloud_images.materialize_cycle_images(
        dataset,
        cycle_name,
        fetch_cloud=True,
        cloud_root=cloud,
        cleanup_downloaded=True,
    ) as available:
        assert (available / "front" / "frame.jpg").is_file()

    assert not (dataset / "images" / cycle_name).exists()
    assert archive.is_file()


def test_materialize_cycle_images_keeps_fresh_download_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zipfile import ZipFile

    from dataloader import cloud_images

    cycle_name = "frost_cycle_000022"
    dataset = tmp_path / "dataset"
    cloud = tmp_path / "cloud"
    cloud.mkdir()
    with ZipFile(cloud / f"{cycle_name}.zip", "w") as bundle:
        bundle.writestr(f"{cycle_name}/front/frame.jpg", b"rgb")
    _allow_image_download(monkeypatch)

    with (
        pytest.raises(RuntimeError, match="shard failed"),
        cloud_images.materialize_cycle_images(
            dataset,
            cycle_name,
            fetch_cloud=True,
            cloud_root=cloud,
            cleanup_downloaded=True,
        ) as available,
    ):
        assert available.is_dir()
        raise RuntimeError("shard failed")

    assert (dataset / "images" / cycle_name / "front" / "frame.jpg").is_file()


def test_materialize_cycle_images_never_cleans_preexisting_local_directory(
    tmp_path: Path,
) -> None:
    from dataloader.cloud_images import materialize_cycle_images

    cycle_name = "frost_cycle_000023"
    local = tmp_path / "dataset" / "images" / cycle_name
    local.mkdir(parents=True)
    (local / "frame.jpg").write_bytes(b"local")

    with materialize_cycle_images(
        tmp_path / "dataset",
        cycle_name,
        cleanup_downloaded=True,
    ) as available:
        assert available == local

    assert (local / "frame.jpg").read_bytes() == b"local"


def test_materialize_cycle_images_stops_before_crossing_free_space_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections import namedtuple
    from zipfile import ZipFile

    from dataloader import cloud_images

    cycle_name = "frost_cycle_000020"
    dataset = tmp_path / "dataset"
    cloud = tmp_path / "cloud"
    cloud.mkdir()
    with ZipFile(cloud / f"{cycle_name}.zip", "w") as bundle:
        bundle.writestr(f"{cycle_name}/front/frame.jpg", b"rgb")
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        cloud_images.shutil,
        "disk_usage",
        lambda _path: usage(100 * 1024**3, 51 * 1024**3, 49 * 1024**3),
    )

    with (
        pytest.raises(OSError, match="50 GiB safety floor"),
        cloud_images.materialize_cycle_images(
            dataset, cycle_name, fetch_cloud=True, cloud_root=cloud
        ),
    ):
        pass


def test_materialize_cycle_images_uses_custom_free_space_floor_for_both_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zipfile import ZipFile

    from dataloader import cloud_images

    cycle_name = "frost_cycle_000020"
    dataset = tmp_path / "dataset"
    cloud = tmp_path / "cloud"
    cloud.mkdir()
    with ZipFile(cloud / f"{cycle_name}.zip", "w") as bundle:
        bundle.writestr(f"{cycle_name}/front/frame.jpg", b"rgb")
    usage = namedtuple("usage", "total used free")
    free = iter((6 * 1024**3, 4 * 1024**3))
    monkeypatch.setattr(
        cloud_images.shutil,
        "disk_usage",
        lambda _path: usage(10 * 1024**3, 0, next(free)),
    )

    with (
        pytest.raises(
            OSError, match=r"extracting frost_cycle_000020\.zip.*5 GiB safety floor"
        ),
        cloud_images.materialize_cycle_images(
            dataset,
            cycle_name,
            fetch_cloud=True,
            cloud_root=cloud,
            minimum_free_gib=5,
        ),
    ):
        pass


def test_materialize_cycle_images_downloads_default_cloud_zip_with_rclone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zipfile import ZipFile

    from dataloader import cloud_images

    cycle_name = "frost_cycle_000020"
    monkeypatch.setattr(cloud_images, "DEFAULT_CLOUD_IMAGES_REMOTE", "remote:images")
    calls: list[tuple[list[str], dict[str, str]]] = []

    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "RCLONE_HTTP_PROXY",
    ):
        monkeypatch.setenv(name, "http://127.0.0.1:7890")
    monkeypatch.setenv("NO_PROXY", "inherited.example")
    monkeypatch.setenv("no_proxy", "inherited.example")

    def fake_run(
        command: list[str], **options: object
    ) -> subprocess.CompletedProcess[str]:
        env = cast(dict[str, str], options["env"])
        calls.append((command, env))
        if command[1] == "lsf":
            return subprocess.CompletedProcess(command, 0, stdout="123\n")
        assert command[1] == "copyto"
        with ZipFile(Path(command[3]), "w") as bundle:
            bundle.writestr(f"{cycle_name}/front_center/frame.jpg", b"rgb")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cloud_images.subprocess, "run", fake_run)
    _allow_image_download(monkeypatch)

    with cloud_images.materialize_cycle_images(
        tmp_path / "dataset", cycle_name, fetch_cloud=True
    ) as available:
        assert (available / "front_center" / "frame.jpg").read_bytes() == b"rgb"

    assert [command[1] for command, _env in calls] == ["lsf", "copyto"]
    assert calls[0][0][:3] == [
        "rclone",
        "lsf",
        f"remote:images/{cycle_name}.zip",
    ]
    assert calls[1][0][:3] == [
        "rclone",
        "copyto",
        f"remote:images/{cycle_name}.zip",
    ]
    assert calls[1][0][calls[1][0].index("--multi-thread-streams") + 1] == "8"
    for command, env in calls:
        assert command[command.index("--http-proxy") + 1] == ""
        assert (
            not {
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
                "RCLONE_HTTP_PROXY",
            }
            & env.keys()
        )
        assert env["NO_PROXY"] == "*"
        assert env["no_proxy"] == "*"
    assert (
        tmp_path / "dataset" / "images" / cycle_name / "front_center" / "frame.jpg"
    ).read_bytes() == b"rgb"


def test_materialize_cycle_images_treats_missing_default_remote_as_no_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataloader import cloud_images

    calls: list[list[str]] = []

    def fake_run(command: list[str], **_options: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="")

    monkeypatch.setattr(cloud_images.subprocess, "run", fake_run)

    with cloud_images.materialize_cycle_images(
        tmp_path / "dataset", "frost_cycle_000099", fetch_cloud=True
    ) as available:
        assert not available.exists()

    assert [command[1] for command in calls] == ["lsf"]


def test_materialize_cycle_image_members_reads_only_requested_remote_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zipfile import ZIP_DEFLATED, ZipFile

    from dataloader import cloud_images

    cycle_name = "frost_cycle_000006"
    archive_path = tmp_path / f"{cycle_name}.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as bundle:
        bundle.writestr(f"{cycle_name}/front/rb.jpg", b"rb-image")
        bundle.writestr(f"{cycle_name}/front/unused.jpg", b"unused-image")
    archive = archive_path.read_bytes()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_options: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        if command[1] == "lsf":
            return subprocess.CompletedProcess(command, 0, stdout=f"{len(archive)}\n".encode())
        assert command[1] == "cat"
        offset = int(command[command.index("--offset") + 1])
        count = int(command[command.index("--count") + 1])
        return subprocess.CompletedProcess(command, 0, stdout=archive[offset : offset + count])

    monkeypatch.setattr(cloud_images, "DEFAULT_CLOUD_IMAGES_REMOTE", "remote:images")
    monkeypatch.setattr(cloud_images.subprocess, "run", fake_run)

    with cloud_images.materialize_cycle_image_members(
        tmp_path / "dataset",
        cycle_name,
        ["rb.jpg"],
        fetch_cloud=True,
        minimum_free_gib=0,
    ) as available:
        assert (available / "front" / "rb.jpg").read_bytes() == b"rb-image"
        assert not (available / "front" / "unused.jpg").exists()

    local = tmp_path / "dataset" / "images" / cycle_name
    assert (local / "front" / "rb.jpg").read_bytes() == b"rb-image"

    with cloud_images.materialize_cycle_image_members(
        tmp_path / "dataset",
        cycle_name,
        ["rb.jpg", "unused.jpg"],
        fetch_cloud=True,
        minimum_free_gib=0,
    ) as available:
        assert available == local
        assert (available / "front" / "unused.jpg").read_bytes() == b"unused-image"

    assert [command[1] for command in calls] == [
        "lsf",
        "cat",
        "cat",
        "lsf",
        "cat",
        "cat",
    ]
    assert not any(command[1] == "copyto" for command in calls)
    assert all("--count" in command for command in calls if command[1] == "cat")


def _write_renderable_dataset(dataset_dir: Path) -> tuple[str, dict[str, str]]:
    from dataloader.files import write_json

    cycle_name = "frost_cycle_000001"
    assets = {
        "parquet": f"cycles/{cycle_name}.parquet",
        "csv": f"cycles/{cycle_name}.csv",
        "original_csv": f"cycles_original/{cycle_name}.csv",
        "publication": f"cycles/{cycle_name}.png",
        "rgb_coverage": f"cycles/{cycle_name}_rgb_coverage.png",
        "rgb_panel": f"cycles/{cycle_name}_rgb_panel.png",
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
    (dataset_dir / assets["rgb_panel"]).write_bytes(b"panel")
    metadata_columns = [
        "cycle_name",
        "camera_role",
        "file_name",
        "frame_index",
        "image_time",
        "cycle_stage",
    ]
    pd.DataFrame(columns=metadata_columns).to_parquet(
        dataset_dir / "image_metadata.parquet", index=False
    )
    write_json(
        {
            "dataset_schema_version": 3,
            "dataset_id": "frost_cycle_dataset",
            "experiments": [
                {
                    "experiment_id": "exp",
                    "experiment_date": "2026-07-14",
                    "camera_roles": [],
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from types import SimpleNamespace

    import dataloader.operations as dataset_module
    from dataloader import metadata as dataset_metadata
    from dataloader.builder.build import ExperimentBuild
    from dataloader.operations import add_dataset

    input_dir = tmp_path / "0714"
    input_dir.mkdir()
    camera_dir = input_dir / "front_center"
    camera_dir.mkdir()
    image_path = camera_dir / "20260714100005000.jpg"
    from PIL import Image

    Image.new("RGB", (4, 4)).save(image_path)
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
    build = ExperimentBuild(
        input_dir=input_dir,
        config=SimpleNamespace(
            experiment_id="exp_0714",
            experiment_date="2026-07-14",
            project_root=tmp_path,
            input_dir=input_dir,
            sensor_globs=("*.xls",),
            image_extensions=(".jpg",),
            cycles=SimpleNamespace(stable_heating_seconds=180),
            process=SimpleNamespace(
                resample_interval_seconds=10,
                baseline=SimpleNamespace(baseline_seconds=60),
            ),
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
        original=frame.assign(p1__undeclared_point=["raw-a", "raw-b"]),
    )
    monkeypatch.setattr(dataset_module, "_validate_date_input", lambda *_args: None)
    monkeypatch.setattr(
        dataset_module,
        "_load_config_for_input",
        lambda *_args: build.config,
    )
    monkeypatch.setattr(dataset_module, "build_experiment", lambda *_args: build)
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
    assert set(manifest) == {
        "dataset_schema_version",
        "dataset_id",
        "experiments",
        "images_root",
    }
    assert manifest["images_root"] == "images"
    assert set(manifest["experiments"][0]) == {"experiment_id", "experiment_date"}
    processed = pd.read_parquet(dataset_dir / "cycles/frost_cycle_000001.parquet")
    assert not {
        "dataset_id",
        "dataset_schema_version",
        "dataset_cycle_index",
    } & set(processed)
    original = pd.read_csv(dataset_dir / "cycles_original/frost_cycle_000001.csv")
    assert original["p1__undeclared_point"].tolist() == ["raw-a", "raw-b"]
    metadata = pd.read_parquet(dataset_dir / "image_metadata.parquet")
    assert metadata.columns.tolist() == [
        "cycle_name",
        "camera_role",
        "file_name",
        "frame_index",
        "image_time",
        "cycle_stage",
    ]
    assert metadata[["cycle_name", "camera_role", "file_name"]].to_dict("records") == [
        {
            "cycle_name": "frost_cycle_000001",
            "camera_role": "front_center",
            "file_name": image_path.name,
        }
    ]
    catalog = json.loads((dataset_dir / "cycle_catalog.json").read_text())
    assert catalog["cycles"][0]["image"] == {"image_count": 1}
    assert "[add] rendering cycles: 1/1 frost_cycle_000001" in capsys.readouterr().out


def test_load_config_for_input_gets_date_from_xls_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataloader.builder.config as config_module
    from dataloader.operations import _load_config_for_input

    original = Config(
        project_root=tmp_path,
        experiment_id="exp_test",
        experiment_date="2027-01-02",
        input_dir=tmp_path / "old-input",
        sensor_globs=("*.xls",),
        image_extensions=(".jpg",),
        timestamp_column="时间",
        expected_sensor_interval_seconds=1,
        image_match_tolerance_seconds=2,
        edf_pair_tolerance_seconds=1.0,
        process=ProcessSettings(resample_interval_seconds=10),
    )
    new_input = tmp_path / "input"
    new_input.mkdir()
    for name in (
        "2027-01-02 09-00-00参数1.xls",
        "2027-01-02 09-00-00参数2.xls",
        "2027-01-02 09-00-00参数5.xls",
        "2027-01-03_09-00-00-CC_AllSensors.edf",
    ):
        (new_input / name).touch()
    calls: list[dict[str, object]] = []

    def fake_load(**kwargs: object) -> Config:
        calls.append(kwargs)
        return original

    monkeypatch.setattr(config_module, "load_config", fake_load)

    loaded = _load_config_for_input(new_input, tmp_path)

    assert loaded is original
    assert calls == [
        {
            "project_root": tmp_path,
            "experiment_date": "2027-01-02",
            "input_dir": new_input,
        }
    ]


def test_load_config_for_input_rejects_multiple_xls_dates(tmp_path: Path) -> None:
    from dataloader.operations import _load_config_for_input

    (tmp_path / "2026-07-24参数1.xls").touch()
    (tmp_path / "2026-07-25参数2.xls").touch()

    with pytest.raises(ValueError, match="one shared date"):
        _load_config_for_input(tmp_path, tmp_path)


def test_build_experiment_prints_processing_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from types import SimpleNamespace

    from dataloader.builder import channels, prepare, process
    from dataloader.builder import prepared as validation
    from dataloader.builder.build import build_experiment

    frame = pd.DataFrame({"cycle_id": ["cycle_001", "cycle_002"]})
    summary = pd.DataFrame({"cycle_id": ["cycle_001", "cycle_002"]})
    monkeypatch.setattr(channels, "load_channels", lambda: {})
    monkeypatch.setattr(prepare, "prepare", lambda _config, _channels: (frame, summary))
    monkeypatch.setattr(process, "process", lambda *_args: (frame, summary))
    monkeypatch.setattr(prepare, "prepare_original", lambda _config, _frame: frame)
    monkeypatch.setattr(validation, "validate_prepared", lambda *_args: None)
    monkeypatch.setattr(validation, "validate_processed", lambda *_args: None)

    build_experiment(tmp_path, SimpleNamespace())

    assert capsys.readouterr().out.splitlines() == [
        "[add] prepare sensors",
        "[add] validate prepared",
        "[add] process cycles",
        "[add] validate processed",
        "[add] preserve original sensors",
        "[add] cycles=2",
    ]


def test_metadata_readers_allow_extra_review_fields(tmp_path: Path) -> None:
    from dataloader.files import write_json
    from dataloader.metadata import read_catalog, read_manifest, write_catalog

    write_json(
        {
            "dataset_schema_version": 3,
            "dataset_id": "frost_cycle_dataset",
            "experiments": [
                {
                    "experiment_id": "exp",
                    "experiment_date": "2026-07-14",
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
    from dataloader import images as dataset_images
    from dataloader.operations import review_cycle
    from plots import publication as visualization

    cycle_name, _ = _write_renderable_dataset(tmp_path)
    manifest_before = (tmp_path / "dataset_manifest.json").read_bytes()
    render_calls: list[str] = []

    def fail_scan(*args: object, **kwargs: object) -> pd.DataFrame:
        pytest.fail("review-cycle must not scan current image directories")

    monkeypatch.setattr(dataset_images, "scan_cycle_images", fail_scan)
    monkeypatch.setattr(
        visualization,
        "render_cycle_publication",
        lambda _frame, _record, _path, **_kwargs: render_calls.append(cycle_name),
    )

    review_cycle(
        tmp_path,
        cycle_name,
        status="valid",
        reason="reviewed",
        rgb_frost="invalid",
        rgb_defrost="valid",
    )

    catalog = json.loads((tmp_path / "cycle_catalog.json").read_text())
    assert catalog["cycles"][0]["status"] == "valid"
    assert catalog["cycles"][0]["status_reason"] == "reviewed"
    assert catalog["cycles"][0]["rgb_frost_status"] == "invalid"
    assert catalog["cycles"][0]["rgb_defrost_status"] == "valid"
    assert render_calls == [cycle_name]
    assert (tmp_path / "dataset_manifest.json").read_bytes() == manifest_before


def test_review_cycle_rejects_nonbinary_status(tmp_path: Path) -> None:
    from dataloader.operations import review_cycle

    with pytest.raises(ValueError, match="invalid Dataset status"):
        review_cycle(tmp_path, "frost_cycle_000001", status="incomplete")


def test_validate_dataset_rejects_nonbinary_status(tmp_path: Path) -> None:
    from dataloader.check import validate_dataset

    _write_renderable_dataset(tmp_path)
    catalog = json.loads((tmp_path / "cycle_catalog.json").read_text())
    catalog["cycles"][0]["status"] = "incomplete"
    (tmp_path / "cycle_catalog.json").write_text(json.dumps(catalog))

    with pytest.raises(ValueError, match="status must be valid or invalid"):
        validate_dataset(tmp_path)


def test_render_only_draws_without_writing_catalog_or_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataloader import files as dataset_io
    from dataloader.operations import render_dataset
    from plots import publication as visualization

    cycle_name, _ = _write_renderable_dataset(tmp_path)
    catalog_before = (tmp_path / "cycle_catalog.json").read_bytes()
    manifest_before = (tmp_path / "dataset_manifest.json").read_bytes()
    writes: list[Path] = []
    publication_calls: list[str] = []
    panel_calls: list[str] = []

    monkeypatch.setattr(
        dataset_io,
        "write_json",
        lambda _data, path: writes.append(path),
    )
    monkeypatch.setattr(
        visualization,
        "render_cycle_publication",
        lambda _frame, _record, _path, **_kwargs: publication_calls.append(cycle_name),
    )
    monkeypatch.setattr(
        visualization,
        "render_rgb_panel",
        lambda _record, _frame, _images, _intervals, _roles, _path: panel_calls.append(
            cycle_name
        ),
    )

    render_dataset(tmp_path, cycle_name, publication=True, panel=True)

    assert writes == []
    assert publication_calls == [cycle_name]
    assert panel_calls == [cycle_name]
    assert (tmp_path / "cycle_catalog.json").read_bytes() == catalog_before
    assert (tmp_path / "dataset_manifest.json").read_bytes() == manifest_before


def test_render_can_explicitly_fetch_cloud_cycle_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zipfile import ZipFile

    from dataloader import cloud_images
    from dataloader.operations import render_dataset
    from plots import publication as visualization

    cycle_name, _ = _write_renderable_dataset(tmp_path)
    shutil.rmtree(tmp_path / "images" / cycle_name)
    file_name = "20260714100000000.jpg"
    metadata = pd.DataFrame(
        [
            {
                "cycle_name": cycle_name,
                "camera_role": "front_center",
                "file_name": file_name,
                "frame_index": 1,
                "image_time": pd.Timestamp("2026-07-14 10:00:00"),
                "cycle_stage": "recovery",
            }
        ]
    )
    metadata.to_parquet(tmp_path / "image_metadata.parquet", index=False)
    cloud = tmp_path / "OneDrive" / "images"
    cloud.mkdir(parents=True)
    archive = cloud / f"{cycle_name}.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(f"{cycle_name}/front_center/{file_name}", b"rgb")

    def fake_rclone(
        command: list[str], **_options: object
    ) -> subprocess.CompletedProcess[str]:
        if command[1] == "lsf":
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{archive.stat().st_size}\n"
            )
        shutil.copyfile(archive, Path(command[3]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cloud_images.subprocess, "run", fake_rclone)
    _allow_image_download(monkeypatch)
    seen: list[list[Path]] = []
    monkeypatch.setattr(
        visualization,
        "render_rgb_panel",
        lambda _record, _frame, images, _intervals, _roles, _path: seen.append(
            images["path"].tolist()
        ),
    )

    render_dataset(tmp_path, cycle_name, publication=False, panel=True)
    assert seen == [[]]
    assert not (tmp_path / "images" / cycle_name).exists()

    render_dataset(
        tmp_path,
        cycle_name,
        publication=False,
        panel=True,
        fetch_cloud_images=True,
    )

    assert seen[1] == [tmp_path / "images" / cycle_name / "front_center" / file_name]
    assert (tmp_path / "images" / cycle_name / "front_center" / file_name).is_file()
    assert archive.is_file()


def test_cycle_assets_has_publication_and_panel_without_coverage() -> None:
    from dataloader.metadata import cycle_assets

    assets = cycle_assets("frost_cycle_000001")

    assert assets["rgb_panel"] == "cycles/frost_cycle_000001_rgb_panel.png"
    assert "rgb_coverage" not in assets


def test_validate_scientific_schema_ignores_catalog_counts(
    tmp_path: Path,
) -> None:
    from dataloader.check import validate_dataset
    from dataloader.files import write_json
    from dataloader.operations import make_cycle_uid

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
        "cycle_name",
        "camera_role",
        "file_name",
        "frame_index",
        "image_time",
        "cycle_stage",
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
                    "camera_roles": [],
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


def test_cli_default_dataset_path_is_local(monkeypatch: object) -> None:
    import main_data

    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        main_data,
        "edit_dataset",
        lambda path, **kwargs: calls.append(("edit", path)) or path,
    )

    assert main_data.main(["edit", "--baseline-seconds", "60"]) == 0
    assert calls == [("edit", Path("dataset"))]


def test_cli_edit_can_skip_rgb_panel_rendering(monkeypatch: object, tmp_path: Path) -> None:
    import main_data

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        main_data, "edit_dataset", lambda _path, **kwargs: calls.append(kwargs) or tmp_path
    )

    assert (
        main_data.main(
            [
                "edit",
                "--dataset",
                str(tmp_path),
                "--recovery-end-by",
                "ts-minus",
                "--skip-rgb-panels",
            ]
        )
        == 0
    )
    assert calls[0]["render_rgb_panels"] is False


def test_recovery_transform_recomputes_stage_coordinates_features_and_images() -> None:
    from dataloader.edit import apply_recovery

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
    )
    metadata = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000001"] * 2,
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
    assert not any(
        "__lag_" in column or "__delta_" in column or "__rolling_mean_" in column
        for column in new_processed.columns
    )
    assert new_metadata["cycle_stage"].tolist() == ["recovery", "frost_development"]


def test_recovery_and_baseline_share_single_scientific_entrypoints() -> None:
    from dataloader.builder.baseline import apply_fixed_baseline
    from dataloader.builder.cycles import resolve_stable_heating_start

    timestamps = pd.date_range("2026-07-14 10:00:00", periods=6, freq="10s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cycle_stage": ["recovery"] + ["frost_development"] * 5,
            "signal": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "signal__imputed": [False] * 6,
            "water_out_temperature": [30.0, 31.0, 33.5, 35.0, 35.5, 36.0],
            "water_temperature_setpoint": [36.0] * 6,
        }
    )

    criterion_start = resolve_stable_heating_start(
        frame,
        timestamps[0],
        timestamps[-1] + pd.Timedelta(seconds=10),
        {"stable_heating_seconds": 20},
    )
    seconds_start = resolve_stable_heating_start(
        frame,
        timestamps[0],
        timestamps[-1] + pd.Timedelta(seconds=10),
        {"stable_heating_seconds": 20},
        mode="seconds",
        seconds=20,
    )
    seconds_at_defrost = resolve_stable_heating_start(
        frame,
        timestamps[0],
        timestamps[2],
        {"stable_heating_seconds": 20},
        mode="seconds",
        seconds=20,
    )
    baselined, unavailable = apply_fixed_baseline(
        frame[["timestamp", "cycle_stage", "signal", "signal__imputed"]].copy(),
        ["signal"],
        start=timestamps[1],
        end=timestamps[4],
        minimum_observed_coverage=0.8,
        stage="frost_development",
    )

    assert criterion_start == timestamps[3]
    assert seconds_start == timestamps[2]
    assert seconds_at_defrost == timestamps[2]
    assert unavailable == []
    assert baselined["signal__baseline"].dropna().unique().tolist() == [12.0]
    assert baselined.loc[4, "signal__baseline_residual"] == pytest.approx(2.0)


def test_partial_stage_context_keeps_known_defrost_without_temperature_evidence() -> None:
    from dataloader.builder.cycles import _partial_stage_context

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
    from dataloader.images import _sensor_coverage_intervals

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
    import dataloader.check as validation
    from dataloader.files import write_json
    from dataloader.operations import refresh_dataset
    from plots import publication as visualization

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
            "experiments": [
                {
                    "experiment_id": "exp",
                    "experiment_date": "2026-07-14",
                    "camera_roles": [],
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
                        "rgb_panel": "cycles/frost_cycle_000001_rgb_panel.png",
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
            "cycle_name",
            "camera_role",
            "file_name",
            "frame_index",
            "image_time",
            "cycle_stage",
        ]
    ).to_parquet(tmp_path / "image_metadata.parquet", index=False)

    monkeypatch.setattr(
        visualization,
        "render_cycle_publication",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(visualization, "render_rgb_panel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(validation, "validate_dataset", lambda _path: None)

    manifest_before = (tmp_path / "dataset_manifest.json").read_bytes()
    refresh_dataset(tmp_path, "figures")

    assert (tmp_path / "dataset_manifest.json").read_bytes() == manifest_before


def test_dataset_add_append_edit_refresh_loader_validate_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import dataloader.operations as dataset_module
    from dataloader.builder.build import ExperimentBuild
    from dataloader.check import validate_dataset
    from dataloader.loader import DatasetLoader
    from dataloader.operations import add_dataset

    def make_build(input_dir: Path) -> ExperimentBuild:
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
            input_dir=input_dir,
            sensor_globs=("*.xls",),
            image_extensions=(".jpg",),
            cycles=SimpleNamespace(stable_heating_seconds=20),
            process=SimpleNamespace(
                resample_interval_seconds=10,
                baseline=SimpleNamespace(baseline_seconds=60),
            ),
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
        return ExperimentBuild(
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
    (first_input / "2026-07-14参数1.xls").touch()
    (second_input / "2026-07-15参数1.xls").touch()
    monkeypatch.setattr(dataset_module, "_validate_date_input", lambda *_args: None)
    monkeypatch.setattr(
        dataset_module,
        "build_experiment",
        lambda input_dir, _project_root: make_build(input_dir),
    )
    dataset_dir = tmp_path / "dataset"

    add_dataset(first_input, dataset_dir)
    initial_registry = json.loads((dataset_dir / "channel_registry.json").read_text())
    initial_catalog = json.loads((dataset_dir / "cycle_catalog.json").read_text())
    assert initial_registry["baseline_managed"] is False
    assert initial_registry["recovery_edit"]["managed"] is False
    assert all("asset_sha256" not in record for record in initial_catalog["cycles"])
    add_dataset(second_input, dataset_dir)
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    assert all("source_fingerprint" not in item for item in manifest["experiments"])

    from dataloader import images as dataset_images
    from dataloader.operations import edit_dataset, refresh_dataset

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
    assert "processing_settings" not in edited_registry
    assert edited_registry["baseline_managed"] is True
    assert edited_registry["recovery_edit"]["managed"] is True
    third_input = tmp_path / "0716"
    third_input.mkdir()
    (third_input / "2026-07-16参数1.xls").touch()
    add_dataset(third_input, dataset_dir)
    refresh_dataset(dataset_dir, "all")
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
    import dataloader.files as dataset_io

    assert hasattr(dataset_io, "write_json")
    assert hasattr(dataset_io, "write_csv")
    assert hasattr(dataset_io, "write_parquet")
    assert not hasattr(dataset_io, "mutate_dataset")
    assert not hasattr(dataset_io, "publish_with_rollback")


def test_dataset_rebuild_is_not_a_public_operation(capsys: pytest.CaptureFixture[str]) -> None:
    import dataloader.operations as dataset_module
    from main_data import main

    with pytest.raises(SystemExit):
        main(["--help"])

    assert "rebuild" not in capsys.readouterr().out
    assert not hasattr(dataset_module, "rebuild_dataset")


def test_dataset_replace_is_a_public_one_input_operation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import dataloader.operations as dataset_module
    from main_data import main

    with pytest.raises(SystemExit):
        main(["replace", "--help"])

    help_text = capsys.readouterr().out
    assert "input_dir" in help_text
    assert hasattr(dataset_module, "replace_dataset")


def test_dataset_remove_deletes_only_selected_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import dataloader.check as validation
    from dataloader.operations import remove_dataset

    first_name, _ = _write_renderable_dataset(tmp_path)
    second_name = "frost_cycle_000002"
    first_catalog = json.loads((tmp_path / "cycle_catalog.json").read_text())
    first_catalog["cycles"][0]["status"] = "valid"
    first_catalog["cycles"][0]["status_reason"] = "manual_review"
    second = json.loads(json.dumps(first_catalog["cycles"][0]))
    second.update(
        {
            "cycle_name": second_name,
            "cycle_uid": "exp_20260722::cycle_001",
            "experiment_id": "exp_20260722",
            "experiment_date": "2026-07-22",
            "status": "invalid",
            "status_reason": "wrong_sensor_date",
        }
    )
    second["assets"] = {
        key: value.replace(first_name, second_name)
        for key, value in second["assets"].items()
    }
    first_catalog["cycles"].append(second)
    (tmp_path / "cycle_catalog.json").write_text(json.dumps(first_catalog))

    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    manifest["experiments"].append(
        {"experiment_id": "exp_20260722", "experiment_date": "2026-07-22"}
    )
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest))
    for key, relative in second["assets"].items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if key == "parquet":
            pd.read_parquet(tmp_path / first_catalog["cycles"][0]["assets"][key]).to_parquet(
                path, index=False
            )
        elif key.endswith("csv") or key == "csv":
            path.write_text("timestamp\n2026-07-22T10:00:00\n")
        else:
            path.write_bytes(b"asset")
    image_dir = tmp_path / "images" / second_name / "front"
    image_dir.mkdir(parents=True)
    (image_dir / "20260722100000000.jpg").write_bytes(b"image")
    metadata = pd.read_parquet(tmp_path / "image_metadata.parquet")
    metadata.loc[len(metadata)] = {
        "cycle_name": second_name,
        "camera_role": "front",
        "file_name": "20260722100000000.jpg",
        "frame_index": 1,
        "image_time": pd.Timestamp("2026-07-22 10:00:00"),
        "cycle_stage": "frost_development",
    }
    metadata.to_parquet(tmp_path / "image_metadata.parquet", index=False)
    validated: list[Path] = []
    monkeypatch.setattr(validation, "validate_dataset", lambda path: validated.append(path))

    remove_dataset(tmp_path, "0722")

    result = json.loads((tmp_path / "cycle_catalog.json").read_text())
    assert [record["cycle_name"] for record in result["cycles"]] == [first_name]
    assert result["cycles"][0]["status"] == "valid"
    assert result["cycles"][0]["status_reason"] == "manual_review"
    assert not (tmp_path / "images" / second_name).exists()
    assert all(not (tmp_path / relative).exists() for relative in second["assets"].values())
    assert pd.read_parquet(tmp_path / "image_metadata.parquet").empty
    assert validated == [tmp_path]
    assert "[remove] done" in capsys.readouterr().out


def test_refresh_roles_uses_folder_names_and_preserves_human_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import dataloader.check as validation
    from dataloader.operations import refresh_dataset
    from plots import publication as visualization

    cycle_name, _ = _write_renderable_dataset(tmp_path)
    image_dir = tmp_path / "images" / cycle_name / "front_center"
    image_dir.mkdir()
    file_name = "20260714100000000.jpg"
    (image_dir / file_name).write_bytes(b"image")
    metadata = pd.DataFrame(
        [
            {
                "cycle_name": cycle_name,
                "camera_role": "unverified_camera_05",
                "file_name": file_name,
                "frame_index": 1,
                "image_time": pd.Timestamp("2026-07-14 10:00:00"),
                "cycle_stage": "recovery",
            }
        ]
    )
    metadata.to_parquet(tmp_path / "image_metadata.parquet", index=False)
    catalog = json.loads((tmp_path / "cycle_catalog.json").read_text())
    catalog["cycles"][0].update(
        {
            "status": "valid",
            "status_reason": "manual_review",
            "rgb_frost_status": "invalid",
            "rgb_defrost_status": "not_applicable",
        }
    )
    (tmp_path / "cycle_catalog.json").write_text(json.dumps(catalog))
    monkeypatch.setattr(visualization, "render_cycle_publication", lambda *_a, **_k: None)
    monkeypatch.setattr(visualization, "render_rgb_panel", lambda *_a, **_k: None)
    validated: list[Path] = []
    monkeypatch.setattr(validation, "validate_dataset", lambda path: validated.append(path))

    refresh_dataset(tmp_path, "roles")

    refreshed = pd.read_parquet(tmp_path / "image_metadata.parquet")
    assert refreshed.loc[0, "camera_role"] == "front_center"
    result = json.loads((tmp_path / "cycle_catalog.json").read_text())["cycles"][0]
    assert (result["status"], result["status_reason"]) == ("valid", "manual_review")
    assert result["rgb_frost_status"] == "invalid"
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    assert "camera_roles" not in manifest["experiments"][0]
    assert "rgb_coverage" not in result["assets"]
    assert not (tmp_path / "cycles" / f"{cycle_name}_rgb_coverage.png").exists()
    assert validated == [tmp_path.resolve()]
    assert "unverified_camera_05 -> front_center" in capsys.readouterr().out


def test_refresh_images_rebuilds_metadata_from_current_cycle_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataloader.check as validation
    from dataloader.operations import refresh_dataset
    from plots import publication as visualization

    cycle_name, _ = _write_renderable_dataset(tmp_path)
    image_dir = tmp_path / "images" / cycle_name / "top_center"
    image_dir.mkdir()
    file_name = "20260714100000000.jpg"
    (image_dir / file_name).write_bytes(b"image")
    monkeypatch.setattr(visualization, "render_cycle_publication", lambda *_a, **_k: None)
    monkeypatch.setattr(visualization, "render_rgb_panel", lambda *_a, **_k: None)
    monkeypatch.setattr(validation, "validate_dataset", lambda _path: None)

    refresh_dataset(tmp_path, "images")

    metadata = pd.read_parquet(tmp_path / "image_metadata.parquet")
    assert metadata[["cycle_name", "camera_role", "file_name"]].to_dict("records") == [
        {
            "cycle_name": cycle_name,
            "camera_role": "top_center",
            "file_name": file_name,
        }
    ]
    assert "matched_timestamp" not in metadata
    assert "offset_seconds" not in metadata


def test_refresh_images_refuses_file_outside_its_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataloader.check as validation
    from dataloader.operations import refresh_dataset

    cycle_name, _ = _write_renderable_dataset(tmp_path)
    image_dir = tmp_path / "images" / cycle_name / "front"
    image_dir.mkdir()
    (image_dir / "20260714110000000.jpg").write_bytes(b"image")
    before = (tmp_path / "image_metadata.parquet").read_bytes()
    monkeypatch.setattr(validation, "validate_dataset", lambda _path: None)

    with pytest.raises(ValueError, match="1 images fall outside their cycle"):
        refresh_dataset(tmp_path, "images")

    assert (tmp_path / "image_metadata.parquet").read_bytes() == before


def test_refresh_figures_does_not_rewrite_image_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataloader.check as validation
    from dataloader.operations import refresh_dataset
    from plots import publication as visualization

    _write_renderable_dataset(tmp_path)
    before = (tmp_path / "image_metadata.parquet").read_bytes()
    calls: list[str] = []
    monkeypatch.setattr(
        visualization,
        "render_cycle_publication",
        lambda *_a, **_k: calls.append("publication"),
    )
    monkeypatch.setattr(
        visualization, "render_rgb_panel", lambda *_a, **_k: calls.append("panel")
    )
    monkeypatch.setattr(validation, "validate_dataset", lambda _path: None)

    refresh_dataset(tmp_path, "figures")

    assert (tmp_path / "image_metadata.parquet").read_bytes() == before
    assert calls == ["publication", "panel"]


def test_loader_uses_manifest_images_root_without_metadata_rewrite(tmp_path: Path) -> None:
    from dataloader.loader import DatasetLoader

    dataset_dir = tmp_path / "dataset"
    cycle_name, _ = _write_renderable_dataset(dataset_dir)
    external_root = tmp_path / "OneDrive" / "images"
    image_dir = external_root / cycle_name / "front_center"
    image_dir.mkdir(parents=True)
    file_name = "20260714100000000.jpg"
    (image_dir / file_name).write_bytes(b"image")
    metadata = pd.DataFrame(
        [
            {
                "cycle_name": cycle_name,
                "camera_role": "front_center",
                "file_name": file_name,
                "frame_index": 1,
                "image_time": pd.Timestamp("2026-07-14 10:00:00"),
                "cycle_stage": "recovery",
            }
        ]
    )
    metadata.to_parquet(dataset_dir / "image_metadata.parquet", index=False)
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    manifest["images_root"] = str(external_root)
    (dataset_dir / "dataset_manifest.json").write_text(json.dumps(manifest))
    shutil.rmtree(dataset_dir / "images")

    images = DatasetLoader(dataset_dir).load_cycle_images(cycle_name)

    assert images.loc[0, "path"] == image_dir / file_name


def test_baseline_edit_does_not_require_original_or_image_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataloader.files import write_json
    from dataloader.operations import edit_dataset

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
        },
        tmp_path / "channel_registry.json",
    )
    monkeypatch.setattr(
        "plots.publication.render_cycle_publication",
        lambda *_args, **_kwargs: None,
    )

    edit_dataset(tmp_path, baseline_seconds=60)


def test_loader_uses_camera_directory_as_role(
    tmp_path: Path,
) -> None:
    from dataloader.files import write_json
    from dataloader.loader import DatasetLoader

    cycle_name = "frost_cycle_000001"
    camera_dir = tmp_path / "images" / cycle_name / "front"
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
                    "camera_roles": ["front"],
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
                "cycle_name": cycle_name,
                "camera_role": "front",
                "file_name": "frame_0001.jpg",
                "frame_index": 1,
                "image_time": frame["timestamp"].iloc[0],
                "cycle_stage": "frost_development",
            }
        ]
    ).to_parquet(tmp_path / "image_metadata.parquet", index=False)

    images = DatasetLoader(tmp_path).load_cycle_images(cycle_name)

    assert images["camera_role"].tolist() == ["front"]
    assert images["path"].tolist() == [camera_dir / "frame_0001.jpg"]

    assert camera_dir.is_dir()


def test_rgb_stage_metrics_require_all_expected_roles() -> None:
    from dataloader.images import rgb_stage_metrics

    times = pd.date_range("2026-07-14 10:00:00", periods=5, freq="10s")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "cycle_stage": ["frost_development"] * 5,
        }
    )
    intervals = {
        "front": {"available": [(times[0], times[-1] + pd.Timedelta(seconds=10))]},
        "top": {"available": [(times[0], times[4])]},
    }

    metrics = rgb_stage_metrics(frame, intervals, ("front", "top"))

    assert metrics["rgb_frost_coverage"] == pytest.approx(0.8)
    assert metrics["rgb_frost_auto_status"] == "valid"
    assert metrics["rgb_defrost_coverage"] is None
    assert metrics["rgb_defrost_auto_status"] == "not_applicable"


def test_rgb_stage_metrics_mark_missing_expected_role_invalid() -> None:
    from dataloader.images import rgb_stage_metrics

    times = pd.date_range("2026-07-14 10:00:00", periods=2, freq="10s")
    frame = pd.DataFrame(
        {"timestamp": times, "cycle_stage": ["defrost", "defrost"]}
    )
    metrics = rgb_stage_metrics(
        frame,
        {"front": {"available": [(times[0], times[-1] + pd.Timedelta(seconds=10))]}},
        ("front", "top"),
    )

    assert metrics["rgb_defrost_coverage"] == 0.0
    assert metrics["rgb_defrost_auto_status"] == "invalid"


def test_update_cycle_columns_writes_parquet_and_csv_by_timestamp(tmp_path: Path) -> None:
    from dataloader.operations import update_cycle_columns

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
    import dataloader.operations as dataset_module
    from dataloader.files import write_json
    from dataloader.operations import add_dataset

    input_dir = tmp_path / "0714"
    input_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    write_json(
        {
            "dataset_schema_version": 3,
            "dataset_id": "frost_cycle_dataset",
            "experiments": [
                {
                    "experiment_id": "exp_0714",
                    "experiment_date": "2026-07-14",
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
        "build_experiment",
        lambda *_args: pytest.fail("same experiment must not rerun Prepare/Process"),
    )

    assert (
        add_dataset(input_dir, dataset_dir)
        == dataset_dir.resolve()
    )


def test_aggregate_original_restores_10s_and_builds_30s_with_new_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataloader.builder.channels as channels_module
    import dataloader.builder.config as config_module
    import dataloader.operations as dataset_module
    from dataloader.files import write_json
    from dataloader.operations import aggregate_original

    dataset = tmp_path / "dataset"
    (dataset / "cycles").mkdir(parents=True)
    (dataset / "cycles_original").mkdir()
    config = Config(
        project_root=tmp_path,
        experiment_id="exp_20260715",
        experiment_date="2026-07-15",
        input_dir=tmp_path / "data-that-must-not-be-read",
        sensor_globs=("*.xls",),
        image_extensions=(),
        timestamp_column="时间",
        expected_sensor_interval_seconds=1,
        image_match_tolerance_seconds=2,
        edf_pair_tolerance_seconds=1,
        process=ProcessSettings(resample_interval_seconds=10),
    )
    monkeypatch.setattr(dataset_module, "_resolve_project_root", lambda: tmp_path)
    monkeypatch.setattr(config_module, "load_config", lambda **_kwargs: config)
    monkeypatch.setattr(
        channels_module,
        "load_channels",
        lambda: {
            "added_temperature": {
                "source_names": ["p1__new_point"],
                "unit": "degC",
                "kind": "continuous",
                "role": "sensor",
                "resample": "mean",
                "missing": "interpolate",
                "analysis_candidate": False,
                "coverage_required": True,
                "valid_range": [-50, 120],
            }
        },
    )

    cycle_name = "frost_cycle_000001"
    original = pd.DataFrame(
        {
            "experiment_id": "exp_20260715",
            "experiment_date": "2026-07-15",
            "timestamp": pd.date_range("2026-07-15", periods=60, freq="s"),
            "cycle_id": "partial_001",
            "cycle_stage": "partial",
            "cycle_status": "valid",
            "cycle_status_reason": "outside_complete_cycle",
            "p1__new_point": range(60),
        }
    )
    original.to_csv(dataset / "cycles_original" / f"{cycle_name}.csv", index=False)
    write_json(
        {
            "cycles": [
                {
                    "cycle_name": cycle_name,
                    "cycle_uid": "exp_20260715::partial_001",
                    "experiment_id": "exp_20260715",
                    "experiment_date": "2026-07-15",
                    "cycle_id": "partial_001",
                    "pipeline_status": "valid",
                    "pipeline_status_reason": "outside_complete_cycle",
                    "boundaries": {},
                    "assets": {
                        "csv": f"cycles/{cycle_name}.csv",
                        "parquet": f"cycles/{cycle_name}.parquet",
                        "original_csv": f"cycles_original/{cycle_name}.csv",
                    },
                }
            ]
        },
        dataset / "cycle_catalog.json",
    )
    write_json(
        {"resample_interval_seconds": 10, "channels": {}, "columns": []},
        dataset / "channel_registry.json",
    )

    aggregate_original(dataset, seconds=10)
    aggregate_original(dataset, seconds=30)

    restored = pd.read_csv(dataset / "cycles" / f"{cycle_name}.csv")
    aggregated = pd.read_csv(dataset / "cycles_30s" / f"{cycle_name}.csv")
    assert restored["added_temperature"].tolist() == [4.5, 14.5, 24.5, 34.5, 44.5, 54.5]
    assert aggregated["added_temperature"].tolist() == [14.5, 44.5]
    assert not config.input_dir.exists()


def test_cli_aggregates_original_at_requested_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import main_data

    calls: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        main_data,
        "aggregate_original",
        lambda dataset, *, seconds: calls.append((dataset, seconds)) or dataset,
        raising=False,
    )

    assert (
        main_data.main(
            [
                "aggregate-original",
                "--dataset",
                str(tmp_path),
                "--seconds",
                "30",
            ]
        )
        == 0
    )
    assert calls == [(tmp_path, 30)]


def test_cli_add_needs_no_date_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import main_data

    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        main_data,
        "add_dataset",
        lambda input_dir, dataset: calls.append((input_dir, dataset)) or input_dir,
    )

    assert (
        main_data.main(
            [
                "add",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert calls == [(tmp_path, Path("dataset"))]


def test_cli_render_cloud_fetch_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import main_data

    calls: list[bool] = []
    monkeypatch.setattr(
        main_data,
        "render_dataset",
        lambda _dataset, _cycle, **options: calls.append(
            bool(options["fetch_cloud_images"])
        )
        or tmp_path,
    )

    main_data.main(["render", "frost_cycle_000020", "--panel"])
    main_data.main(
        [
            "render",
            "frost_cycle_000020",
            "--panel",
            "--fetch-cloud-images",
        ]
    )

    assert calls == [False, True]
