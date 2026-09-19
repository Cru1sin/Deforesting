from __future__ import annotations

import subprocess
from pathlib import Path
from zipfile import ZipFile


def test_upload_date_data_packs_uploads_and_preserves_sources(
    tmp_path: Path, monkeypatch
) -> None:
    from dataset_tools import cloud_upload

    data = tmp_path / "data"
    (data / "0806").mkdir(parents=True)
    (data / "0806" / "sensor.csv").write_text("raw", encoding="utf-8")
    (data / "0807.zip").write_bytes(b"existing-zip")
    remote_sizes: dict[str, int] = {}
    uploaded_members: dict[str, list[str]] = {}

    def rclone(command: list[str], **options: object) -> subprocess.CompletedProcess:
        if command[1] == "copyto":
            remote_sizes[command[3]] = Path(command[2]).stat().st_size
            if command[2].endswith("0806.zip"):
                with ZipFile(command[2]) as archive:
                    uploaded_members["0806"] = archive.namelist()
            return subprocess.CompletedProcess(command, 0)
        assert command[1] == "lsf"
        return subprocess.CompletedProcess(
            command, 0, stdout=f"{remote_sizes[command[2]]}\n"
        )

    monkeypatch.setattr(cloud_upload.subprocess, "run", rclone)
    cloud_upload.upload_date_data(data, None, minimum_free_gib=0)

    assert sorted(Path(name).name for name in remote_sizes) == ["0806.zip", "0807.zip"]
    assert uploaded_members == {"0806": ["0806/sensor.csv"]}
    assert (data / "0806" / "sensor.csv").is_file()
    assert (data / "0807.zip").read_bytes() == b"existing-zip"
    assert not (data / ".date_archive_tmp" / "0806.zip").exists()


def test_upload_cycle_images_packs_and_only_cleans_up_when_requested(
    tmp_path: Path, monkeypatch
) -> None:
    from dataset_tools import cloud_upload

    dataset = tmp_path / "dataset"
    cycle = "frost_cycle_000001"
    source = dataset / "images" / cycle
    source.mkdir(parents=True)
    (source / "front.jpg").write_bytes(b"rgb")
    remote_sizes: dict[str, int] = {}
    uploaded_members: list[str] = []

    def rclone(command: list[str], **options: object) -> subprocess.CompletedProcess:
        if command[1] == "lsf":
            size = remote_sizes.get(command[2])
            return subprocess.CompletedProcess(
                command,
                0 if size is not None else 1,
                stdout="" if size is None else f"{size}\n",
            )
        assert command[1] == "copyto"
        with ZipFile(command[2]) as archive:
            uploaded_members.extend(archive.namelist())
        remote_sizes[command[3]] = Path(command[2]).stat().st_size
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cloud_upload.subprocess, "run", rclone)
    cloud_upload.upload_cycle_images(
        dataset, None, cleanup_uploaded_images=False, minimum_free_gib=0
    )

    assert uploaded_members == [f"{cycle}/front.jpg"]
    assert source.is_dir()
    assert not (dataset / ".image_archive_tmp" / f"{cycle}.zip").exists()

    cloud_upload.upload_cycle_images(
        dataset, [cycle], cleanup_uploaded_images=False, minimum_free_gib=0
    )
    assert uploaded_members == [f"{cycle}/front.jpg"]

    second = "frost_cycle_000002"
    second_source = dataset / "images" / second
    second_source.mkdir()
    (second_source / "front.jpg").write_bytes(b"rgb")
    cloud_upload.upload_cycle_images(
        dataset,
        [second],
        cleanup_uploaded_images=True,
        minimum_free_gib=0,
    )

    assert not second_source.exists()
    assert source.is_dir()
