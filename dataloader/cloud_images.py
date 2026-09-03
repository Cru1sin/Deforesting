"""Materialize Dataset images from local or rclone-hosted ZIP archives."""

from __future__ import annotations

import binascii
import os
import shutil
import struct
import subprocess
import time
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

DEFAULT_CLOUD_IMAGES_REMOTE = "onedrive_hkust:HKUST/Project/Defrost/dataset/images"


def _direct_rclone_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "RCLONE_HTTP_PROXY",
    ):
        env.pop(name, None)
    env["NO_PROXY"] = env["no_proxy"] = "*"
    return env


def _require_free_space(
    path: Path, required: int, action: str, minimum_free_gib: float
) -> None:
    if required + minimum_free_gib * 1024**3 > shutil.disk_usage(path).free:
        raise OSError(
            f"{action} would cross the {minimum_free_gib:g} GiB safety floor; "
            "review local cycle images before deleting anything"
        )


def _read_zip_range(
    archive: Path | str,
    offset: int,
    count: int,
    *,
    remote: str | None = None,
    transferred: list[int] | None = None,
) -> bytes:
    if count == 0:
        return b""
    if remote is None:
        with Path(archive).open("rb") as stream:
            stream.seek(offset)
            data = stream.read(count)
    else:
        result = subprocess.run(
            [
                "rclone",
                "cat",
                remote,
                "--offset",
                str(offset),
                "--count",
                str(count),
                "--http-proxy",
                "",
            ],
            check=True,
            capture_output=True,
            env=_direct_rclone_env(),
        )
        data = result.stdout
        if isinstance(data, str):
            data = data.encode()
    if len(data) != count:
        raise OSError(f"short ZIP range read: expected {count}, got {len(data)}")
    if transferred is not None:
        transferred[0] += len(data)
    return data


def _zip_member_index(
    archive: Path | str,
    archive_size: int,
    *,
    remote: str | None = None,
    transferred: list[int] | None = None,
) -> dict[str, tuple[int, int, int, int, int, int]]:
    tail_offset = max(0, archive_size - 65_557)
    tail = _read_zip_range(
        archive,
        tail_offset,
        archive_size - tail_offset,
        remote=remote,
        transferred=transferred,
    )
    eocd_offset = tail.rfind(b"PK\x05\x06")
    if eocd_offset < 0:
        raise ValueError("ZIP end-of-central-directory record is missing")
    fields = struct.unpack_from("<4s4H2LH", tail, eocd_offset)
    _, disk, central_disk, entries_disk, entries, central_size, central_offset, _ = fields
    if (
        disk
        or central_disk
        or entries_disk == 0xFFFF
        or entries == 0xFFFF
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
    ):
        raise NotImplementedError("ZIP64 or multi-disk ZIP is not supported")
    if entries == 0:
        return {}
    if central_offset + central_size <= archive_size and central_offset >= tail_offset:
        central = tail[central_offset - tail_offset : central_offset - tail_offset + central_size]
    else:
        central = _read_zip_range(
            archive,
            central_offset,
            central_size,
            remote=remote,
            transferred=transferred,
        )
    index: dict[str, tuple[int, int, int, int, int, int]] = {}
    cursor = 0
    for _ in range(entries):
        if central[cursor : cursor + 4] != b"PK\x01\x02":
            raise ValueError("invalid ZIP central-directory entry")
        values = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        filename_size, extra_size, comment_size = values[10:13]
        start = cursor + 46
        filename = central[start : start + filename_size].decode("cp437")
        compressed_size, uncompressed_size, crc, local_offset = (
            values[8],
            values[9],
            values[7],
            values[16],
        )
        index[filename] = (
            local_offset,
            compressed_size,
            uncompressed_size,
            crc,
            filename_size,
            extra_size,
        )
        cursor = start + filename_size + extra_size + comment_size
    return index


def _read_zip_member(
    archive: Path | str,
    member: tuple[int, int, int, int, int, int],
    *,
    archive_size: int,
    remote: str | None = None,
    transferred: list[int] | None = None,
) -> bytes:
    (
        local_offset,
        compressed_size,
        uncompressed_size,
        expected_crc,
        filename_size,
        extra_size,
    ) = member
    payload = _read_zip_range(
        archive,
        local_offset,
        min(
            30 + filename_size + 65_535 + compressed_size,
            archive_size - local_offset,
        ),
        remote=remote,
        transferred=transferred,
    )
    header = payload[:30]
    if header[:4] != b"PK\x03\x04":
        raise ValueError("invalid ZIP local-file header")
    (
        _,
        _version,
        flags,
        method,
        _time,
        _date,
        _crc,
        _size,
        _uncompressed,
        filename_size,
        extra_size,
    ) = (
        struct.unpack("<4s5H3L2H", header)
    )
    if flags & 0x1:
        raise NotImplementedError("encrypted ZIP members are not supported")
    data_offset = 30 + filename_size + extra_size
    payload = payload[data_offset : data_offset + compressed_size]
    if method == 0:
        data = payload
    elif method == 8:
        data = zlib.decompress(payload, -15)
    else:
        raise NotImplementedError(f"ZIP compression method {method} is not supported")
    if len(data) != uncompressed_size or (binascii.crc32(data) & 0xFFFFFFFF) != expected_crc:
        raise ValueError("ZIP member checksum or size mismatch")
    return data


@contextmanager
def materialize_cycle_image_members(
    dataset_dir: Path,
    cycle_name: str,
    file_names: Iterator[str] | list[str] | tuple[str, ...],
    *,
    fetch_cloud: bool = False,
    cloud_root: Path | None = None,
    minimum_free_gib: float = 50,
) -> Iterator[Path]:
    """Retain selected ``front`` JPEGs from a ZIP using range reads."""
    if not cycle_name.startswith("frost_cycle_") or not cycle_name[12:].isdigit():
        raise ValueError(f"invalid cycle name: {cycle_name}")
    images_root = Path(dataset_dir).resolve() / "images"
    cycle_dir = images_root / cycle_name
    names = sorted({str(name) for name in file_names if str(name)})
    if not names:
        yield cycle_dir
        return
    if any(
        PurePosixPath(name).is_absolute()
        or ".." in PurePosixPath(name).parts
        or PurePosixPath(name).name != name
        for name in names
    ):
        raise ValueError("unsafe requested ZIP member name")
    names = [name for name in names if not (cycle_dir / "front" / name).is_file()]
    if not names or not fetch_cloud:
        yield cycle_dir
        return

    archive_name = f"{cycle_name}.zip"
    default_cloud = cloud_root is None
    remote: str | None = None
    if default_cloud:
        remote = f"{DEFAULT_CLOUD_IMAGES_REMOTE}/{archive_name}"
        result = subprocess.run(
            [
                "rclone",
                "lsf",
                remote,
                "--files-only",
                "--format",
                "s",
                "--http-proxy",
                "",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=_direct_rclone_env(),
        )
        raw_size = result.stdout.strip()
        if result.returncode != 0 or not raw_size.isdigit():
            print(f"[images] no local directory or cloud ZIP: {cycle_name}", flush=True)
            yield cycle_dir
            return
        archive_size = int(raw_size)
        archive: Path | str = archive_name
    else:
        archive = Path(cloud_root) / archive_name
        if not archive.is_file():
            print(f"[images] no local directory or cloud ZIP: {cycle_name}", flush=True)
            yield cycle_dir
            return
        archive_size = archive.stat().st_size

    images_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    transferred = [0]
    index = _zip_member_index(
        archive,
        archive_size,
        remote=remote,
        transferred=transferred,
    )
    members = [index.get(f"{cycle_name}/front/{name}") for name in names]
    if any(member is None for member in members):
        missing = [name for name, member in zip(names, members, strict=True) if member is None]
        raise FileNotFoundError("ZIP members are missing: " + ", ".join(missing))
    required = sum(member[2] for member in members if member is not None)
    _require_free_space(
        images_root,
        required,
        f"extracting selected images from {archive_name}",
        minimum_free_gib,
    )
    front = cycle_dir / "front"
    front.mkdir(parents=True, exist_ok=True)
    for name, member in zip(names, members, strict=True):
        assert member is not None
        target = front / name
        temporary = target.with_suffix(f"{target.suffix}.part")
        temporary.write_bytes(
            _read_zip_member(
                archive,
                member,
                archive_size=archive_size,
                remote=remote,
                transferred=transferred,
            )
        )
        temporary.replace(target)
    print(
        f"[images] range retained {len(names)} member(s) for {cycle_name}: "
        f"{transferred[0]} bytes in {time.monotonic() - started:.2f}s",
        flush=True,
    )
    yield cycle_dir


@contextmanager
def materialize_cycle_images(  # noqa: C901
    dataset_dir: Path,
    cycle_name: str,
    *,
    fetch_cloud: bool = False,
    cloud_root: Path | None = None,
    cleanup_downloaded: bool = False,
    minimum_free_gib: float = 50,
) -> Iterator[Path]:
    """Materialize one cloud ZIP; optionally remove this call's copy after success."""
    if not cycle_name.startswith("frost_cycle_") or not cycle_name[12:].isdigit():
        raise ValueError(f"invalid cycle name: {cycle_name}")
    images_root = Path(dataset_dir).resolve() / "images"
    cycle_dir = images_root / cycle_name
    if cycle_dir.is_dir() or not fetch_cloud:
        yield cycle_dir
        return

    archive_name = f"{cycle_name}.zip"
    default_cloud = cloud_root is None
    if default_cloud:
        archive = Path(archive_name)
        remote_file = f"{DEFAULT_CLOUD_IMAGES_REMOTE}/{archive_name}"
        remote = subprocess.run(
            [
                "rclone",
                "lsf",
                remote_file,
                "--files-only",
                "--format",
                "s",
                "--http-proxy",
                "",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=_direct_rclone_env(),
        )
        remote_size = remote.stdout.strip()
        if remote.returncode != 0 or not remote_size.isdigit():
            print(f"[images] no local directory or cloud ZIP: {cycle_name}", flush=True)
            yield cycle_dir
            return
        archive_bytes = int(remote_size)
    else:
        archive = Path(cloud_root) / archive_name
        if not archive.is_file():
            print(f"[images] no local directory or cloud ZIP: {cycle_name}", flush=True)
            yield cycle_dir
            return
        archive_bytes = archive.stat().st_size

    images_root.mkdir(parents=True, exist_ok=True)
    _require_free_space(
        images_root,
        archive_bytes,
        f"downloading {archive.name}",
        minimum_free_gib,
    )

    with TemporaryDirectory(prefix=f".{cycle_name}-", dir=images_root) as temporary:
        work = Path(temporary)
        local_archive = work / archive.name
        print(f"[images] copying cloud ZIP: {archive.name}", flush=True)
        if default_cloud:
            subprocess.run(
                [
                    "rclone",
                    "copyto",
                    f"{DEFAULT_CLOUD_IMAGES_REMOTE}/{archive.name}",
                    str(local_archive),
                    "--progress",
                    "--stats",
                    "1s",
                    "--multi-thread-streams",
                    "8",
                    "--multi-thread-cutoff",
                    "256M",
                    "--timeout",
                    "2m",
                    "--retries",
                    "10",
                    "--retries-sleep",
                    "10s",
                    "--http-proxy",
                    "",
                ],
                check=True,
                env=_direct_rclone_env(),
            )
        else:
            shutil.copyfile(archive, local_archive)
        with ZipFile(local_archive) as bundle:
            members = bundle.infolist()
            for member in members:
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or path.parts[:1] != (cycle_name,):
                    raise ValueError(f"unsafe cycle ZIP member: {member.filename}")
            required = sum(member.file_size for member in members)
            _require_free_space(
                images_root, required, f"extracting {archive.name}", minimum_free_gib
            )
            print(f"[images] extracting local copy: {cycle_name}", flush=True)
            bundle.extractall(work)

        extracted = work / cycle_name
        if not extracted.is_dir():
            raise ValueError(f"cycle ZIP has no {cycle_name} directory")
        if cycle_dir.exists():
            yield cycle_dir
            return
        extracted.replace(cycle_dir)
        print(f"[images] retained local copy: {cycle_name}", flush=True)
        completed = False
        try:
            yield cycle_dir
            completed = True
        finally:
            if cleanup_downloaded and completed:
                shutil.rmtree(cycle_dir)
                print(f"[images] cleaned downloaded copy: {cycle_name}", flush=True)
