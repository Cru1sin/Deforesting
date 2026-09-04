"""Materialize Dataset images from local or rclone-hosted ZIP archives."""

from __future__ import annotations

import binascii
import os
import shutil
import struct
import subprocess
import time
import zlib
from collections import Counter
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ZipMember = tuple[int, int, int, int, int, int]
ImageJob = tuple[str, Path | str, str | None, int, ZipMember, Path]

DEFAULT_CLOUD_IMAGES_REMOTE = os.environ.get(
    "HEAT_PUMP_DEFROST_IMAGE_REMOTE",
    "onedrive_hkust:HKUST/Project/Defrost/dataset/images",
)


def _cloud_file(name: str) -> str:
    if not DEFAULT_CLOUD_IMAGES_REMOTE:
        raise ValueError(
            "cloud image access requires HEAT_PUMP_DEFROST_IMAGE_REMOTE or cloud_root"
        )
    return f"{DEFAULT_CLOUD_IMAGES_REMOTE}/{name}"


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
    zip64 = (
        disk == 0xFFFF
        or central_disk == 0xFFFF
        or entries_disk == 0xFFFF
        or entries == 0xFFFF
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
    )
    if zip64:
        locator_offset = tail.rfind(b"PK\x06\x07", 0, eocd_offset)
        if locator_offset < 0:
            raise ValueError("ZIP64 end-of-central-directory locator is missing")
        _, locator_disk, record_offset, total_disks = struct.unpack_from(
            "<4sLQL", tail, locator_offset
        )
        if locator_disk or total_disks != 1:
            raise NotImplementedError("multi-disk ZIP is not supported")
        relative_offset = record_offset - tail_offset
        if 0 <= relative_offset <= len(tail) - 56:
            record = tail[relative_offset : relative_offset + 56]
        else:
            record = _read_zip_range(
                archive,
                record_offset,
                56,
                remote=remote,
                transferred=transferred,
            )
        values = struct.unpack("<4sQ2H2L4Q", record)
        if values[0] != b"PK\x06\x06" or values[1] < 44:
            raise ValueError("invalid ZIP64 end-of-central-directory record")
        _, _, _, _, disk, central_disk, entries_disk, entries, central_size, central_offset = (
            values
        )
    if disk or central_disk or entries_disk != entries:
        raise NotImplementedError("multi-disk ZIP is not supported")
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
        extra = central[start + filename_size : start + filename_size + extra_size]
        compressed_size, uncompressed_size, crc, local_offset = (
            values[8],
            values[9],
            values[7],
            values[16],
        )
        if 0xFFFFFFFF in (compressed_size, uncompressed_size, local_offset):
            extra_cursor = 0
            while extra_cursor + 4 <= len(extra):
                field_id, field_size = struct.unpack_from("<2H", extra, extra_cursor)
                field = extra[extra_cursor + 4 : extra_cursor + 4 + field_size]
                if field_id == 0x0001:
                    value_cursor = 0
                    if uncompressed_size == 0xFFFFFFFF:
                        uncompressed_size = struct.unpack_from("<Q", field, value_cursor)[0]
                        value_cursor += 8
                    if compressed_size == 0xFFFFFFFF:
                        compressed_size = struct.unpack_from("<Q", field, value_cursor)[0]
                        value_cursor += 8
                    if local_offset == 0xFFFFFFFF:
                        local_offset = struct.unpack_from("<Q", field, value_cursor)[0]
                    break
                extra_cursor += 4 + field_size
            if 0xFFFFFFFF in (compressed_size, uncompressed_size, local_offset):
                raise ValueError("ZIP64 member extra field is missing")
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


def _plan_image_members(
    dataset_dir: Path,
    cycle_name: str,
    file_names: Iterator[str] | list[str] | tuple[str, ...],
    cloud_root: Path | None,
    minimum_free_gib: float,
) -> tuple[list[ImageJob], list[int], float]:
    if not cycle_name.startswith("frost_cycle_") or not cycle_name[12:].isdigit():
        raise ValueError(f"invalid cycle name: {cycle_name}")
    images_root = Path(dataset_dir).resolve() / "images"
    cycle_dir = images_root / cycle_name
    requested = sorted({PurePosixPath(str(name)) for name in file_names if str(name)}, key=str)
    if any(
        path.is_absolute() or ".." in path.parts or len(path.parts) > 2 for path in requested
    ):
        raise ValueError("unsafe requested ZIP member name")
    requested = [
        path if len(path.parts) == 2 else PurePosixPath("front") / path for path in requested
    ]
    requested = [path for path in requested if not (cycle_dir / Path(*path.parts)).is_file()]
    if not requested:
        return [], [0], time.monotonic()

    archive_name = f"{cycle_name}.zip"
    remote: str | None = None
    if cloud_root is None:
        remote = _cloud_file(archive_name)
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
            return [], [0], time.monotonic()
        archive_size = int(raw_size)
        archive: Path | str = archive_name
    else:
        archive = Path(cloud_root) / archive_name
        if not archive.is_file():
            print(f"[images] no local directory or cloud ZIP: {cycle_name}", flush=True)
            return [], [0], time.monotonic()
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
    members = [index.get(f"{cycle_name}/{path.as_posix()}") for path in requested]
    if any(member is None for member in members):
        missing = [
            str(path)
            for path, member in zip(requested, members, strict=True)
            if member is None
        ]
        raise FileNotFoundError("ZIP members are missing: " + ", ".join(missing))
    required = sum(member[2] for member in members if member is not None)
    _require_free_space(
        images_root,
        required,
        f"extracting selected images from {archive_name}",
        minimum_free_gib,
    )
    jobs = [
        (
            cycle_name,
            archive,
            remote,
            archive_size,
            member,
            cycle_dir / Path(*path.parts),
        )
        for path, member in zip(requested, members, strict=True)
        if member is not None
    ]
    return jobs, transferred, started


def _download_image_member(job: ImageJob, transferred: list[int] | None = None) -> Path:
    _, archive, remote, archive_size, member, target = job
    target.parent.mkdir(parents=True, exist_ok=True)
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
    return target


@contextmanager
def materialize_image_members(
    dataset_dir: Path,
    requests: Mapping[str, list[str]],
    *,
    cloud_root: Path | None = None,
    cleanup_downloaded: bool = False,
    minimum_free_gib: float = 50,
    n_jobs: int = 10,
) -> Iterator[Iterator[str]]:
    """Download all requested OneDrive ZIP members through one bounded pool."""
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    cycles = list(requests)
    downloaded: list[Path] = []
    executor = ThreadPoolExecutor(max_workers=n_jobs)
    try:
        plans: dict[str, list[ImageJob]] = {}
        planning = {
            executor.submit(
                _plan_image_members,
                dataset_dir,
                cycle,
                requests[cycle],
                cloud_root,
                minimum_free_gib,
            ): cycle
            for cycle in cycles
        }
        for completed, future in enumerate(as_completed(planning), start=1):
            cycle = planning[future]
            plans[cycle] = future.result()[0]
            print(f"[images:index] {completed}/{len(cycles)} {cycle}", flush=True)
        jobs = [job for cycle in cycles for job in plans[cycle]]
        downloaded = [job[-1] for job in jobs]
        print(f"[images] {len(jobs)} missing panel images with {n_jobs} workers", flush=True)
        remaining = Counter(job[0] for job in jobs)
        futures: dict[Future[Path], str] = {
            executor.submit(_download_image_member, job): job[0] for job in jobs
        }

        def completed_cycles() -> Iterator[str]:
            for cycle in cycles:
                if remaining[cycle] == 0:
                    yield cycle
            for completed, future in enumerate(as_completed(futures), start=1):
                cycle = futures[future]
                future.result()
                print(f"[images] {completed}/{len(jobs)} {cycle}", flush=True)
                remaining[cycle] -= 1
                if remaining[cycle] == 0:
                    yield cycle

        yield completed_cycles()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        if cleanup_downloaded:
            for target in downloaded:
                target.unlink(missing_ok=True)


@contextmanager
def materialize_cycle_image_members(
    dataset_dir: Path,
    cycle_name: str,
    file_names: Iterator[str] | list[str] | tuple[str, ...],
    *,
    fetch_cloud: bool = False,
    cloud_root: Path | None = None,
    cleanup_downloaded: bool = False,
    minimum_free_gib: float = 50,
) -> Iterator[Path]:
    """Retain selected camera JPEGs from a ZIP using range reads."""
    if not cycle_name.startswith("frost_cycle_") or not cycle_name[12:].isdigit():
        raise ValueError(f"invalid cycle name: {cycle_name}")
    cycle_dir = Path(dataset_dir).resolve() / "images" / cycle_name
    if not fetch_cloud:
        yield cycle_dir
        return
    jobs, transferred, started = _plan_image_members(
        dataset_dir,
        cycle_name,
        file_names,
        cloud_root,
        minimum_free_gib,
    )
    downloaded = [_download_image_member(job, transferred) for job in jobs]
    print(
        f"[images] range retained {len(jobs)} member(s) for {cycle_name}: "
        f"{transferred[0]} bytes in {time.monotonic() - started:.2f}s",
        flush=True,
    )
    try:
        yield cycle_dir
    finally:
        if cleanup_downloaded:
            for target in downloaded:
                target.unlink(missing_ok=True)


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
        remote_file = _cloud_file(archive_name)
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
                    _cloud_file(archive.name),
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
