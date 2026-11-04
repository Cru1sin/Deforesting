"""Package raw dates or cycle images and upload them to OneDrive."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from output_layout import PROJECT

from .cloud_images import _direct_rclone_env
from .cycle_metadata import image_root
from .dataset_paths import require_external_storage

DATE_REMOTE = "onedrive_hkust:HKUST/Project/Defrost/data"
IMAGE_REMOTE = "onedrive_hkust:HKUST/Project/Defrost/dataset/images"


def upload_date_data(
    data_dir: Path, dates: Sequence[str] | None, *, minimum_free_gib: float = 5
) -> Path:
    """Upload selected four-digit dates, or every date when dates is None."""
    data = Path(data_dir).resolve()
    require_external_storage(data)
    if not data.is_dir():
        raise FileNotFoundError(f"data directory not found: {data}")
    selected = list(dates) if dates is not None else _available_dates(data)
    if not selected:
        raise ValueError(f"no four-digit date directories or ZIPs in {data}")
    invalid = [date for date in selected if re.fullmatch(r"\d{4}", date) is None]
    if invalid:
        raise ValueError("dates must contain four digits: " + ", ".join(invalid))

    temporary = data / ".date_archive_tmp"
    temporary.mkdir(exist_ok=True)
    for index, date in enumerate(selected, start=1):
        source_zip = data / f"{date}.zip"
        source_dir = data / date
        generated = not source_zip.is_file()
        archive = temporary / f"{date}.zip" if generated else source_zip
        print(f"[upload-data] {index}/{len(selected)} {date}", flush=True)
        if generated and not archive.is_file():
            if not source_dir.is_dir():
                raise FileNotFoundError(f"date source not found: {source_dir}")
            _pack_tree(data, source_dir, archive, minimum_free_gib, "upload-data")
        _upload_archive(archive, f"{DATE_REMOTE}/{date}.zip")
        if generated:
            archive.unlink()
        print(f"[upload-data] verified {date}.zip", flush=True)
    return data


def upload_cycle_images(
    dataset_dir: Path,
    cycles: Sequence[str] | None,
    cleanup_uploaded_images: bool = False,
    *,
    minimum_free_gib: float = 5,
) -> Path:
    """Upload selected cycle-image directories, or every cycle when cycles is None."""
    dataset = Path(dataset_dir).resolve()
    images = image_root(dataset)
    if not images.is_dir():
        raise FileNotFoundError(f"cycle images directory not found: {images}")
    selected = list(cycles) if cycles is not None else _available_cycles(images)
    if not selected:
        raise ValueError(f"no frost_cycle_* image directories in {images}")
    invalid = [cycle for cycle in selected if re.fullmatch(r"frost_cycle_\d+", cycle) is None]
    if invalid:
        raise ValueError("invalid cycle names: " + ", ".join(invalid))

    temporary = images.parent / ".image_archive_tmp"
    temporary.mkdir(exist_ok=True)
    for index, cycle in enumerate(selected, start=1):
        source = images / cycle
        if not source.is_dir():
            raise FileNotFoundError(f"cycle images not found: {source}")
        remote = f"{IMAGE_REMOTE}/{cycle}.zip"
        print(f"[upload-images] {index}/{len(selected)} {cycle}", flush=True)
        if _remote_size(remote):
            print(f"[upload-images] already on OneDrive: {cycle}.zip", flush=True)
            continue
        archive = temporary / f"{cycle}.zip"
        if not archive.is_file():
            _pack_tree(images, source, archive, minimum_free_gib, "upload-images")
        _upload_archive(archive, remote)
        archive.unlink()
        if cleanup_uploaded_images:
            shutil.rmtree(source)
        print(f"[upload-images] verified {cycle}.zip", flush=True)
    return dataset


def _available_dates(data: Path) -> list[str]:
    return sorted(
        {
            path.stem if path.is_file() else path.name
            for path in data.iterdir()
            if re.fullmatch(r"\d{4}(?:\.zip)?", path.name)
            and (path.is_dir() or path.suffix == ".zip")
        }
    )


def _available_cycles(images: Path) -> list[str]:
    return sorted(
        path.name
        for path in images.iterdir()
        if path.is_dir() and re.fullmatch(r"frost_cycle_\d+", path.name)
    )


def _pack_tree(
    base: Path, source: Path, archive: Path, minimum_free_gib: float, label: str
) -> None:
    archive.parent.mkdir(exist_ok=True)
    partial = archive.with_suffix(".zip.part")
    partial.unlink(missing_ok=True)
    files = sorted(path for path in source.rglob("*") if path.is_file())
    required = sum(path.stat().st_size for path in files)
    if required + minimum_free_gib * 1024**3 > shutil.disk_usage(archive.parent).free:
        raise OSError(f"not enough space to pack {source.name}; keep {minimum_free_gib:g} GiB free")
    print(f"[{label}] packing {source.name}", flush=True)
    with ZipFile(partial, "w", compression=ZIP_STORED, allowZip64=True) as bundle:
        for index, path in enumerate(files, start=1):
            bundle.write(path, path.relative_to(base))
            if index % 100 == 0 or index == len(files):
                print(f"[{label}] packed {index}/{len(files)}", flush=True)
    partial.replace(archive)


def _upload_archive(archive: Path, remote: str) -> None:
    subprocess.run(
        [
            "rclone",
            "copyto",
            str(archive),
            remote,
            "--size-only",
            "--onedrive-chunk-size",
            "100Mi",
            "--progress",
            "--stats",
            "1s",
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
    if _remote_size(remote) != archive.stat().st_size:
        raise OSError(f"OneDrive size verification failed: {archive.name}")


def _remote_size(remote: str) -> int | None:
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
    size = result.stdout.strip()
    return int(size) if result.returncode == 0 and size.isdigit() else None


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    upload = actions.add_parser("upload-data", help="package raw dates and upload to OneDrive")
    upload.add_argument("dates", nargs="+")
    upload.add_argument("--data", type=Path, default=(PROJECT / "data"))

    images = actions.add_parser("upload-images", help="package cycle images and upload to OneDrive")
    images.add_argument("cycles", nargs="+")
    images.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    images.add_argument("--cleanup-uploaded-images", action="store_true")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "upload-data":
        if "all" in args.dates and args.dates != ["all"]:
            parser.error("'all' cannot be combined with explicit dates")
        print(upload_date_data(args.data, None if args.dates == ["all"] else args.dates))
    elif args.action == "upload-images":
        if "all" in args.cycles and args.cycles != ["all"]:
            parser.error("'all' cannot be combined with explicit cycles")
        print(
            upload_cycle_images(
                args.dataset,
                None if args.cycles == ["all"] else args.cycles,
                args.cleanup_uploaded_images,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
