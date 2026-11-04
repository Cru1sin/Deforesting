"""Unpack raw date ZIP/7z files, rename cameras, and flatten the xxxxS folder."""

from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from dataset_tools.dataset_paths import require_external_storage
from output_layout import PROJECT

CAMERAS = {
    "192.168.1.1_1": "top",
    "192.168.1.2_1": "left_close",
    "192.168.1.11_1": "top_close",
    "192.168.1.12_1": "extreme",
    "192.168.1.14_1": "left",
    "192.168.1.15_1": "front",
}


def _merge_verified(source: Path, target: Path) -> None:
    """Merge identical prior extraction; never overwrite differing contents."""
    if source.name == ".DS_Store":
        source.unlink()
    elif not target.exists():
        source.rename(target)
    elif source.is_dir() and target.is_dir():
        for child in source.iterdir():
            _merge_verified(child, target / child.name)
        (source / ".DS_Store").unlink(missing_ok=True)
        source.rmdir()
    elif source.is_file() and target.is_file() and filecmp.cmp(source, target, shallow=False):
        source.unlink()
    else:
        raise FileExistsError(target)


def organize_date(date_dir: Path, apply: bool = False, *, keep_archives: bool = False) -> None:
    """Extract, flatten, rename and verify; optionally retain source archives."""
    require_external_storage(date_dir)
    archives = sorted(
        p
        for p in date_dir.iterdir()
        if p.suffix.lower() in {".zip", ".7z"}
        and (p.stem in CAMERAS or p.stem == date_dir.name + "S")
    )
    print(
        f"{date_dir}: {len(archives)} archives; extract -> flatten -> rename -> verify"
        f" -> {'retain' if keep_archives else 'delete'}",
        flush=True,
    )
    if not apply:
        return
    with tempfile.TemporaryDirectory(prefix=".unpack-", dir=date_dir) as temporary:
        staging = Path(temporary)
        for archive in archives:
            print(f"[extract] {archive.name}", flush=True)
            if archive.suffix.lower() == ".7z":
                members = subprocess.run(
                    ["/usr/bin/tar", "-tf", str(archive)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.splitlines()
                if not members:
                    raise ValueError(f"Empty archive: {archive}")
                for member in members:
                    target = (staging / member).resolve()
                    if not target.is_relative_to(staging):
                        raise ValueError(f"Unsafe 7z member: {member}")
                    if target.exists() and not member.endswith("/"):
                        raise FileExistsError(target)
                subprocess.run(
                    ["/usr/bin/tar", "-xf", str(archive), "-C", str(staging)], check=True
                )
                continue
            with zipfile.ZipFile(archive, metadata_encoding="gbk") as bundle:
                if not any(not m.is_dir() for m in bundle.infolist()):
                    raise ValueError(f"Empty archive: {archive}")
                for member in bundle.infolist():
                    target = (staging / member.filename).resolve()
                    if not target.is_relative_to(staging):
                        raise ValueError(f"Unsafe ZIP member: {member.filename}")
                    if target.exists() and not member.is_dir():
                        raise FileExistsError(target)
                bundle.extractall(staging)  # ZIP CRC checked while reading.

        nested = staging / (date_dir.name + "S")
        if nested.is_dir():
            for source in nested.iterdir():
                target = staging / source.name
                _merge_verified(source, target)
            (nested / ".DS_Store").unlink(missing_ok=True)
            nested.rmdir()
        for old, new in CAMERAS.items():
            source = staging / old
            if source.is_dir():
                target = staging / new
                _merge_verified(source, target)

        # Existing uncompressed folders follow the same flatten/rename order.
        nested = date_dir / (date_dir.name + "S")
        if nested.is_dir():
            for source in nested.iterdir():
                target = date_dir / source.name
                _merge_verified(source, target)
            (nested / ".DS_Store").unlink(missing_ok=True)
            nested.rmdir()
        for old, new in CAMERAS.items():
            source = date_dir / old
            if source.is_dir():
                target = date_dir / new
                _merge_verified(source, target)

        files = [p for p in staging.rglob("*") if p.is_file() and p.name != ".DS_Store"]
        for source in files:
            target = date_dir / source.relative_to(staging)
            if target.exists() and not filecmp.cmp(source, target, shallow=False):
                raise FileExistsError(target)
        for source in files:
            target = date_dir / source.relative_to(staging)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            if not filecmp.cmp(source, target, shallow=False):
                raise ValueError(f"Extracted data differs: {target}")
        if not keep_archives:
            for archive in archives:
                archive.unlink()
        print(f"[done] {date_dir.name}: {len(files)} verified files", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data")
    parser.add_argument("--date", action="append", help="date folder name; repeatable")
    parser.add_argument("--apply", action="store_true", help="modify files; default: preview")
    parser.add_argument("--keep-archives", action="store_true", help="retain ZIP/7z sources")
    args = parser.parse_args()
    require_external_storage(args.data)
    dates = (
        [args.data / name for name in args.date]
        if args.date
        else sorted(p for p in args.data.iterdir() if p.is_dir() and p.name.isdigit())
    )
    for date_dir in dates:
        organize_date(date_dir, args.apply, keep_archives=args.keep_archives)


if __name__ == "__main__":
    main()
