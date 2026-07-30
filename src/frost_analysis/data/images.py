"""Inventory-backed image indexing without modifying image pixels."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Final

import pandas as pd
from PIL import Image, UnidentifiedImageError

_IMAGE_TIMESTAMP_FORMAT: Final = "%Y%m%d%H%M%S%f"
_IP_PATTERN = re.compile(r"(?<!\d)(?P<ip>(?:\d{1,3}\.){3}\d{1,3})(?!\d)")
_SAFE_ID_PATTERN = re.compile(r"[^A-Za-z0-9]+")

MANIFEST_COLUMNS = [
    "sample_id",
    "experiment_id",
    "camera_id",
    "ip_address",
    "camera_role",
    "image_time",
    "image_time_raw",
    "timestamp_ok",
    "timestamp_source",
    "clock_verified",
    "image_path",
    "image_extension",
    "width",
    "height",
    "image_mode",
    "file_size_bytes",
    "image_ok",
    "image_error",
]


def extract_ip(path: Path) -> str | None:
    """Extract a valid IPv4-looking identifier from a camera directory name."""
    match = _IP_PATTERN.search(path.name)
    if match is None:
        return None
    octets = match.group("ip").split(".")
    if any(int(octet) > 255 for octet in octets):
        return None
    return match.group("ip")


def build_image_manifest(
    input_dir: Path,
    inventory: pd.DataFrame,
    *,
    experiment_id: str,
    camera_roles: dict[str, str],
    unknown_role: str,
) -> pd.DataFrame:
    """Inspect inventory-classified images and emit one stable row per file."""
    required = {"file_class", "relative_path"}
    missing = sorted(required - set(inventory))
    if missing and not inventory.empty:
        raise ValueError(f"inventory missing image columns: {missing}")
    image_rows = (
        inventory.loc[inventory["file_class"].eq("image")]
        if required <= set(inventory)
        else pd.DataFrame()
    )
    records: list[dict[str, object]] = []
    for relative_path in sorted(image_rows.get("relative_path", pd.Series(dtype=str)).astype(str)):
        image_path = input_dir / relative_path
        camera_dir = Path(relative_path).parent
        ip = extract_ip(camera_dir)
        camera_id = _camera_id(camera_dir, ip)
        records.append(
            _inspect_image(
                image_path,
                relative_path=relative_path,
                experiment_id=experiment_id,
                camera_id=camera_id,
                ip_address=ip,
                camera_role=camera_roles.get(ip or "", unknown_role),
                image_time=_parse_image_time(image_path.stem),
            )
        )
    if not records:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    manifest = pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS)
    duplicate_ids = manifest["sample_id"].duplicated(keep=False)
    manifest.loc[duplicate_ids, "sample_id"] += manifest.loc[duplicate_ids, "image_path"].map(
        lambda value: f"__{_path_digest(str(value))}"
    )
    return manifest.sort_values(
        ["image_time", "camera_id", "image_path"],
        na_position="last",
        ignore_index=True,
    )


def _camera_id(camera_dir: Path, ip: str | None) -> str:
    identifier = ip or camera_dir.name
    return f"cam_{_SAFE_ID_PATTERN.sub('_', identifier).strip('_')}"


def _parse_image_time(stem: str) -> pd.Timestamp | None:
    if len(stem) != 17 or not stem.isdigit():
        return None
    parsed = pd.to_datetime(stem, format=_IMAGE_TIMESTAMP_FORMAT, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _inspect_image(
    image_path: Path,
    *,
    relative_path: str,
    experiment_id: str,
    camera_id: str,
    ip_address: str | None,
    camera_role: str,
    image_time: pd.Timestamp | None,
) -> dict[str, object]:
    width: int | None = None
    height: int | None = None
    image_mode: str | None = None
    image_ok = False
    image_error = ""
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            image_mode = image.mode
            image.verify()
        image_ok = True
    except (OSError, UnidentifiedImageError) as error:
        image_error = f"{type(error).__name__}: {error}"
    return {
        "sample_id": _sample_id(experiment_id, camera_id, image_path.stem, relative_path),
        "experiment_id": experiment_id,
        "camera_id": camera_id,
        "ip_address": ip_address or "",
        "camera_role": camera_role,
        "image_time": image_time,
        "image_time_raw": image_path.stem,
        "timestamp_ok": image_time is not None,
        "timestamp_source": "filename",
        "clock_verified": False,
        "image_path": relative_path,
        "image_extension": image_path.suffix.lower(),
        "width": width,
        "height": height,
        "image_mode": image_mode,
        "file_size_bytes": image_path.stat().st_size,
        "image_ok": image_ok,
        "image_error": image_error,
    }


def _sample_id(experiment_id: str, camera_id: str, stem: str, relative_path: str) -> str:
    if len(stem) == 17 and stem.isdigit():
        return f"{experiment_id}__{camera_id}__{stem}"
    return f"{experiment_id}__{camera_id}__invalid_{_path_digest(relative_path)}"


def _path_digest(relative_path: str) -> str:
    return hashlib.sha1(relative_path.encode()).hexdigest()[:12]
