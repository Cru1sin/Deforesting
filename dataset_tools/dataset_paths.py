"""Direct file writers used by the self-contained Dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def require_external_storage(path: Path) -> None:
    """Fail before IO when a Proust directory link points to an unmounted drive."""
    if path.resolve().is_relative_to("/Volumes/Proust") and not Path("/Volumes/Proust").is_mount():
        raise FileNotFoundError("external drive is not mounted: /Volumes/Proust")


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
