from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def write_sensor_file():
    def _write(
        path: Path,
        header: Sequence[str],
        rows: Iterable[Sequence[object]],
        *,
        encoding: str = "gb18030",
        delimiter: str = "\t",
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding=encoding, newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter, lineterminator="\r\n")
            writer.writerow([*header, ""])
            for row in rows:
                writer.writerow([*row, ""])
        return path

    return _write


@pytest.fixture
def write_image():
    def _write(path: Path, *, size: tuple[int, int] = (24, 12)) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color=(20, 40, 60)).save(path)
        return path

    return _write
