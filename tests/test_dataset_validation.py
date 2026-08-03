from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from test_dataset_integration import _write_run

from frost_analysis.dataset import build_dataset
from frost_analysis.dataset_validation import validate_dataset


def _build_fixture(tmp_path: Path) -> Path:
    run = _write_run(tmp_path, "frost_0715", "2026-07-15")
    output = tmp_path / "outputs" / "datasets" / "frost_cycles_v1"
    build_dataset([run], output)
    return output


def test_validate_rejects_orphan_image_and_wrong_hash(tmp_path: Path) -> None:
    dataset = _build_fixture(tmp_path)
    image_path = next((dataset / "images").glob("*.jpg"))
    image_path.write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="SHA mismatch"):
        validate_dataset(dataset)

    image_path.write_bytes(b"image")
    (dataset / "images" / "orphan.jpg").write_bytes(b"orphan")
    with pytest.raises(ValueError, match="orphan"):
        validate_dataset(dataset)


def test_validate_rejects_dataset_directory_rename(tmp_path: Path) -> None:
    dataset = _build_fixture(tmp_path)
    renamed = dataset.parent / "renamed_dataset"
    dataset.rename(renamed)

    with pytest.raises(ValueError, match="directory name"):
        validate_dataset(renamed)


def test_validate_keeps_nullable_index_dtype(tmp_path: Path) -> None:
    dataset = _build_fixture(tmp_path)
    cycle_index = pd.read_parquet(dataset / "cycle_index.parquet")

    assert str(cycle_index["dataset_cycle_index"].dtype) == "Int64"
