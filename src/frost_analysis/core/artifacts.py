"""Atomic dataframe artifact writing with an auditable Parquet-to-CSV fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd


@dataclass(frozen=True)
class FrameWriteResult:
    requested_path: Path
    actual_path: Path
    storage_format: str
    fallback_reason: str = ""


def write_dataframe(frame: pd.DataFrame, parquet_path: Path) -> FrameWriteResult:
    """Write Parquet atomically, falling back to an atomic CSV artifact on failure."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary_parquet = parquet_path.with_name(f".{parquet_path.name}.{token}.tmp")
    try:
        frame.to_parquet(temporary_parquet, index=False)
        temporary_parquet.replace(parquet_path)
        return FrameWriteResult(parquet_path, parquet_path, "parquet")
    except Exception as error:  # the CSV attempt remains independently fail-fast
        temporary_parquet.unlink(missing_ok=True)
        csv_path = parquet_path.with_suffix(".csv")
        temporary_csv = csv_path.with_name(f".{csv_path.name}.{token}.tmp")
        try:
            frame.to_csv(temporary_csv, index=False)
            temporary_csv.replace(csv_path)
        finally:
            temporary_csv.unlink(missing_ok=True)
        return FrameWriteResult(
            requested_path=parquet_path,
            actual_path=csv_path,
            storage_format="csv",
            fallback_reason=f"{type(error).__name__}: {error}",
        )
