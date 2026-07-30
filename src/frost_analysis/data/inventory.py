"""Recursive, checksum-backed raw-data inventory."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_TABLE_EXTENSIONS = {".xls", ".xlsx", ".csv", ".tsv", ".txt"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class TableMetadata:
    encoding: str
    delimiter: str
    row_count: int
    column_count: int
    time_field: str | None
    time_start: pd.Timestamp | None
    time_end: pd.Timestamp | None
    sampling_median_s: float | None
    duplicate_timestamps: int
    invalid_timestamps: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in _TABLE_EXTENSIONS and re.search(r"参数\d+", path.stem):
        return "monitoring_table"
    return "other"


def read_monitoring_table(path: Path) -> tuple[pd.DataFrame, TableMetadata]:
    encoding, sample = _decode_sample(path)
    delimiter = _detect_delimiter(sample)
    frame = pd.read_csv(
        path,
        sep=delimiter,
        encoding=encoding,
        dtype=str,
        keep_default_na=False,
        engine="python",
    )
    frame = frame.loc[:, [str(column).strip() != "" for column in frame.columns]]
    frame.columns = [str(column).strip() for column in frame.columns]
    time_field = _find_time_field(frame.columns)
    times = (
        pd.to_datetime(frame[time_field], errors="coerce")
        if time_field
        else pd.Series(dtype="datetime64[ns]")
    )
    valid = times.dropna().sort_values()
    deltas = valid.drop_duplicates().diff().dt.total_seconds()
    positive = deltas[deltas.gt(0)]
    metadata = TableMetadata(
        encoding=encoding,
        delimiter=delimiter,
        row_count=len(frame),
        column_count=len(frame.columns),
        time_field=time_field,
        time_start=None if valid.empty else valid.min(),
        time_end=None if valid.empty else valid.max(),
        sampling_median_s=None if positive.empty else float(positive.median()),
        duplicate_timestamps=int(valid.duplicated().sum()),
        invalid_timestamps=int(times.isna().sum()) if time_field else len(frame),
    )
    return frame, metadata


def inventory_directory(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inventory all files; parse only monitoring exports, never image pixels."""
    rows: list[dict[str, object]] = []
    columns: list[dict[str, object]] = []
    for path in sorted(item for item in input_dir.rglob("*") if item.is_file()):
        file_class = classify_file(path)
        base: dict[str, object] = {
            "file_name": path.name,
            "relative_path": path.relative_to(input_dir).as_posix(),
            "file_format": path.suffix.lower(),
            "file_class": file_class,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if file_class != "monitoring_table":
            rows.append(
                {**base, "status": "classified_only", "row_count": pd.NA, "column_count": pd.NA}
            )
            continue
        try:
            table, metadata = read_monitoring_table(path)
        except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
            rows.append({**base, "status": "read_error", "error": str(error)})
            continue
        rows.append(
            {
                **base,
                "status": "analyzed",
                "encoding": metadata.encoding,
                "delimiter": repr(metadata.delimiter),
                "row_count": metadata.row_count,
                "column_count": metadata.column_count,
                "time_field": metadata.time_field,
                "time_start": metadata.time_start,
                "time_end": metadata.time_end,
                "sampling_median_s": metadata.sampling_median_s,
                "duplicate_timestamps": metadata.duplicate_timestamps,
                "invalid_timestamps": metadata.invalid_timestamps,
                "overall_missing_rate": float(table.replace("", pd.NA).isna().mean().mean()),
            }
        )
        group_match = re.search(r"参数(?P<group>\d+)", path.stem)
        for column in table.columns:
            if column == metadata.time_field:
                continue
            values = table[column].replace("", pd.NA)
            numeric = pd.to_numeric(values, errors="coerce")
            nonmissing = int(values.notna().sum())
            columns.append(
                {
                    "source_file": path.relative_to(input_dir).as_posix(),
                    "parameter_group": group_match.group("group") if group_match else "",
                    "source_column": column,
                    "canonical_column": f"p{group_match.group('group')}__{column}"
                    if group_match
                    else column,
                    "inferred_dtype": "numeric"
                    if nonmissing and int(numeric.notna().sum()) == nonmissing
                    else "string",
                    "missing_rate": float(values.isna().mean()),
                    "unit": "unknown",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(columns)


def _decode_sample(path: Path) -> tuple[str, str]:
    sample = path.read_bytes()[:131_072]
    if sample.startswith((b"\xd0\xcf\x11\xe0", b"PK\x03\x04")):
        raise ValueError("binary Excel workbook is not a supported text export")
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return encoding, sample.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"cannot decode {path}")


def _detect_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error as error:
        counts = {delimiter: sample.count(delimiter) for delimiter in ("\t", ",", ";")}
        delimiter, count = max(counts.items(), key=lambda item: item[1])
        if count == 0:
            raise ValueError("no supported delimiter") from error
        return delimiter


def _find_time_field(columns: pd.Index[str]) -> str | None:
    for position, column in enumerate(columns):
        normalized = re.sub(r"[\s_-]+", "", str(column)).lower()
        if normalized in {"时间", "time", "timestamp", "datetime"} or position == 0:
            return str(column)
    return None
