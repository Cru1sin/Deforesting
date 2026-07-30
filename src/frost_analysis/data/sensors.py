"""Deterministic monitoring-fragment merge and quality-preserving cleaning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.artifacts import FrameWriteResult, write_dataframe
from .inventory import read_monitoring_table


@dataclass(frozen=True)
class ParameterMergeResult:
    group: str
    frame: pd.DataFrame
    conflicts: pd.DataFrame
    schema: pd.DataFrame
    sampling_summary: pd.DataFrame


@dataclass(frozen=True)
class PreprocessResult:
    frame: pd.DataFrame
    conflicts: pd.DataFrame
    schema: pd.DataFrame
    quality_summary: pd.DataFrame
    missing_intervals: pd.DataFrame
    sampling_summary: pd.DataFrame


def merge_parameter_fragments(
    group: str,
    paths: list[Path],
    input_dir: Path,
    *,
    short_gap_max_seconds: float = 0,
    transition_guard_seconds: float = 30,
) -> ParameterMergeResult:
    """Merge one parameter group with auditable deterministic conflict resolution."""
    if not paths:
        raise ValueError(f"parameter group {group} has no files")
    fragments: list[tuple[pd.Timestamp, str, pd.DataFrame]] = []
    schema_rows: list[dict[str, object]] = []
    sampling_rows: list[dict[str, object]] = []
    for path in paths:
        table, metadata = read_monitoring_table(path)
        if metadata.time_field is None:
            raise ValueError(f"no timestamp field in {path}")
        time = pd.to_datetime(table.pop(metadata.time_field), errors="coerce")
        table = table.loc[time.notna()].copy()
        time = time.loc[time.notna()]
        renamed: dict[str, str] = {}
        seen: dict[str, int] = {}
        for position, column in enumerate(table.columns, start=2):
            base = str(column).strip() or f"unnamed_{position:03d}"
            seen[base] = seen.get(base, 0) + 1
            canonical = f"p{group}__{base}"
            if seen[base] > 1:
                canonical += f"__col_{position:03d}"
            renamed[str(column)] = canonical
            schema_rows.append(
                {
                    "parameter_group": group,
                    "source_column": base,
                    "canonical_column": canonical,
                    "source_file": path.relative_to(input_dir).as_posix(),
                    "unit": "unknown",
                }
            )
        table = table.rename(columns=renamed)
        table.insert(0, "sensor_time", time.to_numpy())
        table["source_file"] = path.relative_to(input_dir).as_posix()
        table["source_row"] = np.arange(2, len(table) + 2)
        first_time = table["sensor_time"].min()
        fragments.append((first_time, path.name, table))
        valid_time = table["sensor_time"].sort_values().drop_duplicates()
        delta = valid_time.diff().dt.total_seconds().dropna()
        median = float(delta[delta.gt(0)].median()) if delta.gt(0).any() else np.nan
        irregular = (
            int((delta.gt(0) & ~np.isclose(delta, median)).sum()) if np.isfinite(median) else 0
        )
        sampling_rows.append(
            {
                "parameter_group": group,
                "source_file": path.relative_to(input_dir).as_posix(),
                "sampling_median_s": median,
                "sampling_min_s": float(delta.min()) if not delta.empty else np.nan,
                "sampling_max_s": float(delta.max()) if not delta.empty else np.nan,
                "irregular_interval_count": irregular,
            }
        )
    fragments.sort(key=lambda item: (item[0], item[1]))
    ordered: list[pd.DataFrame] = []
    for order, (_, _, fragment) in enumerate(fragments):
        current = fragment.copy()
        current["_fragment_order"] = order
        ordered.append(current)
    merged = pd.concat(ordered, ignore_index=True, sort=False).sort_values(
        ["sensor_time", "_fragment_order", "source_file", "source_row"], kind="stable"
    )
    value_columns = [column for column in merged.columns if column.startswith(f"p{group}__")]
    signatures = pd.util.hash_pandas_object(merged[value_columns].fillna(""), index=False)
    merged["_signature"] = signatures
    merged["duplicate_count"] = merged.groupby("sensor_time")["sensor_time"].transform("size")
    merged["duplicate_conflict"] = (
        merged.groupby("sensor_time")["_signature"].transform("nunique").gt(1)
    )
    conflicts = (
        merged.loc[merged["duplicate_count"].gt(1)]
        .drop(columns=["_fragment_order", "_signature"])
        .reset_index(drop=True)
    )
    selected = (
        merged.drop_duplicates("sensor_time", keep="last")
        .sort_values("sensor_time")
        .reset_index(drop=True)
    )
    selected = selected.drop(columns=["_fragment_order", "_signature"])
    selected = selected.rename(
        columns={
            "source_file": f"p{group}__source_file",
            "source_row": f"p{group}__source_row",
            "duplicate_count": f"p{group}__duplicate_count",
            "duplicate_conflict": f"p{group}__duplicate_conflict",
        }
    )
    selected = _separate_and_clean(
        selected,
        value_columns,
        short_gap_max_seconds=short_gap_max_seconds,
        transition_guard_seconds=transition_guard_seconds,
    )
    schema = pd.DataFrame(schema_rows).drop_duplicates(["canonical_column", "source_file"])
    return ParameterMergeResult(group, selected, conflicts, schema, pd.DataFrame(sampling_rows))


def preprocess_directory(
    input_dir: Path,
    *,
    short_gap_max_seconds: float = 0,
    transition_guard_seconds: float = 30,
) -> PreprocessResult:
    groups: dict[str, list[Path]] = {}
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        match = re.search(r"参数(?P<group>\d+)", path.stem)
        if match and path.suffix.lower() in {".xls", ".csv", ".tsv", ".txt"}:
            groups.setdefault(match.group("group"), []).append(path)
    if not groups:
        raise ValueError(f"no monitoring tables under {input_dir}")
    results = {
        group: merge_parameter_fragments(
            group,
            paths,
            input_dir,
            short_gap_max_seconds=short_gap_max_seconds,
            transition_guard_seconds=transition_guard_seconds,
        )
        for group, paths in sorted(groups.items(), key=lambda item: int(item[0]))
    }
    indexed = [result.frame.set_index("sensor_time") for result in results.values()]
    frame = pd.concat(indexed, axis=1, join="outer").sort_index().reset_index()
    conflicts = pd.concat(
        [result.conflicts.assign(parameter_group=group) for group, result in results.items()],
        ignore_index=True,
    )
    schema = pd.concat([result.schema for result in results.values()], ignore_index=True)
    sampling = pd.concat(
        [result.sampling_summary for result in results.values()], ignore_index=True
    )
    quality = _quality_summary(frame)
    missing = _missing_intervals(frame)
    return PreprocessResult(frame, conflicts, schema, quality, missing, sampling)


def write_preprocessed(
    result: PreprocessResult, processed_dir: Path, tables_dir: Path
) -> FrameWriteResult:
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    storage = write_dataframe(result.frame, processed_dir / "processed_timeseries.parquet")
    result.quality_summary.to_csv(tables_dir / "data_quality_summary.csv", index=False)
    result.missing_intervals.to_csv(tables_dir / "missing_intervals.csv", index=False)
    result.sampling_summary.to_csv(tables_dir / "sampling_interval_summary.csv", index=False)
    result.conflicts.to_csv(tables_dir / "duplicate_conflicts.csv", index=False)
    return storage


def _separate_and_clean(
    frame: pd.DataFrame,
    value_columns: list[str],
    *,
    short_gap_max_seconds: float,
    transition_guard_seconds: float,
) -> pd.DataFrame:
    state_column = next(
        (
            column
            for column in value_columns
            if column.lower().endswith("__deforst") or column.lower().endswith("__defrost")
        ),
        None,
    )
    generated: dict[str, pd.Series] = {}
    for column in value_columns:
        raw = frame[column].fillna("").astype(str).str.strip()
        numeric = pd.to_numeric(raw.replace("", pd.NA), errors="coerce")
        nonempty = raw.ne("")
        numeric_ratio = float(numeric.notna().sum() / nonempty.sum()) if nonempty.any() else 0.0
        is_numeric = numeric_ratio >= 0.5 and numeric.notna().any()
        if is_numeric:
            working = pd.DataFrame(
                {
                    "sensor_time": frame["sensor_time"],
                    column: numeric.astype(float),
                    f"{column}__interpolated": False,
                },
                index=frame.index,
            )
            if state_column is not None and state_column != column:
                working[state_column] = frame[state_column]
            if short_gap_max_seconds > 0:
                _interpolate_short_gaps(
                    working,
                    column,
                    raw,
                    state_column=state_column,
                    max_seconds=short_gap_max_seconds,
                    transition_guard_seconds=transition_guard_seconds,
                )
            generated[column] = working[column]
            generated[f"{column}__interpolated"] = working[f"{column}__interpolated"]
            generated[f"{column}__invalid"] = nonempty & numeric.isna()
        else:
            generated[column] = raw.replace("", pd.NA).astype("string")
            generated[f"{column}__invalid"] = pd.Series(False, index=frame.index)
            generated[f"{column}__interpolated"] = pd.Series(False, index=frame.index)
        generated[f"{column}__raw"] = raw
        generated[f"{column}__missing"] = ~nonempty
    base = frame.drop(columns=value_columns)
    return pd.concat([base, pd.DataFrame(generated, index=frame.index)], axis=1)


def _interpolate_short_gaps(
    frame: pd.DataFrame,
    column: str,
    raw: pd.Series,
    *,
    state_column: str | None,
    max_seconds: float,
    transition_guard_seconds: float,
) -> None:
    values = frame[column]
    times = frame["sensor_time"]
    missing_positions = np.flatnonzero(raw.eq("").to_numpy())
    states = frame[state_column].astype("string") if state_column else None
    for raw_position in missing_positions:
        position = int(raw_position)
        previous = position - 1
        following = position + 1
        if previous < 0 or following >= len(frame):
            continue
        if pd.isna(values.iloc[previous]) or pd.isna(values.iloc[following]):
            continue
        elapsed = (times.iloc[following] - times.iloc[previous]).total_seconds()
        if elapsed > max_seconds:
            continue
        if states is not None and not (
            states.iloc[previous] == states.iloc[position] == states.iloc[following]
        ):
            continue
        if states is not None and transition_guard_seconds > 0:
            nearby = times.between(
                times.iloc[position] - pd.Timedelta(seconds=transition_guard_seconds),
                times.iloc[position] + pd.Timedelta(seconds=transition_guard_seconds),
            )
            if states.loc[nearby].dropna().nunique() > 1:
                continue
        fraction = (times.iloc[position] - times.iloc[previous]).total_seconds() / elapsed
        frame.loc[frame.index[position], column] = float(
            values.iloc[previous] + fraction * (values.iloc[following] - values.iloc[previous])
        )
        frame.loc[frame.index[position], f"{column}__interpolated"] = True


def _quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    value_columns = [
        column
        for column in frame.columns
        if column != "sensor_time"
        and not column.endswith(("__raw", "__invalid", "__missing", "__interpolated"))
        and "source_" not in column
        and column not in {"duplicate_count", "duplicate_conflict"}
    ]
    for column in value_columns:
        rows.append(
            {
                "column": column,
                "dtype": str(frame[column].dtype),
                "row_count": len(frame),
                "missing_count": int(frame[column].isna().sum()),
                "missing_rate": float(frame[column].isna().mean()),
                "invalid_count": int(
                    frame.get(f"{column}__invalid", pd.Series(False, index=frame.index)).sum()
                ),
                "interpolated_count": int(
                    frame.get(f"{column}__interpolated", pd.Series(False, index=frame.index)).sum()
                ),
                "unique_count": int(frame[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def _missing_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        if column == "sensor_time" or column.endswith(
            ("__raw", "__invalid", "__missing", "__interpolated")
        ):
            continue
        mask = frame[column].isna()
        groups = mask.ne(mask.shift(fill_value=False)).cumsum()
        for _, part in frame.loc[mask, ["sensor_time"]].groupby(groups[mask]):
            rows.append(
                {
                    "column": column,
                    "start": part["sensor_time"].min(),
                    "end": part["sensor_time"].max(),
                    "row_count": len(part),
                    "elapsed_seconds": float(
                        (part["sensor_time"].max() - part["sensor_time"].min()).total_seconds()
                    ),
                }
            )
    return pd.DataFrame(rows, columns=["column", "start", "end", "row_count", "elapsed_seconds"])
