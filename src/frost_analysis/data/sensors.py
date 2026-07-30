"""Deterministic monitoring-fragment merge and quality-preserving cleaning."""

from __future__ import annotations

import re
from collections.abc import Mapping
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
class SensorLoadResult:
    data: pd.DataFrame
    conflicts: pd.DataFrame
    schema: pd.DataFrame
    quality_summary: pd.DataFrame
    missing_intervals: pd.DataFrame
    sampling_summary: pd.DataFrame
    warnings: tuple[str, ...] = ()
    metrics: Mapping[str, int | float] | None = None

    @property
    def frame(self) -> pd.DataFrame:
        """Compatibility view for callers migrating from ``PreprocessResult``."""
        return self.data


def merge_parameter_fragments(
    group: str,
    paths: list[Path],
    input_dir: Path,
    *,
    duplicate_conflict_policy: str = "warn_keep_stable",
) -> ParameterMergeResult:
    """Merge one parameter group without reconstructing missing values."""
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
        merged.loc[merged["duplicate_count"].gt(1) & merged["duplicate_conflict"]]
        .drop(columns=["_fragment_order", "_signature"])
        .reset_index(drop=True)
    )
    if duplicate_conflict_policy not in {"warn_keep_stable", "error"}:
        raise ValueError(
            "duplicate_conflict_policy must be 'warn_keep_stable' or 'error'"
        )
    if duplicate_conflict_policy == "error" and not conflicts.empty:
        raise ValueError(f"conflicting duplicate timestamps in parameter group {group}")
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
    selected = _separate_and_clean(selected, value_columns)
    schema = pd.DataFrame(schema_rows).drop_duplicates(["canonical_column", "source_file"])
    return ParameterMergeResult(group, selected, conflicts, schema, pd.DataFrame(sampling_rows))


def load_sensor_data(
    input_dir: Path,
    *,
    duplicate_conflict_policy: str = "warn_keep_stable",
) -> SensorLoadResult:
    """Load observed sensor records and retain missing values without filling them."""
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
            duplicate_conflict_policy=duplicate_conflict_policy,
        )
        for group, paths in sorted(groups.items(), key=lambda item: int(item[0]))
    }
    indexed = [result.frame.set_index("sensor_time") for result in results.values()]
    frame = pd.concat(indexed, axis=1, join="outer").sort_index().reset_index()
    frame = _mark_not_sampled_states(frame)
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
    warnings = tuple(
        f"duplicate_conflict:{group}:{len(result.conflicts)}"
        for group, result in results.items()
        if not result.conflicts.empty
    )
    metrics: dict[str, int | float] = {
        "parameter_group_count": len(results),
        "sensor_row_count": len(frame),
        "conflicting_duplicate_count": int(len(conflicts)),
    }
    return SensorLoadResult(frame, conflicts, schema, quality, missing, sampling, warnings, metrics)


def write_preprocessed(
    result: SensorLoadResult, processed_dir: Path, tables_dir: Path
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
) -> pd.DataFrame:
    generated: dict[str, pd.Series] = {}
    for column in value_columns:
        raw = frame[column].fillna("").astype(str).str.strip()
        numeric = pd.to_numeric(raw.replace("", pd.NA), errors="coerce")
        nonempty = raw.ne("")
        numeric_ratio = float(numeric.notna().sum() / nonempty.sum()) if nonempty.any() else 0.0
        is_numeric = numeric_ratio >= 0.5 and numeric.notna().any()
        invalid = nonempty & numeric.isna()
        if is_numeric:
            generated[column] = numeric.astype(float)
        else:
            generated[column] = raw.replace("", pd.NA).astype("string")
        generated[f"{column}__invalid"] = invalid
        generated[f"{column}__raw"] = raw
        generated[f"{column}__missing"] = ~nonempty
        state = pd.Series("observed", index=frame.index, dtype="string")
        state.loc[~nonempty] = "missing"
        state.loc[invalid] = "invalid"
        generated[f"{column}__source_state"] = state
    base = frame.drop(columns=value_columns)
    return pd.concat([base, pd.DataFrame(generated, index=frame.index)], axis=1)


def _mark_not_sampled_states(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    value_columns = [
        column
        for column in result.columns
        if str(column).startswith("p")
        and "__" in str(column)
        and not str(column).endswith(
            ("__raw", "__missing", "__invalid", "__source_state", "__source_file")
        )
    ]
    for column in value_columns:
        state_column = f"{column}__source_state"
        if state_column not in result:
            continue
        states = result[state_column].astype("string")
        result[state_column] = states.fillna("not_sampled")
    return result


def _quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    value_columns = [
        column
        for column in frame.columns
        if column != "sensor_time"
        and not column.endswith(
            ("__raw", "__invalid", "__missing", "__source_state", "__interpolated")
        )
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
            ("__raw", "__invalid", "__missing", "__source_state", "__interpolated")
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
