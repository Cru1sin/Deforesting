"""Explicit scientific columns for the Cycle Dataset."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

SCIENTIFIC_CHANNEL_FIELDS = (
    "kind",
    "unit",
    "role",
    "resample",
    "formula",
    "dependencies",
    "scale",
    "offset",
    "analysis_candidate",
    "expected_frost_direction",
    "coverage_required",
)

_IMAGE_COLUMN_RE = re.compile(r"^image_.+_(?:path|time|offset_seconds)$")
_SOURCE_QUALITY_SUFFIXES = (
    "__missing",
    "__invalid",
    "__duplicate",
    "__conflict",
)


def is_image_column(name: object) -> bool:
    return _IMAGE_COLUMN_RE.fullmatch(str(name)) is not None


def drop_image_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[name for name in frame if is_image_column(name)])


def export_original_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        str(name)
        for name in frame
        if str(name) not in {"cycle_elapsed_seconds", "cycle_progress"}
        and not is_image_column(name)
        and not str(name).endswith(_SOURCE_QUALITY_SUFFIXES)
        and not str(name).endswith("__baseline")
        and not str(name).endswith("__baseline_residual")
    ]
    if "timestamp" not in columns:
        raise ValueError("Prepared data has no timestamp column")
    return frame.loc[:, columns].sort_values("timestamp", kind="stable").reset_index(drop=True)


def merge_original_columns(builds: Sequence[Any]) -> list[str]:
    columns: list[str] = []
    for build in builds:
        source = build.original if build.original is not None else build.prepared
        for name in export_original_frame(source):
            if str(name) not in columns:
                columns.append(str(name))
    return columns


def registry_from_frame(
    frame: pd.DataFrame,
    channels: Mapping[str, Mapping[str, Any]],
    *,
    resample_interval_seconds: int = 10,
) -> dict[str, Any]:
    scientific = drop_image_columns(frame)
    return {
        "resample_interval_seconds": int(resample_interval_seconds),
        "channels": {
            str(name): {field: settings.get(field) for field in SCIENTIFIC_CHANNEL_FIELDS}
            for name, settings in channels.items()
        },
        "columns": [str(name) for name in scientific.columns],
    }


def merge_registries(existing: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    if int(existing.get("resample_interval_seconds", 10)) != int(
        candidate.get("resample_interval_seconds", 10)
    ):
        raise ValueError("channel registry resample interval changed")
    old_channels = dict(existing.get("channels", {}))
    new_channels = dict(candidate.get("channels", {}))
    for name in old_channels.keys() & new_channels.keys():
        if old_channels[name] != new_channels[name]:
            raise ValueError(f"channel definition changed: {name}")
    columns = list(dict.fromkeys([*existing.get("columns", []), *candidate.get("columns", [])]))
    return {
        "resample_interval_seconds": int(existing.get("resample_interval_seconds", 10)),
        "channels": {**old_channels, **new_channels},
        "columns": columns,
    }


def build_registry(builds: Sequence[Any]) -> dict[str, Any]:
    registry: dict[str, Any] | None = None
    for build in builds:
        candidate = registry_from_frame(
            build.processed,
            build.channels,
            resample_interval_seconds=int(build.config.process.resample_interval_seconds),
        )
        registry = candidate if registry is None else merge_registries(registry, candidate)
    if registry is None:
        raise ValueError("Dataset requires at least one processed build")
    process = builds[0].config.process
    baseline = getattr(process, "baseline", None)
    registry.update(
        {
            "baseline_seconds": int(getattr(baseline, "baseline_seconds", 60)),
            "baseline_managed": False,
            "recovery_edit": {
                "mode": "ts-minus",
                "seconds": None,
                "fallback_seconds": float(
                    getattr(builds[0].config.cycles, "stable_heating_seconds", 180)
                ),
                "managed": False,
            },
            "image_coverage": {"max_image_gap_seconds": 40.0},
        }
    )
    return registry


def canonical_frame(frame: pd.DataFrame, registry: Mapping[str, Any]) -> pd.DataFrame:
    source = drop_image_columns(frame).copy()
    columns = [str(name) for name in registry.get("columns", [])]
    unknown = [str(name) for name in source if str(name) not in columns]
    if unknown:
        raise ValueError(f"Processed columns are not in channel registry: {unknown}")
    return source.reindex(columns=columns)


def build_processed_frame(
    frame: pd.DataFrame,
    registry: Mapping[str, Any],
    *,
    cycle_name: str,
    cycle_uid: str,
) -> pd.DataFrame:
    result = canonical_frame(frame, registry)
    result.insert(0, "cycle_uid", cycle_uid)
    result.insert(0, "cycle_name", cycle_name)
    return result


def align_original_schema(
    dataset_dir: Path,
    records: Sequence[dict[str, Any]],
    original_columns: Sequence[str],
) -> None:
    from ..dataset_paths import write_csv

    expected = [str(name) for name in original_columns]
    for record in records:
        path = dataset_dir / str(record["assets"]["original_csv"])
        if pd.read_csv(path, nrows=0).columns.tolist() != expected:
            write_csv(pd.read_csv(path).reindex(columns=expected), path)
