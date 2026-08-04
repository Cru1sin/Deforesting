"""The read-only entry point for every self-contained Cycle Dataset consumer."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import DATASET_V2_SCHEMA_VERSION
from .dataset_images import scan_cycle_images
from .dataset_manifest import ASSESSMENT_STATUSES
from .dataset_v3 import V3_DATASET_SCHEMA_VERSION


class DatasetLoader:
    """Read a published Dataset without consulting source runs or YAML files."""

    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root.resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")
        self._manifest = _read_json(self.dataset_root / "dataset_manifest.json")
        version = self._manifest.get("dataset_schema_version")
        if version not in {DATASET_V2_SCHEMA_VERSION, V3_DATASET_SCHEMA_VERSION}:
            raise ValueError("DatasetLoader requires dataset schema version 2 or 3")
        self._version = int(version)
        self._cycle_index = pd.read_parquet(self.dataset_root / "cycle_index.parquet")
        self._image_metadata = pd.read_parquet(
            self.dataset_root / "image_metadata.parquet"
        )
        if self._version == V3_DATASET_SCHEMA_VERSION:
            self._registry = _read_json(self.dataset_root / "channel_registry.json")
        else:
            self._registry = {}
        if not (self.dataset_root / "cycles").is_dir():
            raise FileNotFoundError("dataset is missing cycles/")
        canonical_manifest = self._version == V3_DATASET_SCHEMA_VERSION and set(self._manifest) == {
            "dataset_schema_version",
            "dataset_id",
            "created_at",
            "updated_at",
            "source_experiments",
            "cycles",
        }
        if canonical_manifest and not (
            self.dataset_root / "cycles_original"
        ).is_dir():
            raise FileNotFoundError("Dataset v3 is missing cycles_original/")
        if not (self.dataset_root / "images").is_dir():
            raise FileNotFoundError("dataset is missing images/")

    @property
    def manifest(self) -> dict[str, object]:
        return self._manifest

    @property
    def schema_version(self) -> int:
        return self._version

    @property
    def cycle_index(self) -> pd.DataFrame:
        return self._cycle_index.copy()

    def list_cycles(
        self,
        *,
        statuses: set[str] | None = None,
        experiment_ids: set[str] | None = None,
    ) -> pd.DataFrame:
        """Return cycles joined with their current canonical status."""
        if statuses is not None and not statuses <= ASSESSMENT_STATUSES:
            raise ValueError(f"invalid cycle statuses: {sorted(statuses - ASSESSMENT_STATUSES)}")
        records = self._manifest.get("cycles")
        if not isinstance(records, list):
            raise ValueError("dataset manifest is missing cycles")
        assessments = {
            str(record["cycle_name"]): record
            for record in records
            if isinstance(record, dict)
        }
        result = self._cycle_index.copy()
        if self._version == V3_DATASET_SCHEMA_VERSION and not any(
            isinstance(record, dict) and "assessment" in record for record in records
        ):
            result["status"] = [
                str(assessments.get(str(name), {}).get("cycle_status", "invalid"))
                for name in result["cycle_name"]
            ]
            result["assessment_reasons"] = [[] for _ in result["cycle_name"]]
            result["assessment_note"] = [None for _ in result["cycle_name"]]
        else:
            result["status"] = [
                str(assessments.get(str(name), {}).get("assessment", {}).get("status", "invalid"))
                for name in result["cycle_name"]
            ]
            result["assessment_reasons"] = [
                assessments.get(str(name), {}).get("assessment", {}).get("reasons", [])
                for name in result["cycle_name"]
            ]
            result["assessment_note"] = [
                assessments.get(str(name), {}).get("assessment", {}).get("note")
                for name in result["cycle_name"]
            ]
        if statuses is not None:
            result = result.loc[result["status"].isin(statuses)]
        if experiment_ids is not None:
            result = result.loc[result["experiment_id"].isin(experiment_ids)]
        if "start_time" in result:
            result = result.assign(
                _start=pd.to_datetime(result["start_time"], errors="coerce")
            ).sort_values(["_start", "cycle_name"], kind="stable").drop(columns="_start")
        return result.reset_index(drop=True)

    def get_cycle_record(self, cycle_name: str) -> dict[str, object]:
        records = self._manifest.get("cycles")
        if not isinstance(records, list):
            raise ValueError("dataset manifest is missing cycles")
        for record in records:
            if isinstance(record, dict) and record.get("cycle_name") == cycle_name:
                return dict(record)
        raise KeyError(f"unknown cycle: {cycle_name}")

    def load_cycle(
        self,
        cycle_name: str,
        *,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        self.get_cycle_record(cycle_name)
        path = self.dataset_root / "cycles" / f"{cycle_name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"cycle parquet does not exist: {path}")
        frame = pd.read_parquet(path, columns=columns)
        if self._version == V3_DATASET_SCHEMA_VERSION:
            return frame
        path_columns = [
            column
            for column in frame.columns
            if str(column).startswith("image_") and str(column).endswith("_path")
        ]
        if path_columns:
            current_images = self.load_cycle_images(cycle_name)
            current_paths = {
                str(row["image_id"]): Path(str(row["path"]))
                .relative_to(self.dataset_root)
                .as_posix()
                for row in current_images.to_dict(orient="records")
            }
            for column in path_columns:
                present = frame[column].notna()
                frame.loc[present, column] = frame.loc[present, column].map(
                    lambda value: current_paths.get(Path(str(value)).stem, value)
                )
        return frame

    def load_image_metadata(self, cycle_name: str | None = None) -> pd.DataFrame:
        result = self._image_metadata.copy()
        if cycle_name is not None:
            result = result.loc[result["cycle_name"].eq(cycle_name)].reset_index(drop=True)
        return result

    def load_cycle_images(self, cycle_name: str) -> pd.DataFrame:
        self.get_cycle_record(cycle_name)
        if self._version == V3_DATASET_SCHEMA_VERSION:
            return _scan_v3_cycle_images(self.dataset_root, cycle_name, self._image_metadata)
        return scan_cycle_images(self.dataset_root, cycle_name, self._image_metadata)

    def load_cycle_original(
        self, cycle_name: str, *, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Load the preserved Prepared-resolution cycle CSV."""
        if self._version != V3_DATASET_SCHEMA_VERSION:
            raise ValueError("load_cycle_original requires Dataset schema version 3")
        self.get_cycle_record(cycle_name)
        path = self.dataset_root / "cycles_original" / f"{cycle_name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"original cycle CSV does not exist: {path}")
        return pd.read_csv(path, usecols=columns)

    @property
    def registry(self) -> dict[str, object]:
        return dict(self._registry)

    def iter_cycle_frames(
        self,
        *,
        statuses: set[str] | None = None,
        experiment_ids: set[str] | None = None,
        columns: list[str] | None = None,
    ) -> Iterator[tuple[dict[str, object], pd.DataFrame]]:
        cycles = self.list_cycles(statuses=statuses, experiment_ids=experiment_ids)
        for cycle_name in cycles["cycle_name"].astype(str):
            yield self.get_cycle_record(cycle_name), self.load_cycle(
                cycle_name, columns=columns
            )

    def publication_path(self, cycle_name: str) -> Path:
        self.get_cycle_record(cycle_name)
        return self.dataset_root / "cycles" / f"{cycle_name}.png"

    def rgb_coverage_path(self, cycle_name: str) -> Path:
        self.get_cycle_record(cycle_name)
        return self.dataset_root / "cycles" / f"{cycle_name}_rgb_coverage.png"


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset is missing {path.name}")
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"dataset manifest must be an object: {path}")
    return payload


def _scan_v3_cycle_images(
    dataset_root: Path, cycle_name: str, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Join current mutable role folders to immutable v3 image metadata."""
    columns = [
        "image_id",
        "cycle_name",
        "camera_role",
        "path",
        "cycle_uid",
        "frame_index",
        "source_camera_id",
        "initial_camera_slot",
        "image_time",
        "matched_timestamp",
        "offset_seconds",
        "cycle_stage",
        "source_relative_path",
        "file_size_bytes",
        "sha256",
    ]
    root = dataset_root / "images" / cycle_name
    rows: list[dict[str, object]] = []
    if root.is_dir():
        for role_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for image_path in sorted(role_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tif",
                    ".tiff",
                }:
                    rows.append(
                        {
                            "image_id": image_path.stem,
                            "cycle_name": cycle_name,
                            "camera_role": role_dir.name,
                            "path": image_path,
                        }
                    )
    scanned = pd.DataFrame(rows)
    scoped = metadata.loc[metadata["cycle_name"].eq(cycle_name)].copy()
    if scoped["image_id"].duplicated().any():
        raise ValueError(f"image metadata has duplicate image_id in {cycle_name}")
    if scanned.empty and scoped.empty:
        return pd.DataFrame(columns=columns)
    joined = scanned.merge(scoped, on=["image_id", "cycle_name"], how="outer", indicator=True)
    if joined["_merge"].ne("both").any():
        raise ValueError(f"image metadata and files are not a closed set: {cycle_name}")
    result = joined.drop(columns="_merge")[[column for column in columns if column in joined]]
    if not result.empty:
        result = result.sort_values(
            ["image_time", "source_relative_path", "image_id"],
            kind="stable",
        ).reset_index(drop=True)
    return result
