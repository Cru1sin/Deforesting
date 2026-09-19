"""Read-only access to the final, self-contained Dataset format."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .cycle_metadata import read_catalog, read_manifest
from .local_images import scan_cycle_images


class DatasetLoader:
    """Load Dataset schema 3 without consulting Raw data or configuration files."""

    def __init__(self, dataset_root: Path) -> None:
        self.recovery_settings = None
        self.dataset_root = Path(dataset_root).resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")
        self._manifest = read_manifest(self.dataset_root)
        self._catalog = read_catalog(self.dataset_root)
        self._registry = _read_object(self.dataset_root / "channel_registry.json")
        metadata_path = self.dataset_root / "image_metadata.parquet"
        if not metadata_path.is_file():
            raise FileNotFoundError("Dataset is missing image_metadata.parquet")
        self._image_metadata = pd.read_parquet(metadata_path)
        for name in ("cycles", "cycles_original"):
            if not (self.dataset_root / name).is_dir():
                raise FileNotFoundError(f"Dataset is missing {name}/")

    def configure_recovery(self, settings):
        """Apply one offline boundary definition in memory; never rewrite raw data."""
        from .builder.detect_cycles import heating_episode_bounds, recovery_control_trace, find_defrost_preparation_start

        self.recovery_settings = dict(settings)
        rows = []
        for record in self._catalog["cycles"]:
            frame = self.load_cycle_original(str(record["cycle_name"]))
            bounds = heating_episode_bounds(self, str(record["cycle_name"]), frame)
            frame["timestamp"] = pd.to_datetime(frame.timestamp)
            if not bounds.get("defrost_preparation_start"):
                bounds["defrost_preparation_start"] = find_defrost_preparation_start(
                    frame, pd.to_datetime(bounds.get("stable_heating_start")),
                    pd.to_datetime(bounds.get("defrost_start")), settings)
            start = pd.to_datetime(bounds.get("heating_start"))
            end = pd.to_datetime(
                bounds.get("defrost_preparation_start") or bounds.get("defrost_start")
            )
            frame = frame.loc[frame.timestamp.ge(start)] if pd.notna(start) else frame.iloc[:0]
            if pd.notna(end):
                frame = frame.loc[frame.timestamp.lt(end)]
            trace = recovery_control_trace(frame, settings)
            confirmed = trace.loc[trace.normal_heating, "timestamp"]
            boundary = confirmed.iloc[0] if len(confirmed) else None
            bounds["stable_heating_start"] = boundary
            record["boundaries"] = bounds
            record["recovery_status"] = (
                trace.recovery_status.iloc[-1] if len(trace) else "no_heating_observations"
            )
            rows.append(
                {
                    "cycle_name": record["cycle_name"],
                    "experiment_id": record["experiment_id"],
                    "heating_start": bounds["heating_start"],
                    "recorded_heating_start": bounds["recorded_heating_start"],
                    "heating_origin": bounds["heating_origin"],
                    "excluded_prestart_seconds": bounds["excluded_prestart_seconds"],
                    "stable_heating_start": boundary,
                    "recovery_status": record["recovery_status"],
                }
            )
        return pd.DataFrame(rows)

    @property
    def manifest(self) -> dict[str, object]:
        return dict(self._manifest)

    @property
    def catalog(self) -> dict[str, object]:
        result = dict(self._catalog)
        result["cycles"] = [dict(record) for record in self._catalog["cycles"]]
        return result

    def list_cycles(
        self,
        *,
        statuses: set[str] | None = None,
        experiment_ids: set[str] | None = None,
    ) -> pd.DataFrame:
        records = [record for record in self._catalog["cycles"] if isinstance(record, Mapping)]
        rows: list[dict[str, Any]] = []
        for record in records:
            boundaries = record.get("boundaries")
            data = record.get("data")
            image = record.get("image")
            row: dict[str, Any] = dict(record)
            if isinstance(boundaries, Mapping):
                row.update(boundaries)
            if isinstance(data, Mapping):
                row.update(data)
            if isinstance(image, Mapping):
                row["image_count"] = image.get("image_count", 0)
            row["status"] = str(record.get("status", "invalid"))
            row["status_reason"] = record.get("status_reason")
            row["pareto_knee_status"] = str(record.get("pareto_knee_status", "invalid"))
            row["rgb_knee_coverage_status"] = str(record.get("rgb_knee_coverage_status", "invalid"))
            row["pareto_extrapolated_knee_status"] = str(
                record.get("pareto_extrapolated_knee_status", "invalid")
            )
            row["rgb_extrapolated_knee_coverage_status"] = str(
                record.get("rgb_extrapolated_knee_coverage_status", "invalid")
            )
            rows.append(row)
        result = pd.DataFrame(rows)
        if result.empty:
            return result
        if statuses is not None:
            result = result.loc[result["status"].isin(statuses)]
        if experiment_ids is not None:
            result = result.loc[result["experiment_id"].isin(experiment_ids)]
        if "start_time" in result:
            result = (
                result.assign(_start=pd.to_datetime(result["start_time"], errors="coerce"))
                .sort_values(["_start", "cycle_name"], kind="stable")
                .drop(columns="_start")
            )
        return result.reset_index(drop=True)

    def list_valid_cycles(self, *, require_rgb: bool = False) -> pd.DataFrame:
        """Return cycles valid both now and before COP-derived quality updates."""
        cycles = self.list_cycles(statuses={"valid"})
        missing = pd.Series(False, index=cycles.index)
        pre_cop_valid = cycles.get("pre_cop_status", missing).map(
            lambda value: isinstance(value, Mapping) and value.get("status") == "valid"
        )
        selected = pre_cop_valid
        if require_rgb:
            selected &= cycles.get("rgb_valid", missing).eq(True)
        return cycles.loc[selected].reset_index(drop=True)

    def get_cycle_record(self, cycle_name: str) -> dict[str, object]:
        for record in self._catalog["cycles"]:
            if isinstance(record, Mapping) and record.get("cycle_name") == cycle_name:
                return dict(record)
        raise KeyError(f"unknown cycle: {cycle_name}")

    def load_cycle(self, cycle_name: str, *, columns: list[str] | None = None) -> pd.DataFrame:
        record = self.get_cycle_record(cycle_name)
        assets = record.get("assets")
        if not isinstance(assets, Mapping):
            raise ValueError(f"cycle assets are missing: {cycle_name}")
        path = self.dataset_root / str(assets["parquet"])
        if not path.is_file():
            raise FileNotFoundError(f"cycle parquet does not exist: {path}")
        frame = pd.read_parquet(path, columns=columns)
        if (
            self.recovery_settings is not None
            and "timestamp" in frame
            and record["boundaries"].get("heating_origin") in {"cold_start", "not_started"}
        ):
            frame = frame.loc[
                pd.to_datetime(frame.timestamp).ge(
                    pd.to_datetime(record["boundaries"]["heating_start"])
                )
            ].copy()
        if self.recovery_settings is not None and {"cycle_stage", "timestamp"} <= set(frame):
            from .builder.detect_cycles import recovery_stages

            frame["cycle_stage"] = recovery_stages(frame, record["boundaries"])
        return frame

    def load_cycle_original(
        self, cycle_name: str, *, columns: list[str] | None = None
    ) -> pd.DataFrame:
        record = self.get_cycle_record(cycle_name)
        assets = record.get("assets")
        if not isinstance(assets, Mapping):
            raise ValueError(f"cycle assets are missing: {cycle_name}")
        path = self.dataset_root / str(assets["original_csv"])
        if not path.is_file():
            raise FileNotFoundError(f"original cycle CSV does not exist: {path}")
        return pd.read_csv(path, usecols=columns)

    def load_image_metadata(self, cycle_name: str | None = None) -> pd.DataFrame:
        result = self._image_metadata.copy()
        if cycle_name is not None:
            result = result.loc[result["cycle_name"].astype(str).eq(cycle_name)]
        return result.reset_index(drop=True)

    def load_cycle_images(self, cycle_name: str) -> pd.DataFrame:
        return scan_cycle_images(
            self.dataset_root,
            cycle_name,
            self._image_metadata,
        )

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
        for cycle_name in self.list_cycles(statuses=statuses, experiment_ids=experiment_ids)[
            "cycle_name"
        ].astype(str):
            yield self.get_cycle_record(cycle_name), self.load_cycle(cycle_name, columns=columns)

    def publication_path(self, cycle_name: str) -> Path:
        record = self.get_cycle_record(cycle_name)
        assets = record.get("assets")
        if not isinstance(assets, Mapping):
            raise ValueError(f"cycle assets are missing: {cycle_name}")
        return self.dataset_root / str(assets["publication"])

    def rgb_panel_path(self, cycle_name: str) -> Path:
        record = self.get_cycle_record(cycle_name)
        assets = record.get("assets")
        if not isinstance(assets, Mapping):
            raise ValueError(f"cycle assets are missing: {cycle_name}")
        return self.dataset_root / str(assets["rgb_panel"])


def _read_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset is missing {path.name}")
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload
