"""Read-only access to the final, self-contained Dataset format."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .cycle_metadata import image_root, read_catalog, read_manifest
from .local_images import scan_cycle_images


class DatasetLoader:
    """Load Dataset schema 3 without consulting Raw data or configuration files."""

    training_directory = "training_data"

    def __init__(self, dataset_root: Path) -> None:
        self.recovery_settings = None
        self.dataset_root = Path(dataset_root).resolve()
        if self.dataset_root.name == self.training_directory:
            self.dataset_root = self.dataset_root.parent
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")
        self._manifest = read_manifest(self.dataset_root)
        self._catalog = read_catalog(self.dataset_root)
        self._registry = _read_object(self.dataset_root / "channel_registry.json")
        self._image_metadata = None
        for name in ("cycles", "cycles_original"):
            if not (self.dataset_root / name).is_dir():
                raise FileNotFoundError(f"Dataset is missing {name}/")

    def load_rb_reference(self, cycles, *, path=None):
        """Read frozen trigger times; missing cycles require an explicit RB refresh."""
        path = Path(path) if path is not None else self.dataset_root / "priors/rb_reference.csv"
        reference = pd.read_csv(path, parse_dates=["t_RB"])
        reference = reference.set_index("cycle_name", verify_integrity=True)
        missing = set(cycles) - set(reference.index)
        if missing:
            raise ValueError(f"RB prior missing cycles; refresh explicitly: {sorted(missing)}")
        reference = reference.loc[list(cycles)].copy()
        catalog = self.list_cycles().set_index("cycle_name")
        if not reference.experiment_id.eq(catalog.loc[reference.index, "experiment_id"]).all():
            raise ValueError("RB prior experiment IDs differ from Dataset")
        if not reference.rb_status.eq("triggered").eq(reference.t_RB.notna()).all():
            raise ValueError("RB prior timestamps disagree with trigger status")
        return reference

    def configure_recovery(self, settings):
        """Apply one offline boundary definition in memory; never rewrite raw data."""
        from .builder.detect_cycles import (
            find_defrost_preparation_start,
            heating_episode_bounds,
            recovery_control_trace,
        )

        if settings.get("recovery_rule") == "dataset":
            return pd.DataFrame(
                [
                    {
                        "cycle_name": record["cycle_name"],
                        "experiment_id": record["experiment_id"],
                        **record["boundaries"],
                    }
                    for record in self._catalog["cycles"]
                ]
            )
        self.recovery_settings = dict(settings)
        rows = []
        for record in self._catalog["cycles"]:
            frame = self.load_cycle_original(str(record["cycle_name"]))
            bounds = heating_episode_bounds(self, str(record["cycle_name"]), frame)
            frame["timestamp"] = pd.to_datetime(frame.timestamp)
            if not bounds.get("defrost_preparation_start"):
                bounds["defrost_preparation_start"] = find_defrost_preparation_start(
                    frame,
                    pd.to_datetime(bounds.get("stable_heating_start")),
                    pd.to_datetime(bounds.get("defrost_start")),
                    settings,
                )
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
        include_rb: bool = True,
    ) -> pd.DataFrame:
        records = [record for record in self._catalog["cycles"] if isinstance(record, Mapping)]
        rows: list[dict[str, Any]] = []
        for record in records:
            if not include_rb and str(record["assets"]["parquet"]).startswith("cycles_RB/"):
                continue
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

    def list_valid_cycles(
        self, *, require_rgb: bool = False, include_rb: bool = True
    ) -> pd.DataFrame:
        """Return cycles valid both now and before COP-derived quality updates."""
        cycles = self.list_cycles(statuses={"valid"}, include_rb=include_rb)
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

    def load_cycle_measurements(self, cycle_name: str) -> pd.DataFrame:
        """Calculation inputs from published offline interpolation and smoothing.

        Legacy Datasets retain the original measurement path.

        A processed snapshot is held only within its stage and for at most one
        Dataset interval. Original missing observations remain missing.
        """
        from heat_smoothing.methods import MEASUREMENT_PROCESSING, PROCESSED_CHANNELS

        from .builder.detect_cycles import recovery_stages

        if self.recovery_settings is not None:
            raise ValueError(
                "processed measurements require the saved Dataset boundaries; use --recovery-rule dataset"
            )
        if self.registry.get("measurement_processing") == MEASUREMENT_PROCESSING:
            from heat_smoothing.methods import measurement_values

            return measurement_values(self.load_cycle(cycle_name))
        raw = self.load_cycle_original(cycle_name).copy()
        raw["timestamp"] = pd.to_datetime(raw.timestamp)
        raw = raw.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
        columns = [name + "_processed" for name in PROCESSED_CHANNELS]
        processed = self.load_cycle(
            cycle_name, columns=["timestamp", "cycle_stage", *columns]
        ).copy()
        processed["timestamp"] = pd.to_datetime(processed.timestamp)
        raw["cycle_stage"] = recovery_stages(raw, self.get_cycle_record(cycle_name)["boundaries"])
        raw["cycle_stage"] = raw.cycle_stage.astype("string")
        processed["cycle_stage"] = processed.cycle_stage.astype("string")
        result = pd.merge_asof(
            raw,
            processed.sort_values("timestamp"),
            on="timestamp",
            by="cycle_stage",
            direction="backward",
            tolerance=pd.Timedelta(seconds=self.registry["resample_interval_seconds"]),
        )
        for name in PROCESSED_CHANNELS:
            result[name + "_processed"] = result[name + "_processed"].where(result[name].notna())
        return result

    @property
    def training_settings(self):
        return _read_object(self.dataset_root / self.training_directory / "settings.json")

    def camera_training_settings(self, camera_role="front"):
        suffix = "" if camera_role == "front" else f"__{camera_role}"
        return _read_object(self.dataset_root / self.training_directory / f"settings{suffix}.json")

    def load_screening_inputs(self, camera_role="front"):
        """Full candidates and measured quality evidence, independent of source paths."""
        training = self.dataset_root / self.training_directory
        suffix = "" if camera_role == "front" else f"__{camera_role}"
        paths = [
            training / "candidate_decisions.parquet",
            training / f"quality_metrics{suffix}.csv",
            training / f"action_validity{suffix}.parquet",
        ]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing published screening evidence: {path}; "
                    "run python -m dataset_tools.training_data screening-inputs before training or evaluation"
                )
        return pd.read_parquet(paths[0]), pd.read_csv(paths[1]), pd.read_parquet(paths[2])

    def training_cohort(self, camera_role="front"):
        """Camera-specific membership in the shared flat publication."""
        cohort = pd.read_csv(self.dataset_root / self.training_directory / "cohort.csv")
        if "camera_role" not in cohort:
            cohort["camera_role"] = "front"
        return cohort.loc[cohort.camera_role.eq(camera_role)].copy()

    def load_training_data(self, *, objective="set_stop", visual_columns=None, camera_role="front"):
        """Load published trajectories and frozen features; never prepare data here."""
        import numpy as np

        if objective not in {"set_stop", "cdf_ramp"}:
            raise ValueError("unknown training objective")
        training = self.dataset_root / self.training_directory
        cohort = self.training_cohort(camera_role)
        if cohort.empty:
            raise ValueError(f"No published training cycles for camera {camera_role}")
        visual = (
            visual_columns
            if visual_columns is not None
            else [f"dinov2_{i:03d}" for i in range(384)]
        )
        pieces = []
        for record in cohort.itertuples():
            name = record.cycle_name
            suffix = "" if camera_role == "front" else f"__{camera_role}"
            rows = pd.read_parquet(training / f"{name}{suffix}.parquet")
            features = pd.read_parquet(
                self.dataset_root / "image_DINO_feature" / f"{name}.parquet",
                columns=["cycle_name", "file_name", *visual],
                filters=[("camera_role", "==", camera_role)],
            )
            rows = rows.merge(
                features, on=["cycle_name", "file_name"], how="left", validate="many_to_one"
            )
            if not np.isfinite(rows.loc[rows.input_valid, visual]).all().all():
                raise ValueError(
                    f"published valid actions have missing frozen DINO vectors: {name}"
                )
            rows["complete_reward"] = bool(record.complete_reward)
            rows["training_eligible"] = True
            rows["evaluation_eligible"] = bool(record.evaluation_eligible)
            pieces.append(rows)
        reference = pd.read_parquet(training / "reference.parquet")
        return pd.concat(pieces, ignore_index=True), reference.loc[
            reference.cycle_name.isin(cohort.cycle_name)
        ].copy()

    @staticmethod
    def split_deployment_data(rows, training_cycles, test_cycles):
        """One final model: standard development cycles versus untouched later experiments."""
        training_cycles, test_cycles = set(training_cycles), set(test_cycles)
        if training_cycles & test_cycles:
            raise ValueError("Deployment training and test cycles overlap")
        requested_cycles = training_cycles | test_cycles
        result = rows.loc[rows.cycle_name.isin(requested_cycles)].copy()
        if set(result.cycle_name) != requested_cycles:
            raise ValueError("Deployment cohort includes unpublished camera cycles")
        result["split"] = result.cycle_name.map(
            lambda name: "train" if name in training_cycles else "test"
        )
        training_experiments = set(result.loc[result.split.eq("train"), "experiment_id"])
        test_experiments = set(result.loc[result.split.eq("test"), "experiment_id"])
        if training_experiments & test_experiments:
            raise ValueError("Deployment training and test experiments overlap")
        return result

    @staticmethod
    def inner_folds(conditions, test, n_splits=2, coverage="all"):
        """Experiment-disjoint folds preserving setpoint coverage; metadata only."""
        from sklearn.model_selection import StratifiedGroupKFold

        pool = conditions.loc[conditions.experiment_id.ne(test)].reset_index(drop=True)
        labels = pool.water_temperature_setpoint.to_numpy()
        if coverage not in {"all", "repeated"}:
            raise ValueError("Unknown inner setpoint coverage rule")
        required = set(labels)
        if coverage == "repeated":
            counts = pool.groupby("water_temperature_setpoint").experiment_id.nunique()
            required = set(counts.index[counts.ge(n_splits)])
        for seed in range(100):
            splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
            splits = list(splitter.split(pool, labels, pool.experiment_id))
            if all(
                required.issubset(set(labels[train])) and required.issubset(set(labels[val]))
                for train, val in splits
            ):
                return [sorted(pool.experiment_id.iloc[val].unique()) for _, val in splits]
        raise ValueError(f"{n_splits} inner folds cannot preserve setpoint coverage")

    @staticmethod
    def training_folds(
        rows,
        *,
        nested=False,
        validation_count=5,
        cohort="reviewer",
        validation_rule="chronological",
        inner_setpoint_coverage="all",
    ):
        experiments = sorted(rows.experiment_id.unique())
        if len(experiments) < 3:
            raise ValueError("training, validation and test need at least three experiments")
        eligible = rows.evaluation_eligible if cohort == "reviewer" else rows.training_eligible
        tests = sorted(rows.loc[eligible, "experiment_id"].unique())
        if len(tests) < 2:
            raise ValueError("At least two reviewer experiments are needed for validation and test")
        if nested and len(tests) < 3:
            raise ValueError(
                "Nested two-fold COP validation needs at least three reviewer experiments"
            )
        if nested or validation_rule == "stratified":
            conditions = (
                rows.groupby(["cycle_name", "experiment_id"])
                .water_temperature_setpoint.median()
                .reset_index()
            )
            if nested:
                return {
                    test: DatasetLoader.inner_folds(
                        conditions, test, coverage=inner_setpoint_coverage
                    )
                    for test in tests
                }
            if cohort != "training":
                raise ValueError(
                    "Stratified holdout currently requires training-cohort diagnostics"
                )
            return {
                test: DatasetLoader.inner_folds(conditions, test, n_splits=5)[0] for test in tests
            }
        count = min(validation_count, len(experiments) - 2)
        folds = {}
        for test in tests:
            ordered = [
                experiments[(experiments.index(test) + i + 1) % len(experiments)]
                for i in range(len(experiments) - 1)
            ]
            # Validation must contain a reviewer experiment; train-only cycles in it remain excluded.
            reviewers = [name for name in ordered if name in tests]
            if not reviewers:
                raise ValueError(
                    "At least two reviewer experiments are needed for validation and test"
                )
            folds[test] = [reviewers[0], *[name for name in ordered if name != reviewers[0]]][
                :count
            ]
        return folds

    @staticmethod
    def split_training_data(rows, test, validation, *, cohort="reviewer"):
        """A held-out experiment contributes no training-only cycles to fitting."""
        result = rows.copy()
        result["split"] = "train"
        held = result.experiment_id.isin([test, *validation])
        result.loc[held, "split"] = "excluded"
        eligible = result.evaluation_eligible if cohort == "reviewer" else result.training_eligible
        result.loc[eligible & result.experiment_id.isin(validation), "split"] = "validation"
        result.loc[eligible & result.experiment_id.eq(test), "split"] = "test"
        return result

    def load_image_metadata(self, cycle_name: str | None = None) -> pd.DataFrame:
        if self._image_metadata is None:
            metadata_path = image_root(self.dataset_root, self._manifest) / "image_metadata.parquet"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Dataset image metadata is missing: {metadata_path}")
            self._image_metadata = pd.read_parquet(metadata_path)
        result = self._image_metadata.copy()
        if cycle_name is not None:
            result = result.loc[result["cycle_name"].astype(str).eq(cycle_name)]
        return result.reset_index(drop=True)

    def load_cycle_images(self, cycle_name: str) -> pd.DataFrame:
        return scan_cycle_images(
            self.dataset_root,
            cycle_name,
            self.load_image_metadata(),
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
