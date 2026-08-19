from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from frost_analysis.dataset_loader import DatasetLoader
from frost_analysis.evidence import EvidenceSettings


def settings(
    *,
    minimum_feature_points: int = 3,
    minimum_feature_coverage: float = 0.8,
    minimum_valid_pairs: int = 2,
    minimum_pair_coverage: float = 0.8,
    targets: tuple[str, ...] = ("heating_capacity", "cop"),
    eligible_statuses: tuple[str, ...] = ("valid",),
    minimum_cycle_minutes: float = 0.0,
) -> EvidenceSettings:
    return EvidenceSettings(
        targets=targets,
        primary_target=targets[0],
        minimum_feature_points=minimum_feature_points,
        minimum_feature_coverage=minimum_feature_coverage,
        minimum_valid_pairs=minimum_valid_pairs,
        minimum_pair_coverage=minimum_pair_coverage,
        eligible_statuses=eligible_statuses,
        minimum_cycle_minutes=minimum_cycle_minutes,
    )


def frame_for(
    *,
    elapsed: Iterable[float] = (0, 60, 120, 180, 240, 300),
    stage: Iterable[str] | None = None,
    feature_a: Iterable[float] | None = None,
    feature_b: Iterable[float] | None = None,
    heating_capacity: Iterable[float] | None = None,
    cop: Iterable[float] | None = None,
) -> pd.DataFrame:
    elapsed_values = np.asarray(list(elapsed), dtype=float)
    count = len(elapsed_values)
    stage_values = list(stage) if stage is not None else ["frost_development"] * count
    values_a = (
        np.asarray(list(feature_a), dtype=float)
        if feature_a is not None
        else np.linspace(1.0, float(count), count)
    )
    values_b = (
        np.asarray(list(feature_b), dtype=float)
        if feature_b is not None
        else np.linspace(float(count), 1.0, count)
    )
    target_values = (
        np.asarray(list(heating_capacity), dtype=float)
        if heating_capacity is not None
        else np.linspace(10.0, 5.0, count)
    )
    cop_values = (
        np.asarray(list(cop), dtype=float)
        if cop is not None
        else np.linspace(4.0, 2.0, count)
    )
    progress = np.full(count, np.nan)
    timestamps = pd.Timestamp("2026-07-01T00:00:00") + pd.to_timedelta(
        elapsed_values, unit="s"
    )
    frost = np.asarray(stage_values, dtype=object) == "frost_development"
    if frost.any() and np.nanmax(elapsed_values[frost]) > np.nanmin(elapsed_values[frost]):
        progress[frost] = (
            elapsed_values[frost] - np.nanmin(elapsed_values[frost])
        ) / (np.nanmax(elapsed_values[frost]) - np.nanmin(elapsed_values[frost]))
    return pd.DataFrame(
        {
            "cycle_stage": stage_values,
            "timestamp": timestamps,
            "cycle_elapsed_seconds": elapsed_values,
            "cycle_progress": progress,
            "feature_a__baseline_residual": values_a,
            "feature_a__imputed": False,
            "feature_b__baseline_residual": values_b,
            "feature_b__imputed": False,
            "heating_capacity__baseline_residual": target_values,
            "heating_capacity": target_values,
            "heating_capacity__baseline": float(target_values[0]),
            "heating_capacity__imputed": False,
            "cop__baseline_residual": cop_values,
            "cop": cop_values,
            "cop__baseline": float(cop_values[0]),
            "cop__imputed": False,
            "ambient_temperature": 5.0,
            "ambient_temperature__imputed": False,
            "environment_relative_humidity": 80.0,
            "environment_relative_humidity__imputed": False,
            "water_in_temperature": 35.0,
            "water_in_temperature__imputed": False,
            "water_flow": 1.0,
            "water_flow__imputed": False,
            "compressor_frequency": 50.0,
            "compressor_frequency__imputed": False,
        }
    )


def write_dataset(
    root: Path,
    cycles: Iterable[tuple[str, str, str, pd.DataFrame]],
) -> DatasetLoader:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cycles").mkdir()
    (root / "cycles_original").mkdir()
    (root / "images").mkdir()
    cycle_records = []
    experiment_rows: dict[tuple[str, str], None] = {}
    registry = {
        "registry_version": 1,
        "channels": {
            "feature_a": {
                "analysis_candidate": True,
                "expected_frost_direction": "increase",
            },
            "feature_b": {
                "analysis_candidate": True,
                "expected_frost_direction": "decrease",
            },
            "heating_capacity": {"analysis_candidate": False},
            "cop": {"analysis_candidate": False},
        },
        "fields": [],
    }
    for cycle_name, experiment_date, status, frame in cycles:
        cycle_uid = f"exp::{cycle_name}"
        frame.to_parquet(root / "cycles" / f"{cycle_name}.parquet", index=False)
        frame.to_csv(root / "cycles_original" / f"{cycle_name}.csv", index=False)
        (root / "cycles" / f"{cycle_name}.png").write_bytes(b"publication")
        (root / "cycles" / f"{cycle_name}_rgb_coverage.png").write_bytes(b"coverage")
        (root / "images" / cycle_name).mkdir()
        cycle_records.append(
            {
                "cycle_name": cycle_name,
                "cycle_uid": cycle_uid,
                "experiment_id": "exp",
                "experiment_date": experiment_date,
                "status": status,
                "boundaries": {
                    "stable_heating_start": pd.to_datetime(frame["timestamp"])
                    .min()
                    .isoformat(),
                    "baseline_end": (
                        pd.to_datetime(frame["timestamp"]).min()
                        + pd.Timedelta(seconds=60)
                    ).isoformat(),
                    "defrost_start": (
                        pd.to_datetime(frame["timestamp"]).max() + pd.Timedelta(seconds=10)
                    ).isoformat(),
                    "end_time": (
                        pd.to_datetime(frame["timestamp"]).max() + pd.Timedelta(seconds=10)
                    ).isoformat(),
                },
                "assets": {
                    "parquet": f"cycles/{cycle_name}.parquet",
                    "csv": f"cycles/{cycle_name}.csv",
                    "original_csv": f"cycles_original/{cycle_name}.csv",
                    "publication": f"cycles/{cycle_name}.png",
                    "rgb_coverage": f"cycles/{cycle_name}_rgb_coverage.png",
                },
            }
        )
        experiment_rows[("exp", experiment_date)] = None
    for record in cycle_records:
        frame = pd.read_parquet(root / str(record["assets"]["parquet"]))
        frame.to_csv(root / str(record["assets"]["csv"]), index=False)
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_schema_version": 3,
                "dataset_id": "frost_cycle_dataset",
                "experiments": [
                    {"experiment_id": exp, "experiment_date": date}
                    for exp, date in sorted(experiment_rows)
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "cycle_catalog.json").write_text(
        json.dumps({"cycles": cycle_records}), encoding="utf-8"
    )
    (root / "channel_registry.json").write_text(json.dumps(registry), encoding="utf-8")
    pd.DataFrame(columns=["cycle_name"]).to_parquet(
        root / "image_metadata.parquet", index=False
    )
    return DatasetLoader(root)
