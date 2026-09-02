"""Prepare the one shared feature table used by frozen classifiers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from labels.build import CAMERA_GROUPS

CACHE_ROOT = Path("output/models/_cache")


def even_sample_groups(
    rows: pd.DataFrame, group_columns: list[str], *, maximum_per_group: int
) -> pd.DataFrame:
    if maximum_per_group < 1:
        raise ValueError("maximum_per_group must be positive")
    ordered = rows.reset_index(drop=True).copy()
    ordered["_source_position"] = np.arange(len(ordered))
    sampled: list[pd.DataFrame] = []
    for _, group in ordered.sort_values("image_time").groupby(
        group_columns, sort=True, observed=True
    ):
        positions = np.linspace(
            0, len(group) - 1, min(len(group), maximum_per_group), dtype=int
        )
        sampled.append(group.iloc[positions])
    if not sampled:
        return ordered.drop(columns="_source_position")
    return (
        pd.concat(sampled)
        .sort_values("_source_position")
        .drop(columns="_source_position")
        .reset_index(drop=True)
    )


def image_color_features(path: Path) -> np.ndarray:
    """Return the established 34-value color-and-gradient descriptor."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB").resize((64, 36)), dtype=np.float32) / 255
    histograms = [
        np.histogram(pixels[..., channel], bins=8, range=(0, 1), density=True)[0] / 8
        for channel in range(3)
    ]
    grey = pixels.mean(axis=2)
    dx = np.abs(np.diff(grey, axis=1))
    dy = np.abs(np.diff(grey, axis=0))
    result: np.ndarray = np.concatenate(
        [
            *histograms,
            pixels.mean(axis=(0, 1)),
            pixels.std(axis=(0, 1)),
            [dx.mean(), dx.std(), dy.mean(), dy.std()],
        ]
    ).astype(np.float32)
    return result


def _extract_handcrafted(
    rows: pd.DataFrame, absolute_paths: list[str], camera: str
) -> pd.DataFrame:
    roles = CAMERA_GROUPS[camera]
    role_index = {role: index for index, role in enumerate(roles)}
    role_eye: np.ndarray = np.eye(len(roles), dtype=np.float32)
    values = [
        np.concatenate(
            [
                image_color_features(Path(path)),
                role_eye[role_index[str(row.camera_role)]],
            ]
        )
        for row, path in zip(rows.itertuples(index=False), absolute_paths, strict=True)
    ]
    width = 34 + len(roles)
    matrix = np.stack(values) if values else np.empty((0, width), dtype=np.float32)
    result = pd.DataFrame({"absolute_path": absolute_paths})
    for index in range(width):
        result[f"feature_{index:03d}"] = matrix[:, index]
    return result


def prepare_features(
    rows: pd.DataFrame,
    *,
    dataset_root: Path,
    representation: str,
    camera: str,
    modality: str,
    state_column: str,
    maximum_per_group: int,
    cache_root: Path = CACHE_ROOT,
) -> tuple[pd.DataFrame, list[str]]:
    """Sample rows, reuse their image descriptor cache, and select one modality."""
    if representation != "handcrafted":
        raise ValueError(f"frozen features do not implement {representation}")
    sampled = even_sample_groups(
        rows,
        [state_column, "camera_role", "cycle_name"],
        maximum_per_group=maximum_per_group,
    )
    # Stable heating start is an observed physical boundary preceding each image;
    # this uses neither a future cycle end nor any sensor value.
    image_time = pd.to_datetime(sampled["image_time"], errors="raise", format="mixed")
    stable_start = pd.to_datetime(
        sampled["stable_heating_start"], errors="raise", format="mixed"
    )
    sampled["time_minutes"] = (image_time - stable_start).dt.total_seconds() / 60
    if modality == "time":
        return sampled.reset_index(drop=True), ["time_minutes"]

    cache_path = cache_root / representation / camera / "features.parquet"
    cached = pd.read_parquet(cache_path) if cache_path.is_file() else pd.DataFrame()
    absolute_paths = [
        str((dataset_root / str(path)).resolve()) for path in sampled["image_path"]
    ]
    feature_columns = [
        str(column) for column in cached.columns if str(column).startswith("feature_")
    ]
    reusable = (
        cached.get("absolute_path", pd.Series(dtype="string")).astype(str).tolist()
        == absolute_paths
        and len(feature_columns) == 34 + len(CAMERA_GROUPS[camera])
    )
    if not reusable:
        cached = _extract_handcrafted(sampled, absolute_paths, camera)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(cache_path, index=False)
        feature_columns = [
            str(column)
            for column in cached.columns
            if str(column).startswith("feature_")
        ]

    result = sampled.copy()
    result[feature_columns] = cached[feature_columns].to_numpy()
    selected_columns = (
        feature_columns if modality == "rgb" else [*feature_columns, "time_minutes"]
    )
    return result.reset_index(drop=True), selected_columns
