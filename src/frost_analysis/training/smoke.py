"""Small reusable pieces for the local RGB model smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_MAXIMUM_PER_GROUP = 48


def selected_names(requested: str, available: tuple[str, ...]) -> tuple[str, ...]:
    """Return all model names or the one explicitly requested."""
    return available if requested == "all" else (requested,)


def even_sample_groups(
    frame: pd.DataFrame,
    group_columns: list[str],
    *,
    maximum_per_group: int,
) -> pd.DataFrame:
    """Keep evenly spaced rows from each ordered group."""
    sampled: list[pd.DataFrame] = []
    for _, group in frame.sort_values("image_time").groupby(
        group_columns, sort=True, observed=True
    ):
        positions = np.linspace(
            0,
            len(group) - 1,
            min(len(group), maximum_per_group),
            dtype=int,
        )
        sampled.append(group.iloc[positions])
    return pd.concat(sampled, ignore_index=True)


def image_color_features(path: Path) -> np.ndarray:
    """Return a compact color-and-gradient descriptor for one image."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB").resize((64, 36)), dtype=np.float32) / 255
    histograms = [
        np.histogram(pixels[..., channel], bins=8, range=(0, 1), density=True)[0] / 8
        for channel in range(3)
    ]
    grey = pixels.mean(axis=2)
    dx = np.abs(np.diff(grey, axis=1))
    dy = np.abs(np.diff(grey, axis=0))
    return np.concatenate(
        [
            *histograms,
            pixels.mean(axis=(0, 1)),
            pixels.std(axis=(0, 1)),
            [dx.mean(), dx.std(), dy.mean(), dy.std()],
        ]
    ).astype(np.float32)


def image_feature_matrix(
    rows: pd.DataFrame,
    role_order: tuple[str, ...],
) -> tuple[np.ndarray, list[int], pd.DataFrame]:
    """Extract shared color features and record unreadable images once."""
    role_eye = np.eye(len(role_order), dtype=np.float32)
    role_index = {name: index for index, name in enumerate(role_order)}
    features: list[np.ndarray] = []
    good_positions: list[int] = []
    errors: list[dict[str, object]] = []
    for position, row in enumerate(rows.itertuples(index=False)):
        try:
            color = image_color_features(Path(row.absolute_path))
        except OSError as error:
            errors.append(
                {
                    "cycle_name": row.cycle_name,
                    "camera_role": row.camera_role,
                    "file_name": row.file_name,
                    "error": str(error),
                }
            )
            continue
        features.append(np.concatenate([color, role_eye[role_index[row.camera_role]]]))
        good_positions.append(position)
        if (position + 1) % 500 == 0:
            print(f"[features] {position + 1}/{len(rows)}", flush=True)
    width = 34 + len(role_order)
    matrix = np.stack(features) if features else np.empty((0, width), dtype=np.float32)
    return matrix, good_positions, pd.DataFrame(errors)


def cycle_feature_shard(
    rows: pd.DataFrame,
    cycle_dir: Path,
    role_order: tuple[str, ...],
    *,
    maximum_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample one cycle and persistable features without retaining its images."""
    rows = rows.copy()
    rows["absolute_path"] = rows.apply(
        lambda row: str(cycle_dir / row["camera_role"] / row["file_name"]), axis=1
    )
    sampled = even_sample_groups(
        rows,
        ["cost_state", "camera_role"],
        maximum_per_group=maximum_per_group,
    )
    features, good_positions, excluded = image_feature_matrix(sampled, role_order)
    shard = sampled.iloc[good_positions].drop(columns="absolute_path").reset_index(drop=True)
    for index in range(features.shape[1]):
        shard[f"feature_{index:03d}"] = features[:, index]
    return shard, excluded
