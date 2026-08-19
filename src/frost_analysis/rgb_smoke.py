"""Small reusable pieces for the local RGB model smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


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
