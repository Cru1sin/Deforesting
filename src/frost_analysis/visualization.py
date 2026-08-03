"""Pure renderers for self-contained cycle Dataset artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .dataset_loader import DatasetLoader


def render_cycle_publication(
    cycle_frame: pd.DataFrame,
    cycle_record: Mapping[str, object],
    output_path: Path,
) -> None:
    """Render Dataset values through the existing publication renderer."""
    from .report import _plot_one_cycle_publication

    report_cycle = pd.Series(
        {
            **dict(cycle_record),
            "cycle_id": cycle_record.get("source_cycle_id", cycle_record.get("cycle_id")),
        }
    )
    _plot_one_cycle_publication(
        cycle_frame,
        cycle_frame,
        report_cycle,
        output_path,
        [],
        processed_only=True,
        include_humidity=True,
    )


def generate_cycle_publication(loader: DatasetLoader, cycle_name: str) -> Path:
    """Load one cycle through DatasetLoader and write its publication figure."""
    from .dataset_loader import DatasetLoader
    from .dataset_manifest import refresh_cycle_asset_hashes

    if not isinstance(loader, DatasetLoader):
        raise TypeError("generate_cycle_publication requires DatasetLoader")
    path = loader.publication_path(cycle_name)
    render_cycle_publication(
        loader.load_cycle(cycle_name), loader.get_cycle_record(cycle_name), path
    )
    refresh_cycle_asset_hashes(loader.dataset_root, cycle_name)
    return path


def generate_rgb_coverage(loader: DatasetLoader, cycle_name: str) -> Path:
    """Load one cycle and its current role folders through DatasetLoader."""
    from .dataset_loader import DatasetLoader
    from .dataset_manifest import refresh_cycle_asset_hashes

    if not isinstance(loader, DatasetLoader):
        raise TypeError("generate_rgb_coverage requires DatasetLoader")
    path = loader.rgb_coverage_path(cycle_name)
    if loader.schema_version == 3:
        from .dataset_coverage_v3 import render_rgb_coverage

        render_rgb_coverage(
            loader.load_cycle(cycle_name),
            loader.load_cycle_images(cycle_name),
            loader.get_cycle_record(cycle_name),
            path,
            registry=loader.registry,
        )
    else:
        from .dataset_coverage import render_rgb_coverage as render_legacy_rgb_coverage

        render_legacy_rgb_coverage(
            loader.load_cycle(cycle_name),
            loader.load_cycle_images(cycle_name),
            loader.get_cycle_record(cycle_name),
            path,
        )
    refresh_cycle_asset_hashes(loader.dataset_root, cycle_name)
    return path
