"""Shared publication export defaults."""

from __future__ import annotations

import shutil
from pathlib import Path

from matplotlib.figure import Figure


def apply_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"}
    )


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not overwrite)):
        raise FileExistsError(f"plot output exists; pass --overwrite: {output}")
    if output.is_dir() and overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def save_publication(fig: Figure, stem: Path) -> None:
    import matplotlib.pyplot as plt

    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
