"""Plot existing RGB-label class counts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import apply_style, prepare_output, save_publication


def plot(balance: Path, output: Path, threshold: float, cameras: list[str], overwrite: bool) -> None:
    prepare_output(output, overwrite)
    table = pd.read_csv(balance)
    table = table.loc[table["regret_threshold"].eq(threshold) & table["camera_group"].isin(cameras)]
    counts = table.groupby(["camera_group", "cost_state"], as_index=False)["image_count"].sum()
    if counts.empty:
        raise ValueError("no label counts match the requested threshold and cameras")
    apply_style()
    fig, axis = plt.subplots(figsize=(max(6, len(cameras) * 1.5), 4))
    bottom = np.zeros(len(cameras))
    for state in sorted(counts["cost_state"].unique()):
        values = [
            counts.loc[
                counts["camera_group"].eq(camera) & counts["cost_state"].eq(state), "image_count"
            ].sum()
            for camera in cameras
        ]
        axis.bar(cameras, values, bottom=bottom, label=state)
        bottom += values
    axis.set(ylabel="Image count", title=f"Label balance at relative-regret threshold {threshold:g}")
    axis.legend(title="Cost state")
    fig.tight_layout()
    save_publication(fig, output / "label_balance")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-balance", type=Path, default=Path("output/labels/v1/label_balance.csv"))
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--cameras", nargs="+", default=["front", "all"])
    parser.add_argument("--output", type=Path, default=Path("output/plots/labels"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    plot(args.label_balance, args.output, args.threshold, args.cameras, args.overwrite)


if __name__ == "__main__":
    main()
