"""Plot held-out macro-F1 values already written by model evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import apply_style, prepare_output, save_publication

SETTING = ("source", "representation", "head", "camera", "modality")


def _label(row: pd.Series[Any]) -> str:
    return "; ".join(f"{column}={row[column]}" for column in SETTING)


def plot(evaluation: Path, output: Path, overwrite: bool = False) -> None:
    prepare_output(output, overwrite)
    folds = pd.read_csv(evaluation / "experiment_metrics.csv")
    summary = pd.read_csv(evaluation / "summary.csv")
    settings = summary.loc[:, list(SETTING)].drop_duplicates().reset_index(drop=True)
    apply_style()
    fig, axis = plt.subplots(figsize=(8, max(3, len(settings) * 0.7)))
    for position, (_, setting) in enumerate(settings.iterrows()):
        selected = folds
        for column in SETTING:
            selected = selected.loc[selected[column].eq(setting[column])]
        values = pd.to_numeric(
            selected.loc[selected["status"].eq("ok"), "macro_f1"], errors="coerce"
        ).dropna()
        offsets = np.linspace(-0.08, 0.08, len(values), dtype=float)
        axis.scatter(
            values.to_numpy(),
            np.full(len(values), position, dtype=float) + offsets,
            color="C0",
            alpha=0.75,
            label="Held-out fold" if position == 0 else None,
        )
        row = summary.loc[
            (summary.loc[:, list(SETTING)] == setting.loc[list(SETTING)]).all(axis=1)
        ].iloc[0]
        mean = float(row["macro_f1_mean"])
        std = pd.to_numeric(row["macro_f1_std"], errors="coerce")
        axis.errorbar(mean, position, xerr=0 if pd.isna(std) else float(std), fmt="o", color="black", label="Experiment-equal mean ± std" if position == 0 else None)
    axis.set(
        yticks=range(len(settings)),
        yticklabels=[_label(row) for _, row in settings.iterrows()],
        xlabel="Held-out macro-F1",
    )
    axis.legend()
    fig.tight_layout()
    save_publication(fig, output / "model_macro_f1")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=Path("output/models/evaluation"))
    parser.add_argument("--output", type=Path, default=Path("output/plots/models"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    plot(args.evaluation, args.output, args.overwrite)


if __name__ == "__main__":
    main()
