#!/usr/bin/env python3
"""Compare defrost cost optima and render cycle publication PNGs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frost_analysis.dataset_loader import DatasetLoader
from frost_analysis.visualization import (
    match_decision_rgb_images,
    render_decision_publication,
)

STYLES = {
    "v1": ("#2166AC", "D", "V1 optimum"),
    "v2": ("#D97706", "s", "V2 optimum"),
    "RB": ("#2E7D5B", "o", "Rule defrost"),
}
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    }
)


def _save_png(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _read_tables(sources: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for label, path in sources.items():
        table = pd.read_csv(path)
        selected = table["valid"].fillna(False)
        if "is_censored" in table:
            selected &= ~table["is_censored"].fillna(False)
        table = table.loc[selected].copy()
        algorithm = str(table["algorithm"].dropna().iloc[0]).lower()
        tables[algorithm] = table
        tables[algorithm].attrs["label"] = str(label)
    return tables


def _cycle_points(table: pd.DataFrame, optimum_column: str = "t_star") -> pd.DataFrame:
    values = table.copy()
    for column in ("candidate_time", "t_heating_stable", optimum_column, "t_RB"):
        values[column] = pd.to_datetime(
            values[column], errors="coerce", format="mixed"
        )
    rows = []
    for cycle_name, cycle in values.groupby("cycle_name", sort=True):
        first = cycle.iloc[0]
        stable = first["t_heating_stable"]
        rows.append(
            {
                "cycle_name": str(cycle_name),
                "length_minutes": (
                    cycle["candidate_time"].max() - stable
                ).total_seconds()
                / 60,
                "optimum_minutes": (
                    first[optimum_column] - stable
                ).total_seconds()
                / 60,
                "rb_minutes": (
                    (first["t_RB"] - stable).total_seconds() / 60
                    if first.get("rb_status") == "triggered" and pd.notna(first["t_RB"])
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("cycle_name")


def _comparison_figure(
    tables: Mapping[str, pd.DataFrame], algorithms: tuple[str, ...]
) -> plt.Figure:
    points = {algorithm: _cycle_points(tables[algorithm]) for algorithm in algorithms}
    cycles = sorted(set().union(*(set(values.index) for values in points.values())))
    cycle_ids = [int(cycle.rsplit("_", 1)[-1]) for cycle in cycles]
    y = np.asarray(cycle_ids)
    figure, axis = plt.subplots(figsize=(7.2, max(2.4, 0.25 * len(cycles) + 1.2)))
    lengths = pd.concat(
        [points[name].reindex(cycles)["length_minutes"] for name in algorithms], axis=1
    ).max(axis=1)
    axis.barh(y, lengths, height=0.72, color="#E5E7EB", edgecolor="none", label="Candidate length")
    for offset, algorithm in zip(
        np.linspace(-0.14, 0.14, len(algorithms)), algorithms, strict=True
    ):
        color, marker, label = STYLES[algorithm]
        axis.scatter(
            points[algorithm].reindex(cycles)["optimum_minutes"],
            y + offset,
            color=color,
            marker=marker,
            s=24,
            label=label,
            zorder=3,
        )
    rb = points[algorithms[0]].reindex(cycles)["rb_minutes"]
    color, marker, label = STYLES["RB"]
    axis.scatter(rb, y, color=color, marker=marker, facecolors="none", s=28, label=label, zorder=4)
    axis.set(
        xlabel="Minutes from stable heating start",
        ylabel="Cycle index",
        yticks=y,
        yticklabels=cycle_ids,
    )
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.6, alpha=0.7)
    axis.legend(frameon=False, ncols=min(4, len(algorithms) + 2), fontsize=7)
    figure.tight_layout()
    return figure


def _publication_curve(table: pd.DataFrame, label: str) -> pd.DataFrame:
    curve = table.copy()
    if label == "water_reference":
        curve["inverse_cop"] = curve["water_reference_inverse_cop"]
        curve["relative_regret"] = curve["water_reference_relative_regret"]
    return curve


def _decision_images(
    metadata: pd.DataFrame, images: pd.DataFrame, curve: pd.DataFrame
) -> dict[str, dict[str, object]]:
    eligible = curve["optimization_eligible"].fillna(False)
    optimum = curve.loc[
        pd.to_numeric(curve["inverse_cop"], errors="coerce").where(eligible).idxmin(),
        "candidate_time",
    ]
    first = curve.iloc[0]
    rb = first["t_RB"] if first.get("rb_status") == "triggered" else pd.NaT
    matches = match_decision_rgb_images(
        metadata,
        images,
        {"rb": rb, "optimal": optimum},
    )
    return {
        str(row["target_type"]): row.to_dict() for _, row in matches.iterrows()
    }


def _render_cycle_sets(
    tables: Mapping[str, pd.DataFrame], loader: DatasetLoader, output: Path
) -> None:
    titles = {
        "水侧制热量_cycle": "Water-heat optimum",
        "cost_function_v1_cycle": "Unit-heat V1 optimum",
        "cost_function_v2_cycle": "Updated V2 optimum",
    }
    suites: dict[str, pd.DataFrame] = {}
    if "v1" in tables:
        suites["水侧制热量_cycle"] = _publication_curve(
            tables["v1"], "water_reference"
        )
        suites["cost_function_v1_cycle"] = _publication_curve(tables["v1"], "v1")
    if "v2" in tables:
        suites["cost_function_v2_cycle"] = _publication_curve(tables["v2"], "v2")
    cycles = sorted(set().union(*(set(table["cycle_name"]) for table in suites.values())))
    for cycle_name in cycles:
        cycle_name = str(cycle_name)
        record = loader.get_cycle_record(cycle_name)
        frame = loader.load_cycle(cycle_name)
        metadata = loader.load_image_metadata(cycle_name)
        images = loader.load_cycle_images(cycle_name)
        for label, table in suites.items():
            curve = table.loc[table["cycle_name"].eq(cycle_name)]
            if curve.empty:
                continue
            filename = f"cycle_{int(cycle_name.rsplit('_', 1)[-1]):03d}_publication.png"
            render_decision_publication(
                frame,
                record,
                curve,
                _decision_images(metadata, images, curve),
                output / label / filename,
                optimal_label=titles[label],
                full_candidate_domain=True,
            )


def plot_cost_function_comparison(
    sources: Mapping[str, Path], *, output: Path
) -> None:
    """Plot the algorithms present in comprehensive CSV sources against RB."""
    tables = _read_tables(sources)
    algorithms = tuple(name for name in ("v1", "v2") if name in tables)
    _save_png(_comparison_figure(tables, algorithms), output)


def generate_cost_function_figures(
    sources: Mapping[str, Path], loader: DatasetLoader, output: Path
) -> None:
    """Read comprehensive cost CSVs and write comparison/publication PNGs."""
    tables = _read_tables(sources)
    for algorithm in ("v1", "v2"):
        if algorithm in tables:
            _save_png(
                _comparison_figure(tables, (algorithm,)),
                output / f"comparison_{algorithm}_RB.png",
            )
    if {"v1", "v2"}.issubset(tables):
        _save_png(
            _comparison_figure(tables, ("v1", "v2")),
            output / "comparison_v1_v2_RB.png",
        )
    _render_cycle_sets(tables, loader, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost", nargs="+", required=True, metavar="LABEL=CSV")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("output/成本函数"))
    args = parser.parse_args()
    sources = {label: Path(path) for label, path in (item.split("=", 1) for item in args.cost)}
    generate_cost_function_figures(sources, DatasetLoader(args.dataset), args.output)


if __name__ == "__main__":
    main()
