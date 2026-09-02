"""Compare already-calculated cost curves without changing their science."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from dataloader import DatasetLoader

from .style import apply_style, prepare_output, save_publication

Run = tuple[str, str, pd.DataFrame, Path]
SourceMap = dict[str, str]


def _version(recipe: dict[str, object]) -> str:
    base = str(recipe["base_cost"])
    variant = recipe.get("variant")
    return base if variant in (None, "") else f"{base} ({variant})"


def _load(result_dirs: Sequence[Path]) -> list[Run]:
    runs: list[Run] = []
    for directory in result_dirs:
        recipe = json.loads((directory / "recipe.json").read_text(encoding="utf-8"))
        basis = str(recipe.get("heat_basis"))
        if basis not in {"unit", "water"}:
            raise ValueError(f"result has no explicit heat basis: {directory}")
        runs.append((_version(recipe), basis, pd.read_csv(directory / "cost.csv"), directory))
    return runs


def _connected_interval(curve: pd.DataFrame, optimum: int) -> tuple[pd.Series, pd.Series]:
    near = (
        curve["optimization_eligible"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(curve["relative_regret"], errors="coerce"))
        & pd.to_numeric(curve["relative_regret"], errors="coerce").le(0.01)
    ).to_numpy()
    left = right = optimum
    while left and near[left - 1]:
        left -= 1
    while right + 1 < len(curve) and near[right + 1]:
        right += 1
    return curve.iloc[left], curve.iloc[right]


def _optima(runs: list[Run], metadata: pd.DataFrame, dataset: Path, camera: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for version, basis, table, source in runs:
        for cycle, values in table.groupby("cycle_name", sort=True):
            curve = values.sort_values("candidate_elapsed_minutes", kind="stable").reset_index(drop=True)
            regret = pd.to_numeric(curve["relative_regret"], errors="coerce")
            eligible = curve["optimization_eligible"].fillna(False).astype(bool) & np.isfinite(regret)
            if not eligible.any():
                continue
            optimum = int(regret.loc[eligible].idxmin())
            start, end = _connected_interval(curve, optimum)
            selected = curve.iloc[optimum]
            row: dict[str, object] = {
                "source": str(source),
                "version": version,
                "cycle": str(cycle),
                "heat_basis": basis,
                "optimum_time": selected["candidate_time"],
                "optimum_elapsed_minutes": selected["candidate_elapsed_minutes"],
                "interval_start_time": start["candidate_time"],
                "interval_end_time": end["candidate_time"],
                "interval_start_elapsed_minutes": start["candidate_elapsed_minutes"],
                "interval_end_elapsed_minutes": end["candidate_elapsed_minutes"],
                "interval_width_minutes": (
                    float(end["candidate_elapsed_minutes"]) - float(start["candidate_elapsed_minutes"])
                ),
                "image_path": pd.NA,
                "image_time": pd.NaT,
                "image_delta_seconds": np.nan,
            }
            images = metadata.loc[
                metadata["cycle_name"].astype(str).eq(str(cycle))
                & metadata["camera_role"].astype(str).eq(camera)
            ].copy()
            if not images.empty:
                images["image_time"] = pd.to_datetime(images["image_time"], errors="coerce")
                images["image_path"] = images["file_name"].map(
                    lambda name: dataset / "images" / str(cycle) / camera / str(name)
                )
                images = images.loc[images["image_time"].notna() & images["image_path"].map(Path.is_file)]
            if not images.empty:
                target = pd.Timestamp(selected["candidate_time"])
                images["delta"] = (images["image_time"] - target).abs()
                image = images.sort_values(["delta", "image_time"], kind="stable").iloc[0]
                row.update(
                    image_path=str(Path(image["image_path"]).relative_to(dataset)),
                    image_time=image["image_time"],
                    image_delta_seconds=image["delta"].total_seconds(),
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _point(optima: pd.DataFrame, source: Path, cycle: str) -> pd.Series | None:
    rows = optima.loc[optima["source"].eq(str(source)) & optima["cycle"].eq(cycle)]
    return None if rows.empty else rows.iloc[0]


def _cycle_figure(
    cycle: str, runs: list[Run], optima: pd.DataFrame, colors: SourceMap, labels: SourceMap
) -> Figure:
    import matplotlib.pyplot as plt

    bases = sorted({basis for _, basis, table, _ in runs if cycle in set(table["cycle_name"])})
    fig, axes = plt.subplots(1 + len(bases), 1, figsize=(7, 2.6 * (1 + len(bases))))
    axes = np.atleast_1d(axes)
    for _, basis, table, source in runs:
        values = table.loc[table["cycle_name"].astype(str).eq(cycle)].sort_values(
            "candidate_elapsed_minutes", kind="stable"
        )
        if values.empty:
            continue
        point = _point(optima, source, cycle)
        color = colors[str(source)]
        axes[0].plot(
            values["candidate_elapsed_minutes"], values["relative_regret"], label=labels[str(source)], color=color
        )
        if point is not None:
            axes[0].scatter(point["optimum_elapsed_minutes"], 0, color=color, zorder=3)
            axes[0].axvspan(
                point["interval_start_elapsed_minutes"],
                point["interval_end_elapsed_minutes"],
                alpha=0.12,
                color=color,
            )
        for axis, required_basis in zip(axes[1:], bases, strict=True):
            if basis != required_basis:
                continue
            axis.plot(
                values["candidate_elapsed_minutes"], values["inverse_cop"], label=labels[str(source)], color=color
            )
            if point is not None:
                axis.scatter(
                    point["optimum_elapsed_minutes"],
                    values.loc[
                        values["candidate_elapsed_minutes"].eq(point["optimum_elapsed_minutes"]),
                        "inverse_cop",
                    ].iloc[0],
                    color=color,
                    zorder=3,
                )
                axis.axvspan(
                    point["interval_start_elapsed_minutes"],
                    point["interval_end_elapsed_minutes"],
                    alpha=0.12,
                    color=color,
                )
    axes[0].set(title=f"{cycle} — relative regret", ylabel="Relative regret")
    axes[0].legend()
    for axis, basis in zip(axes[1:], bases, strict=True):
        axis.set(title=f"Absolute inverse COP — {basis} heat basis", ylabel="Inverse COP")
        axis.legend()
    axes[-1].set_xlabel("Candidate elapsed time (min)")
    fig.tight_layout()
    return cast(Figure, fig)


def _comparison_figure(
    optima: pd.DataFrame, colors: SourceMap, labels: SourceMap
) -> Figure:
    import matplotlib.pyplot as plt

    cycles = sorted(optima["cycle"].unique())
    sources = list(optima["source"].unique())
    offsets = {
        str(source): 0.4 * (index - (len(sources) - 1) / 2) / max(1, len(sources) - 1)
        for index, source in enumerate(sources)
    }
    fig, axis = plt.subplots(figsize=(min(14, max(7.2, 0.19 * len(cycles))), 4))
    for source, rows in optima.groupby("source", sort=False):
        x = np.array([cycles.index(cycle) for cycle in rows["cycle"]]) + offsets[str(source)]
        lower = rows["optimum_elapsed_minutes"] - rows["interval_start_elapsed_minutes"]
        upper = rows["interval_end_elapsed_minutes"] - rows["optimum_elapsed_minutes"]
        axis.errorbar(
            x,
            rows["optimum_elapsed_minutes"],
            yerr=np.vstack((lower.to_numpy(), upper.to_numpy())),
            fmt="o",
            color=colors[str(source)],
            label=labels[str(source)],
        )
    axis.set(
        xticks=range(len(cycles)),
        xticklabels=cycles,
        ylabel="Optimum elapsed time (min)",
        title="Cost-function optima and connected 1% intervals",
    )
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    axis.legend()
    fig.tight_layout()
    return cast(Figure, fig)


def _images(cycle: str, optima: pd.DataFrame, dataset: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = optima.loc[optima["cycle"].eq(cycle) & optima["image_path"].notna()]
    if rows.empty:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(3 * len(rows), 3))
    for axis, (_, row) in zip(np.atleast_1d(axes), rows.iterrows(), strict=True):
        axis.imshow(plt.imread(dataset / str(row["image_path"])))
        axis.set_title(
            f"{row['version']}\n{row['optimum_time']} (Δ {row['image_delta_seconds']:.0f} s)",
            fontsize=8,
        )
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / f"{cycle}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compare_results(
    result_dirs: Sequence[Path], dataset: Path, output: Path, *, camera: str, overwrite: bool = False
) -> pd.DataFrame:
    """Write one comparison bundle from cost.csv files and local Dataset images."""
    prepare_output(output, overwrite)
    apply_style()
    runs = _load(result_dirs)
    colors = {str(source): f"C{index}" for index, (*_, source) in enumerate(runs)}
    versions = {str(source): version for version, _, _, source in runs}
    names = list(versions.values())
    labels = {
        source: version if names.count(version) == 1 else f"{version} ({Path(source).name})"
        for source, version in versions.items()
    }
    metadata = DatasetLoader(dataset).load_image_metadata()
    optima = _optima(runs, metadata, dataset, camera)
    if optima.empty:
        raise ValueError("no finite eligible cost optima")
    optima.to_csv(output / "optima.csv", index=False)
    save_publication(_comparison_figure(optima, colors, labels), output / "optimum_comparison")
    cycles, images = output / "cycles", output / "images"
    cycles.mkdir()
    images.mkdir()
    for cycle in sorted({str(value) for _, _, table, _ in runs for value in table["cycle_name"]}):
        save_publication(_cycle_figure(cycle, runs, optima, colors, labels), cycles / cycle)
        _images(cycle, optima, dataset, images)
    return optima
