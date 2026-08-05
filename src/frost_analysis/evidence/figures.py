"""Evidence figures built from the new Evidence tables and Loader API."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..dataset_loader import DatasetLoader
from .metrics import observed_mask
from .models import EvidenceBundle
from .settings import EvidenceSettings

FIGURE_NAMES = (
    "figure_1_cycle_progress",
    "figure_2_feature_profiles",
    "figure_3_future_horizon",
    "figure_s2_availability",
)


def plot_cycle_progress(loader: DatasetLoader, settings: EvidenceSettings) -> Figure:
    """Plot fixed cycle-progress bins, preserving Dataset ``cycle_progress``."""
    figure, axes = plt.subplots(
        len(settings.targets),
        1,
        figsize=(8, max(2.8, 2.8 * len(settings.targets))),
        squeeze=False,
        sharex=True,
    )
    flat_axes = list(axes[:, 0])
    edges = np.linspace(0.0, 1.0, 101)
    centers = (edges[:-1] + edges[1:]) / 2.0
    for axis, target in zip(flat_axes, settings.targets, strict=True):
        target_column = f"{target}__baseline_residual"
        quality_column = f"{target}__imputed"
        cycle_rows: list[pd.DataFrame] = []
        for record, frame in loader.iter_cycle_frames(statuses={"valid"}):
            if (
                "cycle_progress" not in frame
                or target_column not in frame
                or quality_column not in frame
            ):
                continue
            progress = cast(
                NDArray[np.float64],
                pd.to_numeric(frame["cycle_progress"], errors="coerce").to_numpy(
                    dtype=float, na_value=np.nan
                ),
            )
            values = cast(
                NDArray[np.float64],
                pd.to_numeric(frame[target_column], errors="coerce").to_numpy(
                    dtype=float, na_value=np.nan
                ),
            )
            observed = observed_mask(frame, target_column).to_numpy(dtype=bool)
            valid = (
                (progress >= 0.0)
                & (progress <= 1.0)
                & np.isfinite(progress)
                & np.isfinite(values)
                & observed
            )
            if not valid.any():
                continue
            progress_subset = [float(value) for value in progress[valid]]
            binned = pd.cut(
                progress_subset,
                bins=[float(value) for value in edges],
                labels=False,
                include_lowest=True,
                right=True,
            )
            grouped = (
                pd.DataFrame({"bin": binned, "value": values[valid]})
                .groupby("bin", sort=True)["value"]
                .median()
            )
            if grouped.empty:
                continue
            date = str(record.get("experiment_date", ""))[:10]
            cycle_rows.append(
                pd.DataFrame(
                    {
                        "experiment_date": date,
                        "bin": grouped.index.astype(int),
                        "value": grouped.to_numpy(dtype=float),
                    }
                )
            )
        if not cycle_rows:
            axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
            axis.set_title(target)
            axis.set_ylabel("residual")
            continue
        cycles = pd.concat(cycle_rows, ignore_index=True)
        by_date = cycles.groupby(["experiment_date", "bin"], sort=False)["value"].median()
        plotted = by_date.groupby(level="bin", sort=True).median()
        x = centers[plotted.index.to_numpy(dtype=int)]
        axis.plot(x, plotted.to_numpy(dtype=float), linewidth=1.8)
        axis.set_title(target)
        axis.set_ylabel("residual")
        axis.set_xlim(0.0, 1.0)
        axis.grid(alpha=0.2)
    flat_axes[-1].set_xlabel("cycle_progress")
    figure.tight_layout()
    return figure


def plot_feature_profiles(
    bundle: EvidenceBundle,
    loader: DatasetLoader,
    settings: EvidenceSettings,
) -> Figure:
    """Plot one feature row and four metric columns in registry order."""
    features = _candidate_feature_names(loader)
    rows = max(1, len(features))
    figure, axes = plt.subplots(
        rows,
        4,
        figsize=(12, max(2.5, rows * 1.35)),
        squeeze=False,
    )
    metric_columns = (
        "signed_effect",
        "trend_slope_per_min",
        "onset_minutes",
        "primary_future_effect",
    )
    metrics = bundle.feature_cycle_metrics
    profile = bundle.feature_profile
    for row_index, feature in enumerate(features):
        metric_rows = metrics.loc[
            metrics["feature"].eq(feature) & metrics["metric_status"].eq("available")
        ]
        profile_rows = profile.loc[profile["feature"].eq(feature)]
        for column_index, column in enumerate(metric_columns):
            axis = axes[row_index, column_index]
            if column == "primary_future_effect":
                values = (
                    pd.to_numeric(profile_rows[column], errors="coerce")
                    if not profile_rows.empty
                    else pd.Series(dtype=float)
                )
            else:
                values = (
                    pd.to_numeric(metric_rows[column], errors="coerce")
                    if not metric_rows.empty
                    else pd.Series(dtype=float)
                )
            values = values.loc[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]
            if values.empty:
                axis.text(0.5, 0.5, "Unavailable", ha="center", va="center", fontsize=8)
            else:
                axis.scatter(np.arange(len(values)), values.to_numpy(dtype=float), s=14)
                axis.axhline(float(values.median()), color="black", linewidth=0.8)
            axis.set_xticks([])
            axis.grid(alpha=0.15)
            if row_index == 0:
                axis.set_title(column)
            if column_index == 0:
                axis.set_ylabel(feature)
    if not features:
        axes[0, 0].text(0.5, 0.5, "Unavailable", ha="center", va="center")
    figure.tight_layout()
    return figure


def plot_future_horizon_summary(
    summary: pd.DataFrame, settings: EvidenceSettings
) -> Figure:
    """Plot only the already-aggregated future horizon summary."""
    features = list(dict.fromkeys(summary.get("feature", pd.Series(dtype=str)).astype(str)))
    if not features:
        features = ["Unavailable"]
    figure, axes = plt.subplots(
        len(settings.targets),
        1,
        figsize=(8, max(2.8, len(settings.targets) * 2.8)),
        squeeze=False,
    )
    for axis, target in zip(list(axes[:, 0]), settings.targets, strict=True):
        matrix = np.full((len(features), len(settings.horizons_minutes)), np.nan)
        for row_index, feature in enumerate(features):
            if feature == "Unavailable":
                continue
            for col_index, horizon in enumerate(settings.horizons_minutes):
                selected = summary.loc[
                    summary["feature"].eq(feature)
                    & summary["target"].eq(target)
                    & summary["horizon_minutes"].eq(horizon)
                ]
                if not selected.empty:
                    value = pd.to_numeric(selected.iloc[0]["effect"], errors="coerce")
                    if pd.notna(value):
                        matrix[row_index, col_index] = float(value)
                        axis.text(
                            col_index,
                            row_index,
                            f"{float(value):.2g}\n"
                            f"cycle={int(selected.iloc[0]['valid_cycle_count'])}\n"
                            f"date={int(selected.iloc[0]['valid_date_count'])}",
                            ha="center",
                            va="center",
                            fontsize=7,
                        )
        if np.isfinite(matrix).any():
            image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
            figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
            axis.set_yticks(np.arange(len(features)), labels=features)
            axis.set_xticks(
                np.arange(len(settings.horizons_minutes)),
                labels=[f"{value} min" for value in settings.horizons_minutes],
            )
        else:
            axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
        axis.set_title(target)
        axis.set_ylabel("feature")
    figure.tight_layout()
    return figure


def plot_availability_audit(bundle: EvidenceBundle) -> Figure:
    """Plot feature availability and future pair coverage on separate panels."""
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), squeeze=False)
    feature_axis, future_axis = axes[:, 0]
    eligibility = bundle.cycle_eligibility
    valid_cycles = eligibility.loc[eligibility["status"].eq("valid"), "cycle_name"].astype(str)
    metrics = bundle.feature_cycle_metrics
    features = list(dict.fromkeys(metrics.get("feature", pd.Series(dtype=str)).astype(str)))
    feature_matrix = np.zeros((len(valid_cycles), len(features)), dtype=float)
    for row_index, cycle_name in enumerate(valid_cycles):
        for col_index, feature in enumerate(features):
            selected = metrics.loc[
                metrics["cycle_name"].eq(cycle_name)
                & metrics["feature"].eq(feature)
                & metrics["metric_status"].eq("available")
            ]
            feature_matrix[row_index, col_index] = float(not selected.empty)
    if feature_matrix.size:
        feature_axis.imshow(feature_matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        feature_axis.set_xticks(np.arange(len(features)), labels=features, rotation=45, ha="right")
        feature_axis.set_yticks(np.arange(len(valid_cycles)), labels=valid_cycles)
    else:
        feature_axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
    feature_axis.set_title("S2 Dataset valid cycle audit — feature availability")
    feature_axis.set_ylabel("cycle")

    future = bundle.future_association
    future_columns = [
        f"{feature}|{target}|{horizon}min"
        for feature in dict.fromkeys(future.get("feature", pd.Series(dtype=str)).astype(str))
        for target in dict.fromkeys(future.get("target", pd.Series(dtype=str)).astype(str))
        for horizon in sorted(future.get("horizon_minutes", pd.Series(dtype=int)).dropna().unique())
    ]
    future_matrix = np.full((len(valid_cycles), len(future_columns)), np.nan)
    for col_index, label in enumerate(future_columns):
        feature, target, horizon_text = label.split("|")
        horizon = int(horizon_text.removesuffix("min"))
        selected = future.loc[
            future["feature"].eq(feature)
            & future["target"].eq(target)
            & future["horizon_minutes"].eq(horizon)
            & future["cycle_name"].isin(valid_cycles)
        ]
        values = (
            selected.set_index("cycle_name")["pair_coverage"]
            if not selected.empty
            else pd.Series(dtype=float)
        )
        for row_index, cycle_name in enumerate(valid_cycles):
            if cycle_name in values.index:
                future_matrix[row_index, col_index] = float(values.loc[cycle_name])
    if future_matrix.size:
        future_axis.imshow(
            future_matrix,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        future_axis.set_xticks(
            np.arange(len(future_columns)), labels=future_columns, rotation=45, ha="right"
        )
        future_axis.set_yticks(np.arange(len(valid_cycles)), labels=valid_cycles)
    else:
        future_axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
    future_axis.set_title("S2 Dataset valid cycle audit — future pair coverage")
    future_axis.set_ylabel("cycle")
    figure.tight_layout()
    return figure


def write_figures(
    output_dir: Path,
    bundle: EvidenceBundle,
    loader: DatasetLoader,
    settings: EvidenceSettings,
) -> tuple[str, ...]:
    """Write the four required figures as PNG and PDF files."""
    figures = (
        plot_cycle_progress(loader, settings),
        plot_feature_profiles(bundle, loader, settings),
        plot_future_horizon_summary(bundle.future_horizon_summary, settings),
        plot_availability_audit(bundle),
    )
    files: list[str] = []
    for name, figure in zip(FIGURE_NAMES, figures, strict=True):
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            figure.savefig(output_dir / filename)
            files.append(filename)
        plt.close(figure)
    return tuple(files)


def _candidate_feature_names(loader: DatasetLoader) -> list[str]:
    channels = loader.registry.get("channels")
    if not isinstance(channels, Mapping):
        return []
    return [
        str(name)
        for name, value in channels.items()
        if isinstance(value, Mapping) and bool(value.get("analysis_candidate", False))
    ]


__all__ = [
    "FIGURE_NAMES",
    "plot_availability_audit",
    "plot_cycle_progress",
    "plot_feature_profiles",
    "plot_future_horizon_summary",
    "write_figures",
]
