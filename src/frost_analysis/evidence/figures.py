"""Evidence figures built from the Dataset Loader and Evidence tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..dataset_loader import DatasetLoader
from .contracts import EvidenceBundle
from .metrics import observed_mask
from .settings import EvidenceSettings

FIGURE_NAMES = (
    "figure_1_cycle_progress",
    "figure_2_feature_profiles",
    "figure_3_future_horizon",
    "figure_s2_availability",
)
_PROGRESS_BIN_COUNT = 100


def plot_cycle_progress(loader: DatasetLoader, settings: EvidenceSettings) -> Figure:
    """Plot target baseline residuals on fixed 100-bin cycle progress axes."""
    targets = settings.targets or ("Unavailable",)
    figure, axes = plt.subplots(
        len(targets),
        1,
        figsize=(8, max(2.8, 2.8 * len(targets))),
        squeeze=False,
        sharex=True,
    )
    flat_axes = list(axes[:, 0])
    centers = (np.arange(_PROGRESS_BIN_COUNT, dtype=float) + 0.5) / _PROGRESS_BIN_COUNT
    for axis, target in zip(flat_axes, targets, strict=True):
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
                np.isfinite(progress)
                & (progress >= 0.0)
                & (progress <= 1.0)
                & np.isfinite(values)
                & observed
            )
            if not valid.any():
                continue
            bins = np.minimum(
                (progress[valid] * _PROGRESS_BIN_COUNT).astype(int),
                _PROGRESS_BIN_COUNT - 1,
            )
            binned = pd.DataFrame({"bin": bins, "value": values[valid]}).groupby(
                "bin", sort=True
            )["value"].median()
            values_by_bin = np.full(_PROGRESS_BIN_COUNT, np.nan)
            values_by_bin[binned.index.to_numpy(dtype=int)] = binned.to_numpy(dtype=float)
            cycle_rows.append(
                pd.DataFrame(
                    {
                        "experiment_date": str(record.get("experiment_date", ""))[:10],
                        "cycle_name": str(record.get("cycle_name", "")),
                        "bin": np.arange(_PROGRESS_BIN_COUNT),
                        "value": values_by_bin,
                    }
                )
            )
        if not cycle_rows:
            _mark_unavailable(axis, target, "residual")
            continue

        cycles = pd.concat(cycle_rows, ignore_index=True)
        date_cycle = (
            cycles.groupby(["experiment_date", "cycle_name", "bin"], sort=False)["value"]
            .median()
            .reset_index()
        )
        date_medians: pd.DataFrame = date_cycle.groupby(
            ["experiment_date", "bin"], sort=False, as_index=False
        ).agg(value=("value", "median"))
        for experiment_date, date_values in date_medians.groupby(
            "experiment_date", sort=False
        ):
            y_values = np.full(_PROGRESS_BIN_COUNT, np.nan)
            bins = pd.to_numeric(date_values["bin"], errors="coerce").to_numpy(dtype=int)
            y_values[bins] = pd.to_numeric(
                date_values["value"], errors="coerce"
            ).to_numpy(dtype=float)
            axis.plot(
                centers,
                y_values,
                color="0.70",
                linewidth=0.8,
                alpha=0.85,
                label=str(experiment_date),
            )
        across_dates = date_medians.groupby("bin", sort=True)["value"].median()
        y_values = np.full(_PROGRESS_BIN_COUNT, np.nan)
        bins = across_dates.index.to_numpy(dtype=int)
        y_values[bins] = across_dates.to_numpy(dtype=float)
        axis.plot(centers, y_values, color="black", linewidth=2.0, label="date median")
        axis.set_title(target)
        axis.set_ylabel("baseline residual")
        axis.set_xlim(0.0, 1.0)
        axis.grid(alpha=0.2)
    flat_axes[-1].set_xlabel("cycle_progress")
    figure.tight_layout()
    return figure


def plot_feature_profiles(
    bundle: EvidenceBundle,
    settings: EvidenceSettings,
) -> Figure:
    """Plot date-level cycle medians and cross-date medians for three metrics."""
    features = bundle.feature_profile["feature"].astype(str).tolist()
    rows = max(1, len(features))
    figure, axes = plt.subplots(
        rows,
        3,
        figsize=(12, max(4.0, rows * 2.4)),
        squeeze=False,
    )
    metric_columns = (
        "signed_effect",
        "trend_slope_per_min",
        "primary_future_degradation_support",
    )
    metrics = bundle.feature_cycle_metrics
    future = bundle.future_association
    for row_index, feature in enumerate(features):
        metric_rows = metrics.loc[metrics["feature"].eq(feature)]
        future_rows = future.loc[
            future["feature"].eq(feature)
            & future["target"].eq(settings.primary_target)
            & future["horizon_minutes"].eq(settings.primary_horizon_minutes)
        ]
        for column_index, column in enumerate(metric_columns):
            axis = axes[row_index, column_index]
            source = (
                future_rows
                if column == "primary_future_degradation_support"
                else metric_rows
            )
            value_column = (
                "degradation_support"
                if column == "primary_future_degradation_support"
                else column
            )
            values = _date_level_values(source, value_column)
            if values.empty:
                axis.text(0.5, 0.5, "Unavailable", ha="center", va="center", fontsize=8)
            else:
                x_values = np.arange(len(values), dtype=float)
                y_values = values.to_numpy(dtype=float)
                median = float(np.median(y_values))
                axis.scatter(
                    x_values,
                    y_values,
                    color="0.70",
                    marker="o",
                    s=20,
                )
                axis.scatter(
                    [float(np.median(x_values))],
                    [median],
                    color="black",
                    marker="D",
                )
            if column in {"signed_effect", "primary_future_degradation_support"}:
                axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
                axis.set_ylim(-1.0, 1.0)
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
    """Plot only degradation support already aggregated in the horizon summary."""
    features = list(dict.fromkeys(summary.get("feature", pd.Series(dtype=str)).astype(str)))
    targets = settings.targets or ("Unavailable",)
    if not features:
        features = ["Unavailable"]
    figure, axes = plt.subplots(
        len(targets),
        1,
        figsize=(10, max(2.8, len(targets) * 2.8)),
        squeeze=False,
    )
    image = None
    for axis, target in zip(list(axes[:, 0]), targets, strict=True):
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
                if selected.empty:
                    continue
                selected_row = selected.iloc[0]
                value_raw: object = selected_row.get("degradation_support")
                value = pd.to_numeric(
                    pd.Series([value_raw]), errors="coerce"
                ).iloc[0]
                cycle_count = int(selected_row.get("valid_cycle_count", 0))
                date_count = int(selected_row.get("valid_date_count", 0))
                label = f"cycle={cycle_count}\ndate={date_count}"
                if pd.notna(value):
                    matrix[row_index, col_index] = float(value)
                    label = f"{float(value):.2g}\n{label}"
                axis.text(
                    col_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        axis.set_yticks(np.arange(len(features)), labels=features)
        axis.set_xticks(
            np.arange(len(settings.horizons_minutes)),
            labels=[f"{value} min" for value in settings.horizons_minutes],
        )
        axis.set_title(target)
        axis.set_ylabel("feature")
    if image is not None:
        figure.colorbar(image, ax=list(axes[:, 0]), fraction=0.03, pad=0.02)
    figure.subplots_adjust(left=0.30, right=0.86, bottom=0.12, top=0.92, hspace=0.3)
    return figure


def plot_availability_audit(
    bundle: EvidenceBundle, settings: EvidenceSettings
) -> Figure:
    """Plot valid-cycle feature availability and primary pair coverage."""
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), squeeze=False)
    feature_axis, future_axis = axes[:, 0]
    eligibility = bundle.cycle_eligibility
    valid_cycles = eligibility.loc[
        eligibility["status"].eq("valid"), "cycle_name"
    ].astype(str)
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
        feature_axis.imshow(
            feature_matrix,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        feature_axis.set_xticks(
            np.arange(len(features)), labels=features, rotation=45, ha="right"
        )
        feature_axis.set_yticks(np.arange(len(valid_cycles)), labels=valid_cycles)
    else:
        feature_axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
    feature_axis.set_title("S2 valid cycle × feature — availability")
    feature_axis.set_ylabel("cycle")

    future = bundle.future_association
    target = settings.primary_target
    horizon = settings.primary_horizon_minutes
    future_matrix = np.full((len(valid_cycles), len(features)), np.nan)
    selected_future = future.loc[
        future["target"].eq(target) & future["horizon_minutes"].eq(horizon)
    ]
    for row_index, cycle_name in enumerate(valid_cycles):
        for col_index, feature in enumerate(features):
            selected = selected_future.loc[
                selected_future["cycle_name"].eq(cycle_name)
                & selected_future["feature"].eq(feature)
            ]
            if not selected.empty:
                value = pd.to_numeric(selected.iloc[0]["pair_coverage"], errors="coerce")
                if pd.notna(value):
                    future_matrix[row_index, col_index] = float(value)
    if future_matrix.size:
        future_axis.imshow(
            future_matrix,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        future_axis.set_xticks(
            np.arange(len(features)), labels=features, rotation=45, ha="right"
        )
        future_axis.set_yticks(np.arange(len(valid_cycles)), labels=valid_cycles)
    else:
        future_axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
    future_axis.set_title(
        f"S2 valid cycle × feature — pair coverage ({target}, {horizon} min)"
    )
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
        plot_feature_profiles(bundle, settings),
        plot_future_horizon_summary(bundle.future_horizon_summary, settings),
        plot_availability_audit(bundle, settings),
    )
    files: list[str] = []
    for name, figure in zip(FIGURE_NAMES, figures, strict=True):
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            figure.savefig(output_dir / filename)
            files.append(filename)
        plt.close(figure)
    return tuple(files)


def _date_level_values(frame: pd.DataFrame, value_column: str) -> pd.Series[Any]:
    if frame.empty or value_column not in frame or "experiment_date" not in frame:
        return pd.Series(dtype=float)
    selected = frame.loc[:, ["experiment_date", value_column]].copy()
    if "metric_status" in frame:
        selected = selected.loc[frame["metric_status"].eq("available")]
    values = pd.to_numeric(selected[value_column], errors="coerce")
    selected = selected.loc[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]
    if selected.empty:
        return pd.Series(dtype=float)
    selected[value_column] = pd.to_numeric(selected[value_column], errors="coerce")
    return selected.groupby("experiment_date", sort=True)[value_column].median()


def _mark_unavailable(axis: Axes, title: str, ylabel: str) -> None:
    axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
