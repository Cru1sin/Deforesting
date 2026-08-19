"""Evidence figures built from the Dataset Loader and Evidence tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
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
    "figure_4_readiness_decision",
)
_PROGRESS_BIN_COUNT = 100
_NAVY = "#0F4D92"
_TEAL = "#42949E"
_RED = "#B64342"
_GREY = "#767676"
_LIGHT_GREY = "#D8D8D8"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False


def plot_cycle_progress(
    loader: DatasetLoader,
    settings: EvidenceSettings,
    eligible_cycle_names: set[str] | None = None,
) -> Figure:
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
        for record, frame in loader.iter_cycle_frames(
            statuses=set(settings.eligible_statuses)
        ):
            if (
                eligible_cycle_names is not None
                and str(record.get("cycle_name", "")) not in eligible_cycle_names
            ):
                continue
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
    eligible_cycles = eligibility.loc[
        eligibility["eligible"], "cycle_name"
    ].astype(str)
    metrics = bundle.feature_cycle_metrics
    features = list(dict.fromkeys(metrics.get("feature", pd.Series(dtype=str)).astype(str)))
    feature_matrix = np.zeros((len(eligible_cycles), len(features)), dtype=float)
    for row_index, cycle_name in enumerate(eligible_cycles):
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
            cmap=ListedColormap(["#E5E5E5", _TEAL]),
        )
        feature_axis.set_xticks(
            np.arange(len(features)), labels=features, rotation=45, ha="right"
        )
        feature_axis.set_yticks(np.arange(len(eligible_cycles)), labels=eligible_cycles)
    else:
        feature_axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
    feature_axis.set_title("S2 eligible cycle × feature — availability")
    feature_axis.set_ylabel("cycle")

    future = bundle.future_association
    target = settings.primary_target
    horizon = settings.primary_horizon_minutes
    future_matrix = np.full((len(eligible_cycles), len(features)), np.nan)
    selected_future = future.loc[
        future["target"].eq(target) & future["horizon_minutes"].eq(horizon)
    ]
    for row_index, cycle_name in enumerate(eligible_cycles):
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
            cmap="Blues",
        )
        future_axis.set_xticks(
            np.arange(len(features)), labels=features, rotation=45, ha="right"
        )
        future_axis.set_yticks(np.arange(len(eligible_cycles)), labels=eligible_cycles)
    else:
        future_axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
    future_axis.set_title(
        f"S2 eligible cycle × feature — pair coverage ({target}, {horizon} min)"
    )
    future_axis.set_ylabel("cycle")
    figure.tight_layout()
    return figure


def plot_readiness_decision(  # noqa: C901
    bundle: EvidenceBundle, settings: EvidenceSettings
) -> Figure:
    """Plot the target, lead, horizon, and incremental-skill decision chain."""
    figure = plt.figure(figsize=(7.2, 5.1))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.35, 1.0),
        height_ratios=(1.0, 0.9),
        wspace=0.48,
        hspace=0.55,
    )
    lead_axis = figure.add_subplot(grid[0, 0])
    skill_axis = figure.add_subplot(grid[0, 1])
    status_axis = figure.add_subplot(grid[1, 0])
    event_axis = figure.add_subplot(grid[1, 1])

    summary = bundle.readiness_summary
    primary = summary.loc[
        summary["target"].eq(settings.primary_target)
        & summary["horizon_minutes"].eq(settings.primary_horizon_minutes)
    ].copy()
    features = primary["feature"].astype(str).tolist()
    positions = np.arange(len(features), dtype=float)[::-1]
    if features:
        q25 = pd.to_numeric(primary["lead_q25_minutes"], errors="coerce").to_numpy(
            dtype=float
        )
        medians = pd.to_numeric(primary["lead_median_minutes"], errors="coerce").to_numpy(
            dtype=float
        )
        counts = pd.to_numeric(primary["lead_valid_cycle_count"], errors="coerce").fillna(0)
        for y_value, lower, median, count in zip(
            positions, q25, medians, counts, strict=True
        ):
            if np.isfinite(lower) and np.isfinite(median):
                lead_axis.plot([lower, median], [y_value, y_value], color=_NAVY, lw=2.0)
                lead_axis.scatter(lower, y_value, color=_TEAL, marker="s", s=20, zorder=3)
                lead_axis.scatter(median, y_value, color=_NAVY, s=24, zorder=3)
                lead_axis.annotate(
                    f"n={int(count)}",
                    (median, y_value),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    color=_GREY,
                    fontsize=6,
                )
        lead_axis.set_yticks(positions, labels=features)
    else:
        _center_unavailable(lead_axis)
    lead_axis.axvline(0.0, color=_GREY, lw=0.8, linestyle="--")
    lead_axis.set_xlabel("Lead before $T_{perf}$ (min)")
    lead_axis.set_title("Robust lead: Q25 to median", loc="left", fontweight="bold")

    if features:
        level = pd.to_numeric(primary["level_skill_median"], errors="coerce").to_numpy(
            dtype=float
        )
        dynamic = pd.to_numeric(
            primary["dynamic_skill_median"], errors="coerce"
        ).to_numpy(dtype=float)
        skill_axis.scatter(level, positions + 0.10, color=_TEAL, s=24, label="Level")
        skill_axis.scatter(
            dynamic, positions - 0.10, color=_RED, marker="D", s=20, label="History"
        )
        skill_axis.set_yticks(positions, labels=features)
        skill_axis.legend(loc="best", fontsize=6)
    else:
        _center_unavailable(skill_axis)
    skill_axis.axvline(0.0, color=_GREY, lw=0.8, linestyle="--")
    skill_axis.set_xlabel("Held-out skill increment")
    skill_axis.set_title("Information beyond baseline", loc="left", fontweight="bold")

    horizons = list(settings.horizons_minutes)
    status_rows = summary.loc[summary["target"].eq(settings.primary_target)]
    status_order = {
        "target_not_evaluable": 0,
        "insufficient_validation_data": 0,
        "state_marker_candidate": 1,
        "no_stable_linear_increment": 2,
        "level_increment_supported": 3,
        "dynamic_increment_supported": 4,
    }
    short_status = {
        "target_not_evaluable": "target?",
        "insufficient_validation_data": "n?",
        "state_marker_candidate": "state",
        "no_stable_linear_increment": "lead only",
        "level_increment_supported": "level+",
        "dynamic_increment_supported": "history+",
    }
    matrix = np.full((len(features), len(horizons)), np.nan)
    for row_index, feature in enumerate(features):
        for column_index, horizon in enumerate(horizons):
            selected = status_rows.loc[
                status_rows["feature"].eq(feature)
                & status_rows["horizon_minutes"].eq(horizon)
            ]
            if selected.empty:
                continue
            status = str(selected.iloc[0]["readiness_status"])
            matrix[row_index, column_index] = status_order.get(status, 0)
            status_axis.text(
                column_index,
                row_index,
                short_status.get(status, "?"),
                ha="center",
                va="center",
                fontsize=6,
            )
    if matrix.size:
        status_axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=4,
            cmap=ListedColormap(["#E5E5E5", "#F6CFCB", "#CFCECE", "#AADCA9", "#42949E"]),
        )
        status_axis.set_yticks(np.arange(len(features)), labels=features)
        status_axis.set_xticks(np.arange(len(horizons)), labels=[f"{h} min" for h in horizons])
    else:
        _center_unavailable(status_axis)
    status_axis.set_title("Decision boundary by horizon", loc="left", fontweight="bold")

    audits = bundle.target_audit.loc[
        bundle.target_audit["target"].eq(settings.primary_target)
    ]
    observed = int(audits["primary_event_status"].eq("event_observed").sum())
    censored = int(
        audits["primary_event_status"].astype(str).str.startswith("right_censored").sum()
    )
    unavailable = int(len(audits) - observed - censored)
    event_counts = [observed, censored, unavailable]
    event_labels = ["Observed", "Right-censored", "Unavailable"]
    event_axis.barh(
        np.arange(3)[::-1],
        event_counts,
        color=[_NAVY, _TEAL, _LIGHT_GREY],
        edgecolor="white",
    )
    event_axis.set_yticks(np.arange(3)[::-1], labels=event_labels)
    event_axis.set_xlabel("Cycles")
    event_axis.set_title("$T_{perf}$ observability", loc="left", fontweight="bold")
    for y_value, value in zip(np.arange(3)[::-1], event_counts, strict=True):
        event_axis.text(value, y_value, f" {value}", va="center", fontsize=6)

    for label, axis in zip("abcd", (lead_axis, skill_axis, status_axis, event_axis), strict=True):
        _add_panel_label(axis, label)
    return figure


def write_figures(
    output_dir: Path,
    bundle: EvidenceBundle,
    loader: DatasetLoader,
    settings: EvidenceSettings,
) -> tuple[str, ...]:
    """Write the four required figures as PNG and PDF files."""
    eligible_cycle_names = set(
        bundle.cycle_eligibility.loc[
            bundle.cycle_eligibility["eligible"], "cycle_name"
        ].astype(str)
    )
    figures = (
        plot_cycle_progress(loader, settings, eligible_cycle_names),
        plot_feature_profiles(bundle, settings),
        plot_future_horizon_summary(bundle.future_horizon_summary, settings),
        plot_availability_audit(bundle, settings),
        plot_readiness_decision(bundle, settings),
    )
    files: list[str] = []
    for name, figure in zip(FIGURE_NAMES, figures, strict=True):
        for suffix in ("svg", "pdf", "png", "tiff"):
            filename = f"{name}.{suffix}"
            figure.savefig(
                output_dir / filename,
                dpi=600 if suffix == "tiff" else 300,
                bbox_inches="tight",
            )
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
    return cast(
        "pd.Series[Any]",
        selected.groupby("experiment_date", sort=True)[value_column].median(),
    )


def _mark_unavailable(axis: Axes, title: str, ylabel: str) -> None:
    axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
    axis.set_title(title)
    axis.set_ylabel(ylabel)


def _center_unavailable(axis: Axes) -> None:
    axis.text(0.5, 0.5, "Unavailable", transform=axis.transAxes, ha="center", va="center")
    axis.set_xticks([])
    axis.set_yticks([])


def _add_panel_label(axis: Axes, label: str) -> None:
    axis.text(
        -0.14,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
