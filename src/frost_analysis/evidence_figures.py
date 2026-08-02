"""Exploratory, read-only figures for frost-cycle evidence outputs.

The public functions return Matplotlib Figure objects and do not write files or
mutate their inputs. The figures are descriptive views of the existing
Evidence contract; they do not add metrics, states, or model rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from textwrap import fill
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from .config import EvidencePolicy
from .evidence_cycle import build_channel_evidence, build_cycle_slices

_CYCLE_KEYS = ("experiment_id", "experiment_date", "cycle_id")
_DISPLAY_MODES = frozenset({"raw", "reference_normalized"})
_VARIANTS = ("residual_level", "past_slope_5min")
_PROGRESS_EDGES = np.linspace(0.0, 1.0, 101)
_PROGRESS_CENTERS = (_PROGRESS_EDGES[:-1] + _PROGRESS_EDGES[1:]) / 2
_PALETTE = ("#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2")


def _validate_non_empty_settings(
    groups: Sequence[tuple[str, tuple[str, ...]]], channels: Sequence[str]
) -> None:
    if not groups:
        raise ValueError("feature_groups must not be empty")
    if not channels:
        raise ValueError("cycle_channels must not be empty")


@dataclass(frozen=True)
class EvidenceFigureSettings:
    """Display-only settings for the exploratory Evidence figures."""

    feature_groups: tuple[tuple[str, tuple[str, ...]], ...]
    cycle_channels: tuple[str, ...]
    display_modes: Mapping[str, str]
    date_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        groups = tuple(
            (str(name), tuple(str(feature) for feature in features))
            for name, features in self.feature_groups
        )
        channels = tuple(str(channel) for channel in self.cycle_channels)
        dates = tuple(str(date) for date in self.date_order)
        modes = {str(channel): str(mode) for channel, mode in self.display_modes.items()}
        object.__setattr__(self, "feature_groups", groups)
        object.__setattr__(self, "cycle_channels", channels)
        object.__setattr__(self, "date_order", dates)
        object.__setattr__(self, "display_modes", modes)

        _validate_non_empty_settings(groups, channels)
        group_names = [name for name, _ in groups]
        if len(set(group_names)) != len(group_names):
            raise ValueError("feature group names must be unique")
        if any(not name.strip() for name in group_names):
            raise ValueError("feature group names must not be empty")
        if any(not features for _, features in groups):
            raise ValueError("feature groups must not be empty")
        features = [feature for _, group in groups for feature in group]
        if len(set(features)) != len(features):
            raise ValueError("duplicate feature in feature_groups")
        if len(set(channels)) != len(channels):
            raise ValueError("cycle_channels must be unique")
        missing_modes = [channel for channel in channels if channel not in modes]
        if missing_modes:
            raise ValueError(f"cycle channel display modes are missing: {missing_modes}")
        invalid_modes = sorted(set(modes.values()) - _DISPLAY_MODES)
        if invalid_modes:
            raise ValueError(
                f"display modes must be raw or reference_normalized: {invalid_modes}"
            )
        if len(set(dates)) != len(dates):
            raise ValueError("date_order must be unique")

    @property
    def feature_order(self) -> tuple[str, ...]:
        """Flattened feature order; the only feature sorting source."""
        return tuple(feature for _, group in self.feature_groups for feature in group)


def plot_cycle_evolution(
    processed: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    cycle_eligibility: pd.DataFrame,
    channels: Mapping[str, Mapping[str, Any]],
    policy: EvidencePolicy,
    settings: EvidenceFigureSettings,
    grid_interval_seconds: int,
) -> Figure:
    """Plot eligible cycle trajectories on formal normalized-progress bins."""
    _require_columns(
        cycle_eligibility,
        (*_CYCLE_KEYS, "eligibility_status"),
        "cycle_eligibility",
    )
    missing = [channel for channel in settings.cycle_channels if channel not in processed]
    if missing:
        raise ValueError(f"requested figure channels are missing: {missing}")
    eligible_keys = _eligible_keys(cycle_eligibility)
    cycles = [
        cycle
        for cycle in build_cycle_slices(processed, cycle_summary, grid_interval_seconds)
        if cycle.eligible and cycle.key in eligible_keys
    ]
    dates = _ordered_dates([cycle.key[1] for cycle in cycles], settings)
    date_colors = _date_colors(dates)
    figure, axes = plt.subplots(
        len(settings.cycle_channels),
        1,
        figsize=(7.2, max(2.4, 2.35 * len(settings.cycle_channels))),
        sharex=True,
        squeeze=False,
    )
    for index, channel in enumerate(settings.cycle_channels):
        axis = axes[index, 0]
        mode = settings.display_modes[channel]
        entries = []
        for cycle in cycles:
            evidence = build_channel_evidence(
                cycle,
                channel,
                policy,
                target=channel in policy.targets,
                interval_seconds=grid_interval_seconds,
            )
            entries.append(
                {
                    "date": cycle.key[1],
                    "values": _progress_bins(
                        _formal_progress(cycle.grid, cycle.start, cycle.end),
                        _display_values(evidence, mode),
                    ),
                    "imputed_fraction": float(evidence.imputed.mean()),
                    "reference_progress": _reference_progress(cycle, evidence),
                }
            )
        _draw_trajectory_panel(
            axis, entries, dates, date_colors, channel, channels, mode
        )
    axes[-1, 0].set_xlabel("Formal frost progress")
    axes[-1, 0].set_xlim(0.0, 1.0)
    axes[-1, 0].set_xticks(np.linspace(0.0, 1.0, 5))
    figure.text(
        0.5,
        0.002,
        "Vertical ticks indicate date-level median reference-valid progress.",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    return figure


def plot_evidence_profiles(
    feature_profile: pd.DataFrame,
    feature_cycle_metrics: pd.DataFrame,
    future_association: pd.DataFrame,
    policy: EvidencePolicy,
    settings: EvidenceFigureSettings,
) -> Figure:
    """Plot existing date-balanced profile values and date-level points."""
    _require_columns(
        feature_profile,
        (
            "feature",
            "global_spearman_median",
            "global_spearman_iqr",
            "signed_sensitivity_median",
            "signed_sensitivity_iqr",
            "primary_future_effect_median",
            "primary_future_effect_iqr",
            "primary_future_valid_cycle_count",
            "primary_future_valid_date_count",
            "primary_future_evidence_status",
            "trend_valid_cycle_count",
            "trend_valid_date_count",
            "trend_evidence_status",
            "reference_scope",
        ),
        "feature_profile",
    )
    _require_columns(
        feature_cycle_metrics,
        (*_CYCLE_KEYS, "feature", "global_spearman", "signed_sensitivity"),
        "feature_cycle_metrics",
    )
    _require_columns(
        future_association,
        (
            *_CYCLE_KEYS,
            "feature",
            "feature_variant",
            "target",
            "target_type",
            "horizon_minutes",
            "effect",
        ),
        "future_association",
    )
    order = _validate_requested_features(
        settings.feature_order, feature_profile["feature"], "feature_profile"
    )
    profile = feature_profile.set_index("feature", drop=False).loc[list(order)]
    primary = _select_primary_future(future_association, policy)
    dates = _ordered_dates(
        [
            *feature_cycle_metrics["experiment_date"].dropna().astype(str),
            *primary["experiment_date"].dropna().astype(str),
        ],
        settings,
    )
    date_colors = _date_colors(dates)
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(14.5, max(3.0, 0.42 * len(order) + 1.4)),
        sharey=True,
    )
    y = np.arange(len(order))[::-1]
    for axis, column, iqr_column, title in (
        (axes[0], "global_spearman_median", "global_spearman_iqr", "Global Spearman"),
        (axes[1], "signed_sensitivity_median", "signed_sensitivity_iqr", "Signed sensitivity"),
        (
            axes[2],
            "primary_future_effect_median",
            "primary_future_effect_iqr",
            "Primary future association",
        ),
    ):
        values = pd.to_numeric(profile[column], errors="coerce").to_numpy(dtype=float)
        iqr = pd.to_numeric(profile[iqr_column], errors="coerce").to_numpy(dtype=float)
        axis.errorbar(values, y, xerr=iqr / 2, fmt="o", color="#222222", ms=5, capsize=2, zorder=3)
        axis.axvline(0, color="#999999", lw=0.6, zorder=0)
        _draw_date_points(
            axis,
            feature_cycle_metrics,
            primary,
            order,
            column,
            dates,
            date_colors,
            y,
        )
        axis.set_title(title)
        axis.grid(axis="x", color="#dddddd", lw=0.4)
    for position, feature in zip(y, order, strict=True):
        row = profile.loc[feature]
        separator = chr(10)
        text = (
            f"trend n={_text_int(row['trend_valid_cycle_count'])}/"
            f"d={_text_int(row['trend_valid_date_count'])} "
            f"{row['trend_evidence_status']}{separator}"
            f"future n={_text_int(row['primary_future_valid_cycle_count'])}/"
            f"d={_text_int(row['primary_future_valid_date_count'])} "
            f"{row['primary_future_evidence_status']}{separator}"
            f"ref={row['reference_scope']}"
        )
        axes[3].text(
            0.02,
            position,
            text,
            transform=axes[3].get_yaxis_transform(),
            va="center",
            fontsize=6.5,
        )
    axes[3].set_title("Reliability / reference")
    axes[3].set_xlim(0, 1)
    axes[3].set_xticks([])
    axes[0].set_yticks(y, order)
    axes[0].set_xlabel("rho")
    axes[1].set_xlabel("scaled level")
    axes[2].set_xlabel("rho")
    for axis in axes:
        axis.tick_params(labelsize=7)
    figure.tight_layout()
    return figure


def plot_future_horizon_map(
    future_association: pd.DataFrame,
    policy: EvidencePolicy,
    settings: EvidenceFigureSettings,
) -> Figure:
    """Plot future-change effects by target, variant, and configured horizon."""
    _require_columns(
        future_association,
        (
            *_CYCLE_KEYS,
            "feature",
            "feature_variant",
            "target",
            "target_type",
            "horizon_minutes",
            "effect",
            "exclusion_reason",
        ),
        "future_association",
    )
    order = _validate_requested_features(
        settings.feature_order, future_association["feature"], "future_association"
    )
    selected = future_association.loc[
        future_association["feature"].isin(order)
        & future_association["target"].isin(policy.targets)
    ].copy()
    selected = selected.loc[selected["target_type"].eq("future_change")]
    figure, axes = plt.subplots(
        len(policy.targets),
        len(_VARIANTS),
        figsize=(8.4, max(3.4, 0.5 * len(order) + 1.8) * len(policy.targets)),
        squeeze=False,
        constrained_layout=True,
    )
    images = []
    primary_index = list(policy.horizons_minutes).index(policy.primary_horizon_minutes)
    for row_index, target in enumerate(policy.targets):
        for column_index, variant in enumerate(_VARIANTS):
            axis = axes[row_index, column_index]
            matrix = np.full((len(order), len(policy.horizons_minutes)), np.nan)
            annotations: dict[tuple[int, int], str] = {}
            reasons: dict[tuple[int, int], str] = {}
            for feature_index, feature in enumerate(order):
                for horizon_index, horizon in enumerate(policy.horizons_minutes):
                    frame = selected.loc[
                        selected["feature"].eq(feature)
                        & selected["target"].eq(target)
                        & selected["feature_variant"].eq(variant)
                        & selected["horizon_minutes"].eq(horizon)
                    ]
                    value, n_cycle, n_date, reason = _balanced_effect(frame)
                    if np.isfinite(value):
                        matrix[feature_index, horizon_index] = value
                        annotations[(feature_index, horizon_index)] = (
                            f"n={n_cycle}/{n_date}"
                        )
                    else:
                        reasons[(feature_index, horizon_index)] = reason
            image = axis.imshow(
                np.ma.masked_invalid(matrix),
                vmin=-1,
                vmax=1,
                cmap="coolwarm",
                aspect="auto",
            )
            images.append(image)
            axis.set_title(f"{target} · {variant}", fontsize=9)
            axis.set_xticks(
                np.arange(len(policy.horizons_minutes)),
                [f"{h} min" for h in policy.horizons_minutes],
            )
            axis.set_yticks(np.arange(len(order)), order)
            axis.tick_params(labelsize=7)
            for feature_index in range(len(order)):
                for horizon_index in range(len(policy.horizons_minutes)):
                    text = annotations.get(
                        (feature_index, horizon_index),
                        reasons.get((feature_index, horizon_index), "X"),
                    )
                    axis.text(
                        horizon_index,
                        feature_index,
                        text,
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
            axis.add_patch(
                Rectangle(
                    (primary_index - 0.5, -0.5),
                    1,
                    len(order),
                    fill=False,
                    ec="black",
                    lw=1.5,
                )
            )
            if column_index == 0:
                axis.set_ylabel("feature")
    figure.colorbar(
        images[0],
        ax=axes.ravel().tolist(),
        label="within-cycle Spearman",
        pad=0.06,
        shrink=0.85,
    )
    figure.suptitle("Future-change temporal association")
    figure.text(
        0.01,
        0.002,
        "Unavailable: R=reference  C=coverage  P=pairs  V=zero variability "
        "N=no structural anchors  X=other",
        fontsize=7,
    )
    return figure


def plot_pair_similarity(
    feature_pair_similarity: pd.DataFrame,
    settings: EvidenceFigureSettings,
) -> Figure:
    """Plot existing dynamic-similarity values in a lower-triangle matrix."""
    _require_columns(
        feature_pair_similarity,
        (
            "feature_a",
            "feature_b",
            "dynamic_spearman_median",
            "valid_date_count",
            "evaluated_cycle_count",
            "valid_cycle_count",
            "pair_coverage_median",
            "definition_dependency",
            "similarity_status",
            "similarity_reason",
        ),
        "feature_pair_similarity",
    )
    pair_features = pd.concat(
        [feature_pair_similarity["feature_a"], feature_pair_similarity["feature_b"]]
    )
    order = _validate_requested_features(
        settings.feature_order, pair_features, "feature_pair_similarity"
    )
    positions = {feature: index for index, feature in enumerate(order)}
    matrix = np.full((len(order), len(order)), np.nan)
    weak_cycles: list[tuple[int, int]] = []
    annotations: dict[tuple[int, int], str] = {}
    for _, row in feature_pair_similarity.iterrows():
        left = str(row["feature_a"])
        right = str(row["feature_b"])
        if left not in positions or right not in positions or left == right:
            continue
        row_index = max(positions[left], positions[right])
        column_index = min(positions[left], positions[right])
        value = _finite_value(row["dynamic_spearman_median"])
        status = str(row["similarity_status"])
        if value is not None and status != "no_valid_evidence":
            matrix[row_index, column_index] = value
            coverage = _finite_value(row["pair_coverage_median"])
            coverage_text = f"cov={coverage:.2f}" if coverage is not None else "cov=NA"
            dependency_text = " D" if bool(row["definition_dependency"]) else ""
            reason = str(row["similarity_reason"]).strip()
            reason_text = (
                fill(
                    f"reason: {reason.replace('_', ' ')}",
                    width=18,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                if reason and reason != "nan"
                else ""
            )
            annotations[(row_index, column_index)] = (
                f"e={_text_int(row['evaluated_cycle_count'])}/"
                f"v={_text_int(row['valid_cycle_count'])}/"
                f"d={_text_int(row['valid_date_count'])}\n"
                f"{coverage_text}{dependency_text}\n{reason_text}"
            )
            if status == "insufficient_cycles":
                weak_cycles.append((row_index, column_index))
        elif status == "no_valid_evidence":
            reason = str(row["similarity_reason"]).strip()
            annotations[(row_index, column_index)] = (
                fill(
                    f"reason: {reason.replace('_', ' ')}",
                    width=18,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                if reason and reason != "nan"
                else "—"
            )
    figure, axis = plt.subplots(
        figsize=(7.5, max(5.5, 0.55 * len(order) + 1.3))
    )
    image = axis.imshow(np.ma.masked_invalid(matrix), vmin=-1, vmax=1, cmap="coolwarm")
    labels = [fill(feature, width=17) for feature in order]
    axis.set_xticks(np.arange(len(order)), labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(order)), labels)
    axis.set_title(
        "Dynamic similarity (lower triangle; D=definition dependency)", fontsize=10
    )
    axis.tick_params(labelsize=7)
    for row_index, column_index in weak_cycles:
        axis.add_patch(
            Rectangle(
                (column_index - 0.5, row_index - 0.5),
                1,
                1,
                fill=False,
                ec="#666666",
                lw=2,
            )
        )
    for (row_index, column_index), text in annotations.items():
        axis.text(column_index, row_index, text, ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, label="dynamic Spearman", pad=0.04, shrink=0.85)
    figure.tight_layout()
    return figure


def plot_evidence_coverage(
    cycle_eligibility: pd.DataFrame,
    feature_cycle_metrics: pd.DataFrame,
    future_association: pd.DataFrame,
    policy: EvidencePolicy,
    settings: EvidenceFigureSettings,
) -> Figure:
    """Plot per-cycle eligibility and coverage audits using complete cycle keys."""
    _require_columns(
        cycle_eligibility,
        (
            *_CYCLE_KEYS,
            "eligibility_status",
            "frost_development_grid_coverage",
            "eligible_feature_count",
            "total_candidate_count",
            "exclusion_reason",
        ),
        "cycle_eligibility",
    )
    _require_columns(
        feature_cycle_metrics,
        (*_CYCLE_KEYS, "feature", "reference_source"),
        "feature_cycle_metrics",
    )
    _require_columns(
        future_association,
        (
            *_CYCLE_KEYS,
            "feature",
            "target",
            "target_type",
            "horizon_minutes",
            "feature_variant",
            "pair_coverage",
        ),
        "future_association",
    )
    _validate_requested_features(
        settings.feature_order, feature_cycle_metrics["feature"], "feature_cycle_metrics"
    )
    rows = _coverage_rows(
        cycle_eligibility, feature_cycle_metrics, future_association, policy
    )
    dates = _ordered_dates([row["experiment_date"] for row in rows], settings)
    date_rank = {date: index for index, date in enumerate(dates)}
    rows.sort(
        key=lambda row: (
            date_rank.get(row["experiment_date"], len(date_rank)),
            row["experiment_id"],
            row["cycle_id"],
        )
    )
    labels = [_short_cycle_label(row) for row in rows]
    y = np.arange(len(rows))
    figure, (coverage_axis, text_axis) = plt.subplots(
        1,
        2,
        figsize=(11.5, max(2.5, 0.38 * len(rows) + 1.5)),
        sharey=True,
        gridspec_kw={"width_ratios": (1.2, 2.0)},
    )
    grid = np.array([row["grid_coverage"] for row in rows], dtype=float)
    primary = np.array([row["primary_coverage"] for row in rows], dtype=float)
    coverage_axis.barh(y + 0.16, grid, height=0.28, label="frost grid")
    coverage_axis.barh(y - 0.16, primary, height=0.28, label="primary pair")
    coverage_axis.set_xlim(0, 1)
    coverage_axis.set_xlabel("coverage")
    coverage_axis.set_yticks(y, labels)
    coverage_axis.legend(frameon=False, fontsize=7)
    coverage_axis.set_title("Eligibility / coverage")
    for index, row in enumerate(rows):
        reference = (
            f"cfg={row['reference_configured']:.2f} "
            f"auto={row['reference_auto']:.2f} "
            f"unavailable={row['reference_unavailable']:.2f}"
        )
        text_axis.text(
            0.01,
            index,
            f"{row['eligibility_status']} | eligible "
            f"{row['eligible_features']}/{row['total_features']} | {reference} | "
            f"{row['exclusion_reason']}",
            transform=text_axis.get_yaxis_transform(),
            va="center",
            fontsize=7,
        )
    text_axis.set_xlim(0, 1)
    text_axis.set_xticks([])
    text_axis.set_title("Audit details")
    text_axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
    figure.tight_layout()
    return figure


def _draw_trajectory_panel(
    axis: Any,
    entries: list[dict[str, Any]],
    dates: list[str],
    date_colors: Mapping[str, str],
    channel: str,
    channels: Mapping[str, Mapping[str, Any]],
    mode: str,
) -> None:
    for entry in entries:
        axis.plot(
            _PROGRESS_CENTERS,
            entry["values"],
            color=date_colors.get(entry["date"], "#999999"),
            alpha=0.18,
            lw=0.7,
        )
    date_arrays: dict[str, NDArray[np.float64]] = {}
    for date in dates:
        values = [entry["values"] for entry in entries if entry["date"] == date]
        if values:
            date_array = _nanmedian(np.asarray(values, dtype=float), axis=0)
            date_arrays[date] = date_array
            axis.plot(
                _PROGRESS_CENTERS,
                date_array,
                color=date_colors[date],
                lw=1.4,
                label=date,
            )
    if date_arrays:
        axis.plot(
            _PROGRESS_CENTERS,
            _nanmedian(np.asarray(list(date_arrays.values())), axis=0),
            color="#222222",
            lw=2.0,
            label="cross-date median",
        )
    for date in dates:
        refs = [
            entry["reference_progress"]
            for entry in entries
            if entry["date"] == date and np.isfinite(entry["reference_progress"])
        ]
        if refs:
            reference_progress = float(np.median(refs))
            axis.plot(
                [reference_progress, reference_progress],
                [0.96, 1.0],
                transform=axis.get_xaxis_transform(),
                color=date_colors[date],
                lw=2,
            )
    mode_label = (
        "Reference-normalized residual"
        if mode == "reference_normalized"
        else _raw_label(channel, channels)
    )
    axis.set_ylabel(mode_label, fontsize=7, labelpad=9)
    axis.tick_params(labelsize=7)
    axis.axvspan(0, 0.25, color="#d9e8f5", alpha=0.35)
    axis.axvspan(0.25, 0.75, color="#eeeeee", alpha=0.35)
    axis.axvspan(0.75, 1.0, color="#f8dfd5", alpha=0.35)
    if entries:
        fractions = np.asarray(
            [entry["imputed_fraction"] for entry in entries], dtype=float
        )
        fraction = float(np.median(fractions[np.isfinite(fractions)]))
        axis.text(
            0.99,
            0.04,
            f"median imputed fraction={fraction:.2f}",
            transform=axis.transAxes,
            ha="right",
            fontsize=7,
        )
    axis.spines["top"].set_visible(False)


def _draw_date_points(
    axis: Any,
    metrics: pd.DataFrame,
    primary: pd.DataFrame,
    order: Sequence[str],
    column: str,
    dates: Sequence[str],
    colors: Mapping[str, str],
    y: NDArray[np.int64],
) -> None:
    source = primary if column == "primary_future_effect_median" else metrics
    source_column = (
        "effect" if column == "primary_future_effect_median" else column.removesuffix("_median")
    )
    for feature_index, feature in enumerate(order):
        for date in dates:
            frame = source.loc[
                source["feature"].eq(feature)
                & source["experiment_date"].astype(str).eq(date)
            ]
            values = pd.to_numeric(frame[source_column], errors="coerce").dropna()
            if not values.empty:
                axis.scatter(
                    float(values.median()),
                    y[feature_index],
                    color=colors[date],
                    s=12,
                    zorder=4,
                )


def _coverage_rows(
    eligibility: pd.DataFrame,
    metrics: pd.DataFrame,
    future: pd.DataFrame,
    policy: EvidencePolicy,
) -> list[dict[str, Any]]:
    primary = _select_primary_future(future, policy)
    rows: list[dict[str, Any]] = []
    for _, eligibility_row in eligibility.iterrows():
        key = (
            str(eligibility_row["experiment_id"]),
            str(eligibility_row["experiment_date"]),
            str(eligibility_row["cycle_id"]),
        )
        metric_rows = metrics.loc[_key_mask(metrics, key)]
        future_rows = primary.loc[_key_mask(primary, key)]
        total = _as_count(eligibility_row["total_candidate_count"])
        configured = int(metric_rows["reference_source"].eq("configured_baseline").sum())
        auto = int(metric_rows["reference_source"].eq("auto_cycle_initial_reference").sum())
        finite_coverage = pd.to_numeric(
            future_rows["pair_coverage"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "experiment_id": key[0],
                "experiment_date": key[1],
                "cycle_id": key[2],
                "eligibility_status": str(eligibility_row["eligibility_status"]),
                "grid_coverage": _number_or_nan(
                    eligibility_row["frost_development_grid_coverage"]
                ),
                "eligible_features": _as_count(eligibility_row["eligible_feature_count"]),
                "total_features": total,
                "reference_configured": configured / total if total else 0.0,
                "reference_auto": auto / total if total else 0.0,
                "reference_unavailable": max(total - configured - auto, 0) / total
                if total
                else 0.0,
                "primary_coverage": float(finite_coverage.median())
                if not finite_coverage.empty
                else np.nan,
                "exclusion_reason": str(eligibility_row["exclusion_reason"] or ""),
            }
        )
    return rows


def _select_primary_future(frame: pd.DataFrame, policy: EvidencePolicy) -> pd.DataFrame:
    return frame.loc[
        frame["target"].eq(policy.primary_target)
        & frame["target_type"].eq(policy.primary_target_type)
        & frame["horizon_minutes"].eq(policy.primary_horizon_minutes)
        & frame["feature_variant"].eq(policy.primary_feature_variant)
    ].copy()


def _balanced_effect(frame: pd.DataFrame) -> tuple[float, int, int, str]:
    if frame.empty:
        return np.nan, 0, 0, "N"
    values = pd.to_numeric(frame["effect"], errors="coerce")
    finite = frame.loc[values.notna()].copy()
    finite["_effect"] = values.loc[finite.index].to_numpy(dtype=float)
    if finite.empty:
        reasons = [str(value) for value in frame["exclusion_reason"].dropna()]
        return np.nan, 0, 0, _reason_code(reasons)
    cycle_values = (
        finite.groupby(list(_CYCLE_KEYS), sort=False)["_effect"].median().reset_index()
    )
    date_values = cycle_values.groupby("experiment_date", sort=True)["_effect"].median()
    return float(date_values.median()), len(cycle_values), len(date_values), ""


def _reason_code(reasons: Sequence[str]) -> str:
    text = " ".join(str(reason).lower() for reason in reasons)
    if "reference" in text:
        return "R"
    if "coverage" in text:
        return "C"
    if "pair" in text:
        return "P"
    if "zero_variability" in text or "zero variability" in text:
        return "V"
    if "structural" in text or "anchor" in text:
        return "N"
    return "X"


def _display_values(evidence: Any, mode: str) -> pd.Series[Any]:
    if mode == "raw":
        return cast("pd.Series[Any]", evidence.values)
    scale = evidence.reference.scale
    if (
        evidence.reference.source == "unavailable"
        or not np.isfinite(scale)
        or np.isclose(scale, 0.0)
    ):
        return pd.Series(np.nan, index=evidence.analysis_residual.index, dtype=float)
    return cast("pd.Series[Any]", evidence.analysis_residual / scale)


def _progress_bins(progress: pd.Series[Any], values: pd.Series[Any]) -> NDArray[np.float64]:
    result = np.full(len(_PROGRESS_CENTERS), np.nan)
    buckets: list[list[float]] = [[] for _ in _PROGRESS_CENTERS]
    numeric_progress = pd.to_numeric(progress, errors="coerce").to_numpy(dtype=float)
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = (
        np.isfinite(numeric_progress)
        & np.isfinite(numeric_values)
        & (numeric_progress >= 0)
        & (numeric_progress < 1)
    )
    for index in np.flatnonzero(valid):
        bin_index = int(
            np.searchsorted(_PROGRESS_EDGES, numeric_progress[index], side="right") - 1
        )
        if 0 <= bin_index < len(result):
            buckets[bin_index].append(float(numeric_values[index]))
    for bin_index, bucket in enumerate(buckets):
        if bucket:
            result[bin_index] = float(np.median(bucket))
    return result


def _formal_progress(
    grid: pd.DatetimeIndex,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.Series[float]:
    if start is None or end is None or end <= start:
        return pd.Series(np.nan, index=grid, dtype=float)
    elapsed = (grid - start).total_seconds()
    return pd.Series(
        elapsed / (end - start).total_seconds(), index=grid, dtype=float
    )


def _reference_progress(cycle: Any, evidence: Any) -> float:
    if cycle.start is None or cycle.end is None or evidence.reference.source == "unavailable":
        return np.nan
    return float(
        (evidence.reference.valid_from - cycle.start).total_seconds()
        / (cycle.end - cycle.start).total_seconds()
    )


def _eligible_keys(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    result: set[tuple[str, str, str]] = set()
    for _, row in frame.iterrows():
        if str(row["eligibility_status"]) in {"eligible", "eligible_exploratory"}:
            result.add(
                (
                    str(row["experiment_id"]),
                    str(row["experiment_date"]),
                    str(row["cycle_id"]),
                )
            )
    return result


def _key_mask(frame: pd.DataFrame, key: tuple[str, str, str]) -> pd.Series[bool]:
    return (
        frame["experiment_id"].astype(str).eq(key[0])
        & frame["experiment_date"].astype(str).eq(key[1])
        & frame["cycle_id"].astype(str).eq(key[2])
    )


def _validate_requested_features(
    requested: Sequence[str],
    values: pd.Series[Any],
    label: str,
) -> tuple[str, ...]:
    available = set(values.dropna().astype(str))
    missing = [feature for feature in requested if feature not in available]
    if missing:
        raise ValueError(f"requested figure features are missing: {missing} in {label}")
    return tuple(requested)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _ordered_dates(values: Sequence[str], settings: EvidenceFigureSettings) -> list[str]:
    actual = sorted({str(value) for value in values if pd.notna(value)})
    configured = [date for date in settings.date_order if date in actual]
    return configured + [date for date in actual if date not in configured]


def _date_colors(dates: Sequence[str]) -> dict[str, str]:
    return {date: _PALETTE[index % len(_PALETTE)] for index, date in enumerate(dates)}


def _raw_label(channel: str, channels: Mapping[str, Mapping[str, Any]]) -> str:
    unit = str(channels.get(channel, {}).get("unit", "")).strip()
    return f"{channel} [{unit}]" if unit else channel


def _short_cycle_label(row: Mapping[str, Any]) -> str:
    date_value = str(row["experiment_date"])
    date_label = date_value.replace("-", "")[-4:]
    return f"{row['experiment_id']} · {date_label} · {row['cycle_id']}"


def _finite_value(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _number_or_nan(value: Any) -> float:
    result = _finite_value(value)
    return result if result is not None else np.nan


def _as_count(value: Any) -> int:
    result = _finite_value(value)
    return int(result) if result is not None else 0


def _text_int(value: Any) -> str:
    return str(_as_count(value))


def _nanmedian(
    values: NDArray[np.float64], axis: int
) -> NDArray[np.float64]:
    if axis == 0:
        result = np.full(values.shape[1], np.nan)
        for index in range(values.shape[1]):
            finite = values[:, index][np.isfinite(values[:, index])]
            if len(finite):
                result[index] = float(np.median(finite))
        return result
    result = np.full(values.shape[0], np.nan)
    for index in range(values.shape[0]):
        finite = values[index, :][np.isfinite(values[index, :])]
        if len(finite):
            result[index] = float(np.median(finite))
    return result


__all__ = [
    "EvidenceFigureSettings",
    "plot_cycle_evolution",
    "plot_evidence_coverage",
    "plot_evidence_profiles",
    "plot_future_horizon_map",
    "plot_pair_similarity",
]
