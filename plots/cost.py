# mypy: ignore-errors
#!/usr/bin/env python3
"""Compare defrost cost optima and render cycle publication PNGs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataloader import DatasetLoader
from dataloader.images import (
    materialize_cycle_image_members,
    scan_cycle_images,
)
from plots.publication import (
    _plot_decision_image,
    match_decision_rgb_images,
    render_decision_publication,
)

STYLES = {
    "v1": ("#0072B2", "D", "V1 optimum"),
    "v2": ("#E69F00", "s", "V2 optimum"),
    "v2.1": ("#009E73", "^", "V2.1 optimum"),
    "v2.2": ("#D55E00", "P", "V2.2 all-water optimum"),
    "v2.3": ("#CC79A7", "X", "V2.3 fixed-9-min optimum"),
    "v2.4": ("#56B4E9", "*", "V2.4 fixed-boundary optimum"),
    "v2.5": ("#6A3D9A", "v", "V2.5 current-cycle optimum"),
    "v2.6": ("#333333", "h", "V2.6 unit-heat optimum"),
    "v3": ("#1B7F79", ">", "V3 robust optimum"),
    "renewal_water": ("#B2182B", "*", "Renewal-water optimum"),
    "v2.6.8": ("#333333", "h", "V2.6.8 diagnostic minimum"),
    "RB": ("#2E7D5B", "o", "Rule defrost"),
}
CURVE_LINESTYLES = {
    "v1": "-",
    "v2": "--",
    "v2.1": "-.",
    "v2.2": ":",
    "v2.3": "-",
    "v2.4": "--",
    "v2.5": "-.",
    "v2.6": ":",
    "v3": "-",
    "renewal_water": "-",
    "v2.6.8": ":",
}
V26_PATCHES = tuple(f"v2.6.{patch}" for patch in range(1, 8))
V26_PATCH_STYLES = {
    "v2.6.1": ("#4C566A", "h", "V2.6.1 baseline"),
    "v2.6.2": ("#3B75AF", "D", "V2.6.2 closed cycle"),
    "v2.6.3": ("#D99032", "s", "V2.6.3 degradation"),
    "v2.6.4": ("#7A5AA6", "^", "V2.6.4 marginal"),
    "v2.6.5": ("#B64A50", "*", "V2.6.5 decision"),
    "v2.6.6": ("#1B7F79", "p", "V2.6.6 diagnostic minimum"),
    "v2.6.7": ("#9A4D8E", "o", "V2.6.7 diagnostic minimum"),
}
CURVE_LINESTYLES.update(
    {
        "v2.6.1": "-",
        "v2.6.2": "--",
        "v2.6.3": "-.",
        "v2.6.4": ":",
        "v2.6.5": (0, (5, 1)),
        "v2.6.6": (0, (3, 1, 1, 1)),
        "v2.6.7": (0, (5, 2)),
    }
)
DATE_BANDS = ("#EAF2F8", "#FFF3E6")
V266_STATUS_MARKERS = {
    "identified_curve": ("p", True, "identified"),
    "measurement_limited": ("s", False, "measurement-limited"),
    "component_extrapolated": ("^", False, "component-extrapolated"),
    "right_censored": (">", False, "right-censored"),
}
V267_STATUS_MARKERS = {
    "identified_curve": ("o", True, "identified"),
    "model_support_limited": ("^", False, "model-support-limited"),
    "measurement_limited": ("s", False, "measurement-limited"),
}
STATUS_MARKERS = {
    "v2.6.6": V266_STATUS_MARKERS,
    "v2.6.7": V267_STATUS_MARKERS,
}
V267_DISPLAY_METRIC = "display_only_inverse_cop"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)


def _save_png(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_svg_png(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style(algorithm: str) -> tuple[str, str, str]:
    base, separator, variant = algorithm.partition("__")
    color, marker, label = (
        V26_PATCH_STYLES[base] if base in V26_PATCH_STYLES else STYLES[base]
    )
    suffix = "diagnostic minimum" if base == "v2.6.8" else "optimum"
    return color, marker, f"{base.upper()} ({variant}) {suffix}" if separator else label


def _run_key(recipe: Mapping[str, object]) -> tuple[str, str]:
    base_cost = str(recipe.get("base_cost", "")).strip().lower()
    if not base_cost:
        raise ValueError("result recipe has no base_cost")
    variant = recipe.get("variant")
    key = base_cost if variant in (None, "") else f"{base_cost}__{variant}"
    label = base_cost.upper() if variant in (None, "") else f"{base_cost.upper()} ({variant})"
    return key, label


def _load_result_tables(
    result_dirs: Sequence[Path], loader: DatasetLoader
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for directory in result_dirs:
        recipe = json.loads((directory / "recipe.json").read_text(encoding="utf-8"))
        key, label = _run_key(recipe)
        if key in tables:
            raise ValueError(f"result runs have the same key: {key}")
        heat_basis = recipe.get("heat_basis")
        if heat_basis not in {"unit", "water"}:
            raise ValueError(f"result has no explicit heat basis: {directory}")
        table = pd.read_csv(directory / "cost.csv")
        table["algorithm"] = key
        table["heat_basis"] = heat_basis
        if "t_RB" not in table:
            table["t_RB"] = pd.NaT
        if "rb_status" not in table:
            table["rb_status"] = "unavailable"
        for _cycle_name, curve in table.groupby("cycle_name", sort=False):
            if key.split("__", 1)[0] == "v2.6.8":
                minimum = pd.to_datetime(
                    curve["diagnostic_minimum"], errors="coerce", format="mixed"
                ).dropna()
                selected = curve.loc[
                    pd.to_datetime(curve["candidate_time"], errors="coerce", format="mixed").eq(
                        minimum.iloc[0] if not minimum.empty else pd.NaT
                    )
                ]
                support_column = "model_supported"
            else:
                selected = curve.loc[curve["is_optimum"].fillna(False)]
                support_column = "supported"
            if selected.empty:
                table.loc[curve.index, "t_star"] = pd.NaT
                table.loc[curve.index, "t_star_model_supported"] = pd.NA
                continue
            selected_row = selected.iloc[0]
            table.loc[curve.index, "t_star"] = pd.to_datetime(
                selected_row["candidate_time"], errors="coerce", format="mixed"
            )
            table.loc[curve.index, "t_star_model_supported"] = selected_row.get(
                support_column, selected_row["optimization_eligible"]
            )
        table.attrs["display_label"] = label
        table.attrs["heat_basis"] = str(heat_basis)
        tables[key] = table
    cycles = sorted(
        set().union(*(set(table["cycle_name"].astype(str)) for table in tables.values()))
    )
    records = {cycle: loader.get_cycle_record(cycle) for cycle in cycles}
    for table in tables.values():
        table["cycle_start"] = table["cycle_name"].map(
            lambda cycle: records[str(cycle)]["boundaries"]["start_time"]
        )
    return tables


def _cycle_points(table: pd.DataFrame, optimum_column: str = "t_star") -> pd.DataFrame:
    values = table.copy()
    for column in ("candidate_time", "cycle_start", optimum_column, "t_RB"):
        values[column] = pd.to_datetime(values[column], errors="coerce", format="mixed")
    rows = []
    for cycle_name, cycle in values.groupby("cycle_name", sort=True):
        first = cycle.iloc[0]
        start = first["cycle_start"]
        support = first.get("t_star_model_supported", True)
        rows.append(
            {
                "cycle_name": str(cycle_name),
                "experiment_id": str(first.get("experiment_id", "unknown")),
                "length_minutes": (cycle["candidate_time"].max() - start).total_seconds() / 60,
                "optimum_minutes": (first[optimum_column] - start).total_seconds() / 60,
                "optimum_supported": (
                    pd.NA
                    if pd.isna(support)
                    else bool(support)
                    and first.get("cycle_status", "identified_curve") == "identified_curve"
                ),
                "cycle_status": str(first.get("cycle_status", "identified_curve")),
                "rb_minutes": (
                    (first["t_RB"] - start).total_seconds() / 60
                    if first.get("rb_status") == "triggered" and pd.notna(first["t_RB"])
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("cycle_name")


def _experiment_date_label(experiment_id: str) -> str:
    date = pd.to_datetime(str(experiment_id).removeprefix("exp_"), format="%Y%m%d")
    return date.strftime("%m-%d")


def _shade_experiment_dates(axis: plt.Axes, experiments: list[str]) -> None:
    start = 0
    groups: list[tuple[int, int, str]] = []
    for index in range(1, len(experiments) + 1):
        if index == len(experiments) or experiments[index] != experiments[start]:
            groups.append((start, index - 1, experiments[start]))
            start = index
    for index, (left, right, experiment_id) in enumerate(groups):
        axis.axvspan(
            left - 0.5,
            right + 0.5,
            color=DATE_BANDS[index % len(DATE_BANDS)],
            zorder=-3,
        )
        if left:
            axis.axvline(left - 0.5, color="#AEB7C2", linewidth=0.6, zorder=-1)
        axis.text(
            (left + right) / 2,
            1.01,
            _experiment_date_label(experiment_id),
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6,
            color="#59636E",
        )


def _comparison_figure(  # noqa: C901
    tables: Mapping[str, pd.DataFrame], algorithms: tuple[str, ...]
) -> plt.Figure:
    points = {algorithm: _cycle_points(tables[algorithm]) for algorithm in algorithms}
    cycle_sets = {algorithm: set(values.index) for algorithm, values in points.items()}
    if len(cycle_sets) > 1 and len({frozenset(cycles) for cycles in cycle_sets.values()}) != 1:
        raise ValueError("comparison families must contain identical cycle sets")
    cycles = sorted(set().union(*(set(values.index) for values in points.values())))
    cycle_ids = [int(cycle.rsplit("_", 1)[-1]) for cycle in cycles]
    x = np.arange(len(cycles))
    figure, axis = plt.subplots(figsize=(max(7.2, 0.19 * len(cycles)), 5.2))
    experiments = [
        next(
            str(values.loc[cycle, "experiment_id"])
            for values in points.values()
            if cycle in values.index
        )
        for cycle in cycles
    ]
    _shade_experiment_dates(axis, experiments)
    lengths = pd.concat(
        [points[name].reindex(cycles)["length_minutes"] for name in algorithms], axis=1
    ).max(axis=1)
    axis.bar(
        x,
        lengths,
        width=0.72,
        color="#D8D8D8",
        alpha=0.65,
        edgecolor="none",
        label="Candidate length",
        zorder=-2,
    )
    offsets = np.zeros(1) if len(algorithms) == 1 else np.linspace(-0.13, 0.13, len(algorithms))
    for offset, algorithm in zip(offsets, algorithms, strict=True):
        color, marker, label = (
            ("#D55E00", "^", "V3 offline decision (supported/RB)")
            if algorithm == "v3_recommended"
            else _style(algorithm)
        )
        values = points[algorithm].reindex(cycles)
        base = algorithm.split("__", 1)[0]
        if base in STATUS_MARKERS:
            markers = STATUS_MARKERS[base]
            no_minimum = values["optimum_minutes"].isna()
            unknown = set(values["cycle_status"].dropna()) - set(markers)
            if unknown or values["cycle_status"].isna().any():
                raise ValueError(
                    f"unrecognized {base.upper()} cycle_status: {sorted(unknown)}"
                )
            for status, (status_marker, filled, status_label) in markers.items():
                selected = values["cycle_status"].eq(status) & ~no_minimum
                if selected.any():
                    axis.scatter(
                        (x + offset)[selected],
                        values.loc[selected, "optimum_minutes"],
                        marker=status_marker,
                        facecolors=color if filled else "none",
                        edgecolors=color,
                        s=24 if filled else 30,
                        label=f"{label} ({status_label})",
                        zorder=3,
                    )
            if no_minimum.any():
                axis.scatter(
                    (x + offset)[no_minimum],
                    np.full(int(no_minimum.sum()), -0.04),
                    transform=axis.get_xaxis_transform(),
                    marker="x",
                    color=color,
                    s=24,
                    linewidths=0.8,
                    clip_on=False,
                    label=f"{label} (no diagnostic minimum)",
                    zorder=4,
                )
            continue
        no_minimum = values["optimum_minutes"].isna()
        supported = values["optimum_supported"].eq(True) & ~no_minimum
        unsupported = values["optimum_supported"].eq(False) & ~no_minimum
        unknown = values["optimum_supported"].isna() & ~no_minimum
        axis.scatter(
            (x + offset)[supported],
            values.loc[supported, "optimum_minutes"],
            color=color,
            marker=marker,
            s=24,
            label=label,
            zorder=3,
        )
        if unsupported.any():
            axis.scatter(
                (x + offset)[unsupported],
                values.loc[unsupported, "optimum_minutes"],
                facecolors="none",
                edgecolors=color,
                marker=marker,
                s=30,
                label=f"{label} (extrapolated)",
                zorder=3,
            )
        if unknown.any():
            axis.scatter(
                (x + offset)[unknown],
                values.loc[unknown, "optimum_minutes"],
                facecolors="#D8D8D8",
                edgecolors=color,
                marker=marker,
                s=30,
                label=f"{label} (support unknown)",
                zorder=3,
            )
        if no_minimum.any():
            axis.scatter(
                (x + offset)[no_minimum],
                np.full(int(no_minimum.sum()), -0.04),
                transform=axis.get_xaxis_transform(),
                marker="x",
                color=color,
                s=24,
                linewidths=0.8,
                clip_on=False,
                label=f"{label} (no diagnostic minimum)",
                zorder=4,
            )
    rb = points[algorithms[0]].reindex(cycles)["rb_minutes"]
    color, marker, label = STYLES["RB"]
    axis.scatter(
        x,
        rb,
        color=color,
        marker=marker,
        facecolors="none",
        s=28,
        label=label,
        zorder=4,
    )
    axis.set(
        xlabel="Cycle ID",
        ylabel="Minutes from cycle start",
        xticks=x,
        xticklabels=cycle_ids,
    )
    axis.tick_params(axis="x", labelrotation=90, labelsize=6)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)
    axis.legend(
        frameon=False,
        ncols=min(5, len(algorithms) + 2),
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _publication_curve(table: pd.DataFrame, label: str) -> pd.DataFrame:
    curve = table.copy()
    if label == "water_reference":
        curve["inverse_cop"] = curve["water_reference_inverse_cop"]
        curve["relative_regret"] = curve["water_reference_relative_regret"]
        curve["t_star"] = curve["water_reference_t_star"]
    return curve


def _with_v267_display_extension(curve: pd.DataFrame) -> pd.DataFrame:
    """Add a plotting-only unsupported ratio without changing formal eligibility."""
    result = curve.copy()
    numerator = pd.to_numeric(result["heating_electricity_kwh"], errors="coerce") + pd.to_numeric(
        result["E_T_hat_kwh"], errors="coerce"
    )
    denominator = pd.to_numeric(result["unit_heating_kwh"], errors="coerce") + pd.to_numeric(
        result["Q_T_hat_kwh"], errors="coerce"
    )
    display = numerator / denominator.where(denominator.gt(0))
    mask = result["measurement_eligible"].eq(True) & result["model_supported"].eq(False)
    result[V267_DISPLAY_METRIC] = display.where(mask & np.isfinite(display))
    return result


def _cost_curve_figure(
    tables: Mapping[str, pd.DataFrame],
    cycle_name: str,
) -> plt.Figure:
    figure, (cost_axis, regret_axis) = plt.subplots(
        2,
        1,
        figsize=(14.4, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    reference = next(iter(tables.values()))
    first = reference.loc[reference["cycle_name"].eq(cycle_name)].iloc[0]
    start = pd.Timestamp(first["cycle_start"])
    for algorithm, table in tables.items():
        curve = table.loc[table["cycle_name"].eq(cycle_name)].copy()
        if curve.empty:
            continue
        curve["candidate_time"] = pd.to_datetime(
            curve["candidate_time"], errors="coerce", format="mixed"
        )
        minutes = (curve["candidate_time"] - start).dt.total_seconds() / 60
        eligible = curve["optimization_eligible"].fillna(False)
        cost = pd.to_numeric(curve["inverse_cop"], errors="coerce").where(eligible)
        regret = (100 * pd.to_numeric(curve["relative_regret"], errors="coerce")).where(eligible)
        color, marker, _label = _style(algorithm)
        linestyle = CURVE_LINESTYLES.get(
            algorithm.split("__", 1)[0], CURVE_LINESTYLES["v2.6"]
        )
        cost_axis.plot(
            minutes,
            cost,
            color=color,
            ls=linestyle,
            lw=1.2,
            label=algorithm.upper(),
        )
        regret_axis.plot(minutes, regret, color=color, ls=linestyle, lw=1.2)
        if algorithm == "v2.6.7":
            extension = _with_v267_display_extension(curve)[V267_DISPLAY_METRIC]
            extension_label = "V2.6.7 unsupported model extension, display only"
            cost_axis.plot(
                minutes,
                extension,
                color="#A69AA8",
                ls="--",
                lw=0.8,
                marker=".",
                ms=2.5,
                alpha=0.75,
                label=extension_label,
            )
        target = pd.to_datetime(curve["t_star"].iloc[0], errors="coerce")
        if pd.notna(target):
            optimum = (curve["candidate_time"] - target).abs().idxmin()
        elif algorithm == "v2.6.7" or cost.dropna().empty:
            continue
        else:
            optimum = cost.idxmin()
        if pd.isna(cost.loc[optimum]) or pd.isna(regret.loc[optimum]):
            continue
        optimum_minute = float(minutes.loc[optimum])
        cost_axis.scatter(
            optimum_minute,
            cost.loc[optimum],
            color=color,
            marker=marker,
            s=26,
            zorder=3,
        )
        regret_axis.scatter(
            optimum_minute,
            regret.loc[optimum],
            color=color,
            marker=marker,
            s=22,
            zorder=3,
        )
    rb = pd.to_datetime(first.get("t_RB"), errors="coerce")
    if first.get("rb_status") == "triggered" and pd.notna(rb):
        rb_minute = (rb - start).total_seconds() / 60
        for axis in (cost_axis, regret_axis):
            axis.axvline(rb_minute, color=STYLES["RB"][0], ls="--", lw=0.8)
    for axis in (cost_axis, regret_axis):
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.45)
    regret_axis.axhline(1, color="#767676", ls=":", lw=0.8, label="1% threshold")
    regret_axis.set_ylim(-0.15, 5)
    cost_axis.set_ylabel("Cost J = 1/COP")
    regret_axis.set(
        xlabel="Minutes from cycle start",
        ylabel="Relative cost [%]",
    )
    cycle_id = int(cycle_name.rsplit("_", 1)[-1])
    status_algorithm = next(
        (
            algorithm
            for algorithm in tables
            if algorithm.split("__", 1)[0] in {"v2.6.7", "v2.6.6"}
        ),
        None,
    )
    status_curve = tables.get(status_algorithm, reference)
    status_row = status_curve.loc[status_curve["cycle_name"].eq(cycle_name)].iloc[0]
    status = (
        f" · {status_algorithm.upper()} {str(status_row['cycle_status']).replace('_', ' ')}"
        if status_algorithm and "cycle_status" in status_row.index
        else ""
    )
    cost_axis.set_title(
        f"Cycle {cycle_id}: cost-function variants{status}", loc="left"
    )
    cost_axis.legend(frameon=False, ncols=4, fontsize=7)
    figure.tight_layout()
    return figure


def _optimal_rgb_figures(
    front_images: Mapping[str, Mapping[str, object]],
    algorithms: tuple[str, ...],
    cycle_name: str,
    start: pd.Timestamp,
):
    """Yield readable four-method front-image plates for one cycle."""
    page_size = 6 if len(algorithms) == 5 else 4
    for page_start in range(0, len(algorithms), page_size):
        page = algorithms[page_start : page_start + page_size]
        rows = max(1, (len(page) + 1) // 2)
        columns = min(2, len(page))
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(6 * columns, 4.4 * rows),
            squeeze=False,
        )
        flat_axes = axes.ravel()
        for axis, algorithm in zip(flat_axes, page, strict=False):
            base = algorithm.split("__", 1)[0]
            info = front_images.get(algorithm, {})
            _plot_decision_image(axis, info, algorithm.upper(), start, pd.NaT)
            target = pd.to_datetime(info.get("target_time"), errors="coerce")
            minute = (
                (pd.Timestamp(target) - start).total_seconds() / 60 if pd.notna(target) else np.nan
            )
            offset = pd.to_numeric(info.get("offset_seconds"), errors="coerce")
            support = info.get("target_supported")
            target_status = str(info.get("target_status", "")).replace("_", " ")
            support_text = (
                " · within support"
                if support is True
                else f" · {target_status or 'extrapolated'}"
                if support is False
                else ""
            )
            detail = (
                f"{minute:.1f} min · image offset {offset:.0f} s{support_text}"
                if bool(info.get("available"))
                else " · ".join(
                    filter(
                        None,
                        (
                            target_status if base in STATUS_MARKERS else "",
                            (
                                "no eligible diagnostic minimum"
                                if base in {"v2.6.7", "v2.6.8"}
                                and info.get("status") == "no_valid_optimal"
                                else str(info.get("status", "unavailable")).replace("_", " ")
                            ),
                        ),
                    )
                )
            )
            axis.set_title(
                (
                    f"{algorithm.upper()} diagnostic minimum\n{detail}"
                    if base in {"v2.6.6", "v2.6.7", "v2.6.8"}
                    else f"{algorithm.upper()} optimum\n{detail}"
                ),
                loc="left",
                fontsize=8,
                pad=5,
            )
            for spine in axis.spines.values():
                spine.set_color(_style(algorithm)[0])
                spine.set_linewidth(1.2)
        for axis in flat_axes[len(page) :]:
            axis.set_visible(False)
        cycle_id = int(cycle_name.rsplit("_", 1)[-1])
        figure.suptitle(
            f"Cycle {cycle_id}: frost appearance at selected/diagnostic cost-function times"
            if {"v2.6.6", "v2.6.7", "v2.6.8"}.intersection(
                algorithm.split("__", 1)[0] for algorithm in page
            )
            else f"Cycle {cycle_id}: frost appearance at cost-function optima",
            x=0.02,
            ha="left",
            fontsize=10,
            fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        yield figure


def _match_optimal_front_images(
    tables: Mapping[str, pd.DataFrame],
    cycle_name: str,
    metadata: pd.DataFrame,
    images: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    matched = {}
    for algorithm, table in tables.items():
        curve = table.loc[table["cycle_name"].eq(cycle_name)]
        target = pd.to_datetime(curve["t_star"].iloc[0], errors="coerce")
        result = match_decision_rgb_images(metadata, images, {"optimal": target}).set_index(
            "target_type"
        )
        matched[algorithm] = result.loc["optimal"].to_dict()
        cycle_status = str(curve.iloc[0].get("cycle_status", "identified_curve"))
        support = curve.get("t_star_model_supported", pd.Series([None])).iloc[0]
        matched[algorithm]["target_supported"] = (
            None
            if support is None or pd.isna(support)
            else bool(support)
            and (
                algorithm.split("__", 1)[0] not in STATUS_MARKERS
                or cycle_status == "identified_curve"
            )
        )
        matched[algorithm]["target_status"] = (
            cycle_status
            if algorithm.split("__", 1)[0] in STATUS_MARKERS
            and cycle_status != "identified_curve"
            else ""
        )
    return matched


def _render_cost_curve_comparisons(
    tables: Mapping[str, pd.DataFrame],
    loader: DatasetLoader,
    output: Path,
    *,
    fetch_cloud: bool = False,
    minimum_free_gib: float = 5,
) -> None:
    algorithms = tuple(tables)
    selected = {algorithm: tables[algorithm] for algorithm in algorithms}
    cycle_sets = {
        algorithm: set(table["cycle_name"].astype(str)) for algorithm, table in selected.items()
    }
    reference_cycles = next(iter(cycle_sets.values()))
    if any(cycles != reference_cycles for cycles in cycle_sets.values()):
        raise ValueError("cost-curve families must contain identical cycle sets")
    cycles = reference_cycles
    reference = next(iter(selected.values()))
    for cycle_name in sorted(cycles):
        cycle_id = int(cycle_name.rsplit("_", 1)[-1])
        start = pd.Timestamp(
            reference.loc[reference["cycle_name"].eq(cycle_name), "cycle_start"].iloc[0]
        )
        metadata = loader.load_image_metadata(cycle_name)
        images = loader.load_cycle_images(cycle_name)
        front_images = _match_optimal_front_images(selected, cycle_name, metadata, images)
        missing = sorted(
            {
                str(info["file_name"])
                for info in front_images.values()
                if info.get("status") == "physical_image_missing" and info.get("file_name")
            }
        )
        source = (
            materialize_cycle_image_members(
                loader.dataset_root,
                cycle_name,
                missing,
                fetch_cloud=True,
                minimum_free_gib=minimum_free_gib,
            )
            if fetch_cloud and missing
            else nullcontext(None)
        )
        with source as cycle_dir:
            if cycle_dir is not None:
                downloaded = scan_cycle_images(
                    loader.dataset_root,
                    cycle_name,
                    metadata,
                    cycle_dir=cycle_dir,
                )
                images = (
                    downloaded
                    if images.empty
                    else pd.concat([images, downloaded], ignore_index=True)
                )
                front_images = _match_optimal_front_images(selected, cycle_name, metadata, images)
            _save_png(
                _cost_curve_figure(selected, cycle_name),
                output / f"cycle_{cycle_id:03d}_cost_curves.png",
            )
            for page, figure in enumerate(
                _optimal_rgb_figures(
                    front_images,
                    algorithms,
                    cycle_name,
                    start,
                ),
                start=1,
            ):
                _save_png(
                    figure,
                    output / "optimal_rgb" / f"cycle_{cycle_id:03d}_optimal_rgb_{page:02d}.png",
                )


def _decision_images(
    metadata: pd.DataFrame, images: pd.DataFrame, curve: pd.DataFrame
) -> dict[str, dict[str, object]]:
    first = curve.iloc[0]
    optimum = (
        first.get("recommended_time") if first.get("algorithm") == "v3" else first.get("t_star")
    )
    rb = first["t_RB"] if first.get("rb_status") == "triggered" else pd.NaT
    matches = match_decision_rgb_images(
        metadata,
        images,
        {"rb": rb, "optimal": optimum},
    )
    return {str(row["target_type"]): row.to_dict() for _, row in matches.iterrows()}


def _render_cycle_sets(  # noqa: C901
    tables: Mapping[str, pd.DataFrame],
    loader: DatasetLoader,
    records: Mapping[str, Mapping[str, object]],
    output: Path,
) -> None:
    suites = {
        f"cost_function_{algorithm}_cycle": (table, _decision_title(algorithm))
        for algorithm, table in tables.items()
    }
    water_reference_columns = {
        "water_reference_t_star",
        "water_reference_inverse_cop",
        "water_reference_relative_regret",
    }
    if "v1" in tables and water_reference_columns.issubset(tables["v1"].columns):
        suites["水侧制热量_cycle"] = (
            _publication_curve(tables["v1"], "water_reference"),
            "Water-heat optimum",
        )
    cycles = sorted(set().union(*(set(table["cycle_name"]) for table, _ in suites.values())))
    for cycle_name in cycles:
        cycle_name = str(cycle_name)
        record = records[cycle_name]
        frame = loader.load_cycle(cycle_name)
        metadata = loader.load_image_metadata(cycle_name)
        images = loader.load_cycle_images(cycle_name)
        for label, (table, title) in suites.items():
            curve = table.loc[table["cycle_name"].eq(cycle_name)]
            if curve.empty:
                continue
            display_metric = None
            minimum_support_label = None
            algorithm = str(curve.iloc[0].get("algorithm", "")).split("__", 1)[0]
            if algorithm == "v2.6.7":
                curve = _with_v267_display_extension(curve)
                display_metric = V267_DISPLAY_METRIC
                cycle_status = str(curve.iloc[0].get("cycle_status", "identified_curve"))
                if cycle_status != "identified_curve":
                    minimum_support_label = cycle_status.replace("_", " ")
            filename = f"cycle_{int(cycle_name.rsplit('_', 1)[-1]):03d}_publication.png"
            render_decision_publication(
                frame,
                record,
                curve,
                _decision_images(metadata, images, curve),
                output / label / filename,
                optimal_label=(
                    f"{title} ({str(curve.iloc[0]['cycle_status']).replace('_', ' ')})"
                    if algorithm in STATUS_MARKERS
                    else title
                ),
                full_candidate_domain=True,
                display_metric=display_metric,
                minimum_label=(
                    "Diagnostic/raw minimum"
                    if algorithm == "v2.6.7"
                    else "Diagnostic minimum"
                    if algorithm == "v2.6.8"
                    else "Minimum"
                ),
                minimum_support_label=minimum_support_label,
            )


def _decision_title(algorithm: str) -> str:
    base = algorithm.split("__", 1)[0]
    if algorithm == "v1":
        return "Unit-heat V1 optimum"
    if algorithm == "v2":
        return "Updated V2 optimum"
    if base == "v3":
        return "V3 offline decision"
    if base in {"v2.6.6", "v2.6.7"}:
        return f"{base.upper()} diagnostic identification minimum"
    return _style(algorithm)[2]


def _plot_bootstrap_stability(bootstrap: pd.DataFrame) -> plt.Figure:
    required = {
        "cycle_name",
        "experiment_id",
        "two_candidate_repeat_fraction",
        "argmin_in_original_5pct_basin_fraction",
    }
    missing = required - set(bootstrap)
    if missing:
        raise ValueError(f"bootstrap audit missing columns: {sorted(missing)}")
    values = bootstrap.copy()
    values["cycle_id"] = values["cycle_name"].str.rsplit("_", n=1).str[-1].astype(int)
    values = values.sort_values("cycle_id").reset_index(drop=True)
    stable = values["two_candidate_repeat_fraction"].ge(0.8) & values[
        "argmin_in_original_5pct_basin_fraction"
    ].ge(0.75)
    stable_fraction = float(stable.mean())
    median_basin = float(values["argmin_in_original_5pct_basin_fraction"].median())
    gate_passes = stable_fraction >= 0.75 and median_basin >= 0.80
    x = np.arange(len(values))
    figure, (cycle_axis, experiment_axis) = plt.subplots(
        2,
        1,
        figsize=(12.8, 5.8),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    _shade_experiment_dates(cycle_axis, values["experiment_id"].astype(str).tolist())
    cycle_axis.plot(
        x,
        values["two_candidate_repeat_fraction"],
        color="#3775BA",
        marker="o",
        markersize=3,
        linewidth=1,
        label="Within two candidates",
    )
    cycle_axis.plot(
        x,
        values["argmin_in_original_5pct_basin_fraction"],
        color="#9A4D8E",
        marker="s",
        markersize=3,
        linewidth=1,
        label="Argmin in original 5% basin",
    )
    cycle_axis.axhline(0.8, color="#3775BA", linestyle=":", linewidth=0.9, label="0.80 gate")
    cycle_axis.axhline(0.75, color="#9A4D8E", linestyle=":", linewidth=0.9, label="0.75 gate")
    cycle_axis.set(
        ylabel="Bootstrap fraction",
        xticks=x,
        xticklabels=values["cycle_id"],
        ylim=(0, 1.04),
    )
    cycle_axis.tick_params(axis="x", labelrotation=90, labelsize=6)
    cycle_axis.legend(frameon=False, ncols=4, fontsize=7, loc="lower left")
    cycle_axis.grid(axis="y", color="#D8D8D8", linewidth=0.5)

    experiment = (
        pd.DataFrame({"experiment_id": values["experiment_id"], "stable": stable})
        .groupby("experiment_id", sort=False)["stable"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )
    colors = ["#7884B4" if value >= 0.75 else "#C6C6CC" for value in experiment["mean"]]
    experiment_axis.bar(np.arange(len(experiment)), experiment["mean"], color=colors, width=0.72)
    experiment_axis.axhline(
        0.75,
        color="#767676",
        linestyle=":",
        linewidth=0.8,
        label="0.75 descriptive reference",
    )
    for index, row in experiment.iterrows():
        experiment_axis.text(
            index,
            min(1.02, float(row["mean"]) + 0.04),
            f"{int(row['sum'])}/{int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    experiment_axis.set(
        ylabel="Stable cycles",
        xlabel="Held-out experiment date",
        xticks=np.arange(len(experiment)),
        xticklabels=[_experiment_date_label(item) for item in experiment["experiment_id"]],
        ylim=(0, 1.12),
        title="Per-experiment descriptive fractions; global gate uses all cycles",
    )
    experiment_axis.tick_params(axis="x", labelrotation=45, labelsize=6)
    experiment_axis.grid(axis="y", color="#D8D8D8", linewidth=0.5)
    figure.suptitle(
        f"Whole-experiment bootstrap {'passes' if gate_passes else 'fails'} "
        f"the hard-label gate: "
        f"{int(stable.sum())}/{len(values)} stable "
        f"({stable_fraction:.1%}) [gate >=75%]; median basin hit "
        f"{median_basin:.1%} (gate >=80%)",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def _plot_ticket_loeo(loeo: pd.DataFrame, target: str) -> plt.Figure:
    required = {
        "experiment_id",
        "target",
        "observed_kwh",
        "loeo_prediction_kwh",
        "training_mean_kwh",
        "supported",
    }
    missing = required - set(loeo)
    if missing:
        raise ValueError(f"ticket LOEO audit missing columns: {sorted(missing)}")
    values = loeo.loc[loeo["target"].eq(target)].copy()
    if values.empty:
        raise ValueError(f"ticket LOEO audit has no {target} rows")
    supported = values["supported"].fillna(False).astype(bool)
    gate = values.loc[supported].copy()
    gate["model_sq_error"] = (gate["observed_kwh"] - gate["loeo_prediction_kwh"]) ** 2
    gate["baseline_sq_error"] = (gate["observed_kwh"] - gate["training_mean_kwh"]) ** 2
    event_ratio = gate["model_sq_error"].mean() / gate["baseline_sq_error"].mean()
    macro = gate.groupby("experiment_id")[["model_sq_error", "baseline_sq_error"]].mean()
    macro_ratio = macro["model_sq_error"].mean() / macro["baseline_sq_error"].mean()
    figure, axis = plt.subplots(figsize=(6.8, 5.8))
    unsupported = values.loc[~supported]
    if not unsupported.empty:
        axis.scatter(
            unsupported["observed_kwh"],
            unsupported["loeo_prediction_kwh"],
            color="#C8C8CC",
            marker="x",
            s=22,
            label=f"Outside support (display only, n={len(unsupported)})",
            zorder=1,
        )
    axis.scatter(
        gate["observed_kwh"],
        gate["training_mean_kwh"],
        facecolors="none",
        edgecolors="#D99032",
        marker="s",
        s=28,
        linewidths=0.8,
        label="Training-mean baseline",
        zorder=2,
    )
    axis.scatter(
        gate["observed_kwh"],
        gate["loeo_prediction_kwh"],
        color="#3775BA",
        marker="o",
        s=25,
        label="LOEO prediction (supported gate cohort)",
        zorder=3,
    )
    bounds = pd.concat(
        [
            values["observed_kwh"],
            values["loeo_prediction_kwh"],
            values["training_mean_kwh"],
        ]
    )
    low, high = float(bounds.min()), float(bounds.max())
    pad = max((high - low) * 0.06, 0.01)
    axis.plot([low - pad, high + pad], [low - pad, high + pad], color="#767676", ls=":", lw=0.9)
    axis.set(
        xlabel=f"Observed {target} [kWh]",
        ylabel=f"Predicted {target} [kWh]",
        xlim=(low - pad, high + pad),
        ylim=(low - pad, high + pad),
        title=(
            f"{target} experiment-LOEO: supported n={len(gate)}/{len(values)}, "
            f"experiments={gate['experiment_id'].nunique()}\n"
            f"MSE ratio vs training mean: event={event_ratio:.3f}, macro={macro_ratio:.3f}"
        ),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#D8D8D8", linewidth=0.45)
    axis.legend(frameon=False, fontsize=7, loc="upper left")
    figure.tight_layout()
    return figure


def generate_v267_evidence(
    bootstrap: pd.DataFrame,
    loeo: pd.DataFrame,
    output: Path,
    cycle_audit: pd.DataFrame | None = None,
) -> None:
    """Write the three independent V2.6.7 gate-evidence PNGs."""
    if "experiment_id" not in bootstrap:
        if cycle_audit is None:
            raise ValueError("cycle audit is required to map bootstrap experiments")
        bootstrap = bootstrap.merge(
            cycle_audit[["cycle_name", "experiment_id"]], on="cycle_name", validate="one_to_one"
        )
    _save_png(_plot_bootstrap_stability(bootstrap), output / "bootstrap_stability_by_cycle.png")
    for target in ("E_T", "Q_T"):
        _save_png(_plot_ticket_loeo(loeo, target), output / f"ticket_{target}_loeo.png")


def generate_cost_function_figures(
    result_dirs: Sequence[Path],
    loader: DatasetLoader,
    output: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Render cost figures from standardized cost run directories."""
    tables = _load_result_tables(result_dirs, loader)
    cycles = sorted(set().union(*(set(table["cycle_name"]) for table in tables.values())))
    records = {cycle: loader.get_cycle_record(str(cycle)) for cycle in cycles}
    algorithms = tuple(tables)
    families = tuple((algorithm,) for algorithm in algorithms)
    if len(algorithms) > 1:
        families += (algorithms,)
    comparisons = [
        (
            family,
            output / f"comparison_{'_'.join(family)}_RB.png",
            _save_svg_png
            if any(algorithm.split("__", 1)[0] == "renewal_water" for algorithm in family)
            else _save_png,
        )
        for family in families
    ]
    if not overwrite:
        for _family, comparison, saver in comparisons:
            targets = (
                (comparison, comparison.with_suffix(".svg"))
                if saver is _save_svg_png
                else (comparison,)
            )
            for target in targets:
                if target.exists():
                    raise FileExistsError(f"comparison exists; pass --overwrite: {target}")
    for family, comparison, saver in comparisons:
        saver(_comparison_figure(tables, family), comparison)
    _render_cycle_sets(tables, loader, records, output)
    for heat_basis in dict.fromkeys(table.attrs["heat_basis"] for table in tables.values()):
        _render_cost_curve_comparisons(
            {
                algorithm: table
                for algorithm, table in tables.items()
                if table.attrs["heat_basis"] == heat_basis
            },
            loader,
            output / "cost_curves" / heat_basis,
        )
