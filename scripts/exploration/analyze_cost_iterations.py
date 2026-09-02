#!/usr/bin/env python3
"""Summarize and visualize the five V2.6 cost-function iterations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataloader.dataloader import DatasetLoader
from frost_analysis.cost.core import water_side_heating_kw

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = tuple(f"v2.6.{number}" for number in range(1, 6))
COLORS = dict(zip(VERSIONS, ("#4C566A", "#3B75AF", "#D99032", "#7A5AA6", "#B64A50"), strict=True))
MARKERS = dict(zip(VERSIONS, ("o", "D", "s", "^", "*"), strict=True))
FORMULAS = {
    "v2.6.1": "(E_H + E_D) / (Q_H,u + Q_prep,w - Q_D,w)",
    "v2.6.2": "(E_H,S->t + E_D + E_R) / (Q_H,u,S->t + Q_prep - Q_D + Q_R)",
    "v2.6.3": "lambda_0 + (L + K) / Q_H",
    "v2.6.4": "lambda_0 + |Delta G_5min| / Delta Q_H,5min",
    "v2.6.5": "latest supported Delta G>=0 point in connected 1% average-cost basin",
}
MODIFICATIONS = {
    "v2.6.1": "Current-cycle ratio baseline",
    "v2.6.2": "Stable-to-stable recovery accounting",
    "v2.6.3": "Baseline-normalized frost degradation",
    "v2.6.4": "Five-minute LOEO marginal balance",
    "v2.6.5": "Average basin + marginal confirmation + support",
}
EARLY_PROBLEMS = {
    "v2.6.1": "Recovery COP is rewarded in a finite ratio",
    "v2.6.2": "Recovery reward removed; cumulative ratio remains",
    "v2.6.3": "Total cost separated from avoidable degradation",
    "v2.6.4": "Local zero crossings recreate early optima",
    "v2.6.5": "Early action avoided unless degradation is supported",
}
NEW_PROBLEMS = {
    "v2.6.1": "High-COP recovery dominates short cycles",
    "v2.6.2": "Flat cumulative minima remain ambiguous",
    "v2.6.3": "Average objective still smooths transient evidence",
    "v2.6.4": "Noise-sensitive local marginal optimum",
    "v2.6.5": "13 labels remain extrapolated or right-censored",
}


def _save(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(f"{output}.svg", bbox_inches="tight", facecolor="white")
    figure.savefig(f"{output}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _read_tables(cost_root: Path, loader: DatasetLoader) -> dict[str, pd.DataFrame]:
    starts = loader.list_cycles().set_index("cycle_name")["start_time"].pipe(pd.to_datetime)
    tables = {}
    for version in VERSIONS:
        table = pd.read_csv(cost_root / f"cost_function_{version}.csv")
        for column in (
            "candidate_time",
            "t_star",
            "raw_t_star",
            "t_RB",
            "actual_preparation_time",
        ):
            if column in table:
                table[column] = pd.to_datetime(table[column], errors="coerce", format="mixed")
        table["cycle_start"] = table["cycle_name"].map(starts)
        table["cycle_id"] = table["cycle_name"].str.rsplit("_", n=1).str[-1].astype(int)
        table["minutes"] = (table["candidate_time"] - table["cycle_start"]).dt.total_seconds() / 60
        tables[version] = table
    return tables


def connected_basin_width(curve: pd.DataFrame, raw_time: pd.Timestamp, threshold: float) -> float:
    """Width of the near-optimal segment that contains the raw minimum."""
    values = curve.sort_values("candidate_time", kind="stable").reset_index(drop=True)
    raw = values["candidate_time"].eq(pd.Timestamp(raw_time))
    near = values["relative_regret"].le(threshold).fillna(False)
    if not raw.any() or not near.loc[raw].all():
        return np.nan
    segments = near.ne(near.shift(fill_value=False)).cumsum()
    basin = near & segments.eq(segments.loc[raw].iloc[0])
    return float(values.loc[basin, "minutes"].max() - values.loc[basin, "minutes"].min())


def _status(first: pd.Series, version: str) -> str:
    if version == "v2.6.5":
        return str(first["decision_status"])
    if str(first["minimum_location"]).startswith("right"):
        return "right_censored_lower_bound"
    return "supported_optimal" if bool(first["t_star_model_supported"]) else "extrapolated"


def _cycle_rows(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for version, table in tables.items():
        for cycle_name, curve in table.groupby("cycle_name", sort=False):
            first = curve.iloc[0]
            raw_time = first.get("raw_t_star", first["t_star"])
            if pd.isna(raw_time):
                raw_time = first["t_star"]
            status = _status(first, version)
            t_star = pd.Timestamp(first["t_star"])
            start = pd.Timestamp(first["cycle_start"])
            rows.append(
                {
                    "version": version,
                    "cycle_name": cycle_name,
                    "cycle_id": int(first["cycle_id"]),
                    "experiment_id": str(first["experiment_id"]),
                    "date": start.normalize(),
                    "t_star": t_star,
                    "raw_t_star": pd.Timestamp(raw_time),
                    "t_star_minutes": (t_star - start).total_seconds() / 60,
                    "raw_t_star_minutes": (pd.Timestamp(raw_time) - start).total_seconds() / 60,
                    "rb_minutes": (
                        (pd.Timestamp(first["t_RB"]) - start).total_seconds() / 60
                        if first.get("rb_status") == "triggered" and pd.notna(first.get("t_RB"))
                        else np.nan
                    ),
                    "actual_minutes": (
                        (pd.Timestamp(first["actual_preparation_time"]) - start).total_seconds()
                        / 60
                        if pd.notna(first.get("actual_preparation_time"))
                        else np.nan
                    ),
                    "candidate_length_minutes": float(curve["minutes"].max()),
                    "status": status,
                    "minimum_location": str(first["minimum_location"]),
                    "supported": bool(first["t_star_model_supported"]),
                    "hard_label": (
                        bool(first["hard_label_eligible"])
                        if version == "v2.6.5"
                        else status == "supported_optimal"
                    ),
                    "basin_1pct_minutes": connected_basin_width(curve, raw_time, 0.01),
                    "basin_5pct_minutes": connected_basin_width(curve, raw_time, 0.05),
                }
            )
    return pd.DataFrame(rows)


def build_version_summary(cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    previous = None
    for version in VERSIONS:
        values = cycles.loc[cycles["version"].eq(version)].sort_values("cycle_name")
        times = values.set_index("cycle_name")["t_star_minutes"]
        spearman = np.nan if previous is None else times.corr(previous, method="spearman")
        rows.append(
            {
                "Version": version.upper(),
                "formula_short": FORMULAS[version],
                "main_modification": MODIFICATIONS[version],
                "early_problem": EARLY_PROBLEMS[version],
                "new_problem": NEW_PROBLEMS[version],
                "median_minutes": times.median(),
                "IQR_minutes": times.quantile(0.75) - times.quantile(0.25),
                "range_minutes": f"{times.min():.2f}-{times.max():.2f}",
                "Spearman_vs_previous": spearman,
                "early_count_lt40": int(times.lt(40).sum()),
                "boundary_count": int(values["minimum_location"].ne("interior").sum()),
                "supported_count": int(values["supported"].sum()),
                "hard_label_count": int(values["hard_label"].sum()),
                "basin_1pct_median_minutes": values["basin_1pct_minutes"].median(),
                "basin_5pct_median_minutes": values["basin_5pct_minutes"].median(),
            }
        )
        previous = times
    return pd.DataFrame(rows)


def _date_bands(axis: plt.Axes, values: pd.DataFrame) -> None:
    ordered = values.sort_values("cycle_id").drop_duplicates("cycle_id")
    ids = ordered["cycle_id"].to_numpy()
    for index, (_date, group) in enumerate(ordered.groupby("date", sort=False)):
        positions = np.flatnonzero(ordered["date"].eq(group["date"].iloc[0]))
        left_i, right_i = positions[0], positions[-1]
        left = ids[left_i] - (ids[left_i] - ids[left_i - 1]) / 2 if left_i else ids[left_i] - 1
        right = (
            ids[right_i] + (ids[right_i + 1] - ids[right_i]) / 2
            if right_i + 1 < len(ids)
            else ids[right_i] + 1
        )
        axis.axvspan(left, right, color=("#EAF2F8", "#FFF3E6")[index % 2], zorder=-3)
        axis.text(
            (left + right) / 2,
            0.985,
            group["date"].iloc[0].strftime("%m-%d"),
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6,
            color="#59636E",
        )


def plot_summary_table(summary: pd.DataFrame, output: Path) -> None:
    shown = summary.assign(
        median_IQR=lambda x: x.apply(
            lambda row: f"{row.median_minutes:.1f} ({row.IQR_minutes:.1f})", axis=1
        ),
        Spearman=lambda x: x["Spearman_vs_previous"].map(
            lambda v: "-" if pd.isna(v) else f"{v:.2f}"
        ),
        basin=lambda x: x.apply(
            lambda row: (
                f"{row.basin_1pct_median_minutes:.1f} / {row.basin_5pct_median_minutes:.1f}"
            ),
            axis=1,
        ),
        status=lambda x: x.apply(
            lambda row: f"{row.supported_count}/{row.hard_label_count}/{row.boundary_count}", axis=1
        ),
    )
    columns = [
        "Version",
        "median_IQR",
        "range_minutes",
        "early_count_lt40",
        "Spearman",
        "status",
        "basin",
    ]
    labels = [
        "Version",
        "Median (IQR), min",
        "Range, min",
        "<40 min",
        "rho prev.",
        "Support / hard / edge",
        "Basin 1%/5%, min",
    ]
    figure, axis = plt.subplots(figsize=(10.2, 2.8))
    axis.axis("off")
    axis.set_title(
        "Closing the recovery boundary removes the dominant early-defrost artefact",
        loc="left",
        weight="bold",
        pad=14,
    )
    table = axis.table(
        cellText=shown[columns].values, colLabels=labels, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.55)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#E8EDF3" if row == 0 else ("#F6F8FA" if row % 2 else "white"))
        if row == 0:
            cell.set_text_props(weight="bold")
    axis.text(
        0,
        -0.02,
        "Source: five formal candidate-level cost-function CSVs; n=69 cycles.",
        transform=axis.transAxes,
        color="#606060",
        fontsize=7,
    )
    _save(figure, output)


def plot_optimum_distribution(cycles: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(13.0, 5.2))
    reference = cycles.loc[cycles["version"].eq(VERSIONS[0])].sort_values("cycle_id")
    _date_bands(axis, reference)
    axis.vlines(
        reference["cycle_id"],
        0,
        reference["candidate_length_minutes"],
        color="#C8CDD3",
        lw=1.0,
        label="Candidate duration",
    )
    offsets = dict(zip(VERSIONS, np.linspace(-0.28, 0.28, len(VERSIONS)), strict=True))
    for version in VERSIONS:
        values = cycles.loc[cycles["version"].eq(version)]
        for status, marker, fill in (
            ("supported_optimal", MARKERS[version], COLORS[version]),
            ("extrapolated", MARKERS[version], "white"),
            ("extrapolated_raw_optimum", MARKERS[version], "white"),
            ("right_censored_lower_bound", "x", COLORS[version]),
        ):
            selected = values.loc[values["status"].eq(status)]
            style = (
                {"color": COLORS[version]}
                if marker == "x"
                else {
                    "facecolor": fill,
                    "edgecolor": COLORS[version],
                }
            )
            axis.scatter(
                selected["cycle_id"] + offsets[version],
                selected["t_star_minutes"],
                s=24 if version != VERSIONS[-1] else 38,
                marker=marker,
                linewidth=0.9,
                zorder=3,
                label=version.upper() if status == "supported_optimal" else None,
                **style,
            )
    axis.scatter(
        reference["cycle_id"],
        reference["rb_minutes"],
        s=20,
        marker="|",
        color="#238B57",
        linewidth=1.6,
        label="Rule defrost",
    )
    axis.set(
        xlabel="Cycle ID",
        ylabel="Minutes from cycle start",
        title="Five iterations change cycle ranking, not merely the global timing offset",
    )
    axis.set_title(axis.get_title(), pad=12)
    axis.text(
        0.01,
        0.94,
        "solid: supported   open: extrapolated   x: censored",
        transform=axis.transAxes,
        fontsize=7,
        color="#4D5660",
    )
    axis.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    axis.grid(axis="y", color="#D7DCE1", lw=0.5)
    _save(figure, output)


def plot_scatter(cycles: pd.DataFrame, output: Path) -> None:
    left = cycles.loc[cycles["version"].eq("v2.6.1")].set_index("cycle_name")
    right = cycles.loc[cycles["version"].eq("v2.6.5")].set_index("cycle_name")
    values = left[["t_star_minutes"]].join(
        right[["t_star_minutes", "status", "cycle_id"]], lsuffix="_v261", rsuffix="_v265"
    )
    palette = {
        "supported_optimal": "#3B75AF",
        "extrapolated_raw_optimum": "#D99032",
        "right_censored_lower_bound": "#B64A50",
    }
    figure, axis = plt.subplots(figsize=(5.4, 5.0))
    for status, group in values.groupby("status"):
        axis.scatter(
            group["t_star_minutes_v261"],
            group["t_star_minutes_v265"],
            s=28,
            color=palette.get(status, "#777777"),
            label=status.replace("_", " "),
            alpha=0.85,
        )
    limits = [
        min(values.filter(like="t_star").min()) - 4,
        max(values.filter(like="t_star").max()) + 4,
    ]
    axis.plot(limits, limits, ls="--", color="#777777", lw=0.9)
    for cycle_id in (85, 86, 30, 60):
        hit = values.loc[values["cycle_id"].eq(cycle_id)]
        if not hit.empty:
            row = hit.iloc[0]
            axis.annotate(
                str(cycle_id),
                (row["t_star_minutes_v261"], row["t_star_minutes_v265"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
    axis.set(
        xlim=limits,
        ylim=limits,
        xlabel="V2.6.1 optimum (min)",
        ylabel="V2.6.5 decision (min)",
        title="V2.6.5 selectively corrects early optima",
    )
    axis.legend(fontsize=7)
    axis.grid(color="#D7DCE1", lw=0.5)
    _save(figure, output)


def _curve(tables: dict[str, pd.DataFrame], version: str, cycle_name: str) -> pd.DataFrame:
    return (
        tables[version]
        .loc[tables[version]["cycle_name"].eq(cycle_name)]
        .sort_values("candidate_time")
    )


def plot_cost_shape(
    tables: dict[str, pd.DataFrame], cycle_name: str, kind: str, output: Path
) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    for version in VERSIONS:
        curve = _curve(tables, version, cycle_name)
        axis.plot(
            curve["minutes"],
            100 * curve["relative_regret"],
            color=COLORS[version],
            lw=1.6,
            label=version.upper(),
        )
        first = curve.iloc[0]
        chosen = curve.loc[curve["candidate_time"].eq(first["t_star"])]
        if not chosen.empty:
            axis.scatter(
                chosen["minutes"],
                100 * chosen["relative_regret"],
                color=COLORS[version],
                marker=MARKERS[version],
                s=30,
                zorder=4,
            )
        if version == "v2.6.5" and pd.notna(first.get("raw_t_star")):
            raw = curve.loc[curve["candidate_time"].eq(first["raw_t_star"])]
            axis.scatter(
                raw["minutes"],
                100 * raw["relative_regret"],
                facecolor="white",
                edgecolor="#222222",
                s=38,
                zorder=5,
                label="V2.6.5 raw minimum",
            )
    axis.axhline(1, color="#777777", ls="--", lw=0.8, label="1% regret")
    axis.set(
        xlabel="Minutes from cycle start",
        ylabel="Relative regret (%)",
        title=f"{kind.title()} cycle {int(cycle_name[-6:])}: decision movement follows curve shape",
    )
    axis.set_ylim(bottom=-0.15)
    axis.grid(color="#D7DCE1", lw=0.5)
    axis.legend(ncol=2, fontsize=7)
    _save(figure, output)


def plot_flatness(cycles: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    positions, data, colors, labels = [], [], [], []
    for index, version in enumerate(VERSIONS):
        values = cycles.loc[cycles["version"].eq(version)]
        for offset, column, _shade, suffix in (
            (-0.17, "basin_1pct_minutes", 1.0, "1%"),
            (0.17, "basin_5pct_minutes", 0.45, "5%"),
        ):
            positions.append(index + offset)
            data.append(values[column].dropna())
            colors.append(COLORS[version])
            labels.append(suffix)
    boxes = axis.boxplot(
        data,
        positions=positions,
        widths=0.27,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020"},
    )
    for index, patch in enumerate(boxes["boxes"]):
        patch.set_facecolor(colors[index])
        patch.set_alpha(0.85 if labels[index] == "1%" else 0.35)
        patch.set_edgecolor(colors[index])
    axis.set_xticks(range(len(VERSIONS)), [version.upper() for version in VERSIONS])
    axis.set(
        ylabel="Connected basin width (min)",
        title="Near-optimal ambiguity is measured around the raw minimum, not a global envelope",
    )
    axis.grid(axis="y", color="#D7DCE1", lw=0.5)
    axis.text(
        0.99,
        0.97,
        "dark: 1%   pale: 5%",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )
    _save(figure, output)


def plot_v264_failure(tables: dict[str, pd.DataFrame], cycle_name: str, output: Path) -> None:
    v3, v4, v5 = (_curve(tables, version, cycle_name) for version in ("v2.6.3", "v2.6.4", "v2.6.5"))
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), sharex=True)
    axes[0].plot(v3["minutes"], 100 * v3["relative_regret"], color=COLORS["v2.6.3"], lw=1.6)
    axes[0].fill_between(
        v3["minutes"],
        0,
        100 * v3["relative_regret"],
        where=v3["near_optimal_1pct"].astype(bool),
        color=COLORS["v2.6.3"],
        alpha=0.2,
    )
    axes[0].set(title="Average-cost basin", ylabel="Relative regret (%)")
    marginal = v4["marginal_delta_g_kwh"] / v4["marginal_delta_heating_kwh"]
    axes[1].plot(v4["minutes"], marginal, color=COLORS["v2.6.4"], lw=1.2)
    axes[1].axhline(0, color="#555555", ls="--", lw=0.8)
    axes[1].axvline(
        (v4.iloc[0]["t_star"] - v4.iloc[0]["cycle_start"]).total_seconds() / 60,
        color=COLORS["v2.6.4"],
        lw=1.2,
    )
    axes[1].set(title="Local marginal balance", ylabel="Delta G / Delta Q_H")
    axes[2].plot(v5["minutes"], 100 * v5["relative_regret"], color=COLORS["v2.6.5"], lw=1.6)
    first = v5.iloc[0]
    for label, column, style in (("raw", "raw_t_star", "--"), ("decision", "t_star", "-")):
        minute = (pd.Timestamp(first[column]) - first["cycle_start"]).total_seconds() / 60
        axes[2].axvline(
            minute,
            color=("#555555" if label == "raw" else COLORS["v2.6.5"]),
            ls=style,
            lw=1.2,
            label=label,
        )
    axes[2].set(title="Basin-confirmed decision", ylabel="Relative regret (%)")
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.set_xlabel("Minutes from cycle start")
        axis.grid(color="#D7DCE1", lw=0.5)
    figure.suptitle(
        f"Cycle {int(cycle_name[-6:])}: a local zero crossing is not a defensible global optimum",
        weight="bold",
    )
    _save(figure, output)


def plot_label_status(cycles: pd.DataFrame, output: Path) -> None:
    values = cycles.loc[cycles["version"].eq("v2.6.5")]
    counts = (
        values["status"]
        .value_counts()
        .reindex(
            ["supported_optimal", "extrapolated_raw_optimum", "right_censored_lower_bound"],
            fill_value=0,
        )
    )
    figure, axis = plt.subplots(figsize=(5.7, 3.8))
    bars = axis.bar(
        ["Supported\noptimal", "Extrapolated", "Right-censored\nlower bound"],
        counts,
        color=("#3B75AF", "#D99032", "#B64A50"),
        width=0.62,
    )
    axis.bar_label(bars, padding=3)
    hard = int(values["hard_label"].sum())
    axis.text(
        0.98,
        0.95,
        f"Hard CV labels: {hard}/69",
        transform=axis.transAxes,
        ha="right",
        va="top",
        weight="bold",
    )
    axis.set(
        ylabel="Number of cycles", title="V2.6.5 separates defensible labels from domain limits"
    )
    axis.grid(axis="y", color="#D7DCE1", lw=0.5)
    _save(figure, output)


def build_early_diagnostics(tables: dict[str, pd.DataFrame], cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    component_columns = {
        "EH_kwh": ("stable_heating_electricity_kwh", "heating_electricity_kwh"),
        "QH_kwh": ("stable_unit_heating_kwh", "unit_heating_kwh"),
        "ED_kwh": ("defrost_electricity_kwh",),
        "Qprep_kwh": ("preparation_heat_kwh",),
        "QD_kwh": ("defrost_absorbed_heat_kwh",),
        "ER_kwh": ("projected_recovery_electricity_kwh", "recovery_electricity_kwh"),
        "QR_kwh": ("projected_recovery_heat_kwh", "recovery_heat_kwh"),
        "L_kwh": ("heating_degradation_electricity_kwh",),
        "K_kwh": ("transition_excess_electricity_kwh",),
    }
    for version in VERSIONS:
        selected = cycles.loc[cycles["version"].eq(version)].nsmallest(5, "t_star_minutes")
        for _, summary in selected.iterrows():
            curve = _curve(tables, version, summary["cycle_name"])
            point = curve.loc[curve["candidate_time"].eq(summary["t_star"])].iloc[0]
            row = {
                "version": version,
                "cycle_name": summary["cycle_name"],
                "cycle_id": summary["cycle_id"],
                "t_star_minutes": summary["t_star_minutes"],
                "raw_t_star_minutes": summary["raw_t_star_minutes"],
                "minimum_location": summary["minimum_location"],
                "decision_status": summary["status"],
                "delta_vs_RB_minutes": summary["t_star_minutes"] - summary["rb_minutes"],
                "delta_vs_actual_minutes": summary["t_star_minutes"] - summary["actual_minutes"],
                "model_supported": bool(point.get("model_supported", False)),
                "hard_label_eligible": summary["hard_label"],
            }
            for target, sources in component_columns.items():
                source = next((name for name in sources if name in point.index), None)
                row[target] = (
                    pd.to_numeric(point.get(source), errors="coerce") if source else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _early_cycles(cycles: pd.DataFrame) -> dict[str, list[str]]:
    result = {}
    for version in VERSIONS:
        values = cycles.loc[cycles["version"].eq(version)].sort_values("t_star_minutes")
        eligible = values.loc[~values["status"].eq("right_censored_lower_bound")]
        result[version] = [str(eligible.iloc[0]["cycle_name"])]
    v4 = cycles.loc[cycles["version"].eq("v2.6.4") & cycles["cycle_id"].isin([30, 85])].nsmallest(
        1, "t_star_minutes"
    )
    result["v2.6.4"] = [str(v4.iloc[0]["cycle_name"])]
    v5 = cycles.loc[cycles["version"].eq("v2.6.5")].sort_values("t_star_minutes")
    earliest, hard = v5.iloc[0], v5.loc[v5["hard_label"]].iloc[0]
    result["v2.6.5"] = list(dict.fromkeys([str(earliest["cycle_name"]), str(hard["cycle_name"])]))
    return result


def plot_early_diagnosis(
    tables: dict[str, pd.DataFrame], version: str, cycle_names: list[str], output: Path
) -> None:
    figure, axes = plt.subplots(
        1, 3, figsize=(11.2, 3.5), gridspec_kw={"width_ratios": [1.45, 1.25, 1.0]}
    )
    component_names = ["EH", "ED", "ER", "QH", "Qprep", "QD", "QR"]
    text = []
    for index, cycle_name in enumerate(cycle_names):
        curve = _curve(tables, version, cycle_name)
        first = curve.iloc[0]
        label = f"cycle {int(cycle_name[-6:])}"
        axes[0].plot(
            curve["minutes"],
            100 * curve["relative_regret"],
            color=COLORS[version],
            lw=1.5,
            ls=("-" if index == 0 else "--"),
            label=label,
        )
        chosen = curve.loc[curve["candidate_time"].eq(first["t_star"])].iloc[0]
        axes[0].scatter(
            chosen["minutes"],
            100 * chosen["relative_regret"],
            color=COLORS[version],
            marker=("o" if index == 0 else "D"),
            zorder=4,
        )
        values = np.array(
            [
                chosen.get("stable_heating_electricity_kwh", chosen.get("heating_electricity_kwh")),
                chosen.get("defrost_electricity_kwh"),
                chosen.get(
                    "projected_recovery_electricity_kwh", chosen.get("recovery_electricity_kwh")
                ),
                chosen.get("stable_unit_heating_kwh", chosen.get("unit_heating_kwh")),
                chosen.get("preparation_heat_kwh"),
                chosen.get("defrost_absorbed_heat_kwh"),
                chosen.get("projected_recovery_heat_kwh", chosen.get("recovery_heat_kwh")),
            ],
            dtype=float,
        )
        normalized = values.copy()
        normalized[:3] /= np.nanmax(np.abs(values[:3]))
        normalized[3:] /= np.nanmax(np.abs(values[3:]))
        axes[1].plot(
            component_names,
            normalized,
            marker=("o" if index == 0 else "D"),
            color=COLORS[version],
            ls=("-" if index == 0 else "--"),
            label=label,
        )
        status = first.get("decision_status", _status(first, version))
        raw_minutes = (
            pd.Timestamp(first.get("raw_t_star", first["t_star"])) - first["cycle_start"]
        ).total_seconds() / 60
        text.append(
            f"{label}\n"
            f"decision = {chosen['minutes']:.1f} min\n"
            f"raw min = {raw_minutes:.1f} min\n"
            f"status = {str(status).replace('_', ' ')}\n"
            f"model support = {bool(chosen.get('model_supported', False))}"
        )
    axes[0].axhline(1, color="#777777", ls="--", lw=0.8)
    axes[0].set(xlabel="Minutes from cycle start", ylabel="Relative regret (%)", title="Cost shape")
    axes[0].set_ylim(bottom=-0.15)
    axes[0].legend(fontsize=7)
    axes[1].axvline(2.5, color="#9AA1A8", ls=":", lw=0.8)
    axes[1].set(ylabel="Within E/Q group normalized", title="Decision-point components")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend(fontsize=7)
    axes[2].axis("off")
    axes[2].set_title("Decision audit", loc="left")
    axes[2].text(0, 0.95, "\n\n".join(text), va="top", fontsize=7.5, linespacing=1.25)
    for axis in axes[:2]:
        axis.grid(color="#D7DCE1", lw=0.5)
    note = (
        "V2.6.4: local marginal zero crossing is the failure mode."
        if version == "v2.6.4"
        else "V2.6.5: earliest lower bound and earliest hard label are shown together."
        if version == "v2.6.5"
        else EARLY_PROBLEMS[version]
    )
    figure.suptitle(f"{version.upper()} early-cycle diagnosis — {note}", weight="bold")
    _save(figure, output)


def plot_sensor_diagnosis(
    tables: dict[str, pd.DataFrame], loader: DatasetLoader, cycle_name: str, output: Path
) -> None:
    """Plot unsmoothed operating evidence at all five version decisions."""
    frame = loader.load_cycle(cycle_name).sort_values("timestamp", kind="stable").copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    start = pd.Timestamp(loader.get_cycle_record(cycle_name)["boundaries"]["start_time"])
    minutes = (frame["timestamp"] - start).dt.total_seconds() / 60

    def observed(column: str) -> pd.Series:
        values = pd.to_numeric(frame[column], errors="coerce")
        imputed = frame.get(f"{column}__imputed")
        return values if imputed is None else values.mask(imputed.astype(bool))

    water_inputs = ("water_flow", "water_in_temperature", "water_out_temperature", "power_total")
    water_cop = water_side_heating_kw(frame).div(observed("power_total")).mask(
        pd.concat([observed(column) for column in water_inputs], axis=1).isna().any(axis=1)
    )
    figure, axes = plt.subplots(5, 1, figsize=(9.2, 10.4), sharex=True)
    panels = (
        (
            axes[0],
            (
                (observed("cop"), "Refrigerant COP", "#009E73"),
                (water_cop, "Water COP", "#0072B2"),
            ),
            "COP [-]",
        ),
        (
            axes[1],
            (
                (observed("heating_capacity"), "Unit heat", "#D55E00"),
                (observed("evaporator_capacity"), "Evaporator heat", "#B24C63"),
                (observed("power_total"), "Input power", "#4D4D4D"),
            ),
            "Capacity / power [kW]",
        ),
        (
            axes[2],
            (
                (
                    observed("water_out_temperature") - observed("water_in_temperature"),
                    "Water delta T",
                    "#0072B2",
                ),
            ),
            "Water delta T [degC]",
        ),
        (
            axes[3],
            (
                (observed("coil_temperature"), "T3", "#7B2CBF"),
                (observed("evaporating_temperature"), "Te", "#56B4E9"),
            ),
            "Evaporator temperature [degC]",
        ),
        (
            axes[4],
            (
                (observed("compressor_frequency"), "Compressor frequency", "#0072B2"),
                (observed("fan_speed"), "Fan speed", "#D99032"),
            ),
            "Control signal",
        ),
    )
    for axis, series, ylabel in panels:
        for values, label, color in series:
            axis.plot(minutes, values, color=color, lw=1.1, label=label)
        axis.set_ylabel(ylabel)
        axis.grid(color="#D7DCE1", lw=0.45)
        axis.legend(ncol=len(series), fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0))

    frost = frame["cycle_stage"].astype("string").eq("frost_development")
    frost_cop = pd.concat([observed("cop").where(frost), water_cop.where(frost)], axis=1).stack()
    lower, upper = float(frost_cop.min()), float(frost_cop.max())
    padding = max(0.1, 0.08 * (upper - lower))
    axes[0].set_ylim(lower - padding, upper + padding)

    pe_axis = axes[3].twinx()
    pe_axis.plot(minutes, observed("evaporating_pressure"), color="#B64A50", lw=1.0, label="Pe")
    pe_axis.set_ylabel("Pe [MPa]", color="#B64A50")
    pe_axis.tick_params(axis="y", colors="#B64A50")

    markers = []
    for version in VERSIONS:
        curve = _curve(tables, version, cycle_name)
        first = curve.iloc[0]
        target = pd.Timestamp(first["t_star"])
        x = (target - start).total_seconds() / 60
        markers.append(f"{version.upper()} {x:.1f}")
        for axis in axes:
            axis.axvline(x, color=COLORS[version], ls="--", lw=0.8)
    first = _curve(tables, VERSIONS[0], cycle_name).iloc[0]
    for column, label, color, style in (
        ("t_RB", "RB", "#238B57", ":"),
        ("actual_preparation_time", "Observed preparation", "#222222", "-."),
    ):
        target = pd.Timestamp(first[column])
        x = (target - start).total_seconds() / 60
        markers.append(f"{label} {x:.1f}")
        for axis in axes:
            axis.axvline(x, color=color, ls=style, lw=0.9)
        if column == "actual_preparation_time":
            observed_preparation = x
    axes[-1].set_xlabel("Minutes from cycle start")
    axes[-1].set_xlim(left=0, right=observed_preparation)
    axes[-1].text(
        0,
        -0.34,
        "   ".join(markers),
        transform=axes[-1].transAxes,
        fontsize=6.5,
        color="#4D5660",
    )
    cycle_id = int(cycle_name.rsplit("_", 1)[-1])
    figure.suptitle(
        f"Cycle {cycle_id}: early decisions checked against raw unsmoothed operation",
        x=0.1,
        ha="left",
        weight="bold",
    )
    figure.subplots_adjust(left=0.10, right=0.91, bottom=0.09, top=0.94, hspace=0.54)
    _save(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--cost-root", type=Path, default=ROOT / "output/成本函数")
    parser.add_argument("--output", type=Path, default=ROOT / "output/test/成本函数/成本函数迭代")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    loader = DatasetLoader(args.dataset)
    tables = _read_tables(args.cost_root, loader)
    cycles = _cycle_rows(tables)
    summary = build_version_summary(cycles)
    summary.to_csv(args.output / "version_summary.csv", index=False)
    diagnostics = build_early_diagnostics(tables, cycles)
    diagnostics.to_csv(args.output / "early_diagnostics.csv", index=False)

    plot_summary_table(summary, args.output / "figure_version_summary")
    plot_optimum_distribution(cycles, args.output / "figure_optimum_distribution")
    plot_scatter(cycles, args.output / "figure_v261_vs_v265_scatter")
    baseline = cycles.loc[cycles["version"].eq("v2.6.1")]
    picks = {
        "early": baseline.nsmallest(1, "t_star_minutes").iloc[0]["cycle_name"],
        "median": baseline.iloc[
            (baseline["t_star_minutes"] - baseline["t_star_minutes"].median()).abs().argmin()
        ]["cycle_name"],
        "late": baseline.nlargest(1, "t_star_minutes").iloc[0]["cycle_name"],
    }
    for kind, cycle_name in picks.items():
        plot_cost_shape(tables, str(cycle_name), kind, args.output / f"figure_cost_shape_{kind}")
    plot_flatness(cycles, args.output / "figure_flatness")
    plot_v264_failure(tables, "frost_cycle_000085", args.output / "figure_v264_failure")
    plot_label_status(cycles, args.output / "figure_v265_label_status")
    early = _early_cycles(cycles)
    for version, cycle_names in early.items():
        plot_early_diagnosis(
            tables, version, cycle_names, args.output / f"figure_early_diagnosis_{version}"
        )
    for cycle_name in sorted({cycle for names in early.values() for cycle in names}):
        cycle_id = int(cycle_name.rsplit("_", 1)[-1])
        plot_sensor_diagnosis(
            tables,
            loader,
            cycle_name,
            args.output / f"figure_sensor_diagnosis_cycle_{cycle_id:03d}",
        )


if __name__ == "__main__":
    main()
