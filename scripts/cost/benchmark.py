#!/usr/bin/env python3
"""Build the parallel C/H/O decision benchmark and its evidence figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frost_analysis.cost.benchmark import (
    FINAL_METRICS,
    absolute_rate_metric_tables,
    benchmark_table,
    bootstrap_absolute_rate_trajectories,
    bootstrap_failure_anatomy,
    bootstrap_fixed_support_stability,
    bootstrap_ho_cofailure,
    bootstrap_stability,
    bootstrap_validity_taxonomy,
    ch_high_value_overlap,
    ch_tradeoff_diagnostic,
    cross_objective_regret,
    cycle_trigger_validation,
    experiment_leverage,
    final_metric_tables,
    ho_paired_decisions,
    local_ratio_attribution,
    matched_decision_regret,
    outdoor_event_model_ablation,
    pareto_nondominated,
    regret_coverage,
    regret_distribution,
    same_cycle_regret,
    stability_to_basin_ratio,
)

COLORS = {
    "cop_cyc_evt": "#484878",
    "eta_h_cyc": "#B64A50",
    "eta_e_cyc": "#2A788E",
}
LABELS = {
    "cop_cyc_evt": r"$COP_{cyc,evt}$",
    "eta_h_cyc": r"$\eta_H$",
    "eta_e_cyc": r"$\eta_{out}$",
    "h_abs_rate": r"$H_{abs}$",
    "o_abs_rate": r"$O_{abs}$",
}
COP_VERSIONS = (
    "v1",
    "v2",
    "v2.1",
    "v2.2",
    "v2.3",
    "v2.4",
    "v2.5",
    "v2.6",
    "v2.6.1",
    "v2.6.2",
    "v2.6.3",
    "v2.6.4",
    "v2.6.5",
    "v2.6.6",
    "v2.6.7",
    "v2.6.8",
)
COP_HISTORY = {
    "v1": (
        "stable heating → candidate",
        "unit",
        "ED model + fixed ER; QD=QR=0",
        "permanent V0 baseline",
    ),
    "v2": (
        "stable heating → candidate",
        "water",
        "predicted ED/QD + setpoint ER/QR",
        "event-heat ablation",
    ),
    "v2.1": (
        "stable heating → candidate",
        "unit",
        "adds Qprep; unit-side QR",
        "heat-basis ablation",
    ),
    "v2.2": (
        "stable heating → candidate",
        "water",
        "Qprep/QD + variable recovery",
        "water-side full event",
    ),
    "v2.3": (
        "stable heating → candidate",
        "water",
        "fixed 9-min recovery",
        "recovery-duration ablation",
    ),
    "v2.4": (
        "fixed 9-min → fixed 9-min",
        "water",
        "closed fixed-boundary cycle",
        "boundary ablation",
    ),
    "v2.5": (
        "cycle start → candidate",
        "water",
        "recovery folded into observed prefix",
        "current-cycle water baseline",
    ),
    "v2.6": (
        "cycle start → candidate",
        "unit",
        "recovery folded into observed prefix",
        "current-cycle unit baseline",
    ),
    "v2.6.1": (
        "cycle start → candidate",
        "unit",
        "exact V2.6 calculation",
        "identity/control check",
    ),
    "v2.6.2": (
        "stable → stable",
        "unit",
        "projected post-defrost recovery",
        "closed-cycle boundary",
    ),
    "v2.6.3": (
        "stable → stable",
        "unit",
        "baseline-normalized excess electricity",
        "degradation form",
    ),
    "v2.6.4": (
        "stable → stable",
        "unit",
        "5-min marginal Dinkelbach balance",
        "marginal diagnostic",
    ),
    "v2.6.5": (
        "stable → stable",
        "unit",
        "average curve + supported marginal basin",
        "decision-basin diagnostic",
    ),
    "v2.6.6": (
        "stable → candidate",
        "unit",
        "LOEO loss + event-ticket components",
        "identification diagnostic",
    ),
    "v2.6.7": (
        "stable → candidate + event",
        "unit",
        "LOEO independent ET/QT prediction",
        "ticket-model candidate",
    ),
    "v2.6.8": (
        "fixed-9 stable → candidate + event",
        "water",
        "LOEO joint event-outcome framework",
        "current COP representative",
    ),
}
CROSS_FITTED_COP = {"v2.6.6", "v2.6.7", "v2.6.8"}
PUBLICATION_COLUMNS = (
    "cycle_name",
    "experiment_id",
    "candidate_time",
    "cycle_start",
    "stable_start_fixed9",
    "actual_preparation_time",
    "t_RB",
    "rb_status",
    "algorithm",
    "metric_id",
    "objective_label",
    "objective_unit",
    "objective_value",
    "display_only_objective",
    "optimization_direction",
    "optimization_eligible",
    "supported",
    "model_supported",
    "physical_valid",
    "continuous_support",
    "relative_optimality_gap",
    "t_star",
    "cycle_status",
    "decision_status",
    "near_optimal_1pct",
    "near_optimal_2pct",
    "near_optimal_5pct",
    "basin_1pct_start",
    "basin_1pct_end",
    "basin_1pct_width_minutes",
    "basin_2pct_start",
    "basin_2pct_end",
    "basin_2pct_width_minutes",
    "basin_5pct_start",
    "basin_5pct_end",
    "basin_5pct_width_minutes",
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def _save(figure: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _benchmark_figure(summary: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(8.1, 5.8))
    positions = np.arange(len(FINAL_METRICS))
    for metric, position in zip(FINAL_METRICS, positions, strict=True):
        values = summary.loc[summary.metric_id.eq(metric)]
        axes[0, 0].boxplot(
            values["t_star_cycle_minutes"].dropna(),
            positions=[position],
            widths=0.55,
            patch_artist=True,
            boxprops={"facecolor": COLORS[metric], "alpha": 0.65},
            medianprops={"color": "white"},
        )
        axes[1, 1].scatter(
            values["frosting_progress"],
            values["heating_attenuation_fraction"],
            s=13,
            alpha=0.55,
            color=COLORS[metric],
            label=LABELS[metric],
        )
    axes[0, 0].set(
        xticks=positions,
        xticklabels=[LABELS[m] for m in FINAL_METRICS],
        ylabel="Selected time from cycle start [min]",
        title="Independent decisions occupy different times",
    )
    width = (
        summary.groupby("metric_id")[["W1_minutes", "W2_minutes", "W5_minutes"]]
        .median()
        .reindex(FINAL_METRICS)
    )
    width.plot.bar(
        ax=axes[0, 1],
        color=["#B4C0E4", "#7884B4", "#484878"],
        width=0.78,
    )
    axes[0, 1].set(
        xticklabels=[LABELS[m] for m in FINAL_METRICS],
        ylabel="Median connected-basin width [min]",
        title="1/2/5% basins quantify decision resolution",
    )
    axes[0, 1].tick_params(axis="x", rotation=0)
    axes[0, 1].legend(["1%", "2%", "5%"], ncol=3, fontsize=7)
    support = (
        summary.groupby("metric_id")
        .agg(
            identified=("t_star", lambda values: values.notna().mean()),
            interior=("extreme_location", lambda values: values.eq("interior").mean()),
            support=("support_fraction", "median"),
        )
        .reindex(FINAL_METRICS)
    )
    support.plot.bar(ax=axes[1, 0], color=["#D8D8D8", "#7884B4", "#2A788E"])
    axes[1, 0].set(
        ylim=(0, 1.05),
        xticklabels=[LABELS[m] for m in FINAL_METRICS],
        ylabel="Fraction",
        title="Coverage, interior extrema and candidate support",
    )
    axes[1, 0].tick_params(axis="x", rotation=0)
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].axvline(1, color="#767676", lw=0.8, ls="--")
    axes[1, 1].axhline(0, color="#767676", lw=0.8, ls="--")
    axes[1, 1].set(
        xlabel="Selected frosting progress (stable start = 0, actual defrost = 1)",
        ylabel="Selected water-heat attenuation",
        title="The selected physical state is measured, not assumed",
    )
    axes[1, 1].legend(fontsize=7)
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontweight="bold")
    figure.suptitle(
        "C, H and O remain parallel objectives on one candidate grid",
        x=0.08,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def _regret_figure(regret: pd.DataFrame) -> plt.Figure:
    decisions = ("point", "latest_W1", "latest_W2", "latest_W5")
    figure, axes = plt.subplots(1, len(decisions), figsize=(11.3, 2.8), sharey=True)
    for axis, decision in zip(axes, decisions, strict=True):
        values = matched_decision_regret(regret, decision)
        matrix = (
            values.groupby(["selector_metric", "target_metric"])["cross_objective_regret"]
            .median()
            .unstack()
            .reindex(index=FINAL_METRICS, columns=FINAL_METRICS)
        )
        image = axis.imshow(100 * matrix, cmap="magma_r", vmin=0, vmax=10)
        for row in range(3):
            for column in range(3):
                value = 100 * matrix.iloc[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.1f}" if np.isfinite(value) else "—",
                    ha="center",
                    va="center",
                    color="white" if value > 5 else "black",
                    fontsize=7,
                )
        axis.set(
            xticks=range(3),
            xticklabels=[LABELS[m] for m in FINAL_METRICS],
            yticks=range(3),
            yticklabels=[LABELS[m] for m in FINAL_METRICS],
            xlabel="Target objective",
            title=(
                decision.replace("latest_", "Latest in ")
                if decision != "point"
                else "Point optimum"
            )
            + f"\nmatched cycles n={values['cycle_name'].nunique()}",
        )
    axes[0].set_ylabel("Selector objective")
    colorbar = figure.colorbar(
        image,
        ax=axes,
        label="Median cross-objective regret [%]",
        shrink=0.8,
        pad=0.03,
    )
    colorbar.ax.set_position([0.925, 0.20, 0.012, 0.56])
    figure.suptitle(
        "Each objective is judged by the consequences of its own decision",
        x=0.08,
        ha="left",
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.08, right=0.89, bottom=0.20, top=0.76, wspace=0.28)
    return figure


def _bootstrap_figure(stability: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))
    fields = (
        ("IQR_tau_minutes", "Optimum-time IQR [min]"),
        ("MAD_tau_minutes", "Optimum-time MAD [min]"),
        ("p90_self_regret", "P90 self-regret"),
    )
    for axis, (field, label) in zip(axes, fields, strict=True):
        values = [
            stability.loc[stability.metric_id.eq(metric), field].dropna()
            for metric in FINAL_METRICS
        ]
        boxes = axis.boxplot(values, patch_artist=True, showfliers=False)
        for box, metric in zip(boxes["boxes"], FINAL_METRICS, strict=True):
            box.set_facecolor(COLORS[metric])
            box.set_alpha(0.65)
        axis.set(xticks=range(1, 4), xticklabels=[LABELS[m] for m in FINAL_METRICS], ylabel=label)
    figure.suptitle(
        "Experiment-level bootstrap separates location uncertainty from decision regret",
        x=0.08,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return figure


def _physical_state_figure(summary: pd.DataFrame) -> plt.Figure:
    fields = (
        ("frosting_progress", "Frosting progress"),
        ("heating_attenuation_fraction", "Water-heat attenuation"),
        ("instant_water_cop", "Instantaneous water-side COP"),
        ("instant_evaporator_capacity_kw", "Outdoor-side heat transfer [kW]"),
        ("evaporating_pressure", "Evaporating pressure [MPa]"),
        ("coil_temperature", "Coil temperature [°C]"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(9.4, 5.2))
    for axis, (field, label) in zip(axes.flat, fields, strict=True):
        values = [
            pd.to_numeric(summary.loc[summary.metric_id.eq(metric), field], errors="coerce")
            .dropna()
            .to_numpy()
            for metric in FINAL_METRICS
        ]
        boxes = axis.boxplot(values, patch_artist=True, showfliers=False)
        for box, metric in zip(boxes["boxes"], FINAL_METRICS, strict=True):
            box.set_facecolor(COLORS[metric])
            box.set_alpha(0.65)
        axis.set(
            xticks=range(1, 4),
            xticklabels=[LABELS[metric] for metric in FINAL_METRICS],
            ylabel=label,
        )
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle(
        "Each objective independently selects a measurable physical state",
        x=0.07,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _cop_history_screen(cost_root: Path, metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    indexed = {
        (metric, cycle): curve.assign(
            candidate_time=pd.to_datetime(curve["candidate_time"], errors="coerce")
        )
        .dropna(subset=["candidate_time"])
        .sort_values("candidate_time", kind="stable")
        for metric, table in metrics.items()
        for cycle, curve in table.groupby("cycle_name", sort=False)
    }
    rows = [
        _cop_version_row(cost_root, version, indexed)
        for version in COP_VERSIONS
        if (cost_root / f"cost_function_{version}.csv").exists()
    ]
    result = pd.DataFrame(rows)
    definitions = pd.DataFrame.from_dict(
        COP_HISTORY,
        orient="index",
        columns=["cycle_boundary", "heating_heat_basis", "event_accounting", "study_role"],
    )
    result = result.join(definitions, on="version")
    result["passes_evidence_gate"] = (
        result["cross_fitted"]
        & result["identified_fraction"].ge(0.8)
        & result["median_support_fraction"].ge(0.6)
        & result["bootstrap_valid_fraction"].ge(0.8)
    )
    regret_columns = [f"median_regret_{metric}" for metric in FINAL_METRICS]
    result["worst_target_median_regret"] = result[regret_columns].max(axis=1)
    result["pareto"] = False
    candidates = result.loc[result["passes_evidence_gate"]]
    for index, row in candidates.iterrows():
        others = candidates.drop(index)
        minimize = ["median_W1_minutes", *regret_columns]
        no_worse = others[minimize].le(row[minimize]).all(axis=1) & others[
            "bootstrap_basin_return"
        ].ge(row["bootstrap_basin_return"])
        strictly_better = others[minimize].lt(row[minimize]).any(axis=1) | others[
            "bootstrap_basin_return"
        ].gt(row["bootstrap_basin_return"])
        dominated = (no_worse & strictly_better).any()
        result.loc[index, "pareto"] = not dominated
    pareto = result.loc[result["pareto"]]
    result["provisional_cop_choice"] = False
    if not pareto.empty:
        winner = pareto["worst_target_median_regret"].idxmin()
        result.loc[winner, "provisional_cop_choice"] = True
    return result


def _cop_version_row(
    cost_root: Path,
    version: str,
    indexed: dict[tuple[str, str], pd.DataFrame],
) -> dict[str, object]:
    table = pd.read_csv(cost_root / f"cost_function_{version}.csv", low_memory=False)
    cycle_rows: list[dict[str, float]] = []
    for cycle, curve in table.groupby("cycle_name", sort=False):
        selected_time = pd.to_datetime(curve["t_star"].iloc[0], errors="coerce")
        if pd.isna(selected_time):
            continue
        result: dict[str, float] = {}
        for metric in FINAL_METRICS:
            target = indexed.get((metric, cycle))
            if target is None:
                continue
            position = np.abs(
                (target["candidate_time"] - selected_time).dt.total_seconds().to_numpy()
            ).argmin()
            selected = target.iloc[position]
            if abs((selected["candidate_time"] - selected_time).total_seconds()) > 31 or not bool(
                selected["optimization_eligible"]
            ):
                continue
            result[f"regret_{metric}"] = float(selected["relative_optimality_gap"])
            stable = pd.to_datetime(selected.get("stable_start_fixed9"), errors="coerce")
            actual = pd.to_datetime(selected.get("actual_preparation_time"), errors="coerce")
            result["frosting_progress"] = (
                (selected["candidate_time"] - stable) / (actual - stable)
                if pd.notna(stable) and pd.notna(actual) and actual > stable
                else np.nan
            )
            result["heating_attenuation_fraction"] = float(selected["heating_attenuation_fraction"])
        cycle_rows.append(result)
    per_cycle = pd.DataFrame(cycle_rows)
    grouped = table.groupby("cycle_name", sort=False)
    bootstrap_valid, bootstrap_return = _cop_bootstrap_summary(cost_root, version)
    row: dict[str, object] = {
        "version": version,
        "display_name": "V0 / original (stored as V1)" if version == "v1" else version.upper(),
        "preserved_baseline": version == "v1",
        "cross_fitted": version in CROSS_FITTED_COP,
        "identified_fraction": float(grouped["t_star"].first().notna().mean()),
        "median_support_fraction": float(grouped["optimization_eligible"].mean().median()),
        "median_W1_minutes": float(
            grouped.apply(lambda curve: _history_width(curve, 1), include_groups=False).median()
        ),
        "bootstrap_valid_fraction": bootstrap_valid,
        "bootstrap_basin_return": bootstrap_return,
        "rgb_learnability_tested": False,
    }
    for metric in FINAL_METRICS:
        row[f"median_regret_{metric}"] = pd.to_numeric(
            per_cycle.get(f"regret_{metric}"), errors="coerce"
        ).median()
    for column in ("frosting_progress", "heating_attenuation_fraction"):
        row[column] = pd.to_numeric(per_cycle.get(column), errors="coerce").median()
    return row


def _history_width(curve: pd.DataFrame, percent: int) -> float:
    column = f"basin_{percent}pct_width_minutes"
    if column in curve and pd.notna(curve[column].iloc[0]):
        return float(curve[column].iloc[0])
    near = curve.get(f"near_optimal_{percent}pct", pd.Series(False, index=curve.index)).fillna(
        False
    )
    times = pd.to_datetime(curve.loc[near, "candidate_time"], errors="coerce")
    return (times.max() - times.min()).total_seconds() / 60 if times.notna().any() else np.nan


def _cop_bootstrap_summary(cost_root: Path, version: str) -> tuple[float, float]:
    if version == "v2.6.7":
        table = pd.read_csv(cost_root / "cost_function_v2.6.7_bootstrap_audit.csv")
        return (
            float(table["two_candidate_repeat_fraction"].median()),
            float(table["argmin_in_original_5pct_basin_fraction"].median()),
        )
    if version == "v2.6.8":
        table = pd.read_csv(cost_root / "cost_function_v2.6.8_bootstrap.csv")
        return (
            float(table["valid_minimum_fraction"].median()),
            float(table["argmin_in_original_5pct_basin_fraction"].median()),
        )
    return np.nan, np.nan


def _cop_history_evidence_figure(screen: pd.DataFrame) -> plt.Figure:
    """Show every historical version and distinguish missing from failed evidence."""
    columns = [
        "curve_available",
        "cross_fitted",
        "bootstrap_available",
        "rgb_learnability_tested",
        "passes_evidence_gate",
        "pareto",
        "provisional_cop_choice",
    ]
    values = screen.assign(
        curve_available=True,
        bootstrap_available=screen["bootstrap_valid_fraction"].notna(),
    )
    matrix = values[columns].astype(float).to_numpy()
    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    axis.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    for row in range(len(values)):
        for column in range(len(columns)):
            axis.text(
                column,
                row,
                "yes" if matrix[row, column] else "—",
                ha="center",
                va="center",
                color="white" if matrix[row, column] else "#555555",
                fontsize=7,
            )
    axis.set(
        yticks=np.arange(len(values)),
        yticklabels=values["display_name"],
        xticks=np.arange(len(columns)),
        xticklabels=[
            "Curve",
            "Cross-fit",
            "Bootstrap",
            "RGB",
            "Gate",
            "Pareto",
            "Selected",
        ],
        title="All 16 historical COP versions are present; evidence depth is unequal",
    )
    axis.tick_params(axis="x", rotation=28)
    axis.text(
        0,
        -0.09,
        "— means not established under this protocol, not zero performance. "
        "V1-r is an oracle audit, not a deployable version.",
        transform=axis.transAxes,
        fontsize=7,
    )
    figure.tight_layout()
    return figure


def _cop_history_consequence_figure(screen: pd.DataFrame) -> plt.Figure:
    """Put names next to all historical resolution and consequence values."""
    figure, axes = plt.subplots(1, 4, figsize=(12.6, 7.0), sharey=True)
    y = np.arange(len(screen))
    fields = ["median_W1_minutes", *[f"median_regret_{metric}" for metric in FINAL_METRICS]]
    titles = ["1% basin width", *[f"Regret under {LABELS[m]}" for m in FINAL_METRICS]]
    colors = ["#888888", *[COLORS[m] for m in FINAL_METRICS]]
    for axis, field, title, color in zip(axes, fields, titles, colors, strict=True):
        values = pd.to_numeric(screen[field], errors="coerce")
        plotted = values if field == "median_W1_minutes" else 100 * values
        axis.barh(y, plotted, color=color, alpha=0.78)
        axis.set(title=title, xlabel="min" if field == "median_W1_minutes" else "%")
        axis.grid(axis="x", alpha=0.18)
    axes[0].set(yticks=y, yticklabels=screen["display_name"])
    axes[0].invert_yaxis()
    figure.suptitle(
        "Historical COP family: resolution and three consequences remain separate",
        x=0.07,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def _cop_history_finalists_figure(screen: pd.DataFrame) -> plt.Figure:
    """Compare the historically important versions without hiding evidence gaps."""
    keep = {"v1", "v2.4", "v2.5", "v2.6.6", "v2.6.7", "v2.6.8"}
    values = screen.loc[screen["version"].isin(keep)].copy()
    x = np.arange(len(values))
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.5))
    axes[0].bar(x, values["median_W1_minutes"], color="#777777")
    axes[0].set(ylabel="Minutes", title="Decision resolution (W1)")
    width = 0.24
    for offset, metric in zip((-width, 0, width), FINAL_METRICS, strict=True):
        axes[1].bar(
            x + offset,
            100 * values[f"median_regret_{metric}"],
            width,
            color=COLORS[metric],
            label=LABELS[metric],
        )
    axes[1].set(ylabel="Median regret [%]", title="Independent consequences")
    axes[1].legend(fontsize=6.5)
    bootstrap = 100 * pd.to_numeric(values["bootstrap_basin_return"], errors="coerce")
    axes[2].bar(x, bootstrap.fillna(0), color="#3B75AF")
    for position, value in zip(x, bootstrap, strict=True):
        if pd.isna(value):
            axes[2].text(position, 2, "not run", ha="center", va="bottom", rotation=90)
    axes[2].set(ylim=(0, 105), ylabel="Basin return [%]", title="Model-refit bootstrap evidence")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(values["display_name"], rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle(
        "V2.6.8 is the current representative; older versions are ablations, not erased",
        x=0.06,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure


def _cop_history_definition_figure(screen: pd.DataFrame) -> plt.Figure:
    """Make the historical ablation path readable without opening the code."""
    figure, axis = plt.subplots(figsize=(12.0, 7.0))
    axis.axis("off")
    table = axis.table(
        cellText=screen[
            [
                "display_name",
                "cycle_boundary",
                "heating_heat_basis",
                "event_accounting",
                "study_role",
            ]
        ].to_numpy(),
        colLabels=["Version", "Candidate cycle boundary", "QH basis", "Event accounting", "Role"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.12, 0.23, 0.08, 0.32, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.45)
    for column in range(5):
        table[(0, column)].set_facecolor("#DCE6F1")
        table[(0, column)].set_text_props(fontweight="bold")
    for row in range(1, len(screen) + 1):
        if row % 2 == 0:
            for column in range(5):
                table[(row, column)].set_facecolor("#F5F5F5")
    axis.set_title(
        "Historical COP family: each version changes a boundary, heat basis, or event model",
        loc="left",
        fontweight="bold",
        pad=12,
    )
    figure.tight_layout()
    return figure


def _regret_coverage_figure(coverage: pd.DataFrame) -> plt.Figure:
    point = coverage.loc[coverage["decision_type"].eq("point")]
    matrix = point.pivot(
        index="selector_metric", columns="target_metric", values="coverage_fraction"
    )
    matrix = matrix.reindex(index=FINAL_METRICS, columns=FINAL_METRICS)
    figure, axis = plt.subplots(figsize=(4.8, 4.0))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for row in range(3):
        for column in range(3):
            value = matrix.iloc[row, column]
            axis.text(column, row, f"{100 * value:.0f}%", ha="center", va="center")
    axis.set(
        xticks=range(3),
        xticklabels=[LABELS[m] for m in FINAL_METRICS],
        yticks=range(3),
        yticklabels=[LABELS[m] for m in FINAL_METRICS],
        xlabel="Target objective",
        ylabel="Selector objective",
        title="Coverage reveals asymmetric missingness",
    )
    figure.colorbar(image, ax=axis, label="Available decision fraction")
    figure.tight_layout()
    return figure


def _bootstrap_taxonomy_figure(taxonomy: pd.DataFrame) -> plt.Figure:
    order = [
        "valid_interior",
        "valid_endpoint",
        "support_or_measurement_limited",
        "formula_unavailable",
    ]
    fractions = (
        (taxonomy.groupby(["metric_id", "status"]).size() / taxonomy.groupby("metric_id").size())
        .unstack(fill_value=0)
        .reindex(index=FINAL_METRICS, columns=order, fill_value=0)
    )
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    bottom = np.zeros(len(fractions))
    palette = ["#3B75AF", "#8CB6D9", "#D8904F", "#B64A50"]
    for status, color in zip(order, palette, strict=True):
        axis.bar(
            range(len(fractions)),
            fractions[status],
            bottom=bottom,
            color=color,
            label=status.replace("_", " "),
        )
        bottom += fractions[status].to_numpy()
    axis.set(
        xticks=range(3),
        xticklabels=[LABELS[m] for m in FINAL_METRICS],
        ylim=(0, 1),
        ylabel="Bootstrap replicate-cycle fraction",
        title="Invalid bootstrap curves are support-limited, not silently treated as zero",
    )
    axis.legend(fontsize=6.5, ncol=2)
    axis.grid(axis="y", alpha=0.18)
    figure.tight_layout()
    return figure


def _estimability_anatomy_figure(
    anatomy: pd.DataFrame,
    cofailure: pd.DataFrame,
    full: pd.DataFrame,
    conditional: pd.DataFrame,
) -> plt.Figure:
    metrics = ("eta_h_cyc", "eta_e_cyc")
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.1))
    grouped = anatomy.assign(
        display_reason=anatomy["failure_reason"].map(
            lambda reason: (
                "component support absent"
                if "_support" in reason and not reason.startswith("joint_")
                else reason.replace("_", " ")
            )
        )
    )
    reasons = (
        grouped.loc[~grouped["valid"]]
        .groupby(["metric_id", "display_reason"])
        .size()
        .unstack(fill_value=0)
    )
    reasons = reasons.div(anatomy.groupby("metric_id").size(), axis=0).reindex(metrics)
    bottom = np.zeros(len(metrics))
    for reason in reasons.columns:
        axes[0, 0].bar(range(2), reasons[reason], bottom=bottom, label=reason)
        bottom += reasons[reason].to_numpy()
    axes[0, 0].set(
        xticks=range(2),
        xticklabels=[LABELS[m] for m in metrics],
        ylim=(0, 1),
        ylabel="Replicate-cycle fraction",
        title="Which gate makes a curve inestimable?",
    )
    axes[0, 0].legend(fontsize=5.5, ncol=2)

    components = ("Q_T", "Qw0", "D_T", "Pcomp0", "E_comp_T")
    requirements = {
        "eta_h_cyc": {"Q_T", "Qw0", "D_T"},
        "eta_e_cyc": set(components),
    }
    missing = np.array(
        [
            [
                (
                    1
                    - pd.to_numeric(
                        anatomy.loc[
                            anatomy.metric_id.eq(metric), f"support_{name}_any"
                        ],
                        errors="coerce",
                    ).mean()
                    if name in requirements[metric]
                    else np.nan
                )
                for name in components
            ]
            for metric in metrics
        ]
    )
    image = axes[0, 1].imshow(missing, cmap="Reds", vmin=0, vmax=max(0.01, np.nanmax(missing)))
    for row in range(2):
        for column in range(len(components)):
            value = missing[row, column]
            axes[0, 1].text(
                column,
                row,
                f"{100 * value:.0f}%" if np.isfinite(value) else "N/A",
                ha="center",
                va="center",
            )
    axes[0, 1].set(
        xticks=range(len(components)),
        xticklabels=components,
        yticks=range(2),
        yticklabels=[LABELS[m] for m in metrics],
        title="Component support absent for the whole curve",
    )
    figure.colorbar(image, ax=axes[0, 1], label="Absent fraction", shrink=0.75)

    modes = {"Full pipeline": full, "Fixed point support": conditional}
    x = np.arange(2)
    for offset, (name, table) in zip((-0.18, 0.18), modes.items(), strict=True):
        values = [
            table.loc[table.metric_id.eq(metric), "valid_fraction"].median()
            for metric in metrics
        ]
        axes[1, 0].bar(x + offset, values, 0.34, label=name)
    axes[1, 0].axhline(0.8, color="#333333", ls="--", lw=0.8)
    axes[1, 0].set(
        xticks=x,
        xticklabels=[LABELS[m] for m in metrics],
        ylim=(0, 1.05),
        ylabel="Median valid fraction",
        title="Fixed support isolates estimator uncertainty",
    )
    axes[1, 0].legend(fontsize=6.5)

    axes[1, 1].barh(
        range(len(cofailure)),
        cofailure["value"],
        color=(COLORS["eta_h_cyc"], COLORS["eta_e_cyc"], "#777777"),
    )
    axes[1, 1].set(
        yticks=range(len(cofailure)),
        yticklabels=(
            "P(H invalid | O invalid)",
            "P(O invalid | H invalid)",
            "P(H and O invalid)",
        ),
        xlim=(0, 1),
        xlabel="Probability",
        title="Do H and O fail in the same resamples?",
    )
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontweight="bold")
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle(
        "H/O estimability is decomposed before changing either objective",
        x=0.06,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _leverage_figure(leverage: pd.DataFrame) -> plt.Figure:
    order = (
        leverage.groupby("source_experiment_id")["leverage"]
        .apply(lambda values: values.abs().max())
        .sort_values()
        .index
    )
    figure, axis = plt.subplots(figsize=(7.8, max(3.8, 0.25 * len(order))))
    for metric, offset in zip(("eta_h_cyc", "eta_e_cyc"), (-0.10, 0.10), strict=True):
        values = leverage.loc[leverage.metric_id.eq(metric)].set_index("source_experiment_id")
        axis.scatter(
            values.reindex(order)["leverage"],
            np.arange(len(order)) + offset,
            s=24,
            color=COLORS[metric],
            label=LABELS[metric],
        )
    axis.axvline(0, color="#555555", lw=0.8)
    axis.set(
        yticks=range(len(order)),
        yticklabels=order,
        xlabel=r"$P(invalid\mid absent)-P(invalid\mid present)$",
        ylabel="Source experiment",
        title="Positive leverage identifies experiments that protect H/O estimability",
    )
    axis.legend()
    axis.grid(axis="x", alpha=0.18)
    figure.tight_layout()
    return figure


def _ho_family_figure(paired: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(8.8, 5.8))
    axes[0, 0].hist(paired["delta_t_O_minus_H_minutes"].dropna(), bins=16, color="#7E6AA2")
    axes[0, 0].axvline(0, color="#333333", ls="--", lw=0.8)
    axes[0, 0].set(xlabel=r"$\tau_O-\tau_H$ [min]", ylabel="Cycles", title="Timing agreement")
    axes[0, 1].scatter(
        100 * paired["C_regret_at_H"],
        100 * paired["C_regret_at_O"],
        s=18,
        alpha=0.65,
        color="#555555",
    )
    limit = max(0.1, 100 * paired[["C_regret_at_H", "C_regret_at_O"]].max().max())
    axes[0, 1].plot([0, limit], [0, limit], color="#333333", ls="--", lw=0.8)
    axes[0, 1].set(
        xlabel="C regret at H decision [%]",
        ylabel="C regret at O decision [%]",
        title="Paired energy consequence",
    )
    overlaps = []
    labels = []
    for percent in (1, 2, 5):
        for direction in ("H_in_O", "O_in_H"):
            overlaps.append(paired[f"{direction}_W{percent}"].mean())
            labels.append(f"{direction.replace('_in_', ' in ')} W{percent}")
    axes[1, 0].bar(range(len(overlaps)), overlaps, color=["#B64A50", "#2A788E"] * 3)
    axes[1, 0].set(
        xticks=range(len(labels)),
        xticklabels=labels,
        ylim=(0, 1),
        ylabel="Cycle fraction",
        title="Mutual near-optimal membership",
    )
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 1].boxplot(
        [100 * paired["H_regret_at_O"], 100 * paired["O_regret_at_H"]],
        tick_labels=[r"$r_H(\tau_O)$", r"$r_O(\tau_H)$"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#A9CCD3"},
    )
    axes[1, 1].set(ylabel="Cross regret [%]", title="Are H and O decision-equivalent?")
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontweight="bold")
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle(
        f"H and O are compared cycle-by-cycle on identical evidence (n={len(paired)})",
        x=0.06,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _same_cycle_regret_figure(common: pd.DataFrame) -> plt.Figure:
    decisions = ("point", "latest_W1", "latest_W2", "latest_W5")
    figure, axes = plt.subplots(2, 3, figsize=(10.2, 6.2))
    for axis, decision in zip(axes.flat[:4], decisions, strict=True):
        values = common.loc[common.decision_type.eq(decision)]
        matrix = (
            values.groupby(["selector_metric", "target_metric"])["cross_objective_regret"]
            .median()
            .unstack()
            .reindex(index=FINAL_METRICS, columns=FINAL_METRICS)
        )
        axis.imshow(100 * matrix, cmap="magma_r", vmin=0, vmax=10)
        for row in range(3):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    f"{100 * matrix.iloc[row, column]:.1f}",
                    ha="center",
                    va="center",
                )
        axis.set(
            xticks=range(3),
            xticklabels=[LABELS[m] for m in FINAL_METRICS],
            yticks=range(3),
            yticklabels=[LABELS[m] for m in FINAL_METRICS],
            title=decision.replace("latest_", "Latest in "),
        )
    tails = pd.concat(
        [regret_distribution(common, decision) for decision in decisions], ignore_index=True
    )
    c_tail = tails.loc[tails.target_metric.eq("cop_cyc_evt")]
    x = np.arange(len(decisions))
    for metric in FINAL_METRICS:
        values = c_tail.loc[c_tail.selector_metric.eq(metric)].set_index("decision_type")
        axes[1, 1].plot(
            x,
            100 * values.reindex(decisions)["p90_regret"],
            marker="o",
            label=LABELS[metric],
            color=COLORS[metric],
        )
        axes[1, 2].plot(
            x,
            values.reindex(decisions)["P_regret_lt_1pct"],
            marker="o",
            label=LABELS[metric],
            color=COLORS[metric],
        )
    for axis, ylabel, title in (
        (axes[1, 1], "P90 C regret [%]", "Tail consequence"),
        (axes[1, 2], r"$P(r_C<1\%)$", "Engineering tolerance"),
    ):
        axis.set(xticks=x, xticklabels=["Point", "W1", "W2", "W5"], ylabel=ylabel, title=title)
        axis.legend(fontsize=6.5)
        axis.grid(alpha=0.18)
    for label, axis in zip("abcdef", axes.flat, strict=True):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontweight="bold")
    figure.suptitle(
        "Selector semantics are compared on one fixed cycle subset "
        f"(n={common.cycle_name.nunique()})",
        x=0.05,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _rho_figure(rho: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(9.3, 3.2))
    values = [
        rho.loc[rho.metric_id.eq(metric), "rho_IQR_over_W5"].dropna()
        for metric in FINAL_METRICS
    ]
    boxes = axes[0].boxplot(values, patch_artist=True, showfliers=False)
    for box, metric in zip(boxes["boxes"], FINAL_METRICS, strict=True):
        box.set_facecolor(COLORS[metric])
        box.set_alpha(0.65)
    axes[0].axhline(1, color="#333333", ls="--", lw=0.8)
    axes[0].set(
        xticks=range(1, 4),
        xticklabels=[LABELS[m] for m in FINAL_METRICS],
        ylabel=r"$\rho=IQR(\tau^*)/W_5$",
        title="Location uncertainty / value tolerance",
    )
    fractions = [
        rho.loc[rho.metric_id.eq(metric), "uncertainty_within_W5"].mean()
        for metric in FINAL_METRICS
    ]
    axes[1].bar(range(3), fractions, color=[COLORS[m] for m in FINAL_METRICS])
    axes[1].set(
        xticks=range(3),
        xticklabels=[LABELS[m] for m in FINAL_METRICS],
        ylim=(0, 1),
        ylabel=r"Fraction with $\rho<1$",
        title="Bootstrap movement remains inside W5",
    )
    for metric in FINAL_METRICS:
        selected = rho.loc[rho.metric_id.eq(metric)]
        axes[2].scatter(
            selected["W5_minutes"],
            selected["IQR_tau_minutes"],
            s=16,
            alpha=0.55,
            color=COLORS[metric],
            label=LABELS[metric],
        )
    limit = max(rho["W5_minutes"].max(), rho["IQR_tau_minutes"].max())
    axes[2].plot([0, limit], [0, limit], color="#333333", ls="--", lw=0.8)
    axes[2].set(
        xlabel="W5 width [min]",
        ylabel="Bootstrap IQR [min]",
        title="Cycle-level diagnostic",
    )
    axes[2].legend(fontsize=6.5)
    for label, axis in zip("abc", axes, strict=True):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontweight="bold")
        axis.grid(alpha=0.18)
    figure.suptitle(
        "A moving argmax matters only relative to the objective's near-optimal basin",
        x=0.05,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return figure


def _semantic_ablation_figure(
    comparison: pd.DataFrame,
    stability: pd.DataFrame,
    consequences: pd.DataFrame,
) -> plt.Figure:
    families = ("H", "O")
    family_colors = {"H": COLORS["eta_h_cyc"], "O": COLORS["eta_e_cyc"]}
    figure, axes = plt.subplots(2, 3, figsize=(10.2, 6.2))
    timing = [
        comparison.loc[comparison.family.eq(family), "delta_abs_minus_ret_minutes"].dropna()
        for family in families
    ]
    boxes = axes[0, 0].boxplot(timing, patch_artist=True, showfliers=False)
    for box, family in zip(boxes["boxes"], families, strict=True):
        box.set_facecolor(family_colors[family])
        box.set_alpha(0.65)
    axes[0, 0].axhline(0, color="#333333", ls="--", lw=0.8)
    axes[0, 0].set(
        xticks=(1, 2),
        xticklabels=families,
        ylabel=r"$\tau_{abs}-\tau_{ret}$ [min]",
        title="Does healthy normalization move the decision?",
    )
    x = np.arange(2)
    for offset, minutes in zip((-0.18, 0.18), (3, 5), strict=True):
        fraction = [
            comparison.loc[comparison.family.eq(family), "abs_delta_minutes"].le(minutes).mean()
            for family in families
        ]
        axes[0, 1].bar(x + offset, fraction, 0.34, label=fr"$|\Delta\tau|\leq{minutes}$ min")
    axes[0, 1].set(
        xticks=x,
        xticklabels=families,
        ylim=(0, 1),
        ylabel="Cycle fraction",
        title="Cycle-level decision equivalence",
    )
    axes[0, 1].legend(fontsize=6.5)

    for offset, form in zip((-0.18, 0.18), ("retention", "absolute_rate"), strict=True):
        values = [
            stability.loc[
                stability.family.eq(family) & stability.objective_form.eq(form),
                "valid_fraction",
            ].median()
            for family in families
        ]
        axes[0, 2].bar(x + offset, values, 0.34, label=form.replace("_", " "))
    axes[0, 2].axhline(0.8, color="#333333", ls="--", lw=0.8)
    axes[0, 2].set(
        xticks=x,
        xticklabels=families,
        ylim=(0, 1.05),
        ylabel="Median bootstrap-valid fraction",
        title="Does removing the counterfactual improve estimability?",
    )
    axes[0, 2].legend(fontsize=6.5)

    positions = np.arange(4)
    labels = ("H ret", "H abs", "O ret", "O abs")
    values = [
        100
        * consequences.loc[
            consequences.family.eq(family) & consequences.objective_form.eq(form),
            "C_regret",
        ].dropna()
        for family, form in (
            ("H", "retention"),
            ("H", "absolute_rate"),
            ("O", "retention"),
            ("O", "absolute_rate"),
        )
    ]
    boxes = axes[1, 0].boxplot(values, positions=positions, patch_artist=True, showfliers=False)
    for box, color in zip(
        boxes["boxes"],
        (family_colors["H"], family_colors["H"], family_colors["O"], family_colors["O"]),
        strict=True,
    ):
        box.set_facecolor(color)
        box.set_alpha(0.65)
    axes[1, 0].set(
        xticks=positions,
        xticklabels=labels,
        ylabel="C regret [%]",
        title="Frozen-C consequence",
    )

    selected = [
        comparison.loc[comparison.family.eq(family), f"{form}_cycle_minutes"].dropna()
        for family, form in (
            ("H", "retention"),
            ("H", "absolute"),
            ("O", "retention"),
            ("O", "absolute"),
        )
    ]
    axes[1, 1].boxplot(selected, positions=positions, showfliers=False)
    axes[1, 1].set(
        xticks=positions,
        xticklabels=labels,
        ylabel="Selected time from cycle start [min]",
        title="Absolute timing, not only paired difference",
    )
    states = [
        comparison.loc[comparison.family.eq(family), f"{form}_frosting_progress"].dropna()
        for family, form in (
            ("H", "retention"),
            ("H", "absolute"),
            ("O", "retention"),
            ("O", "absolute"),
        )
    ]
    axes[1, 2].boxplot(states, positions=positions, showfliers=False)
    axes[1, 2].set(
        xticks=positions,
        xticklabels=labels,
        ylabel="Frosting progress",
        title="Selected physical state",
    )
    for label, axis in zip("abcdef", axes.flat, strict=True):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontweight="bold")
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle(
        "Healthy-normalized retention is tested against counterfactual-free cycle rate",
        x=0.05,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _outdoor_event_ablation_figure(values: pd.DataFrame) -> plt.Figure:
    """Show whether direct event heat improves on component reconstruction."""
    observed = values["Qe_T_observed_kwh"].to_numpy(dtype=float)
    predictions = {
        "Component": values["Qe_T_component_prediction_kwh"].to_numpy(dtype=float),
        "Direct": values["Qe_T_direct_prediction_kwh"].to_numpy(dtype=float),
    }
    colors = {"Component": "#888888", "Direct": COLORS["eta_e_cyc"]}
    errors = {name: prediction - observed for name, prediction in predictions.items()}
    support = {
        "Component": values["Qe_T_component_supported"].astype(bool).mean(),
        "Direct": values["Qe_T_direct_supported"].astype(bool).mean(),
    }
    rmse = {name: float(np.sqrt(np.mean(error**2))) for name, error in errors.items()}
    improvement = 1 - rmse["Direct"] / rmse["Component"]
    support_gain = support["Direct"] - support["Component"]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    low = float(np.nanmin([observed, *predictions.values()]))
    high = float(np.nanmax([observed, *predictions.values()]))
    for name, prediction in predictions.items():
        axes[0, 0].scatter(observed, prediction, s=28, alpha=0.65, color=colors[name], label=name)
    axes[0, 0].plot([low, high], [low, high], "--", color="#333333", lw=1)
    axes[0, 0].set(
        xlabel="Observed signed outdoor-event heat [kWh]",
        ylabel="LOEO prediction [kWh]",
        title="Same events and features",
    )
    axes[0, 0].legend()

    axes[0, 1].boxplot(
        [np.abs(errors[name]) for name in predictions],
        tick_labels=list(predictions),
        patch_artist=True,
        boxprops={"facecolor": "#D9D9D9"},
        medianprops={"color": "#222222"},
    )
    axes[0, 1].set(ylabel="Absolute error [kWh]", title="Event-level error distribution")

    names = list(predictions)
    x = np.arange(len(names))
    mae = [float(np.mean(np.abs(errors[name]))) for name in names]
    axes[1, 0].bar(x - 0.18, mae, 0.36, color="#B4C0E4", label="MAE")
    axes[1, 0].bar(x + 0.18, [rmse[name] for name in names], 0.36, color="#484878", label="RMSE")
    axes[1, 0].set(
        xticks=x,
        xticklabels=names,
        ylabel="Error [kWh]",
        title=f"Direct RMSE improvement = {improvement:.1%}",
    )
    axes[1, 0].legend()

    experiment = values.assign(
        component_ae=np.abs(errors["Component"]),
        direct_ae=np.abs(errors["Direct"]),
    ).groupby("experiment_id", sort=False)[["component_ae", "direct_ae"]].mean()
    difference = (experiment["direct_ae"] - experiment["component_ae"]).sort_values()
    axes[1, 1].bar(
        np.arange(len(difference)),
        difference,
        color=np.where(difference.le(0), "#2E9E44", "#E53935"),
    )
    axes[1, 1].axhline(0, color="#333333", lw=1)
    axes[1, 1].set(
        xlabel="Held-out experiment",
        ylabel=r"Direct $-$ component MAE [kWh]",
        title=f"Support gain = {support_gain:+.1%}",
        xticks=[],
    )
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Direct event prediction is tested before changing the outdoor objective",
        fontweight="bold",
    )
    return figure


def _ch_tradeoff_figure(values: pd.DataFrame) -> plt.Figure:
    """Show the oracle H value available for specified native C tolerances."""
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)
    cycles = values.drop_duplicates("cycle_name")
    axes[0, 0].boxplot(
        [100 * cycles["H_regret_at_C"], 100 * cycles["C_regret_at_H"]],
        tick_labels=[r"$r_H(\tau_C)$", r"$r_C(\tau_H)$"],
        patch_artist=True,
        boxprops={"facecolor": "#D9D9E8"},
        medianprops={"color": "#333333"},
    )
    axes[0, 0].set(ylabel="Regret [%]", title="Native optima expose the endpoint trade-off")

    summary = values.groupby("epsilon_C")["H_gain_upper_bound"].agg(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        p90=lambda x: x.quantile(0.90),
    )
    x = 100 * summary.index.to_numpy(dtype=float)
    axes[0, 1].plot(x, 100 * summary["median"], marker="o", color=COLORS["eta_h_cyc"])
    axes[0, 1].fill_between(
        x,
        100 * summary["q25"].to_numpy(),
        100 * summary["q75"].to_numpy(),
        color=COLORS["eta_h_cyc"],
        alpha=0.2,
        label="IQR",
    )
    axes[0, 1].plot(x, 100 * summary["p90"], ls="--", color="#555555", label="P90")
    axes[0, 1].set(
        xlabel=r"Allowed native $C$ regret, $\epsilon_C$ [%]",
        ylabel="Oracle H gain over H at C optimum [%]",
        title="Potential value gain, not a selected policy",
    )
    axes[0, 1].legend()

    for threshold, color in ((0.01, "#7884B4"), (0.03, "#B64A50"), (0.05, "#2A788E")):
        fraction = values["H_gain_upper_bound"].ge(threshold).groupby(
            values["epsilon_C"]
        ).mean(
        )
        axes[1, 0].plot(
            100 * fraction.index,
            fraction,
            marker="o",
            color=color,
            label=f"H gain >= {100 * threshold:.0f}%",
        )
    axes[1, 0].set(
        xlabel=r"$\epsilon_C$ [%]",
        ylabel="Cycle fraction",
        ylim=(0, 1),
        title="Large service gains are not population-wide",
    )
    axes[1, 0].legend()

    experiment = values.groupby(["experiment_id", "epsilon_C"])[
        "H_gain_upper_bound"
    ].median().unstack()
    for _, row in experiment.iterrows():
        axes[1, 1].plot(100 * row.index, 100 * row, color="#AAAAAA", alpha=0.55, lw=0.8)
    axes[1, 1].plot(x, 100 * summary["median"], color="#222222", marker="o", lw=2)
    axes[1, 1].set(
        xlabel=r"$\epsilon_C$ [%]",
        ylabel="Experiment-median oracle H gain [%]",
        title="Held-out experiments are heterogeneous",
    )
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18)
    figure.suptitle(
        "G1 | Small C tolerance provides limited and heterogeneous H value",
        fontweight="bold",
    )
    return figure


def _ch_overlap_figure(overlap: pd.DataFrame, uncertainty: pd.DataFrame) -> plt.Figure:
    """Test whether native C/H near-optimal overlap exceeds timing uncertainty."""
    merged = overlap.merge(uncertainty, on="cycle_name", how="left", validate="many_to_one")
    merged["timing_IQR_minutes"] = merged[["C_IQR_minutes", "H_IQR_minutes"]].max(axis=1)
    merged["resolved"] = merged["longest_overlap_minutes"].gt(merged["timing_IQR_minutes"])
    width = merged.pivot_table(
        index="epsilon_H", columns="epsilon_C", values="longest_overlap_minutes", aggfunc="median"
    )
    coverage = merged.assign(has_overlap=merged["overlap_candidate_count"].gt(0)).pivot_table(
        index="epsilon_H", columns="epsilon_C", values="has_overlap", aggfunc="mean"
    )
    resolved = merged.pivot_table(
        index="epsilon_H", columns="epsilon_C", values="resolved", aggfunc="mean"
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)
    for axis, table, title, label, cmap, vmax in (
        (axes[0, 0], width, "Median longest overlap", "Minutes", "Blues", None),
        (axes[0, 1], coverage, "Cycles with any overlap", "Fraction", "Greens", 1),
        (axes[1, 1], resolved, "Overlap wider than bootstrap IQR", "Fraction", "Purples", 1),
    ):
        image = axis.imshow(table, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        for row in range(len(table.index)):
            for column in range(len(table.columns)):
                value = table.iloc[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.0f}" if label == "Minutes" else f"{value:.0%}",
                    ha="center",
                    va="center",
                )
        axis.set(
            xticks=range(len(table.columns)),
            xticklabels=[f"{100 * value:g}%" for value in table.columns],
            yticks=range(len(table.index)),
            yticklabels=[f"{100 * value:g}%" for value in table.index],
            xlabel=r"Native $C$ tolerance",
            ylabel=r"Native $H$ tolerance",
            title=title,
        )
        figure.colorbar(image, ax=axis, label=label, shrink=0.75)

    one = merged.loc[merged["epsilon_C"].eq(0.01) & merged["epsilon_H"].eq(0.01)].dropna(
        subset=["longest_overlap_minutes", "timing_IQR_minutes"]
    )
    axes[1, 0].scatter(
        one["timing_IQR_minutes"],
        one["longest_overlap_minutes"],
        color="#484878",
        alpha=0.65,
        s=26,
    )
    limit = max(1.0, float(one[["timing_IQR_minutes", "longest_overlap_minutes"]].max().max()))
    axes[1, 0].plot([0, limit], [0, limit], "--", color="#333333", lw=1)
    axes[1, 0].set(
        xlabel="Larger of C/H bootstrap timing IQR [min]",
        ylabel="Longest 1%/1% overlap [min]",
        title="A real plateau should lie above the diagonal",
    )
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold")
    figure.suptitle(
        "G2 | Apparent C-H overlap is usually narrower than model uncertainty",
        fontweight="bold",
    )
    return figure


def _g1_g2_decision_gate_figure(
    tradeoff: pd.DataFrame,
    overlap: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> plt.Figure:
    """Summarize the evidence gate from value objectives to physical triggering."""
    g1 = tradeoff.loc[tradeoff["epsilon_C"].eq(0.01)]
    g2 = overlap.loc[overlap["epsilon_C"].eq(0.01) & overlap["epsilon_H"].eq(0.01)].merge(
        uncertainty, on="cycle_name", how="left", validate="one_to_one"
    )
    timing_iqr = g2[["C_IQR_minutes", "H_IQR_minutes"]].max(axis=1)
    overlap_count = int(g2["overlap_candidate_count"].gt(0).sum())
    resolved_count = int(g2["longest_overlap_minutes"].gt(timing_iqr).sum())
    cycle_count = len(g2)

    figure, axis = plt.subplots(figsize=(11.5, 4.6), constrained_layout=True)
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    nodes = (
        (
            0.12,
            0.64,
            "C: protect cycle efficiency\n(native objective and support)",
            "#E5E5EF",
        ),
        (
            0.38,
            0.64,
            "G1: local C-H value trade-off\n"
            f"1% C allowance → {100 * g1['H_gain_upper_bound'].median():.2f}% median oracle H gain",
            "#F1E1E3",
        ),
        (
            0.64,
            0.64,
            "G2: independent native basins\n"
            f"1%/1% overlap: {overlap_count}/{cycle_count} cycles",
            "#E4EEF6",
        ),
        (
            0.88,
            0.64,
            "Conservative G2 screen\n"
            f"overlap > timing IQR: {resolved_count}/{cycle_count}",
            "#F7D9D7",
        ),
    )
    for x, y, label, color in nodes:
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": color, "edgecolor": "#555555"},
        )
    for start, end in ((0.21, 0.29), (0.47, 0.55), (0.73, 0.80)):
        axis.annotate("", xy=(end, 0.64), xytext=(start, 0.64), arrowprops={"arrowstyle": "->"})

    axis.text(
        0.5,
        0.25,
        "Engineering selector: valid → guardrail → Pareto → latest\n"
        "Latest is an explicit operational preference, not a statistically unique optimum",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#8F2F2F",
        bbox={"boxstyle": "round,pad=0.65", "facecolor": "#FFF4F3", "edgecolor": "#B64A50"},
    )
    axis.annotate(
        "",
        xy=(0.5, 0.36),
        xytext=(0.88, 0.53),
        arrowprops={"arrowstyle": "->", "color": "#B64A50", "lw": 1.5},
    )
    figure.suptitle(
        "G1–G2 screening | Evidence supports a simpler lexicographic policy",
        fontweight="bold",
    )
    return figure


def _gate_sensitivity(final_screen: pd.DataFrame, cop_screen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.5, 1.0, 11):
        for _, row in final_screen.iterrows():
            rows.append(
                {
                    "family": "parallel_objective",
                    "candidate": row["metric_id"],
                    "bootstrap_threshold": threshold,
                    "passes": bool(
                        row["identified_fraction"] >= 0.8
                        and row["median_support_fraction"] >= 0.8
                        and row["bootstrap_valid_fraction"] >= threshold
                    ),
                }
            )
        for _, row in cop_screen.loc[cop_screen["cross_fitted"]].iterrows():
            rows.append(
                {
                    "family": "historical_COP",
                    "candidate": row["version"],
                    "bootstrap_threshold": threshold,
                    "passes": bool(
                        row["identified_fraction"] >= 0.8
                        and row["median_support_fraction"] >= 0.6
                        and pd.notna(row["bootstrap_valid_fraction"])
                        and row["bootstrap_valid_fraction"] >= threshold
                    ),
                }
            )
    return pd.DataFrame(rows)


def _gate_sensitivity_figure(sensitivity: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for axis, family in zip(axes, ("historical_COP", "parallel_objective"), strict=True):
        values = sensitivity.loc[sensitivity["family"].eq(family)]
        for candidate, rows in values.groupby("candidate", sort=False):
            axis.step(
                rows["bootstrap_threshold"],
                rows["passes"].astype(int),
                where="post",
                label=LABELS.get(candidate, candidate.upper()),
            )
        axis.set(
            xlabel="Bootstrap-validity threshold",
            yticks=(0, 1),
            yticklabels=("Fail", "Pass"),
            title="COP family" if family == "historical_COP" else "Parallel objectives",
        )
        axis.legend(fontsize=6.5)
        axis.grid(alpha=0.18)
    figure.suptitle(
        "The 80% gate does not change the COP representative, but it does gate H/O",
        x=0.06,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure


def _physical_attribution(
    complete_cycle: pd.DataFrame,
    benchmark: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    local = local_ratio_attribution(complete_cycle)
    local["heating_contribution"] = (
        local["heating_energy_contribution"] + local["heating_heat_contribution"]
    ) / local["duration_minutes"]
    local["event_contribution"] = (
        local["event_energy_contribution"] + local["event_heat_contribution"]
    ) / local["duration_minutes"]
    local["inverse_cop_slope"] = local["delta_inverse_cop"] / local["duration_minutes"]
    result = local.loc[local["segment"].eq("optimum_to_after")].copy()
    selected = benchmark.loc[benchmark["metric_id"].eq("cop_cyc_evt")].set_index("cycle_name")
    uncertainty = stability.loc[stability["metric_id"].eq("cop_cyc_evt")].set_index("cycle_name")
    result = result.join(
        selected[["frosting_progress", "W5_minutes", "selected_objective"]],
        on="cycle_name",
    ).join(
        uncertainty[["valid_fraction", "IQR_tau_minutes"]],
        on="cycle_name",
    )
    result["relative_cost_slope_per_min"] = result["inverse_cop_slope"] / (
        1 / pd.to_numeric(result["selected_objective"], errors="coerce")
    )
    unreliable = result["valid_fraction"].lt(0.8) | result["IQR_tau_minutes"].gt(
        result["W5_minutes"]
    )
    suspicious_flat = result["frosting_progress"].lt(0.5) & result[
        "relative_cost_slope_per_min"
    ].abs().le(0.001)
    result["attribution_type"] = np.select(
        [
            unreliable,
            suspicious_flat,
            result["heating_contribution"].ge(result["event_contribution"]),
        ],
        [
            "D_unstable_label",
            "C_early_flat_suspicious",
            "A_heating_degradation",
        ],
        default="B_event_cost",
    )
    return result


def _physical_attribution_figure(attribution: pd.DataFrame) -> plt.Figure:
    values = attribution.sort_values("frosting_progress", kind="stable").reset_index(drop=True)
    x = np.arange(len(values))
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.0))
    axes[0, 0].bar(x, values["heating_contribution"], label="Heating contribution")
    axes[0, 0].bar(
        x,
        values["event_contribution"],
        bottom=values["heating_contribution"],
        label="Event contribution",
    )
    axes[0, 0].axhline(0, color="#333333", lw=0.7)
    axes[0, 0].set(
        xlabel="Cycles ordered by selected frosting progress",
        ylabel=r"Contribution to $\Delta(1/COP)$/min",
        title="Actual V2.6.8 components explain the post-optimum rise",
    )
    axes[0, 0].legend(fontsize=6.5)
    type_colors = {
        "A_heating_degradation": "#3B75AF",
        "B_event_cost": "#D8904F",
        "C_early_flat_suspicious": "#B64A50",
        "D_unstable_label": "#888888",
    }
    for kind, rows in values.groupby("attribution_type", sort=False):
        axes[0, 1].scatter(
            rows["frosting_progress"],
            100 * rows["relative_cost_slope_per_min"],
            color=type_colors[kind],
            label=kind.replace("_", " "),
            alpha=0.8,
        )
    axes[0, 1].axhline(0.1, color="#555555", ls="--", lw=0.8)
    axes[0, 1].set(
        xlabel="Selected frosting progress",
        ylabel="Post-optimum cost rise [%/min]",
        title="Early and flat optima remain explicit counterexamples",
    )
    axes[0, 1].legend(fontsize=6.2)
    counts = values["attribution_type"].value_counts().reindex(type_colors, fill_value=0)
    axes[1, 0].barh(
        [name.replace("_", " ") for name in counts.index],
        counts,
        color=[type_colors[name] for name in counts.index],
    )
    axes[1, 0].set(xlabel="Cycles", title="Attribution is diagnostic, not a new objective")
    axes[1, 1].scatter(
        values["IQR_tau_minutes"],
        values["W5_minutes"],
        c=[type_colors[kind] for kind in values["attribution_type"]],
        alpha=0.8,
    )
    limit = np.nanmax(values[["IQR_tau_minutes", "W5_minutes"]].to_numpy())
    axes[1, 1].plot([0, limit], [0, limit], color="#333333", ls="--", lw=0.8)
    axes[1, 1].set(
        xlabel="Bootstrap optimum IQR [min]",
        ylabel="Original 5% basin width [min]",
        title="Location uncertainty is compared with decision tolerance",
    )
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontweight="bold")
        axis.grid(alpha=0.16)
    figure.suptitle(
        "V2.6.8 early decisions are audited from implemented E/Q components",
        x=0.06,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _trigger_figure(
    trigger: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: dict[str, pd.DataFrame],
    frame_scores: pd.DataFrame,
) -> plt.Figure:
    modalities = [name for name in ("rgb", "time") if name in set(trigger["modality"])]
    labels = {"rgb": "Frozen RGB", "time": "Time only"}
    colors = {"rgb": "#484878", "time": "#D8904F"}
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 6.4))
    x = np.arange(len(modalities))
    f1 = (
        frame_scores.loc[
            frame_scores["metric"].eq("macro_f1") & frame_scores["camera_group"].eq("front")
        ]
        .set_index("modality")
        .reindex(modalities)
    )
    axes[0, 0].bar(x, f1["estimate"], color=[colors[m] for m in modalities])
    axes[0, 0].set(
        xticks=x,
        xticklabels=[labels[m] for m in modalities],
        ylim=(0.8, 1.01),
        ylabel="Macro-F1",
        title="Frame separability is only the first check",
    )
    timing = [
        trigger.loc[trigger["modality"].eq(modality), "signed_error_minutes"].dropna()
        for modality in modalities
    ]
    boxes = axes[0, 1].boxplot(timing, patch_artist=True, showfliers=False)
    for box, modality in zip(boxes["boxes"], modalities, strict=True):
        box.set_facecolor(colors[modality])
        box.set_alpha(0.7)
    axes[0, 1].axhline(0, color="#333333", ls="--", lw=0.8)
    axes[0, 1].set(
        xticks=range(1, len(modalities) + 1),
        xticklabels=[labels[m] for m in modalities],
        ylabel="Trigger minus oracle [min]",
        title="Three-frame trigger timing",
    )
    width = 0.22
    for offset, percent in zip((-width, 0, width), (1, 2, 5), strict=True):
        hits = [
            trigger.loc[trigger["modality"].eq(modality), f"W{percent}_hit"].mean()
            for modality in modalities
        ]
        axes[0, 2].bar(x + offset, hits, width, label=f"W{percent}")
    axes[0, 2].set(
        xticks=x,
        xticklabels=[labels[m] for m in modalities],
        ylim=(0, 1),
        ylabel="Cycle hit fraction",
        title="Near-optimal decision hits",
    )
    axes[0, 2].legend(fontsize=6.5)
    for offset, metric in zip((-width, 0, width), FINAL_METRICS, strict=True):
        regrets = [
            100 * trigger.loc[trigger["modality"].eq(modality), f"regret_{metric}"].median()
            for modality in modalities
        ]
        axes[1, 0].bar(x + offset, regrets, width, label=LABELS[metric], color=COLORS[metric])
    axes[1, 0].set(
        xticks=x,
        xticklabels=[labels[m] for m in modalities],
        ylabel="Median regret [%]",
        title="Implemented decisions are scored under C/H/O separately",
    )
    axes[1, 0].legend(fontsize=6.5)
    paired = trigger.pivot(index="cycle_name", columns="modality", values="absolute_error_minutes")
    if set(modalities) == {"rgb", "time"}:
        paired = paired.dropna(subset=modalities)
        axes[1, 1].scatter(paired["time"], paired["rgb"], color="#484878", alpha=0.75)
        limit = float(np.nanmax(paired[modalities].to_numpy())) if len(paired) else 1.0
        axes[1, 1].plot([0, limit], [0, limit], color="#333333", ls="--", lw=0.8)
    axes[1, 1].set(
        xlabel="Time-only absolute error [min]",
        ylabel="RGB absolute error [min]",
        title="Below the diagonal means RGB adds state information",
    )
    representative = (
        (paired["time"] - paired["rgb"]).idxmax()
        if set(modalities) == {"rgb", "time"} and len(paired)
        else trigger["cycle_name"].iloc[0]
    )
    for modality in modalities:
        curve = predictions.loc[
            predictions["cycle_name"].eq(representative)
            & predictions["modality"].eq(modality)
            & predictions["fold_evaluable"].fillna(False)
        ].sort_values("image_time")
        if curve.empty:
            continue
        time = pd.to_datetime(curve["image_time"], errors="coerce")
        start = time.min()
        axes[1, 2].plot(
            (time - start).dt.total_seconds() / 60,
            curve["decision_score"],
            color=colors[modality],
            label=labels[modality],
            alpha=0.85,
        )
    c_curve = metrics["cop_cyc_evt"].loc[metrics["cop_cyc_evt"]["cycle_name"].eq(representative)]
    if not c_curve.empty:
        start = pd.to_datetime(
            predictions.loc[predictions["cycle_name"].eq(representative), "image_time"]
        ).min()
        for field, alpha in (("basin_5pct_start", 0.12), ("basin_1pct_start", 0.22)):
            left = pd.to_datetime(c_curve[field].iloc[0], errors="coerce")
            end = pd.to_datetime(c_curve[field.replace("start", "end")].iloc[0], errors="coerce")
            axes[1, 2].axvspan(
                (left - start).total_seconds() / 60,
                (end - start).total_seconds() / 60,
                color="#B64A50",
                alpha=alpha,
            )
    axes[1, 2].axhline(0.5, color="#333333", ls="--", lw=0.8)
    axes[1, 2].set(
        ylim=(0, 1),
        xlabel="Cycle image time [min]",
        ylabel="Post-optimal probability",
        title=f"Representative trigger trajectory: {representative}",
    )
    axes[1, 2].legend(fontsize=6.5)
    for label, axis in zip("abcdef", axes.flat, strict=True):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontweight="bold")
        axis.grid(alpha=0.16)
    figure.suptitle(
        "Static RGB separability is tested as an executable cycle trigger",
        x=0.06,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _rgb_learnability(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = {
        "v1": "v0_dinov2_logistic_20260902",
        "v2.6.7": "v267_dinov2_logistic_20260902",
        "v2.6.8": "v268_dinov2_logistic_20260902",
    }
    metrics: list[pd.DataFrame] = []
    monotonicity: list[dict[str, object]] = []
    for version, run_id in runs.items():
        run = root / "runs" / run_id
        if not (run / "summary_metrics.csv").exists():
            continue
        summary = pd.read_csv(run / "summary_metrics.csv")
        summary["version"] = version
        summary["experiment_total"] = len(pd.read_csv(run / "cohort_holdout_audit.csv"))
        metrics.append(summary)
        predictions = pd.read_parquet(run / "predictions.parquet")
        valid = predictions.loc[
            predictions["fold_evaluable"] & predictions["decision_score"].notna()
        ]
        for keys, values in valid.groupby(
            ["camera_group", "cycle_name", "camera_role"], sort=False
        ):
            camera, cycle, role = keys
            time_rank = pd.to_datetime(values["image_time"], errors="coerce").rank()
            probability_rank = pd.to_numeric(values["decision_score"], errors="coerce").rank()
            correlation = (
                time_rank.corr(probability_rank)
                if time_rank.nunique() > 1 and probability_rank.nunique() > 1
                else np.nan
            )
            monotonicity.append(
                {
                    "version": version,
                    "camera_group": camera,
                    "cycle_name": cycle,
                    "camera_role": role,
                    "spearman_time_probability": correlation,
                }
            )
    return pd.concat(metrics, ignore_index=True), pd.DataFrame(monotonicity)


def _rgb_learnability_figure(metrics: pd.DataFrame, monotonicity: pd.DataFrame) -> plt.Figure:
    versions = [
        version for version in ("v1", "v2.6.7", "v2.6.8") if version in set(metrics.version)
    ]
    labels = ["V0 / original" if version == "v1" else version.upper() for version in versions]
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 5.6))
    colors = {"front": "#484878", "all": "#2A788E"}
    x = np.arange(len(versions))
    for offset, camera in zip((-0.18, 0.18), ("front", "all"), strict=True):
        for axis, metric in zip(axes[0], ("macro_f1", "balanced_accuracy"), strict=True):
            values = (
                metrics.loc[metrics.metric.eq(metric) & metrics.camera_group.eq(camera)]
                .set_index("version")
                .reindex(versions)
            )
            estimate = values["estimate"].to_numpy()
            axis.bar(x + offset, estimate, 0.36, color=colors[camera], label=camera)
            axis.errorbar(
                x + offset,
                estimate,
                yerr=np.vstack((estimate - values["lower"], values["upper"] - estimate)),
                fmt="none",
                ecolor="#333333",
                lw=0.7,
                capsize=2,
            )
    for axis, title in zip(axes[0], ("Macro-F1", "Balanced accuracy"), strict=True):
        axis.set(xticks=x, xticklabels=labels, ylim=(0.85, 1.01), ylabel=title, title=title)
        axis.tick_params(axis="x", rotation=12)
    axes[0, 0].legend(fontsize=7)
    coverage = (
        metrics.loc[metrics.metric.eq("macro_f1") & metrics.camera_group.eq("front")]
        .set_index("version")
        .reindex(versions)
    )
    axes[1, 0].bar(
        x,
        coverage["evaluable_experiment_count"] / coverage["experiment_total"],
        color=["#777777", "#9A4D8E", "#C44E52"][: len(versions)],
    )
    axes[1, 0].axhline(0.8, color="#333333", ls="--", lw=0.8)
    axes[1, 0].set(
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1.05),
        ylabel="Evaluable experiment fraction",
        title="A high score is insufficient without experiment coverage",
    )
    axes[1, 0].tick_params(axis="x", rotation=12)
    mono = monotonicity.loc[monotonicity.camera_group.eq("front")]
    values = [
        mono.loc[mono.version.eq(version), "spearman_time_probability"].dropna()
        for version in versions
    ]
    boxes = axes[1, 1].boxplot(values, patch_artist=True, showfliers=False)
    for box, color in zip(boxes["boxes"], ("#777777", "#9A4D8E", "#C44E52"), strict=False):
        box.set_facecolor(color)
        box.set_alpha(0.7)
    axes[1, 1].axhline(0, color="#333333", ls="--", lw=0.8)
    axes[1, 1].set(
        xticks=range(1, len(versions) + 1),
        xticklabels=labels,
        ylabel="Cycle-level Spearman ρ",
        title="Post-optimal probability should generally rise with frosting time",
    )
    axes[1, 1].tick_params(axis="x", rotation=12)
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontweight="bold")
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle(
        "RGB learnability uses one frozen label-neutral representation and separate labels",
        x=0.06,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _final_metric_screen(
    benchmark: pd.DataFrame,
    stability: pd.DataFrame,
    regret: pd.DataFrame,
) -> pd.DataFrame:
    result = benchmark.groupby("metric_id").agg(
        identified_fraction=("t_star", lambda values: values.notna().mean()),
        median_support_fraction=("support_fraction", "median"),
        interior_fraction=("extreme_location", lambda values: values.eq("interior").mean()),
        median_W1_minutes=("W1_minutes", "median"),
        median_W5_minutes=("W5_minutes", "median"),
    )
    result = result.join(
        stability.groupby("metric_id").agg(
            bootstrap_valid_fraction=("valid_fraction", "median"),
            median_IQR_tau_minutes=("IQR_tau_minutes", "median"),
            median_MAD_tau_minutes=("MAD_tau_minutes", "median"),
            median_p90_self_regret=("p90_self_regret", "median"),
        )
    )
    result["passes_evidence_gate"] = (
        result["identified_fraction"].ge(0.8)
        & result["median_support_fraction"].ge(0.8)
        & result["bootstrap_valid_fraction"].ge(0.8)
    )
    for decision in ("point", "latest_W1"):
        matched = matched_decision_regret(regret, decision)
        vectors = matched.pivot_table(
            index="selector_metric",
            columns="target_metric",
            values="cross_objective_regret",
            aggfunc="median",
        ).reindex(index=FINAL_METRICS, columns=FINAL_METRICS)
        result = result.join(vectors.add_prefix(f"{decision}_median_regret_"))
        result[f"{decision}_matched_cycles"] = matched["cycle_name"].nunique()
        regret_columns = [f"{decision}_median_regret_{metric}" for metric in FINAL_METRICS]
        candidates = result.loc[result["passes_evidence_gate"], regret_columns]
        result[f"{decision}_pareto"] = pareto_nondominated(candidates).reindex(
            result.index, fill_value=False
        )
    return result.reset_index()


def _final_metric_screen_figure(screen: pd.DataFrame) -> plt.Figure:
    values = screen.set_index("metric_id").reindex(FINAL_METRICS)
    x = np.arange(len(FINAL_METRICS))
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 5.6))
    axes[0, 0].bar(x - 0.22, values["identified_fraction"], 0.22, label="Identified")
    axes[0, 0].bar(x, values["median_support_fraction"], 0.22, label="Support")
    axes[0, 0].bar(
        x + 0.22,
        values["bootstrap_valid_fraction"],
        0.22,
        label="Bootstrap valid",
    )
    axes[0, 0].axhline(0.8, color="#333333", ls="--", lw=0.8)
    axes[0, 0].set(ylim=(0, 1.05), ylabel="Fraction", title="Evidence gate is applied first")
    axes[0, 0].legend(fontsize=6.5)
    axes[0, 1].bar(x - 0.18, values["median_W1_minutes"], 0.36, label="W1")
    axes[0, 1].bar(x + 0.18, values["median_IQR_tau_minutes"], 0.36, label="Bootstrap IQR")
    axes[0, 1].set(ylabel="Minutes", title="Point resolution and resampling stability")
    axes[0, 1].legend(fontsize=7)
    width = 0.24
    for offset, target in zip((-width, 0, width), FINAL_METRICS, strict=True):
        axes[1, 0].bar(
            x + offset,
            100 * values[f"point_median_regret_{target}"],
            width,
            label=LABELS[target],
            color=COLORS[target],
        )
    axes[1, 0].set(ylabel="Median regret [%]", title="Point decisions: matched-cycle consequences")
    axes[1, 0].legend(fontsize=6.5)
    gates = values[["passes_evidence_gate", "point_pareto", "latest_W1_pareto"]].astype(float)
    axes[1, 1].imshow(gates, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[1, 1].set(
        xticks=range(3),
        xticklabels=["Gate", "Point Pareto", "Latest-W1 Pareto"],
        title="Pareto uses only the three physical regrets",
    )
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(x)
        axis.set_xticklabels([LABELS[metric] for metric in FINAL_METRICS])
        axis.grid(axis="y", alpha=0.18)
    axes[1, 1].set_yticks(x)
    axes[1, 1].set_yticklabels([LABELS[metric] for metric in FINAL_METRICS])
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontweight="bold")
    figure.suptitle(
        "Parallel objectives: evidence qualification precedes decision-level Pareto",
        x=0.06,
        ha="left",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-root", type=Path, default=Path("output/成本函数"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/test/成本函数/最终平行指标"),
    )
    parser.add_argument("--bootstrap-trajectory", type=Path)
    parser.add_argument(
        "--rgb-root",
        type=Path,
        default=Path("output/test/model/成本函数RGB可学习性"),
    )
    parser.add_argument("--trigger-run", type=Path)
    args = parser.parse_args()

    metrics = final_metric_tables(
        pd.read_csv(args.cost_root / "cost_function_v2.6.8.csv", low_memory=False),
        pd.read_csv(args.cost_root / "cost_function_v2.7.4.csv", low_memory=False),
        pd.read_csv(args.cost_root / "cost_function_v2.7.0.csv", low_memory=False),
    )
    summary = benchmark_table(metrics)
    regret = cross_objective_regret(metrics)
    absolute_metrics = absolute_rate_metric_tables(metrics)
    absolute_regret = cross_objective_regret(
        {"cop_cyc_evt": metrics["cop_cyc_evt"], **absolute_metrics},
        metric_order=("cop_cyc_evt", "h_abs_rate", "o_abs_rate"),
    )
    absolute_paired = ho_paired_decisions(
        absolute_regret,
        h_metric="h_abs_rate",
        o_metric="o_abs_rate",
        metric_order=("cop_cyc_evt", "h_abs_rate", "o_abs_rate"),
    )
    ch_tradeoff = ch_tradeoff_diagnostic(metrics["cop_cyc_evt"], absolute_metrics["h_abs_rate"])
    ch_overlap = ch_high_value_overlap(metrics["cop_cyc_evt"], absolute_metrics["h_abs_rate"])
    args.output.mkdir(parents=True, exist_ok=True)
    for metric_id, table in metrics.items():
        table[[column for column in PUBLICATION_COLUMNS if column in table]].to_csv(
            args.output / f"cost_function_v2.7_final_{metric_id}.csv", index=False
        )
    summary.to_csv(args.output / "benchmark.csv", index=False)
    regret.to_csv(args.output / "cross_objective_regret.csv", index=False)
    absolute_paired.to_csv(args.output / "absolute_HO_paired_decisions.csv", index=False)
    ch_tradeoff.to_csv(args.output / "CH_tradeoff_diagnostic.csv", index=False)
    ch_overlap.to_csv(args.output / "CH_high_value_overlap.csv", index=False)
    for metric_id, table in absolute_metrics.items():
        table.to_csv(args.output / f"semantic_ablation_{metric_id}.csv", index=False)
    coverage = regret_coverage(regret)
    coverage.to_csv(args.output / "cross_objective_regret_coverage.csv", index=False)
    cop_screen = _cop_history_screen(args.cost_root, metrics)
    if args.rgb_root.exists():
        rgb_metrics, monotonicity = _rgb_learnability(args.rgb_root)
        rgb_metrics.to_csv(args.output / "rgb_learnability.csv", index=False)
        monotonicity.to_csv(args.output / "rgb_probability_monotonicity.csv", index=False)
        front_f1 = rgb_metrics.loc[
            rgb_metrics.metric.eq("macro_f1") & rgb_metrics.camera_group.eq("front")
        ].set_index("version")
        for index, row in cop_screen.iterrows():
            if row["version"] not in front_f1.index:
                continue
            score = front_f1.loc[row["version"]]
            cop_screen.loc[index, "rgb_learnability_tested"] = True
            cop_screen.loc[index, "rgb_front_macro_f1"] = score["estimate"]
            cop_screen.loc[index, "rgb_evaluable_experiment_fraction"] = (
                score["evaluable_experiment_count"] / score["experiment_total"]
            )
        cop_screen["passes_rgb_gate"] = cop_screen["rgb_front_macro_f1"].ge(0.8) & cop_screen[
            "rgb_evaluable_experiment_fraction"
        ].ge(0.8)
        cop_screen["final_cop_choice"] = (
            cop_screen["provisional_cop_choice"] & cop_screen["passes_rgb_gate"]
        )
        _save(
            _rgb_learnability_figure(rgb_metrics, monotonicity),
            args.output,
            "16_RGB可学习性",
        )
    cop_screen.to_csv(args.output / "cop_history_screen.csv", index=False)
    _save(_benchmark_figure(summary), args.output, "11_三条平行指标统一benchmark")
    _save(_regret_figure(regret), args.output, "12_cross_objective_regret")
    _save(_regret_coverage_figure(coverage), args.output, "12B_cross_objective_regret覆盖率")
    _save(_physical_state_figure(summary), args.output, "15_三个平行指标选择的物理状态")
    _save(
        _ho_family_figure(absolute_paired),
        args.output,
        "28_Habs与Oabs是否可作为同一触发代理",
    )
    _save(
        _ch_tradeoff_figure(ch_tradeoff),
        args.output,
        "29_G1_C容忍度能否换来稳定H增益",
    )
    _save(
        _cop_history_evidence_figure(cop_screen),
        args.output,
        "14A_COP历史全家族_证据状态",
    )
    _save(
        _cop_history_consequence_figure(cop_screen),
        args.output,
        "14B_COP历史全家族_W1与三目标regret",
    )
    _save(
        _cop_history_finalists_figure(cop_screen),
        args.output,
        "14C_COP关键版本公平比较",
    )
    _save(
        _cop_history_definition_figure(cop_screen),
        args.output,
        "14D_COP历史全家族_定义与作用",
    )
    validation_path = args.cost_root / "cost_function_v2.7_validation.csv"
    if validation_path.exists():
        outdoor_event = outdoor_event_model_ablation(
            pd.read_csv(validation_path, low_memory=False)
        )
        outdoor_event.to_csv(
            args.output / "outdoor_event_direct_vs_component.csv", index=False
        )
        _save(
            _outdoor_event_ablation_figure(outdoor_event),
            args.output,
            "27_室外侧事件热量_direct与component消融",
        )

    if args.bootstrap_trajectory and args.bootstrap_trajectory.exists():
        trajectories = pd.read_parquet(args.bootstrap_trajectory)
        stability = bootstrap_stability(trajectories, metrics)
        stability.to_csv(args.output / "bootstrap_stability.csv", index=False)
        _save(_bootstrap_figure(stability), args.output, "13_bootstrap稳定性与self_regret")
        taxonomy = bootstrap_validity_taxonomy(trajectories)
        taxonomy.to_csv(args.output / "bootstrap_validity_taxonomy.csv", index=False)
        _save(
            _bootstrap_taxonomy_figure(taxonomy),
            args.output,
            "13B_bootstrap无效原因拆分",
        )
        final_screen = _final_metric_screen(summary, stability, regret)
        final_screen.to_csv(args.output / "final_metric_screen.csv", index=False)
        _save(
            _final_metric_screen_figure(final_screen),
            args.output,
            "17_三条平行指标_证据门槛与决策Pareto",
        )
        sensitivity = _gate_sensitivity(final_screen, cop_screen)
        sensitivity.to_csv(args.output / "gate_sensitivity.csv", index=False)
        _save(
            _gate_sensitivity_figure(sensitivity),
            args.output,
            "18_证据门槛敏感性",
        )
        attribution = _physical_attribution(metrics["cop_cyc_evt"], summary, stability)
        attribution.to_csv(args.output / "v268_local_component_attribution.csv", index=False)
        _save(
            _physical_attribution_figure(attribution),
            args.output,
            "19_V2.6.8早期最优点组成归因",
        )
        anatomy = bootstrap_failure_anatomy(trajectories)
        cofailure = bootstrap_ho_cofailure(anatomy)
        conditional = bootstrap_fixed_support_stability(trajectories, metrics)
        anatomy.to_csv(args.output / "bootstrap_failure_anatomy.csv", index=False)
        cofailure.to_csv(args.output / "bootstrap_ho_cofailure.csv", index=False)
        conditional.to_csv(
            args.output / "bootstrap_fixed_support_stability.csv", index=False
        )
        _save(
            _estimability_anatomy_figure(anatomy, cofailure, stability, conditional),
            args.output,
            "21_HO_bootstrap可估计性解剖",
        )
        draws_path = args.cost_root / "cost_function_v2.7_bootstrap_draws.csv"
        if draws_path.exists():
            leverage = experiment_leverage(anatomy, pd.read_csv(draws_path))
            leverage.to_csv(args.output / "bootstrap_experiment_leverage.csv", index=False)
            _save(
                _leverage_figure(leverage),
                args.output,
                "22_HO实验状态leverage",
            )
        paired = ho_paired_decisions(regret)
        paired.to_csv(args.output / "ho_paired_decisions.csv", index=False)
        _save(
            _ho_family_figure(paired),
            args.output,
            "23_HO是否属于同一退化决策族",
        )
        common_regret = same_cycle_regret(regret)
        common_regret.to_csv(args.output / "same_cycle_selector_regret.csv", index=False)
        distributions = pd.concat(
            [
                regret_distribution(common_regret, decision)
                for decision in ("point", "latest_W1", "latest_W2", "latest_W5")
            ],
            ignore_index=True,
        )
        distributions.to_csv(args.output / "regret_distribution.csv", index=False)
        _save(
            _same_cycle_regret_figure(common_regret),
            args.output,
            "24_同循环决策语义与regret尾部",
        )
        rho = stability_to_basin_ratio(summary, stability)
        rho.to_csv(args.output / "bootstrap_IQR_to_W5.csv", index=False)
        _save(
            _rho_figure(rho),
            args.output,
            "25_bootstrap位置不确定性相对近优盆地",
        )
        absolute_summary = benchmark_table(absolute_metrics)
        comparison_rows = []
        for family, retention_id, absolute_id in (
            ("H", "eta_h_cyc", "h_abs_rate"),
            ("O", "eta_e_cyc", "o_abs_rate"),
        ):
            retention = summary.loc[summary.metric_id.eq(retention_id), [
                "cycle_name",
                "t_star_cycle_minutes",
                "frosting_progress",
            ]].rename(
                columns={
                    "t_star_cycle_minutes": "retention_cycle_minutes",
                    "frosting_progress": "retention_frosting_progress",
                }
            )
            absolute = absolute_summary.loc[absolute_summary.metric_id.eq(absolute_id), [
                "cycle_name",
                "t_star_cycle_minutes",
                "frosting_progress",
            ]].rename(
                columns={
                    "t_star_cycle_minutes": "absolute_cycle_minutes",
                    "frosting_progress": "absolute_frosting_progress",
                }
            )
            paired = retention.merge(absolute, on="cycle_name", validate="one_to_one")
            paired["family"] = family
            paired["delta_abs_minus_ret_minutes"] = (
                paired["absolute_cycle_minutes"] - paired["retention_cycle_minutes"]
            )
            paired["abs_delta_minutes"] = paired["delta_abs_minus_ret_minutes"].abs()
            comparison_rows.append(paired)
        semantic_comparison = pd.concat(comparison_rows, ignore_index=True)
        semantic_comparison.to_csv(
            args.output / "semantic_ablation_timing.csv", index=False
        )
        absolute_trajectories = bootstrap_absolute_rate_trajectories(trajectories)
        absolute_stability = bootstrap_stability(
            absolute_trajectories,
            absolute_metrics,
            metrics=("h_abs_rate", "o_abs_rate"),
        )
        retention_stability = stability.loc[
            stability.metric_id.isin(("eta_h_cyc", "eta_e_cyc"))
        ].copy()
        retention_stability["family"] = retention_stability["metric_id"].map(
            {"eta_h_cyc": "H", "eta_e_cyc": "O"}
        )
        retention_stability["objective_form"] = "retention"
        absolute_stability["family"] = absolute_stability["metric_id"].map(
            {"h_abs_rate": "H", "o_abs_rate": "O"}
        )
        absolute_stability["objective_form"] = "absolute_rate"
        timing_uncertainty = stability.loc[
            stability["metric_id"].eq("cop_cyc_evt"), ["cycle_name", "IQR_tau_minutes"]
        ].rename(columns={"IQR_tau_minutes": "C_IQR_minutes"}).merge(
            absolute_stability.loc[
                absolute_stability["metric_id"].eq("h_abs_rate"),
                ["cycle_name", "IQR_tau_minutes"],
            ].rename(columns={"IQR_tau_minutes": "H_IQR_minutes"}),
            on="cycle_name",
            how="inner",
            validate="one_to_one",
        )
        timing_uncertainty.to_csv(
            args.output / "CH_bootstrap_timing_uncertainty.csv", index=False
        )
        _save(
            _ch_overlap_figure(ch_overlap, timing_uncertainty),
            args.output,
            "30_G2_C_H共同高价值窗口是否超过不确定性",
        )
        _save(
            _g1_g2_decision_gate_figure(ch_tradeoff, ch_overlap, timing_uncertainty),
            args.output,
            "31_G1_G2证据门与下一步",
        )
        semantic_stability = pd.concat(
            [retention_stability, absolute_stability], ignore_index=True, sort=False
        )
        semantic_stability.to_csv(
            args.output / "semantic_ablation_bootstrap.csv", index=False
        )
        retention_consequence = regret.loc[
            regret.decision_type.eq("point")
            & regret.target_metric.eq("cop_cyc_evt")
            & regret.selector_metric.isin(("eta_h_cyc", "eta_e_cyc"))
        ].copy()
        retention_consequence["family"] = retention_consequence["selector_metric"].map(
            {"eta_h_cyc": "H", "eta_e_cyc": "O"}
        )
        retention_consequence["objective_form"] = "retention"
        absolute_consequence = absolute_regret.loc[
            absolute_regret.decision_type.eq("point")
            & absolute_regret.target_metric.eq("cop_cyc_evt")
            & absolute_regret.selector_metric.isin(("h_abs_rate", "o_abs_rate"))
        ].copy()
        absolute_consequence["family"] = absolute_consequence["selector_metric"].map(
            {"h_abs_rate": "H", "o_abs_rate": "O"}
        )
        absolute_consequence["objective_form"] = "absolute_rate"
        semantic_consequence = pd.concat(
            [retention_consequence, absolute_consequence], ignore_index=True, sort=False
        ).rename(columns={"cross_objective_regret": "C_regret"})
        semantic_consequence = pd.concat(
            [
                values.loc[
                    values.cycle_name.isin(
                        set(values.loc[values.objective_form.eq("retention"), "cycle_name"])
                        & set(
                            values.loc[
                                values.objective_form.eq("absolute_rate"), "cycle_name"
                            ]
                        )
                    )
                ]
                for _, values in semantic_consequence.groupby("family", sort=False)
            ],
            ignore_index=True,
        )
        semantic_consequence.to_csv(
            args.output / "semantic_ablation_C_regret.csv", index=False
        )
        _save(
            _semantic_ablation_figure(
                semantic_comparison,
                semantic_stability,
                semantic_consequence,
            ),
            args.output,
            "26_HO健康保持率与绝对循环能力语义消融",
        )
    if args.trigger_run and (args.trigger_run / "predictions.parquet").exists():
        predictions = pd.read_parquet(args.trigger_run / "predictions.parquet")
        trigger = cycle_trigger_validation(
            predictions.loc[predictions["camera_group"].eq("front")], metrics
        )
        trigger.to_csv(args.output / "rgb_trigger_validation.csv", index=False)
        frame_scores = pd.read_csv(args.trigger_run / "summary_metrics.csv")
        _save(
            _trigger_figure(trigger, predictions, metrics, frame_scores),
            args.output,
            "20_RGB与time_only触发时刻验证",
        )


if __name__ == "__main__":
    main()
