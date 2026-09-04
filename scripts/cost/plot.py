#!/usr/bin/env python3
"""Compare defrost cost optima and render cycle publication PNGs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from frost_analysis.dataset.images import (
    materialize_cycle_image_members,
    scan_cycle_images,
)
from frost_analysis.dataset.loader import DatasetLoader
from frost_analysis.figures.visualization import (
    _plot_decision_image,
    ch_pareto_table,
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
}
V26_PATCHES = tuple(f"v2.6.{patch}" for patch in range(1, 9))
V26_PATCH_STYLES = {
    "v2.6.1": ("#4C566A", "h", "V2.6.1 baseline"),
    "v2.6.2": ("#3B75AF", "D", "V2.6.2 closed cycle"),
    "v2.6.3": ("#D99032", "s", "V2.6.3 degradation"),
    "v2.6.4": ("#7A5AA6", "^", "V2.6.4 marginal"),
    "v2.6.5": ("#B64A50", "*", "V2.6.5 decision"),
    "v2.6.6": ("#1B7F79", "p", "V2.6.6 diagnostic minimum"),
    "v2.6.7": ("#9A4D8E", "o", "V2.6.7 diagnostic minimum"),
    "v2.6.8": ("#C44E52", "X", "V2.6.8 diagnostic minimum"),
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
        "v2.6.8": (0, (2, 1)),
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
    "v2.6.8": V267_STATUS_MARKERS,
}
V267_DISPLAY_METRIC = "display_only_inverse_cop"
V268_DISPLAY_METRIC = "display_only_J_model"
V27_METRIC_STYLES = {
    "cop_cyc_evt": ("#484878", "D", r"$COP_{cyc,evt}$"),
    "eta_h_cyc": ("#B64A50", "^", r"$\eta_{H,cyc}$"),
    "eta_e_cyc": ("#0F4D92", "o", r"$\eta_{e,cyc}$"),
    "cop_e": ("#42949E", "s", r"$COP_e$"),
    "epsilon_hl": ("#B64342", "^", r"$\epsilon_{HL}$"),
    "epsilon_hl_t0_proxy": ("#E9A6A1", "v", r"$\epsilon_{HL}$ t0 proxy"),
    "cop_cyc_k": ("#7C6CCF", "D", r"$COP_{cyc,K}$"),
    "epsilon_hl_2a": ("#E28E2C", "P", "Two-anchor loss"),
}
V27_BOUNDARY_LABELS = {
    "cop_cyc_evt": r"$COP_{cyc,evt}$ · fixed-9 stable-to-stable",
    "eta_h_cyc": r"$\eta_{H,cyc}$ · fixed-9 stable-to-stable",
    "epsilon_hl": r"$\epsilon_{HL}$ · fixed-9 stable-to-stable",
    "epsilon_hl_t0_proxy": r"$\epsilon_{HL}$ t0 proxy · fixed-9 stable-to-stable",
    "cop_cyc_k": r"$COP_{cyc,K}$ · leading recovery + heating + prep/D",
}
V27_VALIDATION_MODELS = ("mean_baseline", "static_5", "physical_static_6", "dynamic_8")
V27_VALIDATION_MODEL_LABELS = {
    "mean_baseline": "Mean baseline",
    "static_5": "Static-5",
    "physical_static_6": "Physical-static-6",
    "dynamic_8": "Dynamic-8",
}
V27_VALIDATION_MODEL_COLORS = {
    "mean_baseline": "#7A7A7A",
    "static_5": "#7884B4",
    "physical_static_6": "#D99032",
    "dynamic_8": "#B64A50",
}
V27_VALIDATION_MODEL_MARKERS = {
    "mean_baseline": "o",
    "static_5": "s",
    "physical_static_6": "^",
    "dynamic_8": "D",
}
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
    if algorithm in V27_METRIC_STYLES:
        return V27_METRIC_STYLES[algorithm]
    return V26_PATCH_STYLES[algorithm] if algorithm in V26_PATCH_STYLES else STYLES[algorithm]


def _read_tables(sources: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for label, path in sources.items():
        table = pd.read_csv(path)
        algorithm = str(table["algorithm"].dropna().iloc[0]).lower()
        usable = table["valid" if "valid" in table else "optimization_eligible"].fillna(False)
        admitted = (
            pd.Series(True, index=table.index)
            if algorithm in {"v2.6.7", "v2.6.8"}
            else usable.groupby(table["cycle_name"]).transform("any")
        )
        table = table.loc[admitted].copy()
        tables[algorithm] = table
        tables[algorithm].attrs["label"] = str(label)
    return tables


def _read_v27_metrics(sources: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    """Split V2.7 CSVs by native metric without changing objective direction."""
    metrics: dict[str, pd.DataFrame] = {}
    for path in sources.values():
        table = pd.read_csv(path)
        if "metric_id" not in table:
            continue
        for metric_id, values in table.groupby("metric_id", sort=False):
            metrics[str(metric_id)] = values.copy()
    return metrics


def _read_v27_validation(shared_root: Path) -> pd.DataFrame:
    """Read V2.7 validation rows, with a CSV-only compatibility fallback.

    Early V2.7 exports contain one row per event because the V2.7 event-outcome
    columns were joined onto the V2.6.8 validation artifact.  The four model
    rows and the ``J_w`` fields are already present in that sibling CSV, so
    joining those exported rows here keeps plotting read-only and avoids
    silently presenting a single model as a four-model comparison.
    """
    path = shared_root / "cost_function_v2.7_validation.csv"
    if not path.exists():
        return pd.DataFrame()
    validation = pd.read_csv(path)
    if "model_name" in validation.columns:
        return validation

    fallback_path = shared_root / "cost_function_v2.6.8_validation.csv"
    if not fallback_path.exists():
        return validation
    fallback = pd.read_csv(fallback_path)
    if "model_name" not in fallback.columns or "cycle_name" not in validation.columns:
        return validation

    derived_columns = [
        column
        for column in (
            "L_T_dynamic_observed_kwh",
            "L_T_dynamic_prediction_kwh",
            "L_T_t0_observed_kwh",
            "L_T_t0_prediction_kwh",
        )
        if column in validation.columns
    ]
    if not derived_columns:
        return fallback
    derived = validation[["cycle_name", *derived_columns]].drop_duplicates("cycle_name")
    return fallback.merge(derived, on="cycle_name", how="left", validate="many_to_one")


def _write_v27_summaries(
    metrics: Mapping[str, pd.DataFrame],
    historical: Mapping[str, pd.DataFrame],
    sources: Mapping[str, Path],
    output: Path,
    *,
    diagnostics: bool,
) -> None:
    """Append V2.7 comparison and diagnostic summaries; never render cycle figures."""
    metric_algorithms = tuple(
        metric_id for metric_id in V27_METRIC_STYLES if metric_id in metrics
    )
    metric_algorithms += tuple(
        metric_id for metric_id in metrics if metric_id not in metric_algorithms
    )
    _save_svg_png(
        _comparison_figure(metrics, metric_algorithms), output / "comparison_v2.7_RB.png"
    )
    required = {"v1", "v2.5", "v2.6.7", "v2.6.8"}
    if required <= set(historical):
        cross_tables = {
            **metrics,
            **{algorithm: historical[algorithm] for algorithm in sorted(required)},
        }
        cross_algorithms = metric_algorithms + tuple(sorted(required))
        _save_svg_png(
            _comparison_figure(cross_tables, cross_algorithms),
            output / "comparison_v1_v2.5_v2.6.7_v2.6.8_v2.7_RB.png",
        )
    if not diagnostics:
        return
    v27_paths = [
        path for path in sources.values() if str(path.name).startswith("cost_function_v2.7")
    ]
    shared_root = v27_paths[0].parent if v27_paths else output
    identifiability_path = shared_root / "cost_function_v2.7_identifiability.csv"
    validation = _read_v27_validation(shared_root)
    identifiability = (
        pd.read_csv(identifiability_path) if identifiability_path.exists() else pd.DataFrame()
    )
    _write_v27_diagnostic_figures(
        metrics,
        historical,
        output,
        validation=validation,
        identifiability=identifiability,
    )


def _v27_summary(metrics: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric_id, table in metrics.items():
        for cycle_name, curve in table.groupby("cycle_name", sort=False):
            first = curve.iloc[0]
            location = str(first.get("extreme_location", "")).removesuffix("_boundary")
            rows.append(
                {
                    "metric_id": metric_id,
                    "cycle_name": cycle_name,
                    "experiment_id": first.get("experiment_id", "unknown"),
                    "identified": pd.notna(pd.to_datetime(first.get("t_star"), errors="coerce")),
                    "supported_fraction": float(curve["supported"].fillna(False).mean()),
                    "physical_fraction": float(curve["physical_valid"].fillna(False).mean()),
                    "identifiable_fraction": float(curve["identifiable"].fillna(False).mean()),
                    "interior": location == "interior",
                    "endpoint": location in {"left", "right"},
                    "basin_1pct_width_minutes": pd.to_numeric(
                        first.get("basin_1pct_width_minutes"), errors="coerce"
                    ),
                    "basin_5pct_width_minutes": pd.to_numeric(
                        first.get("basin_5pct_width_minutes"), errors="coerce"
                    ),
                    "bootstrap_basin_fraction": pd.to_numeric(
                        first.get("bootstrap_in_original_5pct_basin_fraction"), errors="coerce"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _preferred_metric(summary: pd.DataFrame) -> str:
    """Return a Pareto-dominant metric or explicitly abstain."""
    aggregate = summary.groupby("metric_id").agg(
        identified=("identified", "mean"),
        supported=("supported_fraction", "mean"),
        interior=("interior", "mean"),
        basin_1=("basin_1pct_width_minutes", "median"),
        bootstrap=("bootstrap_basin_fraction", "median"),
    )
    eligible = aggregate.loc[aggregate["identified"].ge(0.8) & aggregate["supported"].ge(0.8)]
    for metric_id, row in eligible.iterrows():
        others = eligible.drop(index=metric_id)
        if others.empty:
            continue
        no_worse = (
            row["interior"] >= others["interior"].max()
            and row["basin_1"] <= others["basin_1"].min()
            and row["bootstrap"] >= others["bootstrap"].max()
        )
        strictly_better = (
            row["interior"] > others["interior"].max()
            or row["basin_1"] < others["basin_1"].min()
            or row["bootstrap"] > others["bootstrap"].max()
        )
        if no_worse and strictly_better:
            return str(metric_id)
    return "no_unique_winner"


def _diagnostic_root(output: Path) -> Path:
    return (
        output.parent / "test" / output.name / "评价指标比较"
        if output.name == "成本函数"
        else output / "评价指标比较"
    )


def _metric_positioning_figure(preferred: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(12.4, 5.6))
    axis.axis("off")
    families = (
        (0.03, "COP family", "Useful heat / electricity\nNative objective: maximize", "#EAF2F8"),
        (
            0.35,
            "Heating-loss family",
            "Signed loss / healthy heat\nNative objective: minimize",
            "#FCE8E6",
        ),
        (
            0.67,
            "Evaporator family",
            "Water heat − compressor work\nNative objective: maximize",
            "#E7F4F1",
        ),
    )
    for left, title, body, color in families:
        axis.add_patch(
            plt.Rectangle((left, 0.42), 0.28, 0.38, facecolor=color, edgecolor="#4D4D4D")
        )
        axis.text(left + 0.02, 0.73, title, weight="bold", fontsize=11, va="top")
        axis.text(left + 0.02, 0.62, body, fontsize=9, va="top", linespacing=1.5)
    axis.annotate(
        "Same V2.6.8 candidate grid · raw water-side integration · cross-fitted event outcomes",
        xy=(0.5, 0.31),
        ha="center",
        fontsize=10,
        weight="bold",
    )
    axis.text(
        0.5,
        0.06,
        (
            "Literature anchors: Tan et al. (2025) · Klingebiel et al. (2023) DRL · "
            "Klingebiel et al. (2025) optimality evaluation / experimental validation"
        ),
        ha="center",
        fontsize=8,
        color="#59636E",
    )
    axis.text(
        0.5,
        0.12,
        f"Hierarchical comparison result: {preferred}",
        ha="center",
        fontsize=12,
        color="#0F4D92" if preferred != "no_unique_winner" else "#B64342",
        weight="bold",
    )
    axis.set_title(
        "V2.7 transfers evaluation ideas without claiming strict literature reproduction",
        loc="left",
    )
    return figure


def _metric_formula_boundary_figure() -> plt.Figure:
    """Separate the common metric definitions from the current V2.7 estimators."""
    figure = plt.figure(figsize=(14.4, 9.4))
    grid = figure.add_gridspec(3, 1, height_ratios=[1.08, 1.05, 1.55], hspace=0.18)

    boundary = figure.add_subplot(grid[0])
    boundary.axis("off")
    segments = (
        (0.04, 0.24, "Leading recovery", "#DCEAF1"),
        (0.24, 0.59, r"Observed heating $H$", "#FDE9D6"),
        (0.59, 0.77, "Preparation + defrost", "#E8DDF0"),
        (0.77, 0.96, "Fixed-9 future recovery", "#DCEAF1"),
    )
    for left, right, label, color in segments:
        boundary.add_patch(
            plt.Rectangle((left, 0.49), right - left, 0.24, facecolor=color, edgecolor="#66717C")
        )
        boundary.text((left + right) / 2, 0.61, label, ha="center", va="center", fontsize=8.5)
    for x, label in (
        (0.04, "cycle start"),
        (0.24, "fixed-9 stable start"),
        (0.59, r"candidate $\tau$"),
        (0.77, "defrost end"),
        (0.96, "event end"),
    ):
        boundary.text(x, 0.43, label, ha="center", va="top", fontsize=7.2, color="#3F4851")
    boundary.text(0.04, 0.91, "a  Define the candidate cycle once", fontsize=11.5, weight="bold")
    boundary.text(
        0.04,
        0.80,
        r"$evt=$ preparation + defrost + fixed-9 future recovery",
        fontsize=9.5,
    )
    boundary.text(
        0.04,
        0.18,
        r"Actual: $Q(\tau)=Q_H(\tau)+\hat Q_{evt}(\tau)$   ·   "
        r"$E(\tau)=E_H(\tau)+\hat E_{evt}(\tau)$",
        fontsize=10.5,
        weight="bold",
    )
    boundary.text(
        0.04,
        0.04,
        r"Healthy counterfactual: $Q_0(\tau)=Q_{H,0}(\tau)+\hat Q_{evt,0}(\tau)$   ·   "
        r"present estimator: $\hat Q_{evt,0}=\hat{\dot Q}_{w,0}(\tau)"
        r"\hat{\Delta t}_{evt}(\tau)/60$",
        fontsize=9,
    )

    concepts = figure.add_subplot(grid[1])
    concepts.axis("off")
    concepts.text(
        0.04,
        0.94,
        "b  Three concepts, one direction: larger is better",
        fontsize=11.5,
        weight="bold",
    )
    cards = (
        (
            0.04,
            "System efficiency",
            r"$COP_{cyc}=Q/E$",
            "Heat delivered per unit electricity",
            "#EAF2F8",
        ),
        (
            0.36,
            "Heating-service retention",
            r"$\eta_H=Q/Q_0$",
            r"$\epsilon_{HL}=1-\eta_H$ reports the same result as loss",
            "#FCE8E6",
        ),
        (
            0.68,
            "Outdoor-side retention",
            r"$\eta_{out}=Q_{out}/Q_{out,0}$",
            r"$Q_{out}=Q-E_{comp}$ (signed, inferred by energy balance)",
            "#E7F4F1",
        ),
    )
    for left, title, formula, meaning, color in cards:
        concepts.add_patch(
            plt.Rectangle((left, 0.16), 0.28, 0.62, facecolor=color, edgecolor="#AAB2BA")
        )
        concepts.text(left + 0.018, 0.69, title, fontsize=9.5, weight="bold", va="top")
        concepts.text(left + 0.14, 0.49, formula, fontsize=15, ha="center", va="center")
        concepts.text(left + 0.018, 0.27, meaning, fontsize=7.8, va="center")

    mapping = figure.add_subplot(grid[2])
    mapping.axis("off")
    mapping.text(
        0.04,
        0.95,
        "c  Current V2.7 fields are estimators or sensitivity analyses—"
        "not six new physical concepts",
        fontsize=11.5,
        weight="bold",
    )
    rows = (
        (
            "eta_e_cyc",
            "MAXIMIZE",
            r"$\eta_{out}$ estimator",
            "dynamic healthy reference",
            "full H + evt",
            "main outdoor-side view",
        ),
        (
            "cop_e",
            "MAXIMIZE",
            r"$Q_{out}/E$",
            "no healthy denominator",
            "full H + evt",
            "supplementary efficiency view",
        ),
        (
            "epsilon_hl",
            "MINIMIZE",
            r"$1-\eta_H$ estimator",
            r"dynamic healthy reference + independent $\hat L_{evt}$",
            "full H + evt",
            "definition is not yet algebraically constrained",
        ),
        (
            "epsilon_hl_t0_proxy",
            "MINIMIZE",
            r"$1-\eta_H$ sensitivity",
            "earliest stable 120-s proxy",
            "full H + evt",
            "baseline-window sensitivity",
        ),
        (
            "cop_cyc_k",
            "MAXIMIZE",
            r"$COP_{cyc}$ boundary sensitivity",
            "no healthy denominator",
            r"leading R + H + prep/D",
            "only future recovery is excluded; leading recovery: included",
        ),
        (
            "epsilon_hl_2a",
            "MINIMIZE",
            r"$1-\eta_H$ sensitivity",
            "same-experiment 5%/35% anchors",
            "full H + evt",
            "anchor-domain sensitivity",
        ),
    )
    headers = (
        "CODE FIELD",
        "NATIVE DIRECTION",
        "CONCEPTUAL MAP",
        "CURRENT ESTIMATOR",
        "ACTUAL BOUNDARY",
        "ROLE / CAUTION",
    )
    x = (0.04, 0.16, 0.27, 0.43, 0.62, 0.77)
    widths = (0.12, 0.11, 0.16, 0.19, 0.15, 0.19)
    for left, width, header in zip(x, widths, headers, strict=True):
        mapping.add_patch(
            plt.Rectangle((left, 0.80), width, 0.10, facecolor="#374151", edgecolor="white")
        )
        mapping.text(
            left + 0.008,
            0.85,
            header,
            color="white",
            fontsize=7.2,
            weight="bold",
            va="center",
        )
    for row_index, row in enumerate(rows):
        y = 0.80 - (row_index + 1) * 0.115
        fill = "#F7F8FA" if row_index % 2 == 0 else "white"
        for left, width, value in zip(x, widths, row, strict=True):
            mapping.add_patch(
                plt.Rectangle(
                    (left, y),
                    width,
                    0.115,
                    facecolor=fill,
                    edgecolor="#D4D8DD",
                    linewidth=0.6,
                )
            )
            mapping.text(left + 0.008, y + 0.057, value, fontsize=7.2, va="center")
    mapping.text(
        0.04,
        0.035,
        r"Notation: $H$ starts at the fixed-9 stable boundary unless a row states otherwise; "
        "hats are experiment-LOEO predictions; subscript 0 denotes the healthy "
        "counterfactual. Existing CSV field names remain unchanged.",
        fontsize=7.8,
        color="#4B5563",
    )
    figure.suptitle(
        "V2.7 notation hierarchy: define the cycle, then choose the evaluation question",
        x=0.03,
        y=0.995,
        ha="left",
        fontsize=14,
        weight="bold",
    )
    figure.subplots_adjust(left=0.03, right=0.985, bottom=0.035, top=0.955)
    return figure


def _quality_overview_figure(summary: pd.DataFrame, preferred: str) -> plt.Figure:
    order = [metric for metric in V27_METRIC_STYLES if metric in set(summary["metric_id"])]
    aggregate = (
        summary.groupby("metric_id")
        .agg(
            identified=("identified", "mean"),
            interior=("interior", "mean"),
            endpoint=("endpoint", "mean"),
            support=("supported_fraction", "mean"),
            basin_1=("basin_1pct_width_minutes", "median"),
            basin_5=("basin_5pct_width_minutes", "median"),
            bootstrap=("bootstrap_basin_fraction", "median"),
        )
        .reindex(order)
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 7.2))
    x = np.arange(len(order))
    colors = [V27_METRIC_STYLES[name][0] for name in order]
    axes[0, 0].bar(x, aggregate["identified"], color=colors, alpha=0.85, label="Identified")
    axes[0, 0].plot(x, aggregate["support"], color="#272727", marker="o", label="Support")
    axes[0, 0].set_ylabel("Cycle fraction")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].bar(x - 0.18, aggregate["interior"], width=0.36, color=colors, label="Interior")
    axes[0, 1].bar(
        x + 0.18,
        aggregate["endpoint"],
        width=0.36,
        color="#CFCECE",
        label="Endpoint",
    )
    axes[0, 1].set_ylabel("Extreme fraction")
    axes[0, 1].legend(frameon=False, fontsize=7)
    axes[1, 0].bar(x - 0.18, aggregate["basin_1"], width=0.36, color=colors, label="1%")
    axes[1, 0].bar(
        x + 0.18,
        aggregate["basin_5"],
        width=0.36,
        color="#A8A8A8",
        label="5%",
    )
    axes[1, 0].set_ylabel("Connected basin width [min]")
    axes[1, 0].legend(frameon=False, fontsize=7)
    axes[1, 1].bar(x, aggregate["bootstrap"], color=colors)
    axes[1, 1].set_ylabel("Bootstrap extreme in original 5% basin")
    for axis in axes.ravel():
        axis.set_xticks(x)
        axis.set_xticklabels(order, rotation=30, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(f"Metric quality hierarchy · {preferred}", x=0.06, ha="left", weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def _support_heatmap_figure(summary: pd.DataFrame) -> plt.Figure:
    pivot = summary.pivot(index="metric_id", columns="cycle_name", values="supported_fraction")
    pivot = pivot.reindex([metric for metric in V27_METRIC_STYLES if metric in pivot.index])
    figure, axis = plt.subplots(figsize=(14.2, 3.5))
    image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axis.set_yticks(range(len(pivot.index)))
    axis.set_yticklabels(pivot.index, fontsize=7)
    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels(
        [str(int(str(name).rsplit("_", 1)[-1])) for name in pivot.columns],
        rotation=90,
        fontsize=5,
    )
    axis.set_xlabel("Cycle id")
    axis.set_title(
        "Applicability-domain coverage is shown, never used to truncate display curves", loc="left"
    )
    figure.colorbar(image, ax=axis, label="Supported candidate fraction")
    figure.tight_layout()
    return figure


def _normalized_gap_figure(metrics: Mapping[str, pd.DataFrame]) -> plt.Figure:
    cycles = sorted(set.intersection(*(set(table["cycle_name"]) for table in metrics.values())))
    informative = [
        cycle
        for cycle in cycles
        if all(
            pd.to_numeric(
                table.loc[table["cycle_name"].eq(cycle), "relative_optimality_gap"],
                errors="coerce",
            )
            .notna()
            .any()
            for table in metrics.values()
        )
    ]

    def extreme_minutes(cycle_name: str) -> float:
        values: list[float] = []
        for table in metrics.values():
            curve = table.loc[table["cycle_name"].eq(cycle_name)]
            gap = pd.to_numeric(curve["relative_optimality_gap"], errors="coerce")
            if curve.empty or gap.notna().sum() == 0:
                continue
            extreme = pd.to_datetime(
                curve.loc[gap.idxmin(), "candidate_time"], errors="coerce", format="mixed"
            )
            start = pd.to_datetime(
                curve["cycle_start"].iloc[0], errors="coerce", format="mixed"
            )
            if pd.notna(extreme) and pd.notna(start):
                values.append((extreme - start).total_seconds() / 60)
        return float(np.median(values)) if values else np.inf

    ordered = sorted(informative, key=lambda cycle: (extreme_minutes(cycle), cycle))
    positions = sorted(set((0, len(ordered) // 2, len(ordered) - 1))) if ordered else []
    selected = [ordered[position] for position in positions]
    figure, axes = plt.subplots(len(selected) or 1, 1, figsize=(12.8, 2.8 * max(len(selected), 1)))
    axes = np.atleast_1d(axes)
    for axis, cycle_name in zip(axes, selected, strict=False):
        for metric_id, table in metrics.items():
            curve = table.loc[table["cycle_name"].eq(cycle_name)].copy()
            if curve.empty:
                continue
            start = pd.Timestamp(curve["cycle_start"].iloc[0])
            minutes = (
                pd.to_datetime(curve["candidate_time"], errors="coerce", format="mixed") - start
            ).dt.total_seconds() / 60
            color, _marker, label = V27_METRIC_STYLES[metric_id]
            axis.plot(
                minutes,
                100 * pd.to_numeric(curve["relative_optimality_gap"], errors="coerce"),
                color=color,
                linewidth=1.1,
                label=label,
            )
        axis.axhline(1, color="#767676", linestyle=":", linewidth=0.8)
        axis.axhline(5, color="#A8A8A8", linestyle=":", linewidth=0.8)
        axis.set_ylabel("Optimality gap [%]")
        axis.set_title(str(cycle_name), loc="left", fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Time from cycle start [min]")
    axes[0].legend(frameon=False, fontsize=7, ncols=3)
    figure.tight_layout()
    return figure


def _v27_validation_model_rows(validation: pd.DataFrame) -> pd.DataFrame:
    """Return only valid four-model rows used by the V2.7 LOEO figure."""
    if validation.empty:
        return validation.copy()
    values = validation.copy()
    if "model_name" not in values.columns:
        # This is only a last-resort display path for an old wide export.  The
        # normal entry point first joins the long V2.6.8 validation CSV.
        values["model_name"] = "dynamic_8"
    values = values.loc[values["model_name"].isin(V27_VALIDATION_MODELS)].copy()
    if "event_valid" in values.columns:
        values = values.loc[values["event_valid"].fillna(False)].copy()
    return values


def _v27_validation_summary(
    values: pd.DataFrame, observed: str, predicted: str
) -> dict[str, float]:
    """Compute one target's macro/micro errors and calibration slope."""
    if observed not in values or predicted not in values:
        return {"n": 0, "macro_mse": np.nan, "macro_mae": np.nan, "micro_mse": np.nan,
                "micro_mae": np.nan, "r2": np.nan, "calibration_slope": np.nan}
    columns = [observed, predicted]
    if "experiment_id" in values:
        columns.insert(0, "experiment_id")
    frame = values[columns].copy()
    frame[observed] = pd.to_numeric(frame[observed], errors="coerce")
    frame[predicted] = pd.to_numeric(frame[predicted], errors="coerce")
    frame = frame.dropna(subset=[observed, predicted])
    if frame.empty:
        return {"n": 0, "macro_mse": np.nan, "macro_mae": np.nan, "micro_mse": np.nan,
                "micro_mae": np.nan, "r2": np.nan, "calibration_slope": np.nan}
    residual = frame[observed] - frame[predicted]
    squared = residual.pow(2)
    absolute = residual.abs()
    groups = (
        frame["experiment_id"]
        if "experiment_id" in frame
        else pd.Series("all", index=frame.index)
    )
    grouped_errors = pd.DataFrame({"error": squared, "group": groups}).groupby("group")
    macro_mse = float(grouped_errors["error"].mean().mean())
    grouped_absolute = pd.DataFrame({"error": absolute, "group": groups}).groupby("group")
    macro_mae = float(grouped_absolute["error"].mean().mean())
    denominator = float(np.square(frame[observed] - frame[observed].mean()).sum())
    slope = (
        float(np.polyfit(frame[observed], frame[predicted], 1)[0])
        if frame[observed].nunique() > 1
        else np.nan
    )
    return {
        "n": int(len(frame)),
        "macro_mse": macro_mse,
        "macro_mae": macro_mae,
        "micro_mse": float(squared.mean()),
        "micro_mae": float(absolute.mean()),
        "r2": 1 - float(squared.sum()) / denominator if denominator > 0 else np.nan,
        "calibration_slope": slope,
    }


def _v27_parity_axis(
    axis: plt.Axes,
    values: pd.DataFrame,
    observed: str,
    predicted: str,
    xlabel: str,
    ylabel: str,
    title: str,
    *,
    calibration: bool = False,
) -> dict[str, dict[str, float]]:
    """Draw a four-model parity panel, optionally with calibration fits."""
    summaries: dict[str, dict[str, float]] = {}
    if observed not in values or predicted not in values:
        axis.text(0.5, 0.5, "Validation columns unavailable", ha="center", va="center")
        axis.set_title(title, loc="left")
        axis.set_axis_off()
        return summaries
    bounds = pd.concat(
        [pd.to_numeric(values[observed], errors="coerce"),
         pd.to_numeric(values[predicted], errors="coerce")]
    ).dropna()
    if bounds.empty:
        axis.text(0.5, 0.5, "No finite LOEO rows", ha="center", va="center")
        axis.set_title(title, loc="left")
        axis.set_axis_off()
        return summaries
    lower, upper = float(bounds.min()), float(bounds.max())
    pad = max((upper - lower) * 0.06, 1e-4)
    axis.plot(
        [lower - pad, upper + pad],
        [lower - pad, upper + pad],
        color="#767676",
        linestyle=":",
        linewidth=0.8,
        label="1:1",
    )
    for model_name in V27_VALIDATION_MODELS:
        model_values = values.loc[values["model_name"].eq(model_name)]
        summary = _v27_validation_summary(model_values, observed, predicted)
        summaries[model_name] = summary
        if summary["n"] == 0:
            continue
        x_values = pd.to_numeric(model_values[observed], errors="coerce")
        y_values = pd.to_numeric(model_values[predicted], errors="coerce")
        valid = x_values.notna() & y_values.notna()
        axis.scatter(
            x_values.loc[valid],
            y_values.loc[valid],
            s=18,
            alpha=0.78,
            color=V27_VALIDATION_MODEL_COLORS[model_name],
            marker=V27_VALIDATION_MODEL_MARKERS[model_name],
            label=V27_VALIDATION_MODEL_LABELS[model_name],
        )
        if calibration and summary["n"] >= 2 and np.isfinite(summary["calibration_slope"]):
            coefficient = np.polyfit(x_values.loc[valid], y_values.loc[valid], 1)
            fit_x = np.array([lower - pad, upper + pad])
            axis.plot(
                fit_x,
                coefficient[0] * fit_x + coefficient[1],
                color=V27_VALIDATION_MODEL_COLORS[model_name],
                linewidth=0.85,
                alpha=0.9,
            )
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.set_xlim(lower - pad, upper + pad)
    axis.set_ylim(lower - pad, upper + pad)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#D8D8D8", linewidth=0.4)
    if calibration:
        slopes = [
            f"{V27_VALIDATION_MODEL_LABELS[name]}: {summary['calibration_slope']:.2f}"
            for name, summary in summaries.items()
            if np.isfinite(summary["calibration_slope"])
        ]
        if slopes:
            axis.text(
                0.03,
                0.97,
                "Calibration slope (prediction ~ target)\n" + "\n".join(slopes),
                transform=axis.transAxes,
                va="top",
                fontsize=6.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
            )
    return summaries


def _v27_dynamic_loss_rows(values: pd.DataFrame) -> pd.DataFrame:
    """Select one copy of the healthy-reference-derived dynamic loss target."""
    required = {"L_T_dynamic_observed_kwh", "L_T_dynamic_prediction_kwh"}
    if not required <= set(values.columns):
        return pd.DataFrame()
    selected = values.loc[values["model_name"].eq("dynamic_8")].copy()
    if selected.empty:
        selected = values.copy()
    if "cycle_name" in selected.columns:
        selected = selected.drop_duplicates("cycle_name")
    return selected


def _validation_figure(validation: pd.DataFrame) -> plt.Figure:
    """Render the complete V2.7 event-outcome LOEO evidence chain from CSV rows."""
    values = _v27_validation_model_rows(validation)
    figure = plt.figure(figsize=(13.4, 10.2))
    grid = figure.add_gridspec(3, 3, height_ratios=[3.1, 1.6, 2.2], hspace=0.45)
    target_specs = (
        ("E_T_observed_kwh", "E_T_prediction_kwh", r"Observed $E_T$ [kWh]",
         r"LOEO prediction $E_T$ [kWh]", r"$E_T$ event outcome LOEO", False),
        ("Q_T_observed_kwh", "Q_T_prediction_kwh", r"Observed $Q_T$ [kWh]",
         r"LOEO prediction $Q_T$ [kWh]", r"$Q_T$ event outcome LOEO", False),
        ("J_w_observed", "J_w_prediction", r"Observed $J_w$",
         r"LOEO prediction $J_w$", r"$J_w$ calibration", True),
    )
    scatter_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for axis, (observed, predicted, xlabel, ylabel, title, calibration) in zip(
        scatter_axes, target_specs, strict=True
    ):
        summaries[observed] = _v27_parity_axis(
            axis,
            values,
            observed,
            predicted,
            xlabel,
            ylabel,
            title,
            calibration=calibration,
        )
    if not values.empty:
        scatter_axes[0].legend(fontsize=6.5, loc="upper left")

    metric_labels = ("Macro MSE", "Macro MAE", "Micro MSE", "Micro MAE", "$R^2$", "Cal. slope")
    metric_axes = [figure.add_subplot(grid[1, index]) for index in range(3)]
    for axis, (observed, _predicted, _xlabel, _ylabel, title, _calibration) in zip(
        metric_axes, target_specs, strict=True
    ):
        axis.set_axis_off()
        table_values = summaries[observed]
        table = axis.table(
            cellText=[
                [
                    f"{table_values.get(model, {}).get(key, np.nan):.3g}"
                    for key in ("macro_mse", "macro_mae", "micro_mse", "micro_mae", "r2",
                                "calibration_slope")
                ]
                for model in V27_VALIDATION_MODELS
            ],
            rowLabels=[V27_VALIDATION_MODEL_LABELS[model] for model in V27_VALIDATION_MODELS],
            colLabels=metric_labels,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(5.8)
        table.scale(1.0, 1.32)
        axis.set_title(f"{title} · complete experiment-LOEO metrics", fontsize=7.5, pad=4)

    dynamic_axis = figure.add_subplot(grid[2, :2])
    dynamic_values = _v27_dynamic_loss_rows(values)
    _v27_parity_axis(
        dynamic_axis,
        dynamic_values,
        "L_T_dynamic_observed_kwh",
        "L_T_dynamic_prediction_kwh",
        r"Cross-fitted healthy-reference-derived target $L_T$ [kWh]",
        r"LOEO prediction $L_T$ [kWh]",
        r"Dynamic $L_T$ — cross-fitted healthy-reference-derived target",
    )
    if not dynamic_values.empty:
        dynamic_axis.text(
            0.03,
            0.03,
            (
                "Target is derived from the cross-fitted healthy reference; "
                "it is not an independent measurement."
            ),
            transform=dynamic_axis.transAxes,
            fontsize=6.5,
            color="#59636E",
        )
    note_axis = figure.add_subplot(grid[2, 2])
    note_axis.set_axis_off()
    note_axis.text(
        0.02,
        0.96,
        "V2.7 validation contract",
        va="top",
        weight="bold",
        fontsize=9,
    )
    note_axis.text(
        0.02,
        0.82,
        (
            "Four models use the same held-out experiment split:\n"
            "Mean baseline · Static-5 · Physical-static-6 · Dynamic-8\n\n"
            "$J_w$ points and fitted lines show calibration on the actual event rows.\n\n"
            "Dynamic $L_T$ is a healthy-reference-derived target, not an independent "
            "observed target."
        ),
        va="top",
        fontsize=7.2,
        linespacing=1.45,
    )
    figure.suptitle(
        r"V2.7 event-outcome LOEO: $E_T$, $Q_T$, $J_w$ and calibration",
        x=0.06,
        ha="left",
        fontsize=11,
        weight="bold",
    )
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.07, top=0.93)
    return figure


def _v27_optimal_rows(metrics: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for metric_id, table in metrics.items():
        for _, curve in table.groupby("cycle_name", sort=False):
            target = pd.to_datetime(
                curve["t_star"].iloc[0], errors="coerce", format="mixed"
            )
            if pd.isna(target):
                continue
            index = (
                pd.to_datetime(curve["candidate_time"], errors="coerce", format="mixed")
                - target
            ).abs().idxmin()
            row = curve.loc[index].copy()
            row["metric_id"] = metric_id
            rows.append(row)
    return pd.DataFrame(rows)


def _optimal_physical_state_figure(metrics: Mapping[str, pd.DataFrame]) -> plt.Figure:
    values = _v27_optimal_rows(metrics)
    figure, axes = plt.subplots(3, 1, figsize=(14.2, 8.2), sharex=True)
    if values.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No identified diagnostic extreme", ha="center")
        return figure
    values["cycle_id"] = values["cycle_name"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    for column in (
        "heating_attenuation_fraction",
        "instant_water_cop",
        "instant_unit_cop",
        "instant_evaporator_capacity_kw",
    ):
        if column not in values:
            values[column] = np.nan
    for metric_id, group in values.groupby("metric_id", sort=False):
        group = group.sort_values("cycle_id")
        color, marker, label = V27_METRIC_STYLES[metric_id]
        axes[0].plot(
            group["cycle_id"],
            100 * pd.to_numeric(group["heating_attenuation_fraction"], errors="coerce"),
            color=color,
            marker=marker,
            markersize=3,
            linewidth=0.7,
            label=label,
        )
        axes[1].scatter(
            group["cycle_id"],
            pd.to_numeric(group["instant_water_cop"], errors="coerce"),
            color=color,
            marker=marker,
            s=15,
        )
        axes[1].plot(
            group["cycle_id"],
            pd.to_numeric(group["instant_unit_cop"], errors="coerce"),
            color=color,
            linewidth=0.65,
            alpha=0.45,
        )
        axes[2].plot(
            group["cycle_id"],
            pd.to_numeric(group["instant_evaporator_capacity_kw"], errors="coerce"),
            color=color,
            marker=marker,
            markersize=3,
            linewidth=0.7,
        )
    axes[0].axhspan(5, 20, color="#E28E2C", alpha=0.12, label="5–20% loss diagnostic")
    axes[0].set_ylabel("Water-heat attenuation [%]")
    axes[1].set_ylabel("Instantaneous COP [-]\npoints: water; lines: unit")
    axes[2].set_ylabel("Evaporator capacity [kW]")
    axes[2].set_xlabel("Cycle id")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=6.5, ncols=4)
    figure.suptitle("Physical state at each metric's diagnostic extreme", x=0.06, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def _baseline_boundary_sensitivity_figure(
    metrics: Mapping[str, pd.DataFrame], historical: Mapping[str, pd.DataFrame]
) -> plt.Figure:
    series: dict[str, pd.DataFrame] = {}
    for metric_id in ("epsilon_hl", "epsilon_hl_t0_proxy", "cop_cyc_k"):
        if metric_id in metrics:
            series[metric_id] = _cycle_points(metrics[metric_id])
    if "v2.6.8" in historical:
        series["v2.6.8"] = _cycle_points(historical["v2.6.8"])
    cycles = sorted(set().union(*(set(frame.index) for frame in series.values())))
    figure, axis = plt.subplots(figsize=(14.0, 4.6))
    x = np.arange(len(cycles))
    for name, frame in series.items():
        aligned = frame.reindex(cycles)
        if name in V27_METRIC_STYLES:
            color, marker, label = V27_METRIC_STYLES[name]
            label = V27_BOUNDARY_LABELS.get(name, label)
        else:
            color, marker, label = (*_style(name)[:2], name.upper())
            if name == "v2.6.8":
                label = "V2.6.8 J_model · fixed-9 stable-to-stable"
        axis.plot(
            x, aligned["optimum_minutes"], color=color, marker=marker, ms=3, lw=0.8, label=label
        )
    axis.set_xticks(x)
    axis.set_xticklabels([str(int(str(cycle).rsplit("_", 1)[-1])) for cycle in cycles], fontsize=6)
    axis.set(xlabel="Cycle id", ylabel="Time from cycle start [min]")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, fontsize=7, ncols=4)
    ts_columns = {
        "ts_diagnostic_minimum",
        "J_ts_model",
    }
    ts_available = "v2.6.8" in historical and ts_columns <= set(historical["v2.6.8"])
    if ts_available:
        ts_note = "Ts-dependent 9/13 min: not plotted in this panel (available in V2.6.8 CSV)"
    else:
        ts_note = "Ts-dependent 9/13 min: not available in the plotted/source CSV"
    if "cop_cyc_k" in metrics:
        rr_note = (
            "COP_cyc,K: leading recovery + heating + prep/D; only future recovery excluded"
        )
    elif "v2.6.8" in historical and "rr_diagnostic_minimum" in historical["v2.6.8"]:
        rr_note = (
            "recovery-start-to-recovery-start: not plotted in this panel "
            "(available in V2.6.8 CSV)"
        )
    else:
        rr_note = "recovery-start-to-recovery-start: not available in the plotted/source CSV"
    figure.text(
        0.01,
        0.015,
        "Boundary protocol audit · " + ts_note + " · " + rr_note,
        ha="left",
        va="bottom",
        fontsize=7,
        color="#59636E",
    )
    axis.set_title("Healthy-reference and recovery-boundary sensitivity", loc="left")
    figure.tight_layout(rect=(0, 0.065, 1, 1))
    return figure


def _identifiability_figure(identifiability: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(12.8, 5.2))
    axis.axis("off")
    if identifiability.empty:
        axis.text(0.5, 0.5, "Identifiability audit CSV unavailable", ha="center")
        return figure
    strict = identifiability.loc[identifiability["audit_id"].ne("project_two_anchor")]
    anchor = identifiability.loc[identifiability["audit_id"].eq("project_two_anchor")]
    y = 0.82
    for _, row in strict.iterrows():
        axis.text(0.04, y, str(row["audit_id"]), weight="bold", fontsize=9)
        axis.text(0.28, y, str(row.get("reason", "")), fontsize=8)
        y -= 0.11
    available = int(anchor.get("available", pd.Series(dtype=bool)).fillna(False).sum())
    axis.text(
        0.04,
        0.29,
        f"Project two-anchor identifiable cycles: {available}/{len(anchor)}",
        fontsize=10,
        weight="bold",
    )
    axis.text(
        0.04,
        0.18,
        (
            "Unavailable strict metrics remain audit rows; "
            "they are not silently registered as algorithms."
        ),
        fontsize=9,
        color="#B64342",
    )
    axis.set_title("Strict-reproduction missing conditions and project identifiability", loc="left")
    return figure


def _historical_experience_figure() -> plt.Figure:
    figure, axis = plt.subplots(figsize=(13.2, 6.0))
    axis.axis("off")
    columns = (
        (0.02, "V1", "Simple cycle COP\nStable reference for comparison"),
        (0.21, "V2.5", "Full current-cycle ratio\nKnown boundary/model sensitivity"),
        (0.40, "V2.6.7", "Observational ticket outcome\nSupport-aware diagnostic"),
        (0.59, "V2.6.8", "Complete prep–D–R outcome\nFixed 9-min stable boundary"),
        (0.78, "V2.7", "Same boundary, new indicators\nCOP · loss · evaporator ability"),
    )
    for left, title, body in columns:
        axis.add_patch(
            plt.Rectangle((left, 0.42), 0.17, 0.28, facecolor="#F5F7FA", edgecolor="#4D4D4D")
        )
        axis.text(left + 0.015, 0.64, title, weight="bold", fontsize=11)
        axis.text(left + 0.015, 0.57, body, fontsize=8, va="top", linespacing=1.45)
        if left < 0.78:
            axis.annotate(
                "",
                xy=(left + 0.19, 0.56),
                xytext=(left + 0.17, 0.56),
                arrowprops={"arrowstyle": "->"},
            )
    axis.text(
        0.5,
        0.23,
        (
            "Accumulated lesson: boundary completeness → signed energy → "
            "cross-fitting/support → metric discrimination"
        ),
        ha="center",
        fontsize=10,
        weight="bold",
    )
    axis.text(
        0.5,
        0.12,
        "Timing shift is diagnostic evidence, not the optimization target",
        ha="center",
        fontsize=10,
        color="#B64342",
    )
    axis.set_title(
        "V2.7 extends the historical cost-function line; it does not restart it", loc="left"
    )
    return figure


def _write_v27_diagnostic_figures(
    metrics: Mapping[str, pd.DataFrame],
    historical: Mapping[str, pd.DataFrame],
    output: Path,
    *,
    validation: pd.DataFrame,
    identifiability: pd.DataFrame,
) -> None:
    summary = _v27_summary(metrics)
    preferred = _preferred_metric(summary)
    root = _diagnostic_root(output)
    figures = {
        "00_六指标公式方向与循环边界.png": _metric_formula_boundary_figure(),
        "01_评价指标迁移与文献定位.png": _metric_positioning_figure(preferred),
        "02_完整性极值位置区分度与稳定性.png": _quality_overview_figure(summary, preferred),
        "03_支持域与可识别覆盖.png": _support_heatmap_figure(summary),
        "04_方向感知成本形状比较.png": _normalized_gap_figure(metrics),
        "05_新增事件目标LOEO.png": _validation_figure(validation),
        "06_全历史成本函数经验链.png": _historical_experience_figure(),
        "07_最优点物理状态.png": _optimal_physical_state_figure(metrics),
        "08_健康基准与recovery边界敏感性.png": _baseline_boundary_sensitivity_figure(
            metrics, historical
        ),
        "09_严格复现缺失条件与双锚点可识别性.png": _identifiability_figure(identifiability),
    }
    for name, figure in figures.items():
        _save_svg_png(figure, root / name)


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
        if algorithm in STATUS_MARKERS:
            markers = STATUS_MARKERS[algorithm]
            no_minimum = values["optimum_minutes"].isna()
            unknown = set(values["cycle_status"].dropna()) - set(markers)
            if unknown or values["cycle_status"].isna().any():
                raise ValueError(
                    f"unrecognized {algorithm.upper()} cycle_status: {sorted(unknown)}"
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
            if algorithm in {"v2.6.7", "v2.6.8"} and no_minimum.any():
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
        supported = values["optimum_supported"].eq(True)
        unsupported = values["optimum_supported"].eq(False)
        unknown = values["optimum_supported"].isna()
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
        if algorithm in V27_METRIC_STYLES and values["optimum_minutes"].isna().any():
            no_minimum = values["optimum_minutes"].isna()
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


def _with_v268_display_extension(curve: pd.DataFrame) -> pd.DataFrame:
    result = curve.copy()
    model = pd.to_numeric(result["J_model"], errors="coerce")
    excluded = ~result["optimization_eligible"].fillna(False)
    result[V268_DISPLAY_METRIC] = model.where(excluded & np.isfinite(model))
    return result


def _cost_curve_figure(  # noqa: C901
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
        v27_metric = algorithm in V27_METRIC_STYLES
        metric_column = (
            "objective_value"
            if v27_metric
            else "J_model"
            if algorithm == "v2.6.8"
            else "inverse_cop"
        )
        raw_cost = pd.to_numeric(curve[metric_column], errors="coerce")
        cost = raw_cost.where(eligible)
        regret_column = "relative_optimality_gap" if v27_metric else "relative_regret"
        regret = (100 * pd.to_numeric(curve[regret_column], errors="coerce")).where(eligible)
        color, marker, label = _style(algorithm)
        linestyle = CURVE_LINESTYLES.get(algorithm, CURVE_LINESTYLES["v2.6"])
        cost_axis.plot(
            minutes,
            cost,
            color=color,
            ls=linestyle,
            lw=1.2,
            label=label if v27_metric else algorithm.upper(),
        )
        if algorithm == "v2.6.8":
            outside = ~curve["supported"].fillna(False) & raw_cost.notna()
            cost_axis.plot(
                minutes,
                raw_cost,
                color=color,
                ls="--",
                lw=0.75,
                alpha=0.35,
                label="_nolegend_",
            )
            cost_axis.scatter(
                minutes.loc[outside],
                raw_cost.loc[outside],
                color=color,
                marker="x",
                s=18,
                linewidths=0.8,
                label="V2.6.8 outside applicability domain",
                zorder=3,
            )
        elif v27_metric:
            display = pd.to_numeric(
                curve.get("display_only_objective", raw_cost), errors="coerce"
            ).where(~eligible)
            if display.notna().any():
                cost_axis.plot(
                    minutes,
                    display,
                    color=color,
                    ls="--",
                    lw=0.75,
                    marker=".",
                    ms=2.5,
                    alpha=0.45,
                    label=f"{label} outside formal support, display only",
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
        elif algorithm in {"v2.6.7", "v2.6.8"} or v27_metric or cost.dropna().empty:
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
    regret_axis.axhline(1, color="#767676", ls=":", lw=0.8, label="1% basin")
    regret_axis.axhline(5, color="#A8A8A8", ls="--", lw=0.8, label="5% basin")
    regret_axis.set_ylim(-0.15, 5.5)
    objective_label = (
        "Metric objective"
        if any(name in V27_METRIC_STYLES for name in tables)
        else "Cost J = 1/COP"
    )
    cost_axis.set_ylabel(
        objective_label
    )
    regret_axis.set(
        xlabel="Minutes from cycle start",
        ylabel="Relative cost [%]",
    )
    cycle_id = int(cycle_name.rsplit("_", 1)[-1])
    status_algorithm = next(
        (algorithm for algorithm in ("v2.6.8", "v2.6.7", "v2.6.6") if algorithm in tables),
        None,
    )
    status_curve = tables.get(status_algorithm, reference)
    status_row = status_curve.loc[status_curve["cycle_name"].eq(cycle_name)].iloc[0]
    status = (
        f" · {status_algorithm.upper()} {str(status_row['cycle_status']).replace('_', ' ')}"
        if status_algorithm and "cycle_status" in status_row.index
        else ""
    )
    cost_axis.set_title(f"Cycle {cycle_id}: cost-function variants{status}", loc="left")
    cost_axis.legend(frameon=False, ncols=4, fontsize=7)
    figure.tight_layout()
    return figure


def _optimal_rgb_figures(
    front_images: Mapping[str, Mapping[str, object]],
    algorithms: tuple[str, ...],
    cycle_name: str,
    start: pd.Timestamp,
    stable: pd.Timestamp | None = pd.NaT,
):
    """Yield readable paginated front-image plates for one cycle."""
    page_size = (
        6
        if any(name in V27_METRIC_STYLES for name in algorithms)
        else 6
        if len(algorithms) == 5
        else 4
    )
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
            info = front_images.get(algorithm, {})
            display_name = str(info.get("display_label", algorithm.upper()))
            _plot_decision_image(axis, info, display_name, start, stable)
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
                            target_status
                            if algorithm in STATUS_MARKERS or algorithm in V27_METRIC_STYLES
                            else "",
                            (
                                "no eligible diagnostic minimum"
                                if (
                                    algorithm in {"v2.6.7", "v2.6.8"}
                                    or algorithm in V27_METRIC_STYLES
                                )
                                and info.get("status") == "no_valid_optimal"
                                else str(info.get("status", "unavailable")).replace("_", " ")
                            ),
                        ),
                    )
                )
            )
            diagnostic = (
                algorithm in {"v2.6.6", "v2.6.7", "v2.6.8"}
                or algorithm in V27_METRIC_STYLES
            )
            direction = str(info.get("optimization_direction", "min"))
            label = (
                f"{display_name} diagnostic {'maximum' if direction == 'max' else 'minimum'}"
                if algorithm in V27_METRIC_STYLES
                else f"{algorithm.upper()} diagnostic minimum"
                if diagnostic
                else f"{algorithm.upper()} optimum"
            )
            axis.set_title(
                f"{label}\n{detail}",
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
            if {"v2.6.6", "v2.6.7"}.intersection(page)
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
            and (algorithm not in STATUS_MARKERS or cycle_status == "identified_curve")
        )
        matched[algorithm]["target_status"] = (
            cycle_status
            if algorithm in STATUS_MARKERS and cycle_status != "identified_curve"
            else ""
        )
        if algorithm in V27_METRIC_STYLES:
            matched[algorithm]["display_label"] = str(
                curve.get("objective_label", pd.Series([algorithm])).iloc[0]
            )
            matched[algorithm]["optimization_direction"] = str(
                curve.get("optimization_direction", pd.Series(["min"])).iloc[0]
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
    v27_algorithms = tuple(name for name in V27_METRIC_STYLES if name in tables)
    historical_algorithms = tuple(
        name for name in (*STYLES, *V26_PATCHES) if name != "RB" and name in tables
    )
    algorithms = v27_algorithms + historical_algorithms if v27_algorithms else historical_algorithms
    selected = {algorithm: tables[algorithm] for algorithm in algorithms}
    cycle_sets = {
        algorithm: set(table["cycle_name"].astype(str)) for algorithm, table in selected.items()
    }
    reference_cycles = next(iter(cycle_sets.values()))
    if any(cycles != reference_cycles for cycles in cycle_sets.values()):
        raise ValueError("cost-curve families must contain identical cycle sets")
    cycles = reference_cycles
    reference = next(iter(selected.values()))
    v27 = bool(v27_algorithms)
    for cycle_name in sorted(cycles):
        cycle_id = int(cycle_name.rsplit("_", 1)[-1])
        start = pd.Timestamp(
            reference.loc[reference["cycle_name"].eq(cycle_name), "cycle_start"].iloc[0]
        )
        stable = pd.NaT
        if v27:
            metric_curve = selected[v27_algorithms[0]].loc[
                selected[v27_algorithms[0]]["cycle_name"].eq(cycle_name)
            ]
            stable_values = pd.to_datetime(
                metric_curve["stable_start_fixed9"]
                if "stable_start_fixed9" in metric_curve
                else pd.Series(dtype="datetime64[ns]"),
                errors="coerce",
                format="mixed",
            ).dropna()
            if not stable_values.empty:
                stable = pd.Timestamp(stable_values.iloc[0])
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
            saver = _save_svg_png if v27 else _save_png
            saver(
                _cost_curve_figure(selected, cycle_name),
                output / f"cycle_{cycle_id:03d}_cost_curves.png",
            )
            for page, figure in enumerate(
                _optimal_rgb_figures(
                    front_images,
                    algorithms,
                    cycle_name,
                    start,
                    stable,
                ),
                start=1,
            ):
                saver(
                    figure,
                    output / "optimal_rgb" / f"cycle_{cycle_id:03d}_optimal_rgb_{page:02d}.png",
                )


def _decision_images(
    metadata: pd.DataFrame,
    images: pd.DataFrame,
    curve: pd.DataFrame,
    fallback_to_extreme: bool = True,
) -> dict[str, dict[str, object]]:
    eligible = curve["optimization_eligible"].fillna(False)
    first = curve.iloc[0]
    optimum = (
        first.get("recommended_time") if first.get("algorithm") == "v3" else first.get("t_star")
    )
    if pd.isna(optimum) and fallback_to_extreme:
        metric = "objective_value" if "objective_value" in curve else "inverse_cop"
        formal = pd.to_numeric(curve[metric], errors="coerce").where(eligible).dropna()
        direction = str(first.get("optimization_direction", "min"))
        if formal.empty:
            optimum = pd.NaT
        else:
            extreme = formal.idxmax() if direction == "max" else formal.idxmin()
            optimum = curve.loc[extreme, "candidate_time"]
    rb = first["t_RB"] if first.get("rb_status") == "triggered" else pd.NaT
    matches = match_decision_rgb_images(
        metadata,
        images,
        {"rb": rb, "optimal": optimum},
    )
    return {str(row["target_type"]): row.to_dict() for _, row in matches.iterrows()}


def _render_parallel_policy_cycle(
    cycle_name: str,
    metrics: Mapping[str, pd.DataFrame],
    loader: DatasetLoader,
    record: Mapping[str, object],
    output: Path,
    *,
    guardrail: float = 0.05,
    pareto_selector: str = "knee",
    fetch_cloud: bool = False,
    minimum_free_gib: float = 5,
) -> pd.DataFrame:
    """Render one shared C/H/O time-and-Pareto publication."""
    metric_ids = {"C": "cop_cyc_evt", "H": "h_abs_rate", "O": "o_abs_rate"}
    frame = loader.load_cycle(cycle_name)
    origin = pd.to_datetime(frame["timestamp"], errors="coerce").min()
    curves = {
        label: metrics[metric_id].loc[
            metrics[metric_id]["cycle_name"].astype(str).eq(cycle_name)
        ]
        for label, metric_id in metric_ids.items()
    }
    boundaries = record.get("boundaries")
    boundaries = boundaries if isinstance(boundaries, Mapping) else record
    stable_start = pd.to_datetime(boundaries.get("stable_heating_start"), errors="coerce")
    if pd.notna(stable_start):
        curves = {
            label: curve.loc[
                pd.to_datetime(curve["candidate_time"], errors="coerce").ge(stable_start)
            ].copy()
            for label, curve in curves.items()
        }
    candidates = ch_pareto_table(curves, pd.Timestamp(origin), guardrail=guardrail)
    candidates.insert(0, "cycle_name", cycle_name)
    selected = candidates.loc[
        candidates[f"pareto_{pareto_selector}"], "candidate_time"
    ]
    policy_curve = curves["C"].copy()
    policy_curve["algorithm"] = f"pareto_{pareto_selector}"
    policy_curve["t_star"] = selected.iloc[0] if not selected.empty else pd.NaT
    metadata = loader.load_image_metadata(cycle_name)
    images = loader.load_cycle_images(cycle_name)
    decisions = _decision_images(metadata, images, policy_curve, not selected.empty)
    cycle_id = int(cycle_name.rsplit("_", 1)[-1])
    missing = sorted(
        {
            str(info["file_name"])
            for info in decisions.values()
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
                loader.dataset_root, cycle_name, metadata, cycle_dir=cycle_dir
            )
            images = downloaded if images.empty else pd.concat(
                [images, downloaded], ignore_index=True
            )
            decisions = _decision_images(metadata, images, policy_curve, not selected.empty)
        render_decision_publication(
            frame,
            record,
            policy_curve,
            decisions,
            output / "cycles" / f"cycle_{cycle_id:03d}_publication.png",
            optimal_label=f"Pareto-{pareto_selector}",
            full_candidate_domain=True,
            parallel_curves=curves,
            pareto_guardrail=guardrail,
            pareto_selector=pareto_selector,
        )
    return candidates


def _render_parallel_policy_cycles(
    metrics: Mapping[str, pd.DataFrame],
    loader: DatasetLoader,
    records: Mapping[str, Mapping[str, object]],
    output: Path,
    *,
    guardrail: float = 0.05,
    pareto_selector: str = "knee",
    fetch_cloud: bool = False,
    minimum_free_gib: float = 5,
    n_jobs: int = 1,
) -> None:
    """Render all matched C/H/O cycles with explicit outer parallelism."""
    metric_ids = ("cop_cyc_evt", "h_abs_rate", "o_abs_rate")
    cycles = sorted(
        set.intersection(*(set(metrics[metric_id]["cycle_name"]) for metric_id in metric_ids))
    )
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        candidate_tables = Parallel(verbose=5)(
            delayed(_render_parallel_policy_cycle)(
                str(cycle_name),
                metrics,
                loader,
                records[str(cycle_name)],
                output,
                guardrail=guardrail,
                pareto_selector=pareto_selector,
                fetch_cloud=fetch_cloud,
                minimum_free_gib=minimum_free_gib,
            )
            for cycle_name in cycles
        )
    if candidate_tables:
        output.mkdir(parents=True, exist_ok=True)
        pd.concat(candidate_tables, ignore_index=True).to_csv(
            output / f"pareto_{pareto_selector}_candidates.csv", index=False
        )


def _render_cycle_sets(  # noqa: C901
    tables: Mapping[str, pd.DataFrame],
    loader: DatasetLoader,
    records: Mapping[str, Mapping[str, object]],
    output: Path,
    *,
    fetch_cloud: bool = False,
    minimum_free_gib: float = 5,
) -> None:
    v27_metrics = tuple(name for name in V27_METRIC_STYLES if name in tables)
    titles = {
        "水侧制热量_cycle": "Water-heat optimum",
        "cost_function_v1_cycle": "Unit-heat V1 optimum",
        "cost_function_v2_cycle": "Updated V2 optimum",
    }
    suites: dict[str, pd.DataFrame] = {}
    if "v1" in tables:
        suites["水侧制热量_cycle"] = _publication_curve(tables["v1"], "water_reference")
        suites["cost_function_v1_cycle"] = _publication_curve(tables["v1"], "v1")
    if "v2" in tables:
        suites["cost_function_v2_cycle"] = _publication_curve(tables["v2"], "v2")
    for algorithm in (
        "v2.1",
        "v2.2",
        "v2.3",
        "v2.4",
        "v2.5",
        "v2.6",
        *V26_PATCHES,
        "v3",
    ):
        if algorithm in tables:
            directory = f"cost_function_{algorithm}_cycle"
            titles[directory] = (
                "V3 offline decision"
                if algorithm == "v3"
                else "V2.6.8 pre-action outcome diagnostic minimum"
                if algorithm == "v2.6.8"
                else "V2.6.7 diagnostic identification minimum"
                if algorithm == "v2.6.7"
                else "V2.6.6 diagnostic identification minimum"
                if algorithm == "v2.6.6"
                else f"{algorithm.upper()} optimum"
            )
            suites[directory] = _publication_curve(tables[algorithm], algorithm)
    if "renewal_water" in tables:
        directory = "cost_function_renewal_water_cycle"
        titles[directory] = "Renewal-water optimum"
        suites[directory] = tables["renewal_water"]
    for metric_id in v27_metrics:
        directory = f"cost_function_v2.7_cycle/{metric_id}"
        curve = tables[metric_id]
        label = str(curve.get("objective_label", pd.Series([metric_id])).iloc[0])
        direction = str(curve.get("optimization_direction", pd.Series(["min"])).iloc[0])
        titles[directory] = f"{label} diagnostic {'maximum' if direction == 'max' else 'minimum'}"
        suites[directory] = _publication_curve(curve, metric_id)
    cycles = sorted(set().union(*(set(table["cycle_name"]) for table in suites.values())))
    for cycle_name in cycles:
        cycle_name = str(cycle_name)
        record = records[cycle_name]
        frame = loader.load_cycle(cycle_name)
        metadata = loader.load_image_metadata(cycle_name)
        images = loader.load_cycle_images(cycle_name)
        for label, table in suites.items():
            cycle_mask = (
                table["cycle_name"].astype(str).eq(cycle_name)
                if v27_metrics
                else table["cycle_name"].eq(cycle_name)
            )
            curve = table.loc[cycle_mask]
            if curve.empty:
                continue
            display_metric = None
            display_label = "Unsupported model extension, display only"
            cost_label = None
            minimum_support_label = None
            metric_id = label.removeprefix("cost_function_v2.7_cycle/") if v27_metrics else ""
            if metric_id in v27_metrics:
                curve = curve.copy()
                eligible = curve["optimization_eligible"].fillna(False)
                curve["display_only_objective"] = pd.to_numeric(
                    curve.get("display_only_objective", curve["objective_value"]), errors="coerce"
                ).where(~eligible)
                display_metric = "display_only_objective"
                display_label = "Outside formal support/identifiability, display only"
                first = curve.iloc[0]
                cost_label = (
                    f"{V27_METRIC_STYLES[metric_id][2]} "
                    f"[{first.get('objective_unit', '-')}]"
                )
            elif curve.iloc[0].get("algorithm") == "v2.6.7":
                curve = _with_v267_display_extension(curve)
                display_metric = V267_DISPLAY_METRIC
                cycle_status = str(curve.iloc[0].get("cycle_status", "identified_curve"))
                if cycle_status != "identified_curve":
                    minimum_support_label = cycle_status.replace("_", " ")
            elif curve.iloc[0].get("algorithm") == "v2.6.8":
                curve = _with_v268_display_extension(curve)
                display_metric = V268_DISPLAY_METRIC
                cycle_status = str(curve.iloc[0].get("cycle_status", "identified_curve"))
                if cycle_status != "identified_curve":
                    minimum_support_label = cycle_status.replace("_", " ")
            filename = f"cycle_{int(cycle_name.rsplit('_', 1)[-1]):03d}_publication.png"
            decisions = _decision_images(metadata, images, curve)
            missing = sorted(
                {
                    str(info["file_name"])
                    for info in decisions.values()
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
                cycle_images = images
                if cycle_dir is not None:
                    downloaded = scan_cycle_images(
                        loader.dataset_root,
                        cycle_name,
                        metadata,
                        cycle_dir=cycle_dir,
                    )
                    cycle_images = (
                        downloaded
                        if images.empty
                        else pd.concat([images, downloaded], ignore_index=True)
                    )
                    decisions = _decision_images(metadata, cycle_images, curve)
                render_decision_publication(
                    frame,
                    record,
                    curve,
                    decisions,
                    output / label / filename,
                    optimal_label=(
                        f"{titles[label]} ({str(curve.iloc[0]['cycle_status']).replace('_', ' ')})"
                        if curve.iloc[0].get("algorithm") in STATUS_MARKERS
                        else titles[label]
                    ),
                    cost_label=cost_label or "Cycle inverse COP [-]",
                    full_candidate_domain=True,
                    display_metric=display_metric,
                    display_label=(
                        "Non-eligible V2.6.8 model curve, display only"
                        if curve.iloc[0].get("algorithm") == "v2.6.8"
                        else display_label
                    ),
                    minimum_label=(
                        "Diagnostic/raw minimum"
                        if curve.iloc[0].get("algorithm") in {"v2.6.7", "v2.6.8"}
                        else "Maximum"
                        if metric_id in v27_metrics
                        and str(curve.iloc[0].get("optimization_direction")) == "max"
                        else "Minimum"
                    ),
                    minimum_support_label=minimum_support_label,
                )


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


def generate_cost_function_figures(  # noqa: C901
    sources: Mapping[str, Path],
    loader: DatasetLoader,
    output: Path,
    *,
    comparison_only: bool = False,
    curves_only: bool = False,
    fetch_cloud: bool = False,
    minimum_free_gib: float = 5,
    diagnostics: bool = True,
    n_jobs: int = 6,
    pareto_selector: str = "knee",
) -> None:
    """Read comprehensive cost CSVs and write comparison/publication PNGs."""
    metrics = _read_v27_metrics(sources)
    historical_sources = {
        label: path
        for label, path in sources.items()
        if "metric_id" not in pd.read_csv(path, nrows=0).columns
    }
    tables = _read_tables(historical_sources) if historical_sources else {}
    all_tables = {**tables, **metrics}
    if not all_tables:
        raise ValueError("no cost-function tables were provided")
    cycles = sorted(set().union(*(set(table["cycle_name"]) for table in all_tables.values())))
    records = {cycle: loader.get_cycle_record(str(cycle)) for cycle in cycles}
    for table in all_tables.values():
        table["cycle_start"] = table["cycle_name"].map(
            lambda cycle: records[str(cycle)]["boundaries"]["start_time"]
        )
    parallel_ids = {"cop_cyc_evt", "h_abs_rate", "o_abs_rate"}
    if parallel_ids.issubset(metrics) and not comparison_only and not curves_only:
        _render_parallel_policy_cycles(
            metrics,
            loader,
            records,
            output / f"Pareto_{pareto_selector}",
            pareto_selector=pareto_selector,
            fetch_cloud=fetch_cloud,
            minimum_free_gib=minimum_free_gib,
            n_jobs=n_jobs,
        )
        if set(metrics) == parallel_ids and not tables:
            return
    if metrics:
        _write_v27_summaries(
            metrics,
            tables,
            sources,
            output,
            diagnostics=diagnostics and not comparison_only and not curves_only,
        )
    if curves_only:
        family = (
            "cost_function_v1_v2.5_v2.6.7_v2.6.8_v2.7_cycle"
            if metrics
            else
            "cost_function_v1_v2.5_v2.6.7_v2.6.8_cycle"
            if {"v1", "v2.5", "v2.6.7", "v2.6.8"}.issubset(tables)
            else "cost_function_v1_v2.5_v2.6.5_v2.6.6_v2.6.7_cycle"
            if {"v1", "v2.5", "v2.6.5", "v2.6.6", "v2.6.7"}.issubset(tables)
            else "cost_function_v1_v2.5_v2.6.5_v2.6.6_cycle"
            if {"v1", "v2.5", "v2.6.5", "v2.6.6"}.issubset(tables)
            else "cost_function_v1_to_v2.6_cycle"
        )
        _render_cost_curve_comparisons(
            all_tables,
            loader,
            output / family,
            fetch_cloud=fetch_cloud,
            minimum_free_gib=minimum_free_gib,
        )
        return
    for algorithm in (
        "v1",
        "v2",
        "v2.1",
        "v2.2",
        "v2.3",
        "v2.4",
        "v2.5",
        "v2.6",
        *V26_PATCHES,
        "v3",
        "renewal_water",
    ):
        if algorithm in tables:
            figure = _comparison_figure(tables, (algorithm,))
            path = output / f"comparison_{algorithm}_RB.png"
            (_save_svg_png if algorithm == "renewal_water" else _save_png)(figure, path)
    patches = tuple(algorithm for algorithm in V26_PATCHES if algorithm in tables)
    if len(patches) > 1:
        _save_png(
            _comparison_figure(tables, patches),
            output / f"comparison_{'_'.join(patches)}_RB.png",
        )
    for algorithms in (
        ("v1", "v2"),
        ("v1", "v2.1"),
        ("v1", "v2.2"),
        ("v1", "v2.1", "v2.2"),
        ("v1", "v2.2", "v2.3"),
        ("v1", "v2.3", "v2.4"),
        ("v1", "v2.3", "v2.5"),
        ("v2.5", "v2.6"),
        ("v1", "v2.5", "v2.6"),
        ("v1", "v2.5", "v2.6.5", "v2.6.6"),
        ("v1", "v2.5", "v2.6.5", "v2.6.6", "v2.6.7"),
        ("v1", "v2.5", "v2.6.7", "v2.6.8"),
        ("v2.5", "v2.6", "v3"),
        ("v1", "v3"),
        ("v1", "v2.2", "v2.5", "renewal_water"),
    ):
        if set(algorithms).issubset(tables):
            figure = _comparison_figure(tables, algorithms)
            path = output / f"comparison_{'_'.join(algorithms)}_RB.png"
            (_save_svg_png if "renewal_water" in algorithms else _save_png)(figure, path)
    if "v3" in tables and {"recommended_time", "recommended_rule"}.issubset(tables["v3"]):
        recommended = tables["v3"].copy()
        recommended["algorithm"] = "v3_recommended"
        recommended["t_star"] = recommended["recommended_time"]
        recommended["t_star_model_supported"] = True
        _save_png(
            _comparison_figure(
                {**tables, "v3_recommended": recommended},
                ("v3", "v3_recommended"),
            ),
            output / "comparison_v3_raw_recommended_RB.png",
        )
    if not comparison_only:
        _render_cycle_sets(
            all_tables,
            loader,
            records,
            output,
            fetch_cloud=fetch_cloud,
            minimum_free_gib=minimum_free_gib,
        )
        if metrics:
            _render_cost_curve_comparisons(
                all_tables,
                loader,
                output / "cost_function_v1_v2.5_v2.6.7_v2.6.8_v2.7_cycle",
                fetch_cloud=fetch_cloud,
                minimum_free_gib=minimum_free_gib,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost", nargs="+", required=True, metavar="LABEL=CSV")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("output/成本函数"))
    parser.add_argument("--comparison-only", action="store_true")
    parser.add_argument("--curves-only", action="store_true")
    parser.add_argument("--fetch-cloud", action="store_true")
    parser.add_argument("--minimum-free-gib", type=float, default=5)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--pareto-selector", choices=("knee", "latest"), default="knee")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--v267-evidence-output", type=Path)
    args = parser.parse_args()
    sources = {label: Path(path) for label, path in (item.split("=", 1) for item in args.cost)}
    loader = DatasetLoader(args.dataset)
    generate_cost_function_figures(
        sources,
        loader,
        args.output,
        comparison_only=args.comparison_only,
        curves_only=args.curves_only,
        fetch_cloud=args.fetch_cloud,
        minimum_free_gib=args.minimum_free_gib,
        diagnostics=not args.no_diagnostics,
        n_jobs=args.n_jobs,
        pareto_selector=args.pareto_selector,
    )
    if args.v267_evidence_output is not None:
        v267_path = next((path for path in sources.values() if "v2.6.7" in path.name), None)
        if v267_path is None:
            parser.error("--v267-evidence-output requires a V2.6.7 cost source")
        stem = v267_path.with_suffix("")
        generate_v267_evidence(
            pd.read_csv(stem.with_name(f"{stem.name}_bootstrap_audit.csv")),
            pd.read_csv(stem.with_name(f"{stem.name}_ticket_loeo.csv")),
            args.v267_evidence_output,
            pd.read_csv(stem.with_name(f"{stem.name}_cycle_audit.csv")),
        )


if __name__ == "__main__":
    main()
