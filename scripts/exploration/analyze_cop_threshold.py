#!/usr/bin/env python3
"""Show why cycle-COP optima are not a fixed instantaneous-COP threshold."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataloader.loader import DatasetLoader
from plots.publication import (
    _STAGE_COLORS,
    _observed_values,
    _stage_spans,
)

OUTPUT = Path("output/test/成本函数/模型筛选")
METHODS = {
    "V2.5 · water COP": (
        Path("output/test/成本函数/cost_function_v2.5.csv"),
        "water_cop",
        "#0072B2",
    ),
    "V2.6 · refrigerant COP": (
        Path("output/test/成本函数/cost_function_v2.6.csv"),
        "cop",
        "#009E73",
    ),
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def best_threshold(
    traces: list[tuple[np.ndarray, np.ndarray, float]], thresholds: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return the optimistic fixed threshold and its absolute timing errors."""
    candidates = []
    for threshold in thresholds:
        errors = []
        for minutes, cop, optimum in traces:
            crossings = np.flatnonzero(cop <= threshold)
            trigger = minutes[crossings[0]] if len(crossings) else minutes[-1]
            errors.append(abs(trigger - optimum))
        values = np.asarray(errors, dtype=float)
        candidates.append((np.median(values), np.mean(values), float(threshold), values))
    _, _, threshold, errors = min(candidates, key=lambda item: item[:2])
    return threshold, errors


def _read_curve(path: Path) -> pd.DataFrame:
    curve = pd.read_csv(path)
    curve = curve.loc[curve["valid"].fillna(False)].copy()
    for column in ("candidate_time", "t_star", "t_RB"):
        curve[column] = pd.to_datetime(curve[column], errors="coerce", format="mixed")
    return curve


def _rolling_cop(loader: DatasetLoader, cycle: str, signal: str) -> pd.Series:
    frame = loader.load_cycle(cycle)
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", format="mixed")
    values = _observed_values(frame, signal)
    return (
        pd.Series(values.to_numpy(), index=timestamps)
        .sort_index()
        .rolling("5min", min_periods=10)
        .median()
        .dropna()
    )


def _threshold_data(
    loader: DatasetLoader, method: str, curve: pd.DataFrame, signal: str
) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray, float]]]:
    rows = []
    traces = []
    for cycle, group in curve.groupby("cycle_name", sort=True):
        start, end = group["candidate_time"].min(), group["candidate_time"].max()
        optimum = pd.Timestamp(group["t_star"].iloc[0])
        cop = _rolling_cop(loader, str(cycle), signal).loc[start:end]
        if cop.empty:
            continue
        minutes = (cop.index - start).total_seconds().to_numpy() / 60
        optimum_minute = (optimum - start).total_seconds() / 60
        optimum_cop = float(cop.iloc[cop.index.get_indexer([optimum], method="nearest")[0]])
        rows.append(
            {
                "method": method,
                "cycle_name": cycle,
                "optimal_minutes_from_search_start": optimum_minute,
                "cop_at_optimum_5min_median": optimum_cop,
            }
        )
        traces.append((minutes, cop.to_numpy(), optimum_minute))
    return pd.DataFrame(rows), traces


def _shade_stages(axis: plt.Axes, frame: pd.DataFrame, minutes: pd.Series) -> None:
    for stage, left, right in _stage_spans(frame, minutes):
        axis.axvspan(left, right, color=_STAGE_COLORS.get(stage, "#D9DEE5"), alpha=0.09)


def plot_cycle_90(loader: DatasetLoader, curve: pd.DataFrame) -> None:
    cycle = "frost_cycle_000090"
    frame = loader.load_cycle(cycle).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", format="mixed")
    start = pd.Timestamp(loader.get_cycle_record(cycle)["boundaries"]["start_time"])
    minutes = (frame["timestamp"] - start).dt.total_seconds() / 60
    selected = curve.loc[curve["cycle_name"].eq(cycle)].copy()
    selected["minute"] = (selected["candidate_time"] - start).dt.total_seconds() / 60
    optimum_minute = (selected["t_star"].iloc[0] - start).total_seconds() / 60
    rb_minute = (selected["t_RB"].iloc[0] - start).total_seconds() / 60

    unit = _rolling_cop(loader, cycle, "cop")
    water = _rolling_cop(loader, cycle, "water_cop")
    unit_minutes = (unit.index - start).total_seconds() / 60
    water_minutes = (water.index - start).total_seconds() / 60
    eligible = selected["optimization_eligible"].fillna(False)
    cost = selected["inverse_cop"].where(eligible)
    regret = 100 * (cost / cost.min() - 1)
    late = selected.loc[selected["minute"].between(80, 110) & eligible]
    peak = late.loc[late["inverse_cop"].idxmax()]
    recovered = selected.loc[
        selected["minute"].between(float(peak["minute"]), float(peak["minute"]) + 20) & eligible
    ].loc[lambda values: values["inverse_cop"].idxmin()]
    peak_regret = 100 * (peak["inverse_cop"] / cost.min() - 1)
    recovered_regret = 100 * (recovered["inverse_cop"] / cost.min() - 1)

    figure, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    _shade_stages(axes[0], frame, minutes)
    axes[0].plot(
        unit_minutes, unit, color="#009E73", lw=1.4, label="Refrigerant COP · 5-min median"
    )
    axes[0].plot(water_minutes, water, color="#0072B2", lw=1.4, label="Water COP · 5-min median")
    axes[0].plot(
        selected["minute"],
        selected["cycle_cop"].where(eligible),
        color="#606060",
        lw=1.2,
        ls="--",
        label="Objective-equivalent cycle COP = 1/J",
    )
    axes[0].set(ylabel="COP [-]", ylim=(1.35, 2.55))
    axes[0].legend(ncols=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    axes[0].set_title(
        "Cycle 90: a short COP rebound lowers cumulative cost, but cannot erase the preceding loss",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

    _shade_stages(axes[1], frame, minutes)
    axes[1].plot(selected["minute"], regret, color="#B2182B", lw=1.6)
    axes[1].axhspan(0, 1, color="#E28E2C", alpha=0.15, label="1% near-optimal")
    axes[1].scatter(
        [optimum_minute, peak["minute"], recovered["minute"]],
        [0, peak_regret, recovered_regret],
        color=["#E28E2C", "#B2182B", "#0072B2"],
        zorder=4,
    )
    axes[1].annotate(
        f"Optimum\n{optimum_minute:.1f} min · J={cost.min():.4f}",
        (optimum_minute, 0),
        xytext=(8, 16),
        textcoords="offset points",
        fontsize=8,
    )
    axes[1].annotate(
        f"Accumulated loss\n{peak['minute']:.1f} min · +{peak_regret:.1f}%",
        (peak["minute"], peak_regret),
        xytext=(-72, 12),
        textcoords="offset points",
        fontsize=8,
    )
    axes[1].annotate(
        f"After rebound\n{recovered['minute']:.1f} min · +{recovered_regret:.1f}%",
        (recovered["minute"], recovered_regret),
        xytext=(10, -2),
        textcoords="offset points",
        fontsize=8,
    )
    axes[1].set(
        xlabel="Minutes from cycle start",
        ylabel="Cost above optimum [%]",
        ylim=(-0.2, max(6, float(regret.max()) * 1.08)),
    )

    for axis in axes:
        axis.axvline(optimum_minute, color="#E28E2C", ls=":", lw=1.2, label="V2.5 optimum")
        axis.axvline(rb_minute, color="#2E7D5B", ls=":", lw=1.2, label="Rule defrost")
        axis.grid(axis="y", color="#D8D8D8", lw=0.5)
    figure.tight_layout()
    _save(figure, OUTPUT / "figure_cycle090_cop_rebound")


def plot_threshold_test(
    summaries: list[pd.DataFrame],
    threshold_results: dict[str, tuple[float, np.ndarray]],
) -> None:
    summary = pd.concat(summaries, ignore_index=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [0.9, 1.2]})
    rng = np.random.default_rng(7)
    metrics = []
    for index, (method, (_path, _signal, color)) in enumerate(METHODS.items()):
        values = summary.loc[summary["method"].eq(method)]
        x = index + rng.uniform(-0.13, 0.13, len(values))
        axes[0].scatter(x, values["cop_at_optimum_5min_median"], s=13, alpha=0.55, color=color)
        axes[0].boxplot(
            values["cop_at_optimum_5min_median"],
            positions=[index],
            widths=0.32,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": color},
            medianprops={"color": color, "linewidth": 1.5},
            whiskerprops={"color": color},
            capprops={"color": color},
        )
        cycle90 = values.loc[values["cycle_name"].eq("frost_cycle_000090")]
        axes[0].scatter(
            index,
            cycle90["cop_at_optimum_5min_median"],
            marker="*",
            s=90,
            color="#B2182B",
            zorder=5,
        )
        threshold, errors = threshold_results[method]
        axes[0].plot(
            [index - 0.22, index + 0.22], [threshold, threshold], color="#272727", lw=1.5, ls="--"
        )
        ordered = np.sort(errors)
        axes[1].plot(
            ordered,
            100 * np.arange(1, len(ordered) + 1) / len(ordered),
            color=color,
            lw=1.8,
            label=f"{method} · θ={threshold:.2f}",
        )
        metrics.append(
            {
                "method": method,
                "best_in_sample_threshold": threshold,
                "within_5min": float(np.mean(errors <= 5)),
                "within_10min": float(np.mean(errors <= 10)),
                "median_error_minutes": float(np.median(errors)),
                "p90_error_minutes": float(np.percentile(errors, 90)),
            }
        )

    axes[0].set(
        xticks=range(len(METHODS)),
        xticklabels=["V2.5\nwater COP", "V2.6\nrefrigerant COP"],
        ylabel="5-min COP at cost optimum [-]",
        title="Optimal points do not share one COP",
    )
    axes[0].text(
        0.02,
        0.98,
        "Red star: Cycle 90   -- best fixed threshold",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
    )
    axes[1].axvspan(0, 5, color="#AADCA9", alpha=0.22)
    axes[1].axvline(10, color="#767676", ls=":", lw=1)
    axes[1].set(
        xlabel="Absolute timing error [min]",
        ylabel="Cycles captured [%]",
        xlim=(0, 50),
        ylim=(0, 101),
        title="Even the best in-sample threshold misses many cycles",
    )
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(color="#D8D8D8", lw=0.5)
    figure.suptitle(
        "A fixed COP threshold is only a rough proxy for the cumulative-cost optimum",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    figure.tight_layout()
    _save(figure, OUTPUT / "figure_fixed_cop_threshold_test")

    metrics_frame = pd.DataFrame(metrics)
    summary = summary.merge(metrics_frame[["method", "best_in_sample_threshold"]], on="method")
    metrics_frame.to_csv(OUTPUT / "cop_threshold_metrics.csv", index=False)
    summary.to_csv(OUTPUT / "cop_threshold_cycle_summary.csv", index=False)


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    loader = DatasetLoader(Path("dataset"))
    summaries = []
    threshold_results = {}
    curves = {}
    for method, (path, signal, _color) in METHODS.items():
        curve = _read_curve(path)
        curves[method] = curve
        summary, traces = _threshold_data(loader, method, curve, signal)
        summaries.append(summary)
        threshold_results[method] = best_threshold(traces, np.linspace(1.4, 3.4, 201))
    plot_cycle_90(loader, curves["V2.5 · water COP"])
    plot_threshold_test(summaries, threshold_results)


if __name__ == "__main__":
    main()
