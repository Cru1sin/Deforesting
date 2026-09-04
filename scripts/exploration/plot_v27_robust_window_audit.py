#!/usr/bin/env python3
"""Show what the current V2.7 artifacts do—and do not—prove about time windows."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "output" / "成本函数"
OUTPUT = INPUT / ".." / "test" / "成本函数" / "评价指标比较"
METRICS = {
    "eta_e_cyc": ("Outdoor heat retention", "#2A788E"),
    "cop_e": ("Outdoor heat / electricity", "#4FA3A5"),
    "epsilon_hl": ("Dynamic healthy heat loss", "#C85C5C"),
    "epsilon_hl_t0_proxy": ("Early-window heat loss", "#E7A3A3"),
    "cop_cyc_k": ("Recovery-boundary cycle COP", "#7A6BB7"),
    "epsilon_hl_2a": ("Two-anchor heat loss", "#D9903D"),
}


def load_curves() -> pd.DataFrame:
    frames = [
        pd.read_csv(INPUT / f"cost_function_v2.7.{i}.csv", low_memory=False) for i in range(4)
    ]
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(["cycle_name", "candidate_time", "metric_id"])
    data["candidate_time"] = pd.to_datetime(data["candidate_time"])
    for column in (
        "t_star",
        "bootstrap_extreme_q25_time",
        "bootstrap_extreme_q75_time",
    ):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    return data


def metric_summary(data: pd.DataFrame) -> pd.DataFrame:
    first = data.drop_duplicates(["cycle_name", "metric_id"]).copy()
    first["bootstrap_iqr_minutes"] = (
        first["bootstrap_extreme_q75_time"] - first["bootstrap_extreme_q25_time"]
    ).dt.total_seconds() / 60
    first["bootstrap_valid_fraction"] = (
        first["bootstrap_valid_extreme_count"] / first["repeat_count"]
    )
    rows = []
    for metric_id, group in first.groupby("metric_id", sort=False):
        identified = group.loc[group["t_star"].notna()]
        gate = identified["bootstrap_valid_fraction"].ge(0.8) & identified[
            "bootstrap_in_original_5pct_basin_fraction"
        ].ge(0.8)
        rows.append(
            {
                "metric_id": metric_id,
                "identified_cycles": len(identified),
                "median_basin_1pct_minutes": identified["basin_1pct_width_minutes"].median(),
                "median_basin_5pct_minutes": identified["basin_5pct_width_minutes"].median(),
                "median_bootstrap_iqr_minutes": identified["bootstrap_iqr_minutes"].median(),
                "median_bootstrap_valid_fraction": identified["bootstrap_valid_fraction"].median(),
                "median_bootstrap_5pct_hit_fraction": identified[
                    "bootstrap_in_original_5pct_basin_fraction"
                ].median(),
                "cycles_passing_location_screen": int(gate.sum()),
            }
        )
    return pd.DataFrame(rows).set_index("metric_id").loc[list(METRICS)].reset_index()


def _longest_run(group: pd.DataFrame, columns: list[tuple[str, str]]) -> float:
    valid = np.logical_and.reduce(
        [group[column].fillna(False).astype(bool).to_numpy() for column in columns]
    )
    times = group.index.get_level_values("candidate_time")
    runs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = previous = None
    for time, keep in zip(times, valid, strict=True):
        if keep and (start is None or (time - previous).total_seconds() <= 90):
            start = time if start is None else start
            previous = time
        else:
            if start is not None:
                runs.append((start, previous))
            start = previous = time if keep else None
    if start is not None:
        runs.append((start, previous))
    return max(((end - begin).total_seconds() / 60 for begin, end in runs), default=np.nan)


def intersection_summary(data: pd.DataFrame) -> pd.DataFrame:
    selected = data.loc[data["metric_id"].isin(["eta_e_cyc", "epsilon_hl", "cop_cyc_k"])]
    wide = selected.pivot(
        index=["cycle_name", "candidate_time"],
        columns="metric_id",
        values=["near_optimal_1pct", "near_optimal_5pct"],
    )
    combinations = {
        "eta + COP": ["eta_e_cyc", "cop_cyc_k"],
        "epsilon + COP": ["epsilon_hl", "cop_cyc_k"],
        "eta + epsilon + COP": ["eta_e_cyc", "epsilon_hl", "cop_cyc_k"],
    }
    rows = []
    for percent in (1, 5):
        for label, metrics in combinations.items():
            widths = pd.Series(
                [
                    _longest_run(
                        group, [(f"near_optimal_{percent}pct", metric) for metric in metrics]
                    )
                    for _, group in wide.groupby(level=0)
                ]
            )
            rows.append(
                {
                    "tolerance_percent": percent,
                    "intersection": label,
                    "nonempty_cycles": int(widths.notna().sum()),
                    "positive_width_cycles": int(widths.gt(0).sum()),
                    "median_width_minutes": widths.dropna().median(),
                }
            )
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, intersections: pd.DataFrame) -> plt.Figure:
    labels = [METRICS[key][0] for key in summary["metric_id"]]
    colors = [METRICS[key][1] for key in summary["metric_id"]]
    y = np.arange(len(summary))
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), constrained_layout=True)
    figure.get_layout_engine().set(rect=(0.01, 0.055, 0.995, 0.95))

    axis = axes[0, 0]
    stages = [
        (0.87, "Point curve", "Connected 1% / 5% basin\navailable now", "#DDEBF0"),
        (0.55, "Bootstrap optimum", "Median + IQR + basin-hit rate\navailable now", "#E8E3F3"),
        (
            0.23,
            "Robust feasible window",
            "P(regret <= delta) and joint\nconstraint probability NOT stored",
            "#F5DFDF",
        ),
    ]
    for y0, title, text, color in stages:
        axis.text(0.05, y0, title, weight="bold", fontsize=9, va="center")
        axis.text(0.43, y0, text, fontsize=8.2, va="center")
        axis.add_patch(plt.Rectangle((0.02, y0 - 0.11), 0.95, 0.22, color=color, zorder=-1))
    axis.annotate("does not imply", (0.25, 0.67), (0.25, 0.75), arrowprops={"arrowstyle": "->"})
    axis.annotate("does not imply", (0.25, 0.35), (0.25, 0.43), arrowprops={"arrowstyle": "->"})
    axis.set(xlim=(0, 1), ylim=(0.06, 1), title="a  Evidence ladder: basin is not a robust window")
    axis.axis("off")

    axis = axes[0, 1]
    axis.barh(
        y + 0.20,
        summary["median_basin_1pct_minutes"],
        0.20,
        color=colors,
        alpha=0.95,
        label="Point 1% basin",
    )
    axis.barh(
        y,
        summary["median_basin_5pct_minutes"],
        0.20,
        color=colors,
        alpha=0.55,
        label="Point 5% basin",
    )
    axis.barh(
        y - 0.20,
        summary["median_bootstrap_iqr_minutes"],
        0.20,
        color="#4F4F4F",
        label="Bootstrap optimum IQR",
    )
    axis.set(
        yticks=y,
        yticklabels=labels,
        xlabel="Median width [min]",
        title="b  Curve width and optimum-location uncertainty",
    )
    axis.invert_yaxis()
    axis.legend(fontsize=7, loc="lower right")

    axis = axes[1, 0]
    for row, color in zip(summary.itertuples(), colors, strict=True):
        axis.scatter(
            row.median_bootstrap_valid_fraction,
            row.median_bootstrap_5pct_hit_fraction,
            s=70,
            color=color,
        )
        axis.annotate(
            METRICS[row.metric_id][0],
            (row.median_bootstrap_valid_fraction, row.median_bootstrap_5pct_hit_fraction),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.axvline(0.8, color="#888888", ls="--", lw=0.8)
    axis.axhline(0.8, color="#888888", ls="--", lw=0.8)
    axis.set(
        xlim=(0, 1.03),
        ylim=(0, 1.03),
        xlabel="Median valid bootstrap fraction",
        ylabel="Median optimum in original 5% basin",
        title="c  No metric clears the location-stability screen",
    )

    axis = axes[1, 1]
    triple = intersections.loc[intersections["intersection"].eq("eta + epsilon + COP")]
    x = np.arange(len(triple))
    axis.bar(x - 0.18, triple["nonempty_cycles"], 0.36, color="#7396B8", label="Non-empty")
    axis.bar(x + 0.18, triple["positive_width_cycles"], 0.36, color="#D08A5B", label="> 0 min")
    for index, row in enumerate(triple.itertuples()):
        axis.text(
            index,
            row.nonempty_cycles + 1.5,
            f"median {row.median_width_minutes:g} min",
            ha="center",
            fontsize=8,
        )
    axis.set(
        xticks=x,
        xticklabels=[f"{p}% native near-opt" for p in triple["tolerance_percent"]],
        ylim=(0, 69),
        ylabel="Cycles [n = 69]",
        title="d  Point-estimate three-metric intersection only",
    )
    axis.legend(fontsize=7)

    figure.suptitle(
        "V2.7 robust-window audit — current artifacts locate optima "
        "but do not estimate decision probability",
        x=0.01,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    figure.text(
        0.01,
        0.005,
        "A deployable window still requires bootstrap objective trajectories at every "
        "candidate time and predeclared engineering thresholds. "
        "The 0.8 lines are an audit screen, not a control specification.",
        fontsize=7.5,
    )
    return figure


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    data = load_curves()
    summary = metric_summary(data)
    intersections = intersection_summary(data)
    assert len(summary) == 6 and set(summary["identified_cycles"]) <= set(range(70))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "10_稳健可行窗口证据审计_指标数据.csv", index=False)
    intersections.to_csv(OUTPUT / "10_稳健可行窗口证据审计_交集数据.csv", index=False)
    figure = plot(summary, intersections)
    for suffix, options in (("png", {"dpi": 300}), ("svg", {}), ("pdf", {})):
        figure.savefig(OUTPUT / f"10_稳健可行窗口证据审计.{suffix}", bbox_inches="tight", **options)
    plt.close(figure)


if __name__ == "__main__":
    main()
