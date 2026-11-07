"""Re-screen the full published candidates, then render a descriptive data overview."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from dataset_tools import DatasetLoader
from image_models.cycle_selection import screen_cv_cycles
from output_layout import PROJECT, prepare_output
from plots.publication import _RGB_PANEL_FONT, save_publication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    parser.add_argument(
        "--output", type=Path, default=(PROJECT / "output/03_数据与物理依据/历史队列构成_101循环")
    )
    args = parser.parse_args()
    prepare_output(args)
    training = args.dataset / "training_data"
    settings = json.loads((training / "settings.json").read_text())
    decisions = pd.read_parquet(training / "candidate_decisions.parquet")
    catalog = DatasetLoader(args.dataset).list_cycles()
    terminal = screen_cv_cycles(decisions, catalog)
    quality = pd.read_csv(training / "cycle_screening.csv")
    # The saved audit includes selection outputs; retain only measured quality inputs.
    redundant = set(terminal.columns) - {"cycle_name", "experiment_id"}
    redundant.update(
        [
            "legacy_selected",
            "legacy_exclusion_reasons",
            "selected_train",
            "selected_evaluation",
            "gate_a",
            "gate_b",
            "gate_a_reasons",
            "gate_b_reasons",
        ]
    )
    quality = quality.drop(columns=list(redundant), errors="ignore")
    screened = screen_cv_cycles(
        decisions,
        catalog,
        quality=quality,
        exclude_rb=settings["exclude_rb"],
        input_rule=settings["input_rule"],
        cohort_rule=settings["cohort_rule"],
        maximum_interpolated_fraction=settings["limits"]["maximum_interpolated_fraction"],
    )
    selected = screened.loc[screened.selected]
    published = DatasetLoader(args.dataset).training_cohort()
    if set(selected.cycle_name) != set(published.cycle_name):
        raise ValueError("Re-screened cohort differs from published inputs")
    if settings["cohort_rule"] != "observed" or settings["input_rule"] != "rgb70":
        raise ValueError("This overview describes the observed RGB70 cohort")
    cop = screened.long_enough.fillna(False)
    limits = settings["limits"]
    visual = (
        cop
        & screened.raw_rgb_coverage.ge(limits["rgb_coverage"])
        & screened.dino_action_coverage.ge(limits["dino_coverage"])
    )
    stages = []
    for name, mask in [
        ("all", pd.Series(True, index=screened.index)),
        ("terminal_cop", cop),
        ("rgb_dino", visual),
        ("selected", screened.selected),
    ]:
        part = screened.loc[mask]
        stages.append(
            dict(
                stage=name,
                cycles=len(part),
                experiments=part.experiment_id.nunique(),
                actions=int(part.actions.sum()),
            )
        )
    groups = (
        selected.groupby("setpoint_c")
        .agg(
            cycles=("cycle_name", "size"),
            experiments=("experiment_id", "nunique"),
            baseline_cycles=("baseline_cycle", "sum"),
        )
        .reset_index()
    )
    assert groups.cycles.sum() == len(selected)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stages).to_csv(args.output / "screening_stages.csv", index=False)
    groups.to_csv(args.output / "setpoint_counts.csv", index=False)
    render_data_overview(args.output)
    print(pd.DataFrame(stages).to_string(index=False))
    print(groups.to_string(index=False))
    print(f"Exported PNG, SVG and PDF to {args.output}")


def render_data_overview(output):
    """Compact cohort waterfall and setpoint counts from authoritative screening."""
    output = Path(output)
    flow = pd.read_csv(output / "screening_stages.csv")
    groups = pd.read_csv(output / "setpoint_counts.csv")
    blue, ink, grey = "#436B85", "#253D50", "#66727D"
    with plt.rc_context(
        {
            "font.family": _RGB_PANEL_FONT,
            "font.size": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    ):
        fig = plt.figure(figsize=(10.8, 4.6))
        left = fig.add_axes([0.07, 0.26, 0.55, 0.58])
        right = fig.add_axes([0.745, 0.26, 0.225, 0.58])
        fig.text(0.025, 0.92, "a", fontsize=16, fontfamily="Arial", fontweight="bold", color=ink)
        fig.text(0.05, 0.92, "循环筛选", fontsize=15, color=ink)
        fig.text(
            0.62,
            0.925,
            f"{int(flow.experiments.iloc[0])} → {int(flow.experiments.iloc[-1])}个实验",
            ha="right",
            fontsize=10,
            color=grey,
        )
        fig.text(0.7, 0.92, "b", fontsize=16, fontfamily="Arial", fontweight="bold", color=ink)
        fig.text(0.725, 0.92, "水温工况分布", fontsize=15, color=ink)
        counts = flow.cycles.to_numpy(int)
        retained = ["#DEE5EB", "#C9DAE6", "#ADC7D9", blue]
        loss, loss_ink = "#E8CBB9", "#98694E"
        # Each full-width plateau is the remaining cohort; drops use the true count scale.
        for i, count in enumerate(counts):
            start, end = i - 0.5, i + 0.5
            left.add_patch(
                Rectangle((start, 0), 1, count, facecolor=retained[i], edgecolor="none", zorder=2)
            )
            left.plot([start, end], [count] * 2, color=blue, lw=1.6, zorder=4)
            left.text(
                i,
                count + 12,
                str(count),
                ha="center",
                fontsize=19,
                fontfamily="Arial",
                fontweight="bold",
                color=blue if i else ink,
            )
            if i:
                boundary = i - 0.5
                previous = counts[i - 1]
                removed = previous - count
                left.add_patch(
                    Rectangle(
                        (boundary - 0.055, count),
                        0.11,
                        removed,
                        facecolor=loss,
                        edgecolor="none",
                        zorder=3,
                    )
                )
                left.plot([boundary] * 2, [count, previous], color=blue, lw=1.3, zorder=4)
                target = (boundary, (previous + count) / 2)
                label_y = target[1] if i == 1 else counts[0] * 0.52
                left.annotate(
                    f"−{removed}",
                    xy=target,
                    xytext=(boundary, label_y),
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontfamily="Arial",
                    color=loss_ink,
                    zorder=6,
                    bbox={
                        "boxstyle": "round,pad=.35,rounding_size=.2",
                        "facecolor": "#F7EDE6",
                        "edgecolor": "none",
                    },
                    arrowprops=None
                    if i == 1
                    else {"arrowstyle": "-", "color": loss_ink, "lw": 0.8, "shrinkB": 3},
                )
        left.text(
            len(counts) - 1,
            counts[-1] * 0.5,
            f"{100 * counts[-1] / counts[0]:.1f}%保留",
            ha="center",
            va="center",
            fontsize=11,
            color="white",
        )
        left.plot([0.47, 0.51], [0.96, 0.96], color=blue, lw=3, transform=left.transAxes)
        left.text(
            0.53, 0.96, "保留循环", transform=left.transAxes, va="center", fontsize=9, color=grey
        )
        left.plot([0.76, 0.8], [0.96, 0.96], color=loss, lw=5, transform=left.transAxes)
        left.text(
            0.82, 0.96, "排除循环", transform=left.transAxes, va="center", fontsize=9, color=grey
        )
        left.set(
            xlim=(-0.65, len(counts) - 0.35),
            ylim=(0, counts[0] * 1.13),
            xticks=range(len(counts)),
            xticklabels=[
                "全部循环",
                "有效精确终点COP\n<99%峰值",
                "RGB与DINO覆盖\n各 ≥70%",
                "最终数据集\n全循环插值≤30%",
            ],
            yticks=np.arange(0, counts[0] + 1, 100),
            ylabel="循环数",
        )
        bars = right.bar(range(len(groups)), groups.cycles, width=0.6, color=blue, zorder=3)
        upper = int(np.ceil(groups.cycles.max() / 10) * 10 + 10)
        for bar, row in zip(bars, groups.itertuples(), strict=True):
            right.text(
                bar.get_x() + bar.get_width() / 2,
                row.cycles + 1.7,
                str(row.cycles),
                ha="center",
                fontsize=16,
                color=ink,
            )
        right.set(
            xticks=range(len(groups)),
            xticklabels=[f"{v:g}" for v in groups.setpoint_c],
            ylim=(0, upper),
            yticks=np.arange(0, upper + 1, 10),
            xlabel="水温设定值 (°C)",
            ylabel="循环数",
        )
        right.xaxis.labelpad = 8
        for ax in [left, right]:
            ax.tick_params(axis="x", length=0, pad=7, labelsize=10)
            ax.tick_params(axis="y", colors=grey, labelsize=9)
            ax.yaxis.labelpad = 7
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E7EBEF", lw=0.6)
            ax.spines["left"].set_color("#AEB6BD")
            ax.spines["bottom"].set_color("#AEB6BD")
        left.grid(False)
        left.spines["bottom"].set_visible(False)
        selected = flow.iloc[-1]
        fig.text(
            0.07,
            0.065,
            f"{int(selected.cycles)}个循环  ·  {int(selected.experiments)}个实验  ·  "
            f"{int(selected.actions):,}个30 s网格候选动作",
            fontsize=11,
            color=blue,
        )
        fig.text(
            0.97,
            0.065,
            f"含{int(groups.baseline_cycles.sum())}个RB循环",
            ha="right",
            fontsize=10,
            color=grey,
        )
        fig.canvas.draw()
        for ax in [left, right]:
            for artist in [
                *ax.texts,
                ax.xaxis.label,
                ax.yaxis.label,
                *ax.get_xticklabels(),
                *ax.get_yticklabels(),
            ]:
                if artist.get_visible() and artist.get_text():
                    box = artist.get_window_extent(fig.canvas.get_renderer())
                    if (
                        box.x0 < 0
                        or box.y0 < 0
                        or box.x1 > fig.bbox.width
                        or box.y1 > fig.bbox.height
                    ):
                        raise ValueError(f"Overview label outside canvas: {artist.get_text()}")
        save_publication(fig, output, "data_overview", "", layout=False, dpi=300)


if __name__ == "__main__":
    main()
