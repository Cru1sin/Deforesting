"""Relations between observed frost state and independent COP decisions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

from plots.defrost_decision import _plot_decision_image
from plots.publication import save_publication

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_coverage_steps(axis, coverage, *, color, label, scale=1.0, linewidth=1.8):
    """Shared exact interval-coverage curve; never smooth probability steps."""
    edges = np.r_[coverage.left.iloc[0], coverage.right.to_numpy()]
    return axis.stairs(
        scale * coverage.coverage.to_numpy(),
        edges,
        color=color,
        linewidth=linewidth,
        baseline=None,
        label=label,
    )


def plot_overlap(
    screening,
    intervals,
    coverage,
    output,
    *,
    cycle_names=None,
    cohort_name=None,
    grade_max=None,
    source_label=None,
    name="frost_grade_overlap",
    figure=None,
    axes=None,
    condition_color="#587FA0",
):
    """Shared-threshold evidence: interval membership and its actual denominator."""
    if grade_max is None:
        plt.style.use("seaborn-v0_8-whitegrid")
    cycles = (
        screening.loc[screening.core_overlap, "cycle_name"].tolist()
        if cycle_names is None
        else cycle_names
    )
    fixed = "common_support" in coverage
    size = (
        (183 / 25.4, max(4.8, len(cycles) * 0.085 + 2.8))
        if grade_max is not None
        else (12, max(9, len(cycles) * 0.17 + 4))
    )
    if grade_max is not None and len(cycles) > 60:
        size = (size[0], len(cycles) * 0.13 + 2.8)
    if axes is not None:
        top, bottom = axes
        fig = top.figure
    elif figure is None:
        fig, (top, bottom) = plt.subplots(
            2,
            1,
            figsize=size,
            sharex=True,
            gridspec_kw={"height_ratios": [max(2, len(cycles) / 20), 1]},
        )
    else:
        fig = figure
        top, bottom = fig.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [max(2, len(cycles) / 20), 1]}
        )
    if grade_max is not None:
        for y in range(0, len(cycles), 2):
            top.axhspan(y - 0.5, y + 0.5, color="#F3F6F9", zorder=0)
    availability = screening.set_index("cycle_name").reference_available
    for y, cycle in enumerate(cycles):
        rows = intervals.loc[intervals.cycle_name.eq(cycle)]
        for kind, color, width in [("observed", "#d9dfe5", 3), ("feasible", "#176b9b", 5)]:
            for row in rows.loc[rows.kind.eq(kind)].itertuples():
                top.plot(
                    [row.left, row.right],
                    [y, y],
                    color=condition_color if grade_max is not None else color,
                    lw=2.8 if grade_max is not None else width,
                    solid_capstyle="round" if grade_max is not None else "butt",
                    marker="|" if grade_max is not None and row.left == row.right else None,
                )
        if grade_max is None and rows.loc[rows.kind.eq("feasible")].empty:
            available = availability.get(cycle, False)
            message = (
                ("无观测近优等级" if grade_max is not None else "No feasible first crossing")
                if available
                else "Visual coordinate unavailable"
            )
            top.text(
                0.99 if grade_max is None else grade_max * 0.99,
                y,
                message,
                ha="right",
                va="center",
                fontsize=6,
                color="#777777",
            )
    top.set_yticks(range(len(cycles)), [str(int(c.rsplit("_", 1)[-1])) for c in cycles], fontsize=7)
    top.invert_yaxis()
    top.set_ylabel("Cycle ID")
    label = (
        cohort_name.replace("_", " ").replace("1pct", "1% COP")
        if cohort_name
        else f"{len(cycles)} core cycles"
    )
    if grade_max is not None:
        from plots.publication import save_publication

        normalized = "mean_normalized_cop" in coverage
        regret = "mean_cop_regret" in coverage
        observed_cop = "mean_observed_normalized_cop" in coverage
        observed_metric = regret or observed_cop
        values = (
            coverage.mean_observed_normalized_cop
            if observed_cop
            else (
                coverage.mean_cop_regret
                if regret
                else (coverage.mean_normalized_cop if normalized else coverage.coverage)
            )
        )
        metric_label = (
            "平均 COP regret"
            if regret
            else ("平均归一化 COP" if normalized or observed_cop else "近优区间命中重叠度")
        )
        top.set_title("循环 COP 的1%近优区间", loc="left", fontsize=8, pad=9)
        top.tick_params(axis="y", labelsize=6, length=0, pad=5)
        top.set_xlabel("霜层等级")
        top.grid(axis="y", visible=False)
        edges = np.r_[coverage.left.iloc[0], coverage.right.to_numpy()]
        if observed_metric:
            bottom.plot(coverage.grade, 100 * values, color="#345B7B", lw=1.4, zorder=3)
        else:
            bottom.stairs(
                100 * values,
                edges,
                fill=True,
                color=condition_color,
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )
            plot_coverage_steps(
                bottom,
                coverage.assign(coverage=values),
                color=condition_color,
                label=metric_label,
                scale=100,
                linewidth=1.4,
            )
        best = coverage.iloc[:0] if observed_metric else coverage.loc[values.eq(values.max())]
        best_grades = (best.left + best.right) / 2
        components = best.groupby(best.left.diff().ne(1).cumsum())
        for _, part in components:
            for ax in (top, bottom):
                ax.axvspan(
                    part.left.min(), part.right.max(), color="#D6AB6D", alpha=0.16, zorder=2, lw=0
                )
            bottom.plot(
                [part.left.min(), part.right.max()],
                [100 * values.loc[part.index[0]]] * 2,
                color="#B57935",
                lw=2.2,
                zorder=4,
                solid_capstyle="round",
            )
        bottom.set(
            xlabel="霜层等级",
            ylabel=f"{metric_label} (%)",
            ylim=(0, 105),
            yticks=[0, 25, 50, 75, 100],
        )
        bottom.set_title(metric_label, loc="left", fontsize=8, pad=9)
        grade_text = ", ".join(f"{value:g}" for value in best_grades)
        if not observed_metric:
            bottom.text(
                0.025,
                0.94,
                f"{values.max():.2%}"
                if normalized and len(best)
                else ("无共同有效等级" if normalized else f"{values.max():.1%}"),
                transform=bottom.transAxes,
                va="top",
                fontsize=17,
                fontfamily="Arial",
                fontweight="bold",
                color="#345B7B",
            )
            if normalized:
                support = coverage.loc[coverage.common_support].copy()
                ranges = []
                for _, part in support.groupby(support.left.diff().ne(1).cumsum()):
                    low = (part.left.iloc[0] + part.right.iloc[0]) / 2
                    high = (part.left.iloc[-1] + part.right.iloc[-1]) / 2
                    ranges.append(f"{low:g}" if low == high else f"{low:g}–{high:g}")
                detail = f"固定 {len(cycles)} 个循环等权平均\n共同有效等级 {', '.join(ranges)}\n仅在该范围内最大：等级 {grade_text}"
            else:
                detail = (
                    f"{int(best.n_near_optimal.max())}/{len(cycles)} cycles\n霜层等级 {grade_text}"
                )
            bottom.text(
                0.025,
                0.70,
                detail,
                transform=bottom.transAxes,
                va="top",
                fontsize=6.8,
                color="#52606C",
                linespacing=1.6,
            )
        else:
            bottom.set_ylim(
                0, 105 if observed_cop else max(5, np.ceil(values.max() * 100 / 5) * 5 + 5)
            )
            counts_axis = bottom.twinx()
            counts_axis.plot(
                coverage.grade, coverage.n_evaluated, color="#9DA6AD", lw=0.7, ls="--", zorder=1
            )
            counts_axis.set(ylabel="参与循环数 n(s)", ylim=(0, len(cycles) * 1.1))
            counts_axis.tick_params(labelsize=6, colors="#78858E")
            counts_axis.yaxis.label.set_color("#78858E")
            counts_axis.yaxis.label.set_fontsize(6)
            counts_axis.grid(False)
            counts_axis.spines["top"].set_visible(False)
            counts_axis.spines["right"].set_visible(True)
            counts_axis.spines["right"].set_color("#9DA6AD")
            legend = (
                "蓝线：COP / COP最大值（越高越好）" if observed_cop else "蓝线：regret（越低越好）"
            )
            bottom.text(
                0.975,
                0.05 if observed_cop else 0.95,
                legend + "  ·  灰虚线：参与循环数",
                transform=bottom.transAxes,
                ha="right",
                va="bottom" if observed_cop else "top",
                fontsize=6,
                color="#52606C",
            )
        if normalized:
            for _, part in coverage.groupby(
                coverage.common_support.ne(coverage.common_support.shift()).cumsum()
            ):
                if not part.common_support.iloc[0]:
                    bottom.axvspan(
                        part.left.min(),
                        part.right.max(),
                        color="#E9EDF0",
                        alpha=0.6,
                        zorder=0,
                        lw=0,
                    )
            bottom.text(
                0.975,
                0.05,
                "灰色区域：固定循环集合的数据不完整",
                transform=bottom.transAxes,
                ha="right",
                fontsize=6,
                color="#78858E",
            )
            if values.notna().any():
                lower = max(0, np.floor(values.min() * 100 / 5) * 5 - 5)
                bottom.set_ylim(lower, 102)
                bottom.set_yticks(np.arange(lower, 101, 5))
        for letter, ax in zip("ab", (top, bottom)):
            ax.text(
                -0.075,
                1.025,
                letter,
                transform=ax.transAxes,
                fontsize=9,
                fontfamily="Arial",
                fontweight="bold",
                va="bottom",
            )
            ax.set_xlim(-0.5, grade_max + 0.5)
            ax.set_xticks(np.r_[np.arange(0, grade_max + 1, 10), grade_max])
            if "grade_scale" in coverage and coverage.grade_scale.eq("vertices").all():
                ax.set_xlim(0.5, grade_max + 0.5)
                ax.set_xticks(coverage.grade)
            ax.tick_params(axis="x", labelbottom=True, labelsize=7, length=3)
            ax.grid(axis="x", visible=False)
            for spine in ax.spines.values():
                spine.set_color("#6F7C85")
                spine.set_linewidth(0.55)
        if axes is not None:
            for ax in axes:
                for text in list(ax.texts):
                    text.remove()
                ax.set_title("")
            top.set_xlabel("")
            top.set_ylim(len(cycles) - 0.5, -0.5)
            top.tick_params(axis="x", labelbottom=False)
            bottom.set_ylabel("Coverage (%)")
            bottom.set_xlabel("Visual frost-state grade")
            top.set_title(
                label, loc="left", fontsize=10, fontweight="bold", color=condition_color, pad=10
            )
            bottom.set_title(
                f"Max {values.max():.1%} · {int(best.n_near_optimal.max())}/{len(cycles)} · grade {grade_text}",
                loc="left",
                fontsize=7,
                color=condition_color,
                pad=7,
            )
            for line in bottom.lines:
                line.set_color(condition_color)
            bottom.grid(axis="y", color="#DCE1E6", lw=0.4, alpha=0.65)
            return
        top.set_ylim(len(cycles) - 0.5, -0.5)
        bottom.grid(axis="y", color="#DCE1E6", lw=0.4, alpha=0.65, zorder=0)
        long_panel = figure is None and len(cycles) > 60
        header_y = (lambda inches: 1 - inches / fig.get_figheight()) if long_panel else None
        fig.text(
            0.13,
            header_y(0.20) if long_panel else 0.97,
            label.split(" · ")[0],
            fontsize=13,
            fontfamily="Arial",
            fontweight="bold",
            color="#243F53",
            va="top",
        )
        fig.text(
            0.98,
            header_y(0.23) if long_panel else 0.963,
            f"n = {len(cycles)} cycles",
            fontsize=8,
            fontfamily="Arial",
            color="#52606C",
            ha="right",
            va="top",
        )
        fig.text(
            0.13,
            header_y(0.55) if long_panel else 0.925,
            source_label,
            fontsize=7,
            fontfamily="Arial",
            color="#78858E",
            va="top",
        )
        if normalized:
            mapping_note = (
                "相邻有效记录间局部线性插值，不外推；固定循环集合，仅比较共同有效等级"
                if "grade_mapping" in coverage
                else "首次达到等级时的 COP / 各循环自身有效 COP 最大值；仅比较共同有效等级"
            )
            fig.text(
                0.13,
                header_y(0.80) if long_panel else 0.89,
                mapping_note,
                fontsize=6,
                color="#78858E",
                va="top",
            )
        if regret:
            fig.text(
                0.13,
                header_y(0.80) if long_panel else 0.89,
                "实际观测等级：regret = 1 − COP / 各循环自身最大有效 COP；同等级先循环内平均，再循环间等权平均",
                fontsize=6,
                color="#78858E",
                va="top",
            )
        if observed_cop:
            fig.text(
                0.13,
                header_y(0.80) if long_panel else 0.89,
                "实际观测等级：COP / 各循环自身最大有效 COP；同等级先循环内平均，再对有数据的循环等权平均",
                fontsize=6,
                color="#78858E",
                va="top",
            )
        if (
            "grade_scale" in coverage
            and coverage.grade_scale.eq("vertices").all()
            and not observed_metric
        ):
            fig.text(
                0.13,
                header_y(0.80) if long_panel else 0.89,
                "1–16级视觉霜态；仅统计照片支持的近优估计等级，不填补缺失等级；每个循环一票。",
                fontsize=6,
                color="#78858E",
                va="top",
            )
        if "grade_scale" in coverage and coverage.grade_scale.eq("vertices").all():
            top.set_title("", loc="left")
            bottom.set_title("", loc="left")
        fig.subplots_adjust(
            left=0.13,
            right=0.93 if observed_metric else 0.98,
            bottom=0.55 / fig.get_figheight() if long_panel else 0.09,
            top=header_y(1.3) if long_panel else (0.80 if normalized or observed_metric else 0.85),
            hspace=0.075 if long_panel else 0.36,
        )
        if figure is None:
            save_publication(fig, Path(output), name, "", formats=("png",), layout=False, dpi=600)
        return
    top.set_title(f"a  Is there a shared 1% COP-optimal visual interval?  ({label})", loc="left")
    from matplotlib.lines import Line2D

    top.legend(
        handles=[
            Line2D([0], [0], color="#176b9b", lw=5, label="Feasible first crossing: (left, right]"),
            Line2D([0], [0], color="#d9dfe5", lw=3, label="Identified first-crossing support"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.045),
        ncol=2,
        frameon=False,
    )
    x = np.r_[coverage.left.iloc[0], coverage.right.to_numpy()]
    plot_coverage_steps(bottom, coverage, color="#176b9b", label="Coverage")
    bottom.set_ylim(0, 1.05)
    bottom.set_ylabel("Coverage", color="#176b9b")
    counts = bottom.twinx()
    counts.stairs(coverage.n_observed, x, color="#b36a24", linewidth=1.4, baseline=None)
    counts.set_ylim(0, max(1, len(cycles)) * 1.05)
    counts.set_ylabel("Evaluable cycles (N obs)", color="#b36a24")
    counts.grid(False)
    peak = coverage.loc[coverage.coverage.eq(coverage.coverage.max())]
    if fixed:
        if len(peak):
            title = (
                f"b  Common-support maximum = {int(peak.n_near_optimal.max())}/{len(cycles)} "
                f"({coverage.coverage.max():.1%}); denominator fixed at {len(cycles)}"
            )
        else:
            title = (
                f"b  No common evaluable support; maximum N obs = {coverage.n_observed.max()}/{len(cycles)}\n"
                "Fixed-cohort Coverage is undefined"
            )
        segments = coverage.groupby(
            coverage.common_support.ne(coverage.common_support.shift()).cumsum()
        )
        for _, rows in segments:
            if rows.common_support.iloc[0]:
                for ax in (top, bottom):
                    ax.axvspan(rows.left.iloc[0], rows.right.iloc[-1], color="#398454", alpha=0.07)
    else:
        title = (
            f"b  Maximum Coverage = {coverage.coverage.max():.1%}; "
            f"N obs at maximum = {peak.n_observed.min()}–{peak.n_observed.max()}"
        )
        full = coverage.loc[coverage.n_observed.eq(len(cycles))]
        if len(full):
            title = (
                f"b  With all {len(cycles)} cycles evaluable: maximum Coverage = {full.coverage.max():.1%}\n"
                + title.removeprefix("b  ")
            )
    bottom.set_title(title, loc="left")
    bottom.set_xlabel("Visual frost-state threshold ρ")
    bottom.set_xlim(0, 1)
    bottom.set_xticks(np.linspace(0, 1, 11))
    note = (
        f"Green: identified common support; all {len(cycles)} cycles evaluable. Coverage is undefined outside it."
        if fixed
        else "Conditional Coverage uses the actual number of evaluable cycles at each threshold."
    )
    if fixed and not coverage.common_support.any():
        note = f"No identified common support across all {len(cycles)} cycles; no fixed-cohort Coverage can be reported."
    criterion = (
        "Cohort: aligned visual coordinate and a supported below-0.99 point after the last 1% COP-optimal point; no coverage/endpoint filter.\n"
        if "included_in_overlap" in screening
        else ""
    )
    fig.text(
        0.08,
        0.015,
        "30 s actions; supported COP / supported maximum ≥ 0.99. Initial reference and unknown crossings are excluded.\n"
        + criterion
        + note
        + " No threshold selected; the coordinate is visual state, not calibrated frost mass.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for extension in ("png", "pdf"):
        suffix = (f"_{cohort_name}" if cohort_name else "") + ("_common_support" if fixed else "")
        fig.savefig(
            Path(output) / f"threshold_overlap{suffix}.{extension}", dpi=200, bbox_inches="tight"
        )
    plt.close(fig)


def plot_mean_frost_cop(
    coverage,
    output,
    *,
    cohort_label,
    name,
    source_label,
    statistic="mean",
    images=None,
    axes=None,
    condition_color="#6C3F8D",
    compact=False,
):
    """Cycle-level mean/median COP with participation on a shared state axis."""
    from plots.publication import save_publication

    column = {"mean": "mean_observed_normalized_cop", "median": "median_normalized_cop"}[statistic]
    value = coverage[column]
    near = coverage.loc[value.ge(0.99), "grade"].to_numpy(dtype=int)
    blocks = np.split(near, np.flatnonzero(np.diff(near) > 1) + 1) if len(near) else []
    intervals = ", ".join(str(b[0]) if len(b) == 1 else f"{b[0]}–{b[-1]}" for b in blocks)
    best = coverage.loc[value.eq(value.max())]
    peak = best.iloc[0]
    label = statistic.capitalize()
    total = int(peak.total_cycles)
    height = 6.8 if images is not None else 4.8
    y = lambda position: 1 - (1 - position) * 4.8 / height
    if axes is None:
        fig, axis = plt.subplots(figsize=(183 / 25.4, height))
        count = axis.twinx()
    else:
        axis, count = axes
        fig = axis.figure
    count.set_zorder(0)
    axis.set_zorder(1)
    axis.patch.set_visible(False)
    count.fill_between(coverage.grade, 0, coverage.n_evaluated, color="#158A86", alpha=0.16, lw=0)
    count.plot(
        coverage.grade, coverage.n_evaluated, color="#158A86", lw=0.9, label="Contributing cycles"
    )
    interpolated = "n_interpolated_cycles" in coverage
    if interpolated:
        count.plot(
            coverage.grade,
            coverage.n_evaluated - coverage.n_interpolated_cycles,
            color="#7E828A",
            ls=":",
            lw=0.8,
            label="Observed cycles only",
        )
    count.set(ylabel="Contributing cycles, n(s)", ylim=(0, total * 1.05))
    count.spines["right"].set_visible(True)
    count.spines["right"].set_color("#158A86")
    count.tick_params(axis="y", colors="#158A86", labelsize=7)
    count.yaxis.label.set_color("#158A86")
    for block in blocks:
        axis.axvspan(block[0] - 0.5, block[-1] + 0.5, color=condition_color, alpha=0.06, lw=0)
    axis.plot(
        coverage.grade,
        value,
        color=condition_color,
        lw=1.8,
        zorder=4,
        label=f"{label} normalized COP",
    )
    axis.axhline(0.99, color="#D07C2C", ls="--", lw=0.9, zorder=3, label="0.99 threshold")
    for grade in best.grade:
        axis.axvline(grade, color=condition_color, ls=":", lw=0.8, alpha=0.7)
    axis.set(
        xlabel="Frost-state grade",
        ylabel=rf"{label} normalized COP ($COP/COP_{{\max}}$)",
        xlim=(0, 99),
        ylim=(0.9, 1),
    )
    axis.set_yticks(np.arange(0.9, 1.001, 0.02))
    if "grade_scale" in coverage and coverage.grade_scale.eq("vertices").all():
        axis.set_xlim(0.5, coverage.grade.max() + 0.5)
        axis.set_xticks(coverage.grade)
        lower = max(0, np.floor(value.min() * 100) / 100 - 0.01)
        axis.set_ylim(lower, 1.005)
        axis.set_yticks(np.linspace(lower, 1, 6))
        axis.set_xlabel("Visual frost-state grade (1–16)")
        axis.lines[0].set_marker("o")
        axis.lines[0].set_markersize(3)
    axis.grid(axis="x", alpha=0.10, lw=0.5)
    if compact:
        axis.set(xlabel="Visual frost-state grade", ylabel="Mean normalized COP")
        axis.set_title(
            f"Peak {peak[column]:.2%} · grade {int(peak.grade)} · n = {int(peak.n_evaluated)}/{total}",
            loc="left",
            fontsize=7,
            color=condition_color,
            pad=6,
        )
        count.set(ylabel="", yticks=[0, total])
        count.xaxis.set_visible(False)
        count.tick_params(axis="y", colors="#8C969D", labelsize=5.5, length=2, pad=1)
        count.lines[0].set_color("#9AA8B2")
        count.lines[0].set_linewidth(0.65)
        count.collections[0].set_facecolor("#9AA8B2")
        count.collections[0].set_alpha(0.12)
        count.spines["right"].set_color("#C8CFD4")
        count.spines["right"].set_linewidth(0.5)
        count.text(1.0, 1.035, "n", transform=count.transAxes, fontsize=5.5, color="#8C969D")
        return
    if axes is not None:
        axis.set_title(
            cohort_label, loc="left", fontsize=10, fontweight="bold", color=condition_color, pad=10
        )
        axis.set_xlabel("")
        axis.tick_params(axis="x", labelbottom=False)
        axis.text(
            0.43,
            0.10,
            f"Max {peak[column]:.5f}\nGrade {int(peak.grade)} · n = {int(peak.n_evaluated)}/{total}",
            transform=axis.transAxes,
            fontsize=6.5,
            color=condition_color,
            va="bottom",
            linespacing=1.6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=2),
        )
        axis.set_ylabel("Mean normalized COP")
        count.set(
            xlim=(0.5, 16.5),
            xticks=np.arange(1, 17),
            xlabel="Visual frost-state grade",
            ylabel="Contributing cycles",
        )
        count.spines["right"].set_visible(False)
        count.yaxis.label.set_color("#52606C")
        count.tick_params(axis="y", colors="#52606C")
        count.lines[0].set_color(condition_color)
        count.collections[0].set_facecolor(condition_color)
        return
    fig.text(0.12, y(0.95), cohort_label, fontsize=13, fontweight="bold")
    fig.text(
        0.12,
        y(0.90),
        f"Maximum {statistic}: {peak[column]:.5f} at grade {int(peak.grade)}"
        f" | n = {int(peak.n_evaluated)}/{total}",
        fontsize=8,
        color="#6C3F8D",
    )
    method = "local interpolation" if interpolated else "observed grades"
    fig.text(
        0.12,
        y(0.855),
        f"{source_label} • {method} • cycle-equal {statistic}",
        fontsize=7,
        color=".4",
    )
    fig.text(
        0.12,
        y(0.808),
        f"{label} COP ≥ 0.99: grades {intervals or 'none'}",
        fontsize=8,
        color="#D07C2C",
    )
    fig.subplots_adjust(left=0.12, right=0.87, bottom=y(0.14), top=y(0.74))
    handles = [axis.lines[0], *count.lines, axis.lines[1]]
    axis.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.025),
        ncol=len(handles),
        frameon=False,
        fontsize=6.1,
        handlelength=2.5,
        columnspacing=1.25,
        borderaxespad=0,
    )
    if images is not None:
        headings = {
            "left": "Left boundary",
            "peak": f"Maximum {statistic}",
            "right": "Right boundary",
        }
        for group, position in enumerate(["left", "peak", "right"]):
            rows = images.loc[images.position.eq(position)].sort_values("rank")
            left = 0.12 + group * 0.265
            grade = int(rows.target_grade.iloc[0])
            fig.text(
                left,
                0.26,
                f"{headings[position]} · grade {grade}",
                fontsize=7,
                fontweight="bold",
                color="#35313A",
            )
            for index, row in enumerate(rows.itertuples()):
                photo = fig.add_axes([left + index * 0.081, 0.05, 0.075, 0.185])
                if row.available:
                    _plot_decision_image(photo, row._asdict(), "", row.image_time, row.image_time)
                    photo.set_aspect("equal")
                else:
                    photo.set_xticks([])
                    photo.set_yticks([])
                    photo.set_facecolor("white")
                    photo.set_box_aspect(1.75)
                    photo.text(
                        0.5,
                        0.5,
                        "磁盘缺图",
                        transform=photo.transAxes,
                        ha="center",
                        va="center",
                        fontsize=4,
                        color="#78858E",
                    )
                photo.set_title("", loc="left")
                photo.set_title(
                    f"C{int(row.cycle_name.rsplit('_', 1)[-1])} · {row.setpoint_c:g}°C",
                    fontsize=5.6,
                    pad=3,
                    color="#35313A",
                )
                for spine in photo.spines.values():
                    spine.set_color("#D2CFD5")
                    spine.set_linewidth(0.4)
                photo.text(
                    0.5,
                    -0.06,
                    f"Grade {int(row.observed_grade)}",
                    transform=photo.transAxes,
                    ha="center",
                    va="top",
                    fontsize=5.2,
                    color=".4",
                )
    save_publication(fig, output, name, "", layout=False, formats=("png",), dpi=600)
    return dict(
        statistic=statistic,
        maximum=float(peak[column]),
        maximum_grades=",".join(map(str, best.grade)),
        n_at_maximum=",".join(map(str, best.n_evaluated)),
        total_cycles=total,
        threshold=0.99,
        threshold_intervals=intervals,
        threshold_grades=",".join(map(str, near)),
    )


def render_frost_condition_comparison(output, *, images_dir=None, seed=0, metric="coverage"):
    """Compose the shared coverage and mean renderers from their saved numerical sources."""

    output = Path(output)

    settings = json.loads((output / "run_settings.json").read_text())
    source_label = (
        "Native global Harmony · Slingshot MST-guided polyline · offline (future frames) · empirical rate bound"
        if settings.get("uses_future_frames")
        else "Native Harmony + Slingshot MST · 16 ordered visual grades"
    )
    actions = pd.read_parquet(output / "figure_actions.parquet")
    screening = pd.read_csv(output / "cycle_screening.csv")
    screening["reference_available"] = screening.cycle_name.isin(
        actions.loc[actions.frost_grade.notna(), "cycle_name"]
    )
    intervals = pd.read_csv(output / "near_optimal_grade_intervals.csv").assign(kind="feasible")
    temperatures = actions.groupby("cycle_name").Ts.median()
    colors = ["#4C799B", "#8B73A8", "#C68B68", "#476D68"]
    cohorts = [(ts, sorted(group.index)) for ts, group in temperatures.groupby(temperatures)]
    with_images = images_dir is not None
    normalized = metric == "normalized-cop"
    if with_images:
        cohorts.append((None, sorted(temperatures.index)))
        if not Path(images_dir).is_dir():
            raise FileNotFoundError(f"RGB directory is not mounted: {images_dir}")
        coverage_figure = plt.figure(figsize=(183 / 25.4 * len(cohorts) / 3, 4.75))
        # Shared column edges; the two square photos together span the plot width.
        gap = 0.035
        width = (0.985 - 0.075 - gap * (len(cohorts) - 1)) / len(cohorts)
        lefts = [0.075 + col * (width + gap) for col in range(len(cohorts))]
        coverage_axes = np.array(
            [
                [coverage_figure.add_axes([left, y, width, height]) for left in lefts]
                for y, height in [(0.65, 0.18), (0.35, 0.22)]
            ]
        )
        photo_width = (width - 0.008) / 2
        photo_height = (
            photo_width * coverage_figure.get_figwidth() / coverage_figure.get_figheight()
        )
        photo_axes = [
            [
                coverage_figure.add_axes(
                    [left + rank * (photo_width + 0.008), 0.075, photo_width, photo_height]
                )
                for rank in range(2)
            ]
            for left in lefts
        ]
        rng = np.random.default_rng(seed)
        selected_images = []
    else:
        coverage_figure, coverage_axes = plt.subplots(
            2,
            len(cohorts),
            figsize=(183 / 25.4, 7.4),
            gridspec_kw=dict(height_ratios=[4.2, 1]),
            sharex=True,
        )
    mean_figure, mean_axes = plt.subplots(
        2,
        len(cohorts),
        figsize=(183 / 25.4, 4.25),
        gridspec_kw=dict(height_ratios=[3, 1]),
        sharex=True,
        sharey="row",
    )
    tables = []
    for col, ((ts, names), color) in enumerate(zip(cohorts, colors)):
        suffix = "all" if ts is None else f"{ts:g}C"
        coverage = pd.read_csv(output / f"grade_coverage_{suffix}.csv").assign(
            grade_scale="vertices"
        )
        mean = pd.read_csv(output / f"mean_observed_normalized_cop_{suffix}.csv").assign(
            grade_scale="vertices"
        )
        tables.append(mean)
        label = (
            f"All cycles · n = {len(names)}" if ts is None else f"{ts:g} °C · {len(names)} cycles"
        )
        plot_overlap(
            screening,
            intervals,
            coverage,
            output,
            cycle_names=names,
            cohort_name=label,
            grade_max=16,
            source_label="",
            axes=coverage_axes[:, col],
            condition_color=color,
        )
        plot_mean_frost_cop(
            mean,
            output,
            cohort_label=label,
            name="",
            source_label="",
            axes=mean_axes[:, col],
            condition_color=color,
        )
        coverage_axes[0, col].tick_params(axis="y", labelsize=5.4, pad=2)
        if with_images:
            # Labels identify actual cycles; each categorical row retains equal spacing.
            ticks = np.unique(np.linspace(0, len(names) - 1, 5, dtype=int))
            coverage_axes[0, col].set_yticks(
                ticks, [str(int(names[i].rsplit("_", 1)[-1])) for i in ticks]
            )
            coverage_axes[0, col].set_ylabel("Cycle ID")
            linewidth = (
                0.65
                * max(len(group) for temperature, group in cohorts if temperature is not None)
                / len(names)
            )
            for line in coverage_axes[0, col].lines:
                line.set_linewidth(linewidth)
                if line.get_marker() == "|":
                    line.set_marker("_")
                    line.set_markersize(3)
                    line.set_markeredgewidth(linewidth)
            pd.DataFrame(dict(row=np.arange(1, len(names) + 1), cycle_name=names)).to_csv(
                output / f"循环行号_{suffix}.csv", index=False
            )
            coverage_axes[0, col].set_title(label, loc="left", fontsize=9, color=color, pad=6)
            metric_table = mean if normalized else coverage
            column = "mean_observed_normalized_cop" if normalized else "coverage"
            peak = metric_table.loc[metric_table[column].eq(metric_table[column].max())]
            grades = peak.grade.astype(int).tolist()
            if normalized:
                bottom = coverage_axes[1, col]
                bottom.clear()
                plot_mean_frost_cop(
                    mean,
                    output,
                    cohort_label=label,
                    name="",
                    source_label="",
                    axes=[bottom, bottom.twinx()],
                    condition_color=color,
                    compact=True,
                )
                for patch in list(coverage_axes[0, col].patches):
                    if patch.get_zorder() == 2:
                        patch.remove()
                for grade in grades:
                    coverage_axes[0, col].axvspan(
                        grade - 0.5, grade + 0.5, color=color, alpha=0.12, zorder=0.5, lw=0
                    )
            else:
                from matplotlib.ticker import MultipleLocator

                coverage_axes[1, col].set_yticks(np.arange(0, 101, 10))
                coverage_axes[1, col].yaxis.set_minor_locator(MultipleLocator(5))
                coverage_axes[1, col].tick_params(axis="y", which="minor", length=1.5)
                coverage_axes[1, col].set_title(
                    f"Peak {coverage.coverage.max():.1%} · {int(peak.n_near_optimal.max())}/{len(names)} · grade "
                    + ", ".join(map(str, grades)),
                    loc="left",
                    fontsize=7,
                    color=color,
                    pad=6,
                )
            pool = actions.loc[actions.cycle_name.isin(names)].copy()
            pool["normalized_cop"] = pool.raw_effective_cop / pool.groupby(
                "cycle_name"
            ).raw_effective_cop.transform("max")
            eligible = pool.normalized_cop.notna() if normalized else pool.normalized_cop.ge(0.99)
            pool = pool.loc[pool.frost_grade.isin(grades) & eligible & pool.file_name.notna()]
            pool = pool.drop_duplicates(["cycle_name", "file_name"]).sort_values(
                ["cycle_name", "file_name"]
            )
            pool["image_path"] = [
                str(Path(images_dir) / r.cycle_name / "front" / r.file_name)
                for r in pool.itertuples()
            ]
            pool = pool.loc[pool.image_path.map(lambda p: Path(p).is_file())]
            if pool.empty:
                raise FileNotFoundError(f"No available RGB at selected {metric} grades for {label}")
            from PIL import Image

            choices = rng.choice(pool.cycle_name.unique(), size=2, replace=False)
            for rank, name in enumerate(choices):
                frames = pool.loc[pool.cycle_name.eq(name)]
                sample = frames.iloc[int(rng.integers(len(frames)))]
                with Image.open(sample.image_path) as image:
                    image = np.rot90(np.asarray(image.convert("RGB")))
                height, image_width = image.shape[:2]
                side = min(height, image_width)
                image = image[
                    (height - side) // 2 : (height + side) // 2,
                    (image_width - side) // 2 : (image_width + side) // 2,
                ]
                photo = photo_axes[col][rank]
                photo.imshow(image, aspect="equal")
                photo.set_box_aspect(1)
                photo.set_xticks([])
                photo.set_yticks([])
                for spine in photo.spines.values():
                    spine.set_visible(True)
                    spine.set_color(color)
                    spine.set_linewidth(0.8)
                cycle_id = int(sample.cycle_name.rsplit("_", 1)[-1])
                photo.text(
                    0.035,
                    0.035,
                    f"Grade {int(sample.frost_grade)} · C{cycle_id}\n{pd.Timestamp(sample.image_time):%H:%M:%S}",
                    transform=photo.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=5.6,
                    color="#243F53",
                    linespacing=1.35,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2),
                )
                selected_images.append(
                    dict(
                        cohort=suffix,
                        setpoint_c=ts,
                        rank=rank + 1,
                        grade=int(sample.frost_grade),
                        cycle_name=sample.cycle_name,
                        file_name=sample.file_name,
                        image_time=sample.image_time,
                        image_path=sample.image_path,
                        normalized_cop=sample.normalized_cop,
                        seed=seed,
                        rotation_ccw=90,
                        crop="center square",
                    )
                )
        for ax in [*coverage_axes[:, col], *mean_axes[:, col]]:
            ax.tick_params(axis="x", labelsize=6, length=2)
            ax.tick_params(axis="y", labelsize=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.label.set_fontsize(7)
            ax.yaxis.label.set_fontsize(7)
        if col:
            coverage_axes[0, col].set_ylabel("")
            coverage_axes[1, col].set_ylabel("")
            coverage_axes[1, col].tick_params(labelleft=False)
            mean_axes[0, col].set_ylabel("")
            mean_axes[1, col].set_ylabel("")
    lower = max(
        0, np.floor(min(table.mean_observed_normalized_cop.min() for table in tables) * 10) / 10
    )
    if with_images and normalized:
        for axis in coverage_axes[1]:
            axis.set_ylim(0.9, 1.0)
            axis.set_yticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.0])
    for ax in mean_axes[0]:
        ax.set_ylim(lower, 1.015)
        ax.set_yticks(np.arange(lower, 1.001, 0.2))
    count_limit = max(len(names) for _, names in cohorts)
    for ax in mean_axes[1]:
        ax.set_ylim(0, count_limit * 1.05)
        ax.set_yticks([0, count_limit // 2, count_limit])
        ax.grid(axis="y", alpha=0.12, lw=0.4)
    if not with_images:
        coverage_figure.subplots_adjust(
            left=0.075, right=0.985, bottom=0.08, top=0.88, wspace=0.23, hspace=0.20
        )
    mean_figure.subplots_adjust(
        left=0.085, right=0.985, bottom=0.13, top=0.80, wspace=0.20, hspace=0.12
    )
    for fig, title in [
        (
            coverage_figure,
            "Mean normalized COP across visual frost states"
            if with_images and normalized
            else "COP near-optimal frost-state ranges",
        ),
        (mean_figure, "Mean normalized COP across visual frost states"),
    ]:
        fig.text(0.075, 0.97, title, fontsize=12, fontweight="bold", color="#243F53", va="top")
        fig.text(0.075, 0.935, source_label, fontsize=7, color="#78858E", va="top")
    coverage_figure.text(
        0.075,
        0.037 if with_images else 0.025,
        "Means: within-cycle, then equal-cycle over available cycles. Grey area/right axis: contributing n. Dashed line: 0.99."
        if with_images and normalized
        else "Near-optimal: COP ≥ 0.99 × own-cycle maximum. One row per cycle, sorted and labelled by actual cycle ID.",
        fontsize=6,
        color="#78858E",
    )
    if with_images:
        coverage_figure.text(
            0.075,
            0.012,
            "Top: cycle COP ≥ 0.99 × own maximum. Photos: two random cycles at peak-mean grades; 90° CCW, square centre crop."
            if normalized
            else "Photos: two random cycles per condition at a peak-coverage grade; near-optimal frames, 90° CCW, square centre crop.",
            fontsize=6,
            color="#78858E",
        )
        photo_name = "最高平均COP等级_随机照片.csv" if normalized else "最高覆盖等级_随机照片.csv"
        pd.DataFrame(selected_images).to_csv(output / photo_name, index=False)
    mean_figure.text(
        0.075,
        0.025,
        "Within-cycle means, then equal-cycle means; no interpolation. Dashed line: 0.99. Lower panels: contributing cycles.",
        fontsize=6,
        color="#78858E",
    )
    figure_name = (
        "三工况_平均归一化COP与霜态等级"
        if with_images and normalized
        else "三工况_近优区间与覆盖率"
    )
    save_publication(
        coverage_figure, output, figure_name, "", layout=False, formats=("png",), dpi=600
    )
    if not with_images:
        save_publication(
            mean_figure,
            output,
            "三工况_平均归一化COP与霜态等级",
            "",
            layout=False,
            formats=("png",),
            dpi=600,
        )
    else:
        plt.close(mean_figure)


def plot_frost_cop_mapping(
    curves, values, coverage, output, *, cohort_label, name, source_label, cycle_gallery=False
):
    """Aligned time traces, individual state curves, and grade-wise mean with participation."""
    from matplotlib.lines import Line2D

    colors = {50: "#467B9B", 55: "#7C8FAB", 60: "#AD8277"}
    names = sorted(curves.cycle_name.unique())
    setpoints = curves.groupby("cycle_name").Ts.median()
    durations = curves.groupby("cycle_name").candidate_defrost_time.agg(
        lambda x: (x.max() - x.min()).total_seconds()
    )
    example = (durations - durations.median()).abs().sort_values(kind="stable").index[0]
    g = curves.loc[curves.cycle_name.eq(example)].sort_values("candidate_defrost_time")
    elapsed = (g.candidate_defrost_time - g.candidate_defrost_time.min()).dt.total_seconds() / 60
    norm = g.raw_effective_cop / g.raw_effective_cop.max()
    color = colors.get(setpoints[example], "#467B9B")
    fig = plt.figure(figsize=(183 / 25.4, 6.6))
    grid = fig.add_gridspec(
        3,
        2,
        left=0.095,
        right=0.97,
        bottom=0.09,
        top=0.83,
        height_ratios=[1, 1.1, 0.3],
        hspace=0.55,
        wspace=0.32,
    )
    time_cop = fig.add_subplot(grid[0, 0])
    time_grade = fig.add_subplot(grid[0, 1], sharex=time_cop)
    individual = fig.add_subplot(grid[1, 0])
    mean = fig.add_subplot(grid[1, 1])
    count = fig.add_subplot(grid[2, 1], sharex=mean)
    fig.text(
        0.095, 0.97, "时间曲线 → 霜态曲线 → 逐等级平均", fontsize=12, color="#243F53", va="top"
    )
    fig.text(
        0.095,
        0.925,
        f"{cohort_label} · n = {len(names)}  |  {source_label}",
        fontsize=7,
        color="#52606C",
    )
    fig.text(
        0.095,
        0.885,
        f"a–b 示例：C{int(example.rsplit('_', 1)[-1])}（按循环时长中位数选取）；c–d 纳入全部循环",
        fontsize=6.5,
        color="#78858E",
    )
    time_cop.plot(elapsed, 100 * norm, color=color, lw=1.4)
    time_grade.plot(elapsed, g.frost_grade, color=color, lw=1.4)
    time_cop.set(xlabel="循环内时间 (min)", ylabel="归一化 COP (%)", ylim=(0, 105))
    time_grade.set(xlabel="循环内时间 (min)", ylabel="霜层等级", ylim=(-1, 100))
    # Mark the same candidate time in both source curves without introducing another selector.
    peak_position = int(np.nanargmax(norm.to_numpy()))
    for ax in [time_cop, time_grade]:
        ax.axvline(elapsed.iloc[peak_position], color="#BBC5CE", ls="--", lw=0.7)
    for cycle, part in values.groupby("cycle_name"):
        y = part.set_index("grade").normalized_cop.reindex(np.arange(100))
        individual.plot(
            y.index, 100 * y, color=colors.get(setpoints[cycle], "#467B9B"), lw=0.6, alpha=0.25
        )
    individual.set(xlabel="霜层等级", ylabel="归一化 COP (%)", xlim=(0, 99), ylim=(0, 105))
    individual.legend(
        handles=[
            Line2D([], [], color=colors[ts], lw=1.3, label=f"{ts:g}°C")
            for ts in sorted(setpoints.unique())
        ],
        loc="lower right",
        frameon=False,
        fontsize=6,
    )
    mean.plot(coverage.grade, 100 * coverage.mean_observed_normalized_cop, color="#345B7B", lw=1.7)
    mean.set(ylabel="平均归一化 COP (%)", xlim=(0, 99), ylim=(0, 105))
    mean.tick_params(labelbottom=False)
    count.fill_between(coverage.grade, 0, coverage.n_evaluated, color="#C6D6E2", alpha=0.65, lw=0)
    count.plot(coverage.grade, coverage.n_evaluated, color="#6F8798", lw=0.7)
    count.set(xlabel="霜层等级", ylabel="n(s)", ylim=(0, len(names) * 1.1))
    best = coverage.loc[
        coverage.mean_observed_normalized_cop.eq(coverage.mean_observed_normalized_cop.max())
    ]
    grade = int(best.grade.iloc[0])
    score = float(best.mean_observed_normalized_cop.iloc[0])
    n = int(best.n_evaluated.iloc[0])
    for ax in [individual, mean, count]:
        ax.axvspan(grade - 0.5, grade + 0.5, color="#D6AB6D", alpha=0.18, lw=0)
    mean.text(
        0.04,
        0.08,
        f"最高均值：等级 {grade}\n{score:.2%} · {n}/{len(names)} 个循环",
        transform=mean.transAxes,
        fontsize=7,
        color="#345B7B",
        linespacing=1.5,
    )
    fig.text(
        0.095,
        0.155,
        "同一时刻配对 COP 与霜态\n循环内补相邻跳级，不跨缺口、不外推\n同等级先循环内平均，再循环等权平均",
        fontsize=6.5,
        color="#52606C",
        linespacing=1.6,
    )
    for letter, title, ax in zip(
        "abcd",
        ["COP 随时间变化", "霜态随时间变化", "每个循环的霜态–COP 曲线", "逐等级均值及参与循环数"],
        [time_cop, time_grade, individual, mean],
    ):
        ax.text(
            -0.16,
            1.08,
            letter,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
            color="#243F53",
        )
        ax.set_title(title, loc="left", fontsize=7.5, pad=8)
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=6, length=2.5)
        ax.grid(axis="y", color="#E3E8EC", lw=0.4)
    save_publication(fig, Path(output), name, "", formats=("png",), layout=False, dpi=600)
    if cycle_gallery:
        columns = 7
        rows = int(np.ceil(len(names) / columns))
        fig, axes = plt.subplots(
            rows, columns, figsize=(183 / 25.4, 0.8 + rows * 0.78), squeeze=False
        )
        fig.subplots_adjust(
            left=0.06,
            right=0.98,
            bottom=0.045,
            top=1 - 0.55 / fig.get_figheight(),
            hspace=0.9,
            wspace=0.4,
        )
        fig.text(
            0.06, 0.98, "每个循环：霜层等级与归一化 COP", fontsize=11, color="#243F53", va="top"
        )
        for ax, cycle in zip(axes.ravel(), names):
            y = (
                values.loc[values.cycle_name.eq(cycle)]
                .set_index("grade")
                .normalized_cop.reindex(np.arange(100))
            )
            ax.plot(y.index, 100 * y, color=colors.get(setpoints[cycle], "#467B9B"), lw=0.85)
            ax.axhspan(99, 100, color="#DCE7ED", alpha=0.7, lw=0)
            ax.set(xlim=(0, 99), ylim=(0, 105), xticks=[0, 50, 99], yticks=[0, 100])
            ax.set_title(
                f"C{int(cycle.rsplit('_', 1)[-1])} · {setpoints[cycle]:g}°C",
                loc="left",
                fontsize=5.5,
                pad=3,
            )
            ax.tick_params(labelsize=4.5, length=2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        for ax in axes.ravel()[len(names) :]:
            ax.set_axis_off()
        fig.text(
            0.5,
            0.012,
            "横轴：霜层等级；纵轴：归一化 COP (%)；缺口留空，浅色带为 1% 近优范围",
            ha="center",
            fontsize=6,
            color="#52606C",
        )
        save_publication(
            fig,
            Path(output),
            "frost_cop_individual_cycles",
            "",
            formats=("png",),
            layout=False,
            dpi=600,
        )


def plot_frost_overlap_images(
    rows,
    summary,
    output,
    *,
    combined=False,
    name="maximum_overlap_frost_images",
    common_grade=False,
    source_label="Native Harmony + Slingshot",
):
    """All hit cycles, with actual observed grades and undistorted raw RGB frames."""
    from plots.publication import save_publication

    columns = 12 if common_grade else (14 if combined else 10)
    count = len(rows)
    grid_rows = int(np.ceil(count / columns))
    temperature = rows.setpoint_c.iloc[0]
    grade = int(rows.target_grade.iloc[0])
    ordered = rows.sort_values(["setpoint_c", "cycle_name"])
    if common_grade:
        groups = [
            (f"近优区间覆盖等级 {grade}", ordered.loc[ordered.interval_hit]),
            (
                f"近优区间未覆盖等级 {grade}",
                ordered.loc[~ordered.interval_hit & ordered.near_grade_available],
            ),
            ("近优时段霜层等级不可用 · 照片留空", ordered.loc[~ordered.near_grade_available]),
        ]
        groups = [(label, g) for label, g in groups if len(g)]
        heights = [0.35 + np.ceil(len(g) / columns) * 1.15 for _, g in groups]
        fig = plt.figure(figsize=(183 / 25.4, 1.05 + sum(heights)))
        outer = fig.add_gridspec(
            len(groups),
            1,
            left=0.035,
            right=0.985,
            bottom=0.02,
            top=1 - 0.95 / fig.get_figheight(),
            height_ratios=heights,
            hspace=0.08,
        )
        axes = []
        for i, (label, g) in enumerate(groups):
            nrows = int(np.ceil(len(g) / columns))
            panel = outer[i].subgridspec(
                nrows + 1, columns, height_ratios=[0.28] + [1] * nrows, hspace=0.42, wspace=0.17
            )
            header = fig.add_subplot(panel[0, :])
            header.set_axis_off()
            header.text(
                0,
                0.6,
                f"{chr(97 + i)}  {label}",
                transform=header.transAxes,
                fontsize=8,
                fontweight="bold",
                color="#243F53",
                va="center",
            )
            header.text(
                1,
                0.6,
                f"n = {len(g)}",
                transform=header.transAxes,
                fontsize=7,
                color="#52606C",
                ha="right",
                va="center",
            )
            for j in range(len(g)):
                axes.append(fig.add_subplot(panel[1 + j // columns, j % columns]))
        ordered = pd.concat([g for _, g in groups])
    else:
        fig, grid_axes = plt.subplots(
            grid_rows,
            columns,
            figsize=(183 / 25.4, 1.2 + grid_rows * (1.05 if combined else 1.3)),
            squeeze=False,
        )
        fig.subplots_adjust(
            left=0.035,
            right=0.985,
            bottom=0.025,
            top=1 - 1.1 / fig.get_figheight(),
            wspace=0.12,
            hspace=0.38 if combined else 0.23,
        )
        axes = grid_axes.ravel()
    heading = (
        "最大覆盖霜层等级附近的结霜图像"
        if common_grade
        else (f"全部筛选循环 · n = {count}" if combined else f"{temperature:g} °C")
    )
    fig.text(
        0.035,
        0.975,
        heading,
        fontsize=11 if common_grade else 12,
        fontweight="normal" if combined else "bold",
        color="#243F53",
        va="top",
    )
    stat = (
        f"{int(rows.interval_hit.sum())}/{count} 区间命中"
        if combined
        else f"{int(summary.maximum_hits)}/{int(summary.total_cycles)} cycles · {summary.maximum_overlap:.1%}"
    )
    if common_grade:
        total = (
            int(summary.total_cycles.sum())
            if isinstance(summary, pd.DataFrame)
            else int(summary.total_cycles)
        )
        hits = (
            int(summary.maximum_hits.sum())
            if isinstance(summary, pd.DataFrame)
            else int(summary.maximum_hits)
        )
        stat = f"{hits / total:.1%} · {hits}/{total}"
    fig.text(
        0.985,
        0.965,
        stat,
        fontfamily=plt.rcParams["font.family"] if combined else "Arial",
        fontsize=8,
        color="#52606C",
        va="top",
        ha="right",
    )
    targets = " · ".join(
        f"{ts:g}°C: {int(g.target_grade.iloc[0])}" for ts, g in rows.groupby("setpoint_c")
    )
    target_label = (
        f"共同霜层等级 {grade} · 展示 {count} 个循环"
        if common_grade
        else f"最大重叠等级 {targets if combined else grade} · 展示 {count} 个循环"
    )
    fig.text(
        0.035,
        0.975 - 0.28 / fig.get_figheight(),
        f"{target_label}  |  {source_label}",
        fontsize=7,
        color="#78858E",
        va="top",
    )
    note = (
        "每格为实际观测照片；Δg 为等级差，未命中循环与缺等级循环保留。"
        if combined
        else "每格标注照片实际等级；Δg 为照片等级 − 展示点等级。"
    )
    if common_grade:
        note = "g 为照片实际等级；Δg = g − 共同等级。照片保持原始比例，逆时针旋转 90°。"
    fig.text(
        0.035, 0.975 - 0.5 / fig.get_figheight(), note, fontsize=6.5, color="#78858E", va="top"
    )
    for ax, (_, row) in zip(axes, ordered.iterrows()):
        info = dict(
            image_path=row.image_path,
            available=row.available,
            status="physical image missing",
            target_time=row.candidate_defrost_time,
            image_time=row.image_time,
            offset_seconds=(
                pd.Timestamp(row.image_time) - pd.Timestamp(row.candidate_defrost_time)
            ).total_seconds(),
        )
        if not row.available or (common_grade and not row.near_grade_available):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("white")
            ax.set_box_aspect(1.75)
            if row.near_grade_available and not row.available:
                ax.text(
                    0.5,
                    0.5,
                    "磁盘缺图",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=4,
                    color="#78858E",
                )
        else:
            _plot_decision_image(ax, info, "", pd.NaT, pd.NaT)
            ax.set_aspect("equal")
        actual = row.observed_grade
        delta = actual - row.target_grade
        suffix = (
            ""
            if delta == 0 or pd.isna(delta)
            else f" · {delta:+g}"
            if combined
            else f" · Δg {delta:+g}"
        )
        state = "无等级" if pd.isna(actual) else f"g = {actual:g}{suffix}"
        title = (
            f"{row.setpoint_c:g}° · C{int(row.cycle_name.rsplit('_', 1)[-1])}\n{state}\n"
            + ("区间命中" if row.interval_hit else "未命中")
            if combined
            else f"Cycle {int(row.cycle_name.rsplit('_', 1)[-1])}\n{state}"
        )
        if common_grade:
            state = (
                "—"
                if pd.isna(actual)
                else f"g {actual:g}" + ("" if delta == 0 else f" · Δg {delta:+g}")
            )
            title = f"C{int(row.cycle_name.rsplit('_', 1)[-1])} · {row.setpoint_c:g}°C\n{state}"
        ax.set_title(
            title,
            loc="left",
            fontsize=5.2 if common_grade else (4.6 if combined else 5.4),
            pad=3,
            color="#243F53" if common_grade or delta == 0 else "#8C683E",
        )
        for spine in ax.spines.values():
            spine.set_color("#D7DFE5")
            spine.set_linewidth(0.45)
    for ax in axes[count:]:
        ax.set_axis_off()
    save_publication(
        fig,
        Path(output),
        "screened_cycles_frost_images"
        if combined and name == "maximum_overlap_frost_images"
        else name,
        "",
        formats=("png",),
        layout=False,
        dpi=600,
    )
