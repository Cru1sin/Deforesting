"""Geometry and construction of the native frost-state trajectory."""

from __future__ import annotations

from pathlib import Path

import matplotlib

from plots.frost_state import FROST_CLUSTER_COLORS
from plots.publication import _RGB_PANEL_FONT, save_publication

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def plot_cycle_frost_time(frames, output, *, source_label):
    """Unmodified image projections on the frozen global 0–1 coordinate."""
    from image_models.slingshot import monotonicity_summary

    frames = frames.sort_values(["image_time", "file_name"]).copy()
    times = pd.to_datetime(frames.image_time)
    elapsed = (times - times.iloc[0]).dt.total_seconds() / 60
    score = frames.visual_frost_score.copy()
    gaps = times.diff().dt.total_seconds().gt(60)
    # Break missing-image gaps without deleting the first observation after each gap.
    x = np.insert(elapsed.to_numpy(), np.flatnonzero(gaps), np.nan)
    y = np.insert(score.to_numpy(), np.flatnonzero(gaps), np.nan)
    metric = monotonicity_summary(frames).iloc[0]
    Path(output).mkdir(parents=True, exist_ok=True)
    frames[
        ["cycle_name", "image_time", "file_name", "visual_frost_score", "is_beyond_reference"]
    ].to_csv(Path(output) / "frost_state_time_source.csv", index=False)
    with plt.rc_context(
        {
            "font.family": _RGB_PANEL_FONT,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.plot(x, y, color="#377EAB", lw=1)
        ax.scatter(elapsed, score, s=3, color="#377EAB", alpha=0.45, linewidths=0)
        ax.set(
            xlabel="首张有效图像后的时间（min）", ylabel="连续视觉霜态坐标 s", ylim=(-0.02, 1.04)
        )
        ax.grid(axis="y", alpha=0.12)
        fig.text(
            0.13,
            0.96,
            f"{frames.cycle_name.iloc[0]}｜霜态是否随时间增长？",
            fontsize=10,
            color="#243F53",
        )
        fig.text(0.13, 0.89, source_label, fontsize=6.5, color="#52606C")
        fig.text(
            0.13,
            0.83,
            f"n={len(frames)} images · Spearman ρ={metric.time_spearman:.3f} · 最大回落={metric.max_drawdown:.3f}",
            fontsize=7,
            color="#52606C",
        )
        fig.text(
            0.13,
            0.025,
            "固定全局等级；未作时间单调修正。超过60 s图像缺口不连线；s=1为参考终点。",
            fontsize=6.5,
            color="#52606C",
        )
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.18, top=0.76)
        save_publication(
            fig, Path(output), "frost_state_time", "", formats=("png",), layout=False, dpi=600
        )


def plot_slingshot_geometry(
    points,
    centers,
    curve,
    edges,
    output,
    pcs=(1, 2),
    baseline=None,
    curve_label="Slingshot 主曲线",
    feature_label="DINOv2 CLS",
    n_dimensions=30,
    polyline=False,
    compact=False,
    baseline_label="原始默认平滑曲线",
    clusters_only=False,
    recipe=None,
):
    """Saved topology and fitted curve shown in the same two coordinate dimensions."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    recipe = recipe or {}

    with plt.rc_context(
        {
            "font.family": _RGB_PANEL_FONT,
            "font.size": 9,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#7F8A92",
            "axes.linewidth": 0.65,
            "xtick.color": "#7F8A92",
            "ytick.color": "#7F8A92",
            "axes.labelcolor": "#52616D",
        }
    ):
        fig, ax = plt.subplots(
            figsize=(7.2, 4.7) if compact else (5.6, 4.5 if baseline is not None else 4.25)
        )
        fig.subplots_adjust(
            left=0.09 if compact else 0.13,
            right=0.70 if compact else 0.97,
            bottom=(0.20 if baseline is not None else 0.17)
            if compact
            else (0.31 if baseline is not None else 0.26),
            top=0.82 if compact else 0.82,
        )
        comparison_color = "#BD7B5A" if compact else "#929BA4"
        point_colors = (
            [
                FROST_CLUSTER_COLORS[int(k) % len(FROST_CLUSTER_COLORS)]
                for k in points.get("color_group", points.get("display_cluster", points.cluster))
            ]
            if clusters_only or "kind" in centers
            else "#A9BBC9"
        )
        ax.scatter(
            points.x,
            points.y,
            s=1.2,
            color=point_colors,
            alpha=0.3 if clusters_only else 0.12,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        lookup = centers.set_index("cluster")
        for edge in [] if compact else edges.itertuples():
            pair = lookup.loc[[edge.parent, edge.child]]
            ax.plot(pair.x, pair.y, color="#BD9278", lw=1.35, zorder=3)
        if baseline is not None:
            ax.plot(
                baseline.x,
                baseline.y,
                color=comparison_color,
                lw=1.5,
                linestyle=(0, (4, 3)),
                zorder=3,
            )
        if not clusters_only:
            ax.plot(curve.x, curve.y, color="white", lw=3.8, zorder=4)
            ax.plot(curve.x, curve.y, color="#377EAB", lw=2.1, zorder=5)
        ax.scatter(
            centers.x,
            centers.y,
            s=34 if compact else 65,
            facecolors="white",
            edgecolors="#BD9278",
            linewidths=1.3,
            zorder=6,
        )
        root = centers.loc[centers.is_start].iloc[0]
        ax.scatter(
            [root.x],
            [root.y],
            s=50 if compact else 100,
            color="#54998F",
            edgecolors="white",
            linewidths=1.5,
            zorder=7,
        )
        for row in centers.itertuples():
            offset = (5, 6)
            if "kind" in centers and row.parent_order == centers.parent_order.max():
                local = list(
                    centers.loc[centers.parent_order.eq(row.parent_order), "cluster"]
                ).index(row.cluster)
                offset = (2, -20) if local % 2 == 0 else (2, 16)
                if row.kind == "terminal_anchor":
                    offset = (12, -18)
            ax.annotate(
                str(getattr(row, "display_label", getattr(row, "display_cluster", row.cluster))),
                (row.x, row.y),
                xytext=offset,
                textcoords="offset points",
                color="#344954",
                fontsize=8,
                arrowprops=(
                    dict(arrowstyle="-", color="#93A0A9", lw=0.5) if offset != (5, 6) else None
                ),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5),
                zorder=8,
            )
        # Direction follows the saved pseudotime, not the projected MST geometry.
        for fraction in [] if clusters_only else [0.25, 0.75]:
            ax.annotate(
                "",
                xy=(
                    np.interp(fraction + 0.025, curve.phase, curve.x),
                    np.interp(fraction + 0.025, curve.phase, curve.y),
                ),
                xytext=(
                    np.interp(fraction, curve.phase, curve.x),
                    np.interp(fraction, curve.phase, curve.y),
                ),
                arrowprops=dict(arrowstyle="-|>", color="#377EAB", lw=1.1, mutation_scale=10),
                zorder=9,
            )
        ax.set(
            xlabel=f"Harmony 校正后的 PC{pcs[0]} 坐标", ylabel=f"Harmony 校正后的 PC{pcs[1]} 坐标"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0.035 if compact else 0.09)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=8, length=3, width=0.6)
        if not compact:
            fig.text(
                0.13,
                0.94,
                "Harmony + MST：折线霜态轨迹" if polyline else "Harmony + Slingshot：连续霜态轨迹",
                fontsize=12,
                color="#213E54",
            )
            fig.text(
                0.13,
                0.875,
                f"{feature_label} · {points.cycle_name.nunique()} 个循环 · "
                f"{len(points):,} 张图像 · {len(centers)} 簇 · 单路径",
                fontsize=8,
                color="#7B8791",
            )
        handles = [
            Line2D(
                [],
                [],
                color="#BD9278",
                lw=1.5,
                marker="o",
                markerfacecolor="white",
                markersize=5,
                label="MST / 聚类中心",
            ),
            Line2D([], [], color="#377EAB", lw=2, label=curve_label),
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                color="#54998F",
                markersize=6,
                label=f"起点簇 {int(root.cluster)}",
            ),
        ]
        if baseline is not None:
            handles.insert(
                2,
                Line2D(
                    [], [], color=comparison_color, lw=1.5, linestyle="--", label=baseline_label
                ),
            )
        if compact:
            if not clusters_only:
                fig.legend(
                    handles=handles[1:-1],
                    loc="lower left",
                    bbox_to_anchor=(0.09, 0.015),
                    frameon=False,
                    fontsize=7.5,
                    handlelength=1.8,
                    labelspacing=0.45,
                )
            refined = "kind" in centers
            title = (
                (
                    "原始簇内如何划分更细的组？"
                    if clusters_only
                    else "簇内分组与尾部样本如何形成连续等级？"
                )
                if refined
                else (
                    "图像如何按特征相似度聚类？"
                    if clusters_only
                    else (
                        f"{len(centers)}个簇中心如何连接成初始折线？"
                        if polyline
                        else "簇连接顺序与 Slingshot 主曲线拟合"
                    )
                )
            )
            fig.text(0.09, 0.96, title, fontsize=11, color="#213E54", va="top")
            fig.text(
                0.09,
                0.905,
                f"仅展示 PCA 第{pcs[0]}、{pcs[1]}维经 Harmony 校正后的坐标；计算使用全部{n_dimensions}维。",
                fontsize=7.7,
                color="#52616D",
                va="top",
            )
            fig.text(
                0.09,
                0.86,
                f"每点代表1张图像｜{points.cycle_name.nunique()}个循环｜{len(points):,}张图像",
                fontsize=7.5,
                color="#52616D",
                va="top",
            )
            if refined:
                n_groups = int(centers.kind.eq("visual_stage").sum())
                target = recipe.get("subcluster_target_images", 2000)
                tail = centers.loc[centers.kind.eq("terminal_anchor")].iloc[0]
                fraction = 100 * (1 - recipe.get("terminal_quantile", 0.99))
                notes = [
                    (
                        "编号从哪里来？",
                        f"在原{centers.parent_order.nunique()}个 K-means 簇内，\n按初始折线投影位置排序。\n分组数 = n簇/{target}向上取整；\n按分位数切分，共{n_groups}组；\n相同投影位置不强行拆开。",
                    ),
                    (
                        "颜色和圆点",
                        f"同色 = 同一原始 K-means 簇。\n0–{n_groups - 1} = 簇内细分后的组号；\n圆点 = 各组{n_dimensions}维特征均值。\n组号不等于人工霜态标签。",
                    ),
                    (
                        "尾部样本中心",
                        f"最后一簇沿末段方向最靠后{fraction:g}%：\n{int(tail.image_count)}张图，{int(tail.cycle_count)}循环、{int(tail.experiment_count)}实验。\n先每循环取均值，再平均，\n作为延长折线后的终点。",
                    ),
                    (
                        "连续等级 s",
                        f"将图像投影到{n_dimensions}维折线上；\ns = 起点至投影点弧长 / 总弧长。\n范围固定；未强制随时间单调。",
                    ),
                ]
                ys = [0.81, 0.59, 0.39, 0.19]
            elif clusters_only:
                notes = [
                    (
                        "聚类算法",
                        f"K-means，K={len(centers)}。\n输入：全部{n_dimensions}维校正特征。\n按特征相似度分组，\n未按时间切分图像。",
                    ),
                    (
                        "图形含义",
                        f"颜色 = 图像所属的簇；\n空心圆 = 簇内{n_dimensions}维均值；\n绿色圆 = 首图投票的起点。",
                    ),
                    (
                        "编号含义",
                        f"原始 K-means 编号无先后。\n本图按连接顺序标为0–{len(centers) - 1}。\n这些簇不是人工霜态标签。",
                    ),
                ]
                ys = [0.81, 0.57, 0.36]
            else:
                notes = [
                    (
                        "最小生成树（MST）",
                        f"连接全部{len(centers)}个簇中心；\n共{len(centers) - 1}条边，无闭环，\n所有边的权重总和最小。\n本数据得到一条不分叉路径。",
                    ),
                    (
                        "怎样定义“近”？",
                        "采用 Slingshot 的中心距离：\n用两簇协方差之和校正差异；\n不是仅看图中的二维距离。",
                    ),
                    (
                        "怎样确定先后？",
                        f"每个循环的首图投1票；\n票数最多的簇作为起点。\n沿树编号0–{len(centers) - 1}；\n"
                        + (
                            "本版本使用折线，不拟合样条。"
                            if polyline
                            else "之后拟合 Slingshot 主曲线。"
                        ),
                    ),
                ]
                ys = [0.81, 0.56, 0.35]
            for (heading, body), y in zip(notes, ys):
                fig.text(
                    0.735, y, heading, va="top", fontsize=8.2, color="#344954", fontweight="bold"
                )
                fig.text(
                    0.735,
                    y - 0.043,
                    body,
                    va="top",
                    fontsize=7.3,
                    linespacing=1.35,
                    color="#52616D",
                )
        else:
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.54, 0.075),
                frameon=False,
                ncol=3 if baseline is None else 2,
                fontsize=8,
                handlelength=1.6,
                columnspacing=1.4,
            )
            fig.text(
                0.13,
                0.045,
                f"连接{'' if polyline else '与拟合'}在 {n_dimensions} 维空间完成；图中为相同坐标的二维投影。",
                fontsize=7.3,
                color="#7B8791",
            )
        filename = ("mst_center_polyline" if polyline else "slingshot_mst_principal_curve") + (
            "_comparison" if baseline is not None else ""
        )
        if "kind" in centers:
            filename = "mst_guided_refined_polyline" + (
                "_comparison" if baseline is not None else ""
            )
        if compact:
            filename += "_compact"
        if clusters_only:
            filename = (
                "mst_guided_visual_stages_compact"
                if "kind" in centers
                else "kmeans_clusters_compact"
            )
        save_publication(fig, Path(output), filename, "", formats=("png",), layout=False, dpi=600)


def plot_frost_construction(stages, variance, cases, output, *, chosen_clusters):
    """Cohort, retained PCA variance and cluster trade-off; three single panels."""
    from matplotlib.ticker import MaxNLocator

    style = {
        "font.family": _RGB_PANEL_FONT,
        "font.size": 8,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "axes.edgecolor": "#7F8A92",
        "axes.labelcolor": "#52616D",
        "xtick.color": "#7F8A92",
        "ytick.color": "#52616D",
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(4.8, 2.3))
        fig.subplots_adjust(left=0.38, right=0.98, bottom=0.20, top=0.98)
        ax.barh(
            np.arange(len(stages)),
            stages.cycles,
            height=0.58,
            color=["#AEBCC7"] * (len(stages) - 1) + ["#548F85"],
        )
        ax.set(yticks=np.arange(len(stages)), yticklabels=stages.stage, xlabel="循环数")
        ax.invert_yaxis()
        ax.set_xlim(0, stages.cycles.max() * 1.15)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        for row, value in enumerate(stages.cycles):
            ax.text(
                value + 5,
                row,
                str(value),
                va="center",
                fontsize=8,
                fontweight="bold" if row == len(stages) - 1 else "normal",
                color="#344954",
            )
        save_publication(
            fig, output, "cycle_selection", "", formats=("png",), layout=False, dpi=600
        )

        fig, ax = plt.subplots(figsize=(3.65, 2.65))
        fig.subplots_adjust(left=0.17, right=0.98, bottom=0.19, top=0.98)
        ax.plot(variance.pc, variance.cumulative_variance * 100, color="#377EAB", lw=1.8)
        ax.fill_between(
            variance.pc, 0, variance.cumulative_variance * 100, color="#DCE9EF", alpha=0.65
        )
        last = variance.iloc[-1]
        ax.scatter([last.pc], [last.cumulative_variance * 100], s=28, color="#548F85", zorder=3)
        ax.annotate(
            f"{int(last.pc)} PC：{last.cumulative_variance:.2%}",
            (last.pc, last.cumulative_variance * 100),
            xytext=(-8, -18),
            textcoords="offset points",
            ha="right",
            color="#344954",
            fontsize=8,
        )
        ax.set(
            xlabel="保留的主成分数",
            ylabel="累计解释方差（%）",
            xlim=(1, last.pc + 1),
            ylim=(0, 100),
        )
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        save_publication(
            fig, output, "pca_retained_variance", "", formats=("png",), layout=False, dpi=600
        )

        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.98)
        valid = cases.loc[cases.status.eq("evaluated")]
        means = valid.groupby("n_clusters")[
            ["late_mean_squared_distance", "cycles_drawdown_above_01"]
        ].median()
        ax.plot(
            means.late_mean_squared_distance,
            means.cycles_drawdown_above_01,
            color="#BBC5CB",
            lw=0.85,
            zorder=1,
        )
        for k, group in valid.groupby("n_clusters"):
            x, y = group.late_mean_squared_distance, group.cycles_drawdown_above_01
            color = "#548F85" if k == chosen_clusters else "#587D96"
            ax.errorbar(
                x.median(),
                y.median(),
                xerr=[[x.median() - x.min()], [x.max() - x.median()]],
                yerr=[[y.median() - y.min()], [y.max() - y.median()]],
                fmt="o",
                ms=6 if k == chosen_clusters else 4,
                capsize=2,
                color=color,
                lw=0.9,
                zorder=3,
            )
            offset = (8, 6) if k == chosen_clusters else ((-18, 10) if k == 9 else (6, 7))
            ax.annotate(
                f"K={k}",
                (x.median(), y.median()),
                xytext=offset,
                textcoords="offset points",
                fontsize=7.5,
                color=color,
                fontweight="bold" if k == chosen_clusters else "normal",
            )
        branch_counts = sorted(
            cases.loc[cases.status.eq("branched_stopped"), "n_clusters"].unique()
        )
        ax.text(
            0.98,
            0.96,
            "K=" + ", ".join(map(str, branch_counts)) + "：出现分叉",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.4,
            color="#7B8791",
        )
        ax.text(
            0.98,
            0.86,
            "5 个随机种子：中位数及范围",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.1,
            color="#7B8791",
        )
        ax.set(
            xlabel="后半段平方距离误差（30维）",
            ylabel="霜态回退 >0.1 的循环数",
            ylim=(0, means.cycles_drawdown_above_01.max() + 8),
        )
        ax.margins(x=0.13)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
        save_publication(
            fig, output, "cluster_count_tradeoff", "", formats=("png",), layout=False, dpi=600
        )


def plot_cluster_image_strip(representatives, output, *, recipe=None):
    """Nearest observed image within each cluster, full frame and one shared axis."""
    from plots.publication import _RGB_PANEL_FONT, save_publication

    recipe = recipe or {}

    with plt.rc_context({"font.family": _RGB_PANEL_FONT, "font.size": 8}):
        rows = int(np.ceil(len(representatives) / 7))
        columns = int(np.ceil(len(representatives) / rows))
        height = rows * 1.90 + 0.52
        width = (columns - 1) * 0.86 + 0.80
        fig, ax = plt.subplots(figsize=(7.2, 7.2 * height / width))
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.015, top=0.99)
        for j, row in enumerate(representatives.itertuples()):
            photo = np.rot90(plt.imread(row.image_path))
            left = (j % columns) * 0.86
            bottom = (rows - 1 - j // columns) * 1.90
            ax.imshow(
                photo, extent=(left, left + 0.7875, bottom + 0.27, bottom + 1.67), aspect="equal"
            )
            display_cluster = getattr(row, "display_cluster", row.path_order - 1)
            color = FROST_CLUSTER_COLORS[
                getattr(row, "color_group", display_cluster) % len(FROST_CLUSTER_COLORS)
            ]
            ax.add_patch(
                Rectangle(
                    (left, bottom + 0.27), 0.7875, 1.4, facecolor="none", edgecolor=color, lw=1.4
                )
            )
            ax.text(
                left + 0.39375,
                bottom + 1.73,
                getattr(row, "display_label", f"簇 {display_cluster}"),
                ha="center",
                va="bottom",
                color="#344954",
                fontsize=7.5,
            )
            ax.text(
                left + 0.39375,
                bottom + 0.16,
                f"s={row.visual_frost_score:.2f}",
                ha="center",
                va="center",
                color="#52616D",
                fontsize=7.5,
            )
            if hasattr(row, "group_image_count"):
                scope = "尾部" if getattr(row, "display_label", "") == "尾部样本中心" else "本组"
                ax.text(
                    left + 0.39375,
                    bottom + 0.045,
                    f"{scope} n={row.group_image_count:,}图 / {row.group_cycle_count}循环",
                    ha="center",
                    va="center",
                    color="#52616D",
                    fontsize=6.6,
                )
        ax.text(
            0,
            height - 0.06,
            "各组及尾部样本：展示距对应30维中心最近的一张实际照片；颜色表示原始簇。",
            va="top",
            fontsize=8.5,
            color="#213E54",
        )
        ax.text(
            0,
            height - 0.22,
            "s为照片的实际投影等级；图像选择与等级计算使用全部30维，非仅使用 PCA 前两维。",
            va="top",
            fontsize=7.4,
            color="#52616D",
        )
        ax.text(
            0,
            height - 0.36,
            "照片为完整原图，仅逆时针旋转90°；仅特征散点图展示经 Harmony 校正的 PCA 前两维。",
            va="top",
            fontsize=7.4,
            color="#52616D",
        )
        occupied = len(representatives) % columns
        if occupied and columns - occupied >= 2 and recipe.get("subcluster_target_images"):
            target = recipe["subcluster_target_images"]
            fraction = 100 * (1 - recipe.get("terminal_quantile", 0.99))
            ax.text(
                occupied * 0.86 + 0.03,
                1.68,
                f"组号从哪里来？\n原 K-means 簇内，按折线投影位置排序；\n每簇图数 / {target} 向上取整，\n再按分位数切分；同位置保持同组。\n组号不是人工定义的结霜阶段。\n\n连续等级 s = 累计弧长 / 总弧长。\n尾部中心：最后簇末段方向最高{fraction:g}%图像，\n先取各循环均值，再对循环均值平均。\n参考终点 s=1，不等于最大结霜程度。",
                va="top",
                fontsize=7.2,
                linespacing=1.5,
                color="#52616D",
            )
        ax.set(xlim=(-0.01, width), ylim=(0, height))
        ax.axis("off")
        name = (
            "visual_stage_representative_images"
            if "representation_method" in representatives
            and representatives.representation_method.eq("mst_guided_visual_segments").all()
            else "cluster_representative_images"
        )
        save_publication(fig, output, name, "", formats=("png",), layout=False, dpi=600)


def plot_frost_method(recipe, stages, output, *, counts):
    """One annotated method diagram; all counts come from the current saved model."""
    from matplotlib.patches import FancyBboxPatch

    groups = stages.loc[stages.kind.eq("visual_stage")]
    density = groups.groupby("parent_order").agg(
        n=("image_count", "sum"), m=("fine_cluster", "size")
    )
    tail = stages.loc[stages.kind.eq("terminal_anchor")].iloc[0]
    target = recipe["subcluster_target_images"]
    q = recipe["terminal_quantile"]
    cards = [
        (
            "图像 → 固定视觉特征",
            f"{counts['cycles']}个筛选循环，{counts['experiments']}个实验，{counts['images']:,}张正面结霜图像。\n"
            "每图取 DINOv2 patch 特征均值（384维），再作 L2 归一化。",
        ),
        (
            "PCA 降维 → Harmony 实验偏移校正",
            f"PCA 保留{recipe['pca_components']}维，累计解释方差 {counts['pca_variance']:.2%}。\n"
            "Harmony 以实验编号为批次，在软聚类内迭代估计并校正实验偏移；不输入 COP 或时间。",
        ),
        (
            "K-means：先得到7个原始簇",
            "在全部30维校正坐标中，按欧氏距离聚类；每个簇中心是簇内特征均值。\n"
            "最小化图像到所属中心的平方距离总和；原簇号不代表先后，按后续路径编号0–6。",
        ),
        (
            "Slingshot 最小生成树（minimum spanning tree, MST）：连接簇中心",
            "连接7个中心，共6条边、无闭环，并使边权总和最小；每循环首图投票确定起点。\n"
            r"$w_{ab}=(c_a-c_b)^T(\Sigma_a+\Sigma_b)^{-1}(c_a-c_b)$"
            + "（c为中心，Σ为簇内协方差）。\n"
            "本数据只有一条路径；连接中心得到初始折线，本版本不拟合平滑样条。",
        ),
        (
            "原簇内按折线投影位置分组：分组不是再次 K-means",
            f"每簇分组数 m = ceil(n簇/{target})（向上取整），按投影位置的分位数切分；同位置不强行拆开。\n"
            "端部排序沿相邻线段方向延长计算，不按图像时间切分。\n"
            "各原簇图数："
            + " / ".join(str(n) for n in density.n)
            + "；分组数："
            + " / ".join(str(m) for m in density.m)
            + f" → 共{len(groups)}组。",
        ),
        (
            "用实际尾部样本延长末端",
            f"在最后原始簇内，沿末段方向的位置 u ≥ Q{q:g}：{int(tail.image_count)}张图，{int(tail.cycle_count)}循环、{int(tail.experiment_count)}实验。\n"
            r"$u=(z-c_5)^T(c_6-c_5)/\|c_6-c_5\|_2^2$"
            + "；先计算各循环的尾部均值，再对循环均值平均。\n"
            "由此得到尾部样本中心，而不是取单张最远图像作为终点。",
        ),
        (
            "分组均值连线 → 连续视觉等级 s",
            f"按顺序连接{len(groups)}个组的30维均值中心，追加尾部样本中心；将特征 z 投影到该折线。\n"
            r"$s=\ell(\Pi_\gamma(z))/L_\gamma$"
            + "：累计弧长 / 总弧长；首组中心为0，尾部中心为1。\n"
            "折线和归一化范围保存后固定；不逐循环缩放，不强制等级随时间单调。",
        ),
        (
            "原图对应与末端边界",
            "每组展示距其30维均值最近的实际图像；所标 s 是该图像的实际投影等级。\n"
            "照片仅逆时针旋转90°，未裁剪或调整对比度；超过末端仍记 s=1，并另外标记超界。\n"
            "s 是视觉轨迹坐标，不是霜厚、霜质量或 COP；二维特征图仅展示校正后的 PCA 前两维。",
        ),
    ]
    with plt.rc_context({"font.family": _RGB_PANEL_FONT, "font.size": 8}):
        fig, ax = plt.subplots(figsize=(7.2, 7.4))
        fig.subplots_adjust(left=0.015, right=0.995, bottom=0.015, top=0.925)
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        fig.text(
            0.04,
            0.98,
            "从结霜图像到连续视觉等级：每一步如何计算？",
            va="top",
            fontsize=12,
            color="#213E54",
        )
        fig.text(
            0.04,
            0.945,
            "计算使用全部30维 Harmony 校正特征；时间只确定初始方向并用于结果检查。",
            va="top",
            fontsize=8,
            color="#52616D",
        )
        for i, (heading, body) in enumerate(cards):
            bottom = 0.88 - i * 0.123
            ax.add_patch(
                FancyBboxPatch(
                    (0.09, bottom),
                    0.90,
                    0.105,
                    boxstyle="round,pad=.004,rounding_size=.007",
                    facecolor="#F0F5F7" if i < 4 else "#F1F6F3",
                    edgecolor="#C7D4DA",
                    lw=0.65,
                )
            )
            ax.text(
                0.043,
                bottom + 0.055,
                str(i + 1),
                ha="center",
                va="center",
                fontsize=11,
                color="#377EAB",
                bbox=dict(boxstyle="circle,pad=.35", fc="#E2EDF2", ec="none"),
            )
            if i < len(cards) - 1:
                ax.annotate(
                    "",
                    xy=(0.043, bottom - 0.007),
                    xytext=(0.043, bottom + 0.024),
                    arrowprops=dict(arrowstyle="->", color="#94ABB7", lw=0.9),
                )
            ax.text(
                0.11,
                bottom + 0.091,
                heading,
                va="top",
                fontsize=8.7,
                color="#344954",
                fontweight="bold",
            )
            ax.text(
                0.11,
                bottom + 0.064,
                body,
                va="top",
                fontsize=7.4,
                linespacing=1.28,
                color="#52616D",
            )
        save_publication(
            fig, output, "frost_state_method", "", formats=("png",), layout=False, dpi=600
        )


def plot_observed_refinement(
    summary, stages, output, *, counts, chosen_stages=15, target_images=2000
):
    """Resolution, endpoint saturation and stage density as separate single panels."""
    from matplotlib.ticker import MaxNLocator

    with plt.rc_context(
        {
            "font.family": _RGB_PANEL_FONT,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#7F8A92",
            "axes.labelcolor": "#52616D",
        }
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        fig.subplots_adjust(left=0.10, right=0.985, bottom=0.16, top=0.70)
        fig.text(0.10, 0.98, "如何选择原始簇内的分组数量？", va="top", fontsize=11, color="#213E54")
        fig.text(
            0.10,
            0.925,
            f"同一批{counts['images']:,}张图像、{counts['cycles']}个循环；仅样本内比较，非独立测试。",
            va="top",
            fontsize=7.7,
            color="#52616D",
        )
        fig.text(
            0.10,
            0.875,
            r"$E_{\mathrm{late}}=\frac{1}{N_{\mathrm{late}}}\sum_{\tau_i\geq0.5}\|z_i-\Pi_\gamma(z_i)\|_2^2$"
            + "：图像到折线的30维几何误差，非 COP 预测误差。",
            va="top",
            fontsize=8,
            color="#52616D",
        )
        fig.text(
            0.10,
            0.825,
            r"$\tau_i=(t_i-t_{\mathrm{first},c})/(t_{\mathrm{last},c}-t_{\mathrm{first},c})$"
            + f"；每循环首张至末张有效图像的后50%时段，共{counts['late_images']:,}张图。",
            va="top",
            fontsize=7.7,
            color="#52616D",
        )
        fig.text(
            0.10,
            0.775,
            r"$D_c=\max_j(\max_{i\leq j}s_{ci}-s_{cj})$"
            + "；纵轴统计等级从此前最高值回落超过0.1的循环。",
            va="top",
            fontsize=7.7,
            color="#52616D",
        )
        for row in summary.itertuples():
            chosen = row.method == "refined" and row.fine_stages == chosen_stages
            color = "#548F85" if chosen else "#587D96"
            ax.scatter(
                row.late_mean_squared_distance,
                row.cycles_drawdown_above_01,
                s=42 if chosen else 24,
                color=color,
            )
            ax.annotate(
                row.label,
                (row.late_mean_squared_distance, row.cycles_drawdown_above_01),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=7.5,
                color=color,
            )
        ax.set(
            xlabel="循环后50%观测时段：图像到折线的平均平方距离（30维）",
            ylabel=f"等级回落 >0.1 的循环数（共{counts['cycles']}个）",
        )
        ax.margins(x=0.35, y=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        fig.text(
            0.10,
            0.02,
            "各方法按各自折线总长归一化；0.1是相对等级差，不是固定霜厚或固定弧长。",
            fontsize=7.2,
            color="#52616D",
        )
        save_publication(
            fig, output, "observed_refinement_tradeoff", "", formats=("png",), layout=False, dpi=600
        )

        fig, ax = plt.subplots(figsize=(7.2, 3.5))
        fig.subplots_adjust(left=0.11, right=0.985, bottom=0.20, top=0.72)
        fig.text(
            0.11,
            0.98,
            "延长末端后，多少图像仍被投影到终点？",
            va="top",
            fontsize=11,
            color="#213E54",
        )
        fig.text(
            0.11,
            0.905,
            f"末端饱和率 = 投影到终点（s=1）的图像数 / {counts['images']:,}；包含全部{counts['cycles']}个循环。",
            va="top",
            fontsize=7.7,
            color="#52616D",
        )
        fig.text(
            0.11,
            0.845,
            f"延长末端：最后簇的末段方向位置最高{counts['terminal_fraction']:g}%图像 → 各循环均值 → 尾部样本中心。",
            va="top",
            fontsize=7.7,
            color="#52616D",
        )
        fig.text(
            0.11,
            0.785,
            "各方法使用各自固定终点；s=1表示参考折线终点，不表示最大结霜程度。",
            va="top",
            fontsize=7.7,
            color="#52616D",
        )
        colors = [
            "#548F85" if r.method == "refined" and r.fine_stages == chosen_stages else "#B3C4CF"
            for r in summary.itertuples()
        ]
        ax.bar(np.arange(len(summary)), summary.endpoint_fraction * 100, width=0.58, color=colors)
        for i, row in enumerate(summary.itertuples()):
            ax.text(
                i,
                row.endpoint_fraction * 100 + 0.18,
                f"{row.endpoint_fraction:.2%}\nn={int(row.endpoint_images):,}",
                ha="center",
                fontsize=7.5,
            )
        labels = [
            "原方法\n7中心连线"
            if row.method == "original"
            else (
                "7中心连线\n延长末端"
                if row.fine_stages == 7
                else f"原簇内分{row.fine_stages}组\n延长末端"
            )
            for row in summary.itertuples()
        ]
        ax.set_xticks(np.arange(len(summary)), labels)
        ax.set(ylabel="末端饱和率（%）", ylim=(0, summary.endpoint_fraction.max() * 100 * 1.25))
        save_publication(
            fig, output, "endpoint_saturation", "", formats=("png",), layout=False, dpi=600
        )

        fig, ax = plt.subplots(figsize=(7.2, 3.5))
        fig.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.70)
        fig.text(
            0.10, 0.98, "原始 K-means 簇中的图像如何细分？", va="top", fontsize=11, color="#213E54"
        )
        fig.text(
            0.10,
            0.905,
            f"每簇分组数 m = ceil(n簇/{target_images})（向上取整）；按初始折线投影位置的分位数切分。",
            va="top",
            fontsize=7.7,
            color="#52616D",
        )
        fig.text(
            0.10,
            0.845,
            f"虚线为每组目标{target_images}张，并非硬上限：投影位置相同的图像不强行拆开。",
            va="top",
            fontsize=7.7,
            color="#52616D",
        )
        fig.text(
            0.10,
            0.785,
            f"同色来自同一原始簇；0–{chosen_stages - 1}是计算分组编号，不是人工定义的结霜阶段。",
            va="top",
            fontsize=7.7,
            color="#52616D",
        )
        groups = stages.loc[stages.kind.eq("visual_stage")]
        colors = [FROST_CLUSTER_COLORS[int(i)] for i in groups.parent_order]
        ax.bar(groups.fine_cluster, groups.image_count, color=colors, width=0.72)
        ax.axhline(target_images, color="#8A969E", lw=0.8, linestyle="--")
        ax.set_xticks(groups.fine_cluster)
        ax.set(xlabel="簇内按投影位置细分后的组号", ylabel="每组图像数量")
        for row in groups.itertuples():
            ax.text(
                row.fine_cluster,
                row.image_count + 40,
                f"{row.image_count:,}",
                ha="center",
                va="bottom",
                fontsize=6.4,
            )
        for parent, group in groups.groupby("parent_order"):
            ax.text(
                group.fine_cluster.mean(),
                groups.image_count.max() * 1.22,
                f"原簇{parent}\n分{len(group)}组",
                ha="center",
                va="top",
                fontsize=7,
                color=FROST_CLUSTER_COLORS[int(parent)],
            )
        ax.set_ylim(0, groups.image_count.max() * 1.26)
        save_publication(
            fig, output, "visual_stage_sample_counts", "", formats=("png",), layout=False, dpi=600
        )


def plot_harmony_coordinates(points, output, pcs=(1, 2), compact=False, n_dimensions=30):
    """Paired native coordinates; identical samples, colors, axes and draw order."""
    import matplotlib.dates as mdates
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import MaxNLocator

    dates = pd.to_datetime(points.experiment_id.str.removeprefix("exp_"), format="%Y%m%d")
    date_values = mdates.date2num(dates)
    norm = Normalize(date_values.min(), date_values.max())
    cmap = LinearSegmentedColormap.from_list(
        "experiment_dates", ["#425A9B", "#4B8FB5", "#75B9BD", "#C7C9AA", "#DCA976", "#C77868"]
    )
    order = np.random.default_rng(0).permutation(len(points))
    limits = []
    for pc in pcs:
        values = points[[f"raw_PC{pc}", f"corrected_PC{pc}"]].to_numpy()
        lo, hi = values.min(), values.max()
        pad = (hi - lo) * 0.06
        limits.append((lo - pad, hi + pad))
    variants = [
        ("raw", "PCA 特征空间", "pca_experiment_offsets"),
        ("corrected", "Harmony 校正后的特征空间", "harmony_corrected_space"),
    ]
    counts = (
        f"{points.cycle_name.nunique()} 个循环 · "
        f"{points.experiment_id.nunique()} 个实验 · {len(points):,} 张图像"
    )
    style = {
        "font.family": _RGB_PANEL_FONT,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#67737B",
        "axes.labelcolor": "#34434E",
        "xtick.color": "#67737B",
        "ytick.color": "#67737B",
        "axes.unicode_minus": False,
    }

    def draw(ax, variant, title, label=None):
        ax.scatter(
            points[f"{variant}_PC{pcs[0]}"].to_numpy()[order],
            points[f"{variant}_PC{pcs[1]}"].to_numpy()[order],
            c=date_values[order],
            cmap=cmap,
            norm=norm,
            s=2.2,
            alpha=0.32,
            linewidths=0,
            rasterized=True,
        )
        ax.set(
            xlim=limits[0],
            ylim=limits[1],
            xlabel=f"PC{pcs[0]}" if variant == "raw" else f"Harmony 校正后的 PC{pcs[0]} 坐标",
            ylabel=f"PC{pcs[1]}" if variant == "raw" else f"Harmony 校正后的 PC{pcs[1]} 坐标",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(length=3, width=0.6, labelsize=8)
        if not compact:
            ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=12)
        if label:
            ax.text(
                -0.17,
                1.065,
                label,
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                color="#172B3A",
            )

    def colorbar(fig, rect):
        bar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            cax=fig.add_axes(rect),
            orientation="vertical" if compact else "horizontal",
        )
        bar.set_ticks(np.linspace(date_values.min(), date_values.max(), 4))
        (bar.ax.yaxis if compact else bar.ax.xaxis).set_major_formatter(
            mdates.DateFormatter("%m/%d")
        )
        bar.ax.tick_params(length=2, width=0.5, labelsize=8)
        bar.outline.set_visible(False)
        bar.set_label(
            "实验日期" if compact else "实验日期（同一实验在两图中颜色一致）",
            fontsize=8,
            labelpad=4,
        )

    with plt.rc_context(style):
        for variant, title, name in variants:
            fig, ax = plt.subplots(figsize=(6.2, 4.2) if compact else (4.1, 3.85))
            fig.subplots_adjust(
                left=0.15 if compact else 0.18,
                right=0.82 if compact else 0.96,
                bottom=0.16 if compact else 0.31,
                top=0.76 if compact else 0.83,
            )
            draw(ax, variant, title)
            if compact:
                fig.text(
                    0.15,
                    0.975,
                    "校正前：PCA 降维后的图像特征"
                    if variant == "raw"
                    else "Harmony 校正后：按实验编号调整特征偏移",
                    va="top",
                    fontsize=10.5,
                    color="#213E54",
                )
                fig.text(
                    0.15,
                    0.915,
                    f"仅展示 PCA 第{pcs[0]}、{pcs[1]}维"
                    + ("。" if variant == "raw" else "经 Harmony 校正后的坐标。"),
                    va="top",
                    fontsize=7.8,
                    color="#52616D",
                )
                fig.text(
                    0.15,
                    0.865,
                    f"Harmony 与后续聚类均使用全部{n_dimensions}维；两图样本、颜色和坐标范围完全一致。",
                    va="top",
                    fontsize=7.8,
                    color="#52616D",
                )
                fig.text(
                    0.15,
                    0.815,
                    f"{points.cycle_name.nunique()}个循环｜{points.experiment_id.nunique()}个实验｜{len(points):,}张图；颜色代表实验日期。",
                    va="top",
                    fontsize=7.8,
                    color="#52616D",
                )
            else:
                fig.text(0.18, 0.88, counts, fontsize=7.5, color="#74818A")
            colorbar(fig, [0.855, 0.19, 0.025, 0.54] if compact else [0.24, 0.16, 0.66, 0.022])
            save_publication(fig, Path(output), name, "", formats=("png",), layout=False, dpi=600)
        if compact:
            return
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.85))
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.31, top=0.83, wspace=0.30)
        for ax, (variant, title, _), label in zip(axes, variants, ["a", "b"]):
            draw(ax, variant, title, label)
        fig.text(0.09, 0.92, counts, fontsize=8, color="#74818A")
        colorbar(fig, [0.28, 0.15, 0.48, 0.023])
        save_publication(
            fig, Path(output), "pca_harmony_comparison", "", formats=("png",), layout=False, dpi=600
        )


def plot_representation_audit(
    diagnostics, ranges, support, edges, output, pcs, *, space_label="Raw PCA"
):
    """Numerical batch/state evidence; no image pixels are loaded."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["pdf.fonttype"] = 42
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    rho = diagnostics.median_abs_cycle_spearman
    axes[0].scatter(rho, diagnostics.experiment_eta2, color="#4779a4", s=30)
    for row in diagnostics.itertuples():
        if (
            row.pc == 9
            or row.pc in pcs
            or (row.experiment_eta2 > 0.5 and row.median_abs_cycle_spearman < 0.2)
        ):
            axes[0].annotate(
                f"PC{row.pc}",
                (row.median_abs_cycle_spearman, row.experiment_eta2),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axes[0].set(
        xlabel="Median within-cycle |Spearman(time, PC)|",
        ylabel="Experiment η² (equal total weight per cycle)",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        title="a  Temporal signal versus experiment association",
    )
    axes[1].plot(
        diagnostics.pc,
        diagnostics.experiment_eta2,
        "o-",
        ms=3,
        color="#b36a24",
        label="Experiment η²",
    )
    axes[1].plot(
        diagnostics.pc,
        diagnostics.experiment_partial_r2_after_time,
        "o-",
        ms=3,
        color="#4779a4",
        label="Partial R² after relative time",
    )
    axes[1].plot(
        diagnostics.pc,
        diagnostics.experiment_partial_r2_after_reference,
        "o-",
        ms=3,
        color="#687e54",
        label="Partial R² after existing reference",
    )
    axes[1].set(
        xlabel="PCA component",
        ylabel="Explained fraction",
        ylim=(-0.02, 1.02),
        title="b  Does association remain after state adjustment?",
    )
    axes[1].legend(fontsize=8, loc="best")
    y = np.arange(len(edges))
    for field, offset, color in [
        ("euclidean", -0.16, "#4779a4"),
        ("mahalanobis_squared", 0.16, "#b36a24"),
    ]:
        axes[2].barh(
            y + offset,
            edges[field] / edges[field].median(),
            height=0.3,
            color=color,
            label=field.replace("_", " "),
        )
    axes[2].set_yticks(y, [f"{r.parent} → {r.child}" for r in edges.itertuples()])
    axes[2].set(
        xlabel="Edge length / median edge length", title="c  Is any MST connection exceptional?"
    )
    axes[2].legend(fontsize=8)
    fig.suptitle(
        f"{space_label} audit: association is evidence, not a watermark causal test", fontsize=13
    )
    for ext in ("png", "pdf"):
        fig.savefig(Path(output) / f"pca_batch_diagnostics.{ext}", dpi=200)
    plt.close(fig)

    fields = ["t_rel"] + [f"PC{pc}" for pc in pcs] + ["existing_reference"]
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(17, max(8, len(ranges) * 0.28)),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 2]},
        layout="constrained",
    )
    y = np.arange(len(ranges))
    for ax, field in zip(axes[:4], fields):
        for k, row in enumerate(ranges.to_dict("records")):
            ax.plot([row[f"{field}_min"], row[f"{field}_max"]], [k, k], color="#d9dfe5", lw=2)
            ax.plot(
                [row[f"{field}_q05"], row[f"{field}_q95"]],
                [k, k],
                color="#4779a4",
                lw=4,
                solid_capstyle="butt",
            )
            ax.plot(row[f"{field}_median"], k, "|", color="#252525", ms=7)
        ax.set_title(field.replace("existing_reference", "Existing visual reference"))
    axes[0].set_yticks(
        y,
        [f"{r.experiment_id.removeprefix('exp_')} ({r.cycle_count})" for r in ranges.itertuples()],
        fontsize=8,
    )
    axes[0].set_ylabel("Experiment (number of cycles)")
    axes[0].set_xlim(0, 1)
    axes[3].set_xlim(0, 1)
    matrix = support.pivot(
        index="experiment_id", columns="state_bin", values="cycle_count"
    ).reindex(ranges.experiment_id)
    heat = axes[4].imshow(matrix.to_numpy(), aspect="auto", cmap="Blues", origin="upper")
    axes[4].grid(False)
    for row in range(len(matrix)):
        for column in range(10):
            value = int(matrix.iloc[row, column])
            axes[4].text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > matrix.to_numpy().max() * 0.55 else "#252525",
            )
    axes[4].set_xticks(range(10), [f"{i / 10:.1f}" for i in range(10)], fontsize=8)
    axes[4].set_xlabel("Observed reference bin lower edge (last bin includes 1)")
    axes[4].set_title("Actual cycles with observations in each state bin")
    axes[4].set_ylim(len(ranges) - 0.5, -0.5)
    fig.colorbar(heat, ax=axes[4], shrink=0.4, label="Cycle count")
    fig.suptitle(
        f"{space_label}: is shared frost-state support plausible across experiments?\n"
        "Gray: observed min–max; blue: image 5–95%; tick: median. Existing reference is a proxy, not ground truth.",
        fontsize=13,
    )
    for ext in ("png", "pdf"):
        fig.savefig(Path(output) / f"experiment_state_support.{ext}", dpi=200)
    plt.close(fig)
