"""Model comparisons, component ablations, and training evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from plots.publication import _RGB_PANEL_FONT, save_publication

matplotlib.use("Agg")


def render_stopping_ablation(source_csv):
    """One dimension per figure, drawn only from its same-name source CSV."""
    source_csv = Path(source_csv)
    table = pd.read_csv(source_csv)
    main = table.loc[table.subset.eq("all") & table.setpoint_c.eq("all")].copy()
    main = (
        main.sort_values("variant_order")
        if "variant_order" in main
        else main.sort_values("baseline", ascending=False, kind="stable")
    )
    order = main.version.tolist()
    n = len(order)
    height = max(3.9, 0.24 * n + 2.1)
    with plt.rc_context(
        {
            "font.family": _RGB_PANEL_FONT,
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(
            1, 3, figsize=(7.2, height), gridspec_kw={"width_ratios": [1.5, 1.0, 1.15]}
        )
        positions = np.arange(n)
        colors = ["#353C45" if baseline else "#789DA8" for baseline in main.baseline]
        axes[0].scatter(main.headroom_captured_pct, positions, c=colors, s=24, zorder=3)
        reference = main.loc[main.baseline, "headroom_captured_pct"].iloc[0]
        axes[0].axvline(reference, color="#8A9199", lw=0.7, ls="--")
        labels = [f"{row.version}  ({int(row.parameters):,})" for row in main.itertuples()]
        axes[0].set(
            yticks=positions, yticklabels=labels, xlabel="$H_{RB}$ [%]", title="a  效率提升空间捕获"
        )
        for index, row in enumerate(main.itertuples()):
            if not row.baseline and np.isfinite(row.experiment_equal_delta_cop):
                axes[1].plot(
                    [row.bootstrap_lower_95, row.bootstrap_upper_95],
                    [index, index],
                    color="#789DA8",
                    lw=1.2,
                )
                axes[1].plot(row.experiment_equal_delta_cop, index, "o", color="#456B78", ms=3)
        axes[1].axvline(0, color="#8A9199", lw=0.7, ls="--")
        axes[1].set(
            xlabel="相对基准 ΔCOP", title="b  配对差异与95%区间", yticks=positions, yticklabels=[]
        )
        condition = table.loc[table.subset.eq("all") & table.setpoint_c.ne("all")].copy()
        condition["setpoint_c"] = pd.to_numeric(condition.setpoint_c)
        for offset, temp, color in [
            (-0.16, 50, "#A0B6C9"),
            (0, 55, "#A5A0BF"),
            (0.16, 60, "#BCAA94"),
        ]:
            g = condition.loc[condition.setpoint_c.eq(temp)].set_index("version").reindex(order)
            axes[2].scatter(
                g.headroom_captured_pct, positions + offset, color=color, s=15, label=f"{temp}℃"
            )
        axes[2].set(xlabel="$H_{RB}$ [%]", title="c  各水温工况", yticks=positions, yticklabels=[])
        axes[2].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            fontsize=6,
            handlelength=1.0,
            handletextpad=0.3,
            columnspacing=0.7,
        )
        for ax in axes:
            ax.set_ylim(n - 0.5, -0.6)
            ax.grid(axis="x", alpha=0.14, lw=0.5)
        best = main.loc[main.headroom_captured_pct.idxmax()]
        caption = (
            f"问题：{table.dimension.iloc[0]}是否改变留出策略表现？\n"
            f"证据：同一101循环、29实验日LOEO，seed {int(main.seed.iloc[0])}；括号为可训练参数量。\n"
            "b：共同可评价循环的实验日等权ΔCOP；2000次实验块bootstrap的95%区间，并非种子波动。\n"
            f"点估计最高：{best.version}，H_RB={best.headroom_captured_pct:.2f}%；最高点不代表稳定改进。\n"
            f"可评价覆盖：{int(main.policy_cycles.min())}–101/101；命中率、COP增益和延后时间见同名CSV。"
        )
        if table.dimension.iloc[0] == "触发规则":
            caption += "\n固定CDF训练权重，仅替换回放规则；不代表各规则分别重训后的最佳性能。"
        if table.dimension.iloc[0] == "层数":
            caption += "\n固定每层宽度；增加层数同时增加参数量，此处不是参数量匹配的深度比较。"
        if table.dimension.iloc[0] == "循环单元":
            caption += "\nLSTM32同宽度；LSTM25近似匹配总参数量（+2.13%）；无循环记忆仍保留运行时间与累计CDF。"
        if table.dimension.iloc[0] == "逐时刻模型":
            caption += (
                "\nMLP两层各84单元，参数量接近GRU32（+0.37%）；Logistic无dropout；两者保留累计CDF。"
            )
        if table.dimension.iloc[0] == "MLP的FiLM":
            caption += "\n同一两层84单元MLP，仅删除FiLM；其余输入、初始化、dropout、训练和触发规则保持一致。"
        fig.subplots_adjust(left=0.28, right=0.99, top=0.9, bottom=0.22, wspace=0.42)
        save_publication(fig, source_csv.parent, source_csv.stem, caption, layout=False, dpi=600)


def render_model_comparison(source_csv, layout="ranked", metric="H_RB"):
    """One ranked architecture comparison with aligned parameter and utility columns."""
    source_csv = Path(source_csv)
    source = pd.read_csv(source_csv).sort_values("H_RB", ascending=False).reset_index(drop=True)
    repeated = "H_RB_std" in source
    context = {
        "font.family": "Arial",
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
    }
    palette = {
        "Linear": "#A7B8CC",
        "MLP": "#91A5D8",
        "GRU": "#72B9B2",
        "LSTM": "#B39AD6",
        "RNN": "#D4AE91",
        "Attention": "#85B6D2",
    }
    if layout == "waterfall":
        source = source.sort_values(metric).reset_index(drop=True)
        source["waterfall_base"] = source[metric].shift(fill_value=0)
        source["waterfall_increment"] = source[metric] - source.waterfall_base
        source["waterfall_metric"] = metric
        assert np.isclose(source.waterfall_increment.sum(), source[metric].iloc[-1])
        source.to_csv(source_csv.with_name(source_csv.stem + "_瀑布分解.csv"), index=False)
        std_field = "H_RB_std" if metric == "H_RB" else "COP_gain_pct_std"
        seed_field = "H_RB_seeds" if metric == "H_RB" else "COP_gain_seeds"
        spread = source[std_field].fillna(0)
        lower = source[metric].min() - (source[metric].max() - source[metric].min()) * 0.08
        extent = (source[metric] + spread).max() - lower
        upper = lower + extent * 1.23
        offset = extent * 0.035
        decimals = 2 if metric == "H_RB" else 3
        with plt.rc_context(context):
            fig, ax = plt.subplots(
                figsize=(max(175, min(240, 35 * (len(source) + 1))) / 25.4, 84 / 25.4)
            )
            fig.subplots_adjust(left=0.105, right=0.99, bottom=0.245, top=0.84)
            for i, row in enumerate(source.itertuples()):
                multi = row.seeds > 1
                color = palette[row.family]
                score = getattr(row, metric)
                ax.bar(
                    i,
                    row.waterfall_increment,
                    bottom=row.waterfall_base,
                    width=0.66,
                    color=color,
                    alpha=1 if multi else 0.58,
                    zorder=2,
                )
                ax.plot([i + 0.33, i + 0.67], [score, score], color="#A9AFB7", lw=0.7, zorder=1)
                if multi:
                    ax.errorbar(
                        i,
                        score,
                        yerr=getattr(row, std_field),
                        fmt="none",
                        ecolor="#303640",
                        capsize=3,
                        lw=1,
                        zorder=4,
                    )
                    for xoffset, value in zip(
                        [-0.13, 0, 0.13], json.loads(getattr(row, seed_field)), strict=True
                    ):
                        ax.scatter(
                            i + xoffset,
                            value,
                            s=12,
                            facecolors="white",
                            edgecolors="#303640",
                            lw=0.7,
                            zorder=5,
                        )
                top = score + (getattr(row, std_field) if multi else 0)
                value = (
                    f"{score:.{decimals}f}"
                    if i == 0
                    else f"+{row.waterfall_increment:.{decimals}f} pp\n{score:.{decimals}f}"
                )
                ax.text(
                    i,
                    top + offset,
                    value,
                    ha="center",
                    fontsize=6.5,
                    fontweight="bold" if multi else "normal",
                )
            last = source.iloc[-1]
            ax.bar(
                len(source),
                last[metric],
                width=0.66,
                color=palette[last.family],
                alpha=1 if last.seeds > 1 else 0.58,
                zorder=2,
            )
            ax.text(
                len(source),
                last[metric] + offset,
                f"{last[metric]:.{decimals}f}",
                ha="center",
                fontsize=7.5,
                fontweight="bold",
            )
            for i in [0, len(source)]:
                for y in [lower + extent * 0.04, lower + extent * 0.065]:
                    ax.plot(
                        [i - 0.15, i + 0.15],
                        [y - extent * 0.01, y + extent * 0.01],
                        color="white",
                        lw=1.2,
                        zorder=6,
                    )
            labels = [
                row.label.replace(" + ", " +\n").replace(" (", "\n(")
                + ("\n3 seeds" if row.seeds > 1 else "\n1 seed")
                + f"\n{row.parameters / 1000:.1f}k | +{row.COP_gain_pct:.3f}%"
                for row in source.itertuples()
            ]
            labels += [
                "Highest score\n"
                + last.label.replace(" + ", " +\n").replace(" (", "\n(")
                + "\nTotal"
            ]
            ax.set(
                xticks=range(len(labels)),
                xticklabels=labels,
                ylim=(lower, upper),
                ylabel="Ideal improvement captured, $H_{RB}$ (%)"
                if metric == "H_RB"
                else "COP gain over RB (%)",
            )
            ax.tick_params(axis="x", length=0, pad=7, labelsize=5.8)
            ax.grid(axis="y", color="#E0E3E8", lw=0.5, zorder=0)
            if metric == "H_RB":
                ax.axhline(80, color="#89919A", ls=(0, (3, 3)), lw=0.7, zorder=1)
            title = (
                (
                    "Ideal improvement captured in RB early-defrost cycles"
                    if source_csv.stem.startswith("RB早除霜")
                    else "Typical model performance waterfall"
                )
                if metric == "H_RB"
                else "COP improvement in RB early-defrost cycles"
            )
            fig.text(0.06, 0.955, title, fontsize=10, fontweight="bold")
            fig.text(
                0.06,
                0.90,
                f"Frozen DINOv2 + 7 numerical inputs • {int(source.cycles.iloc[0])} cycles • {int(source.experiments.iloc[0])} experiment-level LOEO folds • CDF threshold 0.5",
                fontsize=7,
            )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for text in fig.findobj(match=lambda artist: isinstance(artist, plt.Text)):
                if text.get_visible() and text.get_text():
                    box = text.get_window_extent(renderer)
                    assert (
                        box.x0 >= -1
                        and box.y0 >= -1
                        and box.x1 <= fig.bbox.width + 1
                        and box.y1 <= fig.bbox.height + 1
                    ), text.get_text()
            save_publication(
                fig,
                source_csv.parent,
                source_csv.stem,
                None,
                layout=False,
                dpi=600,
                formats=("png", "pdf", "svg", "tiff"),
            )
        return
    with plt.rc_context(context):
        fig = plt.figure(figsize=(183 / 25.4, (78 if repeated else 128) / 25.4))
        axes = [
            fig.add_axes(rect)
            for rect in (
                [0.405, 0.19, 0.245, 0.68],
                [0.682, 0.19, 0.09, 0.68],
                [0.80, 0.19, 0.17, 0.68],
            )
        ]
        h, params, cop = axes
        n = len(source)
        for ax in axes:
            ax.set(ylim=(n - 0.5, -0.5), yticks=[])
            for i in range(n):
                if i % 2 == 0:
                    ax.axhspan(i - 0.5, i + 0.5, color="#F3F4F7", zorder=0, lw=0)
            ax.spines["left"].set_visible(False)
        h.set(xlim=(65, 87), xticks=[65, 70, 75, 80, 85], xlabel="$H_{RB}$ (%)")
        h.grid(axis="x", color="#DDE0E7", lw=0.5, zorder=1)
        cop.set(xlim=(0.98, 1.28), xticks=[1.0, 1.1, 1.2], xlabel="COP gain over RB (%)")
        cop.grid(axis="x", color="#DDE0E7", lw=0.5, zorder=1)
        params.set(xlim=(0, 1))
        params.axis("off")
        for i, row in enumerate(source.itertuples()):
            color = palette[row.family]
            multi = repeated and pd.notna(row.H_RB_std)
            y = axes[0].get_position().y1 - (i + 0.5) / n * axes[0].get_position().height
            fig.text(
                0.025,
                y,
                row.label,
                va="center",
                fontsize=7,
                fontweight="normal"
                if repeated
                else ("bold" if row.family == "MLP" and "FiLM" in row.label else "normal"),
            )
            fig.text(0.385, y, "●", ha="center", va="center", fontsize=8, color=color)
            if multi:
                h.errorbar(
                    row.H_RB,
                    i,
                    xerr=row.H_RB_std,
                    fmt="none",
                    ecolor=color,
                    capsize=3,
                    lw=1.2,
                    zorder=2,
                )
                cop.errorbar(
                    row.COP_gain_pct,
                    i,
                    xerr=row.COP_gain_pct_std,
                    fmt="none",
                    ecolor=color,
                    capsize=3,
                    lw=1.2,
                    zorder=2,
                )
                for field, ax in [("H_RB_seeds", h), ("COP_gain_seeds", cop)]:
                    for offset, value in zip(
                        [-0.16, 0, 0.16], json.loads(getattr(row, field)), strict=True
                    ):
                        ax.scatter(
                            value,
                            i + offset,
                            s=13,
                            facecolors="white",
                            edgecolors=color,
                            linewidths=0.8,
                            zorder=3,
                        )
            h.scatter(row.H_RB, i, s=30, color=color, edgecolors="white", linewidths=0.6, zorder=3)
            h.text(
                row.H_RB + (row.H_RB_std + 0.55 if multi else 0.55),
                i,
                f"{row.H_RB:.2f}",
                va="center",
                fontsize=6.7,
                color="#323741",
            )
            params.text(
                0.5,
                i,
                f"{row.parameters / 1000:.1f}k" if row.parameters >= 1000 else str(row.parameters),
                ha="center",
                va="center",
                fontsize=6.7,
                color="#4D5460",
            )
            cop.scatter(
                row.COP_gain_pct, i, s=25, color=color, edgecolors="white", linewidths=0.6, zorder=3
            )
            cop.text(
                row.COP_gain_pct + (row.COP_gain_pct_std + 0.008 if multi else 0.008),
                i,
                f"{row.COP_gain_pct:.3f}",
                va="center",
                fontsize=6.4,
                color="#323741",
            )
        fig.text(0.025, 0.925, "Model architecture", fontsize=8, fontweight="bold")
        fig.text(
            0.527, 0.925, "Energy headroom captured", ha="center", fontsize=7.5, fontweight="bold"
        )
        fig.text(0.727, 0.925, "Parameters", ha="center", fontsize=7.5, fontweight="bold")
        fig.text(0.885, 0.925, "Energy efficiency", ha="center", fontsize=7.5, fontweight="bold")
        fig.text(
            0.025,
            0.065,
            "Frozen DINOv2 + 7 numerical inputs · 101 cycles · 29 experiment-level LOEO folds",
            fontsize=6.3,
        )
        fig.text(
            0.025,
            0.035,
            "CDF ramp loss; first CDF ≥ 0.5. "
            + (
                "Filled: mean ± sample SD; open: individual runs."
                if repeated
                else "Points show estimates. Architecture labels give hidden widths."
            ),
            fontsize=6.3,
        )
        fig.text(
            0.025,
            0.008,
            (
                "Common inner-validation epoch selection; development comparison. LSTM is most stable; mean differences are small."
                if repeated
                else "Development comparison; parameter budgets and training recipes differ. Memory reset retains elapsed time and CDF."
            ),
            fontsize=6.1,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for text in fig.findobj(match=lambda artist: isinstance(artist, plt.Text)):
            if text.get_visible() and text.get_text():
                bounds = text.get_window_extent(renderer)
                assert (
                    bounds.x0 >= -1
                    and bounds.y0 >= -1
                    and bounds.x1 <= fig.bbox.width + 1
                    and bounds.y1 <= fig.bbox.height + 1
                ), text.get_text()
        save_publication(
            fig,
            source_csv.parent,
            source_csv.stem,
            None,
            layout=False,
            dpi=600,
            formats=("png", "pdf", "svg", "tiff"),
        )


def render_stopping_evidence(output):
    """Fig. 7-style evidence chain; every panel reads its own saved numerical source."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    output = Path(output)
    cases = pd.read_csv(output / "case_selection.csv")
    predictions = pd.read_csv(output / "panel_a_predictions.csv")
    metrics = pd.read_csv(output / "cycle_results.csv")
    models = pd.read_csv(output / "panel_b_models.csv")
    components = pd.read_csv(output / "panel_c_components.csv")
    learning = pd.read_csv(output / "panel_d_learning.csv")
    predictions["candidate_defrost_time"] = pd.to_datetime(
        predictions.candidate_defrost_time, format="mixed"
    )
    colors = {"MLP64x64_FiLM": "#789DA8", "GRU32_FiLM": "#C59273", "LSTM25_FiLM": "#8176AD"}
    names = {
        "MLP64x64_FiLM": "MLP + FiLM (64–64)",
        "GRU32_FiLM": "GRU + FiLM (32)",
        "LSTM25_FiLM": "LSTM + FiLM (25)",
    }
    context = {
        "font.family": "Arial",
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
    }
    with plt.rc_context(context):
        fig = plt.figure(figsize=(200 / 25.4, 148 / 25.4))
        grid = fig.add_gridspec(
            2,
            2,
            left=0.085,
            right=0.98,
            bottom=0.09,
            top=0.87,
            width_ratios=[1.35, 1],
            height_ratios=[1.15, 1],
            hspace=0.49,
            wspace=0.38,
        )
        fig.text(
            0.055,
            0.965,
            "LSTM-based visual defrost control",
            fontsize=11,
            weight="bold",
            color="#343345",
        )
        fig.text(
            0.055,
            0.935,
            "Frozen DINOv2 + 7 numerical inputs · 101 cycles · 29 experiment-level LOEO folds · CDF threshold 0.5",
            fontsize=7,
            color="#64646E",
        )

        def panel(ax, letter, title, subtitle):
            ax.text(-0.08, 1.09, letter, transform=ax.transAxes, fontsize=11, weight="bold")
            ax.text(0, 1.09, title, transform=ax.transAxes, fontsize=8, weight="bold")
            ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=6.3, color="#686873")

        mini = grid[0, 0].subgridspec(2, 2, height_ratios=[1.5, 1], hspace=0.12, wspace=0.25)
        top = []
        for j, case in enumerate(cases.itertuples()):
            ax = fig.add_subplot(mini[0, j])
            cdf = fig.add_subplot(mini[1, j], sharex=ax)
            top.append(ax)
            trace = predictions.loc[predictions.cycle_name.eq(case.cycle_name)]
            scores = metrics.loc[metrics.cycle_name.eq(case.cycle_name)]
            origin = pd.to_datetime(scores.cycle_start.iloc[0])
            curve = trace.drop_duplicates("candidate_defrost_time").sort_values(
                "candidate_defrost_time"
            )
            t = (curve.candidate_defrost_time - origin).dt.total_seconds() / 60
            ax.plot(t, curve.cycle_cop, color="#4F7288", lw=1.25)
            near = curve.near_optimal.fillna(False).to_numpy(bool)
            for axis in [ax, cdf]:
                for left, right in zip(
                    np.flatnonzero(np.diff(np.r_[False, near, False].astype(int)) == 1),
                    np.flatnonzero(np.diff(np.r_[False, near, False].astype(int)) == -1),
                ):
                    axis.axvspan(
                        t.iloc[left] - 0.25,
                        t.iloc[right - 1] + 0.25,
                        color="#D8EBDD",
                        alpha=0.8,
                        lw=0,
                        zorder=0,
                    )
                axis.grid(axis="y", alpha=0.12, lw=0.5)
            selected = pd.to_datetime(scores.selected_time, format="mixed")
            triggers = (selected - origin).dt.total_seconds() / 60
            rb = (pd.to_datetime(scores.t_RB.iloc[0]) - origin).total_seconds() / 60
            for axis in [ax, cdf]:
                axis.axvline(rb, color="#B18A65", ls="--", lw=0.9)
                axis.axvline(triggers.median(), color=colors["LSTM25_FiLM"], lw=1.0)
            q = trace.groupby("candidate_defrost_time").probability.agg(["median", "min", "max"])
            qt = (q.index - origin).total_seconds() / 60
            cdf.fill_between(qt, q["min"], q["max"], color=colors["LSTM25_FiLM"], alpha=0.18, lw=0)
            cdf.plot(qt, q["median"], color=colors["LSTM25_FiLM"], lw=1.25)
            cdf.axhline(0.5, color="#A4A3AA", ls=":", lw=0.8)
            cdf.scatter(
                triggers, [0.5] * len(triggers), s=10, color=colors["LSTM25_FiLM"], zorder=5
            )
            cdf.set(ylim=(-0.025, 1.025), yticks=[0, 0.5, 1], xlabel="Elapsed cycle time [min]")
            cdf.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.set_ylabel("Effective COP" if j == 0 else "")
            cdf.set_ylabel("Stop CDF" if j == 0 else "")
            ax.tick_params(labelbottom=False)
            number = int(case.cycle_name.rsplit("_", 1)[-1])
            role = "Largest gain" if j == 0 else "Largest remaining regret"
            delta = (scores.selected_cop - scores.rb_cop).mean()
            ax.set_title(
                f"Cycle {number} · {role}\nMean ΔCOP {delta:+.4f}", loc="left", fontsize=6.7, pad=4
            )
            if j == 1:
                ax.tick_params(labelleft=False)
                cdf.tick_params(labelleft=False)
        box = top[0].get_position()
        fig.text(box.x0 - 0.04, 0.915, "a", fontsize=11, weight="bold")
        fig.text(box.x0, 0.915, "Held-out decisions", fontsize=8, weight="bold")
        handles = [
            Line2D([], [], color="#B18A65", ls="--", label="RB"),
            Line2D([], [], color=colors["LSTM25_FiLM"], label="LSTM25"),
            Rectangle((0, 0), 1, 1, fc="#D8EBDD", label="Exact 1% region"),
        ]
        top[1].legend(
            handles=handles, loc="lower right", fontsize=5.7, frameon=False, handlelength=1.1
        )
        width_path = output / "panel_b_width_screen.csv"
        if width_path.exists():
            benchmark_grid = grid[0, 1].subgridspec(2, 1, height_ratios=[1.9, 1], hspace=0.65)
            ax = fig.add_subplot(benchmark_grid[0])
        else:
            ax = fig.add_subplot(grid[0, 1])
        order = list(names)
        for i, model in enumerate(order):
            g = models.loc[models.version.eq(model)].sort_values("seed")
            values = g.headroom_captured_pct.to_numpy()
            color = colors[model]
            ax.barh(i, values.mean(), height=0.48, color=color, alpha=0.17)
            ax.errorbar(
                values.mean(),
                i,
                xerr=values.std(ddof=1),
                fmt="o",
                color=color,
                ms=5,
                capsize=3,
                lw=1.4,
            )
            ax.scatter(
                values, i + np.array([-0.08, 0, 0.08]), s=9, color=color, alpha=0.8, zorder=4
            )
            ax.text(
                0.025,
                i - 0.32,
                names[model],
                transform=ax.get_yaxis_transform(),
                fontsize=7,
                color="#393946",
            )
            ax.text(
                0.025,
                i + 0.29,
                f"{int(g.parameters.iloc[0]):,} parameters",
                transform=ax.get_yaxis_transform(),
                fontsize=6,
                color="#73737D",
            )
            ax.text(
                0.985,
                i + 0.29,
                f"{values.mean():.2f} ± {values.std(ddof=1):.2f}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=7,
                color=color,
                weight="bold",
            )
        ax.set(
            xlim=(0, 100),
            ylim=(2.65, -0.65),
            yticks=[],
            xticks=[0, 25, 50, 75, 100],
            xlabel="$H_{RB}$ [%]" if not width_path.exists() else "",
        )
        ax.grid(axis="x", alpha=0.12)
        if width_path.exists():
            benchmark_box = ax.get_position()
            fig.text(benchmark_box.x0 - 0.035, 0.905, "b", fontsize=11, weight="bold")
            fig.text(benchmark_box.x0, 0.907, "Tuned model comparison", fontsize=8, weight="bold")
            fig.text(
                benchmark_box.x0,
                0.882,
                "Common inner epoch rule · mean ± sample SD",
                fontsize=6.3,
                color="#686873",
            )
        else:
            panel(
                ax,
                "b",
                "Tuned model comparison",
                "Common inner epoch rule · mean ± sample SD",
            )
        if width_path.exists():
            widths = pd.read_csv(width_path).sort_values("hidden_width")
            capacity = fig.add_subplot(benchmark_grid[1])
            x = np.arange(len(widths))
            capacity.plot(
                x, widths.validation_headroom, color=colors["LSTM25_FiLM"], marker="o", ms=4, lw=1.1
            )
            for position, row in zip(x, widths.itertuples(), strict=True):
                capacity.annotate(
                    f"{row.validation_headroom:.2f}",
                    (position, row.validation_headroom),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6,
                    color=colors["LSTM25_FiLM"],
                )
            capacity.set(
                xticks=x,
                xticklabels=[
                    f"{row.hidden_width} units\n{row.parameters / 1000:.1f}k params"
                    for row in widths.itertuples()
                ],
                xlim=(-0.35, 2.35),
                ylim=(81.55, 82.35),
                yticks=[81.6, 82.0],
                ylabel="$H_{RB}$ [%]",
            )
            capacity.tick_params(labelsize=5.8, pad=2)
            capacity.yaxis.label.set_size(6)
            capacity.set_title(
                "LSTM width · 99 training-side cycles",
                loc="left",
                fontsize=6,
                pad=8,
                color="#686873",
            )
            capacity.grid(axis="y", alpha=0.12, lw=0.5)
        ax = fig.add_subplot(grid[1, 0])
        quick = components.evaluation_scope.eq("training_side_quick").any()
        absolute = "H_RB" in components and components.H_RB.notna().all()
        for i, row in enumerate(components.itertuples()):
            value = row.H_RB if absolute else row.delta_cop
            color = "#AD8178" if value < 0 else "#6F9690"
            if absolute:
                formal = np.isfinite(row.H_RB_std)
                color = colors["LSTM25_FiLM"] if formal else "#789DA8"
                ax.barh(i, value, height=0.36, color=color, alpha=0.15)
                ax.errorbar(
                    value,
                    i,
                    xerr=row.H_RB_std if formal else None,
                    fmt="o",
                    color=color,
                    ms=4,
                    capsize=2,
                    lw=1.2,
                )
                ax.text(
                    0.98,
                    i,
                    f"{value:.2f}%",
                    transform=ax.get_yaxis_transform(),
                    fontsize=6.6,
                    ha="right",
                    va="center",
                    color=color,
                    weight="bold",
                )
            else:
                ax.errorbar(
                    value,
                    i,
                    xerr=[[value - row.lower], [row.upper - value]],
                    fmt="o",
                    color=color,
                    capsize=2,
                    ms=4,
                    lw=1.2,
                )
            label = row.label + (
                f" · {row.cycles}/101 evaluable" if absolute and row.cycles != 101 else ""
            )
            ax.text(
                0.02,
                i - 0.30,
                label,
                transform=ax.get_yaxis_transform(),
                fontsize=6.6,
                color="#444452",
            )
            protocol = (
                (
                    "Formal LOEO"
                    if row.evaluation_scope == "full_experiment_loeo"
                    else "Quick screen"
                )
                if quick
                else ""
            )
            if quick or not absolute:
                ax.text(
                    0.98,
                    i + 0.27 if quick else i,
                    f"{protocol} · n={row.cycles}" if quick else f"n={row.cycles}",
                    transform=ax.get_yaxis_transform(),
                    fontsize=5.7,
                    ha="right",
                    va="center",
                    color="#7C7C84",
                )
        if absolute:
            ax.set_xlim(0, 100)
        else:
            ax.axvline(0, color="#9999A3", ls="--", lw=0.8)
        ax.set(
            ylim=(len(components) - 0.5, -0.65),
            yticks=[],
            xlabel="$H_{RB}$ [%]" if absolute else "Paired ΔCOP vs the full reference model",
        )
        ax.grid(axis="x", alpha=0.12)
        detail = (
            (
                (
                    "Full/reset: LOEO · other deletions: quick screen"
                    if components.evaluation_scope.eq("full_experiment_loeo").sum() > 1
                    else "Full: LOEO · deletions: 40-epoch quick screen"
                )
                if quick
                else "All: 29-fold LOEO · full: repeated runs · deletions: one run"
            )
            if absolute
            else "Same LSTM25 recipe · 95% experiment-block CI"
        )
        panel(ax, "c", "LSTM component ablations", detail)
        ax = fig.add_subplot(grid[1, 1])
        learning_summary = []
        for model in order:
            h = learning.loc[learning.model.eq(model)]
            q = h.groupby("epoch").validation_cdf_loss.agg(
                median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
            )
            ax.fill_between(q.index, q.q25, q.q75, color=colors[model], alpha=0.12, lw=0)
            ax.plot(q.index, q["median"], color=colors[model], lw=1.3, label=names[model])
            learning_summary.append(q.reset_index().assign(model=model))
        pd.concat(learning_summary, ignore_index=True).to_csv(
            output / "panel_d_learning_summary.csv", index=False
        )
        ax.set(xlim=(1, 160), xlabel="Training epoch", ylabel="Validation CDF loss")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
        ax.grid(axis="y", alpha=0.12)
        ax.legend(loc="upper right", fontsize=6, frameon=False, handlelength=1.3)
        panel(
            ax,
            "d",
            "Optimization trajectories",
            "Inner validation only · median / IQR across folds and runs",
        )
        FigureCanvasAgg(fig).draw()
        canvas = fig.bbox
        for axis in fig.axes:
            box = axis.get_tightbbox(fig.canvas.get_renderer())
            if box.x0 < 0 or box.y0 < 0 or box.x1 > canvas.width + 1 or box.y1 > canvas.height + 1:
                raise ValueError("Evidence panel text exceeds the figure canvas")
        save_publication(fig, output, "LSTM25_FiLM_Fig7_abcd", None, layout=False, dpi=600)


def plot_temporal_stopping(summary, output):
    """Lag summary ablation; retain failure coverage next to conditional regret."""
    from plots.publication import save_publication

    source = summary.copy()
    source["history_minutes"] = source.version.str.extract(r"history_(\d+)m")[0].astype(int)
    source["no_trigger_pct"] = 100 * source.no_trigger_count / source.cohort_cycles
    source["not_evaluable_pct"] = 100 * (1 - source.evaluable_coverage)
    source.to_csv(output / "history_figure_source.csv", index=False)
    means = source.groupby("history_minutes").mean(numeric_only=True)
    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 9}):
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
        for field, color, label in [
            ("p90_regret_pct", "#A26E42", "P90 regret"),
            ("median_regret_pct", "#467899", "Median regret"),
        ]:
            axes[0].plot(means.index, means[field], "-o", color=color, label=label, lw=1.5, ms=4)
            axes[0].scatter(
                source.history_minutes,
                source[field],
                s=16,
                facecolors="none",
                edgecolors=color,
                alpha=0.5,
            )
        axes[0].axhline(1, color="#888888", lw=0.7, ls="--")
        axes[0].set_ylabel("Effective-COP regret (%)")
        axes[0].legend(frameon=False)
        for field, color, label in [
            ("not_evaluable_pct", "#806D99", "All unevaluable decisions"),
            ("no_trigger_pct", "#57867C", "No crossing, input available"),
        ]:
            axes[1].plot(means.index, means[field], "-o", color=color, label=label, lw=1.5, ms=4)
            axes[1].scatter(
                source.history_minutes,
                source[field],
                s=16,
                facecolors="none",
                edgecolors=color,
                alpha=0.5,
            )
        axes[1].set(
            xlabel="Requested causal visual lag (minutes)",
            ylabel="Fraction of all cycles (%)",
            xticks=means.index,
        )
        axes[1].legend(frameon=False)
        for ax in axes:
            ax.grid(axis="y", alpha=0.15)
        fig.suptitle("Does visual growth history improve first-trigger decisions?", y=0.98)
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.25, top=0.92, hspace=0.25)
        caption = (
            "Evidence: frozen DINO current + change from one lagged frame; same 768-input logistic head, set loss and threshold 0.5.\n"
            "Outer experiment holdout; five separate validation experiments; student scaling fits training folds only. Energy teacher remains frozen.\n"
            "Lines average three seed-wise statistics; hollow points are seeds, not independent samples. Regret is conditional on evaluable COP.\n"
            f"Observed P90: {means.p90_regret_pct.iloc[0]:.2f}% at {means.index[0]} min versus {means.p90_regret_pct.iloc[-1]:.2f}% at {means.index[-1]} min; paired tables assess common-cycle COP.\n"
            "First frame pads startup; missing lag frames give zero change without removing actions. A lag summary is not a full temporal encoder."
        )
        save_publication(fig, output, "cop_regret_vs_visual_history", caption, layout=False)
