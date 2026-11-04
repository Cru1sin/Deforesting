"""Render the historical RB strategy as one Chinese operating-condition table."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from output_layout import PROJECT
from plots.publication import _RGB_PANEL_FONT, save_publication

matplotlib.use("Agg")

OUTPUT = PROJECT / "output/03_数据与物理依据/RB触发规则"
INK, MUTED, BLUE, ORANGE = "#18283C", "#526579", "#165C88", "#AA4B12"

# Chinese condition names summarize temperature bins, not frost severity.
CASES = (
    ("Case1", "较暖环境", "T4 ≥ −2 °C", (40, 35, 30), "T3 < T3o − 3 °C"),
    ("Case2", "轻寒环境", "−5 ≤ T4 < −2 °C", (40, 38, 33), "T3 < T3o − 5 °C"),
    ("Case3", "寒冷环境", "−8 ≤ T4 < −5 °C", (80, 60, 40), "T3 < T3o − 5 °C"),
    ("Case4", "更低温环境", "−10 ≤ T4 < −8 °C", (90, 70, 50), "T3 < T4 − 5 °C"),
    ("Case5", "严寒环境", "T4 < −10 °C", (150, 120, 90), "T3 < T4 − 5 °C"),
)


def plot_rb_strategy():
    plt.rcParams.update(
        {
            "font.family": ["Arial Unicode MS", "DejaVu Sans"],
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(20, 11.25), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def text(x, y, value, size=16, color=INK, ha="left"):
        return fig.text(
            x, y, value, fontsize=size, color=color, ha=ha, va="center", linespacing=1.45
        )

    text(0.025, 0.953, "RB 除霜策略｜工况与触发条件一览", 31)
    ax.add_patch(Rectangle((0.025, 0.872), 0.95, 0.043, color=BLUE))
    text(
        0.5,
        0.893,
        "任意一行的全部条件满足 → 触发除霜；全部未满足 → 继续制热、继续观察",
        20,
        "white",
        ha="center",
    )
    text(
        0.025,
        0.843,
        "读表顺序：① 找到适用工况   ② 核对 T1、T2 计时门槛   ③ 检查盘管温度及持续要求",
        16,
        MUTED,
    )

    widths = (0.085, 0.17, 0.08, 0.085, 0.08, 0.065, 0.25, 0.135)
    edges = [0.025]
    for width in widths:
        edges.append(edges[-1] + width)

    def cell(col, y, height, value, *, span=1, face="white", color=INK, size=16):
        x, width = edges[col], edges[col + span] - edges[col]
        ax.add_patch(
            Rectangle((x, y), width, height, facecolor=face, edgecolor="#CFDAE5", linewidth=0.7)
        )
        return text(x + width / 2, y + height / 2, value, size, color, ha="center")

    top, header_h, half = 0.809, 0.08, 0.04
    headers = (
        (0, "规则编号"),
        (1, "适用工况\n环境温度 T4"),
        (5, "T2 门槛\n（min）"),
        (6, "盘管温度条件\n本栏各项须同时满足"),
        (7, "持续要求"),
    )
    for col, label in headers:
        cell(col, top - header_h, header_h, label, face=INK, color="white", size=15)
    cell(
        2,
        top - half,
        half,
        "T1 门槛（min）｜按出水温度 Twout 选列",
        span=3,
        face=BLUE,
        color="white",
        size=14,
    )
    for col, label in zip((2, 3, 4), ("≥ 35 °C", "25–<35 °C", "< 25 °C"), strict=True):
        cell(col, top - header_h, half, label, face=BLUE, color="white", size=15)

    y, height = top - header_h, 0.093
    y -= height
    face = "#F1F5FA"
    cell(0, y, height, "Condition1", face="#E6EDF7", color=BLUE, size=15)
    cell(1, y, height, "盘管已冷且继续降温\n不限定环境温度分档", face=face, size=15)
    cell(2, y, height, "> 35（各出水温度档相同）", span=3, face=face)
    cell(5, y, height, "≥ 6", face=face, size=18)
    cell(
        6,
        y,
        height,
        "T3 ≤ −1 °C，且\n过去 50–600 秒的最高有效 T3\n比当前 T3 至少高 1 °C",
        face=face,
        size=15,
    )
    cell(7, y, height, "当秒满足即触发\n无 20 秒连续要求", face=face, size=15)

    for index, (case, condition, interval, limits, temperature) in enumerate(CASES):
        height = 0.063
        y -= height
        face = "#F0F7F7" if index % 2 == 0 else "#FFFFFF"
        cell(0, y, height, case, face="#DFEEEE", color="#176B68", size=18)
        cell(1, y, height, f"{condition}\n{interval}", face=face, size=15)
        for col, limit in zip((2, 3, 4), limits, strict=True):
            cell(col, y, height, f"≥ {limit}", face=face, size=20)
        cell(5, y, height, "≥ 6", face=face, size=18)
        cell(6, y, height, temperature, face=face, size=19)
        cell(7, y, height, "温度条件\n连续满足 20 秒", face=face, size=15)

    height = 0.076
    y -= height
    face = "#F0EDF7"
    cell(0, y, height, "Case7", face="#E4DCF1", color="#674399", size=18)
    cell(1, y, height, "盘管低温工况\n不限定环境温度分档", face=face, size=15)
    cell(2, y, height, "≥ 30（各出水温度档相同）", span=3, face=face)
    cell(5, y, height, "无要求", face=face, size=15)
    cell(6, y, height, "T3 ≤ −10 °C，且\nT3 ≤ 0.8 × T4 − 12 °C", face=face, size=18)
    cell(7, y, height, "两个温度条件\n均连续满足 20 秒", face=face, size=15)

    height = 0.064
    y -= height
    face = "#FFF0E4"
    cell(0, y, height, "Case8", face="#F9DCC5", color=ORANGE, size=18)
    cell(1, y, height, "计时超时工况\n150 分钟兜底", face=face, size=15)
    cell(2, y, height, "≥ 150（各出水温度档相同）", span=3, face=face)
    cell(5, y, height, "无要求", face=face, size=15)
    cell(6, y, height, "无温度要求", face=face, size=19, color=ORANGE)
    cell(7, y, height, "当秒满足即触发\n无需持续确认", face=face, size=15)

    text(
        0.025,
        0.142,
        "20 秒只约束温度条件；计时门槛在触发当秒检查。20 秒 = 连续 20 个一秒采样点，缺秒不补值。",
        15,
        BLUE,
    )
    text(
        0.025,
        0.106,
        "T3 = 盘管温度   ·   T4 = 环境温度   ·   Twout = 出水温度   ·   min = 分钟",
        15,
    )
    text(
        0.025,
        0.075,
        "T1 / T2 = 机内 DefTim1 / DefTim2 计时器（原始秒值 ÷ 60）；"
        "T3o = 机内 T3o 字段（原始值 ÷ 10，单位 °C）。",
        13,
        MUTED,
    )
    text(
        0.025,
        0.046,
        "工况名称为便于讲解的概括，以温度区间为准；Case1–5 按环境分档，"
        "Condition1 / Case7 / Case8 独立判断。历史实现没有 Case6。",
        12,
        MUTED,
    )
    text(
        0.025,
        0.02,
        "历史 RB 回放规则｜记录首次触发；同秒标签优先级："
        "Condition1 → Case1–5 → Case7 → Case8（仅决定标签）。",
        12,
        MUTED,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"RB策略_工况与触发条件总表.{extension}", dpi=160, facecolor="white")
    plt.close(fig)


def render_rb_opportunity(source_path: Path, summary_path: Path) -> None:
    """Vertical bars preserve the historical gain bins and ratios of group mean COP."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as PlotPath

    source = pd.read_csv(source_path)
    summary = pd.read_csv(summary_path)
    labels = ["≤1%", "1–2%", "2–3%", ">3%"]
    groups = pd.cut(source.potential_gain_pct, [-np.inf, 1, 2, 3, np.inf], labels=labels)
    counts = groups.value_counts(sort=False).reindex(labels, fill_value=0)
    distribution = pd.DataFrame(
        {
            "gain_range": labels,
            "cycles": counts.to_numpy(),
            "total": len(source),
            "percent": 100 * counts.to_numpy() / len(source),
        }
    )
    distribution.to_csv(source_path.parent / "gain_distribution.csv", index=False)
    by_setpoint = pd.concat([summary.iloc[1:], summary.iloc[:1]], ignore_index=True)
    ink, grey = "#193F5B", "#6C8293"
    blues = ["#477F9E", "#79A9BD", "#AACAD6"]
    with plt.rc_context(
        {
            "font.family": _RGB_PANEL_FONT,
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.linewidth": 0.6,
            "axes.edgecolor": grey,
            "axes.labelcolor": ink,
            "text.color": ink,
            "xtick.color": ink,
            "ytick.color": grey,
            "axes.unicode_minus": False,
        }
    ):
        figure = plt.figure(figsize=(183 / 25.4, 76 / 25.4), facecolor="white")
        left = figure.add_axes((0.08, 0.25, 0.39, 0.69))
        right = figure.add_axes((0.59, 0.25, 0.39, 0.69))
        left.set(
            xlim=(-0.6, 3.6),
            ylim=(0, 65),
            yticks=[0, 20, 40, 60],
            xticks=range(4),
            xticklabels=labels,
            ylabel="循环占比 (%)",
            xlabel="单循环潜在 COP 提升幅度",
        )
        right.set(
            xlim=(-0.6, len(by_setpoint) - 0.4),
            ylim=(0, max(by_setpoint.pooled_potential_gain_pct) * 1.25),
            yticks=[0, 1, 2],
            xticks=range(len(by_setpoint)),
            xticklabels=[
                f"{float(row.group):g}°C" if row.group != "All" else "总体"
                for row in by_setpoint.itertuples()
            ],
            ylabel="平均 COP 改善上限 (%)",
            xlabel="水温设定与总体",
        )
        for axis, heights, bar_counts, colors in [
            (left, distribution.percent, distribution.cycles, [*blues, "#E8A458"]),
            (right, by_setpoint.pooled_potential_gain_pct, by_setpoint.cycles, [*blues, "#E8A458"]),
        ]:
            axis.tick_params(axis="x", length=0, pad=7)
            axis.tick_params(axis="y", length=3, width=0.6, pad=5)
            axis.xaxis.labelpad = 30
            # Equal physical corner radii; the baseline stays exactly at zero.
            rx = (
                3
                / (figure.get_figwidth() * 72 * axis.get_position().width)
                * np.ptp(axis.get_xlim())
            )
            ry = (
                3
                / (figure.get_figheight() * 72 * axis.get_position().height)
                * np.ptp(axis.get_ylim())
            )
            for index, (height, count, color) in enumerate(zip(heights, bar_counts, colors)):
                x0, x1 = index - 0.32, index + 0.32
                r = min(ry, height / 2)
                vertices = [
                    (x0, 0),
                    (x0, height - r),
                    (x0, height),
                    (x0 + rx, height),
                    (x1 - rx, height),
                    (x1, height),
                    (x1, height - r),
                    (x1, 0),
                    (x0, 0),
                ]
                codes = [
                    PlotPath.MOVETO,
                    PlotPath.LINETO,
                    PlotPath.CURVE3,
                    PlotPath.CURVE3,
                    PlotPath.LINETO,
                    PlotPath.CURVE3,
                    PlotPath.CURVE3,
                    PlotPath.LINETO,
                    PlotPath.CLOSEPOLY,
                ]
                axis.add_patch(
                    PathPatch(PlotPath(vertices, codes), facecolor=color, edgecolor="none")
                )
                value = f"{height:.1f}%" if axis is left else f"{height:.2f}%"
                axis.annotate(
                    value,
                    (index, height),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontfamily="Arial",
                    fontweight="bold",
                )
                axis.text(
                    index,
                    -0.16,
                    f"{int(count)} 个" if axis is left else f"n = {int(count)}",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=7.5,
                    color=grey,
                )
        save_publication(
            figure,
            source_path.parent,
            "RB_improvement_headroom",
            "",
            formats=("svg", "pdf", "png", "tiff"),
            layout=False,
            dpi=600,
        )


def render_rb_timing(source_path: Path) -> None:
    """Exact 99%-COP membership and trigger provenance, without filling near-set holes."""
    source = pd.read_csv(source_path)
    order = ["early", "hit", "late", "gap", "unsupported", "no_trigger", "no_near_set"]
    names = [
        "偏早",
        "近优区间内",
        "偏晚",
        "区间内非近优",
        "触发点无有效COP",
        "无触发记录",
        "无近优集合",
    ]
    colors = ["#C99D7C", "#527F9E", "#9BAAB7", "#B4A6BD", "#BBBBBB", "#BBBBBB", "#BBBBBB"]
    counts = source.groupby(["trigger_status", "rb_trigger_source"]).size().unstack(fill_value=0)
    counts = counts.reindex(order, fill_value=0)
    distribution = counts.reset_index()
    distribution["cycles"] = counts.sum(axis=1).to_numpy()
    distribution["total"] = len(source)
    distribution["percent"] = 100 * distribution.cycles / len(source)
    distribution.to_csv(source_path.parent / "rb_timing_distribution.csv", index=False)
    shown = [i for i, status in enumerate(order) if counts.loc[status].sum() or status in order[:3]]
    with plt.rc_context(
        {
            "font.family": _RGB_PANEL_FONT,
            "font.size": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.unicode_minus": False,
        }
    ):
        figure = plt.figure(figsize=(7.4, 3.8), facecolor="white")
        axis = figure.add_axes((0.115, 0.235, 0.85, 0.535))
        hit_pct = 100 * source.trigger_status.eq("hit").mean()
        figure.text(
            0.115,
            0.97,
            f"{hit_pct:.1f}% 的RB触发位于近优区间",
            fontsize=18,
            color="#253D50",
            va="top",
        )
        figure.text(
            0.115,
            0.845,
            f"n = {len(source)} · 近优：COP ≥ 单循环最优COP的99%",
            fontsize=10.5,
            color="#526579",
        )
        for x, i in enumerate(shown):
            status, color = order[i], colors[i]
            bottom = 0
            for provenance, hatch in [
                ("replayed_rb_in_experiment", None),
                ("rb_acquired_cycle", "////"),
            ]:
                n = counts.loc[status].get(provenance, 0)
                pct = 100 * n / len(source)
                axis.bar(
                    x,
                    pct,
                    bottom=bottom,
                    width=0.70,
                    color=color,
                    edgecolor="#FFFFFF" if hatch is None else "#526579",
                    linewidth=0.6,
                    hatch=hatch,
                )
                bottom += pct
            n = int(counts.loc[status].sum())
            axis.text(
                x,
                bottom + 1.6,
                f"{bottom:.1f}%（{n}个）",
                ha="center",
                va="bottom",
                fontsize=13,
                color="#253D50",
            )
        axis.set(
            xticks=range(len(shown)),
            xticklabels=[names[i] for i in shown],
            ylabel="循环占比 (%)",
            ylim=(0, 65),
            xlim=(-0.50, len(shown) - 0.50),
        )
        axis.set_yticks([0, 20, 40, 60])
        axis.tick_params(axis="x", length=0, pad=6)
        axis.spines["bottom"].set_color("#BAC3CB")
        for status, prefix, xpos in [("early", "提前中位", 0.257), ("late", "延后中位", 0.823)]:
            part = source.loc[source.trigger_status.eq(status), "outside_near_minutes"]
            if len(part):
                figure.text(
                    xpos,
                    0.11,
                    f"{prefix} {part.median():.1f} min",
                    ha="center",
                    fontsize=10,
                    color="#526579",
                )
        replay = source.loc[source.rb_trigger_source.eq("replayed_rb_in_experiment")]
        acquired = source.loc[source.rb_trigger_source.eq("rb_acquired_cycle")]
        figure.text(
            0.115,
            0.035,
            f"纯色：实验内RB回放（{len(replay)}个）    斜线：RB采集循环（{len(acquired)}个）",
            fontsize=10,
            color="#526579",
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for text in [
            *figure.texts,
            *axis.texts,
            axis.xaxis.label,
            axis.yaxis.label,
            *axis.get_xticklabels(),
            *axis.get_yticklabels(),
        ]:
            box = text.get_window_extent(renderer)
            if text.get_text() and (
                box.x0 < 0
                or box.y0 < 0
                or box.x1 > figure.bbox.width + 1
                or box.y1 > figure.bbox.height + 1
            ):
                raise ValueError(f"Text outside figure bounds: {text.get_text()}")
        save_publication(
            figure, source_path.parent, "RB_trigger_timing_compact", "", layout=False, dpi=400
        )


def render_rb_table_audit(output):
    """Matched RB branches and each mismatch transition are mutually exclusive rows."""
    source = pd.read_csv(output / "cycle_trigger_audit.csv")
    rb = source.loc[source.group.eq("cycle_RB")].copy()
    regular = source.loc[source.group.eq("cycle")]
    rb["display_class"] = rb.recorded_branch
    incomplete = rb.audit_category.eq("触发记录或边界不完整")
    rb.loc[incomplete, "display_class"] = rb.loc[incomplete, "recorded_branch"].map(
        lambda branch: "记录缺失" if pd.isna(branch) else "边界不完整"
    )
    classified = rb.audit_category.eq("工况分类不一致")
    rb.loc[classified, "display_class"] = (
        "应属"
        + rb.loc[classified, "ambient_case_at_trigger"]
        + " → 实机"
        + rb.loc[classified, "recorded_branch"]
    )
    different = rb.audit_category.eq("分支不一致")
    rb.loc[different, "display_class"] = (
        "实机"
        + rb.loc[different, "recorded_branch"]
        + " → 回放"
        + rb.loc[different, "table_branch"].replace("未触发", "未检出")
    )
    branches = ["Condition1", *[f"Case{i}" for i in range(1, 6)], "Case7", "Case8"]
    if "case8_enabled" in source and not source.case8_enabled.any():
        branches.remove("Case8")
    transitions = sorted(
        rb.loc[different, "display_class"].unique(),
        key=lambda value: ("未检出" not in value, "Case8" not in value, value),
    )
    classes = [
        *branches,
        "观测内未检出触发",
        "记录缺失",
        "边界不完整",
        *sorted(rb.loc[classified, "display_class"].unique()),
        *transitions,
    ]
    rb_counts = rb.display_class.value_counts()
    regular_counts = regular.table_branch.replace("未触发", "观测内未检出触发").value_counts()
    classes = [
        label for label in classes if rb_counts.get(label, 0) or regular_counts.get(label, 0)
    ]
    reasons = {}
    for label, group in rb.groupby("display_class"):
        if label == "记录缺失":
            reasons[label] = "无有效实机触发记录"
        elif label == "边界不完整":
            reasons[label] = "缺少准备除霜边界或触发前数据"
        elif label.startswith("应属"):
            temperatures = group.T4_at_recorded_trigger_C
            value = (
                f"{temperatures.min():g}"
                if temperatures.nunique() == 1
                else f"{temperatures.min():g}–{temperatures.max():g}"
            )
            reasons[label] = f"触发时 T4 = {value}°C，应属{group.ambient_case_at_trigger.iloc[0]}"
        elif label.startswith("实机") and group.table_branch.eq("未触发").all():
            if group.recorded_branch.eq("Condition1").all():
                drops = group.last_historical_drop_c
                reasons[label] = (
                    f"50–600 s 可见温降仅 {drops.min():.1f}–{drops.max():.1f}°C，未达2°C"
                )
            elif not group.last_temperature_confirmed.any():
                missing = group.last_temperature_window_missing_seconds.max()
                failed = group.last_temperature_window_failed_seconds.max()
                reasons[label] = (
                    f"最后20 s判定窗口缺测 {missing:g} s（允许≤3 s）；有效点均满足"
                    if failed == 0 and missing > 3
                    else f"最后20 s窗口缺测 {missing:g} s、实测不满足 {failed:g} s"
                )
            else:
                reasons[label] = "未达到对应工况的计时门槛"
        elif label.startswith("实机"):
            advance = (
                pd.to_datetime(group.recorded_trigger_time) - pd.to_datetime(group.table_time)
            ).dt.total_seconds()
            duration = (
                f"{advance.iloc[0]:g}"
                if len(advance) == 1
                else f"{advance.min():g}–{advance.max():g}"
            )
            reasons[label] = f"回放提前 {duration} s 满足{group.table_branch.iloc[0]}"
    displayed = []
    colors = ["#527F9E", "#84AFAB"]
    with plt.rc_context(
        {
            "font.family": _RGB_PANEL_FONT,
            "font.size": 12,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.linewidth": 0.7,
            "axes.unicode_minus": False,
        }
    ):
        main = [label for label in classes if label in branches or label == "观测内未检出触发"]
        details = [label for label in classes if label not in main]
        main_y = {label: index * 1.35 for index, label in enumerate(main)}
        detail_start = max(main_y.values()) + 1.35
        detail_y = {label: detail_start + index * 0.53 for index, label in enumerate(details)}
        positions = {**main_y, **detail_y}
        fig, ax = plt.subplots(figsize=(13.5, 8))
        fig.subplots_adjust(left=0.025, right=0.985, bottom=0.025, top=0.925)
        ax.set(xlim=(0, 90), ylim=(max(positions.values()) + 0.43, -0.65))
        ax.axis("off")
        for label in classes:
            row = positions[label]
            is_main = label in main
            if is_main:
                ax.add_patch(
                    Rectangle((19.5, row - 0.50), 64, 1, facecolor="#F5F7F9", edgecolor="none")
                )
                short = "未检出触发" if label == "观测内未检出触发" else label
                ax.text(
                    7.5 if label.startswith("Case") else 0,
                    row,
                    short,
                    va="center",
                    fontsize=14,
                    color="#253D50",
                )
            else:
                short = label.replace("实机", "").replace("回放", "").replace("应属", "")
                if label.startswith("应属"):
                    short = label
                ax.text(0, row, short, va="center", fontsize=10.5, color="#526579")
            for col, (column, counts, total) in enumerate(
                [
                    ("cycle_RB 实机核对", rb_counts, len(rb)),
                    ("cycle 按表回放", regular_counts, len(regular)),
                ]
            ):
                available = (
                    col == 0
                    and label != "观测内未检出触发"
                    or col == 1
                    and label in [*branches, "观测内未检出触发"]
                )
                n = int(counts.get(label, 0)) if available else np.nan
                displayed.append(
                    dict(
                        column=column,
                        branch=label,
                        n=n,
                        total=total,
                        percent=100 * n / total,
                        reason=reasons.get(label, ""),
                    )
                )
                if not available:
                    continue
                fraction = 100 * n / total
                if is_main:
                    y = row + (-0.23 if col == 0 else 0.23) if label in branches else row
                    ax.barh(y, fraction, left=20, height=0.36, color=colors[col], zorder=2)
                    ax.text(
                        20 + fraction + 0.9,
                        y,
                        f"{fraction:.1f}% ({n})",
                        va="center",
                        fontsize=11,
                        color="#253D50",
                    )
                else:
                    ax.text(
                        27,
                        row,
                        f"{n} ({fraction:.1f}%)",
                        va="center",
                        fontsize=10.5,
                        color="#526579",
                    )
            if not is_main:
                ax.text(43, row, reasons[label], va="center", fontsize=10.5, color="#526579")
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(facecolor=colors[0], label=f"cycle_RB · n={len(rb)}"),
                Patch(facecolor=colors[1], label=f"cycle · n={len(regular)}"),
            ],
            loc="lower left",
            bbox_to_anchor=(20 / 90, 1.015),
            ncol=2,
            frameon=False,
            fontsize=13,
            borderaxespad=0,
        )
        case_y = [main_y[label] for label in main if label.startswith("Case")]
        if case_y:
            lower, upper = min(case_y) - 0.5, max(case_y) + 0.5
            ax.plot([5.5, 5, 5, 5.5], [lower, lower, upper, upper], color="#84AFAB", lw=1.4)
            ax.text(
                1.6,
                (lower + upper) / 2,
                "Condition2",
                rotation=90,
                ha="center",
                va="center",
                fontsize=12,
                color="#547F7C",
            )
        heading_y = detail_start - 0.58
        ax.plot([0, 89], [heading_y - 0.30, heading_y - 0.30], color="#D8DDE1", lw=0.7)
        for x, label in [(0, "实机 → 回放 / 记录状态"), (27, "数量（RB占比）"), (43, "原因")]:
            ax.text(x, heading_y, label, va="center", fontsize=10.5, color="#526579")
        pd.DataFrame(displayed).to_csv(output / "displayed_trigger_proportions.csv", index=False)
        rb["display_reason"] = rb.display_class.map(reasons).fillna("")
        rb[
            [
                "cycle_name",
                "display_class",
                "display_reason",
                "recorded_branch",
                "table_branch",
                "ambient_case_at_trigger",
                "T4_at_recorded_trigger_C",
            ]
        ].to_csv(output / "rb_display_classification.csv", index=False)
        save_publication(
            fig,
            output,
            "RB_trigger_distribution_and_table_audit_balanced",
            "",
            layout=False,
            dpi=300,
        )


if __name__ == "__main__":
    plot_rb_strategy()
