#!/usr/bin/env python3
"""Render an evidence-backed cost-function design guide as standalone figures."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Heiti SC", "Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "output/test/成本函数/成本函数设计框架"
NAVY = "#0F4D92"
BLUE = "#5B8FD6"
TEAL = "#42949E"
GREEN = "#2E7D5B"
GOLD = "#E28E2C"
RED = "#B64342"
INK = "#272727"
GREY = "#767676"
LIGHT = "#EEF1F5"
DOMAIN_LABELS = {
    "building_energy": "building",
    "data_driven_control": "data control",
    "dynamic_programming": "dynamic prog.",
    "embedded_control": "embedded",
    "mathematical_optimization": "math opt.",
    "model_based_rl": "model RL",
    "offline_rl": "offline RL",
    "operations_research": "OR",
    "predict_then_optimize": "predict-opt",
    "quantitative_finance": "finance",
    "reinforcement_learning": "RL",
    "reward_engineering": "reward",
    "robust_optimization": "robust opt.",
}


def _wrap(text: object, width: int) -> str:
    return textwrap.fill(str(text), width=width, break_long_words=False)


def _canvas(title: str, subtitle: str, height: float = 6.0) -> tuple[plt.Figure, plt.Axes]:
    figure, axis = plt.subplots(figsize=(8.3, height))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    axis.text(0.04, 0.96, title, fontsize=15, fontweight="bold", color=INK, va="top")
    axis.text(0.04, 0.905, subtitle, fontsize=8.5, color=GREY, va="top")
    return figure, axis


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str,
    color: str = NAVY,
    face: str = "white",
    title_size: float = 9.5,
    body_size: float = 7.6,
) -> None:
    x, y = xy
    width, height = wh
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.009,rounding_size=0.015",
            facecolor=face,
            edgecolor=color,
            linewidth=1.2,
        )
    )
    axis.text(
        x + 0.018,
        y + height - 0.028,
        title,
        fontsize=title_size,
        fontweight="bold",
        color=color,
        va="top",
    )
    axis.text(
        x + 0.018,
        y + height - 0.075,
        body,
        fontsize=body_size,
        color=INK,
        va="top",
        linespacing=1.35,
    )


def _arrow(
    axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = GREY
) -> None:
    axis.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, lw=1.1, color=color)
    )


def _footer(axis: plt.Axes, sources: str, conclusion: str) -> None:
    axis.text(0.04, 0.035, conclusion, fontsize=8.2, color=INK, fontweight="bold", va="bottom")
    axis.text(0.96, 0.035, sources, fontsize=6.5, color=GREY, ha="right", va="bottom")


def _save(figure: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    figure.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_evidence_map(source: pd.DataFrame, output: Path) -> None:
    counts_by_type = source["source_type"].value_counts()
    figure, axis = _canvas(
        f"{len(source)}项跨领域证据共同指向五条成本函数设计规则",
        f"同行评审{counts_by_type.get('peer_reviewed', 0)}篇定方法；"
        "GitHub"
        f"{source.loc[source['source_type'].eq('github_official'), 'source'].nunique()}"
        "个高星库定实现；"
        f"小红书{counts_by_type.get('xiaohongshu', 0)}篇非推广笔记只作失败模式线索。",
        6.3,
    )
    type_order = ["peer_reviewed", "github_official", "community", "xiaohongshu"]
    type_names = ["同行评审", "GitHub官方", "社区讨论", "小红书"]
    colors = [NAVY, TEAL, GOLD, RED]
    counts = source["source_type"].value_counts().reindex(type_order).fillna(0)
    maximum = counts.max()
    for index, (name, value, color) in enumerate(zip(type_names, counts, colors, strict=True)):
        y = 0.80 - 0.09 * index
        axis.text(0.06, y, name, fontsize=8.5, va="center")
        axis.add_patch(
            FancyBboxPatch(
                (0.18, y - 0.018),
                0.31 * value / maximum,
                0.036,
                boxstyle="round,pad=0",
                facecolor=color,
                edgecolor="none",
            )
        )
        axis.text(
            0.51, y, f"{int(value)}项", fontsize=9, fontweight="bold", color=color, va="center"
        )

    groups = [
        (
            "长期/循环目标",
            {"operations_research", "dynamic_programming", "mathematical_optimization"},
            NAVY,
        ),
        ("瞬态、终端与约束", {"building_energy", "embedded_control"}, BLUE),
        ("模型误差与风险", {"robust_optimization", "offline_rl", "quantitative_finance"}, TEAL),
        (
            "决策导向学习",
            {"predict_then_optimize", "data_driven_control", "reward_engineering"},
            GREEN,
        ),
        ("学习控制", {"reinforcement_learning", "model_based_rl"}, GOLD),
        ("工程失败模式", set(), RED),
    ]
    for index, (title, domains, color) in enumerate(groups):
        col, row = index % 2, index // 2
        x, y = 0.58 + 0.20 * col, 0.745 - 0.19 * row
        selected = (
            source["source_type"].isin(["community", "xiaohongshu"])
            if not domains
            else (
                source["domain"].isin(domains)
                & ~source["source_type"].isin(["community", "xiaohongshu"])
            )
        )
        _box(
            axis,
            (x, y),
            (0.18, 0.145),
            title,
            f"{int(selected.sum())}项证据",
            color=color,
            face="#FAFBFC",
            title_size=8.2,
            body_size=5.8,
        )

    _box(
        axis,
        (0.06, 0.20),
        (0.46, 0.18),
        "共同规则",
        "① 主目标保持物理量纲；② 瞬态通过状态、约束和终端价值进入；\n"
        "③ 模型误差必须传播到决策；④ 曲线平坦时输出窗口而非伪精确点；\n"
        "⑤ 无动作覆盖时，反事实不可识别。",
        color=GREEN,
        face="#F4FAF7",
        body_size=8.0,
    )
    _footer(
        axis,
        "完整来源、规则、URL：source_matrix.csv",
        "证据不是投票：A层定理论，B层定实现，C层只提醒哪里容易踩坑。",
    )
    _save(figure, output, "figure_01_evidence_map")


def plot_source_atlas(source: pd.DataFrame, output: Path, page_size: int = 19) -> None:
    evidence_colors = {"A1": NAVY, "A2": BLUE, "R": TEAL, "B1": GREEN, "B2": GOLD, "C": RED}
    pages = int(np.ceil(len(source) / page_size))
    for page in range(pages):
        values = source.iloc[page * page_size : (page + 1) * page_size]
        figure, axis = _canvas(
            f"来源索引 {page + 1}/{pages}：每条规则均可追溯",
            "编号与source_matrix.csv一致；A1/A2/R为论文，B1/B2为官方或高质量工程证据，C为经验线索。",
            10.3,
        )
        top, row_height = 0.865, 0.043
        for row, item in enumerate(values.itertuples(index=False)):
            y = top - row * row_height
            color = evidence_colors.get(item.evidence, GREY)
            axis.add_patch(
                FancyBboxPatch(
                    (0.045, y - 0.013),
                    0.035,
                    0.028,
                    boxstyle="round,pad=0.003",
                    facecolor=color,
                    edgecolor="none",
                )
            )
            axis.text(
                0.0625,
                y + 0.001,
                str(item.id),
                fontsize=6.8,
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
            )
            axis.text(0.092, y + 0.001, _wrap(item.title, 57), fontsize=6.7, color=INK, va="center")
            axis.text(0.76, y + 0.001, str(item.year), fontsize=6.4, color=GREY, va="center")
            axis.text(
                0.83,
                y + 0.001,
                item.evidence,
                fontsize=6.3,
                color=color,
                va="center",
                fontweight="bold",
            )
            axis.text(
                0.955,
                y + 0.001,
                DOMAIN_LABELS.get(item.domain, str(item.domain).replace("_", " ")),
                fontsize=5.0,
                color=GREY,
                va="center",
                ha="right",
            )
            axis.plot([0.045, 0.955], [y - 0.020, y - 0.020], color="#E6E8EC", lw=0.45)
        _footer(
            axis,
            f"本页编号 {values.id.min()}–{values.id.max()}",
            "标题是索引；方法规则和限制请按编号查看source_matrix.csv。",
        )
        _save(figure, output, f"figure_02_source_atlas_{page + 1:02d}")


def plot_global_transient(output: Path) -> None:
    figure, axis = _canvas(
        "好的全局算法会兼顾瞬态，但不会重复惩罚瞬时COP",
        "长期效率决定优化方向；瞬态通过状态转移、硬约束、终端价值和风险进入。",
        6.2,
    )
    _box(
        axis,
        (0.31, 0.66),
        (0.38, 0.16),
        "全局主目标",
        r"$\rho^\pi=\mathrm{E}_\pi[E_{cycle}]/\mathrm{E}_\pi[Q_{u,cycle}]$"
        + "\n长期单位有效供热电耗",
        color=NAVY,
        face="#F2F6FB",
        body_size=10,
    )
    items = [
        (0.05, "状态转移", "霜、温度、储热和\n下一循环恢复状态", TEAL),
        (0.275, "逐时/逐循环约束", "出水温、压力、安全、\n最长结霜和最小驻留", RED),
        (0.50, "终端/继续价值", "有限预测窗之后\n仍会发生什么", GOLD),
        (0.725, "尾部风险", "低概率严重失供热\n或硬件风险", GREEN),
    ]
    for x, title, body, color in items:
        _box(
            axis,
            (x, 0.37),
            (0.20, 0.17),
            title,
            body,
            color=color,
            face="white",
            title_size=8.5,
            body_size=7.5,
        )
        _arrow(axis, (x + 0.10, 0.55), (0.43 + 0.04 * (x > 0.5), 0.66), color)
    _box(
        axis,
        (0.15, 0.15),
        (0.70, 0.14),
        "不推荐",
        r"$J=E/Q+\lambda\,(1/COP_{instant})+\cdots$"
        + "\n瞬时COP已影响E和Q；再加一次通常是重复计数，且权重不可解释。",
        color=RED,
        face="#FFF6F4",
        body_size=8.0,
    )
    _footer(axis, "来源[2–20,28–31,43–52]", "瞬态不是另一个目标；它是动态后果、约束和风险。")
    _save(figure, output, "figure_05_global_and_transient")


def plot_future_decline(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.3, 5.4))
    x = np.linspace(0, 145, 300)
    observed = np.interp(
        x, [0, 20, 36.6, 70, 94.6, 108.6, 145], [6.8, 2.0, 0.0, 2.4, 5.2, 4.2, 6.1]
    )
    counter = np.interp(x, [0, 20, 40, 75, 105, 135, 145], [6.0, 1.8, 0.5, 3.0, 1.0, -1.1, -0.6])
    axis.plot(x, observed, color=NAVY, lw=2.2, label="完整轨迹：后期回落但未低于当前全局最小")
    axis.plot(
        x, counter, color=RED, lw=1.8, ls="--", label="反例：后期低于当前点 → 当前只能叫局部最小"
    )
    axis.axhline(0, color=GREY, lw=0.8)
    axis.axvline(36.6, color=GOLD, lw=1.3, ls=":")
    axis.scatter([36.6, 94.6, 108.6], [0, 5.2, 4.2], color=[GOLD, RED, BLUE], s=35, zorder=4)
    axis.annotate(
        "循环90最优\n36.6 min", (36.6, 0), xytext=(5, 28), textcoords="offset points", fontsize=8
    )
    axis.annotate(
        "94.6 min\n+5.2%", (94.6, 5.2), xytext=(-12, 12), textcoords="offset points", fontsize=8
    )
    axis.annotate(
        "COP回升后108.6 min\n仍+4.2%",
        (108.6, 4.2),
        xytext=(8, -28),
        textcoords="offset points",
        fontsize=8,
    )
    axis.set(
        xlabel="从循环开始的时间 [min]",
        ylabel="相对候选最小成本 [%]",
        xlim=(0, 145),
        ylim=(-1.8, 7.5),
    )
    axis.set_title(
        "“未来还有下降空间”不否定当前最优；“未来成本更低”才否定",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=13,
    )
    axis.legend(loc="upper center", fontsize=7.5)
    axis.text(
        0.01,
        -0.22,
        "判断式：若存在 $t>t^*$ 使 $J(t)<J(t^*)$，则 $t^*$ 不是全局最优。"
        "在线时必须用继续价值预测，而非偷看未来。",
        transform=axis.transAxes,
        fontsize=8.2,
        fontweight="bold",
    )
    axis.text(
        0.99,
        -0.22,
        "来源[2–5,10–13,17,43]；循环90为本项目数据",
        transform=axis.transAxes,
        fontsize=6.3,
        color=GREY,
        ha="right",
    )
    figure.tight_layout()
    _save(figure, output, "figure_06_future_decline_and_global_optimum")


def plot_cost_anatomy(output: Path) -> None:
    figure, axis = _canvas(
        "好成本函数的六层结构：每一层只做一件事",
        "从物理目标到实际触发逐层写；某层缺失时，不应靠调整其他层权重补洞。",
        7.0,
    )
    layers = [
        ("1 物理主目标", r"长期 $E/Q_u$ 或经济成本率；量纲、正负号、控制体必须一致", NAVY),
        ("2 动作后果", r"$E_a,Q_{u,a},s'_a$ 都随“继续/除霜”变化，包含准备、除霜、恢复和重置", TEAL),
        ("3 继续价值", r"$h(s')$ 关闭预测窗，防止把未来恢复或继续结霜截掉", GOLD),
        (
            "4 服务与安全",
            r"水温、压力、最长结霜、最小驻留：hard / chance / robust constraints",
            RED,
        ),
        (
            "5 不确定性与支持域",
            r"情景、CVaR、置信上界、OOD惩罚或abstain；不得把外推点当同等证据",
            GREEN,
        ),
        ("6 决策与执行", r"近优窗口、latest-safe tie-break、迟滞、连续帧确认和fallback", BLUE),
    ]
    for index, (title, body, color) in enumerate(layers):
        y = 0.75 - 0.128 * index
        _box(
            axis,
            (0.08, y),
            (0.84, 0.125),
            title,
            body,
            color=color,
            face="#FAFBFC",
            title_size=8.6,
            body_size=7.2,
        )
    _footer(
        axis,
        "来源[1–20,26–35,42–52]",
        "只有第1层决定“节能是什么”；其余层决定“能否安全可信地实现”。",
    )
    _save(figure, output, "figure_07_cost_function_anatomy")


def plot_formula_contract(output: Path) -> None:
    figure, axis = _canvas(
        "成本函数最清晰的写法：四行定义，不写一个巨型权重汤",
        "先写问题，再写目标，再写约束，最后写触发；所有符号都说明单位和何时可获得。",
        6.5,
    )
    rows = [
        (
            "① 决策问题",
            r"$s_t$: 在线可观测状态；$a\in\{H,D\}$；$T(s,a)$: 驻留时间；$s'$: 下一状态",
            NAVY,
        ),
        (
            "② 物理目标",
            r"$\rho^\pi=\mathrm{E}_\pi[E]/\mathrm{E}_\pi[Q_u]$；分母为正、同一控制体、统一kWh",
            TEAL,
        ),
        (
            "③ 动作评分",
            r"$G_\rho(s,a)=\widehat E_a-\rho\widehat Q_{u,a}"
            r"+\mathrm{E}[\widehat h(s'_a)]$；另报风险和模型支持",
            GOLD,
        ),
        (
            "④ 可行动作与策略",
            r"$a\in\mathcal{A}_{safe}(s)$；仅当除霜优势超过不确定性裕量和迟滞时触发",
            GREEN,
        ),
    ]
    for index, (title, body, color) in enumerate(rows):
        _box(
            axis,
            (0.06, 0.75 - index * 0.145),
            (0.61, 0.115),
            title,
            body,
            color=color,
            face="#FAFBFC",
            title_size=8.5,
            body_size=7.5,
        )
    _box(
        axis,
        (0.71, 0.46),
        (0.24, 0.36),
        "反例：含糊写法",
        r"$C=\sum_i w_i f_i$"
        + "\n\n把kWh、温度、COP、切换、\n风险和终端残差全部相加。"
        + "\n\n后果：单位不明、重复计数、\n权重不可识别；换工况即变。",
        color=RED,
        face="#FFF6F4",
        title_size=9,
        body_size=7.6,
    )
    _box(
        axis,
        (0.71, 0.22),
        (0.24, 0.16),
        "最低书写契约",
        "符号｜单位｜积分边界｜正负号\n动作依赖｜在线/事后\n支持域｜不确定性",
        color=BLUE,
        face="#F2F6FB",
        title_size=8.5,
        body_size=7.2,
    )
    _footer(
        axis,
        "来源[1,4–6,13–16,28–31,45–48,53–57]",
        "权重只表示真实交换率；安全和服务优先写成约束。",
    )
    _save(figure, output, "figure_08_clear_formula_contract")


def _current_metrics(output: Path) -> dict[str, float]:
    threshold = pd.read_csv(ROOT / "output/test/成本函数/模型筛选/cop_threshold_metrics.csv")
    v25 = pd.read_csv(
        ROOT / "output/成本函数/cost_function_v2.5.csv", usecols=["cycle_name", "t_star"]
    ).drop_duplicates("cycle_name")
    v26 = pd.read_csv(
        ROOT / "output/成本函数/cost_function_v2.6.csv", usecols=["cycle_name", "t_star"]
    ).drop_duplicates("cycle_name")
    v3 = pd.read_csv(
        ROOT / "output/成本函数/cost_function_v3.csv",
        usecols=[
            "cycle_name",
            "candidate_time",
            "near_optimal_1pct",
            "near_optimal_5pct",
            "t_star_model_supported",
            "recommended_rule",
        ],
    )
    v3["candidate_time"] = pd.to_datetime(v3["candidate_time"], format="mixed")

    def median_span(flag: str) -> float:
        selected = v3.loc[v3[flag].fillna(False)]
        spans = selected.groupby("cycle_name")["candidate_time"].agg(
            lambda values: (values.max() - values.min()).total_seconds() / 60
        )
        return float(spans.median())

    merged = v25.merge(v26, on="cycle_name", suffixes=("_25", "_26"))
    for column in ["t_star_25", "t_star_26"]:
        merged[column] = pd.to_datetime(merged[column], format="mixed")
    shift = (merged["t_star_26"] - merged["t_star_25"]).dt.total_seconds().abs() / 60
    supported = v3.groupby("cycle_name")["t_star_model_supported"].first().fillna(False)
    decisions = v3.groupby("cycle_name")["recommended_rule"].first()
    water_threshold = threshold.loc[threshold["method"].str.contains("water")].iloc[0]
    unit_threshold = threshold.loc[threshold["method"].str.contains("refrigerant")].iloc[0]
    metrics = {
        "band_1": median_span("near_optimal_1pct"),
        "band_5": median_span("near_optimal_5pct"),
        "supported": int(supported.sum()),
        "cycles": int(len(supported)),
        "latest_supported": int(decisions.eq("latest_supported_in_1pct_basin").sum()),
        "rb_fallback": int(decisions.eq("rb_fallback").sum()),
        "shift": float(shift.median()),
        "shift_gt5": float((shift > 5).mean()),
        "water5": float(water_threshold["within_5min"]),
        "unit5": float(unit_threshold["within_5min"]),
    }
    pd.DataFrame([metrics]).to_csv(output / "current_project_metrics.csv", index=False)
    return metrics


def plot_uncertainty(metrics: dict[str, float], output: Path) -> None:
    figure, axis = _canvas(
        "模型不准时：先判断“支持域 × 决策间隔”，再决定输出点、窗口或放弃",
        "预测MSE小不代表动作排序稳；真正关键是候选之间的regret差是否大于模型不确定性。",
        6.8,
    )
    cells = [
        (0.10, 0.52, "域内 × 间隔大", "可输出动作点\n仍需跨实验验证", GREEN, "A"),
        (0.52, 0.52, "域外 × 间隔大", "先做稳健情景/定向实验\n不直接贴‘最优’标签", GOLD, "B"),
        (0.10, 0.28, "域内 × 间隔小", "输出近优窗口\n窗口内取latest safe", BLUE, "C"),
        (0.52, 0.28, "域外 × 间隔小", "abstain → 规则基线\n这是信息不足，不是算法失败", RED, "D"),
    ]
    for x, y, title, body, color, label in cells:
        _box(
            axis,
            (x, y),
            (0.36, 0.18),
            f"{label}  {title}",
            body,
            color=color,
            face="#FAFBFC",
            title_size=8.8,
            body_size=7.6,
        )
    axis.text(0.49, 0.765, "模型支持域 →", fontsize=8, color=GREY, ha="center")
    axis.text(
        0.055, 0.49, "决策间隔\n大 ↑", fontsize=8, color=GREY, ha="center", va="center", rotation=90
    )
    data = (
        f"本项目现状：1%近优带中位 {metrics['band_1']:.0f} min；5%带 {metrics['band_5']:.1f} min；"
        f"仅 {metrics['supported']}/{metrics['cycles']} 个最优点位于联合模型支持域。\n"
        f"仅更换供热口径，t*中位移动 {metrics['shift']:.0f} min，"
        f"{metrics['shift_gt5']:.1%} 超过5 min；"
        f"最佳固定COP阈值在±5 min内仅 {metrics['water5']:.1%}/{metrics['unit5']:.1%}。"
    )
    _box(
        axis,
        (0.10, 0.08),
        (0.78, 0.13),
        "数据把当前模型放在 C 与 D，而不是 A",
        data,
        color=RED,
        face="#FFF6F4",
        title_size=8.5,
        body_size=7.2,
    )
    _footer(
        axis,
        "来源[21–34,42,44,46–47]；本项目LOEO与支持域审计",
        "曲线平坦时最诚实的输出是窗口；人为把曲线变尖会制造置信度。",
    )
    _save(figure, output, "figure_09_low_accuracy_decision_matrix")


def plot_counterfactual(output: Path) -> None:
    figure, axis = _canvas(
        "小数据循环最难的不是拟合，而是“提前动作后会怎样”没有被观测",
        "历史规则只在一个时刻除霜；更早分支没有动作覆盖，不能靠同一张真实循环自动补出来。",
        7.1,
    )
    axis.plot([0.08, 0.90], [0.70, 0.70], color=INK, lw=2)
    for x, label in zip([0.10, 0.25, 0.40, 0.55], ["H", "H", "H", "H"], strict=True):
        axis.scatter(x, 0.70, s=260, color=BLUE, edgecolor="white", zorder=3)
        axis.text(x, 0.70, label, color="white", ha="center", va="center", fontweight="bold")
    axis.scatter(0.70, 0.70, s=300, color=RED, edgecolor="white", zorder=3)
    axis.text(0.70, 0.70, "D", color="white", ha="center", va="center", fontweight="bold")
    axis.text(0.82, 0.70, "恢复被观测", fontsize=8, color=GREEN, va="center")
    axis.plot([0.40, 0.40, 0.70], [0.68, 0.47, 0.47], color=GOLD, lw=1.5, ls="--")
    axis.scatter(0.40, 0.47, s=300, facecolor="white", edgecolor=GOLD, lw=2, zorder=3)
    axis.text(0.40, 0.47, "D?", color=GOLD, ha="center", va="center", fontweight="bold")
    axis.text(
        0.48,
        0.49,
        "提前除霜后的ED/QD、恢复\n与下一状态均未观测",
        fontsize=8,
        color=RED,
        va="center",
    )
    rules = [
        ("可识别", "有相近状态下的动作覆盖；\n按完整实验分组交叉拟合。", GREEN),
        ("半识别", "低维物理模型 + ensemble/区间；\n只在训练支持域内排序。", TEAL),
        ("不可识别", "行为动作概率为0：输出区间\n或abstain，不给伪精确点。", RED),
        ("补数据", "近优窗内做±5–10 min安全扰动；\nshadow→小范围在线探索。", GOLD),
    ]
    for index, (title, body, color) in enumerate(rules):
        x = 0.06 + 0.47 * (index % 2)
        y = 0.25 - 0.16 * (index // 2)
        _box(
            axis,
            (x, y),
            (0.41, 0.13),
            title,
            body,
            color=color,
            face="#FAFBFC",
            title_size=8.5,
            body_size=7.0,
        )
    _footer(
        axis, "来源[7–12,21–27,32–36,41]", "无overlap时，复杂模型只会把不可识别包装成更漂亮的数字。"
    )
    _save(figure, output, "figure_10_small_data_counterfactual")


def plot_defrost_objective(output: Path) -> None:
    figure, axis = _canvas(
        "V3：当前数据可直接计算的稳健闭合循环成本",
        "每个候选时刻均用同一物理单位计算；残差裕量由LOEO误差给出，不手调权重。",
        7.3,
    )
    _box(
        axis,
        (0.06, 0.73),
        (0.88, 0.12),
        "闭合边界",
        "从上一轮除霜后的 recovery 起点，积分到候选时刻触发的下一次除霜结束。\n"
        "recovery 已在循环开头实测，不再重复增加一次未来 recovery。",
        color=NAVY,
        face="#F2F6FB",
        body_size=8.5,
    )
    _box(
        axis,
        (0.06, 0.56),
        (0.88, 0.12),
        "电耗上界",
        r"$E_U(\tau)=E_{R+H}^{meas}(\tau)+\widehat E_{prep+D}(\tau)+\Delta E_{0.90}$"
        + "\n"
        + r"$\Delta E_{0.90}=0.00486\ \mathrm{kWh}$",
        color=TEAL,
        face="#F4FAF7",
        body_size=9.2,
    )
    _box(
        axis,
        (0.06, 0.37),
        (0.88, 0.14),
        "有效供热下界",
        r"$Q_L(\tau)=\min(Q_{R+H,w}^{meas}+\widehat Q_{prep,w}-\widehat Q_{D,w},"
        r"\ Q_{R+H,u}^{meas}+\widehat Q_{prep,w}-\widehat Q_{D,w})-\Delta Q_{0.90}$"
        + "\n"
        + r"$\Delta Q_{0.90}=0.03580\ \mathrm{kWh}$",
        color=GOLD,
        face="#FFF9F0",
        body_size=8.2,
    )
    _box(
        axis,
        (0.06, 0.17),
        (0.42, 0.14),
        "主指标与可解释差值",
        r"$J_R(\tau)=E_U(\tau)/Q_L(\tau)$" + "\n"
        r"$G(\tau)=E_U(\tau)-J_R^*Q_L(\tau)\geq0$",
        color=GREEN,
        face="#F4FAF7",
        body_size=7.8,
    )
    _box(
        axis,
        (0.52, 0.17),
        (0.42, 0.14),
        "最终离线标签",
        r"$\mathcal{W}_{1\%}=\{\tau:J_R(\tau)\leq1.01J_R^*\}$"
        + "\n取与全局最小点同一连续盆地中最晚的支持域点；无则回退RB。",
        color=BLUE,
        face="#F2F6FB",
        body_size=7.8,
    )
    _footer(
        axis,
        "证据索引：source_matrix.csv；裕量：配对LOEO残差",
        "V3现在是稳健离线标签器；它还不是有在线最优性保证的SMDP控制器。",
    )
    _save(figure, output, "figure_11_recommended_defrost_objective")


def plot_acceptance_framework(metrics: dict[str, float], output: Path) -> None:
    figure, axis = _canvas(
        "继续设计成本函数的验收框架：八道门，未通过就降级输出",
        "顺序不能反：先物理与因果，再不确定性和区分度，最后才讨论模型复杂度。",
        7.3,
    )
    gates = [
        ("1 闭合边界", "积分从再生状态到下一再生状态；储热差可解释", "PASS", GREEN),
        ("2 统一控制体", "E与Q口径、符号、单位、准备/除霜/恢复不重不漏", "PASS", GREEN),
        ("3 在线因果", "仍借用实测准备时长和规则除霜时长", "FAIL", RED),
        ("4 动作相关转移", "不同τ会改变ED/QD/恢复和下一循环reset", "FAIL", RED),
        (
            "5 联合支持域",
            f"当前仅{metrics['supported']}/{metrics['cycles']}个t*在组件模型联合支持域",
            "FAIL",
            RED,
        ),
        (
            "6 决策可区分",
            f"1%近优带中位{metrics['band_1']:.0f} min：输出点的证据不足",
            "WARN",
            GOLD,
        ),
        ("7 稳健排名", "跨实验、bootstrap、口径敏感性后候选顺序仍稳定", "FAIL", RED),
        ("8 样本外策略价值", "在新循环上优于RB且服务/安全不退化", "TODO", BLUE),
    ]
    for index, (title, body, status, color) in enumerate(gates):
        col, row = index % 2, index // 2
        x, y = 0.055 + col * 0.47, 0.75 - row * 0.16
        _box(
            axis,
            (x, y),
            (0.42, 0.125),
            title,
            body,
            color=color,
            face="#FAFBFC",
            title_size=8.4,
            body_size=6.9,
        )
        axis.text(
            x + 0.39, y + 0.098, status, fontsize=7.0, color=color, ha="right", fontweight="bold"
        )
    _box(
        axis,
        (0.12, 0.08),
        (0.76, 0.13),
        "当前最准确的定位",
        f"V3是稳健离线标签器：{metrics['latest_supported']}/{metrics['cycles']}个循环取支持域内近优晚点，"
        f"{metrics['rb_fallback']}个回退RB。\n"
        "它能计算当前全部循环，但还不是在线SMDP控制器。",
        color=NAVY,
        face="#F2F6FB",
        title_size=8.8,
        body_size=7.5,
    )
    _footer(
        axis,
        "来源：source_matrix.csv；当前项目成本、支持域与阈值审计",
        "下一步：在近优窗内安全改变τ，采集完整动作后果。",
    )
    _save(figure, output, "figure_12_cost_design_acceptance_framework")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = pd.read_csv(args.output / "source_matrix.csv")
    counts = source["source_type"].value_counts()
    assert source["id"].is_unique
    assert counts.get("peer_reviewed", 0) >= 60
    assert counts.get("xiaohongshu", 0) >= 30
    assert source.loc[source["source_type"].eq("github_official"), "source"].nunique() >= 10
    metrics = _current_metrics(args.output)
    plot_evidence_map(source, args.output)
    plot_source_atlas(source, args.output)
    plot_global_transient(args.output)
    plot_future_decline(args.output)
    plot_cost_anatomy(args.output)
    plot_formula_contract(args.output)
    plot_uncertainty(metrics, args.output)
    plot_counterfactual(args.output)
    plot_defrost_objective(args.output)
    plot_acceptance_framework(metrics, args.output)


if __name__ == "__main__":
    main()
