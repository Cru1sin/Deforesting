"""Render the historical RB strategy as one Chinese operating-condition table."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

matplotlib.use("Agg")

OUTPUT = Path("output/RB规则说明")
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


if __name__ == "__main__":
    plot_rb_strategy()
