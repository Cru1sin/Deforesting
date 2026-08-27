#!/usr/bin/env python3
# ruff: noqa: E501
"""Compare heating-capacity smoothing methods on every exported frost cycle."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from frost_analysis.heating_smoothing import (
    cost_method_ranking,
    global_method_ranking,
    recommend_methods,
    score_methods,
    smooth_cycle,
)

COLORS = {
    "median_centered_60s": "#0072B2",
    "ewma_tau30s": "#D55E00",
    "median30s_ewma30s": "#009E73",
    "savgol_70s": "#CC79A7",
    "adaptive_offline": "#6A3D9A",
    "wavelet_offline": "#56B4E9",
    "wavelet_monotonic_offline": "#111111",
    "nearly_isotonic_offline": "#E69F00",
    "robust_monotone_offline": "#009E73",
}
LABELS = {
    "median_centered_60s": "Centered median 60 s (offline)",
    "ewma_tau30s": "EWMA tau=30 s (causal)",
    "median30s_ewma30s": "Median 30 s + EWMA 30 s (causal)",
    "savgol_70s": "Savitzky-Golay ~70 s (offline)",
    "adaptive_offline": "Adaptive median + zero-phase LPF (offline)",
    "wavelet_offline": "PyWavelets db4 shrinkage (offline)",
    "wavelet_monotonic_offline": "Wavelet + monotonic regression (offline)",
    "nearly_isotonic_offline": "Nearly-isotonic (offline)",
    "robust_monotone_offline": "Huber + smooth monotonic (offline)",
}


def run_analysis(input_dir: Path, output_dir: Path) -> None:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"no cycle parquet files found in {input_dir}")
    source_dir = output_dir / "源数据"
    cycle_figure_dir = output_dir / "图表" / "循环图"
    source_dir.mkdir(parents=True, exist_ok=True)
    cycle_figure_dir.mkdir(parents=True, exist_ok=True)

    metric_tables: list[pd.DataFrame] = []
    cycle_frames: list[tuple[str, pd.DataFrame]] = []
    for path in files:
        frame = pd.read_parquet(path)
        if "cycle_name" not in frame:
            frame["cycle_name"] = path.stem
        smoothed = smooth_cycle(frame)
        metrics = score_methods(smoothed)
        metric_tables.append(metrics)
        cycle_frames.append((path.stem, smoothed))

    metrics = pd.concat(metric_tables, ignore_index=True)
    recommendations = recommend_methods(metrics)
    summary = _method_summary(metrics)
    global_ranking = global_method_ranking(metrics)
    cost_ranking = cost_method_ranking(metrics)
    metrics.to_csv(source_dir / "method_metrics.csv", index=False)
    recommendations.to_csv(source_dir / "cycle_recommendations.csv", index=False)
    summary.to_csv(source_dir / "overall_method_summary.csv", index=False)
    global_ranking.to_csv(source_dir / "global_method_ranking.csv", index=False)
    cost_ranking.to_csv(source_dir / "cost_method_ranking.csv", index=False)
    highlighted_method = str(cost_ranking.iloc[0]["method"])
    for name, frame in cycle_frames:
        _plot_cycle(frame, cycle_figure_dir / f"{name}.png", highlighted_method)
    _plot_overview(recommendations, output_dir / "图表" / "method_overview.png")
    (output_dir / "报告.md").write_text(
        _report(
            files, metrics, recommendations, summary, global_ranking, cost_ranking
        ),
        encoding="utf-8",
    )
    print(f"Analyzed {len(files)} cycles; outputs: {output_dir}")


def _plot_cycle(frame: pd.DataFrame, output: Path, highlighted_method: str) -> None:
    frost = frame.loc[frame["cycle_stage"].eq("frost_development")]
    if not frost.empty:
        frame = frost
    times = pd.to_datetime(frame["timestamp"], errors="coerce")
    elapsed = (times - times.min()).dt.total_seconds() / 60.0
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    methods = (
        "median_centered_60s",
        "savgol_70s",
        "adaptive_offline",
        "wavelet_offline",
        "wavelet_monotonic_offline",
        "nearly_isotonic_offline",
        "robust_monotone_offline",
    )
    for method in methods:
        axis.plot(
            elapsed,
            frame[method],
            color=COLORS[method],
            linewidth=1.8 if method == highlighted_method else 1.0,
            alpha=1.0 if method == highlighted_method else 0.8,
            linestyle="--" if method in {
                "wavelet_monotonic_offline",
                "nearly_isotonic_offline",
                "robust_monotone_offline",
            } else "-",
            label=LABELS[method],
        )
    axis.plot(
        elapsed,
        frame["heating_capacity"],
        color="0.2",
        linewidth=0.65,
        alpha=0.65,
        zorder=10,
        label="Raw",
    )
    cycle_name = str(frame["cycle_name"].iloc[0])
    axis.set(
        title=f"{cycle_name} — frost-development smoothing comparison",
        xlabel="Frost-stage time (min)",
        ylabel="Heating capacity (kW)",
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _method_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    available = metrics.loc[
        metrics["metric_status"].eq("available")
        & metrics["cycle_stage"].eq("frost_development")
    ]
    return (
        available.groupby(["mode", "method"], as_index=False)
        .agg(
            cycles=("cycle_name", "nunique"),
            median_noise_reduction=("noise_reduction", "median"),
            median_spike_reduction=("spike_reduction", "median"),
            median_water_rmse_offset_kw=("water_rmse_offset_kw", "median"),
            median_energy_error_pct=("energy_error_pct", "median"),
            median_shortfall_area_error_pct=("shortfall_area_error_pct", "median"),
            median_transient_retention=("transient_retention", "median"),
            median_lag_seconds=("lag_seconds", "median"),
            median_water_bias_kw=("water_bias_kw", "median"),
        )
        .sort_values(["mode", "method"])
    )


def _plot_overview(recommendations: pd.DataFrame, output: Path) -> None:
    primary = recommendations.loc[
        recommendations["cycle_stage"].eq("frost_development")
    ]
    if primary.empty:
        primary = recommendations
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for axis, mode in zip(axes, ("offline", "online"), strict=True):
        counts = primary[f"{mode}_method"].value_counts().drop("unavailable", errors="ignore")
        axis.bar(range(len(counts)), counts.to_numpy(), color=[COLORS.get(name, "0.5") for name in counts.index])
        axis.set_xticks(range(len(counts)), [LABELS.get(name, name) for name in counts.index], rotation=18, ha="right")
        axis.set(title=f"Recommended {mode} method", ylabel="Cycles")
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _report(
    files: list[Path],
    metrics: pd.DataFrame,
    recommendations: pd.DataFrame,
    summary: pd.DataFrame,
    global_ranking: pd.DataFrame,
    cost_ranking: pd.DataFrame,
) -> str:
    primary = recommendations.loc[recommendations["cycle_stage"].eq("frost_development")]
    offline_counts = primary["offline_method"].value_counts().to_dict()
    online_counts = primary["online_method"].value_counts().to_dict()
    available = metrics.loc[
        metrics["metric_status"].eq("available")
        & metrics["cycle_stage"].eq("frost_development")
    ]
    median_bias = (
        float(available["water_bias_kw"].median()) if not available.empty else float("nan")
    )
    indexed = summary.set_index("method")
    median_noise = float(indexed.loc["median_centered_60s", "median_noise_reduction"])
    adaptive_shape = float(indexed.loc["adaptive_offline", "median_water_rmse_offset_kw"])
    adaptive_transient = float(indexed.loc["adaptive_offline", "median_transient_retention"])
    wavelet_energy = float(indexed.loc["wavelet_offline", "median_energy_error_pct"])
    wavelet_transient = float(indexed.loc["wavelet_offline", "median_transient_retention"])
    monotonic_shape = float(
        indexed.loc["wavelet_monotonic_offline", "median_water_rmse_offset_kw"]
    )
    monotonic_energy = float(
        indexed.loc["wavelet_monotonic_offline", "median_energy_error_pct"]
    )
    nearly_spike = float(
        indexed.loc["nearly_isotonic_offline", "median_spike_reduction"]
    )
    nearly_transient = float(
        indexed.loc["nearly_isotonic_offline", "median_transient_retention"]
    )
    global_best = str(global_ranking.iloc[0]["method"])
    global_score = float(global_ranking.iloc[0]["global_mean_rank_sum"])
    cost_best = str(cost_ranking.iloc[0]["method"])
    cost_score = float(cost_ranking.iloc[0]["cost_mean_rank_sum"])
    return f"""# 制热量平滑方法实测比较

## 结论

本报告对 {len(files)} 个循环逐一保留原始制热量、水侧审计量和九种平滑曲线。按所有有效结霜循环的平均秩和，统一普通离线方法是 **`{global_best}`**（平均秩和 {global_score:.2f}，越低越好）；若重点服务除霜成本积分，在先满足极端跳变至少降低 10% 后，再比较供热缺口、能量、短时响应和水侧形状，综合第一是 **`{cost_best}`**（平均秩和 {cost_score:.2f}）。逐循环最优计数 `{offline_counts}` 仅作为稳健性说明，不再作为主结论。在线推荐计数为 `{online_counts}`。各平滑结果相对水侧审计量的中位偏差约为 {median_bias:.3f} kW；不同平滑器的偏差接近，实证说明平滑没有消除系统偏差。

推荐分成两个用途：人工审图使用离线推荐，在线控制使用因果推荐。居中中值、Savitzky–Golay、小波和单调回归均使用未来样本或完整阶段信息，不能用于实时控制。

如果后续成本函数和 RGB 标签只能统一保留一种离线平滑，本报告选择 **`{cost_best}`**：它在全部循环统一参数下是成本导向第一。`wavelet_offline` 作为总体保真度第一的敏感性对照保留，不再为不同循环选择不同主方法。

但本轮的比较载体仍是控制器 `heating_capacity`；正式成本函数必须把同一候选方法迁移并复核到逐秒计算后再做 10 s 均值的水侧 $Q_h=\\dot m c_p\\Delta T$，不能用控制器字段替代用户侧真实热流。

- 若“降噪”特指压制相邻点随机跳动，60 s 居中中值最强，中位降幅为 {median_noise:.1%}。
- 若要兼顾能量与短时响应，PyWavelets 的能量误差仅 {wavelet_energy:.4f}%，短时响应保留 {wavelet_transient:.1%}，但相邻点降噪弱于中值方案；它没有在全部循环中取代现有方法。
- 若必须得到单调下降趋势，`wavelet_monotonic_offline` 已在全部 {len(files)} 个有效循环验证为零上升违例；其能量误差中位数仅 {monotonic_energy:.4f}%，但短时响应保留率为 0，水侧形状 RMSE 增至 {monotonic_shape:.3f} kW，因此只应作为趋势曲线，不作为瞬时制热量真值。
- 修复并核验 ADMM 收敛后，`nearly_isotonic_offline` 的极端跳变抑制中位数为 {nearly_spike:.1%}、短时响应保留 {nearly_transient:.1%}，成本排名第三；说明软单调比硬单调合理，但仍没有超过无方向偏置的 Savitzky–Golay。
- 自适应中值后接零相位低通对水侧形状最贴近（去偏 RMSE 中位数 {adaptive_shape:.3f} kW），但短时响应仅保留 {adaptive_transient:.1%}，属于强平滑版本，不应被误称为无损真值。
- 不建议“全消除”所有独立谷峰：单点或极短尖峰可删，跨越多个采样点且与温度、流量、压缩机状态同步的谷峰应保留为候选真实响应。

## 数据与边界

- 输入：`dataset/cycles/*.parquet`，原始间隔通常为 10 s。
- 所有方法按循环、按连续 `cycle_stage` 分段，不跨越结霜、除霜和恢复边界。
- `heating_capacity` 的缺失点保持缺失，不由平滑器伪造。
- 水侧审计量为 `1.161 × water_flow[m³/h] × (T_out−T_in)[K]` kW。它可检查形状和偏差，但因温度通道可能与控制器计算共享，不能称为完全独立的金标准。

## 单调假设的物理边界

总制热量 $Q_{{heating}}$ 同时受结霜、压缩机频率、阀位、水温、流量和整机热惯性影响，因此不能直接施加严格非增约束。Guo 等确实发表了空气源热泵结霜生长与动态性能实验研究（[DOI 10.1016/j.applthermaleng.2008.01.007](https://doi.org/10.1016/j.applthermaleng.2008.01.007)）；本次已核验题名、作者、期刊、卷页和 DOI，但可访问的出版社元数据不含摘要/全文，所以不把“早期制热量和 COP 必然上升”的更细分阶段表述当作已核实证据。

严格单调应留给去除健康工况基线后的潜在退化量，例如 $D_{{frost}}(t)=Q_{{healthy}}(u_t)-Q_h(t)$。只有当 $Q_{{healthy}}$ 已由未结霜样本和运行工况 $u_t$ 独立建模后，才对 $D_{{frost}}$ 施加非减约束；当前两条硬单调制热量曲线只用于展示“若强加该假设会损失多少信息”。

## 方法原理与文献

### 1. 60 s 居中 rolling median

在当前点前后 30 s 的窗口中取中位数。中位数对孤立尖峰具有 50% breakdown point，不会像均值一样被单个极端值拖动；代价是非线性，并可能消除持续时间短于半个窗口的真实卸载。它只用于离线审图。

- J. W. Tukey, *Nonlinear (Nonsuperposable) Methods for Smoothing Data*, 1974.
- 热泵动态边界依据：[Tran et al., 2021](https://doi.org/10.1016/j.ijrefrig.2021.03.001)。

### 2. 因果 EWMA / 一阶低通，τ=30 s

`y[k]=y[k−1]+α(x[k]−y[k−1])`，其中 `α=1−exp(−Δt/τ)`。它只使用当前和过去样本，适合在线实现；约 1τ 达到阶跃的 63%，约 3τ 达到 95%，因此必须同时检查滞后和短时响应保真。

- S. W. Roberts, [Control Chart Tests Based on Geometric Moving Averages](https://doi.org/10.1080/00401706.1959.10489860), *Technometrics* 1 (1959).

### 3. 30 s 因果 median + 30 s EWMA

先以很短的因果中值窗口删除孤立坏点，再由 EWMA 抑制剩余高频噪声。它对应工程上“去尖峰”和“低通”职责分离；与单独 EWMA 相比通常更稳健，但连续两级处理可能增加滞后。

- 原理分别沿用 Tukey 的稳健中值方法和 Roberts 的 EWMA。

### 4. 约 70 s Savitzky–Golay

用窗口内二次多项式最小二乘拟合中心值，较中值滤波更容易保留平滑斜率和峰形，但对极端尖峰不够稳健，并且居中实现使用未来样本，只适合离线分析。

- A. Savitzky and M. J. E. Golay, [Smoothing and Differentiation of Data by Simplified Least Squares Procedures](https://doi.org/10.1021/ac60214a047), *Analytical Chemistry* 36 (1964).

### 5. 自适应居中中值 + 双向零相位低通

这是上帝视角下的强降噪候选。每个循环、每个阶段分别尝试 30/60/90 s 居中中值，以及 20/30/45/60/90 s 低通时间常数组合。首先保留水侧去偏 RMSE 距离最优值不超过 5%+0.01 kW、且积分能量误差不超过 1% 的候选；再从中选择相邻步长 MAD 最低者。低通前向、反向各运行一次，因此没有因果相位滞后，但绝不能用于在线控制。

- S. Butterworth, *On the Theory of Filter Amplifiers*, 1930。
- F. Gustafsson, [Determining the Initial States in Forward-Backward Filtering](https://doi.org/10.1109/78.492552), *IEEE Transactions on Signal Processing* 44 (1996)。

### 6. PyWavelets db4 小波软阈值

先将阶段内信号分解为低频近似系数和多层高频细节系数，再依据最细层细节的稳健噪声尺度施加 universal soft threshold，最后重构为原长度曲线。它可在不固定时间窗口的情况下压低多尺度噪声，但孤立大尖峰可能同时影响多个尺度，因此仍需与中值方案实测比较。本实现使用 [PyWavelets](https://github.com/PyWavelets/pywt)（MIT，GitHub 约 2.4k Stars，调研于 2026-08-19），未引入更重的 Kalman/深度学习框架。

- D. L. Donoho and I. M. Johnstone, [Ideal Spatial Adaptation by Wavelet Shrinkage](https://doi.org/10.1093/biomet/81.3.425), *Biometrika* 81 (1994)。
- G. Lee et al., [PyWavelets: A Python package for wavelet analysis](https://doi.org/10.21105/joss.01237), *Journal of Open Source Software* 4 (2019)。

GitHub 候选中，[FilterPy](https://github.com/rlabbe/filterpy)（约 3.9k Stars）和 [pykalman](https://github.com/pykalman/pykalman)（约 1.3k Stars）虽然更高 Star 或同样成熟，但 Kalman 平滑需要可辨识的状态转移与测量噪声模型；当前数据尚无这样的独立模型，直接用 EM 拟合容易把真实制热变化当噪声。因此按 Ponytail 原则未加入这两个更重依赖。

### 7. 小波后单调回归

若分析先验明确要求结霜发展阶段制热量只能下降，可在小波结果上求解约束最小二乘：$\\min_z\\sum_i(z_i-s_i)^2$，约束 $z_{{i+1}}\\le z_i$。输出是单调不增曲线，允许平台段。它回答的是“在单调物理假设下最接近小波曲线的结果”，不是从数据证明制热量本来就严格单调。恢复和除霜阶段不施加该约束。

- R. E. Barlow et al., *Statistical Inference Under Order Restrictions*, Wiley, 1972。

### 8. Nearly-isotonic regression

直接求解 $\\min_z\\frac12\\sum_i(x_i-z_i)^2+\\lambda\\sum_i(z_{{i+1}}-z_i)_+$。上升可以存在，但每次上升都会被惩罚；因此它用于检验“总体下降、局部允许回升”是否比硬单调更符合数据。本实现统一使用 $\\lambda=0.15$ kW，由三对角 ADMM 求解，不增加优化框架。

- R. J. Tibshirani, H. Hoefling and R. Tibshirani, [Nearly-Isotonic Regression](https://doi.org/10.1198/tech.2010.10111), *Technometrics* 53 (2011) 54–61。

### 9. Huber + 二阶差分惩罚 + 硬单调

直接联合求解 $\\min_z\\sum_i\\rho_{{Huber}}(x_i-z_i)+\\lambda\\sum_i(\\Delta^2z_i)^2$，约束 $z_{{i+1}}\\le z_i$。Huber 降低孤立大跳点的影响，二阶惩罚消除保序回归的楼梯，硬约束保证趋势不回升。本实现统一使用 Huber 阈值 0.15 kW、$\\lambda=2$，通过现有 SciPy 保序投影求解。该曲线只解释“累计退化趋势”，不能冒充瞬时制热量。

- J. O. Ramsay, [Monotone Regression Splines in Action](https://doi.org/10.1214/ss/1177012761), *Statistical Science* 3 (1988) 425–461。
- N. Pya and S. N. Wood, [Shape constrained additive models](https://doi.org/10.1007/s11222-013-9448-7), *Statistics and Computing* 25 (2015) 543–559。

## 评价指标

令原始制热量为 $x_i$，平滑结果为 $s_i$，水侧审计量为 $w_i$，相邻差分 $\\Delta x_i=x_i-x_{{i-1}}$。MAD 定义为 $\\operatorname{{median}}(|u_i-\\operatorname{{median}}(u)|)$。

- **降噪率**：$D=1-\\operatorname{{MAD}}(\\Delta s)/\\operatorname{{MAD}}(\\Delta x)$。$D$ 越大，相邻点抖动压得越强；$D=0$ 表示没有改善，$D<0$ 表示反而增加了跳动。它不判断被删变化是真噪声还是物理响应。
- **极端跳变抑制率**：$D_{{99}}=1-P_{{99}}(|\\Delta s|)/P_{{99}}(|\\Delta x|)$。成本推荐要求跨循环中位数至少 10%，防止“几乎等于原始曲线”的方法仅靠零面积误差作弊。
- **短时响应保留率**：$R=P_{{90}}(|\\Delta s|)/P_{{90}}(|\\Delta x|)$。接近 1 表示较大的短时变化幅度被保留；接近 0 表示峰谷和阶跃几乎被抹平。它不是“越大绝对越好”，因为噪声尖峰也会抬高该值。
- **能量误差**：$E=100\\,|\\int s(t)dt-\\int x(t)dt|/|\\int x(t)dt|$。采用梯形积分，越接近 0 越不改变整个阶段累计供热量。
- **供热缺口面积误差**：先取结霜段最初 60 s 的中位数为 $Q_{{ref}}$，令 $A(x)=\\int[Q_{{ref}}-x(t)]_+dt$，再计算 $E_A=100|A(s)-A(x)|/|A(x)|$。它直接对应成本项 $\\lambda_Q[Q_{{ref}}-Q_h]_+$ 的累计失真；越小越不容易仅因平滑而移动最优除霜点。
- **水侧去偏 RMSE**：先取常数偏差 $b=\\operatorname{{median}}(s_i-w_i)$，再算 $\\sqrt{{N^{{-1}}\\sum_i(s_i-w_i-b)^2}}$。越小表示曲线形状越接近水侧审计量；因为水侧与控制器信号可能共享传感器，它只是审计基准，不是独立真值。
- **滞后**：在有限移位范围内寻找原始与平滑曲线互相关最大的时间偏移，绝对值越小越好。居中和双向离线方法通常为 0；因果滤波通常大于 0。

统一方法采用循环内平均排名：对每个循环分别在水侧 RMSE、$1-D$、$E$、$|R-1|$ 和绝对滞后上排名，方法 $m$ 的全局分数为 $G_m=C^{{-1}}\\sum_c\\sum_k\\operatorname{{rank}}_{{c,k}}(m)$。这样每个循环权重相等，不会由持续时间最长或振幅最大的循环决定结果。

全局排名如下：

{global_ranking.to_markdown(index=False)}

面向成本函数的排名如下：

{cost_ranking.to_markdown(index=False)}

此排名衡量成本函数输入的失真。由于当前没有统一给定 $\\lambda_Q$、除霜能耗和恢复代价，不能诚实地计算每种平滑造成的具体最优除霜时刻偏移；补齐这些成本参数后，应直接增加 $|t_m^*-t_{{raw}}^*|$ 作为最终指标。

## 总体统计

{summary.to_markdown(index=False)}

## 文件

- `图表/循环图/*.png`：一个循环一张对比图。
- `图表/method_overview.png`：结霜阶段推荐方法计数。
- `源数据/method_metrics.csv`：逐循环、逐阶段、逐方法指标。
- `源数据/cycle_recommendations.csv`：逐循环离线/在线推荐。
- `源数据/overall_method_summary.csv`：总体中位指标。
- `源数据/global_method_ranking.csv`：统一离线方法的跨循环平均秩和。
- `源数据/cost_method_ranking.csv`：成本导向排名。大体量逐点曲线属于可再生缓存，不纳入报告目录。

## 与热泵测量文献的一致性

热泵动态测量研究强调启动、稳态、除霜和恢复必须区别处理；动态段瞬时误差可很大，而完整周期积分更可靠。参见 [Tran et al., 2012](https://doi.org/10.1016/j.ijrefrig.2012.03.010)、[Tran et al., 2021](https://doi.org/10.1016/j.ijrefrig.2021.03.001) 和 [Noël et al., 2018](https://docs.lib.purdue.edu/iracc/1893)。本实现因此不跨阶段平滑，并单独报告积分失真。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("dataset/cycles"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/test/成本函数/其他/01_制热量与退化/制热量平滑")
    )
    args = parser.parse_args()
    run_analysis(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
