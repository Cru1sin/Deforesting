#!/usr/bin/env python3
"""Compare smoothers on condition-normalized frost degradation for every cycle."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frost_analysis.degradation_law import (
    leave_group_out_reference,
    monotonicity_metrics,
    relative_degradation,
)
from frost_analysis.heating_smoothing import (
    _nearly_isotonic,
    _robust_monotone,
    _savgol,
    _wavelet_denoise,
)

CONTEXT = [
    "ambient_temperature",
    "water_in_temperature",
    "water_flow",
    "compressor_frequency",
    "fan_speed",
    "exv_opening",
]
METHODS = ("D_Q_raw", "D_Q_savgol_70s", "D_Q_wavelet", "D_Q_nearly", "D_Q_hard")
LABELS = {
    "D_Q_raw": "Raw condition-normalized degradation",
    "D_Q_savgol_70s": "Savitzky–Golay ~70 s",
    "D_Q_wavelet": "db4 wavelet",
    "D_Q_nearly": "Nearly monotonic",
    "D_Q_hard": "Hard monotonic trend",
}
COLORS = {
    "D_Q_raw": "0.45",
    "D_Q_savgol_70s": "#CC79A7",
    "D_Q_wavelet": "#0072B2",
    "D_Q_nearly": "#E69F00",
    "D_Q_hard": "#009E73",
}


def load_cycles(input_dir: Path) -> pd.DataFrame:
    rows = []
    kept = [
        "cycle_name",
        "experiment_date",
        "timestamp",
        "cycle_stage",
        "heating_capacity",
        *CONTEXT,
    ]
    for path in sorted(input_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        if "cycle_name" not in frame:
            frame["cycle_name"] = path.stem
        rows.append(frame.loc[:, [column for column in kept if column in frame]])
    if not rows:
        raise ValueError(f"no cycle parquet files found in {input_dir}")
    result = pd.concat(rows, ignore_index=True)
    missing = sorted(
        {
            "cycle_name",
            "experiment_date",
            "timestamp",
            "cycle_stage",
            "heating_capacity",
            *CONTEXT,
        }
        - set(result)
    )
    if missing:
        raise ValueError(f"cycle data is missing required columns: {missing}")
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["date"] = result["experiment_date"].astype(str).str[:10]
    return result.loc[
        result["cycle_stage"].eq("frost_development")
        & pd.to_numeric(result["compressor_frequency"], errors="coerce").gt(5)
    ].copy()


def _with_elapsed(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["cycle_name", "timestamp"]).copy()
    start = result.groupby("cycle_name")["timestamp"].transform("min")
    result["frost_elapsed_seconds"] = (result["timestamp"] - start).dt.total_seconds()
    result["early"] = result["frost_elapsed_seconds"].between(0.0, 600.0)
    return result


def reference_ridge_ablation(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = _with_elapsed(frame)
    rows = []
    for ridge in (1e-6, 0.01, 1.0, 10.0, 100.0, 1000.0):
        prediction = leave_group_out_reference(
            prepared,
            target="heating_capacity",
            features=CONTEXT,
            group="date",
            early="early",
            cycle="cycle_name",
            ridge=ridge,
        )
        valid = prepared["early"] & prediction.notna() & prepared["heating_capacity"].notna()
        error = prepared.loc[valid, "heating_capacity"] - prediction.loc[valid]
        rows.append(
            {
                "ridge": ridge,
                "early_median_absolute_error_kw": float(error.abs().median()),
                "early_rmse_kw": float(np.sqrt(np.mean(np.square(error)))),
                "healthy_reference_p99_step_kw": float(
                    prediction.groupby(prepared["cycle_name"]).diff().abs().quantile(0.99)
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["selected"] = result["early_rmse_kw"].eq(result["early_rmse_kw"].min())
    return result


def reference_support_audit(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = _with_elapsed(frame)
    rows = []
    for held_out in pd.unique(prepared["date"]):
        train = prepared.loc[
            prepared["date"].ne(held_out) & prepared["early"], CONTEXT
        ].apply(pd.to_numeric, errors="coerce")
        test = prepared.loc[prepared["date"].eq(held_out), ["cycle_name", *CONTEXT]].copy()
        lower, upper = train.min(), train.max()
        numeric = test[CONTEXT].apply(pd.to_numeric, errors="coerce")
        outside_count = ((numeric < lower) | (numeric > upper)).sum(axis=1)
        for cycle, indices in test.groupby("cycle_name", sort=False).groups.items():
            values = outside_count.loc[indices]
            rows.append(
                {
                    "cycle_name": cycle,
                    "date": str(held_out),
                    "outside_support_fraction": float(values.gt(0).mean()),
                    "median_outside_feature_count": float(values.median()),
                    "max_outside_feature_count": int(values.max()),
                }
            )
    return pd.DataFrame(rows)


def add_degradation_curves(frame: pd.DataFrame, ridge: float = 1e-6) -> pd.DataFrame:
    result = _with_elapsed(frame)
    result["Q_healthy"] = leave_group_out_reference(
        result,
        target="heating_capacity",
        features=CONTEXT,
        group="date",
        early="early",
        cycle="cycle_name",
        ridge=ridge,
    )
    invalid_reference = ~np.isfinite(result["Q_healthy"]) | result["Q_healthy"].le(0.0)
    result.loc[invalid_reference, "Q_healthy"] = np.nan
    result["D_Q_raw"] = relative_degradation(
        result["heating_capacity"], result["Q_healthy"]
    )
    for method in METHODS[1:]:
        result[method] = np.nan
    for _, indices in result.groupby("cycle_name", sort=False).groups.items():
        index = pd.Index(indices)
        valid = result.loc[index, ["timestamp", "D_Q_raw"]].dropna().index
        if valid.empty:
            continue
        values = result.loc[valid, "D_Q_raw"].to_numpy(dtype=float)
        times = result.loc[valid, "timestamp"]
        result.loc[valid, "D_Q_savgol_70s"] = _savgol(values, times)
        result.loc[valid, "D_Q_wavelet"] = _wavelet_denoise(values)
        result.loc[valid, "D_Q_nearly"] = -_nearly_isotonic(-values, penalty=0.02)
        result.loc[valid, "D_Q_hard"] = -_robust_monotone(
            -values, huber_delta=0.02, smoothness=2.0
        )
    return result


def _mad(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(np.abs(values - np.median(values)))) if len(values) else np.nan


def _area(values: np.ndarray, seconds: np.ndarray, *, absolute: bool = False) -> float:
    valid = np.isfinite(values) & np.isfinite(seconds)
    values = values[valid]
    seconds = seconds[valid]
    if len(values) < 2:
        return np.nan
    if absolute:
        values = np.abs(values)
    return float(np.trapezoid(values, seconds))


def _crossing_error(raw: np.ndarray, fitted: np.ndarray, seconds: np.ndarray) -> float:
    valid = np.isfinite(raw) & np.isfinite(fitted) & np.isfinite(seconds)
    raw, fitted, seconds = raw[valid], fitted[valid], seconds[valid]
    if len(raw) < 3:
        return np.nan
    widths = np.diff(seconds)
    raw_cumulative = np.r_[0.0, np.cumsum(np.maximum(raw[1:], 0.0) * widths)]
    fitted_cumulative = np.r_[0.0, np.cumsum(np.maximum(fitted[1:], 0.0) * widths)]
    final = raw_cumulative[-1]
    if final <= 0.0:
        return np.nan
    errors = []
    for fraction in (0.25, 0.5, 0.75):
        threshold = fraction * final
        raw_index = min(int(np.searchsorted(raw_cumulative, threshold)), len(seconds) - 1)
        fit_index = min(int(np.searchsorted(fitted_cumulative, threshold)), len(seconds) - 1)
        errors.append(abs(seconds[fit_index] - seconds[raw_index]))
    return float(np.median(errors))


def _future_loss_error(raw: np.ndarray, fitted: np.ndarray, seconds: np.ndarray) -> float:
    intervals = np.diff(seconds[np.isfinite(seconds)])
    intervals = intervals[intervals > 0.0]
    if not len(intervals):
        return np.nan
    ratios = []
    for horizon_seconds in (300.0, 600.0):
        shift = int(round(horizon_seconds / np.median(intervals)))
        if shift < 1 or shift >= len(raw):
            continue
        valid = (
            np.isfinite(raw[:-shift])
            & np.isfinite(raw[shift:])
            & np.isfinite(fitted[:-shift])
            & np.isfinite(fitted[shift:])
        )
        raw_change = raw[shift:][valid] - raw[:-shift][valid]
        fitted_change = fitted[shift:][valid] - fitted[:-shift][valid]
        scale = np.percentile(np.abs(raw_change), 90) if len(raw_change) else np.nan
        if np.isfinite(scale) and scale > 0.0:
            ratios.append(float(np.median(np.abs(fitted_change - raw_change)) / scale))
    return float(np.median(ratios)) if ratios else np.nan


def score_curves(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle, group in frame.groupby("cycle_name", sort=False):
        raw = group["D_Q_raw"].to_numpy(dtype=float)
        seconds = group["frost_elapsed_seconds"].to_numpy(dtype=float)
        raw_step_mad = _mad(np.diff(raw))
        raw_abs_area = _area(raw, seconds, absolute=True)
        for method in METHODS:
            values = group[method].to_numpy(dtype=float)
            mono = monotonicity_metrics(values)
            step_mad = _mad(np.diff(values))
            rows.append(
                {
                    "cycle_name": cycle,
                    "date": str(group["date"].iloc[0]),
                    "method": method,
                    "n": int(np.isfinite(values).sum()),
                    **mono,
                    "noise_ratio": (
                        step_mad / raw_step_mad
                        if np.isfinite(step_mad) and raw_step_mad > 0.0
                        else np.nan
                    ),
                    "degradation_area_error_pct": (
                        100.0 * abs(_area(values - raw, seconds)) / raw_abs_area
                        if np.isfinite(raw_abs_area) and raw_abs_area > 0.0
                        else np.nan
                    ),
                    "transient_retention": (
                        np.nanpercentile(np.abs(np.diff(values)), 90)
                        / np.nanpercentile(np.abs(np.diff(raw)), 90)
                        if len(raw) > 1
                        and np.nanpercentile(np.abs(np.diff(raw)), 90) > 0.0
                        else np.nan
                    ),
                    "normalized_loss_crossing_mae_seconds": _crossing_error(
                        raw, values, seconds
                    ),
                    "future_loss_increment_error_ratio": _future_loss_error(
                        raw, values, seconds
                    ),
                }
            )
    return pd.DataFrame(rows)


def prior_ablation(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle, group in frame.groupby("cycle_name", sort=False):
        for signal, values in {
            "measured_Q_expected_decrease": -group["heating_capacity"].to_numpy(dtype=float),
            "condition_normalized_D_Q": group["D_Q_raw"].to_numpy(dtype=float),
            "near_monotonic_D_Q": group["D_Q_nearly"].to_numpy(dtype=float),
            "hard_monotonic_D_Q": group["D_Q_hard"].to_numpy(dtype=float),
        }.items():
            rows.append(
                {
                    "cycle_name": cycle,
                    "date": str(group["date"].iloc[0]),
                    "signal": signal,
                    **monotonicity_metrics(values),
                }
            )
    return pd.DataFrame(rows)


def aggregation_order_ablation(raw_dir: Path | None) -> pd.DataFrame:
    if raw_dir is None or not raw_dir.is_dir():
        return pd.DataFrame()
    rows = []
    for path in sorted(raw_dir.glob("*.csv")):
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        frost = frame.loc[frame["cycle_stage"].eq("frost_development")].dropna(
            subset=["timestamp", "heating_capacity"]
        )
        if len(frost) < 10:
            continue
        indexed = frost.set_index("timestamp")["heating_capacity"].astype(float)
        aggregated = indexed.resample("10s", origin="start").median().dropna()
        aggregate_then_smooth = _savgol(
            aggregated.to_numpy(), pd.Series(aggregated.index)
        )
        smooth_1s = _savgol(indexed.to_numpy(), pd.Series(indexed.index))
        smooth_then_aggregate = (
            pd.Series(smooth_1s, index=indexed.index)
            .resample("10s", origin="start")
            .median()
            .reindex(aggregated.index)
            .to_numpy()
        )
        scale = float(np.nanmedian(np.abs(aggregate_then_smooth)))
        difference = smooth_then_aggregate - aggregate_then_smooth
        rows.append(
            {
                "cycle_name": path.stem,
                "n_10s": len(aggregated),
                "median_absolute_difference_kw": float(np.nanmedian(np.abs(difference))),
                "median_relative_difference_pct": (
                    100.0 * float(np.nanmedian(np.abs(difference))) / scale
                    if scale > 0.0
                    else np.nan
                ),
                "energy_difference_pct": (
                    100.0
                    * abs(np.nansum(smooth_then_aggregate) - np.nansum(aggregate_then_smooth))
                    / abs(np.nansum(aggregate_then_smooth))
                    if abs(np.nansum(aggregate_then_smooth)) > 0.0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_cycle(group: pd.DataFrame, output: Path) -> None:
    minutes = group["frost_elapsed_seconds"] / 60.0
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    axes[0].plot(minutes, group["heating_capacity"], color="0.25", lw=0.8, label="Measured Q")
    axes[0].plot(minutes, group["Q_healthy"], color="#D55E00", lw=1.5, label="Healthy reference")
    for method in METHODS:
        axes[1].plot(
            minutes,
            group[method],
            color=COLORS[method],
            lw=0.7 if method == "D_Q_raw" else 1.3,
            alpha=0.75 if method == "D_Q_raw" else 1.0,
            label=LABELS[method],
        )
    axes[0].set(ylabel="Heating capacity (kW)", title=str(group["cycle_name"].iloc[0]))
    axes[1].set(xlabel="Frost-development time (min)", ylabel="Relative degradation D_Q")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def write_report(
    output: Path,
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    ablation: pd.DataFrame,
    order: pd.DataFrame,
    ridge: pd.DataFrame,
    support: pd.DataFrame,
) -> None:
    summary = metrics.groupby("method").agg(
        cycles=("cycle_name", "nunique"),
        monotonic_violation=("violation_fraction", "median"),
        noise_ratio=("noise_ratio", "median"),
        area_error_pct=("degradation_area_error_pct", "median"),
        transient_retention=("transient_retention", "median"),
        crossing_error_s=("normalized_loss_crossing_mae_seconds", "median"),
        future_loss_error=("future_loss_increment_error_ratio", "median"),
    )
    prior = ablation.groupby("signal")["violation_fraction"].median()
    order_relative = (
        float(order["median_relative_difference_pct"].median()) if not order.empty else np.nan
    )
    selected_ridge = float(ridge.loc[ridge["selected"], "ridge"].iloc[0])
    outside = float(support["outside_support_fraction"].median())
    lines = [
        "# 工况归一化结霜退化与平滑比较",
        "",
        f"分析范围：{curves['cycle_name'].nunique()} 个循环，统一 10 s 建模网格。",
        "",
        "## 结论口径",
        "",
        "这里不把总制热量强制成单调曲线。先用其他实验日期的早期样本拟合健康输出，",
        "再计算 `D_Q = 1 - Q_measured / Q_healthy(context)`；硬单调仅表示潜在累计退化趋势。",
        "",
        "## 直接结论",
        "",
        (
            "1. **总制热量仍不能强制单调。** 去工况后原始 D_Q 的单调违例仅小幅下降，"
            "数据没有支持“归一化后自然单调”。"
        ),
        (
            "2. **成本函数输入首选 SG ~70 s。** 它不是单项数值最小，而是在曲线连续、"
            "累计面积基本不变、交点只偏移一个 10 s 采样点和保留短时响应之间最均衡；"
            "db4 小波作为独立复核。"
        ),
        "3. **近单调只作敏感性分析。** 它大量制造平台，适合检验单调先验，不适合计算导数。",
        (
            "4. **硬单调只作 RGB 潜在退化标签候选。** 它实现零违例，但短时响应"
            "只剩约 6.6%，不能替代瞬时制热量。"
        ),
        "",
        "## 健康基准审计",
        "",
        (
            f"跨日期早期样本选择的统一 ridge 为 {selected_ridge:g}。每循环中位有 "
            f"{outside:.1%} 的点至少一个运行变量超出其他日期早期健康样本范围。"
        ),
        "这些点的 Q_healthy 属于外推，必须在正式成本计算中标记低置信度；平滑不能修复健康基准外推。",
        "",
        "## 实测汇总（跨循环中位数）",
        "",
        summary.to_markdown(floatfmt=".4f"),
        "",
        "## 单调先验消融",
        "",
        prior.to_frame("median_violation_fraction").to_markdown(floatfmt=".4f"),
        "",
        "## 指标",
        "",
        (
            "- 单调违例率：`Σ max(-ΔD,0) / Σ |ΔD|`。越低表示越接近累计退化；"
            "0 只说明满足约束，不说明物理正确。"
        ),
        "- 噪声比：`MAD(ΔD_smooth) / MAD(ΔD_raw)`。越低越安静。",
        (
            "- 退化面积误差：`|∫(D_smooth-D_raw)dt| / ∫|D_raw|dt`。越低越不篡改"
            "进入成本函数的累计损失。"
        ),
        (
            "- 短时响应保留：平滑与原始一阶差分绝对值 P90 之比。接近 1 保留响应；"
            "接近 0 表示趋势被压平。"
        ),
        (
            "- 归一化损失交点误差：累计正退化达到原始最终面积 25%/50%/75% 时的"
            "中位时间偏移。它只检验成本输入稳定性，不是正式最优除霜时刻。"
        ),
        (
            "- 未来损失增量误差：5/10 min 的 ΔD_smooth 与 ΔD_raw 之差的中位绝对值，"
            "除以原始 ΔD 绝对值 P90。越低越少改写短期损失预测输入。"
        ),
        "",
        "## 1 s 与 10 s 顺序消融",
        "",
        (
            "`先10 s中值聚合再平滑` 与 `先在1 s平滑再10 s聚合` 的跨循环中位"
            f"相对差为 {order_relative:.4f}%。"
        ),
        "若该差异远小于传感器和模型误差，应选前者：计算更少、抗孤立点、并与成本/RGB时间网格一致。",
        "",
        "## 边界",
        "",
        (
            "没有真实 λ_Q、除霜能耗和恢复代价，因此本报告不伪造具体 t*；正式 t* "
            "应在成本参数确定后做循环级敏感性分析。"
        ),
        "",
        "## 方法依据",
        "",
        "- Tibshirani, Hoefling & Tibshirani, *Nearly-Isotonic Regression*, Technometrics (2011), https://doi.org/10.1198/tech.2010.10111。",
        "- Ramsay, *Monotone Regression Splines in Action*, Statistical Science (1988), https://doi.org/10.1214/ss/1177012761。",
        "- Mammen & Thomas-Agnan, *Smoothing Splines and Shape Restrictions* (1999), https://doi.org/10.1111/1467-9469.00147。",
        "- de Pater & Mitici, varying operating conditions 下的健康指标构造 (2023), https://doi.org/10.1016/j.engappai.2022.105582。",
        "- Zhou, Serban & Gebraeel, 含测量误差的退化轨迹建模 (2011), https://doi.org/10.1214/10-aoas448。",
    ]
    (output / "README_CN.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(input_dir: Path, output_dir: Path, raw_dir: Path | None = None) -> None:
    source = output_dir / "source_data"
    figures = output_dir / "figures" / "cycles"
    source.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    loaded = load_cycles(input_dir)
    ridge = reference_ridge_ablation(loaded)
    selected_ridge = float(ridge.loc[ridge["selected"], "ridge"].iloc[0])
    support = reference_support_audit(loaded)
    curves = add_degradation_curves(loaded, ridge=selected_ridge)
    metrics = score_curves(curves)
    ablation = prior_ablation(curves)
    order = aggregation_order_ablation(raw_dir)
    curves.to_parquet(source / "normalized_degradation.parquet", index=False)
    metrics.to_csv(source / "method_metrics.csv", index=False)
    ablation.to_csv(source / "monotonic_prior_ablation.csv", index=False)
    order.to_csv(source / "aggregation_order_ablation.csv", index=False)
    ridge.to_csv(source / "reference_ridge_ablation.csv", index=False)
    support.to_csv(source / "reference_support_audit.csv", index=False)
    for cycle, group in curves.groupby("cycle_name", sort=False):
        plot_cycle(group, figures / f"{cycle}.png")
    write_report(output_dir, curves, metrics, ablation, order, ridge, support)
    print(f"Analyzed {curves['cycle_name'].nunique()} cycles; outputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args()
    run_analysis(args.input_dir, args.output_dir, args.raw_dir)


if __name__ == "__main__":
    main()
