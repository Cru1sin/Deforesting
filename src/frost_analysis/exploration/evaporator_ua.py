"""Image-aligned evaporator apparent-UA labels."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROLLING_COLUMNS = [
    "ambient_temperature",
    "evaporating_temperature",
    "coil_temperature",
    "evaporator_capacity",
    "compressor_power",
    "power_total",
    "cop",
]


def image_evaporator_ua(
    frame: pd.DataFrame,
    images: pd.DataFrame,
    record: Mapping[str, Any],
    *,
    window_seconds: int = 60,
    match_tolerance_seconds: int = 15,
    min_driving_temperature_k: float = 2.0,
) -> pd.DataFrame:
    """Attach a causal inlet-reference evaporator UA label to every image."""
    if images.empty:
        return _empty_image_result(images)
    required = {"timestamp", "cycle_stage", "defrost_active", *_ROLLING_COLUMNS}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"cycle frame is missing evaporator-UA columns: {missing}")

    sensor = frame[list(required)].copy()
    sensor["timestamp"] = pd.to_datetime(sensor["timestamp"], errors="coerce")
    sensor = sensor.dropna(subset=["timestamp"]).sort_values("timestamp")
    indexed = sensor.set_index("timestamp")
    rolling = indexed[_ROLLING_COLUMNS].rolling(
        f"{window_seconds}s", min_periods=3
    ).median()
    rolling["defrost_active"] = (
        indexed["defrost_active"]
        .astype(bool)
        .rolling(f"{window_seconds}s", min_periods=1)
        .max()
    )
    rolling["sensor_stage"] = indexed["cycle_stage"].astype("string")
    rolling = rolling.reset_index().rename(columns={"timestamp": "sensor_time"})

    rolling["driving_temperature_te_k"] = (
        rolling["ambient_temperature"] - rolling["evaporating_temperature"]
    )
    primary_valid = (
        rolling[
            [
                "ambient_temperature",
                "evaporating_temperature",
                "evaporator_capacity",
            ]
        ]
        .notna()
        .all(axis=1)
        & rolling["evaporator_capacity"].gt(0)
        & rolling["driving_temperature_te_k"].gt(min_driving_temperature_k)
    )
    rolling["ua_evaporator_kw_per_k"] = np.where(
        primary_valid,
        rolling["evaporator_capacity"] / rolling["driving_temperature_te_k"],
        np.nan,
    )

    rolling["driving_temperature_t3_k"] = (
        rolling["ambient_temperature"] - rolling["coil_temperature"]
    )
    t3_valid = (
        rolling[["ambient_temperature", "coil_temperature", "evaporator_capacity"]]
        .notna()
        .all(axis=1)
        & rolling["evaporator_capacity"].gt(0)
        & rolling["driving_temperature_t3_k"].gt(min_driving_temperature_k)
    )
    rolling["ua_t3_diagnostic_kw_per_k"] = np.where(
        t3_valid,
        rolling["evaporator_capacity"] / rolling["driving_temperature_t3_k"],
        np.nan,
    )

    baseline_start = _boundary(record, "baseline_start")
    baseline_end = _boundary(record, "baseline_end")
    baseline_mask = rolling["sensor_time"].ge(baseline_start) & rolling["sensor_time"].lt(
        baseline_end
    )
    baseline_te = _median(rolling.loc[baseline_mask, "ua_evaporator_kw_per_k"])
    baseline_t3 = _median(rolling.loc[baseline_mask, "ua_t3_diagnostic_kw_per_k"])
    rolling["ua_baseline_kw_per_k"] = baseline_te
    rolling["ua_t3_baseline_kw_per_k"] = baseline_t3
    rolling["ua_over_baseline"] = rolling["ua_evaporator_kw_per_k"] / baseline_te

    reconstructed_heat = (
        rolling["ua_evaporator_kw_per_k"] * rolling["driving_temperature_te_k"]
        + rolling["compressor_power"]
    )
    rolling["cop_from_ua"] = reconstructed_heat.div(rolling["power_total"]).where(
        rolling["power_total"].gt(0)
    )
    rolling["cop_from_ua_error"] = rolling["cop_from_ua"] - rolling["cop"]

    result = images.copy()
    result["image_time"] = pd.to_datetime(result["image_time"], errors="coerce")
    result = pd.merge_asof(
        result.sort_values("image_time"),
        rolling.sort_values("sensor_time"),
        left_on="image_time",
        right_on="sensor_time",
        direction="backward",
        tolerance=pd.Timedelta(seconds=match_tolerance_seconds),
    )
    result["cycle_status"] = str(record.get("status", "invalid"))
    result["ua_temperature_basis"] = "evaporating_temperature"
    result["quality_status"] = "available"
    result.loc[result["sensor_time"].isna(), "quality_status"] = "sensor_time_unmatched"
    result.loc[
        result["sensor_stage"].astype(str).ne("frost_development"), "quality_status"
    ] = "not_frost_development"
    result.loc[result["defrost_active"].fillna(False).astype(bool), "quality_status"] = (
        "defrost_active"
    )
    result.loc[
        result["evaporator_capacity"].le(0), "quality_status"
    ] = "nonpositive_evaporator_capacity"
    result.loc[
        result["driving_temperature_te_k"].le(min_driving_temperature_k),
        "quality_status",
    ] = "small_driving_temperature"
    result.loc[result["ua_evaporator_kw_per_k"].isna(), "quality_status"] = (
        "sensor_value_unavailable"
    )
    label_columns = [
        "ua_evaporator_kw_per_k",
        "ua_over_baseline",
        "ua_t3_diagnostic_kw_per_k",
        "cop_from_ua",
        "cop_from_ua_error",
    ]
    result.loc[result["quality_status"].ne("available"), label_columns] = np.nan
    return result.sort_values(["image_time", "camera_role", "file_name"]).reset_index(drop=True)


def summarize_evaporator_ua(
    images: pd.DataFrame, records: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    """Summarize absolute UA and normalized continuity once per image time."""
    rows: list[dict[str, object]] = []
    for record in records:
        cycle_name = str(record["cycle_name"])
        scoped = images.loc[images.get("cycle_name", pd.Series(dtype="string")).eq(cycle_name)]
        if scoped.empty:
            rows.append(_empty_cycle_summary(cycle_name, str(record.get("status", "invalid"))))
            continue
        times = (
            scoped.groupby("image_time", as_index=False)
            .agg(
                quality_status=("quality_status", "first"),
                ua=("ua_evaporator_kw_per_k", _median),
                ua_t3=("ua_t3_diagnostic_kw_per_k", _median),
                baseline=("ua_baseline_kw_per_k", _median),
                baseline_t3=("ua_t3_baseline_kw_per_k", _median),
            )
            .sort_values("image_time")
        )
        if "cop_from_ua_error" in scoped and scoped["cop_from_ua_error"].notna().any():
            cop_error = (
                scoped.groupby("image_time", as_index=False)["cop_from_ua_error"]
                .median()
                .rename(columns={"cop_from_ua_error": "cop_error"})
            )
            times = times.merge(cop_error, on="image_time", how="left")
        else:
            times["cop_error"] = np.nan
        valid = times.loc[times["quality_status"].eq("available") & times["ua"].notna()]
        row = _empty_cycle_summary(cycle_name, str(record.get("status", "invalid")))
        row.update(
            {
                "image_count": int(len(scoped)),
                "unique_image_times": int(len(times)),
                "available_unique_times": int(len(valid)),
                "valid_fraction": float(len(valid) / len(times)),
            }
        )
        if valid.empty:
            row["continuity_status"] = "no_available_values"
            rows.append(row)
            continue

        values = valid["ua"].to_numpy(dtype=float)
        baseline = float(valid["baseline"].median())
        scale = baseline if np.isfinite(baseline) and baseline > 0 else float(np.median(values))
        steps_fraction = np.abs(np.diff(values)) / scale
        t3_values = valid["ua_t3"].dropna().to_numpy(dtype=float)
        t3_baseline = float(valid["baseline_t3"].median())
        t3_steps_fraction = (
            np.abs(np.diff(t3_values)) / t3_baseline
            if len(t3_values) >= 2 and np.isfinite(t3_baseline) and t3_baseline > 0
            else np.array([], dtype=float)
        )
        valid_times = pd.to_datetime(valid["image_time"])
        elapsed_hours = (
            valid_times - valid_times.iloc[0]
        ).dt.total_seconds().to_numpy(dtype=float) / 3600.0
        time_gaps = valid_times.diff().dt.total_seconds().to_numpy(dtype=float)[1:] / 60.0
        median_step = _quantile(steps_fraction, 0.5)
        p95_step = _quantile(steps_fraction, 0.95)
        t3_p95_step = _quantile(t3_steps_fraction, 0.95)
        row.update(
            {
                "ua_baseline_kw_per_k": baseline,
                "ua_median_kw_per_k": float(np.median(values)),
                "ua_q05_kw_per_k": float(np.quantile(values, 0.05)),
                "ua_q95_kw_per_k": float(np.quantile(values, 0.95)),
                "ua_min_kw_per_k": float(np.min(values)),
                "ua_max_kw_per_k": float(np.max(values)),
                "ua_cv": float(np.std(values) / abs(np.mean(values))),
                "ua_slope_kw_per_k_per_hour": (
                    float(np.polyfit(elapsed_hours, values, 1)[0])
                    if len(values) >= 2 and elapsed_hours[-1] > 0
                    else np.nan
                ),
                "median_abs_step_fraction": median_step,
                "p95_abs_step_fraction": p95_step,
                "max_abs_step_fraction": (
                    float(np.max(steps_fraction)) if len(steps_fraction) else np.nan
                ),
                "large_step_count": int(np.sum(steps_fraction > 0.15)),
                "max_time_gap_minutes": float(np.max(time_gaps)) if len(time_gaps) else np.nan,
                "t3_p95_abs_step_fraction": t3_p95_step,
                "preferred_temperature_basis": "Te",
                "smoother_temperature_basis": (
                    "unavailable"
                    if not np.isfinite(t3_p95_step)
                    else "Te"
                    if p95_step <= t3_p95_step
                    else "T3"
                ),
                "cop_from_ua_mae": _median(valid["cop_error"].abs()),
                "continuity_status": (
                    "insufficient_valid_points"
                    if len(values) < 10
                    else "stable"
                    if median_step <= 0.03 and p95_step <= 0.15
                    else "unstable"
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_evaporator_ua(
    loader: Any,
    cycle_names: Sequence[str],
    output_dir: Path,
) -> Path:
    """Load Dataset cycles and write image-aligned evaporator-UA artifacts."""
    records = [loader.get_cycle_record(name) for name in cycle_names]
    results: list[pd.DataFrame] = []
    columns = ["timestamp", "cycle_stage", "defrost_active", *_ROLLING_COLUMNS]
    for record in records:
        cycle_name = str(record["cycle_name"])
        frame = loader.load_cycle(cycle_name, columns=columns)
        images = loader.load_image_metadata(cycle_name)
        result = image_evaporator_ua(frame, images, record)
        if not result.empty:
            results.append(result)
    image_table = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    summary = summarize_evaporator_ua(image_table, records)
    return write_evaporator_ua_outputs(image_table, summary, output_dir)


def write_evaporator_ua_outputs(
    images: pd.DataFrame, summary: pd.DataFrame, output_dir: Path
) -> Path:
    """Write source tables, a Nature-style figure bundle, and findings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images.to_csv(output_dir / "image_evaporator_ua.csv", index=False)
    summary.to_csv(output_dir / "cycle_summary.csv", index=False)
    payload = {
        "label": "inlet-reference apparent evaporator UA",
        "formula": "UA = evaporator_capacity / (ambient_temperature - evaporating_temperature)",
        "unit": "kW/K",
        "temperature_basis": {
            "primary": "evaporating_temperature (Te)",
            "diagnostic_only": "coil_temperature (T3)",
        },
        "limitation": (
            "air outlet temperature and air mass flow are unavailable; "
            "this is not LMTD/NTU UA"
        ),
        "cop_bridge": "COP = (UA*(T4-Te) + compressor_power) / power_total",
        "quality": {
            "trailing_window_seconds": 60,
            "sensor_match_tolerance_seconds": 15,
            "minimum_driving_temperature_k": 2.0,
            "stable_median_abs_step_fraction_max": 0.03,
            "stable_p95_abs_step_fraction_max": 0.15,
        },
        "counts": {
            "cycles": int(len(summary)),
            "images": int(len(images)),
            "available_images": int(
                images.get("quality_status", pd.Series(dtype="string"))
                .eq("available")
                .sum()
            ),
            "stable_cycles": int(summary["continuity_status"].eq("stable").sum()),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_figure(images, summary, output_dir / "ua_timeseries")
    _write_findings(summary, output_dir / "findings.md")
    return output_dir


def _write_figure(images: pd.DataFrame, summary: pd.DataFrame, base: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
    with mpl.rc_context(style):
        figure, (axis, diagnostic) = plt.subplots(
            1,
            2,
            figsize=(183 / 25.4, 82 / 25.4),
            gridspec_kw={"width_ratios": [2.2, 1]},
        )
        available_mask = images["quality_status"].eq("available")
        if "cycle_status" in images:
            available_mask &= images["cycle_status"].eq("valid")
        available = images.loc[available_mask].copy()
        if not available.empty:
            reduced = (
                available.groupby(["cycle_name", "image_time"], as_index=False)[
                    "ua_evaporator_kw_per_k"
                ]
                .median()
                .sort_values("image_time")
            )
            colors = plt.get_cmap("Blues")(
                np.linspace(0.38, 0.9, reduced["cycle_name"].nunique())
            )
            for color, (cycle_name, group) in zip(
                colors, reduced.groupby("cycle_name", sort=True), strict=False
            ):
                elapsed = (
                    pd.to_datetime(group["image_time"])
                    - pd.to_datetime(group["image_time"]).iloc[0]
                ).dt.total_seconds() / 60.0
                axis.plot(
                    elapsed,
                    group["ua_evaporator_kw_per_k"],
                    color=color,
                    linewidth=0.9,
                    label=str(cycle_name).removeprefix("frost_cycle_"),
                )
        axis.set(
            xlabel="Time from first valid image (min)",
            ylabel=r"Apparent evaporator $UA$ (kW K$^{-1}$)",
        )
        if axis.lines:
            axis.legend(title="Cycle", ncol=2, fontsize=5.8, title_fontsize=6.2)

        compared = summary.loc[
            summary["cycle_status"].eq("valid")
            & summary["p95_abs_step_fraction"].notna()
            & summary["t3_p95_abs_step_fraction"].notna()
        ]
        for _, row in compared.iterrows():
            values = [row["p95_abs_step_fraction"], row["t3_p95_abs_step_fraction"]]
            diagnostic.plot([0, 1], values, color="#B8BEC7", linewidth=0.7, zorder=1)
            diagnostic.scatter(
                [0, 1], values, color=["#2C6E9B", "#9A7B64"], s=12, zorder=2
            )
        diagnostic.set(
            xticks=[0, 1],
            xticklabels=[r"$T_e$", r"$T_3$"],
            ylabel="P95 relative step",
        )
        diagnostic.set_title("Temperature-basis sensitivity", fontsize=7)
        axis.text(-0.13, 1.02, "a", transform=axis.transAxes, fontweight="bold", fontsize=8)
        diagnostic.text(
            -0.32,
            1.02,
            "b",
            transform=diagnostic.transAxes,
            fontweight="bold",
            fontsize=8,
        )
        figure.tight_layout(w_pad=2.0)
        figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
        figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        figure.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
        plt.close(figure)


def _write_findings(summary: pd.DataFrame, path: Path) -> None:
    columns = [
        "cycle_name",
        "cycle_status",
        "available_unique_times",
        "ua_baseline_kw_per_k",
        "ua_median_kw_per_k",
        "ua_q05_kw_per_k",
        "ua_q95_kw_per_k",
        "p95_abs_step_fraction",
        "t3_p95_abs_step_fraction",
        "cop_from_ua_mae",
        "max_time_gap_minutes",
        "continuity_status",
    ]
    evaluated = summary.loc[summary["available_unique_times"].gt(0)]
    valid = evaluated.loc[evaluated["cycle_status"].eq("valid")]
    stable = valid.loc[valid["continuity_status"].eq("stable")]
    comparable = valid.loc[valid["t3_p95_abs_step_fraction"].notna()]
    te_smoother = comparable.loc[comparable["smoother_temperature_basis"].eq("Te")]
    te_step = _median(comparable["p95_abs_step_fraction"])
    t3_step = _median(comparable["t3_p95_abs_step_fraction"])
    unavailable = summary.loc[summary["available_unique_times"].eq(0), "cycle_name"]

    def names(values: pd.Series) -> str:
        return ", ".join(values.astype(str).str.removeprefix("frost_cycle_")) or "无"

    path.write_text(
        "# 图片对齐的蒸发器表观 UA\n\n"
        "## 主标签\n\n"
        "$$\nQ_{evap}=Q_{heat}-W_{comp}\n$$\n\n"
        "$$\nUA_{app,in}=\\frac{Q_{evap}}{T_4-T_e}\n$$\n\n"
        "其中，$T_4$ 是入口环境空气温度，$T_e$ 是制冷剂蒸发温度。"
        "输出单位为 kW/K；绝对 UA 是 RGB 回归主标签，`ua_over_baseline` 只用于质量检查。\n\n"
        "## 为什么主公式用 Te 而不是 T3\n\n"
        "分子是整台蒸发器的换热量，因此分母应使用代表制冷剂侧整体状态的蒸发温度。"
        "T3 是局部盘管测点，把整机热量除以局部温差存在空间不匹配；"
        "代码仅保留其结果作为敏感性诊断。\n\n"
        f"在 {len(comparable)} 个可同时比较的 valid cycles 中，Te 版本在 "
        f"{len(te_smoother)}/{len(comparable)} 个 cycle 上具有不高于 T3 版本的 P95 相对步长；"
        f"两者跨 cycle 中位数分别为 {te_step:.4f} 和 {t3_step:.4f}。"
        "因此平滑性没有给 T3 明确优势，主选择依据仍是温度定义与整机热量的空间一致性。"
        f"按预注册连续性阈值，{len(stable)}/{len(valid)} 个 cycle 整体稳定。"
        f"无可用图片标签的 cycles：{names(unavailable)}。\n\n"
        "## 不能声称为严格 LMTD-UA\n\n"
        "严格换热器 UA 应使用对数平均温差：\n\n"
        "$$\nUA=\\frac{Q_{evap}}{\\Delta T_{lm}}\n$$\n\n"
        "$$\n\\Delta T_{lm}=\\frac{(T_{air,in}-T_e)-(T_{air,out}-T_e)}"
        "{\\ln\\left[\\frac{T_{air,in}-T_e}{T_{air,out}-T_e}\\right]}\n$$\n\n"
        "当前数据没有蒸发器出口空气温度，也没有可校准的空气质量流量，"
        "因此不能从现有点位辨识严格 LMTD/NTU-UA。"
        "本标签应写作 inlet-reference apparent UA。\n\n"
        "## 从 UA 连接到 COP\n\n"
        "$$\n\\widehat{Q}_{evap}=\\widehat{UA}_{app,in}(T_4-T_e)\n$$\n\n"
        "$$\n\\widehat{COP}=\\frac{\\widehat{Q}_{evap}+W_{comp}}{W_{total}}\n$$\n\n"
        "部署时仍必须提供或预测 $T_4$、$T_e$、压缩机功率和总功率；"
        "UA 不能单独唯一决定 COP。当前 `cop_from_ua_mae` 只是代数闭合检查，"
        "因为 $Q_{evap}$ 本身由 $Q_{heat}-W_{comp}$ 得到，不能当作独立预测验证。\n\n"
        "## Cycle 汇总\n\n"
        + summary[columns].to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


def _boundary(record: Mapping[str, Any], name: str) -> pd.Timestamp:
    value = record.get(name)
    boundaries = record.get("boundaries")
    if value is None and isinstance(boundaries, Mapping):
        value = boundaries.get(name)
    return pd.to_datetime(value, errors="coerce")


def _median(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.median()) if not clean.empty else np.nan


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if len(values) else np.nan


def _empty_image_result(images: pd.DataFrame) -> pd.DataFrame:
    result = images.copy()
    for column in (
        "sensor_time",
        "ambient_temperature",
        "evaporating_temperature",
        "coil_temperature",
        "evaporator_capacity",
        "driving_temperature_te_k",
        "driving_temperature_t3_k",
        "ua_evaporator_kw_per_k",
        "ua_t3_diagnostic_kw_per_k",
        "ua_baseline_kw_per_k",
        "ua_t3_baseline_kw_per_k",
        "ua_over_baseline",
        "cop_from_ua",
        "cop_from_ua_error",
        "cycle_status",
        "quality_status",
    ):
        result[column] = pd.Series(dtype="object")
    return result


def _empty_cycle_summary(cycle_name: str, cycle_status: str) -> dict[str, object]:
    return {
        "cycle_name": cycle_name,
        "cycle_status": cycle_status,
        "image_count": 0,
        "unique_image_times": 0,
        "available_unique_times": 0,
        "valid_fraction": 0.0,
        "ua_baseline_kw_per_k": np.nan,
        "ua_median_kw_per_k": np.nan,
        "ua_q05_kw_per_k": np.nan,
        "ua_q95_kw_per_k": np.nan,
        "ua_min_kw_per_k": np.nan,
        "ua_max_kw_per_k": np.nan,
        "ua_cv": np.nan,
        "ua_slope_kw_per_k_per_hour": np.nan,
        "median_abs_step_fraction": np.nan,
        "p95_abs_step_fraction": np.nan,
        "max_abs_step_fraction": np.nan,
        "large_step_count": 0,
        "max_time_gap_minutes": np.nan,
        "t3_p95_abs_step_fraction": np.nan,
        "preferred_temperature_basis": "Te",
        "smoother_temperature_basis": "unavailable",
        "cop_from_ua_mae": np.nan,
        "continuity_status": "no_images",
    }
