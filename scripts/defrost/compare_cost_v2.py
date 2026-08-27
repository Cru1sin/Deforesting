#!/usr/bin/env python3
"""Compare an experiment-held-out full-cycle cost proxy with existing defrost points."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

ENERGY_FEATURES = ("ambient_temperature", "evaporating_temperature")
SIGNED_HEAT_FEATURES = ("coil_temperature", "minutes_from_stable")


def _fit_ridge(
    train: pd.DataFrame,
    features: tuple[str, ...],
    outcome: str,
    *,
    alpha: float = 1.0,
) -> dict[str, np.ndarray | float]:
    raw = train.loc[:, features].apply(pd.to_numeric, errors="coerce")
    medians = raw.median().to_numpy(dtype=float)
    filled = raw.fillna(pd.Series(medians, index=features))
    centers = filled.mean().to_numpy(dtype=float)
    scales = filled.std(ddof=0).replace(0.0, 1.0).to_numpy(dtype=float)
    design = np.column_stack(
        [np.ones(len(filled)), (filled.to_numpy(dtype=float) - centers) / scales]
    )
    target = pd.to_numeric(train[outcome], errors="coerce").to_numpy(dtype=float)
    penalty = np.diag([0.0, *(alpha for _ in features)])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return {
        "medians": medians,
        "centers": centers,
        "scales": scales,
        "coefficients": coefficients,
        "minimums": raw.min().to_numpy(dtype=float),
        "maximums": raw.max().to_numpy(dtype=float),
        "outcome_minimum": float(np.nanmin(target)),
        "outcome_maximum": float(np.nanmax(target)),
    }


def _predict_ridge(
    model: dict[str, np.ndarray | float],
    values: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    raw = values.loc[:, features].apply(pd.to_numeric, errors="coerce")
    raw_array = raw.to_numpy(dtype=float)
    supported = np.isfinite(raw_array).all(axis=1)
    supported &= (raw_array >= np.asarray(model["minimums"])).all(axis=1)
    supported &= (raw_array <= np.asarray(model["maximums"])).all(axis=1)
    filled = raw.fillna(pd.Series(np.asarray(model["medians"]), index=features))
    standardized = (
        filled.to_numpy(dtype=float) - np.asarray(model["centers"])
    ) / np.asarray(model["scales"])
    design = np.column_stack([np.ones(len(filled)), standardized])
    predicted = design @ np.asarray(model["coefficients"])
    predicted = np.clip(
        predicted,
        float(model["outcome_minimum"]),
        float(model["outcome_maximum"]),
    )
    return predicted, supported


def build_v2_curves(events: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """Predict a held-out event ticket and calculate the V2 full-cycle proxy."""
    predictions = []
    for experiment, candidates in curves.groupby("experiment_id", sort=True):
        train = events.loc[~events["experiment_id"].eq(experiment)]
        if train.empty:
            raise ValueError(f"no held-out training events for experiment {experiment}")
        energy_model = _fit_ridge(train, ENERGY_FEATURES, "electricity_kwh")
        heat_model = _fit_ridge(
            train,
            SIGNED_HEAT_FEATURES,
            "measured_signed_transient_user_heat_kwh",
        )
        candidates = candidates.copy()
        energy, energy_supported = _predict_ridge(
            energy_model, candidates, ENERGY_FEATURES
        )
        signed_heat, heat_supported = _predict_ridge(
            heat_model, candidates, SIGNED_HEAT_FEATURES
        )
        candidates["predicted_ticket_electricity_kwh"] = energy
        candidates["predicted_ticket_signed_heat_kwh"] = signed_heat
        candidates["v2_energy_model_supported"] = energy_supported
        candidates["v2_signed_heat_model_supported"] = heat_supported
        predictions.append(candidates)

    result = pd.concat(predictions, ignore_index=True)
    result["v2_total_electricity_kwh"] = (
        result["heating_electricity_kwh"]
        + result["predicted_ticket_electricity_kwh"]
    )
    result["v2_total_water_heat_kwh"] = (
        result["water_heating_kwh"] + result["predicted_ticket_signed_heat_kwh"]
    )
    result["v2_model_supported"] = (
        result["v2_energy_model_supported"]
        & result["v2_signed_heat_model_supported"]
    )
    result["v2_eligible"] = (
        result["optimization_eligible"].fillna(False)
        & result["v2_model_supported"]
        & result["v2_total_electricity_kwh"].gt(0)
        & result["v2_total_water_heat_kwh"].gt(0)
    )
    result["inverse_cop_v2"] = (
        result["v2_total_electricity_kwh"] / result["v2_total_water_heat_kwh"]
    ).where(result["v2_eligible"])
    result["inverse_cop_v2_zero_transient_heat"] = (
        result["v2_total_electricity_kwh"] / result["water_heating_kwh"]
    ).where(result["v2_eligible"] & result["water_heating_kwh"].gt(0))
    result["inverse_cop_v2_unit_ablation"] = (
        result["v2_total_electricity_kwh"] / result["unit_heating_kwh"]
    ).where(result["v2_eligible"] & result["unit_heating_kwh"].gt(0))
    return result


def _earliest_minimum(values: pd.DataFrame, column: str) -> pd.Timestamp | pd.NaT:
    finite = values[column].notna()
    if not finite.any():
        return pd.NaT
    minimum = values.loc[finite, column].min()
    return pd.Timestamp(values.loc[finite & values[column].eq(minimum), "candidate_time"].iloc[0])


def _nearest_v2_regret(
    values: pd.DataFrame,
    target: pd.Timestamp | pd.NaT,
    *,
    tolerance_minutes: float = 0.51,
) -> tuple[pd.Timestamp | pd.NaT, float]:
    if pd.isna(target) or values.empty:
        return pd.NaT, float("nan")
    distance = (
        pd.to_datetime(values["candidate_time"]) - pd.Timestamp(target)
    ).abs().dt.total_seconds() / 60.0
    index = distance.idxmin()
    if float(distance.loc[index]) > tolerance_minutes:
        return pd.NaT, float("nan")
    minimum = float(values["inverse_cop_v2"].min())
    return (
        pd.Timestamp(values.loc[index, "candidate_time"]),
        float(values.loc[index, "inverse_cop_v2"] / minimum - 1.0),
    )


def compare_points(curves: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """Return one row per cycle for the V2, published unit, and rule points."""
    rows = []
    for cycle_name, values in curves.groupby("cycle_name", sort=True):
        values = values.sort_values("candidate_time", kind="stable").copy()
        values = values.loc[values["v2_eligible"].fillna(False)]
        t_star_v2 = _earliest_minimum(values, "inverse_cop_v2")
        if pd.isna(t_star_v2):
            minimum_location = "no_supported_candidate"
        else:
            position = int(
                np.flatnonzero(
                    pd.to_datetime(values["candidate_time"]).eq(t_star_v2).to_numpy()
                )[0]
            )
            minimum_location = (
                "left_boundary"
                if position == 0
                else "right_boundary"
                if position == len(values) - 1
                else "interior"
            )
        rows.append(
            {
                "cycle_name": cycle_name,
                "t_star_v2": t_star_v2,
                "v2_minimum_location": minimum_location,
                "v2_supported_start": pd.to_datetime(values["candidate_time"]).min(),
                "v2_supported_end": pd.to_datetime(values["candidate_time"]).max(),
                "v2_supported_candidate_count": len(values),
                "t_star_v2_zero_transient_heat": _earliest_minimum(
                    values, "inverse_cop_v2_zero_transient_heat"
                ),
                "t_star_v2_unit_ablation": _earliest_minimum(
                    values, "inverse_cop_v2_unit_ablation"
                ),
                "t_star_unit_common_support": _earliest_minimum(
                    values, "inverse_cop_unit"
                )
                if "inverse_cop_unit" in values
                else pd.NaT,
            }
        )
    comparison = pd.DataFrame(rows).merge(
        points[
            ["cycle_name", "t_star_unit", "t_RB", "rb_status", "trigger_type"]
        ],
        on="cycle_name",
        how="left",
        validate="one_to_one",
    )
    comparison = comparison.rename(columns={"t_star_unit": "t_star_unit_original"})
    comparison["t_rule"] = pd.to_datetime(comparison["t_RB"], errors="coerce").where(
        comparison["rb_status"].eq("triggered")
    )
    for column in (
        "t_star_v2",
        "t_star_v2_zero_transient_heat",
        "t_star_v2_unit_ablation",
        "t_star_unit_common_support",
        "t_star_unit_original",
        "t_rule",
    ):
        comparison[column] = pd.to_datetime(comparison[column], errors="coerce")
    comparison["unit_minus_v2_minutes"] = (
        comparison["t_star_unit_original"] - comparison["t_star_v2"]
    ).dt.total_seconds() / 60.0
    comparison["unit_common_minus_v2_minutes"] = (
        comparison["t_star_unit_common_support"] - comparison["t_star_v2"]
    ).dt.total_seconds() / 60.0
    comparison["rule_minus_v2_minutes"] = (
        comparison["t_rule"] - comparison["t_star_v2"]
    ).dt.total_seconds() / 60.0
    comparison["zero_transient_heat_minus_v2_minutes"] = (
        comparison["t_star_v2_zero_transient_heat"] - comparison["t_star_v2"]
    ).dt.total_seconds() / 60.0
    comparison["unit_ablation_minus_v2_minutes"] = (
        comparison["t_star_v2_unit_ablation"] - comparison["t_star_v2"]
    ).dt.total_seconds() / 60.0
    comparison["unit_ablation_minus_unit_original_minutes"] = (
        comparison["t_star_v2_unit_ablation"]
        - comparison["t_star_unit_original"]
    ).dt.total_seconds() / 60.0
    unit_candidates = []
    unit_regrets = []
    rule_candidates = []
    rule_regrets = []
    curve_groups = {
        cycle: values.loc[values["v2_eligible"].fillna(False)].copy()
        for cycle, values in curves.groupby("cycle_name", sort=False)
    }
    for row in comparison.itertuples(index=False):
        values = curve_groups.get(row.cycle_name, pd.DataFrame())
        unit_candidate, unit_regret = _nearest_v2_regret(
            values, row.t_star_unit_original
        )
        rule_candidate, rule_regret = _nearest_v2_regret(values, row.t_rule)
        unit_candidates.append(unit_candidate)
        unit_regrets.append(unit_regret)
        rule_candidates.append(rule_candidate)
        rule_regrets.append(rule_regret)
    comparison["unit_original_candidate_time"] = unit_candidates
    comparison["v2_regret_at_unit_original"] = unit_regrets
    comparison["rule_candidate_time"] = rule_candidates
    comparison["v2_regret_at_rule"] = rule_regrets
    return comparison.drop(columns="t_RB")


def ticket_model_predictions(events: pd.DataFrame) -> pd.DataFrame:
    """Return event-level held-out predictions for the two frozen V2 models."""
    rows = []
    for experiment, test in events.groupby("experiment_id", sort=True):
        train = events.loc[~events["experiment_id"].eq(experiment)]
        models = {
            "electricity": (
                _fit_ridge(train, ENERGY_FEATURES, "electricity_kwh"),
                ENERGY_FEATURES,
                "electricity_kwh",
            ),
            "signed_transient_heat": (
                _fit_ridge(
                    train,
                    SIGNED_HEAT_FEATURES,
                    "measured_signed_transient_user_heat_kwh",
                ),
                SIGNED_HEAT_FEATURES,
                "measured_signed_transient_user_heat_kwh",
            ),
        }
        for label, (model, features, outcome) in models.items():
            predicted, supported = _predict_ridge(model, test, features)
            actual = pd.to_numeric(test[outcome], errors="coerce").to_numpy(dtype=float)
            baseline = np.repeat(pd.to_numeric(train[outcome]).mean(), len(test))
            for index, event in enumerate(test.itertuples(index=False)):
                rows.extend(
                    [
                        {
                            "cycle_name": event.cycle_name,
                            "experiment_id": experiment,
                            "outcome": label,
                            "model": "held_out_mean",
                            "actual": actual[index],
                            "predicted": baseline[index],
                            "feature_supported": True,
                        },
                        {
                            "cycle_name": event.cycle_name,
                            "experiment_id": experiment,
                            "outcome": label,
                            "model": "ridge_frozen_features",
                            "actual": actual[index],
                            "predicted": predicted[index],
                            "feature_supported": supported[index],
                        },
                    ]
                )
    return pd.DataFrame(rows)


def ticket_model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize experiment-held-out event prediction errors."""
    rows = []
    for (outcome, model), values in predictions.groupby(["outcome", "model"]):
        error = values["predicted"] - values["actual"]
        experiment_mae = error.abs().groupby(values["experiment_id"]).mean()
        rows.append(
            {
                "outcome": outcome,
                "model": model,
                "event_count": len(values),
                "experiment_count": values["experiment_id"].nunique(),
                "mae": error.abs().mean(),
                "rmse": float(np.sqrt(error.pow(2).mean())),
                "bias": error.mean(),
                "macro_experiment_mae": experiment_mae.mean(),
                "feature_supported_fraction": values["feature_supported"].mean(),
            }
        )
    return pd.DataFrame(rows)


def comparison_summary(comparison: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """Return compact descriptive summaries without significance claims."""
    rows = []
    valid_v2 = comparison["t_star_v2"].notna()
    columns = (
        "unit_minus_v2_minutes",
        "unit_common_minus_v2_minutes",
        "rule_minus_v2_minutes",
        "zero_transient_heat_minus_v2_minutes",
        "unit_ablation_minus_v2_minutes",
        "unit_ablation_minus_unit_original_minutes",
        "v2_regret_at_unit_original",
        "v2_regret_at_rule",
    )
    for column in columns:
        values = pd.to_numeric(comparison[column], errors="coerce").dropna()
        rows.append(
            {
                "metric": column,
                "count": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "p10": values.quantile(0.1),
                "p90": values.quantile(0.9),
            }
        )
    rows.extend(
        [
            {
                "metric": "v2_cycle_count",
                "count": valid_v2.sum(),
            },
            {
                "metric": "v2_supported_candidate_fraction",
                "count": len(curves),
                "mean": curves["v2_model_supported"].mean(),
            },
            {
                "metric": "v2_interior_minimum_fraction",
                "count": valid_v2.sum(),
                "mean": comparison.loc[valid_v2, "v2_minimum_location"]
                .eq("interior")
                .mean(),
            },
        ]
    )
    return pd.DataFrame(rows)


def plot_comparison(comparison: pd.DataFrame, output: Path) -> None:
    """Plot unit and rule timing offsets relative to the V2 optimum."""
    values = comparison.dropna(subset=["t_star_v2"]).sort_values(
        "unit_minus_v2_minutes", kind="stable"
    )
    figure = Figure(figsize=(8.0, 4.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    x = np.arange(len(values))
    axis.axhline(0.0, color="black", linewidth=0.8, label="V2 optimum")
    axis.scatter(
        x,
        values["unit_minus_v2_minutes"],
        s=18,
        color="#2878B5",
        label="Original unit-heat optimum − V2",
    )
    axis.scatter(
        x,
        values["rule_minus_v2_minutes"],
        s=18,
        marker="x",
        color="#C82423",
        label="Rule point − V2",
    )
    axis.set(
        xlabel="Cycles ordered by original unit-heat offset",
        ylabel="Time difference from V2 optimum (min)",
    )
    axis.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)


def _minutes_after(timestamp: object, stable: pd.Timestamp) -> float:
    value = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(value):
        return float("nan")
    return float((pd.Timestamp(value) - stable).total_seconds() / 60.0)


def prepare_cycle_overlay(
    curves: pd.DataFrame,
    comparison: pd.Series,
    stable: pd.Timestamp,
) -> pd.DataFrame:
    """Prepare the established per-cycle time axis and both cost curves."""
    values = curves.copy()
    values["candidate_time"] = pd.to_datetime(
        values["candidate_time"], errors="coerce"
    )
    values = values.dropna(subset=["candidate_time"]).sort_values(
        "candidate_time", kind="stable"
    )
    values["minutes_from_stable"] = (
        values["candidate_time"] - stable
    ).dt.total_seconds() / 60.0
    original_eligible = values.get(
        "optimization_eligible", pd.Series(True, index=values.index)
    ).fillna(False)
    v2_eligible = values.get(
        "v2_eligible", pd.Series(False, index=values.index)
    ).fillna(False)
    values["inverse_cop_unit_plot"] = pd.to_numeric(
        values["inverse_cop_unit"], errors="coerce"
    ).where(original_eligible)
    values["inverse_cop_v2_plot"] = pd.to_numeric(
        values["inverse_cop_v2"], errors="coerce"
    ).where(v2_eligible)
    values.attrs.update(
        {
            "v2_optimum_minutes": _minutes_after(
                comparison.get("t_star_v2"), stable
            ),
            "unit_optimum_minutes": _minutes_after(
                comparison.get("t_star_unit_original"), stable
            ),
            "rule_minutes": _minutes_after(comparison.get("t_rule"), stable),
            "v2_minimum_location": str(
                comparison.get("v2_minimum_location", "")
            ),
        }
    )
    return values


def _cycle_overlay_figure(cycle_name: str, values: pd.DataFrame) -> Figure:
    figure = Figure(figsize=(7.2, 4.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot(
        values["minutes_from_stable"],
        values["inverse_cop_unit_plot"],
        color="#3775BA",
        linewidth=1.3,
        label="Original unit-heat cost",
    )
    axis.plot(
        values["minutes_from_stable"],
        values["inverse_cop_v2_plot"],
        color="#E28E2C",
        linewidth=1.5,
        label="V2 water-side full-cycle proxy",
    )
    unsupported = values["inverse_cop_v2_plot"].isna()
    if unsupported.any():
        axis.scatter(
            values.loc[unsupported, "minutes_from_stable"],
            np.full(int(unsupported.sum()), 0.02),
            marker="x",
            s=12,
            linewidths=0.7,
            color="#A7ADB3",
            label="V2 outside joint support",
            transform=axis.get_xaxis_transform(),
            zorder=3,
        )
    markers = (
        (
            values.attrs["unit_optimum_minutes"],
            "Original optimum",
            "#3775BA",
            "--",
        ),
        (values.attrs["v2_optimum_minutes"], "V2 optimum", "#E28E2C", "-"),
        (values.attrs["rule_minutes"], "Rule trigger", "#C82423", "-."),
    )
    marker_positions = []
    for position, label, color, linestyle in markers:
        if np.isfinite(position):
            marker_positions.append(float(position))
            axis.axvline(
                position,
                color=color,
                linewidth=1.0,
                linestyle=linestyle,
                label=label,
                zorder=2,
            )
    observed_positions = pd.to_numeric(
        values["minutes_from_stable"], errors="coerce"
    ).dropna()
    if not observed_positions.empty:
        left = min([0.0, float(observed_positions.min()), *marker_positions])
        right = max([float(observed_positions.max()), *marker_positions])
        pad = max((right - left) * 0.03, 0.5)
        axis.set_xlim(left - pad, right + pad)
    location = values.attrs["v2_minimum_location"].replace("_", " ")
    axis.set(
        xlabel="Time from stable heating start [min]",
        ylabel="Electricity per delivered heat, E/Q [-]",
        title=f"{cycle_name}  |  V2 minimum: {location}",
    )
    axis.legend(frameon=False, fontsize=7, ncols=2)
    return figure


def plot_cycle_overlays(
    curves: pd.DataFrame,
    comparison: pd.DataFrame,
    points: pd.DataFrame,
    output: Path,
) -> list[str]:
    """Write one established cycle-time overlay per cycle and a PDF atlas."""
    output.mkdir(parents=True, exist_ok=True)
    atlas_path = output.parent / "逐循环成本曲线图集.pdf"
    point_rows = points.set_index("cycle_name")
    comparison_rows = comparison.set_index("cycle_name")
    exported = []
    with PdfPages(atlas_path) as atlas:
        for cycle_name, cycle_curves in curves.groupby("cycle_name", sort=True):
            if cycle_name not in point_rows.index or cycle_name not in comparison_rows.index:
                continue
            stable = pd.to_datetime(
                point_rows.loc[cycle_name].get("t_heating_stable"), errors="coerce"
            )
            if pd.isna(stable):
                continue
            values = prepare_cycle_overlay(
                cycle_curves,
                comparison_rows.loc[cycle_name],
                pd.Timestamp(stable),
            )
            figure = _cycle_overlay_figure(str(cycle_name), values)
            figure.savefig(
                output / f"{cycle_name}.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            atlas.savefig(figure, bbox_inches="tight", facecolor="white")
            figure.clear()
            exported.append(str(cycle_name))
    expected = {f"{cycle}.png" for cycle in exported}
    for path in output.glob("frost_cycle_*.png"):
        if path.name not in expected:
            path.unlink()
    return exported


def _summary_value(summary: pd.DataFrame, metric: str, column: str) -> float:
    selected = summary.loc[summary["metric"].eq(metric), column]
    return float(selected.iloc[0]) if not selected.empty else float("nan")


def write_report(
    output: Path,
    comparison: pd.DataFrame,
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Write the fixed formula, evidence, comparison, and limitations."""
    energy = metrics.loc[
        metrics["outcome"].eq("electricity")
        & metrics["model"].eq("ridge_frozen_features")
    ].iloc[0]
    heat = metrics.loc[
        metrics["outcome"].eq("signed_transient_heat")
        & metrics["model"].eq("ridge_frozen_features")
    ].iloc[0]
    supported_fraction = _summary_value(
        summary, "v2_supported_candidate_fraction", "mean"
    )
    unit_median = _summary_value(summary, "unit_minus_v2_minutes", "median")
    unit_p10 = _summary_value(summary, "unit_minus_v2_minutes", "p10")
    unit_p90 = _summary_value(summary, "unit_minus_v2_minutes", "p90")
    unit_common_median = _summary_value(
        summary, "unit_common_minus_v2_minutes", "median"
    )
    unit_common_p10 = _summary_value(summary, "unit_common_minus_v2_minutes", "p10")
    unit_common_p90 = _summary_value(summary, "unit_common_minus_v2_minutes", "p90")
    rule_median = _summary_value(summary, "rule_minus_v2_minutes", "median")
    rule_p10 = _summary_value(summary, "rule_minus_v2_minutes", "p10")
    rule_p90 = _summary_value(summary, "rule_minus_v2_minutes", "p90")
    heat_shift = _summary_value(
        summary, "zero_transient_heat_minus_v2_minutes", "median"
    )
    valid_count = int(comparison["t_star_v2"].notna().sum())
    total_count = comparison["cycle_name"].nunique()
    location_counts = comparison.loc[
        comparison["t_star_v2"].notna(), "v2_minimum_location"
    ].value_counts()
    interior_count = int(location_counts.get("interior", 0))
    left_count = int(location_counts.get("left_boundary", 0))
    right_count = int(location_counts.get("right_boundary", 0))
    energy_line = (
        f"- ticket 电耗：n={int(energy['event_count'])}，"
        f"MAE={energy['mae']:.4f} kWh，RMSE={energy['rmse']:.4f} kWh，"
        f"实验宏平均 MAE={energy['macro_experiment_mae']:.4f} kWh；"
    )
    heat_line = (
        f"- 带符号瞬态热量：n={int(heat['event_count'])}，"
        f"MAE={heat['mae']:.4f} kWh，RMSE={heat['rmse']:.4f} kWh，"
        f"实验宏平均 MAE={heat['macro_experiment_mae']:.4f} kWh。"
    )
    report = f"""# 成本函数 V2 与除霜点比较

## 1. 每个量怎样计算

候选时刻记为 `tau`，稳定制热起点记为 `t_s`。每个候选点之前的累计量来自同一循环实测积分：

```text
E_H(tau)       = integral[t_s,tau] P_e(t) dt
Qdot_water(t)   = 1.161 Vdot(t) [T_out(t) - T_in(t)]
Q_water,H(tau) = integral[t_s,tau] max(Qdot_water(t), 0) dt
Q_unit,H(tau)  = integral[t_s,tau] max(Qdot_unit(t), 0) dt
```

其中水侧瞬时制热量由流量、比热和供回水温差得到；`Q_unit`
是原机组上报的制热量口径。所有积分沿用原分析的覆盖率、内部缺口和端点外推审计。

### 1.1 原机组制热量成本曲线

原曲线不是“只最大化瞬时制热量”，而是：

```text
J_unit(tau) = [E_H(tau) + E_PD_hat(p_evap(tau)) + E_R_fixed]
              / Q_unit,H(tau)
```

`E_PD_hat = b0 + b1 p_evap` 是按实验留出的蒸发压力线性模型，
`E_R_fixed = 0.2799019 kWh`。它把准备/除霜/恢复的电耗加进分子，
但分母只使用候选点前累计的机组制热量，没有加入除霜和恢复阶段的带符号热量。

### 1.2 V2 的两个票价模型

V2 只用两个自由度很低的 ridge 模型：

```text
E_ticket_hat = beta_0 + beta_1 z(T_ambient) + beta_2 z(T_evaporating)
Q_ticket_hat = gamma_0 + gamma_1 z(T_coil) + gamma_2 z(minutes_from_stable)
z(x_j)        = (x_j - mean_train,j) / sd_train,j
```

对每个待评估实验 `e`，训练集只含其他实验 `e' != e`。设标准化后的设计矩阵为 `Z=[1,z_1,z_2]`，则：

```text
theta_hat = (Z'Z + alpha diag(0,1,1))^(-1) Z'y,  alpha = 1
```

截距不惩罚；训练缺失值以该训练折中位数填充。预测再裁剪到该训练折真实
outcome 的最小值和最大值，避免 ridge 给出超出已见事件量级的结果。
裁剪只能限制数值范围，不能证明反事实正确。

电耗 outcome 是实际准备—除霜—恢复 ticket 电耗；热量 outcome 是同一
ticket 内的带符号水侧瞬态热量，因此热泵从水侧吸热时可以为负。

### 1.3 V2 完整周期代理

```text
J_V2(tau) = [E_H(tau) + E_ticket_hat(x_tau)]
            / [Q_water,H(tau) + Q_ticket_hat(x_tau)]
```

只在总电耗和总热量均为正时计算。V2 与原曲线的主要结构差异有两个：
票价电耗模型不同；V2 在分母加入了带符号瞬态水侧热量。

### 1.4 规则除霜点

规则点不是第三条成本函数，而是原控制器的一秒级因果回放：

```text
t_RB = inf{{t: C1(t) or C2(t) or C7(t) or C8(t)}}
```

例如 `C1` 要求 `T1>35 min, T2>=6 min, T3<=-1 C` 且过去窗口盘管温降
至少 `1 C`；`C2` 使用由环境温度、出水温度和基准盘管温度决定的时间/温度阈值，
并连续满足 20 s；`C7` 是低盘管温度的 20 s 持续条件；`C8` 是
`T1>=150 min` 的兜底。它回答“原规则何时会触发”，不直接最小化 `E/Q`。

## 2. 联合支持率是什么意思

对候选时刻 `tau`，电耗模型支持要求 `T_ambient` 与 `T_evaporating`
均非缺失，并分别位于其他实验训练事件的 min–max 内；热量模型支持要求
`T_coil` 与 `minutes_from_stable` 也满足同一条件：

```text
S_E(tau) = all_j [x_E,j is finite and min_train,j <= x_E,j <= max_train,j]
S_Q(tau) = all_j [x_Q,j is finite and min_train,j <= x_Q,j <= max_train,j]
S_joint(tau) = S_E(tau) and S_Q(tau)
```

主 V2 候选还必须满足原积分 `optimization_eligible`，以及 `E_total>0`、
`Q_total>0`。因此候选联合支持率的分母是全部候选时刻，不是循环数：

```text
joint support rate = count(S_joint=True) / count(all candidate times)
```

这里的“联合”只表示两个模型的四个特征分别通过边缘 min–max 检查。
**联合支持不是四维联合密度支持**：它不检查特征组合是否真的在训练云内部，
也不等于凸包、KDE 支持或因果 overlap。它只是一个保守程度有限、
容易审计的“禁止单变量范围外推”门槛。

`Q_ticket_hat` 尚未扣除水箱/管路储热，所以 V2 是水侧完整周期代理，不是真实用户侧全局最优。

## 3. 留出预测

{energy_line}
{heat_line}

## 4. 三类除霜点比较

- V2 有效最优点：{valid_count}/{total_count} 个循环；
  候选联合支持率 {supported_fraction:.1%}；
  内部/左边界/右边界最小值为 {interior_count}/{left_count}/{right_count}；
- 原机组制热量最优点相对 V2：中位数 {unit_median:.1f} min，
  P10–P90 为 {unit_p10:.1f}–{unit_p90:.1f} min；
- 限制到 V2 共同支持域后，机组制热量最优点相对 V2：
  中位数 {unit_common_median:.1f} min，
  P10–P90 为 {unit_common_p10:.1f}–{unit_common_p90:.1f} min；
- 规则点相对 V2：中位数 {rule_median:.1f} min，
  P10–P90 为 {rule_p10:.1f}–{rule_p90:.1f} min；
- 将瞬态热量固定为 0 后，最优点相对完整 V2 的位移中位数
  {heat_shift:.1f} min；这是当前储热/热量口径敏感性的直接审计。

正时间差表示原方法或规则点晚于 V2，负值表示早于 V2。
`v2_regret_at_*` 仅在相应时刻能匹配到 0.51 min 内的 V2 支持候选时计算。

## 结论边界

该比较检验的是三种离线决策定义在同一历史数据上的差异，
不能验证未执行候选的真实反事实结果。特征集合在本次探索中按物理含义冻结，
但未经过独立外部数据确认；结果应称为“模型依赖的 V2 最优点”。
下一步应优先补储热修正与整实验日不确定性传播，而不是据此直接上线控制。

## 5. 怎样用实验判断成本函数

历史数据里每个循环只执行了一个除霜时刻，所以同一循环其他曲线点都是反事实预测。
不能用某个模型自己算出的 `J` 去证明它自己正确。验证分两阶段：

### 阶段 A：离线淘汰明显不可信的模型

1. 在真实执行过的 ticket 上做 experiment-LOEO，比较电耗和带符号热量的 MAE、RMSE、bias 与区间覆盖；
2. 检查候选支持率、边界最小值比例、整实验留出后最优点位移；
3. 对储热修正、recovery 终点和积分边界做敏感性分析；
4. 若模型在真实执行点都预测不好、最优点大量落在支持边界，直接淘汰。

这一步只能排除坏模型，不能证明某条曲线的未执行最优点是真实最优。

### 阶段 B：前瞻分歧集随机试验

先把每套回顾性 `argmin J(tau)` 冻结成真正因果可执行的策略。
不能等一个循环结束后再把已发生曲线的最低点当作当时可执行的动作。
每套策略必须只读当前及过去数据，例如预先承诺的候选时刻，
或带持续确认/迟滞的单侧在线停止规则；两套策略使用同样的信息截止点。

只保留两套最终策略 A/B。仅当两者建议时刻相差至少预先规定的 `Delta` 时进入分歧集：

```text
D = {{cycle: abs(tau_A - tau_B) >= Delta}}
```

在每个分歧循环进入决策窗口前，以 1:1 随机指定整条 A 或 B 策略；
不要求相邻循环或环境相同。可按实验日、环境温度区间等做分层随机，
避免某策略系统性只在冷湿时执行。两者一致的循环不消耗稀缺验证样本。

主要结局必须来自统一控制体和统一终点的真实量：

```text
rho_A = sum(E_real,A) / sum(Q_useful,real,A)
rho_B = sum(E_real,B) / sum(Q_useful,real,B)
```

不要平均逐循环 COP。若储热不可忽略，先冻结
`Q_useful = Q_measured - Delta U_storage` 的口径。也可预先固定 `rho_0`，
比较可加结局 `Y_i=E_i-rho_0 Q_useful,i`。按实验日聚类 bootstrap/随机化检验
并报告效应区间。

约 100 个循环时不要同时比较四五套函数：先离线淘汰到两套，再把约 50/50
的稀缺动作给 A/B。若 A/A 试验显示测量噪声已经大于最小有意义差异，
则现有样本无法判定优胜者；正确结论是“不可区分”，不是选择样本均值略低者。
若完全不能改变真实除霜策略，则只能验证执行点的部件预测，
无法验证哪个离线最优点正确。
"""
    (output / "报告.md").write_text(report, encoding="utf-8")


def analyze(source: Path, evidence: Path, output: Path) -> None:
    """Build the deterministic V2 comparison from existing evidence tables."""
    events = pd.read_csv(evidence / "ticket_event_features_and_predictions.csv")
    candidate_source = evidence / "conditional_candidate_costs.parquet"
    if not candidate_source.exists():
        candidate_source = source / "candidate_cost_curves.parquet"
    curves = pd.read_parquet(candidate_source)
    points = pd.read_csv(source / "cycle_optimal_points.csv")
    curves["candidate_time"] = pd.to_datetime(curves["candidate_time"], errors="coerce")
    for column in ("t_heating_stable", "t_star_unit", "t_RB"):
        if column not in points:
            continue
        points[column] = pd.to_datetime(points[column], errors="coerce")

    v2_curves = build_v2_curves(events, curves)
    comparison = compare_points(v2_curves, points)
    predictions = ticket_model_predictions(events)
    metrics = ticket_model_metrics(predictions)
    summary = comparison_summary(comparison, v2_curves)

    source_output = output / "源数据"
    evidence_output = output / "证据"
    source_output.mkdir(parents=True, exist_ok=True)
    evidence_output.mkdir(parents=True, exist_ok=True)
    v2_curves.to_parquet(source_output / "cost_v2_candidate_curves.parquet", index=False)
    comparison.to_csv(source_output / "cost_v2_point_comparison.csv", index=False)
    metrics.to_csv(evidence_output / "cost_v2_ticket_model_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    plot_comparison(comparison, output / "比较图.png")
    plot_cycle_overlays(
        v2_curves,
        comparison,
        points,
        output / "图表" / "逐循环成本曲线",
    )
    write_report(output, comparison, v2_curves, metrics, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.evidence, args.output)


if __name__ == "__main__":
    main()
