"""Analyze catalog positions 0–48 and write the scientific report source data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from frost_analysis.dataset_loader import DatasetLoader
from frost_analysis.degradation_law import (
    fit_hinge,
    leave_group_out_reference,
    select_valid_catalog_positions,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "01_制热量与退化" / "退化规律_0至48循环"
FIGURES = OUT / "图表"
SOURCE = OUT / "源数据"

CONTEXT = [
    "ambient_temperature",
    "water_in_temperature",
    "water_flow",
    "compressor_frequency",
    "fan_speed",
    "exv_opening",
]
TARGETS = ["heating_capacity", "cop"]
SENSORS = [
    "evaporating_temperature",
    "coil_temperature",
    "suction_temperature",
    "evaporator_inlet_temperature",
    "plate_heat_exchanger_outlet_temperature",
    "pressure_ratio",
]
LOAD_COLUMNS = [
    "timestamp",
    "cycle_stage",
    "defrost_active",
    *CONTEXT,
    "environment_temperature",
    "environment_relative_humidity",
    *TARGETS,
    "evaporator_capacity",
    "evaporating_pressure",
    *SENSORS,
]
STATE_NAMES = {
    "state_evaporating_temperature": "Evaporating-temperature deficit",
    "state_coil_temperature": "Coil-temperature deficit",
    "state_suction_temperature": "Suction-temperature deficit",
    "state_evaporator_inlet_temperature": "Evaporator-inlet-temperature deficit",
    "state_plate_heat_exchanger_outlet_temperature": "PHE-outlet-temperature deficit",
    "state_pressure_ratio": "Pressure-ratio excess",
    "minute": "Elapsed time",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    loader = DatasetLoader(ROOT / "dataset")
    cohort = select_valid_catalog_positions(loader.list_cycles(), last_position=48)
    minute = build_minute_frame(loader, cohort)
    analyzed = add_normal_references(minute)
    active = analyzed.loc[analyzed["compressor_frequency"].gt(5)].copy()

    comparison = compare_laws(active)
    by_date = date_parameters(active)
    dynamics = compare_dynamics(active)
    rgb = rgb_coverage(loader, cohort)
    cycle_summary = summarize_cycles(active, cohort)
    availability = channel_availability(minute)
    sensitivity = compare_reference_windows(minute)
    bootstrap = bootstrap_law(active)

    cohort.to_csv(SOURCE / "cohort_0_48_valid.csv", index=False)
    cycle_summary.to_csv(SOURCE / "cycle_summary.csv", index=False)
    active.to_csv(SOURCE / "minute_state_performance.csv", index=False)
    comparison.to_csv(SOURCE / "law_comparison.csv", index=False)
    by_date.to_csv(SOURCE / "law_parameters_by_date.csv", index=False)
    dynamics.to_csv(SOURCE / "state_dynamics.csv", index=False)
    rgb.to_csv(SOURCE / "rgb_coverage.csv", index=False)
    availability.to_csv(SOURCE / "channel_availability_by_date.csv", index=False)
    sensitivity.to_csv(SOURCE / "reference_window_sensitivity.csv", index=False)

    best = active[["state_evaporating_temperature", "D_COP"]].dropna()
    law = fit_hinge(best["state_evaporating_temperature"], best["D_COP"])
    summary = {
        "scope": {
            "catalog_positions": "0-48 inclusive",
            "cycle_names": "frost_cycle_000001-frost_cycle_000049",
            "valid_cycles": int(len(cohort)),
            "experiment_dates": int(cohort["experiment_date"].astype(str).str[:10].nunique()),
            "cycles_over_30_minutes": int(cycle_summary["analysis_minutes"].gt(30).sum()),
        },
        "normal_reference": {
            "method": "leave-one-experiment-date-out early-window context model plus cycle offset",
            "early_window_minutes": 10,
            "dataset_baseline_managed": bool(loader.registry.get("baseline_managed", False)),
        },
        "primary_law": {
            "target": "D_COP = 1 - COP/COP0(c,u)",
            "state": "z = Te0(c,u) - Te",
            "form": "D_COP = alpha * max(z-zc, 0)",
            "threshold_degC": law.threshold,
            "slope_per_degC": law.slope,
            "fit_rmse": law.rmse,
            "bootstrap_threshold_95pct": bootstrap["threshold"],
            "bootstrap_slope_95pct": bootstrap["slope"],
        },
        "rgb": {
            "metadata_cycles": int(rgb["metadata_images"].gt(0).sum()),
            "local_raw_image_cycles": int(rgb["local_images"].gt(0).sum()),
            "panel_cycles": int(rgb["panel_available"].sum()),
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    configure_plotting()
    figure_cycle_atlas(cycle_summary, active)
    figure_target_comparison(active, comparison)
    figure_collapse(active, law)
    figure_stability(comparison, by_date)
    figure_dynamics_and_reversal(analyzed, dynamics)
    figure_rgb_montage(loader)


def build_minute_frame(loader: DatasetLoader, cohort: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for record in cohort.to_dict(orient="records"):
        frame = loader.load_cycle(str(record["cycle_name"]), columns=LOAD_COLUMNS)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        start = pd.to_datetime(record.get("baseline_end"), errors="coerce")
        if pd.isna(start):
            start = pd.to_datetime(record["stable_heating_start"]) + pd.Timedelta(minutes=1)
        end = pd.to_datetime(record.get("defrost_start"), errors="coerce")
        if pd.isna(end):
            end = pd.to_datetime(record["end_time"])
        frame = frame.loc[frame["timestamp"].between(start, end)].copy()
        stages = frame.set_index("timestamp")["cycle_stage"].resample("1min").agg(_mode)
        numeric = (
            frame.set_index("timestamp").select_dtypes(include="number").resample("1min").median()
        )
        reduced = numeric.join(stages).reset_index()
        reduced["cycle"] = str(record["cycle_name"])
        reduced["date"] = str(record["experiment_date"])[:10]
        reduced["reason"] = str(record.get("status_reason") or "")
        reduced["minute"] = (reduced["timestamp"] - start).dt.total_seconds() / 60.0
        reduced["early"] = reduced["minute"].between(0, 10)
        rows.append(reduced)
    return pd.concat(rows, ignore_index=True)


def _mode(values: pd.Series) -> str:
    modes = values.dropna().astype(str).mode()
    return "" if modes.empty else str(modes.iloc[0])


def add_normal_references(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in [*TARGETS, *SENSORS]:
        result[f"{column}_normal"] = leave_group_out_reference(
            result,
            target=column,
            features=CONTEXT,
            group="date",
            early="early",
            cycle="cycle",
        )
    result["D_Q"] = 1.0 - result["heating_capacity"] / result["heating_capacity_normal"]
    result["D_COP"] = 1.0 - result["cop"] / result["cop_normal"]
    for sensor in SENSORS:
        if sensor == "pressure_ratio":
            result[f"state_{sensor}"] = result[sensor] - result[f"{sensor}_normal"]
        else:
            result[f"state_{sensor}"] = result[f"{sensor}_normal"] - result[sensor]
    return result


def compare_laws(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    states = ["minute", *[f"state_{sensor}" for sensor in SENSORS]]
    for target in ["D_Q", "D_COP"]:
        for state in states:
            for form in ["linear", "hinge"]:
                errors: list[float] = []
                parameters: list[tuple[float, float]] = []
                for date in pd.unique(frame["date"]):
                    train = frame.loc[frame["date"].ne(date), [state, target]].dropna()
                    test = frame.loc[frame["date"].eq(date), [state, target]].dropna()
                    if len(train) < 30 or len(test) < 5:
                        continue
                    if form == "linear":
                        x = np.maximum(train[state].to_numpy(dtype=float), 0.0)
                        y = train[target].to_numpy(dtype=float)
                        slope = max(0.0, float(x @ y) / float(x @ x))
                        threshold = 0.0
                        prediction = slope * np.maximum(test[state].to_numpy(), 0.0)
                    else:
                        fitted = fit_hinge(train[state], train[target])
                        threshold, slope = fitted.threshold, fitted.slope
                        prediction = fitted.predict(test[state])
                    errors.extend((test[target].to_numpy() - prediction).tolist())
                    parameters.append((threshold, slope))
                error = np.asarray(errors)
                rows.append(
                    {
                        "target": target,
                        "state": state,
                        "state_label": STATE_NAMES[state],
                        "form": form,
                        "rmse": float(np.sqrt(np.mean(np.square(error)))),
                        "mae": float(np.mean(np.abs(error))),
                        "median_threshold": float(np.median([item[0] for item in parameters])),
                        "median_slope": float(np.median([item[1] for item in parameters])),
                        "held_out_dates": len(parameters),
                    }
                )
    rows.extend(compare_extra_dimensions(frame))
    return pd.DataFrame(rows)


def compare_extra_dimensions(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for features, label in [
        (["state_evaporating_temperature", "state_coil_temperature"], "Te deficit + coil deficit"),
        (["state_evaporating_temperature", "state_pressure_ratio"], "Te deficit + pressure ratio"),
    ]:
        errors: list[float] = []
        for date in pd.unique(frame["date"]):
            train = frame.loc[frame["date"].ne(date), [*features, "D_COP"]].dropna()
            test = frame.loc[frame["date"].eq(date), [*features, "D_COP"]].dropna()
            design = np.column_stack([np.ones(len(train)), train[features].to_numpy()])
            coefficients = np.linalg.lstsq(design, train["D_COP"].to_numpy(), rcond=None)[0]
            prediction = np.column_stack([np.ones(len(test)), test[features]]) @ coefficients
            errors.extend((test["D_COP"].to_numpy() - prediction).tolist())
        error = np.asarray(errors)
        rows.append(
            {
                "target": "D_COP",
                "state": "+".join(features),
                "state_label": label,
                "form": "two-variable linear",
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mae": float(np.mean(np.abs(error))),
                "median_threshold": np.nan,
                "median_slope": np.nan,
                "held_out_dates": int(frame["date"].nunique()),
            }
        )
    return rows


def date_parameters(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("date"):
        values = group[["state_evaporating_temperature", "D_COP"]].dropna()
        x = np.maximum(values["state_evaporating_temperature"].to_numpy(), 0.0)
        y = values["D_COP"].to_numpy()
        slope = max(0.0, float(x @ y) / float(x @ x))
        rows.append(
            {
                "date": date,
                "slope_per_degC": slope,
                "rmse": float(np.sqrt(np.mean(np.square(y - slope * x)))),
                "observations": len(values),
                "cycles": int(group["cycle"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def compare_dynamics(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["cycle", "minute"]).copy()
    state = "state_evaporating_temperature"
    ordered["state_rate_5min"] = (ordered[state] - ordered.groupby("cycle")[state].shift(5)) / 5.0
    rows: list[dict[str, object]] = []
    for horizon in [5, 10, 20, 30]:
        ordered["future_state"] = ordered.groupby("cycle")[state].shift(-horizon)
        for label, features in [
            ("persistence", []),
            ("current state", [state]),
            ("state + rate", [state, "state_rate_5min"]),
            ("state + rate + ambient", [state, "state_rate_5min", "ambient_temperature"]),
        ]:
            errors: list[float] = []
            for date in pd.unique(ordered["date"]):
                needed = ["future_state", state, *features]
                test = ordered.loc[ordered["date"].eq(date)].dropna(
                    subset=list(dict.fromkeys(needed))
                )
                if not features:
                    prediction = test[state].to_numpy()
                else:
                    train = ordered.loc[ordered["date"].ne(date)].dropna(
                        subset=["future_state", *features]
                    )
                    center = train[features].mean()
                    scale = train[features].std().replace(0.0, 1.0)
                    design = np.column_stack(
                        [np.ones(len(train)), ((train[features] - center) / scale).to_numpy()]
                    )
                    coefficients = np.linalg.lstsq(
                        design, train["future_state"].to_numpy(), rcond=None
                    )[0]
                    prediction = (
                        np.column_stack(
                            [np.ones(len(test)), ((test[features] - center) / scale).to_numpy()]
                        )
                        @ coefficients
                    )
                errors.extend((test["future_state"].to_numpy() - prediction).tolist())
            error = np.asarray(errors)
            rows.append(
                {
                    "horizon_minutes": horizon,
                    "model": label,
                    "rmse_degC": float(np.sqrt(np.mean(np.square(error)))),
                    "mae_degC": float(np.mean(np.abs(error))),
                    "observations": len(error),
                }
            )
    return pd.DataFrame(rows)


def compare_reference_windows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window in [5, 10, 20]:
        window_frame = frame.copy()
        window_frame["early"] = window_frame["minute"].between(0, window)
        analyzed = add_normal_references(window_frame)
        analyzed = analyzed.loc[analyzed["compressor_frequency"].gt(5)]
        state = analyzed["state_evaporating_temperature"]
        for target in ["D_Q", "D_COP"]:
            values = pd.DataFrame({"state": state, "loss": analyzed[target]}).dropna()
            fitted = fit_hinge(values["state"], values["loss"])
            x = np.maximum(values["state"].to_numpy(), 0.0)
            y = values["loss"].to_numpy()
            slope = max(0.0, float(x @ y) / float(x @ x))
            rows.append(
                {
                    "early_window_minutes": window,
                    "target": target,
                    "threshold_degC": fitted.threshold,
                    "hinge_slope": fitted.slope,
                    "hinge_rmse": fitted.rmse,
                    "linear_slope": slope,
                    "linear_rmse": float(np.sqrt(np.mean(np.square(y - slope * x)))),
                }
            )
    return pd.DataFrame(rows)


def rgb_coverage(loader: DatasetLoader, cohort: pd.DataFrame) -> pd.DataFrame:
    metadata = loader.load_image_metadata()
    counts = metadata.groupby("cycle_name").size()
    rows = []
    for cycle in cohort["cycle_name"].astype(str):
        rows.append(
            {
                "cycle": cycle,
                "metadata_images": int(counts.get(cycle, 0)),
                "local_images": int(len(loader.load_cycle_images(cycle))),
                "panel_available": loader.rgb_panel_path(cycle).is_file(),
            }
        )
    return pd.DataFrame(rows)


def channel_availability(frame: pd.DataFrame) -> pd.DataFrame:
    channels = [
        "environment_temperature",
        "environment_relative_humidity",
        "heating_capacity",
        "cop",
        "evaporating_temperature",
        "coil_temperature",
        "suction_temperature",
        "evaporator_inlet_temperature",
        "pressure_ratio",
    ]
    rows = []
    for date, group in frame.groupby("date"):
        row: dict[str, object] = {"date": date, "cycles": int(group["cycle"].nunique())}
        row.update({channel: float(group[channel].notna().mean()) for channel in channels})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_cycles(frame: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for cycle, group in frame.groupby("cycle", sort=False):
        tail = group.sort_values("minute").tail(10)
        record = cohort.loc[cohort["cycle_name"].eq(cycle)].iloc[0]
        summaries.append(
            {
                "cycle": cycle,
                "date": str(record["experiment_date"])[:10],
                "reason": str(record.get("status_reason") or ""),
                "analysis_minutes": float(group["minute"].max()),
                "terminal_D_Q": float(tail["D_Q"].median()),
                "terminal_D_COP": float(tail["D_COP"].median()),
                "terminal_state_degC": float(tail["state_evaporating_temperature"].median()),
                "shutdown_minutes": int(group["compressor_frequency"].le(5).sum()),
            }
        )
    return pd.DataFrame(summaries)


def bootstrap_law(frame: pd.DataFrame) -> dict[str, list[float]]:
    rng = np.random.default_rng(20260811)
    cycles = frame["cycle"].unique()
    thresholds: list[float] = []
    slopes: list[float] = []
    for _ in range(500):
        sampled = rng.choice(cycles, size=len(cycles), replace=True)
        blocks = [frame.loc[frame["cycle"].eq(cycle)] for cycle in sampled]
        values = pd.concat(blocks)[["state_evaporating_temperature", "D_COP"]].dropna()
        fitted = fit_hinge(values.iloc[:, 0], values.iloc[:, 1])
        thresholds.append(fitted.threshold)
        slopes.append(fitted.slope)
    return {
        "threshold": np.quantile(thresholds, [0.025, 0.975]).tolist(),
        "slope": np.quantile(slopes, [0.025, 0.975]).tolist(),
    }


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    for suffix, dpi in [("svg", 300), ("pdf", 300), ("png", 300)]:
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.12, 1.06, label, transform=axis.transAxes, fontsize=10, fontweight="bold")


def figure_cycle_atlas(cycles: pd.DataFrame, frame: pd.DataFrame) -> None:
    ordered = cycles.sort_values(["date", "cycle"]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 7.0), gridspec_kw={"width_ratios": [0.8, 1.6]})
    colors = ordered["reason"].map(
        lambda value: (
            "#D95F59" if "关机" in value else ("#78A6B8" if "薄霜" in value else "#305F72")
        )
    )
    y = np.arange(len(ordered))
    axes[0].barh(y, ordered["analysis_minutes"], color=colors, height=0.72)
    axes[0].set_yticks(y, ordered["cycle"].str[-3:])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Observed frost-development time (min)")
    axes[0].set_ylabel("Cycle")
    axes[0].axvline(30, color="#777777", ls="--", lw=0.8)
    axes[0].set_title("Different endpoints and durations")
    panel_label(axes[0], "a")

    maximum = int(np.ceil(ordered["analysis_minutes"].max()))
    matrix = np.full((len(ordered), maximum + 1), np.nan)
    for row, cycle in enumerate(ordered["cycle"]):
        values = frame.loc[frame["cycle"].eq(cycle), ["minute", "D_COP"]].dropna()
        indices = values["minute"].round().astype(int).clip(0, maximum)
        matrix[row, indices] = values["D_COP"].clip(-0.05, 0.55)
    image = axes[1].imshow(
        matrix, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=0.5
    )
    axes[1].set_yticks(y, ordered["cycle"].str[-3:])
    axes[1].set_xlabel("Observed frost-development time (min)")
    axes[1].set_title("Context-adjusted COP loss")
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.04, pad=0.02)
    colorbar.set_label(r"$D_{COP}$")
    panel_label(axes[1], "b")
    fig.suptitle("The valid cohort spans unequal and right-censored physical trajectories", y=0.995)
    fig.tight_layout()
    save_figure(fig, "figure_1_cycle_atlas")


def binned_curve(axis: plt.Axes, x: pd.Series, y: pd.Series, color: str) -> None:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    data["bin"] = pd.qcut(data["x"], 20, duplicates="drop")
    summary = data.groupby("bin", observed=True).agg(x=("x", "median"), y=("y", "median"))
    axis.plot(summary["x"], summary["y"], color=color, lw=2.0)


def figure_target_comparison(frame: pd.DataFrame, comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    state = frame["state_evaporating_temperature"]
    for axis, target, label in zip(
        axes[:2], ["D_Q", "D_COP"], [r"$D_Q$", r"$D_{COP}$"], strict=True
    ):
        axis.scatter(state, frame[target], s=3, alpha=0.08, color="#2F6F7E", rasterized=True)
        binned_curve(axis, state, frame[target], "#173F4F")
        axis.set(xlabel=r"$z=T_{e,0}-T_e$ (°C)", ylabel=label, xlim=(-3, 28), ylim=(-0.15, 0.75))
    axes[0].set_title("Heating-capacity loss")
    axes[1].set_title("COP loss")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    subset = comparison.loc[
        comparison["state"].eq("state_evaporating_temperature") & comparison["form"].eq("linear")
    ]
    axes[2].bar([r"$D_Q$", r"$D_{COP}$"], 100 * subset["rmse"], color=["#78A6B8", "#305F72"])
    axes[2].set_ylabel("Held-out-date RMSE (percentage points)")
    axes[2].set_title("Cross-date generalization")
    panel_label(axes[2], "c")
    fig.suptitle("COP loss is the more stable performance coordinate", y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure_2_performance_target")


def figure_collapse(frame: pd.DataFrame, law: object) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw={"width_ratios": [1.55, 1]})
    dates = sorted(frame["date"].unique())
    cmap = plt.get_cmap("viridis")
    for index, date in enumerate(dates):
        group = frame.loc[frame["date"].eq(date)]
        axes[0].scatter(
            group["state_evaporating_temperature"],
            group["D_COP"],
            s=4,
            alpha=0.18,
            color=cmap(index / max(1, len(dates) - 1)),
            rasterized=True,
        )
    x = np.linspace(0, 28, 200)
    axes[0].plot(x, law.predict(x), color="#B13B3B", lw=2.2, label="linear law (threshold ≈ 0)")
    axes[0].axvline(law.threshold, color="#B13B3B", ls="--", lw=0.8)
    axes[0].set(
        xlabel=r"State $z=T_{e,0}(c,u)-T_e$ (°C)",
        ylabel=r"$D_{COP}$",
        xlim=(-3, 28),
        ylim=(-0.15, 0.72),
    )
    axes[0].legend(frameon=False)
    axes[0].set_title("Cycles collapse in state–performance space")
    panel_label(axes[0], "a")
    axes[1].scatter(
        frame["minute"], frame["D_COP"], s=4, alpha=0.10, color="#6D8490", rasterized=True
    )
    binned_curve(axes[1], frame["minute"], frame["D_COP"], "#173F4F")
    axes[1].set(xlabel="Elapsed time (min)", ylabel=r"$D_{COP}$", ylim=(-0.15, 0.72))
    axes[1].set_title("Time does not align degradation")
    panel_label(axes[1], "b")
    fig.suptitle("A one-dimensional thermodynamic state replaces elapsed time", y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure_3_state_performance_collapse")


def figure_stability(comparison: pd.DataFrame, by_date: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    chosen = comparison.loc[
        comparison["target"].eq("D_COP")
        & comparison["form"].isin(["linear", "hinge", "two-variable linear"])
        & comparison["state"].isin(
            [
                "minute",
                "state_evaporating_temperature",
                "state_coil_temperature",
                "state_evaporating_temperature+state_coil_temperature",
            ]
        )
    ].copy()
    chosen["label"] = chosen["state_label"] + "\n" + chosen["form"]
    axes[0].barh(np.arange(len(chosen)), 100 * chosen["rmse"], color="#527E8D")
    axes[0].set_yticks(np.arange(len(chosen)), chosen["label"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Held-out-date RMSE (percentage points)")
    axes[0].set_title("Complexity does not improve generalization")
    panel_label(axes[0], "a")
    axes[1].plot(by_date["date"].str[5:], 100 * by_date["slope_per_degC"], "o-", color="#305F72")
    axes[1].axhline(100 * by_date["slope_per_degC"].median(), color="#B13B3B", ls="--", lw=1)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylabel("COP-loss slope (percentage points per °C)")
    axes[1].set_title("Date-specific law slopes")
    panel_label(axes[1], "b")
    fig.tight_layout()
    save_figure(fig, "figure_4_law_stability")


def figure_dynamics_and_reversal(frame: pd.DataFrame, dynamics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw={"width_ratios": [1, 1.35]})
    for model, group in dynamics.groupby("model"):
        axes[0].plot(group["horizon_minutes"], group["mae_degC"], marker="o", label=model)
    axes[0].set(xlabel="Forecast horizon (min)", ylabel="Held-out-date MAE (°C)")
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[0].set_title("State rate adds modest forecast information")
    panel_label(axes[0], "a")
    cycle = frame.loc[frame["cycle"].eq("frost_cycle_000043")].sort_values("minute")
    axes[1].plot(
        cycle["minute"],
        cycle["state_evaporating_temperature"],
        color="#305F72",
        lw=1.5,
        label="state z",
    )
    axes[1].set(xlabel="Minute in analyzed interval", ylabel=r"State $z$ (°C)")
    off = cycle["compressor_frequency"].le(5)
    if off.any():
        axes[1].fill_between(
            cycle["minute"],
            axes[1].get_ylim()[0],
            axes[1].get_ylim()[1],
            where=off,
            color="#D95F59",
            alpha=0.18,
            label="shutdown",
        )
    twin = axes[1].twinx()
    twin.plot(cycle["minute"], cycle["D_COP"], color="#B8792D", lw=1.1, label=r"$D_{COP}$")
    twin.set_ylabel(r"$D_{COP}$")
    axes[1].set_title("The disturbed cycle is not monotonic")
    panel_label(axes[1], "b")
    handles = axes[1].get_lines() + twin.get_lines()
    axes[1].legend(handles, [line.get_label() for line in handles], frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "figure_5_state_dynamics_reversal")


def figure_rgb_montage(loader: DatasetLoader) -> None:
    cycles = [
        "frost_cycle_000023",
        "frost_cycle_000024",
        "frost_cycle_000043",
        "frost_cycle_000042",
    ]
    labels = [
        "thin-frost endpoint",
        "thick-frost endpoint",
        "shutdown disturbance",
        "thin, right-censored",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 7.1))
    for axis, cycle, label in zip(axes.flat, cycles, labels, strict=True):
        with Image.open(loader.rgb_panel_path(cycle)) as image:
            # Old panel headers can contain status text from an earlier catalog
            # revision; retain the image evidence and use the current catalog label.
            top = int(image.height * 0.045)
            axis.imshow(image.crop((0, top, image.width, image.height)).copy())
        axis.set_title(f"{cycle[-3:]}: {label}")
        axis.axis("off")
    fig.suptitle(
        "RGB confirms heterogeneous visible states; raw frames exist for only four cycles", y=0.99
    )
    fig.tight_layout()
    save_figure(fig, "figure_6_rgb_state_examples")


if __name__ == "__main__":
    main()
