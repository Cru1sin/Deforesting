#!/usr/bin/env python3
"""Audit empirical defrost tickets and summarize optimal windows, COP and RGB."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from frost_analysis.dataset import render_publication_asset
from frost_analysis.dataset_loader import DatasetLoader
from frost_analysis.defrost_cost import (
    build_partial_pool_curves,
    leave_one_event_out_partial_pool,
    water_side_heating_kw,
)

RAW_COLUMNS = [
    "timestamp",
    "water_flow",
    "water_in_temperature",
    "water_out_temperature",
    "power_total",
    "ambient_temperature",
    "environment_relative_humidity",
    "compressor_frequency",
    "evaporating_temperature",
]
STATE_FEATURES = [
    "minutes_from_stable",
    "cop",
    "ambient_temperature",
    "environment_relative_humidity",
    "water_in_temperature",
    "compressor_frequency",
    "evaporating_temperature",
]
DYNAMIC_BASE_FEATURES = [
    "q_heating_kw",
    "cop",
    "power_total",
    "water_flow",
    "water_delta_temperature",
    "ambient_temperature",
    "environment_relative_humidity",
    "water_in_temperature",
    "water_out_temperature",
    "compressor_frequency",
    "evaporating_temperature",
]
DYNAMIC_FEATURES = [
    "minutes_from_stable",
    *DYNAMIC_BASE_FEATURES,
    *(f"{column}_slope_per_min" for column in DYNAMIC_BASE_FEATURES),
    *(f"{column}_iqr" for column in DYNAMIC_BASE_FEATURES),
]
OUTCOMES = {
    "cost": "equivalent_cost_kwh",
    "duration": "duration_minutes",
    "electricity": "electricity_kwh",
    "thermal_shortfall": "thermal_shortfall_kwh",
}
STRATEGIES = (
    "mean",
    "time",
    "state",
    "dynamic",
    "nonlinear",
    "component",
    "partial_pool",
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def preceding_features(
    frame: pd.DataFrame,
    end: pd.Timestamp,
    *,
    seconds: int = 60,
    include_dynamics: bool = False,
) -> dict[str, float]:
    """Return raw levels, and optionally dynamics, before one candidate action."""
    values = (
        frame
        if {"q_heating_kw", "cop", "water_delta_temperature"} <= set(frame)
        else _prepare_raw_features(frame)
    )

    window = values.loc[
        values["timestamp"].gt(end - pd.Timedelta(seconds=seconds))
        & values["timestamp"].le(end)
    ]

    columns = [
        "q_heating_kw",
        "cop",
        "power_total",
        "water_flow",
        "water_delta_temperature",
        "ambient_temperature",
        "environment_relative_humidity",
        "water_in_temperature",
        "water_out_temperature",
        "compressor_frequency",
        "evaporating_temperature",
    ]
    result = {}
    for column in columns:
        if column not in window:
            result[column] = float("nan")
            if include_dynamics:
                result[f"{column}_iqr"] = float("nan")
                result[f"{column}_slope_per_min"] = float("nan")
            continue
        numeric = pd.to_numeric(window[column], errors="coerce")
        valid = numeric.notna()
        observed = numeric.loc[valid]
        result[column] = float(observed.median()) if not observed.empty else float("nan")
        if not include_dynamics:
            continue
        result[f"{column}_iqr"] = (
            float(observed.quantile(0.75) - observed.quantile(0.25))
            if not observed.empty
            else float("nan")
        )
        elapsed = (
            (window.loc[valid, "timestamp"] - window.loc[valid, "timestamp"].min())
            .dt.total_seconds()
            .to_numpy(dtype=float)
            / 60.0
        )
        result[f"{column}_slope_per_min"] = (
            float(np.polyfit(elapsed, observed.to_numpy(dtype=float), 1)[0])
            if len(observed) >= 2 and np.ptp(elapsed) > 0
            else float("nan")
        )
    return result


def _prepare_raw_features(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.copy()
    values["timestamp"] = pd.to_datetime(values["timestamp"], errors="coerce")
    values["q_heating_kw"] = water_side_heating_kw(values)
    power = pd.to_numeric(values["power_total"], errors="coerce")
    values["cop"] = values["q_heating_kw"] / power.where(power.gt(0))
    values["water_delta_temperature"] = (
        pd.to_numeric(values["water_out_temperature"], errors="coerce")
        - pd.to_numeric(values["water_in_temperature"], errors="coerce")
    )
    return values


def _ridge_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    outcome: str,
) -> np.ndarray:
    usable = [column for column in features if train[column].notna().any()]
    if not usable:
        return np.repeat(train[outcome].mean(), len(test))
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[usable], train[outcome])
    return model.predict(test[usable])


def _nonlinear_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    outcome: str,
) -> np.ndarray:
    usable = [column for column in features if train[column].notna().any()]
    if not usable:
        return np.repeat(train[outcome].mean(), len(test))
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=8,
            l2_regularization=1.0,
            random_state=0,
        ),
    )
    model.fit(train[usable], train[outcome])
    return model.predict(test[usable])


def leave_one_experiment_out_ticket_predictions(
    events: pd.DataFrame,
    state_features: list[str],
    dynamic_features: list[str] | None = None,
) -> pd.DataFrame:
    """Predict held-out ticket outcomes using only other experiments."""
    predictions = []
    dynamic_features = dynamic_features or state_features
    for experiment in sorted(events["experiment_id"].unique()):
        train = events.loc[~events["experiment_id"].eq(experiment)]
        test = events.loc[events["experiment_id"].eq(experiment)].copy()
        test["training_event_count"] = len(train)
        for label, outcome in OUTCOMES.items():
            test[f"predicted_mean_{label}"] = train[outcome].mean()
            test[f"predicted_time_{label}"] = _ridge_predict(
                train, test, ["minutes_from_stable"], outcome
            )
            test[f"predicted_state_{label}"] = _ridge_predict(
                train, test, state_features, outcome
            )
            test[f"predicted_dynamic_{label}"] = _ridge_predict(
                train, test, dynamic_features, outcome
            )
            test[f"predicted_nonlinear_{label}"] = _nonlinear_predict(
                train, test, dynamic_features, outcome
            )
        thermal_weight = (
            (train["equivalent_cost_kwh"] - train["electricity_kwh"])
            / train["thermal_shortfall_kwh"].where(train["thermal_shortfall_kwh"].gt(0))
        ).median()
        test["predicted_component_cost"] = (
            test["predicted_dynamic_electricity"]
            + thermal_weight * test["predicted_dynamic_thermal_shortfall"]
        )
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True)


def predict_candidate_tickets(
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    state_features: list[str],
) -> pd.DataFrame:
    """Apply an experiment-held-out state model at every candidate time."""
    predictions = []
    for experiment, test in candidates.groupby("experiment_id", sort=True):
        train = events.loc[~events["experiment_id"].eq(experiment)]
        if train.empty:
            train = events
        test = test.copy()
        for label, outcome in OUTCOMES.items():
            predicted = _ridge_predict(train, test, state_features, outcome)
            test[f"predicted_ticket_{label}"] = np.clip(
                predicted, train[outcome].min(), train[outcome].max()
            )
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True)


def conditional_optimal_points(
    curves: pd.DataFrame, *, threshold: float = 0.01
) -> pd.DataFrame:
    """Return conditional minima and near-optimal windows for every cycle."""
    rows = []
    for cycle, values in curves.groupby("cycle_name", sort=True):
        values = values.sort_values("candidate_time", kind="stable")
        minimum = values["renewal_cost_conditional"].min()
        optimum = values.loc[values["renewal_cost_conditional"].eq(minimum)].iloc[0]
        near = values.loc[
            values["renewal_cost_conditional"].le((1 + threshold) * minimum)
        ]
        position = values.index.get_loc(optimum.name)
        rows.append(
            {
                "cycle_name": cycle,
                "t_star_conditional": pd.Timestamp(optimum["candidate_time"]),
                "rho_min_conditional": minimum,
                "near_opt_start_conditional": pd.to_datetime(near["candidate_time"]).min(),
                "near_opt_end_conditional": pd.to_datetime(near["candidate_time"]).max(),
                "minimum_location_conditional": "left_boundary"
                if position == 0
                else "right_boundary"
                if position == len(values) - 1
                else "interior",
            }
        )
    return pd.DataFrame(rows)


def _load_raw(loader: DatasetLoader, cycle: str) -> pd.DataFrame:
    frame = loader.load_cycle_original(cycle, columns=RAW_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame.sort_values("timestamp", kind="stable").drop_duplicates("timestamp")


def build_ticket_features(
    loader: DatasetLoader,
    tickets: pd.DataFrame,
    points: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    valid = tickets.loc[tickets["valid"]].merge(
        points[["cycle_name", "actual_minutes_from_stable"]], on="cycle_name", how="left"
    )
    valid = valid.rename(columns={"actual_minutes_from_stable": "minutes_from_stable"})
    rows = []
    for event in valid.itertuples(index=False):
        frame = _prepare_raw_features(_load_raw(loader, event.cycle_name))
        rows.append(
            {
                "cycle_name": event.cycle_name,
                **preceding_features(
                    frame,
                    pd.Timestamp(event.defrost_start),
                    include_dynamics=True,
                ),
            }
        )
    return valid.merge(pd.DataFrame(rows), on="cycle_name").merge(
        catalog[["cycle_name", "experiment_id"]], on="cycle_name", how="left"
    )


def build_candidate_features(
    loader: DatasetLoader,
    curves: pd.DataFrame,
    points: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    stable = points.set_index("cycle_name")["t_heating_stable"]
    for cycle, candidates in curves.groupby("cycle_name", sort=True):
        frame = _prepare_raw_features(_load_raw(loader, cycle))
        start = pd.Timestamp(stable.loc[cycle])
        for candidate in candidates.itertuples(index=False):
            time = pd.Timestamp(candidate.candidate_time)
            rows.append(
                {
                    "cycle_name": cycle,
                    "candidate_time": time,
                    "minutes_from_stable": (time - start).total_seconds() / 60,
                    **preceding_features(frame, time),
                }
            )
    return curves.merge(pd.DataFrame(rows), on=["cycle_name", "candidate_time"]).merge(
        catalog[["cycle_name", "experiment_id"]], on="cycle_name", how="left"
    )


def ticket_model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, values in predictions.groupby("experiment_id", sort=True):
        for strategy in STRATEGIES:
            for label, outcome in OUTCOMES.items():
                prediction = f"predicted_{strategy}_{label}"
                if prediction not in values:
                    continue
                error = (
                    values[prediction] - values[outcome]
                ).abs()
                rows.append(
                    {
                        "experiment_id": experiment,
                        "strategy": strategy,
                        "outcome": label,
                        "protocol": (
                            "leave_one_event_out_within_experiment"
                            if strategy == "partial_pool"
                            else "leave_one_experiment_out"
                        ),
                        "mae": error.mean(),
                        "event_count": len(values),
                    }
                )
    return pd.DataFrame(rows)


def partial_pool_optimal_points(curves: pd.DataFrame) -> pd.DataFrame:
    renamed = curves.rename(
        columns={"renewal_cost_partial_pool": "renewal_cost_conditional"}
    )
    return conditional_optimal_points(renamed).rename(
        columns={
            "t_star_conditional": "t_star_partial_pool",
            "rho_min_conditional": "rho_min_partial_pool",
            "near_opt_start_conditional": "near_opt_start_partial_pool",
            "near_opt_end_conditional": "near_opt_end_partial_pool",
            "minimum_location_conditional": "minimum_location_partial_pool",
        }
    )


def render_representative_cost_publication(
    loader: DatasetLoader,
    points: pd.DataFrame,
    curves: pd.DataFrame,
    output: Path,
) -> str:
    primary = points.loc[points["valid"] & points["primary_analysis"]].copy()
    interior = primary.loc[primary["minimum_location"].eq("interior")]
    pool = interior if not interior.empty else primary
    median_advance = pool["minutes_earlier_than_actual"].median()
    representative = pool.iloc[
        (pool["minutes_earlier_than_actual"] - median_advance).abs().argmin()
    ]
    cycle_name = str(representative["cycle_name"])
    record = loader.get_cycle_record(cycle_name)
    curve = curves.loc[
        curves["cycle_name"].eq(cycle_name),
        ["candidate_time", "renewal_cost_partial_pool", "relative_regret_partial_pool"],
    ].rename(
        columns={
            "renewal_cost_partial_pool": "renewal_cost_kw",
            "relative_regret_partial_pool": "relative_regret",
        }
    )
    for path in (output, output.with_suffix(".pdf")):
        render_publication_asset(
            loader.dataset_root,
            record,
            output_path=path,
            cost_curve=curve,
        )
    return cycle_name


def build_window_overview(
    loader: DatasetLoader,
    points: pd.DataFrame,
    image_labels: pd.DataFrame,
    dataset: Path,
) -> pd.DataFrame:
    rows = []
    images = image_labels.loc[image_labels["camera_role"].eq("front")].copy()
    images["image_time"] = pd.to_datetime(images["image_time"], errors="coerce")
    for point in points.loc[points["valid"]].itertuples(index=False):
        t_star = pd.Timestamp(point.t_star)
        features = preceding_features(_load_raw(loader, point.cycle_name), t_star)
        available = images.loc[images["cycle_name"].eq(point.cycle_name)].copy()
        if available.empty:
            image_time = pd.NaT
            image_path = ""
        else:
            available["distance"] = (available["image_time"] - t_star).abs()
            nearest = available.loc[available["distance"].idxmin()]
            image_time = nearest["image_time"]
            candidate = dataset / str(nearest["image_path"])
            image_path = str(candidate) if candidate.is_file() else ""
        rows.append(
            {
                "cycle_name": point.cycle_name,
                "cohort_tier": point.cohort_tier,
                "minimum_location": point.minimum_location,
                "is_censored": point.is_censored,
                "minutes_from_stable": point.minutes_from_stable,
                "near_opt_start_minutes": (
                    pd.Timestamp(point.near_opt_start) - pd.Timestamp(point.t_heating_stable)
                ).total_seconds()
                / 60,
                "near_opt_end_minutes": (
                    pd.Timestamp(point.near_opt_end) - pd.Timestamp(point.t_heating_stable)
                ).total_seconds()
                / 60,
                "t_star": t_star,
                "cop_at_t_star_60s": features["cop"],
                "q_at_t_star_60s_kw": features["q_heating_kw"],
                "power_at_t_star_60s_kw": features["power_total"],
                "front_image_time": image_time,
                "front_image_path": image_path,
                "front_image_available": bool(image_path),
                "front_image_offset_seconds": abs((image_time - t_star).total_seconds())
                if pd.notna(image_time)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["minutes_from_stable", "cycle_name"], kind="stable"
    ).reset_index(drop=True)


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def plot_ticket_audit(
    events: pd.DataFrame,
    metrics: pd.DataFrame,
    shifts: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), gridspec_kw={"hspace": 0.42, "wspace": 0.34})
    valid = events.sort_values("equivalent_cost_kwh").reset_index(drop=True)
    thermal_equivalent = valid["equivalent_cost_kwh"] - valid["electricity_kwh"]
    axes[0, 0].bar(range(len(valid)), valid["electricity_kwh"], color="#72B7B2", width=0.9)
    axes[0, 0].bar(
        range(len(valid)),
        thermal_equivalent,
        bottom=valid["electricity_kwh"],
        color="#A6C8E0",
        width=0.9,
    )
    axes[0, 0].axhline(valid["equivalent_cost_kwh"].mean(), color="#D95F02", lw=1)
    axes[0, 0].set(xlabel="Defrost/recovery event (sorted)", ylabel="Ticket cost (kWh-eq.)")
    axes[0, 0].legend(
        handles=[
            Patch(color="#72B7B2", label="Electricity"),
            Patch(color="#A6C8E0", label="Thermal shortfall equivalent"),
            Line2D([], [], color="#D95F02", lw=1, label="Mean"),
        ],
        fontsize=5.5,
    )

    axes[0, 1].scatter(
        valid["minutes_from_stable"], valid["equivalent_cost_kwh"], s=15, color="#4C78A8", alpha=0.8
    )
    rho = valid[["minutes_from_stable", "equivalent_cost_kwh"]].corr(method="spearman").iloc[0, 1]
    axes[0, 1].text(0.04, 0.94, f"Spearman ρ = {rho:.2f}", transform=axes[0, 1].transAxes, va="top")
    axes[0, 1].set(xlabel="Observed defrost time (min)", ylabel="Ticket cost (kWh-eq.)")

    labels = {"mean": "Mean", "time": "Time linear", "state": "State ridge"}
    positions = np.arange(3)
    for offset, outcome in ((-0.13, "cost"), (0.13, "duration")):
        raw = [
            metrics.loc[
                metrics["strategy"].eq(strategy) & metrics["outcome"].eq(outcome), "mae"
            ].mean()
            for strategy in labels
        ]
        values = np.asarray(raw) / raw[0]
        axes[1, 0].bar(
            positions + offset,
            values,
            width=0.24,
            color="#4C78A8" if outcome == "cost" else "#F2A65A",
            label="Ticket cost" if outcome == "cost" else "Duration",
        )
    axes[1, 0].axhline(1, color="#555555", ls="--", lw=0.8)
    axes[1, 0].set(
        xticks=positions,
        xticklabels=list(labels.values()),
        ylabel="Held-out MAE / mean-baseline MAE",
    )
    axes[1, 0].legend(fontsize=5.5)

    shift_data = [
        shifts["median_ticket_shift_minutes"].abs().dropna(),
        shifts["conditional_shift_minutes"].abs().dropna(),
    ]
    axes[1, 1].boxplot(
        shift_data,
        tick_labels=["Mean → median", "Mean → state ridge"],
        patch_artist=True,
        boxprops={"facecolor": "#C6DBEF", "edgecolor": "#4C78A8"},
        medianprops={"color": "#D95F02"},
        whiskerprops={"color": "#4C78A8"},
        capprops={"color": "#4C78A8"},
        flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "#777777"},
    )
    rng = np.random.default_rng(0)
    for index, values in enumerate(shift_data, start=1):
        axes[1, 1].scatter(
            index + rng.uniform(-0.07, 0.07, len(values)),
            values,
            s=7,
            color="#4C78A8",
            alpha=0.35,
        )
    axes[1, 1].set(ylabel="Absolute optimum shift (min)")

    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontsize=9, fontweight="bold")
    _save_figure(fig, output)


def plot_window_cop_rgb(overview: pd.DataFrame, output: Path) -> None:
    colors = overview["minimum_location"].map(
        {"interior": "#4C78A8", "left_boundary": "#E6A34A", "right_boundary": "#8C8C8C"}
    ).to_numpy()
    x = np.arange(len(overview))
    fig = plt.figure(figsize=(7.2, 8.9))
    grid = fig.add_gridspec(
        4, 2, height_ratios=[1.15, 1.0, 0.18, 5.0], hspace=0.28, wspace=0.28
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, :])
    ax_a.vlines(
        x,
        overview["near_opt_start_minutes"],
        overview["near_opt_end_minutes"],
        color=colors,
        alpha=0.55,
        lw=1.2,
    )
    ax_a.scatter(x, overview["minutes_from_stable"], color=colors, s=10, zorder=3)
    ax_a.set(ylabel="Time from stable heating (min)", xticks=[])
    ax_a.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color="#4C78A8", label="Interior minimum"),
            Line2D([], [], marker="o", ls="", color="#8C8C8C", label="Right boundary"),
        ],
        loc="upper left",
        ncol=2,
        fontsize=5.5,
    )
    ax_b.scatter(x, overview["cop_at_t_star_60s"], color=colors, s=12)
    ax_b.axhline(overview["cop_at_t_star_60s"].median(), color="#D95F02", lw=0.9)
    ax_b.set(ylabel="COP in preceding 60 s", xticks=[])
    ax_a.text(-0.04, 1.05, "a", transform=ax_a.transAxes, fontsize=9, fontweight="bold")
    ax_b.text(-0.04, 1.05, "b", transform=ax_b.transAxes, fontsize=9, fontweight="bold")

    ax_c = fig.add_subplot(grid[2, :])
    ax_c.axis("off")
    ax_c.text(-0.04, 0.5, "c", transform=ax_c.transAxes, fontsize=9, fontweight="bold")
    ax_c.text(
        0,
        0.5,
        "Front-view RGB available for "
        f"{int(overview['front_image_available'].sum())}/{len(overview)} cycles; "
        "grey tiles await cloud retrieval",
        va="center",
        fontsize=5.5,
    )

    image_grid = grid[3, :].subgridspec(9, 7, hspace=0.08, wspace=0.04)
    for index in range(63):
        axis = fig.add_subplot(image_grid[index // 7, index % 7])
        axis.set_xticks([])
        axis.set_yticks([])
        if index >= len(overview):
            axis.axis("off")
            continue
        row = overview.iloc[index]
        image_path = row["front_image_path"]
        path = Path(image_path) if isinstance(image_path, str) and image_path else None
        if path and path.is_file():
            with Image.open(path) as image:
                axis.imshow(np.asarray(image.convert("RGB")))
        else:
            axis.set_facecolor("#ECECEC")
            axis.text(
                0.5,
                0.52,
                "cloud\npending",
                ha="center",
                va="center",
                fontsize=4.2,
                color="#666666",
            )
        cycle = int(str(row["cycle_name"]).rsplit("_", 1)[-1])
        axis.text(
            0.02,
            0.98,
            f"C{cycle:03d} · {row['minutes_from_stable']:.0f}m\nCOP {row['cop_at_t_star_60s']:.2f}",
            transform=axis.transAxes,
            va="top",
            fontsize=3.9,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 0.5},
        )
        color = colors[index]
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(0.8)
    fig.text(0.5, 0.01, "Cycles ordered by empirical optimum time", ha="center", fontsize=7)
    _save_figure(fig, output)


def analyze(dataset: Path, source: Path, output: Path) -> None:
    # Stage 1: load the raw-data cost baseline.
    loader = DatasetLoader(dataset)
    catalog = loader.list_cycles()
    tickets = pd.read_csv(source / "defrost_ticket_events.csv")
    points = pd.read_csv(source / "cycle_optimal_points.csv")
    curves = pd.read_parquet(source / "candidate_cost_curves.parquet")
    for column in ("t_heating_stable", "t_star", "near_opt_start", "near_opt_end"):
        points[column] = pd.to_datetime(points[column], errors="coerce")
    curves["candidate_time"] = pd.to_datetime(curves["candidate_time"], errors="coerce")

    # Stage 2: audit observed defrost-ticket cost models.
    event_features = build_ticket_features(loader, tickets, points, catalog)
    event_predictions = leave_one_experiment_out_ticket_predictions(
        event_features, STATE_FEATURES, DYNAMIC_FEATURES
    )
    event_predictions = event_predictions.merge(
        leave_one_event_out_partial_pool(event_features), on="cycle_name", how="left"
    )
    metrics = ticket_model_metrics(event_predictions)
    # Stage 3: recover empirical economic windows under each ticket model.
    partial_pool_curves = build_partial_pool_curves(curves, event_features, catalog)
    partial_pool_points = partial_pool_optimal_points(partial_pool_curves)
    candidate_features = build_candidate_features(
        loader, curves, points.loc[points["valid"]], catalog
    )
    conditional_curves = predict_candidate_tickets(
        event_features, candidate_features, STATE_FEATURES
    )
    conditional_curves["renewal_cost_conditional"] = (
        conditional_curves["heating_cost_kwh"] + conditional_curves["predicted_ticket_cost"]
    ) / (
        conditional_curves["heating_hours"]
        + conditional_curves["predicted_ticket_duration"] / 60
    )
    conditional = conditional_optimal_points(conditional_curves)
    shifts = points.loc[
        points["valid"], ["cycle_name", "t_star", "median_ticket_shift_minutes"]
    ].merge(
        conditional, on="cycle_name", how="left"
    )
    shifts["conditional_shift_minutes"] = (
        shifts["t_star_conditional"] - shifts["t_star"]
    ).dt.total_seconds() / 60
    # Stage 4: connect windows to COP and RGB evidence.
    labels = pd.read_parquet("report/rgb_cost_labels/image_cost_labels.parquet")
    overview = build_window_overview(loader, points, labels, dataset)

    # Stage 5: write publication evidence from the same tables.
    output.mkdir(parents=True, exist_ok=True)
    event_predictions.to_csv(output / "ticket_event_features_and_predictions.csv", index=False)
    metrics.to_csv(output / "ticket_model_metrics_by_experiment.csv", index=False)
    partial_pool_curves.to_parquet(output / "partial_pool_candidate_costs.parquet", index=False)
    partial_pool_points.to_csv(output / "partial_pool_optimal_points.csv", index=False)
    conditional_curves.to_parquet(output / "conditional_candidate_costs.parquet", index=False)
    shifts.to_csv(output / "conditional_optimal_points.csv", index=False)
    overview.to_csv(output / "optimal_window_cop_rgb.csv", index=False)
    plot_ticket_audit(
        event_predictions,
        metrics,
        shifts,
        output.parent / "figures" / "figure_ticket_cost_audit",
    )
    plot_window_cop_rgb(
        overview,
        output.parent / "figures" / "figure_window_cop_rgb_overview",
    )
    render_representative_cost_publication(
        loader,
        points,
        partial_pool_curves,
        output.parent / "figures" / "cycles" / "representative_publication_cost.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--source", type=Path, default=Path("report/raw_optimal_defrost/source_data")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("report/raw_optimal_defrost/evidence")
    )
    args = parser.parse_args()
    analyze(args.dataset, args.source, args.output)


if __name__ == "__main__":
    main()
