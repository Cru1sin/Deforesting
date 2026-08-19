#!/usr/bin/env python3
# ruff: noqa: E501
"""Estimate empirical defrost optima from unsmoothed original cycle data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from frost_analysis.dataset_loader import DatasetLoader
from frost_analysis.defrost_cost import (
    count_true_runs,
    find_recovery_time,
    integrate_energy_kwh,
    optimize_renewal_cost,
    water_side_heating_kw,
)

RAW_COLUMNS = [
    "timestamp",
    "water_flow",
    "water_in_temperature",
    "water_out_temperature",
    "power_total",
    "cycle_stage",
]
ANCHOR_SECONDS = 60
MINIMUM_HEATING_MINUTES = 10
CANDIDATE_STEP_MINUTES = 1
RECOVERY_FRACTION = 0.9
RECOVERY_SECONDS = 30
MINIMUM_INTEGRATION_COVERAGE = 0.95

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


def _timestamp(value: object) -> pd.Timestamp | None:
    result = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(result) else pd.Timestamp(result)


def _raw(loader: DatasetLoader, cycle_name: str) -> pd.DataFrame:
    frame = loader.load_cycle_original(cycle_name, columns=RAW_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["q_heating_kw"] = water_side_heating_kw(frame)
    frame["power_total"] = pd.to_numeric(frame["power_total"], errors="coerce")
    return frame.sort_values("timestamp", kind="stable").drop_duplicates("timestamp")


def _anchor(frame: pd.DataFrame, start: pd.Timestamp | None) -> dict[str, object]:
    if start is None:
        return {"valid": False, "invalid_reason": "missing_stable_heating_start"}
    window = frame.loc[
        frame["timestamp"].ge(start)
        & frame["timestamp"].lt(start + pd.Timedelta(seconds=ANCHOR_SECONDS))
    ]
    valid = window[["q_heating_kw", "power_total"]].dropna()
    if len(valid) < 0.8 * ANCHOR_SECONDS:
        return {"valid": False, "invalid_reason": "clean_anchor_coverage_below_80pct"}
    q_kw = float(valid["q_heating_kw"].median())
    power_kw = float(valid["power_total"].median())
    if q_kw <= 0 or power_kw <= 0:
        return {"valid": False, "invalid_reason": "nonpositive_clean_anchor"}
    return {
        "valid": True,
        "invalid_reason": "",
        "anchor_start": start,
        "q_clean_kw": q_kw,
        "power_clean_kw": power_kw,
        "cop_clean": q_kw / power_kw,
        "observed_points": len(valid),
    }


def _reference_kw(
    timestamps: pd.Series,
    start_time: pd.Timestamp,
    start_kw: float,
    end_time: pd.Timestamp,
    end_kw: float,
) -> np.ndarray:
    x = pd.to_datetime(timestamps).astype("int64").to_numpy(dtype=float)
    return np.interp(
        x,
        [float(start_time.value), float(end_time.value)],
        [start_kw, end_kw],
    )


def _candidate_costs(
    frame: pd.DataFrame,
    *,
    stable_start: pd.Timestamp,
    defrost_start: pd.Timestamp,
    q_start_kw: float,
    next_stable_start: pd.Timestamp,
    q_end_kw: float,
    lambda_q: float,
) -> pd.DataFrame:
    heating = frame.loc[
        frame["timestamp"].ge(stable_start) & frame["timestamp"].le(defrost_start)
    ].copy()
    heating["q_reference_kw"] = _reference_kw(
        heating["timestamp"], stable_start, q_start_kw, next_stable_start, q_end_kw
    )
    heating["thermal_shortfall_kw"] = np.maximum(
        heating["q_reference_kw"] - heating["q_heating_kw"], 0.0
    )
    heating["equivalent_power_kw"] = (
        heating["power_total"] + lambda_q * heating["thermal_shortfall_kw"]
    )
    first = stable_start + pd.Timedelta(minutes=MINIMUM_HEATING_MINUTES)
    candidates = list(pd.date_range(first, defrost_start, freq=f"{CANDIDATE_STEP_MINUTES}min"))
    if candidates and candidates[-1] != defrost_start:
        candidates.append(defrost_start)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        observed = heating.loc[heating["timestamp"].le(candidate)]
        energy, coverage = integrate_energy_kwh(
            observed["timestamp"], observed["equivalent_power_kw"]
        )
        rows.append(
            {
                "candidate_time": candidate,
                "heating_hours": (candidate - stable_start).total_seconds() / 3600,
                "heating_cost_kwh": energy,
                "integration_coverage": coverage,
            }
        )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _plot_cycle(
    frame: pd.DataFrame,
    result: pd.Series,
    curve: pd.DataFrame,
    *,
    q_start_kw: float,
    next_stable_start: pd.Timestamp,
    q_end_kw: float,
    output: Path,
) -> None:
    stable = pd.Timestamp(result["t_heating_stable"])
    actual = pd.Timestamp(result["t_actual_defrost"])
    optimum = pd.Timestamp(result["t_star"])
    shown = frame.loc[frame["timestamp"].between(stable, actual)].copy()
    shown["minutes"] = (shown["timestamp"] - stable).dt.total_seconds() / 60
    shown["q_reference_kw"] = _reference_kw(
        shown["timestamp"], stable, q_start_kw, next_stable_start, q_end_kw
    )
    curve = curve.copy()
    curve["minutes"] = (pd.to_datetime(curve["candidate_time"]) - stable).dt.total_seconds() / 60
    x_star = (optimum - stable).total_seconds() / 60
    x_actual = (actual - stable).total_seconds() / 60

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    axes[0].plot(
        shown["minutes"],
        shown["q_heating_kw"],
        color="#7A8793",
        lw=0.45,
        alpha=0.75,
        label="Raw water-side $Q_h$",
    )
    axes[0].plot(
        shown["minutes"], shown["q_reference_kw"], color="#2166AC", lw=1.3, label="$Q_{ref}$"
    )
    axes[0].set_ylabel("Heating capacity (kW)")
    axes[0].legend(loc="best")
    axes[1].plot(curve["minutes"], curve["renewal_cost_kw"], color="#4C78A8", lw=1.3)
    near = curve["renewal_cost_kw"].le(1.05 * curve["renewal_cost_kw"].min())
    axes[1].fill_between(
        curve["minutes"],
        0,
        1,
        where=near,
        transform=axes[1].get_xaxis_transform(),
        color="#9ECAE1",
        alpha=0.35,
        label="5% near-optimal",
    )
    for axis in axes:
        axis.axvline(x_star, color="#D95F02", lw=1.1, label="Empirical optimum")
        axis.axvline(x_actual, color="#222222", lw=0.9, ls="--", label="Observed defrost")
    axes[1].set_ylabel("Renewal cost (kW-eq.)")
    axes[1].set_xlabel("Minutes from stable heating")
    axes[1].legend(loc="best", ncol=3)
    fig.suptitle(str(result["cycle_name"]), fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_main(
    representative_frame: pd.DataFrame,
    representative: pd.Series,
    representative_curve: pd.DataFrame,
    valid_results: pd.DataFrame,
    tickets: pd.DataFrame,
    *,
    q_start_kw: float,
    next_stable_start: pd.Timestamp,
    q_end_kw: float,
    output: Path,
) -> None:
    stable = pd.Timestamp(representative["t_heating_stable"])
    actual = pd.Timestamp(representative["t_actual_defrost"])
    optimum = pd.Timestamp(representative["t_star"])
    shown = representative_frame.loc[
        representative_frame["timestamp"].between(stable, actual)
    ].copy()
    shown["minutes"] = (shown["timestamp"] - stable).dt.total_seconds() / 60
    shown["q_reference_kw"] = _reference_kw(
        shown["timestamp"], stable, q_start_kw, next_stable_start, q_end_kw
    )
    curve = representative_curve.copy()
    curve["minutes"] = (pd.to_datetime(curve["candidate_time"]) - stable).dt.total_seconds() / 60
    x_star = (optimum - stable).total_seconds() / 60
    x_actual = (actual - stable).total_seconds() / 60

    fig = plt.figure(figsize=(7.2, 6.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.42, wspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    ax_a.plot(shown["minutes"], shown["q_heating_kw"], color="#7A8793", lw=0.4, alpha=0.75)
    ax_a.plot(shown["minutes"], shown["q_reference_kw"], color="#2166AC", lw=1.25)
    ax_a.axvline(x_star, color="#D95F02", lw=1.0)
    ax_a.axvline(x_actual, color="#222222", lw=0.9, ls="--")
    ax_a.set(xlabel="Minutes from stable heating", ylabel="Heating capacity (kW)")
    ax_a.legend(
        handles=[
            Line2D([], [], color="#7A8793", lw=0.8, label="Raw $Q_h$"),
            Line2D([], [], color="#2166AC", lw=1.2, label="$Q_{ref}$"),
            Line2D([], [], color="#D95F02", lw=1.0, label="Optimum"),
            Line2D([], [], color="#222222", lw=0.9, ls="--", label="Observed"),
        ],
        loc="lower left",
        ncol=2,
        fontsize=5.5,
    )

    ax_b.plot(curve["minutes"], curve["renewal_cost_kw"], color="#4C78A8", lw=1.35)
    near = curve["renewal_cost_kw"].le(1.05 * curve["renewal_cost_kw"].min())
    ax_b.fill_between(
        curve["minutes"],
        0,
        1,
        where=near,
        transform=ax_b.get_xaxis_transform(),
        color="#9ECAE1",
        alpha=0.35,
    )
    ax_b.axvline(x_star, color="#D95F02", lw=1.0)
    ax_b.axvline(x_actual, color="#222222", lw=0.9, ls="--")
    ax_b.set(xlabel="Candidate start (min)", ylabel="Renewal cost (kW-eq.)")

    colors = valid_results["minimum_location"].map(
        {"interior": "#4C78A8", "left_boundary": "#E6A34A", "right_boundary": "#8C8C8C"}
    )
    ax_c.scatter(
        valid_results["minutes_from_stable"],
        valid_results["actual_minutes_from_stable"],
        c=colors,
        s=18,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.3,
    )
    limit = float(
        np.nanmax(valid_results[["minutes_from_stable", "actual_minutes_from_stable"]].to_numpy())
    )
    ax_c.plot([0, limit], [0, limit], color="#777777", lw=0.8, ls="--")
    ax_c.set(xlabel="Empirical optimum (min)", ylabel="Observed defrost (min)")
    ax_c.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color="#4C78A8", label="Interior"),
            Line2D([], [], marker="o", ls="", color="#8C8C8C", label="Right boundary"),
        ],
        loc="upper left",
        fontsize=5.5,
    )

    advance = valid_results["minutes_earlier_than_actual"].dropna()
    ax_d.boxplot(
        advance,
        orientation="vertical",
        widths=0.35,
        patch_artist=True,
        boxprops={"facecolor": "#C6DBEF", "edgecolor": "#4C78A8"},
        medianprops={"color": "#D95F02", "linewidth": 1.2},
        whiskerprops={"color": "#4C78A8"},
        capprops={"color": "#4C78A8"},
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": "#7A8793",
            "markeredgecolor": "none",
        },
    )
    jitter = np.random.default_rng(0).uniform(-0.07, 0.07, len(advance))
    ax_d.scatter(1 + jitter, advance, s=8, color="#4C78A8", alpha=0.45, zorder=3)
    ax_d.axhline(0, color="#777777", lw=0.8, ls="--")
    ax_d.set(xticks=[], ylabel="Minutes earlier than observed")
    ax_d.text(
        0.03,
        0.97,
        f"n = {len(valid_results)}\nvalid tickets = {tickets['valid'].sum():.0f}",
        transform=ax_d.transAxes,
        va="top",
    )

    for label, axis in zip("abcd", [ax_a, ax_b, ax_c, ax_d], strict=True):
        axis.text(-0.18, 1.08, label, transform=axis.transAxes, fontsize=9, fontweight="bold")
    fig.suptitle(
        "Raw-data empirical defrost timing under the observed fixed-duration policy",
        fontsize=8.5,
        y=0.995,
    )
    _save_figure(fig, output)


def analyze(dataset_root: Path, output_root: Path) -> None:  # noqa: C901
    loader = DatasetLoader(dataset_root)
    catalog = loader.list_cycles().sort_values(["experiment_id", "start_time"], kind="stable")
    records = {row["cycle_name"]: row for _, row in catalog.iterrows()}
    ordered = list(catalog["cycle_name"].astype(str))
    next_cycle: dict[str, str] = {}
    for current, following in zip(ordered, ordered[1:], strict=False):
        if records[current]["experiment_id"] == records[following]["experiment_id"]:
            next_cycle[current] = following

    frames = {name: _raw(loader, name) for name in ordered}
    anchors: dict[str, dict[str, object]] = {}
    anchor_rows: list[dict[str, object]] = []
    for name in ordered:
        start = _timestamp(records[name].get("stable_heating_start"))
        anchors[name] = _anchor(frames[name], start)
        anchor_rows.append({"cycle_name": name, **anchors[name]})
    anchor_table = pd.DataFrame(anchor_rows)
    clean_cop = anchor_table.loc[anchor_table["valid"], "cop_clean"].median()
    if not np.isfinite(clean_cop) or clean_cop <= 0:
        raise ValueError("No positive clean-anchor COP is available")
    lambda_q = 1 / float(clean_cop)

    ticket_rows: list[dict[str, object]] = []
    for name in ordered:
        row = records[name]
        reason = ""
        following = next_cycle.get(name)
        defrost_start = _timestamp(row.get("defrost_start"))
        defrost_end = _timestamp(row.get("defrost_end"))
        if row["status"] != "valid":
            reason = f"catalog_{row['status']}"
        elif defrost_start is None or defrost_end is None:
            reason = "missing_defrost_boundary"
        elif following is None:
            reason = "missing_next_cycle_recovery"
        elif not anchors[name]["valid"] or not anchors[following]["valid"]:
            reason = "invalid_clean_anchor"
        if reason:
            ticket_rows.append({"cycle_name": name, "valid": False, "invalid_reason": reason})
            continue

        assert following is not None and defrost_start is not None and defrost_end is not None
        next_stable = _timestamp(records[following].get("stable_heating_start"))
        assert next_stable is not None
        event = pd.concat([frames[name], frames[following]], ignore_index=True).sort_values(
            "timestamp"
        )
        recovery_search = event.loc[event["timestamp"].between(defrost_end, next_stable)]
        recovery = find_recovery_time(
            recovery_search["timestamp"],
            recovery_search["q_heating_kw"],
            reference_kw=float(anchors[following]["q_clean_kw"]),
            threshold_fraction=RECOVERY_FRACTION,
            continuous_seconds=RECOVERY_SECONDS,
        )
        if recovery is None:
            ticket_rows.append(
                {"cycle_name": name, "valid": False, "invalid_reason": "recovery_not_observed"}
            )
            continue
        event = event.loc[event["timestamp"].between(defrost_start, recovery)].copy()
        event["q_reference_kw"] = _reference_kw(
            event["timestamp"],
            pd.Timestamp(anchors[name]["anchor_start"]),
            float(anchors[name]["q_clean_kw"]),
            pd.Timestamp(anchors[following]["anchor_start"]),
            float(anchors[following]["q_clean_kw"]),
        )
        event["thermal_shortfall_kw"] = np.maximum(
            event["q_reference_kw"] - event["q_heating_kw"], 0.0
        )
        electricity, power_coverage = integrate_energy_kwh(event["timestamp"], event["power_total"])
        shortfall, heat_coverage = integrate_energy_kwh(
            event["timestamp"], event["thermal_shortfall_kw"]
        )
        valid = min(power_coverage, heat_coverage) >= MINIMUM_INTEGRATION_COVERAGE
        ticket_rows.append(
            {
                "cycle_name": name,
                "defrost_start": defrost_start,
                "recovery_stable": recovery,
                "electricity_kwh": electricity,
                "thermal_shortfall_kwh": shortfall,
                "equivalent_cost_kwh": electricity + lambda_q * shortfall,
                "duration_minutes": (recovery - defrost_start).total_seconds() / 60,
                "integration_coverage": min(power_coverage, heat_coverage),
                "valid": valid,
                "invalid_reason": "" if valid else "event_integration_coverage_below_95pct",
            }
        )
    tickets = pd.DataFrame(ticket_rows)
    valid_tickets = tickets.loc[tickets["valid"]]
    if valid_tickets.empty:
        raise ValueError("No valid empirical defrost tickets are available")
    mean_ticket_cost = float(valid_tickets["equivalent_cost_kwh"].mean())
    mean_ticket_hours = float(valid_tickets["duration_minutes"].mean() / 60)
    median_ticket_cost = float(valid_tickets["equivalent_cost_kwh"].median())
    median_ticket_hours = float(valid_tickets["duration_minutes"].median() / 60)

    result_rows: list[dict[str, object]] = []
    curves: list[pd.DataFrame] = []
    band_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for name in ordered:
        row = records[name]
        following = next_cycle.get(name)
        stable = _timestamp(row.get("stable_heating_start"))
        actual = _timestamp(row.get("defrost_start"))
        reason = ""
        if row["status"] != "valid":
            reason = f"catalog_{row['status']}"
        elif actual is None:
            reason = "missing_actual_defrost"
        elif following is None:
            reason = "missing_next_cycle_reference"
        elif not anchors[name]["valid"] or not anchors[following]["valid"]:
            reason = "invalid_clean_anchor"
        elif stable is None or (actual - stable) < pd.Timedelta(minutes=MINIMUM_HEATING_MINUTES):
            reason = "heating_interval_shorter_than_10min"
        if reason:
            result_rows.append({"cycle_name": name, "valid": False, "invalid_reason": reason})
            audit_rows.append({"cycle_name": name, "included": False, "reason": reason})
            continue

        assert following is not None and stable is not None and actual is not None
        next_stable = _timestamp(records[following].get("stable_heating_start"))
        assert next_stable is not None
        candidates = _candidate_costs(
            frames[name],
            stable_start=stable,
            defrost_start=actual,
            q_start_kw=float(anchors[name]["q_clean_kw"]),
            next_stable_start=next_stable,
            q_end_kw=float(anchors[following]["q_clean_kw"]),
            lambda_q=lambda_q,
        )
        candidates = candidates.loc[
            candidates["integration_coverage"].ge(MINIMUM_INTEGRATION_COVERAGE)
        ].reset_index(drop=True)
        if candidates.empty:
            reason = "candidate_integration_coverage_below_95pct"
            result_rows.append({"cycle_name": name, "valid": False, "invalid_reason": reason})
            audit_rows.append({"cycle_name": name, "included": False, "reason": reason})
            continue
        try:
            curve, optimum = optimize_renewal_cost(
                candidates,
                ticket_cost_kwh=mean_ticket_cost,
                ticket_duration_hours=mean_ticket_hours,
                required_end_time=actual,
            )
        except ValueError:
            reason = "candidate_domain_truncated_before_actual_defrost"
            result_rows.append({"cycle_name": name, "valid": False, "invalid_reason": reason})
            audit_rows.append({"cycle_name": name, "included": False, "reason": reason})
            continue
        curve.insert(0, "cycle_name", name)
        curve["relative_regret"] = curve["renewal_cost_kw"] / float(optimum["renewal_cost_kw"]) - 1
        curve["is_near_optimal"] = curve["renewal_cost_kw"].le(
            1.05 * float(optimum["renewal_cost_kw"])
        )
        for fraction in (0.01, 0.02, 0.05, 0.10):
            band = curve.loc[curve["relative_regret"].le(fraction), "candidate_time"]
            band_rows.append(
                {
                    "cycle_name": name,
                    "relative_regret_threshold": fraction,
                    "band_start": band.min(),
                    "band_end": band.max(),
                    "band_width_minutes": (band.max() - band.min()).total_seconds() / 60,
                    "segment_count": count_true_runs(
                        curve["relative_regret"].le(fraction).tolist()
                    ),
                }
            )
        curves.append(curve)
        t_star = pd.Timestamp(optimum["candidate_time"])
        _, median_ticket_optimum = optimize_renewal_cost(
            candidates,
            ticket_cost_kwh=median_ticket_cost,
            ticket_duration_hours=median_ticket_hours,
            required_end_time=actual,
        )
        median_ticket_t_star = pd.Timestamp(median_ticket_optimum["candidate_time"])
        constant_reference_candidates = _candidate_costs(
            frames[name],
            stable_start=stable,
            defrost_start=actual,
            q_start_kw=float(anchors[name]["q_clean_kw"]),
            next_stable_start=next_stable,
            q_end_kw=float(anchors[name]["q_clean_kw"]),
            lambda_q=lambda_q,
        )
        constant_reference_candidates = constant_reference_candidates.loc[
            constant_reference_candidates["integration_coverage"].ge(MINIMUM_INTEGRATION_COVERAGE)
        ].reset_index(drop=True)
        _, constant_reference_optimum = optimize_renewal_cost(
            constant_reference_candidates,
            ticket_cost_kwh=mean_ticket_cost,
            ticket_duration_hours=mean_ticket_hours,
            required_end_time=actual,
        )
        constant_reference_t_star = pd.Timestamp(constant_reference_optimum["candidate_time"])
        near_start = pd.Timestamp(optimum["near_opt_start"])
        near_end = pd.Timestamp(optimum["near_opt_end"])
        result_rows.append(
            {
                "cycle_name": name,
                "t_heating_stable": stable,
                "t_actual_defrost": actual,
                "t_star": t_star,
                "t_star_median_ticket": median_ticket_t_star,
                "median_ticket_shift_minutes": (median_ticket_t_star - t_star).total_seconds() / 60,
                "t_star_constant_current_reference": constant_reference_t_star,
                "constant_reference_shift_minutes": (
                    constant_reference_t_star - t_star
                ).total_seconds()
                / 60,
                "minutes_from_stable": (t_star - stable).total_seconds() / 60,
                "actual_minutes_from_stable": (actual - stable).total_seconds() / 60,
                "minutes_earlier_than_actual": (actual - t_star).total_seconds() / 60,
                "rho_min_kw_equivalent": optimum["renewal_cost_kw"],
                "near_opt_start": near_start,
                "near_opt_end": near_end,
                "near_opt_width_minutes": (near_end - near_start).total_seconds() / 60,
                "near_opt_segment_count": count_true_runs(
                    curve["relative_regret"].le(0.05).tolist()
                ),
                "minimum_location": optimum["minimum_location"],
                "valid": True,
                "invalid_reason": "",
            }
        )
        audit_rows.append({"cycle_name": name, "included": True, "reason": ""})

    results = pd.DataFrame(result_rows)
    candidate_curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    band_sensitivity = pd.DataFrame(band_rows)
    source = output_root / "source_data"
    figures = output_root / "figures"
    source.mkdir(parents=True, exist_ok=True)
    (figures / "cycles").mkdir(parents=True, exist_ok=True)
    anchor_table.to_csv(source / "clean_anchor_summary.csv", index=False)
    tickets.to_csv(source / "defrost_ticket_events.csv", index=False)
    results.to_csv(source / "cycle_optimal_points.csv", index=False)
    candidate_curves.to_parquet(source / "candidate_cost_curves.parquet", index=False)
    pd.DataFrame(audit_rows).to_csv(source / "cohort_audit.csv", index=False)
    band_sensitivity.to_csv(source / "near_optimal_band_sensitivity.csv", index=False)
    pd.DataFrame(
        [
            {
                "clean_cop": clean_cop,
                "lambda_q": lambda_q,
                "valid_ticket_count": len(valid_tickets),
                "mean_ticket_cost_kwh_equivalent": mean_ticket_cost,
                "median_ticket_cost_kwh_equivalent": median_ticket_cost,
                "mean_ticket_duration_minutes": mean_ticket_hours * 60,
                "median_ticket_duration_minutes": median_ticket_hours * 60,
            }
        ]
    ).to_csv(source / "empirical_policy_summary.csv", index=False)

    valid_results = results.loc[results["valid"]].copy()
    if valid_results.empty:
        raise ValueError("No valid cycle optimum is available")
    for _, result in valid_results.iterrows():
        name = str(result["cycle_name"])
        following = next_cycle[name]
        next_stable = _timestamp(records[following].get("stable_heating_start"))
        assert next_stable is not None
        _plot_cycle(
            frames[name],
            result,
            candidate_curves.loc[candidate_curves["cycle_name"].eq(name)],
            q_start_kw=float(anchors[name]["q_clean_kw"]),
            next_stable_start=next_stable,
            q_end_kw=float(anchors[following]["q_clean_kw"]),
            output=figures / "cycles" / f"{name}.png",
        )

    interior = valid_results.loc[valid_results["minimum_location"].eq("interior")]
    pool = interior if not interior.empty else valid_results
    median_advance = pool["minutes_earlier_than_actual"].median()
    representative = pool.iloc[
        (pool["minutes_earlier_than_actual"] - median_advance).abs().argmin()
    ]
    representative_name = str(representative["cycle_name"])
    following = next_cycle[representative_name]
    next_stable = _timestamp(records[following].get("stable_heating_start"))
    assert next_stable is not None
    _plot_main(
        frames[representative_name],
        representative,
        candidate_curves.loc[candidate_curves["cycle_name"].eq(representative_name)],
        valid_results,
        tickets,
        q_start_kw=float(anchors[representative_name]["q_clean_kw"]),
        next_stable_start=next_stable,
        q_end_kw=float(anchors[following]["q_clean_kw"]),
        output=figures / "figure_1_empirical_optimal_defrost",
    )

    counts = valid_results["minimum_location"].value_counts()
    invalid_counts = results.loc[~results["valid"], "invalid_reason"].value_counts()
    median_ticket_shift = valid_results["median_ticket_shift_minutes"].abs()
    constant_reference_shift = valid_results["constant_reference_shift_minutes"].abs()
    summary = f"""# 原始数据经验最优除霜点：论文初稿级 demo

## 结论边界

本阶段直接使用约 1 s 原始水侧与电功率数据，不进行平滑。结果是“当前固定时长除霜策略门票假设下的经验等效能耗最优启动时刻”，不是包含电价、真实热舒适与任意反事实动作的因果全局经济最优点。

## 固定方法

- 水侧制热量：`1.161 × water_flow × (water_out_temperature - water_in_temperature)`，单位 kW。
- clean reference：每循环稳定制热开始后的 60 s 中位数，与下一循环 clean anchor 线性连接。
- clean COP：{clean_cop:.3f}；热量缺口等效电量系数 `lambda_Q = 1/COP = {lambda_q:.3f}`。
- 恢复：除霜结束后原始制热量连续 {RECOVERY_SECONDS} s 达到下一 clean anchor 的 {RECOVERY_FRACTION:.0%}。
- 经验门票：{len(valid_tickets)} 个有效事件；均值成本 {mean_ticket_cost:.3f} kWh-eq.，均值时长 {mean_ticket_hours * 60:.2f} min。
- 候选：稳定制热后 {MINIMUM_HEATING_MINUTES} min 起，以 {CANDIDATE_STEP_MINUTES} min 网格搜索，并包含实际除霜时刻。
- 目标：`rho(tau) = [C_H(tau) + mean(K_D)] / [T_H(tau) + mean(T_D)]`。

## 当前结果

- catalog 循环：{len(results)}；得到有效经验最优点：{len(valid_results)}。
- 未给出点估计的 30 个循环包括：无实际除霜边界 {int(invalid_counts.get("missing_actual_defrost", 0))} 个、catalog 无效 {int(invalid_counts.get("catalog_invalid", 0))} 个、候选域被长缺口截断 {int(invalid_counts.get("candidate_domain_truncated_before_actual_defrost", 0))} 个、clean anchor 无效 {int(invalid_counts.get("invalid_clean_anchor", 0))} 个。
- 内部最小值：{int(counts.get("interior", 0))}；左边界：{int(counts.get("left_boundary", 0))}；右边界：{int(counts.get("right_boundary", 0))}。
- 相对实际除霜的提前量中位数：{valid_results["minutes_earlier_than_actual"].median():.1f} min。
- 5% near-optimal envelope 宽度中位数：{valid_results["near_opt_width_minutes"].median():.1f} min；其中 {int(valid_results["near_opt_segment_count"].gt(1).sum())} 个循环含不连续低-regret 段，因此图像标签必须使用逐图 regret，不能把 envelope 内全部时刻视为 near-optimal。
- 均值门票改为中位数门票后，最优点绝对移动量中位数：{median_ticket_shift.median():.1f} min；90% 分位：{median_ticket_shift.quantile(0.9):.1f} min。
- 双 clean anchor 改为仅用当前 clean anchor 后，最优点绝对移动量中位数：{constant_reference_shift.median():.1f} min；90% 分位：{constant_reference_shift.quantile(0.9):.1f} min；超过 30 min 的循环占比：{constant_reference_shift.gt(30).mean():.1%}。

若右边界最小值占比高，含义是观察区间尚未跨过最优点，不能强制制造内部 optimum。若左边界占比高，需优先检查固定门票是否过低或最小运行时长是否设置过晚。下一阶段是否增加工况条件化门票，只由门票残差诊断决定。

本阶段的 `lambda_Q × thermal_shortfall` 是供热服务缺口的等效电量代理。由于当前数据没有室内空气温度、PMV/PPD、占用人数或暴露时长，它不能被表述为直接测得的热舒适损失。

## 可追溯输出

- `source_data/cycle_optimal_points.csv`：全部循环结果与无效原因。
- `source_data/defrost_ticket_events.csv`：经验除霜门票分解。
- `source_data/candidate_cost_curves.parquet`：每个候选时刻的成本曲线。
- `source_data/clean_anchor_summary.csv`：clean anchor 与 COP。
- `source_data/cohort_audit.csv`：队列纳入审计。
- `source_data/empirical_policy_summary.csv`：经验门票和换算系数。
- `source_data/near_optimal_band_sensitivity.csv`：1%、2%、5%、10% regret 阈值对应的 envelope 宽度与连续段数。
- `figures/figure_1_empirical_optimal_defrost.*`：PNG/SVG/PDF/TIFF 主图。
- `figures/cycles/`：每个有效循环一张原始曲线与成本曲线图。
"""
    (output_root / "README_CN.md").write_text(summary, encoding="utf-8")
    figure_qa = f"""# Figure 1 QA contract

- Core conclusion: under the observed fixed-duration defrost policy, raw-data renewal cost identifies an empirical optimum before the observed defrost in many complete cycles, but the broad near-optimal regions limit point-label precision.
- Archetype: quantitative grid with one representative raw-data example and cross-cycle validation.
- Backend: Python/matplotlib only.
- Final size: 183 mm wide; 7.2 × 6.2 in working canvas.
- n definition: {len(valid_results)} complete cycles with an untruncated candidate domain; {len(valid_tickets)} valid defrost/recovery tickets.
- Center/spread: panel d reports all cycle values and a box plot (median, interquartile range, 1.5×IQR whiskers); no hypothesis test is claimed.
- Source data: `source_data/cycle_optimal_points.csv`, `source_data/candidate_cost_curves.parquet`, and `source_data/defrost_ticket_events.csv`.
- Editable exports: SVG text preserved; PDF uses TrueType fonts; TIFF is 600 dpi; PNG is 300 dpi.
- Image integrity: no microscopy or image manipulation in this figure; panel a uses unsmoothed original sensor points.
- Representative cycle: {representative_name}, selected as the interior-minimum cycle nearest the cohort median advance.
"""
    (output_root / "FIGURE_QA.md").write_text(figure_qa, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("report/raw_optimal_defrost"))
    args = parser.parse_args()
    analyze(args.dataset, args.output)


if __name__ == "__main__":
    main()
