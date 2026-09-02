#!/usr/bin/env python3
"""Compare measured water-side and controller-estimated heat during recovery."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataloader.loader import DatasetLoader
from dataloader.metadata import following_cycle_names
from frost_analysis.cost.core import (
    integrate_energy_curve_kwh,
    integrate_energy_kwh,
    water_side_heating_kw,
)
from frost_analysis.cost.selected import MINIMUM_INTEGRATION_COVERAGE, RECOVERY

DATE_BANDS = ("#EAF2F8", "#FFF3E6")
WATER_COLOR = "#2166AC"
UNIT_COLOR = "#D97706"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)


def build_recovery_heat_comparison(dataset: Path, tickets: Path) -> pd.DataFrame:
    """Integrate both heat-capacity channels over the current fixed V2 window."""
    loader = DatasetLoader(dataset)
    catalog = loader.list_cycles()
    records = catalog.set_index("cycle_name")
    following = following_cycle_names(catalog)
    valid = pd.read_csv(tickets).loc[lambda frame: frame["valid"].fillna(False)]
    rows = []
    columns = [
        "timestamp",
        "water_flow",
        "water_in_temperature",
        "water_out_temperature",
        "water_temperature_setpoint",
        "power_total",
        "heating_capacity",
    ]
    for cycle_name in valid["cycle_name"].astype(str):
        recovery_cycle = following[cycle_name]
        record = records.loc[cycle_name]
        recovery_record = records.loc[recovery_cycle]
        start = pd.Timestamp(recovery_record["heating_start"])
        frame = loader.load_cycle_original(recovery_cycle, columns=columns)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        first_30s = frame.loc[
            frame["timestamp"].between(start, start + pd.Timedelta(seconds=30))
        ]
        measured_setpoint = pd.to_numeric(
            first_30s["water_temperature_setpoint"], errors="coerce"
        ).median()
        setpoint = min(RECOVERY, key=lambda value: abs(value - measured_setpoint))
        duration_minutes = RECOVERY[setpoint][0]
        end = start + pd.Timedelta(minutes=duration_minutes)
        window = frame.loc[frame["timestamp"].between(start, end)].copy()
        water_heat, water_coverage = integrate_energy_kwh(
            window["timestamp"], water_side_heating_kw(window)
        )
        unit_heat, unit_coverage = integrate_energy_kwh(
            window["timestamp"], window["heating_capacity"]
        )
        electricity, electricity_coverage = integrate_energy_kwh(
            window["timestamp"], window["power_total"]
        )
        rows.append(
            {
                "cycle_name": cycle_name,
                "cycle_id": int(cycle_name.rsplit("_", 1)[-1]),
                "recovery_cycle_name": recovery_cycle,
                "experiment_id": record["experiment_id"],
                "experiment_date": str(record["experiment_date"])[:10],
                "defrost_start": pd.Timestamp(record["defrost_start"]),
                "defrost_end": pd.Timestamp(record["defrost_end"]),
                "recovery_start": start,
                "recovery_end": end,
                "stable_end": end + pd.Timedelta(minutes=5),
                "setpoint_c": setpoint,
                "duration_minutes": duration_minutes,
                "electricity_kwh": electricity,
                "water_heat_kwh": water_heat,
                "unit_heat_kwh": unit_heat,
                "water_cop": water_heat / electricity,
                "unit_cop": unit_heat / electricity,
                "water_to_unit_ratio": water_heat / unit_heat,
                "electricity_coverage": electricity_coverage,
                "water_heat_coverage": water_coverage,
                "unit_heat_coverage": unit_coverage,
            }
        )
    return pd.DataFrame(rows).sort_values("cycle_id").reset_index(drop=True)


def build_heating_stage_heat_comparison(dataset: Path) -> pd.DataFrame:
    """Integrate both heat channels over the two-stage heating interval."""
    loader = DatasetLoader(dataset)
    catalog = loader.list_cycles().dropna(
        subset=["heating_start", "defrost_preparation_start"]
    )
    rows = []
    columns = [
        "timestamp",
        "water_flow",
        "water_in_temperature",
        "water_out_temperature",
        "heating_capacity",
    ]
    for record in catalog.itertuples(index=False):
        start = pd.Timestamp(record.heating_start)
        end = pd.Timestamp(record.defrost_preparation_start)
        if end <= start:
            continue
        frame = loader.load_cycle_original(record.cycle_name, columns=columns)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        window = frame.loc[
            frame["timestamp"].ge(start) & frame["timestamp"].lt(end)
        ]
        water_heat, water_coverage = integrate_energy_kwh(
            window["timestamp"], water_side_heating_kw(window)
        )
        unit_heat, unit_coverage = integrate_energy_kwh(
            window["timestamp"], window["heating_capacity"]
        )
        stable_start = pd.Timestamp(record.stable_heating_start)
        rows.append(
            {
                "cycle_name": record.cycle_name,
                "cycle_id": int(record.cycle_name.rsplit("_", 1)[-1]),
                "experiment_id": record.experiment_id,
                "experiment_date": str(record.experiment_date)[:10],
                "start": start,
                "end": end,
                "heating_start": start,
                "stable_heating_start": stable_start,
                "defrost_preparation_start": end,
                "recovery_duration_minutes": (stable_start - start).total_seconds()
                / 60,
                "stable_duration_minutes": (end - stable_start).total_seconds() / 60,
                "duration_minutes": (end - start).total_seconds() / 60,
                "water_heat_kwh": water_heat,
                "unit_heat_kwh": unit_heat,
                "water_to_unit_ratio": water_heat / unit_heat,
                "water_heat_coverage": water_coverage,
                "unit_heat_coverage": unit_coverage,
            }
        )
    result = pd.DataFrame(rows).sort_values("cycle_id").reset_index(drop=True)
    result["eligible"] = result[
        ["water_heat_coverage", "unit_heat_coverage"]
    ].min(axis=1).ge(MINIMUM_INTEGRATION_COVERAGE)
    return result


def _plot_heat_comparison(
    events: pd.DataFrame,
    output: Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    show_relative_difference: bool = False,
) -> None:
    source = events.sort_values("cycle_id").copy()
    groups = list(source.groupby("experiment_date", sort=False))
    starts = [group["cycle_id"].min() for _, group in groups]
    ends = [group["cycle_id"].max() for _, group in groups]
    boundaries = [starts[0] - 0.75]
    boundaries += [(ends[i] + starts[i + 1]) / 2 for i in range(len(groups) - 1)]
    boundaries += [ends[-1] + 0.75]

    if show_relative_difference:
        figure, (axis, difference_axis) = plt.subplots(
            2,
            1,
            figsize=(183 / 25.4, 108 / 25.4),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        figure, axis = plt.subplots(figsize=(183 / 25.4, 82 / 25.4))
        difference_axis = None
    for index, ((date, group), left, right) in enumerate(
        zip(groups, boundaries[:-1], boundaries[1:], strict=True)
    ):
        eligible = group.get("eligible", pd.Series(True, index=group.index))
        for current_axis in (axis, difference_axis):
            if current_axis is None:
                continue
            current_axis.axvspan(left, right, color=DATE_BANDS[index % 2], zorder=0)
            if index:
                current_axis.axvline(left, color="#AEB7C2", linewidth=0.6, zorder=1)
        axis.plot(
            group["cycle_id"],
            group["water_heat_kwh"].where(eligible),
            color=WATER_COLOR,
            marker="o",
            markersize=3,
            linewidth=1,
            label="Water-side measurement" if index == 0 else None,
            zorder=3,
        )
        if difference_axis is not None:
            difference_axis.plot(
                group["cycle_id"],
                (100 * (group["water_heat_kwh"] / group["unit_heat_kwh"] - 1)).where(
                    eligible
                ),
                color="#606060",
                marker="o",
                markersize=2.5,
                linewidth=0.8,
                zorder=3,
            )
        axis.plot(
            group["cycle_id"],
            group["unit_heat_kwh"].where(eligible),
            color=UNIT_COLOR,
            marker="s",
            markersize=2.8,
            linewidth=1,
            label="Refrigerant-side estimate" if index == 0 else None,
            zorder=3,
        )
        axis.text(
            (left + right) / 2,
            1.015,
            pd.Timestamp(date).strftime("%m-%d"),
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=5.5,
            color="#59636E",
        )

    tick_start = 5 * int(np.floor(source["cycle_id"].min() / 5))
    tick_end = 5 * int(np.ceil(source["cycle_id"].max() / 5))
    axis.set(
        xlim=(boundaries[0], boundaries[-1]),
        xticks=np.arange(tick_start, tick_end + 1, 5),
        xlabel=xlabel,
        ylabel=ylabel,
    )
    axis.set_title(title, loc="left", y=1.10, pad=0)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.45)
    axis.legend(frameon=False, loc="upper left", ncol=2)
    if difference_axis is not None:
        difference_axis.axhline(0, color="#767676", linewidth=0.7)
        difference_axis.set(
            xlabel=xlabel,
            ylabel="Water − refrigerant [%]",
        )
        difference_axis.grid(axis="y", color="#D8D8D8", linewidth=0.45)
        axis.set_xlabel("")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_recovery_heat_comparison(events: pd.DataFrame, output: Path) -> None:
    """Plot paired recovery heat curves against defrost cycle ID."""
    _plot_heat_comparison(
        events,
        output,
        xlabel="Defrost cycle ID (recovery measured in the following cycle)",
        ylabel="Recovery heat [kWh]",
        title="Unit estimates exceed water-side heat during post-defrost recovery",
    )


def plot_heating_stage_heat_comparison(events: pd.DataFrame, output: Path) -> None:
    """Compare heating-stage water- and refrigerant-side cumulative heat."""
    _plot_heat_comparison(
        events,
        output,
        xlabel="Cycle ID",
        ylabel="Heating-stage heat [kWh]",
        title="Water- and refrigerant-side heat over the complete heating stage (≥95% coverage)",
        show_relative_difference=True,
    )


def plot_defrost_recovery_cycle_curve(
    frame: pd.DataFrame, event: pd.Series, output: Path
) -> None:
    """Plot heat capacity and total power through defrost and adjacent recovery."""
    source = frame.copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="coerce")
    start = pd.Timestamp(event["defrost_start"])
    recovery_start = pd.Timestamp(event["recovery_start"])
    recovery_end = pd.Timestamp(event["recovery_end"])
    stable_end = pd.Timestamp(event["stable_end"])
    source = source.loc[source["timestamp"].between(start, stable_end)].copy()
    source["stage"] = np.select(
        [source["timestamp"].lt(recovery_start), source["timestamp"].lt(recovery_end)],
        ["defrost", "recovery"],
        default="stable",
    )
    source["minutes"] = (source["timestamp"] - start).dt.total_seconds() / 60
    source["water_heat_kw"] = water_side_heating_kw(source)
    source["unit_heat_kw"] = pd.to_numeric(
        source["heating_capacity"], errors="coerce"
    ).where(source["stage"].ne("defrost"))
    source["power_total"] = pd.to_numeric(source["power_total"], errors="coerce")
    positive_power = source["power_total"].where(source["power_total"].gt(0))
    source["water_cop_instant"] = source["water_heat_kw"] / positive_power
    source["unit_cop_instant"] = source["unit_heat_kw"] / positive_power
    power_energy = integrate_energy_curve_kwh(
        source["timestamp"], source["power_total"], source["timestamp"]
    )["energy_kwh"]
    water_energy = integrate_energy_curve_kwh(
        source["timestamp"], source["water_heat_kw"], source["timestamp"]
    )["energy_kwh"]
    source["water_cop_integrated"] = water_energy.div(power_energy.replace(0, np.nan))
    unit = source.loc[source["stage"].ne("defrost")]
    unit_power_energy = integrate_energy_curve_kwh(
        unit["timestamp"], unit["power_total"], unit["timestamp"]
    )["energy_kwh"]
    unit_heat_energy = integrate_energy_curve_kwh(
        unit["timestamp"], unit["unit_heat_kw"], unit["timestamp"]
    )["energy_kwh"]
    source["unit_cop_integrated"] = np.nan
    source.loc[unit.index, "unit_cop_integrated"] = unit_heat_energy.div(
        unit_power_energy.replace(0, np.nan)
    ).to_numpy()
    recovery_minute = (recovery_start - start).total_seconds() / 60
    recovery_end_minute = (recovery_end - start).total_seconds() / 60
    stable_end_minute = (stable_end - start).total_seconds() / 60

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(150 / 25.4, 112 / 25.4),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [2, 1]},
    )
    rate_axis, cop_axis = axes
    for axis in axes:
        axis.axvspan(0, recovery_minute, color="#FCE8E6", zorder=0)
        axis.axvspan(
            recovery_minute, recovery_end_minute, color="#EAF2F8", zorder=0
        )
        axis.axvspan(
            recovery_end_minute, stable_end_minute, color="#EDF5EA", zorder=0
        )
        axis.axvline(recovery_minute, color="#8A8A8A", lw=0.7, zorder=1)
        axis.axvline(recovery_end_minute, color="#8A8A8A", lw=0.7, zorder=1)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.45)
    rate_axis.text(
        recovery_minute / 2,
        0.97,
        "Defrost",
        transform=rate_axis.get_xaxis_transform(),
        ha="center",
        va="top",
        color="#7A5550",
    )
    rate_axis.text(
        (recovery_minute + recovery_end_minute) / 2,
        0.97,
        "Recovery",
        transform=rate_axis.get_xaxis_transform(),
        ha="center",
        va="top",
        color="#496A7A",
    )
    rate_axis.text(
        (recovery_end_minute + stable_end_minute) / 2,
        0.97,
        "Stable heating (5 min)",
        transform=rate_axis.get_xaxis_transform(),
        ha="center",
        va="top",
        color="#4E7048",
    )
    rate_axis.plot(
        source["minutes"],
        source["water_heat_kw"],
        color=WATER_COLOR,
        lw=0.9,
        label="Water-side measurement",
    )
    rate_axis.plot(
        source["minutes"],
        source["unit_heat_kw"],
        color=UNIT_COLOR,
        lw=0.9,
        label="Unit controller estimate",
    )
    rate_axis.plot(
        source["minutes"],
        source["power_total"],
        color="#3F3F3F",
        lw=0.9,
        label="Total electric power",
    )
    rate_axis.axhline(0, color="#777777", lw=0.6)
    rate_axis.set(ylabel="Heat / electric power [kW]")
    rate_axis.legend(frameon=False, loc="best")
    cop_axis.plot(
        source["minutes"],
        source["water_cop_instant"],
        color=WATER_COLOR,
        alpha=0.35,
        lw=0.55,
        label="Instantaneous water-side COP",
    )
    cop_axis.plot(
        source["minutes"],
        source["unit_cop_instant"],
        color=UNIT_COLOR,
        alpha=0.35,
        lw=0.55,
        label="Instantaneous unit COP",
    )
    cop_axis.plot(
        source["minutes"],
        source["water_cop_integrated"],
        color=WATER_COLOR,
        lw=1.2,
        ls="--",
        label="Integrated water COP (from defrost)",
    )
    cop_axis.plot(
        source["minutes"],
        source["unit_cop_integrated"],
        color=UNIT_COLOR,
        lw=1.2,
        ls="--",
        label="Integrated unit COP (from recovery)",
    )
    cop_axis.axhline(0, color="#777777", lw=0.6)
    cop_axis.set(
        xlim=(0, stable_end_minute),
        xlabel="Time from defrost start [min]",
        ylabel="COP [–]",
    )
    cop_axis.legend(frameon=False, loc="best", ncol=2)
    rate_axis.set_title(
        f"{event['cycle_name']} · {event['experiment_date']} · "
        f"$T_s$={event['setpoint_c']:.0f} °C\n"
        f"Recovery: $Q_w$={event['water_heat_kwh']:.3f} kWh, "
        f"$Q_u$={event['unit_heat_kwh']:.3f} kWh · "
        f"$COP_w$={event['water_cop']:.2f}, $COP_u$={event['unit_cop']:.2f}",
        loc="left",
        pad=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--tickets",
        type=Path,
        default=Path(
            "output/test/成本函数/ED模型/经验经济窗口/源数据/defrost_ticket_events.csv"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/test/成本函数/QR模型")
    )
    parser.add_argument(
        "--heating-output", type=Path, default=Path("output/test/成本函数/QH口径")
    )
    args = parser.parse_args()
    events = build_recovery_heat_comparison(args.dataset, args.tickets)
    args.output.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output / "recovery_heat_comparison.csv", index=False)
    plot_recovery_heat_comparison(
        events, args.output / "recovery_heat_comparison.png"
    )
    complete_heating = build_heating_stage_heat_comparison(args.dataset)
    args.heating_output.mkdir(parents=True, exist_ok=True)
    complete_heating.to_csv(
        args.heating_output / "heating_stage_heat_comparison.csv", index=False
    )
    plot_heating_stage_heat_comparison(
        complete_heating, args.heating_output / "heating_stage_heat_comparison.png"
    )
    loader = DatasetLoader(args.dataset)
    curve_columns = [
        "timestamp",
        "water_flow",
        "water_in_temperature",
        "water_out_temperature",
        "heating_capacity",
        "power_total",
    ]
    for _, event in events.iterrows():
        defrost = loader.load_cycle_original(
            str(event["cycle_name"]), columns=curve_columns
        )
        recovery = loader.load_cycle_original(
            str(event["recovery_cycle_name"]), columns=curve_columns
        )
        plot_defrost_recovery_cycle_curve(
            pd.concat([defrost, recovery], ignore_index=True),
            event,
            args.output
            / "cycles"
            / f"{event['cycle_name']}_defrost_recovery_stable.png",
        )


if __name__ == "__main__":
    main()
