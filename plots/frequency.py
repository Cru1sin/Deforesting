"""Frequency changes and their COP response."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from plots.publication import _RGB_PANEL_FONT, save_publication

matplotlib.use("Agg")


def render_cop_stopping_audit(output):
    """Shared publication export of unchanged COP labels and control diagnostics."""
    with plt.rc_context(
        {
            "font.family": _RGB_PANEL_FONT,
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    ):
        checks = pd.read_csv(output / "crossing_peak_checks.csv")
        windows = pd.read_parquet(output / "marginal_cop_windows.parquet")
        traces = pd.read_csv(output / "temperature_frequency_source.csv")
        rises = pd.read_csv(output / "first_frequency_rises.csv")
        names = sorted(checks.cycle_name.unique())
        available = windows.window_seconds.unique()
        seconds = available[np.argmin(abs(available - 60))]
        fig, axes = plt.subplots(
            int(np.ceil(len(names) / 4)), 4, figsize=(16, 3 * np.ceil(len(names) / 4))
        )
        for ax, name in zip(axes.flat, names):
            g = windows.loc[windows.cycle_name.eq(name) & windows.window_seconds.eq(seconds)].copy()
            t = pd.to_datetime(g.candidate_defrost_time)
            x = (t - t.iloc[0]).dt.total_seconds() / 60
            for col, label, color in [
                ("cycle_cop", "周期有效COP", "#253D50"),
                ("instant_cop", f"{seconds:g}s供热/电耗", "#D98845"),
                ("corrected_marginal_cop", "含事件成本变化", "#2A788E"),
            ]:
                y = g[col].where(g.supported_pair)
                ax.plot(x, y, label=label, color=color, lw=0.8)
            peak = pd.Timestamp(g.peak_time.iloc[0])
            ax.axvline((peak - t.iloc[0]).total_seconds() / 60, color="#B64A50", ls="--", lw=0.8)
            ax.set(
                title=name.replace("frost_cycle_", "循环 "), xlabel="稳定制热后 / min", ylabel="COP"
            )
        for ax in list(axes.flat)[len(names) :]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=6)
        fig.suptitle("判据是否吻合标签峰值？当前 cross-fitted without_trunc 有效COP")
        save_publication(
            fig,
            output,
            "01_marginal_vs_cycle_cop",
            f"证据：{seconds:g} s完整累计能量差分；红线为原标签全局峰值；曲线横坐标为区间左端。\n"
            "结论：含事件电耗变化的判据检查公式自洽；固定成本交点不能单独验证真实最优除霜。",
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (definition, g) in enumerate(rises.groupby("definition")):
            g = g.sort_values("lag_to_peak_minutes")
            label = (
                "首个原生正增量" if definition.startswith("first_native") else "首个图示≥3 Hz升频段"
            )
            ax.scatter(g.lag_to_peak_minutes, np.arange(len(g)), label=label, s=25, alpha=0.7)
        ax.axvline(0, color="#B64A50", ls="--")
        ax.set(
            xlabel="COP峰值时刻 − 首次升频时刻 / min",
            ylabel="按滞后排序的循环",
            title="升频是否提供稳定的提前量？",
        )
        ax.legend()
        save_publication(
            fig,
            output,
            "02_first_rise_peak_lag",
            "证据：稳定制热起点后原生实际频率；两种既有升频定义；负值表示峰值已在升频前出现。\n"
            "结论：滞后分布需按升频定义解释；观察到升频不等于已识别结霜原因。",
        )
        fig, axes = plt.subplots(
            int(np.ceil(len(names) / 4)), 4, figsize=(16, 3 * np.ceil(len(names) / 4))
        )
        for ax, name in zip(axes.flat, names):
            g = traces.loc[traces.cycle_name.eq(name)]
            artist = ax.scatter(
                g.compressor_frequency,
                g.evaporating_temperature,
                c=g.progress,
                cmap="viridis",
                vmin=0,
                vmax=1,
                s=5,
                alpha=0.65,
            )
            ax.set(
                title=name.replace("frost_cycle_", "循环 "),
                xlabel="实际频率 / Hz",
                ylabel="蒸发温度 / ℃",
            )
        for ax in list(axes.flat)[len(names) :]:
            ax.set_visible(False)
        fig.colorbar(
            artist,
            ax=list(axes.flat)[: len(names)],
            label="稳定制热段相对时间",
            fraction=0.015,
            pad=0.01,
        )
        fig.subplots_adjust(top=0.95, bottom=0.09, hspace=0.5, wspace=0.35, right=0.92)
        fig.suptitle("同频率蒸发温度是否后移？原生30 s中位数")
        save_publication(
            fig,
            output,
            "03_temperature_frequency_time",
            "证据：每箱≥15个原生观测，未用插值填补；配对环境、水温和风机变化见源数据。\n"
            "结论：同频率温度下移是描述性证据；仍需排除环境、负荷、控制及传感器平台值的影响。",
            layout=False,
        )


def frequency_activity_segments(
    progress, frequency, valid, *, half_width_per_hz=0.005, direction="both"
):
    """Extend each observed change by ±0.5% of a cycle per Hz, for display only."""
    delta = np.diff(frequency)
    amplitude = np.abs(delta)
    changes = valid & (amplitude > 1e-9)
    sign = {"both": 0, "increase": 1, "decrease": -1}[direction]
    if sign:
        changes &= sign * delta > 1e-9
    midpoint = (progress[:-1] + progress[1:]) / 2
    onsets = progress[1:][changes]
    widths = amplitude[changes] * half_width_per_hz
    coloured = valid & (
        changes
        | np.any(np.abs(midpoint[:, None] - onsets[None, :]) <= widths[None, :] + 1e-12, axis=1)
    )
    if sign:
        coloured &= (
            sign * delta >= -1e-9
        )  # Never colour an opposite-direction step through display extension.
    return coloured


def render_cop_slope(
    output,
    *,
    rise_markers="first",
    activity_only=False,
    include_all=False,
    all_styles=False,
    cop_lower=1.5,
    activity_direction="both",
    curves=None,
    cohort=None,
    x_column="cycle_progress",
    y_column="cycle_cop",
    styles=None,
    source_label="Native Harmony + Slingshot",
    n_jobs=6,
):
    """Individual absolute COP curves coloured by actual compressor frequency."""
    from joblib import Parallel, delayed, parallel_config

    jobs = []
    output = Path(output)
    curves = pd.read_csv(output / "cop_curves.csv") if curves is None else curves.copy()
    cohort = pd.read_csv(output / "cohort.csv") if cohort is None else cohort
    rises = (
        pd.read_csv(output / "frequency_rise_markers.csv")
        if rise_markers != "none"
        else pd.DataFrame(columns=["cycle_name"])
    )
    shown = curves.loc[
        curves.cycle_cop_eligible.eq(True)
        & (curves.cycle_cop.ge(cop_lower) | (x_column == "frost_grade"))
        & np.isfinite(curves.compressor_frequency)
    ]
    frequency_limits = (
        np.floor(shown.compressor_frequency.min()),
        np.ceil(shown.compressor_frequency.max()),
    )
    cop_limits = (shown[y_column].min(), shown[y_column].max())
    groups = list(cohort.groupby("setpoint_c"))
    if include_all:
        groups.insert(0, (None, cohort))
    for temperature, group in groups:
        names = set(group.cycle_name)
        tag = "all_cycles" if temperature is None else f"{temperature:g}C"
        combinations = (
            styles
            if styles is not None
            else [(False, "both"), (True, "both"), (True, "increase"), (True, "decrease")]
            if all_styles
            else [(activity_only, activity_direction)]
        )
        for activity, direction in combinations:
            suffix = (
                {
                    "both": "frequency_changes",
                    "increase": "frequency_increases",
                    "decrease": "frequency_decreases",
                }[direction]
                if activity
                else "full_gradient"
            )
            name = (
                f"effective_cop_{tag}_{suffix}"
                if all_styles or direction != "both"
                else f"effective_cop_slope_{tag}"
            )
            if x_column == "frost_grade":
                name = f"frost_state_{'absolute' if y_column == 'cycle_cop' else 'normalized'}_cop_{tag}_{suffix}"
            jobs.append(
                delayed(_render_cop_slope_condition)(
                    output,
                    temperature,
                    group,
                    curves.loc[curves.cycle_name.isin(names)],
                    rises.loc[rises.cycle_name.isin(names)],
                    frequency_limits,
                    cop_limits,
                    rise_markers=rise_markers,
                    activity_only=activity,
                    cop_lower=cop_lower,
                    output_name=name,
                    activity_direction=direction,
                    x_column=x_column,
                    y_column=y_column,
                    source_label=source_label,
                )
            )
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        Parallel(verbose=5)(jobs)


def _render_cop_slope_condition(
    output,
    temperature,
    cohort,
    curves,
    rises,
    frequency_limits,
    cop_limits,
    *,
    rise_markers,
    activity_only,
    cop_lower,
    output_name,
    activity_direction,
    x_column="cycle_progress",
    y_column="cycle_cop",
    source_label="Native Harmony + Slingshot",
):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    peaks = curves.loc[curves.groupby("cycle_name").relative_cop.idxmax()]
    peak_progress = peaks.set_index("cycle_name").cycle_progress
    cmap = matplotlib.colormaps["turbo"]
    norm = Normalize(*frequency_limits)
    marker_sizes = {"3–<5 Hz": 9, "5–<10 Hz": 16, "≥10 Hz": 25}
    peak_activity = {}

    def gradient_line(axis, part, y):
        points = np.column_stack(
            [part[x_column] * (100 if x_column == "cycle_progress" else 1), part[y]]
        )
        frequency = part.compressor_frequency.to_numpy()
        segments = np.stack([points[:-1], points[1:]], axis=1)
        valid = (
            np.isfinite(segments).all(axis=(1, 2))
            & np.isfinite(frequency[:-1])
            & np.isfinite(frequency[1:])
        )
        coloured = valid
        if activity_only:
            coloured = frequency_activity_segments(
                part.cycle_progress.to_numpy(), frequency, valid, direction=activity_direction
            )
            grey = LineCollection(
                segments[valid & ~coloured], colors="#B6B8BE", linewidths=1.1, alpha=0.7
            )
            grey.set_capstyle("round")
            axis.add_collection(grey)
            name = part.cycle_name.iloc[0]
            peak_x = peak_progress[name]
            if np.any(
                coloured
                & (part.cycle_progress.to_numpy()[:-1] <= peak_x)
                & (part.cycle_progress.to_numpy()[1:] >= peak_x)
            ):
                peak_activity[name] = True
        lines = LineCollection(segments[coloured], cmap=cmap, norm=norm, linewidths=1.5, alpha=0.8)
        lines.set_array((frequency[:-1][coloured] + frequency[1:][coloured]) / 2)
        lines.set_capstyle("round")
        axis.add_collection(lines)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "legend.frameon": False,
        }
    ):
        fig, a = plt.subplots(figsize=(183 / 25.4, 105 / 25.4))
        for name in peak_progress.sort_values().index:
            g = curves.loc[curves.cycle_name.eq(name)].sort_values("cycle_progress")
            g.loc[
                ~(
                    g.cycle_cop_eligible.eq(True)
                    & (g.cycle_cop.ge(cop_lower) | (x_column == "frost_grade"))
                ),
                y_column,
            ] = np.nan
            gap = g.elapsed_minutes.diff().gt(0.502)
            if x_column == "frost_grade":
                gap |= pd.to_datetime(g.image_time).diff().gt(pd.Timedelta(seconds=45))
            for _, part in g.groupby(gap.cumsum()):
                gradient_line(a, part, y_column)
        peak_colors = cmap(norm(peaks.compressor_frequency))
        if activity_only:
            peak_colors[~peaks.cycle_name.map(peak_activity).fillna(False).to_numpy(dtype=bool)] = (
                matplotlib.colors.to_rgba("#8C909A")
            )
        a.scatter(
            peaks[x_column] * (100 if x_column == "cycle_progress" else 1),
            peaks[y_column],
            c=peak_colors,
            s=10,
            edgecolor="white",
            linewidth=0.35,
            zorder=4,
        )
        if rise_markers != "none":
            a.scatter(
                100 * rises.marker_progress,
                rises.pre_cop,
                s=rises.rise_size_class.map(marker_sizes),
                marker="^",
                facecolor="white",
                edgecolor="#252A3A",
                linewidth=0.65,
                zorder=5,
            )
        padding = 0.03 * (cop_limits[1] - cop_limits[0])
        a.set(
            xlim=(0, 99 if x_column == "frost_grade" else 100),
            ylim=(0.9, 1)
            if y_column == "relative_cop"
            else (cop_lower if y_column == "cycle_cop" else cop_limits[0], cop_limits[1] + padding),
            xlabel="Frost-state grade"
            if x_column == "frost_grade"
            else "Time from heating restart (%)",
            ylabel="Effective COP"
            if y_column == "cycle_cop"
            else (
                r"Normalized cycle COP ($COP/COP_{\max}$)"
                if y_column == "relative_cop"
                else "Normalized cycle COP (%)"
            ),
        )
        if y_column == "relative_cop":
            a.set_yticks(np.arange(0.9, 1.001, 0.02))
            a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        a.tick_params(width=0.65, length=3)
        fig.text(
            0.10,
            0.94,
            "All cycles" if temperature is None else f"{temperature:g} °C",
            fontsize=14,
            fontweight="bold",
        )
        fig.text(
            0.31 if temperature is None and x_column == "frost_grade" else 0.23,
            0.943,
            f"{len(cohort)} cycles • {cohort.experiment_id.nunique()} experiments",
            fontsize=9,
            color=".3",
        )
        if x_column == "frost_grade":
            style_label = (
                {
                    "both": "Frequency changes highlighted",
                    "increase": "Frequency increases highlighted",
                    "decrease": "Frequency decreases highlighted",
                }[activity_direction]
                if activity_only
                else "All frequency segments coloured"
            )
            fig.text(0.10, 0.897, f"{source_label} • {style_label}", fontsize=7, color=".4")
        fig.subplots_adjust(left=0.10, right=0.82, bottom=0.15, top=0.88)
        color_axis = fig.add_axes([0.855, 0.15, 0.018, 0.73])
        bar = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=color_axis,
            orientation="vertical",
            ticks=[frequency_limits[0], 60, 70, 80, 90, frequency_limits[1]],
        )
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=7, length=2, pad=2)
        bar.set_label("Frequency (Hz)", fontsize=8, labelpad=5)
        save_publication(
            fig,
            output,
            output_name,
            "",
            layout=False,
            dpi=600,
            formats=("png",) if x_column == "frost_grade" else ("png", "pdf", "svg"),
        )


def render_frequency_cop_grid(output, *, frequency_gradient=False):
    """Standalone six-category maps; each cell is one complete physical-time interval."""
    output = Path(output)
    cells = pd.read_csv(output / "frequency_cop_cells.csv")
    cohort = pd.read_csv(output / "cohort.csv")
    # Control changes carry the visual emphasis; unchanged frequency is neutral.
    colors = ["#E39A4B", "#C44B64", "#DEE2E8", "#BCC3CE", "#47A6AF", "#4C70B7"]
    if frequency_gradient:
        from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm

        cmap = LinearSegmentedColormap.from_list(
            "actual_frequency", ["#FFF1A8", "#FBC34A", "#F78636", "#E6493C", "#C51D42", "#790F35"]
        )
        lower = np.floor(cells[["frequency_start_hz", "frequency_end_hz"]].min().min())
        upper = np.ceil(cells[["frequency_start_hz", "frequency_end_hz"]].max().max())
        # Expand perceptual resolution where the observed high-frequency values cluster.
        norm = PowerNorm(gamma=2.5, vmin=lower, vmax=upper)
        cop_cmap = LinearSegmentedColormap.from_list(
            "absolute_effective_cop",
            ["#D4F5EF", "#64D9C6", "#21AEB6", "#2476C4", "#5946B5", "#38206B"],
        )
        cop_norm = Normalize(vmin=cells.cop_end_absolute.min(), vmax=cells.cop_end_absolute.max())
        stages = pd.read_csv(output / "frequency_rise_stages.csv")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
        }
    ):
        for temperature, group in cohort.groupby("setpoint_c"):
            names = sorted(group.cycle_name)
            data = cells.loc[cells.cycle_name.isin(names)]
            seconds = 60 * data.window_minutes.iloc[0]
            fig, ax = plt.subplots(figsize=(183 / 25.4, 135 / 25.4))
            fig.subplots_adjust(left=0.095, right=0.97, bottom=0.13, top=0.76)
            ax.set_facecolor("#F3F4F7")
            for row, name in enumerate(names):
                g = data.loc[data.cycle_name.eq(name)]
                if frequency_gradient:
                    starts = 100 * g.cell_start_progress
                    widths = 100 * (g.cell_end_progress - g.cell_start_progress)
                    spans = list(zip(starts + 0.03 * widths, 0.94 * widths))
                    ax.broken_barh(
                        spans,
                        (row + 0.06, 0.43),
                        facecolors=cmap(norm(g.frequency_end_hz)),
                        edgecolors="none",
                    )
                    ax.broken_barh(
                        spans,
                        (row + 0.52, 0.24),
                        facecolors=cop_cmap(cop_norm(g.cop_end_absolute)),
                        edgecolors="none",
                    )
                    for stage in stages.loc[stages.cycle_name.eq(name)].itertuples():
                        x0, x1 = 100 * stage.stage_start_progress, 100 * stage.stage_end_progress
                        ax.add_patch(
                            Rectangle(
                                (x0, row + 0.025),
                                x1 - x0,
                                0.755,
                                facecolor="none",
                                edgecolor="#39334D",
                                linewidth=0.65,
                                zorder=4,
                            )
                        )
                else:
                    for category, ink in enumerate(colors):
                        part = g.loc[g.category.eq(category)]
                        starts = 100 * part.cell_start_progress
                        widths = 100 * (part.cell_end_progress - part.cell_start_progress)
                        ax.broken_barh(
                            list(zip(starts + 0.03 * widths, 0.94 * widths)),
                            (row + 0.06, 0.7),
                            facecolors=ink,
                            edgecolors="none",
                        )
                near = g.loc[g.within_1pct_peak_entire_cell.eq(True)]
                starts = 100 * near.cell_start_progress
                widths = 100 * (near.cell_end_progress - near.cell_start_progress)
                ax.broken_barh(
                    list(zip(starts, widths)),
                    (row + 0.8, 0.14),
                    facecolors="#263547",
                    edgecolors="none",
                )
            ax.set(
                xlim=(0, 100),
                ylim=(len(names), 0),
                xlabel="Time from heating restart (% of observed cycle)",
                ylabel="Individual cycles",
                xticks=[0, 20, 40, 60, 80, 100],
            )
            ticks = sorted(set([0, *range(4, len(names), 5), len(names) - 1]))
            ax.set_yticks(np.array(ticks) + 0.5, labels=np.array(ticks) + 1)
            ax.tick_params(length=3, width=0.65)
            fig.text(0.095, 0.958, f"{temperature:g} °C", fontsize=14, fontweight="bold")
            fig.text(
                0.095,
                0.913,
                f"{len(names)} cycles · {group.experiment_id.nunique()} experiments · {seconds:g}-s cells",
                fontsize=9,
                color=".3",
            )
            if frequency_gradient:
                fig.text(0.095, 0.865, "Frequency above · absolute effective COP below", fontsize=8)
                fig.text(
                    0.095,
                    0.792,
                    "Boxes: first to last rise; holds included; falls and observation gaps split stages",
                    fontsize=7,
                    color=".35",
                )
                color_axis = fig.add_axes([0.68, 0.93, 0.28, 0.023])
                bar = fig.colorbar(
                    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=color_axis,
                    orientation="horizontal",
                )
                bar.set_ticks([lower, 70, 80, 90, 95, 100, upper])
                bar.set_label(
                    "Actual frequency (Hz; nonlinear colour scale)", fontsize=6.5, labelpad=2
                )
                bar.ax.tick_params(labelsize=7, length=2, pad=1)
                bar.outline.set_visible(False)
                cop_axis = fig.add_axes([0.68, 0.855, 0.28, 0.023])
                cop_bar = fig.colorbar(
                    matplotlib.cm.ScalarMappable(norm=cop_norm, cmap=cop_cmap),
                    cax=cop_axis,
                    orientation="horizontal",
                )
                cop_bar.set_label(
                    "Effective COP (absolute; linear colour scale)", fontsize=6.5, labelpad=2
                )
                cop_bar.ax.tick_params(labelsize=7, length=2, pad=1)
                cop_bar.outline.set_visible(False)
            else:
                fig.text(
                    0.095,
                    0.865,
                    "Actual compressor frequency × effective COP direction",
                    fontsize=8,
                )
                fig.text(
                    0.095,
                    0.817,
                    "Warm: frequency ↑ · cool: frequency ↓ · grey: unchanged",
                    fontsize=7,
                    color=".35",
                )
                legend = fig.add_axes([0.64, 0.815, 0.32, 0.16])
                legend.set_axis_off()
                for column, label in enumerate(["COP ↑", "COP ↓"]):
                    legend.text(
                        0.53 + 0.27 * column, 0.99, label, ha="center", va="top", fontsize=7
                    )
                for row, label in enumerate(["Frequency ↑", "Frequency =", "Frequency ↓"]):
                    y = 0.65 - 0.27 * row
                    legend.text(0.36, y + 0.06, label, ha="right", va="center", fontsize=7)
                    for column in range(2):
                        legend.add_patch(
                            Rectangle(
                                (0.43 + 0.27 * column, y),
                                0.2,
                                0.15,
                                facecolor=colors[2 * row + column],
                                edgecolor="none",
                            )
                        )
            fig.text(
                0.095,
                0.055,
                f"Dark strip beneath cells: COP stays within 1% of its cycle maximum throughout the {seconds:g}-s cell.",
                fontsize=6.5,
                color="#263547",
            )
            fig.text(
                0.095,
                0.025,
                "Only COP / maximum ≥ 0.6. Blank background: outside range, missing support, or unchanged COP.",
                fontsize=6.5,
                color=".35",
            )
            name = f"frequency_cop_grid_{temperature:g}C"
            save_publication(fig, output, name, "", layout=False, dpi=600)


def render_pre_frequency_evidence(output):
    """An illustrative temporal sequence, with instantaneous and cycle COP kept separate."""
    output = Path(output)
    data = pd.read_csv(output / "example_curve.csv")
    example = pd.read_csv(output / "example_selection.csv").iloc[0]
    all_data = data
    data = data.loc[data.cycle_name.eq(example.cycle_name)]
    panels = [
        ("compressor_frequency", "压缩机频率", "Hz"),
        ("instantaneous_cop", "实测瞬时 COP", "COP"),
        ("heating_capacity_processed", "供热功率与总输入功率", "kW"),
        ("evaporating_pressure_processed", "平滑蒸发压力", "MPa"),
        ("visual_frost_score", "图像霜态坐标", "0–1"),
        ("cycle_cop", "已保存的有效循环 COP", "COP"),
    ]
    with plt.rc_context(
        {
            "font.family": _RGB_PANEL_FONT,
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig, axes = plt.subplots(3, 2, figsize=(8, 7.5), sharex=True)
        for i, (ax, (field, title, unit)) in enumerate(zip(axes.flat, panels)):
            valid = data.dropna(subset=[field])
            ax.plot(
                valid.minutes_from_onset,
                valid[field],
                color="#789DA8",
                lw=1.2,
                marker="o" if field == "visual_frost_score" else None,
                markersize=2,
                label="供热功率" if field == "heating_capacity_processed" else None,
            )
            if field == "heating_capacity_processed":
                power = data.dropna(subset=["power_total"])
                ax.plot(
                    power.minutes_from_onset,
                    power.power_total,
                    color="#D3A781",
                    lw=1.2,
                    label="输入功率",
                )
                ax.legend(frameon=False, fontsize=7)
            ax.axvline(0, color="#444444", ls="--", lw=0.8)
            ax.axvspan(-10, -5, color="#BBA2C8", alpha=0.12)
            ax.axvspan(-5, 0, color="#789DA8", alpha=0.12)
            ax.set(title=f"{chr(97 + i)}  {title}", ylabel=unit, xlim=(-10, 5))
            ax.grid(alpha=0.12)
        for ax in axes[-1]:
            ax.set_xlabel("距升频开始的时间（min）")
        save_publication(
            fig,
            output,
            "升频前退化不等于立即除霜",
            f"{example.cycle_name}；两色背景对应升频前两个5 min窗口；虚线为升频开始。\n"
            "霜态为原生Harmony＋Slingshot坐标；循环COP含预测除霜成本，仅为估计。示例按所述现象筛选，不代表总体。",
            dpi=400,
        )
        selection = pd.read_csv(output / "example_selection.csv")
        fig, axes = plt.subplots(
            len(selection), 3, figsize=(9, 3 * len(selection)), squeeze=False, sharex=True
        )
        for row, sample in enumerate(selection.itertuples()):
            part = all_data.loc[all_data.cycle_name.eq(sample.cycle_name)]
            for col, field in enumerate(["compressor_frequency", "instantaneous_cop", "cycle_cop"]):
                ax = axes[row, col]
                valid = part.dropna(subset=[field])
                ax.plot(
                    valid.minutes_from_onset,
                    valid[field],
                    lw=1.3,
                    color="#789DA8" if sample.onset_side == "left" else "#D3A781",
                )
                ax.axvline(0, color="#444444", ls="--", lw=0.8)
                ax.set_xlabel("距升频开始时间（min）")
                ax.set_ylabel("Hz" if field == "compressor_frequency" else "COP")
                ax.set_title(["压缩机频率", "实测瞬时COP", "已保存的有效循环COP"][col])
                ax.grid(alpha=0.12)
                ax.set_xlim(-10, 5)
            axes[row, 0].text(
                0.02,
                0.97,
                f"{'峰值左侧' if sample.onset_side == 'left' else '峰值右侧'}\n{sample.cycle_name}",
                transform=axes[row, 0].transAxes,
                va="top",
                fontsize=7,
            )
        save_publication(
            fig,
            output,
            "峰值左右侧升频对照",
            "按各循环已保存的有效COP最高点划分左右；所示窗口均未跨峰值。\n"
            "瞬时COP不含Ridge成本；循环COP含预测成本。两个示例不能代替总体，最高点仅在离线分析时可知。",
            dpi=400,
        )


def render_frequency_cop_response(points, summary, curves, examples, output):
    """Compare observed synchronous changes and show both outcomes on shared axes."""
    with plt.rc_context({"font.family": _RGB_PANEL_FONT, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
        order = [
            ("increase", "any"),
            ("increase", "at_least_2Hz"),
            ("unchanged", "any"),
            ("decrease", "any"),
        ]
        for offset, seconds, color in [(-0.18, 30, "#0072B2"), (0.18, 120, "#D55E00")]:
            s = (
                summary.loc[summary.scope.eq("all_valid") & summary.window_seconds.eq(seconds)]
                .set_index(["direction", "threshold"])
                .loc[order]
            )
            bars = axes[0].bar(
                np.arange(4) + offset,
                100 * s.falling_fraction,
                width=0.34,
                color=color,
                label=f"{seconds} 秒窗口",
            )
            for bar, row in zip(bars, s.itertuples()):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{row.falling_fraction:.1%}\nn={row.n_windows}",
                    ha="center",
                    fontsize=8,
                )
        axes[0].set(
            xticks=range(4),
            xticklabels=["所有升频", "升频≥2 Hz", "频率不变", "降频"],
            ylim=(0, 105),
            ylabel="同一窗口内 COP 下降的比例 [%]",
            title="a  升频是否必然伴随下降？",
        )
        axes[0].legend(fontsize=8)
        p = points.loc[points.window_seconds.eq(30)]
        axes[1].hexbin(
            p.frequency_change_hz, p.cop_change, gridsize=45, mincnt=1, bins="log", cmap="Blues"
        )
        axes[1].axhline(0, color="#555555", lw=0.7)
        axes[1].axvline(0, color="#555555", lw=0.7)
        axes[1].set(
            xlabel="30 秒频率变化 [Hz]",
            ylabel="同一窗口 ΔCOP",
            title="b  全部有效窗口；深色代表更多观测",
        )
        falling = p.loc[p.direction.eq("increase") & p.falling]
        values = [falling.measured_cop_change, falling.ridge_cop_change, falling.cop_change]
        axes[2].boxplot(
            values,
            tick_labels=["实测 Q / E 项", "预测成本项", "总变化"],
            showfliers=False,
            widths=0.5,
        )
        axes[2].axhline(0, color="#555555", lw=0.7)
        axes[2].set(ylabel="ΔCOP / 30 s", title=f"c  升频且下降的 {len(falling)} 个窗口")
        for ax in axes:
            ax.grid(axis="y", alpha=0.15)
        fig.suptitle("升频与有效供热 COP：同步关系及下降来源", fontsize=14)
        save_publication(
            fig,
            output,
            "frequency_cop_population",
            "问题：升频是否同步降低 COP？证据：正式全事件三输入 Ridge、without_trunc；外推点同等参与。\n"
            "每个窗口两端及内部均须有效；窗口有重叠，比例仅描述关联，不证明因果。c 为精确代数分解，箱线不显示离群值；各项中位数不必相加。",
        )

        fig, axes = plt.subplots(4, len(examples), figsize=(14, 12), sharex="col", squeeze=False)
        for column, row in enumerate(examples.itertuples()):
            g = curves.loc[curves.selection.eq(row.selection)]
            x = g.minutes_from_event
            ax = axes[:, column]
            outcome = "COP 同步下降" if row.falling else "COP 没有同步下降"
            ax[0].plot(x, g.compressor_frequency, color="#333333")
            ax[0].set(
                ylabel="压缩机频率 [Hz]",
                title=f"{row.cycle_name}\n{outcome}：30 s 升频 {row.frequency_change_hz:g} Hz",
            )
            for field, label, color, style in [
                ("cycle_cop", "正式 COP", "#0072B2", "-"),
                ("fixed_cost_cop", "成本固定在起点值（诊断）", "#D55E00", "--"),
                ("zero_cost_cop", "无未来除霜成本（诊断）", "#777777", ":"),
            ]:
                ax[1].plot(x, g[field], label=label, color=color, ls=style)
            ax[1].set_ylabel("累计有效供热 COP")
            ax[1].legend(fontsize=8)
            outside = g.cycle_cop_uses_model_extrapolation.eq(True)
            ax[1].scatter(
                x[outside],
                g.loc[outside, "cycle_cop"],
                s=15,
                facecolors="none",
                edgecolors="#0072B2",
                label="外推",
            )
            ax[2].plot(x, 1000 * g.defrost_event_electricity_kwh, color="#7B2CBF")
            ax[2].scatter(
                x[outside],
                1000 * g.loc[outside, "defrost_event_electricity_kwh"],
                s=15,
                facecolors="none",
                edgecolors="#7B2CBF",
            )
            ax[2].set_ylabel("预测准备＋除霜电耗 [Wh]")
            w = points.loc[
                points.cycle_name.eq(row.cycle_name) & points.window_seconds.eq(30)
            ].copy()
            w["x"] = (
                pd.to_datetime(w.candidate_defrost_time) - row.candidate_defrost_time
            ).dt.total_seconds() / 60
            w = w.loc[w.x.between(x.min(), x.max())]
            for field, label, color in [
                ("measured_cop_change", "实测 Q / E 项", "#009E73"),
                ("ridge_cop_change", "预测成本项", "#D55E00"),
                ("cop_change", "总变化", "#0072B2"),
            ]:
                ax[3].plot(w.x, w[field], label=label, color=color)
            ax[3].axhline(0, color="#555555", lw=0.7)
            ax[3].set(xlabel="距所选升频窗口起点 [min]", ylabel="随后 30 秒的 ΔCOP")
            ax[3].legend(fontsize=8)
            for a in ax:
                a.axvspan(0, 0.5, color="#E6B84F", alpha=0.2)
                a.grid(alpha=0.15)
        fig.suptitle("同样升频，两种结果：黄色区间为判定“同步”的 30 秒", fontsize=14)
        save_publication(
            fig,
            output,
            "frequency_cop_examples",
            "问题：频率、成本和 COP 如何同步变化？证据：两类窗口各取频率变化/COP变化/成本贡献秩最接近组内中位数者；不按最大跳变选例。\n"
            "空心点是外推；两条诊断参照共享实测 Q、E，仅改变未来成本假设，不是实测真值。底行各分量在每个窗口精确相加。",
        )


def frequency_episode_folder(episode):
    amplitude = abs(float(episode["end_frequency_hz"]) - float(episode["start_frequency_hz"]))
    return "变化幅度＜=5HZ" if amplitude <= 5 else "变化幅度>=5HZ"


def render_frequency_episode(curve, raw, episode, origin, output, name):
    """Absolute COP and frequency throughout one staircase frequency episode."""
    with plt.rc_context({"font.family": _RGB_PANEL_FONT, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
        begin = (episode["start_time"] - origin).total_seconds() / 60
        end = (episode["end_time"] - origin).total_seconds() / 60
        anchor = (episode["anchor_time"] - origin).total_seconds() / 60
        axes[0].step(
            raw.cycle_minutes, raw.compressor_frequency, where="post", color="#333333", lw=1.4
        )
        axes[0].set(
            ylabel="压缩机频率 [Hz]",
            title=f"① 完整阶梯调频：{episode['start_frequency_hz']:g} → {episode['end_frequency_hz']:g} Hz；{episode['n_steps']} 次同向变化",
        )
        axes[1].plot(raw.cycle_minutes, raw.power_total, color="#555555", label="实测总电功率 [kW]")
        axes[1].plot(
            raw.cycle_minutes,
            raw.heating_capacity_processed,
            color="#009E73",
            label="60 s 平滑供热功率 [kW]",
        )
        energy_axis = axes[1].twinx()
        energy_axis.plot(
            curve.cycle_minutes,
            curve.pre_defrost_electricity_kwh,
            color="#7B2CBF",
            ls=":",
            label="累计实测耗电量 [kWh]（缺测时仅为已积分部分）",
        )
        energy_axis.plot(
            curve.cycle_minutes,
            curve.pre_defrost_heat_kwh,
            color="#D55E00",
            ls="--",
            label="累计实测有效供热量 [kWh]（平滑后积分；缺测时仅为已积分部分）",
        )
        energy_axis.set_ylabel("累计供热量／耗电量 [kWh]")
        axes[1].set(
            ylabel="功率 [kW]", title="② 总电功率、平滑供热功率与累计供热量／耗电量（右轴）"
        )
        axes[1].legend(loc="upper left", fontsize=8)
        energy_axis.legend(loc="upper right", fontsize=8)
        axes[2].plot(
            raw.cycle_minutes, raw.instantaneous_cop, lw=1.5, color="#0072B2", label="瞬时供热 COP"
        )
        axes[2].set(
            ylabel="瞬时供热 COP", title="③ 处理后供热功率 ÷ 实测总电功率；不含未来除霜成本"
        )
        axes[2].legend(fontsize=9)
        axes[3].plot(
            curve.cycle_minutes, curve.cop_current, color="#0072B2", lw=1.5, label="实时 Ridge 成本"
        )
        fixed_label = (
            f"固定成本 = {episode['fixed_event_cost_kwh'] * 1000:.2f} Wh（起点前 1 秒）"
            if np.isfinite(episode["fixed_event_cost_kwh"])
            else "起点前 1 秒成本缺失：无法绘制固定成本曲线"
        )
        axes[3].plot(
            curve.cycle_minutes,
            curve.cop_fixed_cost,
            color="#D55E00",
            lw=1.5,
            ls="--",
            label=fixed_label,
        )
        outside = curve.cycle_cop_uses_model_extrapolation.eq(True)
        axes[3].scatter(
            curve.loc[outside, "cycle_minutes"],
            curve.loc[outside, "cop_current"],
            s=12,
            facecolors="none",
            edgecolors="#0072B2",
            label="外推标记（参与计算）",
        )
        axes[3].set(
            ylabel="循环有效供热 COP", title="④ 同一实测累计供热／用电；只改变预测的准备＋除霜电耗"
        )
        if not curve.cop_current.notna().any():
            axes[3].text(
                0.5,
                0.45,
                "无有效循环 COP：累计积分或输入有效性未通过\n空白不代表 COP 为零；外推未被排除",
                transform=axes[3].transAxes,
                ha="center",
                fontsize=11,
            )
        axes[3].legend(fontsize=9)
        axes[4].plot(
            curve.cycle_minutes,
            curve.defrost_event_electricity_kwh * 1000,
            color="#0072B2",
            lw=1.5,
            label="实时 Ridge 预测电耗",
        )
        axes[4].plot(
            curve.cycle_minutes,
            curve.fixed_event_cost_kwh * 1000,
            color="#D55E00",
            lw=1.5,
            ls="--",
            label=fixed_label,
        )
        axes[4].scatter(
            curve.loc[outside, "cycle_minutes"],
            curve.loc[outside, "defrost_event_electricity_kwh"] * 1000,
            s=12,
            facecolors="none",
            edgecolors="#0072B2",
            label="外推标记（参与计算）",
        )
        axes[4].set(
            ylabel="预测准备＋除霜电耗 [Wh]",
            xlabel="从循环开始经过的时间 [min]",
            title="⑤ 实时预测电耗与调频前 1 秒固定电耗",
        )
        axes[4].legend(fontsize=9)
        for ax in axes:
            if end > begin:
                ax.axvspan(begin, end, color="#E6B84F", alpha=0.15)
            ax.axvline(anchor, color="#666666", ls=":", lw=0.8)
            ax.grid(alpha=0.2)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        direction = "升频" if episode["direction"] == "increase" else "降频"
        fig.suptitle(
            f"{episode['cycle_name']} · 第 {episode['episode']} 段{direction} · 循环起点 {origin}",
            fontsize=13,
        )
        difference = episode["maximum_absolute_cop_difference"]
        conclusion = (
            f"图示范围内两种循环 COP 最大绝对差 {difference:.4f}。"
            if np.isfinite(difference)
            else "本段缺少可配对的循环 COP；不推断差异。"
        )
        folder = frequency_episode_folder(episode)
        (output / folder).mkdir(parents=True, exist_ok=True)
        save_publication(
            fig,
            output / folder,
            name,
            "问题：整段调频时 Ridge 成本变化如何影响 COP？黄色为从首次到末次同向调频，中间平台保留；反向调频另图。\n"
            "证据：直接读取 Dataset 已发布的累计量、Ridge 电耗与 COP；固定对照仅补起点单值；外推不剔除。\n"
            + conclusion
            + " 固定成本仅作诊断，不能据此证明预测成本的物理真值。",
            formats=("png",),
        )
        return str(Path(folder) / (name + ".png"))
