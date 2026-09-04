"""Plot model-comparison and optional visual-concentration evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plots.cost import _shade_experiment_dates

__all__ = [
    "plot_model_figures",
    "plot_probability_curves",
    "plot_trigger_error_figures",
    "trigger_error_table",
    "two_of_three_trigger",
]

_MODALITY_COLORS = {
    "rgb": "#3775BA",
    "rgb_sensor": "#E28E2C",
    "rgb_sensor_slope": "#2E7D5B",
}
_MODALITY_MARKERS = {"rgb": "o", "rgb_sensor": "s", "rgb_sensor_slope": "^"}

_CAMERA_ORDER = [
    "top",
    "top_close",
    "left",
    "left_close",
    "front",
    "extreme",
    "top_pair",
    "left_pair",
    "all",
]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def _export(figure: plt.Figure, stem: Path, formats: tuple[str, ...] = ("png",)) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for figure_format in formats:
        figure.savefig(
            stem.with_suffix(f".{figure_format}"),
            dpi=300 if figure_format == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def _ordered(values: pd.Series, preferred: list[str]) -> list[str]:
    present = values.astype(str).drop_duplicates().tolist()
    return [value for value in preferred if value in present] + sorted(
        set(present).difference(preferred)
    )


def two_of_three_trigger(
    times: pd.Series | pd.DatetimeIndex,
    probabilities: pd.Series,
    threshold: float = 0.5,
) -> tuple[pd.Timestamp, pd.Series]:
    """Return the first frame where at least two of the latest three are positive."""
    positive = probabilities.ge(threshold)
    rolling = positive.rolling(3, min_periods=2).sum().ge(2)
    positions = np.flatnonzero(rolling.to_numpy())
    trigger = pd.NaT
    if len(positions):
        trigger = pd.Timestamp(
            times.iloc[positions[0]] if hasattr(times, "iloc") else times[positions[0]]
        )
    return trigger, rolling


def trigger_error_table(
    predictions: pd.DataFrame,
    policy: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compare sampled-frame trigger times with each selected Pareto knee."""
    selected = policy.loc[policy["selected"].fillna(False), ["cycle_name", "selected_time"]]
    selected = selected.drop_duplicates("cycle_name").set_index("cycle_name")["selected_time"]
    selected = pd.to_datetime(selected, errors="coerce", format="mixed")
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(
        ["experiment_id", "cycle_name", "camera", "modality"], sort=True
    ):
        experiment_id, cycle_name, camera, modality = keys
        if cycle_name not in selected:
            continue
        group = group.sort_values("image_time", kind="stable")
        times = pd.to_datetime(group["image_time"], errors="coerce", format="mixed")
        positive = group["decision_score"].ge(threshold)
        first_positive = times.loc[positive].iloc[0] if positive.any() else pd.NaT
        two_of_three, _ = two_of_three_trigger(times, group["decision_score"], threshold)
        optimum = pd.Timestamp(selected[cycle_name])
        for strategy, trigger in (
            ("first_positive", first_positive),
            ("two_of_three", two_of_three),
        ):
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "cycle_name": cycle_name,
                    "cycle_id": int(str(cycle_name).rsplit("_", 1)[-1]),
                    "camera": camera,
                    "modality": modality,
                    "strategy": strategy,
                    "selected_time": optimum,
                    "trigger_time": trigger,
                    "trigger_error_minutes": (
                        (trigger - optimum).total_seconds() / 60
                        if pd.notna(trigger)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_trigger_error_figures(
    *,
    predictions: pd.DataFrame,
    policy: pd.DataFrame,
    output: Path,
    source_output: Path,
    representation: str,
    head: str,
    continuous_stream: bool = False,
    figure_formats: tuple[str, ...] = ("png",),
) -> None:
    """Plot signed trigger-minus-Pareto timing errors for two control rules."""
    values = predictions.loc[
        predictions["representation"].eq(representation)
        & predictions["head"].eq(head)
        & predictions["modality"].isin(_MODALITY_COLORS)
    ].copy()
    errors = trigger_error_table(values, policy)
    errors["continuous_stream"] = continuous_stream
    source_output.mkdir(parents=True, exist_ok=True)
    errors.to_csv(source_output / "trigger_error_by_cycle.csv", index=False)
    cameras = [camera for camera in _CAMERA_ORDER[:6] if camera in set(errors["camera"])]
    finite = errors["trigger_error_minutes"].dropna().abs()
    limit = max(5.0, float(finite.max()) * 1.05)
    names = {
        "two_of_three": "Two positives within three frames",
        "first_positive": "First positive frame",
    }
    offsets = {"rgb": -0.16, "rgb_sensor": 0.0, "rgb_sensor_slope": 0.16}
    for strategy, strategy_name in names.items():
        subset = errors.loc[errors["strategy"].eq(strategy)]
        for camera in cameras:
            camera_rows = subset.loc[subset["camera"].eq(camera)]
            cycles = (
                camera_rows[["cycle_name", "cycle_id", "experiment_id"]]
                .drop_duplicates("cycle_name")
                .sort_values("cycle_id")
                .reset_index(drop=True)
            )
            positions = pd.Series(cycles.index, index=cycles["cycle_name"])
            figure, axis = plt.subplots(figsize=(max(7.2, 0.15 * len(cycles)), 3.4))
            _shade_experiment_dates(axis, cycles["experiment_id"].astype(str).tolist())
            for modality, rows in camera_rows.groupby("modality", sort=False):
                rows = rows.sort_values("cycle_id")
                axis.scatter(
                    rows["cycle_name"].map(positions) + offsets[str(modality)],
                    rows["trigger_error_minutes"],
                    color=_MODALITY_COLORS[str(modality)],
                    marker=_MODALITY_MARKERS[str(modality)],
                    s=18,
                    label=str(modality).replace("_", " + "),
                    zorder=3,
                )
            axis.axhline(0, color="#333333", lw=0.8)
            axis.set_ylim(-limit, limit)
            axis.grid(axis="y", color="#DDDDDD", lw=0.45)
            axis.set(
                xlabel="Cycle ID",
                ylabel="Trigger − Pareto knee (min)",
                xticks=np.arange(len(cycles)),
                xticklabels=cycles["cycle_id"],
            )
            axis.tick_params(axis="x", labelrotation=90, labelsize=6)
            handles = [
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker=_MODALITY_MARKERS[name],
                    ms=4,
                    linestyle="none",
                    label=name.replace("_", " + "),
                )
                for name, color in _MODALITY_COLORS.items()
            ]
            axis.legend(handles=handles, loc="upper right", ncol=3, fontsize=6)
            prefix = "Continuous stream" if continuous_stream else "Sampled-frame diagnostic"
            figure.suptitle(
                f"{prefix}: {camera.replace('_', ' ')} — {strategy_name}", fontsize=8
            )
            figure.text(
                0.5,
                0.008,
                "Positive = later than Pareto knee; negative = earlier. "
                "Missing points never triggered.",
                ha="center",
                fontsize=5.8,
            )
            figure.tight_layout(rect=(0, 0.06, 1, 0.93))
            _export(figure, output / "trigger_error" / strategy / camera, figure_formats)


def plot_probability_curves(
    *,
    predictions: pd.DataFrame,
    policy: pd.DataFrame,
    output: Path,
    source_output: Path,
    representation: str,
    head: str,
    window_minutes: float = 10,
    figure_formats: tuple[str, ...] = ("png",),
    continuous_stream: bool = False,
) -> None:
    """Plot held-out probabilities and the 2-of-3 trigger around each Pareto knee."""
    values = predictions.loc[
        predictions["representation"].eq(representation)
        & predictions["head"].eq(head)
        & predictions["modality"].isin(_MODALITY_COLORS)
    ].copy()
    values["image_time"] = pd.to_datetime(values["image_time"], errors="coerce", format="mixed")
    selected = policy.loc[policy["selected"].fillna(False)].copy()
    selected["selected_time"] = pd.to_datetime(
        selected["selected_time"], errors="coerce", format="mixed"
    )
    selected_times = selected.set_index("cycle_name")["selected_time"]
    cameras = [camera for camera in _CAMERA_ORDER[:6] if camera in set(values["camera"])]
    source_rows: list[pd.DataFrame] = []
    trigger_rows: list[dict[str, object]] = []
    cycle_output = output / "probability_cycles"
    for cycle_name, cycle in values.groupby("cycle_name", sort=True):
        if cycle_name not in selected_times:
            continue
        optimum = pd.Timestamp(selected_times[cycle_name])
        figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
        for axis, camera in zip(axes.flat, cameras, strict=False):
            camera_rows = cycle.loc[cycle["camera"].eq(camera)]
            for modality, rows in camera_rows.groupby("modality", sort=False):
                rows = rows.sort_values("image_time", kind="stable").copy()
                trigger, rolling = (
                    two_of_three_trigger(rows["image_time"], rows["decision_score"])
                    if continuous_stream
                    else (pd.NaT, pd.Series(False, index=rows.index))
                )
                rows["two_of_three"] = rolling.to_numpy()
                rows["minutes_from_selected"] = (
                    rows["image_time"] - optimum
                ).dt.total_seconds() / 60
                rows["trigger_time"] = trigger
                source_rows.append(rows)
                trigger_rows.append(
                    {
                        "cycle_name": cycle_name,
                        "camera": camera,
                        "modality": modality,
                        "selected_time": optimum,
                        "trigger_time": trigger,
                        "continuous_stream": continuous_stream,
                        "trigger_error_minutes": (
                            (trigger - optimum).total_seconds() / 60
                            if pd.notna(trigger)
                            else np.nan
                        ),
                    }
                )
                shown = rows.loc[
                    rows["minutes_from_selected"].between(-window_minutes, window_minutes)
                ]
                color = _MODALITY_COLORS[str(modality)]
                axis.plot(
                    shown["minutes_from_selected"],
                    shown["decision_score"],
                    color=color,
                    lw=1.0,
                    label=str(modality).replace("_", " + "),
                )
                if pd.notna(trigger):
                    trigger_minute = (trigger - optimum).total_seconds() / 60
                    if -window_minutes <= trigger_minute <= window_minutes:
                        axis.axvline(trigger_minute, color=color, lw=0.7, ls=":")
            axis.axhline(0.5, color="#555555", lw=0.7, ls="--")
            axis.axvline(0, color="#B64342", lw=1.0)
            axis.axvspan(-window_minutes, 0, color="#3775BA", alpha=0.04)
            axis.axvspan(0, window_minutes, color="#B64342", alpha=0.04)
            axis.set_title(camera.replace("_", " "))
            axis.grid(axis="y", color="#DDDDDD", lw=0.45)
        for axis in axes.flat[len(cameras) :]:
            axis.set_visible(False)
        axes[1, 0].set_xlabel("Minutes from Pareto knee")
        axes[1, 1].set_xlabel("Minutes from Pareto knee")
        axes[1, 2].set_xlabel("Minutes from Pareto knee")
        axes[0, 0].set_ylabel("Post-knee probability")
        axes[1, 0].set_ylabel("Post-knee probability")
        axes[0, 0].set_ylim(0, 1.02)
        handles = [
            Line2D([0], [0], color=color, lw=1.2, label=name.replace("_", " + "))
            for name, color in _MODALITY_COLORS.items()
        ]
        handles += [
            Line2D([0], [0], color="#B64342", lw=1.0, label="Pareto knee"),
            Line2D([0], [0], color="#555555", lw=0.7, ls="--", label="p = 0.5"),
        ]
        if continuous_stream:
            handles.append(
                Line2D([0], [0], color="#555555", lw=0.7, ls=":", label="2-of-3 trigger")
            )
        figure.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=5.7)
        figure.suptitle(
            f"{cycle_name}: held-out probabilities"
            + (" and 2-of-3 triggers" if continuous_stream else " (sampled cached features)"),
            fontsize=8,
        )
        figure.text(
            0.5,
            0.005,
            (
                "Question: can each held-out camera reproduce the C-H Pareto decision boundary?"
                if continuous_stream
                else "Sampled cached features: the 2-of-3 control trigger is not evaluated."
            ),
            ha="center",
            fontsize=5.8,
        )
        figure.tight_layout(rect=(0, 0.035, 1, 0.91))
        _export(figure, cycle_output / cycle_name, figure_formats)
    source_output.mkdir(parents=True, exist_ok=True)
    if source_rows:
        pd.concat(source_rows, ignore_index=True).to_parquet(
            source_output / "probability_curves.parquet", index=False
        )
    pd.DataFrame(trigger_rows).to_csv(source_output / "two_of_three_triggers.csv", index=False)


def _plot_model_comparison(
    summary: pd.DataFrame,
    output: Path,
    source_output: Path,
    figure_formats: tuple[str, ...],
) -> None:
    values = summary.copy()
    values["model_setting"] = values["head"].str.replace("_", " ") + " + " + values[
        "modality"
    ].str.replace("_", " ")
    if values["representation"].nunique() > 1:
        values["model_setting"] = (
            values["representation"].str.replace("_", " ")
            + " + "
            + values["model_setting"]
        )
    if "source" in values and values["source"].nunique() > 1:
        sources = values["source"].astype(str)
        run_names = sources.map(lambda source: Path(source).name)
        if run_names.nunique() != sources.nunique():
            run_names = sources
        duplicated = values.groupby("model_setting")["source"].transform("nunique").gt(1)
        values.loc[duplicated, "model_setting"] += " + run=" + run_names[duplicated]
    cameras = _ordered(values["camera"], _CAMERA_ORDER)
    settings = values["model_setting"].drop_duplicates().tolist()
    values["camera"] = pd.Categorical(values["camera"], cameras, ordered=True)
    values["model_setting"] = pd.Categorical(values["model_setting"], settings, ordered=True)
    values = values.sort_values(["camera", "model_setting"])
    values.to_csv(source_output / "figure_5_model_comparison.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"wspace": 0.32})
    colors = dict(zip(settings, plt.cm.tab10(np.linspace(0, 0.85, len(settings))), strict=True))
    offsets = np.linspace(-0.26, 0.26, len(settings))
    metrics = (
        ("macro_f1", "Macro-F1 (mean ± SD)"),
        ("balanced_accuracy", "Balanced accuracy (mean ± SD)"),
    )
    for axis, (metric, ylabel) in zip(axes, metrics, strict=True):
        for offset, setting in zip(offsets, settings, strict=True):
            rows = values.loc[values["model_setting"].eq(setting)]
            x = rows["camera"].cat.codes.to_numpy() + offset
            axis.errorbar(
                x,
                rows[f"{metric}_mean"],
                yerr=rows[f"{metric}_std"],
                fmt="o",
                ms=2.8,
                capsize=1.5,
                lw=0.7,
                color=colors[setting],
                label=setting,
            )
        axis.set_xticks(range(len(cameras)), cameras, rotation=45, ha="right")
        axis.set(ylabel=ylabel, xlabel="Camera")
        axis.set_ylim(0.0, 1.01)
    axes[0].legend(fontsize=5.4, ncol=2, loc="lower left")
    for label, axis in zip("ab", axes, strict=True):
        axis.text(
            -0.16,
            1.03,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
        )
    title = "Leave-one-experiment-out model and camera comparison"
    if values["representation"].astype(str).eq("dinov2").all():
        title += "\nExisting cached-feature cohort; error bars span held-out experiments"
    figure.suptitle(title, fontsize=8)
    _export(figure, output / "figure_5_model_camera_comparison", figure_formats)


def _plot_concentration(
    optima: pd.DataFrame,
    concentration: pd.DataFrame,
    output: Path,
    source_output: Path,
    figure_formats: tuple[str, ...],
) -> None:
    primary = optima.loc[optima["cohort_tier"].eq("A_observed_policy")].copy()
    concentration = concentration.copy()
    cameras = _ordered(concentration["camera_group"], _CAMERA_ORDER)
    concentration["camera_group"] = pd.Categorical(
        concentration["camera_group"], cameras, ordered=True
    )
    concentration = concentration.sort_values("camera_group")
    primary[["cycle_name", "minutes_from_stable", "minimum_location"]].to_csv(
        source_output / "figure_6_optimal_times.csv", index=False
    )
    concentration.to_csv(source_output / "figure_6_visual_concentration.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.34})
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.08, 0.08, len(primary))
    axes[0].scatter(jitter, primary["minutes_from_stable"], s=11, color="#4C78A8", alpha=0.65)
    axes[0].boxplot(
        primary["minutes_from_stable"],
        positions=[0],
        widths=0.3,
        patch_artist=True,
        boxprops={"facecolor": "#C6DBEF", "edgecolor": "#4C78A8"},
        medianprops={"color": "#D95F02"},
    )
    axes[0].set(
        xticks=[0],
        xticklabels=["Observed-policy\ncycles"],
        ylabel="Empirical optimum (min)",
    )
    dispersion = concentration["time_optimum_iqr_over_median"].iloc[0]
    axes[0].text(
        0.03,
        0.97,
        f"n={len(primary)}\nIQR/median={dispersion:.3f}",
        transform=axes[0].transAxes,
        va="top",
    )

    y = np.arange(len(concentration))
    axes[1].axvline(0, color="#555555", lw=0.8, ls="--")
    axes[1].errorbar(
        concentration["estimate"],
        y,
        xerr=[
            concentration["estimate"] - concentration["lower"],
            concentration["upper"] - concentration["estimate"],
        ],
        fmt="o",
        color="#D95F02",
        ms=3.5,
        capsize=2,
        lw=0.9,
    )
    axes[1].set_yticks(y, concentration["camera_group"].astype(str))
    axes[1].set(
        xlabel="Optimum-neighbourhood minus fixed-time\ncentroid dispersion",
        ylabel="Camera group",
    )
    axes[1].text(
        0.02,
        0.98,
        "<0 supports visual concentration",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
    )
    for label, axis in zip("ab", axes, strict=True):
        axis.text(
            -0.17,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
        )
    figure.suptitle(
        "Optimal timing varies, but single-frame visual concentration is not supported",
        fontsize=8,
    )
    _export(figure, output / "figure_6_time_visual_concentration", figure_formats)


def plot_model_figures(
    *,
    summary: pd.DataFrame,
    output: Path,
    source_output: Path,
    figure_formats: tuple[str, ...] = ("png",),
    optima: pd.DataFrame | None = None,
    concentration: pd.DataFrame | None = None,
) -> None:
    """Export Figure 5 and, when both evidence tables are supplied, Figure 6."""
    if (optima is None) != (concentration is None):
        raise ValueError("optima and concentration must be provided together")
    source_output.mkdir(parents=True, exist_ok=True)
    _plot_model_comparison(summary, output, source_output, figure_formats)
    if optima is not None and concentration is not None:
        _plot_concentration(optima, concentration, output, source_output, figure_formats)
