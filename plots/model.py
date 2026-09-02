"""Plot model-comparison and optional visual-concentration evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["plot_model_figures"]

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


def _plot_model_comparison(
    summary: pd.DataFrame,
    output: Path,
    source_output: Path,
    figure_formats: tuple[str, ...],
) -> None:
    values = summary.copy()
    values["model_setting"] = (
        values["representation"].astype(str)
        + " + "
        + values["head"].astype(str)
        + " + "
        + values["modality"].astype(str)
    )
    if "source" in values and values["source"].nunique() > 1:
        sources = values["source"].astype(str)
        run_names = sources.map(lambda source: Path(source).name)
        if run_names.nunique() != sources.nunique():
            run_names = sources
        values["model_setting"] += " + run=" + run_names
    cameras = _ordered(values["camera"], _CAMERA_ORDER)
    settings = values["model_setting"].drop_duplicates().tolist()
    values["camera"] = pd.Categorical(values["camera"], cameras, ordered=True)
    values["model_setting"] = pd.Categorical(values["model_setting"], settings, ordered=True)
    values = values.sort_values(["camera", "model_setting"])
    values.to_csv(source_output / "figure_5_model_comparison.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"wspace": 0.32})
    colors = dict(zip(settings, plt.cm.tab10(np.linspace(0, 0.85, len(settings))), strict=True))
    offsets = np.linspace(-0.26, 0.26, len(settings))
    for offset, setting in zip(offsets, settings, strict=True):
        rows = values.loc[values["model_setting"].eq(setting)]
        x = rows["camera"].cat.codes.to_numpy() + offset
        axes[0].errorbar(
            x,
            rows["balanced_accuracy_mean"],
            yerr=rows["balanced_accuracy_std"],
            fmt="o",
            ms=2.8,
            capsize=1.5,
            lw=0.7,
            color=colors[setting],
            label=setting.replace("_", " "),
        )
    axes[0].set_xticks(range(len(cameras)), cameras, rotation=45, ha="right")
    axes[0].set(ylabel="Balanced accuracy (mean ± SD)", xlabel="Camera")
    axes[0].set_ylim(0.0, 1.01)
    axes[0].legend(fontsize=5.4, ncol=2, loc="lower left")

    camera_colors = plt.cm.tab10(np.linspace(0, 0.85, len(cameras)))
    offsets = np.linspace(-0.3, 0.3, len(cameras))
    for offset, camera, color in zip(offsets, cameras, camera_colors, strict=True):
        rows = values.loc[values["camera"].eq(camera)]
        x = rows["model_setting"].cat.codes.to_numpy() + offset
        axes[1].plot(
            x,
            rows["balanced_accuracy_mean"],
            "o",
            ms=2.5,
            color=color,
            label=camera,
        )
    axes[1].set_xticks(range(len(settings)), [name.replace(" + ", "\n+ ") for name in settings])
    axes[1].set(ylabel="Balanced accuracy (mean)", xlabel="Model setting")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].legend(fontsize=4.8, ncol=3, loc="lower left")
    for label, axis in zip("ab", axes, strict=True):
        axis.text(
            -0.16,
            1.03,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
        )
    figure.suptitle("Leave-one-experiment-out model and camera comparison", fontsize=8)
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
