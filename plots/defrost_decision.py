"""Effective heating COP decisions, using the shared publication renderer."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from plots.cop import COP_COLOR, OPTIMAL_COLOR, RB_COLOR
from plots.publication import match_decision_rgb_images, render_decision_publication


def render_current_decision_cycle(decisions, loader, output_path, *, images=None):
    name = decisions.cycle_name.iloc[0]
    selected = decisions.loc[decisions.is_selected.fillna(False)]
    optimum = selected.candidate_defrost_time.iloc[0] if len(selected) else pd.NaT
    matches = match_decision_rgb_images(
        loader.load_image_metadata(name),
        loader.load_cycle_images(name) if images is None else images,
        {"rb": decisions.t_RB.iloc[0], "optimal": optimum},
    )
    render_decision_publication(
        loader.load_cycle(name),
        loader.get_cycle_record(name),
        decisions,
        {row.target_type: row._asdict() for row in matches.itertuples(index=False)},
        output_path,
        optimal_label="Effective heating COP maximum",
        parallel_curve=decisions,
    )


def render_current_decision_figures(decisions, loader, output, n_jobs=6, fetch_cloud_images=False):
    from select_defrost_time import summarize_decisions

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if fetch_cloud_images:
        from dataset_tools.cloud_images import materialize_image_members

        requests = {}
        for name, curve in decisions.groupby("cycle_name", sort=True):
            selected = curve.loc[curve.is_selected.fillna(False)]
            matches = match_decision_rgb_images(
                loader.load_image_metadata(name),
                loader.load_cycle_images(name),
                {
                    "rb": curve.t_RB.iloc[0],
                    "optimal": selected.candidate_defrost_time.iloc[0] if len(selected) else pd.NaT,
                },
            )
            requests[name] = sorted(
                {
                    f"front/{row.file_name}"
                    for row in matches.itertuples()
                    if row.status == "physical_image_missing"
                }
            )
        with materialize_image_members(loader.dataset_root, requests, n_jobs=n_jobs) as ready:
            for name in ready:
                print(f"[decision images] {name}", flush=True)
    summary = summarize_decisions(decisions)
    summary.to_csv(output / "timing_vs_RB.csv", index=False)
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    x = np.arange(len(summary))
    axes[0].scatter(
        x, summary.COP_eff_selected, color=OPTIMAL_COLOR, s=12, label="Effective COP maximum"
    )
    axes[0].scatter(x, summary.COP_eff_RB, facecolors="none", edgecolors=RB_COLOR, s=16, label="RB")
    axes[0].set_ylabel("Effective COP")
    axes[0].legend(frameon=False)
    axes[1].scatter(x, summary.COP_relative_change_pct, s=12, color=COP_COLOR)
    axes[1].set(
        ylabel="Change vs RB [%]",
        xlabel="Cycle",
        xticks=x,
        xticklabels=summary.cycle_name.str.rsplit("_", n=1).str[-1],
    )
    axes[1].tick_params(axis="x", labelrotation=90, labelsize=5)
    figure.tight_layout()
    _save_png(figure, output / "effective_COP_vs_RB.png")
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        Parallel()(
            delayed(render_current_decision_cycle)(
                curve,
                loader,
                output
                / "cost_function_effective_cop_cycle"
                / f"cycle_{int(name.rsplit('_', 1)[-1]):03d}_publication.png",
            )
            for name, curve in decisions.groupby("cycle_name", sort=True)
        )


def _save_png(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_preparation_comparison(paired, output):
    timing = "time_difference_minutes" in paired
    if not timing:
        paired = paired.rename(
            columns={f"COP_eff_{mode}": f"COP_eff_selected_{mode}" for mode in ("include", "zero")}
        )
        paired = paired.assign(common_supported=paired.COP_difference.notna())
    figure, axes = plt.subplots(
        3 if timing else 2, 1, figsize=(12, 7 if timing else 5), sharex=True
    )
    x = np.arange(len(paired))
    axes[0].scatter(x, paired.COP_difference, s=12, color=COP_COLOR)
    axes[0].set_ylabel(("Optimum COP" if timing else "COP") + " include − zero [-]")
    if timing:
        axes[1].scatter(x, paired.time_difference_minutes, s=12, color=COP_COLOR)
        axes[1].set_ylabel("Optimum include − zero [min]")
    for level, mode in enumerate(("include", "zero")):
        supported = paired[f"COP_eff_selected_{mode}"].notna()
        axes[-1].scatter(
            x[supported],
            np.full(supported.sum(), level),
            s=14,
            label=f"{mode}: {supported.sum()}/{len(paired)}",
        )
        axes[-1].scatter(
            x[~supported], np.full((~supported).sum(), level), s=14, marker="x", color="0.65"
        )
    axes[-1].set(
        yticks=[0, 1],
        yticklabels=["include", "zero"],
        xlabel="Cycle",
        ylim=(-0.5, 1.5),
        xticks=x,
        xticklabels=paired.cycle_name.str.rsplit("_", n=1).str[-1],
    )
    axes[-1].tick_params(axis="x", labelrotation=90, labelsize=5)
    axes[-1].legend(frameon=False, fontsize=8)
    for axis in axes[:-1]:
        axis.axhline(0, color="0.7", linewidth=0.7)
    figure.suptitle(
        (
            "Preparation heat sensitivity (mode-specific support)"
            if timing
            else "Measured cycles at fixed defrost times"
        )
        + " · paired available cycles: "
        f"{paired.common_supported.sum()}/{len(paired)}"
    )
    figure.tight_layout()
    _save_png(figure, output)
