#!/usr/bin/env python3
"""Compare camera-specific binary ResNet-50 models on identical test images."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "auroc", "class0_f1", "class1_f1")
CAMERAS = ("front", "left", "left_close", "top", "top_close", "extreme")
CAMERA_LABELS = {
    "front": "Front",
    "left": "Left",
    "left_close": "Left close",
    "top": "Top",
    "top_close": "Top close",
    "extreme": "Extreme",
}
DEDICATED_RUNS = {
    "front": "unit_front_boundary_20260827",
    "left": "unit_left_selected_20260827",
    "left_close": "unit_left_close_selected_20260827",
    "top": "unit_top_selected_20260827",
    "top_close": "unit_top_close_selected_20260827",
    "extreme": "unit_extreme_selected_20260827",
}
ADAPTED_RUNS = {camera: f"unit_{camera}_adapted_20260827" for camera in CAMERAS}
PAIR_RUNS = {
    "left_pair": ("unit_left_pair_selected_20260827", ("left", "left_close")),
    "top_pair": ("unit_top_pair_selected_20260827", ("top", "top_close")),
}
SAMPLE_NOTE = (
    "n=image frames; each camera: 8 test cycles / 3 held-out experiments; "
    "adjacent frames are not independent replicates; no seed/fold CI; no error bars."
)
FIGURE_SAMPLE_NOTE = (
    "n=image frames; each camera: 8 test cycles / 3 held-out experiments;\n"
    "adjacent frames are not independent replicates; no seed/fold CI; no error bars."
)
FORMAL_SELECTION_BASIS = "validation-selected stage within predeclared formal run"
ADAPTATION_SELECTION_BASIS = "validation-selected stage within predeclared adaptation run"
ELIGIBILITY_RULE = (
    "Eligible formal core schedule: head_epochs=5 and finetune_epochs=5. "
    "Exploratory unit_front_boundary_lr1e4_20260827 excluded: 1+1 schedule is not comparable."
)
ADAPTATION_ELIGIBILITY_RULE = (
    "Eligible adaptation schedule: initialized from the mixed-model checkpoint "
    "and trained with the configured adaptation epochs and learning rate."
)
MIXED_COLOR, SINGLE_COLOR, PAIR_COLOR = "#A8A8A8", "#7884B4", "#E4CCD8"
UP_COLOR, DOWN_COLOR, DARK = "#8BCF8B", "#E9A6A1", "#484878"


def _align_predictions(
    *frames: pd.DataFrame, require_equal_regret: bool = True
) -> tuple[pd.DataFrame, ...]:
    """Return frames in identical image-key order or reject an unfair comparison."""
    if len(frames) < 2:
        raise ValueError("at least two prediction tables are required")
    required = {
        "image_path",
        "camera",
        "cycle",
        "experiment",
        "relative_regret",
        "target",
        "prediction",
        "p1",
    }
    ordered = []
    for frame in frames:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"missing prediction columns: {sorted(missing)}")
        if frame["image_path"].duplicated().any():
            raise ValueError("duplicate image keys")
        ordered.append(frame.sort_values("image_path").reset_index(drop=True))
    keys = ordered[0]["image_path"].tolist()
    if any(frame["image_path"].tolist() != keys for frame in ordered[1:]):
        raise ValueError("image keys differ between compared models")
    for column in ("camera", "cycle", "experiment", "target"):
        reference = ordered[0][column].to_numpy()
        if any(not np.array_equal(frame[column].to_numpy(), reference) for frame in ordered[1:]):
            raise ValueError(f"{column} differs for aligned image keys")
    regret = ordered[0]["relative_regret"].to_numpy(float)
    if require_equal_regret and any(
        not np.allclose(frame["relative_regret"].to_numpy(float), regret, equal_nan=True)
        for frame in ordered[1:]
    ):
        raise ValueError("relative_regret differs for aligned image keys")
    return tuple(ordered)


def _read_selected_predictions(run: Path) -> tuple[pd.DataFrame, str]:
    """Read the validation-selected stage without inspecting test performance."""
    selection = json.loads((run / "selected_stage.json").read_text())
    stage = selection["stage"]
    predictions = pd.read_parquet(run / "test_predictions.parquet")
    selected = predictions.loc[predictions["stage"].eq(stage)].copy()
    if selected.empty:
        raise ValueError(f"selected stage {stage!r} has no test predictions in {run.name}")
    return selected, stage


def _require_formal_schedule(config: dict[str, object], run_name: str) -> None:
    if (config["head_epochs"], config["finetune_epochs"]) != (5, 5):
        raise ValueError(f"ineligible non-formal training schedule in {run_name}")


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return dict.fromkeys(METRICS, np.nan)
    target = frame["target"].to_numpy(int)
    predicted = frame["prediction"].to_numpy(int)
    class_f1 = f1_score(target, predicted, labels=[0, 1], average=None, zero_division=0)
    if pd.Series(target).nunique() == 2:
        auroc = roc_auc_score(target, frame["p1"].to_numpy(float))
    else:
        auroc = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A single label was found", category=UserWarning)
        balanced_accuracy = balanced_accuracy_score(target, predicted)
    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(
            f1_score(target, predicted, labels=[0, 1], average="macro", zero_division=0)
        ),
        "auroc": float(auroc),
        "class0_f1": float(class_f1[0]),
        "class1_f1": float(class_f1[1]),
    }


def _sample_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "n_images": len(frame),
        "n_cycles": frame["cycle"].nunique(),
        "n_experiments": frame["experiment"].nunique(),
    }


def _comparison_rows(
    camera: str,
    baseline: pd.DataFrame,
    dedicated: pd.DataFrame,
    adapted: pd.DataFrame,
) -> list[dict[str, object]]:
    baseline, dedicated, adapted = _align_predictions(
        baseline, dedicated, adapted, require_equal_regret=False
    )
    rows = []
    for scope, mask in (
        ("full", np.ones(len(baseline), dtype=bool)),
        ("near_1pct", adapted["relative_regret"].le(0.01).to_numpy()),
    ):
        baseline_scores = _metrics(baseline.loc[mask])
        dedicated_scores = _metrics(dedicated.loc[mask])
        adapted_scores = _metrics(adapted.loc[mask])
        counts = _sample_counts(baseline.loc[mask])
        for metric in METRICS:
            rows.append(
                {
                    "camera": camera,
                    "scope": scope,
                    "metric": metric,
                    "baseline": baseline_scores[metric],
                    "dedicated": dedicated_scores[metric],
                    "delta": dedicated_scores[metric] - baseline_scores[metric],
                    "adapted": adapted_scores[metric],
                    "adapted_minus_baseline": adapted_scores[metric]
                    - baseline_scores[metric],
                    "adapted_minus_dedicated": adapted_scores[metric]
                    - dedicated_scores[metric],
                    **counts,
                }
            )
    return rows


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def _export(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white", bbox_inches="tight")


def _plot_delta(source: pd.DataFrame, stem: Path) -> None:
    near = source["scope"].eq("near_1pct").all()
    rows = source.set_index("camera").reindex(CAMERAS)
    series = (
        ("delta", "Dedicated − mixed", SINGLE_COLOR, -0.17),
        ("adapted_minus_baseline", "Adapted − mixed", UP_COLOR, 0.17),
    )
    values = 100 * rows[[item[0] for item in series]].to_numpy(float)
    y = np.arange(len(rows))
    span = max(1.0, float(np.nanmax(np.abs(values))) + 1.4)
    height_mm = 78 if near else 66
    fig, ax = plt.subplots(figsize=(89 / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.28, right=0.97, bottom=0.38 if near else 0.25, top=0.93)
    for column, label, color, offset in series:
        bars = 100 * rows[column].to_numpy(float)
        ax.barh(
            y + offset,
            bars,
            color=color,
            edgecolor=DARK,
            linewidth=0.45,
            height=0.3,
            label=label,
        )
        for yi, value in zip(y + offset, bars, strict=True):
            text_offset = 0.10 if value >= 0 else -0.10
            ax.text(
                value + text_offset,
                yi,
                f"{value:+.1f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=5.7,
            )
    ax.axvline(0, color="#767676", lw=0.8)
    ax.set_yticks(y, [CAMERA_LABELS[camera] for camera in CAMERAS])
    ax.invert_yaxis()
    xlabel = (
        "Model − mixed Macro-F1 in 1% near-optimal subset\n(percentage points)"
        if near
        else "Model − mixed Macro-F1 (percentage points)"
    )
    ax.set(xlabel=xlabel, xlim=(-span, span))
    ax.grid(axis="x", color="#E5E5E5", lw=0.55, zorder=0)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=5.8)
    note = f"Scope: relative_regret<=0.01.\n{FIGURE_SAMPLE_NOTE}" if near else FIGURE_SAMPLE_NOTE
    fig.text(0.28, 0.018 if near else 0.035, note, fontsize=5.1, color="#606060")
    _export(fig, stem)
    plt.close(fig)


def _plot_pair(source: pd.DataFrame, stem: Path) -> None:
    cameras = ("left", "left_close", "top", "top_close")
    rows = source.set_index("camera").reindex(cameras)
    y = np.arange(len(rows))
    series = (
        ("mixed", "Mixed", MIXED_COLOR, -0.18),
        ("single_dedicated", "Single dedicated", SINGLE_COLOR, 0.0),
        ("pair", "Pair", PAIR_COLOR, 0.18),
    )
    values = rows[[item[0] for item in series]].to_numpy(float)
    lower, upper = float(np.nanmin(values)) - 0.008, float(np.nanmax(values)) + 0.008
    fig, ax = plt.subplots(figsize=(89 / 25.4, 66 / 25.4))
    fig.subplots_adjust(left=0.28, right=0.97, bottom=0.25, top=0.82)
    for column, label, color, offset in series:
        ax.scatter(
            rows[column],
            y + offset,
            s=25,
            color=color,
            edgecolor=DARK,
            linewidth=0.45,
            label=label,
            zorder=3,
        )
    ax.set_yticks(y, [CAMERA_LABELS[camera] for camera in cameras])
    ax.invert_yaxis()
    ax.set(xlabel="Full-test Macro-F1", xlim=(lower, upper))
    ax.grid(axis="x", color="#E5E5E5", lw=0.55)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=6)
    fig.text(0.28, 0.035, FIGURE_SAMPLE_NOTE, fontsize=5.1, color="#606060")
    _export(fig, stem)
    plt.close(fig)


def _write_outputs(
    comparison: pd.DataFrame, pair: pd.DataFrame, summary: pd.DataFrame, output: Path
) -> None:
    _style()
    source_dir, figure_dir = output / "源数据", output / "图表"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    comparison = comparison.assign(statistical_note=SAMPLE_NOTE)
    pair = pair.assign(statistical_note=SAMPLE_NOTE)
    comparison.to_csv(source_dir / "camera_specific_comparison.csv", index=False)
    pair.to_csv(source_dir / "pair_model_breakdown.csv", index=False)
    summary.to_csv(source_dir / "selected_model_summary.csv", index=False)

    full = comparison.loc[
        comparison["scope"].eq("full") & comparison["metric"].eq("macro_f1")
    ].copy()
    near = comparison.loc[
        comparison["scope"].eq("near_1pct") & comparison["metric"].eq("macro_f1")
    ].copy()
    pair_figure = pair.loc[
        pair["camera"].ne("all")
        & pair["scope"].eq("full")
        & pair["metric"].eq("macro_f1")
    ].copy()
    figure_sources = (
        ("figure_camera_specific_macro_f1_delta", full, _plot_delta),
        ("figure_camera_specific_near_1pct_macro_f1_delta", near, _plot_delta),
        ("figure_pair_model_macro_f1_comparison", pair_figure, _plot_pair),
    )
    for name, source, plotter in figure_sources:
        source.to_csv(source_dir / f"{name}.csv", index=False)
        plotter(source, figure_dir / name)


def _pair_rows(
    pair_group: str,
    camera: str,
    mixed: pd.DataFrame,
    single: pd.DataFrame,
    pair: pd.DataFrame,
) -> list[dict[str, object]]:
    mixed, single, pair = _align_predictions(mixed, single, pair)
    rows = []
    for scope, mask in (
        ("full", np.ones(len(mixed), dtype=bool)),
        ("near_1pct", mixed["relative_regret"].le(0.01).to_numpy()),
    ):
        scores = [_metrics(frame.loc[mask]) for frame in (mixed, single, pair)]
        counts = _sample_counts(mixed.loc[mask])
        for metric in METRICS:
            mixed_value, single_value, pair_value = (score[metric] for score in scores)
            rows.append(
                {
                    "pair_group": pair_group,
                    "camera": camera,
                    "scope": scope,
                    "metric": metric,
                    "mixed": mixed_value,
                    "single_dedicated": single_value,
                    "pair": pair_value,
                    "pair_minus_mixed": pair_value - mixed_value,
                    "pair_minus_single": pair_value - single_value,
                    **counts,
                }
            )
    return rows


def _selected_summary(camera_root: Path, adaptation_root: Path) -> pd.DataFrame:
    rows = []
    run_groups = [
        (group, camera_root / run_name, "from_scratch_dedicated")
        for group, run_name in DEDICATED_RUNS.items()
    ] + [
        (group, camera_root / run[0], "pair_from_scratch")
        for group, run in PAIR_RUNS.items()
    ] + [
        (camera, adaptation_root / run_name, "mixed_to_camera_adapted")
        for camera, run_name in ADAPTED_RUNS.items()
    ]
    for group, run, model_family in run_groups:
        run_name = run.name
        selection = json.loads((run / "selected_stage.json").read_text())
        stage = selection["stage"]
        stage_metrics = pd.read_csv(run / "stage_metrics.csv")
        selected = stage_metrics.loc[stage_metrics["stage"].eq(stage)].set_index("split")
        config = json.loads((run / "config.json").read_text())
        adapted = model_family == "mixed_to_camera_adapted"
        if not adapted:
            _require_formal_schedule(config, run_name)
        for split in ("validation", "near_1pct_validation", "test", "near_1pct_test"):
            if split not in selected.index:
                raise ValueError(f"missing {stage}/{split} metrics in {run_name}")
        if not np.isclose(
            selection["validation_macro_f1"], selected.loc["validation", "macro_f1"]
        ):
            raise ValueError(f"selected validation metric mismatch in {run_name}")
        rows.append(
            {
                "group": group,
                "run": run_name,
                "selected_stage": stage,
                "checkpoint": selection.get("checkpoint", ""),
                "model_family": model_family,
                "init_checkpoint": config.get("init_checkpoint", "") if adapted else "",
                "adapt_epochs": config.get("adapt_epochs", "") if adapted else "",
                "adapt_lr": config.get("adapt_lr", "") if adapted else "",
                "full_validation_macro_f1": selection["validation_macro_f1"],
                "near_1pct_validation_macro_f1": selection[
                    "near_1pct_validation_macro_f1"
                ],
                "full_test_macro_f1": selected.loc["test", "macro_f1"],
                "near_1pct_test_macro_f1": selected.loc["near_1pct_test", "macro_f1"],
                "elapsed_seconds": config["elapsed_seconds"],
                "elapsed_hours": config["elapsed_seconds"] / 3600,
                "selection_basis": ADAPTATION_SELECTION_BASIS
                if adapted
                else FORMAL_SELECTION_BASIS,
                "eligibility_rule": ADAPTATION_ELIGIBILITY_RULE
                if adapted
                else ELIGIBILITY_RULE,
            }
        )
    return pd.DataFrame(rows)


def render(
    mixed_run: Path, camera_root: Path, adaptation_root: Path, output: Path
) -> None:
    mixed_predictions = pd.read_parquet(mixed_run / "test_predictions.parquet")
    mixed_predictions = mixed_predictions.loc[mixed_predictions["stage"].eq("finetune")].copy()
    if mixed_predictions.empty:
        raise ValueError("mixed baseline finetune predictions are missing")

    selected_predictions = {}
    comparison_rows = []
    for camera, run_name in DEDICATED_RUNS.items():
        dedicated, _ = _read_selected_predictions(camera_root / run_name)
        adapted, _ = _read_selected_predictions(adaptation_root / ADAPTED_RUNS[camera])
        selected_predictions[camera] = dedicated
        baseline = mixed_predictions.loc[mixed_predictions["camera"].eq(camera)].copy()
        comparison_rows.extend(_comparison_rows(camera, baseline, dedicated, adapted))

    pair_rows = []
    for pair_group, (run_name, cameras) in PAIR_RUNS.items():
        pair_predictions, _ = _read_selected_predictions(camera_root / run_name)
        aligned_groups = [[], [], []]
        for camera in cameras:
            mixed = mixed_predictions.loc[mixed_predictions["camera"].eq(camera)].copy()
            single = selected_predictions[camera]
            paired = pair_predictions.loc[pair_predictions["camera"].eq(camera)].copy()
            mixed, single, paired = _align_predictions(mixed, single, paired)
            pair_rows.extend(_pair_rows(pair_group, camera, mixed, single, paired))
            for collection, frame in zip(aligned_groups, (mixed, single, paired), strict=True):
                collection.append(frame)
        aggregate = [pd.concat(frames, ignore_index=True) for frames in aligned_groups]
        pair_rows.extend(_pair_rows(pair_group, "all", *aggregate))

    comparison = pd.DataFrame(comparison_rows)
    pair = pd.DataFrame(pair_rows)
    summary = _selected_summary(camera_root, adaptation_root)
    if len(summary) != 14:
        raise ValueError(f"expected fourteen selected models, found {len(summary)}")
    _write_outputs(comparison, pair, summary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mixed-run",
        type=Path,
        default=Path("output/model/resnet50_binary_20260825/resnet50_binary_unit_latest_20260825"),
    )
    parser.add_argument(
        "--camera-root", type=Path, default=Path("output/model/camera_models_20260827")
    )
    parser.add_argument(
        "--adaptation-root",
        type=Path,
        default=Path("output/model/camera_adaptation_20260827"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/model/camera_models_20260827/论文分析")
    )
    args = parser.parse_args()
    render(args.mixed_run, args.camera_root, args.adaptation_root, args.output)


if __name__ == "__main__":
    main()
