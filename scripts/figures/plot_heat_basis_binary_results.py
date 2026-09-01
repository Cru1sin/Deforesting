#!/usr/bin/env python3
"""Plot single-panel Nature-style comparisons for heat-basis binary models."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    basis: ROOT
    / "output/model/resnet50_binary_20260825"
    / f"resnet50_binary_{basis}_latest_20260825"
    for basis in ("water", "unit")
}
OUT = ROOT / "output/model/rgb_binary_results_latest/热量口径二分类/图表"
SRC = OUT / "源数据"

COLORS = {
    "water": "#7884B4",
    "unit": "#D49AAA",
    "rgb_z": "#484878",
    "z_current": "#7884B4",
    "z_current_slope": "#D49AAA",
    "head": "#7884B4",
    "finetune": "#D49AAA",
    "positive": "#4F8A63",
    "negative": "#B85C5C",
    "neutral": "#767676",
}
METHOD_LABEL = {
    "rgb_z": "RGB embedding",
    "z_current": "+ current state",
    "z_current_slope": "+ 5-min slope",
}
BEST_INPUT = {"water": "z_current", "unit": "z_current_slope"}
CAMERAS = ["front", "left", "left_close", "top", "top_close", "extreme"]
CAMERA_LABEL = {
    "front": "Front",
    "left": "Left",
    "left_close": "Left close",
    "top": "Top",
    "top_close": "Top close",
    "extreme": "Extreme",
}
META = {
    "experiment_split": "fixed experiment-level split",
    "replication": "single training run; no error bars",
}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.8)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def source(df, name, **notes):
    SRC.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for key, value in {**META, **notes}.items():
        out[key] = value
    out.to_csv(SRC / f"{name}.csv", index=False)


def loss_resnet(basis):
    name = f"01_{basis}_resnet_train_loss" if basis == "water" else f"02_{basis}_resnet_train_loss"
    df = pd.read_csv(RUNS[basis] / "history.csv")
    df["global_epoch"] = np.arange(1, len(df) + 1)
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    for stage in ("head", "finetune"):
        d = df[df.stage == stage]
        ax.plot(
            d.global_epoch,
            d.train_loss,
            marker="o",
            ms=3.2,
            lw=1.6,
            color=COLORS[stage],
            label=stage.capitalize(),
        )
    ax.axvline(5.5, color="#A8A8A8", lw=0.9, ls="--")
    ax.text(
        5.5,
        ax.get_ylim()[1],
        "stage boundary",
        ha="center",
        va="bottom",
        color=COLORS["neutral"],
        fontsize=6,
    )
    ax.set(xlabel="Training epoch", ylabel="Cross-entropy loss", xticks=range(1, 11))
    ax.legend(loc="upper right")
    source(
        df[["stage", "epoch", "global_epoch", "train_loss"]],
        name,
        split="training",
        stage_boundary="between global epochs 5 and 6",
    )
    save(fig, name)


def loss_fusion(basis):
    name = f"03_{basis}_fusion_train_loss" if basis == "water" else f"04_{basis}_fusion_train_loss"
    df = pd.read_csv(RUNS[basis] / "sensor_fusion/history.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    for method in METHOD_LABEL:
        d = df[df.input == method]
        ax.plot(
            d.epoch,
            d.train_loss,
            marker="o",
            ms=2.8,
            lw=1.45,
            color=COLORS[method],
            label=METHOD_LABEL[method],
        )
    ax.set(xlabel="Training epoch", ylabel="Cross-entropy loss", xticks=range(1, 11))
    ax.legend()
    source(df[["input", "epoch", "train_loss"]], name, split="training")
    save(fig, name)


def metric_table():
    return pd.concat(
        [
            pd.read_csv(RUNS[b] / "sensor_fusion/metrics.csv").assign(heat_basis=b)
            for b in ("water", "unit")
        ],
        ignore_index=True,
    )


def macro_f1(split, number):
    name = f"{number:02d}_{'full_test' if split == 'test' else 'near_1pct'}_macro_f1"
    df = metric_table().query("split == @split").copy()
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    x = np.arange(2)
    width = 0.23
    for i, method in enumerate(METHOD_LABEL):
        vals = [
            df.loc[(df.heat_basis == b) & (df.input == method), "macro_f1"].iloc[0]
            for b in ("water", "unit")
        ]
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            color=COLORS[method],
            label=METHOD_LABEL[method],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, val in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.003,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                rotation=90,
            )
    low = max(0, df.macro_f1.min() - (0.035 if split == "test" else 0.06))
    ax.set_ylim(low, min(1.02, df.macro_f1.max() + 0.055))
    ax.set_xticks(x, ["Water-side heat", "Unit heat"])
    ax.set_ylabel("Macro-F1")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=6)
    subset = "full held-out test" if split == "test" else "held-out test, relative regret <= 1%"
    source(
        df[["heat_basis", "input", "split", "images", "macro_f1"]], name, evaluation_subset=subset
    )
    save(fig, name)


def accuracy(split, number):
    name = f"{number:02d}_{'full_test' if split == 'test' else 'near_1pct'}_accuracy"
    df = metric_table().query("split == @split").copy()
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    x = np.arange(2)
    width = 0.23
    for i, method in enumerate(METHOD_LABEL):
        vals = [
            df.loc[(df.heat_basis == b) & (df.input == method), "accuracy"].iloc[0]
            for b in ("water", "unit")
        ]
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            color=COLORS[method],
            label=METHOD_LABEL[method],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, val in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.003,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                rotation=90,
            )
    low = max(0, df.accuracy.min() - (0.035 if split == "test" else 0.06))
    ax.set_ylim(low, min(1.02, df.accuracy.max() + 0.055))
    ax.set_xticks(x, ["Water-side heat", "Unit heat"])
    ax.set_ylabel("Accuracy")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=6)
    subset = "full held-out test" if split == "test" else "held-out test, relative regret <= 1%"
    source(
        df[["heat_basis", "input", "split", "images", "accuracy"]], name, evaluation_subset=subset
    )
    save(fig, name)


def sensor_gain(metric="macro_f1", number=7):
    name = f"{number:02d}_sensor_gain_delta_{metric}"
    metrics = metric_table()
    rows = []
    for basis in ("water", "unit"):
        for split in ("test", "near_1pct_test"):
            d = metrics.query("heat_basis == @basis and split == @split").set_index("input")
            for method in ("z_current", "z_current_slope"):
                rows.append(
                    {
                        "heat_basis": basis,
                        "split": split,
                        "input": method,
                        f"delta_{metric}_vs_rgb_z": (
                            d.loc[method, metric] - d.loc["rgb_z", metric]
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    groups = [(b, s) for b in ("water", "unit") for s in ("test", "near_1pct_test")]
    labels = ["Water\nfull", "Water\nnear 1%", "Unit\nfull", "Unit\nnear 1%"]
    fig, ax = plt.subplots(figsize=(4.1, 2.65))
    x = np.arange(4)
    width = 0.32
    for i, method in enumerate(("z_current", "z_current_slope")):
        vals = [
            df.loc[
                (df.heat_basis == b) & (df.split == s) & (df.input == method),
                f"delta_{metric}_vs_rgb_z",
            ].iloc[0]
            for b, s in groups
        ]
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            color=COLORS[method],
            label=METHOD_LABEL[method],
            edgecolor="white",
            linewidth=0.6,
        )
    ax.axhline(0, color="#767676", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel(
        rf"$\Delta$ {'Macro-F1' if metric == 'macro_f1' else 'accuracy'} vs RGB embedding"
    )
    ax.legend(ncol=2, loc="upper left")
    source(df, name, baseline="rgb_z within the same heat basis and evaluation subset")
    save(fig, name)


def training_history(basis, number):
    name = f"{number:02d}_{basis}_resnet_loss_validation_accuracy"
    df = pd.read_csv(RUNS[basis] / "history.csv")
    df["global_epoch"] = np.arange(1, len(df) + 1)
    fig, ax_loss = plt.subplots(figsize=(3.7, 2.55))
    ax_acc = ax_loss.twinx()
    ax_loss.plot(
        df.global_epoch,
        df.train_loss,
        color=COLORS["rgb_z"],
        marker="o",
        ms=3.0,
        lw=1.5,
        label="Training loss",
    )
    ax_acc.plot(
        df.global_epoch,
        df.validation_accuracy,
        color=COLORS["z_current_slope"],
        marker="s",
        ms=2.8,
        lw=1.5,
        label="Validation accuracy",
    )
    ax_loss.axvline(5.5, color="#A8A8A8", lw=0.8, ls="--")
    ax_loss.set(xlabel="Training epoch", ylabel="Cross-entropy loss", xticks=range(1, 11))
    ax_acc.set_ylabel("Validation accuracy")
    lines = [ax_loss.lines[0], ax_acc.lines[0]]
    ax_loss.legend(
        lines,
        [line.get_label() for line in lines],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
    )
    ax_acc.spines["top"].set_visible(False)
    source(
        df[["stage", "epoch", "global_epoch", "train_loss", "validation_accuracy"]],
        name,
        split="training loss and held-out validation accuracy",
        chen_analogue="Fig. 9; test accuracy by epoch was not logged",
    )
    save(fig, name)


def camera_accuracy(basis, number):
    name = f"{number:02d}_{basis}_camera_accuracy"
    method = BEST_INPUT[basis]
    pred = pd.read_parquet(RUNS[basis] / "sensor_fusion/predictions.parquet")
    pred = pred[(pred.split == "test") & pred.input.isin(["rgb_z", method])].copy()
    pred["correct"] = pred.target.eq(pred.prediction.astype(int))
    df = pred.groupby(["camera_role", "input"], as_index=False).agg(
        accuracy=("correct", "mean"), images=("correct", "size")
    )
    fig, ax = plt.subplots(figsize=(4.55, 2.65))
    x = np.arange(len(CAMERAS))
    width = 0.34
    for i, current in enumerate(("rgb_z", method)):
        vals = [
            df.loc[(df.camera_role == camera) & (df.input == current), "accuracy"].iloc[0]
            for camera in CAMERAS
        ]
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            color=COLORS[current],
            label=METHOD_LABEL[current],
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_ylim(max(0, df.accuracy.min() - 0.025), min(1.005, df.accuracy.max() + 0.012))
    ax.set_xticks(x, [CAMERA_LABEL[c] for c in CAMERAS], rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.legend(ncol=2, loc="lower center")
    source(
        df,
        name,
        split="full held-out test",
        interpretation="descriptive camera comparison; not a causal angle or illumination test",
    )
    save(fig, name)


def confusion(basis, method, number, camera=None):
    suffix = f"_{camera}" if camera else ""
    name = f"{number:02d}_{basis}{suffix}_{method}_normalized_confusion"
    pred = pd.read_parquet(
        RUNS[basis] / "sensor_fusion/predictions.parquet",
        columns=["camera_role", "split", "input", "target", "prediction"],
    )
    pred = pred[(pred["split"] == "test") & (pred["input"] == method)].copy()
    if camera:
        pred = pred[pred["camera_role"] == camera]
    pred["prediction"] = pred.prediction.astype(int)
    counts = pd.crosstab(pred.target, pred.prediction).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0
    )
    norm = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(2.55, 2.45))
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            val = norm.iloc[i, j]
            ax.text(
                j,
                i,
                f"{val:.1%}\n(n={counts.iloc[i, j]:,})",
                ha="center",
                va="center",
                color="white" if val > 0.52 else "#272727",
                fontsize=7,
            )
    ax.set_xticks([0, 1], ["Pre", "Post"])
    ax.set_yticks([0, 1], ["Pre", "Post"])
    ax.set(xlabel="Predicted class", ylabel="True class")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.04)
    cbar.set_label("Row proportion")
    out = counts.stack().rename("count").reset_index()
    out.columns = ["true_class", "predicted_class", "count"]
    out["row_proportion"] = [
        norm.loc[i, j] for i, j in zip(out.true_class, out.predicted_class, strict=True)
    ]
    source(
        out,
        name,
        split="full held-out test",
        input=method,
        camera=camera or "all",
        normalization="within true class",
    )
    save(fig, name)


def write_chen_figure_map():
    rows = [
        (8, "unsupported", "", "No K-means clustering or silhouette experiment."),
        (
            9,
            "analogue",
            "14-15",
            "Training loss and validation accuracy are logged; epoch-wise test accuracy is not.",
        ),
        (
            10,
            "analogue",
            "18-29",
            "Per-camera normalized confusion matrices for the best current models.",
        ),
        (11, "unsupported", "", "Requires retraining on multiple labelled-sample budgets."),
        (
            12,
            "descriptive analogue",
            "16-17",
            "Per-camera accuracy is available, but angle and illumination were not controlled.",
        ),
        (13, "unsupported", "", "No deployed monthly control experiment."),
        (14, "unsupported", "", "No deployed per-unit defrost control comparison."),
        (15, "unsupported", "", "Requires a matched conventional-CNN baseline retraining."),
        (16, "unsupported", "", "No clustering and augmentation ablation."),
        (17, "unsupported", "", "No augmentation-versus-GAN experiment."),
        (18, "unsupported", "", "No backbone, optimizer, learning-rate or batch-size ablation."),
    ]
    pd.DataFrame(rows, columns=["chen_figure", "status", "current_figure", "reason"]).to_csv(
        SRC / "chen_figure_reproduction_map.csv", index=False
    )


def tstar_shift():
    name = "10_unit_minus_water_tstar_shift"
    columns = ["cycle_uid", "cycle_name", "camera_role", "image_time", "t_star", "target"]
    water = pd.read_parquet(RUNS["water"] / "manifest.parquet", columns=columns)
    unit = pd.read_parquet(RUNS["unit"] / "manifest.parquet", columns=columns)
    times = (
        water.groupby("cycle_uid", as_index=False)
        .agg(cycle_name=("cycle_name", "first"), water_t_star=("t_star", "first"))
        .merge(
            unit.groupby("cycle_uid", as_index=False).agg(unit_t_star=("t_star", "first")),
            on="cycle_uid",
            validate="one_to_one",
        )
    )
    matched = water.merge(
        unit,
        on=["cycle_uid", "cycle_name", "camera_role", "image_time"],
        suffixes=("_water", "_unit"),
        validate="one_to_one",
    )
    matched["label_flipped"] = matched.target_water != matched.target_unit
    flips = (
        matched.groupby("cycle_uid")
        .label_flipped.agg([("label_flip_count", "sum"), ("image_count", "size")])
        .reset_index()
    )
    times = times.merge(flips, on="cycle_uid", validate="one_to_one")
    times["unit_minus_water_minutes"] = (
        pd.to_datetime(times.unit_t_star) - pd.to_datetime(times.water_t_star)
    ).dt.total_seconds() / 60
    times["label_flip_fraction"] = times.label_flip_count / times.image_count
    times["global_label_flip_count"] = int(matched.label_flipped.sum())
    times["global_label_flip_fraction"] = matched.label_flipped.mean()
    times = times.sort_values("unit_minus_water_minutes").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.7, 7.1))
    y = np.arange(len(times))
    values = times.unit_minus_water_minutes.to_numpy()
    colors = np.where(values >= 0, COLORS["positive"], COLORS["negative"])
    ax.barh(y, values, color=colors, height=0.72)
    ax.axvline(0, color="#606060", lw=0.8)
    ax.set_yticks(y, times.cycle_name.str.replace("frost_cycle_", "C", regex=False), fontsize=5.5)
    ax.set_xlabel(r"$t^*_{unit} - t^*_{water}$ (min)")
    ax.set_ylabel("Local cycle")
    ax.margins(y=0.01)
    source(
        times,
        name,
        snapshot="two 2026-08-25 manifests; exact local-frame match",
        sign="positive means later unit-heat boundary",
    )
    save(fig, name)


def main():
    for basis in ("water", "unit"):
        loss_resnet(basis)
        loss_fusion(basis)
    macro_f1("test", 5)
    macro_f1("near_1pct_test", 6)
    sensor_gain()
    confusion("water", "z_current", 8)
    confusion("unit", "z_current_slope", 9)
    tstar_shift()
    accuracy("test", 11)
    accuracy("near_1pct_test", 12)
    sensor_gain("accuracy", 13)
    training_history("water", 14)
    training_history("unit", 15)
    camera_accuracy("water", 16)
    camera_accuracy("unit", 17)
    number = 18
    for basis in ("water", "unit"):
        for camera in CAMERAS:
            confusion(basis, BEST_INPUT[basis], number, camera)
            number += 1
    write_chen_figure_map()
    print(f"Wrote 29 figures (SVG/PDF/PNG) and source CSVs to {OUT}")


if __name__ == "__main__":
    main()
