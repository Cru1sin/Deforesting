#!/usr/bin/env python3
"""Plot post-optimum probabilities along each held-out test cycle."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Arial", "DejaVu Sans", "Liberation Sans"]
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
    }
)

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/resnet50_binary/resnet50_binary_unit_latest_20260825"
CAMERA_RUN_ROOT = ROOT / "outputs/resnet50_binary_camera_models"
COST = ROOT / "report/02_经济除霜窗口/经验经济窗口/源数据"
OUT = ROOT / "report/03_RGB标签与模型/热量口径二分类/图表/循环概率曲线"
CAMERA_OUT = OUT.parent / "单机位模型循环概率曲线"
SOURCE = OUT.parent / "源数据"
THRESHOLD = 0.5

CAMERAS = ("front", "left", "left_close", "top", "top_close", "extreme")
CAMERA_LABELS = {
    "front": "正面",
    "left": "左侧",
    "left_close": "左侧近景",
    "top": "顶部",
    "top_close": "顶部近景",
    "extreme": "极端角度",
}
CAMERA_COLORS = {
    "front": "#0F4D92",
    "left": "#009E73",
    "left_close": "#56B4E9",
    "top": "#D55E00",
    "top_close": "#E69F00",
    "extreme": "#8E6C8A",
}

CAMERA_RUNS = {
    camera: (
        "unit_front_boundary_20260827"
        if camera == "front"
        else f"unit_{camera}_selected_20260827"
    )
    for camera in CAMERAS
}


def load_camera_specific_runs(
    run_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    predictions, manifests, loaded = [], [], []
    required = ("selected_stage.json", "test_predictions.parquet", "manifest.parquet")
    for camera in CAMERAS:
        run = run_root / CAMERA_RUNS[camera]
        if not all((run / name).is_file() for name in required):
            continue
        selected_stage = json.loads((run / "selected_stage.json").read_text(encoding="utf-8"))[
            "stage"
        ]
        values = pd.read_parquet(run / "test_predictions.parquet")
        values = values.loc[
            values["camera"].eq(camera) & values["stage"].eq(selected_stage)
        ].copy()
        if values.empty:
            if camera == "front":
                raise ValueError(
                    f"front run {run.name} has no predictions for stage {selected_stage}"
                )
            continue
        values["model_run"] = run.name
        values["selected_stage"] = selected_stage
        predictions.append(values)
        manifests.append(
            pd.read_parquet(
                run / "manifest.parquet", columns=["cycle_name", "stable_heating_start"]
            )
        )
        loaded.append(camera)
    if not loaded or loaded[0] != "front":
        front_run = run_root / CAMERA_RUNS["front"]
        raise FileNotFoundError(
            f"complete front run required: {front_run} ({', '.join(required)})"
        )
    manifest = (
        pd.concat(manifests, ignore_index=True)
        .drop_duplicates("cycle_name")
        .set_index("cycle_name")
    )
    return pd.concat(predictions, ignore_index=True), manifest, tuple(loaded)


def minutes(timestamp: object, origin: pd.Timestamp) -> float:
    return (pd.Timestamp(timestamp) - origin).total_seconds() / 60


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.13, 1, 1), pad=0.9)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_cycle(
    cycle: str,
    probabilities: pd.DataFrame,
    boundary: pd.Series,
    point: pd.Series,
    near_band: pd.DataFrame,
    cameras: tuple[str, ...] = CAMERAS,
    output_dir: Path = OUT,
    camera_specific: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    stable = pd.Timestamp(boundary["stable_heating_start"])
    optimum = minutes(point["t_star_unit"], stable)
    near_start = minutes(near_band["candidate_time"].min(), stable)
    near_end = minutes(near_band["candidate_time"].max(), stable)

    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    ax.axvspan(near_start, near_end, color="#AADCA9", alpha=0.22, lw=0)

    trigger = pd.NaT
    for camera in cameras:
        values = probabilities.loc[probabilities["camera"].eq(camera)].sort_values("time").copy()
        values["frost_minutes"] = (values["time"] - stable).dt.total_seconds() / 60
        # The rolling minimum crosses theta iff all three consecutive frames cross theta.
        values["rolling_min_3"] = values["p1"].rolling(3, min_periods=3).min()
        values["cycle"] = cycle
        values["phase"] = values["frost_minutes"].lt(0).map({True: "recovery", False: "frosting"})
        columns = ["cycle", "camera", "time", "frost_minutes", "p1", "rolling_min_3", "phase"]
        if camera_specific:
            columns += [
                column for column in ("stage", "model_run", "selected_stage") if column in values
            ]
        rows.append(values[columns])
        color = CAMERA_COLORS[camera]
        ax.plot(values["frost_minutes"], values["p1"], color=color, lw=0.65, alpha=0.28)
        ax.plot(values["frost_minutes"], values["rolling_min_3"], color=color, lw=1.45)
        if camera == "front":
            crossed = values.loc[values["rolling_min_3"].ge(THRESHOLD), "time"]
            trigger = crossed.iloc[0] if len(crossed) else pd.NaT

    rule_trigger = minutes(trigger, stable) if pd.notna(trigger) else None
    ax.axhline(THRESHOLD, color="#272727", lw=1.0, ls=(0, (2, 2)))
    ax.axvline(optimum, color="#B64342", lw=1.35)
    if rule_trigger is not None and near_start <= rule_trigger <= near_end:
        ax.axvline(rule_trigger, color="#9A4D8E", lw=1.25, ls="-.")
    elif rule_trigger is not None:
        side = "早于" if rule_trigger < near_start else "晚于"
        edge = near_start if rule_trigger < near_start else near_end
        ax.scatter(edge, THRESHOLD, marker="<" if rule_trigger < near_start else ">",
                   s=26, color="#9A4D8E", zorder=5, clip_on=False)
        ax.text(edge, 0.54, f"触发{side}窗口", color="#9A4D8E", fontsize=6,
                ha="left" if rule_trigger < near_start else "right", va="bottom")

    camera_handles = [
        Line2D([0], [0], color=CAMERA_COLORS[camera], lw=1.8, label=CAMERA_LABELS[camera])
        for camera in cameras
    ]
    camera_legend = ax.legend(
        handles=camera_handles, ncol=len(cameras), loc="lower center", bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.1, handlelength=2.0, fontsize=6.4,
    )
    ax.add_artist(camera_legend)
    event_handles = [
        Line2D([0], [0], color="#B64342", lw=1.35, label=f"最优点 {optimum:.1f} min"),
        Line2D(
            [0],
            [0],
            color="#9A4D8E",
            lw=1.25,
            ls="-.",
            label=(
                f"正面三帧触发 {rule_trigger:.1f} min"
                if rule_trigger is not None
                else "正面三帧未触发"
            ),
        ),
        Line2D([0], [0], color="#272727", lw=1.0, ls=(0, (2, 2)), label=r"触发阈值 $\theta=0.5$"),
        Line2D([0], [0], color="#767676", lw=0.65, alpha=0.35, label="原始概率"),
        Line2D([0], [0], color="#767676", lw=1.45, label="三帧连续判定（滚动最小值）"),
    ]
    ax.legend(handles=event_handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.19),
              fontsize=6.0, columnspacing=1.0, handlelength=2.2)

    ax.set(
        xlim=(near_start, near_end),
        ylim=(0, 1.02),
        xlabel="实际结霜时间 / min（0 = 稳定制热开始）",
        ylabel=r"已过最优点概率 $p_t$",
    )
    title = (
        f"循环 {int(cycle.rsplit('_', 1)[1])}：1%近优区间内的{len(cameras)}机位专用模型概率轨迹"
        if camera_specific
        else f"循环 {int(cycle.rsplit('_', 1)[1])}：1%近优区间内的六机位概率轨迹"
    )
    ax.set_title(title, pad=28)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color="#D8D8D8", lw=0.55, alpha=0.65)

    plotted = pd.concat(rows, ignore_index=True)
    assert set(plotted["camera"]) == set(cameras)
    assert plotted["rolling_min_3"].dropna().between(0, 1).all()
    model = "ResNet50 finetune checkpoint"
    if camera_specific:
        selected_stages = probabilities.drop_duplicates("camera").set_index("camera")[
            "selected_stage"
        ]
        stage_summary = ";".join(f"{camera}={selected_stages[camera]}" for camera in cameras)
        model = f"ResNet50 camera-specific checkpoints; selected stages: {stage_summary}"
    event = {
        "cycle": cycle,
        "stable_heating_start": stable,
        "optimal_point_minutes": optimum,
        "near_optimum_start_minutes": near_start,
        "near_optimum_end_minutes": near_end,
        "front_three_frame_trigger_minutes": rule_trigger,
        "threshold": THRESHOLD,
        "rolling_rule": "minimum of current and previous two front-camera probabilities >= 0.5",
        "model": model,
        "heat_basis": "unit",
        "split": "held-out test",
    }
    if camera_specific:
        event.update(camera_count=len(cameras), cameras=",".join(cameras))
    save(fig, output_dir / f"cycle_probability_{cycle}")
    return plotted, event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-specific",
        action="store_true",
        help="plot completed camera-specific model runs selected by selected_stage.json",
    )
    args = parser.parse_args()
    if args.camera_specific:
        predictions, manifest, cameras = load_camera_specific_runs(CAMERA_RUN_ROOT)
        output_dir = CAMERA_OUT
        source_prefix = "31_camera_specific_probability"
    else:
        predictions = pd.read_parquet(RUN / "test_predictions.parquet")
        predictions = predictions.loc[predictions["stage"].eq("finetune")].copy()
        manifest = pd.read_parquet(
            RUN / "manifest.parquet",
            columns=["cycle_name", "stable_heating_start"],
        ).drop_duplicates("cycle_name").set_index("cycle_name")
        cameras = CAMERAS
        output_dir = OUT
        source_prefix = "30_cycle_probability"
    predictions["time"] = pd.to_datetime(predictions["time"], format="mixed")
    points = pd.read_csv(COST / "cycle_optimal_points.csv").set_index("cycle_name")
    curves = pd.read_parquet(
        COST / "candidate_cost_curves.parquet",
        columns=["cycle_name", "candidate_time", "optimization_eligible", "relative_regret_unit"],
    )
    curves["candidate_time"] = pd.to_datetime(curves["candidate_time"], format="mixed")

    plotted, events = [], []
    for cycle, values in predictions.groupby("cycle", sort=True):
        near = curves.loc[
            curves["cycle_name"].eq(cycle)
            & curves["optimization_eligible"].fillna(False)
            & curves["relative_regret_unit"].le(0.01)
        ]
        source, event = plot_cycle(
            cycle,
            values,
            manifest.loc[cycle],
            points.loc[cycle],
            near,
            cameras=cameras,
            output_dir=output_dir,
            camera_specific=args.camera_specific,
        )
        plotted.append(source)
        events.append(event)

    SOURCE.mkdir(parents=True, exist_ok=True)
    pd.concat(plotted, ignore_index=True).to_csv(
        SOURCE / f"{source_prefix}_curves.csv", index=False
    )
    pd.DataFrame(events).to_csv(SOURCE / f"{source_prefix}_events.csv", index=False)
    print(f"generated {len(events)} cycle figures for {len(cameras)} cameras in {output_dir}")


if __name__ == "__main__":
    main()
