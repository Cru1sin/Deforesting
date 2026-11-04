"""Effective heating COP decisions, using the shared publication renderer."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from dataset_tools.local_images import RGB_PANEL_MAX_OFFSET
from output_layout import PROJECT
from plots.cop import (
    COP_COLOR,
    OPTIMAL_COLOR,
    RB_COLOR,
    _plot_supported_curve,
    original_missing_spans,
    plot_normalized,
    plot_objectives,
)
from plots.publication import (
    _RGB_PANEL_FONT,
    _STAGE_COLORS,
    _display_label,
    _plot_cycle_panel,
    _shade_cycle_stages,
    _stage_spans,
    save_publication,
)

DEFAULT_FROST_PROGRESSION = (
    Path(__file__).resolve().parents[1]
    / "output/99_历史归档/legacy_test/slingshot_harmony_20261001/progression.csv"
)


def load_native_frost_frames(progression=DEFAULT_FROST_PROGRESSION):
    """Use the frozen native global trajectory for every effective-COP rendering path."""

    progression = Path(progression)
    settings = json.loads((progression.parent / "run_settings.json").read_text())
    if settings.get("coordinate_correction") != "harmony" or settings.get("method") != "slingshot":
        raise ValueError("Effective figures require native global Harmony + Slingshot")
    frames = pd.read_csv(progression)
    if not frames.reference_method.eq("slingshot").all():
        raise ValueError("Frost frames must use native Slingshot scores")
    frames.image_time = pd.to_datetime(frames.image_time)
    return frames


def render_current_decision_cycle(
    decisions, loader, output_path, *, images=None, near_set_rule="exact", frost_frames=None
):
    from defrost_decision.performance_objectives import calculate_cycle_cop, select_cycle_optimum

    decisions = calculate_cycle_cop(
        decisions.sort_values("candidate_defrost_time").reset_index(drop=True), effective=True
    )
    decisions = select_cycle_optimum(decisions)
    name = decisions.cycle_name.iloc[0]
    if frost_frames is None:
        frames = load_native_frost_frames()
        frost_frames = frames.loc[frames.cycle_name.eq(name)].copy()
    selected = decisions.loc[decisions.is_selected.fillna(False)]
    optimum = selected.candidate_defrost_time.iloc[0] if len(selected) else pd.NaT
    images = loader.load_cycle_images(name) if images is None else images
    if not frost_frames.empty and not images.empty:
        keys = ["cycle_name", "camera_role", "file_name"]
        frost_frames = frost_frames.merge(
            images[keys + ["path"]].rename(columns={"path": "image_path"}),
            on=keys,
            how="left",
            validate="one_to_one",
        )
    matches = match_decision_rgb_images(
        loader.load_image_metadata(name),
        images,
        {"rb": decisions.t_RB.iloc[0], "optimal": optimum},
    )
    frame = loader.load_cycle(name)
    render_decision_publication(
        frame,
        loader.get_cycle_record(name),
        decisions,
        {row.target_type: row._asdict() for row in matches.itertuples(index=False)},
        output_path,
        optimal_label=(
            "Effective COP maximum (without temperature truncation)"
            if decisions.algorithm.eq("effective_cop_without_trunc").all()
            else "Effective heating COP maximum"
        ),
        parallel_curve=decisions,
        full_near_sets=True,
        connected_near_sets=near_set_rule == "connected",
        frost_frames=frost_frames,
    )


def render_current_decision_figures(
    decisions,
    loader,
    output,
    n_jobs=6,
    fetch_cloud_images=False,
    progression=DEFAULT_FROST_PROGRESSION,
):
    from defrost_decision.selection_results import summarize_decisions

    if (
        "effective_heat_rule" in decisions
        and not decisions.effective_heat_rule.eq("without_trunc").all()
    ):
        raise ValueError("Only without_trunc effective COP figures are published")
    output = Path(output)
    frames = load_native_frost_frames(progression)
    frames = frames.loc[frames.cycle_name.isin(decisions.cycle_name)].copy()
    frost = dict(tuple(frames.groupby("cycle_name")))
    output.mkdir(parents=True, exist_ok=True)
    frames.to_csv(output / "frost_state_frames.csv", index=False)
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
    figure.savefig(
        output / "effective_COP_vs_RB.png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        Parallel(verbose=10)(
            delayed(render_current_decision_cycle)(
                curve,
                loader,
                output
                / "cost_function_effective_cop_cycle"
                / f"cycle_{int(name.rsplit('_', 1)[-1]):03d}_publication.png",
                frost_frames=frost.get(name, frames.iloc[:0]),
            )
            for name, curve in decisions.groupby("cycle_name", sort=True)
        )


def main():
    """Render current COP decisions and their native frost grades."""
    import argparse

    from dataset_tools import DatasetLoader
    from image_models.cycle_selection import screen_published_cycles
    from output_layout import analysis_output

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    parser.add_argument(
        "--cohort-dataset",
        type=Path,
        help="re-screen this published cohort for a fixed-cohort Dataset comparison",
    )
    parser.add_argument(
        "--publish-dataset-figures",
        action="store_true",
        help="refresh existing Dataset COP image assets and copies",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", nargs="+", help="optional subset after authoritative screening")
    parser.add_argument(
        "--frost-overlap",
        type=Path,
        metavar="COST_COMPARISON",
        help="plot four saved cost recipes side by side on the native frost grade",
    )
    parser.add_argument(
        "--progression",
        type=Path,
        default=DEFAULT_FROST_PROGRESSION,
        help="native Harmony+Slingshot progression.csv",
    )
    parser.add_argument("--n-jobs", type=int, default=6)
    args = parser.parse_args()
    output = analysis_output(args.output, replay=True)
    if output.exists():
        raise FileExistsError(output)
    loader = DatasetLoader(args.dataset)
    screening = screen_published_cycles(args.cohort_dataset or args.dataset)
    if args.frost_overlap is not None:
        if args.progression is None:
            parser.error("--progression is required for frost overlap")
        from plots.frost_comparison import render_cost_grade_comparison

        render_cost_grade_comparison(args.frost_overlap, args.progression, screening, output)
        return
    names = set(screening.loc[screening.selected, "cycle_name"])
    if args.publish_dataset_figures:
        from dataset_tools.cycle_metadata import read_catalog

        names = {
            r["cycle_name"]
            for r in read_catalog(args.dataset)["cycles"]
            if "effective_cop_without_trunc" in r["assets"]
        }
    if args.cycles:
        if not set(args.cycles) <= names:
            raise ValueError("Requested cycles must belong to the re-screened main cohort")
        names &= set(args.cycles)
    source = args.dataset / "training_data"
    if args.progression is None:
        parser.error("--progression is required for the native frost panel")
    decisions = pd.read_parquet(source / "candidate_decisions.parquet")
    decisions = decisions.loc[decisions.cycle_name.isin(names)].copy()
    decisions.candidate_defrost_time = pd.to_datetime(
        decisions.candidate_defrost_time, format="mixed"
    )
    output.mkdir(parents=True)
    screening.loc[screening.cycle_name.isin(names)].to_csv(output / "cohort.csv", index=False)
    decisions.to_parquet(output / "candidate_decisions.parquet", index=False)
    (output / "settings.json").write_text(
        json.dumps(
            dict(
                dataset=str(args.dataset),
                candidates=str(source / "candidate_decisions.parquet"),
                progression=str(args.progression),
                coordinate_correction="harmony",
                caption=False,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    render_current_decision_figures(
        decisions, loader, output, n_jobs=args.n_jobs, progression=args.progression
    )
    if args.publish_dataset_figures:
        from plots.dataset import refresh_effective_decision_figures

        sources = {
            name: output
            / "cost_function_effective_cop_cycle"
            / f"cycle_{int(name.rsplit('_', 1)[-1]):03d}_publication.png"
            for name in names
        }
        refresh_effective_decision_figures(args.dataset, sources)
    return


def render_realtime_cost_comparison(frame, output, name):
    """Same quantitative renderer for every transient cost example."""
    with plt.rc_context({"font.family": _RGB_PANEL_FONT, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
        x = frame.seconds_from_event / 60
        for column, label, color, style in [
            ("zero_cost_new", "新边界：未来成本=0", "#555555", "--"),
            ("cop_formal", "正式 Ridge（原边界）", "#D55E00", "-"),
            ("cop_new", "实测本循环恢复＋准备/除霜预测", "#0072B2", "-"),
            ("zero_cost_formal", "原边界：未来成本=0", "#999999", ":"),
        ]:
            axes[0].plot(x, frame[column], label=label, color=color, ls=style, lw=1.2)
        axes[0].set(ylabel="有效供热 COP", title="a  相同实测过程；两种边界各有零成本参照")
        axes[0].legend(fontsize=8, ncol=2)
        for label, color, text in [
            ("formal", "#D55E00", "正式 Ridge"),
            ("new", "#0072B2", "准备＋除霜"),
        ]:
            axes[1].plot(
                x,
                1000 * frame[f"cost_{label}_kwh"].where(frame[f"inputs_valid_{label}"]),
                color=color,
                label=text,
            )
            outside = ~frame[f"in_domain_{label}"].eq(True) & frame[f"inputs_valid_{label}"]
            axes[1].scatter(
                x[outside],
                1000 * frame.loc[outside, f"cost_{label}_kwh"],
                s=7,
                facecolors="none",
                edgecolors=color,
                label=text + " 外推",
            )
            axes[4].plot(x, frame[f"cost_cop_step_{label}"], color=color, label=text)
        axes[1].set(ylabel="未来预测电耗 [Wh]", title="b  空心点为训练范围外；不作物理真值解释")
        axes[1].legend(fontsize=8, ncol=2)
        axes[2].plot(x, frame.compressor_frequency, color="#222222", label="模型可见实际频率")
        right = axes[2].twinx()
        right.plot(x, frame.evaporating_pressure, color="#009E73", label="入模蒸发压力")
        right.set_ylabel("蒸发压力 [MPa]")
        axes[2].set(ylabel="频率 [Hz]", title="c  连续调频过程；无稳定段筛选")
        axes[2].legend(loc="upper left", fontsize=8)
        right.legend(loc="upper right", fontsize=8)
        axes[3].plot(
            x, frame.actual_water_temperature_setpoint, color="#333333", label="实际设定水温"
        )
        axes[3].plot(
            x,
            frame.water_temperature_setpoint,
            color="#7B2CBF",
            ls="--",
            label="入模设定水温：历史60秒中位数",
        )
        axes[3].set(ylabel="设定水温 [°C]", title="d  区分用户调温与单纯调频")
        axes[3].legend(fontsize=8)
        axes[4].axhline(0, color="#777777", lw=0.6)
        axes[4].set(
            ylabel="成本项引起的 ΔCOP / 1 s",
            xlabel="相对所选30秒调频区间起点 [min]",
            title="e  精确分解成本变化引起的逐秒跳变",
        )
        axes[4].legend(fontsize=8)
        for ax in axes:
            ax.axvline(0, color="#888888", lw=0.6, ls=":")
            ax.grid(alpha=0.15)
        fig.suptitle(name, fontsize=12)
        save_publication(
            fig,
            output,
            name,
            "问题：调频时未来成本预测是否制造COP跳变？证据：共享特征、积分及模型的1秒回放，与保存时刻对齐校验。\n"
            "零成本仅是诊断参照；新旧循环边界不同。成本项跳变减小不能单独证明物理成本更准确；热量边界仍沿用历史定义。",
        )


def match_decision_rgb_images(
    metadata: pd.DataFrame,
    images: pd.DataFrame,
    targets: Mapping[str, object],
    *,
    max_offset: pd.Timedelta = RGB_PANEL_MAX_OFFSET,
) -> pd.DataFrame:
    """Match the nearest physical front frame to each causal decision time.

    Metadata is kept separate from the physically materialized images so that a
    missing local file is reported instead of silently being treated as missing
    metadata.  The same two-minute limit as the existing RGB panel is used.
    """
    metadata = metadata.loc[
        metadata.get("camera_role", pd.Series(dtype=str)).astype(str).eq("front")
    ].copy()
    if not metadata.empty:
        metadata["image_time"] = pd.to_datetime(metadata["image_time"], errors="coerce")
        metadata = metadata.dropna(subset=["image_time"]).sort_values(
            ["image_time", "file_name"], kind="stable"
        )
    physical = images.loc[
        images.get("camera_role", pd.Series(dtype=str)).astype(str).eq("front")
    ].copy()
    physical_by_name = {
        str(row["file_name"]): Path(cast(Any, row["path"]))
        for _, row in physical.iterrows()
        if "file_name" in row and "path" in row
    }
    rows: list[dict[str, object]] = []
    for target_type in ("rb", "optimal"):
        raw_target = targets.get(target_type)
        target_time = pd.to_datetime(raw_target, errors="coerce")
        target_time = pd.NaT if pd.isna(target_time) else pd.Timestamp(target_time)
        if pd.isna(target_time):
            rows.append(
                {
                    "target_type": target_type,
                    "target_time": pd.NaT,
                    "image_time": pd.NaT,
                    "offset_seconds": np.nan,
                    "file_name": "",
                    "image_path": "",
                    "available": False,
                    "status": ("rb_right_censored" if target_type == "rb" else "no_valid_optimal"),
                }
            )
            continue
        if metadata.empty:
            rows.append(
                {
                    "target_type": target_type,
                    "target_time": target_time,
                    "image_time": pd.NaT,
                    "offset_seconds": np.nan,
                    "file_name": "",
                    "image_path": "",
                    "available": False,
                    "status": "front_metadata_missing",
                }
            )
            continue
        distance = (metadata["image_time"] - target_time).abs()
        nearest = metadata.loc[distance.idxmin()]
        image_time = pd.Timestamp(nearest["image_time"])
        offset_seconds = float((image_time - target_time).total_seconds())
        file_name = str(nearest.get("file_name", ""))
        path = physical_by_name.get(file_name)
        if abs(offset_seconds) > max_offset.total_seconds():
            status = "offset_exceeds_2min"
            available = False
            image_path = ""
        elif path is None or not path.is_file():
            status = "physical_image_missing"
            available = False
            image_path = ""
        else:
            status = "matched"
            available = True
            image_path = str(path)
        rows.append(
            {
                "target_type": target_type,
                "target_time": target_time,
                "image_time": image_time,
                "offset_seconds": offset_seconds,
                "file_name": file_name,
                "image_path": image_path,
                "available": available,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def cost_curve_optimal_time(
    cost_curve: pd.DataFrame,
    origin: pd.Timestamp,
    stage_spans: list[tuple[str, float, float]],
) -> pd.Timestamp | None:
    """Return the same eligible minimum used by ``_plot_cost_panel``."""
    curve = cost_curve.copy()
    if "candidate_defrost_time" not in curve:
        return None
    curve["candidate_defrost_time"] = pd.to_datetime(
        curve["candidate_defrost_time"], errors="coerce"
    )
    curve["minutes"] = (curve["candidate_defrost_time"] - origin).dt.total_seconds() / 60.0
    metric = "inverse_cop" if "inverse_cop" in curve else "renewal_cost_kw"
    if metric not in curve:
        return None
    curve[metric] = pd.to_numeric(curve[metric], errors="coerce")
    frost_spans = [
        (left, right) for stage, left, right in stage_spans if stage == "frost_development"
    ]
    is_frost = pd.Series(False, index=curve.index)
    for left, right in frost_spans:
        is_frost |= curve["minutes"].ge(left) & curve["minutes"].le(right)
    curve = curve.loc[is_frost & curve["minutes"].notna()].copy()
    eligible = (
        curve["optimization_eligible"].fillna(False).astype(bool)
        if "optimization_eligible" in curve
        else pd.Series(True, index=curve.index)
    )
    curve.loc[~eligible, metric] = np.nan
    values = curve.loc[eligible, metric].dropna()
    if values.empty:
        return None
    return pd.Timestamp(curve.loc[values.idxmin(), "candidate_defrost_time"])


def _plot_effective_cop_panel(
    axis: Any,
    curve: pd.DataFrame,
    origin: pd.Timestamp,
    stage_spans: list[tuple[str, float, float]],
    *,
    full_near_sets: bool = False,
    connected_near_sets: bool = False,
    observations=None,
) -> Any:
    """Draw the effective-COP panel shared by publication and trigger figures."""
    curve = curve.copy()
    if observations is not None:
        curve.attrs["original_missing_spans"] = original_missing_spans(
            observations,
            origin,
            columns=(
                "power_total",
                "heating_capacity",
                "compressor_frequency",
                "ambient_temperature",
                "coil_temperature",
            ),
        )
    plot_objectives(axis, curve, origin, stage_spans, _shade_cycle_stages, metrics=("cycle_cop",))
    outside = curve.get("cycle_cop_uses_model_extrapolation", ~curve.cycle_cop_eligible.eq(True))
    if outside.eq(True).any():
        axis.plot([], [], color=COP_COLOR, linestyle="--", label="Ridge extrapolation")
    if curve.attrs.get("original_missing_spans"):
        axis.plot([], [], color="0.6", linestyle="-", label="Interpolated inputs")
    axis.set_ylabel("Effective cycle COP [-]", fontsize=8)
    relative_axis = axis.twinx()
    plot_normalized(relative_axis, curve, origin, metrics=("cycle_cop",))
    relative_axis.spines["right"].set_visible(True)
    relative_axis.spines["right"].set_color("black")
    relative_axis.tick_params(axis="y", colors="black")
    relative_axis.yaxis.label.set_color("black")
    relative_axis.get_legend().remove()
    first = curve.iloc[0]
    for percent, alpha in ((5, 0.07), (2, 0.10), (1, 0.14)):
        left = pd.to_datetime(first.get(f"cycle_cop_basin_{percent}pct_start"), errors="coerce")
        right = pd.to_datetime(first.get(f"cycle_cop_basin_{percent}pct_end"), errors="coerce")
        spans = [(left, right)] if pd.notna(left) and pd.notna(right) else []
        if full_near_sets:
            times = pd.to_datetime(curve.candidate_defrost_time)
            eligible = curve.cycle_cop_eligible.eq(True)
            peak = curve.loc[eligible, "cycle_cop"].max()
            near = eligible & curve.cycle_cop.ge((1 - percent / 100) * peak)
            starts = ~near.shift(fill_value=False) | times.diff().gt(pd.Timedelta(seconds=30))
            spans = [
                (times.loc[g.index[0]], times.loc[g.index[-1]])
                for _, g in curve.loc[near].groupby(starts.cumsum()[near])
            ]
            if connected_near_sets and spans:
                spans = [(spans[0][0], spans[-1][1])]
        for span_index, (left, right) in enumerate(spans):
            axis.axvspan(
                (left - origin).total_seconds() / 60,
                (right - origin).total_seconds() / 60,
                color=COP_COLOR,
                alpha=alpha,
                label=(
                    (
                        f"连片近优 ({percent}%)"
                        if connected_near_sets
                        else f"Within {percent}% of best COP"
                    )
                    if span_index == 0
                    else "_nolegend_"
                ),
            )
    candidate_times = pd.to_datetime(curve["candidate_defrost_time"], errors="coerce")
    eligible = curve["cycle_cop_eligible"].fillna(False).astype(bool)
    for field, color in (("t_star", OPTIMAL_COLOR), ("t_RB", RB_COLOR)):
        value = pd.to_datetime(first.get(field), errors="coerce")
        selected = curve.loc[candidate_times.eq(value) & eligible]
        if len(selected):
            axis.scatter(
                (pd.to_datetime(selected.candidate_defrost_time) - origin).dt.total_seconds() / 60,
                selected.cycle_cop,
                color=color,
                marker="D" if field == "t_star" else "o",
                s=26,
                zorder=5,
            )
    if not eligible.any():
        axis.text(
            0.02,
            0.5,
            str(first.get("cycle_status", "no supported reference")).replace("_", " "),
            transform=axis.transAxes,
            fontsize=8,
        )
    axis.legend(frameon=False, fontsize=7, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01))
    return relative_axis


def _plot_cop_cost_comparison(cost_axis, cop_axis, near_axis, curves, origin, stage_spans):
    """One colour per independent cost recipe, including its exact 1% components."""
    colors = ("#555555", "#8B7FA5", COP_COLOR, "#C18457")
    limits = []
    for lane, ((label, curve), color) in enumerate(zip(curves.items(), colors, strict=True)):
        x = (pd.to_datetime(curve.candidate_defrost_time) - origin).dt.total_seconds() / 60
        valid = curve.cycle_cop_eligible.eq(True)
        outside = curve.cycle_cop_uses_model_extrapolation.eq(True)
        for axis, field in ((cost_axis, "cost_over_heating_kwh"), (cop_axis, "cycle_cop")):
            _plot_supported_curve(
                axis,
                x,
                curve[field].where(valid),
                valid,
                color=color,
                label=label,
                interpolate=False,
                extrapolated=outside,
            )
        limits.extend(curve.loc[valid, "cycle_cop"].tolist())
        selected = curve.is_selected.eq(True)
        cop_axis.scatter(
            x[selected],
            curve.loc[selected, "cycle_cop"],
            s=16,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            zorder=4,
        )
        near = curve.near_optimal.eq(True)
        starts = ~near.shift(fill_value=False) | pd.to_datetime(
            curve.candidate_defrost_time
        ).diff().gt(pd.Timedelta(seconds=30))
        for _, component in curve.loc[near].groupby(starts.cumsum()[near]):
            left, right = x.loc[component.index[[0, -1]]]
            near_axis.plot([left, right], [lane, lane], color=color, lw=4, solid_capstyle="butt")
            if left == right:
                near_axis.scatter([left], [lane], s=9, color=color)
        near_axis.scatter(
            x[selected],
            np.full(selected.sum(), lane),
            s=18,
            marker="D",
            color=color,
            edgecolors="white",
            linewidths=0.4,
            zorder=4,
        )
    for axis in (cost_axis, cop_axis):
        _shade_cycle_stages(axis, stage_spans, [])
        axis.grid(axis="x", alpha=0.12)
        axis.tick_params(labelsize=6.5)
    cost_axis.set_ylabel("Cost beyond heating [kWh]", fontsize=8)
    cost_axis.set_ylim(bottom=0)
    cost_axis.legend(
        frameon=False,
        fontsize=6.5,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        columnspacing=1.4,
    )
    cop_axis.set_ylabel("Effective cycle COP [-]", fontsize=8)
    if limits:
        peak = max(limits)
        cop_axis.set_ylim(0.75 * peak, 1.025 * peak)
    near_axis.set(
        yticks=range(len(curves)),
        yticklabels=["Zero prep+D", "Mean prep+D", "New Ridge", "Old Ridge"],
        ylim=(-0.5, len(curves) - 0.5),
    )
    near_axis.invert_yaxis()
    near_axis.set_ylabel("Exact 1% set", fontsize=8)
    near_axis.tick_params(labelsize=6.5)
    near_axis.spines["left"].set_visible(False)
    near_axis.grid(axis="x", alpha=0.12)


def render_decision_publication(
    cycle_frame: pd.DataFrame,
    cycle_record: Mapping[str, object],
    cost_curve: pd.DataFrame,
    decision_images: Mapping[str, Mapping[str, object]],
    output_path: Path,
    *,
    optimal_label: str = "Economic optimum",
    cost_label: str = "Cycle inverse COP [-]",
    full_candidate_domain: bool = False,
    display_metric: str | None = None,
    minimum_label: str = "Minimum",
    minimum_support_label: str | None = None,
    parallel_curve: pd.DataFrame | None = None,
    figure_note: str | None = None,
    full_near_sets: bool = False,
    connected_near_sets: bool = False,
    comparison_curves: Mapping[str, pd.DataFrame] | None = None,
    frost_frames: pd.DataFrame | None = None,
) -> None:
    """Render two decision frames above the shared historical or selection panels."""
    frame = cycle_frame.sort_values("timestamp", kind="stable").copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    origin = pd.Timestamp(frame["timestamp"].min())
    minutes = (frame["timestamp"] - origin).dt.total_seconds() / 60.0
    stage_spans = _stage_spans(frame, minutes)
    duration = max(float(minutes.max()), 1.0)
    boundaries = cycle_record.get("boundaries")
    boundaries = boundaries if isinstance(boundaries, Mapping) else cycle_record
    stable = pd.to_datetime(boundaries.get("stable_heating_start"), errors="coerce")
    stable = pd.NaT if pd.isna(stable) else pd.Timestamp(stable)
    parallel = parallel_curve is not None
    effective = (
        parallel
        and "algorithm" in parallel_curve
        and parallel_curve.algorithm.isin(["effective_cop", "effective_cop_without_trunc"]).all()
    )
    show_frost = effective and frost_frames is not None and comparison_curves is None
    show_frost_images = (
        show_frost and "image_path" in frost_frames and frost_frames.image_path.notna().any()
    )
    figure = plt.figure(
        figsize=(7.2, 10.2)
        if comparison_curves is not None
        else (9, 11.2 + 4.8 * show_frost + 4.5 * show_frost_images)
        if effective
        else (10.4, 16.8)
        if parallel
        else (7.2, 8.25),
        dpi=300,
    )
    grid = figure.add_gridspec(
        6
        if comparison_curves is not None or show_frost
        else 5
        if effective
        else 6
        if parallel
        else 4,
        2,
        height_ratios=[2.6, 0.9, 0.9, 1.1, 1.8, 0.65]
        if comparison_curves is not None
        else [3.0, 1.1, 1.1, 1.1, 2.0] + ([3.6 if show_frost_images else 1.2] if show_frost else [])
        if effective
        else [2.15, 0.92, 1.05, 1.45, 1.05, 10.0]
        if parallel
        else [2.15, 0.92, 1.05, 1.08],
        hspace=0.62 if effective else 0.42 if parallel else 0.62,
        wspace=0.08,
    )
    image_axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    image_specs = (("rb", "Hypothetical RB trigger"), ("optimal", optimal_label))
    for axis, (target_type, label) in zip(image_axes, image_specs, strict=True):
        info = decision_images.get(target_type, {})
        _plot_decision_image(axis, info, label, origin, stable)

    panel_axes = [
        figure.add_subplot(grid[1, :]),
        figure.add_subplot(grid[2, :]),
        figure.add_subplot(grid[3, :]),
    ]
    lower_grid = (
        grid[4:6, :].subgridspec(
            2, 1, height_ratios=[2.0, 3.6 if show_frost_images else 1.2], hspace=0.16
        )
        if show_frost
        else None
    )
    if effective:
        panel_axes.append(figure.add_subplot(lower_grid[0] if show_frost else grid[4, :]))
    cop_axis = panel_axes[-1] if effective else panel_axes[2]
    if parallel and not effective:
        panel_axes.extend([figure.add_subplot(grid[4, :]), figure.add_subplot(grid[5, :])])
    _plot_cycle_panel(
        panel_axes[0],
        frame,
        minutes,
        ("cop", "water_cop"),
        "COP [-]",
        stage_spans,
        [],
    )
    _plot_cycle_panel(
        panel_axes[1],
        frame,
        minutes,
        ("water_in_temperature", "water_out_temperature", "water_temperature_setpoint"),
        "Water temperature [degC]",
        stage_spans,
        [],
    )
    if effective and "reference_outlet_temperature" in parallel_curve:
        tref = parallel_curve.reference_outlet_temperature.iloc[0]
        if pd.notna(tref):
            panel_axes[1].axhline(
                tref, color="#333333", ls=":", lw=1, label=f"Tref = {tref:.2f} °C (recovery end)"
            )
            panel_axes[1].legend(fontsize=6, frameon=False, loc="best")
    if comparison_curves is not None:
        near_axis = figure.add_subplot(grid[5, :])
        _plot_cop_cost_comparison(
            panel_axes[2], cop_axis, near_axis, comparison_curves, origin, stage_spans
        )
        panel_axes.append(near_axis)
    elif effective:
        costs = parallel_curve.copy()
        costs.attrs["original_missing_spans"] = original_missing_spans(
            frame,
            origin,
            columns=("compressor_frequency", "ambient_temperature", "coil_temperature"),
        )
        if "defrost_event_electricity_prediction_available" in costs:
            costs["defrost_event_electricity_kwh"] = costs.defrost_event_electricity_kwh.where(
                costs.defrost_event_electricity_prediction_available.eq(True)
            )
        candidate_minutes = (
            pd.to_datetime(costs.candidate_defrost_time) - origin
        ).dt.total_seconds() / 60
        _plot_cycle_panel(
            panel_axes[2],
            costs,
            candidate_minutes,
            ("defrost_event_electricity_kwh",),
            "Predicted defrost cost [kWh]",
            stage_spans,
            [],
        )
        axis = cop_axis
        relative_axis = _plot_effective_cop_panel(
            axis,
            parallel_curve,
            origin,
            stage_spans,
            full_near_sets=full_near_sets,
            connected_near_sets=connected_near_sets,
            observations=frame,
        )
        if show_frost:
            from plots.frost_state import _plot_frost_reference

            frost_axis = figure.add_subplot(lower_grid[1], sharex=cop_axis)
            frost = frost_frames.sort_values("image_time").copy()
            frost.image_time = pd.to_datetime(frost.image_time)
            frost = frost.loc[
                frost.image_time.between(origin, origin + pd.Timedelta(minutes=duration))
            ]
            frost = frost.rename(columns={"image_time": "candidate_defrost_time"})
            _shade_cycle_stages(frost_axis, stage_spans, [])
            for _, part in frost.groupby(
                frost.candidate_defrost_time.diff().gt(pd.Timedelta(seconds=120)).cumsum()
            ):
                _plot_frost_reference(frost_axis, part, origin, metric="visual_frost_score")
            frost_axis.set(ylabel="Frost state [-]\nHarmony + Slingshot", ylim=(-0.02, 1.02))
            if show_frost_images:
                frost_axis.set(xlim=(0, duration), ylim=(-0.02, 2.02), yticks=np.linspace(0, 1, 6))
                _plot_frost_images(frost_axis, frost, origin, output_path)
            if frost.visual_frost_score.notna().sum() == 0:
                frost_axis.text(
                    0.02,
                    0.5,
                    "No native frost-state observations",
                    transform=frost_axis.transAxes,
                    fontsize=8,
                )
            panel_axes.append(frost_axis)
    else:
        curve = cost_curve if cost_curve is not None else pd.DataFrame()
        _plot_cost_panel(
            cop_axis,
            curve,
            origin,
            stage_spans,
            cost_label=cost_label,
            full_candidate_domain=full_candidate_domain,
            display_metric=display_metric,
            minimum_label=minimum_label,
            minimum_support_label=minimum_support_label,
        )
    time_axes = panel_axes if effective else panel_axes[:4] if parallel else panel_axes
    for axis in time_axes:
        axis.set_xlim(0.0, duration)
    point_curves = None
    if effective and comparison_curves is None:
        point_curves = {
            panel_axes[0]: (
                frame.timestamp,
                frame.get("cop", pd.Series(np.nan, index=frame.index)),
            ),
            panel_axes[1]: (
                frame.timestamp,
                frame.get("water_out_temperature", pd.Series(np.nan, index=frame.index)),
            ),
            panel_axes[2]: (costs.candidate_defrost_time, costs.defrost_event_electricity_kwh),
            cop_axis: (parallel_curve.candidate_defrost_time, parallel_curve.cycle_cop),
        }
        if show_frost:
            point_curves[frost_axis] = (frost.candidate_defrost_time, frost.visual_frost_score)
    _plot_decision_markers(
        time_axes,
        decision_images,
        origin,
        optimal_label,
        target_times=parallel,
        legend_axis=cop_axis if show_frost else None,
        point_curves=point_curves,
    )
    if effective and comparison_curves is None:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        axis = cop_axis
        handles, labels = axis.get_legend_handles_labels()
        other_handles, other_labels = relative_axis.get_legend_handles_labels()
        handles.extend(other_handles)
        labels.extend(other_labels)
        unique = dict(zip(labels, handles))
        labels, handles = list(unique), list(unique.values())
        for label, color, marker, style in (
            ("RB defrost trigger", RB_COLOR, "o", "--"),
            ("COP-optimal defrost trigger", OPTIMAL_COLOR, "D", "-."),
        ):
            if label in labels:
                handles[labels.index(label)] = Line2D(
                    [], [], color=color, marker=marker, linestyle=style, markersize=4, linewidth=1
                )
        axis.legend(
            handles,
            labels,
            frameon=False,
            prop={"family": _RGB_PANEL_FONT, "size": 6.7},
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncol=3,
            columnspacing=1.1,
        )
        stage_legend = figure.legend(
            handles=[
                Patch(
                    facecolor=(
                        "#76528F"
                        if stage == "defrost_preparation"
                        else _STAGE_COLORS.get(stage, "#D9DEE5")
                    ),
                    alpha=0.20 if stage == "defrost_preparation" else 0.10,
                    label=_display_label(stage),
                )
                for stage in dict.fromkeys(stage for stage, _, _ in stage_spans)
            ],
            loc="lower left" if show_frost else "lower center",
            bbox_to_anchor=(0.12, 0.95) if show_frost else (0.5, 0.015),
            borderaxespad=0,
            ncol=4,
            frameon=False,
            fontsize=7,
        )
    labels = cop_axis.get_legend_handles_labels()
    if labels[0] and not parallel:
        cop_axis.legend(
            *labels,
            frameon=False,
            fontsize=6.5,
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncol=min(len(labels[0]), 4),
        )
    cycle_name = str(cycle_record.get("cycle_name", cycle_record.get("cycle_id", "Cycle")))
    from dataset_tools.cycle_metadata import cycle_operating_condition

    condition = cycle_operating_condition(frame)
    if condition:
        cycle_name += " | " + condition
    if comparison_curves is not None:
        cycle_name += " | Cost definition comparison"
    elif effective:
        cycle_name += " | Refrigerant effective COP | preparation: " + str(
            parallel_curve.preparation_heat.iloc[0]
        )
    title = figure.suptitle(cycle_name, x=0.08, ha="left", fontsize=10, fontweight="bold")
    if figure_note and not effective:
        figure.text(0.12, 0.045, figure_note, fontsize=7, fontfamily=_RGB_PANEL_FONT)
    (
        near_axis
        if comparison_curves is not None
        else panel_axes[-1]
        if effective
        else panel_axes[3 if parallel else 2]
    ).set_xlabel("Time from cycle start [min]", fontsize=8)
    figure.subplots_adjust(
        left=0.12 if parallel else 0.14,
        right=0.98
        if comparison_curves is not None
        else 0.88
        if effective
        else 0.82
        if parallel
        else 0.98,
        bottom=0.055 if comparison_curves is not None else 0.10 if effective else 0.06,
        top=0.92,
    )
    if show_frost:
        # Pack by the visible labels and legends, while preserving every panel's size.
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        groups = [image_axes] + [
            [axis, relative_axis] if axis is cop_axis else [axis] for axis in panel_axes
        ]
        bounds = [
            (
                min(axis.get_tightbbox(renderer).y0 for axis in group),
                max(axis.get_tightbbox(renderer).y1 for axis in group),
            )
            for group in groups
        ]
        gap = max(8 * figure.dpi / 72, bounds[-2][0] - bounds[-1][1])
        old_bottom = panel_axes[-1].get_position().y0
        previous_bottom = bounds[0][0]
        for group, (bottom, top) in zip(groups[1:], bounds[1:]):
            shift = (previous_bottom - gap - top) / figure.bbox.height
            for axis, box in [(axis, axis.get_position().frozen()) for axis in group]:
                axis.set_position([box.x0, box.y0 + shift, box.width, box.height])
            previous_bottom = bottom + shift * figure.bbox.height
        removed = (panel_axes[-1].get_position().y0 - old_bottom) * figure.get_figheight()
        old_height = figure.get_figheight()
        positions = [(axis, axis.get_position().frozen()) for axis in figure.axes]
        figure.set_size_inches(figure.get_figwidth(), old_height - removed)
        for axis, box in positions:
            axis.set_position(
                [
                    box.x0,
                    (box.y0 * old_height - removed) / (old_height - removed),
                    box.width,
                    box.height * old_height / (old_height - removed),
                ]
            )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        header_gap = 8 * figure.dpi / 72
        image_top = max(axis.get_tightbbox(renderer).y1 for axis in image_axes)
        stage_legend.set_bbox_to_anchor(
            (image_axes[0].get_position().x0, (image_top + header_gap) / figure.bbox.height),
            transform=figure.transFigure,
        )
        legend_top = stage_legend.get_window_extent(renderer).y1
        title.set_position(
            (image_axes[0].get_position().x0, (legend_top + header_gap) / figure.bbox.height)
        )
        title.set_verticalalignment("bottom")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if comparison_curves is not None:
        save_publication(figure, output_path.parent, output_path.stem, "", layout=False, dpi=300)
    else:
        figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_frost_images(axis, frost, origin, output_path):
    """Place native photos directly above their observations with equal vertical leaders."""
    available = frost.loc[frost.image_path.notna() & frost.visual_frost_score.notna()].copy()
    available = available.loc[available.image_path.map(lambda path: Path(path).is_file())]
    duration = axis.get_xlim()[1] - axis.get_xlim()[0]
    width, height, leader = 0.14 * duration, 0.70, 0.24
    minutes = (pd.to_datetime(available.candidate_defrost_time) - origin).dt.total_seconds() / 60
    available = available.loc[minutes.between(width / 2, duration - width / 2)]
    if available.empty:
        return
    times = pd.to_datetime(available.candidate_defrost_time)
    seconds = (times - times.iloc[0]).dt.total_seconds().to_numpy()
    indices = np.unique(
        [
            np.abs(seconds - target).argmin()
            for target in np.linspace(0, seconds[-1], min(5, len(available)))
        ]
    )
    # Keep true timestamps; omit crowded thumbnails rather than shifting photographs.
    spaced = []
    for index in indices:
        if not spaced or seconds[index] - seconds[spaced[-1]] >= width * 60 * 1.15:
            spaced.append(index)
    selected = available.iloc[spaced]
    trace_minutes = (pd.to_datetime(frost.candidate_defrost_time) - origin).dt.total_seconds() / 60
    for row in selected.itertuples(index=False):
        minute = (row.candidate_defrost_time - origin).total_seconds() / 60
        nearby = frost.loc[
            trace_minutes.between(minute - width / 2, minute + width / 2), "visual_frost_score"
        ]
        leader = max(leader, float(nearby.max()) - row.visual_frost_score + 0.06)
    axis.set_ylim(
        axis.get_ylim()[0],
        max(axis.get_ylim()[1], float(selected.visual_frost_score.max()) + leader + height + 0.08),
    )
    for row in selected.itertuples(index=False):
        minute = (row.candidate_defrost_time - origin).total_seconds() / 60
        bottom = row.visual_frost_score + leader
        photo = axis.inset_axes(
            [minute - width / 2, bottom, width, height], transform=axis.transData, zorder=6
        )
        photo.imshow(np.rot90(plt.imread(row.image_path)))
        photo.set_anchor("S")  # Preserve image proportions and the leader's bottom endpoint.
        photo.set_xticks([])
        photo.set_yticks([])
        for spine in photo.spines.values():
            spine.set_color("#7B8790")
            spine.set_linewidth(0.6)
        axis.plot(minute, row.visual_frost_score, "o", color="#435663", markersize=3, zorder=5)
        axis.plot(
            [minute, minute],
            [row.visual_frost_score, bottom],
            color="#7B8790",
            linewidth=0.6,
            zorder=4,
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    selected.rename(columns={"candidate_defrost_time": "image_time"}).to_csv(
        Path(output_path).with_suffix(".frost_images.csv"), index=False
    )


def _plot_decision_image(
    axis: Any,
    info: Mapping[str, object],
    label: str,
    origin: pd.Timestamp,
    stable: pd.Timestamp,
) -> None:
    path_value = info.get("image_path", info.get("path", ""))
    path = Path(str(path_value)) if path_value else None
    available = bool(info.get("available")) and path is not None and path.is_file()
    axis.set_xticks([])
    axis.set_yticks([])
    if available:
        axis.imshow(np.rot90(plt.imread(path)), aspect="auto")
    else:
        axis.set_facecolor("#ECEFF1")
        status = str(info.get("status", "unavailable")).replace("_", " ")
        axis.text(
            0.5,
            0.5,
            f"RGB unavailable\n{status}",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#59636E",
        )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(RB_COLOR if label.startswith("RB") else OPTIMAL_COLOR)
        spine.set_linewidth(1.0)
    target_time = pd.to_datetime(info.get("target_time"), errors="coerce")
    image_time = pd.to_datetime(info.get("image_time"), errors="coerce")
    target_text = (
        f"target {pd.Timestamp(target_time):%H:%M:%S}"
        if not pd.isna(target_time)
        else "target unavailable"
    )
    if not pd.isna(image_time):
        title = (
            f"{label} · {target_text}\n"
            f"image {pd.Timestamp(image_time):%H:%M:%S} · "
            f"Δ {float(info.get('offset_seconds', np.nan)):+.1f} s"
        )
    else:
        title = f"{label} · {target_text}\nRGB unavailable"
    axis.set_title(title, fontsize=7.2, pad=5, loc="left")


def _plot_decision_markers(
    axes: list[Any],
    decision_images: Mapping[str, Mapping[str, object]],
    origin: pd.Timestamp,
    optimal_label: str,
    *,
    target_times: bool = False,
    legend_axis: Any = None,
    point_curves: Mapping[Any, tuple[pd.Series, pd.Series]] | None = None,
) -> None:
    markers = (
        ("rb", "RB RGB frame", RB_COLOR),
        ("optimal", f"{optimal_label} RGB frame", OPTIMAL_COLOR),
    )
    for target_type, label, color in markers:
        info = decision_images.get(target_type, {})
        if target_times:
            label = "RB defrost trigger" if target_type == "rb" else "COP-optimal defrost trigger"
        marker_time = pd.to_datetime(
            info.get("target_time") if target_times else info.get("image_time"),
            errors="coerce",
        )
        if (not target_times and not bool(info.get("available"))) or pd.isna(marker_time):
            continue
        x = (pd.Timestamp(marker_time) - origin).total_seconds() / 60.0
        for axis_index, axis in enumerate(axes):
            ymax = 1.0
            if point_curves is not None:
                times, values = point_curves[axis]
                minutes = (pd.to_datetime(times) - origin).dt.total_seconds().to_numpy() / 60
                if not len(minutes):
                    continue
                y = np.interp(
                    x, minutes, np.asarray(values, dtype=float), left=np.nan, right=np.nan
                )
                index = np.searchsorted(minutes, x)
                if "Frost state" in axis.get_ylabel() and 0 < index < len(minutes):
                    if minutes[index] != x and minutes[index] - minutes[index - 1] > 2:
                        continue  # Do not invent a frost state across missing-frame gaps.
                low, high = axis.get_ylim()
                if not np.isfinite(y) or y < low:
                    continue
                ymax = min((y - low) / (high - low), 1.0)
            axis.axvline(
                x,
                ymax=ymax,
                color=color,
                linestyle="--" if target_type == "rb" else "-.",
                linewidth=0.9,
                label=label
                if (axis is legend_axis if legend_axis is not None else axis_index == len(axes) - 1)
                else "_nolegend_",
                zorder=4,
            )


def _plot_cost_panel(  # noqa: C901
    axis: Any,
    cost_curve: pd.DataFrame,
    origin: pd.Timestamp,
    stage_spans: list[tuple[str, float, float]],
    *,
    cost_label: str = "Cycle inverse COP [-]",
    full_candidate_domain: bool = False,
    display_metric: str | None = None,
    minimum_label: str = "Minimum",
    minimum_support_label: str | None = None,
) -> None:
    """Plot empirical cost only where the cycle is in frost development."""
    curve = cost_curve.copy()
    curve["candidate_defrost_time"] = pd.to_datetime(
        curve["candidate_defrost_time"], errors="coerce"
    )
    curve["minutes"] = (curve["candidate_defrost_time"] - origin).dt.total_seconds() / 60.0
    metric = "inverse_cop" if "inverse_cop" in curve else "renewal_cost_kw"
    curve[metric] = pd.to_numeric(curve[metric], errors="coerce")
    frost_spans = [
        (left, right) for stage, left, right in stage_spans if stage == "frost_development"
    ]
    is_frost = pd.Series(False, index=curve.index)
    for left, right in frost_spans:
        is_frost |= curve["minutes"].ge(left) & curve["minutes"].lt(right)
    curve = curve.loc[
        curve["minutes"].notna() & (True if full_candidate_domain else is_frost)
    ].copy()
    curve["_raw_metric"] = curve[metric]
    if display_metric is not None:
        if display_metric not in curve:
            raise ValueError(f"missing display-only metric: {display_metric}")
        curve[display_metric] = pd.to_numeric(curve[display_metric], errors="coerce")
    eligible = (
        curve["optimization_eligible"].fillna(False).astype(bool)
        if "optimization_eligible" in curve
        else pd.Series(True, index=curve.index)
    )
    pe_supported = (
        curve["pe_supported"].fillna(False).astype(bool)
        if "pe_supported" in curve
        else curve["support_status"].eq("supported")
        if "support_status" in curve
        else eligible.copy()
    )
    if "model_supported" in curve:
        support_state = curve["model_supported"].astype("boolean")
        model_supported = support_state.eq(True).fillna(False)
        outside_support = support_state.eq(False).fillna(False)
        outside_support_label = "Outside empirical-model support"
    else:
        support_state = None
        model_supported = pe_supported
        outside_support = ~model_supported
        outside_support_label = "Outside Pe support"
    integration_eligible = (
        curve["integration_eligible"].fillna(False).astype(bool)
        if "integration_eligible" in curve
        else eligible | ~pe_supported
    )
    insufficient_integration = pe_supported & ~integration_eligible
    interpolated_gap = (
        curve["candidate_in_interpolated_gap"].fillna(False).astype(bool)
        if "candidate_in_interpolated_gap" in curve
        else pd.Series(False, index=curve.index)
    )
    extrapolated_endpoint = (
        curve["candidate_in_extrapolated_endpoint"].fillna(False).astype(bool)
        if "candidate_in_extrapolated_endpoint" in curve
        else pd.Series(False, index=curve.index)
    )
    curve.loc[~eligible, metric] = np.nan

    _shade_cycle_stages(axis, stage_spans, [])
    if curve.loc[eligible, metric].dropna().empty:
        axis.text(
            0.5,
            0.5,
            "No optimization-eligible candidates",
            transform=axis.transAxes,
            ha="center",
        )
    else:
        axis.plot(
            curve["minutes"],
            curve[metric],
            color="#3775BA",
            linewidth=1.25,
            label="Cycle inverse COP" if metric == "inverse_cop" else "Empirical cost",
        )
        minimum_index = curve.loc[eligible, metric].idxmin()
        minimum_x = float(curve.loc[minimum_index, "minutes"])
        axis.axvline(
            minimum_x,
            color="#E28E2C",
            linewidth=1.05,
            label=minimum_label,
            zorder=4,
        )
        regret = (
            pd.to_numeric(curve["relative_regret"], errors="coerce")
            if "relative_regret" in curve
            else curve[metric] / curve[metric].min() - 1.0
        )
        near = eligible & regret.le(0.01)
        groups = near.ne(near.shift(fill_value=False)).cumsum()
        for segment_index, (_, segment) in enumerate(
            curve.loc[near].groupby(groups[near], sort=False), start=1
        ):
            axis.axvspan(
                float(segment["minutes"].min()),
                float(segment["minutes"].max()),
                color="#E28E2C",
                alpha=0.18,
                label=f"1% near-optimal segment {segment_index}",
                zorder=0.2,
            )
        if "minimum_location" in curve:
            location = str(curve.loc[minimum_index, "minimum_location"]).replace("_", " ")
        else:
            location = (
                "left boundary"
                if minimum_index == curve.index[0]
                else "right boundary"
                if minimum_index == curve.index[-1]
                else "interior"
            )
        if minimum_support_label is not None:
            support_text = minimum_support_label
        elif support_state is not None:
            minimum_support = support_state.loc[minimum_index]
            support_text = (
                "support unknown"
                if pd.isna(minimum_support)
                else "within support"
                if bool(minimum_support)
                else "extrapolated"
            )
        axis.text(
            0.01,
            0.95,
            (
                f"Minimum: {location} · {support_text}"
                if minimum_support_label is not None or support_state is not None
                else f"Minimum: {location}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#4B5563",
        )
        decision_time = pd.to_datetime(
            curve.get("t_star", pd.Series(dtype=object)).iloc[:1], errors="coerce"
        )
        if not decision_time.empty and pd.notna(decision_time.iloc[0]):
            decision_index = (
                (curve["candidate_defrost_time"] - decision_time.iloc[0]).abs().idxmin()
            )
            if decision_index != minimum_index:
                axis.axvline(
                    float(curve.loc[decision_index, "minutes"]),
                    color="#B64A50",
                    linestyle="--",
                    linewidth=1.05,
                    label="Selected decision",
                    zorder=4,
                )
                status = str(curve.get("decision_status", pd.Series(["near-optimal"])).iloc[0])
                axis.text(
                    0.01,
                    0.87,
                    f"Decision: {status.replace('_', ' ')} · "
                    f"+{100 * float(regret.loc[decision_index]):.2f}%",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="#B64A50",
                )

    if display_metric is not None and curve[display_metric].notna().any():
        axis.plot(
            curve["minutes"],
            curve[display_metric],
            color="#A69AA8",
            linestyle="--",
            linewidth=0.8,
            marker=".",
            markersize=2.5,
            alpha=0.75,
            label="Unsupported model extension, display only",
        )

    _plot_cost_quality_markers(
        axis,
        curve,
        outside_support,
        outside_support_label,
        insufficient_integration,
        interpolated_gap,
        extrapolated_endpoint,
    )

    _plot_rb_trigger(axis, curve, origin, metric, eligible)

    preparation = (
        pd.to_datetime(curve["observed_defrost_preparation_start"], errors="coerce").dropna()
        if "observed_defrost_preparation_start" in curve
        else pd.Series(dtype="datetime64[ns]")
    )
    if not preparation.empty:
        axis.axvline(
            (pd.Timestamp(preparation.iloc[0]) - origin).total_seconds() / 60.0,
            color="#777777",
            linewidth=0.9,
            linestyle="--",
            label="Observed preparation",
        )
    else:
        defrost_starts = [left for stage, left, _ in stage_spans if stage == "defrost"]
        if defrost_starts:
            axis.axvline(
                defrost_starts[0],
                color="#777777",
                linewidth=0.9,
                linestyle="--",
                label="Observed defrost",
            )
    axis.set_ylabel(
        cost_label if metric == "inverse_cop" else "Renewal cost [kW-eq.]",
        fontsize=8,
    )
    axis.grid(axis="x", alpha=0.12)
    if axis.lines:
        handle_count = len(axis.get_legend_handles_labels()[0])
        axis.legend(
            frameon=False,
            fontsize=7,
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncol=min(handle_count, 4),
        )


def _plot_cost_quality_markers(
    axis: Any,
    curve: pd.DataFrame,
    outside_support: pd.Series,
    outside_support_label: str,
    insufficient_integration: pd.Series,
    interpolated_gap: pd.Series,
    extrapolated_endpoint: pd.Series,
) -> None:
    valid_metric = curve["_raw_metric"].notna()
    marker_specs = (
        (
            outside_support,
            11,
            "x",
            {"linewidths": 0.65, "color": "#A7ADB3"},
            outside_support_label,
            3,
        ),
        (
            insufficient_integration,
            13,
            "+",
            {"linewidths": 0.75, "color": "#C66A00"},
            "Insufficient integration coverage",
            3,
        ),
        (
            interpolated_gap,
            16,
            "s",
            {"facecolors": "none", "edgecolors": "#9AA0A6", "linewidths": 0.65},
            "Internal-gap interpolation",
            2,
        ),
        (
            extrapolated_endpoint,
            18,
            "D",
            {"facecolors": "none", "edgecolors": "#76528F", "linewidths": 0.7},
            "Endpoint linear extrapolation",
            2,
        ),
    )
    for mask, size, marker, style, label, zorder in marker_specs:
        visible = mask & valid_metric
        if visible.any():
            x = curve.loc[visible, "minutes"]
            y = curve.loc[visible, "_raw_metric"]
        elif mask.any():
            # Keep the diagnostic label when the corresponding cost is NaN.
            x = []
            y = []
        else:
            continue
        axis.scatter(
            x,
            y,
            s=size,
            marker=marker,
            label=label,
            zorder=zorder,
            **style,
        )


def _plot_rb_trigger(
    axis: Any,
    curve: pd.DataFrame,
    origin: pd.Timestamp,
    metric: str,
    eligible: pd.Series,
) -> None:
    """Mark the recorded first RB trigger without changing its causal timestamp."""
    if "rb_status" not in curve or "t_RB" not in curve:
        return
    status = curve["rb_status"].dropna().astype(str)
    rb_times = pd.to_datetime(curve["t_RB"], errors="coerce").dropna()
    if status.empty or status.iloc[0] != "triggered" or rb_times.empty:
        return
    rb_time = pd.Timestamp(rb_times.iloc[0])
    axis.axvline(
        (rb_time - origin).total_seconds() / 60.0,
        color="#2E7D5B",
        linewidth=0.9,
        linestyle="--",
        label="RB trigger",
        zorder=2.5,
    )
    valid_cost = eligible & curve[metric].notna()
    if valid_cost.any():
        nearest = (curve.loc[valid_cost, "candidate_defrost_time"] - rb_time).abs().idxmin()
        if abs(curve.loc[nearest, "candidate_defrost_time"] - rb_time) > pd.Timedelta(minutes=0.51):
            return
        axis.scatter(
            curve.loc[nearest, "minutes"],
            curve.loc[nearest, metric],
            s=18,
            color="#2E7D5B",
            label="Nearest eligible cost",
            zorder=3.5,
        )


if __name__ == "__main__":
    main()
