"""Camera comparison and representative frost photographs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from dataset_tools import DatasetLoader
from image_models.cycle_selection import screen_cv_cycles
from image_models.frost_progression import match_trajectory_frames
from output_layout import OUTPUT
from plots.publication import _RGB_PANEL_FONT, save_publication

matplotlib.use("Agg")


def render_camera_frost_images(source, output, *, layout="full", name="camera_frost_images"):
    """Six complete real photographs, optionally resized directly to square panels."""
    from PIL import Image

    output = Path(output)
    source = source.sort_values("display_order").copy()
    source["rotation_degrees"] = 90
    source["image_layout"] = layout
    ink, grey = "#253D50", "#707B86"
    with plt.rc_context(
        {"font.family": _RGB_PANEL_FONT, "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42}
    ):
        fig = plt.figure(figsize=(183 / 25.4, (39 if layout == "square" else 90) / 25.4))
        for i, row in enumerate(source.itertuples()):
            left = 0.025 + i * 0.16
            with Image.open(row.image_path) as original:
                photo = np.rot90(np.asarray(original.convert("RGB")))
            height, width = photo.shape[:2]
            if layout == "square":
                photo = np.asarray(
                    Image.fromarray(photo).resize((width, width), Image.Resampling.LANCZOS)
                )
            source.loc[
                row.Index,
                [
                    "crop_left_px",
                    "crop_top_px",
                    "crop_width_px",
                    "crop_height_px",
                    "resize_width_px",
                    "resize_height_px",
                ],
            ] = [0, 0, width, height, photo.shape[1], photo.shape[0]]
            axis = fig.add_axes(
                (
                    left,
                    0.25 if layout == "square" else 0.235,
                    0.15,
                    0.15 * 183 / 39 if layout == "square" else 0.54,
                )
            )
            axis.imshow(photo, aspect="equal", interpolation="none")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#DCE2E7")
                spine.set_linewidth(0.5)
            if layout == "full":
                fig.text(
                    left + 0.075,
                    0.805,
                    f"g = {row.reference_grade:.1f}",
                    ha="center",
                    color=grey,
                    fontsize=7,
                )
            fig.text(
                left + 0.075,
                0.09 if layout == "square" else 0.18,
                row.camera_role,
                ha="center",
                color=ink,
                fontsize=8 if layout == "square" else 9,
            )
            plt.imsave(output / f"{name.removesuffix('_images')}_{row.camera_role}.png", photo)
        if layout == "full":
            fig.text(
                0.5, 0.965, "六机位的早期结霜形态", ha="center", va="top", fontsize=13, color=ink
            )
            fig.text(0.025, 0.88, "无霜", ha="left", va="center", fontsize=8, color=grey)
            fig.text(0.975, 0.88, "薄霜", ha="right", va="center", fontsize=8, color=grey)
            fig.add_artist(
                matplotlib.patches.FancyArrowPatch(
                    (0.10, 0.88),
                    (0.90, 0.88),
                    transform=fig.transFigure,
                    arrowstyle="->",
                    mutation_scale=9,
                    lw=0.7,
                    color="#9AAAB5",
                )
            )
            fig.text(
                0.5,
                0.075,
                "霜态参考：" + source.source_method.iloc[0].split(";")[0] + " · 同周期取图",
                ha="center",
                fontsize=6.5,
                color=grey,
            )
        save_publication(fig, output, name, None, layout=False, dpi=600)
    return source


def render_camera_comparison(output, *, coverage_only=False, layout="overview"):
    """Camera coverage, held-out utility and paired experiment-block effects share one renderer."""
    output = Path(output)
    from textwrap import fill

    cameras = ["front", "top", "top_close", "left", "left_close", "extreme"]
    blue, ink, grey = "#6F9EB6", "#253D50", "#707B86"
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    }
    if layout in ("grouped", "diagnostic"):
        summary = pd.read_csv(output / "comparison_summary.csv")
        scopes = {"first_common": "共同集合", "second_common": "跨工况集合"}
        fields = [
            "evaluation_scope",
            "camera_role",
            "seed",
            "experiments",
            "cohort_cycles",
            "rb_paired_cycles",
            "headroom_captured_pct",
        ]
        if layout == "diagnostic":
            fields += ["mean_paired_model_cop", "mean_rb_cop", "mean_optimal_cop"]
        source = summary.loc[summary.evaluation_scope.isin(scopes), fields].copy()
        if layout == "diagnostic":
            # Ratios of paired mean COP match the H_RB finalizer, not means of cycle ratios.
            source["model_loss_pct"] = (
                100
                * (source.mean_optimal_cop - source.mean_paired_model_cop)
                / source.mean_optimal_cop
            )
            source["rb_loss_pct"] = (
                100 * (source.mean_optimal_cop - source.mean_rb_cop) / source.mean_optimal_cop
            )
        order = (
            source.loc[source.evaluation_scope.eq("first_common")]
            .groupby("camera_role")
            .headroom_captured_pct.mean()
            .sort_values(kind="stable")
            .index.tolist()
        )
        source["display_order"] = source.camera_role.map(dict(zip(order, range(len(order)))))
        source["集合"] = source.evaluation_scope.map(scopes)
        name = "camera_test_diagnostics" if layout == "diagnostic" else "camera_core_grouped"
        source.sort_values(["display_order", "evaluation_scope", "seed"]).to_csv(
            output / f"{name}_source.csv", index=False
        )
        # Front follows the same continuous palette as its two neighbours.
        warm = np.array(matplotlib.colors.to_rgb("#C8A08F"))
        cool = np.array(matplotlib.colors.to_rgb(blue))
        colors = np.array([(1 - t) * warm + t * cool for t in np.linspace(0, 1, len(order))])
        light = 0.60 * colors + 0.40
        with plt.rc_context({**style, "font.family": _RGB_PANEL_FONT, "hatch.linewidth": 0.35}):
            if layout == "diagnostic":
                fig = plt.figure(figsize=(183 / 25.4, 160 / 25.4))
                ax = fig.add_axes((0.10, 0.53, 0.88, 0.28))
                baseline = fig.add_axes((0.10, 0.12, 0.36, 0.25))
                window = fig.add_axes((0.64, 0.12, 0.34, 0.25))
            else:
                fig, ax = plt.subplots(figsize=(183 / 25.4, 105 / 25.4))
                fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.78)
            metric = "model_loss_pct" if layout == "diagnostic" else "headroom_captured_pct"
            for column, (scope, label) in enumerate(scopes.items()):
                values = (
                    source.loc[source.evaluation_scope.eq(scope)]
                    .groupby("camera_role")[metric]
                    .mean()
                    .reindex(order)
                )
                bars = ax.bar(
                    np.arange(len(order)) + (column - 0.5) * 0.36,
                    values,
                    width=0.33,
                    color=colors if column == 0 else light,
                    edgecolor=colors,
                    linewidth=0.5,
                    hatch=None if column == 0 else "//",
                    zorder=3,
                )
                for bar, value in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + (0.008 if layout == "diagnostic" else 1.2),
                        f"{value:.3f}" if layout == "diagnostic" else f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=ink,
                    )
            middle = colors.mean(axis=0)
            handles = [
                Rectangle((0, 0), 1, 1, facecolor=middle, edgecolor=middle),
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=0.60 * middle + 0.40,
                    edgecolor=middle,
                    hatch="//",
                    linewidth=0.5,
                ),
            ]
            fig.legend(
                handles,
                list(scopes.values()),
                loc="upper center",
                bbox_to_anchor=(0.54, 0.90),
                ncol=2,
                columnspacing=2.5,
                handlelength=1.8,
                fontsize=9,
            )
            ax.set(
                xticks=np.arange(len(order)),
                xticklabels=order,
                ylim=(0, source.model_loss_pct.max() * 1.22)
                if layout == "diagnostic"
                else (0, 105),
                ylabel="距最优的 COP 损失（%）" if layout == "diagnostic" else r"$H_{RB}$（%）",
            )
            if layout == "grouped":
                ax.set_yticks(np.arange(0, 101, 20))
            else:
                counts = source.groupby(["camera_role", "evaluation_scope"]).rb_paired_cycles.mean()
                ax.set_xticklabels(
                    [
                        f"{camera}\nn={int(counts.loc[camera, 'first_common'])}/"
                        f"{int(counts.loc[camera, 'second_common'])}"
                        for camera in order
                    ]
                )
            ax.tick_params(axis="x", length=0, pad=7, labelsize=8)
            ax.grid(axis="y", color="#E6E8EB", lw=0.5, zorder=0)
            if layout == "diagnostic":
                ax.set_title(
                    "a  六机位的实际损失：越低越好", loc="left", fontsize=9, pad=12, color=ink
                )
                front = (
                    source.loc[source.camera_role.eq("front")]
                    .groupby("evaluation_scope")
                    .mean(numeric_only=True)
                )
                sets = pd.read_csv(output / "camera_near_set_source.csv")
                set_colors = [middle, 0.60 * middle + 0.40]
                for i, scope in enumerate(scopes):
                    row = front.loc[scope]
                    baseline.plot([i, i], [row.model_loss_pct, row.rb_loss_pct], color=grey, lw=1.2)
                    baseline.plot(i, row.rb_loss_pct, "s", ms=4, color=grey)
                    baseline.plot(
                        i, row.model_loss_pct, "o", ms=5, color=set_colors[i], mec=middle, mew=0.7
                    )
                    for value, label in [
                        (row.rb_loss_pct, f"RB {row.rb_loss_pct:.2f}"),
                        (row.model_loss_pct, f"模型 {row.model_loss_pct:.3f}"),
                    ]:
                        baseline.text(i + 0.09, value, label, va="center", fontsize=7, color=ink)
                    part = sets.loc[sets.evaluation_scope.eq(scope)]
                    offsets = np.random.default_rng(42).uniform(-0.13, 0.13, len(part))
                    window.scatter(
                        i + offsets,
                        part.near_minutes,
                        s=9,
                        facecolor=set_colors[i],
                        edgecolor=middle,
                        lw=0.3,
                        alpha=0.65,
                        zorder=3,
                    )
                    mean = part.near_minutes.mean()
                    window.plot([i - 0.18, i + 0.18], [mean, mean], color=ink, lw=1.4, zorder=4)
                    window.text(
                        i,
                        sets.near_minutes.max() + 4,
                        f"均值 {mean:.1f}",
                        ha="center",
                        fontsize=7,
                        color=ink,
                    )
                labels = [
                    f"{label}\n{int(front.loc[scope, 'cohort_cycles'])}周期 · "
                    f"{int(front.loc[scope, 'experiments'])}实验"
                    for scope, label in scopes.items()
                ]
                baseline.set(
                    xticks=[0, 1],
                    xticklabels=labels,
                    xlim=(-0.35, 1.8),
                    ylim=(0, front.rb_loss_pct.max() * 1.18),
                    ylabel="距最优的 COP 损失（%）",
                )
                baseline.axhline(0, color=grey, lw=0.6)
                baseline.set_title(
                    "b  RB 基线离最优更远（front）", loc="left", fontsize=9, pad=12, color=ink
                )
                window.set(
                    xticks=[0, 1],
                    xticklabels=labels,
                    xlim=(-0.4, 1.4),
                    ylim=(0, sets.near_minutes.max() + 12),
                    ylabel="近优时段合计（分钟）",
                )
                window.set_title(
                    "c  近优时间窗有多宽？（front）", loc="left", fontsize=9, pad=12, color=ink
                )
                for axis in [baseline, window]:
                    axis.tick_params(axis="x", length=0, pad=7, labelsize=7)
                    axis.grid(axis="y", color="#E6E8EB", lw=0.5, zorder=0)
                fig.suptitle(
                    r"为什么二测的 $H_{RB}$ 普遍较高？", x=0.54, y=0.975, fontsize=14, color=ink
                )
            else:
                fig.suptitle("不同机位的除霜预测性能", x=0.54, y=0.975, fontsize=14, color=ink)
            save_publication(fig, output, name, None, layout=False, dpi=600)
        return
    cohort = pd.read_csv(output / "camera_cohorts.csv")
    shared_second = False
    if "data_role" in cohort:
        second = cohort.loc[cohort.data_role.eq("second")]
        shared_second = (
            len(second) > 0
            and second.groupby("cycle_name").camera_role.nunique().eq(len(cameras)).all()
        )
        cohort = cohort.loc[cohort.data_role.eq("first")]
    counts = pd.read_csv(output / "cohort_counts.csv").set_index("camera_role").reindex(cameras)
    membership = {
        camera: set(cohort.loc[cohort.camera_role.eq(camera), "cycle_name"]) for camera in cameras
    }
    overlap = pd.DataFrame(
        [[len(membership[a] & membership[b]) for b in cameras] for a in cameras],
        index=cameras,
        columns=cameras,
    )
    overlap.to_csv(output / "pairwise_cycle_overlap.csv")
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            1, 2, figsize=(183 / 25.4, 83 / 25.4), gridspec_kw={"width_ratios": [1, 1.3]}
        )
        y = np.arange(6)
        axes[0].barh(y, counts.cycles, color=blue)
        axes[0].axvline(counts.intersection_cycles.iloc[0], color=ink, ls="--", lw=0.9)
        axes[0].set(
            yticks=y,
            yticklabels=cameras,
            xlabel="Admitted cycles",
            title="a  Camera-specific coverage",
        )
        axes[0].invert_yaxis()
        for i, count in enumerate(counts.cycles):
            axes[0].text(count + 1, i, str(count), va="center", fontsize=7)
        axes[0].set_xlim(0, counts.cycles.max() * 1.2)
        axes[1].imshow(overlap, cmap="Blues", vmin=0, vmax=overlap.to_numpy().max(), aspect="equal")
        axes[1].set(
            xticks=y,
            xticklabels=cameras,
            yticks=y,
            yticklabels=cameras,
            title="b  Shared cycles between cameras",
        )
        axes[1].tick_params(axis="x", rotation=45)
        for i in y:
            for j in y:
                value = overlap.iloc[i, j]
                axes[1].text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > overlap.to_numpy().max() * 0.55 else ink,
                )
        save_publication(
            fig,
            output,
            "camera_coverage",
            fill(
                f"Question: how much comparable data exists? Evidence: authoritative screening per camera. "
                f"All-camera intersection: {int(counts.intersection_cycles.iloc[0])} cycles; dashed line in a. "
                + (
                    "Primary scope: front-defined standard-cycle reference. Additional eligible standard cycles are listed in source data. "
                    if "outside_front_reference" in counts
                    else ""
                )
                + "Unequal camera cohorts require paired evaluation.",
                125,
            ),
            dpi=600,
        )
        if coverage_only:
            return
        summary = pd.read_csv(output / "comparison_summary.csv")
        seeds = sorted(summary.seed.unique())
        seed_evidence = (
            f"Seed {int(seeds[0])}; one fitted model per outer experiment. "
            if len(seeds) == 1
            else f"Dots: {len(seeds)} seeds; ticks: seed means, not independent replicates. "
        )
        two_tests = "first_common" in set(summary.evaluation_scope)
        scopes = (
            ["first_own", "first_common", "second_own", "second_common"]
            if two_tests
            else ["own", "common_test", "common_train_test"]
        )
        titles = (
            [
                "First: own cycles",
                "First: common cycles",
                "Second: own cycles",
                "Second: common cycles",
            ]
            if two_tests
            else ["Own train / own test", "Own train / common test", "Common train / common test"]
        )
        if two_tests and shared_second:
            scopes, titles = (
                [scopes[0], scopes[1], scopes[3]],
                [titles[0], titles[1], "Second: all cameras"],
            )
        paired_scopes = ["first_common", "second_common"] if two_tests else scopes[1:]
        paired_titles = (
            ["First: development LOEO", "Second: independent tests"] if two_tests else titles[1:]
        )
        fig, axes = plt.subplots(
            3, len(scopes), figsize=(183 / 25.4, 175 / 25.4), sharey=True, sharex="row"
        )
        measures = [
            ("gain_vs_rb_pct", "COP gain vs RB (%)"),
            ("headroom_captured_pct", r"$H_{RB}$ captured (%)"),
            (
                "hit_rate_all_pct",
                "Near-optimal hits (%)" if two_tests else "Near-optimal hit rate, all cycles (%)",
            ),
        ]
        for row, (metric, xlabel) in enumerate(measures):
            for col, (scope, title) in enumerate(zip(scopes, titles)):
                ax = axes[row, col]
                table = summary.loc[summary.evaluation_scope.eq(scope)]
                for i, camera in enumerate(cameras):
                    values = table.loc[table.camera_role.eq(camera), metric].dropna()
                    ax.scatter(values, [i] * len(values), s=13, color=blue, alpha=0.75)
                    if len(values) > 1:
                        ax.plot(values.mean(), i, "|", color=ink, ms=11, mew=1.5)
                ax.axvline(0, color=grey, ls="--", lw=0.7)
                ax.set(
                    xlabel=xlabel,
                    title=f"{chr(97 + len(scopes) * row + col)}  {title}",
                    yticks=y,
                    yticklabels=cameras,
                )
        axes[0, 0].invert_yaxis()
        matched_means = (
            summary.loc[summary.evaluation_scope.isin(paired_scopes)]
            .groupby(["evaluation_scope", "camera_role"])
            .headroom_captured_pct.mean()
        )
        save_publication(
            fig,
            output,
            "camera_policy_comparison",
            fill(
                "Question: does the camera ranking survive matched data? Evidence: "
                + (
                    "seven-input LSTM25 FiLM + frozen DINO. "
                    if two_tests
                    else "experiment-held-out LSTM FiLM. "
                )
                + seed_evidence
                + (
                    "First: fixed-recipe LOEO shares standard data with global tuning (development evidence). "
                    "Second: one full-standard weight per camera, untouched later experiments. Common cycles restrict testing only. "
                    if two_tests
                    else ""
                )
                + (
                    "Second own/common results coincide; shown once. "
                    if two_tests and shared_second
                    else ""
                )
                + "COP and H use supported policy-RB pairs; hit rates retain every cycle. Own-cohort differences include sample composition. "
                + (
                    "Matched point estimates differ across cameras; paired intervals assess uncertainty."
                    if matched_means.groupby(level=0).nunique().gt(1).all()
                    else "The matched point estimates do not separate all cameras; paired intervals assess uncertainty."
                ),
                125,
            ),
            dpi=600,
        )
        fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 87 / 25.4), sharex=True)
        below = []
        for ax, scope, title, label in zip(axes, paired_scopes, paired_titles, "ab"):
            effects = pd.read_csv(output / scope / "experiment_bootstrap.csv")
            effects = effects.loc[effects.left.eq("front")].set_index("right").reindex(cameras[1:])
            below.append(set(effects.index[effects.pooled_H_upper_95.lt(0)]))
            for i, (_, row) in enumerate(effects.iterrows()):
                if np.isfinite(row.pooled_H_delta_pp):
                    ax.plot(
                        [row.pooled_H_lower_95, row.pooled_H_upper_95], [i, i], color=blue, lw=1.5
                    )
                    ax.plot(row.pooled_H_delta_pp, i, "o", color=ink, ms=3.5)
            ax.axvline(0, color=grey, ls="--", lw=0.7)
            ax.set(
                xlabel=r"$\Delta H_{RB}$ vs front (percentage points)",
                title=f"{label}  {title}",
                yticks=np.arange(5),
                yticklabels=[
                    f"{camera} (n={int(row.paired_cycles)}, E={int(row.test_experiments)})"
                    for camera, row in effects.iterrows()
                ],
            )
            ax.invert_yaxis()
        save_publication(
            fig,
            output,
            "camera_paired_effects",
            fill(
                "Question: which cameras outperform front? "
                + (
                    f"Single-seed paired effects (seed {int(seeds[0])}); "
                    if len(seeds) == 1
                    else "Seed-averaged paired effects; "
                )
                + "pointwise 95% intervals: "
                "2,000 experiment-block bootstrap draws, fitted models fixed. n=cycles and E=experiments supported by both cameras in every seed. "
                "Exploratory comparisons; zero-crossing intervals do not establish advantage. "
                "Below front in both panels: "
                + (", ".join(c for c in cameras if c in set.intersection(*below)) or "none")
                + ".",
                125,
            ),
            dpi=600,
        )


def select_camera_frost_images(args):
    """One actual view per increasing saved front grade, all from one screened cycle."""
    from dataset_tools.cycle_metadata import image_root
    from image_models.camera_data import CAMERAS
    from plots.defrost_decision import DEFAULT_FROST_PROGRESSION, load_native_frost_frames

    loader = DatasetLoader(args.dataset)
    screening = screen_cv_cycles(
        pd.read_parquet(args.dataset / "training_data/candidate_decisions.parquet"),
        loader.list_cycles(),
        exclude_rb=False,
    )
    if args.frost_scale == "groups16":
        progression = (
            args.frost_progression
            or OUTPUT
            / "analysis/霜层等级分析/16级视觉霜态_近优覆盖与平均COP_20261015/discrete_progression.csv"
        )
        settings = json.loads((progression.parent / "run_settings.json").read_text())
        frames = pd.read_csv(progression)
        if (
            settings.get("coordinate_correction") != "harmony"
            or settings.get("method") != "mst_guided_visual_segments"
            or not frames.reference_method.eq("mst_guided_visual_segments").all()
            or set(frames.reference_index.dropna().unique()) != set(range(1, 17))
        ):
            raise ValueError("Provide the saved native Harmony 16-group visual-stage assignments")
        frames.image_time = pd.to_datetime(frames.image_time)
        frames["selection_grade"] = frames.reference_index
        source_method = (
            "Native Harmony · MST-guided 16 visual groups (front); other views matched by time"
        )
        defaults = [1, 4, 7, 10, 13, 16]
    else:
        frames = load_native_frost_frames(args.frost_progression or DEFAULT_FROST_PROGRESSION)
        frames["selection_grade"] = 100 * frames.visual_frost_score
        source_method = "Native Harmony + Slingshot (front); other views matched by time"
        defaults = [0, 5, 10, 15, 20, 25]
    frames = frames.loc[
        frames.cycle_name.isin(screening.loc[screening.selected, "cycle_name"])
        & frames.embedding_available
        & frames.visual_frost_score.notna()
    ]
    if args.image_cycle:
        frames = frames.loc[frames.cycle_name.eq(args.image_cycle)]
    targets = pd.Series(args.frost_grades or defaults, dtype=float)
    valid = (
        targets.isin(range(1, 17)) if args.frost_scale == "groups16" else targets.between(0, 100)
    )
    if len(targets) != len(CAMERAS) or not valid.all() or not targets.diff().iloc[1:].gt(0).all():
        raise ValueError("Provide six strictly increasing reference grades within the chosen scale")
    candidates = []
    for name, group in frames.groupby("cycle_name"):
        selected = []
        for target in targets:
            pool = (
                group
                if not selected
                else group.loc[
                    group.image_time.gt(selected[-1].image_time)
                    & group.selection_grade.gt(selected[-1].selection_grade)
                ]
            )
            if args.frost_scale == "groups16":
                pool = pool.loc[pool.selection_grade.eq(target)]
            if pool.empty:
                break
            selected.append(
                pool.sort_values(["projection_distance", "image_time"]).iloc[0]
                if args.frost_scale == "groups16"
                else pool.loc[(pool.selection_grade - target).abs().idxmin()]
            )
        if len(selected) != len(targets):
            continue
        chosen = pd.DataFrame(selected).reset_index(drop=True)
        error = (chosen.selection_grade - targets).abs()
        distance = (
            float(chosen.projection_distance.mean()) if args.frost_scale == "groups16" else 0.0
        )
        candidates.append((float(error.iloc[0]), float(error.sum()), distance, name, chosen))
    metadata = loader.load_image_metadata()
    root = image_root(args.dataset)
    for _, _, _, name, chosen in sorted(candidates, key=lambda item: item[:4]):
        records = []
        for i, camera in enumerate(CAMERAS):
            anchor = chosen.iloc[[i]][["cycle_name", "image_time", "selection_grade"]].rename(
                columns={
                    "image_time": "candidate_defrost_time",
                    "selection_grade": "reference_grade",
                }
            )
            views = metadata.loc[
                metadata.cycle_name.eq(name) & metadata.camera_role.eq(camera)
            ].assign(indexed=True)
            row = match_trajectory_frames(anchor, views, metric="indexed", max_age_seconds=30).iloc[
                0
            ]
            path = root / name / camera / str(row.file_name)
            if pd.isna(row.file_name) or not path.is_file():
                break
            records.append(
                dict(
                    camera_role=camera,
                    cycle_name=name,
                    display_order=i,
                    target_grade=float(targets.iloc[i]),
                    reference_grade=float(row.reference_grade),
                    reference_time=row.candidate_defrost_time,
                    image_time=row.image_time,
                    image_path=str(path),
                    offset_seconds=(row.image_time - row.candidate_defrost_time).total_seconds(),
                    source_method=source_method,
                )
            )
        if len(records) == len(CAMERAS):
            source = pd.DataFrame(records)
            return source
    raise ValueError(
        "No screened cycle has all six requested saved grades and physical camera frames"
    )


def render_camera_images(args):
    """Select frozen frost examples and save them with the shared renderer."""
    from zipfile import ZIP_DEFLATED, ZipFile

    name = (
        "camera_frost_images"
        if args.frost_scale == "native" and args.image_layout == "full"
        else f"camera_frost_images_{args.frost_scale}_{args.image_layout}"
    )
    if args.image_layout == "square":
        name += "_resized"
    source = select_camera_frost_images(args)
    source = render_camera_frost_images(source, args.output, layout=args.image_layout, name=name)
    source.to_csv(args.output / f"{name}_source.csv", index=False)
    prefix = name.removesuffix("_images")
    with ZipFile(
        args.output / f"{prefix}_six_images.zip", "w", compression=ZIP_DEFLATED
    ) as archive:
        for camera in source.camera_role:
            path = args.output / f"{prefix}_{camera}.png"
            archive.write(path, path.name)
