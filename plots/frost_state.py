"""Compare frozen RGB states at supported optima, replayed/observed RB or CV decisions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

from dataset_tools import DatasetLoader
from output_layout import PROJECT

FROST_CLUSTER_COLORS = ["#78A9BC", "#B5A0C5", "#79ADA0", "#D9AE85", "#92A8CB", "#C58E9C", "#B8BA8B"]


def render(root: Path, output: Path, *, seed: int = 0, **options) -> pd.DataFrame:
    from image_models.frost_progression import collect_states

    states = collect_states(root, seed=seed, **options)
    output.parent.mkdir(parents=True, exist_ok=True)
    states.to_csv(output.with_suffix(".csv"), index=False)
    all_states = states
    states = states.loc[states.embedding_available].copy()
    if states.empty:
        raise ValueError("no available embeddings; missing-point audit saved to CSV")
    title = {
        "cv": f"CV model: {options.get('model', 'B0')}",
        "optimal": "Supported COP optimum",
        "rb": "Replayed RB",
        "observed": "Observed defrost",
    }[options.get("point", "cv")]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (scatter, consistency) = plt.subplots(
        1, 2, figsize=(14, max(7, len(states) * 0.19)), constrained_layout=True
    )
    scatter.scatter(states.pca1, states.pca2, s=35, alpha=0.8, color="#4779a4")
    for row in states.itertuples():
        scatter.annotate(
            str(row.cycle_id),
            (row.pca1, row.pca2),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    if len(states) >= 3:
        covariance = np.cov(states[["pca1", "pca2"]].to_numpy().T)
        values, vectors = np.linalg.eigh(covariance)
        order = values.argsort()[::-1]
        values, vectors = values[order], vectors[:, order]
        angle = np.degrees(np.arctan2(*vectors[:, 0][::-1]))
        scatter.add_patch(
            Ellipse(
                states[["pca1", "pca2"]].mean().to_numpy(),
                2 * 1.96 * np.sqrt(values[0]),
                2 * 1.96 * np.sqrt(values[1]),
                angle=angle,
                facecolor="#5b8ff9",
                alpha=0.12,
                edgecolor="#3867d6",
                lw=1.5,
            )
        )
    scatter.axhline(0, color="0.7", lw=0.7)
    scatter.axvline(0, color="0.7", lw=0.7)
    scatter.set(
        xlabel="PCA 1 (unit-normalized DINOv2)", ylabel="PCA 2", title=f"Frozen RGB state: {title}"
    )

    metric = "cosine_to_reference" if options.get("reference_states") else "cosine_to_centroid"
    ordered = states.sort_values(metric)
    values = ordered[metric].to_numpy()
    y = np.arange(len(ordered))
    consistency.scatter(values, y, s=24, color="#c77e57")
    consistency.set(
        yticks=y,
        yticklabels=ordered.cycle_id.astype(str),
        xlabel="Cosine to fixed optimal centroid"
        if options.get("reference_states")
        else "Cosine to cohort centroid",
        ylabel="Cycle ID (least similar first)",
        title="Identify inconsistent cycles",
        xlim=(max(-1, values.min() - 0.02), 1.005),
    )
    consistency.tick_params(axis="y", labelsize=8)
    consistency.invert_yaxis()
    consistency.axvline(
        np.median(values), color="#777777", lw=1, label=f"median = {np.median(values):.3f}"
    )
    consistency.legend(frameon=True, loc="lower right")
    figure.suptitle(
        f"{title} | {options.get('cohort', 'regular')} cycles | "
        f"{len(states)}/{len(all_states)} available states",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return all_states


def _plot_frost_reference(axis, curve, origin, visual_top=1.02, *, metric=None):
    """Shared reference-time panel for historical and multi-objective cycle figures."""
    metric = metric or (
        "visual_frost_score" if "visual_frost_score" in curve else "visual_distance"
    )
    x = (curve.candidate_defrost_time - origin).dt.total_seconds() / 60
    axis.plot(x, curve[metric], color="#4779a4", lw=1.2, marker=".", ms=2)
    if "in_basin_grid" in curve:
        near = curve.in_basin_grid & curve[metric].notna()
        axis.scatter(
            x[near], curve.loc[near, metric], color="#3C6E8F", s=8, label="valid 1% basin grid"
        )
    axis.set(
        xlabel="Minutes since stable heating start",
        ylabel="Visual frost-state coordinate"
        if metric == "visual_frost_score"
        else "1 − cosine to early center\n(descriptive, not frost severity)",
        ylim=(-0.02, visual_top),
    )
    if metric == "visual_frost_score":
        if "is_beyond_reference" in curve:
            beyond = curve.is_beyond_reference.eq(True)
            axis.scatter(
                x[beyond],
                np.ones(beyond.sum()),
                marker="x",
                color="0.4",
                s=12,
                label="beyond reference (score capped at 1)",
            )
        axis.set_yticks(np.linspace(0, 1, 6))


def _render_trajectory_cycle(
    curve: pd.DataFrame,
    output: Path,
    visual_top: float,
    dataset: Path = (PROJECT / "dataset"),
    images_dir=None,
    include_images=True,
) -> None:
    from image_models.frost_progression import trajectory_snapshots
    from plots.cop import COP_COLOR, OPTIMAL_COLOR, RB_COLOR

    plt.style.use("seaborn-v0_8-whitegrid")
    cycle = curve.cycle_name.iloc[0]
    first = curve.iloc[0]
    origin = (
        first.stable_heating_start
        if pd.notna(first.stable_heating_start)
        else pd.Timestamp(first.cycle_start)
    )
    metric = "visual_frost_score" if "visual_frost_score" in curve else "visual_distance"
    reference_score = metric == "visual_frost_score"
    fig = plt.figure(figsize=(11, 9 if include_images else 6), constrained_layout=True)
    grid = fig.add_gridspec(
        3 if include_images else 2, 4, height_ratios=[1, 1, 1.1] if include_images else [1, 1]
    )
    axes = [fig.add_subplot(grid[0, :])]
    axes.append(fig.add_subplot(grid[1, :], sharex=axes[0]))
    decisions = (
        curve.loc[curve.is_decision_candidate] if "is_decision_candidate" in curve else curve
    )
    supported = decisions.cycle_cop.where(decisions.cycle_cop_eligible & decisions.on_action_grid)
    axes[0].plot(
        (decisions.candidate_defrost_time - origin).dt.total_seconds() / 60,
        supported / first.COP_eff_selected,
        color=COP_COLOR,
        lw=1.4,
    )
    axes[0].axhline(0.99, color="0.55", ls="--", lw=0.8)
    title = "visual frost-state coordinate" if reference_score else "descriptive visual distance"
    axes[0].set(ylabel="Effective COP / C*", title=f"{cycle} | COP and {title}")
    if pd.isna(first.t_star):
        axes[0].text(0.5, 0.5, "No supported COP optimum", ha="center", transform=axes[0].transAxes)
    _plot_frost_reference(axes[1], curve, origin, visual_top)
    origin_label = "stable heating start" if pd.notna(first.stable_heating_start) else "cycle start"
    axes[1].set_xlabel(f"Minutes since {origin_label}")
    for ax in axes:
        if pd.notna(first.cycle_cop_basin_1pct_start) and pd.notna(first.cycle_cop_basin_1pct_end):
            left = (first.cycle_cop_basin_1pct_start - origin).total_seconds() / 60
            right = (first.cycle_cop_basin_1pct_end - origin).total_seconds() / 60
            ax.axvspan(left, right, color=COP_COLOR, alpha=0.10)
        for field, color, label in (("t_star", OPTIMAL_COLOR, "t*"), ("t_RB", RB_COLOR, "RB")):
            if pd.notna(first[field]):
                ax.axvline(
                    (first[field] - origin).total_seconds() / 60,
                    color=color,
                    lw=1,
                    ls="--",
                    label=label,
                )
    axes[1].legend(fontsize=7, loc="best")
    from plots.defrost_decision import _plot_decision_image

    labels = ("A 1% start", "B 1% end", "C max visual score", "D min visual score")
    snapshots = trajectory_snapshots(curve) if include_images else []
    for i, (label, row) in enumerate(zip(labels, snapshots)):
        info = {}
        if row is not None:
            path = (
                (Path(images_dir) if images_dir is not None else dataset / "images")
                / cycle
                / "front"
                / str(row.file_name)
            )
            info = dict(
                image_path=path,
                available=path.is_file(),
                target_time=row.candidate_defrost_time,
                image_time=row.image_time,
                offset_seconds=-row.image_age_seconds,
            )
            position = (row.candidate_defrost_time - origin).total_seconds() / 60
            for ax in axes:
                ax.axvline(position, color="0.45", lw=0.7, ls=":")
                ax.text(
                    position,
                    0.96 - i * 0.10,
                    label[0],
                    transform=ax.get_xaxis_transform(),
                    fontsize=8,
                    ha="center",
                    va="top",
                    backgroundcolor="white",
                )
        _plot_decision_image(fig.add_subplot(grid[2, i]), info, label, origin, origin)
    fig.savefig(output / f"{cycle}.png", dpi=160)
    plt.close(fig)


def render_trajectories(
    decisions: Path,
    output: Path,
    *,
    dataset=(PROJECT / "dataset"),
    feature_caches=(),
    max_age_seconds=120,
    n_jobs=6,
    progression=None,
    cycle_list=None,
    fetch_cloud=False,
    images_dir=None,
    include_images=True,
) -> pd.DataFrame:
    from joblib import Parallel, delayed, parallel_config

    from image_models.dinov2_features import DINO_COLUMNS, merge_cached_features
    from image_models.frost_progression import trajectory_scores, trajectory_snapshots
    from image_models.inputs import reliable_rgb_cohort
    from plots.cop import OPTIMAL_COLOR, RB_COLOR

    loader = DatasetLoader(dataset)
    cycles = pd.read_csv(decisions / "cycle_comparison.csv")
    if progression is None:
        cohort = reliable_rgb_cohort(loader, include_rb=False, require_complete_peak=True)
        cycles = cycles.loc[cycles.cycle_name.isin(cohort.cycle_name) & cycles.t_star.notna()]
    else:
        frames = pd.read_csv(progression)
        cycles = cycles.loc[cycles.cycle_name.isin(frames.cycle_name)]
    if cycle_list is not None:
        selected = pd.read_csv(cycle_list).cycle_name
        cycles = cycles.loc[cycles.cycle_name.isin(selected)]
        if set(cycles.cycle_name) != set(selected):
            raise ValueError(
                "Requested trajectory cycles are missing decisions or reference frames"
            )
    if progression is not None:
        frames = frames.loc[frames.cycle_name.isin(cycles.cycle_name)]
    cycles = cycles.loc[
        :,
        [
            "cycle_name",
            "experiment_id",
            "stable_heating_start",
            "t_star",
            "t_RB",
            "COP_eff_selected",
            "cycle_cop_basin_1pct_start",
            "cycle_cop_basin_1pct_end",
            "cycle_start",
        ],
    ]
    if progression is not None:
        bounds = loader.list_cycles(include_rb=True).set_index("cycle_name")
        end = (
            bounds[["defrost_preparation_start", "defrost_start", "end_time"]]
            .apply(pd.to_datetime)
            .min(axis=1)
        )
        cycles["frost_end"] = cycles.cycle_name.map(end)
    candidates = pd.read_csv(
        decisions / "candidate_decisions.csv",
        usecols=[
            "cycle_name",
            "candidate_defrost_time",
            "cycle_cop",
            "cycle_cop_eligible",
            "on_action_grid",
        ],
    )
    candidates = candidates.loc[candidates.cycle_name.isin(cycles.cycle_name)].copy()
    if progression is not None:
        # Plot every observed RGB time, without adding points to the COP decision support.
        candidates["is_decision_candidate"] = True
        image_points = frames[["cycle_name", "image_time"]].rename(
            columns={"image_time": "candidate_defrost_time"}
        )
        image_points = image_points.assign(
            is_decision_candidate=False, on_action_grid=False, cycle_cop_eligible=False
        )
        candidates["candidate_defrost_time"] = pd.to_datetime(candidates.candidate_defrost_time)
        image_points["candidate_defrost_time"] = pd.to_datetime(image_points.candidate_defrost_time)
        candidates = pd.concat([candidates, image_points], ignore_index=True).drop_duplicates(
            ["cycle_name", "candidate_defrost_time"]
        )
    if progression is None:
        paths = [Path(p) for p in feature_caches]
        frames = []
        for cycle in cycles.cycle_name:
            images = loader.load_image_metadata(cycle)
            images = images.loc[
                images.camera_role.eq("front"),
                ["cycle_name", "camera_role", "file_name", "image_time"],
            ].copy()
            if images.empty:
                continue
            images["local_image_available"] = images.file_name.map(
                lambda name: (loader.dataset_root / "images" / cycle / "front" / name).is_file()
            )
            cache_files = [p / f"{cycle}.parquet" if p.is_dir() else p for p in paths]
            features = merge_cached_features(images, cache_files)
            merged = images.merge(
                features, on=["cycle_name", "camera_role", "file_name"], validate="one_to_one"
            )
            merged.loc[~merged.local_image_available, DINO_COLUMNS] = np.nan
            frames.append(merged)
        if not frames:
            raise ValueError("no local front images in the selected cohort")
        frames = pd.concat(frames, ignore_index=True)
    result = trajectory_scores(cycles, candidates, frames, max_age_seconds=max_age_seconds)
    if fetch_cloud and include_images:
        from dataset_tools.cloud_images import materialize_image_members

        requests = {
            name: [
                str(row.file_name)
                for row in trajectory_snapshots(curve)
                if row is not None and pd.notna(row.file_name)
            ]
            for name, curve in result.groupby("cycle_name", sort=True)
        }
        with materialize_image_members(
            dataset, requests, n_jobs=n_jobs, images_dir=images_dir
        ) as completed:
            for _ in completed:
                pass
    metric = "visual_frost_score" if progression is not None else "visual_distance"
    output.mkdir(parents=True, exist_ok=True)
    if progression is None:
        result.attrs["early_reference"].to_csv(output / "early_reference.csv", index=False)
    else:
        result["reference_cycle"] = frames.reference_cycle.iloc[0]
    result.to_csv(output / "trajectory_source.csv", index=False)
    plt.style.use("seaborn-v0_8-whitegrid")
    summary = []
    visual_top = 1.08 if progression is not None else max(0.05, result.visual_distance.max() + 0.02)
    for cycle, curve in result.groupby("cycle_name", sort=True):
        first = curve.iloc[0]
        near = curve.loc[curve.in_basin_grid & curve[metric].notna()]
        decision_rows = (
            curve.loc[curve.is_decision_candidate] if "is_decision_candidate" in curve else curve
        )
        row = {
            "cycle_name": cycle,
            "experiment_id": first.experiment_id,
            "candidate_count": len(decision_rows),
            "matched_candidate_count": int(decision_rows[metric].notna().sum()),
            "plotted_score_count": int(curve[metric].notna().sum()),
            "near_grid_count": int(curve.in_basin_grid.sum()),
            "near_matched_count": len(near),
            "near_visual_min": near[metric].min(),
            "near_visual_max": near[metric].max(),
            "visual_metric": metric,
            "reference_count": first.get("reference_count", np.nan),
            "reference_cycle": first.get("reference_cycle", ""),
        }
        selected = curve.loc[curve.candidate_defrost_time.eq(first.t_star), metric].dropna()
        row["t_star_visual"] = selected.iloc[0] if len(selected) else np.nan
        rb = curve.loc[curve.candidate_defrost_time.eq(first.t_RB), metric].dropna()
        row["rb_visual"] = rb.iloc[0] if len(rb) else np.nan
        summary.append(row)

    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        Parallel()(
            delayed(_render_trajectory_cycle)(
                curve, output, visual_top, dataset, images_dir, include_images
            )
            for _, curve in result.groupby("cycle_name", sort=True)
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, max(6, len(summary) * 0.17)), constrained_layout=True)
    ordered = summary.sort_values("cycle_name").reset_index(drop=True)
    for i, name in enumerate(ordered.cycle_name):
        points = result.loc[
            result.cycle_name.eq(name) & result.in_basin_grid & result[metric].notna(), metric
        ]
        ax.scatter(points, np.full(len(points), i), s=5, color="#4779a4", alpha=0.5)
    ax.scatter(
        ordered.t_star_visual,
        np.arange(len(ordered)),
        s=16,
        marker="D",
        color=OPTIMAL_COLOR,
        label="t*",
    )
    ax.scatter(
        ordered.rb_visual,
        np.arange(len(ordered)),
        s=16,
        facecolors="none",
        edgecolors=RB_COLOR,
        label="RB",
    )
    ax.set(
        yticks=np.arange(len(ordered)),
        yticklabels=ordered.cycle_name.str.rsplit("_", n=1).str[-1],
        xlabel=f"{metric} at valid 1% basin grid points",
        ylabel="Cycle",
        title="Offline visual comparison; overlap does not establish a shared frost state",
    )
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(fontsize=8)
    fig.savefig(output / "near_basin_values.png", dpi=180)
    plt.close(fig)
    return result


def plot_progression(
    result, counts, dataset, output, *, seed=0, fetch_cloud=False, n_jobs=6, images_dir=None
):
    """Independently sample six different cycles' local images from each phase bin."""
    from image_models.frost_progression import sample_phase_images
    from plots.defrost_decision import _plot_decision_image

    labels = (
        counts["label"].tolist()
        if "label" in counts
        else [f"{x:.1f}" for x in counts.reference_phase]
    )
    image_root = Path(images_dir) if images_dir is not None else Path(dataset) / "images"
    pool = result.copy()
    local = pool.apply(
        lambda row: (image_root / row.cycle_name / "front" / row.file_name).is_file(), axis=1
    )
    if fetch_cloud:
        from dataset_tools.cloud_images import available_image_archives

        readable_cycles = available_image_archives()
        pool = pool.loc[local | pool.cycle_name.isin(readable_cycles)]
    else:
        pool = pool.loc[local]
    selected = sample_phase_images(pool, seed=seed)
    if fetch_cloud:
        from dataset_tools.cloud_images import materialize_image_members

        requests = selected.groupby("cycle_name").file_name.agg(list).to_dict()
        with materialize_image_members(
            Path(dataset), requests, n_jobs=n_jobs, images_dir=image_root
        ) as completed:
            for _ in completed:
                pass
    paths = selected.apply(
        lambda row: image_root / row.cycle_name / "front" / row.file_name, axis=1
    )
    if not paths.map(Path.is_file).all():
        raise FileNotFoundError("Selected grid images are unavailable; no blank cells are rendered")
    Path(output).mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(22, 15), constrained_layout=True)
    grid = fig.add_gridspec(7, 11, height_ratios=[1] * 6 + [1.4])
    source = []
    for b in range(11):
        for i, (_, chosen) in enumerate(selected.loc[selected.phase_bin.eq(b)].iterrows()):
            axis = fig.add_subplot(grid[i, b])
            name = chosen.cycle_name
            info = dict(
                image_path=image_root / name / "front" / chosen.file_name,
                available=True,
                image_time=chosen.image_time,
                target_time=chosen.image_time,
                offset_seconds=0,
            )
            _plot_decision_image(axis, info, "", pd.NaT, pd.NaT)
            heading = f"{labels[b]}\n" if i == 0 else ""
            axis.set_title(
                f"{heading}Cycle {int(name.rsplit('_', 1)[-1])}", fontsize=10, loc="left"
            )
            if b == 0:
                axis.set_ylabel(f"Sample {i + 1}", fontsize=11)
            source.append(
                {
                    **chosen.to_dict(),
                    "reference_phase": b / 10,
                    "sample_row": i + 1,
                    "panel_image_path": str(info["image_path"]),
                }
            )
    axis = fig.add_subplot(grid[-1, :])
    axis.bar(counts.reference_phase, counts.image_count, width=0.075, color="#4779a4")
    for row in counts.itertuples():
        axis.text(
            row.reference_phase,
            row.image_count,
            f"{row.image_count}\n{row.cycle_count} cycles",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set(
        xlabel="Reference phase bin",
        ylabel="Number of unique front images",
        xticks=counts.reference_phase,
        xticklabels=labels,
        xlim=(-0.055, counts.reference_phase.max() + 0.055),
        title=f"B | All {result.cycle_name.nunique()} cycles; each image counted once",
    )
    axis.margins(y=0.18)
    reference = result.reference_cycle.iloc[0]
    method = result.reference_method.iloc[0]
    if "frost_only" in result and result.frost_only.iloc[0]:
        method += ", frost-only"
    algorithm = "PCA + Slingshot" if method == "slingshot" else "PCA + Slingshot MST polyline"
    if "coordinate_correction" in result:
        method += f", {result.coordinate_correction.iloc[0]}"
    fig.suptitle(
        f"A | Six independently sampled images per bin; distinct cycles (seed {seed}); readable pixel sources\n"
        f"Reference: {reference} ({method}); offline DINO + {algorithm}; visual consistency unverified",
        fontsize=15,
    )
    pd.DataFrame(source).to_csv(Path(output) / "grid_source.csv", index=False)
    fig.savefig(Path(output) / "reference_front_grid.png", dpi=160)
    plt.close(fig)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, help="saved model run (required for --point cv)")
    parser.add_argument(
        "--output",
        type=Path,
        default=(PROJECT / "output/90_复现材料/05_霜态轨迹/RGB霜态/only_rgb_frost_state.png"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    parser.add_argument(
        "--point",
        choices=(
            "optimal",
            "rb",
            "observed",
            "cv",
            "trajectory",
            "progression",
        ),
        default="cv",
    )
    parser.add_argument("--cohort", choices=("regular", "rb", "all"), default="regular")
    parser.add_argument("--decisions", type=Path)
    parser.add_argument(
        "--progression", type=Path, help="saved progression.csv for trajectory plots"
    )
    parser.add_argument(
        "--fetch-cloud", action="store_true", help="retain only images used by requested panels"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="explicit panel-image destination; Dataset paths stay unchanged",
    )
    parser.add_argument(
        "--no-rgb-panels",
        action="store_true",
        help="trajectory diagnostics: COP and frost panels only",
    )
    parser.add_argument("--feature-cache", type=Path, nargs="*", default=[])
    parser.add_argument("--model", help="model name in cycle_metrics.csv")
    parser.add_argument(
        "--cycle-list", type=Path, help="CSV with cycle_name for a fixed paired cohort"
    )
    parser.add_argument(
        "--reference-states", type=Path, help="optimal-state CSV for fixed centroid and PCA"
    )
    parser.add_argument("--max-age-seconds", type=int, default=120)
    parser.add_argument("--n-jobs", type=int, default=6)
    args = parser.parse_args()
    from output_layout import prepare_output

    prepare_output(args, training=False)
    if args.point == "cv" and (args.root is None or args.model is None):
        parser.error("--point cv requires --root and --model from a retained model run")
    if args.point == "progression":
        if args.progression is None:
            parser.error("--point progression requires a saved --progression CSV")
        plot_progression(
            pd.read_csv(args.progression),
            pd.read_csv(args.progression.parent / "phase_counts.csv"),
            args.dataset,
            args.output,
            seed=args.seed,
            fetch_cloud=args.fetch_cloud,
            n_jobs=args.n_jobs,
            images_dir=args.images_dir,
        )
        raise SystemExit
    if args.point == "trajectory":
        if args.cohort != "regular":
            parser.error("trajectory supports only the complete regular cohort")
        rows = render_trajectories(
            args.decisions,
            args.output,
            dataset=args.dataset,
            feature_caches=args.feature_cache,
            max_age_seconds=args.max_age_seconds,
            n_jobs=args.n_jobs,
            progression=args.progression,
            cycle_list=args.cycle_list,
            fetch_cloud=args.fetch_cloud,
            images_dir=args.images_dir,
            include_images=not args.no_rgb_panels,
        )
        metric = "visual_frost_score" if args.progression is not None else "visual_distance"
        print(
            f"{rows.cycle_name.nunique()} cycles; {rows[metric].notna().sum()}/{len(rows)} candidate rows with {metric}"
        )
        raise SystemExit
    states = render(
        args.root,
        args.output,
        seed=args.seed,
        dataset=args.dataset,
        point=args.point,
        cohort=args.cohort,
        decisions=args.decisions,
        feature_caches=args.feature_cache,
        model=args.model,
        max_age_seconds=args.max_age_seconds,
        cycle_list=args.cycle_list,
        reference_states=args.reference_states,
    )
    valid = states.loc[states.embedding_available, "cosine_to_centroid"]
    print(
        f"{len(valid)}/{len(states)} states; median cosine={valid.median():.6f}; "
        f"IQR={valid.quantile(0.25):.6f}–{valid.quantile(0.75):.6f}"
    )
