"""Audit a frozen PCA space for experiment/date information and observed state support.

Relative time is the fraction of each cycle's observed front-image span, not frost mass.
Batch association is descriptive: experiment also contains genuine operating conditions.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import rankdata
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits


def cycle_correlations(cycle, frames, coordinates):
    t = rankdata(frames.t_rel.to_numpy())
    t -= t.mean()
    ranks = rankdata(coordinates, axis=0)
    ranks -= ranks.mean(axis=0)
    denominator = np.linalg.norm(t) * np.linalg.norm(ranks, axis=0)
    rho = np.divide(
        t @ ranks, denominator, out=np.full(coordinates.shape[1], np.nan), where=denominator > 0
    )
    return pd.DataFrame({"cycle_name": cycle, "pc": np.arange(1, len(rho) + 1), "spearman": rho})


def batch_eta2(values, groups, weights):
    """Between-group / total variance, with one total unit of weight per cycle."""
    mean = np.average(values, axis=0, weights=weights)
    total = np.sum(weights[:, None] * (values - mean) ** 2, axis=0)
    between = np.zeros(values.shape[1])
    for group in pd.unique(groups):
        mask = groups == group
        group_mean = np.average(values[mask], axis=0, weights=weights[mask])
        between += weights[mask].sum() * (group_mean - mean) ** 2
    return np.divide(between, total, out=np.full(values.shape[1], np.nan), where=total > 0)


def partial_batch_r2(values, state, groups, weights):
    """Extra variance explained by experiment after a common smooth state curve."""
    basis = SplineTransformer(n_knots=5, degree=3, include_bias=False).fit_transform(state[:, None])
    state_design = np.column_stack([np.ones(len(values)), basis])
    batches = pd.get_dummies(groups, drop_first=True, dtype=float).to_numpy()
    full_design = np.column_stack([state_design, batches])
    root_weight = np.sqrt(weights)
    target = values * root_weight[:, None]
    errors = []
    for design in (state_design, full_design):
        design = design * root_weight[:, None]
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        errors.append(np.sum((target - design @ coefficients) ** 2, axis=0))
    time_error, full_error = errors
    total = np.sum(
        weights[:, None] * (values - np.average(values, axis=0, weights=weights)) ** 2, axis=0
    )
    result = np.divide(
        time_error - full_error,
        time_error,
        out=np.full(values.shape[1], np.nan),
        where=time_error > total * 1e-12,
    )
    return np.clip(result, 0, 1)


def pc_diagnostics(frames, coordinates, explained_variance, n_jobs=6):
    groups = list(frames.groupby("cycle_name", sort=True).indices.items())
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        tables = Parallel()(
            delayed(cycle_correlations)(name, frames.iloc[ids], coordinates[ids])
            for name, ids in groups
        )
    correlations = pd.concat(tables, ignore_index=True)
    grouped = correlations.groupby("pc").spearman
    table = pd.DataFrame(
        {
            "pc": np.arange(1, coordinates.shape[1] + 1),
            "explained_variance": explained_variance,
            "median_cycle_spearman": grouped.median().to_numpy(),
            "median_abs_cycle_spearman": grouped.apply(lambda s: s.abs().median()).to_numpy(),
            "q25_cycle_spearman": grouped.quantile(0.25).to_numpy(),
            "q75_cycle_spearman": grouped.quantile(0.75).to_numpy(),
        }
    )
    weights = 1 / frames.groupby("cycle_name").cycle_name.transform("size").to_numpy()
    mean = np.average(coordinates, axis=0, weights=weights)
    table["coordinate_variance"] = np.average((coordinates - mean) ** 2, axis=0, weights=weights)
    experiment = frames.experiment_id.to_numpy()
    table["experiment_eta2"] = batch_eta2(coordinates, experiment, weights)
    table["date_eta2"] = batch_eta2(coordinates, frames.image_date.to_numpy(), weights)
    table["experiment_eta2_images"] = batch_eta2(coordinates, experiment, np.ones(len(frames)))
    table["experiment_partial_r2_after_time"] = partial_batch_r2(
        coordinates, frames.t_rel.to_numpy(), experiment, weights
    )
    known = frames.existing_reference.notna().to_numpy()
    table["experiment_partial_r2_after_reference"] = partial_batch_r2(
        coordinates[known],
        frames.existing_reference.to_numpy()[known],
        experiment[known],
        weights[known],
    )
    return table, correlations


def state_support(frames):
    """Count actual observations in tenth bins, including score=1 in the final bin."""
    known = frames.loc[frames.existing_reference.notna()].copy()
    known["state_bin"] = np.minimum(np.floor(known.existing_reference * 10).astype(int), 9)
    counts = known.groupby(["experiment_id", "state_bin"]).agg(
        image_count=("cycle_name", "size"), cycle_count=("cycle_name", "nunique")
    )
    grid = pd.MultiIndex.from_product(
        [sorted(frames.experiment_id.unique()), range(10)], names=["experiment_id", "state_bin"]
    )
    counts = counts.reindex(grid, fill_value=0).reset_index()
    summary = (
        counts.groupby("state_bin")
        .agg(
            experiment_count=("cycle_count", lambda s: int((s > 0).sum())),
            image_count=("image_count", "sum"),
            experiment_cycle_count=("cycle_count", "sum"),
        )
        .reset_index()
    )
    return counts, summary


def mst_edges(sling):
    from pyslingshot.util import mahalanobis

    covariances = [
        np.cov(sling.data[sling.cluster_label_indices == i].T) for i in range(sling.num_clusters)
    ]
    rows = []
    for parent, children in sling.tree.items():
        for child in children:
            difference = sling.cluster_centres[parent] - sling.cluster_centres[child]
            squared = difference**2
            rows.append(
                dict(
                    parent=parent,
                    child=child,
                    euclidean=float(np.linalg.norm(difference)),
                    mahalanobis_squared=np.asarray(
                        mahalanobis(
                            sling.cluster_centres[parent],
                            sling.cluster_centres[child],
                            covariances[parent],
                            covariances[child],
                        )
                    ).item(),
                    largest_difference_pc=int(np.argmax(squared) + 1),
                    pc9_euclidean_fraction=float(squared[8] / squared.sum())
                    if len(squared) > 8
                    else np.nan,
                )
            )
    return pd.DataFrame(rows)


def audit_representation(source, progression, output, n_jobs=6, admission_settings=None):
    from image_models.dinov2_features import KEYS

    state = joblib.load(Path(source) / "fit_state.joblib")
    frames = state["frames"].copy().reset_index(drop=True)
    coordinates = state["sling"].data.copy()
    if admission_settings is not None:
        from image_models.cycle_selection import screen_admission

        screening = screen_admission(admission_settings)
        Path(output).mkdir(parents=True, exist_ok=True)
        screening.to_csv(Path(output) / "cycle_screening.csv", index=False)
        selected = set(screening.loc[screening.selected, "cycle_name"])
        mask = frames.cycle_name.isin(selected).to_numpy()
        if selected - set(frames.loc[mask, "cycle_name"]):
            raise ValueError("Some screened cycles lack saved representation coordinates")
        frames, coordinates = frames.loc[mask].reset_index(drop=True), coordinates[mask]
    frames["image_time"] = pd.to_datetime(frames.image_time, format="mixed")
    first = frames.groupby("cycle_name").image_time.transform("min")
    last = frames.groupby("cycle_name").image_time.transform("max")
    frames["t_rel"] = (frames.image_time - first).dt.total_seconds() / (
        last - first
    ).dt.total_seconds()
    if frames.t_rel.isna().any():
        raise ValueError("Relative-time audit requires a nonzero observed span for every cycle")
    frames["image_date"] = frames.image_time.dt.strftime("%Y-%m-%d")
    reference = pd.read_csv(
        progression, usecols=KEYS + ["visual_frost_score", "is_beyond_reference"]
    )
    reference.loc[reference.is_beyond_reference.eq(True), "visual_frost_score"] = 1.0
    frames = frames.merge(
        reference[KEYS + ["visual_frost_score"]], on=KEYS, how="left", validate="one_to_one"
    )
    frames = frames.rename(columns={"visual_frost_score": "existing_reference"})
    if frames.existing_reference.isna().all():
        raise ValueError("No matching frozen visual reference frames")
    print(
        f"[audit] {len(frames)} frames; {frames.cycle_name.nunique()} cycles; "
        f"{frames.experiment_id.nunique()} experiments",
        flush=True,
    )
    diagnostics, correlations = pc_diagnostics(
        frames, coordinates, state["pca"].explained_variance_ratio_, n_jobs
    )
    pcs = diagnostics.sort_values("median_abs_cycle_spearman", ascending=False).pc.head(2).tolist()
    for pc in pcs:
        frames[f"PC{pc}"] = coordinates[:, pc - 1]
    ranges = []
    for experiment, group in frames.groupby("experiment_id", sort=True):
        row = dict(
            experiment_id=experiment,
            image_count=len(group),
            cycle_count=group.cycle_name.nunique(),
            dates=";".join(sorted(group.image_date.unique())),
        )
        for field in ["t_rel", "existing_reference"] + [f"PC{pc}" for pc in pcs]:
            for name, quantile in [
                ("min", 0),
                ("q05", 0.05),
                ("median", 0.5),
                ("q95", 0.95),
                ("max", 1),
            ]:
                row[f"{field}_{name}"] = group[field].quantile(quantile)
        ranges.append(row)
    ranges = pd.DataFrame(ranges)
    counts, support = state_support(frames)
    cluster_experiments = (
        frames.groupby(["cluster", "experiment_id"])
        .agg(image_count=("cycle_name", "size"), cycle_count=("cycle_name", "nunique"))
        .reset_index()
    )
    cluster_experiments["within_cluster_fraction"] = (
        cluster_experiments.image_count
        / cluster_experiments.groupby("cluster").image_count.transform("sum")
    )
    edges = mst_edges(state["sling"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [
        ("pc_diagnostics", diagnostics),
        ("cycle_pc_correlations", correlations),
        ("experiment_ranges", ranges),
        ("experiment_state_support", counts),
        ("state_support_summary", support),
        ("cluster_experiments", cluster_experiments),
        ("mst_edges", edges),
    ]:
        table.to_csv(output / f"{name}.csv", index=False)
    from plots.frost_trajectory import plot_representation_audit

    method = "Harmony-corrected PCA" if "harmony" in state else "Raw PCA"
    plot_representation_audit(diagnostics, ranges, counts, edges, output, pcs, space_label=method)
    print(diagnostics.to_string(index=False))
    print("Cluster 5 experiments:")
    print(cluster_experiments.loc[cluster_experiments.cluster.eq(5)].to_string(index=False))
    print("Actual observed state support:")
    print(support.to_string(index=False))
    print("MST edges:")
    print(edges.to_string(index=False))
    return diagnostics, ranges, counts


def compare_harmony_coordinates(source, admission_settings, output, pcs=(1, 2), compact=False):
    """Show paired frozen coordinates, after authoritative main-cohort screening."""
    from image_models.cycle_selection import screen_admission
    from plots.frost_trajectory import plot_harmony_coordinates

    screening = screen_admission(admission_settings)
    source = Path(source)
    settings = json.loads((source / "run_settings.json").read_text())
    if settings.get("coordinate_correction") != "harmony":
        raise ValueError("Use the native global Harmony run, not harmony-linear")
    state = joblib.load(source / "fit_state.joblib")
    frames = state["frames"].reset_index(drop=True)
    selected = set(screening.loc[screening.selected, "cycle_name"])
    mask = frames.cycle_name.isin(selected).to_numpy()
    if selected - set(frames.loc[mask, "cycle_name"]):
        raise ValueError("Some screened cycles have no saved trajectory coordinates")
    raw = state["raw_coordinates"][mask]
    corrected = state["sling"].data[mask]
    frames = frames.loc[mask].reset_index(drop=True)
    if len(set(pcs)) != 2 or min(pcs) < 1 or max(pcs) > raw.shape[1]:
        raise ValueError("Choose two distinct saved coordinate dimensions")
    weights = 1 / frames.groupby("cycle_name").cycle_name.transform("size").to_numpy()
    experiments = frames.experiment_id.to_numpy()
    metrics = pd.DataFrame(
        dict(
            pc=np.arange(1, raw.shape[1] + 1),
            raw_pca_variance_fraction=state["pca"].explained_variance_ratio_,
            raw_experiment_eta2=batch_eta2(raw, experiments, weights),
            corrected_experiment_eta2=batch_eta2(corrected, experiments, weights),
        )
    )
    columns = ["cycle_name", "experiment_id", "image_time", "file_name"]
    points = frames[columns].copy()
    for pc in pcs:
        points[f"raw_PC{pc}"] = raw[:, pc - 1]
        points[f"corrected_PC{pc}"] = corrected[:, pc - 1]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    screening.to_csv(output / "cycle_screening.csv", index=False)
    points.to_csv(output / "harmony_coordinate_source.csv", index=False)
    metrics.to_csv(output / "experiment_association_by_pc.csv", index=False)
    plot_harmony_coordinates(points, output, pcs, compact, n_dimensions=raw.shape[1])
    print(
        f"{len(points)} images; {points.cycle_name.nunique()} cycles; "
        f"{points.experiment_id.nunique()} experiments; frozen PCA {raw.shape[1]}D"
    )
    print(metrics.loc[metrics.pc.isin(pcs)].to_string(index=False))
    return points, metrics


def render_slingshot_geometry(
    source, output, pcs=(1, 2), comparison_source=None, compact=False, clusters_only=False
):
    """Display the frozen global model, without selecting or refitting a cohort."""
    from image_models.slingshot import trajectory_geometry
    from plots.frost_trajectory import plot_slingshot_geometry

    source, output = Path(source), Path(output)
    settings = json.loads((source / "run_settings.json").read_text())
    if settings.get("coordinate_correction") != "harmony":
        raise ValueError("Use the native global Harmony + Slingshot trajectory")
    state = joblib.load(source / "fit_state.joblib")
    sling = state["sling"]
    if len(sling.lineages) != 1:
        raise ValueError("The single-path figure requires one saved lineage")
    coordinates = sling.data
    if len(set(pcs)) != 2 or min(pcs) < 1 or max(pcs) > coordinates.shape[1]:
        raise ValueError("Choose two distinct saved coordinate dimensions")
    dims = np.asarray(pcs) - 1
    points = state["frames"][
        ["cycle_name", "experiment_id", "image_time", "file_name", "cluster"]
    ].copy()
    points[["x", "y"]] = coordinates[:, dims]
    centers = pd.read_csv(source / "clusters.csv")
    centers[["x", "y"]] = sling.cluster_centres[centers.cluster.to_numpy()][:, dims]
    if len(sling.lineages) == 1:
        display_order = {
            int(cluster): order for order, cluster in enumerate(sling.lineages[0].clusters)
        }
        centers["display_cluster"] = centers.cluster.map(display_order)
        points["display_cluster"] = points.cluster.map(display_order)
    if "fine_stages" in state:
        points["native_cluster"] = points.cluster
        points["cluster"] = state["frames"].fine_cluster.to_numpy()
        points["display_cluster"] = points.cluster
        centers = state["fine_stages"].rename(columns={"fine_cluster": "cluster"}).copy()
        centers[["x", "y"]] = sling.polyline_points[:, dims]
        centers["display_cluster"] = centers.cluster
        centers["display_label"] = centers.cluster.astype(str).where(
            centers.kind.eq("visual_stage"), "末端"
        )
        centers["is_start"] = centers.cluster.eq(0)
        parent_order = centers.set_index("cluster").parent_order
        points["color_group"] = points.cluster.map(parent_order)
    phase, geometry = trajectory_geometry(sling)
    unique, ids = np.unique(phase, return_index=True)
    path = pd.DataFrame(dict(phase=unique, x=geometry[ids, dims[0]], y=geometry[ids, dims[1]]))
    edges = mst_edges(sling)
    if "fine_stages" in state:
        edges = edges.iloc[:0]
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [
        ("trajectory_points", points),
        ("cluster_centers", centers),
        (
            "center_polyline"
            if settings.get("trajectory_type") == "mst_polyline"
            else "principal_curve",
            path,
        ),
        ("mst_edges", edges),
    ]:
        prefix = "refined" if "fine_stages" in state else "native"
        table.to_csv(output / f"{prefix}_{name}.csv", index=False)
    baseline = None
    baseline_label = "原始默认平滑曲线"
    if comparison_source is not None:
        prior_settings = json.loads((Path(comparison_source) / "run_settings.json").read_text())
        if prior_settings.get("trajectory_type") == "mst_polyline":
            baseline_label = f"原方法：{prior_settings['n_clusters']}个 K-means 中心连线"
        elif prior_settings.get("spline_df") is not None:
            baseline_label = f"样条曲线（df={prior_settings['spline_df']}）"
        elif prior_settings.get("spline_residual") is not None:
            baseline_label = f"样条曲线（平滑预算 {prior_settings['spline_residual']:g}）"
        prior = joblib.load(Path(comparison_source) / "fit_state.joblib")["sling"]
        if not np.array_equal(prior.data, sling.data):
            raise ValueError("Curve overlays require identical corrected coordinates")
        phase, prior_geometry = trajectory_geometry(prior)
        unique, ids = np.unique(phase, return_index=True)
        baseline = pd.DataFrame(
            dict(phase=unique, x=prior_geometry[ids, dims[0]], y=prior_geometry[ids, dims[1]])
        )
        baseline.to_csv(
            output
            / (
                "baseline_center_polyline.csv"
                if prior_settings.get("trajectory_type") == "mst_polyline"
                else "baseline_principal_curve.csv"
            ),
            index=False,
        )
    curve_label = (
        "Slingshot / B-spline (df=" + str(settings["spline_df"]) + ")"
        if settings.get("spline_df") is not None
        else "Slingshot 主曲线"
    )
    if settings.get("spline_knot_placement") == "uniform":
        curve_label += " / 弧长均匀节点"
    if settings.get("spline_residual") is not None:
        curve_label = f"Slingshot / 平滑预算 {settings['spline_residual']:g}"
    polyline = settings.get("trajectory_type") == "mst_polyline"
    if polyline:
        curve_label = "聚类中心折线" if compact else "聚类中心折线 / 弧长霜态坐标"
        if "fine_stages" in state:
            curve_label = f"改进方法：{len(state['fine_stages']) - 1}个分组中心连线＋末端延长"
    pooling = settings.get("feature_pooling", "cls")
    feature_label = {
        "cls": "DINOv2 CLS",
        "patch_mean": "DINOv2 patch mean",
        "patch_grid2": "DINOv2 2×2 patch grid",
    }[pooling]
    if baseline is None or not compact:
        plot_slingshot_geometry(
            points,
            centers,
            path,
            edges,
            output,
            pcs,
            curve_label=curve_label,
            feature_label=feature_label,
            n_dimensions=coordinates.shape[1],
            polyline=polyline,
            compact=compact,
            clusters_only=clusters_only,
            recipe=settings,
        )
    if baseline is not None:
        plot_slingshot_geometry(
            points,
            centers,
            path,
            edges,
            output,
            pcs,
            baseline,
            curve_label=curve_label,
            feature_label=feature_label,
            n_dimensions=coordinates.shape[1],
            polyline=polyline,
            compact=compact,
            baseline_label=baseline_label,
            recipe=settings,
        )
    geometry_label = (
        f"{len(centers) - 1} visual stages + observed terminal anchor"
        if "fine_stages" in state
        else f"{len(centers)} clusters"
    )
    print(
        f"Frozen global model: {points.cycle_name.nunique()} cycles; {len(points)} images; "
        f"{geometry_label}; native root {sling.start_node}; {state['completed_epochs']} epochs"
    )


def render_core_panels(
    source,
    admission_settings,
    cluster_study,
    comparison_source,
    output,
    *,
    fetch_images=False,
    n_jobs=6,
):
    """Actual cohort, feature, topology and photo evidence; one panel per PNG."""
    from image_models.cycle_selection import screen_admission
    from plots.frost_trajectory import plot_frost_construction

    screening = screen_admission(admission_settings)
    state = joblib.load(source / "fit_state.joblib")
    recipe = json.loads((source / "run_settings.json").read_text())
    frames = state["frames"]
    if (
        recipe.get("coordinate_correction") != "harmony"
        or recipe.get("trajectory_type") != "mst_polyline"
    ):
        raise ValueError("Core panels require the chosen native Harmony + MST polyline")
    if set(frames.cycle_name) != set(screening.loc[screening.selected, "cycle_name"]):
        raise ValueError("Rebuild trajectory when the authoritative cohort changes")
    output.mkdir(parents=True, exist_ok=True)
    screening.to_csv(output / "cycle_screening.csv", index=False)
    # Sequential display of already-computed gate outcomes, not a second selector.
    stages = [dict(stage="全部候选循环", cycles=len(screening))]
    retained = pd.Series(True, index=screening.index)
    for label, failures in [
        ("决策时点可用", "no_actions"),
        ("RGB / DINO 覆盖率 ≥70%", "rgb_coverage|dino_coverage"),
        ("插值比例 ≤30%", "interpolation"),
    ]:
        retained &= ~screening.gate_a_reasons.fillna("").str.contains(failures, regex=True)
        stages.append(dict(stage=label, cycles=int(retained.sum())))
    if not retained.equals(screening.gate_a):
        raise ValueError("The displayed gate recipe must match the current input rule")
    stages.append(dict(stage="终点 COP < 峰值的99%", cycles=int(screening.selected.sum())))
    stages = pd.DataFrame(stages)
    stages.to_csv(output / "selection_stages.csv", index=False)
    variance = pd.DataFrame(
        dict(
            pc=np.arange(1, state["pca"].n_components_ + 1),
            variance_fraction=state["pca"].explained_variance_ratio_,
            cumulative_variance=np.cumsum(state["pca"].explained_variance_ratio_),
        )
    )
    variance.to_csv(output / "pca_variance.csv", index=False)
    cases = pd.concat(
        [pd.read_csv(p) for p in sorted(cluster_study.rglob("cluster_cases.csv"))],
        ignore_index=True,
    )
    if cases.duplicated(["n_clusters", "seed"]).any():
        raise ValueError("Cluster studies must not duplicate count/seed cases")
    cases.to_csv(output / "cluster_count_source.csv", index=False)
    plot_frost_construction(stages, variance, cases, output, chosen_clusters=recipe["n_clusters"])
    compare_harmony_coordinates(source, admission_settings, output, compact=True)
    render_slingshot_geometry(source, output, compact=True, clusters_only=True)
    render_slingshot_geometry(source, output, compact=True)
    if comparison_source is not None:
        render_slingshot_geometry(source, output, comparison_source=comparison_source, compact=True)
    render_cluster_representatives(source, output, fetch_images=fetch_images, n_jobs=n_jobs)
    print(stages.to_string(index=False))
    print("All core panels saved as individual 600 dpi PNGs")


def render_cluster_representatives(source, output, *, fetch_images=False, n_jobs=6):
    """Nearest actual images for native clusters or explicit refined visual stages."""
    from dataset_tools.cycle_metadata import image_root
    from image_models.slingshot import project_curve, trajectory_geometry
    from plots.frost_trajectory import plot_cluster_image_strip

    state = joblib.load(source / "fit_state.joblib")
    recipe = json.loads((source / "run_settings.json").read_text())
    frames, sling = state["frames"], state["sling"]
    output.mkdir(parents=True, exist_ok=True)
    phase, points = trajectory_geometry(sling)
    scores, _ = project_curve(sling.data, phase, points, None)
    try:
        images = image_root(Path(recipe["dataset"]))
    except FileNotFoundError:
        images = Path("temp/test/frost_figure_photos").resolve()
    cache = Path("temp/test/frost_figure_photos").resolve()
    image_paths = [images / r.cycle_name / "front" / r.file_name for r in frames.itertuples()]
    image_paths = [
        p if p.is_file() else cache / r.cycle_name / "front" / r.file_name
        for p, r in zip(image_paths, frames.itertuples())
    ]
    readable = np.array([p.is_file() for p in image_paths])
    if fetch_images:
        from dataset_tools.cloud_images import available_image_archives

        readable |= frames.cycle_name.isin(available_image_archives()).to_numpy()
    representatives = []
    if "fine_stages" in state:
        groups = []
        for row in state["fine_stages"].itertuples():
            mask = (
                frames.fine_cluster.eq(row.fine_cluster)
                if row.kind == "visual_stage"
                else frames.cluster.eq(row.parent_cluster)
                & frames.backbone_coordinate.ge(row.guide_min)
            )
            groups.append(
                (
                    row.fine_cluster,
                    np.flatnonzero(mask),
                    sling.polyline_points[row.fine_cluster],
                    row.parent_order,
                    f"组 {row.fine_cluster}｜原簇 {row.parent_order}"
                    if row.kind == "visual_stage"
                    else "尾部样本中心",
                )
            )
    else:
        groups = [
            (
                int(c),
                np.flatnonzero(frames.cluster.to_numpy() == c),
                sling.cluster_centres[c],
                i,
                f"簇 {i}",
            )
            for i, c in enumerate(sling.lineages[0].clusters)
        ]
    for position, (cluster, ids, center, color_group, label) in enumerate(groups):
        distances = np.linalg.norm(sling.data[ids] - center, axis=1)
        order = np.argsort(distances, kind="stable")
        available_ranks = np.flatnonzero(readable[ids[order]])
        if not len(available_ranks):
            raise FileNotFoundError(f"No retrievable representative photo for cluster {cluster}")
        rank = int(available_ranks[0])
        nearest = ids[order[rank]]
        row = frames.iloc[nearest][["cycle_name", "image_time", "file_name"]].to_dict()
        representatives.append(
            dict(
                **row,
                cluster=int(cluster),
                path_order=position + 1,
                display_cluster=position,
                display_label=label,
                color_group=color_group,
                group_image_count=len(ids),
                group_cycle_count=frames.iloc[ids].cycle_name.nunique(),
                visual_frost_score=scores[nearest],
                center_distance=float(distances[order[rank]]),
                nearest_rank=rank + 1,
                selection_rule="nearest retrievable image within assigned cluster/stage or terminal support in Harmony 30D",
            )
        )
        representatives[-1]["image_path"] = str(image_paths[nearest])
    representatives = pd.DataFrame(representatives)
    representatives["representation_method"] = recipe["method"]
    representatives["image_processing"] = (
        "full frame; 90 degree counterclockwise rotation; no contrast/crop changes"
    )
    representatives.to_csv(output / "cluster_representatives.csv", index=False)
    if fetch_images:
        from dataset_tools.cloud_images import materialize_image_members

        missing = representatives.loc[~representatives.image_path.map(lambda p: Path(p).is_file())]
        requests = missing.groupby("cycle_name").file_name.agg(list).to_dict()
        with materialize_image_members(
            Path(recipe["dataset"]), requests, images_dir=cache, minimum_free_gib=0, n_jobs=n_jobs
        ) as completed:
            for _ in completed:
                pass
    if not representatives.image_path.map(lambda p: Path(p).is_file()).all():
        raise FileNotFoundError(
            "Core numerical panels are saved; representative photos require --fetch-images or the image drive"
        )
    plot_cluster_image_strip(representatives, output, recipe=recipe)
    return representatives


def render_refinement_panels(source, comparison_source, admission_settings, study, output):
    """Current-cohort evidence for the explicit observed-tail refinement recipe."""
    from image_models.cycle_selection import screen_admission
    from plots.frost_trajectory import plot_frost_method, plot_observed_refinement

    selected = set(screen_admission(admission_settings).query("selected").cycle_name)
    for run in (source, comparison_source):
        if set(pd.read_csv(run / "progression.csv").cycle_name) != selected:
            raise ValueError("Refinement figures require identical current screened cycle support")
    recipe = json.loads((source / "run_settings.json").read_text())
    cases = pd.read_csv(study / "observed_refinement_summary.csv")
    cases = cases.loc[cases.terminal_quantile.eq(recipe["terminal_quantile"])].copy()
    cases["method"] = "refined"
    cases["label"] = cases.fine_stages.map(
        lambda k: "7中心＋延长末端" if k == 7 else f"簇内分为{k}组＋延长末端"
    )
    old_frames = pd.read_csv(comparison_source / "progression.csv")
    old_cycles = pd.read_csv(comparison_source / "cycle_summary.csv")
    baseline = (
        pd.read_csv(source / "refinement_comparison.csv")
        .query('method == "baseline" and cohort == "main" and section == "late"')
        .iloc[0]
    )
    original = dict(
        method="original",
        label="原7中心连线",
        fine_stages=7,
        late_mean_squared_distance=baseline.mean_squared_distance,
        cycles_drawdown_above_01=int(old_cycles.max_drawdown.gt(0.1).sum()),
        endpoint_fraction=float(old_frames.is_endpoint.mean()),
        endpoint_images=int(old_frames.is_endpoint.sum()),
    )
    summary = pd.concat(
        [pd.DataFrame([original]), cases.sort_values("fine_stages")], ignore_index=True
    )
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "refinement_comparison_source.csv", index=False)
    stages = pd.read_csv(source / "fine_stages.csv")
    stages.to_csv(output / "fine_stages.csv", index=False)
    plot_observed_refinement(
        summary,
        stages,
        output,
        counts=dict(
            images=len(old_frames),
            cycles=len(old_cycles),
            late_images=int(baseline.image_count),
            terminal_fraction=100 * (1 - recipe["terminal_quantile"]),
        ),
        chosen_stages=int(stages.kind.eq("visual_stage").sum()),
        target_images=recipe["subcluster_target_images"],
    )
    state = joblib.load(source / "fit_state.joblib")
    frames = state["frames"]
    plot_frost_method(
        recipe,
        stages,
        output,
        counts=dict(
            images=len(frames),
            cycles=frames.cycle_name.nunique(),
            experiments=frames.experiment_id.nunique(),
            pca_variance=float(state["pca"].explained_variance_ratio_.sum()),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="frozen Slingshot run")
    parser.add_argument("--progression", type=Path, help="frozen existing reference progression")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument(
        "--compare-harmony", action="store_true", help="paired native PCA/Harmony panels"
    )
    parser.add_argument(
        "--trajectory", action="store_true", help="saved global MST and principal curve"
    )
    parser.add_argument(
        "--comparison-source", type=Path, help="overlay a frozen curve in identical coordinates"
    )
    parser.add_argument(
        "--compact", action="store_true", help="one tight trajectory panel without title or caption"
    )
    parser.add_argument(
        "--core-panels",
        action="store_true",
        help="individual paper panels for the chosen frost-state construction",
    )
    parser.add_argument("--cluster-study", type=Path, help="saved count/seed sensitivity tables")
    parser.add_argument(
        "--fetch-images",
        action="store_true",
        help="materialize only the selected cluster representative photos",
    )
    parser.add_argument(
        "--representative-images",
        action="store_true",
        help="actual native-cluster or refined-stage image plate",
    )
    parser.add_argument(
        "--refinement-panels",
        action="store_true",
        help="observed-tail saturation and stage-resolution evidence",
    )
    parser.add_argument("--admission-settings", type=Path, help="recompute the training cohort")
    parser.add_argument(
        "--pcs", type=int, nargs=2, default=[1, 2], help="same saved dimensions in both panels"
    )
    args = parser.parse_args()
    if args.compare_harmony and args.admission_settings is None:
        parser.error("--compare-harmony requires --admission-settings")
    if args.trajectory and args.compare_harmony:
        parser.error("Choose --trajectory or --compare-harmony")
    if args.core_panels and (args.admission_settings is None or args.cluster_study is None):
        parser.error("--core-panels requires --admission-settings and --cluster-study")
    if args.refinement_panels and any(
        x is None for x in (args.admission_settings, args.cluster_study, args.comparison_source)
    ):
        parser.error(
            "--refinement-panels requires --admission-settings, --cluster-study and --comparison-source"
        )
    if (
        not args.core_panels
        and not args.refinement_panels
        and not args.compare_harmony
        and not args.trajectory
        and not args.representative_images
        and args.progression is None
    ):
        parser.error("--progression is required for the representation audit")
    from output_layout import prepare_output

    prepare_output(args, training=False)
    with threadpool_limits(limits=1):
        if args.refinement_panels:
            render_refinement_panels(
                args.source,
                args.comparison_source,
                args.admission_settings,
                args.cluster_study,
                args.output,
            )
        elif args.core_panels:
            render_core_panels(
                args.source,
                args.admission_settings,
                args.cluster_study,
                args.comparison_source,
                args.output,
                fetch_images=args.fetch_images,
                n_jobs=args.n_jobs,
            )
        elif args.trajectory:
            render_slingshot_geometry(
                args.source, args.output, args.pcs, args.comparison_source, args.compact
            )
        elif args.representative_images:
            render_cluster_representatives(
                args.source, args.output, fetch_images=args.fetch_images, n_jobs=args.n_jobs
            )
        elif args.compare_harmony:
            compare_harmony_coordinates(
                args.source, args.admission_settings, args.output, args.pcs, args.compact
            )
        else:
            audit_representation(
                args.source, args.progression, args.output, args.n_jobs, args.admission_settings
            )


if __name__ == "__main__":
    main()
