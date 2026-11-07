"""Pooled front-DINO → PCA → native Slingshot; stop if the MST has multiple lineages.

This coordinate is relative visual geometry, not a calibrated physical frost mass.
Time anchors the start cluster and audits the result; it never repairs pseudotime.
"""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from image_models.dinov2_features import KEYS, POOL_COLUMNS
from image_models.frost_progression import load_front_features, phase_bins
from output_layout import PROJECT


def scalar_distances():
    """Upstream's 1×1 Mahalanobis value as a scalar; restore the function on exit."""
    from pyslingshot import slingshot as implementation

    distance = implementation.mahalanobis
    return patch.object(
        implementation, "mahalanobis", lambda *args: np.asarray(distance(*args)).item()
    )


def bounded_curve_projection(n_points):
    """Apply the explicit curve approximation to upstream's repeated projection passes."""
    from pcurvepy2 import PrincipalCurve

    original = PrincipalCurve.project_to_curve

    def project(curve, data, points=None, stretch=0.0):
        polygon = curve.points if points is None else points
        if polygon is not None and n_points and len(polygon) > n_points:
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(polygon, axis=0), axis=1))]
            arc, indices = np.unique(arc, return_index=True)
            targets = np.linspace(arc[0], arc[-1], n_points)
            polygon = np.column_stack(
                [np.interp(targets, arc, polygon[indices, j]) for j in range(polygon.shape[1])]
            )
        return original(curve, data, points=polygon, stretch=stretch)

    return patch.object(PrincipalCurve, "project_to_curve", project)


def aggregate_curve_samples(parameter, data, weights):
    """Exact weighted least-squares sufficient statistics at tied curve positions.

    SciPy spline weights multiply residuals, so the statistical weights are w².
    Within-position scatter is constant with respect to the fitted spline.
    """
    positions, inverse = np.unique(parameter, return_inverse=True)
    squared = np.square(weights)
    mass = np.bincount(inverse, weights=squared)
    totals = np.zeros((len(positions), data.shape[1]))
    np.add.at(totals, inverse, data * squared[:, None])
    positive = mass > 0
    return positions[positive], totals[positive] / mass[positive, None], np.sqrt(mass[positive])


def fit_slingshot(sling, epochs=1):
    """Use the same scalar adapter when native fit recomputes its initial MST."""
    if getattr(sling, "trajectory_type", "principal_curve") == "mst_polyline":
        return
    from pcurvepy2 import PrincipalCurve, pcurve
    from scipy.interpolate import LSQUnivariateSpline

    original = PrincipalCurve.fit
    native_spline = pcurve.UnivariateSpline
    residual = getattr(sling, "spline_residual", None)
    degrees = getattr(sling, "spline_df", None)
    knot_placement = getattr(sling, "spline_knot_placement", "quantile")
    aggregate = getattr(sling, "aggregate_ties", False)
    if residual is not None and (not np.isfinite(residual) or residual <= 0):
        raise ValueError("Spline residual budget must be finite and positive")
    if degrees is not None and (degrees < 4 or residual is not None):
        raise ValueError("Cubic spline df must be >=4; choose df or residual budget")
    if knot_placement not in {"quantile", "uniform"} or (
        knot_placement == "uniform" and degrees is None
    ):
        raise ValueError("Uniform spline knots require a fixed spline df")

    def fit(curve, data, *args, **kwargs):
        # Native SciPy s=None permits roughly one unit of squared residual per image
        # in each PC. An explicit budget is needed for small PCA coordinate units.
        if residual is not None:
            kwargs["param_s"] = residual * len(data)
        statistics = None
        within_scatter = None
        if aggregate:
            weights = kwargs.get("weights", np.ones(len(data)))
            statistics = aggregate_curve_samples(curve.pseudotimes_interp, data, weights)
            positions, means, _ = statistics
            ids = np.searchsorted(positions, curve.pseudotimes_interp)
            positive = weights > 0
            within_scatter = np.sum(
                (weights[positive, None] * (data[positive] - means[ids[positive]])) ** 2, axis=0
            )
        dimension = 0

        def spline(x, y, k=3, s=None, w=None):
            nonlocal dimension
            if statistics is not None:
                x, means, w = statistics
                y = means[:, dimension]
                if residual is not None:
                    # Group scatter is constant: subtract it from the full-data budget.
                    # If the budget is below that floor, interpolate the group means.
                    s = max(0.0, s - within_scatter[dimension])
                dimension += 1
            if degrees is None:
                return native_spline(x, y, k=k, s=s, w=w)
            # df coefficients = interior knots + degree + 1.
            fractions = np.arange(1, degrees - k) / (degrees - k)
            knots = (
                x[0] + fractions * (x[-1] - x[0])
                if knot_placement == "uniform"
                else np.quantile(x, fractions)
            )
            return LSQUnivariateSpline(x, y, knots, k=k, w=w)

        with patch.object(pcurve, "UnivariateSpline", spline):
            return original(curve, data, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(scalar_distances())
        stack.enter_context(bounded_curve_projection(sling.approx_points))
        if residual is not None or degrees is not None or aggregate:
            stack.enter_context(patch.object(PrincipalCurve, "fit", fit))
        sling.fit(num_epochs=epochs)


def initialize_slingshot(coordinates, labels, frames, *, approx_points=150):
    """One first-frame vote per cycle anchors the native MST; no forced lineage."""
    from pyslingshot import Slingshot

    metadata = frames[["cycle_name", "image_time", "file_name"]].copy()
    metadata["cluster"] = labels
    first = metadata.sort_values(["cycle_name", "image_time", "file_name"]).drop_duplicates(
        "cycle_name"
    )
    clusters = metadata.groupby("cluster").agg(
        image_count=("file_name", "size"), cycle_count=("cycle_name", "nunique")
    )
    clusters["first_frame_votes"] = first.cluster.value_counts().reindex(
        clusters.index, fill_value=0
    )
    start = int(clusters.first_frame_votes.idxmax())
    clusters["is_start"] = clusters.index == start
    onehot = np.eye(len(clusters))[labels]
    sling = Slingshot(
        coordinates,
        cluster_labels_onehot=onehot,
        start_node=start,
        approx_points=approx_points,
        is_debugging=False,
    )
    with scalar_distances():
        sling.get_lineages()
    lineages = pd.DataFrame(
        [
            dict(lineage=i, position=j, cluster=int(c))
            for i, lineage in enumerate(sling.lineages)
            for j, c in enumerate(lineage.clusters)
        ]
    )
    return sling, clusters.reset_index(), lineages


def monotonicity_summary(frames):
    """Audit unmodified projections; quantify tiny reversals as well as large drawdowns."""
    rows = []
    for cycle, group in frames.groupby("cycle_name", sort=True):
        group = group.sort_values(["image_time", "file_name"])
        s = group.visual_frost_score.to_numpy(float)
        changes = np.diff(s)
        decreases = changes < -1e-12
        elapsed = (
            pd.to_datetime(group.image_time) - pd.to_datetime(group.image_time).iloc[0]
        ).dt.total_seconds()
        rho = float(spearmanr(elapsed, s).statistic) if len(s) > 1 and np.ptp(s) > 0 else np.nan
        rows.append(
            dict(
                cycle_name=cycle,
                image_count=len(s),
                transition_count=len(changes),
                n_decreases=int(decreases.sum()),
                decrease_fraction=float(decreases.mean()) if len(changes) else np.nan,
                max_step_decrease=float(max(0.0, -changes.min())) if len(changes) else 0.0,
                max_drawdown=float((np.maximum.accumulate(s) - s).max()),
                time_spearman=rho,
                reference_min_observed=s.min(),
                max_visual_frost_score=s.max(),
                reached_endpoint=bool(s.max() >= 1.0),
            )
        )
    return pd.DataFrame(rows)


def curve_anchors(curve_phase, curve_points, n_points):
    """Uniform positions on the fitted curve's fixed 0–1 parameterization."""
    phase, indices = np.unique(curve_phase, return_index=True)
    points = curve_points[indices]
    targets = np.linspace(0.0, 1.0, n_points)
    anchors = np.column_stack(
        [np.interp(targets, phase, points[:, j]) for j in range(points.shape[1])]
    )
    return targets, anchors


def trajectory_geometry(sling):
    """Saved principal curve or exact cluster-center path; never smooth the latter."""
    if getattr(sling, "trajectory_type", "principal_curve") == "mst_polyline":
        if len(sling.lineages) != 1:
            raise ValueError("A shared polyline coordinate requires one native MST lineage")
        points = (
            sling.polyline_points
            if hasattr(sling, "polyline_points")
            else sling.cluster_centres[np.asarray(sling.lineages[0].clusters, dtype=int)]
        )
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if arc[-1] <= 0:
            raise ValueError("Cluster-center path has zero length")
        return arc / arc[-1], points
    curve = sling.curves[0]
    raw = np.asarray(curve.pseudotimes_interp)
    return (raw - raw.min()) / np.ptp(raw), curve.points_interp


def refine_observed_polyline(sling, frames, target_images, terminal_quantile=0.99):
    """Density-adaptive visual stages on the frozen native MST, never time sorting.

    The last native edge orders its observed overflow. Within each native cluster,
    equal-count visual-coordinate bins get empirical 30D means. The terminal anchor
    averages per-cycle means of the observed upper tail, avoiding one outlier endpoint.
    This is MST-guided segmentation, not a newly inferred native Slingshot MST.
    """
    if target_images < 1 or not 0.9 <= terminal_quantile < 1:
        raise ValueError("Positive target image count and terminal quantile in [0.9, 1) required")
    if getattr(sling, "trajectory_type", None) != "mst_polyline" or len(sling.lineages) != 1:
        raise ValueError("Refinement requires a frozen single native center polyline")
    native = sling.cluster_centres[np.asarray(sling.lineages[0].clusters, dtype=int)]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(native, axis=0), axis=1))]
    guide, _ = project_curve(sling.data, arc / arc[-1], native, None)
    delta = native[-1] - native[-2]
    u = (sling.data - native[-2]) @ delta / (delta @ delta)
    overflow = (guide >= 1.0) & (u > 1.0)
    guide[overflow] = 1.0 + (u[overflow] - 1.0) * (arc[-1] - arc[-2]) / arc[-1]
    first_delta = native[1] - native[0]
    first_u = (sling.data - native[0]) @ first_delta / (first_delta @ first_delta)
    before = (guide <= 0.0) & (first_u < 0.0)
    guide[before] = first_u[before] * arc[1] / arc[-1]
    refined = frames.copy()
    refined["backbone_coordinate"] = guide
    labels = np.full(len(frames), -1, dtype=int)
    vertices, rows = [], []
    for parent_order, parent in enumerate(sling.lineages[0].clusters):
        ids = np.flatnonzero(frames.cluster.to_numpy() == parent)
        bins = pd.qcut(
            guide[ids],
            max(1, int(np.ceil(len(ids) / target_images))),
            labels=False,
            duplicates="drop",
        )
        groups = [ids] if np.isnan(bins).all() else [ids[bins == b] for b in np.unique(bins)]
        for group in groups:
            stage = len(rows)
            labels[group] = stage
            vertices.append(sling.data[group].mean(axis=0))
            rows.append(
                dict(
                    fine_cluster=stage,
                    parent_cluster=int(parent),
                    parent_order=parent_order,
                    image_count=len(group),
                    cycle_count=frames.iloc[group].cycle_name.nunique(),
                    experiment_count=frames.iloc[group].experiment_id.nunique(),
                    guide_min=float(guide[group].min()),
                    guide_max=float(guide[group].max()),
                    guide_mean=float(guide[group].mean()),
                    kind="visual_stage",
                )
            )
    terminal = frames.cluster.eq(sling.lineages[0].clusters[-1]).to_numpy()
    tail = terminal & (u >= np.quantile(u[terminal], terminal_quantile))
    if tail.sum() < 2 or frames.loc[tail, "cycle_name"].nunique() < 2:
        raise ValueError("Observed tail requires support from at least two cycles")
    per_cycle = [
        sling.data[np.flatnonzero(tail & frames.cycle_name.eq(cycle).to_numpy())].mean(axis=0)
        for cycle in sorted(frames.loc[tail, "cycle_name"].unique())
    ]
    endpoint = np.mean(per_cycle, axis=0)
    if (endpoint - vertices[-1]) @ delta <= 0:
        raise ValueError("Tail support does not extend beyond the final substage center")
    vertices.append(endpoint)
    rows.append(
        dict(
            fine_cluster=len(rows),
            parent_cluster=int(sling.lineages[0].clusters[-1]),
            parent_order=len(sling.lineages[0].clusters) - 1,
            image_count=int(tail.sum()),
            cycle_count=frames.loc[tail, "cycle_name"].nunique(),
            experiment_count=frames.loc[tail, "experiment_id"].nunique(),
            guide_min=float(guide[tail].min()),
            guide_max=float(guide[tail].max()),
            guide_mean=float(guide[tail].mean()),
            kind="terminal_anchor",
        )
    )
    sling.polyline_points = np.asarray(vertices)
    refined["fine_cluster"] = labels
    return refined, pd.DataFrame(rows)


def project_curve(query_coordinates, curve_phase, curve_points, n_points=150):
    """Nearest polygon segment, with its fixed parameter; independent of other queries.

    None uses every stored curve vertex, for the approximation audit.
    """
    if n_points is None:
        targets, ids = np.unique(curve_phase, return_index=True)
        anchors = curve_points[ids]
    else:
        targets, anchors = curve_anchors(curve_phase, curve_points, n_points)
    score = np.zeros(len(query_coordinates))
    squared = np.full(len(query_coordinates), np.inf)
    for j, delta in enumerate(np.diff(anchors, axis=0)):
        length_sq = delta @ delta
        if length_sq == 0:
            continue
        offset = query_coordinates - anchors[j]
        u = np.clip(offset @ delta / length_sq, 0.0, 1.0)
        residual = offset - u[:, None] * delta
        distance_sq = np.einsum("ij,ij->i", residual, residual)
        nearer = distance_sq < squared
        squared[nearer] = distance_sq[nearer]
        score[nearer] = targets[j] + u[nearer] * (targets[j + 1] - targets[j])
    return score, np.sqrt(squared)


def reference_atlas(frames, coordinates, curve_phase, curve_points, n_reference=100):
    """Nearest real-image embedding to each uniformly spaced curve position.

    Repeated nearest images are recorded honestly; no uniqueness constraint changes the rule.
    reference_phase is the target, visual_frost_score is the chosen image's actual projection.
    """
    targets, anchors = curve_anchors(curve_phase, curve_points, n_reference)
    # Cached features still identify real images even when pixels are stored remotely.
    distances, nearest = cKDTree(coordinates).query(anchors, workers=1)
    atlas = frames.iloc[nearest].copy().reset_index(drop=True)
    atlas["reference_index"] = np.arange(n_reference)
    atlas["reference_phase"] = targets
    atlas["atlas_distance"] = distances
    return atlas


def harmony_coordinates(coordinates, frames, seed=0, **options):
    """Harmony on all PCs; no time/reference labels enter correction."""
    from harmonypy import run_harmony

    model = run_harmony(
        coordinates.copy(),
        frames[["experiment_id"]],
        "experiment_id",
        random_state=seed,
        verbose=True,
        **options,
    )
    corrected = np.asarray(model.Z_corr.T, dtype=float)
    if corrected.shape != coordinates.shape or not np.isfinite(corrected).all():
        raise ValueError("Harmony returned invalid coordinates")
    return corrected, model


def prepare_slingshot(
    valid,
    *,
    pca_components=30,
    n_clusters=12,
    seed=0,
    approx_points=150,
    correction="none",
    pca_solver="randomized",
    pca_whiten=False,
    kmeans_n_init=10,
    kmeans_max_iter=300,
    kmeans_tol=1e-4,
    spline_residual=None,
    spline_df=None,
    spline_knot_placement="quantile",
    aggregate_ties=False,
    feature_pooling="cls",
    trajectory_type="principal_curve",
    **harmony_options,
):
    """Prepare pooled features and fit the native frost trajectory."""
    columns = POOL_COLUMNS[feature_pooling]
    z = valid[columns].to_numpy(float)
    z /= np.linalg.norm(z, axis=1)[:, None]
    pca = PCA(
        n_components=pca_components, svd_solver=pca_solver, whiten=pca_whiten, random_state=seed
    )
    print(
        f"[PCA] {len(valid)} images, {valid.cycle_name.nunique()} cycles "
        f"→ {pca_components} components",
        flush=True,
    )
    coordinates = pca.fit_transform(z)
    raw_coordinates = coordinates.copy()
    harmony = None
    if correction == "harmony":
        print("[Harmony] experiment_id correction; all PCs and frames retained", flush=True)
        coordinates, harmony = harmony_coordinates(coordinates, valid, seed, **harmony_options)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=kmeans_n_init,
        max_iter=kmeans_max_iter,
        tol=kmeans_tol,
    )
    labels = kmeans.fit_predict(coordinates)
    valid = valid.drop(columns=columns).assign(cluster=labels)
    print(f"[Slingshot] native MST over {n_clusters} clusters", flush=True)
    sling, clusters, lineages = initialize_slingshot(
        coordinates, labels, valid, approx_points=approx_points
    )
    sling.spline_residual = spline_residual
    sling.spline_df = spline_df
    sling.spline_knot_placement = spline_knot_placement
    sling.aggregate_ties = aggregate_ties
    sling.trajectory_type = trajectory_type
    state = dict(frames=valid, pca=pca, kmeans=kmeans, sling=sling, completed_epochs=0)
    if feature_pooling != "cls":
        state["feature_pooling"] = feature_pooling
    if harmony is not None:
        state.update(raw_coordinates=raw_coordinates, harmony=harmony)
    return state, clusters, lineages


def compute_slingshot(
    dataset,
    feature_caches,
    output,
    *,
    pca_components=30,
    n_clusters=12,
    epochs=10,
    approx_points=150,
    n_reference=100,
    seed=0,
    resume=False,
    correction="none",
    refine_source=None,
    recluster=False,
    admission_settings=None,
    image_list=None,
    subcluster_target_images=None,
    terminal_quantile=0.99,
    **model_options,
):
    """Shared front loader and schemas; one native curve, or an explicit branched result."""
    output = Path(output)
    polyline = model_options.get("trajectory_type", "principal_curve") == "mst_polyline"
    settings = dict(
        method="mst_polyline" if polyline else "slingshot",
        dataset=str(Path(dataset).resolve()),
        feature_caches=[str(Path(p).resolve()) for p in feature_caches],
        frost_only=True,
        feature_normalization="unit_l2",
        pca_components=pca_components,
        n_clusters=n_clusters,
        epochs=0 if polyline else epochs,
        approx_points=approx_points,
        n_reference=n_reference,
        seed=seed,
        start_rule="first_frame_vote_per_cycle",
        branching_rule="record_and_stop",
        monotonicity_correction="none",
    )
    if correction == "harmony":
        settings["coordinate_correction"] = correction
        settings["harmony_version"] = "0.0.10"
    if model_options:
        defaults = model_parameters(add_model_arguments(argparse.ArgumentParser()).parse_args([]))
        # Preserve the frozen recipe's settings when the added knobs retain their old defaults.
        settings.update(
            {key: value for key, value in model_options.items() if value != defaults[key]}
        )
    if refine_source is not None:
        settings["refine_source"] = str(Path(refine_source).resolve())
    if admission_settings is not None:
        settings["admission_settings"] = str(Path(admission_settings).resolve())
    if subcluster_target_images is not None:
        if refine_source is None or not polyline or recluster or admission_settings is None:
            raise ValueError(
                "Observed refinement requires a native polyline source and current admission settings"
            )
        settings.update(
            method="mst_guided_visual_segments",
            subcluster_target_images=subcluster_target_images,
            terminal_quantile=terminal_quantile,
            stage_rule="within-native-cluster visual-coordinate quantiles; equal sample density",
            endpoint_rule="upper observed terminal-direction tail; equal-cycle mean; frozen arc normalization",
        )
    if image_list is not None:
        if refine_source is not None:
            raise ValueError("Image support changes require a fresh fit")
        settings["image_list"] = str(Path(image_list).resolve())
    if recluster:
        if refine_source is None:
            raise ValueError("Reclustering requires a frozen coordinate source")
        settings["recluster"] = True
    if resume:
        saved = json.loads((output / "run_settings.json").read_text())
        if saved != settings:
            raise ValueError(
                "Resume requires the same inputs and recipe; use a new output for changes"
            )
        state = joblib.load(output / "fit_state.joblib")
    else:
        if (output / "run_settings.json").exists():
            raise FileExistsError("Existing run: use --resume or a new output directory")
        output.mkdir(parents=True, exist_ok=True)
        selected = None
        if admission_settings is not None:
            from image_models.cycle_selection import screen_admission

            screening = screen_admission(admission_settings)
            screening.to_csv(output / "cycle_screening.csv", index=False)
            selected = set(screening.loc[screening.selected, "cycle_name"])
        if refine_source is None:
            pooling = model_options.get("feature_pooling", "cls")
            columns = POOL_COLUMNS[pooling]
            frames = load_front_features(
                dataset,
                feature_caches,
                pooling=pooling,
                cycles=selected,
                image_list=image_list,
            )
            frames.drop(columns=columns).to_csv(output / "frame_inventory.csv", index=False)
            if selected is not None and selected - set(
                frames.loc[frames.embedding_available, "cycle_name"]
            ):
                raise ValueError(
                    "Some screened cycles lack embeddings; prepare inputs before fitting"
                )
            valid = frames.loc[frames.embedding_available].reset_index(drop=True)
            if len(valid) <= max(pca_components, n_clusters):
                raise ValueError("Insufficient frozen embeddings for the requested PCA/clusters")
            state, clusters, lineages = prepare_slingshot(
                valid,
                pca_components=pca_components,
                n_clusters=n_clusters,
                seed=seed,
                approx_points=approx_points,
                correction=correction,
                **model_options,
            )
        else:
            source = Path(refine_source)
            prior = json.loads((source / "run_settings.json").read_text())
            if correction != "harmony" or prior.get("coordinate_correction") != "harmony":
                raise ValueError("Refinement reuses native global Harmony coordinates")
            if pca_components != prior["pca_components"] or (
                not recluster and n_clusters != prior["n_clusters"]
            ):
                raise ValueError(
                    "Preserve PCA and cluster count, or explicitly request --recluster"
                )
            defaults = model_parameters(
                add_model_arguments(argparse.ArgumentParser()).parse_args([])
            )
            if seed != prior["seed"] or any(
                value != prior.get(key, defaults[key])
                for key, value in model_options.items()
                if key
                not in {
                    "spline_residual",
                    "spline_df",
                    "spline_knot_placement",
                    "aggregate_ties",
                    "trajectory_type",
                }
            ):
                raise ValueError(
                    "Frozen-coordinate refinement must retain the source PCA/Harmony/KMeans recipe"
                )
            state = joblib.load(source / "fit_state.joblib")
            frames = state["frames"]
            if selected is not None and set(frames.cycle_name) != selected:
                raise ValueError("A changed screened cohort requires a fresh fit")
            if recluster:
                state["kmeans"] = clone(state["kmeans"]).set_params(n_clusters=n_clusters)
                frames = frames.assign(cluster=state["kmeans"].fit_predict(state["sling"].data))
                state["frames"] = frames
            pd.read_csv(source / "frame_inventory.csv").to_csv(
                output / "frame_inventory.csv", index=False
            )
            sling, clusters, lineages = initialize_slingshot(
                state["sling"].data, frames.cluster.to_numpy(), frames, approx_points=approx_points
            )
            sling.spline_residual = model_options.get("spline_residual")
            sling.spline_df = model_options.get("spline_df")
            sling.spline_knot_placement = model_options.get("spline_knot_placement", "quantile")
            sling.aggregate_ties = model_options.get("aggregate_ties", False)
            sling.trajectory_type = model_options.get("trajectory_type", "principal_curve")
            state.update(sling=sling, completed_epochs=0)
            if subcluster_target_images is not None:
                frames, stages = refine_observed_polyline(
                    sling, frames, subcluster_target_images, terminal_quantile
                )
                state.update(frames=frames, fine_stages=stages)
                stages.to_csv(output / "fine_stages.csv", index=False)
        clusters.to_csv(output / "clusters.csv", index=False)
        lineages.to_csv(output / "lineages.csv", index=False)
        output.joinpath("run_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        joblib.dump(state, output / "fit_state.joblib")
    frames, sling = state["frames"], state["sling"]
    inventory = pd.read_csv(output / "frame_inventory.csv", dtype={key: str for key in KEYS}).drop(
        columns="cluster", errors="ignore"
    )
    inventory = inventory.merge(
        frames[KEYS + ["cluster"]], on=KEYS, how="left", validate="one_to_one"
    )
    inventory.to_csv(output / "frame_inventory.csv", index=False)
    summary = dict(
        image_count=len(frames),
        cycle_count=frames.cycle_name.nunique(),
        pca_explained_variance=float(state["pca"].explained_variance_ratio_.sum()),
        lineage_count=len(sling.lineages),
        start_cluster=int(sling.start_node),
        projection_approx_points=None if polyline else approx_points,
    )
    if "fine_stages" in state:
        summary["fine_stage_count"] = int(state["fine_stages"].kind.eq("visual_stage").sum())
        summary["terminal_anchor_images"] = int(state["fine_stages"].iloc[-1].image_count)
    if len(sling.lineages) != 1:
        summary["status"] = "branched_stopped"
        pd.DataFrame([summary]).to_csv(output / "fit_summary.csv", index=False)
        print(
            f"[STOP] {len(sling.lineages)} lineages: no shared frost coordinate or atlas generated",
            flush=True,
        )
        return None, None
    for epoch in range(state["completed_epochs"], 0 if polyline else epochs):
        print(f"[curve] epoch {epoch + 1}/{epochs}", flush=True)
        fit_slingshot(sling)
        state["completed_epochs"] = epoch + 1
        curve = sling.curves[0]
        raw = np.asarray(curve.pseudotimes_interp)
        score, distance = project_curve(
            sling.data, (raw - raw.min()) / np.ptp(raw), curve.points_interp, approx_points
        )
        metrics = monotonicity_summary(frames.assign(visual_frost_score=score))
        record = dict(
            epoch=epoch + 1,
            mean_squared_distance=float(np.mean(distance**2)),
            q95_distance=float(np.quantile(distance, 0.95)),
            median_time_spearman=float(metrics.time_spearman.median()),
            cycles_rho_below_09=int(metrics.time_spearman.lt(0.9).sum()),
        )
        history_path = output / "fit_history.csv"
        history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
        pd.DataFrame([*history, record]).to_csv(history_path, index=False)
        joblib.dump(state, output / "fit_state.joblib")
    phase, points = trajectory_geometry(sling)
    if polyline:
        pd.DataFrame(
            points, columns=[f"coordinate_{j + 1}" for j in range(points.shape[1])]
        ).assign(phase=phase).to_csv(output / "trajectory_vertices.csv", index=False)
    if not np.isfinite(phase).all() or np.ptp(phase) <= 0:
        raise ValueError("Slingshot produced nonfinite or constant pseudotime")
    scores, distances = project_curve(
        sling.data, phase, points, None if polyline else approx_points
    )
    beyond = np.zeros(len(frames), dtype=bool)
    if "fine_stages" in state:
        delta = points[-1] - points[-2]
        u = (sling.data - points[-2]) @ delta / (delta @ delta)
        beyond = (scores >= 1.0) & (u > 1.0)
    raw = scores if polyline else np.asarray(sling.curves[0].pseudotimes_interp)
    frames = frames.assign(
        visual_frost_score=scores,
        projection_distance=distances,
        reference_index=np.floor(scores * (n_reference - 1) + 0.5).astype(int),
        phase_bin=phase_bins(scores),
        reference_cycle="pooled",
        reference_method=settings["method"],
        coordinate_correction=correction,
        frost_only=True,
        is_beyond_reference=beyond,
        is_endpoint=scores >= 1.0,
    )
    frames["mst_polyline_pseudotime" if polyline else "slingshot_pseudotime"] = raw
    frames.to_csv(output / "progression.csv", index=False)
    monotonicity_summary(frames).to_csv(output / "cycle_summary.csv", index=False)
    atlas = reference_atlas(frames, sling.data, phase, points, n_reference)
    atlas.to_csv(output / "reference.csv", index=False)
    counts = (
        frames.groupby("phase_bin")
        .agg(image_count=("file_name", "size"), cycle_count=("cycle_name", "nunique"))
        .reindex(range(11), fill_value=0)
    )
    counts["reference_phase"] = counts.index / 10
    counts["label"] = [f"{i / 10:.1f}" for i in range(11)]
    counts.to_csv(output / "phase_counts.csv", index=False)
    summary.update(
        status="fitted",
        completed_epochs=0 if polyline else epochs,
        atlas_unique_images=atlas[["cycle_name", "file_name"]].drop_duplicates().shape[0],
    )
    pd.DataFrame([summary]).to_csv(output / "fit_summary.csv", index=False)
    print(f"[done] {summary}", flush=True)
    return frames, counts


def add_model_arguments(parser, *, correction="none"):
    """Expose the PCA, clustering and trajectory fitting parameters together."""
    model = parser.add_argument_group("PCA / KMeans / Slingshot hyperparameters")
    model.add_argument("--pca-components", type=int, default=30)
    model.add_argument("--pca-solver", choices=("randomized", "full"), default="randomized")
    model.add_argument("--pca-whiten", action="store_true")
    model.add_argument("--n-clusters", type=int, default=12)
    model.add_argument("--kmeans-n-init", type=int, default=10)
    model.add_argument("--kmeans-max-iter", type=int, default=300)
    model.add_argument("--kmeans-tol", type=float, default=1e-4)
    model.add_argument("--epochs", type=int, default=10)
    model.add_argument("--approx-points", type=int, default=150)
    model.add_argument(
        "--spline-residual",
        type=float,
        default=None,
        help="per-image squared-residual budget in each PC; None preserves native smoothing",
    )
    model.add_argument(
        "--spline-df",
        type=int,
        default=None,
        help="fixed cubic B-spline coefficient count; >=4; mutually exclusive with residual",
    )
    model.add_argument(
        "--spline-knot-placement",
        choices=("quantile", "uniform"),
        default="quantile",
        help="quantile preserves the fixed-df recipe; uniform uses equal arc-parameter spacing",
    )
    model.add_argument(
        "--aggregate-ties",
        action="store_true",
        help="retain all images at tied projections with exact weighted spline statistics",
    )
    model.add_argument("--feature-pooling", choices=list(POOL_COLUMNS), default="cls")
    model.add_argument(
        "--trajectory-type",
        choices=("principal_curve", "mst_polyline"),
        default="principal_curve",
        help="native smooth curve, or exact arc-length projection onto ordered cluster centers",
    )
    model.add_argument("--seed", type=int, default=0)
    model.add_argument(
        "--correction",
        choices=("none", "harmony"),
        default=correction,
        help="Harmony is a separate recipe; audit shared state support first",
    )
    harmony = parser.add_argument_group("Harmony hyperparameters (harmonypy 0.0.10 defaults)")
    harmony.add_argument("--harmony-theta", type=float, default=1.0)
    harmony.add_argument("--harmony-lambda", type=float, default=1.0)
    harmony.add_argument("--harmony-sigma", type=float, default=0.1)
    harmony.add_argument(
        "--harmony-clusters", type=int, default=None, help="default: min(N/30, 100)"
    )
    harmony.add_argument("--harmony-tau", type=float, default=0.0)
    harmony.add_argument("--harmony-block-size", type=float, default=0.05)
    harmony.add_argument("--harmony-max-iter", type=int, default=10)
    harmony.add_argument("--harmony-kmeans-max-iter", type=int, default=20)
    harmony.add_argument("--harmony-cluster-tol", type=float, default=1e-5)
    harmony.add_argument("--harmony-tol", type=float, default=1e-4)
    return parser


def model_parameters(args):
    """Explicit CLI → model mapping; epochs belongs to fitting, not preparation."""
    return dict(
        pca_components=args.pca_components,
        pca_solver=args.pca_solver,
        pca_whiten=args.pca_whiten,
        n_clusters=args.n_clusters,
        kmeans_n_init=args.kmeans_n_init,
        kmeans_max_iter=args.kmeans_max_iter,
        kmeans_tol=args.kmeans_tol,
        epochs=args.epochs,
        approx_points=args.approx_points,
        spline_residual=args.spline_residual,
        spline_df=args.spline_df,
        spline_knot_placement=args.spline_knot_placement,
        aggregate_ties=args.aggregate_ties,
        feature_pooling=args.feature_pooling,
        trajectory_type=args.trajectory_type,
        seed=args.seed,
        correction=args.correction,
        theta=args.harmony_theta,
        lamb=args.harmony_lambda,
        sigma=args.harmony_sigma,
        nclust=args.harmony_clusters,
        tau=args.harmony_tau,
        block_size=args.harmony_block_size,
        max_iter_harmony=args.harmony_max_iter,
        max_iter_kmeans=args.harmony_kmeans_max_iter,
        epsilon_cluster=args.harmony_cluster_tol,
        epsilon_harmony=args.harmony_tol,
    )


def main():
    parser = add_model_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    parser.add_argument("--feature-cache", type=Path, nargs="+")
    parser.add_argument(
        "--refine-source", type=Path, help="reuse frozen native coordinates and clustering"
    )
    parser.add_argument(
        "--recluster",
        action="store_true",
        help="refit KMeans in frozen coordinates at --n-clusters",
    )
    parser.add_argument(
        "--subcluster-target-images",
        type=int,
        help="MST-guided visual-coordinate segmentation; adaptive stages within native clusters",
    )
    parser.add_argument(
        "--terminal-quantile",
        type=float,
        default=0.99,
        help="observed terminal tail used for an equal-cycle endpoint anchor",
    )
    parser.add_argument(
        "--admission-settings",
        type=Path,
        help="screen full saved candidates before image filtering",
    )
    parser.add_argument(
        "--image-list", type=Path, help="fixed image keys for matched feature comparisons"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-reference", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.refine_source is not None:
        args.feature_cache = json.loads((args.refine_source / "run_settings.json").read_text())[
            "feature_caches"
        ]
    elif args.feature_cache is None:
        parser.error("--feature-cache is required without --refine-source")
    if args.spline_residual is not None and args.spline_residual <= 0:
        parser.error("--spline-residual must be positive")
    if args.spline_df is not None and (args.spline_df < 4 or args.spline_residual is not None):
        parser.error("--spline-df must be >=4 and cannot accompany --spline-residual")
    if args.spline_knot_placement == "uniform" and args.spline_df is None:
        parser.error("--spline-knot-placement uniform requires --spline-df")
    if args.trajectory_type == "mst_polyline" and (
        args.spline_residual is not None or args.spline_df is not None or args.aggregate_ties
    ):
        parser.error("The center polyline does not use spline settings")
    from output_layout import prepare_output

    prepare_output(args, training=False)
    if (
        min(args.pca_components, args.epochs, args.approx_points) < 1
        or min(args.n_clusters, args.n_reference) < 2
    ):
        parser.error(
            "Positive components/epochs/approx-points; at least 2 clusters/reference positions"
        )
    with threadpool_limits(limits=1):
        compute_slingshot(
            args.dataset,
            args.feature_cache,
            args.output,
            n_reference=args.n_reference,
            resume=args.resume,
            refine_source=args.refine_source,
            recluster=args.recluster,
            admission_settings=args.admission_settings,
            image_list=args.image_list,
            subcluster_target_images=args.subcluster_target_images,
            terminal_quantile=args.terminal_quantile,
            **model_parameters(args),
        )


if __name__ == "__main__":
    main()
