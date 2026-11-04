"""Check native Slingshot, raw reversals, and real-image atlas selection."""

import numpy as np
import pandas as pd


def test_tied_curve_samples_retain_full_weighted_fit_and_ignore_input_order():
    from scipy.interpolate import LSQUnivariateSpline

    from image_models.slingshot import aggregate_curve_samples

    rng = np.random.default_rng(4)
    x = np.repeat(np.linspace(0.0, 1.0, 60), 5)
    y = np.sin(5 * x) + rng.normal(0.0, 0.2, len(x))
    weights = rng.uniform(0.2, 1.0, len(x))
    unique, means, effective = aggregate_curve_samples(x, y[:, None], weights)
    knots = [0.25, 0.5, 0.75]
    complete = LSQUnivariateSpline(x, y, knots, w=weights)
    reduced = LSQUnivariateSpline(unique, means[:, 0], knots, w=effective)
    query = np.linspace(0.0, 1.0, 200)
    np.testing.assert_allclose(reduced(query), complete(query), atol=1e-12)
    # A smoothing budget applies to full residuals, including fixed group scatter.
    mean_at_x = means[np.searchsorted(unique, x), 0]
    scatter = np.sum((weights * (y - mean_at_x)) ** 2)
    full_error = np.sum((weights * (y - reduced(x))) ** 2)
    reduced_error = np.sum((effective * (means[:, 0] - reduced(unique))) ** 2)
    np.testing.assert_allclose(full_error, scatter + reduced_error, atol=1e-12)
    order = rng.permutation(len(x))
    repeated = aggregate_curve_samples(x[order], y[order, None], weights[order])
    for original, permuted in zip((unique, means, effective), repeated):
        np.testing.assert_allclose(original, permuted, atol=1e-12)


def test_explicit_spline_smoothing_improves_a_nonlinear_tail():
    from threadpoolctl import threadpool_limits

    from image_models.slingshot import fit_slingshot, initialize_slingshot, project_curve

    rng = np.random.default_rng(8)
    t = np.linspace(0.0, 1.0, 400)
    z = np.column_stack([3 * t, 0.18 * np.sin(8 * t), 0.05 * t]) + rng.normal(0.0, 0.004, (400, 3))
    frames = pd.DataFrame(
        dict(
            cycle_name="a",
            file_name=np.arange(400).astype(str),
            image_time=pd.date_range("2026-01-01", periods=400, freq="30s"),
        )
    )
    errors = []
    with threadpool_limits(1):
        for budget, degrees, aggregate in [
            (None, None, False),
            (0.0002, None, False),
            (0.0002, None, True),
            (None, 6, False),
            (None, 8, False),
        ]:
            sling, _, _ = initialize_slingshot(
                z, np.minimum((t * 8).astype(int), 7), frames, approx_points=100
            )
            sling.spline_residual = budget
            sling.spline_df = degrees
            sling.aggregate_ties = aggregate
            fit_slingshot(sling, epochs=4)
            curve = sling.curves[0]
            raw = curve.pseudotimes_interp
            scores, distance = project_curve(
                z, (raw - raw.min()) / np.ptp(raw), curve.points_interp, 100
            )
            errors.append(np.mean(distance[t > 0.5] ** 2))
            assert np.corrcoef(t, scores)[0, 1] > 0.99
    assert max(errors[1:]) < 0.5 * errors[0]


def test_uniform_spline_knots_do_not_follow_frame_density():
    from unittest.mock import patch

    from scipy.interpolate import LSQUnivariateSpline
    from threadpoolctl import threadpool_limits

    from image_models.slingshot import fit_slingshot, initialize_slingshot

    rng = np.random.default_rng(8)
    t = np.r_[np.linspace(0.0, 0.6, 100, endpoint=False), np.linspace(0.6, 1.0, 300)]
    z = np.column_stack([3 * t, 0.18 * np.sin(8 * t), 0.05 * t]) + rng.normal(0.0, 0.004, (400, 3))
    frames = pd.DataFrame(
        dict(
            cycle_name="a",
            file_name=np.arange(400).astype(str),
            image_time=pd.date_range("2026-01-01", periods=400, freq="30s"),
        )
    )
    captured = []

    def spline(x, y, knots, **kwargs):
        captured.append((x.copy(), knots.copy()))
        return LSQUnivariateSpline(x, y, knots, **kwargs)

    with threadpool_limits(1), patch("scipy.interpolate.LSQUnivariateSpline", spline):
        sling, _, _ = initialize_slingshot(z, np.minimum((t * 8).astype(int), 7), frames)
        sling.spline_df = 8
        sling.spline_knot_placement = "uniform"
        sling.aggregate_ties = True
        fit_slingshot(sling)
    x, knots = captured[0]
    np.testing.assert_allclose(np.diff(np.r_[x[0], knots, x[-1]]), (x[-1] - x[0]) / 5)
    assert not np.allclose(knots, np.quantile(x, [0.2, 0.4, 0.6, 0.8]))


def test_dense_frozen_projection_interpolates_the_nearest_segment_parameter():
    from image_models.slingshot import project_curve

    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    phase = np.array([0.0, 1 / 3, 1.0])
    queries = np.array([[0.6, -0.1], [1.1, 1.2], [-1.0, 0.0], [1.0, 3.0]])
    score, distance = project_curve(queries, phase, points, n_points=None)
    np.testing.assert_allclose(score, [0.2, 2.2 / 3, 0.0, 1.0])
    np.testing.assert_allclose(distance, [0.1, 0.1, 1.0, 1.0])
    alone, _ = project_curve(queries[1:2], phase, points, n_points=None)
    np.testing.assert_allclose(alone, score[1:2])


def test_polyline_uses_exact_center_edges_without_spline_or_corner_shortcut():
    from types import SimpleNamespace

    from image_models.slingshot import fit_slingshot, project_curve, trajectory_geometry

    sling = SimpleNamespace(
        trajectory_type="mst_polyline",
        cluster_centres=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]]),
        lineages=[SimpleNamespace(clusters=[0, 1, 2])],
    )
    fit_slingshot(sling)  # The polyline never calls the native spline fitter.
    phase, points = trajectory_geometry(sling)
    np.testing.assert_allclose(phase, [0.0, 1 / 3, 1.0])
    score, distance = project_curve(np.array([[1.0, 0.0], [0.6, -0.1]]), phase, points, None)
    np.testing.assert_allclose(score, [1 / 3, 0.2])
    np.testing.assert_allclose(distance, [0.0, 0.1])


def test_observed_refinement_extends_tail_and_keeps_ties_in_same_stage():
    from types import SimpleNamespace

    from image_models.slingshot import project_curve, refine_observed_polyline, trajectory_geometry

    x = np.tile(np.linspace(-0.2, 2.0, 120), 3)
    data = np.column_stack([x, np.zeros(len(x))])
    frames = pd.DataFrame(
        dict(
            cluster=(x > 0.5).astype(int),
            cycle_name=np.repeat(["a", "b", "c"], 120),
            experiment_id=np.repeat(["a", "b", "c"], 120),
        )
    )

    def model():
        return SimpleNamespace(
            trajectory_type="mst_polyline",
            data=data,
            cluster_centres=np.array([[0.0, 0.0], [1.0, 0.0]]),
            lineages=[SimpleNamespace(clusters=[0, 1])],
        )

    sling = model()
    fine, stages = refine_observed_polyline(sling, frames, 50, 0.95)
    phase, points = trajectory_geometry(sling)
    assert points[-1, 0] > 1.8 and stages.iloc[-1].cycle_count == 3
    assert fine.fine_cluster.nunique() > 2
    assert fine.groupby(x).fine_cluster.nunique().eq(1).all()
    scores, _ = project_curve(data, phase, points, None)
    assert np.mean(scores == 1.0) < 0.05
    # The fitted bounds remain fixed when projecting a single future sample.
    np.testing.assert_allclose(project_curve(data[20:21], phase, points, None)[0], scores[20:21])


def test_raw_monotonicity_keeps_reversals():
    from image_models.slingshot import monotonicity_summary

    frames = pd.DataFrame(
        {
            "cycle_name": ["a"] * 4,
            "file_name": list("abcd"),
            "image_time": pd.date_range("2026-01-01", periods=4, freq="30s"),
            "visual_frost_score": [0.0, 0.7, 0.3, 1.0],
        }
    )
    summary = monotonicity_summary(frames).iloc[0]
    assert summary.n_decreases == 1 and summary.decrease_fraction == 1 / 3
    assert np.isclose(summary.max_step_decrease, 0.4)
    assert np.isclose(summary.max_drawdown, 0.4)
    assert frames.visual_frost_score.tolist() == [0.0, 0.7, 0.3, 1.0]


def test_atlas_uses_nearest_real_images_to_uniform_curve_positions():
    from image_models.slingshot import reference_atlas

    z = np.array([[0.0, 0.1], [0.5, 0.0], [1.0, -0.1], [0.0, 0.0]])
    frames = pd.DataFrame(
        {
            "cycle_name": ["a"] * 4,
            "file_name": list("abcd"),
            "visual_frost_score": [0.0, 0.5, 1.0, 0.0],
            "local_image_available": [True, True, True, False],
        }
    )
    curve = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    atlas = reference_atlas(frames, z, np.array([0.0, 0.5, 1.0]), curve, 3)
    assert atlas.file_name.tolist() == list("dbc")
    assert not atlas.local_image_available.iloc[0]  # A real cached image needs no download.
    np.testing.assert_allclose(atlas.reference_phase, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(atlas.atlas_distance, [0.0, 0.0, 0.1])


def test_native_slingshot_fits_in_numpy2_and_preserves_a_single_lineage():
    from image_models.slingshot import fit_slingshot, initialize_slingshot

    rng = np.random.default_rng(4)
    t = np.linspace(0.0, 1.0, 120)
    z = np.column_stack([3 * t, 0.2 * t * t, 0.1 * t]) + rng.normal(0, 0.025, (120, 3))
    labels = np.minimum((t * 4).astype(int), 3)
    frames = pd.DataFrame(
        {
            "cycle_name": ["a"] * 120,
            "file_name": [str(i) for i in range(120)],
            "image_time": pd.date_range("2026-01-01", periods=120, freq="30s"),
        }
    )
    sling, clusters, lineages = initialize_slingshot(z, labels, frames, approx_points=30)
    assert sling.start_node == 0 and lineages.lineage.nunique() == 1
    assert lineages.cluster.tolist() == [0, 1, 2, 3]
    fit_slingshot(sling, epochs=2)
    raw = sling.curves[0].pseudotimes_interp
    assert np.isfinite(raw).all() and np.corrcoef(t, raw)[0, 1] > 0.98


def test_branched_run_stops_before_scoring_or_atlas(tmp_path, monkeypatch):
    from image_models import slingshot
    from image_models.dinov2_features import DINO_COLUMNS

    rng = np.random.default_rng(7)
    centers = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
    z = rng.normal(0.0, 0.01, (240, 384))
    z[:, :2] += np.repeat(centers, 60, axis=0)
    z[:, -1] += 10.0
    frames = pd.DataFrame(z, columns=DINO_COLUMNS)
    frames = frames.assign(
        cycle_name="a",
        file_name=[str(i) for i in range(240)],
        image_time=pd.date_range("2026-01-01", periods=240, freq="30s"),
        camera_role="front",
        embedding_available=True,
        local_image_available=True,
    )
    monkeypatch.setattr(slingshot, "load_front_features", lambda *args, **kwargs: frames)
    result = slingshot.compute_slingshot(
        tmp_path, [], tmp_path / "run", pca_components=3, n_clusters=4, epochs=1
    )
    assert result == (None, None)
    summary = pd.read_csv(tmp_path / "run/fit_summary.csv").iloc[0]
    assert summary.status == "branched_stopped" and summary.lineage_count == 2
    assert not (tmp_path / "run/progression.csv").exists()
    assert not (tmp_path / "run/reference.csv").exists()


def test_harmony_preserves_frame_order_and_shape():
    from image_models.slingshot import harmony_coordinates

    rng = np.random.default_rng(3)
    t = np.tile(np.linspace(0.0, 1.0, 100), 2)
    z = np.column_stack([t + 1.0, 0.1 * np.repeat([0.0, 1.0], 100), np.sin(t)])
    z += rng.normal(0.0, 0.01, z.shape)
    frames = pd.DataFrame({"experiment_id": np.repeat(["a", "b"], 100)})
    before = z.copy()
    corrected, model = harmony_coordinates(z, frames, seed=0)
    assert corrected.shape == z.shape and np.isfinite(corrected).all()
    np.testing.assert_array_equal(z, before)
    assert model.Z_corr.shape == (3, 200)
    assert np.corrcoef(t[:100], corrected[:100, 0])[0, 1] > 0.9


def test_curve_projection_approximation_preserves_nearest_geometry():
    from pcurvepy2 import PrincipalCurve

    from image_models.slingshot import bounded_curve_projection

    rng = np.random.default_rng(2)
    angle = np.linspace(0.0, 1.5, 401)
    polygon = np.column_stack([np.cos(angle), np.sin(angle)])
    chosen = np.linspace(0.0, 1.5, 60)
    data = np.column_stack([np.cos(chosen), np.sin(chosen)]) + rng.normal(0.0, 0.01, (60, 2))
    exact = PrincipalCurve()
    exact_distance, _ = exact.project_to_curve(data, polygon)
    approximate = PrincipalCurve()
    with bounded_curve_projection(150):
        distance, _ = approximate.project_to_curve(data, polygon)
    np.testing.assert_allclose(approximate.points_interp, exact.points_interp, atol=0.002)
    np.testing.assert_allclose(distance, exact_distance, atol=0.0001)


def test_frozen_projection_is_independent_of_query_batch_and_query_range():
    from image_models.slingshot import project_curve

    points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    phase = np.array([0.0, 0.5, 1.0])
    queries = np.array([[0.2, 0.1], [0.8, -0.1]])
    score, distance = project_curve(queries, phase, points, n_points=30)
    one, _ = project_curve(queries[:1], phase, points, n_points=30)
    np.testing.assert_allclose(score, [0.2, 0.8], atol=1e-5)
    np.testing.assert_allclose(one, score[:1], atol=1e-5)
    np.testing.assert_allclose(distance, [0.1, 0.1], atol=1e-5)
