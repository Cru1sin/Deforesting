from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("scale,layout", [("native", "full"), ("groups16", "square")])
def test_camera_frost_photos_use_screened_native_grades_and_preserve_pixels(
    tmp_path, monkeypatch, scale, layout
):
    import json

    from PIL import Image

    from image_models import camera_data
    from plots import camera as publication
    from plots import camera as rgb_frost_state

    dataset = tmp_path / "dataset"
    (dataset / "training_data").mkdir(parents=True)
    full = pd.DataFrame(dict(cycle_name=["good", "excluded"], marker=[1, 2]))
    full.to_parquet(dataset / "training_data/candidate_decisions.parquet")
    times = pd.date_range("2026-01-01", periods=16 if scale == "groups16" else 6, freq="30s")
    frames = pd.DataFrame(
        dict(
            cycle_name="good",
            image_time=times,
            visual_frost_score=np.linspace(0, 1, 16)
            if scale == "groups16"
            else [0.03, 0.05, 0.10, 0.15, 0.20, 0.25],
            embedding_available=True,
            projection_distance=0.1,
        )
    )
    if scale == "groups16":
        frames = frames.assign(
            reference_index=range(1, 17), reference_method="mst_guided_visual_segments"
        )
    frames = pd.concat([frames, frames.assign(cycle_name="excluded", visual_frost_score=0.0)])
    pixels = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    pixels[:, 0] = [255, 0, 0]
    pixels[:, -1] = [0, 0, 255]
    metadata = []
    for i, camera in enumerate(camera_data.CAMERAS):
        path = tmp_path / "images" / "good" / camera / "photo.png"
        path.parent.mkdir(parents=True)
        Image.fromarray(pixels).save(path)
        metadata.append(
            dict(
                cycle_name="good",
                camera_role=camera,
                file_name="photo.png",
                image_time=times[3 * i if scale == "groups16" else i] - pd.Timedelta(seconds=1),
            )
        )
    catalog = pd.DataFrame(dict(cycle_name=["good", "excluded"]))
    loader = SimpleNamespace(
        list_cycles=lambda: catalog, load_image_metadata=lambda: pd.DataFrame(metadata)
    )
    monkeypatch.setattr(rgb_frost_state, "DatasetLoader", lambda _: loader)

    def screen(decisions, cycles, **kwargs):
        pd.testing.assert_frame_equal(decisions, full)
        pd.testing.assert_frame_equal(cycles, catalog)
        return catalog.assign(selected=[True, False])

    monkeypatch.setattr(rgb_frost_state, "screen_cv_cycles", screen)
    monkeypatch.setattr("plots.defrost_decision.load_native_frost_frames", lambda _: frames)
    monkeypatch.setattr("dataset_tools.cycle_metadata.image_root", lambda _: tmp_path / "images")
    progression = tmp_path / "discrete_progression.csv"
    frames.to_csv(progression, index=False)
    (tmp_path / "run_settings.json").write_text(
        json.dumps(dict(coordinate_correction="harmony", method="mst_guided_visual_segments"))
    )
    args = SimpleNamespace(
        dataset=dataset,
        output=tmp_path,
        frost_progression=progression,
        image_cycle=None,
        frost_grades=None,
        frost_scale=scale,
    )
    source = rgb_frost_state.select_camera_frost_images(args)
    assert source.cycle_name.eq("good").all() and len(source) == 6
    if scale == "groups16":
        np.testing.assert_array_equal(source.reference_grade, [1, 4, 7, 10, 13, 16])
    else:
        assert source.reference_grade.iloc[0] == 3.0 and source.target_grade.iloc[0] == 0.0
    assert source.reference_grade.diff().iloc[1:].gt(0).all()
    assert pd.to_datetime(source.image_time).diff().iloc[1:].gt(pd.Timedelta(0)).all()
    assert source.offset_seconds.eq(-1).all()
    captured = []
    monkeypatch.setattr(
        publication, "save_publication", lambda fig, *args, **kwargs: captured.append(fig)
    )
    rendered = publication.render_camera_frost_images(source, tmp_path, layout=layout)
    assert len(captured[0].axes) == 6
    if layout == "square":
        assert [text.get_text() for text in captured[0].texts] == list(camera_data.CAMERAS)
    expected = np.rot90(pixels)
    for i, (camera, axis) in enumerate(zip(camera_data.CAMERAS, captured[0].axes)):
        assert axis.get_aspect() == 1
        shown = np.asarray(axis.images[0].get_array())
        if layout == "square":
            assert shown.shape[:2] == (2, 2)
            assert (
                shown[0, :, 2].mean() > shown[0, :, 0].mean()
            )  # Full-frame blue top edge survives.
            assert (
                shown[-1, :, 0].mean() > shown[-1, :, 2].mean()
            )  # Full-frame red bottom edge survives.
            assert rendered.crop_top_px.iloc[i] == 0 and rendered.crop_height_px.iloc[i] == 4
        else:
            np.testing.assert_array_equal(shown, expected)
        with Image.open(tmp_path / f"camera_frost_{camera}.png") as photo:
            np.testing.assert_array_equal(np.asarray(photo.convert("RGB")), shown)


def test_vertex_grades_and_discrete_near_sets_preserve_missing_support():
    from image_models.overlap import (
        normalized_cop_by_grade,
        observed_grade_intervals,
        vertex_grades,
    )

    np.testing.assert_equal(
        vertex_grades([0, 0.25, 0.75, 1, np.nan], [0, 0.5, 1]), [1, 2, 3, 3, np.nan]
    )
    curves = pd.DataFrame(
        dict(
            cycle_name=["a"] * 4 + ["b"],
            Ts=50,
            frost_grade=[1, 3, 3, 4, 3],
            raw_effective_cop=[1.0, 1.0, 0.9, 0.8, 2.0],
        )
    )
    spans = observed_grade_intervals(curves)
    a = spans.loc[spans.cycle_name.eq("a")]
    assert list(zip(a.left, a.right)) == [(1, 1), (3, 3)]
    values = normalized_cop_by_grade(curves, np.arange(1, 5), mapping="observed-grade")
    assert 2 not in values.grade.values
    mean = values.loc[values.grade.eq(3)].normalized_cop.mean()
    assert np.isclose(mean, 0.975)  # equal-cycle means: (mean(1,.9) + 1) / 2


def test_normalized_cop_state_axis_preserves_time_order_support_and_gaps():
    import matplotlib.pyplot as plt

    from plots.cop import plot_normalized

    origin = pd.Timestamp("2026-01-01")
    times = origin + pd.to_timedelta([0, 10, 20, 30, 80, 90], unit="s")
    curve = pd.DataFrame(
        dict(
            candidate_defrost_time=times,
            image_time=times,
            frost_grade=[10, 20, 15, 25, 40, 45],
            cycle_cop=[1, 2, 1.5, 3, 1.8, 1.9],
            cycle_cop_eligible=[True, True, True, False, True, True],
        )
    )
    figure, axis = plt.subplots()
    plot_normalized(axis, curve, origin, metrics=("cycle_cop",), x_column="frost_grade")
    first, second = axis.lines[:2]
    np.testing.assert_equal(first.get_xdata(), [10, 20, 15, 25])
    np.testing.assert_allclose(first.get_ydata(), [50, 100, 75, np.nan])
    np.testing.assert_equal(second.get_xdata(), [40, 45])
    np.testing.assert_allclose(second.get_ydata(), [90, 95])
    plt.close(figure)


def test_grade_intervals_preserve_saved_near_mask_and_missing_grade_gaps():
    from image_models.overlap import reference_basin_intervals

    curve = pd.DataFrame(
        dict(
            cycle_name="cycle",
            Ts=50,
            candidate_defrost_time=pd.date_range("2026-01-01", periods=5, freq="30s"),
            cycle_cop=[2.0, 1.98, 2.0, 1.98, 2.1],
            near_optimal=[True, True, True, True, False],
            frost_grade=[10.0, 11.0, np.nan, 20.0, 30.0],
        )
    )
    intervals = reference_basin_intervals(
        curve,
        objectives=[("new", "cycle_cop")],
        coordinates=["frost_grade"],
        near_column="near_optimal",
    )
    assert intervals[["left", "right"]].to_numpy().tolist() == [[10.0, 11.0], [20.0, 20.0]]
    assert intervals.actions.tolist() == [2, 1]


def test_frost_image_selection_stays_in_the_component_covering_the_target():
    from image_models.overlap import reference_basin_intervals
    from plots.frost_comparison import select_frost_overlap_images

    times = pd.date_range("2026-01-01", periods=5, freq="30s")
    curves = pd.DataFrame(
        dict(
            cycle_name="cycle_000001",
            Ts=50,
            candidate_defrost_time=times,
            raw_effective_cop=[2.0, 2.0, 1.0, 2.0, 2.0],
            frost_grade=[10.0, 30.0, 20.0, 19.0, 19.0],
            image_time=times,
            file_name=[f"{k}.jpg" for k in range(5)],
        )
    )
    spans = reference_basin_intervals(
        curves, objectives=[("cop", "raw_effective_cop")], coordinates=["frost_grade"]
    )
    loader = SimpleNamespace(
        load_cycle_images=lambda n: pd.DataFrame(columns=["camera_role", "file_name", "path"])
    )
    rows = select_frost_overlap_images(curves, spans, 20, loader)
    assert len(rows) == 1 and rows.observed_grade.iloc[0] == 10
    assert rows.grade_distance.iloc[0] == 10 and not rows.exact_grade.iloc[0]
    assert rows.cop_fraction.iloc[0] == 1 and not rows.available.iloc[0]
    extra = curves.assign(cycle_name="cycle_000002", frost_grade=np.arange(50.0, 55.0))
    all_curves = pd.concat([curves, extra], ignore_index=True)
    all_spans = reference_basin_intervals(
        all_curves, objectives=[("cop", "raw_effective_cop")], coordinates=["frost_grade"]
    )
    all_rows = select_frost_overlap_images(all_curves, all_spans, 20, loader, all_cycles=True)
    assert len(all_rows) == 2 and all_rows.interval_hit.tolist() == [True, False]
    assert all_rows.observed_grade.iloc[1] == 50
    missing = curves.assign(cycle_name="cycle_000003", frost_grade=np.nan)
    blank = select_frost_overlap_images(
        pd.concat([all_curves, missing]), all_spans, 20, loader, all_cycles=True
    ).iloc[-1]
    assert not blank.near_grade_available and not blank.available
    assert pd.isna(blank.observed_grade) and not blank.image_path and pd.isna(blank.file_name)


def test_explicit_rgb_root_keeps_missing_nearest_frame_without_loader_fallback(tmp_path):
    from image_models.overlap import reference_basin_intervals
    from plots.frost_comparison import select_frost_overlap_images

    times = pd.date_range("2026-01-01", periods=2, freq="30s")
    curves = pd.DataFrame(
        dict(
            cycle_name="cycle_000001",
            Ts=50,
            candidate_defrost_time=times,
            raw_effective_cop=2.0,
            frost_grade=[10.0, 30.0],
            image_time=times,
            file_name=["missing.jpg", "present.jpg"],
        )
    )
    spans = reference_basin_intervals(
        curves, objectives=[("cop", "raw_effective_cop")], coordinates=["frost_grade"]
    )
    folder = tmp_path / "cycle_000001" / "front"
    folder.mkdir(parents=True)
    (folder / "present.jpg").touch()
    loader = SimpleNamespace(load_cycle_images=lambda _: 1 / 0)
    row = select_frost_overlap_images(curves, spans, 11, loader, images_dir=tmp_path).iloc[0]
    assert row.file_name == "missing.jpg" and row.observed_grade == 10
    assert not row.available and row.interval_hit
    assert row.image_path == str(folder / "missing.jpg")
    replacement = select_frost_overlap_images(
        curves, spans, 11, loader, images_dir=tmp_path, available_only=True
    )
    assert replacement.empty


def test_explicit_summary_rgb_root_records_the_missing_external_path(tmp_path):
    from plots.frost_comparison import select_frost_summary_images

    curves = pd.DataFrame(
        dict(
            cycle_name="cycle_000001",
            Ts=50,
            frost_grade=[10, 11, 12],
            raw_effective_cop=[1.99, 2.0, 1.99],
            candidate_defrost_time=pd.date_range("2026-01-01", periods=3, freq="30s"),
            file_name=["a.jpg", "b.jpg", "c.jpg"],
        )
    )
    curves["image_time"] = curves.candidate_defrost_time
    values = (
        curves[["cycle_name", "frost_grade"]]
        .rename(columns={"frost_grade": "grade"})
        .assign(normalized_cop=[0.995, 1.0, 0.995])
    )
    coverage = pd.DataFrame(
        dict(grade=[10, 11, 12], mean_observed_normalized_cop=[0.995, 1.0, 0.995])
    )
    root = tmp_path / "external"
    rows = select_frost_summary_images(
        curves,
        values,
        coverage,
        "mean",
        SimpleNamespace(dataset_root=tmp_path),
        tmp_path,
        images_dir=root,
    )
    assert len(rows) == 3 and not rows.available.any()
    assert rows.image_path.str.startswith(str(root / "cycle_000001" / "front") + "/").all()
    other = curves.assign(cycle_name="cycle_000002")
    folder = root / "cycle_000002" / "front"
    folder.mkdir(parents=True)
    for name in other.file_name:
        (folder / name).touch()
    replacements = select_frost_summary_images(
        pd.concat([curves, other]),
        pd.concat([values, values.assign(cycle_name="cycle_000002")]),
        coverage,
        "mean",
        SimpleNamespace(dataset_root=tmp_path),
        tmp_path,
        images_dir=root,
        available_only=True,
    )
    assert len(replacements) == 3 and replacements.available.all()
    assert replacements.cycle_name.eq("cycle_000002").all()


def test_frost_image_plate_rotates_and_keeps_dense_header_clear(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from plots.frost_cop import plot_frost_overlap_images
    from plots.publication import _RGB_PANEL_FONT

    path = tmp_path / "image.jpg"
    path.touch()
    pixels = np.arange(9 * 16 * 3).reshape(9, 16, 3) / 500
    monkeypatch.setattr(plt, "imread", lambda p: pixels)
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: figures.append(self))
    rows = pd.DataFrame(
        dict(
            cycle_name=[f"cycle_{k:06d}" for k in range(47)],
            setpoint_c=55,
            target_grade=82,
            observed_grade=82,
            image_path=str(path),
            available=True,
            interval_hit=True,
            candidate_defrost_time=pd.Timestamp("2026-01-01"),
            image_time=pd.Timestamp("2026-01-01"),
        )
    )
    summary = pd.Series(dict(maximum_hits=47, total_cycles=48, maximum_overlap=47 / 48))
    with plt.rc_context({"font.family": _RGB_PANEL_FONT}):
        plot_frost_overlap_images(rows, summary, tmp_path, combined=True)
    fig = figures[0]
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), np.rot90(pixels))
    assert fig.axes[0].get_aspect() == 1
    assert sum(bool(ax.images) for ax in fig.axes) == 47
    assert (
        fig.axes[0]._left_title.get_window_extent(renderer).y1
        < fig.texts[3].get_window_extent(renderer).y0
    )
    rows["near_grade_available"] = True
    rows.loc[0, ["near_grade_available", "interval_hit"]] = False
    rows.loc[0, "observed_grade"] = np.nan
    rows.loc[1, "interval_hit"] = False
    with plt.rc_context({"font.family": _RGB_PANEL_FONT}):
        plot_frost_overlap_images(rows, summary, tmp_path, combined=True, common_grade=True)
    grouped = figures[-1]
    FigureCanvasAgg(grouped)
    grouped.canvas.draw()
    assert sum(bool(ax.images) for ax in grouped.axes) == 46
    blank = next(ax for ax in grouped.axes if ax.get_title(loc="left").startswith("C0 "))
    assert not blank.images and blank.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert any("霜层等级不可用" in t.get_text() for ax in grouped.axes for t in ax.texts)
    renderer = grouped.canvas.get_renderer()
    for ax in grouped.axes:
        if not ax.get_title(loc="left"):
            continue
        label = ax._left_title.get_window_extent(renderer)
        assert not any(
            label.overlaps(other.get_window_extent(renderer))
            for other in grouped.axes
            if other is not ax and other.images
        )


def test_single_axis_frost_cop_mean_median_threshold_ranges_and_header(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from plots.frost_cop import plot_mean_frost_cop

    coverage = pd.DataFrame(
        dict(
            grade=np.arange(10, 15),
            mean_observed_normalized_cop=[0.98, 0.995, 0.996, np.nan, 0.992],
            median_normalized_cop=[0.99, 0.997, 0.998, np.nan, 0.999],
            n_evaluated=[3, 3, 3, 0, 2],
            total_cycles=3,
            n_interpolated_cycles=[0, 1, 1, 0, 0],
        )
    )
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: figures.append(self))
    for statistic in ["mean", "median"]:
        record = plot_mean_frost_cop(
            coverage,
            tmp_path,
            cohort_label="Test",
            name="test",
            source_label="Native Harmony + Slingshot",
            statistic=statistic,
        )
        figure = figures[-1]
        assert len(figure.axes) == 2
        np.testing.assert_allclose(
            figure.axes[0].get_position().bounds, figure.axes[1].get_position().bounds
        )
        column = "mean_observed_normalized_cop" if statistic == "mean" else "median_normalized_cop"
        np.testing.assert_allclose(
            figure.axes[0].lines[0].get_ydata(), coverage[column], equal_nan=True
        )
        assert figure.axes[0].get_ylim() == (0.9, 1)
        assert figure.axes[0].lines[1].get_ydata() == [0.99, 0.99]
        assert any(
            t.get_text().startswith("Maximum " + statistic) and t.get_position()[1] == 0.90
            for t in figure.texts
        )
        assert record["threshold_intervals"] == (
            "11–12, 14" if statistic == "mean" else "10–12, 14"
        )
        assert record["maximum_grades"] == ("12" if statistic == "mean" else "14")
    plt.close("all")


def test_observed_grade_cop_averages_all_values_and_weights_cycles_equally():
    from image_models.overlap import normalized_cop_by_grade

    curves = pd.DataFrame(
        dict(
            cycle_name=["one"] * 12 + ["two"] * 2,
            frost_grade=[10, 20, 20, 30, 30, 30, 40, 40, 40, 40, 50, np.nan, 20, np.nan],
            raw_effective_cop=[8, 7, 9, 9, 6, 8, 9, 6, 8, 7, np.nan, 10, 1, 2],
        )
    )
    table = normalized_cop_by_grade(curves, np.arange(100), mapping="observed-grade")
    one = table.loc[table.cycle_name.eq("one")].set_index("grade")
    assert one.n_actions.to_dict() == {10.0: 1, 20.0: 2, 30.0: 3, 40.0: 4}
    np.testing.assert_allclose(one.normalized_cop, [0.8, 0.8, 2.3 / 3, 0.75])
    at20 = table.loc[table.grade.eq(20)]
    assert len(at20) == 2 and np.isclose(at20.normalized_cop.mean(), 0.65)
    assert not table.grade.eq(50).any()


def test_normalized_cop_grade_uses_own_maximum_first_crossing_and_keeps_gaps():
    from image_models.overlap import normalized_cop_by_grade

    times = pd.date_range("2026-01-01", periods=4, freq="30s")
    curves = pd.DataFrame(
        dict(
            cycle_name=["one"] * 4 + ["two"] * 4,
            candidate_defrost_time=list(times) * 2,
            frost_grade=[10, 20, 10, 20, 10, np.nan, 20, 30],
            raw_effective_cop=[2, 1, 2, 2, 4, 2, 3, 2],
        )
    )
    table = normalized_cop_by_grade(curves, [10, 20, 30]).set_index(["cycle_name", "grade"])
    assert table.loc[("one", 20), "normalized_cop"] == 0.5
    assert table.loc[("two", 30), "normalized_cop"] == 0.5
    assert table.loc[("two", 20), "status"] == "reference_gap"
    assert pd.isna(table.loc[("two", 20), "normalized_cop"])
    assert pd.isna(table.loc[("one", 10), "normalized_cop"])
    assert table.loc[("one", 30), "status"] == "no_crossing"
    observed = normalized_cop_by_grade(curves, [10, 20, 30], mapping="observed-grade").set_index(
        ["cycle_name", "grade"]
    )
    assert observed.loc[("one", 20), "normalized_cop"] == 0.75
    assert observed.loc[("one", 20), "n_actions"] == 2
    assert observed.loc[("two", 20), "normalized_cop"] == 0.75
    assert ("one", 30) not in observed.index
    curves["image_time"] = curves.candidate_defrost_time
    interpolated = normalized_cop_by_grade(
        curves, np.arange(32), mapping="interpolated-grade"
    ).set_index(["cycle_name", "grade"])
    assert np.isclose(interpolated.loc[("one", 15), "normalized_cop"], (0.75 + 0.75 + 1) / 3)
    assert interpolated.loc[("one", 20), "normalized_cop"] == 0.75
    assert interpolated.loc[("one", 20), "n_interpolated_segments"] == 0
    assert interpolated.loc[("two", 25), "normalized_cop"] == 0.625
    assert ("two", 15) not in interpolated.index
    assert ("one", 5) not in interpolated.index and ("two", 31) not in interpolated.index
    gaps = curves.copy()
    gaps.loc[7, "candidate_defrost_time"] += pd.Timedelta(seconds=120)
    assert (
        not normalized_cop_by_grade(gaps, [25], mapping="interpolated-grade")
        .query('cycle_name=="two" and grade==25')
        .shape[0]
    )
    gaps = curves.copy()
    gaps.loc[7, "image_time"] += pd.Timedelta(seconds=60)
    assert (
        not normalized_cop_by_grade(gaps, [25], mapping="interpolated-grade")
        .query('cycle_name=="two" and grade==25')
        .shape[0]
    )


def test_time_to_frost_mapping_renders_exact_grade_means_and_counts(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from plots.frost_cop import plot_frost_cop_mapping
    from plots.publication import _RGB_PANEL_FONT

    times = pd.date_range("2026-01-01", periods=3, freq="30s")
    curves = pd.DataFrame(
        dict(
            cycle_name=["cycle_000001"] * 3 + ["cycle_000002"] * 3,
            Ts=[50] * 3 + [60] * 3,
            candidate_defrost_time=list(times) * 2,
            raw_effective_cop=[1, 2, 1, 2, 4, 3],
            frost_grade=[10, 20, 30, 20, 30, 40],
        )
    )
    values = pd.DataFrame(
        dict(
            cycle_name=curves.cycle_name,
            grade=curves.frost_grade,
            normalized_cop=[0.5, 1, 0.5, 0.5, 1, 0.75],
        )
    )
    table = (
        values.groupby("grade")
        .normalized_cop.agg(mean_observed_normalized_cop="mean", n_evaluated="count")
        .reset_index()
    )
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: figures.append(self))
    with plt.rc_context({"font.family": _RGB_PANEL_FONT}):
        plot_frost_cop_mapping(
            curves,
            values,
            table,
            tmp_path,
            cohort_label="All cycles",
            name="mapping",
            source_label="Native Harmony + Slingshot",
        )
    time_cop, time_grade, individual, mean, count = figures[0].axes
    np.testing.assert_allclose(time_cop.lines[0].get_ydata(), [50, 100, 50])
    np.testing.assert_array_equal(time_grade.lines[0].get_ydata(), [10, 20, 30])
    np.testing.assert_allclose(mean.lines[0].get_ydata(), 100 * table.mean_observed_normalized_cop)
    np.testing.assert_array_equal(count.lines[0].get_ydata(), table.n_evaluated)
    assert len(individual.lines) == 2


def test_grade_overlap_keeps_gaps_counts_cycles_once_and_reuses_two_panels(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    from image_models.overlap import interval_cycle_counts, reference_basin_intervals
    from plots.frost_cop import plot_overlap
    from plots.publication import _RGB_PANEL_FONT

    times = pd.date_range("2026-01-01", periods=5, freq="30s")
    curves = pd.DataFrame(
        dict(
            cycle_name="cycle_000001",
            Ts=50,
            candidate_defrost_time=times,
            raw_effective_cop=[2.0, 1.0, 2.0, 2.0, 2.0],
            frost_grade=[10.0, 20.0, 30.0, np.nan, 10.0],
        )
    )
    intervals = reference_basin_intervals(
        curves, objectives=[("cop", "raw_effective_cop")], coordinates=["frost_grade"]
    )
    assert intervals[["left", "right"]].to_numpy().tolist() == [
        [10.0, 10.0],
        [30.0, 30.0],
        [10.0, 10.0],
    ]
    grades = np.arange(40)
    counts = interval_cycle_counts(intervals, grades, closed=True)
    assert counts[10] == 1 and counts[30] == 1 and counts[20] == 0
    coverage = pd.DataFrame(
        dict(left=grades - 0.5, right=grades + 0.5, n_near_optimal=counts, coverage=counts / 2)
    )
    screening = pd.DataFrame(
        dict(cycle_name=["cycle_000001", "cycle_000002"], reference_available=[True, False])
    )
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: figures.append(self))
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": _RGB_PANEL_FONT}):
        plot_overlap(
            screening,
            intervals.assign(kind="feasible"),
            coverage,
            tmp_path,
            cycle_names=screening.cycle_name.tolist(),
            grade_max=39,
            source_label="Harmony + Slingshot",
        )
    top, bottom = figures[0].axes
    assert len(figures[0].axes) == 2
    assert top.get_shared_x_axes().joined(top, bottom)
    assert [tick.get_text() for tick in top.get_yticklabels()] == ["1", "2"]
    assert any("1/2" in text.get_text() for text in bottom.texts)
    assert sum("50.0%" in text.get_text() for text in bottom.texts) == 1
    assert len(figures) == 1 and any("Harmony" in text.get_text() for text in figures[0].texts)
    assert not any("证据" in text.get_text() for text in figures[0].texts)
    assert top.get_xlabel() == bottom.get_xlabel() == "霜层等级"
    mean = coverage.drop(columns=["coverage", "n_near_optimal"]).assign(
        mean_normalized_cop=np.where(
            np.isin(grades, [10, 11]), [0.8 if g == 10 else 0.9 for g in grades], np.nan
        ),
        common_support=np.isin(grades, [10, 11]),
        n_evaluated=np.where(np.isin(grades, [10, 11]), 2, 1),
    )
    with plt.rc_context({"font.family": _RGB_PANEL_FONT}):
        plot_overlap(
            screening,
            intervals.assign(kind="feasible"),
            mean,
            tmp_path,
            cycle_names=screening.cycle_name.tolist(),
            grade_max=39,
            source_label="Harmony + Slingshot",
        )
    assert figures[-1].axes[1].get_ylabel() == "平均归一化 COP (%)"
    assert any("90.00%" in t.get_text() for t in figures[-1].axes[1].texts)
    regret = mean.rename(columns={"mean_normalized_cop": "mean_cop_regret"}).assign(grade=grades)
    regret["mean_cop_regret"] = 1 - regret.mean_cop_regret
    with plt.rc_context({"font.family": _RGB_PANEL_FONT}):
        plot_overlap(
            screening,
            intervals.assign(kind="feasible"),
            regret,
            tmp_path,
            cycle_names=screening.cycle_name.tolist(),
            grade_max=39,
            source_label="Harmony + Slingshot",
        )
    bottom = figures[-1].axes[1]
    assert bottom.get_ylabel() == "平均 COP regret (%)"
    np.testing.assert_allclose(
        bottom.lines[0].get_ydata(), 100 * regret.mean_cop_regret, equal_nan=True
    )
    observed = regret.rename(columns={"mean_cop_regret": "mean_observed_normalized_cop"})
    observed["mean_observed_normalized_cop"] = 1 - observed.mean_observed_normalized_cop
    with plt.rc_context({"font.family": _RGB_PANEL_FONT}):
        plot_overlap(
            screening,
            intervals.assign(kind="feasible"),
            observed,
            tmp_path,
            cycle_names=screening.cycle_name.tolist(),
            grade_max=39,
            source_label="Harmony + Slingshot",
        )
    bottom = figures[-1].axes[1]
    assert bottom.get_ylabel() == "平均归一化 COP (%)"
    np.testing.assert_allclose(
        bottom.lines[0].get_ydata(), 100 * observed.mean_observed_normalized_cop, equal_nan=True
    )


def test_state_matching_is_causal_and_missing_features_remain_blank(tmp_path):
    from image_models.dinov2_features import DINO_COLUMNS
    from image_models.frost_progression import match_states

    points = pd.DataFrame(
        {
            "cycle_name": ["one", "two"],
            "experiment_id": ["e", "e"],
            "selected_time": pd.to_datetime(["2026-01-01 00:01:00"] * 2),
        }
    )
    metadata = pd.DataFrame(
        {
            "cycle_name": ["one", "one", "two"],
            "camera_role": ["front"] * 3,
            "file_name": ["past.jpg", "future.jpg", "absent.jpg"],
            "image_time": pd.to_datetime(
                ["2026-01-01 00:00:50", "2026-01-01 00:01:01", "2026-01-01 00:00:50"]
            ),
        }
    )
    cache = metadata.iloc[:2][["cycle_name", "camera_role", "file_name"]].copy()
    cache = pd.concat([cache, pd.DataFrame(np.ones((2, 384)), columns=DINO_COLUMNS)], axis=1)
    cache.to_parquet(tmp_path / "one.parquet")
    loader = SimpleNamespace(
        load_image_metadata=lambda n: metadata.loc[metadata.cycle_name.eq(n)], dataset_root=tmp_path
    )
    result = match_states(points, loader, [tmp_path])
    assert result.file_name.tolist() == ["past.jpg", "absent.jpg"]
    assert result.loc[0, DINO_COLUMNS].eq(1).all()
    assert result.loc[1, DINO_COLUMNS].isna().all()


def test_common_model_cycles_excludes_missing_decisions():
    from image_models.experiments import common_cycle_metrics

    metrics = pd.DataFrame(
        dict(
            cycle_name=["a", "a", "b", "b"],
            seed=[0] * 4,
            version=["old", "future", "old", "future"],
            selected_cop=[2, 2, 2, np.nan],
            rb_cop=[1] * 4,
        )
    )
    assert set(common_cycle_metrics(metrics).cycle_name) == {"a"}


def test_fixed_optimal_reference_is_not_refit_to_model_states():
    from image_models.frost_progression import state_coordinates

    reference = pd.DataFrame({"dinov2_000": [1.0, 1.0], "dinov2_001": [0.0, 0.1]})
    states = pd.DataFrame({"dinov2_000": [0.0], "dinov2_001": [1.0]})
    result = state_coordinates(states, reference)
    assert result.cosine_to_centroid.iloc[0] == 1
    assert result.cosine_to_reference.iloc[0] < 0.1


def test_optimal_cohort_is_not_limited_by_old_model_outputs(tmp_path, monkeypatch):
    import dataset_tools
    import image_models.frost_progression as module
    import image_models.inputs

    cycles = pd.DataFrame(dict(cycle_name=["frost_cycle_000200"], experiment_id=["new_exp"]))
    loader = SimpleNamespace(list_cycles=lambda **kwargs: cycles)
    monkeypatch.setattr(dataset_tools, "DatasetLoader", lambda dataset: loader)
    monkeypatch.setattr(image_models.inputs, "reliable_rgb_cohort", lambda *a, **k: cycles)
    monkeypatch.setattr(
        module,
        "match_states",
        lambda points, *a, **k: points.assign(
            dinov2_000=1.0, dinov2_001=0.0, local_image_available=True
        ),
    )
    cycles.assign(t_star="2026-09-01 10:00:00").to_csv(
        tmp_path / "cycle_comparison.csv", index=False
    )
    result = module.collect_states(
        tmp_path / "no_old_model_run",
        point="optimal",
        cohort="regular",
        decisions=tmp_path,
        feature_caches=[tmp_path],
    )
    assert result.cycle_name.tolist() == ["frost_cycle_000200"]


def test_trajectory_uses_early_reference_and_only_supported_basin_grid():
    from image_models.frost_progression import trajectory_scores

    start = pd.Timestamp("2026-01-01 00:00:00")
    cycles = pd.DataFrame(
        {
            "cycle_name": ["a", "b"],
            "stable_heating_start": [start, start],
            "t_star": [start + pd.Timedelta(seconds=180)] * 2,
            "t_RB": [start + pd.Timedelta(seconds=150)] * 2,
            "COP_eff_selected": [2.0] * 2,
            "cycle_cop_basin_1pct_start": [start + pd.Timedelta(seconds=120)] * 2,
            "cycle_cop_basin_1pct_end": [start + pd.Timedelta(seconds=180)] * 2,
        }
    )
    times = [start + pd.Timedelta(seconds=x) for x in (90, 120, 150, 180)]
    candidates = pd.DataFrame(
        {
            "cycle_name": ["a"] * 4 + ["b"] * 4,
            "candidate_defrost_time": times * 2,
            "cycle_cop": [1.9, 1.99, 1.99, 2.0] * 2,
            "cycle_cop_eligible": [True] * 8,
            "on_action_grid": [True, True, False, True] * 2,
        }
    )
    frames = pd.DataFrame(
        {
            "cycle_name": ["a", "a", "b"],
            "image_time": [start + pd.Timedelta(seconds=x) for x in (30, 150, 150)],
            "file_name": ["early.jpg", "late.jpg", "late_b.jpg"],
            "dinov2_000": [1.0, 0.0, 0.0],
            "dinov2_001": [0.0, 1.0, 1.0],
        }
    )
    result = trajectory_scores(cycles, candidates, frames)
    assert (
        result.loc[
            result.cycle_name.eq("a") & result.candidate_defrost_time.eq(times[0]),
            "visual_distance",
        ].iloc[0]
        == 0
    )
    assert (
        result.loc[
            result.cycle_name.eq("b") & result.candidate_defrost_time.eq(times[-1]),
            "visual_distance",
        ].iloc[0]
        == 1
    )
    assert result.loc[result.candidate_defrost_time.eq(times[2]), "in_basin_grid"].eq(False).all()
    assert result.loc[result.candidate_defrost_time.eq(times[1]), "in_basin_grid"].all()
    assert result.attrs["early_reference"].cycle_name.tolist() == ["a"]
    assert pd.isna(
        result.loc[
            result.cycle_name.eq("b") & result.candidate_defrost_time.eq(times[0]),
            "visual_distance",
        ].iloc[0]
    )
    stale = candidates.iloc[[0]].copy()
    stale["candidate_defrost_time"] = start + pd.Timedelta(seconds=300)
    assert trajectory_scores(cycles, stale, frames).visual_distance.isna().all()
    missing = frames.copy()
    missing.loc[missing.file_name.eq("late.jpg"), ["dinov2_000", "dinov2_001"]] = np.nan
    latest = trajectory_scores(cycles, candidates, missing)
    assert pd.isna(
        latest.loc[
            latest.cycle_name.eq("a") & latest.candidate_defrost_time.eq(times[-1]),
            "visual_distance",
        ].iloc[0]
    )


def test_trajectory_snapshots_keep_missing_boundaries_and_exclude_outside_points():
    from image_models.frost_progression import trajectory_snapshots

    times = pd.date_range("2026-01-01", periods=6, freq="30s")
    curve = pd.DataFrame(
        {
            "candidate_defrost_time": times,
            "cycle_cop_basin_1pct_start": times[1],
            "cycle_cop_basin_1pct_end": times[4],
            "in_basin_grid": [False, False, True, True, True, False],
            "visual_distance": [9.0, np.nan, 0.3, 0.1, 0.2, -1.0],
        }
    )
    assert [row.name for row in trajectory_snapshots(curve)] == [1, 4, 2, 3]
    assert trajectory_snapshots(curve.iloc[[0]]) == [None] * 4


def test_reference_trajectory_preserves_scores_beyond_and_preparation_boundary():
    from image_models.frost_progression import trajectory_scores

    times = pd.date_range("2026-01-01", periods=5, freq="30s")
    cycles = pd.DataFrame(
        dict(
            cycle_name=["a"],
            stable_heating_start=[times[0]],
            t_star=[times[1]],
            t_RB=[times[1]],
            COP_eff_selected=[2.0],
            cycle_cop_basin_1pct_start=[times[1]],
            cycle_cop_basin_1pct_end=[times[3]],
            frost_end=[times[3]],
        )
    )
    candidates = pd.DataFrame(
        dict(
            cycle_name=["a"] * 5,
            candidate_defrost_time=times,
            cycle_cop=[2.0] * 5,
            cycle_cop_eligible=[True] * 5,
            on_action_grid=[True] * 5,
        )
    )
    frames = pd.DataFrame(
        dict(
            cycle_name=["a"] * 3,
            image_time=times[:3],
            file_name=["a.jpg", "b.jpg", "c.jpg"],
            visual_frost_score=[0.2, 1.0, np.nan],
            is_beyond_reference=[False, False, True],
            reference_cycle=["reference"] * 3,
        )
    )
    result = trajectory_scores(cycles, candidates, frames)
    np.testing.assert_allclose(result.visual_frost_score[:3], [0.2, 1.0, 1.0])
    assert result.visual_frost_score[3:].isna().all()
    assert result.is_beyond_reference.iloc[2]
    assert not result.is_beyond_reference.iloc[3:].any()
    assert "visual_distance" not in result
    assert result.cycle_cop.eq(2.0).all()
    # A valid last frost frame must not be carried into preparation either.
    result = trajectory_scores(cycles, candidates, frames.iloc[:2])
    assert result.visual_frost_score.iloc[3:].isna().all()


def test_trajectory_grade_matching_keeps_unique_columns_and_uses_past_frames():
    from image_models.frost_progression import match_trajectory_frames

    rows = pd.DataFrame(
        {
            "cycle_name": ["a", "a"],
            "candidate_defrost_time": pd.to_datetime(["2026-01-01 00:01", "2026-01-01 00:04"]),
        }
    )
    frames = pd.DataFrame(
        {
            "cycle_name": ["a", "a"],
            "image_time": pd.to_datetime(["2026-01-01 00:00:30", "2026-01-01 00:01:30"]),
            "file_name": ["a.jpg", "b.jpg"],
            "reference_index": [12, 20],
        }
    )
    result = match_trajectory_frames(rows, frames, metric="reference_index")
    assert result.columns.is_unique
    assert result.reference_index.iloc[0] == 12
    assert pd.isna(result.reference_index.iloc[1])


def test_summary_photos_use_exact_observed_threshold_and_peak_grades(tmp_path, monkeypatch):
    from plots.frost_comparison import select_frost_summary_images

    monkeypatch.setattr("dataset_tools.cycle_metadata.image_root", lambda _: tmp_path)
    output = tmp_path / "analysis" / "figures"
    output.parent.mkdir()
    origin = pd.Timestamp("2026-01-01")
    rows = []
    for cycle, ts in [
        ("frost_cycle_000001", 50),
        ("frost_cycle_000002", 55),
        ("frost_cycle_000003", 60),
    ]:
        for grade, cop in [(9, 0.98), (10, 0.99), (11, 1.0), (12, 0.995), (13, 0.98)]:
            rows.append(
                dict(
                    cycle_name=cycle,
                    Ts=ts,
                    frost_grade=grade,
                    raw_effective_cop=cop,
                    file_name=f"{grade}.jpg",
                    image_time=origin,
                    candidate_defrost_time=origin,
                )
            )
    curves = pd.DataFrame(rows)
    values = curves.rename(columns={"frost_grade": "grade", "raw_effective_cop": "normalized_cop"})
    coverage = pd.DataFrame(
        dict(grade=[9, 10, 11, 12, 13], mean_observed_normalized_cop=[0.98, 0.99, 1.0, 0.995, 0.98])
    )
    pictures = select_frost_summary_images(
        curves, values, coverage, "mean", SimpleNamespace(dataset_root=tmp_path), output
    )
    assert len(pictures) == 9
    assert pictures.target_grade.eq(pictures.observed_grade).all()
    assert pictures.groupby("position").cycle_name.nunique().eq(3).all()
    assert pictures.groupby("position").target_grade.first().to_dict() == {
        "left": 10,
        "peak": 11,
        "right": 12,
    }
    assert pictures.groupby("position").setpoint_c.nunique().eq(3).all()
