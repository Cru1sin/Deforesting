import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_operation_panels_preserve_seed_ranges_and_black_axes(tmp_path, monkeypatch):
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure
    from plots.external_policy import render_stopping_results

    rows = []
    for cycle in range(101):
        for seed, delta in [(42, -0.02), (1010, 0.04), (2026, 0.10)]:
            rows.append(dict(
                cycle_name=f"cycle_{cycle:03d}", seed=seed,
                rb_cop=2.0, selected_cop=2.0 + delta,
                delta_cop=delta, gain_vs_rb_pct=50 * delta,
                rb_heating_duration_minutes=40.0,
                heating_duration_minutes=40.0 + 100 * delta,
                defrost_electricity_change_kwh=delta / 10,
                defrost_electricity_change_pct=100 * delta,
            ))
    pd.DataFrame(rows).to_csv(tmp_path / "cycle_results.csv", index=False)
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *a, **kw: figures.append(fig))
    render_stopping_results(tmp_path, decision_view="operations")
    figure = figures[0]
    assert len(figure.axes) == 7  # Four panels, three with a second y-axis.
    assert [ax.get_title(loc="left")[:3] for ax in figure.axes[:4]] == [
        "(a)", "(b)", "(c)", "(d)"
    ]
    for axis in figure.axes[:4]:
        assert 1.05 <= axis.bbox.width / axis.bbox.height <= 1.6
    for axis in figure.axes:
        labels = [axis.xaxis.label, axis.yaxis.label,
                  *axis.get_xticklabels(), *axis.get_yticklabels()]
        assert all(to_rgba(label.get_color()) == to_rgba("black") for label in labels)
        assert all(spine.get_edgecolor() == to_rgba("black")
                   for spine in axis.spines.values())
    source = pd.read_csv(tmp_path / "Fig2e_heating_electricity_relationship.csv")
    assert len(source) == 101
    for metric, expected in [
        ("stable_heating_increase_minutes", [4.0, -2.0, 10.0]),
        ("stable_heating_increase_pct", [10.0, -5.0, 25.0]),
        ("defrost_electricity_change_wh", [4.0, -2.0, 10.0]),
        ("defrost_electricity_change_pct", [4.0, -2.0, 10.0]),
    ]:
        for statistic, value in zip(["mean", "minimum", "maximum"], expected):
            np.testing.assert_allclose(source[f"{statistic}_{metric}"], value)


def test_camera_diagnostics_preserve_cop_ratios_and_disconnected_near_sets(tmp_path, monkeypatch):
    import pickle
    from types import SimpleNamespace

    from image_models import camera_comparison
    from plots import camera as publication

    times = pd.date_range("2026-01-01", periods=5, freq="30s")
    metrics = pd.DataFrame(
        [
            dict(
                evaluation_scope=scope,
                camera_role="front",
                seed=42,
                cycle_name=name,
                experiment_id=name,
                selected_time=times[0],
                unsupported_trigger=False,
                near_components=2,
            )
            for scope, name in [("first_common", "a"), ("second_common", "b")]
        ]
    )
    metrics.to_csv(tmp_path / "comparison_cycles.csv", index=False)
    paths = []
    for name in ["a", "b"]:
        path = tmp_path / f"{name}.pkl"
        paths.append(path)
        predictions = pd.DataFrame(
            dict(
                cycle_name=name,
                split="test",
                candidate_defrost_time=times,
                near_optimal=[True, True, False, True, True],
                input_valid=[True, False, True, True, True],
                cycle_cop_eligible=True,
            )
        )
        with path.open("wb") as stream:
            pickle.dump(dict(checkpoint={"seed": 42}, predictions=predictions), stream)
    monkeypatch.setattr(
        camera_comparison, "model_arguments", lambda *_: SimpleNamespace(output=tmp_path)
    )
    monkeypatch.setattr("image_models.checkpoints.checkpoint_files", lambda _: paths)
    camera_comparison.summarize_near_sets(SimpleNamespace(output=tmp_path))
    near = pd.read_csv(tmp_path / "camera_near_set_source.csv")
    np.testing.assert_allclose(
        near.near_minutes, [1.0, 1.0]
    )  # Two 30-second bands, not the 2-minute envelope.
    np.testing.assert_allclose(near.near_action_fraction, [0.75, 0.75])
    cameras = ["front", "top", "top_close", "left", "left_close", "extreme"]
    rows = [
        dict(
            evaluation_scope=scope,
            camera_role=camera,
            seed=42,
            experiments=1,
            cohort_cycles=1,
            rb_paired_cycles=1,
            headroom_captured_pct=90 - index if scope == "first_common" else 94 - index,
            mean_optimal_cop=2.0,
            mean_rb_cop=1.9 if scope == "first_common" else 1.8,
            mean_paired_model_cop=2.0
            - (0.01 if scope == "first_common" else 0.012)
            - 0.001 * index,
        )
        for scope in ["first_common", "second_common"]
        for index, camera in enumerate(cameras)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "comparison_summary.csv", index=False)
    captured = []
    monkeypatch.setattr(
        publication, "save_publication", lambda fig, *args, **kwargs: captured.append(fig)
    )
    publication.render_camera_comparison(tmp_path, layout="diagnostic")
    fig = captured[0]
    assert len(fig.axes) == 3 and len(fig.texts) == 1
    source = pd.read_csv(tmp_path / "camera_test_diagnostics_source.csv")
    front = source.loc[source.camera_role.eq("front")].set_index("evaluation_scope")
    np.testing.assert_allclose(
        front.loc[["first_common", "second_common"], "model_loss_pct"], [0.5, 0.6]
    )
    np.testing.assert_allclose(
        front.loc[["first_common", "second_common"], "rb_loss_pct"], [5.0, 10.0]
    )
    np.testing.assert_allclose(
        [bar.get_height() for bar in fig.axes[0].patches[:6]], [0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
    )
    np.testing.assert_allclose(fig.axes[2].lines[0].get_ydata(), [1.0, 1.0])


def test_camera_grouped_plot_uses_absolute_values_and_common_cohort_order(tmp_path, monkeypatch):
    from plots import camera as publication

    cameras = ["front", "top", "top_close", "left", "left_close", "extreme"]
    first = [85.0, 75.0, 80.0, 86.0, 83.0, 69.0]
    second = [86.0, 87.0, 84.0, 91.0, 88.0, 89.0]
    rows = [
        dict(
            camera_role=camera,
            evaluation_scope=scope,
            seed=42,
            experiments=27 if scope == "first_common" else 6,
            cohort_cycles=96 if scope == "first_common" else 12,
            rb_paired_cycles=95
            if camera == "extreme" and scope == "first_common"
            else 96
            if scope == "first_common"
            else 12,
            headroom_captured_pct=value,
        )
        for scope, values in [("first_common", first), ("second_common", second)]
        for camera, value in zip(cameras, values)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "comparison_summary.csv", index=False)
    captured = []
    monkeypatch.setattr(
        publication,
        "save_publication",
        lambda fig, output, name, caption, **kwargs: captured.append((fig, caption)),
    )
    publication.render_camera_comparison(tmp_path, layout="grouped")
    fig, caption = captured[0]
    assert len(fig.axes) == 1 and caption is None and len(fig.texts) == 1
    ax = fig.axes[0]
    order = ["extreme", "top", "top_close", "left_close", "front", "left"]
    assert [label.get_text() for label in ax.get_xticklabels()] == order
    np.testing.assert_allclose(
        [bar.get_height() for bar in ax.patches[:6]], [69, 75, 80, 83, 85, 86]
    )
    np.testing.assert_allclose(
        [bar.get_height() for bar in ax.patches[6:]], [89, 87, 84, 88, 86, 91]
    )
    for start in [0, 6]:
        np.testing.assert_allclose(
            ax.patches[start + 4].get_facecolor(),
            np.mean(
                [ax.patches[start + 3].get_facecolor(), ax.patches[start + 5].get_facecolor()],
                axis=0,
            ),
        )
    assert [text.get_text() for text in fig.legends[0].get_texts()] == ["共同集合", "跨工况集合"]
    source = pd.read_csv(tmp_path / "camera_core_grouped_source.csv")
    assert (
        len(source) == 12
        and source.loc[
            source.camera_role.eq("extreme") & source.evaluation_scope.eq("first_common"),
            "rb_paired_cycles",
        ].item()
        == 95
    )


@pytest.mark.parametrize(
    "seeds,scopes",
    [
        ([42], ["first_own", "first_common", "second_own", "second_common"]),
        ([42], ["own", "common_test", "common_train_test"]),
        ([0, 1, 2], ["own", "common_test", "common_train_test"]),
    ],
)
def test_camera_renderer_exports_matched_metrics_and_editable_text(
    tmp_path, monkeypatch, seeds, scopes
):
    from itertools import product

    from plots import camera as publication

    cameras = ["front", "top", "top_close", "left", "left_close", "extreme"]
    cohort = pd.DataFrame(
        [dict(camera_role=camera, cycle_name=name) for camera, name in product(cameras, ["a", "b"])]
    )
    if len(scopes) == 4:
        cohort = pd.concat(
            [cohort.assign(data_role=role) for role in ["first", "second"]], ignore_index=True
        )
    cohort.to_csv(tmp_path / "camera_cohorts.csv", index=False)
    counts = pd.DataFrame(dict(camera_role=cameras, cycles=2, experiments=2, intersection_cycles=2))
    if len(scopes) == 4:
        counts["all_standard_eligible_cycles"], counts["outside_front_reference"] = 3, 1
    counts.to_csv(tmp_path / "cohort_counts.csv", index=False)
    pd.DataFrame(
        [
            dict(
                evaluation_scope=scope,
                camera_role=camera,
                seed=seed,
                headroom_captured_pct=30 + seed,
                gain_vs_rb_pct=1 + seed / 10,
                hit_rate_all_pct=70 + seed,
            )
            for scope, camera, seed in product(scopes, cameras, seeds)
        ]
    ).to_csv(tmp_path / "comparison_summary.csv", index=False)
    paired_scopes = ["first_common", "second_common"] if len(scopes) == 4 else scopes[1:]
    for scope in paired_scopes:
        folder = tmp_path / scope
        folder.mkdir()
        paired = 2 if scope == paired_scopes[0] else 1
        pd.DataFrame(
            dict(
                left="front",
                right=cameras[1:],
                paired_cycles=paired,
                test_experiments=paired,
                pooled_H_delta_pp=1.0,
                pooled_H_lower_95=-1.0,
                pooled_H_upper_95=3.0,
            )
        ).to_csv(folder / "experiment_bootstrap.csv", index=False)
    save = publication.save_publication
    captured = []

    def export(fig, output, name, caption, **kwargs):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.text import Text

        captured.append((name, len(fig.axes)))
        save(fig, output, name, caption, formats=("svg",))
        FigureCanvasAgg(fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        off_view = set()
        for ax in fig.axes:
            for axis in [ax.xaxis, ax.yaxis]:
                low, high = sorted(axis.get_view_interval())
                for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                    if not low - 1e-9 <= tick.get_loc() <= high + 1e-9:
                        off_view.update([tick.label1, tick.label2])
        for text in fig.findobj(match=Text):
            if text.get_visible() and text.get_text() and text not in off_view:
                box = text.get_window_extent(renderer)
                assert (
                    box.x0 >= -1
                    and box.y0 >= -1
                    and box.x1 <= fig.bbox.width + 1
                    and box.y1 <= fig.bbox.height + 1
                ), (name, text.get_text())
        footer = fig.texts[-1].get_window_extent(renderer)
        assert not any(
            footer.overlaps(ax.xaxis.label.get_window_extent(renderer)) for ax in fig.axes
        )

    monkeypatch.setattr(publication, "save_publication", export)
    publication.render_camera_comparison(tmp_path)
    assert [count for _, count in captured] == [2, 9, 2]
    assert all("font-family" in (tmp_path / f"{name}.svg").read_text() for name, _ in captured)
    paired_svg = (tmp_path / "camera_paired_effects.svg").read_text()
    assert "n=2, E=2" in paired_svg and "n=1, E=1" in paired_svg
    assert (
        "Single-seed paired effects" if len(seeds) == 1 else "Seed-averaged paired effects"
    ) in paired_svg
    overlap = pd.read_csv(tmp_path / "pairwise_cycle_overlap.csv", index_col=0)
    assert overlap.eq(2).all().all()
    (tmp_path / "comparison_summary.csv").unlink()
    captured.clear()
    publication.render_camera_comparison(tmp_path, coverage_only=True)
    assert captured == [("camera_coverage", 2)]


import matplotlib.pyplot as plt

from plots.defrost_decision import match_decision_rgb_images


def test_effective_screening_plot_counts_extrapolated_max_and_continuous_intervals(
    monkeypatch, tmp_path
):
    import plots.defrost_decision as renderer
    from plots.defrost_decision import _plot_effective_cop_panel, render_current_decision_cycle

    times = pd.date_range("2026-01-01", periods=5, freq="30s")
    curve = pd.DataFrame(
        dict(
            candidate_defrost_time=times,
            cycle_name="sample",
            algorithm="effective_cop",
            cycle_cop=[1.0, 2.0, 1.0, 1.99, 1.0],
            cycle_cop_eligible=[True, False, True, True, True],
            is_selected=[False, False, False, True, False],
            t_RB=times[-1],
            pre_defrost_heat_kwh=[1.0, 2.0, 1.0, 1.99, 1.0],
            pre_defrost_electricity_kwh=0.9,
            defrost_event_electricity_kwh=0.1,
            defrost_event_net_heat_kwh=0.0,
            pre_defrost_feature_window_valid=True,
            pre_defrost_electricity_measurement_valid=True,
            pre_defrost_heat_measurement_valid=True,
            defrost_event_electricity_prediction_available=True,
            defrost_event_net_heat_prediction_available=True,
            defrost_event_electricity_in_training_domain=[True, False, True, True, True],
            defrost_event_net_heat_in_training_domain=True,
        )
    )

    class Loader:
        def load_image_metadata(self, name):
            return pd.DataFrame()

        def load_cycle_images(self, name):
            return pd.DataFrame(
                dict(
                    cycle_name=["sample"],
                    camera_role=["front"],
                    file_name=["frame_1.jpg"],
                    path=[tmp_path / "frame_1.jpg"],
                )
            )

        def load_cycle(self, name):
            return pd.DataFrame()

        def get_cycle_record(self, name):
            return {}

    captured = {}
    monkeypatch.setattr(renderer, "match_decision_rgb_images", lambda *a: pd.DataFrame())
    native = pd.DataFrame(
        dict(
            cycle_name="sample",
            camera_role="front",
            file_name=[f"frame_{i}.jpg" for i in range(len(times))],
            image_time=times,
            visual_frost_score=np.linspace(0, 1, len(times)),
        )
    )
    monkeypatch.setattr(renderer, "load_native_frost_frames", lambda: native)
    monkeypatch.setattr(
        renderer, "render_decision_publication", lambda *a, **kw: captured.update(kw)
    )
    render_current_decision_cycle(curve, Loader(), tmp_path / "test.png")
    plotted = captured["parallel_curve"]
    pd.testing.assert_frame_equal(captured["frost_frames"].drop(columns="image_path"), native)
    assert captured["frost_frames"].image_path.notna().tolist() == [
        False,
        True,
        False,
        False,
        False,
    ]
    assert captured["frost_frames"].image_path.iloc[1] == tmp_path / "frame_1.jpg"
    assert plotted.t_star.eq(times[1]).all()
    assert plotted.is_selected.tolist() == [False, True, False, False, False]
    assert plotted.cycle_cop_uses_model_extrapolation.tolist() == [False, True, False, False, False]
    assert not curve.cycle_cop_eligible.iloc[1]
    fig, ax = plt.subplots()
    right = _plot_effective_cop_panel(
        ax, plotted, times[0], [], full_near_sets=captured["full_near_sets"]
    )
    # Each interval spans the low-COP point between the two qualifying times.
    assert len(ax.patches) == 3
    assert plotted.cycle_cop_basin_1pct_start.eq(times[1]).all()
    assert plotted.cycle_cop_basin_1pct_end.eq(times[3]).all()
    assert np.nanmax(right.lines[1].get_ydata()) == pytest.approx(100.0)
    assert np.nanmax(right.lines[0].get_ydata()) == pytest.approx(99.5)
    assert right.lines[1].get_linestyle() == "--"  # Extrapolation remains dashed.
    plt.close(fig)
    fig, ax = plt.subplots()
    plotted.loc[2, "cycle_cop_eligible"] = False  # Missing validity must also be bridged.
    _plot_effective_cop_panel(
        ax, plotted, times[0], [], full_near_sets=True, connected_near_sets=True
    )
    assert len(ax.patches) == 3  # One connected interval for each 1/2/5% band.
    assert all("连续近优区间" in patch.get_label() for patch in ax.patches)
    assert not plotted.cycle_cop_eligible.iloc[2]  # Shading never imputes a COP value.
    plt.close(fig)


def test_decision_image_offset_keeps_before_after_direction(tmp_path: Path) -> None:
    target = pd.Timestamp("2026-01-01 00:01:00")
    image_path = tmp_path / "front.jpg"
    image_path.touch()
    metadata = pd.DataFrame(
        {
            "camera_role": ["front"],
            "image_time": [target - pd.Timedelta(seconds=11)],
            "file_name": [image_path.name],
        }
    )
    images = metadata.assign(path=[image_path])

    matched = match_decision_rgb_images(metadata, images, {"rb": target, "optimal": target})

    assert matched["offset_seconds"].tolist() == [-11.0, -11.0]


@pytest.mark.parametrize("with_frost_images", [False, True])
def test_effective_publication_merges_cop_axes_and_interpolates_display(
    tmp_path, monkeypatch, with_frost_images
):
    from matplotlib.figure import Figure

    from plots.cop import OPTIMAL_COLOR, RB_COLOR
    from plots.defrost_decision import render_decision_publication

    start = pd.Timestamp("2026-01-01")
    times = pd.date_range(start, periods=5, freq="min")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "cycle_stage": "frost_development",
            "cop": [2.0, np.nan, 2.2, 2.3, 2.4],
            "water_in_temperature": 20.0,
            "water_out_temperature": 25.0,
            "water_temperature_setpoint": 30.0,
        }
    )
    curve = pd.DataFrame(
        {
            "candidate_defrost_time": times,
            "algorithm": "effective_cop",
            "preparation_heat": "include",
            "cycle_status": "identified_curve",
            "cycle_cop": [2.0, np.nan, 2.2, 2.1, 2.4],
            "cycle_cop_eligible": [True, True, True, True, False],
            "t_star": times[2],
            "t_RB": times[3],
            "defrost_event_electricity_kwh": [0.1, 0.12, 0.14, 0.16, 0.18],
        }
    )
    captured = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *_a, **_k: captured.append(figure))
    frost_frames = pd.DataFrame(
        dict(image_time=times[[0, 1, 4]], visual_frost_score=[0.1, 0.6, 0.3])
    )
    if with_frost_images:
        frost_frames["image_time"] = start + pd.to_timedelta([0.4, 1.2, 3.6], unit="min")
        image_path = tmp_path / "frost.png"
        plt.imsave(image_path, np.zeros((10, 20, 3)))
        frost_frames["image_path"] = str(image_path)
    render_decision_publication(
        frame,
        {},
        curve,
        {"rb": {"target_time": times[3]}, "optimal": {"target_time": times[2]}},
        tmp_path / "effective.png",
        parallel_curve=curve,
        figure_note="Old caption must be removed",
        frost_frames=frost_frames,
    )
    figure = captured[0]
    assert len(figure.axes) == 8
    image, _, cop, water, cost, absolute, relative, frost = figure.axes[:8]
    if with_frost_images:
        sources = pd.read_csv(tmp_path / "effective.frost_images.csv")
        assert pd.to_datetime(sources.image_time).tolist() == frost_frames.image_time.tolist()
        assert sources.visual_frost_score.tolist() == [0.1, 0.6, 0.3]
        assert len(frost.child_axes) == 3
        leaders = [
            line
            for line in frost.lines
            if len(line.get_xdata()) == 2
            and line.get_xdata()[0] == line.get_xdata()[1]
            and np.isclose(np.diff(line.get_ydata())[0], 0.24)
        ]
        assert len(leaders) == 3
        assert [line.get_xdata()[0] for line in leaders] == pytest.approx([0.4, 1.2, 3.6])
        assert [np.diff(line.get_ydata())[0] for line in leaders] == pytest.approx([0.24] * 3)
    assert frost.get_position().y1 < absolute.get_position().y0
    assert frost.get_xlim() == absolute.get_xlim()
    assert frost.lines[0].get_ydata().tolist() == [0.1, 0.6]
    assert frost.lines[1].get_ydata().tolist() == [0.3]  # Preserve decrease and missing-frame gap.
    assert "Harmony + Slingshot" in frost.get_ylabel()
    assert frost.get_xlabel() == "Time from cycle start [min]"
    assert not any(text.get_text() == "Old caption must be removed" for text in figure.texts)
    assert image.get_position().height > 1.5 * cop.get_position().height
    assert np.isnan(cop.lines[0].get_ydata()[1])  # Missing observations are not measured COP.
    assert water.get_ylabel() == "Water temperature [degC]"
    assert cost.get_ylabel() == "Predicted defrost cost [kWh]"
    assert cost.lines[0].get_ydata().tolist() == [0.1, 0.12, 0.14, 0.16, 0.18]
    assert cost.get_position().y0 > absolute.get_position().y0
    assert absolute.get_ylabel() == "Effective cycle COP [-]"
    trigger_lines = {
        line.get_label(): line
        for line in absolute.lines
        if line.get_label() in {"RB defrost trigger", "COP-optimal defrost trigger"}
    }
    low, high = absolute.get_ylim()
    for label, value in [("RB defrost trigger", 2.1), ("COP-optimal defrost trigger", 2.2)]:
        assert trigger_lines[label].get_ydata()[0] == 0
        assert low + trigger_lines[label].get_ydata()[1] * (high - low) == pytest.approx(value)
    assert not any(
        line.get_color() in {RB_COLOR, OPTIMAL_COLOR} for line in frost.lines
    )  # Both trigger times lie in the native-frame gap.
    assert absolute.lines[0].get_linestyle() == "-"
    assert absolute.lines[0].get_alpha() is None
    assert absolute.lines[0].get_ydata()[1] == pytest.approx(2.1)
    assert absolute.lines[1].get_ydata()[-2] == pytest.approx(2.4)
    assert np.nanmax(relative.lines[0].get_ydata()) == pytest.approx(100)
    assert relative.get_ylim() == (90, 100.8)
    assert relative.spines["right"].get_visible()
    assert relative.spines["right"].get_edgecolor() == (0, 0, 0, 1)
    assert {text.get_text() for text in relative.texts} >= {"1%", "2%", "5%"}
    legend = {text.get_text() for text in absolute.get_legend().get_texts()}
    assert legend >= {
        "RB defrost trigger",
        "COP-optimal defrost trigger",
        "Ridge extrapolation",
        "Normalized cycle COP",
    }
    assert "Interpolated inputs" not in legend  # No input-interpolation evidence in this fixture.
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    image_top = max(axis.get_tightbbox(renderer).y1 for axis in [image, figure.axes[1]])
    stage_box = figure.legends[0].get_window_extent(renderer)
    title_box = figure._suptitle.get_window_extent(renderer)
    assert stage_box.y0 - image_top == pytest.approx(8 * figure.dpi / 72)
    assert title_box.y0 - stage_box.y1 == pytest.approx(8 * figure.dpi / 72)
    groups = [[image, figure.axes[1]], [cop], [water], [cost], [absolute, relative], [frost]]
    bounds = [
        (
            min(axis.get_tightbbox(renderer).y0 for axis in group),
            max(axis.get_tightbbox(renderer).y1 for axis in group),
        )
        for group in groups
    ]
    gaps = [upper[0] - lower[1] for upper, lower in zip(bounds[:-1], bounds[1:])]
    assert min(gaps) > 0
    assert gaps == pytest.approx([gaps[-1]] * len(gaps), abs=1)
    if with_frost_images:
        panel_box = frost.get_window_extent(renderer)
        for photo, leader in zip(frost.child_axes, leaders):
            box = photo.get_window_extent(renderer)
            endpoint = frost.transData.transform((leader.get_xdata()[-1], leader.get_ydata()[-1]))
            assert (box.x0 + box.x1) / 2 == pytest.approx(endpoint[0])
            assert box.y0 == pytest.approx(endpoint[1])
            assert panel_box.x0 <= box.x0 < box.x1 <= panel_box.x1
            assert panel_box.y0 <= box.y0 < box.y1 <= panel_box.y1
    assert (
        not absolute.get_legend()
        .get_window_extent(renderer)
        .overlaps(cost.get_window_extent(renderer))
    )
    assert pd.isna(curve.cycle_cop.iloc[1])


@pytest.mark.parametrize("folder", ["cycles", "cycles_RB"])
@pytest.mark.parametrize("without_trunc", [False, True])
def test_publish_effective_decision_preserves_history_and_manual_status(
    tmp_path, folder, without_trunc, monkeypatch
):
    from plots.dataset import publish_effective_decision_assets

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("plots.dataset.PROJECT", tmp_path)

    dataset = tmp_path / "dataset"
    (dataset / folder).mkdir(parents=True)
    name = "frost_cycle_000001"
    old = dataset / folder / f"{name}_pareto_decision.png"
    old.write_bytes(b"historical figure")
    record = {
        "cycle_name": name,
        "status": "valid",
        "assets": {"csv": f"{folder}/{name}.csv", "pareto_decision": f"{folder}/{old.name}"},
        "pareto_decision": {"status": "old"},
    }
    if without_trunc:
        legacy = dataset / folder / f"{name}_effective_cop.png"
        legacy.write_bytes(b"original effective COP")
        record["assets"]["effective_cop"] = f"{folder}/{legacy.name}"
        record["effective_cop"] = {"status": "original"}
    (dataset / "cycle_catalog.json").write_text(json.dumps({"cycles": [record]}))
    run = tmp_path / "run"
    run.mkdir()
    if without_trunc:
        (run / "run_settings.json").write_text(json.dumps({"effective_heat_rule": "without_trunc"}))
    pd.DataFrame(
        [{"cycle_name": name, "cycle_status": "identified_curve", "preparation_heat": "zero"}]
    ).to_csv(run / "cycle_comparison.csv", index=False)
    figures = tmp_path / "figures"
    source = figures / "cost_function_effective_cop_cycle" / "cycle_001_publication.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"effective figure")
    publish_effective_decision_assets(dataset, run, figures)
    current = json.loads((dataset / "cycle_catalog.json").read_text())["cycles"][0]
    assert current["status"] == "valid"
    key = "effective_cop_without_trunc" if without_trunc else "effective_cop"
    assert ("pareto_decision" in current["assets"]) == without_trunc
    assert current[key]["preparation_heat"] == "zero"
    published = dataset / current["assets"][key]
    assert published.parent.name == folder
    assert published.read_bytes() == b"effective figure"
    if without_trunc:
        assert old.read_bytes() == b"historical figure"
        assert legacy.read_bytes() == b"effective figure"
        assert (
            Path("output/99_历史归档/dataset_effective_cop") / run.name / legacy.name
        ).read_bytes() == b"original effective COP"
        assert "effective_cop" not in current
    else:
        assert (
            Path("output/99_历史归档/dataset_effective_cop") / run.name / old.name
        ).read_bytes() == b"historical figure"
        assert not old.exists()

    table = pd.read_csv(run / "cycle_comparison.csv")
    table["preparation_heat"] = "include"
    table.to_csv(run / "cycle_comparison.csv", index=False)
    with pytest.raises(ValueError, match="zero preparation heat"):
        publish_effective_decision_assets(dataset, run, figures)
    assert published.name == f"{name}_effective_cop.png"
    assert published.parent.name == folder
    assert published.read_bytes() == b"effective figure"


def test_refresh_effective_figures_updates_catalog_target_and_copies_without_changing_metadata(
    tmp_path, monkeypatch
):
    from plots.dataset import refresh_effective_decision_figures

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("plots.dataset.PROJECT", tmp_path)
    dataset = tmp_path / "dataset"
    name = "frost_cycle_000001"
    destinations = [
        dataset / "cycles" / f"{name}_effective_cop.png",
        dataset / "training_data" / f"{name}_effective_COP.png",
        dataset / "RB失败循环" / "早除霜" / f"{name}_effective_COP.png",
    ]
    for target in destinations:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"old figure")
    metadata = json.dumps(
        {
            "cycles": [
                dict(
                    cycle_name=name,
                    status="valid",
                    assets={
                        "effective_cop_without_trunc": str(destinations[0].relative_to(dataset))
                    },
                    effective_cop_without_trunc={
                        "source_table": "frozen.csv",
                        "status": "identified_curve",
                    },
                )
            ]
        }
    )
    (dataset / "cycle_catalog.json").write_text(metadata)
    source = tmp_path / "new.png"
    source.write_bytes(b"taller frost figure")
    refresh_effective_decision_figures(dataset, {name: source})
    assert all(p.read_bytes() == source.read_bytes() for p in destinations)
    assert (dataset / "cycle_catalog.json").read_text() == metadata
    backups = list(Path("output/99_历史归档/dataset_effective_cop").rglob("*.png"))
    assert len(backups) == 3 and all(p.read_bytes() == b"old figure" for p in backups)


def test_decision_summary_preserves_missing_optimum_and_rb_times():
    from defrost_decision.selection_results import summarize_decisions

    curve = pd.DataFrame(
        dict(
            cycle_name=["missing"],
            experiment_id="day",
            candidate_defrost_time=pd.Timestamp("2026-01-01"),
            is_selected=False,
            cycle_cop_eligible=False,
            cycle_cop=np.nan,
            t_star=None,
            t_RB=None,
            cycle_status="unavailable",
            rb_status="unavailable",
            preparation_heat="zero",
            cycle_start=pd.Timestamp("2026-01-01"),
            stable_heating_start=None,
        )
    )
    summary = summarize_decisions(curve)
    assert (
        summary[["t_star", "t_RB", "time_difference_minutes", "COP_eff_selected", "COP_eff_RB"]]
        .isna()
        .all()
        .all()
    )


def test_event_prediction_cycle_preserves_energy_step(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    from plots import event_calibration as publication

    assert hasattr(publication, "render_event_prediction_cycle")
    start = pd.Timestamp("2026-01-01")
    times = pd.date_range(start, periods=4, freq="min")
    frame = pd.DataFrame(
        dict(
            timestamp=times, cycle_stage="frost_development", heating_capacity=6.0, power_total=2.0
        )
    )
    curve = pd.DataFrame(
        dict(
            candidate_defrost_time=times,
            cycle_cop=[2.0, 2.1, 2.5, 2.4],
            cycle_cop_eligible=[True, True, True, False],
            defrost_event_net_heat_kwh=0.0,
            defrost_event_electricity_kwh=[0.3, 0.3, 0.2, 0.2],
            defrost_event_electricity_in_training_domain=[True, True, True, False],
            preparation_heat="zero",
            t_star=times[2],
            t_RB=times[1],
        )
    )
    original = curve.copy(deep=True)
    captured = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *_a, **_k: captured.append(fig))
    publication.render_event_prediction_cycle(
        frame, {"cycle_name": "example"}, curve, tmp_path / "predictions.png"
    )
    fig = captured[0]
    q_axes = [ax for ax in fig.axes if ax.get_ylabel() == "Event effective heat [kWh]"]
    e_axes = [ax for ax in fig.axes if ax.get_ylabel() == "Predicted event electricity [kWh]"]
    assert not q_axes
    assert len(e_axes) == 2
    for ax in e_axes:
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0.3, 0.3, 0.2, 0.2])
        assert np.isnan(ax.lines[1].get_ydata()[-1])
    pd.testing.assert_frame_equal(curve, original)


def test_supported_curve_can_preserve_frontier_gaps_without_interpolation():
    import matplotlib.pyplot as plt

    from plots.cop import _plot_supported_curve

    fig, axis = plt.subplots()
    _plot_supported_curve(
        axis,
        pd.Series([0.0, 1.0, 2.0, 3.0]),
        pd.Series([1.0, np.nan, 0.95, 0.98]),
        pd.Series([True, False, False, True]),
        color="blue",
        label="frontier",
        interpolate=False,
        dashed_alpha=0.4,
    )
    assert np.isnan(axis.lines[0].get_ydata()[1])
    assert axis.lines[0].get_ydata()[2] == 0.95
    assert np.isnan(axis.lines[1].get_ydata()[2])
    plt.close(fig)


def test_effective_cop_styles_split_at_original_missing_intervals():
    from plots.cop import _plot_supported_curve, original_missing_spans

    origin = pd.Timestamp("2026-01-01")
    sensor = pd.DataFrame(
        dict(
            timestamp=origin + pd.to_timedelta([0, 15, 30, 90, 150, 240], unit="s"),
            is_interpolated=[False, True, False, True, False, False],
        )
    )
    spans = original_missing_spans(sensor, origin)
    assert spans == [(0.25, 0.5), (1.5, 2.5)]
    fig, axis = plt.subplots()
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    _plot_supported_curve(
        axis,
        pd.Series([0.0, 1.0, 2.0, 3.0, 4.0]),
        values,
        pd.Series(True, index=values.index),
        color="blue",
        label="COP",
        extrapolated=[False, False, True, True, False],
        missing_spans=spans,
    )
    assert [(line.get_color(), line.get_linestyle()) for line in axis.lines] == [
        ("blue", "-"),
        ("blue", "--"),
        ("0.6", "-"),
        ("0.6", "--"),
    ]
    np.testing.assert_allclose(axis.lines[2].get_xdata()[:-1], [0.25, 0.5])
    np.testing.assert_allclose(axis.lines[3].get_xdata()[:-1], [1.5, 2.0, 2.5])
    np.testing.assert_allclose(axis.lines[3].get_ydata()[:-1], [2.5, 3.0, 3.5])
    np.testing.assert_allclose(values, [1.0, 2.0, 3.0, 4.0, 5.0])
    plt.close(fig)


def test_interpolation_style_uses_only_the_metric_inputs():
    from plots.cop import original_missing_spans

    origin = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame(
        dict(
            timestamp=pd.date_range(origin, periods=4, freq="s"),
            is_interpolated=True,
            power_total=2.0,
            power_total_interpolated=2.0,
            heating_capacity=6.0,
            heating_capacity_interpolated=6.0,
        )
    )
    assert original_missing_spans(frame, origin, columns=["power_total", "heating_capacity"]) == []
    frame.loc[1, "power_total"] = np.nan
    assert original_missing_spans(frame, origin, columns=["power_total"]) == [(1 / 60, 2 / 60)]


def test_disconnected_near_sets_have_one_legend_entry_each():
    from plots.defrost_decision import _plot_effective_cop_panel

    times = pd.date_range("2026-01-01", periods=41, freq="30s")
    curve = pd.DataFrame(
        dict(
            candidate_defrost_time=times,
            cycle_cop=np.where(np.arange(41) % 2, 1.5, 2.0),
            cycle_cop_eligible=True,
            cycle_cop_uses_model_extrapolation=False,
        )
    )
    before = curve.copy()
    figure, axis = plt.subplots()
    _plot_effective_cop_panel(axis, curve, times[0], [], full_near_sets=True)
    assert len(axis.patches) == 63  # All 21 components of each set remain visible.
    labels = axis.get_legend_handles_labels()[1]
    for percent in (1, 2, 5):
        assert labels.count(f"Within {percent}% of best COP") == 1
    assert "Ridge extrapolation" not in labels
    pd.testing.assert_frame_equal(curve, before)
    plt.close(figure)
