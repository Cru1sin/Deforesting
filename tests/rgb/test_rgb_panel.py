from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex


def test_build_rgb_panel_targets_uses_stages_boundaries_and_fallbacks() -> None:
    from frost_analysis.visualization import build_rgb_panel_targets

    times = pd.date_range("2026-07-14 10:00:00", periods=13, freq="10s")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "cycle_stage": ["recovery"] * 2 + ["frost_development"] * 8 + ["defrost"] * 3,
        }
    )
    record = {
        "boundaries": {
            "start_time": None,
            "end_time": None,
            "heating_start": times[0].isoformat(),
            "stable_heating_start": times[2].isoformat(),
            "defrost_start": times[10].isoformat(),
            "defrost_end": times[12].isoformat(),
        }
    }

    targets = build_rgb_panel_targets(record, frame)

    assert [target["label"] for target in targets] == [
        "Start",
        "Recovery End",
        "Frost 25%",
        "Frost 50%",
        "Frost 75%",
        "Defrost Start",
        "Defrost Mid",
        "End",
    ]
    assert [target["target_time"] for target in targets] == [
        times[0],
        times[2],
        times[4],
        times[6],
        times[8],
        times[10],
        times[11],
        times[12],
    ]


def test_rgb_overall_intervals_requires_every_expected_role() -> None:
    from frost_analysis.dataset_images import rgb_overall_intervals

    times = pd.date_range("2026-07-14 10:00:00", periods=4, freq="10s")
    frame = pd.DataFrame({"timestamp": times})
    intervals = {
        "front": {"available": [(times[0], times[3] + pd.Timedelta(seconds=10))]},
        "top": {"available": [(times[1], times[3])]},
    }

    result = rgb_overall_intervals(frame, intervals, ("front", "top"))

    assert result["available"] == [(times[1], times[3])]
    assert result["missing"] == [
        (times[0], times[1]),
        (times[3], times[3] + pd.Timedelta(seconds=10)),
    ]


def test_publication_contains_sensor_and_rgb_availability_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis.visualization import render_cycle_publication

    times = pd.date_range("2026-07-14 10:00:00", periods=4, freq="10s")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "cycle_stage": ["recovery", "frost_development", "frost_development", "defrost"],
        }
    )
    available = {
        "available": [(times[0], times[-1] + pd.Timedelta(seconds=10))],
        "missing": [],
    }
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda _figure: None)

    render_cycle_publication(
        frame,
        {"cycle_name": "frost_cycle_000001", "status": "valid", "boundaries": {}},
        tmp_path / "publication.png",
        sensor_intervals=available,
        rgb_intervals=available,
    )

    figure = plt.gcf()
    labels = {text.get_text() for text in figure.axes[0].texts}
    assert {"Sensor", "RGB", "Recovery", "Frost Development", "Defrost"} <= labels
    assert "Capacity / power [kW]" in {axis.get_ylabel() for axis in figure.axes}
    assert len(figure.axes) == 6
    original_close(figure)


def test_publication_cost_panel_uses_full_cycle_axis_but_only_plots_frosting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis.visualization import render_cycle_publication

    times = pd.date_range("2026-07-14 10:00:00", periods=4, freq="1min")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "cycle_stage": ["recovery", "frost_development", "frost_development", "defrost"],
        }
    )
    cost = pd.DataFrame(
        {
            "candidate_time": times,
            "inverse_cop": [9.0, 0.5, 0.4, 8.0],
            "cycle_cop": [1 / 9, 2.0, 2.5, 0.125],
            "relative_regret": [3.5, 0.5, 0.0, 3.0],
        }
    )
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda _figure: None)

    render_cycle_publication(
        frame,
        {
            "cycle_name": "frost_cycle_000001",
            "status": "valid",
            "boundaries": {"defrost_start": times[-1].isoformat()},
        },
        tmp_path / "publication.png",
        cost_curve=cost,
    )

    axis = next(axis for axis in plt.gcf().axes if axis.get_ylabel() == "Cycle inverse COP [-]")
    curve = next(line for line in axis.lines if line.get_label() == "Cycle inverse COP")
    assert list(curve.get_xdata()) == [1.0, 2.0]
    assert axis.get_xlim()[0] <= 0.0
    assert axis.get_xlim()[1] >= 3.0
    assert len(axis.patches) >= 3
    assert {line.get_label() for line in axis.lines} >= {
        "Cycle inverse COP",
        "Minimum",
        "Observed defrost",
    }
    original_close(plt.gcf())


def test_publication_cost_panel_preserves_support_holes_and_preparation_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis.visualization import render_cycle_publication

    times = pd.date_range("2026-07-14 10:00:00", periods=5, freq="1min")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "cycle_stage": ["recovery", *(["frost_development"] * 4)],
        }
    )
    cost = pd.DataFrame(
        {
            "candidate_time": times[1:],
            "inverse_cop": [0.50, 0.51, 0.49, 0.515],
            "cycle_cop": [2.0, 1 / 0.51, 1 / 0.49, 1 / 0.515],
            "relative_regret": [0.0, 0.02, np.nan, 0.03],
            "pe_supported": [True, True, False, True],
            "integration_eligible": [True, False, True, True],
            "optimization_eligible": [True, False, False, True],
            "support_status": ["supported", "supported", "above", "supported"],
            "minimum_location": ["interior"] * 4,
            "actual_preparation_time": times[-1],
        }
    )
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda _figure: None)

    render_cycle_publication(
        frame,
        {"cycle_name": "cycle", "status": "valid"},
        tmp_path / "publication.png",
        cost_curve=cost,
    )

    axis = next(axis for axis in plt.gcf().axes if axis.get_ylabel() == "Cycle inverse COP [-]")
    curve = next(line for line in axis.lines if line.get_label() == "Cycle inverse COP")
    assert np.isnan(curve.get_ydata()[2])
    assert len([patch for patch in axis.patches if str(patch.get_label()).startswith("5%")]) == 2
    assert {line.get_label() for line in axis.lines} >= {
        "Minimum",
        "Observed preparation",
    }
    assert {collection.get_label() for collection in axis.collections} >= {
        "Outside Pe support",
        "Insufficient integration coverage",
    }
    original_close(plt.gcf())


def test_publication_cost_panel_marks_triggered_rb_at_nearest_valid_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis.visualization import render_cycle_publication

    times = pd.date_range("2026-07-14 10:00:00", periods=5, freq="1min")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "cycle_stage": ["recovery", *("frost_development" for _ in range(4))],
        }
    )
    cost = pd.DataFrame(
        {
            "candidate_time": times[1:],
            "inverse_cop": [0.55, 0.50, 0.51, 0.54],
            "optimization_eligible": [True, True, False, True],
            "t_RB": times[2] + pd.Timedelta(seconds=20),
            "rb_status": "triggered",
        }
    )
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda _figure: None)

    render_cycle_publication(
        frame,
        {"cycle_name": "cycle", "status": "valid"},
        tmp_path / "publication.png",
        cost_curve=cost,
    )

    axis = next(axis for axis in plt.gcf().axes if axis.get_ylabel() == "Cycle inverse COP [-]")
    rb_line = next(line for line in axis.lines if line.get_label() == "RB trigger")
    rb_cost = next(
        collection
        for collection in axis.collections
        if collection.get_label() == "Nearest eligible cost"
    )
    minimum = next(line for line in axis.lines if line.get_label() == "Minimum")
    assert rb_line.get_xdata()[0] == pytest.approx(2 + 20 / 60)
    assert rb_line.get_linestyle() == "--"
    assert to_hex(rb_line.get_color()).upper() == "#2E7D5B"
    assert rb_cost.get_offsets()[0].tolist() == pytest.approx([2.0, 0.50])
    assert minimum.get_zorder() > rb_line.get_zorder()
    original_close(plt.gcf())


def test_publication_cost_panel_does_not_fake_right_censored_rb_trigger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis.visualization import render_cycle_publication

    times = pd.date_range("2026-07-14 10:00:00", periods=4, freq="1min")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "cycle_stage": ["recovery", *("frost_development" for _ in range(3))],
        }
    )
    cost = pd.DataFrame(
        {
            "candidate_time": times[1:],
            "inverse_cop": [0.55, 0.50, 0.51],
            "optimization_eligible": True,
            "t_RB": times[-1],
            "rb_status": "right_censored",
        }
    )
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda _figure: None)

    render_cycle_publication(
        frame,
        {"cycle_name": "cycle", "status": "valid"},
        tmp_path / "publication.png",
        cost_curve=cost,
    )

    axis = next(axis for axis in plt.gcf().axes if axis.get_ylabel() == "Cycle inverse COP [-]")
    labels = [*axis.get_legend_handles_labels()[1]]
    assert not any(label.startswith("RB") for label in labels)
    original_close(plt.gcf())


def test_publication_cost_panel_does_not_attach_rb_to_distant_eligible_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis.visualization import render_cycle_publication

    times = pd.date_range("2026-07-14 10:00:00", periods=4, freq="1min")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "cycle_stage": ["recovery", *("frost_development" for _ in range(3))],
        }
    )
    cost = pd.DataFrame(
        {
            "candidate_time": times[1:],
            "inverse_cop": [0.55, 0.50, 0.51],
            "optimization_eligible": [True, True, False],
            "t_RB": times[2] + pd.Timedelta(seconds=40),
            "rb_status": "triggered",
        }
    )
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda _figure: None)

    render_cycle_publication(
        frame,
        {"cycle_name": "cycle", "status": "valid"},
        tmp_path / "publication.png",
        cost_curve=cost,
    )

    axis = next(axis for axis in plt.gcf().axes if axis.get_ylabel() == "Cycle inverse COP [-]")
    assert "RB trigger" in {line.get_label() for line in axis.lines}
    assert "Nearest eligible cost" not in {
        collection.get_label() for collection in axis.collections
    }
    original_close(plt.gcf())


def test_publication_cost_panel_labels_zero_joint_eligibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis.visualization import render_cycle_publication

    times = pd.date_range("2026-07-14 10:00:00", periods=4, freq="1min")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "cycle_stage": ["recovery", *("frost_development" for _ in range(3))],
        }
    )
    cost = pd.DataFrame(
        {
            "candidate_time": times[1:],
            "inverse_cop": [np.nan, np.nan, np.nan],
            "optimization_eligible": [False, False, False],
            "support_status": ["supported", "supported", "supported"],
            "actual_preparation_time": times[-1],
        }
    )
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda _figure: None)

    render_cycle_publication(
        frame,
        {"cycle_name": "cycle", "status": "valid"},
        tmp_path / "publication.png",
        cost_curve=cost,
    )

    axis = next(axis for axis in plt.gcf().axes if axis.get_ylabel() == "Cycle inverse COP [-]")
    messages = {text.get_text() for text in axis.texts}
    assert "No optimization-eligible candidates" in messages
    assert "No frosting cost candidates" not in messages
    assert "Insufficient integration coverage" in {
        collection.get_label() for collection in axis.collections
    }
    original_close(plt.gcf())


def test_publication_humidity_uses_stage_and_missing_backgrounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis.visualization import render_cycle_publication

    times = pd.date_range("2026-07-14 10:00:00", periods=4, freq="10s")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "cycle_stage": ["recovery", "frost_development", "frost_development", "defrost"],
            "environment_relative_humidity": [90.0, 89.0, 88.0, 91.0],
        }
    )
    intervals = {
        "available": [(times[0], times[1]), (times[2], times[-1] + pd.Timedelta(seconds=10))],
        "missing": [(times[1], times[2])],
    }
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda _figure: None)

    render_cycle_publication(
        frame,
        {"cycle_name": "frost_cycle_000001", "status": "valid", "boundaries": {}},
        tmp_path / "publication.png",
        sensor_intervals=intervals,
        rgb_intervals=intervals,
    )

    humidity_axis = plt.gcf().axes[-1]
    assert humidity_axis.get_ylabel() == "Relative humidity [%]"
    assert len(humidity_axis.patches) == 4
    assert humidity_axis.patches[-1].get_hatch() == "////"
    original_close(plt.gcf())


def test_build_rgb_panel_targets_leaves_absent_stages_empty_but_keeps_frost() -> None:
    from frost_analysis.visualization import build_rgb_panel_targets

    times = pd.date_range("2026-07-14 10:00:00", periods=9, freq="10s")
    frame = pd.DataFrame(
        {"timestamp": times, "cycle_stage": ["frost_development"] * len(times)}
    )
    record = {
        "boundaries": {
            "start_time": times[0].isoformat(),
            "end_time": times[-1].isoformat(),
            "heating_start": times[0].isoformat(),
            "stable_heating_start": None,
            "defrost_start": None,
            "defrost_end": None,
        }
    }

    targets = build_rgb_panel_targets(record, frame)

    assert targets[1]["enabled"] is False
    assert [target["target_time"] for target in targets[2:5]] == [
        times[2],
        times[4],
        times[6],
    ]
    assert targets[5]["enabled"] is False
    assert targets[6]["enabled"] is False


def test_select_rgb_panel_cells_chooses_nearest_image_per_camera() -> None:
    from frost_analysis.visualization import select_rgb_panel_cells

    target = pd.Timestamp("2026-07-14 10:00:10")
    images = pd.DataFrame(
        {
            "camera_role": ["camera_01", "camera_01", "camera_02"],
            "image_time": [
                target - pd.Timedelta(seconds=8),
                target + pd.Timedelta(seconds=3),
                target,
            ],
            "path": [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")],
        }
    )
    targets = [
        {"label": "Start", "target_time": target, "enabled": True},
        {"label": "Recovery End", "target_time": None, "enabled": False},
    ]

    cells = select_rgb_panel_cells(images, targets, ["camera_01", "camera_02", "camera_03"])

    assert cells["camera_01"] == [Path("b.jpg"), None]
    assert cells["camera_02"] == [Path("c.jpg"), None]
    assert cells["camera_03"] == [None, None]


def test_select_rgb_panel_cells_requires_image_within_two_minutes() -> None:
    from frost_analysis.visualization import select_rgb_panel_cells

    target = pd.Timestamp("2026-07-14 10:10:00")
    images = pd.DataFrame(
        {
            "camera_role": ["before_edge", "after_edge", "before_far", "after_far"],
            "image_time": [
                target - pd.Timedelta(minutes=2),
                target + pd.Timedelta(minutes=2),
                target - pd.Timedelta(seconds=121),
                target + pd.Timedelta(seconds=121),
            ],
            "path": [Path(f"{name}.jpg") for name in ("a", "b", "c", "d")],
        }
    )
    targets = [{"label": "Frost 50%", "target_time": target, "enabled": True}]

    cells = select_rgb_panel_cells(
        images,
        targets,
        ["before_edge", "after_edge", "before_far", "after_far"],
    )

    assert cells == {
        "before_edge": [Path("a.jpg")],
        "after_edge": [Path("b.jpg")],
        "before_far": [None],
        "after_far": [None],
    }


def test_render_rgb_panel_writes_review_png_with_role_order(tmp_path: Path) -> None:
    from frost_analysis.dataset_images import build_rgb_coverage_intervals
    from frost_analysis.visualization import render_rgb_panel

    times = pd.date_range("2026-07-14 10:00:00", periods=9, freq="10s")
    frame = pd.DataFrame(
        {"timestamp": times, "cycle_stage": ["frost_development"] * len(times)}
    )
    rows = []
    for camera, color in (("camera_02", (1.0, 0.0, 0.0)), ("camera_01", (0.0, 1.0, 0.0))):
        camera_dir = tmp_path / camera
        camera_dir.mkdir()
        path = camera_dir / "frame.jpg"
        plt.imsave(path, np.full((6, 10, 3), color))
        rows.append(
            {
                "camera_role": {"camera_01": "front", "camera_02": "left"}[camera],
                "image_time": times[4],
                "path": path,
            }
        )
    images = pd.DataFrame(rows)
    intervals = {
        row["camera_role"]: build_rgb_coverage_intervals(
            times[0], times[-1], pd.Series([times[4]]), max_image_gap_seconds=20
        )
        for row in rows
    }
    output = tmp_path / "panel.png"

    render_rgb_panel(
        {"cycle_name": "frost_cycle_000001", "status": "valid", "boundaries": {}},
        frame,
        images,
        intervals,
        ["front", "left", "top"],
        output,
    )

    assert output.is_file()
    rendered = plt.imread(output)
    assert rendered.shape[0] > 100
    assert rendered.shape[1] > rendered.shape[0]


def test_rgb_panel_available_segments_use_stage_colors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis.visualization import render_rgb_panel

    times = pd.date_range("2026-07-14 10:00:00", periods=6, freq="10s")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "cycle_stage": [
                "recovery",
                "recovery",
                "frost_development",
                "frost_development",
                "defrost",
                "defrost",
            ],
        }
    )
    images = pd.DataFrame(columns=["camera_role", "image_time", "path"])
    intervals = {
        "front": {
            "available": [(times[0], times[-1] + pd.Timedelta(seconds=10))],
            "missing": [],
        }
    }
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda _figure: None)

    render_rgb_panel(
        {"cycle_name": "frost_cycle_000001", "status": "valid", "boundaries": {}},
        frame,
        images,
        intervals,
        ["front"],
        tmp_path / "panel.png",
    )

    figure = plt.gcf()
    coverage_axis = figure.axes[-1]
    solid_colors = {
        to_hex(patch.get_facecolor())
        for patch in coverage_axis.patches
        if patch.get_alpha() in (None, 1.0)
        and patch.get_facecolor()[-1] > 0
        and patch.get_width() > 0
    }
    original_close(figure)
    assert solid_colors == {"#78a6bc", "#f2a35e", "#70b184"}


def test_stage_ribbon_omits_labels_that_do_not_fit() -> None:
    from frost_analysis.visualization import _plot_stage_ribbon

    figure, axis = plt.subplots()
    _plot_stage_ribbon(
        axis,
        [
            ("recovery", 0.0, 20.0),
            ("frost_development", 20.0, 90.0),
            ("defrost_preparation", 90.0, 92.0),
            ("defrost", 92.0, 100.0),
        ],
    )

    assert {text.get_text() for text in axis.texts} == {
        "Recovery",
        "Frost Development",
        "Defrost",
    }
    plt.close(figure)


def test_match_decision_rgb_images_keeps_target_status_and_two_minute_limit(
    tmp_path: Path,
) -> None:
    from frost_analysis.visualization import match_decision_rgb_images

    target = pd.Timestamp("2026-07-14 10:00:10")
    metadata = pd.DataFrame(
        {
            "camera_role": ["front", "front"],
            "file_name": ["near.jpg", "far.jpg"],
            "image_time": [target + pd.Timedelta(seconds=8), target + pd.Timedelta(minutes=3)],
        }
    )
    images = metadata.iloc[[0]].copy()
    near_path = tmp_path / "near.jpg"
    plt.imsave(near_path, np.ones((4, 4, 3)))
    images["path"] = [near_path]

    result = match_decision_rgb_images(
        metadata,
        images,
        {"rb": target, "optimal": target + pd.Timedelta(minutes=6)},
    )

    assert result.loc[result["target_type"].eq("rb"), "status"].item() == "matched"
    assert result.loc[result["target_type"].eq("rb"), "available"].item() is True
    assert result.loc[result["target_type"].eq("rb"), "offset_seconds"].item() == pytest.approx(8.0)
    assert result.loc[result["target_type"].eq("optimal"), "status"].item() == "offset_exceeds_2min"
    assert result.loc[result["target_type"].eq("optimal"), "available"].item() is False


def test_render_decision_publication_has_rgb_and_three_aligned_time_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis.visualization import render_decision_publication

    times = pd.date_range("2026-07-14 10:00:00", periods=6, freq="1min")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "cycle_stage": ["recovery", *(["frost_development"] * 4), "defrost"],
            "cop": [2.2, 2.1, 2.0, 1.9, 1.8, 1.0],
            "water_in_temperature": [30.0] * 6,
            "water_out_temperature": [35.0] * 6,
            "water_temperature_setpoint": [35.0] * 6,
        }
    )
    rgb = {
        "rb": {
            "target_type": "rb",
            "target_time": times[2],
            "image_time": times[2] + pd.Timedelta(seconds=8),
            "offset_seconds": 8.0,
            "image_path": str(tmp_path / "rb.jpg"),
            "available": True,
            "status": "matched",
        },
        "optimal": {
            "target_type": "optimal",
            "target_time": times[3],
            "image_time": times[3] - pd.Timedelta(seconds=6),
            "offset_seconds": 6.0,
            "image_path": str(tmp_path / "optimal.jpg"),
            "available": True,
            "status": "matched",
        },
    }
    plt.imsave(rgb["rb"]["image_path"], np.ones((8, 8, 3)))
    plt.imsave(rgb["optimal"]["image_path"], np.ones((8, 8, 3)))
    cost = pd.DataFrame(
        {
            "candidate_time": times[1:5],
            "inverse_cop": [0.5, 0.45, 0.46, 0.5],
            "optimization_eligible": [True] * 4,
            "relative_regret": [0.1, 0.0, 0.02, 0.1],
        }
    )
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda _figure: None)

    render_decision_publication(
        frame,
        {
            "cycle_name": "cycle",
            "status": "valid",
            "boundaries": {"stable_heating_start": times[1].isoformat()},
        },
        cost,
        rgb,
        tmp_path / "publication.svg",
    )

    figure = plt.gcf()
    assert len(figure.axes) == 5
    assert {axis.get_ylabel() for axis in figure.axes[2:]} >= {
        "COP [-]",
        "Water temperature [degC]",
        "Cycle inverse COP [-]",
    }
    for axis in figure.axes[2:]:
        assert {line.get_color() for line in axis.lines} >= {"#2E7D5B", "#E28E2C"}
    assert figure.axes[0].get_title(loc="left").startswith("RB trigger")
    assert figure.axes[1].get_title() == ""
    original_close(figure)


def test_cost_curve_optimal_time_includes_observed_frost_right_boundary() -> None:
    from frost_analysis.visualization import cost_curve_optimal_time

    origin = pd.Timestamp("2026-01-01 00:00")
    curve = pd.DataFrame(
        {
            "candidate_time": [origin + pd.Timedelta(minutes=9), origin + pd.Timedelta(minutes=10)],
            "inverse_cop": [0.5, 0.4],
            "optimization_eligible": [True, True],
        }
    )

    assert cost_curve_optimal_time(
        curve, origin, [("frost_development", 1.0, 10.0)]
    ) == origin + pd.Timedelta(minutes=10)


def test_unit_cost_panel_uses_full_candidate_domain_past_processed_frost_span() -> None:
    from frost_analysis.visualization import _plot_cost_panel

    origin = pd.Timestamp("2026-01-01 00:00")
    curve = pd.DataFrame(
        {
            "candidate_time": [origin + pd.Timedelta(minutes=9), origin + pd.Timedelta(minutes=10)],
            "inverse_cop": [0.5, 0.4],
            "optimization_eligible": [True, True],
            "relative_regret": [0.25, 0.0],
        }
    )
    figure, axis = plt.subplots()

    _plot_cost_panel(
        axis,
        curve,
        origin,
        [("frost_development", 1.0, 9.5)],
        full_candidate_domain=True,
    )

    minimum = next(line for line in axis.lines if line.get_label() == "Minimum")
    assert list(minimum.get_xdata()) == [10.0, 10.0]
    plt.close(figure)
