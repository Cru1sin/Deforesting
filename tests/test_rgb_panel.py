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
