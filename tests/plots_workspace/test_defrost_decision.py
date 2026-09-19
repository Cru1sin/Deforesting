from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from plots.publication import match_decision_rgb_images

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

def test_effective_publication_merges_cop_axes_and_interpolates_display(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    from plots.publication import render_decision_publication

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
        }
    )
    captured = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *_a, **_k: captured.append(figure))
    render_decision_publication(
        frame,
        {},
        curve,
        {"rb": {"target_time": times[3]}, "optimal": {"target_time": times[2]}},
        tmp_path / "effective.png",
        parallel_curve=curve,
    )
    figure = captured[0]
    assert len(figure.axes) == 6  # Two front images, three panels, one right axis.
    image, _, cop, water, absolute, relative = figure.axes
    assert image.get_position().height > 1.5 * cop.get_position().height
    assert np.isnan(cop.lines[0].get_ydata()[1])  # Missing observations are not measured COP.
    assert water.get_ylabel() == "Water temperature [degC]"
    assert absolute.get_ylabel() == "Effective cycle COP [-]"
    assert absolute.lines[0].get_linestyle() == "--"
    assert absolute.lines[0].get_alpha() is None
    assert absolute.lines[1].get_ydata()[1] == pytest.approx(2.1)
    assert np.isnan(absolute.lines[1].get_ydata()[-1])
    assert relative.lines[1].get_ydata()[2] == pytest.approx(100)
    assert relative.get_ylim() == (90, 100.8)
    assert relative.spines["right"].get_visible()
    assert relative.spines["right"].get_edgecolor() == (0, 0, 0, 1)
    assert {text.get_text() for text in relative.texts} >= {"1%", "2%", "5%"}
    legend = {text.get_text() for text in absolute.get_legend().get_texts()}
    assert legend >= {
        "RB defrost trigger",
        "COP-optimal defrost trigger",
        "Assumed extrapolation",
        "Normalized cycle COP",
    }
    assert pd.isna(curve.cycle_cop.iloc[1])

def test_chen_cycle_figure_reuses_effective_cop_panel_and_keeps_missing_probabilities():
    from plots.publication import _effective_cop_probability_figure

    origin = pd.Timestamp("2026-01-01 00:00:00")
    times = pd.date_range(origin, periods=6, freq="30s")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "cycle_stage": [
                "recovery",
                "recovery",
                "frost_development",
                "frost_development",
                "frost_development",
                "defrost_preparation",
            ],
        }
    )
    curve = pd.DataFrame(
        {
            "candidate_defrost_time": times,
            "cycle_cop": [0.0, 1.0, 2.0, 2.2, 2.1, 2.0],
            "cycle_cop_eligible": [False, False, True, True, True, False],
            "t_star": times[3],
            "t_RB": times[4],
            "cycle_status": "identified_curve",
            "cycle_cop_basin_1pct_start": times[3],
            "cycle_cop_basin_1pct_end": times[3],
            "cycle_cop_basin_2pct_start": times[3],
            "cycle_cop_basin_2pct_end": times[4],
            "cycle_cop_basin_5pct_start": times[2],
            "cycle_cop_basin_5pct_end": times[4],
        }
    )
    trace = pd.DataFrame(
        {
            "candidate_defrost_time": times,
            "score": [0.1, np.nan, 0.6, 0.8, 0.7, 0.9],
            "threshold": 0.5,
        }
    )
    metric = pd.Series({"trigger_time": times[3], "status": "scored"})

    figure = _effective_cop_probability_figure(
        frame, curve, trace, metric, "frost_cycle_000001"
    )

    effective_axis = next(
        axis for axis in figure.axes if axis.get_ylabel() == "Effective cycle COP [-]"
    )
    probability_axis = next(
        axis for axis in figure.axes if axis.get_ylabel() == "Trigger probability"
    )
    assert effective_axis.get_shared_x_axes().joined(effective_axis, probability_axis)
    assert effective_axis.patches and probability_axis.patches
    labels = {
        line.get_label()
        for axis in (effective_axis, probability_axis)
        for line in axis.lines
    }
    assert {
        "COP optimum",
        "RB trigger",
        "2/3 confirmed trigger",
        "Positive threshold (0.5)",
    } <= labels
    probability = next(
        line for line in probability_axis.lines if line.get_label() == "Positive-class probability"
    )
    assert np.isnan(probability.get_ydata()[1])
    assert "(a)" in {text.get_text() for text in effective_axis.texts}
    assert "(b)" in {text.get_text() for text in probability_axis.texts}

def test_publish_effective_decision_preserves_history_and_manual_status(tmp_path):
    from dataset_tools.dataset_maintenance import publish_effective_decision_assets

    dataset = tmp_path / "dataset"
    (dataset / "cycles").mkdir(parents=True)
    name = "frost_cycle_000001"
    old = dataset / "cycles" / f"{name}_pareto_decision.png"
    old.write_bytes(b"historical figure")
    record = {
        "cycle_name": name,
        "status": "valid",
        "assets": {"pareto_decision": f"cycles/{old.name}"},
        "pareto_decision": {"status": "old"},
    }
    (dataset / "cycle_catalog.json").write_text(json.dumps({"cycles": [record]}))
    run = tmp_path / "run"
    run.mkdir()
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
    assert "pareto_decision" not in current["assets"]
    assert current["effective_cop"]["preparation_heat"] == "zero"
    published = dataset / current["assets"]["effective_cop"]
    assert published.read_bytes() == b"effective figure"
    assert (figures / "previous_dataset_decisions" / old.name).read_bytes() == b"historical figure"
    assert not old.exists()

    table = pd.read_csv(run / "cycle_comparison.csv")
    table["preparation_heat"] = "include"
    table.to_csv(run / "cycle_comparison.csv", index=False)
    with pytest.raises(ValueError, match="zero preparation heat"):
        publish_effective_decision_assets(dataset, run, figures)
    assert published.name == f"{name}_effective_cop.png"
    assert published.read_bytes() == b"effective figure"
