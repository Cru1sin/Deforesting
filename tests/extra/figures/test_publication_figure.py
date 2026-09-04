from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.patches import FancyArrowPatch


def _panel_frame() -> tuple[pd.DataFrame, pd.Series[float]]:
    frame = pd.DataFrame(
        {
            "ambient_temperature": [5.0, 4.0, 3.0, 2.0],
            "coil_temperature": [1.0, 0.0, -1.0, -2.0],
            "evaporating_temperature": [0.0, -1.0, -2.0, -3.0],
            "cycle_stage": [
                "recovery",
                "frost_development",
                "frost_development",
                "defrost",
            ],
        }
    )
    return frame, pd.Series([0.0, 1.0, 2.0, 3.0])


def test_stage_spans_keeps_a_single_sample_first_stage() -> None:
    from frost_analysis.figures.visualization import _stage_spans

    frame, minutes = _panel_frame()

    assert _stage_spans(frame, minutes)[0] == ("recovery", 0.0, 1.0)


def test_cycle_panel_legend_labels_are_arranged_horizontally() -> None:
    from frost_analysis.figures.visualization import _plot_cycle_panel

    frame, minutes = _panel_frame()
    figure, axis = plt.subplots()
    _plot_cycle_panel(
        axis,
        frame,
        minutes,
        ("ambient_temperature", "coil_temperature", "evaporating_temperature"),
        "Temperature [degC]",
        [("recovery", 0.0, 1.0), ("frost_development", 1.0, 3.0), ("defrost", 3.0, 4.0)],
        [],
        np.nan,
        np.nan,
    )

    assert axis.get_legend()._ncols == 3
    plt.close(figure)


def test_publication_display_label_hides_machine_field_formatting() -> None:
    from frost_analysis.figures.visualization import _display_label

    assert _display_label("environment_relative_humidity") == "Relative Humidity"
    assert _display_label("ambient_temperature") == "Ambient Temperature"


def test_publication_combines_heating_and_evaporator_capacity() -> None:
    from frost_analysis.figures.visualization import _COLORS, _PANELS

    assert (
        (
            "heating_capacity",
            "evaporator_capacity",
            "compressor_power",
            "power_total",
        ),
        "Capacity / power [kW]",
    ) in _PANELS
    assert (("heating_capacity",), "Heating capacity [kW]") not in _PANELS
    assert (("evaporator_capacity",), "Evaporator capacity [kW]") not in _PANELS
    assert "evaporator_capacity" in _COLORS
    assert "compressor_power" in _COLORS
    assert "power_total" in _COLORS


def test_cycle_panel_uses_stage_colors_and_hatched_missing_state() -> None:
    from frost_analysis.figures.visualization import _STAGE_COLORS, _plot_cycle_panel

    frame, minutes = _panel_frame()
    figure, axis = plt.subplots()
    _plot_cycle_panel(
        axis,
        frame,
        minutes,
        ("ambient_temperature",),
        "Temperature [degC]",
        [
            ("recovery", 0.0, 1.0),
            ("frost_development", 1.0, 3.0),
            ("defrost_preparation", 3.0, 3.5),
            ("defrost", 3.5, 4.0),
        ],
        [(1.4, 1.7)],
        np.nan,
        np.nan,
    )

    stage_colors = [to_hex(patch.get_facecolor()) for patch in axis.patches[:4]]
    assert stage_colors == [
        _STAGE_COLORS["recovery"].lower(),
        _STAGE_COLORS["frost_development"].lower(),
        "#76528f",
        _STAGE_COLORS["defrost"].lower(),
    ]
    assert axis.patches[2].get_alpha() == 0.20
    assert axis.patches[4].get_hatch() == "////"
    plt.close(figure)


def test_cop_panel_focuses_normal_range_without_inset() -> None:
    from frost_analysis.figures.visualization import _plot_cycle_panel

    frame = pd.DataFrame(
        {
            "cop": [30.0, 8.0, 3.8, 3.4, 3.0, 4.2],
            "water_flow": [1.0] * 6,
            "power_total": [1.161] * 6,
            "water_in_temperature": [0.0] * 6,
            "water_out_temperature": [30.0, 8.0, 3.8, 3.4, 3.0, -15.0],
            "cycle_stage": [
                "recovery",
                "recovery",
                "frost_development",
                "frost_development",
                "frost_development",
                "defrost",
            ],
        }
    )
    minutes = pd.Series(np.arange(len(frame), dtype=float))
    figure, axis = plt.subplots()
    _plot_cycle_panel(
        axis,
        frame,
        minutes,
        ("cop", "water_cop"),
        "COP [-]",
        [("recovery", 0.0, 2.0), ("frost_development", 2.0, 5.0), ("defrost", 5.0, 6.0)],
        [],
        np.nan,
        np.nan,
    )

    assert axis.child_axes == []
    assert axis.get_ylim()[0] > 0.0
    assert axis.get_ylim()[1] < 10.0
    plt.close(figure)


def test_cost_panel_display_extension_does_not_change_formal_minimum() -> None:
    from frost_analysis.figures.visualization import _plot_cost_panel

    start = pd.Timestamp("2026-01-01")
    curve = pd.DataFrame(
        {
            "candidate_time": start + pd.to_timedelta([1, 2, 3], unit="min"),
            "inverse_cop": [0.5, 0.4, np.nan],
            "optimization_eligible": [True, True, False],
            "model_supported": [True, True, False],
            "display_only_inverse_cop": [np.nan, np.nan, 0.1],
        }
    )
    figure, axis = plt.subplots()

    _plot_cost_panel(
        axis,
        curve,
        start,
        [("frost_development", 0.0, 4.0)],
        full_candidate_domain=True,
        display_metric="display_only_inverse_cop",
        minimum_label="Diagnostic/raw minimum",
    )

    raw_minimum = next(line for line in axis.lines if line.get_label() == "Diagnostic/raw minimum")
    extension = next(
        line
        for line in axis.lines
        if line.get_label() == "Unsupported model extension, display only"
    )
    assert list(raw_minimum.get_xdata()) == [2.0, 2.0]
    assert np.allclose(extension.get_ydata(), [np.nan, np.nan, 0.1], equal_nan=True)
    plt.close(figure)


def test_cost_panel_preserves_native_maximum_direction_for_v27_objective() -> None:
    from frost_analysis.figures.visualization import _plot_cost_panel

    start = pd.Timestamp("2026-01-01")
    curve = pd.DataFrame(
        {
            "candidate_time": start + pd.to_timedelta([1, 2, 3], unit="min"),
            "objective_value": [0.5, 0.8, 0.9],
            "optimization_direction": ["max"] * 3,
            "optimization_eligible": [True, True, False],
            "model_supported": [True, True, False],
            "relative_optimality_gap": [0.375, 0.0, np.nan],
            "display_only_objective": [np.nan, np.nan, 0.9],
        }
    )
    figure, axis = plt.subplots()

    _plot_cost_panel(
        axis,
        curve,
        start,
        [("frost_development", 0.0, 4.0)],
        cost_label="Evaporator cycle capacity efficiency [-]",
        full_candidate_domain=True,
        display_metric="display_only_objective",
        minimum_label="Maximum",
    )

    maximum = next(line for line in axis.lines if line.get_label() == "Maximum")
    outside = next(
        line
        for line in axis.lines
        if line.get_label() == "Unsupported model extension, display only"
    )
    assert list(maximum.get_xdata()) == [2.0, 2.0]
    assert np.allclose(outside.get_ydata(), [np.nan, np.nan, 0.9], equal_nan=True)
    assert axis.get_ylabel() == "Evaporator cycle capacity efficiency [-]"
    plt.close(figure)


def test_cost_panel_shades_connected_one_and_five_percent_regions() -> None:
    from frost_analysis.figures.visualization import _plot_cost_panel

    start = pd.Timestamp("2026-01-01")
    curve = pd.DataFrame(
        {
            "candidate_time": start + pd.to_timedelta([1, 2, 3, 4], unit="min"),
            "inverse_cop": [1.0, 1.005, 1.03, 1.07],
            "optimization_eligible": True,
        }
    )
    figure, axis = plt.subplots()

    _plot_cost_panel(
        axis,
        curve,
        start,
        [("frost_development", 0.0, 5.0)],
        full_candidate_domain=True,
    )

    regions = {patch.get_label(): patch for patch in axis.patches if patch.get_label()}
    assert {
        "1% connected near-optimal region",
        "5% connected near-optimal region",
    } <= set(regions)
    assert regions["1% connected near-optimal region"].get_alpha() > regions[
        "5% connected near-optimal region"
    ].get_alpha()
    plt.close(figure)


def test_cost_panel_does_not_mark_unknown_model_support_as_unsupported() -> None:
    from frost_analysis.figures.visualization import _plot_cost_panel

    start = pd.Timestamp("2026-01-01")
    curve = pd.DataFrame(
        {
            "candidate_time": start + pd.to_timedelta([1, 2, 3], unit="min"),
            "inverse_cop": [0.5, 0.6, 0.4],
            "optimization_eligible": [True, True, True],
            "model_supported": pd.Series([True, False, np.nan], dtype=object),
        }
    )
    figure, axis = plt.subplots()

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        _plot_cost_panel(
            axis,
            curve,
            start,
            [("frost_development", 0.0, 4.0)],
            full_candidate_domain=True,
        )

    unsupported = next(
        item for item in axis.collections if item.get_label() == "Outside empirical-model support"
    )
    assert unsupported.get_offsets().tolist() == [[2.0, 0.6]]
    assert any("support unknown" in text.get_text() for text in axis.texts)
    plt.close(figure)


def test_parallel_objective_panel_uses_three_raw_axes_and_requested_line_styles() -> None:
    from frost_analysis.figures.visualization import _plot_parallel_objectives_panel

    start = pd.Timestamp("2026-01-01")
    curves = {
        metric: pd.DataFrame(
            {
                "candidate_time": pd.date_range(start, periods=3, freq="min"),
                "objective_value": values,
                "optimization_eligible": True,
            }
        )
        for metric, values in {
            "C": [2.8, 3.0, 2.9],
            "H": [8.0, 8.2, 8.1],
            "O": [5.0, 4.8, 4.5],
        }.items()
    }
    figure, axis = plt.subplots()

    axes = _plot_parallel_objectives_panel(axis, curves, start)

    assert len(axes) == 3
    assert [item.get_ylabel() for item in axes] == [
        "cycle COP",
        "Heating rate [kW]",
        "Evaporator capacity [kW]",
    ]
    assert [item.lines[0].get_linestyle() for item in axes] == ["-", "-", "--"]
    assert axes[2].spines["right"].get_position() == ("axes", 1.09)
    assert [item.yaxis.label.get_color() for item in axes] == ["black"] * 3
    assert [item.yaxis.label.get_fontsize() for item in axes] == [8.0] * 3
    assert axes[1].yaxis.labelpad == axes[2].yaxis.labelpad == 8

    for item, expected_ticks in zip(
        axes,
        ([2.4, 2.6, 2.8, 3.0], [6.5, 7.0, 7.5, 8.0], [4.0, 4.5, 5.0]),
        strict=True,
    ):
        assert np.allclose(item.get_yticks(), expected_ticks)
        assert np.allclose(np.diff(item.get_yticks()), np.diff(item.get_yticks())[0])
    assert len({round(item.get_position().height, 8) for item in axes}) == 1
    plt.close(figure)


def test_parallel_table_promotes_only_physically_valid_extrapolation() -> None:
    from frost_analysis.figures.visualization import _parallel_objective_table

    start = pd.Timestamp("2026-01-01")
    times = pd.date_range(start, periods=4, freq="min")
    curves = {
        metric: pd.DataFrame(
            {
                "candidate_time": times,
                "objective_value": values,
                "optimization_eligible": [True, True, False, False],
                "model_supported": [True, True, False, False],
                "physical_valid": [True, True, True, False],
                "measurement_eligible": True,
            }
        )
        for metric, values in {
            "C": [2.0, 4.0, 8.0, 9.0],
            "H": [3.0, 5.0, 7.0, 10.0],
            "O": [4.0, 6.0, 9.0, 11.0],
        }.items()
    }

    values = _parallel_objective_table(curves)

    assert values["C_native_eligible"].tolist() == [True, True, False, False]
    assert values["C_model_supported"].tolist() == [True, True, False, False]
    assert values["C_eligible"].tolist() == [True, True, True, False]


def test_normalized_objective_panel_scales_by_full_decision_domain_maximum() -> None:
    from frost_analysis.figures.visualization import _plot_normalized_objectives_panel

    start = pd.Timestamp("2026-01-01")
    curves = {
        metric: pd.DataFrame(
            {
                "candidate_time": pd.date_range(start, periods=3, freq="min"),
                "objective_value": values,
                "optimization_eligible": eligible,
                "model_supported": supported,
                "physical_valid": True,
                "measurement_eligible": True,
            }
        )
        for metric, values, eligible, supported in (
            ("C", [2.0, 4.0, 8.0], [True, True, False], [True, True, False]),
            ("H", [6.0, 3.0, 1.5], [True, True, True], [True, True, True]),
            ("O", [5.0, 10.0, 2.5], [True, True, True], [True, True, True]),
        )
    }
    figure, axis = plt.subplots()

    _plot_normalized_objectives_panel(axis, curves, start)

    assert axis.get_ylabel() == "Relative to best performance [%]"
    assert axis.yaxis.label.get_color() == "black"
    assert axis.yaxis.label.get_fontsize() == 8.0
    assert axis.yaxis.labelpad == 8
    assert [text.get_text() for text in axis.get_legend().texts] == [
        "cycle COP",
        "Heating rate",
        "Evaporator capacity",
    ]
    assert axis.get_legend()._loc == 3
    assert np.allclose(axis.get_legend().get_bbox_to_anchor()._bbox.bounds, [0, 1.01, 0, 0])
    assert np.allclose(axis.lines[0].get_ydata(), [25, 50, 100], equal_nan=True)
    assert np.allclose(axis.lines[1].get_ydata(), [100, 50, 25], equal_nan=True)
    assert np.allclose(axis.lines[2].get_ydata(), [50, 100, 25], equal_nan=True)
    assert np.allclose(axis.get_ylim(), [90, 100.8])
    thresholds = {
        float(np.asarray(line.get_ydata())[0])
        for line in axis.lines
        if len(np.asarray(line.get_ydata())) == 2
    }
    assert {95.0, 98.0, 99.0, 100.0} <= thresholds
    assert {text.get_text() for text in axis.texts} >= {"1%", "2%", "5%"}
    plt.close(figure)


def test_raw_objective_panel_includes_unsupported_extension_in_display_range() -> None:
    from frost_analysis.figures.visualization import _plot_parallel_objectives_panel

    start = pd.Timestamp("2026-01-01")
    curves = {
        metric: pd.DataFrame(
            {
                "candidate_time": pd.date_range(start, periods=3, freq="min"),
                "objective_value": values,
                "optimization_eligible": [True, True, False],
            }
        )
        for metric, values in {"C": [2.0, 4.0, 8.0], "H": [4, 6, 9], "O": [3, 5, 7]}.items()
    }
    figure, axis = plt.subplots()

    axes = _plot_parallel_objectives_panel(axis, curves, start)

    assert np.allclose([item.get_ylim()[1] for item in axes], np.array([8, 9, 7]) * 1.02)
    plt.close(figure)


def test_ch_pareto_panel_maps_each_point_to_o_with_a_colorbar() -> None:
    from frost_analysis.figures.visualization import _plot_ch_pareto_panel

    start = pd.Timestamp("2026-01-01")
    times = pd.date_range(start, periods=5, freq="min")
    curves = {
        metric: pd.DataFrame(
            {
                "candidate_time": times,
                "objective_value": values,
                "optimization_eligible": True,
            }
        )
        for metric, values in {
            "C": [2.97, 2.99, 3.0, 1.0, 2.86],
            "H": [8.2, 8.1, 8.0, 1.0, 8.3],
            "O": [5.0, 4.8, 4.6, 0.0, 4.4],
        }.items()
    }
    figure, axis = plt.subplots()

    plotted = _plot_ch_pareto_panel(axis, curves, start, guardrail=0.05)

    assert len(plotted) == 5
    assert plotted["O"].tolist() == [5.0, 4.8, 4.6, 0.0, 4.4]
    assert plotted["pareto"].tolist() == [True, True, True, False, True]
    assert plotted.loc[plotted["pareto_latest"], "minutes"].tolist() == [4.0]
    assert plotted.loc[plotted["pareto_knee"], "minutes"].tolist() == [0.0]
    assert axis.get_xlabel() == "cycle COP"
    assert axis.get_ylabel() == "Heating rate [kW]"
    assert axis.child_axes[0].get_ylabel() == "Evaporator capacity [kW]"
    assert axis.xaxis.label.get_color() == "black"
    assert axis.yaxis.label.get_color() == "black"
    assert axis.xaxis.label.get_fontsize() == 8.0
    assert axis.yaxis.label.get_fontsize() == 8.0
    assert axis.get_box_aspect() == 1
    candidates = next(
        collection for collection in axis.collections if collection.get_array() is not None
    )
    assert len(candidates.get_offsets()) == len(plotted)
    assert candidates.norm.vmin == 4.4
    assert candidates.norm.vmax == 5.0
    assert candidates.get_sizes().tolist() == [24]
    assert candidates.colorbar.extend == "neither"
    pareto = next(
        collection
        for collection in axis.collections
        if collection.get_label() == "Pareto front"
    )
    selected = next(
        collection for collection in axis.collections if collection.get_label() == "Selected"
    )
    assert pareto.get_linewidths().max() <= 0.6
    assert pareto.get_facecolors().size == 0
    assert pareto.get_sizes().tolist() == [56]
    assert selected.get_facecolors().size == 0
    assert selected.get_sizes().tolist() == [80]
    assert np.allclose(selected.get_offsets(), [[2.97, 8.2]])
    assert {text.get_text() for text in axis.get_legend().texts} == {"Pareto front", "Selected"}
    assert axis.get_title(loc="left") == (
        "Local Pareto view · local O scale · full range 0–4 min"
    )
    assert not any(text.get_text().startswith("selected") for text in axis.texts)
    time_labels = [text for text in axis.texts if text.get_text() in {"0", "1", "2", "4"}]
    assert {text.get_text() for text in time_labels} == {"0", "1", "2", "4"}
    leader_lines = [
        patch for patch in axis.patches if isinstance(patch, FancyArrowPatch)
    ]
    assert len(leader_lines) == len(time_labels)
    assert all(line.get_linewidth() == 0.35 for line in leader_lines)
    assert not any(text.get_text() == "increasing defrost delay" for text in axis.texts)
    assert len(axis.lines) == 1
    assert np.allclose(axis.lines[0].get_xdata(), [2.97, 2.99, 3.0, 1.0, 2.86])
    assert np.allclose(axis.lines[0].get_ydata(), [8.2, 8.1, 8.0, 1.0, 8.3])
    assert axis.lines[0].get_color() == "#D1D5DB"
    plt.close(figure)


def test_ch_pareto_panel_labels_each_displayed_minute_once() -> None:
    from frost_analysis.figures.visualization import _plot_ch_pareto_panel

    start = pd.Timestamp("2026-01-01")
    times = pd.date_range(start, periods=7, freq="10s")
    curves = {
        metric: pd.DataFrame(
            {
                "candidate_time": times,
                "objective_value": values,
                "optimization_eligible": True,
            }
        )
        for metric, values in {
            "C": np.arange(1.0, 8.0),
            "H": np.arange(7.0, 0.0, -1.0),
            "O": np.linspace(3.0, 3.6, 7),
        }.items()
    }
    figure, axis = plt.subplots()

    _plot_ch_pareto_panel(axis, curves, start)

    labels = [text.get_text() for text in axis.texts if text.get_text() in {"0", "1"}]
    assert labels.count("0") == 1
    assert labels.count("1") == 1
    plt.close(figure)


def test_ch_pareto_table_selects_normalized_chord_knee_not_latest() -> None:
    from frost_analysis.figures.visualization import ch_pareto_table

    start = pd.Timestamp("2026-01-01")
    times = pd.date_range(start, periods=4, freq="min")
    curves = {
        metric: pd.DataFrame(
            {
                "candidate_time": times,
                "objective_value": values,
                "optimization_eligible": True,
            }
        )
        for metric, values in {
            "C": [1.0, 1.4, 1.7, 2.0],
            "H": [2.0, 1.95, 1.6, 1.0],
            "O": [5.0, 4.9, 4.8, 4.7],
        }.items()
    }

    values = ch_pareto_table(curves, start)

    assert values.loc[values["pareto_knee"], "candidate_time"].tolist() == [times[1]]
    assert values.loc[values["pareto_latest"], "candidate_time"].tolist() == [times[3]]
    assert values.loc[values["pareto_knee"], "pareto_knee_method"].item() == (
        "normalized_chord_distance"
    )


def test_ch_pareto_panel_keeps_full_front_and_rb_point_inside_view() -> None:
    from frost_analysis.figures.visualization import _plot_ch_pareto_panel

    start = pd.Timestamp("2026-01-01")
    times = pd.date_range(start, periods=4, freq="min")
    curves = {
        metric: pd.DataFrame(
            {
                "candidate_time": times,
                "objective_value": values,
                "optimization_eligible": True,
            }
        )
        for metric, values in {
            "C": [3.0, 2.99, 2.8, 2.7],
            "H": [7.0, 8.1, 8.3, 6.5],
            "O": [5.0, 4.9, 4.8, 4.7],
        }.items()
    }
    figure, axis = plt.subplots()

    plotted = _plot_ch_pareto_panel(
        axis,
        curves,
        start,
        rb_time=times[1],
    )

    front = plotted.loc[plotted["pareto"]]
    rb = next(item for item in axis.collections if item.get_label() == "RB trigger")
    assert axis.get_xlim()[0] < front["C"].min() < front["C"].max() < axis.get_xlim()[1]
    assert axis.get_ylim()[0] < front["H"].min() < front["H"].max() < axis.get_ylim()[1]
    assert np.allclose(rb.get_offsets(), [[2.99, 8.1]])
    assert "RB 1" in {text.get_text() for text in axis.texts}
    assert axis.get_legend()._ncols == 3
    plt.close(figure)


def test_decision_markers_use_target_time_when_rgb_is_missing() -> None:
    from frost_analysis.figures.visualization import _plot_decision_markers

    start = pd.Timestamp("2026-01-01")
    figure, axes = plt.subplots(2)
    decisions = {
        "rb": {"target_time": start + pd.Timedelta(minutes=2), "available": False},
        "optimal": {"target_time": start + pd.Timedelta(minutes=4), "available": False},
    }

    _plot_decision_markers(list(axes), decisions, start, "Pareto-latest")

    assert [axis.lines[0].get_xdata()[0] for axis in axes] == [2.0, 2.0]
    assert [axis.lines[1].get_xdata()[0] for axis in axes] == [4.0, 4.0]
    plt.close(figure)
