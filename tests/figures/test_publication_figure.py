from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex


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
    from frost_analysis.visualization import _stage_spans

    frame, minutes = _panel_frame()

    assert _stage_spans(frame, minutes)[0] == ("recovery", 0.0, 1.0)


def test_cycle_panel_legend_labels_are_arranged_horizontally() -> None:
    from frost_analysis.visualization import _plot_cycle_panel

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
    from frost_analysis.visualization import _display_label

    assert _display_label("environment_relative_humidity") == "Relative Humidity"
    assert _display_label("ambient_temperature") == "Ambient Temperature"


def test_publication_combines_heating_and_evaporator_capacity() -> None:
    from frost_analysis.visualization import _COLORS, _PANELS

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
    from frost_analysis.visualization import _STAGE_COLORS, _plot_cycle_panel

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
    from frost_analysis.visualization import _plot_cycle_panel

    frame = pd.DataFrame(
        {
            "cop": [30.0, 8.0, 3.8, 3.4, 3.0, 4.2],
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
        ("cop",),
        "COP [-]",
        [("recovery", 0.0, 2.0), ("frost_development", 2.0, 5.0), ("defrost", 5.0, 6.0)],
        [],
        np.nan,
        np.nan,
    )

    assert axis.child_axes == []
    assert axis.get_ylim()[1] < 10.0
    plt.close(figure)
