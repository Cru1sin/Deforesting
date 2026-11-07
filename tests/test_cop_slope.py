import numpy as np
import pandas as pd

from tests.cost_workspace.cop_slope import slope_windows


def test_frequency_coloured_state_cop_keeps_time_order_and_own_normalization(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from plots import frequency as publication

    times = pd.date_range("2026-01-01", periods=5, freq="10s")
    curves = pd.DataFrame(
        dict(
            cycle_name="cycle_1",
            cycle_progress=np.arange(5) / 10,
            elapsed_minutes=np.arange(5) / 6,
            image_time=times,
            frost_grade=[10, 20, 15, 30, 40],
            cycle_cop=[1, 2, 1.5, 3, 1.8],
            cycle_cop_eligible=[True, True, True, False, True],
            compressor_frequency=[70, 71, 70, 72, 73],
            relative_cop=[0.5, 1, 0.75, np.nan, 0.9],
            normalized_cop_percent=[50, 100, 75, np.nan, 90],
        )
    )
    cohort = pd.DataFrame(dict(cycle_name=["cycle_1"], experiment_id=["day1"]))
    captured = []

    def save(fig, *args, **kwargs):
        collection = next(c for c in fig.axes[0].collections if isinstance(c, LineCollection))
        captured.append(collection.get_segments())
        plt.close(fig)

    monkeypatch.setattr(publication, "save_publication", save)
    for y in ["cycle_cop", "normalized_cop_percent"]:
        publication._render_cop_slope_condition(
            tmp_path,
            None,
            cohort,
            curves,
            pd.DataFrame(),
            (70, 73),
            (1, 100),
            rise_markers="none",
            activity_only=False,
            cop_lower=0,
            output_name="test",
            activity_direction="both",
            x_column="frost_grade",
            y_column=y,
        )
    assert len(captured[0]) == 2
    np.testing.assert_allclose(captured[0][1], [[20, 2], [15, 1.5]])
    np.testing.assert_allclose(captured[1][1], [[20, 100], [15, 75]])


def test_secants_preserve_physical_time_and_do_not_bridge_invalid_candidates():
    t = np.arange(0.5, 40.51, 0.5)
    origin = pd.Timestamp("2026-01-01")
    curve = pd.DataFrame(
        dict(
            candidate_defrost_time=origin + pd.to_timedelta(t, unit="min"),
            heating_accounting_start=origin,
            cycle_cop=4 - (t - 20) ** 2 / 400,
            cycle_cop_eligible=True,
        )
    )
    windows = slope_windows(curve, 5)
    assert len(windows) == 8
    assert np.allclose(windows.slope_change.dropna(), -0.625)
    edges = np.arange(0.5, 40.51, 5)
    assert np.allclose(windows.slope_cop_per_minute, np.diff(4 - (edges - 20) ** 2 / 400) / 5)
    # One missing candidate removes the entire affected secant and breaks comparison.
    curve.loc[t == 10.5, "cycle_cop_eligible"] = False
    windows = slope_windows(curve, 5)
    assert not windows.midpoint_minutes.isin([8.0, 13.0]).any()
    assert windows.block.nunique() == 2
    assert windows.slope_change.isna().sum() == 2


def test_slope_direction_support_requires_both_windows_in_the_displayed_cop_range():
    from tests.cost_workspace.cop_slope import slope_direction_support

    curves = pd.DataFrame(
        dict(
            cycle_name="cycle",
            elapsed_minutes=np.arange(0.0, 31.0),
            cycle_progress=np.arange(0.0, 31.0) / 30,
            recovery_minutes=0.0,
            relative_cop=np.linspace(0.4, 1.0, 31),
        )
    )
    windows = pd.DataFrame(
        dict(
            cycle_name="cycle",
            midpoint_minutes=[7.5, 12.5, 17.5],
            window_minutes=5,
            slope_change=[-1.0, -1.0, 1.0],
        )
    )
    result = slope_direction_support(curves, windows)
    assert result.comparison_above_cop_minimum.tolist() == [False, False, True]
    assert np.allclose(result.minimum_relative_cop_in_comparison, [0.4, 0.5, 0.6])
    assert np.allclose(result.comparison_start_progress, np.array([5.0, 10.0, 15.0]) / 30)


def test_first_visible_rise_requires_sustained_supported_stage_and_selects_once():
    from tests.cost_workspace.cop_slope import visible_frequency_rises

    times = pd.date_range("2026-01-01", periods=6, freq="5min")
    stages = pd.DataFrame(
        dict(
            cycle_name=["a"] * 5 + ["b"],
            rise_start=times,
            rise_end=times + pd.to_timedelta([60, 60, 0, 60, 60, 60], unit="s"),
            rise_hz=[2, 10, 10, 5, 12, 3],
            cop_supported=[True, True, True, True, True, False],
            pre_over_peak=[1, 0.5, 1, 0.995, 0.98, 1],
        )
    )
    first = visible_frequency_rises(stages)
    assert first.index.tolist() == [3]
    assert first.rise_size_class.iloc[0] == "5–<10 Hz"
    assert first.pre_within_1pct_peak.iloc[0]
    assert visible_frequency_rises(stages, first_only=False).index.tolist() == [3, 4]


def test_frequency_change_colour_extends_both_directions_without_using_invalid_changes():
    from plots.frequency import frequency_activity_segments

    progress = np.arange(9) / 100
    frequency = np.array([70, 70, 71, 71, 71, 71, 71, 70, 70])
    valid = np.ones(8, dtype=bool)
    assert frequency_activity_segments(
        progress, frequency, valid, half_width_per_hz=0.01
    ).tolist() == [False, True, True, False, False, False, True, True]
    assert frequency_activity_segments(
        progress, frequency, valid, half_width_per_hz=0.01, direction="increase"
    ).tolist() == [False, True, True, False, False, False, False, False]
    assert frequency_activity_segments(
        progress, frequency, valid, half_width_per_hz=0.01, direction="decrease"
    ).tolist() == [False, False, False, False, False, False, True, True]
    reversal = np.array([70, 70, 71, 70, 70, 70, 70, 70, 70])
    marked = frequency_activity_segments(
        progress, reversal, valid, direction="increase", half_width_per_hz=0.05
    )
    assert marked[1] and not marked[2]  # A nearby downstep cannot inherit rise colour.
    valid[1] = False
    assert frequency_activity_segments(
        progress, frequency, valid, half_width_per_hz=0.01
    ).tolist() == [False, False, False, False, False, False, True, True]
    progress = np.linspace(0, 0.1, 101)
    valid = np.ones(100, dtype=bool)
    counts = []
    for delta in [1, 2, 3]:
        frequency = np.where(progress < 0.05, 70, 70 + delta)
        counts.append(frequency_activity_segments(progress, frequency, valid).sum())
    assert counts == [10, 20, 30]


def test_frequency_cop_cells_classify_six_directions_without_crossing_missing_support():
    from tests.cost_workspace.cop_slope import frequency_cop_cells

    origin = pd.Timestamp("2026-01-01")
    t = np.arange(0.5, 4.0, 0.5)
    g = pd.DataFrame(
        dict(
            candidate_defrost_time=origin + pd.to_timedelta(t, unit="min"),
            heating_accounting_start=origin,
            measured_cycle_start=origin,
            cycle_cop=[0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8],
            cycle_cop_eligible=True,
            compressor_frequency=[40.0, 41.0, 42.0, 42.0, 42.0, 41.0, 40.0],
        )
    )
    result = frequency_cop_cells(g)
    assert result.category.tolist() == [0, 1, 2, 3, 4, 5]
    assert not result.within_1pct_peak_entire_cell.any()
    g.loc[3, "cycle_cop_eligible"] = False
    result = frequency_cop_cells(g)
    assert result.category.tolist() == [0, 1, 4, 5]


def test_near_peak_strip_does_not_bridge_an_internal_cop_dip():
    from tests.cost_workspace.cop_slope import frequency_cop_cells

    origin = pd.Timestamp("2026-01-01")
    t = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
    g = pd.DataFrame(
        dict(
            candidate_defrost_time=origin + pd.to_timedelta(t, unit="min"),
            heating_accounting_start=origin,
            measured_cycle_start=origin,
            cycle_cop=[0.995, 0.98, 1.0, 0.998, 0.997],
            cycle_cop_eligible=True,
            compressor_frequency=40.0,
        )
    )
    result = frequency_cop_cells(g)
    assert result.within_1pct_peak_entire_cell.tolist() == [False, True]


def test_frequency_rises_merge_ramps_and_use_strictly_pre_rise_cop():
    from tests.cost_workspace.cop_slope import frequency_rise_events

    times = pd.date_range("2026-01-01", periods=7, freq="10s")
    raw = pd.DataFrame(dict(timestamp=times, frequency=[40.0, 41.0, 42.0, 42.0, 40.0, 41.0, 41.0]))
    curve = pd.DataFrame(
        dict(
            candidate_defrost_time=times,
            heating_accounting_start=times[0],
            cycle_cop=[0.8, 0.9, 0.95, 0.99, 1.0, 0.98, 0.96],
            cycle_cop_eligible=True,
        )
    )
    result = frequency_rise_events(raw, curve, frequency_column="frequency")
    assert result.rise_start.tolist() == [times[1], times[5]]
    assert result.rise_hz.tolist() == [2.0, 1.0]
    assert result.pre_cop.tolist() == [0.8, 1.0]
    assert result.post_peak.tolist() == [False, True]
    assert (result.pre_cop_time < result.rise_start).all()


def test_rise_stages_include_holds_but_stop_at_falls_and_observation_gaps():
    from tests.cost_workspace.cop_slope import frequency_rise_events

    origin = pd.Timestamp("2026-01-01")
    seconds = np.r_[np.arange(0, 101, 10), 200, 210]
    raw = pd.DataFrame(
        dict(
            timestamp=origin + pd.to_timedelta(seconds, unit="s"),
            frequency=[
                40.0,
                41.0,
                41.0,
                41.0,
                41.0,
                41.0,
                41.0,
                42.0,
                40.0,
                40.0,
                41.0,
                41.0,
                42.0,
            ],
        )
    )
    curve = pd.DataFrame(
        dict(
            candidate_defrost_time=origin + pd.to_timedelta(np.arange(0, 211, 10), unit="s"),
            heating_accounting_start=origin,
            cycle_cop=1.0,
            cycle_cop_eligible=True,
        )
    )
    stages = frequency_rise_events(raw, curve, frequency_column="frequency", merge_seconds=np.inf)
    assert ((stages.rise_start - origin).dt.total_seconds()).tolist() == [10, 100, 210]
    assert ((stages.rise_end - origin).dt.total_seconds()).tolist() == [70, 100, 210]
    assert stages.rise_hz.tolist() == [2.0, 1.0, 1.0]
