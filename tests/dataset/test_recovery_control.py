import numpy as np
import pandas as pd

from dataset_tools.builder import detect_cycles


def startup():
    seconds = np.arange(1001)
    frequency = np.where(
        seconds < 30,
        30.0,
        np.where(seconds < 150, 42.0, np.minimum(42 + 3 * np.floor((seconds - 150) / 30), 87.0)),
    )
    return pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(seconds, unit="s"),
            "compressor_frequency_setpoint": frequency,
            "compressor_frequency": frequency - 1,
            "condensing_pressure": 0.5 + frequency / 70,
            "evaporating_pressure": 0.4,
        }
    )


def test_offline_boundary_is_the_transition_without_confirmation_padding():
    frame = startup()
    trace = detect_cycles.recovery_control_trace(frame, {})
    boundary = trace.loc[trace.normal_heating, "timestamp"].iloc[0]
    assert abs((boundary - frame.timestamp.iloc[600]).total_seconds()) <= 1
    assert not trace.normal_heating.iloc[:600].any()
    assert trace.normal_heating.iloc[601:].all()


def test_missing_signal_has_no_fallback_and_later_gap_does_not_move_boundary():
    assert hasattr(detect_cycles, "recovery_control_trace")
    frame = startup()
    trace = detect_cycles.recovery_control_trace(
        frame.drop(columns="compressor_frequency_setpoint"), {}
    )
    assert not trace.normal_heating.any()
    assert trace.recovery_status.iloc[-1] == "missing_signal"
    frame.loc[650:850, "compressor_frequency_setpoint"] = np.nan
    trace = detect_cycles.recovery_control_trace(frame, {})
    assert trace.loc[trace.normal_heating, "timestamp"].iloc[0] == frame.timestamp.iloc[600]


def test_isolated_missing_samples_do_not_restart_observed_ramp():
    frame = startup()
    frame.loc[::45, "compressor_frequency_setpoint"] = np.nan
    trace = detect_cycles.recovery_control_trace(frame, {})
    assert trace.normal_heating.any()
    assert trace.loc[trace.normal_heating, "timestamp"].iloc[0] < frame.timestamp.iloc[820]


def test_normal_downregulation_and_hidden_transition():
    frame = startup()
    seconds = np.arange(len(frame))
    frame.loc[seconds >= 600, "compressor_frequency_setpoint"] = 87 - np.floor(
        (seconds[600:] - 600) / 20
    )
    trace = detect_cycles.recovery_control_trace(frame, {})
    assert trace.normal_heating.iloc[820]
    frame = startup()
    frame.loc[500:850, "compressor_frequency_setpoint"] = np.nan
    trace = detect_cycles.recovery_control_trace(frame, {})
    assert not trace.normal_heating.any()
    assert trace.recovery_status.iloc[-1] == "transition_hidden_by_gap"


def test_missing_preparation_boundary_keeps_unconfirmed_stage(monkeypatch, tmp_path):
    from dataset_tools.load_dataset import DatasetLoader

    path = tmp_path / "cycle.parquet"
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="s"),
            "cycle_stage": ["frost_development", "defrost"],
        }
    )
    frame.to_parquet(path)
    loader = DatasetLoader.__new__(DatasetLoader)
    loader.dataset_root = tmp_path
    loader.recovery_settings = {}
    loader._catalog = {
        "cycles": [{"cycle_name": "test", "assets": {"parquet": path.name}, "boundaries": {}}]
    }
    assert loader.load_cycle("test").cycle_stage.tolist() == ["recovery", "defrost"]


def test_real_offline_boundaries_stay_inside_observed_heating():
    from pathlib import Path

    import pytest

    from dataset_tools import DatasetLoader

    dataset = Path(__file__).resolve().parents[4] / "dataset"
    if not dataset.exists():
        pytest.skip("local 138-cycle Dataset unavailable")
    loader = DatasetLoader(dataset)
    for name in loader.list_cycles().cycle_name:
        bounds = loader.get_cycle_record(name)["boundaries"]
        frame = loader.load_cycle_original(name)
        frame.timestamp = pd.to_datetime(frame.timestamp)
        frame = frame.loc[frame.timestamp.ge(pd.to_datetime(bounds["heating_start"]))]
        end = pd.to_datetime(bounds.get("defrost_preparation_start") or bounds.get("defrost_start"))
        if pd.notna(end):
            frame = frame.loc[frame.timestamp.lt(end)]
        trace = detect_cycles.recovery_control_trace(frame, {})
        identified = trace.loc[trace.normal_heating, "timestamp"]
        if len(identified):
            assert frame.timestamp.min() <= identified.iloc[0] <= frame.timestamp.max(), name
            expected_seconds = {120: 740, 133: 740, 134: 741, 135: 740, 138: 710}
            cycle_number = int(name.rsplit("_", 1)[-1])
            if cycle_number in expected_seconds:
                seconds = (
                    identified.iloc[0] - pd.Timestamp(bounds["heating_start"])
                ).total_seconds()
                assert seconds == expected_seconds[cycle_number], name
            assert trace.loc[trace.timestamp.ge(identified.iloc[0]), "normal_heating"].all(), name


def test_slow_positive_regulation_uses_frozen_ramp_reference():
    frame = startup()
    seconds = np.arange(len(frame))
    frame.loc[seconds >= 600, "compressor_frequency_setpoint"] = 87 + np.floor(
        (seconds[600:] - 600) / 50
    )
    trace = detect_cycles.recovery_control_trace(frame, {})
    reference = trace.compressor_frequency_setpoint_fast_reference.dropna()
    assert reference.nunique() == 1
    confirmed = trace.loc[trace.normal_heating, "timestamp"]
    assert len(confirmed)
    assert abs((confirmed.iloc[0] - frame.timestamp.iloc[600]).total_seconds()) <= 1


def test_high_frequency_left_truncation_cannot_start_a_new_recovery():
    frame = startup()
    frame["compressor_frequency_setpoint"] = np.minimum(76 + np.arange(len(frame)) / 60, 88)
    trace = detect_cycles.recovery_control_trace(frame, {})
    assert not trace.normal_heating.any()
    assert trace.recovery_status.iloc[-1] == "awaiting_startup_observation"


def test_staircase_pause_does_not_end_recovery():
    frame = startup()
    seconds = np.arange(1601)
    ramp_clock = seconds - np.clip(seconds - 600, 0, 45)
    frequency = np.where(seconds < 150, 30.0, 42 + 2 * np.floor((ramp_clock - 150) / 60))
    frequency = np.minimum(frequency, 74.0)
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(seconds, unit="s"),
            "compressor_frequency_setpoint": frequency,
        }
    )
    trace = detect_cycles.recovery_control_trace(frame, {})
    assert not trace.loc[seconds < 1155, "normal_heating"].any()
    assert trace.normal_heating.iloc[-1]


def test_sampling_jitter_does_not_delay_offline_control_transition():
    seconds = np.arange(901)
    commands = np.array(
        [
            0,
            30,
            180,
            211,
            242,
            273,
            303,
            333,
            363,
            393,
            423,
            453,
            483,
            513,
            543,
            573,
            603,
            663,
            723,
            783,
            843,
        ]
    )
    levels = np.array([30, 42, *range(44, 74, 2), 73, 74, 75, 76])
    frequency = levels[np.searchsorted(commands, seconds, side="right") - 1]
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(seconds, unit="s"),
            "compressor_frequency_setpoint": frequency,
            "compressor_frequency": np.r_[30, 30, frequency[:-2]],
        }
    )
    for rule, expected in [("frequency-setpoint", 603), ("frequency-actual", 605)]:
        trace = detect_cycles.recovery_control_trace(frame, {"recovery_rule": rule})
        assert (
            trace.loc[trace.normal_heating, "timestamp"].iloc[0] == frame.timestamp.iloc[expected]
        )
    assert not detect_cycles.recovery_control_trace(frame.iloc[:630], {}).normal_heating.any()


def test_old_online_settings_cannot_silently_change_boundary_definition():
    import pytest

    with pytest.raises(ValueError, match="obsolete recovery settings"):
        detect_cycles.recovery_control_trace(startup(), {"slow_seconds": 15})


def test_publication_uses_exact_offline_boundary_and_observed_pressure(monkeypatch, tmp_path):
    from matplotlib.figure import Figure

    from plots import publication

    frame = startup().iloc[::10].copy()
    boundary = frame.timestamp.iloc[60] + pd.Timedelta(seconds=3)
    frame["cycle_stage"] = np.where(frame.timestamp.lt(boundary), "recovery", "frost_development")
    frame["condensing_pressure__imputed"] = False
    frame.loc[frame.index[1], "condensing_pressure__imputed"] = True
    captured = {}
    monkeypatch.setattr(Figure, "savefig", lambda figure, *a, **k: captured.update(figure=figure))
    record = {
        "cycle_name": "test",
        "recovery_status": "identified_offline",
        "boundaries": {
            "stable_heating_start": boundary,
            "baseline_start": boundary,
            "baseline_end": boundary + pd.Timedelta(seconds=60),
        },
    }
    publication.render_cycle_publication(frame, record, tmp_path / "cycle.png")
    axis = next(ax for ax in captured["figure"].axes if ax.get_ylabel() == "Pc − Pe [MPa]")
    pressure = axis.lines[0].get_ydata()
    assert np.isnan(pressure[1])
    assert np.isclose(
        pressure[0], frame.condensing_pressure.iloc[0] - frame.evaporating_pressure.iloc[0]
    )
    # Historical baseline fields must not add a third shading patch.
    assert len(axis.patches) == 2
    spans = [(patch.get_x(), patch.get_width()) for patch in axis.patches]
    assert any(np.isclose(start + width, 10.05) for start, width in spans)
    assert any(np.isclose(start, 10.05) for start, width in spans)
    record["boundaries"].update(heating_origin="cold_start", heating_start=frame.timestamp.iloc[3])
    publication.render_cycle_publication(frame, record, tmp_path / "cold_start.png")
    frequency_axis = captured["figure"].axes[1]
    assert frequency_axis.lines[0].get_xdata().min() == 0
    assert any(text.get_text() == "Cold Start" for text in captured["figure"].axes[0].texts)


def test_fast_start_and_isolated_small_step_are_not_normal_regulation():
    # 8 Hz/min startup, then 4 Hz/min ramp containing a 1 Hz step.
    seconds = np.arange(1001)
    commands = np.array([0, 30, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 510, 570])
    levels = np.array([30, 42, 48, 52, 56, 58, 59, 62, 64, 66, 68, 70, 72, 73, 74])
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(seconds, unit="s"),
            "compressor_frequency_setpoint": levels[
                np.searchsorted(commands, seconds, side="right") - 1
            ],
        }
    )
    trace = detect_cycles.recovery_control_trace(frame, {})
    assert trace.loc[trace.normal_heating, "timestamp"].iloc[0] == frame.timestamp.iloc[450]


def test_only_observed_initial_standby_is_excluded():
    from types import SimpleNamespace

    frame = startup()
    frame.loc[:29, "compressor_frequency_setpoint"] = 0
    bounds = {"heating_start": frame.timestamp.iloc[0]}
    loader = SimpleNamespace(
        get_cycle_record=lambda name: {"boundaries": bounds, "experiment_id": "e"},
        list_cycles=lambda: pd.DataFrame(
            {
                "cycle_name": ["first", "next"],
                "experiment_id": ["e", "e"],
                "defrost_end": [frame.timestamp.iloc[0], pd.NaT],
            }
        ),
    )
    initial = detect_cycles.heating_episode_bounds(loader, "first", frame)
    subsequent = detect_cycles.heating_episode_bounds(loader, "next", frame)
    assert initial["heating_start"] == frame.timestamp.iloc[30]
    assert initial["heating_origin"] == "cold_start"
    assert initial["excluded_prestart_seconds"] == 30
    assert subsequent["heating_start"] == bounds["heating_start"]
    assert subsequent["heating_origin"] == "post_defrost"
    frame["cycle_stage"] = "recovery"
    assert detect_cycles.recovery_stages(frame, initial).iloc[:30].isna().all()
    assert detect_cycles.recovery_stages(frame, subsequent).iloc[:30].eq("recovery").all()
    truncated = detect_cycles.heating_episode_bounds(loader, "first", frame.iloc[30:])
    assert truncated["heating_origin"] == "initial_recording"
    assert truncated["excluded_prestart_seconds"] == 0
    loader.list_cycles = lambda: pd.DataFrame(
        {
            "cycle_name": ["first", "next"],
            "experiment_id": ["e", "e"],
            "defrost_end": [pd.NaT, pd.NaT],
        }
    )
    restart = detect_cycles.heating_episode_bounds(loader, "next", frame)
    assert restart["heating_origin"] == "cold_start"
    assert restart["heating_start"] == frame.timestamp.iloc[30]


def test_preparation_command_at_defrost_timestamp_is_not_missing():
    times = pd.date_range('2026-01-01', periods=5, freq='s')
    frame = pd.DataFrame(dict(timestamp=times, compressor_frequency_setpoint=[70,70,70,70,52]))
    assert detect_cycles.find_defrost_preparation_start(frame,times[0],times[-1],{}) == times[-1]
