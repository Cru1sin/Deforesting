import numpy as np
import pandas as pd


def test_processed_alignment_accepts_string_and_object_identity_columns():
    from heat_smoothing.methods import process_measurements

    raw = pd.DataFrame(
        dict(
            timestamp=pd.date_range("2026-09-17", periods=2, freq="s"),
            experiment_id="a",
            cycle_id="one",
            cycle_stage="recovery",
            heating_capacity=[6.0, 8.0],
        )
    )
    raw["cycle_id"] = raw.cycle_id.astype("string")
    target = raw.copy()
    target["cycle_id"] = target.cycle_id.astype(object)
    before = raw.copy()
    result = process_measurements(raw, target)
    np.testing.assert_allclose(result.heating_capacity_processed, [6.0, 7.0])
    pd.testing.assert_frame_equal(raw, before)
    pd.testing.assert_series_equal(result.cycle_id, target.cycle_id)


def test_offline_interpolation_precedes_smoothing_and_keeps_one_flag():
    from defrost_event_models.training_data import candidate_integral_table, window_audit
    from heat_smoothing.methods import MEASUREMENT_PROCESSING, process_measurements

    origin = pd.Timestamp("2026-01-01")
    raw = pd.DataFrame(
        dict(
            timestamp=pd.date_range(origin, periods=83, freq="s"),
            experiment_id="a",
            cycle_id="one",
            cycle_stage="frost_development",
            heating_capacity=np.arange(83, dtype=float),
            coil_temperature=np.arange(83, dtype=float),
            power_total=2.0,
            compressor_frequency=50.0,
        )
    )
    raw.loc[
        1:79, ["heating_capacity", "coil_temperature", "power_total", "compressor_frequency"]
    ] = np.nan
    raw.loc[82, ["heating_capacity", "coil_temperature"]] = np.nan
    raw["heating_capacity_low"] = 999.0
    raw["heating_capacity__imputed"] = False
    result = process_measurements(raw, raw)
    assert result.heating_capacity_interpolated.iloc[40] == 40.0
    assert result.coil_temperature_interpolated.iloc[40] == 40.0
    assert result.heating_capacity_processed.iloc[40] == 20.0
    assert result.compressor_frequency_filled.iloc[40] == 50.0
    assert pd.isna(result.heating_capacity_interpolated.iloc[-1])
    assert result.is_interpolated.sum() == 79
    assert not any(n.endswith("__imputed") or n.startswith("heating_capacity_low") for n in result)
    pd.testing.assert_series_equal(result.heating_capacity, raw.heating_capacity)
    frame = result.copy()
    frame.attrs["measurement_processing"] = MEASUREMENT_PROCESSING
    ends = [origin + pd.Timedelta(seconds=v) for v in [0, 40, 40.5, 80]]
    batch = candidate_integral_table(frame, origin, ends, "heating_capacity_processed")
    single = pd.DataFrame(
        [window_audit(frame, origin, end, "heating_capacity_processed") for end in ends]
    )
    pd.testing.assert_frame_equal(batch, single, check_exact=False, rtol=1e-12, atol=1e-12)
    assert batch.valid.all()
    assert np.isclose(batch.energy.iloc[1], 400 / 3600)


def test_processed_bucket_start_never_reads_future_raw_rows():
    from heat_smoothing.methods import process_measurements

    origin = pd.Timestamp("2026-01-01")
    raw = pd.DataFrame(
        dict(
            timestamp=origin + pd.to_timedelta([0, 1, 9, 10], unit="s"),
            experiment_id="a",
            cycle_id="one",
            cycle_stage="frost_development",
            heating_capacity=[6.0, 100.0, 100.0, 6.0],
            evaporating_pressure=[0.6, 10.0, 10.0, 0.6],
        )
    )
    target = raw.iloc[[0, 3]].copy()
    result = process_measurements(raw, target)
    assert result.heating_capacity_processed.iloc[0] == 6
    altered = raw.copy()
    altered.loc[1:, "heating_capacity"] = 999
    assert process_measurements(altered, target).heating_capacity_processed.iloc[0] == 6


def test_ridge_coil_reads_processed_once_and_no_legacy_recipe():
    from defrost_event_models.training_data import extract_pre_defrost_features
    from tests.cost_workspace.test_v268_science import _raw_frame

    start = pd.Timestamp("2026-01-01")
    frame = _raw_frame(start)
    frame["coil_temperature_processed"] = 1.25
    frame["coil_temperature"] = 999
    got = extract_pre_defrost_features(frame, [start + pd.Timedelta(minutes=6)], start)
    assert got.coil_temperature.iloc[0] == 1.25


def test_three_channels_share_stage_reset_mean_and_ridge_does_not_smooth_twice():
    from defrost_event_models.training_data import extract_pre_defrost_features
    from heat_smoothing.methods import PROCESSED_CHANNELS, process_measurements
    from tests.cost_workspace.test_v268_science import _raw_frame

    start = pd.Timestamp("2026-01-01")
    raw = _raw_frame(start, 131)
    raw["experiment_id"], raw["cycle_id"] = "one", "one"
    raw["cycle_stage"] = np.where(np.arange(131) < 70, "recovery", "frost_development")
    for name in PROCESSED_CHANNELS:
        raw[name] = np.r_[np.full(70, 1000.0), np.arange(61, dtype=float)]
    processed = process_measurements(raw, raw)
    for name in PROCESSED_CHANNELS:
        assert processed[name + "_processed"].iloc[70] == 0.0
        assert processed[name + "_processed"].iloc[129] == 29.5
    processed["coil_temperature"] = 999.0
    processed.loc[130, "coil_temperature_processed"] = 999.0
    got = extract_pre_defrost_features(processed, [start + pd.Timedelta(seconds=130)], start)
    assert got.coil_temperature.iloc[0] == 29.5


def test_processed_measurements_reject_relabelled_boundaries():
    import pytest

    from dataset_tools import DatasetLoader

    loader = object.__new__(DatasetLoader)
    loader.recovery_settings = {"recovery_rule": "frequency-setpoint"}
    with pytest.raises(ValueError, match="saved Dataset boundaries"):
        loader.load_cycle_measurements("cycle")


def test_processed_entries_default_to_dataset_boundaries():
    import pytest

    from defrosting_electricity_model import build_parser

    # Current fitting always uses saved Dataset boundaries.
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--recovery-rule", "fixed-nine-minutes"])


def test_three_input_features_need_only_the_three_input_channels():
    from defrost_event_models.event_electricity_ridge import THREE_INPUTS
    from defrost_event_models.training_data import extract_pre_defrost_features

    start = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=61, freq="s"),
            "compressor_frequency": 60.0,
            "ambient_temperature": 5.0,
            "coil_temperature_processed": -2.0,
        }
    )
    features = extract_pre_defrost_features(frame, [start + pd.Timedelta(seconds=60)], start)
    assert features[[name + "_valid_second_count" for name in THREE_INPUTS]].eq(60).all().all()
    assert "evaporating_pressure_slope_5m" not in features
    np.testing.assert_allclose(features[list(THREE_INPUTS)], [[60.0, 5.0, -2.0]])
