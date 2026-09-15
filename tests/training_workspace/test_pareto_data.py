import importlib.util

import numpy as np
import pandas as pd

from defrost_decision import candidate_quantities
from image_models import sensor_features


def test_statistics_mask_imputation_and_are_available_at_bucket_end():
    frame = pd.DataFrame(
        {
            "cycle_name": "a",
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="10s"),
            "x": np.arange(8.0),
            "x__imputed": [False] * 7 + [True],
        }
    )
    assert hasattr(sensor_features, "build_past_only_sensor_statistics")
    build = sensor_features.build_past_only_sensor_statistics
    result = build(frame, current_sensors=("x",))
    assert result.sensor_timestamp.iloc[0] == frame.timestamp.iloc[0] + pd.Timedelta(seconds=10)
    assert result.stat_x_mean.iloc[-1] == 3
    assert np.isclose(result.stat_x_slope.iloc[-1], 6)
    assert result.stat_x_valid_count.iloc[-1] == 7
    assert result.stat_x_age_seconds.iloc[-1] == 10
    assert not result.stat_x_mean_missing.iloc[-1]
    assert result.stat_x_std_missing.iloc[0]
    pd.testing.assert_frame_equal(result.iloc[:5], build(frame.iloc[:5], current_sensors=("x",)))
    tiny = frame.copy()
    tiny["x"] = [20.0, np.nextafter(20.0, 21.0)] * 4
    assert np.isfinite(build(tiny, current_sensors=("x",)).stat_x_entropy.iloc[-1])


def test_measured_inputs_do_not_integrate_before_accounting_start():
    assert hasattr(candidate_quantities, "build_measured_candidate_quantities")
    from defrost_event_models.training_data import RAW_COLUMNS, build_candidate_boundaries

    start = pd.Timestamp("2026-01-01")

    class Loader:
        def get_cycle_record(self, name):
            return {
                "experiment_id": "a",
                "heating_start": start,
                "defrost_preparation_start": start + pd.Timedelta(minutes=20),
            }

        def load_cycle_original(self, name, columns):
            frame = pd.DataFrame(1.0, index=range(1201), columns=RAW_COLUMNS[1:])
            frame["timestamp"] = pd.date_range(start, periods=len(frame), freq="s")
            return frame

    times = build_candidate_boundaries("a", "a", start, start + pd.Timedelta(minutes=20))
    times.loc[0, "candidate_defrost_time"] = start + pd.Timedelta(minutes=2)
    result = candidate_quantities.build_measured_candidate_quantities(Loader(), "a", times)
    assert np.isnan(result.pre_defrost_electricity_kwh.iloc[0])
    assert not result.pre_defrost_electricity_measurement_valid.iloc[0]
    assert np.isclose(result.pre_defrost_electricity_kwh.iloc[1], 2 / 60)


def test_online_accounting_never_uses_a_future_gap_endpoint():
    from defrost_event_models.training_data import RAW_COLUMNS, build_candidate_boundaries

    start = pd.Timestamp("2026-01-01")

    class Loader:
        def __init__(self, future_power=3.0):
            self.future_power = future_power

        def get_cycle_record(self, name):
            return {
                "experiment_id": "a", "heating_start": start,
                "defrost_preparation_start": start + pd.Timedelta(minutes=20),
            }

        def load_cycle_original(self, name, columns):
            frame = pd.DataFrame(1.0, index=range(1201), columns=RAW_COLUMNS[1:])
            frame["timestamp"] = pd.date_range(start, periods=len(frame), freq="s")
            gap = frame.timestamp.between(start + pd.Timedelta(minutes=10),
                                          start + pd.Timedelta(minutes=10, seconds=59))
            frame.loc[gap, "power_total"] = np.nan
            frame.loc[frame.timestamp.ge(start + pd.Timedelta(minutes=11)), "power_total"] = (
                self.future_power
            )
            return frame

    times = build_candidate_boundaries("a", "a", start, start + pd.Timedelta(minutes=20))
    times.loc[1, "candidate_defrost_time"] = start + pd.Timedelta(minutes=10, seconds=30)
    first = candidate_quantities.build_measured_candidate_quantities(
        Loader(3), "a", times, allow_measurement_reconstruction=True
    )
    changed = candidate_quantities.build_measured_candidate_quantities(
        Loader(300), "a", times, allow_measurement_reconstruction=True
    )
    assert first.online_pre_defrost_electricity_kwh.iloc[1] == (
        changed.online_pre_defrost_electricity_kwh.iloc[1]
    )
    assert not first.online_pre_defrost_electricity_measurement_valid.iloc[1]
    assert first.pre_defrost_electricity_kwh.iloc[1] != changed.pre_defrost_electricity_kwh.iloc[1]


def test_fold_teacher_uses_only_grid_and_preserves_native_rows():
    assert importlib.util.find_spec("image_models.pareto_data") is not None
    from image_models.pareto_data import apply_fold_teacher

    times = pd.date_range("2026-01-01", periods=40, freq="10s")
    base = pd.DataFrame(
        {
            "cycle_name": "a",
            "candidate_defrost_time": times,
            "image_time": times,
            "is_teacher_candidate": True,
            "is_frame": True,
            "stable_heating_start": times[0],
            "heating_accounting_start": times[0] - pd.Timedelta(minutes=10),
            "pre_defrost_feature_window_valid": True,
        }
    )
    for name in ("electricity", "heat", "compressor_electricity"):
        base[f"pre_defrost_{name}_kwh"] = np.linspace(1, 2, len(base)) * (
            3 if name == "heat" else 1
        )
        base[f"pre_defrost_{name}_measurement_valid"] = True
    parameters = {}
    for name, value in (
        ("event_electricity", 1),
        ("event_net_heat", -1),
        ("event_compressor_electricity", 0.2),
        ("event_duration", 5),
    ):
        parameters[name] = {
            "feature_order": [],
            "imputer_median": [],
            "scaler_mean": [],
            "scaler_scale": [],
            "coefficients": [],
            "intercept": value,
            "training_standardized_references": [[]],
            "support_threshold": 0,
        }
    selected = apply_fold_teacher(base, parameters)
    native = base.iloc[[5]].copy()
    native["candidate_defrost_time"] += pd.Timedelta(seconds=1)
    native["image_time"] = native.candidate_defrost_time
    native["is_teacher_candidate"] = False
    native["pre_defrost_heat_kwh"] = 10000
    union = apply_fold_teacher(pd.concat([base, native], ignore_index=True), parameters)
    assert union.teacher_time.nunique() == 1
    assert union.teacher_time.iloc[0] == selected.teacher_time.iloc[0]
    assert union.is_knee.sum() == 1
    assert union.loc[~union.is_teacher_candidate, "pareto_selection_score"].isna().all()


def test_neural_pareto_reuses_formula_and_selects_only_the_frozen_grid():
    from defrost_event_models.ridge_models import OUTCOME_TARGETS
    from image_models.pareto_data import apply_fold_teacher, apply_neural_pareto

    times = pd.date_range("2026-01-01", periods=40, freq="10s")
    base = pd.DataFrame({
        "row_id": [f"a:{time.value}" for time in times],
        "cycle_name": "a", "candidate_defrost_time": times, "image_time": times,
        "is_teacher_candidate": True, "is_frame": True,
        "stable_heating_start": times[0],
        "heating_accounting_start": times[0] - pd.Timedelta(minutes=10),
        "pre_defrost_feature_window_valid": True,
    })
    for name in ("electricity", "heat", "compressor_electricity"):
        base[f"pre_defrost_{name}_kwh"] = np.linspace(1, 2, len(base)) * (
            3 if name == "heat" else 1
        )
        base[f"pre_defrost_{name}_measurement_valid"] = True
    parameters = {}
    for name, value in (
        ("event_electricity", 1), ("event_net_heat", -1),
        ("event_compressor_electricity", .2), ("event_duration", 5),
    ):
        parameters[name] = {
            "feature_order": [], "imputer_median": [], "scaler_mean": [],
            "scaler_scale": [], "coefficients": [], "intercept": value,
            "training_standardized_references": [[]], "support_threshold": 0,
        }
    ridge = apply_fold_teacher(base, parameters, allow_model_extrapolation=True)
    native = ridge.iloc[[5]].copy()
    native["row_id"] = "native"
    native["candidate_defrost_time"] += pd.Timedelta(seconds=1)
    native["is_teacher_candidate"] = False
    union = pd.concat([ridge, native], ignore_index=True)
    predictions = union[["row_id"]].copy()
    for outcome, target in OUTCOME_TARGETS.items():
        unit = "minutes" if outcome == "event_duration" else "kwh"
        predictions[f"predicted_{target}"] = union[f"defrost_{outcome}_{unit}"]

    neural = apply_neural_pareto(union, predictions)

    assert neural.neural_teacher_time.nunique() == 1
    assert neural.neural_teacher_time.iloc[0] == ridge.teacher_time.iloc[0]
    assert neural.neural_is_knee.sum() == 1
    assert not neural.loc[~neural.is_teacher_candidate, "neural_is_knee"].any()
    grid = neural.is_teacher_candidate
    np.testing.assert_allclose(neural.loc[grid, "neural_c"], neural.loc[grid, "cycle_cop"])
    np.testing.assert_allclose(
        neural.loc[grid, "neural_h"], neural.loc[grid, "cycle_heating_rate_kw"]
    )
    assert neural.neural_event_prediction_domain.eq("unknown").all()


def test_online_economic_features_are_pointwise_and_past_only():
    from image_models.pareto_data import add_online_economic_features

    times = pd.date_range("2026-01-01", periods=4, freq="min")
    rows = pd.DataFrame(
        {
            "cycle_name": "a", "candidate_defrost_time": times,
            "heating_accounting_start": times[0] - pd.Timedelta(minutes=1),
            "online_pre_defrost_electricity_kwh": [1.0, 2.0, np.nan, 4.0],
            "online_pre_defrost_heat_kwh": [3.0, 6.0, np.nan, 12.0],
            "online_pre_defrost_compressor_electricity_kwh": [0.5, 1.0, np.nan, 2.0],
            "online_pre_defrost_electricity_measurement_valid": [True, True, False, True],
            "online_pre_defrost_heat_measurement_valid": [True, True, False, True],
            "online_pre_defrost_compressor_measurement_valid": [True, True, False, True],
            "pre_defrost_feature_window_valid": True,
            "defrost_event_electricity_kwh": 1.0,
            "defrost_event_net_heat_kwh": -0.5,
            "defrost_event_compressor_electricity_kwh": 0.2,
            "defrost_event_duration_minutes": 5.0,
        }
    )
    for name in ("electricity", "net_heat", "compressor_electricity", "duration"):
        rows[f"defrost_event_{name}_prediction_available"] = True
        rows[f"defrost_event_{name}_in_training_domain"] = True
    result = add_online_economic_features(rows)
    assert result.online_pointwise_valid.tolist() == [True, True, False, True]
    assert np.isnan(result.online_c.iloc[2])
    assert result.online_c_valid_count.iloc[1] == 2
    assert result.online_c_valid_count.iloc[2] == 2
    assert result.online_c_age_seconds.iloc[2] == 60
    earlier = add_online_economic_features(rows.iloc[:2])
    pd.testing.assert_series_equal(
        result.online_c.iloc[:2].reset_index(drop=True),
        earlier.online_c.reset_index(drop=True), check_names=False,
    )
    neural = add_online_economic_features(rows, feature_prefix="neural")
    np.testing.assert_allclose(neural.neural_c, result.online_c, equal_nan=True)
    assert neural.neural_c_valid_count.tolist() == result.online_c_valid_count.tolist()


def test_fold_parameters_exclude_all_heldout_experiments():
    from defrost_event_models.ridge_models import DYNAMIC_STATE_8, OUTCOME_TARGETS
    from image_models.pareto_data import fit_fold_parameters

    events = pd.DataFrame(np.random.default_rng(1).normal(size=(20, 8)), columns=DYNAMIC_STATE_8)
    events["experiment_id"] = np.repeat(list("abcde"), 4)
    events["event_id"] = [str(value) for value in range(20)]
    events["event_valid"] = True
    for target in OUTCOME_TARGETS.values():
        events[target] = np.arange(20.0)
    fitted = fit_fold_parameters(events, {"c", "d"})
    for parameters in fitted.values():
        assert set(parameters["training_experiment_ids"]) == {"a", "b", "e"}
        assert parameters["training_event_count"] == 12
