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
