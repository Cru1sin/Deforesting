import numpy as np
import pandas as pd
import pytest


def _processed_frame(frame):
    result = frame.copy()
    from heat_smoothing.methods import PROCESSED_CHANNELS

    for name in PROCESSED_CHANNELS:
        if name in result:
            result[name + "_processed"] = result[name]
    return result


@pytest.mark.parametrize("energy,supported", [(-1.0, True), (0.0, True), (1.0, False)])
def test_nonpositive_energy_is_ineligible_and_extrapolation_is_annotation(energy, supported):
    from defrost_decision.performance_objectives import calculate_cycle_cop

    frame = pd.DataFrame(
        {
            "candidate_defrost_time": [pd.Timestamp("2026-01-01")],
            "pre_defrost_electricity_kwh": [10.0],
            "pre_defrost_heat_kwh": [30.0],
            "defrost_event_electricity_kwh": [energy],
            "defrost_event_net_heat_kwh": [0.0],
            "pre_defrost_feature_window_valid": [True],
            "pre_defrost_electricity_measurement_valid": [True],
            "pre_defrost_heat_measurement_valid": [True],
            "defrost_event_electricity_prediction_available": [True],
            "defrost_event_net_heat_prediction_available": [True],
            "defrost_event_electricity_in_training_domain": [supported],
            "defrost_event_net_heat_in_training_domain": [True],
        }
    )
    result = calculate_cycle_cop(frame, effective=True)
    assert result.cycle_cop_eligible.any() == (energy > 0)
    if energy > 0 and not supported:
        assert result.cycle_cop_uses_model_extrapolation.all()


def test_energy_loading_does_not_require_optional_heat_or_compressor_channels():
    from defrost_event_models.training_data import load_frame

    class Loader:
        def load_cycle_original(self, name, columns=None):
            return pd.DataFrame({"timestamp": ["2026-01-01"], "power_total": [2.0]})

        def load_cycle_measurements(self, name):
            return _processed_frame(self.load_cycle_original(name))

    result = load_frame(Loader(), "cycle", {}, extra_columns=("heating_capacity",))
    assert result.power_total.iloc[0] == 2.0
    assert result.heating_capacity.isna().all()
    assert result.compressor_power.isna().all()


@pytest.mark.parametrize(
    "status,defrost,available,pipeline_status",
    [
        ("valid", None, True, None),
        ("partial", None, True, None),
        ("invalid", None, True, None),
        ("valid", 20, False, None),
        ("invalid", None, True, "valid"),
        ("reference", None, True, "valid"),
        ("valid", None, True, "invalid"),
    ],
)
@pytest.mark.parametrize("missing_preparation", ["exclude", "defrost-start"])
def test_hypothetical_defrost_uses_observed_open_heating(
    status, defrost, available, pipeline_status, monkeypatch, missing_preparation
):
    from types import SimpleNamespace

    from defrost_decision import candidate_quantities as decision

    start = pd.Timestamp("2026-01-01")
    end = start + pd.Timedelta(minutes=30)
    bounds = {
        "heating_start": start,
        "stable_heating_start": start + pd.Timedelta(minutes=10),
        "end_time": end,
        "defrost_preparation_start": None,
        "defrost_start": start + pd.Timedelta(minutes=defrost) if defrost else None,
    }
    record = {"boundaries": bounds, "status": status, "experiment_id": "e"}
    if pipeline_status is not None:
        record["pipeline_status"] = pipeline_status
    loader = SimpleNamespace(
        get_cycle_record=lambda _: record,
        load_cycle_measurements=lambda _: pd.DataFrame(
            {"timestamp": pd.date_range(start, end, freq="s")}
        ),
    )
    calls = []

    def evaluate(frame, times, *args, **kwargs):
        calls.append(times)
        return pd.DataFrame(
            {"candidate_defrost_time": times, "cycle_cop": 2.0, "cycle_cop_eligible": True}
        )

    monkeypatch.setattr(decision, "effective_candidate_cop", evaluate)
    result = decision.calculate_cycle(
        loader,
        "cycle",
        {},
        rb={"rb_status": "triggered", "t_RB": start + pd.Timedelta(minutes=25)},
        missing_preparation=missing_preparation,
    )
    fallback = defrost is not None and missing_preparation == "defrost-start"
    available = available or fallback
    assert bool(calls) == available
    if available:
        # Insert the off-grid prior only within the observed heating support.
        assert (start + pd.Timedelta(minutes=25) in calls[0]) == (not fallback)
        assert result.t_RB.eq(start + pd.Timedelta(minutes=25)).all()
        assert max(calls[0]) == (
            bounds["defrost_start"] - pd.Timedelta(seconds=1) if fallback else end
        )
        assert result.cycle_cop_eligible.all()
        assert result.candidate_end_source.eq(
            "last_observation_before_defrost" if fallback else "recording_end"
        ).all()
    assert bounds["defrost_preparation_start"] is None
    assert record["status"] == status


def test_peak_after_relaxation_requires_one_observed_lower_point():
    from dataset_tools.cycle_metadata import complete_peak_screen

    frame = pd.DataFrame(
        {
            "cycle_name": "test",
            "experiment_id": "e",
            "candidate_defrost_time": pd.date_range("2026-01-01", periods=5, freq="30s"),
            "cycle_cop": [0.97, 0.97, 0.97, 1.0, 0.98],
            "cycle_cop_eligible": True,
        }
    )
    assert not complete_peak_screen(frame).selected.iloc[0]
    assert complete_peak_screen(frame, minimum_after_seconds=0).selected.iloc[0]
    frame.loc[4, "cycle_cop_eligible"] = False
    assert not complete_peak_screen(frame, minimum_after_seconds=0).selected.iloc[0]


@pytest.mark.parametrize(
    "origin,first,expected",
    [
        ("initial_recording", True, True),
        ("cold_start", True, True),
        ("post_defrost", True, False),
        ("cold_start", False, False),
    ],
)
def test_first_cycle_can_use_observed_running_without_recovery(
    origin, first, expected, monkeypatch
):
    from types import SimpleNamespace

    from defrost_decision import candidate_quantities as decision

    start = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=1801, freq="s"),
            "compressor_frequency_setpoint": np.r_[np.zeros(60), np.full(1741, 40.0)],
        }
    )
    record = {
        "experiment_id": "e",
        "status": "valid",
        "boundaries": {
            "heating_start": start,
            "heating_origin": origin,
            "stable_heating_start": None,
            "end_time": frame.timestamp.max(),
        },
    }
    loader = SimpleNamespace(
        get_cycle_record=lambda _: record,
        load_cycle_measurements=lambda _: frame.copy(),
        list_cycles=lambda: pd.DataFrame(
            {"cycle_name": ["cycle"] if first else ["earlier", "cycle"], "experiment_id": "e"}
        ),
    )
    seen = []

    def evaluate(frame, times, recovery, *args, **kwargs):
        seen.append(recovery)
        return pd.DataFrame(
            {"candidate_defrost_time": times, "cycle_cop": 2.0, "cycle_cop_eligible": True}
        )

    monkeypatch.setattr(decision, "effective_candidate_cop", evaluate)
    result = decision.calculate_cycle(
        loader,
        "cycle",
        {},
        rb={"rb_status": "right_censored", "t_RB": None},
        initial_heating_start="observed",
    )
    assert bool(seen) == expected
    if expected:
        assert seen[0] == start + pd.Timedelta(seconds=60)
        assert result.heating_accounting_start_source.eq("first_observed_running_first_cycle").all()
    assert record["boundaries"]["stable_heating_start"] is None


def test_action_grid_constrains_optimum_without_changing_support_or_basin():
    from defrost_decision.performance_objectives import add_single_objective_optima

    rows = pd.DataFrame(
        {
            "candidate_defrost_time": pd.date_range("2026-01-01", periods=4, freq="10s"),
            "cycle_cop": [2.0, 2.01, 2.005, 2.001],
            "cycle_cop_eligible": True,
        }
    )
    result = add_single_objective_optima(
        rows, names=("cycle_cop",), selection_mask=pd.Series([True, False, False, True])
    )
    assert result.cycle_cop_t_star.eq(rows.candidate_defrost_time.iloc[3]).all()
    assert result.cycle_cop_eligible.all()
    assert result.cycle_cop_basin_1pct_start.eq(rows.candidate_defrost_time.iloc[0]).all()


def test_current_prefix_needs_only_three_input_channels_and_heat_electricity(monkeypatch):
    from defrost_decision import candidate_quantities as quantities

    start = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=61, freq="s"),
            "power_total": 2.0,
            "compressor_frequency": 60.0,
            "ambient_temperature": 5.0,
            "heating_capacity_processed": 6.0,
            "coil_temperature_processed": -2.0,
            "cycle_stage": "frost_development",
        }
    )

    def evaluate(values, candidates, recovery, heating, models, experiment, **kwargs):
        assert recovery == heating == start
        assert candidates == [frame.timestamp.max()]
        return pd.DataFrame({"cycle_cop": [2.0], "cycle_cop_eligible": [True]})

    monkeypatch.setattr(quantities, "effective_candidate_cop", evaluate)
    result = quantities.current_effective_cop(
        frame, {"recovery_settings": {"recovery_rule": "dataset"}}, "new"
    )
    assert result["status"] == "supported"


def test_production_cost_counts_measured_recovery_and_rejects_negative_phase():
    from defrost_decision.candidate_quantities import apply_current_cycle_cost

    start = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame(
        dict(timestamp=pd.date_range(start, periods=181, freq="s"), power_total=6.0)
    )
    values = pd.DataFrame(dict(candidate_defrost_time=[start + pd.Timedelta(minutes=3)]))
    for name in [
        "compressor_frequency",
        "evaporating_pressure",
        "water_in_temperature",
        "water_out_temperature",
        "ambient_temperature",
        "coil_temperature",
    ]:
        values[name], values[name + "_valid_second_count"] = 1.0, 60
    parameters = dict(
        feature_order=["compressor_frequency"],
        imputer_median=[1.0],
        scaler_mean=[1.0],
        scaler_scale=[1.0],
        coefficients=[0.0],
        intercept=0.02,
        training_standardized_references=[[0.0]],
        support_threshold=1.0,
    )
    phases = {"E_prep_kwh": parameters, "E_D_kwh": dict(parameters, intercept=0.05)}
    result = apply_current_cycle_cost(values, frame, start, phases)
    np.testing.assert_allclose(result.pre_defrost_electricity_kwh, 0.3)
    np.testing.assert_allclose(result.defrost_event_electricity_kwh, 0.07)
    assert result.defrost_event_electricity_prediction_available.all()
    joint = apply_current_cycle_cost(
        values.drop(
            columns=[
                "evaporating_pressure",
                "water_in_temperature",
                "water_out_temperature",
                "ambient_temperature",
                "coil_temperature",
            ]
        ),
        frame,
        start,
        {"E_prep_D_kwh": dict(parameters, intercept=0.07)},
    )
    np.testing.assert_allclose(
        joint.defrost_event_electricity_kwh, result.defrost_event_electricity_kwh
    )
    assert joint.defrost_event_electricity_prediction_available.all()
    phases["E_prep_kwh"] = dict(parameters, intercept=-0.02)
    assert not apply_current_cycle_cost(
        values, frame, start, phases
    ).defrost_event_electricity_prediction_available.any()
