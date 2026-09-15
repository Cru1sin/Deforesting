import numpy as np
import pandas as pd
import pytest

from defrost_event_models.training_data import measure_defrost_event_quantities


def test_effective_event_retains_all_electricity_and_only_optional_preparation_heat():
    t = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(t, periods=241, freq="s"),
            "power_total": 2.0,
            "compressor_power": 1.0,
            "heating_capacity": 6.0,
            "water_flow": 1.0,
            "water_in_temperature": 20.0,
            "water_out_temperature": -20.0,
        }
    )
    kwargs = dict(
        preparation_start=t,
        defrost_start=t + pd.Timedelta(seconds=60),
        defrost_end=t + pd.Timedelta(seconds=120),
        recovery_end=t + pd.Timedelta(seconds=240),
    )
    for mode, expected in [("include", 0.1), ("zero", 0)]:
        result = measure_defrost_event_quantities(frame, frame, preparation_heat=mode, **kwargs)
        assert result["defrost_event_electricity_observed_kwh"] == pytest.approx(2 * 240 / 3600)
        assert result["defrost_event_net_heat_observed_kwh"] == pytest.approx(expected)
        assert result["Q_D_kwh"] == result["Q_R_kwh"] == 0
        assert result["energy_event_valid"] and result["heat_event_valid"]
    frame["heating_capacity"] = np.nan
    result = measure_defrost_event_quantities(frame, frame, preparation_heat="zero", **kwargs)
    assert result["energy_event_valid"] and result["heat_event_valid"]


@pytest.mark.parametrize("mode", ["zero", "include"])
def test_zero_length_preparation_is_a_valid_empty_half_open_phase(mode):
    start = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=121, freq="s"),
            "power_total": 2.0,
            "compressor_power": 1.0,
            "heating_capacity": 6.0,
            "water_flow": 1.0,
            "water_in_temperature": 20.0,
            "water_out_temperature": 25.0,
        }
    )

    result = measure_defrost_event_quantities(
        frame,
        frame,
        preparation_start=start,
        defrost_start=start,
        defrost_end=start + pd.Timedelta(seconds=60),
        recovery_end=start + pd.Timedelta(seconds=120),
        preparation_heat=mode,
    )

    assert result["phase_partition_valid"]
    for quantity in ("E", "Q", "E_comp"):
        assert result[f"{quantity}_prep_kwh"] == 0
        assert result[f"{quantity}_prep_coverage"] == 1
        assert result[f"{quantity}_prep_maximum_gap_seconds"] == 0
        assert result[f"{quantity}_prep_start_fresh"]
        assert result[f"{quantity}_prep_end_fresh"]
        assert result[f"{quantity}_prep_valid"]
    assert result["E_D_kwh"] == pytest.approx(2 * 60 / 3600)
    assert result["E_R_kwh"] == pytest.approx(2 * 60 / 3600)
    assert result["energy_event_valid"] and result["heat_event_valid"]


def test_live_candidate_prefix_and_heat_mode_share_the_same_energy():
    from defrost_decision.candidate_quantities import effective_candidate_cop
    from defrost_decision.performance_objectives import add_single_objective_optima
    from defrost_event_models.ridge_models import mean_outcome_model

    start = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame({"timestamp": pd.date_range(start, periods=1201, freq="s")})
    for name, value in dict(
        power_total=2.0,
        heating_capacity=6.0,
        water_in_temperature=40.0,
        water_out_temperature=45.0,
        water_temperature_setpoint=50.0,
        coil_temperature=-10.0,
        ambient_temperature=-5.0,
        evaporating_pressure=0.3,
        compressor_frequency=80.0,
    ).items():
        frame[name] = value
    events = pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "experiment_id": ["a", "b", "c"],
            "E": [1.0, 1.0, 1.0],
            "Q": [0.1, 0.1, 0.1],
        }
    )
    targets = {
        key: {"full_data_model": mean_outcome_model(events, col)}
        for key, col in [("event_electricity", "E"), ("event_net_heat", "Q")]
    }
    models = {
        "cop_definition": "refrigerant_effective_heat",
        "models": {"ridge_dynamic_state_8": targets},
    }
    now = start + pd.Timedelta(minutes=10)
    kwargs = dict(
        recovery_start=start + pd.Timedelta(minutes=3),
        heating_start=start,
        models=models,
        experiment_id="new",
        prediction_mode="full-model",
    )
    full = effective_candidate_cop(frame, [now], preparation_heat="include", **kwargs)
    prefix = effective_candidate_cop(
        frame[frame.timestamp.lt(now)], [now], preparation_heat="include", **kwargs
    )
    pd.testing.assert_frame_equal(full, prefix)
    zero = effective_candidate_cop(frame, [now], preparation_heat="zero", **kwargs)
    assert full.cycle_cop.iloc[0] > zero.cycle_cop.iloc[0]
    assert full.pre_defrost_electricity_kwh.iloc[0] == zero.pre_defrost_electricity_kwh.iloc[0]
    assert bool(full.cycle_cop_eligible.iloc[0])
    full["cycle_heating_rate_kw"] = -999.0
    selected = add_single_objective_optima(full, names=("cycle_cop",))
    assert selected.cycle_cop_t_star.iloc[0] == now


def test_live_input_without_required_observations_reports_unavailable():
    from dataset_tools.builder.detect_cycles import RECOVERY_DEFAULTS
    from defrost_decision.candidate_quantities import current_effective_cop

    result = current_effective_cop(
        pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")]}),
        {"recovery_settings": RECOVERY_DEFAULTS},
        "new",
    )
    assert result["status"].startswith("missing_channels:")
    assert np.isnan(result["cycle_cop"])


@pytest.mark.parametrize("energy,supported", [(-1.0, True), (0.0, True), (1.0, False)])
def test_nonpositive_or_unsupported_event_energy_cannot_select(energy, supported):
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
    assert not calculate_cycle_cop(frame, effective=True).cycle_cop_eligible.any()


def test_energy_loading_does_not_require_optional_heat_or_compressor_channels():
    from defrost_event_models.training_data import load_frame

    class Loader:
        def load_cycle_original(self, name, columns=None):
            return pd.DataFrame({"timestamp": ["2026-01-01"], "power_total": [2.0]})

    result = load_frame(Loader(), "cycle", {}, extra_columns=("heating_capacity",))
    assert result.power_total.iloc[0] == 2.0
    assert result.heating_capacity.isna().all()
    assert result.compressor_power.isna().all()


def test_validation_mse_keeps_units_and_experiment_denominators():
    from defrost_event_models.validation import summarize_validation

    rows = pd.DataFrame(
        {
            "model_name": ["ridge"] * 3,
            "experiment_id": ["a", "a", "b"],
            "defrost_event_electricity_observed_kwh": [1.0, 2.0, 3.0],
            "defrost_event_electricity_prediction_kwh": [2.0, 2.0, 1.0],
            "event_electricity_in_training_domain": [True, True, False],
        }
    )
    result = summarize_validation(rows).set_index("experiment_id")
    assert result.loc["all", "mse"] == pytest.approx(5 / 3)
    assert result.loc["all", "rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert result.loc["all", "relative_rmse"] == pytest.approx(np.sqrt(5 / 3) / 2)
    assert result.loc["a", "mse"] == 0.5
    assert result.loc["b", "mse"] == 4
    assert result.loc["all", "experiment_macro_mse"] == 2.25
    assert result.loc["all", "supported_mse"] == 0.5
    assert result.loc["all", "in_training_domain_fraction"] == pytest.approx(2 / 3)


def test_observed_cop_heat_comparison_uses_same_complete_energy():
    from types import SimpleNamespace

    from defrost_event_models.training_data import observed_cycle_cop_comparison

    start = pd.Timestamp("2026-01-01")
    loader = SimpleNamespace(
        get_cycle_record=lambda name: {"boundaries": {"stable_heating_start": start}},
        load_cycle_original=lambda name: pd.DataFrame(
            {
                "timestamp": pd.date_range(start, periods=61, freq="s").astype(str),
                "heating_capacity": 6.0,
                "power_total": 2.0,
            }
        ),
    )
    events = pd.DataFrame(
        {
            "cycle_name": ["test"],
            "experiment_id": ["e"],
            "defrost_preparation_start": [start + pd.Timedelta(seconds=60)],
            "defrost_event_electricity_observed_kwh": [2 * 120 / 3600],
            "defrost_event_net_heat_observed_kwh": [0.05],
            "energy_event_valid": [True],
            "heat_event_valid": [True],
        }
    )
    row = observed_cycle_cop_comparison(loader, events).iloc[0]
    assert row.COP_eff_zero == pytest.approx(1.0)
    assert row.COP_eff_include == pytest.approx(1.5)
    assert row.COP_relative_difference_pct == pytest.approx(50.0)

    events["heat_event_valid"] = False
    events["defrost_event_net_heat_observed_kwh"] = np.nan
    missing_heat = observed_cycle_cop_comparison(loader, events).iloc[0]
    assert pd.isna(missing_heat.COP_eff_include)
    assert missing_heat.COP_eff_zero == pytest.approx(1.0)


@pytest.mark.parametrize('status,defrost,available', [
    ('valid', None, True), ('partial', None, True),
    ('invalid', None, False), ('valid', 20, False),
])
def test_hypothetical_defrost_uses_observed_open_heating(status, defrost, available, monkeypatch):
    from types import SimpleNamespace

    import select_defrost_time as decision

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
    loader = SimpleNamespace(
        get_cycle_record=lambda _: record,
        load_cycle_original=lambda _: pd.DataFrame({"timestamp": [start, end]}),
    )
    monkeypatch.setattr(
        decision.rule_based,
        "calculate_cycle",
        lambda *_: {"rb_status": "triggered", "t_RB": start + pd.Timedelta(minutes=25)},
    )
    calls = []

    def evaluate(frame, times, *args, **kwargs):
        calls.append(times)
        return pd.DataFrame(
            {"candidate_defrost_time": times, "cycle_cop": 2.0, "cycle_cop_eligible": True}
        )

    monkeypatch.setattr(decision, "effective_candidate_cop", evaluate)
    result = decision.calculate_cycle(loader, "cycle", {})
    assert bool(calls) == available
    if available:
        assert max(calls[0]) == end
        assert result.cycle_cop_eligible.all()
        assert result.candidate_end_source.eq("recording_end").all()
    assert bounds["defrost_preparation_start"] is None
    assert record["status"] == status


def test_outlet_threshold_integrates_crossings_without_low_temperature_heat():
    from defrost_event_models.training_data import window_audit

    start = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame({
        "timestamp": pd.date_range(start, periods=5, freq="s"),
        "heating_capacity": [2., 4., 6., 8., 10.],
        "power_total": 1.,
        "water_out_temperature": [41., 39., 39., 41., 41.],
    })
    end = start + pd.Timedelta(seconds=4)
    heat = window_audit(frame, start, end, "heating_capacity",
                        minimum_outlet_temperature=40.)
    # 0..0.5: integral(2+2t)=1.25; 2.5..3: 3.75; trailing 3..4: 8.
    assert heat["energy"] == pytest.approx(13 / 3600)
    assert window_audit(frame, start, end, "power_total")["energy"] == pytest.approx(4/3600)
    frame.water_out_temperature = 40.
    assert window_audit(frame, start, end, "heating_capacity",
                        minimum_outlet_temperature=40.)["energy"] == pytest.approx(23/3600)
    frame.water_out_temperature = 39.
    assert window_audit(frame, start, end, "heating_capacity",
                        minimum_outlet_temperature=40.)["energy"] == 0
    assert not window_audit(frame, start, end, "heating_capacity",
                            minimum_outlet_temperature=np.nan)["valid"]
