from __future__ import annotations

import pandas as pd
import pytest

from defrost_decision.baselines.unit_heat_inverse_cop_v1 import DEFAULT_RECIPE as V1_RECIPE
from defrost_decision.baselines.unit_heat_inverse_cop_v1 import calculate_cycle as calculate_v1
from defrost_decision.baselines.water_heat_inverse_cop_v25 import DEFAULT_RECIPE as V25_RECIPE
from defrost_decision.baselines.water_heat_inverse_cop_v25 import calculate_cycle as calculate_v25


class FakeDataset:
    def __init__(self) -> None:
        timestamps = pd.date_range("2026-01-01", periods=1501, freq="s")
        self.frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "power_total": 6.0,
                "heating_capacity": 12.0,
                "water_flow": 1.0,
                "water_in_temperature": 45.0,
                "water_out_temperature": 50.0,
                "water_temperature_setpoint": 50.0,
                "coil_temperature": -10.0,
                "evaporating_pressure": 0.3,
            }
        )
        self.frame.loc[self.frame["timestamp"].ge("2026-01-01 00:19:00"), "coil_temperature"] = -10
        self.frame.loc[self.frame["timestamp"].ge("2026-01-01 00:23:20"), "coil_temperature"] = 20
        self.record = {
            "cycle_name": "cycle_a",
            "experiment_id": "exp_20260714",
            "boundaries": {
                "heating_start": "2026-01-01 00:00:00",
                "stable_heating_start": "2026-01-01 00:02:00",
                "defrost_preparation_start": "2026-01-01 00:18:30",
                "defrost_start": "2026-01-01 00:19:00",
                "defrost_end": "2026-01-01 00:24:00",
            },
        }

    def get_cycle_record(self, cycle_name: str) -> dict[str, object]:
        assert cycle_name == "cycle_a"
        return self.record

    def load_cycle_original(
        self, cycle_name: str, *, columns: list[str] | None = None
    ) -> pd.DataFrame:
        assert cycle_name == "cycle_a"
        return self.frame if columns is None else self.frame[columns]


def test_default_recipes_are_canonical_and_explicit() -> None:
    assert V1_RECIPE == {
        "base_cost": "v1",
        "version": "v1",
        "run_name": None,
        "label_eligible": True,
        "heat_basis": "unit",
        "event_scope": "stable_heating_start_to_actual_preparation",
        "heating_start_rule": "stable_heating_start",
        "integration_protocol": "historical_reconstruction",
        "state_protocol": "historical_interpolation",
        "candidate_start_rule": "stable_heating_start_plus_10_minutes",
        "candidate_end_rule": "observed_defrost_preparation_start",
        "candidate_cadence": "1_minute_plus_exact_endpoint",
        "state_window": "[tau-60s,tau)",
        "transition_scope": "preparation_defrost_recovery",
        "transition_window": "candidate_state_at_tau",
        "transition_provenance": "candidate_defrost_time_state_plus_fixed_recovery",
        "decision_rule": "supported_argmin_inverse_cop",
        "heating_energy_model": "measured_total_power",
        "heating_heat_model": "measured_unit_heat",
        "transition_energy_model": "pe_quadratic_plus_fixed_recovery",
        "transition_heat_model": "zero_transition_heat",
    }
    assert V25_RECIPE["base_cost"] == "v2.5"
    assert V25_RECIPE["heat_basis"] == "water"
    assert V25_RECIPE["event_scope"] == "heating_start_to_actual_preparation"
    assert V25_RECIPE["integration_protocol"] == "historical_reconstruction"
    assert V25_RECIPE["state_protocol"] == "historical_interpolation"
    assert V25_RECIPE["candidate_start_rule"] == "stable_heating_start_plus_10_minutes"
    assert V25_RECIPE["candidate_end_rule"] == "observed_defrost_preparation_start"
    assert V25_RECIPE["candidate_cadence"] == "1_minute_plus_exact_endpoint"
    assert V25_RECIPE["state_window"] == "[tau-60s,tau)"
    assert V25_RECIPE["decision_rule"] == "supported_argmin_inverse_cop"
    assert V25_RECIPE["label_eligible"] is False
    assert "observed_preparation_and_defrost_durations" in str(V25_RECIPE["transition_window"])
    assert "offline_diagnostic" in str(V25_RECIPE["transition_provenance"])


def test_version_module_rejects_recipe_from_another_base_cost() -> None:
    with pytest.raises(ValueError, match="V1 module"):
        calculate_v1(FakeDataset(), "cycle_a", V25_RECIPE)
    with pytest.raises(ValueError, match="V2.5 module"):
        calculate_v25(FakeDataset(), "cycle_a", V1_RECIPE)


def test_v1_single_cycle_uses_stable_unit_heat_fixed_recovery_and_zero_qt() -> None:
    result = calculate_v1(FakeDataset(), "cycle_a")
    first = result.iloc[0]

    assert not result["is_censored"].any()
    assert first["minutes_since_heating_start"] == 10.0
    assert first["pre_defrost_electricity_kwh"] == pytest.approx(1.0)
    assert first["pre_defrost_heat_kwh"] == pytest.approx(2.0)
    assert first["preparation_energy_kwh"] == 0.0
    assert first["recovery_energy_kwh"] == 0.279901897467
    assert first["defrost_event_net_heat_kwh"] == 0.0
    assert first["base_cost"] == "v1"
    assert first["label_eligible"]
    assert first["heating_energy_rule"] == (
        "offline_historical_reconstruction_bridged_internal_gaps_"
        "endpoint_extrapolation_from_stable_heating_start"
    )
    assert first["heating_heat_rule"] == first["heating_energy_rule"]
    assert first["transition_energy_rule"] == ("offline_historical_interpolation_[tau-60s,tau)")
    assert {
        "preparation_energy_kwh",
        "defrost_energy_kwh",
        "recovery_energy_kwh",
        "preparation_heat_kwh",
        "defrost_heat_kwh",
        "recovery_heat_kwh",
    } <= set(result)
    assert result["supported"].all()
    assert result["is_optimum"].sum() == 1


def test_v25_single_cycle_uses_cycle_start_water_heat_and_signed_qd() -> None:
    result = calculate_v25(FakeDataset(), "cycle_a")
    first = result.iloc[0]

    assert first["minutes_since_heating_start"] == 12.0
    assert first["pre_defrost_electricity_kwh"] == pytest.approx(1.2)
    assert first["pre_defrost_heat_kwh"] == pytest.approx(1.161)
    assert first["preparation_energy_kwh"] == 0.0
    assert first["recovery_energy_kwh"] == 0.0
    assert first["recovery_heat_kwh"] == 0.0
    assert first["defrost_heat_kwh"] <= 0.0
    assert first["defrost_event_net_heat_kwh"] == (
        first["preparation_heat_kwh"] + first["defrost_heat_kwh"]
    )
    assert first["base_cost"] == "v2.5"
    assert not first["label_eligible"]
    assert first["heating_energy_rule"] == (
        "offline_historical_reconstruction_stable_block_bridged_internal_gaps_"
        "endpoint_extrapolation_plus_bridged_observed_heating_start_prefix"
    )
    assert first["heating_heat_rule"] == first["heating_energy_rule"]
    assert first["transition_energy_rule"] == ("offline_historical_interpolation_[tau-60s,tau)")
    assert first["transition_heat_rule"] == first["transition_energy_rule"]
    assert result["is_optimum"].sum() == 1


def test_strict_mixed_variant_is_causal_and_label_ineligible() -> None:
    recipe = dict(V1_RECIPE)
    recipe.update(run_name="strict_state", state_protocol="strict_causal")
    result = calculate_v1(FakeDataset(), "cycle_a", recipe)

    assert not result["label_eligible"].any()
    assert result["heating_energy_rule"].str.startswith("offline_historical").all()
    assert result["transition_energy_rule"].eq("strict_causal_[tau-60s,tau)").all()


def test_strict_integration_variant_threads_through_v1_pipeline() -> None:
    dataset = FakeDataset()
    gap = dataset.frame["timestamp"].between(
        "2026-01-01 00:03:00",
        "2026-01-01 00:03:10",
    )
    dataset.frame = dataset.frame.loc[~gap]
    recipe = dict(V1_RECIPE)
    recipe.update(run_name="strict_integration", integration_protocol="strict_causal")

    result = calculate_v1(dataset, "cycle_a", recipe)

    assert not result["label_eligible"].any()
    assert (
        result["heating_energy_rule"]
        .eq("strict_causal_gap_aware_5s_from_stable_heating_start")
        .all()
    )
    assert result["pre_defrost_electricity_kwh"].equals(
        result["strict_pre_defrost_electricity_kwh"]
    )
    assert result["pre_defrost_heat_kwh"].equals(result["strict_pre_defrost_heat_kwh"])


def test_v25_strict_state_variant_threads_through_transition_heat() -> None:
    recipe = dict(V25_RECIPE)
    recipe.update(run_name="strict_state", state_protocol="strict_causal")

    result = calculate_v25(FakeDataset(), "cycle_a", recipe)

    assert not result["label_eligible"].any()
    assert result["transition_energy_rule"].eq("strict_causal_[tau-60s,tau)").all()
    assert result["transition_heat_rule"].eq("strict_causal_[tau-60s,tau)").all()
    assert result["defrost_event_electricity_kwh"].equals(
        result["strict_defrost_event_electricity_kwh"]
    )
    assert result["defrost_event_net_heat_kwh"].equals(result["strict_defrost_event_net_heat_kwh"])
    assert result["defrost_event_net_heat_evaluable"].equals(result["strict_QT_supported"])
    assert "QT_supported" not in result
