from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _raw_frame(start: pd.Timestamp, periods: int = 601) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq="s")
    elapsed = np.arange(periods) / 60
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "power_total": 2.0,
            "compressor_power": 1.0,
            "heating_capacity": 6.0,
            "water_flow": 1.0,
            "water_in_temperature": 35.0,
            "water_out_temperature": 40.0,
            "coil_temperature": -8.0,
            "evaporating_pressure": 0.4 + 0.01 * elapsed,
            "water_temperature_setpoint": 50.0,
            "ambient_temperature": 2.0,
            "compressor_frequency": 70.0,
        }
    )


def test_half_open_integral_excludes_right_boundary_and_does_not_bridge_gap() -> None:
    from defrost_event_models.training_data import window_audit

    start = pd.Timestamp("2026-01-01")
    frame = _raw_frame(start, 61)
    frame.loc[60, "power_total"] = 100.0
    end = start + pd.Timedelta(seconds=60)
    audit = window_audit(frame, start, end, "power_total")
    assert audit["energy"] == pytest.approx(2 * 60 / 3600)

    gapped = frame.drop(index=range(11, 50))
    audit = window_audit(gapped, start, end, "power_total")
    assert audit["energy"] == pytest.approx(2 * 20 / 3600)
    assert not audit["valid"]


def test_candidate_measurement_reconstruction_is_explicit_and_traceable() -> None:
    from defrost_decision.candidate_quantities import build_measured_candidate_quantities

    start = pd.Timestamp("2026-01-01")
    frame = _raw_frame(start, 1201).drop(index=range(700, 820))

    class Loader:
        def get_cycle_record(self, _: str) -> dict[str, object]:
            return {
                "experiment_id": "experiment",
                "boundaries": {
                    "heating_start": start,
                    "defrost_preparation_start": start + pd.Timedelta(minutes=20),
                },
            }

        def load_cycle_original(self, _: str, *, columns: list[str]) -> pd.DataFrame:
            return frame[columns]

    strict = build_measured_candidate_quantities(Loader(), "cycle")
    reconstructed = build_measured_candidate_quantities(
        Loader(), "cycle", allow_measurement_reconstruction=True
    )

    assert not strict.pre_defrost_electricity_measurement_valid.iloc[-1]
    assert reconstructed.pre_defrost_electricity_measurement_valid.iloc[-1]
    assert reconstructed.pre_defrost_electricity_uses_measurement_reconstruction.iloc[-1]
    assert reconstructed.pre_defrost_heat_uses_measurement_reconstruction.iloc[-1]
    assert reconstructed.pre_defrost_compressor_uses_measurement_reconstruction.iloc[-1]


def test_candidate_before_accounting_start_retains_missing_causal_integrals() -> None:
    from defrost_decision.candidate_quantities import build_measured_candidate_quantities

    start = pd.Timestamp("2026-01-01")
    frame = _raw_frame(start, 61)

    class Loader:
        def get_cycle_record(self, _: str) -> dict[str, object]:
            return {
                "experiment_id": "experiment",
                "boundaries": {
                    "heating_start": start,
                    "defrost_preparation_start": start + pd.Timedelta(seconds=9),
                },
            }

        def load_cycle_original(self, _: str, *, columns: list[str]) -> pd.DataFrame:
            return frame[columns]

    candidates = pd.DataFrame({
        "cycle_name": ["cycle"], "experiment_id": ["experiment"],
        "candidate_defrost_time": [start + pd.Timedelta(seconds=9)],
        "minutes_since_heating_start": [.15],
        "heating_accounting_start": [start + pd.Timedelta(minutes=9)],
        "heating_accounting_start_rule": ["fixed_post_defrost_9min"],
        "heating_start": [start],
        "observed_defrost_preparation_start": [start + pd.Timedelta(seconds=9)],
    })

    result = build_measured_candidate_quantities(
        Loader(), "cycle", candidates, allow_measurement_reconstruction=True
    ).iloc[0]

    assert pd.isna(result.online_pre_defrost_electricity_kwh)
    assert not result.online_pre_defrost_electricity_measurement_valid
    assert pd.isna(result.pre_defrost_heat_kwh)
    assert not result.pre_defrost_heat_measurement_valid


def test_features_are_strictly_pre_action_and_require_counts() -> None:
    from defrost_event_models.training_data import extract_pre_defrost_features

    start = pd.Timestamp("2026-01-01")
    frame = _raw_frame(start)
    tau = start + pd.Timedelta(minutes=10)
    frame.loc[frame["timestamp"].eq(tau), "water_in_temperature"] = 999
    result = extract_pre_defrost_features(frame, [tau], start).iloc[0]
    assert result["water_in_temperature"] == 35.0
    assert result["evaporating_pressure_slope_5m"] == pytest.approx(0.01)
    assert bool(result["pre_defrost_feature_window_valid"])

    sparse = frame.drop(index=range(540, 553))
    assert not bool(
        extract_pre_defrost_features(sparse, [tau], start).iloc[0][
            "pre_defrost_feature_window_valid"
        ]
    )


def test_signed_water_heat_is_not_clipped() -> None:
    from defrost_event_models.training_data import window_audit

    start = pd.Timestamp("2026-01-01")
    frame = _raw_frame(start, 61)
    frame["water_out_temperature"] = 34.0
    result = window_audit(frame, start, start + pd.Timedelta(seconds=60), "water_heat")
    assert result["energy"] < 0


def test_window_maximum_gap_includes_leading_window_gap() -> None:
    from defrost_event_models.training_data import window_audit

    start = pd.Timestamp("2026-01-01")
    frame = _raw_frame(start, 61).iloc[25:]

    result = window_audit(frame, start, start + pd.Timedelta(seconds=60), "power_total")

    assert result["maximum_gap_seconds"] == 25


def test_fixed9_candidates_start_at_heating_plus_ten_and_keep_exact_end() -> None:
    from defrost_event_models.training_data import build_candidate_boundaries

    heating = pd.Timestamp("2026-01-01")
    preparation = heating + pd.Timedelta(minutes=12, seconds=30)
    result = build_candidate_boundaries("cycle", "experiment", heating, preparation)
    assert result["heating_accounting_start"].iloc[0] == heating + pd.Timedelta(minutes=9)
    assert result["candidate_defrost_time"].tolist() == [
        heating + pd.Timedelta(minutes=10),
        heating + pd.Timedelta(minutes=11),
        heating + pd.Timedelta(minutes=12),
        preparation,
    ]

    ten_seconds = build_candidate_boundaries(
        "cycle", "experiment", heating, preparation - pd.Timedelta(seconds=5), step_seconds=10
    )
    assert ten_seconds["candidate_defrost_time"].iloc[:3].tolist() == [
        heating + pd.Timedelta(minutes=10),
        heating + pd.Timedelta(minutes=10, seconds=10),
        heating + pd.Timedelta(minutes=10, seconds=20),
    ]
    assert ten_seconds["candidate_defrost_time"].iloc[-1] == preparation - pd.Timedelta(seconds=5)










def test_event_compressor_audit_has_independent_validity() -> None:
    from defrost_event_models.training_data import measure_defrost_event_quantities

    start = pd.Timestamp("2026-01-01")
    current = _raw_frame(start, 120)
    recovery = _raw_frame(start + pd.Timedelta(minutes=2), 60)
    boundaries = {
        "preparation_start": start,
        "defrost_start": start + pd.Timedelta(minutes=1),
        "defrost_end": start + pd.Timedelta(minutes=2),
        "recovery_end": start + pd.Timedelta(minutes=3),
    }

    valid = measure_defrost_event_quantities(current, recovery, **boundaries)
    assert valid["E_comp_prep_kwh"] == pytest.approx(1 / 60)
    assert valid["defrost_event_compressor_electricity_observed_kwh"] == pytest.approx(3 / 60)
    assert valid["defrost_event_duration_observed_minutes"] == 3
    assert valid["event_valid"]
    assert valid["energy_event_valid"]
    assert valid["heat_event_valid"]
    assert valid["compressor_event_valid"]
    assert valid["duration_event_valid"]

    current.loc[current["timestamp"].ge(boundaries["defrost_start"]), "compressor_power"] = np.nan
    invalid = measure_defrost_event_quantities(current, recovery, **boundaries)
    assert invalid["event_valid"]
    assert invalid["energy_event_valid"]
    assert invalid["heat_event_valid"]
    assert not invalid["compressor_event_valid"]
    assert invalid["duration_event_valid"]
    assert np.isnan(invalid["defrost_event_compressor_electricity_observed_kwh"])


def test_compressor_event_valid_requires_phase_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from defrost_event_models import training_data as module

    monkeypatch.setattr(
        module,
        "window_audit",
        lambda *_args: {
            "energy": 1.0,
            "coverage": 1.0,
            "maximum_gap_seconds": 1.0,
            "start_fresh": True,
            "end_fresh": True,
            "valid": True,
        },
    )
    start = pd.Timestamp("2026-01-01")
    result = module.measure_defrost_event_quantities(
        pd.DataFrame(),
        pd.DataFrame(),
        preparation_start=start,
        defrost_start=start + pd.Timedelta(minutes=2),
        defrost_end=start + pd.Timedelta(minutes=1),
        recovery_end=start + pd.Timedelta(minutes=3),
    )

    assert not result["phase_partition_valid"]
    assert not result["event_valid"]
    assert not result["compressor_event_valid"]
    assert np.isnan(result["defrost_event_compressor_electricity_observed_kwh"])


def test_event_audit_retains_any_defrost_with_missing_preparation() -> None:
    from defrost_event_models.training_data import build_defrost_event_training_table

    class Loader:
        def list_cycles(self, **_: object) -> pd.DataFrame:
            start = pd.Timestamp("2026-01-01")
            return pd.DataFrame(
                [
                    {
                        "cycle_name": "arbitrary_defrost_name",
                        "experiment_id": "exp",
                        "status": "valid",
                        "start_time": start,
                        "heating_start": start,
                        "defrost_preparation_start": pd.NaT,
                        "defrost_start": start + pd.Timedelta(minutes=10),
                        "defrost_end": start + pd.Timedelta(minutes=11),
                    },
                    {
                        "cycle_name": "following",
                        "experiment_id": "exp",
                        "status": "invalid",
                        "start_time": start + pd.Timedelta(minutes=11),
                        "heating_start": start + pd.Timedelta(minutes=11),
                        "defrost_preparation_start": pd.NaT,
                        "defrost_start": pd.NaT,
                        "defrost_end": pd.NaT,
                    },
                ]
            )

    result = build_defrost_event_training_table(Loader())
    assert result["cycle_name"].tolist() == ["arbitrary_defrost_name"]
    assert result["event_invalid_reason"].iloc[0] == "missing_defrost_preparation_start"
    assert pd.notna(result["defrost_start"].iloc[0])
    assert pd.notna(result["defrost_end"].iloc[0])


def test_candidate_cohort_excludes_experiments_without_parameter_folds() -> None:
    from defrost_event_models.training_data import candidate_cohort

    start = pd.Timestamp("2026-01-01")

    class Loader:
        def list_cycles(self, **_: object) -> pd.DataFrame:
            rows = []
            for index, experiment in enumerate(("with_fold", "missing_fold")):
                heating = start + pd.Timedelta(hours=index)
                rows.append(
                    {
                        "cycle_name": experiment,
                        "experiment_id": experiment,
                        "status": "valid",
                        "start_time": heating,
                        "heating_start": heating,
                        "stable_heating_start": heating + pd.Timedelta(minutes=9),
                        "defrost_preparation_start": heating + pd.Timedelta(minutes=15),
                        "defrost_start": heating + pd.Timedelta(minutes=16),
                        "defrost_end": heating + pd.Timedelta(minutes=18),
                    }
                )
            return pd.DataFrame(rows)

        def load_cycle_original(
            self, cycle_name: str, *, columns: list[str] | None = None
        ) -> pd.DataFrame:
            heating = start + pd.Timedelta(hours=int(cycle_name == "missing_fold"), minutes=9)
            frame = pd.DataFrame(
                {
                    "timestamp": pd.date_range(heating, periods=60, freq="s"),
                    "water_flow": 1.0,
                    "water_in_temperature": 40.0,
                    "water_out_temperature": 45.0,
                    "power_total": 2.0,
                }
            )
            return frame if columns is None else frame[columns]

    assert candidate_cohort(Loader(), {"with_fold"}) == (["with_fold"], 6)
