from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import fit_defrost_event_models as fit_command
from defrost_event_models.ridge_models import load_defrost_event_models


def _events() -> pd.DataFrame:
    rows = []
    for group_index, experiment in enumerate(("a", "b", "c", "d")):
        for index in range(2):
            value = float(group_index + index)
            rows.append(
                {
                    "event_id": f"{experiment}_{index}",
                    "cycle_name": f"{experiment}_{index}",
                    "experiment_id": experiment,
                    "event_valid": True,
                    "energy_event_valid": True,
                    "heat_event_valid": True,
                    "compressor_event_valid": True,
                    "duration_event_valid": True,
                    "event_invalid_reason": "",
                    "water_in_temperature": 35 + value,
                    "water_out_temperature": 40 + value,
                    "coil_temperature": -8 + value,
                    "evaporating_pressure": 0.4 + value / 100,
                    "water_temperature_setpoint": 50.0,
                    "ambient_temperature": 2 + value,
                    "mean_water_temperature": 37.5 + value,
                    "setpoint_outlet_difference": 10 - value,
                    "ambient_coil_difference": 10.0,
                    "compressor_frequency": 70 + value,
                    "heating_elapsed_minutes": 15 + value,
                    "evaporating_pressure_slope_5m": -0.01 + value / 1000,
                    "defrost_event_electricity_observed_kwh": 0.3 + value / 100,
                    "defrost_event_net_heat_observed_kwh": -0.1 + value / 100,
                    "defrost_event_compressor_electricity_observed_kwh": 0.2 + value / 100,
                    "defrost_event_duration_observed_minutes": 12 + value,
                }
            )
    rows.append(
        {
            "event_id": "excluded",
            "cycle_name": "excluded",
            "experiment_id": "a",
            "event_valid": False,
            "energy_event_valid": False,
            "heat_event_valid": False,
            "compressor_event_valid": False,
            "duration_event_valid": False,
            "event_invalid_reason": "missing_defrost_preparation_start",
        }
    )
    result = pd.DataFrame(rows)
    result.loc[result["event_id"].eq("a_0"), ["event_valid", "heat_event_valid"]] = False
    result.loc[result["event_id"].eq("a_0"), "defrost_event_net_heat_observed_kwh"] = float("nan")
    result.loc[result["event_id"].eq("b_0"), "compressor_event_valid"] = False
    result.loc[
        result["event_id"].eq("b_0"), "defrost_event_compressor_electricity_observed_kwh"
    ] = float("nan")
    return result


def test_fit_writes_only_review_candidate_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fit_command, "DatasetLoader", lambda _: object())
    monkeypatch.setattr(fit_command, "build_defrost_event_training_table", lambda _: _events())

    status = fit_command.main(
        [
            "--run-name",
            "review_a",
            "--workers",
            "1",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert status == 0
    run = tmp_path / "defrost_event_models" / "review_a"
    assert {path.name for path in run.iterdir()} == {
        "run_settings.json",
        "defrost_events.csv",
        "model_validation.csv",
        "candidate_model_parameters.json",
    }
    artifact = json.loads((run / "candidate_model_parameters.json").read_text())
    assert (run / "candidate_model_parameters.json").read_text() == json.dumps(
        artifact, sort_keys=True, allow_nan=False, separators=(",", ":")
    )
    assert artifact["run_name"] == "review_a"
    assert artifact["training_cohort_rule"] == "complete_case_across_all_four_outcomes"
    model_names = {
        "experiment_balanced_mean",
        "ridge_basic_state_5",
        "ridge_physical_state_6",
        "ridge_dynamic_state_8",
    }
    assert set(artifact["models"]) == model_names
    for model in artifact["models"].values():
        assert set(model) == {
            "event_electricity",
            "event_net_heat",
            "event_compressor_electricity",
            "event_duration",
        }
    dynamic = artifact["models"]["ridge_dynamic_state_8"]
    assert {model["full_data_model"]["training_event_count"] for model in dynamic.values()} == {6}
    assert (
        len({tuple(model["full_data_model"]["training_event_ids"]) for model in dynamic.values()})
        == 1
    )
    promoted = load_defrost_event_models()
    assert set(promoted["models"]) == model_names
    assert set(promoted["models"]["ridge_dynamic_state_8"]) == {
        "event_electricity",
        "event_net_heat",
        "event_compressor_electricity",
        "event_duration",
    }
    for model_name in model_names - {"experiment_balanced_mean"}:
        for target in (
            "event_electricity",
            "event_net_heat",
            "event_compressor_electricity",
            "event_duration",
        ):
            folds = promoted["models"][model_name][target]["folds"]
            assert folds
            assert all(fold["training_standardized_references"] for fold in folds.values())
            assert all("support_threshold" in fold for fold in folds.values())
    validation = pd.read_csv(run / "model_validation.csv")
    assert len(validation) == 8 * 4 + 1
    assert validation["common_training_event_count"].eq(6).all()
    assert validation["available_event_count_event_electricity"].eq(8).all()
    assert validation["available_event_count_event_net_heat"].eq(7).all()
    assert validation["available_event_count_event_compressor_electricity"].eq(7).all()
    assert validation["available_event_count_event_duration"].eq(8).all()

    assert (
        fit_command.main(
            [
                "--run-name",
                "review_a",
                "--workers",
                "1",
                "--output-root",
                str(tmp_path),
                "--overwrite",
            ]
        )
        == 0
    )
    assert {path.name for path in run.iterdir()} == {
        "run_settings.json",
        "defrost_events.csv",
        "model_validation.csv",
        "candidate_model_parameters.json",
    }


def test_fit_overwrite_does_not_reject_directory_with_stale_members(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = tmp_path / "defrost_event_models" / "review_a"
    run.mkdir(parents=True)
    (run / "stale.txt").write_text("old")
    monkeypatch.setattr(fit_command, "DatasetLoader", lambda _: object())
    monkeypatch.setattr(
        fit_command, "build_defrost_event_training_table", lambda _: _events().iloc[0:0]
    )

    with pytest.raises(ValueError, match="no valid observed defrost events"):
        fit_command.main(
            [
                "--run-name",
                "review_a",
                "--output-root",
                str(tmp_path),
                "--overwrite",
            ]
        )
