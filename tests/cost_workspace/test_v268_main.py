from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import main_cost
from cost.fit_v2_6_8 import load_artifacts


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
                    "E_T_observed_kwh": 0.3 + value / 100,
                    "Q_T_observed_kwh": -0.1 + value / 100,
                }
            )
    rows.append(
        {
            "event_id": "excluded",
            "cycle_name": "excluded",
            "experiment_id": "a",
            "event_valid": False,
            "event_invalid_reason": "missing_defrost_preparation_start",
        }
    )
    return pd.DataFrame(rows)


def test_fit_writes_only_review_candidate_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: object())
    monkeypatch.setattr(main_cost, "build_event_table", lambda _: _events())
    monkeypatch.setattr(main_cost, "candidate_cohort", lambda *_: (["a_0"], 6))
    monkeypatch.setattr(
        main_cost.cost_function_v2_6_8,
        "calculate_cycle",
        lambda *_: pd.DataFrame(
            {
                "cycle_name": "a_0",
                "experiment_id": "a",
                "candidate_time": pd.date_range("2026-01-01", periods=6, freq="min"),
                "heating_energy_kwh": 1.0,
                "heating_heat_kwh": 2.0,
                "heating_measurement_valid": True,
                "pre_action_window_valid": True,
            }
        ),
    )
    monkeypatch.setattr(
        main_cost,
        "bootstrap_minima",
        lambda *_: pd.DataFrame({"cycle_name": ["a_0"], "repeat_count": [200], "seed": [268]}),
    )

    status = main_cost.main(
        [
            "--action",
            "fit",
            "--cost",
            "v2.6.8",
            "--variant",
            "review_a",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert status == 0
    run = tmp_path / "cost" / "fit" / "review_a"
    assert {path.name for path in run.iterdir()} == {
        "command.txt",
        "recipe.json",
        "events.csv",
        "validation.csv",
        "bootstrap.csv",
        "params_candidate.json",
    }
    artifact = json.loads((run / "params_candidate.json").read_text())
    recipe = json.loads((run / "recipe.json").read_text())
    assert recipe == main_cost.cost_function_v2_6_8.DEFAULT_RECIPE
    assert artifact["fit_variant"] == "review_a"
    model_names = {
        "experiment_mean",
        "ticket_ridge_static5",
        "ticket_ridge_physical6",
        "ticket_ridge_dynamic8",
    }
    assert set(artifact["models"]) == model_names
    promoted = load_artifacts()
    assert set(promoted["models"]) == model_names
    for model_name in model_names - {"experiment_mean"}:
        for target in ("energy", "heat"):
            folds = promoted["models"][model_name][target]["folds"]
            assert folds
            assert all(fold["training_standardized_references"] for fold in folds.values())
            assert all("support_threshold" in fold for fold in folds.values())
    validation = pd.read_csv(run / "validation.csv")
    assert len(validation) == 8 * 4 + 1

    assert (
        main_cost.main(
            [
                "--action",
                "fit",
                "--cost",
                "v2.6.8",
                "--variant",
                "review_a",
                "--output-root",
                str(tmp_path),
                "--overwrite",
            ]
        )
        == 0
    )
    assert {path.name for path in run.iterdir()} == main_cost.FIT_ARTIFACT_NAMES


def test_fit_overwrite_rejects_directory_with_stale_members(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = tmp_path / "cost" / "fit" / "review_a"
    run.mkdir(parents=True)
    (run / "stale.txt").write_text("old")
    monkeypatch.setattr(
        main_cost,
        "DatasetLoader",
        lambda _: pytest.fail("stale directory must fail before loading Dataset"),
    )

    with pytest.raises(FileExistsError, match="unexpected member.*stale.txt"):
        main_cost.main(
            [
                "--action",
                "fit",
                "--cost",
                "v2.6.8",
                "--variant",
                "review_a",
                "--output-root",
                str(tmp_path),
                "--overwrite",
            ]
        )
