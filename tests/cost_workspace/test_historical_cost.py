from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main_cost
from cost.cost_function_v1 import calculate_cycle as calculate_v1
from cost.cost_function_v2_5 import calculate_cycle as calculate_v25
from cost.energy_models import load_parameters
from dataloader import DatasetLoader

ORIGINAL_ROOT = Path("/Users/cruisin/Documents/DeforestingSensor")
CYCLE = "frost_cycle_000070"


def _formal_cycles(loader: DatasetLoader) -> list[str]:
    metadata_cycles = main_cost._cycle_names(  # noqa: SLF001
        loader, None, set(load_parameters()["pe_quadratic"])
    )
    selected, _ = main_cost._clean_anchor_cycles(  # noqa: SLF001
        loader, metadata_cycles, explicit=False
    )
    return selected


def _assert_numeric_parity(
    current: pd.DataFrame,
    historical: pd.DataFrame,
    columns: dict[str, str],
) -> None:
    for new, old in columns.items():
        assert np.allclose(
            pd.to_numeric(current[new], errors="coerce"),
            pd.to_numeric(historical[old], errors="coerce"),
            rtol=0,
            atol=5e-13,
            equal_nan=True,
        ), f"{new} differs from formal {old}"


def test_real_dataset_frozen_science_cohort_is_69() -> None:
    dataset = ORIGINAL_ROOT / "dataset"
    if not dataset.exists():
        pytest.skip("read-only original Dataset integration assets are unavailable")
    loader = DatasetLoader(dataset)
    metadata_cycles = main_cost._cycle_names(  # noqa: SLF001
        loader, None, set(load_parameters()["pe_quadratic"])
    )
    selected, excluded = main_cost._clean_anchor_cycles(  # noqa: SLF001
        loader, metadata_cycles, explicit=False
    )

    assert len(metadata_cycles) == 70
    assert len(selected) == 69
    assert excluded == 1
    assert "frost_cycle_000005" not in selected


@pytest.mark.parametrize(
    ("name", "calculate", "heat_column", "optimum"),
    [
        ("v1", calculate_v1, "unit_heating_kwh", pd.Timestamp("2026-07-31 10:19:29")),
        ("v2.5", calculate_v25, "water_heating_kwh", pd.Timestamp("2026-07-31 10:05:29")),
    ],
)
def test_dataset_native_curve_matches_historical_selected_output(
    name: str, calculate: object, heat_column: str, optimum: pd.Timestamp
) -> None:
    dataset = ORIGINAL_ROOT / "dataset"
    historical_path = ORIGINAL_ROOT / "output/成本函数" / f"cost_function_{name}.csv"
    if not dataset.exists() or not historical_path.exists():
        pytest.skip("read-only original Dataset integration assets are unavailable")
    result = calculate(DatasetLoader(dataset), CYCLE)  # type: ignore[operator]
    historical = pd.read_csv(historical_path)
    historical = historical.loc[historical["cycle_name"].eq(CYCLE)].reset_index(drop=True)

    assert len(result) == len(historical) == 107
    assert np.array_equal(
        pd.to_datetime(result["candidate_time"], format="mixed").to_numpy(),
        pd.to_datetime(historical["candidate_time"], format="mixed").to_numpy(),
    )
    comparisons = {
        "heating_energy_kwh": "heating_electricity_kwh",
        "heating_heat_kwh": heat_column,
        "defrost_energy_kwh": "defrost_electricity_kwh",
        "recovery_energy_kwh": "recovery_electricity_kwh",
        "inverse_cop": "inverse_cop",
    }
    if name == "v2.5":
        comparisons.update(
            {
                "preparation_heat_kwh": "preparation_heat_kwh",
                "defrost_heat_kwh": "defrost_heat_kwh",
            }
        )
    for current, old in comparisons.items():
        assert np.allclose(result[current], historical[old], rtol=0, atol=5e-13)
    assert "strict_heating_energy_kwh" in result
    assert "strict_heating_heat_kwh" in result
    assert "strict_ET_supported" in result
    assert result.loc[result["is_optimum"], "candidate_time"].tolist() == [optimum]
    assert pd.Timestamp(historical["t_star"].iloc[0]) == optimum


@pytest.mark.parametrize(
    ("name", "calculate", "heat_column"),
    [
        ("v1", calculate_v1, "unit_heating_kwh"),
        ("v2.5", calculate_v25, "water_heating_kwh"),
    ],
)
def test_all_dataset_native_curves_match_formal_cost_outputs(
    name: str, calculate: object, heat_column: str
) -> None:
    dataset = ORIGINAL_ROOT / "dataset"
    historical_path = ORIGINAL_ROOT / "output/成本函数" / f"cost_function_{name}.csv"
    if not dataset.exists() or not historical_path.exists():
        pytest.skip("read-only original Dataset integration assets are unavailable")
    loader = DatasetLoader(dataset)
    current = pd.concat(
        [calculate(loader, cycle) for cycle in _formal_cycles(loader)],  # type: ignore[operator]
        ignore_index=True,
    )
    historical = pd.read_csv(historical_path).sort_values(
        ["cycle_name", "candidate_time"], kind="stable"
    )
    current = current.sort_values(["cycle_name", "candidate_time"], kind="stable")

    assert len(current) == len(historical) == 7418
    assert current["cycle_name"].tolist() == historical["cycle_name"].tolist()
    assert np.array_equal(
        pd.to_datetime(current["candidate_time"], format="mixed").to_numpy(),
        pd.to_datetime(historical["candidate_time"], format="mixed").to_numpy(),
    )
    numeric = {
        "heating_energy_kwh": "heating_electricity_kwh",
        "heating_heat_kwh": heat_column,
        "defrost_energy_kwh": "defrost_electricity_kwh",
        "recovery_energy_kwh": "recovery_electricity_kwh",
        "inverse_cop": "inverse_cop",
        "relative_regret": "relative_regret",
    }
    if name == "v2.5":
        numeric.update(
            {
                "preparation_heat_kwh": "preparation_heat_kwh",
                "defrost_heat_kwh": "defrost_heat_kwh",
            }
        )
    _assert_numeric_parity(current, historical, numeric)
    expected_et = historical["defrost_electricity_kwh"] + historical["recovery_electricity_kwh"]
    expected_qt = (
        historical["preparation_heat_kwh"] + historical["defrost_heat_kwh"]
        if name == "v2.5"
        else pd.Series(0.0, index=historical.index)
    )
    assert np.allclose(current["transition_energy_kwh"], expected_et, rtol=0, atol=5e-13)
    assert np.allclose(
        current["transition_heat_kwh"],
        expected_qt,
        rtol=0,
        atol=5e-13,
        equal_nan=True,
    )
    assert current["optimization_eligible"].tolist() == historical["optimization_eligible"].tolist()
    assert (
        current["heating_energy_supported"].tolist() == historical["integration_eligible"].tolist()
    )
    assert current["heating_heat_supported"].tolist() == historical["integration_eligible"].tolist()
    assert (
        current["ET_supported"].tolist() == historical["evaporating_pressure_mpa"].notna().tolist()
    )
    expected_qt_support = (
        historical["qprep_eligible"].fillna(False) & historical["qd_eligible"].fillna(False)
        if name == "v2.5"
        else pd.Series(True, index=historical.index)
    )
    assert current["QT_supported"].tolist() == expected_qt_support.tolist()
    expected_optima = pd.to_datetime(historical["candidate_time"], format="mixed").eq(
        pd.to_datetime(historical["t_star"], format="mixed")
    )
    assert current["is_optimum"].tolist() == expected_optima.tolist()
