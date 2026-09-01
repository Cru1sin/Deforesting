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
        "defrost_energy_kwh": "defrost_electricity_kwh",
        "recovery_energy_kwh": "recovery_electricity_kwh",
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
    assert np.allclose(
        result["heating_energy_legacy_bridged_kwh"],
        historical["heating_electricity_kwh"],
        rtol=0,
        atol=5e-13,
    )
    assert np.allclose(
        result["heating_heat_legacy_bridged_kwh"],
        historical[heat_column],
        rtol=0,
        atol=5e-13,
    )
    assert not np.allclose(result["heating_energy_kwh"], historical["heating_electricity_kwh"])
    if name == "v1":
        assert np.allclose(result["heating_heat_kwh"], historical[heat_column], rtol=0, atol=5e-13)
    else:
        assert not np.allclose(result["heating_heat_kwh"], historical[heat_column])
    assert result.loc[result["is_optimum"], "candidate_time"].tolist() == [optimum]
    assert pd.Timestamp(historical["t_star"].iloc[0]) == optimum
