from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cost.cost_function_v1 import calculate_cycle as calculate_v1
from cost.cost_function_v2_5 import calculate_cycle as calculate_v25
from dataloader import DatasetLoader

ORIGINAL_ROOT = Path("/Users/cruisin/Documents/DeforestingSensor")
CYCLE = "frost_cycle_000070"


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
    assert result.loc[result["is_optimum"], "candidate_time"].tolist() == [optimum]
    assert pd.Timestamp(historical["t_star"].iloc[0]) == optimum
