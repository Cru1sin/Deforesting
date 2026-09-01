from __future__ import annotations

import importlib

import pandas as pd
import pytest


def _gate(cost: pd.DataFrame) -> None:
    try:
        build = importlib.import_module("labels.build")
    except ModuleNotFoundError:
        pytest.fail("labels.build cost gate is missing", pytrace=False)
    build.validate_cost(cost)


def _canonical_cost() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "cycle_a"],
            "candidate_time": pd.date_range("2026-01-01", periods=2, freq="min"),
            "relative_regret": [0.0, 0.1],
            "optimization_eligible": [True, True],
            "is_censored": [False, False],
            "label_eligible": [True, True],
            "variant": [None, None],
        }
    )


def test_gate_accepts_only_canonical_label_eligible_cost() -> None:
    _gate(_canonical_cost())


def test_gate_rejects_missing_columns_clearly() -> None:
    cost = _canonical_cost().drop(columns="candidate_time")

    with pytest.raises(ValueError, match="missing required columns: candidate_time"):
        _gate(cost)


def test_gate_rejects_any_label_ineligible_row() -> None:
    cost = _canonical_cost()
    cost.loc[1, "label_eligible"] = False

    with pytest.raises(ValueError, match="label_eligible must be True for every row"):
        _gate(cost)


def test_gate_rejects_any_named_variant() -> None:
    cost = _canonical_cost()
    cost.loc[1, "variant"] = "strict_state"

    with pytest.raises(ValueError, match="named cost variant cannot produce hard labels"):
        _gate(cost)
