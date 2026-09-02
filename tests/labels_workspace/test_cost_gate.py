from __future__ import annotations

import pandas as pd
import pytest

from labels.build import validate_cost


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
    validate_cost(_canonical_cost())


def test_gate_rejects_missing_columns_clearly() -> None:
    cost = _canonical_cost().drop(columns="candidate_time")

    with pytest.raises(ValueError, match="missing required columns: candidate_time"):
        validate_cost(cost)


def test_gate_rejects_any_label_ineligible_row() -> None:
    cost = _canonical_cost()
    cost.loc[1, "label_eligible"] = False

    with pytest.raises(ValueError, match="label_eligible must be True for every row"):
        validate_cost(cost)


def test_gate_rejects_any_named_variant() -> None:
    cost = _canonical_cost()
    cost.loc[1, "variant"] = "strict_state"

    with pytest.raises(ValueError, match="named cost variant cannot produce hard labels"):
        validate_cost(cost)
