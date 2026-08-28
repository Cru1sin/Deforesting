from __future__ import annotations

import numpy as np
import pandas as pd

from frost_analysis.exploration.sensor_model import (
    add_cycle_future,
    apply_reference_model,
    fit_reference_model,
    fit_weighted_ridge,
    shared_complete_cases,
    split_replication_cohort,
)


def test_compared_models_use_one_shared_complete_case() -> None:
    frame = pd.DataFrame(
        {
            "future": [1.0, 2.0, 3.0],
            "z": [0.0, 1.0, 2.0],
            "rate": [np.nan, 0.1, 0.2],
            "context": [5.0, 6.0, np.nan],
        }
    )

    result = shared_complete_cases(
        frame,
        target="future",
        models={"state": ["z"], "dynamic": ["z", "rate"], "context": ["z", "context"]},
    )

    assert result.index.tolist() == [1]


def test_replication_scope_freezes_first_49_and_tests_next_10() -> None:
    cycles = pd.DataFrame(
        {
            "cycle_name": [f"frost_cycle_{index:06d}" for index in range(1, 62)],
            "status": ["invalid" if index == 11 else "valid" for index in range(1, 62)],
        }
    )

    old, new, stress = split_replication_cohort(cycles)

    assert old["cycle_name"].iloc[-1] == "frost_cycle_000049"
    assert new["cycle_name"].tolist() == [f"frost_cycle_{index:06d}" for index in range(50, 60)]
    assert stress["cycle_name"].tolist() == ["frost_cycle_000011"]
    assert "frost_cycle_000060" not in set(pd.concat([old, new])["cycle_name"])


def test_reference_model_uses_training_context_and_only_cycle_offset_at_test() -> None:
    train = pd.DataFrame(
        {
            "cycle": ["a"] * 3 + ["b"] * 3,
            "early": [True, True, False] * 2,
            "context": [0.0, 1.0, 2.0] * 2,
            "target": [1.0, 3.0, 5.0, 11.0, 13.0, 15.0],
        }
    )
    test = pd.DataFrame(
        {
            "cycle": ["held"] * 4,
            "early": [True, True, False, False],
            "context": [0.0, 1.0, 2.0, 3.0],
            "target": [21.0, 23.0, 25.0, 27.0],
        }
    )

    model = fit_reference_model(train, target="target", features=["context"], early="early")
    reference = apply_reference_model(
        model,
        test,
        observed="target",
        cycle="cycle",
        early="early",
    )

    assert np.allclose(reference, test["target"], atol=1e-5)


def test_reference_without_valid_early_calibration_is_not_silently_used() -> None:
    train = pd.DataFrame(
        {"early": [True, True, True], "context": [0.0, 1.0, 2.0], "target": [1.0, 3.0, 5.0]}
    )
    test = pd.DataFrame(
        {
            "cycle": ["held", "held"],
            "early": [True, False],
            "context": [0.0, 1.0],
            "target": [np.nan, 23.0],
        }
    )

    model = fit_reference_model(train, target="target", features=["context"], early="early")
    reference = apply_reference_model(
        model,
        test,
        observed="target",
        cycle="cycle",
        early="early",
    )

    assert reference.isna().all()


def test_future_target_never_crosses_cycle_boundary() -> None:
    frame = pd.DataFrame(
        {
            "cycle": ["a", "a", "a", "b", "b", "b"],
            "value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )

    result = add_cycle_future(frame, column="value", horizon_steps=2, cycle="cycle")

    assert result.iloc[0] == 3.0
    assert result.iloc[3] == 30.0
    assert result.iloc[[1, 2, 4, 5]].isna().all()


def test_future_target_rejects_irregular_time_gap() -> None:
    frame = pd.DataFrame(
        {
            "cycle": ["a"] * 4,
            "minute": [0.0, 1.0, 4.0, 5.0],
            "value": [1.0, 2.0, 4.0, 5.0],
        }
    )

    result = add_cycle_future(
        frame,
        column="value",
        horizon_steps=2,
        cycle="cycle",
        time="minute",
        step_minutes=1.0,
    )

    assert result.isna().all()


def test_weighted_ridge_gives_cycles_equal_total_weight() -> None:
    frame = pd.DataFrame(
        {
            "cycle": ["long"] * 100 + ["short"] * 2,
            "x": [0.0] * 102,
            "y": [0.0] * 100 + [10.0, 10.0],
        }
    )

    model = fit_weighted_ridge(
        frame,
        target="y",
        features=["x"],
        cycle="cycle",
    )

    assert abs(model.intercept - 5.0) < 1e-6


def test_weighted_ridge_normalization_is_also_cycle_balanced() -> None:
    frame = pd.DataFrame(
        {
            "cycle": ["long"] * 100 + ["short"] * 2,
            "x": [0.0] * 100 + [10.0, 10.0],
            "y": [0.0] * 100 + [10.0, 10.0],
        }
    )

    model = fit_weighted_ridge(
        frame,
        target="y",
        features=["x"],
        cycle="cycle",
    )

    assert abs(model.center[0] - 5.0) < 1e-6
