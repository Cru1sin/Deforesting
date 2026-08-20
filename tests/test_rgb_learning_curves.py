from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _module():  # type: ignore[no-untyped-def]
    path = Path("scripts/evaluate_rgb_learning_curves.py")
    spec = importlib.util.spec_from_file_location("rgb_learning_curves", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_experiment_subsets_are_nested_and_exclude_test() -> None:
    module = _module()
    subsets = module.nested_training_sets(["a", "b", "c", "d", "e"], "c", [2, 4], 0)

    assert "c" not in subsets[4]
    assert set(subsets[2]).issubset(subsets[4])
    assert len(subsets[2]) == 2
    assert len(subsets[4]) == 4


def test_summary_averages_repeats_before_experiment_bootstrap() -> None:
    module = _module()
    rows = []
    for held_out in ("a", "b"):
        for repeat in (0, 1):
            rows.append(
                {
                    "camera_group": "front",
                    "model": "logistic",
                    "training_experiment_count": 2,
                    "held_out_experiment": held_out,
                    "balanced_accuracy": 0.8 + 0.1 * repeat,
                    "macro_f1": 0.8,
                    "auroc": 0.9,
                    "balanced_misclassification_regret": 0.01,
                    "fit_predict_seconds": 1.0,
                    "training_cycle_count": 4,
                    "training_image_count": 100,
                }
            )
    summary = module.summarize(pd.DataFrame(rows))
    balanced = summary.loc[summary["metric"].eq("balanced_accuracy")].iloc[0]

    assert balanced["estimate"] == pytest.approx(0.85)
    assert balanced["held_out_experiment_count"] == 2
