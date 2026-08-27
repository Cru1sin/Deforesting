from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _module():  # type: ignore[no-untyped-def]
    path = Path("scripts/rgb/evaluate_rgb_illumination_robustness.py")
    spec = importlib.util.spec_from_file_location("rgb_illumination_robustness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_low_light_augmentation_excludes_held_out_experiment() -> None:
    module = _module()
    rows = []
    for experiment in ("a", "b", "c"):
        for target, value in ((0, -1.0), (1, 1.0)):
            for _ in range(2):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "camera_role": "front",
                        "target": target,
                        "relative_regret": 0.1,
                        "dinov2_000": value,
                        "efficientnet_000": value,
                    }
                )
    cohort = pd.DataFrame(rows)
    conditions = {name: cohort.copy() for name in module.CONDITIONS}

    results = module.evaluate_conditions(cohort, conditions, ["front"])

    native_count = results.loc[
        results["training_strategy"].eq("native_only"), "train_image_count"
    ].unique()
    augmented_count = results.loc[
        results["training_strategy"].eq("low_light_augmented"), "train_image_count"
    ].unique()
    assert native_count.tolist() == [8]
    assert augmented_count.tolist() == [32]


def test_ood_evaluation_can_run_one_selected_backbone() -> None:
    module = _module()
    rows = []
    for experiment in ("a", "b", "c"):
        for target, value in ((0, -1.0), (1, 1.0)):
            rows.append(
                {
                    "experiment_id": experiment,
                    "camera_role": "front",
                    "target": target,
                    "relative_regret": 0.1,
                    "dinov2_000": value,
                    "efficientnet_000": value,
                }
            )
    cohort = pd.DataFrame(rows)
    conditions = {name: cohort.copy() for name in module.CONDITIONS}

    results = module.evaluate_conditions(
        cohort, conditions, ["front"], representations=["dinov2"]
    )

    assert results["representation"].unique().tolist() == ["dinov2"]
