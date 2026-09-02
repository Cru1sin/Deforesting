from __future__ import annotations

import math
import subprocess
import sys

import pandas as pd
import pytest

from model.evaluate import evaluate_run

SETTING = {
    "representation": "handcrafted",
    "head": "logistic",
    "camera": "front",
    "modality": "rgb",
}


def _metrics(*, held_out: str, status: str = "ok", test_images: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "held_out_experiment": held_out,
                **SETTING,
                "train_images": 4,
                "test_images": test_images,
                "status": status,
                "message": "bad fold" if status == "invalid" else "",
                "accuracy": 0.99,
                "balanced_accuracy": 0.99,
                "macro_f1": 0.99,
            }
        ]
    )


def _predictions(
    held_out: str,
    targets: list[int],
    predictions: list[int],
    *,
    setting: dict[str, str] | None = None,
) -> pd.DataFrame:
    values = setting or SETTING
    return pd.DataFrame(
        [
            {
                "experiment_id": held_out,
                "held_out_experiment": held_out,
                **values,
                "target": target,
                "prediction": prediction,
            }
            for target, prediction in zip(targets, predictions, strict=True)
        ]
    )


def test_binary_fold_metrics_use_full_expected_labels_when_heldout_lacks_class() -> None:
    experiment_metrics, _ = evaluate_run(
        _metrics(held_out="a", test_images=2),
        _predictions("a", [0, 0], [0, 0]),
        task="binary",
    )

    row = experiment_metrics.iloc[0]
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["balanced_accuracy"] == pytest.approx(0.5)
    assert row["macro_f1"] == pytest.approx(0.5)
    assert row["test_images"] == 2


def test_invalid_fold_is_retained_with_nan_metrics() -> None:
    experiment_metrics, _ = evaluate_run(
        _metrics(held_out="a", status="invalid"),
        pd.DataFrame(
            columns=[
                "experiment_id",
                "held_out_experiment",
                "representation",
                "head",
                "camera",
                "modality",
                "target",
                "prediction",
            ]
        ),
        task="binary",
    )

    row = experiment_metrics.iloc[0]
    assert row["status"] == "invalid"
    assert row["message"] == "bad fold"
    assert math.isnan(row["accuracy"])
    assert math.isnan(row["balanced_accuracy"])
    assert math.isnan(row["macro_f1"])


def test_summary_uses_equal_experiment_weight_not_frame_pooling() -> None:
    metrics = pd.concat(
        [_metrics(held_out="a", test_images=1), _metrics(held_out="b", test_images=3)],
        ignore_index=True,
    )
    predictions = pd.concat(
        [
            _predictions("a", [0], [0]),
            _predictions("b", [0, 0, 0], [0, 1, 1]),
        ],
        ignore_index=True,
    )

    experiment_metrics, summary = evaluate_run(metrics, predictions, task="binary")

    assert experiment_metrics["accuracy"].tolist() == [1.0, pytest.approx(1 / 3)]
    row = summary.iloc[0]
    assert row["total_folds"] == 2
    assert row["valid_folds"] == 2
    assert row["total_test_images"] == 4
    assert row["accuracy_mean"] == pytest.approx(2 / 3)
    assert row["accuracy_std"] == pytest.approx(math.sqrt(2) / 3)
    assert row["accuracy_mean"] != pytest.approx(0.5)


def test_prediction_leakage_is_rejected() -> None:
    predictions = _predictions("a", [0], [0])
    predictions.loc[0, "experiment_id"] = "other"

    with pytest.raises(ValueError, match="experiment_id.*held_out_experiment"):
        evaluate_run(_metrics(held_out="a", test_images=1), predictions, task="binary")


def test_prediction_group_must_map_to_one_metrics_row() -> None:
    predictions = _predictions("unknown", [0], [0])

    with pytest.raises(ValueError, match="does not map to exactly one metrics row"):
        evaluate_run(_metrics(held_out="a", test_images=1), predictions, task="binary")


def test_evaluation_module_imports_without_torch() -> None:
    code = """
import builtins
real_import = builtins.__import__
def import_without_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('evaluation imported torch')
    return real_import(name, *args, **kwargs)
builtins.__import__ = import_without_torch
import model.evaluate
"""
    subprocess.run([sys.executable, "-c", code], check=True)
