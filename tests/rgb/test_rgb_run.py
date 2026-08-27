from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from frost_analysis.rgb_run import RunStore, stable_task_id, validate_completed_run


def _task(status: str, task_id: str, heldout: str = "a") -> dict[str, object]:
    return {
        "task_id": task_id,
        "status": status,
        "elapsed": 0.1,
        "heldout": heldout,
        "stage": "MATRIX",
        "camera_group": "all",
        "regret_threshold": 0.01,
        "representation": "dinov3",
        "model": "logistic",
        "modality": "rgb",
        "warning_count": 0,
        "error_type": "",
        "message": "",
    }


def test_resume_skips_only_ok_or_invalid_with_prediction(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "same-run", {"task": "three"}, {"jobs": 3})
    ok = _task("ok", "ok")
    invalid = _task("invalid", "invalid")
    failed = _task("failed", "failed")
    store.record(ok, pd.DataFrame({"sample": [1]}))
    store.record(invalid, pd.DataFrame({"sample": [2]}))
    store.record(failed, pd.DataFrame({"sample": [3]}))
    (store.predictions_dir / "invalid.parquet").unlink()

    resumed = RunStore(tmp_path, "same-run", {"task": "three"}, {"jobs": 1})

    assert resumed.completed_task_ids() == {"ok"}
    assert "failed" not in resumed.completed_task_ids()


def test_resume_rejects_changed_config(tmp_path: Path) -> None:
    RunStore(tmp_path, "same-run", {"task": "three"})

    with pytest.raises(ValueError, match="configuration"):
        RunStore(tmp_path, "same-run", {"task": "binary"})


def test_failed_or_incomplete_run_does_not_publish_latest(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "failed-run", {"jobs": 3})
    store.record(_task("failed", "failed"), pd.DataFrame({"sample": [1]}))
    store.mark_failed("one failed task")

    assert not (tmp_path / "latest.json").exists()
    assert json.loads(store.manifest_path.read_text())["status"] == "failed"


def test_complete_run_atomically_publishes_latest_without_touching_legacy_files(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "predictions.parquet"
    legacy.write_bytes(b"legacy")
    store = RunStore(tmp_path, "complete-run", {"jobs": 3})
    store.mark_complete()

    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["run_id"] == "complete-run"
    assert Path(latest["run_dir"]) == store.run_dir
    assert legacy.read_bytes() == b"legacy"
    assert not list(tmp_path.rglob("*.tmp"))


def test_resuming_complete_run_withdraws_its_latest_pointer(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "resumed-run", {"task": "three"})
    store.mark_complete()

    RunStore(tmp_path, "resumed-run", {"task": "three"}, {"jobs": 1})

    assert not (tmp_path / "latest.json").exists()
    assert json.loads(store.manifest_path.read_text())["status"] == "running"


def test_stable_task_id_depends_on_combination_and_holdout() -> None:
    assert stable_task_id(1, "experiment/a") == "combo_001__heldout_experiment_a"
    assert stable_task_id(1, "a") != stable_task_id(2, "a")


def test_completed_run_validation_rejects_duplicate_prediction_identity() -> None:
    ledger = pd.DataFrame([_task("ok", "a"), _task("ok", "b", heldout="b")])
    predictions = pd.DataFrame(
        {
            "task_id": ["a", "b"],
            "camera_group": ["all", "all"],
            "regret_threshold": [0.01, 0.01],
            "representation": ["dinov3", "dinov3"],
            "model": ["logistic", "logistic"],
            "modality": ["rgb", "rgb"],
            "held_out_experiment": ["a", "a"],
            "cycle_name": ["cycle", "cycle"],
            "camera_role": ["front", "front"],
            "image_time": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        }
    )

    with pytest.raises(ValueError, match="prediction keys"):
        validate_completed_run(ledger, predictions, expected_tasks=2, folds_per_combination=2)


def test_completed_run_validation_rejects_any_task_warning() -> None:
    ledger = pd.DataFrame([_task("ok", "a")])
    ledger.loc[0, "warning_count"] = 1

    with pytest.raises(ValueError, match="task warnings"):
        validate_completed_run(ledger, pd.DataFrame(), expected_tasks=1, folds_per_combination=1)
