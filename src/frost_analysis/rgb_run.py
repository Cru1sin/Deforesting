"""Small, crash-safe state store for RGB matrix runs."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

COMBINATION_COLUMNS = (
    "camera_group",
    "regret_threshold",
    "representation",
    "model",
    "modality",
)


def stable_task_id(combination_index: int, heldout: object) -> str:
    """Return a stable, filesystem-safe identifier for one held-out task."""
    safe_heldout = re.sub(r"[^\w.-]+", "_", str(heldout)).strip("_")
    return f"combo_{combination_index:03d}__heldout_{safe_heldout}"


def _atomic(path: Path, write: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        write(temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_json(path: Path, value: object) -> None:
    _atomic(
        path,
        lambda temporary: temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        ),
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic(path, lambda temporary: frame.to_csv(temporary, index=False))


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    _atomic(path, lambda temporary: frame.to_parquet(temporary, index=False))


class RunStore:
    """Own the isolated files and resume state for one local run."""

    def __init__(
        self,
        output: Path,
        run_id: str,
        config: dict[str, Any],
        execution: dict[str, Any] | None = None,
    ) -> None:
        self.output = Path(output)
        self.run_id = run_id
        self.run_dir = self.output / "runs" / run_id
        self.predictions_dir = self.run_dir / "predictions"
        self.manifest_path = self.run_dir / "manifest.json"
        self.ledger_path = self.run_dir / "task_ledger.csv"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("config") != config:
                raise ValueError("run configuration does not match existing manifest")
            latest_path = self.output / "latest.json"
            if manifest.get("status") == "complete" and latest_path.exists():
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
                if latest.get("run_id") == run_id:
                    latest_path.unlink()
            manifest["status"] = "running"
            manifest["execution"] = execution or {}
        else:
            manifest = {
                "run_id": run_id,
                "status": "running",
                "config": config,
                "execution": execution or {},
            }
        atomic_json(self.manifest_path, manifest)

    def ledger(self) -> pd.DataFrame:
        return (
            pd.read_csv(
                self.ledger_path,
                keep_default_na=False,
                dtype={"task_id": "string"},
            )
            if self.ledger_path.exists()
            else pd.DataFrame()
        )

    def completed_task_ids(self) -> set[str]:
        ledger = self.ledger()
        if ledger.empty:
            return set()
        return {
            str(row.task_id)
            for row in ledger.itertuples()
            if row.status in {"ok", "invalid"}
            and (self.predictions_dir / f"{row.task_id}.parquet").is_file()
        }

    def record(self, result: dict[str, Any], predictions: pd.DataFrame) -> None:
        task_id = str(result["task_id"])
        atomic_parquet(self.predictions_dir / f"{task_id}.parquet", predictions)
        ledger = self.ledger()
        row = pd.DataFrame([result])
        if not ledger.empty:
            ledger = ledger.loc[ledger["task_id"].astype(str).ne(task_id)]
        atomic_csv(self.ledger_path, pd.concat([ledger, row], ignore_index=True))

    def _set_status(self, status: str, message: str = "") -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest.update(status=status, message=message)
        atomic_json(self.manifest_path, manifest)

    def mark_failed(self, message: str) -> None:
        self._set_status("failed", message)

    def mark_complete(self) -> None:
        self._set_status("complete")
        atomic_json(
            self.output / "latest.json",
            {"run_id": self.run_id, "run_dir": str(self.run_dir.resolve())},
        )


def validate_completed_run(
    ledger: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    expected_tasks: int,
    folds_per_combination: int,
) -> None:
    """Enforce the task and prediction invariants before aggregation/publication."""
    if len(ledger) != expected_tasks or ledger["task_id"].nunique() != expected_tasks:
        raise ValueError(f"expected {expected_tasks} unique tasks")
    counts = ledger["status"].value_counts()
    if set(counts.index) - {"ok", "invalid", "failed"}:
        raise ValueError("unknown task status")
    if int(counts.sum()) != expected_tasks:
        raise ValueError("task status totals do not match expected tasks")
    if int(counts.get("failed", 0)):
        raise ValueError("run contains failed tasks")
    if pd.to_numeric(ledger["warning_count"], errors="coerce").fillna(0).gt(0).any():
        raise ValueError("run contains task warnings")
    if ledger.duplicated([*COMBINATION_COLUMNS, "heldout"]).any():
        raise ValueError("combination and held-out task keys must be unique")
    combination_counts = ledger.groupby(list(COMBINATION_COLUMNS), dropna=False).size()
    if not combination_counts.eq(folds_per_combination).all():
        raise ValueError("each combination must contain every held-out task")
    identity = [
        column
        for column in (
            "cycle_uid",
            "cycle_name",
            "camera_role",
            "file_name",
            "frame_index",
            "image_time",
        )
        if column in predictions
    ]
    keys = [*COMBINATION_COLUMNS, "held_out_experiment", *identity]
    if not predictions.empty and predictions.duplicated(keys).any():
        raise ValueError("prediction keys must be unique")
