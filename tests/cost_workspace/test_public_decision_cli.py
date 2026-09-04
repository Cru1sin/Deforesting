from __future__ import annotations

import subprocess
from pathlib import Path

import fit_defrost_event_models
import select_defrost_time
from defrost_decision.candidate_quantities import DEFAULT_OUTCOME_MODEL
from defrost_event_models.ridge_models import OUTCOME_TARGETS


def test_public_defrost_commands_have_direct_help() -> None:
    root = Path(__file__).parents[2]
    for script in (
        "fit_defrost_event_models.py",
        "select_defrost_time.py",
        "calculate_v1_label_reference.py",
    ):
        result = subprocess.run(
            [str(root / ".venv/bin/python"), script, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--action" not in result.stdout
        assert "--variant" not in result.stdout


def test_candidate_quantities_use_released_dynamic_ridge_models() -> None:
    assert DEFAULT_OUTCOME_MODEL == "ridge_dynamic_state_8"


def test_fit_dry_run_does_not_read_dataset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        fit_defrost_event_models,
        "DatasetLoader",
        lambda *_: (_ for _ in ()).throw(AssertionError("dataset must not be read")),
    )
    assert (
        fit_defrost_event_models.main(
            ["--run-name", "check", "--output-root", str(tmp_path), "--dry-run"]
        )
        == 0
    )


def test_selection_dry_run_checks_model_cohort_without_raw_cycles(
    monkeypatch, tmp_path: Path
) -> None:
    folds = {"exp": {"support_threshold": 1.0}}
    models = {
        "models": {"ridge_dynamic_state_8": {name: {"folds": folds} for name in OUTCOME_TARGETS}}
    }

    class Loader:
        pass

    monkeypatch.setattr(select_defrost_time, "load_defrost_event_models", lambda _: models)
    monkeypatch.setattr(select_defrost_time, "DatasetLoader", lambda _: Loader())
    monkeypatch.setattr(
        select_defrost_time,
        "metadata_eligible_cycles",
        lambda *_: ["cycle_001"],
    )
    assert (
        select_defrost_time.main(
            [
                "--model-file",
                str(tmp_path / "models.json"),
                "--output-root",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )
