from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

import fit_defrost_event_models
import select_defrost_time
from defrost_decision.candidate_quantities import DEFAULT_OUTCOME_MODEL
from defrost_decision.candidate_times import catalog_exclusion_reason
from defrost_event_models.ridge_models import OUTCOME_TARGETS


def test_public_defrost_commands_have_direct_help() -> None:
    root = Path(__file__).parents[2]
    for script in (
        "fit_defrost_event_models.py",
        "select_defrost_time.py",
        "calculate_v1_label_reference.py",
        "calculate_rb_reference.py",
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


def test_shared_batch_uses_each_cycle_prediction_mode(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def calculate(_loader, cycle: str, _models, **options):
        calls.append((cycle, options["prediction_mode"]))
        return pd.DataFrame(
            {"cycle_name": [cycle], "candidate_defrost_time": [pd.Timestamp("2026-01-01")]}
        )

    monkeypatch.setattr(select_defrost_time, "calculate_cycle", calculate)
    table = select_defrost_time.calculate_cycles(
        object(),
        ["cycle_a", "cycle_b"],
        {},
        candidate_step_seconds=10,
        allow_extrapolation=True,
        prediction_modes={"cycle_a": "cross-fitted", "cycle_b": "full-model"},
        workers=1,
    )

    assert calls == [("cycle_a", "cross-fitted"), ("cycle_b", "full-model")]
    assert table["cycle_name"].tolist() == ["cycle_a", "cycle_b"]


def test_metadata_gate_excludes_cycles_without_the_fixed_ten_minute_candidate_start() -> None:
    start = pd.Timestamp("2026-01-01")
    record = {
        "experiment_id": "exp",
        "boundaries": {
            "heating_start": start,
            "stable_heating_start": start,
            "defrost_preparation_start": start + pd.Timedelta(minutes=9, seconds=59),
            "defrost_start": start + pd.Timedelta(minutes=10),
            "defrost_end": start + pd.Timedelta(minutes=11),
        },
    }

    assert catalog_exclusion_reason(record, None) == "candidate interval is shorter than 10 minutes"


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


def test_full_model_dry_run_does_not_restrict_cycles_to_saved_folds(
    monkeypatch, tmp_path: Path
) -> None:
    parameters = {"support_threshold": 1.0}
    models = {
        "models": {
            "ridge_dynamic_state_8": {
                name: {"folds": {}, "full_data_model": parameters}
                for name in OUTCOME_TARGETS
            }
        }
    }
    received: list[set[str] | None] = []

    monkeypatch.setattr(select_defrost_time, "load_defrost_event_models", lambda _: models)
    monkeypatch.setattr(select_defrost_time, "DatasetLoader", lambda _: object())
    monkeypatch.setattr(
        select_defrost_time,
        "metadata_eligible_cycles",
        lambda _loader, _cycles, experiments: received.append(experiments) or ["cycle_new"],
    )

    assert select_defrost_time.main(
        [
            "--prediction-mode",
            "full-model",
            "--model-file",
            str(tmp_path / "models.json"),
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    ) == 0
    assert received == [None]
