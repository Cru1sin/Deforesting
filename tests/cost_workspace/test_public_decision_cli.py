from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import calculate_rb_reference
import defrosting_electricity_model
import optimal_defrost_time
from plots import dataset as dataset_figures


@pytest.mark.parametrize("publish", [False, True])
def test_rb_refresh_publishes_prior_only_when_requested(tmp_path, monkeypatch, publish):
    class Loader:
        def list_cycles(self):
            return pd.DataFrame({"cycle_name": ["a", "b"]})

    monkeypatch.setattr(calculate_rb_reference, "DatasetLoader", lambda _: Loader())
    monkeypatch.setattr(
        calculate_rb_reference.rule_based,
        "calculate_cycle",
        lambda _, name: {
            "cycle_name": name,
            "experiment_id": "exp",
            "t_RB": "2026-01-01 01:00:00" if name == "a" else None,
            "rb_status": "triggered" if name == "a" else "right_censored",
        },
    )
    args = ["--dataset", str(tmp_path), "--output", str(tmp_path / "runs"), "--n-jobs", "1"]
    if publish:
        args.append("--publish-prior")
        with pytest.raises(ValueError, match="requires all cycles"):
            calculate_rb_reference.main([*args, "--cycles", "a"])
    assert calculate_rb_reference.main(args) == 0
    prior = tmp_path / "priors/rb_reference.csv"
    assert prior.exists() == publish
    if publish:
        saved = pd.read_csv(prior)
        assert saved.cycle_name.tolist() == ["a", "b"]
        assert saved.rule.eq("RB-20260923").all()
        assert pd.isna(saved.loc[1, "t_RB"])


@pytest.mark.parametrize("publish", [False, True])
def test_cop_updates_dataset_only_when_explicitly_published(tmp_path, monkeypatch, publish):
    from dataset_tools import cycle_metadata
    from plots import defrost_decision

    models = dict(
        phase_models={"folds": {"exp": {}}},
        cop_definition="refrigerant_effective_heat",
        recovery_settings=optimal_defrost_time.recovery_settings(
            optimal_defrost_time.build_parser().parse_args([])
        ),
    )

    class Loader:
        registry = {"measurement_processing": "test"}

        def load_rb_reference(self, cycles, *, path=None):
            return pd.DataFrame(
                {"cycle_name": cycles, "t_RB": pd.NaT, "rb_status": "right_censored"}
            ).set_index("cycle_name")

        def list_cycles(self, **kwargs):
            return pd.DataFrame(dict(cycle_name=["cycle"], experiment_id=["exp"]))

        def configure_recovery(self, settings):
            return pd.DataFrame(dict(cycle_name=["cycle"], experiment_id=["exp"]))

    decisions = pd.DataFrame(
        dict(cycle_name=["cycle"], candidate_defrost_time=[pd.Timestamp("2026-01-01")])
    )
    monkeypatch.setattr(optimal_defrost_time, "DatasetLoader", lambda path: Loader())
    monkeypatch.setattr(optimal_defrost_time, "load_defrost_event_models", lambda path: models)
    monkeypatch.setattr(optimal_defrost_time, "calculate_cycle", lambda *a, **kw: decisions)
    monkeypatch.setattr(
        optimal_defrost_time,
        "summarize_decisions",
        lambda rows: pd.DataFrame(dict(cycle_status=["identified_curve"])),
    )
    calls = []
    monkeypatch.setattr(
        cycle_metadata, "update_effective_cop_quality", lambda *a, **kw: calls.append("catalog")
    )
    monkeypatch.setattr(
        dataset_figures,
        "publish_effective_decision_assets",
        lambda *a, **kw: calls.append("assets"),
    )
    monkeypatch.setattr(defrost_decision, "render_current_decision_figures", lambda *a, **kw: None)
    args = [
        "--dataset",
        str(tmp_path / "dataset"),
        "--output",
        str(tmp_path / "result"),
        "--n-jobs",
        "1",
    ]
    if publish:
        args += ["--publish-dataset", "--figures"]
    assert optimal_defrost_time.main(args) == 0
    assert calls == (["catalog", "assets"] if publish else [])
    assert (tmp_path / "result/candidate_decisions.csv").exists()


def test_public_defrost_commands_have_direct_help() -> None:
    root = Path(__file__).parents[2]
    for script in (
        "defrosting_electricity_model.py",
        "optimal_defrost_time.py",
        "calculate_rb_reference.py",
    ):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert not re.search(r"--action(?![\w-])", result.stdout)
        assert "--variant" not in result.stdout


def test_effective_cop_rejects_retired_heat_options() -> None:
    import pytest

    for rule in ("without_trunc", "outlet_at_least_recovery_temperature"):
        with pytest.raises(SystemExit):
            optimal_defrost_time.build_parser().parse_args(["--effective-heat-rule", rule])


def test_ridge_uses_admitted_count_and_updates_same_output(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from defrost_event_models.event_electricity_ridge import THREE_INPUTS

    events = pd.DataFrame(
        {"E_prep_valid": [True, False, True], "E_D_valid": True, "E_prep_kwh": 0.1, "E_D_kwh": 0.2}
    )
    for feature in THREE_INPUTS:
        events[feature] = 1.0
        events[feature + "_valid_second_count"] = 60
    loader = SimpleNamespace(
        registry={"resample_interval_seconds": 1}, configure_recovery=lambda _: pd.DataFrame()
    )
    monkeypatch.setattr(defrosting_electricity_model, "DatasetLoader", lambda _: loader)
    monkeypatch.setattr(
        defrosting_electricity_model, "build_defrost_event_training_table", lambda _: events
    )
    calls = []

    def fit(loader, admitted, run, workers, figures):
        calls.append(run)
        assert admitted.selected_for_training.tolist() == [True, False, True]
        run.mkdir(parents=True, exist_ok=True)
        (run / "model.txt").write_text(str(len(calls)))

    monkeypatch.setattr(defrosting_electricity_model, "fit_preparation_defrost_model", fit)
    for _ in range(2):
        assert defrosting_electricity_model.main(["--dataset", str(tmp_path)]) == 0
    expected = tmp_path / "除霜预测模型/三输入Ridge-两阶段-2-1"
    assert calls == [expected, expected]
    assert (expected / "model.txt").read_text() == "2"
    explicit = tmp_path / "chosen_model"
    defrosting_electricity_model.main(["--dataset", str(tmp_path), "--output", str(explicit)])
    assert calls[-1] == explicit
    assert (explicit / "model.txt").exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_reselection_respects_preview_and_does_not_render_by_default(
    tmp_path, monkeypatch, dry_run
):
    from defrost_decision import performance_objectives, selection_results
    from plots import defrost_decision

    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame(
        {"cycle_name": ["a"], "candidate_defrost_time": ["2026-01-01"], "is_selected": [True]}
    ).to_csv(source / "candidate_decisions.csv", index=False)
    (source / "run_settings.json").write_text("{}")

    def calculate(rows, **kwargs):
        assert not dry_run, "Preview must not recalculate"
        return rows

    monkeypatch.setattr(performance_objectives, "calculate_cycle_cop", calculate)
    monkeypatch.setattr(performance_objectives, "select_cycle_optimum", lambda rows: rows)
    monkeypatch.setattr(selection_results, "summarize_decisions", lambda rows: rows)
    monkeypatch.setattr(
        defrost_decision,
        "render_current_decision_figures",
        lambda *a, **k: pytest.fail("Rendering was not requested"),
    )
    args = ["--reselect-run", str(source), "--output", str(tmp_path / "results")]
    if dry_run:
        args.append("--dry-run")
    assert optimal_defrost_time.main(args) == 0
    result = tmp_path / "results"
    assert result.exists() == (not dry_run)


def test_selection_dry_run_checks_model_cohort_without_raw_cycles(
    monkeypatch, tmp_path: Path
) -> None:
    folds = {"exp": {"support_threshold": 1.0}}
    models = {"phase_models": {"folds": folds}}

    class Loader:
        registry = {"measurement_processing": "test"}

        def load_rb_reference(self, cycles, *, path=None):
            return pd.DataFrame(
                {"cycle_name": cycles, "t_RB": pd.NaT, "rb_status": "right_censored"}
            ).set_index("cycle_name")

        def list_cycles(self, **kwargs):
            return pd.DataFrame({"cycle_name": ["cycle_001"], "experiment_id": ["exp"]})

    monkeypatch.setattr(optimal_defrost_time, "load_defrost_event_models", lambda _: models)
    monkeypatch.setattr(optimal_defrost_time, "DatasetLoader", lambda _: Loader())
    models.update(
        cop_definition="refrigerant_effective_heat", recovery_settings={"recovery_rule": "dataset"}
    )
    assert (
        optimal_defrost_time.main(
            [
                "--model-file",
                str(tmp_path / "models.json"),
                "--output",
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
    models = {"phase_models": {"folds": {}, "full_data_model": {"E_prep_D_kwh": parameters}}}
    models.update(
        cop_definition="refrigerant_effective_heat", recovery_settings={"recovery_rule": "dataset"}
    )

    class Loader:
        registry = {"measurement_processing": "test"}

        def load_rb_reference(self, cycles, *, path=None):
            return pd.DataFrame(
                {"cycle_name": cycles, "t_RB": pd.NaT, "rb_status": "right_censored"}
            ).set_index("cycle_name")

        def list_cycles(self, **kwargs):
            return pd.DataFrame({"cycle_name": ["cycle_new"], "experiment_id": ["exp"]})

    monkeypatch.setattr(optimal_defrost_time, "load_defrost_event_models", lambda _: models)
    monkeypatch.setattr(optimal_defrost_time, "DatasetLoader", lambda _: Loader())

    assert (
        optimal_defrost_time.main(
            [
                "--prediction-mode",
                "full-model",
                "--model-file",
                str(tmp_path / "models.json"),
                "--output",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )


def test_current_csv_requires_processing_registry_and_reads_published_values(tmp_path, monkeypatch):
    import json

    import pytest

    csv = tmp_path / "cycle.csv"
    pd.DataFrame(
        {"timestamp": ["2026-01-01"], "power_total": [None], "power_total_interpolated": [2.0]}
    ).to_csv(csv, index=False)
    model = {
        "cop_definition": "refrigerant_effective_heat",
        "measurement_processing": "dataset_offline_linear_60s",
    }
    monkeypatch.setattr(optimal_defrost_time, "load_defrost_event_models", lambda _: model)
    with pytest.raises(ValueError, match="requires its --current-registry"):
        optimal_defrost_time.main(["--current-csv", str(csv)])
    registry = tmp_path / "channel_registry.json"
    registry.write_text(json.dumps({"measurement_processing": "dataset_offline_linear_60s"}))

    def evaluate(frame, models, experiment):
        assert frame.attrs["measurement_processing"] == model["measurement_processing"]
        assert frame.power_total.iloc[0] == 2.0
        return {"status": "supported"}

    monkeypatch.setattr(optimal_defrost_time, "current_effective_cop", evaluate)
    assert (
        optimal_defrost_time.main(["--current-csv", str(csv), "--current-registry", str(registry)])
        == 0
    )
