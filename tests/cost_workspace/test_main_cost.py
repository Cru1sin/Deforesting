from __future__ import annotations

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest

import main_cost


def _policy_artifact(path: Path) -> Path:
    targets = ("energy", "heat", "compressor_energy", "duration")
    path.write_text(
        json.dumps(
            {
                "artifact_version": "v2.6.8",
                "models": {
                    "ticket_ridge_dynamic8": {
                        name: {"folds": {"exp_20260714": {}}} for name in targets
                    }
                },
            }
        )
    )
    return path


class MetadataOnlyDataset:
    def list_cycles(self, **_: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cycle_name": ["cycle_a", "cycle_b"],
                "experiment_id": "exp_20260714",
                "heating_start": "2026-01-01 00:00:00",
                "stable_heating_start": "2026-01-01 00:02:00",
                "defrost_preparation_start": "2026-01-01 00:18:30",
                "defrost_start": "2026-01-01 00:19:00",
                "defrost_end": "2026-01-01 00:24:00",
            }
        )

    def load_cycle_original(self, *_: object, **__: object) -> pd.DataFrame:
        raise AssertionError("dry-run must not read cycle time series")

    def get_cycle_record(self, cycle_name: str) -> dict[str, object]:
        assert cycle_name in {"cycle_a", "cycle_b"}
        return {
            "cycle_name": cycle_name,
            "experiment_id": "exp_20260714",
            "boundaries": {
                "heating_start": "2026-01-01 00:00:00",
                "stable_heating_start": "2026-01-01 00:02:00",
                "defrost_preparation_start": "2026-01-01 00:18:30",
                "defrost_start": "2026-01-01 00:19:00",
                "defrost_end": "2026-01-01 00:24:00",
            },
        }


def test_cost_cli_help_imports_the_moved_plot_consumer() -> None:
    result = subprocess.run(
        [sys.executable, "main_cost.py", "--help"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert "--action" in result.stdout


class AnchorDataset(MetadataOnlyDataset):
    def load_cycle_original(
        self, cycle_name: str, *, columns: list[str] | None = None
    ) -> pd.DataFrame:
        periods = 60 if cycle_name == "cycle_a" else 47
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01 00:02:00", periods=periods, freq="s"),
                "water_flow": 1.0,
                "water_in_temperature": 40.0,
                "water_out_temperature": 45.0,
                "power_total": 6.0,
            }
        )
        return frame if columns is None else frame[columns]


def test_dry_run_checks_recipe_output_and_cycle_count_without_time_series(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: MetadataOnlyDataset())

    status = main_cost.main(
        [
            "--action",
            "calculate",
            "--cost",
            "v1",
            "--dataset",
            str(tmp_path / "dataset"),
            "--cycles",
            "cycle_a",
            "--output-root",
            str(tmp_path / "output"),
            "--dry-run",
        ]
    )

    assert status == 0
    output = capsys.readouterr().out
    assert "Dry-run OK" in output
    assert "1 metadata-eligible cycle" in output
    assert "raw clean-anchor gate deferred" in output
    assert not (tmp_path / "output").exists()


def test_implicit_calculation_applies_raw_clean_anchor_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Module:
        DEFAULT_RECIPE = main_cost.cost_function_v1.DEFAULT_RECIPE
        validate_recipe = staticmethod(main_cost.cost_function_v1.validate_recipe)

        @staticmethod
        def calculate(_: object, cycles: list[str], __: object) -> pd.DataFrame:
            return pd.DataFrame({"cycle_name": cycles, "inverse_cop": 0.5})

    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: AnchorDataset())
    monkeypatch.setitem(main_cost.COST_MODULES, "v1", Module)
    output = tmp_path / "output"

    assert (
        main_cost.main(["--action", "calculate", "--cost", "v1", "--output-root", str(output)]) == 0
    )

    result = pd.read_csv(output / "cost/v1/cost.csv")
    assert result["cycle_name"].tolist() == ["cycle_a"]
    message = capsys.readouterr().out
    assert "Selected 1 cycle" in message
    assert "excluded 1 by raw clean-anchor gate" in message


def test_explicit_cycle_failing_clean_anchor_has_precise_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: AnchorDataset())

    with pytest.raises(
        ValueError, match="cycle_b excluded: clean anchor has 47 complete rows; requires 48"
    ):
        main_cost.main(
            [
                "--action",
                "calculate",
                "--cost",
                "v1",
                "--cycles",
                "cycle_b",
                "--output-root",
                str(tmp_path),
            ]
        )


def test_explicit_cycle_with_unordered_boundaries_has_precise_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class UnorderedDataset(MetadataOnlyDataset):
        def list_cycles(self, **_: object) -> pd.DataFrame:
            values = super().list_cycles()
            values.loc[0, "defrost_start"] = "2026-01-01 00:18:00"
            return values.iloc[:1]

    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: UnorderedDataset())

    with pytest.raises(ValueError, match="cycle_a excluded: required boundaries are not ordered"):
        main_cost.main(
            [
                "--action",
                "calculate",
                "--cost",
                "v1",
                "--cycles",
                "cycle_a",
                "--output-root",
                str(tmp_path),
                "--dry-run",
            ]
        )


def test_dry_run_loads_empirical_parameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: MetadataOnlyDataset())
    monkeypatch.setattr(
        main_cost,
        "load_parameters",
        lambda: (_ for _ in ()).throw(ValueError("bad empirical parameters")),
    )

    with pytest.raises(ValueError, match="bad empirical parameters"):
        main_cost.main(
            [
                "--action",
                "calculate",
                "--cost",
                "v1",
                "--cycles",
                "cycle_a",
                "--output-root",
                str(tmp_path),
                "--dry-run",
            ]
        )


def test_dry_run_checks_selected_experiment_has_parameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: AnchorDataset())
    monkeypatch.setattr(
        main_cost,
        "load_parameters",
        lambda: {"pe_quadratic": {}, "v1": {}, "v2.5": {}},
    )

    with pytest.raises(ValueError, match="exp_20260714"):
        main_cost.main(
            [
                "--action",
                "calculate",
                "--cost",
                "v1",
                "--cycles",
                "cycle_a",
                "--output-root",
                str(tmp_path),
                "--dry-run",
            ]
        )


def test_existing_canonical_and_variant_run_directories_require_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: AnchorDataset())
    canonical = tmp_path / "output" / "cost" / "v1"
    canonical.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        main_cost.main(
            [
                "--action",
                "calculate",
                "--cost",
                "v1",
                "--output-root",
                str(tmp_path / "output"),
                "--dry-run",
            ]
        )

    variant = tmp_path / "output" / "cost" / "v1__water_trial"
    variant.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        main_cost.main(
            [
                "--action",
                "calculate",
                "--cost",
                "v1",
                "--variant",
                "water_trial",
                "--heating-heat-model",
                "measured_water_heat",
                "--output-root",
                str(tmp_path / "output"),
                "--dry-run",
            ]
        )


def test_fit_is_scoped_to_named_v268_candidates() -> None:
    with pytest.raises(ValueError, match="only for --cost v2.6.8"):
        main_cost.main(["--action", "fit"])
    with pytest.raises(ValueError, match="requires --variant"):
        main_cost.main(["--action", "fit", "--cost", "v2.6.8"])


def test_fit_dry_run_stops_before_reading_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        main_cost,
        "DatasetLoader",
        lambda _: (_ for _ in ()).throw(AssertionError("dry-run must not read Dataset")),
    )

    status = main_cost.main(
        [
            "--action",
            "fit",
            "--cost",
            "v2.6.8",
            "--variant",
            "dry_run",
            "--output-root",
            str(tmp_path / "output"),
            "--dry-run",
        ]
    )

    assert status == 0
    assert "Fit V2.6.8" in capsys.readouterr().out
    assert not (tmp_path / "output").exists()


def test_compare_dry_run_stops_before_rendering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        main_cost,
        "generate_cost_function_figures",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run must not render")
        ),
    )

    status = main_cost.main(
        [
            "--action",
            "compare",
            "--results",
            str(tmp_path / "run_a"),
            str(tmp_path / "run_b"),
            "--output-root",
            str(tmp_path / "output"),
            "--dry-run",
        ]
    )

    assert status == 0
    assert "Compare dry-run" in capsys.readouterr().out
    assert not (tmp_path / "output").exists()


def test_action_is_a_required_option_not_a_positional() -> None:
    parser = main_cost.build_parser()

    assert parser.parse_args(["--action", "calculate"]).action == "calculate"
    parsed = parser.parse_args(
        [
            "--action",
            "calculate",
            "--integration-protocol",
            "strict_causal",
            "--state-protocol",
            "strict_causal",
        ]
    )
    assert parsed.integration_protocol == "strict_causal"
    assert parsed.state_protocol == "strict_causal"
    with pytest.raises(SystemExit):
        parser.parse_args(["calculate"])


def test_policy_dry_run_validates_four_targets_and_defaults_without_raw_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: MetadataOnlyDataset())
    parameters = _policy_artifact(tmp_path / "models.json")

    status = main_cost.main(
        [
            "--action",
            "policy",
            "--parameters",
            str(parameters),
            "--variant",
            "offline_a",
            "--cycles",
            "cycle_a",
            "--output-root",
            str(tmp_path / "output"),
            "--dry-run",
        ]
    )

    assert status == 0
    output = capsys.readouterr().out
    assert "Policy dry-run" in output
    assert "candidate_step_seconds=10" in output
    assert "n_jobs=6" in output
    assert not (tmp_path / "output").exists()


def test_policy_requires_parameters_variant_and_positive_jobs(tmp_path: Path) -> None:
    parameters = _policy_artifact(tmp_path / "models.json")
    with pytest.raises(ValueError, match="--parameters"):
        main_cost.main(["--action", "policy", "--variant", "x", "--dry-run"])
    with pytest.raises(ValueError, match="--variant"):
        main_cost.main(
            ["--action", "policy", "--parameters", str(parameters), "--dry-run"]
        )
    with pytest.raises(ValueError, match="n-jobs"):
        main_cost.main(
            [
                "--action",
                "policy",
                "--parameters",
                str(parameters),
                "--variant",
                "x",
                "--n-jobs",
                "0",
                "--dry-run",
            ]
        )


def test_policy_uses_single_parallel_layer_and_shared_run_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parameters = _policy_artifact(tmp_path / "models.json")
    calls: dict[str, object] = {}
    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: AnchorDataset())
    monkeypatch.setattr(
        main_cost,
        "_clean_anchor_cycles",
        lambda _loader, cycles, *, explicit: (cycles, 0),
    )
    monkeypatch.setattr(
        main_cost.cho,
        "calculate_cycle",
        lambda _loader, cycle, _artifacts, **kwargs: (
            calls.setdefault("workers", []).append((cycle, kwargs))
            or pd.DataFrame(
                {
                    "cycle_name": [cycle, cycle],
                    "candidate_time": ["2026-01-01 00:00:20", "2026-01-01 00:00:10"],
                    "C": [2.0, 2.1],
                    "H": [3.0, 2.9],
                }
            )
        ),
    )
    monkeypatch.setattr(
        main_cost,
        "parallel_config",
        lambda **kwargs: calls.update(config=kwargs) or nullcontext(),
    )

    def fake_parallel(**kwargs):
        calls["parallel"] = kwargs

        def run(tasks):
            for function, args, keywords in reversed(list(tasks)):
                yield function(*args, **keywords)

        return run

    monkeypatch.setattr(main_cost, "Parallel", fake_parallel)
    monkeypatch.setattr(
        main_cost,
        "delayed",
        lambda function: lambda *args, **kwargs: (function, args, kwargs),
    )

    assert (
        main_cost.main(
            [
                "--action",
                "policy",
                "--parameters",
                str(parameters),
                "--variant",
                "offline_a",
                "--candidate-step-seconds",
                "20",
                "--n-jobs",
                "2",
                "--output-root",
                str(tmp_path / "output"),
            ]
        )
        == 0
    )

    assert calls["config"] == {"backend": "loky", "n_jobs": 2, "inner_max_num_threads": 1}
    assert calls["parallel"] == {
        "return_as": "generator_unordered",
        "batch_size": "auto",
        "pre_dispatch": "2*n_jobs",
    }
    assert calls["workers"] == [
        ("cycle_b", {"step_seconds": 20, "allow_extrapolation": True}),
        ("cycle_a", {"step_seconds": 20, "allow_extrapolation": True}),
    ]
    run = tmp_path / "output/cost/policy/offline_a"
    assert {path.name for path in run.iterdir()} == {
        "cost.csv",
        "cycles",
        "recipe.json",
        "command.txt",
    }
    assert (run / "cycles/cycle_a.csv").exists()
    assert (run / "cycles/cycle_b.csv").exists()
    cost = pd.read_csv(run / "cost.csv")
    assert list(zip(cost["cycle_name"], cost["candidate_time"], strict=True)) == [
        ("cycle_a", "2026-01-01 00:00:10"),
        ("cycle_a", "2026-01-01 00:00:20"),
        ("cycle_b", "2026-01-01 00:00:10"),
        ("cycle_b", "2026-01-01 00:00:20"),
    ]
    recipe = json.loads((run / "recipe.json").read_text())
    assert recipe["working_mode"] == "offline_policy"
    assert recipe["O_role"] == "reference_only"
    assert recipe["label_eligible"] is False


def test_calculate_writes_cost_command_and_recipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Module:
        DEFAULT_RECIPE = main_cost.cost_function_v1.DEFAULT_RECIPE
        validate_recipe = staticmethod(main_cost.cost_function_v1.validate_recipe)

        @staticmethod
        def calculate(*_: object) -> pd.DataFrame:
            return pd.DataFrame({"cycle_name": ["cycle_a", "cycle_b"], "inverse_cop": [0.5, 0.6]})

    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: AnchorDataset())
    monkeypatch.setitem(main_cost.COST_MODULES, "v1", Module)
    output = tmp_path / "output"

    assert (
        main_cost.main(["--action", "calculate", "--cost", "v1", "--output-root", str(output)]) == 0
    )

    run = output / "cost" / "v1"
    assert (run / "cost.csv").exists()
    assert (run / "command.txt").exists()
    assert (run / "cycles/cycle_a.csv").exists()
    assert (run / "cycles/cycle_b.csv").exists()
    assert (
        (run / "command.txt")
        .read_text()
        .startswith("uv run python main_cost.py --action calculate")
    )
    recipe = json.loads((run / "recipe.json").read_text())
    assert recipe["heat_basis"] == "unit"
    assert recipe["integration_protocol"] == "historical_reconstruction"
    assert recipe["state_protocol"] == "historical_interpolation"


def test_strict_protocol_dry_run_requires_named_variant_and_records_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Module:
        DEFAULT_RECIPE = main_cost.cost_function_v1.DEFAULT_RECIPE
        validate_recipe = staticmethod(main_cost.cost_function_v1.validate_recipe)

        @staticmethod
        def calculate(_: object, cycles: list[str], __: object) -> pd.DataFrame:
            return pd.DataFrame({"cycle_name": cycles, "inverse_cop": 0.5})

    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: AnchorDataset())
    monkeypatch.setitem(main_cost.COST_MODULES, "v1", Module)
    common = [
        "--action",
        "calculate",
        "--cost",
        "v1",
        "--cycles",
        "cycle_a",
        "--integration-protocol",
        "strict_causal",
        "--state-protocol",
        "strict_causal",
        "--output-root",
        str(tmp_path),
        "--dry-run",
    ]

    with pytest.raises(ValueError, match="named variant"):
        main_cost.main(common)

    assert main_cost.main([*common, "--variant", "strict"]) == 0
    output = capsys.readouterr().out
    assert "integration_protocol=strict_causal" in output
    assert "state_protocol=strict_causal" in output

    calculate = [value for value in common if value != "--dry-run"]
    assert main_cost.main([*calculate, "--variant", "strict"]) == 0
    recipe = json.loads((tmp_path / "cost/v1__strict/recipe.json").read_text())
    assert recipe["integration_protocol"] == "strict_causal"
    assert recipe["state_protocol"] == "strict_causal"
    assert recipe["label_eligible"] is False


def test_overwrite_removes_stale_per_cycle_csvs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Module:
        DEFAULT_RECIPE = main_cost.cost_function_v1.DEFAULT_RECIPE
        validate_recipe = staticmethod(main_cost.cost_function_v1.validate_recipe)

        @staticmethod
        def calculate(_: object, cycles: list[str], __: object) -> pd.DataFrame:
            return pd.DataFrame({"cycle_name": cycles, "inverse_cop": 0.5})

    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: AnchorDataset())
    monkeypatch.setitem(main_cost.COST_MODULES, "v1", Module)
    output = tmp_path / "output"
    common = ["--action", "calculate", "--cost", "v1", "--output-root", str(output)]

    assert main_cost.main(common) == 0
    assert main_cost.main([*common, "--cycles", "cycle_a", "--overwrite"]) == 0

    assert {path.name for path in (output / "cost/v1/cycles").iterdir()} == {"cycle_a.csv"}


@pytest.mark.parametrize(
    ("cost", "overrides", "expected"),
    [
        (
            "v1",
            ["--transition-heat-model", "linear_qprep_plus_signed_quadratic_qd"],
            {
                "transition_scope": "preparation_defrost_recovery",
                "transition_window": "observed_preparation_and_defrost_durations",
                "transition_provenance": (
                    "offline_diagnostic_future_boundary_observed_durations_plus_fixed_recovery"
                ),
            },
        ),
        (
            "v1",
            ["--transition-energy-model", "pe_quadratic"],
            {
                "transition_scope": "preparation_defrost_recovery",
                "transition_window": "candidate_state_at_tau",
                "transition_provenance": "candidate_time_state",
            },
        ),
        (
            "v2.5",
            ["--transition-heat-model", "zero_transition_heat"],
            {
                "transition_scope": "preparation_defrost_recovery",
                "transition_window": "candidate_state_at_tau",
                "transition_provenance": "candidate_time_state",
            },
        ),
    ],
)
def test_variant_recipe_metadata_matches_selected_transition_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cost: str,
    overrides: list[str],
    expected: dict[str, str],
) -> None:
    source = main_cost.COST_MODULES[cost]

    class Module:
        DEFAULT_RECIPE = source.DEFAULT_RECIPE
        validate_recipe = staticmethod(source.validate_recipe)

        @staticmethod
        def calculate(*_: object) -> pd.DataFrame:
            return pd.DataFrame({"cycle_name": ["cycle_a"], "inverse_cop": [0.5]})

    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: AnchorDataset())
    monkeypatch.setitem(main_cost.COST_MODULES, cost, Module)
    output = tmp_path / cost

    assert (
        main_cost.main(
            [
                "--action",
                "calculate",
                "--cost",
                cost,
                "--variant",
                "semantic_trial",
                "--cycles",
                "cycle_a",
                "--output-root",
                str(output),
                *overrides,
            ]
        )
        == 0
    )

    recipe = json.loads((output / "cost" / f"{cost}__semantic_trial" / "recipe.json").read_text())
    assert {key: recipe[key] for key in expected} == expected


def test_compare_delegates_results_and_dataset_to_moved_cost_plotter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = [tmp_path / "v1", tmp_path / "v25"]
    loader = object()
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        main_cost, "DatasetLoader", lambda dataset: (calls.append((dataset,)), loader)[1]
    )
    monkeypatch.setattr(
        main_cost,
        "generate_cost_function_figures",
        lambda result_dirs, actual_loader, output, *, overwrite: calls.append(
            (result_dirs, actual_loader, output, overwrite)
        ),
    )

    assert (
        main_cost.main(
            [
                "--action",
                "compare",
                "--dataset",
                str(tmp_path / "dataset"),
                "--results",
                *(str(run) for run in runs),
                "--output-root",
                str(tmp_path / "output"),
                "--overwrite",
            ]
        )
        == 0
    )

    assert calls == [
        (tmp_path / "dataset",),
        (runs, loader, tmp_path / "output" / "plots", True),
    ]
