from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import main_cost


class MetadataOnlyDataset:
    def list_cycles(self, **_: object) -> pd.DataFrame:
        return pd.DataFrame({"cycle_name": ["cycle_a", "cycle_b"]})

    def load_cycle_original(self, *_: object, **__: object) -> pd.DataFrame:
        raise AssertionError("dry-run must not read cycle time series")

    def get_cycle_record(self, cycle_name: str) -> dict[str, str]:
        assert cycle_name in {"cycle_a", "cycle_b"}
        return {"experiment_id": "exp_20260714"}


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
    assert "1 cycle" in output
    assert not (tmp_path / "output").exists()


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
                "--output-root",
                str(tmp_path),
                "--dry-run",
            ]
        )


def test_dry_run_checks_selected_experiment_has_parameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: MetadataOnlyDataset())
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
                "--output-root",
                str(tmp_path),
                "--dry-run",
            ]
        )


def test_existing_canonical_and_variant_run_directories_require_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: MetadataOnlyDataset())
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
                "--heat-basis",
                "water",
                "--heating-heat-model",
                "measured_water_heat",
                "--output-root",
                str(tmp_path / "output"),
                "--dry-run",
            ]
        )


def test_fit_is_explicitly_reserved(capsys: pytest.CaptureFixture[str]) -> None:
    assert main_cost.main(["--action", "fit"]) == 2
    assert "V2.6.8 fit not migrated yet" in capsys.readouterr().err


def test_action_is_a_required_option_not_a_positional() -> None:
    parser = main_cost.build_parser()

    assert parser.parse_args(["--action", "calculate"]).action == "calculate"
    with pytest.raises(SystemExit):
        parser.parse_args(["calculate"])


def test_calculate_writes_cost_command_and_recipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Module:
        DEFAULT_RECIPE = main_cost.cost_function_v1.DEFAULT_RECIPE

        @staticmethod
        def calculate(*_: object) -> pd.DataFrame:
            return pd.DataFrame({"cycle_name": ["cycle_a", "cycle_b"], "inverse_cop": [0.5, 0.6]})

    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: MetadataOnlyDataset())
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
    assert json.loads((run / "recipe.json").read_text())["heat_basis"] == "unit"


def test_overwrite_removes_stale_per_cycle_csvs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Module:
        DEFAULT_RECIPE = main_cost.cost_function_v1.DEFAULT_RECIPE

        @staticmethod
        def calculate(_: object, cycles: list[str], __: object) -> pd.DataFrame:
            return pd.DataFrame({"cycle_name": cycles, "inverse_cop": 0.5})

    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: MetadataOnlyDataset())
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

        @staticmethod
        def calculate(*_: object) -> pd.DataFrame:
            return pd.DataFrame({"cycle_name": ["cycle_a"], "inverse_cop": [0.5]})

    monkeypatch.setattr(main_cost, "DatasetLoader", lambda _: MetadataOnlyDataset())
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


def test_compare_separates_absolute_inverse_cop_by_heat_basis(tmp_path: Path) -> None:
    runs = []
    for name, basis, cost in (("v1", "unit", 0.5), ("v25", "water", 0.4)):
        run = tmp_path / name
        run.mkdir()
        (run / "recipe.json").write_text(json.dumps({"base_cost": name, "heat_basis": basis}))
        pd.DataFrame(
            {
                "cycle_name": ["cycle_a", "cycle_a"],
                "candidate_elapsed_minutes": [10, 11],
                "relative_regret": [0.1, 0.0],
                "inverse_cop": [cost + 0.1, cost],
            }
        ).to_csv(run / "cost.csv", index=False)
        runs.append(run)

    figure, path = main_cost.compare_results(runs, tmp_path / "plots")

    assert path.exists()
    assert figure.axes[0].get_ylabel() == "Relative regret"
    assert {axis.get_title() for axis in figure.axes[1:]} == {
        "Absolute inverse COP — unit heat basis",
        "Absolute inverse COP — water heat basis",
    }
    with pytest.raises(FileExistsError, match="overwrite"):
        main_cost.compare_results(runs, tmp_path / "plots")
