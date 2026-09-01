"""Calculate and compare Dataset-native defrost cost curves."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pandas as pd

from cost import cost_function_v1, cost_function_v2_5
from cost.boundaries import catalog_exclusion_reason, clean_anchor_exclusion_reason
from cost.cost_curve import transition_semantics, validate_recipe
from cost.energy_models import load_parameters
from dataloader import DatasetLoader

if TYPE_CHECKING:
    from matplotlib.figure import Figure

COST_MODULES: dict[str, ModuleType] = {
    "v1": cost_function_v1,
    "v2.5": cost_function_v2_5,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=("calculate", "fit", "compare"))
    parser.add_argument("--cost", choices=tuple(COST_MODULES), default="v1")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--cycles", nargs="*")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--results", nargs="+", type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--heat-basis", choices=("unit", "water"))
    parser.add_argument(
        "--event-scope",
        choices=(
            "stable_heating_start_to_actual_preparation",
            "heating_start_to_actual_preparation",
        ),
    )
    parser.add_argument("--heating-start-rule", choices=("stable_heating_start", "heating_start"))
    parser.add_argument("--heating-energy-model", choices=("measured_total_power",))
    parser.add_argument(
        "--heating-heat-model", choices=("measured_unit_heat", "measured_water_heat")
    )
    parser.add_argument(
        "--transition-energy-model",
        choices=("pe_quadratic_plus_fixed_recovery", "pe_quadratic"),
    )
    parser.add_argument(
        "--transition-heat-model",
        choices=("zero_transition_heat", "linear_qprep_plus_signed_quadratic_qd"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _recipe(module: Any, args: argparse.Namespace) -> dict[str, object]:
    recipe = dict(module.DEFAULT_RECIPE)
    recipe["variant"] = args.variant
    for argument, key in (
        (args.heat_basis, "heat_basis"),
        (args.event_scope, "event_scope"),
        (args.heating_start_rule, "heating_start_rule"),
        (args.heating_energy_model, "heating_energy_model"),
        (args.heating_heat_model, "heating_heat_model"),
        (args.transition_energy_model, "transition_energy_model"),
        (args.transition_heat_model, "transition_heat_model"),
    ):
        if argument is not None:
            recipe[key] = argument
    recipe.update(
        transition_semantics(recipe["transition_energy_model"], recipe["transition_heat_model"])
    )
    return validate_recipe(recipe)


def _run_directory(output_root: Path, recipe: dict[str, object]) -> Path:
    variant = recipe["variant"]
    if variant is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", str(variant)):
        raise ValueError("variant may contain only letters, numbers, dot, underscore, and hyphen")
    name = str(recipe["base_cost"]) if variant is None else f"{recipe['base_cost']}__{variant}"
    return output_root / "cost" / name


def _cycle_names(
    loader: Any, requested: list[str] | None, parameter_experiments: set[str]
) -> list[str]:
    catalog = loader.list_cycles(statuses={"valid"})
    available = [str(value) for value in catalog["cycle_name"].tolist()]
    if requested is None or not requested:
        selected = available
    else:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"unknown or invalid cycles: {', '.join(missing)}")
        selected = requested
    records = {str(row["cycle_name"]): row.to_dict() for _, row in catalog.iterrows()}
    eligible = []
    for cycle_name in selected:
        reason = catalog_exclusion_reason(records[cycle_name], parameter_experiments)
        if reason is not None:
            if requested:
                raise ValueError(f"{cycle_name} excluded: {reason}")
            continue
        eligible.append(cycle_name)
    if not eligible:
        raise ValueError("no metadata-eligible cycles selected")
    return eligible


def _clean_anchor_cycles(
    loader: Any, cycles: list[str], *, explicit: bool
) -> tuple[list[str], int]:
    selected = []
    columns = [
        "timestamp",
        "water_flow",
        "water_in_temperature",
        "water_out_temperature",
        "power_total",
    ]
    for cycle_name in cycles:
        frame = loader.load_cycle_original(cycle_name, columns=columns)
        reason = clean_anchor_exclusion_reason(frame, loader.get_cycle_record(cycle_name))
        if reason is not None:
            if explicit:
                raise ValueError(f"{cycle_name} excluded: {reason}")
            continue
        selected.append(cycle_name)
    if not selected:
        raise ValueError("no cycles pass the raw clean-anchor gate")
    return selected, len(cycles) - len(selected)


def _validate_cycle_artifact_names(table: pd.DataFrame) -> None:
    for value in table["cycle_name"]:
        if (
            not isinstance(value, str)
            or not value.strip()
            or value in {".", ".."}
            or Path(value).is_absolute()
            or Path(value).name != value
            or "/" in value
            or "\\" in value
        ):
            raise ValueError(f"unsafe cycle name for artifact: {value!r}")


def compare_results(
    result_dirs: Sequence[Path], output_dir: Path, *, overwrite: bool = False
) -> tuple[Figure, Path]:
    """Plot relative regret together and absolute inverse COP in heat-basis panels."""
    import matplotlib.pyplot as plt

    path = output_dir / "cost_comparison.png"
    if path.exists() and not overwrite:
        raise FileExistsError(f"comparison exists; pass --overwrite: {path}")
    loaded: list[tuple[str, str, pd.DataFrame]] = []
    for directory in result_dirs:
        recipe = json.loads((directory / "recipe.json").read_text(encoding="utf-8"))
        basis = recipe.get("heat_basis")
        if basis not in {"unit", "water"}:
            raise ValueError(f"result has no explicit heat basis: {directory}")
        loaded.append((directory.name, str(basis), pd.read_csv(directory / "cost.csv")))
    bases = sorted({basis for _, basis, _ in loaded})
    figure, axes = plt.subplots(1 + len(bases), 1, figsize=(7, 3 * (1 + len(bases))))
    axes_list = list(axes) if hasattr(axes, "__len__") else [axes]
    for label, _, table in loaded:
        for _, cycle in table.groupby("cycle_name", sort=False):
            axes_list[0].plot(
                cycle["candidate_elapsed_minutes"], cycle["relative_regret"], label=label
            )
    axes_list[0].set_ylabel("Relative regret")
    axes_list[0].set_title("Cost comparison — relative regret")
    for axis, basis in zip(axes_list[1:], bases, strict=True):
        for label, result_basis, table in loaded:
            if result_basis != basis:
                continue
            for _, cycle in table.groupby("cycle_name", sort=False):
                axis.plot(cycle["candidate_elapsed_minutes"], cycle["inverse_cop"], label=label)
        axis.set_ylabel("Inverse COP")
        axis.set_title(f"Absolute inverse COP — {basis} heat basis")
    axes_list[-1].set_xlabel("Candidate elapsed time (min)")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    return figure, path


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    arguments = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(arguments)
    if args.action == "fit":
        print("V2.6.8 fit not migrated yet", file=sys.stderr)
        return 2
    if args.action == "compare":
        if not args.results:
            raise ValueError("compare requires --results run directories")
        _, path = compare_results(
            args.results, args.output_root / "plots", overwrite=args.overwrite
        )
        print(f"Comparison written: {path}")
        return 0

    module = COST_MODULES[args.cost]
    recipe = _recipe(module, args)
    parameters = load_parameters()
    if not {"pe_quadratic", "v1", "v2.5"} <= parameters.keys():
        raise ValueError("empirical parameters are incomplete")
    run = _run_directory(args.output_root, recipe)
    if run.exists() and not args.overwrite:
        raise FileExistsError(f"run directory exists; pass --overwrite: {run}")
    loader = DatasetLoader(args.dataset)
    cycles = _cycle_names(loader, args.cycles, set(parameters["pe_quadratic"]))
    if (
        recipe["transition_energy_model"] == "pe_quadratic_plus_fixed_recovery"
        and "fixed_recovery_electricity_kwh" not in parameters["v1"]
    ):
        raise ValueError("V1 fixed recovery parameter is missing")
    if (
        recipe["transition_heat_model"] == "linear_qprep_plus_signed_quadratic_qd"
        and not {
            "preparation_heat",
            "defrost_heat",
        }
        <= parameters["v2.5"].keys()
    ):
        raise ValueError("V2.5 transition heat parameters are incomplete")
    if args.dry_run:
        print(
            "Dry-run OK: recipe, parameters, variant, output, and "
            f"{len(cycles)} metadata-eligible cycle(s) checked; "
            "raw clean-anchor gate deferred"
        )
        return 0

    cycles, anchor_excluded = _clean_anchor_cycles(loader, cycles, explicit=bool(args.cycles))
    print(f"Selected {len(cycles)} cycle(s); excluded {anchor_excluded} by raw clean-anchor gate")
    table = module.calculate(loader, cycles, recipe)
    _validate_cycle_artifact_names(table)
    run.mkdir(parents=True, exist_ok=True)
    table.to_csv(run / "cost.csv", index=False)
    cycles_dir = run / "cycles"
    cycles_dir.mkdir(exist_ok=True)
    if args.overwrite:
        for stale in cycles_dir.glob("*.csv"):
            stale.unlink()
    for cycle_name, cycle in table.groupby("cycle_name", sort=False):
        cycle.to_csv(cycles_dir / f"{cycle_name}.csv", index=False)
    (run / "recipe.json").write_text(
        json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run / "command.txt").write_text(
        shlex.join(["uv", "run", "python", "main_cost.py", *arguments]) + "\n",
        encoding="utf-8",
    )
    print(f"Cost result written: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
