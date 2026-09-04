"""Calculate the frozen V1 inverse-COP reference used for image labels."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from dataset_tools import DatasetLoader
from defrost_decision.baselines import unit_heat_inverse_cop_v1
from defrost_decision.baselines.electricity import load_parameters
from defrost_decision.candidate_times import clean_anchor_cycles, metadata_eligible_cycles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--cycles", nargs="*")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    run = args.output_root / "defrost_decisions" / "v1_label_reference"
    if run.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {run}")
    parameters = load_parameters()
    experiments = set(parameters["pe_quadratic"])
    loader = DatasetLoader(args.dataset)
    cycles = metadata_eligible_cycles(loader, args.cycles, experiments)
    if args.dry_run:
        print(f"V1 label reference: cycles={len(cycles)}, output={run}")
        return 0
    cycles, excluded = clean_anchor_cycles(loader, cycles, explicit=bool(args.cycles))
    decisions = unit_heat_inverse_cop_v1.calculate(
        loader, cycles, unit_heat_inverse_cop_v1.DEFAULT_RECIPE
    )
    run.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(run / "candidate_decisions.csv", index=False)
    settings = {
        "method": "frozen_v1_inverse_cop_reference",
        "excluded_by_clean_anchor_gate": excluded,
        "command": shlex.join(
            ["uv", "run", "python", "calculate_v1_label_reference.py", *arguments]
        ),
    }
    (run / "run_settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
