"""Command line entry points for the independent data stages and tasks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .analysis.correlation import run_correlation_analysis
from .config import load_app_config
from .pipelines.prepare import prepare_dataset
from .pipelines.process import process_dataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="frost_analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "process"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, type=Path)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--task", required=True, choices=("correlation",))
    analyze.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    config = load_app_config(arguments.config)
    if arguments.command == "prepare":
        output = prepare_dataset(config)
        stage = "prepare"
    elif arguments.command == "process":
        output = process_dataset(config)
        stage = "process"
    else:
        if arguments.task != config.analysis.task:
            raise ValueError(
                f"requested task {arguments.task!r} does not match config task "
                f"{config.analysis.task!r}"
            )
        output = run_correlation_analysis(config)
        stage = "analyze"
    _write_manifest(config.paths.state_dir, stage, output)
    print(output)
    return 0


def _write_manifest(state_dir: Path, stage: str, output: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "run_manifest.json"
    previous: dict[str, object] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except json.JSONDecodeError:
            previous = {}
    stages = previous.get("stages", {})
    if not isinstance(stages, dict):
        stages = {}
    stages[stage] = {"output": str(output), "exists": output.is_file()}
    path.write_text(
        json.dumps({"stages": stages}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
