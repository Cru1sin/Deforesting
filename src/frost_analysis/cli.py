"""Command line entry points for Dataset construction and Evidence analysis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from frost_analysis.config import find_project_root
from frost_analysis.dataset import (
    add_dataset,
    edit_dataset,
    rebuild_dataset,
    refresh_dataset,
    render_dataset,
    review_cycle,
)
from frost_analysis.dataset_loader import DatasetLoader
from frost_analysis.dataset_validation import validate_dataset
from frost_analysis.evidence import EvidenceSettings, build_evidence, write_evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="frost_analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_dataset_commands(subparsers)
    evidence_parser = subparsers.add_parser("evidence")
    evidence_parser.add_argument("--dataset", type=Path)
    evidence_parser.add_argument("--config", required=True, type=Path)
    evidence_parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "dataset":
        return _run_dataset_command(arguments)
    dataset_path = arguments.dataset or (_project_root() / "dataset")
    loader = DatasetLoader(dataset_path)
    settings = EvidenceSettings.from_yaml(arguments.config)
    bundle = build_evidence(loader, settings)
    print(write_evidence(bundle, arguments.output, loader=loader, settings=settings))
    return 0


def _add_dataset_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    dataset = subparsers.add_parser("dataset")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)

    rebuild_parser = dataset_commands.add_parser("rebuild")
    rebuild_parser.add_argument("input_dirs", nargs="+", type=Path)
    rebuild_parser.add_argument("--dataset", type=Path)

    validate_parser = dataset_commands.add_parser("validate")
    validate_parser.add_argument("--dataset", type=Path)

    add_parser = dataset_commands.add_parser("add")
    add_parser.add_argument("input_dir", type=Path)
    add_parser.add_argument("--dataset", type=Path)

    refresh_parser = dataset_commands.add_parser("refresh")
    refresh_parser.add_argument("--dataset", type=Path)

    review_parser = dataset_commands.add_parser("review-cycle")
    review_parser.add_argument("--dataset", type=Path)
    review_parser.add_argument("cycle")
    review_parser.add_argument(
        "--status",
        required=True,
        choices=["valid", "partial", "incomplete", "invalid"],
    )
    review_parser.add_argument("--reason")

    edit_parser = dataset_commands.add_parser("edit")
    edit_parser.add_argument("--dataset", type=Path)
    edit_parser.add_argument("--baseline-seconds", type=int)
    recovery_group = edit_parser.add_mutually_exclusive_group()
    recovery_group.add_argument("--recovery-seconds", type=int)
    recovery_group.add_argument("--recovery-end-by", choices=["ts-minus"])

    render_parser = dataset_commands.add_parser("render")
    render_parser.add_argument("--dataset", type=Path)
    render_parser.add_argument("cycle")
    render_parser.add_argument("--publication", action="store_true")
    render_parser.add_argument("--coverage", action="store_true")


def _run_dataset_command(arguments: argparse.Namespace) -> int:  # noqa: C901
    if arguments.dataset_command == "add":
        print(add_dataset(arguments.input_dir, arguments.dataset))
        return 0
    if arguments.dataset_command == "rebuild":
        print(rebuild_dataset(arguments.input_dirs, arguments.dataset))
        return 0
    if arguments.dataset_command == "refresh":
        dataset = arguments.dataset or (_project_root() / "dataset")
        print(refresh_dataset(dataset))
        return 0
    if arguments.dataset_command == "review-cycle":
        dataset = arguments.dataset or (_project_root() / "dataset")
        review_cycle(
            dataset,
            arguments.cycle,
            status=arguments.status,
            reason=arguments.reason,
        )
        print(dataset)
        return 0
    if arguments.dataset_command == "edit":
        dataset = arguments.dataset or (_project_root() / "dataset")
        print(
            edit_dataset(
                dataset,
                baseline_seconds=arguments.baseline_seconds,
                recovery_seconds=arguments.recovery_seconds,
                recovery_end_by=arguments.recovery_end_by,
            )
        )
        return 0
    if arguments.dataset_command == "render":
        dataset = arguments.dataset or (_project_root() / "dataset")
        print(
            render_dataset(
                dataset,
                arguments.cycle,
                publication=arguments.publication or not arguments.coverage,
                coverage=arguments.coverage or not arguments.publication,
            )
        )
        return 0
    dataset = arguments.dataset or (_project_root() / "dataset")
    validate_dataset(dataset)
    loader = DatasetLoader(dataset)
    print(
        f"dataset valid: {len(loader.list_cycles())} cycles, "
        f"{len(loader.load_image_metadata())} images"
    )
    return 0


def _project_root() -> Path:
    root = find_project_root(Path(__file__))
    if root is None:
        raise FileNotFoundError("could not find project root")
    return root
