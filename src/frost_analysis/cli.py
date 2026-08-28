"""Command line entry points for Dataset construction and Evidence analysis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from frost_analysis.dataset import DatasetLoader
from frost_analysis.dataset.check import validate_dataset
from frost_analysis.dataset.config import find_project_root
from frost_analysis.dataset.core import (
    add_dataset,
    aggregate_original,
    edit_dataset,
    refresh_dataset,
    remove_dataset,
    render_dataset,
    review_cycle,
)
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

    validate_parser = dataset_commands.add_parser("validate")
    validate_parser.add_argument("--dataset", type=Path)

    add_parser = dataset_commands.add_parser("add")
    add_parser.add_argument("input_dir", type=Path)
    add_parser.add_argument("--dataset", type=Path)

    replace_parser = dataset_commands.add_parser("replace")
    replace_parser.add_argument("input_dir", type=Path)
    replace_parser.add_argument("--dataset", type=Path)

    aggregate_parser = dataset_commands.add_parser("aggregate-original")
    aggregate_parser.add_argument("--dataset", type=Path)
    aggregate_parser.add_argument("--seconds", type=int, default=10)

    remove_parser = dataset_commands.add_parser("remove")
    remove_parser.add_argument("date")
    remove_parser.add_argument("--dataset", type=Path)

    refresh_parser = dataset_commands.add_parser("refresh")
    refresh_parser.add_argument("mode", choices=["roles", "images", "figures", "all"])
    refresh_parser.add_argument("--dataset", type=Path)

    review_parser = dataset_commands.add_parser("review-cycle")
    review_parser.add_argument("--dataset", type=Path)
    review_parser.add_argument("cycle")
    review_parser.add_argument(
        "--status",
        required=True,
        choices=["valid", "invalid"],
    )
    review_parser.add_argument("--reason")
    review_parser.add_argument(
        "--rgb-frost", choices=["valid", "invalid", "not_applicable"]
    )
    review_parser.add_argument(
        "--rgb-defrost", choices=["valid", "invalid", "not_applicable"]
    )

    edit_parser = dataset_commands.add_parser("edit")
    edit_parser.add_argument("--dataset", type=Path)
    edit_parser.add_argument("--baseline-seconds", type=int)
    recovery_group = edit_parser.add_mutually_exclusive_group()
    recovery_group.add_argument("--recovery-seconds", type=int)
    recovery_group.add_argument("--recovery-end-by", choices=["ts-minus"])
    edit_parser.add_argument("--defrost-preparation", action="store_true")
    edit_parser.add_argument("--skip-rgb-panels", action="store_true")

    render_parser = dataset_commands.add_parser("render")
    render_parser.add_argument("--dataset", type=Path)
    render_parser.add_argument("cycle")
    render_parser.add_argument("--publication", action="store_true")
    render_parser.add_argument("--panel", action="store_true")
    render_parser.add_argument("--fetch-cloud-images", action="store_true")


def _run_dataset_command(arguments: argparse.Namespace) -> int:  # noqa: C901
    if arguments.dataset_command == "add":
        print(add_dataset(arguments.input_dir, arguments.dataset))
        return 0
    if arguments.dataset_command == "replace":
        from frost_analysis.dataset.core import replace_dataset

        print(replace_dataset(arguments.input_dir, arguments.dataset))
        return 0
    if arguments.dataset_command == "aggregate-original":
        dataset = arguments.dataset or (_project_root() / "dataset")
        print(aggregate_original(dataset, seconds=arguments.seconds))
        return 0
    if arguments.dataset_command == "remove":
        dataset = arguments.dataset or (_project_root() / "dataset")
        print(remove_dataset(dataset, arguments.date))
        return 0
    if arguments.dataset_command == "refresh":
        dataset = arguments.dataset or (_project_root() / "dataset")
        print(refresh_dataset(dataset, arguments.mode))
        return 0
    if arguments.dataset_command == "review-cycle":
        dataset = arguments.dataset or (_project_root() / "dataset")
        review_cycle(
            dataset,
            arguments.cycle,
            status=arguments.status,
            reason=arguments.reason,
            rgb_frost=arguments.rgb_frost,
            rgb_defrost=arguments.rgb_defrost,
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
                defrost_preparation=arguments.defrost_preparation,
                render_rgb_panels=not arguments.skip_rgb_panels,
            )
        )
        return 0
    if arguments.dataset_command == "render":
        dataset = arguments.dataset or (_project_root() / "dataset")
        print(
            render_dataset(
                dataset,
                arguments.cycle,
                publication=arguments.publication or not arguments.panel,
                panel=arguments.panel or not arguments.publication,
                fetch_cloud_images=arguments.fetch_cloud_images,
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
