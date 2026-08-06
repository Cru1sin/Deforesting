"""Command line entry points for the explicit three-stage pipeline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from frost_analysis import run_pipeline
from frost_analysis.analysis import run_analysis
from frost_analysis.channels import load_channels
from frost_analysis.config import find_project_root, load_config
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
from frost_analysis.io import (
    ensure_output_outside_input,
    write_prepare_outputs,
    write_process_outputs,
)
from frost_analysis.prepare import prepare
from frost_analysis.process import process
from frost_analysis.report import generate_report
from frost_analysis.validation import validate_prepared, validate_processed


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    parser = argparse.ArgumentParser(prog="frost_analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    _add_config_and_output(run)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--report", action="store_true")
    prepare_parser = subparsers.add_parser("prepare")
    _add_config_and_output(prepare_parser)
    prepare_parser.add_argument("--overwrite", action="store_true")
    process_parser = subparsers.add_parser("process")
    _add_config_input_cycles_output(process_parser)
    process_parser.add_argument("--overwrite", action="store_true")
    analysis_parser = subparsers.add_parser("analysis")
    _add_dataset_analysis_arguments(analysis_parser)
    report_parser = subparsers.add_parser("report")
    _add_report_input_output(report_parser)
    report_parser.add_argument("--overwrite", action="store_true")
    _add_dataset_commands(subparsers)
    arguments = parser.parse_args(argv)
    if arguments.command == "dataset":
        return _run_dataset_command(arguments)
    if arguments.command == "analysis":
        dataset_path = arguments.dataset or (_project_root() / "dataset")
        loader = DatasetLoader(dataset_path)
        statuses = set(arguments.status) if arguments.status else None
        experiments = set(arguments.experiment) if arguments.experiment else None
        cycle_names = set(arguments.cycle) if arguments.cycle else None
        print(
            run_analysis(
                loader,
                statuses=statuses,
                experiment_ids=experiments,
                cycle_names=cycle_names,
                output_dir=arguments.output,
            )
        )
        return 0
    if arguments.command == "run":
        run_dir = run_pipeline(arguments.config, arguments.output, arguments.overwrite)
        if arguments.report:
            try:
                generate_report(run_dir, run_dir / "qa", overwrite=arguments.overwrite)
            except Exception as error:
                print(
                    f"scientific run succeeded, QA report failed: {error}",
                    file=sys.stderr,
                )
                return 1
        print(run_dir)
        return 0
    if arguments.command == "report":
        try:
            print(generate_report(arguments.input, arguments.output, arguments.overwrite))
        except Exception as error:
            print(f"QA report failed: {error}", file=sys.stderr)
            return 1
        return 0
    config = load_config(arguments.config)
    channels = load_channels(config.channels_path)
    if arguments.command == "prepare":
        prepared, summary, prepare_summary = prepare(config, channels)
        validate_prepared(prepared, summary)
        write_prepare_outputs(
            prepared,
            summary,
            prepare_summary,
            arguments.output,
            config.input_dir,
            overwrite=arguments.overwrite,
        )
        print(arguments.output)
        return 0
    cycle_summary = _read_cycle_summary(arguments.cycles)
    input_frame = pd.read_parquet(arguments.input)
    ensure_output_outside_input(arguments.output, config.input_dir)
    if arguments.command == "process":
        validate_prepared(input_frame, cycle_summary)
        processed, final_summary = process(input_frame, cycle_summary, config, channels)
        validate_processed(processed, final_summary)
        write_process_outputs(
            processed,
            final_summary,
            arguments.output,
            config.input_dir,
            overwrite=arguments.overwrite,
        )
        print(arguments.output)
        return 0
    parser.error(f"unsupported command: {arguments.command}")


def _add_config_and_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)


def _add_config_input_cycles_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--cycles", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)


def _add_dataset_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--status",
        action="append",
        choices=["valid", "partial", "incomplete", "invalid"],
    )
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--cycle", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)


def _add_report_input_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)


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


def _read_cycle_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in (
        "heating_start",
        "stable_heating_start",
        "defrost_start",
        "defrost_end",
        "baseline_start",
        "baseline_end",
    ):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame
