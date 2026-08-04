"""Command line entry points for the explicit three-stage pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from frost_analysis import run_pipeline
from frost_analysis.analysis import run_analysis
from frost_analysis.channels import load_channels
from frost_analysis.config import load_config
from frost_analysis.dataset import add_dataset, append_dataset, build_dataset
from frost_analysis.dataset_loader import DatasetLoader
from frost_analysis.dataset_manifest import edit_dataset, refresh_manifest, review_cycle
from frost_analysis.dataset_v3 import resolve_project_root
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
from frost_analysis.visualization import (
    generate_cycle_publication,
    generate_rgb_coverage,
)


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
        dataset_path = arguments.dataset or resolve_project_root() / "dataset"
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

    build_parser = dataset_commands.add_parser("build")
    build_parser.add_argument("--run", action="append", required=True, type=Path)
    build_parser.add_argument("--output", required=True, type=Path)

    append_parser = dataset_commands.add_parser("append")
    append_parser.add_argument("--run", required=True, type=Path)
    append_parser.add_argument("--dataset", required=True, type=Path)

    validate_parser = dataset_commands.add_parser("validate")
    validate_parser.add_argument("--input", type=Path)
    validate_parser.add_argument("--dataset", type=Path)
    validate_parser.add_argument("--require-assigned", action="store_true")

    add_parser = dataset_commands.add_parser("add")
    add_parser.add_argument("input_dir", nargs="?", type=Path)
    add_parser.add_argument("--run", type=Path)
    add_parser.add_argument("--dataset", type=Path)

    refresh_parser = dataset_commands.add_parser("refresh-manifest")
    refresh_parser.add_argument("--dataset", type=Path)
    refresh_parser = dataset_commands.add_parser("refresh")
    refresh_parser.add_argument("--dataset", type=Path)

    review_parser = dataset_commands.add_parser("review-cycle")
    review_parser.add_argument("--dataset", type=Path)
    review_parser.add_argument("cycle", nargs="?")
    review_parser.add_argument("--cycle", dest="cycle_option")
    review_parser.add_argument(
        "--status",
        required=True,
        choices=["valid", "partial", "incomplete", "invalid"],
    )
    review_parser.add_argument("--note")

    edit_parser = dataset_commands.add_parser("edit")
    edit_parser.add_argument("--dataset", type=Path)
    edit_parser.add_argument("--baseline-seconds", type=int)
    recovery_group = edit_parser.add_mutually_exclusive_group()
    recovery_group.add_argument("--recovery-seconds", type=int)
    recovery_group.add_argument("--recovery-end-by", choices=["ts-minus"])
    edit_parser.add_argument("--status", action="append", default=[])
    edit_parser.add_argument("--rename-camera", action="append", default=[])

    render_parser = dataset_commands.add_parser("render")
    render_parser.add_argument("--dataset", type=Path)
    render_parser.add_argument("cycle", nargs="?", type=str)
    render_parser.add_argument("--cycle", dest="cycle_option")
    render_parser.add_argument("--publication", action="store_true")
    render_parser.add_argument("--coverage", action="store_true")


def _run_dataset_command(arguments: argparse.Namespace) -> int:  # noqa: C901
    if arguments.dataset_command == "add":
        if arguments.input_dir is not None:
            print(add_dataset(arguments.input_dir, arguments.dataset))
            return 0
        if arguments.run is None or arguments.dataset is None:
            raise ValueError("dataset add requires INPUT_DIR or --run with --dataset")
        print(add_dataset(arguments.run, arguments.dataset))
        return 0
    if arguments.dataset_command in {"refresh-manifest", "refresh"}:
        dataset = arguments.dataset or (resolve_project_root() / "dataset")
        print(refresh_manifest(dataset))
        return 0
    if arguments.dataset_command == "review-cycle":
        dataset = arguments.dataset or (resolve_project_root() / "dataset")
        cycle = arguments.cycle_option or arguments.cycle
        if cycle is None:
            raise ValueError("dataset review-cycle requires a cycle name")
        review_cycle(
            dataset,
            cycle,
            status=arguments.status,
            note=arguments.note,
        )
        print(dataset)
        return 0
    if arguments.dataset_command == "edit":
        dataset = arguments.dataset or (resolve_project_root() / "dataset")
        print(
            edit_dataset(
                dataset,
                baseline_seconds=arguments.baseline_seconds,
                recovery_seconds=arguments.recovery_seconds,
                recovery_end_by=arguments.recovery_end_by,
                statuses=arguments.status,
                camera_renames=arguments.rename_camera,
            )
        )
        return 0
    if arguments.dataset_command == "render":
        dataset = arguments.dataset or (resolve_project_root() / "dataset")
        cycle = arguments.cycle_option or arguments.cycle
        if cycle is None:
            raise ValueError("dataset render requires a cycle name")
        loader = DatasetLoader(dataset)
        if not arguments.publication and not arguments.coverage:
            arguments.publication = True
            arguments.coverage = True
        if arguments.publication:
            generate_cycle_publication(loader, cycle)
        if arguments.coverage:
            generate_rgb_coverage(loader, cycle)
        print(dataset)
        return 0
    if arguments.dataset_command == "build":
        print(build_dataset(arguments.run, arguments.output))
        return 0
    if arguments.dataset_command == "append":
        print(append_dataset(arguments.run, arguments.dataset))
        return 0
    dataset = arguments.input or arguments.dataset or (resolve_project_root() / "dataset")
    validate_dataset(dataset)
    if arguments.require_assigned:
        _require_assigned_roles(dataset)
    manifest = json.loads(
        (dataset / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("dataset_schema_version") == 2:
        cycle_index = pd.read_parquet(dataset / "cycle_index.parquet")
        image_metadata = pd.read_parquet(dataset / "image_metadata.parquet")
        print(
            "dataset valid: "
            f"{len(cycle_index)} cycles, {len(image_metadata)} images"
        )
        return 0
    if manifest.get("dataset_schema_version") == 3:
        cycle_index = pd.read_parquet(dataset / "cycle_index.parquet")
        image_metadata = pd.read_parquet(dataset / "image_metadata.parquet")
        print(
            "dataset valid: "
            f"{len(cycle_index)} cycles, "
            f"{len(image_metadata)} images"
        )
    else:
        print(
            "dataset valid: "
            f"{manifest.get('published_cycle_count', 0)} published cycles, "
            f"{manifest.get('image_count', 0)} images"
        )
    return 0


def _require_assigned_roles(dataset: Path) -> None:
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_schema_version") == 3:
        from frost_analysis.dataset_validation import validate_canonical_dataset

        try:
            validate_canonical_dataset(dataset, require_assigned=True)
        except ValueError as error:
            if "canonical Dataset manifest" not in str(error):
                raise
            from frost_analysis.dataset_validation_v3 import validate_v3_dataset

            validate_v3_dataset(dataset, require_assigned=True)
        return
    image_root = dataset / "images"
    unassigned = sorted(
        path.as_posix()
        for path in image_root.glob("*/unassigned_*")
        if path.is_dir()
    )
    if unassigned:
        raise ValueError(f"unassigned camera roles remain: {unassigned}")


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
