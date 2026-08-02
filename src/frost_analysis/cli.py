"""Command line entry points for the explicit three-stage pipeline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from frost_analysis import run_pipeline
from frost_analysis.analysis import analyze
from frost_analysis.channels import load_channels
from frost_analysis.config import find_project_root, load_config, load_evidence_settings
from frost_analysis.evidence import build_evidence_bundle
from frost_analysis.io import (
    ensure_output_outside_input,
    load_evidence_runs,
    optional_sha256,
    write_analysis_outputs,
    write_evidence_outputs,
    write_prepare_outputs,
    write_process_outputs,
)
from frost_analysis.prepare import prepare
from frost_analysis.process import process
from frost_analysis.report import generate_report
from frost_analysis.validation import validate_analysis, validate_prepared, validate_processed


def main(argv: Sequence[str] | None = None) -> int:
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
    analyze_parser = subparsers.add_parser("analyze")
    _add_analyze_arguments(analyze_parser)
    report_parser = subparsers.add_parser("report")
    _add_report_input_output(report_parser)
    report_parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
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
    if arguments.command == "analyze":
        has_run_dirs = bool(arguments.run_dirs)
        has_legacy_inputs = arguments.input is not None or arguments.cycles is not None
        if has_run_dirs and has_legacy_inputs:
            parser.error("analyze cannot combine --run-dir with --input/--cycles")
        if not has_run_dirs and not (arguments.input is not None and arguments.cycles is not None):
            parser.error("analyze requires --input/--cycles or one or more --run-dir")
    if arguments.command == "analyze" and arguments.run_dirs:
        _run_evidence_analyze(arguments)
        print(arguments.output)
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
    validate_processed(input_frame, cycle_summary)
    evidence = analyze(input_frame, cycle_summary, config, channels)
    validate_analysis(evidence)
    write_analysis_outputs(
        evidence,
        arguments.output,
        config.input_dir,
        overwrite=arguments.overwrite,
    )
    print(arguments.output)
    return 0


def _add_config_and_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)


def _add_config_input_cycles_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--cycles", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)


def _add_analyze_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--cycles", type=Path)
    parser.add_argument("--run-dir", action="append", type=Path, dest="run_dirs", default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")


def _add_report_input_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)


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


def _run_evidence_analyze(arguments: argparse.Namespace) -> None:
    run_dirs = [path.resolve() for path in arguments.run_dirs]
    settings = load_evidence_settings(
        arguments.config,
        allow_date_config=len(run_dirs) == 1,
    )
    channels = load_channels(settings.channels_path)
    registry_hash = optional_sha256(settings.channels_path)
    if registry_hash is None:
        raise FileNotFoundError(f"channels registry does not exist: {settings.channels_path}")
    loaded = load_evidence_runs(run_dirs, registry_hash=registry_hash)
    bundle = build_evidence_bundle(
        loaded.processed,
        loaded.cycle_summary,
        settings,
        channels,
        grid_interval_seconds=loaded.grid_interval_seconds,
    )
    legacy_evidence = None
    if len(run_dirs) == 1 and _is_date_config(arguments.config):
        legacy_config = load_config(arguments.config)
        legacy_evidence = analyze(loaded.processed, loaded.cycle_summary, legacy_config, channels)
        validate_analysis(legacy_evidence)
    write_evidence_outputs(
        bundle,
        arguments.output,
        run_dirs,
        settings=settings,
        load_result=loaded,
        candidate_registry_path=settings.channels_path,
        project_root=find_project_root(arguments.config),
        legacy_evidence=legacy_evidence,
        overwrite=arguments.overwrite,
    )


def _is_date_config(path: Path) -> bool:
    try:
        value = _read_yaml(path)
    except OSError:
        return False
    return isinstance(value, dict) and value.get("schema_version") == 2


def _read_yaml(path: Path) -> object:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _is_iso_date(value: str) -> bool:
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False
