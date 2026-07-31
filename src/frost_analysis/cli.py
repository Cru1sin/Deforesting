"""Command line entry points for the explicit three-stage pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from frost_analysis import run_pipeline
from frost_analysis.analysis import analyze
from frost_analysis.channels import load_channels
from frost_analysis.config import load_config
from frost_analysis.io import (
    ensure_output_outside_input,
    write_analysis_outputs,
    write_prepare_outputs,
    write_process_outputs,
)
from frost_analysis.prepare import prepare
from frost_analysis.process import process
from frost_analysis.validation import validate_analysis, validate_prepared, validate_processed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="frost_analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    _add_config_and_output(run)
    run.add_argument("--overwrite", action="store_true")
    prepare_parser = subparsers.add_parser("prepare")
    _add_config_and_output(prepare_parser)
    prepare_parser.add_argument("--overwrite", action="store_true")
    process_parser = subparsers.add_parser("process")
    _add_config_input_cycles_output(process_parser)
    process_parser.add_argument("--overwrite", action="store_true")
    analyze_parser = subparsers.add_parser("analyze")
    _add_config_input_cycles_output(analyze_parser)
    analyze_parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        print(run_pipeline(arguments.config, arguments.output, arguments.overwrite))
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
    evidence = analyze(input_frame, cycle_summary, config, channels)
    validate_processed(input_frame, cycle_summary)
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


def _read_cycle_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("heating_start", "stable_heating_start", "defrost_start", "defrost_end"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame
