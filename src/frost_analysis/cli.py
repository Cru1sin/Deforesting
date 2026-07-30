"""Command line entry points for the explicit three-stage pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .analysis import analyze
from .channels import load_channels
from .config import load_config
from .io import write_prepare_outputs
from .pipeline import run_pipeline
from .prepare import prepare
from .process import process


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
    analyze_parser = subparsers.add_parser("analyze")
    _add_config_input_cycles_output(analyze_parser)
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        print(run_pipeline(arguments.config, arguments.output, arguments.overwrite))
        return 0
    config = load_config(arguments.config)
    channels = load_channels(config.channels_path)
    if arguments.command == "prepare":
        prepared, summary, prepare_summary = prepare(config, channels)
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
    if arguments.command == "process":
        processed, final_summary = process(input_frame, cycle_summary, config, channels)
        arguments.output.mkdir(parents=True, exist_ok=True)
        processed.to_parquet(arguments.output / "processed_data.parquet", index=False)
        final_summary.to_csv(arguments.output / "cycle_summary.csv", index=False)
        print(arguments.output)
        return 0
    evidence = analyze(input_frame, cycle_summary, config, channels)
    arguments.output.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(arguments.output / "candidate_channel_evidence.csv", index=False)
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
