"""The one small orchestration entry point for a complete run."""

from __future__ import annotations

from pathlib import Path

from .analysis import analyze
from .channels import load_channels
from .config import load_config
from .io import remove_manifest_for_overwrite, write_run_outputs
from .prepare import prepare
from .process import process
from .validation import validate_analysis, validate_prepared, validate_processed


def run_pipeline(config_path: Path, output_dir: Path, overwrite: bool = False) -> Path:
    """Run Prepare, Process, and Analyze in their documented order."""
    config = load_config(config_path)
    remove_manifest_for_overwrite(output_dir, config.input_dir, overwrite=overwrite)
    channels = load_channels(config.channels_path)
    prepared, initial_summary, prepare_summary = prepare(config, channels)
    validate_prepared(prepared, initial_summary)
    processed, final_summary = process(prepared, initial_summary, config, channels)
    validate_processed(processed, final_summary)
    evidence = analyze(processed, final_summary, config, channels)
    validate_analysis(evidence)
    write_run_outputs(
        prepared,
        processed,
        final_summary,
        evidence,
        prepare_summary,
        config,
        config_path,
        output_dir,
        config.input_dir,
        overwrite=overwrite,
    )
    return output_dir
