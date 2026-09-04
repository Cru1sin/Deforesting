"""Build one experiment from raw inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExperimentBuild:
    input_dir: Path
    config: Any
    channels: Mapping[str, Mapping[str, Any]]
    prepared: pd.DataFrame
    summary: pd.DataFrame
    processed: pd.DataFrame
    original: pd.DataFrame | None = None


def build_experiment(input_dir: Path, config: Any) -> ExperimentBuild:
    """Run the existing Prepare -> Process path for one experiment."""
    from .build_cycle_tables import process
    from .channel_mapping import load_channels
    from .prepare_measurements import prepare, prepare_original
    from .validate_prepared_measurements import validate_prepared, validate_processed

    channels = load_channels()
    print("[add] prepare sensors", flush=True)
    prepared, initial_summary = prepare(config, channels)
    print("[add] validate prepared", flush=True)
    validate_prepared(prepared, initial_summary)
    print("[add] process cycles", flush=True)
    processed, final_summary = process(prepared, initial_summary, config, channels)
    print("[add] validate processed", flush=True)
    validate_processed(processed, final_summary)
    print("[add] preserve original sensors", flush=True)
    original = prepare_original(config, prepared)
    print(f"[add] cycles={final_summary['cycle_id'].nunique()}", flush=True)
    return ExperimentBuild(
        input_dir=input_dir.resolve(),
        config=config,
        channels=channels,
        prepared=prepared,
        summary=final_summary,
        processed=processed,
        original=original,
    )
