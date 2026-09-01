"""Run the Prepare contract against explicitly listed experiment configs."""

from __future__ import annotations

import argparse
from pathlib import Path

from frost_analysis.dataset.channels import load_channels
from frost_analysis.dataset.config import load_config
from frost_analysis.dataset.prepare import prepare
from frost_analysis.dataset.prepared import validate_prepared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", type=Path)
    arguments = parser.parse_args()
    for config_path in arguments.config_paths:
        config = load_config(config_path)
        channels = load_channels(config.channels_path)
        prepared, summary, prepare_summary = prepare(config, channels)
        validate_prepared(prepared, summary)
        print(
            f"{config.experiment_id}: rows={prepare_summary['prepared_row_count']} "
            f"cycles={prepare_summary['cycle_count']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
