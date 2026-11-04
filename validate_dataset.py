"""Validate an existing processed Dataset without changing it."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dataset_tools import DatasetLoader
from dataset_tools.validate_dataset import validate_dataset
from output_layout import PROJECT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "dataset")
    parser.add_argument(
        "--check-images", action="store_true", help="额外检查原图索引；DINO 特征交付不需要"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_dataset(args.dataset, check_images=args.check_images)
    loader = DatasetLoader(args.dataset)
    print(f"dataset valid: {len(loader.list_cycles())} cycles; images checked: {args.check_images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
