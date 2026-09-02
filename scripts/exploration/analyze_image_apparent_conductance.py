"""Calculate image-aligned evaporator apparent UA for a catalog cycle range."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataloader.loader import DatasetLoader
from frost_analysis.exploration.evaporator_ua import analyze_evaporator_ua

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/test/成本函数/其他/表观导热分析_cycles_020_030"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=20)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT,
    )
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start must not exceed --end")

    cycle_names = [
        f"frost_cycle_{cycle_number:06d}"
        for cycle_number in range(args.start, args.end + 1)
    ]
    output = analyze_evaporator_ua(
        DatasetLoader(ROOT / "dataset"), cycle_names, args.output
    )
    print(output)


if __name__ == "__main__":
    main()
