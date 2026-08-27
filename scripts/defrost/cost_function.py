#!/usr/bin/env python3
"""Export candidate-level defrost cost functions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from analyze_raw_optimal_defrost import (
    build_cost_function_table,
    write_cost_function_csv,
)

from frost_analysis.dataset_loader import DatasetLoader

BASE_CURVES = Path(
    "output/test/成本函数/其他/经验经济窗口/源数据/candidate_cost_curves.parquet"
)
OPTIMAL_POINTS = Path(
    "output/test/成本函数/其他/经验经济窗口/源数据/cycle_optimal_points.csv"
)
TICKET_FEATURES = Path(
    "output/test/成本函数/ED模型/经验经济窗口/证据/ticket_event_features_and_predictions.csv"
)
QD_COEFFICIENTS = Path(
    "output/test/成本函数/QD模型/经验经济窗口/证据/defrost_absorbed_heat_model_coefficients.csv"
)


def _require_valid(table: pd.DataFrame, algorithm: str) -> None:
    valid_by_cycle = table["valid"].fillna(False).groupby(table["cycle_name"]).any()
    failed = valid_by_cycle.index[~valid_by_cycle].astype(str).tolist()
    if failed:
        raise RuntimeError(f"{algorithm} produced failed cycles: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", nargs="+", choices=("v1", "v2"), default=("v1", "v2")
    )
    parser.add_argument("--output", type=Path, default=Path("output/成本函数"))
    args = parser.parse_args()

    base = pd.read_parquet(BASE_CURVES)
    points = pd.read_csv(OPTIMAL_POINTS)
    tickets = pd.read_csv(TICKET_FEATURES)
    qd = pd.read_csv(QD_COEFFICIENTS)
    loader = DatasetLoader(Path("dataset"))
    for algorithm in args.algorithm:
        table = build_cost_function_table(
            base, points, tickets, qd, loader, algorithm
        )
        _require_valid(table, algorithm)
        write_cost_function_csv(table, args.output, algorithm)


if __name__ == "__main__":
    main()
