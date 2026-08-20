#!/usr/bin/env python3
"""Build an isolated policy-conditional cost label set for the disrupted cycle11."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from analyze_raw_optimal_defrost import (
    MINIMUM_INTEGRATION_COVERAGE,
    _anchor,
    _candidate_costs,
    _raw,
    _timestamp,
)

from frost_analysis.dataset_loader import DatasetLoader
from frost_analysis.defrost_cost import optimize_renewal_cost
from frost_analysis.rgb_cost_labels import assign_image_cost_states

OOD_CYCLE = "frost_cycle_000011"
FOLLOWING_CYCLE = "frost_cycle_000012"


def analyze(dataset: Path, policy_path: Path, output: Path) -> None:
    loader = DatasetLoader(dataset)
    catalog = loader.list_cycles().set_index("cycle_name")
    record = catalog.loc[OOD_CYCLE]
    following = catalog.loc[FOLLOWING_CYCLE]
    stable = _timestamp(record.get("stable_heating_start"))
    actual = _timestamp(record.get("defrost_start"))
    next_stable = _timestamp(following.get("stable_heating_start"))
    if stable is None or actual is None or next_stable is None:
        raise ValueError("cycle11 OOD boundaries are incomplete")
    frame = _raw(loader, OOD_CYCLE)
    next_frame = _raw(loader, FOLLOWING_CYCLE)
    current_anchor = _anchor(frame, stable)
    next_anchor = _anchor(next_frame, next_stable)
    if not current_anchor["valid"] or not next_anchor["valid"]:
        raise ValueError("cycle11 OOD clean anchor is invalid")
    policy = pd.read_csv(policy_path).iloc[0]
    candidates = _candidate_costs(
        frame,
        stable_start=stable,
        candidate_end=actual,
        q_start_kw=float(current_anchor["q_clean_kw"]),
        next_stable_start=next_stable,
        q_end_kw=float(next_anchor["q_clean_kw"]),
        lambda_q=float(policy["lambda_q"]),
    )
    candidates = candidates.loc[
        candidates["integration_coverage"].ge(MINIMUM_INTEGRATION_COVERAGE)
    ].reset_index(drop=True)
    curve, optimum = optimize_renewal_cost(
        candidates,
        ticket_cost_kwh=float(policy["mean_ticket_cost_kwh_equivalent"]),
        ticket_duration_hours=float(policy["mean_ticket_duration_minutes"]) / 60,
        required_end_time=actual,
        near_optimal_fraction=0.01,
    )
    curve.insert(0, "cycle_name", OOD_CYCLE)
    curve["relative_regret"] = curve["renewal_cost_kw"] / float(
        optimum["renewal_cost_kw"]
    ) - 1
    images = loader.load_image_metadata(OOD_CYCLE)
    labels = assign_image_cost_states(
        images["image_time"], curve, regret_threshold=0.01
    )
    labels = pd.concat([images.reset_index(drop=True), labels.drop(columns="image_time")], axis=1)
    labels["experiment_id"] = str(record["experiment_id"])
    labels["split"] = "ood_only"
    output.mkdir(parents=True, exist_ok=True)
    curve.to_parquet(output / "candidate_cost_curve.parquet", index=False)
    labels.to_parquet(output / "image_cost_labels.parquet", index=False)
    pd.DataFrame(
        [
            {
                "cycle_name": OOD_CYCLE,
                "catalog_status": record["status"],
                "catalog_reason": record["status_reason"],
                "training_use": "excluded",
                "t_star": optimum["candidate_time"],
                "minimum_location": optimum["minimum_location"],
                "minutes_from_stable": (
                    pd.Timestamp(optimum["candidate_time"]) - stable
                ).total_seconds()
                / 60,
                "image_count": len(labels),
            }
        ]
    ).to_csv(output / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("report/raw_optimal_defrost/source_data/empirical_policy_summary.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("report/cycle11_ood"))
    args = parser.parse_args()
    analyze(args.dataset, args.policy, args.output)


if __name__ == "__main__":
    main()
