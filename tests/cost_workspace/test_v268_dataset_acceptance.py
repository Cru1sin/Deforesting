from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

import main_cost
from cost import cost_function_v2_6_8
from dataloader import DatasetLoader


@pytest.mark.slow
def test_current_dataset_freezes_v268_review_counts_and_outcomes(tmp_path: Path) -> None:
    dataset_value = os.environ.get("DEFROST_DATASET")
    if not dataset_value:
        pytest.skip("set DEFROST_DATASET to opt in to the full V2.6.8 acceptance fit")
    dataset = Path(dataset_value)

    assert (
        main_cost.main(
            [
                "--action",
                "fit",
                "--cost",
                "v2.6.8",
                "--variant",
                "dataset_acceptance",
                "--dataset",
                str(dataset),
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )

    run = tmp_path / "cost" / "fit" / "dataset_acceptance"
    events = pd.read_csv(run / "events.csv")
    validation = pd.read_csv(run / "validation.csv")
    bootstrap = pd.read_csv(run / "bootstrap.csv")
    assert (len(events), int(events["event_valid"].sum())) == (83, 72)
    assert int((~events["event_valid"]).sum()) == 11
    assert len(validation) == 299
    assert len(bootstrap) == 69
    assert bootstrap["repeat_count"].eq(200).all()
    assert bootstrap["seed"].eq(268).all()

    loader = DatasetLoader(dataset)
    experiments = set(events.loc[events["event_valid"], "experiment_id"].astype(str))
    cohort, candidate_rows = cost_function_v2_6_8.candidate_cohort(loader, experiments)
    cohort_experiments = {
        str(loader.get_cycle_record(cycle_name)["experiment_id"]) for cycle_name in cohort
    }
    assert (len(cohort), len(cohort_experiments), candidate_rows) == (69, 15, 8461)

    artifact = json.loads((run / "params_candidate.json").read_text(encoding="utf-8"))
    curves = pd.concat(
        [
            cost_function_v2_6_8.calculate_cycle(
                loader, cycle_name, cost_function_v2_6_8.DEFAULT_RECIPE, artifact
            )
            for cycle_name in cohort
        ],
        ignore_index=True,
    )
    identified = curves.groupby("cycle_name")["diagnostic_minimum"].first().notna()
    assert int(identified.sum()) == 63
    assert not bool(identified["frost_cycle_000003"])
    assert not bool(identified["frost_cycle_000029"])

    cycle_021 = curves.loc[curves["cycle_name"].eq("frost_cycle_000021")].iloc[0]
    assert pd.Timestamp(cycle_021["diagnostic_minimum"]) == pd.Timestamp("2026-07-20 13:37:39")
    assert cycle_021["basin_1pct_width_minutes"] == 0
    assert cycle_021["basin_5pct_width_minutes"] == 2
