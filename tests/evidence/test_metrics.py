from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frost_analysis.evidence import build_evidence
from frost_analysis.evidence.metrics import observed_mask

from .conftest import frame_for, settings, write_dataset


def test_feature_coverage_denominator_is_all_frost_rows(tmp_path: Path) -> None:
    frame = frame_for(
        elapsed=(0, 60, 120, 180, 240, 300, 360, 420),
        stage=(
            "recovery",
            "recovery",
            "frost_development",
            "frost_development",
            "frost_development",
            "frost_development",
            "frost_development",
            "frost_development",
        ),
        feature_a=(np.nan, np.nan, 1, 2, 3, 4, 5, 6),
    )
    loader = write_dataset(tmp_path / "dataset", [("c1", "2026-07-01", "valid", frame)])

    bundle = build_evidence(
        loader,
        settings(minimum_feature_points=3, minimum_feature_coverage=0.8),
    )

    metric = bundle.feature_cycle_metrics.loc[
        bundle.feature_cycle_metrics["feature"].eq("feature_a")
    ].iloc[0]
    assert metric["observed_fraction"] == 1.0
    assert metric["metric_status"] == "available"


def test_missing_quality_column_is_local_to_feature_and_target(tmp_path: Path) -> None:
    frame = frame_for()
    frame = frame.drop(columns=["feature_a__imputed"])
    loader = write_dataset(tmp_path / "dataset", [("c1", "2026-07-01", "valid", frame)])

    bundle = build_evidence(
        loader,
        settings(targets=("heating_capacity",), horizons=(1,)),
    )

    feature = bundle.feature_cycle_metrics.loc[
        bundle.feature_cycle_metrics["feature"].eq("feature_a")
    ].iloc[0]
    future = bundle.future_association.loc[
        bundle.future_association["feature"].eq("feature_a")
    ].iloc[0]
    assert feature["exclusion_reason"] == "missing_quality_column"
    assert future["exclusion_reason"] == "missing_quality_column"

    target_frame = frame_for().drop(columns=["heating_capacity__imputed"])
    target_loader = write_dataset(
        tmp_path / "target_dataset",
        [("c1", "2026-07-01", "valid", target_frame)],
    )
    target_bundle = build_evidence(
        target_loader,
        settings(targets=("heating_capacity",), horizons=(1,)),
    )
    target_future = target_bundle.future_association.iloc[0]
    assert target_future["exclusion_reason"] == "missing_target_quality"


def test_future_anchor_requires_exact_elapsed_value_in_same_stage(tmp_path: Path) -> None:
    frame = frame_for(
        elapsed=(0, 60, 120, 180, 240, 300),
        stage=(
            "frost_development",
            "frost_development",
            "recovery",
            "frost_development",
            "frost_development",
            "frost_development",
        ),
        feature_a=(1, 2, 3, 4, 5, 6),
        heating_capacity=(0, 1, 100, 90, 95, 101),
    )
    loader = write_dataset(tmp_path / "dataset", [("c1", "2026-07-01", "valid", frame)])

    bundle = build_evidence(
        loader,
        settings(
            targets=("heating_capacity",),
            horizons=(1,),
            minimum_valid_pairs=2,
            minimum_pair_coverage=0.5,
        ),
    )

    future = bundle.future_association.loc[
        bundle.future_association["feature"].eq("feature_a")
    ].iloc[0]
    assert future["valid_pairs"] == 3
    assert future["pair_coverage"] == 1.0
    assert future["metric_status"] == "available"


def test_observed_mask_requires_finite_non_imputed_values() -> None:
    frame = pd.DataFrame(
        {
            "signal__baseline_residual": [1.0, np.nan, np.inf, -np.inf, 5.0],
            "signal__imputed": [False, False, False, False, pd.NA],
        }
    )

    assert observed_mask(frame, "signal__baseline_residual").tolist() == [
        True,
        False,
        False,
        False,
        True,
    ]


def test_future_effect_and_degradation_support_use_exact_change_direction(
    tmp_path: Path,
) -> None:
    frame = frame_for(
        elapsed=(0, 60, 120, 180, 240, 300),
        feature_a=(1, 2, 3, 4, 5, 6),
        feature_b=(6, 5, 4, 3, 2, 1),
        heating_capacity=(10, 9, 7, 4, 0, -5),
    )
    loader = write_dataset(tmp_path / "dataset", [("c1", "2026-07-01", "valid", frame)])

    bundle = build_evidence(
        loader,
        settings(
            targets=("heating_capacity",),
            horizons=(1,),
            minimum_valid_pairs=2,
            minimum_pair_coverage=1.0,
        ),
    )

    future = bundle.future_association.set_index("feature")
    assert future.loc["feature_a", "effect"] == pytest.approx(-1.0)
    assert future.loc["feature_a", "degradation_support"] == pytest.approx(1.0)
    assert future.loc["feature_b", "effect"] == pytest.approx(1.0)
    assert future.loc["feature_b", "degradation_support"] == pytest.approx(1.0)
