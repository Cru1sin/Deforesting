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
        settings(targets=("heating_capacity",)),
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
        settings(targets=("heating_capacity",)),
    )
    target_future = target_bundle.future_association.iloc[0]
    assert target_future["exclusion_reason"] == "missing_target_quality"


def test_future_anchor_requires_exact_elapsed_value_in_same_stage(tmp_path: Path) -> None:
    frame = frame_for(
        elapsed=(0, 300, 600, 900, 1200, 1500),
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


def test_metric_rows_do_not_revalidate_candidate_direction() -> None:
    from frost_analysis.evidence.metrics import feature_cycle_rows

    rows = feature_cycle_rows(
        frame_for(),
        {
            "cycle_name": "c1",
            "cycle_uid": "exp::c1",
            "experiment_id": "exp",
            "experiment_date": "2026-07-01",
            "status": "valid",
        },
        (("feature_a", "invalid"),),
        settings(),
    )

    assert rows[0]["feature"] == "feature_a"


def test_future_effect_and_degradation_support_use_exact_change_direction(
    tmp_path: Path,
) -> None:
    frame = frame_for(
        elapsed=(0, 300, 600, 900, 1200, 1500),
        feature_a=(1, 2, 3, 4, 5, 6),
        feature_b=(6, 5, 4, 3, 2, 1),
        heating_capacity=(10, 9, 7, 4, 0, -5),
    )
    loader = write_dataset(tmp_path / "dataset", [("c1", "2026-07-01", "valid", frame)])

    bundle = build_evidence(
        loader,
        settings(
            targets=("heating_capacity",),
            minimum_valid_pairs=2,
            minimum_pair_coverage=1.0,
        ),
    )

    future = bundle.future_association.loc[
        bundle.future_association["horizon_minutes"].eq(5)
    ].set_index("feature")
    assert future.loc["feature_a", "effect"] == pytest.approx(-1.0)
    assert future.loc["feature_a", "degradation_support"] == pytest.approx(1.0)
    assert future.loc["feature_b", "effect"] == pytest.approx(1.0)
    assert future.loc["feature_b", "degradation_support"] == pytest.approx(1.0)


def test_degradation_support_uses_explicit_target_direction() -> None:
    from frost_analysis.evidence.metrics import _degradation_support

    assert _degradation_support(-0.5, "increase", "decrease") == pytest.approx(0.5)
    assert _degradation_support(-0.5, "increase", "increase") == pytest.approx(-0.5)


def test_future_degradation_support_reads_target_direction_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frost_analysis.evidence.contracts import TARGET_DEGRADATION_DIRECTION

    frame = frame_for(
        elapsed=(0, 300, 600, 900, 1200, 1500),
        feature_a=(1, 2, 3, 4, 5, 6),
        heating_capacity=(10, 9, 7, 4, 0, -5),
    )
    loader = write_dataset(tmp_path / "dataset", [("c1", "2026-07-01", "valid", frame)])
    monkeypatch.setitem(TARGET_DEGRADATION_DIRECTION, "heating_capacity", "increase")

    bundle = build_evidence(
        loader,
        settings(
            targets=("heating_capacity",),
            minimum_valid_pairs=2,
            minimum_pair_coverage=1.0,
        ),
    )

    future = bundle.future_association.loc[
        bundle.future_association["horizon_minutes"].eq(5)
    ].iloc[0]
    assert future["effect"] == pytest.approx(-1.0)
    assert future["degradation_support"] == pytest.approx(-1.0)
