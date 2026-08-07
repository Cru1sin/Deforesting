from __future__ import annotations

from pathlib import Path

import numpy as np

from frost_analysis.evidence import build_evidence

from .conftest import frame_for, settings, write_dataset


def test_real_loader_eligibility_uses_all_statuses_and_valid_stage_is_not_a_gate(
    tmp_path: Path,
) -> None:
    loader = write_dataset(
        tmp_path / "dataset",
        [
            ("valid", "2026-07-01", "valid", frame_for()),
            ("invalid", "2026-07-01", "invalid", frame_for()),
            (
                "valid_without_stage",
                "2026-07-02",
                "valid",
                frame_for(stage=["recovery"] * 6),
            ),
        ],
    )

    bundle = build_evidence(loader, settings())

    eligibility = bundle.cycle_eligibility.set_index("cycle_name")
    assert bool(eligibility.loc["valid", "eligible"])
    assert not bool(eligibility.loc["invalid", "eligible"])
    assert eligibility.loc["invalid", "exclusion_reason"] == "cycle_status_not_valid"
    assert bool(eligibility.loc["valid_without_stage", "eligible"])
    missing_stage = bundle.feature_cycle_metrics.loc[
        bundle.feature_cycle_metrics["cycle_name"].eq("valid_without_stage")
    ]
    assert set(missing_stage["exclusion_reason"]) == {"missing_frost_stage"}
    assert "invalid" not in set(bundle.feature_cycle_metrics["cycle_name"])


def test_registry_order_direction_and_finite_quality_mask_are_applied(
    tmp_path: Path,
) -> None:
    frame = frame_for(
        elapsed=(0, 60, 120, 180, 240, 300),
        feature_a=(0, 1, 2, 3, np.nan, np.inf),
        feature_b=(6, 5, 4, 3, 2, 1),
    )
    frame.loc[4, "feature_a__imputed"] = True
    loader = write_dataset(tmp_path / "dataset", [("cycle", "2026-07-01", "valid", frame)])

    bundle = build_evidence(
        loader,
        settings(minimum_feature_points=3, minimum_feature_coverage=0.5),
    )

    metrics = bundle.feature_cycle_metrics
    assert metrics["feature"].tolist() == ["feature_a", "feature_b"]
    first = metrics.iloc[0]
    second = metrics.iloc[1]
    assert first["observed_fraction"] == 4 / 6
    assert first["metric_status"] == "available"
    assert first["spearman"] > 0
    assert second["spearman"] < 0
    assert first["signed_effect"] > 0
    assert second["signed_effect"] > 0


def test_future_effect_uses_exact_same_stage_elapsed_anchors(tmp_path: Path) -> None:
    frame = frame_for(
        elapsed=(0, 150, 300, 600, 900, 1200),
        feature_a=(1, 2, 3, 4, 5, 6),
        feature_b=(6, 5, 4, 3, 2, 1),
        heating_capacity=(0, 0, 1, 3, 6, 10),
        cop=(0, 0, -1, -3, -6, -10),
    )
    loader = write_dataset(tmp_path / "dataset", [("cycle", "2026-07-01", "valid", frame)])
    bundle = build_evidence(
        loader,
        settings(
            targets=("heating_capacity",),
            minimum_valid_pairs=2,
            minimum_pair_coverage=1.0,
        ),
    )

    future = bundle.future_association.iloc[0]
    assert future["valid_pairs"] == 4
    assert future["pair_coverage"] == 1.0
    assert future["metric_status"] == "available"
    assert future["effect"] > 0


def test_one_valid_cycle_builds_readiness_without_model_admission(tmp_path: Path) -> None:
    frame = frame_for(
        elapsed=range(0, 901, 10),
        feature_a=np.linspace(-1.0, 5.0, 91),
        feature_b=np.linspace(2.0, -3.0, 91),
        heating_capacity=np.linspace(10.0, 7.0, 91),
        cop=np.linspace(4.0, 2.5, 91),
    )
    loader = write_dataset(
        tmp_path / "dataset", [("cycle", "2026-07-01", "valid", frame)]
    )

    bundle = build_evidence(
        loader,
        settings(
            targets=("heating_capacity",),
            minimum_valid_pairs=10,
            minimum_pair_coverage=0.5,
        ),
    )

    assert len(bundle.target_audit) == 1
    five_minute = bundle.readiness_split.loc[
        bundle.readiness_split["horizon_minutes"].eq(5)
    ]
    assert len(five_minute) == 2
    assert set(five_minute["exclusion_reason"]) == {
        "no_training_cycles_after_holdout"
    }
    five_minute_summary = bundle.readiness_summary.loc[
        bundle.readiness_summary["horizon_minutes"].eq(5)
    ]
    assert set(five_minute_summary["readiness_status"]) == {
        "insufficient_validation_data"
    }
