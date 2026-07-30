from __future__ import annotations

import pandas as pd

from frost_analysis.analysis.qualification import evaluate_task_eligibility


def test_task_eligibility_is_scoped_to_cycle_variable_and_modality() -> None:
    start = pd.Timestamp("2026-07-15")
    frame = pd.DataFrame(
        {
            "timestamp": [start, start + pd.Timedelta(seconds=30)] * 2,
            "cycle_id": ["cycle_001"] * 2 + ["cycle_002"] * 2,
            "cycle_status": "valid",
            "cycle_stage": "frost_development",
            "cycle_gap_contaminated": [True, True, False, False],
            "signal": [1.0, 3.0, 2.0, float("nan")],
            "signal__observed": [True, False, True, False],
            "signal__imputed": [False, True, False, False],
            "cop": [2.0, 2.5, 2.0, float("nan")],
            "image_front_path": ["cycle_001.jpg", None, None, None],
        }
    )
    summary = pd.DataFrame(
        {
            "cycle_id": ["cycle_001", "cycle_002"],
            "cycle_status": ["valid", "valid"],
            "rgb_quality": ["complete", "missing"],
        }
    )

    result = evaluate_task_eligibility(
        frame,
        summary,
        task="multimodal",
        required_features=["signal"],
        required_targets=["cop"],
        required_modalities={
            "sensor": {"required": True},
            "rgb": {"required": True, "required_camera_roles": ["front"]},
        },
        minimum_available_coverage=0.5,
    )

    signal_one = result.query("cycle_id == 'cycle_001' and variable == 'signal'").iloc[0]
    signal_two = result.query("cycle_id == 'cycle_002' and variable == 'signal'").iloc[0]
    rgb_one = result.query("cycle_id == 'cycle_001' and variable == 'front'").iloc[0]
    rgb_two = result.query("cycle_id == 'cycle_002' and variable == 'front'").iloc[0]

    assert bool(signal_one["qualified"])
    assert bool(signal_one["task_qualified"])
    assert bool(signal_two["qualified"])
    assert not bool(signal_two["task_qualified"])
    assert bool(rgb_one["qualified"])
    assert not bool(rgb_two["qualified"])
    assert rgb_two["reason"] == "missing_required_camera_role"
